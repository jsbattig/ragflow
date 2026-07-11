#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Regression coverage for RedisDB.reap_consumer_pending.

Incident context (RAGFlow deployment, NAS-Docs/Emails knowledgebases,
2026-07-10/11): a Redis Streams consumer group
(rag_flow_svr_task_broker) accumulated 215 permanently-orphaned PEL
(pending-entries-list) entries across two container restarts. Each
restart gives task_executor.py workers a fresh, differently-named
consumer identity (task_executor_common_<host_id>_<idx>) - any message
XREADGROUP-delivered to a worker from a *previous* container instance,
and not yet XACK'd when that instance was replaced, is never revisited:
Redis Streams do not redeliver PEL entries on their own, and this
codebase has no XCLAIM/XAUTOCLAIM anywhere. Of the 137 entries under
dead consumer names, 28 corresponded to documents still shown RUNNING
in the application database - permanently stuck, since nothing will
ever XACK them.

task_executor.py's report_status() loop already detects dead workers
via a heartbeat timeout (WORKER_HEARTBEAT_TIMEOUT) and deregisters
them from the "TASKEXE" tracking set - but stops there, never touching
the dead worker's stream PEL entries. RedisDB.reap_consumer_pending()
is the missing half: given a queue/group/consumer_name already known
to be dead, it reclaims every PEL entry still owned by that consumer,
re-delivering each as a fresh stream message (via the existing
requeue_msg() primitive) so a live consumer can pick it up.

These tests run against a real Redis-compatible server (Messi Rule #1,
anti-mock: Redis Streams consumer-group/PEL semantics are exactly the
behavior under test and cannot be faithfully faked).

No committed conftest/CI job provisions that server: test/unit_test/ is not
wired into any CI workflow (CI only runs test/testcases/), so there is no
established precedent in this repo for provisioning Redis for a unit test.
To keep this suite self-contained and to degrade gracefully instead of
hard-failing when nothing is listening, `redis_test_port` below reuses an
externally-provisioned Redis at REDIS_CONN_TEST_PORT if one is reachable
(e.g. a future CI service container), otherwise starts and tears down a
disposable redis-server/valkey-server for the test session, and pytest.skip()s
- rather than erroring - if neither is available.
"""

import json
import os
import shutil
import socket
import subprocess
import time

import pytest
import valkey

# common.settings and rag.utils.redis_conn import each other (settings needs
# REDIS_CONN, redis_conn needs settings.decrypt_database_config). Production
# code always enters via common.settings first, which is what lets the cycle
# resolve; importing redis_conn directly here would hit a
# partially-initialized-module ImportError, so prime settings first.
import common.settings  # noqa: F401,E402 - import order matters, see comment above

from rag.utils import redis_conn


def _make_test_redis_db(port: int) -> "redis_conn.RedisDB":  # noqa: F821 - string forward ref, class name shadowed by singleton wrapper
    """Build a RedisDB instance wired to the test Redis on `port`, bypassing
    both the module's @singleton wrapper (which would return the one
    production-config-bound instance) and __init__ (which reads connection
    params from global settings, not our test port).
    """
    real_class = type(redis_conn.REDIS_CONN)
    instance = object.__new__(real_class)
    instance.config = {}
    instance.REDIS = valkey.Redis(host="localhost", port=port, db=0, decode_responses=True)
    return instance


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_reachable(port: int, timeout_s: float = 5.0) -> Exception | None:
    client = valkey.Redis(host="localhost", port=port, db=0, socket_connect_timeout=1)
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.ping()
            return None
        except Exception as e:  # noqa: BLE001 - probing, any failure means "not ready yet"
            last_error = e
            time.sleep(0.05)
    return last_error


@pytest.fixture(scope="session")
def redis_test_port():
    """Yield a port with a reachable Redis-compatible server for the whole
    test session; skip the dependent tests if none can be reached or started.

    `proc` stays None on the "reuse an externally-provisioned Redis" path so
    teardown never depends on which branch was taken.
    """
    proc: subprocess.Popen | None = None
    try:
        explicit_port = os.environ.get("REDIS_CONN_TEST_PORT")
        if explicit_port:
            port = int(explicit_port)
            error = _wait_until_reachable(port, timeout_s=1.0)
            if error is not None:
                pytest.skip(f"REDIS_CONN_TEST_PORT={port} set but not reachable: {error}")
            yield port
            return

        server_bin = shutil.which("redis-server") or shutil.which("valkey-server")
        if server_bin is None:
            pytest.skip("REDIS_CONN_TEST_PORT not set and no redis-server/valkey-server binary found on PATH")

        port = _find_free_port()
        proc = subprocess.Popen(
            [server_bin, "--port", str(port), "--save", "", "--appendonly", "no"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        error = _wait_until_reachable(port)
        if error is not None:
            pytest.skip(f"ephemeral {os.path.basename(server_bin)} on port {port} never became ready: {error}")
        yield port
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


@pytest.fixture
def redis_db(redis_test_port):
    db = _make_test_redis_db(redis_test_port)
    yield db


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{time.time_ns()}"


def _deliver(redis_db, queue: str, group: str, consumer: str, payload: dict) -> str:
    """XADD payload then XREADGROUP-deliver it to `consumer`, leaving it
    unacked (i.e. pending). Returns the stream message id.
    """
    redis_db.REDIS.xadd(queue, {"message": json.dumps(payload)})
    try:
        redis_db.REDIS.xgroup_create(queue, group, id="0", mkstream=True)
    except valkey.exceptions.ResponseError as e:
        if "busygroup" not in str(e).lower():
            raise
    messages = redis_db.REDIS.xreadgroup(group, consumer, {queue: ">"}, count=1)
    return messages[0][1][0][0]


pytestmark = pytest.mark.p1


def test_reap_consumer_pending_reclaims_dead_consumers_entries(redis_db):
    queue = _unique_name("te.0.common")
    group = "rag_flow_svr_task_broker"
    dead_consumer = "task_executor_common_deadhost_0"

    original_msg_id = _deliver(redis_db, queue, group, dead_consumer, {"id": "task-1", "doc_id": "doc-1"})

    reaped = redis_db.reap_consumer_pending(queue, group, dead_consumer)

    assert reaped == 1
    # The stale entry must be acked (gone from the dead consumer's PEL)...
    remaining_for_dead = redis_db.REDIS.xpending_range(queue, group, "-", "+", 10, consumername=dead_consumer)
    assert remaining_for_dead == []
    # ...and a fresh, deliverable copy must now exist in the stream.
    fresh = redis_db.REDIS.xrange(queue, original_msg_id, "+")
    # xrange from original_msg_id (exclusive isn't supported by plain xrange,
    # so filter out the original id itself) - anything newer proves a
    # replacement entry was appended.
    fresh_new_entries = [e for e in fresh if e[0] != original_msg_id]
    assert len(fresh_new_entries) == 1
    assert fresh_new_entries[0][1]["message"] == '{"id": "task-1", "doc_id": "doc-1"}'


def test_reap_consumer_pending_leaves_other_consumers_untouched(redis_db):
    queue = _unique_name("te.0.common")
    group = "rag_flow_svr_task_broker"
    dead_consumer = "task_executor_common_deadhost_1"
    live_consumer = "task_executor_common_livehost_0"

    _deliver(redis_db, queue, group, dead_consumer, {"id": "task-dead", "doc_id": "doc-dead"})
    live_msg_id = _deliver(redis_db, queue, group, live_consumer, {"id": "task-live", "doc_id": "doc-live"})

    reaped = redis_db.reap_consumer_pending(queue, group, dead_consumer)

    assert reaped == 1
    live_pending = redis_db.REDIS.xpending_range(queue, group, "-", "+", 10, consumername=live_consumer)
    assert len(live_pending) == 1
    assert live_pending[0]["message_id"] == live_msg_id


def test_reap_consumer_pending_returns_zero_when_consumer_has_no_pending(redis_db):
    queue = _unique_name("te.0.common")
    group = "rag_flow_svr_task_broker"
    consumer = "task_executor_common_idlehost_0"

    redis_db.REDIS.xadd(queue, {"message": "{}"})
    redis_db.REDIS.xgroup_create(queue, group, id="0", mkstream=True)

    reaped = redis_db.reap_consumer_pending(queue, group, consumer)

    assert reaped == 0


def test_reap_consumer_pending_returns_zero_for_nonexistent_queue(redis_db):
    reaped = redis_db.reap_consumer_pending(_unique_name("te.0.common.missing"), "rag_flow_svr_task_broker", "any_consumer")

    assert reaped == 0


def test_reap_consumer_pending_respects_batch_limit(redis_db):
    queue = _unique_name("te.0.common")
    group = "rag_flow_svr_task_broker"
    dead_consumer = "task_executor_common_deadhost_2"

    for i in range(3):
        _deliver(redis_db, queue, group, dead_consumer, {"id": f"task-{i}", "doc_id": f"doc-{i}"})

    reaped = redis_db.reap_consumer_pending(queue, group, dead_consumer, batch=2)

    assert reaped == 2
    still_pending = redis_db.REDIS.xpending_range(queue, group, "-", "+", 10, consumername=dead_consumer)
    assert len(still_pending) == 1


def test_requeue_msg_does_not_duplicate_the_message(redis_db):
    """requeue_msg()'s `for _ in range(3)` retry loop must stop on success.

    Without a return/break on the happy path, the loop body (xadd + xack)
    runs all 3 iterations regardless of outcome, silently re-adding the
    same message 3 times to the stream - every reap_consumer_pending()
    reclaim would then triple-process the underlying task. This was
    previously undetectable: requeue_msg() was dead code (defined, never
    called anywhere) until reap_consumer_pending() wired it up.
    """
    queue = _unique_name("te.0.common")
    group = "rag_flow_svr_task_broker"
    consumer = "task_executor_common_deadhost_3"

    msg_id = _deliver(redis_db, queue, group, consumer, {"id": "task-once", "doc_id": "doc-once"})

    redis_db.requeue_msg(queue, group, msg_id)

    all_entries = redis_db.REDIS.xrange(queue, "-", "+")
    fresh_entries = [e for e in all_entries if e[0] != msg_id]
    assert len(fresh_entries) == 1


def test_requeue_msg_acks_phantom_pel_entry_when_message_was_trimmed(redis_db):
    """A PEL entry can outlive its stream entry (e.g. trimmed via MAXLEN;
    here XDEL'd directly to simulate that without configuring MAXLEN), so
    xrange(msg_id, msg_id) then returns empty. requeue_msg() must still
    xack msg_id in that case: there is nothing left to re-xadd, but the
    phantom PEL entry must be cleared - otherwise it is never revisited
    again (the dead consumer that owned it has already been srem'd from
    "TASKEXE" tracking by the time reap_consumer_pending() runs) and stays
    permanently orphaned, defeating the entire point of the reaper for
    that entry.
    """
    queue = _unique_name("te.0.common")
    group = "rag_flow_svr_task_broker"
    consumer = "task_executor_common_deadhost_trimmed"

    msg_id = _deliver(redis_db, queue, group, consumer, {"id": "task-trimmed", "doc_id": "doc-trimmed"})
    redis_db.REDIS.xdel(queue, msg_id)

    redis_db.requeue_msg(queue, group, msg_id)

    remaining_pending = redis_db.REDIS.xpending_range(queue, group, "-", "+", 10, consumername=consumer)
    assert remaining_pending == []


def test_reap_consumer_pending_only_counts_genuine_requeue_successes(redis_db, monkeypatch):
    """reap_consumer_pending() must only increment `reaped` for entries
    whose requeue_msg() call actually succeeded.

    Previously it did `self.requeue_msg(...); reaped += 1` unconditionally,
    so a persistent Redis error on one entry (requeue_msg exhausting its
    retries and giving up, returning False) was still logged/counted as
    "reclaimed" even though nothing was actually reclaimed for it.
    """
    queue = _unique_name("te.0.common")
    group = "rag_flow_svr_task_broker"
    dead_consumer = "task_executor_common_deadhost_partial_fail"

    ok_msg_id = _deliver(redis_db, queue, group, dead_consumer, {"id": "task-ok", "doc_id": "doc-ok"})
    fail_msg_id = _deliver(redis_db, queue, group, dead_consumer, {"id": "task-fail", "doc_id": "doc-fail"})

    real_requeue_msg = redis_db.requeue_msg

    def flaky_requeue_msg(queue_arg, group_arg, msg_id_arg):
        if msg_id_arg == fail_msg_id:
            return False
        return real_requeue_msg(queue_arg, group_arg, msg_id_arg)

    monkeypatch.setattr(redis_db, "requeue_msg", flaky_requeue_msg)

    reaped = redis_db.reap_consumer_pending(queue, group, dead_consumer)

    assert reaped == 1
    still_pending = redis_db.REDIS.xpending_range(queue, group, "-", "+", 10, consumername=dead_consumer)
    still_pending_ids = {p["message_id"] for p in still_pending}
    assert still_pending_ids == {fail_msg_id}
    assert ok_msg_id not in still_pending_ids


def test_reap_consumer_pending_excludes_entries_below_min_idle_ms(redis_db):
    """The min_idle_ms gate must exclude entries that have not been pending
    long enough yet, even though the owning consumer is already known dead.

    A worker can be falsely flagged dead (GC pause, network flap, heartbeat
    coroutine briefly starved) while genuinely mid-task; if it picked up
    that task moments ago, yanking the in-flight PEL entry and
    re-delivering it risks double-processing. Combining the caller's
    external death signal with this internal idle floor is
    belt-and-suspenders against that race.
    """
    # A gate far above how long the entry has actually been pending (~0ms,
    # just delivered) - it must be excluded. A gate below the real sleep
    # below - it must become eligible once genuinely that idle.
    TOO_LARGE_MIN_IDLE_MS = 60_000
    SLEEP_BEFORE_RETRY_S = 0.2
    SMALL_MIN_IDLE_MS = 100

    queue = _unique_name("te.0.common")
    group = "rag_flow_svr_task_broker"
    dead_consumer = "task_executor_common_deadhost_idle_gate"

    msg_id = _deliver(redis_db, queue, group, dead_consumer, {"id": "task-recent", "doc_id": "doc-recent"})

    reaped_too_soon = redis_db.reap_consumer_pending(queue, group, dead_consumer, min_idle_ms=TOO_LARGE_MIN_IDLE_MS)
    assert reaped_too_soon == 0
    still_pending = redis_db.REDIS.xpending_range(queue, group, "-", "+", 10, consumername=dead_consumer)
    assert len(still_pending) == 1
    assert still_pending[0]["message_id"] == msg_id

    # After it has genuinely been idle longer than SMALL_MIN_IDLE_MS, it
    # becomes eligible - proven against real Redis idle-time reporting via
    # an actual sleep, not a mock.
    time.sleep(SLEEP_BEFORE_RETRY_S)
    reaped_after_idle = redis_db.reap_consumer_pending(queue, group, dead_consumer, min_idle_ms=SMALL_MIN_IDLE_MS)
    assert reaped_after_idle == 1
