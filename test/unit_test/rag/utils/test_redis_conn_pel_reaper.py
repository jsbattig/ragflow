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

These tests run against a real, ephemeral redis-server (Messi Rule #1,
anti-mock: Redis Streams consumer-group/PEL semantics are exactly the
behavior under test and cannot be faithfully faked) started out-of-band
by the test runner on REDIS_CONN_TEST_PORT (defaults to 16399).
"""

import json
import os
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


def _make_test_redis_db() -> "redis_conn.RedisDB":  # noqa: F821 - string forward ref, class name shadowed by singleton wrapper
    """Build a RedisDB instance wired to the ephemeral test Redis, bypassing
    both the module's @singleton wrapper (which would return the one
    production-config-bound instance) and __init__ (which reads connection
    params from global settings, not our test port).
    """
    real_class = type(redis_conn.REDIS_CONN)
    port = int(os.environ.get("REDIS_CONN_TEST_PORT", "16399"))
    instance = object.__new__(real_class)
    instance.config = {}
    instance.REDIS = valkey.Redis(host="localhost", port=port, db=0, decode_responses=True)
    return instance


@pytest.fixture
def redis_db():
    db = _make_test_redis_db()
    db.REDIS.ping()  # fail fast with a clear error if the test Redis isn't up
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
