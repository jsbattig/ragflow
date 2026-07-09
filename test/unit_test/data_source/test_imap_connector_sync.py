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

"""Regression tests for IMAP connector sync robustness fixes.

Covers six fixes made to ``common/data_source/imap_connector.py``:

- UID search: ``_fetch_email_ids_in_mailbox`` must call
  ``mail_client.uid("SEARCH", ...)`` instead of ``mail_client.search(...)``.
  Sequence numbers returned by plain ``SEARCH``/``FETCH`` are only valid for
  the lifetime of the current ``SELECT`` and are **not** stable identifiers;
  using them elsewhere (e.g. across reconnects) can silently fetch the wrong
  message. UIDs are the stable, session-independent identifier IMAP defines
  for exactly this purpose.
- UID fetch: ``_fetch_email`` must call ``mail_client.uid("FETCH", ...)``
  instead of ``mail_client.fetch(...)``, for the same reason.
- Graceful mailbox failure: a failed ``SEARCH`` on one mailbox must not abort
  the entire sync via ``RuntimeError`` - it should log a warning and return an
  empty id list so the rest of the mailboxes still get processed.
- Optional sender: ``EmailHeaders.sender`` must accept ``None``. A message
  with no ``From:`` header (malformed/spam mail happens) must not raise a
  pydantic ``ValidationError`` and crash the run.
- Timezone-aware ``doc_updated_at``: both ``Document`` construction sites
  (the email document and its attachments) must wrap ``email_headers.date``
  with ``_as_utc`` so a timezone-naive ``Date:`` header (e.g. ``-0000``,
  which RFC 5322 defines as "no timezone information") cannot produce a
  naive ``doc_updated_at`` that later crashes a ``max(...)`` comparison
  against timezone-aware values in the sync driver's watermark logic.
- Transient-abort resilience: a large/long Gmail pull gets server-side
  killed mid-session (``imaplib.IMAP4.abort: command: UID => System Error``).
  ``ImapConnector._load_from_checkpoint`` must catch that (and the other
  transient IMAP/network exception types) around the per-message
  ``_fetch_email`` call, reconnect with a fresh ``IMAP4_SSL`` session, and
  retry the *same* UID (fetching is UID-based, so this is safe/idempotent) -
  instead of letting the abort propagate and kill the whole sync task. If
  retries are exhausted for one UID, that single message is logged + skipped
  (counted) rather than aborting the rest of the mailbox.
- Durable per-mailbox UID checkpoint: a from-beginning pull that gets killed
  at the *task* level (timeout, container restart, Gmail's daily bandwidth
  cap) must be able to resume without re-scanning/re-fetching everything.
  ``ImapConnector`` tracks a per-mailbox ``{uidvalidity, last_uid}`` high-water
  mark (``uid_checkpoints`` / ``load_uid_checkpoints``), advanced as UIDs are
  processed (success or permanent skip) and consulted on the next mailbox
  SEARCH to skip UIDs ``<= last_uid``. If UIDVALIDITY changed since the saved
  checkpoint, the mailbox restarts from UID 0 (the old UIDs are no longer
  meaningful per RFC 3501).
"""

import imaplib
import logging
from datetime import datetime, timezone
from email.message import Message
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock

import pytest

from common.data_source import imap_connector
from common.data_source.imap_connector import (
    CurrentMailbox,
    EmailHeaders,
    ImapCheckpoint,
    ImapConnector,
    _convert_email_headers_and_body_into_document,
    _fetch_email,
    _fetch_email_ids_in_mailbox,
    attachment_to_document,
    extract_attachments,
)

pytestmark = pytest.mark.p2

_START = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
_END = datetime(2026, 1, 31, tzinfo=timezone.utc).timestamp()


class TestFetchEmailIdsUsesUidSearch:
    """#1: ``_fetch_email_ids_in_mailbox`` must SEARCH by UID."""

    def test_uses_uid_search_not_plain_search(self):
        mail_client = MagicMock()
        mail_client.select.return_value = ("OK", [b"1"])
        mail_client.uid.return_value = ("OK", [b"101 102 103"])

        ids = _fetch_email_ids_in_mailbox(mail_client=mail_client, mailbox="INBOX", start=_START, end=_END)

        assert ids == ["101", "102", "103"]
        mail_client.uid.assert_called_once()
        assert mail_client.uid.call_args.args[0] == "SEARCH"
        mail_client.search.assert_not_called()

    def test_search_criteria_uses_since_before_range(self):
        mail_client = MagicMock()
        mail_client.select.return_value = ("OK", [b"1"])
        mail_client.uid.return_value = ("OK", [b""])

        _fetch_email_ids_in_mailbox(mail_client=mail_client, mailbox="INBOX", start=_START, end=_END)

        called_args = mail_client.uid.call_args.args
        assert called_args[0] == "SEARCH"
        assert called_args[1] is None
        criteria = called_args[2]
        assert "SINCE" in criteria
        assert "BEFORE" in criteria


class TestFetchEmailUsesUidFetch:
    """#2: ``_fetch_email`` must FETCH by UID."""

    def test_uses_uid_fetch_not_plain_fetch(self):
        mail_client = MagicMock()
        raw_email = b"Subject: Test\r\n\r\nBody"
        mail_client.uid.return_value = ("OK", [(b"1 (RFC822 {22}", raw_email)])

        msg = _fetch_email(mail_client=mail_client, email_id="55")

        assert isinstance(msg, Message)
        assert msg.get("Subject") == "Test"
        mail_client.uid.assert_called_once_with("FETCH", "55", "(RFC822)")
        mail_client.fetch.assert_not_called()

    def test_non_ok_status_returns_none(self):
        mail_client = MagicMock()
        mail_client.uid.return_value = ("NO", None)

        assert _fetch_email(mail_client=mail_client, email_id="55") is None


class TestFetchEmailIdsGracefulFailure:
    """#3: a bad mailbox SEARCH must warn + skip, not abort the whole sync."""

    def test_returns_empty_list_and_warns_on_non_ok_status(self, caplog):
        mail_client = MagicMock()
        mail_client.select.return_value = ("OK", [b"1"])
        mail_client.uid.return_value = ("NO", None)

        with caplog.at_level(logging.WARNING):
            ids = _fetch_email_ids_in_mailbox(mail_client=mail_client, mailbox="Archive", start=_START, end=_END)

        assert ids == []
        assert any("Archive" in rec.message for rec in caplog.records), f"expected a warning naming the failed mailbox, got: {caplog.records}"

    def test_does_not_raise_on_search_failure(self):
        # Explicit guard: before the fix this path raised RuntimeError,
        # which aborted the entire sync over a single bad mailbox.
        mail_client = MagicMock()
        mail_client.select.return_value = ("OK", [b"1"])
        mail_client.uid.return_value = ("NO", None)

        try:
            ids = _fetch_email_ids_in_mailbox(mail_client=mail_client, mailbox="Archive", start=_START, end=_END)
        except RuntimeError as e:  # pragma: no cover - guard only
            pytest.fail(f"_fetch_email_ids_in_mailbox unexpectedly raised: {e}")
        assert ids == []

    def test_empty_id_byte_array_also_returns_empty_list(self):
        mail_client = MagicMock()
        mail_client.select.return_value = ("OK", [b"1"])
        mail_client.uid.return_value = ("OK", [])

        ids = _fetch_email_ids_in_mailbox(mail_client=mail_client, mailbox="Archive", start=_START, end=_END)

        assert ids == []


class TestEmailHeadersOptionalSender:
    """#4: a message with no From: header must not crash header parsing."""

    def test_missing_from_header_yields_sender_none_without_raising(self):
        raw = "Subject: No sender here\r\nTo: someone@example.com\r\nDate: Mon, 1 Jan 2026 12:00:00 +0000\r\n\r\nBody text"
        msg = Message()
        for line in raw.split("\r\n\r\n")[0].split("\r\n"):
            name, _, value = line.partition(": ")
            msg[name] = value
        msg.set_payload(raw.split("\r\n\r\n")[1])

        assert msg.get("From") is None

        headers = EmailHeaders.from_email_msg(email_msg=msg)

        assert headers.sender is None
        assert headers.subject == "No sender here"

    def test_present_from_header_still_populates_sender(self):
        raw = "From: Alice <alice@example.com>\r\nSubject: Hi\r\nDate: Mon, 1 Jan 2026 12:00:00 +0000\r\n\r\nBody"
        msg = Message()
        for line in raw.split("\r\n\r\n")[0].split("\r\n"):
            name, _, value = line.partition(": ")
            msg[name] = value
        msg.set_payload(raw.split("\r\n\r\n")[1])

        headers = EmailHeaders.from_email_msg(email_msg=msg)

        assert headers.sender == "Alice <alice@example.com>"


def _build_naive_date_multipart_msg() -> MIMEMultipart:
    """A message whose Date: header carries no timezone info (RFC 5322 -0000).

    ``email.utils.parsedate_to_datetime`` returns a timezone-*naive*
    ``datetime`` for this format, which is the exact condition the #5 fix
    guards against.
    """
    msg = MIMEMultipart()
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Report"
    msg["Date"] = "Mon, 1 Jan 2026 12:00:00 -0000"
    msg.attach(MIMEText("Please see attached.", "plain"))

    attachment = MIMEApplication(b"file-bytes", Name="report.txt")
    attachment["Content-Disposition"] = 'attachment; filename="report.txt"'
    msg.attach(attachment)
    return msg


class TestDocUpdatedAtIsTimezoneAware:
    """#5: doc_updated_at must be tz-aware even from a tz-naive Date: header."""

    def test_convert_email_headers_produces_tz_aware_doc_updated_at(self):
        msg = _build_naive_date_multipart_msg()
        headers = EmailHeaders.from_email_msg(email_msg=msg)
        assert headers.date.tzinfo is None  # sanity: the source Date really is naive

        doc = _convert_email_headers_and_body_into_document(
            email_msg=msg,
            email_headers=headers,
            include_perm_sync=False,
        )

        assert doc.doc_updated_at.tzinfo is not None

    def test_attachment_to_document_produces_tz_aware_doc_updated_at(self):
        msg = _build_naive_date_multipart_msg()
        headers = EmailHeaders.from_email_msg(email_msg=msg)
        assert headers.date.tzinfo is None  # sanity: the source Date really is naive

        parent_doc = _convert_email_headers_and_body_into_document(
            email_msg=msg,
            email_headers=headers,
            include_perm_sync=False,
        )

        attachments = extract_attachments(msg)
        assert len(attachments) == 1

        att_doc = attachment_to_document(parent_doc, attachments[0], headers)

        assert att_doc.doc_updated_at.tzinfo is not None


def _drain(generator):
    """Exhaust a ``CheckpointOutput`` generator, returning (yielded_items, final_checkpoint).

    ``_load_from_checkpoint`` is a generator that yields ``Document``s and
    ``return``s the next ``ImapCheckpoint`` - which Python surfaces as the
    ``StopIteration.value``.
    """
    items = []
    try:
        while True:
            items.append(next(generator))
    except StopIteration as stop:
        return items, stop.value


def _raw_email_bytes(subject: str, message_id: str) -> bytes:
    return (
        f"From: Alice <alice@example.com>\r\nTo: bob@example.com\r\nSubject: {subject}\r\nDate: Mon, 15 Jan 2026 12:00:00 +0000\r\nMessage-ID: {message_id}\r\n\r\nBody text for {subject}"
    ).encode()


def _fetch_response_for_uid(uid: str) -> tuple:
    raw = _raw_email_bytes(subject=f"Msg {uid}", message_id=f"<msg-{uid}@example.com>")
    return "OK", [(f"{uid} (RFC822 {{{len(raw)}}}".encode(), raw)]


class TestLoadFromCheckpointReconnectsOnTransientAbort:
    """#6: a mid-session ``imaplib.IMAP4.abort`` during the per-message fetch
    loop must reconnect and resume the *same* UID, not kill the whole sync.

    This is the exact failure observed against live Gmail: a ~51,820-message
    ``[Gmail]/All Mail`` pull ran ~10.4 hours, ingested ~35,024 messages, then
    died with ``imaplib.IMAP4.abort: command: UID => System Error`` inside
    ``_fetch_email``'s ``mail_client.uid("FETCH", ...)`` call.
    """

    def test_reconnects_after_transient_abort_and_resumes_same_uid(self, monkeypatch):
        monkeypatch.setattr(imap_connector.time, "sleep", lambda _seconds: None)

        dying_client = MagicMock(name="dying_client")
        dying_client.select.return_value = ("OK", [b"1"])

        def dying_uid(command, uid, *_rest):
            assert command == "FETCH"
            assert uid == "10"
            raise imaplib.IMAP4.abort("command: UID => System Error")

        dying_client.uid.side_effect = dying_uid

        recovered_client = MagicMock(name="recovered_client")
        recovered_client.select.return_value = ("OK", [b"1"])
        fetched_uids: list[str] = []

        def recovered_uid(command, uid, *_rest):
            assert command == "FETCH"
            fetched_uids.append(uid)
            return _fetch_response_for_uid(uid)

        recovered_client.uid.side_effect = recovered_uid

        connector = ImapConnector(host="imap.example.com")
        clients = [dying_client, recovered_client]
        monkeypatch.setattr(connector, "_get_mail_client", lambda: clients.pop(0))

        checkpoint = ImapCheckpoint(
            has_more=True,
            todo_mailboxes=[],
            current_mailbox=CurrentMailbox(mailbox='"INBOX"', todo_email_ids=["10", "11"]),
        )

        docs, _final_checkpoint = _drain(connector._load_from_checkpoint(start=_START, end=_END, checkpoint=checkpoint, include_perm_sync=False))

        # Both messages made it through, exactly once each: no loss, no duplication.
        assert [doc.id for doc in docs] == ["<msg-10@example.com>", "<msg-11@example.com>"]
        # The reconnected client resumed on the SAME uid that aborted (10), then continued to 11.
        assert fetched_uids == ["10", "11"]
        # The dying client was only ever asked for uid 10 once - it wasn't retried on the dead client.
        assert dying_client.uid.call_count == 1
        # No permanent failures: the abort was fully recovered.
        assert connector.failed_fetch_count == 0
        # Both mailboxes/clients were exhausted (no leftover un-reconnected client).
        assert clients == []

    def test_skips_uid_after_exhausting_retries_and_continues(self, monkeypatch):
        monkeypatch.setattr(imap_connector.time, "sleep", lambda _seconds: None)

        fetch_attempts: list[str] = []

        def make_client() -> MagicMock:
            client = MagicMock(name="client")
            client.select.return_value = ("OK", [b"1"])

            def uid_side_effect(command, uid, *_rest):
                assert command == "FETCH"
                fetch_attempts.append(uid)
                if uid == "10":
                    # uid 10 always aborts, no matter how many times we reconnect.
                    raise imaplib.IMAP4.abort("command: UID => System Error")
                return _fetch_response_for_uid(uid)

            client.uid.side_effect = uid_side_effect
            return client

        connector = ImapConnector(host="imap.example.com")
        monkeypatch.setattr(connector, "_get_mail_client", make_client)

        checkpoint = ImapCheckpoint(
            has_more=True,
            todo_mailboxes=[],
            current_mailbox=CurrentMailbox(mailbox='"INBOX"', todo_email_ids=["10", "11"]),
        )

        docs, _final_checkpoint = _drain(connector._load_from_checkpoint(start=_START, end=_END, checkpoint=checkpoint, include_perm_sync=False))

        # uid 10 permanently failed and was skipped; uid 11 still made it through.
        assert [doc.id for doc in docs] == ["<msg-11@example.com>"]
        assert connector.failed_fetch_count == 1
        # uid 10 was retried up to the bound, not aborted-and-given-up after one try,
        # and not retried unboundedly either.
        assert fetch_attempts.count("10") == imap_connector._MAX_TRANSIENT_RETRIES
        assert fetch_attempts.count("11") == 1


def _make_search_client(search_uids: list[str], uidvalidity: int, fetch_log: list[str]) -> MagicMock:
    """A mock IMAP4_SSL client that answers SELECT, STATUS (UIDVALIDITY), UID SEARCH
    (returning `search_uids` verbatim - filtering is the connector's job, not the
    server's, in these tests) and UID FETCH (via `_fetch_response_for_uid`), while
    logging every fetched uid into `fetch_log` for assertions."""
    client = MagicMock()
    client.select.return_value = ("OK", [b"1"])
    client.status.return_value = ("OK", [f'"INBOX" (UIDVALIDITY {uidvalidity})'.encode()])

    def uid_dispatch(command, *args):
        if command == "SEARCH":
            return "OK", [" ".join(search_uids).encode()]
        if command == "FETCH":
            uid = args[0]
            fetch_log.append(uid)
            return _fetch_response_for_uid(uid)
        raise AssertionError(f"unexpected uid command {command!r}")

    client.uid.side_effect = uid_dispatch
    return client


class TestUidCheckpointResumability:
    """#7: a from-beginning pull killed at the *task* level (timeout, container
    restart, Gmail's daily bandwidth cap) must resume without re-scanning/
    re-fetching the whole mailbox. `ImapConnector` tracks a per-mailbox
    `{uidvalidity, last_uid}` high-water mark, restorable via
    `load_uid_checkpoints` / readable via `uid_checkpoints` so
    `rag/svr/sync_data_source.py` can persist it into the connector `config`,
    mirroring `imap_initial_sync_start`.
    """

    def test_uid_checkpoint_advances_to_max_processed_uid_after_batch(self, monkeypatch):
        fetch_log: list[str] = []
        client = _make_search_client(search_uids=["10", "11"], uidvalidity=1001, fetch_log=fetch_log)

        connector = ImapConnector(host="imap.example.com")
        monkeypatch.setattr(connector, "_get_mail_client", lambda: client)

        checkpoint = ImapCheckpoint(has_more=True, todo_mailboxes=['"INBOX"'], current_mailbox=None)

        docs, _final_checkpoint = _drain(connector._load_from_checkpoint(start=_START, end=_END, checkpoint=checkpoint, include_perm_sync=False))

        assert [doc.id for doc in docs] == ["<msg-10@example.com>", "<msg-11@example.com>"]
        assert connector.uid_checkpoints == {'"INBOX"': {"uidvalidity": 1001, "last_uid": 11}}

    def test_fresh_connector_restores_checkpoint_and_skips_already_processed_uids(self, monkeypatch):
        # First "task invocation": processes uids 10 and 11, then persists its cursor.
        first_fetch_log: list[str] = []
        first_client = _make_search_client(search_uids=["10", "11"], uidvalidity=1001, fetch_log=first_fetch_log)
        first_connector = ImapConnector(host="imap.example.com")
        monkeypatch.setattr(first_connector, "_get_mail_client", lambda: first_client)
        first_checkpoint = ImapCheckpoint(has_more=True, todo_mailboxes=['"INBOX"'], current_mailbox=None)
        _drain(first_connector._load_from_checkpoint(start=_START, end=_END, checkpoint=first_checkpoint, include_perm_sync=False))
        saved_checkpoints = first_connector.uid_checkpoints
        assert saved_checkpoints == {'"INBOX"': {"uidvalidity": 1001, "last_uid": 11}}

        # Second "task invocation": a brand-new connector instance (simulating a
        # fresh process after a task kill/container restart), restored from the
        # persisted config. The server now also has a new message, uid 12.
        second_fetch_log: list[str] = []
        second_client = _make_search_client(search_uids=["10", "11", "12"], uidvalidity=1001, fetch_log=second_fetch_log)
        second_connector = ImapConnector(host="imap.example.com")
        monkeypatch.setattr(second_connector, "_get_mail_client", lambda: second_client)
        second_connector.load_uid_checkpoints(saved_checkpoints)

        second_checkpoint = ImapCheckpoint(has_more=True, todo_mailboxes=['"INBOX"'], current_mailbox=None)
        docs, _final_checkpoint = _drain(second_connector._load_from_checkpoint(start=_START, end=_END, checkpoint=second_checkpoint, include_perm_sync=False))

        # Only the new uid was fetched - 10 and 11 were skipped, not re-fetched.
        assert second_fetch_log == ["12"]
        assert [doc.id for doc in docs] == ["<msg-12@example.com>"]
        assert second_connector.uid_checkpoints == {'"INBOX"': {"uidvalidity": 1001, "last_uid": 12}}

    def test_uidvalidity_change_restarts_mailbox_from_zero(self, monkeypatch):
        saved_checkpoints = {'"INBOX"': {"uidvalidity": 1001, "last_uid": 11}}

        fetch_log: list[str] = []
        # UIDVALIDITY changed (e.g. the mailbox was recreated) - the old uids 10/11
        # are no longer meaningful per RFC 3501, so the server handing them back
        # again must NOT be treated as "already processed".
        client = _make_search_client(search_uids=["10", "11"], uidvalidity=2002, fetch_log=fetch_log)

        connector = ImapConnector(host="imap.example.com")
        monkeypatch.setattr(connector, "_get_mail_client", lambda: client)
        connector.load_uid_checkpoints(saved_checkpoints)

        checkpoint = ImapCheckpoint(has_more=True, todo_mailboxes=['"INBOX"'], current_mailbox=None)
        docs, _final_checkpoint = _drain(connector._load_from_checkpoint(start=_START, end=_END, checkpoint=checkpoint, include_perm_sync=False))

        # Both uids were re-fetched - the mailbox restarted from uid 0, not from 11.
        assert fetch_log == ["10", "11"]
        assert [doc.id for doc in docs] == ["<msg-10@example.com>", "<msg-11@example.com>"]
        assert connector.uid_checkpoints == {'"INBOX"': {"uidvalidity": 2002, "last_uid": 11}}
