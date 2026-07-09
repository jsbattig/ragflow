import copy
import email
import hashlib
from email.header import decode_header
import imaplib
import logging
import os
import re
import socket
import ssl
import time
from datetime import datetime, timedelta
from datetime import timezone
from email.message import Message
from email.utils import collapse_rfc2231_value, getaddresses
from enum import Enum
from typing import Any
from typing import Callable
from typing import TypeVar
from typing import cast

import bs4
from pydantic import BaseModel

from common.data_source.config import IMAP_CONNECTOR_SIZE_THRESHOLD, DocumentSource
from common.data_source.interfaces import (
    CheckpointOutput,
    CheckpointedConnectorWithPermSync,
    CredentialsConnector,
    CredentialsProviderInterface,
)
from common.data_source.models import (
    BasicExpertInfo,
    ConnectorCheckpoint,
    Document,
    ExternalAccess,
    GenerateSlimDocumentOutput,
    SecondsSinceUnixEpoch,
    SlimDocument,
)

_DEFAULT_IMAP_PORT_NUMBER = int(os.environ.get("IMAP_PORT", 993))
_IMAP_OKAY_STATUS = "OK"
_PAGE_SIZE = 100
_USERNAME_KEY = "imap_username"
_PASSWORD_KEY = "imap_password"

# Gmail (and other IMAP servers) will kill a long-running / high-volume IMAP
# session out from under the client mid-command - most commonly surfaced as
# `imaplib.IMAP4.abort: command: UID => System Error`. `imaplib.IMAP4_SSL`
# sessions are ephemeral (see `ImapConnector._get_mail_client`'s docstring):
# once the socket/session is dead, the only way back is a brand-new login.
# These are the exception types that indicate "the session/connection died",
# as opposed to "the request itself was rejected" - and are therefore safe
# to retry after reconnecting. `ssl.SSLError`/`socket.error` are already
# subclasses of `OSError`; they're listed explicitly for clarity.
_TRANSIENT_IMAP_ERRORS: tuple[type[BaseException], ...] = (
    imaplib.IMAP4.abort,
    imaplib.IMAP4.error,
    OSError,
    ssl.SSLError,
    socket.error,
)
_MAX_TRANSIENT_RETRIES = 5
_INITIAL_RETRY_BACKOFF_SECONDS = 1.0
_MAX_RETRY_BACKOFF_SECONDS = 30.0

_T = TypeVar("_T")


class _ImapRetryExhausted(Exception):
    """All transient-error retries for one IMAP operation were exhausted.

    Carries the last (possibly reconnected) `mail_client` so the caller can
    keep using a live connection for subsequent operations even though this
    particular operation ultimately failed.
    """

    def __init__(self, last_exc: BaseException, mail_client: imaplib.IMAP4_SSL) -> None:
        super().__init__(str(last_exc))
        self.last_exc = last_exc
        self.mail_client = mail_client


def _call_with_imap_retry(
    mail_client: imaplib.IMAP4_SSL,
    reconnect: Callable[[], imaplib.IMAP4_SSL],
    description: str,
    op: Callable[[imaplib.IMAP4_SSL], _T],
) -> tuple[_T, imaplib.IMAP4_SSL]:
    """
    Invoke `op(mail_client)`, transparently reconnecting (via `reconnect`) and
    retrying on a transient IMAP session failure - the prototypical case
    being Gmail's mid-session `imaplib.IMAP4.abort: ... System Error` on
    long-running / high-volume pulls.

    Returns `(result, mail_client)`; `mail_client` may be a *new* object if a
    reconnect happened, so callers must use the returned client for all
    subsequent calls against this mailbox.

    Raises `_ImapRetryExhausted` if all `_MAX_TRANSIENT_RETRIES` attempts
    fail - callers decide whether that means skipping one message or giving
    up on a whole mailbox, and can recover the latest live client via the
    raised exception's `.mail_client` attribute.
    """
    backoff = _INITIAL_RETRY_BACKOFF_SECONDS
    last_exc: BaseException | None = None

    for attempt in range(1, _MAX_TRANSIENT_RETRIES + 1):
        try:
            return op(mail_client), mail_client
        except _TRANSIENT_IMAP_ERRORS as exc:
            last_exc = exc
            logging.warning(f"IMAP session error while {description} (attempt {attempt}/{_MAX_TRANSIENT_RETRIES}): {exc!r}")
            if attempt == _MAX_TRANSIENT_RETRIES:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, _MAX_RETRY_BACKOFF_SECONDS)
            try:
                mail_client = reconnect()
                logging.info(f"Reconnected to IMAP server; resuming {description}")
            except _TRANSIENT_IMAP_ERRORS as reconnect_exc:
                last_exc = reconnect_exc
                logging.warning(f"Reconnect attempt {attempt} failed while {description}: {reconnect_exc!r}")

    assert last_exc is not None, "retry loop must set last_exc before exhausting attempts"
    raise _ImapRetryExhausted(last_exc=last_exc, mail_client=mail_client)


class Header(str, Enum):
    SUBJECT_HEADER = "subject"
    FROM_HEADER = "from"
    TO_HEADER = "to"
    CC_HEADER = "cc"
    DELIVERED_TO_HEADER = "Delivered-To"  # Used in mailing lists instead of the "to" header.
    DATE_HEADER = "date"
    MESSAGE_ID_HEADER = "Message-ID"


class EmailHeaders(BaseModel):
    """
    Model for email headers extracted from IMAP messages.
    """

    id: str
    subject: str
    sender: str | None = None
    recipients: str | None
    cc: str | None
    date: datetime

    @classmethod
    def from_email_msg(cls, email_msg: Message) -> "EmailHeaders":
        def _decode(header: str, default: str | None = None) -> str | None:
            value = email_msg.get(header, default)
            if not value:
                return None

            decoded_fragments = decode_header(value)
            decoded_strings: list[str] = []

            for decoded_value, encoding in decoded_fragments:
                if isinstance(decoded_value, bytes):
                    try:
                        decoded_strings.append(decoded_value.decode(encoding or "utf-8", errors="replace"))
                    except LookupError:
                        decoded_strings.append(decoded_value.decode("utf-8", errors="replace"))
                elif isinstance(decoded_value, str):
                    decoded_strings.append(decoded_value)
                else:
                    decoded_strings.append(str(decoded_value))

            return "".join(decoded_strings)

        def _parse_date(date_str: str | None) -> datetime | None:
            if not date_str:
                return None
            try:
                return email.utils.parsedate_to_datetime(date_str)
            except (TypeError, ValueError):
                return None

        # It's possible for the subject line to not exist or be an empty string.
        subject = _decode(header=Header.SUBJECT_HEADER) or "Unknown Subject"
        from_ = _decode(header=Header.FROM_HEADER)
        to = _decode(header=Header.TO_HEADER)
        if not to:
            to = _decode(header=Header.DELIVERED_TO_HEADER)
        cc = _decode(header=Header.CC_HEADER)
        date_str = _decode(header=Header.DATE_HEADER)
        parsed_date = _parse_date(date_str=date_str)
        date = parsed_date

        if not date:
            date = datetime.now(tz=timezone.utc)

        message_id = _decode(header=Header.MESSAGE_ID_HEADER)
        if not message_id:
            message_id = _build_stable_generated_message_id(
                email_msg=email_msg,
                subject=subject,
                sender=from_ or "",
                recipients=to or "",
                cc=cc or "",
                date_key=(_as_utc(parsed_date).isoformat() if parsed_date else (date_str or "")),
            )

        # If any of the above are `None`, model validation will fail.
        # Therefore, no guards (i.e.: `if <header> is None: raise RuntimeError(..)`) were written.
        return cls.model_validate(
            {
                "id": message_id,
                "subject": subject,
                "sender": from_,
                "recipients": to,
                "cc": cc,
                "date": date,
            }
        )


class CurrentMailbox(BaseModel):
    mailbox: str
    todo_email_ids: list[str]


# An email has a list of mailboxes.
# Each mailbox has a list of email-ids inside of it.
#
# Usage:
# To use this checkpointer, first fetch all the mailboxes.
# Then, pop a mailbox and fetch all of its email-ids.
# Then, pop each email-id and fetch its content (and parse it, etc..).
# When you have popped all email-ids for this mailbox, pop the next mailbox and repeat the above process until you're done.
#
# For initial checkpointing, set both fields to `None`.
class ImapCheckpoint(ConnectorCheckpoint):
    todo_mailboxes: list[str] | None = None
    current_mailbox: CurrentMailbox | None = None


class LoginState(str, Enum):
    LoggedIn = "logged_in"
    LoggedOut = "logged_out"


class ImapConnector(
    CredentialsConnector,
    CheckpointedConnectorWithPermSync,
):
    def __init__(
        self,
        host: str,
        port: int = _DEFAULT_IMAP_PORT_NUMBER,
        mailboxes: list[str] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._mailboxes = mailboxes
        self._credentials: dict[str, Any] | None = None
        # Count of messages permanently skipped after exhausting transient-error
        # retries (see `_call_with_imap_retry`). A message is only ever counted
        # here, never silently dropped without a log line.
        self._failed_fetch_count = 0
        # Per-mailbox durable resume state: `{mailbox: {"uidvalidity": int, "last_uid": int}}`.
        # Restored via `load_uid_checkpoints` and read back via `uid_checkpoints` so the
        # caller (`rag/svr/sync_data_source.py`) can persist it into the connector's
        # `config`, the same way it already persists `imap_initial_sync_start`. This lets
        # a from-beginning pull killed at the task level (timeout, container restart,
        # Gmail's daily bandwidth cap) resume without re-scanning/re-fetching everything.
        self._uid_checkpoints: dict[str, dict[str, int]] = {}

    @property
    def credentials(self) -> dict[str, Any]:
        if not self._credentials:
            raise RuntimeError("Credentials have not been initialized; call `set_credentials_provider` first")
        return self._credentials

    @property
    def failed_fetch_count(self) -> int:
        return self._failed_fetch_count

    @property
    def uid_checkpoints(self) -> dict[str, dict[str, int]]:
        """Current per-mailbox `{uidvalidity, last_uid}` high-water marks - JSON-serializable,
        suitable for persisting into the connector's `config` (see `load_uid_checkpoints`)."""
        return copy.deepcopy(self._uid_checkpoints)

    def load_uid_checkpoints(self, checkpoints: dict[str, dict[str, int]] | None) -> None:
        """Restores previously-persisted per-mailbox UID high-water marks (see `uid_checkpoints`)
        so a resumed pull can skip UIDs already processed in a prior task invocation instead of
        re-scanning the whole mailbox from UID 0."""
        restored: dict[str, dict[str, int]] = {}
        for mailbox, entry in (checkpoints or {}).items():
            try:
                restored[mailbox] = {"uidvalidity": int(entry["uidvalidity"]), "last_uid": int(entry["last_uid"])}
            except (KeyError, TypeError, ValueError):
                logging.warning(f"Ignoring malformed persisted UID checkpoint for mailbox {mailbox!r}: {entry!r}")
        self._uid_checkpoints = restored

    def _get_mail_client(self) -> imaplib.IMAP4_SSL:
        """
        Returns a new `imaplib.IMAP4_SSL` instance.

        The `imaplib.IMAP4_SSL` object is supposed to be an "ephemeral" object; it's not something that you can login,
        logout, then log back into again. I.e., the following will fail:

        ```py
        mail_client.login(..)
        mail_client.logout();
        mail_client.login(..)
        ```

        Therefore, you need a fresh, new instance in order to operate with IMAP. This function gives one to you.

        # Notes
        This function will throw an error if the credentials have not yet been set.
        """

        def get_or_raise(name: str) -> str:
            value = self.credentials.get(name)
            if not value:
                raise RuntimeError(f"Credential item {name=} was not found")
            if not isinstance(value, str):
                raise RuntimeError(f"Credential item {name=} must be of type str, instead received {type(name)=}")
            return value

        username = get_or_raise(_USERNAME_KEY)
        password = get_or_raise(_PASSWORD_KEY)

        mail_client = imaplib.IMAP4_SSL(host=self._host, port=self._port)
        status, _data = mail_client.login(user=username, password=password)

        if status != _IMAP_OKAY_STATUS:
            raise RuntimeError(f"Failed to log into imap server; {status=}")

        return mail_client

    def _make_reconnect(self, mailbox: str) -> Callable[[], imaplib.IMAP4_SSL]:
        """
        Builds a zero-arg callable that produces a brand-new, already-`mailbox`-selected
        `IMAP4_SSL` session - for use as the `reconnect` callback passed to
        `_call_with_imap_retry` when a transient IMAP session failure (e.g. a Gmail
        mid-session `abort`) needs a fresh connection to resume on.
        """

        def reconnect() -> imaplib.IMAP4_SSL:
            mail_client = self._get_mail_client()
            if not _select_mailbox(mail_client=mail_client, mailbox=mailbox):
                raise RuntimeError(f"Failed to re-select mailbox {mailbox!r} after reconnecting")
            return mail_client

        return reconnect

    def _resolve_resume_uid(self, mailbox: str, uidvalidity: int | None) -> int:
        """
        Determines the UID to resume `mailbox` from (SEARCH results with UID <= this
        value are skipped as already-processed), reconciling the freshly-fetched
        `uidvalidity` against any saved checkpoint for this mailbox:

        - No saved checkpoint: start from 0; seed a checkpoint if `uidvalidity` is known.
        - Saved checkpoint, `uidvalidity` unknown (e.g. STATUS failed): trust the saved
          checkpoint rather than lose resume progress over a transient hiccup.
        - Saved checkpoint, `uidvalidity` unchanged: resume from the saved `last_uid`.
        - Saved checkpoint, `uidvalidity` changed: the server may have reused UID
          numbers (RFC 3501) - the old high-water mark is meaningless. Restart from 0.
        """
        existing = self._uid_checkpoints.get(mailbox)

        if existing is None:
            if uidvalidity is not None:
                self._uid_checkpoints[mailbox] = {"uidvalidity": uidvalidity, "last_uid": 0}
            return 0

        if uidvalidity is None:
            return existing["last_uid"]

        if existing["uidvalidity"] != uidvalidity:
            logging.warning(f"UIDVALIDITY changed for mailbox {mailbox!r} ({existing['uidvalidity']} -> {uidvalidity}); restarting mailbox from UID 0")
            self._uid_checkpoints[mailbox] = {"uidvalidity": uidvalidity, "last_uid": 0}
            return 0

        return existing["last_uid"]

    def _advance_uid_checkpoint(self, mailbox: str, email_id: str) -> None:
        """
        Records that `email_id` has been dealt with (fetched successfully, or
        permanently given up on after exhausting retries) for `mailbox`, advancing
        the saved high-water mark if `email_id` is the largest seen so far.

        No-op if `mailbox` has no checkpoint entry yet (UIDVALIDITY was never
        successfully established for it) - there's nothing to safely anchor a
        resume cursor to in that case.
        """
        existing = self._uid_checkpoints.get(mailbox)
        if existing is None:
            return

        uid_int = _safe_int(email_id)
        if uid_int > existing["last_uid"]:
            existing["last_uid"] = uid_int

    def _load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: ImapCheckpoint,
        include_perm_sync: bool,
    ) -> CheckpointOutput[ImapCheckpoint]:
        checkpoint = cast(ImapCheckpoint, copy.deepcopy(checkpoint))
        checkpoint.has_more = True

        mail_client = self._get_mail_client()

        if checkpoint.todo_mailboxes is None:
            # This is the dummy checkpoint.
            # Fill it with mailboxes first.
            if self._mailboxes:
                checkpoint.todo_mailboxes = _sanitize_mailbox_names(self._mailboxes)
            else:
                fetched_mailboxes = _fetch_all_mailboxes_for_email_account(mail_client=mail_client)
                if not fetched_mailboxes:
                    raise RuntimeError("Failed to find any mailboxes for this email account")
                checkpoint.todo_mailboxes = _sanitize_mailbox_names(fetched_mailboxes)

            return checkpoint

        if not checkpoint.current_mailbox or not checkpoint.current_mailbox.todo_email_ids:
            if not checkpoint.todo_mailboxes:
                checkpoint.has_more = False
                return checkpoint

            mailbox = checkpoint.todo_mailboxes.pop()
            try:
                uidvalidity, mail_client = _call_with_imap_retry(
                    mail_client=mail_client,
                    reconnect=self._make_reconnect(mailbox=mailbox),
                    description=f"fetching UIDVALIDITY for mailbox {mailbox!r}",
                    op=lambda client: _fetch_uidvalidity(mail_client=client, mailbox=mailbox),
                )
            except _ImapRetryExhausted as exc:
                mail_client = exc.mail_client
                logging.warning(f"Failed to fetch UIDVALIDITY for mailbox {mailbox!r}: {exc.last_exc!r}; resume filtering may be limited to a previously-saved checkpoint")
                uidvalidity = None

            min_uid_exclusive = self._resolve_resume_uid(mailbox=mailbox, uidvalidity=uidvalidity)

            try:
                email_ids, mail_client = _call_with_imap_retry(
                    mail_client=mail_client,
                    reconnect=self._make_reconnect(mailbox=mailbox),
                    description=f"searching mailbox {mailbox!r}",
                    op=lambda client: _fetch_email_ids_in_mailbox(mail_client=client, mailbox=mailbox, start=start, end=end, min_uid_exclusive=min_uid_exclusive),
                )
            except _ImapRetryExhausted as exc:
                mail_client = exc.mail_client
                logging.warning(f"Giving up on mailbox {mailbox!r} after repeated IMAP session errors: {exc.last_exc!r}; skipping mailbox")
                email_ids = []
            checkpoint.current_mailbox = CurrentMailbox(
                mailbox=mailbox,
                todo_email_ids=email_ids,
            )

        _select_mailbox(mail_client=mail_client, mailbox=checkpoint.current_mailbox.mailbox)
        current_todos = cast(list, copy.deepcopy(checkpoint.current_mailbox.todo_email_ids[:_PAGE_SIZE]))
        checkpoint.current_mailbox.todo_email_ids = checkpoint.current_mailbox.todo_email_ids[_PAGE_SIZE:]

        for email_id in current_todos:
            try:
                email_msg, mail_client = _call_with_imap_retry(
                    mail_client=mail_client,
                    reconnect=self._make_reconnect(mailbox=checkpoint.current_mailbox.mailbox),
                    description=f"fetching message uid={email_id!r}",
                    op=lambda client, email_id=email_id: _fetch_email(mail_client=client, email_id=email_id),
                )
            except _ImapRetryExhausted as exc:
                self._failed_fetch_count += 1
                logging.error(
                    f"Permanently failed to fetch message uid={email_id!r} after {_MAX_TRANSIENT_RETRIES} attempts: "
                    f"{exc.last_exc!r}; skipping this message ({self._failed_fetch_count} total failures so far)"
                )
                mail_client = exc.mail_client
                self._advance_uid_checkpoint(mailbox=checkpoint.current_mailbox.mailbox, email_id=email_id)
                continue

            # This UID has now been definitively dealt with (fetched, whether or not
            # it turns out to be usable below) - safe to advance the resume cursor
            # past it regardless of what happens next in this iteration.
            self._advance_uid_checkpoint(mailbox=checkpoint.current_mailbox.mailbox, email_id=email_id)

            if not email_msg:
                logging.warning(f"Failed to fetch message {email_id=}; skipping")
                continue

            email_headers = EmailHeaders.from_email_msg(email_msg=email_msg)
            msg_dt = _as_utc(email_headers.date)
            start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end, tz=timezone.utc)

            if not (start_dt < msg_dt <= end_dt):
                continue

            email_doc = _convert_email_headers_and_body_into_document(
                email_msg=email_msg,
                email_headers=email_headers,
                include_perm_sync=include_perm_sync,
            )
            yield email_doc
            attachments = extract_attachments(email_msg)
            for att in attachments:
                yield attachment_to_document(email_doc, att, email_headers)

        return checkpoint

    # impls for BaseConnector

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        self._credentials = credentials
        return None

    def validate_connector_settings(self) -> None:
        self._get_mail_client()

    # impls for CredentialsConnector

    def set_credentials_provider(self, credentials_provider: CredentialsProviderInterface) -> None:
        self._credentials = credentials_provider.get_credentials()

    # impls for CheckpointedConnector

    def load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: ImapCheckpoint,
    ) -> CheckpointOutput[ImapCheckpoint]:
        return self._load_from_checkpoint(start=start, end=end, checkpoint=checkpoint, include_perm_sync=False)

    def build_dummy_checkpoint(self) -> ImapCheckpoint:
        return ImapCheckpoint(has_more=True)

    def validate_checkpoint_json(self, checkpoint_json: str) -> ImapCheckpoint:
        return ImapCheckpoint.model_validate_json(json_data=checkpoint_json)

    # impls for CheckpointedConnectorWithPermSync

    def load_from_checkpoint_with_perm_sync(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: ImapCheckpoint,
    ) -> CheckpointOutput[ImapCheckpoint]:
        return self._load_from_checkpoint(start=start, end=end, checkpoint=checkpoint, include_perm_sync=True)

    def retrieve_all_slim_docs_perm_sync(
        self,
        start: SecondsSinceUnixEpoch | None = None,
        end: SecondsSinceUnixEpoch | None = None,
        callback: Any = None,
    ) -> GenerateSlimDocumentOutput:
        del callback
        mail_client = self._get_mail_client()
        start_ts = start if start is not None else 0
        end_ts = end if end is not None else datetime.now(tz=timezone.utc).timestamp()
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

        if self._mailboxes:
            mailboxes = _sanitize_mailbox_names(self._mailboxes)
        else:
            mailboxes = _sanitize_mailbox_names(_fetch_all_mailboxes_for_email_account(mail_client=mail_client))

        slim_doc_batch: list[SlimDocument] = []
        for mailbox in mailboxes:
            email_ids = _fetch_email_ids_in_mailbox(
                mail_client=mail_client,
                mailbox=mailbox,
                start=start_ts,
                end=end_ts,
            )
            _select_mailbox(mail_client=mail_client, mailbox=mailbox)

            for email_id in email_ids:
                email_msg = _fetch_email(mail_client=mail_client, email_id=email_id)
                if not email_msg:
                    logging.warning(f"Failed to fetch message {email_id=}; skipping")
                    continue

                email_headers = EmailHeaders.from_email_msg(email_msg=email_msg)
                msg_dt = _as_utc(email_headers.date)
                if not (start_dt < msg_dt <= end_dt):
                    continue

                slim_doc_batch.append(SlimDocument(id=email_headers.id))
                for att in extract_attachments(email_msg):
                    slim_doc_batch.append(SlimDocument(id=_attachment_document_id(email_headers.id, att)))

                if len(slim_doc_batch) >= _PAGE_SIZE:
                    yield slim_doc_batch
                    slim_doc_batch = []

        if slim_doc_batch:
            yield slim_doc_batch


def _fetch_all_mailboxes_for_email_account(mail_client: imaplib.IMAP4_SSL) -> list[str]:
    status, mailboxes_data = mail_client.list('""', "*")
    if status != _IMAP_OKAY_STATUS:
        raise RuntimeError(f"Failed to fetch mailboxes; {status=}")

    mailboxes = []

    for mailboxes_raw in mailboxes_data:
        if isinstance(mailboxes_raw, bytes):
            mailboxes_str = mailboxes_raw.decode()
        elif isinstance(mailboxes_raw, str):
            mailboxes_str = mailboxes_raw
        else:
            logging.warning(f"Expected the mailbox data to be of type str, instead got {type(mailboxes_raw)=} {mailboxes_raw}; skipping")
            continue

        # The mailbox LIST response output can be found here:
        # https://www.rfc-editor.org/rfc/rfc3501.html#section-7.2.2
        #
        # The general format is:
        # `(<name-attributes>) <hierarchy-delimiter> <mailbox-name>`
        #
        # The below regex matches on that pattern; from there, we select the 3rd match (index 2), which is the mailbox-name.
        match = re.match(r'\([^)]*\)\s+"([^"]+)"\s+"?(.+?)"?$', mailboxes_str)
        if not match:
            logging.warning(f"Invalid mailbox-data formatting structure: {mailboxes_str=}; skipping")
            continue

        mailbox = match.group(2)
        mailboxes.append(mailbox)
    if not mailboxes:
        logging.warning("No mailboxes parsed from LIST response; falling back to INBOX")
        return ["INBOX"]

    return mailboxes


def _select_mailbox(mail_client: imaplib.IMAP4_SSL, mailbox: str) -> bool:
    try:
        status, _ = mail_client.select(mailbox=mailbox, readonly=True)
        if status != _IMAP_OKAY_STATUS:
            return False
        return True
    except _TRANSIENT_IMAP_ERRORS:
        # Session/connection died mid-SELECT (e.g. a Gmail abort) - let this propagate so
        # `_call_with_imap_retry` (wrapping `_fetch_email_ids_in_mailbox`) can reconnect and
        # retry, instead of it being silently swallowed into an "unselectable mailbox" False.
        raise
    except Exception:
        return False


def _fetch_uidvalidity(mail_client: imaplib.IMAP4_SSL, mailbox: str) -> int | None:
    """
    Fetches the mailbox's current UIDVALIDITY via `STATUS ... (UIDVALIDITY)`
    (RFC 3501 §6.3.10) - the value a persisted per-mailbox UID checkpoint must
    be anchored to. If UIDVALIDITY differs from what was last observed for a
    mailbox, every previously-remembered UID is meaningless (the server is
    free to reuse UID numbers after a UIDVALIDITY change) and any saved
    high-water mark for that mailbox must be discarded.

    Returns `None` if the value could not be determined (caller should treat
    this as "unknown" rather than assume anything about resumability).
    """
    status, data = mail_client.status(mailbox, "(UIDVALIDITY)")
    if status != _IMAP_OKAY_STATUS or not data or not data[0]:
        return None

    raw = data[0]
    text = raw.decode() if isinstance(raw, bytes) else raw
    match = re.search(r"UIDVALIDITY\s+(\d+)", text)
    if not match:
        return None

    return int(match.group(1))


def _safe_int(value: str) -> int:
    """`int(value)`, or -1 if `value` isn't a valid integer - so a malformed
    UID string can never crash a `> min_uid_exclusive` comparison; -1 simply
    sorts below any real UID and gets filtered out like any other."""
    try:
        return int(value)
    except ValueError:
        return -1


def _fetch_email_ids_in_mailbox(
    mail_client: imaplib.IMAP4_SSL,
    mailbox: str,
    start: SecondsSinceUnixEpoch,
    end: SecondsSinceUnixEpoch,
    min_uid_exclusive: int = 0,
) -> list[str]:
    if not _select_mailbox(mail_client, mailbox):
        logging.warning(f"Skip mailbox: {mailbox}")
        return []

    start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end, tz=timezone.utc) + timedelta(days=1)

    start_str = start_dt.strftime("%d-%b-%Y")
    end_str = end_dt.strftime("%d-%b-%Y")
    search_criteria = f'(SINCE "{start_str}" BEFORE "{end_str}")'

    status, email_ids_byte_array = mail_client.uid("SEARCH", None, search_criteria)

    if status != _IMAP_OKAY_STATUS or not email_ids_byte_array:
        logging.warning(f"Failed to fetch email ids for mailbox {mailbox}; {status=}; skipping")
        return []

    email_ids: bytes = email_ids_byte_array[0]
    all_ids = [email_id.decode() for email_id in email_ids.split()]

    if min_uid_exclusive <= 0:
        return all_ids

    # Durable-resume filtering: skip UIDs already processed in a prior task
    # invocation (per the persisted `{uidvalidity, last_uid}` checkpoint),
    # instead of re-fetching a from-beginning pull's entire history on every
    # resume. Client-side filtering (rather than a server-side UID range in
    # the SEARCH itself) keeps this simple and robust across IMAP server
    # implementations; the SEARCH response is cheap (just UID numbers), so
    # the extra bandwidth here is negligible compared to re-FETCHing bodies.
    resumed_ids = [email_id for email_id in all_ids if _safe_int(email_id) > min_uid_exclusive]
    skipped = len(all_ids) - len(resumed_ids)
    if skipped:
        logging.info(f"Resuming mailbox {mailbox!r} from UID > {min_uid_exclusive}: skipping {skipped} already-processed UID(s)")
    return resumed_ids


def _fetch_email(mail_client: imaplib.IMAP4_SSL, email_id: str) -> Message | None:
    status, msg_data = mail_client.uid("FETCH", email_id, "(RFC822)")
    if status != _IMAP_OKAY_STATUS or not msg_data:
        return None

    data = msg_data[0]
    if not isinstance(data, tuple):
        raise RuntimeError(f"Message data should be a tuple; instead got a {type(data)=} {data=}")

    _, raw_email = data
    return email.message_from_bytes(raw_email)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_stable_generated_message_id(
    email_msg: Message,
    subject: str,
    sender: str,
    recipients: str,
    cc: str,
    date_key: str,
) -> str:
    body = _extract_email_body_text(email_msg)
    raw_digest = hashlib.sha256(email_msg.as_bytes()).hexdigest()
    body_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    digest = hashlib.sha256(
        "\n".join(
            [
                subject,
                date_key,
                sender,
                recipients,
                cc,
                body_digest,
                raw_digest,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"generated:{digest}"


def _convert_email_headers_and_body_into_document(
    email_msg: Message,
    email_headers: EmailHeaders,
    include_perm_sync: bool,
) -> Document:
    sender_name, sender_addr = _parse_singular_addr(raw_header=email_headers.sender)
    to_addrs = _parse_addrs(email_headers.recipients) if email_headers.recipients else []
    cc_addrs = _parse_addrs(email_headers.cc) if email_headers.cc else []
    all_participants = to_addrs + cc_addrs

    expert_info_map = {recipient_addr: BasicExpertInfo(display_name=recipient_name, email=recipient_addr) for recipient_name, recipient_addr in all_participants}
    if sender_addr not in expert_info_map:
        expert_info_map[sender_addr] = BasicExpertInfo(display_name=sender_name, email=sender_addr)

    email_body = _parse_email_body(email_msg=email_msg, email_headers=email_headers)
    primary_owners = list(expert_info_map.values())
    external_access = (
        ExternalAccess(
            external_user_emails=set(expert_info_map.keys()),
            external_user_group_ids=set(),
            is_public=False,
        )
        if include_perm_sync
        else None
    )
    return Document(
        id=email_headers.id,
        title=email_headers.subject,
        blob=email_body,
        size_bytes=len(email_body),
        semantic_identifier=email_headers.subject,
        metadata={},
        extension=".txt",
        doc_updated_at=_as_utc(email_headers.date),
        source=DocumentSource.IMAP,
        primary_owners=primary_owners,
        external_access=external_access,
    )


def extract_attachments(email_msg: Message, max_bytes: int = IMAP_CONNECTOR_SIZE_THRESHOLD):
    attachments = []

    if not email_msg.is_multipart():
        return attachments

    for part in email_msg.walk():
        if part.get_content_maintype() == "multipart":
            continue

        disposition = (part.get("Content-Disposition") or "").lower()
        filename = part.get_filename()

        if not (disposition.startswith("attachment") or (disposition.startswith("inline") and filename)):
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        if len(payload) > max_bytes:
            continue

        attachments.append(
            {
                "filename": filename or "attachment.bin",
                "content_type": part.get_content_type(),
                "content_bytes": payload,
                "size_bytes": len(payload),
            }
        )

    return attachments


def decode_mime_filename(raw: str | None) -> str | None:
    if not raw:
        return None

    try:
        raw = collapse_rfc2231_value(raw)
    except Exception:
        pass

    parts = decode_header(raw)
    decoded = []

    for value, encoding in parts:
        if isinstance(value, bytes):
            decoded.append(value.decode(encoding or "utf-8", errors="replace"))
        else:
            decoded.append(value)

    return "".join(decoded)


def _attachment_document_id(parent_doc_id: str, att: dict) -> str:
    raw_filename = att["filename"]
    filename = decode_mime_filename(raw_filename) or "attachment.bin"
    return f"{parent_doc_id}#att:{filename}"


def attachment_to_document(
    parent_doc: Document,
    att: dict,
    email_headers: EmailHeaders,
):
    raw_filename = att["filename"]
    filename = decode_mime_filename(raw_filename) or "attachment.bin"
    ext = "." + filename.split(".")[-1] if "." in filename else ""

    return Document(
        id=_attachment_document_id(parent_doc.id, att),
        source=DocumentSource.IMAP,
        semantic_identifier=filename,
        extension=ext,
        blob=att["content_bytes"],
        size_bytes=att["size_bytes"],
        doc_updated_at=_as_utc(email_headers.date),
        primary_owners=parent_doc.primary_owners,
        metadata={
            "parent_email_id": parent_doc.id,
            "parent_subject": email_headers.subject,
            "attachment_filename": filename,
            "attachment_content_type": att["content_type"],
        },
    )


def _parse_email_body(
    email_msg: Message,
    email_headers: EmailHeaders,
) -> str:
    body = _extract_email_body_text(email_msg)
    if not body:
        logging.warning(f"Email with {email_headers.id=} has an empty body; returning an empty string")
    return body


def _extract_email_body_text(email_msg: Message) -> str:
    body = None
    for part in email_msg.walk():
        if part.is_multipart():
            # Multipart parts are *containers* for other parts, not the actual content itself.
            # Therefore, we skip until we find the individual parts instead.
            continue

        charset = part.get_content_charset() or "utf-8"

        try:
            raw_payload = part.get_payload(decode=True)
            if not isinstance(raw_payload, bytes):
                logging.warning(f"Payload section from email was expected to be an array of bytes, instead got {type(raw_payload)=}, {raw_payload=}")
                continue
            body = raw_payload.decode(charset)
            break
        except (UnicodeDecodeError, LookupError) as e:
            logging.warning(f"Could not decode part with charset {charset}. Error: {e}")
            continue

    if not body:
        return ""

    soup = bs4.BeautifulSoup(markup=body, features="html.parser")

    return " ".join(str_section for str_section in soup.stripped_strings)


def _sanitize_mailbox_names(mailboxes: list[str]) -> list[str]:
    """
    Mailboxes with special characters in them must be enclosed by double-quotes, as per the IMAP protocol.
    Just to be safe, we wrap *all* mailboxes with double-quotes.
    """
    return [f'"{mailbox}"' for mailbox in mailboxes if mailbox]


def _parse_addrs(raw_header: str) -> list[tuple[str, str]]:
    if not raw_header:
        return []
    return getaddresses([raw_header])


def _parse_singular_addr(raw_header: str) -> tuple[str, str]:
    addrs = _parse_addrs(raw_header=raw_header)
    if not addrs:
        return ("Unknown", "unknown@example.com")
    if len(addrs) >= 2:
        logging.warning(
            "Multiple addresses in header expected to be singular; using first. parsed_count=%d",
            len(addrs),
        )
    return addrs[0]


if __name__ == "__main__":
    import time
    import uuid
    from types import TracebackType
    from common.data_source.utils import load_all_docs_from_checkpoint_connector

    class OnyxStaticCredentialsProvider(CredentialsProviderInterface["OnyxStaticCredentialsProvider"]):
        """Implementation (a very simple one!) to handle static credentials."""

        def __init__(
            self,
            tenant_id: str | None,
            connector_name: str,
            credential_json: dict[str, Any],
        ):
            self._tenant_id = tenant_id
            self._connector_name = connector_name
            self._credential_json = credential_json

            self._provider_key = str(uuid.uuid4())

        def __enter__(self) -> "OnyxStaticCredentialsProvider":
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            pass

        def get_tenant_id(self) -> str | None:
            return self._tenant_id

        def get_provider_key(self) -> str:
            return self._provider_key

        def get_credentials(self) -> dict[str, Any]:
            return self._credential_json

        def set_credentials(self, credential_json: dict[str, Any]) -> None:
            self._credential_json = credential_json

        def is_dynamic(self) -> bool:
            return False

    # from tests.daily.connectors.utils import load_all_docs_from_checkpoint_connector
    # from onyx.connectors.credentials_provider import OnyxStaticCredentialsProvider

    host = os.environ.get("IMAP_HOST")
    mailboxes_str = os.environ.get("IMAP_MAILBOXES", "INBOX")
    username = os.environ.get("IMAP_USERNAME")
    password = os.environ.get("IMAP_PASSWORD")

    mailboxes = [mailbox.strip() for mailbox in mailboxes_str.split(",")] if mailboxes_str else []

    if not host:
        raise RuntimeError("`IMAP_HOST` must be set")

    imap_connector = ImapConnector(
        host=host,
        mailboxes=mailboxes,
    )

    imap_connector.set_credentials_provider(
        OnyxStaticCredentialsProvider(
            tenant_id=None,
            connector_name=DocumentSource.IMAP,
            credential_json={
                _USERNAME_KEY: username,
                _PASSWORD_KEY: password,
            },
        )
    )
    END = time.time()
    START = END - 1 * 24 * 60 * 60
    for doc in load_all_docs_from_checkpoint_connector(
        connector=imap_connector,
        start=START,
        end=END,
    ):
        print(doc.id, doc.extension)
