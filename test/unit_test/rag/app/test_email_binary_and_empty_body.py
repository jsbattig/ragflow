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

"""Regression tests for the email parser (``rag/app/email.py``).

Covers a production bug in the deployed IMAP email knowledge base where 136
documents failed to parse with::

    [ERROR]Internal server error while chunking: [Errno 2] No such file or
    directory: <subject>.txt

Root cause: ``chunk()`` decided whether to read the in-memory ``binary``
payload or fall back to ``open(filename, "rb")`` using ``if binary:``.
Python bytes objects are falsy when empty, so an empty-body email
(``binary=b""``, which the IMAP connector produces for messages whose
extracted text body is empty) took the ``open(filename, "rb")`` branch --
but ``filename`` is the email subject (frequently containing mojibake, e.g.
``"...m�s...�a.txt"``), never a real filesystem path, so
``open()`` always raised ``FileNotFoundError``.

Import-chain note: ``rag.app.email`` transitively imports ``rag.app.naive``
(docx/pandas/xgboost/DB-backed LLM services -- entirely unrelated to email
chunking; only reached for attachment parsing, which none of these tests
exercise) and ``deepdoc.parser`` (whose ``__init__.py`` unconditionally
imports every document parser, pulling in the same heavy stack, even though
the email chunker only uses ``HtmlParser``/``TxtParser``). This host has none
of those heavy/native dependencies installed. Rather than standing up that
infrastructure, this module installs lightweight stand-ins -- but only when
the real module genuinely cannot be imported, so a fuller environment with
the real dependencies installed is completely unaffected.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import types

import pytest


def _ensure_stub(name: str, **attrs) -> None:
    """Install a bare stand-in module at ``sys.modules[name]``, but only if
    the real module truly cannot be imported here. Heavy/irrelevant to the
    email-chunking logic under test (attachment parsing via naive_chunk uses
    them, wrapped in try/except by ``chunk()``; none of these tests touch
    attachments)."""
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


_ensure_stub("docx", Document=object)
_ensure_stub("cv2")
_ensure_stub("xgboost")
_ensure_stub("ollama")


class _StubNativeTokenizer:
    """Stand-in base class for ``infinity.rag_tokenizer.RagTokenizer`` -- a
    compiled native binding not installable on this host. It only needs to
    exist as an importable base class: the ``_stub_rag_tokenizer`` fixture
    below monkeypatches the module-level ``tokenize``/``fine_grained_tokenize``
    functions that ``chunk()`` actually calls."""

    def __init__(self, *_args, **_kwargs):
        pass

    def tokenize(self, line):
        return line

    def fine_grained_tokenize(self, tks):
        return tks

    def tag(self, *_args, **_kwargs):
        return ""

    def freq(self, *_args, **_kwargs):
        return 0

    def _tradi2simp(self, s):
        return s

    def _strQ2B(self, s):
        return s

    def set_language(self, *_args, **_kwargs):
        pass


try:
    importlib.import_module("infinity.rag_tokenizer")
except ImportError:
    _infinity_pkg = types.ModuleType("infinity")
    _infinity_rt = types.ModuleType("infinity.rag_tokenizer")
    _infinity_rt.RagTokenizer = _StubNativeTokenizer
    _infinity_rt.is_chinese = lambda _s: False
    _infinity_rt.is_number = lambda _s: False
    _infinity_rt.is_alphabet = lambda _s: False
    _infinity_rt.naive_qie = lambda t: t
    _infinity_pkg.rag_tokenizer = _infinity_rt
    sys.modules["infinity"] = _infinity_pkg
    sys.modules["infinity.rag_tokenizer"] = _infinity_rt

try:
    importlib.import_module("rag.app.naive")
except ImportError:
    # rag/app/naive.py pulls in docx submodules, api.db.services.llm_service
    # (DB-backed LLM config), and the full deepdoc parser stack. It is only
    # used by chunk() for attachment parsing (wrapped in try/except), which
    # none of these tests exercise.
    _naive_stub = types.ModuleType("rag.app.naive")
    _naive_stub.chunk = lambda *_args, **_kwargs: []
    sys.modules["rag.app.naive"] = _naive_stub

try:
    importlib.import_module("deepdoc.parser").HtmlParser  # noqa: B018 - probe for real HtmlParser
except (ImportError, AttributeError):
    # deepdoc/parser/__init__.py unconditionally imports EVERY parser
    # (docx_parser, pdf_parser, ppt_parser, ...), pulling in pandas/torch/etc,
    # even though the email chunker only uses HtmlParser/TxtParser. Register a
    # bare namespace package exposing just those two real implementations --
    # same technique test/unit_test/data_source/conftest.py uses to skip
    # common/data_source/__init__.py's heavy connector imports.
    import deepdoc  # noqa: F401 - real, lightweight (beartype_this_package() only)

    _deepdoc_parser_pkg = types.ModuleType("deepdoc.parser")
    _deepdoc_parser_pkg.__path__ = [os.path.join(os.path.dirname(deepdoc.__file__), "parser")]
    _deepdoc_parser_pkg.__package__ = "deepdoc.parser"
    sys.modules["deepdoc.parser"] = _deepdoc_parser_pkg
    _html_mod = importlib.import_module("deepdoc.parser.html_parser")
    _txt_mod = importlib.import_module("deepdoc.parser.txt_parser")
    _deepdoc_parser_pkg.HtmlParser = _html_mod.RAGFlowHtmlParser
    _deepdoc_parser_pkg.TxtParser = _txt_mod.RAGFlowTxtParser

    try:
        importlib.import_module("deepdoc.parser.pdf_parser").RAGFlowPdfParser
    except (ImportError, AttributeError):
        # rag.nlp.naive_merge() -- real production logic that chunk() calls
        # for every chunk, including the email path -- does a deferred
        # `from deepdoc.parser.pdf_parser import RAGFlowPdfParser` just to
        # call its `remove_tag` staticmethod. The real pdf_parser module also
        # unconditionally imports huggingface_hub/sklearn/deepdoc.vision
        # (OCR + layout-model machinery) at module scope, none of which
        # affect remove_tag. Reproduce remove_tag verbatim from source
        # (deepdoc/parser/pdf_parser.py:1864-1865) rather than installing an
        # OCR/ML stack to exercise one regex.
        _pdf_parser_stub = types.ModuleType("deepdoc.parser.pdf_parser")

        class _StubRAGFlowPdfParser:
            @staticmethod
            def remove_tag(txt):
                return re.sub(r"@@[\t0-9.-]+?##", "", txt)

        _pdf_parser_stub.RAGFlowPdfParser = _StubRAGFlowPdfParser
        sys.modules["deepdoc.parser.pdf_parser"] = _pdf_parser_stub
        _deepdoc_parser_pkg.PdfParser = _StubRAGFlowPdfParser

from rag.app import email as email_mod  # noqa: E402 - must follow the import-chain shims above


def _noop_callback(*_args, **_kwargs):
    pass


@pytest.fixture(autouse=True)
def _stub_rag_tokenizer(monkeypatch):
    def fake_tokenize(text):
        return str(text)

    monkeypatch.setattr("rag.nlp.rag_tokenizer.tokenize", fake_tokenize)
    monkeypatch.setattr("rag.nlp.rag_tokenizer.fine_grained_tokenize", fake_tokenize)


pytestmark = pytest.mark.p1


class TestChunkUsesBinaryNotFilesystem:
    def test_valid_binary_with_slash_and_replacement_char_name_never_opens_filesystem(self, monkeypatch):
        """(i) A subject-derived filename containing '/' and U+FFFD must never be
        dereferenced as a filesystem path when binary content is supplied -- chunking
        must come exclusively from ``binary``."""

        def _fail_open(*args, **kwargs):
            raise AssertionError("email chunker must not touch the filesystem when binary is provided")

        # Scope the patch to this module's own global namespace. Per Python's
        # LEGB name resolution, a bare `open(...)` call inside a function
        # defined in `email_mod` resolves against `email_mod`'s globals
        # BEFORE falling back to builtins -- so setting `email_mod.open`
        # intercepts every `open()` call made by that module's own code
        # without touching the process-wide builtin (verified empirically:
        # a real filesystem open() from the module raises FileNotFoundError
        # before this patch is applied, and raises this test's AssertionError
        # after -- unrelated `open()` calls made deeper in the
        # tokenizer/model-loading chain, which live in other modules, are
        # unaffected either way).
        monkeypatch.setattr(email_mod, "open", _fail_open, raising=False)

        raw_email = b"Subject: Test\r\nFrom: a@example.com\r\n\r\nHello world, this is the email body content used for chunking.\r\n"
        name = "Some/Weird Subject with � replacement chars.txt"

        chunks = email_mod.chunk(name, binary=raw_email, lang="English", callback=_noop_callback)

        assert len(chunks) >= 1
        assert any("Hello world" in c.get("content_with_weight", "") for c in chunks)


class TestEmptyBodyEmailHandledGracefully:
    def test_empty_binary_with_mojibake_subject_does_not_raise_and_indexes_subject(self):
        """(ii) Reproduces the exact production failure: binary=b"" (the empty-body
        case the IMAP connector produces) must not raise FileNotFoundError. Since a
        usable subject exists, it must be indexed as a subject-only chunk instead of
        the document being silently dropped."""
        name = "Una de las novelas de Dickens m�s desconocidas en Espa�a.txt"

        chunks = email_mod.chunk(name, binary=b"", lang="English", callback=_noop_callback)

        assert len(chunks) == 1
        assert "Dickens" in chunks[0]["content_with_weight"]

    def test_empty_binary_with_unusable_subject_skips_cleanly(self):
        """No exception and no fabricated content when there is neither a body nor a
        usable subject to fall back on (skip cleanly, not a crash)."""
        chunks = email_mod.chunk(".txt", binary=b"", lang="English", callback=_noop_callback)

        assert chunks == []
