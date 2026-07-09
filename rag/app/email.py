#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

import logging
from email import policy
from email.parser import BytesParser
from rag.app.naive import chunk as naive_chunk
from common.constants import MAXIMUM_PAGE_NUMBER
import re
from rag.nlp import rag_tokenizer, naive_merge, tokenize_chunks
from deepdoc.parser import HtmlParser, TxtParser
from timeit import default_timer as timer
import io


def chunk(
    filename,
    binary=None,
    from_page=0,
    to_page=MAXIMUM_PAGE_NUMBER,
    lang="Chinese",
    callback=None,
    **kwargs,
):
    """
    Only eml is supported
    """
    eng = lang.lower() == "english"  # is_english(cks)
    parser_config = kwargs.get(
        "parser_config",
        {"chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"},
    )
    title_text = re.sub(r"\.[a-zA-Z]+$", "", filename)
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(title_text),
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    main_res = []
    attachment_res = []

    # `filename` is the caller-supplied name (e.g. an IMAP email subject) --
    # never assume it is a real filesystem path when in-memory content was
    # provided. Bytes objects are falsy when empty (`bool(b"") is False`), so
    # `if binary:` would treat an empty-body email's `binary=b""` as "no
    # binary given" and fall through to `open(filename, "rb")`, crashing with
    # `[Errno 2] No such file or directory` on a subject that just happens to
    # look like one. `binary is not None` is the correct "was content
    # supplied" check.
    if binary is not None:
        with io.BytesIO(binary) as buffer:
            msg = BytesParser(policy=policy.default).parse(buffer)
    else:
        with open(filename, "rb") as buffer:
            msg = BytesParser(policy=policy.default).parse(buffer)

    text_txt, html_txt = [], []
    # get the email header info
    for header, value in msg.items():
        text_txt.append(f"{header}: {value}")

    #  get the email main info
    def _add_content(msg, content_type):
        def _decode_payload(payload, charset, target_list):
            try:
                target_list.append(payload.decode(charset))
            except (UnicodeDecodeError, LookupError):
                for enc in ["utf-8", "gb2312", "gbk", "gb18030", "latin1"]:
                    try:
                        target_list.append(payload.decode(enc))
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    target_list.append(payload.decode("utf-8", errors="ignore"))

        if content_type == "text/plain":
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            _decode_payload(payload, charset, text_txt)
        elif content_type == "text/html":
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            _decode_payload(payload, charset, html_txt)
        elif "multipart" in content_type:
            if msg.is_multipart():
                for part in msg.iter_parts():
                    _add_content(part, part.get_content_type())

    _add_content(msg, msg.get_content_type())

    sections = TxtParser.parser_txt("\n".join(text_txt)) + [(line, "") for line in HtmlParser.parser_txt("\n".join(html_txt), chunk_token_num=parser_config["chunk_token_num"]) if line]

    st = timer()
    chunks = naive_merge(
        sections,
        int(parser_config.get("chunk_token_num", 128)),
        parser_config.get("delimiter", "\n!?。；！？"),
    )

    main_res.extend(tokenize_chunks(chunks, doc, eng, None, language=lang))
    logging.debug("naive_merge({}): {}".format(filename, timer() - st))

    if not main_res:
        # The body was empty or entirely undecodable (e.g. an empty-body
        # email). Don't silently drop the document -- index whatever usable
        # content the subject itself carries instead. naive_merge()/
        # tokenize_chunks() already skip blank content, so an unusable
        # subject (nothing left after stripping the extension) falls straight
        # through to an empty result here too, same as a normal empty
        # document -- not a crash.
        subject_chunks = naive_merge(
            title_text,
            int(parser_config.get("chunk_token_num", 128)),
            parser_config.get("delimiter", "\n!?。；！？"),
        )
        main_res.extend(tokenize_chunks(subject_chunks, doc, eng, None, language=lang))

    # get the attachment info
    for part in msg.iter_attachments():
        content_disposition = part.get("Content-Disposition")
        if content_disposition:
            dispositions = content_disposition.strip().split(";")
            if dispositions[0].lower() == "attachment":
                filename = part.get_filename()
                payload = part.get_payload(decode=True)
                try:
                    attachment_res.extend(naive_chunk(filename, payload, callback=callback, **kwargs))
                except Exception:
                    pass

    return main_res + attachment_res


if __name__ == "__main__":
    import sys

    def dummy(prog=None, msg=""):
        pass

    chunk(sys.argv[1], callback=dummy)
