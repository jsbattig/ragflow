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
"""Regression tests for the MySQL/OceanBase connection charset default.

peewee's ``MySQLDatabase`` (and therefore our pooled/retrying subclasses in
``api.db.db_models``) silently defaults the connection charset to ``utf8`` --
MySQL's alias for the legacy 3-byte utf8mb3 -- whenever the caller does not
pass one explicitly. Every table/column in the RAGFlow schema is
``utf8mb4_0900_ai_ci``, so a utf8mb3 connection literal mixed with a utf8mb4
column trips MySQL error 1267 ("Illegal mix of collations
(utf8mb4_0900_ai_ci,IMPLICIT) and (utf8mb3_general_ci,COERCIBLE) for
operation 'concat'") on any query that CONCATs a column with a string
literal -- e.g. FileService's file-existence/dedup lookups during connector
sync. ``rag/svr/sync_data_source.py`` catches that error, logs "Skipping N
document(s) due to collation conflict", and silently drops the affected
documents.

``BaseDataBase.__init__`` must default the connection charset to
``utf8mb4`` for MySQL-family backends (mysql, oceanbase) unless the operator
already set one explicitly in ``service_conf.yaml``, and must leave postgres
untouched.
"""

import pytest

from common import settings


@pytest.mark.p0
def test_db_connection_charset_defaults_to_utf8mb4():
    """The live, singleton DB connection (as configured by service_conf.yaml
    in this environment) must use utf8mb4, not peewee's utf8mb3 default, for
    MySQL-family backends. No mocking: this imports the real, fully wired
    ``DB`` object and inspects the actual connect kwargs peewee will use.
    """
    if settings.DATABASE_TYPE.upper() not in ("MYSQL", "OCEANBASE"):
        pytest.skip(f"charset default only applies to MySQL-family backends, not {settings.DATABASE_TYPE}")

    from api.db.db_models import DB

    assert DB.connect_params.get("charset") == "utf8mb4"


@pytest.mark.p0
def test_apply_charset_default_mysql_adds_utf8mb4():
    from api.db.db_models import _apply_charset_default

    config = {"host": "mysql", "port": 3306, "user": "root", "password": "x"}

    result = _apply_charset_default(config, "mysql")

    assert result["charset"] == "utf8mb4"


@pytest.mark.p1
def test_apply_charset_default_oceanbase_adds_utf8mb4():
    from api.db.db_models import _apply_charset_default

    config = {"host": "oceanbase", "port": 2881, "user": "root", "password": "x"}

    result = _apply_charset_default(config, "oceanbase")

    assert result["charset"] == "utf8mb4"


@pytest.mark.p1
def test_apply_charset_default_preserves_explicit_charset():
    """An operator-set ``charset`` in service_conf.yaml must win over the
    utf8mb4 default -- ``setdefault`` semantics, not an unconditional set.
    """
    from api.db.db_models import _apply_charset_default

    config = {"host": "mysql", "charset": "latin1"}

    result = _apply_charset_default(config, "mysql")

    assert result["charset"] == "latin1"


@pytest.mark.p1
def test_apply_charset_default_postgres_untouched():
    """Postgres has no utf8mb3/utf8mb4 collation mismatch -- peewee's
    PostgresqlDatabase does not inject a charset default, and passing an
    unknown ``charset`` kwarg to it would break the connection. The helper
    must leave non-MySQL-family configs alone.
    """
    from api.db.db_models import _apply_charset_default

    config = {"host": "postgres", "port": 5432, "user": "root", "password": "x"}

    result = _apply_charset_default(config, "postgres")

    assert "charset" not in result
