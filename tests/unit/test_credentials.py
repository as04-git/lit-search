"""Credentials: names, the credentials file, ``config`` and ``doctor``."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lit_search import credentials
from lit_search.cli.main import main


@pytest.fixture
def cred_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "credentials.env"
    monkeypatch.setenv(credentials.ENV_FILE, str(path))
    for setting in credentials.SETTINGS:
        monkeypatch.delenv(setting.name, raising=False)
    return path


class TestParse:
    def test_plain_quoted_exported_and_shell_lines(self, tmp_path: Path) -> None:
        source = tmp_path / "keys.env"
        source.write_text(
            "# comment\n"
            "set -a\n"
            ". $HOME/other.env\n"
            "PLAIN=abc\n"
            "export EXPORTED=def\n"
            "SINGLE='g$h'\n"
            'DOUBLE="ijk"\n'
            'REFERENCE="$PLAIN"\n'
            "BARE_REF=$PLAIN\n"
            "EMPTY=\n",
            encoding="utf-8",
        )
        assert credentials.parse_env_file(source) == {
            "PLAIN": "abc",
            "EXPORTED": "def",
            "SINGLE": "g$h",
            "DOUBLE": "ijk",
            "EMPTY": "",
        }


class TestLookup:
    def test_environment_wins_then_file(
        self, cred_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        credentials.write({"OPENALEX_API_KEY": "from-file"})
        assert credentials.get("OPENALEX_API_KEY") == "from-file"
        assert credentials.origin("OPENALEX_API_KEY") == "file"
        monkeypatch.setenv("OPENALEX_API_KEY", "from-env")
        assert credentials.get("OPENALEX_API_KEY") == "from-env"
        assert credentials.origin("OPENALEX_API_KEY") == "environment"

    def test_empty_environment_value_does_not_hide_the_file(
        self, cred_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        credentials.write({"CORE_API_KEY": "real"})
        monkeypatch.setenv("CORE_API_KEY", "")
        assert credentials.get("CORE_API_KEY") == "real"

    def test_old_names_are_not_read(self, cred_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINDPAPERS_SEMANTIC_SCHOLAR_API_TOKEN", "old")
        assert credentials.get("SEMANTIC_SCHOLAR_API_KEY") is None
        assert "FINDPAPERS_SEMANTIC_SCHOLAR_API_TOKEN" in credentials.legacy_in_environment()

    def test_unknown_name_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            credentials.get("NOT_A_SETTING")


class TestWrite:
    def test_updates_keep_other_lines_and_mode_600(self, cred_file: Path) -> None:
        credentials.write({"OPENALEX_API_KEY": "a", "CORE_API_KEY": "b"})
        credentials.write({"OPENALEX_API_KEY": "c", "CORE_API_KEY": None})
        text = cred_file.read_text(encoding="utf-8")
        assert "OPENALEX_API_KEY=c" in text
        assert "CORE_API_KEY" not in text
        assert stat.S_IMODE(cred_file.stat().st_mode) == 0o600

    def test_line_breaks_are_refused(self, cred_file: Path) -> None:
        with pytest.raises(ValueError, match="line break"):
            credentials.write({"OPENALEX_API_KEY": "a\nB=c"})


class TestConfigCommand:
    def test_import_converts_old_names_and_skips_shell_references(
        self, cred_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "old.env"
        source.write_text(
            "FINDPAPERS_SEMANTIC_SCHOLAR_API_TOKEN=s2key\n"
            'FINDPAPERS_CORE_API_TOKEN="$CORE_API_KEY"\n'
            "OPENALEX_API_KEY=oakey\n"
            "ZOTERO_API_KEY=z\n",
            encoding="utf-8",
        )
        assert main(["config", "import", str(source)]) == 0
        out = capsys.readouterr().out
        assert "FINDPAPERS_SEMANTIC_SCHOLAR_API_TOKEN -> SEMANTIC_SCHOLAR_API_KEY" in out
        assert "ZOTERO_API_KEY" in out
        assert "s2key" not in out
        assert credentials.get("SEMANTIC_SCHOLAR_API_KEY") == "s2key"
        assert credentials.get("OPENALEX_API_KEY") == "oakey"
        assert credentials.get("CORE_API_KEY") is None

    def test_import_keeps_existing_values_unless_overwrite(
        self, cred_file: Path, tmp_path: Path
    ) -> None:
        credentials.write({"OPENALEX_API_KEY": "kept"})
        source = tmp_path / "new.env"
        source.write_text("OPENALEX_API_KEY=new\n", encoding="utf-8")
        main(["config", "import", str(source)])
        assert credentials.get("OPENALEX_API_KEY") == "kept"
        main(["config", "import", str(source), "--overwrite"])
        assert credentials.get("OPENALEX_API_KEY") == "new"

    def test_set_from_stdin_and_list_never_prints_values(
        self, cred_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdin", MagicMock(read=lambda: "secret-value\n"))
        assert main(["config", "set", "CORE_API_KEY", "--stdin"]) == 0
        assert credentials.get("CORE_API_KEY") == "secret-value"
        capsys.readouterr()
        assert main(["config", "list", "--json"]) == 0
        out = capsys.readouterr().out
        assert "secret-value" not in out
        rows = {r["name"]: r for r in json.loads(out)["settings"]}
        assert rows["CORE_API_KEY"]["from"] == "file"
        assert rows["IEEE_API_KEY"]["set"] is False


def _response(code: int, payload: object | None = None) -> MagicMock:
    response = MagicMock(status_code=code)
    response.json.return_value = payload or {}
    return response


class TestDoctor:
    def test_offline_reports_missing_core_keys_with_fixes(
        self, cred_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["doctor", "--offline"]) == 3
        out = capsys.readouterr().out
        assert "MISS semantic_scholar" in out
        assert "lit-search config set SEMANTIC_SCHOLAR_API_KEY --stdin" in out
        assert "not ready" in out

    def test_live_checks_pass_and_a_rejected_key_fails(
        self, cred_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        credentials.write(
            {
                "SEMANTIC_SCHOLAR_API_KEY": "s2",
                "OPENALEX_API_KEY": "oa",
                "OPENAIRE_ACCESS_TOKEN": "tok",
            }
        )
        with (
            patch("lit_search.cli.setup.requests.get", return_value=_response(200)),
            patch(
                "lit_search.connectors.openaire_auth.OpenAIRETokenManager.get_access_token",
                return_value="tok",
            ),
        ):
            assert main(["doctor"]) == 0
        assert "ready" in capsys.readouterr().out

        def by_host(url: str, **_: object) -> MagicMock:
            return _response(403 if "openalex" in url else 200)

        with (
            patch("lit_search.cli.setup.requests.get", side_effect=by_host),
            patch(
                "lit_search.connectors.openaire_auth.OpenAIRETokenManager.get_access_token",
                return_value="tok",
            ),
        ):
            assert main(["doctor"]) == 3
        out = capsys.readouterr().out
        assert "FAIL openalex" in out
        assert "credential rejected (HTTP 403)" in out

    def test_old_names_in_environment_are_flagged(
        self, cred_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINDPAPERS_OPENALEX_API_TOKEN", "x")
        main(["doctor", "--offline"])
        assert "FINDPAPERS_OPENALEX_API_TOKEN" in capsys.readouterr().out
