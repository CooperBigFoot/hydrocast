from __future__ import annotations

from launchpad.cli import app
from typer.testing import CliRunner

runner = CliRunner()


class TestCLIHelp:
    def test_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "launchpad" in result.output.lower() or "Usage" in result.output

    def test_train_help(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_evaluate_help(self) -> None:
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0

    def test_sweep_help(self) -> None:
        result = runner.invoke(app, ["sweep", "--help"])
        assert result.exit_code == 0

    def test_status_help(self) -> None:
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_logs_help(self) -> None:
        result = runner.invoke(app, ["logs", "--help"])
        assert result.exit_code == 0

    def test_pull_help(self) -> None:
        result = runner.invoke(app, ["pull", "--help"])
        assert result.exit_code == 0

    def test_push_help(self) -> None:
        result = runner.invoke(app, ["push", "--help"])
        assert result.exit_code == 0

    def test_down_help(self) -> None:
        result = runner.invoke(app, ["down", "--help"])
        assert result.exit_code == 0

    def test_volume_help(self) -> None:
        result = runner.invoke(app, ["volume", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output.lower()
        assert "ls" in result.output.lower()
        assert "upload" in result.output.lower()
        assert "delete" in result.output.lower()
        assert "setup" in result.output.lower()

    def test_volume_create_help(self) -> None:
        result = runner.invoke(app, ["volume", "create", "--help"])
        assert result.exit_code == 0
        assert "name" in result.output.lower()

    def test_volume_ls_help(self) -> None:
        result = runner.invoke(app, ["volume", "ls", "--help"])
        assert result.exit_code == 0
