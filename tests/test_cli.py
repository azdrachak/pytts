from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import pytts.cli as cli
from pytts.domain import ConversionResult
from pytts.errors import InputError, ModelError, SynthesisError

runner = CliRunner()


class FakePipeline:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def list_voices(self) -> tuple[str, ...]:
        return ("aidar", "xenia")

    def convert(self, request: object) -> ConversionResult:
        self.requests.append(request)
        return ConversionResult(Path("article.mp3"), "xenia", 3)


def _factory(pipeline: object):
    return lambda progress: pipeline


def test_list_voices_needs_no_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_pipeline_factory", _factory(FakePipeline()))

    result = runner.invoke(cli.app, ["--list-voices"])

    assert result.exit_code == 0
    assert result.stdout == "aidar\nxenia\n"
    assert result.stderr == ""


def test_input_and_list_voices_is_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_pipeline_factory", _factory(FakePipeline()))

    result = runner.invoke(cli.app, ["article.md", "--list-voices"])

    assert result.exit_code == 2
    assert "cannot be used together" in result.stderr
    assert result.stdout == ""


def test_input_is_required_without_list_voices() -> None:
    result = runner.invoke(cli.app, [])

    assert result.exit_code == 2
    assert "INPUT is required" in result.stderr
    assert result.stdout == ""


def test_conversion_builds_request(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = FakePipeline()
    monkeypatch.setattr(cli, "_pipeline_factory", _factory(pipeline))

    result = runner.invoke(
        cli.app,
        [
            "article.md",
            "--output",
            "narration.mp3",
            "--config",
            "custom.yaml",
            "--voice",
            "xenia",
            "--speed",
            "fast",
            "--force",
        ],
    )

    assert result.exit_code == 0
    request = pipeline.requests[0]
    assert request.input_path == Path("article.md")
    assert request.output_path == Path("narration.mp3")
    assert request.config_path == Path("custom.yaml")
    assert request.voice == "xenia"
    assert request.rate.value == "fast"
    assert request.force
    assert "article.mp3" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("error", "code"),
    [(InputError("bad input"), 3), (ModelError("bad model"), 4), (SynthesisError("bad audio"), 5)],
)
def test_maps_application_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, code: int
) -> None:
    class FailingPipeline(FakePipeline):
        def convert(self, request: object) -> ConversionResult:
            raise error

    monkeypatch.setattr(cli, "_pipeline_factory", _factory(FailingPipeline()))

    result = runner.invoke(cli.app, ["article.md"])

    assert result.exit_code == code
    assert str(error) in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_debug_prints_traceback_to_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingPipeline(FakePipeline):
        def convert(self, request: object) -> ConversionResult:
            raise ModelError("details")

    monkeypatch.setattr(cli, "_pipeline_factory", _factory(FailingPipeline()))

    result = runner.invoke(cli.app, ["article.md", "--debug"])

    assert result.exit_code == 4
    assert "Traceback" in result.stderr
    assert "ModelError" in result.stderr
    assert result.stdout == ""


def test_keyboard_interrupt_returns_130(monkeypatch: pytest.MonkeyPatch) -> None:
    class InterruptedPipeline(FakePipeline):
        def convert(self, request: object) -> ConversionResult:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_pipeline_factory", _factory(InterruptedPipeline()))

    result = runner.invoke(cli.app, ["article.md"])

    assert result.exit_code == 130
    assert "Interrupted" in result.stderr
    assert result.stdout == ""
