from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Protocol

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from pytts.domain import ConversionRequest, ConversionResult, SpeechRate
from pytts.errors import PyTTSError

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console(stderr=True)


class ProgressEvent(Protocol):
    stage: str
    completed: int | None
    total: int | None
    message: str | None
    warning: bool


ProgressSink = Callable[[ProgressEvent], None]


class Pipeline(Protocol):
    def list_voices(self) -> tuple[str, ...]: ...

    def convert(self, request: ConversionRequest) -> ConversionResult: ...


class RichProgressReporter:
    def __init__(self, target: Console) -> None:
        self._console = target
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=target,
            transient=True,
        )
        self._tasks: dict[str, int] = {}
        self._started = False

    def __call__(self, event: ProgressEvent) -> None:
        if event.warning and event.message:
            self._console.print(f"[yellow]Warning:[/] {event.message}")
        if event.completed is not None:
            if not self._started:
                self._progress.start()
                self._started = True
            stage = str(event.stage)
            task_id = self._tasks.get(stage)
            if task_id is None:
                task_id = self._progress.add_task(stage, total=event.total)
                self._tasks[stage] = task_id
            self._progress.update(task_id, completed=event.completed, total=event.total)
        elif event.message and not event.warning:
            self._console.print(event.message)

    def stop(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False


def build_pipeline(progress: ProgressSink) -> Pipeline:
    from pytts.audio import AtomicMp3Writer
    from pytts.config import find_project_root
    from pytts.model_store import ModelStore, load_model_spec
    from pytts.pipeline import ConversionPipeline
    from pytts.readers.input import InputReader
    from pytts.tts import SileroRuntime

    spec = load_model_spec()
    store = ModelStore()
    return ConversionPipeline(
        input_reader=InputReader(),
        model_spec=spec,
        runtime_factory=lambda download: SileroRuntime.load(store, spec, download),
        writer_factory=lambda output, force: AtomicMp3Writer(
            output, force, sample_rate=spec.sample_rate
        ),
        project_root=find_project_root(Path(__file__)),
        progress=progress,
    )


PipelineFactory = Callable[[ProgressSink], Pipeline]
_pipeline_factory: PipelineFactory = build_pipeline


def _usage(message: str) -> None:
    console.print(f"[red]Error:[/] {message}")
    raise typer.Exit(2)


@app.command()
def main(
    input_path: Annotated[Path | None, typer.Argument(metavar="[INPUT]")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    voice: Annotated[str | None, typer.Option("--voice")] = None,
    speed: Annotated[SpeechRate, typer.Option("--speed")] = SpeechRate.NORMAL,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    list_voices: Annotated[bool, typer.Option("--list-voices")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    if list_voices and input_path is not None:
        _usage("INPUT and --list-voices cannot be used together")
    if not list_voices and input_path is None:
        _usage("INPUT is required unless --list-voices is used")

    reporter = RichProgressReporter(console)
    try:
        pipeline = _pipeline_factory(reporter)
        if list_voices:
            for name in pipeline.list_voices():
                typer.echo(name)
            return

        source = input_path
        assert source is not None
        result = pipeline.convert(
            ConversionRequest(
                input_path=source,
                output_path=output,
                config_path=config,
                voice=voice,
                rate=speed,
                force=force,
            )
        )
        typer.echo(f"Wrote {result.output_path} with voice {result.voice}")
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/]")
        raise typer.Exit(130) from None
    except PyTTSError as error:
        console.print(f"[red]Error:[/] {error}")
        if debug:
            console.print_exception(show_locals=False)
        raise typer.Exit(error.exit_code) from None
    finally:
        reporter.stop()
