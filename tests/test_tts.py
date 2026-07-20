from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pytts.domain import SpeechChunk
from pytts.errors import ModelError, SynthesisError
from pytts.model_store import ModelSpec
from pytts.tts import SileroRuntime


_DEFAULT_AUDIO = object()


def _spec(preferred: str = "xenia") -> ModelSpec:
    return ModelSpec(
        "v5_5_ru",
        "https://example.test/model.pt",
        "a" * 64,
        preferred,
        48000,
        800,
    )


class FakeModel:
    def __init__(self, speakers: list[str], audio: object = _DEFAULT_AUDIO) -> None:
        self.speakers = speakers
        self.device: torch.device | None = None
        self.evaluated = False
        self.calls: list[dict[str, object]] = []
        self.audio = (
            torch.tensor([0.0, 0.25, -0.25]) if audio is _DEFAULT_AUDIO else audio
        )

    def to(self, device: torch.device) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> FakeModel:
        self.evaluated = True
        return self

    def apply_tts(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.audio


class FailingModel(FakeModel):
    def apply_tts(self, **kwargs: object) -> object:
        raise RuntimeError("boom")


class InferenceOnlyModel:
    def __init__(self) -> None:
        self.speakers = ["xenia"]
        self.device: torch.device | None = None
        self.calls: list[dict[str, object]] = []

    def to(self, device: torch.device) -> InferenceOnlyModel:
        self.device = device
        return self

    def apply_tts(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.tensor([0.0, 0.25, -0.25])


def _chunk() -> SpeechChunk:
    return SpeechChunk('<speak><prosody rate="fast">Текст &amp; ещё</prosody></speak>', 350)


def test_loads_and_synthesizes_inference_wrapper_without_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = InferenceOnlyModel()
    path = tmp_path / "v5_5_ru.pt"

    class FakeImporter:
        def __init__(self, source: str) -> None:
            assert source == str(path)

        def load_pickle(self, package: str, resource: str) -> InferenceOnlyModel:
            assert (package, resource) == ("tts_models", "model")
            return model

    class FakeStore:
        def ensure(self, *_: object) -> Path:
            return path

    monkeypatch.setattr(torch.package, "PackageImporter", FakeImporter)

    runtime = SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]
    audio = runtime.synthesize(_chunk(), "xenia")

    assert model.device == torch.device("cpu")
    assert model.calls
    assert audio.shape == (3,)


def test_ignores_non_callable_eval_metadata() -> None:
    model = InferenceOnlyModel()
    model.eval = "inference wrapper metadata"  # type: ignore[attr-defined]

    runtime = SileroRuntime(model, _spec())

    assert runtime.speakers == ("xenia",)


def test_prefers_manifest_voice_when_runtime_has_it() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "xenia"]), _spec())

    assert runtime.resolve_voice(None).name == "xenia"
    assert runtime.resolve_voice(None).warning is None


def test_falls_back_to_first_runtime_voice_with_warning() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "baya"]), _spec())

    selection = runtime.resolve_voice(None)

    assert selection.name == "aidar"
    assert selection.warning is not None
    assert "xenia" in selection.warning


def test_resolves_requested_voice_only_against_runtime_speakers() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "baya"]), _spec())

    assert runtime.resolve_voice("baya").name == "baya"
    with pytest.raises(ModelError, match="aidar, baya"):
        runtime.resolve_voice("xenia")


def test_synthesizes_chunk_ssml_on_cpu_at_manifest_sample_rate() -> None:
    model = FakeModel(["xenia"])
    runtime = SileroRuntime(model, _spec())
    chunk = _chunk()

    audio = runtime.synthesize(chunk, "xenia")

    assert model.device == torch.device("cpu")
    assert model.evaluated
    assert model.calls == [
        {
            "ssml_text": chunk.ssml_text,
            "speaker": "xenia",
            "sample_rate": 48000,
        }
    ]
    assert audio.shape == (3,)
    assert audio.device.type == "cpu"


@pytest.mark.parametrize(
    "source",
    [
        torch.tensor([1, -1], dtype=torch.int64),
        torch.tensor([True, False], dtype=torch.bool),
        torch.tensor([0.25, -0.25], dtype=torch.float64),
    ],
)
def test_normalizes_returned_pcm_to_detached_cpu_float32(source: torch.Tensor) -> None:
    runtime = SileroRuntime(FakeModel(["xenia"], source), _spec())

    audio = runtime.synthesize(_chunk(), "xenia")

    assert audio.device.type == "cpu"
    assert audio.dtype is torch.float32
    assert not audio.requires_grad


def test_returned_pcm_is_detached() -> None:
    source = torch.tensor([0.25, -0.25], dtype=torch.float64, requires_grad=True)
    runtime = SileroRuntime(FakeModel(["xenia"], source), _spec())

    audio = runtime.synthesize(_chunk(), "xenia")

    assert not audio.requires_grad


def test_maps_pcm_conversion_failure_to_synthesis_error() -> None:
    class FloatFailureTensor(torch.Tensor):
        @staticmethod
        def __new__(cls) -> FloatFailureTensor:
            return torch.Tensor._make_subclass(cls, torch.tensor([0.25]), False)

        def detach(self) -> FloatFailureTensor:
            return self

        def cpu(self) -> FloatFailureTensor:
            return self

        def float(self) -> FloatFailureTensor:
            raise RuntimeError("float conversion failed")

    runtime = SileroRuntime(FakeModel(["xenia"], FloatFailureTensor()), _spec())

    with pytest.raises(SynthesisError, match="float conversion failed"):
        runtime.synthesize(_chunk(), "xenia")


def test_wraps_model_synthesis_failure_as_synthesis_error() -> None:
    runtime = SileroRuntime(FailingModel(["xenia"]), _spec())

    with pytest.raises(SynthesisError, match="boom"):
        runtime.synthesize(_chunk(), "xenia")


@pytest.mark.parametrize(
    "audio",
    [
        None,
        torch.empty(0),
        torch.ones((2, 3)),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([0.0, float("inf")]),
    ],
)
def test_rejects_non_mono_empty_or_non_finite_pcm(audio: object) -> None:
    runtime = SileroRuntime(FakeModel(["xenia"], audio), _spec())

    with pytest.raises(SynthesisError, match="non-empty finite mono PCM"):
        runtime.synthesize(_chunk(), "xenia")


def test_loads_a_verified_cached_package_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel(["xenia"])
    events: list[tuple[object, ...]] = []
    path = tmp_path / "v5_5_ru.pt"

    class FakeImporter:
        def __init__(self, source: str) -> None:
            events.append(("open", source))

        def load_pickle(self, package: str, resource: str) -> FakeModel:
            events.append(("load_pickle", package, resource))
            return model

    class FakeStore:
        def ensure(self, spec: object, validator: object, progress: object) -> Path:
            events.append(("ensure", spec, validator, progress))
            return path

    monkeypatch.setattr(torch.package, "PackageImporter", FakeImporter)

    runtime = SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]

    assert runtime.speakers == ("xenia",)
    assert model.device == torch.device("cpu")
    assert [event[0] for event in events] == ["ensure", "open", "load_pickle"]
    assert events[1:] == [
        ("open", str(path)),
        ("load_pickle", "tts_models", "model"),
    ]


def test_load_validates_a_fresh_download_before_runtime_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel(["xenia"])
    events: list[tuple[object, ...]] = []
    path = tmp_path / "v5_5_ru.pt"

    class FakeImporter:
        def __init__(self, source: str) -> None:
            events.append(("open", source))

        def load_pickle(self, package: str, resource: str) -> FakeModel:
            events.append(("load_pickle", package, resource))
            return model

    class FreshDownloadStore:
        def ensure(self, spec: object, validator: object, progress: object) -> Path:
            events.append(("verified", spec, progress))
            validator(path)  # type: ignore[operator]
            events.append(("validated",))
            return path

    monkeypatch.setattr(torch.package, "PackageImporter", FakeImporter)

    SileroRuntime.load(FreshDownloadStore(), _spec())  # type: ignore[arg-type]

    assert events == [
        ("verified", _spec(), None),
        ("open", str(path)),
        ("validated",),
        ("open", str(path)),
        ("load_pickle", "tts_models", "model"),
    ]


def test_load_reuses_its_single_deserialized_model_for_every_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel(["xenia"])
    imports: list[Path] = []
    path = tmp_path / "v5_5_ru.pt"

    class FakeImporter:
        def __init__(self, source: str) -> None:
            imports.append(Path(source))

        def load_pickle(self, package: str, resource: str) -> FakeModel:
            assert (package, resource) == ("tts_models", "model")
            return model

    class FakeStore:
        def ensure(self, *_: object) -> Path:
            return path

    monkeypatch.setattr(torch.package, "PackageImporter", FakeImporter)
    runtime = SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]

    runtime.synthesize(_chunk(), "xenia")
    runtime.synthesize(_chunk(), "xenia")

    assert imports == [path]
    assert len(model.calls) == 2


def test_maps_package_load_failure_to_model_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenImporter:
        def __init__(self, _: str) -> None:
            raise RuntimeError("broken package")

    class FakeStore:
        def ensure(self, *_: object) -> Path:
            return tmp_path / "v5_5_ru.pt"

    monkeypatch.setattr(torch.package, "PackageImporter", BrokenImporter)

    with pytest.raises(ModelError, match="broken package"):
        SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]


def test_maps_cpu_setup_failure_to_model_error() -> None:
    class BrokenCpuModel(FakeModel):
        def to(self, _: torch.device) -> FakeModel:
            raise RuntimeError("cpu unavailable")

    with pytest.raises(ModelError, match="cpu unavailable"):
        SileroRuntime(BrokenCpuModel(["xenia"]), _spec())


def test_maps_eval_setup_failure_to_model_error() -> None:
    class BrokenEvalModel(FakeModel):
        def eval(self) -> FakeModel:
            raise RuntimeError("eval unavailable")

    with pytest.raises(ModelError, match="eval unavailable"):
        SileroRuntime(BrokenEvalModel(["xenia"]), _spec())


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(130)])
def test_does_not_wrap_control_flow_from_eval(error: BaseException) -> None:
    class InterruptedEvalModel(FakeModel):
        def eval(self) -> FakeModel:
            raise error

    with pytest.raises(type(error)):
        SileroRuntime(InterruptedEvalModel(["xenia"]), _spec())


def test_rejects_absent_invalid_or_empty_runtime_speakers() -> None:
    absent = FakeModel(["xenia"])
    del absent.speakers
    invalid = FakeModel(["xenia"])
    invalid.speakers = "xenia"
    empty = FakeModel([])

    for model, message in [
        (absent, "missing or invalid"),
        (invalid, "missing or invalid"),
        (empty, "speakers is empty"),
    ]:
        with pytest.raises(ModelError, match=message):
            SileroRuntime(model, _spec())


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(130)])
def test_does_not_wrap_control_flow_from_synthesis(error: BaseException) -> None:
    class InterruptedModel(FakeModel):
        def apply_tts(self, **kwargs: object) -> object:
            raise error

    runtime = SileroRuntime(InterruptedModel(["xenia"]), _spec())

    with pytest.raises(type(error)):
        runtime.synthesize(_chunk(), "xenia")


def test_does_not_wrap_control_flow_from_package_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedImporter:
        def __init__(self, _: str) -> None:
            raise KeyboardInterrupt

    class FakeStore:
        def ensure(self, *_: Any) -> Path:
            return tmp_path / "v5_5_ru.pt"

    monkeypatch.setattr(torch.package, "PackageImporter", InterruptedImporter)

    with pytest.raises(KeyboardInterrupt):
        SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]
