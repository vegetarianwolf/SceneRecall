"""Worker regression cases with explicitly injected synthetic media/model adapters."""
from copy import deepcopy

import pytest
from PIL import Image

from scenerecall.jobs import JobQueue, StopJob, canonical_ocr_frames
from scenerecall.library import Library, read_json
from scenerecall.models import AnalysisInput, AssetInput
from scenerecall.providers import AIResult, ProviderError
from scenerecall.search import SearchEngine


class InjectedProvider:
    def __init__(self):
        self.profile = {"id": "test", "name": "test double", "base_url": "http://test.local/v1", "model": "synthetic",
                        "capabilities": ["vision", "subtitle", "embedding"], "secret_mode": "none",
                        "input_price_per_million": 1, "output_price_per_million": 1}
        self.calls = []
        self.subtitle_inputs = []
        self.fail_embedding = False
        self.fail_window = None
        self.failed_once = False

    def get(self, id):
        assert id == "test"
        return deepcopy(self.profile)

    async def recognize_subtitles(self, id, frames):
        self.calls.append("subtitle")
        self.subtitle_inputs.append(deepcopy(frames))
        return AIResult([{"frame_id": frame["id"], "lines": [{"text": "合成字幕", "language": "zh", "uncertain": False}]}
                         for frame in frames], {}, 1, 0.01)

    async def analyze(self, id, frames, start_ms, end_ms):
        self.calls.append(("vision", start_ms))
        if start_ms == self.fail_window and not self.failed_once:
            self.failed_once = True
            raise ProviderError("测试故障", request_count=1, cost=0)
        return AIResult({"summary": "测试用白色杯子", "entities": [], "events": [],
                         "evidence_frame_ids": [frame["id"] for frame in frames]}, {}, 1, 0.01)

    async def embed(self, id, texts):
        self.calls.append("embedding")
        if self.fail_embedding:
            raise ProviderError("测试向量故障", request_count=1, cost=0)
        return AIResult([[1.0, 0.1] for _ in texts], {}, 1, 0.01)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    source = tmp_path / "explicit-synthetic-video.mp4"
    source.write_bytes(b"test fixture; decoded only by injected extractor")
    monkeypatch.setattr("scenerecall.media.probe", lambda path: {
        "duration_ms": 4000, "width": 32, "height": 32, "video_codec": "synthetic", "streams": []})
    monkeypatch.setattr("scenerecall.media.detect_shots", lambda path, start, end: [
        {"id": "synthetic-shot", "start_ms": start, "end_ms": end}])
    extractions = []
    def extract(path, times_ms, out_paths, crop=None):
        extractions.append(list(times_ms))
        frames = []
        for at, output in zip(times_ms, out_paths, strict=True):
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (at % 255, (at // 100) % 255, 0)).save(output)
            frames.append({"id": output.stem, "path": str(output), "at_ms": at + 125, "requested_at_ms": at})
        return frames
    monkeypatch.setattr("scenerecall.media.extract_frames", extract)
    library = Library(tmp_path / "library")
    asset = library.register(AssetInput(video_path=str(source), title="Synthetic only", subtitle_mode="embedded"))
    providers = InjectedProvider()
    search = SearchEngine(library.root / "indexes", providers)
    queue = JobQueue(library, providers, search)
    return queue, asset, extractions


def test_unknown_cost_remains_unknown_and_pauses_cost_budget(worker):
    queue, _asset, _extractions = worker
    job = queue.add("index", {"max_cost": 100, "max_requests": 100})
    queue.account(job["id"], AIResult({}, {}, 1, 0.1))
    queue.account(job["id"], AIResult({}, {}, 1, None))
    queue.account(job["id"], AIResult({}, {}, 1, 0.2))
    state = queue.get(job["id"])
    assert state["cost"] is None and state["cost_incomplete"]
    assert state["known_cost"] == pytest.approx(0.3)
    with pytest.raises(StopJob):
        queue.before_call(job["id"], queue.providers.get("test"))
    assert queue.get(job["id"])["status"] == "paused"
    queue.control(job["id"], "resume", {"max_cost": None})
    queue.before_call(job["id"], queue.providers.get("test"))


def test_inflight_request_reservation_survives_restart(worker):
    queue, _asset, _extractions = worker
    job = queue.add("index", {"max_requests": 20})
    queue.update(job["id"], status="running")
    queue.before_call(job["id"], queue.providers.get("test"))
    restarted = JobQueue(queue.library, queue.providers, queue.search)
    result = restarted.get(job["id"])
    assert result["request_count"] == 3
    assert result["uncertain_request_count"] == 3
    assert result["cost"] is None and result["cost_incomplete"]
    assert result["inflight_attempts"] == 0 and result["status"] == "queued"


@pytest.mark.parametrize("override", [{"max_requests": True}, {"max_requests": 1.5}, {"max_cost": float("nan")},
                                      {"max_requests": float("inf")}])
def test_budget_overrides_validate_type_and_finiteness(worker, override):
    queue, _asset, _extractions = worker
    job = queue.add("index", {})
    queue.update(job["id"], status="paused")
    with pytest.raises(ValueError):
        queue.control(job["id"], "resume", override)


@pytest.mark.asyncio
async def test_batch_ocr_preserves_actual_pts_and_lexical_survives_embedding_failure(worker):
    queue, asset, extractions = worker
    request = AnalysisInput(asset_id=asset["id"], stages=["subtitle"], max_requests=100)
    bindings = {"bindings": {"subtitle": "test", "embedding": "test"}}
    _, config = queue.validate(request, bindings)
    queue.providers.fail_embedding = True
    job = queue.add("analysis", config, asset["id"])
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "failed"
    assert extractions == [[0, 500, 1000, 1500, 2000, 2500, 3000, 3500]]
    assert queue.providers.subtitle_inputs[0][0]["at_ms"] == 125
    cues = queue.library.subtitles(asset["id"])
    assert cues[0]["start_ms"] == 125
    assert (await queue.search.search("合成字幕"))["results"]
    cache = next(queue.library.asset_dir(asset["id"]).glob("analyses/*/subtitle-batches/*.json"))
    assert read_json(cache)[0]["requested_times_ms"] == [0]
    # Updating non-model fields must not invalidate successful OCR or block resume.
    queue.providers.profile.update(name="renamed", input_price_per_million=2, secret_mode="session")
    queue.providers.fail_embedding = False
    queue.control(job["id"], "retry")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "completed"
    assert queue.providers.calls == ["subtitle", "embedding", "embedding"]
    assert len(extractions) == 1


@pytest.mark.asyncio
async def test_force_retry_only_retries_failed_window_and_frame_batches_are_cached(worker):
    queue, asset, extractions = worker
    queue.providers.fail_window = 1000
    request = AnalysisInput(asset_id=asset["id"], stages=["vision"], window_ms=1000, force=True, max_requests=100)
    _, config = queue.validate(request, {"bindings": {"vision": "test"}})
    job = queue.add("analysis", config, asset["id"])
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "partial"
    assert len(extractions) == 4
    assert len(queue.library.observations(asset["id"])) == 3
    queue.control(job["id"], "retry")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "completed"
    assert queue.providers.calls == [("vision", 0), ("vision", 1000), ("vision", 2000), ("vision", 3000), ("vision", 1000)]
    assert len(extractions) == 4
    assert len(queue.library.observations(asset["id"])) == 4


def test_same_pts_ocr_conflicts_become_uncertain_without_zero_length_cues():
    frames = [
        {"frame_id": "a", "at_ms": 100, "lines": [{"text": "你好", "language": "zh", "uncertain": False}]},
        {"frame_id": "b", "at_ms": 100, "lines": []},
        {"frame_id": "c", "at_ms": 500, "lines": []},
    ]
    normalized = canonical_ocr_frames(frames, 0, 1000, 500)
    assert len(normalized) == 2
    assert normalized[0]["end_ms"] == 500
    assert normalized[0]["lines"][0]["uncertain"] is True


def test_switching_model_during_job_pauses_before_request(worker):
    queue, _asset, _extractions = worker
    job = queue.add("index", {})
    original = queue.providers.get("test")
    queue.providers.profile["model"] = "changed-model"
    with pytest.raises(StopJob):
        queue.before_call(job["id"], original)
    assert queue.get(job["id"])["status"] == "paused"
    assert queue.providers.calls == []
