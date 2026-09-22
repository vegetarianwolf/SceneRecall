import json

import httpx
import pytest
from scenerecall.providers import ProviderError, ProviderManager
from scenerecall.search import SearchEngine, normalize


def record(id, text, kind="subtitle", start=0, end=1000, **extra):
    return {"id": id, "record_id": id, "asset_id": "film-one", "work_id": "work-one", "title": "合成测试片",
            "kind": kind, "text": text, "start_ms": start, "end_ms": end, "thumbnail": None,
            "evidence_frame_ids": ["frame-1"] if kind == "visual" else [], **extra}


@pytest.fixture
def local(tmp_path):
    providers = ProviderManager(tmp_path / "settings")
    return SearchEngine(tmp_path / "index", providers)


@pytest.mark.asyncio
async def test_chinese_substring_and_adjacent_cue_retrieval(local):
    await local.rebuild([
        record("s1", "我们一起回家。"),
        record("s2", "明天再来看你。", start=1100, end=2000),
        record("s3", "不会跨过长空白。", start=9000, end=10000),
    ])
    direct = await local.search("一起回家")
    assert direct["results"][0]["id"] == "s1"
    assert direct["results"][0]["match_type"] == "full"
    across = await local.search("回家明天再来")
    context = next(row for row in across["results"] if row.get("is_context"))
    assert context["source_ids"] == ["s1", "s2"]
    assert context["start_ms"] == 0 and context["end_ms"] == 2000
    assert local.status()["contexts"] == 1


@pytest.mark.asyncio
async def test_filters_character_aliases_and_combination_is_not_full(local):
    await local.rebuild([
        record("v1", "左边的人站立；右边的人拿着杯子", kind="visual", season=1, episode=2,
               entities=[{"id": "p1", "name": "小明", "aliases": ["阿明"], "description": "穿蓝衣"}]),
        record("v2", "左边的人拿着杯子", kind="visual", asset_id="film-two", season=2, episode=2),
    ])
    result = await local.search("左边的人拿杯子", {"season": 1, "episode": 2, "character": "阿明"})
    assert [row["id"] for row in result["results"]] == ["v1"]
    assert result["results"][0]["match_type"] == "partial"
    assert (await local.search("杯子", {"character": "未知角色"}))["results"] == []


@pytest.mark.asyncio
async def test_search_input_cannot_inject_sql_or_fts(local):
    await local.rebuild([record("s1", "我们回家")])
    assert (await local.search("' OR 1=1 --", {"asset_id": "' OR 1=1 --"}))["results"] == []
    assert local.status()["records"] == 1
    assert normalize("ＡＢＣ，回 家！") == "abc回家"


@pytest.mark.asyncio
async def test_function_words_do_not_create_unrelated_negative_results(local):
    await local.rebuild([record("v1", "穿红衣的人在画面左侧拿着杯子", kind="visual")])
    assert (await local.search("不存在的海豚跳舞"))["results"] == []
    assert (await local.search("找到拿杯子的那个人"))["results"][0]["id"] == "v1"


@pytest.mark.asyncio
async def test_rebuild_is_complete_projection_and_upsert_replaces_context(local):
    await local.rebuild([record("s1", "你好"), record("s2", "世界", start=1200, end=2000)])
    local.upsert([record("s1", "再见")])
    assert (await local.search("你好"))["results"] == []
    assert any(row.get("source_ids") == ["s1", "s2"] for row in (await local.search("再见世界"))["results"])
    await local.rebuild([record("s1", "再见")])
    assert local.status()["records"] == 1
    assert local.status()["contexts"] == 0


def vector_provider(tmp_path):
    calls = []
    def handler(request):
        calls.append((request.url.path, json.loads(request.content)))
        assert request.url.path.endswith("/embeddings"), "search must never request vision/subtitles"
        payload = json.loads(request.content)
        vectors = []
        for text in payload["input"]:
            if "杯子" in text or "喝水" in text:
                vector = [1, 0, 0] if payload["model"] == "embed-one" else [0, 1, 0]
            else:
                vector = [0, 0, 1]
            vectors.append({"index": len(vectors), "embedding": vector})
        return httpx.Response(200, json={"data": vectors, "usage": {"prompt_tokens": 20, "total_tokens": 20}})
    providers = ProviderManager(tmp_path / "profiles", transport=httpx.MockTransport(handler))
    providers.upsert({"id": "embed", "name": "向量一", "base_url": "http://test.local/v1", "model": "embed-one",
                      "capabilities": ["embedding"], "secret_mode": "none"})
    return providers, calls


@pytest.mark.asyncio
async def test_real_lancedb_semantic_index_cache_namespaces_and_accounting(tmp_path):
    providers, calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    records = [record("v1", "人物拿着杯子", kind="visual"), record("v2", "树上飞鸟", kind="visual")]
    before, after = [], []
    built = await engine.rebuild(records, "embed", before_batch=before.append, on_batch=after.append)
    assert before == [2] and after[0].request_count == 1
    assert built["request_count"] == 1 and built["usage"]["input_tokens"] == 20
    result = await engine.search("喝水", embedding_profile_id="embed")
    assert [row["id"] for row in result["results"]] == ["v1"]
    assert result["results"][0]["match_type"] == "partial"
    cached = await engine.rebuild(records, "embed")
    assert cached["request_count"] == 0 and cached["cache_hits"] == 2
    assert len(calls) == 2  # one indexing request and one query embedding
    providers.upsert({"id": "embed", "model": "embed-two"})
    result = await engine.search("喝水", embedding_profile_id="embed")
    assert result["results"] == [] and "尚未建立索引" in result["degraded"][0]
    await engine.rebuild(records, "embed")
    assert len(engine.status()["embedding_namespaces"]) == 2
    assert (await engine.search("喝水", embedding_profile_id="embed"))["results"][0]["id"] == "v1"


@pytest.mark.asyncio
async def test_failed_embedding_rebuild_keeps_previous_lexical_and_vector_index(tmp_path):
    providers, _calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    await engine.rebuild([record("v1", "人物拿着杯子", kind="visual")], "embed")
    table = engine.status()["embedding_namespaces"][0]["table_name"]
    providers._transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "no key"}))
    errors = []
    with pytest.raises(ProviderError):
        await engine.rebuild([record("v2", "全新资料", kind="visual")], "embed", on_batch=errors.append)
    assert errors[0].request_count == 1
    assert engine.status()["embedding_namespaces"][0]["table_name"] == table
    assert (await engine.search("杯子"))["results"][0]["id"] == "v1"


@pytest.mark.asyncio
async def test_semantic_prefilter_and_stale_vectors_do_not_return_old_evidence(tmp_path):
    providers, _calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    await engine.rebuild([record("v1", "拿杯子", kind="visual"),
                          record("v2", "另一个杯子", kind="visual", asset_id="film-two")], "embed")
    result = await engine.search("喝水", {"asset_id": "film-two"}, embedding_profile_id="embed")
    assert [row["id"] for row in result["results"]] == ["v2"]
    engine.upsert([record("v2", "纠正为树上飞鸟", kind="visual", asset_id="film-two")])
    stale = await engine.search("喝水", {"asset_id": "film-two"}, embedding_profile_id="embed")
    assert stale["results"] == []
    assert any("旧向量已跳过" in warning for warning in stale["degraded"])


@pytest.mark.asyncio
async def test_decision_can_reject_keyword_cooccurrence_without_vision_calls(tmp_path):
    paths = []
    def handler(request):
        paths.append(request.url.path)
        payload = json.loads(request.content)
        assert all("image_url" not in str(message["content"]) for message in payload["messages"])
        data = {"candidates": [{"id": "v1", "score": 0.1, "match_type": "none", "evidence_ids": []}], "expand_search": False}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(data)}}]})
    providers = ProviderManager(tmp_path / "profiles", transport=httpx.MockTransport(handler))
    providers.upsert({"id": "judge", "name": "决策", "base_url": "http://test.local/v1", "model": "text-model",
                      "capabilities": ["decision"], "secret_mode": "none"})
    engine = SearchEngine(tmp_path / "index", providers)
    await engine.rebuild([record("v1", "左边的人站立；右边的人拿杯子", kind="visual")])
    result = await engine.search("左边的人拿杯子", decision_profile_id="judge")
    assert result["results"] == [] and "未找到" in result["explanation"]
    assert paths == ["/v1/chat/completions"]


@pytest.mark.asyncio
async def test_rebuild_budget_callback_stops_before_network(tmp_path):
    providers, calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    def stop(_):
        raise ValueError("budget exhausted")
    with pytest.raises(ValueError, match="budget"):
        await engine.rebuild([record("v1", "杯子", kind="visual")], "embed", before_batch=stop)
    assert not calls and engine.status()["records"] == 0


@pytest.mark.asyncio
async def test_concurrent_annotation_update_is_not_replaced_by_old_rebuild_snapshot(tmp_path):
    providers, _calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    await engine.rebuild([record("v1", "原来的描述", kind="visual")])
    def edit_while_model_is_running(_):
        engine.upsert([record("v1", "人工纠正为杯子", kind="visual")])
    with pytest.raises(ValueError, match="资料发生变化"):
        await engine.rebuild([record("v1", "原来的描述", kind="visual")], "embed",
                             before_batch=edit_while_model_is_running)
    assert (await engine.search("人工纠正"))["results"][0]["text"] == "人工纠正为杯子"
