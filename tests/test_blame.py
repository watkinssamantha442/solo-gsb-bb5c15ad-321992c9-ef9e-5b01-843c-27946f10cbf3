"""GET /documents/{id}/blame 历史字段归因端到端集成测试（独立临时数据库）。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.blame import RevisionSnapshot, blame_snapshots
from app.config import settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 与 test_api 相同：直接改 settings.db_path，避免模块单例身份漂移
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _create(client, content=None, author="alice", doc_id="cfg"):
    body = {"id": doc_id, "author": author}
    if content is not None:
        body["content"] = content
    r = client.post("/documents", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _patch(client, base_revision, request_id, patch, author=None, doc_id="cfg"):
    body = {
        "base_revision": base_revision,
        "request_id": request_id,
        "patch": patch,
    }
    if author is not None:
        body["author"] = author
    r = client.patch(f"/documents/{doc_id}", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _revert(client, revision, request_id, author=None, doc_id="cfg"):
    body = {"revision": revision, "request_id": request_id}
    if author is not None:
        body["author"] = author
    r = client.post(f"/documents/{doc_id}/reverts", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _blame(client, revision, doc_id="cfg"):
    return client.get(f"/documents/{doc_id}/blame", params={"revision": revision})


def _by_path(body):
    return {item["path"]: item for item in body["items"]}


# ---------------- 基本归因 ----------------


def test_revision_one_owns_all_created_leaves(client):
    created = _create(client, {"a": 1, "c": {"x": 1}, "note": None, "tags": [1]})
    r = _blame(client, 1)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document_id"] == "cfg"
    assert body["revision"] == 1
    assert [item["path"] for item in body["items"]] == ["/a", "/c/x", "/note", "/tags"]
    for item in body["items"]:
        assert item["last_changed_revision"] == 1
        assert item["author"] == "alice"
        assert item["request_id"] is None
        assert item["created_at"] == created["created_at"]
    assert _by_path(body)["/note"]["value"] is None
    assert _by_path(body)["/tags"]["value"] == [1]


def test_each_change_is_attributed_to_its_revision(client):
    _create(client, {"a": 1, "b": 2, "c": {"x": 1}})
    p2 = _patch(client, 1, "r2", {"a": 100}, author="bob")
    p3 = _patch(client, 2, "r3", {"c": {"y": 2}}, author="carol")
    body = _blame(client, 3).json()
    items = _by_path(body)
    assert items["/a"] == {
        "path": "/a",
        "value": 100,
        "last_changed_revision": 2,
        "author": "bob",
        "request_id": "r2",
        "created_at": p2["created_at"],
    }
    assert items["/b"]["last_changed_revision"] == 1
    assert items["/c/x"]["last_changed_revision"] == 1
    assert items["/c/y"] == {
        "path": "/c/y",
        "value": 2,
        "last_changed_revision": 3,
        "author": "carol",
        "request_id": "r3",
        "created_at": p3["created_at"],
    }
    # 结果按 path 稳定排序
    assert [item["path"] for item in body["items"]] == [
        "/a",
        "/b",
        "/c/x",
        "/c/y",
    ]


def test_unchanged_leaf_keeps_earlier_attribution_across_revisions(client):
    _create(client, {"a": 1, "b": 2})
    _patch(client, 1, "r2", {"a": 10}, author="bob")
    _patch(client, 2, "r3", {"a": 100}, author="carol")
    body = _blame(client, 3).json()
    items = _by_path(body)
    assert items["/a"]["last_changed_revision"] == 3
    # /b 从未改动，始终归属于创建修订（即使每一步都重新经过快照比较）
    assert items["/b"]["last_changed_revision"] == 1
    assert items["/b"]["author"] == "alice"


def test_target_revision_only_includes_leaves_present_then(client):
    _create(client, {"a": 1, "b": 2})
    _patch(client, 1, "r2", {"b": None, "c": 3})
    body2 = _blame(client, 2).json()
    assert [item["path"] for item in body2["items"]] == ["/a", "/c"]
    # 目标是历史修订 1：只含 rev1 当时的叶子
    body1 = _blame(client, 1).json()
    assert [item["path"] for item in body1["items"]] == ["/a", "/b"]
    assert _by_path(body1)["/b"]["last_changed_revision"] == 1


# ---------------- 删除后重建 ----------------


def test_recreated_leaf_is_owned_by_rebuild_revision_even_with_same_value(client):
    _create(client, {"a": 1})
    _patch(client, 1, "r2", {"a": None}, author="bob")          # 删除 /a
    rebuild = _patch(client, 2, "r3", {"a": 1}, author="carol")  # 同值重建
    body = _blame(client, 3).json()
    assert body["items"] == [
        {
            "path": "/a",
            "value": 1,
            "last_changed_revision": 3,
            "author": "carol",
            "request_id": "r3",
            "created_at": rebuild["created_at"],
        }
    ]
    # 中间的 rev2 目标快照里 /a 不存在
    assert _blame(client, 2).json()["items"] == []


def test_recreated_nested_subtree_owned_by_rebuild_revision(client):
    _create(client, {"cfg": {"x": 1, "y": 2}})
    _patch(client, 1, "r2", {"cfg": None}, author="bob")
    rebuild = _patch(
        client, 2, "r3", {"cfg": {"x": 1, "y": 2}}, author="carol"
    )  # 同值整树重建
    body = _blame(client, 3).json()
    assert [item["path"] for item in body["items"]] == ["/cfg/x", "/cfg/y"]
    assert all(item["last_changed_revision"] == 3 for item in body["items"])
    assert all(item["request_id"] == "r3" for item in body["items"])
    assert all(item["created_at"] == rebuild["created_at"] for item in body["items"])


def test_subtree_swap_flips_all_leaves_inside(client):
    # /n 在 rev1 是对象，rev2 变成标量，rev3 再变回对象：内部叶子全部重新归属
    _create(client, {"n": {"x": 1}})
    _patch(client, 1, "r2", {"n": 42}, author="bob")
    _patch(client, 2, "r3", {"n": {"x": 1}}, author="carol")
    body = _blame(client, 3).json()
    assert body["items"] == [
        {
            "path": "/n/x",
            "value": 1,
            "last_changed_revision": 3,
            "author": "carol",
            "request_id": "r3",
            "created_at": body["items"][0]["created_at"],
        }
    ]
    # rev2 目标：/n 是标量叶子
    items2 = _blame(client, 2).json()["items"]
    assert [i["path"] for i in items2] == ["/n"]
    assert items2[0]["value"] == 42
    assert items2[0]["last_changed_revision"] == 2


# ---------------- 数组 / 标量整体叶子 ----------------


def test_arrays_are_whole_leaves_not_indexed(client):
    _create(client, {"tags": [1, 2], "o": {"list": ["a"]}})
    _patch(client, 1, "r2", {"tags": [1, 2, 3]}, author="bob")
    p3 = _patch(client, 2, "r3", {"z": 1, "o": {"list": ["a"]}}, author="carol")
    body = _blame(client, 3).json()
    items = _by_path(body)
    # 数组不按下标展开：/tags 是一个叶子，改动后归 rev2
    assert items["/tags"]["value"] == [1, 2, 3]
    assert items["/tags"]["last_changed_revision"] == 2
    assert items["/tags"]["request_id"] == "r2"
    # 数组原样保留（即使外层对象被重新提交）→ 沿用旧归属
    assert items["/o/list"]["value"] == ["a"]
    assert items["/o/list"]["last_changed_revision"] == 1
    assert items["/z"]["last_changed_revision"] == 3
    assert p3["revision"] == 3  # rev3 确实存在，但不改变 /o/list 的归属


def test_scalar_root_document_is_single_leaf(client):
    _create(client, {"a": 1})
    _patch(client, 1, "r2", [1, 2], author="bob")  # 根整体替换为数组
    body = _blame(client, 2).json()
    assert body["items"] == [
        {
            "path": "",
            "value": [1, 2],
            "last_changed_revision": 2,
            "author": "bob",
            "request_id": "r2",
            "created_at": body["items"][0]["created_at"],
        }
    ]
    _patch(client, 2, "r3", "hello", author="carol")
    body3 = _blame(client, 3).json()
    assert len(body3["items"]) == 1
    assert body3["items"][0]["path"] == ""
    assert body3["items"][0]["value"] == "hello"
    assert body3["items"][0]["last_changed_revision"] == 3


def test_null_root_is_leaf_and_empty_object_has_no_leaves(client):
    _create(client, {"a": 1})
    _patch(client, 1, "r2", None, author="bob")  # 根整体替换为真实 null
    body = _blame(client, 2).json()
    assert body["items"] == [
        {
            "path": "",
            "value": None,
            "last_changed_revision": 2,
            "author": "bob",
            "request_id": "r2",
            "created_at": body["items"][0]["created_at"],
        }
    ]
    _patch(client, 2, "r3", {}, author="carol")  # 空对象：没有叶子
    assert _blame(client, 3).json()["items"] == []


def test_number_type_byte_change_counts_as_change(client):
    # 1 → 1.0 / 1 → true：Python == 判等但落库字节不同，归因必须前移
    _create(client, {"i": 1, "b": 1})
    _patch(client, 1, "r2", {"i": 1.0, "b": True}, author="bob")
    items = _by_path(_blame(client, 2).json())
    assert [items["/b"]["last_changed_revision"], items["/i"]["last_changed_revision"]] == [
        2,
        2,
    ]


# ---------------- 路径转义与排序 ----------------


def test_special_keys_escaped_per_rfc6901_and_sorted(client):
    _create(client, {"a/b": 1, "t~k": 1, "m": 1})
    _patch(client, 1, "r2", {"a/b": 2}, author="bob")
    body = _blame(client, 2).json()
    assert [item["path"] for item in body["items"]] == ["/a~1b", "/m", "/t~0k"]
    items = _by_path(body)
    assert items["/a~1b"]["last_changed_revision"] == 2
    assert items["/m"]["last_changed_revision"] == 1
    assert items["/t~0k"]["last_changed_revision"] == 1


# ---------------- 撤销等普通修订 ----------------


def test_revert_rebuilds_owned_by_revert_revision(client):
    # rev1 a=1 → rev2 a=100 → rev3 撤销 rev2（a 回到 1）：
    # /a 被“重建”为旧值，但归属于撤销修订 rev3
    _create(client, {"a": 1})
    _patch(client, 1, "r2", {"a": 100}, author="bob")
    reverted = _revert(client, 2, "undo-1", author="dave")
    assert reverted["revision"] == 3
    body = _blame(client, 3).json()
    assert body["items"] == [
        {
            "path": "/a",
            "value": 1,
            "last_changed_revision": 3,
            "author": "dave",
            "request_id": "undo-1",
            "created_at": reverted["created_at"],
        }
    ]


# ---------------- 404 / 422 / 只读 ----------------


def test_missing_document_returns_404(client):
    r = _blame(client, 1, doc_id="ghost")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "not_found"


def test_missing_revision_returns_404(client):
    _create(client, {"a": 1})
    r = _blame(client, 99)
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "revision_not_found"
    assert detail["document_id"] == "cfg"
    assert detail["revision"] == 99


def test_revision_param_required_and_positive(client):
    _create(client, {"a": 1})
    assert client.get("/documents/cfg/blame").status_code == 422
    assert _blame(client, 0).status_code == 422
    assert client.get("/documents/cfg/blame?revision=x").status_code == 422
    assert client.get("/documents/cfg/blame?revision=-3").status_code == 422


def test_blame_is_read_only_and_does_not_advance_history(client):
    _create(client, {"a": 1})
    _patch(client, 1, "r2", {"a": 2}, author="bob")
    before = client.get("/documents/cfg/revisions?order=asc").json()["items"]
    for _ in range(3):
        r = _blame(client, 2)
        assert r.status_code == 200
    after = client.get("/documents/cfg/revisions?order=asc").json()["items"]
    assert after == before  # 没有产生任何新修订
    doc = client.get("/documents/cfg").json()
    assert doc["revision"] == 2  # 当前修订不变


# ---------------- 纯函数单元测试 ----------------


def _snap(revision, content, author=None, request_id=None, created_at=None):
    return RevisionSnapshot(
        revision=revision,
        content=content,
        author=author,
        request_id=request_id,
        created_at=created_at or f"2026-01-01T00:00:0{revision}.000000Z",
    )


def test_blame_snapshots_pure_function_basic():
    items = blame_snapshots(
        [
            _snap(1, {"a": 1, "b": {"c": 2}}, author="alice"),
            _snap(2, {"a": 9, "b": {"c": 2}}, author="bob", request_id="r2"),
        ]
    )
    assert [i["path"] for i in items] == ["/a", "/b/c"]
    assert items[0]["last_changed_revision"] == 2
    assert items[0]["request_id"] == "r2"
    assert items[1]["last_changed_revision"] == 1
    assert items[1]["author"] == "alice"


def test_blame_snapshots_delete_and_recreate_with_same_value():
    items = blame_snapshots(
        [
            _snap(1, {"a": 1}),
            _snap(2, {}, author="bob", request_id="del"),
            _snap(3, {"a": 1}, author="carol", request_id="add"),
        ]
    )
    assert items[0]["last_changed_revision"] == 3
    assert items[0]["request_id"] == "add"
