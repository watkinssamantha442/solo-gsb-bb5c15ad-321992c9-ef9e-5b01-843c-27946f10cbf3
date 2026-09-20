"""历史字段归因（blame）：目标快照中每个叶子值最后一次被改动的修订。

规则概述
--------
* 按修订顺序比较**相邻快照**（1→2→…→目标修订）：
  - 对象递归到叶子；数组、标量与 ``null`` 整体视为一个叶子，
    与差异审计 / RFC 7386 的整体值语义一致；
  - 修订 1（创建快照）中的全部叶子归属于修订 1；
  - 每个后续修订：叶子路径仅存在于新快照（新增 / 子树重建）或两侧
    规范化值不同（替换）→ 归属于当前修订；路径两侧都在且值相同 →
    沿用旧归属；仅存在于旧快照（删除）→ 从归属表移除。
* 因此路径被删除后再次出现时，即使值与删除前完全相同，也归属于
  **重建它的修订**——中间的删除已使旧归属失效。
* 结果按 RFC 6901 JSON Pointer 路径稳定排序；每项给出目标快照中的
  值与产生该叶子的修订的元数据（revision / author / request_id /
  created_at）。

算法本身不依赖 FastAPI / SQLite，可独立单测。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .merge import json_pointer


@dataclass(frozen=True)
class RevisionSnapshot:
    """归因所用的单个修订快照及其元数据。"""

    revision: int
    content: Any
    author: str | None
    request_id: str | None
    created_at: str


def _canonical(value: Any) -> str:
    """确定性序列化（与 db.canonical 同参数），用于整体值相等判断。

    区分 ``true``/``1``、``1``/``1.0`` 这类 Python ``==`` 会判等、
    但落库字节不同的值，保证“快照间字节不同即视为叶子被改动”。
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _leaf_values(value: Any) -> dict[str, Any]:
    """递归收集 *value* 的全部叶子：``{RFC6901 路径: 叶子值}``。

    对象按键递归（空对象没有叶子）；数组与标量（含 ``null``）整体
    作为一个叶子挂在当前路径上，根本身是标量 / 数组时键为 ``""``。
    """
    leaves: dict[str, Any] = {}

    def node(current: Any, path: list[str]) -> None:
        if isinstance(current, dict):
            for key, child in current.items():
                node(child, path + [key])
        else:
            leaves[json_pointer(path)] = current

    node(value, [])
    return leaves


def blame_snapshots(
    snapshots: list[RevisionSnapshot],
) -> list[dict[str, Any]]:
    """按修订顺序归并相邻快照，返回目标快照（最后一个 *snapshots*）
    每个叶子的归因项，按路径稳定排序。

    *snapshots* 必须为修订 1..N 的连续非空序列。每项形如::

        {"path", "value", "last_changed_revision", "author",
         "request_id", "created_at"}
    """
    if not snapshots:
        raise ValueError("snapshots 不能为空")

    # path -> 归属该叶子的修订快照（修订 1 的全部叶子归属于创建修订）
    attributed: dict[str, RevisionSnapshot] = {}
    previous_leaves: dict[str, Any] = {}

    for index, snapshot in enumerate(snapshots):
        current_leaves = _leaf_values(snapshot.content)
        if index == 0:
            attributed = {path: snapshot for path in current_leaves}
        else:
            next_attributed: dict[str, RevisionSnapshot] = {}
            for path, value in current_leaves.items():
                old = previous_leaves.get(path)
                # 路径两侧都在且整体值字节相同 → 沿用旧归属；
                # 新增（路径缺失过，含“删除后又出现”）或值不同 → 归当前修订。
                if path in previous_leaves and _canonical(old) == _canonical(value):
                    next_attributed[path] = attributed[path]
                else:
                    next_attributed[path] = snapshot
            attributed = next_attributed
        previous_leaves = current_leaves

    items = [
        {
            "path": path,
            "value": previous_leaves[path],
            "last_changed_revision": owner.revision,
            "author": owner.author,
            "request_id": owner.request_id,
            "created_at": owner.created_at,
        }
        for path, owner in attributed.items()
    ]
    # 路径互不重复，按指针字符串排序即完全确定（Python sort 本身稳定）
    items.sort(key=lambda item: item["path"])
    return items
