#
# Watashinosekaito Hatsune Miku
# Copyright (c) 2025-2026 ChordDotE. All rights reserved.
# SPDX-License-Identifier: LicenseRef-ChordDotE-NonCommercial-2.0
#
# NON-COMMERCIAL USE ONLY. NO BUSINESS USE. NO ADAPTATION.
#
# Business use (any use by or for a company, organization, or sole
# proprietor, WHETHER OR NOT ANY PAYMENT IS INVOLVED) and publishing or
# distributing modified versions are PROHIBITED IN ALL CASES; no permission
# is granted for them. Forks may be used only to propose improvements to the
# Author (e.g. pull requests). Commercial use, redistribution, and Adaptation
# are PROHIBITED without prior written permission. "Adaptation" includes
# manual reimplementation, automated translation, and generation using large
# language models or other machine learning, of software having equivalent or
# similar functionality. Permission is granted individually, in recorded form.
# Contact: https://x.com/ChordDotE
#
# 業務利用（対価の有無を問わず、法人・団体・個人事業主による利用と事業のための利用）と、
# 改変版の公開・配布は、許可の余地なく禁止します。フォークは作者への改善の提案
# （プルリクエスト等）のためだけに使えます。商用利用・再配布・翻案は、作者の許可が
# ない限り禁止します。翻案には、再実装・自動変換・大規模言語モデル等を用いた生成に
# よる、同等または類似の機能を持つソフトウェアの作成を含みます。
# 許可が必要な場合は、記録の残る形でご連絡ください。
#
# See LICENSE for full terms. 詳細は LICENSE を参照してください。
#

"""Long Memory VDB MCP サーバー。

旧 long_memory_mcp_server.py (SQLite + long_memory_db_service) を置き換える。
fastmcp で VDB_service の `/vdb/long_memory/*` エンドポイントを叩くツールを提供する。
memory_update_db.py が MCPServerStdio 経由でサブプロセスとして起動する。

提供ツール（旧サーバーとシグネチャ互換）:
  get_events / add_event / update_event / delete_event
  get_string_items / add_string_item / delete_string_item
  get_records / add_or_update_record / delete_record
  get_full_memory_json（VDB の always_in_context=True 全件 + 必要時は events/visited も）
  get_memory_size（件数集計）

環境変数:
  VDB_BASE_URL  VDB サービスの URL（デフォルト: http://127.0.0.1:8020）

データ規約: VDB ドキュメント metadata 構造は VDB_service.py のコメント参照。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import requests
from fastmcp import FastMCP

_BASE_URL = os.environ.get("VDB_BASE_URL", "http://127.0.0.1:8020").rstrip("/")
JST = timezone(timedelta(hours=9))

mcp = FastMCP("long-memory-vdb")


# =============================================================================
# string_items category → VDB metadata.category マッピング
# =============================================================================
_USER_CATEGORIES = {"fact", "like", "strength", "value"}
_AGENT_CATEGORIES = {"personality", "interest", "skill", "can_do", "cant_do", "state", "want_to"}

_RECORD_TYPE_MAP = {
    "visited":  ("visited",    False),
    "promise":  ("promises",   True),
    "habit":    ("procedural", False),
    "rule":     ("procedural", False),
    "todo":     ("working",    True),
    "goal":     ("working",    True),
    "context":  ("working",    True),
    "person":   ("user",       True),
}

_ID_PREFIX = {
    "fact": "FC", "like": "LK", "strength": "ST", "value": "VL",
    "personality": "PN", "interest": "IT", "skill": "SK",
    "can_do": "CD", "cant_do": "CN", "state": "SL", "want_to": "WT",
}


# =============================================================================
# HTTP ヘルパー
# =============================================================================


def _post(path: str, body: dict, timeout: int = 30) -> dict:
    r = requests.post(f"{_BASE_URL}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path: str, params: dict | None = None, timeout: int = 10) -> dict:
    r = requests.get(f"{_BASE_URL}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _delete(path: str, timeout: int = 10) -> dict:
    r = requests.delete(f"{_BASE_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _now_iso() -> str:
    return datetime.now(JST).isoformat()


def _vdb_upsert(items: list[dict]) -> dict:
    return _post("/vdb/long_memory/upsert", {"items": items})


def _vdb_get_by_category(category: str | None = None,
                         always_in_context: bool | None = None) -> list[dict]:
    params: dict = {}
    if category is not None:
        params["category"] = category
    if always_in_context is not None:
        params["always_in_context"] = "true" if always_in_context else "false"
    return _get("/vdb/long_memory/by_category", params=params).get("items", [])


def _vdb_get_by_id(doc_id: str) -> dict | None:
    """単一 ID で取得（簡易: by_category で全件取って絞り込み）。"""
    # VDB に直接 by_id エンドポイントは無いので、簡易に always_in_context=true/false 両方検索
    for aic in (True, False):
        items = _vdb_get_by_category(always_in_context=aic)
        for it in items:
            if it.get("id") == doc_id:
                return it
    return None


# =============================================================================
# Events
# =============================================================================


def _event_document_text(start_dt: str, place: str, what: str,
                          points: list[str], feelings: list[str], who: list[str]) -> str:
    parts = [start_dt]
    if place:
        parts.append(place)
    parts.append(what)
    if points:
        parts.append("ポイント: " + " / ".join(points))
    if feelings:
        parts.append("感情: " + " / ".join(feelings))
    if who:
        parts.append("関係者: " + " / ".join(map(str, who)))
    return "\n".join(parts)


@mcp.tool()
def get_events(importance: str = "") -> list[dict]:
    """イベント一覧を取得する。importance で絞り込み可（大/中/小）。"""
    items = _vdb_get_by_category(category="events")
    result: list[dict] = []
    for it in items:
        try:
            data = json.loads(it.get("metadata", {}).get("data_json") or "{}")
        except json.JSONDecodeError:
            continue
        if importance and data.get("importance", "中") != importance:
            continue
        inner = json.loads(data.get("data") or "{}") if isinstance(data.get("data"), str) else (data.get("data") or {})
        result.append({
            "id": data.get("id", it.get("id", "")),
            "start_dt": data.get("start_dt", ""),
            "end_dt": data.get("end_dt", ""),
            "place": data.get("place", ""),
            "what": data.get("what", ""),
            "importance": data.get("importance", "中"),
            "points": inner.get("points", []),
            "feelings": inner.get("feelings", []),
            "who": inner.get("who", []),
        })
    return result


@mcp.tool()
def add_event(
    id: str,
    start: str,
    what: str,
    importance: str = "中",
    end: str = "",
    place: str = "",
    points: list[str] | None = None,
    feelings: list[str] | None = None,
    who: list[str] | None = None,
) -> str:
    """イベントを追加する。id は E20260524-01 形式。

    end_dt が空（LLM が省略した）場合、start_dt と同じ値を補完する。
    これにより end_dt なしのレコードが生まれることを防ぐ（スキーマレベルガード）。
    """
    pts = points or []
    fls = feelings or []
    whs = who or []
    # end が空文字 or 空白なら start を補完
    end_filled = end.strip() if isinstance(end, str) else ""
    if not end_filled:
        end_filled = start
    inner_data = json.dumps({"points": pts, "feelings": fls, "who": whs}, ensure_ascii=False)
    row_data = {
        "id": id,
        "start_dt": start,
        "end_dt": end_filled,
        "place": place,
        "what": what,
        "importance": importance,
        "data": inner_data,
    }
    doc = _event_document_text(start, place, what, pts, fls, whs)
    metadata = {
        "category": "events",
        "type": "event",
        "always_in_context": False,
        "data_json": json.dumps(row_data, ensure_ascii=False),
        "date": start,
        "importance": importance,
        "registered_at": _now_iso(),
    }
    _vdb_upsert([{"id": id, "document": doc, "metadata": metadata}])
    return f"追加しました: {id}"


@mcp.tool()
def update_event(
    id: str,
    start_dt: str | None = None,
    end_dt: str | None = None,
    place: str | None = None,
    what: str | None = None,
    importance: str | None = None,
) -> str:
    """イベントのフィールドを更新する。None のフィールドはスキップ。"""
    existing = _vdb_get_by_id(id)
    if not existing:
        return f"[エラー] イベントが見つかりません: {id}"

    try:
        row_data = json.loads(existing.get("metadata", {}).get("data_json") or "{}")
    except json.JSONDecodeError:
        return f"[エラー] 既存データ JSON 破損: {id}"

    if start_dt is not None and start_dt.strip(): row_data["start_dt"] = start_dt
    # end_dt は空文字を渡された場合「未指定」扱いにしてスキップ（既存値保持）。
    # これにより LLM が誤って空文字を渡しても既存の end_dt が消えない。
    if end_dt is not None and end_dt.strip():     row_data["end_dt"] = end_dt
    if place is not None:    row_data["place"] = place
    if what is not None and what.strip(): row_data["what"] = what
    if importance is not None and importance.strip(): row_data["importance"] = importance

    # end_dt が依然として空なら start_dt と同値で補完（add_event と同じガード）
    if not row_data.get("end_dt", "").strip():
        row_data["end_dt"] = row_data.get("start_dt", "")

    try:
        inner = json.loads(row_data.get("data") or "{}")
    except json.JSONDecodeError:
        inner = {}

    doc = _event_document_text(
        row_data.get("start_dt", ""), row_data.get("place", ""), row_data.get("what", ""),
        inner.get("points", []), inner.get("feelings", []), inner.get("who", []),
    )
    metadata = {
        "category": "events", "type": "event", "always_in_context": False,
        "data_json": json.dumps(row_data, ensure_ascii=False),
        "date": row_data.get("start_dt", ""),
        "importance": row_data.get("importance", "中"),
        "registered_at": _now_iso(),
    }
    _vdb_upsert([{"id": id, "document": doc, "metadata": metadata}])
    return f"更新しました: {id}"


@mcp.tool()
def delete_event(id: str) -> str:
    """イベントを削除する。"""
    _delete(f"/vdb/long_memory/{id}")
    return f"削除しました: {id}"


# =============================================================================
# string_items
# =============================================================================


@mcp.tool()
def get_string_items(category: str = "") -> list[dict]:
    """string_items を取得する。category で絞り込み可。

    category は旧 SQLite の細分（fact/like/personality 等）。
    metadata.original_category で照合する。
    """
    # user / agent 両方から取得
    all_items: list[dict] = []
    for vcat in ("user", "agent"):
        for it in _vdb_get_by_category(category=vcat):
            meta = it.get("metadata", {})
            if meta.get("type") != "string_item":
                continue
            oc = meta.get("original_category", "")
            if category and oc != category:
                continue
            try:
                row_data = json.loads(meta.get("data_json") or "{}")
            except json.JSONDecodeError:
                continue
            all_items.append({
                "id": row_data.get("id", it.get("id", "")),
                "category": oc,
                "value": row_data.get("value", ""),
            })
    return all_items


@mcp.tool()
def add_string_item(category: str, value: str) -> str:
    """string_item を追加する。同じ category + value が既にあれば無視。"""
    # 重複チェック
    existing = get_string_items(category)
    for item in existing:
        if item.get("value") == value:
            return f"既に存在: ({category}: {value})"

    prefix = _ID_PREFIX.get(category, "SI")
    new_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    if category in _USER_CATEGORIES:
        vdb_category = "user"
    elif category in _AGENT_CATEGORIES:
        vdb_category = "agent"
    else:
        vdb_category = "other"

    row_data = {"id": new_id, "category": category, "value": value}
    metadata = {
        "category": vdb_category,
        "type": "string_item",
        "always_in_context": True,
        "data_json": json.dumps(row_data, ensure_ascii=False),
        "original_category": category,
        "registered_at": _now_iso(),
    }
    _vdb_upsert([{"id": new_id, "document": value, "metadata": metadata}])
    return f"追加しました ({category}: {value})"


@mcp.tool()
def delete_string_item(id: str) -> str:
    """string_item を ID で削除する。"""
    _delete(f"/vdb/long_memory/{id}")
    return f"削除しました: {id}"


# =============================================================================
# records
# =============================================================================


def _record_document_text(rec_type: str, data: dict) -> str:
    if rec_type == "visited":
        return f"{data.get('name','')} ({data.get('date','')}): {data.get('note','')}"
    if rec_type == "promise":
        return f"{data.get('content','')} ({data.get('status','')}, {data.get('date','')})"
    if rec_type == "habit":
        return f"{data.get('what','')} when {data.get('when','')} confidence={data.get('confidence','')}"
    if rec_type == "rule":
        return f"{data.get('what','')} active={data.get('active','')}"
    if rec_type == "todo":
        return f"{data.get('title','')} (status={data.get('status','')}, due={data.get('due','')}, priority={data.get('priority','')})"
    if rec_type == "goal":
        return f"{data.get('what','')} owner={data.get('owner','')} status={data.get('status','')}"
    if rec_type == "context":
        return f"{data.get('what','')} expires={data.get('expires','')}"
    if rec_type == "person":
        return f"relationship={data.get('relationship','')} closeness={data.get('closeness','')} notes={data.get('notes','')}"
    return json.dumps(data, ensure_ascii=False)


@mcp.tool()
def get_records(type: str = "") -> list[dict]:
    """records を取得する。type で絞り込み可（visited/promise/habit/rule/todo/goal/context/person）。"""
    # 対象 category を type から決定（type 未指定なら全カテゴリ）
    target_categories: list[str]
    if type:
        cat, _ = _RECORD_TYPE_MAP.get(type, ("other", False))
        target_categories = [cat]
    else:
        target_categories = list(set(v[0] for v in _RECORD_TYPE_MAP.values()))

    result: list[dict] = []
    for vcat in target_categories:
        for it in _vdb_get_by_category(category=vcat):
            meta = it.get("metadata", {})
            if meta.get("type") != "record":
                continue
            rt = meta.get("record_type", "")
            if type and rt != type:
                continue
            try:
                row_data = json.loads(meta.get("data_json") or "{}")
                inner = json.loads(row_data.get("data") or "{}")
            except json.JSONDecodeError:
                continue
            result.append({
                "id": row_data.get("id", it.get("id", "")),
                "type": rt,
                **inner,
            })
    return result


@mcp.tool()
def add_or_update_record(id: str, type: str, data: dict) -> str:
    """record を追加または更新する。data は任意の dict。"""
    vdb_category, always_in_context = _RECORD_TYPE_MAP.get(type, ("other", False))
    row_data = {"id": id, "type": type, "data": json.dumps(data, ensure_ascii=False)}
    doc = _record_document_text(type, data)
    metadata = {
        "category": vdb_category,
        "type": "record",
        "always_in_context": always_in_context,
        "data_json": json.dumps(row_data, ensure_ascii=False),
        "record_type": type,
        "registered_at": _now_iso(),
    }
    _vdb_upsert([{"id": id, "document": doc, "metadata": metadata}])
    return f"保存しました: {id} ({type})"


@mcp.tool()
def delete_record(id: str) -> str:
    """record を削除する。"""
    _delete(f"/vdb/long_memory/{id}")
    return f"削除しました: {id}"


# =============================================================================
# ユーティリティ
# =============================================================================


@mcp.tool()
def get_full_memory_json() -> str:
    """全記憶を旧 LongMemoryJSON 形式の JSON 文字列で返す（圧縮フェーズで使用）。

    VDB から always_in_context=True/False 両方を取得し、旧 SQLite 構造に近い形式で組み立てる。
    """
    result: dict = {
        "meta": {},
        "episodic": {"events": []},
        "semantic": {"user": {}, "agent": {}, "promises": [], "visited": []},
        "procedural": {"habits": [], "rules": []},
        "working": {"todos": [], "goals": [], "context": ""},
    }

    # 全件取得（always_in_context 関係なく全カテゴリ）
    all_items: list[dict] = []
    for cat in ["meta", "events", "user", "agent", "promises", "visited",
                "procedural", "working"]:
        all_items.extend(_vdb_get_by_category(category=cat))

    for it in all_items:
        meta = it.get("metadata") or {}
        cat = meta.get("category")
        type_ = meta.get("type")
        try:
            orig = json.loads(meta.get("data_json") or "{}")
        except json.JSONDecodeError:
            continue

        if cat == "meta":
            result["meta"][orig.get("key", "")] = orig.get("value", "")
        elif cat == "events":
            try:
                inner = json.loads(orig.get("data") or "{}")
            except json.JSONDecodeError:
                inner = {}
            result["episodic"]["events"].append({
                "id": orig.get("id", ""),
                "start_dt": orig.get("start_dt", ""),
                "end_dt": orig.get("end_dt", ""),
                "place": orig.get("place", ""),
                "what": orig.get("what", ""),
                "importance": orig.get("importance", "中"),
                "points": inner.get("points", []),
                "feelings": inner.get("feelings", []),
                "who": inner.get("who", []),
            })
        elif type_ == "string_item":
            oc = meta.get("original_category", "other")
            bucket = result["semantic"]["user"] if cat == "user" else result["semantic"]["agent"]
            bucket.setdefault(oc, []).append(orig.get("value", ""))
        elif cat == "promises":
            try: d = json.loads(orig.get("data") or "{}")
            except json.JSONDecodeError: d = {}
            result["semantic"]["promises"].append(d)
        elif cat == "visited":
            try: d = json.loads(orig.get("data") or "{}")
            except json.JSONDecodeError: d = {}
            result["semantic"]["visited"].append(d)
        elif cat == "procedural":
            rt = meta.get("record_type", "")
            try: d = json.loads(orig.get("data") or "{}")
            except json.JSONDecodeError: d = {}
            if rt == "habit":
                result["procedural"]["habits"].append(d)
            elif rt == "rule":
                result["procedural"]["rules"].append(d)
        elif cat == "working":
            rt = meta.get("record_type", "")
            try: d = json.loads(orig.get("data") or "{}")
            except json.JSONDecodeError: d = {}
            if rt == "todo":
                result["working"]["todos"].append(d)
            elif rt == "goal":
                result["working"]["goals"].append(d)
            elif rt == "context":
                existing = result["working"]["context"]
                new = d.get("what", "")
                result["working"]["context"] = (existing + "\n" + new).strip() if existing else new

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_memory_size() -> dict:
    """件数集計を返す（圧縮判定用）。"""
    counts = {"events": 0, "string_items": 0, "records": 0}
    estimated_chars = 0
    for cat in ["meta", "events", "user", "agent", "promises", "visited",
                "procedural", "working"]:
        for it in _vdb_get_by_category(category=cat):
            meta = it.get("metadata") or {}
            t = meta.get("type")
            if t == "event":
                counts["events"] += 1
            elif t == "string_item":
                counts["string_items"] += 1
            elif t == "record":
                counts["records"] += 1
            estimated_chars += len(it.get("document", "") or "")
    counts["estimated_chars"] = estimated_chars
    return counts


def main() -> None:
    """単体起動の入口。配布物では `.pyd` 化されるため、起動処理は関数に置く。"""
    mcp.run()


if __name__ == "__main__":
    main()
