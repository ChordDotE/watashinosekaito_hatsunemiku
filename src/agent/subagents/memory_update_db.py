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

"""会話文記憶更新 Subagent — VDB 版。

長期記憶は VDB collection `long_memory` で管理し、`long_memory_vdb_mcp_server.py` の
MCP ツール群経由で更新する。

フロー:
  - 登録待機会話取得（KV: registration_waiting_conversation）
  - always_in_context セクションのサイズ確認（圧縮判定）
  - MCPServerStdio で long_memory_vdb_mcp_server.py を起動
  - LLM が get_full_memory_json() → 差分操作（add_event / add_string_item / add_or_update_record / delete_*）
  - サイズが閾値超えの場合はプロンプトで通知し、同一セッション内で圧縮も実施
  - 更新前後のサイズ比率をログ出力（バックアップは /vdb/long_memory/full_json 経由）

テスト時のモック境界:
  _execute_mcp_update(user_prompt)            — MCP エージェント起動（AsyncMock 可）
  _get_always_in_context_size()               — サイズ確認（return_value mock 可）
  _get_full_memory_json_str()                 — バックアップ用 JSON 取得（return_value mock 可）
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from agent.agent_logger import record_llm_usage, log, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import (
    configure_env_from_settings,
    get_kv_base_url,
    get_memory_compress_threshold,
    get_pydantic_ai_flash_model,
    get_vdb_base_url,
    load_settings,
)
from agent.schemas import MemoryUpdateInput, MemoryUpdateResult
from utils import conversation_log

_logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_MCP_SERVER_SCRIPT = Path(__file__).resolve().parents[2] / "long_memory_vdb_mcp_server.py"

_settings = load_settings()
configure_env_from_settings(_settings)
_KV_BASE_URL = get_kv_base_url(_settings)
_VDB_BASE_URL = get_vdb_base_url(_settings)

# always_in_context セクションの圧縮発動閾値（settings.json: memory_update.compress_threshold）
_COMPRESS_THRESHOLD = get_memory_compress_threshold(_settings)

_BASE_PROMPT = (_PROMPTS_DIR / "base_prompt.md").read_text(encoding="utf-8")
_UPDATE_SYSTEM = (_PROMPTS_DIR / "memory_update_system_db.md").read_text(encoding="utf-8")


# ── KV ヘルパー ────────────────────────────────────────────────────────────

def _kv_get(key: str) -> str:
    r = requests.get(f"{_KV_BASE_URL}/kv", params={"key": key}, timeout=10)
    if r.status_code == 404:
        return ""
    r.raise_for_status()
    return r.text


def _kv_set(key: str, value: str) -> None:
    r = requests.post(f"{_KV_BASE_URL}/kv", json={"key": key, "value": value}, timeout=10)
    r.raise_for_status()


# ── VDB ヘルパー ──────────────────────────────────────────────────────────

def _get_always_in_context_size() -> int:
    """always_in_context=True ドキュメントの合計文字数を返す（圧縮判定用）。"""
    r = requests.get(
        f"{_VDB_BASE_URL}/vdb/long_memory/by_category",
        params={"always_in_context": "true"},
        timeout=15,
    )
    r.raise_for_status()
    items = r.json().get("items", [])
    return sum(len(it.get("document", "") or "") for it in items)


def _get_full_memory_json_str() -> str:
    """更新後の全記憶を JSON 文字列で返す（バックアップ用）。"""
    r = requests.get(f"{_VDB_BASE_URL}/vdb/long_memory/full_json", timeout=30)
    r.raise_for_status()
    return r.text


# ── プロンプトビルダー ────────────────────────────────────────────────────

_JST = timezone(timedelta(hours=9))


def _build_update_user_prompt(new_conversation: str, memory_size: int) -> str:
    """更新エージェントへのユーザープロンプトを組み立てる。

    memory_size が閾値を超えている場合は圧縮指示を末尾に追加する。
    """
    current_time = datetime.now(_JST).strftime("%Y-%m-%d %H:%M JST")

    compress_hint = ""
    if memory_size >= _COMPRESS_THRESHOLD:
        compress_hint = (
            f"\n\n## 圧縮指示\n"
            f"現在の always_in_context セクションサイズ: {memory_size}字"
            f"（上限 {_COMPRESS_THRESHOLD}字 超過）\n"
            "記憶の追加・更新作業が完了したら、不要な記憶を削除・統合して"
            "15,000字程度まで圧縮してください。"
        )

    return (
        f"## 現在日時\n{current_time}\n\n"
        "## 更新せず固定で持っているプロンプト(1. ベースプロンプト)\n"
        "----------------\n"
        f"{{\n{_BASE_PROMPT}\n}}\n"
        "----------------\n\n"
        f"{conversation_log.LOG_READING_GUIDE}\n"
        "## 新たに記憶に追加してほしい会話・行動内容\n"
        "----------------\n"
        f"{{\n{new_conversation}\n}}\n"
        "----------------"
        + compress_hint
    )


# ── MCP エージェント実行（テスト時は patch.object でモック可） ────────────

async def _execute_mcp_update(user_prompt: str) -> str:
    """MCPServerStdio で long_memory_vdb_mcp_server を起動し VDB 差分更新する。

    テストでは ``patch.object(module, '_execute_mcp_update', AsyncMock(...))``
    でモック可能。
    """
    mcp_server = MCPServerStdio(
        command=sys.executable,
        args=[str(_MCP_SERVER_SCRIPT)],
        env={**os.environ, "VDB_BASE_URL": _VDB_BASE_URL},
        timeout=30,
    )
    agent: Agent[None, str] = Agent(
        model=get_pydantic_ai_flash_model(_settings),
        output_type=str,
        system_prompt=_UPDATE_SYSTEM,
        toolsets=[mcp_server],
    )
    log_subagent_request(
        "memory_update",
        mark_prompt("memory_update_system", _UPDATE_SYSTEM),
        user_prompt,
    )
    async with agent:
        result = await agent.run(user_prompt)
    record_llm_usage(agent, result, kind="memory_update")
    log_subagent_response("memory_update", result.output)
    return result.output


# ── 公開エントリポイント ──────────────────────────────────────────────────

async def run_memory_update(input_: MemoryUpdateInput) -> MemoryUpdateResult | None:
    """Orchestrator / scheduler / long_memory_patch から呼ぶ公開関数。

    戻り値が None または skip_reason != None の場合はスキップ/失敗。
    """
    is_transfer = input_.is_transfer

    if is_transfer:
        new_conversation = input_.conversation_text
    else:
        new_conversation = _kv_get("registration_waiting_conversation")
        if not new_conversation or not new_conversation.strip():
            _logger.info("エラー: 登録内容が空白によりスキップ")
            return MemoryUpdateResult(skip_reason="エラー: 登録内容が空白によりスキップ")

    log(
        "memory_update_inputs",
        "memory_update WF の入力素材",
        input={"new_conversation": new_conversation},
        meta={"is_transfer": is_transfer},
    )

    # 更新前サイズ取得（失敗しても圧縮なしで続行）
    size_before = 0
    try:
        size_before = _get_always_in_context_size()
    except Exception as e:
        _logger.warning("更新前サイズ取得失敗: %s", e)
    print(f"[memory_update] 更新前サイズ: {size_before} chars")

    # MCP エージェント実行（更新 + 必要なら圧縮）
    update_user_prompt = _build_update_user_prompt(new_conversation, size_before)
    update_summary = await _execute_mcp_update(update_user_prompt)
    print(f"[memory_update] MCP更新完了: {update_summary[:100]}")

    if not is_transfer:
        _kv_set("registration_waiting_conversation", "")

    # 更新後サイズ取得・圧縮率ログ
    compression_rate = 1.0
    try:
        size_after = _get_always_in_context_size()
        if size_before > 0:
            compression_rate = size_after / size_before
        print(
            f"[memory_update] 更新後サイズ: {size_after} chars  "
            f"圧縮率: {compression_rate:.3f}"
            f" ({size_before} → {size_after})"
        )
    except Exception as e:
        _logger.warning("更新後サイズ取得失敗: %s", e)

    # バックアップ用 JSON 取得（失敗しても結果は返す）
    updated_json = ""
    try:
        updated_json = _get_full_memory_json_str()
    except Exception as e:
        _logger.warning("バックアップ用 JSON 取得失敗: %s", e)

    return MemoryUpdateResult(compression_rate=compression_rate, updated_memory_json=updated_json)
