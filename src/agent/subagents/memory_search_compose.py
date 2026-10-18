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

"""記憶検索クエリ考案 Subagent。

Dify の「記憶検索ワード考案」LLM ノードを再現する。
会話履歴・新たなユーザー入力・AI の判断・既存ツール結果から、
RAG 向けの検索クエリ（"query: ..." 形式）を生成する。

戻り値は `query: ...` の prefix を取り除いた純粋な検索クエリ文字列。

権限レベル: 0 (utility / LLM のみ・副作用なし)
副作用: なし
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agent.agent_logger import record_llm_usage, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import configure_env_from_settings, get_pydantic_ai_flash_model, load_settings

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_settings = load_settings()
configure_env_from_settings(_settings)
_SYSTEM_PROMPT = (_PROMPTS_DIR / "memory_search_compose" / "system.md").read_text(encoding="utf-8")


class MemorySearchComposeInput(BaseModel):
    conversation_history: str = Field(default="", description="ここまでの会話内容（Dify body 相当）")
    user_input: str = Field(default="", description="ユーザーの新たな入力")
    ai_judgements: str = Field(default="", description="これまでの AI の判断（structured_output）")
    tool_results: str = Field(default="", description="これまでに得られたツール結果（conversation.tool_result）")


_memory_search_compose_agent: Agent[None, str] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=str,
    system_prompt=_SYSTEM_PROMPT,
)


def _build_user_prompt(input_: MemorySearchComposeInput) -> str:
    """Dify「記憶検索ワード考案」LLM の user テンプレートを再現する。

    各セクションは値の有無に関わらず常に描画される（Dify テンプレ文字列の挙動）。
    """
    return (
        f"## ここまでの会話内容\n{input_.conversation_history}\n\n"
        f"## ユーザーの新たな入力\n{input_.user_input}\n\n"
        f"## これまでの会話のあなた（=AI）が実行したのツール等の判断\n{input_.ai_judgements}\n\n"
        f"## これまでに今回の返答のためにあなた（=AI）が実行して得られたツール結果\n{input_.tool_results}"
    )


def _strip_query_prefix(text: str) -> str:
    """`query: ...` の prefix を取り除いて検索ワードのみ返す。"""
    line = text.strip().splitlines()[0] if text.strip() else ""
    if line.lower().startswith("query:"):
        return line[len("query:"):].strip()
    return line.strip()


async def run_memory_search_compose(input_: MemorySearchComposeInput) -> str:
    """LLM で memory_search 用の検索クエリを生成する。

    失敗時は "[エラー] ..." 文字列を返す（例外を raise しない）。
    """
    try:
        user_prompt = _build_user_prompt(input_)
        log_subagent_request("memory_search_compose", mark_prompt("memory_search_compose.system", _SYSTEM_PROMPT), user_prompt)
        result = await _memory_search_compose_agent.run(user_prompt)
        record_llm_usage(_memory_search_compose_agent, result, kind="memory_search")
        log_subagent_response("memory_search_compose", result.output)
        return _strip_query_prefix(result.output)
    except Exception as exc:
        log_subagent_response("memory_search_compose", None, error=repr(exc))
        return f"[エラー] memory_search_compose 失敗: {exc!r}"
