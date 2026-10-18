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

"""Web 検索 Subagent。

Dify の「web検索エージェント」ノードを再現する。
`search_web` ツールを使って最大 3 回検索し、
### 結論 + #### 検索結果N 形式のマークダウンを返す。

権限レベル: 1 (retrieval)
副作用: なし（受信のみ）
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agent.agent_logger import record_llm_usage, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import configure_env_from_settings, get_pydantic_ai_flash_model, load_settings
from agent.schemas import Failure
from agent.tools.retrieval.web_search import WebSearchInput, run_web_search
from agent.tools.utility.get_current_time import GetCurrentTimeInput, run_get_current_time

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_settings = load_settings()
configure_env_from_settings(_settings)
_SYSTEM_PROMPT = (_PROMPTS_DIR / "web_search_agent" / "system.md").read_text(encoding="utf-8")


class WebSearchAgentInput(BaseModel):
    conversation_history: str = Field(default="", description="ここまでの会話内容（Dify body 相当）")
    user_input: str = Field(default="", description="ユーザーの新たな入力")
    ai_judgements: str = Field(default="", description="これまでの AI の判断（inner_thoughts/purpose/motion 等の structured_output）")
    tool_results: str = Field(default="", description="これまでに得られたツール結果（conversation.tool_result）")
    max_results: int = Field(default=5, ge=1, le=10, description="1回の検索で取得する件数")


class WebSearchAgentOutput(BaseModel):
    result: str = Field(description="検索結果のマークダウン（### 結論 + #### 検索結果N 形式）")


_web_search_agent: Agent[None, WebSearchAgentOutput] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=WebSearchAgentOutput,
    system_prompt=_SYSTEM_PROMPT,
)


@_web_search_agent.tool_plain
def search_web(query: str, max_results: int = 5) -> str:
    """Tavily API で Web 検索を実行する。"""
    result = run_web_search(WebSearchInput(query=query, max_results=max_results))
    if isinstance(result, Failure):
        return f"[検索エラー] {result.reason}"
    return result.value.formatted or "（結果なし）"


@_web_search_agent.tool_plain
def get_current_time(timezone: str = "Asia/Tokyo") -> str:
    """現在時刻を取得する。検索結果が時間・地域に依存する場合に先に呼び出すこと。"""
    result = run_get_current_time(GetCurrentTimeInput(timezone=timezone))
    if isinstance(result, Failure):
        return f"[時刻取得エラー] {result.reason}"
    return result.value.text


def _build_user_prompt(input_: WebSearchAgentInput) -> str:
    """Dify「web検索エージェント」の query テンプレートを再現する。

    Dify はテンプレート文字列なので、各セクションは値の有無に関わらず常に描画される
    （値が空なら空白行になるだけ）。
    """
    return (
        f"## ここまでの会話内容\n{input_.conversation_history}\n\n"
        f"## ユーザーの新たな入力\n{input_.user_input}\n\n"
        f"## これまでの会話のあなた（=AI）が実行したのツール等の判断\n{input_.ai_judgements}\n\n"
        f"## これまでに今回の返答のためにあなた（=AI）が実行して得られたツール結果\n{input_.tool_results}"
    )


# 調べた結果を「自分の興味」として受け取ってよい、と添える一文。
# これが無いと、拾う話題がすべてマスター向けになる（check_news で実測済み）。
# 実ログでも、一文がある check_news 経由では notice が 3 件出たのに対し、
# web_search だけで完結したターンは 6 回とも 0 件だった（2026-08-29 集計）。
# 要請ではなく許可のままにするのは、強めると捏造が出るため。
_INTEREST_NOTE = (
    "\n\n※ この結果の中に、マスターの役に立つかどうかとは別に、"
    "自分が面白いと思ったことがあれば、それは自分の興味として残してよい。"
)


async def run_web_search_agent(input_: WebSearchAgentInput) -> str:
    """Web 検索 subagent を実行し、マークダウン形式の結果を返す。

    失敗時は "[エラー] ..." 文字列を返す（例外を raise しない）。
    """
    try:
        user_prompt = _build_user_prompt(input_)
        log_subagent_request("web_search_agent", mark_prompt("web_search_agent.system", _SYSTEM_PROMPT), user_prompt)
        result = await _web_search_agent.run(user_prompt)
        record_llm_usage(_web_search_agent, result, kind="web_search")
        log_subagent_response("web_search_agent", result.output)
        return result.output.result + _INTEREST_NOTE
    except Exception as exc:
        log_subagent_response("web_search_agent", None, error=repr(exc))
        return f"[エラー] web_search_agent 失敗: {exc!r}"
