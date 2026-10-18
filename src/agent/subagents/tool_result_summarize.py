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

"""ツール結果の要点まとめ Subagent。

ツール結果は本来ターンをまたぐと失われる。会話ログに残せる長さへ縮めるための
軽量な要約 LLM（設計: docs/plan/tool_result_carryover.md）。

対象は**生の結果が長い想起系のみ**（memory_search / recall_long_memory）。
web_search は subagent が既に「### 結論」を書いており、短いツールはそのまま残せるため、
ここを通す必要がない。

**会話文脈は渡さない**。「どんな言葉で調べて、何が分かったか」だけを見て要約する
（文脈を渡すと「これからどう返事するか」に引きずられ、結果の記録ではなくなる）。

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
_SYSTEM_PROMPT = (_PROMPTS_DIR / "tool_result_summarize" / "system.md").read_text(encoding="utf-8")

# 生結果が極端に長いと要約自体が重くなるので頭を切る。
# 検索結果はスコア降順で並ぶため、先頭を残せば要点は拾える。
_MAX_INPUT_CHARS = 12_000


class ToolResultSummarizeInput(BaseModel):
    tool_name: str = Field(description="要約対象のツール名（出所ラベルとして要約文に含めさせる）")
    result: str = Field(description="ツールが返した生の結果")


_tool_result_summarize_agent: Agent[None, str] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=str,
    system_prompt=_SYSTEM_PROMPT,
)


def _build_user_prompt(input_: ToolResultSummarizeInput) -> str:
    body = input_.result
    truncated = ""
    if len(body) > _MAX_INPUT_CHARS:
        body = body[:_MAX_INPUT_CHARS]
        truncated = "\n（※ 結果が長いため先頭のみ。以降は省略）"
    return (
        f"## 使ったツール\n{input_.tool_name}\n\n"
        f"## ツールが返した結果\n{body}{truncated}"
    )


async def run_tool_result_summarize(input_: ToolResultSummarizeInput) -> str:
    """ツール結果を1〜3文の要点へ縮める。

    失敗時は空文字を返す（例外を raise しない）。**要約に失敗しても会話は止めない**——
    要点が残らないだけで、そのターンの応答自体は生の結果を見て作られる。
    """
    try:
        user_prompt = _build_user_prompt(input_)
        log_subagent_request(
            "tool_result_summarize",
            mark_prompt("tool_result_summarize.system", _SYSTEM_PROMPT),
            user_prompt,
        )
        result = await _tool_result_summarize_agent.run(user_prompt)
        record_llm_usage(_tool_result_summarize_agent, result, kind="tool_result_summarize")
        summary = (result.output or "").strip()
        log_subagent_response("tool_result_summarize", summary)
        return summary
    except Exception as exc:  # noqa: BLE001
        log_subagent_response("tool_result_summarize", None, error=repr(exc))
        print(f"[WARN] tool_result_summarize 失敗: {exc!r}")
        return ""
