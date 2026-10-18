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

"""高度思考 Subagent。

Dify の「高度思考」LLM ノードを再現する。
pro モデル（gemini pro 系）に grounding / url_context / code_execution を有効化して、
段階的推論結果をマークダウン形式で返す。

Dify オリジナルの prompt 構造を踏襲:
  system: 基本プロンプト + ループ状況 + 行動指示
  user:   参考情報（context） + ユーザー新規入力

戻り値はマークダウン文字列（structured_output ではない）。

権限レベル: 1 (retrieval / 副作用なし)
副作用: なし（grounding 内部で Google 検索が走るが LLM 内部のみ）
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, WebFetchTool, WebSearchTool
from pydantic_ai.models.google import GoogleModelSettings

from agent.agent_logger import record_llm_usage, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import configure_env_from_settings, get_pydantic_ai_pro_model, load_settings
from agent.io_helpers import to_binary_contents

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_settings = load_settings()
configure_env_from_settings(_settings)
_SYSTEM_PROMPT = (_PROMPTS_DIR / "deep_think" / "system.md").read_text(encoding="utf-8")


class DeepThinkInput(BaseModel):
    base_prompt: str = Field(default="", description="基本のプロンプト（Dify {{#1764431466242.output#}} 相当）")
    loop_counter: int = Field(default=1, description="現在のループ実行回数（1 始まり）")
    max_loops: int = Field(default=5, description="最大許容ループ回数")
    context: str = Field(default="", description="参考情報（会話履歴・ナレッジ・ツール結果等）")
    user_input: str = Field(default="", description="ユーザー側の新たな入力・発言")
    attached_images: list[str] = Field(
        default_factory=list,
        description="添付画像のローカルパスリスト（Dify vision: enabled 相当）",
    )


_deep_think_agent: Agent[None, str] = Agent(
    model=get_pydantic_ai_pro_model(_settings),
    output_type=str,
    builtin_tools=[WebSearchTool(), WebFetchTool(), CodeExecutionTool()],
    model_settings=GoogleModelSettings(temperature=0.7),
    system_prompt=_SYSTEM_PROMPT,
)


def _build_user_prompt(input_: DeepThinkInput) -> str:
    """Dify「高度思考」の system+user prompt 構造を user_prompt に統合して再現する。

    各セクションは値の有無に関わらず常に描画される（Dify テンプレ文字列の挙動）。

    ★**人格の本文（base_prompt）は渡さない**（2026-10-10 本人判断）。deep_think が作るのはミクの言葉ではなく、
    ミクが使う考えの材料。人格とテンションを渡すと材料がミクの期待に寄る（10/9 に盛った結論が会話へ広がった）。
    ミクの好みや価値観は context（記憶・手帳）に入っているので、創作の手がかりはそこから取れる。
    `input_.base_prompt` は受け取るが使わない（raw_input は全ツール共通の形のため）。
    """
    loop_block = (
        "------\n"
        "ループに関する指示\n\n"
        "# 現在の処理状況（システムステータス）\n"
        "この処理は解決のためにループ実行されています。あなたは今、そのループの中にいます。\n"
        "現在、ツールとして deep_think が選ばれたので、高度な思考をする必要があります。\n"
        f"- 現在のループ実行回数: {input_.loop_counter} 回目の中で deep_think（本LLM）ツールに分岐中\n"
        f"- 最大許容ループ回数: {input_.max_loops} 回"
    )

    context_block = (
        "# 参考情報（返答内容の参考にすること）\n"
        "{\n"
        f"{input_.context}\n"
        "}"
    )

    user_input_block = (
        "# ユーザー側の新たな入力・発言など\n"
        "- こちらに返答することが最終的なゴール\n"
        "- 返答は別のLLMで行うため、今回は回答に至るための要素を構築する\n"
        f"{input_.user_input}"
    )

    return f"{loop_block}\n\n{context_block}\n\n{user_input_block}"


async def run_deep_think(input_: DeepThinkInput) -> str:
    """pro モデルで深い思考を実行し、マークダウン形式の結果を返す。

    添付画像がある場合は BinaryContent に変換して並列で渡す（Dify vision: enabled 相当）。

    失敗時は "[エラー] ..." 文字列を返す（例外を raise しない）。
    """
    try:
        user_prompt_text = _build_user_prompt(input_)
        if input_.attached_images:
            user_prompt: list = [user_prompt_text, *to_binary_contents(input_.attached_images)]
        else:
            user_prompt = user_prompt_text
        log_subagent_request(
            "deep_think",
            mark_prompt("deep_think.system", _SYSTEM_PROMPT),
            user_prompt_text,
            attached_images=input_.attached_images,
        )
        result = await _deep_think_agent.run(user_prompt)
        record_llm_usage(_deep_think_agent, result, kind="deep_think")
        log_subagent_response("deep_think", result.output)
        return result.output
    except Exception as exc:
        log_subagent_response("deep_think", None, error=repr(exc))
        return f"[エラー] deep_think 失敗: {exc!r}"
