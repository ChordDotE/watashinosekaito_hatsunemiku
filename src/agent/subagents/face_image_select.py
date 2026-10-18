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

"""顔認識用画像選択 Subagent。

複数枚の添付画像を実際に見て、顔認識に最も適した1枚を選ぶ。
インカメラバッファ（最大3枚）や写真添付が複数ある場合に使用する。

権限レベル: 1 (retrieval / 副作用なし)
副作用: なし
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModelSettings

from agent.agent_logger import record_llm_usage, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import configure_env_from_settings, get_pydantic_ai_flash_model, load_settings
from agent.io_helpers import to_binary_contents

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_settings = load_settings()
configure_env_from_settings(_settings)
_SYSTEM_PROMPT = (_PROMPTS_DIR / "face_image_select" / "system.md").read_text(encoding="utf-8")


class FaceImageSelectInput(BaseModel):
    image_file_names: list[str] = Field(description="画像ファイル名リスト（attached_images と並列）")
    attached_images: list[str] = Field(description="画像のローカルパスリスト")


class FaceImageSelectOutput(BaseModel):
    selected_filename: Optional[str] = Field(
        description="選択した画像のファイル名。顔認識に適した画像がない場合は null"
    )
    reason: str = Field(
        description="選択理由。selected_filename が null の場合は適切な画像がない理由"
    )


_face_image_select_agent: Agent[None, FaceImageSelectOutput] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=FaceImageSelectOutput,
    # 3.8 Flash は temperature を受け取るが無視する（2026-09-05 実測・0.0 でも 8 回中 7 種類）。
    # 同じ写真の組には同じ答えを返したいので seed も付ける（ベストエフォートで効く・3/8）。2026-10-10 に 3.8 へ切り替え
    model_settings=GoogleModelSettings(temperature=0.0, seed=0),
    system_prompt=_SYSTEM_PROMPT,
)


async def run_face_image_select(input_: FaceImageSelectInput) -> FaceImageSelectOutput:
    """画像一覧を LLM に見せ、顔認識に最適な1枚を選ばせる。

    失敗時は selected_filename=None、reason にエラー内容を返す。
    """
    file_list_text = "\n".join(
        f"- {name}" for name in input_.image_file_names
    )
    user_prompt_text = (
        f"以下の画像ファイルがあります：\n{file_list_text}\n\n"
        "顔認識処理に最も適した1枚を選んでください。"
    )

    try:
        if input_.attached_images:
            user_prompt: list = [user_prompt_text, *to_binary_contents(input_.attached_images)]
        else:
            user_prompt = user_prompt_text

        log_subagent_request(
            "face_image_select",
            mark_prompt("face_image_select.system", _SYSTEM_PROMPT),
            user_prompt_text,
            attached_images=input_.attached_images,
        )
        result = await _face_image_select_agent.run(user_prompt)
        record_llm_usage(_face_image_select_agent, result, kind="face_image_select")
        log_subagent_response("face_image_select", result.output.model_dump_json())
        return result.output

    except Exception as exc:
        log_subagent_response("face_image_select", None, error=repr(exc))
        return FaceImageSelectOutput(
            selected_filename=None,
            reason=f"画像選択処理でエラーが発生しました: {exc!r}",
        )
