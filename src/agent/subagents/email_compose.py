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

"""メール内容考案 Subagent。

Dify の以下のノード群を再現する:
  - メール内容考案 LLM（件名・本文を生成）
  - メール送信（SMTP 送信）

戻り値は生成したメール内容の JSON 文字列
（Dify「メール内容考案」structured_output 互換: {"subject": "...", "body": "..."}）。

権限レベル: 3 (external)
副作用: 実際にメールを送信する
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agent.agent_logger import record_llm_usage, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import configure_env_from_settings, get_pydantic_ai_flash_model, load_settings
from agent.io_helpers import derive_file_names, to_binary_contents
from agent.schemas import Failure
from agent.tools.external.send_email import SendEmailInput, run_send_email

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_settings = load_settings()
configure_env_from_settings(_settings)
_SYSTEM_PROMPT = (_PROMPTS_DIR / "email_compose" / "system.md").read_text(encoding="utf-8")


class EmailComposeInput(BaseModel):
    base_prompt: str = Field(default="", description="基本のプロンプト（Dify {{#1764431466242.output#}} 相当）")
    loop_counter: int = Field(default=1, description="現在のループ実行回数")
    max_loops: int = Field(default=5, description="最大許容ループ回数")
    context: str = Field(default="", description="会話コンテキスト（参考情報）")
    current_time: str = Field(default="", description="現在時刻")
    user_input: str = Field(default="", description="ユーザーの新たな入力（{{#conversation.input#}} 相当）")
    file_names: str = Field(default="", description="今回添付されたファイルの名称（camera_* 含む）")
    attached_images: list[str] = Field(
        default_factory=list,
        description="添付画像のローカルパスリスト（Dify vision: enabled 相当）",
    )


class EmailComposeOutput(BaseModel):
    subject: str = Field(description="メールの件名")
    body: str = Field(description="メールの本文（初音ミクらしい口調・敬語不要・マークダウン禁止・改行は \\n で表現）")


_email_compose_agent: Agent[None, EmailComposeOutput] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=EmailComposeOutput,
    system_prompt=_SYSTEM_PROMPT,
)


def _build_user_prompt(input_: EmailComposeInput, file_names: str) -> str:
    """Dify「メール内容考案」LLM の system+user prompt 構造を user_prompt に統合して再現する。

    各セクションは値の有無に関わらず常に描画される（Dify テンプレ文字列の挙動）。
    """
    base_prompt_block = (
        "------\n"
        "基本のプロンプト\n"
        f"{input_.base_prompt}"
    )

    loop_block = (
        "------\n"
        "ループに関する指示\n\n"
        "# 現在の処理状況（システムステータス）\n"
        "現在、ツールとして send_email が選ばれたので、マスターにメールを送ります。\n"
        "あなたは初音ミクの内部メール内容考案専用アシスタントです。"
        "このノードの目的は、ミクからマスターに送るメールの件名・本文を作成することです。"
        "ユーザーに見せる返答文は作成しません。\n\n"
        f"- 現在のループ実行回数: {input_.loop_counter} 回目の中で send_email（本LLM）ツールに分岐中\n"
        f"- 最大許容ループ回数: {input_.max_loops} 回"
    )

    context_block = (
        "# 参考情報（返答内容の参考にすること）\n"
        "{\n"
        f"{input_.context}\n"
        "}"
    )

    user_input_block = (
        f"# 現在の時刻\n{input_.current_time}\n\n"
        f"# ユーザー側の新たな入力・発言（会話においてはこれに返答すること）\n{input_.user_input}\n\n"
        f"# 今回添付されたファイルの名称\n{file_names}\n"
        "※camera_0～camera_2の画像は、ミクの「目」から見た画像で、"
        "その他のファイルはユーザーが意図的に添付した画像・ファイルなので注意すること"
    )

    return f"{base_prompt_block}\n\n{loop_block}\n\n{context_block}\n\n{user_input_block}"


async def run_email_compose(input_: EmailComposeInput) -> str:
    """メール内容を生成・送信し、生成したメール内容の JSON 文字列を返す。

    失敗時は "[エラー] ..." 文字列を返す（例外を raise しない）。
    """
    try:
        file_names = input_.file_names or derive_file_names(input_.attached_images)
        user_prompt_text = _build_user_prompt(input_, file_names=file_names)

        if input_.attached_images:
            user_prompt: list = [user_prompt_text, *to_binary_contents(input_.attached_images)]
        else:
            user_prompt = user_prompt_text

        log_subagent_request(
            "email_compose",
            mark_prompt("email_compose.system", _SYSTEM_PROMPT),
            user_prompt_text,
            attached_images=input_.attached_images,
        )
        result = await _email_compose_agent.run(user_prompt)
        record_llm_usage(_email_compose_agent, result, kind="email_compose")
        subject = result.output.subject
        body = result.output.body.replace('\\n', '\n')
        log_subagent_response("email_compose", {"subject": subject, "body": body})

        send_result = run_send_email(SendEmailInput(subject=subject, body=body))
        if isinstance(send_result, Failure):
            return f"[エラー] メール送信失敗: {send_result.reason}"

        return json.dumps({"subject": subject, "body": body}, ensure_ascii=False)

    except Exception as exc:
        log_subagent_response("email_compose", None, error=repr(exc))
        return f"[エラー] email_compose 失敗: {exc!r}"
