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

"""ミクメモ更新 Subagent。

現在のメモを読み、`note_replace` / `note_append` で**変えたい箇所だけ**を書き換える。
更新後の全文は作らない（2026-08-29 に全文上書きから移行。経緯は
`agent/tools/memory/miku_note_update.py` の docstring）。

編集ツールは失敗を文字列で返す。LLM はそれを読んでやり直せるので、
一致しない書き換えが黙って通ることがない。

権限レベル: 2 (memory)
副作用: KV サービスの miku_note を更新する
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agent.agent_logger import record_llm_usage, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import configure_env_from_settings, get_kv_base_url, get_pydantic_ai_flash_model, load_settings
from agent.io_helpers import derive_file_names, to_binary_contents
from agent.schemas import Failure
from agent.tools.memory.miku_note_update import (
    MikuNoteAppendInput,
    MikuNoteReplaceInput,
    run_miku_note_append,
    run_miku_note_replace,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_settings = load_settings()
configure_env_from_settings(_settings)
_KV_BASE_URL = get_kv_base_url(_settings)
_SYSTEM_PROMPT = (_PROMPTS_DIR / "miku_note_compose" / "system.md").read_text(encoding="utf-8")

# 1 回の更新で許す編集回数の上限。実ログ 145 回ぶんの実測では、書き写しによる
# 「揺れ」を除いた本当の編集は中央値 2 回・最大 7 回だった（2026-08-29）。
# 失敗してやり直すぶんに 3 倍弱の余裕を見てこの値にしている。暴走の歯止め。
_MAX_EDITS = 20


class MikuNoteComposeInput(BaseModel):
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


class MikuNoteOutput(BaseModel):
    what_changed: str = Field(
        default="",
        description=(
            "今回このメモをどう更新したかを1〜2文で。何を書き足したか・何を書き直したかを、"
            "あとで自分が読んで思い出せるように書く（例:「ボカコレ2026夏のStem配布リストを"
            "Page 4 に追加した」）。書き換える必要が無かったならそう書く。"
        ),
    )


_miku_note_compose_agent: Agent[None, MikuNoteOutput] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=MikuNoteOutput,
    system_prompt=_SYSTEM_PROMPT,
)


class _EditBudget:
    """1 回の compose 実行で使える編集回数を数える。

    Agent のツールは引数しか受け取れないので、実行ごとに差し替えるモジュール変数で持つ。
    compose は runner から直列に呼ばれるため、同時に 2 つ走ることはない。
    """

    def __init__(self) -> None:
        self.used = 0

    def spend(self) -> bool:
        if self.used >= _MAX_EDITS:
            return False
        self.used += 1
        return True


_budget = _EditBudget()


def _apply(result) -> str:
    """ツール結果を LLM に見せる 1 行にする。"""
    if isinstance(result, Failure):
        return f"[失敗] {result.reason}"
    return f"[成功] {result.summary}"


@_miku_note_compose_agent.tool_plain
def note_replace(old: str, new: str = "") -> str:
    """メモの中の一箇所を書き換える。new を空にするとその箇所を削除する。

    old はメモに一字一句そのままある文字列であること。見つからない場合・
    2 箇所以上ある場合は失敗が返るので、メモを読み直してやり直すこと。
    """
    if not _budget.spend():
        return f"[失敗] 1 回の更新で編集できるのは {_MAX_EDITS} 回までです。ここで終了してください"
    return _apply(run_miku_note_replace(MikuNoteReplaceInput(old=old, new=new)))


@_miku_note_compose_agent.tool_plain
def note_append(text: str) -> str:
    """メモの末尾に書き足す。既存のどこにも属さない新しい話題を書くときに使う。"""
    if not _budget.spend():
        return f"[失敗] 1 回の更新で編集できるのは {_MAX_EDITS} 回までです。ここで終了してください"
    return _apply(run_miku_note_append(MikuNoteAppendInput(text=text)))


def _kv_get_note() -> str:
    try:
        resp = requests.get(f"{_KV_BASE_URL}/kv", params={"key": "miku_note"}, timeout=10)
        return resp.text if resp.status_code != 404 else ""
    except Exception:
        return ""


def _build_user_prompt(input_: MikuNoteComposeInput, current_note: str, file_names: str) -> str:
    """Dify「ミクメモ更新」LLM の system+user prompt 構造を user_prompt に統合して再現する。

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
        "現在、ツールとして miku_note_update が選ばれたので、メモの更新を行う必要があります。\n"
        "あなたは初音ミクの内部メモ編集専用アシスタントです。"
        "このノードの目的は、ミクの内部メモ帳（miku_note）を必要なだけ書き換えることです。"
        "ユーザーに見せる返答文は作成しません。\n\n"
        f"- 現在のループ実行回数: {input_.loop_counter} 回目の中で miku_note_update（本LLM）ツールに分岐中\n"
        f"- 最大許容ループ回数: {input_.max_loops} 回\n"
        f"- 1回の更新で編集できる回数: {_MAX_EDITS} 回まで"
        "（note_replace / note_append の呼び出し回数。失敗した呼び出しも数える）"
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

    current_note_block = (
        "# 現在のミクのメモの内容（この中の変えたい箇所だけを書き換える）\n"
        f"{current_note}"
    )

    return f"{base_prompt_block}\n\n{loop_block}\n\n{context_block}\n\n{user_input_block}\n\n{current_note_block}"


async def run_miku_note_compose(input_: MikuNoteComposeInput) -> str:
    """miku_note を編集し、何を変えたかの JSON 文字列を返す。

    失敗時は "[エラー] ..." 文字列を返す（例外を raise しない）。
    """
    global _budget
    try:
        current_note = _kv_get_note()
        file_names = input_.file_names or derive_file_names(input_.attached_images)
        user_prompt_text = _build_user_prompt(input_, current_note, file_names=file_names)

        if input_.attached_images:
            user_prompt: list = [user_prompt_text, *to_binary_contents(input_.attached_images)]
        else:
            user_prompt = user_prompt_text

        log_subagent_request(
            "miku_note_compose",
            mark_prompt("miku_note_compose.system", _SYSTEM_PROMPT),
            user_prompt_text,
            attached_images=input_.attached_images,
            current_note=current_note,
        )

        _budget = _EditBudget()
        result = await _miku_note_compose_agent.run(user_prompt)
        record_llm_usage(_miku_note_compose_agent, result, kind="miku_note")
        what_changed = (result.output.what_changed or "").strip()
        edits = _budget.used

        # 全文ではなく「何回どう編集したか」を残す。ノート全文は毎ターン context にある
        log_subagent_response("miku_note_compose", f"edits={edits} / {what_changed}")

        # what_changed は会話ログへ残す要点として runner が取り出す
        return json.dumps(
            {"what_changed": what_changed, "edits": edits}, ensure_ascii=False)

    except Exception as exc:
        log_subagent_response("miku_note_compose", None, error=repr(exc))
        return f"[エラー] miku_note_compose 失敗: {exc!r}"
