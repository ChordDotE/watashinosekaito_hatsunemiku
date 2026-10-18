# -*- coding: utf-8 -*-
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

"""どの曲・どの版を歌うかを選ぶ軽量サブエージェント（2026-08-16 追加）。

## なぜ LLM が要るか

以前は文字列マッチだけで選んでいた（曲名の部分一致・頭4文字の一致）。これでは
**指示語と版指定がまったく解釈できない**:

  「さっきのバージョンと聞き比べたい」「前のやつ」「新しいほうで」「バンドアレンジの」

さらに「歌える曲が1つなら名前を照合せずに歌う」という近道があり、**名指しされた曲が
書き出し待ちで除外されると、無関係な曲が黙って鳴った**（2026-08-16 の実害。ミクは
『なついろのそら』を歌うと宣言したのに『シャイニングスター』が流れた）。

## なぜ軽量モデルか

判断は「候補の中から1つ選ぶ」だけで、文章生成を伴わない。フィラーと同じ
`lite_model`（flash-lite）で足りる。しかも **ミクが「歌うね♪」と喋っている間に走る**
ので、体感のレイテンシに出にくい。

## 失敗しても止めない

LLM が落ちたら呼び出し側が文字列マッチにフォールバックする（歌えなくなるより、
多少雑でも歌えるほうがよい）。
"""
from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agent.config import get_pydantic_ai_lite_model, load_settings
from agent.agent_logger import record_llm_usage

_settings = load_settings()

_SYSTEM_PROMPT = """あなたは「どの曲のどの版を再生するか」を決める係です。

与えられるもの:
- マスターの言葉（曲名がはっきりしないことも、指示語のこともある）
- 歌える曲の一覧。各曲には次が付く:
    latest_version   … いまの版番号
    playable_versions… **完成音源が実在する版**（ここに無い版は鳴らせない）
    stale            … true なら楽譜を直した後で書き出し直しが済んでいない
    version_forms    … 鳴らせる版ごとの曲の形、latest_form … いまの版の形（歌もの／アカペラ／インスト）。
                       **インストは歌の入らない演奏だけの曲で、歌わずに流す**
    version_notes    … 版ごとの、作った日時と何を直したか
- いまの日時

決め方:
1. 曲名・話の流れから、どの曲かを決める。**手がかりが無く候補が複数あるなら決めない**
2. 版の指定（「さっきの」「前の」「新しいほう」「朝作った」「〜を直す前の」）があれば version に入れる。
   日時や直した内容で言われたら、version_notes といまの日時から決める。
   指定が無ければ version は null（＝最新の固定名で鳴らす）
3. 次のときは song_id を空にして reason に理由を日本語で書く:
   - どの曲か絞れない
   - 名指しされた曲が stale（書き出し直しが要る）
   - 指定された版が playable_versions に無い（その版の音源がまだ無い）

**マスターが名前を出しているのに、それと違う曲を選んではいけません。**
一覧に1曲しかなくても、その曲を指していないなら選ばないこと。
"""


class SongSelectOutput(BaseModel):
    song_id: str = Field(default="", description="選んだ曲。決められないときは空")
    version: int | None = Field(default=None, description="版指定。無指定は null")
    reason: str = Field(default="", description="選べなかった理由（マスターへの説明に使う）")


def _build_agent() -> Agent[None, SongSelectOutput]:
    """呼び出しごとに生成する（フィラーと同じ理由＝使い捨てイベントループへの束縛を避ける）。"""
    return Agent(
        model=get_pydantic_ai_lite_model(_settings),
        output_type=SongSelectOutput,
        system_prompt=_SYSTEM_PROMPT,
    )


async def run_song_select(hint: str, candidates: list[dict]) -> SongSelectOutput:
    """曲と版を選ぶ。失敗時は song_id 空・reason 付きで返す（呼び出し側がフォールバック）。"""
    if not candidates:
        return SongSelectOutput(reason="歌える曲が無い")
    prompt = (
        f"【いまの日時】{datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"【マスターの言葉】\n{hint or '（手がかりなし）'}\n\n"
        f"【歌える曲】\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )
    try:
        _agent = _build_agent()
        result = await _agent.run(prompt)
        record_llm_usage(_agent, result, kind="song_select")
        return result.output
    except Exception as e:                                  # ネットワーク・スキーマ不整合など
        print(f"[song_select] 選曲に失敗（文字列マッチに委ねる）: {type(e).__name__}: {e}")
        return SongSelectOutput(reason=f"選曲の判断に失敗: {e}")
