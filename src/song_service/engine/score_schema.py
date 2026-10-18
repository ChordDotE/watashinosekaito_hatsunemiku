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

"""楽曲の中間表現スキーマ（PoC 版）。

docs/plan/song_generation_study.md の SongScore 案を簡略実装したもの。
本実装時は miku/src/agent/tools/music/score_schema.py に移植する想定。

tick 基準: 480 tick = 四分音符（LibreSVIP と同じ）。
"""

from pydantic import BaseModel, Field

TICKS_PER_QUARTER = 480


class SongNote(BaseModel):
    lyric: str = Field(description="かな1〜2モーラ（例: 'き', 'きゃ'）")
    key: int = Field(ge=0, le=127, description="MIDIノート番号（C4=60）")
    start_tick: int = Field(ge=0, description="曲頭からの位置")
    duration_tick: int = Field(gt=0)


class ChordSpan(BaseModel):
    """伴奏用のコード指定。1要素 = 1コード区間。"""

    chord: str = Field(description="コード名（例: 'C', 'Am', 'F', 'G7'）")
    start_tick: int = Field(ge=0)
    duration_tick: int = Field(gt=0)


class SongScore(BaseModel):
    title: str
    tempo: float = Field(gt=0, description="BPM")
    time_signature: str = Field(default="4/4", pattern=r"^\d+/\d+$")
    lyrics_full: str = Field(default="", description="歌詞全文（表示用）")
    vocal_notes: list[SongNote]
    chords: list[ChordSpan] = Field(default_factory=list)

    def duration_ticks(self) -> int:
        ends = [n.start_tick + n.duration_tick for n in self.vocal_notes]
        ends += [c.start_tick + c.duration_tick for c in self.chords]
        return max(ends) if ends else 0
