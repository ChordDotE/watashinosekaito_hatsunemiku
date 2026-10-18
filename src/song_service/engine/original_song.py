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

"""オリジナル曲「ひかりのメロディ」(Claude 作曲・作詞)。

構成: イントロ4 + Aメロ8 + Bメロ4 + サビ8 + アウトロ4 = 28小節
BPM 100 / 4/4 / ハ長調。ボーカル音域 C4(60)〜E5(76)。
歌詞はモーラ分割しやすいよう「っ」「ん」「ー」を避けて作詞。
"""

from score_schema import TICKS_PER_QUARTER as Q
from score_schema import ChordSpan, SongNote, SongScore

TITLE = "ひかりのメロディ"
BPM = 100.0

# 歌詞全文（表示用）
LYRICS_FULL = (
    "ひかりのなかで めをさましたら あたらしいうたが きこえてくるよ\n"
    "とけいのはりが きざむりずむに こころがおどる くちずさむうた\n"
    "ことばにならない おもいをのせて そらにひびけと ねがいをこめて\n"
    "うたうよ みらいへ このこえ とどけ きみと みつけた ひかりの うたを"
)

# (開始小節, [(かな, MIDIキー, 拍数), ...])  小節は1始まり
_PHRASES: list[tuple[int, list[tuple[str, int, float]]]] = [
    # ---- Aメロ1 (5-8小節) ----
    (5, [("ひ", 64, 0.5), ("か", 64, 0.5), ("り", 67, 0.5), ("の", 67, 0.5),
         ("な", 69, 1.0), ("か", 67, 0.5), ("で", 64, 0.5)]),
    (6, [("め", 62, 0.5), ("を", 62, 0.5), ("さ", 64, 0.5), ("ま", 62, 0.5),
         ("し", 60, 1.0), ("た", 62, 0.5), ("ら", 64, 0.5)]),
    (7, [("あ", 69, 0.5), ("た", 69, 0.5), ("ら", 72, 0.5), ("し", 69, 0.5),
         ("い", 67, 0.5), ("う", 64, 0.5), ("た", 67, 0.5), ("が", 67, 0.5)]),
    (8, [("き", 65, 0.5), ("こ", 65, 0.5), ("え", 67, 0.5), ("て", 64, 0.5),
         ("く", 62, 0.5), ("る", 62, 0.5), ("よ", 60, 1.0)]),
    # ---- Aメロ2 (9-12小節) ----
    (9, [("と", 64, 0.5), ("け", 64, 0.5), ("い", 67, 0.5), ("の", 67, 0.5),
         ("は", 69, 1.0), ("り", 67, 0.5), ("が", 64, 0.5)]),
    (10, [("き", 62, 0.5), ("ざ", 62, 0.5), ("む", 64, 0.5), ("り", 62, 0.5),
          ("ず", 60, 1.0), ("む", 62, 0.5), ("に", 64, 0.5)]),
    (11, [("こ", 69, 0.5), ("こ", 69, 0.5), ("ろ", 72, 0.5), ("が", 69, 0.5),
          ("お", 67, 0.5), ("ど", 64, 0.5), ("る", 67, 1.0)]),
    (12, [("く", 65, 0.5), ("ち", 65, 0.5), ("ず", 67, 0.5), ("さ", 64, 0.5),
          ("む", 62, 0.5), ("う", 62, 0.5), ("た", 60, 1.0)]),
    # ---- Bメロ (13-16小節) ----
    (13, [("こ", 65, 0.5), ("と", 65, 0.5), ("ば", 69, 0.5), ("に", 69, 0.5),
          ("な", 67, 0.5), ("ら", 65, 0.5), ("な", 67, 0.5), ("い", 67, 0.5)]),
    (14, [("お", 67, 0.5), ("も", 67, 0.5), ("い", 71, 0.5), ("を", 71, 0.5),
          ("の", 69, 0.5), ("せ", 67, 0.5), ("て", 69, 1.0)]),
    (15, [("そ", 64, 0.5), ("ら", 67, 0.5), ("に", 71, 0.5), ("ひ", 71, 0.5),
          ("び", 72, 0.5), ("け", 71, 0.5), ("と", 69, 1.0)]),
    (16, [("ね", 67, 0.5), ("が", 67, 0.5), ("い", 69, 0.5), ("を", 71, 0.5),
          ("こ", 74, 0.5), ("め", 72, 0.5), ("て", 71, 1.0)]),
    # ---- サビ (17-24小節) ----
    (17, [("う", 72, 1.0), ("た", 72, 0.5), ("う", 74, 0.5), ("よ", 76, 2.0)]),
    (18, [("み", 74, 1.0), ("ら", 72, 0.5), ("い", 72, 0.5), ("へ", 71, 2.0)]),
    (19, [("こ", 72, 1.0), ("の", 72, 0.5), ("こ", 74, 0.5), ("え", 76, 2.0)]),
    (20, [("と", 74, 1.0), ("ど", 72, 0.5), ("け", 71, 2.5)]),
    (21, [("き", 72, 1.0), ("み", 72, 0.5), ("と", 74, 0.5),
          ("み", 76, 1.0), ("つ", 74, 0.5), ("け", 72, 0.5)]),
    (22, [("た", 71, 1.0), ("ひ", 71, 0.5), ("か", 72, 0.5),
          ("り", 74, 1.0), ("の", 72, 1.0)]),
    (23, [("う", 72, 1.0), ("た", 69, 1.0), ("を", 65, 2.0)]),
    # 24小節はボーカル休み（伴奏フィル）→ アウトロへ
]

# 1小節 = 1コード × 28小節
_CHORDS: list[str] = [
    # イントロ
    "C", "Am", "F", "G",
    # Aメロ1
    "C", "G", "Am", "F",
    # Aメロ2
    "C", "G", "Am", "F",
    # Bメロ
    "F", "G", "Em", "G",
    # サビ
    "C", "G", "Am", "F", "C", "G", "F", "G",
    # アウトロ
    "F", "G", "C", "C",
]

BEATS_PER_BAR = 4


def build_original_score() -> SongScore:
    notes: list[SongNote] = []
    for bar, phrase in _PHRASES:
        tick = (bar - 1) * BEATS_PER_BAR * Q
        for lyric, key, beats in phrase:
            dur = int(beats * Q)
            notes.append(SongNote(lyric=lyric, key=key, start_tick=tick, duration_tick=dur))
            tick += dur

    chords = [
        ChordSpan(chord=c, start_tick=i * BEATS_PER_BAR * Q, duration_tick=BEATS_PER_BAR * Q)
        for i, c in enumerate(_CHORDS)
    ]

    return SongScore(
        title=TITLE,
        tempo=BPM,
        time_signature="4/4",
        lyrics_full=LYRICS_FULL,
        vocal_notes=notes,
        chords=chords,
    )
