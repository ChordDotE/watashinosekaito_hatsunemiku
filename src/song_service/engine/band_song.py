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

"""オリジナル曲2「はしりだせ」(Claude 作曲・作詞、バンド編成)。

構成: イントロ4 + Aメロ8 + Bメロ4 + サビ8 + 間奏4 + アウトロ4 = 32小節
BPM 128 / 4/4 / イ短調(Am)。ボーカル音域 C4(60)〜E5(76)。
編成: ボーカル + ドラム + エレキベース + ギター(パワーコード刻み)。
"""

from score_schema import TICKS_PER_QUARTER as Q
from score_schema import ChordSpan, SongNote, SongScore

TITLE = "はしりだせ"
BPM = 128.0

LYRICS_FULL = (
    "めざましとめて とびだすあさは かぜがせなかを おしてくれるよ\n"
    "まちのざわめき りずむにかえて きょうのぼくらは とまらないから\n"
    "むねのおくがわ あつくなるたび ちいさなゆめが さけびだすのさ\n"
    "はしりだせ かぜよりはやく ときめきを むねにかかえて\n"
    "はしりだせ ひかりのなかを あしたへとつづく みちを"
)

_PHRASES: list[tuple[int, list[tuple[str, int, float]]]] = [
    # ---- Aメロ1 (5-8) ----
    (5, [("め", 69, 0.5), ("ざ", 69, 0.5), ("ま", 67, 0.5), ("し", 69, 0.5),
         ("と", 72, 0.5), ("め", 69, 0.5), ("て", 67, 1.0)]),
    (6, [("と", 65, 0.5), ("び", 65, 0.5), ("だ", 67, 0.5), ("す", 65, 0.5),
         ("あ", 64, 0.5), ("さ", 62, 0.5), ("は", 60, 1.0)]),
    (7, [("か", 64, 0.5), ("ぜ", 64, 0.5), ("が", 67, 0.5), ("せ", 64, 0.5),
         ("な", 62, 0.5), ("か", 64, 0.5), ("を", 67, 1.0)]),
    (8, [("お", 67, 0.5), ("し", 67, 0.5), ("て", 69, 0.5), ("く", 67, 0.5),
         ("れ", 64, 0.5), ("る", 62, 0.5), ("よ", 62, 1.0)]),
    # ---- Aメロ2 (9-12) ----
    (9, [("ま", 69, 0.5), ("ち", 69, 0.5), ("の", 67, 0.5), ("ざ", 69, 0.5),
         ("わ", 72, 0.5), ("め", 69, 0.5), ("き", 67, 1.0)]),
    (10, [("り", 65, 0.5), ("ず", 65, 0.5), ("む", 67, 0.5), ("に", 65, 0.5),
          ("か", 64, 0.5), ("え", 62, 0.5), ("て", 60, 1.0)]),
    (11, [("きょ", 64, 0.5), ("う", 64, 0.5), ("の", 67, 0.5), ("ぼ", 64, 0.5),
          ("く", 62, 0.5), ("ら", 64, 0.5), ("は", 67, 1.0)]),
    (12, [("と", 67, 0.5), ("ま", 67, 0.5), ("ら", 69, 0.5), ("な", 67, 0.5),
          ("い", 64, 0.5), ("か", 64, 0.5), ("ら", 62, 1.0)]),
    # ---- Bメロ (13-16) ----
    (13, [("む", 65, 0.5), ("ね", 65, 0.5), ("の", 69, 0.5), ("お", 69, 0.5),
          ("く", 67, 0.5), ("が", 67, 0.5), ("わ", 65, 1.0)]),
    (14, [("あ", 67, 0.5), ("つ", 67, 0.5), ("く", 71, 0.5), ("な", 71, 0.5),
          ("る", 69, 0.5), ("た", 67, 0.5), ("び", 67, 1.0)]),
    (15, [("ち", 64, 0.5), ("い", 67, 0.5), ("さ", 71, 0.5), ("な", 71, 0.5),
          ("ゆ", 72, 0.5), ("め", 71, 0.5), ("が", 71, 1.0)]),
    (16, [("さ", 67, 0.5), ("け", 67, 0.5), ("び", 69, 0.5), ("だ", 71, 0.5),
          ("す", 74, 0.5), ("の", 72, 0.5), ("さ", 71, 1.0)]),
    # ---- サビ (17-24) ----
    (17, [("は", 72, 0.5), ("し", 72, 0.5), ("り", 74, 0.5), ("だ", 76, 1.5), ("せ", 72, 1.0)]),
    (18, [("か", 74, 0.5), ("ぜ", 74, 0.5), ("よ", 74, 0.5), ("り", 72, 0.5),
          ("は", 71, 0.5), ("や", 69, 0.5), ("く", 71, 1.0)]),
    (19, [("と", 72, 0.5), ("き", 72, 0.5), ("め", 74, 0.5), ("き", 76, 1.5), ("を", 72, 1.0)]),
    (20, [("む", 72, 0.5), ("ね", 72, 0.5), ("に", 72, 0.5), ("か", 69, 0.5),
          ("か", 67, 0.5), ("え", 65, 0.5), ("て", 65, 1.0)]),
    (21, [("は", 72, 0.5), ("し", 72, 0.5), ("り", 74, 0.5), ("だ", 76, 1.5), ("せ", 72, 1.0)]),
    (22, [("ひ", 74, 0.5), ("か", 74, 0.5), ("り", 74, 0.5), ("の", 72, 0.5),
          ("な", 71, 0.5), ("か", 69, 0.5), ("を", 71, 1.0)]),
    (23, [("あ", 69, 0.5), ("し", 69, 0.5), ("た", 72, 0.5), ("へ", 72, 0.5),
          ("と", 71, 0.5), ("つ", 69, 0.5), ("づ", 71, 0.5), ("く", 72, 0.5)]),
    (24, [("み", 74, 1.0), ("ち", 72, 0.5), ("を", 71, 2.5)]),
    # 25-32 は間奏・アウトロ（インスト）
]

_CHORDS: list[str] = [
    # イントロ
    "Am", "F", "C", "G",
    # Aメロ1 / Aメロ2
    "Am", "F", "C", "G",
    "Am", "F", "C", "G",
    # Bメロ
    "F", "G", "Em", "G",
    # サビ
    "C", "G", "Am", "F", "C", "G", "F", "G",
    # 間奏
    "Am", "F", "C", "G",
    # アウトロ
    "F", "G", "Am", "Am",
]

BEATS_PER_BAR = 4


def build_band_score() -> SongScore:
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
