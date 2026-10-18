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

"""**調声**——音符ごとの表情（`VocalTuning`）を規則で付ける。

## なぜ要るか

vpr は音符ごとに `aiExp`（ビブラート・ピッチ遷移・音量の出入り）を持てて、
V6 はこれを尊重する。ところが**全音符が既定値 0.5 のまま**で書き出されていた——
V6 で開いても**表情がまったく付いていない**状態だった。

人が歌うときに自然にやることを、そのまま規則にする:

| 場所 | 付けるもの | なぜ |
|---|---|---|
| 長い音（1拍以上）| ビブラート | 伸ばす音は揺れる。短い音では揺らさない |
| フレーズの頭 | 入りのピッチ遷移（しゃくり）| 下から音程へ入る、歌い出しの癖 |
| フレーズの終わり | 抜き（減衰）| 語尾は自然に減衰する。切りっぱなしにしない |
| 最も熱い区間 | 音量を上げる | 区間の起伏は音量にも出る |

★**揺れは「揺れる部分（音の後半）が 0.5秒以上ある音」にだけ付ける**（`vibrato_fits`・2026-09-27 に書き出しから移した）。
  短い音に掛けると揺れが周りきらずに下がったまま終わり、音程が低く聞こえる（実測 −48セント → 外すと −3）。
  以前は書き出し（`score_to_vpr._write_vibrato`）が黙って落としていた——**楽譜には揺れがあるのに鳴らない**形で、
  歌い方の工程も知らなかった（目録 表情-7）。いまは**決める側**（ここと歌い方の当てはめ）が秒で見て、
  書き出しは楽譜のとおり写す。歌い方の指示書（`song_expression.md`）にもこの条件を書いてある。

**LLM に音符数ぶんのパラメータを書かせない**（重いうえに一貫しない）。規則で十分に効く。

## カバーにも使う

オリジナルだけでなく**カバーにも当てる**（2026-09-01 移植）。
調声は**音高もタイミングも変えない**ので、原曲への忠実さを損なわずに表情だけが付く。
フレーズの切れ目が分からない場合は `phrases_by_gap` で休符から推定する。
"""
from __future__ import annotations

from song_schema import TICKS_PER_QUARTER as Q
from song_schema import SongNote, VocalTuning


def phrases_by_gap(notes: list[SongNote], gap_beats: float = 0.5) -> tuple[set[int], set[int]]:
    """休符の空きから**フレーズの頭と終わり**を推定する（区間情報が無いとき用）。

    `gap_beats` 拍以上あいたら、そこがフレーズの切れ目。戻り値は (頭, 終わり) の添字集合。
    """
    head: set[int] = set()
    tail: set[int] = set()
    if not notes:
        return head, tail
    head.add(0)
    for i in range(1, len(notes)):
        prev = notes[i - 1]
        if notes[i].start_tick - (prev.start_tick + prev.duration_tick) >= gap_beats * Q:
            head.add(i)
            tail.add(i - 1)
    tail.add(len(notes) - 1)
    return head, tail


#: 揺れを**音のどこから始めるか**（音の長さに対する割合）。歌は伸ばしてから揺れ始める。
#: 書き出し（`score_to_vpr._write_vibrato`）も同じ値を使う（**基準は1つ**）。
VIBRATO_FROM = 0.5

#: 揺れを掛ける**最短の長さ（秒）**——揺れる部分（音の後半）がこれに満たない音には掛けない（2026-09-27）。
#: ★短い音に掛けると**揺れが1周半ほどで切れ、下がったまま終わる**＝音程が低く聞こえる。実測（『風の航路へ』
#:   23.8秒の「あ」・0.73秒）: 揺れ 0.8 のままだと中ほどが **-48セント**、揺れを外すと **-3セント**
#:   （入りのしゃくりを外しても -23 までしか戻らない＝主因は揺れ）。人の歌も揺らすのは伸ばす音。
#:   0.5秒＝速さ 4.9Hz で約2.5周。`s5065b362_scoop_ab.py` の A/B が根拠。
VIBRATO_MIN_SEC = 0.5


def vibrato_fits(duration_tick: int, tempo: float | None) -> bool:
    """この長さの音に**揺れが効くか**（揺れる部分＝音の後半が `VIBRATO_MIN_SEC` 秒以上あるか）。

    テンポが分からない（`None`・0 以下）ときは判断できないので True（従来どおり長さの拍だけで決める）。
    """
    span = int(int(duration_tick) * (1.0 - VIBRATO_FROM))
    if span <= 0:
        return False
    if not tempo or tempo <= 0:
        return True
    return span * (60.0 / float(tempo) / Q) >= VIBRATO_MIN_SEC


def apply_tuning(notes: list[SongNote], phrase_head: set[int], phrase_tail: set[int],
                 peak_ticks: tuple[int, int] | None, *, tempo: float | None = None) -> int:
    """`notes` を**その場で**書き換え、調声を付けた音符の数を返す。

    `peak_ticks` は最も熱い区間の (開始tick, 終了tick)。**「サビ」という名前では探さない**
    ——名前で探すと、Cメロが山の曲や独自の呼び方をした曲で当たらない。

    `tempo`（BPM）を渡すと、**揺れが周りきらない短い音には揺れを付けない**（`vibrato_fits`）。
    ★呼び側は必ず渡す——渡さないと、短い音にも揺れが楽譜に残り、そのまま書き出される（2026-09-27 まで
      書き出しが落としていた。目録 表情-7）。

    ★**山より後ろの音にも「控えめ」を付ける**（2026-09-27 修正・目録 表情-9）。以前は条件が
      `not peak_ticks or 開始 < 山の頭` で、**山より後ろ（2番・ラスサビ前の区間・後奏の歌）には音量を
      書いていなかった**＝V6 の既定 0.5 で鳴り、山の前（0.45）より大きかった。この関数の説明（「それ以外は
      控えめ」）とも違い、わざとそうした記録も無い（2026-09-01 の移植のときから）ので、説明どおりに揃えた。
      差は 0.05＝実測で約 0.7dB（`amplitude` は 0→1 で 13.3dB）。
    """
    n_set = 0
    for i, n in enumerate(notes):
        beats = n.duration_tick / Q
        t: dict[str, float] = {}
        if vibrato_fits(n.duration_tick, tempo):
            if beats >= 1.0:
                t["vibrato_depth"] = 0.8                   # 伸ばす音は揺らす
            elif beats <= 0.3:
                t["vibrato_depth"] = 0.1                   # 細かい音は揺らさない（濁る）
        if i in phrase_head:
            t["pitch_transition_in"] = 0.7                 # 歌い出しはしゃくり上げる
            t["amplitude_attack"] = 0.6
        if i in phrase_tail:
            t["amplitude_release"] = 0.7 if beats >= 1.0 else 0.55   # 語尾は抜く
        if peak_ticks and peak_ticks[0] <= n.start_tick < peak_ticks[1]:
            t["amplitude"] = 0.68                          # 山の区間は前に出す
        else:
            t["amplitude"] = 0.45                          # それ以外は控えめ（起伏を音量でも作る）
        if t:
            notes[i] = n.model_copy(update={"tuning": VocalTuning(**t)})
            n_set += 1
    return n_set


def apply_by_gap(notes: list[SongNote], gap_beats: float = 0.5, *, tempo: float | None = None) -> int:
    """**区間情報が無いとき**の入口（カバー・声部向け）。休符からフレーズを推定して調声を当てる。

    `tempo` は `apply_tuning` と同じ（短い音に揺れを付けないために渡す）。
    """
    head, tail = phrases_by_gap(notes, gap_beats)
    return apply_tuning(notes, head, tail, None, tempo=tempo)


def vibrato_findings(notes: list[SongNote], tempo: float | None, *, where=None) -> list[str]:
    """**揺れが周りきらない短い音に、揺れが書かれている**所（所見だけ。直さない）。

    決める側（`apply_tuning`・歌い方の当てはめ）がテンポを知っていれば出ない。出るのは、テンポを渡さずに
    規則を当てた経路・人の手の直し・古い曲。書き出し（`score_to_vpr`）は楽譜のとおり写し、これを所見に出す。
    `where(tick)` を渡すと位置をその書き方で出す。
    """
    out: list[str] = []
    if not tempo or tempo <= 0:
        return out
    for n in notes:
        d = getattr(n.tuning, "vibrato_depth", None) if n.tuning else None
        if d and float(d) > 0 and not vibrato_fits(n.duration_tick, tempo):
            at = where(n.start_tick) if where else f"{n.start_tick}tick"
            sec = n.duration_tick * 60.0 / float(tempo) / Q
            out.append(f"{at}『{n.lyric}』（{sec:.2f}秒）に揺れ {float(d):g} が書いてある——揺れる部分が "
                       f"{VIBRATO_MIN_SEC:g}秒に満たず、揺れが周りきらずに音程が下がって聞こえる")
    return out
