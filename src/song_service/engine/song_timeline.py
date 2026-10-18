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

"""**曲の時間軸**——「区間がどこにあるか」を知る唯一の場所。

## なぜ要るか

2026-08-22 の1日で音の不具合が7件見つかり、**うち6件が同じ型**だった——
「一致していなければならない2つの表現があるのに、一致を要求する場所が無い」。

同じ事実が独立に計算されていた:

    bars_of（区間→小節）      … compose_v2 / compose_v3 / cover_load の**3実装**
    sec_of_note（音符→区間）  … compose_v3 / cover_load の**2実装**
    sec_span（区間→歌の実時間）… compose_v3 だけだが bars_of と一致必須

しかも**順序が逆**だった。歌を並べてから、その結果を測って時間軸にしていたので、
配置の都合（手前休符・小節への切り上げ）が時間軸に染み出し、伴奏とずれた。
位置ずれを直すために入れた切り上げが、その場で**累積ずれ**（最大4.1小節）を生んだ。

## 直し方

**時間軸を先に決めて、歌も伴奏もそこへ置く。**

    ① 測る … 旋律から各区間の必要な長さを出す（**まだ置かない**）
    ② 組む … 区間を曲頭から並べて SectionTimeline を作る
    ③ 置く … 各区間の頭から歌を置く。コードも伴奏も**同じ timeline から**引く

同じ場所から座標を引くので、**ずれること自体が起きない**。

## 決めごと（守らないと1小節ずれる）

**小節番号は 1始まり・両端を含む。** `bar_filter` にこれが書かれていなかったことが、
解釈が3実装に割れた原因（ドラムはその判定を呼び忘れていた）。
tick は 0始まり。`bar_from=1` の区間の tick は 0 から始まる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class TimelineError(ValueError):
    """時間軸が壊れている。**黙って進めない**（音だけおかしい不具合になるため）。"""


@dataclass(frozen=True)
class SectionSpan:
    """区間1つ。**小節は1始まり・両端を含む**。"""
    kind: str
    bar_from: int
    bar_to: int
    has_vocal: bool = True
    intensity: int = 3

    @property
    def bars(self) -> int:
        return self.bar_to - self.bar_from + 1


#: 曲の頭に置く空の小節の区間名（設計には無い・歌も伴奏も鳴らない）。`SectionTimeline.build(lead_bars=)` が置く
LEAD_IN = "lead_in"


@dataclass
class SectionTimeline:
    """曲の時間軸。**区間の位置を知るのはここだけ**。

    `bars_of` も `sec_of_note` も**ここから導出する**（保存しない）。
    保存していたせいで、音符数と区間対応の長さが食い違う事故が起きた。
    """
    bar_ticks: int
    spans: list[SectionSpan] = field(default_factory=list)

    # ---- 組み立て ----------------------------------------------------------
    @classmethod
    def build(cls, sections: Iterable[Any], needed_ticks: dict[str, int],
              bar_ticks: int, lead_bars: int = 0) -> "SectionTimeline":
        """設計の区間と「歌が必要とする長さ」から時間軸を組む。

        Args:
            sections: 設計が並べた区間（`kind` / `bars` / `has_vocal` / `intensity`）
            needed_ticks: 区間 → 歌が必要とする tick 数（**歌のある区間だけ**）。
                旋律の実長で決めるため。宣言された `bars` より長くても短くても、
                **実際に歌う長さに合わせる**（合わせないと歌が伴奏をはみ出す）
            bar_ticks: 1小節の tick 数（4/4 なら 1920）
            lead_bars: 曲の頭に置く**空の小節**の数（区間名 `LEAD_IN`・歌も伴奏も鳴らない）。歌で始まる曲で
                0秒から声が出ないようにする。**曲全体を同じだけ後ろへずらす**ので、歌とコード・伴奏の関係は変わらない
                （2026-09-28。以前は最初の区間の歌だけを1拍ずらしていて、旋律がコードに対して1拍ずれていた）
        """
        spans: list[SectionSpan] = []
        bar = 1
        if lead_bars > 0:
            spans.append(SectionSpan(kind=LEAD_IN, bar_from=1, bar_to=lead_bars, has_vocal=False, intensity=1))
            bar += lead_bars
        for s in sections:
            kind = getattr(s, "kind", None) or (s.get("kind") if isinstance(s, dict) else None)
            if not kind:
                continue
            declared = int(getattr(s, "bars", None) or (s.get("bars") if isinstance(s, dict) else 0) or 1)
            sings = bool(getattr(s, "has_vocal", None) if not isinstance(s, dict)
                         else s.get("has_vocal", True))
            need = needed_ticks.get(kind)
            if need:                                       # 歌のある区間は**実際に歌う長さ**で
                n_bars = max(1, -(-int(need) // bar_ticks))    # 切り上げ
            else:
                n_bars = max(1, declared)                  # 歌なし区間は設計どおり
            inten = int(getattr(s, "intensity", None) if not isinstance(s, dict)
                        else s.get("intensity", 3) or 3)
            spans.append(SectionSpan(kind=kind, bar_from=bar, bar_to=bar + n_bars - 1,
                                     has_vocal=sings, intensity=inten))
            bar += n_bars
        tl = cls(bar_ticks=bar_ticks, spans=spans)
        tl.validate()
        return tl

    # ---- 不変条件 ----------------------------------------------------------
    def validate(self) -> None:
        """**守るべき状態**を確かめる。壊れていたら例外（黙って進めない）。"""
        if self.bar_ticks <= 0:
            raise TimelineError(f"1小節の長さが不正: {self.bar_ticks}")
        seen: set[str] = set()
        expect = 1
        for s in self.spans:
            if s.kind in seen:                             # ③ 名前は一意
                raise TimelineError(f"区間名が重複している: 『{s.kind}』")
            seen.add(s.kind)
            if s.bar_from > s.bar_to:                      # ② 負の長さ
                raise TimelineError(f"『{s.kind}』の小節範囲が逆: {s.bar_from}〜{s.bar_to}")
            if s.bar_from != expect:                       # ① 隙間なく連続
                raise TimelineError(
                    f"『{s.kind}』が小節{s.bar_from}から始まっている（{expect}のはず）"
                    f"——区間の間に穴か重なりがある")
            expect = s.bar_to + 1

    # ---- 引く --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.spans)

    def get(self, kind: str) -> SectionSpan | None:
        return next((s for s in self.spans if s.kind == kind), None)

    def tick_from(self, kind: str) -> int:
        """区間の**先頭 tick**（小節は1始まり・tick は0始まり）。"""
        s = self.get(kind)
        if s is None:
            raise TimelineError(f"知らない区間『{kind}』")
        return (s.bar_from - 1) * self.bar_ticks

    def tick_to(self, kind: str) -> int:
        """区間の**終わり tick**（この tick は含まない）。"""
        s = self.get(kind)
        if s is None:
            raise TimelineError(f"知らない区間『{kind}』")
        return s.bar_to * self.bar_ticks

    def at(self, tick: int) -> str:
        """その tick がどの区間か。範囲外なら最も近い端の区間。"""
        for s in self.spans:
            if (s.bar_from - 1) * self.bar_ticks <= tick < s.bar_to * self.bar_ticks:
                return s.kind
        if not self.spans:
            return ""
        return self.spans[0].kind if tick < 0 else self.spans[-1].kind

    @property
    def total_bars(self) -> int:
        return self.spans[-1].bar_to if self.spans else 0

    @property
    def total_ticks(self) -> int:
        return self.total_bars * self.bar_ticks

    # ---- 導出（保存しない）-------------------------------------------------
    def bars_of(self) -> dict[str, tuple[int, int]]:
        """区間 → (開始小節, 終了小節)。**1始まり・両端を含む**。"""
        return {s.kind: (s.bar_from, s.bar_to) for s in self.spans}

    def sections_of(self, notes: Iterable[Any]) -> list[str]:
        """音符 → 区間名。**導出する**ので、音符数と必ず同じ長さになる。

        以前は配置のときに別途ためており、音符を足し引きすると長さが食い違った
        （実際に手で直す羽目になった）。**始まりの tick** で決める——
        次の区間へ伸びている音は、始まった区間のものとして数える。
        """
        return [self.at(int(getattr(n, "start_tick", None) if not isinstance(n, dict)
                            else n["start_tick"])) for n in notes]

    def intensity_of(self) -> dict[str, int]:
        return {s.kind: s.intensity for s in self.spans}

    def key_ranges(self) -> list[tuple[str, int, int]]:
        """(区間名, 開始tick, 終了tick) の並び（曲頭から順）。"""
        return [(s.kind, (s.bar_from - 1) * self.bar_ticks, s.bar_to * self.bar_ticks)
                for s in self.spans]

    # ---- 保存・復元 --------------------------------------------------------
    def to_json(self) -> dict:
        return {"bar_ticks": self.bar_ticks,
                "spans": [{"kind": s.kind, "bar_from": s.bar_from, "bar_to": s.bar_to,
                           "has_vocal": s.has_vocal, "intensity": s.intensity}
                          for s in self.spans]}

    @classmethod
    def from_json(cls, d: dict) -> "SectionTimeline":
        tl = cls(bar_ticks=int(d.get("bar_ticks") or 1920),
                 spans=[SectionSpan(**s) for s in (d.get("spans") or [])])
        tl.validate()
        return tl

    def describe(self) -> str:
        return " → ".join(f"{s.kind}({s.bar_from}-{s.bar_to}"
                          + ("" if s.has_vocal else "・歌なし") + ")" for s in self.spans)


def measure_phrases(phrases: Iterable[Any], q: int, head_tick: int = 0) -> dict[str, int]:
    """**①測る**——旋律から、各区間が必要とする tick 数を出す（まだ置かない）。

    手前休符も含める。同じ区間の複数フレーズは足し合わせる。
    `head_tick` は曲の**いちばん最初**に置く間（0秒ちょうどから歌わせないため）で、
    最初の区間の必要長へ加算する。
    """
    need: dict[str, int] = {}
    first = True
    for p in phrases:
        kind = p.section
        # ★★**枠（鳴る長さ＋空き）で数える**（2026-09-13 修正）★★
        #   `beats` だけを足していたが、置く側（`_compose_once`）は
        #   **`beats + gap_after`＝枠ぶん**進む。2026-09-09 に「音を鳴らし切らず少し空ける」
        #   （`gap_after`）を入れてから、**区間の必要長が空きのぶん短く見積もられ**
        #   （1音 0.25拍×5音＝1行 1.25拍、verse 16小節で約2.5小節）、次の区間が早く固定されて
        #   **あふれた歌が区間の終わりで押し潰されていた**（0.08秒の音が並ぶ「一気に歌う」箇所）。
        #   本人:「初めはぽつぽつなのに、その後一気にまとめて歌うところがある」（『名前のない衝動』23〜26秒）。
        span = int(round(p.rest_before * q)) + sum(
            max(1, int(round((n.beats + float(getattr(n, "gap_after", 0.0) or 0.0)) * q)))
            for n in p.notes)
        if first:
            span += head_tick
            first = False
        need[kind] = need.get(kind, 0) + span
    return need
