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

"""**奏法**——音の中で高さが動くもの（2026-09-23）。

本人の指摘「**伴奏のところは、初めの音と入りの音と終わりの音が違うって多分あるんじゃないの。
吹く楽器に多い気がするけど**」から。調べたら、伴奏は **1音＝1つの固定した高さ**しか持てず、
トロンボーンのグリッサンドもブラスのフォールもサックスのベンドも**全部落ちていた**。
音色は GM の128種から選べるようになったのに、**奏法は選べないまま**だった。

## 分担（原則12「原因の構造から作る」）

```
LLM        「この音はフォール」と**名前で選ぶだけ**
   ↓
カタログ    楽器の族ごとに「どこからどこへ・どれだけの時間で・どんなカーブで」を持つ
（ここ）
   ↓
書き出す側   点列へ展開する。MIDI なら pitchwheel、vpr なら controllers
```

★**深さや時間を LLM に数値で書かせない。** 同じ「フォール」でも楽器で深さが違うのは
**楽器の側の性質**なので、そちらに持たせる。`Tone` に `key` を足したときと同じ形。

★**実物の楽器ができないことはしない。** ピアノ・オルガン・鍵盤打楽器は音の中で高さが動かない。
指定されても**落として所見に出す**（黙って鳴らすと「ピアノがしゃくり上げる」ことになる）。
★2026-09-27: 落とす前に、`artic_plan` の検査が所見として奏法の工程へ返し、差分で直させる（`song_fix`）。


## 音になるまで（実測済み・2026-09-23）

  ・MIDI → FluidSynth  … `pitchwheel` は鳴る。既定のベンド幅 ±2半音では
                          フォール（-7）が入らないので **RPN で ±12 へ広げる**
  ・**ベンドはチャンネル単位** … 同じチャンネルの音は**全部**曲がる。
                          だから**和音を鳴らすパートには奏法を付けない**（`song_schema` の注記）

実測と経緯は [accomp_articulation.md](../../../../docs/plan/accomp_articulation.md)。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

#: ベンド幅（半音）。**RPN でチャンネルごとに設定する**。
#: 既定の ±2 ではフォールもグリッサンドも入らない（実測で確認）。
BEND_RANGE = 12


class BendPoint(BaseModel):
    """奏法の折れ点1つ。**この3つで「どこからどこへ・どれだけの時間で・どんなカーブで」が揃う。**"""
    at: float = Field(ge=0.0, le=1.0, description="音の中の位置（0=頭 / 1=終わり）。**音の長さに対する割合**")
    st: float = Field(default=0.0, description="そこでの半音のずれ（0＝書かれた音高）")
    to_next: bool = Field(default=False, description="次の音の高さへ（`st` より優先。コードが計算する）")
    curve: str = Field(default="linear", description="ここへ**向かう**動き方: linear / ease_in / ease_out / step")


#: 奏法 → 折れ点。**楽器の族ごとに深さと時間が変わる**（下の `_BY_FAMILY` で上書き）。
#: ここに書いてあるのは**基準となる形**で、数字は聴いて詰める前提の起案値。
_BASE: dict[str, list[BendPoint]] = {
    # ---- 入り（音の頭）----
    "scoop":   [BendPoint(at=0.00, st=-2.0), BendPoint(at=0.10, st=0.0, curve="ease_out")],
    "plop":    [BendPoint(at=0.00, st=+4.0), BendPoint(at=0.08, st=0.0, curve="ease_out")],
    # ---- 終わり（音の尻）----
    "fall":    [BendPoint(at=0.80, st=0.0), BendPoint(at=1.00, st=-7.0, curve="ease_in")],
    "doit":    [BendPoint(at=0.85, st=0.0), BendPoint(at=1.00, st=+5.0, curve="ease_in")],
    # ---- 音の中 ----
    "bend_up": [BendPoint(at=0.15, st=0.0), BendPoint(at=0.35, st=+2.0, curve="ease_out")],
    "bend_release": [BendPoint(at=0.15, st=0.0), BendPoint(at=0.35, st=+2.0, curve="ease_out"),
                     BendPoint(at=0.65, st=0.0, curve="ease_in")],
    # ---- 次の音へ ----
    "gliss":   [BendPoint(at=0.70, st=0.0), BendPoint(at=1.00, to_next=True, curve="linear")],
    "portamento": [BendPoint(at=0.55, st=0.0), BendPoint(at=1.00, to_next=True, curve="ease_in")],
}

#: 楽器の族（GM プログラム番号の範囲）→ **その楽器にできる奏法**。
#: ★**実物ができないことはできない**。鍵盤とオルガンは音の中で高さが動かないので空。
_FAMILY: list[tuple[range, str, set[str]]] = [
    (range(0, 8),    "ピアノ",       set()),
    (range(8, 16),   "鍵盤打楽器",   set()),
    (range(16, 24),  "オルガン",     set()),
    # ★ギターとベースの `fall` を足した（2026-09-23・実機で LLM が選んで弾かれたため）。
    #   **指板を滑り下ろすのも、アームで落とすのも実在する奏法**——狭く作りすぎていた。
    #   「実物ができないことはしない」は守るが、**できることを落とさない**ほうも同じくらい大事。
    (range(24, 32),  "ギター",       {"bend_up", "bend_release", "gliss", "scoop", "fall"}),
    (range(32, 40),  "ベース",       {"bend_up", "gliss", "scoop", "fall"}),
    (range(40, 56),  "弦",           {"portamento", "gliss", "scoop", "fall"}),
    (range(56, 64),  "金管",         {"fall", "doit", "scoop", "plop", "gliss", "bend_up"}),
    (range(64, 80),  "リード・笛",   {"scoop", "fall", "doit", "bend_up", "bend_release"}),
    (range(80, 104), "シンセ",       set(_BASE)),          # 電子楽器なので制約が無い
    (range(104, 128), "その他",      {"scoop", "bend_up"}),
]

#: 族ごとの味つけ（**同じ奏法でも楽器で深さと時間が違う**）。(奏法, 族名) → 深さの倍率
_DEPTH: dict[tuple[str, str], float] = {
    ("fall", "金管"): 1.0,          # ブラスのフォールは派手に落とす（基準）
    ("fall", "弦"): 0.45,           # 弦は控えめ
    ("fall", "リード・笛"): 0.7,
    ("scoop", "ギター"): 0.5,       # ギターのしゃくりは半音ぶん
    ("scoop", "ベース"): 0.5,
    ("bend_up", "ギター"): 1.0,     # チョーキングは全音（基準）
    ("bend_up", "金管"): 0.5,       # リップベンドは半音
    ("bend_up", "ベース"): 0.5,
}


def family_of(program: int) -> tuple[str, set[str]]:
    """GM プログラム番号（0始まり）→ (族の名前, できる奏法)。"""
    for rng, name, allowed in _FAMILY:
        if program in rng:
            return name, allowed
    return "その他", {"scoop", "bend_up"}


def names() -> list[str]:
    """カタログにある奏法の名前（プロンプトに出す用）。"""
    return sorted(_BASE)


#: 深さの上限（半音）。ベンド幅（`BEND_RANGE`）を超えると音源側で潰れ、**書いたとおりに鳴らない**。
#: ★2026-09-27（目録の 伴奏-10）: 上限は奏法の指示書に書き、超えた深さは `artic_plan` の検査が所見にして
#:   奏法の工程へ返す（差分で直させる）。
#: ★2026-09-28（組み立ては補正しない）: 直しても残った深さは**頭打ちにせず、その奏法を掛けない**（`shape` が None を返す）。
#:   以前はここで 12半音に切り詰めて鳴らしていた（書いたものと違う深さで鳴る）。
MAX_DEPTH = float(BEND_RANGE)


def too_deep(depth_st: float | None) -> bool:
    """書いた深さがベンドの幅（`MAX_DEPTH`）を超えるか。超えたら**その奏法は掛けない**（2026-09-28）。"""
    return depth_st is not None and abs(float(depth_st)) > MAX_DEPTH


def _peak(base: list[BendPoint]) -> float:
    """いちばん大きく動く点の半音（次の音へ向かう点は数えない）。`gliss`・`portamento` は 0。"""
    return max((abs(p.st) for p in base if not p.to_next), default=0.0)


def depth_unusable(name: str, depth_st: float | None) -> bool:
    """書いた深さをこの奏法に**掛けられない**か（2026-10-02・洗い出し H1）。0 以下の深さと、深さを持たない奏法
    （`gliss`・`portamento`＝行き先は次の音）に書いた深さ。型（`artic_plan.ArticNote`・`vocal_artic.VocalArticNote`）と
    奏法の検査が先に差し戻すので、組み立てに来たら鳴らさない（`shape` が None）。

    ★以前は 0 を「カタログの基準の形」で鳴らし（楽器ごとの倍率も掛けない＝弦のフォールが標準 3.1 のはずが 7 半音）、
      `gliss`・`portamento` に書いた深さは黙って無視していた。
    """
    if depth_st is None:
        return False
    base = _BASE.get(name or "")
    return float(depth_st) <= 0 or (base is not None and _peak(base) <= 0)


def depth_why(name: str, depth_st: float | None) -> str:
    """`depth_unusable` の所見の文（奏法の検査と `why_not` で同じ文）。"""
    base = _BASE.get(name or "")
    if base is not None and _peak(base) <= 0:
        return (f"『{name}』は次の音へ向かう奏法で、深さは書けない（行き先は次の音で決まる）——"
                f"depth_st は書かない（直しでは name に同じ『{name}』を書くと深さが消える）")
    return (f"深さ {float(depth_st):g}半音は書けない——深さは動かす半音の大きさ（0 より大きい数）。"
            "表の標準を目安に数を書く。動かさないならその奏法をやめる")


def has_depth(name: str) -> bool:
    """この奏法が**深さを持つ**か（`gliss`・`portamento`＝行き先は次の音、は持たない）。
    ★2026-10-04（本人「深さを書かないのは良くない。書き忘れたのか標準にしたのかが曖昧になる」）: 深さを持つ奏法は、
      LLM が深さを必ず書く（型 `artic_plan.ArticNote`・`vocal_artic.VocalArticNote`）。組み立てに空が来たら不具合（`AssemblyError`）"""
    return base_depth(name) is not None


def shape(name: str, program: int, depth_st: float | None = None) -> list[BendPoint] | None:
    """奏法の名前と楽器 → 折れ点。**その楽器にできなければ None**（呼ぶ側が所見に出す）。
    **書いた深さがベンドの幅（`MAX_DEPTH`）を超えても None**（頭打ちにして鳴らさない・2026-09-28）。
    **0 以下の深さ・深さを持たない奏法（`gliss`・`portamento`）に書いた深さも None**（2026-10-02・`depth_unusable`）。

    `depth_st` を渡すと**深さだけ**そちらで上書きする（2026-09-23）。
    ★**形（いつ動き始めて・どんなカーブで）はカタログのまま**で、**大きさだけ**を変える。
      本人の判断——「標準でこれぐらい落ちる、と渡したうえで、
      **違う値にするならその理由を考えさせれば問題ない**」。
    ★**符号は奏法が決める**（フォールは下・ドゥイットは上）。書く側は大きさだけ考えればよい（伴奏は正の数だけ・型が見る）。
    ★名前は**カタログの綴りそのまま**（2026-10-02・洗い出し）。以前は大小・前後の空白を無視して読んでいた（型 `ArticNote.name` が選択肢で閉じた）。
    """
    key = name or ""
    base = _BASE.get(key)
    if base is None:
        return None
    fam, allowed = family_of(program)
    if key not in allowed or too_deep(depth_st) or depth_unusable(key, depth_st):
        return None
    if depth_st is None:
        scale = _DEPTH.get((key, fam), 1.0)
        return [p.model_copy(update={"st": p.st * scale}) for p in base]
    # 深さの上書き: **いちばん大きく動く点を基準**に、全体を比で伸縮する
    k = float(depth_st) / _peak(base)
    return [p.model_copy(update={"st": p.st * k}) for p in base]


def shape_of(name: str, depth_st: float | None = None) -> list[BendPoint] | None:
    """**楽器の縛りを通さずに**形だけ引く（2026-09-23）。

    ★**声のために足した。** 人の声は**この表の楽器ではない**ので、
      「サックスで代用する」と**サックスにできないものが声でもできないことになる**
      （`plop`＝上から落として入る、は人の歌にあるのにサックスの持ち物には無い）。
      **代用は筋が悪い**ので、声の側は**自分の使える名前の一覧**（`vocal_artic.VOCAL_NAMES`）で
      縛り、ここでは形だけを渡す。

    **楽器に使ってはいけない**——楽器の「できる／できない」は `shape()` が見ている。
    ★深さは**大きさ**（0 より大きい数）で渡す（歌の崩しの型も 2026-10-02 から 0 より大きい数だけ。以前は負も受けて呼び手が `abs` を取った）。
      空なら**カタログの基準の形**（形の確かめ用）。歌の崩しは 2026-10-04 から LLM が深さを必ず書く（`vocal_artic.VocalArticNote`）ので、
      書き出し（`score_to_vpr`）に空は来ない——来たら不具合として止める（以前は空を歌の標準 `VOCAL_STANDARD` で埋めていた）。
    """
    base = _BASE.get(name or "")
    if base is None or too_deep(depth_st) or depth_unusable(name, depth_st):   # 深すぎる・0 以下の深さは掛けない
        return None
    if depth_st is None:
        return [p.model_copy() for p in base]
    k = float(depth_st) / _peak(base)
    return [p.model_copy(update={"st": p.st * k}) for p in base]


def base_depth(name: str) -> float | None:
    """カタログの基準の形の深さ（半音・楽器の倍率なし）。深さを持たない奏法・知らない名前は None。"""
    base = _BASE.get(name or "")
    peak = _peak(base) if base else 0.0
    return peak or None


def standard_depth(name: str, program: int) -> float | None:
    """その楽器での**標準の深さ**（半音）。プロンプトに出して「基準」を渡すために使う。"""
    pts = shape(name, program)
    if not pts:
        return None
    peak = _peak(pts)
    return round(peak, 1) if peak else None


def why_not(name: str, program: int, depth_st: float | None = None) -> str:
    """鳴らせない理由を**人が読める形**で（所見に出す）。"""
    key = name or ""
    fam, allowed = family_of(program)
    if key not in _BASE:
        return f"知らない奏法『{name}』（使えるのは {' / '.join(names())}）"
    if key in allowed and too_deep(depth_st):
        return (f"『{name}』の深さ {float(depth_st):g}半音はベンドの幅（{MAX_DEPTH:g}半音）を超えるので掛けなかった"
                f"（{MAX_DEPTH:g}半音に切り詰めて鳴らすことはしない）")
    if key in allowed and depth_unusable(key, depth_st):
        return depth_why(key, depth_st)
    if not allowed:
        return f"『{name}』は{fam}では鳴らせない——**実物が音の中で高さを変えられない**"
    return (f"『{name}』は{fam}では鳴らせない（この楽器にできるのは "
            f"{' / '.join(sorted(allowed))}）")


def _ease(u: float, curve: str) -> float:
    """0〜1 の進み具合 → 0〜1 の動き具合。**`step` はその点で跳ぶ**。"""
    if curve == "step":
        return 1.0 if u >= 1.0 else 0.0
    if curve == "ease_in":                                 # じわっと始めて速く終わる
        return u * u
    if curve == "ease_out":                                # 速く始めて着地で緩める
        return 1.0 - (1.0 - u) ** 2
    return u                                               # linear（既定）


def expand(points: list[BendPoint], dur_tick: int, *, next_st: float | None = None,
           per_tick: int = 24) -> list[tuple[int, float]]:
    """折れ点 → **(音の頭からの tick, 半音)** の列。書き出す側はこれを生値に直すだけ。

    `next_st` は「次の音が何半音ずれているか」（`to_next` の行き先）。
    **次の音が無ければ `to_next` の点は落とす**——行き先が無いのに滑らせない。

    `per_tick` は折れ点のあいだを何点で刻むか（曲線をなめらかに見せるための刻み数）。
    """
    out: list[tuple[int, float]] = []
    prev_tick, prev_st = 0, 0.0
    for p in points:
        st = next_st if p.to_next else p.st
        if st is None:                                     # 行き先の無い to_next
            continue
        tick = int(round(p.at * dur_tick))
        if p.curve == "step":
            out.append((max(0, tick - 1), prev_st))
            out.append((tick, st))
        else:
            n = max(1, per_tick)
            for i in range(1, n + 1):
                u = i / n
                out.append((prev_tick + int(round((tick - prev_tick) * u)),
                            prev_st + (st - prev_st) * _ease(u, p.curve)))
        prev_tick, prev_st = tick, st
    # **同じ tick に複数の点が来たら、後の値を採る**（重なると音源の解釈が割れる）
    merged: dict[int, float] = {}
    for tick, st in out:
        merged[max(0, min(dur_tick, tick))] = st
    return sorted(merged.items())


def ends_bent(points: list[BendPoint]) -> bool:
    """★**音の終わりで 0 に戻っていないか**。戻っていないなら、書き出す側が
    **次の音の直前まで待ってから 0 へ戻す**必要がある。

    `note_off` と同時に戻すと**リリースの余韻が跳ね上がる**（2026-09-23 に実測で踏んだ）。
    """
    return bool(points) and (points[-1].to_next or abs(points[-1].st) > 1e-6)
