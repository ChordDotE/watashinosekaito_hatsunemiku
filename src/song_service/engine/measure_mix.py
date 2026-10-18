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

"""**鳴らした音を実測する**。見積もりではなく、書き出した wav を測る。

## なぜ見積もりを捨てたか（2026-08-22 の実測）

当初は「鳴るパートのゲイン×音数」から音量差を**見積もって**いた。しかし——

    見積もり  7.9 dB の差がある
    実測      2.9 dB しか無い

**見積もりは当てにならない**。しかも外れ方が危険な方向で、
**平坦な曲を「差がある」と判定して通してしまった**（今日の不具合をもう一度通す網）。

理由は単純で、RMS は「鳴っている音の数×ゲイン」に比例しない。
持続音・残響・音域が効くうえ、実際に踏んだ不具合（ドラムが全小節で鳴る）は
**楽譜を見ても分からない**——鳴らして初めて出る。

そこで**鳴らした後に測る**方式へ変えた。曲を組み終えたあと、伴奏の wav を区間ごとに
切って RMS を出す。**検査が捕まえたい対象は音**なのだから、音を測るのが筋。

## 使いどころ

工程の途中（まだ鳴らしていない）では呼べない。**仕上げの後**に呼んで、
所見を関門へ渡す。
"""
from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any


def _load(path: Path) -> tuple[list[float], int]:
    """wav を読む（モノラル化）。numpy を使わずに済ませる（依存を増やさない）。"""
    import array
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = array.array("h")
        raw.frombytes(w.readframes(w.getnframes()))
    if ch > 1:
        raw = array.array("h", [int(sum(raw[i:i + ch]) / ch) for i in range(0, len(raw), ch)])
    return [x / 32768.0 for x in raw], sr


def _rms_db(xs: list[float]) -> float:
    if not xs:
        return -99.0
    r = math.sqrt(sum(x * x for x in xs) / len(xs))
    return 20 * math.log10(max(r, 1e-9))


def section_levels(wav: Path, timeline: Any, tempo: float) -> dict[str, float]:
    """区間ごとの音量（dB）を**実測**する。"""
    a, sr = _load(Path(wav))
    bar_s = 60.0 / tempo * (timeline.bar_ticks / 480 / 4) * 4    # 1小節の秒数
    out: dict[str, float] = {}
    for s in timeline.spans:
        i0 = int((s.bar_from - 1) * bar_s * sr)
        i1 = min(int(s.bar_to * bar_s * sr), len(a))
        out[s.kind] = _rms_db(a[i0:i1]) if i1 > i0 else -99.0
    return out


def inspect_levels(levels: dict[str, float], timeline: Any) -> list[tuple[str, str]]:
    """実測した音量から**所見**を作る（判定はしない・関門が決める）。

    戻りは (見つけたこと, 根拠) の並び。
    """
    spans = [s for s in timeline.spans if s.kind in levels and levels[s.kind] > -90]
    if len(spans) < 2:
        return []
    top = max(levels[s.kind] for s in spans)
    rel = {s.kind: levels[s.kind] - top for s in spans}
    spread = max(rel.values()) - min(rel.values())
    detail = " / ".join(f"{s.kind}:強さ{s.intensity}→{rel[s.kind]:+.1f}dB" for s in spans)

    out: list[tuple[str, str]] = []
    if spread < 4.0:
        out.append((
            f"**鳴らして測ったら、曲全体の強弱の幅が {spread:.1f}dB しかなかった**"
            f"——区間が変わったことが聴いて分からない恐れがある",
            f"{detail}／目安: 3dB未満=差が伝わらない・5〜8dB=はっきり分かる。"
            f"※ずっと静かな曲・淡々と進める曲なら、これで正しい"))

    # 宣言した強さと実際の音が食い違う組（強さが3以上離れているのに音は3dB も違わない）
    bad = []
    for i, s in enumerate(spans):
        for t in spans[i + 1:]:
            if abs(s.intensity - t.intensity) >= 3 and abs(rel[s.kind] - rel[t.kind]) < 3.0:
                bad.append(f"『{s.kind}』(強さ{s.intensity}) と 『{t.kind}』(強さ{t.intensity}) "
                           f"の音の差が {abs(rel[s.kind] - rel[t.kind]):.1f}dB")
    if bad:
        out.append((
            "**宣言した強さと、鳴らした音の差が食い違っている**区間の組がある",
            " ; ".join(bad[:6])
            + "／※楽器の出入りで差を付けていて音量では差が出ない作りなら、そのままでよい"))

    # 逆方向: 強さが低いのに音が大きい（順序が逆転している）
    rev = []
    for i, s in enumerate(spans):
        for t in spans[i + 1:]:
            if s.intensity - t.intensity >= 2 and rel[s.kind] - rel[t.kind] < -3.0:
                rev.append(f"『{s.kind}』(強さ{s.intensity}) より "
                           f"『{t.kind}』(強さ{t.intensity}) の方が "
                           f"{rel[t.kind] - rel[s.kind]:.1f}dB 大きい")
    if rev:
        out.append(("**強さの順序と、鳴っている音の大小が逆転している**区間がある",
                    " ; ".join(rev[:6])))
    return out
