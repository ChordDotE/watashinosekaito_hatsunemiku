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

"""**区間の対比**——区間どうしがどれだけ違うかを、機械が数えて並べる（2026-10-11・本人承認）。

## 前提・背景

『てのひらのシグナル』第16版の振り返りで、Bメロとサビのコード・リズム・高さがほぼ同じで「単調」に聴こえた（本人も「単調だな」）。
設計の工程は区間ごとの強さ・高さを決め、指示書にも「抑揚は差で作る」とあり、最後の点検役も盛り上がりを見ていた。それでも通ったのは、
点検役に渡す表（`song_finish_gate.finish_material`）が区間ごとに**何が鳴っているか**だけで、**区間どうしの差**を数えていなかったから。
手直しのあとの点検（`song_final_gate.judge_edit`）も同じ表を読む。

## 決めたこと

- **機械は事実を並べるだけで、良し悪しを決めない**（あえて差を付けない曲もある——判断は点検役が設計の意図と照らして行う）。
- 区間ごとに: 歌（1小節あたりの音数・高さの幅と中心・最高音とその最長の長さ・裏から入るフレーズの割合）、伴奏（楽器の数・
  1小節あたりの音数・打鍵の強さの平均）、コード（並び）。
- 隣り合う区間の違い: 歌のリズムの似かた（後の区間の各小節が、前の区間でいちばん似た小節と入りの位置がどれだけ重なるか）・
  共通のコードのいちばん長い並び・中心の高さの差・最高音・伴奏の音数と楽器の数と打鍵の強さ。
- 曲の最高音を最初に使う区間と、その長さ。
- 新しい曲の最後の点検役と、手直しのあとの点検の両方に、同じ表を渡す（`song_final_gate.final_material`）。

触れる軸: リアルタイム性・継続性には触れない。体験（作った曲を人に聴かせられる水準へ）と公開性（点検役の材料が数百字増えるだけ）。
"""
from __future__ import annotations

from collections import Counter
from statistics import median

#: 4分音符の tick
Q = 480
#: リズムの型を比べる細かさ（16分音符）
GRID = Q // 4


def _label(spans) -> dict[int, str]:
    """区間の添字 → 表の名前（同じ名前の区間が2つ以上あれば小節を添える）。"""
    n = Counter(s.kind for s in spans)
    return {i: (f"{s.kind}({s.bar_from}-{s.bar_to})" if n[s.kind] > 1 else s.kind) for i, s in enumerate(spans)}


def _in(span, bt: int, tick: int) -> bool:
    return (span.bar_from - 1) * bt <= tick < span.bar_to * bt


def _rhythm(notes, span, bt: int) -> list[tuple]:
    """区間の小節ごとの、歌の入りの位置の型（16分の格子・音の無い小節は除く）。"""
    out = []
    for b in range(span.bar_from, span.bar_to + 1):
        t0 = (b - 1) * bt
        sig = tuple(sorted({round((n.start_tick - t0) / GRID) for n in notes if t0 <= n.start_tick < t0 + bt}))
        if sig:
            out.append(sig)
    return out


def _chords(chords, span, bt: int) -> list[str]:
    seq: list[str] = []
    for c in chords:
        if _in(span, bt, c.start_tick) and (not seq or seq[-1] != c.chord):
            seq.append(c.chord)
    return seq


def _common_run(a: list[str], b: list[str]) -> list[str]:
    """2つのコードの並びに共通する、いちばん長いひと続きの並び。"""
    best: list[str] = []
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            if k > len(best):
                best = a[i:i + k]
    return best


def _rhythm_alike(a: list[tuple], b: list[tuple]) -> int:
    """後の区間の各小節が、前の区間でいちばん似た小節とどれだけ入りの位置が重なるか（重なりの割合の平均・%）。"""
    def jac(x, y):
        return len(set(x) & set(y)) / len(set(x) | set(y))
    return round(100 * sum(max(jac(r, q) for q in a) for r in b) / len(b))


def _accomp(st, score, span, bt: int) -> tuple[int, int, float | None]:
    """区間の伴奏: (楽器の数, 音の数, 強さの平均)。手直しで伴奏の MIDI を直接書き換えたあとは MIDI から（強さは数えない）。"""
    midi = getattr(st, "accomp_midi", None)
    if midi is not None:
        hit = [(name, drum) for a, _z, _k, name, drum in midi if _in(span, bt, a)]
        return len({n for n, _d in hit}), len(hit), None
    names, count, vels = set(), 0, []
    spec = getattr(score, "arrangement", None)
    if spec is None:
        return 0, 0, None
    from song_audio import pitched_events
    for p in spec.parts:
        if getattr(p, "drum_steps", None):
            continue                                       # ドラムは楽器の数にだけ入れる（下）
        try:
            evs = pitched_events(score, p)
        except Exception:                                  # noqa: BLE001（数えられないパートは飛ばす）
            continue
        got = [(on, v, len(keys)) for on, _d, keys, v in evs if _in(span, bt, on)]
        if got:
            names.add(getattr(p, "role", "") + str(getattr(p, "timbre", "")))
            count += sum(k for _on, _v, k in got)
            vels += [v for _on, v, _k in got]
    from song_audio import bars_of_filter
    for p in spec.parts:
        if getattr(p, "drum_steps", None):
            bars = bars_of_filter(p.bar_filter, span.bar_to)
            if any(span.bar_from <= b <= span.bar_to for b in bars):
                names.add("ドラム")
    return len(names), count, (sum(vels) / len(vels) if vels else None)


def contrast_text(st) -> str:
    """区間の対比の表（区間が2つ未満・楽譜が無ければ空）。"""
    from song_design import bar_ticks
    from song_table import spell_pitch as _spell
    from song_timeline import LEAD_IN
    import vocal_expression
    score = getattr(st, "score", None)
    tl = getattr(st, "timeline", None)
    spans = [s for s in (getattr(tl, "spans", None) or []) if s.kind != LEAD_IN]
    if score is None or len(spans) < 2:
        return ""
    bt = bar_ticks(getattr(score, "time_signature", None) or "4/4")
    notes = sorted(getattr(score, "vocal_notes", None) or [], key=lambda n: n.start_tick)
    chords = sorted(getattr(score, "chords", None) or [], key=lambda c: c.start_tick)
    heads = {notes[s].start_tick for s, _e in vocal_expression.phrases(notes)} if notes else set()
    label = _label(spans)
    key = getattr(getattr(getattr(st, "song", None), "design", None), "key", None)
    spell_pitch = lambda k: _spell(int(k), key)                # noqa: E731（曲の調で綴る＝Eb の曲は D# でなく Eb）
    rows = ["【区間の対比】（機械が数えた事実だけ。差を付けない狙いの曲もあるので、良し悪しは設計の強さ・狙いと照らして判断する）"]
    info = []
    for i, s in enumerate(spans):
        ns = [n for n in notes if _in(s, bt, n.start_tick)]
        bars = s.bar_to - s.bar_from + 1
        inst, acc_n, vel = _accomp(st, score, s, bt)
        d = {"ns": ns, "rhythm": _rhythm(notes, s, bt), "chords": _chords(chords, s, bt),
             "inst": inst, "acc": acc_n / bars, "vel": vel,
             "mid": median(n.key for n in ns) if ns else None, "top": max((n.key for n in ns), default=None)}
        info.append(d)
        parts = [f"■{label[i]}（小節 {s.bar_from}-{s.bar_to}・設計の強さ{getattr(s, 'intensity', '?')}）"]
        if ns:
            top_len = max(n.duration_tick for n in ns if n.key == d["top"]) / Q
            off = [n for n in ns if n.start_tick in heads]
            parts.append(f"歌 1小節あたり{len(ns) / bars:.1f}音・高さ {spell_pitch(min(n.key for n in ns))}〜{spell_pitch(d['top'])}"
                         f"（中心 {spell_pitch(round(d['mid']))}）・最高音の最長 {top_len:g}拍"
                         + (f"・裏から入るフレーズ {sum(1 for n in off if n.start_tick % Q)}/{len(off)}" if off else ""))
        else:
            parts.append("歌なし")
        parts.append(f"伴奏 楽器{inst}・1小節あたり{d['acc']:.1f}音" + (f"・打鍵の強さの平均{vel:.0f}" if vel is not None else ""))
        if d["chords"]:
            parts.append("コード " + "-".join(d["chords"][:12]) + ("…" if len(d["chords"]) > 12 else ""))
        rows.append(" ／ ".join(parts))
    rows.append("隣り合う区間の違い:")
    for i in range(1, len(spans)):
        a, b = info[i - 1], info[i]
        bits = []
        if a["rhythm"] and b["rhythm"]:
            same = sum(1 for r in b["rhythm"] if r in set(a["rhythm"]))
            bits.append(f"歌のリズムの似かた {_rhythm_alike(a['rhythm'], b['rhythm'])}%（入りの位置がまったく同じ小節 {same}/{len(b['rhythm'])}）")
        if a["chords"] and b["chords"]:
            run = _common_run(a["chords"], b["chords"])
            bits.append(f"共通のコードの並び {'-'.join(run)}（{len(run)}個）" if len(run) >= 2 else "共通のコードの並びなし")
        if a["mid"] is not None and b["mid"] is not None:
            bits.append(f"中心の高さ {b['mid'] - a['mid']:+g}半音・最高音 {spell_pitch(a['top'])}→{spell_pitch(b['top'])}")
        bits.append(f"伴奏の音数 {a['acc']:.1f}→{b['acc']:.1f}／小節・楽器 {a['inst']}→{b['inst']}"
                    + (f"・打鍵の強さ {a['vel']:.0f}→{b['vel']:.0f}" if a["vel"] is not None and b["vel"] is not None else ""))
        rows.append(f"  {label[i - 1]}→{label[i]}: " + "・".join(bits))
    if notes:
        top = max(n.key for n in notes)
        first = next(n for n in notes if n.key == top)
        where = next((label[i] for i, s in enumerate(spans) if _in(s, bt, first.start_tick)), "区間の外")
        rows.append(f"曲の最高音 {spell_pitch(top)}: 最初に使うのは {where}（{first.start_tick // bt + 1}小節・{first.duration_tick / Q:g}拍）")
    return "\n".join(rows)
