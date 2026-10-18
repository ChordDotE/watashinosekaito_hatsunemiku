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

"""**仕上げの点検役の材料のうち、楽譜から数える事実**（2026-10-03・組G4・`docs/plan/song_viewpoints_review.md` §1-5）。

## 前提・背景

仕上げの点検役（`song_finish_gate`）の指示書は「歌い方が詞の意味と噛み合うか」「アカペラで和声として成り立つか」
「ハモリの書き方の基本（主旋律を超える・交差・平行5度8度の連続）」を見よと言うのに、材料に詞の本文・コード・声部の音が無く、
数え方も音の数（伸ばすパッド1音と速い分散和音何十音が同じ重み）・曲全体の割合（区間ごとの単調さを見落とす）だった
（`dev_tools/scratch/s5065b362_content_check_memo.md` B-4）。

## 決めたこと

- ここは**数えるだけ**（良し悪しを断じない・閾値で出す出さないを決めない）。読み方は指示書 `prompts/song_finish_judge.md`。
- 高さはどれも**鳴る高さ**（楽譜の歌・声部・コード・彩りは組み立てが区間の調へ移したあと）。
- 音は `(開始tick, 終わりtick, 高さ)` の組で受け取る（楽譜の型に依らない＝テストで組を渡せる）。
"""
from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter

#: 4分音符の tick（`song_schema.TICKS_PER_QUARTER`）。拍の長さはこれで数える
Q = 480
#: 「高さが近い」の幅（半音）。主旋律と彩りの比べ方（`song_finish_gate.colour_clashes`）と同じ
NEAR = 4
#: 「伸ばす音」の長さ（tick）。1拍（4分音符）以上
LONG_NOTE = Q
#: アカペラの和声の材料に載せる小節数（区間ごと・頭から）
HARMONY_BARS = 4
#: 区間ごとに載せる見せる歌詞の行数・字数の上限（超えたら切ったことを書く）
LYRIC_LINES = 4
LYRIC_CHARS = 120

Note3 = tuple[int, int, int]


def _beats(t: float) -> str:
    return f"{t / Q:.1f}"


def _union_len(ivs: list[tuple[int, int]]) -> int:
    total, cur_a, cur_z = 0, None, None
    for a, z in sorted(ivs):
        if cur_z is None or a > cur_z:
            if cur_z is not None:
                total += cur_z - cur_a
            cur_a, cur_z = a, z
        else:
            cur_z = max(cur_z, z)
    return total + ((cur_z - cur_a) if cur_z is not None else 0)


def voice_names(tracks) -> list[str]:
    """声部の呼び名（`title`。同じ名前が2本以上なら番号を付ける）。"""
    seen = Counter(t.title or "声部" for t in tracks)
    num: Counter = Counter()
    out = []
    for t in tracks:
        n = t.title or "声部"
        num[n] += 1
        out.append(f"{n}{num[n]}" if seen[n] > 1 else n)
    return out


def beat_ticks(score) -> int:
    """拍の長さ（tick）。拍子の分母の音符（4/4 なら4分・6/8 なら8分）。"""
    den = int((getattr(score, "time_signature", None) or "4/4").split("/")[1])
    return Q * 4 // den


# ---- 彩りと主旋律・声部の重なり（長さで数える）-------------------------------------------------

def near_overlap(targets: list[Note3], colour: list[tuple[int, int, int, str]], sec_at,
                 near: int = NEAR) -> dict[str, list]:
    """`targets`（主旋律か声部の音）の鳴っている時間のうち、**高さが `near` 半音以内の彩りの音が同時に鳴っている長さ**を区間ごとに。

    戻り値は 区間 → [重なった長さ(tick), targets の鳴っている長さ(tick), 重なった彩りのパート名の Counter(tick)]。
    1音の中で複数の彩りの音が重なっても時間は1回だけ数える（伸ばすパッド1音と速い分散和音何十音を、鳴っている長さで比べる）。
    区間は targets の音の頭で決める。
    """
    out: dict[str, list] = {}
    for a, z, k in targets:
        row = out.setdefault(sec_at(a), [0, 0, Counter()])
        row[1] += z - a
        hits = [(max(a, ca), min(z, cz), name) for ca, cz, ck, name in colour if ca < z and a < cz and abs(ck - k) <= near]
        row[0] += _union_len([(x, y) for x, y, _n in hits])
        for x, y, name in hits:
            row[2][name] += y - x
    return out


def overlap_text(by_sec: dict[str, list], order: list[str]) -> str:
    """`near_overlap` の結果を区間の順に1行へ（`chorus 12.5/32.0拍（笛 10.0）`）。重なりの無い区間も 0 で出す。"""
    rows = []
    for k in order:
        if k not in by_sec:
            continue
        hit, total, names = by_sec[k]
        who = "・".join(f"{n} {_beats(t)}" for n, t in names.most_common(3))
        rows.append(f"{k} {_beats(hit)}/{_beats(total)}拍" + (f"（{who}）" if who else ""))
    return "・".join(rows)


# ---- 歌と伴奏が同じ高さで同時に鳴る（複数の工程にまたがる: 旋律×伴奏）-----------------------------
# 2026-10-04（本人の実機確認『水たまりのプリズム』8〜9秒）: 歌「う・つ・む・い・た」に、ギターのアルペジオが同じ F#4・A4 を
#   同時に鳴らし、歌声が二重に聞こえた。設計に歌と同じ高さに寄せる意図は無かった。それまでの材料は「歌の音域に入る音の数」で、
#   同じ高さかを見ていなかった。ここは数えるだけ（ユニゾン・オクターブ重ねで寄り添う意図なら正しい。読み方は点検役の指示書）

def same_pitch(melody: list[Note3], acc: list[tuple[int, int, int, str]]) -> list[tuple[int, str, int]]:
    """歌の音と**同じ高さ（音名・オクターブ込み＝同じ MIDI 番号）で同時に鳴る伴奏の音**。
    戻り値は (歌の音の頭 tick, 伴奏のパート名, 高さ) の並び。歌の1音 × パートで1件（同じパートの分散和音が1音の間に
    何度当たっても1件）。同時＝鳴っている時間が少しでも重なる。高さは鳴る高さどうしで比べる。"""
    out: list[tuple[int, str, int]] = []
    for a, z, k in sorted(melody):
        names: list[str] = []
        for ca, cz, ck, name in acc:
            if ck == k and ca < z and a < cz and name not in names:
                names.append(name)
        out += [(a, nm, k) for nm in names]
    return out


def same_pitch_rows(hits: list[tuple[int, str, int]], sec_at, where, spell, order: list[str],
                    lyric_at=None, show: int = 8) -> dict[str, str]:
    """`same_pitch` の結果 → 区間 → 1行（`3か所: 4:1.00「う」F#4 ギター…`）。区間ごとに頭から `show` か所（ほかの数も書く）。
    `lyric_at`（tick → 歌詞）を渡すと、歌の音の歌詞を添える。"""
    by: dict[str, list[str]] = {}
    parts: dict[str, Counter] = {}
    for t, name, k in hits:
        ly = lyric_at(t) if lyric_at else ""
        sec = sec_at(t)
        by.setdefault(sec, []).append(f"{where(t)}" + (f"「{ly}」" if ly else "") + f" {spell(k)} {name}")
        parts.setdefault(sec, Counter())[name] += 1
    out = {}
    for sec in order:
        items = by.get(sec)
        if items:
            per = "・".join(f"{n} {c}" for n, c in parts[sec].most_common())
            out[sec] = (f"{len(items)}か所（{per}）: " + "・".join(items[:show])
                        + (f" ほか{len(items) - show}か所" if len(items) > show else ""))
    return out


# ---- 跳躍のあとの長い伸ばし（旋律だけで分かる部分 ＋ 歌の崩しとの重なり）-------------------------------
#: 「大きな跳躍」の目安（半音・完全5度以上）。2026-10-04 本人の実機確認（22秒の「だ」・E4→C#5 の9半音・2拍・fall 2）
LEAP = 7
#: 「長く伸ばす」の目安（tick・1.5拍）
LEAP_HOLD = Q * 3 // 2
#: 直前の音とのあいだがこれより空いたら、跳躍に数えない（休みをはさんだ別のひと息）
LEAP_GAP = Q


def leap_holds(notes, leap: int = LEAP, hold: int = LEAP_HOLD, gap: int = LEAP_GAP) -> list[tuple[int, int, int, int, int]]:
    """**直前の音から `leap` 半音以上跳んで、`hold` 以上伸ばす音**。`notes` は (開始tick, 終わりtick, 高さ) の並び。
    戻り値は (歌の音の番号〈並べ替えたあと〉, 頭 tick, 跳躍〈符号つき半音〉, 長さ tick, 高さ)。
    直前の音とのあいだが `gap` より空いたら数えない（休みをはさむ＝ひと息が別）。長さはその音だけ（後ろの伸ばしの音は足さない）。"""
    ns = sorted(notes)
    out = []
    for i in range(1, len(ns)):
        a, z, k = ns[i][:3]
        pa, pz, pk = ns[i - 1][:3]
        if a - pz > gap or abs(k - pk) < leap or z - a < hold:
            continue
        out.append((i, a, k - pk, z - a, k))
    return out


# ---- 声部進行（主旋律を超える・交差・平行5度8度の連続）----------------------------------------

class _Line:
    """1本の声（主旋律か声部）。ある時刻に鳴っている高さを引く。"""

    def __init__(self, name: str, notes: list[Note3]):
        self.name = name
        self.notes = sorted(notes)
        self.starts = [n[0] for n in self.notes]
        self.ends = {n[1] for n in self.notes}

    def at(self, t: int) -> int | None:
        """`t` に鳴っている高さ（いちばん後に始まった音。無ければ None）。声は1本1音なので、さかのぼるのは数音まで。"""
        i = bisect_right(self.starts, t) - 1
        for a, z, k in reversed(self.notes[max(0, i - 3):i + 1]):
            if a <= t < z:
                return k
        return None


def voice_leading(melody: list[Note3], voices: list[tuple[str, list[Note3]]], spell, melody_name: str = "歌"):
    """**声部進行の事実**（声部進行の基本: 上の声が主旋律を超えない・声部が交差しない・平行5度と平行8度を連続させない）。

    戻り値は (超えた所, 交差した所, 平行5度・8度の所)。それぞれ (tick, 説明) の並び。
    - 超えた所: 声部の音の頭で、そのとき鳴っている主旋律より高い（上ハモは主旋律の上に置く書き方なので、多いこと自体は事実でしかない）
    - 交差: 2本の声（声部どうし・声部と主旋律）の上下が入れ替わった所。どちらかの音の頭ごとに上下を見て、前と逆になったら数える
      （同じ高さは入れ替わりに数えない）
    - 平行5度・8度: 2本がどちらも動き、同じ向きに動いて、前も後も5度（または8度・その複音程）。同じ高さどうし（ユニゾン＝重ねて
      厚くする書き方）は数えない。2本のどちらかが鳴っていない時刻（音の終わりも見る）をはさんだら続きとみなさない
    """
    lines = [_Line(melody_name, melody)] + [_Line(n, ns) for n, ns in voices]
    above: list[tuple[int, str]] = []
    mel = lines[0]
    for ln in lines[1:]:
        for a, _z, k in ln.notes:
            m = mel.at(a)
            if m is not None and k > m:
                above.append((a, f"{ln.name} {spell(k)}＞{melody_name} {spell(m)}"))
    cross: list[tuple[int, str]] = []
    parallel: list[tuple[int, str]] = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x, y = lines[i], lines[j]
            if not x.notes or not y.notes:
                continue
            order = 0
            prev = None
            for t in sorted(set(x.starts) | set(y.starts) | x.ends | y.ends):   # 終わりも見る（休みをはさんだら続きとみなさない）
                p, q = x.at(t), y.at(t)
                if p is None or q is None:
                    prev = None
                    continue
                s = (p > q) - (p < q)
                if s and order and s != order:
                    cross.append((t, f"{x.name}と{y.name}（{spell(p)}・{spell(q)}）"))
                if s:
                    order = s
                if prev is not None:
                    p0, q0 = prev
                    dp, dq = p - p0, q - q0
                    if dp and dq and (dp > 0) == (dq > 0):
                        iv0, iv1 = abs(p0 - q0), abs(p - q)
                        if iv0 % 12 == iv1 % 12 == 7:
                            parallel.append((t, f"5度 {x.name}と{y.name}（{spell(p0)}・{spell(q0)}→{spell(p)}・{spell(q)}）"))
                        elif iv0 % 12 == iv1 % 12 == 0 and iv0 and iv1:
                            parallel.append((t, f"8度 {x.name}と{y.name}（{spell(p0)}・{spell(q0)}→{spell(p)}・{spell(q)}）"))
                prev = (p, q)
    return sorted(above), sorted(cross), sorted(parallel)


def by_section_text(items: list[tuple[int, str]], sec_at, where, order: list[str], show: int = 8) -> tuple[str, str]:
    """(tick, 説明) の並び → (区間ごとの数の1行, 頭から `show` か所の位置と説明。ほかの数も書く)。"""
    c = Counter(sec_at(t) for t, _d in items)
    per = "・".join(f"{k} {c[k]}" for k in order if c.get(k))
    pos = "・".join(f"{where(t)}（{sec_at(t)}）{d}" for t, d in items[:show]) + (f" ほか{len(items) - show}か所" if len(items) > show else "")
    return per, pos


# ---- 声部が主旋律に乗る割合（区間ごと）・歌の崩しの置き場所 -------------------------------------------

def voice_coverage(melody_starts: list[int], voice_starts: set[int], sec_at) -> dict[str, tuple[int, int]]:
    """区間 → (同じ位置に声部の音がある主旋律の音の数, 主旋律の音の数)。曲全体でなく区間ごと（サビだけに重ねる曲の単調さを見るため）。"""
    out: dict[str, list[int]] = {}
    for t in melody_starts:
        row = out.setdefault(sec_at(t), [0, 0])
        row[1] += 1
        row[0] += t in voice_starts
    return {k: (v[0], v[1]) for k, v in out.items()}


# ---- 声部の厚み（層・同時に鳴る声の数・いちばん低い声部・音の長さ）---------------------------------------
# 2026-10-04（本人の実機確認『夜のしじまに灯す声』・アカペラ）: 「後ろの支えが弱すぎる。スカスカ」。声部12本が主旋律とほぼ同じ帯域
#   （声部 A#3〜G5・主旋律 C4〜D#5）・同じ長さ（中央値1拍）に集まり、同時に鳴る声は2声が45%だった。仕上げ・最終の点検役は
#   「低域の重心が軽い」と書いたが、材料に層・声の数・長さが無く、戻させなかった。ここは数えるだけ（アカペラかどうかで出す・
#   出さないを切り替えない。声部があれば出す。読み方は点検役の指示書）
#: 声部の音の長さの区切り（tick）。1拍（4分音符）未満＝短い／1拍以上2拍未満＝1拍前後／2拍以上＝長い
SHORT_NOTE, HELD_NOTE = Q, Q * 2


def _segments(lines: list[list[Note3]]) -> list[tuple[int, int, list[int]]]:
    """時間を音の頭と終わりで区切り、区切りごとに (頭, 終わり, 鳴っている声ごとの高さ) を返す（声は1本につき1つまで数える）。"""
    ls = [_Line("", ns) for ns in lines if ns]
    cuts = sorted({t for ln in ls for a, z, _k in ln.notes for t in (a, z)})
    out = []
    for a, z in zip(cuts, cuts[1:]):
        ks = [k for ln in ls if (k := ln.at(a)) is not None]
        if ks:
            out.append((a, z, ks))
    return out


def _weighted_median(items: list[tuple[int, int]]) -> int:
    """(値, 重み) の並びの、重みつきの中央値。"""
    items = sorted(items)
    half, acc = sum(w for _v, w in items) / 2, 0
    for v, w in items:
        acc += w
        if acc >= half:
            return v
    return items[-1][0]


def voice_texture(melody: list[Note3], voices: list[list[Note3]], sec_at, spell) -> dict[str, str]:
    """**声部の厚み**の事実（区間ごと・`"全体"` に曲全体）。声部が無ければ空。

    - 層: 声部の音を主旋律の音域（曲全体の最低〜最高・鳴る高さ）と比べて、下／同じ帯域／上に分けた音の数と、
      その区間での中央の高さがそこにある声部の数（固定の音高の帯でなく、主旋律との相対）
    - 同時に鳴る声の数（歌も含む）の時間の割合: 何かが鳴っている時間のうち、1声・2声・3声・4声以上
    - いちばん低い声部: 声部が鳴っている時間の、声部のうちいちばん低い高さの最低・中央（時間の重み）・最高と、主旋律の最低音との差（半音）
    - 声部の音の長さ: 1拍未満・1〜2拍・2拍以上の割合（比べるため歌の割合も添える）
    主旋律が無い曲（インスト）では層と差を出さない。
    """
    voices = [v for v in voices if v]
    if not voices:
        return {}
    lo = min((k for _a, _z, k in melody), default=None)
    hi = max((k for _a, _z, k in melody), default=None)
    secs = list(dict.fromkeys(sec_at(a) for v in voices for a, _z, _k in v))
    out: dict[str, str] = {}

    def layer(k: int) -> int:
        return 0 if k < lo else 2 if k > hi else 1

    def pct(c: Counter, keys, total) -> str:
        return "・".join(f"{name} {100 * c[key] / total:.0f}%" for key, name in keys) if total else "なし"

    for sec in secs + ["全体"]:
        pick = (lambda t: True) if sec == "全体" else (lambda t, s=sec: sec_at(t) == s)  # noqa: E731
        vs = [[n for n in v if pick(n[0])] for v in voices]
        vs = [v for v in vs if v]
        if not vs:
            continue
        parts = []
        if lo is not None:
            notes_n, voice_n = Counter(), Counter()
            for v in vs:
                for _a, _z, k in v:
                    notes_n[layer(k)] += 1
                voice_n[layer(_weighted_median([(k, z - a) for a, z, k in v]))] += 1
            parts.append(f"声部の層（主旋律の音域 {spell(lo)}〜{spell(hi)} に対して）: "
                         + "・".join(f"{name} {voice_n[i]}声（{notes_n[i]}音）"
                                     for i, name in enumerate(("主旋律より下", "同じ帯域", "主旋律より上"))))
        mel = [n for n in melody if pick(n[0])]
        seg = [s for s in _segments([mel] + vs) if pick(s[0])]
        cnt: Counter = Counter()
        for a, z, ks in seg:
            cnt[min(len(ks), 4)] += z - a
        parts.append("同時に鳴る声の数（歌も含む・時間の割合）: "
                     + pct(cnt, ((1, "1声"), (2, "2声"), (3, "3声"), (4, "4声以上")), sum(cnt.values())))
        low = [(min(ks), z - a) for a, z, ks in _segments(vs) if pick(a)]
        if low:
            mn, md, mx = min(k for k, _w in low), _weighted_median(low), max(k for k, _w in low)
            parts.append(f"いちばん低い声部（声部が鳴っている時間）: 最低 {spell(mn)}・中央 {spell(md)}・最高 {spell(mx)}"
                         + (f"（主旋律の最低音 {spell(lo)} との差: 最低 {mn - lo:+d}・中央 {md - lo:+d}半音）" if lo is not None else ""))

        def lens(ns) -> tuple[Counter, int]:
            c = Counter(0 if z - a < SHORT_NOTE else 1 if z - a < HELD_NOTE else 2 for a, z, _k in ns)
            return c, len(ns)
        keys = ((0, "1拍未満"), (1, "1〜2拍"), (2, "2拍以上"))
        vc, vt = lens([n for v in vs for n in v])
        mc, mt = lens(mel)
        parts.append(f"声部の音の長さ: {pct(vc, keys, vt)}" + (f"（歌は {pct(mc, keys, mt)}）" if mt else ""))
        out[sec] = "／".join(parts)
    return out


def bend_placement(notes, phrase_ends: set[int], sec_at) -> dict[str, tuple[int, int]]:
    """歌の崩しの**置き場所**（理論: 装飾は伸ばす音・フレーズの終わりに付け、乱発しない）。
    区間 → (伸ばす音〈1拍以上〉かフレーズの終わりの音に付いた数, それ以外〈1拍未満でフレーズの途中・頭〉に付いた数)。"""
    out: dict[str, list[int]] = {}
    for i, n in enumerate(notes):
        if not getattr(n, "artic", ""):
            continue
        row = out.setdefault(sec_at(n.start_tick), [0, 0])
        row[0 if (n.duration_tick >= LONG_NOTE or i in phrase_ends) else 1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


# ---- アカペラの和声（区間の頭の数小節・拍の頭で鳴る高さ）-----------------------------------------

def harmony_grid(bar_from: int, bar_to: int, bar_ticks: int, beat_ticks: int, chords: list[tuple[int, int, str]],
                 lines: list[tuple[str, list[Note3]]], spell, bars: int = HARMONY_BARS) -> list[str]:
    """1区間ぶんの和声の行（小節ごと）: その小節のコードと、声ごとに**拍の頭で鳴っている高さ**（鳴っていなければ `-`）。
    頭から `bars` 小節まで（区間がそれより長ければ、切ったことを最後の行に書く）。"""
    rows = []
    ls = [_Line(n, ns) for n, ns in lines]
    last = min(bar_to, bar_from + bars - 1)
    for b in range(bar_from, last + 1):
        t0 = (b - 1) * bar_ticks
        cs = [name for a, z, name in chords if a < t0 + bar_ticks and t0 < z]
        heads = range(t0, t0 + bar_ticks, beat_ticks)
        parts = [f"{ln.name} " + " ".join(spell(k) if (k := ln.at(t)) is not None else "-" for t in heads) for ln in ls]
        rows.append(f"{b}小節 コード {'→'.join(cs) or 'なし'} ／ " + " ／ ".join(parts))
    if bar_to > last:
        rows.append(f"（この区間は {bar_to - bar_from + 1}小節。頭の{last - bar_from + 1}小節だけ載せた）")
    return rows


# ---- 見せる歌詞（区間ごとの頭の数行）---------------------------------------------------------------

def lyrics_by_section(song) -> dict[str, list[str]]:
    """区間 → その区間の**見せる歌詞**の行（詞のフレーズ i ＝旋律のフレーズ i。区間は旋律のフレーズの `section`）。
    詞か旋律が無い・フレーズの数が合わないときは空（材料は曲名の材料の見せる歌詞へ落ちる）。"""
    ly, mel = getattr(song, "lyrics", None), getattr(song, "melody", None)
    lph, mph = list(getattr(ly, "phrases", None) or []), list(getattr(mel, "phrases", None) or [])
    if not lph or len(lph) != len(mph):
        return {}
    out: dict[str, list[str]] = {}
    for p in sorted(lph, key=lambda p: p.index):
        if not 1 <= p.index <= len(mph):
            return {}
        line = " ".join(((getattr(p, "display", "") or "").strip() or "").split())
        if line:
            out.setdefault(mph[p.index - 1].section, []).append(line)
    return out


def _kana(s: str) -> str:
    return re.sub(r"[^ぁ-ゖァ-ヺー]", "", s or "")


def display_matches_sung(song, notes) -> bool:
    """詞の工程の詞（`text`＝歌う読み）が、いまの楽譜の歌の文字と一致するか（かなだけを比べる）。

    ★**手直しのあとは食い違う**（2026-10-11）。詞の工程の出力は曲を作ったときのままで、編集の道具（区間の置き換えなど）は楽譜の歌だけを
      書き換える。『てのひらのシグナル』v16 は詞の工程の詞が v7 のころのまま（「はじめてのこえを」）で、実際は「ねえきこえる」を歌っていた——
      点検役は歌っていない詞を読んでいた。"""
    lph = list(getattr(getattr(song, "lyrics", None), "phrases", None) or [])
    if not lph or not notes:
        return False
    return _kana("".join(p.text for p in sorted(lph, key=lambda p: p.index))) == _kana("".join(n.lyric or "" for n in notes))


def sung_by_section(notes, sec_at) -> dict[str, list[str]]:
    """区間 → その区間で**実際に歌う詞**（かな）の行（フレーズ＝1行・`vocal_expression.phrases` の切り方・区間はフレーズの頭の音で決める）。"""
    import vocal_expression
    srt = sorted(notes, key=lambda n: n.start_tick)
    out: dict[str, list[str]] = {}
    for s, e in vocal_expression.phrases(srt):
        line = "".join(n.lyric or "" for n in srt[s:e])
        if line:
            out.setdefault(sec_at(srt[s].start_tick), []).append(line)
    return out


def lyric_head(lines: list[str]) -> str:
    """区間の見せる歌詞を頭から `LYRIC_LINES` 行・`LYRIC_CHARS` 字まで（切ったら書く）。"""
    shown = "／".join(lines[:LYRIC_LINES])
    cut = len(shown) > LYRIC_CHARS
    rest = len(lines) - LYRIC_LINES
    return (shown[:LYRIC_CHARS] + ("…" if cut else "")
            + (f"（ほか{rest}行）" if rest > 0 else "（以下略）" if cut else ""))


#: **声にならない（息の音だけになる）声の型**。V6 の MIKU Whisper は AIR=+100 で、AIR は +64 を超えると声が失われ
#: +100 で完全に無声（ヤマハの説明）。実測でも無声（2026-10-05『夜の底の呼吸』: D#3 を V6 で書き出すと Whisper は
#: −76dBFS・4kHz 以上が89%・スペクトル平坦度 0.81＝息の音だけ。Default は −41dBFS の声。`docs/plan/song_acappella_instrumental.md` §14-2）。
#: 高さによらないので、境の高さは持たない——音程のある声部に使われていれば、その声部と音数を事実として出す
WHISPER = "MIKU Whisper"


def voice_styles(names: list[str], styles: list[tuple[str, bool]], voices: list[list[Note3]],
                 sec_at, spell) -> dict[str, str]:
    """**声部ごとの声の型と、その声部の音の高さ**（区間ごと・`"全体"` に曲全体）。声部が無ければ空。

    `styles` … 声部ごとに (声の型, 全体の歌い方から引いたか)。声部の欄（`VoiceTrack.voice_style`）が空なら、
    呼ぶ側が編成の `harmony_style` を入れて True にする（それも無ければ空の文字＝既定の歌い方）。
    `MIKU Whisper`（声にならず息の音だけ）の声部すべてと、その音数・音域を添える。事実だけ（断じない）。
    """
    out: dict[str, str] = {}
    secs = list(dict.fromkeys(sec_at(a) for v in voices for a, _z, _k in v))
    for sec in secs + ["全体"]:
        pick = (lambda t: True) if sec == "全体" else (lambda t, s=sec: sec_at(t) == s)  # noqa: E731
        each, whisper = [], []
        for name, (style, inherited), v in zip(names, styles, voices):
            ks = [k for a, _z, k in v if pick(a)]
            if not ks:
                continue
            label = (style or "既定の歌い方") + ("（全体の歌い方）" if inherited else "")
            each.append(f"{name} {label} {spell(min(ks))}〜{spell(max(ks))}")
            if style == WHISPER:
                whisper.append(f"{name} {len(ks)}音")
        if not each:
            continue
        out[sec] = ("・".join(each)
                    + f"／うち {WHISPER} の声部: "
                    + (f"{len(whisper)}本（" + "・".join(whisper) + "）" if whisper else "なし"))
    return out
