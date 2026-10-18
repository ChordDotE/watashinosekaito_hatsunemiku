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

"""**前の稿と新しい稿の違いを、機械が並べる**（2026-10-04・組P2a・`docs/plan/song_final_reviewer.md`「追記: 監督者を人として見たとき」の3）。

## 前提・背景

点検役がやり直させた稿・差し戻した後の稿を見るとき、**実際にどこが変わったか**を渡していなかった。人の監督者なら、
出し直された作業は前の稿と見比べる（指示した所が変わったか・指示していない所が崩れていないか）。

## 決めたこと

- **機械は事実を並べるだけで、良し悪しを断じない**（変わった／変わらなかった、だけ）。
- 比べるのは工程の出力の dict（`model_dump`）。**単位は工程ごとに読みやすい単位**を、出力の形から決める（工程ごとの手書きの表は持たない——
  工程を足しても形が同じなら同じ単位で並ぶ）:
    - 区間名の欄（`section`・`kind`）を持つ要素の並び … 区間ごと。同じ区間に複数あれば**区間の中の何行目**（旋律の `phrases` ＝行）
    - 番号の欄（`index`・`no`）を持つ要素の並び … 番号ごと（詞のフレーズ `P3`・歌い方の `#3`）
    - 区間の並び（`sections: [...]`）を持つ要素の並び … 区間ごと（伴奏・彩りのパターン、ハモリの狙い）
    - パートの並び（`parts` で、要素が `patterns` を持つ）… パート（役と楽器番号）ごとに、パートの欄と区間ごとのパターン
    - ドラム（`drums` で `patterns` を持つ）… 区間ごとのパターン
    - それ以外の並び … 何個中何個が変わったか。値（文字・数など）… 欄の名前
- 見ない欄: 自己申告（`status`・`trouble`）・指摘への対応（`response`）・理由（`reason`）——理由は毎回書き直されるので、
  違いに混ぜると肝心の変化が埋もれる（対応と理由は点検役へ別に渡す）。旋律の `sketch`（ABC の記録）は `phrases` と同じ中身なので見ない。
- 長さに上限（`DIFF_LIMIT`）を付け、切ったら切ったと書く。
"""
from __future__ import annotations

from typing import Any

#: 違いに数えない欄（全工程）
IGNORED = frozenset({"status", "trouble", "response", "reason"})
#: 工程の出力名 → 違いに数えない欄（その工程だけ）
IGNORED_OF: dict[str, frozenset[str]] = {"melody": frozenset({"sketch"})}
#: 1工程ぶん・全体の上限（字）
DIFF_LIMIT = 1500
#: 区間名の欄・番号の欄（並びの要素がこれを持てば、その単位で比べる）
_SECTION_KEYS = ("section", "kind")
_NUMBER_KEYS = ("index", "no")
_NUMBER_MARK = {"index": "P", "no": "#"}


def as_dict(x: Any) -> dict | None:
    """工程の出力（型・保存から読んだ dict）→ dict。無い・走らなかったものは None。"""
    if x is None:
        return None
    d = x.model_dump() if hasattr(x, "model_dump") else x
    if not isinstance(d, dict) or d.get("status") == "skipped":
        return None
    return d


def _nums(ns: list[int]) -> str:
    """[1, 2, 3, 5] → 1〜3・5"""
    ns = sorted(set(ns))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if i == j else f"{ns[i]}〜{ns[j]}")
        i = j + 1
    return "・".join(out)


def _strip(d: Any, drop: tuple[str, ...]) -> Any:
    return {k: v for k, v in d.items() if k not in drop} if isinstance(d, dict) else d


def _dicts(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def _key_of(items: list, keys: tuple[str, ...]) -> str:
    return next((k for k in keys if all(k in x for x in items)), "")


def _by_section(field: str, b: list, a: list, key: str) -> tuple[list[str], list[str]]:
    """区間名の欄で比べる。同じ区間に複数あれば区間の中の何行目で（旋律の行）。"""
    order: list[str] = []
    gb: dict[str, list] = {}
    ga: dict[str, list] = {}
    for src, g in ((b, gb), (a, ga)):
        for x in src:
            name = str(x.get(key, ""))
            if name not in order:
                order.append(name)
            g.setdefault(name, []).append(_strip(x, (key,)))
    multi = any(len(v) > 1 for g in (gb, ga) for v in g.values())
    changed, same = [], []
    for name in order:
        xb, xa = gb.get(name, []), ga.get(name, [])
        if xb == xa:
            same.append(name)
            continue
        if not xb:
            changed.append(f"{name}（新しく入った）")
            continue
        if not xa:
            changed.append(f"{name}（無くなった）")
            continue
        if not multi:
            diff_fields = sorted(k for k in set(xb[0]) | set(xa[0]) if xb[0].get(k) != xa[0].get(k))
            changed.append(f"{name}（{'・'.join(diff_fields)}）")
            continue
        rows = [i + 1 for i in range(min(len(xb), len(xa))) if xb[i] != xa[i]]
        text = f"{name} の {_nums(rows)}行目" if rows else name
        if len(xb) != len(xa):
            text += f"（行数 {len(xb)}→{len(xa)}）"
        changed.append(text)
    return ([f"{field}: " + "／".join(changed)] if changed else []), ([f"{field}: " + "・".join(same)] if same else [])


def _by_number(field: str, b: list, a: list, key: str) -> tuple[list[str], list[str]]:
    mark = _NUMBER_MARK.get(key, "#")
    mb = {x.get(key): x for x in b}
    ma = {x.get(key): x for x in a}
    keys = [k for k in dict.fromkeys([*mb, *ma])]
    changed = [k for k in keys if mb.get(k) != ma.get(k) and isinstance(k, int)]
    same = [k for k in keys if mb.get(k) == ma.get(k) and isinstance(k, int)]
    out_c = [f"{field}: {mark}{_nums(changed)}（{len(changed)}/{len(keys)}）"] if changed else []
    out_s = [f"{field}: ほかの {len(same)} 個は同じ"] if same and changed else ([f"{field}: 全部（{len(same)} 個）"] if same else [])
    return out_c, out_s


def _per_section_items(items: list) -> dict[str, list]:
    """区間の並び（`sections`）を持つ要素 → 区間名 → その区間を受け持つ要素（`sections` を外したもの）の並び。"""
    out: dict[str, list] = {}
    for x in items:
        for s in (x.get("sections") or ["（区間なし）"]):
            out.setdefault(str(s), []).append(_strip(x, ("sections",)))
    return out


def _by_sections_list(field: str, b: list, a: list) -> tuple[list[str], list[str]]:
    sb, sa = _per_section_items(b), _per_section_items(a)
    names = list(dict.fromkeys([*sb, *sa]))
    changed = [n for n in names if sb.get(n) != sa.get(n)]
    same = [n for n in names if sb.get(n) == sa.get(n)]
    return ([f"{field}: 区間 " + "・".join(changed)] if changed else []), ([f"{field}: 区間 " + "・".join(same)] if same else [])


def _part_label(p: dict) -> str:
    role = p.get("role") or p.get("title") or p.get("name") or "パート"
    inst = p.get("instrument")
    return f"{role}（楽器{inst}）" if inst is not None else str(role)


def _parts(field: str, b: list, a: list) -> tuple[list[str], list[str]]:
    """パートの並び: パートごとに、パートの欄（音量など）と区間ごとのパターン。"""
    pb = {_part_label(p): p for p in b}
    pa = {_part_label(p): p for p in a}
    changed, same = [], []
    for name in dict.fromkeys([*pb, *pa]):
        xb, xa = pb.get(name), pa.get(name)
        if xb == xa:
            same.append(name)
        elif xb is None:
            changed.append(f"{name}（新しく入った）")
        elif xa is None:
            changed.append(f"{name}（無くなった）")
        else:
            bits = [k for k in sorted(set(xb) | set(xa)) if k != "patterns" and xb.get(k) != xa.get(k)]
            sb, sa = _per_section_items(xb.get("patterns") or []), _per_section_items(xa.get("patterns") or [])
            secs = [n for n in dict.fromkeys([*sb, *sa]) if sb.get(n) != sa.get(n)]
            what = ([f"パターン（区間 {'・'.join(secs)}）"] if secs else []) + bits
            changed.append(f"{name}: {'・'.join(what)}")
    return ([f"{field}: " + "／".join(changed)] if changed else []), ([f"{field}: " + "・".join(same)] if same else [])


def _compare(field: str, b: Any, a: Any) -> tuple[list[str], list[str], bool]:
    """1つの欄を比べる → (変わった所, 変わらなかった所, 単位で並べたか)。単位で並べない欄は呼び手が欄の名前で数える。"""
    if field in ("parts",) and (_dicts(b) or _dicts(a)) and all("patterns" in x for x in (b or []) + (a or [])):
        return (*_parts(field, b or [], a or []), True)
    if isinstance(b, dict) and isinstance(a, dict) and ("patterns" in b or "patterns" in a):
        c, s = _by_sections_list(f"{field}のパターン", b.get("patterns") or [], a.get("patterns") or [])
        bits = [k for k in sorted(set(b) | set(a)) if k != "patterns" and b.get(k) != a.get(k)]
        if bits:
            c = c + [f"{field}: {'・'.join(bits)}"]
        return c, s, True
    if _dicts(b) or _dicts(a):
        items = (b or []) + (a or [])
        if (k := _key_of(items, _NUMBER_KEYS)):
            return (*_by_number(field, b or [], a or [], k), True)
        if (k := _key_of(items, _SECTION_KEYS)):
            return (*_by_section(field, b or [], a or [], k), True)
        if all(isinstance(x.get("sections"), list) for x in items):
            return (*_by_sections_list(field, b or [], a or []), True)
        if b == a:
            return [], [f"{field}: 全部（{len(b or [])} 個）"], True
        lb, la = list(b or []), list(a or [])
        n = max(len(lb), len(la))
        diff = sum(1 for i in range(n) if (lb[i] if i < len(lb) else None) != (la[i] if i < len(la) else None))
        return [f"{field}: {n} 個中 {diff} 個" + (f"（数 {len(lb)}→{len(la)}）" if len(lb) != len(la) else "")], [], True
    return [], [], False


def diff_lines(key: str, before: Any, after: Any) -> tuple[list[str], list[str]]:
    """**前の稿と新しい稿の違い**（工程の出力名 `key`・型か dict）→ (変わった所, 変わらなかった所)。どちらかが無ければ両方空。"""
    b, a = as_dict(before), as_dict(after)
    if b is None or a is None:
        return [], []
    skip = IGNORED | IGNORED_OF.get(key, frozenset())
    changed, same, ch_fields, same_fields = [], [], [], []
    for f in dict.fromkeys([*b, *a]):
        if f in skip:
            continue
        c, s, unit = _compare(f, b.get(f), a.get(f))
        if unit:
            changed += c
            same += s
        elif b.get(f) == a.get(f):
            same_fields.append(f)
        else:
            ch_fields.append(f)
    if ch_fields:
        changed.append("欄: " + "・".join(ch_fields))
    if same_fields:
        same.append(f"ほかの欄（{len(same_fields)}個）")
    return changed, same


def diff_text(label: str, key: str, before: Any, after: Any, *, limit: int = DIFF_LIMIT) -> str:
    """1工程ぶんの違いの文（無ければ空）。`label` は工程名。上限 `limit` 字で切ったら書く。"""
    if as_dict(before) is None or as_dict(after) is None:
        return ""
    changed, same = diff_lines(key, before, after)
    lines = [f"■{label}"]
    lines.append("  変わった所: " + ("／".join(changed) if changed else "なし（前の稿と同じ）"))
    if same:
        lines.append("  変わらなかった所: " + "／".join(same))
    body = "\n".join(lines)
    if len(body) > limit:
        body = body[:limit] + f"…（以下略・全{len(body)}字を上限 {limit}字で切った）"
    return body


# ---- 前の版の楽譜といまの楽譜（編集のあとの最終の点検・2026-10-04・案C）-------------------------------------
#
# 追加編集（移調・ハモリ・歌の置き換え・伴奏の書き換え）のあと、最終の点検役に「前の版から変わった所」を渡す。工程の出力は変わらず
# 楽譜（と伴奏 MIDI）だけが変わるので、上の工程の出力どうしの比べ方は使えない。**区間は記録の時間軸**（`SectionTimeline`）で分ける。
# 事実を並べるだけで、良し悪しは書かない。伴奏 MIDI は `manual_diff.read_midi`（人の手直しを読む読み手）を使い回す。
# ★移調のように**全部の音が同じ幅だけ動いた**ときは、音ごとに並べず「全体を +3 半音移した」と1行で書く（同じことが何百行並ぶと埋もれる）

#: 楽譜の違いの上限（字・全体）
SCORE_DIFF_LIMIT = 6000
#: 区間ごとの行の上限（パート・声部ごと）
_ROWS_SHOWN = 24
NO_PREVIOUS = "前の版なし（この曲には前の版の楽譜が無い——今回が最初の版）"


def _sec_of(timeline):
    return (lambda t: timeline.at(int(t)) or "（区間の外）") if timeline is not None else (lambda t: "（区間不明）")


def _order(timeline) -> list[str]:
    return [s.kind for s in (getattr(timeline, "spans", None) or [])]


def uniform_shift(before: list[tuple], after: list[tuple]) -> int:
    """音の並び `(開始, 長さ, 高さ, ほか…)` が、高さだけ**全部同じ幅**で動いたならその幅（同じなら 0・形が違えば None）。"""
    if len(before) != len(after) or not before:
        return 0 if before == after else None
    b = sorted(before, key=lambda x: (x[0], x[1], x[2]))
    a = sorted(after, key=lambda x: (x[0], x[1], x[2]))
    ds = {y[2] - x[2] for x, y in zip(b, a)}
    if len(ds) != 1 or any((x[0], x[1], *x[3:]) != (y[0], y[1], *y[3:]) for x, y in zip(b, a)):
        return None
    return ds.pop()


def _score_by_section(before: list[tuple], after: list[tuple], sec) -> dict[str, tuple[int, int, int, int]]:
    """区間 → (前の音の数, いまの音の数, 消えた音, 増えた音)。変わらなかった区間は入れない（音は `(開始, 長さ, 高さ, ほか…)`）。"""
    from collections import Counter
    b, a = Counter(before), Counter(after)
    gone, new = b - a, a - b
    out: dict[str, list[int]] = {}
    for n in before:
        out.setdefault(sec(n[0]), [0, 0, 0, 0])[0] += 1
    for n in after:
        out.setdefault(sec(n[0]), [0, 0, 0, 0])[1] += 1
    for n in gone.elements():
        out[sec(n[0])][2] += 1
    for n in new.elements():
        out[sec(n[0])][3] += 1
    return {k: tuple(v) for k, v in out.items() if v[2] or v[3]}


def _section_rows(label: str, before: list[tuple], after: list[tuple], sec, order: list[str]) -> list[str]:
    """1つのパート（歌・声部・伴奏のトラック）の、区間ごとの違いの行（変わらなければ空）。"""
    k = uniform_shift(before, after)
    if k == 0:
        return []
    if k is not None:
        return [f"{label}: 全体を {k:+d} 半音移した（音の位置・長さは同じ・{len(after)}音）"]
    per = _score_by_section(before, after, sec)
    keys = sorted(per, key=lambda s: order.index(s) if s in order else len(order))
    rows = [f"{s}: {b}音 → {a}音（消えた音 {g}・増えた音 {n}）" for s, (b, a, g, n) in ((s, per[s]) for s in keys)]
    return [f"{label}: " + "／".join(rows[:_ROWS_SHOWN]) + (f" ほか{len(rows) - _ROWS_SHOWN}区間" if len(rows) > _ROWS_SHOWN else "")]


def _notes(xs, lyric: bool = False) -> list[tuple]:
    return [(int(n.start_tick), int(n.duration_tick), int(n.key)) + ((getattr(n, "lyric", "") or "",) if lyric else ())
            for n in (xs or [])]


def _midi_rows(before_mid, after_mid, sec, order: list[str]) -> list[str]:
    """伴奏 MIDI のトラック（楽器の名前）ごとの違い。どちらかの MIDI が無ければ空。"""
    from pathlib import Path
    if not before_mid or not after_mid or not Path(before_mid).exists() or not Path(after_mid).exists():
        return []
    from manual_diff import read_midi
    (b, _tb), (a, _ta) = read_midi(Path(before_mid)), read_midi(Path(after_mid))
    rows = []
    for name in sorted(set(b) | set(a)):
        bn = [(t, d, k, v) for t, k, d, v in b.get(name, [])]
        an = [(t, d, k, v) for t, k, d, v in a.get(name, [])]
        if name not in a:
            rows.append(f"伴奏『{name}』: トラックが消えた（{len(bn)}音）")
        elif name not in b:
            secs = sorted({sec(n[0]) for n in an}, key=lambda s: order.index(s) if s in order else len(order))
            rows.append(f"伴奏『{name}』: トラックが増えた（{len(an)}音・区間 {'・'.join(secs)}）")
        else:
            rows += _section_rows(f"伴奏『{name}』", bn, an, sec, order)
    return rows


def score_changes(before, after, timeline, *, before_mid=None, after_mid=None, limit: int = SCORE_DIFF_LIMIT) -> str:
    """**前の版の楽譜といまの楽譜の違い**（区間ごと・事実だけ）。前の版が無ければ `NO_PREVIOUS`、違いが無ければ「変わった所なし」。

    比べるもの: 速さ・拍子・コード（同じ幅の移しは1行）・歌の音（区間ごとの数・消えた音・増えた音）・声部（増えた・消えた・区間ごと）・
    主旋律の楽器・伴奏の楽譜のパート（役と音色）・伴奏 MIDI（`before_mid`・`after_mid` があればトラックごと・区間ごと）。
    区間は記録の時間軸（`timeline`）。無ければ区間を分けずに数える。"""
    if before is None:
        return NO_PREVIOUS
    sec, order = _sec_of(timeline), _order(timeline)
    out: list[str] = []
    if float(before.tempo) != float(after.tempo):
        out.append(f"速さ: {float(before.tempo):g} → {float(after.tempo):g}")
    if (before.time_signature or "4/4") != (after.time_signature or "4/4"):
        out.append(f"拍子: {before.time_signature} → {after.time_signature}")
    bc = [(c.start_tick, c.duration_tick, c.chord) for c in (before.chords or [])]
    ac = [(c.start_tick, c.duration_tick, c.chord) for c in (after.chords or [])]
    if bc != ac:
        changed = sum(1 for x, y in zip(bc, ac) if x != y) + abs(len(bc) - len(ac))
        out.append(f"コード: {len(bc)}個 → {len(ac)}個（違う所 {changed}）——移調ならコード名も一緒に動く")
    out += _section_rows("歌", _notes(before.vocal_notes, True), _notes(after.vocal_notes, True), sec, order)
    bv = {v.title or "声部": v for v in (before.extra_voices or [])}
    av = {v.title or "声部": v for v in (after.extra_voices or [])}
    for t in bv.keys() - av.keys():
        out.append(f"声部『{t}』: 消えた（{len(bv[t].notes)}音）")
    for t in [x for x in av if x not in bv]:
        secs = sorted({sec(n.start_tick) for n in av[t].notes}, key=lambda s: order.index(s) if s in order else len(order))
        out.append(f"声部『{t}』: 増えた（{len(av[t].notes)}音・音量 {float(av[t].gain_db):+g}dB・区間 {'・'.join(secs) or 'なし'}）")
    for t in [x for x in av if x in bv]:
        out += _section_rows(f"声部『{t}』", _notes(bv[t].notes), _notes(av[t].notes), sec, order)
        if float(bv[t].gain_db) != float(av[t].gain_db):
            out.append(f"声部『{t}』: 音量 {float(bv[t].gain_db):+g}dB → {float(av[t].gain_db):+g}dB")
    out += _section_rows("主旋律の楽器", _notes(before.lead.notes), _notes(after.lead.notes), sec, order)
    bp = [(p.role, p.timbre) for p in before.arrangement.parts]
    ap = [(p.role, p.timbre) for p in after.arrangement.parts]
    if bp != ap:
        from collections import Counter
        gone, new = Counter(bp) - Counter(ap), Counter(ap) - Counter(bp)
        out.append("伴奏の楽譜のパート: " + "／".join(
            [f"消えた {r}（{t}）" for r, t in gone.elements()] + [f"増えた {r}（{t}）" for r, t in new.elements()]
            or ["並びだけ変わった"]))
    out += _midi_rows(before_mid, after_mid, sec, order)
    if not out:
        return "変わった所なし（前の版の楽譜・伴奏 MIDI と同じ）"
    body = "\n".join(f"  ・{x}" for x in out)
    if len(body) > limit:
        body = body[:limit] + f"…（以下略・全{len(body)}字を上限 {limit}字で切った）"
    return body
