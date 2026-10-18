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

"""**版を確定するとき、前の版との差をこの回の頼みと照らして並べる**（2026-10-11・本人承認）。

## 前提・背景

『てのひらのシグナル』第1〜16版の振り返りで、頼んでいないのに変わった所・頼んだのに変わっていない所が、報告に載らないまま
何度も起きた（v15 で v14 の伴奏の手直しが黙って v3 に戻り、11〜12小節の歌が古いリズムに戻った／「ねえ、聴こえる？」が2回消えた／
画面に出す歌詞が v7 のまま）。手直しのあとの点検役には前の版との差（`song_draft_diff.score_changes`）を渡していたが、
呼ばれるのは大きい編集を書き出す時だけで、最終報告には載らなかった。

## 決めたこと

- **版を確定したあと、コーディネーターの実行の最後に1回**（`coordinator.run`）。前の版（編集の元にした版）と新しい版の楽譜・
  伴奏 MIDI・音量の設定・歌詞を比べ、この回に台帳へ積んだ頼み（`record_v{N}.json` の `requests` の差分）の範囲と照らす。
- **直さない・止めない**。最終報告の後ろに足すだけ（作業記録・`_last_run` の報告・ミクへの申し送りに載る）。
- 範囲（`scope`）の書かれていない頼みがあれば、「範囲の外で変わった」は判断しない（曲全体への頼みかもしれない）。

触れる軸: 継続性（頼みが黙って消えない）。費用は掛からない（LLM を呼ばない）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

#: 足す文の上限（字）
LIMIT = 1800
_SCOPE = re.compile(r"(\d+):([\d.]+)\s*[〜~-]\s*(\d+):([\d.]+)")


def _load_score(song_dir: Path, v: int):
    from song_schema import SongScore
    p = song_dir / f"score_v{v}.json"
    return SongScore.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def _requests(song_dir: Path, v: int) -> list[dict]:
    try:
        return json.loads((song_dir / f"record_v{v}.json").read_text(encoding="utf-8")).get("requests") or []
    except Exception:
        return []


def _bars_of_scope(scope: str) -> tuple[int, int] | None:
    m = _SCOPE.search(scope or "")
    if not m:
        return None
    b0, b1, beat1 = int(m.group(1)), int(m.group(3)), float(m.group(4))
    return b0, (b1 - 1 if beat1 == 0 else b1)


def _changed_sections(before, after, timeline, mid0: Path, mid1: Path) -> dict[str, list[str]]:
    """区間 → 変わったもの（歌・伴奏のトラック名）。"""
    from song_draft_diff import _notes, _score_by_section, _sec_of
    sec = _sec_of(timeline)
    out: dict[str, list[str]] = {}
    for s in _score_by_section(_notes(before.vocal_notes, True), _notes(after.vocal_notes, True), sec):
        out.setdefault(s, []).append("歌")
    if mid0.exists() and mid1.exists():
        from manual_diff import read_midi
        (b, _), (a, _) = read_midi(mid0), read_midi(mid1)
        for name in set(b) | set(a):
            bn = [(t, d, k, v) for t, k, d, v in b.get(name, [])]
            an = [(t, d, k, v) for t, k, d, v in a.get(name, [])]
            for s in _score_by_section(bn, an, sec):
                if f"伴奏『{name}』" not in out.setdefault(s, []):
                    out[s].append(f"伴奏『{name}』")
    return out


def _vocal_bars(before, after, bt: int) -> set[int]:
    """歌の音（開始・長さ・高さ・文字）が変わった小節。"""
    def by_bar(score):
        out: dict[int, set] = {}
        for n in score.vocal_notes or []:
            out.setdefault(n.start_tick // bt + 1, set()).add((n.start_tick, n.duration_tick, n.key, n.lyric or ""))
        return out
    b, a = by_bar(before), by_bar(after)
    return {k for k in set(b) | set(a) if b.get(k) != a.get(k)}


def _ranges(bars: list[int]) -> str:
    out, i = [], 0
    while i < len(bars):
        j = i
        while j + 1 < len(bars) and bars[j + 1] == bars[j] + 1:
            j += 1
        out.append(f"{bars[i]}" if i == j else f"{bars[i]}〜{bars[j]}")
        i = j + 1
    return "・".join(out) + "小節"


def _mix_rows(song_dir: Path, base: int, ver: int) -> list[str]:
    try:
        b = json.loads((song_dir / f"mix_v{base}.json").read_text(encoding="utf-8"))
        a = json.loads((song_dir / f"mix_v{ver}.json").read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = [f"{k}: {b.get(k)} → {a.get(k)}" for k in sorted(set(b) | set(a), key=str) if b.get(k) != a.get(k)]
    return [f"音量の設定: {'／'.join(rows[:6])}" + (f" ほか{len(rows) - 6}件" if len(rows) > 6 else "")] if rows else []


#: 頼みの文の言葉 → 変わったはずのもの（2026-10-14）。範囲の外の変化しか見ていなかったので、「コードも変えて」と頼まれて
#: コードが一度も変わらなかったこと（『てのひらのシグナル』v17〜27）が報告に載らなかった。言葉で拾う目安なので「確かめる」と書く
_ASKED_PARTS = (
    (("コード", "和音", "進行"), "コード", lambda b, a, mids: [(c.start_tick, c.duration_tick, c.chord) for c in (b.chords or [])]
     == [(c.start_tick, c.duration_tick, c.chord) for c in (a.chords or [])]),
    (("伴奏",), "伴奏", lambda b, a, mids: _same_file(*mids) and b.arrangement.model_dump() == a.arrangement.model_dump()),
    (("歌詞",), "歌う詞", lambda b, a, mids: (b.lyrics_full or "") == (a.lyrics_full or "")),
    (("字幕", "見せる歌詞", "表示用"), "見せる歌詞（字幕）", lambda b, a, mids: (b.lyrics_display or "") == (a.lyrics_display or "")),
    (("テンポ", "速さ", "BPM"), "速さ", lambda b, a, mids: float(b.tempo) == float(a.tempo)),
)


def _same_file(p: Path, q: Path) -> bool:
    try:
        return p.read_bytes() == q.read_bytes()
    except OSError:
        return True                                        # 比べられないときは「変わっていない」と言わない側に倒す


def _master_words(instruction: str) -> str:
    """依頼文からマスターの言葉の部分（【マスターの依頼】〜【ミクの受け取り方】の手前）。見つからなければ全体。"""
    t = instruction or ""
    if "【マスターの依頼】" in t:
        t = t.split("【マスターの依頼】", 1)[1]
    return t.split("【ミクの受け取り方", 1)[0]


def asked_but_unchanged(before, after, instruction: str, mids: tuple[Path, Path]) -> list[str]:
    """頼みの文に出てくるのに、前の版から変わっていないもの。"""
    words = _master_words(instruction)
    out = []
    for keys, label, same in _ASKED_PARTS:
        if any(k in words for k in keys):
            try:
                if same(before, after, mids):
                    out.append(f"頼みの文に『{next(k for k in keys if k in words)}』があるのに、{label}は前の版から変わっていない")
            except Exception:                              # noqa: BLE001（比べられなければ書かない）
                pass
    return out


def version_check_text(song_dir: Path, base: int, ver: int, instruction: str = "") -> str:
    """前の版 `base` → 新しい版 `ver` の差と、この回の頼みとの照らし合わせ（比べられなければ空）。"""
    if not base or ver <= base:
        return ""
    before, after = _load_score(song_dir, base), _load_score(song_dir, ver)
    if before is None or after is None:
        return ""
    from session import load_record
    from song_draft_diff import score_changes
    rec = load_record(song_dir, "", ver)
    timeline = getattr(rec, "timeline", None)
    mid0, mid1 = song_dir / f"accompaniment_v{base}.mid", song_dir / f"accompaniment_v{ver}.mid"
    lines = [f"\n【前の版（第{base}版）から第{ver}版で変わった所】（機械が数えた事実）",
             score_changes(before, after, timeline, before_mid=mid0, after_mid=mid1)]
    lines += [f"  ・{x}" for x in _mix_rows(song_dir, base, ver)]
    # 伴奏のパートの音量（楽器ごとにまとめる。パートは小節ごとに分かれていることが多い）
    import math
    ratio: dict[str, list[float]] = {}
    for q, p in zip(before.arrangement.parts, after.arrangement.parts):
        if (q.role, q.timbre) == (p.role, p.timbre) and float(q.gain) > 0 and abs(float(q.gain) - float(p.gain)) > 1e-6:
            ratio.setdefault(p.role, []).append(float(p.gain) / float(q.gain))
    if ratio:
        lines.append("  ・伴奏のパートの音量: " + "／".join(
            f"{role} {len(r)}か所 {20 * math.log10(sorted(r)[len(r) // 2]):+.1f}dB" for role, r in ratio.items()))
    if (before.lyrics_full or "") != (after.lyrics_full or ""):
        lines.append("  ・歌う詞（lyrics_full）が変わった"
                     + ("——**画面に出す歌詞（lyrics_display）は変わっていない**（歌と画面の歌詞が食い違う）"
                        if (before.lyrics_display or "") == (after.lyrics_display or "") else ""))
    # ---- この回の頼みと照らす（歌は小節単位・伴奏は区間単位）----
    old = {r.get("id") for r in _requests(song_dir, base)}
    new = [r for r in _requests(song_dir, ver) if r.get("id") not in old]
    if new and timeline is not None:
        from song_design import bar_ticks
        vb = _vocal_bars(before, after, bar_ticks(getattr(after, "time_signature", None) or "4/4"))
        changed = _changed_sections(before, after, timeline, mid0, mid1)
        spans = list(getattr(timeline, "spans", None) or [])
        asked_bars: set[int] = set()
        unscoped, missing = [], []
        for r in new:
            bars = _bars_of_scope(str(r.get("scope") or ""))
            words = " ".join(str(r.get("words") or "").split())
            words = (words.split(": ", 1)[0] if "@" in words and ": " in words else words)[:40]
            if bars is None:
                unscoped.append(f"「{words}」")
                continue
            rng = set(range(bars[0], bars[1] + 1))
            asked_bars |= rng
            secs = {s.kind for s in spans if s.bar_from <= bars[1] and bars[0] <= s.bar_to}
            if not (rng & vb) and not any(w != "歌" for s in secs for w in changed.get(s, [])):
                missing.append(f"「{words}」（{bars[0]}〜{bars[1]}小節）")
        lines.append("【この回の頼みと照らす】")
        out_v = sorted(vb - asked_bars)
        acc = {s: [w for w in ws if w != "歌"] for s, ws in changed.items()}
        acc = {s: ws for s, ws in acc.items() if ws}
        if out_v:
            lines.append(f"  ・**範囲つきの頼みの外で歌が変わった小節**: {_ranges(out_v)}")
        if acc:
            lines.append("  ・伴奏が変わった区間: " + "／".join(f"{s}（{'・'.join(w[:4])}）" for s, w in acc.items()))
        if unscoped:
            lines.append("  ・範囲の書かれていない頼み: " + "／".join(unscoped[:3])
                         + "——上の変化がこの頼みによるものか、頼んでいない変化かを確かめる")
        if not out_v and not acc and not unscoped:
            lines.append("  ・頼みの範囲の外で変わった所はない")
        if missing:
            lines.append("  ・**頼んだ範囲で変化が見つからない頼み**: " + "／".join(missing))
    unchanged = asked_but_unchanged(before, after, instruction, (mid0, mid1))
    if unchanged:
        if "【この回の頼みと照らす】" not in lines:
            lines.append("【この回の頼みと照らす】")
        lines += [f"  ・**{x}**（頼みどおりか確かめる）" for x in unchanged]
    text = "\n".join(x for x in lines if x)
    return text[:LIMIT] + ("…（以下略）" if len(text) > LIMIT else "")
