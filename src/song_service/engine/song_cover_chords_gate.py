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

"""**カバーのコードの中身の点検役**（2026-10-03 本人決定で J7 を変更・組K4。`song_cover_chords.py` から機能の境目で分けた）。

以前は点検役を置かず（J7）、推定と違う小節を返事に出して本人が聴いて判断していた。いまは**機械が数えた事実を材料にして**
点検役（`song_gate.judge`・新しい曲の点検役と同じ呼び出し口）が「通す／コードだけやり直す／所見つきで進む」を決める。
直し方を決めるのは点検役で、機械は推定に合わせて書き換えない（推定は外れる）。やり直しは1回まで。
指示書は `prompts/song_cover_chords_judge.md`（点検役の共通の指示書 `song_gate_judge.md` は変えず、依頼文の頭に置く）。

- 材料（`gate_findings`）: 推定と違う小節（曲の小節・区間の中の小節・LLM のコード・**書いたコードが推定の上位3に入っているか**
  〈入っていない小節を先に並べる〉・推定の上位3と確からしさ・その小節の旋律〈音ごとに拍の位置と長さ・前の小節から伸びる音〉）・
  推定が無ければ無いこと・Web の要約との和音の顔ぶれの違い。読むもの（`gate_material`）: 書いたコード（C 基準）・LLM の理由・
  Web の要約・マスターの頼み
- 判断の記録は `chords_gate`（伴奏の `arrange_gate` と同じ形）。返事の一文は `song_cover_chords._note_of`（`cover_skeleton["gate"]`）

import の向きは**このファイル → `song_cover_chords`**。`song_cover_chords.prepare_cover` は `review_chords` を関数の中で読む
（入口の名前を変えずに、ここをモジュールの読み込みの輪に入れないため）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from song_cover_chords import (CoverChords, _differ_bars, _guess_text, _melody_text, _triad_id,   # noqa: E402
                               _web_block, differing, place_chords_on, write_chords)
from song_cover_load import cover_bars                      # noqa: E402
from song_design import CHORD_BAR, key_shift, transpose_chord   # noqa: E402
from song_fix import cut_note                               # noqa: E402（切ったら書く・組F2）
from song_model import Song, StepRecord                     # noqa: E402
from song_prompts import prompt                             # noqa: E402

#: 点検役へ並べる「推定と違う小節」の上限（多すぎると肝心の行が埋もれる。超えた数は所見に書く）
GATE_ROWS = 48


def _web_gap(chords: CoverChords, web: dict) -> list[str]:
    """Web の要約と書いたコードの**和音の顔ぶれ**の違い（根音と長短で比べる。小節の位置は合っていない前提なので並びは比べない）。"""
    if not web.get("used"):
        return []

    def faces(names) -> dict:
        out: dict = {}
        for c in names:
            if (t := _triad_id(c)) is not None:
                out.setdefault(t, c)
        return out
    mine = faces(c for s in chords.sections for cell in s.chords_per_bar for c in CHORD_BAR.read(cell))
    sheet = faces(c for sec in web.get("sections") or [] for c in sec.get("コード（C基準）") or [])
    out = []
    if only := [c for t, c in sheet.items() if t not in mine]:
        out.append("Web の要約にあって、書いたコードに無い和音: " + " ".join(only))
    if only := [c for t, c in mine.items() if t not in sheet]:
        out.append("書いたコードにあって、Web の要約に無い和音: " + " ".join(only))
    return out


def gate_findings(song: Song, chords: CoverChords, mats: dict) -> list:
    """点検役へ渡す**機械が数えた事実**（`song_gate.Finding`）。推定と違う小節ごとに、LLM のコード・推定（上位3と確からしさ）・
    その小節の旋律。推定が無ければ無いこと。Web の要約があれば和音の顔ぶれの違い。良し悪しは書かない。"""
    from song_gate import Finding
    score = song.score
    _ts, bar, _n = cover_bars(score)
    sh = key_shift(song.design["key"])
    guesses = mats.get("midi") or mats.get("audio")
    rows, n_cmp, _miss = _differ_bars(song, chords, guesses)
    # ★**書いたコードが推定の上位3に入っているか**を添え、入っていない小節を先に並べる（2026-10-03 組G3）。比べるのは上位1つとの
    #   違い（`_differ_bars`）のままなので、確からしさがほぼ同じ2位を書いた小節も「違う」に入る。以前はその行で上限48小節が埋まり、
    #   本当に違う小節が埋もれた。並べ替えるだけで、どちらが正しいかは書かない
    def in_top3(b: int, cell: str) -> bool:
        mine = {_triad_id(c) for c in CHORD_BAR.read(cell)}
        return any(_triad_id(transpose_chord(n, -sh)) in mine for n, _p in (guesses[b].top or [])[:3])
    rows = sorted(((r, in_top3(r[0], r[3])) for r in rows), key=lambda x: x[1])   # 安定ソート（同じ側は曲の順のまま）
    out = []
    for (b, kind, i, cell, est), top3 in rows[:GATE_ROWS]:
        # 旋律は拍の位置と長さつき（`E4(1拍目・2拍)`・前の小節から伸びる音は「前から」・組G3）
        ev = [f"旋律 {_melody_text(score, b, bar, sh, beats=True)}"] + [
            f"{label}: {_guess_text(mats[k][b], sh)}" for k, label in (("midi", "伴奏MIDI"), ("audio", "音声")) if mats.get(k)]
        out.append(Finding(what=f"曲の{b + 1}小節目（{kind} の{i}小節目）: 書いたコード {cell}／推定 {est}"
                                f"（書いたコードは推定の上位3に{'入っている' if top3 else '入っていない'}）",
                           evidence=" ／ ".join(ev)))
    if len(rows) > GATE_ROWS:
        out.append(Finding(what=f"推定と違う小節はほかに {len(rows) - GATE_ROWS} 小節ある（並べたのは先頭の {GATE_ROWS} 小節）"))
    if not guesses:
        out.append(Finding(what="機械の推定が無い", evidence=f"原曲の伴奏 MIDI: {mats.get('midi_how', '')} ／ "
                                                          f"原曲の音声: {mats.get('audio_how', '')}"))
    elif not rows:
        out.append(Finding(what=f"機械の推定と違う小節は無い（比べた {n_cmp} 小節）"))
    return out + [Finding(what=w) for w in _web_gap(chords, mats.get("web") or {})]


def gate_material(chords: CoverChords, mats: dict, request: str = "") -> str:
    """点検役が自分の目で読むもの: 書いたコード（C 基準・区間ごと）・LLM の理由・Web の要約（あれば）・マスターの頼み（あれば）。"""
    lines = ([f"【マスターの頼み】{request}（コードはこの頼みに応えて書き直したもの。頼みで変えた小節は推定と違って当然）"]
             if request else [])
    lines += ["【書いたコード】（C 基準・区間ごと・1要素＝1小節）",
              json.dumps({s.kind: s.chords_per_bar for s in chords.sections}, ensure_ascii=False),
              f"【コードの LLM が書いた理由】（この曲を知っているか: {chords.known}）{cut_note(chords.reason or '', 1500)}"]
    return "\n".join(lines + _web_block(mats.get("web") or {}))


def gate_chords(song: Song, chords: CoverChords, mats: dict, *, request: str = "", attempt: int = 0,
                before: CoverChords | None = None, verbose: bool = True):
    """**書いたコードを点検役に1回通す**（`song_gate.judge`・ラベル `関門`＝費用は `song_llm.run_sync` に乗る）。
    前の工程へは戻らない（`redo_from_options=[]`）。やり直しは1回まで——やり直したあと（`attempt=1`）は通すか所見つきで進むだけ。
    判断は `chords_gate` の記録に残す（伴奏の点検役 `arrange_gate` と同じ形）。
    `before` … やり直す前のコード（2026-10-04 組F4）。やり直したあとの点検役に、前回の自分の判断・コードの LLM の対応（`response`）・
    変わった所（`song_draft_diff`）が入る（`song_gate_brief.gate_brief` のカバーの枝）。"""
    import song_context as sc                               # 点検役の呼び出し口（再生が差し替える）
    sh = key_shift(song.design["key"])
    _ts, _bar, n_bars = cover_bars(song.score)
    ctx_line = (prompt("song_cover_chords_judge")
                + f"\n\nカバー曲『{song.score.title}』／旋律と歌詞は原曲のまま／調 {song.design['key']}（原曲の旋律から推定）・"
                  f"C 基準へは {-sh:+d}半音／全{n_bars}小節")
    keep, sc.CURRENT_SONG = sc.CURRENT_SONG, song          # 点検役の答えが型を通らず止まるときの記録の置き場
    try:
        v = sc.judge("コード", f"カバーのコードを全小節書いた（この曲を知っているか: {chords.known}）",
                     chords.status, chords.trouble, gate_findings(song, chords, mats), attempt=attempt, verbose=verbose,
                     material=gate_material(chords, mats, request), context=ctx_line,
                     redo_from_options=[], can_redo=attempt == 0, before=before, after=chords)
    finally:
        sc.CURRENT_SONG = keep
    song.record(StepRecord(step="chords_gate", reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                           status="ok" if v.action == "accept" else "compromised", request=v.action))
    return v


def _review_text(v, differ: list[str]) -> str:
    """点検役の差し戻しを、コードの LLM への依頼文の1節にする（違う小節と理由を添える）。
    ★指摘ごとの対応（`response`）も頼む（2026-10-04 組F4。新しい曲の自工程のやり直しと同じ `RESPOND_ASK`）。やり直したあとの点検役が、
    対応と前のコードからの実際の変化を見比べる。"""
    from song_gate_brief import RESPOND_ASK                # noqa: E402
    return ("【点検役の差し戻し】前回のコードを点検役が読んで、直してほしいと言った所です（直す所以外は前回のまま残す）。"
            f"\n{v.guidance or v.reason}"
            + (f"\n（点検役が見た、機械の推定と違う小節: {' ／ '.join(differ[:GATE_ROWS])}）" if differ else "")
            + f"\n{RESPOND_ASK}")


def review_chords(song: Song, out: CoverChords, mats: dict, *, request: str = "", requests=None,
                  verbose: bool = True) -> tuple[CoverChords, object, dict]:
    """**点検役に通し、「やり直し」ならコードだけ1回書き直させる**（前回のコードを土台に、点検役の指示と推定と違う小節を添える）。
    戻り値は (使うコード, 最後の判断, 記録に残す `cover_skeleton["gate"]`)。書き直したコードは曲に載せ直してある。"""
    guesses = mats.get("midi") or mats.get("audio")
    v = gate_chords(song, out, mats, request=request, verbose=verbose)
    first = None
    if v.action == "redo":
        first = {"action": v.action, "reason": v.reason, "guidance": v.guidance, "notes": v.notes}
        if verbose:
            print(f"  [関門] コードを書き直す: {(v.guidance or v.reason)[:100]}")
        for r in song.history:                             # 捨てた案の直しの記録を、いまのコードのものとして残さない（伴奏と同じ）
            if r.step in ("chords_fix", "chords_machine"):
                r.step = "chords_discarded"
        before = out
        out = write_chords(song, mats, request=request, prev=out, requests=requests, verbose=verbose,
                           review=_review_text(v, differing(song, out, guesses)[0]))
        place_chords_on(song, out)
        v = gate_chords(song, out, mats, request=request, attempt=1, before=before, verbose=verbose)
    return out, v, {"action": v.action, "reason": v.reason, "notes": v.notes or "", "redone": first is not None,
                    "first": first}
