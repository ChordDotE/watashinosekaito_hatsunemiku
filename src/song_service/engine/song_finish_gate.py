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

"""**仕上げの点検役**——後ろの工程（曲名・歌い方・歌の崩し・ハモリ・声部・彩り・奏法）の内容をまとめて見る（2026-10-03・判-21 案A＋判-18。
曲名は同日に工程を足したときに加えた——曲名と理由を材料に載せ、詞・テーマと噛み合うかを見る）。

## 前提・背景

点検役がいるのは旋律・詞・伴奏の3工程だけで、**後ろの6工程は内容を誰も見ておらず、戻り道も無かった**
（`docs/plan/song_content_decisions.md` §3・`docs/plan/song_step_review.md` 判-21）。
アカペラでは伴奏の代わりに声部が和声とリズムの全部を担うのに、伴奏の点検役が無いので編成も声部も誰も見ていなかった。
本人方針（2026-10-03）「検査して、引っかかった点の修正判断を委ねる」に沿わせる。

## 決めたこと

- 曲の形で走る**最後の工程のあと**に1回だけ呼ぶ（`song_compose._compose_once_body` から1行）。
- 材料は**機械が数えた事実だけ**（数字の帯で断じない）。良し悪しは点検役（LLM・指示書 `prompts/song_finish_judge.md`）が決める:
  ・区間ごとに鳴っているもの（歌・歌い方〔歌の強さ・歌い回し〕・入り方・歌の崩し・声部・主旋律の楽器・楽器・彩り・奏法）
  ・歌い方の分布（歌の強さと歌い回しを別々に・2026-10-06）と、隣り合うフレーズの歌の強さの段差、区間の強さ
  ・声部の内容の所見（判-15。`harmony.voice_content_facts`＝担当 K2a の関数）と、後ろの工程の記録に残った所見（判-22）
  ・**判-18** 彩りと主旋律・声部の重なり（区間ごと）。組G4（同日）から**重なった長さ（拍）**で、声部とも高さが4半音以内だけ
  ・各工程の方針（reason）・ハモリの方針と計画・（アカペラ・インストでは）編成
  ・**組G4**（同日・`song_viewpoints_review.md` §1-5）: 材料の頭に**マスターの頼み**（2026-10-04 から点検役の呼び出し口が
    依頼文の頭に置く・`song_gate_brief`）、区間ごとの表に**見せる歌詞の頭の数行**・
    歌の崩しの**置き場所**・声部が乗る割合（**区間ごと**）、アカペラでは**区間ごとのコードと拍の頭の音**、声部があれば
    **声部進行**（主旋律を超えた所・交差・平行5度8度の連続）。数え方は `song_finish_facts`
- **差し戻せるのは後ろの工程だけ・1曲で1回まで**（`FINISH_MAX_REDO`）。既存の差し戻し（`song_flow._maybe_redo_from` →
  `song_compose._compose_song_rounds`）と、曲全体の上限（`song_gate.MAX_REDO_SONG`）に乗る。「この工程だけやり直す」は無い。
  戻り先は曲の形で決める（`finish_targets`）: 歌もの＝曲名と後ろの6工程／アカペラ＝曲名・歌い方・歌の崩し・ハモリ・声部と**編成**
  （伴奏の点検役が無いので、編成を戻せるのはここだけ）／インスト＝曲名・編成・彩り・奏法
- 再生（`song_replay`）では LLM を呼ばない——呼び口 `ctx.judge` を通すので、保存した判断（記録の `finish_gate`）を戻すか「通す」。
  続きから作る周で、この周に本物で作り直した工程があれば本物の点検役に見せる（`_live_if_remade`）。

- ★**後ろの工程の点検に専念させた**（2026-10-04・組P2b・`docs/plan/song_final_reviewer.md`「C」）。曲全体と頼みの照合
  （区間のつなぎ・頼みどおりか・アカペラの編成が頼みに合うか）は、2段目の**最終の点検役**（`song_final_gate`）の指示書へ移した。
  材料はそのまま（区間ごとの表は最終の点検役も `finish_material(policies=False)` で使い回す）。

触れる軸: リアルタイム性（曲づくりが1呼び出しぶん延びる）・公開性（1曲あたり約2%の費用）。
"""
from __future__ import annotations

from collections import Counter

from pydantic_ai import Agent

import song_context as ctx                                 # 呼び出し口（点検役＝再生が差し替える）
import song_flow as flow                                   # 差し戻し（`_maybe_redo_from`・打ち止め `NO_MORE_REDO`）
from song_gate import (FORM_STEPS, GATE_RECORD, STEP_ORDER, Finding, Verdict, note_redo_sent,
                       redo_sent)
from song_fix import cut_note                              # 切ったら「…（以下略・全N字）」（黙って切らない・組F2）
from song_llm import pick_model
from song_model import StepRecord
from song_prompts import prompt

#: 点検役の名前（`ctx.judge` の工程名・記録の名前は `song_gate.GATE_RECORD`）
FINISH = "仕上げ"
#: 記録の名前（`finish_gate`）
FINISH_RECORD = GATE_RECORD[FINISH]
#: 1曲のうちに仕上げの点検役が差し戻せる回数
FINISH_MAX_REDO = 1
#: 後ろの工程（どの形でも、走っていれば戻せる）。★曲名（2026-10-03）も入れる——曲名が詞・テーマと噛み合うかを見るのはここだけ
#:   （曲名を読む工程は無いので、戻しても曲名だけが作り直る）
FINISH_BACK = ("曲名", "歌い方", "歌の崩し", "ハモリ", "声部", "彩り", "奏法")
#: 工程名 → 記録の名前（後ろの工程と編成。申告の状態・困りごとを読む）
_RECORD_OF = {"曲名": "naming", "歌い方": "expression", "歌の崩し": "vocal_artic", "編成": "instrumentation",
              "ハモリ": "harmony", "声部": "voices", "彩り": "colour", "奏法": "articulation"}
#: 材料に載せる見せる歌詞の上限（字）。超えたら切ったことを書く（黙って切らない）。区間ごとの表に載せられないとき（詞と旋律の
#: フレーズが対応しない）だけ曲名の材料に出す
LYRICS_SHOWN = 1500
#: ※マスターの頼み（`_request_block`・`REQUEST_SHOWN`・`REQUEST_ROW`）は 2026-10-04（組P1）に `song_gate_brief` へ移した（全部の点検役へ）

_finish_judge: Agent[None, Verdict] = Agent(
    pick_model("COORD", "flash"), output_type=Verdict, system_prompt=prompt("song_finish_judge"))


def finish_targets(form: str) -> tuple[str, ...]:
    """**仕上げの点検役から差し戻せる工程**（作る順）。その形で走る後ろの工程と、歌もの以外では編成。

    ★編成: アカペラは伴奏の点検役が無く、編成（ハモリの方針・声の種類）を戻せるのはここだけ。インストも親の指示（2026-10-03）で戻せる。
    """
    runs = FORM_STEPS[form]
    extra = ("編成",) if form != "song" else ()
    return tuple(s for s in STEP_ORDER if s in runs and (s in FINISH_BACK or s in extra))


def last_step(form: str) -> str:
    """その形で走る最後の工程（点検役の答えが型を通らなかったとき、止める工程）。"""
    return next(s for s in reversed(STEP_ORDER) if s in FORM_STEPS[form])


def redo_options(form: str) -> list[str]:
    """いま実際に戻れる工程（打ち止め・この曲で既に1回差し戻したなら無し）。"""
    if flow.NO_MORE_REDO or redo_sent(FINISH) >= FINISH_MAX_REDO:
        return []
    return list(finish_targets(form))


# ---- 材料（機械が数えた事実だけ）--------------------------------------------------------------

def _spans(st) -> list:
    """設計の区間（曲頭の空き小節 `lead_in` を除く）を曲の順に。"""
    from song_timeline import LEAD_IN
    tl = getattr(st, "timeline", None)
    return [s for s in (getattr(tl, "spans", None) or []) if s.kind != LEAD_IN]


def _sec_at(st, tick: int) -> str:
    tl = getattr(st, "timeline", None)
    return tl.at(int(tick)) if tl is not None else ""


def _counts(c: Counter) -> str:
    return "・".join(f"{k} {n}" for k, n in c.most_common())


def _part_name(p) -> str:
    """伴奏の組み立てのパートの呼び名（役割と楽器名）。"""
    if getattr(p, "drum_steps", None):
        return "ドラム"
    from timbre_map import gm_name, parse_gm_bank
    _bank, prog = parse_gm_bank(getattr(p, "timbre", "") or "")
    return f"{p.role}（{gm_name(prog)}）" if prog is not None else (p.role or "?")


def _pitched_notes(score, spec) -> list[tuple[int, int, int, str]]:
    """組み立てた伴奏（彩り）の鳴る音を (開始tick, 終わりtick, 高さ, パート名) に（和音は1音ずつ）。"""
    from song_audio import pitched_events
    out: list[tuple[int, int, int, str]] = []
    for p in getattr(spec, "parts", None) or []:
        if p.drum_steps:
            continue
        name = _part_name(p)
        for on, dur, keys, _vel in pitched_events(score, p):
            out += [(on, on + dur, k, name) for k in keys]
    return out


def accomp_from_midi(mid_path, spec=None) -> list[tuple[int, int, int, str, bool]]:
    """**伴奏 MIDI（作業台の `accompaniment.mid`）の鳴る音**を (開始tick, 終わりtick, 高さ, パート名, ドラムか) に（2026-10-04）。

    編集のあとの最終の点検役（`song_final_gate.edit_state`）が、伴奏を MIDI で直接書き換える道具
    （`replace_accompaniment_region`・`remove_accompaniment_track`）を使った編集のあとに読む。その道具は MIDI だけを書き換え、
    楽譜の `arrangement` は変えないので、楽譜から数えると編集前の伴奏を見てしまう（実機『水たまりのプリズム』v2 で、
    5小節目のギターを1オクターブ下げたのに「歌と同じ高さで鳴る」と出て、誤った NG になった）。
    - tick は MIDI 自身の速さで秒から戻す（楽譜の原点 tick 0＝MIDI の頭。`score_to_midi` が同じ原点で書く）。
    - 主旋律の楽器のトラック（`midi_out.LEAD_TRACK`）は入れない（楽譜から数えるときも伴奏のパートに入っていない）。
    - パート名は楽譜の数え方と同じ呼び名（`役割（楽器名）`・ドラムは「ドラム」）。トラック名＝役割で、楽譜のパートに同じ役割が
      あればその呼び名を使う。無ければ（道具が足した新しいトラック）MIDI の音色から付ける。
    """
    import mido
    from accomp_edit import load_accomp
    from midi_out import LEAD_TRACK
    from song_finish_facts import Q
    from timbre_map import gm_name
    m = mido.MidiFile(mid_path)
    tempo = next((msg.tempo for tr in m.tracks for msg in tr if msg.type == "set_tempo"), 500000)
    spb = tempo / 1_000_000                                 # 1拍の秒（MIDI 自身の速さ）
    shown: dict[str, str] = {}
    for p in getattr(spec, "parts", None) or []:
        shown.setdefault(p.role, _part_name(p))
    tracks = load_accomp(mid_path)
    prog_of: dict[int, int] = {}                           # チャンネルの音色（区間ごとのトラックは頭の1本だけが音色を持つ）
    for t in tracks:
        prog_of.setdefault(t.channel, t.program)
    out: list[tuple[int, int, int, str, bool]] = []
    for t in tracks:
        if t.name == LEAD_TRACK:
            continue
        drum = t.is_drums
        name = "ドラム" if drum else shown.get(t.name) or f"{t.name}（{gm_name(prog_of.get(t.channel, t.program) + 1)}）"
        for n in t.notes:
            a = int(round(n.start_sec / spb * Q))
            out.append((a, max(a + 1, int(round((n.start_sec + n.dur_sec) / spb * Q))), n.key, name, drum))
    return out


def _colour_spec(st):
    """**彩りだけ**を組み立てたもの（彩りの無い曲は None）。骨組みを空にして `song_band.run_colour` と同じ組み立てに通す。"""
    from song_arrange import merge_colour, to_spec
    from song_design import grid_slots
    song = st.song
    colour, plan = getattr(song, "colour", None), getattr(song, "arrangement", None)
    if colour is None or plan is None or not colour.parts:
        return None
    base = plan.model_copy(update={"parts": [], "drums": None})   # 骨組みのリフは残す（彩りが `use_riff` で指すことがある）
    return to_spec(merge_colour(base, colour), st.bars_of, grid_slots(st.ts), st.inst_gms, st.allow_drums, None,
                   intensity_of=st.intensity_of, shift_of=st.shift_of, drum_kit="")


def colour_clashes(colour: list[tuple[int, int, int, str]], melody: list[tuple[int, int, int]],
                   voices: list[tuple[int, int, int]], sec_at) -> tuple[Counter, Counter, Counter]:
    """**判-18**: 彩りの音を区間ごとに数える（区間 → 数）。戻り値は (全部, 主旋律と同時に鳴って4半音以内, 声部と同時に鳴る)。

    同時に鳴る＝鳴っている時間が少しでも重なる。高さは鳴る高さ（移調後）。数えるだけで、濁るかどうかは断じない。
    """
    total, near, with_voice = Counter(), Counter(), Counter()
    for a, z, k, _name in colour:
        sec = sec_at(a)
        total[sec] += 1
        if any(ma < z and a < mz and abs(mk - k) <= 4 for ma, mz, mk in melody):
            near[sec] += 1
        if any(va < z and a < vz for va, vz, _vk in voices):
            with_voice[sec] += 1
    return total, near, with_voice


def _voice_findings(st) -> list[str]:
    """声部の内容の所見（判-15・推奨音域の外・曲の9割に乗る）。`harmony.voice_content_facts`（担当 K2a）で楽譜から取る。

    ★楽譜（`song.score`）は流れの最後（`song_compose._finish`）で曲に入るので、ここでは組み立て中の楽譜（`st.score`）を持たせて渡す。
    """
    import harmony
    from types import SimpleNamespace
    # 曲全体の「9割に乗る」は出さない——乗る割合は区間ごとの表に出す（組G4・`song_finish_facts.voice_coverage`）
    return list(harmony.voice_content_facts(SimpleNamespace(score=st.score, form=st.song.form), whole_song_share=False) or [])


def _record_findings(st) -> list[Finding]:
    """**後ろの工程の記録に残った所見**（`StepRecord.warnings`）。点検役のいない工程の所見（判-22）を、この点検役が材料として読む。

    工程ごとに最後の記録だけ。声部は楽譜から取り直す（`_voice_findings`）ので記録の所見は使わない（同じ所見が2つ並ぶ）。
    差分の直しの経過（`*_fix` の reason と同じ文）と、失敗の控え（トレースバック）は除く。
    ※要約（`song_model.FINDINGS_ROUTE`）では、ここで読む記録を `finish_gate` 宛てにすると二重に出ない。
    """
    song = st.song
    fixed = {r.reason for r in song.history if r.step.endswith("_fix")}
    out: list[Finding] = []
    for s in finish_targets(song.form):
        if s not in FINISH_BACK or s == "声部":
            continue
        rec = song.last(_RECORD_OF[s])
        for w in (rec.warnings if rec else []):
            if w in fixed or "Traceback" in w or 'File "' in w:
                continue
            out.append(Finding(what=f"{s}の記録の所見: {cut_note(' '.join(w.split()), 300)}"))
    return out


def _sound_choice_facts(st, notes, tracks, sec_at) -> list[Finding]:
    """**形式から外した音の判断**の材料（2026-10-03・`docs/plan/song_viewpoints_review.md` §2 d1〜d3）。事実だけ（断じない）:
    d1 歌の崩しが目安の5半音を超える音／d2 `しゃくる` の入りに `scoop` が二重に掛かった音／d3 主旋律より大きい（`gain_db` > 0）声部。
    以前は d1・d3 を型の上限（5半音・0dB）で、d2 を歌の崩しの形式の所見で差し戻していた。読み方は指示書 `prompts/song_finish_judge.md`。"""
    import vocal_artic
    import vocal_expression
    from artic_plan import tick_key
    from song_design import bar_ticks
    bt = bar_ticks(getattr(st.score, "time_signature", None) or "4/4")
    where = lambda idx: "・".join(f"{tick_key(notes[i].start_tick, bt)}（{sec_at(notes[i].start_tick)}）"   # noqa: E731
                                  for i in idx[:12]) + (f" ほか{len(idx) - 12}か所" if len(idx) > 12 else "")
    out: list[Finding] = []
    if notes:
        deep = vocal_artic.deep_bend_notes(notes)
        if deep:
            out.append(Finding(
                what=f"歌の崩しが目安（{vocal_artic.VOCAL_GUIDE_DEPTH:g}半音）を超える音: {len(deep)}か所",
                evidence="位置と深さ: " + "・".join(
                    f"{tick_key(notes[i].start_tick, bt)} {notes[i].artic} {notes[i].artic_depth if notes[i].artic_depth is not None else '標準'}半音"
                    for i in deep[:12]) + (f" ほか{len(deep) - 12}か所" if len(deep) > 12 else "")))
        entries = vocal_expression.entries_of_heads(notes, getattr(st.song, "expression", None))
        dbl = vocal_artic.double_scoop_notes(notes, entries)
        if dbl:
            out.append(Finding(
                what=f"歌い方の入り方『しゃくる』の音に歌の崩し scoop も掛かっている（二重のしゃくり）: {len(dbl)}か所",
                evidence="位置: " + where(dbl)))
    loud = [t for t in tracks if float(getattr(t, "gain_db", 0.0) or 0.0) > 0]
    if loud:
        out.append(Finding(
            what=f"主旋律（0dB）より大きい声部: {len(loud)}本",
            evidence="声部と音量: " + "・".join(f"{t.title or '声部'} {float(t.gain_db):+g}dB" for t in loud)))
    return out


def _expression_by_section(st) -> tuple[dict[str, Counter], dict[str, Counter], dict[str, Counter], Counter, Counter, int]:
    """歌い方（フレーズ単位）を区間ごとに数える。戻り値は (区間 → 強さの数, 区間 → 歌い回しの数, 区間 → 入り方の数,
    全体の強さ, 全体の歌い回し, 書かなかったフレーズ数)。強さ（音量）と歌い回し（語尾・揺れ）は 2026-10-06 から別の欄なので別々に数える。"""
    import vocal_expression
    plan = getattr(st.song, "expression", None)
    notes = list(st.notes or [])
    by_sec: dict[str, Counter] = {}
    how_sec: dict[str, Counter] = {}
    entry_sec: dict[str, Counter] = {}
    whole: Counter = Counter()
    how_all: Counter = Counter()
    if not notes or plan is None:
        return by_sec, how_sec, entry_sec, whole, how_all, 0
    segs = vocal_expression.phrases(notes)
    of = {ps.no: ps for ps in (plan.phrases or [])}
    unwritten = 0
    for i, (s, _e) in enumerate(segs, 1):
        sec = _sec_at(st, notes[s].start_tick)
        ps = of.get(i)
        if ps is None:
            unwritten += 1
            continue
        by_sec.setdefault(sec, Counter())[ps.strength] += 1
        whole[ps.strength] += 1
        if ps.how != "ふつう":
            how_sec.setdefault(sec, Counter())[ps.how] += 1
        how_all[ps.how] += 1
        if ps.entry and ps.entry != "ふつう":
            entry_sec.setdefault(sec, Counter())[ps.entry] += 1
    return by_sec, how_sec, entry_sec, whole, how_all, unwritten


def _strength_jump_text(st) -> str:
    """隣り合うフレーズの強さの段差（2026-10-06）。段差ごとの数と、2段以上の所の番号・名前。**事実だけ**（対比を狙うことがある）。"""
    import vocal_expression
    jumps = vocal_expression.strength_jumps(getattr(st.song, "expression", None))
    if not jumps:
        return ""
    by = Counter(abs(j) for *_x, j in jumps)
    out = "・".join(f"{k}段 {by[k]}" for k in sorted(by))
    big = [f"{a}→{b}（{x}→{y}）" for a, b, x, y, j in jumps if abs(j) >= 2]
    return out + (f"／2段以上: {'・'.join(big)}" if big else "")


def _cut_ending_text(st) -> str:
    """歌い回し `切って` を選んだフレーズごとの最後の音の長さ（2026-10-06）。長い順で、目安（`CUT_LONG_SEC`）以上に ★。**事実だけ**。
    発端は『火花になって飛べ』v3 フレーズ4（最後の「く」が2拍＝0.60秒で `切って`。語尾が消えて聞こえた）。"""
    import vocal_expression as ve
    tempo = getattr(st.score, "tempo", None)
    rows = ve.cut_endings(list(st.notes or []), getattr(st.song, "expression", None), tempo)
    if not rows:
        return ""
    over = sum(1 for *_x, s in rows if s is not None and s >= ve.CUT_LONG_SEC)
    items = [f"{no}「{ly}」{b:g}拍" + (f"・{s:.2f}秒" + ("★" if s >= ve.CUT_LONG_SEC else "") if s is not None else "")
             for no, ly, b, s in rows]
    return (f"{len(rows)}フレーズ（★＝目安の{ve.CUT_LONG_SEC:g}秒以上 {over}）: " + "・".join(items))


def _reason(obj, limit: int = 300) -> str:
    r = (getattr(obj, "reason", "") or "").replace("\n", " ").strip()
    return cut_note(r, limit)


def _title_block(song, score, lyrics_in_table: bool = False) -> str:
    """**曲名と理由**（2026-10-03・曲名の工程）と、噛み合いを見る相手（テーマの伝えたいこと・心の動き・決め台詞・見せる歌詞）。
    曲名の工程の出力が無ければ空。歌詞は楽譜の見せる歌詞（`SongScore.lyrics_display`）を `LYRICS_SHOWN` 字まで（切ったら書く）。
    `lyrics_in_table` … 見せる歌詞を区間ごとの表に載せたなら、ここでは二重に出さない（組G4）。"""
    from song_model import made
    nm = getattr(song, "naming", None)
    if not made(nm):
        return ""
    th = song.theme
    work = getattr(th, "working_title", "") or ""
    out = [f"【曲名】『{nm.title}』"
           + ("（テーマの仮の曲名のまま）" if nm.title == work else f"（テーマの仮の曲名『{work}』から付け直した）"),
           f"   理由: {_reason(nm, 400) or '—'}",
           f"   テーマの伝えたいこと: {getattr(th, 'message', '')}／心の動き: {getattr(th, 'arc', '')}"]
    lp = getattr(song, "lyric_plan", None)
    if made(lp) and getattr(lp, "hook", ""):
        out.append(f"   決め台詞（hook）: 「{lp.hook}」")
    shown = " ".join((getattr(score, "lyrics_display", "") or "").split())
    if shown and lyrics_in_table:
        out.append("   見せる歌詞: 区間ごとの表の『詞』の行（各区間の頭の数行）")
    elif shown:
        out.append("   見せる歌詞: " + shown[:LYRICS_SHOWN]
                   + (f"…（以下略・全{len(shown)}字）" if len(shown) > LYRICS_SHOWN else ""))
    return "\n".join(out)


def finish_material(st, *, policies: bool = True) -> tuple[str, list[Finding]]:
    """仕上げの点検役に読ませる**出来たもの**（区間ごとの表と各工程の方針）と、**機械が数えた所見**。

    `policies=False` … 区間ごとの表だけを返す（各工程の方針の節を付けない）。最終の点検役（`song_final_gate`・2026-10-04 組P2b）が
    曲全体を見る材料に、同じ表を使い回す（表の作り方を2か所に持たない）。"""
    from song_audio import bars_of_filter, song_bars
    song, score = st.song, st.score
    form = song.form
    runs = FORM_STEPS[form]
    spans = _spans(st)
    sec_at = lambda t: _sec_at(st, t)                      # noqa: E731
    inten = {s.kind: s.intensity for s in (getattr(song.design, "sections", None) or [])}
    notes = list(getattr(score, "vocal_notes", None) or [])
    tracks = list(getattr(score, "extra_voices", None) or [])
    lead = score.lead                                      # 無ければ中身が空（`LeadPart.is_empty`・F1）
    spec = score.arrangement                               # 無ければパートが0本（F1）

    # ---- 区間ごとに数える ----
    sung = Counter(sec_at(n.start_tick) for n in notes)
    bend = {}
    for n in notes:
        if getattr(n, "artic", ""):
            bend.setdefault(sec_at(n.start_tick), Counter())[n.artic] += 1
    voice_n: dict[str, Counter] = {}
    for t in tracks:
        for n in t.notes:
            voice_n.setdefault(sec_at(n.start_tick), Counter())[t.title or "声部"] += 1
    lead_n = Counter(sec_at(n.start_tick) for n in lead.notes)
    artic_n: dict[str, Counter] = {}
    for a in list(spec.articulations) + list(lead.articulations):
        artic_n.setdefault(sec_at(a.tick), Counter())[a.name] += 1
    expr_sec, how_sec, entry_sec, expr_all, how_all, unwritten = _expression_by_section(st)
    # ---- 組G4: 詞・歌の崩しの置き場所・声部の乗る割合（区間ごと）----
    import song_finish_facts as ff
    import vocal_expression
    from song_design import bar_ticks
    from song_table import spell_pitch
    bt = bar_ticks(getattr(score, "time_signature", None) or "4/4")
    order = [s.kind for s in spans]
    # ★**見せる歌詞が実際の歌と食い違うとき（手直しのあと）は、実際に歌う詞（かな）を載せる**（2026-10-11・`display_matches_sung`）
    sung_only = bool(notes) and not ff.display_matches_sung(song, notes)
    lyr = (ff.sung_by_section(notes, sec_at) if sung_only else ff.lyrics_by_section(song)) if notes else {}
    place = ff.bend_placement(notes, {e - 1 for _s, e in vocal_expression.phrases(notes)} if notes else set(), sec_at)
    mel3 = [(n.start_tick, n.start_tick + n.duration_tick, n.key) for n in notes]
    cover = ff.voice_coverage([n.start_tick for n in notes], {n.start_tick for t in tracks for n in t.notes}, sec_at) \
        if tracks and form != "acappella" else {}
    names = ff.voice_names(tracks)
    # 声部の厚み（層・同時に鳴る声の数・いちばん低い声部・音の長さ）。形によらず、声部があれば出す（2026-10-04『夜のしじまに灯す声』）
    texture = ff.voice_texture(mel3, [[(n.start_tick, n.start_tick + n.duration_tick, n.key) for n in t.notes] for t in tracks],
                               sec_at, spell_pitch) if tracks else {}
    # 声部ごとの声の型×音の高さ（2026-10-05『夜の底の呼吸』: MIKU Whisper の声部が声にならず息の音だけになった。
    # Whisper は AIR=+100＝高さによらず無声なので、Whisper の声部はすべて音数つきで出す）。
    # 声部の欄が空なら編成の `harmony_style`（楽譜の `voice_choice` → 編成の決定）。形によらず、声部があれば出す
    whole_style = ((getattr(score, "voice_choice", None) or {}).get("harmony_style", "")
                   or getattr(getattr(song, "instrumentation", None), "harmony_style", "") or "")
    styles = [((getattr(t, "voice_style", "") or whole_style), not getattr(t, "voice_style", "")) for t in tracks]
    vstyle = ff.voice_styles(names, styles, [[(n.start_tick, n.start_tick + n.duration_tick, n.key) for n in t.notes]
                                             for t in tracks], sec_at, spell_pitch) if tracks else {}
    # ---- 複数の工程にまたがる事実（2026-10-04・本人の実機確認『水たまりのプリズム』）----
    # 歌×伴奏: 歌と同じ高さ（音名・オクターブ込み）で同時に鳴る伴奏の音（8〜9秒のギターで歌声が二重に聞こえた）
    # 旋律×歌の崩し: 大きく跳んで長く伸ばす音と、そこに掛かった崩し（22秒の「だ」・9半音跳んで2拍・fall）
    from artic_plan import tick_key
    where_t = lambda t: tick_key(t, bt)                    # noqa: E731
    lyric_at = {n.start_tick: (n.lyric or "") for n in notes}.get
    # ★**伴奏の MIDI を直接書き換えた編集のあと**は、楽譜の `arrangement` ではなく実際の伴奏（作業台の MIDI）で数える
    #   （`st.accomp_midi`＝`accomp_from_midi` の結果・`song_final_gate.edit_state` が置く。新しい曲の流れでは置かない＝None）
    acc_midi = getattr(st, "accomp_midi", None)
    acc_pitched = [(a, z, k, nm) for a, z, k, nm, drum in acc_midi if not drum] if acc_midi is not None else None
    same_rows = (ff.same_pitch_rows(ff.same_pitch(mel3, acc_pitched if acc_pitched is not None else _pitched_notes(score, spec)),
                                    sec_at, where_t, spell_pitch, order, lyric_at=lyric_at)
                 if notes and "伴奏" in runs and (acc_midi is not None or spec.parts) else {})
    leap_rows: dict[str, list[str]] = {}
    srt = sorted(notes, key=lambda n: n.start_tick)
    for i, t, jump, dur, k in ff.leap_holds([(n.start_tick, n.start_tick + n.duration_tick, n.key) for n in srt]):
        n = srt[i]
        bend_txt = (f"・崩し {n.artic} {n.artic_depth:g}半音" if getattr(n, "artic", "") and n.artic_depth is not None
                    else f"・崩し {n.artic}" if getattr(n, "artic", "") else "・崩しなし")
        leap_rows.setdefault(sec_at(t), []).append(
            f"{where_t(t)}「{n.lyric or ''}」 {spell_pitch(srt[i - 1].key)}→{spell_pitch(k)}（{jump:+d}半音）{dur / ff.Q:g}拍{bend_txt}")

    # 楽器（伴奏と彩りを合わせた楽譜）がその区間で鳴るか
    inst_in: dict[str, list[str]] = {}
    if acc_midi is not None:                               # 編集のあと（MIDI を直接書き換えた）: 実際の伴奏で、鳴った区間を見る
        for a, _z, _k, name, _drum in acc_midi:            # トラックの順（＝楽譜のパートの順）のまま
            sec = sec_at(a)
            if name not in inst_in.setdefault(sec, []):
                inst_in[sec].append(name)
    elif spec.parts:
        from song_audio import pitched_events
        n_bars = song_bars(score)
        for p in spec.parts:
            name = _part_name(p)
            if p.drum_steps:
                bars = bars_of_filter(p.bar_filter, n_bars)
                secs = {s.kind for s in spans if any(s.bar_from <= b <= s.bar_to for b in bars)}
            else:
                secs = {sec_at(on) for on, _d, _k, _v in pitched_events(score, p)}
            for sec in secs:
                if name not in inst_in.setdefault(sec, []):
                    inst_in[sec].append(name)

    # ---- 判-18 彩りと主旋律・声部のぶつかり（機械が数える）----
    findings: list[Finding] = []
    col_total: Counter = Counter()
    col_parts: dict[str, Counter] = {}
    if "彩り" in runs and getattr(song, "colour", None) is not None:
        try:
            cspec = _colour_spec(st)
            cnotes = _pitched_notes(score, cspec) if cspec is not None else []
        except flow.NOT_SWALLOW:
            raise
        except Exception as e:                             # noqa: BLE001（数えられなくても点検役は呼ぶ。数えられなかったことを渡す）
            cnotes = []
            findings.append(Finding(what="彩りの音を数えられなかった（機械の側の失敗）", evidence=cut_note(f"{type(e).__name__}: {e}", 200)))
        for a, _z, _k, name in cnotes:
            col_parts.setdefault(sec_at(a), Counter())[name] += 1
        melody = mel3 or [(n.start_tick, n.start_tick + n.duration_tick, n.key) for n in lead.notes]
        mel_name = "歌" if notes else "主旋律の楽器"
        vnotes = [(n.start_tick, n.start_tick + n.duration_tick, n.key) for t in tracks for n in t.notes]
        col_total, _near, _voice = colour_clashes(cnotes, melody, vnotes, sec_at)   # 音の数は【彩り】の行に使う
        # ★**重なった長さ（拍）で数える**（組G4）。以前は音の数で、伸ばすパッド1音と速い分散和音何十音が同じ重みだった。
        #   声部とも、高さが4半音以内のものだけ（以前は時間が少しでも重なれば数え、低い対旋律と高いハモリも入った）
        if cnotes:
            findings.append(Finding(
                what=f"彩りが主旋律（{mel_name}）と同時に、高さ{ff.NEAR}半音以内で鳴っている長さ（区間ごと）",
                evidence=f"重なった長さ/{mel_name}の鳴っている長さ: {ff.overlap_text(ff.near_overlap(melody, cnotes, sec_at), order)}"
                         "／※同時に近い高さで鳴る事実だけ。重ねる意図（ユニゾン・3度で寄り添う）なら正しい"))
            if tracks:
                findings.append(Finding(
                    what=f"彩りが声部と同時に、高さ{ff.NEAR}半音以内で鳴っている長さ（区間ごと）",
                    evidence=f"重なった長さ/声部の鳴っている長さ: {ff.overlap_text(ff.near_overlap(vnotes, cnotes, sec_at), order)}"
                             "／※彩りは声部の入った所を避けて置く工程（声部が先）。避けずに重ねた所か、重ねる意図のある所かを見る"))

    # ---- 組G4: 声部進行（主旋律を超えた所・交差・平行5度8度の連続）----
    if tracks and (mel3 or len(tracks) > 1):
        from artic_plan import tick_key
        above, cross, para = ff.voice_leading(mel3, [(names[i], [(n.start_tick, n.start_tick + n.duration_tick, n.key)
                                                                 for n in t.notes]) for i, t in enumerate(tracks)], spell_pitch)
        where = lambda t: tick_key(t, bt)                  # noqa: E731
        for label, items in (("声部が主旋律（歌）より上に出た音", above),
                             ("声が交差した所（声部どうし・声部と歌の上下が入れ替わった）", cross),
                             ("平行5度・平行8度の連続（2本が同じ向きに動いて前後とも5度か8度。同じ高さどうしは数えない）", para)):
            per, pos = ff.by_section_text(items, sec_at, where, order)
            findings.append(Finding(what=f"声部進行: {label} {len(items)}か所",
                                    evidence=(f"区間ごと: {per}／位置: {pos}" if items else "")))

    # ---- 声部の厚み（曲全体）・声部の内容の所見（判-15）・後ろの工程の記録の所見（判-22）----
    if texture.get("全体"):
        findings.append(Finding(what="声部の厚み（曲全体）", evidence=texture["全体"]
                                + "／※数えた事実だけ。区間ごとは表の「声部の厚み」の行"))
    if vstyle.get("全体"):
        findings.append(Finding(what="声部の声の型と音の高さ（曲全体）", evidence=vstyle["全体"]
                                + f"／※数えた事実だけ（{ff.WHISPER} は AIR=+100 で、高さによらず声にならず息の音だけになる）。"
                                  "区間ごとは表の「声部の声の型と音の高さ」の行"))
    if "声部" in runs:
        findings += [Finding(what=f"声部の所見: {w}") for w in _voice_findings(st)]
    findings += _sound_choice_facts(st, notes, tracks, sec_at)
    findings += _record_findings(st)

    # ---- 区間ごとの表 ----
    rows = ["【区間ごとに鳴っているもの】（音・か所の数。良し悪しの目安ではない。歌の崩しの「伸ばす音」は1拍以上の音）"]
    if sung_only and getattr(score, "lyrics_display", ""):
        rows.append("※見せる歌詞（字幕・投稿用・漢字の歌詞）は、実際に歌う詞と食い違っている（手直しのあと更新されていない）。表の詞は実際に歌う詞")
    if form == "acappella":
        chords = [(c.start_tick, c.start_tick + c.duration_tick, c.chord) for c in (getattr(score, "chords", None) or [])]
        grid_lines = [("歌", mel3)] + [(names[i], [(n.start_tick, n.start_tick + n.duration_tick, n.key) for n in t.notes])
                                       for i, t in enumerate(tracks)]
    for s in spans:
        k = s.kind
        # ★「設計の強さ」と書く（2026-10-11）。伴奏を弾く強さ（打鍵の強さ）と区別が無く、点検役が読み違えた
        line = [f"■{k}（小節 {s.bar_from}-{s.bar_to}・設計の強さ{inten.get(k, '?')}）"]
        if lyr.get(k):
            line.append(("   詞（実際に歌う詞・かな）: " if sung_only else "   詞: ") + ff.lyric_head(lyr[k]))
        if "歌い方" in runs or notes:
            vo = f"歌 {sung.get(k, 0)}音"
            if expr_sec.get(k):
                vo += f"／歌の強さ: {_counts(expr_sec[k])}"
            if how_sec.get(k):
                vo += f"／歌い回し: {_counts(how_sec[k])}"
            if entry_sec.get(k):
                vo += f"／入り方: {_counts(entry_sec[k])}"
            if "歌の崩し" in runs:
                vo += (f"／歌の崩し: {sum(bend[k].values())}か所（{_counts(bend[k])}）" if bend.get(k) else "／歌の崩し: なし")
                if place.get(k):
                    vo += f"（伸ばす音・フレーズの終わりに {place[k][0]}・それ以外の音に {place[k][1]}）"
            line.append("   " + vo)
        if leap_rows.get(k):
            line.append(f"   跳躍（{ff.LEAP}半音以上）のあとの長い伸ばし（{ff.LEAP_HOLD / ff.Q:g}拍以上）: " + "・".join(leap_rows[k]))
        if same_rows.get(k):
            line.append("   歌と同じ高さで同時に鳴る伴奏（パート）: " + same_rows[k])
        if "声部" in runs:
            line.append("   声部: " + (_counts(voice_n[k]) + "（音）" if voice_n.get(k) else "なし")
                        + (f"／歌の音のうち同じ位置に声部の音がある: {cover[k][0]}/{cover[k][1]}" if voice_n.get(k) and cover.get(k) else ""))
        if texture.get(k):
            line.append("   声部の厚み: " + texture[k])
        if vstyle.get(k):
            line.append("   声部の声の型と音の高さ: " + vstyle[k])
        if form == "acappella" and (notes or tracks):
            line.append("   和声（拍の頭で鳴っている高さ・鳴っていなければ -）:")
            line += ["     " + r for r in ff.harmony_grid(s.bar_from, s.bar_to, bt, ff.beat_ticks(score), chords, grid_lines, spell_pitch)]
        if not lead.is_empty():
            line.append(f"   主旋律の楽器: {lead_n.get(k, 0)}音")
        if "伴奏" in runs:
            line.append("   楽器（伴奏と彩り）: " + ("・".join(inst_in.get(k, [])) or "なし"))
        if "彩り" in runs:
            line.append("   彩り: " + (_counts(col_parts[k]) + "（音）" if col_parts.get(k) else "なし"))
        if "奏法" in runs:
            line.append("   奏法: " + (_counts(artic_n[k]) if artic_n.get(k) else "なし"))
        rows.append("\n".join(line))
    if not policies:
        return "\n".join(rows), findings

    # ---- 工程ごとの方針と全体の数 ----
    parts: list[str] = []
    if "曲名" in runs and (named := _title_block(song, score, lyrics_in_table=bool(lyr))):
        parts.append(named)
    iplan = getattr(song, "instrumentation", None)
    if form != "song" and iplan is not None:
        from instrumentation import instrument_name
        ins = " / ".join(f"{instrument_name(i)}({i.part})" for i in (iplan.instruments or [])) or "楽器なし"
        parts.append(f"【編成】{ins}／ドラム {iplan.drum_kit}／声 {iplan.voice}・{iplan.vocal_style}"
                     + (f"／主旋律の楽器 {iplan.lead}番（区間 {'・'.join(iplan.lead_sections) or '主旋律のある区間の全部'}）"
                        if getattr(iplan, "lead", None) else "")
                     + f"\n   方針: {_reason(iplan)}")
    if "歌い方" in runs:
        jumps = _strength_jump_text(st)
        cuts = _cut_ending_text(st)
        parts.append(f"【歌い方】全体: 歌の強さ {_counts(expr_all) or '書かれていない'}"
                     + (f"／歌い回し {_counts(how_all)}" if how_all else "")
                     + (f"（書かなかったフレーズ {unwritten}＝規則の表情のまま）" if unwritten else "")
                     + (f"\n   隣り合うフレーズの歌の強さの段差（歌の強さの順 {' < '.join(vocal_expression.LOUDNESS)}）: {jumps}" if jumps else "")
                     + (f"\n   歌い回し『切って』のフレーズの最後の音の長さ（番号「最後の音節」・長い順）: {cuts}" if cuts else "")
                     + f"\n   方針: {_reason(getattr(song, 'expression', None)) or '—'}")
    if "歌の崩し" in runs:
        va = getattr(song, "vocal_artic", None)
        allb = Counter(n.artic for n in notes if getattr(n, "artic", ""))
        at_long = sum(v[0] for v in place.values())
        parts.append(f"【歌の崩し】全体: {_counts(allb) or 'なし'}"
                     + (f"（伸ばす音・フレーズの終わりに {at_long}・それ以外の音に {sum(allb.values()) - at_long}）" if allb else "")
                     + f"\n   方針: {_reason(va) or '—'}")
    if "ハモリ" in runs:
        pol = getattr(iplan, "harmony", "none") if iplan is not None else "?"
        hp = getattr(song, "harmony", None)
        plan_lines = "".join(f"\n   ・{'／'.join(p.sections)}: {p.aim}（{p.density}）" for p in (getattr(hp, "parts", None) or []))
        parts.append(f"【ハモリ】編成の方針: {pol}"
                     + (f"（{cut_note(getattr(iplan, 'harmony_reason', ''), 200)}）" if getattr(iplan, "harmony_reason", "") else "")
                     + (plan_lines or "\n   計画: 重ねない")
                     + (f"\n   方針: {_reason(hp)}" if hp is not None else ""))
    if "声部" in runs:
        parts.append(f"【声部】{len(tracks)}本・{sum(len(t.notes) for t in tracks)}音"
                     + (f"\n   方針: {_reason(getattr(song, 'voices', None))}" if getattr(song, "voices", None) is not None else ""))
    if "彩り" in runs:
        col = getattr(song, "colour", None)
        parts.append(f"【彩り】{sum(col_total.values())}音（{len(getattr(col, 'parts', None) or [])}パート）"
                     + (f"\n   方針: {_reason(col)}" if col is not None else ""))
    if "奏法" in runs:
        allart = Counter(n for c in artic_n.values() for n in c.elements())
        parts.append(f"【奏法】全体: {_counts(allart) or 'なし'}\n   方針: {_reason(getattr(song, 'articulation', None)) or '—'}")
    # ※マスターの頼みは、2026-10-04（組P1）から点検役の呼び出し口が依頼文の頭に置く（`song_gate_brief.request_block`・全部の点検役）
    return "\n".join(rows) + "\n\n" + "\n".join(parts), findings


# ---- 点検役を呼ぶ --------------------------------------------------------------------------------

def _live_if_remade(form: str) -> None:
    """**続きから作る周で、この周に本物で作った工程があれば、本物の点検役に見せる**（再生では何もしない）。

    再生の差し替え（`song_replay.Replay.judge`）は「直前の呼び出しが本物だったか」で本物の点検役を呼ぶかを決める。
    仕上げの点検役は曲の後ろ半分をまとめて読むので、歌い方だけ作り直した周でも、その後の編成〜奏法が保存から戻ると
    「直前は保存から」になり、作り直した歌い方を見ずに「通す」になる。この周で本物を呼んだ工程（`missing`）があれば本物に回す。
    """
    import song_replay
    rp = song_replay._state
    if rp is None or rp.strict:
        return
    if any(ctx.step_of_label(lb) in FORM_STEPS[form] for lb in rp.missing):
        rp.last_was_live = True


def run_finish_gate(st) -> None:
    """**仕上げの点検役**（曲の形で走る最後の工程のあと・1回）。後ろの工程にだけ、1曲で1回まで差し戻せる。

    読む: 楽譜（`st.score`）・時間軸（`st.timeline`）・後ろの工程の出力（`song.expression` ほか）・編成・設計の強さ。
    書く: 記録 `finish_gate`（判断・理由・気になったところ・機械の所見）。差し戻すなら `RedoFrom` が抜ける。
    """
    song, verbose = st.song, st.verbose
    form = song.form
    material, findings = finish_material(st)
    opts = redo_options(form)
    ran = [s for s in finish_targets(form)]
    recs = {s: song.last(_RECORD_OF[s]) for s in ran if s in _RECORD_OF}
    status = "・".join(f"{s}: {r.status}" for s, r in recs.items() if r is not None and r.status != "ok") or "ok"
    trouble = " ／ ".join(f"{s}: {r.trouble}" for s, r in recs.items() if r is not None and r.trouble)
    design = song.design
    ctx_line = (f"曲『{st.title or song.title()}』（{st.genre_name}）／構成: "
                + " → ".join(f"{s.kind}(強さ{getattr(s, 'intensity', 3)})" for s in (getattr(design, "sections", None) or [])))
    _live_if_remade(form)
    v = ctx.judge(FINISH, f"後ろの工程（{'・'.join(ran)}）を作り終えた。出来たものを下にまとめる", status, trouble,
                  findings, verbose=verbose, material=material, context=ctx_line,
                  redo_from_options=opts, can_redo=False, agent=_finish_judge, stop_step=last_step(form))
    song.record(StepRecord(step=FINISH_RECORD, reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                           status="ok" if v.action == "accept" else "compromised",
                           warnings=[f.what + (f"（{f.evidence}）" if f.evidence else "") for f in findings],
                           request=v.action))
    st.log["finish_gate"] = f"{v.action}: {v.reason}"
    if v.action == "redo" and (v.redo_from or "").strip() in opts:
        note_redo_sent(FINISH)                             # 1曲で1回まで（次の周では戻り先を見せない）
    flow._maybe_redo_from(v, FINISH, song)
