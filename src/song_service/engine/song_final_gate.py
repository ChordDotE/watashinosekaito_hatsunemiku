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

"""**最終の点検役**（2段目）——1段目の点検役（旋律・詞・伴奏・仕上げ）を束ね、曲全体を見る（2026-10-04・組P2b）。

## 前提・背景

設計は `docs/plan/song_final_reviewer.md`（決まったこと・追記）。本人（2026-10-04）「最後の点検役が NG を出したら作り直すべき。
どこでどういう NG だったからこの工程からやり直して、と伝えて以降をやり直す。妥協した状態でマスターに出しても困る」
「最終判断は点検役を1つ作って分けたほうがいい。音楽コーディネーターが最後の点検役をするのはしんどい」。
それまで曲全体を最後に見ていたのは、作曲担当（コーディネーター）が書き出しの前に呼ぶ「最終検問」（`tools/diagnosis.review_progress`）で、
NG のとき作り直すか出すかも作曲担当が決めていた。仕上げの点検役は後ろの工程の点検と曲全体・頼みの照合を兼ねていた。

## 決めたこと

- **2つの階層**。1段目＝旋律・詞・伴奏・仕上げ（仕上げは後ろの工程の点検に専念）。2段目＝ここ。**別の工程・別の呼び出し**。
- **いつ**: 仕上げの点検役のあと・組み立てのあと（楽譜ができてから）・書き出しの前（`song_compose._compose_once_body` から1行）。
  新しい曲と作り直し（`remake_from_step`・`change_form`＝`song_remake._run_rebuild`）で走る。カバー・尺の変更・手書き取り込みは
  この流れを通らないので走らない。
- **見るもの**（人の監督者が判断に要るものに絞る。全部の出力は渡さない）:
  ①頼みどおりか … 頼みの原文・作り直しで頼まれたこと（依頼文の頭・`song_gate_brief.request_block`）・テーマの伝えたいこと・心の動き
  ②1段目の指摘が直ったか … 1段目の点検役の判断の流れと残した弱い所（依頼文の頭・`song_gate_brief.gate_history_text`）
  ③曲全体 … 区間ごとの表（仕上げの点検役の表を使い回す・`song_finish_gate.finish_material(policies=False)`）・構成・曲名と理由
  ④2周目以降 … 自分の前回の NG・差し戻した工程群の対応（`response`）と前の稿から変わった所・その間の1段目の再判断
    （`song_gate_brief.final_rejudge_text`）
  ①②④は呼び出し口（`song_gate.judge` → `song_gate_brief.gate_brief`）が足す。ここは③とテーマを材料にする。
- **答え**は「通す（弱い所があれば `notes`）」か「NG（`redo`＋`redo_from`＝原因の工程・`guidance`＝どこがどう NG か・どう直すか）」。
  **stop（所見つきで進む）は無い**（`allow_stop=False`。妥協して出さない）。差し戻せる先は、その形で走る工程のどれでも（テーマを除く）。
- **NG なら、その工程へ差し戻す**（`song_flow.RedoFrom`・名前つき）。いまの差し戻しの周回（`song_compose._compose_song_rounds`）が
  その工程から先を捨てて作り直す。申し送りは `carry_note`（「最終の点検役の指摘」）。差し戻された工程は前の稿を土台に直して `response` に
  対応を書き、後ろの工程は「前回のまま」も選べ、途中の1段目の点検役も通る。そのあと、ここがもう一度走る。
- **差し戻しは1曲2回まで**（`FINAL_MAX_REDO`・1段目の差し戻し `MAX_REDO_SONG` とは別に数える）。使い切っても NG なら、
  完成扱いにせず止める（`song_fix.FinalUnpassed`＝`StepUnfixed` の子。残った NG と「ここから続けられる」工程を返す）。
- 記録は `final_gate`（判断・理由・指示 `guidance`・弱い所 `notes`）。再生（`song_replay`）では LLM を呼ばない——呼び口 `ctx.judge` を
  通すので、保存した判断を戻すか「通す」。続きから作る周で本物の工程が1つでもあれば本物に見せる（`_live_if_remade`）。

触れる軸: 継続性（頼みと前の指摘を最後まで持って判断する）・公開性（1曲あたり点検役1回ぶんの費用。NG の曲だけ作り直しの費用）。
"""
from __future__ import annotations

from pydantic_ai import Agent

import song_context as ctx                                 # 呼び出し口（点検役＝再生が差し替える）
from song_finish_gate import _title_block, finish_material, last_step
from song_flow import RedoFrom
from song_fix import FinalUnpassed
from song_gate import FORM_STEPS, GATE_RECORD, STEP_ORDER, Verdict, note_redo_sent, redo_sent
from song_llm import pick_model
from song_model import StepRecord, made
from song_prompts import prompt

#: 点検役の名前（`ctx.judge` の工程名・`song_gate_brief.GATE_SCOPE` の行・差し戻しの申し送りの「◯◯の点検役」）
FINAL = "最終"
#: 記録の名前（`final_gate`）
FINAL_RECORD = GATE_RECORD[FINAL]
#: 1曲のうちに最終の点検役が差し戻せる回数（1段目の差し戻し `song_gate.MAX_REDO_SONG` とは別に数える）
FINAL_MAX_REDO = 2
#: 依頼文の結びの問い（工程の点検役の「破綻していますか」とは違う物差し——曲として聴いた人が気づく NG か）
ASK = ("**①この曲を、頼みどおりの曲として通せますか**——曲として聴いた人が気づく NG があれば `redo`（`redo_from` に原因の工程・"
       "`guidance` にどこがどう NG か・どう直すか）、無ければ `accept`。"
       "**②通すなら、弱い所はどこですか**（`notes`）。")

_final_judge: Agent[None, Verdict] = Agent(
    pick_model("COORD", "flash"), output_type=Verdict, system_prompt=prompt("song_final_judge"))


def final_targets(form: str) -> tuple[str, ...]:
    """**最終の点検役から差し戻せる工程**（作る順）。その形で走る工程のどれでも（テーマは `STEP_ORDER` に無い＝戻せない）。"""
    return tuple(s for s in STEP_ORDER if s in FORM_STEPS[form])


def final_material(st) -> str:
    """最終の点検役に読ませる**曲全体**: テーマ（伝えたいこと・心の動き）・区間ごとの表・曲名と理由。

    区間ごとの表は仕上げの点検役の表（`song_finish_gate.finish_material`）をそのまま使う。機械が数えた所見は渡さない——
    後ろの工程の細部は仕上げの点検役が見た（その判断は【1段目の点検役の判断】で届く）。構成は依頼文の `context` に載る。
    """
    song = st.song
    th = song.theme
    out = [f"【テーマ】伝えたいこと: {getattr(th, 'message', '') or '—'}／心の動き: {getattr(th, 'arc', '') or '—'}"]
    if getattr(th, "subject", ""):
        out.append(f"   題材: {th.subject}／情景: {getattr(th, 'scene', '')}／感情: {getattr(th, 'emotion', '')}")
    table, _ = finish_material(st, policies=False)
    out.append(table)
    # ★**区間どうしの差も渡す**（2026-10-11・本人承認）。区間ごとの表だけでは、Bメロとサビが同じ型でも差が見えなかった
    #   （『てのひらのシグナル』v16）。新しい曲と手直しのあとの点検の両方がここを読む
    try:
        from song_contrast import contrast_text
        out.append(contrast_text(st))
    except Exception as e:                                 # noqa: BLE001（数えられなくても点検役は呼ぶ。数えられなかったことを渡す）
        out.append(f"【区間の対比】数えられなかった（機械の側の失敗・{type(e).__name__}）")
    if made(getattr(song, "naming", None)):
        out.append(_title_block(song, st.score, lyrics_in_table="   詞" in table))
    return "\n\n".join(t for t in out if t)


def _live_if_remade() -> None:
    """**続きから作る周で、この周に本物で作った工程があれば、本物の点検役に見せる**（再生では何もしない）。

    最終の点検役は曲全体を読むので、どの工程を作り直した周でも本物に見せる（仕上げの点検役の `_live_if_remade` と同じ考え。
    あちらは後ろの工程だけを見る）。この周で本物を呼んだ工程（`missing`）が1つでもあれば本物に回す。
    """
    import song_replay
    rp = song_replay._state
    if rp is None or rp.strict:
        return
    if rp.missing:
        rp.last_was_live = True


def _ng_rows(v: Verdict, back: str) -> list[str]:
    """止めるときに返す、残った NG の行（どこがどう・どう直すか）。"""
    rows = [f"NG（原因の工程: {back or '書かれていない'}）: {' '.join((v.reason or '').split())}"]
    if v.guidance:
        rows.append(f"どう直すか: {' '.join(v.guidance.split())}")
    return rows


def run_final_gate(st) -> None:
    """**最終の点検役**（仕上げの点検役のあと・書き出しの前・1周に1回）。

    読む: 楽譜（`st.score`）・時間軸・曲の記録（テーマ・曲名・1段目の点検役の判断）。依頼文の頭の頼み・1段目の判断と、2周目以降の材料は
    呼び出し口が足す（`song_gate_brief`）。
    書く: 記録 `final_gate`（判断・理由・指示・弱い所）。NG なら差し戻す（`RedoFrom`）か、使い切っていれば止める（`FinalUnpassed`）。
    """
    song, verbose = st.song, st.verbose
    form = song.form
    opts = list(final_targets(form))
    design = song.design
    ctx_line = (f"曲『{st.title or song.title()}』（{st.genre_name}）／構成: "
                + " → ".join(f"{s.kind}(強さ{getattr(s, 'intensity', 3)})" for s in (getattr(design, "sections", None) or [])))
    _live_if_remade()
    v = ctx.judge(FINAL, "", "", "", None, verbose=verbose, material=final_material(st), context=ctx_line,
                  redo_from_options=opts, can_redo=False, allow_stop=False, ask=ASK,
                  agent=_final_judge, stop_step=last_step(form))
    back = (v.redo_from or "").strip()
    song.record(StepRecord(step=FINAL_RECORD, reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                           status="ok" if v.action == "accept" else "compromised", request=v.action))
    st.log["final_gate"] = f"{v.action}: {v.reason}"
    if v.action == "accept":
        return
    sent = redo_sent(FINAL)
    if v.action == "redo" and back in opts and sent < FINAL_MAX_REDO:
        note_redo_sent(FINAL)                              # 1曲で2回まで（1段目の差し戻しとは別に数える）
        raise RedoFrom(back, song, v.guidance or v.reason, who=FINAL)
    # ★**使い切っても NG・選べる判断を2回とも外した** → 完成扱いにせず止める（妥協して出さない）。止めた工程の記録は上に積んだ。
    #   続きは最終の点検役が原因と見た工程から（分からなければ空——文が「NG を読んで工程を選ぶ」と案内する）
    raise FinalUnpassed(back if back in opts else "", _ng_rows(v, back), song=song, rounds=sent)


# ---- 編集のあとの最終の点検（2026-10-04・案C・本人決定）-----------------------------------------------------------
#
# 設計と調査は `docs/plan/song_final_reviewer.md`「追記: 編集のあとの点検（案C）」。要点:
# - 追加編集の道具（移調・テンポ・ハモリ・歌の区間の置き換え・伴奏の書き換え）は、道具の中では点検役を呼ばず「点検待ちの束」に積む
#   （`tools/editing.EDIT_SIZE`・`note_edit`）。**書き出しの入口でまとめて1回**ここを呼ぶ（`tools/editing.review_edits`）。
# - **NG のときは所見を返すだけ**（差し戻さない・投げない）。判断は `Verdict` の形のまま、`redo_from` は「直すならこの工程から」の提案。
#   NG でも書き出す（マスターが聴いて決める）。記録は `final_gate_edit`（`final_gate` と分ける）。
# - 1回の頼み（＝1版）で最大2回（`EDIT_MAX_RUNS`）。回数はコーディネーターの実行（`MusicSession`）が数える。
# - 材料は新しい曲と同じ曲全体の表（`final_material`）を、記録（テーマ・設計・時間軸）といまの楽譜から組む（`edit_state`）。
#   足すもの（この編集で頼まれたこと・したこと・前の版から変わった所・前に通した判断）は `song_gate_brief.edit_brief` が足す。

#: 点検役の名前（`ctx.judge` の工程名・費用のラベル `関門(最終（編集のあと）)`・記録の名前は `GATE_RECORD`）
FINAL_EDIT = "最終（編集のあと）"
#: 記録の名前（`final_gate_edit`）
FINAL_EDIT_RECORD = GATE_RECORD[FINAL_EDIT]
#: 1回の頼み（＝1版）で編集のあとの点検を呼べる回数（初回と、コーディネーターが直したあとの1回）
EDIT_MAX_RUNS = 2
#: 編集のあとの結びの問い（新しい曲の `ASK` とは物差しが違う——この編集で NG が生じたか。編集していない所は蒸し返さない）
EDIT_ASK = ("**①この編集で、聴いた人が気づく NG が生じましたか**——頼んだ変化になっているか・編集した所が周りと噛み合うかを見て、"
            "NG なら `redo`（`redo_from` に、直すならどの工程から作り直すのがよいかの提案・`guidance` にどこがどう NG か・どう直すか）、"
            "無ければ `accept`。編集していない所の弱さは、前に通したときの判断に従います（蒸し返さない）。"
            # ★2026-10-14: 前の頼みに反する編集を「蒸し返さない」で通していた（最高音が Eb6 になった v26 など）
            "**ただし【続いている頼み】に反している所は、編集した所でなくても NG にします**（マスターがこの先も守ってと言ったこと）。"
            "**②通すなら、この編集で弱くなった所・気になる所はどこですか**（`notes`）。")
#: 歌の区間を置き換えて歌い方の数がずれうるとき、材料に添える一文
EXPRESSION_DROPPED = ("※この編集で歌の区間を置き換え、歌のフレーズの数が前の版と変わったので、区間ごとの「歌い方・入り方」の数は出していない"
                      "（歌い方はフレーズの番号で数えるため、番号がずれて別の区間に数えてしまう）。")


def genre_name_of(song) -> str:
    """曲の種類の呼び名（新しい曲では演奏の工程が `song_context._GENRE_NAME[テーマの genre]` で引く）。引けなければ空。
    記録から読んだ曲のテーマは dict のこともある（`Song.load_record` は工程の出力を dict のまま持つ）。"""
    th = getattr(song, "theme", None)
    genre = th.get("genre") if isinstance(th, dict) else getattr(th, "genre", None)
    return ctx._GENRE_NAME.get(genre, "") if genre else ""


def typed(song):
    """**工程の出力を型に戻した写し**（記録から読んだ曲 `Song.load_record` は工程の出力を dict のまま持つ。材料の表は型で読む）。
    型は再生と同じ表（`song_replay._types`）。戻せない出力は dict のまま（材料の表がその行を出さないだけ）。元の曲は変えない。"""
    import copy
    import song_replay
    out = copy.copy(song)
    for key, model in song_replay._types().items():
        v = getattr(song, key, None)
        if isinstance(v, dict) and model is not None:
            try:
                setattr(out, key, model.model_validate(v))
            except Exception:                              # noqa: BLE001（古い形の出力は dict のまま）
                pass
    return out


def _phrase_count(score) -> int:
    import vocal_expression
    notes = list(getattr(score, "vocal_notes", None) or [])
    return len(vocal_expression.phrases(notes)) if notes else 0


def edit_state(song, score, *, prev_score=None, region_replaced: bool = False, accomp_mid=None):
    """記録（テーマ・設計・時間軸・各工程の出力）といまの楽譜から、最終の点検役の材料を作る `ComposeState` を組む。
    戻り値は (状態, 材料に添える一文)。

    ★**`accomp_mid`（作業台の伴奏 MIDI）を渡したら、伴奏はそこから数える**（2026-10-04・実機『水たまりのプリズム』v2）。
      伴奏を MIDI で直接書き換える道具（`replace_accompaniment_region`・`remove_accompaniment_track`）は楽譜の `arrangement` を
      変えないので、楽譜から数えると編集前の伴奏の「歌と同じ高さで鳴る伴奏」「楽器」の行が出て、点検役が直った所を NG にした。
      渡すかは呼び手（`tools/editing.review_edits`）が束の道具で決める。読めなければ楽譜から数える（黙らない——材料に一文を添える）。

    ★**歌の区間を置き換えた（`replace_region`）ときの歌い方の数**: 歌い方はフレーズの番号で計画と突き合わせる
      （`song_finish_gate._expression_by_section`）。置き換えでフレーズの数が前の版から変わったら番号がずれるので、
      歌い方の行を出さない（記録の写しから歌い方を外す）。前の版が無いときも出さない。数が同じなら出す。"""
    import copy
    from song_flow import ComposeState
    note = ""
    if region_replaced and getattr(song, "expression", None) is not None and (
            prev_score is None or _phrase_count(prev_score) != _phrase_count(score)):
        song = copy.copy(song)
        song.expression = None
        note = EXPRESSION_DROPPED
    st = ComposeState(song=song, verbose=False)
    st.score = score
    st.timeline = getattr(song, "timeline", None)
    st.notes = list(getattr(score, "vocal_notes", None) or [])
    st.title = getattr(score, "title", "") or song.title()
    st.genre_name = genre_name_of(song)
    st.ts = getattr(score, "time_signature", None) or "4/4"
    st.bars_of = dict(getattr(song, "bars_of", None) or {})
    st.intensity_of = {s.kind: getattr(s, "intensity", 3) for s in (getattr(song.design, "sections", None) or [])}
    st.accomp_midi = None                                  # 楽譜の `arrangement` から数える（`song_finish_gate.finish_material`）
    if accomp_mid is not None:
        from song_finish_gate import accomp_from_midi
        try:
            st.accomp_midi = accomp_from_midi(accomp_mid, getattr(score, "arrangement", None))
        except Exception as e:                             # noqa: BLE001（読めなくても点検役は呼ぶ。読めなかったことを渡す）
            note = (note + "\n" if note else "") + (
                f"※伴奏の MIDI を読めなかった（{type(e).__name__}）ので、区間ごとの表の伴奏は楽譜の編成から数えた"
                "——この編集で MIDI を直接書き換えた伴奏は、表に映っていない。")
    return st, note


def judge_edit(song, score, *, instruction: str = "", ids=(), did=(), changes: str = "", earlier=(),
               prev_score=None, region_replaced: bool = False, accomp_mid=None, verbose: bool = False) -> Verdict:
    """**編集のあとの最終の点検**（投げない・判断を返す）。記録 `final_gate_edit` を `song` に積む（保存は呼び手）。

    `instruction` … この実行のマスターの言葉 ／ `ids` … この編集で台帳に積んだ行 ／ `did` … 点検待ちの束の各件の文 ／
    `changes` … 前の版から変わった所（`song_draft_diff.score_changes`）／ `earlier` … この頼みの中で前に出した判断の文 ／
    `accomp_mid` … 伴奏を MIDI で直接書き換えた編集なら、作業台の伴奏 MIDI（表の伴奏をそこから数える・`edit_state`）。
    呼び出し口は新しい曲と同じ `ctx.judge`（再生では LLM を呼ばない）。NG（`redo`）でも投げない——`redo_from` は提案として返す。
    """
    import song_gate_brief as gb
    from song_usage import USAGE
    form = song.form
    rec_song, song = song, typed(song)                     # 記録は元の曲へ積む（呼び手が保存する）。材料は型に戻した写しで読む
    st, note = edit_state(song, score, prev_score=prev_score, region_replaced=region_replaced, accomp_mid=accomp_mid)
    design = song.design
    ctx_line = (f"曲『{st.title}』（{st.genre_name or '曲の種類は記録から引けなかった'}）／編集のあと／構成: "
                + " → ".join(f"{s.kind}(強さ{getattr(s, 'intensity', 3)})" for s in (getattr(design, "sections", None) or [])))
    material = final_material(st) + (f"\n\n{note}" if note else "")
    saved_song, saved_edit = ctx.CURRENT_SONG, gb.EDIT
    ctx.CURRENT_SONG = song                                # 依頼文の頭（`gate_brief`）が読む「いまの曲」
    gb.EDIT = {"gate": FINAL_EDIT, "instruction": instruction, "ids": list(ids), "did": list(did),
               "changes": changes, "earlier": list(earlier)}
    n0 = len(USAGE)
    try:
        v = ctx.judge(FINAL_EDIT, "", "", "", None, verbose=verbose, material=material, context=ctx_line,
                      redo_from_options=list(final_targets(form)), can_redo=False, allow_stop=False, ask=EDIT_ASK,
                      agent=_final_judge, stop_step=last_step(form))
    finally:
        ctx.CURRENT_SONG, gb.EDIT = saved_song, saved_edit
    from song_flow import usage_record_name
    usd = sum(float(u.get("usd", 0.0) or 0.0) for u in USAGE[n0:]
              if usage_record_name(u.get("step", "")) == FINAL_EDIT_RECORD)
    back = (v.redo_from or "").strip()
    rec_song.record(StepRecord(step=FINAL_EDIT_RECORD, reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                           status="ok" if v.action == "accept" else "compromised", request=v.action,
                           trouble=(f"直すならの提案: {back}から。" if v.action != "accept" and back else "")
                                   + "この編集でしたこと: " + " ／ ".join(did)[:600],
                           usd=usd))
    return v
