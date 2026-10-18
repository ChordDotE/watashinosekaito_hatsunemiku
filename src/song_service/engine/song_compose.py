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

"""**曲先**で作る（v2 は詞先）。テーマ → 設計 → コード → **旋律** → **詞** → 伴奏。

## なぜ順序を変えるか

v2（詞先）は「モーラ配分を決める → 詞を書く → 各モーラに1音ずつ当てる」だったため、
**旋律が詞の形に完全に従属**していた。実測（『Morning Step!』）では:

- 全フレーズが「8分を並べて最後に伸ばす」の**一択**
- フレーズ開始が **4拍ごとに固定**、休符が**全部 0.5拍**
- 別のフレーズ3つが**音高まで完全に同一**

「無理やり歌詞を合わせている・伸び伸びしていない」という実機の指摘そのもの。
配分（8分半分／16分2〜3割／伸ばし1割）は満たしていたので、**数値を満たしても音楽にならない**
という実例になった。

## この版

**旋律を先に、歌詞のことを考えずに書く**。フレーズの長さも休符も旋律の都合で決める。
そのあと「このフレーズは7音」という**枠に合う詞**を当てる（ボカロP の主流手順）。
旋律が詞に縛られないので、フレーズ長・音価・休符が自然に不揃いになるはず。

## このファイル（2026-09-30 に工程ごとのファイルへ分けた）

ここは**曲づくりの入口（`compose_song`）と、工程を依存の順に呼ぶ本体（`_compose_once_body`）だけ**。
本人「今のままだと直列でしか作業できないから、まず並列作業できるようにした方がいい」「工程ごとのファイルに分けてほしい」。
それまでは約3,900行の1ファイルで、`_compose_once_body`（約1,200行の1関数）がテーマから奏法まで全工程を上から順に動かしており、
どの工程を直してもこの関数に触ることになった。

| ファイル | 中身 |
|---|---|
| `song_context` | 共有の状態（頼まれごと・前回の出力・いまの曲・拍子）・LLM の呼び出し口（再生の差し替え口）・前段の渡し方・申し送り（`brief`） |
| `song_flow` | 工程の記録（`_rec`）・途中保存・途中で一度出す・差し戻しの合図・受け渡しの入れ物（`ComposeState`） |
| `song_outline` | テーマ・設計・コード・グルーヴ |
| `song_melody`／`song_melody_prompt`／`song_melody_step` | 旋律の型／依頼文／工程（検査と直し・点検役） |
| `song_rhythm` | 旋律を測る側（リズムの物差し・息継ぎの所見・入るモーラ） |
| `song_lyrics` | 詞の型・記法・検査・詞の工程 |
| `song_naming` | 曲名（詞のあとに最後の曲名を決める・2026-10-03。テーマは仮の曲名） |
| `song_place` | 歌の組み立て（詞を音へ当てる・移調・時間軸・コード・調声・楽譜） |
| `vocal_expression`・`vocal_artic` | 歌い方・歌の崩し（`run_expression`・`run_vocal_artic`） |
| `song_band` | 編成・伴奏（点検役とやり直し）・ハモリの計画・声部・彩り |
| `artic_plan` | 奏法（`run_articulation`） |
| `song_finish_gate` | 仕上げの点検役（後ろの工程をまとめて見る・`run_finish_gate`。曲の形で走る最後の工程のあと） |
| `song_final_gate` | 最終の点検役（2段目・曲全体を見る・`run_final_gate`。仕上げの点検役のあと・組み立てのあと・書き出しの前） |

依存の向き: `song_context` → `song_flow` → 工程のファイル → ここ（逆向きの import は無い）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import song_context as ctx                                 # noqa: E402（共有の状態・呼び出し口）
import song_flow as flow                                   # noqa: E402（記録・途中保存・差し戻し）
import artic_plan                                          # noqa: E402（8.7 奏法）
import song_band as band                                   # noqa: E402（編成・伴奏・ハモリ・声部・彩り）
import song_finish_gate                                    # noqa: E402（仕上げの点検役・2026-10-03 判-21）
import song_final_gate                                     # noqa: E402（最終の点検役・2段目・2026-10-04 組P2b）
import song_lyrics                                         # noqa: E402（詞）
import song_melody_step                                    # noqa: E402（旋律）
import song_naming                                         # noqa: E402（曲名・2026-10-03）
import song_outline as outline                            # noqa: E402（テーマ・設計・コード・グルーヴ）
import song_place as place                                 # noqa: E402（歌の組み立て・楽譜）
import vocal_artic                                         # noqa: E402（5.8 歌の崩し）
import vocal_expression                                    # noqa: E402（5.7 歌い方）
from song_fix import StepUnfixed                           # noqa: E402（形式の所見が直らずに止まった合図）
from song_flow import ComposeState, RedoFrom, ShowProgress  # noqa: E402
from song_gate_brief import carry_note                      # noqa: E402（差し戻しの申し送りの書き方・2026-10-04 組P2a）
from song_final_gate import FINAL, FINAL_MAX_REDO           # noqa: E402（最終の点検役・2段目・2026-10-04 組P2b）
from song_gate import FORM_STEPS, GATE_RECORD, MAX_REDO_SONG, Verdict, gate_of_record, redo_sent   # noqa: E402
from song_model import STEP_LABEL, Song, StepRecord, made  # noqa: E402
from song_usage import USAGE                               # noqa: E402（費用は1箇所で数える）

#: **差し戻しの記録**（2026-09-30・出-1／出-4）。差し戻しのたびに `_compose_song_rounds` が積み、次の周で作る曲の
#: 記録の頭に入れる（`_start_song`）。以前は差し戻しが曲の記録に残らず（点検役の記録は最後に通したものだけ）、
#: 捨てた周回の費用もどこにも載らなかった。1曲ごとに空にする
_ROUND_RECORDS: list[StepRecord] = []

#: **前の周の途中保存の場所**（2026-10-03・曲名の工程）。差し戻しの次の周の曲に「前に書いた場所」として渡し、最初の途中保存で
#: 名前が違えば古いほうを消す（`song_flow._autosave`）。曲名の工程を足して、周の途中で名前が変わる（仮の曲名 → 曲名）ようになったため——
#: 渡さないと、前の周の曲名の途中保存が迷子として残る。1曲ごと（`_compose_song_rounds` の頭と終わり）に空にする。
#: ※`song_flow.LAST_AUTOSAVE` は使わない（作り直し `song_remake._run_rebuild` は `compose_song` を通らず、別の曲の途中保存を指したままのことがある）
_PREV_AUTOSAVE: list[Path] = []


# ※旧 steps.json の形（`_meta`・`_song`）で (score, log) を返す `compose_v3`・`_legacy_meta` は 2026-09-30 に外した
#   （呼び出し元が無く、`Song.load` の `_song` の枝と一緒に消した。曲の記録は `Song.to_json` だけ）。


def compose_song(instruction: str, *, verbose: bool = True, stop_after: str = "",
                 form: str | None = None) -> Song:
    """1曲作る。**関門が前工程を指したら、そこから作り直す**（2026-09-01）。

    `form`（2026-10-03・`docs/plan/song_acappella_instrumental.md` §2）: 曲の形（song／acappella／instrumental・`song_files.as_form`
    で読む＝空なら song・読めない綴りは `ValueError`）。走る工程は `song_gate.FORM_STEPS[form]`。省くと、再生・続きから作る途中なら
    その記録の形、それ以外は song。決めるのはコーディネーター（`draft_melody(form=…)`）。

    やり直しの実体は `_compose_once` を呼び直すことだが、**前段はもう作ってある**ので
    `replay.resume` で $0 のまま復元し、**戻した工程から先だけ本物を呼ぶ**。

    差し戻しをやめる条件は**回数だけ**（`MAX_REDO_SONG` 回。2026-09-30・出-1）。やめたら、**その時点までに作った
    工程を全部残して**続きから最後まで作る（差し戻しを投げた点検役は「所見つきで進む」になる）。
    ※以前は費用の上限（最初に差し戻した時点の費用の1.5倍）もあった。基準が早い工程での1回目の費用なので、
      2回目はほぼ必ず上限に達し、しかも打ち止めの周回が1回目の保存から作り直していた（点検役を通った工程まで
      作り直し、1周目で退けられた稿を「前回の出力」として見せた。『孤独のまま』で約5分・$0.61 余計にかかり、
      2周目で得た「サビの伸ばし」が消えた）。回数だけにし、打ち止めでは何も捨てない
    ※差し戻しは**内容**（点検役の判断）の話。**形式**の所見が差分で直らなかったら、差し戻しの回数と関係なく
      その工程で曲づくりを止める（`song_fix.StepUnfixed` がここを抜けて呼び手へ届く・2026-09-30）。
      打ち止めのあとの「最後まで作る」周回でも同じ——形式の壊れたものを鳴らさずに最後まで作ることはしない。
    """
    import song_replay as replay                           # 循環 import を避けるため中で読む
    from song_files import as_form
    from song_gate import reset_progress
    the_form = as_form(form) if form is not None else (replay.form() if replay._state is not None else "song")
    reset_progress()                                       # 無進展の記録は**1曲ごと**に消す
    flow.LAST_AUTOSAVE = None                              # 前の曲の途中保存を指したまま落ちないように
    flow.AUTOSAVE_FAILED = False                           # 前の曲の保存の失敗を持ち越さない（2026-10-03）
    flow.NO_MORE_REDO = False                              # 差し戻しの打ち止めも**1曲ごと**に戻す

    if replay._state is not None:                          # 既に再生/続きから作る途中なら重ねない
        saved = ctx.FORM                                   # 片づけは呼び手（再生を入れた側）の仕事。形だけ元に戻す
        ctx.FORM = the_form
        try:
            return _compose_once(instruction, verbose=verbose, stop_after=stop_after)
        finally:
            ctx.FORM = saved

    ctx.FORM = the_form                                    # 曲の形（1曲ごと・下の finally で song へ戻す）
    try:
        return _compose_song_rounds(instruction, verbose=verbose, stop_after=stop_after)
    finally:
        # ★**頼まれごとは1曲ごと**（2026-09-25）。持ち込んだものと台帳の参照を片づけ、
        #   同じプロセスで次の曲に前の曲の頼みが混ざらないようにする（台帳そのものは曲の記録に残っている）
        ctx.PENDING_REQUESTS.clear()
        ctx.ACTIVE_REQUESTS = []
        ctx.PREV_OUTPUTS.clear()                           # 前回の出力も1曲ごと
        ctx.FINAL_PREV.clear()                             # 最終の点検役が差し戻したときの稿も（2026-10-04 組P2b）
        ctx.FINAL_REDONE.clear()                           # その差し戻しの範囲も（2026-10-04 組F1）
        ctx.CURRENT_SONG = None
        ctx.KEEP_ALLOWED.clear()
        ctx.KEPT.clear()
        __import__("song_llm").KEEP_REJECTED.clear()      # 「前回のまま」を機械が捨てた記録も1曲ごと（道具-12）
        ctx.METER = "4/4"                                  # 拍子も1曲ごと（次の曲へ持ち越さない）
        ctx.FORM, ctx.FORM_FROM, ctx.KEEP_CHECK, ctx.FORM_REQUEST = "song", "", None, ""   # 曲の形も1曲ごと


def _compose_song_rounds(instruction: str, *, verbose: bool, stop_after: str) -> Song:
    """`compose_song` の本体（差し戻しの周回）。

    差し戻しのたびに、**その時点の曲**（差し戻しを投げた周の出力）を保存し、次の周はそこから続ける:
      ・差し戻し（`MAX_REDO_SONG` 回まで）… 戻す工程から先（依存の木）を捨て、残りは保存から $0 で戻す。
        捨てる工程の出力は「前回の出力」（`PREV_OUTPUTS`）として作り直す工程に見せる——**その時点の出力で入れ替える**
        （以前は足し込むだけで、1回目の差し戻しで退けられた稿が2回目以降も「前回の出力」に残っていた）
      ・打ち止め（それ以上の差し戻し）… **何も捨てずに**全部残して続ける。差し戻しを投げた点検役は、保存から戻した
        判断を「所見つきで進む」（stop）に切り替えて通る（`song_replay.Replay.stop_gate`）——作り直しも自工程のやり直しも起きない
    申し送り（`CARRY`）は、1段目の点検役のぶんはいまの周の理由だけを載せ、1曲の終わりに外す（以前は前の差し戻しの理由が残り続け、
    打ち止めでは実際と違う「直せなかった（このまま進めた）」を、作り直している最中の工程へ渡していた）。
    ★最終の点検役のぶんは、差し戻した工程とそれを読む工程を作り直し終えるまで残す（2026-10-04 組F1・1‑3）。
    ★差し戻しの保存は、戻している保存の工程＋この周に作った工程から、作り直す工程を引いたもの（2026-10-04 組F1・1‑4。以前はこの周に
      作った工程だけで、差し戻しに関係ない工程まで前の稿なしで作り直していた）。作り直す工程には、周をまたいで最後に作った稿を渡す。

    ★**最終の点検役（2段目）の差し戻し**（2026-10-04・組P2b・`song_final_gate`）も同じ周回に乗る。違いは:
      回数を1段目と別に数える（`FINAL_MAX_REDO`＝2・1段目が打ち止めでも差し戻せる）／差し戻したときの稿を控える（`ctx.FINAL_PREV`）／
      巻き込まれた後ろの工程は「前回のまま」を選べる／最終の点検役の判断の写しを次の周の記録の頭に入れる（2周目に前回の NG を読む）。
      使い切ったあとの NG は差し戻さず、最終の点検役が `song_fix.FinalUnpassed` で止める（ここを抜けて呼び手へ届く）。

    ★**作り直し（`song_remake._run_rebuild`＝`remake_from_step`・`change_form`）もここを通る**（2026-10-03・再レビュー）。
      1周目は呼び手が入れた「続きから」（前段を保存から戻す）のまま走り、差し戻しが作り直しの範囲の外（前）を指せば、
      その工程から先を捨てて範囲を広げる。以前は作り直しが `_compose_once` を直に呼んでいて、差し戻しが例外のまま道具まで抜け、
      作り直しの費用ごと捨てられた。
    """
    import dataclasses
    import song_replay as replay                           # 循環 import を避けるため中で読む
    resume_from: Path | None = None
    stop_gate: tuple[str, Verdict] | None = None           # 打ち止めで「所見つきで進む」にする点検役
    mine: list[str] = []                                   # ここが積んだ1段目の申し送り（次の差し戻しと1曲の終わりに外す）
    # ★**最終の点検役の申し送りは別に持つ**（2026-10-04 組F1・1‑3）。差し戻した工程を直し終えるまで残す——次の周で1段目の点検役も
    #   差し戻しても消さない（以前は周の頭で全部外し、最終の点検役の指摘が修正役と点検役から消えた）。作り直し（`song_remake.rebuild`）が
    #   前の版で残った NG を積んでいれば、それもここで扱う
    from song_gate import affected
    from song_gate_brief import parse_carry
    final_notes: list[str] = [n for n in ctx.CARRY if (g := parse_carry(n)) and g[1] == FINAL]
    # ★**周をまたいで、工程ごとに最後に作った出力**（キー → 出力・2026-10-04 組F1・1‑4）。差し戻しの保存と「前回の出力」の元
    known: dict = {}
    first = 0                                              # 1段目の点検役の差し戻しの回数（最終の点検役のぶんは別に数える）
    _ROUND_RECORDS.clear()
    _PREV_AUTOSAVE.clear()
    ctx.FINAL_PREV.clear()
    ctx.FINAL_REDONE.clear()
    try:
        # +2 は「差し戻しの上限に達したあと、**打ち止めのまま最後まで作る**1周」と最初の1周を足したもの。
        # ★**最終の点検役の差し戻し（`FINAL_MAX_REDO` 回）は1段目と別に数える**（2026-10-04・組P2b）——そのぶん周を足す。
        #   使い切ったあとの NG は差し戻さず止める（`song_fix.FinalUnpassed`）ので、周はこれで足りる
        for _round in range(MAX_REDO_SONG + FINAL_MAX_REDO + 2):
            if resume_from is not None:
                # 保存から戻した工程の点検役は、保存した判断をそのまま返す（`Replay.judge`）。
                # 打ち止めの周では、差し戻しを投げた点検役だけ「所見つきで進む」にする
                _st = replay.resume(resume_from)
                if _st is not None:
                    _st.stop_gate = stop_gate
            round_start = len(USAGE)
            try:
                song = _compose_once(instruction, verbose=verbose, stop_after=stop_after)
                # 「前回のまま」を記録に残す（報告と「なぜ」に使う）。作り直し（`song_remake._run_rebuild`）と、最終の点検役の
                # 差し戻しで後ろの工程が選んだもの（2026-10-04 組P2b）。以前は作り直しの側だけが書いていた
                for _s, _why in ctx.KEPT:
                    song.record(StepRecord(step=ctx._STEP_KEY.get(_s, _s), reason=f"前回のまま: {_why}", request="keep"))
                return song
            except RedoFrom as e:
                spent = sum(float(u.get("usd", 0.0) or 0.0) for u in USAGE[round_start:])
                gate = next((r for r in reversed(e.song.history) if r.step.endswith("_gate")), None)
                # 工程名でない関門（仕上げの点検役＝`finish_gate`・最終の点検役＝`final_gate`）は `GATE_RECORD` から引く（2026-10-03・判-21）
                who = e.who or ((gate_of_record(gate.step) or STEP_LABEL.get(gate.step.removesuffix("_gate"), "")) if gate else "")
                final = who == FINAL                       # 最終の点検役（2段目）の差し戻し（2026-10-04 組P2b）
                if flow.NO_MORE_REDO and not final:        # 打ち止めの周では1段目は投げない（`_maybe_redo_from`）はず
                    raise RuntimeError(f"差し戻しの打ち止めのあとに、また差し戻しが来た: {e}") from e
                data = e.song.to_json()
                # 作った工程だけ残す（走らなかった工程の空の出力は次の周の `fill_skipped` が入れ直す・F1）
                # ★**戻している保存の工程も残す**（2026-10-04 組F1・1‑4）。以前はこの周に作った工程（`e.song`）だけを残したので、
                #   2周目以降に差し戻すと、この周でまだ届いていない工程（差し戻しに関係ない編成・伴奏など）が保存から落ち、
                #   前の稿なしで一から作り直された。保存（`replay._state.raw`）に、この周に作った工程を上書きする
                base = dict(getattr(replay._state, "raw", None) or {}) if replay._state is not None else {}
                steps = {**base, **{k: v for k, v in (data.get("steps") or {}).items() if made(v)}}
                known.update(steps)
                for note in mine + final_notes:            # 申し送りは下で積み直す（1段目のぶんは、いまの周の理由だけ）
                    if note in ctx.CARRY:
                        ctx.CARRY.remove(note)
                mine.clear()
                ctx.PREV_OUTPUTS.clear()
                if not final and first >= MAX_REDO_SONG:
                    # ★**打ち止め: 何も捨てずに、いまの曲の続きから最後まで作る**（途中の曲を返さない）。
                    #   差し戻しを投げた点検役の工程は保存から戻り、点検役は「所見つきで進む」で通る。
                    #   ※止めないのは点検役の差し戻し（内容）だけ。形式の所見が直らなければ、この周でも止まる（`StepUnfixed`）
                    flow.NO_MORE_REDO = True
                    why = (f"差し戻しの上限（{MAX_REDO_SONG}回）に達したので、点検役が求めた「{e.step}から作り直す」をやめ、"
                           f"ここまで作った工程をすべて残して所見つきで進めた")
                    if who:
                        stop_gate = (who, Verdict(reason=f"【機械】{why}。点検役の理由: {gate.reason}", action="stop",
                                                  guidance=e.guidance, notes=gate.notes, redo_from="", machine=why))
                    note = f"【{who or '点検役'}の点検役の所見（{why}）】{e.guidance}"
                    rec = StepRecord(step="redo", request=e.step, status="ok" if who else "compromised", guidance=e.guidance,
                                     reason=f"差し戻しの打ち止め: {who or '点検役'}の点検役 — {why}。所見: {e.guidance[:300]}"
                                            "（費用は差し戻しまでの周で使ったもの。この周の工程は保存から戻して使い続けた）",
                                     usd=spent)
                    print(f"  [差し戻し] ★上限（{MAX_REDO_SONG}回）に達した。**差し戻しはやめて、"
                          f"ここまでの工程を全部残したまま最後まで作る**／残す工程: {sorted(steps)}")
                else:
                    # **戻す工程から先を捨てて**、残りは保存から戻す
                    drop = flow.drop_keys(e.step)
                    snapshot = dict(steps)                 # 差し戻したときの全工程（最終の点検役の2周目が比べる相手）
                    steps = {k: v for k, v in steps.items() if k not in drop}
                    # ★差し戻された工程にも**前回の出力**を渡す（2026-09-26）。理由（`CARRY`）だけでは、
                    #   直す必要の無い所まで一から書き直す。人は差し戻された稿に手を入れる。
                    #   ★作り直すのは差し戻した工程とそれを読む工程（`affected` を形で重ねたもの）と、前の周で捨ててまだ作り直していない工程。
                    #   どれにも周をまたいで最後に作った稿を渡す（2026-10-04 組F1・1‑4）
                    prev = flow.prev_outputs_for(known, steps, ctx.FORM)
                    ctx.PREV_OUTPUTS.update(prev)
                    # ★**作り直し（`song_remake._run_rebuild`）の途中の差し戻し**（2026-10-03・再レビュー）: 差し戻された工程は
                    #   「前回のまま」にできない（点検役が退けた稿に戻るだけ）。作り直す工程の「前回のまま」の控えも捨てる
                    #   （次の周で作り直すか、また選べば積み直す）。新しい曲では両方とも空なので何も変わらない
                    ctx.KEEP_ALLOWED.discard(e.step)
                    ctx.KEPT[:] = [k for k in ctx.KEPT if ctx._STEP_KEY.get(k[0], k[0]) not in drop]
                    # ★**どの点検役の指摘かを書く**（2026-10-04・組P2a）。差し戻された工程の受け持ちの点検役（1段目）が
                    #   「上からの指摘」として読む（`song_gate_brief.upper_notes`）。自分が出した指摘は「前回のあなたの判断」で別に見る
                    note = carry_note(e.step, who, e.guidance)
                    if final:
                        # ★**最終の点検役の差し戻し**（2026-10-04・組P2b）。回数は最終の点検役が数えた（`song_gate.note_redo_sent`）。
                        #   ①差し戻したときの稿を控える（2周目に「前の稿から変わった所」を比べる相手・1段目の差し戻しで入れ替わらない）
                        #   ②作り直しに巻き込まれた後ろの工程は「前回のまま」を選べる（差し戻された当の工程は選べない）
                        #   ③最終の点検役の判断の写しを次の周の記録の頭に入れる（2周目に前回の NG を記録から読む。費用は差し戻しの記録に載る）
                        #   ④（2026-10-04 組F1・1‑4）控える稿は**全工程の写し**、差し戻しの範囲は `FINAL_REDONE`。2周目は範囲と、
                        #     その間の1段目の差し戻しで変わった工程を並べる（`song_gate_brief.final_rejudge_text`）
                        n, cap = redo_sent(FINAL), FINAL_MAX_REDO
                        ctx.FINAL_PREV.clear()
                        ctx.FINAL_PREV.update({s: snapshot[k] for s, k in ctx._STEP_KEY.items() if made(snapshot.get(k))})
                        ctx.FINAL_REDONE.clear()
                        ctx.FINAL_REDONE.update({s for s in affected(e.step) | {e.step} if s in FORM_STEPS[ctx.FORM]})
                        ctx.KEEP_ALLOWED.update(set(prev) - {e.step})
                        if gate is not None and gate.step == GATE_RECORD[FINAL]:
                            _ROUND_RECORDS.append(dataclasses.replace(gate, usd=0.0, warnings=list(gate.warnings)))
                        label = f"最終の点検役の差し戻し（{n}/{cap}回目・1段目の差し戻しとは別に数える）"
                    else:
                        first += 1
                        label = f"差し戻し（{first}/{MAX_REDO_SONG}回目）"
                    rec = StepRecord(step="redo", request=e.step, guidance=e.guidance,
                                     reason=f"{label}: {who or '点検役'}の点検役が"
                                            f"「{e.step}から作り直す」を選んだ——{e.guidance[:300]}"
                                            "（費用は差し戻しまでの周で使ったもの。保存から戻して使い続けた工程のぶんも含む）",
                                     usd=spent)
                    print(f"  [差し戻し] **{e.step}から作り直す**（{label}）"
                          f"／残す工程: {sorted(steps)}\n      理由: {e.guidance[:150]}")
                data["steps"] = steps
                _ROUND_RECORDS.append(rec)
                _PREV_AUTOSAVE[:] = [p] if (p := getattr(e.song, "_autosave_path", None)) is not None else []
                # 名前は曲名の slug（2026-10-03・`?`・`:`・`/` を含む曲名で書けなかった）
                resume_from = (flow.AUTOSAVE_DIR or HERE) / f"_redo_{flow.autosave_name(e.song.title())}.steps.json"
                resume_from.parent.mkdir(parents=True, exist_ok=True)
                resume_from.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                # 作り直す側・後の工程へ理由を届ける（届け先は `song_context.asks_for`＝宛先に1回・巻き込まれた工程には要旨と変わった所）。
                # ★最終の点検役の申し送りは、差し戻した工程かそれを読む工程を次の周でまだ作り直すあいだ残す（1‑3）。最終の点検役が
                #   また差し戻したら入れ替える
                if final:
                    final_notes[:] = [note]
                else:
                    remade = flow.remade_steps(steps, ctx.FORM)
                    final_notes[:] = [n for n in final_notes
                                      if (g := parse_carry(n)) and (affected(g[0]) | {g[0]}) & remade]
                    mine.append(note)
                ctx.CARRY.extend(final_notes + mine)
            finally:
                if replay._state is not None:              # 次の周回で入れ直す
                    replay.uninstall()
                    replay._state = None
        raise RuntimeError("差し戻しの周回が終わらなかった")   # 届かない（打ち止めの周は差し戻さない）
    finally:
        for note in mine + final_notes:
            if note in ctx.CARRY:
                ctx.CARRY.remove(note)
        _ROUND_RECORDS.clear()
        _PREV_AUTOSAVE.clear()


def _compose_once(instruction: str, *, verbose: bool = True, stop_after: str = "") -> Song:
    """1周ぶん作る。`stop_after` を渡すと**その工程まで作って `ShowProgress` で抜ける**。

    ★張った印は **`finally` で必ず戻す**——モジュール変数なので、戻し忘れると
    **次の曲まで途中で止まる**。
    """
    flow.STOP_AFTER = stop_after
    flow.begin_usage()                                     # この周の費用を記録へ割り当てる（出-4）
    try:
        song = _compose_once_body(instruction, verbose=verbose)
        if flow.settle_usage(song, final=True):            # `_finish` のあとに積まれた呼び出しがあれば（通常は無い）
            flow._autosave(song)
        return song
    except StepUnfixed as e:
        # ★**形式の所見が直らずに止まった**（2026-09-30）。前の工程は `_rec` のたびに途中保存してあるが、
        #   止まった工程の記録（`<工程>_fix`・直らなかった所見）はその後に積まれたので、ここでもう一度保存する。
        #   続きは `remake_from_step`（止まった工程から・前の工程は費用なしで戻す）
        flow.settle_usage(e.song or ctx.CURRENT_SONG, final=True)
        flow._autosave(e.song or ctx.CURRENT_SONG)
        if verbose:
            print(f"  [止めた] {e}")
        raise
    except ShowProgress as e:                              # 一度出す: そこまでの費用を記録に載せて保存し直す
        if flow.settle_usage(e.song, final=True):
            flow._autosave(e.song)
        raise
    finally:
        flow.STOP_AFTER = ""
        flow.end_usage()


def _compose_once_body(instruction: str, *, verbose: bool = True) -> Song:
    """1周ぶんの本体。**工程を依存の順に呼ぶだけ**（2026-09-30 に工程ごとの関数へ分けた）。

    工程の中身（依頼・検査・差分の直し・点検役・記録）は各工程のファイルにあり、ここは順番だけを持つ。
    工程の出力は曲（`song`）が持ち、それ以外に後の工程が使う途中の値は `ComposeState` が運ぶ。
    **順番を入れ替える・工程を足すときは**、依存の木（`song_gate.DEPS`）と、記録の並び・保存済みの曲の再生で確かめる
    （伴奏の点検役がハモリの計画のあと・声部が彩りの前、などは記録の並びと工程の読み合わせで決まっている）。
    """
    song = _start_song(instruction)
    st = ComposeState(song=song, instruction=instruction, verbose=verbose)
    # ★**曲の形で走る工程を決める**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §3）。集合は `song_gate.FORM_STEPS` の1か所
    #   （依存の木 `DEPS` は変えない）。走らない工程の出力は、ほかの工程と同じ型で**中身が空・状態 skipped**
    #   （`_start_song` が `flow.fill_skipped` で入れる・2026-10-03 F1。以前は `None` で、記録の `steps` の形が曲の形で変わっていた）。
    #   アカペラ: 伴奏・彩り・奏法と、伴奏の組み立て・点検役が無い。インスト: 詞の設計〜詞・歌い方・歌の崩し・ハモリ・声部が無く、
    #   歌の組み立ての代わりに主旋律を詞なしで置く（`place_lead`）。工程の中身の形ごとの違いは各工程（形は `ctx.FORM`・`song.form`）
    runs = FORM_STEPS[song.form].__contains__

    outline.run_theme(st)                                  # テーマ
    outline.run_design(st)                                 # 設計
    outline.run_chords(st)                                 # コード
    outline.run_groove(st)                                 # グルーヴ（拍子もここで決まる）
    song_melody_step.run_melody(st)                        # 旋律（検査・差分の直し・点検役）
    if runs("詞"):
        song_lyrics.run_lyrics(st)                         # 詞の設計 → サビの詞 → 詞（検査・点検役）
    if runs("曲名"):
        song_naming.run_naming(st)                         # 曲名（詞のあと・インストでは旋律のあと・2026-10-03）
    place.build_timeline(st)                              # 時間軸・区間の調と移調量・コードの配置（全部の形）
    if runs("詞"):
        place.place_vocal(st)                              # 歌の組み立て（詞を音へ当てる・調声）
    else:
        place.place_lead(st)                               # 主旋律を詞なしで置く（インスト）

    # ★**申し送りをここで作る**（2026-09-23）。**歌の工程が使うので、使う場所より前**。
    #   当初は編成工程の手前（下のほう）でだけ作っていたため、5.7 と 5.8 が
    #   **`UnboundLocalError` で毎回落ちていた**（実機で発覚・`try` に包まれていたので
    #   曲は落ちず `compromised` に残った＝**2つの工程が一度も動いていなかった**）。
    #   `brief()` は theme/design/chords/melody から作るだけなので、ここで作れる。
    st.brief_text = ctx.brief(song.theme, song.design, song.chords, song.melody)

    if runs("歌い方"):
        vocal_expression.run_expression(st)                # 5.7 歌い方
    if runs("歌の崩し"):
        vocal_artic.run_vocal_artic(st)                    # 5.8 歌の崩し
    band.run_instrumentation(st)                           # 6 編成（どの形にもある）
    if runs("伴奏"):
        band.run_arrangement(st)                           # 7 伴奏（骨組み）
    if runs("ハモリ"):
        band.run_harmony_plan(st)                          # 8 ハモリの計画（と声の種類）
    if runs("伴奏"):
        band.run_arrange_gate(st)                          # 伴奏の組み立て・点検役・自工程のやり直し
    place.make_score(st)                                   # 楽譜にまとめる
    if runs("声部"):
        band.run_voices(st)                                # 8.5 声部
    if runs("彩り"):
        band.run_colour(st)                                # 7.5 彩り（声部のあと）
    if runs("奏法"):
        artic_plan.run_articulation(st)                    # 8.7 奏法（彩りのあと）
    if runs("歌の崩し"):
        vocal_artic.record_unrendered(st)                  # V6 へ描かない歌の奏法の記録
    song_finish_gate.run_finish_gate(st)                   # 仕上げの点検役（後ろの工程をまとめて見る・後ろへ1回まで差し戻す）
    song_final_gate.run_final_gate(st)                     # 最終の点検役（2段目・曲全体。どの工程へも1曲2回まで差し戻す・残れば止める）
    return _finish(st)


def _start_song(instruction: str) -> Song:
    """1周の曲を用意する（台帳・いまの曲・それまでの差し戻しの記録）。"""
    import dataclasses
    song = Song(instruction=instruction, form=ctx.FORM)    # 曲の形は入口が `ctx.FORM` に入れる（2026-10-03）
    flow.fill_skipped(song)                                # その形で走らない工程の出力は空・skipped（F1・形でデータの形を変えない）
    # ★**頼まれごとの台帳を持ち込む**（2026-09-25・段階2）。作り直しや鼻歌が積んだものを曲に載せ、
    #   工程へ届ける元にする（`song_llm.run_sync` が `ACTIVE_REQUESTS` を読む）。
    song.requests = list(ctx.PENDING_REQUESTS)
    ctx.ACTIVE_REQUESTS = song.requests
    ctx.CURRENT_SONG = song                                # 前段の意図を下流へ届けるため（`upstream_intents`）
    # ★**それまでの差し戻しを記録の頭に残す**（2026-09-30・出-1／出-4）。捨てた周の費用もここに載る
    for r in _ROUND_RECORDS:
        song.record(dataclasses.replace(r, warnings=list(r.warnings)))
    # ★前の周の途中保存の場所を引き継ぐ（2026-10-03・曲名の工程）。この周の最初の途中保存で、名前が違えば古いほうを消す
    if _PREV_AUTOSAVE:
        song._autosave_path = _PREV_AUTOSAVE[0]              # type: ignore[attr-defined]
    return song


def _finish(st: ComposeState) -> Song:
    """**曲の状態を Song にまとめる**（D9）。以降、編集もエラー復帰もここを見る。"""
    song, verbose = st.song, st.verbose
    song.score = st.score
    # ★**楽譜にも曲の形を書く**（2026-10-03・§2）。曲づくりの外（再生・状態・申し送り・書き出し）は楽譜から形を読む。
    #   記録（`song.form`）→楽譜（`SongScore.form`）の写しはここ1か所
    if song.score is not None and getattr(song.score, "form", "song") != song.form:
        song.score.form = song.form
    song.chords_of = st.ch_map
    song.timeline = st.timeline                            # **区間の位置はここが正**
    song.bars_of = st.bars_of                              # 以下2つは timeline からの導出（互換用）
    song.sec_of_note = st.sec_of_note
    song.key_of = st.key_of
    song.instruments = st.inst_gms
    song.allow_drums = st.allow_drums
    if verbose and (bad := song.troubles()):
        print("  【うまくいかなかった工程】")
        for r in bad:
            print(f"      {r.summary()}")
    # ★**最後にもう一度保存する**（2026-09-02）。工程ごとの保存は `_rec` の中で走るが、
    #   **時間軸（`timeline`）・コード表・音符と区間の対応は工程の後に入る**ので、
    #   そこまでで止めると記録に `timeline: None` / `bars_of: {}` が残る。
    #   実際そうなっていて、**書き出した音を区間ごとに測る道具（`measure_mix`）が
    #   保存済みの曲に対して使えなかった**（実機で発覚）。
    flow.settle_usage(song, final=True)                    # 最後の工程の費用も記録へ（出-4）
    flow._autosave(song)
    return song
