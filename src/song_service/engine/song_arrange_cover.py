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

"""**カバー曲をアレンジする**（編成 → 伴奏 → ハモリ）。

カバーは原曲の伴奏をそのまま使うため、作った時点では `chords` も `arrangement` も持っていない。
編集の入口が区間・調・コードを用意したあと（`song_cover_chords.prepare_cover`＝曲の記録にあれば読み、無ければ LLM が書いて記録に残す）、
この工程で**新しい伴奏を組む**。

    python arrange_cover.py <曲のjson> "和楽器風にアレンジして"

**旋律と歌詞は触らない**（原曲のまま）。変えるのは伴奏・ハモリ・音色だけ。
旋律を変えるのは「原曲から離れる」判断なので、**明示の指示があるときだけ**（別工程）。

## 高さの物差しは C 基準（2026-10-01・判-C 本人「c基準で良い」）

オリジナル曲と同じく、LLM に見せる高さ・書かせる高さは **C メジャー／A マイナー基準（移調前）**で、機械がこの曲の調へ移す。
移す量は `song_cover_load.add_sections` が推定した調（`Song.key_of`）から `song_place.keys_shift` が出す（新しい曲と同じ計算）。

| 何か | 誰が決めた高さか | 扱い |
|---|---|---|
| 歌の音符・歌詞・伴奏 MIDI | 原曲の写し | 実音のまま（移さない） |
| コード（`song.chords`） | コードの LLM が書いた（2026-10-02 から。以前は伴奏 MIDI から機械が推定） | C 基準で書かせ、機械が実音へ移して `score.chords`・`song.chords_of` に持つ |
| 伴奏のコード表・歌の音域・声部の表 | 見せるだけ | C 基準で見せる（コード表は `song.chords` のまま・ほかは移調量を引く） |
| 伴奏の `pitches`・リフ、声部の音名 | LLM が書いた | C 基準で書かせ、移調量だけ移す（`to_spec`・`voices_for` の `shift_of`） |
| `tones`（コードの構成音で弾く音） | コードに従う | 実音のコードから鳴るので移さない |

以前は伴奏の指示書が「C 基準で書けば機械が移す」と言うのに、カバーではコード表を実音で見せ、書いた高さを移さずに鳴らしていた
（依頼文の末尾は「移調はしません」）。推定した調が外れても、見せるものと書いたものが同じ量だけ移るので鳴る高さどうしの関係はずれない
（推定と合い具合は記録と返事に出る・`song_cover_load.key_fit`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from song_arrange import arrange, fix_arrangement, to_spec    # noqa: E402
from song_design import CHORD_BAR, Section, chord_readable, grid_slots, transpose_chord   # noqa: E402
from song_checks import strip_ascii_ws                     # noqa: E402（半角の名前の前後の空白・§0-6・2026-10-03）
from song_place import keys_shift                           # noqa: E402（移調量の計算は新しい曲と同じ1か所）
from harmony import plan_harmony                            # noqa: E402
from instrumentation import InstrumentationStep, describe, instrument_name, plan_instruments   # noqa: E402
from song_model import Song, StepRecord                     # noqa: E402


#: この経路でやり直せる工程（順番どおり）。**カバーが持っているのはここだけ**——
#: 設計・グルーヴ・旋律・詞は**原曲が決めている**ので、作り直す対象が存在しない。
#: ★**コードも作り直せる**（2026-10-02・組「カバーのコード」・J6）。カバーのコードは LLM が書く段になった（`song_cover_chords`）ので、
#:   「サビのコードが違う」を `remake_from_step('コード', 頼み)` で頼み直せる。コードは編集の入口が用意し（`prepare_cover`）、
#:   ここは「コード」から始めても編成から作る（コードが変われば編成から先も作り直す）
STEPS = ("コード", "編成", "伴奏", "ハモリ")
#: この関数（`arrange_song`）が組む工程
_ARRANGE = ("編成", "伴奏", "ハモリ")


def arrange_song(song: Song, request: str, *, genre: str = "ミドルのポップス",
                 start: str = "編成", verbose: bool = True) -> Song:
    """編成 → 伴奏 → ハモリ を、**指示と過去の記録を渡して**やり直す。

    `start` から先だけを走らせる（2026-09-16 追加）。**カバーにも
    `remake_from_step` を通すため**——それまでは3工程が一塊で、「ハモリだけ付け直す」が
    できなかった（本人:「**カバーでもできるようにして**」）。
    既定は `編成` なので、`rearrange_song` の従来の呼び方は何も変わらない。

    ★この間は `song_context.CURRENT_SONG` にこの曲を置く——各工程の LLM 呼び出しに、前段の出力が丸ごと
      自動で付く（`song_context.upstream_intents`・2026-09-26）。新規作成と同じ仕組みに乗せる。
    """
    import song_context as _sc
    prev, _sc.CURRENT_SONG = _sc.CURRENT_SONG, song
    try:
        return _arrange_song_body(song, request, genre=genre, start=start, verbose=verbose)
    finally:
        _sc.CURRENT_SONG = prev


def _arrange_song_body(song: Song, request: str, *, genre: str, start: str, verbose: bool) -> Song:
    if start not in STEPS:
        raise ValueError(f"カバーで作り直せるのは {' / '.join(STEPS)}（{start!r} は原曲が決めている）")
    _from = _ARRANGE.index("編成" if start == "コード" else start)   # コードは呼び手が用意済み（`song_cover_chords.prepare_cover`）
    d = song.design or {}
    secs = [Section.model_validate(s) for s in (d.get("sections") or [])]
    ts = song.score.time_signature or "4/4"
    bpm = int(song.score.tempo)
    # ★**C 基準で見せ・書かせ、推定した調の分だけ移す**（2026-10-01・判-C）。移調量は新しい曲と同じ計算（`keys_shift`）
    key = str(d.get("key") or "")
    shift_of = keys_shift(song.key_of or {}, key)
    sh = shift_of(secs[0].kind) if secs else shift_of("")
    brief = (song.build_log() + f"\n\n■**マスターの指示**: {request}\n"
             "\n**旋律と歌詞は変えません**（原曲のまま）。伴奏・音色・ハモリで応えてください。"
             f"\n■**高さの物差し**: コード表・歌の音域・あなたが書く高さ（`pitches`・リフ・声部）は、新しく作る曲と同じ"
             f" **C メジャー／A マイナー基準（移調前）**です。この曲の調は原曲の旋律から推定した **{key or '（不明）'}** で、"
             f"機械が書いた高さを **{sh:+d}半音** 移して鳴らします（旋律・歌詞は原曲の高さのまま。"
             "コードはコードの工程が同じ C 基準で書き、同じ量だけ移して鳴らします）。"
             # ★カバーは主旋律の楽器（編成の `lead`・`lead_sections`）を書けない（2026-10-03・再レビュー 記A・仕様書 §15-4）
             "\n■**主旋律の楽器**: カバーでは編成の `lead`・`lead_sections` を書きません（主旋律は原曲の歌のまま。楽器では重ねない）。")
    c_chords = cover_c_chords(song, shift_of)
    c_range = c_basis_range(song, shift_of)

    # ⑥編成——`start` が先の工程でも、**編成が無ければ作る**（伴奏がそれを読むため）
    ip = getattr(song, "instrumentation", None)
    if _from <= 0 or ip is None or not hasattr(ip, "instruments"):
        if _from > 0 and verbose:
            print("  [編成] 前回の編成が残っていないので、ここも作り直します")
        ip = plan_instruments(song.score.title, genre, bpm, brief)
        ip, _ = InstrumentationStep(cover=True).run(ip, song, verbose=verbose)   # カバーは lead を書けない（記A・2026-10-03）
        song.instrumentation = ip
        song.instruments = [(i.bank, i.gm) for i in ip.instruments]
        song.allow_drums = strip_ascii_ws(ip.drum_kit) != "none"   # 綴りどおり（`none` は小文字だけ・§15-3。新しい曲の `song_band` と同じ読み方）
        song.record(StepRecord(step="instrumentation", reason=ip.reason, status=ip.status,
                               trouble=ip.trouble, request=request))
        if verbose:
            print("  [編成] " + " / ".join(f"{instrument_name(i)}" for i in ip.instruments)
                  + f"  ドラム={ip.drum_kit}")

    # ⑦伴奏
    if _from <= 1:
        ap = arrange(song.score.title, bpm, secs, c_chords, genre,
                     brief + "\n\n" + describe(ip), ts, vocal_range=c_range, shift_of=shift_of)
        ap, _ = fix_arrangement(ap, song, bars_of=song.bars_of, instruments=song.instruments,
                                allow_drums=song.allow_drums, slots=grid_slots(ts), inst_desc=describe(ip),
                                shift_of=shift_of,   # 移したあとの鳴る高さも工程で見る（`to_spec` と同じもの・2026-10-03）
                                verbose=verbose)
        warns: list[str] = []
        spec = to_spec(ap, song.bars_of, grid_slots(ts), song.instruments, song.allow_drums, warns,
                       shift_of=shift_of, drum_kit=ip.drum_kit)   # 書いた高さを推定した調へ移す（判-C）
        song.arrangement = ap
        song.score = song.score.model_copy(update={"arrangement": spec})
        song.record(StepRecord(step="arrange", reason=ap.reason, status=ap.status,
                               trouble=ap.trouble, warnings=warns, request=request))
        if verbose:
            print(f"  [伴奏] {ap.reason[:140]}")
            for w in warns:
                print(f"      ★{w}")
        # ★**伴奏の点検役を1回置く**（2026-10-03・判-27）。新しい曲にはある（`song_band.run_arrange_gate`）のに、カバーの編曲には無く、
        #   マスターの頼み（「和楽器風に」）と合っているか・薄い／厚いを見る目が無かった。やり直しは伴奏だけ（上限は `MAX_REDO`）
        _arrange_gate(song, ap, spec, warns, request, secs=secs, ts=ts, bpm=bpm, genre=genre, ip=ip,
                      brief=brief, c_chords=c_chords, c_range=c_range, shift_of=shift_of, verbose=verbose)

    # ⑧ハモリ
    # ★**編成が決めた声の種類を楽譜に写す**（2026-10-03・再レビュー）。新しい曲は `song_band.run_harmony_plan` が置き
    #   `song_place.make_score` が楽譜の `voice_choice` へ写すが、カバーは写しておらず、編成が選んだ声・歌い方・ハモリの声に
    #   関わらず既定の声で書き出していた（vpr へ焼くのは楽譜の `voice_choice` だけ・`score_to_vpr.voice_choice_for`）。
    #   写し方は新しい曲と同じ（`voice_style.set_choice` → `CHOICE` の写し）
    import voice_style as _vs
    _vs.set_choice(voice=ip.voice, style=ip.vocal_style, harmony_style=ip.harmony_style)   # 編成の型が必ず書かせる（2026-10-04）
    song.score = song.score.model_copy(update={"voice_choice": dict(_vs.CHOICE)})
    # ★**編成のハモリの方針（none / accent / full）を読む**（2026-10-03・再レビュー）。新しい曲と同じく none ならハモリの計画を呼ばない
    #   （`song_band.run_harmony_plan`）。以前は方針を読まずに毎回呼び、編成が「使わない」と決めた曲にも声部が付いた。
    #   方針は小文字の3つだけ（編成の検査 `InstrumentationStep` が一覧に無い値を所見にする）
    want_harmony = strip_ascii_ws(ip.harmony)
    if want_harmony == "none":
        why = "使わない" + (f"（{ip.harmony_reason}）" if ip.harmony_reason else "")
        hp, voices, hw = None, [], []
        song.voices = None
        if verbose:
            print(f"  [ハモリ] **使わない** — {ip.harmony_reason or '編成の判断'}")
    else:
        # 曲を渡すと区間名の直しの記録（`harmony_fix`）が残る（新しい曲と同じ・`harmony.plan_harmony`）
        hp = plan_harmony(song.score.title, secs, brief
                          + f"\n\n**ハモリの方針は『{want_harmony}』**"
                          + ("（**ここぞという所だけ**——ラスサビだけ等に絞る）"
                             if want_harmony == "accent" else "（サビでしっかり重ねる）"),
                          song=song, verbose=verbose)
        why = ""
        # ★**2段目を差し替えた**（2026-09-23）。主旋律を±2ずらすのでなく、LLM が位置も長さも書く
        from harmony import voices_for                      # noqa: E402
        # 形式の所見が直らなければ声部の工程で止まる（`song_fix.StepUnfixed`・2026-09-30。以前は捨てた音を記録して進んだ）
        # 声部も C 基準の表で見せ、書いた音を推定した調へ移す（判-C・新しい曲と同じ `shift_of`）
        voices, hw, song.voices = voices_for(hp, song.score, song=song, shift_of=shift_of)
    # ---- 奏法（2026-09-23）。**編曲をやり直したら奏法も引き直す**——
    #      前の編曲に付いていた位置は、新しいパターンでは音が無いかもしれない
    from song_flow import NOT_SWALLOW                       # noqa: E402（止める・途中で見せる・差し戻しは握らない）
    try:
        from artic_plan import plan as _artic_plan          # noqa: E402
        _a, _aw, song.articulation = _artic_plan(song.score, brief, verbose=verbose, song=song)
        song.score.arrangement.articulations = _a           # 伴奏が無ければパートが0本（F1）——奏法も空で来る
        hw = list(hw) + _aw
    except NOT_SWALLOW:                                         # 形式の所見が直らない・組み立ての不具合は止める（握らない・2026-09-30）
        raise
    except Exception as e:                                  # 奏法で曲を落とさない
        hw = list(hw) + [f"★奏法を書けなかった: {type(e).__name__}: {e}"]
    song.harmony = hp
    song.score = song.score.model_copy(update={"extra_voices": voices})
    if hp is None:                                          # 編成が「使わない」と決めた（新しい曲と同じ記録の形）
        song.record(StepRecord(step="harmony", reason=why, warnings=hw, request=request))
    else:
        song.record(StepRecord(step="harmony", reason=hp.reason, status=hp.status,
                               trouble=hp.trouble, warnings=hw, request=request))
    if verbose and hp is not None:
        print("  [ハモリ] " + (" / ".join(f"{v.title}({len(v.notes)}音)" for v in voices) or "なし"))
        for w in hw:
            print(f"      ★{w}")
    return song


# ---- 伴奏の点検役（2026-10-03・判-27）------------------------------------------------------------

def _arrange_gate(song: Song, ap, spec, warns: list[str], request: str, *, secs: list, ts: str, bpm: int,
                  genre: str, ip, brief: str, c_chords: dict, c_range: dict, shift_of, verbose: bool) -> None:
    """**カバーの伴奏を点検役に1回通し、「やり直し」なら伴奏だけ組み直す**（新しい曲の `song_band.run_arrange_gate` と同じ点検役）。

    材料は新しい曲と同じパターンの表（`song_band._arr_material`）に、**マスターの頼み**と主旋律の楽器の行（判-25）を足したもの。
    前の工程（編成）へは戻らない（`redo_from_options=[]`）。やり直しの上限・堂々巡りの止め方は点検役の呼び出し口
    （`song_gate.judge`・`MAX_REDO`）のまま。費用は点検役の呼び出しが `song_llm.run_sync`（ラベル `関門`）に乗せて数える。
    書く: （やり直したら）`song.arrangement`・楽譜の `arrangement`・`arrange` の記録。点検役の判断は `arrange_gate` の記録。
    """
    import song_context as ctx                              # noqa: E402（点検役の呼び出し口・再生が差し替える）
    from song_band import REDO_BASE_NOTE, _arr_material, arrange_facts, inspect_arrange   # noqa: E402（新しい曲と同じ材料と機械の所見）
    from song_gate_brief import redo_ask                    # noqa: E402（前回の指摘と「指摘への対応」の頼み）
    from song_gate import MAX_REDO                          # noqa: E402

    kit = ip.drum_kit
    title = song.score.title
    # テンポと拍子も見せる（新しい曲と同じ・2026-10-03 組G3。ジャンル・テンポとドラムの刻みの食い違いを点検役が見る）
    ctx_line = (f"カバー曲『{title}』（{genre}）／BPM {bpm}・拍子 {ts}／旋律と歌詞は原曲のまま・伴奏をマスターの頼みで組み直した／構成: "
                + " → ".join(f"{s.kind}(強さ{getattr(s, 'intensity', 3)})" for s in secs))
    inten = {s.kind: s.intensity for s in secs}

    def _judge(plan, sp, ws, attempt: int, before=None):
        # 機械が数えた表（区間ごとに鳴る楽器・歌の音域との重なり・ドラムの細かさ）は、楽譜の歌とコードにこの伴奏を置いて数える
        facts = arrange_facts(song.score.model_copy(update={"arrangement": sp}), song.bars_of, intensity_of=inten)
        # ★`before`/`after`（前の回の伴奏と新しい伴奏）を渡す（2026-10-04 組F4）。2回目以降の点検役に、前回の自分の判断・
        #   作業者の対応（`response`）・変わった所（`song_draft_diff`）が入る（`song_gate_brief.gate_brief` のカバーの枝）
        v = ctx.judge("伴奏", plan.reason, plan.status, plan.trouble, inspect_arrange(sp, song.bars_of, ws, secs),
                      attempt=attempt, verbose=verbose,
                      material=_arr_material(plan, song.instruments, c_range, kit, shift_of, ins=ip, request=request,
                                             facts=facts),
                      redo_from_options=[], context=ctx_line,  # 戻るのは伴奏だけ（編成へは戻らない）
                      before=before, after=plan)
        song.record(StepRecord(step="arrange_gate", reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                               status="ok" if v.action == "accept" else "compromised",
                               warnings=list(ws), request=v.action))
        return v

    v = _judge(ap, spec, warns, 0)
    att = 0
    while v.action == "redo" and att < MAX_REDO:
        att += 1
        guide = v.guidance or v.reason
        if verbose:
            print(f"  [関門] 伴奏を作り直す（{att}回目）: {guide[:100]}")
        # 捨てた案の直しの記録を、いまの伴奏のものとして後へ渡さない（新しい曲と同じ・名前を変えて残す）
        for r in song.history:
            if r.step in ("arrangement_fix", "arrangement_machine") and not r.reason.startswith("終わり方"):
                r.step = "arrangement_discarded"
        # ★**前回の伴奏を土台として渡し、指摘ごとの対応（`response`）を頼む**（2026-10-04 組F4・1‑2。新しい曲の伴奏と同じ口）。
        #   以前は指摘の全文だけで、前の伴奏も対応の頼みも無く、伴奏を一から書いていた
        before = ap
        with ctx.redo_base("伴奏", before):
            ap = arrange(title, bpm, secs, c_chords, genre,
                         brief + "\n\n" + describe(ip)
                         + f"\n\n★**前回の伴奏は作り直しになりました。**\n{redo_ask(guide)}\n{REDO_BASE_NOTE}", ts,
                         vocal_range=c_range, shift_of=shift_of)
        ap, _ = fix_arrangement(ap, song, bars_of=song.bars_of, instruments=song.instruments,
                                allow_drums=song.allow_drums, slots=grid_slots(ts), inst_desc=describe(ip),
                                shift_of=shift_of, verbose=verbose)
        warns = []
        spec = to_spec(ap, song.bars_of, grid_slots(ts), song.instruments, song.allow_drums, warns,
                       shift_of=shift_of, drum_kit=kit)
        song.arrangement = ap
        song.score = song.score.model_copy(update={"arrangement": spec})
        song.record(StepRecord(step="arrange", reason=ap.reason, status=ap.status,
                               trouble=ap.trouble, warnings=warns, request=guide))
        v = _judge(ap, spec, warns, att, before=before)


# ---- C 基準で見せる（2026-10-01・判-C）--------------------------------------------------------

def cover_c_chords(song: Song, shift_of) -> dict[str, list[str]]:
    """伴奏の依頼文の【コード】表（C 基準）。**コードの LLM が書いたもの（`song.chords`・C 基準）があればそのまま**見せる
    （2026-10-02・組「カバーのコード」）——実音から戻すと綴りが変わる（C 基準の `Bb` が D の曲で `C` になり、戻すと `A#`）。
    無ければ（コードの段を通っていない呼び方）区間ごとの実音（`chords_of`）を移調量だけ戻す。"""
    ch = getattr(song, "chords", None)
    secs = getattr(ch, "sections", None) if ch is not None else None
    if secs:
        return {s.kind: list(s.chords_per_bar) for s in secs}
    return c_basis_chords(song.chords_of or {}, shift_of)


def c_basis_chords(chords_of: dict[str, list[str]], shift_of) -> dict[str, list[str]]:
    """区間 → コードの並び（実音）を、**C 基準（移調前）で見せる形**へ（区間の移調量だけ戻す）。コードの段の出力（`song.chords`）が無いときだけ
    （`cover_c_chords`）。曲の中のコード（`song.chords_of`・`score.chords`）は実音で持ち（案A・2026-10-01 本人）、ここは見せるためだけ。
    1小節に複数あれば1つずつ戻す。和音を鳴らさない小節は `N.C.`（移すものが無いのでそのまま）。
    """
    out: dict[str, list[str]] = {}
    for kind, cells in (chords_of or {}).items():
        sh = int(shift_of(kind))
        row = []
        for cell in cells:
            names = CHORD_BAR.read(str(cell))
            # ★**空の小節・読めないコード名は止める**（2026-10-02）。コードはコードの段が書いたもの（`song_cover_chords`・検査済み）か
            #   尺の変更が置いたもので、どちらも読めるコード名しか作らない。来たら不具合。以前は空を `or [cell]` で、
            #   推定が和音の取れない小節に書いた `-` を「機械の印」として、どちらも書かれたまま見せていた（2026-10-02 本人「おすすめで」）。
            #   読めるかは組み立てと同じ部品（`song_design.chord_readable`）
            bad = [c for c in names if not chord_readable(c)]
            if not names or bad:
                from song_fix import AssemblyError
                raise AssemblyError("カバーのコードに" + (f"読めないコード名『{'・'.join(bad)}』" if bad else "空の小節")
                                    + f"が来た: 区間『{kind}』の『{cell}』（和音を鳴らさない小節は `N.C.`）")
            row.append(" ".join(transpose_chord(c, -sh) for c in names))
        out[kind] = row
    return out


def c_basis_range(song: Song, shift_of) -> dict[str, tuple[int, int]]:
    """区間 → 歌の最低音・最高音を、**C 基準（移調前）**で（伴奏の依頼文の歌の音域・新しい曲の `song_band` と同じ物差し）。

    以前はカバーの伴奏に歌の音域を渡していなかった（`pitches` の説明は「依頼文の歌の音域もこの物差しで出ています」と言っていた）。
    歌の音符は原曲の実音なので、区間の移調量を引く。区間は `song_cover_load.place_sections` の `sec_of_note`。
    """
    out: dict[str, tuple[int, int]] = {}
    notes = list(getattr(song.score, "vocal_notes", None) or [])
    secs = list(getattr(song, "sec_of_note", None) or [])
    for n, kind in zip(notes, secs):
        k = int(n.key) - int(shift_of(kind))
        lo, hi = out.get(kind, (999, -1))
        out[kind] = (min(lo, k), max(hi, k))
    return out
