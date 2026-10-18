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

"""既存曲の編集ツール（読み込み/参照配線/調整/区間置換/再配置/補正/参照整合/mix）。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext

from align_reference import align_score_lyrics
from align_reference import align_to_reference_jp as _align_jp
from align_reference import check_timing, fix_timing, load_reference
from align_cover import (accompaniment_for_section, align_section, lyrics_for_section,
                         render_vocal_and_mix, save_cut_source, section_from_audio, section_from_midi,
                         write_accompaniment)
from machine_eval import similarity
from song_llm import PITCH, generate_added_voice, pick_model, pitch_name_to_midi
from song_checks import JsonBool, JsonFloat, JsonInt  # LLM が書く数と真偽の欄の型・2026-10-02 判断5
from song_checks import strip_ascii_ws  # 半角の記法の前後の空白（§0-6・2026-10-03）
from song_table import _hold_shown, spell_pitch        # LLM に見せる・書かせる高さは国際式の音名、伸ばしは `_`（表と同じ・2026-10-01）
from song_fix import StepUnfixed, cut_note             # 形式の所見が直らずに工程で止まった合図（2026-09-30）・切ったら書く切り方
from song_flow import NOT_SWALLOW                      # 「失敗しても落とさない」except が握ってはいけない合図
from reference_to_wav import SR
from song_pipeline import compose
from song_schema import SongScore
from reference_fetch import fetch_audio_reference as _fetch_audio_ref
from reference_fetch import fetch_reference as _fetch_ref

import song_files                                      # 曲フォルダのファイルの名前の決まり（2026-10-03・完成品・ログ・印）
from session import (SONGS, MusicSession, _append_log, _trace, _traced, _locked, _save_section,
                     _save_ref_index, _save_accomp_bounds, _save_section_bounds, _write_fetch_log,
                     _clarify_msg, _build_log_digest, _slug, _song_status, _current_version,
                     begin_session, end_session, _snapshot_version_if_changed, _read_index, _update_index,
                     _retire_scoreless_session, _working_version, NEXT_EXPORT, retitle_song_folder)
from coordinator import coordinator


@coordinator.tool
@_traced
def load_existing_song(ctx: RunContext[MusicSession], song_id: str, version: JsonInt | None = None) -> str:
    """作成済みの曲を修正対象として読み込む（UC-2）。score＋前回の作成ログに加え、
    採用参照（source/reference.json）も自動復元して以降この曲フォルダ対象にする＝**修正で再fetch不要**。
    **『〜を作って/カバーして』という新規作成依頼ではこのツールを呼ばない**（既存一覧に同名の曲があっても新規作成する。
    既存曲を勝手に修正対象にした事故の再発防止）。使うのは『さっきの/その曲を直して・mixして』等の明示的な編集依頼だけ。
    **version=N を指定すると過去版 `score_v{N}.json` を非破壊で読み込む**（固定名ファイルは上書きしない＝どの版も消さない）。
    『v3に戻して直して』等に対応＝v3の内容を編集して次の確定で新版として保存（v3〜現在版はそのまま残る）。"""
    song_dir = SONGS / song_id
    # 曲の実体は版付き（`score_v{N}.json`）。固定名は作業中しか存在しないので、
    # ここで固定名だけを見ると「作業していない曲は全部“見つからない”」になる（2026-08-18）
    if not song_dir.exists() or not (
            (song_dir / "score.json").exists() or any(song_dir.glob("score_v*.json"))):
        # ★**途中で止めた曲は、ここでは開けない**（2026-09-23）。`draft_melody(show_after=...)` で
        #   手を止めた曲には**楽譜がまだ無い**ので、上の判定に落ちる。だが曲は生きていて、
        #   工程の記録（`_wip/<曲名>.json`）から **$0 で復元して続けられる**——入口が違うだけ。
        #   **「見つからない」で返すと、コーディネーターは作り直しを選ぶ**（実際にそうなった。
        #   2回試して両方この文言が返り、テーマ・設計・コード・グルーヴを作り直した＝余計な費用）。
        #   ★**行き先を返す。** 止めたときの案内は前のセッションの出力にしか無く、
        #   **次のセッションのコーディネーターは読んでいない**ので、失敗した場所で教える。
        if (song_dir / song_files.PROGRESS).exists():
            # ★**この曲を開いてから返す**（2026-09-25）。会話ごとに新しい作業フォルダが発番され、
            #   既存の曲へ切り替える手段はこの関数だけ。以前は**切り替える行（下の
            #   `ctx.deps.song_id = song_id`）より前で返していた**ので、案内どおり
            #   `remake_from_step` を呼んでも**新しい空のフォルダ**を見て、曲名が取れず断っていた。
            #   `show_after` で止めた曲の「次の会話で続き」も、同じ理由で届いていなかったはず。
            ctx.deps.song_id = song_id
            ctx.deps.record_version = None
            try:
                pg = json.loads((song_dir / song_files.PROGRESS).read_text(encoding="utf-8"))
            except Exception:
                pg = {}
            # ★**失敗で止まった曲も同じ入口で続けられる**（2026-09-25）。止め方が違うだけで、
            #   済んだ工程が `_wip` に残っているのは同じ。**作り直しを選ばせない**のが要点——
            #   ここで「見つからない」を返すと、テーマから全部作り直して費用を二重に払う。
            # ★**形式の所見が直らずに止まった曲**（2026-09-30）。止まった工程から、マスターの頼みを読んで続きを作る
            if isinstance(pg.get("stopped"), dict):
                _stp = pg["stopped"].get("step") or pg.get("next") or ""
                # 1件 120字・5件まで。★切ったら切ったと書く（2026-10-04 組F1・4‑1。全文は止めた印と記録に残っていて、
                #   続きから作るとき `song_remake.rebuild` が最終の点検役の NG を工程へ申し送る）
                _fs = [str(f) for f in (pg["stopped"].get("findings") or [])]
                _left = " / ".join(cut_note(f, 120) for f in _fs[:5]) + (f" ほか{len(_fs) - 5}件" if len(_fs) > 5 else "")
                if pg.get("unsaved"):                      # 途中保存が書けなかった＝続きの材料が無い（2026-10-03・再レビュー）
                    return (f"『{pg.get('title') or song_id}』は**{_stp}の工程で止まった曲**（形式の所見が、書き直させても残った: "
                            f"{_left}）。★止まったときに途中保存が書けなかったので、済んだ工程は残っていない——remake_from_step では"
                            "続きを作れない。作るなら draft_melody でもう一度（費用は最初から）。マスターに止まったことと所見を伝えてから決める。")
                if pg["stopped"].get("why", "").startswith("最終の点検役"):
                    # ★**最終の点検役の NG が残って止まった曲**（2026-10-04 組P2b・`song_fix.FinalUnpassed`）。全工程は済んでいて保存してある。
                    #   続きは最終の点検役が原因と見た工程から（選べなかったときは NG を読んで工程を選ぶ）
                    _where = (f"`remake_from_step(step='{_stp}', request=残った NG を直すように＋マスターの頼み)`" if _stp
                              else "残った NG を読んで原因の工程を選び、`remake_from_step(step=その工程, request=NG を直すように)`")
                    return (f"『{pg.get('title') or song_id}』は**最終の点検役の NG が残って止まった曲**（完成扱いにしていない。"
                            f"残った NG: {_left}）。このツールでは開けない（楽譜は書き出していない）。**{_where} で続きから作る**"
                            "——済んだ工程は費用なしで戻る。**新しく作り直さない**（済んだぶんの費用を二重に払うことになる）。"
                            + _requests_note(ctx, pg.get("title") or ""))
                if _stp == "テーマ":                       # テーマは作り直しの起点にならない（済んだ工程が無い・2026-09-30）
                    return (f"『{pg.get('title') or song_id}』は**テーマの工程で止まった曲**（形式の所見が、書き直させても残った: "
                            f"{_left}）。済んだ工程が無いので続きは作れない。作るなら draft_melody でもう一度（費用は最初から）。"
                            "マスターに止まったことと所見を伝えてから決める。")
                return (f"『{pg.get('title') or song_id}』は**{_stp}の工程で止まった曲**"
                        f"（書いたものの形式の所見が、差分で直させても残った: {_left}）。このツールでは開けない（楽譜がまだ無い）。"
                        f"**`remake_from_step(step='{_stp}', request=マスターの頼み)` で続きから作る**——済んだ工程は費用なしで戻り、"
                        f"{_stp}の工程が頼みを読んで書き直す（特に頼みが無ければ、上の所見が直るように頼む）。"
                        "**新しく作り直さない**（済んだぶんの費用を二重に払うことになる）。"
                        + _requests_note(ctx, pg.get("title") or ""))
            if pg.get("failed"):
                nxt = pg.get("next") or ""
                return (f"『{pg.get('title') or song_id}』は**途中で失敗して止まった曲**"
                        f"（{pg['failed'][:120]}）。このツールでは開けない（楽譜がまだ無い）。"
                        f"**済んだ工程は保存されているので、`remake_from_step` で続きから作る**"
                        + (f"——step は **{nxt}**（まだ作っていない最初の工程）。" if nxt else "。")
                        + "**新しく作り直さない**（済んだぶんの費用を二重に払うことになる）。"
                        + _requests_note(ctx, pg.get("title") or ""))
            return (f"『{song_id}』は**途中まで作って手を止めた曲**なので、このツールでは開けない"
                    f"（楽譜がまだ無い）。**`remake_from_step` を使う**——工程の記録から"
                    f"**作り直しの費用なしで**復元して、指定した工程から続きを作れる。"
                    f"曲名は {song_files.PROGRESS} にある。" + _requests_note(ctx, pg.get("title") or ""))
        return f"見つからない: {song_dir}"
    begin_session(song_dir)                                # セッション開始: 作業する版を決め、現在版を作業台（固定名）へ展開する
                                                           # （前回が途中で終わっていれば、その版の続きから作業する）
    p = song_dir / "score.json"                            # begin_session が展開済み＝ここから先は従来どおり固定名で扱える
    load_path, note = p, ""
    if version is not None:                                # 過去版を**非破壊で**読込（固定名は上書きしない＝現在版を消さない）
        vp = song_dir / f"score_v{version}.json"
        if not vp.exists():
            avail = sorted(int(m.group(1)) for q in song_dir.glob("score_v*.json")
                           if (m := re.match(r"score_v(\d+)\.json$", q.name)))
            return (f"v{version} が見つからない（score_v{version}.json なし）。利用可能な版: {avail}\n"
                    f"[版の記録（作った日時・直した内容）]\n{song_files.versions_text(song_dir, last=len(avail))}")
        load_path = vp
        note = (f"\n[版読込] v{version} を**非破壊で**読込（固定名ファイルは未変更＝現在版・他版とも消さない）。"
                f"これを編集→次の確定で新版として保存される。")
    ctx.deps.score = SongScore.model_validate_json(load_path.read_text(encoding="utf-8"))
    if version is not None and ctx.deps.score.form != "instrumental":     # インストは vpr を持たない（§4-4）
        note += f"VOCALOIDで見るなら {song_files.versioned(song_files.VPR, version)} を開く。"
    # ★**読んだ時点の姿を控える**（2026-09-20）。セッション末尾の命綱が
    #   「編集したのに書き出し忘れた」と「読んだだけ」を区別できるようにする。
    #   控えが無かったころ、**調べただけの曲が作り直された**（→ session.py の score_at_load）。
    ctx.deps.score_at_load = ctx.deps.score.model_dump_json(indent=2)
    ctx.deps.song_id = song_id                             # 以降の修正/ログ/書出はこの曲フォルダ対象
    # ★**記録も開いた版のものを使う**（2026-09-25・段階3）。作業台は書き換えず、
    #   最初に記録を書くとき（編集・作り直し）に `record_v{N}.json` から始める。
    ctx.deps.record_version = version
    # 編集の元にした版（編集のあとの最終の点検が「前の版から変わった所」の伴奏 MIDI を引く・2026-10-04・案C）
    ctx.deps.edit_base_version = version or _current_version(song_dir) or None
    ctx.deps.reference_json = ctx.deps.reference_midi = None
    ctx.deps.melody_channel = ctx.deps.reference_audio_accomp = None
    ctx.deps.faithful = False
    msg = f"読込: 『{ctx.deps.score.title}』{len(ctx.deps.score.vocal_notes)}音"
    if ctx.deps.score.form != "song":                      # 形を先に見せる（形に合わない道具は断られる・§7-2）
        msg += (f"（この曲は{song_files.FORM_LABEL[ctx.deps.score.form]}。"
                + ("歌は無く、主旋律は楽器が弾く" if ctx.deps.score.form == "instrumental" else "伴奏は無く、声だけ")
                + "。形を変えるなら change_form）")
    ref_idx = song_dir / "source" / "reference.json"       # 採用参照を復元（再fetch=LLM不要・同じ原曲素材で修正）
    if ref_idx.exists():
        import json as _json
        try:
            r = _json.loads(ref_idx.read_text(encoding="utf-8"))
            sd = song_dir / "source"
            if r.get("type") == "midi" and (sd / r["midi"]).exists():
                ctx.deps.reference_midi = str(sd / r["midi"])
                ctx.deps.melody_channel = r.get("melody_channel")
                from midi_to_reference import build_reference       # reference_json は solfa.json（.mid でない）
                ref = build_reference(str(sd / r["midi"]), r.get("melody_channel"))
                ctx.deps.reference_json = ref["reference_json"]
                ctx.deps.faithful = True                   # カバー＝忠実コピー（render_finalで合成伴奏を再生成させない）
                msg += f"\n参照復元(MIDI): source/{r['midi']} ch{r.get('melody_channel')}"
            elif r.get("type") == "audio" and (sd / r["solfa"]).exists():
                ctx.deps.reference_json = str(sd / r["solfa"])
                if r.get("accomp") and (sd / r["accomp"]).exists():
                    ctx.deps.reference_audio_accomp = str(sd / r["accomp"])
                ctx.deps.faithful = True                   # 音声カバーも実楽器伴奏を保持（再生成しない）
                msg += f"\n参照復元(音声): source/{r['solfa']}（再fetch不要）"
            # 中間生成物（区間情報・歌詞）を復元＝**align_vocal を再実行して再整列できる**（カバーのタイミング修正はこれ）
            sec_p, lyr_p = sd / "section.json", sd / "lyrics.txt"
            if sec_p.exists():
                sec = _json.loads(sec_p.read_text(encoding="utf-8"))
                sec["log"] = []
                if sec.get("vocals_wav"):                  # パスを現フォルダへ修正（render_final の rename 後でも解決）
                    sec["vocals_wav"] = str(sd / Path(sec["vocals_wav"]).name)
                if sec.get("accomp_wav"):
                    sec["accomp_wav"] = str(sd / Path(sec["accomp_wav"]).name)
                ctx.deps.section = sec
                msg += "\n区間(中間状態)復元→**align_vocal を再実行すれば再整列**できる（タイミング修正はこれ・記号整列は使わない）"
            if lyr_p.exists():
                # ★**いまの歌詞のかなの決まりに合わない保存（2026-10-02 より前の、空白・改行入り）は読み込まない**（案B）。
                #   整列の読み手はもう空白・句読点を落とさないので、黙って直さずに取り直しを頼む
                from song_lyrics import kana_text_problems     # noqa: E402
                _lk = lyr_p.read_text(encoding="utf-8")
                if _bad := kana_text_problems(_lk):
                    msg += (f"\n保存された歌詞（source/lyrics.txt）はいまの歌詞のかなの決まりに合わないので読み込まなかった"
                            f"（{_bad[0]}）——再整列するときは先に fetch_section_lyrics で歌詞を取り直す")
                else:
                    ctx.deps.lyrics = _lk.strip()
            ab_p = sd / "accomp_bounds.json"               # 伴奏を最後に切った区間を復元→編集セッションでも stale 検知が働く
            if ab_p.exists():                              # （railgun実害: 再整列で始端が動いたのに伴奏が旧区間のまま＝約1秒ズレ）
                ab = _json.loads(ab_p.read_text(encoding="utf-8"))
                ctx.deps.accomp_bounds = (ab.get("start_sec"), ab.get("end_sec"))
        except Exception:
            pass
    digest = _build_log_digest(song_dir)                   # 前回の作り方＝修正の手がかり（巨大なノート列は除外し経緯の要点を渡す）
    if digest:
        msg += ("\n--- 前回の作成経緯（build_log 要点・区間特定/歌詞/再整列/整合/隙間/評価） ---\n"
                f"{digest}\n--- ここまで。これを読んで何をどう直すか判断する ---")
    # ★**版の記録を見せる**（2026-10-11・本人了承）。「朝の版」「直す前の版」と頼まれたとき、どの版か番号で決められるように
    vers = song_files.versions_text(song_dir, last=6)
    if vers:
        msg += f"\n[版の記録（新しい6つ・作った日時と直した内容）]\n{vers}"
    # ★**見せる歌詞（字幕）の古い行を知らせる**（2026-10-14）。手直しで歌う詞だけ変わった行が、頼まれても直らずに残っていた
    try:
        from song_edits import numbered_display, score_bar_ticks, stale_display_lines
        _rec = _record_of(ctx)
        _stale = stale_display_lines(_rec, ctx.deps.score) if _rec is not None else []
        if _stale:
            msg += ("\n★見せる歌詞（字幕・投稿用）が、実際に歌う詞と食い違っていそうな行: "
                    + "・".join(f"{i}行" for i in _stale)
                    + "（手直しで歌う詞だけ変わった）。直すなら set_display_lyrics で行番号を指して書き直す"
                    "（歌う詞は変えない）。このセッションで直さないなら、最終報告にそう書く:\n"
                    + numbered_display(_rec, ctx.deps.score, score_bar_ticks(ctx.deps.score)))
    except Exception:                                      # noqa: BLE001（知らせられなくても開くのは止めない）
        pass
    return msg + note + _requests_note(ctx)


# ---- 頼まれごとの台帳（2026-09-25・段階2→3）----------------------------------------------
# 編集の道具は、頼まれたことを**曲の工程の記録の台帳にも書く**。楽譜だけを書き換えると、
# あとで作り直したときに**頼んだことが黙って消える**（キーを上げてから詞を直すとキーが戻る）。
# 段階3から記録は**曲フォルダに版ごと**（`record.json` → `record_v{N}.json`・session.load_record）。
# 速さ・移調・声部・音符には**機械が当て直すための値**（`value`）も残す（`song_edits.apply_edits`）。

def _title_of(ctx: RunContext[MusicSession]) -> str:
    return (ctx.deps.score.title if getattr(ctx.deps, "score", None) else "") or ""


def _record_of(ctx: RunContext[MusicSession], title: str = ""):
    """この曲の工程の記録（開いた版のもの）。無ければ None。"""
    from session import load_record
    return load_record(SONGS / ctx.deps.song_id, title or _title_of(ctx),
                       getattr(ctx.deps, "record_version", None))


def _added_voice_basis(ctx: RunContext[MusicSession], transpose: int = 0):
    """ハモリ追加（`generate_added_voice`）に渡す物差し `(shift_of, bars_of, keys)`。**決められなければ理由の文字列**。

    ★**C 基準で見せて書かせ、区間の調へ移す**（2026-10-02・仕様書 §0-2・ず-12①）。移調量は声部の工程と同じ計算
      （`song_place.keys_shift`＝区間の調 → 移調量）に、曲の記録の区間の調（`Song.key_of`。オリジナルは組み立てが決めた調・
      カバーは原曲から推定した調）と区間の位置（`Song.bars_of`）を渡す——2本目の計算を作らない。
    ★**あとから移調した曲**（台帳の生きている移調 `組み立て.移調` と、この呼び出しの `transpose`）は、記録の調からその分ずれて
      鳴っているので、移調量にその分を足す（見せる高さと書いた音を移す量が同じなので、足し損ねても音は主旋律と合う）。
      その曲では記録の調の名前が今の調と違うので、依頼文に調の名前は出さない（移す量だけ）。
    ★**分からない曲（記録が無い・区間の調か位置が無い・調が読めない）は黙って 0 にしない**——理由を返し、呼び手が足さずに止める。
    """
    from song_edits import ASSEMBLY_TRANSPOSE
    from song_place import keys_shift
    from song_table import key_regions
    try:
        song = _record_of(ctx)
    except Exception as e:                                 # 記録が読めない＝物差しが決められない（足さずに止める）
        return f"この曲の記録が読めず（{type(e).__name__}）、ハモリを書かせる高さの物差しが決められない"
    if song is None or not getattr(song, "key_of", None) or not getattr(song, "bars_of", None):
        return ("この曲には区間の調と区間の位置の記録（作ったときの記録 record の key_of・bars_of）が無く、"
                "ハモリを書かせる高さの物差し（C 基準から区間の調へ移す量）が決められない")
    d = song.design or {}
    key = d.get("key") if isinstance(d, dict) else getattr(d, "key", "")
    try:
        base = keys_shift(dict(song.key_of), str(key or ""))
        base("")                                           # 区間名の無い所の既定（設計の調）も読めるか
        for k in song.key_of:
            base(k)
    except ValueError as e:
        return f"記録の区間の調が読めず、ハモリを書かせる高さの物差しが決められない（{e}）"
    offset = int(transpose or 0) + sum(int((r.value or {}).get("semitones", 0)) for r in song.requests or []
                                       if r.status == "active" and r.target == ASSEMBLY_TRANSPOSE)
    keys = (key_regions(song) or None) if offset == 0 else None
    return (lambda k: base(k) + offset), dict(song.bars_of), keys


def _cover_lacks_skeleton(ctx: RunContext[MusicSession], song_dir: Path) -> bool:
    """**区間の調と位置（`key_of`・`bars_of`）を記録に持たないカバー**か（2026-10-03・再レビュー）。

    新しく作ったカバー（`draft_melody` → … → `render_final`）は区間と調を記録に残さない。ハモリ追加の物差しはそれを読むので、
    以前は「記録が無い」と断っていた（断りの文は「新しく取り込んだカバーなら記録がある」と事実と違った）。カバーなら
    カバーのコードの入口（`song_cover_chords.prepare_cover(chords=False)`＝区間と調だけ）で用意できる。
    カバーかどうか: 台帳だけの記録（テーマが無い）があればカバー。記録が無ければ、参照（原曲のお手本）を持つ曲。"""
    try:
        rec = _record_of(ctx)
    except Exception:                                      # 記録が読めない＝用意もできない（物差しの側が理由を返す）
        return False
    if rec is not None and getattr(rec, "key_of", None) and getattr(rec, "bars_of", None):
        return False
    if _is_original(rec):                                  # オリジナル曲の区間と調は工程の記録が持つ（無ければ古い曲＝足せない）
        return False
    return rec is not None or bool(getattr(ctx.deps, "faithful", False)) \
        or (song_dir / "source" / "reference.json").exists()


def _is_original(song) -> bool:
    """工程の記録を持つ曲か（カバーは台帳だけの記録になる＝テーマが無い）。"""
    return song is not None and song.theme is not None


# ---- 曲の形と、形に合わない道具（2026-10-03・`docs/plan/song_acappella_instrumental.md` §7-2）---------------
# ★第1段は**形に合わない道具は理由を返して断る**（黙って作らない・黙って足さない）。断るときは何も書き換えない。
#   表はここ1か所（道具 → 形 → 断る理由と行き先）。診断・伴奏・手直しの道具もここを読む（`form_refusal`）。

def form_of(ctx: RunContext[MusicSession]) -> str:
    """いま開いている曲の形（`song`／`acappella`／`instrumental`）。楽譜（`SongScore.form`）が正、楽譜が無ければ曲の記録
    （`Song.form`）、どちらも無ければ `song`（欄の無い古い曲も song）。"""
    s = getattr(ctx.deps, "score", None)
    if s is not None:
        return getattr(s, "form", "song") or "song"
    try:
        rec = _record_of(ctx)
    except Exception:
        rec = None
    return (getattr(rec, "form", "song") or "song") if rec is not None else "song"


_NO_VOCAL_WHY = ("インストなので歌が無い（主旋律は楽器が弾く）——この道具は歌の音符を直す・調べるもの。"
                 "主旋律を変えるなら remake_from_step('旋律', 頼み)、歌を乗せるなら change_form('song', 頼み)")

#: **道具 → 形 → 断る理由と行き先**（§7-2 の表の「断る」の行）。ここに無い組み合わせはいまのまま動く。
#: 尺の変更（`resize_song`）の断りは `song_resize.form_refusal`（尺の変更の部品の側）、診断の「対象外」は各道具が返す
FORM_REFUSE: dict[str, dict[str, str]] = {
    "replace_region": {"instrumental": _NO_VOCAL_WHY},
    "redistribute_by_gaps": {"instrumental": _NO_VOCAL_WHY},
    "fix_song_timing": {"instrumental": _NO_VOCAL_WHY},
    "find_cramped_regions": {"instrumental": _NO_VOCAL_WHY},
    "find_misplaced_regions": {"instrumental": _NO_VOCAL_WHY},
    "adopt_vocal_export": {"instrumental": "インストなので歌声の書き出しは無い——完成品（master_inst_v<版>.wav）は render_final が作る"},
    # ★2026-10-03・再レビューで漏れを足した。render_and_mix はインストでも vpr を書こうとして例外で落ち、inspect_region は歌0音を調べていた
    "render_and_mix": {"instrumental": ("インストなので歌声の楽譜・vpr・仮歌は無い（この道具はそれを書き出すもの）——"
                                        "伴奏と完成品（master_inst_v<版>.wav）は render_final が作る")},
    "inspect_region": {"instrumental": _NO_VOCAL_WHY},
    "rearrange_song": {"acappella": (
        "アカペラなのでアレンジのやり直し（編成→伴奏→ハモリ）はしない——伴奏を作ってしまう。"
        "声だけで響きを変えるなら remake_from_step('ハモリ', 頼み)（声の種類なら '編成'）、"
        "伴奏を付けたいなら change_form('song', 頼み)")},
    "adjust_song.vocal_lead_db": {
        "acappella": "アカペラなので伴奏が無く、歌を伴奏より前に出す量（vocal_lead_db）は意味が無い——声部どうしの音量は adjust_voice(gain_db=…)",
        "instrumental": "インストなので歌が無く、歌を伴奏より前に出す量（vocal_lead_db）は意味が無い"},
    "adjust_song.add_voice": {"instrumental": (
        "インストなので重ねる歌（主旋律）が無く、声部（ハモリ）は足せない——歌を乗せるなら change_form('song', 頼み)")},
}


def form_refusal(ctx: RunContext[MusicSession], tool: str, head: str = "何も変えなかった") -> str:
    """形に合わない道具なら、断る文（`head`＝何も変えなかった／調べなかった と、理由・行き先）。合えば空。"""
    form = form_of(ctx)
    why = FORM_REFUSE.get(tool, {}).get(form, "")
    if not why:
        return ""
    _trace(ctx.deps.song_id, f"{tool.split('.')[0]}.refused", reason=f"form-{form}")
    return f"{head}: この曲は{song_files.FORM_LABEL[form]}。{why}。"


def lead_refusal(ctx: RunContext[MusicSession], tool: str, score=None) -> str:
    """**主旋律を楽器でも重ねている歌もの**（`SongScore.lead`・F10②）で、歌の音符だけを変える道具なら断る文。そうでなければ空。

    楽器の主旋律は旋律の工程の音そのもの（楽譜の `lead`）なので、歌の音符だけ書き換える・並べ直すと、楽器は古い旋律のまま鳴る
    （黙って古いまま残さない・2026-10-03）。旋律の工程から作り直せば、楽器の主旋律も一緒に作り直る。"""
    s = score if score is not None else getattr(ctx.deps, "score", None)
    lead = s.lead if s is not None else None
    if lead is None or lead.is_empty():                    # 楽譜が無い／主旋律の楽器が無い（中身が空・F1）
        return ""
    _trace(ctx.deps.song_id, f"{tool}.refused", reason="lead-overlay")
    where = "・".join(lead.sections) if getattr(lead, "sections", None) else "主旋律のある区間の全部"
    how = ("尺・構成から変えるなら remake_from_step('設計', 頼み)" if tool == "resize_song"
           else "旋律を直すなら remake_from_step('旋律', 頼み)")
    return (f"何も変えなかった: この曲は主旋律を楽器でも重ねて弾いている（{where}）。{tool} で歌の音符だけを変えると、"
            f"楽器の主旋律が古い旋律のまま残って歌とずれる。{how}（楽器の主旋律も一緒に作り直る）。"
            "重ねるのをやめるなら remake_from_step('編成', 頼み)。その旨をマスターに伝える。")


def finish_hint(form: str) -> str:
    """直したあとの「どうすれば完成品になるか」（形で分ける）。完成品の名前の形は `song_files` の決まり（版の番号は書き出すときに決まる）。"""
    from session import next_step                          # 次にやることの文は形ごとに1か所（`session.NEXT_BY_FORM`）
    name = f"{song_files.MASTER_PREFIX}{song_files.FORM_TAG[song_files.as_form(form)]}_v<版>.wav"
    return f"{next_step(form)}（完成品は {name}）"


# ---- 編集のあとの最終の点検（2026-10-04・案C・本人決定。`docs/plan/song_final_reviewer.md`「追記: 編集のあとの点検」）-------------
# ★**頼みの大きさで切り替える**。物差しは「曲の中の音を、誰かが新しく書いたか・曲全体の土台を動かしたか」（道具と引数だけで機械が決める）:
#   大＝LLM・コーディネーターが音を新しく書いた／移調・テンポで曲全体の土台を動かした（`replace_region` は長さに関係なく大・テンポは常に大）
#   中＝引き算・高さの移しだけ（単独では通さない。同じ頼みに大があれば、まとめて見せる）
#   小＝音量だけ・機械の決まった直し（通さない・束に積まない）
#   対象外＝人の手直しの取り込み・手で書き出した歌声の取り込み・参照の付け外し・尺の変更（記録の時間軸が古いので後回し）
# ★**表はここ1か所**（道具ごとに散らさない）。コーディネーターの道具は全部に行を持つ——`tests/song_service/test_edit_gate.py` が
#   登録された道具を数え、行の無い道具があれば落とす（道具を足したら、ここに行を足す）。キーは「道具」か「道具.引数」。
EDIT_BIG, EDIT_MID, EDIT_SMALL, EDIT_OUT = "大", "中", "小", "対象外"
#: 曲を変えない道具（読む・調べる）と、ほかの流れが受け持つ道具（区分の対象にならない）
EDIT_READ = "読むだけ"
EDIT_MAKE = "作る（新しい曲・カバーの作成工程）"
EDIT_REMAKE = "作り直し（中で最終の点検役が走る。カバーは編曲の点検役）"
EDIT_EXPORT = "書き出し（ここで点検待ちの束を見る）"
EDIT_SIZE: dict[str, str] = {
    # ---- 大・中・小（点検待ちの束に積むのは大と中）----
    "adjust_song.tempo": EDIT_BIG,                         # 曲全体の土台（テンポは常に大）
    "adjust_song.transpose": EDIT_BIG,                     # 曲全体の土台
    "adjust_song.add_voice": EDIT_BIG,                     # LLM が声部を書く（差し替えも）
    "adjust_song.vocal_lead_db": EDIT_SMALL,               # 音量だけ
    "adjust_voice.transpose": EDIT_BIG,                    # ±12 以外の幅は声部の和声が変わる
    "adjust_voice.transpose_octave": EDIT_MID,             # ±12（オクターブ移し）
    "adjust_voice.remove": EDIT_MID,                       # 引き算
    "adjust_voice.gain_db": EDIT_SMALL,                    # 音量だけ
    "replace_region": EDIT_BIG,                            # コーディネーターが歌の音を書く（長さに関係なく）
    "replace_accompaniment_region.notes": EDIT_BIG,        # 伴奏の区間に音を置く
    "replace_accompaniment_region.program": EDIT_BIG,      # 音色を変える
    "replace_accompaniment_region.new_track": EDIT_BIG,    # 新しいトラック（楽器を足す）
    "replace_accompaniment_region.silence": EDIT_MID,      # notes=[]（区間を無音にする・引き算）
    "remove_accompaniment_track": EDIT_MID,                # 楽器を外す（引き算）
    "redistribute_by_gaps": EDIT_SMALL,                    # 機械の決まった直し（位置だけ）
    "fix_song_timing": EDIT_SMALL,                         # 機械の決まった直し（格子寄せ）
    #: 作り直しのあとに台帳の手直し（移調・速さ・声部・音符）と人の手直しを当て直した（§3.7。作り直しの中の最終の点検役は当て直す前の
    #: 楽譜を見ているので、当て直した姿を書き出しの前にもう一度見る）。道具ではない行
    "当て直し": EDIT_BIG,
    # ---- 対象外（通さない）----
    "take_in_manual_edits": EDIT_OUT, "adopt_vocal_export": EDIT_OUT, "set_reference": EDIT_OUT,
    "resize_song": EDIT_OUT,                               # 記録の時間軸が古いまま（材料が作れない）。当面は対象外（本人決定 2026-10-04）
    # ---- 区分の外 ----
    "remake_from_step": EDIT_REMAKE, "change_form": EDIT_REMAKE, "rearrange_song": EDIT_REMAKE,
    "draft_melody": EDIT_MAKE, "fetch_section_lyrics": EDIT_MAKE, "align_vocal": EDIT_MAKE, "build_accompaniment": EDIT_MAKE,
    "fetch_reference": EDIT_MAKE, "read_humming": EDIT_MAKE,
    "render_and_mix": EDIT_EXPORT, "render_final": EDIT_EXPORT, "export_vocal": EDIT_EXPORT,
    "load_existing_song": EDIT_READ, "research_style": EDIT_READ, "recall_making": EDIT_READ, "review_manual_edits": EDIT_READ,
    "review_progress": EDIT_READ, "evaluate_song": EDIT_READ, "evaluate_timing": EDIT_READ, "find_cramped_regions": EDIT_READ,
    "find_misplaced_regions": EDIT_READ, "inspect_song": EDIT_READ, "inspect_region": EDIT_READ, "check_mix_quality": EDIT_READ,
    "check_consistency": EDIT_READ, "inspect_accompaniment": EDIT_READ,
    # 2026-10-14: 音を変えない記録の道具（続いている頼みを残す・見せる歌詞の行だけ直す）＝点検の束に積まない
    "keep_request": EDIT_READ, "set_display_lyrics": EDIT_READ,
}
_SIZE_RANK = {EDIT_BIG: 2, EDIT_MID: 1}
#: **伴奏を MIDI（作業台の `accompaniment.mid`）で直接書き換える道具**。楽譜の `arrangement` を変えない（`remove_accompaniment_track`
#: は名前で外すときだけ楽譜からも外す）ので、束にこれがあれば、編集のあとの最終の点検役の表の伴奏は MIDI から数える（2026-10-04・
#: 実機『水たまりのプリズム』v2——楽譜から数えて、直したギターを「まだ歌と同じ高さ」と見せ、誤った NG になった）
ACCOMP_MIDI_TOOLS = ("replace_accompaniment_region", "remove_accompaniment_track")


def edit_size(*keys: str) -> str:
    """表の行（`EDIT_SIZE` のキー）の並び → この呼び出しの区分（大 > 中 > 小）。大・中・小の行が無ければ空。表に無いキーは不具合。"""
    sizes = [EDIT_SIZE[k] for k in keys]                    # 無いキーは KeyError（表に足し忘れた）
    got = [s for s in sizes if s in (EDIT_BIG, EDIT_MID, EDIT_SMALL)]
    return max(got, key=lambda s: _SIZE_RANK.get(s, 0)) if got else ""


#: 大の編集の返事に添える一文（オリジナル曲／カバー）
EDIT_GATE_NOTE = "（この編集は、書き出しの前に最終の点検役がまとめて見る）"
COVER_REVIEW_NOTE = ("★カバーなので最終の点検役は通らない——大きい編集のあとは、書き出し（export_vocal）の前に "
                     "**review_progress（最終検問）を必ず呼ぶ**。")


def note_edit(ctx: RunContext[MusicSession], keys, what: str) -> str:
    """編集の道具の終わりに、**点検待ちの束へ1件積む**（大・中だけ。小・対象外は積まない）。返事に添える一文を返す（無ければ空）。

    `keys` … `EDIT_SIZE` の行（1つか並び）／ `what` … 何をしたか（道具の返事の頭など・1行）。
    カバー（工程の記録を持たない曲）は最終の点検役を通らないので、大なら最終検問（`review_progress`）の案内を返す（束にも積む——
    書き出しの入口で、カバーだと分かったら呼ばずに捨てる）。"""
    keys = [keys] if isinstance(keys, str) else list(keys)
    size = edit_size(*keys)
    if size not in (EDIT_BIG, EDIT_MID):
        return ""
    tool = keys[0].split(".")[0]
    ctx.deps.pending_edits.append({"tool": tool, "size": size, "keys": keys, "what": " ".join((what or "").split())[:300]})
    _trace(ctx.deps.song_id, "edit_gate.pending", tool=tool, size=size, n=len(ctx.deps.pending_edits))
    if size != EDIT_BIG:
        return ""
    try:
        cover = not _is_original(_record_of(ctx))
    except Exception:
        cover = False
    return COVER_REVIEW_NOTE if cover else EDIT_GATE_NOTE


def _did_lines(pend: list[dict]) -> list[str]:
    return [f"{p['tool']}（{p['size']}）: {p['what']}" for p in pend]


def _base_files(ctx: RunContext[MusicSession], song_dir: Path):
    """前の版の楽譜と、前の版・いまの伴奏 MIDI（比べる相手）。前の版が無ければ楽譜は None。
    楽譜は読み込んだときの姿（`score_at_load`）が正——過去の版を開いて編集したときも、その版と比べる。無ければ前の版の楽譜ファイル。"""
    prev = None
    if getattr(ctx.deps, "score_at_load", None):
        prev = SongScore.model_validate_json(ctx.deps.score_at_load)
    base = getattr(ctx.deps, "edit_base_version", None) or _current_version(song_dir)
    if prev is None and base and (p := song_dir / song_files.versioned(song_files.SCORE, base)).exists():
        prev = SongScore.model_validate_json(p.read_text(encoding="utf-8"))
    mid0 = song_dir / song_files.versioned(song_files.ACCOMP_MID, base) if base else None
    return prev, mid0, song_dir / song_files.ACCOMP_MID


def review_edits(ctx: RunContext[MusicSession], where: str) -> str:
    """**書き出しの入口で、点検待ちの束を見て、最終の点検役を1回呼ぶ**（2026-10-04・案C）。返事に添える文を返す（呼ばなければ空）。

    `where` … 呼んだ所（記録用: export_vocal / render_final / 伴奏の編集 / 実行の終わり）。
    - 束に大が無ければ呼ばずに束を捨てる（中だけの頼みは通さない）
    - カバー・工程の記録や時間軸の無い曲は呼ばない（カバーは最終検問 `review_progress` の受け持ち）
    - 1回の頼みで2回まで（`song_final_gate.EDIT_MAX_RUNS`）。使い切ったら呼ばずに「点検はここまで・残った NG」を返す
    - **NG でも止めない**（書き出しは続ける）。返事に NG と直す案（提案）を載せる。判断は記録 `final_gate_edit` に残す
    """
    pend = list(ctx.deps.pending_edits or [])
    if not pend:
        return ""
    ctx.deps.pending_edits = []
    if not any(p["size"] == EDIT_BIG for p in pend):
        _trace(ctx.deps.song_id, "edit_gate.skip", where=where, reason="中だけ")
        return ""
    import song_final_gate as fin
    sid = ctx.deps.song_id
    song_dir = SONGS / sid
    try:
        song = _record_of(ctx)
    except Exception as e:
        song, why = None, f"記録が読めない（{type(e).__name__}）"
    else:
        why = ""
    if song is None or not _is_original(song):
        _trace(sid, "edit_gate.skip", where=where, reason=why or "カバー・記録なし")
        return "" if not why else f"\n【最終の点検役（編集のあと）】呼ばなかった: {why}。"
    if getattr(song, "timeline", None) is None:
        _trace(sid, "edit_gate.skip", where=where, reason="時間軸なし")
        return ("\n【最終の点検役（編集のあと）】呼ばなかった: この曲の記録に区間の時間軸が無い（古い曲）ので、曲全体の表が作れない。"
                "大きい編集の仕上がりは点検していない——最終報告でマスターにそう伝える。")
    if ctx.deps.edit_gate_runs >= fin.EDIT_MAX_RUNS:
        last = ctx.deps.edit_gate_log[-1] if ctx.deps.edit_gate_log else "（なし）"
        _trace(sid, "edit_gate.skip", where=where, reason="回数を使い切った")
        return (f"\n【最終の点検役（編集のあと）】この頼みで点検できるのは{fin.EDIT_MAX_RUNS}回まで——**点検はここまで**（今回の直しは見ていない）。"
                f"前の点検で残った判断: {last}。最終報告でマスターに、点検は使い切ったこと・残った NG を伝える。")
    score = ctx.deps.score
    try:
        from song_draft_diff import score_changes
        prev, mid0, mid1 = _base_files(ctx, song_dir)
        changes = score_changes(prev, score, song.timeline, before_mid=mid0, after_mid=mid1)
        v = fin.judge_edit(song, score, instruction=getattr(ctx.deps, "instruction", "") or "",
                           ids=list(ctx.deps.run_request_ids or []), did=_did_lines(pend), changes=changes,
                           earlier=list(ctx.deps.edit_gate_log), prev_score=prev,
                           region_replaced=any(p["tool"] == "replace_region" for p in pend),
                           # 伴奏を MIDI で直接書き換えた（楽譜の `arrangement` は変わらない）なら、表の伴奏は作業台の MIDI から数える
                           accomp_mid=(mid1 if mid1.exists() and any(p["tool"] in ACCOMP_MIDI_TOOLS for p in pend) else None))
        from session import save_record
        save_record(song_dir, song)                        # 判断はこの版の記録に入る（書き出しの前・版の確定の前）
        ctx.deps.record_version = None
    except NOT_SWALLOW:
        raise
    except Exception as e:                                 # 点検役が呼べなくても書き出しは止めない——ただし黙らない
        _trace(sid, "edit_gate.error", where=where, reason=f"{type(e).__name__}: {e}"[:200])
        return (f"\n【最終の点検役（編集のあと）】呼べなかった（{type(e).__name__}: {str(e)[:160]}）。"
                "大きい編集の仕上がりは点検していない——最終報告でマスターにそう伝える。")
    ctx.deps.edit_gate_runs += 1
    left = fin.EDIT_MAX_RUNS - ctx.deps.edit_gate_runs
    _trace(sid, "edit_gate.out", where=where, action=v.action, runs=ctx.deps.edit_gate_runs)
    if v.action == "accept":
        ctx.deps.edit_gate_log.append(f"通した: {v.reason}" + (f"／気になったところ: {v.notes}" if v.notes else ""))
        return (f"\n【最終の点検役（編集のあと）】通した — {' '.join(v.reason.split())[:400]}"
                + (f"\n  ▲気になったところ: {' '.join(v.notes.split())[:400]}" if v.notes else ""))
    back = (v.redo_from or "").strip()
    ctx.deps.edit_gate_log.append(f"NG: {v.reason}" + (f"／直すなら {back} から: {v.guidance}" if back else ""))
    return (f"\n【最終の点検役（編集のあと）】★NG — {' '.join(v.reason.split())[:600]}"
            + (f"\n  直す案（提案・まだ何もしていない）: {back + ' から作り直す。' if back else ''}"
               f"{' '.join((v.guidance or '').split())[:900]}" if (back or v.guidance) else "")
            + "\n  書き出しは止めていない（NG を添えてマスターに渡す。マスターが聴いて決める）。"
              "最終報告で、**マスターの頼みそのものが原因か**（頼みどおりに足したが合わない）・**やり方が原因か**（音の書き方）を分けて伝える。"
              "やり方が原因なら、直す案を1つ添えて、直すかを聞く。"
            + (f"（マスターに頼まれて直したら、この頼みの中で点検はあと{left}回。）" if left
               else "（この頼みの中の点検は、これで使い切った。）"))


class _hand_scope:
    """作り直しのあいだだけ、**「この範囲は人が直している」**を申し送り（`song_context.CARRY`）に載せる
    （2026-09-27・目録の道具-8）。作り直しが落ちても、出るときに必ず外す（次の曲へ持ち越さない）。"""

    def __init__(self, prev_score, step: str):
        from manual_import import hand_edit_scope
        self.note = hand_edit_scope(prev_score, step)

    def __enter__(self):
        if self.note:
            import song_context as _sc
            _sc.CARRY.append(self.note)
        return self

    def __exit__(self, *exc):
        if self.note:
            import song_context as _sc
            if self.note in _sc.CARRY:
                _sc.CARRY.remove(self.note)
        return False


#: 編集の道具が台帳に書く行の出どころ（マスターの頼みを受けて作曲担当がしたこと）
EDIT_ORIGIN = "直した記録（マスターの頼みを受けて）"


def _split_ids(ids: str) -> list[str]:
    return [s.strip() for s in (ids or "").split(",") if s.strip()]


def _note_request(ctx: RunContext[MusicSession], words: str, target: str, *, op: str = "set",
                  scope: str = "", value: dict | None = None, supersedes: str = "",
                  history: list | None = None, origin: str = "") -> str:
    """台帳に1行書いて、作業台の記録（`record.json`）を保存する。書いた行の id を返す。

    ★**カバーにも台帳だけの記録を作る**——カバーも編成からやり直せるので、足した声部や移調は
      やり直したあとに当て直す必要がある。
    `history`（工程の記録 `StepRecord` の並び）を渡すと記録にも足す——カバーの作り直しが**推定した調と合い具合**を
    曲の記録に残すため（2026-10-01・判-C。カバーの記録は台帳だけで、作り直しの工程の記録は保存されないため）。
    """
    try:
        from session import save_record
        from song_model import Song
        from song_requests import add as _add
        song = _record_of(ctx)
        if song is None:
            song = Song(instruction=getattr(ctx.deps, "instruction", "") or "", origin="cover")
        if getattr(ctx.deps, "score", None) is not None:
            song.score = ctx.deps.score                    # 記録の楽譜も手直し後の姿にしておく
        # ★**出どころは「直した記録」**（2026-10-11・本人承認）。ここで書く `words` は編集の道具が書いた「したこと」（「音符を置き直した: …」）で、
        #   マスターの言葉そのものではない。以前は「マスター」と書いていて、最終の点検役が「マスターが自分で置いた音＝意図」と読み、
        #   弱点を打ち消す根拠に使った（`test/final_judge_role_ab`）。マスターの言葉はこの実行の依頼文（`instruction`）にある
        r = _add(song.requests, words, target, op=op, scope=scope, origin=origin or EDIT_ORIGIN,
                 version=song.version, value=value, supersedes=_split_ids(supersedes))
        for h in history or []:
            song.record(h)
        save_record(SONGS / ctx.deps.song_id, song)
        ctx.deps.record_version = None                     # 以後は作業台の記録が正
        ids = getattr(ctx.deps, "run_request_ids", None)
        if isinstance(ids, list):                          # この実行で積んだ行（編集のあとの最終の点検の【この編集】の印）
            ids.append(r.id)
        return r.id
    except Exception as e:                                 # 台帳が書けなくても編集そのものは止めない——ただし黙らない
        _trace(ctx.deps.song_id, "note_request.warn", target=target, reason=f"{type(e).__name__}: {e}"[:160])
        return ""


def _ledger_note(*ids: str) -> str:
    """台帳に書けなかった行があれば、返事に添える一文（2026-10-02・洗い出し 2-2）。以前は書けなくても成功と返し、
    作り直したときに手直しが黙って消えた。`_note_request` が返した id を渡す（空＝書けなかった）。"""
    if all(ids):
        return ""
    return ("\n★**この手直しを台帳に書けなかった**（曲の記録が保存できない）。楽譜には入っているが、"
            "あとで工程から作り直すとこの手直しは消える——最終報告でマスターに伝える。")


def _supersedes_problem(ctx: RunContext[MusicSession], supersedes: str, title: str = "") -> str:
    """`supersedes` に書いた id のうち、台帳に生きていない（無い・もう置き換えられた）もの の所見。無ければ空（2026-10-02・洗い出し 2-2）。
    以前は知らない id を黙って無視し、置き換えたつもりの古い頼みが生きたまま残った。**道具は何も変えずに返す**。"""
    ids = _split_ids(supersedes)
    if not ids:
        return ""
    try:
        song = _record_of(ctx, title)
    except Exception:
        song = None
    live = [r for r in (song.requests if song is not None else []) if r.status in ("active", "reapply")]
    gone = [i for i in ids if i not in {r.id for r in live}]
    if not gone:
        return ""
    return (f"何も変えなかった: supersedes の {', '.join(gone)} は、この曲の台帳で生きている頼みに無い"
            "（無い id か、もう置き換えられた頼み）。生きている頼み: "
            + (" / ".join(f"{r.id}（{r.target}・{r.words[:40]}）" for r in live) or "なし")
            + "。置き換える頼みが無いなら supersedes を空にして呼び直す。")


class DisplayEdit(BaseModel):
    """**見せる歌詞（字幕・投稿用の漢字まじりの詞）の1行**の書き直し（replace_region の `display`・set_display_lyrics・2026-10-12）。"""
    line: int = Field(ge=1, description="見せる歌詞の行番号（1始まり）。返事や inspect_song の「見せる歌詞」に出る番号")
    text: str = Field(description="その行の新しい見せる詞（漢字・カタカナ可・1行・読みは歌うかなと同じ。譜割りの印は入れない）")


def _requests_note(ctx: RunContext[MusicSession], title: str = "") -> str:
    """コーディネーターへ見せる「この曲で生きている頼みごと」（無ければ空）。続いている頼みは別の見出しで先に出す。"""
    try:
        from song_requests import listing, standing
        song = _record_of(ctx, title)
        reqs = list(song.requests) if song is not None else []
        keep = standing(reqs)
        got = listing([r for r in reqs if r not in keep])
    except Exception:
        return ""
    head = ("\n--- 【続いている頼み】（マスターがこの先も守ってほしいと言ったこと。直すたびに、反していないか確かめる。"
            "取り下げは keep_request(drop=True, supersedes=id)） ---\n"
            + "\n".join(f"  [{r.id}] {r.target}" + (f"（{r.scope}）" if r.scope else "") + f": 「{r.words}」（{r.time}）"
                        for r in keep)) if keep else ""
    return head + (f"\n--- この曲で生きている頼みごと（台帳・作り直しても守られる。置き換えるときは id を"
                   f" `supersedes` に指す） ---\n{got}" if got else "")


@coordinator.tool
@_traced
def keep_request(ctx: RunContext[MusicSession], words: str, target: str = "全体", scope: str = "",
                 supersedes: str = "", drop: JsonBool = False) -> str:
    """**マスターの頼みのうち、この先の直しでも守り続けるもの**を、曲の台帳に【続いている頼み】として残す（2026-10-14）。

    例: 「Bメロの最高音はサビより低く（いちばん高い音はサビまで取っておく）」「『シグナル』は『シ』が高く下がる言い方」
    「見せる歌詞（字幕）は、歌う詞と揃えておく」。今回の直しだけで済む頼み（「この音を半音上げて」）は残さない。
    残した頼みは、この曲を開くたびに【続いている頼み】として見え、編集のあとの最終の点検役も照らす（反した変化は NG）。
    ★一度通った頼みが、後の版の手直しで黙って崩れていた（『てのひらのシグナル』: シ↓グナルが v19 で戻り、最高音が v20 から崩れた）。

    - `words` … マスターの意図を1文で（マスターの言葉に近く。音符の羅列は書かない）
    - `target` … かかる工程（`旋律`・`詞`・`コード`・`伴奏` など工程の一覧の名前）。曲全体にかかるなら `全体`
    - `scope` … 範囲（`Bメロ`・`サビ`・`11:0.00〜19:0.00`）。曲全体なら空
    - マスターが「もう守らなくていい」と言ったら、`drop=True` と `supersedes` にその id を書いて取り下げる（words は理由を1文で）
    """
    from song_gate import STEP_ORDER
    from song_requests import standing
    words = " ".join((words or "").split())
    if not words:
        return "残さなかった: words が空。マスターの意図を1文で書く。"
    if drop:
        ids = _split_ids(supersedes)
        if not ids:
            return "取り下げなかった: supersedes に取り下げる頼みの id を書く。"
        try:
            from session import save_record
            song = _record_of(ctx)
            keep = {r.id: r for r in standing(list(song.requests))} if song is not None else {}
            bad = [i for i in ids if i not in keep]
            if bad:
                return (f"取り下げなかった: {', '.join(bad)} は続いている頼みに無い。続いている頼み: "
                        + (" / ".join(f"{k}（{r.words[:40]}）" for k, r in keep.items()) or "なし"))
            for i in ids:
                keep[i].status = "superseded"
                keep[i].superseded_by = f"取り下げ: {words[:80]}"
            save_record(SONGS / ctx.deps.song_id, song)
            ctx.deps.record_version = None
        except NOT_SWALLOW:
            raise
        except Exception as e:                             # noqa: BLE001
            return f"取り下げなかった: 曲の記録を読み書きできない（{type(e).__name__}）"
        return f"続いている頼み {', '.join(ids)} を取り下げた（理由: {words}）。"
    if target != "全体" and target not in STEP_ORDER:
        return f"残さなかった: target『{target}』は工程の名前ではない。工程の一覧の名前（{'・'.join(STEP_ORDER)}）か `全体` を書く。"
    if why := _supersedes_problem(ctx, supersedes):
        return why
    rid = _note_request(ctx, words, target, op="keep", scope=scope, supersedes=supersedes, origin="マスター")
    return (f"続いている頼みとして残した [{rid}]: {target}" + (f"（{scope}）" if scope else "") + f"「{words}」。"
            "この先この曲を直すときは、これに反していないかを確かめる。")


@coordinator.tool
@_traced
def set_display_lyrics(ctx: RunContext[MusicSession], display: list[DisplayEdit]) -> str:
    """**見せる歌詞（字幕・投稿用の漢字まじりの詞）の行だけを書き直す**（歌う詞・音は変えない・2026-10-14）。

    手直しで歌う詞だけ変わり、見せる歌詞が古いまま残った行を揃えるのに使う（load_existing_song が古そうな行を知らせる）。
    行番号は load_existing_song・inspect_song の「見せる歌詞」の番号。読みは、その行で実際に歌っているかなと同じにする。
    歌う詞も一緒に変えるなら replace_region の `display` を使う。"""
    from song_edits import apply_display_edits, display_problems
    s = ctx.deps.score
    if s is None:
        return "曲が未生成。先に load_existing_song を。"
    if not getattr(s, "lyrics_display", ""):
        return "書き直さなかった: この曲には見せる歌詞が無い（カバーなど）。"
    edits = {int(d.line): d.text for d in (display or [])}
    if not edits:
        return "書き直さなかった: display が空。"
    if bad := display_problems(edits, s, []):
        return "書き直さなかった: " + "／".join(bad)
    before = s.lyrics_display.split("\n")
    ctx.deps.score = s.model_copy(update={"lyrics_display": apply_display_edits(s, edits)})
    song_dir = SONGS / ctx.deps.song_id
    (song_dir / song_files.SCORE).write_text(ctx.deps.score.model_dump_json(indent=2), encoding="utf-8")
    rows = [f"行{k}「{before[k - 1]}」→「{' '.join(t.split())}」" for k, t in sorted(edits.items())]
    rid = _note_request(ctx, "見せる歌詞を書き直した: " + "／".join(rows)[:300], "詞",
                        value={"display": {str(k): v for k, v in edits.items()}})
    return "見せる歌詞を書き直した（歌う詞・音は変えていない）: " + "／".join(rows) + _ledger_note(rid)


def _stopped_edit(sid: str, tool: str, err: StepUnfixed, *, restart: str = "", how: str = "") -> str:
    """直し（作り直し・アレンジ・尺の変更）が、**形式の所見が直らずに途中で止まった**ときの返事（2026-09-30）。

    曲の版は切らない（元の版はそのまま）。どの工程で何が直らなかったか・どう頼み直すかは `StepUnfixed.message` が書く。
    """
    _trace(sid, f"{tool}.out", kind="stopped", step=err.step, findings=len(err.findings))
    return err.message(restart=restart, edit=True, how=how)


def _resize_how(step: str) -> str:
    """`resize_song` が止まったときの「どうなったか・どう頼み直すか」を、**止まった工程で分ける**（2026-10-02）。
    以前は詞のことだけ書いていたが、尺の変更は 構成を起こす（カバーの構成推定）→ 並べ替えの構成（尺の変更の構成）→
    新しい区間の詞（詞）の3か所で止まりうる。"""
    head = "**尺は変えていない**（元の版はそのまま残っている）。\n"
    if step == "カバーの構成推定":
        return (head + "いまの曲の区間（構成）を旋律から起こす所で止まった（区間の切れ目の形式が直らなかった）。"
                "**resize_song(request=同じ頼み)** でもう一度呼ぶと起こし直す。同じ所で止まるなら、この曲は区間が取れない——"
                "最終報告でマスターに伝えて終える。")
    if step == "尺の変更の構成":
        return (head + "どう並べ替えるか（新しい構成）を決める所で止まった。続けるときは **resize_song(request=…)** でもう一度——"
                "頼みを具体的に（どの区間を足す・削る・繰り返すかを、上の所見に出たいまの区間名で）書き直してよい。")
    if step == "詞":
        return (head + "新しく歌う区間の詞の所で止まった。続けるときは **resize_song(request=同じ頼み)** でもう一度"
                "（新しい区間の詞を書き直す。上の所見が直るように、という一言を頼みに足してよい）。")
    return (head + f"『{step}』の所で止まった。続けるときは **resize_song(request=同じ頼み)** でもう一度呼ぶ"
            "（上の所見が直るように、という一言を頼みに足してよい）。同じ所で止まるなら最終報告でマスターに伝えて終える。")


def _cover_step(step: str, start: str) -> str:
    """カバーで止まった工程 → `remake_from_step` に渡せる工程（カバーが作り直せるのは コード / 編成 / 伴奏 / ハモリ だけ）。
    声部と奏法はハモリの中で作るのでハモリから。コードと区間（カバーの構成推定）は、記録に無いときに作り直しの前に用意する
    ものなので、作り直しを始めた工程から（もう一度呼べば用意し直す・2026-10-02 組「カバーのコード」）。それ以外（来ないはず）も同じ。"""
    from song_arrange_cover import STEPS as COVER_STEPS
    if step == "コード":
        return start
    if step in COVER_STEPS:
        return step
    return "ハモリ" if step in ("声部", "奏法") else start


def _cover_ready(ctx: RunContext[MusicSession], song_dir: Path, song, rec, **kw) -> list:
    """**カバーの区間・調・コードを用意する**（2026-10-02・組「カバーのコード」・J1）。曲の記録にあれば読み、無ければ作って
    （区間と調は LLM「カバーの構成推定」・コードは LLM「コード」）**すぐ曲の記録へ残す**——このあとの編曲が止まっても、
    次の編集で同じ費用をかけ直さない。戻り値は今回作った工程の記録（台帳の記録 `_note_request(history=…)` へ渡す）。"""
    from song_cover_chords import dirty, prepare_cover
    made = prepare_cover(song, rec, verbose=False, **kw)
    if dirty(song):
        _cover_save(ctx, song_dir, song, rec)
    return made


def _cover_save(ctx: RunContext[MusicSession], song_dir: Path, song, rec) -> None:
    """カバーの区間・調・コードを作業台の曲の記録（`record.json`）へ残す（`song_cover_chords.save_into`）。"""
    try:
        from session import save_record
        from song_cover_chords import save_into
        from song_model import Song
        r = rec if rec is not None else Song(instruction=getattr(ctx.deps, "instruction", "") or "", origin="cover")
        if r.score is None:
            r.score = ctx.deps.score if getattr(ctx.deps, "score", None) is not None else song.score
        save_into(r, song)
        save_record(song_dir, r)
        ctx.deps.record_version = None                     # 以後は作業台の記録が正
    except Exception as e:                                 # 記録に残せなくても編集は続ける——ただし黙らない（次の編集で作り直しになる）
        _trace(ctx.deps.song_id, "cover_skeleton.warn", reason=f"{type(e).__name__}: {e}"[:160])

@coordinator.tool
@_traced
def set_reference(ctx: RunContext[MusicSession], reference_json_path: str) -> str:
    """取得済みの参照（採譜/MIDIの階名JSON）をそのまま使う。検索が不要/不安定な時用。

    ★**空文字を渡すと参照を外す**（2026-09-20 追加）。カバーとして原曲に合わせるのをやめて、
      **オリジナルとして作り直したい**ときに使う。外すまでは `faithful`（原曲の伴奏を保つ）が
      効き続け、`render_final` が伴奏を作り直さないので、編成を変えても反映されない。
      **付けたら外せない**状態だったのを直した。
    """
    if not reference_json_path:                            # 参照を外す（カバー → オリジナルへ戻す）
        had = ctx.deps.reference_json or ctx.deps.reference_midi or ctx.deps.reference_audio_accomp
        ctx.deps.reference_json = None
        ctx.deps.reference_midi = None
        ctx.deps.reference_audio_accomp = None
        ctx.deps.melody_channel = None
        ctx.deps.faithful = False
        return ("参照を外した（以降はオリジナル扱い＝伴奏は楽譜から作り直す）。"
                if had else "参照はもともと設定されていない。")
    if not Path(reference_json_path).exists():
        return f"見つからない: {reference_json_path}"
    ctx.deps.reference_json = reference_json_path
    n = len(load_reference(reference_json_path)["notes"])
    return f"参照を設定: {reference_json_path}（{n}音）"

# ★**`RegionGain`（区間ごとの音量）と `accomp_gain`（伴奏全体の音量）は外した**（2026-09-27・目録の 音量-4）。
#   2026-09-06 に仮歌ミックスを廃止してから**どちらも効いていなかった**のに、`adjust_song` は
#   「伴奏音量(絶対)X」「区間音量N件」と**効いたように返し**、台帳にも「組み立て.音量」と残していた。
#   歌と伴奏の釣り合いは `vocal_lead_db`（vpr の伴奏トラックの初期音量）、区間の伴奏を変えるなら
#   `replace_accompaniment_region`。

#: ★**道具の音量の範囲**（2026-10-02・洗い出し 2-2・記法の仕様書 §15-2）。以前は範囲が無く、vpr を書くときに
#:   フェーダーの範囲（声部 −60〜+6dB・伴奏 −40〜+6dB）へ黙って丸め、返事は書いた値を言っていた。型（道具の引数）で差し戻す。
#:   声部の音量は工程のハモリ（`harmony.HarmonyPart.gain_db`）と同じ物差し（主旋律 0dB に対する dB）。
#: ★**範囲は書き出しの限界に合わせた**（2026-10-03・`docs/plan/song_viewpoints_review.md` §2 c10・c11）。以前は声部 −18〜0・
#:   歌の前に出す量 −6〜+12 で、機械の限界より狭かった（−18 未満・主旋律より大きい声部は音の判断＝仕上げの点検役の材料・d3）。
#:   声部: vpr の声部フェーダー `v6_export._FADER_MIN/MAX`（−60〜+6dB）。`harmony.VOICE_GAIN_DB` と同じ（1か所）
from harmony import VOICE_GAIN_DB  # noqa: E402
#:   歌を伴奏より前に出す量（既定 `v6_export.LEAD_DB`＝+4）。伴奏のフェーダーは「歌と伴奏の実測の差 − この値」で決まり
#:   （`score_to_vpr._accomp_gain_db`・フェーダーは `_GAIN_MIN/MAX`＝−40〜+6dB に収める）、ふつうの曲（差が −10dB 前後）で
#:   フェーダーがその範囲に収まる値＝ −10 − 6 〜 −10 + 40 。差が曲ごとに違うので、端の近くでは曲によってフェーダーの端に届きうる
_TYPICAL_VOCAL_MINUS_ACCOMP_DB = -10.0
VOCAL_LEAD_DB = (_TYPICAL_VOCAL_MINUS_ACCOMP_DB - 6.0, _TYPICAL_VOCAL_MINUS_ACCOMP_DB + 40.0)


def _design_bpm_range() -> tuple[float, float]:
    """設計の BPM の範囲（`song_design.Design.bpm` の型の `ge`・`le`＝60〜200）。`adjust_song(tempo=)` も同じ範囲
    （2026-10-03・再レビュー 記O・仕様書 §15-2）。手で書いた数を2か所に持たない（`song_cover_load.section_bars_range` と同じ形）。"""
    from song_design import Design                          # noqa: E402
    lo, hi = 60.0, 200.0
    for m in Design.model_fields["bpm"].metadata:
        lo = float(m.ge) if getattr(m, "ge", None) is not None else lo
        hi = float(m.le) if getattr(m, "le", None) is not None else hi
    return lo, hi


#: ★**速さの範囲**（2026-10-03・記O）。以前は範囲が無く、−120 も通り、0 は「変えない」と読んでいた。変えないなら書かない
TEMPO_RANGE = _design_bpm_range()


@coordinator.tool
@_traced
def adjust_song(ctx: RunContext[MusicSession], *,
                tempo: Annotated[JsonFloat | None, Field(
                    ge=TEMPO_RANGE[0], le=TEMPO_RANGE[1],
                    description=(f"新しい速さ（BPM・4分音符の数・{TEMPO_RANGE[0]:g}〜{TEMPO_RANGE[1]:g}＝設計の BPM と同じ範囲）。"
                                 "変えないなら書かない（0 は書けない）"))] = None,
                transpose: JsonInt = 0,
                vocal_lead_db: Annotated[JsonFloat | None, Field(
                    ge=VOCAL_LEAD_DB[0], le=VOCAL_LEAD_DB[1],
                    description=(f"歌声を伴奏より前に出す量（dB・{VOCAL_LEAD_DB[0]:g}〜+{VOCAL_LEAD_DB[1]:g}・既定は +4・"
                                 "大きいほど歌が前・0 で同じくらい）。省略＝いまの設定のまま"))] = None,
                add_voice: str | None = None, replace_voice: str = "",
                supersedes: str = "") -> str:
    """確定スコアの編集（修正用）。歌声(score)と伴奏(accompaniment.wav)を同期編集し再レンダ＋再ミックス。
    ・tempo変更 / transpose(半音・移調。追加声部も移調) / 歌と伴奏の音量バランス。
    ・**add_voice**: 人間の指示文（例『サビにハモリ』『2拍遅れの輪唱』『オクターブ下で重ねて』）を渡すと、
      メイン旋律に重ねる**追加声部をLLMが生成**して多トラックvprに足す（種類は指示から自動判別）。
    ・**replace_voice**: 足すのではなく**差し替える**ときに、置き換える声部の名前（部分一致）を渡す。
      『上ハモを下ハモに変えて』なら `replace_voice='上'` ＋ `add_voice='下ハモ'`。
      ★**足すか差し替えるかは、マスターの言葉から決める**——指定しないと足すだけなので、
        差し替えのつもりで呼ぶと**二重になる**（2026-09-20 に上ハモが残ったまま3声になった）。
    ・**音量**: 既定は自動バランス。vocal_lead_db（dB・−16〜+30・ふつうは 0〜+8）で歌声を伴奏より前に出す量を決める（以後も踏襲）。
      **特定区間の伴奏だけ**大小・薄く厚くしたいなら、伴奏そのものを replace_accompaniment_region で変える。
    直したあとは export_vocal で歌声を書き出し直す（VOCALOID6 を自動で操作する）。
    ・**supersedes**: 作り直しで「当て直し待ち」になった頼みを当て直すときに、その id（`r…`・カンマ区切り）。
      台帳に生きていない id を書くと、何も変えずに返す。
    ★速さ・移調・足した声部は台帳に残り、**あとで工程から作り直しても当て直される**。
    ★**移調・テンポ・声部の追加は「大きい編集」**: 書き出し（export_vocal）の前に最終の点検役がまとめて見る（NG なら所見と直す案が
      返事に載る。書き出しは止まらない）。カバーは最終の点検役を通らないので、書き出しの前に review_progress（最終検問）を必ず呼ぶ。
    """
    from song_edits import transpose_score
    from song_pipeline import _render_accomp
    from build_lipsync import write_lipsync
    from song_pipeline import render_placeholder_vocal
    from score_to_vpr import score_to_vpr
    if ctx.deps.score is None:
        return "曲が未生成。先に compose か load_existing_song を。"
    # ★**形に合わない頼みは、何も変えずに断る**（2026-10-03・§7-2）。アカペラ・インストに歌と伴奏の釣り合いは無く、
    #   インストには声部を重ねる歌が無い（黙って作らない・黙って足さない）
    form = form_of(ctx)
    if vocal_lead_db is not None and (why := form_refusal(ctx, "adjust_song.vocal_lead_db")):
        return why
    if add_voice and (why := form_refusal(ctx, "adjust_song.add_voice")):
        return why
    # ★**replace_voice だけでは何もしない**（2026-10-02・洗い出し 2-2）。以前は黙って無視して「調整完了」と返した
    if replace_voice and not add_voice:
        return ("何も変えなかった: replace_voice は add_voice と一緒に書く（差し替えて足す声部の頼み）。"
                "声部を消すだけなら adjust_voice(title=…, remove=True)。")
    if why := _supersedes_problem(ctx, supersedes):
        return why
    s = ctx.deps.score
    old_tempo = float(s.tempo)
    new_tempo = float(tempo) if tempo is not None else old_tempo   # 範囲は型（TEMPO_RANGE・0 は書けない・2026-10-03 記O）
    song_dir = SONGS / ctx.deps.song_id
    acc_path = song_dir / "accompaniment.wav"
    _moves = bool(transpose or (tempo and abs(new_tempo - old_tempo) > 1e-6))
    # ★**伴奏が楽譜から作られていない曲（カバー）は、伴奏 MIDI を動かす。無ければ断る**（2026-09-27・目録の道具-4）。
    #   以前はカバーでも楽譜（`arrangement`＝無い）から鳴らし直していたので、**移調・テンポ変更で伴奏が空になった**
    #   （`midi_out.score_to_midi` はパートの無い楽譜から音を1つも出さない）。歌だけ動くと伴奏とずれるので、
    #   伴奏を動かせないときは歌も動かさずに返す。
    acc_mid = song_dir / "accompaniment.mid"
    no_accomp_score = s.arrangement.is_empty()             # 伴奏が楽譜に無い＝パートが0本（F1。以前は None）
    if acc_path.exists() and _moves and no_accomp_score and not acc_mid.exists():
        _trace(ctx.deps.song_id, "adjust_song.refused", reason="no-accomp-score")
        return ("移調・テンポ変更はしなかった: この曲の伴奏は楽譜から作られていない（原曲の MIDI か録音から用意した伴奏）うえ、"
                "動かせる伴奏 MIDI（accompaniment.mid）も無い。歌だけ動かすと伴奏とずれるので、どちらも変えていない。"
                "伴奏ごと移調したいなら、先に rearrange_song で伴奏を楽譜から作り直す（原曲の伴奏ではなくなる）。"
                "その旨をマスターに伝えて、どうするか聞く。")
    # ★**伴奏 MIDI を動かすと鳴らせる高さからはみ出すなら、移調しない**（2026-10-02・洗い出し 2-3）。以前は端の高さに張り付けた
    if acc_path.exists() and transpose and no_accomp_score and acc_mid.exists():
        from song_edits import shift_midi_problem
        if _why := shift_midi_problem(acc_mid, int(transpose)):
            _trace(ctx.deps.song_id, "adjust_song.refused", reason="accomp-out-of-range")
            return f"移調はしなかった（ほかの変更もしていない）: {_why}。移調の量を小さくするか、マスターに伝える。"
    # ★**読めないコード名があるなら移調しない**（2026-09-27・監査）。以前は `transpose_chord` が読めないコードを
    #   黙ってそのまま返し、そのコードだけ元の調で鳴っていた。
    if transpose:
        from song_edits import untransposable
        _bad = untransposable(s)
        if _bad:
            _trace(ctx.deps.song_id, "adjust_song.refused", reason="unreadable-chord")
            return ("移調はしなかった（ほかの変更もしていない）: 楽譜に読めないコード名があり、移調するとそのコードだけ元の調で鳴る。"
                    + " / ".join(_bad[:3]) + "。コードを直してから（remake_from_step('コード', …) など）移調する。"
                    "その旨をマスターに伝える。")
        # ★**移調で鳴らせる高さ（0〜127）をはみ出すなら移調しない**（2026-10-03・再レビュー）。以前は歌・声部・伴奏・主旋律の楽器が
        #   はみ出しても保存し、次にその曲を開けなくなった（カバーの伴奏 MIDI だけ確かめていた）。判定は組み立ての側の1つ
        from song_edits import transpose_problem
        if _why := transpose_problem(s, int(transpose)):
            _trace(ctx.deps.song_id, "adjust_song.refused", reason="out-of-range")
            return f"移調はしなかった（ほかの変更もしていない）: {_why}。移調の量を小さくするか、マスターに伝える。"
    # メイン＋追加声部＋**コード**をまとめて移調（`song_edits.transpose_score`）。
    # ★**コードを一緒に動かさないと、歌だけが調を変えて伴奏から離れる**（2026-09-07 修正）。
    #   実際に起きた事故: Coordinator が音域を下げようとして −12 → −8 → +14（合計 −6半音）と
    #   呼び、**歌は A♭メジャー・伴奏は D メジャー**という三全音違いの曲が出荷された。
    #   142音中109音（77%）がコードと半音でぶつかり、本人には「リズムに乗ってない」と聞こえた。
    #   当時は伴奏を `librosa` で**音声のままピッチシフト**していたが、
    #   **直後の `render_final` が楽譜から伴奏を作り直すたびに、その加工が捨てられていた**。
    #   編曲パートは**コードからの度数**（`degree`）で書かれているので、
    #   **コードさえ動かせば伴奏は丸ごと追随する**——音声を触る必要は元から無かった。
    # ★**ハモリ追加の物差しが決められない曲は、何も変えずに返す**（2026-10-02・ず-12①）。C 基準で見せて書かせるので、
    #   区間の調が分からないと移す量が決まらない（黙って 0＝鳴る高さで書かせると、物差しが2通りになる）
    basis = None
    made: list = []                                        # 今回カバーの区間と調を用意したときの工程の記録（台帳へ）
    prepared = ""
    if add_voice:
        # ★**区間と調を記録に持たないカバー（新しく作ったカバー）は、ここで用意してから足す**（2026-10-03・再レビュー）。
        #   カバーのコードの入口の「区間と調だけ」（`prepare_cover(chords=False)`・LLM「カバーの構成推定」）を通し、曲の記録へ残す
        if _cover_lacks_skeleton(ctx, song_dir):
            try:
                from song_cover_load import key_note, load_cover
                _cv = load_cover(song_dir, verbose=False)
                made = _cover_ready(ctx, song_dir, _cv, _record_of(ctx), chords=False)
                prepared = key_note(_cv)
            except StepUnfixed as e:                       # 区間と調の形式が直らずに止まった（何も変えていない）
                return _stopped_edit(ctx.deps.song_id, "adjust_song", e, how=(
                    "**ハモリは足していない**（ほかの変更もしていない）。この曲の区間と調（ハモリを書かせる高さの物差し）を"
                    "旋律から起こす所で止まった。**adjust_song(add_voice=同じ頼み)** でもう一度呼ぶと起こし直す。"
                    "同じ所で止まるなら最終報告でマスターに伝えて終える。"))
            except NOT_SWALLOW:
                raise
            except Exception as e:                         # 用意できなかった＝足さない（理由を返す）
                _trace(ctx.deps.song_id, "adjust_song.refused", reason=f"cover-skeleton-{type(e).__name__}")
                return (f"ハモリは足さなかった（ほかの変更もしていない）: この曲（カバー）の区間と調を用意できなかった"
                        f"（{type(e).__name__}: {str(e)[:160]}）。その旨をマスターに伝える。")
        basis = _added_voice_basis(ctx, int(transpose or 0))
        if isinstance(basis, str):
            _trace(ctx.deps.song_id, "adjust_song.refused", reason="no-key-for-voice")
            return (f"ハモリは足さなかった（ほかの変更もしていない）: {basis}。"
                    "記録が残っていない古いオリジナル曲には、この道具で声部を足せない。その旨をマスターに伝える"
                    "（工程から作り直した曲には記録がある。カバーは区間と調をこの道具が旋律から起こして記録に残す）。")
    s = transpose_score(s, int(transpose or 0)).model_copy(update={"tempo": new_tempo})
    added = None
    replaced = ""
    vt = None
    voice_warns: list[str] = []
    if add_voice and replace_voice:                        # 差し替え: 先に消す（見つからなければ何もせず止める）
        hit = [v for v in s.extra_voices if replace_voice in v.title]
        if len(hit) != 1:
            return (f"差し替える声部『{replace_voice}』が"
                    + ("見つからない" if not hit else "複数ある")
                    + "。いまある声部: " + (" / ".join(v.title for v in s.extra_voices) or "なし"))
        replaced = hit[0].title
        s = s.model_copy(update={"extra_voices": [v for v in s.extra_voices if v is not hit[0]]})
    if add_voice:                                          # 追加声部をLLM生成して足す（ハモリ/輪唱/重ね/コーラス）
        shift_of, bars_of, keys = basis
        # ★**型の差し戻しが直らなければ止まった旨を返す**（2026-10-03・再レビュー）。以前は例外のまま作曲担当ごと落ちた。
        #   ここまでは楽譜もファイルも書いていない（移調・差し替えも手元の写しだけ）＝何も変えずに返す
        try:
            vt, voice_warns = generate_added_voice(s, add_voice, shift_of=shift_of, bars_of=bars_of, keys=keys)
        except StepUnfixed as e:
            return _stopped_edit(ctx.deps.song_id, "adjust_song", e, how=(
                "**ハモリは足していない**（移調・速さ・音量などほかの変更もしていない）。\n"
                "続けるときは **adjust_song(add_voice=頼み)** でもう一度（上の所見が直るように、という一言を頼みに足してよい）。"
                "同じ所で止まるなら最終報告でマスターに伝えて終える。"))
        s = s.model_copy(update={"extra_voices": list(s.extra_voices) + [vt]})
        added = f"{vt.title}({len(vt.notes)}音, {vt.gain_db:+.0f}dB)"
    ctx.deps.score = s

    acc_note = ""
    if acc_path.exists() and _moves and s.arrangement.is_empty():
        # カバー: 伴奏 MIDI を移調・テンポ変更して鳴らし直す（楽譜には伴奏が無い）
        from song_edits import shift_midi
        from sf2_render import render_midi
        n_moved = shift_midi(acc_mid, acc_mid, semitones=int(transpose or 0),
                             tempo_from=old_tempo, tempo_to=new_tempo)
        render_midi(acc_mid, acc_path)
        acc_note = f"伴奏は伴奏 MIDI を動かして鳴らし直した（{n_moved}音を移調・打楽器は据え置き）"
    elif (acc_path.exists() or form == "instrumental") and _moves and form != "acappella":
        # **楽譜から作り直す**（音声を加工しない）。編曲は「コードからの度数」と「拍」で
        # 書かれているので、移調もテンポ変更も**再レンダリングで無劣化に反映される**。
        # 以前の `librosa` 加工は劣化するうえ、`render_final` の再レンダで毎回消えていた。
        # ★アカペラは伴奏を持たない（空の楽譜を鳴らさない）。インストは主旋律も伴奏の楽譜の1パートなので、ここで一緒に動く（§5-2）
        _render_accomp(s, acc_path)
    if form == "instrumental":
        # ★**インストは vpr・仮歌・口の動きを持たない**（§4-4・`song_files.FORM_FILES`）。楽譜だけを書く
        (song_dir / "score.json").write_text(s.model_dump_json(indent=2), encoding="utf-8")
    else:
        score_to_vpr(s, song_dir / "vocal.vpr")           # G0: vprと常に同時に永続化（声のIDは既定）
        (song_dir / "score.json").write_text(s.model_dump_json(indent=2), encoding="utf-8")  # （移調等がscore.jsonに残らず次セッションで
        render_placeholder_vocal(s, song_dir / "placeholder.wav")                            #   消えるバグの修正。scoreはvpr成功後=SoTを最後に）
        write_lipsync(s, song_dir)                        # 口の動きも score に追随させる（歌詞・音符が変わったので）
    if vocal_lead_db is not None:                          # 指定した音量バランスを永続化（後続mixも踏襲）
        from song_pipeline import set_mix_setting
        set_mix_setting(song_dir, vocal_lead_db=float(vocal_lead_db))
    # ★仮歌ミックスは 2026-09-06 に廃止（`export_vocal` が実歌声を90秒で出すので、
    #   サイン波で仕上がりを判断する意味が無くなった）。**音量バランスは vpr に焼き込む**
    #   経路が正で、上の `score_to_vpr` が `mix.json` の `vocal_lead_db` を見て反映する。
    #   ★`accomp_gain`・`region_gains` は効いていなかったので引数ごと外した（2026-09-27・音量-4）。
    ch = []
    if tempo:
        ch.append(f"tempo {old_tempo}→{new_tempo}")
    if transpose:
        ch.append(f"移調 {transpose:+d}半音")
    if added:
        ch.append(f"声部{'差し替え（' + replaced + '→）' if replaced else '追加'}={added}")
    if vocal_lead_db is not None:
        ch.append(f"自動バランス(歌声+{vocal_lead_db:.0f}dB・以後踏襲)")
    elif form == "song":                                   # 歌と伴奏の釣り合いは歌ものだけ（アカペラ・インストに釣り合いは無い）
        ch.append("自動バランス(既定/踏襲)")
    if acc_note:
        ch.append(acc_note)
    _trace(ctx.deps.song_id, "adjust_song", changes=" / ".join(ch), voices=len(s.extra_voices))
    # ★**台帳にも書く**（2026-09-25）。楽譜だけ直すと、あとで作り直したときに黙って消える。
    #   速さ・移調・足した声部は**当て直すための値**も残し、作り直しのあとに機械が当てる
    #   （`song_edits.apply_edits`・段階3）。音量は `mix.json` に残るので値は要らない。
    #   `supersedes` は最初の1行にだけ付ける（当て直し待ちを当て直した、の対応を1つにする）。
    #   ★**書く順は、この関数が当てた順**（移調 → 声部）。当て直しは台帳の順に当てるので、
    #     逆にすると、移調後の高さで記録した声部をもう一度移調してしまう。
    _sup = supersedes
    _ids: list[str] = []
    if transpose:
        _ids.append(_note_request(ctx, f"移調 {transpose:+d}半音", "組み立て.移調", op="add",
                                  value={"semitones": int(transpose)}, supersedes=_sup))
        _sup = ""
    if tempo and abs(new_tempo - old_tempo) > 1e-6:
        _ids.append(_note_request(ctx, f"速さを {old_tempo:g} → {new_tempo:g}", "組み立て.速さ",
                                  value={"tempo": new_tempo}, supersedes=_sup))
        _sup = ""
    if vt is not None:
        _ids.append(_note_request(ctx, add_voice, "組み立て.声部", op="set" if replaced else "add", scope=replaced,
                                  value={"track": vt.model_dump(), "replaces": replaced}, supersedes=_sup,
                                  history=made))           # カバーの区間と調を今回用意したなら、その工程の記録も
        _sup = ""
    if vocal_lead_db is not None:
        _ids.append(_note_request(ctx, " / ".join(c for c in ch if "音量" in c or "バランス" in c),
                                  "組み立て.音量", supersedes=_sup))
    # ★**点検待ちの束へ**（2026-10-04・案C）。移調・テンポ・声部は大、音量は小（積まない）
    _keys = ([("adjust_song.transpose")] if transpose else []) \
        + (["adjust_song.tempo"] if tempo and abs(new_tempo - old_tempo) > 1e-6 else []) \
        + (["adjust_song.add_voice"] if vt is not None else []) \
        + (["adjust_song.vocal_lead_db"] if vocal_lead_db is not None else [])
    _gate = note_edit(ctx, _keys, "／".join(ch)) if _keys else ""
    # 足した声部の所見（声部の工程と同じ: 移したあとの鳴る高さで推奨音域の外・16分の格子の外など。直してはいない）
    vw = ("\n足した声部の所見（直していない）: " + " / ".join(voice_warns[:6])) if voice_warns else ""
    vw += f"\n{_gate}" if _gate else ""
    if prepared:                                           # カバーの区間と調をこの呼び出しで起こした（記録に残した）
        vw = (f"\nこの曲（カバー）には区間と調の記録が無かったので、旋律から起こして記録に残した（ハモリはこの調へ移した）: "
              f"{prepared}。調が違っていたらマスターに伝える。") + vw
    if form == "instrumental":                             # 歌も vpr も無い。伴奏（主旋律を含む）を鳴らし直しただけ
        return (f"調整完了（{' / '.join(ch) or '変更なし'}）。伴奏（主旋律を含む）を鳴らし直した（インストなので歌と vpr は無い）。"
                f"{finish_hint(form)}" + (f"\n{_gate}" if _gate else "") + _ledger_note(*_ids))
    if form == "acappella":
        return (f"調整完了（{' / '.join(ch) or '変更なし'}）。vpr({1 + len(s.extra_voices)}トラック) 更新（アカペラなので伴奏は無い）。"
                f"{finish_hint(form)}{vw}" + _ledger_note(*_ids))
    return (f"調整完了（{' / '.join(ch)}）。vpr({1 + len(s.extra_voices)}トラック)/伴奏 更新。{NEXT_EXPORT}{vw}"
            + _ledger_note(*_ids))

@coordinator.tool
@_traced
def adjust_voice(ctx: RunContext[MusicSession], title: str, *,
                 gain_db: Annotated[JsonFloat | None, Field(
                     ge=VOICE_GAIN_DB[0], le=VOICE_GAIN_DB[1],
                     description=(f"その声部の音量（主旋律 0dB に対する dB・{VOICE_GAIN_DB[0]:g}〜{VOICE_GAIN_DB[1]:g}・"
                                  "負で後ろへ下げる。工程のハモリの音量と同じ物差し）。省略＝変えない"))] = None,
                 transpose: JsonInt = 0,
                 remove: JsonBool = False, supersedes: str = "") -> str:
    """**追加声部（ハモリ・輪唱・重ね）を1つだけ直す／消す。**

    `title` は声部の名前（`third_up` / `harmony_lower` 等。**`check_consistency` の1行目に出る**）。
    **部分一致でよい**（名前が分からなければ適当に呼べば、いまある声部が一覧で返る）。
    ・`remove=True` … その声部を**消す**（他の声部と主旋律はそのまま）
    ・`transpose` … その声部だけ移調（**−12 でオクターブ下**。音域から外れたハモリを収める）
    ・`gain_db` … その声部だけ音量（主旋律 0dB に対する dB・−60〜+6。負で後ろへ下げる。ハモリは −3〜−6 くらいが目安）

    ★**`add_voice` は「足す」だけ**なので、『上ハモを下ハモに変えて』のつもりで呼ぶと
      **二重になる**（2026-09-20 に実際に起きた: 上ハモが残ったまま3声）。
      **差し替えたいときは、ここで消してから足す。**
    ※工程ごと作り直す `remake_from_step('ハモリ')` もあるが、**下流を全部作り直す**ので粗い。
      1つだけ直したいならこちら。
    ★台帳に残るので、**あとで工程から作り直しても、消した声部は戻らない**（同じ名前の声部に当て直す）。
    `supersedes` … 「当て直し待ち」を当て直すときに、その id。
    ★±12 以外の幅の移調は「大きい編集」（書き出しの前に最終の点検役がまとめて見る）。消す・オクターブ移しは、同じ頼みに大きい編集があれば一緒に見る。
    """
    from build_lipsync import write_lipsync                # ※`adjust_song` と同じ遅延import一式
    from song_pipeline import render_placeholder_vocal     #   （モジュール先頭には無い。抜くと実行時に NameError）
    from score_to_vpr import score_to_vpr
    if ctx.deps.score is None:
        return "曲が未生成。先に compose か load_existing_song を。"
    if why := _supersedes_problem(ctx, supersedes):
        return why
    s = ctx.deps.score
    voices = list(s.extra_voices)
    if not voices:
        return "追加声部が無い（ハモリ等は付いていない）。"
    hit = [i for i, v in enumerate(voices) if title and title in v.title]
    if not hit:
        return f"『{title}』という声部が見つからない。いまあるのは: " + \
               " / ".join(f"{v.title}({len(v.notes)}音)" for v in voices)
    if len(hit) > 1:
        return f"『{title}』に当てはまる声部が複数ある: " + \
               " / ".join(voices[i].title for i in hit) + "。名前を絞って。"
    i = hit[0]
    v = voices[i]
    # ★**消すときに移調・音量を一緒に書いたら、何も変えずに返す**（2026-10-03・再レビュー）。以前は黙って捨てて消していた
    if remove and (transpose or gain_db is not None):
        return ("何も変えなかった: remove=True（声部を消す）と transpose・gain_db（声部を直す）は一緒に書けない。"
                "消すなら remove=True だけ、直すなら remove を外して呼び直す。")
    # ★**移調で鳴らせる高さ（0〜127）をはみ出すなら移調しない**（2026-10-03・再レビュー）。判定は組み立ての側の1つ
    #   （この声部だけを動かすので、この声部だけを見る `voice=`）
    if transpose and not remove:
        from song_edits import transpose_problem
        if _why := transpose_problem(s, int(transpose), voice=v.title):
            return f"何も変えなかった: {_why}。移調の量を小さくする（オクターブなら ±12）か、マスターに伝える。"
    if remove:
        voices.pop(i)
        what = f"『{v.title}』({len(v.notes)}音) を消した"
    else:
        upd = {}
        if transpose:
            upd["notes"] = [n.model_copy(update={"key": n.key + int(transpose)}) for n in v.notes]
        if gain_db is not None:
            upd["gain_db"] = float(gain_db)
        if not upd:
            return "何も指定されていない（remove / transpose / gain_db のどれかを指定して）。"
        voices[i] = v.model_copy(update=upd)
        parts = []
        if transpose:
            keys = [n.key for n in voices[i].notes]
            parts.append(f"移調 {transpose:+d}半音（音域 {spell_pitch(min(keys))}〜{spell_pitch(max(keys))}）")   # 国際式（以前は MIDI 番号）
        if gain_db is not None:
            parts.append(f"音量 {gain_db:+.1f}dB")
        what = f"『{v.title}』を直した（{' / '.join(parts)}）"

    s = s.model_copy(update={"extra_voices": voices})
    ctx.deps.score = s
    song_dir = SONGS / ctx.deps.song_id
    score_to_vpr(s, song_dir / "vocal.vpr")               # 声のIDは既定（MIKU_V6_COMP_ID）
    (song_dir / "score.json").write_text(s.model_dump_json(indent=2), encoding="utf-8")
    render_placeholder_vocal(s, song_dir / "placeholder.wav")
    write_lipsync(s, song_dir)                            # 口は主旋律だけだが、念のため追随させる
    _trace(ctx.deps.song_id, "adjust_voice", changes=what, voices=len(voices))
    # ★**台帳にも書く**（2026-09-25）。楽譜だけ直すと、ハモリや声部を作り直したときに
    #   **消した声部が戻ってくる**。作り直しのあとに機械が同じ名前の声部へ当て直す
    #   （`song_edits.apply_edits`）。移調は足し算、音量は最後の値。
    _rid = _note_request(ctx, what, "組み立て.声部", op="remove" if remove else "set", scope=v.title,
                         value={"title": v.title,
                                **({} if remove else {"transpose": int(transpose or 0), "gain_db": gain_db})},
                         supersedes=supersedes)
    # ★**点検待ちの束へ**（2026-10-04・案C）。消す・オクターブ移しは中、それ以外の幅の移調は大、音量は小
    _keys = (["adjust_voice.remove"] if remove else
             (["adjust_voice.transpose_octave" if abs(int(transpose)) == 12 else "adjust_voice.transpose"] if transpose else [])
             + (["adjust_voice.gain_db"] if gain_db is not None else []))
    _gate = note_edit(ctx, _keys, what)
    rest = " / ".join(f"{x.title}({len(x.notes)}音)" for x in voices) or "なし"
    return (f"{what}。いまの追加声部: {rest}。vpr({1 + len(voices)}トラック) 更新。"
            f"★**歌声は書き出し直しが要る**（前の歌声には直す前の声部が残っている）。{NEXT_EXPORT}"
            + (f"\n{_gate}" if _gate else "") + _ledger_note(_rid))


# ★`RegionNote` の docstring は道具の引数の説明としてコーディネーターに見える——**道具の説明だけを書き、経緯はここ（コメント）に置く**
#   （2026-10-03・再レビュー。以前は docstring に `start_beat`・MIDI 番号の `key` の経緯があり、使えない書き方を見せていた）。
#   ・高さは国際式の音名（2026-10-01）。以前は MIDI 番号で書かせ、`inspect_region` も MIDI 番号で見せていた——曲の表・ハモリ追加・
#     伴奏の `pitches` は国際式なので、1つの曲を2つの物差しで読み書きしていた。読み手は `song_llm.PITCH` の1つ
#     （読めない・鳴らせない音名は型の検証で差し戻す）。`key` プロパティは楽譜へ写すときの MIDI 番号
#   ・位置は曲の表と同じ `小節:拍`、言葉は声部と同じ1モーラ（2026-10-01・記法の仕様書 §15-2・ず-18）。以前は位置が楽譜の頭からの拍、
#     言葉が「かな1〜2モーラ」で伸ばす印の決まりが無かった。位置の拍が1小節の長さ以上かは曲の拍子を知る道具の中で見る
class RegionNote(BaseModel):
    """replace_region で渡す1つの音（区間の部分上書き用）。言葉は1モーラ、高さは国際式の音名、位置は `小節:拍`、長さは拍。"""
    lyric: str = Field(description="この音で歌う**1モーラのひらがな**（`か`・`きゃ`・`っ`・`ん`）。前の音を伸ばすなら `_`"
                                   "（`-` や「ー」ではない）。助詞は発音どおり（「は」→「わ」・「へ」→「え」・「を」→「お」）。"
                                   "inspect_region の現行ノートと同じ書き方")
    note: str = Field(description="高さ。国際式の音名（`C4`＝中央のド・`E5`・`F#4`・`Bb3`。変化記号は `#` と `b`、"
                                  "オクターブの数字は必ず付ける）。inspect_region の現行ノート・推定音高と同じ書き方")
    at: str = Field(description="開始位置 `小節:拍`（inspect_song の表・inspect_region と同じ書き方。小節は楽譜の頭から1始まり、"
                                "拍は4分音符＝1.00の0始まり・小数2桁。例 `12:1.50`＝12小節目の2拍目の裏）")
    beats: JsonFloat = Field(gt=0, description="長さ（拍・4分音符＝1.0。4分の3等分は `0.33`・`0.67`、8分の3等分は `0.17`・`0.83`）")

    @field_validator("note")
    @classmethod
    def _readable_note(cls, v: str) -> str:
        if bad := PITCH.problems(v):
            raise ValueError(bad[0])
        return v

    @field_validator("beats")
    @classmethod
    def _sounding_beats(cls, v: float) -> float:
        # 長さの決まりは声部と同じ1つ（§14・tick にして 0 になる長さは書けない）。2026-10-02 まで道具は 1/8拍未満を黙って伸ばしていた
        from harmony import beats_problem
        if why := beats_problem(v):
            raise ValueError(why)
        return v

    @field_validator("lyric")
    @classmethod
    def _one_mora(cls, v: str) -> str:
        # 1つの音の言葉の記法は声部と同じ1つ。助詞の「を」もここで差し戻る（詞と同じ `check_kana`。以前は機械が黙って「お」に直していた）
        from harmony import VOICE_WORD
        if bad := VOICE_WORD.problems(v):
            raise ValueError(bad[0])
        return v

    @field_validator("at")
    @classmethod
    def _readable_at(cls, v: str) -> str:
        from song_checks import POSITION                    # 位置の記法は表・声部と同じ1つ（拍の上限は道具の中で）
        if bad := POSITION.problems(v):
            raise ValueError(bad[0])
        return v

    @property
    def key(self) -> int:
        """MIDI 番号（楽譜の `SongNote.key` へ写す値。`song_edits.region_notes` が読む）。"""
        return int(pitch_name_to_midi(self.note))

@coordinator.tool
@_traced
def replace_region(ctx: RunContext[MusicSession], start: str, end: str,
                   notes: list[RegionNote], supersedes: str = "", display: list[DisplayEdit] | None = None) -> str:
    """確定スコアの区間 [start, end)（どちらも `小節:拍`＝inspect_song の表と同じ書き方。終わりは含まない）のノートを、
    渡した notes で**部分上書き**（全文でなく区間のみ）。各音は `at`（`小節:拍`）・`beats`（長さ・拍）・`note`（国際式の音名）・
    `lyric`（1モーラ・伸ばすなら `_`）で書く。
    ★**音は区間の中に、重ならずに並べる**: `at` は区間 [start, end) の中だけ（外の元の音は消えないので、外に置くと同じ位置に2音になる）、
      歌は1度に1音なので、書いた音どうし・区間の外の次の音に長さが重ならないように書く。外れていたら**何も書き換えずに返す**
      （区間の外まで直すなら区間を広げる）。区間の最初の音に「っ」は書けない（前の音を保つ形に訳すため、前の音も区間に含める）。
    詰め込み等で壊れた区間を inspect_region の実オンセット/音高を見て作り直すのに使う。グリッド量子化や全再整列は
    しない＝カバーの他区間は保持。書換後 vpr/仮歌を再書出。

    ★**詞は「発音どおり」に書く**（2026-09-20）——助詞『は』→『わ』／『へ』→『え』／『を』→『お』。
      **V6 は字のまま発音する**ので「さいれん**は**」は「さいれんハ」と歌われる。
      新規作成の詞はこの決まりで作られているが、**書き直すとここが戻ってしまう**
      （実際に『季節を追い越せ』が v1「さいれんわ」→ v2「さいれんは」に戻っていた）。
      ※「を」は書けない（差し戻す）。**「は」「へ」は語の一部のこともある**ので直さず報せる。

    ★**促音は「っ」とそのまま書く**（2026-09-20）。**前の音を伸ばすときだけ `_` を書く**（`-` は書けない）。
      「っ」は機械が**前の音を保つ形＋終わり際の無音**に変える（新規作成と同じ作法）。
      伸ばす印で書かれると「ただ伸ばす」のか「促音」のかが区別できず、**詰まりが入らない**
      ——実際に「くらくな**っ**ていく」が「くらくな**ー**ていく」になっていた。

    ★**歌い方と崩しは、元の音と同じ開始位置に置いた音にだけ引き継がれる**（2026-09-27）。
      開始位置を動かした音・新しく足した音は素で歌う（返事に「歌い方・崩しが付いていない音」として出る）。
      歌い方を保ちたい音は、元の開始位置のまま置く。

    ★**歌う詞が変わるときは、見せる歌詞（`display`）も一緒に渡す**（2026-10-12）。見せる歌詞は字幕・投稿に使う漢字まじりの詞で、
      歌う詞だけ直すと古いまま残る。区間の歌うかなが変わるのに `display` が無ければ、**何も書き換えずに**行番号つきの見せる歌詞を返す
      ——区間にかかる行を `display=[{line, text}]` で書き直して、もう一度呼ぶ。歌う詞が変わらないとき（高さ・長さだけ）は要らない。

    ★置いた音符は台帳に残り、あとで工程から作り直しても**旋律と詞が変わらなければ当て直される**。
      旋律か詞を作り直すと「当て直し待ち」になる——新しい楽譜の同じ意図の場所に置き直し、
      `supersedes` にその id を指す。
    ★**長さに関係なく「大きい編集」**: 書き出し（export_vocal）の前に最終の点検役がまとめて見る（書いた音は工程の点検役を通っていないため）。
      カバーは最終の点検役を通らないので、書き出しの前に review_progress（最終検問）を必ず呼ぶ。"""
    s = ctx.deps.score
    if s is None:
        return "曲が未生成。先に align_vocal か load_existing_song を。"
    if why := form_refusal(ctx, "replace_region"):        # インストは歌0音——ここで書くと歌の無い曲に歌が生える（§7-2）
        return why
    if why := lead_refusal(ctx, "replace_region"):        # 楽器で重ねた主旋律が古いまま残る（F10②）
        return why
    from song_edits import at_text, notes_value, patch_region, region_notes, score_bar_ticks, span_text, span_ticks
    Qd = 480
    bar = score_bar_ticks(s)
    if why := _supersedes_problem(ctx, supersedes):
        return why
    # 位置が読めない・拍が1小節を超える・区間の外の音・書いた音どうしや区間の外の次の音との重なり＝書き換えずに返す
    # （区間の外と重なりは 2026-10-02・洗い出し H11・H12。以前は外の音を置いて2音にし、重なりは前の音を黙って切った）
    try:
        s_tick, e_tick = span_ticks(start, end, bar)
        new, bare = region_notes(s.vocal_notes, s_tick, e_tick, notes, bar)
    except ValueError as e:
        return f"書き換えなかった: {e}"
    # ★**見せる歌詞を同じ場面で書き換える**（2026-10-12・本人了承）。歌う詞が変わるのに見せる歌詞が古いまま残っていた
    #   （『てのひらのシグナル』v16 は v7 の詞のまま）。変わるのに `display` が無ければ、何も書かずに行番号つきで返す
    from song_edits import (apply_display_edits, display_problems, numbered_display, sung_kana_of, _fold)
    disp_edits = {int(d.line): d.text for d in (display or [])}
    shown_before = getattr(s, "lyrics_display", "") or ""
    if shown_before:
        _key = lambda ns: _fold(sung_kana_of(ns)).replace("っ", "")      # noqa: E731（っ は前の音を保つ形に訳されて残る）
        old_in = [n for n in s.vocal_notes if s_tick <= n.start_tick < e_tick]
        words_changed = _key(old_in) != _key(new)
        if disp_edits:
            if bad := display_problems(disp_edits, s, new):
                return "書き換えなかった: 見せる歌詞の指定が読めない——" + "／".join(bad)
        elif words_changed:
            try:
                _rec = _record_of(ctx)
            except Exception:                                # noqa: BLE001（記録が読めなくても行番号は見せる）
                _rec = None
            return ("書き換えなかった: この区間の歌う詞が変わる（"
                    f"「{sung_kana_of(old_in)}」→「{sung_kana_of(new)}」）のに、見せる歌詞（字幕・投稿用）の書き直しが無い。"
                    "歌う詞だけ直すと見せる歌詞が古いまま残る。区間にかかる行を `display=[{line, text}]` で書き直して、もう一度呼ぶ"
                    "（読みは歌うかなと同じにする。新しい詞が別の行にまたがるなら、かかる行を全部）:\n"
                    + numbered_display(_rec, s, bar, near=(s_tick, e_tick)))
    elif disp_edits:
        disp_edits = {}                                      # 見せる歌詞が無い曲（カバーなど）には書かない
    # ★**区間の最初の音に「っ」は書けない**（2026-10-02・洗い出し 2-3）。「っ」は前の音を保つ形に訳すが、区間の最初の音には
    #   前の音が無い（区間の外の音は書き換えない）。以前は訳さずに「っ」のまま V6 の歌詞にしていた
    if new and new[0].lyric == "っ":
        return (f"書き換えなかった: 区間の最初の音（{at_text(new[0].start_tick, bar)}）が「っ」。「っ」は前の音を保つ形＋終わり際の無音に"
                "訳すので、前の音が区間の中に要る——区間の始まりを前の音まで広げ、前の音も一緒に書く。")
    # ★★**促音「っ」をここでも仕上げる**（2026-09-20）。
    #   新規作成の `fit_phrase` は「っ」を **`-`（前の音を保つ）＋終わり際の無音**に変えるが、
    #   **区間の書き直しには同じ処理が無かった**。そのため `replace_region` で直した区間だけ
    #   促音の詰まりが消え、「くらくな**っ**ていく」が「くらくな**ー**ていく」に聞こえていた
    #   （『加速させて』の 17・77小節で実測——詰まり 0ms。直っていた所は 83ms あった）。
    #   ★**`-` を LLM に書かせない**——`-` は「ただ伸ばす」なのか「促音」なのか区別が付かない。
    #   「っ」と書いてもらい、**機械がここで詰まりに変える**（新規作成と同じ作法）。
    # ★★**歌う表記へ直す**（2026-09-20）。新規作成の詞は「発音どおりに書く」（を→お・は→わ・
    #   へ→え）という決まりで作られるが、**区間を書き直す経路には何の正規化も無かった**。
    #   実害: 『季節を追い越せ』は v1 で「さいれん**わ**／ぼく**お**」だったのに、
    #   書き直した v2 で「さいれん**は**／ぼく**を**」に戻っていた。V6 は字のまま発音するので
    #   **「さいれんハ」と歌う**。
    #   ・**「を」は必ず助詞**なので書けない（型の検証 `RegionNote._one_mora`＝`check_kana` が差し戻す。2026-10-01 まではここで黙って「お」に直していた
    #     ——記法の仕様書 §0-7「黙って読み替えない」）
    #   ・**「は」「へ」は語の一部のこともある**（はな・へや）ので、**直さずに報せる**
    #     ——機械が勝手に直すと「はな」が「わな」になる
    warn_ha = [f"{at_text(n.start_tick, bar)}の「{n.lyric}」" for n in new if n.lyric in ("は", "へ")]

    from song_place import _SOKUON_GAP_BEATS
    for i, n in enumerate(new):                            # `new` は開始位置の順（`region_notes`）＝1つ前が前の音
        if n.lyric != "っ" or i == 0:
            continue
        cut = min(_SOKUON_GAP_BEATS, n.duration_tick / Qd * 0.3)
        new[i] = n.model_copy(update={
            "lyric": "-",                                      # 前の音を再発音せず保つ（V6 のタイ）
            # 鳴る長さは書いた長さの7割以上（1/8拍へ伸ばすのは 2026-10-02 にやめた——書いた長さを超えて次の音に重なる）
            "duration_tick": max(1, int(round((n.duration_tick / Qd - cut) * Qd)))})
    _trimmed: list[str] = []
    merged, old_n = patch_region(s.vocal_notes, s_tick, e_tick, new, _trimmed, bar)   # 重なりは次音の開始までに切る
    song_dir = SONGS / ctx.deps.song_id
    # ★**詞の全文も音符から組み直す**（2026-09-27・監査）。区間を書き直しても `lyrics_full` が古いままで、
    #   作ったときの詞と音符が食い違っていた（`song_table` が「食い違っている」と出していたもの）。
    _upd = {"vocal_notes": merged, "lyrics_full": "".join(n.lyric for n in merged)}
    if disp_edits:
        _upd["lyrics_display"] = apply_display_edits(s, disp_edits)       # 歌う詞と見せる歌詞を同じ場面で
    ctx.deps.score = s.model_copy(update=_upd)
    st = ctx.deps.section
    render_vocal_and_mix(ctx.deps.score, song_dir, log=(st.get("log") if st else None))
    _trace(ctx.deps.song_id, "replace_region", region=(start, end), old=old_n, new=len(new))
    # ★**台帳にも書く**（2026-09-25）。旋律と詞が変わらない作り直しなら機械が同じ音符を当て直し、
    #   変わったら「当て直し待ち」にしてコーディネーターが新しい旋律へ当て直す
    #   （`docs/plan/song_revision_design.md` §10-2）。その材料として「どこを・どう置いたか」を残す。
    _rid = _note_request(ctx,
                         "音符を置き直した: " + " ".join(f"{n.lyric}:{n.note}@{n.at}×{n.beats:g}"
                                                        for n in notes)[:300],
                         "音符", scope=span_text(s_tick, e_tick, bar),
                         value=notes_value(s_tick, e_tick, new, disp_edits), supersedes=supersedes)
    # 返事は inspect_region の現行ノートと同じ書き方（かな:音名@小節:拍×長さ拍・伸ばしは `_`・音名は区間の調の向き）
    from tools.diagnosis import pitch_speller
    sp = pitch_speller(ctx)
    nl = " ".join(f"{_hold_shown(n.lyric)}:{sp(n.key, n.start_tick)}@{at_text(n.start_tick, bar)}×{n.duration_tick / Qd:.2f}"
                  for n in sorted(new, key=lambda n: n.start_tick))
    ha = (f"\n★**確認して**: {' / '.join(warn_ha)} が残っている。"
          "**助詞なら「わ」「え」に直す**（V6 は字のまま発音するので『さいれんハ』と歌う）。"
          "語の一部（はな・へや）ならそのままでよい——**機械では区別できないので、あなたが決める**。"
          if warn_ha else "")
    out_of_range = ("\n★**範囲の外も変えた**: " + "／".join(_trimmed)) if _trimmed else ""
    kept_expr = sum(1 for n in new if n.tuning is not None or n.artic)
    expr = (f"\n歌い方・崩しは、元の音と同じ開始位置の音に引き継いだ（{kept_expr}音）。" if kept_expr else "")
    if bare:
        expr += ("\n★**歌い方・崩しが付いていない音**: " + "／".join(bare)
                 + "——元の音と開始位置が違うので、機械は歌い方（しゃくり・揺れ・音量）も崩しも付けていない"
                 "（素で歌う）。この道具では付けられないので、周りと歌い方が揃わず気になるなら最終報告でマスターに伝える。")
    # ★**点検待ちの束へ**（2026-10-04・案C）。コーディネーターが書いた歌の音は1段目の点検役を通っていない＝長さに関係なく大
    _gate = note_edit(ctx, "replace_region", f"区間 {span_text(s_tick, e_tick, bar)} の歌を {old_n}音→{len(new)}音に置き換えた")
    shown = ("\n見せる歌詞を更新した: " + "／".join(f"行{k}「{t}」" for k, t in sorted(disp_edits.items()))) if disp_edits else ""
    # ★**小節ごとの文字の数を機械が数えて返す**（2026-10-14）。「1小節に6文字」と「2小節に6文字」を取り違えたり、
    #   「ひびかせて」を4文字と数えたりして、頼まれた密度と違う形になった（『てのひらのシグナル』v23〜25・v27）
    from collections import Counter as _Counter
    _per_bar = _Counter(int(n.at.split(":")[0]) for n in notes if n.lyric != "_")
    if _per_bar:
        shown += ("\n小節ごとの文字の数（書いた1モーラずつ・伸ばしの `_` は数えない・機械が数えた）: "
                  + "／".join(f"{b}小節 {_per_bar.get(b, 0)}文字"
                             for b in range(s_tick // bar + 1, (e_tick - 1) // bar + 2))
                  + "——頼まれた文字の数（1小節にいくつか・2小節でいくつか）と合っているか確かめる")
    return (f"区間 {span_text(s_tick, e_tick, bar)}（終わりは含まない）を上書き: {old_n}音→{len(new)}音。新ノート: {nl}。"
            f"vpr/仮歌 更新。{shown}{ha}{out_of_range}{expr}"
            f"\nまだズレていれば inspect_region→replace_region を繰返す。{NEXT_EXPORT}"
            + (f"\n{_gate}" if _gate else "") + _ledger_note(_rid))

@coordinator.tool
@_traced
def redistribute_by_gaps(ctx: RunContext[MusicSession], start: str, end: str) -> str:
    """区間 [start, end)（どちらも `小節:拍`＝inspect_song の表と同じ書き方。終わりは含まない）の音節を
    **無音ギャップで区切った各セグメントの実オンセット数に比例**して前後へ割り振り直す（決定論・LLM判断不要）。
    フレーズがギャップを跨いで分断/詰め込みされた時に「ギャップ前N音／後M音」を機械的に正す＝**手前セグメント末尾音の巻き込みも詰め込みも同時解消**。
    各音節の**音高はそのまま保持**（歌詞・音高は触らずタイミングのみ）。位置ズレ修正の下処理に使い、残りはLLMが微調整。"""
    import numpy as _np

    import librosa
    from render_base import load_wav
    from song_edits import at_text, score_bar_ticks, span_ticks
    from song_schema import SongNote
    s, st = ctx.deps.score, ctx.deps.section
    if s is None:
        return "曲が未生成。先に align_vocal か load_existing_song を。"
    if why := form_refusal(ctx, "redistribute_by_gaps"):
        return why
    Qd, tempo = 480, s.tempo
    bar = score_bar_ticks(s)
    try:
        s_tick, e_tick = span_ticks(start, end, bar)
    except ValueError as e:
        return f"動かさなかった: {e}"
    start_beat, end_beat = s_tick / Qd, e_tick / Qd        # ここから先の計算は楽譜の頭からの拍（返事は `小節:拍` で書く）
    reg = sorted([n for n in s.vocal_notes if s_tick <= n.start_tick < e_tick], key=lambda n: n.start_tick)
    if len(reg) < 2:
        return "区間の音節が少なく割り振り不要。"
    syl = [(n.lyric, n.key) for n in reg]                  # 既存の(歌詞,音高)を順に保持
    vw = (st or {}).get("vocals_wav") if st else None
    origin = s.origin_abs_sec if s.origin_abs_sec is not None else (st or {}).get("start_sec")
    if not vw or origin is None or not Path(vw).exists():
        return "音声参照なし→割り振り不可（音声カバーのみ対応）。"
    y = load_wav(Path(vw)).astype(_np.float32)
    ws, we = origin + start_beat * 60.0 / tempo, origin + end_beat * 60.0 / tempo
    seg = y[int(ws * SR):int(we * SR)]
    on = [start_beat + float(o) * tempo / 60.0 for o in librosa.onset.onset_detect(y=seg, sr=SR, units="time", backtrack=True)]
    hop = int(0.02 * SR)                                   # 無音ギャップ検出（score基準＝音符と同フレーム）
    rms = _np.array([_np.sqrt(_np.mean(seg[i:i + hop] ** 2)) for i in range(0, max(1, len(seg) - hop), hop)])
    thr = float(rms.max()) * 0.08
    gaps, i, m = [], 0, len(rms)
    while i < m:
        if rms[i] < thr:
            j = i
            while j < m and rms[j] < thr:
                j += 1
            if (j - i) * 0.02 > 0.35:
                gaps.append((start_beat + (i * 0.02) * tempo / 60.0, start_beat + (j * 0.02) * tempo / 60.0))
            i = j
        else:
            i += 1
    segr, prev = [], start_beat                            # ギャップ間のセグメント
    for gs, ge in sorted(gaps):
        segr.append((prev, gs))
        prev = ge
    segr.append((prev, end_beat))
    seg_on = [[b for b in on if s0 - 0.1 <= b < s1 + 0.1] for s0, s1 in segr]
    total = sum(len(o) for o in seg_on) or 1
    L, bounds, acc = len(syl), [0], 0                      # 音節を各セグメントへ「オンセット数に比例」して割り当て
    for i in range(len(segr) - 1):
        acc += len(seg_on[i])
        bounds.append(min(L, round(L * acc / total)))
    bounds.append(L)
    placed = []                                            # (lyric, key, beat)
    for i, (s0, s1) in enumerate(segr):
        chunk, ons, C = syl[bounds[i]:bounds[i + 1]], seg_on[i], bounds[i + 1] - bounds[i]
        for j, (ly, ky) in enumerate(chunk):
            if len(ons) >= C and C > 0:                    # オンセットが足りる→実アタックへ
                t = ons[round(j * (len(ons) - 1) / (C - 1))] if C > 1 else ons[len(ons) // 2]
            else:                                          # 足りない→セグメント内に等間隔
                t = s0 + (s1 - s0) * (j / max(1, C))
            placed.append((ly, ky, t))
    placed.sort(key=lambda x: x[2])
    for k in range(1, len(placed)):                        # 単調増加を保証
        if placed[k][2] <= placed[k - 1][2]:
            placed[k] = (placed[k][0], placed[k][1], placed[k - 1][2] + 0.1)
    rns = []
    for k, (ly, ky, t) in enumerate(placed):
        nb = placed[k + 1][2] if k + 1 < len(placed) else end_beat
        rns.append(SongNote(lyric=ly, key=ky if 0 <= ky <= 127 else 67,
                            start_tick=int(round(t * Qd)), duration_tick=max(Qd // 8, int(round((nb - t) * Qd)))))
    kept = [n for n in s.vocal_notes if not (s_tick <= n.start_tick < e_tick)]
    merged = sorted(kept + rns, key=lambda n: n.start_tick)
    for i in range(len(merged) - 1):
        ov = merged[i].start_tick + merged[i].duration_tick - merged[i + 1].start_tick
        if ov > 0:
            merged[i] = merged[i].model_copy(update={"duration_tick": max(Qd // 8, merged[i + 1].start_tick - merged[i].start_tick)})
    ctx.deps.score = s.model_copy(update={"vocal_notes": merged})
    render_vocal_and_mix(ctx.deps.score, SONGS / ctx.deps.song_id, log=(st.get("log") if st else None))
    info = " ／ ".join(f"[{at_text(s0 * Qd, bar)}〜{at_text(s1 * Qd, bar)}]{bounds[i + 1] - bounds[i]}音←{len(seg_on[i])}オンセット"
                       for i, (s0, s1) in enumerate(segr))
    _trace(ctx.deps.song_id, "redistribute_by_gaps", region=(start, end), segs=len(segr))
    return (f"ギャップ割り振り直し: {L}音を{len(segr)}セグメントへ決定論配置（{info}）。"
            f"音高は保持・タイミングのみ。vpr/仮歌 更新。VOCALOIDで確認し、必要なら inspect_region→replace_region で微調整。")

@coordinator.tool
@_traced
def fix_song_timing(ctx: RunContext[MusicSession], start: str = "", end: str = "") -> str:
    """evaluate_song が指摘した timing 問題（**拍グリッド外で始まる音／音の重なり**）だけを決定論的に直す。
    **歌い出しの位置は動かさない**（設計で拍の裏から入る歌を、伴奏に対してずらさないため）。
    ・格子の外で始まる音 → 開始と終わりを16分の格子へ寄せる
    ・重なり → 前の音を次の音の開始までに縮める
    ・**それ以外の音は長さも位置も触らない**（スタッカートや短い音をそのまま残す）
    ・`start`〜`end`（どちらも `小節:拍`＝inspect_song の表と同じ書き方。終わりは含まない。片方だけでもよい）を渡すと
      その範囲だけ直す。**スウィング・タメなど、意図して格子から外した区間は
      範囲に入れない**（直すと意図が消える）。1か所ずつ直したいなら replace_region のほうが確か。
    返事に**動かした音を全部**書く。
    ※**オリジナル(記憶生成)専用**。カバー（実演奏タイミングを再整列したもの）には使わない。"""
    if ctx.deps.score is None:
        return "曲が未生成。"
    if why := form_refusal(ctx, "fix_song_timing"):
        return why
    if why := lead_refusal(ctx, "fix_song_timing"):
        return why
    if ctx.deps.faithful:    # カバーをグリッド量子化すると vad再整列(実演奏タイミング)が壊れ、前奏削除＋ノート押し出しで尺も狂う＝禁止
        return ("カバーは原曲の実演奏タイミングを再現したものなので fix_song_timing（拍グリッド量子化）は使わない"
                "（再整列が壊れ・前奏が消え・尺がずれる）。タイミングは evaluate_timing で参考確認のこと。")
    from song_edits import at_tick, score_bar_ticks, span_ticks
    bar = score_bar_ticks(ctx.deps.score)
    try:                                                   # 範囲は `小節:拍`（空＝その端は曲の端まで）
        lo = at_tick(start, bar) / 480 if strip_ascii_ws(start) else None   # 前後は半角の空白とタブだけ（§0-6）
        hi = at_tick(end, bar) / 480 if strip_ascii_ws(end) else None
        if lo is not None and hi is not None:              # 両端を書いたら、逆向き・空の範囲は断る（区間の読み手は1つ・§15-2・2026-10-03）
            span_ticks(start, end, bar)
    except ValueError as e:
        return f"直さなかった: {e}"
    # ★**動かした音を全部返す**（2026-09-27・目録の道具-2）。以前は全音符の長さを16分へ丸め、重なりは後ろへ押し出し、
    #   「補正完了」としか返さなかった（弱起が半拍ずれ、曲頭の間が潰れても分からなかった）。
    moved: list[str] = []
    ctx.deps.score = fix_timing(ctx.deps.score, start_beat=lo, end_beat=hi, moved=moved)
    rest = check_timing(ctx.deps.score)
    _trace(ctx.deps.song_id, "fix_song_timing", moved=len(moved), rest=len(rest),
           region=(start, end))
    if moved:
        _append_log(ctx.deps.song_id, "[fix_song_timing] 動かした音:\n" + "\n".join(f"  {m}" for m in moved))
    return (f"補正: {len(moved)}件動かした" + (f"（範囲 {start or '曲頭'}〜{end or '曲の終わり'}）"
                                            if lo is not None or hi is not None else "")
            + ("。\n" + "\n".join(f"・{m}" for m in moved[:40])
               + (f"\n・ほか {len(moved) - 40}件（全部は {song_files.BUILD_LOG}）" if len(moved) > 40 else "")
               if moved else "（直す所は無かった）")
            + "\n残課題: " + ("なし" if not rest else " / ".join(rest))
            + "\n★意図して置いた位置を動かしていたら、replace_region で元に戻す。")

# ★`align_to_reference`（参照の音高と歌い回しに合わせ直す道具）は外した（2026-09-27・目録の道具-3）。
#   旋律を作り直して歌い方・歌の崩し・調声を落とし、音節の過不足は切り捨てか埋めで吸収していた（LLM の決定を機械が変える）。
#   いまのオリジナル曲は参照を持たず、カバーでは使わない決まりだったので、使われる場面が無かった。

# mix_song ツールは 2026-08-10 に削除。vpr が伴奏入り（音量バランス設定済み）になり、
# VOCALOID6 のオーディオミックスダウン一発で完成ミックスが出るため、
# 「歌声だけを書き出してもらいプログラム側で混ぜる」工程そのものが不要になった。
# 代わりに、V6 の**既定の名前**（vpr と同じ名前の wav＝`song_files.v6_raw`）で保存された完成ミックスを取り込むのが下の adopt_vocal_export。


def _v6_raw_names(song_dir: Path) -> list[str]:
    """曲フォルダにある V6 の生の書き出し（`song_files.v6_raw` の形＝vpr と同じ名前の wav）。"""
    return sorted(p.name for p in song_dir.iterdir()
                  if p.is_file() and (got := song_files.split_version(p.name)) and got[0] == "vocal.wav")


@coordinator.tool
@_traced
def adopt_vocal_export(ctx: RunContext[MusicSession], song_id: str = "", vocal_filename: str = "") -> str:
    """マスターが VOCALOID6 で書き出した**完成ミックスを取り込んで完成にする**（台帳を「完成」へ）。

    ★**ふだんは使わない**——歌声は `export_vocal` が自動で書き出す（2026-09-05〜）。これは
      **`export_vocal` が失敗して、マスターに手で書き出してもらった**ときの受け口。

    vpr は伴奏入りなので、V6 のミックスダウンは歌＋伴奏が揃った完成品（アカペラは声だけ）。**混ぜ直さない**
    （足すと伴奏が二重になる）。V6 は vpr と同名で出すので、マスターは**既定の名前のまま**保存すればよい。
    取り込むと仕上げて完成品（歌もの `master_song_v{N}.wav`・アカペラ `master_acappella_v{N}.wav`）にする。
    **インストには歌声の書き出しが無い**ので使わない（断る。完成品は render_final が作る）。
    『書き出したよ』『V6 でミックスダウンした』と言われたら呼ぶ。
    vocal_filename=使う書き出しを明示指定（例 'vocal_v2.wav'。版を指定されたときだけ）。
    """
    from song_pipeline import adopt_real_vocal_as_final
    sid = song_id or ctx.deps.song_id
    song_dir = SONGS / sid
    if not sid or not song_dir.exists():
        return f"見つからない: {song_dir}（song_id={sid!r}）"
    ctx.deps.song_id = sid
    if why := form_refusal(ctx, "adopt_vocal_export"):
        return why
    if vocal_filename and not (song_dir / vocal_filename).exists():
        avail = _v6_raw_names(song_dir)
        return f"指定の書き出しが無い: {vocal_filename}。この曲にあるのは: {', '.join(avail) or '（書き出しなし）'}"
    # 仕上げて `master_<形>_v{N}.wav` へ付け替える（書き出しの側の1か所）。名前に版が無い書き出し（`vocal.wav`）は作業中の版として
    out = adopt_real_vocal_as_final(song_dir, vocal_filename or None,
                                    version=_working_version(song_dir) or _current_version(song_dir) or None)
    if out is None:
        return ("まだ書き出しが見つからない。**先に export_vocal で自動の書き出しを試す**。"
                f"それが失敗したときだけ、{song_files.versioned(song_files.VPR, _current_version(song_dir) or 1)} を VOCALOID6 で開いて"
                "オーディオミックスダウンし、曲のフォルダに保存してもらう（名前は既定のままでいい）。"
                "そのあともう一度呼ぶ。")
    _update_index(sid)                                     # 台帳の状態を「完成」へ
    return (f"完成品にした: {out.name}（{out.stat().st_size / 1e6:.1f}MB・V6 の書き出しを仕上げて付け替えた）。"
            f"これで歌えるようになった＝マスターに聴かせられる。")
# 完成品は `master_<形>_v{N}.wav`（`song_files.master`。songs.md の状態はファイルの有無から自動導出）。
# **編集中の試聴ミックス（draft_mix）は 2026-09-06 に廃止**
# ——`export_vocal` が実歌声を90秒で出すので、サイン波で仕上がりを判断する場面が無くなった。


@coordinator.tool
@_locked
def rearrange_song(ctx: RunContext[MusicSession], request: str) -> str:
    """**曲を作り直さずにアレンジだけ変える**（編成→伴奏→ハモリをやり直す）。

    「和楽器風にして」「バンドアレンジにして」「サビをもっと厚く」のような、
    **旋律も歌詞も変えずに響きだけ変えたい**依頼に使う。**カバーにも使える**——
    カバーは区間・調・コードを曲の記録から読み（無ければ初めの1回だけ、原曲の旋律・伴奏・音声を材料に LLM が書いて記録に残す）、
    そのコードの上で組み直す。旋律・歌詞・タイミングは触らない。

    ※ 部分的な差し替えで足りるなら `replace_accompaniment_region` のほうが安い。
      こちらは**編成から選び直す**ので、曲全体の響きが変わる。
    ※ オリジナル曲では `remake_from_step('編成', request)` と同じ動きになる（工程の記録から作り直す）。
    ※ **アカペラでは使えない**（伴奏を作ってしまう。断って行き先を返す）。インストは編成から作り直す（ハモリは無い）。
    """
    sid = ctx.deps.song_id
    song_dir = SONGS / sid
    if not sid or not song_dir.exists():
        return f"曲が未ロード。先に load_existing_song を。（song_id={sid!r}）"
    if why := form_refusal(ctx, "rearrange_song"):        # アカペラに伴奏を作らない（§7-2）
        return why
    _trace(sid, "rearrange_song.in", request=request[:120])
    # ★**オリジナル曲は「編成から作り直す」と同じにする**（2026-09-25・段階3）。
    #   以前はオリジナルもカバーの経路（伴奏 MIDI から和音を起こし直す）を通っていて、
    #   **工程の記録に何も書かなかった**——次に詞を直すと、記録にある前の編成へ戻った。
    #   記録から作り直せば、彩り・声部・奏法も新しい編成に合わせて作り直され、台帳にも残る。
    #   （`remake_from_step` 本体を呼ぶ。こちらもロックを持っているので、ラッパーは通さない）
    try:
        _rec = _record_of(ctx)
    except Exception:
        _rec = None
    if _is_original(_rec):
        _trace(sid, "rearrange_song.out", kind="to_remake", step="編成")
        return remake_from_step.__wrapped__(ctx, "編成", request)
    try:
        from song_arrange_cover import arrange_song
        from song_cover_load import load_cover
        song = load_cover(song_dir)                        # 曲フォルダ → Song（構造に載せる）
        made = _cover_ready(ctx, song_dir, song, _rec)     # 区間・調・コード（記録にあれば読む・無ければ LLM が書いて残す）
        with _hand_scope(ctx.deps.score, "編成"):          # 人が直した範囲を先に知らせる（道具-8）
            song = arrange_song(song, request, verbose=False)  # 編成 → 伴奏 → ハモリ
    except StepUnfixed as e:                               # 形式の所見が直らずに止まった（2026-09-30）
        return _stopped_edit(sid, "rearrange_song", e,
                             how="**アレンジは変えていない**（元の版はそのまま残っている）。\n"
                                 "続けるときは **rearrange_song(request=同じ頼み)** でもう一度"
                                 "（上の所見が直るように、という一言を頼みに足してよい）。")
    except Exception as e:
        _trace(sid, "rearrange_song.out", kind="error", reason=f"{type(e).__name__}")
        return (f"アレンジをやり直せなかった（{type(e).__name__}: {e}）。"
                f"伴奏 MIDI が無い曲かもしれない。部分的な差し替えなら "
                f"replace_accompaniment_region、音量だけなら adjust_song を使う。")
    # ★**いまの曲から作り直したので、作り直しで消える声部の手直しだけを当て直す**（2026-10-01・別件①）。以前は台帳を全部当て直し、
    #   版に入った移調をもう一度かけていた（+2 が +4）。移調・速さ・置き直した音符は読んだ楽譜にもう入っている
    from song_edits import apply_edits, cover_reapply
    song.score, _done, _ = apply_edits(song.score, cover_reapply(list(_rec.requests) if _rec is not None else []),
                                       notes_changed=False)
    from manual_import import carry_hand_edits             # 人が手で直したもの（楽譜の手の直しの欄）
    _carried: list[str] = []
    song.score, _hand = carry_hand_edits(ctx.deps.score, song.score, "編成", carried=_carried)
    ctx.deps.score = song.score
    from song_pipeline import compose as _compose
    _compose(sid, song.score)                              # 楽譜・伴奏・vpr・仮歌・mix を書き直す
    _rid = _note_request(ctx, request, "編成", history=made)    # 台帳だけの記録に残す（今回作った区間・コードの記録も）
    _snapshot_version_if_changed(song_dir)
    _update_index(sid)
    ins = song.instrumentation
    _trace(sid, "rearrange_song.out", instruments=len(getattr(ins, "instruments", []) or []),
           harmony=getattr(ins, "harmony", "none"))
    from song_cover_load import chords_note, key_note
    return (f"アレンジをやり直した: 編成 {len(getattr(ins, 'instruments', []) or [])}本"
            f"（ハモリ={getattr(ins, 'harmony', 'none')}）。"
            f"旋律と歌詞は原曲のまま。export_vocal で実歌声にして確認できる。"
            f"\n{key_note(song)}"
            + (f"\n{chords_note(song)}" if chords_note(song) else "")   # コードがどこから来たか（LLM が書いた／記録から・2026-10-02）
            + "\n（コード・伴奏の対旋律・リフと声部は C 基準で書かせ、この調へ移した。調が違っていたらマスターに伝える）"
            + (f"\n前に手で直したことを当て直した: {' / '.join(_done)}" if _done else "")
            + (f"\n人が手で直したものを引き継いだ: {' / '.join(_carried)}" if _carried else "")
            + (f"\n★{' / '.join(_hand)}" if _hand else "") + _ledger_note(_rid))


@coordinator.tool
@_locked
def remake_from_step(ctx: RunContext[MusicSession], step: str, request: str,
                     scope: str = "", supersedes: str = "") -> str:
    """**任意の工程から曲を作り直す**（人間と同じように、途中からやり直す）。

    「サビの歌詞を書き直して」「メロディをもっと跳ねる感じに」「コード進行を明るく」
    「テンポと構成を変えて」のような、**響き以外を直したい**依頼に使う。
    その工程を**読んでいる工程も一緒にやり直す**ので、下流が置いていかれない
    （読んでいない工程はそのまま——歌い方を直しても編曲は作り直さない）。

    ★**頼まれたことは台帳に残る**（2026-09-25）。この先また作り直しても、この頼みは守られる。
    `scope` … 範囲（例: `サビ`。空なら全体）。
    `supersedes` … **この頼みで置き換える古い頼みの id**（`load_existing_song` の一覧に出る `r…`。
    複数ならカンマ区切り）。「さっきの『明るく』はやめて、落ち着いた感じに」のように
    **前の頼みを打ち消すときだけ**指す。指さない古い頼みは生きたまま残る。

    ★**手で直したこと（移調・速さ・消した声部・足した声部・置き直した音符）は作り直しのあとに当て直す**。
    旋律か詞を作り直したときは、足した声部と置き直した音符は当てずに**「当て直し待ち」**として返す——
    新しい楽譜の同じ意図の場所に当て直し（`supersedes` にその番号）、置き場が無ければ最終報告に書く。

    `step` に指定できるのは: 設計 / コード / グルーヴ / 旋律 / 詞の設計 / サビの詞 / 詞 / 曲名 /
    歌い方 / 歌の崩し / 編成 / 伴奏 / ハモリ / 声部 / 彩り / 奏法 の名前を**この綴りのまま**
    （英語のキー名・言い換え〔リズム・しゃくり・リフ …〕は読まずに書き直しを頼む。2026-10-02 まで読み替えていた）。
    **テーマからは作り直せない**（別の曲になるので、新しく作るほうが早い）。
    **曲名だけを変えたい**（『タイトルを変えて』『もっと短い曲名に』）ときは `曲名` から。ほかは作り直さない。
    **曲の形で無い工程も作り直せない**——インストに詞の設計〜詞・歌い方・歌の崩し・ハモリ・声部は無く、
    アカペラに伴奏・彩り・奏法は無い（断って、作り直せる工程を返す）。歌や伴奏を足す・外すのは `change_form`。

    ★**途中で止まった曲の続きもこれで作る**——`show_after` で手を止めた曲も、
    **途中で失敗して止まった曲**も、**工程が書いたものの形式が直らずに止まった曲**（2026-09-30）も同じ。
    止まった曲は、止まった工程を `step` に、マスターの頼みを `request` に渡す（その工程の LLM が頼みを読んで書き直す）。
    作り直しの途中でまた止まったら、元の版はそのまま（返事にどう頼み直すかが出る）。`load_existing_song` が「どの工程から」を教えるので、
    その工程を指定すれば済んだぶんは費用なしで復元される（**新しく作り直さない**）。

    ★**カバーでも使える**（2026-09-16）。ただしカバーがやり直せるのは
    **コード / 編成 / 伴奏 / ハモリ**だけ——旋律・詞・設計は**原曲が決めている**ので、
    作り直す工程がそもそも存在しない（そう言って断る）。
    カバーのコードは LLM が原曲の旋律・伴奏・音声を材料に書いて曲の記録に残したもの（2026-10-02）で、
    「サビのコードが違う」は `remake_from_step('コード', 頼み)`（前のコードを土台に書き直し、編成から先も作り直す）。

    ※**編成・伴奏・ハモリだけで足りるなら `rearrange_song` のほうが安い**
      （こちらは前段の復元と後工程の作り直しを伴う）。
    ※**音符や音量のピンポイント修正には使わない**（`replace_region` / `adjust_song`）。
    ※元の版は消えない。作り直したものが**新しい版**になる。
    """
    sid = ctx.deps.song_id
    song_dir = SONGS / sid
    if not sid or not song_dir.exists():
        return f"曲が未ロード。先に load_existing_song を。（song_id={sid!r}）"
    from song_remake import describe_steps, rebuild, resolve_step, step_problem
    name = resolve_step(step)
    if not name:                                           # 工程名は正式な名前だけ（別名は読まない・2026-10-02 判断6）
        _trace(sid, "remake_from_step.out", kind="refused", reason="step-name", given=str(step)[:40])
        _f = form_of(ctx)                                  # 一覧はその曲の形の工程だけ（2026-10-03）
        return f"作り直さなかった: {step_problem(step, form=_f)}\n{describe_steps(form=_f)}"
    _trace(sid, "remake_from_step.in", step=name, request=request[:120])

    # 工程の記録が要る——曲フォルダの楽譜だけでは前段を復元できない。
    # ★記録は**曲フォルダに版ごと**（2026-09-25・段階3）。段階3より前の曲と、途中で止まった曲は
    #   `_wip/<曲名>` にあるので、曲名も引いておく（`session.load_record` が順に探す）。
    title = _title_of(ctx)
    if not title:
        try:
            title = json.loads((song_dir / "score.json").read_text(encoding="utf-8")).get("title", "")
        except Exception:
            title = ""
    # ★**途中で止まった曲は、曲名が `progress.json` にしか無い**（2026-09-25）。
    #   楽譜がまだ無いので上の2つでは取れず、カバー扱いに落ちて断っていた。
    #   `session._update_index` は既にこの順で探している——ここだけ揃っていなかった。
    if not title:
        try:
            title = json.loads((song_dir / song_files.PROGRESS).read_text(encoding="utf-8")).get("title", "")
        except Exception:
            title = ""
    from session import save_record
    from song_edits import apply_edits, notes_redone, reapply_note
    if why := _supersedes_problem(ctx, supersedes, title):   # 台帳に生きていない id は黙って無視しない（2026-10-02・洗い出し 2-2）
        return why
    try:
        rec = _record_of(ctx, title)
    except Exception:
        rec = None
    if not _is_original(rec):
        # ★**カバーには工程の記録が無い**（11工程を通らず、原曲の歌詞とタイミングを使う）。
        #   だが**編成・伴奏・ハモリは自分で決めている**ので、そこはやり直せる。
        #   本人:「**カバーでもできるようにして**」（2026-09-16）。
        #   ★**コードも作り直せる**（2026-10-02・組「カバーのコード」・J6）——カバーのコードは LLM が書いて曲の記録に残す段になった。
        #   それより前の工程（設計・グルーヴ・旋律・詞）は**原曲が決めている**ので、
        #   作り直す対象がそもそも無い——断る理由をそう伝える。
        from song_arrange_cover import STEPS as COVER_STEPS
        from song_arrange_cover import arrange_song as _cover_arrange
        from song_cover_load import load_cover
        if name not in COVER_STEPS:
            _trace(sid, "remake_from_step.out", kind="refused", reason="cover-upstream")
            return (f"この曲はカバーなので、{name}は**原曲が決めている**——作り直す工程が無い。"
                    f"カバーでやり直せるのは **{' / '.join(COVER_STEPS)}** だけ。"
                    f"ほかは 尺なら resize_song、音符や音量なら replace_region / adjust_song、"
                    f"テンポ・移調なら adjust_song が使える。"
                    f"※原曲から離れて作り直したいなら、それは新しいオリジナル曲を作ること。")
        try:
            song = load_cover(song_dir)
            # 区間・調・コード（記録にあれば読む・無ければ LLM が書いて残す）。「コード」から作り直すときは、記録のコードを土台に頼みで書き直す
            made = _cover_ready(ctx, song_dir, song, rec, redo_chords=(name == "コード"),
                                request=request if name == "コード" else "")
            with _hand_scope(ctx.deps.score, name):        # 人が直した範囲を先に知らせる（道具-8）
                song = _cover_arrange(song, request, start=name, verbose=False)
        except StepUnfixed as e:                           # 形式の所見が直らずに止まった（2026-09-30）
            return _stopped_edit(sid, "remake_from_step", e, restart=_cover_step(e.step, name))
        except Exception as e:
            _trace(sid, "remake_from_step.out", kind="error", reason=f"{type(e).__name__}")
            return f"{name}から作り直せなかった（{type(e).__name__}: {e}）。"
        # ★**カバーはいまの曲から作り直すので、作り直しで消える声部の手直しだけを当て直す**（2026-10-01・別件①）。
        #   以前は台帳を全部当て直し、版に入った移調をもう一度かけていた（+2 が +4）
        from song_cover_load import chords_note, key_note
        from song_edits import cover_reapply
        song.requests = list(rec.requests) if rec is not None else []
        song.score, done, _ = apply_edits(song.score, cover_reapply(song.requests), notes_changed=False)
        from manual_import import carry_hand_edits
        _carried: list[str] = []
        song.score, _hand = carry_hand_edits(ctx.deps.score, song.score, name, carried=_carried)
        ctx.deps.score = song.score
        from song_pipeline import compose as _compose
        _compose(sid, song.score)
        _rid = _note_request(ctx, request, name, scope=scope, supersedes=supersedes,
                             history=made)                 # 台帳だけの記録に残す（今回作った区間・コードの記録も）
        _snapshot_version_if_changed(song_dir)
        _update_index(sid)
        _trace(sid, "remake_from_step.out", step=name, kind="cover",
               notes=len(song.score.vocal_notes), reapplied=len(done))
        _after = [s for s in COVER_STEPS[COVER_STEPS.index(name):]]
        return (f"『{song.score.title}』（カバー）を**{name}から作り直した**"
                f"（{' → '.join(_after)}）。旋律と歌詞は原曲のまま。"
                f"\n{key_note(song)}"
                + (f"\n{chords_note(song)}" if chords_note(song) else "")   # コードがどこから来たか（LLM が書いた／記録から・2026-10-02）
                + "\n（コード・伴奏の対旋律・リフと声部は C 基準で書かせ、この調へ移した。調が違っていたらマスターに伝える）\n"
                + (f"前に手で直したこと（{' / '.join(done)}）は当て直した。" if done else "")
                + (f"\n人が手で直したものを引き継いだ: {' / '.join(_carried)}" if _carried else "")
                + (f"\n★{' / '.join(_hand)}" if _hand else "") + _ledger_note(_rid))
    # ★**その形で走らない工程は作り直さない**（2026-10-03・§3）。インストに詞の工程は無く、アカペラに伴奏の工程は無い——
    #   形そのものを変えるのは change_form（§4）。断りの文は作り直しの側（`song_remake.absent_step_problem`）の1つ
    from song_remake import absent_step_problem
    if why := absent_step_problem(name, getattr(rec, "form", "song") or "song"):
        _trace(sid, "remake_from_step.out", kind="refused", reason=f"form-{rec.form}")
        return why
    _prev_score = ctx.deps.score or (getattr(rec, "score", None) if rec is not None else None)
    import song_llm as _llm
    _llm.KEEP_REJECTED.clear()
    try:
        song = rec
        # ★**人が手で直した範囲を、作り直す工程に先に知らせる**（2026-09-27・目録の道具-8）。
        #   作り直したあと機械がその上に重ねるので、知らないと工程は手直しとぶつかる形で作る
        with _hand_scope(_prev_score, name):
            new = rebuild(song, name, request, songs_dir=SONGS, verbose=False, scope=scope,
                          supersedes=_split_ids(supersedes))
    except ValueError as e:                                # 工程名が不正・テーマ起点
        _trace(sid, "remake_from_step.out", kind="refused", reason=str(e)[:80])
        return f"{e}"
    except StepUnfixed as e:
        # ★**形式の所見が直らずに止まった**（2026-09-30）。作り直しの途中まで作った工程は作業場（`_wip/_remaking`）にしか無く、
        #   曲の記録には入らない——続きは**作り直しを始めた工程**から、同じ頼みでもう一度（止まった工程からは続けられない）
        _how = ("" if _prev_score is not None else          # 途中で止まった曲の続き（楽譜がまだ無い）
                f"**まだ曲はできていない**（前に止まった所までの保存はそのまま）。続けるときは "
                f"**remake_from_step(step='{name}', request=同じ頼み)** でもう一度（上の所見が直るように、という一言を足してよい）。")
        return _stopped_edit(sid, "remake_from_step", e, restart=name, how=_how)
    except Exception as e:
        _trace(sid, "remake_from_step.out", kind="error", reason=f"{type(e).__name__}")
        return (f"{name}から作り直せなかった（{type(e).__name__}: {e}）。"
                f"アレンジだけなら rearrange_song が使える。")

    # ★**手で直したことを当て直す**（2026-09-25・段階3）。作り直しは工程の記録から楽譜を組み立て直すので、
    #   楽譜だけに当てた移調・速さ・声部・音符はここで当てないと消える（キーを上げてから詞を直すとキーが戻る）。
    #   旋律か詞が変わったら、音符に乗った手直しは当てずに「当て直し待ち」にする（§10-2）。
    #   ★**設計を作り直したら、速さ・移調も当てずに「当て直し待ち」**（2026-09-27・目録の道具-7）——
    #     設計の LLM が決めた新しい BPM を黙って戻さない／新しい調の上に前の差を足さない。
    #   ★**歌い方・歌の崩しを作り直したら、置き直した音符にも新しい歌い方・崩しを使う**（2026-09-27・監査）。
    #     以前は台帳の前の値ごと当て直し、書き直した区間だけ前の歌い方が残っていた。どちらを使ったかは記録に残す。
    from song_edits import design_redone, expression_redone
    changed = notes_redone(name)
    _tu, _ar = expression_redone(name)
    new.score, done, wait = apply_edits(new.score, new.requests, notes_changed=changed,
                                        design_changed=design_redone(name),
                                        tuning_changed=_tu, artic_changed=_ar)
    from song_fix import record_change
    for _d in done:
        if _d.startswith("音符"):
            record_change(new, "expression", f"台帳の音符を当て直した: {_d}",
                          "作り直した楽譜に、前に置き直した音符を重ねた")
    # 人が V6・MIDI で手で直したもの（楽譜の手の直しの欄）は、前の版の楽譜から引き継ぐ。下が変わったら引き継がず報せる
    from manual_import import carry_hand_edits
    _carried: list[str] = []
    new.score, _hand = carry_hand_edits(_prev_score, new.score, name, carried=_carried)
    # ★**当て直したら、点検待ちの束へ大として積む**（2026-10-04・案C §3.7）。作り直しの中の最終の点検役（`rebuild`）は
    #   当て直す前の楽譜を見ている（移調 +3・ハモリを足したあとで詞を作り直すと、移調もハモリも無い曲を通していた）
    _gate = note_edit(ctx, "当て直し", f"{name}から作り直したあと当て直した: " + " / ".join([*done, *_carried])) \
        if (done or _carried) else ""
    ctx.deps.score = new.score
    from song_pipeline import compose as _compose
    _compose(sid, new.score)                               # 楽譜・伴奏・vpr・仮歌・mix を書き直す
    save_record(song_dir, new)                             # 工程の記録を作業台へ（版は楽譜と一緒に切る）
    ctx.deps.record_version = None
    _snapshot_version_if_changed(song_dir)
    _update_index(sid)
    # ★**曲名が変わったら、曲フォルダの名前も付け替える**（2026-10-03・本人了承）。曲名を読む工程は『曲名』だけなので、
    #   ほかの工程から作り直して曲名の工程まで巻き込まれた場合も同じ。台帳の行・song_id・同じ会話の絶対パスも一緒に直す。
    #   開いていて変えられなければ元の名前のまま続ける（次の書き出しと会話の終わりにもう一度試す）。
    #   ★このあとは新しい名前で書く——古い名前で `_trace` すると古いフォルダが作り直される
    retitle_song_folder(ctx.deps, title, new.score.title, SONGS)
    sid = ctx.deps.song_id
    song_dir = SONGS / sid
    _new_form = getattr(new.score, "form", "song") or "song"
    after = [s for s in STEP_ORDER_FOR_MSG(name, _new_form)]
    _trace(sid, "remake_from_step.out", step=name, notes=len(new.score.vocal_notes),
           reapplied=len(done), waiting=len(wait))
    from song_requests import listing as _req_listing
    _asks = _req_listing(new.requests)
    _wait = reapply_note(wait, new.score)
    _kept = [h for h in new.history if h.request == "keep"]
    _rejected = [f"{s}（{why[:60]}）" for s, why, _e in _llm.KEEP_REJECTED]
    return (f"『{new.score.title}』を**{name}から作り直した**"
            + (f"\n変える必要が無いと判断して**前回のまま**にした工程: "
               + " / ".join(f"{h.step}（{h.reason[len('前回のまま: '):][:60]}）" for h in _kept) if _kept else "")
            + (f"（{' → '.join(after)} も一緒にやり直した）" if after else "（ほかの工程はそのまま）")
            + (f"。歌 {len(new.score.vocal_notes)}音。" if _new_form != "instrumental" else "。")
            + f"**新しい版として保存**したので、前の版は残っている。"
            + (f"export_vocal で実歌声にして確認できる。" if _new_form == "song" else finish_hint(_new_form))
            + (f"\n★「前回のまま」を選んだが、前回の出力が今の形に合わず**作り直させた**工程: "
               + " / ".join(_rejected) if _rejected else "")
            + (f"\n前に手で直したことを当て直した: {' / '.join(done)}" if done else "")
            + (f"\n{_wait}" if _wait else "")
            + (f"\n人が手で直したものを引き継いだ（作り直した工程にも「この範囲は人が直している」と伝えた）: "
               + " / ".join(_carried) if _carried else "")
            + (f"\n★{' / '.join(_hand)}——同じ直しが要るかはマスターに聞く" if _hand else "")
            + (f"\n{_gate}" if _gate else "")
            + (f"\n【この曲で生きている頼みごと】（次に作り直しても守られる）\n{_asks}" if _asks else ""))


def _song_title(ctx: RunContext[MusicSession], song_dir: Path) -> str:
    """曲名（開いている楽譜 → 曲フォルダの楽譜 → 途中で止めた印 の順。`session._update_index` と同じ順）。"""
    title = _title_of(ctx)
    for name in ("score.json", song_files.PROGRESS):
        if title:
            break
        try:
            title = json.loads((song_dir / name).read_text(encoding="utf-8")).get("title", "")
        except Exception:
            title = ""
    return title


def _has_score(song_dir: Path) -> bool:
    """楽譜がある曲か（途中で止めた曲には楽譜がまだ無い・`load_existing_song` と同じ見分け方）。"""
    return (song_dir / song_files.SCORE).exists() or any(
        (got := song_files.split_version(p.name)) and got[0] == song_files.SCORE for p in song_dir.iterdir())


def _step_json(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)


def _earliest_step(steps, new_form: str) -> str:
    """形を変えたときの「手の直しの引き継ぎの基準」の工程＝**足した・外した・作り直した工程のうち作る順で最初のもの**（無ければ空）。
    インストにするなら歌が無くなるので、旋律より後にはしない（歌の手の直しを引き継がない＝旋律から作り直したのと同じ扱い）。"""
    from song_gate import STEP_ORDER
    got = [s for s in steps if s in STEP_ORDER] + (["旋律"] if new_form == "instrumental" else [])
    return min(got, key=STEP_ORDER.index) if got else ""


def _form_basis(sid: str, rec, old_form: str, new_form: str) -> str:
    """作り直しの前に、`change_form` が作り直す工程を引いて基準の工程を決める（`_hand_scope` に渡す）。作り直しの側
    （`song_remake.rebuild_form`）と同じ答え（`song_gate.form_change`＋新しい形の形式の検査 `song_form_checks.form_findings`）。
    引けなければ、形が変われば必ず作り直す編成（`song_gate.form_change` の決まり）を基準にする。"""
    try:
        from song_form_checks import form_findings
        from song_gate import form_change
        fc = form_change(old_form, new_form, check=lambda s: form_findings(rec, s, new_form))
        return _earliest_step([*fc.add, *fc.remove, *fc.remake], new_form) or "編成"
    except Exception as e:                                 # 基準が引けなくても形の変更は止めない——ただし黙らない
        _trace(sid, "change_form.warn", reason=f"basis-{type(e).__name__}: {e}"[:160])
        return _earliest_step(["編成"], new_form)


@coordinator.tool
@_locked
def change_form(ctx: RunContext[MusicSession], form: str, request: str = "") -> str:
    """**曲の形を変える**（歌もの・アカペラ・インストのあいだで。2026-10-03）。

    「このアカペラに伴奏を付けて」「インストに歌を乗せて」「歌を抜いて伴奏だけの曲にして」
    「伴奏を外して声だけにして」のように、**歌や伴奏を足す・外す**依頼に使う。
    `form` … 新しい形。**song**（歌もの）／**acappella**（アカペラ）／**instrumental**（インスト）を**この綴りのまま**
    （日本語・別名・大文字は読まない）。
    `request` … マスターの頼み（例「サビだけ厚いコーラスにして」）。台帳に残り、作り直す工程に届く。特に無ければ空。

    ★**どの工程を作り直すかは機械が決める**（形の差と、新しい形の決まりに合わない工程。工程名は書かない）。
      足す工程は一から書き、巻き込まれた工程は「前回のまま」を選べる（新しい形の決まりに合わなければ作り直す）。
      ふつうは**旋律は変えない**——旋律も変えたければ、そのあと remake_from_step('旋律', 頼み)。
      ★例外: 前の版の工程が新しい形の決まりに合わないときは、その工程から作り直す。たとえば**終わり方が歌（vocal）の曲を
      歌もの以外にすると、設計から作り直す**ので、旋律（と詞）も変わる。どの工程を作り直したかは返事に出る。
    ★**元の版は消えない**。作り直したものが**新しい版**になる（外した工程の出力も前の版に残る）。
    ★手で直したこと（移調・速さ・声部・置き直した音符）は作り直しのあとに当て直す。新しい形に届け先が無い頼み
      （インストにした曲の声部・音符など）は当てずに返事に出す（台帳には残る）。
    ※**いまの曲はそのままで、歌の無い音（伴奏だけ）が欲しいだけ**なら使わない（形は変えない）。
    断る（何も変えない）: いまと同じ形・カバー・工程の記録が無い曲・途中で止まった曲（先に remake_from_step で続きを作る）。
    """
    sid = ctx.deps.song_id
    song_dir = SONGS / sid
    if not sid or not song_dir.exists():
        return f"曲が未ロード。先に load_existing_song を。（song_id={sid!r}）"
    try:                                                   # 形の綴りは `song_files.as_form` の1つ（空＝song・読めない綴りは断る）
        new_form = song_files.as_form(form)
    except ValueError as e:
        _trace(sid, "change_form.out", kind="refused", reason="form-spelling", given=str(form)[:40])
        return f"形は変えなかった: {e}"
    _trace(sid, "change_form.in", form=new_form, request=request[:120])
    title = _song_title(ctx, song_dir)
    try:
        rec = _record_of(ctx, title)
    except Exception:
        rec = None
    if rec is None:
        _trace(sid, "change_form.out", kind="refused", reason="no-record")
        return ("形は変えなかった: この曲には工程の記録（record）が無く、どの工程から作り直すか決められない"
                "（記録の無い古い曲）。その旨をマスターに伝える。")
    if not _is_original(rec):
        _trace(sid, "change_form.out", kind="refused", reason="cover")
        return ("形は変えなかった: この曲はカバー。カバーは原曲の歌と伴奏の両方を前提に組んでいるので、形（歌もの以外）にはできない。"
                "その旨をマスターに伝える。")
    old_form = getattr(rec, "form", "song") or "song"
    if new_form == old_form:
        _trace(sid, "change_form.out", kind="refused", reason="same-form")
        return (f"形は変えなかった: この曲はもう{song_files.FORM_LABEL[new_form]}。"
                "中身を直したいなら remake_from_step（工程から）・rearrange_song（響き）などを使う。")
    _prev_score = ctx.deps.score or getattr(rec, "score", None)
    if _prev_score is None or not _has_score(song_dir):
        _trace(sid, "change_form.out", kind="refused", reason="unfinished")
        return ("形は変えなかった: この曲は途中で止まっていて、まだ楽譜が無い。先に load_existing_song の案内どおり "
                "remake_from_step で続きを作り、できあがってから形を変える。")
    from song_context import step_output_key
    from song_edits import ASSEMBLY_VOICE, NOTE_STEPS, NOTES, apply_edits, reapply_note
    from song_gate import FORM_STEPS
    from song_model import StepRecord
    before = {s: _step_json((rec.to_json().get("steps") or {}).get(step_output_key(s))) for s in FORM_STEPS[old_form]}
    import song_llm as _llm
    _llm.KEEP_REJECTED.clear()
    try:
        from song_remake import rebuild_form                # 中身は作り直しと同じ（保存から前段を $0 で戻す・作業場を隔離・新しい版）
        # 人が手で直した範囲を先に知らせる（道具-8）。★**基準は作り直す工程から決める**（2026-10-03・再レビュー）。以前は「編成」固定で、
        # 終わり方が歌の曲を歌もの以外にして設計から作り直したときも、歌の手の直しを引き継ぐと工程に伝えていた。
        # 作り直す工程は作り直しの側と同じ答え（`song_gate.form_change`＋新しい形の形式の検査）で前もって引く
        _basis = _form_basis(sid, rec, old_form, new_form)
        with _hand_scope(_prev_score, _basis):
            new = rebuild_form(rec, new_form, request, songs_dir=SONGS, verbose=False)
    except ValueError as e:                                # 形が読めない・同じ形（入口で断っているので来ないはず）
        _trace(sid, "change_form.out", kind="refused", reason=str(e)[:80])
        return f"形は変えなかった: {e}"
    except StepUnfixed as e:
        return _stopped_edit(sid, "change_form", e, how=(
            "**形は変えていない**（元の版はそのまま残っている）。\n"
            f"続けるときは **change_form(form='{new_form}', request=同じ頼み)** でもう一度"
            "（上の所見が直るように、という一言を頼みに足してよい）。"))
    except Exception as e:
        _trace(sid, "change_form.out", kind="error", reason=f"{type(e).__name__}")
        return f"形を変えられなかった（{type(e).__name__}: {e}）。元の版はそのまま残っている。"

    # ---- 何が起きたかは、前後の記録の工程の出力を突き合わせて書く（事実から・推測しない）----
    after_steps = new.to_json().get("steps") or {}
    old_steps, new_steps = FORM_STEPS[old_form], FORM_STEPS[new_form]
    added = [s for s in new_steps if s not in old_steps]
    removed = [s for s in old_steps if s not in new_steps]
    redone = [s for s in new_steps if s in old_steps and before.get(s) != _step_json(after_steps.get(step_output_key(s)))]
    kept = [s for s in new_steps if s in old_steps and s not in redone]
    if not any(h.step == "form" for h in new.history):   # 記録に「形」の行が無ければ積む（§4-5。作り直しの側が積んでいれば重ねない）
        new.record(StepRecord(step="form", request=new_form, reason=(
            f"{song_files.FORM_LABEL[old_form]}→{song_files.FORM_LABEL[new_form]}。"
            f"足した工程: {'・'.join(added) or 'なし'}／外した工程: {'・'.join(removed) or 'なし'}／"
            f"作り直した工程: {'・'.join(redone) or 'なし'}／前の版のままの工程: {'・'.join(kept) or 'なし'}")))

    # ---- 手で直したことの当て直し（§4-3）。新しい形に届け先が無い頼みは当てずに出す（台帳には残す＝形を戻せばまた届く）----
    touched = set(added) | set(redone) | set(removed)
    _no_vocal = new_form == "instrumental"
    unreachable = [r for r in new.requests if r.status in ("active", "reapply")
                   and (r.target in removed or (_no_vocal and r.target in (ASSEMBLY_VOICE, NOTES)))]
    new.score, done, wait = apply_edits(new.score, [r for r in new.requests if r not in unreachable],
                                        notes_changed=bool(set(NOTE_STEPS) & touched),
                                        design_changed="設計" in touched,
                                        tuning_changed="歌い方" in touched, artic_changed="歌の崩し" in touched)
    from manual_import import carry_hand_edits
    _carried: list[str] = []
    # 引き継ぎの基準は**実際に足した・外した・作り直した工程**のうち最初のもの（「前回のまま」を選んで変わらなかった工程は数えない）
    _basis = _earliest_step(added + removed + redone, new_form) or _basis
    new.score, _hand = carry_hand_edits(_prev_score, new.score, _basis, carried=_carried)
    # ★当てたら点検待ちの束へ大として積む（2026-10-04・案C §3.7。作り直しの中の最終の点検役は当て直す前の楽譜を見ている）
    _gate = note_edit(ctx, "当て直し", f"形を変えたあと当て直した: " + " / ".join([*done, *_carried])) \
        if (done or _carried) else ""
    ctx.deps.score = new.score
    from song_pipeline import compose as _compose
    _compose(sid, new.score)                               # 楽譜と、その形の持ち物を書き直す（前の形のファイルは書き出しが消す・§4-4）
    from session import save_record
    save_record(song_dir, new)
    ctx.deps.record_version = None
    _snapshot_version_if_changed(song_dir)
    _update_index(sid)
    _trace(sid, "change_form.out", old=old_form, new=new_form, added=len(added), removed=len(removed),
           redone=len(redone), kept=len(kept), reapplied=len(done), waiting=len(wait), unreachable=len(unreachable))
    from song_requests import listing as _req_listing
    _asks = _req_listing(new.requests)
    _kept = [h for h in new.history if h.request == "keep"]
    _rejected = [f"{s}（{why[:60]}）" for s, why, _e in _llm.KEEP_REJECTED]
    # ★**書き出しのあとの版を読む**（2026-10-03・再レビュー）。完成のあと別の形で書き出すと、書き出し（`compose` →
    #   `session.open_version_for_form`）が版を進める——確定済みの版（`_current_version`）は前の版の番号になる。
    #   書き出し・完成品が使う版（`export_vocal` と同じ決め方＝作業中の版、無ければ確定済みの版）を出す
    _cur = _working_version(song_dir) or _current_version(song_dir)
    return (f"『{new.score.title}』を**{song_files.FORM_LABEL[old_form]}から{song_files.FORM_LABEL[new_form]}にした**"
            f"（**新しい版として保存**したので、前の版は残っている）。"
            f"\n足した工程: {' / '.join(added) or 'なし'}"
            f"\n外した工程: {' / '.join(removed) or 'なし'}（出力は前の版の記録に残っている）"
            f"\n作り直した工程: {' / '.join(redone) or 'なし'}"
            f"\n前の版のままの工程: {' / '.join(kept) or 'なし'}"
            + (f"\n変える必要が無いと判断して**前回のまま**にした工程: "
               + " / ".join(f"{h.step}（{h.reason[len('前回のまま: '):][:60]}）" for h in _kept) if _kept else "")
            + (f"\n★「前回のまま」を選んだが、前回の出力が新しい形に合わず**作り直させた**工程: "
               + " / ".join(_rejected) if _rejected else "")
            + (f"\n前に手で直したことを当て直した: {' / '.join(done)}" if done else "")
            + (f"\n{reapply_note(wait, new.score)}" if wait else "")
            + (f"\n★**新しい形に届け先が無い頼み**（当てていない・台帳には残っている＝形を戻せばまた届く）: "
               + " / ".join(f"{r.id}（{r.target}・{r.words[:40]}）" for r in unreachable)
               + "——最終報告でマスターに伝える" if unreachable else "")
            + (f"\n人が手で直したものを引き継いだ: {' / '.join(_carried)}" if _carried else "")
            + (f"\n★{' / '.join(_hand)}——同じ直しが要るかはマスターに聞く" if _hand else "")
            + f"\n{finish_hint(new_form)}" + (f"（いまの版は v{_cur}）" if _cur else "")
            + (f"\n{_gate}" if _gate else "")
            + (f"\n【この曲で生きている頼みごと】（次に作り直しても守られる）\n{_asks}" if _asks else ""))


def STEP_ORDER_FOR_MSG(step: str, form: str = "song") -> list[str]:
    """`step` を作り直したときに**一緒にやり直した工程**（報告文のため）。

    ★依存の木を辿る（2026-09-25）。列の後ろ全部を書くと、実際にはやり直していない工程まで
      「やり直した」と報告してしまう。
    ★**その形で走る工程だけ**（2026-10-03・§3）。インストの旋律を作り直しても、詞の工程は無い
    """
    from song_remake import redone_after
    return redone_after(step, form=form)


@coordinator.tool
@_locked
def resize_song(ctx: RunContext[MusicSession], request: str) -> str:
    """**曲の尺を変える**（2番を足す／TVサイズに縮める／構成を組み替える）。

    「フルコーラスにして」「90秒のTVサイズに」「2番を足して」のような、
    **構成そのものが変わる**依頼に使う。他の編集と性質が違う——区間の並びが変わり、
    **音符の位置が全部ずれる**ため。

    旋律はコピーして並べ直す（機械）。**新しく歌う区間（2番など）の詞だけ書き直す**
    ——旋律は同じなので、**元の区間と同じ音符数**で書く（新しい曲と同じ譜割りの記法。数が合わなければ詞の検査が差分で直させ、
    直らなければ尺は変えずに止まる・2026-09-30）。

    ★**カバーの伴奏（伴奏 MIDI）も、音符と同じ区間の対応で並べ替える**（2026-09-27）。伴奏が音声のまま
      （伴奏 MIDI が無い）の曲は並べ替えられないので、尺を変えずに断る。
    """
    sid = ctx.deps.song_id
    song_dir = SONGS / sid
    if not sid or not song_dir.exists():
        return f"曲が未ロード。先に load_existing_song を。（song_id={sid!r}）"
    from song_resize import form_refusal as _resize_refusal
    if why := _resize_refusal(form_of(ctx)):              # アカペラ・インストは費用をかける前に断る（2026-10-03・§7-2）
        _trace(sid, "resize_song.refused", reason=f"form-{form_of(ctx)}")
        return why
    _acc_mid = None
    _made: list = []
    try:
        _rec0 = _record_of(ctx)
    except Exception:
        _rec0 = None
    # 楽器で重ねた主旋律（楽譜の `lead`）は並べ直しの対象に入っていない——古い構成のまま残るので、費用をかける前に断る（F10②）
    if why := lead_refusal(ctx, "resize_song", ctx.deps.score or getattr(_rec0, "score", None)):
        return why
    _trace(sid, "resize_song.in", request=request[:120])
    _orig = _is_original(_rec0)
    try:
        from song_cover_load import add_machine_chords, add_sections, load_cover
        from song_resize import apply_resize, plan_resize, write_new_lyrics
        song = load_cover(song_dir)                        # 曲フォルダ → Song
        # ★**伴奏が楽譜に無い曲（カバー）は、伴奏 MIDI を並べ替える**（2026-09-27・監査）。以前は楽譜から
        #   鳴らし直していたので、**伴奏が空で上書きされていた**（楽譜に伴奏のパートが無い）。
        #   伴奏 MIDI も無く伴奏が音声だけなら、並べ替えられないので**費用をかける前に断る**（黙って消さない）。
        if song.score.arrangement.is_empty():               # 伴奏が楽譜に無い（パートが0本・F1）
            _m = (song.source or {}).get("accompaniment_mid")
            _acc_mid = (Path(song.source["folder"]) / _m) if _m else None
            if _acc_mid is None and (song_dir / "accompaniment.wav").exists():
                _trace(sid, "resize_song.out", kind="refused", reason="accomp-audio-only")
                return ("尺は変えなかった: この曲の伴奏は楽譜にも伴奏 MIDI にも無く、音声（accompaniment.wav）だけ。"
                        "音声のままでは区間ごとに並べ替えられないので、尺を変えると**伴奏が消えるかずれる**。"
                        "伴奏ごと尺を変えたいなら、先に rearrange_song で伴奏を楽譜から作り直す（原曲の伴奏ではなくなる）。"
                        "その旨をマスターに伝えて、どうするか聞く。")
        _guessed = ""
        if _orig:
            # ★**オリジナル曲は工程の記録の区間・調・コードをそのまま読む**（2026-10-03 本人決定）。以前は区間を LLM に起こし直し、
            #   コードを伴奏 MIDI から推定し直していた。記録が使えないとき（古い曲・前に尺を変えた曲）だけ推定へ戻り、そう返事と記録に出す
            from song_record_frame import from_record
            _guessed = from_record(song, _rec0)
            if _guessed:
                add_sections(song, verbose=False)
                add_machine_chords(song, verbose=False)
                from song_fix import record_change
                record_change(song, "chords", "区間を旋律から起こし直し、コードをいまの伴奏 MIDI から機械で推定した（コードの工程が書いたコードではない）",
                              f"工程の記録が使えない: {_guessed}")
        else:                                              # カバー: 区間・調・コード（記録にあれば読む・無ければ LLM が書いて残す・2026-10-02）
            # ★今回作った区間・コードの工程の記録は台帳へ（rearrange_song・remake_from_step と同じ。以前は捨てていた・2026-10-03）
            _made = _cover_ready(ctx, song_dir, song, _rec0)
        plan = plan_resize(song, request)                  # どう変えるか（LLM）
        # 並べ直す（機械）。オリジナル曲は工程の記録を渡す——旋律の行を持ち回り、2番の詞のフレーズを旋律の行で切る
        # （渡さないと休符で割ったフレーズになり、2番の詞が旧い休符割りのまま・2026-10-03 組み立ての組）
        warns = apply_resize(song, plan, verbose=False, record=_rec0 if _orig else None)
        rec = write_new_lyrics(song, verbose=False)        # 新しい区間の詞だけ書く
        if not _orig:                                      # 並べ替えた区間とコードを曲の記録へ（次の編集が前の構成を読まないように）
            from song_cover_chords import resized
            resized(song)
            _cover_save(ctx, song_dir, song, _rec0)
    except StepUnfixed as e:                               # 形式の所見が直らずに止まった（2026-09-30・止まる工程は下の `_resize_how`）
        return _stopped_edit(sid, "resize_song", e, how=_resize_how(e.step))
    except Exception as e:
        _trace(sid, "resize_song.out", kind="error", reason=f"{type(e).__name__}")
        return (f"尺を変えられなかった（{type(e).__name__}: {e}）。"
                f"区間が取れない曲かもしれない。部分的な入れ替えなら replace_region を使う。")
    ctx.deps.score = song.score
    _acc_msgs: list[str] = []
    if _acc_mid is not None:
        # カバー: 伴奏 MIDI を音符と同じ対応表で並べ替えて鳴らし直す（楽譜から作らない＝伴奏を空にしない）
        from accomp_edit import load_accomp, save_accomp
        from align_cover import render_vocal_and_mix
        from sf2_render import render_midi
        from song_design import bar_ticks as _bar_ticks
        from song_fix import record_change
        from song_resize import rearrange_accomp
        tracks, _acc_msgs = rearrange_accomp(load_accomp(_acc_mid), song.score.tempo,
                                             _bar_ticks(song.score.time_signature or "4/4"),
                                             getattr(song, "_resize_placed", []))
        save_accomp(tracks, song.score.tempo, song_dir / "accompaniment.mid")
        render_midi(song_dir / "accompaniment.mid", song_dir / "accompaniment.wav")
        render_vocal_and_mix(song.score, song_dir)         # 楽譜・vpr（新しい伴奏を同梱）・仮歌
        for m in _acc_msgs:
            record_change(song, "arrangement", m, "尺の変更でカバーの伴奏 MIDI を区間ごとに並べ替えた")
    else:
        from song_pipeline import compose as _compose
        _compose(sid, song.score)
    _rid = "-"                                             # オリジナル曲は台帳に書かない（尺の変更は工程の記録に入らない・下の返事）
    _cover_lines = ""
    if not _orig:
        # ★**カバーは頼みと、今回作った区間・調・コードの記録を台帳だけの記録に残す**（2026-10-03・再レビュー）。以前は戻り値を捨て、
        #   初めて作ったコードが返事にも台帳にも出なかった
        _rid = _note_request(ctx, request, "設計", history=_made)
        if _made:
            from song_cover_load import chords_note, key_note
            _cover_lines = "".join(f"\n{x}" for x in (key_note(song), chords_note(song)) if x)
    _snapshot_version_if_changed(song_dir)
    _update_index(sid)
    n = len(song.score.vocal_notes)
    _trace(sid, "resize_song.out", notes=n, warns=len(warns))
    # ★**機械が捨てた・埋めたことを全部返す**（2026-09-27・目録の道具-6）。以前は★注意の先頭3件だけで、
    #   ハモリを全部外した・前奏が消えた・新しい区間を4小節 C で置いた・詞を「ー」で埋めた、が見えなかった。
    #   ※詞の「ー」埋めは 2026-09-29 に外した（数のずれは詞の検査が直させ、残れば鳴らさない）
    #   **オリジナル曲は工程の記録に反映していない**——あとで remake_from_step すると元の尺に戻る。
    _chords = [r.reason for r in song.history if r.step == "chords_machine"]
    _omit = list(getattr(song, "_resize_omitted", []) or [])
    _all = _chords + list(warns) + (list(rec.warnings) if rec else []) + _omit
    # ★**伴奏が新しい構成と食い違う所**（2026-09-27・監査）。伴奏のパートは前の構成の小節番号で鳴る所を
    #   持つので、並べ替えた構成では別の区間で鳴る・鳴らない。機械は合わせない（`song_resize.arrangement_mismatch`）
    _arr = list(getattr(song, "_resize_arrangement", []) or [])
    if _all:
        _append_log(sid, "[resize_song] 機械が並べ直したときに起きたこと:\n" + "\n".join(f"  {w}" for w in _all))
    if _arr:
        _append_log(sid, "[resize_song] 伴奏が新しい構成と食い違う所:\n" + "\n".join(f"  {w}" for w in _arr))
    if _acc_mid is not None:
        _append_log(sid, "[resize_song] 伴奏 MIDI を並べ替えた:\n" + "\n".join(f"  {w}" for w in _acc_msgs or ["（捨てた・切った音なし）"]))
    return (f"尺を変えた: {plan.reason[:120]}／{n}音"
            + (f"／新しい詞: {rec.reason[:60]}" if rec else "")
            + (f"\n【機械が並べ直したときに捨てた・埋めたこと（{len(_all)}件）】\n"
               + "\n".join(f"・{w}" for w in _all) if _all else "")
            + ("\n伴奏（原曲の伴奏 MIDI）も、歌と同じ区間の対応で並べ替えて鳴らし直した"
               + ("。そのとき:\n" + "\n".join(f"・{w}" for w in _acc_msgs) if _acc_msgs else "（捨てた・切った音は無い）。")
               if _acc_mid is not None else "")
            + (f"\n【★伴奏が新しい構成と食い違う所（{len(_arr)}件・機械は合わせていない）】\n"
               + "\n".join(f"・{w}" for w in _arr)
               + "\n伴奏は前の構成の小節番号のまま鳴っている。**新しい構成に合った伴奏は作られていない**ので、"
                 "聴く前にマスターへ伝え、伴奏をどうするか（作り直すか・このままか）を聞く。" if _arr else "")
            + ("\n★**この曲はオリジナル曲だが、尺の変更は工程の記録（設計・コード・詞…）に入っていない**。"
               + ("区間・調・コードは工程の記録（設計とコードの工程が書いたもの）から読んだ。" if not _guessed else
                  f"★工程の記録が使えなかった（{_guessed}）ので、区間は旋律から起こし直し、コードはいまの伴奏 MIDI から機械で推定した"
                  "（コードの工程が書いたコードではない）。")
               + "あとで remake_from_step で作り直すと**元の尺に戻る**。"
               "作り直しでも尺を保ちたいなら、remake_from_step('設計', 同じ頼み) で設計から作り直す"
               "（旋律・詞も作り直しになる）。どちらにするかはマスターに聞く。" if _orig else "")
            + _cover_lines
            + "\nexport_vocal で実歌声にして確認できる。" + _ledger_note(_rid))
