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

"""伴奏の観察・区間×トラック編集（アレンジ用）。決定論層は engine/accomp_edit.py。"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent, RunContext

from align_reference import align_score_lyrics
from align_reference import align_to_reference_jp as _align_jp
from align_reference import check_timing, fix_timing, load_reference
from align_cover import (accompaniment_for_section, align_section, lyrics_for_section,
                         render_vocal_and_mix, save_cut_source, section_from_audio, section_from_midi,
                         write_accompaniment)
from machine_eval import similarity
from song_llm import PITCH, generate_added_voice, pick_model, pitch_name_to_midi
from song_checks import JsonFloat, JsonInt  # LLM が書く数と真偽の欄の型・2026-10-02 判断5
from song_checks import strip_ascii_ws  # 半角の記法の前後の空白（§0-6・2026-10-03）
from timbre_map import DRUM_NAMES, drum_name_problem, drum_note   # 打楽器は伴奏のドラムの行と同じ名前（2026-10-01）＝GM の正式な名前（10-02）
from accomp_edit import drum_shown                     # 打楽器の番号 → 見せる名前（伴奏の道具の返事）
from reference_to_wav import SR
from song_pipeline import compose
from song_schema import SongScore
from reference_fetch import fetch_audio_reference as _fetch_audio_ref
from reference_fetch import fetch_reference as _fetch_ref

import song_files                                      # 曲フォルダのファイルの名前の決まり（伴奏 wav・完成品）
from session import (SONGS, MusicSession, with_scope, _append_log, _trace, _traced, _locked, _save_section,
                     _save_ref_index, _save_accomp_bounds, _save_section_bounds, _write_fetch_log,
                     _clarify_msg, _build_log_digest, _slug, _song_status, _current_version,
                     begin_session, end_session, _snapshot_version_if_changed, _read_index, _update_index,
                     _retire_scoreless_session)
from coordinator import coordinator


# ★`AccompNote` の docstring は道具の引数の説明としてコーディネーターに見える——**道具の説明だけを書き、経緯はここ（コメント）に置く**
#   （2026-10-03・再レビュー。以前は docstring に `start_beat`・MIDI 番号の `key`・`kick` の経緯があり、使えない書き方を見せていた）。
#   ・音の高さは国際式の音名（`note`）、打楽器は名前（`drum`）（2026-10-01）。以前は1つの欄 `key` に MIDI 番号で両方を書かせていた。
#     曲の表・伴奏の `pitches`・ハモリは国際式なので、高さはそれにそろえた。打楽器は高さではない（どの太鼓か）ので欄を分けた
#   ・打楽器は伴奏のドラムの行と同じ名前（2026-10-01 本人決定）。いったん GM の打楽器番号にしたが、伴奏の工程は名前で書くので
#     名前にそろえた。2026-10-02 から名前は GM の正式な名前だけ（以前の略称と別名はやめた）。読み手は伴奏と同じ `timbre_map.drum_canon`
#   ・位置は曲の表と同じ `小節:拍`（記法の仕様書 §15-2・ず-18）。以前は楽譜の頭からの拍
class AccompNote(BaseModel):
    """replace_accompaniment_region で渡す伴奏の1つの音（歌詞なし）。高さのトラックは `note`（国際式の音名）、
    ドラムのトラックは `drum`（打楽器の名前）のどちらか1つ。位置は `小節:拍`、長さは拍、強さは 1〜127。"""
    note: str | None = Field(default=None, description=(
        "音の高さ（ドラム以外のトラック）。国際式の音名（`C4`＝中央のド・`E2`・`F#3`・`Bb2`。変化記号は `#` と `b`、"
        "オクターブの数字は必ず付ける）。inspect_accompaniment の一覧と同じ書き方。ドラムトラックでは書かない"))
    drum: str | None = Field(default=None, description=(
        "ドラムトラック（ch9）だけ: 打楽器の名前（伴奏のドラムの行と同じ名前＝GM の打楽器の正式な名前を小文字・`_` つなぎにしたもの。"
        "inspect_accompaniment の一覧と同じ書き方。kick・hihat のような略称は書けない。"
        f"{' / '.join(DRUM_NAMES)}）。高さのトラックでは書かない"))
    at: str = Field(description=(
        "開始位置 `小節:拍`（inspect_song の表・inspect_accompaniment の一覧と同じ書き方。小節は楽譜の頭から1始まり、"
        "拍は4分音符＝1.00の0始まり・小数2桁。例 `5:2.50`）"))
    beats: JsonFloat = Field(gt=0, description="長さ（拍・4分音符＝1.0。4分の3等分は `0.33`・`0.67`、8分の3等分は `0.17`・`0.83`）")
    # ★強さは必ず書く（2026-10-02・洗い出し 2-3。以前は省くと 80 で埋めていた）
    vel: JsonInt = Field(ge=1, le=127, description="強さ 1〜127（打点の記号の X＝110・x＝88・.＝62 と同じ物差し）")

    @field_validator("note")
    @classmethod
    def _readable_note(cls, v: str | None) -> str | None:
        if v is not None and (bad := PITCH.problems(v)):
            raise ValueError(bad[0])
        return v

    @field_validator("beats")
    @classmethod
    def _sounding_beats(cls, v: float) -> float:
        # 長さの決まりは声部・replace_region と同じ1つ（§14。tick にして 0 になる長さは書けない）。以前は 10tick 未満を黙って伸ばした
        from harmony import beats_problem
        if why := beats_problem(v):
            raise ValueError(why)
        return v

    @field_validator("drum")
    @classmethod
    def _known_drum(cls, v: str | None) -> str | None:
        if v is not None and (why := drum_name_problem(v)):   # 打楽器名の読み手は伴奏のドラムの行と同じ1つ
            raise ValueError(f"{why}（書ける名前: {' / '.join(DRUM_NAMES)}）")
        return v

    @field_validator("at")
    @classmethod
    def _readable_at(cls, v: str) -> str:
        from song_checks import POSITION                    # 位置の記法は表・声部と同じ1つ（拍の上限は道具の中で）
        if bad := POSITION.problems(v):
            raise ValueError(bad[0])
        return v

    @model_validator(mode="after")
    def _one_of(self) -> "AccompNote":
        if (self.note is None) == (self.drum is None):
            raise ValueError("note（音名）と drum（打楽器名）のどちらか1つだけを書く"
                             "（ドラムトラックは drum、それ以外は note）")
        return self

    @property
    def key(self) -> int:
        """伴奏 MIDI に書く番号（音名なら MIDI の高さ、打楽器なら GM の打楽器番号）。"""
        return int(drum_note(self.drum)) if self.drum is not None else int(pitch_name_to_midi(self.note))

def _accomp_ctx(ctx: RunContext[MusicSession]) -> tuple[Path, float] | str:
    """伴奏編集ツール共通の前提検査 → (mid_path, tempo) か エラーメッセージ。"""
    song_dir = SONGS / ctx.deps.song_id
    mid = song_dir / "accompaniment.mid"
    if ctx.deps.score is None:
        return "曲が未生成。先に load_existing_song（または新規作成の工程）を。"
    if getattr(ctx.deps.score, "form", "song") == "acappella":   # 前の形の伴奏 MIDI が残っていても触らない（形で決める・2026-10-03・§7-2）
        return ("この曲はアカペラなので伴奏は無い（伴奏の観察・書き換え・楽器を外す道具は使えない）。"
                "伴奏を付けたいなら change_form('song', 頼み)。")
    if not mid.exists():
        return ("この曲には伴奏MIDI（accompaniment.mid）が無く、伴奏の譜面編集はできない"
                "（記憶生成オリジナルの合成伴奏など）。伴奏を変えたい場合はマスターに相談を。")
    return mid, float(ctx.deps.score.tempo)


def _vocal_side(ctx: RunContext[MusicSession], song_dir: Path) -> None:
    """伴奏を書き換えたあと、歌の側（楽譜・vpr・仮歌・口の動き）を書き直す。
    ★**インストは vpr・仮歌・口の動きを持たない**（2026-10-03・§4-4・`song_files.FORM_FILES`）ので、楽譜だけを書く。"""
    if getattr(ctx.deps.score, "form", "song") == "instrumental":
        (song_dir / "score.json").write_text(ctx.deps.score.model_dump_json(indent=2), encoding="utf-8")
        return
    st = ctx.deps.section
    render_vocal_and_mix(ctx.deps.score, song_dir, log=(st.get("log") if st else None))


def _after_accomp_edit(ctx: RunContext[MusicSession], song_dir: Path) -> str:
    """伴奏を書き換えたあと、歌の側を書き直し（`_vocal_side`）、返事に添える「完成品まで」の一文を返す（形で分ける）。

    ★**インストは直した伴奏から完成品を作り直す**（2026-10-03・再レビュー）。インストの完成品は伴奏 wav に仕上げを当てた写しで
      （`render_final` → `song_pipeline.compose` → `v6_export.finish_as_master`）、以前はここで楽譜だけを書き、返事は export_vocal
      （インストには無い）を案内していた——直しが完成品に届かなかった。`render_final` は伴奏を楽譜の編成から作り直すので、
      MIDI の直しを消さないよう、ここでは同じ仕上げ（`finish_as_master`）だけを当てる。歌もの・アカペラは `session.next_step`。"""
    from session import next_step
    _vocal_side(ctx, song_dir)
    form = getattr(ctx.deps.score, "form", "song") or "song"
    if form != "instrumental":
        return next_step(form)
    # ★**インストは完成品をここで作り直すので、その前に点検待ちの束を見る**（2026-10-04・案C）。NG でも作り直す（所見を返すだけ）
    from tools.editing import review_edits
    gate = review_edits(ctx, "伴奏の編集")
    # 版の決め方は書き出し（`song_pipeline.compose`）と同じ: 作業中の版に別の形の完成品があれば版を進め、印が無ければ置く
    from session import ensure_working_version, open_version_for_form
    from v6_export import finish_as_master
    ver = open_version_for_form(song_dir, "instrumental") or ensure_working_version(song_dir)
    dst, note = finish_as_master(song_dir / song_files.ACCOMP_WAV, song_dir, "instrumental", ver, copy=True)
    _trace(ctx.deps.song_id, "accomp_edit.master", master=dst.name)
    return (f"インストなので、直した伴奏から完成品 {dst.name} を作り直した{note}＝このまま play_song で流せる"
            "（歌声の書き出しは無い）。" + gate)


#: トラックの番号の書き方（`inspect_accompaniment` の `[番号]`＝0始まり・半角の数字・先頭に 0 を付けない）
_TRACK_NO = re.compile(r"0|[1-9][0-9]*")


def _tracks_named(tracks: list, track: str) -> tuple[list[int], str]:
    """道具の `track`（番号か名前）→ (当たったトラックの番号, 所見)。所見があれば何も当たっていない。

    ★**番号は `inspect_accompaniment` の `[番号]` だけを読む**（2026-10-02・洗い出し H13・2-3）。以前は `str.isdigit` で、
      全角の「１」も番号として読み、範囲外の番号（10本の曲で "12"）は名前として扱って新しいトラックを作っていた。
      全部が数字なのに番号として読めないもの（全角・先頭の 0・範囲外）は所見にする。名前は同じ名前のトラックを全部返す。
    """
    t = (track or "").strip()
    if _TRACK_NO.fullmatch(t):
        if int(t) < len(tracks):
            return [int(t)], ""
        return [], (f"トラックの番号 {t} は無い（いまの伴奏は [0]〜[{len(tracks) - 1}] の {len(tracks)} 本。"
                    "inspect_accompaniment の番号を書く。新しい楽器を足すなら番号ではなく名前を書く）")
    if t.isdigit():
        return [], (f"トラックの番号『{track}』は読めない（番号は inspect_accompaniment の [番号] と同じ半角の数字で、"
                    "先頭に 0 を付けない）")
    if not t:
        return [], "track が空（inspect_accompaniment の番号か、トラックの名前を書く）"
    return [i for i, x in enumerate(tracks) if x.name == t], ""


@coordinator.tool
@_traced
@with_scope
def inspect_accompaniment(ctx: RunContext[MusicSession], start: str = "", end: str = "") -> str:
    """伴奏の観察（アレンジ用）。区間省略時=**全体要約**（トラック表・音域・活動範囲＝リードシート級）、
    区間指定時（`start`・`end` は `小節:拍`＝inspect_song の表と同じ書き方。終わりは含まない）=要約＋その区間の
    **小節コード推定（和音の取れない小節は `N.C.`）＋トラック別ノート一覧**（高さ@小節:拍×長さ拍v強さ。高さは国際式の音名 `C4`＝中央のド、
    ドラムトラックは `bass_drum_1` `acoustic_snare` のように打楽器名＝伴奏のドラムの行と同じ GM の正式な名前）。
    全ノートは一度に見ない（3000音規模）＝人間のアレンジャーと同じく全体は要約で把握し、書き換える区間だけ詳細を見る。"""
    from accomp_edit import estimate_bar_chords, load_accomp, summarize
    from song_edits import at_text, score_bar_ticks, span_text, span_ticks
    pre = _accomp_ctx(ctx)
    if isinstance(pre, str):
        return pre
    mid, tempo = pre
    tracks = load_accomp(mid)
    bar = score_bar_ticks(ctx.deps.score)
    out = [summarize(tracks, tempo, bar)]
    if not (strip_ascii_ws(start) or strip_ascii_ws(end)):     # 位置は半角の記法（前後は半角の空白とタブだけ・§0-6）
        out.append("→ 書き換えたい区間を start/end（`小節:拍`）で指定して詳細（コード＋ノート）を観察してから "
                   "replace_accompaniment_region で置き換える。")
        return "\n".join(out)
    try:
        s_tick, e_tick = span_ticks(start, end, bar)
    except ValueError as e:
        return "\n".join(out + [f"区間を見られなかった: {e}"])
    spb = 60.0 / tempo
    w0, w1 = s_tick / 480 * spb, e_tick / 480 * spb
    b0, b1 = s_tick // bar, -(-e_tick // bar)
    chords = estimate_bar_chords(tracks, tempo, b0, b1, bar / 480)
    out.append(f"区間 {span_text(s_tick, e_tick, bar)}（終わりは含まない）")
    out.append("小節コード推定(参考): " + " ".join(f"{b0 + i + 1}小節={c}" for i, c in enumerate(chords)))
    shown = total = 0
    unnamed: set[str] = set()
    from tools.diagnosis import pitch_speller
    sp = pitch_speller(ctx)                                # 鳴る高さを区間の調の向きで（曲の表と同じ綴り）
    for i, tr in enumerate(tracks):
        sel = [n for n in tr.notes if w0 <= n.start_sec < w1]
        total += len(sel)
        if not sel:
            continue
        # 高さは国際式の音名（`replace_accompaniment_region` の `note`）、打楽器は名前（`drum`）。以前は MIDI 番号・GM 番号
        names = [drum_shown(n.key) if tr.is_drums else sp(n.key, n.start_sec / spb * 480) for n in sel[:60]]
        unnamed |= {x for x in names if tr.is_drums and x.startswith("gm")}
        body = " ".join(f"{nm}@{at_text(n.start_sec / spb * 480, bar)}×{n.dur_sec / spb:.2f}v{n.vel}"
                        for nm, n in zip(names, sel[:60]))
        shown += min(len(sel), 60)
        out.append(f"  [{i}] {tr.name}: {len(sel)}音 {body}" + ("（以降省略）" if len(sel) > 60 else ""))
    # ★2026-10-02 から GM の打楽器 35〜81 は全部名前がある（`timbre_map.GM_DRUM`）。番号のまま出るのは、その外の番号
    #   （GS 音源の拡張の 27〜34・82〜87 など）だけ。以前は採譜（YourMT3）に出る 35・40・41・43・48・57 も `gm35` と出て書けなかった
    if unnamed:
        out.append(f"※{' '.join(sorted(unnamed))} は GM の打楽器（35〜81）の外の番号で、名前が無い（番号のまま出した）。"
                   "drum には書けないので、その区間を置き換えると消える——残したいなら置き換えない（名前の一覧: "
                   + " / ".join(DRUM_NAMES) + "）。")
    if total > shown:
        out.append(f"※区間が広く{total - shown}音を省略した。区間を狭めて（8小節程度以下推奨）見直すと全音見える。")
    return "\n".join(out)

@coordinator.tool
@_traced
def replace_accompaniment_region(
        ctx: RunContext[MusicSession], start: str, end: str, track: str, notes: list[AccompNote],
        # ★**音色番号は1始まり**（2026-10-02 本人決定・案①）。以前は MIDI の program change の値（0始まり）で、編成の
        #   `Instrument.gm`（1始まり）と物差しが2つあった。MIDI へ書くときだけ機械が1を引く（組み立て＝訳すだけ）
        program: Annotated[JsonInt | None, Field(ge=1, le=128, description=(
            "GM の音色番号（**1始まり**・1〜128＝GM の音色表の番号・編成の楽器の番号と同じ物差し。例 1=アコースティックピアノ "
            "33=アコースティックベース 66=アルトサックス。inspect_accompaniment の `prog` と同じ番号）。"
            "既存のトラックの音色を差し替える・新しいトラックを作るときに書く"
            "（新しい高さのトラックでは必ず書く）。ドラムのトラックには書かない"))] = None,
        channel: Annotated[JsonInt | None, Field(ge=0, le=15, description=(
            "新しいトラックの MIDI チャンネル（0〜15・ドラムは 9）。省略すると空いているチャンネル（9 以外）"))] = None) -> str:
    """伴奏の指定トラック・区間 [start, end)（どちらも `小節:拍`＝inspect_song の表と同じ書き方。終わりは含まない）を
    notes で置き換える（**歌には触れない**・スタイルアレンジ用）。
    track=inspect_accompaniment の番号（`[番号]`・半角の数字）or トラック名。**既存に無い名前なら新規トラック追加**
    （channel 指定可・ドラムは ch9。高さのトラックを足すときは program を必ず書く）。同じ名前のトラックが何本もあるときは番号で指す。
    program=GM音色番号（1始まり・1〜128＝GM の音色表の番号）で楽器変更可（例: 33=アコースティックベース 66=アルトサックス。既存トラックの音色もこれで差替）。
    notes=[] で区間を無音化。置換後は自動で MIDI 再構成→FluidSynth レンダ。
    各音は `at`（`小節:拍`・**区間の中だけ**）・`beats`（長さ・拍）・`vel`（強さ 1〜127・必ず書く）と、高さを `note`（国際式の音名 `C4`＝中央のド・`F#3`・`Bb2`）で、
    ドラムトラック（ch9）だけ `drum`（打楽器名 `bass_drum_1` `acoustic_snare` `closed_hi_hat` …＝伴奏のドラムの行と同じ GM の正式な名前）で書く。
    区間の外の `at`・読めない番号・同じ名前が何本もある名前は、**何も書き換えずに返す**。
    アレンジは全体方針（スタイル・編成・コード）を決めてから構成単位（8小節程度）ごとに。仕上げに evaluate_timing で整合確認。
    ★音を置く・音色を変える・楽器を足すのは「大きい編集」: 書き出しの前に最終の点検役がまとめて見る（インストは完成品を作り直す前）。
      カバーは最終の点検役を通らないので、書き出しの前に review_progress（最終検問）を必ず呼ぶ。"""
    from accomp_edit import ANote, ATrack, load_accomp, save_accomp
    from harmony import beats_to_ticks                     # 長さの読み方は声部・replace_region と同じ1つ（3等分の 0.33 も）
    from sf2_render import render_midi
    from song_edits import at_tick, score_bar_ticks, span_text, span_ticks
    pre = _accomp_ctx(ctx)
    if isinstance(pre, str):
        return pre
    mid, tempo = pre
    spb = 60.0 / tempo
    bar = score_bar_ticks(ctx.deps.score)
    try:                                                   # 位置が読めない・拍が1小節を超える＝書き換えずに返す（ファイルはまだ書いていない）
        s_tick, e_tick = span_ticks(start, end, bar)
        at_ticks = [at_tick(n.at, bar) for n in notes]
    except ValueError as e:
        return f"書き換えなかった: {e}"
    # ★**区間の外の音は置かない**（2026-10-02・洗い出し H11）。区間の外の元の音は消えないので、以前は同じ位置に2音が重なって鳴った
    outside = [f"{i}番目（{n.at}）" for i, (n, t) in enumerate(zip(notes, at_ticks)) if not (s_tick <= t < e_tick)]
    if outside:
        return (f"書き換えなかった: 区間 {span_text(s_tick, e_tick, bar)}（終わりは含まない）の外に置いた音がある: "
                f"{', '.join(outside[:8])}{' ほか' if len(outside) > 8 else ''}。区間の外の元の音は消えないので、"
                "そのまま置くと同じ位置に2音になる。区間を広げるか、その音を外して書く。")
    tracks = load_accomp(mid)
    hit, why = _tracks_named(tracks, track)
    if why:
        return f"書き換えなかった: {why}"
    if len(hit) > 1:                                       # 同じ名前が区間ごとに何本もある（どれを書き換えるか機械は選ばない）
        return (f"書き換えなかった: 『{track.strip()}』という名前のトラックが {len(hit)} 本ある（"
                + " ".join(f"[{i}]" for i in hit[:12]) + (" ほか" if len(hit) > 12 else "")
                + "）。inspect_accompaniment で区間に鳴っているトラックを見て、番号で指す。")
    created = not hit
    # ★**channel は新しいトラックを作るときだけ**（2026-10-03・再レビュー）。以前は既存のトラックでは黙って捨て、
    #   使用中のチャンネル（ドラムの 9 以外）に新しいトラックを重ねても断らなかった（同じチャンネルの音色が取り合いになる）
    if not created and channel is not None and channel != tracks[hit[0]].channel:
        return (f"書き換えなかった: channel は新しいトラックを作るときだけ書く（既存のトラック『{tracks[hit[0]].name}』は"
                f" ch{tracks[hit[0]].channel} のまま変えられない）。channel を外して呼び直す。")
    if created and channel is not None and channel != 9 and (used := [t for t in tracks if t.channel == channel]):
        return (f"書き換えなかった: チャンネル {channel} は使用中（{' / '.join(t.name for t in used[:4])}）——同じチャンネルに"
                "別の楽器を重ねると音色が取り合いになる。channel を省く（空いているチャンネルを機械が選ぶ）か、空いている番号を書く。")
    if created:                                                # 新規トラック（楽器追加。サックス等）
        if channel is not None:
            ch = channel                                       # 明示指定（ドラム追加なら ch9 を指定）
        else:                                                  # 自動採番: ch9(GMドラム固定)と使用中chを避ける
            free = [c for c in range(16) if c != 9 and c not in {t.channel for t in tracks}]
            if not free:                                       # （旧実装は max+1 で ch9 に落ち、旋律楽器が打楽器音になるバグ）
                return "MIDIチャンネルが満杯（16）。既存トラックを指定して置き換えて。"
            ch = free[0]
        # ★**高さのトラックを足すときは音色を必ず書く**（2026-10-02・洗い出し 2-2）。以前は省くと黙ってピアノ（0）にした
        if ch != 9 and program is None:
            return (f"書き換えなかった: 『{track.strip()}』という名前のトラックは無いので新しく作ることになるが、program（GM の音色番号・"
                    "1始まり）が無い。足す楽器の program を書く（名前の打ち間違いなら、inspect_accompaniment の番号か名前で指す）。")
        # `ATrack.program` は MIDI の program change の値（0始まり）。道具の番号（1始まり）から1を引くのはここと下の差し替えだけ
        sel = ATrack(track.strip(), ch, program - 1 if program is not None else 0)
        tracks.append(sel)
    else:
        sel = tracks[hit[0]]
    if program is not None and sel.is_drums:               # ドラムは音色番号を持たない（キット固定）＝黙って捨てない
        return "書き換えなかった: ドラムのトラック（ch9）には program を書けない（音色はキットで決まる）。program を外して呼び直す。"
    # ★ドラムトラックは `drum`、ほかは `note`（音名）。食い違いは書き換えずに返す（ファイルはまだ書いていない）
    wrong = [i for i, n in enumerate(notes) if (n.drum is None) == sel.is_drums]
    if wrong:
        return (f"書き換えなかった: トラック『{sel.name}』は{'ドラム（ch9）' if sel.is_drums else '高さのある楽器'}なので、"
                f"各音は {'drum（打楽器名 `bass_drum_1` `acoustic_snare` …＝GM の正式な名前）' if sel.is_drums else 'note（国際式の音名 `C4`＝中央のド）'} で書く"
                f"（{len(wrong)}音が逆の欄: {', '.join(str(i) for i in wrong[:8])}番目）。")
    if program is not None:
        sel.program = program - 1                          # 1始まりの GM 番号 → MIDI の値
    w0, w1 = s_tick / 480 * spb, e_tick / 480 * spb
    kept = [n for n in sel.notes if not (w0 <= n.start_sec < w1)]
    new = [ANote(n.key, t / 480 * spb, beats_to_ticks(n.beats) / 480 * spb, n.vel) for n, t in zip(notes, at_ticks)]
    sel.notes = sorted(kept + new, key=lambda n: n.start_sec)
    song_dir = SONGS / ctx.deps.song_id
    save_accomp(tracks, tempo, mid)
    render_midi(mid, song_dir / "accompaniment.wav", gain=0.8)
    # ★**点検待ちの束へ**（2026-10-04・案C）。音を置く・音色を変える・新しいトラックは大、区間を無音にするだけは中。
    #   インストは次の `_after_accomp_edit` が完成品を作り直す前に束を見るので、その前に積む
    from tools.editing import note_edit
    _keys = (["replace_accompaniment_region.new_track"] if created else []) \
        + (["replace_accompaniment_region.program"] if program is not None else []) \
        + (["replace_accompaniment_region.notes"] if notes else []) or ["replace_accompaniment_region.silence"]
    _gate = note_edit(ctx, _keys, f"伴奏『{sel.name}』の区間 {span_text(s_tick, e_tick, bar)} を{len(new)}音に置き換えた"
                                  + (f"（音色 prog{program}）" if program is not None else ""))
    nxt = _after_accomp_edit(ctx, song_dir)                # vpr/仮歌を更新（歌は不変）。インストは直した伴奏から完成品を作り直す
    nxt = (f"{_gate}\n" if _gate and getattr(ctx.deps.score, "form", "song") != "instrumental" else "") + nxt
    gone = "" if sel.notes else "（このトラックは音がゼロになったので**編成から外れた**）"
    # ★**オリジナル曲では、この編集は `render_final` で戻る**（2026-09-20 に実測）。
    #   伴奏MIDIは楽譜の `arrangement` から作り直される派生物で、区間単位の書き換えは
    #   楽譜側に写す先が無い。黙って戻ると「直したのに直っていない」になるので、必ず言う。
    tmp = ("★**この編集はオリジナル曲では `render_final` で元に戻る**"
           "（伴奏は楽譜の編成から作り直されるため）。恒久的に変えるなら "
           "`rearrange_song` か `remake_from_step('編成')` を使う。"
           if ctx.deps.score is not None and not ctx.deps.score.arrangement.is_empty() and not ctx.deps.faithful
           else "")                                         # 伴奏が楽譜に無い（パートが0本・F1）なら書かない
    # 返事の `prog` は道具に書く番号と同じ1始まり（MIDI の値＋1）
    made = (f"★**新しいトラック『{sel.name}』を作った**（ch{sel.channel}・prog{sel.program + 1}。既存の名前・番号ではなかったため）。"
            if created else "")
    return (f"{made}伴奏トラック『{sel.name}』の区間 {span_text(s_tick, e_tick, bar)}（終わりは含まない）を置換: {len(new)}音を配置"
            f"（トラック計{len(sel.notes)}音・prog{sel.program + 1}）{gone}。伴奏レンダ済み。{tmp}"
            f"次の区間へ進むか、仕上げる。{nxt}")


@coordinator.tool
@_traced
def remove_accompaniment_track(ctx: RunContext[MusicSession], track: str) -> str:
    """**伴奏の楽器を1つ、編成から外す**（『ストリングスは要らない』『ベースを抜いて』）。

    `track` は**楽器の名前**（`drums` / `bass` / `chord` / `lead` 等）か、
    `inspect_accompaniment` に出る**番号**。**歌には触れない。**

    ★★**名前で指定すると、その名前のトラックを全部外す。** 伴奏MIDIは**区間ごとに
      トラックが分かれている**ので、同じ楽器が何本もある（『加速させて』は 43トラック中
      **ドラムだけで16本**）。1本だけ外すと**鳴り続けて「消えていない」ように聞こえる**。
      1本だけ狙うときは**番号**で指定する。
    ★**区間を無音化する（`replace_accompaniment_region(notes=[])`）のとは違う。**
      あちらは音を消すだけで**トラック自体は編成に残る**（鳴らないのに楽器としては居る）。
      楽器そのものを減らしたいときはこちら。
    ※全部を組み直したいなら `rearrange_song` / `remake_from_step('編成')`。**こちらは1つの楽器だけ**。
    """
    from accomp_edit import load_accomp, save_accomp
    from sf2_render import render_midi
    pre = _accomp_ctx(ctx)
    if isinstance(pre, str):
        return pre
    mid, tempo = pre
    tracks = load_accomp(mid)
    hit, why = _tracks_named(tracks, track)                # 番号＝その1本だけ／名前＝同名を全部（区間ごとに分かれているため）
    if why:                                                # 読めない番号・範囲外の番号は名前として探さない（2026-10-02）
        return f"外さなかった: {why}"
    gone = [tracks[i] for i in hit]
    # ★**主旋律の楽器のトラックは外さない**（2026-10-03・再レビュー）。主旋律の楽器は編成が決めたもの（楽譜の `lead`）で、
    #   伴奏のパート（`arrangement.parts`）には無い——MIDI から外しても楽譜は変わらず、次の書き出しで黙って戻っていた（歌ものの重ねでも）
    from midi_out import LEAD_TRACK
    if any(t.name == LEAD_TRACK for t in gone):
        _trace(ctx.deps.song_id, "remove_accompaniment_track.refused", reason="lead")
        return (f"外さなかった: 『{LEAD_TRACK}』は主旋律を弾く楽器（編成が決めたもの）。伴奏 MIDI から外しても楽譜の編成は変わらず、"
                "次の書き出しで戻る。主旋律の楽器をやめる・替えるなら remake_from_step('編成', 頼み)（編成から作り直す）。"
                + ("インストは主旋律を楽器で弾く曲なので、楽器をやめるなら歌を乗せる change_form('song', 頼み) も行き先。"
                   if getattr(ctx.deps.score, "form", "song") == "instrumental" else "")
                + "その旨をマスターに伝える。")
    by_number = bool(_TRACK_NO.fullmatch(track.strip()))
    if not gone:
        names = {}
        for t in tracks:
            names[t.name] = names.get(t.name, 0) + len(t.notes)
        return (f"『{track}』という楽器が無い。いまの編成: "
                + " / ".join(f"{k}({v}音)" for k, v in names.items()))
    rest = [t for t in tracks if t not in gone]
    if not rest:
        return "それを外すと伴奏が無くなる。減らすのではなく作り直しを。"
    song_dir = SONGS / ctx.deps.song_id

    # ★★**オリジナル曲の伴奏MIDIは、楽譜から毎回作り直される派生物**（2026-09-20）。
    #   `render_final` → `compose` → `score_to_midi` が `accompaniment.mid` を上書きするので、
    #   **MIDI だけ直しても次の書き出しで戻る**。実際に起きた: ドラム16トラック・1230音を
    #   外したのに、4分後の `render_final` で**全部復活**していた（ログ上は「外した」と出ている
    #   ので、報告だけ見ると成功に見える）。**編成の正は `score.arrangement.parts`** なので、
    #   そちらからも同じ役割（role）を外す。カバー（faithful）は MIDI が正なのでこの処理は要らない。
    back = ""
    if ctx.deps.score is not None and not ctx.deps.score.arrangement.is_empty():   # 伴奏が無ければパートが0本（F1）
        arr = ctx.deps.score.arrangement
        if by_number:                                      # 番号＝1本だけ。役割の単位では消せない
            back = ("※番号で1本だけ外したので、**次の `render_final` で戻る**"
                    "（オリジナル曲の伴奏は楽譜の編成から作り直される）。"
                    "恒久的に外すなら**楽器の名前**で呼び直して。")
        else:
            keep = [p for p in arr.parts if p.role != track.strip()]
            if len(keep) != len(arr.parts):
                ctx.deps.score = ctx.deps.score.model_copy(
                    update={"arrangement": arr.model_copy(update={"parts": keep})})
                (song_dir / "score.json").write_text(
                    ctx.deps.score.model_dump_json(indent=2), encoding="utf-8")
                back = f"楽譜の編成からも外した（{len(arr.parts)}→{len(keep)}パート）＝作り直しても戻らない。"

    save_accomp(rest, tempo, mid)
    render_midi(mid, song_dir / "accompaniment.wav", gain=0.8)
    from tools.editing import note_edit                    # 点検待ちの束へ（楽器を外す＝引き算＝中・2026-10-04・案C）
    note_edit(ctx, "remove_accompaniment_track", f"伴奏から『{gone[0].name}』を外した（{len(gone)}トラック）")
    nxt = _after_accomp_edit(ctx, song_dir)                # インストは直した伴奏から完成品を作り直す
    _trace(ctx.deps.song_id, "remove_accompaniment_track",
           removed=gone[0].name, n_tracks=len(gone), rest=len(rest))
    left = {}
    for t in rest:
        left[t.name] = left.get(t.name, 0) + len(t.notes)
    return (f"伴奏から『{gone[0].name}』を外した（{len(gone)}トラック・{sum(len(t.notes) for t in gone)}音）。"
            f"残りの編成: " + " / ".join(f"{k}({v}音)" for k, v in left.items())
            + f"。{back}伴奏レンダ済み。"
            + (nxt if getattr(ctx.deps.score, "form", "song") == "instrumental" else
               "★**歌声は書き出し直しが要る**（vpr に伴奏が入っているので、前の完成品には外した楽器が残っている）。" + nxt))
