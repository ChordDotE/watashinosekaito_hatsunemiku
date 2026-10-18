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

"""伴奏MIDI（accompaniment.mid）の検分・区間×トラック編集の決定論層。

アレンジ（ジャズ化等のスタイル変更）を **LLMの譜面補正だけ** で行うための基盤:
MIDI → 構造データ（トラック別ノートリスト・秒→拍換算）→ LLMが区間×トラック単位で書き換え →
MIDI 再構成 → FluidSynth レンダ。歌側の inspect_region/replace_region の伴奏版。

- 全ノートをLLMに渡さない（シャイニングスターで約3000音≒3万トークン超・書き戻しで写し間違い）。
  全体は要約（トラック表＋活動範囲＋コード推定）、詳細は区間指定で渡す（人間のリードシート方式）
- 拍(beat)は score.tempo 基準（歌の replace_region と同じ座標系）。MIDI内部の秒は tempo map で吸収
- 楽器は固定しない: トラックごとに GM program を持ち、変更・新規トラック追加可（サックス等）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mido

TPB = 480
_GM_NAMES = {0: "Piano", 24: "Guitar(clean)", 30: "Guitar(dist)", 32: "AcousticBass",
             33: "FingerBass", 48: "Strings", 52: "Choir", 56: "Trumpet", 64: "SopranoSax",
             65: "AltoSax", 66: "TenorSax", 67: "BaritoneSax"}


@dataclass
class ANote:
    key: int
    start_sec: float
    dur_sec: float
    vel: int


@dataclass
class ATrack:
    name: str
    channel: int
    program: int                 # MIDI の program change の値（0始まり）。道具が見せる・受け取る番号はこれ＋1（1始まり・2026-10-02）。ch9(ドラム)は意味を持たない（キット固定）
    notes: list[ANote] = field(default_factory=list)

    @property
    def is_drums(self) -> bool:
        return self.channel == 9


def load_accomp(mid_path: Path) -> list[ATrack]:
    """accompaniment.mid → トラック別ノート（絶対秒）。tempo map は merge して吸収。"""
    m = mido.MidiFile(mid_path)
    # 絶対秒はトラック横断の tempo map に依存 → merged で一括計算し (track元, msg, abs_sec) を得る
    abs_sec: list[list[tuple[float, mido.Message]]] = [[] for _ in m.tracks]
    for ti, tr in enumerate(m.tracks):
        t = 0
        # 本pocの生成MIDIは set_tempo が先頭トラックに1個だけ（YourMT3/自作とも単一テンポ）なので、
        # 全トラック共通の初期テンポとして先頭トラックの set_tempo を先に拾う
        tempo = next((msg.tempo for msg in m.tracks[0] if msg.type == "set_tempo"), 500000)
        for msg in tr:
            t += mido.tick2second(msg.time, m.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            abs_sec[ti].append((t, msg))
    # 先頭トラックに set_tempo がある形式（YourMT3/自作とも）はトラック別計算でズレない（全て単一テンポ）
    tracks: list[ATrack] = []
    # ★**音色の指定が無いトラックは、同じチャンネルで先に指定された音色を引き継ぐ**（2026-10-10）。
    #   伴奏 MIDI は区間ごとにトラックが分かれ、program_change を出すのは楽器の最初のトラックだけ
    #   （midi_out.py の done_pc）。ここで指定の無いトラックを 0 番（ピアノ）と読んでいたため、伴奏を1区間でも手直しすると
    #   保存し直しで全トラックに「ピアノ」が書かれ、シンセベースも和音もピアノで鳴っていた（『てのひらのシグナル』v8・v10・v11）
    chan_prog: dict[int, int] = {}
    for ti, evs in enumerate(abs_sec):
        name, ch, prog = f"Track{ti}", None, None
        on: dict[tuple[int, int], tuple[float, int]] = {}
        notes: list[ANote] = []
        for t, msg in evs:
            if msg.type == "track_name":
                name = msg.name
            elif msg.type == "program_change":
                prog, ch = msg.program, msg.channel
                chan_prog[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                ch = msg.channel if ch is None else ch
                on[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on") and (msg.channel, getattr(msg, "note", -1)) in on:
                s, v = on.pop((msg.channel, msg.note))
                notes.append(ANote(msg.note, s, max(t - s, 0.03), v))
        if notes or name != f"Track{ti}":
            ch = ch if ch is not None else 0
            prog = prog if prog is not None else chan_prog.get(ch, 0)
            tracks.append(ATrack(name, ch, prog, sorted(notes, key=lambda n: n.start_sec)))
    return [tr for tr in tracks if tr.notes]


def save_accomp(tracks: list[ATrack], tempo_bpm: float, mid_path: Path) -> None:
    """トラック別ノート（絶対秒）→ MIDI（tpb=480・単一テンポ=score.tempo）。

    ★**音がゼロになったトラックは書き出さない**（2026-09-20）。
      `replace_accompaniment_region(notes=[])` で区間を無音化すると、
      **音の無い空トラックが MIDI に残り続けていた**——鳴らないのに編成には居るので、
      `inspect_accompaniment` には出るし、数えると楽器が多く見える。
      本人:「無音化は実質そうなんだけど、**データとしては汚くなる**」。
      消したいときは消えるのが正しい。
    """
    mid = mido.MidiFile(ticks_per_beat=TPB)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    mid.tracks.append(meta)
    spb = 60.0 / tempo_bpm
    tracks = [t for t in tracks if t.notes]                # 空トラックは残さない
    for tr in tracks:
        mt = mido.MidiTrack()
        mt.append(mido.MetaMessage("track_name", name=tr.name, time=0))
        if not tr.is_drums:
            mt.append(mido.Message("program_change", program=tr.program, channel=tr.channel, time=0))
        ev: list[tuple[int, int, mido.Message]] = []
        for n in tr.notes:
            s, e = int(round(n.start_sec / spb * TPB)), int(round((n.start_sec + n.dur_sec) / spb * TPB))
            # 長さ 0 の音だけは作らない（2026-10-02 まで 10tick 未満を伸ばしていた。道具に書く長さは型が tick にして 1 以上を保証し、
            # 読み込んだ MIDI の音は 0.03秒以上＝ふつうの速さで 10tick を超えるので、どちらも書いた長さのまま）
            e = max(e, s + 1)
            ev.append((s, 1, mido.Message("note_on", channel=tr.channel, note=n.key, velocity=max(1, min(n.vel, 127)))))
            ev.append((e, 0, mido.Message("note_off", channel=tr.channel, note=n.key, velocity=0)))
        prev = 0
        for tick, _, msg in sorted(ev, key=lambda x: (x[0], x[1])):
            msg.time = tick - prev
            mt.append(msg)
            prev = tick
        mid.tracks.append(mt)
    mid.save(mid_path)


# ★音名の綴りは `song_table.spell_pitch` の1つ（2026-10-01）。ここに同じことをする `_pitch_name` があった（2本目）。


def drum_shown(gm: int) -> str:
    """GM の打楽器番号 → 道具が見せる打楽器名（伴奏のドラムの行と同じ名前＝GM の正式な名前・`timbre_map.GM_DRUM_NAME`）。

    ★**打楽器は名前で見せ・名前で書かせる**（2026-10-01 本人決定・記法の仕様書 §15-2）。以前は `drum36` と GM の番号で見せていた。
    ★2026-10-02 から名前は GM の正式な名前だけ（`bass_drum_1`・`closed_hi_hat` …）で、GM の打楽器 35〜81 は番号と名前が1対1
    （以前は短い独自名で、42 に hihat と closedhat の2つがあり「表で先に書いた名前」を見せていた）。
    35〜81 の外の番号（GS 音源の拡張の 27〜34・82〜87 など）だけ `gm27` と番号のまま見せる（`replace_accompaniment_region` の `drum` には書けない）。
    """
    from timbre_map import GM_DRUM_NAME
    return GM_DRUM_NAME.get(gm, f"gm{gm}")


# ★**小節ごとのコードの推定は `chord_estimate` の1か所**（2026-10-02・組「カバーのコード」）。ここにあった `estimate_bar_chords` を
#   そのまま移した（伴奏 MIDI からと原曲の音声からを、同じ24の型・同じ出力の形にするため）。呼び手（曲の表・伴奏の観察）のために名前だけ残す。
#   和音の取れない小節は `N.C.`（2026-10-02 本人「おすすめで」・記法の仕様書 §3。以前は記法に無い `-` を書いていた）
from chord_estimate import NO_CHORD_MARK, estimate_bar_chords  # noqa: E402,F401


def summarize(tracks: list[ATrack], tempo_bpm: float, bar_ticks: int = 4 * TPB) -> str:
    """全体要約（常時コンテキスト用・リードシート級）: トラック表＋鳴っている範囲。

    ★**位置は曲の表と同じ `小節:拍`、高さは国際式の音名、打楽器は名前**（2026-10-01・記法の仕様書 §15-2・ず-18）。
      以前は活動範囲を楽譜の頭からの拍（`活動拍[0.0,64.0]`）で出し、「小節=4拍」と決め打ちしていた。
      `bar_ticks` は曲の1小節の tick（曲の拍子から）。ドラムは高さではないので音域でなく使っている打楽器名を出す。
    """
    from song_edits import at_text
    from song_table import spell_pitch
    spb = 60.0 / tempo_bpm
    lines = [f"伴奏MIDI全体: {len(tracks)}トラック tempo={tempo_bpm:.1f}（1小節={bar_ticks / TPB:g}拍・"
             "位置は `小節:拍`＝拍は4分音符＝1.00の0始まり）"]
    for i, tr in enumerate(tracks):
        ks = [n.key for n in tr.notes]
        t0 = min(n.start_sec for n in tr.notes) / spb * TPB
        t1 = max(n.start_sec + n.dur_sec for n in tr.notes) / spb * TPB
        # ★`prog` は GM の音色番号を**1始まり**で見せる（2026-10-02 本人決定・案①。`replace_accompaniment_region` の `program` と
        #   編成の楽器の番号と同じ物差し）。`ATrack.program` は MIDI の値（0始まり）のまま持つ
        inst = "Drums" if tr.is_drums else _GM_NAMES.get(tr.program, f"GM{tr.program + 1}")
        what = ("打楽器 " + " ".join(dict.fromkeys(drum_shown(k) for k in sorted(ks))) if tr.is_drums
                else f"音域{spell_pitch(min(ks))}-{spell_pitch(max(ks))}")
        lines.append(f"  [{i}] {tr.name}（{inst}/prog{tr.program + 1}/ch{tr.channel}）"
                     f" {len(tr.notes)}音 {what} 鳴っている範囲 {at_text(t0, bar_ticks)}〜{at_text(t1, bar_ticks)}")
    return "\n".join(lines)
