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

"""バンド編成曲のフルパイプライン（band_song.py → output/hashiridase/）。

編成: ボーカル + ドラム(8ビート) + エレキベース(8分刻み) + ギター(パワーコード刻み)。
ピッチ系の合成は make_original_song の render_part を流用し、
ドラムだけ専用のシンセ（キック=ピッチスイープ / スネア=ノイズ+胴鳴り / ハット=高域ノイズ）。
"""

import sys
import wave
from pathlib import Path

import mido
import numpy as np
from libresvip.extension.manager import plugin_manager
from libresvip.model.base import Note, Project, SingingTrack, SongTempo, TimeSignature

from band_song import BEATS_PER_BAR, build_band_score
from render_base import (
    SR,
    _TIMBRES,
    Part,
    chord_root_and_triad,
    load_wav,
    render_part,
    write_wav,
)
from score_schema import TICKS_PER_QUARTER as Q
from score_schema import SongScore

OUTPUT_DIR = Path(__file__).parents[1] / "output" / "hashiridase"        # レガシーデモ用（runtimeでは未使用）

# ギター: 倍音多め・短めの減衰（歪み風の明るさ）
_TIMBRES["guitar"] = (0.005, 2.2, [(1, 1.0), (2, 0.60), (3, 0.40), (4, 0.22)])

EIGHTH = Q // 2


# ---------------------------------------------------------------- パート生成
def make_bass_part(score: SongScore) -> Part:
    """8分刻みのルート弾き。小節最後の8分は5度でアプローチ。"""
    events: Part = []
    for span in score.chords:
        root, _ = chord_root_and_triad(span.chord)
        low = root - 24
        for i in range(8):
            key = low + 7 if i == 7 else low
            vel = 100 if i % 2 == 0 else 85
            events.append((key, span.start_tick + i * EIGHTH, EIGHTH, vel))
    return events


def make_guitar_part(score: SongScore) -> Part:
    """パワーコード(根+5度+オクターブ)のシンコペ刻み: 1拍・2拍裏・4拍。"""
    stab_steps = [0, 3, 6]  # 8分グリッド
    events: Part = []
    for span in score.chords:
        root, _ = chord_root_and_triad(span.chord)
        for step in stab_steps:
            for key in (root - 12, root - 5, root):
                events.append((key, span.start_tick + step * EIGHTH, EIGHTH, 85))
    return events


def make_vocal_part(score: SongScore) -> Part:
    return [(n.key, n.start_tick, n.duration_tick, 100) for n in score.vocal_notes]


# ドラム: [(楽器名, 開始tick, ベロシティ)]
DrumPart = list[tuple[str, int, int]]


def make_drum_part(score: SongScore) -> DrumPart:
    """8ビート。kick=1拍・2拍裏・3拍 / snare=2・4拍 / hihat=8分。4小節毎にフィル。"""
    events: DrumPart = []
    n_bars = len(score.chords)
    for bar in range(n_bars):
        t0 = bar * BEATS_PER_BAR * Q
        fill = (bar + 1) % 4 == 0
        for step in range(8):
            t = t0 + step * EIGHTH
            if fill and step >= 6:
                continue  # フィル区間はハット休み
            events.append(("hihat", t, 80 if step % 2 == 0 else 55))
        for step in (0, 3, 4):
            events.append(("kick", t0 + step * EIGHTH, 110))
        for step in (2, 6):
            if not (fill and step == 6):
                events.append(("snare", t0 + step * EIGHTH, 100))
        if fill:
            for i in range(4):  # 4拍目スネア16分フィル
                events.append(("snare", t0 + 3 * Q + i * (Q // 4), 80 + i * 8))
    return events


# ---------------------------------------------------------------- ドラムシンセ
def _kick(vel: float) -> np.ndarray:
    n = int(0.20 * SR)
    t = np.arange(n) / SR
    freq = 45 + 75 * np.exp(-t * 28)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return vel * np.sin(phase) * np.exp(-t * 16)


def _snare(vel: float) -> np.ndarray:
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(1).standard_normal(n)
    return vel * (0.7 * noise * np.exp(-t * 22) + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30))


def _hihat(vel: float) -> np.ndarray:
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(2).standard_normal(n + 1)
    return vel * np.diff(noise) * np.exp(-t * 70) * 0.8


def render_drums(part: DrumPart, score: SongScore) -> np.ndarray:
    sec_per_tick = 60.0 / score.tempo / Q
    total = int((score.duration_ticks() * sec_per_tick + 1.5) * SR)
    buf = np.zeros(total)
    synth = {"kick": _kick, "snare": _snare, "hihat": _hihat}
    for inst, tick, vel in part:
        sig = synth[inst](vel / 127)
        i0 = int(tick * sec_per_tick * SR)
        buf[i0 : i0 + len(sig)] += sig[: max(0, total - i0)]
    return buf


# ---------------------------------------------------------------- MIDI / ボーカル譜面
_GM_DRUM = {"kick": 36, "snare": 38, "hihat": 42}


def write_midi(score: SongScore, path: Path, bass: Part, guitar: Part, drums: DrumPart) -> None:
    mid = mido.MidiFile(ticks_per_beat=Q)

    def add_pitched(name: str, part: Part, program: int, channel: int) -> None:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        if not mid.tracks[0] is track:
            pass
        track.append(mido.Message("program_change", program=program, channel=channel, time=0))
        msgs = []
        for key, start, dur, vel in part:
            msgs.append((start, "note_on", key, vel))
            msgs.append((start + dur, "note_off", key, 0))
        msgs.sort(key=lambda m: (m[0], m[1] == "note_on"))
        prev = 0
        for tick, kind, key, vel in msgs:
            track.append(mido.Message(kind, note=key, velocity=vel, channel=channel, time=tick - prev))
            prev = tick

    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(score.tempo), time=0))

    add_pitched("bass", bass, 34, 0)      # picked bass
    add_pitched("guitar", guitar, 30, 1)  # distortion guitar

    drum_track = mido.MidiTrack()
    mid.tracks.append(drum_track)
    drum_track.append(mido.MetaMessage("track_name", name="drums", time=0))
    msgs = []
    for inst, tick, vel in drums:
        msgs.append((tick, "note_on", _GM_DRUM[inst], vel))
        msgs.append((tick + Q // 4, "note_off", _GM_DRUM[inst], 0))
    msgs.sort(key=lambda m: (m[0], m[1] == "note_on"))
    prev = 0
    for tick, kind, key, vel in msgs:
        drum_track.append(mido.Message(kind, note=key, velocity=vel, channel=9, time=tick - prev))
        prev = tick
    mid.save(path)


def dump_vocal_files(score: SongScore, out_dir: Path) -> None:
    numerator, denominator = (int(x) for x in score.time_signature.split("/"))
    project = Project(
        song_tempo_list=[SongTempo(position=0, bpm=score.tempo)],
        time_signature_list=[TimeSignature(bar_index=0, numerator=numerator, denominator=denominator)],
        track_list=[SingingTrack(
            title=score.title,
            note_list=[Note(start_pos=n.start_tick, length=n.duration_tick,
                            key_number=n.key, lyric=n.lyric) for n in score.vocal_notes],
        )],
    )
    options = {"vpr": {"default_lang_id": 0}, "vsqx": {"default_lang_id": 0}, "ustx": {}}
    for fmt in ("vpr", "vsqx", "ustx"):
        path = out_dir / f"vocal.{fmt}"
        plugin_manager.plugins["svs"][fmt].dump(path, project, options[fmt])
        print(f"  [OK] {fmt}: {path.name} ({path.stat().st_size} bytes)")


# ---------------------------------------------------------------- main
def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score = build_band_score()
    (OUTPUT_DIR / "score.json").write_text(score.model_dump_json(indent=2), encoding="utf-8")
    dur_sec = score.duration_ticks() * 60.0 / score.tempo / Q
    bars = score.duration_ticks() // (BEATS_PER_BAR * Q)
    print(f"楽曲: {score.title} / {bars}小節 / BPM {score.tempo} / 約{dur_sec:.0f}秒 / "
          f"{len(score.vocal_notes)}ノート")

    print("-- ボーカル譜面 (NT2 用) --")
    dump_vocal_files(score, OUTPUT_DIR)

    print("-- バンド3パート --")
    bass, guitar, drums = make_bass_part(score), make_guitar_part(score), make_drum_part(score)
    write_midi(score, OUTPUT_DIR / "accompaniment.mid", bass, guitar, drums)
    bass_w = render_part(bass, score, "bass", 0.75)
    gtr_w = render_part(guitar, score, "guitar", 0.45)
    drum_w = render_drums(drums, score) * 0.55
    accomp = bass_w + gtr_w + drum_w
    for name, data in (("stem_bass", bass_w), ("stem_guitar", gtr_w),
                       ("stem_drums", drum_w), ("accompaniment", accomp)):
        write_wav(OUTPUT_DIR / f"{name}.wav", data)
    print("  [OK] accompaniment.mid / stem×3 / accompaniment.wav")

    print("-- ボーカル＋ミックス --")
    # 本物（ボカロエディタ書き出し）= vocal.wav / 仮歌 = placeholder.wav
    real_vocal_path = OUTPUT_DIR / "vocal.wav"
    if real_vocal_path.exists():
        vocal = load_wav(real_vocal_path)
        print("  本物の vocal.wav を使用")
    else:
        vocal = render_part(make_vocal_part(score), score, "lead", 0.9)
        write_wav(OUTPUT_DIR / "placeholder.wav", vocal)
        print("  [OK] 仮歌（正弦波リード）を生成: placeholder.wav")
    n = max(len(vocal), len(accomp))
    mix = np.zeros(n)
    mix[: len(vocal)] += vocal * 1.0
    mix[: len(accomp)] += accomp * 0.75
    write_wav(OUTPUT_DIR / "final.wav", mix)
    print(f"  [OK] final.wav ({n / SR:.1f}秒)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
