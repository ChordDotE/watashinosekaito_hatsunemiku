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

"""オリジナル曲のフルパイプライン。

original_song.py の SongScore から:
  1. score.json / vocal.vpr / vocal.vsqx / vocal.ustx（NT2 用）
  2. 伴奏3パート: pad(コード) / bass / arpeggio → MIDI + 各 stem WAV + 合成 WAV
  3. 仮歌（正弦波リード）+ 伴奏 → final.wav（試聴用）
を output/hikari_no_melody/ に生成する。

本物の歌声にするには vocal.vpr を Piapro Studio NT2 で開いて
vocal.wav として書き出し → 本スクリプトを再実行（仮歌の代わりに使われる）。
"""

import sys
import wave
from pathlib import Path

import mido
import numpy as np
from libresvip.extension.manager import plugin_manager
from libresvip.model.base import Note, Project, SingingTrack, SongTempo, TimeSignature

from original_song import BEATS_PER_BAR, build_original_score
from score_schema import TICKS_PER_QUARTER as Q
from score_schema import SongScore

OUTPUT_DIR = Path(__file__).parents[1] / "output" / "hikari_no_melody"   # レガシーデモ用（runtimeでは未使用）
SR = 44100

def chord_root_and_triad(chord: str) -> tuple[int, list[int]]:
    """コード名 → (根音, [根音・3度・5度])（MIDI・根音は `C4`〜`B4` あたり）。手書きの固定曲のデモ（`band_song_build`）が使う。

    ★**構成音は曲づくりの組み立てと同じ `song_audio.chord_notes` で読む**（2026-10-01・仕様書 §3 の突き合わせ）。以前はここに
      根音と m だけを読む2本目があり、`Bdim`・`Caug`・`Csus4` を長三和音で鳴らし、`N.C.` で `KeyError` になっていた
      （2026-10-01 の朝に根音の ♭ だけ足していた・判-I）。読めない名前・`N.C.` は `ValueError`（デモは和音のある小節しか書かない）。
    """
    from song_audio import chord_notes                     # 呼ぶ時に読む（song_audio → band_song_build → ここ、の循環を避ける）
    notes = chord_notes(chord)
    if not notes:
        raise ValueError(f"コード名『{chord}』は和音にならない（読めない名前か N.C.）")
    return notes[0], notes[:3]


# ---------------------------------------------------------------- パート生成
# 各パート = [(MIDIキー, 開始tick, 長さtick, ベロシティ), ...]
Part = list[tuple[int, int, int, int]]


def make_pad_part(score: SongScore) -> Part:
    """コードを全音符で鳴らすパッド（中音域）。"""
    events: Part = []
    for span in score.chords:
        _, triad = chord_root_and_triad(span.chord)
        for key in triad:
            events.append((key, span.start_tick, span.duration_tick, 70))
    return events


def make_bass_part(score: SongScore) -> Part:
    """ルート中心のベース。1拍目(2拍)・3拍目(1拍)・4拍目は5度(1拍)。"""
    events: Part = []
    for span in score.chords:
        root, _ = chord_root_and_triad(span.chord)
        low = root - 24
        t = span.start_tick
        events.append((low, t, 2 * Q, 90))
        events.append((low, t + 2 * Q, Q, 85))
        events.append((low + 7, t + 3 * Q, Q, 80))
    return events


def make_arpeggio_part(score: SongScore) -> Part:
    """8分音符のアルペジオ（高音域・控えめ）。"""
    pattern = [0, 7, 12, 16, 12, 16, 12, 7]  # ルート基準の度数（半音）
    events: Part = []
    for span in score.chords:
        root, triad = chord_root_and_triad(span.chord)
        minor_third = triad[1] - root  # 3 or 4
        for i, off in enumerate(pattern):
            off = minor_third + 12 if off == 16 else off  # 10度はコードに追従
            key = root + 12 + off
            events.append((key, span.start_tick + i * (Q // 2), Q // 2, 55))
    return events


def make_vocal_part(score: SongScore) -> Part:
    return [(n.key, n.start_tick, n.duration_tick, 100) for n in score.vocal_notes]


# ---------------------------------------------------------------- シンセ（numpy）
_TIMBRES = {
    # name: (attack_s, decay_rate, [(harmonic, gain), ...])
    "pad": (0.06, 0.25, [(1, 1.0), (2, 0.30), (3, 0.10)]),
    "bass": (0.01, 0.45, [(1, 1.0), (2, 0.15)]),
    "pluck": (0.005, 3.0, [(1, 1.0), (2, 0.45), (3, 0.20)]),
    "lead": (0.02, 0.12, [(1, 1.0), (2, 0.18)]),
}


# ★GM の音色 id（`gm0-28`）→ 合成音の近い形（`_GM_FALLBACK`）は 2026-09-29 に消した。伴奏を SoundFont で
#   鳴らせないときに numpy の合成音へ落とす経路（`song_audio.render_accompaniment`）でだけ使っていた。
#   ここに残る合成音は、手書きの固定曲のデモ（このファイルの `main`・`band_song_build`）が使う4種だけ。


def _timbre_of(timbre: str) -> tuple:
    """音色名 → 合成のパラメータ（デモ用の4種＋`band_song_build` が足す guitar）。知らない名前は pad。"""
    if timbre in _TIMBRES:
        return _TIMBRES[timbre]
    return _TIMBRES["pad"]                                 # 最後の受け皿（デモ用）


def render_part(part: Part, score: SongScore, timbre: str, gain: float) -> np.ndarray:
    sec_per_tick = 60.0 / score.tempo / Q
    total = int((score.duration_ticks() * sec_per_tick + 1.5) * SR)
    buf = np.zeros(total)
    attack, decay, harmonics = _timbre_of(timbre)
    for key, start_tick, dur_tick, vel in part:
        freq = 440.0 * 2 ** ((key - 69) / 12)
        i0 = int(start_tick * sec_per_tick * SR)
        n = int(dur_tick * sec_per_tick * SR)
        t = np.arange(n) / SR
        env = np.minimum(t / attack, 1.0) * np.exp(-decay * t)
        env *= np.minimum((n - np.arange(n)) / (0.04 * SR), 1.0)  # リリース
        sig = np.zeros(n)
        for h, g in harmonics:
            sig += g * np.sin(2 * np.pi * freq * h * t)
        buf[i0 : i0 + n] += (vel / 127) * env * sig
    return gain * buf


def write_wav(path: Path, data: np.ndarray) -> None:
    peak = np.max(np.abs(data)) or 1.0
    pcm = (data / peak * 0.85 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1)
    return data.astype(np.float64) / 32768.0


# ---------------------------------------------------------------- MIDI / vpr
def write_midi(score: SongScore, parts: dict[str, tuple[Part, int]], path: Path) -> None:
    """parts: name -> (events, program番号)"""
    mid = mido.MidiFile(ticks_per_beat=Q)
    for ch, (name, (part, program)) in enumerate(parts.items()):
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        if ch == 0:
            track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(score.tempo), time=0))
        track.append(mido.Message("program_change", program=program, channel=ch, time=0))
        msgs: list[tuple[int, str, int, int]] = []
        for key, start, dur, vel in part:
            msgs.append((start, "note_on", key, vel))
            msgs.append((start + dur, "note_off", key, 0))
        msgs.sort(key=lambda m: (m[0], m[1] == "note_on"))
        prev = 0
        for tick, kind, key, vel in msgs:
            track.append(mido.Message(kind, note=key, velocity=vel, channel=ch, time=tick - prev))
            prev = tick
    mid.save(path)


def dump_vocal_files(score: SongScore) -> None:
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
        path = OUTPUT_DIR / f"vocal.{fmt}"
        plugin_manager.plugins["svs"][fmt].dump(path, project, options[fmt])
        print(f"  [OK] {fmt}: {path.name} ({path.stat().st_size} bytes)")


# ---------------------------------------------------------------- main
def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score = build_original_score()
    (OUTPUT_DIR / "score.json").write_text(score.model_dump_json(indent=2), encoding="utf-8")
    dur_sec = score.duration_ticks() * 60.0 / score.tempo / Q
    bars = score.duration_ticks() // (BEATS_PER_BAR * Q)
    print(f"楽曲: {score.title} / {bars}小節 / BPM {score.tempo} / 約{dur_sec:.0f}秒 / "
          f"{len(score.vocal_notes)}ノート")

    print("-- ボーカル譜面 (NT2 用) --")
    dump_vocal_files(score)

    print("-- 伴奏3パート --")
    pad, bass, arp = make_pad_part(score), make_bass_part(score), make_arpeggio_part(score)
    write_midi(score, {"pad": (pad, 0), "bass": (bass, 33), "arpeggio": (arp, 46)},
               OUTPUT_DIR / "accompaniment.mid")
    pad_w = render_part(pad, score, "pad", 0.50)
    bass_w = render_part(bass, score, "bass", 0.80)
    arp_w = render_part(arp, score, "pluck", 0.40)
    accomp = pad_w + bass_w + arp_w
    for name, data in (("stem_pad", pad_w), ("stem_bass", bass_w),
                       ("stem_arpeggio", arp_w), ("accompaniment", accomp)):
        write_wav(OUTPUT_DIR / f"{name}.wav", data)
    print(f"  [OK] accompaniment.mid / stem×3 / accompaniment.wav")

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
    mix[: len(accomp)] += accomp * 0.7
    write_wav(OUTPUT_DIR / "final.wav", mix)
    print(f"  [OK] final.wav ({n / SR:.1f}秒)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
