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

"""参照 MIDI を簡易シンセで WAV 化（確認試聴用）。

FluidSynth/サウンドフォント無しでも聞けるよう、各ノートを正弦波＋倍音で鳴らす。
ドラム(ch9)は除外。デフォルトは全メロディ系チャンネル。melody_only=ch0 のみ。

※**手で聞いて確かめる用だけ**。曲づくりの伴奏はここを使わない——MIDI カバーの伴奏は
  `align_cover.accompaniment_for_section` が SoundFont（`sf2_render`）で鳴らす（2026-09-30 に
  ここにあった簡易合成の伴奏 `render_accompaniment_from_midi` を外した）。`SR` は曲づくりの各所が読む。
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import mido
import numpy as np

SR = 44100


def render_midi(path: Path, *, only_channel: int | None = None) -> np.ndarray:
    notes, active, t = [], {}, 0.0
    for msg in mido.MidiFile(str(path)):     # 反復は秒（テンポ考慮済み）
        t += msg.time
        ch = getattr(msg, "channel", 99)
        if msg.type == "note_on" and msg.velocity > 0:
            active[(ch, msg.note)] = t
        elif msg.type in ("note_off", "note_on") and (ch, msg.note) in active:
            s = active.pop((ch, msg.note))
            if ch == 9:                       # ドラムは除外
                continue
            if only_channel is not None and ch != only_channel:
                continue
            notes.append((s, t, msg.note))
    if not notes:
        return np.zeros(SR)
    total = int((max(e for _, e, _ in notes) + 0.4) * SR)
    buf = np.zeros(total)
    for s, e, m in notes:
        n = int((e - s) * SR)
        if n < 200:
            continue
        i0 = int(s * SR)
        f = 440.0 * 2 ** ((m - 69) / 12)
        tt = np.arange(n) / SR
        env = np.clip(tt / 0.01, 0, 1) * np.clip((n - np.arange(n)) / SR / 0.08, 0, 1)
        buf[i0:i0 + n] += 0.25 * env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt))
    peak = np.max(np.abs(buf)) or 1.0
    return (buf / peak * 0.85)


def write_wav(buf: np.ndarray, out: Path) -> None:
    pcm = (np.clip(buf, -1, 1) * 32767).astype("<i2")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "reference/hotaru.mid")
    only = None if (len(sys.argv) > 2 and sys.argv[2] == "full") else 0
    full = render_midi(src, only_channel=None)
    mel = render_midi(src, only_channel=0)
    write_wav(full, src.with_name(src.stem + "_full.wav"))
    write_wav(mel, src.with_name(src.stem + "_melody.wav"))
    print(f"書き出し: {src.with_name(src.stem + '_full.wav')}（全パート）")
    print(f"書き出し: {src.with_name(src.stem + '_melody.wav')}（メロディ ch0 のみ）")
