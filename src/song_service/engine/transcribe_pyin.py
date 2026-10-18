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

"""分離済みモノフォニック音声の採譜（pyin f0 → 中央値平滑 → 音価分割）。

basic-pitch はビブラートを半音刻みの別音に割ってしまうため、単音 f0 追跡 pyin で
連続ピッチを取り、中央値平滑→半音丸め→同音連結→短音除去でビブラートを1音に均す。
統合 venv(py3.13) で実行。出力 <input>.pyin.solfa.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter

_INV = {0: "ド", 1: "ド#", 2: "レ", 3: "レ#", 4: "ミ", 5: "ファ",
        6: "ファ#", 7: "ソ", 8: "ソ#", 9: "ラ", 10: "ラ#", 11: "シ"}


def midi_to_solfa(m: int) -> str:
    return f"{_INV[m % 12]}{m // 12 - 1}"


def _mode_filter(rounded: np.ndarray, win: int) -> np.ndarray:
    """各フレームを窓内の最頻値に置換（ビブラートの±半音うねりを中心音へ）。"""
    out = rounded.copy()
    half = win // 2
    n = len(rounded)
    for i in range(n):
        seg = rounded[max(0, i - half): min(n, i + half + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            vals, cnt = np.unique(seg, return_counts=True)
            out[i] = vals[cnt.argmax()]
    return out


def transcribe_pyin(path: Path, *, fmin: float = 130.0, fmax: float = 1000.0,
                    hop: int = 256, mode_win: int = 23, min_note: float = 0.13):
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    midi = np.where(voiced & ~np.isnan(midi), midi, np.nan)
    rounded = np.round(midi)
    # 最頻値フィルタでビブラートを中心音に丸める
    rounded = _mode_filter(rounded, mode_win)

    # 同音連結 → ノート化（短すぎる音は捨てる）
    raw, cur, start = [], None, 0.0
    for t, m in zip(times, rounded):
        val = None if np.isnan(m) else int(m)
        if val != cur:
            if cur is not None:
                raw.append((start, t, cur))
            cur, start = val, t
    if cur is not None:
        raw.append((start, times[-1], cur))

    # 短音マージ: <min_note のノートは前後の長いノートへ吸収（同音は連結）
    notes: list[tuple[float, float, int]] = []
    for s, e, m in raw:
        if e - s < min_note and notes:
            notes[-1] = (notes[-1][0], e, notes[-1][2])  # 前へ吸収
        elif notes and notes[-1][2] == m:
            notes[-1] = (notes[-1][0], e, m)
        else:
            notes.append((s, e, m))
    return [(s, e, m) for s, e, m in notes if e - s >= min_note]


def transcribe_onsets(path: Path, *, fmin: float = 130.0, fmax: float = 1000.0,
                      hop: int = 256, min_note: float = 0.10):
    """振幅オンセット検出で『発声タイミング』を実測し、各区間のピッチを pyin で推定。

    pyin の“ピッチ変化で区切る”方式と違い、音節の立ち上がり（振幅トランジェント）に音符が揃う。
    戻り値 [(start秒, end秒, midi), ...]。
    """
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    midi = np.where(voiced & ~np.isnan(midi), midi, np.nan)
    # 振幅オンセット（backtrack=ローカル極小へ寄せて音節頭に合わせる）
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True, units="time")
    bounds = list(onsets) + [float(times[-1])]
    notes: list[tuple[float, float, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < min_note:
            continue
        mask = (times >= a) & (times < b)
        seg, seg_t = midi[mask], times[mask]
        vmask = ~np.isnan(seg)
        if vmask.sum() < 2:                                 # ほぼ無声（休符/子音/間奏）→音にしない
            continue
        # 終端は『実際に声がある最後』まで（間奏で次オンセットまで伸ばさない＝真のギャップを残す）
        end = min(b, float(seg_t[vmask][-1]) + 0.06)
        vals, cnt = np.unique(np.round(seg[vmask]), return_counts=True)  # 区間の最頻音高
        notes.append((float(a), end, int(vals[cnt.argmax()])))
    return notes


def main() -> int:
    audio = Path(sys.argv[1] if len(sys.argv) > 1 else "reference/momotaro_vocal.vocals.wav")
    notes = transcribe_pyin(audio)
    solfa = [midi_to_solfa(m) for _, _, m in notes]
    side = audio.with_suffix(".pyin.solfa.json")
    side.write_text(json.dumps({"audio": audio.name, "solfa": solfa,
                                "midi": [m for _, _, m in notes]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"採譜ノート数: {len(notes)}  (階名: {side})")
    print("階名列:")
    print("  " + " ".join(solfa))
    print("\n時刻つき（先頭40音）:")
    for s, e, m in notes[:40]:
        print(f"  {s:6.2f}-{e:6.2f}  {midi_to_solfa(m)} (midi {m})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
