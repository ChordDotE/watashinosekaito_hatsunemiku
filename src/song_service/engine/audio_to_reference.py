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

"""音声(MP3等) → 参照記譜（venv_transcribe・py3.10 で実行）。

demucs でボーカル分離 → pyin で採譜 → テンポ推定で秒→拍に変換 → MIDI参照と同形式の階名JSON。
統合 venv(py3.13) で実行（旧 venv_transcribe は廃止）。
出力: <audio>.solfa.json（solfa / midi / notes[[pitch,dur拍,start拍]] / phrase_breaks / beats_per_bar / tempo）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa

from separate_vocals import separate_vocals
from transcribe_pyin import midi_to_solfa, transcribe_onsets


def audio_to_reference(audio_path: str, *, beats_per_bar: int = 4, q: float = 0.125) -> dict:
    path = Path(audio_path)
    voc, accomp = separate_vocals(path)                          # 1. 音源分離（vocals＋伴奏インスト）
    notes = transcribe_onsets(voc)                               # 2. 採譜＝振幅オンセットで発声タイミング実測
    y, sr = librosa.load(str(path), sr=22050, mono=True)         # 3. テンポ推定
    import numpy as _np                                          # 新しい librosa は配列を返す→スカラ化
    _bt = _np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0]).ravel()
    tempo = float(_bt[0]) if _bt.size else 0.0
    if tempo < 40:
        tempo = 120.0
    bps = tempo / 60.0

    def qb(sec: float) -> float:
        return round((sec * bps) / q) * q

    midis, out_notes = [], []
    for s, e, m in notes:                                        # 4. 秒→拍・量子化
        st = qb(s)
        du = max(q, round(((e - s) * bps) / q) * q)
        midis.append(int(m))
        out_notes.append([int(m), du, st])
    breaks = [0]                                                 # 5. 休符でフレーズ境界
    for i in range(1, len(out_notes)):
        if out_notes[i][2] - (out_notes[i - 1][2] + out_notes[i - 1][1]) >= 0.75:
            breaks.append(i)
    out = path.with_suffix(".solfa.json")
    out.write_text(json.dumps({"audio": path.name, "tempo": round(tempo, 1),
                               "beats_per_bar": beats_per_bar,
                               "accompaniment_wav": str(accomp),   # 伴奏の忠実コピー用（実楽器）
                               "vocals_wav": str(voc),             # 採譜のLLM音声グラウンディング補正用
                               "solfa": [midi_to_solfa(m) for m in midis], "midi": midis,
                               "notes": out_notes, "phrase_breaks": breaks},
                              ensure_ascii=False), encoding="utf-8")
    return {"reference_json": str(out), "note_count": len(out_notes), "tempo": round(tempo, 1),
            "accompaniment_wav": str(accomp)}


if __name__ == "__main__":
    print(json.dumps(audio_to_reference(sys.argv[1]), ensure_ascii=False))
