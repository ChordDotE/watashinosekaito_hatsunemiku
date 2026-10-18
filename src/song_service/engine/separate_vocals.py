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

"""音源分離（demucs）でボーカル stem ＋ インスト(伴奏)stem を抽出 → soundfile で WAV 保存。

torchaudio 2.11 の save は torchcodec 必須になったため、読み込み=librosa / 保存=soundfile で回避。
htdemucs は drums/bass/other/vocals の4音源に分離。vocals=採譜用、それ以外の合算=伴奏の忠実コピー用。
出力: <input>.vocals.wav / <input>.accomp.wav。統合 venv(py3.13) で実行。
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def separate_vocals(in_path: Path, out_path: Path | None = None, *, save_stems: bool = True) -> tuple[Path, Path]:
    """(vocals.wav, accomp.wav) を返す。accomp = vocals 以外（drums+bass+other）の合算＝実楽器の伴奏。
    save_stems=True なら **drums/bass/other/vocals の各stemも `<input>.<stem>.wav` で保存**
    （stem別採譜＝楽器ごとの得意手法で採譜するため。参照素材として source/ に残る）。"""
    out_path = out_path or in_path.with_suffix(".vocals.wav")
    accomp_path = in_path.with_suffix(".accomp.wav")
    device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU があれば自動で使う（大幅高速化）
    print(f"[demucs] device={device}")
    model = get_model("htdemucs")
    model.eval()
    sr = model.samplerate

    audio, _ = librosa.load(str(in_path), sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    wav = torch.tensor(audio, dtype=torch.float32)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav = (wav - mean) / std

    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, progress=True)[0]
    sources = sources * std + mean
    vi = model.sources.index("vocals")
    vocals = sources[vi]                                   # (channels, n)
    accomp = sources.sum(0) - vocals                       # vocals 以外の合算＝伴奏（実楽器）
    sf.write(str(out_path), vocals.T.numpy(), sr)
    sf.write(str(accomp_path), accomp.T.numpy(), sr)
    if save_stems:                                         # 各楽器stemも保存（stem別採譜の素材）
        for i, name in enumerate(model.sources):
            sf.write(str(in_path.with_suffix(f".{name}.wav")), sources[i].T.numpy(), sr)
    return out_path, accomp_path


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "reference/momotaro_vocal.mp3")
    voc, acc = separate_vocals(src)
    print(f"分離完了: voc={voc} accomp={acc}")
