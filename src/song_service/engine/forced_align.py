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

"""WhisperX 強制アラインメント: 分離ボーカル＋既知のかな歌詞 → 各かなの発声時刻＋音高（実測）。

ASR の聞き取りに頼らず、**正しい歌詞（ネット取得）を音声に強制対応**させて各かなの時刻を得る。
音高は pyin f0 を各かな区間で中央値。統合 venv(py3.13) で実行。
引数: <vocal.wav> <text.txt(utf-8)>  出力 stdout: [{"char","start","end","midi"}, ...] の JSON 行。
"""
from __future__ import annotations

import json
import sys

import librosa
import numpy as np
import whisperx


def force_align_words(wav_path: str, text: str, *, language: str = "en",
                      device: str | None = None) -> list[dict]:
    """英語等の**単語単位**強制アラインメント。言語別 wav2vec2（en=英語モデル）で
    正しい原文テキストを整列し [{word,start,end}] を返す（日本語FAの前詰めを受けない実測時刻）。"""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    audio = whisperx.load_audio(wav_path)
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    segs = [{"text": text, "start": 0.0, "end": len(audio) / 16000.0}]
    res = whisperx.align(segs, model_a, metadata, audio, device, return_char_alignments=False)
    words = []
    for s in res["segments"]:
        for w in s.get("words", []):
            if w.get("start") is not None:
                words.append({"word": w["word"].strip(), "start": round(float(w["start"]), 3),
                              "end": round(float(w.get("end", w["start"])), 3)})
    return words


def force_align(wav_path: str, text: str, *, language: str = "ja", device: str | None = None) -> list[dict]:
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    audio = whisperx.load_audio(wav_path)                      # 16kHz mono
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    segs = [{"text": text, "start": 0.0, "end": len(audio) / 16000.0}]
    res = whisperx.align(segs, model_a, metadata, audio, device, return_char_alignments=True)
    chars = []
    for s in res["segments"]:
        for c in s.get("chars", []):
            if c.get("start") is not None and c.get("char", "").strip():
                chars.append({"char": c["char"], "start": round(float(c["start"]), 3),
                              "end": round(float(c.get("end", c["start"])), 3)})
    if not chars:
        return chars
    # f0（音高用）＋ 母音オンセット化(A)＋ 振幅オンセットスナップ(B) で発声位置を精緻化
    hop = 256
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=130.0, fmax=1000.0, sr=sr, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    midi = np.where(voiced & ~np.isnan(midi), midi, np.nan)
    amp = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True, units="time")
    prev = -1.0
    for c in chars:
        os, oe = c["start"], c["end"]
        seg = midi[(times >= os) & (times < oe + 0.08)]      # 音高＝区間のf0中央値
        seg = seg[~np.isnan(seg)]
        c["midi"] = int(round(float(np.median(seg)))) if seg.size else None
        # A: 母音の立ち上がり＝窓内で最初に有声(f0あり)になる時刻（子音ぶん後ろへ）
        vt = times[(times >= os - 0.02) & (times <= oe) & ~np.isnan(midi)]
        cand = float(vt[0]) if vt.size else os
        # B: 近傍(±0.06s)の実振幅オンセットへスナップ
        if amp.size:
            k = int(np.argmin(np.abs(amp - cand)))
            if abs(float(amp[k]) - cand) <= 0.06:
                cand = float(amp[k])
        cand = max(cand, prev + 0.02)                        # 単調維持
        c["start"], prev = round(cand, 3), cand
    return chars


if __name__ == "__main__":
    wav = sys.argv[1]
    text = open(sys.argv[2], encoding="utf-8").read().strip()
    lang = sys.argv[3] if len(sys.argv) > 3 else "ja"
    if lang == "ja":
        print(json.dumps(force_align(wav, text), ensure_ascii=False))
    else:                                                      # en等: 単語単位で出力
        print(json.dumps(force_align_words(wav, text, language=lang), ensure_ascii=False))
