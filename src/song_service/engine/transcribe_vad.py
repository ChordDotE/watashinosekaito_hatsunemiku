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

"""faster-whisper で分離ボーカルを文字起こし → 各語の (word, start, end)。統合 venv(py3.13) で実行。

**vad_filter=True 必須**: 既定（VADなし）は伴奏/不明瞭部に「作詞作曲編曲…」等のクレジット文言を**幻覚**する。
VADで非音声を除くと幻覚が消え、実際に歌われた語と時刻だけが残る（区間特定/タイミング再整列の語時刻に使う）。
引数: <vocal.wav>  出力 stdout: [{"word","start","end"}, ...] の JSON 行（秒はファイル先頭=0基準）。
"""
from __future__ import annotations

import json
import sys

from faster_whisper import WhisperModel


def _run(wav_path: str, language: str, device: str, ct: str) -> list[dict]:
    model = WhisperModel("large-v3", device=device, compute_type=ct)
    segs, _ = model.transcribe(wav_path, language=language, word_timestamps=True,
                               vad_filter=True, condition_on_previous_text=False)  # ★幻覚防止
    out = []
    for s in segs:                       # ★生成器なので、実際の計算はここで初めて走る
        for w in (s.words or []):
            t = (w.word or "").strip()
            if t:
                out.append({"word": t, "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3)})
    return out


def transcribe_vad(wav_path: str, *, language: str = "ja") -> list[dict]:
    """GPU があれば使い、**使えなければ CPU でやり直す**。

    ★`torch.cuda.is_available()` だけで device を決めてはいけない（2026-09-23 に実害）。
    faster-whisper の実体は **ctranslate2** で、CUDA を使うには cuBLAS/cuDNN の DLL が要る。
    それらは torch の同梱物とは**別配布**なので、`torch` が「CUDA あり」と答えても
    `RuntimeError: Library cublas64_12.dll is not found` で落ちることがある
    （この機械で実際に発生。`ctranslate2.get_cuda_device_count()` も 1 を返すので、
    **数えるだけでは判定できない**——走らせて初めて分かる）。
    配布先は GPU 無しが前提なので、落ちたら黙って CPU へ下りる。
    """
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if has_cuda:
        try:
            return _run(wav_path, language, "cuda", "float16")
        except (RuntimeError, OSError) as ex:
            print(f"[transcribe_vad] GPU で動かせなかったので CPU でやり直す"
                  f"（{type(ex).__name__}: {ex}）", file=sys.stderr)
    return _run(wav_path, language, "cpu", "int8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(transcribe_vad(sys.argv[1]), ensure_ascii=False))
