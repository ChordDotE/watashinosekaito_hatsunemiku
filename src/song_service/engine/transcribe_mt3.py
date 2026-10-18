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

"""YourMT3（mt3-infer）で伴奏を stem個別にポリフォニック採譜→多楽器MIDI。**venv_mt3 で実行**。

本番(py3.13)からは subprocess で呼ぶ（transcribe_vad.py / forced_align.py と同じパターン）。
本番venvでは mt3-infer/YourMT3 が動かない（transformers不整合）ので、専用 venv_mt3(py3.10) に隔離。

使い方:
    venv_mt3/Scripts/python.exe transcribe_mt3.py <source_dir> <start_sec> <end_sec> <out_mid> [stem_prefix]
  ・source_dir に aud_0.{drums,bass,other}.wav（demucs分離stem）がある前提。
  ・区間[start_sec,end_sec]を各stemから切り、YourMT3でMIDI化して1つのMIDIに統合。
  ・stem個別＝混合一括より楽器が多彩に採れる（drums/bass/otherを別々に推論）。
最終行に "OK <総ノート数> <楽器数>" を出力（呼び出し側はこれで成否判定）。
"""
import sys
import tempfile
from pathlib import Path

import librosa
import pretty_midi

from mt3_infer import transcribe

STEMS = ("drums", "bass", "other")


def transcribe_stems(source_dir: str, start: float, end: float, out_mid: str,
                     stem_prefix: str = "aud_0") -> dict:
    src = Path(source_dir)
    comb = pretty_midi.PrettyMIDI()
    per = {}
    tmp = Path(tempfile.gettempdir()) / "_mt3_stem.mid"
    for stem in STEMS:
        p = src / f"{stem_prefix}.{stem}.wav"
        if not p.exists():
            continue
        y = librosa.load(str(p), sr=16000, mono=True, offset=start, duration=end - start)[0]
        midi = transcribe(y, sr=16000, model="yourmt3")       # 戻り値は mido MidiFile
        try:
            midi.save(str(tmp))
        except Exception:
            midi.write(str(tmp))
        pm = pretty_midi.PrettyMIDI(str(tmp))
        for ins in pm.instruments:                            # stem内の全楽器を統合MIDIへ
            comb.instruments.append(ins)
        per[stem] = sum(len(i.notes) for i in pm.instruments)
    comb.write(out_mid)
    return {"notes": sum(len(i.notes) for i in comb.instruments), "instruments": len(comb.instruments), "per_stem": per}


if __name__ == "__main__":
    source_dir, start, end, out_mid = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    prefix = sys.argv[5] if len(sys.argv) > 5 else "aud_0"
    r = transcribe_stems(source_dir, start, end, out_mid, prefix)
    print(f"per_stem={r['per_stem']}")
    print(f"OK {r['notes']} {r['instruments']}")
