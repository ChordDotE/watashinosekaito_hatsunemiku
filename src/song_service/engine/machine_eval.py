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

"""非LLMの機械評価: 旋律の参照一致度を決定論的に算出（DTW＋12移調正規化 / mir_eval）。

LLM官能評価は実行ごとにブレるが、これは入力が同じなら必ず同じ値を返す（決定論的）。
カバー精度（参照曲との一致）の測定に向く。move-do（移調不変）でピッチ列を比較する。
venv_transcribe(py3.10・mir_eval/numpy あり) で実行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def seq_from_score(p: Path) -> list[int]:
    return [n["key"] for n in json.loads(Path(p).read_text(encoding="utf-8"))["vocal_notes"]]


def seq_from_solfa(p: Path) -> list[int]:
    return json.loads(Path(p).read_text(encoding="utf-8"))["midi"]


def _dtw(a: list[int], b: list[int]) -> tuple[float, int]:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), 1e9)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m]), n + m


def best_match(ours: list[int], ref: list[int]) -> tuple[int, float]:
    """12通り移調を試し、平均音程誤差（半音）が最小の移調と誤差を返す。"""
    out = []
    for shift in range(-12, 13):
        dist, norm = _dtw(ours, [r + shift for r in ref])
        out.append((shift, dist / norm))
    return min(out, key=lambda x: x[1])


def similarity(ours: list[int], ref: list[int]) -> dict:
    shift, err = best_match(ours, ref)
    return {"best_transpose": shift, "avg_semitone_error": round(err, 3),
            "similarity_100": max(0, round(100 - err * 15))}


# ★DTW の対応の道（`_dtw_path`）は 2026-10-02 に外した。本番で使っていたのはカバーの音高補正（`align_cover`）が
#   頼んだ数と違う数を返したときの割り振りだけで、それを型の差し戻しにした（数が違えば直させ、直らなければ止める）ので呼び手が無くなった。
#   試作（`test/song_generation_poc/`）は自分の `machine_eval.py` の写しを使う。


if __name__ == "__main__":
    ours = seq_from_score(sys.argv[1] if len(sys.argv) > 1 else "output/20260619_proto_douyou/score.json")
    ref = seq_from_solfa(sys.argv[2] if len(sys.argv) > 2 else "reference/momotaro_vocal.vocals.pyin.solfa.json")
    r = similarity(ours, ref)
    print(f"自作 {len(ours)}音 vs 参照 {len(ref)}音")
    print(f"最適移調 {r['best_transpose']:+d}半音 / 平均音程誤差 {r['avg_semitone_error']}半音 / "
          f"類似度 {r['similarity_100']}/100")
