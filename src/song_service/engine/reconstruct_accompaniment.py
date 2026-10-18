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

"""音声カバーの伴奏を **YourMT3 で多楽器 MIDI に採譜** → FluidSynth（フリー音源）で鳴らす（`tools/creation._audio_accomp` が呼ぶ）。

★2026-10-03 に **DSP 版の下書き**（stem 別にオンセット・pyin・クロマで MIDI を組む `reconstruct_accompaniment()` と
`_drum_events`・`_bass_events`・`_chord_events`・`_build_midi`・`_triad` ほか）を消した（本人承認）。どこからも呼ばれておらず、
クロマとの照合が `chord_estimate` と同じ役割の3つ目の実装になりかけていた（組「カバーのコード」）。
"""
from __future__ import annotations

from pathlib import Path

from sf2_render import render_midi


def reconstruct_accompaniment_mt3(source_dir: Path, start_sec: float, end_sec: float,
                                  out_dir: Path, *, stem_prefix: str = "aud_0", gain: float = 0.7) -> dict:
    """**YourMT3（SOTA・最高精度）**で stem個別にポリフォニック採譜→多楽器MIDI→FluidSynth(フリー音源)レンダ。
    外部公開用＝原曲音源でなく"自作MIDI"の伴奏。YourMT3は本番venvで動かないので **venv_mt3 を subprocess** で呼ぶ。
    戻り: {midi, wav}。out_dir に accompaniment.mid と accompaniment.wav を書く。"""
    import os
    import subprocess
    here = Path(__file__).parent                           # engine/（transcribe_mt3.py の置き場）
    service = here.parent                                   # song_service/（venv等のデータ基準）
    mt3_py = service / "venv_mt3" / "Scripts" / "python.exe"
    if not mt3_py.exists():
        raise RuntimeError(f"venv_mt3 未構築: {mt3_py}（YourMT3用 Python3.10 venv が必要）")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi = out_dir / "accompaniment.mid"
    r = subprocess.run([str(mt3_py), str(here / "transcribe_mt3.py"), str(source_dir),
                        f"{start_sec:.4f}", f"{end_sec:.4f}", str(midi), stem_prefix],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=1200, cwd=str(service), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if not midi.exists():
        raise RuntimeError(f"YourMT3採譜失敗(rc={r.returncode}): {(r.stderr or r.stdout or '')[-400:]}")
    wav = render_midi(midi, out_dir / "accompaniment.wav", gain=gain)   # フリーGM SoundFont（MIKU_SOUNDFONT で差替可）
    info = [ln for ln in (r.stdout or "").splitlines() if ln.startswith(("per_stem", "OK"))]
    return {"midi": str(midi), "wav": str(wav), "info": " ".join(info)}
