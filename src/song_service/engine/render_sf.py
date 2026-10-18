# -*- coding: utf-8 -*-
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

"""**伴奏を実音源で鳴らす**。実体は本番の `engine/sf2_render.py`（FluidSynth + GM SoundFont）。

## このファイルが薄い理由（2026-08-22）

当初ここに FluidSynth の呼び出しを新規実装したが、**本番に `sf2_render.py` が既にあった**。
カバー経路（`reconstruct_accompaniment`）と伴奏編集（`accomp.py`）は前から実音で鳴っており、
**オリジナル曲だけが numpy 合成音のまま取り残されていた**のが問題の正体だった。

  ・FluidSynth 本体  … `song_service/tools_bin/fluidsynth/bin/fluidsynth.exe`（配置済み）
  ・SoundFont       … `song_service/tools_bin/GeneralUser-GS.sf2`（32MB・**無制限配布可**）
  ・差し替え        … `MIKU_FLUIDSYNTH` / `MIKU_SOUNDFONT` の環境変数

そこで**重複を捨て、本番の関数を呼ぶだけ**にした。ここに残る仕事は
「**オリジナル曲の編曲データ → MIDI**」（`midi_out.py`）だけで、それは本当に無かった部分。

    python render_sf.py out/v3/19_*.json      # 1曲だけ鳴らす
    python render_sf.py --all                 # 既存曲をまとめて鳴らし直す（LLM 費用ゼロ）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import song_design as compose_v2                                          # noqa: E402,F401（bootstrap）
from midi_out import score_to_midi                          # noqa: E402
from sf2_render import available, render_midi, soundfont    # noqa: E402（本番の実音源レンダラ）
from song_schema import SongScore                           # noqa: E402


def render(score: SongScore, out_wav: Path, gain: float = 0.8,
           warns: list[str] | None = None) -> Path:
    """伴奏を実音源で wav 化。歌は入れない（V6 が歌うので二重に鳴らさない）。
    主旋律を楽器で弾くパート（`SongScore.lead`・インストと歌ものの重ね）は伴奏の1トラックとして入る（`midi_out._lead_track`）。

    前提:
      1. `tools_bin/` に FluidSynth と SoundFont が配置済み（セットアップが行う）
      2. 歌は含めない——`score_to_midi(include_vocal=False)` を守ること
      3. 同じ場所に `.mid` も残る（**V6/Cubase で音色を差し替えられる**ようにするため）
    """
    # ★**アカペラは伴奏の音を作らない**（2026-10-03・§5-3）。伴奏の楽譜が無いので空の MIDI になり、FluidSynth は
    #   約2秒の無音の wav を書く（2026-10-03 に使い捨ての場所で確かめた）——それを伴奏として同梱・計測しないよう、呼ばれたら止める
    if getattr(score, "form", "song") == "acappella":
        raise ValueError("アカペラの曲は伴奏の音を作らない（書き出しの形の分かれ方・song_pipeline.compose）")
    if not available():
        raise RuntimeError("FluidSynth / SoundFont が見つからない（tools_bin を確認）")
    mid = out_wav.with_suffix(".mid")
    # `warns` は**奏法を掛けられなかった所見**（和音パート・楽器が持たない・重なり）。
    # ★**機械は証拠を並べ、判定は人**——落としたことを黙っていない。
    score_to_midi(score, mid, include_vocal=False, warns=warns)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    return render_midi(mid, out_wav, gain=gain)


def _load(p: Path) -> SongScore:
    return SongScore.model_validate(json.loads(p.read_text(encoding="utf-8")))
