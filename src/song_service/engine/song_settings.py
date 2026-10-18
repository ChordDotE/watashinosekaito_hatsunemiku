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

"""**曲づくりが設定（settings.json）を読む入口**——ここ1か所（2026-10-01）。

以前は `song_llm`（モデル・API キー）と `song_runner`（実行時間の上限・KV と本体のポート・自己診断）の5か所が、
それぞれ `MIKU_SETTINGS` か標準の置き場から settings.json を直接読んでいた。

## 検証用の重ね書き（`MIKU_SETTINGS_OVERLAY`）

モデルを替えて試すとき（例: 曲づくりだけ Gemini 3.8 Flash）、以前は本番の settings.json を**丸ごと写して**
書き換え、`MIKU_SETTINGS` で渡していた。写しには **API キーが入る**——終わったら消すが、途中で落ちると残る
（本人「検証用にセッティング JSON のダミーを作ってそこから読み込むようにできる？」）。

いまは**変えたい所だけを書いた JSON**（`dev_tools/settings_overlays/*.json`）を、環境変数
`MIKU_SETTINGS_OVERLAY` で渡す。本番の設定に**重ねて**読む（辞書は入れ子まで重ね、それ以外は上書き）。

- 重ね書きに `api_keys` を書いたら**読まずに止める**——重ね書きは git に入れて使い回すもので、キーを置く場所ではない
- 本番の settings.json は書き換えない。キーの写しも作らない
- `MIKU_SETTINGS`（丸ごと差し替え）はそのまま使える（配布先で設定の置き場を変える用途）。重ね書きはその上に重なる

## 曲づくりのモデル（`song_service.pro_model` / `flash_model` / `lite_model`・2026-10-03）

曲づくりは本体の会話と別のモデルで動かす（本人決定: 曲づくりは Gemini 3.8 Flash）。モデルの段を読むのは `models()` の1か所で、
`song_llm` の `_PRO` / `_FLASH` / `_LITE`（`pick_model` の pro/flash/lite・コーディネーター・点検役・検索）は全部ここから来る。
キーが無い古い設定では、以前どおり本体の `llm.pro_model` / `llm.flash_model` / `llm.lite_model` を読む（`llm.filler_model` は 2026-10-10 に `lite_model` へ改名。本体の起動時に寄せられる）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

#: 標準の置き場（`miku/src/settings.json`。engine/ から2階層上）
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "settings.json"
OVERLAY_ENV = "MIKU_SETTINGS_OVERLAY"


def base_path() -> Path:
    """本番の設定の置き場（`MIKU_SETTINGS` があればそれ）。"""
    return Path(os.environ.get("MIKU_SETTINGS") or DEFAULT_PATH)


def overlay_path() -> Path | None:
    """重ね書きの置き場（`MIKU_SETTINGS_OVERLAY`。無ければ None）。"""
    v = os.environ.get(OVERLAY_ENV)
    return Path(v) if v else None


def merge(base: dict, over: dict) -> dict:
    """`over` を `base` に重ねた新しい辞書（辞書は入れ子まで重ね、それ以外は `over` が勝つ）。`base` は変えない。"""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))   # BOM 付きでも読む（2026-09-25）


def load() -> dict:
    """本番の設定に重ね書きを重ねたもの。本番の設定が読めなければ例外（黙って空にしない）。"""
    data = _read(base_path())
    op = overlay_path()
    if op is not None:
        over = _read(op)
        if "api_keys" in over:
            raise ValueError(f"重ね書き {op} に api_keys がある——キーは本番の settings.json だけに置く（重ね書きは git に入れて使い回す）")
        data = merge(data, over)
    return data


#: 曲づくりのモデルの段 → （`song_service` のキー, キーが無いときに読む `llm` のキー）
MODEL_KEYS = {"pro": ("pro_model", "pro_model"),
              "flash": ("flash_model", "flash_model"),
              "lite": ("lite_model", "lite_model")}


def models(data: dict) -> dict[str, str]:
    """曲づくりが使うモデル `{"pro", "flash", "lite"}`。段ごとに `song_service.<段>_model` → `llm.<旧キー>` の順に読む。

    pro と flash はどちらにも無ければ例外（黙って別のモデルで動かさない）。lite は以前と同じく、無ければ flash で代える。
    """
    song, llm = data.get("song_service") or {}, data.get("llm") or {}
    out: dict[str, str] = {}
    for tier, (song_key, llm_key) in MODEL_KEYS.items():
        v = song.get(song_key) or llm.get(llm_key)
        if not v and tier == "lite":
            v = out["flash"]
        if not v:
            raise KeyError(f"曲づくりのモデルが無い: song_service.{song_key} も llm.{llm_key} も設定されていない")
        out[tier] = v
    return out
