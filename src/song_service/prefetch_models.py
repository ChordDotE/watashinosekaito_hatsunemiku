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

"""学習済みモデルの明示プリフェッチ（setup_song_service.ps1 が統合venvで実行する）。

初回の曲作成中に数GBの暗黙ダウンロードが走るのを防ぐ（プロジェクト規約:
モデルの重みはセットアップで明示DL。実行中のDLは 30分 watchdog に殺されるリスクもある）。

キャッシュ先は環境変数 HF_HOME / TORCH_HOME に従う（setup_song_service.ps1 と
system_start.ps1 がともに miku\\runtime\\models\\ 配下を指すため、セットアップで
取得した実体をそのまま実行時に使う。ユーザープロファイルには何も残さない）。

何度実行しても取得済みの分はスキップされる（各ライブラリのキャッシュ判定に委ねる）。
使い方: venv\\Scripts\\python.exe prefetch_models.py
"""
from __future__ import annotations

import sys


def fetch_whisper() -> str:
    """**faster-whisper large-v3 だけ**を取る（約3.1GB）。

    ★**GPU が無い環境（`-OriginalOnly`）でもこれは取る**（2026-09-23）。
    以前は「文字起こしはカバー専用＝カバーは GPU 必須」だったので軽い導入から外していたが、
    **鼻歌の肉声から歌詞を採る**という **CPU だけで成立する使い道**ができた。前提が変わった。

    ★**実体は ctranslate2 で、`torch` は要らない**（torch を入れずに動くことを実測で確認済み）。
    ★**導入時に取る**（本人判断 2026-09-23）——「**初回時になぜか時間がかかるのは不自然**」。
    """
    from faster_whisper import download_model
    return str(download_model("large-v3"))


def main() -> int:
    failed: list[str] = []

    # 1) faster-whisper large-v3（採譜のASR/VAD・約3.1GB）。
    #    faster_whisper 自身のダウンローダを使う＝実行時と同じファイル選別・同じキャッシュ配置
    try:
        print(f"[prefetch] faster-whisper large-v3: {fetch_whisper()}")
    except Exception as e:
        failed.append(f"faster-whisper large-v3: {e}")

    # 2) whisperx 日本語整列モデル（forced_align の ja・約1.3GB）。
    #    whisperx は transformers 経由でロードするので同じ経路で取得する
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        name = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
        Wav2Vec2ForCTC.from_pretrained(name)
        Wav2Vec2Processor.from_pretrained(name)
        print(f"[prefetch] whisperx ja整列モデル: {name}")
    except Exception as e:
        failed.append(f"ja整列モデル: {e}")

    # 3) whisperx 英語整列モデル（torchaudio pipeline・約360MB。英語詞の en-FA 上書きで使用）
    try:
        import torchaudio
        torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        print("[prefetch] whisperx en整列モデル: WAV2VEC2_ASR_BASE_960H")
    except Exception as e:
        failed.append(f"en整列モデル: {e}")

    # 4) demucs htdemucs（音源分離・約320MB）
    try:
        from demucs.pretrained import get_model
        get_model("htdemucs")
        print("[prefetch] demucs htdemucs: OK")
    except Exception as e:
        failed.append(f"demucs htdemucs: {e}")

    if failed:
        for f in failed:
            print(f"[prefetch] 失敗: {f}", file=sys.stderr)
        return 1
    print("[prefetch] すべて取得済み")
    return 0


def main_whisper_only() -> int:
    """軽い導入（`-OriginalOnly`）から呼ぶ入口。**whisper だけ**取る。"""
    try:
        print(f"[prefetch] faster-whisper large-v3: {fetch_whisper()}")
        return 0
    except Exception as e:
        print(f"[prefetch] faster-whisper large-v3 を取れなかった: {e}")
        return 1


if __name__ == "__main__":
    # `--whisper-only`: 軽い導入（GPU 無し）向け。文字起こしのモデルだけ取る
    sys.exit(main_whisper_only() if "--whisper-only" in sys.argv[1:] else main())
