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

"""
Fish Audio TTS ストリーマー (tts_service_fish_audio.py)

Flask サービスとしては起動しない。
gateway_service.py から import して FishAudioStreamer クラスを使う。

設計書: miku/src/plan/fish_audio_pcm_streaming.md

使い方:
    from tts_service_fish_audio import FishAudioStreamer

    streamer = FishAudioStreamer()
    streamer.stream(
        text_queue=q,       # queue.Queue。None を入れると終端。
        on_chunk=callback,  # bytes (Int16 PCM) を受け取るコールバック
        on_finish=done,     # 完了時コールバック
    )
"""

import json
import queue
import traceback
from pathlib import Path
from typing import Callable, Optional

from fish_audio_sdk import WebSocketSession, TTSRequest

# ---------------------------------------------------------------------------
# 設定読み込み
# ---------------------------------------------------------------------------

def _load_fish_settings() -> dict:
    """settings.json から tts.fish_audio セクション（＋api_keys.fish_audio）を返す。"""
    settings_path = Path(__file__).parent / "settings.json"
    try:
        with open(settings_path, "r", encoding="utf-8-sig") as f:   # BOM 付きでも読む
            settings = json.load(f)
        cfg = dict(settings.get("tts", {}).get("fish_audio", {}))
        cfg["api_key"] = settings.get("api_keys", {}).get("fish_audio", "")
        return cfg
    except Exception as e:
        print(f"[FishAudio] settings.json 読み込みエラー: {e}")
        return {}

_cfg = _load_fish_settings()

FISH_API_KEY: str = _cfg.get("api_key", "")
FISH_VOICE_ID: str = _cfg.get("voice_id", "")
FISH_SAMPLE_RATE: int = int(_cfg.get("sample_rate", 44100))
FISH_LATENCY: str = _cfg.get("latency", "normal")
FISH_BACKEND: str = _cfg.get("backend", "speech-1.5")
# 声の調子の加工（話速・低域/高域・音量）。**既定で有効**（2026-09-01 ユーザー判断）。
# 設定で false にすれば従来と完全に同じ動作に戻る（加工を通さない素通し）。
# ここを既定 True にしているのは、**キーを持たない古い settings.json のまま更新した人**にも
# 届かせるため。配布の雛形（settings_sample.json）も true にしてある
FISH_TONE_CONTROL: bool = bool(_cfg.get("tone_control", True))

# Fish Audio は1チャンクで10秒超のPCMを返すことがあり、ブラウザのリングバッファを
# 溢れさせる。0.25秒単位に再分割して送ることで、プレイヤー側のプログレッシブ
# フィードがバッファ残量に応じてペーシングできるようにする。
# Int16 mono なので 2バイト境界に揃える（偶数）。
_MAX_EMIT_BYTES: int = (FISH_SAMPLE_RATE // 4) * 2  # 0.25秒分


# ---------------------------------------------------------------------------
# FishAudioStreamer
# ---------------------------------------------------------------------------

class FishAudioStreamer:
    """
    Fish Audio WebSocket 経由でテキストを PCM にストリーム変換する。

    使い方:
        q = queue.Queue()
        streamer = FishAudioStreamer()

        # 別スレッドでストリームを開始
        t = threading.Thread(target=streamer.stream, args=(q, on_chunk, on_finish))
        t.start()

        # テキストを push（何回でも可）
        q.put("こんにちは")
        q.put("今日はいい天気ですね。")

        # 終端シグナルを送ってストリームを閉じる
        q.put(None)
        t.join()

    on_chunk(pcm: bytes):
        Int16 PCM バイト列（44100Hz, mono）を受け取るコールバック。
        Fish Audio から受信するたびに呼ばれる。

    on_finish():
        Fish Audio ストリーム終了（正常・エラー問わず）後に必ず呼ばれる。
    """

    def stream(
        self,
        text_queue: queue.Queue,
        on_chunk: Callable[[bytes], None],
        on_finish: Callable[[], None],
        cancel_event: Optional[object] = None,  # threading.Event
        tone_holder: Optional[object] = None,   # utils.voice_tone.ToneHolder
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        """
        別スレッドで実行する。text_queue から文字列を取り出して Fish Audio に流し、
        返ってくる PCM チャンクを on_chunk へ渡す。

        Args:
            text_queue:   文字列を入れるキュー。None を入れると終端。
            on_chunk:     PCM チャンク（bytes）受信コールバック。
            on_finish:    完了（正常/エラー問わず）コールバック。
            cancel_event: threading.Event。set() されたら emit をスキップする（切断時用）。
            tone_holder:  声の調子を保持する箱。渡すと on_chunk の直前で加工する。
                          **このメソッドはテキストが届く前に起動される**ので、
                          値ではなく箱を受け取る（あとから差し替えられる）。
            on_error:     失敗を**呼び出し元へ運ぶ**コールバック（2026-09-20 追加）。
                          ★ここは別スレッドで、以前は例外を印字して終わっていた。
                          その結果、**Fish の残高が切れると「文字は出るのに声だけ出ない」**
                          状態になり、理由がどこにも出なかった。
        """
        if not FISH_API_KEY or not FISH_VOICE_ID:
            print("[FishAudio] APIキーまたはVoiceIDが設定されていません。settings.json を確認してください。")
            if on_error is not None:
                try:
                    on_error(RuntimeError("APIキーまたは voice_id が設定されていません。"
                                          "スタート画面の「設定」タブで入れてください。"))
                except Exception as cb_err:
                    print(f"[FishAudio] on_error 呼び出しに失敗: {cb_err}")
            on_finish()
            return

        _text_chunk_count = 0
        _text_total_chars = 0

        # 声の調子の加工。無効・箱なしのときは on_chunk をそのまま使う（従来と同じ経路）
        tone_stream = None
        if FISH_TONE_CONTROL and tone_holder is not None:
            try:
                from utils.voice_tone import ToneStream
                tone_stream = ToneStream(FISH_SAMPLE_RATE, tone_holder)
            except Exception as e:
                print(f"[FishAudio] 声の調子の加工を初期化できませんでした（無加工で続行）: {e}")

        def emit(pcm: bytes) -> None:
            if tone_stream is None:
                on_chunk(pcm)
                return
            try:
                y = tone_stream.feed(pcm)
            except Exception as e:
                print(f"[FishAudio] 声の調子の加工に失敗（無加工で続行）: {e}")
                on_chunk(pcm)
                return
            if y:
                on_chunk(y)

        def text_generator():
            nonlocal _text_chunk_count, _text_total_chars
            while True:
                try:
                    text = text_queue.get(timeout=60)
                except queue.Empty:
                    print(f"[FishAudio] text_queue タイムアウト（60s）。ストリームを終了します。"
                          f" (配信済み: {_text_chunk_count}チャンク / {_text_total_chars}字)")
                    return
                if text is None:
                    print(f"[FishAudio] text_generator 終了: {_text_chunk_count}チャンク / {_text_total_chars}字を配信")
                    return
                _text_chunk_count += 1
                _text_total_chars += len(text)
                preview = text[:40].replace('\n', '↵')
                print(f"[FishAudio] text_generator [{_text_chunk_count}] yield: {len(text)}字 「{preview}」")
                yield text

        try:
            print(f"[FishAudio] ストリーム開始 (latency={FISH_LATENCY}, backend={FISH_BACKEND})")
            _total_pcm_bytes = 0
            _raw_chunk_count = 0
            with WebSocketSession(FISH_API_KEY) as ws:
                for pcm_chunk in ws.tts(
                    TTSRequest(
                        text="",
                        reference_id=FISH_VOICE_ID,
                        format="pcm",
                        sample_rate=FISH_SAMPLE_RATE,
                        latency=FISH_LATENCY,
                    ),
                    text_stream=text_generator(),
                    backend=FISH_BACKEND,
                ):
                    if cancel_event is not None and cancel_event.is_set():
                        print("[FishAudio] キャンセルシグナルを検出。emit をスキップして終了します。")
                        break
                    _raw_chunk_count += 1
                    _total_pcm_bytes += len(pcm_chunk)
                    # 大きいチャンクは _MAX_EMIT_BYTES 単位に再分割して送る
                    if len(pcm_chunk) > _MAX_EMIT_BYTES:
                        for i in range(0, len(pcm_chunk), _MAX_EMIT_BYTES):
                            if cancel_event is not None and cancel_event.is_set():
                                break
                            emit(pcm_chunk[i:i + _MAX_EMIT_BYTES])
                    else:
                        emit(pcm_chunk)

            # 加工の重ね合わせに残ったぶんを出し切る（末尾が切れないように）
            if tone_stream is not None and not (
                    cancel_event is not None and cancel_event.is_set()):
                try:
                    tail = tone_stream.flush()
                    if tail:
                        on_chunk(tail)
                except Exception as e:
                    print(f"[FishAudio] 声の調子の後始末に失敗: {e}")

            print(f"[FishAudio] ストリーム正常終了: 合計 {_total_pcm_bytes}bytes"
                  f" / {_total_pcm_bytes/2/FISH_SAMPLE_RATE:.2f}秒 (rawチャンク {_raw_chunk_count}個)")

        except Exception as e:
            print(f"[FishAudio] ストリームエラー: {e}")
            traceback.print_exc()
            if on_error is not None:
                try:
                    on_error(e)
                except Exception as cb_err:
                    print(f"[FishAudio] on_error 呼び出しに失敗: {cb_err}")
        finally:
            on_finish()
