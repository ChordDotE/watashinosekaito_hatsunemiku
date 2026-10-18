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
Fish Audio のバッチ ASR（POST /v1/asr）を使うリアルタイム STT サービス（中継サーバー）。

WebSocket プロトコルを :5006 で提供する。Fish Audio にはストリーミング ASR が無いため、
サーバー側 RMS VAD で文末（無音区間）を検出し、発話 1 回分の PCM を WAV に
まとめてバッチ送信する方式を取る。
（もとはローカル faster-whisper 版 stt_service_realtime.py と同じプロトコルで作った。
 ローカル版は 2026-09-25 に削除した）

ブラウザ向けプロトコル:
  クライアント → サーバー:
    - バイナリ: PCM Int16, 16kHz mono
    - JSON: {"type": "stop"}
    - JSON: {"type": "set_threshold", "value": 0.03}  ← 本サービスでは有効（VAD はローカル）
  サーバー → クライアント:
    - {"type": "model_ready"}   ← 接続直後に即送信（モデルロード不要のため）
    - {"type": "interim", "text"}  ← 疑似 interim（下記）
    - {"type": "final", "text", "confidence", "audio_sec", "latency_ms"[, "turn_complete"]}
    - {"type": "error", "message"}

終話判定（Smart Turn v3・stt.smart_turn.enabled 時のみ）:
  VAD が final を切る瞬間、発話 PCM を Smart Turn v3（stt_smart_turn.py）にかけ、
  「言い終わった話し方か」を ASR と並行して判定する。結果は final の
  `turn_complete`（bool）で通知し、ブラウザは true のとき固定タイマーなしで即送信する。
  無効時は turn_complete フィールド自体を付けない（ブラウザは従来の5秒カウントダウン）。

疑似 interim:
  発話中、interim_interval_sec（既定 2.0s）ごとに「発話開始〜現時点のバッファ全体」を
  ASR にかけて interim として送る。毎回全体を認識し直すため途中経過は言い直されうる
  （partial 相当の暫定扱い）。前回の interim 呼び出しが未完了なら見送る。
  final 確定（バッファ世代の更新）後に返ってきた interim 結果は破棄する。
  会話中リアクション（相槌）は Gateway が interim をトリガーに発火するため、これで機能する。

制約:
  - interim の刻みは interim_interval_sec ＋ API 往復ぶん粗い
  - final は発話終了から「無音判定（既定 500ms）＋ API 往復（実測 0.8〜1.2s）」後に届く
  - keyterms（固有名詞バイアス）のような認識語彙の指定はできない
"""

import asyncio
import io
import json
import sys
import time
import traceback
import wave
from pathlib import Path

import httpx
import numpy as np
import ormsgpack

# ----------------------------------------
# 設定 (settings.json の services.stt.port / stt.fish_audio / api_keys.fish_audio で上書き可能)
# ----------------------------------------
DEFAULT_CONFIG = {
    "port":              5006,
    "host":              "127.0.0.1",  # 固定（設定では変えない。認証が無いので LAN に開かない・2026-10-08）
    "api_key":           "",     # 空なら settings.json の fish_audio.api_key（TTS 用）を流用
    "language":          "ja",
    "speech_threshold":  0.03,   # RMSエネルギー閾値 (0.0〜1.0)
    "silence_chunks":    5,      # 連続無音チャンク数でfinal判定 (100ms × 5 = 500ms)
    # 話者照合の前処理でテキストを区間へ割り当てるのに単語の時刻が要る。
    # 実測 2026-08-09: タイムスタンプ有効化のコストは +27〜43ms（4.8秒/20.8秒の音声）
    "ignore_timestamps": False,
    "max_utterance_sec": 60,     # これを超えたら強制フラッシュ（バッファ暴走防止）
    "flush_idle_sec":    1.0,    # 発話中にチャンクが途絶えたら文末とみなす秒数
    "interim_interval_sec": 2.0, # 疑似 interim の最小間隔（0 以下で無効）
}

_MIN_INTERIM_SEC = 0.5  # これ未満の音声では疑似 interim を出さない

SAMPLE_RATE = 16000
CHUNK_MS_ASSUMED = 100  # silence_chunks の換算基準（ブラウザは約100msごとに送信）
FISH_ASR_URL = "https://api.fish.audio/v1/asr"

REALTIME_PORT = DEFAULT_CONFIG["port"]
REALTIME_HOST = DEFAULT_CONFIG["host"]
_config: dict = dict(DEFAULT_CONFIG)
_smart_turn_analyzer = None  # run_stt_fish_service() で settings に従い初期化（無効なら None）


def load_config() -> dict:
    """settings.json の services.stt.port と stt.fish_audio を読み、
    DEFAULT_CONFIG とマージして返す。api_key は api_keys.fish_audio（TTS と共用）。
    """
    config = DEFAULT_CONFIG.copy()
    settings_path = Path(__file__).parent / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8-sig") as f:   # BOM 付きでも読む
                settings = json.load(f)
            stt_svc = settings.get("services", {}).get("stt", {})
            # host は読まない（127.0.0.1 固定。古い settings.json に host が残っていても LAN へ開かない）
            config.update({k: v for k, v in stt_svc.items() if k == "port"})
            fish_cfg = settings.get("stt", {}).get("fish_audio", {})
            config.update({k: v for k, v in fish_cfg.items() if k in config})
            config["api_key"] = settings.get("api_keys", {}).get("fish_audio", "")
            # 話者照合の呼び先（顔と同居の Recognition サービス。旧キーも読む）
            services = settings.get("services", {})
            rec = services.get("recognition") or services.get("face_recognition", {})
            config["recognition_port"] = rec.get("port", 5007)
        except Exception as e:
            print(f"[Fish STT] settings.json 読み込みエラー（デフォルト値で継続）: {e}")
    return config


async def analyze_speaker(pcm: bytes) -> dict | None:
    """発話の PCM を Recognition サービスへ送り、区間ごとの一致度を得る。

    声が未登録・サービス未起動なら None を返す（照合なしで会話は続く）。
    ASR と**並行**に呼ばれるため、ここでの待ちは ASR の裏に隠れる。
    """
    import base64
    port = _config.get("recognition_port", 5007)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/voice/analyze",
                json={"audio_b64": base64.b64encode(pcm_to_wav(pcm)).decode("ascii")},
                timeout=15.0,
            )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        # サービス未起動・タイムアウトは無視する（声の注釈が無いだけで会話は成立する）
        return None


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """PCM Int16 mono を WAV コンテナに包んで返す。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def chunk_rms(pcm: bytes) -> float:
    """PCM Int16 チャンクの RMS（0.0〜1.0）を返す。"""
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2)))


async def transcribe(client: httpx.AsyncClient, config: dict, pcm: bytes) -> str:
    """発話 1 回分の PCM を Fish Audio /v1/asr へバッチ送信し、認識テキストを返す。

    失敗時は例外を送出する（呼び出し側で error 通知に変換）。
    """
    text, _words = await transcribe_with_words(client, config, pcm)
    return text


async def transcribe_with_words(client: httpx.AsyncClient, config: dict,
                                pcm: bytes) -> tuple[str, list[dict]]:
    """認識テキストと単語タイムスタンプを返す。

    Fish が返す segments は**単語/形態素レベル**（実測 平均 0.23秒・最短 0.08）で、
    そのままでは話者照合できない（埋め込みが安定しない）。区間の境目は音声側の
    無音で決め、ここで得た時刻は「どのテキストがどの区間に属するか」の割り当てにだけ使う。
    """
    payload = {
        "audio":             pcm_to_wav(pcm),
        "language":          config["language"],
        "ignore_timestamps": bool(config["ignore_timestamps"]),
    }
    resp = await client.post(
        FISH_ASR_URL,
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type":  "application/msgpack",
        },
        content=ormsgpack.packb(payload),
        timeout=30.0,
    )
    resp.raise_for_status()
    body = resp.json()
    return (body.get("text") or "").strip(), (body.get("segments") or [])


def assign_words_to_segments(words: list[dict], segments: list[dict]) -> int:
    """単語を各区間へ割り当てる（区間の "text" を埋める）。戻り値は最近傍へ寄せた語数。

    Fish のタイムスタンプと音響区切りは別基準なので境界が数十〜数百ms ずれる。
    **語を区間に配るのではなく、区間へ語を1つずつ配る**ことで取りこぼしを無くす
    （実測 2026-08-09: 重なり50%方式だと長さゼロの語〔`18.08-18.08` のような語を
    Fish は返す〕が必ず落ち、97字中5字が消えた。中心時刻＋最近傍なら 0字）。
    """
    if not segments:
        return 0
    strays = 0
    for w in words:
        try:
            c = (float(w["start"]) + float(w["end"])) / 2
        except (KeyError, TypeError, ValueError):
            continue
        idx = next((i for i, s in enumerate(segments)
                    if s["start"] <= c <= s["end"]), None)
        if idx is None:
            idx = min(range(len(segments)),
                      key=lambda i: min(abs(c - segments[i]["start"]),
                                        abs(c - segments[i]["end"])))
            strays += 1
        segments[idx]["text"] = segments[idx].get("text", "") + (w.get("text") or "")
    return strays


class _UtteranceBuffer:
    """発話バッファと RMS VAD の状態管理（接続 1 本ごと）。

    ブラウザ側 VAD で発話区間のみ送信済みのため全チャンクをバッファに追加する。RMS は speech_detected（発話開始）と
    silence_ms（文末検出）にのみ使用する。
    """

    def __init__(self, config: dict):
        self.threshold = float(config["speech_threshold"])
        self.silence_limit_ms = int(config["silence_chunks"]) * CHUNK_MS_ASSUMED
        self.max_bytes = int(config["max_utterance_sec"]) * SAMPLE_RATE * 2
        self.generation = 0  # final 確定ごとに +1（遅延 interim 結果の破棄判定に使う）
        self.reset()

    def reset(self):
        self.chunks: list[bytes] = []
        self.total_bytes = 0
        self.speech_detected = False
        self.silence_ms = 0.0
        self.speech_start: float | None = None
        self.generation += 1

    def add(self, pcm: bytes) -> bool:
        """チャンクを追加し、文末（フラッシュすべき状態）なら True を返す。"""
        self.chunks.append(pcm)
        self.total_bytes += len(pcm)
        duration_ms = len(pcm) / 2 / SAMPLE_RATE * 1000
        rms = chunk_rms(pcm)
        if rms >= self.threshold:
            if not self.speech_detected:
                self.speech_detected = True
                self.speech_start = time.time()
                print(f"  [VAD] 音声開始 RMS={rms:.4f}  threshold={self.threshold:.3f}")
            self.silence_ms = 0.0
        elif self.speech_detected:
            self.silence_ms += duration_ms
        if self.total_bytes >= self.max_bytes:
            print(f"  [VAD] 最大発話長 {self.max_bytes // (SAMPLE_RATE * 2)}s 到達 → 強制フラッシュ")
            return True
        return self.speech_detected and self.silence_ms >= self.silence_limit_ms

    @property
    def audio_sec(self) -> float:
        return self.total_bytes / (SAMPLE_RATE * 2)

    def snapshot(self) -> bytes:
        """現時点のバッファ内容を（リセットせずに）結合して返す。疑似 interim 用。"""
        return b"".join(self.chunks)

    def take(self) -> bytes:
        """バッファ内容を取り出してリセットする。"""
        pcm = b"".join(self.chunks)
        self.reset()
        return pcm


async def _flush(websocket, client: httpx.AsyncClient, config: dict,
                 buffer: _UtteranceBuffer, smart_session=None):
    """バッファの発話を ASR にかけ、final をクライアントへ送る。

    smart_session（SmartTurnSession）があれば、終話判定を ASR と**並行**して実行し、
    final に turn_complete を付与する（判定十数ms << ASR 往復なので追加遅延ゼロ）。
    """
    audio_sec = buffer.audio_sec
    speech_start = buffer.speech_start
    pcm = buffer.take()
    if not pcm:
        return

    turn_task = None
    if smart_session is not None:
        loop = asyncio.get_running_loop()
        turn_task = loop.run_in_executor(None, smart_session.predict_pcm16, pcm)

    # 話者照合も ASR と並行に走らせる（PCM だけで完結し ASR 結果を待たない）。
    # Smart Turn と同じ並行の作法。実測 2026-08-09: ASR の裏に隠れて追加は +3〜12ms
    voice_task = asyncio.create_task(analyze_speaker(pcm))

    try:
        text, words = await transcribe_with_words(client, config, pcm)
    except Exception as e:
        voice_task.cancel()
        print(f"[Fish STT] ASR リクエスト失敗: {e}")
        await websocket.send(json.dumps(
            {"type": "error", "message": f"Fish Audio ASR エラー: {e}"}, ensure_ascii=False))
        return
    if not text:
        voice_task.cancel()
        print("  [final] （認識結果なし・スキップ）")
        return

    payload = {
        "type":       "final",
        "text":       text,
        "confidence": 1.0,  # API は信頼度を返さないため固定値
        "audio_sec":  round(audio_sec, 2),
        "latency_ms": 0.0,  # 下で上書き
    }

    # 声が登録済みなら、区間ごとの一致度にテキストを割り当てて final に載せる。
    # 未登録・サービス未起動なら何も付けない（従来どおりテキストだけの final）
    try:
        analyzed = await voice_task
    except Exception as e:
        analyzed = None
        print(f"[Fish STT] 話者照合に失敗（final は segments なしで送信）: {e}")
    if analyzed and analyzed.get("enrolled") and analyzed.get("segments"):
        segments = analyzed["segments"]
        strays = assign_words_to_segments(words, segments)
        # テキストが1文字も入らなかった区間は落とす（物音だけの区間を LLM に見せない）
        segments = [s for s in segments if (s.get("text") or "").strip()]
        if segments:
            payload["segments"] = segments
            print(f"  [voice] {len(segments)}区間 "
                  + " ".join(f"{s['start']:.1f}-{s['end']:.1f}s"
                             + ("--" if not s["scored"] else f"{s['score']:+.2f}"
                                + ("*" if s["borrowed"] else ""))
                             for s in segments)
                  + (f"  （最近傍へ寄せた語 {strays}）" if strays else ""))
    if turn_task is not None:
        try:
            result = await turn_task
            payload["turn_complete"] = result.complete
            print(f"  [SmartTurn] {'COMPLETE' if result.complete else 'INCOMPLETE'} "
                  f"p={result.probability:.3f} ({result.elapsed_ms:.0f}ms)")
        except Exception as e:
            # 判定失敗時はフィールドを付けない（ブラウザは従来の5秒カウントダウンで動く）
            print(f"[SmartTurn] 判定失敗（final は turn_complete なしで送信）: {e}")

    now = time.time()
    payload["latency_ms"] = round((now - (speech_start or now)) * 1000, 1)
    await websocket.send(json.dumps(payload, ensure_ascii=False))
    print(f"  [final] '{text}'")


# ----------------------------------------
# クライアント処理
# ----------------------------------------
async def handle_client(websocket):
    client_addr = getattr(websocket, "remote_address", "?")
    print(f"[接続] {client_addr}")

    # クラウド API のためモデルロード不要 → 即 ready を通知（ウォームアップ互換）
    await websocket.send(json.dumps({"type": "model_ready"}))

    buffer = _UtteranceBuffer(_config)
    # 終話判定セッション（接続ごと。INCOMPLETE 継続バッファを他接続と混ぜない）
    smart_session = _smart_turn_analyzer.new_session() if _smart_turn_analyzer else None
    flush_idle = float(_config["flush_idle_sec"])
    interim_interval = float(_config["interim_interval_sec"])
    interim_task: asyncio.Task | None = None
    interim_last_ts = 0.0

    async def _interim_job(client: httpx.AsyncClient, pcm: bytes, generation: int):
        """発話途中のバッファ全体を ASR にかけ、interim として送る。
        final 確定済み（世代不一致）なら結果を破棄する。エラーは interim なので握りつぶす。
        """
        try:
            text = await transcribe(client, _config, pcm)
        except Exception as e:
            print(f"[Fish STT] interim ASR 失敗（無視）: {e}")
            return
        if generation != buffer.generation or not text:
            return
        try:
            await websocket.send(json.dumps({"type": "interim", "text": text},
                                            ensure_ascii=False))
            print(f"  [interim] '{text}'")
        except Exception:
            pass  # クライアント切断済み

    def _maybe_start_interim(client: httpx.AsyncClient):
        """条件（発話中・間隔経過・前回完了・最小長）を満たせば疑似 interim を起動する。"""
        nonlocal interim_task, interim_last_ts
        if interim_interval <= 0 or not buffer.speech_detected:
            return
        if interim_task is not None and not interim_task.done():
            return
        now = time.monotonic()
        if now - interim_last_ts < interim_interval:
            return
        if buffer.audio_sec < _MIN_INTERIM_SEC:
            return
        interim_last_ts = now
        interim_task = asyncio.create_task(
            _interim_job(client, buffer.snapshot(), buffer.generation))

    async with httpx.AsyncClient() as client:
        try:
            while True:
                # 発話中にチャンクが途絶えた場合も文末とみなす（ブラウザ側 VAD 停止対策）
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=flush_idle)
                except asyncio.TimeoutError:
                    if buffer.speech_detected:
                        print("  [VAD] 受信アイドル → フラッシュ")
                        await _flush(websocket, client, _config, buffer, smart_session)
                    continue

                if isinstance(message, (bytes, bytearray)):
                    if len(message) % 2 != 0:
                        await websocket.send(json.dumps(
                            {"type": "error", "message": "PCMバイト長が奇数です"}))
                        continue
                    if buffer.add(bytes(message)):
                        await _flush(websocket, client, _config, buffer, smart_session)
                    else:
                        _maybe_start_interim(client)
                elif isinstance(message, str):
                    try:
                        ctrl = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if ctrl.get("type") == "stop":
                        print("  [stop] 終了要求受信")
                        if buffer.speech_detected:
                            await _flush(websocket, client, _config, buffer, smart_session)
                        break
                    elif ctrl.get("type") == "set_threshold":
                        try:
                            buffer.threshold = float(ctrl.get("value", buffer.threshold))
                            print(f"  [threshold] → {buffer.threshold:.3f}")
                        except (TypeError, ValueError):
                            pass
        except Exception as e:
            import websockets.exceptions as _ws_exc
            if isinstance(e, (_ws_exc.ConnectionClosedOK, _ws_exc.ConnectionClosedError)):
                print(f"[接続終了] {client_addr}: {e}")
            else:
                print(f"[エラー] {client_addr}: {e}")
                traceback.print_exc()

    print(f"[切断] {client_addr}")


# ----------------------------------------
# サーバー起動
# ----------------------------------------
async def start_server():
    import websockets
    print(f"Fish Audio STT 中継サーバー起動: ws://{REALTIME_HOST}:{REALTIME_PORT}/ws")
    async with websockets.serve(handle_client, REALTIME_HOST, REALTIME_PORT):
        await asyncio.Future()  # run forever


def run_stt_fish_service(port: int, host: str = "127.0.0.1"):
    """
    Gateway の別スレッドから呼び出し可能なエントリーポイント。
    threading.Thread(target=run_stt_fish_service, args=(port, host)).start() で起動する。
    """
    global REALTIME_PORT, REALTIME_HOST, _config, _smart_turn_analyzer
    _config = load_config()
    if not _config.get("api_key"):
        # サイレント失敗禁止: キー未設定は起動時に明示エラーで停止する
        raise RuntimeError(
            "api_keys.fish_audio が settings.json に設定されていません")
    # 終話判定（enabled=false なら None。enabled なのにモデル欠損等なら明示エラーで停止）
    from stt_smart_turn import SmartTurnAnalyzer
    _smart_turn_analyzer = SmartTurnAnalyzer.from_settings()
    REALTIME_PORT = port
    REALTIME_HOST = host
    asyncio.run(start_server())


def main() -> None:
    """単体起動の入口。配布物では `.pyd` 化されるため、起動処理は関数に置く。"""
    cfg = load_config()
    try:
        run_stt_fish_service(cfg["port"], cfg["host"])
    except KeyboardInterrupt:
        print("\n終了")
        sys.exit(0)


if __name__ == "__main__":
    main()
