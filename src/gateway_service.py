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
Gateway Service - メインサービス（リファクタリング版）

Features:
- テンプレートフォルダの静的ファイル配信
- Live2D distフォルダの静的ファイル配信
- WebSocket接続管理（Flask-SocketIO）
- REST API（/generate, /ping, /audio）
- Dify連携
- TTS連携（音声生成・配信）
- KVサービス起動
- 会話ログ処理
- 無応答リマインダー

リファクタリング後:
- Flask-Sockを削除（WebSocketはSocket.IOに統一）
- 音声再生はクライアント側（voiceQueue.js）で行う
- Live2D制御はpostMessage経由（ws_control.tsはpostMessageを受信）

Port: 5001
"""

import os
import sys
import json
import queue
import time
import asyncio
import threading
import traceback
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import re

if TYPE_CHECKING:
    from agent.turn_handle import TurnHandle

import requests as http_requests
from flask import (Flask, render_template, request, jsonify, send_file,
                   send_from_directory, session, redirect, url_for)
from flask_socketio import SocketIO, emit

# =============================================================================
# ディレクトリ設定
# =============================================================================

BASE_DIR = Path(__file__).parent  # src/
MIKU_DIR = BASE_DIR.parent        # miku/
TEMPLATES_DIR = BASE_DIR / "templates"
LIVE2D_DIST_DIR = BASE_DIR / "live2d" / "CubismWebSamples" / "Samples" / "TypeScript" / "Demo" / "dist"
# ファビコン。Live2D モデルから生成したもの（セットアップが作る）を優先し、
# 生成前・生成に失敗した場合は同梱の簡易マークを使う
GENERATED_ICON = MIKU_DIR / "runtime" / "miku_icon.ico"
BUNDLED_ICON = MIKU_DIR / "app_icon.ico"

# =============================================================================
# Flaskアプリケーションの初期化
# =============================================================================

# static_folder は持たない。唯一の中身だったキャラクター静止画は、Live2D 必須化にともない
# 廃止した（再配布できない素材を配布物から外す意図もある）。CSS/JS は専用ルートで返している
app = Flask(__name__,
            template_folder=str(TEMPLATES_DIR),
            static_folder=None)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# =============================================================================
# ログイン保護（2026-08-06 追加）
#
# パスワードを設定すれば保護ON・空ならOFF（有効/無効スイッチは持たない）。
# この PC 自身からのアクセスは常に免除（コントロールパネルが叩くため）。
# ロジックは utils/auth.py、パスワードのハッシュ検証は utils/settings_store.py。
# =============================================================================

from utils.auth import (  # noqa: E402
    LoginLimiter, get_or_create_session_key, is_local_request, is_same_origin,
)
from utils.settings_store import verify_password  # noqa: E402
from utils import log_archive  # noqa: E402

# セッション署名鍵は runtime に保存（再起動してもログイン状態を保つため）。
# 以前は全インストール共通の固定文字列だった＝誰でもセッションを偽造できた。
# settings.json に依存しないのでここで確定できる
app.config['SECRET_KEY'] = get_or_create_session_key(MIKU_DIR / "runtime")
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)   # 既定。設定読込後に上書き

# settings は initialize_app() が読み込むまで空なので、依存する設定は後から反映する
_login_limiter = LoginLimiter()

# 認証なしで通すエンドポイント（ログイン画面自体とファビコン）
_AUTH_EXEMPT_ENDPOINTS = {'login', 'favicon', 'static'}


def configure_auth() -> None:
    """settings.json 読込後に、セッション期間と失敗ロックの設定を反映する。

    initialize_app() から呼ぶ（モジュール読込時点では settings がまだ空のため）。
    """
    global _login_limiter
    cfg = (settings.get('security', {}) or {}).get('auth', {}) or {}
    hours = int(cfg.get('session_hours', 24) or 24)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=hours)
    _login_limiter = LoginLimiter(
        threshold=int(cfg.get('lock_threshold', 10) or 10),
        lock_minutes=int(cfg.get('lock_minutes', 5) or 5),
    )
    print(f"[auth] ログイン保護: {'有効' if _auth_password_hash() else '無効（パスワード未設定）'}"
          f" / セッション {hours} 時間")


def _auth_password_hash() -> str:
    """現在のパスワードハッシュ。空なら保護は無効。"""
    return ((settings.get('security', {}) or {}).get('auth', {}) or {}).get('password_hash', '') or ''


def _is_authenticated() -> bool:
    """このリクエストが認証済みか（保護OFF・ローカルからのアクセスも True）。"""
    if not _auth_password_hash():
        return True                     # パスワード未設定＝保護しない
    if is_local_request(request.remote_addr, request.headers):
        return True                     # この PC 自身（コントロールパネル等）
    return bool(session.get('authed'))


@app.before_request
def _reject_cross_site():
    """**他所のサイトから投げられた要求**を断る（2026-09-20 追加）。

    ★接続元アドレスだけを見る従来の判定では、**利用者が開いた別のサイト**から
    `http://localhost:5001` へ投げられた要求も「本人」と扱われていた。
    実測で、別オリジンのページから**ミクに話しかけ、返答を読み取れた**。

    状態を変える要求（POST/PUT/PATCH/DELETE）だけを見る。
    画面の取得（GET）は読み取りを伴わないので従来どおり
    （そもそも他所のサイトからは中身を読めない）。
    """
    if request.method in ('GET', 'HEAD', 'OPTIONS'):
        return None
    if is_same_origin(request.headers.get('Origin'), request.headers.get('Host')):
        return None
    print(f"[SECURITY] 別サイトからの要求を拒否: {request.path} "
          f"origin={request.headers.get('Origin')!r}")
    return jsonify({"success": False, "error": "この要求は受け付けられません"}), 403


@app.before_request
def _require_login():
    """未認証のアクセスをログイン画面へ送る（API/Socket.IO は 401 で返す）。"""
    if request.endpoint in _AUTH_EXEMPT_ENDPOINTS:
        return None
    if request.path.startswith('/socket.io'):
        return None                     # Socket.IO は接続ハンドラ側で検証する
    if _is_authenticated():
        return None
    # fetch / sendBeacon などは画面遷移できないので 401 を返す（ブラウザ側で扱う）
    if request.path.startswith('/api/') or request.path == '/generate':
        return jsonify({"success": False, "error": "認証が必要です"}), 401
    return redirect(url_for('login', next=request.path))


# Flask-SocketIOの初期化（統一WebSocket）
socketio = SocketIO(app, cors_allowed_origins="*")

# =============================================================================
# グローバル変数
# =============================================================================

# 接続中のSocket.IOクライアントを管理する辞書
connected_clients: Dict[str, dict] = {}

# WebSocketセッションIDとクライアントセッションIDのマッピング
session_mapping: Dict[str, str] = {}  # {socketio_session_id: client_session_id}

# リアルタイムSTTストリーミングセッション管理
# {socketio_sid: {'loop': asyncio.EventLoop, 'aio_queue': asyncio.Queue, 'thread': Thread}}
_stream_sessions: Dict[str, dict] = {}

# 音声チャンク受信の計測（2026-08-16 追加）。
# 「録音はできているのに文字起こしが返らない」を切り分けるには、Gateway に音声が
# 届いていたかを知る必要がある。届いた量を発話単位でまとめ、セッション不在で
# 捨てた分は**黙って捨てずに警告**する（捨てた事実が残らないのが従来の穴だった）。
# {socketio_sid: {'chunks': int, 'bytes': int, 'first': float, 'last': float}}
_chunk_stats: Dict[str, dict] = {}
# セッション不在で捨てたチャンク {socketio_sid: {'dropped': int, 'bytes': int, 'warned_at': float}}
_chunk_orphans: Dict[str, dict] = {}

# アクティブセッション管理
active_session_id: Optional[str] = None
last_activity_time: Optional[float] = None

# 自由行動モード（mode=2）管理
FREE_MODE_INTERVAL_SECONDS = 60* 60 *5  # 5時間
MAX_CONSECUTIVE_FREE_MODE_RUNS = 10  # 連続実行上限
FREE_MODE_QUIET_HOURS_START = "00:00"  # 夜間停止開始（HH:MM）
FREE_MODE_QUIET_HOURS_END = "07:00"    # 夜間停止終了（HH:MM）

free_mode_reset_event = threading.Event()  # ユーザー入力でタイマーをリセットするためのevent
last_user_input_time: Optional[float] = None  # 最後にユーザー入力（mode=0）を処理したUNIX時刻
free_mode_consecutive_runs = 0  # 直近で連続実行した自由行動（mode=2）の回数
free_mode_limit_logged = False  # 連続実行上限到達ログを1回だけ出すためのフラグ
free_mode_quiet_hours_logged = False  # 夜間停止中ログを1回だけ出すためのフラグ

# 自発話しかけタイマー（mode=3）管理
_proactive_timer: Optional[threading.Timer] = None

# 定期記憶更新スケジューラー管理
DAILY_MEMORY_UPDATE_HOUR = 4       # 毎日記憶更新を実行する時刻（時）
DAILY_MEMORY_UPDATE_MINUTE = 0     # 毎日記憶更新を実行する時刻（分）
DAILY_MEMORY_QUIET_MINUTES = 20    # 最後の会話からこの分数以内ならスキップ
last_memory_update_date: Optional[str] = None  # 最後に記憶更新を実行した日付（YYYY-MM-DD）

# ログアーカイブスケジューラー管理（記憶更新とは別枠にする。あちらは LLM を叩き、
# こちらは CPU を数秒占有するので、同じ枠で直列にすると所要時間が読めなくなる）
DAILY_LOG_ARCHIVE_HOUR = 4         # 毎日ログをアーカイブする時刻（時）
DAILY_LOG_ARCHIVE_MINUTE = 30      # 同（分）。記憶更新の 4:00 とずらす
last_log_archive_date: Optional[str] = None  # 最後にアーカイブを実行した日付（YYYY-MM-DD）

# 初期化状態
initialization_success = False  # 初期化成功フラグ（/ping, /generate のガードに使用）
path_config = None  # PathConfigインスタンス（初期化成功後に設定される）
settings: dict = {}  # settings.json の読込結果（各サービス設定参照用）

# STT ウォームアップ接続（ブラウザ接続中はSTTモデルをロードしておく共有WS接続）
_stt_warmup: Optional[dict] = None          # {'loop': ..., 'thread': ..., 'stop_event': ...}
_stt_warmup_lock = threading.Lock()
_stt_model_status = "unknown"               # "unknown" | "loading" | "ready"


# 会話中リアクションの emit 共通化。
# 4経路（STT interim / STT final / テキスト入力中 / 送信直後の filler）でクールダウンを
# sid 単位に共有し、経路をまたいだリアクションの連発を防ぐ。
_reaction_cooldown_until: Dict[str, float] = {}
_reaction_emit_lock = threading.Lock()


def _try_emit_reaction_motion(sid: Optional[str], motion: str, source: str) -> bool:
    """リアクションモーションをブラウザへ送る（クールダウン共有・全経路共通の出口）。

    Returns: 実際に emit したら True（呼び出し側の Gate 状態更新に使う）。
    """
    if not sid or not motion or motion == "none":
        return False
    reaction_config = settings.get('conversation_reaction', {})
    if not reaction_config.get('enabled', True):
        return False
    cooldown = float(reaction_config.get('cooldown_seconds', 5))
    now = time.monotonic()
    with _reaction_emit_lock:
        if now < _reaction_cooldown_until.get(sid, 0.0):
            return False
        _reaction_cooldown_until[sid] = now + cooldown
    socketio.emit("reaction_motion", {"motion": motion}, to=sid)
    print(f"[Reaction] motion={motion} を送信 (source={source})")
    return True


# Gateway 以外の内部サービス（KV・VDB・Recognition・STT・Agent Service）の待ち受けアドレス。
# どれも認証を持たないので、この PC の中からしか届かないようにする（2026-10-08）。
# 呼び出し元は全部この PC の中（Gateway・Agent・曲づくり・コントロールパネル）で、
# ブラウザ（スマホ含む）とトンネルは Gateway :5001 だけを通る。設定キーは無い（固定）
INTERNAL_HOST = "127.0.0.1"


def _get_active_stt_config() -> dict:
    """有効なリアルタイム STT の設定を返す。

    settings.stt.provider（none | fish_audio）で有効／無効を決める。
    :5006 の WebSocket プロトコルに揃えてあるため、プロキシ・ウォームアップ・ブラウザは本関数の返す
    port / enabled / auto_send だけ見ればよい。port は services.stt から読む
    （待ち受けアドレスは INTERNAL_HOST 固定）。
    """
    stt_cfg = settings.get('stt', {})
    stt_svc = settings.get('services', {}).get('stt', {})
    provider = stt_cfg.get('provider', 'none')
    return {
        'enabled':   provider != 'none',
        'provider':  provider,
        'port':      stt_svc.get('port', 5006),
        'auto_send': stt_cfg.get('auto_send', {}),
    }

# Fish Audio PCM ストリーミング キャンセル制御
_current_fish_cancel: Optional[threading.Event] = None
# ★**いま声を送っているターンの送り先 sid**（2026-10-14・本人了承）。切断した端末がこの sid のときだけ声を止める。
#   以前は**どの端末が切れても**声を止めたので、約10秒おきに再接続する別の端末がいると、長い返事の声が毎回消えた
#   （2026-10-14 212秒のターンで 0 バイト・「ブラウザから声が出なかった」の原因）
_current_fish_target_sid: Optional[str] = None

# 進行中の mode0 ターンへの「考え中の追加発話」注入ハンドル（P2 畳み込み・2026-08-01）。
# _run_fish_audio_flow がターン開始時に登録し、終了時に解除する。
# /generate は進行中ターンがあれば新規ターンを起動せず try_inject() で畳み込む。
_active_turn_handle = None  # agent.turn_handle.TurnHandle | None

# =============================================================================
# セッション管理
# =============================================================================

def set_active_session(session_id: str) -> None:
    """アクティブセッションを設定する関数"""
    global active_session_id, last_activity_time

    try:
        active_session_id = session_id
        last_activity_time = time.time()
        print(f"アクティブセッションを設定しました: {session_id}")
    except Exception as e:
        print(f"アクティブセッション設定エラー: {str(e)}")


def reset_free_mode_counter() -> None:
    """自由行動モードの連続実行カウンタをリセットする"""
    global free_mode_consecutive_runs, free_mode_limit_logged
    free_mode_consecutive_runs = 0
    free_mode_limit_logged = False


def is_in_free_mode_quiet_hours(current_time: Optional[time.struct_time] = None) -> bool:
    """自由行動モードの夜間停止時間帯かを判定"""
    current = current_time if current_time is not None else time.localtime()
    current_minutes = current.tm_hour * 60 + current.tm_min

    start_hour, start_min = map(int, FREE_MODE_QUIET_HOURS_START.split(':'))
    end_hour, end_min = map(int, FREE_MODE_QUIET_HOURS_END.split(':'))
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min

    # 例: 00:00-07:00（同日内）
    if start_minutes < end_minutes:
        return start_minutes <= current_minutes < end_minutes

    # 例: 23:45-07:00（日跨ぎ）
    if start_minutes > end_minutes:
        return current_minutes >= start_minutes or current_minutes < end_minutes

    # start == end の場合は24時間停止
    return True


def schedule_proactive(minutes: int) -> None:
    """次回の自発話しかけタイマーをセット（既存タイマーはキャンセルして上書き）"""
    global _proactive_timer
    print(f"⏰ schedule_proactive: minutes={minutes}")
    if _proactive_timer is not None:
        _proactive_timer.cancel()
        _proactive_timer = None
    if minutes > 0:
        _proactive_timer = threading.Timer(minutes * 60, trigger_proactive_action)
        _proactive_timer.daemon = True
        _proactive_timer.start()
        print(f"⏰ 自発話しかけタイマーをセット: {minutes}分後")
    else:
        print(f"⏰ 自発話しかけタイマーなし")


def trigger_proactive_action(trigger_note: str = "") -> None:
    """自発話しかけモード（mode=3）を実行する。

    ★**ブラウザが開いていれば、返事を声にしてブラウザへ届ける**（2026-10-09 本人判断）。
      届け方は無応答リマインダーと同じ（確定した全文を `inactivity_reminder` で字幕・モーションへ、
      声は chapter=100 で流す）。閉じていれば従来どおりブラウザへは何も送らず、連絡はメール等で
      エージェント側が完結させる。どちらにするかはミクが `<system_state>` のブラウザの行を見て決める
      （開いていれば response を書き、閉じていれば null にしてメール）。
      進行中のターンがあるときは、その声と重ねないよう終わるまで待つ（上限60秒）。

    trigger_note: このターンの**起点**の内容（例: 裏で進めていた作曲の完了）。
    空なら従来どおり時間経過による自発話しかけ。会話ログには起点=assistant として残る。"""
    from agent_main import process_agent_request

    print(f"⏰ 自発話しかけモード（mode=3）を実行します{'（起点: 裏作業の完了通知）' if trigger_note else ''}")

    try:
        result = process_agent_request(
            input_text=trigger_note or "（自発話しかけ）",
            files_data=[],
            mode=3,
            unresponsive_time="0",
        )
        print(f"⏰ 自発話しかけ完了: success={result.get('success', False)}")
        free_mode_reset_event.set()
        schedule_proactive(result.get('next_proactive_in_minutes', -1))
        _speak_proactive(result)

    except Exception as e:
        print(f"自発話しかけエラー: {str(e)}")


def _speak_proactive(result: dict) -> None:
    """自発話しかけの返事を、ブラウザが開いていれば声で届ける（開いていなければ何もしない）。"""
    text = (result.get('response') or '').strip()
    if not text:
        return
    deadline = time.time() + 60
    while _active_turn_handle is not None and time.time() < deadline:   # 進行中の会話の声と重ねない
        time.sleep(0.5)
    sid = get_active_socketio_sid() if browser_presence()["connected"] else None
    if not sid:
        print("⏰ 自発話しかけ: ブラウザが閉じているので声にはしない（会話ログには残っている）")
        return
    full_tts_text = strip_for_tts(text)
    fish_q: queue.Queue = queue.Queue()
    if full_tts_text:
        fish_q.put(full_tts_text)
    fish_q.put(None)
    socketio.emit('inactivity_reminder', {
        'response': text,
        'session_id': active_session_id,
        'timestamp': time.time(),
        'inactivity_timeout': -1,
        'motion': result.get('motion', '01'),
    })
    print(f"⏰ 自発話しかけ: ブラウザへ声で届ける → SID={sid}, chapter=100（{len(full_tts_text)}字）")
    threading.Thread(target=lambda: _run_fish_audio_from_queue(fish_q, sid, chapter_num=100), daemon=True).start()


def trigger_free_mode_action() -> None:
    """自由行動モード（mode=2）を1回実行する（クライアント配信なし）"""
    global free_mode_consecutive_runs

    if free_mode_consecutive_runs >= MAX_CONSECUTIVE_FREE_MODE_RUNS:
        return

    # Phase 7 切り戻し用: 旧 Dify 実装
    # from agent_main_dify import process_agent_request
    # from agent_main_dify_stream import process_agent_request
    from agent_main import process_agent_request

    print("🧠 自由行動モード（mode=2）を実行します（非配信）")

    try:
        free_mode_response = process_agent_request(
            input_text="（自由行動）",
            files_data=[],
            mode=2,
            unresponsive_time=str(FREE_MODE_INTERVAL_SECONDS)
        )
        print(f"🧠 自由行動モード実行完了: success={free_mode_response.get('success', False)}")
        free_mode_reset_event.set()
        schedule_proactive(free_mode_response.get('next_proactive_in_minutes', -1))
    except Exception as e:
        print(f"自由行動モード実行エラー: {str(e)}")

    free_mode_consecutive_runs += 1
    print(f"🧠 自由行動モード連続実行回数: {free_mode_consecutive_runs}/{MAX_CONSECUTIVE_FREE_MODE_RUNS}")


def free_mode_scheduler_loop() -> None:
    """ユーザー入力から5時間経過後に自由行動モードを実行する監視ループ。
    ユーザー入力があるたびに free_mode_reset_event がセットされ、タイマーがリセットされる。"""
    global free_mode_limit_logged, free_mode_quiet_hours_logged

    print("自由行動モードスケジューラーを開始しました")

    while True:
        try:
            free_mode_reset_event.clear()
            user_input_received = free_mode_reset_event.wait(timeout=FREE_MODE_INTERVAL_SECONDS)

            if user_input_received:
                # いずれかのモードの完了でタイマーリセット → 最初から5時間待つ
                print("🔄 会話イベントを検知 → 自由行動タイマーをリセットします")
                continue

            # 5時間タイムアウト
            if is_in_free_mode_quiet_hours():
                if not free_mode_quiet_hours_logged:
                    print(
                        "🌙 夜間停止時間帯のため自由行動モードをスキップします "
                        f"({FREE_MODE_QUIET_HOURS_START}-{FREE_MODE_QUIET_HOURS_END})"
                    )
                    free_mode_quiet_hours_logged = True
                continue
            elif free_mode_quiet_hours_logged:
                print("🌅 夜間停止時間帯を抜けたため自由行動モード監視を再開します")
                free_mode_quiet_hours_logged = False

            if free_mode_consecutive_runs >= MAX_CONSECUTIVE_FREE_MODE_RUNS:
                if not free_mode_limit_logged:
                    print("🛑 自由行動モードは連続実行上限に達したため停止中です（ユーザー操作でリセット）")
                    free_mode_limit_logged = True
                continue

            trigger_free_mode_action()

        except Exception as e:
            print(f"自由行動モードスケジューラーエラー: {str(e)}")
            time.sleep(5)

def run_log_archive(trigger: str) -> None:
    """WF ログの月次アーカイブを1回実行する（別スレッドから呼ぶ）。

    設定は settings.json の `logs` 節（`archive_enabled` / `keep_days`）。
    GUI は持たない（スタート画面の設定タブには出さない）。
    **失敗しても会話には一切影響させない** —— 例外はここで握って記録だけ残す。
    """
    global settings
    cfg = settings.get('logs', {}) or {}
    if not cfg.get('archive_enabled', True):
        return
    keep_days = int(cfg.get('keep_days', log_archive.DEFAULT_KEEP_DAYS))
    try:
        result = log_archive.run(keep_days=keep_days)
    except Exception as e:
        print(f"ログアーカイブでエラーが発生しました（{trigger}）: {e}")
        traceback.print_exc()
        return
    if result["files"]:
        print(f"🗜 ログをアーカイブしました（{trigger}）: "
              f"{result['files']}件 / {result['months']}か月 / "
              f"{result['freed'] / 1024 / 1024:.1f} MB 削減")
    for message in result["errors"]:
        print(f"ログアーカイブの一部が失敗しました: {message}")


def daily_log_archive_scheduler_loop() -> None:
    """毎日指定時刻にログをアーカイブする監視ループ（1分ごとにチェック）"""
    global last_log_archive_date

    print("ログアーカイブスケジューラーを開始しました")

    while True:
        try:
            time.sleep(60)

            from datetime import datetime
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            # 指定時刻かどうか
            if not (now.hour == DAILY_LOG_ARCHIVE_HOUR and now.minute == DAILY_LOG_ARCHIVE_MINUTE):
                continue

            # 今日すでに実行済みかどうか
            if last_log_archive_date == today_str:
                continue

            last_log_archive_date = today_str
            threading.Thread(
                target=run_log_archive,
                args=("定時",),
                daemon=True,
                name="Log-Archive-Thread"
            ).start()

        except Exception as e:
            print(f"ログアーカイブスケジューラーエラー: {str(e)}")


def context_cache_sweeper_loop() -> None:
    """使われなくなった明示キャッシュを捨てる監視ループ（1分ごと・2026-08-30）。

    明示キャッシュ（`cachedContents`）の保管料は**時間課金**なので、会話が途切れたあと
    持ち続けると払い続けることになる。無通信が続いたら手放す。

    キャッシュ側にも短い TTL が入っているので、**このループが止まっても**
    Google 側で自動的に失効する（掃除は保管料を早く止めるための最適化であって、
    安全性はここに依存していない）。
    """
    from agent import context_cache

    print("明示キャッシュの掃除スケジューラーを開始しました")

    while True:
        try:
            time.sleep(60)
            context_cache.sweep_if_idle()
        except Exception as e:
            print(f"キャッシュ掃除スケジューラーエラー: {str(e)}")
            time.sleep(5)


def daily_memory_scheduler_loop() -> None:
    """毎日指定時刻に記憶更新を実行する監視ループ（1分ごとにチェック）"""
    global last_memory_update_date

    print("定期記憶更新スケジューラーを開始しました")

    while True:
        try:
            time.sleep(60)

            from datetime import datetime
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            # 指定時刻かどうか
            if not (now.hour == DAILY_MEMORY_UPDATE_HOUR and now.minute == DAILY_MEMORY_UPDATE_MINUTE):
                continue

            # 今日すでに実行済みかどうか
            if last_memory_update_date == today_str:
                continue

            # 最後の会話から20分以内ならスキップ
            if last_user_input_time is not None:
                elapsed_minutes = (time.time() - last_user_input_time) / 60
                if elapsed_minutes < DAILY_MEMORY_QUIET_MINUTES:
                    print(
                        f"🧠 定期記憶更新: 最後の会話から {elapsed_minutes:.1f} 分しか経過していないためスキップします"
                        f"（{DAILY_MEMORY_QUIET_MINUTES}分未満）"
                    )
                    last_memory_update_date = today_str  # 翌日まで待つ
                    continue

            print(f"🧠 定期記憶更新を開始します（{today_str} {now.strftime('%H:%M')}）")
            last_memory_update_date = today_str

            def run_daily_memory():
                try:
                    # from models.memory_manager_dify import process_all_conversations  # Phase 7 切り戻し用
                    from models.memory_manager_agent import process_all_conversations
                    process_all_conversations(
                        path_config.conversation_backup_dir,
                        path_config.long_memory_backup_dir
                    )
                    print("🧠 定期記憶更新が完了しました")
                except Exception as e:
                    print(f"定期記憶更新中にエラーが発生しました: {e}")
                    traceback.print_exc()

            threading.Thread(
                target=run_daily_memory,
                daemon=True,
                name="Daily-Memory-Update-Thread"
            ).start()

        except Exception as e:
            print(f"定期記憶更新スケジューラーエラー: {str(e)}")
            time.sleep(5)


# =============================================================================
# TTS テキスト前処理
# =============================================================================

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # 顔文字
    "\U0001F300-\U0001F5FF"  # 記号・絵文字
    "\U0001F680-\U0001F6FF"  # 交通・地図
    "\U0001F700-\U0001F77F"  # 錬金術記号
    "\U0001F780-\U0001F7FF"  # 幾何学図形拡張
    "\U0001F800-\U0001F8FF"  # 追加矢印
    "\U0001F900-\U0001F9FF"  # 補足記号・絵文字
    "\U0001FA00-\U0001FA6F"  # チェス・その他
    "\U0001FA70-\U0001FAFF"  # 絵文字拡張-A
    "\U00002702-\U000027B0"  # 装飾記号
    # \U000024C2-\U0001F251 は削除: 日本語（ひらがな・カタカナ・漢字）を誤って含む広すぎる範囲
    "]+",
    flags=re.UNICODE,
)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def remove_emoji_for_tts(text: str) -> str:
    """絵文字・制御文字のみ除去（空白は保持）。ストリーミング delta 用。

    delta（断片）ごとに呼ぶため、空白の正規化・trim はしない。断片境界の空白を
    消すと単語が繋がってしまうため、除去だけに留める。

    記号（♪ … 〜 ↑ ☆ 等）は**残す**。ミクは記号を句点やテンションの表現に使い、
    Fish はそれを韻律に反映する——除去すると文境界の手がかりが消えて
    「素敵♪また〜」が無ポーズで連結される（実測 155ms。残すと 860ms。
    test/tts_pause_check/results.md）。記号の除去はもともと VOICEVOX の文分割判定を
    守るための処置で、VOICEVOX 経路は 2026-09-25 に削除した。
    """
    text = _EMOJI_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return text


def strip_for_tts(text: str) -> str:
    """TTSに渡す前に絵文字等を除去し、空白を正規化して前後を trim する（全文一括用）。"""
    text = remove_emoji_for_tts(text)
    # 全角スペースも含めて連続空白を1つに
    text = re.sub(r"[ 　\t]+", " ", text)
    return text.strip()


# =============================================================================
# Socket.IO の宛先
# =============================================================================

def browser_presence() -> dict:
    """ブラウザが開いているか（アクティブなセッションの接続があるか）と、最後にマスターが話しかけてからの分数。

    `<system_state>` と自発話しかけ（mode=3）の届け方に使う（2026-10-09）。
    スマホは画面を閉じると接続が切れるので「閉じている」になる。開いたまま席を外していることもあるので分数を添える。
    """
    connected = any(cid == active_session_id for cid in session_mapping.values()) if active_session_id else False
    minutes = (time.time() - last_user_input_time) / 60 if last_user_input_time is not None else None
    return {"connected": connected, "minutes_since_input": minutes}


def get_active_socketio_sid() -> Optional[str]:
    """active_session_id に対応する Socket.IO SID を返す"""
    for sid, cid in session_mapping.items():
        if cid == active_session_id:
            return sid
    print(f"[WARN] get_active_socketio_sid → None (active_session={active_session_id}, mapping={dict(session_mapping)})")
    return None


# =============================================================================
# Fish Audio PCM ストリーミングフロー
# =============================================================================

def _run_fish_audio_from_queue(
    text_queue: queue.Queue,
    target_sid: str,
    chapter_num: int = 0,
) -> None:
    """Fish Audio PCM ストリームをキューから受け取ってブラウザへ送信する（ブロッキング）。

    呼び出し元がテキストを text_queue に put し、終端として None を put することで
    Fish Audio SDK のストリームが閉じる。全文一括投入（リマインダー）でも逐次投入
    （ストリーミング）でも使える。
    """
    from tts_service_fish_audio import FishAudioStreamer

    fish_cfg = settings.get('tts', {}).get('fish_audio', {})
    sample_rate = fish_cfg.get('sample_rate', 44100)
    cancel_event = threading.Event()

    def on_chunk(pcm_bytes: bytes) -> None:
        try:
            socketio.emit('voice_pcm_chunk', {
                'chapter': chapter_num,
                'data': base64.b64encode(pcm_bytes).decode(),
            }, to=target_sid)
        except Exception as e:
            print(f"[ERROR] _run_fish_audio_from_queue voice_pcm_chunk 失敗: {e}")

    def on_finish() -> None:
        try:
            socketio.emit('voice_pcm_end', {'chapter': chapter_num}, to=target_sid)
        except Exception as e:
            print(f"[ERROR] _run_fish_audio_from_queue voice_pcm_end 失敗: {e}")

    try:
        socketio.emit('voice_pcm_start', {
            'sample_rate': sample_rate,
            'channels': 1,
            'chapter': chapter_num,
        }, to=target_sid)
        print(f"[FishAudio] voice_pcm_start emit → {target_sid} (chapter={chapter_num}, sr={sample_rate})")
    except Exception as e:
        print(f"[ERROR] _run_fish_audio_from_queue voice_pcm_start 失敗: {e}")

    FishAudioStreamer().stream(
        text_queue, on_chunk, on_finish,
        cancel_event=cancel_event,
        on_error=lambda exc: _emit_voice_error(exc, target_sid),
    )


def _emit_turn_error_if_failed(result: dict, target_sid: Optional[str]) -> None:
    """ターンが失敗していたら、その理由をブラウザへ知らせる（2026-09-20）。

    **以前はどこへも送っておらず、画面は「考え中」のまま止まっていた。**
    残高切れやAPIキーの誤りは利用者にしか直せないので、必ず外へ出す。

    呼ぶのは `_run_fish_audio_flow` の終わり。※2026-09-20 に、当時あった VOICEVOX 経路
    だけに入れて既定の構成で何も出なかった（経路は 2026-09-25 に1本になった）。
    """
    if result.get('success', True):
        return
    notice = result.get('error_notice') or {}
    message = notice.get('message') or (
        '[本体] エラーが発生しました。スタート画面の「ログ」タブをご覧ください。'
    )
    try:
        socketio.emit('generate_error', {'message': message}, to=target_sid)
        print(f"[ERROR通知] {message.splitlines()[0]}")
    except Exception as e:
        print(f"[ERROR] エラー通知の送信に失敗: {e}")


def _emit_voice_error(exc: BaseException, target_sid: Optional[str]) -> None:
    """声が作れなかったことをブラウザへ知らせる（2026-09-20）。

    **文字は出ているのに声だけ出ない**状態は、利用者からは原因が見えない。
    残高切れ・APIキーの誤りはこちらでは直せないので、出どころを添えて外へ出す。
    """
    try:
        from utils.api_error import describe
        socketio.emit(
            'generate_error',
            {'message': describe(exc, source="Fish Audio")['message']},
            to=target_sid,
        )
    except Exception as e:
        print(f"[ERROR] 声のエラー通知に失敗: {e}")


def _emit_unexpected_error(exc: BaseException, target_sid: Optional[str]) -> None:
    """会話の途中で想定外の例外が出たことをブラウザへ知らせる（2026-09-26）。

    出どころ（`[本体]`／LLM の失敗なら `[Google AI Studio]`）を付け、APIキーらしき文字列は伏せる
    （`utils/api_error.describe`）。**以前は例外の文字列をそのまま送っていた**——見出しと伏せ字は、
    2026-09-25 に削除した VOICEVOX 経路の `except` にしか無かった。
    """
    try:
        from utils.api_error import describe, looks_like_llm_error
        source = "Google AI Studio" if looks_like_llm_error(exc) else "本体"
        message = describe(exc, source=source)['message']
    except Exception:
        message = '[本体] エラーが発生しました。スタート画面の「ログ」タブをご覧ください。'
    try:
        socketio.emit('generate_error', {'message': message}, to=target_sid)
    except Exception as e:
        print(f"[ERROR] エラー通知の送信に失敗: {e}")


def _run_fish_audio_flow(
    input_text: str,
    files_data: list,
    target_sid: Optional[str],
    client_session_id: Optional[str],
    voice_segments_raw: str = '',
    turn_handle: Optional["TurnHandle"] = None,
) -> None:
    """Fish Audio WebSocket PCM ストリーミングで TTS を処理するフロー。

    各 LLM ループのテキストを chapter 単位でストリーミングし、
    PCM チャンクをブラウザへリアルタイム送信する。

    Phase 6 即応フィラー: agent service 側でフィラー LLM を並列起動し、
    on_filler_ready コールバック経由でテキストがここに届く。本フローでは:
      - chap0: フィラー音声（agent からテキスト受信時に開始）
      - chap1+: メイン応答音声
    の順次再生を管理する。フィラーのテキスト生成は agent の責務、TTS chapter
    管理は gateway の責務、という責務分離。

    デバッグ用に主要区間の経過時間（ターン開始からの ms）を [TIMING] として出力する。

    turn_handle: `_active_turn_handle` は**呼び出し元の `generate()` が既に同期的に
    登録済み**（2026-09-22）。以前はここで生成・登録していたが、`_prewarm_chapter()` 等の
    前処理を挟む間に登録が遅れ、ほぼ同時に届いた2件目の `/generate` が「進行中ターンなし」と
    誤判定して独立した新規ターンを並行起動してしまっていた（実機で確認：同じチャプター番号が
    2系統から届いて衝突し、片方の声が届かなかった）。ここでは受け取ったハンドルをそのまま使う。
    """
    global _current_fish_cancel, _current_fish_target_sid, last_user_input_time

    from tts_service_fish_audio import FishAudioStreamer
    from agent_main import process_agent_request

    # ── タイミング計測（ターン開始基準）─────────────────────────────────
    _t0 = time.perf_counter()
    def _ms() -> int:
        return int((time.perf_counter() - _t0) * 1000)

    cancel_event = threading.Event()
    _current_fish_cancel = cancel_event
    _current_fish_target_sid = target_sid

    fish_cfg = settings.get('tts', {}).get('fish_audio', {})
    sample_rate = fish_cfg.get('sample_rate', 44100)

    chapter_state: dict = {
        'text_queue': None,       # queue.Queue | None
        'streamer_thread': None,  # threading.Thread | None
        'chapter_num': 0,
        'chapter_start_ms': 0,    # チャプター開始時刻（ms）
        'first_delta_logged': False,
        # Phase 6 即応フィラー: agent からの on_filler_ready コールバック受信を待つ
        'filler_done': threading.Event(),  # フィラー chap0 終了通知（main は待つ）
        'filler_attempted': False,         # 1ターン内で多重発火を防ぐ
        'lock': threading.Lock(),          # text_queue / streamer_thread への並列アクセス保護
        # いまのループの**本応答**を載せているチャプター番号（フィラーと区別するため）。
        # 「応答音声が実際に鳴り始めたか」を TurnHandle へ伝える判定に使う
        'main_chapter': None,
        # 声の調子。ストリーマはテキストより先に起動されるので、値ではなく箱を渡し、
        # LLM が決めた時点で差し替える（届かなければ「そのまま」＝加工なし）
        'tone_holder': None,
    }

    def _start_chapter() -> None:
        """新しいチャプターのストリーミングを開始する。"""
        q: queue.Queue = queue.Queue()
        chapter_num: int = chapter_state['chapter_num']
        chapter_state['chapter_start_ms'] = _ms()
        print(f"[TIMING] chap{chapter_num} _start_chapter: {chapter_state['chapter_start_ms']}ms")

        # チャプター内で最初の PCM チャンクが届いた時刻だけ記録するためのフラグ
        first_pcm_logged = {'done': False}

        def on_chunk(pcm_bytes: bytes) -> None:
            if cancel_event.is_set():
                return
            if not first_pcm_logged['done']:
                first_pcm_logged['done'] = True
                cur = _ms()
                delta = cur - chapter_state['chapter_start_ms']
                print(f"[TIMING] chap{chapter_num} 初回PCMチャンク受信: {cur}ms "
                      f"(チャプター開始から +{delta}ms, {len(pcm_bytes)}bytes)")
                # ★このループの本応答が**実際に鳴り始めた**ことを runner へ伝える（2026-09-22）。
                # 半二重（鳴っている間はマイクがミュート）なので、以後に声で割り込まれることは
                # 無い＝「鳴り始めたか」が「発話が届いたか」の判定になる。フィラー（chap0）は
                # 応答本文ではないので、本応答を載せているチャプターだけを対象にする
                if turn_handle is not None and chapter_num == chapter_state['main_chapter']:
                    turn_handle.mark_audio_started()
            try:
                socketio.emit('voice_pcm_chunk', {
                    'chapter': chapter_num,
                    'data': base64.b64encode(pcm_bytes).decode(),
                }, to=target_sid)
            except Exception as e:
                print(f"[ERROR] voice_pcm_chunk emit 失敗: {e}")

        def on_finish() -> None:
            print(f"[TIMING] chap{chapter_num} Fish Audio ストリーム終了: {_ms()}ms")
            try:
                socketio.emit('voice_pcm_end', {
                    'chapter': chapter_num,
                }, to=target_sid)
            except Exception as e:
                print(f"[ERROR] voice_pcm_end emit 失敗: {e}")

        try:
            socketio.emit('voice_pcm_start', {
                'sample_rate': sample_rate,
                'channels': 1,
                'chapter': chapter_num,
            }, to=target_sid)
            print(f"[TIMING] chap{chapter_num} voice_pcm_start emit: {_ms()}ms")
        except Exception as e:
            print(f"[ERROR] voice_pcm_start emit 失敗: {e}")

        # 箱はチャプターをまたいで使い回す（同じターンの声の調子は変わらない）
        if chapter_state['tone_holder'] is None:
            from utils.voice_tone import ToneHolder
            chapter_state['tone_holder'] = ToneHolder()

        streamer = FishAudioStreamer()
        t = threading.Thread(
            target=streamer.stream,
            args=(q, on_chunk, on_finish),
            kwargs={'cancel_event': cancel_event,
                    'tone_holder': chapter_state['tone_holder'],
                    'on_error': lambda exc: _emit_voice_error(exc, target_sid)},
            daemon=True,
        )
        t.start()
        chapter_state['text_queue'] = q
        chapter_state['streamer_thread'] = t

    def _prewarm_chapter() -> None:
        """Phase 5: 次のチャプター用 streamer thread を事前起動する（WS HS pre-warm）。

        text_queue は空のままなので、Fish Audio SDK 内の text_generator() は queue.get で
        ブロックする。その間に WS HS と Fish Audio サーバ側のモデル初期化が並行で進むため、
        実テキスト到着時の put→PCM 時間が大幅短縮される（ベンチで -640ms 実測）。
        """
        with chapter_state['lock']:
            if chapter_state['text_queue'] is not None:
                print(f"[WARN] prewarm: chap{chapter_state['chapter_num']} 既に起動済み・スキップ")
                return
            print(f"[TIMING] chap{chapter_state['chapter_num']} 事前起動（WS HS pre-warm）: {_ms()}ms")
            _start_chapter()

    def on_filler_ready(filler_text: str) -> None:
        """Phase 6 即応フィラー: agent service が生成したフィラーテキストを受信して chap0 に流す。

        chap0 streamer thread はターン開始時に _prewarm_chapter() で既に事前起動済み。
        ここでは queue.put(text) → queue.put(None) で実テキストを流すだけ。

        責務:
          - chap0 へテキスト投入（queue.put）
          - 終端 None を送って streamer 完了待ち
          - chapter_num を 1 へ進めて chap1（main 用）も事前起動
          - filler_done をセットして main の wait を解放
        """
        try:
            print(f"[TIMING] on_filler_ready 受信: {_ms()}ms  text={filler_text!r}")
            chapter_state['filler_attempted'] = True

            with chapter_state['lock']:
                q = chapter_state['text_queue']
                if q is None:
                    # 事前起動されていない（filler_enabled=false 等）→ filler 諦め
                    print(f"[WARN] filler: chap0 が事前起動されていない・filler スキップ")
                    chapter_state['filler_done'].set()
                    return
                if filler_text and not cancel_event.is_set():
                    q.put(filler_text)
                q.put(None)  # 終端（text が空でも streamer を閉じるため必ず None を送る）

            # streamer 完了を待つ
            streamer_thread = chapter_state['streamer_thread']
            if streamer_thread is not None:
                streamer_thread.join(timeout=15)
                if streamer_thread.is_alive():
                    print(f"[WARN] filler streamer thread タイムアウト")
                else:
                    print(f"[TIMING] filler streamer thread join 完了: {_ms()}ms")

            # main 用に chap1 へ進め、直ちに pre-warm
            with chapter_state['lock']:
                chapter_state['text_queue'] = None
                chapter_state['streamer_thread'] = None
                chapter_state['chapter_num'] = 1
            _prewarm_chapter()  # chap1 事前起動（main LLM 待機中に WS HS）

        except Exception as e:
            print(f"[ERROR] filler chapter emit 失敗: {e}")
            traceback.print_exc()
        finally:
            chapter_state['filler_done'].set()
            print(f"[TIMING] filler_done set: {_ms()}ms")

    def on_voice_tone(tone: str) -> None:
        """LLM が決めた声の調子を箱へ入れる（response の差分より先に届く）。

        チャプターがまだ起動していなければ箱だけ先に作る。_start_chapter が
        同じ箱を streamer へ渡すので、起動の前後どちらでも取りこぼさない。
        """
        with chapter_state['lock']:
            if chapter_state['tone_holder'] is None:
                from utils.voice_tone import ToneHolder
                chapter_state['tone_holder'] = ToneHolder()
            sound = chapter_state['tone_holder'].set(tone)
        print(f"[TIMING] 声の調子: {tone} → {sound} ({_ms()}ms)")

    def on_text_delta(text: str) -> None:
        # 絵文字・制御文字を除去（空白は残す）。除去後に空なら捨てる。
        # .strip() 判定にすると単語間スペースの delta も落ちて単語が繋がるため not text のまま。
        # 記号（♪…〜↑☆等）は Fish が韻律に使うため残す。
        text = remove_emoji_for_tts(text)
        if not text:
            return
        if not chapter_state['first_delta_logged']:
            chapter_state['first_delta_logged'] = True
            print(f"[TIMING] LLM 初回 on_text_delta（TTFT 相当）: {_ms()}ms")
            # フィラー完了を待つ（既に完了していれば即返る・最大 15 秒）。
            # フィラーが起動していない（filler_attempted=False かつ done もセットされてない）
            # ケースを想定して、念のため timeout 短めで wait する。
            if not chapter_state['filler_done'].is_set():
                wait_timeout = 15 if chapter_state['filler_attempted'] else 0.5
                print(f"[TIMING] main: filler 完了待ち開始: {_ms()}ms (timeout={wait_timeout}s)")
                chapter_state['filler_done'].wait(timeout=wait_timeout)
                print(f"[TIMING] main: filler 完了待ち終了: {_ms()}ms")
        with chapter_state['lock']:
            # ★事前起動したチャプターが、文字を待ちきれずに閉じていたら次のチャプターで出し直す（2026-10-10）。
            #   ツールの前に次のチャプターを起動しておくが、ストリーマは60秒文字が来ないと閉じる。
            #   曲の確認（最大3分待つ）のように長いツールのあとは、閉じた後に返事が届いて声にならなかった。
            st = chapter_state['streamer_thread']
            if chapter_state['text_queue'] is not None and st is not None and not st.is_alive():
                print(f"[TIMING] chap{chapter_state['chapter_num']} は待ちきれずに閉じていた"
                      f"（ツールが長かった）→ 次のチャプターで出し直す: {_ms()}ms")
                chapter_state['text_queue'] = None
                chapter_state['streamer_thread'] = None
                chapter_state['chapter_num'] += 1
            if chapter_state['text_queue'] is None:
                # chap1 が事前起動されていない場合のフォールバック（filler 失敗ケース等）
                _start_chapter()
            # このループの本応答を載せるチャプターを確定する（初回デルタの時点）。
            # ループが変わった＝新しい応答なので「まだ鳴っていない」へ戻す（2026-09-22）
            if chapter_state['main_chapter'] != chapter_state['chapter_num']:
                chapter_state['main_chapter'] = chapter_state['chapter_num']
                if turn_handle is not None:
                    turn_handle.reset_audio_started()
            chapter_state['text_queue'].put(text)

    def on_object_complete(obj: dict, obj_index: int) -> None:
        rt = obj.get("requires_tool")
        is_tool = (isinstance(rt, str) and rt.lower() == "true") or (rt is True)

        response_text = obj.get("response") or ""
        if isinstance(response_text, str) and response_text.lower() == "null":
            response_text = ""

        has_audio = chapter_state['text_queue'] is not None
        chapter_num = chapter_state['chapter_num']
        print(f"[TIMING] chap{chapter_num} on_object_complete 呼び出し: {_ms()}ms "
              f"(is_tool={is_tool}, has_audio={has_audio})")

        # JSON 確定直後（音声合成完了を待たず）に表情・タイムアウトをブラウザへ通知する。
        # ツール呼び出し中も LLM が motion を設定した場合は送信する。
        motion = obj.get("motion", "01") or "01"
        if isinstance(motion, str) and motion.lower() == "null":
            motion = "null"
        inactivity_timeout = obj.get("inactivity_timeout", 60)
        if isinstance(inactivity_timeout, str):
            try:
                inactivity_timeout = int(inactivity_timeout)
            except ValueError:
                inactivity_timeout = 60

        if not is_tool or motion != "null":
            print(f"[DEBUG] on_object_complete (fish): emitting response_ready motion={motion} has_audio={has_audio} is_tool={is_tool}")
            try:
                socketio.emit('response_ready', {
                    'motion': motion if motion != "null" else "01",
                    'inactivity_timeout': inactivity_timeout,
                    'has_audio': has_audio,
                    'response': response_text,
                    'session_id': client_session_id,
                }, to=target_sid)
                print(f"[TIMING] response_ready emit: {_ms()}ms")
            except Exception as e:
                print(f"[ERROR] response_ready emit 失敗: {e}")
                traceback.print_exc()

        # 現チャプターを閉じる（Fish Audio WebSocket を閉じて全 PCM を flush）
        if chapter_state['text_queue'] is not None:
            chapter_state['text_queue'].put(None)
            if chapter_state['streamer_thread'] is not None:
                chapter_state['streamer_thread'].join(timeout=30)
                if chapter_state['streamer_thread'].is_alive():
                    print(f"[WARN] Fish Audio streamer thread タイムアウト (chapter={chapter_num})")
                else:
                    print(f"[TIMING] chap{chapter_num} streamer thread join 完了: {_ms()}ms")
            chapter_state['text_queue'] = None
            chapter_state['streamer_thread'] = None
            chapter_state['chapter_num'] += 1

        # Phase 5: tool 呼び出しが続く場合、次のループ用の chapter を pre-warm する。
        # こうすることで loop2 の LLM が走っている間（数秒〜十数秒）に WS HS が前倒し完了し、
        # loop2 の初音到達 (put→PCM) が大幅短縮される。
        if is_tool:
            _prewarm_chapter()

    # settings の filler_enabled フラグでオン/オフ切替（デバッグ・切り戻し用）
    from agent.config import get_pydantic_ai_filler_enabled
    filler_enabled = get_pydantic_ai_filler_enabled(settings)
    if not filler_enabled:
        # 無効時は最初から filler_done をセットしておき、main は即座に chap0 から開始する
        chapter_state['filler_done'].set()
    print(f"[TIMING] filler_enabled={filler_enabled}")

    # Phase 5: ターン開始と同時に chap0 を事前起動して Fish Audio WS を pre-warm する。
    #   - filler_enabled=true  → chap0 は filler 用（filler テキスト到着まで queue 待ち）
    #   - filler_enabled=false → chap0 は main 用（main 初回 delta 到着まで queue 待ち）
    # WS HS と Fish Audio サーバ側モデル初期化が並行で進むため、put→PCM が ~640ms 短縮される。
    _prewarm_chapter()

    # P2 畳み込み: 進行中ターンへの追加発話注入ハンドル。
    # 生成・`_active_turn_handle` への登録は呼び出し元の `generate()` が同期的に済ませている
    # （このコメントの上にある docstring 参照）。ここでは受け取ったものを使うだけ
    global _active_turn_handle
    assert turn_handle is not None, "_run_fish_audio_flow は generate() から turn_handle 付きで呼ばれる前提"

    try:
        print(f"[TIMING] _run_fish_audio_flow 開始: {_ms()}ms (input={input_text[:30]!r})")
        result = process_agent_request(
            input_text=input_text,
            files_data=files_data,
            mode=0,
            voice_segments_raw=voice_segments_raw,
            on_filler_ready=on_filler_ready if filler_enabled else None,
            on_filler_motion=(
                (lambda motion: _try_emit_reaction_motion(target_sid, motion, "filler_final"))
                if filler_enabled else None
            ),
            on_text_delta=on_text_delta,
            on_object_complete=on_object_complete,
            turn_handle=turn_handle,
            on_voice_tone=on_voice_tone,
        )

        # 失敗したらブラウザへ知らせる（2026-09-20）
        _emit_turn_error_if_failed(result, target_sid)

        last_user_input_time = time.time()
        free_mode_reset_event.set()
        reset_free_mode_counter()
        schedule_proactive(result.get('next_proactive_in_minutes', -1))
        print(f"[TIMING] _run_fish_audio_flow 正常終了: {_ms()}ms (total)")

    except Exception as e:
        print(f"[ERROR] _run_fish_audio_flow 例外: {e}")
        traceback.print_exc()
        _emit_unexpected_error(e, target_sid)
    finally:
        _current_fish_cancel = None
        _current_fish_target_sid = None
        # 例外時も含めターンを確実に閉じる（以後の発話は通常の新規ターンになる）
        turn_handle.force_close()
        if _active_turn_handle is turn_handle:
            _active_turn_handle = None


# =============================================================================
# generate バックグラウンド処理
# =============================================================================

#: 発話の送り先（アクティブな端末）が見つからないとき、つなぎ直しを待つ最長の秒数
TARGET_SID_WAIT_SEC = 3.0


def _process_generate_async(
    input_text: str,
    files_data: list,
    target_sid: Optional[str],
    client_session_id: Optional[str],
    voice_segments_raw: str = '',
    turn_handle: Optional["TurnHandle"] = None,
) -> None:
    """Fire & Forget で呼ばれるバックグラウンドスレッド。`_run_fish_audio_flow` へ委譲する。

    turn_handle: `generate()` が新規ターン判定と同時に**同期的に**生成・登録済みのもの
    （後始末のタイミング競合を避けるため。`_run_fish_audio_flow` の docstring 参照）。
    """
    assert turn_handle is not None, "generate() が turn_handle を渡す"
    global _active_turn_handle
    try:
        # ★**送り先が空なら、つなぎ直しを少し待って探し直す**（2026-10-12・本人了承）。送り先が空だと下の送信は
        #   **全員宛**になり、PC とスマホの両方で声が鳴った（スマホの接続が発話の直前に切れてつなぎ直された）。
        #   つなぎ直しは数百ミリ秒で終わるので、待てば正しい1台に届く。それでも無ければ、従来どおり全員宛のまま進める
        if not target_sid:
            deadline = time.time() + TARGET_SID_WAIT_SEC
            while not target_sid and time.time() < deadline:
                time.sleep(0.1)
                target_sid = get_active_socketio_sid()
            if target_sid:
                print(f"[generate] 送り先を探し直して見つけた: {target_sid}")
        _run_fish_audio_flow(input_text, files_data, target_sid, client_session_id,
                             voice_segments_raw, turn_handle)
    finally:
        # ★安全網（2026-09-22）: `_run_fish_audio_flow` は自前の finally で
        # `_active_turn_handle` を戻すが、その finally へ到達する**前**
        # （chap0 の事前起動・フィラー設定等の準備区間）で例外が飛んだ場合に備え、
        # ここでも確実に戻す。戻し忘れると `_active_turn_handle` が生き残ったままになり、
        # **以後の発話が全部このターンへ畳み込まれようとして誰にも処理されなくなる**
        # （キューに積まれるだけで読むループが無い＝声が永久に返らなくなる）
        if _active_turn_handle is turn_handle:
            _active_turn_handle = None


# =============================================================================
# 初期化処理
# =============================================================================

def _wait_until_ready(name: str, url: str, timeout_sec: float = 60.0) -> bool:
    """内部サービスが実際に応答を返すまで待つ（2026-08-30）。

    **以前は `time.sleep(2)` で「たぶん起動した」ことにしていた**が、これは
    「起動した」ではなく「2秒経った」に過ぎない。VDB は Chroma のロードに 2 秒以上
    かかることがあり、その間に来た要求は失敗する。実際、ブラウザ接続時の
    キャッシュ事前作成が起動直後に走ったとき、記憶が空のまま組み立てられた
    （`Read timed out` → 中身が欠けたキャッシュができ、最初の発話で作り直しになった）。

    固定待ちをやめ、**応答が返るまで待つ**。上限に達したら諦めて先へ進む
    （そのサービスを使う機能が個別に失敗するだけで、起動そのものは止めない）。

    Args:
        name: 表示名
        url: 叩く URL（軽いものを選ぶこと）
        timeout_sec: 上限。超えたら諦める

    Returns:
        応答が返れば True、諦めたら False
    """
    deadline = time.time() + timeout_sec
    started = time.time()
    while time.time() < deadline:
        try:
            r = http_requests.get(url, timeout=3)
            if r.status_code < 500:
                print(f"  {name} の起動を確認しました（{time.time() - started:.1f}秒）")
                return True
        except Exception:
            pass
        time.sleep(0.5)
    print(f"  [WARN] {name} が {timeout_sec:.0f} 秒以内に応答しませんでした（続行します）")
    return False


def initialize_app() -> bool:
    """アプリケーションの初期化"""
    global initialization_success, path_config, settings
    
    try:
        from utils.path_config import PathConfig
        # from models.memory_manager_dify import process_all_conversations  # Phase 7 切り戻し用
        from models.memory_manager_agent import process_all_conversations

        # パス設定の初期化
        path_config = PathConfig.initialize(BASE_DIR)
        path_config.ensure_directories()

        # ブラウザへ届ける能力を Agent 側へ登録する（依存は Gateway → Agent の一方向を保つ）。
        # ツールから `import gateway_service` してはいけない: Gateway は `__main__` として
        # 起動されるため再 import で状態が空の別インスタンスになる（2026-08-15 の実害）
        from agent.host_bridge import set_browser_presence, set_song_playback_reserver
        set_song_playback_reserver(reserve_song_playback)
        set_browser_presence(browser_presence)

        reset_free_mode_counter()
        
        # 旧構造の settings.json は読み込み前に新構造へ一度きり変換する（.bak を残す）。
        # あわせて、あとから増えた設定キーをひな形（settings_sample.json）から補う
        try:
            from utils.settings_migrate import migrate_settings_file_if_needed
            migrate_settings_file_if_needed(path_config.settings_file)
        except Exception as e:
            print(f"設定ファイルのマイグレーション確認に失敗しました: {e}")

        # 設定ファイルを直接読み込む
        try:
            with open(str(path_config.settings_file), 'r', encoding='utf-8-sig') as f:   # BOM 付きでも読む
                settings = json.load(f)
            print("設定ファイルを読み込みました")
        except FileNotFoundError:
            print(f"設定ファイルが見つかりません: {path_config.settings_file}")
            settings = {}
        except json.JSONDecodeError as e:
            print(f"設定ファイルの形式が不正です: {e}")
            settings = {}

        # ログイン保護の設定を反映（セッション期間・失敗ロック。設定読込後でないと決まらない）
        configure_auth()

        # 起動時バリデーション（手動編集対策。設定GUIの保存時と同じルール）
        # 違反機能は明示エラーを表示して該当機能のみ停止し、起動は続行する
        try:
            from utils import settings_validate
            if settings_validate.google_key_missing(settings):
                print("=" * 60)
                print("[settings] 致命的: api_keys.google（Google AI Studio APIキー）が未設定です。")
                print("[settings] 会話が一切できません。/settings ページまたは settings.json で設定してください。")
                print("=" * 60)
            startup_errors = settings_validate.validate_settings(settings)
            if startup_errors:
                print("=" * 60)
                print("[settings] 設定の検証エラーがあります（該当機能のみ停止して続行します）:")
                for err_path, err_msg in startup_errors.items():
                    print(f"[settings]   - {err_path}: {err_msg}")
                for applied in settings_validate.apply_startup_downgrades(settings, startup_errors):
                    print(f"[settings]   → {applied}")
                print("=" * 60)
        except Exception as e:
            print(f"[settings] 起動時バリデーションに失敗しました（続行します）: {e}")

        # KVサービスを別スレッドで起動
        try:
            kv_config = settings.get('services', {}).get('kv', {})
            if kv_config.get('enabled', True):
                from KV_service import run_kv_service
                
                kv_port = kv_config.get('port', 8010)

                kv_service_thread = threading.Thread(
                    target=run_kv_service,
                    args=(path_config, kv_port, INTERNAL_HOST),
                    daemon=True,
                    name="KV-Service-Thread"
                )
                kv_service_thread.start()
                print(f"KVサービスを別スレッドで起動しました (ポート: {kv_port})")
                _wait_until_ready("KVサービス", f"http://127.0.0.1:{kv_port}/health")
            else:
                print("KVサービスは無効化されています")
        except Exception as e:
            print(f"KVサービスの起動中にエラーが発生しました: {e}")
            traceback.print_exc()

        # VDBサービスを別スレッドで起動
        try:
            vdb_config = settings.get('services', {}).get('vdb', {})
            if vdb_config.get('enabled', True):
                from VDB_service import run_vdb_service

                vdb_port = vdb_config.get('port', 8020)
                vdb_root = path_config.app_dir.parent.parent  # miku_agent/

                vdb_service_thread = threading.Thread(
                    target=run_vdb_service,
                    args=(settings, vdb_root, vdb_port, INTERNAL_HOST),
                    daemon=True,
                    name="VDB-Service-Thread"
                )
                vdb_service_thread.start()
                print(f"VDBサービスを別スレッドで起動しました (ポート: {vdb_port})")
                # VDB は Chroma のロードがあり、**起動に最も時間がかかる**。
                # /vdb/stats は件数を返すだけの軽い経路（/health は無い）
                _wait_until_ready("VDBサービス", f"http://127.0.0.1:{vdb_port}/vdb/stats",
                                  timeout_sec=120.0)
            else:
                print("VDBサービスは無効化されています")
        except Exception as e:
            print(f"VDBサービスの起動中にエラーが発生しました: {e}")
            traceback.print_exc()

        # Recognition Service（顔・声）を別スレッドで起動（常時起動）
        try:
            services_cfg = settings.get('services', {})
            # 旧キー face_recognition も読む（settings_migrate 前の設定ファイル対応）
            rec_config = services_cfg.get('recognition') or services_cfg.get('face_recognition', {})
            from recognition_service import run_recognition_service

            rec_port = rec_config.get('port', 5007)

            rec_thread = threading.Thread(
                target=run_recognition_service,
                args=(rec_port, INTERNAL_HOST),
                daemon=True,
                name="Recognition-Service-Thread"
            )
            rec_thread.start()
            print(f"Recognition Service（顔・声）を別スレッドで起動しました (ポート: {rec_port})")
            time.sleep(1)
        except Exception as e:
            print(f"Recognition Serviceの起動中にエラーが発生しました: {e}")
            traceback.print_exc()

        # STT Realtime Serviceを別スレッドで起動
        # （stt.provider: none | fish_audio）
        try:
            stt_active = _get_active_stt_config()
            stt_provider = stt_active.get('provider', 'none')
            stt_rt_port = stt_active.get('port', 5006)

            runner = None
            if stt_provider == 'none':
                print("STT Realtime Serviceは無効化されています (provider: none)")
            elif stt_provider == 'fish_audio':
                from stt_service_fish_audio import run_stt_fish_service
                runner = run_stt_fish_service
            else:
                # サイレント失敗禁止: 不正値は起動時に明示エラー
                raise ValueError(
                    f"stt.provider の値が不正です: '{stt_provider}'"
                    "（none | fish_audio のいずれか）")

            if runner is not None:
                stt_rt_thread = threading.Thread(
                    target=runner,
                    args=(stt_rt_port, INTERNAL_HOST),
                    daemon=True,
                    name=f"STT-{stt_provider}-Service-Thread"
                )
                stt_rt_thread.start()
                print(f"STT Realtime Service ({stt_provider}) を別スレッドで起動しました"
                      f" (ポート: {stt_rt_port})")
                time.sleep(1)
        except Exception as e:
            print(f"STT Realtime Serviceの起動中にエラーが発生しました: {e}")
            traceback.print_exc()



        # 旧 Long Memory DB Service は VDB collection に統合済み（Phase 3 移行完了）。
        # 長期記憶は VDB の long_memory collection で管理し、long_memory_vdb_mcp_server.py
        # 経由で memory_update_db.py がアクセスする。
        # 詳細: .claude/skills/miku-dev/details/latency_benchmark.md

        # Agent Serviceを別スレッドで起動
        try:
            agent_svc_config = settings.get('services', {}).get('agent', {})
            from agent_service import run_agent_service

            agent_svc_port = agent_svc_config.get('port', 5010)

            agent_service_thread = threading.Thread(
                target=run_agent_service,
                args=(agent_svc_port, INTERNAL_HOST),
                daemon=True,
                name="Agent-Service-Thread"
            )
            agent_service_thread.start()
            print(f"Agent Serviceを別スレッドで起動しました (ポート: {agent_svc_port})")
            _wait_until_ready("Agent Service", f"http://127.0.0.1:{agent_svc_port}/ping")
        except Exception as e:
            print(f"Agent Serviceの起動中にエラーが発生しました: {e}")
            traceback.print_exc()

        # 起動時に会話を登録（バックグラウンドで非同期実行）
        # --skip-memory フラグが指定されている場合はスキップ
        if '--skip-memory' in sys.argv:
            print("⏭️ 記憶登録処理はスキップされました")
        else:
            def run_memory_process():
                try:
                    print("会話ファイルの処理を開始します...")
                    process_all_conversations(
                        path_config.conversation_backup_dir,
                        path_config.long_memory_backup_dir
                    )
                    print("会話ファイルの処理が完了しました")
                except Exception as e:
                    print(f"会話ファイルの処理中にエラーが発生しました: {e}")
                    traceback.print_exc()
            
            memory_thread = threading.Thread(
                target=run_memory_process,
                daemon=True,
                name="Memory-Process-Thread"
            )
            memory_thread.start()
            print("会話ファイルの処理をバックグラウンドで開始しました")

        # 自由行動モードスケジューラーを起動
        free_mode_thread = threading.Thread(
            target=free_mode_scheduler_loop,
            daemon=True,
            name="Free-Mode-Scheduler-Thread"
        )
        free_mode_thread.start()
        print("自由行動モードスケジューラーをバックグラウンドで開始しました")

        # 定期記憶更新スケジューラーを起動
        daily_memory_thread = threading.Thread(
            target=daily_memory_scheduler_loop,
            daemon=True,
            name="Daily-Memory-Scheduler-Thread"
        )
        daily_memory_thread.start()
        print("定期記憶更新スケジューラーをバックグラウンドで開始しました")

        # ログアーカイブスケジューラーを起動
        log_archive_thread = threading.Thread(
            target=daily_log_archive_scheduler_loop,
            daemon=True,
            name="Log-Archive-Scheduler-Thread"
        )
        log_archive_thread.start()
        print("ログアーカイブスケジューラーをバックグラウンドで開始しました")

        # 明示キャッシュの掃除スケジューラーを起動（保管料を早く止めるため）
        cache_sweeper_thread = threading.Thread(
            target=context_cache_sweeper_loop,
            daemon=True,
            name="Context-Cache-Sweeper-Thread"
        )
        cache_sweeper_thread.start()
        print("明示キャッシュの掃除スケジューラーをバックグラウンドで開始しました")

        # 起動時にも1回走らせる（キャッチアップ）。
        # **定時トリガーだけでは、その時刻に PC が起動していない環境で一度も走らない**。
        # 記憶更新（上）にはこの配慮が無いが、ここは揃えずに足す（走らなければ機能がゼロになるため）
        threading.Thread(
            target=run_log_archive,
            args=("起動時",),
            daemon=True,
            name="Log-Archive-Catchup-Thread"
        ).start()


        initialization_success = True
        return True
        
    except Exception as e:
        print(f"初期化エラー: {str(e)}")
        traceback.print_exc()
        initialization_success = False
        return False

# =============================================================================
# 静的ファイル配信
# =============================================================================

@app.after_request
def _add_no_cache_headers(response):
    """静的アセットに Cache-Control: no-cache を付ける。

    Cloudflare Tunnel 経由の配信では、オリジンがキャッシュヘッダを返さないと
    Cloudflare が .js/.css/モデルファイルを既定 TTL でエッジキャッシュし、
    改修後もスマホ等に古いファイルが配られ続ける（2026-07-11 リアクション未動作の原因）。
    no-cache は「使う前に必ず再検証」の意味で、ETag による 304 応答が効くため転送コストは小さい。
    """
    p = request.path
    if p == '/' or p.startswith(('/js/', '/css/', '/live2d', '/assets/', '/Core/', '/Resources/', '/settings', '/api/settings')):
        response.headers['Cache-Control'] = 'no-cache'
    return response


@app.route('/css/<path:filename>')
def serve_css(filename: str):
    """CSSファイルを配信（`../` で配信フォルダの外へ出られないよう send_from_directory を使う）"""
    return send_from_directory(str(TEMPLATES_DIR / 'css'), filename, mimetype='text/css')

@app.route('/js/<path:filename>')
def serve_js(filename: str):
    """JSファイルを配信（`../` で配信フォルダの外へ出られないよう send_from_directory を使う）"""
    import mimetypes
    if filename.endswith(('.js', '.mjs')):
        # Windows の mimetypes は .mjs を text/plain と判定し、ES モジュールの
        # 厳格な MIME 検査で import が失敗する（vendor/mediabunny.mjs 等）ため明示する
        mimetype = 'text/javascript'
    else:
        mimetype = mimetypes.guess_type(filename)[0] or 'application/javascript'
    return send_from_directory(str(TEMPLATES_DIR / 'js'), filename, mimetype=mimetype)

# =============================================================================
# Live2D関連
# =============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """ログイン画面（パスワードのみ。ID は持たない）。

    パスワード未設定なら保護は無効なので、そのままトップへ返す。
    失敗が続いた相手は LoginLimiter が一定時間ロックする（総当たり対策）。
    """
    next_path = request.args.get('next') or request.form.get('next') or '/'
    if not next_path.startswith('/'):
        next_path = '/'                      # 外部サイトへの誘導を防ぐ
    if not _auth_password_hash():
        return redirect(next_path)

    # ロック判定の単位はアクセス元。トンネル経由は全員 127.0.0.1 に見えるため、
    # Cloudflare が付ける実クライアント IP を優先する
    who = request.headers.get('CF-Connecting-IP') or request.remote_addr or 'unknown'

    if request.method == 'POST':
        left = _login_limiter.locked_seconds_left(who)
        if left > 0:
            return render_template(
                'login.html', next=next_path,
                error=f"しばらく待ってからお試しください（あと約{max(1, left // 60)}分）"), 429
        if verify_password(_auth_password_hash(), request.form.get('password', '')):
            _login_limiter.record_success(who)
            session['authed'] = True
            session.permanent = True      # PERMANENT_SESSION_LIFETIME まで保持
            print(f"[auth] ログイン成功 ({who})")
            return redirect(next_path)
        locked = _login_limiter.record_failure(who)
        print(f"[auth] ログイン失敗 ({who})" + (f" → {locked // 60}分ロック" if locked else ""))
        msg = ("パスワードが違います。しばらく待ってからお試しください"
               if locked else "パスワードが違います")
        return render_template('login.html', next=next_path, error=msg), 401

    return render_template('login.html', next=next_path, error=None)


@app.route('/logout', methods=['POST'])
def logout():
    """ログアウト（セッションを破棄してログイン画面へ）。"""
    session.pop('authed', None)
    return redirect(url_for('login'))


@app.route('/api/auth/unlock', methods=['POST'])
def api_auth_unlock():
    """ログイン失敗ロックを解除する（コントロールパネルから呼ばれる）。

    **127.0.0.1 限定**。ロックはメモリ上なのでミクを再起動しても消えるが、
    再起動せずに解除できる手段として用意する（ユーザー要望 2026-08-06）。
    """
    if not is_local_request(request.remote_addr, request.headers):
        return jsonify({"success": False, "error": "この操作はこの PC からのみ実行できます"}), 403
    count = _login_limiter.reset_all()
    print(f"[auth] ログインロックを解除しました（対象 {count} 件）")
    return jsonify({"success": True, "unlocked": count})


@app.route('/favicon.ico')
def favicon():
    """ブラウザのタブに出すアイコン。

    デスクトップのショートカットと同じ「ミクの姿から作ったアイコン」を出す。
    セットアップが `runtime\\miku_icon.ico` を作るまでは同梱の簡易マークで代用する。
    """
    icon = GENERATED_ICON if GENERATED_ICON.exists() else BUNDLED_ICON
    if not icon.exists():
        return '', 404
    return send_file(str(icon), mimetype='image/x-icon')


@app.route('/live2d')
def live2d_index():
    """Live2D index.htmlを返す。

    未導入のときはチャット画面の iframe に案内を出す（開発者向けエラーを見せない）。
    Live2D は必須だが、無くても会話自体は成立するため 200 で返す（入れ直しへ誘導する）。
    """
    if not LIVE2D_DIST_DIR.exists():
        print(f"[WARN] Live2D が未導入のため案内を表示します（dist なし: {LIVE2D_DIST_DIR}）")
        return send_from_directory(str(TEMPLATES_DIR), "live2d_unavailable.html")
    return send_from_directory(str(LIVE2D_DIST_DIR), "index.html")

@app.route('/live2d/<path:path>')
def live2d_static_files(path: str):
    """Live2D distフォルダの静的ファイルを配信"""
    return send_from_directory(str(LIVE2D_DIST_DIR), path)


# --- 曲の配信・再生（Phase 2 歌唱） --------------------------------------
# 曲フォルダには原曲音源や分離stem（著作権素材）も入るため、**出すファイルを白名簿で限定**する。
# 鳴らすのに要るのは「完成品」「口の動き」「作りかけの音」の3つだけ。
#
# ★**完成品の名前は song_service と同じ1か所（`song_service/song_files.py`）から引く**（2026-10-03・§7-1）。
#   完成品は `master_<形>_v{N}.wav`（歌もの・アカペラ・インスト）で、名前に形が入る。判定は
#   `song_files.is_master`（**形と桁を固定した正規表現**。前方一致に緩めると `..%2F..` のような細工に道を開く）。
#   V6 の生の書き出し（`vocal_v{N}.wav`）・昔の完成ミックスの写し（固定名からのコピーで名前と中身が食い違った・2026-08-17）は出さない。
# ★**作りかけの音（`preview.wav`）も出す**（2026-10-03）。2026-09-23 から play_song が `preview` で予約していたのに、
#   ここに無かったので**ブラウザの取得が 404 で、作りかけは鳴っていなかった**（白名簿の見落とし）。
#   中身は楽譜から作った音（曲フォルダの外の素材ではない）。
# 口の動きは版つき（`lipsync_v{N}.json`）だけ。固定名は作業中しか無い。
from song_service import song_files  # noqa: E402  標準ライブラリだけで書かれた名前の決まり（miku/src 起点で読める）

_SONG_PUBLIC_FILES = (song_files.PREVIEW,)
# ★**fullmatch で全体を照らす**（2026-10-03）。`match` と `$` だけだと末尾が改行の名前（`%0A`）が通る。
#   版の桁は完成品（`song_files` の `_v[1-9]\d*`）とそろえて制限しない
_SONG_PUBLIC_LIPSYNC = re.compile(r"lipsync_v[1-9]\d*\.json")


def _song_public(filename: str) -> bool:
    """配信してよい曲フォルダのファイル名か（完成品・版つきの口の動き・作りかけの音）。"""
    return (filename in _SONG_PUBLIC_FILES or song_files.is_master(filename)
            or bool(_SONG_PUBLIC_LIPSYNC.fullmatch(filename)))


@app.route('/songs/<song_id>/<filename>')
def song_file(song_id: str, filename: str):
    """曲の再生に必要なファイルだけを配信する（完成品・口の動き・作りかけの音）。

    song_id は利用者の指示から来うるので、**フォルダ名として素直な形だけを許す**
    （パス区切り・`..` を弾き、解決後も output/ の中にあることを確かめる＝二重の歯止め）。
    """
    if not _song_public(filename):
        return '', 404
    from agent.config import get_songs_dir
    out_root = get_songs_dir().resolve()
    if "/" in song_id or "\\" in song_id or song_id in ("", ".", ".."):
        return '', 404
    target = (out_root / song_id / filename).resolve()
    if not target.is_file() or out_root not in target.parents:
        return '', 404
    return send_file(str(target),
                     mimetype='audio/wav' if filename.endswith('.wav') else 'application/json')


def reserve_song_playback(song_id: str, title: str, version: Optional[int] = None,
                          *, audio: Optional[str] = None, mode: str = "sing") -> Optional[str]:
    """ブラウザへ曲の再生を**予約**する（すぐには鳴らない）。

    実際の再生開始はブラウザ側（songPlayer.js）が `voice:end`＝ミクが喋り終わった時に行う。
    サーバーは PCM を「送り終えた」時刻しか知らず、**実際に音が鳴り終わった時刻を知らない**ため、
    ここで鳴らすとミクの声と曲が重なる。

    version: 版番号。指定すると **その版の完成品（`master_<形>_v{N}.wav`・名前の決まりは `song_files`）と
        `lipsync_v{N}.json`（口）をセットで**渡す。歌と口を必ず同じ版に揃えるため、片方だけ版を変えることはしない
        （固定名の2つが別々の版を指し、口と歌がズレる事故が実際に起きた・2026-08-16）。
        None のときは鳴らせる版のうち最新の完成品（play_song は必ず版を決めて渡すので、念のための道）。
    audio: 鳴らすものを名指しする（作りかけの `preview.wav`）。このときは完成品を探さない。
    mode: `sing`（歌う・口が動く）／`play`（歌の入らない曲を流す・口は動かさない・2026-10-03 F8①）／
        `preview`（作りかけを聴かせる）。**完成品がインストなら、渡された mode によらず `play`**（歌の無い曲で口を動かさない）。

    Returns:
        **実際に決めた mode**（`sing`／`play`／`preview`）。予約を送れなければ None
        （ブラウザが繋がっていない・鳴らす完成品が無い＝ツール側が正直に断れる）。
        ★mode を返すのは、ここで `sing` が `play` に落ちうるから（歌ものでも口の形のファイルが無いとき）。
        play_song はこの値で言い方を選ぶ——「歌うね」と言って口が動かない食い違いを作らない（2026-10-03）。
    """
    target_sid = get_active_socketio_sid()
    if not target_sid:
        return None
    from agent.config import get_songs_dir             # 曲の置き場は1か所に寄せてある（配信と同じ）
    song_dir = get_songs_dir() / song_id
    lip: Optional[str] = None
    if audio:                                              # 鳴らすものが指定されている（作りかけなど）
        if version:
            lip = song_files.versioned(song_files.LIPSYNC, version)
    else:
        # 版は play_song が必ず決めて渡す（鳴らせる版のうち最新、または指定された版）
        got = (song_files.master_of(song_dir, version) if version
               else next(reversed(song_files.masters(song_dir)), None))
        if got is None:
            print(f"[song] 鳴らす完成品が無い: {title}（{song_id} / v{version}）")
            return None
        audio, version = got.path.name, got.version
        if got.form == "instrumental":
            mode = "play"                                  # 歌の入らない曲は歌わずに流す（口の形も送らない）
        if mode != "play":
            lip = song_files.versioned(song_files.LIPSYNC, version)

    # ★**口を動かすかは「口の形のデータがあるか」で決まる**（2026-09-23 本人指示）。
    #   フラグで指示すると、データが無いのに「歌う」と言ってしまう食い違いが起きる。
    #   **実在を見て、無ければ口の情報を送らない**＝ミクは歌わず、曲を流すだけになる。
    has_lip = bool(lip) and (song_dir / lip).exists()
    if mode == "sing" and not has_lip:
        mode = "play"                                      # 口が無ければ、歌ではなく流すだけ
    payload = {
        'song_id': song_id,
        'title': title,
        'url': f'/songs/{song_id}/{audio}',
        'mode': mode,
    }
    if has_lip:
        payload['lipsync_url'] = f'/songs/{song_id}/{lip}'
    socketio.emit('song_play', payload, to=target_sid)
    print(f"[song] 再生を予約: {title}（{song_id} / {audio} / {payload['mode']}"
          f"{'' if has_lip else '・口の形なし'}） → {target_sid}")
    return mode

# iframe内からの相対パスアクセス対応
@app.route('/assets/<path:path>')
def live2d_assets(path: str):
    """Live2D assetsフォルダの静的ファイルを配信"""
    return send_from_directory(str(LIVE2D_DIST_DIR / "assets"), path)

@app.route('/Core/<path:path>')
def live2d_core(path: str):
    """Live2D Coreフォルダの静的ファイルを配信"""
    return send_from_directory(str(LIVE2D_DIST_DIR / "Core"), path)

@app.route('/Resources/<path:path>')
def live2d_resources(path: str):
    """Live2D Resourcesフォルダの静的ファイルを配信"""
    return send_from_directory(str(LIVE2D_DIST_DIR / "Resources"), path)

# =============================================================================
# メインルート
# =============================================================================

@app.route('/', methods=['GET'])
def index():
    """メインページを表示"""
    return render_template('index.html')

@app.route('/ping', methods=['POST'])
def ping():
    """疎通確認用のエンドポイント"""
    try:
        if not initialization_success:
            return jsonify({
                'success': False,
                'error': "システムの初期化に失敗しました。ログを確認してください。"
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Connection established'
        })
    except Exception as e:
        print(f"Ping failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config', methods=['GET'])
def api_config():
    """フロントエンド向けの設定情報を返すエンドポイント。"""
    stt_rt_config = _get_active_stt_config()
    camera_config = settings.get('camera', {})

    auto_send_config = stt_rt_config.get('auto_send', {})
    return jsonify({
        'stt_enabled': stt_rt_config.get('enabled', False),
        'stt_realtime_port': stt_rt_config.get('port', 5006),
        'auto_send_enabled': auto_send_config.get('enabled', False),
        'auto_send_delay': auto_send_config.get('delay_seconds', 5),
        'camera_enabled': camera_config.get('enabled', False),
        'camera_buffer_size': camera_config.get('buffer_size', 3),
    })


@app.route('/api/conversation/recent', methods=['GET'])
def api_conversation_recent():
    """直近 N 発話を返す（マイクパネルの履歴表示用・2026-08-01）。

    KV の this_conversation（新形式=正式JSON配列）から speaker+text を持つ発話行だけを
    末尾 N 件抽出する。旧形式（移行前の疑似JSON）の場合は空リスト（履歴なし表示）。
    """
    from utils import conversation_log
    try:
        n = min(max(1, int(request.args.get('n', 5))), 20)
    except (TypeError, ValueError):
        n = 5
    try:
        kv_config = settings.get('services', {}).get('kv', {})
        kv_port = kv_config.get('port', 8010)
        resp = http_requests.get(f"http://127.0.0.1:{kv_port}/kv",
                                 params={'key': 'this_conversation'}, timeout=5)
        text = resp.text if resp.status_code == 200 else ""
    except Exception as e:
        print(f"[conversation/recent] KV 取得失敗: {e}")
        return jsonify({'utterances': []})
    from utils.perception import display_text
    records = conversation_log.tail_utterances(text, n=n, speech_only=True)
    # マスターの発話は、音声認識の区間・声の一致度・カメラの照合結果を外して本文だけ見せる（2026-10-10 本人）
    return jsonify({'utterances': [
        {'speaker': r.get('speaker'), 'time': r.get('time'),
         'text': display_text(r.get('text') or '') if r.get('speaker') == 'user' else r.get('text')}
        for r in records
    ]})

# =============================================================================
# 設定ページ・設定API
# =============================================================================

# 設定変更GUI（/settings・/api/settings）は 2026-08-02 にスタート画面へ移設した。
# 理由: 設定の反映は再起動時で、再起動はスタート画面（PC の前）でしか行えないため
# リモート編集は完結しない。一方 Gateway に置くと Cloudflare トンネル経由で
# **認証なしの設定画面が外から開ける**状態になる。移設によりその穴が消え、
# システム停止中でも設定を直せるようになった。
# 実装は control_panel/ui（画面）＋ control_panel/__main__.py の Api（読み書き）。
# ブラウザ向けの設定ページ（settings.html / settings_main.js / settings.css）は削除済み。

def _apply_voice_segments(text: str, voice_segments_raw: str) -> str:
    """話者照合の区間を反映したテキストを返す（整形の実体は utils/perception.py）。

    畳み込み注入は TurnHandle へ**テキストだけ**を積む経路なので、ここで織り込んでおく。
    そうしないと LLM にも会話ログにも素のテキストで渡り、「誰の声だったか」が
    最初の発話ぶんしか残らない（2026-08-12 に実機で発覚）。
    """
    if not voice_segments_raw:
        return text
    try:
        from utils import perception
        segments = json.loads(voice_segments_raw)
        if isinstance(segments, list) and segments:
            return perception.build_input_text(text, segments, [])
    except Exception as e:
        print(f"[入力] 区間の整形に失敗（素のテキストで続行）: {type(e).__name__}: {e}")
    return text


def _read_attachments(request) -> list:
    """添付ファイルを読み出して files_data 形式にする。"""
    files_data = []
    if 'file_upload[]' not in request.files:
        return files_data
    for file in request.files.getlist('file_upload[]'):
        if not (file and file.filename):
            continue
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        lower_filename = file.filename.lower()
        content_type = file.content_type if hasattr(file, 'content_type') else ''

        file_type = "image"
        file_type_label = "不明"
        if lower_filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg')) or str(content_type).startswith('image/'):
            file_type, file_type_label = "image", "画像"
        elif lower_filename.endswith(('.mp3', '.wav', '.ogg', '.m4a', '.aac', '.amr', '.mpga')) or str(content_type).startswith('audio/'):
            file_type, file_type_label = "audio", "音声"
        elif lower_filename.endswith(('.mp4', '.mov', '.mpeg', '.webm')) or str(content_type).startswith('video/'):
            file_type, file_type_label = "video", "映像"

        print(f"添付ファイル: {file.filename}")
        print(f"  - 種類: {file_type_label}")
        print(f"  - サイズ: {file_size} バイト ({file_size / 1024:.1f} KB)")
        print(f"  - MIME タイプ: {content_type or '不明'}")

        files_data.append({
            'filename': file.filename,
            'type': file_type,
            'size': file_size,
            'content_type': content_type or '不明',
            'content': file.read(),
        })
        file.seek(0)
    return files_data


class UserInput:
    """ブラウザから届いた1回分の入力（`/generate` の**唯一の入口**）。

    生の `request.form` を各分岐が直接読むと、経路が増えたときに前処理の呼び忘れが起きる
    —— 2026-08-12 に実際に起きた（畳み込み経路だけ `voice_segments` が落ち、追加発話の
    一致度が会話ログに残らなかった）。ここを通せば、分岐がいくつ増えても整形は自動で効く。

    **`text` と `raw_text` の使い分け**（重要）:
      - `text`     … 声の区間を織り込み済み。**顔を伴わない経路**（畳み込み注入）で使う
      - `raw_text` … 生のまま。**通常送信**で使う。顔の照合結果は agent_main でしか
                     得られないため、あちらで声と顔をまとめて整形する（二重整形を避ける）
    """

    def __init__(self, request) -> None:
        raw = request.form['input_text']
        self.raw_text: str = raw
        self.voice_segments_raw: str = request.form.get('voice_segments', '')
        self.text: str = _apply_voice_segments(raw.strip(), self.voice_segments_raw)
        self.client_session_id = request.form.get('clientSessionId', None)
        self.files_data: list = _read_attachments(request)

    @property
    def has_files(self) -> bool:
        return bool(self.files_data)

    @property
    def has_text(self) -> bool:
        return bool(self.raw_text.strip())


@app.route('/generate', methods=['POST'])
def generate():
    """テキスト入力から応答を生成するエンドポイント"""
    global _active_turn_handle
    try:
        if not initialization_success:
            return jsonify({
                'success': False,
                'error': "システムの初期化に失敗しました。ログを確認してください。"
            }), 500
        
        # 入力の読み取りはここ1箇所（生の request.form を分岐側で読まない）。
        # UserInput が声の区間を織り込むので、経路が増えても前処理の呼び忘れが起きない
        ui = UserInput(request)

        # --- P2 畳み込み: 進行中ターンがあれば新規ターンを起動せず注入する ---
        # 対象はテキストのみの入力（考え中の音声追加発話）。添付つきは従来どおり新規ターン。
        # try_inject はターン終了と lock で排他されており、閉じた直後なら False → 通常処理へ
        handle = _active_turn_handle
        if handle is not None and ui.has_text and not ui.has_files:
            arrival = time.strftime('%Y-%m-%d %H:%M:%S')
            # 整形済みの text を積む（顔はこの経路に無いので、声だけで確定できる）
            if handle.try_inject(arrival, ui.text):
                print(f"[FOLD] 進行中ターンへ追加発話を注入: {ui.text[:40]!r}")
                return jsonify({'success': True, 'folded': True})

        # セッションIDのログ出力とアクティブセッション設定
        if ui.client_session_id:
            print(f"クライアントセッションID: {ui.client_session_id}")
            set_active_session(ui.client_session_id)
        else:
            print("クライアントセッションIDが指定されていません")

        # ★新規ターンの TurnHandle は**ここで同期的に**登録する（2026-09-22）。
        # 以前は `_run_fish_audio_flow`（バックグラウンドスレッド側、しかも
        # `_prewarm_chapter()` 等の前処理のあと）で登録しており、ほぼ同時に届いた
        # 2件目の /generate がこの登録前に上の判定を通ってしまい、**畳み込まれるべき
        # 発話が独立した新規ターンとして並行実行**されていた（実機ログで確認。
        # ブラウザ側は同じチャプター番号が2系統から届いて衝突し、声が届かなかった）。
        # ここでリクエストスレッドが完了する前に確実にセットすることで、その隙間を無くす。
        from agent.turn_handle import TurnHandle
        wait_sec = _get_active_stt_config().get('auto_send', {}).get('delay_seconds', 5)
        turn_handle = TurnHandle(wait_timeout_sec=wait_sec)
        _active_turn_handle = turn_handle

        # バックグラウンドスレッドで処理（Fire & Forget）。
        # 通常経路は **raw_text** を渡す —— 顔の照合結果は agent_main でしか得られず、
        # あちらで声と顔をまとめて perception にかけるため（ここで整形すると二重になる）
        target_sid = get_active_socketio_sid()
        t = threading.Thread(
            target=_process_generate_async,
            args=(ui.raw_text, ui.files_data, target_sid, ui.client_session_id,
                  ui.voice_segments_raw, turn_handle),
            daemon=True,
        )
        t.start()

        return jsonify({'success': True})

    except Exception as e:
        print(f"Generate response failed: {str(e)}")
        # multipart 解析失敗（400 Bad Request）の切り分け用診断情報。
        # content_length はクライアントが宣言した送信サイズ（ヘッダー）なので、
        # 本文が壊れていても取得できる
        try:
            print(
                f"[Generate診断] content_type={request.content_type!r} "
                f"content_length={request.content_length} "
                f"user_agent={request.headers.get('User-Agent', '')[:120]!r}"
            )
        except Exception:
            pass
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# =============================================================================
# TTS Service からの Webhook 受信
# =============================================================================

@app.route('/webhook/task_done', methods=['POST'])
def webhook_task_done():
    """バックグラウンド作業（曲生成など）の完了通知を受け、ミク起点の自発話しかけ（mode=3）を起動する。

    **ここは「今すぐ気づいてほしい」の合図を受けるだけで、記録は持たない。**
    事実の保存は裏作業自身の責務（KV `background_tasks`・`utils/task_record.py`）。
    分けている理由: 記録を通知経路に相乗りさせると、**通知が抑制された瞬間に事実まで消える**。
    実際 2026-08-11 に夜間抑制で曲の完了が消え、ミクが「まだ作業中」と誤答した。
    いまは通知が届かなくても・抑制されても、記録は残り system_state 経由で次のターンに気づける。

    リクエスト: {"source": "song_service", "summary": "『〜』ができた。実歌声には vpr 書き出しが必要"}
    - **ローカル限定**（127.0.0.1 のみ）。外部プロセスからの誤起動・悪用を防ぐ
    - 夜間（自由行動の停止時間帯）は起動しない＝マスターを起こさない（記録は裏作業側にあるので消えない）
    - 会話ログには起点=assistant（ミク自身の裏作業）として残る（マスターの発言にしない）
    """
    if request.remote_addr not in ('127.0.0.1', '::1'):
        return jsonify({"status": "error", "message": "ローカルからのみ受け付けます"}), 403
    data = request.get_json(silent=True) or {}
    source = str(data.get('source', 'background'))
    summary = str(data.get('summary', '')).strip()
    if not summary:
        return jsonify({"status": "error", "message": "summary が空です"}), 400

    if is_in_free_mode_quiet_hours():
        print(f"🌙 完了通知({source})を受信したが夜間のため自発話しかけは行わない: {summary[:60]}")
        return jsonify({"status": "deferred", "message": "夜間のため次の会話で報告"}), 200

    # 依頼から数十分経っているのが普通なので、マスターは席を外している前提で「メールで知らせる」まで促す
    note = (f"（自分が裏で進めていた作業が完了した・{source}）{summary}\n"
            f"※依頼から時間が経っていて、マスターは席を外している可能性が高い。"
            f"対面していないので声では届かないから、伝える価値のある内容なら send_email で知らせること。"
            f"次にやってほしいこと（あれば）もメール本文に具体的に書く。")
    threading.Thread(target=trigger_proactive_action, args=(note,), daemon=True).start()
    print(f"📣 完了通知({source})→ 自発話しかけを起動: {summary[:60]}")
    return jsonify({"status": "ok"}), 200


# =============================================================================
# ステータスAPI
# =============================================================================

@app.route('/api/status', methods=['GET'])
def api_status():
    """サーバーステータス確認"""
    return jsonify({
        "status": "running",
        "initialized": initialization_success,
        "clients": len(connected_clients),
        "active_session": active_session_id,
        "templates_dir": str(TEMPLATES_DIR),
        "templates_exists": TEMPLATES_DIR.exists(),
        "live2d_dist_dir": str(LIVE2D_DIST_DIR),
        "live2d_exists": LIVE2D_DIST_DIR.exists()
    })


@app.route('/api/system/shutdown', methods=['POST'])
def api_system_shutdown():
    """システムを停止する（コントロールパネルの「停止」から呼ばれる）。

    **ローカルからのみ受理する**。Gateway は Cloudflare トンネルで外部公開されうるため、
    外部から停止させられないようにする必要がある。
    """
    if request.remote_addr not in ('127.0.0.1', '::1', 'localhost'):
        print(f"[shutdown] ローカル以外からの停止要求を拒否しました: {request.remote_addr}")
        return jsonify({"error": "forbidden", "reason": "local only"}), 403

    def _shutdown():
        # レスポンスを返しきってから終了する（呼び出し側が「受理」を確認できるように）
        time.sleep(0.3)
        try:
            cleanup()
        except Exception as e:
            print(f"[shutdown] クリーンアップでエラー: {e}")
        print("停止要求を受け取りました。システムを終了します")
        sys.stdout.flush()
        # 各サービスは daemon スレッドで動いており通常の return では止まらないため、
        # クリーンアップ後にプロセスごと終了する
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify({"status": "stopping"})

# =============================================================================
# STT ウォームアップ接続管理
# =============================================================================

def _run_stt_warmup(loop: asyncio.AbstractEventLoop, stop_event: threading.Event):
    """STT ウォームアップ接続のバックグラウンドスレッド。"""
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_stt_warmup_coro(stop_event))
    except Exception as e:
        print(f"[STT Warmup] スレッドエラー: {e}")
    finally:
        loop.close()
        print("[STT Warmup] スレッド終了")


async def _stt_warmup_coro(stop_event: threading.Event):
    """STT Realtime に接続し、model_loading/model_ready を全ブラウザへ転送する。
    接続が切れた場合は3秒後にリトライする。stop_event がセットされたら終了。
    """
    global _stt_model_status
    import websockets as _ws

    stt_rt_config = _get_active_stt_config()
    port = stt_rt_config.get('port', 5006)
    url = f"ws://localhost:{port}/ws"

    while not stop_event.is_set():
        try:
            async with _ws.connect(url, ping_interval=None) as ws:
                print(f"[STT Warmup] 接続: {url}")
                async for raw in ws:
                    if stop_event.is_set():
                        break
                    try:
                        data = json.loads(raw)
                        t = data.get("type")
                        if t == "model_loading":
                            _stt_model_status = "loading"
                            socketio.emit("stt_model_loading", {})
                            print("[STT Warmup] model_loading → 全ブラウザへ通知")
                        elif t == "model_ready":
                            _stt_model_status = "ready"
                            socketio.emit("stt_model_ready", {})
                            print("[STT Warmup] model_ready → 全ブラウザへ通知")
                    except Exception:
                        pass
        except Exception as e:
            if not stop_event.is_set():
                print(f"[STT Warmup] 接続失敗、3秒後リトライ: {e}")
                await asyncio.sleep(3)

    print("[STT Warmup] 停止")


def _start_stt_warmup():
    """最初のブラウザが接続した時にウォームアップ接続を開始する。"""
    global _stt_warmup, _stt_model_status
    stt_rt_enabled = _get_active_stt_config().get('enabled', False)
    if not stt_rt_enabled:
        return
    with _stt_warmup_lock:
        if _stt_warmup is not None:
            return  # 既に起動中
        stop_event = threading.Event()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_stt_warmup,
            args=(loop, stop_event),
            daemon=True,
            name="STT-Warmup",
        )
        _stt_warmup = {'loop': loop, 'thread': thread, 'stop_event': stop_event}
        _stt_model_status = "unknown"
        thread.start()
    print("[STT Warmup] 開始")


def _stop_stt_warmup():
    """最後のブラウザが切断した時にウォームアップ接続を停止する。"""
    global _stt_warmup, _stt_model_status
    with _stt_warmup_lock:
        w = _stt_warmup
        _stt_warmup = None
    if w:
        w['stop_event'].set()
        try:
            w['loop'].call_soon_threadsafe(w['loop'].stop)
        except Exception:
            pass
        _stt_model_status = "unknown"
        print("[STT Warmup] 停止要求")


# =============================================================================
# WebSocketイベントハンドラー（Socket.IOのみ）
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """WebSocket接続時の処理。未認証の接続は拒否する（HTTP と同じ保護をかける）"""
    # ★**他所のサイトからの接続を断る**（2026-09-20 追加）。
    #   Socket.IO は `cors_allowed_origins="*"` で**どのサイトからでも繋げた**ため、
    #   別オリジンのページが `response_ready` を受け取り、**会話を読めていた**（実測）。
    if not is_same_origin(request.headers.get('Origin'), request.headers.get('Host')):
        print(f"[SECURITY] 別サイトからの WebSocket 接続を拒否: "
              f"origin={request.headers.get('Origin')!r}")
        return False
    if not _is_authenticated():
        print(f"WebSocket接続を拒否（未認証）: {request.remote_addr}")
        return False        # False を返すと Socket.IO が接続を確立しない
    print(f"WebSocket接続: {request.sid}")
    connected_clients[request.sid] = {
        'connected_at': time.time(),
        'last_activity': time.time()
    }
    # 最初のブラウザ接続でSTTウォームアップを開始
    if len(connected_clients) == 1:
        _start_stt_warmup()
        # プロンプトの固定部を明示キャッシュへ先に載せておく（2026-08-30）。
        # **これが無いと、その会話の最初の発話だけ 2 秒余計に待たされる**
        # （キャッシュはループの頭で作られるため）。接続から最初の発話までは
        # タイピングでも音声入力でも数秒あるので、その間に作り終わる。
        # 中で別スレッドを立てるため、接続処理は待たされない
        from agent import context_cache
        context_cache.prewarm_async()
        # 既にready状態ならウォームアップから通知が来ないため即通知
        if _stt_model_status == "ready":
            socketio.emit('stt_model_ready', {}, to=request.sid)
    elif _stt_model_status == "loading":
        # 後から接続したブラウザにもロード中状態を即通知
        socketio.emit('stt_model_loading', {}, to=request.sid)
    elif _stt_model_status == "ready":
        # 後から接続したブラウザにもready状態を即通知
        socketio.emit('stt_model_ready', {}, to=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket切断時の処理"""
    global active_session_id, _current_fish_cancel

    socketio_session_id = request.sid
    print(f"WebSocket切断: {socketio_session_id}")
    
    client_session_id = session_mapping.get(socketio_session_id)
    
    if client_session_id:
        print(f"切断されたクライアントセッションID: {client_session_id}")
        
        if active_session_id == client_session_id:
            print(f"アクティブセッション {client_session_id} が切断されました")
            active_session_id = None
        
        del session_mapping[socketio_session_id]
        print(f"セッションマッピング削除: {socketio_session_id}")
    else:
        print(f"警告: 切断されたWebSocketセッション {socketio_session_id} に対応するクライアントセッションIDが見つかりません")
    
    if socketio_session_id in connected_clients:
        del connected_clients[socketio_session_id]

    # 会話中リアクションの sid 単位状態をクリーンアップ
    _typing_gates.pop(socketio_session_id, None)
    with _reaction_emit_lock:
        _reaction_cooldown_until.pop(socketio_session_id, None)

    # 最後のブラウザが切断したらSTTウォームアップを停止
    if len(connected_clients) == 0:
        _stop_stt_warmup()

    # Fish Audio PCM ストリーミング中で、**送り先の端末が切れたときだけ**キャンセルする（2026-10-14）。
    # 別の端末の切断では止めない（送り先が空＝全員宛のターンも止めない）
    if _current_fish_cancel is not None:
        if _current_fish_target_sid and socketio_session_id == _current_fish_target_sid:
            _current_fish_cancel.set()
            print(f"[disconnect] Fish Audio キャンセル (sid={socketio_session_id}・送り先の端末が切れた)")
        else:
            print(f"[disconnect] 別の端末の切断なので声は止めない (sid={socketio_session_id}・送り先={_current_fish_target_sid})")

    # リアルタイムSTTストリーミング中なら停止
    _flush_chunk_stats(socketio_session_id, '切断')
    _chunk_orphans.pop(socketio_session_id, None)
    stream_session = _stream_sessions.pop(socketio_session_id, None)
    if stream_session:
        stream_session['loop'].call_soon_threadsafe(stream_session['aio_queue'].put_nowait, None)
        print(f"[disconnect] STTストリーミング停止 (sid={socketio_session_id})")

@socketio.on('session_activate')
def handle_session_activate(data):
    """セッションアクティブ化イベント"""
    try:
        client_session_id = data.get('sessionId', request.sid)
        socketio_session_id = request.sid
        
        print(f"セッションアクティブ化（クライアントセッションID）: {client_session_id}")
        print(f"WebSocketセッションID: {socketio_session_id}")
        
        session_mapping[socketio_session_id] = client_session_id
        print(f"セッションマッピング追加: {socketio_session_id} -> {client_session_id}")
        
        set_active_session(client_session_id)
        
        emit('session_activated', {
            'sessionId': client_session_id,
            'timestamp': time.time(),
            'message': 'セッションがアクティブになりました'
        })
        
    except Exception as e:
        print(f"セッションアクティブ化エラー: {str(e)}")
        emit('error', {'message': 'セッションアクティブ化に失敗しました'})

@socketio.on('inactivity_reminder_request')
def handle_inactivity_reminder_request(data):
    """無応答リマインダー要求イベント"""
    global active_session_id
    
    try:
        socketio_session_id = request.sid
        client_session_id = session_mapping.get(socketio_session_id)
        
        unresponsive_time = data.get('timeout', 300)
        
        print(f"🔔 無応答リマインダー要求を受信: セッション {client_session_id}, 無応答時間: {unresponsive_time}秒 (SID={socketio_session_id})")

        if client_session_id != active_session_id:
            print(f"非アクティブセッション {client_session_id} からのリマインダー要求を無視 (アクティブ: {active_session_id})")
            return
        
        # Phase 7 切り戻し用: 旧 Dify 実装
        # from agent_main_dify import process_agent_request
        # from agent_main_dify_stream import process_agent_request
        from agent_main import process_agent_request

        # カメラフレーム（base64）をfiles_dataに変換
        camera_files_data = []
        for i, frame_b64 in enumerate(data.get('camera_frames', [])):
            try:
                encoded = frame_b64.split(',', 1)[1] if ',' in frame_b64 else frame_b64
                image_bytes = base64.b64decode(encoded)
                camera_files_data.append({
                    'filename': f'camera_{i}.jpg',
                    'type': 'image',
                    'size': len(image_bytes),
                    'content_type': 'image/jpeg',
                    'content': image_bytes
                })
            except Exception as ex:
                print(f"カメラフレームデコード失敗 [{i}]: {ex}")

        if camera_files_data:
            print(f"[inactivity] カメラ画像 {len(camera_files_data)} 枚を添付")

        # process_agent_request の戻り値（確定済み全文）を fish_q に投入する方式。
        # on_object_complete の partial output（LLM ストリーミング途中の断片）は使わない。
        # inactivity_reminder emit 後（ブラウザの forceStop 完了後）に Fish Audio スレッドを
        # 起動することで chapter=100 が forceStop に殺されないようにする。
        # chapter=100 は通常チャット (0,1,2,...) と被らない固定値。
        fish_q: queue.Queue = queue.Queue()

        try:
            reminder_response = process_agent_request(
                input_text="（応答なし）",
                files_data=camera_files_data,
                mode=1,
                unresponsive_time=str(unresponsive_time),
            )

            response_text = reminder_response.get('response', 'お疲れ様。何かお手伝いできることはある？')

            # LLM 確定後に fish_q へ投入（on_object_complete の partial ではなく確定値を使う）
            full_tts_text = strip_for_tts(response_text)
            print(f"[inactivity] fish_q 投入: {len(full_tts_text)}字")
            if full_tts_text:
                fish_q.put(full_tts_text)
            fish_q.put(None)

        except Exception as e:
            print(f"無応答リマインダー生成エラー: {str(e)}")
            response_text = "お疲れ様。何かお手伝いできることはある？"
            reminder_response = {}
            fish_q.put(None)  # エラー時もストリームを閉じる

        print(f"無応答リマインダーを送信: {response_text}")
        free_mode_reset_event.set()

        # inactivity_timeout = reminder_response.get('inactivity_timeout', -1)
        inactivity_timeout = -1
        motion = reminder_response.get('motion', '01')

        print(f"⏳ 無応答時はさらなる無応答リマインダーは開始しない")

        # inactivity_reminder を先に emit（ブラウザの forceStop を先に完了させる）
        socketio.emit('inactivity_reminder', {
            'response': response_text,
            'session_id': client_session_id,
            'timestamp': time.time(),
            'inactivity_timeout': inactivity_timeout,
            'motion': motion
        })

        # emit 後に Fish Audio スレッドを起動（fish_q には確定全文 + None が積み済み）
        def _reminder_tts_stream() -> None:
            sid = get_active_socketio_sid()
            if not sid:
                print("[WARN] リマインダーTTS: アクティブな Socket.IO SID が見つかりません")
                return
            print(f"[inactivity] TTS 開始 → SID={sid}, chapter=100")
            _run_fish_audio_from_queue(fish_q, sid, chapter_num=100)
        threading.Thread(target=_reminder_tts_stream, daemon=True).start()
        
    except Exception as e:
        print(f"無応答リマインダー要求処理エラー: {str(e)}")

# =============================================================================
# STT関連 Socket.IOイベント
# =============================================================================

# ---------------------------------------------------------------------------
# リアルタイムSTT WebSocketプロキシ
# ---------------------------------------------------------------------------

def _run_stt_stream(sid: str, aio_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """バックグラウンドスレッドでSTT Realtimeへの接続を管理する。"""
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_stt_stream_coro(sid, aio_queue))
    except Exception as e:
        print(f"[STT Stream] スレッドエラー (sid={sid}): {e}")
    finally:
        loop.close()
        _stream_sessions.pop(sid, None)
        print(f"[STT Stream] スレッド終了 (sid={sid})")


async def _stt_stream_coro(sid: str, aio_queue: asyncio.Queue):
    """STT Realtimeサービスに接続し、音声チャンクを転送・結果をブラウザへ返す。

    合わせて会話中リアクション判定を行う: interim / final テキストを ReactionGate（頻度制御）を
    通して filler_compose(mode="interim") に渡し、motion が返れば reaction_motion で
    ブラウザへ即時配信する（final でも声は出さない。詳細: docs/plan/conversation_reaction_study.md）。
    """
    import websockets

    from reaction_gate import ReactionGate
    from agent.subagents.filler_compose import run_filler_compose

    stt_rt_config = _get_active_stt_config()
    port = stt_rt_config.get('port', 5006)
    url = f"ws://localhost:{port}/ws"

    reaction_config = settings.get('conversation_reaction', {})
    reaction_enabled = reaction_config.get('enabled', True)
    gate = ReactionGate(cooldown_secs=float(reaction_config.get('cooldown_seconds', 5)))

    async def judge_reaction(text: str, source: str = "stt_interim"):
        """STT テキストからリアクション motion を判定し、あればブラウザへ送る。

        mode は常に "interim"（モーションのみ・声なし）。final トリガーでも声を出さないのは、
        声のフィラーは送信直後の Phase 6 経路が担当しており二重になるため。
        """
        reacted = False
        try:
            out = await run_filler_compose(text, mode="interim")
            reacted = _try_emit_reaction_motion(sid, out.motion, source)
        except Exception as e:
            print(f"[Reaction] 判定失敗 (sid={sid}): {e}")
        finally:
            gate.on_judge_done(reacted)

    async def send_loop(ws):
        while True:
            item = await aio_queue.get()
            if item is None:
                await ws.send(json.dumps({"type": "stop"}))
                break
            await ws.send(item)

    async def recv_loop(ws):
        async for raw in ws:
            try:
                data = json.loads(raw)
                t = data.get("type")
                if t == "interim":
                    text = data.get("text", "")
                    socketio.emit("stt_interim", {"text": text}, to=sid)
                    # 相手が話している間に返答の方針を立てておく（2026-08-30）。
                    # **走行中・上限到達なら何もしない**ので、interim のたびに呼んでよい。
                    # 失敗しても会話は成立する（従来どおり1段階で作るだけ）
                    try:
                        from agent import plan_stage
                        from agent_main import build_plan_prompt
                        plan_stage.maybe_start(text, build_plan_prompt)
                    except Exception as e:
                        print(f"[plan_stage] 起動を見送った: {e!r}")
                    if reaction_enabled and gate.should_judge(text):
                        asyncio.create_task(judge_reaction(text.strip()))
                elif t == "final":
                    text = data.get("text", "")
                    # 発話確定時にもモーション相槌を判定する（声なし。声は送信直後のフィラー経路が担当）。
                    # 直前の interim で同等テキストを判定済みなら Gate の成長チェックが自動スキップし、
                    # 連発は _try_emit_reaction_motion の sid 共有クールダウンが防ぐ
                    if reaction_enabled and gate.should_judge(text):
                        asyncio.create_task(judge_reaction(text.strip(), source="stt_final"))
                    gate.on_final()
                    # 発話が確定した。次の発話に備えて段階1の起動回数を戻す
                    # （**立った方針は消さない** —— これから段階2が使う）
                    try:
                        from agent import plan_stage
                        plan_stage.reset_starts()
                    except Exception:
                        pass
                    stt_final_payload = {
                        "text": text,
                        "confidence": data.get("confidence", 0),
                    }
                    # Smart Turn 終話判定（stt.smart_turn.enabled 時のみ STT サービスが付与）。
                    # true=言い終わった → ブラウザは固定タイマーなしで即送信 / false=続きを待つ
                    if "turn_complete" in data:
                        stt_final_payload["turn_complete"] = data["turn_complete"]
                    # 話者照合の区間ごとの一致度（声が登録済みのときだけ STT サービスが付与）。
                    # **ここで詰め替えているので、素通しの指定漏れがそのまま欠落になる**
                    # （2026-08-12: 前処理化のとき STT 側とブラウザ側は実装したのに、
                    #   中継するここだけ漏れていて、区間が一度もブラウザへ届いていなかった）
                    if "segments" in data:
                        stt_final_payload["segments"] = data["segments"]
                    socketio.emit("stt_final", stt_final_payload, to=sid)
            except Exception:
                pass

    try:
        print(f"[STT Stream] 接続: {url} (sid={sid})")
        async with websockets.connect(url) as ws:
            send_task = asyncio.create_task(send_loop(ws))
            recv_task = asyncio.create_task(recv_loop(ws))
            _, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    except Exception as e:
        print(f"[STT Stream] エラー (sid={sid}): {e}")
    finally:
        print(f"[STT Stream] 切断 (sid={sid})")


@socketio.on('audio_stream_start')
def handle_audio_stream_start():
    """リアルタイムSTTストリーミング開始。STT Realtimeへの接続を確立する。"""
    sid = request.sid
    _flush_chunk_stats(sid, 'ストリーム再開')
    orp = _chunk_orphans.pop(sid, None)
    if orp and orp['dropped']:
        print(f"[audio_stream_start] 直前までに破棄した音声: "
              f"{orp['dropped']}チャンク / {orp['bytes']:,}バイト (sid={sid})")
    if sid in _stream_sessions:
        print(f"[audio_stream_start] 既存セッション上書き (sid={sid})")
        old = _stream_sessions.pop(sid)
        old['loop'].call_soon_threadsafe(old['aio_queue'].put_nowait, None)

    loop = asyncio.new_event_loop()
    aio_queue = asyncio.Queue()
    thread = threading.Thread(
        target=_run_stt_stream,
        args=(sid, aio_queue, loop),
        daemon=True,
    )
    _stream_sessions[sid] = {'loop': loop, 'aio_queue': aio_queue, 'thread': thread}
    thread.start()
    print(f"[audio_stream_start] ストリーミング開始 (sid={sid})")


def _flush_chunk_stats(sid: str, reason: str) -> None:
    """溜めていた受信集計を1行にまとめて出す（発話の区切り・停止・切断で呼ぶ）。"""
    st = _chunk_stats.pop(sid, None)
    if not st or not st['chunks']:
        return
    print(f"[audio_chunk] 受信 {st['chunks']}チャンク / {st['bytes']:,}バイト / "
          f"{st['last'] - st['first']:.1f}秒 ({reason}, sid={sid})")


@socketio.on('audio_stream_chunk')
def handle_audio_stream_chunk(data):
    """ブラウザからの音声チャンク (PCM Int16 binary) をSTT Realtimeに転送する。"""
    sid = request.sid
    session = _stream_sessions.get(sid)
    audio_bytes = bytes(data.get('audio', b''))
    now = time.time()

    if not session:
        # **黙って捨てない**。ここに落ちると「録音できているのに文字起こしが返らない」
        # という症状になり、ブラウザ側のログだけでは原因を特定できない
        # （2026-08-16: 2ターン目の発話 17.9 秒が丸ごと消えた件の調査で必要になった）。
        # 100ms ごとに来るので、警告は 1 秒に 1 回へ間引く
        orp = _chunk_orphans.setdefault(sid, {'dropped': 0, 'bytes': 0, 'warned_at': 0.0})
        orp['dropped'] += 1
        orp['bytes'] += len(audio_bytes)
        if now - orp['warned_at'] >= 1.0:
            orp['warned_at'] = now
            print(f"[audio_chunk][WARN] STTセッション不在のため破棄: "
                  f"累計 {orp['dropped']}チャンク / {orp['bytes']:,}バイト (sid={sid})")
        return

    if not audio_bytes:
        return

    try:
        session['loop'].call_soon_threadsafe(session['aio_queue'].put_nowait, audio_bytes)
    except RuntimeError as e:
        # loop が閉じている＝停止済みのセッションが辞書に残ったまま。
        # `audio_stream_stop` はセッションを辞書から消さないので、stop 後に start が
        # 届かないとこの状態になり、**音声が静かに失われる**（従来は例外も握られていた）
        print(f"[audio_chunk][WARN] 停止済みセッションへの転送に失敗: {e} (sid={sid})")
        _stream_sessions.pop(sid, None)
        _flush_chunk_stats(sid, '転送失敗')
        return

    # 受信量の集計。1 秒以上途切れたら「別の発話」とみなして前の分を出す
    st = _chunk_stats.get(sid)
    if st and now - st['last'] > 1.0:
        _flush_chunk_stats(sid, '発話の区切り')
        st = None
    if st is None:
        st = {'chunks': 0, 'bytes': 0, 'first': now, 'last': now}
        _chunk_stats[sid] = st
    st['chunks'] += 1
    st['bytes'] += len(audio_bytes)
    st['last'] = now


@socketio.on('audio_stream_stop')
def handle_audio_stream_stop():
    """リアルタイムSTTストリーミング停止。"""
    sid = request.sid
    _flush_chunk_stats(sid, 'ストリーム停止')
    session = _stream_sessions.get(sid)
    if session:
        session['loop'].call_soon_threadsafe(session['aio_queue'].put_nowait, None)
        print(f"[audio_stream_stop] ストリーミング停止 (sid={sid})")
    else:
        # セッションが無い状態での stop。従来は完全に無音だったため、
        # 「stop を送ったのにログに出ない」＝ start/stop の取りこぼしを追えなかった
        print(f"[audio_stream_stop][WARN] セッション不在のまま停止要求 (sid={sid})")


# ---------------------------------------------------------------------------
# テキスト入力中リアクション（会話中リアクションのタイピング版）
# ---------------------------------------------------------------------------

_typing_gates: dict = {}  # {socketio_sid: ReactionGate}


# ---------------------------------------------------------------------------
# ブラウザ挙動ログの転送受け口（remoteLog.js）
#
# スマホ実機ではコンソールが見られず、タイミング系の偶発バグの記録が残らないため、
# タグ付きコンソールログを常時受けて日別ファイルへ追記する（検知→診断→修正手順の検知部分。
# 判断の記録: docs/dev/decisions.md 2026-08-04）。
# ---------------------------------------------------------------------------

_browser_log_lock = threading.Lock()


def _append_browser_log(session_short: str, lines) -> None:
    """ブラウザから届いたログ行を miku/runtime/browser_YYYYMMDD.log へ追記する。"""
    if not isinstance(lines, list) or not lines:
        return
    session_short = re.sub(r'[^0-9a-zA-Z-]', '', str(session_short))[:8] or 'unknown'
    path = MIKU_DIR / "runtime" / f"browser_{datetime.now().strftime('%Y%m%d')}.log"
    out = []
    for entry in lines[:200]:  # 1バッチ上限（暴走・悪意ある大量送信の抑制）
        if not isinstance(entry, dict):
            continue
        t = str(entry.get('t', ''))[:16]
        level = str(entry.get('level', 'log'))[:5]
        text = str(entry.get('text', ''))[:500].replace('\n', ' ')
        out.append(f"{t} [{session_short}] [{level}] {text}\n")
    if not out:
        return
    with _browser_log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.writelines(out)


@socketio.on('client_log')
def handle_client_log(data):
    """ブラウザログの通常経路（3秒ごとのバッチ送信）。"""
    data = data or {}
    _append_browser_log(data.get('session', ''), data.get('lines'))


@app.route('/api/client_log', methods=['POST'])
def api_client_log():
    """ブラウザログの離脱時経路（sendBeacon。Socket.IO が使えない瞬間の補完）。"""
    if request.content_length and request.content_length > 128 * 1024:
        return jsonify({"success": False, "error": "too large"}), 413
    data = request.get_json(force=True, silent=True) or {}
    _append_browser_log(data.get('session', ''), data.get('lines'))
    return jsonify({"success": True})


@socketio.on('user_speaking')
def handle_user_speaking(data):
    """ユーザーの発話開始・終了の一報（2026-08-11）。

    進行中ターンがあれば TurnHandle へ反映する。話し始めなら runner は生成を中断し、
    追加発話が届くまで確定を保留する（話している最中に返事を始めないため）。
    進行中ターンが無ければ何もしない（ブラウザ状態の複製ではなく、その時点の一報）。
    """
    handle = _active_turn_handle
    if handle is None:
        return
    speaking = bool((data or {}).get('speaking'))
    handle.set_user_speaking(speaking)
    print(f"[FOLD] ユーザー発話の一報: speaking={speaking}")


@socketio.on('text_interim')
def handle_text_interim(data):
    """テキスト入力中のドラフトを受けてリアクションを判定する。

    ブラウザは手止まり時にドラフトを送り、送信・クリア時は空文字を送る（→ ゲートをリセット）。
    判定は STT interim と同じ filler_compose(mode="interim") ＋ 共通クールダウンを使う。
    """
    sid = request.sid
    text = ((data or {}).get('text', '') or '')

    # 相手が入力している間にも返答の方針を立てておく（2026-08-30）。
    # **音声の interim と同じ扱い**——タイピングでも「送信する前に考えておく」は成り立つ。
    # 走行中・上限到達なら何もしないので、ドラフトが届くたびに呼んでよい。
    #
    # **リアクション判定より前に置く**。あちらは conversation_reaction の設定で
    # 無効にできるが、方針づくりはそれとは独立した機能なので、
    # 早期 return の後ろに置くと設定次第で動かなくなる
    try:
        from agent import plan_stage
        from agent_main import build_plan_prompt
        if text.strip():
            plan_stage.maybe_start(text, build_plan_prompt)
        else:
            plan_stage.reset_starts()      # 送信・クリアで次に備えて数え直す
    except Exception as e:
        print(f"[plan_stage] 起動を見送った: {e!r}")

    reaction_config = settings.get('conversation_reaction', {})
    if not reaction_config.get('enabled', True):
        return
    if not reaction_config.get('text_input_enabled', True):
        return

    from reaction_gate import ReactionGate

    gate = _typing_gates.get(sid)
    if gate is None:
        # タイピングは発話より遅いため、成長しきい値は STT より大きめ（8字）
        gate = ReactionGate(
            cooldown_secs=float(reaction_config.get('cooldown_seconds', 5)),
            min_chars=6,
            min_growth=8,
        )
        _typing_gates[sid] = gate

    if not text.strip():
        gate.on_final()  # 送信・入力クリアで次のメッセージに向けてリセット
        return
    if not gate.should_judge(text):
        return

    def _runner() -> None:
        reacted = False
        try:
            from agent.subagents.filler_compose import run_filler_compose
            out = asyncio.run(run_filler_compose(text.strip(), mode="interim"))
            reacted = _try_emit_reaction_motion(sid, out.motion, "text_typing")
        except Exception as e:
            print(f"[Reaction] テキスト判定失敗 (sid={sid}): {e}")
        finally:
            gate.on_judge_done(reacted)

    threading.Thread(target=_runner, daemon=True, name="text-reaction").start()



# =============================================================================
# クリーンアップ
# =============================================================================

def cleanup():
    """アプリケーション終了時のクリーンアップ処理"""
    print("アプリケーションをクリーンアップしています...")

# =============================================================================
# テストページ
# =============================================================================

@app.route('/test')
def test_page():
    """テスト用HTML"""
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Gateway Service Test</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        button { margin: 5px; padding: 10px 20px; }
        .section { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
        h3 { margin-top: 0; }
    </style>
</head>
<body>
    <h1>Gateway Service Test</h1>
    
    <div class="section">
        <h3>ステータス</h3>
        <pre id="status"></pre>
        <button onclick="getStatus()">更新</button>
    </div>
    
    <div class="section">
        <h3>Live2Dテスト</h3>
        <p><a href="/live2d" target="_blank">Live2Dページを開く</a></p>
    </div>
    
    <div class="section">
        <h3>メインページ</h3>
        <p><a href="/" target="_blank">メインページを開く</a></p>
    </div>
    
    <script>
        function getStatus() {
            fetch('/api/status').then(r => r.json()).then(data => {
                document.getElementById('status').textContent = JSON.stringify(data, null, 2);
            });
        }
        
        getStatus();
    </script>
</body>
</html>
"""

# =============================================================================
# メイン
# =============================================================================

def _gateway_port() -> int:
    """Gateway の待ち受けポート（settings.json の services.gateway.port。無い・読めないときは 5001）。"""
    try:
        return int(settings.get('services', {}).get('gateway', {}).get('port', 5001))
    except (TypeError, ValueError):
        return 5001


def main() -> None:
    """単体起動の入口。配布物では `.pyd` 化されるため、起動処理は関数に置く。"""
    print("=" * 60)
    print("  Gateway Service (Miku Agent Main Service)")
    print("=" * 60)
    
    # アプリケーション初期化
    initialize_app()
    
    print(f"Templates folder: {TEMPLATES_DIR}")
    print(f"Templates exists: {TEMPLATES_DIR.exists()}")
    print(f"Live2D dist folder: {LIVE2D_DIST_DIR}")
    print(f"Live2D exists: {LIVE2D_DIST_DIR.exists()}")
    print("")
    # 待ち受けポートは settings.json の services.gateway.port（無ければ 5001）。
    # 以前は 5001 に直書きで、設定を変えても効かなかった（別フォルダに2つ目を入れる用途で問題になる・2026-10-22）
    port = _gateway_port()
    print(f"Open http://localhost:{port} for main application")
    print(f"Open http://localhost:{port}/settings for settings page")
    print(f"Open http://localhost:{port}/live2d for Live2D")
    print(f"Open http://localhost:{port}/test for test page")
    print("=" * 60)
    
    # 終了時のクリーンアップ
    import atexit
    atexit.register(cleanup)
    
    # サーバー起動
    socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
