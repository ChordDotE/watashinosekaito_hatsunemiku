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

"""ミクのシステムの起動・停止・状態把握（コントロールパネルの実処理）。

このモジュールは **監督役（supervisor）** として振る舞う:
  - Gateway を子プロセスとして起動し、標準出力をログファイルへ流す
  - cloudflared（外部アクセス用の外部ソフト）を必要に応じて起動する
  - 起動したプロセスの PID を runtime/running.json に記録する

PID を記録するのは、**コントロールパネルを閉じたり異常終了したりしても、
開き直したときに稼働中のシステムを見つけて操作できるようにする**ため
（親子関係に依存しない）。停止も PID ではなく Gateway の停止APIを第一手段とし、
応答しないときだけプロセスを終了させる。

UI（pywebview / HTML）からは呼ばれるだけで、このモジュール自身は画面を持たない。
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

# miku/src/control_panel/supervisor.py → miku/
MIKU_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = MIKU_DIR / "src"
RUNTIME_DIR = MIKU_DIR / "runtime"
STATE_FILE = RUNTIME_DIR / "running.json"
LOG_FILE = RUNTIME_DIR / "gateway.log"
TUNNEL_LOG_FILE = RUNTIME_DIR / "cloudflared.log"
SETTINGS_FILE = SRC_DIR / "settings.json"

GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 5001

# 起動完了待ちの上限（記憶の整理が走ると時間がかかるため長めに取る）
STARTUP_TIMEOUT_SEC = 300
# 停止要求を出してからプロセス終了を待つ上限
SHUTDOWN_TIMEOUT_SEC = 20
# cloudflared が公開URLを出すまで待つ上限
TUNNEL_URL_TIMEOUT_SEC = 30

# コンソールウィンドウを出さずに子プロセスを起動する（Windows のみ有効）
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    """settings.json を読む。無い・壊れている場合は空 dict を返す。"""
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def save_system_settings(updates: dict) -> bool:
    """settings.json の system セクションだけを更新する（他のキーは触らない）。"""
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    system = data.setdefault("system", {})
    system.update(updates)
    try:
        SETTINGS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def gateway_port(settings: dict | None = None) -> int:
    settings = settings if settings is not None else load_settings()
    try:
        return int(settings["services"]["gateway"]["port"])
    except Exception:
        return DEFAULT_GATEWAY_PORT


def _base_url(settings: dict | None = None) -> str:
    return f"http://{GATEWAY_HOST}:{gateway_port(settings)}"


# ---------------------------------------------------------------------------
# 稼働状態
# ---------------------------------------------------------------------------

def read_state() -> dict:
    """前回起動時に記録した PID などを読む。"""
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(state: dict) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    except Exception:
        pass


def clear_state(keep: dict | None = None) -> None:
    write_state(keep or {})


def is_running(settings: dict | None = None) -> bool:
    """Gateway が応答するかどうか（＝会話できる状態か）。

    親子関係ではなくポートの応答で判定するため、コントロールパネルを開き直した
    場合や、別の手段（bat 等）で起動された場合も正しく「起動中」と分かる。
    """
    try:
        r = requests.get(f"{_base_url(settings)}/api/status", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def is_tunnel_running(state: dict | None = None) -> bool:
    """外部アクセス（cloudflared）のプロセスが生きているか。

    ミクを「停止」した後も `keep_tunnel_on_stop` が真なら cloudflared は残る。
    ウィンドウを閉じるときに置き去りを作らないため、この判定が要る。
    """
    state = state if state is not None else read_state()
    return _pid_alive(state.get("cloudflared_pid"))


def unlock_auth(settings: dict | None = None) -> dict:
    """Gateway のログイン失敗ロックを解除する（設定タブのボタンから呼ばれる）。

    ロックは Gateway のメモリ上にあるため、停止中は既に解除されている状態＝何もしなくてよい。
    """
    settings = settings if settings is not None else load_settings()
    if not is_running(settings):
        return {"success": True, "unlocked": 0,
                "message": "ミクは停止中です（ロックは既に解除されています）"}
    try:
        r = requests.post(f"{_base_url(settings)}/api/auth/unlock", timeout=5)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 200 and data.get("success"):
            count = data.get("unlocked", 0)
            return {"success": True, "unlocked": count,
                    "message": f"ロックを解除しました（{count} 件）" if count
                               else "ロックされている相手はいませんでした"}
        return {"success": False, "message": data.get("error", "解除に失敗しました")}
    except requests.RequestException as e:
        return {"success": False, "message": f"解除に失敗しました: {e}"}


def status(settings: dict | None = None) -> dict:
    """UI へ返す状態一式。"""
    settings = settings if settings is not None else load_settings()
    state = read_state()
    running = is_running(settings)
    return {
        "running": running,
        "port": gateway_port(settings),
        "started_at": state.get("started_at"),
        "tunnel_url": state.get("tunnel_url"),
        "tunnel_running": is_tunnel_running(state),
        "log_path": str(LOG_FILE),
    }


def _pid_alive(pid: int | None) -> bool:
    """プロセスが生きているかを、副作用なしで確認する。

    **os.kill(pid, 0) を使ってはならない**。Unix の「シグナル0＝存在確認」と違い、
    Windows では sig=0 が signal.CTRL_C_EVENT と同値のため **Ctrl+C の送信**になり、
    CREATE_NO_WINDOW（隠しコンソール付き）で起動した子を実際に殺す。
    実害: 5秒毎の状態ポーリングが起動直後の cloudflared を毎回殺し、
    外部アクセスが常に Error 1033 になっていた（2026-08-03 実測で確定）。
    """
    if not pid:
        return False
    try:
        pid_num = int(pid)
    except (TypeError, ValueError):
        return False

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    # PROCESS_QUERY_LIMITED_INFORMATION。存在と終了コードを見るだけなので最小権限で開く
    handle = kernel32.OpenProcess(0x1000, False, pid_num)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259   # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


# ---------------------------------------------------------------------------
# 起動
# ---------------------------------------------------------------------------

def _python_exe() -> str:
    """venv の python.exe。コントロールパネル自身が venv で動いているため sys.executable でよいが、
    pythonw.exe で起動されている場合はコンソール版に読み替える（子の出力を拾うため）。
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        if console.exists():
            return str(console)
    return str(exe)


def _child_env() -> dict:
    """子プロセスへ渡す環境変数。

    モデルの保存先を miku/runtime 配下へ隔離する（アンインストール＝フォルダ削除で完結させる方針）。
    import より前に効かせる必要があるため、ここで明示的に渡す。
    """
    env = os.environ.copy()
    env["TORCH_HOME"] = str(RUNTIME_DIR / "models" / "torch")
    env["HF_HOME"] = str(RUNTIME_DIR / "models" / "hf")
    env["PYTHONUTF8"] = "1"
    # 標準出力はログファイルへリダイレクトする（＝非 TTY）ため、既定ではブロック
    # バッファになり **gateway.log が数分〜数十分遅れて書かれる**。障害の切り分けでは
    # 「今この瞬間の状態」が読めないと役に立たないので、行バッファに落とす
    # （2026-08-16: iPad 調査で 20 分ぶんのサーバー側ログが未書き出しのまま読めなかった）
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _spawn(cmd: list[str], cwd: Path, log_handle=None) -> subprocess.Popen:
    return subprocess.Popen(
        cmd, cwd=str(cwd), env=_child_env(),
        stdout=log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
        creationflags=_NO_WINDOW,
    )


def find_cloudflared() -> str | None:
    """cloudflared の場所。runtime に取得した版を優先し、無ければ PATH 上を使う。

    PATH も見るのは、winget 等で先に入れている環境があるため
    （旧 system_start.ps1 も同じ順で探していた）。
    """
    exe = RUNTIME_DIR / "cloudflared" / "cloudflared.exe"
    if exe.exists():
        return str(exe)
    return shutil.which("cloudflared")


def _extract_tunnel_url(line: str) -> str | None:
    """cloudflared の1行から公開URLを取り出す。

    URL は罫線で囲まれた枠の中に出るため（`|  https://xxx.trycloudflare.com  |`）、
    区切りを空白に均してから探す。
    """
    if "trycloudflare.com" not in line:
        return None
    for token in line.replace("|", " ").split():
        if token.startswith("https://") and "trycloudflare.com" in token:
            return token.strip()
    return None


def start_tunnel(on_progress=None) -> tuple[int | None, str | None]:
    """cloudflared を起動し、出力に現れる公開URLを拾って返す。

    Quick Tunnel の URL は起動のたびに変わるため、取得できたら state に残して
    UI に表示する（ユーザーがスマホへ共有するのに使う）。

    **出力は最後まで読み続ける**。URL を拾ったあとも cloudflared はログを出し続けるので、
    読むのをやめるとパイプが詰まり（Windows のパイプバッファは数十KB）、**cloudflared 自身が
    書き込みでブロックしてトンネルが止まる**。読んだ内容は `runtime/cloudflared.log` に残す
    ——失敗の理由（証明書・ネットワーク・版の古さ）が分かる場所が他に無いため。
    """
    def progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    exe = find_cloudflared()
    if not exe:
        progress("cloudflared が見つかりません（設定タブの「外部アクセス」から取得してください）")
        return None, None

    port = gateway_port()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(exe), "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=_NO_WINDOW,
    )

    found: dict[str, str | None] = {"url": None}
    ready = threading.Event()

    def pump() -> None:
        try:
            with open(TUNNEL_LOG_FILE, "a", encoding="utf-8", errors="replace") as log:
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"cloudflared 起動（--url http://localhost:{port}）=====\n")
                log.flush()
                for line in proc.stdout or []:
                    log.write(line)
                    log.flush()
                    if found["url"] is None:
                        url = _extract_tunnel_url(line)
                        if url:
                            found["url"] = url
                            ready.set()
                # 出力が尽きた＝プロセスが終わるとき。終了コードを残す
                # （2026-08-03 の障害調査で「いつ・どう死んだか」がどこにも残っておらず
                # 特定に苦労したため。強制終了なら 1 や負値、自然終了なら 0 が入る）
                rc = proc.wait()
                log.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"cloudflared 終了 (exit code {rc}) =====\n")
        finally:
            # プロセスが終わったときにも待ち側を解放する（URL が出ないまま落ちた場合）
            ready.set()

    threading.Thread(target=pump, daemon=True, name="cloudflared-log").start()
    ready.wait(TUNNEL_URL_TIMEOUT_SEC)

    if found["url"] is None:
        if proc.poll() is not None:
            progress(f"cloudflared が起動できませんでした（詳細: {TUNNEL_LOG_FILE.name}）")
        else:
            progress(f"アクセス用URLを取得できませんでした（詳細: {TUNNEL_LOG_FILE.name}）")
    return proc.pid, found["url"]


def start(skip_memory: bool = False, on_progress=None) -> dict:
    """システムを起動する。戻り値は status() と同じ形。

    on_progress: 進捗メッセージを受け取るコールバック（UI へ流す用）。
    """
    def progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    settings = load_settings()
    if is_running(settings):
        progress("すでに起動しています")
        return status(settings)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    state = read_state()

    # トンネルは前回のものが生きていれば使い回す（URL が変わらないようにするため）
    if settings.get("system", {}).get("tunnel_enabled"):
        if _pid_alive(state.get("cloudflared_pid")):
            progress("外部アクセスは前回のURLを引き継ぎます")
        else:
            progress("外部アクセス（Cloudflare）を開始しています...")
            pid, url = start_tunnel(on_progress=on_progress)
            state["cloudflared_pid"] = pid
            state["tunnel_url"] = url

    progress("ミクのシステムを起動しています...")
    log = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 起動 =====\n")
    log.flush()
    cmd = [_python_exe(), "gateway_service.py"]
    if skip_memory:
        cmd.append("--skip-memory")
    proc = _spawn(cmd, SRC_DIR, log_handle=log)

    state["gateway_pid"] = proc.pid
    state["started_at"] = time.time()
    write_state(state)

    progress("起動を待っています（記憶の整理に時間がかかることがあります）...")
    deadline = time.time() + STARTUP_TIMEOUT_SEC
    while time.time() < deadline:
        if proc.poll() is not None:
            progress("起動に失敗しました。ログを確認してください")
            return status(settings)
        if is_running(settings):
            progress("起動しました")
            return status(settings)
        time.sleep(1.0)

    progress("起動の確認がタイムアウトしました。ログを確認してください")
    return status(settings)


# ---------------------------------------------------------------------------
# 停止
# ---------------------------------------------------------------------------

def stop(keep_tunnel: bool = True, on_progress=None) -> dict:
    """システムを停止する。

    第一手段は Gateway の停止API（記憶を保存してから終了する）。
    応答しない場合のみプロセスを終了させる。

    keep_tunnel: True なら cloudflared を残す。Quick Tunnel の URL は
        起動のたびに変わるため、残しておくと再起動しても同じURLで繋がる。
    """
    def progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    settings = load_settings()
    state = read_state()

    if is_running(settings):
        progress("ミクの記憶を保存して停止しています...")
        try:
            requests.post(f"{_base_url(settings)}/api/system/shutdown", timeout=5)
        except requests.RequestException:
            pass

        deadline = time.time() + SHUTDOWN_TIMEOUT_SEC
        while time.time() < deadline:
            if not is_running(settings):
                break
            time.sleep(0.5)

    # 停止APIが効かなかった場合の保険（記録した PID を終了させる）
    if is_running(settings):
        progress("応答がないため、プロセスを終了します")
        _terminate(state.get("gateway_pid"))

    if not keep_tunnel:
        _terminate(state.pop("cloudflared_pid", None))
        state.pop("tunnel_url", None)
    elif _pid_alive(state.get("cloudflared_pid")):
        progress("外部アクセスのURLは保持しました")

    state.pop("gateway_pid", None)
    state.pop("started_at", None)
    write_state(state)
    progress("停止しました")
    return status(settings)


def _terminate(pid: int | None) -> None:
    if not _pid_alive(pid):
        return
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 機能の導入・状態（設定タブの「ミクにできること」が使う）
# ---------------------------------------------------------------------------

def has_vocaloid6() -> bool:
    """VOCALOID6 が入っているか。**作曲機能を導入できる条件**（2026-09-06 本人判断）。

    この機能の値打ちは「**初音ミクが歌ってくれる**」ところにある。V6 が無いと曲は作れても
    **ミクの声で歌わせることが一生できない**（`play_song` が歌うのは V6 の書き出しを仕上げた完成品
    `master_song_v{N}.wav`・`master_acappella_v{N}.wav` だけで、仮歌〔サイン波〕は対象外）。聴けない曲が溜まるだけなので**導入させない**。
    ただし**黙って飛ばさない**——理由を伝える。

    **パスは決め打ちにしない**——設置先は利用者が変えられるので、決め打ちだと
    **別のドライブに入れた人を「持っていない」と誤判定**して導入させないことになる。
    ただし**深追いもしない**（本人判断）: ①`.vpr` の関連付け ②既定のパス、の2つだけ。
    歌声ライブラリの有無は見ない（名前照合になり、かえって誤判定を生むため）。

    ※ロジックの正は `song_service/engine/v6_export.py` の `find_v6_exe()`。PowerShell 側は
      `setup_common.ps1` の `Get-MikuVocaloid6Exe`。変えるときは3つとも直す
    """
    import re
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Classes\VOCALOID6.vpr\shell\open\command") as k:
            cmd = winreg.QueryValueEx(k, "")[0]
        if (m := re.match(r'\s*"([^"]+)"', cmd)) and Path(m.group(1)).exists():
            return True
    except OSError:
        pass
    for base in (os.environ.get("ProgramFiles"), r"C:\Program Files"):
        if base and (Path(base) / "VOCALOID6" / "Editor" / "VOCALOID6.exe").exists():
            return True
    return False


def has_nvidia_gpu() -> bool:
    """nvidia-smi の有無で GPU を判定する（本体 venv の torch に依存しない）。"""
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10,
                           creationflags=_NO_WINDOW)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def feature_status() -> dict:
    """機能の導入状況。「設定値」ではなく**実体の有無**で判定する。

    設定が ON でも実体が無ければ動かないため、UI には実体を見せる。
    """
    settings = load_settings()
    system = settings.get("system", {})
    live2d = SRC_DIR / "live2d/CubismWebSamples/Samples/TypeScript/Demo/dist"
    song = SRC_DIR / "song_service"
    return {
        "live2d_ready": (live2d / "index.html").exists(),
        "icon_ready": (RUNTIME_DIR / "miku_icon.ico").exists(),
        "song_ready": (song / "venv/Scripts/python.exe").exists(),
        "song_enabled": bool(settings.get("song_service", {}).get("enabled")),
        "gpu": has_nvidia_gpu(),
        "tunnel_ready": find_cloudflared() is not None,
        "tunnel_enabled": bool(system.get("tunnel_enabled")),
    }


def derive_feature_flags(settings: dict) -> dict:
    """入力された値から、機能の有効・無効を決めて settings に書き込む（破壊的）。

    **画面にトグルを置かない**ための処理。利用者は「キーを入れる／消す」だけを行い、
    `enabled` はその結果として決まる。トグルを別に持つと
    **「有効なのにキーが無い」「導入していないのに有効」**という食い違いを作れてしまう。

    作曲機能だけは値ではなく**実体**（導入済みか）で決まる。
    """
    from utils import settings_store

    def has(path: str) -> bool:
        value = settings_store.get_by_path(settings, path)
        if not isinstance(value, str):
            return bool(value)
        value = value.strip()
        # settings_sample.json のプレースホルダ（"put your ..."）は未設定として扱う
        return bool(value) and not value.startswith("put your")

    settings_store.set_by_path(settings, "web_search.enabled", has("api_keys.tavily"))
    settings_store.set_by_path(settings, "email.enabled", all(
        has(p) for p in ("api_keys.gmail_app_password", "email.smtp_user", "email.recipient")))

    song = SRC_DIR / "song_service" / "venv" / "Scripts" / "python.exe"
    settings_store.set_by_path(settings, "song_service.enabled", song.exists())
    return settings


def install_cloudflared(on_progress=None) -> bool:
    """cloudflared（外部アクセス用トンネル）を runtime へ取得する（単体exe・約60MB）。"""
    if find_cloudflared():
        return True
    exe = RUNTIME_DIR / "cloudflared" / "cloudflared.exe"
    if on_progress:
        on_progress("cloudflared をダウンロードしています（約60MB）...")
    url = ("https://github.com/cloudflare/cloudflared/releases/latest/download/"
           "cloudflared-windows-amd64.exe")
    try:
        exe.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(exe, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    except Exception as e:
        if on_progress:
            on_progress(f"ダウンロードに失敗しました: {e}")
        exe.unlink(missing_ok=True)
        return False
    return exe.exists()


def set_tunnel_enabled(enabled: bool, on_progress=None) -> dict:
    """外部アクセスの有効・無効を切り替える（有効化時は cloudflared を取得する）。"""
    if enabled and not install_cloudflared(on_progress=on_progress):
        return feature_status()
    save_system_settings({"tunnel_enabled": bool(enabled)})
    if on_progress:
        on_progress("外部アクセスを有効にしました（次回起動時からURLが表示されます）"
                    if enabled else "外部アクセスを無効にしました")
    return feature_status()


def _run_powershell(script: Path, on_progress=None, args: list[str] | None = None) -> bool:
    """セットアップ用の PowerShell スクリプトを実行し、出力を進捗として流す。

    Windows 標準の PowerShell 5.1 を固定で使う（PowerShell 7 の有無に依存させない）。
    """
    ps = Path(os.environ.get("SystemRoot", r"C:\Windows")) / \
        "System32/WindowsPowerShell/v1.0/powershell.exe"
    cmd = [str(ps), "-ExecutionPolicy", "Bypass", "-File", str(script), *(args or [])]
    try:
        proc = subprocess.Popen(cmd, cwd=str(script.parent), env=_child_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                creationflags=_NO_WINDOW)
    except OSError as e:
        if on_progress:
            on_progress(f"起動に失敗しました: {e}")
        return False
    for line in proc.stdout or []:
        line = line.rstrip()
        if line and on_progress:
            on_progress(line)
    return proc.wait() == 0


def install_song_service(on_progress=None, *, skip_vocaloid_check: bool = False) -> dict:
    """作曲機能を導入する。

    ★**GPU が無くても導入できる**（2026-09-01）。それまでは入口で一律に弾いていたが、
      **オリジナル曲の経路は GPU を使っていない**（LLM＝クラウド／FluidSynth・numpy・
      libresvip＝CPU）。GPU が要るのは**音源しか無い曲のカバー**（音源分離・採譜・
      強制アラインメント）だけ。楽譜（MIDI）が見つかる曲のカバーは GPU 無しで作れる。

      GPU 無し … `-OriginalOnly`（**約4GB・十数分**。うち約3.1GB は鼻歌の聞き取り用の
                  文字起こしモデル）。音源しか無い曲のカバーを頼まれたら断る
      GPU あり … 全部入れる（ダウンロード約15GB・40〜90分）
      ※数字の根拠は `song_service/setup_song_service.ps1` 冒頭のコメント（実測と改訂の日付つき）
    """
    script = SRC_DIR / "song_service" / "setup_song_service.ps1"
    if not script.exists():
        return feature_status()
    # ★**V6 が無ければ導入しない**（2026-09-06）。理由を伝えてから断る。
    #   `skip_vocaloid_check` は**デバッグ用の抜け道**（検証用の仮想マシンなど、V6 を
    #   入れられない環境で導入まで通したいとき）。**飛ばすのは導入の可否だけ**で、
    #   実行時の書き出しは正直に失敗する。UI からは渡さない＝コードから呼ぶときだけ。
    if skip_vocaloid_check:
        if on_progress:
            on_progress("【デバッグ】VOCALOID6 の確認を飛ばして導入します"
                        "（曲は作れますが、歌声の書き出しは実行時に失敗します）")
    elif not has_vocaloid6():
        if on_progress:
            on_progress("VOCALOID6 が見つからないため、作曲機能は導入できません。"
                        "この機能は『ミクが自分の声で歌う』ためのもので、VOCALOID6"
                        "（＋初音ミク V6・別売）が無いと曲を歌声にできません。"
                        "VOCALOID6 を導入してから、もう一度お試しください")
        return feature_status()
    full = has_nvidia_gpu()
    if on_progress:
        on_progress("作曲機能を導入しています（約15GB・40〜90分かかります）..." if full else
                    "GPU が無いので、音源しか無い曲のカバーに使うものを除いて導入します（約4GB・十数分）。"
                    "オリジナル曲と、楽譜（MIDI）が見つかる曲のカバーは作れます")
    args = [] if full else ["-OriginalOnly"]
    if _run_powershell(script, args=args, on_progress=on_progress):
        save_settings_key("song_service", "enabled", True)
        # ★導入の最後に**一度だけ実際に起動して確かめる**（2026-09-06 本人指定）。
        #   ミクは曲づくりの最後に VOCALOID6 を自分で開くので、それがこの環境で通るかを見る
        #   （実体があってもライセンス認証で止まる等は、起動して初めて分かる）。
        #   曲は作らず数秒で閉じる。**失敗しても導入は止めない**——手で書き出す道は残っている。
        check = SRC_DIR / "song_service" / "export_vocal.ps1"
        if check.exists():
            if on_progress:
                on_progress("動作確認のため VOCALOID6 を一度だけ起動します（数秒で閉じます）...")
            ok = _run_powershell(check, args=["-CheckOnly"], on_progress=None)
            if on_progress:
                on_progress("動作確認 OK: ミクが自分で歌声を書き出せます" if ok else
                            "VOCALOID6 を起動できませんでした。作曲機能は使えますが、"
                            "歌声の書き出しはご自身で行っていただくことになります")
        if on_progress:
            on_progress("作曲機能を導入しました")
    return feature_status()


def save_settings_key(section: str, key: str, value) -> bool:
    """settings.json の任意セクションの1キーを更新する。"""
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    data.setdefault(section, {})[key] = value
    try:
        SETTINGS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def open_live2d_setup() -> bool:
    """Live2D セットアップ（対話式）を別ウィンドウで開く。

    zip の選択ダイアログや Node.js 導入の確認を含むため、**既存の対話スクリプトを
    そのまま使う**（パネル側に作り直さない）。
    """
    bat = SRC_DIR / "live2d" / "setup_live2d.bat"
    if not bat.exists():
        return False
    subprocess.Popen(["cmd", "/c", "start", "", str(bat)], cwd=str(bat.parent), shell=False)
    return True


#: アイコンの生成は一度に1本だけ。起動時の自動の作り直し（`__main__._ensure_generated_icon`）と
#: 設定タブのボタンが重なると、2本が同じ `miku_icon.ico` / `shortcut_icon.ico` へ書く（2026-10-18 のレビュー）
_ICON_LOCK = threading.Lock()


def regenerate_icon(on_progress=None) -> dict:
    """Live2D モデルからアイコンを作り直す（実行中なら、終わるのを待ってから作り直す）。"""
    script = SRC_DIR / "live2d" / "generate_icon.py"
    if not script.exists():
        return feature_status()
    with _ICON_LOCK:
        if on_progress:
            on_progress("ミクの姿からアイコンを作成しています...")
        proc = subprocess.Popen([_python_exe(), str(script)], cwd=str(script.parent),
                                env=_child_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
        for line in proc.stdout or []:
            line = line.rstrip()
            if line and on_progress:
                on_progress(line)
        proc.wait()
    return feature_status()


def read_log_tail(lines: int = 40) -> str:
    """ログの末尾を返す（UI の「起動失敗の原因がその場で見える」ため）。"""
    try:
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "（ログはまだありません）"
    return "\n".join(text.splitlines()[-lines:])
