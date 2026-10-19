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

"""コントロールパネルの入口（ネイティブウィンドウ）。

起動:
    pythonw.exe start_panel.py          （miku/src で実行。黒い窓は出ない）

**`-m control_panel` は使わない**（2026-09-19 に変更）。配布物の保護版ではこのパッケージが
Nuitka で `.pyd` へ機械語化され、`.pyd` に対して `python -m <パッケージ>` は使えないため。
`start_panel.py` はここの `main()` を呼ぶだけの薄いランチャーで、素版でも同じ経路で動く。

画面は HTML（ui/index.html）で、pywebview が Windows 標準の WebView2 で描画する。
Python 側の処理は Api クラス経由で JS から呼ばれる（HTTPサーバーは立てない）。

**このファイルは「窓と橋渡し」だけを担当する**。起動・停止の実処理は supervisor.py。
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

# start_panel.py 経由でも単体実行でも同じように import できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control_panel import supervisor  # noqa: E402

UI_DIR = Path(__file__).resolve().parent / "ui"

# 起動時にコンソールが付いてきたか。付いてきた場合は画面にも注意書きを出す
# （閉じるとスタート画面ごと落ちるため）
CONSOLE_ATTACHED = False
# 利用者向けの呼称は「スタート画面」。Windows 自身の「コントロールパネル」と
# 取り違えられないようにするため（コード上の名前は control_panel のまま）
WINDOW_TITLE = "ワタシノセカイト初音ミク ― スタート画面"


class Api:
    """JS から呼ばれる API。戻り値はそのまま JS の Promise で受け取れる。"""

    def __init__(self) -> None:
        self._window = None
        self._busy = False
        # 処理中に「何をしているか」（start / stop / restart / setup）。
        # 起動は数十秒〜数分かかるため、画面が「停止中」のまま見えないよう
        # 状態表示の出し分けに使う（UI 側で判断させない）
        self._busy_action: str | None = None

    def bind(self, window) -> None:
        self._window = window

    # --- 状態 ---------------------------------------------------------------

    def get_status(self) -> dict:
        st = supervisor.status()
        settings = supervisor.load_settings()
        system = settings.get("system", {})
        st["skip_memory"] = bool(system.get("skip_memory"))
        st["keep_tunnel_on_stop"] = bool(system.get("keep_tunnel_on_stop", True))
        st["tunnel_enabled"] = bool(system.get("tunnel_enabled"))
        st["busy"] = self._busy
        st["busy_action"] = self._busy_action
        st["console_attached"] = CONSOLE_ATTACHED
        return st

    def check_update(self) -> dict:
        """新しい版が出ているか（スタート画面を開いたときに1回だけ呼ぶ・2026-10-20）。"""
        return supervisor.check_update()

    def get_tunnel_qr(self) -> dict:
        """外出先の URL を QR コード（SVG）にして返す（2026-10-20）。

        絵はこの PC の中で作り、外のサービスには送らない。パスワードの有無も返し、
        未設定なら画面が「読んだ人は誰でも入れる」と注意を出す。
        """
        url = supervisor.status().get("tunnel_url")
        if not url:
            return {"ok": False, "message": "外出先の URL がまだありません"}
        try:
            import segno
        except ImportError:
            return {"ok": False, "message": "QR コードを作る部品がありません。インストーラをもう一度実行してください"}
        settings = supervisor.load_settings()
        auth = settings.get("security", {}).get("auth", {})
        return {
            "ok": True,
            "url": url,
            # omitsize: 幅と高さを書かず viewBox だけにする（大きさは panel.css で決める）
            "svg": segno.make(url, error="m").svg_inline(scale=6, border=2, dark="#111", light="#fff",
                                                         omitsize=True),
            "password_set": bool(auth.get("password_hash")),
            "keep_tunnel_on_stop": bool(settings.get("system", {}).get("keep_tunnel_on_stop", True)),
        }

    def get_log(self, lines: int = 40) -> str:
        return supervisor.read_log_tail(lines)

    def set_option(self, key: str, value: bool) -> bool:
        """チェックボックスの状態を settings.json に保存する（次回起動時も保持）。"""
        if key not in ("skip_memory", "keep_tunnel_on_stop"):
            return False
        return supervisor.save_system_settings({key: bool(value)})

    # --- 操作 ---------------------------------------------------------------

    def start(self, skip_memory: bool = False) -> dict:
        if self._busy:
            return self.get_status()
        self._busy = True
        self._busy_action = "start"
        try:
            supervisor.save_system_settings({"skip_memory": bool(skip_memory)})
            return supervisor.start(skip_memory=skip_memory, on_progress=self._progress)
        finally:
            self._busy = False
            self._busy_action = None

    def stop(self, keep_tunnel: bool = True) -> dict:
        if self._busy:
            return self.get_status()
        self._busy = True
        self._busy_action = "stop"
        try:
            supervisor.save_system_settings({"keep_tunnel_on_stop": bool(keep_tunnel)})
            return supervisor.stop(keep_tunnel=keep_tunnel, on_progress=self._progress)
        finally:
            self._busy = False
            self._busy_action = None

    def open_browser(self) -> bool:
        import webbrowser
        webbrowser.open(f"http://localhost:{supervisor.gateway_port()}")
        return True

    # --- 設定タブ -----------------------------------------------------------

    def get_settings(self) -> dict:
        """設定タブの描画に必要なものを一度に返す。

        画面は**スキーマ（並び・文言）と現在値と機能の導入状況**をまとめて受け取る。
        3回に分けて取りに行くと、機能の行と入力欄で状態がずれた瞬間が見えるため。
        """
        from utils import settings_schema, settings_store
        try:
            current = settings_store.load_settings(supervisor.SETTINGS_FILE)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        view = settings_store.build_view(current)
        return {
            "success": True,
            "sections": view["sections"],
            "features": settings_schema.FEATURES,
            "status": supervisor.feature_status(),
            "system_running": supervisor.is_running(),
        }

    def save_settings(self, updates: dict) -> dict:
        """設定を保存する。**機能の有効・無効はここで値から決める**。

        画面にトグルを置かない代わりに、キーが入っていれば有効・空なら無効とする
        （「有効なのにキーが無い」状態を作れなくするため）。
        """
        from utils import settings_store, settings_validate
        if not isinstance(updates, dict) or not updates:
            return {"success": False, "error": "変更はありません"}
        try:
            current = settings_store.load_settings(supervisor.SETTINGS_FILE)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        new_settings, errors = settings_store.apply_updates(current, updates)
        if errors:
            return {"success": False, "errors": errors}

        supervisor.derive_feature_flags(new_settings)

        cross_errors = settings_validate.validate_settings(new_settings)
        if cross_errors:
            return {"success": False, "errors": cross_errors}

        settings_store.save_settings(new_settings, supervisor.SETTINGS_FILE)
        return {
            "success": True,
            "restart_required": True,
            "notices": settings_validate.save_notices(new_settings),
            "status": supervisor.feature_status(),
        }

    def restart(self) -> dict:
        """設定を反映するために再起動する（記憶の整理はスキップする）。

        設定は各サービスが起動時に読むため、保存しただけでは効かない。
        稼働していないときは何もしない（起動はホームタブの操作に任せる）。
        """
        if self._busy or not supervisor.is_running():
            return self.get_status()
        self._busy = True
        self._busy_action = "restart"
        try:
            supervisor.stop(keep_tunnel=True, on_progress=self._progress)
            return supervisor.start(skip_memory=True, on_progress=self._progress)
        finally:
            self._busy = False
            self._busy_action = None

    def get_features(self) -> dict:
        return supervisor.feature_status()

    def install_song_service(self) -> dict:
        if self._busy:
            return supervisor.feature_status()
        self._busy = True
        self._busy_action = "setup"
        try:
            return supervisor.install_song_service(on_progress=self._progress)
        finally:
            self._busy = False
            self._busy_action = None

    def setup_live2d(self) -> bool:
        """Live2D の対話セットアップを別ウィンドウで開く（zip 選択等があるため）。"""
        self._progress("Live2D のセットアップを別ウィンドウで開きました")
        return supervisor.open_live2d_setup()

    def regenerate_icon(self) -> dict:
        if self._busy:
            return supervisor.feature_status()
        self._busy = True
        self._busy_action = "setup"
        try:
            return supervisor.regenerate_icon(on_progress=self._progress)
        finally:
            self._busy = False
            self._busy_action = None

    def set_tunnel(self, enabled: bool) -> dict:
        if self._busy:
            return supervisor.feature_status()
        self._busy = True
        self._busy_action = "setup"
        try:
            return supervisor.set_tunnel_enabled(bool(enabled), on_progress=self._progress)
        finally:
            self._busy = False
            self._busy_action = None

    def open_url(self, url: str) -> bool:
        """メンテページ（VDBビューア等）を既定のブラウザで開く。"""
        import webbrowser
        webbrowser.open(url)
        return True

    def unlock_auth(self) -> dict:
        """ログイン失敗によるロックを解除する（設定タブのボタンから）。

        ロックは Gateway のメモリにあるため、停止中はそもそもロックも消えている。
        """
        return supervisor.unlock_auth()

    def open_log_folder(self) -> bool:
        import subprocess
        subprocess.Popen(["explorer", "/select,", str(supervisor.LOG_FILE)])
        return True

    # --- 内部 ---------------------------------------------------------------

    def _progress(self, message: str) -> None:
        """supervisor の進捗を画面へ流す。"""
        if not self._window:
            return
        payload = message.replace("\\", "\\\\").replace("'", "\\'")
        try:
            self._window.evaluate_js(f"window.onProgress && window.onProgress('{payload}')")
        except Exception:
            pass


def _confirm_close(api: Api) -> bool:
    """✕ が押されたときの確認。False を返すとウィンドウは閉じない。

    「パネルを閉じる＝ミクも止まる」に統一する（置き去りのプロセスを作らない）。
    ミクを動かしたまま席を立つときは最小化する。

    **「停止」ボタンで止めた後**も外部アクセス（cloudflared）は残っているため
    （URL を保つ既定動作）、稼働中でなくてもトンネルが生きていれば確認して終了させる。
    以前は `is_running()` が偽なら素通りしており、停止→✕でトンネルが置き去りになっていた
    （2026-08-06 修正）。
    """
    running = supervisor.is_running()
    tunnel_alive = supervisor.is_tunnel_running()
    if not running and not tunnel_alive:
        return True                      # 何も動いていない → そのまま閉じる

    if running:
        message = ("会話システムを停止して、スタート画面を終了します。\n"
                   "ミクの記憶は保存されてから終了します。\n\n"
                   "※ 外部アクセスも終了します（次回はURLが変わります）。\n"
                   "　 URLを保ちたい場合はキャンセルして「停止」ボタンを使ってください。")
    else:
        # ミクは停止済みで、外部アクセスだけが残っている状態
        message = ("外部アクセスを終了して、スタート画面を終了します。\n"
                   "（ミクの会話システムは停止済みです）\n\n"
                   "※ 次回はアクセス用のURLが変わります。\n"
                   "　 URLを保ちたい場合はキャンセルして、この画面を開いたままにしてください。")

    import webview
    if not webview.windows[0].create_confirmation_dialog("終了しますか？", message):
        return False
    supervisor.stop(keep_tunnel=False)
    return True


def _window_icon() -> Path | None:
    """ウィンドウに出すアイコン。Live2D から起こした版を優先する。

    デスクトップのショートカットと同じ絵にするため、探す順序も揃えてある。
    """
    for path in (supervisor.RUNTIME_DIR / "miku_icon.ico",
                 supervisor.MIKU_DIR / "app_icon.ico"):
        if path.exists():
            return path
    return None


def _apply_window_icon() -> None:
    """タイトルバーとタスクバーのアイコンを設定する。

    **なぜ ctypes なのか**: pywebview の `icon` 引数は GTK/Qt 向けで Windows では効かず、
    何もしないと `pythonw.exe` のアイコンがそのまま出る。Windows のやり方
    （ウィンドウハンドルへ `WM_SETICON`）で設定する。
    アイコンが出ないだけで実害は無いので、失敗しても起動は続ける。
    """
    _ensure_generated_icon()
    icon = _window_icon()
    if icon is None:
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32
        WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
        IMAGE_ICON, LR_LOADFROMFILE = 1, 0x0010

        # ウィンドウが出来るまで少し待つ（GUI ループ開始直後は間に合わないことがある）
        hwnd = 0
        for _ in range(50):
            hwnd = user32.FindWindowW(None, WINDOW_TITLE)
            if hwnd:
                break
            time.sleep(0.1)
        if not hwnd:
            return

        for which, size in ((ICON_BIG, 32), (ICON_SMALL, 16)):
            handle = user32.LoadImageW(None, str(icon), IMAGE_ICON, size, size, LR_LOADFROMFILE)
            if handle:
                user32.SendMessageW(hwnd, WM_SETICON, which, handle)
    except Exception:
        pass


def _ensure_generated_icon() -> None:
    """ミクの姿のアイコンがまだ無ければ、裏で作る（作れたらウィンドウにも付け直す）。

    インストール中の生成は失敗することがある（2026-10-08 の VM で WebView2 の初期化が
    E_ABORT で落ちた）。スタート画面は利用者のふつうの権限で WebView2 が動いている場なので、
    ここでもう一度試せば、利用者が何もしなくてもショートカットの絵が変わる。
    """
    generated = supervisor.RUNTIME_DIR / "miku_icon.ico"
    if generated.exists():
        # 作れているのにショートカットが簡易マークを指したまま、を直す（インストール中の生成に失敗した・
        # 0.2.0 から入れ直した等）。指し先を確かめて、違うときだけ付け替える（裏で・PowerShell を1回）
        def retarget() -> None:
            try:
                from live2d import generate_icon
                generate_icon.publish_to_shortcuts()
            except Exception:
                pass
        threading.Thread(target=retarget, daemon=True).start()
        return
    model = (supervisor.SRC_DIR / "live2d" / "CubismWebSamples" / "Samples" / "Resources"
             / "miku_sample_t04" / "miku_sample_t04.model3.json")
    if not model.exists():
        return

    def work() -> None:
        try:
            supervisor.regenerate_icon()
        except Exception:
            return
        if (supervisor.RUNTIME_DIR / "miku_icon.ico").exists():
            _apply_window_icon()

    threading.Thread(target=work, daemon=True).start()


def _console_process_names() -> list[str]:
    """いまのコンソールに紐づくプロセスの実行ファイル名を返す（小文字）。"""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    pids = (wintypes.DWORD * 32)()
    count = kernel32.GetConsoleProcessList(pids, 32)

    names: list[str] = []
    for i in range(min(count, 32)):
        # PROCESS_QUERY_LIMITED_INFORMATION。名前を引くだけなので最小権限で開く
        handle = kernel32.OpenProcess(0x1000, False, pids[i])
        if not handle:
            names.append("?")
            continue
        try:
            buf = ctypes.create_unicode_buffer(260)
            size = wintypes.DWORD(260)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                names.append(Path(buf.value).name.lower())
            else:
                names.append("?")
        finally:
            kernel32.CloseHandle(handle)
    return names


# 人が開いたターミナルだと判断する目印。これらが紐づいていれば触らない
_SHELL_NAMES = {"cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe",
                "windowsterminal.exe", "wt.exe"}


def _warn_on_console(kernel32) -> None:
    """コンソールに「閉じないで」と出す。タブのタイトルにも出す。

    **なぜ隠さずに書くのか**: Windows 11 で既定端末が Windows Terminal になってから、
    `ShowWindow(GetConsoleWindow(), SW_HIDE)` は効かない（ConPTY 越しのため、
    返るハンドルが実際に見えている窓ではない。microsoft/terminal#15311・#12570）。
    旧来の conhost なら消えるので試みはするが、消えなかったときのために書いておく。
    この窓を閉じるとスタート画面も落ちるので、知らせないと事故になる。
    """
    import ctypes
    from ctypes import wintypes

    kernel32.SetConsoleTitleW("【閉じないでください】ワタシノセカイト初音ミク")

    notice = (
        "\n"
        "============================================================\n"
        "  この画面は閉じないでください\n"
        "\n"
        "  閉じると「スタート画面」も一緒に終了します。\n"
        "  最小化するか、そのままにしておいてください。\n"
        "  （Windows キー ＋ D でまとめてしまえます）\n"
        "============================================================\n"
    )
    # pythonw では sys.stdout が無いことがあるため、コンソールへ直接書く
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW("CONOUT$", 0x40000000, 0x00000002,
                                  None, 3, 0, None)
    if handle and handle != wintypes.HANDLE(-1).value:
        try:
            written = wintypes.DWORD()
            kernel32.WriteConsoleW(handle, ctypes.c_wchar_p(notice),
                                   len(notice), ctypes.byref(written), None)
        finally:
            kernel32.CloseHandle(handle)


def _hide_own_console() -> None:
    """自分のために作られたコンソールを隠す。

    **なぜ必要か**: venv の `pythonw.exe` は本物の Python ではなく中継用の小さな exe で、
    uv が作る環境ではそれが**コンソールアプリとして作られている**
    （2026-08-03 の VM 検証: PEヘッダの Subsystem が 3）。名前に反してコンソールが開き、
    スタート画面はその子プロセスなので、**そのコンソールを閉じるとスタート画面ごと落ちる**。
    exe を差し替える手は使えない（venv の exe は DLL を本体側から解決するための中継で、
    本物を置くと python313.dll が見つからず起動しなくなる。2026-08-03 に実際に壊した）。

    **既存のターミナルから起動した場合は触らない**。`python start_panel.py` を
    手で叩いたときに、そのターミナルごと消えてしまうため。
    判定は「紐づくプロセスの数」ではなく**シェルが紐づいているか**で行う——
    中継用 exe は本物の Python を子として起動するので、自分のためのコンソールでも
    プロセスは2つ以上になる（数で見ると隠せない。2026-08-03 に実際にそうなった）。

    判断材料は `runtime/panel.log` に残す。ここは失敗しても静かに素通りする場所なので、
    記録が無いと「なぜ隠れなかったか」を追えない。
    """
    decision, names = "", []
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            decision = "コンソールなし（何もしない）"
        else:
            names = _console_process_names()
            shells = sorted(set(names) & _SHELL_NAMES)
            if shells:
                decision = f"隠さない（シェルが紐づいている: {', '.join(shells)}）"
            else:
                # 旧来の conhost なら消える。Windows Terminal では効かないので、
                # 消えなかったときのために注意書きを出しておく
                ctypes.windll.user32.ShowWindow(hwnd, 0)    # SW_HIDE
                _warn_on_console(kernel32)
                global CONSOLE_ATTACHED
                CONSOLE_ATTACHED = True
                decision = "隠す試行＋注意書きを表示（Terminal では隠れない）"
    except Exception as e:
        decision = f"判定に失敗（起動は続行）: {e}"

    try:
        supervisor.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        # BOM 付きで書く。PowerShell 5.1 の Get-Content は BOM 無し UTF-8 を
        # cp932 と誤読して文字化けする（人が読むログなので読めないと意味がない）
        with open(supervisor.RUNTIME_DIR / "panel.log", "a",
                  encoding="utf-8-sig", errors="replace") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} コンソール: "
                      f"{decision} / 紐づくプロセス={names}\n")
    except Exception:
        pass


def main() -> int:
    _hide_own_console()

    try:
        import webview
    except ImportError:
        _fatal("pywebview が見つかりません。インストーラ（MikuAgentSetup.exe）をもう一度実行してください。")
        return 1

    # タスクバーに独立したアプリとして並ぶようにする（既定では pythonw.exe 扱いになり、
    # 他の Python 製アプリとまとめられたうえ、そちらのアイコンが使われる）。
    # ウィンドウを作る前に呼ぶ必要がある
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ChordDotE.MikuAgent.StartScreen")
    except Exception:
        pass

    api = Api()
    window = webview.create_window(
        WINDOW_TITLE,
        str(UI_DIR / "index.html"),
        js_api=api,
        width=760, height=780, min_size=(620, 560),
    )
    api.bind(window)
    window.events.closing += lambda: _confirm_close(api)
    # 第1引数の関数は GUI ループ開始後に別スレッドで呼ばれる
    webview.start(_apply_window_icon)
    return 0


def _fatal(message: str) -> None:
    """pythonw.exe では例外が画面に出ないため、明示的にダイアログで知らせる。

    ここで黙って落ちると「ダブルクリックしても何も起きない」状態になり、
    利用者も開発者も原因を追えない（クリーンVM検証で同種の実害あり）。
    """
    detail = f"{message}\n\nログ: {supervisor.LOG_FILE}"
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("ワタシノセカイト初音ミク", detail)
        root.destroy()
    except Exception:
        print(detail)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        _fatal("スタート画面の起動に失敗しました。\n\n" + traceback.format_exc())
        sys.exit(1)
