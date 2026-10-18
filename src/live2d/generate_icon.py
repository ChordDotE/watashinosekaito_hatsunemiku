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

"""Live2D モデルからアプリのアイコン（.ico）を生成する。

**なぜユーザーの PC で生成するのか**:
Live2D 公式サンプルモデルは再配布できないため、モデルから起こした画像を配布物へ同梱できない。
利用者が自分でダウンロードしたモデルから、利用者の PC 上で生成すれば再配布に当たらない。
生成できるまでのあいだは、同梱の簡易マーク（miku/app_icon.ico）が使われる。

**使い方**（セットアップから呼ばれる。単体でも実行できる）:
    python generate_icon.py                # runtime/miku_icon.ico を作る
    python generate_icon.py --keep-png     # 確認用に PNG も残す

ブラウザは Windows 標準の Edge（無ければ Chrome）をヘッドレスで使う。追加のダウンロードは不要。
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

LIVE2D_DIR = Path(__file__).resolve().parent
MIKU_DIR = LIVE2D_DIR.parents[1]
RUNTIME_DIR = MIKU_DIR / "runtime"
OUT_ICO = RUNTIME_DIR / "miku_icon.ico"
# ショートカットが指すアイコン。インストーラが同梱の簡易マークをここへ置き、生成できたら上書きする。
# ショートカット自体は書き換えない——全ユーザー共通のデスクトップにあり、管理者でないと書き換えられないため
# （インストール中の生成に失敗したあと、スタート画面から作り直しても絵が変わらなかった・2026-10-18）
SHORTCUT_ICO = RUNTIME_DIR / "shortcut_icon.ico"

# --- 描画・切り出しの条件 -----------------------------------------------------
# ここの数値は「確認済みのアイコンとまったく同じ絵」を得るための固定値。
# 2026-08-01 にモーションラボで詰めた構図をそのまま再現している（変更すると絵が変わる）。
RENDER_SCALE = 3                      # 論理 700x1000 → 実ピクセル 2100x3000
CANVAS_W, CANVAS_H = 700, 1000
# 切り出し枠（レンダリング画像に対する比率）。
# 内訳: ラボの「顔まわり切り出し」(x 0.22〜0.78 / y 0〜0.45) の中で
#       中心(0.357, 0.356)・一辺 0.467 を取った枠を、全体座標へ直したもの
CROP_RATIO = (566 / 2100, 165 / 3000, 1196 / 2100, 795 / 3000)
ICON_SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
BG_TOLERANCE = 14                     # 背景（#e8e8e8）と見なす色差

PAGE_TIMEOUT_SEC = 60


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _start_server(root: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    """ローカル配信。file:// だと fetch/WebGL が制限されるため HTTP で開く。"""
    handler = functools.partial(_QuietHandler, directory=str(root))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def render_png_webview(url: str, out_png: Path) -> bool:
    """WebView2（pywebview）でページを開き、キャンバスを PNG として取り出す。

    **Edge の `--screenshot` は使えない**: Windows 11 の Edge では終了コード 0 のまま
    ファイルが作られずエラーも出ない（2026-08-02 実測）。同じ Edge のエンジンでも
    WebView2 経由なら描画結果を dataURL で取得できるため、こちらを使う。
    pywebview はコントロールパネルで既に使っているので追加の依存も増えない。
    """
    import base64

    import webview

    result = {"ok": False}
    window = webview.create_window(
        "アイコンを作成しています…", url,
        width=CANVAS_W, height=CANVAS_H, hidden=True,
    )

    def worker() -> None:
        try:
            deadline = time.time() + PAGE_TIMEOUT_SEC
            while time.time() < deadline:
                if window.evaluate_js("window.READY === true"):
                    break
                err = window.evaluate_js("window.LAST_ERROR")
                if err:
                    print(f"警告: レンダラーでエラーが発生しました: {err}")
                    return
                time.sleep(0.5)
            else:
                print("警告: レンダリングがタイムアウトしました")
                return

            data_url = window.evaluate_js("window.snapshot()")
            if not data_url or "," not in data_url:
                return
            out_png.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
            result["ok"] = out_png.exists() and out_png.stat().st_size > 0
        finally:
            window.destroy()

    webview.start(worker, private_mode=True)
    return result["ok"]


def find_chrome() -> str | None:
    """Chrome 系のフォールバック（WebView2 が使えない環境向け）。"""
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    for name in ("chrome", "chromium", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def render_png_chrome(browser: str, url: str, out_png: Path) -> bool:
    """Chrome のヘッドレス撮影（フォールバック）。"""
    with tempfile.TemporaryDirectory(prefix="miku_icon_") as profile:
        cmd = [
            browser, "--headless=new", f"--screenshot={out_png}",
            f"--window-size={CANVAS_W * RENDER_SCALE},{CANVAS_H * RENDER_SCALE}",
            "--force-device-scale-factor=1", "--hide-scrollbars",
            # GPU が無い環境（仮想マシン等）でもソフトウェア描画で通す
            "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            "--virtual-time-budget=20000",
            f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
            url,
        ]
        try:
            subprocess.run(cmd, timeout=PAGE_TIMEOUT_SEC, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return out_png.exists() and out_png.stat().st_size > 0


def crop_to_icon(png_path: Path, out_ico: Path, keep_png: bool = False) -> bool:
    """スクリーンショットを切り出して .ico にする（背景は透過にする）。"""
    from PIL import Image

    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    box = (int(w * CROP_RATIO[0]), int(h * CROP_RATIO[1]),
           int(w * CROP_RATIO[2]), int(h * CROP_RATIO[3]))
    head = img.crop(box).resize((512, 512), Image.LANCZOS)

    # レンダラーの背景色（左上隅の色）を抜く。どのテーマでも馴染むようにする
    bg = head.getpixel((0, 0))[:3]
    px = head.load()
    for y in range(head.height):
        for x in range(head.width):
            r, g, b, a = px[x, y]
            if (abs(r - bg[0]) < BG_TOLERANCE and abs(g - bg[1]) < BG_TOLERANCE
                    and abs(b - bg[2]) < BG_TOLERANCE):
                px[x, y] = (r, g, b, 0)

    out_ico.parent.mkdir(parents=True, exist_ok=True)
    head.save(out_ico, format="ICO", sizes=ICON_SIZES)
    if keep_png:
        head.save(out_ico.with_suffix(".png"))
    return out_ico.exists()


def generate(keep_png: bool = False) -> bool:
    model = LIVE2D_DIR / "CubismWebSamples/Samples/Resources/miku_sample_t04/miku_sample_t04.model3.json"
    if not model.exists():
        print("Live2D モデルが未配置のため、アイコン生成をスキップします（同梱のアイコンを使います）")
        return False
    srv, port = _start_server(LIVE2D_DIR)
    try:
        url = f"http://127.0.0.1:{port}/icon_render.html?scale={RENDER_SCALE}"
        with tempfile.TemporaryDirectory(prefix="miku_icon_shot_") as tmp:
            shot = Path(tmp) / "shot.png"
            print("Live2D モデルからアイコンを生成しています...")

            rendered = False
            try:
                rendered = render_png_webview(url, shot)
            except Exception as e:
                print(f"WebView での描画に失敗しました（Chrome を試します）: {e}")
            if not rendered:
                chrome = find_chrome()
                if chrome:
                    rendered = render_png_chrome(chrome, url, shot)

            if not rendered:
                print("警告: アイコンの描画に失敗しました（同梱のアイコンを使います）")
                return False
            if not crop_to_icon(shot, OUT_ICO, keep_png):
                print("警告: アイコンの書き出しに失敗しました")
                return False
    finally:
        srv.shutdown()

    print(f"アイコンを作成しました: {OUT_ICO}")
    publish_to_shortcuts()
    return True


def publish_to_shortcuts() -> None:
    """生成したアイコンを、ショートカットが指すファイルへ写し、エクスプローラーに描き直させる。"""
    try:
        shutil.copyfile(OUT_ICO, SHORTCUT_ICO)
    except OSError as e:
        print(f"警告: ショートカットのアイコンを差し替えられませんでした: {e}")
        return
    try:
        import ctypes
        SHCNE_ASSOCCHANGED, SHCNF_IDLIST = 0x08000000, 0x0000
        ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)
    except Exception:
        pass   # 描き直しは次のサインインでも起きるので、ここで失敗しても絵は変わる


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-png", action="store_true", help="確認用に PNG も残す")
    a = ap.parse_args()
    # 失敗しても同梱アイコンで動作するため、セットアップを止めないよう常に 0 を返す
    generate(keep_png=a.keep_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
