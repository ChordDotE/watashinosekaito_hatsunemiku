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

"""ログイン保護のロジック（セッション鍵・アクセス元判定・失敗ロック）。

Flask に依存しない純粋ロジックのみ（gateway_service.py のルートから呼ばれる）。
パスワードのハッシュ化・検証は `utils/settings_store.py`（設定の読み書きと同居）。

## 方針（2026-08-06 ユーザーと確定）

- **ID は持たない**。単一マスター前提のシステムで、誰がログインしても同じミク・同じ記憶に
  つながるため、識別子としての ID は意味を持たない（将来マルチテナント提供をするなら、
  そのとき ID は「テナント識別子」として入る。docs/dev/board.md §5 の棚）
- **パスワードを設定すれば保護ON・空ならOFF**（有効/無効スイッチは作らない。
  「スイッチは入っているのに値が無い」食い違いを作れなくする既存方針に揃える）
- この PC からのアクセス（真のローカル）は常に免除。コントロールパネルが叩くため
"""
from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path

# アクセス元がローカルに見えても「経由」を示すヘッダがあれば外部扱いにする。
# cloudflared は http://localhost:<port> へ転送するため、**トンネル経由でも
# remote_addr は 127.0.0.1 になる**。ヘッダで判別しないと認証が素通りする
PROXY_HEADERS = ("CF-Connecting-IP", "CF-Ray", "X-Forwarded-For", "X-Real-IP")

_LOCAL_ADDRS = {"127.0.0.1", "::1", "localhost"}


def get_or_create_session_key(runtime_dir: Path) -> str:
    """セッション署名鍵を runtime から読む。無ければ生成して保存する。

    ファイルに残すことで **Gateway を再起動してもログイン状態が続く**
    （毎回変わるとセッションが全部無効になり、再起動のたびに再ログインになる）。
    """
    runtime_dir = Path(runtime_dir)
    key_path = runtime_dir / "session_key"
    try:
        if key_path.exists():
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                return key
    except OSError:
        pass  # 読めなければ作り直す（既存セッションは無効になるが安全側）

    key = secrets.token_hex(32)
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key, encoding="utf-8")
    except OSError as e:
        # 保存できなくても動作は続ける（再起動でセッションが切れるだけ）
        print(f"[auth] セッション鍵を保存できませんでした（再起動で再ログインが必要）: {e}")
    return key


def is_local_request(remote_addr: str | None, headers=None) -> bool:
    """この PC 自身からのアクセスか（＝認証を免除してよいか）を判定する。

    remote_addr がローカルでも、プロキシ経由を示すヘッダがあれば **外部扱い**にする。
    headers は dict 互換（Flask の request.headers を想定）。
    """
    if remote_addr not in _LOCAL_ADDRS:
        return False
    if headers:
        for name in PROXY_HEADERS:
            if headers.get(name):
                return False
    return True


def is_same_origin(origin: str | None, host: str | None) -> bool:
    """このブラウザ要求が、**自分が配ったページ**から来たものか（2026-09-20 追加）。

    ★**なぜ要るか**: `is_local_request()` は接続元アドレスだけを見るので、
    **利用者が開いた別のサイト**から `http://localhost:5001` へ投げられた要求も
    「本人」と判定してしまう。実測で、別オリジンのページから
    **ミクに話しかけ、返答をそのまま読み取れた**（2026-09-20・検証）。

    判定は**同一オリジンかどうか**だけ。許可リストを持たないので、
    localhost でも LAN の IP でも、外部公開のトンネル URL でも、
    **利用者が実際に開いた住所と同じなら通る**（＝設定不要で、抜けもない）。

    - `origin` が無い要求は**ブラウザ以外**（スタート画面・内部サービス・curl）。
      ここは従来どおり通す（ブラウザの同一生成元ポリシーの外なので、この検査の対象外）。
    - `Origin` は**ブラウザが必ず付け、ページの JavaScript からは偽装できない**。

    Args:
        origin: `Origin` ヘッダ（例: `http://localhost:5001`）
        host:   `Host` ヘッダ（例: `localhost:5001`）
    """
    if not origin:
        return True                      # ブラウザ以外からの要求
    if not host:
        return False
    netloc = origin.split("://", 1)[-1].split("/", 1)[0]
    return netloc.lower() == host.lower()


class LoginLimiter:
    """ログイン失敗の回数を数え、続いたら一定時間ロックする（総当たり対策）。

    状態はメモリだけに持つ。**ミクを再起動すればロックは消える**（＝復旧手段が常にある）。
    コントロールパネルからの明示解除は reset_all() を使う。
    """

    def __init__(self, threshold: int = 10, lock_minutes: int = 5) -> None:
        self.threshold = max(1, int(threshold))
        self.lock_seconds = max(1, int(lock_minutes)) * 60
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}   # key -> 失敗時刻のリスト
        self._locked_until: dict[str, float] = {}

    def locked_seconds_left(self, key: str) -> int:
        """ロック中なら残り秒数、ロックされていなければ 0。"""
        with self._lock:
            until = self._locked_until.get(key, 0.0)
            left = until - time.time()
            if left <= 0:
                self._locked_until.pop(key, None)
                return 0
            return int(left) + 1

    def record_failure(self, key: str) -> int:
        """失敗を1件記録し、ロックされた場合は残り秒数を返す（されなければ 0）。"""
        now = time.time()
        with self._lock:
            # 直近 lock_seconds 以内の失敗だけを数える（古い失敗を持ち越さない）
            recent = [t for t in self._failures.get(key, []) if now - t < self.lock_seconds]
            recent.append(now)
            self._failures[key] = recent
            if len(recent) >= self.threshold:
                self._locked_until[key] = now + self.lock_seconds
                self._failures[key] = []
                return self.lock_seconds
        return 0

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)

    def reset_all(self) -> int:
        """全ロック・全失敗回数を解除する。戻り値は解除したロック数。"""
        with self._lock:
            count = len(self._locked_until)
            self._failures.clear()
            self._locked_until.clear()
        return count

    def locked_count(self) -> int:
        """現在ロック中の相手の数（コントロールパネルの表示用）。"""
        now = time.time()
        with self._lock:
            return sum(1 for until in self._locked_until.values() if until > now)
