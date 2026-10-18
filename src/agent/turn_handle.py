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
turn_handle.py - 進行中ターンへの「考え中の追加発話」注入ハンドル（P2 畳み込み）

ターン開始時に gateway が生成して runner まで渡す共有オブジェクト。
実行中の関数のローカル変数へ外部から干渉するのではなく、**両者が最初から参照を持つ
このハンドルを介した協調方式**で通信する（既存の `_current_fish_cancel`
〔threading.Event を gateway が set し Fish ストリーマーが自分のループで確認して抜ける〕
と同じ流儀の一般化。concurrency.md 参照）。

役割分担:
  gateway  : 進行中ターンがあれば try_inject() で追加発話を積む（新規ターンを起動しない）
  runner   : ストリーム読み取り中は abort_stream を自発チェックして生成を中断し、
             ループ境界で drain() して入力に時系列追記する
  agent_main: ターン終了時、collected を会話ログの user 行としてまとめて KV に書く

クローズ判定（レース対策）:
  「ターンが終わる瞬間」と「追加発話の到着」が競合しうるため、lock の下で
  close_if_idle()（runner 側・キューが空なら閉じる）と try_inject()（gateway 側・
  閉じていたら拒否 → 呼び出し元は通常の新規ターンとして処理）を排他する。

発話中の確定保留（2026-08-11）:
  ユーザーが話している最中にターンを確定させると、その発話を聞かないまま返事が始まる。
  ブラウザは録音の開始・終了を **一報**（Socket.IO の user_speaking）として送り、gateway が
  set_user_speaking() で反映する。runner は
    - 話し始め → abort_stream が立つので生成を中断（デルタが TTS へ流れる前に止める）
    - 確定直前 → 話している間は閉じずに追加発話の到着を待つ（上限 wait_timeout_sec）
  ブラウザ状態の複製ではなく「その時点の一報」である点が重要（decisions.md 2026-08-03 の
  写し＋照合点方式に沿う）。切断・取りこぼしに備え、待ちは必ず上限で打ち切る。
"""

from __future__ import annotations

import queue
import threading

#: 発話中に追加発話の到着を待つ上限（秒）。gateway が settings の
#: stt.auto_send.delay_seconds を渡す。ここは渡されなかった場合の保険値
DEFAULT_WAIT_TIMEOUT_SEC = 5.0


class TurnHandle:
    """1ターンぶんの注入ハンドル。ターンごとに使い捨てる。"""

    def __init__(self, wait_timeout_sec: float = DEFAULT_WAIT_TIMEOUT_SEC) -> None:
        self._lock = threading.Lock()
        self._queue: "queue.Queue[tuple[str, str]]" = queue.Queue()  # (到着時刻, テキスト)
        self.abort_stream = threading.Event()
        self._closed = False
        # ターン中に注入された全発話（KV 保存用）。runner の drain() が積む
        self.collected: list[tuple[str, str]] = []
        # ユーザーが話している最中か（ブラウザからの一報。set_user_speaking が書く）
        self._user_speaking = False
        self.wait_timeout_sec = float(wait_timeout_sec)
        # いまのループの応答音声が**実際に鳴り始めたか**（gateway が初回 PCM 送出時に立てる）。
        # 中断時に「どこまで発話済みか」を二値で決めるために使う（下の mark_audio_started 参照）
        self._audio_started = False

    # ------------------------------------------------------------------
    # gateway 側
    # ------------------------------------------------------------------

    def try_inject(self, arrival_time: str, text: str) -> bool:
        """追加発話を注入する。ターンが既に閉じていれば False（→通常の新規ターンへ）。"""
        with self._lock:
            if self._closed:
                return False
            self._queue.put((arrival_time, text))
            self.abort_stream.set()  # 生成中なら中断してもらう（ツール実行中は無視される）
            return True

    def mark_audio_started(self) -> None:
        """いまのループの応答音声が実際に鳴り始めた（初回 PCM をブラウザへ送った）。

        **半二重（鳴っている間はマイクがミュート）なので、声で割り込めたということは
        その時点で音が鳴っていない**——したがって中断時の「発話済みか」は
        「鳴り始めたか」の二値で決められる（2026-09-22 本人指摘）。
        """
        with self._lock:
            self._audio_started = True

    def reset_audio_started(self) -> None:
        """ループ境界で倒す。フラグは**そのループの応答音声**についてのものなので、
        次のループを作り直すたびに「まだ鳴っていない」へ戻す。"""
        with self._lock:
            self._audio_started = False

    @property
    def audio_started(self) -> bool:
        """いまのループの応答音声が鳴り始めたか。"""
        with self._lock:
            return self._audio_started

    def set_user_speaking(self, speaking: bool) -> None:
        """ユーザーの発話開始・終了の一報を反映する。

        話し始めたら abort_stream を立てて生成を止める。まだ追加発話は届いていないが、
        **デルタが TTS へ流れてしまう前に止める**ことが目的（流れた分は取り消せない）。
        runner は中断後、追加発話が届くまで待ってから作り直す。
        """
        with self._lock:
            if self._closed:
                return
            self._user_speaking = speaking
        if speaking:
            self.abort_stream.set()

    # ------------------------------------------------------------------
    # runner 側
    # ------------------------------------------------------------------

    def is_user_speaking(self) -> bool:
        """ユーザーが話している最中か（一報ベース）。"""
        with self._lock:
            return self._user_speaking

    def has_pending(self) -> bool:
        """未取り込みの追加発話があるか（待機ループの終了判定に使う）。"""
        return not self._queue.empty()

    def drain(self) -> list[tuple[str, str]]:
        """溜まっている追加発話を取り出して返す（collected にも記録）。abort フラグも下ろす。"""
        items: list[tuple[str, str]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        self.abort_stream.clear()
        self.collected.extend(items)
        return items

    def close_if_idle(self) -> bool:
        """キューが空ならターンを閉じて True。追加発話が残っていれば閉じず False。

        runner が最終応答を確定させる直前に呼ぶ。False の場合は drain() して再ループする。
        """
        with self._lock:
            if not self._queue.empty():
                return False
            self._closed = True
            return True

    def force_close(self) -> None:
        """例外時などにターンを確実に閉じる（以後の注入は新規ターン扱いになる）。"""
        with self._lock:
            self._closed = True
