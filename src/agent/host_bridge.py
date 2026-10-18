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

"""ホスト（Gateway）が持つ能力を、Agent 側のツールへ**登録して**渡すための受け口。

## なぜ必要か（2026-08-15 の実害から）

ツールが `from gateway_service import ...` と書くと、**必ず壊れる**:

- Gateway は `python gateway_service.py` で起動されるので、そのモジュール名は **`__main__`**
- そこへ `import gateway_service` すると `sys.modules` に無いため**同じファイルが別モジュールとして
  再読み込み**され、`session_mapping` / `active_session_id` などの状態が**空の別インスタンス**になる
- 結果、ブラウザが繋がっているのに「繋がっていない」と判定される（`play_song` で実際に発生）
- おまけに `app = Flask(...)` / `socketio = SocketIO(...)` まで二重生成される

依存の向きの点でも、このプロジェクトは **Gateway → Agent の一方向**で統一している
（Agent 側から Gateway を import しているのはこの事故の1箇所だけだった）。

## 使い方

- **ホスト側**（Gateway の `initialize_app`）: 起動時に実装を登録する

      from agent.host_bridge import set_song_playback_reserver
      set_song_playback_reserver(reserve_song_playback)

- **ツール側**: 関数を取り出して呼ぶ（未登録なら None が返るので、正直に断れる）

      reserve = get_song_playback_reserver()
      mode = reserve(song_id, title, version) if reserve else None
      if not mode: ...                       # 送れなかった
      # mode（実際に決まった "sing"／"play"／"preview"）で言い方を選ぶ

`agent.host_bridge` という**パッケージ経由の同一モジュール**をホストとツールが共有するので、
`__main__` 問題は起きない。
"""
from __future__ import annotations

from collections.abc import Callable

# 曲の再生をブラウザへ予約する関数。(song_id, title, version, *, audio, mode) -> 実際に決めた mode（送れなければ None）
# version=None は鳴らせる版のうち最新の完成品。数値ならその版の完成品 `master_<形>_v{N}.wav` と
# `lipsync_v{N}.json` を**セットで**渡す（歌と口の版を必ず揃えるため。名前の決まりは `song_service/song_files.py`）
#
# ★**audio / mode は「歌う」以外のために足した**（2026-09-23・play は 2026-10-03）。
#   `audio` … 鳴らすファイル名を明示する（作りかけを聴かせる `preview.wav`）
#   `mode`  … "sing"（実歌声＝歌う）／"play"（歌の入らない曲を流す）／"preview"（まだ歌になっていない音を聴かせる）
#
#   ★**口を動かすかは、フラグではなく「口の形のデータがあるか」で決まる**（本人指示）。
#   受け側が `lipsync_v{N}.json` の実在を見て、無ければ口の情報を送らない＝ミクは歌わず、
#   曲を流すだけになる（"sing" で頼んでも "play" に落ちる）。だから**受け側は実際に決めた mode を返し**、
#   呼ぶ側はその mode で言い方を選ぶ（「歌うね」と言って口が動かない食い違いを作らない・2026-10-03）。
SongPlaybackReserver = Callable[..., str | None]

_song_playback_reserver: SongPlaybackReserver | None = None


def set_song_playback_reserver(fn: SongPlaybackReserver | None) -> None:
    """曲の再生予約の実装を登録する（Gateway の初期化から呼ぶ）。"""
    global _song_playback_reserver
    _song_playback_reserver = fn


def get_song_playback_reserver() -> SongPlaybackReserver | None:
    """登録済みの実装を返す。**未登録なら None**（＝ブラウザへ届ける手段が無い）。

    未登録になるのは「Gateway 以外から Agent を動かしている」場合
    （CLI・テスト・単体実行）。そのときは歌を届けられないので、ツールは断る。
    """
    return _song_playback_reserver


# ブラウザが開いているか（2026-10-09）。() -> {"connected": bool, "minutes_since_input": float | None}
# `<system_state>` に載せ、自発話しかけを声で届けるかメールにするかをミクが決める材料にする。
# 開いていても席を外していることがあるので、最後にマスターが話しかけてからの分数も添える。
BrowserPresence = Callable[[], dict]

_browser_presence: BrowserPresence | None = None


def set_browser_presence(fn: BrowserPresence | None) -> None:
    """ブラウザの接続状態を返す実装を登録する（Gateway の初期化から呼ぶ）。"""
    global _browser_presence
    _browser_presence = fn


def get_browser_presence() -> BrowserPresence | None:
    """登録済みの実装を返す。**未登録なら None**（Gateway 以外から動かしている＝状態は載せない）。"""
    return _browser_presence
