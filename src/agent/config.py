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

"""Agent 側の設定ロード。

settings.json から API キー・Chroma パス・埋め込み設定を読み出し、
Pydantic AI が参照する環境変数（GEMINI_API_KEY 等）に反映する。

読み込み失敗は呼び出し側で捕捉すること（ここでは握り潰さない）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "settings.json"
_PLACEHOLDER_PREFIXES = ("put your", "your-")


def load_settings(path: Path | None = None) -> dict[str, Any]:
    target = path or _DEFAULT_SETTINGS_PATH
    with target.open("r", encoding="utf-8-sig") as f:      # BOM 付きでも読む（settings_store と同じ）
        return json.load(f)


def _is_placeholder(value: str) -> bool:
    v = (value or "").strip().lower()
    return any(v.startswith(p) for p in _PLACEHOLDER_PREFIXES)


def configure_env_from_settings(settings: dict[str, Any]) -> None:
    """API キーを環境変数に登録する（既存の環境変数は上書きしない）。"""
    key = settings.get("api_keys", {}).get("google", "")
    if key and not _is_placeholder(key):
        os.environ.setdefault("GEMINI_API_KEY", key)


def get_pydantic_ai_flash_model(settings: dict[str, Any]) -> str:
    return settings.get("llm", {}).get("flash_model", "google-gla:gemini-3.8-flash")


def get_pydantic_ai_pro_model(settings: dict[str, Any]) -> str:
    return settings.get("llm", {}).get("pro_model", "google-gla:gemini-3.1-pro-preview")


def get_pydantic_ai_lite_model(settings: dict[str, Any]) -> str:
    """軽いモデル（Flash Lite 系）。即応フィラー・曲の選び出しなど、速さが要る小さな仕事に使う。

    ★2026-10-10 にキーを `filler_model` → `lite_model` へ改名した（フィラー専用に見えたため・本人判断）。
    古い設定は起動時の `settings_migrate.rename_lite_model` が寄せる。

    本応答 LLM と並列で動かすため、最速モデル（Flash Lite 系）を固定指定する。
    ベンチで gemini-3.1-flash-lite が最も自然なフィラー生成を確認済み。
    詳細: .claude/skills/miku-dev/details/latency_benchmark.md
    """
    return settings.get("llm", {}).get("lite_model", "google-gla:gemini-3.1-flash-lite")


def get_pydantic_ai_filler_enabled(settings: dict[str, Any]) -> bool:
    """即応フィラー機能を有効化するか。

    True: gateway が on_filler_ready コールバックを agent に渡し、フィラー LLM が並列起動する。
    False: フィラーは生成されず、本応答 LLM の結果のみが TTS される。デバッグや切り戻し用。
    """
    return bool(settings.get("llm", {}).get("filler_enabled", True))


def get_embedding_config(settings: dict[str, Any]) -> dict[str, Any]:
    api_key = settings.get("api_keys", {}).get("google", "")
    return {
        "model_name": settings.get("llm", {}).get("embedding_model", "models/gemini-embedding-2"),
        "api_key": api_key,
    }


def get_knowledge_base_config(settings: dict[str, Any]) -> dict[str, Any]:
    kb = settings.get("memory", {}).get("knowledge_base", {})
    return {
        # Chroma の保存先は固定（miku_agent ルートからの相対）。他のユーザーデータ同様に
        # 保存先は設定項目にしない（settings からキーごと廃止・2026-07-26）
        "path": "miku/data/chroma",
        "chunk_size_child": kb.get("chunk_size_child", 256),
        "top_k": kb.get("top_k", 10),
    }


def get_kv_base_url(settings: dict[str, Any]) -> str:
    kv = settings.get("services", {}).get("kv", {})
    port = kv.get("port", 8010)
    return f"http://127.0.0.1:{port}"


def get_vdb_base_url(settings: dict[str, Any]) -> str:
    vdb = settings.get("services", {}).get("vdb", {})
    port = vdb.get("port", 8020)
    return f"http://127.0.0.1:{port}"


def get_agent_service_base_url(settings: dict[str, Any]) -> str:
    cfg = settings.get("services", {}).get("agent", {})
    port = cfg.get("port", 5010)
    return f"http://127.0.0.1:{port}"


def get_recognition_base_url(settings: dict[str, Any]) -> str:
    """マスター認識サービス（顔・声）のベース URL。

    旧キー `services.face_recognition` も読む（settings_migrate で `recognition` へ
    寄せるが、移行前の設定ファイルでも動くようにするため）。
    """
    services = settings.get("services", {})
    cfg = services.get("recognition") or services.get("face_recognition", {})
    port = cfg.get("port", 5007)
    return f"http://127.0.0.1:{port}"


def get_email_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """SMTP 送信設定を返す。未設定キーがあれば KeyError で即停止（サイレント失敗禁止）。"""
    em = settings.get("email", {})
    return {
        "smtp_host": em["smtp_host"],
        "smtp_port": int(em["smtp_port"]),
        "smtp_user": em["smtp_user"],
        "smtp_password": settings["api_keys"]["gmail_app_password"],
        "recipient": em["recipient"],
    }


def get_memory_compress_threshold(settings: dict[str, Any]) -> int:
    """always_in_context セクションの圧縮発動閾値（文字数）を返す。"""
    return int(settings.get("memory", {}).get("compress_threshold", 20_000))


def get_tavily_api_key(settings: dict[str, Any]) -> str:
    """Tavily API キーを返す。未設定なら KeyError で即停止。"""
    return settings["api_keys"]["tavily"]


def _is_usable_value(value: Any) -> bool:
    """設定値が実質的に設定済みか（空・プレースホルダでないか）。"""
    return bool(value) and isinstance(value, str) and not _is_placeholder(value)


def is_web_search_available(settings: dict[str, Any]) -> bool:
    """web_search 機能が使えるか（enabled かつ Tavily キー設定済み）。

    False のときミクのいまの状態（`<system_state>`）に「使えない」と出て
    （`agent_main._format_tool_availability`）、dispatch 側でも呼び出しを拒否する。
    """
    if not settings.get("web_search", {}).get("enabled", True):
        return False
    return _is_usable_value(settings.get("api_keys", {}).get("tavily", ""))


def get_song_service_dir() -> "Path":
    """song_service の実体ディレクトリ（miku/src/song_service）。"""
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "song_service"


def get_songs_dir() -> "Path":
    """**曲の置き場**（`miku/data/songs`）。曲フォルダ・台帳・実行状態がここに入る。

    2026-09-01 に `song_service/output/` から移した——**ソースツリーの中に成果物を置かない**。
    パスを組み立てる場所が **8箇所に散っていた**ので、ここ1つに寄せる
    （散っていると、片方だけ直したときに「作る場所と読む場所が違う」という形で壊れる）。
    """
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "data" / "songs"


def is_song_service_available(settings: dict[str, Any]) -> bool:
    """work_on_music（曲の作成・編集）が使えるか（enabled かつ実体=song_runner と統合venvが配置済み）。

    False のときミクのいまの状態（`<system_state>`）に「使えない」と出て、dispatch 側でも呼び出しを拒否する。
    実体チェックを含むのは、配布先で setup_song_service.ps1 未実行のまま enabled にされた場合に
    「歌えるのに動かない」を防ぐため。"""
    if not settings.get("song_service", {}).get("enabled", False):
        return False
    svc = get_song_service_dir()
    return (svc / "song_runner.py").exists() and (svc / "venv" / "Scripts" / "python.exe").exists()


def is_audio_cover_available(settings: dict[str, Any]) -> bool:
    """**音源しか無い曲のカバー**まで作れる導入か（2026-09-23）。

    カバーには**2つの経路**があり、要るものが違う:

    | 経路 | 中身 | GPU |
    |---|---|---|
    | **MIDI のお手本が見つかる** | 記号の対応づけ＋元 MIDI を鳴らす | **要らない** |
    | 音源しか無い | 音源分離 → 採譜 → 強制アラインメント | **要る** |

    ★**ここが見るのは「どう導入されたか」だけ**——`-OriginalOnly` で入れた環境には
      `sofa_venv` / `venv_mt3` が無い。**GPU が実際に使えるか**の判定は
      `song_service/tools/creation.py` の `_audio_cover_available()` が実行時に行う
      （あちらは `nvidia-smi` まで見る）。

    ★**同じ判定を2か所に書いているのではない。** サービスをまたいで import しない規約があり
      （例外は標準ライブラリだけで書いた名前の決まり `song_service/song_files.py` だけ。`creation.py` は
      song_service の venv の依存を引き込むので本体からは読めない）、そのうえで**見ている軸が違う**——こちらは**ミクのいまの状態（`<system_state>`）に何と書くか**を決めるための「導入の形」、
      あちらは**実際に走らせてよいか**。**あちらが最後の砦**で、こちらが外しても曲は壊れない
      （ミクが「できます」と言ってから断ることになるだけ）。
    """
    if not is_song_service_available(settings):
        return False
    svc = get_song_service_dir()
    return (svc / "sofa_venv").exists() and (svc / "venv_mt3").exists()


def is_email_available(settings: dict[str, Any]) -> bool:
    """send_email 機能が使えるか（enabled かつ アプリパスワード・送信元設定済み）。"""
    if not settings.get("email", {}).get("enabled", True):
        return False
    return (
        _is_usable_value(settings.get("api_keys", {}).get("gmail_app_password", ""))
        and _is_usable_value(settings.get("email", {}).get("smtp_user", ""))
    )
