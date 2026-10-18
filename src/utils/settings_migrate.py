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

"""settings.json の旧構造 → 新構造（2026-07 整理版）への一度きり自動マイグレーション。

Gateway の initialize_app が settings.json 読込前に呼ぶ。旧構造を検出した場合のみ
変換して保存する（変換前の内容は settings.json.bak に残る）。

新構造でも、**ひな形にあって設定に無いキーは補う**（`fill_missing_keys`）。
新規インストールは settings_sample.json を複製するので新しいキーが入るが、
既存の環境には入らないため、あとから増えた設定が settings.json に現れなかった。

旧 → 新 の対応:
    api.google.api_key                 → api_keys.google
    fish_audio.api_key                 → api_keys.fish_audio
    stt_realtime.fish_audio.api_key    → api_keys.fish_audio（TTS側が未設定の場合のみ）
    web_search.tavily_api_key          → api_keys.tavily
    email.smtp_password                → api_keys.gmail_app_password
    pydantic_ai.* / embedding.model_name → llm.*
    fish_audio.{voice_id等}             → tts.*
    stt_realtime.{provider,auto_send,fish_audio} → stt.*（provider の local_whisper は fish_audio へ）
    knowledge_base / memory_update.compress_threshold → memory.*
    email.{smtp_host,smtp_port,smtp_user,recipient} → email.*（enabled は smtp_user 設定済みなら true）
    web_search                         → web_search.enabled（tavily キー設定済みなら true）
    <svc>_service.{port,host,enabled}  → services.<svc>.*
    stt_realtime.elevenlabs            → 廃止（削除）

新構造でも、**2026-09-25 に削除した構成のキーは外す**（`remove_local_voice_options`）:
    tts.provider / system.voicevox_path / system.rvc_path / services.tts / stt.local_whisper
    stt.provider = "local_whisper" → "fish_audio"

内部サービスの待ち受けアドレスも外す（`remove_internal_hosts`・2026-10-08）:
    services.{kv,vdb,stt,agent,recognition}.host（127.0.0.1 固定にしたので読まない）
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from utils.settings_store import is_unset_secret, load_settings, save_settings

# 旧構造の指標キー（どれか1つでもあれば旧構造とみなす）
_LEGACY_MARKERS = ("api", "pydantic_ai", "stt_realtime", "knowledge_base", "memory_update")


def is_legacy_settings(settings: dict) -> bool:
    return any(key in settings for key in _LEGACY_MARKERS)


def _secret_or_empty(value: Any) -> str:
    """プレースホルダ・空は空文字に正規化して返す。"""
    return "" if is_unset_secret(value) else value


def migrate_settings(old: dict) -> dict:
    """旧構造の settings dict を新構造に変換して返す（入力は変更しない）。"""
    old = copy.deepcopy(old)
    old_fish_tts = old.get("fish_audio", {})
    old_stt = old.get("stt_realtime", {})
    old_stt_fish = old_stt.get("fish_audio", {})
    old_email = old.get("email", {})
    old_pyai = old.get("pydantic_ai", {})
    old_kb = old.get("knowledge_base", {})

    fish_key = _secret_or_empty(old_fish_tts.get("api_key"))
    if not fish_key:
        fish_key = _secret_or_empty(old_stt_fish.get("api_key"))

    smtp_user = _secret_or_empty(old_email.get("smtp_user"))
    tavily_key = _secret_or_empty(old.get("web_search", {}).get("tavily_api_key"))

    def svc(name: str, default_port: int, with_enabled: bool = False,
            with_host: bool = False) -> dict:
        section = old.get(f"{name}_service", {})
        entry = {"port": section.get("port", default_port)}
        if with_host:   # host を持つのは Gateway だけ（内部サービスは 127.0.0.1 固定・2026-10-08）
            entry["host"] = section.get("host", "0.0.0.0")
        if with_enabled:
            entry = {"enabled": bool(section.get("enabled", True)), **entry}
        return entry

    new: dict[str, Any] = {
        "api_keys": {
            "google": _secret_or_empty(old.get("api", {}).get("google", {}).get("api_key")),
            "fish_audio": fish_key,
            "tavily": tavily_key,
            "gmail_app_password": _secret_or_empty(old_email.get("smtp_password")),
        },
        "llm": {
            "flash_model": old_pyai.get("flash_model", "google-gla:gemini-3.8-flash"),
            "pro_model": old_pyai.get("pro_model", "google-gla:gemini-3.1-pro-preview"),
            "lite_model": old_pyai.get("filler_model", "google-gla:gemini-3.1-flash-lite"),
            "filler_enabled": bool(old_pyai.get("filler_enabled", True)),
            "embedding_model": old.get("embedding", {}).get("model_name", "models/gemini-embedding-2"),
        },
        "tts": {
            "fish_audio": {
                "voice_id": _secret_or_empty(old_fish_tts.get("voice_id")),
                "sample_rate": old_fish_tts.get("sample_rate", 44100),
                "latency": old_fish_tts.get("latency", "normal"),
                "backend": old_fish_tts.get("backend", "s2.1-pro"),
            },
        },
        "stt": {
            "provider": _stt_provider(old_stt.get("provider", "none")),
            "auto_send": {
                "enabled": bool(old_stt.get("auto_send", {}).get("enabled", True)),
                "delay_seconds": old_stt.get("auto_send", {}).get("delay_seconds", 5),
            },
            "fish_audio": {
                "language": old_stt_fish.get("language", "ja"),
                "speech_threshold": old_stt_fish.get("speech_threshold", 0.03),
                "silence_chunks": old_stt_fish.get("silence_chunks", 5),
                "ignore_timestamps": bool(old_stt_fish.get("ignore_timestamps", True)),
                "interim_interval_sec": old_stt_fish.get("interim_interval_sec", 2.0),
            },
        },
        "camera": {
            "enabled": bool(old.get("camera", {}).get("enabled", True)),
            "buffer_size": old.get("camera", {}).get("buffer_size", 3),
        },
        "conversation_reaction": {
            "enabled": bool(old.get("conversation_reaction", {}).get("enabled", True)),
            "cooldown_seconds": old.get("conversation_reaction", {}).get("cooldown_seconds", 5),
            "text_input_enabled": bool(old.get("conversation_reaction", {}).get("text_input_enabled", True)),
        },
        "email": {
            "enabled": bool(smtp_user),
            "smtp_host": old_email.get("smtp_host", "smtp.gmail.com"),
            "smtp_port": old_email.get("smtp_port", 587),
            "smtp_user": smtp_user,
            "recipient": _secret_or_empty(old_email.get("recipient")),
        },
        "web_search": {
            "enabled": bool(tavily_key),
        },
        "song_service": {
            "enabled": bool(old.get("song_service", {}).get("enabled", False)),
            "max_runtime_minutes": old.get("song_service", {}).get("max_runtime_minutes", 60),   # 書いてあればその値のまま（既定は 2026-09-30 に 30→60）
        },
        "memory": {
            "knowledge_base": {
                # path は 2026-07-26 に廃止（コード側で miku/data/chroma 固定）
                "chunk_size_child": old_kb.get("chunk_size_child", 1024),
                "top_k": old_kb.get("top_k", 10),
            },
            "compress_threshold": old.get("memory_update", {}).get("compress_threshold", 20000),
        },
        "services": {
            "gateway": svc("gateway", 5001, with_host=True),
            "kv": svc("kv", 8010, with_enabled=True),
            "vdb": svc("vdb", 8020, with_enabled=True),
            "stt": {
                "port": old_stt.get("port", 5006),
            },
            "agent": svc("agent", 5010),
            # 顔・声を1サービスに統合（2026-08-09。旧名 face_recognition）
            "recognition": svc("face_recognition", 5007),
        },
    }
    return new


def _stt_provider(value: str) -> str:
    """聞き取りの方法。削除した local_whisper は fish_audio へ置き換える。"""
    return "fish_audio" if value == "local_whisper" else value


def remove_local_voice_options(settings: dict) -> list[str]:
    """2026-09-25 に削除した構成（VOICEVOX+RVC・local_whisper）のキーを外す。

    声は Fish Audio だけになったので `tts.provider` は選ぶものが無い。
    聞き取りが local_whisper だった設定は fish_audio にする（使えないまま起動するより、
    APIキーが入っていれば聞き取れる方がよい。キーが無ければ起動時の検査が none へ落とす）。
    変えた内容の一覧を返す（空なら何もしていない）。
    """
    changed: list[str] = []

    def pop(section: str, key: str) -> None:
        parent = settings.get(section)
        if isinstance(parent, dict) and key in parent:
            parent.pop(key)
            changed.append(f"{section}.{key} を削除")

    pop("tts", "provider")
    pop("system", "voicevox_path")
    pop("system", "rvc_path")
    pop("services", "tts")
    pop("stt", "local_whisper")
    stt = settings.get("stt")
    if isinstance(stt, dict) and stt.get("provider") == "local_whisper":
        stt["provider"] = "fish_audio"
        changed.append("stt.provider を local_whisper → fish_audio")
    return changed


# 待ち受けアドレスを設定から外した内部サービス（2026-10-08）
INTERNAL_SERVICES = ("kv", "vdb", "stt", "agent", "recognition")


def remove_internal_hosts(settings: dict) -> list[str]:
    """内部サービスの `services.<名前>.host` を外す（2026-10-08）。

    KV（会話ログ）・VDB（記憶）・聞き取り・エージェント・顔と声の認識は**認証を持たない**ので、
    この PC の中（127.0.0.1）だけで待ち受けるようコードで固定した。以前のひな形は
    `0.0.0.0`（同じ Wi-Fi の誰からでも届く）を入れていたため、既存の環境ではその値が残っている。
    **値はもう読まない**ので、残っていると「ここで開ける」と誤解させるだけ——外す。
    変えた内容の一覧を返す（空なら何もしていない）。
    """
    changed: list[str] = []
    services = settings.get("services")
    if not isinstance(services, dict):
        return changed
    for name in INTERNAL_SERVICES + ("face_recognition",):
        entry = services.get(name)
        if isinstance(entry, dict) and "host" in entry:
            entry.pop("host")
            changed.append(f"services.{name}.host を削除")
    return changed


def rename_face_recognition(settings: dict) -> bool:
    """services.face_recognition → services.recognition へ寄せる（2026-08-09 の統合）。

    旧構造からの移行（migrate_settings）とは別に、**新構造だが旧サービス名のまま**の
    設定ファイルを更新する。変更したら True。
    """
    services = settings.get("services")
    if not isinstance(services, dict) or "face_recognition" not in services:
        return False
    old_entry = services.pop("face_recognition")
    services.setdefault("recognition", old_entry)
    return True


def rename_lite_model(settings: dict) -> bool:
    """llm.filler_model → llm.lite_model へ寄せる（2026-10-10・フィラー専用に見えたので改名）。

    値はそのまま移す。両方あれば lite_model を残し、filler_model を外す。変更したら True。
    **ひな形の補完（fill_missing_keys）より前に呼ぶ**——後だと既定値の lite_model が先に入り、利用者の値が消える。
    """
    llm = settings.get("llm")
    if not isinstance(llm, dict) or "filler_model" not in llm:
        return False
    old = llm.pop("filler_model")
    llm.setdefault("lite_model", old)
    return True


def fill_missing_keys(settings: dict, sample: dict, _prefix: str = "") -> list[str]:
    """ひな形にあって設定に無いキーを補う。補ったパスの一覧を返す。

    **既にある値は絶対に触らない。** 足りないものを足すだけ。

    これが無いと、**あとから増えた設定キーは既存の環境にいつまでも現れない**
    （新規インストールは settings_sample.json をそのまま複製するので入る）。
    コードの既定値が効くので壊れはしないが、settings.json を見ても
    「そんな設定がある」と分からない状態が続く。

    辞書の中だけを再帰する。リストは中身を混ぜず、丸ごと無い場合だけ複製する
    （要素単位で足すと、利用者が意図的に削った項目が復活してしまう）。
    """
    added: list[str] = []
    for key, sample_value in sample.items():
        path = f"{_prefix}{key}"
        if key not in settings:
            settings[key] = copy.deepcopy(sample_value)
            added.append(path)
        elif isinstance(sample_value, dict) and isinstance(settings[key], dict):
            added.extend(fill_missing_keys(settings[key], sample_value, f"{path}."))
    return added


def migrate_settings_file_if_needed(settings_path: Path) -> bool:
    """旧構造なら変換し、足りないキーを補って保存する。何かしたら True。

    - 旧構造 → 新構造へ一度きり変換する（変換前は settings.json.bak に残る）
    - `services.face_recognition` が残っていれば `recognition` へ寄せる
    - 削除した構成（VOICEVOX+RVC・local_whisper）のキーを外す
    - 内部サービスの待ち受けアドレス（`services.<名前>.host`）を外す
    - **ひな形（settings_sample.json）にあって設定に無いキーを補う**
    """
    settings = load_settings(settings_path)
    if not settings:
        return False
    if is_legacy_settings(settings):
        new = migrate_settings(settings)
        fill_missing_keys(new, _load_sample(settings_path))
        save_settings(new, settings_path)
        print("[settings_migrate] settings.json を新構造に変換しました"
              "（変換前の内容は settings.json.bak に保存）")
        return True

    changed = False
    if rename_face_recognition(settings):
        print("[settings_migrate] services.face_recognition を services.recognition に変更しました"
              "（顔・声の認識サービス統合）")
        changed = True

    if rename_lite_model(settings):
        print("[settings_migrate] llm.filler_model を llm.lite_model に改名しました（値はそのまま）")
        changed = True

    removed = remove_local_voice_options(settings)
    if removed:
        print("[settings_migrate] 使われなくなった設定を整理しました"
              f"（ローカル音声合成・音声認識の削除）: {', '.join(removed)}")
        changed = True

    hosts = remove_internal_hosts(settings)
    if hosts:
        print("[settings_migrate] 内部サービスの待ち受けアドレスの設定を外しました"
              "（この PC の中だけで待ち受けるよう固定。同じ Wi-Fi の他の端末からは届かない）: "
              f"{', '.join(hosts)}")
        changed = True

    added = fill_missing_keys(settings, _load_sample(settings_path))
    if added:
        print(f"[settings_migrate] 新しい設定キーを補いました（既定値のまま）: {', '.join(added)}")
        changed = True

    if changed:
        save_settings(settings, settings_path)
    return changed


def _load_sample(settings_path: Path) -> dict:
    """settings_sample.json を読む。無ければ空（補完をしないだけで、起動は妨げない）。"""
    sample_path = settings_path.parent / "settings_sample.json"
    return load_settings(sample_path) or {}
