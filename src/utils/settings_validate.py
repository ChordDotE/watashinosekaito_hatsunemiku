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

"""settings.json のクロスフィールド検証（有効にした機能に必要なキーがあるか）。

使う場所は2つ（ルールは共通）:
  1. 設定GUIの保存時（PUT /api/settings）: 違反があれば保存自体を拒否する
  2. Gateway 起動時: 手動編集対策。違反機能はエラー表示のうえ該当機能のみ停止して続行する

エラー dict のキーは設定GUIのフィールドパスに合わせる（該当フィールドの下にエラーが出る）。
"""
from __future__ import annotations

from typing import Any

from utils.settings_store import is_unset_secret

def _key_set(settings: dict, name: str) -> bool:
    return not is_unset_secret(settings.get("api_keys", {}).get(name))


def validate_settings(settings: dict[str, Any]) -> dict[str, str]:
    """有効化と前提条件（APIキー）の整合を検証する。

    Returns:
        {GUIフィールドパス: エラーメッセージ}。空 dict なら違反なし。
        api_keys.google の欠損は含めない（機能トグルではなく常時必須のため、
        起動時警告は google_key_missing() で別途行う）。
    """
    errors: dict[str, str] = {}
    fish_key = _key_set(settings, "fish_audio")

    if not fish_key:
        errors["api_keys.fish_audio"] = "ミクの声を作るには Fish Audio の APIキーの入力が必要です"

    stt_provider = settings.get("stt", {}).get("provider", "none")
    if stt_provider == "fish_audio" and not fish_key:
        errors["stt.provider"] = "Fish Audio を使うには APIキー（api_keys.fish_audio）の入力が必要です"

    if settings.get("web_search", {}).get("enabled", False) and not _key_set(settings, "tavily"):
        errors["web_search.enabled"] = "Web検索を使うには APIキー（api_keys.tavily）の入力が必要です"

    email = settings.get("email", {})
    if email.get("enabled", False):
        if not _key_set(settings, "gmail_app_password"):
            errors["email.enabled"] = "メール送信を使うには Gmail アプリパスワード（api_keys.gmail_app_password）の入力が必要です"
        elif is_unset_secret(email.get("smtp_user")):
            errors["email.enabled"] = "メール送信を使うには送信元メールアドレス（email.smtp_user）の入力が必要です"


    return errors


def save_notices(settings: dict[str, Any]) -> dict[str, str]:
    """保存は通すが、そのままでは機能しない設定への案内（設定GUIの保存後に表示する）。

    errors と分けているのは「設定値としては正しいが、環境側の準備が残っている」ケースだから。
    保存を拒否すると準備後にもう一度 GUI へ戻る必要があり、かえって迷わせる。
    """
    notices: dict[str, str] = {}
    # 外部公開しているのにログイン保護が無い状態（requirements §5.4 が求める認証が欠けている）。
    # 保存も起動も止めないが、URL を知られたら誰でも会話・記憶の閲覧ができるため案内する
    if settings.get("system", {}).get("tunnel_enabled", False):
        auth = (settings.get("security", {}) or {}).get("auth", {}) or {}
        if not (auth.get("password_hash") or ""):
            notices["security.auth.password_hash"] = (
                "外部アクセスが有効ですが、ログインのパスワードが未設定です。"
                "アクセス用URLを知った人は誰でもミクと話せてしまいます"
                "（会話の記録も読めます）。この画面の「ログインの保護」で設定してください"
            )
    if settings.get("song_service", {}).get("enabled", False):
        from agent import config as agent_config
        svc = agent_config.get_song_service_dir()
        if not (svc / "venv" / "Scripts" / "python.exe").exists():
            notices["song_service.enabled"] = (
                "作曲機能はまだ準備できていません。"
                "miku\\src\\song_service\\setup_song_service.ps1 を実行してください"
                "（約15GBのダウンロード・40〜90分）。"
                "実際にミクの歌声にするには VOCALOID6＋初音ミク V6（別売）も必要です"
            )
    return notices


def google_key_missing(settings: dict[str, Any]) -> bool:
    """必須の Google APIキーが未設定か（未設定だと会話自体が成立しない）。"""
    return not _key_set(settings, "google")


def apply_startup_downgrades(settings: dict[str, Any], errors: dict[str, str]) -> list[str]:
    """起動時の違反に対し「該当機能のみ停止」を in-memory settings に適用する。

    settings は Gateway のグローバル設定 dict（ファイルには書き戻さない）。
    api_keys.fish_audio（声）の違反はダウングレード先が無いため停止対象外
    （エラー表示のみ・合成時にも明示エラーになる）。

    Returns:
        適用した停止内容の説明リスト（ログ表示用）。
    """
    applied: list[str] = []
    for path in errors:
        if path == "stt.provider":
            settings.setdefault("stt", {})["provider"] = "none"
            applied.append("stt.provider → none（音声認識を停止）")
        elif path == "web_search.enabled":
            settings.setdefault("web_search", {})["enabled"] = False
            applied.append("web_search.enabled → false")
        elif path == "email.enabled":
            settings.setdefault("email", {})["enabled"] = False
            applied.append("email.enabled → false")
        elif path == "song_service.enabled":
            settings.setdefault("song_service", {})["enabled"] = False
            applied.append("song_service.enabled → false")
    return applied
