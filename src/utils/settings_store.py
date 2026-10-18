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

"""settings.json の読込・マスク・更新適用・保存（設定GUIのバックエンドロジック）。

Flask に依存しない純粋関数のみ（gateway_service.py のルートから呼ばれる）。
秘匿値（type=secret）の扱い:
  - GET 向けビュー（build_view）では実値を返さず MASKED_VALUE に置き換える
  - 更新適用（apply_updates）で MASKED_VALUE が来たら「変更なし」として無視する

パスワード（type=password）の扱い:
  - 保存するのは **scrypt ハッシュ**（werkzeug.security。復元不可）
  - ビューでは設定の有無だけが分かる固定マスク（末尾も文字数も出さない）
  - 更新でマスク・空文字が来たら「変更なし」。実値が来たらハッシュ化して保存する
"""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

from utils.settings_schema import (
    FIELDS_BY_PATH,
    MASKED_VALUE,
    SECRET_PLACEHOLDER_PREFIX,
    SETTINGS_SECTIONS,
)


def load_settings(settings_path: Path) -> dict:
    """settings.json を読み込む。無ければ空 dict。壊れていれば ValueError。"""
    if not Path(settings_path).exists():
        return {}
    try:
        # utf-8-sig: BOM が付いていても読む（2026-09-25）。BOM 付きで保存するエディタで
        # 触られると、strict な utf-8 では「JSON が不正です」になり**システムごと起動しない**
        # ——しかも原因が BOM だとは分からない文言になる。書くときは BOM を付けない
        with open(str(settings_path), "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"settings.json の JSON が不正です: {e}") from e


def get_by_path(settings: dict, path: str) -> Any:
    """ドット区切りパスで値を取得する。途中のキーが無ければ None。"""
    node: Any = settings
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def set_by_path(settings: dict, path: str, value: Any) -> None:
    """ドット区切りパスで値を設定する。途中のキーが無ければ dict を作る。"""
    keys = path.split(".")
    node = settings
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def is_unset_secret(value: Any) -> bool:
    """秘匿値が「未設定」（空 or サンプルのプレースホルダのまま）かを判定する。"""
    if not isinstance(value, str) or value.strip() == "":
        return True
    return value.startswith(SECRET_PLACEHOLDER_PREFIX)


def mask_password(value: Any) -> str:
    """パスワードのマスク表示を作る。設定済みなら固定マスク、未設定なら空文字。

    secret と違い**末尾を残さない**（末尾も文字数も推測させない。ユーザー指示 2026-08-06）。
    保存されているのはハッシュなので、そもそも元の文字は復元できない。
    """
    if not isinstance(value, str) or value.strip() == "":
        return ""
    return MASKED_VALUE


def hash_password(raw: str) -> str:
    """パスワードを scrypt ハッシュ（ソルト自動生成・162文字固定）にする。

    werkzeug は Flask の依存として既に入っているため、追加インストールは不要。
    """
    from werkzeug.security import generate_password_hash
    return generate_password_hash(raw)


def verify_password(password_hash: str, raw: str) -> bool:
    """パスワードがハッシュと一致するか検証する。未設定・不正な形式なら False。"""
    if not password_hash or not raw:
        return False
    from werkzeug.security import check_password_hash
    try:
        return check_password_hash(password_hash, raw)
    except Exception:
        # 手編集で壊れたハッシュが入っていても例外で落とさない（ログインを拒否するだけ）
        return False


def mask_secret(value: Any) -> str:
    """秘匿値のマスク表示を作る。判別用に末尾3文字だけ残す（例: ********b12）。

    キーが8文字未満だと末尾3文字で大半が割れてしまうため全マスクにする。
    未設定なら空文字。
    """
    if is_unset_secret(value):
        return ""
    if len(value) < 8:
        return MASKED_VALUE
    return MASKED_VALUE + value[-3:]


def build_view(settings: dict) -> dict:
    """設定画面向けのビューを組み立てる。secret は実値を出さずマスクする。

    **`ui="none"` のフィールドは返さない**。画面に出さないと決めた項目
    （変えると壊れる・会話画面から操作できる・変える理由が無いもの）は、
    ここで落として画面側へ渡さない（描画側の実装ミスで露出しないようにするため）。
    settings.json を直接編集する経路は塞がない（apply_updates は全フィールドを受け付ける）。
    """
    sections = []
    for section in SETTINGS_SECTIONS:
        fields = []
        for field in section["fields"]:
            if field.get("ui") == "none":
                continue
            value = get_by_path(settings, field["path"])
            if field["type"] == "secret":
                value = mask_secret(value)
            elif field["type"] == "password":
                value = mask_password(value)
            entry = dict(field)
            entry["value"] = value
            fields.append(entry)
        if not fields:
            continue
        sections.append({
            "id": section["id"],
            "title": section["title"],
            "subtitle": section.get("subtitle", ""),
            # 既定で開くのは「ミクにできること」だけ。項目数が多く、全部開くと目的に辿り着けない
            "open": bool(section.get("open")),
            "fields": fields,
        })
    return {"sections": sections}


def _coerce(field: dict, value: Any) -> Any:
    """フィールド型に合わせて値を検証・変換する。不正なら ValueError。"""
    ftype = field["type"]
    if ftype in ("string", "secret"):
        if not isinstance(value, str):
            raise ValueError("文字列で入力してください")
        return value
    if ftype == "password":
        if not isinstance(value, str):
            raise ValueError("文字列で入力してください")
        # 空は「保護しない」を意味するのでそのまま通す（マスク・変更なしは呼び出し側で処理済み）
        if value == "":
            return ""
        if len(value) < 4:
            raise ValueError("4文字以上にしてください")
        return hash_password(value)
    if ftype == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError("ON/OFF の値が不正です")
    if ftype == "int":
        if isinstance(value, bool):
            raise ValueError("整数で入力してください")
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError("整数で入力してください")
    if ftype == "float":
        if isinstance(value, bool):
            raise ValueError("数値で入力してください")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError("数値で入力してください")
    if ftype == "select":
        if value not in field.get("choices", []):
            raise ValueError(f"選択肢にない値です: {value}")
        return value
    if ftype == "list_str":
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return [v.strip() for v in value if v.strip()]
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        raise ValueError("カンマ区切りの文字列で入力してください")
    raise ValueError(f"未知のフィールド型です: {ftype}")


def apply_updates(settings: dict, updates: dict) -> tuple[dict, dict]:
    """更新値を検証して settings のコピーへ適用する。

    Args:
        settings: 現在の settings.json の内容
        updates: {ドット区切りパス: 新しい値}

    Returns:
        (新しい settings, エラー dict {パス: メッセージ})。
        エラーが1件でもあれば settings は変更前のコピーのまま返す（部分適用しない）。
    """
    new_settings = copy.deepcopy(settings)
    errors: dict[str, str] = {}
    applied = copy.deepcopy(settings)

    for path, value in updates.items():
        field = FIELDS_BY_PATH.get(path)
        if field is None:
            errors[path] = "未知の設定キーです"
            continue
        if field["type"] == "secret":
            # マスク表示のまま送り返された場合は「変更なし」として無視する
            current_mask = mask_secret(get_by_path(applied, path))
            if value == MASKED_VALUE or (value and value == current_mask):
                continue
        if field["type"] == "password":
            # マスクのまま／空欄のままなら「変更なし」（既存のパスワードを消さない）。
            # 明示的に消したい場合は settings.json を直接編集する
            if value == MASKED_VALUE or value == "":
                continue
        try:
            set_by_path(applied, path, _coerce(field, value))
        except ValueError as e:
            errors[path] = str(e)

    if errors:
        return new_settings, errors
    return applied, {}


def save_settings(settings: dict, settings_path: Path) -> None:
    """settings.json を保存する。既存があれば .bak を残し、一時ファイル経由で置き換える。"""
    settings_path = Path(settings_path)
    if settings_path.exists():
        shutil.copy2(str(settings_path), str(settings_path) + ".bak")
    tmp_path = str(settings_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, str(settings_path))
