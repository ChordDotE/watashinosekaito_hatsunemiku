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

"""知覚層の整形 — 声・顔の同定結果を LLM へ渡すテキストにする。

## なぜ「前処理」なのか

同定（マスター本人かどうか）は毎ターン必要で、**呼び忘れが許されない**。
ツールにすると LLM が呼ばないことがあり、ゲートとしても働かない。
だから VAD・Smart Turn・〔 〕と同じ層に置く（設計原則は roadmap #22）。
深い分析（オンデマンドの調べ物）は従来どおりツール。

## なぜ整形をここに一本化するのか

照合の実行タイミングは声（STT で ASR と並行）と顔（agent_main で context 取得と並行）で
分かれる。これはレイテンシの制約による物理的な都合で、**整形まで分けると
「LLM へどう見えるか」を決める場所が2箇所になる**。ここに集約して1箇所にする。

## 出力の形（2026-08-09 実測に基づき確定）

    〔音声入力（文字起こし）〕
    [0.0-4.8s 声0.91] 今度さあのみんなで集まる話あったじゃん
    [6.9-8.4s 声0.36] ダンデイ大学の
    〔カメラ入力〕
    [camera_0.jpg 顔0.72] (312,268)-(466,449)
    [camera_1.jpg 顔なし]

- 声と顔で**同じ行の構造**（`[どこ ラベル数値] 内容`）。「どこ」は声=時間区間・顔=ファイル名
- 区間・顔が1つでも複数でも同じ形（形が変わると読み手が混乱する）
- **判定はしない**。数値のまま渡す（顔サービスが is_master を返さないのと同じ方針。
  似た顔がいる場面でサービス側の二値判定が誤動作した経緯による）
- 数値の読み方（何秒ならいくつが本人か）は base_prompt に一度だけ置き、毎ターンは送らない
- 声の数値の後ろの `*` は「短くて単独では測れず、隣の発話と合わせて測った値」、
  `--` は「短く、かつ前後で値が食い違って測れなかった」
"""
from __future__ import annotations

from typing import Any

# ヘッダーに「文字起こし」と入れるのは、`〔 〕` と同じ**不確かさの表明**をここでも保つため。
# segments が付くと user_text の `〔 〕` は使われなくなる（テキストが区間側に含まれるので
# 二重に見せない）。「音声入力」だけだと聞こえた事実のように読めてしまう（2026-08-12）
VOICE_HEADER = "〔音声入力（文字起こし）〕"
CAMERA_HEADER = "〔カメラ入力〕"


def format_voice_segments(segments: list[dict[str, Any]]) -> str:
    """区間ごとの照合結果を音声入力ブロックにする。空なら空文字。"""
    rows = []
    if not segments:
        return ""
    t0 = float(segments[0].get("start", 0.0))
    for s in segments:
        # 改行・連続空白は潰す。本文がそのまま行を作れると、区間行や〔カメラ入力〕を
        # 本文側から偽造できてしまう（check_news._clean と同じ手）
        text = " ".join((s.get("text") or "").split())
        if not text:
            continue
        start = float(s.get("start", 0.0)) - t0
        end = float(s.get("end", 0.0)) - t0
        if s.get("scored", True):
            val = f"{float(s.get('score', 0.0)):.2f}" + ("*" if s.get("borrowed") else "")
        else:
            val = "--"
        rows.append(f"[{start:.1f}-{end:.1f}s 声{val}] {text}")
    return "\n".join([VOICE_HEADER] + rows) if rows else ""


def format_faces(faces: list[dict[str, Any]]) -> str:
    """画像ごとの顔照合結果をカメラ入力ブロックにする。空なら空文字。

    faces の各要素: {"file": str, "confidence": float|None, "box": {top,right,bottom,left}|None}
    confidence/box が None の要素は「顔なし」として1行出す。
    """
    rows = []
    for f in faces:
        name = f.get("file", "?")
        box = f.get("box")
        if not box:
            rows.append(f"[{name} 顔なし]")
            continue
        rows.append(
            f"[{name} 顔{float(f.get('confidence', 0.0)):.2f}] "
            f"({int(box['left'])},{int(box['top'])})-({int(box['right'])},{int(box['bottom'])})"
        )
    return "\n".join([CAMERA_HEADER] + rows) if rows else ""


def display_text(text: str) -> str:
    """`build_input_text` の逆向き——人に見せる用に、知覚情報の印を外した本文だけにする（2026-10-10）。

    マイク長押しの履歴に、区間（`[0.0-2.1s 声0.44]`）・見出し・カメラの照合結果まで出ていた（本人「出さなくていい」）。
    会話の記録そのものは変えない（ミクには知覚情報つきのまま見せる）。形を決めているこのファイルに逆向きも置く。
    """
    import re
    voice_row = re.compile(r"^\[\d+(?:\.\d+)?-\d+(?:\.\d+)?s 声[^\]]*\] ?(.*)$")
    parts: list[str] = []
    in_camera = False
    for line in (text or "").splitlines():
        if line == VOICE_HEADER:
            in_camera = False
            continue
        if line == CAMERA_HEADER:
            in_camera = True
            continue
        if in_camera:
            continue
        m = voice_row.match(line)
        parts.append(m.group(1) if m else line)
    out = " ".join(p for p in (s.strip() for s in parts) if p)
    if out.startswith("〔") and out.endswith("〕"):      # ブラウザが付ける文字起こしのマーク
        out = out[1:-1].strip()
    return out


def build_input_text(user_text: str,
                     voice_segments: list[dict[str, Any]] | None = None,
                     faces: list[dict[str, Any]] | None = None) -> str:
    """ユーザー入力を知覚情報つきのテキストにする。

    - 声の区間情報があれば、テキストは区間側に含まれるので user_text は使わない
      （二重に見せない）
    - 声が無い場合は user_text をそのまま使う（未登録・照合失敗・テキスト編集時。
      文字起こしマーク〔 〕はブラウザ側が付けている）
    - 顔があれば末尾に足す
    """
    blocks = []
    voice_block = format_voice_segments(voice_segments or [])
    blocks.append(voice_block if voice_block else user_text)
    face_block = format_faces(faces or [])
    if face_block:
        blocks.append(face_block)
    return "\n".join(b for b in blocks if b)
