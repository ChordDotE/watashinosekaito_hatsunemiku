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
conversation_log.py - 会話記録（this_conversation）の新形式を一手に扱う共有モジュール

## 形式（2026-07-31 確定・docs/plan/conversation_log_migration_plan.md）

保存形式（KV のテキスト）: **インデント付きの正式 JSON 配列**（人間が読める・目視デバッグ可能）
LLM 投入時: **read 側で 1行=1発話の JSONL ビューへコンパクト化**（整形ぶん約 -25%）
画面表示（履歴オーバーレイ）: 同じパーサで末尾 N 発話を抽出

発話レコードのスキーマ（null/空フィールドは保存段階から省略）:
  ユーザー発話    {"speaker": "user", "time", "text", ("file_names", "file_content_description")}
  ミクの発話      {"speaker": "miku", "time", "before_response_states": [{"inner_thoughts", ("tool"), ("tool_summary"), ("text")}...], "text"}
                  ※ states 内の "text" はツール選択ループの中間発話（発話済み）。最終発話はトップの "text"
                  ※ "tool_summary" は**そのツールが返した結果の要点**（2026-08-26 追加）。
                    ツール結果は本来ターンをまたぐと失われるため、要点だけを会話ログに残す
                    （設計: docs/plan/tool_result_carryover.md）。**生の結果ではない**ので
                    tool_result ではなく tool_summary。隣の "tool" が出所のラベルになる
  ミクの状況認識  {"speaker": "miku", "time", "situation"}          … mode2/3 の起点
  外部要因        {"time", "event"}                                  … mode1 無応答タイマー等（speaker なし）

- `speaker` があれば発話者。`event` 行は外部の出来事（"others" の第三者誤読と、
  過去に却下済みの "system" 表記を両方回避するため speaker を持たせない）
- ミク発話の `text` が無い場合は「発話なし」（自由行動の内部行動など）
- 将来のターンキャンセル（発話済み切り詰め）用に、miku 行への追加フィールド
  （中断メタ等）を許容する前提で読み手は未知フィールドを無視すること

## 旧形式との共存（既存データは変換しない）

旧形式（疑似JSON: `{"user": {...}, "assistant": {...}}` の連結）が残っている間は
書き手も旧形式で追記を続け、記憶整理でファイルが空になった時点から新形式が始まる
（自己解決型移行）。`is_new_format()` で判別する。
"""

from __future__ import annotations

import json
from typing import Any

# LLM ビューの前文（プロンプトの this_conversation セクション見出しの直後に置く説明）。
# 解釈テスト（test/context_format_ab・19/20 同点）で使った文面と同一。
LLM_VIEW_PREAMBLE = (
    "- 1行=1レコードのJSON Linesです。speaker があればその人の発話。"
    "text=発話内容、situation=そのとき本人が認識した状況\n"
    "- speaker の無い event 行は外部要因の出来事（無応答タイマー等）\n"
    "- before_response_states は発話前の内心（inner_thoughts）と使ったツール（tool）の記録。"
    "tool_summary はそのツールが返した結果の要点で、**自分で見聞きしたことではなく道具が返したもの**\n"
    # ★2026-10-10: 記録に並ぶ自分の返事・内心が「話し方の手本」として効き、長い日ほど大げさになっていた
    #   （test/context_ablation/ の再現実験・decisions.md 2026-10-10）。中身は使い、言い回しと強さは写さない
    "- ミクの text と inner_thoughts は、そのとき何を話し・考えたかの記録。言い回しや気持ちの強さはそのときの場面のもので、"
    "いまの返事では写さない（いまの話し方と強さは、いまの場面で決める）\n"
)

# 会話ログを「後から読む」サブエージェント（記憶更新・知識登録など）へ渡すときの説明。
# 本文プロンプトの LLM_VIEW_PREAMBLE と違い、これらのエージェントは会話の場に居合わせて
# おらず、レコードの各フィールドが何を指すか・〔 〕や camera_N.jpg が何かを知らないまま
# 生テキストを渡されていた（2026-08-09 修正）。渡す側が増えても説明がぶれないよう
# ここに一元化する。
LOG_READING_GUIDE = (
    "## 会話ログの読み方\n"
    "会話ログは1レコード=1発話のJSONで、時系列順に並んでいる。\n"
    "複数の会話区切りぶんがまとめて渡される場合、区切りごとのJSON配列が続けて並ぶ。\n"
    "\n"
    "- `speaker` — 発話者。`user`=マスター、`miku`=あなた自身\n"
    "- `time` — その発話の時刻\n"
    "- `text` — 発話内容。ミクの行には無いこともあり、その場合は「発話せずに行動した」ことを表す\n"
    "- `situation` — ミク自身がそのとき認識した状況（自発的に話しかけた・自由行動した起点）\n"
    "- `before_response_states` — 発話前の内心（`inner_thoughts`）と使ったツール（`tool`）の記録。"
    "内心はあなた自身が実際に考えたことなので、記憶の材料として扱ってよい\n"
    "- 古い記録には旧いツール名（`memory_search` / `recall_long_memory`）が出てくることがある。"
    "それぞれ `recall_detail_conversation` / `recall_summary_experience` と同じもの（2026-08-27 改名）\n"
    "- `tool_summary` — `before_response_states` の中にある、**そのツールが返した結果の要点**。"
    "**あなたが自分で見聞きしたことではなく、道具が返してきたもの**である。"
    "隣の `tool` がその出所（どのツールか）を示すので、記憶として残すときも"
    "「（そのツール）で調べた／思い出した」という形を保つこと。"
    "要点であって結果の全文ではない（全文は残っていない）\n"
    "- `event` — `speaker` の無い行。外部要因の出来事（無応答タイマー等）で、誰かの発話ではない\n"
    "- `file_names` / `file_content_description` — その発話に添付されたファイルと、"
    "そのときのあなたが見て書いた内容説明。画像そのものはここには残っていないため、この説明文が唯一の手がかりになる\n"
    "\n"
    "### 音声・カメラ由来の情報\n"
    "- `text` が〔 〕で囲まれている場合、それはマイク音声の文字起こしである。"
    "誤認識や同音異義の聞き間違いを含みうるため、文脈が通らない箇所は似た音の言葉ではないかを考えること\n"
    "- `file_names` の `camera_N.jpg` はマスターが添付した写真ではなく、"
    "会話中のカメラに映っていた光景を自動で取り込んだものである。"
    "「マスターが見せてきた」ではなく「そのときそこに見えていた」として扱うこと\n"
    "\n"
    "### 声・顔の一致度が付いた発話\n"
    "`text` が〔音声入力（文字起こし）〕〔カメラ入力〕で始まる行は、発話が届いた時点で自動計測した"
    "「マスター本人らしさ」の記録である。**判定済みの結論ではなく測定値**なので、そのつもりで読む。\n"
    "- `[0.0-4.8s 声0.91] …` — 発話のまとまりごとの一致度（-1.0〜1.0）と、その区間の長さ。"
    "3秒以上なら本人は 0.85 前後・別人は 0.45 以下だが、1秒程度だと本人でも 0.60 前後まで下がる。"
    "**長さを見ずに数値だけで人を決めない**\n"
    "- `声0.60*` は前後の発話と合わせて測った値、`声--` は短くて測れなかったもの\n"
    "- `[camera_0.jpg 顔0.72] (左,上)-(右,下)` — 画像内の顔ごとの一致度と位置。"
    "本人はおおむね 0.65 以上・別人は 0.4 以下。ポスターや画面の顔も検出される\n"
)


def is_new_format(text: str) -> bool:
    """テキストが新形式（JSON 配列）か判定する。空文字は新形式で開始できるので True。"""
    stripped = (text or "").strip()
    if not stripped:
        return True
    if not stripped.startswith("["):
        return False
    try:
        return isinstance(json.loads(stripped), list)
    except json.JSONDecodeError:
        return False


def parse(text: str) -> list[dict[str, Any]]:
    """新形式テキストを発話レコードのリストにパースする。空なら []。

    旧形式を渡された場合は ValueError（呼び出し側は is_new_format で分岐すること）。
    """
    stripped = (text or "").strip()
    if not stripped:
        return []
    data = json.loads(stripped)
    if not isinstance(data, list):
        raise ValueError("this_conversation が JSON 配列ではありません")
    return data


def _clean(record: dict[str, Any]) -> dict[str, Any]:
    """null・空文字・空リスト・文字列 'null' のフィールドを省いたレコードを返す。"""
    out: dict[str, Any] = {}
    for k, v in record.items():
        if v in (None, "", [], "null"):
            continue
        out[k] = v
    return out


def build_turn_records(
    *,
    mode: int,
    trigger_text: str,
    user_input_time: str,
    assistant_output_time: str,
    final_response: str | None,
    response_objects: list[dict[str, Any]],
    file_names: list[str] | None = None,
    file_content_description: str | None = None,
    additions: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """1ターンぶんの発話レコード（起点行 + ミク応答行）を構築する。

    mode: 0=ユーザー発話 / 1=無応答タイマー（event 行）/ 2,3=ミク自身の起点（situation 行）
    additions: 「考え中の追加発話」（畳み込み・P2）。(到着時刻, テキスト) のリスト。
        起点行の直後に独立した user 行として時系列順に並べる。
    """
    records: list[dict[str, Any]] = []

    if mode == 1:
        records.append(_clean({"time": user_input_time, "event": trigger_text}))
    elif mode in (2, 3):
        records.append(_clean({
            "speaker": "miku", "time": user_input_time, "situation": trigger_text,
        }))
    else:
        records.append(_clean({
            "speaker": "user",
            "time": user_input_time,
            "text": trigger_text,
            "file_names": file_names,
            "file_content_description": file_content_description,
        }))

    # 考え中の追加発話（畳み込み）: 起点の後に独立した user 行として時系列で並べる
    for arrival_time, text in (additions or []):
        records.append(_clean({
            "speaker": "user", "time": arrival_time, "text": text,
        }))

    states = []
    for s in response_objects:
        states.append(_clean({
            "inner_thoughts": s.get("inner_thoughts", ""),
            "tool": s.get("tool_name"),
            # ツールが返した結果の要点。ターンをまたいで残すために持つ（2026-08-26）。
            # 生の結果ではないので tool_result ではなく tool_summary。null は省略される
            "tool_summary": s.get("tool_summary"),
            # ツール選択ループの中間発話（発話済み）。null は省略される
            "text": s.get("response"),
        }))
    records.append(_clean({
        "speaker": "miku",
        "time": assistant_output_time,
        "before_response_states": states,
        "text": final_response,
    }))
    return records


def append_records(existing_text: str, records: list[dict[str, Any]]) -> str:
    """新形式テキストにレコードを追記し、保存用（インデント付き JSON 配列）を返す。"""
    data = parse(existing_text)
    data.extend(records)
    return json.dumps(data, ensure_ascii=False, indent=2)


def to_llm_view(text: str) -> str:
    """LLM 投入用ビューを返す。

    新形式 → 1行=1発話のコンパクト JSONL（整形ぶん約 -25%）
    旧形式・不正 → そのまま返す（既存データは変換しない方針）
    """
    if not is_new_format(text):
        return text
    records = parse(text)
    if not records:
        return ""
    return "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records)


def tail_utterances(text: str, n: int = 5, speech_only: bool = False) -> list[dict[str, Any]]:
    """末尾 N 発話レコードを返す（履歴オーバーレイ用）。旧形式は空リスト。

    speech_only=True なら「speaker と text を持つ発話行」だけに絞る
    （situation / event / 発話なし行は画面表示に出さない）。
    ミクの行は**ツール前に話した言葉**（before_response_states[].text）も、話した順に1行ずつ出す
    （2026-10-09）。最終の text だけを見ていたので、曲を流すターン（最終が空）はミクの発言が丸ごと
    消え、deep_think の回は「待っててね」が出なかった。最終と同じ文の中間発話は重ねない。
    """
    if not is_new_format(text):
        return []
    records = parse(text)
    if speech_only:
        rows: list[dict[str, Any]] = []
        for r in records:
            if not r.get("speaker"):
                continue
            final = r.get("text")
            if r.get("speaker") == "miku":
                for s in r.get("before_response_states") or []:
                    said = s.get("text") if isinstance(s, dict) else None
                    if said and said != final:
                        rows.append({"speaker": "miku", "time": r.get("time"), "text": said})
            if final:
                rows.append(r)
        records = rows
    return records[-n:] if n > 0 else []
