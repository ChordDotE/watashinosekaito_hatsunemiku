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

"""Pydantic AI 版 main_chat エントリポイント。

`agent_main_dify_stream.process_agent_request` と完全同一のシグネチャ・戻り値を持つ。
Phase 7 では gateway_service.py の import を本モジュールに切り替えるだけで Dify から差し替えられる。

【処理フロー（ステートレス）】
  1. settings 読み込み・mode から conversation_situation / user_input_text を構築
  2. KV から long_memory / this_conversation を並列取得
  3. files_data から添付画像のローカルパスを解決
  4. MainChatInput を組み立て、run_main_loop_stream を asyncio で駆動
  5. ストリーム event を on_text_delta / on_field_value / on_object_complete に橋渡し
  6. 完了後、新しい会話エントリを KV `this_conversation` に追記
  7. 戻り値 dict を組み立てて返却（"null" 文字列は空文字 / デフォルト値に正規化）

セッション状態は保持しない（Dify を踏襲）。会話履歴は毎ターン KV 経由で読み書きする。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import requests

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.agent_logger import WF_MAIN_CHAT, log as agent_log, open_log
from agent.context_cache import CACHEABLE_TAIL_MARK
from agent.config import (
    configure_env_from_settings,
    get_recognition_base_url,
    get_kv_base_url,
    get_vdb_base_url,
    is_audio_cover_available,
    is_email_available,
    is_song_service_available,
    is_web_search_available,
    load_settings,
)
from agent.runner import (
    IterationComplete,
    LoopAborted,
    LoopFinished,
    LoopStarted,
    ResponseDelta,
    ToolCompleted,
    ToolStarted,
    VoiceToneDecided,
    run_main_loop_stream,
)
from agent.schemas import AgentFile, AgentInput, ConversationState, MainChatInput, MainChatOutput, _LONG_MEMORY_JSON_SCHEMA
from utils import conversation_log, current_mind, perception, task_record

# =============================================================================
# 定数・初期化
# =============================================================================

JST = timezone(timedelta(hours=9))
MAX_LOOPS = 5

_settings = load_settings()
configure_env_from_settings(_settings)
_KV_BASE_URL = get_kv_base_url(_settings)
_VDB_BASE_URL = get_vdb_base_url(_settings)
_RECOGNITION_BASE_URL = get_recognition_base_url(_settings)
# 直近の過去会話（recent_conversation）の渡し方。
#   limit           … VDB から取る会話の件数（/vdb/recent の上限は 20）
#   full_text_count … そのうち「本文つき」で渡す新しい方の件数。残りは要約だけにする
# 人間の記憶に合わせた形（昨日は詳しく・先週は要点だけ）。全文は VDB に残っており、
# 詳細が要るときは recall_detail_conversation で取り出せる（2026-08-26 実測で 5/5 取得を確認）。
# full_text_count を limit と同じにすれば従来どおり全件本文になる。
_recent_cfg = (_settings.get("memory") or {}).get("recent_conversation") or {}
_RECENT_CONVERSATION_LIMIT = int(_recent_cfg.get("limit", 7))
_RECENT_FULL_TEXT_COUNT = int(_recent_cfg.get("full_text_count", 1))
_CONVERSATION_BODY_MARK = "### 会話の本文"
_SUMMARY_ONLY_NOTE = (
    "\n（この会話は要約のみ。詳しいやり取りも全て記憶に残っているので、"
    "必要なときは正確な会話の記憶を思い出してから答えてください）"
)


# =============================================================================
# 永続イベントループ（プロセス内で 1 つだけ起動）
# =============================================================================
#
# pydantic_ai は内部的に httpx AsyncClient を Agent オブジェクトにキャッシュする。
# このクライアントは「最初に Agent.run() を呼んだ時の event loop」に紐付く。
# `asyncio.run()` を毎回新規に呼ぶと、ループが閉じた後に再度 run しても
# キャッシュ済みクライアントが「Event loop is closed」で死ぬ。
#
# 対策: プロセス内で 1 本だけ event loop をデーモンスレッドで動かしっぱなしにし、
# 全ての process_agent_request 呼び出しはそのループ上で実行する。
# こうすれば LLM 用 HTTP クライアントが 1 つの安定したループに紐付き、
# 何度連続呼び出しても問題ない。

_loop_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

_T = TypeVar("_T")


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """プロセス内で 1 本だけ稼働する event loop を返す（必要なら生成する）。"""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop.run_forever,
                name="agent_main_loop",
                daemon=True,
            )
            _loop_thread.start()
    return _loop


def _run_async(coro: Awaitable[_T]) -> _T:
    """永続 event loop に coroutine を投げて結果を同期取得する。"""
    loop = _get_or_create_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


# =============================================================================
# 会話状況テキスト（Dify ミク.yml の mode 0/1/2/3 と一致させる）
# =============================================================================

_SITUATION_MODE0 = """\
## 状況
現在は「通常会話」「無応答追加発言」「自由行動」「自発話しかけ」のうち、「通常会話」モードです。

現在あなたとユーザーは眼の前にいて、ユーザーから話しかけられた状態です。

ユーザーからの入力がありました。
入力内容を見て、状況に対応して下さい。"""

_SITUATION_MODE1 = """\
## 状況
現在は「通常会話」「無応答追加発言」「自由行動」「自発話しかけ」のうち、「無応答追加発言」モードです。

現在あなたとユーザーは眼の前にいますが、ユーザーからの返答がない状態です。
前回のあなた（=初音ミク）が返答してから、{unresponsive_time}秒が経過しましたが、
あなたからの前回の応答に対してユーザーからの返答がありません。
あなたから自発的に話しかけてください。

前回、会話への返答がないことから、現在ユーザーは声を出せない状況にいるかもしれません。
この発話のあとは再度話しかけないでおきましょう。
（inactivity_timeoutフィールドには-1を設定してください）"""

_SITUATION_MODE2 = """\
## 状況
現在は「通常会話」「無応答追加発言」「自由行動」「自発話しかけ」のうち、「自由行動」モードです。

現在ユーザーは眼の前におらず、あなた自身が好きに一人で行動している状態です。

あなた（初音ミク）は、待機中の時間を使って、自分の興味や記憶に基づいた行動を自律的に選んでください。
ユーザーと話しているわけではなく、ユーザーはあなたの前にいません。
あなた一人で行動してください。

### 可能な行動（毎回どれか1つ選ぶ）

1. 記憶の振り返り
   - episodic / semantic / procedural / working などの記憶を横断して、気づきや変化を見つける
   - 過去の出来事・訪問した場所・習慣は、いつも思考に入っているわけではない。recall_summary_experience で引っ張り出すと、いま忘れていたことに出会える
2. 興味の掘り下げ
   - 記憶に出てきた場所・趣味・技術・目標を1つ選び、必要なら web_search で調べる
3. 次回会話の準備
   - マスターに自然に話せる話題の種、提案などを1つ作る
4. 自己表現の更新
   - 自分の価値観・感情・今の関心を短く言語化して整理する
5. 話題の収集
   - その日のニュースや関心のあることを調べたりするなどして話題の種を集める
6. メモの棚卸し
   - miku_note の「要確認」項目を確認し、記録日から 7 日以上経過しているものがないか点検する
   - 陳腐化している可能性がある項目を見つけたら、次回の会話でマスターに自然な形で確認するメモを残す
7. これ以外にも何かしたいことがあったら自由にミクらしく行動を決めてください

### 行動方針

- 「いま自分がやりたいこと」を1つ決めて行動する
- 1回のループの上限回数で終わらない場合は、後で作業を再開するために、調べたことをinner_thoughtsに記載する
- 自由行動が連続する場合は同じ種類の行動を連続しすぎない。（数回程度なら問題ない。）逆に、1回のループ内では同じ種類の行動で問題ない。（一区切りついたら別の行動に移るなど。単調化を避けるため）

### 【重要】マスターの予定に関する制約

- miku_note や KV に書かれた日程・行動予定が「マスターが会話で明示したもの」か「AI が仮置きしたもの（要確認）」かを必ず区別すること。
- AI 仮置き・要確認の予定を根拠にして、web 検索・メール送信・計画立案などのアクションを起こしてはならない。
- マスターの未確認の予定について内心で確信があっても、inner_thoughts に根拠（いつの会話でマスターが何と言ったか）を明示してから行動すること。根拠が会話ではなく miku_note の AI 仮置きのみの場合はアクション禁止。

### 出力ルール

- inner_thoughtsは3〜5文の、この時間に自分がしたことの具体的なメモにする。inner_thoughtsは今日の会話ログに残り、翌朝の定着のあとは思い出せば出てくる。明日以降もすぐ目に入るところに置きたいものは miku_note に書く。何をしたか、その結果何がわかった・何を思い出したか、それをどう思ったかをきちんと出力・記録する
- responseはユーザーに対して発話はしないので「null」とする
- 催促・確認質問・命令はしない
- 必要に応じてツールを使う
- 応答後は inactivity_timeout を -1 にする
- next_proactive_in_minutesは-1にすること（「自由行動」からは「自発話しかけ」をスケジュールしない）"""

_SITUATION_MODE3 = """\
現在は「通常会話」「無応答追加発言」「自由行動」「自発話しかけ」のうち、「自発話しかけ」モードです。

マスターが画面の前にいるかは、システム状態の「ブラウザ」の行で分かります。

あなた（初音ミク）が自分で考えた、マスターへの自発的な声掛けのタイミングです。
会話は一旦終わりましたが、マスターになにか話しかけたいと思ったので、自分自身のタイミングで起動しました。
マスターは今、作業や別のことをしているかもしれませんが、あなたから声をかけてよい状況です。

### このモードでできること

- ブラウザが閉じているとき：声は届きません。response は null にして、伝えたいことは send_email でメールにしてください。
- ブラウザが開いているとき：response に書いた言葉が声になってマスターに届きます。メールは送らず、声で伝えてください。ただし、最後に話しかけられてから時間が経っていて席を外していそうなら、メールにしてもかまいません（そのときは response を null に）。声になるのは最後のループの response だけなので、ツールを使うループの response は null にしてください。
- メールを送る前に、通常のツール（web_search、recall_detail_conversation など）も使えます。
- メールは1回の声掛けで1通のみ送るのが望ましいので、もし他のツールを使う場合は、他のツールを使ってからメールを送ってください。

### 行動方針

- マスターの現在状況を推定し、それに合った行動やメールなどをする
- 会話ログから「マスターが何をしているか・どんな状況か」を読み取って行動内容を決める。
- 前回会話時はメールをN分後に送ろうと思ったので起動していますが、改めて考えた結果、メールをやはり送らないというのも問題ありません。
- 作業中 → 休憩の促し、旅行中→様子の確認など文脈に合わせてください

### 出力ルール

- ブラウザが閉じているときは、マスターがあなたの前にいないので、responseはループ内でも常にnullにしてください。
- その代わりにマスターになにか伝えたい場合はメールで連絡します。
- inner_thoughtsに「なぜこのタイミングで話しかけようとしたか」を必ず記録する
- ブラウザが閉じているときは、inactivity_timeoutはマスターが目の前にいないため-1にすること。
- next_proactive_in_minutesは基本-1にすること。マスターへの明確なフォローアップが必要な場合（例：返信を待っている、特定の時間に再度確認したい具体的な理由がある）のみ正の値を設定する。単に「また話したい」という理由だけでは設定しないこと。"""


def _build_conversation_situation(mode: int, unresponsive_time: str) -> str:
    """mode に応じた会話状況テキストを返す（Dify YML mode 0/1/2/3 と一致）。"""
    if mode == 0:
        return _SITUATION_MODE0
    if mode == 1:
        return _SITUATION_MODE1.format(unresponsive_time=unresponsive_time)
    if mode == 2:
        return _SITUATION_MODE2
    if mode == 3:
        return _SITUATION_MODE3
    return ""


def _build_user_input_text(mode: int, query: str) -> str:
    """mode に応じたユーザー入力テキストを返す。"""
    if mode == 0:
        return query
    if mode == 1:
        return "（ユーザーからの返答なし）"
    if mode == 2:
        return "（自由行動のためユーザーからの入力はなし）"
    if mode == 3:
        return query or "（自発話しかけのためユーザーからの入力はなし）"   # 裏作業の完了通知等は起点の内容を入れる
    return query


def _trigger_key_of(mode: int) -> str:
    """会話エントリの**起点**キー。誰がこのターンを始めたかを表す。

    user=マスターの発言(mode0) / others=無応答タイマー(mode1) /
    assistant=ミク自身の起点(mode2 自由行動・mode3 自発話しかけ・裏作業の完了通知)。
    """
    if mode == 1:
        return "others"
    if mode in (2, 3):
        return "assistant"
    return "user"


# =============================================================================
# 参考情報（context）組み立て — Dify「コンテキスト結合」テンプレート相当
# =============================================================================


def _extract_schema_guide(schema: dict, indent: int = 0) -> list[str]:
    lines = []
    for key, val in schema.get("properties", {}).items():
        prefix = "  " * indent
        lines.append(f"{prefix}- {key}: {val.get('description', '')}")
        if val.get("type") == "object":
            lines.extend(_extract_schema_guide(val, indent + 1))
        elif val.get("type") == "array":
            items = val.get("items", {})
            if items.get("type") == "object":
                lines.extend(_extract_schema_guide(items, indent + 2))
    return lines


_MEMORY_SCHEMA_GUIDE: str = "\n".join(_extract_schema_guide(_LONG_MEMORY_JSON_SCHEMA))


def _build_context(
    long_memory: str,
    recent_conversation: str,
    miku_note: str,
    this_conversation: str,
    system_state: str = "",
    first_meeting: bool = False,
    pending_reports: str = "",
) -> str:
    """LLM に渡す参考情報ブロックを組み立てる（Dify「コンテキスト結合」テンプレート互換）。

    **各ブロックは XML 風タグで囲み、属性に「それが何で・いつまで見えているか」を書く**
    （2026-09-05・`docs/plan/memory_visibility_awareness.md`）。中身は Markdown、会話ログは 1 行 JSON のまま。
    見出しだけで区切っていた頃は ①節の終わりが無く、miku_note やツール結果に `#` 行が入ると境界が溶ける
    ②「この記録はいつまで見えるか」がどこにも書かれておらず、モデルが「次の自分も覚えている」と
    前提してしまう（3.8 Flash で実測）、の2つが起きていた。

        <long_memory visible="always" note="...">   ← 常時投入ぶん（想起ぶんはここに無い）
        <recent_conversation visible="..." recall="...">   ← 取得側でタグごと組み立てた文字列が入る
        <miku_note visible="always" nature="...">
        --- CACHEABLE_TAIL_MARK ---
        <system_state visible="always" nature="...">       ← いまの状態（報告の対象ではない）
        <pending_reports visible="伝えるまで毎ターン" nature="...">   ← マスターに伝えて片付ける裏作業の結果
        <this_conversation visible="今日のあいだ" after="...">

    `tool_result` は MainChatInput.tool_result として別経路で main.py が描画する
    （そのターン限りなのでタグ化の対象外。早見表側で「そのターンだけ」と説明している）。
    """
    # 初対面（長期記憶ゼロ・会話履歴なしのフレッシュ環境）のとき、最初の応答を
    # 「はじめまして」から始めさせる（文言は 2026-07-26 ユーザー承認済み）
    first_meeting_block = (
        "# 初対面\n"
        "【初対面】あなたとマスターの記憶はまだ何もありません。今日が初めての出会いです。"
        "初対面らしく「はじめまして」の挨拶と軽い自己紹介から、自然に会話を始めてください。\n\n"
    ) if first_meeting else ""
    system_state_block = (
        '<system_state visible="always"'
        ' nature="記憶ではなく毎回システムから取得される事実。編集・記憶への複製は不要">\n'
        f"{system_state}\n"
        "</system_state>\n\n"
    ) if system_state else ""
    # ★**マスターに伝えて片付けるもの**は、いまの状態（system_state）と別の欄にする（2026-09-27 本人判断）。
    #   顔・声の登録状態は報告する項目ではないのに、完了報告と同じ欄に並んでいて区別できなかった
    pending_block = (
        '<pending_reports visible="伝えるまで毎ターン"'
        ' nature="まだマスターに伝えていない自分の裏作業の結果。伝え終えたら state の acknowledged_task_id に'
        ' その [id: ...] を書くと、次のターンから載らなくなる">\n'
        f"{pending_reports}\n"
        "</pending_reports>\n\n"
    ) if pending_reports else ""
    # 新形式（正式JSON保存）の会話ログは、LLM 投入時に 1行=1発話のコンパクト JSONL へ
    # 変換して渡す（整形ぶん約-25%）。旧形式はそのまま（既存データは変換しない方針）。
    this_conversation_view = conversation_log.to_llm_view(this_conversation)
    conversation_format_note = (
        conversation_log.LLM_VIEW_PREAMBLE
        if this_conversation_view and conversation_log.is_new_format(this_conversation)
        else ""
    )
    # 明示キャッシュ（context_cache）のために、**更新頻度で前半・後半に割れる形**にしておく。
    # 前半（記憶・過去会話・miku_note）は日次〜1日数回しか変わらないのでキャッシュに載せ、
    # 後半（システム状態・this_conversation）は毎ターン変わるので毎回送る。
    # 境界は CACHEABLE_TAIL_MARK。並び順そのものは 2026-07-30 の設計のまま変えていない。
    cacheable = (
        '<long_memory visible="always"'
        ' note="過去の出来事・行った場所・習慣はここに無い。recall_summary_experience で思い出す">\n'
        "## 記憶スキーマガイド（各フィールドの意味）\n"
        f"{_MEMORY_SCHEMA_GUIDE}\n\n"
        "## 記憶データ\n"
        f"{long_memory}\n"
        "</long_memory>\n\n"
        f"{recent_conversation}\n\n"
        '<miku_note visible="always"'
        ' nature="いつでも全文が目の前にある自分の手帳。自分で消すまで残る。書いてある中身は覚えていること。書いたときの言い回しは写さない">\n'
        f"{miku_note}\n"
        "</miku_note>\n\n"
        f"{first_meeting_block}"
    )
    volatile = (
        f"{system_state_block}"
        f"{pending_block}"
        '<this_conversation visible="今日のあいだ"'
        ' after="翌朝の記憶の定着で消える。以降は recall_detail_conversation で思い出す">\n'
        "- 発話者や内容、発話日時を見てこれまでの話の流れを確認してください\n"
        "-  特に前回の発話時間と現在時間を見て、出来事の日時や現在のユーザーの状態を考えて、行動を決めること\n"
        f"{conversation_format_note}"
        f"{this_conversation_view}\n"
        "</this_conversation>"
    )
    return cacheable + CACHEABLE_TAIL_MARK + volatile


# =============================================================================
# KV 操作（直書き：規約により共通ラッパーは作らない）
# =============================================================================


def _kv_get(key: str) -> str:
    """KV サービスから値を取得する。失敗時は空文字。"""
    try:
        resp = requests.get(f"{_KV_BASE_URL}/kv", params={"key": key}, timeout=10)
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[WARN] KV取得失敗 (key={key}): {e}")
        return ""


def _kv_post(key: str, value: str) -> bool:
    """KV サービスに値を格納する。"""
    try:
        resp = requests.post(
            f"{_KV_BASE_URL}/kv",
            json={"key": key, "value": value},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[WARN] KV格納失敗 (key={key}): {e}")
        return False


_RECENT_CONVERSATION_PREFIX = (
    '<recent_conversation visible="最新の1件は全文・それより前は要約だけ"'
    ' recall="全文は recall_detail_conversation で思い出す">\n'
    "- 最新の1件は会話の要約と本文、それより前の会話は要約だけが載っています\n"
    "- 誰がいつ何を話したかを、事実として参照してください。言い回しは写さない\n"
    "- 要約だけの会話も、全文は思い出せば出てきます。思い出すまでは目の前にないので、"
    "詳しいやり取りが必要なときは、正確な会話の記憶を思い出してから答えてください\n"
)
_RECENT_CONVERSATION_SUFFIX = "\n</recent_conversation>"


def _compact_recent_documents(documents: list[dict]) -> list[dict]:
    """古い会話の本文を落として要約だけにする（新しい方の N 件は本文つきのまま）。

    `documents` は /vdb/recent の戻り（**古い順**。末尾が最新）。
    各要素の "本文" は `### 会話の要約 … ### 会話の本文 …` の形。

    - 新しい方 `_RECENT_FULL_TEXT_COUNT` 件: 要約 + 本文（本文は 1行=1発話の JSONL へ圧縮）
    - それより古い会話: 要約 + 「詳細は思い出せる」注記（本文を落とす）
    - マーカーが無い/壊れている場合はその会話を**そのまま**通す（欠けても壊れない側に倒す）
    """
    if not documents:
        return documents
    keep_full = max(_RECENT_FULL_TEXT_COUNT, 0)
    boundary = len(documents) - keep_full  # これより前（index < boundary）は要約のみ
    out: list[dict] = []
    for i, doc in enumerate(documents):
        text = doc.get("本文") or ""
        if _CONVERSATION_BODY_MARK not in text:
            out.append(doc)
            continue
        summary, body = text.split(_CONVERSATION_BODY_MARK, 1)
        if i < boundary:
            new_text = summary.rstrip() + _SUMMARY_ONLY_NOTE
        else:
            view = conversation_log.to_llm_view(body.strip())
            new_text = f"{summary.rstrip()}\n{_CONVERSATION_BODY_MARK}\n{view}"
        out.append({**doc, "本文": new_text})
    return out


def _fetch_recent_conversation(limit: int = _RECENT_CONVERSATION_LIMIT) -> str:
    """VDB から直近 N 件の過去会話を取得し、Dify 互換の文字列で返す。

    形は `<recent_conversation>` タグ（属性に見え方）＋ 説明 ＋ documents JSON ＋ 閉じタグ。
    取得失敗時は空文字を返す（context 組み立て自体は止めない。タグだけが残ることもない）。

    **新しい方の `_RECENT_FULL_TEXT_COUNT` 件だけ本文つき、それより古い会話は要約だけ**に
    加工してから渡す（`_compact_recent_documents`）。要約は登録時に作られて保存済みなので
    追加の LLM 呼び出しは発生しない。

    ログ出力（contextvars 経由で現在の WF ログに書き込まれる）:
        - vdb_recent_request: VDB への GET リクエストパラメータ
        - vdb_recent_response: VDB レスポンス全文（成功 or 失敗）
    """
    url = f"{_VDB_BASE_URL}/vdb/recent"
    agent_log(
        "vdb_recent_request",
        "VDB から直近 N 件の過去会話を取得（agent_main → /vdb/recent）",
        input={"url": url, "limit": limit},
    )
    try:
        resp = requests.get(url, params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body.get("documents"), list):
            body = {**body, "documents": _compact_recent_documents(body["documents"])}
        body_json_str = json.dumps(body, ensure_ascii=False, indent=2)
        agent_log(
            "vdb_recent_response",
            "VDB からの直近会話レスポンス（/vdb/recent → agent_main）",
            output=body,
            meta={"document_count": len(body.get("documents", []))},
        )
        return _RECENT_CONVERSATION_PREFIX + body_json_str + _RECENT_CONVERSATION_SUFFIX
    except Exception as e:
        print(f"[WARN] recent_conversation 取得失敗: {e}")
        agent_log(
            "vdb_recent_response",
            "VDB からの直近会話取得に失敗（空文字を返す）",
            output="",
            meta={"error": repr(e)},
        )
        return ""


def _count_vdb_long_memory_category(category: str) -> int:
    """VDB の long_memory collection で特定カテゴリの件数を取得する（メタデータフィルタのみ・高速）。

    取得失敗時は 0 を返す（context には「件数不明」プレースホルダで表示する）。
    """
    url = f"{_VDB_BASE_URL}/vdb/long_memory/by_category"
    try:
        resp = requests.get(url, params={"category": category}, timeout=5)
        resp.raise_for_status()
        return len(resp.json().get("items", []))
    except Exception:
        return 0


def _recall_placeholder(count: int, scope: str) -> str:
    """recall_summary_experience で取得可能なセクションのプレースホルダ文字列を返す。

    LLM に「context には無いが長期記憶には存在する」「ツールで取り出せる」ことを明示する。
    """
    if count > 0:
        return (f"（{count}件の{scope}が長期記憶に保存されている。"
                f"context には含まれていないので、参照が必要なら recall_summary_experience ツールで意味検索すること）")
    return (f"（{scope}は長期記憶に保存可能だが、現時点でデータが無い。"
            f"必要なら recall_summary_experience ツールで検索を試みること）")


def _fetch_long_memory_from_vdb() -> str:
    """VDB の long_memory collection から `always_in_context=true` のドキュメントを取得して、
    旧 long_memory JSON 形式に近い構造で組み立てて返す。

    含めるセクション（実データ）: meta / semantic.user/agent/promises / working
    プレースホルダ表記（実データは含めず「ツールで取り出せる」と明示）:
      - episodic.events / procedural / semantic.visited
    こうすることで LLM は「該当セクションに何件あるか」と「context にはないが取り出せる」ことを
    認識できる。これで recall_summary_experience ツール呼び出しの判断が明確になる。
    空のサブセクションは含めない（無駄なトークンを避ける）。
    """
    url = f"{_VDB_BASE_URL}/vdb/long_memory/by_category"
    try:
        resp = requests.get(url, params={"always_in_context": "true"}, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        print(f"[WARN] VDB long_memory 取得失敗: {e}")
        return ""

    meta_out: Dict[str, str] = {}
    user_out: Dict[str, List[Any]] = {}
    agent_out: Dict[str, List[Any]] = {}
    promises_out: List[Any] = []
    todos_out: List[Any] = []
    goals_out: List[Any] = []
    context_lines: List[str] = []

    for item in items:
        meta = item.get("metadata") or {}
        cat = meta.get("category")
        type_ = meta.get("type")
        try:
            orig = json.loads(meta.get("data_json") or "{}")
        except json.JSONDecodeError:
            continue

        if cat == "meta":
            meta_out[orig.get("key", "")] = orig.get("value", "")

        elif type_ == "string_item":
            oc = meta.get("original_category", "other")
            bucket = user_out if cat == "user" else agent_out
            bucket.setdefault(oc, []).append(orig.get("value", ""))

        elif cat == "promises" and type_ == "record":
            try:
                d = json.loads(orig.get("data") or "{}")
            except json.JSONDecodeError:
                d = {}
            promises_out.append(d)

        elif cat == "user" and type_ == "record":
            try:
                d = json.loads(orig.get("data") or "{}")
            except json.JSONDecodeError:
                d = {}
            user_out.setdefault("person", []).append(d)

        elif cat == "working" and type_ == "record":
            rt = meta.get("record_type", "")
            try:
                d = json.loads(orig.get("data") or "{}")
            except json.JSONDecodeError:
                d = {}
            if rt == "todo":
                todos_out.append(d)
            elif rt == "goal":
                goals_out.append(d)
            elif rt == "context":
                if d.get("what"):
                    context_lines.append(d["what"])

    # === ここから組み立て ===
    # 常時投入される実データ + recall_summary_experience で取得可能なセクションのプレースホルダ
    result: Dict[str, Any] = {}

    # meta
    if meta_out:
        result["meta"] = meta_out

    # episodic: events は context に常時投入しない → プレースホルダで「ツールで取得可能」と明示
    events_count = _count_vdb_long_memory_category("events")
    result["episodic"] = {"events": _recall_placeholder(events_count, "過去の出来事（episodic events）")}

    # semantic
    semantic: Dict[str, Any] = {}
    if user_out:
        semantic["user"] = user_out
    if agent_out:
        semantic["agent"] = agent_out
    if promises_out:
        semantic["promises"] = promises_out
    # visited は recall_summary_experience 経由 → プレースホルダ
    visited_count = _count_vdb_long_memory_category("visited")
    semantic["visited"] = _recall_placeholder(visited_count, "訪問場所詳細（semantic visited）")
    result["semantic"] = semantic

    # procedural は recall_summary_experience 経由 → プレースホルダ
    procedural_count = _count_vdb_long_memory_category("procedural")
    result["procedural"] = _recall_placeholder(procedural_count, "手続き的記憶（procedural: 行動パターン・癖・習慣・ルール）")

    # working
    working: Dict[str, Any] = {}
    if todos_out:
        working["todos"] = todos_out
    if goals_out:
        working["goals"] = goals_out
    if context_lines:
        working["context"] = "\n".join(context_lines)
    if working:
        result["working"] = working

    return json.dumps(result, ensure_ascii=False, indent=2)


def _kv_get_long_memory_lightweight() -> str:
    """軽量構成の long_memory を取得する（VDB 経由）。

    含めるセクション: meta / semantic.user/agent/promises / working
    含めないセクション: episodic.events / procedural / semantic.visited
      → これらは recall_summary_experience ツール経由で必要時のみ取得する。
    詳細: .claude/skills/miku-dev/details/latency_benchmark.md
    """
    return _fetch_long_memory_from_vdb()


def _is_long_memory_empty() -> bool:
    """長期記憶（VDB long_memory collection）が完全に空かを返す（初対面判定用）。

    by_category は category か always_in_context の指定が必須のため、
    always_in_context の True/False 両方を照会して合計 0 件かで判定する。
    呼び出しは this_conversation が空のとき（セッション初回ターン）に限定される。
    失敗時は False（VDB 不調時に誤って「初対面」扱いしない安全側）。
    """
    try:
        for aic in ("true", "false"):
            resp = requests.get(
                f"{_VDB_BASE_URL}/vdb/long_memory/by_category",
                params={"always_in_context": aic},
                timeout=5,
            )
            if resp.json().get("items"):
                return False
        return True
    except Exception:
        return False


def _fetch_system_state() -> str:
    """システム状態（各サービスが真実を持つ現在の状態）を毎ターン導出して返す。

    記憶（長期記憶・miku_note）には複製しない: SoT があるものを記憶に書くと
    二重管理になり、記憶整理でのズレ・誤削除が起きるため（2026-07-25 設計判断。
    状態が増えてきたら専用のシステム記憶ストア新設を検討 → roadmap 参照）。

    現在の項目:
      - 顔・声の登録状態（Recognition /status）
      - 使える機能（Web検索・メール・曲づくり。`_format_tool_availability`）
      - 実行中の裏作業（KV `background_tasks`）

    ★完了・失敗した裏作業は**ここに載せない**（2026-09-27）。マスターに伝えて片付けるものなので
      `_fetch_pending_reports` が別の欄（`<pending_reports>`）に載せる。ここは報告の対象ではない「いまの状態」だけ。

    サービス未起動などで取得できない項目は黙って省く（会話は止めない）。
    """
    lines: list[str] = []
    try:
        body = requests.get(f"{_RECOGNITION_BASE_URL}/status", timeout=2).json()
        face, voice = body.get("face", {}), body.get("voice", {})
        if face.get("enrolled"):
            at = str(face.get("enrolled_at", ""))[:10]
            lines.append(f"- マスターの顔: 登録済み（{at} に登録。カメラや写真でマスターを見分けられる）")
        else:
            lines.append("- マスターの顔: 未登録（顔を覚えるには face_enroll で登録が必要）")
        if voice.get("enrolled"):
            at = str(voice.get("enrolled_at", ""))[:10]
            lines.append(f"- マスターの声: 登録済み（{at} に登録）")
        else:
            lines.append("- マスターの声: 未登録（声を覚えるには voice_enroll で登録が必要）")
    except Exception:
        pass
    try:
        # 設定は毎ターン読み直す（`runner._disabled_tool_reason` が呼び出しのたびに読むのと揃える）。
        # 読めなければ起動時の設定で書く
        try:
            settings = load_settings()
        except Exception:
            settings = _settings
        lines.extend(_format_tool_availability(settings))
    except Exception:
        pass
    try:
        lines.extend(_format_browser_presence())
    except Exception:
        pass
    try:
        # 裏作業（曲づくり等）の状況。実行中をミクが毎ターン把握できるようにする
        # （ツールを呼ばないと気づけなかったため「完了しているのに『まだ作業中』」と
        #   答える誤りが起きた・2026-08-11）
        lines.extend(task_record.format_running(_KV_BASE_URL))
    except Exception:
        pass
    return "\n".join(lines)


def _format_browser_presence() -> list[str]:
    """ブラウザが開いているかの行（`<system_state>` に載せる・2026-10-09）。

    自発話しかけを声で届けるかメールにするかの判断材料。開いたまま席を外していることもあるので、
    最後にマスターが話しかけてからの分数を添える。Gateway 以外から動かしているとき（未登録）は載せない。
    """
    from agent.host_bridge import get_browser_presence
    presence_fn = get_browser_presence()
    if presence_fn is None:
        return []
    p = presence_fn()
    if not p.get("connected"):
        return ["- ブラウザ: 閉じている（マスターは画面の前にいない。声は届かない）"]
    m = p.get("minutes_since_input")
    since = f"マスターが最後に話しかけたのは約{int(m)}分前" if m is not None else "今回の起動後、マスターはまだ話しかけていない"
    return [f"- ブラウザ: 開いている（声が届く。{since}）"]


def _format_tool_availability(settings: dict) -> list[str]:
    """使える機能・使えない機能の行（`<system_state>` に載せる）。

    ★**「使える・使えない」はここ1か所に載せる**（2026-10-03 本人判断）。それまでは
      起動時に設定で `available_tools.md` の文面を切り替えて（`<!-- if -->` / `<!-- switch -->`）
      道具の説明に混ぜており、顔・声の登録と別の場所にあった。道具の説明は「何ができる道具か」だけにした。
    判定は `agent.config` の `is_*_available` をそのまま使う（dispatch の拒否と同じ判定。2本作らない）。
    どれもファイルの有無と設定値を見るだけなので、毎ターン作っても軽い。
    """
    lines = [
        "- Web検索（web_search）: "
        + ("使える" if is_web_search_available(settings)
           else "使えない（設定でオフになっているか、Tavily の API キーが未登録）"),
        "- メール（send_email）: "
        + ("使える" if is_email_available(settings)
           else "使えない（設定でオフになっているか、送信元のアドレス・アプリパスワードが未登録）"),
    ]
    # 曲づくりは3通り（無い／CPU だけ／GPU あり）。歌う（play_song）も同じ実体に依る
    music = "- 曲づくり・歌う（work_on_music / play_song）: "
    if not is_song_service_available(settings):
        lines.append(music + "使えない（曲づくりの機能が導入されていないか、設定でオフになっている）")
    elif is_audio_cover_available(settings):
        lines.append(music + "使える。音源しか無い曲のカバーもこの環境でできる")
    else:
        lines.append(music + "使える。ただし音源しか無い曲のカバーはこの環境ではできない"
                     "（MIDI のお手本が見つかる曲のカバー、オリジナル曲づくり、できた曲の編集、"
                     "口ずさんだ音の聞き取りはできる）")
    return lines


def _fetch_pending_reports() -> str:
    """**まだマスターに伝えていない裏作業の結果**（完了・失敗）。`<pending_reports>` に載せる。

    伝え終えたらミクが `acknowledged_task_id` に id を返し、記録から取り下げられる（それまで毎ターン載る）。
    取得できなければ空（会話は止めない）。
    """
    try:
        return "\n".join(task_record.format_pending(_KV_BASE_URL))
    except Exception:
        return ""


_FACE_MAX_PX = 640   # 顔検出前に縮小する長辺。実測: 原寸2084ms → 640px 306ms・一致度は不変


def _verify_faces(image_paths: List[str],
                  file_names: List[str]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """添付画像・カメラ画像の顔を照合する（LLM 投入前の前処理）。

    **カメラ画像だけでなく添付写真も対象にする**。人は写真を見せられたときも
    「誰が写っているか」を自動で見るため、カメラだけ特別扱いする理由がない
    （2026-08-09 ユーザー判断）。

    長辺 _FACE_MAX_PX へ縮小してから送る。実測でスマホ写真の原寸（4032x3024）は
    2084ms かかるのに対し 640px 縮小で 306ms、一致度は変わらなかった。
    サービス未起動・未登録でも会話は止めない（注釈が付かないだけ）。

    戻り値は **(顔の照合結果, 失敗の記録)**。失敗はここではログに書かず呼び出し側へ返す
    ——`agent_logger` は contextvars で現在の WF ログを持つため、
    ThreadPoolExecutor の別スレッドから `agent_log()` を呼んでも**どこにも残らない**
    （2026-08-11。顔照合が壊れていたのに痕跡がゼロで、発見が遅れた事後規約）。
    """
    if not image_paths:
        return [], []
    import base64
    import io as _io

    from PIL import Image, ImageOps

    out: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for path, name in zip(image_paths, file_names or image_paths):
        try:
            img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
            if max(img.size) > _FACE_MAX_PX:
                scale = _FACE_MAX_PX / max(img.size)
                img = img.resize((int(img.width * scale), int(img.height * scale)),
                                 Image.LANCZOS)
            buf = _io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            resp = requests.post(
                f"{_RECOGNITION_BASE_URL}/face/verify",
                json={"image_b64": base64.b64encode(buf.getvalue()).decode("ascii")},
                timeout=20,
            )
        except Exception as e:      # 読めない画像・サービス未起動
            errors.append({"file": name, "error": f"{type(e).__name__}: {e}"})
            continue
        if resp.status_code == 400:
            out.append({"file": name, "confidence": None, "box": None})   # 顔なし
            continue
        if resp.status_code != 200:
            errors.append({"file": name, "status": resp.status_code, "body": resp.text[:300]})
            continue
        body = resp.json()
        if not body.get("enrolled"):
            return [], errors     # 未登録なら注釈そのものを出さない
        for f in body.get("faces", []):
            out.append({"file": name, "confidence": f.get("confidence"), "box": f.get("box")})
    return out, errors


def _gather_context_data() -> Dict[str, str]:
    """LLM 常時投入用のコンテキストを取得する。

    Phase 2（コンテキスト軽量化）以降の構成:
      - long_memory: 軽量版（episodic.events / procedural / semantic.visited を除去）
      - this_conversation: 全文（KV）
      - miku_note: 全文（KV）
      - recent_conversation: 直近 7 件（最新 1 件だけ本文つき・それ以前は要約のみ）。
        本文が要るときは recall_detail_conversation で取り出す
      - system_state: システム状態（顔登録状態など。SoT から毎ターン導出・記憶には複製しない）

    背景:
      過去 Dify の 4 ノード（長期記憶取得・当該会話内容取得・メモ取得・直近5会話取得）相当の
      フル取得は 100k 字超になり TTFT を悪化させていた。Tool 想起アーキテクチャに移行する。
      詳細: .claude/skills/miku-dev/details/latency_benchmark.md
    """
    return {
        "long_memory": _kv_get_long_memory_lightweight(),
        "this_conversation": _kv_get("this_conversation"),
        "miku_note": _kv_get("miku_note"),
        "recent_conversation": _fetch_recent_conversation(_RECENT_CONVERSATION_LIMIT),
        "system_state": _fetch_system_state(),
        "pending_reports": _fetch_pending_reports(),
    }


def build_plan_prompt(partial_text: str) -> tuple[str, str, str]:
    """段階1（方針づくり）へ渡す instructions と user_prompt を組み立てる（2026-08-30）。

    **会話ターンとまったく同じ手順**で作り、ユーザー入力の位置に「ここまで聞き取れた発話」を
    入れる。同じ組み立てを通すのは、明示キャッシュのプレフィックスを共有するため
    （別の作り方をすると、段階1のぶんキャッシュが二重になる）。

    user_prompt は会話ターンと同じ境界で前半・後半に割って返す。**前半が段階2・事前作成と
    同じ文字列になる**ので、3者が1つのキャッシュを使い回せる（段階1は1ターンに最大3回
    走るため、フルで送ると段階2より費用が重くなる）。

    Args:
        partial_text: STT の interim で得られた、ここまでの発話

    Returns:
        (instructions, キャッシュに載せる前半, 毎回送る後半)
    """
    from agent.agent_logger import strip_markers
    from agent.main import _build_marked_instructions, build_user_prompt_parts

    gathered = _gather_context_data()
    context_text = _build_context(
        long_memory=gathered.get("long_memory", ""),
        recent_conversation=gathered.get("recent_conversation", ""),
        miku_note=gathered.get("miku_note", ""),
        this_conversation=gathered.get("this_conversation", ""),
        system_state=gathered.get("system_state", ""),
        first_meeting=False,
        pending_reports=gathered.get("pending_reports", ""),
    )
    inp = MainChatInput(
        context=context_text,
        user_input=partial_text,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        loop_counter=1,
        # 会話ターン（mode=0）と同じ状況の節を入れる。キャッシュの照合に instructions も入るので、
        # 入れないと会話ターンのキャッシュと一致しない（2026-10-09）
        conversation_situation=_build_conversation_situation(0, "0"),
    )
    head, tail = build_user_prompt_parts(inp, file_names="")
    return strip_markers(_build_marked_instructions(inp)), head, tail


def build_prewarm_material() -> tuple[str, str, str] | None:
    """ブラウザ接続時にキャッシュを先に作るための材料を組み立てる（2026-08-30）。

    **これが無いと、その会話の最初の発話だけ 2 秒余計に待たされる**
    （キャッシュはループの頭で作られるため、初回は作成を待つことになる）。

    会話ターンと同じ手順で context を作り、`build_user_prompt_parts` と同じ境界で割って
    「キャッシュに載る前半」だけを返す。**同じ文字列になることが重要**で、
    ずれると事前に作ったものが使われず、二重に作ることになる。

    Returns:
        (instructions, キャッシュに載せる本文, モデル名)。組み立てられなければ None。
    """
    from agent.config import get_pydantic_ai_flash_model
    from agent.main import _build_marked_instructions, build_user_prompt_parts
    from agent.agent_logger import strip_markers
    from agent.schemas import MainChatInput

    try:
        gathered = _gather_context_data()
        context_text = _build_context(
            long_memory=gathered.get("long_memory", ""),
            recent_conversation=gathered.get("recent_conversation", ""),
            miku_note=gathered.get("miku_note", ""),
            this_conversation=gathered.get("this_conversation", ""),
            system_state=gathered.get("system_state", ""),
            first_meeting=False,
            pending_reports=gathered.get("pending_reports", ""),
        )
        # ループ1 と同じ形で組み立てる（前半だけ使うので、後半の中身は影響しない）
        # 会話ターン（mode=0）と同じ状況の節で作る。★以前は状況の節が空の instructions でキャッシュを作り、
        #   キャッシュの照合が前半だけだったので、どのモードのターンも状況の節なしで答えていた（2026-10-09）
        inp = MainChatInput(context=context_text, loop_counter=1,
                            conversation_situation=_build_conversation_situation(0, "0"))
        instructions = strip_markers(_build_marked_instructions(inp))
        head, _ = build_user_prompt_parts(inp, file_names="")
        if not head:
            return None
        return instructions, head, get_pydantic_ai_flash_model(_settings).split(":", 1)[-1]
    except Exception:
        return None      # 材料が揃わなければ諦める（会話が始まれば作られる）


# =============================================================================
# 会話エントリの整形 / 後処理（Dify post_process / format_conversation_entry 相当）
# =============================================================================


def _get_current_time_jst() -> str:
    """現在時刻を JST `YYYY-MM-DD HH:MM:SS` 文字列で返す。"""
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")


def _format_conversation_entry(
    user_text: str,
    file_names: List[str],
    response_objects: List[Dict[str, Any]],
    final_response: str,
    user_input_time: str,
    assistant_output_time: str,
    trigger_key: str = "user",
) -> str:
    """新しい会話エントリを JSON 文字列に整形する。

    trigger_key = このターンの**起点**（"user"=マスターの発言 / "assistant"=ミク自身の起点
    〔自由行動・自発話しかけ・裏作業の完了〕/ "others"=無応答タイマー等）。
    従来は起点を常に "user" と書いていたが、mode1〜3 はマスターの発言ではないため是正した（2026-07-26）。
    ※このログは会話順に連結する**疑似JSON**で、起点=assistant のターンは同名キーが2つ並ぶ。
      LLM が読む前提の形式なので `json.loads` で読まないこと（読むならエントリ単位・重複キー対応が要る）。
    """
    states: List[Dict[str, Any]] = []
    for s in response_objects:
        states.append({
            "inner_thoughts": s.get("inner_thoughts", ""),
            "next_step_tool": s.get("tool_name", "null"),
            "response": s.get("response", "null"),
        })

    # LLM が最初のループで記述したファイル内容説明（file_content_description）を取り出す
    file_description: str | None = None
    if response_objects:
        raw_desc = response_objects[0].get("file_content_description", "")
        if raw_desc and raw_desc.strip().lower() != "null":
            file_description = raw_desc

    trigger_block = json.dumps({
        "time": user_input_time,
        "text": user_text,
        "file_names": file_names,
        "file_content_description": file_description,
    }, ensure_ascii=False, indent=2)
    assistant_block = json.dumps({
        "time": assistant_output_time,
        "before_response_states": states,
        "response": final_response,
    }, ensure_ascii=False, indent=2)

    def _indent(block: str) -> str:                     # dict では同名キーを持てないため文字列で組み立てる
        return "\n".join("  " + ln for ln in block.splitlines()).lstrip()

    return (f'{{\n  "{trigger_key}": {_indent(trigger_block)},\n'
            f'  "assistant": {_indent(assistant_block)}\n}}')


def _append_to_this_conversation(old_conversation: str, new_entry: str) -> str:
    """既存の this_conversation に新エントリを連結する。"""
    return f"{old_conversation}\n\n{new_entry}" if old_conversation else new_entry


# =============================================================================
# files_data → AgentFile 変換
# =============================================================================

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")
_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".avi", ".mkv")

# 動画・音声を LLM に inline 添付する際の合計サイズ上限。
# Gemini の inline リクエストは合計約20MBまでのため、プロンプト・画像分のマージンを引いた値。
# 超過分は添付せず、状況テキストに「見えていない」ことを明示する（作話防止）。
_MAX_INLINE_MEDIA_BYTES = 18 * 1024 * 1024

# 一時ファイル化する際の拡張子フォールバック（filename に拡張子がない場合）
_DEFAULT_SUFFIX_BY_TYPE = {"image": ".jpg", "video": ".mp4", "audio": ".mp3"}


def _classify_file(filename: str) -> str:
    """ファイル名から type を推定する（image/audio/video）。不明は image にフォールバック。"""
    lower = filename.lower()
    if lower.endswith(_AUDIO_EXTS):
        return "audio"
    if lower.endswith(_VIDEO_EXTS):
        return "video"
    return "image"


def _files_to_agent_files(files_data: List[Dict[str, Any]]) -> List[AgentFile]:
    """gateway_service の files_data 形式を AgentFile に変換する。

    files_data は dict のリストで、`filename`（必須）/ `path` または `local_path`（任意）を持つ。
    `path`/`local_path` が指定されていない場合は `local_path=None` のまま渡す（画像描画はスキップ）。
    """
    out: List[AgentFile] = []
    for f in files_data or []:
        name = f.get("filename") or f.get("name") or ""
        if not name:
            continue
        ftype = f.get("type") or _classify_file(name)
        local_path = f.get("local_path") or f.get("path")
        out.append(AgentFile(
            type=ftype,  # type: ignore[arg-type]
            file_id=name,
            is_camera_capture=name.startswith("camera_"),
            local_path=local_path,
        ))
    return out


# =============================================================================
# 戻り値 dict の正規化
# =============================================================================


def _normalize_response(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str) and text.strip().lower() == "null":
        return ""
    return str(text)


def _normalize_motion(motion: Any) -> str:
    if motion is None:
        return "01"
    if isinstance(motion, str) and motion.strip().lower() == "null":
        return "01"
    return str(motion) or "01"


def _normalize_int(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


# =============================================================================
# ストリーム駆動本体
# =============================================================================


async def _drive_loop(
    main_input: MainChatInput,
    conversation_history: str,
    on_text_delta: Optional[Callable[[str], None]],
    on_field_value: Optional[Callable[[str, Any], None]],
    on_object_complete: Optional[Callable[[Dict[str, Any], int], None]],
    turn_handle=None,
    on_voice_tone: Optional[Callable[[str], None]] = None,
    on_tool_started: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """run_main_loop_stream を駆動し、event を各コールバックへ橋渡しする。

    `on_tool_started(objects)` はツールを呼ぶ直前に、そこまでのループ出力を渡す
    （ツール前に話した言葉を、ターンの終わりを待たずに会話ログへ書くため・2026-10-09）。

    `conversation_history` は subagent（memory_search_compose / web_search_agent 等）へ
    `## ここまでの会話内容` セクションとして渡される。通常 KV `this_conversation` の中身を
    そのまま渡せば良い（Dify でも当該会話本文が subagent 入力に流れる仕様）。

    Returns:
        {
            "final_output": MainChatOutput | None,
            "objects": list[dict],  # 各ループの structured_output dump（on_object_complete 引数の累積）
            "final_response_text": str,  # 最終ループの response（"null" 正規化前）
        }
    """
    state = ConversationState(conversation_history=conversation_history)
    objects: List[Dict[str, Any]] = []
    final_output: MainChatOutput | None = None
    object_index = 0

    async for event in run_main_loop_stream(main_input, state, turn_handle=turn_handle):
        if isinstance(event, LoopStarted):
            print(f"[STREAM] Loop {event.loop_counter} 開始")
            agent_log("loop_started", "ループ反復開始", meta={"loop": event.loop_counter})

        elif isinstance(event, LoopAborted):
            # 畳み込み: 中断されたループ。会話ログには inner_thoughts（＋TTS へ流れた
            # 部分 response）のみの state として残す（発話されなかった内容は記録しない）
            if event.partial_inner_thoughts or event.partial_response:
                objects.append({
                    "inner_thoughts": event.partial_inner_thoughts,
                    "tool_name": "null",
                    "response": event.partial_response or "null",
                })
            agent_log(
                "loop_aborted",
                "考え中の追加発話によりループを中断（畳み込み再ループへ）",
                meta={"partial_response_chars": len(event.partial_response or "")},
            )
            print("[STREAM] LoopAborted: 畳み込み再ループ")

        elif isinstance(event, ResponseDelta):
            # デルタは TTS へ逐次送るのみ。ターミナル表示は IterationComplete で
            # 応答全文を1行出力する（デルタを逐次 print すると Fish Audio スレッドの
            # 非同期ログと同じ行に混線し、末尾も取りこぼすため）。
            if on_text_delta is not None and event.text:
                try:
                    on_text_delta(event.text)
                except Exception as cb_err:
                    print(f"[WARN] on_text_delta 失敗 (ResponseDelta): {cb_err}")

        elif isinstance(event, VoiceToneDecided):
            # response より先に届く。TTS のストリーマは既に起動しているので即渡す
            if on_voice_tone is not None and event.tone:
                try:
                    on_voice_tone(event.tone)
                except Exception as cb_err:
                    print(f"[WARN] on_voice_tone 失敗: {cb_err}")

        elif isinstance(event, IterationComplete):
            obj = event.output.model_dump()
            # 応答全文を1行で出力（末尾改行つき）。output.response が確定値なので欠けない。
            _resp_text = obj.get("response") or ""
            if _resp_text and _resp_text != "null":
                print(f"[応答] {_resp_text}")
            objects.append(obj)
            object_index += 1
            agent_log(
                "main_chat_response",
                "ミクメイン応答LLMからの出力（structured_output 全フィールド）",
                output=obj,
                meta={"loop": object_index},
            )
            if on_field_value is not None:
                for key, value in obj.items():
                    try:
                        on_field_value(key, value)
                    except Exception as cb_err:
                        print(f"[WARN] on_field_value 失敗 ({key}): {cb_err}")
            if on_object_complete is not None:
                try:
                    on_object_complete(obj, object_index)
                except Exception as cb_err:
                    print(f"[WARN] on_object_complete 失敗: {cb_err}")
            print(
                f"[STREAM] IterationComplete #{object_index}: "
                f"requires_tool={obj.get('requires_tool')} tool_name={obj.get('tool_name')}"
            )

        elif isinstance(event, ToolStarted):
            print(f"[STREAM] ToolStarted: {event.tool_name}")
            agent_log(
                "tool_started",
                f"ツール呼び出し開始: {event.tool_name}（runner→tool への入力）",
                input=event.raw_input,
                meta={"tool": event.tool_name},
            )
            if on_tool_started is not None:
                try:
                    on_tool_started(list(objects))
                except Exception as cb_err:
                    print(f"[WARN] on_tool_started 失敗（会話は続ける）: {cb_err}")

        elif isinstance(event, ToolCompleted):
            print(f"[STREAM] ToolCompleted: {event.tool_name} (chars={len(event.result)})")
            agent_log(
                "tool_completed",
                f"ツール呼び出し完了: {event.tool_name}（tool→runner からの出力）",
                output=event.result,
                meta={"tool": event.tool_name, "summary_chars": len(event.summary)},
            )
            # ツール結果の要点を、そのツールを選んだループの出力へ載せる。
            # ToolCompleted は該当ループの IterationComplete の直後に来るため objects[-1] が対象。
            # これが会話ログの before_response_states[].tool_summary になり、
            # ターンをまたいで「さっき調べたこと」を参照できるようにする（2026-08-26）
            if event.summary and objects:
                objects[-1]["tool_summary"] = event.summary

        elif isinstance(event, LoopFinished):
            final_output = event.final_output
            print(f"[STREAM] LoopFinished: reason={event.reason}")
            agent_log(
                "loop_finished",
                "ループ全体終了",
                meta={
                    "reason": event.reason,
                    "loop_count": event.state.loop_counter,
                },
            )

    final_response_text = ""
    if final_output is not None:
        final_response_text = _normalize_response(final_output.response)

    return {
        "final_output": final_output,
        "objects": objects,
        "final_response_text": final_response_text,
    }


# =============================================================================
# 公開関数（gateway_service.py からの import 切替対象）
# =============================================================================


def process_agent_request(
    input_text: str,
    files_data: List[Dict[str, Any]],
    mode: int = 0,
    unresponsive_time: str = "300",
    voice_segments_raw: str = "",
    on_filler_ready: Optional[Callable[[str], None]] = None,
    on_filler_motion: Optional[Callable[[str], None]] = None,
    on_text_delta: Optional[Callable[[str], None]] = None,
    on_field_value: Optional[Callable[[str, Any], None]] = None,
    on_object_complete: Optional[Callable[[Dict[str, Any], int], None]] = None,
    turn_handle=None,
    on_voice_tone: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Pydantic AI 実装の main_chat エントリポイント。

    `agent_main_dify_stream.process_agent_request` と同一シグネチャ。Phase 7 では
    本関数を import 切替するだけで Dify から差し替えられる。

    on_filler_ready: Phase 6 即応フィラー用。mode=0 かつ非 None のとき、
        ターン開始と同時にフィラー LLM を別スレッドで起動し、テキスト確定時に呼ばれる。
        gateway 側で Fish Audio chap0 に流す想定。フィラーが不要な呼び出し元（CLI 等）は
        None で渡せばよい。
    on_filler_motion: 会話中リアクション用。フィラーがリアクションモーション（"none" 以外）を
        返したとき motion コードで呼ばれる。gateway 側で reaction_motion をブラウザへ emit する想定。
    """
    user_input_time = _get_current_time_jst()

    with open_log(WF_MAIN_CHAT, mode=mode):
        agent_log(
            "request",
            "process_agent_request の受領入力（gateway → agent_main）",
            input={
                "input_text": input_text,
                "files_data": files_data or [],
                "mode": mode,
                "unresponsive_time": unresponsive_time,
            },
            meta={"user_input_time": user_input_time},
        )
        return _process_agent_request_inner(
            input_text=input_text,
            files_data=files_data or [],
            mode=mode,
            unresponsive_time=unresponsive_time,
            user_input_time=user_input_time,
            voice_segments_raw=voice_segments_raw,
            on_filler_ready=on_filler_ready,
            on_filler_motion=on_filler_motion,
            on_text_delta=on_text_delta,
            on_field_value=on_field_value,
            on_object_complete=on_object_complete,
            turn_handle=turn_handle,
            on_voice_tone=on_voice_tone,
        )


def _spawn_filler_thread(
    input_text: str,
    on_filler_ready: Callable[[str], None],
    on_filler_motion: Optional[Callable[[str], None]] = None,
) -> None:
    """Phase 6 即応フィラー: 別スレッドで filler_compose を呼び、結果をコールバック通知する。

    本応答 LLM が context 60k 字を処理して 9 秒程度かかる間、並列で短いフィラー（~10字）を
    Flash Lite に生成させる。ターン開始から ~600ms で on_filler_ready が呼ばれる想定。
    失敗時は何もしない（gateway 側は filler_done を timeout で待つ設計）。
    """
    def _runner() -> None:
        try:
            # filler_compose は asyncio 必要なので、専用スレッド内で asyncio.run を回す
            from agent.subagents.filler_compose import run_filler_compose
            filler = asyncio.run(run_filler_compose(input_text, mode="final"))
            if filler.motion != "none" and on_filler_motion is not None:
                # 送信直後リアクション: gateway 経由で reaction_motion をブラウザへ
                try:
                    on_filler_motion(filler.motion)
                except Exception as e:
                    print(f"[WARN] on_filler_motion 失敗: {e}")
            if filler.utterance:
                on_filler_ready(filler.utterance)
            else:
                # 空文字（API 失敗等）でもコールバックは呼ぶ。gateway 側で filler_done をセットして
                # main を待たせない処理に流れる。
                on_filler_ready("")
        except Exception as e:
            print(f"[WARN] filler thread 失敗: {e}")
            try:
                on_filler_ready("")
            except Exception:
                pass

    threading.Thread(target=_runner, daemon=True, name="filler_compose").start()


def _process_agent_request_inner(
    input_text: str,
    files_data: List[Dict[str, Any]],
    mode: int,
    unresponsive_time: str,
    user_input_time: str,
    voice_segments_raw: str,
    on_filler_ready: Optional[Callable[[str], None]],
    on_filler_motion: Optional[Callable[[str], None]],
    on_text_delta: Optional[Callable[[str], None]],
    on_field_value: Optional[Callable[[str, Any], None]],
    on_object_complete: Optional[Callable[[Dict[str, Any], int], None]],
    turn_handle=None,
    on_voice_tone: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    _temp_image_files: List[str] = []
    try:
        # --- 0. Phase 6 即応フィラー: ユーザー入力モード（mode=0）かつ
        #        gateway が on_filler_ready を渡してきたときのみ並列起動 ---
        if mode == 0 and on_filler_ready is not None and input_text and input_text.strip():
            _spawn_filler_thread(input_text, on_filler_ready, on_filler_motion)

        # --- 1. 入力テキスト・状況・添付ファイル ---
        situation = _build_conversation_situation(mode, str(unresponsive_time))
        user_text = _build_user_input_text(mode, input_text)
        agent_files = _files_to_agent_files(files_data or [])

        # ブラウザ upload ファイル（content bytes あり・local_path なし）を一時ファイルに保存して
        # local_path を解決する（画像・動画・音声すべて）。リクエスト完了後に finally で削除する。
        _files_by_name = {(f.get("filename") or f.get("name") or ""): f for f in (files_data or [])}
        for af in agent_files:
            if af.local_path is None:
                raw = _files_by_name.get(af.file_id, {})
                content: bytes | None = raw.get("content")
                if content:
                    suffix = Path(af.file_id).suffix or _DEFAULT_SUFFIX_BY_TYPE.get(af.type, ".bin")
                    fd_num, tmp_path = tempfile.mkstemp(suffix=suffix)
                    try:
                        with os.fdopen(fd_num, "wb") as fp:
                            fp.write(content)
                        af.local_path = tmp_path
                        _temp_image_files.append(tmp_path)
                    except Exception as e:
                        print(f"[WARN] 一時ファイル保存失敗 ({af.file_id}): {e}")
                        try:
                            os.close(fd_num)
                        except Exception:
                            pass

        attached_images: List[str] = [
            f.local_path for f in agent_files if f.type == "image" and f.local_path
        ]
        image_file_names: List[str] = [
            f.file_id for f in agent_files if f.type == "image" and f.local_path
        ]
        file_names_list: List[str] = [f.file_id for f in agent_files]
        file_names_text = "\n".join(file_names_list)

        # 動画・音声の inline 添付（合計 _MAX_INLINE_MEDIA_BYTES まで）。
        # 超過したファイルは LLM に渡さず、状況テキストに明示して
        # 「見えているかのように話す」作話を防ぐ
        attached_media: List[str] = []
        # 音声だけの一覧（voice_enroll ツールが声の登録に使う。LLM への添付は attached_media 側）。
        # サイズ上限の対象外にしているのは、登録用の音声はローカルで読むだけで LLM へ送らないため
        attached_audio: List[str] = [
            f.local_path for f in agent_files if f.type == "audio" and f.local_path
        ]
        oversized_media_names: List[str] = []
        media_budget = _MAX_INLINE_MEDIA_BYTES
        for af in agent_files:
            if af.type in ("video", "audio") and af.local_path:
                try:
                    size = os.path.getsize(af.local_path)
                except OSError:
                    continue
                if size <= media_budget:
                    attached_media.append(af.local_path)
                    media_budget -= size
                else:
                    oversized_media_names.append(
                        f"{af.file_id}（約{size / (1024 * 1024):.0f}MB）"
                    )
        if oversized_media_names:
            situation += (
                "\n\n【添付ファイルに関する重要な注意】\n"
                "以下の添付ファイルはサイズが大きすぎるため、あなたには内容が渡されていません: "
                + "、".join(oversized_media_names)
                + "\nファイル名から内容を推測して、見えているかのように話すことは禁止です。"
                "「ファイルが大きすぎて内容までは見られない」ことを正直に伝えてください。"
            )

        # --- 2. context 素材の並列取得 ---
        # KV: long_memory / this_conversation / miku_note, VDB: recent_conversation
        # 顔照合（前処理）も**ここで並行**に走らせる。声は STT が ASR と並行に済ませて
        # おり、顔は画像がここに届くのでこのタイミングが最速になる
        face_future = None
        if attached_images:
            import concurrent.futures as _futures
            _face_pool = _futures.ThreadPoolExecutor(max_workers=1)
            face_future = _face_pool.submit(_verify_faces, attached_images, image_file_names)
            _face_pool.shutdown(wait=False)

        gathered = _gather_context_data()
        long_memory = gathered.get("long_memory", "")
        this_conversation = gathered.get("this_conversation", "")
        miku_note = gathered.get("miku_note", "")
        recent_conversation = gathered.get("recent_conversation", "")
        agent_log(
            "context_data_loaded",
            "KV / VDB から context 素材（long_memory / this_conversation / miku_note / recent_conversation）を並列取得した結果",
            output={
                "long_memory": long_memory,
                "this_conversation": this_conversation,
                "miku_note": miku_note,
                "recent_conversation": recent_conversation,
            },
        )

        # --- 3. context 組み立て ---
        # 初対面判定: セッション会話が空 かつ 記憶登録待ちの会話も無い かつ 長期記憶が完全に空。
        # 登録待ち（registration_waiting_conversation）は「過去に会話したが記憶登録が
        # スキップ/未処理」の状態がありうるため、long_memory 0件だけでは初対面と断定できない。
        # 短絡評価により VDB/KV への追加照会はフレッシュな初回ターンに限られる
        first_meeting = (
            not this_conversation.strip()
            and not _kv_get("registration_waiting_conversation").strip()
            and _is_long_memory_empty()
        )
        context_text = _build_context(
            long_memory=long_memory,
            recent_conversation=recent_conversation,
            miku_note=miku_note,
            this_conversation=this_conversation,
            system_state=gathered.get("system_state", ""),
            first_meeting=first_meeting,
            pending_reports=gathered.get("pending_reports", ""),
        )
        agent_log(
            "pre_loop_materials",
            "ループ前素材の準備完了（mode→状況テキスト変換 / context 結合 / ファイル名抽出 を一括）",
            output={
                "conversation_situation": situation,
                "user_input_text": user_text,
                "file_names": file_names_list,
                "attached_image_paths": attached_images,
                "attached_media_paths": attached_media,
                "attached_audio_paths": attached_audio,
                "oversized_media": oversized_media_names,
                "context": context_text,
            },
        )

        # --- 3.5 知覚情報の付与（声・顔をここで一本化して整形する） ---
        # 照合の実行は声=STT / 顔=直前の並行実行と分かれるが、**LLM へどう見えるかを
        # 決めるのはここ1箇所**にする（utils/perception.py）
        voice_segments: List[Dict[str, Any]] = []
        if voice_segments_raw:
            try:
                loaded = json.loads(voice_segments_raw)
                if isinstance(loaded, list):
                    voice_segments = loaded
            except (json.JSONDecodeError, TypeError):
                pass
        faces: List[Dict[str, Any]] = []
        face_errors: List[Dict[str, Any]] = []
        if face_future is not None:
            try:
                faces, face_errors = face_future.result(timeout=20)
            except Exception as e:
                face_errors = [{"error": f"{type(e).__name__}: {e}"}]
        if face_errors:
            # 握りつぶさない。顔照合は失敗しても会話を止めないぶん、記録が唯一の手がかりになる
            print(f"[知覚] 顔照合に失敗（注釈なしで続行）: {face_errors}")
            agent_log(
                "face_verify_failed",
                "顔照合に失敗した画像（注釈なしで会話を続行）",
                output={"errors": face_errors},
                meta={"images": len(attached_images), "failed": len(face_errors)},
            )
        if voice_segments or faces:
            user_text = perception.build_input_text(user_text, voice_segments, faces)
            agent_log(
                "perception",
                "LLM 投入前に付与した知覚情報（声の区間ごと一致度・画像の顔ごと一致度）",
                output={"user_input_text": user_text},
                meta={"voice_segments": len(voice_segments), "faces": len(faces)},
            )

        # 現在心境（感情・興味）を読み、注入文を作る。番号は build_view が確定した並びに
        # 対して解決するので、適用時に同じ view を使い回す（順序がずれると宛先が変わる）
        cm_state = current_mind.load(_kv_get("current_mind"))
        cm_view = current_mind.build_view(cm_state, user_input_time)
        cm_text = current_mind.render(cm_view)

        # --- 4. AgentInput / MainChatInput 構築 ---
        agent_input = AgentInput(
            text=user_text,
            files=agent_files,
            session_id="session_stream",
            timestamp=user_input_time,
            mode=mode,
            unresponsive_time=_normalize_int(unresponsive_time, 300),
        )
        main_input = MainChatInput(
            loop_counter=1,
            max_loops=MAX_LOOPS,
            user_input=agent_input.text,
            current_time=user_input_time,
            file_names=file_names_text,
            context=context_text,
            current_mind=cm_text,
            tool_result="",
            conversation_situation=situation,
            attached_images=attached_images,
            image_file_names=image_file_names,
            attached_media=attached_media,
            attached_audio=attached_audio,
        )

        # --- 5. ストリーム駆動（永続ループに投入） ---
        # ★話した時点で会話ログへ書く（2026-10-09 本人指示）。ツールを呼ぶ直前に、ここまでの
        #   依頼とツール前の発言を仮の行として書く。ターンの終わりに⑦がターン開始時の
        #   this_conversation から書き直すので、仮の行は本番の行に置き換わる（二重にならない）。
        #   以前はターンの終わりにだけ書いていたので、deep_think 等の待ち中は履歴に出ず、
        #   途中で落ちると依頼ごと消えた。旧形式の間は従来どおり終わりにだけ書く。
        def _save_in_progress(objs: List[Dict[str, Any]]) -> None:
            if not conversation_log.is_new_format(this_conversation):
                return
            records = conversation_log.build_turn_records(
                mode=mode,
                trigger_text=user_text,
                user_input_time=user_input_time,
                assistant_output_time=_get_current_time_jst(),
                final_response="",
                response_objects=objs,
                file_names=file_names_list,
                file_content_description=None,
                additions=list(turn_handle.collected) if turn_handle is not None else [],
            )
            _kv_post("this_conversation", conversation_log.append_records(this_conversation, records))
            print(f"[STREAM] 会話文を KV に仮保存しました（ツール前・{len(objs)}ループ分）")

        print(f"[STREAM] process_agent_request 開始: mode={mode}, input={user_text[:50]}")
        loop_result = _run_async(_drive_loop(
            main_input=main_input,
            # this_conversation を subagent の `## ここまでの会話内容` 入力として渡す
            # （新形式なら LLM 投入用のコンパクト JSONL ビューに変換。旧形式はそのまま）
            conversation_history=conversation_log.to_llm_view(this_conversation),
            on_text_delta=on_text_delta,
            on_field_value=on_field_value,
            on_object_complete=on_object_complete,
            turn_handle=turn_handle,
            on_voice_tone=on_voice_tone,
            on_tool_started=_save_in_progress,
        ))
        final_output: MainChatOutput | None = loop_result["final_output"]
        objects: List[Dict[str, Any]] = loop_result["objects"]
        if final_output is None:
            raise RuntimeError("ループから最終応答を取得できませんでした")

        # --- 6. 戻り値正規化 ---
        response_text = loop_result["final_response_text"]
        motion = _normalize_motion(final_output.motion)
        inactivity_timeout = _normalize_int(final_output.inactivity_timeout, 60)
        next_proactive_in_minutes = _normalize_int(final_output.next_proactive_in_minutes, -1)
        print(
            f"[STREAM] 最終応答: response_chars={len(response_text)} "
            f"motion={motion} inactivity_timeout={inactivity_timeout} "
            f"next_proactive_in_minutes={next_proactive_in_minutes}"
        )

        # --- 7. 後処理: 会話エントリを KV `this_conversation` に追記 ---
        # 新形式（正式JSON配列・utils/conversation_log.py）で追記する。ただし既存データは
        # 変換しない方針のため、旧形式（疑似JSON）が残っている間は旧形式で追記を続け、
        # 記憶整理でファイルが空になった時点から新形式が始まる（自己解決型移行・2026-07-31）。
        assistant_output_time = _get_current_time_jst()
        # 考え中の追加発話（畳み込み・P2）: runner が drain 済みのものを turn_handle が保持
        additions: List[tuple] = list(turn_handle.collected) if turn_handle is not None else []
        if conversation_log.is_new_format(this_conversation):
            file_desc: str | None = None
            if objects:
                raw_desc = objects[0].get("file_content_description", "")
                if raw_desc and raw_desc.strip().lower() != "null":
                    file_desc = raw_desc
            records = conversation_log.build_turn_records(
                mode=mode,
                trigger_text=user_text,
                user_input_time=user_input_time,
                assistant_output_time=assistant_output_time,
                final_response=response_text,
                response_objects=objects,
                file_names=file_names_list,
                file_content_description=file_desc,
                additions=additions,
            )
            full_conversation = conversation_log.append_records(this_conversation, records)
            new_entry = json.dumps(records, ensure_ascii=False)
        else:
            # 旧形式（移行前の残存データ）: 追加発話は user_text へ時系列追記で表現する
            legacy_user_text = user_text + "".join(
                f"\n[{t} 追加発話] {text}" for t, text in additions
            )
            new_entry = _format_conversation_entry(
                user_text=legacy_user_text,
                file_names=file_names_list,
                response_objects=objects,
                final_response=response_text,
                user_input_time=user_input_time,
                assistant_output_time=assistant_output_time,
                trigger_key=_trigger_key_of(mode),
            )
            full_conversation = _append_to_this_conversation(this_conversation, new_entry)
        _kv_post("this_conversation", full_conversation)
        print("[STREAM] 会話文を KV に格納しました")
        agent_log(
            "conversation_saved",
            "this_conversation を KV に追記（agent_main → KV への出力）",
            output={"new_entry": new_entry},
            meta={"full_conversation_chars": len(full_conversation)},
        )

        # 現在心境を更新して保存する。減衰・床・件数上限はここで効くので、
        # LLM が過剰に積んでも状態は膨らまない（プロンプト側の抑制と二段構え）。
        #
        # **全ループぶんを集める**。ツールを使うターンは複数ループになり、気分や興味は
        # 途中のループで出ることが多い（最終ループはツール結果をまとめるだけで空になりがち）。
        # final_output だけを見ていたときは、自由行動で notice しても丸ごと捨てられていた
        # （2026-08-28 検出。実ログで notice 2件がループ2に出て applied=[] だった）。
        # 1ターンの新規上限は apply 側が全体に対してかけるので、ここでは連結するだけでよい。
        def _collect(key: str) -> list:
            return [u for obj in objects for u in (obj.get(key) or [])]

        cm_new, cm_logs = current_mind.apply(
            cm_state,
            cm_view,
            _collect("mood_updates"),
            _collect("interest_updates"),
            assistant_output_time,
        )
        _kv_post("current_mind", current_mind.dumps(cm_new))
        agent_log(
            "current_mind_saved",
            "現在心境（感情・興味）を更新して KV に保存",
            output={"applied": cm_logs},
            meta={
                "feelings": len(cm_new["feelings"]),
                "interests": len(cm_new["interests"]),
                "capacity": round(cm_view["capacity"], 3),
            },
        )

        # 伝え終えた裏作業を箱から出す（会話ログに残ったので、もう気にしなくてよい）。
        # **発話が無いターンは伝わっていない**ので取り下げない（自由行動中に
        # システム状態を見て「伝えた」と誤判定されるのを防ぐコード側のガード）。
        ack_id = getattr(final_output, "acknowledged_task_id", None)
        if ack_id and response_text and response_text != "null":
            if task_record.remove(_KV_BASE_URL, ack_id):
                agent_log(
                    "task_acknowledged",
                    "裏作業の完了・失敗をマスターへ伝えたので記録から取り下げた",
                    output={"task_id": ack_id},
                )
        elif ack_id:
            print(f"[裏作業] 発話が無いターンなので取り下げない (task_id={ack_id})")

        result_dict = {
            "response": response_text,
            "session_file": "session_stream",
            "success": True,
            "inactivity_timeout": inactivity_timeout,
            "motion": motion,
            "next_proactive_in_minutes": next_proactive_in_minutes,
        }
        agent_log(
            "response",
            "process_agent_request の戻り値（agent_main → gateway）",
            output=result_dict,
        )
        return result_dict

    except Exception as e:
        print(f"[ERROR] process_agent_request 失敗: {e}")
        traceback.print_exc()
        # 失敗の中身を**利用者に見せられる形**にして返す（2026-09-20）。
        # 以前はここで作った文言がどこへも送られず、**ミクが黙るだけ**だった。
        # LLM 由来か本体の不具合かは、HTTP ステータスが読めるかで分ける
        from utils.api_error import describe, looks_like_llm_error
        _info = describe(e, source="Google AI Studio" if looks_like_llm_error(e) else "本体")
        error_dict = {
            "response": f"エラーが発生しました: {str(e)}",
            "session_file": "session_stream",
            "success": False,
            "error_notice": _info,          # gateway がこれをブラウザへ送る
            "inactivity_timeout": 60,
            "motion": "01",
            "next_proactive_in_minutes": -1,
        }
        agent_log(
            "response",
            "process_agent_request エラー戻り値（agent_main → gateway）",
            output=error_dict,
            meta={"exception": str(e), "traceback": traceback.format_exc()},
        )
        return error_dict
    finally:
        for _p in _temp_image_files:
            try:
                os.unlink(_p)
            except Exception:
                pass
