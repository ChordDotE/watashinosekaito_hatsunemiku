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

"""Pydantic AI Orchestrator のディスパッチ層 + ループ層。

【ディスパッチ層（Phase 3）】
各ツール名に対応するハンドラを静的 dict で定義し、
Orchestrator が選んだツールを呼び出して結果文字列を返す（dispatch_tool）。

戻り値 str は Dify WF の「ツールから得られた情報」（{{result}}）に相当する。

【ループ層（Phase 5）】
Dify「ループ」ノード相当。
統合応答ノード（main_agent）↔ dispatch_tool の往復をストリーミングで yield する。
各ループの response フィールドは生成と同時に ResponseDelta として yield されるため、
Gateway が TTS / Browser へリアルタイム送出できる。

Phase 3 時点のブランチ実装範囲:
  self_knowledge / recall_detail_conversation / web_search / deep_think /
  miku_note_update / send_email / long_memory_update
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from agent.agent_logger import log as agent_log, log_llm_usage, strip_markers
from agent.config import get_pydantic_ai_flash_model, load_settings
from agent.io_helpers import derive_file_names, to_binary_contents, to_media_binary_contents
from agent import context_cache, plan_stage
from agent.main import (
    _BASE_PROMPT, _build_marked_instructions, _build_user_prompt,
    build_user_prompt_parts as _build_user_prompt_parts, main_agent,
)
from agent.schemas import (
    ConversationState,
    Failure,
    MainChatInput,
    MainChatOutput,
    Success,
    partial_response_of,
    partial_voice_tone_of,
)
from agent.subagents.deep_think import DeepThinkInput, run_deep_think
from agent.subagents.email_compose import EmailComposeInput, run_email_compose
from agent.subagents.face_image_select import FaceImageSelectInput, run_face_image_select
from agent.subagents.memory_search_compose import MemorySearchComposeInput, run_memory_search_compose
from agent.subagents.recall_summary_experience import (
    RecallSummaryExperienceInput,
    run_recall_summary_experience,
)
from agent.subagents.miku_note_compose import MikuNoteComposeInput, run_miku_note_compose
from agent.tools.retrieval.check_news import CheckNewsInput, run_check_news
from agent.subagents.web_search_agent import WebSearchAgentInput, run_web_search_agent
from agent.tools.memory.long_memory_patch import run_long_memory_patch
from agent.tools.retrieval.recall_detail_conversation import (
    RecallDetailConversationInput,
    run_recall_detail_conversation,
)
from agent.tools.self.face_enroll import FaceEnrollInput, run_face_enroll
from agent.tools.self.voice_enroll import VoiceEnrollInput, run_voice_enroll
from agent.tools.self.self_knowledge import SelfKnowledgeInput, run_self_knowledge


# usage ログ用のモデル名ラベル（"google-gla:" 等のプレフィックスを除いた表記）
_MAIN_MODEL_LABEL = get_pydantic_ai_flash_model(load_settings()).split(":", 1)[-1]


# ループ1（最初の応答）だけ Gemini の内部 thinking を絞る。None にすると従来動作に戻る。
#
# **なぜ削ってよいか**: 出力スキーマが `purpose → inner_thoughts → response` という
# CoT 構造そのもので、思考は構造化出力側で行われている。thought summaries を実際に
# 取得して読むと、内部 thinking の内容は inner_thoughts とほぼ重複していた
# （直近10ターンを再生して検証。test/thinking_budget_ab/results/recent_thinking.md）。
# 実測: 既定 平均 12.20 秒 → LOW 平均 5.49 秒。ツール選択・motion・長期記憶の参照は
# いずれも劣化せず、むしろ具体的になるケースが見られた（2026-08-08 採用）。
#
# 利用者が触る値ではないため設定ファイルには出さない（ユーザー判断 2026-08-08）。
# 切り戻すときはここを None にする。ループ2以降は未検証なので既定のまま触らない。
_LOOP1_THINKING_LEVEL: str | None = "LOW"


def _cache_model_id() -> str:
    """明示キャッシュを作るときのモデル名（`google-gla:` などの接頭辞を外す）。"""
    return get_pydantic_ai_flash_model(load_settings()).split(":", 1)[-1]


def _loop_model_settings(loop_n: int, cached_name: str | None = None):
    """そのループで使う model_settings を返す（不要なら None）。

    `cached_name` があれば明示キャッシュを参照する。ループ1 の thinking 抑制とは
    独立して効くので、**ループ2以降でもキャッシュだけは使う**（loop2 は tool_result で
    プレフィックスが崩れ、暗黙キャッシュが 0% になっていた区間・2026-08-30）。
    """
    thinking = _LOOP1_THINKING_LEVEL if loop_n == 1 else None
    if thinking is None and cached_name is None:
        return None
    try:
        from pydantic_ai.models.google import GoogleModelSettings
    except ImportError:
        return None      # Google 以外のモデル構成でも動くようにする
    kw: dict = {}
    if thinking is not None:
        kw["google_thinking_config"] = {"thinking_level": thinking}
    if cached_name:
        kw["google_cached_content"] = cached_name
    return GoogleModelSettings(**kw)


def _log(tool_name: str, phase: str, detail: str = "") -> None:
    """ツール呼び出しのデバッグ出力。"""
    tag = f"[runner:{tool_name}:{phase}]"
    print(f"{tag} {detail}" if detail else tag)


# ── ブランチハンドラ ──────────────────────────────────────────

async def _handle_self_knowledge(raw_input: dict) -> str:
    _log("self_knowledge", "call", f"input={raw_input}")
    result = run_self_knowledge(SelfKnowledgeInput(**raw_input))
    if isinstance(result, Failure):
        _log("self_knowledge", "failure", result.reason)
        return f"[エラー] {result.reason}"
    _log("self_knowledge", "success", f"chars={len(result.value.text)}")
    return result.value.text


async def _handle_recall_detail_conversation(raw_input: dict) -> str:
    """Dify「記憶検索ワード考案 LLM → 記憶検索 → 検索結果パース」の 3 段を再現する。

    記憶の2層のうち**下層（会話の逐語）**を引く。クエリは `memory_search_compose`
    subagent が会話文脈から生成する（ミク自身が検索語を書くわけではない）。
    """
    conv_history = raw_input.get("conversation_history", "")
    user_input = raw_input.get("user_input", "")
    ai_judgements = raw_input.get("ai_judgements", "")
    tool_results = raw_input.get("tool_results", "")
    top_k = raw_input.get("top_k")

    # Step 1: 検索クエリを LLM で生成
    _log("recall_detail_conversation", "compose",
         f"history={len(conv_history)}c  input={len(user_input)}c")
    query = await run_memory_search_compose(MemorySearchComposeInput(
        conversation_history=conv_history,
        user_input=user_input,
        ai_judgements=ai_judgements,
        tool_results=tool_results,
    ))
    if query.startswith("[エラー]"):
        _log("recall_detail_conversation", "compose_failure", query)
        return query

    # Step 2: 生成したクエリで検索実行
    _log("recall_detail_conversation", "search", f"query={query!r}  top_k={top_k}")
    search_input = (
        RecallDetailConversationInput(query=query) if top_k is None
        else RecallDetailConversationInput(query=query, top_k=top_k)
    )
    result = run_recall_detail_conversation(search_input)
    if isinstance(result, Failure):
        _log("recall_detail_conversation", "failure", result.reason)
        return f"[エラー] {result.reason}"
    _log("recall_detail_conversation", "success", f"hits={len(result.value.hits)}")
    return result.value.formatted or "（検索結果なし）"


async def _handle_check_news(raw_input: dict) -> str:
    limit = raw_input.get("limit", 6)
    _log("check_news", "call", f"limit={limit}")
    result = run_check_news(CheckNewsInput(limit=limit))
    if isinstance(result, Failure):
        _log("check_news", "failure", result.reason)
        return f"[エラー] {result.reason}"
    _log("check_news", "success", f"titles={len(result.value.titles)}")
    return result.value.formatted


async def _handle_web_search(raw_input: dict) -> str:
    conv_history = raw_input.get("conversation_history", "")
    user_input = raw_input.get("user_input", "")
    ai_judgements = raw_input.get("ai_judgements", "")
    tool_results = raw_input.get("tool_results", "")
    max_results = raw_input.get("max_results", 5)
    _log(
        "web_search", "call",
        f"history={len(conv_history)}c  input={len(user_input)}c  "
        f"judge={len(ai_judgements)}c  prev_results={len(tool_results)}c  max={max_results}",
    )
    result = await run_web_search_agent(WebSearchAgentInput(
        conversation_history=conv_history,
        user_input=user_input,
        ai_judgements=ai_judgements,
        tool_results=tool_results,
        max_results=max_results,
    ))
    _log("web_search", "success", f"chars={len(result)}")
    return result


async def _handle_deep_think(raw_input: dict) -> str:
    user_input = raw_input.get("user_input", "")
    context = raw_input.get("context", "")
    loop_counter = raw_input.get("loop_counter", 1)
    attached_images = raw_input.get("attached_images", [])
    _log("deep_think", "call", f"user_input_chars={len(user_input)}  context_chars={len(context)}  loop={loop_counter}  images={len(attached_images)}")
    result = await run_deep_think(DeepThinkInput(
        base_prompt=raw_input.get("base_prompt", ""),
        loop_counter=loop_counter,
        max_loops=raw_input.get("max_loops", 5),
        context=context,
        user_input=user_input,
        attached_images=attached_images,
    ))
    _log("deep_think", "success", f"chars={len(result)}")
    return result


async def _handle_miku_note_update(raw_input: dict) -> str:
    context = raw_input.get("context", "")
    loop_counter = raw_input.get("loop_counter", 1)
    attached_images = raw_input.get("attached_images", [])
    _log("miku_note_update", "call", f"context_chars={len(context)}  loop={loop_counter}  images={len(attached_images)}")
    result = await run_miku_note_compose(MikuNoteComposeInput(
        base_prompt=raw_input.get("base_prompt", ""),
        loop_counter=loop_counter,
        max_loops=raw_input.get("max_loops", 5),
        context=context,
        current_time=raw_input.get("current_time", ""),
        user_input=raw_input.get("user_input", ""),
        file_names=raw_input.get("file_names", ""),
        attached_images=attached_images,
    ))
    _log("miku_note_update", "success", f"chars={len(result)}")
    return result


async def _handle_send_email(raw_input: dict) -> str:
    context = raw_input.get("context", "")
    loop_counter = raw_input.get("loop_counter", 1)
    attached_images = raw_input.get("attached_images", [])
    _log("send_email", "call", f"context_chars={len(context)}  loop={loop_counter}  images={len(attached_images)}")
    result = await run_email_compose(EmailComposeInput(
        base_prompt=raw_input.get("base_prompt", ""),
        loop_counter=loop_counter,
        max_loops=raw_input.get("max_loops", 5),
        context=context,
        current_time=raw_input.get("current_time", ""),
        user_input=raw_input.get("user_input", ""),
        file_names=raw_input.get("file_names", ""),
        attached_images=attached_images,
    ))
    _log("send_email", "success", f"chars={len(result)}")
    return result


async def _handle_long_memory_update(raw_input: dict) -> str:
    """long_memory をバックグラウンドで即時更新する。

    create_task で起動して即座に返すため、LLM は結果を待たずに次のループへ進める。
    会話履歴（conversation_history）と今回の inner_thoughts を渡すことで、
    通常の記憶更新と同じデータを使いながらミクの判断ヒントも反映する。
    """
    import json as _json

    conversation_history = raw_input.get("conversation_history", "")

    # ai_judgements には main LLM の今回の出力 JSON が入っている（inner_thoughts を含む）
    inner_thoughts_hint = ""
    try:
        ai_judgements = _json.loads(raw_input.get("ai_judgements", "{}"))
        inner_thoughts_hint = ai_judgements.get("inner_thoughts", "")
    except (ValueError, TypeError):
        pass

    _log(
        "long_memory_update",
        "fire",
        f"conv_chars={len(conversation_history)}  hint_chars={len(inner_thoughts_hint)}",
    )
    asyncio.create_task(run_long_memory_patch(conversation_history, inner_thoughts_hint))
    return "長期記憶の即時更新をバックグラウンドで開始しました。"


async def _handle_work_on_music(raw_input: dict) -> str:
    """音楽の作成・編集（song_service への単一入口・Fire&Forget）。

    このシステムの Orchestrator は**ツール名だけを選び引数は渡さない**ため、
    音楽コーディネーターへ渡す instruction は**ここで組み立てる**（ユーザー発話＋ミクの意図＋会話文脈）。
    コーディネーター側は LLM なので、多少冗長でも文脈が多いほうが解決しやすい（指示語の解決材料になる）。
    subprocess を起動して即返す（作成15〜20分は待たない）。実行中の新依頼は Failure(retryable)。
    """
    import json as _json

    from agent.tools.music.work_on_music import WorkOnMusicInput, run_work_on_music

    user_input = str(raw_input.get("user_input", "")).strip()
    inner_thoughts = ""
    try:                                                   # ai_judgements＝main LLM の今回の出力JSON
        inner_thoughts = str(_json.loads(raw_input.get("ai_judgements", "{}")).get("inner_thoughts", "")).strip()
    except (ValueError, TypeError):
        pass
    # 会話ログは**全文**を渡す（実測13,696字＝Flashには誤差。末尾1500字だと実質1ターン分しか届かず、
    # 曲名・アーティスト・URL が落ちて同名異曲で失敗した実例がある・2026-07-26）
    history = str(raw_input.get("conversation_history", "")).strip()

    parts = []
    if user_input:
        parts.append(f"【マスターの依頼】{user_input}")
    if inner_thoughts:
        parts.append(f"【ミクの受け取り方・意図】{inner_thoughts}")
    if history:
        parts.append(f"【会話の流れ（指示語の解決に使う。曲名・アーティスト・URL等はここから拾う）】\n{history}")
    instruction = "\n".join(parts)
    if not instruction:
        return "[エラー] 依頼内容が空（会話文脈も取れなかった）。マスターに作りたい曲を聞き直すこと。"

    # ★**口ずさんだ音が添付されていたら、それも渡す**（2026-09-23）。
    #   引数はミクが書かないので**ここで詰める**（`voice_enroll` と同じ作法）。
    #   複数あると**どれが手本か決められない**ので、選んでもらう（音は中身を見て選べない）。
    audios: list[str] = raw_input.get("attached_audio", []) or []
    if len(audios) > 1:
        return ("[エラー] 音声が複数添付されています。旋律の手本にする音を1つだけ添付して"
                "再試行してください。")
    audio_path = audios[0] if audios else ""
    if audio_path:
        instruction += ("\n【マスターが口ずさんだ音が届いている】旋律の手本として渡してある。"
                        "read_humming で読み取ってから進めること。")

    # 数分で終わる依頼は結果を待つ（最大3分・`work_on_music.WAIT_SECONDS`）ので、イベントループを塞がないよう別スレッドで
    result = await asyncio.to_thread(
        run_work_on_music, WorkOnMusicInput(instruction=instruction, audio_path=audio_path))
    _log("work_on_music", result.status,
         f"instr_chars={len(instruction)} audio={bool(audio_path)} user={user_input[:40]!r}")
    return result.model_dump_json(exclude_none=True)


async def _handle_play_song(raw_input: dict) -> str:
    """自分の作った曲を歌う（完成した曲をブラウザで再生）。

    work_on_music と同じ理由で **Orchestrator は引数を渡さない**ので、どの曲かの手がかりは
    ここで組み立てる（マスターの言葉＋直近の会話）。曲の特定はツール側が台帳と突き合わせて行い、
    決まらなければ Failure＝ミクが候補を挙げて聞き返す。
    """
    from agent.tools.music.play_song import PlaySongInput, run_play_song

    user_input = str(raw_input.get("user_input", "")).strip()
    history = str(raw_input.get("conversation_history", "")).strip()
    # 会話の**末尾**だけを見る（「あの曲」の解決に要るのは直近のやりとり。全文を渡すと
    # 昔の曲名まで拾って候補が増え、無駄な聞き返しになる）
    hint = user_input + "\n" + history[-1200:]

    result = run_play_song(PlaySongInput(hint=hint))
    _log("play_song", result.status, f"user={user_input[:40]!r}")
    return result.model_dump_json(exclude_none=True)


async def _resolve_image_path(raw_input: dict, purpose: str) -> tuple[str | None, str]:
    """添付画像から顔認識に使う1枚を選んでローカルパスを返す。

    画像が1枚のみならそのまま返す。複数枚なら face_image_select サブエージェントで選択する。
    戻り値: (local_path, reason)。local_path が None のときは選択不可の理由が reason に入る。
    """
    image_file_names: list[str] = raw_input.get("image_file_names", [])
    attached_images: list[str] = raw_input.get("attached_images", [])

    if not attached_images:
        return None, "添付画像がありません。カメラ画像または写真を添付して再試行してください。"

    if len(attached_images) == 1:
        return attached_images[0], ""

    # 複数枚: face_image_select サブエージェントで選択
    select_result = await run_face_image_select(FaceImageSelectInput(
        image_file_names=image_file_names or [str(i) for i in range(len(attached_images))],
        attached_images=attached_images,
    ))
    if select_result.selected_filename is None:
        return None, select_result.reason

    name_to_path = dict(zip(image_file_names, attached_images))
    path = name_to_path.get(select_result.selected_filename)
    if path is None:
        # フォールバック: 先頭を使う
        path = attached_images[0]
    return path, ""


async def _handle_face_enroll(raw_input: dict) -> str:
    """マスターの顔を登録する。添付画像から最適な1枚を選んで FaceRecognition サービスへ送る。"""
    _log("face_enroll", "call")
    path, reason = await _resolve_image_path(raw_input, "enroll")
    if path is None:
        _log("face_enroll", "no_image", reason)
        return f"[エラー] {reason}"
    result = run_face_enroll(FaceEnrollInput(image_path=path))
    if isinstance(result, Failure):
        _log("face_enroll", "failure", result.reason)
        return f"[エラー] {result.reason}"
    _log("face_enroll", "success", f"enrolled={result.value.enrolled}")
    return result.summary or f"enrolled={result.value.enrolled} faces_detected={result.value.faces_detected}"


async def _handle_voice_enroll(raw_input: dict) -> str:
    """マスターの声を登録する。添付音声を Recognition サービスへ送る。

    画像と違い選択サブエージェントは使わない（音声は中身を見て選べないため、
    複数添付されたら1つに絞ってもらう）。長さの検証はサービス側が行う。
    """
    _log("voice_enroll", "call")
    audios: list[str] = raw_input.get("attached_audio", [])
    if not audios:
        return ("[エラー] 添付音声がありません。マスターの声だけが入った音声ファイルを"
                "添付して再試行してください。")
    if len(audios) > 1:
        return ("[エラー] 音声が複数添付されています。登録に使う音声を1つだけ添付して"
                "再試行してください。")
    result = run_voice_enroll(VoiceEnrollInput(audio_path=audios[0]))
    if isinstance(result, Failure):
        _log("voice_enroll", "failure", result.reason)
        return f"[エラー] {result.reason}"
    _log("voice_enroll", "success", f"seconds={result.value.seconds}")
    return result.summary or f"enrolled={result.value.enrolled} seconds={result.value.seconds}"


# ── ディスパッチテーブル（静的・動的登録禁止） ─────────────────

async def _handle_recall_summary_experience(raw_input: dict) -> str:
    """経験の記憶を思い出すハンドラ（記憶の2層のうち**上層＝要点**）。

    内部 LLM が会話履歴・ユーザー入力・メインエージェントの直前出力（MainChatOutput JSON 全体・
    purpose / inner_thoughts / requires_tool / tool_name / response 等を含む）を読んで
    検索クエリと category を決定し、VDB long_memory collection をベクトル検索する。
    詳細: agent/subagents/recall_summary_experience.py
    """
    conv_history = raw_input.get("conversation_history", "")
    user_input = raw_input.get("user_input", "")
    # ai_judgements は MainChatOutput JSON 全体（model_dump_json 結果）
    # 主要フィールド: purpose, inner_thoughts, requires_tool, tool_name, response 等
    main_chat_output_json = raw_input.get("ai_judgements", "")
    tool_results = raw_input.get("tool_results", "")

    _log("recall_summary_experience","call",
         f"history={len(conv_history)}c  input={len(user_input)}c")
    result = await run_recall_summary_experience(RecallSummaryExperienceInput(
        conversation_history=conv_history,
        user_input=user_input,
        main_chat_output_json=main_chat_output_json,
        tool_results=tool_results,
    ))
    if result.startswith("[エラー]"):
        _log("recall_summary_experience","failure", result)
        return result
    _log("recall_summary_experience","success", f"result_len={len(result)}c")
    return result


_DISPATCH: dict[str, object] = {
    "self_knowledge": _handle_self_knowledge,
    "recall_detail_conversation": _handle_recall_detail_conversation,
    "recall_summary_experience": _handle_recall_summary_experience,
    "check_news": _handle_check_news,
    "web_search": _handle_web_search,
    "deep_think": _handle_deep_think,
    "miku_note_update": _handle_miku_note_update,
    "send_email": _handle_send_email,
    "long_memory_update": _handle_long_memory_update,
    "face_enroll": _handle_face_enroll,
    "voice_enroll": _handle_voice_enroll,
    "work_on_music": _handle_work_on_music,
    "play_song": _handle_play_song,
}


def _disabled_tool_reason(tool_name: str) -> str | None:
    """settings で無効化されたツールなら理由を返す（有効なら None）。

    プロンプト側（available_tools.md の使用不可注記）をすり抜けて LLM が
    呼び出した場合の防御。判定は agent.config の可用性アクセサと共通。
    """
    from agent.config import (is_email_available, is_song_service_available,
                              is_web_search_available, load_settings)
    settings = load_settings()
    if tool_name == "web_search" and not is_web_search_available(settings):
        return "web_search は設定で無効化されています（web_search.enabled / api_keys.tavily を確認）"
    if tool_name == "send_email" and not is_email_available(settings):
        return "send_email は設定で無効化されています（email.enabled / api_keys.gmail_app_password を確認）"
    if tool_name in ("work_on_music", "play_song") and not is_song_service_available(settings):
        return f"{tool_name} は設定で無効化されています（song_service.enabled / song_service の setup 実施を確認）"
    return None


async def dispatch_tool(tool_name: str, raw_input: dict) -> str:
    """ツール名と入力 dict からツールを呼び出し、結果文字列を返す。

    未知のツール名は "[エラー] 未知のツール: ..." を返す（例外を raise しない）。
    設定で無効化されたツールは呼び出さずにエラー文字列を返す。
    ツールの中で想定外の例外が出たときもエラー文字列にして返す（2026-10-09）——
    raise するとターンごと中断し、ミクは失敗を知らず、依頼も発話も会話ログに残らなかった
    （曲づくりの起動で WinError 206・頼まれたこと自体を忘れていた）。
    """
    print(f"[runner:dispatch] tool={tool_name!r}  available={list(_DISPATCH.keys())}")
    handler = _DISPATCH.get(tool_name)
    if handler is None:
        print(f"[runner:dispatch] unknown tool: {tool_name!r}")
        return f"[エラー] 未知のツール: {tool_name!r}"
    disabled_reason = _disabled_tool_reason(tool_name)
    if disabled_reason is not None:
        print(f"[runner:dispatch] disabled tool called: {tool_name!r}")
        return f"[エラー] {disabled_reason}"
    try:
        return await handler(raw_input)  # type: ignore[operator]
    except Exception as e:
        import traceback
        print(f"[runner:dispatch] {tool_name!r} で想定外のエラー: {type(e).__name__}: {e}")
        traceback.print_exc()
        return (f"[エラー] {tool_name} の実行中に想定外のエラーが起きた（{type(e).__name__}）。"
                "結果は得られておらず、頼まれたことはできていない。うまくいかなかったことをマスターに正直に伝える。")


# ════════════════════════════════════════════════════════════
# ツール結果の要点（ターンをまたいで会話ログに残す・2026-08-26）
# ════════════════════════════════════════════════════════════
#
# ツール結果はこのターンが終わると捨てられる。要点だけを会話ログへ残すことで、
# 「さっき調べたこと」を次のターン以降も参照できるようにする。
# 設計・ツールごとの根拠は docs/plan/tool_result_carryover.md。
#
# 方針: **いつでも取り直せるものは残さない。一度きりの外部情報だけ残す。**

# 中身を残さないツール（いつでも同じものを引ける）。成否だけ記録する。
_SUMMARY_STATUS_ONLY = {"self_knowledge"}

# 生結果が長く、要約 LLM を通すツール（呼び出し頻度は全体の 1% 未満）
_SUMMARY_VIA_LLM = {"recall_detail_conversation", "recall_summary_experience"}

# これ以上長い結果はそのまま残さない（短いツールの想定外の肥大に対する保険）
_SUMMARY_MAX_CHARS = 1_000


def _extract_web_search_conclusion(result: str) -> str:
    """web_search_agent の出力から「### 結論」節を取り出す。

    web_search_agent は「### 結論 + #### 検索結果N」形式で返す。結論部分は
    **既に LLM が書いた要約**なので、ここでは機械的に切り出すだけでよい
    （実測で全体の約 39%・中央 453 字）。見出しが無ければ全体を短く切って返す。
    """
    m = re.search(r"###\s*結論\s*\n(.*?)(?=\n\s*####|\n\s*###|\Z)", result, re.S)
    if m:
        return m.group(1).strip()
    return result.strip()[:_SUMMARY_MAX_CHARS]


def _extract_miku_note_change(result: str) -> str:
    """miku_note_update の結果から「何をどう更新したか」を取り出す。

    `miku_note_compose` が構造化出力で `what_changed` を返すため、それを使う。
    ノート全文（中央 10,854 字）は毎ターン context に入っているので残さない。
    """
    try:
        obj = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return ""
    if isinstance(obj, dict):
        return str(obj.get("what_changed") or "").strip()
    return ""


async def summarize_tool_result(tool_name: str, result: str) -> str:
    """ツール結果の要点を作る。会話ログに残せない／残す意味が無いものは空文字。

    失敗しても会話は止めない（要点が残らないだけ）。
    """
    if not result:
        return ""

    # 失敗も残す（「調べられなかった」は次の判断の材料になる。2026-08-26 実測で確認）
    if result.startswith("[エラー]"):
        return f"{tool_name}: {result[:_SUMMARY_MAX_CHARS]}"

    if tool_name in _SUMMARY_STATUS_ONLY:
        return f"{tool_name}: 読み取れた（{len(result):,}字）"

    if tool_name == "web_search":
        return _extract_web_search_conclusion(result)[:_SUMMARY_MAX_CHARS]

    if tool_name == "miku_note_update":
        return _extract_miku_note_change(result)[:_SUMMARY_MAX_CHARS]

    if tool_name in _SUMMARY_VIA_LLM:
        from agent.subagents.tool_result_summarize import (
            ToolResultSummarizeInput,
            run_tool_result_summarize,
        )
        summary = await run_tool_result_summarize(
            ToolResultSummarizeInput(tool_name=tool_name, result=result))
        return summary[:_SUMMARY_MAX_CHARS]

    # 残りは短いツール（check_news / face_* / long_memory_update / work_on_music 等）。
    # そのまま残せるが、想定外に長ければ切る
    return result.strip()[:_SUMMARY_MAX_CHARS]


# ════════════════════════════════════════════════════════════
# ループ層（Phase 5）— 統合応答 ↔ ツール呼び出しの往復ストリーミング
# ════════════════════════════════════════════════════════════


@dataclass
class LoopAborted:
    """考え中の追加発話（畳み込み）により、生成途中で中断されたループ。

    partial_inner_thoughts: 中断時点までに生成できていた inner_thoughts（空のこともある）。
    partial_response: 中断までに ResponseDelta として**既に TTS へ流れた**分の response。
        空でなければ音声としては発話されうるため、会話ログにも text として残す
        （「言った事実は消さない」原則）。空なら inner_thoughts のみの state になる。
    """
    partial_inner_thoughts: str = ""
    partial_response: str = ""


@dataclass
class LoopStarted:
    """新しいループ反復が開始された。"""
    loop_counter: int


@dataclass
class ResponseDelta:
    """統合応答 LLM の `response` フィールドの差分テキスト。

    Gateway はこれを Browser / TTS へ即座に流すことで、
    ループの最終応答だけでなく途中ループの一時応答もリアルタイム発話できる。
    """
    text: str


@dataclass
class VoiceToneDecided:
    """話す声の調子が確定した。**response の差分より先に流れる**。

    TTS のストリーマは発話の前に事前起動されているため、
    確定した時点で渡さないと最初のチャンクの加工に間に合わない。
    """
    tone: str


@dataclass
class IterationComplete:
    """1 ループの統合応答 LLM が完了し、structured_output が確定した。"""
    output: MainChatOutput


@dataclass
class ToolStarted:
    """ツール呼び出しを開始した（Browser の「検索中…」UI 表示など用）。"""
    tool_name: str
    raw_input: dict


@dataclass
class ToolCompleted:
    """ツール呼び出しが完了し、結果文字列が得られた。

    summary: 会話ログへ残す**要点**（生の結果ではない）。ターンをまたぐと結果は
        失われるため、これだけが後から参照できる。残す意味が無いツールでは空文字
        （設計: docs/plan/tool_result_carryover.md）。
    """
    tool_name: str
    result: str
    summary: str = ""


@dataclass
class LoopFinished:
    """ループ全体が終了した（tool_name=null / requires_tool=False / max_loops 到達）。"""
    final_output: MainChatOutput
    state: ConversationState
    reason: str  # "tool_name_null" | "max_loops_reached"


MainChatStreamEvent = (
    LoopStarted | ResponseDelta | VoiceToneDecided | IterationComplete
    | ToolStarted | ToolCompleted | LoopAborted | LoopFinished
)


def _format_tool_annotation(
    loop_counter: int,
    llm_output: MainChatOutput,
    tool_name: str,
    tool_result: str,
) -> str:
    """Dify「ツールアノテーション」テンプレを再現する。

    Dify テンプレ:
      ## ループN回目のエージェント実行結果
      ### ループN回目のエージェント一時応答・判断
      {llm_response}
      ### ループN回目のツール実行結果
      {
      ツール名: {tool_name}
      ツールから得られた情報: {result}
      }

    ★**見出しと中身を対応させる**（2026-09-27）。それまでは「発話済み応答」という見出しの下に、出力の**全欄を平らな形の
      JSON**（内心・タイムアウト・次の自発話しかけ・気分の更新まで）で載せていた。見出しは「もう話したこと」なのに
      中身は判断の全部で、しかも実際の出力（4群の入れ子）と形が違う——**平らな形の手本が入力の中で増える**。
      2026-09-26 夜、内心の欄の中で後ろの欄（タイムアウト・次の自発話しかけ）を読み上げ続ける崩れが4ターン続いた
      （原因とは言い切れないが、取り違えの種）。いまは「そのときの判断（目的・内心・選んだツール）」と「発話済み応答」を
      見出しごとに分け、次の周回に要らない記録の欄（タイムアウト等）は載せない。
    """
    said = llm_output.response if llm_output.response else "（このループでは発話していない）"
    return (
        f"## ループ{loop_counter}回目のエージェント実行結果\n\n"
        f"### ループ{loop_counter}回目の判断（そのときの目的・内心・選んだツール）\n"
        f"目的: {llm_output.purpose}\n"
        f"内心: {llm_output.inner_thoughts}\n"
        f"選んだツール: {tool_name}\n\n"
        f"### ループ{loop_counter}回目の発話済み応答\n"
        f"※ この内容はすでにユーザーに発話・送信されています。次のループで同じ内容を繰り返さないでください。\n"
        f"{said}\n\n"
        f"### ループ{loop_counter}回目のツール実行結果\n"
        "{\n"
        f"ツール名: {tool_name}\n"
        f"ツールから得られた情報: {tool_result}\n"
        "}"
    )


def _build_subagent_raw_input(
    main_input: MainChatInput,
    llm_output: MainChatOutput,
    state: ConversationState,
) -> dict:
    """ツールのハンドラに渡す raw_input を組み立てる。

    各 subagent（web_search / recall_detail_conversation / deep_think / miku_note_compose / email_compose）が
    受け取り得るフィールドを最大限揃えて渡す。subagent 側は必要な分だけ取り出す。

    ★**名前は subagent だが、渡る先は subagent だけではない**（2026-09-23 に補足）。
      `dispatch_tool` は**すべてのツール**へこの dict を渡すので、
      `voice_enroll` の `attached_audio`、`work_on_music` の添付音声もここを通る。
      **「subagent 用だから関係ない」と読み飛ばすと、添付が届かない理由が分からなくなる。**
    """
    return {
        # ループ状態
        "loop_counter": state.loop_counter,
        "max_loops": main_input.max_loops,
        # ユーザー入力・状況
        "user_input": main_input.user_input,
        "current_time": main_input.current_time,
        "file_names": main_input.file_names,
        "attached_images": main_input.attached_images,
        "image_file_names": main_input.image_file_names,
        "attached_audio": main_input.attached_audio,
        # 文脈
        "context": main_input.context,
        "conversation_history": state.conversation_history,
        "ai_judgements": llm_output.model_dump_json(indent=2),
        "tool_results": state.tool_result,  # web_search / recall_detail_conversation 互換
        "tool_result": state.tool_result,    # 一部互換
        "conversation_situation": state.conversation_situation or main_input.conversation_situation,
        # base_prompt.md（ミクのキャラ・人格設定本文）。subagent の `------\n基本のプロンプト\n` セクションに展開される。
        # Dify でも同じ base_prompt 変数が subagent 入力に渡される。
        "base_prompt": _BASE_PROMPT,
    }


def _build_loop_input(input_: MainChatInput, state: ConversationState) -> MainChatInput:
    """同一ターン内の各ループ反復で main_agent に渡す MainChatInput を組み立てる。

    state の loop_counter / tool_result / conversation_situation を流し込んで返す。
    """
    return input_.model_copy(update={
        "loop_counter": state.loop_counter,
        "tool_result": state.tool_result,
        "conversation_situation": state.conversation_situation or input_.conversation_situation,
    })


def _append_additions_to_input(
    input_: MainChatInput, additions: list[tuple[str, str]]
) -> MainChatInput:
    """考え中の追加発話を user_input へ時系列追記した MainChatInput を返す（P2 畳み込み）。"""
    lines = "".join(f"\n\n[{t} 追加発話] {text}" for t, text in additions)
    return input_.model_copy(update={"user_input": (input_.user_input or "") + lines})


#: 発話中の待機を打ち切るハードリミット（秒）。切断・通知の取りこぼしで
#: 「話している」まま張り付いた場合でも、必ずここで確定へ進む
_MAX_SPEECH_WAIT_SEC = 30.0
#: 待機ループのポーリング間隔（秒）。イベントループは塞がない
_SPEECH_POLL_SEC = 0.1


async def _wait_for_additional_speech(turn_handle) -> bool:
    """ユーザーが話し終えて追加発話が届くまで待つ（2026-08-11）。届いたら True。

    「話している最中に確定させない」ための待機。2段構えで打ち切る:
      - 話している間は待ち続ける（上限 _MAX_SPEECH_WAIT_SEC）
      - 話し終えてからは wait_timeout_sec だけ待つ（ブラウザ側の送信カウントダウン分。
        既定は settings の stt.auto_send.delay_seconds）

    待たずに再生成すると同じ入力で作り直すだけで無駄になるため、中断直後にも使う。
    """
    import time as _time

    if turn_handle is None:
        return False
    if turn_handle.has_pending():
        return True
    if not turn_handle.is_user_speaking():
        return False

    print("[STREAM] ユーザーが発話中 → 追加発話の到着を待つ")
    hard_deadline = _time.perf_counter() + _MAX_SPEECH_WAIT_SEC
    settled_at: float | None = None
    while _time.perf_counter() < hard_deadline:
        if turn_handle.has_pending():
            return True
        if turn_handle.is_user_speaking():
            settled_at = None          # まだ話している
        else:
            now = _time.perf_counter()
            if settled_at is None:
                settled_at = now       # いま話し終えた
            elif now - settled_at >= turn_handle.wait_timeout_sec:
                print("[STREAM] 発話終了後も追加発話が届かない → 確定へ進む")
                return False
        await asyncio.sleep(_SPEECH_POLL_SEC)
    print(f"[STREAM] 発話中の待機が上限 {_MAX_SPEECH_WAIT_SEC:.0f}秒に達した → 確定へ進む")
    return False


async def run_main_loop_stream(
    input_: MainChatInput,
    state: ConversationState | None = None,
    turn_handle=None,
) -> AsyncIterator[MainChatStreamEvent]:
    """Dify「ループ」ノード相当: 統合応答 ↔ ツール呼び出しを往復ストリーミング。

    各ループの response フィールドを ResponseDelta として逐次 yield。
    ツール呼び出しの開始・終了も yield。
    最後に LoopFinished を yield してジェネレータ終了。

    state を渡さなければ新規作成。loop_counter は 0 から開始してループ毎に +1。

    turn_handle（agent/turn_handle.py の TurnHandle）を渡すと「考え中の追加発話」の
    畳み込みが有効になる（P2・2026-08-01）:
      - LLM ストリーミング中に abort_stream が立ったら生成を中断（LoopAborted を yield）し、
        追加発話を user_input へ時系列追記して同一ターンの次ループとして再実行する。
        `state.tool_result` の蓄積・発話済み中間発話は保持される（非破壊）
      - ツール実行中はチェックしない（ツールは完走させ、結果と追加発話の両方を次ループへ）
      - 最終応答の確定直前に close_if_idle() でターンを閉じる（gateway の注入とロックで排他。
        閉じ損ねた追加発話があれば確定せずに再ループする）
      - 中断されたループは loop_counter を消費しない（畳み込みで max_loops を使い切らない）

    デバッグ用に主要区間の経過時間（呼び出し開始からの ms）を [TIMING] として出力する。
    """
    import time as _time
    _t0 = _time.perf_counter()
    def _ms() -> int:
        return int((_time.perf_counter() - _t0) * 1000)

    state = state or ConversationState()

    final_output: MainChatOutput | None = None
    finish_reason: str = "max_loops_reached"

    print(f"[TIMING] runner.run_main_loop_stream 開始: {_ms()}ms")

    while state.loop_counter < input_.max_loops:
        # 畳み込み: ツール実行中などに溜まった追加発話をループ境界で取り込む
        if turn_handle is not None:
            _additions = turn_handle.drain()
            if _additions:
                input_ = _append_additions_to_input(input_, _additions)
                print(f"[STREAM] 追加発話 {len(_additions)} 件を取り込み（ループ境界）")

        state.loop_counter += 1
        loop_n = state.loop_counter
        print(f"[TIMING] loop{loop_n} 開始: {_ms()}ms")
        yield LoopStarted(loop_counter=loop_n)

        # 統合応答 LLM への入力組み立て
        loop_input = _build_loop_input(input_, state)
        marked_instructions = _build_marked_instructions(loop_input)
        instructions = strip_markers(marked_instructions)
        file_names = loop_input.file_names or derive_file_names(loop_input.attached_images)

        # 明示キャッシュ（2026-08-30）: プロンプトの固定部（記憶・過去会話・miku_note）を
        # Google 側に置いて使い回す。**作れなければ従来どおり全文を送る**ので、
        # 失敗しても会話は止まらない。
        #
        # キャッシュを使うリクエストには instructions（system_instruction）を載せられない
        # 制約があるため、キャッシュ側へ入れて**こちらは空にする**。
        # 出力スキーマは NativeOutput（response_json_schema）なので tools は送られない。
        cache_head, cache_tail = _build_user_prompt_parts(loop_input, file_names=file_names)
        cached_name = None
        if cache_head:
            cached_name = context_cache.ensure(instructions, cache_head, _cache_model_id())
        if cached_name:
            user_prompt_text = cache_tail
            instructions = ""                     # キャッシュ側に入っている
            context_cache.touch()
        else:
            user_prompt_text = cache_head + cache_tail

        # 相手が話している間に立てた方針があれば、参考として添える（2026-08-30）。
        # **ループ1だけ**——ループ2以降はツール結果を見て考え直す場面なので、
        # 話す前の方針を持ち込むと古い前提に引きずられる。
        # 無ければ何も足さず、従来どおり1段階で作る（壊れない）
        if loop_n == 1:
            got = plan_stage.take()
            if got:
                plan, heard = got
                user_prompt_text += plan_stage.build_plan_block(plan, heard)
                print(f"[plan_stage] 方針を段階2へ渡した（{len(heard)}字まで聞いた時点）")

        attachments = [
            *to_binary_contents(loop_input.attached_images),
            *to_media_binary_contents(loop_input.attached_media),
        ]
        if attachments:
            user_prompt: str | list = [user_prompt_text, *attachments]
        else:
            user_prompt = user_prompt_text

        agent_log(
            "main_chat_request",
            "ミクメイン応答LLMへの入力（runner → main_agent / Gemini Flash）",
            input={
                "instructions": marked_instructions,
                "user_prompt": user_prompt_text,
                "attached_images": loop_input.attached_images,
                "attached_media": loop_input.attached_media,
            },
            meta={"loop": loop_n},
        )

        # ストリーミングで partial output を読み、response フィールドの差分を yield
        prev_response_len = 0
        model_settings = _loop_model_settings(loop_n, cached_name)
        print(f"[TIMING] loop{loop_n} LLM run_stream 呼出直前: {_ms()}ms "
              f"(prompt={len(user_prompt_text):,}字, instructions={len(instructions):,}字"
              + (", thinking=絞る" if model_settings is not None else "") + ")")
        _t_llm_call = _time.perf_counter()
        first_ttft_logged = False
        first_response_logged = False
        aborted = False
        last_partial = None
        tone_sent = False
        async with main_agent.run_stream(user_prompt, instructions=instructions,
                                         model_settings=model_settings) as run:
            async for partial in run.stream_output(debounce_by=None):
                # 畳み込み: 追加発話が来ていたら生成を中断（自発チェックして自分で抜ける）
                if turn_handle is not None and turn_handle.abort_stream.is_set():
                    aborted = True
                    print(f"[STREAM] loop{loop_n} 追加発話により生成を中断: {_ms()}ms")
                    break
                last_partial = partial
                if not first_ttft_logged:
                    first_ttft_logged = True
                    ttft_ms = int((_time.perf_counter() - _t_llm_call) * 1000)
                    print(f"[TIMING] loop{loop_n} LLM 初回 partial（TTFT）: {_ms()}ms "
                          f"(LLM 呼出から +{ttft_ms}ms)")
                # 声の調子は speech 群の先頭なので response より先に確定する。
                # **一度だけ**流す（同じ値が毎デルタ届くため）
                if not tone_sent:
                    tone = partial_voice_tone_of(partial)
                    if tone:
                        tone_sent = True
                        print(f"[TIMING] loop{loop_n} 声の調子が確定: {tone} ({_ms()}ms)")
                        yield VoiceToneDecided(tone=tone)
                # partial は出力の途中状態。response が未確定の場合は空文字。
                # 出力は入れ子（speech.response）だが、**取り出しは partial_response_of が
                # 吸収する**ので、ここから先は形が変わっても影響しない（2026-08-30）
                response_text = partial_response_of(partial)
                if response_text and response_text != "null":
                    if not first_response_logged:
                        first_response_logged = True
                        print(f"[TIMING] loop{loop_n} LLM 初回 response delta: {_ms()}ms")
                    delta = response_text[prev_response_len:]
                    if delta:
                        yield ResponseDelta(text=delta)
                        prev_response_len = len(response_text)
            if aborted:
                # break だけでは stream_output() の async generator を閉じたことにならず、
                # 後始末が Python の GC（asyncgen finalizer）任せになる。GC が動くタイミングは
                # 不定で、次ループの run_stream() と重なると
                # 「aclose(): asynchronous generator is already running」を Task exception
                # として出していた（2026-09-21 実機ログで確認・pydantic_ai issue #5615/#5756 と同型）。
                # pydantic_ai 1.107.6 で追加された正式な中断 API `run.cancel()` を使い、
                # 後始末を GC 任せにせずここで確定させる（[decisions.md 2026-09-22]）。
                with contextlib.suppress(Exception):
                    await run.cancel()
                # 中断しても、そこまでの生成は課金される。取れた分を記録する（応答の終わりまで来ていないと 0 のこともある）
                log_llm_usage(_MAIN_MODEL_LABEL, run, label=f"loop{loop_n} main_agent usage（中断）", kind="main")
            if not aborted:
                raw = await run.get_output()
                # **ここで従来のフラット形へ直す**。これより下流（会話ログ・配信・TTS・記憶）は
                # 形式も中身も現行のまま。入れ子は LLM とのやり取りの中だけで完結させる
                output: MainChatOutput = raw.to_flat() if hasattr(raw, "to_flat") else raw

        if aborted:
            # 中断ループ: inner_thoughts（＋TTS へ流れてしまった部分 response）を記録用に
            # yield し、loop_counter を返上して再ループする（次周のループ境界 drain() が
            # 追加発話を取り込む）。部分 response があれば「発話済み」注記を次ループへ渡す
            # 入れ子（thinking.inner_thoughts / speech.response）でも取れるようにする
            partial_thoughts = ""
            if last_partial is not None:
                th = getattr(last_partial, "thinking", None)
                partial_thoughts = (getattr(th, "inner_thoughts", None) if th is not None
                                    else getattr(last_partial, "inner_thoughts", None)) or ""
            partial_response = ""
            if prev_response_len > 0 and last_partial is not None:
                partial_response = partial_response_of(last_partial)[:prev_response_len]
            # ★**音が鳴り始めていなければ、その部分応答は「無かったこと」にする**（2026-09-22 本人指示）。
            # 半二重なので、声で割り込めた＝その時点で音が鳴っていない。TTS へ送っただけの
            # テキストは Fish の合成待ち・ブラウザの先読みバッファ（最大4秒）に留まったまま
            # 強制停止で捨てられるので、**マスターには一言も届いていない**。
            # それを「発話済み」と申し送ると、聞いていない話を前提に次の発言が組み立てられる
            # （実機で違和感として報告された）。記録にも残さず、次ループで作り直させる
            audio_started = bool(turn_handle is not None and turn_handle.audio_started)
            if partial_response and not audio_started:
                print(f"[STREAM] loop{loop_n} 中断時に音声未開始 → 部分応答 {len(partial_response)}字 を破棄")
                partial_response = ""
            if partial_response:
                note = (f"[畳み込み注記] ループ{loop_n}は追加発話により中断されたが、"
                        f"「{partial_response}」までは既にユーザーへ発話済み。"
                        "重複を避け、続きとして自然に応答すること。")
                state.tool_result = (
                    f"{state.tool_result}\n\n{note}" if state.tool_result else note
                )
            yield LoopAborted(partial_inner_thoughts=partial_thoughts,
                              partial_response=partial_response)
            state.loop_counter -= 1
            # 話し始めを検知して止めた場合、追加発話はまだ届いていない。
            # 届くのを待ってから再ループする（待たずに作り直すと同じ入力で再生成するだけ）
            await _wait_for_additional_speech(turn_handle)
            continue

        # 使用量（取り出しと書き先は agent_logger 1か所。失敗しても止めない）
        log_llm_usage(_MAIN_MODEL_LABEL, run, label=f"loop{loop_n} main_agent usage", kind="main")

        llm_total_ms = int((_time.perf_counter() - _t_llm_call) * 1000)
        resp_len = len(output.response or "")
        print(f"[TIMING] loop{loop_n} LLM 応答完了: {_ms()}ms "
              f"(LLM 全体 {llm_total_ms}ms, response={resp_len}字, "
              f"requires_tool={output.requires_tool}, tool_name={output.tool_name!r})")

        final_output = output
        yield IterationComplete(output=output)

        # ループ終了条件: tool_name が null（または空文字）か requires_tool=False
        tool_name = (output.tool_name or "").strip()
        if not output.requires_tool or tool_name in ("", "null"):
            # 畳み込み: 確定直前にターンを閉じる（gateway の注入とロックで排他）。
            # 生成完了後〜ここまでに追加発話が滑り込んでいた場合、この応答のデルタは
            # 既に TTS へ流れ切っている（＝発話される）ため**捨てずに中間発話として確定**し、
            # 「発話済み」注記を次ループへ渡して続きを生成させる。
            # ※この応答は直前の IterationComplete で objects に記録済み（text 保持）
            #
            # 2026-08-11: 確定の直前にユーザーが話している最中なら、その内容が届くまで
            # 閉じずに待つ（従来は「既に届いた追加発話」しか見ておらず、話している最中でも
            # 確定して返事を始めていた）。待った結果は下の close_if_idle() が判定する
            await _wait_for_additional_speech(turn_handle)
            if turn_handle is not None and not turn_handle.close_if_idle():
                print("[STREAM] 確定直前に追加発話を検出 → 応答を中間発話として畳み込み再ループ")
                note = (f"[畳み込み注記] あなたはループ{loop_n}で既に"
                        f"「{output.response or ''}」と発話済み。"
                        "重複を避け、追加発話を踏まえた続きとして応答すること。")
                state.tool_result = (
                    f"{state.tool_result}\n\n{note}" if state.tool_result else note
                )
                continue
            finish_reason = "tool_name_null"
            break

        # ツール呼び出し
        raw_input = _build_subagent_raw_input(input_, output, state)
        print(f"[TIMING] loop{loop_n} tool dispatch 開始: {_ms()}ms (tool={tool_name!r})")
        _t_tool = _time.perf_counter()
        yield ToolStarted(tool_name=tool_name, raw_input=raw_input)
        tool_result_str = await dispatch_tool(tool_name, raw_input)
        tool_ms = int((_time.perf_counter() - _t_tool) * 1000)
        print(f"[TIMING] loop{loop_n} tool dispatch 完了: {_ms()}ms "
              f"(所要 {tool_ms}ms, result={len(tool_result_str):,}字)")
        # ターンをまたいで残す要点を作る（失敗しても会話は止めない）
        tool_summary = await summarize_tool_result(tool_name, tool_result_str)
        if tool_summary:
            print(f"[runner:{tool_name}:summary] {len(tool_summary):,}字")
        yield ToolCompleted(
            tool_name=tool_name, result=tool_result_str, summary=tool_summary)

        # ツール結果を annotation 形式で state.tool_result に蓄積
        annotation = _format_tool_annotation(
            loop_counter=state.loop_counter,
            llm_output=output,
            tool_name=tool_name,
            tool_result=tool_result_str,
        )
        state.tool_result = (
            f"{state.tool_result}\n\n{annotation}" if state.tool_result else annotation
        )

    # 畳み込み: max_loops 枯渇でここに来た場合はターンが未クローズのことがある。
    # 確実に閉じてから残りの追加発話を回収する（応答はできないが KV 記録には残す）
    if turn_handle is not None:
        turn_handle.force_close()
        leftover = turn_handle.drain()
        if leftover:
            print(f"[STREAM] max_loops 枯渇時に未処理の追加発話 {len(leftover)} 件を記録用に回収")

    assert final_output is not None, "ループが 1 回も実行されていない（max_loops < 1?）"
    # state を MainChatOutput の値で更新（次ターンに引き継ぐ）
    state.inactivity_timeout = final_output.inactivity_timeout

    print(f"[TIMING] runner.run_main_loop_stream 終了: {_ms()}ms "
          f"(reason={finish_reason}, total loops={state.loop_counter})")
    yield LoopFinished(final_output=final_output, state=state, reason=finish_reason)
