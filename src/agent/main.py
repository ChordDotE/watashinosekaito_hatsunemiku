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

"""統合応答ノード（Orchestrator）。

Dify「統合応答ノード（id=1763668287653）」を再現する。
gemini-3-flash-preview に Dify と同じ system+user プロンプト構造を渡し、
structured_output として MainChatOutput（10 フィールド）を返す。

Dify「プロンプト結合ノード（id=1764431466242）」が会話変数を結合した結果を、
本モジュールでは prompts/ 直下の md ファイルから動的に読み込んで再現する。

system プロンプト構造:
  base_prompt + task_instruction + conversation_situation +
  available_tools + attach_file_instruction（すべて静的・毎ターン同一 → キャッシュ対象）
  ※ output_format は 2026-08-11 に廃止（構造化出力のスキーマと二重管理だったため）

user プロンプト構造（並び順の理由は _build_user_prompt のコメント参照）:
  1. 参考情報（context + ループ内ツール結果）… 静的側を先頭に置きキャッシュに乗せる
  2. 現在の時刻 / ループに関する指示        … 毎ターン・毎ループ変わる動的部分
  3. ユーザー側の新たな入力 / 添付ファイル名

戻り値: MainChatOutput（成功）or "[エラー] ..." 文字列（失敗）
"""
from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent, NativeOutput

from agent.agent_logger import record_llm_usage, mark_prompt, strip_markers
from agent.context_cache import CACHEABLE_TAIL_MARK
from agent.config import (
    configure_env_from_settings,
    get_pydantic_ai_flash_model,
    load_settings,
)
from agent.io_helpers import derive_file_names, to_binary_contents, to_media_binary_contents
from agent.schemas import (
    MainChatInput, MainChatOutput, NestedChatOutput,
)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_settings = load_settings()
configure_env_from_settings(_settings)


def _read(name: str) -> str:
    """prompts/ 直下の md ファイルを読み込む。"""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# ★道具の説明（available_tools.md）は**設定で文面を切り替えない**（2026-10-03 本人判断）。
#   以前は `<!-- if:tool -->…<!-- else -->` / `<!-- switch:music -->` で、起動時の設定に応じて
#   「使える／使えない」「CPU だけ／GPU あり」の文面を差し替えていた（`_filter_available_tools`）。
#   使える・使えないは `<system_state>`（`agent_main._format_tool_availability`）の1か所に載せ、
#   道具の説明は「何ができる道具か」だけにした。

# Dify 会話変数相当の静的プロンプト（プロセス起動時に一度だけ読み込む）
_BASE_PROMPT = _read("base_prompt.md")
_TASK_INSTRUCTION = _read("task_instruction.md")
_AVAILABLE_TOOLS = _read("available_tools.md")
_ATTACH_FILE_INSTRUCTION = _read("attach_file_instruction.md")


def _build_loop_instruction(input_: MainChatInput) -> str:
    """ループ状態テキストを生成する（instructions から除外して静的に保つ。
    user_prompt 内では context の**後ろ**に置く — 並び順の理由は _build_user_prompt 参照）。"""
    return (
        "------\n"
        "ループに関する指示\n\n"
        "# 現在の処理状況（システムステータス）\n"
        "この処理は解決のためにループ実行されています。あなたは今、そのループの中にいます。\n"
        f"- 現在のループ実行回数: {input_.loop_counter} 回目\n"
        f"- 最大許容ループ回数: {input_.max_loops} 回まで\n"
        "- 【重要】一般的なエージェントフレームワークでは中間ループの思考・応答はユーザーに見せない挙動が多いですが、"
        "このシステムは**ループ1以降で `response` に設定した内容がすでにユーザーに発話・送信されています**。"
        "ループ2以降では、前のループで発話した内容と重複・類似した表現を避け、ツール結果に基づく新しい情報のみを追加してください。\n"
        "- 4回目以上の場合: 無限ループを防ぐため、ツール呼び出しを強制的に諦め（action の requires_tool を false、"
        "tool_name を null にする）、現時点で持っている情報だけで最善の回答を作成してください。\n"
        "------"
    )


def _build_marked_instructions(input_: MainChatInput) -> str:
    """マーカー付き instructions を組み立てる（ログ用）。LLM に渡す前に strip_markers() で除去する。

    Dify「プロンプト結合ノード」+ 統合応答ノード system 部を再現する。
    各セクションに <<<name:START>>> / <<<name:END>>> マーカーを付与し、
    ログ上で grep によるセクション特定を可能にする。

    current_time / loop_instruction は user_prompt 側（context の後ろ）に置く。
    instructions を毎ターン同じ文字列に保つことで Gemini Implicit Caching を有効にする
    （user_prompt 内の並び順は _build_user_prompt のコメント参照）。
    """
    sections: list[str] = [
        mark_prompt("base_prompt", _BASE_PROMPT),
        mark_prompt("task_instruction", _TASK_INSTRUCTION),
        # output_format は 2026-08-11 に廃止した。構造化出力（response_schema）が
        # 「JSON のみ・指定外フィールド禁止・全キー出力」を保証するため、プロンプト側の
        # 指示は Dify 時代の名残で二重管理になっていた。各フィールドの説明は schemas.py の
        # description、出力例はトップレベル description へ移設（検証: test/output_schema_ab/）
        mark_prompt("conversation_situation", input_.conversation_situation) if input_.conversation_situation.strip() else "",
        mark_prompt("available_tools", _AVAILABLE_TOOLS),
        mark_prompt("attach_file_instruction", _ATTACH_FILE_INSTRUCTION),
    ]
    return "\n\n".join(s for s in sections if s.strip())


def _build_instructions(input_: MainChatInput) -> str:
    """Dify「プロンプト結合ノード」+ 統合応答ノード system 部を再現する。

    pydantic_ai 0.0.x 以降では Agent.run(instructions=...) で system 相当を渡せる。
    """
    return strip_markers(_build_marked_instructions(input_))


def _build_user_prompt(input_: MainChatInput, file_names: str) -> str:
    """Dify 統合応答ノード user 部（user 1 + user 2）を再現する。

    Dify の prompt_template はテンプレート文字列なので、各セクションは
    値の有無に関わらず常に描画される（値が空なら空白行になるだけ）。
    本実装も同じ挙動にすることで、LLM の入力構造を Dify と完全に一致させる。

    file_names は input_.file_names を優先、空なら attached_images から派生する。
    """
    # Dify「コンテキスト結合」テンプレート互換: 参考情報ブロック内に tool_result を含める。
    # ループ層が state.tool_result に annotation を蓄積し、
    # _build_loop_input で MainChatInput.tool_result に流し込んでくる。
    # 1 ループ目は tool_result が空文字 → ヘッダ末尾の注記が「(空白の場合は今回が1ループ目)」。
    #
    # 【並び順とキャッシュ（2026-07-30 変更）】毎ターン/毎ループ変わる要素（現在の時刻・
    # ループ指示）は user_prompt の**末尾側**（ユーザー入力の直前）に置く。
    # かつては先頭に置いていたが、それではプレフィックスが毎ターン変わり、後続の巨大な
    # context（数万字）が Gemini Implicit Caching に一切乗らなかった（実測: ヒットが
    # instructions ぶんの約8.2k tokens 止まり）。静的な context を先頭に置くことで
    # 「instructions + context」がキャッシュ対象になる。tool_result はブロック末尾への
    # 追記型なので、ループをまたいでも context 部分のプレフィックスは崩れない。
    head, tail = build_user_prompt_parts(input_, file_names)
    return head + tail


def build_user_prompt_parts(input_: MainChatInput, file_names: str) -> tuple[str, str]:
    """user_prompt を「キャッシュに載せる前半」と「毎回送る後半」に割って返す。

    2026-08-30 追加。**結合すれば従来と完全に同じ文字列**になるので、
    キャッシュを使わない経路の挙動は変わらない。

    境界は `agent_main._build_context` が埋めた `CACHEABLE_TAIL_MARK`。
    **マークは両方から取り除く**（LLM には渡さない）。

    | | 中身 | 更新頻度 |
    |---|---|---|
    | 前半 | 記憶スキーマ・記憶データ・直近の過去会話・miku_note | 日次〜1日数回 |
    | 後半 | システム状態・this_conversation・tool_result・時刻・現在心境・入力 | 毎ターン |

    マークが無い（旧形式の context など）ときは、**全部を後半に入れて前半を空にする**。
    キャッシュは作られず従来どおり動く。
    """
    ctx = input_.context
    if CACHEABLE_TAIL_MARK in ctx:
        ctx_head, ctx_tail = ctx.split(CACHEABLE_TAIL_MARK, 1)
    else:
        ctx_head, ctx_tail = "", ctx

    inner_tail = (
        f"{ctx_tail}\n\n"
        "# これまでに今回の返答1ターンにおいてループ内であなたが実行して得られた判断・発話済み応答・ツール結果\n"
        "# （発話済み応答はすでにユーザーに発話・送信されています。同じ内容を繰り返すことは禁止です）\n"
        "# （空白の場合は今回が1ループ目）\n"
        f"{input_.tool_result}"
    )
    # 前半は「参考情報ブロックの開き」＋ context の固定部まで
    head = (
        "# 参考情報（返答の中身の材料。話し方の手本ではない）\n"
        "{\n"
        f"{ctx_head}"
    ) if ctx_head else ""
    # 前半が空なら、開きも後半側へ回す（結合結果は同じ）
    tail_open = "" if ctx_head else "# 参考情報（返答の中身の材料。話し方の手本ではない）\n{\n"
    user_1_tail = f"{tail_open}{inner_tail}\n}}"

    dynamic_parts = []
    if input_.current_time:
        dynamic_parts.append(f"# 現在の時刻\n{input_.current_time}")
    # 現在心境（いまの気分・気になっていること）は毎ターン変わるので動的ブロックに置く。
    # context 側（先頭）へ入れるとプレフィックスが毎ターン変わり、キャッシュが崩れる。
    # 記録の container はタグで囲み、属性に見え方を書く（agent_main._build_context と同じ扱い）
    if input_.current_mind:
        dynamic_parts.append(
            '<current_mind visible="always" nature="いまの自分の状態。時間とともに薄れる">\n'
            f"{input_.current_mind}\n"
            "</current_mind>"
        )
    dynamic_parts.append(_build_loop_instruction(input_))
    dynamic_block = "\n\n".join(dynamic_parts)

    user_2 = (
        f"# ユーザー側の新たな入力・発言（会話においてはこれに返答すること）\n{input_.user_input}\n\n"
        f"# 今回添付されたファイルの名称\n{file_names}\n"
        "※camera_0～camera_2の画像は、ミクの「目」から見た画像で、"
        "その他のファイルはユーザーが意図的に添付した画像・ファイルなので注意すること"
    )

    return head, f"{user_1_tail}\n\n{dynamic_block}\n\n{user_2}"


# 出力の形は **NativeOutput**（`response_json_schema`）で強制する（2026-08-30 変更）。
#
# **なぜ既定の tool モードをやめたか**: 明示キャッシュ（`cachedContents`）は
# `system_instruction` / `tools` / `tool_config` をリクエストに載せられない（400）。
# pydantic_ai の既定は構造化出力を `tools` で実現するため、そのままではキャッシュを
# 一切使えなかった。NativeOutput は同じスキーマを `response_json_schema` として送るので、
# **API 側で形が強制される点は変わらないまま**、キャッシュと併用できる
# （プロンプト本文に書く PromptedOutput とは別物。あちらは形が保証されない）。
#
# 実測（8ターン×3回=48リクエスト・2026-08-30）:
#   構造の崩れ  既定 0/24 ・ Native 0/24
#   ツール判断  モード間で割れたターン **0/8**
#   自由行動の response=null  既定 12/12 ・ Native 12/12（明確な規則は両者とも守る）
#   所要 中央値 11.26s → 7.98s ／ 費用 中央値 $0.0247 → $0.0226
#
# **変わること**: ツールを呼ぶとき、既定は稀に無言だったが（null 3/24）、Native は
# 必ず一言添える（0/24）。description の「原則として毎回発話する」に寄った形で、
# 自由行動のような「null とする」規則は崩れない。検証は test/response_planning_ab/。
# 出力は**入れ子4群**（thinking / action / speech / state）にする（2026-08-30）。
# 13 フィールドに性質の違うものが並列に並んでいて、声のトーンを足すと 15 個になり
# 置き場所も自明でなくなるため。実測では群の生成順が守られ（thinking が空 0/10）、
# 速度はフラット比 +0.37 秒、方向転換への追随は 9/9 で変わらなかった。
#
# **入れ子なのは LLM とのやり取りの中だけ**。runner が `to_flat()` で従来の
# MainChatOutput に直すので、会話ログ・配信・TTS・記憶は形式も中身も現行のまま。
main_agent: Agent[None, MainChatOutput] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=NativeOutput(NestedChatOutput),
)


async def run_main_chat(input_: MainChatInput) -> MainChatOutput | str:
    """統合応答ノードを実行し、MainChatOutput を返す。

    添付画像（attached_images）がある場合は pydantic_ai BinaryContent に変換して
    user_prompt と並列に LLM へ渡す（Dify vision: enabled 相当）。

    失敗時は "[エラー] ..." 文字列を返す（例外を raise しない）。
    """
    try:
        # file_names は明示指定があればそれを使い、無ければ attached_images から派生
        file_names = input_.file_names or derive_file_names(input_.attached_images)
        instructions = _build_instructions(input_)
        user_prompt_text = _build_user_prompt(input_, file_names=file_names)

        attachments = [
            *to_binary_contents(input_.attached_images),
            *to_media_binary_contents(input_.attached_media),
        ]
        if attachments:
            user_prompt: list = [user_prompt_text, *attachments]
        else:
            user_prompt = user_prompt_text

        result = await main_agent.run(user_prompt, instructions=instructions)
        record_llm_usage(main_agent, result, kind="main")
        # LLM は入れ子で返すが、**呼び出し元は従来のフラット形しか知らない**ので直す
        # （runner と同じ扱い。会話ログ・配信・TTS・記憶の形式は現行のまま）
        out = result.output
        return out.to_flat() if hasattr(out, "to_flat") else out
    except Exception as exc:
        return f"[エラー] main_chat 失敗: {exc!r}"
