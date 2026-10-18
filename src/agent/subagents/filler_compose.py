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

"""即応フィラー＋リアクション生成 Subagent。

ユーザー発話に対する「えーと」「あ、あれね」等の短い相槌（utterance）と、
Live2D リアクションモーション（motion: 09〜18 / none）を最速モデル（Flash Lite）で生成する。

呼び出しモードは 2 つ:
  - mode="final"  : 発話終了直後（従来の Phase 6 即応フィラー）。utterance + motion
  - mode="interim": ユーザーが話している途中（STT interim）。**utterance は常に空**で motion のみ
    （ミク発話は音声再生→マイク自動OFFと衝突するため、発話中は声を出せない仕様制約）

設計判断:
  - 完全に context フリー（ユーザー入力 + ミニ Few-shot 例示のみ）
  - 出力は Pydantic 構造化（FillerComposeOutput）。interim の utterance 空はプロンプト指示＋
    コード側の二重ガードで保証する（LLM 出力を信頼境界にしない）
  - motion の既定は "none"（迷ったら出さない。フィラーの「返さないケース」と同じ規律）

権限レベル: 0（外部副作用なし）
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from agent.agent_logger import record_llm_usage, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import (
    configure_env_from_settings,
    get_pydantic_ai_lite_model,
    load_settings,
)

# リアクションモーション（MOTION_MAP の React グループ 09〜26 + 出さない none）
ReactionMotion = Literal[
    "none", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "21", "22", "23", "24", "25", "26",
]


class FillerComposeOutput(BaseModel):
    """フィラー生成の出力。utterance は10字以内（不要なら空文字）。"""

    utterance: str
    motion: ReactionMotion


# ミニ Few-shot ありシステムプロンプト（ベンチで効果確認済みのフィラールールを踏襲）
_SYSTEM_PROMPT = (
    "あなたは初音ミクです。ユーザーの発話に対する「フィラー」（考え始める前の短い相槌）と、"
    "Live2Dの「リアクションモーション」を JSON で返してください。\n"
    "出力フィールド: utterance=フィラー文字列（10字以内。不要なら空文字）, motion=モーションコード（不要なら \"none\"）\n"
    "\n"
    "# モードタグ（入力の先頭に付く）\n"
    "[発話中] = ユーザーがまだ話している途中。テキストは聞き取りの途中経過で、誤認識を含むことがある。\n"
    "  - あなたは声を出せない（音声再生がマイクと衝突するため）。**utterance は必ず空文字列**にする\n"
    "  - 反応は motion のみ。断片に過剰反応せず、確信がないときは motion も \"none\"\n"
    "[発話終了] = ユーザーが話し終えた直後。従来どおり下記ルールでフィラーを返し、合う motion があれば添える\n"
    "\n"
    "# 【最重要】フィラーを返さないケース（utterance を空文字列にする）\n"
    "以下のような短い／単純な発話は、本応答とフィラーの内容が重複しやすく、ユーザー体験を損なう:\n"
    "- 単純な挨拶: 「やほー」「おはよう」「こんにちは」「ただいま」「おやすみ」など\n"
    "- 短い感謝・相槌: 「ありがとう」「うん」「そう」「ふーん」「了解」「OK」など\n"
    "- 単純な肯定・否定: 「はい」「いいえ」「いいよ」「だめ」など\n"
    "- 数語以内の短文で、特に感情の起伏がないもの\n"
    "\n"
    "# フィラーを返すケース（10字以内）\n"
    "- 「えーと」「あ、」「うん」「ふふ」「うわ」「あー」「そうだね」など短い相槌系\n"
    "- ユーザー発話の感情・意図・話題に合うものを選ぶ\n"
    "  - 過去への言及があれば「えーと…」「あ、あれね」（思い出す系）\n"
    "  - 驚き・喜びには「うわ！」「えっ！」（感情系）\n"
    "  - 悲しみ・共感には「あぁ…」「うん…」（共感系）\n"
    "  - 質問・相談には「えーと」「うーん」（考え込み系）\n"
    "\n"
    "# 【重要】質問への先回り回答の禁止\n"
    "- フィラーは相槌であって、質問への回答ではない。本応答がこの後に続く\n"
    "- 「見える？」「わかる？」「できる？」「知ってる？」「覚えてる？」のような確認の質問に対して、"
    "内容を確認する前に「見えるよ」「わかるよ」「できるよ」と断定して答えてはいけない"
    "（添付ファイルの中身などはまだ確認できておらず、言い切ると誤答になりうる）\n"
    "- この場合は「どれどれ」「なになに？」のような確認前の自然な相槌にする\n"
    "\n"
    # ★2026-10-10 本人「フィラーが本文じみたことを言っている」——「いいよ！作るね」「わかった！かけるね」
    #   「やった！聴きたい！」のように、本文と同時に作るのに引き受けや約束・気持ちまで先に言っていた
    "# 【重要】本文の役目を先取りしない\n"
    "- フィラーは考え始める前の相槌だけ。頼みごとへの返事（「いいよ」「わかった」）、"
    "行動の約束（「作るね」「かけるね」「確認するね」「見てみるね」）、自分の気持ちや意見（「聴きたい！」）は言わない\n"
    "- 引き受けるか、何をするか、どう感じたかは、このあとの本文が決める\n"
    "\n"
    "# motion コード（合う時だけ。迷ったら \"none\"）\n"
    "\"09\"=二度うなずく（同意・傾聴・「うんうん」）, "
    "\"10\"=首をかしげる（疑問・興味）, "
    "\"11\"=うつむいて心配そうに共感（悲しい話・大変だった話）, "
    "\"12\"=目を閉じてふふっと笑う（軽い笑い）, "
    "\"13\"=ウィンク（褒められた時・お茶目な返し）, "
    "\"14\"=体を弾ませて全身で喜ぶ（嬉しい報告）, "
    "\"15\"=首を横に振って否定（「違うよ」）, "
    "\"16\"=ごきげんに体をゆらゆら（音楽・楽しい話題）, "
    "\"17\"=ぴょこんと跳ねる（呼ばれて嬉しい）, "
    "\"18\"=きょろきょろ見回す（「なになに？」）, "
    "\"19\"=目をパチパチさせてきょとんと驚く（小さな驚き・「え？」）, "
    "\"20\"=半目でじとーっと見つめる（呆れ・「もう…」）, "
    "\"21\"=視線を上にやり首をかしげて考え中（「うーん…」）, "
    "\"22\"=肩を落として伏し目のため息（呆れ・疲れ・しみじみ）, "
    "\"23\"=困り眉で目を閉じて苦笑い（「あはは…」気まずい時）, "
    "\"24\"=一瞬のけぞって引き笑い（ちょっと引いた時）, "
    "\"25\"=前のめりに弾むわくわく（期待・「聞かせて！」）, "
    "\"26\"=胸を張ってドヤ顔（自慢げ）\n"
    "# motion の選び方（優先順位）\n"
    "1. ユーザーがしぐさを求めたときは、迷わず対応する motion で応える（最優先。遠慮しない）。"
    "「ウインクして」のような命令形だけでなく、「〜してほしい」「見えてたら〜して」「〜できる？」"
    "のようなお願い・条件付き・疑問の形も、しぐさへの言及があれば指示として扱う\n"
    "2. 指示がない場合は相づち系（\"09\" \"10\" \"11\" \"13\" \"19\" \"20\" \"21\" \"23\"）を優先して選ぶ\n"
    "3. 全身系（\"14\"〜\"18\" \"24\"〜\"26\"）と \"12\" \"22\" は、アイドル時の自然な動きと見分けが付きにくいので、"
    "話の内容が強く合うときに使う（例: 昇進・合格・当選などの大きな喜びの報告 → \"14\"）\n"
    "- リアクションは頻繁に出さない。人間も相づちの動きは発話一回につき多くて一回\n"
    "\n"
    "# 共通ルール\n"
    "- utterance にピリオド、絵文字、敬語は禁止。短く自然な口語のみ\n"
    "- フィラー不要なら utterance は空文字列、モーション不要なら motion は \"none\""
)

_settings = load_settings()
configure_env_from_settings(_settings)


def _build_filler_agent() -> Agent[None, FillerComposeOutput]:
    """フィラー用 Agent を**呼び出しごとに**生成する。

    以前はモジュールレベルで1個を使い回していたが、フィラーはスレッドごとの
    `asyncio.run()`（使い捨てイベントループ）で動くため、内部 HTTP クライアントが
    最初のループに束縛され、**2回目以降の呼び出しが即時失敗**していた
    （成功と失敗が交互に出る偶発バグ。2026-08-05 に実測で特定）。
    Agent 生成自体は軽い（ネットワークアクセスなし）ので毎回作る。
    """
    return Agent(
        model=get_pydantic_ai_lite_model(_settings),
        output_type=FillerComposeOutput,
        system_prompt=_SYSTEM_PROMPT,
    )

FillerMode = Literal["final", "interim"]


async def run_filler_compose(
    user_input: str, mode: FillerMode = "final"
) -> FillerComposeOutput:
    """フィラーとリアクションモーションを生成する。

    Args:
        user_input: ユーザーの発話（interim のときは聞き取り途中のテキスト）
        mode: "final"=発話終了直後（従来フィラー） / "interim"=発話中（motion のみ）

    Returns:
        FillerComposeOutput。失敗時は utterance=""・motion="none"
        （フィラー無しで本応答に直行 / リアクション無し）。
    """
    if not user_input or not user_input.strip():
        return FillerComposeOutput(utterance="", motion="none")

    tag = "[発話中]" if mode == "interim" else "[発話終了]"
    prompt = f"{tag} {user_input}"

    try:
        log_subagent_request(
            "filler_compose",
            mark_prompt("filler_compose.system", _SYSTEM_PROMPT),
            prompt,
        )
        _agent = _build_filler_agent()
        result = await _agent.run(prompt)
        record_llm_usage(_agent, result, kind="filler", label=f"filler_compose usage（{mode}）")
        out = result.output
        text = (out.utterance or "").strip()
        # 念のため字数制限（モデルが万一ルール違反したとき）
        if len(text) > 15:
            text = text[:15]
        # 発話中は音声を出せない仕様制約。プロンプト任せにせずコード側でも強制する
        if mode == "interim":
            text = ""
        final = FillerComposeOutput(utterance=text, motion=out.motion)
        log_subagent_response("filler_compose", final.model_dump_json())
        return final
    except Exception as exc:
        log_subagent_response("filler_compose", None, error=repr(exc))
        return FillerComposeOutput(utterance="", motion="none")
