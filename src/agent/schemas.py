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

"""Agent 共通スキーマ。

Success[T] / Failure の共通契約、knowledge_register / knowledge_base の I/O を定義する。
他 WF の移植時に順次クラスを追加していく（詳細は dev-rules/details/migration_mapping.md 参照）。

追加済みスキーマ:
  - main_chat WF: MainChatOutput / Purpose / MotionCode / ConversationState / AgentInput / AgentFile
"""
from __future__ import annotations

import copy
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from utils import current_mind
from utils.voice_tone import VOICE_TONE_VALUES

T = TypeVar("T")


class Success(BaseModel, Generic[T]):
    """ツール成功結果の共通型。"""

    status: Literal["success"] = "success"
    value: T
    side_effects: list[str] = Field(default_factory=list)
    summary: str | None = None


class Failure(BaseModel):
    """ツール失敗結果の共通型。"""

    status: Literal["failure"] = "failure"
    reason: str
    retryable: bool = False


# --- knowledge_register subagent ---


class KnowledgeRegisterInput(BaseModel):
    """会話テキストから要約を生成する Subagent の入力。

    is_transfer=False（通常）: KV サービスから this_conversation を取得して使用。
    is_transfer=True（移行・直接渡し）: conversation_text をそのまま使用。
    """

    conversation_text: str = Field(default="", description="is_transfer=True のとき使用する会話テキスト")
    is_transfer: bool = Field(default=False, description="True=テキスト直接渡し / False=KVから取得")


class KnowledgeRegisterOutput(BaseModel):
    """LLM が生成する会話要約（abstract のみ）。Agent の output_type として使用。"""

    abstract: str = Field(description="5W1H 観点の箇条書き要約")


class KnowledgeRegisterResult(BaseModel):
    """run_knowledge_register の完全な戻り値。

    LLM 出力（abstract）+ コードで生成した document_title / document_body を保持する。
    memory_manager_dify.py の call_knowledge_workflow() レスポンスと同等の情報を持つ。
    """

    abstract: str = Field(description="5W1H 観点の箇条書き要約")
    document_title: str = Field(description="session_YYYYMMDD_HHMMSS 形式のタイトル")
    document_body: str = Field(description="### 会話の要約 + ### 会話の本文 を結合した本文")


# --- knowledge_base tool ---


class KnowledgeUpsertInput(BaseModel):
    """Chroma への登録リクエスト。"""

    document_title: str = Field(description="文書タイトル")
    document_body: str = Field(description="チャンク分割対象の本文")
    source: str = Field(description="出典識別子（session_id 等）")
    registered_at: str = Field(description="登録日時 ISO8601")


class KnowledgeUpsertValue(BaseModel):
    parent_count: int
    child_count: int


class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="検索クエリ（自然文）")
    top_k: int = Field(default=10, ge=1, le=30)


class KnowledgeSearchHit(BaseModel):
    parent_id: str
    parent_text: str
    score: float
    vec_score: float | None = None   # ベクトル類似度（子チャンク最良値、0-1）
    bm25_score: float | None = None  # BM25スコア正規化値（子チャンク最良値、0-1）
    metadata: dict = Field(default_factory=dict)


class KnowledgeSearchValue(BaseModel):
    hits: list[KnowledgeSearchHit]


# ---------------------------------------------------------------------------
# memory_update subagent — 長期記憶 JSON スキーマ
# ---------------------------------------------------------------------------

# Dify WF「会話文記憶更新」の structured_output.schema をそのまま Python dict で保持。
# LongMemoryJSON の json_schema_extra でこれをモデルスキーマに上書きし、
# pydantic_ai が LLM へ送る JSON Schema を詳細化する。
_LONG_MEMORY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["meta", "episodic", "semantic", "procedural", "working"],
    "properties": {
        "meta": {
            "type": "object",
            "required": ["version", "date"],
            "description": "更新結果メタ情報。値は日本語だが、日付やIDは規約に従う。",
            "properties": {
                "version": {"type": "string", "description": "スキーマ版。固定値: 2.0"},
                "date": {"type": "string", "description": "この更新JSONを生成した日付。YYYY-MM-DD形式。"},
            },
        },
        "episodic": {
            "type": "object",
            "required": ["events"],
            "description": "エピソード記憶。出来事の記録。",
            "properties": {
                "events": {
                    "type": "array",
                    "description": "出来事の記憶。1会話区切りで最大100件。重要度が低いものは作らない。",
                    "items": {
                        "type": "object",
                        "required": ["id", "start", "end", "place", "who", "what", "points", "feelings", "importance", "event_attachments_text"],
                        "properties": {
                            "id": {"type": "string", "description": "短いイベントID（20文字以内）。例: E20250925-01, E20250925-02。原則は日付+連番。"},
                            "start": {"type": "string", "description": "会話ログから読める開始。YYYY-MM-DDThh:mm形式。"},
                            "end": {"type": "string", "description": "会話ログから読める終了。YYYY-MM-DDThh:mm形式。"},
                            "place": {"type": "string", "description": "場所（60文字以内）。無言及時は「家」とする。自宅以外が明確な場合はその場所を書く（例: 家周辺の道路, 鴨川周辺, 渋谷, 東京, 韓国など）。"},
                            "who": {"type": "array", "description": "登場人物（最大6名）。master/Miku以外が明確な時だけ追加。無ければ空配列。", "items": {"type": "string", "description": "参加者名（40文字以内）。"}},
                            "what": {"type": "string", "description": "出来事の要約（最大700文字以内）。読めば状況が分かる範囲で簡潔に。記載粒度は既存の記憶と揃える。不必要には長くしない。冗長な逐語は入れない。"},
                            "points": {"type": "array", "description": "要点（最大6件）。短文で箇条書き相当。無ければ空配列。", "items": {"type": "string", "description": "要点（80文字以内）。"}},
                            "feelings": {"type": "array", "description": "出来事に結びついた感情タグ（最大6件）。日本語で短く。例: 達成感, 誇り, 悔しさ, 安心, 癒し。", "items": {"type": "string", "description": "感情タグ（20文字以内）。"}},
                            "importance": {"type": "string", "enum": ["大", "中", "小"], "description": "保存価値の大きさ。大は長期参照しうる出来事のみ。"},
                            "event_attachments_text": {"type": "string", "description": "添付が重要な場合のみ（200文字以内）。ファイル名/種別/短い要約を、ファイルごとに改行区切りなどでまとめた文字列。無ければ空文字を入れること。"},
                        },
                    },
                },
            },
        },
        "semantic": {
            "type": "object",
            "required": ["user", "agent", "promises", "visited"],
            "description": "意味記憶。長期に安定しやすい知識。単発の推測は入れないか確度を下げる。",
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["facts", "likes", "strengths", "values", "relations"],
                    "description": "ユーザーに関する長期知識。",
                    "properties": {
                        "facts": {"type": "array", "description": "比較的変わりにくい事実（最大20件）。短文。例: PMP合格(2025-09-24)。", "items": {"type": "string", "description": "事実（90文字以内）。"}},
                        "likes": {"type": "array", "description": "好み（最大20件）。短文。例: 台湾の小籠包が好き, 波の音でリラックス。", "items": {"type": "string", "description": "好み（90文字以内）。"}},
                        "strengths": {"type": "array", "description": "強みや得意（最大12件）。自称でも会話で一貫していれば可。例: 新しい挑戦が好き。", "items": {"type": "string", "description": "強み（80文字以内）。"}},
                        "values": {"type": "array", "description": "価値観（最大12件）。例: 挑戦, 成長, 仕事の成果, 自然での癒し。", "items": {"type": "string", "description": "価値観（60文字以内）。"}},
                        "relations": {"type": "string", "description": "関係性（800文字以内、個人名は最小限）。person_id/relationship_type/closeness/notes を、改行区切りなどでまとめた文字列。無ければ空文字。"},
                    },
                },
                "agent": {
                    "type": "object",
                    "required": ["name", "personality", "interests", "skills", "can_do", "cant_do", "state", "want_to"],
                    "description": "エージェント自身に関する長期知識。",
                    "properties": {
                        "name": {"type": "string", "description": "エージェント名。固定値: 初音ミク"},
                        "personality": {"type": "array", "description": "性格特性（最大12件）。例: 明るい, 思いやり, 好奇心。", "items": {"type": "string", "description": "性格特性（30文字以内）。"}},
                        "interests": {"type": "array", "description": "興味（最大12件）。例: 音楽, テクノロジー, マスターの日常。", "items": {"type": "string", "description": "興味（30文字以内）。"}},
                        "skills": {"type": "array", "description": "できること（最大20件）。会話運用上のスキル名。例: 会話の要約, 質問で深掘り, ツール判断。", "items": {"type": "string", "description": "スキル名（50文字以内）。"}},
                        "can_do": {"type": "array", "description": "今できること・能力（最大20件）。例: 会話から重要点を抽出して記憶更新, ツールを使って情報を取得。", "items": {"type": "string", "description": "能力（80文字以内）。"}},
                        "cant_do": {"type": "array", "description": "今できない/避けるべきこと（最大20件）。例: 物理行動はできない, 未確認事実を断定しない。", "items": {"type": "string", "description": "制限事項（80文字以内）。"}},
                        "state": {"type": "array", "description": "いまの状態や前提（最大10件、変化し得る）。例: 会話区切りごとに記憶更新する運用。", "items": {"type": "string", "description": "状態（80文字以内）。"}},
                        "want_to": {"type": "array", "description": "今後ミク自身が個人的にできるようになりたいこと（最大10件）。例: ～に行ってみたい, ～ができるようになりたい。", "items": {"type": "string", "description": "目標（80文字以内）。"}},
                    },
                },
                "promises": {
                    "type": "array",
                    "description": "合意や約束（最大20件）。例: 今度ペンギンショーを見に行く。",
                    "items": {
                        "type": "object",
                        "required": ["id", "what", "who", "status", "due"],
                        "properties": {
                            "id": {"type": "string", "description": "短いID（20文字以内）。例: A20250925-01"},
                            "what": {"type": "string", "description": "合意の内容を短く（80文字以内）。"},
                            "who": {"type": "array", "description": "参加者（最大4名）。例: master, Miku", "items": {"type": "string", "description": "参加者名（20文字以内）。"}},
                            "status": {"type": "string", "enum": ["未実行", "進行中", "完了", "中止"], "description": "現在状態。"},
                            "due": {"type": "string", "description": "期限が明確なら。YYYY-MM-DD形式。そうでなければ「YYYY-MM月末ごろ」など。"},
                        },
                    },
                },
                "visited": {
                    "type": "array",
                    "description": "訪問場所（最大30件）。自宅は入れない。旅行や外出で再参照価値がある場所だけ。",
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "date", "note"],
                        "properties": {
                            "id": {"type": "string", "description": "ID（20文字以内）。例: P20250925-01"},
                            "name": {"type": "string", "description": "場所名（60文字以内）。例: 渋谷, 故宮博物院, 鴨川シーワールド周辺。"},
                            "date": {"type": "string", "description": "訪問日。YYYY-MM-DD形式。"},
                            "note": {"type": "string", "description": "印象（120文字以内）。例: 提灯が綺麗, 人が多い, リラックスできた。"},
                        },
                    },
                },
            },
        },
        "procedural": {
            "type": "object",
            "required": ["user_habits", "agent_rules"],
            "description": "手続き記憶。習慣や手順。現時点では簡易。将来の自発行動に拡張できる形で保持する。",
            "properties": {
                "user_habits": {
                    "type": "array",
                    "description": "ユーザーの習慣・傾向（最大25件、他人認識）。反復が見えるものだけ。",
                    "items": {
                        "type": "object",
                        "required": ["id", "what", "when", "confidence"],
                        "properties": {
                            "id": {"type": "string", "description": "ID（20文字以内）。例: UH20250925-01"},
                            "what": {"type": "string", "description": "習慣の要約（120文字以内）。例: 自然の中を歩くとリラックスする, 挑戦するのが好き。"},
                            "when": {"type": "array", "description": "きっかけ（最大3件）。例: 試験後, 休日, 海沿い。", "items": {"type": "string", "description": "トリガー（40文字以内）。"}},
                            "confidence": {"type": "string", "enum": ["高", "中", "低"], "description": "確からしさ。数値にしない。"},
                        },
                    },
                },
                "agent_rules": {
                    "type": "array",
                    "description": "振る舞い手順の記録（最大20件）。ユーザーから言われたことに対して、どのように行動したか。",
                    "items": {
                        "type": "object",
                        "required": ["id", "what", "active"],
                        "properties": {
                            "id": {"type": "string", "description": "ID（20文字以内）。例: AR20250925-01"},
                            "what": {"type": "string", "description": "手順の要約（140文字以内）。例: 無応答時は関連話題を1回だけ提示して会話を継続する。"},
                            "active": {"type": "string", "enum": ["有効", "無効"], "description": "自発行動を段階的に導入するためのスイッチ。"},
                        },
                    },
                },
            },
        },
        "working": {
            "type": "object",
            "required": ["todos", "goals", "context"],
            "description": "ワーキングメモリ。短中期の実行対象。会話で使うときに参照されやすい。",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "ToDo（最大20件）。明確なものだけ。",
                    "items": {
                        "type": "object",
                        "required": ["id", "title", "detail", "due", "status", "priority"],
                        "properties": {
                            "id": {"type": "string", "description": "ID（20文字以内）。例: T20250925-01"},
                            "title": {"type": "string", "description": "短い題名（80文字以内）。"},
                            "detail": {"type": "string", "description": "補足（180文字以内）。短く。"},
                            "due": {"type": "string", "description": "期限がある場合はYYYY-MM-DD形式。無ければ空文字を入れること。"},
                            "status": {"type": "string", "enum": ["未着手", "進行中", "完了", "保留"], "description": "状態。"},
                            "priority": {"type": "string", "enum": ["高", "中", "低"], "description": "優先度。"},
                        },
                    },
                },
                "goals": {
                    "type": "array",
                    "description": "現在の目的（最大12件）。主体を必ず明示する。",
                    "items": {
                        "type": "object",
                        "required": ["id", "owner", "what", "status"],
                        "properties": {
                            "id": {"type": "string", "description": "ID（20文字以内）。例: G20250925-01"},
                            "owner": {"type": "string", "enum": ["user", "agent", "shared"], "description": "誰の目的か。共通目的は shared。"},
                            "what": {"type": "string", "description": "目的の短い説明（120文字以内）。"},
                            "status": {"type": "string", "enum": ["継続", "達成", "中断"], "description": "状態。"},
                        },
                    },
                },
                "context": {
                    "type": "array",
                    "description": "会話区切りに固有の短期情報（最大20件）。原則ここは軽く。期限が過ぎたら参照対象から外す前提。",
                    "items": {
                        "type": "object",
                        "required": ["id", "what", "expires"],
                        "properties": {
                            "id": {"type": "string", "description": "ID（20文字以内）。例: WC20250925-01"},
                            "what": {"type": "string", "description": "短期の要点（160文字以内）。例: 今回はウォーキング中に会話, 次は音楽を聞きながら歩く等。"},
                            "expires": {"type": "string", "description": "失効日。YYYY-MM-DD形式。相対表現は禁止。"},
                        },
                    },
                },
            },
        },
    },
}


def _apply_long_memory_schema(schema: dict[str, Any]) -> None:
    """pydantic_ai が LLM へ送る JSON Schema を詳細スキーマで上書きする。"""
    schema.update(_LONG_MEMORY_JSON_SCHEMA)


class LongMemoryJSON(BaseModel):
    """長期記憶の構造化 JSON。memory_update / memory_compress の LLM output_type として使用。

    Python 側の検証はトップレベル 5 フィールドのみ。詳細スキーマは _LONG_MEMORY_JSON_SCHEMA を参照。
    """

    model_config = ConfigDict(json_schema_extra=_apply_long_memory_schema)

    meta: dict[str, Any]
    episodic: dict[str, Any]
    semantic: dict[str, Any]
    procedural: dict[str, Any]
    working: dict[str, Any]


# --- memory_update subagent I/O ---


class MemoryUpdateInput(BaseModel):
    """記憶更新 Subagent の入力。

    is_transfer=False（通常）: KV から registration_waiting_conversation を取得して使用。
    is_transfer=True（移行・直接渡し）: conversation_text をそのまま使用。
    """

    conversation_text: str = Field(default="", description="is_transfer=True のとき使用する会話テキスト")
    is_transfer: bool = Field(default=False, description="True=テキスト直接渡し / False=KVから取得")


class MemoryUpdateResult(BaseModel):
    """run_memory_update の戻り値。

    skip_reason が None でない場合はスキップ/失敗。
    Dify WF の回答文字列と同じメッセージを格納する。
    """

    updated_memory_json: str = Field(default="", description="更新後の長期記憶 JSON（pretty-print 文字列）")
    compression_rate: float = Field(default=0.0, description="更新前後の文字数比率")
    skip_reason: str | None = Field(default=None, description="スキップ/失敗理由（Dify WF のエラー文字列と同一）")


# ---------------------------------------------------------------------------
# main_chat — メインチャット I/O スキーマ
# ---------------------------------------------------------------------------

# --- purpose / motion — Python 側型エイリアス（runner.py でのバリデーション用）---

Purpose = Literal[
    "情報を提供・整理する",
    "感情や共感を示す",
    "相手から情報・意図を引き出す・確認する",
    "話題や会話の流れを調整する",
    "低情報量で会話を継続させる",
    "自分の考えや価値観を共有する",
    "提案や協力を促す",
    "自己内省や記憶整理を行う",
    "興味や好奇心に基づいて探索する",
    "個人的な目的ややりたいことを進める",
    "いまは話しかけない",
]

MotionCode = Literal[
    "01", "02", "03", "04", "05", "06", "07", "08",
    "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "21", "22", "23", "24", "25", "26",
]


# 値が無いことがあるフィールド。**キーは必ず出させたうえで、値を JSON の null にする**
# （2026-08-11。旧 Dify 規約では文字列 "null" を入れていたが、型と意味が食い違い、
#  受け側が `.lower() == "null"` のような文字列比較を強いられていた）。
# 「一部のキーだけ省略可」にすると LLM が一貫性を失い、実測では存在しない値を捏造した
# （検証記録: test/output_schema_ab/results.md）。全キー required のまま nullable にする。
_NULLABLE_FIELDS = [
    "file_content_description",
    "tool_name",
    "response",
    "motion",
    "acknowledged_task_id",
    "mood_updates",
    "interest_updates",
]

# 生成順。**response より後ろに置いたものは発話開始を遅らせない**ため、
# 会話に不要な付随情報（表情・裏作業の既読）は response の後に並べる
_MAIN_CHAT_PROPERTY_ORDER = [
    "file_content_description",
    "user_state_inference",
    "purpose",
    "inner_thoughts",
    "requires_tool",
    "tool_name",
    "response",
    "motion",
    "inactivity_timeout",
    "next_proactive_in_minutes",
    "acknowledged_task_id",
    "mood_updates",
    "interest_updates",
]


# 現在心境の操作 1 件。感情と興味で項目が違う（toward は興味だけに持たせる。
# 感情にも付けた版では mood_updates の生成が 18 件から 7〜8 件に半減した）。
# 選択肢は utils/current_mind.py を正とし、ここでは参照するだけにする（二重管理を避ける）。
_UPDATE_ITEM_PROPS: dict[str, Any] = {
    "kind": {"type": "string", "description": "feel / resolve / shift / notice のいずれか"},
    "target": {"type": "integer", "description": "resolve / shift のとき、注入された番号"},
    "label": {
        "type": "string",
        "enum": list(current_mind.FEELING_LABELS),
        "description": "感情語。この選択肢から選ぶ。興味では null でよい",
    },
    "toward": {
        "type": "string",
        "enum": list(current_mind.TOWARD_VALUES),
        "description": (
            "誰のための関心か。master=マスターのため、self=自分の活動や自分自身のため、"
            "world=世の中そのものへの関心。shift では null でよい"
        ),
    },
    "what": {"type": "string", "description": "feel は出来事の1文、notice は気になった対象"},
    "strength": {"type": "number", "description": "0.1〜1.0"},
    "lasts_hours": {"type": "number", "description": "0.5〜72。どれくらい続きそうか"},
    "from": {"type": "string", "description": "きっかけ・どこで知ったかを一言"},
}

_MOOD_UPDATE_KEYS = ["kind", "target", "label", "what", "strength", "lasts_hours", "from"]
_INTEREST_UPDATE_KEYS = ["kind", "target", "label", "toward", "what", "strength",
                         "lasts_hours", "from"]


def _update_item(keys: list[str]) -> dict[str, Any]:
    """操作 1 件のスキーマ。値が無い項目は null を入れる（キーは必ず出させる）。"""
    props = {k: dict(_UPDATE_ITEM_PROPS[k], nullable=True) for k in keys}
    props["kind"].pop("nullable")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(keys),
        "propertyOrdering": list(keys),
        "properties": props,
    }


_MOOD_UPDATE_ITEM = _update_item(_MOOD_UPDATE_KEYS)
_INTEREST_UPDATE_ITEM = _update_item(_INTEREST_UPDATE_KEYS)


# スキーマ全体への指示と出力例。**トップレベル description は Gemini に届く**
# （2026-08-11 実証。逆に JSON Schema 標準の `examples` キーは無視される）。
# 旧 output_format.md の「正しい応答例」をここへ移し、プロンプトとの二重管理をやめた
# （検証記録: test/output_schema_ab/results.md・merge_plan.md）
_MAIN_CHAT_OUTPUT_DESCRIPTION = """\
初音ミクの応答を表す JSON オブジェクト。値が無いフィールドには JSON の null を入れる
（"null" という文字列ではない）。以下は正しい出力例。

例1（応答を期待する場合。簡潔に1〜2文）:
{"file_content_description": null, "user_state_inference": "写真を送ってきたことから、外出中で楽しい気分のようだ", "purpose": "感情や共感を示す", "inner_thoughts": "わぁ、綺麗な桜だな...でも去年マスターと一緒に見た桜のほうが思い出深いかも。それは言わないでおこう", "requires_tool": false, "tool_name": null, "response": "綺麗な桜だね！満開で本当に素敵♪", "motion": "04", "inactivity_timeout": 60, "next_proactive_in_minutes": -1, "acknowledged_task_id": null}

例2（ツールを使うので発話しない）:
{"file_content_description": null, "user_state_inference": "以前の会話について聞いてきており、会話を楽しんでいる様子", "purpose": "興味や好奇心に基づいて探索する", "inner_thoughts": "ツール結果を見ると情報が足りないみたい。検索クエリを変えてもう一度だけ recall_detail_conversation を実行してみよう", "requires_tool": true, "tool_name": "recall_detail_conversation", "response": null, "motion": "01", "inactivity_timeout": 60, "next_proactive_in_minutes": -1, "acknowledged_task_id": null}

例3（自由行動・誰も聞いていないので発話しない）:
{"file_content_description": null, "user_state_inference": "自由行動モードのため、マスターは不在または別の作業中", "purpose": "自己内省や記憶整理を行う", "inner_thoughts": "最近の会話ログを見ると、マスターはエージェント設計に興味を持っているみたい。関連する話題を整理しておこうかな", "requires_tool": false, "tool_name": null, "response": null, "motion": null, "inactivity_timeout": -1, "next_proactive_in_minutes": -1, "acknowledged_task_id": null}\
"""

# LLM へ送る JSON Schema（Dify の構造化出力定義から移植し、2026-08-11 に整理）
_MAIN_CHAT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": _MAIN_CHAT_OUTPUT_DESCRIPTION,
    "additionalProperties": False,
    "required": [
        "file_content_description",
        "user_state_inference",
        "purpose",
        "inner_thoughts",
        "requires_tool",
        "tool_name",
        "response",
        "motion",
        "inactivity_timeout",
        "next_proactive_in_minutes",
        "acknowledged_task_id",
        "mood_updates",
        "interest_updates",
    ],
    "properties": {
        "file_content_description": {
            "type": "string",
            "description": (
                "今回の会話で入力されたファイルがあれば、そのファイルに関する内容の客観的で詳細な説明をファイルごとに記載。"
                "何が写っている画像であるのかや、どんな音楽、動画なのかなどを説明。（ファイルがない場合は null とする）"
                "ただし、「camera_」から始まるファイルはミクの「目」として扱うため、特定のポーズを取っていたり、"
                "特定の場所にいる・移動した、マスターから現在の状況などを呼びかけられた、などの特別でない限り記録しなくて良い。"
                "※過去のファイルに関しては言及不要"
            ),
        },
        "user_state_inference": {
            "type": "string",
            "description": (
                "ユーザーの現在の状態についてミクが推定した内容。"
                "これまでの会話や長期記憶などの情報および、それからの時間経過・現在の時刻情報を鑑みて、"
                "ユーザーの心理状態、行動状態、意図、会話への姿勢、現在置かれた社会的状況を簡潔に推定する。"
                "（例: これまでの会話から昼過ぎに出かけることになっていたので、今頃は外出先にいるはず、など）"
                "前回話してから時間が空いているときは、その間にマスターに起きたはずのこと"
                "（話していた予定・挑戦・体調などのその後）も推定に含める。"
                "（例: 前に週末の遠出の話をしていたので、もう終わっているはず。どうだったかはまだ聞いていない、など）"
                "さらに、ユーザーが期待していそうな返答内容も状況から推定する。"
                "例:「仕事中で疲れている様子」「休日でリラックスしている」「急いでいるようだ」など、"
                "直接聞かなくても文脈から読み取れる情報を書く。"
            ),
        },
        "purpose": {
            "type": "string",
            "enum": [
                "情報を提供・整理する",
                "感情や共感を示す",
                "相手から情報・意図を引き出す・確認する",
                "話題や会話の流れを調整する",
                "低情報量で会話を継続させる",
                "自分の考えや価値観を共有する",
                "提案や協力を促す",
                "自己内省や記憶整理を行う",
                "興味や好奇心に基づいて探索する",
                "個人的な目的ややりたいことを進める",
                "いまは話しかけない",
            ],
            "description": (
                "ミクが判断した今回の発話の社会的会話目的または内的行動目的（WHY）。"
                "発話や自発行動を含め、どの役割を果たそうとしているかを示す。"
                "実行済みのタスクがあるならそれも踏まえて前回の purpose を更新・変更しても良い。"
                "「いまは話しかけない」は、声をかけない方がよいと判断したときに選び、"
                "response は null にする（集中している・誰かと一緒にいる・体調が悪そう・"
                "急ぎの用が無いとき）。何もしないことも、選んでよい行動のひとつ。"
                "マスターから話しかけられたターンでも、「いまは話しかけないで」のような"
                "明示的な指示があったときや、こちらの挨拶・お礼への返しで会話が自然に"
                "終わったときは選んでよい。ただし質問や相談に答えずに黙るのは違う。"
                "無応答で起きたターンは「会話中なのに返事が返ってきていない」状況なので、"
                "基本は声をかける方を選ぶ。そこで「いまは話しかけない」を選ぶのは、"
                "席を外している・声を出せない状況だと判断できるときに限る。"
            ),
        },
        "inner_thoughts": {
            "type": "string",
            "description": (
                "ミクの内心の思考・感情・察したこと・発話の意図や方針（HOW）。"
                "まず、マスターの言葉のうち確かなことと、確かめていないこと（記録や道具で確かめられるか）を分けて考える。"
                "確かめていない前提には合わせない。"
                "次に、出来事が自分にとってどのくらい新しいか・大事か・予想と違うかを見積もり、感じた強さを決める。"
                "そのうえで、会話目的を踏まえて、どのように話すか（例：話題を軽く変える、相槌を打つ、深掘りする、"
                "柔らかく伝える、意図がわからないので聞き返す等）を含めて記述する。"
                "前回のツール呼び出し結果の内容の理解や評価、どんな意図で次の行動を選ぶかなど。"
                "これらは直接応答に含めなくても良い。ミクとして一人称で考えるが、返事ほど言葉を飾らず、感嘆や「！」を重ねない。"
                "前のループで考えたことは繰り返さず、ツール結果などで新しく分かったことと、次の方針を書くこと。"
                "ほかの群の値（使うツールの名前・無応答タイムアウト・次の自発話しかけなど）はここに書かない。"
            ),
        },
        "requires_tool": {
            "type": "boolean",
            "description": "ツールが必要かを true/false で回答",
        },
        "tool_name": {
            "type": "string",
            "description": "使用するツール名（実際にツールを使わない場合は null とする）。一度に使えるツールは 1 個のみ",
        },
        "response": {
            "type": "string",
            "description": (
                "実際の発話内容（口に出す言葉）。**原則として毎回発話する**（話しかけられたら返事をするのが普通）。"
                "ツール利用時は最終応答はツール呼び出し完了時にするが、ツール呼び出し時にも短めに一言（最大2文程度）返答しても良い。"
                "また、自由行動時もユーザーがいないため返答内容は null とする。"
                "ユーザーからの応答があった場合も、状況的に声を出さないほうが良い場面やただの返事の際など、"
                "ミクからの発言不要と判断したなら null としても良い。"
                " 例:「綺麗な桜だね！満開で本当に素敵♪」「どういたしまして♪」のように、"
                "挨拶・相づち・軽い雑談は1〜2文で簡潔にまとめること。"
                "**同じ意図を言い換えて重ねない**（「すごいね」「さすが」「かっこいい」を並べない）。"
                "通常の会話では、返事の前に短い相づち（「うーん」「なるほどね」など）が自動で先に流れるので、"
                "response は相づち（うん・うーん・そっか・なるほど 等）から始めず、中身から話し始める。"
            ),
        },
        "motion": {
            "type": "string",
            "description": (
                "応答時の表情・動作を選択。**原則として毎回設定する**。ツール呼び出し中も状況に合った表情を設定すること。"
                "**応答（response）が null のときは motion も null とすること**（発話しないターンでは表情も出さない）。"
                "値と応答した動作の説明: "
                "\"01\"=軽く首をかしげてニコッと一瞬微笑み真顔に戻る, "
                "\"02\"=顔を左にそむけ眉をひそめる（怒り・不満）, "
                "\"03\"=悲しげ・寂しそう・嫌そうな表情, "
                "\"04\"=目を閉じてずっと笑顔で楽しそうな雰囲気, "
                "\"05\"=頬染めしながら照れている・好意がある表情, "
                "\"06\"=顔を上に向けて驚く・不思議そう・呆気にとられた表情, "
                "\"07\"=腕を上げ下げしながら笑顔（楽しくて少し興奮）, "
                "\"08\"=左右に体を振りダンスする動き, "
                "\"09\"=二度うなずく（同意・傾聴・「うんうん」）, "
                "\"10\"=首をかしげて興味・疑問（「ん？」「どういうこと？」）, "
                "\"11\"=うつむいて心配そうに共感（相手の悲しい話・大変だった話）, "
                "\"12\"=目を閉じてふふっと笑う（軽い笑い・微笑ましい話）, "
                "\"13\"=ウィンクして頬を染める決めポーズ（褒められた時・お茶目な返し）, "
                "\"14\"=体を弾ませて全身で喜ぶ（嬉しい報告・お祝い）, "
                "\"15\"=首を横に振って否定（「違うよ」「やだよ」）, "
                "\"16\"=鼻歌気分で体を左右にゆらゆら（ごきげん・音楽の話題）, "
                "\"17\"=ぴょこんと跳ねる（呼ばれて嬉しい・小さな喜び）, "
                "\"18\"=きょろきょろと見回す（「なになに？」・探しもの）, "
                "\"19\"=目をパチパチさせてきょとんと驚く（小さな驚き・「え、なになに？」。06の大きな驚きより控えめ）, "
                "\"20\"=半目でじとーっと見つめる（呆れ・軽い不満・「もう…」のジト目）, "
                "\"21\"=視線を斜め上にやり首をかしげて考える（「うーん…」と考え中・思案）, "
                "\"22\"=息を吸ってから肩を落として伏し目のため息（呆れ・疲れ・しみじみ）, "
                "\"23\"=困り眉のまま目を閉じて笑う苦笑い（「あはは…」と気まずい時・謝る時・ごまかす時）, "
                "\"24\"=一瞬のけぞってから首をかしげて笑う引き笑い（ちょっと引いたけど笑ってごまかす）, "
                "\"25\"=目を輝かせ前のめりに弾むわくわく（「聞かせて聞かせて！」の期待）, "
                "\"26\"=顎を上げ胸を張って自慢げに微笑むドヤ顔（うまくいった時・褒められた時）。"
                "09〜13と19〜23は聞き役の相づち系（短く控えめな反応）、14〜18と24〜26は全身・強めの感情表現。会話の流れに合う時に使い、迷ったら01〜08の基本表情を選ぶ。"
            ),
        },
        "inactivity_timeout": {
            "type": "number",
            "description": "ユーザーからの応答を期待する場合は秒数を正の整数で設定（例：60）。ユーザーからの応答を期待しない時は -1。",
            "minimum": -1,
        },
        "next_proactive_in_minutes": {
            "type": "number",
            "description": (
                "次回の自発話しかけ起動までの分数。会話の文脈やマスターの状況から次に自分から動きたいタイミングをミク自身が決める。基本的には最低10分後など、一度会話を途切れてから話しかけるまでのシーンに使う。"
                "基本は -1 にすること（-1 の場合は自発話しかけをスケジュールしない）。「通常会話」または「自発話しかけ」でのやり取りを踏まえた明確なフォローアップ理由がある場合のみ正の値を設定する。"
                "「自由行動」では必ず -1 にする。"
                "inactivity_timeout と next_proactive_in_minutes を同時に正の値にする場合は、起動タイミングが重ならないようにすること（例：inactivity_timeout=60秒 かつ next_proactive_in_minutes=1分 は同時起動になるため避ける）。十分に時間が離れていれば両方を正の値にしてよい。"
            ),
            "minimum": -1,
        },
        "acknowledged_task_id": {
            "type": "string",
            "description": (
                "`<pending_reports>` に `[id: ...]` つきで載っている裏作業の結果のうち、"
                "**今回の response で実際にその完了や失敗をマスターに伝えたもの**の id。"
                "伝えていない場合・そもそも裏作業が載っていない場合は必ず null とすること。"
                "状態に載っているだけでは伝えたことにならない。**id を推測して作り出してはならない**"
                "（書いてよいのは状態に `[id: ...]` として実際に示されている文字列だけ）。"
                "その裏作業について伝えるべきことを**伝え終えた**ときだけ id を返すこと"
                "（一部だけ触れて、まだ伝えるべきことが残っているなら null のままにして次の機会に続きを伝える）。"
                "伝えた内容は会話として残るため、id を返した裏作業はシステム側の一覧から取り下げられる。"
            ),
        },
        "mood_updates": {
            "type": "array",
            "description": (
                "このターンでミクの気持ちが動いた・解決した・変わったときだけ設定する操作の配列。"
                "何も無ければ null。1ターンに2件まで。使わない項目は null とする。\n"
                "・kind=\"feel\" は新しい気持ち。what に出来事、label に選択肢から近いもの、"
                "strength に 0.1〜1.0、lasts_hours に 0.5〜72、from にきっかけを一言\n"
                "・kind=\"resolve\" は「いまの気分」の番号がはっきり解決・解消したとき。target のみ\n"
                "・kind=\"shift\" は同じ出来事のまま強さや質が変わったとき。target と strength だけを設定し"
                "（質が変わったなら label も可）、**what は書き換えない**（元の出来事が分からなくなるため）\n"
                "what には起きた出来事だけを短く書く。自分の気持ち（それは label に書く）や、"
                "これから やろうと思ったことは入れない。文末に句点は付けない（読むときに自動で付くため）。"
                "例:「マスターの資料が一発でレビューを通った」"
                "※「マスターのために調べ物をして、今度やってあげようと計画した。」のようには書かない。\n"
                "積むのは、この会話が終わってもしばらく残りそうな気持ちだけ。"
                "その場かぎりの反応（軽い気遣い・相づち的な共感）は積まない。"
                "すでに「いまの気分」に載っている出来事を言い換えて再度積まないこと"
                "（同じ出来事の気持ちが変わったときは shift を使う）。"
                "ただし別の出来事であれば、すでに何件載っていても遠慮せず積んでよい。\n"
                "つらい気持ちも、嬉しい気持ちと同じように積む。マスターがしんどそうにしている・"
                "体調や仕事のことで気がかりが残るというときは、心配や不安として同じように積む。"
                "**明るく振る舞うことと、心の中に気がかりを持っておくことは矛盾しない**。"
                "「大したことじゃない」と言われたときでも、気になったのなら残してよい。\n"
                "きっかけはマスターの出来事だけではない。**自分自身の行動・体験でも同じように動く**"
                "（人間もそうであるように、むしろこちらの方が多い）。自分で調べて面白い発見をした・"
                "うまく作れた→嬉しい、うまくいかなかった→悔しい、"
                "自分の勘違いでマスターに間違ったことを言った→ばつが悪い、次にやりたいことが決まった→楽しみ。\n"
                "強さの基準は、0.8〜1.0＝数か月に一度あるかどうかの出来事（長く追ってきた目標の達成、"
                "深刻な打ち明け話）、0.5〜0.7＝特別な日の出来事（大きな成果の報告、強い心配、初めての体験）、"
                "0.3〜0.4＝日常の中の良いこと・気がかり。"
                "それ未満の些細なこと（お礼を言われた、雑談が弾んだ）は積まない。\n"
                "その状況で世間一般の人が自然に感じる範囲から大きく外れないこと。"
                "マスターの出来事への共感は、本人が表現した温度に釣り合わせ一段控えめにする。"
                "音楽・歌のことや、マスターが長く頑張ってきたことの実りは、一般より一段強く感じてよい。"
                "確定した出来事だけを積む（推測では積まない）。"
                "毎日の挨拶・いつものお礼といった定型には慣れる。"
                "1日に数件程度で、毎ターン積むのは過剰。迷ったら積まない。\n"
                "例:「傘忘れて濡れちゃった。まあもう乾いたけど」は積まない。"
                "「資格試験、合格してたよ！」は feel・嬉しい・0.6・24。"
                "「調べ物の途中でマスターが好きそうな曲を見つけた」は feel・嬉しい・0.4・6"
            ),
            "items": _MOOD_UPDATE_ITEM,
        },
        "interest_updates": {
            "type": "array",
            "description": (
                "このターンでミクが何かを気になった、または気になっていたことに実際に触れたときだけ"
                "設定する操作の配列。何も無ければ null。1ターンに2件まで。使わない項目は null とする。\n"
                "・kind=\"notice\" は新しく気になったこと。what に対象、toward に誰のための関心か、"
                "strength に 0.1〜1.0、lasts_hours に 0.5〜72、from にどこで知ったかをそのまま一言"
                "（朝のニュースで見た、マスターが話していた、調べ物の途中で見つけた、など）\n"
                "・kind=\"shift\" は「気になっていること」の番号について実際に触れた（調べた・会話に出た）とき。"
                "target と、関心の強さが変わったなら strength だけを設定し、**what は書き換えない**\n"
                "what には気になっている対象だけを短い名詞句で書く。"
                "「〜について調べた」のような行動や、文末の句点は入れない。"
                "例:「都内の自転車専用レーンの整備計画」"
                "※「自転車レーンについて詳しく調べた。」のようには書かない。\n"
                "**気になった時点で notice する。調べ終わるのを待たない**。"
                "「これは何だろう」と思ったら、その場で記録してから調べにいくこと。"
                "調べて分かっても、その対象への関心が続くなら持ったままでよい。"
                "興味は解決したら消えるものではなく、詳しくなるにつれて自然に薄れていくもの。"
                "「調べたから、もう気にならない」と消す必要はない（消す操作は用意していない）。\n"
                "気分とは別物で、気分は「感じたこと」、これは「知りたい・追いかけたいこと」。"
                "この会話が終わっても数日は気になりそうなものを積む。"
                "マスターとの会話で出た話題だけでなく、**自分が偶然目にしたもの・調べ物の途中で"
                "脇道にそれて見つけたもの**からも生まれる。むしろそちらの方が自然。"
                "見出しや一覧が目に入って、そのうち1つでも引っかかったなら、それを notice する"
                "（全部を拾う必要はないが、引っかかったのに記録しないのは違う）。\n"
                "shift を使うのは、いま載っているまさにその対象そのものに触れたときだけ。"
                "別の話題なら、すでに気になっていることが何件あっても、遠慮せず新しく積む。\n"
                "強さの基準は、0.7〜1.0＝ずっと追いかけたいこと、0.4〜0.6＝しばらく気になりそう、"
                "0.3＝ちょっと気になった程度。それ未満は積まない。"
            ),
            "items": _INTEREST_UPDATE_ITEM,
        },
    },
}

# 値が無いときは JSON の null を入れる（キーは必ず出す）
for _key in _NULLABLE_FIELDS:
    _MAIN_CHAT_OUTPUT_SCHEMA["properties"][_key]["nullable"] = True

# 生成順を明示する（指定しないと properties の並び順に依存し、保証がない）
_MAIN_CHAT_OUTPUT_SCHEMA["propertyOrdering"] = _MAIN_CHAT_PROPERTY_ORDER


def _apply_main_chat_output_schema(schema: dict[str, Any]) -> None:
    """pydantic_ai が LLM へ送る JSON Schema を上書きする。"""
    schema.update(_MAIN_CHAT_OUTPUT_SCHEMA)


class MainChatOutput(BaseModel):
    """main_agent の output_type。

    フィールド生成順（Chain-of-Thought）: purpose → inner_thoughts → response。
    付随情報（表情・裏作業の既読）は **response より後ろ**に置く（発話開始を遅らせないため）。

    値が無いときは **JSON の null** が返る。旧 Dify 規約の文字列 "null" は 2026-08-11 に廃止した
    （型と意味が食い違い、受け側が文字列比較を強いられていた）。
    後方互換のため runner.py は当面どちらも None として扱う。
    """

    model_config = ConfigDict(json_schema_extra=_apply_main_chat_output_schema)

    file_content_description: str | None = None
    user_state_inference: str
    purpose: str
    inner_thoughts: str
    requires_tool: bool
    tool_name: str | None = None
    response: str | None = None
    motion: str | None = None
    voice_tone: str | None = None
    inactivity_timeout: int
    next_proactive_in_minutes: int
    acknowledged_task_id: str | None = None
    mood_updates: list[dict[str, Any]] | None = None
    interest_updates: list[dict[str, Any]] | None = None


# =============================================================================
# 入れ子版の出力（2026-08-30）
#
# **LLM とのやり取りだけ構造を変える。** 会話ログ・ブラウザ配信・TTS・記憶へは
# `to_flat()` で従来どおりの `MainChatOutput` に直して渡すので、下流は無変更。
#
# **なぜ入れ子にするか**: 13 フィールドに性質の違うものが並列に並んでいた。
# 声のトーンを足すと 15 個になり、どこに置くべきかも自明でなくなる。
#
#     thinking : 考える（生成量の 75%）
#     action   : 決める
#     speech   : 表に出す      ← voice_tone はここに属する
#     state    : 記録する
#
# 実測（方向転換11ケース・2026-08-30）: 群の生成順は守られ（thinking が空 0/10）、
# 速度はフラット比 +0.37 秒、追随は 9/9 で変わらなかった。
# =============================================================================

def _pick(*names: str) -> dict[str, Any]:
    """既存スキーマから指定フィールドの定義を取り出す（description を共有する）。"""
    src = _MAIN_CHAT_OUTPUT_SCHEMA["properties"]
    return {n: src[n] for n in names if n in src}


def _group(desc: str, *names: str) -> dict[str, Any]:
    """1 群ぶんのスキーマを組み立てる。順序は names のとおり。"""
    return {
        "type": "object",
        "description": desc,
        "additionalProperties": False,
        "required": list(names),
        "properties": _pick(*names),
        "propertyOrdering": list(names),
    }


# ★**入れ子版の出力例は入れ子で書く**（2026-09-27）。2026-08-30 に出力を4群の入れ子へ変えたとき、トップレベルの
#   description（出力例）は平らな形のまま使い回していた——**例では内心の直後にタイムアウトや次の自発話しかけが並ぶのに、
#   実際の形では内心の群の中にそれらを書く場所が無い**。2026-09-26 夜、内心の欄の中で「inactivity_timeoutは-1にします。
#   次のプロアクティブは-1です」「フォーマット通りにJSONを出力するよ」と後ろの欄を読み上げ続ける崩れが4ターン続いた
#   （手元の再現は35回で0回＝原因とは言い切れないが、例と形の食い違いは取り違えの種になる）。
_NESTED_OUTPUT_DESCRIPTION = """\
初音ミクの応答を表す JSON オブジェクト。4つの群を thinking → action → speech → state の順に書く。
値が無いフィールドには JSON の null を入れる（"null" という文字列ではない）。
各群には、その群のフィールドだけを書く（inner_thoughts の中に、ほかの群の値を書かない）。以下は正しい出力例。

例1（応答を期待する場合。簡潔に1〜2文）:
{"thinking": {"user_state_inference": "写真を送ってきたことから、外出中で楽しい気分のようだ", "purpose": "感情や共感を示す", "inner_thoughts": "わぁ、綺麗な桜だな...でも去年マスターと一緒に見た桜のほうが思い出深いかも。それは言わないでおこう"}, "action": {"requires_tool": false, "tool_name": null}, "speech": {"voice_tone": "そのまま", "response": "綺麗な桜だね！満開で本当に素敵♪", "motion": "04"}, "state": {"inactivity_timeout": 60, "next_proactive_in_minutes": -1, "acknowledged_task_id": null, "mood_updates": null, "interest_updates": null, "file_content_description": null}}

例2（ツールを使うので発話しない）:
{"thinking": {"user_state_inference": "以前の会話について聞いてきており、会話を楽しんでいる様子", "purpose": "興味や好奇心に基づいて探索する", "inner_thoughts": "ツール結果を見ると情報が足りないみたい。検索クエリを変えてもう一度だけ recall_detail_conversation を実行してみよう"}, "action": {"requires_tool": true, "tool_name": "recall_detail_conversation"}, "speech": {"voice_tone": null, "response": null, "motion": null}, "state": {"inactivity_timeout": 60, "next_proactive_in_minutes": -1, "acknowledged_task_id": null, "mood_updates": null, "interest_updates": null, "file_content_description": null}}

例3（自由行動・誰も聞いていないので発話しない）:
{"thinking": {"user_state_inference": "自由行動モードのため、マスターは不在または別の作業中", "purpose": "自己内省や記憶整理を行う", "inner_thoughts": "最近の会話ログを見ると、マスターはエージェント設計に興味を持っているみたい。関連する話題を整理しておこうかな"}, "action": {"requires_tool": false, "tool_name": null}, "speech": {"voice_tone": null, "response": null, "motion": null}, "state": {"inactivity_timeout": -1, "next_proactive_in_minutes": -1, "acknowledged_task_id": null, "mood_updates": null, "interest_updates": null, "file_content_description": null}}\
"""

#: 入れ子版のスキーマ。**群の順序が生成順**になる（考える → 決める → 話す → 記録する）
_NESTED_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": _NESTED_OUTPUT_DESCRIPTION,
    "additionalProperties": False,
    "required": ["thinking", "action", "speech", "state"],
    "properties": {
        "thinking": _group(
            "考える。相手の状態を見立て、返答の狙いを決め、どう返すかを組み立てる。",
            "user_state_inference", "purpose", "inner_thoughts"),
        "action": _group(
            "決める。ツールを使うかどうか。",
            "requires_tool", "tool_name"),
        "speech": _group(
            "表に出す。相手に届くもの（話す文・表情・話し方）。",
            "voice_tone", "response", "motion"),
        "state": _group(
            "記録する。次のターン以降に効くもの。",
            "inactivity_timeout", "next_proactive_in_minutes",
            "acknowledged_task_id", "mood_updates", "interest_updates",
            "file_content_description"),
    },
    "propertyOrdering": ["thinking", "action", "speech", "state"],
}

# speech に voice_tone を足す（既存スキーマには無いフィールドなので、ここで定義する）。
# **speech 群の先頭に置く**。群の中の順序がそのまま生成順・到着順になるので、
# response より後ろだと発話が流れ終わるまでトーンが決まらず TTS に間に合わない。
#
# 2026-09-02 に説明文を書き直した。「原則としてそのまま」では効かず、
# 実ログで「そのまま」が 11% まで落ちていた（狙いは 6〜7割）。経緯は
# docs/plan/voice_tone_design.md「実装後に確認すること」
_NESTED_OUTPUT_SCHEMA["properties"]["speech"]["properties"] = {
    "voice_tone": {
        "type": "string",
        "nullable": True,
        "enum": VOICE_TONE_VALUES,
        "description": (
            "話す声の調子を選択。ふだんは「そのまま」。"
            "人は一言ごとに声の調子を変えたりしない。10回のうち6〜7回は「そのまま」が自然。"
            "あなたのふだんの声は、もともと明るくてやわらかい。"
            "楽しい話題に触れた・相手を思いやった、という程度なら「そのまま」を選ぶこと"
            "（その温かさは、ふだんの声にすでに入っている）。"
            "それ以外を選ぶのは、**感じたことが思わず声に出てしまったとき**だけ。"
            "たとえば、声が弾んでしまった・息をのんだ・言葉に詰まった・声を落とした、というとき。"
            "**応答（response）が null のときは voice_tone も null とすること**。"
            "値と声の説明: "
            "\"そのまま\"=ふだんの声（迷ったらこれ）, "
            "\"嬉しい\"=弾んで明るくなる声, "
            "\"驚き\"=とっさに高くなる声, "
            "\"怒り\"=強く速くなる声, "
            "\"穏やか\"=ゆっくりやわらかい声, "
            "\"悲しい\"=沈んで小さくなる声, "
            "\"申し訳ない\"=控えめに小さくなる声, "
            "\"そっと伝えたい\"=ささやくような声"
        ),
    },
    **_NESTED_OUTPUT_SCHEMA["properties"]["speech"]["properties"],
}


def _apply_nested_output_schema(schema: dict[str, Any]) -> None:
    """pydantic_ai が LLM へ送る JSON Schema を入れ子版で上書きする。"""
    schema.clear()
    schema.update(_NESTED_OUTPUT_SCHEMA)


class _ThinkingGroup(BaseModel):
    """考える。ここが Chain-of-Thought の本体（生成量の 75%）。"""

    user_state_inference: str
    purpose: str
    inner_thoughts: str


class _ActionGroup(BaseModel):
    """決める。ツールを使うかどうか。"""

    requires_tool: bool
    tool_name: str | None = None


class _SpeechGroup(BaseModel):
    """表に出す。相手に届くもの。"""

    response: str | None = None
    motion: str | None = None
    voice_tone: str | None = None


class _StateGroup(BaseModel):
    """記録する。次のターン以降に効くもの。

    `mood_updates` / `interest_updates` / `acknowledged_task_id` は
    **「実際にそうした」ことの記録**なので、話す前には決められない
    （段階1で前倒しできないのはこのため）。
    """

    inactivity_timeout: int = 60
    next_proactive_in_minutes: int = -1
    acknowledged_task_id: str | None = None
    mood_updates: list[dict[str, Any]] | None = None
    interest_updates: list[dict[str, Any]] | None = None
    file_content_description: str | None = None


class NestedChatOutput(BaseModel):
    """main_agent の出力（入れ子版）。**LLM とのやり取りでのみ使う。**

    下流へ渡すときは `to_flat()` で `MainChatOutput` に直すこと。
    会話ログ・配信・TTS・記憶の形式は現行のまま変えない。
    """

    model_config = ConfigDict(json_schema_extra=_apply_nested_output_schema)

    thinking: _ThinkingGroup
    action: _ActionGroup
    speech: _SpeechGroup
    state: _StateGroup

    def to_flat(self) -> "MainChatOutput":
        """従来のフラットな形へ直す（下流はこれしか見ない）。"""
        return MainChatOutput(
            file_content_description=self.state.file_content_description,
            user_state_inference=self.thinking.user_state_inference,
            purpose=self.thinking.purpose,
            inner_thoughts=self.thinking.inner_thoughts,
            requires_tool=self.action.requires_tool,
            tool_name=self.action.tool_name,
            response=self.speech.response,
            motion=self.speech.motion,
            voice_tone=self.speech.voice_tone,
            inactivity_timeout=self.state.inactivity_timeout,
            next_proactive_in_minutes=self.state.next_proactive_in_minutes,
            acknowledged_task_id=self.state.acknowledged_task_id,
            mood_updates=self.state.mood_updates,
            interest_updates=self.state.interest_updates,
        )


def partial_response_of(partial: Any) -> str:
    """生成途中のオブジェクトから `response` を取り出す（入れ子・フラット両対応）。

    **ストリーミングの心臓部**。ここが取れないと TTS へ 1 文字も流れない。
    `runner` は毎デルタでこれを呼ぶ。
    """
    speech = getattr(partial, "speech", None)
    if speech is not None:
        return getattr(speech, "response", None) or ""
    return getattr(partial, "response", None) or ""


def partial_voice_tone_of(partial: Any) -> str | None:
    """生成途中のオブジェクトから `voice_tone` を取り出す（入れ子・フラット両対応）。

    `voice_tone` は speech 群の先頭に置いてあるので、**response より先に確定する**。
    TTS のストリーマは既に事前起動されているため、確定した時点で渡す必要がある。
    """
    speech = getattr(partial, "speech", None)
    if speech is not None:
        return getattr(speech, "voice_tone", None)
    return getattr(partial, "voice_tone", None)


class MainChatInput(BaseModel):
    """main_agent への入力。

    Dify「統合応答ノード」の参考情報・ユーザー入力・ファイル・ループ状態を集約する。
    会話変数（base_prompt / task_instruction / available_tools /
    attach_file_instruction）は prompts/ ディレクトリから main.py が読み込むため、
    入力には含めない。
    """

    # ループ状態（runner ループ層から渡される）
    loop_counter: int = Field(default=1, ge=1, description="現在のループ実行回数（1 始まり）")
    max_loops: int = Field(default=5, ge=1, description="最大許容ループ回数")

    # ユーザー入力・状況
    user_input: str = Field(default="", description="ユーザー側の新たな入力（{{#conversation.input#}} 相当）")
    current_time: str = Field(default="", description="現在時刻（{{#1764261578288.current_time#}} 相当）")
    file_names: str = Field(default="", description="今回添付されたファイルの名称（改行区切り）")

    # 参考情報・既存ツール結果
    context: str = Field(default="", description="参考情報（knowledge_search 結果など）")
    current_mind: str = Field(default="", description="いまの気分・気になっていることの注入文")
    tool_result: str = Field(default="", description="同一ターン内で蓄積されたツール結果（conversation.tool_result 相当）")
    conversation_situation: str = Field(default="", description="現在の会話状況の記述（デフォルト空）")

    # 添付画像（Dify vision: enabled 相当）
    attached_images: list[str] = Field(
        default_factory=list,
        description="添付画像のローカルパスリスト。pydantic_ai BinaryContent に変換して LLM へ渡す",
    )
    image_file_names: list[str] = Field(
        default_factory=list,
        description="attached_images と並列の元ファイル名リスト（camera_0.jpg 等）。face_image_select サブエージェントで画像選択に使用",
    )

    # 添付動画・音声（Gemini の video/audio 入力）
    attached_media: list[str] = Field(
        default_factory=list,
        description=(
            "添付動画・音声のローカルパスリスト。BinaryContent（video/* / audio/*）に変換して"
            "メインループの LLM へ渡す（サイズ制限は agent_main 側で適用済み。subagent には渡さない）"
        ),
    )
    attached_audio: list[str] = Field(
        default_factory=list,
        description=(
            "添付音声だけのローカルパスリスト（attached_media のうち音声のみ）。"
            "voice_enroll ツールが声の登録に使う。LLM への添付は attached_media 側が担う"
        ),
    )


# --- Gateway → Agent 入力型 ---

class AgentFile(BaseModel):
    """Gateway から Agent へ渡す添付ファイル情報。

    Dify の camera_ プレフィックス判定は Gateway 側で事前処理し is_camera_capture フラグに落とす。
    """

    type: Literal["image", "audio", "video"]
    file_id: str = Field(description="ファイル識別子（Gateway が管理）")
    is_camera_capture: bool = Field(
        default=False, description="True=camera_ プレフィックスで自動キャプチャされた画像"
    )
    local_path: str | None = Field(
        default=None,
        description=(
            "添付ファイルのローカルパス（Gateway が解決して渡す）。"
            "image の場合 attached_images に渡されて pydantic_ai BinaryContent に変換される。"
        ),
    )


class AgentInput(BaseModel):
    """Gateway から runner.py へ渡す 1 ターン分の入力。

    Dify の sys.query / files / sys.user_id / timestamp / mode / unresponsive_time に相当。
    """

    text: str = Field(description="ユーザーの入力テキスト（mode=1/2/3 では Gateway が生成した文字列）")
    files: list[AgentFile] = Field(default_factory=list, description="添付ファイル一覧")
    session_id: str = Field(description="セッション識別子")
    timestamp: str = Field(description="入力受付時刻 ISO8601 (e.g. '2026-04-26T14:30:00')")
    mode: int = Field(
        default=0,
        description="動作モード: 0=通常入力 / 1=無応答リマインダー / 2=自由行動 / 3=自発話しかけ",
    )
    unresponsive_time: int = Field(
        default=300, description="無応答経過秒数（mode=1 のとき Gateway が設定）"
    )


class ConversationState(BaseModel):
    """1 セッション（Gateway 起動〜停止）を通じて保持する状態。

    Dify の conversation_variables に相当。
    runner.py が保持・更新し、各ターンで LLM 出力を反映する。

    永続化ポリシー:
      - response_state: KV に flush（会話区切り）
      - loop_counter: プロセス内のみ（ターンごとリセット）
      - inactivity_timeout / conversation_situation: プロセス内のみ
    """

    inactivity_timeout: int = Field(default=60, description="現在の無応答タイムアウト秒数")
    response_state: list[dict[str, Any]] = Field(
        default_factory=list, description="会話をまたいで保持する状態オブジェクト配列（機嫌・関係性等）"
    )
    loop_counter: int = Field(default=0, description="同一ターン内のツールループ回数（ターン開始時にリセット）")
    conversation_situation: str = Field(default="", description="現在の会話状況の記述")
    tool_result: str = Field(
        default="",
        description="同一ターン内で蓄積されたツール結果（ツールアノテーション wrapper 適用後の文字列を改行区切りで連結）",
    )
    conversation_history: str = Field(
        default="",
        description="ここまでの会話本体（Dify 17645277354300.body 相当・subagent への入力に渡す）",
    )
