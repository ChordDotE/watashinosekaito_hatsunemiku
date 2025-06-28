"""
LangMemで使用するデータクラスの定義

このモジュールでは、LangMemシステムで使用するデータ構造をPydanticを使って定義します。
これらのクラスは、記憶データベースの構造を表現し、JSONとの相互変換を容易にします。
拡張フィールド（extensions）は動的に処理され、事前定義なしで様々な拡張データを扱えます。
"""

from typing import List, Dict, Optional, Union, Type, TypeVar, Generic, Callable
from pydantic import BaseModel, Field, conint, confloat
import json
import datetime
from pathlib import Path


# 補助クラス

class ParticipantProfile(BaseModel):
    """参加者のプロフィール情報"""
    basic_info: Dict[str, str] = Field(default_factory=dict, description="プロフィール情報")
    # extended_attributes: Dict[str, object] = Field(default_factory=dict, description="拡張属性情報")


# class EmotionalContext(BaseModel):
#     """感情コンテキスト情報"""
#     emotion: str = Field(..., description="感情の種類")
#     intensity: float = Field(..., ge=0.0, le=1.0, description="感情の強度（0.0～1.0）")
#     cause: str = Field(..., description="感情の原因")


# class MediaItem(BaseModel):
#     """メディア項目（画像、音声、動画など）"""
#     type: str = Field(..., description="メディアのタイプ（'image', 'audio', 'video'など）")
#     content_description: str = Field(..., description="メディアの内容説明")
#     file_reference: str = Field(..., description="ファイルの参照パス")
#     duration_seconds: Optional[int] = Field(None, description="メディアの長さ（秒）")
#     captured_by: Optional[str] = Field(None, description="撮影者/録音者のID")
#     capture_device: Optional[str] = Field(None, description="撮影/録音に使用したデバイス")
#     created_by: Optional[str] = Field(None, description="作成者のID")



class Activity(BaseModel):
    """活動データ"""
    time: str = Field(..., description="（現在時刻ではなく）会話文のタイムスタンプから読み取れる活動の日付（YYYY-MM-DD形式）")
    description: str = Field(..., description="活動の説明")
    # participants: Optional[List[str]] = Field(default_factory=list, description="`master`活動の参加者IDリスト")
    # details: Optional[str] = Field(None, description="活動の詳細情報")


# エピソード記憶関連クラス

class EpisodicMemory(BaseModel):
    """エピソード記憶（特定の出来事や体験の記憶）"""
    # 基本識別情報
    # memory_id: str = Field(..., description="記憶の一意識別子")
    
    # 時間情報
    start_time: str = Field(..., description="（現在時刻ではなく）会話文のタイムスタンプから読み取れる記憶の開始時間（YYYY-MM-DDThh:mm形式）")
    end_time: str = Field(..., description="（現在時刻ではなく）会話文のタイムスタンプから読み取れる記憶の終了時間（YYYY-MM-DDThh:mm形式）")
    # duration_minutes: int = Field(..., ge=0, description="記憶の継続時間（分）")
    
    # 場所と活動
    location: Optional[str] = Field(..., description="会話した場所が家以外の場合のみ、会話場所を記録する。特に言及がなければ家で会話したものとして扱う。")
    participants: Optional[List[str]]= Field(..., description="master,Miku以外の参加者がいた場合に参加者IDのリストを記載")
    
    # 内容情報
    summary: str = Field(..., description="記憶・会話の説明。会話や行動などの内容がすべてわかるよう、詳細に記録すること")
    activities: Optional[List[Activity]] = Field(None, description="活動データリスト。日常の会話・行動では最大2個、外出・旅行等では最大5個に制限すること。会話のみの事項は記載せず、行動のみを記載する")
    insights: Optional[List[str]] = Field(None, description="洞察リスト")
    future_improvements: Optional[List[str]] = Field(None, description="将来の改善点リスト")
    
    # 感情情報
    emotion: str = Field(..., description="各人が感じていた感情")
    
    # メディア情報
    # media: List[MediaItem] = Field(default_factory=list, description="関連メディアリスト")
    
    # 重要度と信頼性
    importance: float = Field(..., ge=0.0, le=1.0, description="重要度（0.0～1.0）。会話の内容や、あとから振り返られた回数に応じて決まる。")
    
    # 検索・想起関連
    # recall_count: int = Field(0, ge=0, description="想起回数")
    # last_recalled: str = Field("", description="最後に想起した時間（ISO 8601形式）")
    # retrieval_count: int = Field(0, ge=0, description="検索回数")
    
    # 関連記憶
    # associated_episodic_ids: List[str] = Field(default_factory=list, description="関連するエピソード記憶IDリスト")
    # related_memories: List[str] = Field(default_factory=list, description="関連する記憶IDリスト")
    
    # 拡張情報
    extensions: Dict[str, str] = Field(default_factory=dict, description="拡張フィールド（最大3項目に絞って記録。天気、交通手段、作曲詳細など）")


# 意味記憶関連クラス

class InterestCategory(BaseModel):
    """興味カテゴリ"""
    category: str = Field(..., description="カテゴリ名")
    items: List[str] = Field(..., description="カテゴリ内のアイテムリスト")
    # confidence: float = Field(..., ge=0.0, le=1.0, description="信頼度（0.0～1.0）")
    # last_updated: str = Field(..., description="最終更新時間（YYYY-MM-DDThh:mm形式）")


class Relationship(BaseModel):
    """人間関係"""
    person_id: str = Field(..., description="関係者のID")
    relationship_type: str = Field(..., description="関係の種類")
    closeness: float = Field(..., ge=0.0, le=1.0, description="親密度（0.0～1.0）")
    known_since: str = Field(..., description="知り合った日付")
    shared_activities: List[str] = Field(..., description="共有活動リスト")
    last_interaction: str = Field(..., description="最後の交流時間（YYYY-MM-DDThh:mm形式）")
    notes: str = Field(..., description="備考")


class ImportantDate(BaseModel):
    """重要な日付"""
    date: str = Field(..., description="日付")
    event: str = Field(..., description="イベント名")
    significance: str = Field(..., description="重要性の説明")
    # recurrence: str = Field(..., description="繰り返しパターン（'yearly', 'monthly', 'none'など）")


class VisitedPlace(BaseModel):
    """訪問した場所"""
    name: str = Field(..., description="場所の名前（自宅は記録せず、それ以外の場所を訪れた際のみ記録する）")
    # coordinates: Dict[str, float] = Field(..., description="座標情報（緯度・経度）")
    visit_date: str = Field(..., description="訪問日")
    activities: Optional[List[str]] = Field(..., description="活動リスト")
    user_impression: str = Field(..., description="ユーザーの印象")
    # confidence: float = Field(..., ge=0.0, le=1.0, description="信頼度（0.0～1.0）")


# class ContextualInfo(BaseModel):
#     """文脈情報"""
#     recent_activity: str = Field(..., description="最近の活動")
#     goals_or_challenges: str = Field(..., description="目標または課題")
#     health_status: Dict[str, str] = Field(..., description="健康状態")


class UserProfile(BaseModel):
    """ユーザープロファイル"""
    basic_info: List[str] = Field(..., description="基本情報")
    preferences: Dict[str, List[str]] = Field(default_factory=dict, description="好みの情報（カテゴリ名: アイテムリスト）")
    relationships: Optional[List[Relationship]] = Field(..., description="人間関係リスト")
    important_dates: List[ImportantDate] = Field(..., description="重要な日付リスト")
    visited_places: Optional[List[VisitedPlace]] = Field(..., description="訪問した場所リスト")
    extensions: List[str] = Field(default_factory=dict, description="追加情報")

    recent_activity: Optional[str] = Field(..., description="最近の活動")
    goals_or_challenges: Optional[str] = Field(..., description="目標または課題")
    health_status: Optional[List[str]] = Field(..., description="健康状態")


class AgentProfile(BaseModel):
    """エージェントプロファイル"""
    name: str = Field(..., description="名前")
    created_on: str = Field(..., description="作成日（YYYY-MM-DDThh:mm形式）")
    version: str = Field(..., description="バージョン")
    developer: str = Field(..., description="開発者")
    personality_traits: List[str] = Field(..., description="性格特性リスト")
    interests: List[str] = Field(..., description="興味リスト")
    skills: Optional[List[str]] = Field(..., description="スキルリスト")
    beliefs: Dict[str, str] = Field(..., description="信念")
    # extended_attributes: Dict[str, str] = Field(..., description="拡張属性")
    extensions: Dict[str, object] = Field(..., description="拡張フィールド")


# class WorldKnowledge(BaseModel):
#     """世界知識"""
#     music_genres: Dict[str, Dict[str, object]] = Field(..., description="音楽ジャンル情報")
#     locations: Dict[str, Dict[str, object]] = Field(..., description="場所情報")
#     extensions: Dict[str, object] = Field(default_factory=dict, description="拡張フィールド")

class Agreement(BaseModel):
    """合意事項"""
    agreement_id: str = Field(..., description="合意の一意識別子")
    title: str = Field(..., description="タイトル")
    participants: List[str] = Field(..., description="参加者IDリスト")
    currentStatus: str = Field(..., description="現在の状態")
    # changeLog: List[ChangeLog] = Field(..., description="変更ログリスト")


class SemanticMemories(BaseModel):
    """意味記憶（概念や事実の記憶）"""
    user_profiles: Dict[str, UserProfile] = Field(..., description="ユーザープロファイル")
    agent_profiles: Dict[str, AgentProfile] = Field(..., description="エージェントプロファイル")
    agreements: List[Agreement] = Field(default_factory=list, description="合意事項リスト")
    # world_knowledge: WorldKnowledge = Field(..., description="世界知識")


# 手続き記憶関連クラス

class Observation(BaseModel):
    """観察データ"""
    date: str = Field(..., description="観察日時（YYYY-MM-DDThh:mm形式）")
    description: str = Field(..., description="観察内容")


class BehavioralPattern(BaseModel):
    """行動パターン"""
    pattern_id: str = Field(..., description="パターンの一意識別子")
    owner_id: str = Field(..., description="所有者ID")
    pattern_name: str = Field(..., description="パターン名")
    pattern_type: str = Field(..., description="パターンタイプ（'routine', 'habit', 'workflow'など）")
    description: str = Field(..., description="説明")
    observations: List[Observation] = Field(..., description="観察データリスト")
    triggers: List[str] = Field(..., description="トリガーリスト")
    consequences: List[str] = Field(..., description="結果リスト")
    steps: Optional[List[str]] = Field(None, description="ステップリスト")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="信頼度（0.0～1.0）")
    last_updated: str = Field("", description="最終更新時間（YYYY-MM-DDThh:mm形式）")


class Routine(BaseModel):
    """ルーチン"""
    routine_id: str = Field(..., description="ルーチンの一意識別子")
    owner_id: str = Field(..., description="所有者ID")
    name: str = Field(..., description="ルーチン名")
    typical_sequence: List[str] = Field(..., description="典型的な順序")
    frequency: str = Field(..., description="頻度")
    variations: List[str] = Field(..., description="バリエーションリスト")
    confidence: float = Field(..., ge=0.0, le=1.0, description="信頼度（0.0～1.0）")
    last_updated: str = Field(..., description="最終更新時間（YYYY-MM-DDThh:mm形式）")


class Skill(BaseModel):
    """スキル"""
    skill_id: str = Field(..., description="スキルの一意識別子")
    # owner_id: str = Field(..., description="所有者ID")
    name: str = Field(..., description="スキル名")
    process: List[str] = Field(..., description="プロセスリスト")
    # templates: Dict[str, object] = Field(..., description="テンプレート")
    extensions: Dict[str, str] = Field(default_factory=dict, description="拡張フィールド")
    mastery_level: float = Field(..., ge=0.0, le=1.0, description="習熟度（0.0～1.0）")
    last_used: str = Field(..., description="最終使用時間（YYYY-MM-DDThh:mm形式）")
    # voice_banks: Optional[List[str]] = Field(None, description="ボイスバンクリスト")


class ProceduralMemories(BaseModel):
    """手続き記憶（スキルや行動パターンの記憶）"""
    behavioral_patterns: List[BehavioralPattern] = Field(..., description="行動パターンリスト")
    routines: List[Routine] = Field(..., description="ルーチンリスト")
    skills: Dict[str, Skill] = Field(..., description="スキル")


# ワーキングメモリ関連クラス

class Task(BaseModel):
    """タスク"""
    # taskId: str = Field(..., description="タスクの一意識別子")
    title: str = Field(..., description="タイトル")
    description: str = Field(..., description="説明")
    dueDate: Optional[str] = Field(..., description="期限（YYYY-MM-DDThh:mm形式）")
    status: str = Field(..., description="状態")
    priority: str = Field(..., description="優先度")
    # createdAt: datetime.datetime = Field(..., description="作成時間")
    completedDte: Optional[str] = Field(None, description="完了時間（YYYY-MM-DDThh:mm形式）")


class WorkingMemory(BaseModel):
    """ワーキングメモリ（短期記憶）"""
    todo_list: List[Task] = Field(default_factory=list, description="TODOリスト")


# # 連想記憶関連クラス

# class RelatedConcept(BaseModel):
#     """関連概念"""
#     concept: str = Field(..., description="概念")
#     strength: float = Field(..., ge=0.0, le=1.0, description="関連強度（0.0～1.0）")
#     relationship_type: str = Field(..., description="関係タイプ")


# class ConceptNetwork(BaseModel):
#     """概念ネットワーク"""
#     network_id: str = Field(..., description="ネットワークの一意識別子")
#     central_concept: str = Field(..., description="中心概念")
#     related_concepts: List[RelatedConcept] = Field(..., description="関連概念リスト")
#     last_activated: str = Field(..., description="最終活性化時間（YYYY-MM-DDThh:mm形式）")


# class MemoryLink(BaseModel):
#     """記憶リンク"""
#     link_id: str = Field(..., description="リンクの一意識別子")
#     source_id: str = Field(..., description="ソースID")
#     target_id: str = Field(..., description="ターゲットID")
#     link_type: str = Field(..., description="リンクタイプ")
#     strength: float = Field(..., ge=0.0, le=1.0, description="強度（0.0～1.0）")
#     description: str = Field(..., description="説明")


# class AssociativeMemory(BaseModel):
#     """連想記憶"""
#     concept_networks: List[ConceptNetwork] = Field(..., description="概念ネットワークリスト")
#     memory_links: List[MemoryLink] = Field(..., description="記憶リンクリスト")


# ユーザー体験関連クラス

# class ChangeLog(BaseModel):
#     """変更ログ"""
#     timestamp: str = Field(..., description="タイムスタンプ（YYYY-MM-DDThh:mm形式）")
#     description: str = Field(..., description="説明")
#     reason: str = Field(..., description="理由")





# class Diary(BaseModel):
#     """日記"""
#     diary_id: str = Field(..., description="日記の一意識別子")
#     date: str = Field(..., description="日付(登録実施日ではなく、会話のタイムスタンプから判断すること)")
#     title: str = Field(..., description="タイトル")
#     content: str = Field(..., description="内容")
#     masterMood: str = Field(..., description="マスターの気分")
#     mikuMood: str = Field(..., description="ミクの気分")




# class UserExperience(BaseModel):
#     """ユーザー体験"""
#     agreements: List[Agreement] = Field(..., description="合意事項リスト")
#     diary: List[Diary] = Field(..., description="日記リスト")
#     todo_list: List[Task] = Field(..., description="TODOリスト")


# # メタデータ関連クラス

# class MemoryHealth(BaseModel):
#     """記憶の健全性"""
#     episode_count: int = Field(..., ge=0, description="エピソード数")
#     semantic_fact_count: int = Field(..., ge=0, description="意味的事実数")
#     pattern_count: int = Field(..., ge=0, description="パターン数")
#     average_confidence: float = Field(..., ge=0.0, le=1.0, description="平均信頼度（0.0～1.0）")


# class RetrievalStatistics(BaseModel):
#     """検索統計"""
#     most_accessed_category: str = Field(..., description="最もアクセスされたカテゴリ")
#     least_accessed_category: str = Field(..., description="最もアクセスされていないカテゴリ")
#     total_retrievals: int = Field(..., ge=0, description="総検索回数")


# class ForgettingCurve(BaseModel):
#     """忘却曲線"""
#     decay_rate: float = Field(..., ge=0.0, le=1.0, description="減衰率（0.0～1.0）")
#     reinforcement_threshold: float = Field(..., ge=0.0, le=1.0, description="強化閾値（0.0～1.0）")


# class VectorDBInfo(BaseModel):
#     """ベクトルDB情報"""
#     engine: str = Field(..., description="使用エンジン")
#     embedding_model: str = Field(..., description="埋め込みモデル")
#     total_vectors: int = Field(..., ge=0, description="ベクトル総数")
#     last_indexed: str = Field(..., description="最終インデックス時間（YYYY-MM-DDThh:mm形式）")


# class StorageUsage(BaseModel):
#     """ストレージ使用量"""
#     memory_data_size: str = Field(..., description="メモリデータのサイズ")
#     vector_db_size: str = Field(..., description="ベクトルDBのサイズ")
#     media_references_size: str = Field(..., description="メディア参照のサイズ")


# class SystemInfo(BaseModel):
#     """システム情報"""
#     version: str = Field(..., description="バージョン")
#     last_updated: str = Field(..., description="最終更新時間（YYYY-MM-DDThh:mm形式）")
#     storage_usage: StorageUsage = Field(..., description="ストレージ使用量")


# class MemoryMetadata(BaseModel):
#     """記憶メタデータ"""
#     last_consolidated: str = Field(..., description="最終統合時間（ISO 8601形式）")
    # memory_health: MemoryHealth = Field(..., description="記憶の健全性")
    # retrieval_statistics: RetrievalStatistics = Field(..., description="検索統計")
    # forgetting_curve: ForgettingCurve = Field(..., description="忘却曲線")
    # vector_db_info: VectorDBInfo = Field(..., description="ベクトルDB情報")
    # system_info: SystemInfo = Field(..., description="システム情報")


# class Participant(BaseModel):
#     """会話や記憶に関わる参加者（ユーザーやAIエージェント）"""
#     name: str = Field(..., description="参加者の名前")
#     type: str = Field(..., description="参加者のタイプ（'human'または'ai'）")
#     role: str = Field(..., description="参加者の役割（'master', 'assistant', 'guest', 'companion'など）")
#     profile: ParticipantProfile = Field(default_factory=ParticipantProfile, description="参加者のプロフィール情報")
#     extensions: Dict[str, str] = Field(default_factory=dict, description="拡張フィールド")


class MemorySystem(BaseModel):
    """記憶システム全体"""
    episodic_memories: List[EpisodicMemory] = Field(..., description="エピソード記憶リスト")
    semantic_memories: SemanticMemories = Field(..., description="意味記憶")
    procedural_memories: ProceduralMemories = Field(..., description="手続き記憶")
    working_memory: WorkingMemory = Field(..., description="ワーキングメモリ")
    # associative_memory: AssociativeMemory = Field(..., description="連想記憶")
    # memory_metadata: MemoryMetadata = Field(..., description="記憶メタデータ")
    
    @classmethod
    def create_empty_memory_system(cls) -> 'MemorySystem':
        """
        最小限の初期データだけで記憶システムを作成
        
        Returns:
            初期化された空の記憶システム
        """
        import uuid
        current_time = datetime.datetime.now().isoformat()
        
        # 空の記憶システム
        return cls(
            episodic_memories=[],  # 空のエピソード記憶リスト
            semantic_memories=SemanticMemories(
                user_profiles={},
                agent_profiles={},
                agreements=[]
            ),
            procedural_memories=ProceduralMemories(
                behavioral_patterns=[],
                routines=[],
                skills={},
                todo_list=[]
            ),
            working_memory=WorkingMemory(
                todo_list=[]
            )
            # associative_memory=AssociativeMemory(
            #     concept_networks=[],
            #     memory_links=[]
            # ),
            # memory_metadata=MemoryMetadata(
            #     last_consolidated=current_time
            # )
        )


# 拡張フィールドを動的に処理するためのユーティリティ関数

# def create_extension_model(data: Dict[str, object]) -> Type[BaseModel]:
#     """
#     拡張データから動的にPydanticモデルを生成する
    
#     Args:
#         data: 拡張データの辞書
        
#     Returns:
#         動的に生成されたPydanticモデルクラス
#     """
#     fields = {}
#     for key, value in data.items():
#         if isinstance(value, dict):
#             fields[key] = (Dict[str, object], ...)
#         elif isinstance(value, list):
#             fields[key] = (List[object], ...)
#         elif isinstance(value, str):
#             fields[key] = (str, ...)
#         elif isinstance(value, int):
#             fields[key] = (int, ...)
#         elif isinstance(value, float):
#             fields[key] = (float, ...)
#         elif isinstance(value, bool):
#             fields[key] = (bool, ...)
#         else:
#             fields[key] = (object, ...)
    
#     return type('DynamicExtension', (BaseModel,), {'__annotations__': fields})


# def parse_extensions(data: Dict[str, object]) -> Dict[str, BaseModel]:
#     """
#     拡張データを解析し、適切なモデルに変換する
    
#     Args:
#         data: 拡張データの辞書
        
#     Returns:
#         モデル化された拡張データの辞書
#     """
#     result = {}
#     for key, value in data.items():
#         if isinstance(value, dict):
#             model_class = create_extension_model({key: value})
#             result[key] = model_class(**{key: value})
#         else:
#             result[key] = value
    
#     return result


'''
以下のクラスは、メモリ制限の都合で、メモリシステムとは別のクラスとして管理する。
'''

class Message(BaseModel):
    """会話メッセージ"""
    timestamp: str = Field(..., description="会話文のタイムスタンプから読み取れる、メッセージのタイムスタンプ（ISO 8601形式）")
    role: str = Field(..., description="メッセージの発話者（user/assitant）")
    content: str = Field(..., description="メッセージの本文")
    speaker_name: str = Field(..., description="発言者の名前（会話や文脈から特に指示がない場合は、人間のユーザーは全て「マスター」の発言であるものとする。）")

class Conversation(BaseModel):
    """会話データ"""
    language: str = Field(default="ja", description="会話の言語")
    description: str = Field(..., description="会話の説明。5W1Hなどの観点から、会話の内容を客観的に要約したもの")
    messages: List[Message] = Field(default_factory=list, description="会話メッセージのリスト")
    participant: str = Field(default="マスター, 初音ミク", description="会話参加者の名前（カンマ区切りの文字列）")
    start_time: str = Field(..., description="会話文のタイムスタンプから読み取れる会話の開始時間（ISO 8601形式形式の文字列）")
    end_time: str = Field(..., description="会話文のタイムスタンプから読み取れる会話の終了時間（ISO 8601形式形式の文字列）")
    
    @classmethod
    def create_empty_conversation(cls) -> 'Conversation':
        """空のConversationオブジェクトを作成する"""
        return cls(
            language="ja",  # 日本語を設定
            description="",    # 会話の説明を空白で初期化
            messages=[],     # 空のメッセージリスト
            participant="マスター, 初音ミク",   # 仮のデフォルト値を設定  
            start_time="",
            end_time=""
        )
