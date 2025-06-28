"""
LangMemを使用して会話ファイルを処理するモジュール

このモジュールでは、langmemパッケージを使用して会話ファイルから情報を抽出し、
LangMemのデータ構造に格納します。MemorySystemクラスを使用して、
意味記憶や手続き記憶など全てを含んだクラスをまとめて更新します。
"""

import re
import json
import datetime
import os
import uuid
import pickle
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from langmem import create_memory_manager
from models.memory_data_class import *
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from models.config_manager import ConfigManager
from utils.path_config import PathConfig

def setup_api_keys(use_openrouter: bool = True) -> ChatOpenAI:
    """
    APIキーを設定し、使用するチャットモデルを初期化する
    
    Args:
        use_openrouter: OpenRouterを使用するかどうか（Trueの場合はOpenRouter、Falseの場合はOpenAI）
        
    Returns:
        初期化されたチャットモデル
    """
    # 設定を読み込み
    path_config = PathConfig.get_instance()
    
    # settings.jsonからAPIキーとモデル情報を読み込む
    settings_path = path_config.settings_file
    with open(str(settings_path), 'r') as f:
        settings = json.load(f)
    
    # OpenRouterのAPI情報を取得
    api_config = settings.get("api", {}).get("openrouter", {})
    api_url = api_config.get("url", "https://openrouter.ai/api/v1/chat/completions")
    
    # URLから"/chat/completions"を除去する
    if api_url.endswith("/chat/completions"):
        api_url = api_url[:-len("/chat/completions")]
    
    api_key = api_config.get("api_key", "")
    model = api_config.get("models", {}).get("analysis")  # analysisモデルを使用
    
    # OpenRouterのチャットモデルを初期化
    openrouter_chat_model = ChatOpenAI(
        model=model,  # settings.jsonから取得したanalysisモデル
        api_key=api_key,
        base_url=api_url,
        temperature=0.1,
        # OpenRouterに必要な追加ヘッダーを設定
        default_headers={
            "HTTP-Referer": "http://localhost:8000",  # あなたのサイトのURLに変更してください
            "X-Title": "My Application"  # あなたのアプリケーション名に変更してください
        }
    )
    
    # 環境変数にAPIキーを設定
    if use_openrouter:
        return openrouter_chat_model

# エンベディングモデルとベクトルストアの初期化
def initialize_vector_database() -> Chroma:
    """
    HuggingFaceEmbeddingsモデルとChromaベクトルストアを初期化する
    
    Returns:
        初期化されたChromaインスタンス
    """
    # パス設定を取得
    path_config = PathConfig.get_instance()
    
    # エンベディングモデルの設定 - multilingual-e5-largeを使用
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cuda"}, 
        encode_kwargs={"normalize_embeddings": True}  # E5モデルでは正規化が推奨されています
    )
    
    # 保存先ディレクトリの設定
    persist_directory = str(path_config.chroma_db_dir)
    
    # Chromaの初期化
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_name="conversation_store"
    )
    return vectorstore

# ベクトルストアに会話を保存する関数
def store_conversation(conversation: Conversation, vectorstore: Chroma) -> None:
    """会話をベクトルDBに保存
    
    Args:
        conversation: 保存する会話オブジェクト
        vectorstore: 使用するベクトルストア
    """
    # 会話をE5モデル向けのテキスト形式に変換
    message_texts = []

    message_texts.append("会話の概要: " + conversation.description)
    for msg in conversation.messages:
        message_texts.append(f"{msg.role}: {msg.content}(meta:  {msg.speaker_name} {msg.timestamp})")
    
    # E5モデル向けにフォーマット（'passage:' プレフィックスを使うとより良い結果が得られます）
    page_content = "passage: " + "\n".join(message_texts)
    
    # メタデータを準備
    metadata = {
        "language": conversation.language,
        "participant": conversation.participant,
        "start_time": conversation.start_time,
        "end_time": conversation.end_time,
        "description": conversation.description,
        "message_count": len(conversation.messages)
    }
    
    document = Document(
        page_content=page_content,
        metadata=metadata
    )
    
    vectorstore.add_documents([document])
    print(f"会話が保存されました: {len(conversation.messages)}件のメッセージ")

# 会話を検索する関数（E5モデル向けに最適化）
def search_conversations(query: str, filters: dict = None, k: int = 5) -> list:
    """会話を検索（E5モデル向けに最適化）
    
    内部でベクトルストアを初期化するため、単独で呼び出し可能
    
    Args:
        query: 検索クエリ
        filters: フィルタ条件（オプション）
        k: 返す結果の数（オプション、デフォルト5）
        
    Returns:
        検索結果のリスト
    """
    try:
        # ベクトルストアを内部で初期化
        vectorstore = initialize_vector_database()
        
        # E5モデルでは、クエリに 'query:' プレフィックスを付けると良い結果が得られます
        formatted_query = f"query: {query}"
        
        results = vectorstore.similarity_search(
            query=formatted_query,
            filter=filters,
            k=k
        )
        return results
    except Exception as e:
        print(f"会話検索エラー: {str(e)}")
        return []  # エラー時は空のリストを返す

def initialize_chroma_client():
    """
    エンベディングモデルを使わずにChromaクライアントを直接初期化する
    時系列などに沿って会話をベクトル検索せずに取得する場合に使用する
    
    Returns:
        初期化されたChromaクライアントとコレクション、またはNoneとNone
    """
    try:
        # パス設定を取得
        path_config = PathConfig.get_instance()
        
        # 保存先ディレクトリの設定
        persist_directory = str(path_config.chroma_db_dir)
        
        # Chromaクライアントを直接初期化
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # コレクションが存在するか確認し、存在しない場合は作成
        try:
            collection = chroma_client.get_collection("conversation_store")
        except Exception as e:
            if "Collection conversation_store does not exist" in str(e):
                # コレクションが存在しない場合は作成
                print("コレクション 'conversation_store' が存在しないため、新規作成します。")
                collection = chroma_client.create_collection("conversation_store")
            else:
                # その他のエラーの場合は再スロー
                raise
        
        return chroma_client, collection
    except Exception as e:
        print(f"Chromaクライアントの初期化エラー: {str(e)}")
        return None, None

def get_recent_conversations(limit: int = 5, sort_order: str = "asc") -> list:
    """
    直近の会話を取得し、指定された順序で並べる
    エンベディングモデルやベクトルストアの初期化を省略して高速化
    
    Args:
        limit: 取得する会話の最大数（デフォルト5）
        sort_order: ソート順（"asc"=古い順、"desc"=新しい順）
        
    Returns:
        会話データとメタデータのペアのリスト
    """
    try:
        # Chromaクライアントを直接初期化
        _, collection = initialize_chroma_client()
        
        # コレクションからすべてのデータを取得
        all_results = collection.get()
        
        if all_results and 'documents' in all_results and len(all_results['documents']) > 0:
            # メタデータを取得
            documents = all_results['documents']
            metadatas = all_results['metadatas']
            
            # 会話データとメタデータをペアにする
            conversation_pairs = list(zip(documents, metadatas))
            
            # 常に新しい順（desc）でソートして最新のlimit件を取得
            sorted_pairs = sorted(
                conversation_pairs, 
                key=lambda pair: pair[1].get('start_time', ''),
                reverse=True  # 常に新しい順でソート
            )
            
            # 指定された数だけ取得
            recent_pairs = sorted_pairs[:limit]
            
            # sort_orderが"asc"の場合は、取得したデータを古い順に並べ替える
            if sort_order.lower() == "asc":
                recent_pairs = sorted(
                    recent_pairs,
                    key=lambda pair: pair[1].get('start_time', ''),
                    reverse=False  # 古い順にソート
                )
            
            return recent_pairs
        else:
            return []
    except Exception as e:
        print(f"会話履歴取得エラー: {str(e)}")
        return []  # エラーが発生した場合は空のリストを返す

def parse_conversation_file(file_path: str) -> List[Dict[str, str]]:
    """
    会話ファイルを解析し、メッセージのリストに変換する
    
    Args:
        file_path: 会話ファイルのパス
        
    Returns:
        メッセージのリスト（各メッセージは{"role": "user"/"assistant", "content": "メッセージ内容", "timestamp": "タイムスタンプ"}の形式）
    
    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    messages = []
    current_message = None
    skip_line = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # システムメッセージはスキップ
            if re.match(r'\[(.*?)\] system:', line):
                skip_line = True
                continue
            
            # 新しいメッセージの開始行かチェック
            match = re.match(r'\[(.*?)\] (user|assistant): (.*)', line)
            if match:
                # 前のメッセージがあれば保存
                if current_message:
                    messages.append(current_message)
                
                # 新しいメッセージを開始
                timestamp, speaker, content = match.groups()
                current_message = {
                    "role": speaker,
                    "content": f"## 発言\n\n発言者: {speaker}, 発言日時: {timestamp}, 発言内容: {content.strip()}",
                    "timestamp": timestamp
                }
                skip_line = False
            elif current_message and not skip_line:
                # 現在のメッセージの続きの行（ファイル情報や添付ファイル詳細など）
                current_message["content"] += "\n" + line.strip()
    
    # 最後のメッセージを保存
    if current_message and not skip_line:
        messages.append(current_message)
    
    return messages

def find_latest_memory_file(memory_dir: str) -> Optional[Path]:
    """
    指定されたディレクトリ内の最新のPKL記憶ファイルを探す
    
    Args:
        memory_dir: 記憶ファイルが格納されているディレクトリのパス
        
    Returns:
        最新のPKL記憶ファイルのパス、見つからない場合はNone
    """
    memory_dir_path = Path(memory_dir)
    if not memory_dir_path.exists() or not memory_dir_path.is_dir():
        return None
    
    # PKLファイルのみを検索
    memory_files = list(memory_dir_path.glob("*.pkl"))
    
    # ファイルが見つからない場合
    if not memory_files:
        return None
    
    # 最終更新日時でソートして最新のファイルを返す
    return max(memory_files, key=lambda p: p.stat().st_mtime)

def load_memory_system(memory_dir: str) -> Tuple[Any, Dict[str, Any], str]:
    """
    記憶ファイルを読み込む
    
    Args:
        memory_dir: 記憶ファイルが格納されているディレクトリのパス
        
    Returns:
        (memory_obj, memory_dump, memory_id)のタプル
    """
    # 記憶ディレクトリの存在確認と作成
    memory_dir_path = Path(memory_dir)
    memory_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 最新の記憶ファイルを探す
    latest_memory_file = find_latest_memory_file(memory_dir)
    
    # 記憶システムの初期化
    memory_id = "memory_system_1"
    memory_obj = None
    memory_dump = None
    
    if latest_memory_file:
        # 最新の記憶ファイルが存在する場合は読み込む
        try:
            memory_file_path = str(latest_memory_file)
            with open(latest_memory_file, 'rb') as f:
                memory_obj = pickle.load(f)
                # print(f"PKLファイルから読み込んだオブジェクトの型: {type(memory_obj)}")
                
                # 型を検査
                if hasattr(memory_obj, 'content') and isinstance(memory_obj.content, MemorySystem):
                    # print(f"memory_obj.content の型: {type(memory_obj.content)}")
                    memory_dump = memory_obj.content.model_dump()
                    # print(f"memory_dump の型: {type(memory_dump)}")
                    print(f"最新の記憶ファイルを読み込みました: {memory_file_path}")
                else:
                    print(f"警告: PKLファイルから読み込んだオブジェクトが期待した形式ではありません")
                    # 空のMemorySystemを作成
                    memory_system = MemorySystem.create_empty_memory_system()
                    memory_dump = memory_system.model_dump()
                    memory_obj = type('MemoryObject', (), {'content': memory_system, 'id': memory_id})
        except Exception as e:
            print(f"記憶ファイルの読み込みに失敗しました: {e}")
            # 空のMemorySystemを作成
            memory_system = MemorySystem.create_empty_memory_system()
            memory_dump = memory_system.model_dump()
            memory_obj = type('MemoryObject', (), {'content': memory_system, 'id': memory_id})
    else:
        # 記憶ファイルが見つからない場合は空のシステムを作成
        print(f"記憶ディレクトリ {memory_dir} に記憶ファイルが見つかりません。空の記憶システムを作成します。")
        memory_system = MemorySystem.create_empty_memory_system()
        memory_dump = memory_system.model_dump()
        memory_obj = type('MemoryObject', (), {'content': memory_system, 'id': memory_id})
    
    return memory_obj, memory_dump, memory_id

def update_memory_system(chat_model, conversation: List[Dict[str, str]], memory_dump: Dict[str, Any], memory_id: str) -> Any:
    """
    Memory Managerを使用してMemorySystemを更新する
    
    Args:
        chat_model: 使用するチャットモデル
        conversation: 会話データ
        memory_dump: 記憶システムのダンプデータ
        memory_id: 記憶システムのID
        
    Returns:
        更新された記憶システム
        
    Raises:
        Exception: 記憶システムの更新に失敗した場合
    """
    # 更新前のメモリシステムの文字列表現を取得
    memory_before_str = str(memory_dump)
    memory_before_len = len(memory_before_str)
    
    # Memory Managerの作成
    manager = create_memory_manager(
        chat_model,
        schemas=[MemorySystem],
        instructions="""
        会話から全ての関連情報を抽出し、記憶システムを更新してください。
        
        既存の情報を尊重しつつ、新しい情報で更新してください。
        矛盾する情報がある場合は、最新の情報を優先してください。
        会話の内容から前回の会話が存在すると思われる場合には、タイムスタンプや内容をもとに前回の会話の記録部分を探して、そこに追加してください。

        マスターとは、初音ミクのマスターである。ユーザー（人間）のことです。
       
        すべての説明文は必ず日本語で記述してください。       
        """,
        enable_inserts=False,  # 記憶追加
        enable_updates=True,  # 記憶更新
        enable_deletes=True,  # 記憶削除
    )
    
    updated_memory_systems = manager.invoke({
        "messages": conversation,
        "existing": [(memory_id, memory_dump)]  # タプルのリストとして渡す
    })
    
    # 型検査
    if updated_memory_systems and len(updated_memory_systems) > 0:
        memory = updated_memory_systems[0]
        # print(f"content の内容: {memory.content}")
        
        if not hasattr(memory, 'content') or not isinstance(memory.content, MemorySystem):
            error_msg = f"更新されたメモリが期待した形式ではありません。型: {type(memory)}"
            if hasattr(memory, 'content'):
                error_msg += f", content の型: {type(memory.content)}"
            print(f"警告: {error_msg}")
            raise Exception(error_msg)
        
        # 更新後のメモリシステムの文字列表現を取得
        memory_after_str = str(memory.content)
        memory_after_len = len(memory_after_str)
        
        # 更新前と更新後の文字列長を比較
        if memory_after_len *1.1< memory_before_len :
            error_msg = f"更新後のメモリシステムの文字列長({memory_after_len})が更新前({memory_before_len})と比べて小さすぎます"
            print(f"エラー: {error_msg}")
            raise Exception(error_msg)
        
        return memory
    else:
        error_msg = "記憶システムの更新に失敗しました"
        print(error_msg)
        raise Exception(error_msg)

def update_conversation(chat_model, conversation: List[Dict[str, str]]) -> Conversation:
    """
    会話データからConversationオブジェクトを作成・更新する
    
    Args:
        chat_model: 使用するチャットモデル
        conversation: 会話データ
        
    Returns:
        更新されたConversationオブジェクト
        
    Raises:
        Exception: 会話データの更新に失敗した場合
    """
    # 空のConversationオブジェクトを新規作成
    empty_conversation = Conversation.create_empty_conversation()
    
    # Conversation用のMemory Manager作成
    conversation_manager = create_memory_manager(
        chat_model,
        schemas=[Conversation],
        instructions="""
        会話データから詳細な情報を抽出し、会話オブジェクトを更新してください。
                
        
        特に以下の点に注目してください：
        1. 会話の言語を適切に設定する
        2. すべてのメッセージを正確に保存する（タイムスタンプ、話者、内容）
        3. ファイル添付情報や詳細な説明も含める
        4. 全ての会話を保存する
        
        すべてのメッセージを完全に保存してください。
        会話の全文を漏れなく記録することが重要です。
        
        すべての説明文は必ず日本語で記述してください。
        """,
        enable_inserts=True,
        enable_updates=True,
        enable_deletes=False,
    )
    
    # 空のConversationに対して更新をかける
    conversation_id = "new_conversation"
    updated_conversations = conversation_manager.invoke({
        "messages": conversation,
        "existing": [(conversation_id, empty_conversation.model_dump())]
    })
    
    # 型検査
    if updated_conversations and len(updated_conversations) > 0:
        # print(f"updated_conversations[0] の型: {type(updated_conversations[0])}")
        
        # if hasattr(updated_conversations[0], 'content'):
            # print(f"updated_conversations[0].content の型: {type(updated_conversations[0].content)}")
        
        # 更新されたConversationが正しい形式かチェック
        if not hasattr(updated_conversations[0], 'content') or not isinstance(updated_conversations[0].content, Conversation):
            error_msg = f"更新されたConversationが期待した形式ではありません。型: {type(updated_conversations[0])}"
            print(f"警告: {error_msg}")
            raise Exception(error_msg)
        
        # 更新されたConversationオブジェクトを返す
        return updated_conversations[0].content
    else:
        error_msg = "会話データの更新に失敗しました"
        print(error_msg)
        raise Exception(error_msg)

def save_memory_system(memory: Any, memory_dir: str) -> bool:
    """
    更新されたMemorySystemを保存する
    
    Args:
        memory: 保存するメモリオブジェクト
        memory_dir: 保存先ディレクトリのパス
        
    Returns:
        保存に成功した場合はTrue、失敗した場合はFalse
    """
    # 保存前に型を検査
    if not hasattr(memory, 'content') or not isinstance(memory.content, MemorySystem):
        print(f"警告: 保存しようとしているオブジェクトが期待した形式ではありません")
        print(f"型: {type(memory)}")
        if hasattr(memory, 'content'):
            print(f"content の型: {type(memory.content)}")
        # 保存をスキップ
        return False
    
    # 現在の日時を取得してファイル名に追加
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ディレクトリの作成
    memory_dir_path = Path(memory_dir)
    memory_dir_path.mkdir(parents=True, exist_ok=True)
    
    # ファイル名に日時を追加
    base_name = "memory"
    file_ext = ".pkl"
    time_stamped_path = memory_dir_path / f"{base_name}_{current_time}{file_ext}"
    
    try:
        with open(time_stamped_path, 'wb') as f:
            pickle.dump(memory, f)
        print(f"更新された記憶を保存しました: {time_stamped_path}")
        
        # JSONは参照用に保存するが、読み込みには使用しない
        json_path = str(time_stamped_path).replace('.pkl', '.json')
        memory_dict = {}
        for attr in dir(memory):
            if not attr.startswith('__') and not callable(getattr(memory, attr)):
                try:
                    value = getattr(memory, attr)
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        memory_dict[attr] = value
                    else:
                        memory_dict[attr] = str(value)
                except Exception as e:
                    memory_dict[attr] = f"Error getting attribute: {str(e)}"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(memory_dict, f, ensure_ascii=False, indent=4)
        print(f"参照用にJSONも保存しました: {json_path}")
        
        return True
    except Exception as e:
        print(f"記憶の保存に失敗しました: {e}")
        return False

def load_and_update_memory(memory_dir: str, conversation_file_path: str, use_openrouter: bool = True) -> Tuple[Any, bool]:
    """
    記憶ファイルを読み込み、会話データで更新する
    
    Args:
        memory_dir: 記憶ファイルが格納されているディレクトリのパス
        conversation_file_path: 会話ファイルのフルパス
        use_openrouter: OpenRouterを使用するかどうか
        
    Returns:
        更新された記憶システムと成功フラグのタプル
        
    Raises:
        Exception: 会話ファイルが見つからない場合や記憶システムの更新に失敗した場合
    """
    # チャットモデルを初期化
    chat_model = setup_api_keys(use_openrouter)
    
    # 会話ファイルの解析
    try:
        conversation = parse_conversation_file(conversation_file_path)
        print(f"会話ファイルを解析しました: {conversation_file_path}")
        # print(f"conversation: {conversation}")
    except FileNotFoundError:
        print(f"会話ファイル {conversation_file_path} が見つかりません。")
        return None, False
    
    # 記憶システムの読み込み
    memory_obj, memory_dump, memory_id = load_memory_system(memory_dir)
    
    # 記憶システムの更新（最大10回試行）
    max_retries = 10
    updated_memory = None
    
    for memory_attempt in range(max_retries):
        try:
            print(f"記憶システムの更新を試行中... (試行 {memory_attempt+1}/{max_retries})")
            
            # 記憶システムの更新
            updated_memory = update_memory_system(chat_model, conversation, memory_dump, memory_id)
            print(f"記憶システムの更新に成功しました")

            # 記憶システムの保存
            success = save_memory_system(updated_memory, memory_dir)
            if success:
                print("記憶システムの更新が完了しました。")
                break
            else:
                print("記憶システムの保存に失敗しました。")
                raise Exception("記憶システムの保存に失敗しました")
                
        except Exception as e:
            print(f"記憶システムの更新中にエラーが発生しました: {e} (試行 {memory_attempt+1}/{max_retries})")
            if memory_attempt < max_retries - 1:
                print("再試行します...")
            else:
                error_msg = "最大試行回数に達しました。処理を中止します。"
                print(error_msg)
                raise Exception(error_msg)
    

    # 記憶システムの更新に成功した場合のみ、会話データの更新を試行
    if updated_memory is not None:
        # 会話データの更新（最大3回試行）- エンベディング用に別途保存するため、ここではコメントアウト
        conversation_updated = False
        
        for conv_attempt in range(max_retries):
            try:
                print(f"会話データの更新を試行中... (試行 {conv_attempt+1}/{max_retries})")
                
                # 会話データの更新
                updated_conversation = update_conversation(chat_model, conversation)
                print(f"会話データの更新に成功しました")
                
                # 会話データをベクトルDBに格納
                try:
                    # ベクトルデータベースの初期化（エンベディングモデルとベクトルストア）
                    vectorstore = initialize_vector_database()
                    
                    # 会話をベクトルDBに保存
                    store_conversation(updated_conversation, vectorstore)
                    print(f"会話データのベクトルDBへの格納に成功しました")
                    break
                except Exception as e:
                    print(f"会話データのベクトルDBへの格納中にエラーが発生しました: {e}")
                

            except Exception as e:
                print(f"会話データの更新中にエラーが発生しました: {e} (試行 {conv_attempt+1}/{max_retries})")
                if conv_attempt < max_retries - 1:
                    print("再試行します...")
                else:
                    print("最大試行回数に達しました。会話データの更新をスキップします。")
        
        return updated_memory, success
    
    # ここに到達することはないはず
    error_msg = "予期しない状況が発生しました"
    print(error_msg)
    raise Exception(error_msg)


# def update_memory_component(chat_model, conversation, memory_dump, memory_id, component_type, max_retries=3):
#     """
#     指定されたメモリコンポーネントを更新する共通関数
    
#     Args:
#         chat_model: 使用するチャットモデル
#         conversation: 会話データ
#         memory_dump: 記憶システムのダンプデータ
#         memory_id: 記憶システムのID
#         component_type: 更新するコンポーネントの種類 ('episodic', 'semantic', 'procedural', 'working')
#         max_retries: 最大再試行回数
        
#     Returns:
#         更新されたコンポーネントデータ
        
#     Raises:
#         Exception: 指定された回数の再試行後も更新に失敗した場合
#     """
#     # デバッグ用：インポートされている型を確認
#     print(f"デバッグ: typing.List = {List}")
#     print(f"デバッグ: typing.Dict = {Dict}")
#     print(f"デバッグ: typing.Any = {Any}")
#     print(f"デバッグ: EpisodicMemory = {EpisodicMemory}")
    
#     # コンポーネントタイプに応じた指示とスキーマを設定
#     component_config = {
#         'episodic': {
#             'schema': [List[EpisodicMemory]],
#             'instructions': """
#             会話から関連するエピソード記憶情報を抽出し、更新してください。
#             既存の情報を尊重しつつ、新しい情報で更新してください。
#             矛盾する情報がある場合は、最新の情報を優先してください。
#             すべての説明文は必ず日本語で記述してください。
#             """
#         },
#         'semantic': {
#             'schema': [SemanticMemories],
#             'instructions': """
#             会話から関連する意味記憶情報（ユーザープロファイル、エージェントプロファイル、合意事項など）を抽出し、更新してください。
#             既存の情報を尊重しつつ、新しい情報で更新してください。
#             矛盾する情報がある場合は、最新の情報を優先してください。
#             すべての説明文は必ず日本語で記述してください。
#             """
#         },
#         'procedural': {
#             'schema': [ProceduralMemories],
#             'instructions': """
#             会話から関連する手続き記憶情報（行動パターン、ルーチン、スキルなど）を抽出し、更新してください。
#             既存の情報を尊重しつつ、新しい情報で更新してください。
#             矛盾する情報がある場合は、最新の情報を優先してください。
#             すべての説明文は必ず日本語で記述してください。
#             """
#         },
#         'working': {
#             'schema': [WorkingMemory],
#             'instructions': """
#             会話から関連するワーキングメモリ情報（TODOリストなど）を抽出し、更新してください。
#             既存の情報を尊重しつつ、新しい情報で更新してください。
#             矛盾する情報がある場合は、最新の情報を優先してください。
#             すべての説明文は必ず日本語で記述してください。
#             """
#         }
#     }
    
#     # 指定されたコンポーネントの設定を取得
#     config = component_config.get(component_type)
#     if not config:
#         raise ValueError(f"不明なコンポーネントタイプ: {component_type}")
    
#     # 更新前のコンポーネントデータを取得
#     component_data = None
#     if component_type == 'episodic':
#         component_data = memory_dump.get('episodic_memories', [])
#     elif component_type == 'semantic':
#         component_data = memory_dump.get('semantic_memories', {})
#     elif component_type == 'procedural':
#         component_data = memory_dump.get('procedural_memories', {})
#     elif component_type == 'working':
#         component_data = memory_dump.get('working_memory', {})
    
#     # 更新前のデータサイズを記録
#     component_data_str = str(component_data)
#     component_data_len = len(component_data_str)
    
#     # 再試行ループ
#     for attempt in range(max_retries):
#         try:
#             print(f"{component_type}コンポーネントの更新を試行中... (試行 {attempt+1}/{max_retries})")
            
#             # デバッグ用：スキーマ情報を出力
#             print(f"デバッグ: {component_type}のスキーマ = {config['schema']}")
#             print(f"デバッグ: スキーマの型 = {type(config['schema'])}")
            
#             try:
#                 # Memory Managerの作成
#                 print(f"デバッグ: create_memory_manager呼び出し前")
#                 manager = create_memory_manager(
#                     chat_model,
#                     schemas=config['schema'],
#                     instructions=config['instructions'],
#                     enable_inserts=True,
#                     enable_updates=True,
#                     enable_deletes=False,
#                 )
#                 print(f"デバッグ: create_memory_manager呼び出し後")
                
#                 # コンポーネントの更新
#                 print(f"デバッグ: manager.invoke呼び出し前")
#                 updated_components = manager.invoke({
#                     "messages": conversation,
#                     "existing": [(memory_id, component_data)]
#                 })
#                 print(f"デバッグ: manager.invoke呼び出し後")
#             except Exception as e:
#                 print(f"デバッグ: 例外発生箇所 = {e.__class__.__name__}: {str(e)}")
#                 print(f"デバッグ: 例外のトレースバック:")
#                 import traceback
#                 traceback.print_exc()
#                 raise
            
#             # 更新結果の検証
#             if not updated_components or len(updated_components) == 0:
#                 raise Exception(f"{component_type}コンポーネントの更新に失敗しました: 結果が空です")
            
#             updated_component = updated_components[0].content
            
#             # 型チェック
#             if component_type == 'episodic' and not isinstance(updated_component, list):
#                 raise TypeError(f"更新された{component_type}コンポーネントが期待した型ではありません: {type(updated_component)}")
#             elif component_type == 'semantic' and not isinstance(updated_component, SemanticMemories):
#                 raise TypeError(f"更新された{component_type}コンポーネントが期待した型ではありません: {type(updated_component)}")
#             elif component_type == 'procedural' and not isinstance(updated_component, ProceduralMemories):
#                 raise TypeError(f"更新された{component_type}コンポーネントが期待した型ではありません: {type(updated_component)}")
#             elif component_type == 'working' and not isinstance(updated_component, WorkingMemory):
#                 raise TypeError(f"更新された{component_type}コンポーネントが期待した型ではありません: {type(updated_component)}")
            
#             # 更新後のデータサイズを検証
#             updated_component_str = str(updated_component)
#             updated_component_len = len(updated_component_str)
            
#             # 更新後のデータが極端に小さくなっていないか確認
#             if updated_component_len * 1.1 < component_data_len:
#                 raise Exception(f"更新後の{component_type}コンポーネントのサイズ({updated_component_len})が更新前({component_data_len})と比べて小さすぎます")
            
#             print(f"{component_type}コンポーネントの更新に成功しました")
#             return updated_component
            
#         except Exception as e:
#             print(f"{component_type}コンポーネントの更新中にエラーが発生しました: {e} (試行 {attempt+1}/{max_retries})")
#             if attempt < max_retries - 1:
#                 print("再試行します...")
#             else:
#                 error_msg = f"{component_type}コンポーネントの更新に失敗しました: 最大試行回数({max_retries})に達しました"
#                 print(f"エラー: {error_msg}")
#                 raise Exception(error_msg)


# def load_and_update_memory_hierarchical(memory_dir: str, conversation_file_path: str, use_openrouter: bool = True) -> Tuple[Any, bool]:
#     """
#     記憶ファイルを読み込み、会話データで階層的に更新する
#     各メモリコンポーネント（エピソード記憶、意味記憶、手続き記憶、ワーキングメモリ）を個別に更新し、
#     全てのコンポーネントが正常に更新された場合のみ、メモリシステム全体を更新する
    
#     Args:
#         memory_dir: 記憶ファイルが格納されているディレクトリのパス
#         conversation_file_path: 会話ファイルのフルパス
#         use_openrouter: OpenRouterを使用するかどうか
        
#     Returns:
#         更新された記憶システムと成功フラグのタプル
        
#     Raises:
#         Exception: 会話ファイルが見つからない場合や記憶システムの更新に失敗した場合
#     """
#     # チャットモデルを初期化
#     chat_model = setup_api_keys(use_openrouter)
    
#     # 会話ファイルの解析
#     try:
#         conversation = parse_conversation_file(conversation_file_path)
#         print(f"会話ファイルを解析しました: {conversation_file_path}")
#     except FileNotFoundError:
#         print(f"会話ファイル {conversation_file_path} が見つかりません。")
#         return None, False
    
#     # 記憶システムの読み込み
#     memory_obj, memory_dump, memory_id = load_memory_system(memory_dir)
    
#     try:
#         # 各コンポーネントを個別に更新
#         print("メモリシステムの階層的更新を開始します...")
        
#         # エピソード記憶の更新
#         updated_episodic = update_memory_component(
#             chat_model, conversation, memory_dump, memory_id, 'episodic'
#         )
        
#         # 意味記憶の更新
#         updated_semantic = update_memory_component(
#             chat_model, conversation, memory_dump, memory_id, 'semantic'
#         )
        
#         # 手続き記憶の更新
#         updated_procedural = update_memory_component(
#             chat_model, conversation, memory_dump, memory_id, 'procedural'
#         )
        
#         # ワーキングメモリの更新
#         updated_working = update_memory_component(
#             chat_model, conversation, memory_dump, memory_id, 'working'
#         )
        
#         # 更新されたコンポーネントを統合
#         updated_memory_system = MemorySystem(
#             episodic_memories=updated_episodic,
#             semantic_memories=updated_semantic,
#             procedural_memories=updated_procedural,
#             working_memory=updated_working
#         )
        
#         # メモリオブジェクトを作成
#         updated_memory = type('MemoryObject', (), {
#             'content': updated_memory_system,
#             'id': memory_id
#         })
        
#         # 更新されたメモリシステムを保存
#         success = save_memory_system(updated_memory, memory_dir)
#         if not success:
#             raise Exception("記憶システムの保存に失敗しました")
        
#         print("記憶システムの階層的更新が完了しました。")
        
#         # 会話データの更新
#         max_retries = 3
#         for conv_attempt in range(max_retries):
#             try:
#                 print(f"会話データの更新を試行中... (試行 {conv_attempt+1}/{max_retries})")
                
#                 # 会話データの更新
#                 updated_conversation = update_conversation(chat_model, conversation)
#                 print(f"会話データの更新に成功しました")
                
#                 # 会話データをベクトルDBに格納
#                 try:
#                     # ベクトルデータベースの初期化（エンベディングモデルとベクトルストア）
#                     vectorstore = initialize_vector_database()
                    
#                     # 会話をベクトルDBに保存
#                     store_conversation(updated_conversation, vectorstore)
#                     print(f"会話データのベクトルDBへの格納に成功しました")
#                     break
#                 except Exception as e:
#                     print(f"会話データのベクトルDBへの格納中にエラーが発生しました: {e}")
            
#             except Exception as e:
#                 print(f"会話データの更新中にエラーが発生しました: {e} (試行 {conv_attempt+1}/{max_retries})")
#                 if conv_attempt < max_retries - 1:
#                     print("再試行します...")
#                 else:
#                     print("最大試行回数に達しました。会話データの更新をスキップします。")
        
#         return updated_memory, success
        
#     except Exception as e:
#         print(f"記憶システムの階層的更新中にエラーが発生しました: {e}")
#         return None, False

def find_conversation_files(conversation_dir: str) -> List[Path]:
    """
    指定されたディレクトリ内の「session_」から始まる会話ファイルを取得し、作成日時順にソート
    
    Args:
        conversation_dir: 会話ファイルが格納されているディレクトリのパス
        
    Returns:
        作成日時順にソートされた会話ファイルのリスト
    """
    conversation_dir_path = Path(conversation_dir)
    if not conversation_dir_path.exists() or not conversation_dir_path.is_dir():
        print(f"会話ディレクトリ {conversation_dir} が見つかりません。")
        return []
    
    # session_から始まるファイルを検索
    conversation_files = list(conversation_dir_path.glob("session_*"))
    
    # ファイルが見つからない場合
    if not conversation_files:
        print(f"会話ディレクトリ {conversation_dir} に会話ファイルが見つかりません。")
        return []
    
    # ファイル名でソートして返す（昇順）
    return sorted(conversation_files, key=lambda p: p.name)

def process_conversation_file(conversation_file: Path, memory_dir: str) -> bool:
    """
    1つの会話ファイルを処理する
    
    Args:
        conversation_file: 処理する会話ファイルのパス
        memory_dir: 記憶ディレクトリのパス
        
    Returns:
        処理に成功した場合はTrue、失敗した場合はFalse
    """
    print(f"会話ファイル {conversation_file} の処理を開始します...")
    
    try:
        # 会話ファイルを処理（階層的更新方式を使用）
        # memory, success = load_and_update_memory_hierarchical(memory_dir, str(conversation_file))
        memory, success = load_and_update_memory(memory_dir, str(conversation_file))
        return success
    except Exception as e:
        print(f"会話ファイル {conversation_file} の処理中にエラーが発生しました: {e}")
        return False

def move_file(file_path: Path, success: bool, conversation_dir: str) -> None:
    """
    処理結果に応じてファイルを移動する
    
    Args:
        file_path: 移動するファイルのパス
        success: 処理に成功したかどうか
        conversation_dir: 会話ディレクトリのパス
    """
    # 移動先ディレクトリの作成
    base_dir = Path(conversation_dir)
    success_dir = base_dir / "register_success"
    failed_dir = base_dir / "register_failed"
    
    target_dir = success_dir if success else failed_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 移動先のファイルパス
    target_path = target_dir / file_path.name
    
    try:
        # ファイルの移動
        import shutil
        shutil.move(str(file_path), str(target_path))
        print(f"ファイル {file_path.name} を {target_dir} に移動しました。")
    except Exception as e:
        print(f"ファイル {file_path.name} の移動中にエラーが発生しました: {e}")

def check_memory_file_size(memory_dir: str, size_threshold_kb: int = 100) -> bool:
    """
    最新の記憶ファイルのサイズをチェックする
    
    Args:
        memory_dir: 記憶ディレクトリのパス
        size_threshold_kb: サイズ閾値（KB）
        
    Returns:
        ファイルサイズが閾値を超えている場合はTrue、そうでなければFalse
    """
    latest_memory_file = find_latest_memory_file(memory_dir)
    if not latest_memory_file:
        print(f"記憶ディレクトリ {memory_dir} に記憶ファイルが見つかりません")
        return False
    
    try:
        file_size_bytes = latest_memory_file.stat().st_size
        file_size_kb = file_size_bytes / 1024
        threshold_bytes = size_threshold_kb * 1024
        
        print(f"最新記憶ファイル: {latest_memory_file.name}")
        print(f"ファイルサイズ: {file_size_kb:.1f} KB ({file_size_bytes:,} bytes)")
        print(f"閾値: {size_threshold_kb} KB ({threshold_bytes:,} bytes)")
        
        if file_size_bytes > threshold_bytes:
            print(f"ファイルサイズが閾値を超えています。圧縮が必要です。")
            return True
        else:
            print(f"ファイルサイズは閾値以下です。圧縮は不要です。")
            return False
            
    except Exception as e:
        print(f"ファイルサイズチェック中にエラーが発生しました: {e}")
        return False

def conditional_memory_compression(memory_dir: str, size_threshold_kb: int = 100) -> bool:
    """
    条件付きで記憶圧縮を実行する
    最新の記憶ファイルが指定されたサイズを超えている場合のみ圧縮を実行
    
    Args:
        memory_dir: 記憶ディレクトリのパス
        size_threshold_kb: サイズ閾値（KB、デフォルト100KB）
        
    Returns:
        圧縮を実行して成功した場合、または圧縮が不要だった場合はTrue
        圧縮に失敗した場合はFalse
    """
    print("条件付き記憶圧縮を開始します...")
    
    # ファイルサイズをチェック
    needs_compression = check_memory_file_size(memory_dir, size_threshold_kb)
    
    if not needs_compression:
        print("記憶ファイルサイズが閾値以下のため、圧縮をスキップします")
        return True
    
    # 圧縮を実行
    print("記憶ファイルサイズが閾値を超えているため、圧縮を実行します")
    
    try:
        from models.memory_compressor import compress_latest_memory
        compressed_memory, success = compress_latest_memory(memory_dir)
        
        if success:
            print("条件付き記憶圧縮が正常に完了しました")
            return True
        else:
            print("条件付き記憶圧縮に失敗しました")
            return False
    except Exception as e:
        print(f"条件付き記憶圧縮中にエラーが発生しました: {e}")
        return False

def process_all_conversations(conversation_dir: str, memory_dir: str) -> None:
    """
    指定ディレクトリ内の全ての会話ファイルを処理する
    処理前に最新記憶ファイルのサイズをチェックし、100kBを超えている場合は圧縮を実行
    
    Args:
        conversation_dir: 会話ファイルが格納されているディレクトリのパス
        memory_dir: 記憶ディレクトリのパス
    """
    # パス設定を取得
    path_config = PathConfig.get_instance()
    
    # 会話ディレクトリと記憶ディレクトリを設定
    conversation_dir = str(path_config.conversations_dir)
    memory_dir = str(path_config.langmem_db_dir)
    
    # 最新記憶ファイルのサイズチェックと条件付き圧縮
    print("記憶システムの初期化を開始します...")
    compression_result = conditional_memory_compression(memory_dir)
    if compression_result:
        print("記憶圧縮処理が正常に完了しました（または不要でした）")
    else:
        print("記憶圧縮処理に失敗しましたが、処理を継続します")
    
    # 会話ディレクトリ内のファイルを処理
    conversation_files = find_conversation_files(conversation_dir)
    if not conversation_files:
        print("処理する会話ファイルがありません。")
        return
    
    print(f"{len(conversation_files)}個の会話ファイルを処理します...")
    
    # 各ファイルを順番に処理
    for i, file_path in enumerate(conversation_files):
        print(f"[{i+1}/{len(conversation_files)}] {file_path.name} を処理中...")
        
        # ファイルを処理
        success = process_conversation_file(file_path, memory_dir)
        
        # 処理結果に応じてファイルを移動
        move_file(file_path, success, conversation_dir)
    
    print(f"全ての会話ファイル ({len(conversation_files)}個) の処理が完了しました。")

def load_latest_memory_content_as_string(memory_dir: str) -> Optional[str]:
    """
    最新のPKLファイルからcontentを読み込み、文字列に変換する
    
    Args:
        memory_dir: 記憶ファイルが格納されているディレクトリのパス
        
    Returns:
        contentを文字列に変換したもの、失敗した場合はNone
    """
    # 最新の記憶ファイルを探す
    latest_memory_file = find_latest_memory_file(memory_dir)
    
    if not latest_memory_file:
        print(f"記憶ディレクトリ {memory_dir} に記憶ファイルが見つかりません。")
        return None
    
    try:
        # ファイルを開いてオブジェクトを読み込む
        with open(latest_memory_file, 'rb') as f:
            memory_obj = pickle.load(f)
            
        # contentを文字列に変換
        content_str = str(memory_obj.content)
        print("content_str", content_str)
        return content_str
    except Exception as e:
        print(f"記憶ファイルの読み込みに失敗しました: {e}")
        return None

def main():
    """メイン処理"""
    # パス設定を取得
    path_config = PathConfig.get_instance()
    
    # 会話ディレクトリと記憶ディレクトリを設定
    conversation_dir = str(path_config.conversations_dir)
    memory_dir = str(path_config.langmem_db_dir)
    
    # 全ての会話ファイルを処理
    # process_all_conversations(conversation_dir, memory_dir)

if __name__ == "__main__":
    main()
