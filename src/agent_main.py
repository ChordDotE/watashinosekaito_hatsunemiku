import json
import os
import traceback
from pathlib import Path
import sys
from datetime import datetime
import uuid
import pickle
import threading
import time
from typing import Dict, List, Any, Annotated, Literal, Callable, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


# Stateの定義
# LangGraphで用いるためのクラス
class State(TypedDict, total=False):
    # ユーザーの入力テキスト
    input_text: str
    # 最新のユーザーの入力の添付ファイル情報を解釈し、日本語で記載したもの
    files: List[Dict[str, Any]]
    # 入力テキストとファイルを合わせて理解した文字
    processed_input: str
    # LangGraphで処理するメッセージ履歴
    # 各メッセージはLangChainのメッセージオブジェクト（HumanMessage, AIMessage, SystemMessageなど）
    # LangChainのメッセージオブジェクトのadditional_kwargs属性に以下の情報が含まれる：
    # - node_info: ノード情報（必須）
    #   - node_name: ノード名（例: "input_node", "planner_node"）
    #   - node_type: ノードタイプ（例: "user_facing", "internal"）
    #   - timestamp: 処理時刻
    # - file_info: 添付ファイルの概要情報（例: "1個のファイルが添付されています。(.jpg)"）
    # - file_content: ファイル内容の詳細な説明
    # - understanding: 入力とファイルから得られる本質的な理解
    messages: Annotated[list, add_messages]
    # 最終的なエージェントからの応答テキスト
    response: str
    # 処理成功フラグ
    success: bool
    # セッションファイル
    session_file: str
    # 利用可能なノード情報
    available_nodes: Dict[str, Dict[str, Any]]
    # 次のノード名
    next_node: str
    # 自動応答モードフラグ
    is_auto_response: bool
    # 無応答リマインダーフラグ
    is_inactivity_reminder: bool
    # 無応答タイムアウト秒数
    inactivity_timeout: int

# ノードモジュールのインポート
from nodes.unified_response_node import process_unified_response  # 統合ノード
from nodes.weather_search_node import process_weather_search
from nodes.memory_search_node import process_memory_search
from nodes.end_node import process_end  # 終了ノード
from nodes.registry import get_all_nodes_info
from utils.message_validator import MessageValidator, MessageValidationError

"""
各ノードの実装における必須項目と検証要件：

1. 必須の更新項目:
   - success: bool - 処理の成功/失敗を示すフラグ（必須）
     * 処理成功時は明示的に True を設定
     * 処理失敗時は False を設定し、可能であれば error キーにエラーメッセージを追加
   - messages: list - 処理結果を含むメッセージリスト（必須）
     * 各ノードは自身の処理結果を表すメッセージを追加する
     * メッセージは LangChain のメッセージオブジェクト（HumanMessage, AIMessage, SystemMessage など）
     * additional_kwargs に node_info を含める必要がある

2. メッセージ検証要件:
   - 各メッセージは MessageValidator.validate_message() でチェックされる
   - 検証に失敗すると MessageValidationError が発生
   - 検証項目:
     * LangChain の BaseMessage を継承していること
     * additional_kwargs が辞書型であること
     * node_info が存在し、辞書型であること
     * node_info に node_name, node_type, timestamp が含まれていること

3. エラー処理と再試行:
   - success=False の場合、最新のメッセージが削除され、同じノードが再処理される
   - 最大5回まで再試行される
   - 再試行回数を超えるとエラーとして処理が中止される
"""

def node_wrapper(node_func, node_name):
    """
    ノード関数をラップして、メッセージ検証と時間計測を行う関数
    
    Args:
        node_func (callable): ラップするノード関数
        node_name (str): ノード名
        
    Returns:
        callable: ラップされたノード関数
    """
    def wrapped_func(state):
        # 処理開始時間を記録
        start_time = datetime.now()
        
        # 再試行回数の上限
        MAX_RETRY_COUNT = 10
        retry_count = 0
        
        # 元の状態を保存（ディープコピー）
        import copy
        original_state = copy.deepcopy(state)
        
        while retry_count < MAX_RETRY_COUNT:
            # ノード関数を実行
            result = node_func(state)
            
            # 処理後のメッセージ検証
            try:
                if "messages" in result and result["messages"]:
                    MessageValidator.validate_messages(result["messages"])
            except MessageValidationError as e:
                error_msg = f"{node_name}の処理後にメッセージ検証エラーが発生しました: {str(e)}"
                print(f"エラー: {error_msg}")
                # エラー情報を含む状態を返す
                return {
                    **original_state,  # 元の状態に戻す
                    "success": False,
                    "error": error_msg
                }
            
            # successが設定されていない場合はエラーとして扱う
            if "success" not in result:
                error_msg = f"{node_name}の処理でsuccessフラグが設定されていません"
                print(f"エラー: {error_msg}")
                result["success"] = False
                result["error"] = error_msg
            
            # 処理が成功した場合は結果を返す（明示的にsuccess=Trueを設定）
            if result.get("success", False):
                # 再処理に備えてsuccessをtrueに設定
                result["success"] = True
                
                # 処理終了時間を記録
                end_time = datetime.now()
                
                # 処理時間を計算（ミリ秒単位）
                processing_time = (end_time - start_time).total_seconds() * 1000
                
                # 処理時間をログに出力
                print(f"ノード '{node_name}' の処理時間: {processing_time:.2f}ms")
                
                # stateログを保存
                save_state_log(result, node_name)
                
                return result
            
            # 処理が失敗した場合、元の状態に戻して再試行
            print(f"{node_name}の処理が失敗しました。再試行します（{retry_count + 1}/{MAX_RETRY_COUNT}）")
            
            # 元の状態に戻す（ただしエラー情報は保持）
            state = copy.deepcopy(original_state)
            state["error"] = result.get("error", f"{node_name}の処理が失敗しました")
            
            retry_count += 1
        
        # 最大再試行回数に達した場合
        print(f"{node_name}の処理が{MAX_RETRY_COUNT}回失敗しました。処理を中止します。")
        
        # 処理終了時間を記録（失敗時）
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        print(f"ノード '{node_name}' の処理時間（失敗）: {processing_time:.2f}ms")
        
        # 失敗状態を作成
        failed_state = {
            **original_state,  # 元の状態に戻す
            "success": False,
            "error": f"{node_name}の処理が{MAX_RETRY_COUNT}回失敗しました。"
        }
        
        # 失敗状態のログを保存
        save_state_log(failed_state, f"{node_name}_failed")
        
        return failed_state
    
    return wrapped_func

# グローバル変数の定義（process_agent_requestからinput_node_wrapperへのデータ受け渡し用）
CURRENT_INPUT_TEXT = ""
CURRENT_FILES_DATA = []

# 統合ノードのラッパー関数
def unified_response_wrapper(state: State) -> State:
    """
    統合ノードのラッパー関数
    
    Args:
        state (State): 現在の状態
        
    Returns:
        State: 更新された状態
    """
    # グローバル変数から入力テキストとファイルデータを取得
    # stateからではなく、process_agent_requestで設定されたグローバル変数から取得
    global CURRENT_INPUT_TEXT, CURRENT_FILES_DATA
    input_text = CURRENT_INPUT_TEXT
    files_data = CURRENT_FILES_DATA
    
    # process_unified_response関数を呼び出し
    updated_state = process_unified_response(state, input_text, files_data)
    
    # デバッグ情報: process_unified_response関数の結果を出力
    # print("\n=== process_unified_response 関数の結果 ===")
    # print(f"input_text: {updated_state.get('input_text', '')}")
    # print(f"files: {updated_state.get('files', [])}")
    # print(f"processed_input: {updated_state.get('processed_input', '')}")
    # print(f"messages: {updated_state.get('messages', [])}")
    # print(f"response: {updated_state.get('response', '')}")
    # print(f"next_node: {updated_state.get('next_node', '')}")
    # print("=====================================\n")
    
    return updated_state


# セッション情報をグローバルに保持
# Flaskアプリケーション起動時に一度だけ生成される
SESSION_ID = str(uuid.uuid4())
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_FILE = f"session_{SESSION_TIMESTAMP}.txt"

# グローバル変数としてpath_configを公開
path_config = None

# stateログを保存する関数
def save_state_log(state, node_name):
    """
    stateをpklとjsonで保存する関数
    
    Args:
        state (dict): 保存するstate
        node_name (str): ノード名
    """
    try:
        if not path_config or not hasattr(path_config, 'state_logs_dir'):
            print("警告: path_configが初期化されていないか、state_logs_dirが存在しないため、stateログを保存できません")
            return
        
        # セッション名からフォルダ名を作成（拡張子を除去）
        session_name = os.path.splitext(SESSION_FILE)[0]
        
        # 保存先ディレクトリのパスを作成
        log_dir = path_config.state_logs_dir / session_name
        
        # ディレクトリが存在することを確認
        os.makedirs(log_dir, exist_ok=True)
        
        # 現在のタイムスタンプを取得（ミリ秒まで含める）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # マイクロ秒の下3桁を除去
        
        # ファイル名を作成
        pkl_filename = f"{timestamp}_{node_name}.pkl"
        json_filename = f"{timestamp}_{node_name}.json"
        
        # pklファイルに保存
        pkl_path = log_dir / pkl_filename
        with open(pkl_path, 'wb') as f:
            pickle.dump(state, f)
        
        # jsonファイルに保存（シリアライズできない部分を処理）
        json_path = log_dir / json_filename
        
        # stateのコピーを作成して、JSONに変換できない部分を処理
        json_safe_state = {}
        for key, value in state.items():
            try:
                # 単純な型チェック
                if isinstance(value, (str, int, float, bool, type(None))):
                    json_safe_state[key] = value
                elif isinstance(value, (list, dict)):
                    # リストや辞書の場合は、文字列に変換
                    json_safe_state[key] = str(value)
                else:
                    # その他の型は文字列に変換
                    json_safe_state[key] = f"<{type(value).__name__}>"
            except Exception as e:
                json_safe_state[key] = f"<シリアライズエラー: {str(e)}>"
        
        # JSONファイルに保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_state, f, ensure_ascii=False, indent=2)
        
        print(f"stateログを保存しました: {pkl_path}, {json_path}")
    except Exception as e:
        print(f"警告: stateログの保存に失敗しました: {str(e)}")
        traceback.print_exc()

# 必要なモジュールのインポート
try:
    from utils.path_config import PathConfig

    # パスの取得    
    path_config = PathConfig.get_instance()
    
    # モデル関連のインポート（実際の実装に合わせて調整）
    # from models.llm_manager import LLMManager
    # llm_manager = LLMManager()
    
    # セッションパスの設定
    SESSION_PATH = path_config.conversations_dir / SESSION_FILE if hasattr(path_config, 'conversations_dir') else None
    
    # セッション情報のログ出力
    print(f"新しいセッションを開始: {SESSION_ID}")
    print(f"セッションファイル: {SESSION_FILE}")
    
    initialization_success = True
except Exception as e:
    print(f"エラー: agent_main初期化エラー: {str(e)}")
    traceback.print_exc()
    initialization_success = False
    SESSION_PATH = None
    path_config = None

# 会話履歴を保持する辞書（現在は使用していないため削除）

# ノード情報を取得
node_info = get_all_nodes_info()

# グラフビルダーの作成
graph_builder = StateGraph(State)


# ノードの追加（ラッパー関数を適用）
graph_builder.add_node("unified_response", node_wrapper(unified_response_wrapper, "unified_response"))  # 統合ノード
graph_builder.add_node("weather_search", node_wrapper(process_weather_search, "weather_search"))  # 天気検索ノード
graph_builder.add_node("memory_search", node_wrapper(process_memory_search, "memory_search"))  # 記憶検索ノード
graph_builder.add_node("end", process_end)  # 終了ノード（ラッパーなし）

# エッジの追加
graph_builder.add_edge("weather_search", "unified_response")  # weather_searchノードの後に統合ノードに戻る
graph_builder.add_edge("memory_search", "unified_response")  # memory_searchノードの後に統合ノードに戻る

# 条件付きエッジの定義を動的に構築
conditional_edges = {}
for name, info in node_info.items():
    if name not in ["unified_response", "input", "planner", "output"]:  # 統合ノードと置き換えられたノードを除外
        conditional_edges[name] = name

# デフォルトノードを設定（終了）
conditional_edges["default"] = "end"  # 空文字列の代わりに"end"という名前のノードを使用
print(f"デフォルトエッジを追加: default -> end")

# 条件付きエッジの内容を確認
print("条件付きエッジの内容:")
for key, value in conditional_edges.items():
    print(f"  {key} -> {value}")

# 現在利用可能なノードを表示
print("利用可能なノード:")
for name, info in node_info.items():
    capabilities = ", ".join(info.get("capabilities", []))
    print(f"- {name}: {info.get('description', '説明なし')} ({capabilities})")

# 条件付きエッジを追加
graph_builder.add_conditional_edges(
    "unified_response",
    lambda state: state.get("next_node"),
    {**conditional_edges, "end": "end"}  # "end"を明示的に追加
)

# 開始ノードの指定
graph_builder.set_entry_point("unified_response")

# チェックポインタの作成
memory = MemorySaver()

# グラフのコンパイル
graph = graph_builder.compile(checkpointer=memory)

def process_agent_request(input_text, files_data, is_auto_response=False, is_inactivity_reminder=False):
    """
    入力テキストとファイルデータを処理し、応答を生成する
    
    Args:
        input_text (str): ユーザーからの入力テキスト
        files_data (list): 添付ファイルのデータリスト。各要素は辞書形式で、
                          filename, type, size, content_type, contentを含む
        is_auto_response (bool): 自動応答モードかどうか
        is_inactivity_reminder (bool): 無応答リマインダーかどうか
    
    Returns:
        dict: 処理結果を含む辞書
    """
    try:
        if not initialization_success:
            return {
                'response': "システムの初期化に失敗しました。ログを確認してください。",
                'success': False
            }
        
        print(f"入力テキスト: {input_text}")
        print(f"添付ファイル数: {len(files_data)}")
        
        # ファイル情報のログ出力（デバッグ用）
        for i, file_data in enumerate(files_data):
            print(f"ファイル {i+1}: {file_data['filename']} ({file_data['type']}, {file_data['size']} バイト)")
        
        # 利用可能なノード情報を取得
        available_nodes = get_all_nodes_info()
        
        # デバッグ情報を出力
        # print(f"process_agent_request - 利用可能なノード情報: {available_nodes}")
        
        # デバッグログを追加
        # print(f"=== agent_main.py デバッグ ===")
        # print(f"is_auto_response: {is_auto_response}")
        # print(f"is_inactivity_reminder: {is_inactivity_reminder}")
        # print(f"input_text: '{input_text}'")
        # print("===============================")
        
        # 初期状態を作成（input_textとfiles_dataは含めない）
        initial_state = {
            "processed_input": "",  # 文字列として初期化
            "messages": [],
            "response": "",
            "success": False,
            "session_file": "",
            "available_nodes": available_nodes,  # 利用可能なノード情報を追加
            "is_auto_response": is_auto_response,  # 自動応答モードフラグを追加
            "is_inactivity_reminder": is_inactivity_reminder,  # 無応答リマインダーフラグを追加
            "inactivity_timeout": 60  # デフォルトのタイムアウト値を設定
        }
        
        # グローバル変数に一時的に保存（input_node_wrapperで使用）
        global CURRENT_INPUT_TEXT, CURRENT_FILES_DATA
        CURRENT_INPUT_TEXT = input_text
        CURRENT_FILES_DATA = files_data
        
        # デバッグ情報を出力
        # print(f"process_agent_request - 初期状態のavailable_nodes: {initial_state.get('available_nodes', {})}")
        # print("\n=== 初期状態のstate ===")
        # print(f"state: {initial_state}")
        # print("=====================================\n")
        
        # 処理開始時間を記録
        request_start_time = datetime.now()
        
        # LangGraphを実行（スレッドIDとしてSESSION_IDを使用）
        result = graph.invoke(
            initial_state,
            {"configurable": {"thread_id": SESSION_ID}}
        )
        
        # 処理終了時間を記録
        request_end_time = datetime.now()
        request_processing_time = (request_end_time - request_start_time).total_seconds() * 1000
        print(f"\n全体の処理時間: {request_processing_time:.2f}ms")
        
        # グローバル変数をクリア
        CURRENT_INPUT_TEXT = ""
        CURRENT_FILES_DATA = []
        
        # 結果のstateを出力
        print("\n=== 最終的なstate ===")
        print(f"state: {result}")
        print("=====================================\n")
        
        # 最終結果のメッセージ検証
        try:
            if "messages" in result and result["messages"]:
                MessageValidator.validate_messages(result["messages"])
        except MessageValidationError as e:
            error_msg = f"最終結果のメッセージ検証エラー: {str(e)}"
            print(f"警告: {error_msg}")
        
        # ユーザー入力とアシスタント応答を保存
        # stateからinput_textを取得
        user_text = result.get('input_text', input_text)
        
        # デバッグ情報: filesを出力
        # print("\n=== files デバッグ情報 ===")
        files = result.get('files', [])
        # print(f"files: {files}")
        # print("================================\n")
        
        # メッセージから最後のHumanMessageのadditional_kwargsを取得
        additional_kwargs = None
        messages = result.get('messages', [])
        
        # 最後のHumanMessageを探す
        for i in range(len(messages) - 1, -1, -1):  # 逆順に探索
            message = messages[i]
            if hasattr(message, "additional_kwargs") and hasattr(message, "type") and message.type == "human":
                additional_kwargs = message.additional_kwargs
                break
        
        # デバッグ情報: additional_kwargsを出力
        # print("\n=== additional_kwargs デバッグ情報 ===")
        # print(f"additional_kwargs: {additional_kwargs}")
        # print("================================\n")
        
        # 応答テキストを取得
        response_text = result.get('response', '')
        
        # 応答が空でない場合のみ会話ログに保存
        if response_text and response_text.strip():
            # ユーザーメッセージを保存（stateのfilesとadditional_kwargsを渡す）
            save_message(user_text, True, files, additional_kwargs)
            
            # アシスタントメッセージを保存
            save_message(response_text, False)
        else:
            print("応答が空のため、会話ログへの記録をスキップしました")
        
        # 最終的なstateをログに保存
        save_state_log(result, "final_state")
        
        # 結果を返す
        return {
            'response': result.get('response', ''),
            'session_file': result.get('session_file', ''),
            'success': result.get('success', False),
            'inactivity_timeout': result.get('inactivity_timeout', 60)  # デフォルト値を60に変更
        }
        
    except Exception as e:
        print(f"エラー: エージェント処理エラー: {str(e)}")
        traceback.print_exc()
        return {
            'response': f"エラーが発生しました: {str(e)}",
            'success': False
        }


def save_message(text, is_user, files=None, additional_kwargs=None):
    """
    メッセージをファイルに保存する関数
    
    Args:
        text (str): 保存するメッセージテキスト
        is_user (bool): ユーザーのメッセージかどうか
        files (List[Dict[str, Any]], optional): ファイル情報のリスト
        additional_kwargs (Dict[str, Any], optional): 追加のメタデータ
    """
    if not SESSION_PATH:
        return
        
    try:
        # デバッグ情報を出力
        # print("\n=== save_message デバッグ情報 ===")
        # print(f"text: {text}")
        # print(f"is_user: {is_user}")
        # print(f"files: {files}")
        # print("================================\n")
        
        # ディレクトリが存在することを確認
        os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
        
        # 現在のタイムスタンプを取得
        current_time = datetime.now().isoformat()
        
        # ファイル情報を追加
        message_text = text
        
        # additional_kwargsから情報を取得して追加
        if additional_kwargs:
            # ファイル情報を取得
            file_info = additional_kwargs.get("file_info", "")
            if file_info and file_info != "添付ファイルはありません。":
                message_text += f"\n(ファイル情報: {file_info})"
        
        # ファイル詳細情報を追加
        if files and is_user:  # ユーザーメッセージの場合のみファイル情報を追加
            # print("\n=== ファイル詳細情報 ===")
            
            # ファイル名のリストを作成
            filenames = []
            for file_info in files:
                filename = file_info.get('filename', '')
                file_type = file_info.get('type', '不明')
                # print(f"ファイル: {filename} ({file_type})")
                filenames.append(f"{filename} ({file_type})")
            
            # 最初のファイルの説明を使用
            if files:
                description = files[0].get('description', '説明なし')
                # print(f"説明: {description}")
                
                # ファイル詳細情報を一度だけ追加
                message_text += f"\n(添付ファイル詳細情報:\n{', '.join(filenames)}: {description})"
                # print(f"追加されたファイル情報: {message_text}")
            # else:
            #     print("ファイル詳細情報はありません")
            
            # print("========================\n")
        
        with open(SESSION_PATH, 'a', encoding='utf-8') as f:
            # メッセージを追加（フラグ付き）
            sender = "user" if is_user else "assistant"
            f.write(f"[{current_time}] {sender}: {message_text}\n")
        
        # print(f"メッセージを保存しました: {SESSION_PATH}")
    except Exception as save_error:
        print(f"エラー: メッセージの保存に失敗しました: {str(save_error)}")

# def save_final_output(final_output):
#     """
#     最終出力をファイルに保存する関数
    
#     Args:
#         final_output (str): 最終出力
    
#     Returns:
#         dict: 処理結果を含む辞書
#     """
#     if not SESSION_PATH:
#         return {
#             'success': False,
#             'message': 'セッションパスが設定されていません',
#             'session_file': ""
#         }
        
#     try:
#         # ディレクトリが存在することを確認
#         os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
        
#         # 現在のタイムスタンプを取得
#         current_time = datetime.now().isoformat()
        
#         with open(SESSION_PATH, 'a', encoding='utf-8') as f:
#             # 最終出力を追加
#             f.write(f"[{current_time}] final_output: {final_output}\n")
        
#         # print(f"最終出力を保存しました: {SESSION_PATH}")
#         return {
#             'success': True,
#             'message': '最終出力を保存しました',
#             'session_file': SESSION_FILE
#         }
#     except Exception as save_error:
#         print(f"エラー: 最終出力の保存に失敗しました: {str(save_error)}")
#         return {
#             'success': False,
#             'message': f'最終出力の保存に失敗しました: {str(save_error)}',
#             'session_file': SESSION_FILE
#         }
