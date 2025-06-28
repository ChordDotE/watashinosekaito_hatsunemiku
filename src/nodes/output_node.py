"""
出力ノード - LLMを使って応答を生成する
"""
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime
from utils.api_logger import ApiLogger
from utils.llm_utils import call_llm
from utils.prompt_utils import load_prompt
from nodes.registry import register_node
from langchain.schema import AIMessage  # LangChainのメッセージクラスをインポート

# 会話履歴を抽出する関数
def extract_conversation_history(messages):
    """
    messagesリストから会話履歴を抽出する関数
    
    Args:
        messages (list): メッセージのリスト
        
    Returns:
        str: 整形された会話履歴
    
    Raises:
        ValueError: サポートされていないメッセージ形式の場合
    """
    conversation = []
    
    for msg in messages:
        # LangChainのメッセージオブジェクトの場合
        if hasattr(msg, 'content') and hasattr(msg, 'type'):
            # ノード情報を取得
            node_info = None
            if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                node_info = msg.additional_kwargs.get("node_info", {})
            
            # メッセージタイプに基づいてロールを決定
            if msg.type == "human":
                role = "ユーザー"
            elif msg.type == "ai":
                role = "アシスタント"
            elif msg.type == "system":
                # ノード情報に基づいて表示を調整
                node_name = node_info.get("node_name", "") if node_info else ""
                if node_name == "planner_node":
                    role = "思考プロセス"  # plannerノードの場合は「思考プロセス」として表示
                else:
                    role = "システム"
            elif msg.type == "function" or msg.type == "tool":
                role = "ツール"
                # 関数名/ツール名を取得
                name = getattr(msg, 'name', '不明なツール')
                role = f"{role}({name})"
            else:
                role = msg.type
            
            # 基本メッセージを作成
            message = f"{role}: {msg.content}"
            
            # additional_kwargsから追加情報を取得（すべてのキーを取得）
            # すべてのメッセージタイプで実行
            if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                # すべてのkwargsの情報を追加
                for key, value in msg.additional_kwargs.items():
                    message += f"\n[{key}: {value}]"
                
            conversation.append(message)
        # タプル形式の場合 - エラーを発生させる
        elif isinstance(msg, tuple):
            raise ValueError("タプル形式のメッセージはサポートされていません。LangChainのメッセージオブジェクトを使用してください。")
        # 辞書形式の場合 - エラーを発生させる
        elif isinstance(msg, dict):
            raise ValueError("辞書形式のメッセージはサポートされていません。LangChainのメッセージオブジェクトを使用してください。")
        # その他の形式 - エラーを発生させる
        else:
            raise ValueError(f"サポートされていないメッセージ形式です: {type(msg)}")
    
    return "\n\n".join(conversation)

# 最新のユーザー入力を取得する関数
def get_latest_user_input(messages):
    """
    messagesリストから最新のユーザー入力を取得する関数
    
    Args:
        messages (list): メッセージのリスト
        
    Returns:
        dict: 最新のユーザー入力情報
    
    Raises:
        ValueError: サポートされていないメッセージ形式の場合
    """
    # 最新のHumanMessageを探す
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        
        # LangChainのHumanMessageオブジェクトの場合
        if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "human":
            result = {'content': msg.content}
            
            # additional_kwargsからすべての情報を取得
            if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                # すべてのキーを取得
                for key, value in msg.additional_kwargs.items():
                    if value:
                        result[key] = value
            
            return result
        # タプル形式の場合 - エラーを発生させる
        elif isinstance(msg, tuple):
            raise ValueError("タプル形式のメッセージはサポートされていません。LangChainのメッセージオブジェクトを使用してください。")
    
    # ユーザーメッセージが見つからない場合は空の辞書を返す
    return {'content': ''}

# プロンプトを作成する関数
def create_output_prompt(state):
    """
    出力ノード用のプロンプトを作成する関数
    
    Args:
        state (dict): 現在の状態
        
    Returns:
        tuple: (prompt, system_prompt)
    
    Raises:
        ValueError: サポートされていないメッセージ形式の場合
    """
    try:
        # messagesから会話履歴を抽出
        messages = state.get("messages", [])
        conversation_history = extract_conversation_history(messages)
        
        # 最新のユーザー入力を取得
        latest_input = get_latest_user_input(messages)
        
        # システムプロンプトの読み込み
        system_prompt = load_prompt("output_prompt.txt")
        
        # ユーザープロンプトの作成
        prompt = f"""
        以下の情報を基に、ユーザーへの応答を生成してください。

        ## 最新のユーザー入力
        {latest_input.get('content', '')}
        
        ## 添付ファイル情報
        {latest_input.get('file_info', 'なし')}
        
        ## ファイル内容
        {latest_input.get('file_content', 'なし')}
        
        ## ユーザーの意図理解
        {latest_input.get('understanding', 'なし')}
        
        ## 会話履歴
        注意: 会話履歴には最新のユーザー入力も含まれています。上記の「最新のユーザー入力」と重複している場合がありますが、これは意図的なものです。会話の流れを把握するために、最新の入力を会話の文脈の中で理解してください。
        
        {conversation_history}
        
        ユーザーへの応答を生成してください。会話の文脈とファイル情報を考慮し、自然な応答を心がけてください。
        """
        
        return prompt, system_prompt
    except ValueError as e:
        # エラーが発生した場合はログに出力し、デフォルトのプロンプトを返す
        print(f"プロンプト作成エラー: {str(e)}")
        system_prompt = load_prompt("output_prompt.txt")
        prompt = """
        ユーザーへの応答を生成してください。
        
        エラーが発生したため、デフォルトの応答を返します。
        """
        return prompt, system_prompt

@register_node(
    name="output",
    description="ユーザーへの応答を生成する出力ノード",
    capabilities=["テキスト応答生成", "会話履歴の考慮", "ファイル情報の参照"],
    input_requirements=["messages", "files"],
    output_fields=["response"]
)
def process_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLMを使って応答を生成する関数
    
    Args:
        state (Dict[str, Any]): 現在の状態
        
    Returns:
        Dict[str, Any]: 更新された状態
    """
    try:
        # 状態から情報を取得
        files = state.get("files", [])
        messages = state.get("messages", [])
        
        # 会話履歴のログ出力
        print(f"出力ノード - 過去の会話履歴: {len(messages)}件")
        
        # プロンプトの作成
        prompt, system_prompt = create_output_prompt(state)
        
        # LLMを呼び出し
        response = call_llm(prompt, system_prompt, api_name="output_node")
        
        # responseをそのまま使用（call_llm関数内ですでにパース済み）
        # 応答テキストを取得（responseがdictの場合はcontentを取得、文字列の場合はそのまま使用）
        response_text = response.get("content", response) if isinstance(response, dict) else response
        
        # AIMessageオブジェクトを作成
        ai_message = AIMessage(
            content=response_text,
            additional_kwargs={
                "node_info": {
                    "node_name": "output_node",  # ノード名
                    "node_type": "user_facing",
                    "timestamp": datetime.now().isoformat(),
                }
            }
        )
        
        # Stateに情報を追加
        updated_state = {
            **state,  # 既存の状態を維持
            "success": True,
            "messages": state.get("messages", []) + [ai_message],  # AIMessageを追加
            "response": response_text  # 応答テキストを追加
        }
        
        return updated_state
    except Exception as e:
        print(f"出力ノードエラー: {str(e)}")
        # エラーが発生した場合はデフォルト値を設定
        error_response = f"エラーが発生したため、デフォルトの応答を返します: {str(e)}"
        
        # AIMessageオブジェクトを作成（エラー情報を含む）
        ai_message = AIMessage(
            content=error_response,
            additional_kwargs={
                "node_info": {
                    "node_name": "output_node",  # ノード名
                    "node_type": "user_facing",
                    "timestamp": datetime.now().isoformat(),
                },
                "error": str(e)
            }
        )
        
        # Stateに情報を追加（エラー情報を含む）
        updated_state = {
            **state,  # 既存の状態を維持
            "success": False,
            "messages": state.get("messages", []) + [ai_message],  # AIMessageを追加
            "response": "エラー発生"  # エラー応答を追加
        }
        
        return updated_state
