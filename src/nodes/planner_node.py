from typing import Dict, List, Any, Optional, Literal
import os
import json
from datetime import datetime
from utils.api_logger import ApiLogger
from utils.llm_utils import call_llm, parse_json_response
from utils.prompt_utils import load_prompt
from nodes.registry import register_node
from langchain.schema import HumanMessage, AIMessage, SystemMessage  # LangChainのメッセージクラスをインポート

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
def create_planner_prompt(state):
    """
    プランナーノード用のプロンプトを作成する関数
    
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
        
        # 利用可能なノード情報を整形
        available_nodes = state.get("available_nodes", {})
        available_nodes_str = ""
        for name, info in available_nodes.items():
            if name not in ["input", "planner"]:  # 入力とプランナーノードを除外
                capabilities = ", ".join(info.get("capabilities", []))
                available_nodes_str += f"- {name}: {info.get('description', '説明なし')} ({capabilities})\n"
        
        # システムプロンプトの読み込み
        system_prompt = load_prompt("planner_prompt.txt")
        
        # ユーザープロンプトの作成
        prompt = f"""
        以下の情報を基に、次に何をすべきか判断してください。

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
        
        
        ## 利用可能なノード
        {available_nodes_str if available_nodes_str else "なし"}
        
        以下の形式で回答してください（マークダウンのコードブロックは使わず、直接JSONオブジェクトを返してください）:
        {{
            "action": "output", // 実行すべきアクション（利用可能なノード名から選択）
            "next_action": "次のノードで実行すべき具体的な行動の説明。ユーザーへの応答内容をそのまま書くのではなく、自分が何を次に行うかを述べる。",
            "reasoning": "このアクションと行動を選んだ理由",
            "next_steps": "ユーザーの入力に対し、返答するために、次に考えられるステップの説明",
            "context_usage": "会話履歴とファイル情報をどのように活用したか"
        }}

        余分な説明や装飾は不要です。
        """
        
        return prompt, system_prompt
    except ValueError as e:
        # エラーが発生した場合はログに出力し、デフォルトのプロンプトを返す
        print(f"プロンプト作成エラー: {str(e)}")
        system_prompt = load_prompt("planner_prompt.txt")
        prompt = """
        次に何をすべきか判断してください。
        
        以下のJSON形式で回答してください（マークダウンのコードブロックは使わず、直接JSONオブジェクトを返してください）:
        {
            "action": "output",
            "next_action": "エラーが発生したため、デフォルトの応答を返します。",
            "reasoning": "サポートされていないメッセージ形式が検出されたため、デフォルトの応答を返します。",
            "next_steps": "エラーを修正する",
            "context_usage": "会話履歴は利用できませんでした"
        }

        余分な説明や装飾は不要です。単純なJSONオブジェクトのみを返してください。
        """
        return prompt, system_prompt

@register_node(
    name="planner",
    description="次のアクションを決定するプランナーノード",
    capabilities=["アクション決定", "文脈理解", "ファイル情報参照"],
    input_requirements=["messages"],
    output_fields=["next_node"]
)
def process_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    計画の処理を行い、次のアクションを決定する関数
    
    Args:
        state (Dict[str, Any]): 現在の状態
        
    Returns:
        Dict[str, Any]: 更新された状態（次のアクションを含む）
    """
    try:
        # 状態から情報を取得
        files = state.get("files", [])
        messages = state.get("messages", [])
        
        # 会話履歴のログ出力
        print(f"プランナーノード - 過去の会話履歴: {len(messages)}件")
        
        # プロンプトの作成
        prompt, system_prompt = create_planner_prompt(state)
        
        # LLMを呼び出し
        response = call_llm(prompt, system_prompt, api_name="planner_node")
        
        # responseをそのまま使用（call_llm関数内ですでにパース済み）
        action_decision = response
        
        # 次のノードを決定
        action = action_decision.get("action", "output")
        available_nodes = state.get("available_nodes", {})
        
        # 次のノード名を決定
        available_nodes = state.get("available_nodes", {})
        next_node = ""
        if action in available_nodes:
            next_node = action
            print(f"次のノード: {action} ({available_nodes[action].get('description', '説明なし')})")
        else:
            next_node = "output"
            print(f"アクション '{action}' に対応するノードが見つかりません。outputノードを使用します。")
        
        # SystemMessageオブジェクトを作成 - next_actionには次のノードでの行動を記載
        system_message = SystemMessage(
            content=action_decision.get("next_action", f"次のノード: {action}"),
            additional_kwargs={
                "node_info": {
                    "node_name": "planner_node",  # ノード名
                    "node_type": "internal",
                    "timestamp": datetime.now().isoformat(),
                },
                "action": action,
                "reasoning": action_decision.get("reasoning", ""),
                "next_steps": action_decision.get("next_steps", ""),
                "context_usage": action_decision.get("context_usage", "")
            }
        )
        
        # Stateに情報を追加（action_decisionを含めない）
        updated_state = {
            **state,  # 既存の状態を維持
            "success": True,
            "messages": state.get("messages", []) + [system_message],  # SystemMessageを追加
            "next_node": next_node  # 次のノード名を追加
        }
        
        return updated_state
    except Exception as e:
        print(f"プランナーノードエラー: {str(e)}")
        # エラーが発生した場合はデフォルト値を設定
        error_action_decision = {
            "action": "output",
            "content": f"エラーが発生したため、デフォルトの応答を返します: {str(e)}",
            "reasoning": f"エラーが発生したため、デフォルトの応答を返します: {str(e)}",
            "next_steps": "エラーを修正する",
            "context_usage": "会話履歴は利用できませんでした"
        }
        
        # SystemMessageオブジェクトを作成（エラー情報を含む）
        system_message = SystemMessage(
            content=error_action_decision["content"],
            additional_kwargs={
                "node_info": {
                    "node_name": "planner_node",  # ノード名
                    "node_type": "internal",
                    "timestamp": datetime.now().isoformat(),
                },
                "action": "output",
                "reasoning": error_action_decision["reasoning"],
                "next_steps": error_action_decision["next_steps"],
                "context_usage": error_action_decision["context_usage"],
                "error": str(e)
            }
        )
        
        # action_decisionを更新
        action_decision = error_action_decision
        
        # Stateに情報を追加（エラー情報を含む、action_decisionを含めない）
        updated_state = {
            **state,  # 既存の状態を維持
            "success": False,
            "messages": state.get("messages", []) + [system_message],  # SystemMessageを追加
            "next_node": "output"  # エラー時はoutputノードを使用
        }
        
        return updated_state
