"""
過去の会話から関連する内容を検索するノード
"""
from typing import Dict, List, Any
from datetime import datetime
import uuid
from nodes.registry import register_node
from langchain.schema.messages import ToolMessage
from models.memory_manager import search_conversations

@register_node(
    name="memory_search",
    description="過去の会話から関連する内容を検索して一文ごとに思い出すノード",
    capabilities=["会話検索", "記憶検索", "関連情報取得"],
    input_requirements=["input_text", "processed_input"],
    output_fields=["memory_search_results"]
)
def process_memory_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    過去の会話から関連する内容を検索する関数
    
    Args:
        state (Dict[str, Any]): 現在の状態
        
    Returns:
        Dict[str, Any]: 更新された状態
    """
    try:
        # 状態から情報を取得
        messages = state.get("messages", [])
        input_text = state.get("input_text", "")
        processed_input = state.get("processed_input", "")
        
        # humanとaiのメッセージの内容を全て抽出
        all_contents = []
        
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                if msg.type in ["human", "ai"]:
                    all_contents.append(msg.content)
        
        # 最新の10個のメッセージに制限
        if len(all_contents) > 10:
            all_contents = all_contents[-10:]  # 末尾から10個を取得
        
        # 抽出したメッセージ内容を結合
        messages_text = " ".join(all_contents)
        
        # AIの最新のメッセージからunderstandingを取得（補足情報として）
        understanding = ""
        if messages and len(messages) > 0:
            last_msg = messages[-1]
            if hasattr(last_msg, 'type') and last_msg.type == "ai":
                if hasattr(last_msg, 'additional_kwargs') and 'understanding' in last_msg.additional_kwargs:
                    # successがtrueかどうかをチェック（additional_kwargsに含まれている場合）
                    if not hasattr(last_msg.additional_kwargs, 'success') or last_msg.additional_kwargs.get('success', True):
                        understanding = last_msg.additional_kwargs['understanding']
        
        # 検索クエリを作成
        search_query = f"{messages_text} {processed_input}"
        if understanding:
            search_query += f" {understanding}"
        # search_query = messages_text

        # 会話を検索（上位5件）- 修正後のsearch_conversations関数を使用
        search_results = search_conversations(search_query, k=5)
        
        # 検索結果を整形
        formatted_results = []
        for i, result in enumerate(search_results):
            content = result.page_content
            metadata = result.metadata
            
            # メタデータから情報を取得
            start_time = metadata.get("start_time", "不明")
            participant = metadata.get("participant", "不明")
            
            # 結果を整形
            formatted_result = f"会話 {i+1}:\n"
            formatted_result += f"- 日時: {start_time}\n"
            formatted_result += f"- 参加者: {participant}\n"
            formatted_result += f"- 内容:\n{content}\n"
            
            formatted_results.append(formatted_result)
        
        # 結果をまとめる
        if formatted_results:
            memory_search_results = "関連する過去の会話:\n\n" + "\n\n".join(formatted_results)
        else:
            memory_search_results = "関連する過去の会話は見つかりませんでした。"
        
        # ToolMessageオブジェクトを作成
        memory_message = ToolMessage(
            name="memory_search",
            content=memory_search_results,
            tool_call_id=f"memory_search_{uuid.uuid4()}",
            additional_kwargs={
                "node_info": {
                    "node_name": "memory_search_node",
                    "node_type": "service",
                    "timestamp": datetime.now().isoformat(),
                },
                "memory_info": {
                    "query": search_query,
                    "result_count": len(search_results)
                }
            }
        )
        
        # Stateに情報を追加
        updated_state = {
            **state,
            "success": True,
            "messages": state.get("messages", []) + [memory_message],
            "memory_search_results": memory_search_results,
            "response": memory_search_results,
            "next_node": "unified_response"  # 統合ノードに戻る
        }
        
        return updated_state
    except Exception as e:
        print(f"記憶検索ノードエラー: {str(e)}")
        # エラーが発生した場合はデフォルト値を設定
        error_message = f"記憶検索に失敗しました: {str(e)}"
        
        # ToolMessageオブジェクトを作成（エラー情報を含む）
        error_message_obj = ToolMessage(
            name="memory_search",
            content=error_message,
            tool_call_id=f"memory_search_error_{uuid.uuid4()}",
            additional_kwargs={
                "node_info": {
                    "node_name": "memory_search_node",
                    "node_type": "service",
                    "timestamp": datetime.now().isoformat(),
                },
                "error": str(e)
            }
        )
        
        # Stateに情報を追加（エラー情報を含む）
        updated_state = {
            **state,
            "success": False,
            "messages": state.get("messages", []) + [error_message_obj],
            "memory_search_results": "検索失敗",
            "response": error_message,
            "next_node": "unified_response"  # 統合ノードに戻る
        }
        
        return updated_state
