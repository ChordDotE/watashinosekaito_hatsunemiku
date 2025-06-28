"""
Stateオブジェクトの処理に関するユーティリティクラス
"""
import re
from typing import Dict, List, Any, Optional

class StateUtils:
    """
    Stateオブジェクトの処理に関するユーティリティクラス
    各ノードで共通して使用される処理をまとめています
    """
    
    def __init__(self, max_messages=10, max_files=5):
        """
        初期化
        
        Args:
            max_messages (int): 取得する最大メッセージ数
            max_files (int): 取得する最大ファイル数
        """
        self.max_messages = max_messages
        self.max_files = max_files
    
    def extract_info(self, state) -> Dict[str, Any]:
        """
        Stateから必要な情報を抽出する
        
        Args:
            state (Dict[str, Any]): 現在の状態
            
        Returns:
            Dict[str, Any]: 抽出された情報
        """
        input_text = state.get("input_text", "")
        processed_input = state.get("processed_input", {})
        messages = state.get("messages", [])
        file_contents = state.get("file_contents", [])
        
        # 会話履歴を取得（すべてのメッセージを含む）
        conversation_context = ""
        if messages:
            # すべてのメッセージを処理（制限なし）
            for message in messages:
                # メッセージの形式を確認
                role = None
                content = None
                
                # LangChainのメッセージクラスのインスタンスかどうかを確認
                if hasattr(message, "__class__"):
                    class_name = message.__class__.__name__
                    if class_name in ["SystemMessage", "HumanMessage", "AIMessage"]:
                        if class_name == "SystemMessage":
                            role = "system"
                        elif class_name == "HumanMessage":
                            role = "user"
                        elif class_name == "AIMessage":
                            role = "assistant"
                        
                        if hasattr(message, "content"):
                            content = str(message.content)
                
                # タプル形式のメッセージかどうかを確認
                elif isinstance(message, tuple) and len(message) == 2:
                    role, content = message
                
                # 辞書形式のメッセージかどうかを確認
                elif isinstance(message, dict) and "role" in message and "content" in message:
                    role = message["role"]
                    content = message["content"]
                
                # 文字列形式のメッセージかどうかを確認
                elif isinstance(message, str):
                    content = message
                    # 文字列の先頭に「role:」形式があるか確認
                    if ":" in message and message.split(":", 1)[0].strip() in ["system", "user", "assistant"]:
                        parts = message.split(":", 1)
                        role = parts[0].strip()
                        content = parts[1].strip()
                    else:
                        # デフォルトはユーザーメッセージとして扱う
                        role = "user"
                
                # contentからメタデータを除去
                if role and content:
                    if isinstance(content, str):
                        # additional_kwargs={} response_metadata={} id='xxx' の形式を除去
                        pattern = r"additional_kwargs=\{\}\s*response_metadata=\{\}\s*id='[^']*'"
                        clean_content = re.sub(pattern, "", content).strip()
                        conversation_context += f"{role}: {clean_content}\n"
                    else:
                        conversation_context += f"{role}: {content}\n"
        
        # ファイル内容の文脈を構築（制限なし）
        file_context = ""
        if file_contents:
            file_context = "過去に共有されたファイル情報:\n"
            # すべてのファイルを処理
            for i, file in enumerate(file_contents):
                file_context += f"ファイル{i+1}: {file.get('filename', '不明')} - {file.get('description', '説明なし')}\n"
        
        # 画像内容の説明を取得
        image_description = ""
        interpretation = processed_input.get('interpretation', {})
        image_desc = interpretation.get('image_content_description', '')
        if image_desc and image_desc != "画像の内容を解析できませんでした":
            image_description = image_desc
        
        # 添付ファイル情報を取得
        files_info = processed_input.get('interpretation', {}).get('files_info', 'なし')
        
        # 抽出した情報を返す（デフォルト値を設定）
        return {
            "input_text": input_text,
            "files_info": processed_input.get('interpretation', {}).get('files_info', 'なし'),
            "image_description": image_description,
            "conversation_context": conversation_context,
            "file_context": file_context
        }
