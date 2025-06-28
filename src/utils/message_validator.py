"""
メッセージ検証ユーティリティモジュール
メッセージの型検査と検証を行う機能を提供
"""
from typing import Any, Dict, List, Optional, Union
from langchain.schema import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    FunctionMessage,
    BaseMessage
)

class MessageValidationError(Exception):
    """メッセージ検証エラー"""
    pass

class MessageValidator:
    """メッセージの検証を行うクラス"""
    
    @staticmethod
    def validate_message(msg: Any) -> None:
        """
        メッセージを検証し、無効な場合は例外を発生させる
        
        Args:
            msg (Any): 検証するメッセージ
            
        Raises:
            MessageValidationError: メッセージが無効な場合
        """
        # LangChainのBaseMessageを継承しているか確認
        if not isinstance(msg, BaseMessage):
            raise MessageValidationError(f"無効なメッセージ形式です: {type(msg)}。LangChainのメッセージオブジェクトを使用してください。")
        
        # additional_kwargsが辞書型であることを確認
        if not hasattr(msg, 'additional_kwargs') or not isinstance(msg.additional_kwargs, dict):
            raise MessageValidationError(f"メッセージにadditional_kwargsがないか、辞書型ではありません: {type(msg)}")
        
        # node_infoが存在することを確認
        if "node_info" not in msg.additional_kwargs:
            raise MessageValidationError(f"メッセージにnode_info情報がありません: {msg.type}")
        
        # node_infoが辞書型であることを確認
        node_info = msg.additional_kwargs["node_info"]
        if not isinstance(node_info, dict):
            raise MessageValidationError(f"node_infoが辞書型ではありません: {type(node_info)}")
        
        # 必須フィールドが存在することを確認
        required_fields = ["node_name", "node_type", "timestamp"]
        for field in required_fields:
            if field not in node_info:
                raise MessageValidationError(f"node_infoに必須フィールド '{field}' がありません")
    
    @staticmethod
    def validate_messages(messages: List[Any]) -> None:
        """
        メッセージリストを検証し、無効な場合は例外を発生させる
        
        Args:
            messages (List[Any]): 検証するメッセージリスト
            
        Raises:
            MessageValidationError: メッセージが無効な場合
        """
        if not isinstance(messages, list):
            raise MessageValidationError(f"messagesがリスト型ではありません: {type(messages)}")
        
        for i, msg in enumerate(messages):
            try:
                MessageValidator.validate_message(msg)
            except MessageValidationError as e:
                # インデックス情報を追加してエラーを再発生
                raise MessageValidationError(f"メッセージ[{i}]が無効です: {str(e)}")
        
        # 検証成功時のメッセージ
        print(f"メッセージ検証成功: {len(messages)}件のメッセージを検証しました")
