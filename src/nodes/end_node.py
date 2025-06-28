"""
終了ノード - 処理を終了するためのノード
"""
from typing import Dict, Any
from datetime import datetime
from nodes.registry import register_node
from langchain.schema import AIMessage

@register_node(
    name="end",
    description="処理を終了するノード",
    capabilities=["処理終了"],
    input_requirements=[],
    output_fields=[]
)
def process_end(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    処理を終了する関数
    
    Args:
        state (Dict[str, Any]): 現在の状態
        
    Returns:
        Dict[str, Any]: 更新された状態
    """
    print("終了ノードが実行されました")
    
    # 状態をそのまま返す
    return {
        **state,
        "success": True
    }
