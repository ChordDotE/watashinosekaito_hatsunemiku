"""
ノード情報を管理するレジストリモジュール
"""
from typing import Dict, List, Any, Callable, Optional

# ノード情報を格納する辞書
_NODE_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_node(
    name: str,
    description: str,
    capabilities: List[str],
    input_requirements: List[str],
    output_fields: List[str]
):
    """ノード情報を登録するデコレータ"""
    def decorator(func: Callable):
        _NODE_REGISTRY[name] = {
            "name": name,
            "function": func,
            "description": description,
            "capabilities": capabilities,
            "input_requirements": input_requirements,
            "output_fields": output_fields
        }
        return func
    return decorator

def get_node_info(node_name: str) -> Dict[str, Any]:
    """ノード名から情報を取得する"""
    return _NODE_REGISTRY.get(node_name, {})

def get_all_nodes_info() -> Dict[str, Dict[str, Any]]:
    """すべてのノード情報を取得する"""
    # 関数参照を除外したコピーを返す
    # また、特定のノード（input, planner, output）を除外する
    excluded_nodes = ["input", "planner", "output"]  # 統合ノードに置き換えられたノードを除外
    return {
        name: {k: v for k, v in info.items() if k != "function"}
        for name, info in _NODE_REGISTRY.items()
        if name not in excluded_nodes  # 除外リストにないノードのみを返す
    }

def get_node_function(node_name: str) -> Optional[Callable]:
    """ノード名から関数を取得する"""
    node_info = _NODE_REGISTRY.get(node_name, {})
    return node_info.get("function")
