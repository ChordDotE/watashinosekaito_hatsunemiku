from typing import Dict, List, Any
from datetime import datetime
import random
import uuid  # 一意のIDを生成するためのモジュール
from nodes.registry import register_node
from langchain.schema.messages import ToolMessage  # 正しいインポートパス

@register_node(
    name="weather_search",
    description="都市名から天気情報を検索するノード（モック版）",
    capabilities=["天気検索", "気象情報取得"],
    input_requirements=["city_name"],
    output_fields=["weather_info"]
)
def process_weather_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    都市名から天気情報を検索する関数（モック版）
    
    Args:
        state (Dict[str, Any]): 現在の状態
        
    Returns:
        Dict[str, Any]: 更新された状態
    """
    try:
        # 状態から情報を取得
        messages = state.get("messages", [])
        
        # 最新のメッセージから都市名を抽出
        city_name = extract_city_name(messages)
        if not city_name:
            city_name = "東京"  # デフォルト値
        
        # モックの天気情報を生成
        today_weather = generate_mock_weather()
        tomorrow_weather = generate_mock_weather()
        
        # 天気情報を整形
        weather_info = f"{city_name}の天気情報:\n今日: {today_weather}\n明日: {tomorrow_weather}"
        
        # ToolMessageオブジェクトを作成
        weather_message = ToolMessage(
            name="weather_search",
            content=weather_info,
            tool_call_id=f"weather_search_{uuid.uuid4()}",  # 一意のIDを生成
            additional_kwargs={
                "node_info": {
                    "node_name": "weather_search_node",
                    "node_type": "service",
                    "timestamp": datetime.now().isoformat(),
                },
                "weather_info": {
                    "city": city_name,
                    "today": today_weather,
                    "tomorrow": tomorrow_weather
                }
            }
        )
        
        # Stateに情報を追加
        updated_state = {
            **state,
            "success": True,
            "messages": state.get("messages", []) + [weather_message],
            "weather_info": weather_info,
            "response": weather_info,  # 直接応答として設定
            "next_node": "end"  # 直接endノードに進む
        }
        
        return updated_state
    except Exception as e:
        print(f"天気検索ノードエラー: {str(e)}")
        # エラーが発生した場合はデフォルト値を設定
        error_message = f"天気情報の取得に失敗しました: {str(e)}"
        
        # ToolMessageオブジェクトを作成（エラー情報を含む）
        error_message_obj = ToolMessage(
            name="weather_search",
            content=error_message,
            tool_call_id=f"weather_search_error_{uuid.uuid4()}",  # 一意のIDを生成
            additional_kwargs={
                "node_info": {
                    "node_name": "weather_search_node",
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
            "weather_info": "取得失敗",
            "response": error_message,  # 直接応答として設定
            "next_node": "end"  # 直接endノードに進む
        }
        
        return updated_state

def extract_city_name(messages):
    """
    メッセージから都市名を抽出する関数（簡易版）
    """
    # 簡易的な都市名抽出
    cities = ["東京", "大阪", "名古屋", "福岡", "札幌", "仙台", "広島", "京都"]
    
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if hasattr(msg, 'type') and msg.type == "human":
            content = msg.content
            for city in cities:
                if city in content:
                    return city
    
    return "東京"  # デフォルト値

def generate_mock_weather():
    """
    ランダムな天気情報を生成する関数
    """
    weather_types = ["晴れ", "曇り", "雨", "雪", "晴れ時々曇り", "曇り時々雨", "雨時々晴れ"]
    temperatures = list(range(0, 35))  # 0℃から35℃
    humidity_values = list(range(30, 90))  # 30%から90%
    
    weather = random.choice(weather_types)
    temp = random.choice(temperatures)
    humidity = random.choice(humidity_values)
    
    return f"天気: {weather}, 気温: {temp}°C, 湿度: {humidity}%"
