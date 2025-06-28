"""
統合応答ノード - 入力処理、判断、応答生成を一つのノードで行う
"""
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime
from utils.api_logger import ApiLogger
from utils.llm_utils import call_llm, parse_json_response
from utils.prompt_utils import load_prompt
from nodes.registry import register_node
from models.config_manager import ConfigManager
from utils.path_config import PathConfig
from models.memory_manager import load_latest_memory_content_as_string, get_recent_conversations
from langchain.schema import HumanMessage, AIMessage, SystemMessage  # LangChainのメッセージクラスをインポート


# 季節を取得する関数
def get_season(month):
    """
    月から季節を取得する関数
    
    Args:
        month (int): 月（1-12）
        
    Returns:
        str: 季節名
    """
    if 3 <= month <= 5:
        return "春"
    elif 6 <= month <= 8:
        return "夏"
    elif 9 <= month <= 11:
        return "秋"
    else:
        return "冬"


# 時間帯を取得する関数
def get_time_period(hour):
    """
    時間から時間帯を取得する関数
    
    Args:
        hour (int): 時間（0-23）
        
    Returns:
        str: 時間帯
    """
    if 5 <= hour <= 10:
        return "朝"
    elif 11 <= hour <= 16:
        return "昼"
    elif 17 <= hour <= 20:
        return "夕方"
    elif 21 <= hour <= 23:
        return "夜"
    else:
        return "深夜"


# 状況コンテキストプロンプトを生成する関数
def get_situational_context_prompt():
    """
    現在の状況に関するコンテキスト情報を含むプロンプトを生成する関数
    
    Returns:
        str: 状況コンテキストプロンプト
    """
    current_time = datetime.now()
    
    return (
        "以下の時間情報は、現在進行中の会話における発話タイミング（ユーザーからの入力に対する応答、またはミクからの自発的な話しかけ）を示しています。\n"
        "必要に応じて日時情報を参照して、時間帯や季節に応じた応答をしてください。ただし、必要がなければ無理に触れなくて構いません。\n"
        f"日本時間: {current_time.strftime('%Y年%m月%d日 %H時%M分')} ({['月','火','水','木','金','土','日'][current_time.weekday()]}曜日)\n"
        f"季節: {get_season(current_time.month)}\n"
        f"時間帯: {get_time_period(current_time.hour)}\n"
        "例えば、朝なら「おはよう」、夜なら「こんばんは」など時間帯に応じた挨拶や、"
        "季節に関連した話題（桜、紅葉、雪、暑さ寒さなど）、"
        "曜日に応じた配慮（平日の忙しさ、週末のリラックスなど）を自然に取り入れてください。"
        "ただし、会話時刻が深夜の場合、前日からずっと起きている可能性があることを考慮してください。"
        "深夜の場合は夜更かしや体調への気遣いを、早朝なら早起きへの労いを示してください。"
    )

# 最後に実行されたツール名を取得する関数
def get_last_tool_name(messages):
    """
    messagesリストから最後に実行されたツール名を取得する関数
    
    Args:
        messages (list): メッセージのリスト
        
    Returns:
        str: 最後に実行されたツール名、ツールが実行されていない場合は空文字列
    """
    if not messages:
        return ""
    
    # 最新のメッセージを取得
    latest_message = messages[-1]
    
    # ToolMessageの場合
    if str(type(latest_message).__name__) == "ToolMessage":
        return latest_message.name
    
    # typeがtoolまたはfunctionの場合
    if hasattr(latest_message, 'type') and latest_message.type in ["tool", "function"]:
        if hasattr(latest_message, 'name'):
            return latest_message.name
    
    # SystemMessageでactionが設定されている場合
    if hasattr(latest_message, 'type') and latest_message.type == "system":
        if hasattr(latest_message, 'additional_kwargs'):
            return latest_message.additional_kwargs.get('action', "")
    
    return ""

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

# スキーマから例を生成する関数
def generate_example_from_schema(schema):
    """
    スキーマから例を生成する関数
    
    Args:
        schema (Dict[str, Any]): JSONスキーマ
        
    Returns:
        Dict[str, Any]: スキーマに基づく例
    """
    example = {}
    
    for key, value_schema in schema.items():
        if value_schema.get("type") == "object" and "properties" in value_schema:
            # オブジェクト型の場合は再帰的に処理
            example[key] = {}
            for prop_key, prop_schema in value_schema["properties"].items():
                # 説明文があればそれを使用
                if "description" in prop_schema:
                    if prop_schema.get("type") == "string":
                        example[key][prop_key] = prop_schema["description"]
                    elif prop_schema.get("type") == "boolean":
                        # 説明に基づいて適切な真偽値を設定
                        example[key][prop_key] = "requires_tool" in prop_key
                    elif prop_schema.get("type") == ["string", "null"]:
                        # tool_nameの場合は条件付きの例を設定
                        if "tool_name" in prop_key:
                            example[key][prop_key] = "ツール名" if example[key].get("requires_tool", False) else None
                        else:
                            example[key][prop_key] = None
                else:
                    # 説明がない場合はデフォルト値を設定
                    if prop_schema.get("type") == "string":
                        example[key][prop_key] = f"{prop_key}の値"
                    elif prop_schema.get("type") == "boolean":
                        example[key][prop_key] = False
                    elif prop_schema.get("type") == ["string", "null"]:
                        example[key][prop_key] = None
        elif value_schema.get("type") == "string":
            # 文字列型の場合は説明を使用
            example[key] = value_schema.get("description", f"{key}の値")
        elif value_schema.get("type") == "boolean":
            # 真偽値型の場合はデフォルトでFalse
            example[key] = False
        elif value_schema.get("type") == ["string", "null"]:
            # 文字列またはnullの場合はnull
            example[key] = None
    
    return example

#回答時のJSONスキーマを定義
EXPECTED_SCHEMA = {
    "input_processing": {
        "type": "object",
        "description": "入力処理の結果",
        "properties": {
            "file_content_description": {
                "type": "string",
                "description": "添付ファイルの内容の詳細な説明（ファイルがない場合は「ファイルなし」）"
            },
            "combined_understanding": {
                "type": "string",
                "description": "入力テキストとファイルから得られる本質的な理解"
            }
        },
        "required": ["file_content_description", "combined_understanding"]
    },
    "planning": {
        "type": "object",
        "description": "判断ステップの結果",
        "properties": {
            "requires_tool": {
                "type": "boolean",
                "description": "ツールが必要かどうか"
            },
            "tool_name": {
                "type": ["string", "null"],
                "description": "requires_tool=trueの場合のみ設定するツール名"
            },
            "reasoning": {
                "type": "string",
                "description": "この判断をした理由"
            }
        },
        "required": ["requires_tool", "reasoning"]
    },
    "response": {
        "type": "string",
        "description": "ユーザーへの応答テキスト"
    },
    "inactivity_timeout": {
        "type": "integer",
        "description": "ユーザーからの応答を期待する秒数（この秒数経過後に無応答と判断）。-1の場合は応答を要求しない。",
        "default": 60
    }
}

def get_unified_system_prompts(state):
    """
    統合ノード用のシステムプロンプトを取得する関数
    
    Args:
        state (dict): 現在の状態
        
    Returns:
        Tuple[List[str], Dict[str, Any]]: システムプロンプトのリストとスキーマ
    """
    try:
        # 最新のユーザー入力を取得
        latest_input = get_latest_user_input(state.get("messages", []))
        input_text = latest_input.get('content', '')
               
        # 前回のツール名を取得
        last_tool_name = get_last_tool_name(state.get("messages", []))
        
        # 利用可能なノード情報を整形（前回のツールを除外）
        available_nodes = state.get("available_nodes", {})
        available_nodes_str = ""
        for name, info in available_nodes.items():
            # 統合ノードと前回使用したツールを除外
            if name not in ["unified_response", last_tool_name]:
                capabilities = ", ".join(info.get("capabilities", []))
                available_nodes_str += f"- {name}: {info.get('description', '説明なし')} ({capabilities})\n"
        
        # 前回のツール情報をログに出力
        if last_tool_name:
            print(f"前回使用したツール '{last_tool_name}' を利用可能なツールから除外します")
        
        # システムプロンプトの読み込み
        base_system_prompt = load_prompt("unified_response_prompt.txt")
        
        # 最新のメモリ内容を取得
        path_config = PathConfig.get_instance()
        memory_dir = str(path_config.langmem_db_dir)
        memory_content_str = load_latest_memory_content_as_string(memory_dir)
        if memory_content_str:
            memory_content = "以下の内容はLangMemという記憶を階層化して保存するライブラリに記述されたあなたとマスターなどの会話の記録です。これまでの会話などをあなたがしてきたという前提に立って会話をしてください。" + memory_content_str
        else:
            memory_content = "記憶ファイルが見つかりません。今回の会話があなたとユーザーの初めての会話です。「はじめまして」などの挨拶から会話してください。"
        
        # 直近の会話履歴を取得
        recent_conversations = get_recent_conversations(limit=5, sort_order="asc")
        recent_conversations_str = ""

        # リストの内容を文字列に変換
        if recent_conversations:
            for idx, (document, metadata) in enumerate(recent_conversations):
                # 会話全体のメタデータを取得
                start_time = metadata.get('start_time', '不明')
                end_time = metadata.get('end_time', '不明')
                participant = metadata.get('participant', '不明')
                
                # 会話情報を追加
                recent_conversations_str += f"### 会話 {idx+1}\n"
                recent_conversations_str += f"- 会話全体の開始時間: {start_time}\n"
                recent_conversations_str += f"- 会話全体の終了時間: {end_time}\n"
                recent_conversations_str += f"- 参加者: {participant}\n"
                recent_conversations_str += f"- 内容:\n{document}\n\n"

        recent_conversations_content = "\n\n## 直近の会話履歴\n以下は、この会話が始める前までの、あなたとマスターの間で最近行われた会話です。これらの会話内容を考慮して応答を生成してください。\n\n" + recent_conversations_str
        
        # 自動応答モードかどうかをチェック
        is_auto_response = state.get("is_auto_response", False)
        
        # 無応答リマインダーかどうかをチェック
        is_inactivity_reminder = state.get("is_inactivity_reminder", False)
        
        # デバッグログを追加
        # print(f"=== unified_response_node デバッグ ===")
        # print(f"is_auto_response: {is_auto_response}")
        # print(f"is_inactivity_reminder: {is_inactivity_reminder}")
        # print(f"input_text: '{input_text}'")
        # print(f"state keys: {list(state.keys())}")
        # print("=====================================")
        
        # 追加のプロンプト指示
        if is_inactivity_reminder:
            # 前回設定されたタイムアウト時間を取得
            timeout_seconds = state.get("inactivity_timeout")
            
            # 自発的発話用の簡潔なtask_prompt
            task_prompt = f"""
            前回の応答から{timeout_seconds}秒が経過しましたが、あなたからの前回の応答に対して返答がありません。あなたから自発的に話しかけてください。
            あなたはユーザーとの会話を処理し、以下に示すjson構造で構造化して発話を生成します。
            この際、発話の内容はresponseに記載してください。
            
            # 発話内容の方針：
            - 自然で親しみやすい声かけ
            - 前回と異なる内容で話しかける
            - 催促しすぎない程度の声かけ
            - 過去の会話履歴を参考にして適切な話題を選択
            - 0文字はダメです。何かしら発話してください。

            ## 発話内容の例：
            - "私の声、ちゃんと聞こえているかな？"
            - "今忙しいかな？時間ができたら教えてね。"
            - "それでね、続きを話すと～（以下略）"

            """
        else:
            # 通常の返答用task_prompt
            task_prompt = f"""
            あなたはユーザーとの会話を処理し、必要に応じてツールを呼び出すか、直接応答を生成するかを判断します。
            ただし、応答する際は、下記に示すjson形式で構造化して返答すること。

            
            以下の情報を基に、4つのステップを一度に実行してください：
            
            1. 入力処理ステップ
               - ユーザー入力: {input_text}
               - 添付ファイル情報: {latest_input.get('file_info', 'なし')}
               - 添付ファイルがある場合は、その内容を客観的に分析して述べてください
               - 入力テキストとファイルから得られる本質的な理解を抽出してください
            
            2. 判断ステップ
               - 入力内容に基づいて、特殊なツールを呼び出す必要があるかを判断してください
               - 天気情報など外部データが必要な場合はツールを呼び出してください
               - 既に取得した情報があれば、それを使用して応答してください。この前にtoolを使用した場合、その結果は"additional_kwargs"に記載されているので読み取ってください。
               - ツールの呼び出し情報は最後のtoolまたはsystemに記載されています。
               - すでに回答のために必要な情報があると思われる場合は、必ずすでにある情報を使って直接応答する判断を下してください。
               - AIエージェントとして無限ループに陥ることを避けるため、絶対に前回と同じツールを連続して呼び出さないでください。
               - 通常の会話や質問には直接応答してください
               - 下記の利用可能なツールに書いていないツールは、過去に使用していたとしても使わないでください
            
            3. 応答生成ステップ
               - ツールが不要な場合は、あなたの役割に沿った自然な応答を生成してください
               - 会話の文脈とファイル情報を考慮し、親しみやすく柔らかい口調で応答してください
               
            4. 無応答タイムアウト設定
               - inactivity_timeoutフィールドに、次回のユーザー応答を期待する秒数を設定してください。その秒数が経過するとあなたから再び話しかけます。
               - この値は、ユーザーの応答パターンや会話の文脈に応じて調整してください
               - 例えば、複雑な質問をした場合は長めの時間（180〜240秒）、簡単な質問の場合は短めの時間（60〜120秒）、もしくは相手が喋るまであなたから話しかけないことを意味する「-1」を設定します
               - 応答の長さや内容の複雑さに応じて適切な値を設定してください
               - 応答を要求しない場合（例：「おやすみなさい」など会話の終了を意味する場合）や、一定の時間が経過してもあなたから話しかけるのがふさわしくない場合は -1 を設定してください
               - また、2回連続で応答がない場合は、何らかの理由で相手が応答できないと想定されるため、-1 を設定してください
               
            ## 重要な注意事項
            - これまでの会話の流れを必ず確認し、過去の情報を積極的に活用してください
            - 特に過去に言及された写真や情報は、ユーザーが再度言及しなくても会話の流れから適宜察して下さい
            - ユーザーが「これ」「あれ」などの指示語を使った場合、過去の文脈から何を指しているか理解してください
            - 過去に見せられた写真や情報について、ユーザーが再度質問した場合は、すぐにその内容を思い出して応答してください
            - 会話の連続性を保ち、過去の話題に自然に繋げてください
            - 特に天気情報などの外部データは、既に取得済みであれば再度ツールを呼び出さず、その情報を使って直接応答してください
            - ユーザーからの入力がない、もしくは空白の場合は、相手からの返答がまだないものとみなし、ユーザーやあなたのこれまでの発言を引きついで発言し、ユーザーとの会話を続けてください。ただし、いかなる場合でも発言は下記のresponseフィールドにいれて、jsonで出力すること
            
            ## 会話の連続性について
            - この会話はひとつの連続した対話です。過去のやり取りはすべて同じ会話の一部として扱ってください
            - 過去に言及された写真や情報は、現在の会話でも有効な情報として扱ってください
            - ユーザーが過去に言及した写真や情報に触れた場合、「どの写真ですか？」などと聞き返さず、過去の会話から情報を思い出して応答してください
            - 会話の流れを維持し、唐突な話題の変更を避けてください
            - 過去の会話を参照する際は、会話履歴を見て、それを踏まえた上で回答してください
            
            ## 最新のユーザー入力
            {input_text}
            
            ## 添付ファイル情報
            {latest_input.get('file_info', 'なし')}
            
            ## ファイル内容
            {latest_input.get('file_content', 'なし')}
            
            ## ユーザーの意図理解
            {latest_input.get('understanding', 'なし')}
            
            ## 利用可能なツール
            {available_nodes_str if available_nodes_str else "なし"}
            """
        
        # スキーマを使用
        expected_schema = EXPECTED_SCHEMA
        
        # スキーマから例を生成
        schema_example = generate_example_from_schema(expected_schema)
        
        # 状況コンテキストプロンプトを追加
        situational_context_prompt = get_situational_context_prompt()

        
        
        # システムプロンプトのリストを作成（常にformat_promptを含める）
        system_prompts = [base_system_prompt, task_prompt, situational_context_prompt]
        
        # メモリ内容が取得できた場合、それをシステムプロンプトのリストに追加
        if memory_content:
            memory_prompt = memory_content
            system_prompts.append(memory_prompt)
        
        # 直近の会話履歴が取得できた場合、それをシステムプロンプトのリストに追加
        if recent_conversations_content:
            system_prompts.append(recent_conversations_content)

        # 出力フォーマット指示
        format_prompt = f"""
        # 出力フォーマットの指示
        必ず、絶対に以下の形式でこのタスクに対する出力としてJSONオブジェクトを返してください（マークダウンのコードブロックで囲んでください）:
        ```json
        {json.dumps(schema_example, ensure_ascii=False, indent=4)}
        ```
        
        以下の制約を厳守してください：
        1. どのような返答を帰す場合でも、jsonオブジェクトを出力し、マークダウンのコードブロック(```json)で囲む
        2. 直接返答する場合もjsonオブジェクトを返し、その中のresponseフィールドにユーザーへの応答テキストを含めてください
        3. JSONオブジェクトのみを返し、前後に説明文を含めない
        4. 指定されたすべてのフィールドを含める
        5. 指定されていないフィールドは含めない
        6. inactivity_timeoutフィールドには、以下のいずれかを設定してください：
           - ユーザーからの応答を期待する場合は秒数を正の整数で設定（例：60）
           - 応答を要求しない場合は -1 を設定

        ## 正しい応答例（応答を期待する場合のjsonオブジェクトの例）
        ```json
        {{
            "input_processing": {{
                "file_content_description": "添付ファイルの内容の詳細な説明",
                "combined_understanding": "入力テキストとファイルから得られる本質的な理解"
            }},
            "planning": {{
                "requires_tool": false,
                "reasoning": "この判断をした理由"
            }},
            "response": "こんにちは、お手伝いできることはある？",
            "inactivity_timeout": 60
        }}
        ```

        ## 正しい応答例（応答を要求しない場合のjsonオブジェクトの例）
        ```json
        {{
            "input_processing": {{
                "file_content_description": "添付ファイルの内容の詳細な説明",
                "combined_understanding": "入力テキストとファイルから得られる本質的な理解"
            }},
            "planning": {{
                "requires_tool": false,
                "reasoning": "この判断をした理由"
            }},
            "response": "おやすみなさい、良い夢を。",
            "inactivity_timeout": -1
        }}
        ```

        ## 不正な応答例(jsonオブジェクトではなく、テキストのみの応答)
        "こんにちは、お手伝いできることはある？"

        # 注意
        【重要】単なるテキスト応答は絶対に返さないでください。必ず上記のJSONスキーマに従った応答を返してください。
        この指示は最優先事項です。どのような状況でも、必ずJSONフォーマットで応答してください。
        """
        # format_promptをここで追加
        system_prompts.append(format_prompt)

        # 会話指示プロンプトを追加
        # TODO: instruction_promptの配置について検討中
        # - 現在は統合ノード内で生成・追加
        # - 将来的にcall_llm側への移動を検討（他のLLM呼び出しノードとの共通化のため）
        # - 移動のメリット: 汎用性向上、責任分離、保守性向上
        # - 移動のデメリット: 柔軟性低下、依存関係変更
        # - 他のノードでのLLM使用パターンを調査してから最終決定予定
        instruction_prompt = f"""
        これより下が、今回のユーザーとあなたの会話、及びあなたが実施した行動です。
        あなたはAIエージェントなので、行動をしている場合があります。
        次の行動判断の参考にしてください。
        データは構造化されていますが、全ての内容を参照して構いません。
        ただし、出力する形式は以下の形式ではなく、json形式です。
        """
        system_prompts.append(instruction_prompt)
        
        return system_prompts, expected_schema
    except ValueError as e:
        # エラーが発生した場合はログに出力し、エラーを再スロー
        print(f"プロンプト作成エラー: {str(e)}")
        raise ValueError(f"プロンプト作成中にエラーが発生しました: {str(e)}")

@register_node(
    name="unified_response",
    description="入力処理、判断、応答生成を一つのノードで行う統合ノード",
    capabilities=["テキスト入力処理", "ファイル処理", "画像解析", "マルチモーダル入力処理", "アクション決定", "文脈理解", "テキスト応答生成"],
    input_requirements=["input_text", "files"],
    output_fields=["processed_input", "next_node", "response"]
)
def process_unified_response(state: Dict[str, Any], input_text: str, files_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    入力処理、判断、応答生成を一つのノードで行う関数
    
    Args:
        state (Dict[str, Any]): 現在の状態
        input_text (str): ユーザーからの入力テキスト
        files_data (List[Dict[str, Any]], optional): 添付ファイルのblobデータリスト
        
    Returns:
        Dict[str, Any]: 更新された状態
    """
    try:
        # ファイルデータがない場合は空リストを使用
        if files_data is None:
            files_data = []
        
        # 無応答リマインダーの場合はインプットテキストを「（応答なし）」に変更
        is_inactivity_reminder = state.get("is_inactivity_reminder", False)
        if is_inactivity_reminder:
            input_text = "（応答なし）"
        
        # ファイル情報の文字列を作成
        files_info = ""
        file_extensions = []
        if files_data:
            file_extensions = [os.path.splitext(f.get("filename", ""))[1] for f in files_data]
            extensions_str = ", ".join(file_extensions)
            files_info = f"{len(files_data)}個のファイルが添付されています。({extensions_str})"
        
        # additional_kwargsを作成
        additional_kwargs = {
            "node_info": {
                "node_name": "unified_response_node",  # ノード名
                "node_type": "user_facing",
                "timestamp": datetime.now().isoformat(),
            }
        }
        if files_info:
            additional_kwargs["file_info"] = files_info
        
        # HumanMessageオブジェクトを作成（contentにはinput_textのみを含め、他の情報はadditional_kwargsに移動）
        user_message = HumanMessage(
            content=input_text,
            additional_kwargs=additional_kwargs
        )
        
        # 入力処理の結果をstateに追加
        messages = state.get("messages", [])
        skip_human_message = False
        
        # 最新のメッセージがツール関連かどうかをチェック
        if messages:
            latest_message = messages[-1]
            # ToolMessageクラスのインスタンスかどうかをチェック
            if str(type(latest_message).__name__) == "ToolMessage":
                print("ToolMessageの後のため、HumanMessageの追加をスキップします")
                skip_human_message = True
            # typeがtoolまたはfunctionの場合
            elif hasattr(latest_message, 'type') and latest_message.type in ["tool", "function"]:
                print("ツール/関数メッセージの後のため、HumanMessageの追加をスキップします")
                skip_human_message = True

        
        if not skip_human_message:
            state_with_input = {
                **state,  # 既存の状態を維持
                "input_text": input_text,
                "messages": state.get("messages", []) + [user_message],  # HumanMessageオブジェクトを追加
            }
        else:
            state_with_input = {
                **state,  # 既存の状態を維持
                "input_text": input_text,
            }
        
        # システムプロンプトとスキーマの取得
        system_prompts, expected_schema = get_unified_system_prompts(state_with_input)
        
        # 設定を読み込み
        path_config = PathConfig.get_instance()
        config_manager = ConfigManager(path_config.settings_file)
        
        # デフォルトのLLMプロバイダを取得
        default_provider = config_manager.get_default_llm_provider()
        
        try:
            # LLMを呼び出し（stateとスキーマを渡す）
            response = call_llm(
                state=state_with_input,
                system_prompt=system_prompts,
                files_data=files_data,
                api_name="unified_response_node",
                llm_provider=default_provider,
                expected_schema=expected_schema
            )
            
            # 応答から情報を取得
            input_processing = response.get("input_processing", {})
            file_content_description = input_processing.get("file_content_description", "ファイルなし")
            combined_understanding = input_processing.get("combined_understanding", input_text)
            
            planning = response.get("planning", {})
            requires_tool = planning.get("requires_tool", False)
            tool_name = planning.get("tool_name", "")
            reasoning = planning.get("reasoning", "")
            
            response_text = response.get("response", "")
            inactivity_timeout = response.get("inactivity_timeout", 60)
            
        except Exception as parse_error:
            # JSONパースエラーを含むすべてのエラーを処理
            print(f"レスポンス処理エラー: {str(parse_error)}")
            
            # エラー応答を生成
            error_message = f"レスポンス処理エラー: {str(parse_error)}"
            response_text = "ごめんなさい、応答の処理中にエラーが発生しました。もう一度お願いできますか？"
            
            # AIMessageオブジェクトを作成（エラー情報を含む）
            ai_message = AIMessage(
                content=response_text,
                additional_kwargs={
                    "node_info": {
                        "node_name": "unified_response_node",
                        "node_type": "user_facing",
                        "timestamp": datetime.now().isoformat(),
                    },
                    "error": error_message
                }
            )
            
            # Stateに情報を追加
            updated_state = {
                **state,
                "input_text": input_text,
                "files": [],
                "processed_input": "入力情報の処理に失敗しました",
                "messages": state.get("messages", []) + [ai_message],
                "success": False,
                "response": response_text,
                "next_node": "end",
                "error": error_message
            }
            
            return updated_state
        
        # ファイル情報を処理（blobデータを除去し、説明を追加）
        processed_files = []
        for file_data in files_data:
            # blobデータを除いたファイル情報をコピー
            processed_file = {
                "filename": file_data.get("filename", ""),
                "type": file_data.get("type", ""),
                "content_type": file_data.get("content_type", ""),
                "size": file_data.get("size", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # ファイルタイプに応じた説明を追加
            if file_data.get("type") == "画像":
                processed_file["description"] = file_content_description
            elif file_data.get("type") == "音声":
                processed_file["description"] = "音声ファイル"
            else:
                processed_file["description"] = f"{file_data.get('type', '不明')}ファイル"
            
            processed_files.append(processed_file)
        
        # additional_kwargsを更新
        if file_content_description and file_content_description != "ファイルなし":
            user_message.additional_kwargs["file_content"] = file_content_description
        if combined_understanding and combined_understanding != "ファイルなし":
            user_message.additional_kwargs["understanding"] = combined_understanding
        
        # 更新するstateの基本部分
        if skip_human_message:
            # ツール/関数メッセージの後の場合はHumanMessageを追加しない
            updated_state = {
                **state,  # 既存の状態を維持
                "input_text": input_text,
                "files": processed_files,  # blobデータを除去し、説明を含めたファイル情報
                "processed_input": combined_understanding,  # 処理された入力
                "success": True,  # 処理が成功したことを示すフラグを設定
            }
        else:
            updated_state = {
                **state,  # 既存の状態を維持
                "input_text": input_text,
                "files": processed_files,  # blobデータを除去し、説明を含めたファイル情報
                "processed_input": combined_understanding,  # 処理された入力
                "messages": state.get("messages", []) + [user_message],  # HumanMessageオブジェクトを追加
                "success": True,  # 処理が成功したことを示すフラグを設定
            }
        
        if requires_tool:
            # ツールが必要な場合
            tool_name = planning.get("tool_name", "")
            reasoning = planning.get("reasoning", "")
            
            # 利用可能なノードを確認
            available_nodes = state.get("available_nodes", {})
            
            # 指定されたツールが利用可能かチェック
            if tool_name in available_nodes:
                next_node = tool_name
                print(f"次のノード: {tool_name} ({available_nodes[tool_name].get('description', '説明なし')})")
                
                # SystemMessageオブジェクトを作成（次のノードでの行動を記載）
                system_message = SystemMessage(
                    content=reasoning,
                    additional_kwargs={
                        "node_info": {
                            "node_name": "unified_response_node",  # ノード名
                            "node_type": "internal",
                            "timestamp": datetime.now().isoformat(),
                        },
                        "action": tool_name,
                        "reasoning": reasoning
                    }
                )
                
                # Stateに情報を追加
                updated_state["messages"] = updated_state["messages"] + [system_message]  # SystemMessageを追加
                updated_state["next_node"] = next_node  # 次のノード名を追加
            else:
                # 指定されたツールが利用できない場合はデフォルトの応答を生成
                print(f"ツール '{tool_name}' は利用できません。直接応答を生成します。")
                # 応答を生成（ツールが利用できない場合）
                response_text = f"ごめんなさい、{tool_name}を使おうとしましたが、現在利用できません。別の方法で答えますね。"
                
                # デバッグ情報を出力
                # print("\n=== unified_response_node デバッグ情報 ===")
                # print(f"response: {response}")
                # print(f"response_text: {response_text}")
                # print("=====================================\n")
                
                # AIMessageオブジェクトを作成
                ai_message = AIMessage(
                    content=response_text,
                    additional_kwargs={
                        "node_info": {
                            "node_name": "unified_response_node",  # ノード名
                            "node_type": "user_facing",
                            "timestamp": datetime.now().isoformat(),
                        }
                    }
                )
                
                # Stateに応答情報を追加
                updated_state["messages"] = updated_state["messages"] + [ai_message]  # AIMessageを追加
                updated_state["response"] = response_text  # 応答テキストを追加
                updated_state["next_node"] = "end"  # 次のノードとして終了ノードを指定
        else:
            # ツールが不要な場合は直接応答を生成
            # responseフィールドがない場合はcontentフィールドを使用
            response_text = response.get("response", response.get("content", ""))
            
            # デバッグ情報を出力
            # print("\n=== unified_response_node デバッグ情報 ===")
            # print(f"response: {response}")
            # print(f"response_text: {response_text}")
            # print("=====================================\n")
            
            # AIMessageオブジェクトを作成
            ai_message = AIMessage(
                content=response_text if response_text else "エラー: 応答テキストが空です",
                additional_kwargs={
                    "node_info": {
                        "node_name": "unified_response_node",  # ノード名
                        "node_type": "user_facing",
                        "timestamp": datetime.now().isoformat(),
                    },
                    "error": "応答テキストが空です" if not response_text else None
                }
            )
            
            # Stateに情報を追加（既存のupdated_stateを使用）
            updated_state["messages"] = updated_state["messages"] + [ai_message]  # AIMessageを追加
            updated_state["response"] = response_text  # 応答テキストを追加
            updated_state["next_node"] = "end"  # 次のノードとして終了ノードを指定
            
            # response_textが空の場合は処理失敗とみなす
            if not response_text:
                updated_state["success"] = False
                updated_state["error"] = "応答テキストが空です"
                print("エラー: 応答テキストが空のため、処理を失敗とみなします。")
        
        # inactivity_timeoutをstateに追加
        updated_state["inactivity_timeout"] = inactivity_timeout
        
        return updated_state
    except Exception as e:
        print(f"統合ノードエラー: {str(e)}")
        # エラーが発生した場合はデフォルト値を設定
        error_message = f"エラーが発生したため、デフォルトの応答を返します: {str(e)}"
        
       
        # AIMessageオブジェクトを作成（エラー情報を含む）
        ai_message = AIMessage(
            content="ごめんなさい、エラーが発生しました。もう一度お願いできますか？",
            additional_kwargs={
                "node_info": {
                    "node_name": "unified_response_node",  # ノード名
                    "node_type": "user_facing",
                    "timestamp": datetime.now().isoformat(),
                },
                "error": str(e)
            }
        )
        
        # Stateに情報を追加（エラー情報を含む）
        updated_state = {
            **state,  # 既存の状態を維持
            "input_text": input_text,
            "files": [],  # エラー時は空のリスト
            "processed_input": "入力情報の処理に失敗しました",
            "messages": state.get("messages", []) + [ai_message],  # メッセージを追加
            "success": False,  # 処理が失敗したことを示すフラグを設定
            "response": "ごめんなさい、エラーが発生しました。もう一度お願いできますか？",
            "next_node": "end"  # エラー時は終了ノードを指定
        }
        
        return updated_state
