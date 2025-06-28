"""
LLM呼び出しに関するユーティリティモジュール
LangChainとLangGraphを使用してLLM呼び出しを実装
"""
from typing import Dict, List, Any, Optional, Union
import json
import base64
import re
from utils.api_logger import ApiLogger
from utils.path_config import PathConfig
from models.config_manager import ConfigManager

# LangChainとLangGraphのインポート
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

def validate_schema(data, schema, path=""):
    """JSONデータがスキーマに従っているかを検証する関数"""
    errors = []
    
    for key, value_schema in schema.items():
        current_path = f"{path}.{key}" if path else key
        
        # 必須フィールドのチェック
        if key not in data:
            if "required" in value_schema and value_schema.get("required", []):
                errors.append(f"{current_path} は必須フィールドです")
            continue
            
        # 型チェック
        if "type" in value_schema:
            expected_type = value_schema["type"]
            if expected_type == "object" and not isinstance(data[key], dict):
                errors.append(f"{current_path} はオブジェクト型である必要があります")
            elif expected_type == "string" and not isinstance(data[key], str):
                errors.append(f"{current_path} は文字列型である必要があります")
            elif expected_type == "boolean" and not isinstance(data[key], bool):
                errors.append(f"{current_path} は真偽値型である必要があります")
            elif expected_type == ["string", "null"] and not (isinstance(data[key], str) or data[key] is None):
                errors.append(f"{current_path} は文字列型またはnullである必要があります")
                
        # ネストされたオブジェクトの検証
        if "properties" in value_schema and isinstance(data[key], dict):
            nested_errors = validate_schema(data[key], value_schema["properties"], current_path)
            errors.extend(nested_errors)
    
    return errors

def parse_json_response(content: str, default_values: Optional[Dict[str, Any]] = None, expected_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLMのレスポンスからJSONを抽出する関数
    
    Args:
        content (str): LLMのレスポンス内容
        default_values (Dict[str, Any], optional): パース失敗時のデフォルト値
        expected_schema (Dict[str, Any], optional): 期待するJSONスキーマ
        
    Returns:
        Dict[str, Any]: 抽出されたJSON
        
    Raises:
        json.JSONDecodeError: JSONパースに失敗した場合
        ValueError: スキーマ検証に失敗した場合
    """
    # デバッグ情報: 入力コンテンツを出力
    # print("\n=== parse_json_response 入力 ===")
    # print(content)
    # print(f"入力タイプ: {type(content)}")
    # print("================================\n")
    
    # マークダウンのコードブロックを処理
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
        # print(f"コードブロック抽出: {json_str}")
        try:
            result = json.loads(json_str)
            # print(f"コードブロックからJSONを抽出しました: {result}")
            # print(f"結果タイプ: {type(result)}")
            
            # スキーマ検証
            if expected_schema and isinstance(result, dict):
                validation_errors = validate_schema(result, expected_schema)
                if validation_errors:
                    print(f"スキーマ検証エラー: {validation_errors}")
                    if default_values:
                        return default_values
                    raise ValueError(f"スキーマ検証エラー: {validation_errors}")
            
            return result
        except json.JSONDecodeError:
            # JSONパースエラーを再スロー（キャッチしない）
            print(f"コードブロックのJSONパースに失敗")
            # 失敗した場合は次の方法を試す
    
    # JSONブロックを抽出
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # print(f"JSONブロック抽出: {json_str}")
        try:
            result = json.loads(json_str)
            # print(f"JSONブロックを抽出しました: {result}")
            # print(f"結果タイプ: {type(result)}")
            
            # スキーマ検証を追加
            if expected_schema and isinstance(result, dict):
                validation_errors = validate_schema(result, expected_schema)
                if validation_errors:
                    print(f"スキーマ検証エラー: {validation_errors}")
                    if default_values:
                        return default_values
                    raise ValueError(f"スキーマ検証エラー: {validation_errors}")
            
            return result
        except json.JSONDecodeError:
            # JSONパースエラーを再スロー（キャッチしない）
            print(f"JSONブロックのパースに失敗")
            # 失敗した場合は次の方法を試す
    
    # JSONブロックが見つからない場合は全体をパース
    try:
        # print("コンテンツ全体をJSONとしてパース試行")
        result = json.loads(content)
        # print(f"コンテンツ全体をJSONとしてパースしました: {result}")
        # print(f"結果タイプ: {type(result)}")
        
        # スキーマ検証を追加
        if expected_schema and isinstance(result, dict):
            validation_errors = validate_schema(result, expected_schema)
            if validation_errors:
                print(f"スキーマ検証エラー: {validation_errors}")
                if default_values:
                    return default_values
                raise ValueError(f"スキーマ検証エラー: {validation_errors}")
        
        return result
    except json.JSONDecodeError:
        # JSONパースエラーを再スロー（キャッチしない）
        if default_values:
            print(f"デフォルト値を返します: {default_values}")
            return default_values
        print(f"JSONパースエラー: すべての方法でJSONパースに失敗しました")
        raise

def call_llm(
    state: Dict[str, Any],
    system_prompt: Optional[List[str]] = None,
    files_data: Optional[List[Dict[str, Any]]] = None,
    api_name: str = "",  # 呼び出し元を識別するためのAPI名
    llm_provider: str = "",  # デフォルトはなし。openrouter、geminiが選択可能
    expected_schema: Optional[Dict[str, Any]] = None  # 期待するJSONスキーマ
) -> Dict[str, Any]:
    """
    LangChainを使用してLLMを呼び出す関数
    
    Args:
        state (Dict[str, Any]): 現在の状態（messagesを含む）
        system_prompt (List[str], optional): システムプロンプトのリスト
        files_data (List[Dict[str, Any]], optional): 添付ファイルのデータリスト
        api_name (str, optional): APIログに記録する呼び出し元の名前
        llm_provider (str, optional): 使用するLLMプロバイダ（"openrouter"または"gemini"）
        expected_schema (Dict[str, Any], optional): 期待するJSONスキーマ
        
    Returns:
        Dict[str, Any]: 処理結果（JSONパース済みの辞書オブジェクト）
        
    Note:
        この関数は内部でparse_json_responseを呼び出し、LLMからの応答を
        JSONとしてパースした結果を返します。返り値はすでにパース済みの
        辞書オブジェクトであるため、呼び出し元で再度parse_json_responseを
        呼び出す必要はありません。
    """
    # 設定を読み込み
    path_config = PathConfig.get_instance()
    config_manager = ConfigManager(path_config.settings_file)
    api_settings = config_manager.get_api_settings()
    
    # メッセージの準備
    messages = []
    
    # システムプロンプトがある場合は追加
    if system_prompt:
        for prompt in system_prompt:
            messages.append(SystemMessage(content=prompt))
    
    # stateからメッセージを取得
    state_messages = state.get("messages", [])
    
    # stateのメッセージを適切なLangChainメッセージに変換
    for msg in state_messages:
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            if msg.type == "human":
                # 画像データがある場合はマルチモーダル処理
                if files_data and msg == state_messages[-1]:  # 最新のメッセージの場合のみ
                    content_parts = []
                    content_parts.append({"type": "text", "text": msg.content})
                    
                    for file_data in files_data:
                        if file_data.get("type") == "画像" and file_data.get("content"):
                            try:
                                # 画像データをBase64エンコード
                                image_content = file_data.get("content", b"")
                                image_b64 = base64.b64encode(image_content).decode('utf-8')
                                
                                # 画像のMIMEタイプを取得
                                mime_type = file_data.get("content_type")
                                
                                # 画像データを追加
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_b64}"
                                    }
                                })
                                # print(f"画像データを追加: {file_data.get('filename')} ({len(image_b64)} バイト)")
                            except Exception as img_error:
                                print(f"画像データの処理エラー: {str(img_error)}")
                    
                    if len(content_parts) > 1:
                        human_msg = HumanMessage(content=content_parts)
                        if hasattr(msg, 'additional_kwargs'):
                            human_msg.additional_kwargs = msg.additional_kwargs
                        messages.append(human_msg)
                    else:
                        human_msg = HumanMessage(content=msg.content)
                        if hasattr(msg, 'additional_kwargs'):
                            human_msg.additional_kwargs = msg.additional_kwargs
                        messages.append(human_msg)
                else:
                    human_msg = HumanMessage(content=msg.content)
                    if hasattr(msg, 'additional_kwargs'):
                        human_msg.additional_kwargs = msg.additional_kwargs
                        # print("human_msgに追加しました")
                        # print("human_msg",human_msg)
                    messages.append(human_msg)
            elif msg.type == "ai":
                ai_msg = AIMessage(content=msg.content)
                if hasattr(msg, 'additional_kwargs'):
                    ai_msg.additional_kwargs = msg.additional_kwargs
                messages.append(ai_msg)
            elif msg.type == "system":
                system_msg = SystemMessage(content=msg.content)
                if hasattr(msg, 'additional_kwargs'):
                    system_msg.additional_kwargs = msg.additional_kwargs
                messages.append(system_msg)
            elif msg.type == "function" or msg.type == "tool":
                # ツール名を取得
                name = getattr(msg, 'name', '不明なツール')
                

                use_gemini = True  # geminiを使う場合はtrue。geminiはrokeにtoolを許容していないため
                # Gemini使用時はtoolメッセージをsystemメッセージに変換
                if use_gemini:
                    # ツールメッセージをシステムメッセージに変換
                    formatted_content = f"ツール「{name}」の結果:\n{msg.content}"
                    system_msg = SystemMessage(content=formatted_content)
                    if hasattr(msg, 'additional_kwargs'):
                        system_msg.additional_kwargs = msg.additional_kwargs
                    messages.append(system_msg)
                    # print(f"ツールメッセージをシステムメッセージに変換: {name}")
                else:
                    # OpenAIなど他のLLMではFunctionMessageをそのまま使用
                    from langchain.schema import FunctionMessage
                    function_msg = FunctionMessage(name=name, content=msg.content)
                    if hasattr(msg, 'additional_kwargs'):
                        function_msg.additional_kwargs = msg.additional_kwargs
                    messages.append(function_msg)
    
    # デフォルトのプロバイダを取得（明示的に指定されていない場合）
    default_provider = ""
    if not llm_provider:
        default_provider = config_manager.get_default_llm_provider()
        print(f"デフォルトのLLMプロバイダを使用: {default_provider}")
    
    # LLMプロバイダに応じて処理を分岐
    if llm_provider == "openrouter" or (not llm_provider and default_provider == "openrouter"):
        # OpenRouter APIの設定
        openrouter_config = api_settings.get("openrouter", {})
        api_url = "https://openrouter.ai/api/v1"
        api_key = openrouter_config.get("api_key", "")
        model = openrouter_config.get("models", {}).get("conversation")
        
        if not api_key:
            print("警告: OpenRouter APIキーが設定されていません。")
            return {"error": "APIキーが設定されていません"}
        
        # OpenRouterのLLMを初期化
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_url
        )
    
    elif llm_provider == "gemini" or (not llm_provider and default_provider == "gemini"):
        # Gemini APIの設定
        gemini_config = api_settings.get("gemini", {})
        api_key = gemini_config.get("api_key", "")
        model = gemini_config.get("models", {}).get("conversation", "gemini-pro")
        
        if not api_key:
            print("警告: Gemini APIキーが設定されていません。")
            return {"error": "APIキーが設定されていません"}
        
        # Gemini APIの初期化
        try:
            # LangChain-Google-GenAIをインポート（インストールされていない場合はエラーメッセージを表示）
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # GeminiのLLMを初期化
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                convert_system_message_to_human=True  # Geminiはシステムメッセージを直接サポートしていないため
            )
        except ImportError:
            print("警告: langchain-google-genaiがインストールされていません。")
            return {"error": "Geminiを使用するには、'pip install langchain-google-genai google-generativeai'を実行してください。"}
    
    else:
        print(f"警告: サポートされていないLLMプロバイダです: {llm_provider or default_provider}")
        return {"error": f"サポートされていないLLMプロバイダです: {llm_provider or default_provider}"}
        
    try:
        # LLMを呼び出し
        response = llm.invoke(messages)
        
        # LLMからの生の応答をprint
        # print("\n=== LLMからの生の応答 ===")
        # print(response.content)
        # print("========================\n")
        
        # レスポンスオブジェクトの詳細情報を出力（デバッグ用）
        # print("\n=== レスポンスオブジェクトの詳細 ===")
        # print(f"response: {response}")
        
        # APIログを保存（メッセージの全属性を含める）
        # メッセージの全属性を取得
        messages_with_role = []
        for m in messages:
            if hasattr(m, "type") and hasattr(m, "content"):
                # LangChainのメッセージオブジェクトから全属性を取得
                message_data = {}
                
                # 基本属性を追加
                message_data["role"] = m.type
                message_data["content"] = m.content
                
                # その他の属性を追加
                for attr in dir(m):
                    # 特殊属性とメソッドを除外
                    if not attr.startswith('_') and not callable(getattr(m, attr)):
                        # 既に追加した属性は除外
                        if attr not in ["type", "content"]:
                            try:
                                attr_value = getattr(m, attr)
                                # JSONシリアライズ可能な値のみ追加
                                if attr_value is None or isinstance(attr_value, (str, int, float, bool, list, dict)):
                                    message_data[attr] = attr_value
                            except Exception as e:
                                # 取得できない属性は無視
                                print(f"属性 {attr} の取得に失敗: {str(e)}")
                
                messages_with_role.append(message_data)
            else:
                # 通常の文字列の場合はそのまま追加
                messages_with_role.append({"role": "user", "content": str(m)})
        
        # # デバッグ情報を出力
        # print("\n=== デバッグ情報 ===")
        # print(f"llm_provider: {llm_provider}")
        # print(f"api_settings: {api_settings}")
        # print(f"messages_with_role: {messages_with_role}")
        # print(f"response.content: {response.content}")
        # print("===================\n")
        
        # APIログを保存
        api_url_to_log = "https://generativelanguage.googleapis.com" if llm_provider == "gemini" else api_url
        
        # JSONシリアライズ可能なオブジェクトのみを抽出
        serializable_messages = []
        for m in messages_with_role:
            serializable_message = {}
            for key, value in m.items():
                try:
                    # 試験的にJSONシリアライズしてみる
                    json.dumps({key: value})
                    serializable_message[key] = value
                except TypeError:
                    # シリアライズできない場合は文字列に変換
                    serializable_message[key] = str(value)
            serializable_messages.append(serializable_message)
        
        ApiLogger.save_api_log(
            url=api_url_to_log,
            headers={"HTTP-Referer": "http://localhost:5000", "X-Title": "Miku Agent"},
            request_data={"messages": serializable_messages},
            response_json={"content": str(response.content)},
            api_name=api_name
        )
        
        # JSONパース処理を共通関数で行う
        # 例外が発生した場合は呼び出し元に伝播させる
        parsed_result = parse_json_response(response.content, expected_schema=expected_schema)
        # print("\n=== JSONパース結果 ===")
        # print(json.dumps(parsed_result, ensure_ascii=False, indent=2))
        # print("=====================\n")
        
        return parsed_result
            
    except Exception as e:
        print(f"LLM呼び出しエラー: {str(e)}")
        
        # エラー時もAPIログを保存
        try:
            # メッセージのroleとcontentを取得（成功時と同じ処理）
            messages_with_role = []
            for m in messages:
                if hasattr(m, "type") and hasattr(m, "content"):
                    # LangChainのメッセージオブジェクトからroleとcontentを取得
                    role = m.type
                    content = m.content
                    messages_with_role.append({"role": role, "content": content})
                else:
                    # 通常の文字列の場合はそのまま追加
                    messages_with_role.append({"role": "user", "content": str(m)})
            
            # エラー情報を含めてログを保存
            api_url_to_log = "https://generativelanguage.googleapis.com" if llm_provider == "gemini" else api_url
            
            # JSONシリアライズ可能なオブジェクトのみを抽出
            serializable_messages = []
            for m in messages_with_role:
                serializable_message = {}
                for key, value in m.items():
                    try:
                        # 試験的にJSONシリアライズしてみる
                        json.dumps({key: value})
                        serializable_message[key] = value
                    except TypeError:
                        # シリアライズできない場合は文字列に変換
                        serializable_message[key] = str(value)
                serializable_messages.append(serializable_message)
            
            ApiLogger.save_api_log(
                url=api_url_to_log,
                headers={"HTTP-Referer": "http://localhost:5000", "X-Title": "Miku Agent"},
                request_data={"messages": serializable_messages, "provider": llm_provider or default_provider},
                response_json={"error": str(e)},  # エラー情報をレスポンスとして記録
                api_name=f"{api_name}_error"  # エラーログであることを明示
            )
            
            # デバッグ情報を出力
            print("\n=== エラー時のデバッグ情報 ===")
            print(f"llm_provider: {llm_provider or default_provider}")
            print(f"messages_with_role: {json.dumps(messages_with_role, ensure_ascii=False)}")
            print("===================\n")
        
        except Exception as log_error:
            # ログ保存中にエラーが発生した場合
            print(f"エラーログの保存に失敗: {str(log_error)}")
        
        return {"error": str(e)}
