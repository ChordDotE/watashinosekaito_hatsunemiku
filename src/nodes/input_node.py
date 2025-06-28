from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime
from utils.llm_utils import call_llm, parse_json_response
from nodes.registry import register_node
from langchain.schema import HumanMessage  # LangChainのメッセージクラスをインポート

@register_node(
    name="input",
    description="ユーザー入力とファイルを処理し、テキスト化する入力ノード",
    capabilities=["テキスト入力処理", "ファイル処理", "画像解析", "マルチモーダル入力処理"],
    input_requirements=["input_text", "files"],
    output_fields=["processed_input"]
)
def process_input(state: Dict[str, Any], input_text: str, files_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    ユーザーからの入力テキストとファイルを統合的に理解し、Stateに情報を追加する関数
    
    Args:
        state (Dict[str, Any]): 現在の状態
        input_text (str): ユーザーからの入力テキスト
        files_data (List[Dict[str, Any]], optional): 添付ファイルのblobデータリスト
        
    Returns:
        Dict[str, Any]: 更新された状態
    """
    # ファイルデータがない場合は空リストを使用
    if files_data is None:
        files_data = []
    
    # ファイルがない場合は早期リターン
    if not files_data:
        # additional_kwargsを作成
        additional_kwargs = {
            "node_info": {
                "node_name": "input_node",  # ノード名
                "node_type": "user_facing",
                "timestamp": datetime.now().isoformat(),
            },
            "file_info": "添付ファイルはありません。"
        }
        
        # HumanMessageオブジェクトを作成
        user_message = HumanMessage(
            content=input_text,
            additional_kwargs=additional_kwargs
        )
        
        # Stateに情報を追加
        updated_state = {
            **state,  # 既存の状態を維持
            "input_text": input_text,
            "files": [],  # 空のリスト
            "processed_input": f"ユーザーからの入力「{input_text}」に対する回答を生成します。添付ファイルはありません。",
            "messages": state.get("messages", []) + [user_message],  # HumanMessageオブジェクトを追加
            "success": True  # 処理が成功したことを示すフラグを設定
        }
        return updated_state
    
    # ファイル情報の文字列を作成
    files_info = ""
    file_extensions = []
    if files_data:
        file_extensions = [os.path.splitext(f.get("filename", ""))[1] for f in files_data]
        extensions_str = ", ".join(file_extensions)
        files_info = f"{len(files_data)}個のファイルが添付されています。({extensions_str})"
    
    # 入力の解釈（LLMを使用）
    interpretation = interpret_with_llm(input_text, files_info, file_extensions, files_data)
    
    # エラーチェック
    if "error" in interpretation:
        print(f"入力解釈エラー: {interpretation.get('error')}")
        # エラーが発生した場合でも、Stateの構造に合わせて値を設定
        error_message = f"エラー: {interpretation.get('error')}"
        processed_input_str = "入力情報の処理に失敗しました"
        
        # additional_kwargsを作成（エラー情報を含む）
        additional_kwargs = {
            "node_info": {
                "node_name": "input_node",  # ノード名
                "node_type": "user_facing",
                "timestamp": datetime.now().isoformat(),
            }
        }
        if files_info:
            additional_kwargs["file_info"] = files_info
        additional_kwargs["error"] = error_message
        
        # HumanMessageオブジェクトを作成
        user_message = HumanMessage(
            content=input_text,
            additional_kwargs=additional_kwargs
        )
        
        # Stateに情報を追加（エラー情報を含む）
        updated_state = {
            **state,
            "input_text": input_text,
            "files": [],  # エラー時は空のリスト
            "processed_input": processed_input_str,
            "messages": state.get("messages", []) + [user_message],  # HumanMessageオブジェクトを追加
            "success": False  # 処理が失敗したことを示すフラグを設定
        }
        return updated_state
    
    # ファイル内容の説明を直接取得
    file_description = interpretation.get("file_content_description", "")
    if file_description:
        print(f"ファイル内容の説明: {file_description}")
    
    
    # additional_kwargsを作成
    additional_kwargs = {
        "node_info": {
            "node_name": "input_node",  # ノード名
            "node_type": "user_facing",
            "timestamp": datetime.now().isoformat(),
        }
    }
    if files_info:
        additional_kwargs["file_info"] = files_info
    if file_description and file_description != "ファイルなし":
        additional_kwargs["file_content"] = file_description
    
    combined_understanding = interpretation.get("combined_understanding", "")
    if combined_understanding and combined_understanding != "ファイルなし":
        additional_kwargs["understanding"] = combined_understanding
    
    # HumanMessageオブジェクトを作成（contentにはinput_textのみを含め、他の情報はadditional_kwargsに移動）
    user_message = HumanMessage(
        content=input_text,
        additional_kwargs=additional_kwargs
    )
    
    # processed_inputを文字列として設定
    processed_input_str = interpretation.get("combined_understanding", "")
    
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
            processed_file["description"] = interpretation.get("file_content_description", "画像の説明なし")
        elif file_data.get("type") == "音声":
            processed_file["description"] = "音声ファイル"
        else:
            processed_file["description"] = f"{file_data.get('type', '不明')}ファイル"
        
        processed_files.append(processed_file)
    
    # Stateに情報を追加
    updated_state = {
        **state,  # 既存の状態を維持
        "input_text": input_text,
        "files": processed_files,  # blobデータを除去し、説明を含めたファイル情報
        "processed_input": processed_input_str,  # JSON文字列として設定
        "messages": state.get("messages", []) + [user_message],  # HumanMessageオブジェクトを追加
        "success": True  # 処理が成功したことを示すフラグを設定
    }
    
    return updated_state

def interpret_with_llm(input_text: str, files_info: str, file_extensions: List[str], files_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    LLMを使用して入力情報を処理する
    
    Args:
        input_text (str): ユーザーの入力テキスト
        files_info (str): ファイル情報の文字列
        file_extensions (List[str]): ファイル拡張子のリスト
        files_data (List[Dict[str, Any]], optional): 添付ファイルのデータリスト
        
    Returns:
        Dict[str, Any]: 処理結果
    """
    # ファイルがない場合は直接結果を返す
    if not files_data:
        return {
            "file_content_description": "ファイルなし",
            "combined_understanding": "ファイルなし"
        }
    
    try:
        # LLMへのプロンプト作成
        prompt = f"""
        以下の情報をすべて読み込み、その情報すべてを文字として出力してください。特に添付された画像の内容を詳細に分析してください。JSON形式で回答してください。

        ユーザー入力: {input_text}
        
        添付ファイル情報: {files_info if files_info else "なし"}
        
        これらの情報をマルチモーダルとして理解し、以下の形式で回答してください:
        {{
            "file_content_description": "添付ファイルの内容の詳細な説明（画像なら写っている人物、物体、背景、テキストなど）。。ファイルがない場合は"ファイルなし"とすること。",
            "combined_understanding": "入力テキストとファイルから得られる本質的な理解のみを簡潔に記載。ファイル自体の詳細説明は含めないこと。ユーザーの意図や質問の本質を捉えた内容にすること。",
        }}
        """
        
        # システムプロンプト
        system_prompt = "あなたはユーザー入力を解析するアシスタントです。日本語で答えてください。JSON形式で回答してください。"
        
        # LLM呼び出し関数を使用
        response = call_llm(prompt, system_prompt, files_data, api_name="input_node")
        
        # 応答を返す
        return response
    except Exception as e:
        print(f"LLM解釈エラー: {str(e)}")
        # エラーが発生した場合はデフォルト値を返す
        return {
            "file_content_description": "画像の内容を解析できませんでした",
            "combined_understanding": "エラー",  # 入力テキストをそのまま理解として使用
            "error": str(e)  # エラー情報を追加
        }
