"""
API通信のログを記録するユーティリティモジュール
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# PathConfigクラスのインポート
from utils.path_config import PathConfig

class ApiLogger:
    """APIリクエストとレスポンスをログに記録するクラス"""
    
    @staticmethod
    def get_timestamp() -> str:
        """現在のタイムスタンプを取得"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def save_api_log(
        url: str,
        headers: Dict[str, str],
        request_data: Dict[str, Any],
        response_json: Dict[str, Any],
        timestamp: Optional[str] = None,
        api_name: str = "api"
    ) -> Optional[Path]:
        """
        APIリクエストとレスポンスをログファイルに保存
        
        Args:
            url: APIのURL
            headers: リクエストヘッダー（機密情報は削除済みであること）
            request_data: リクエストデータ
            response_json: レスポンスデータ
            timestamp: タイムスタンプ（省略時は現在時刻）
            api_name: API名（ログファイル名のプレフィックス）
            
        Returns:
            ログファイルのパス（保存に失敗した場合はNone）
        """
        try:
            # シングルトンインスタンスを取得
            path_config = PathConfig.get_instance()
            
            # print(f"APIログディレクトリ: {path_config.api_logs_dir}")
        except Exception as e:
            print(f"警告: PathConfigの取得に失敗しました: {str(e)}")
            return None
        
        # タイムスタンプが指定されていない場合は現在時刻を使用
        if timestamp is None:
            timestamp = ApiLogger.get_timestamp()
            
        # ログファイルのパスを作成
        log_file = path_config.api_logs_dir / f'{timestamp}_log_{api_name}.txt'
        
        try:
            # ディレクトリが存在することを確認
            log_dir = os.path.dirname(log_file)
            # print(f"ログディレクトリを作成: {log_dir}")
            os.makedirs(log_dir, exist_ok=True)
            
            # print(f"APIログファイルに書き込み: {log_file}")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== API Request ===\n")
                f.write(f"URL: {url}\n")
                f.write(f"Headers: {json.dumps(headers, ensure_ascii=False, indent=2)}\n")
                f.write(f"Data: {json.dumps(request_data, ensure_ascii=False, indent=2)}\n")
                f.write("==================\n\n")
                
                f.write("=== API Response ===\n")
                f.write(json.dumps(response_json, ensure_ascii=False, indent=2))
                f.write("\n===================\n")
            
            # print(f"APIログを保存しました: {log_file}")
            return log_file
            
        except Exception as e:
            print(f"APIログの保存に失敗しました: {str(e)}")
            return None
