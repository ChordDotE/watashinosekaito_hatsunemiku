import json
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO)

class ConfigError(Exception):
    """設定関連のエラーを表すカスタム例外クラス"""
    pass

class ConfigManager:
    """設定ファイルの管理クラス"""
    
    def __init__(self, settings_file_path):
        """
        ConfigManagerを初期化する
        
        Args:
            settings_file_path (str): 設定ファイルのパス
        """
        self.settings_file = settings_file_path
        self.settings = self._load_settings()
    
    def _load_settings(self):
        """
        設定ファイルを読み込む
        
        Returns:
            dict: 設定データ
        
        Raises:
            ConfigError: 設定ファイルの読み込みに失敗した場合
        """
        try:
            # 設定ファイルを読み込む
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            return settings
        except Exception as e:
            logging.error(f"設定ファイルの読み込みエラー: {e}")
            raise ConfigError(f"設定ファイルの読み込みに失敗しました: {e}")
    
    def get_audio_output_device(self):
        """
        音声出力デバイス名を取得する
        
        Returns:
            str: 音声出力デバイス名
        """
        return self.settings.get('audio', {}).get('output_device')
    
    def get_webrtc_settings(self):
        """
        WebRTC設定を取得する
        
        Returns:
            dict: WebRTC設定
        """
        return self.settings.get('audio', {}).get('webrtc', {})
    
    def get_api_settings(self):
        """
        API設定を取得する
        
        Returns:
            dict: API設定
        """
        return self.settings.get('api', {})
    
    def get_default_llm_provider(self):
        """
        デフォルトのLLMプロバイダを取得する
        
        Returns:
            str: デフォルトのLLMプロバイダ名
        """
        return self.settings.get('api', {}).get('default_provider', 'openrouter')
