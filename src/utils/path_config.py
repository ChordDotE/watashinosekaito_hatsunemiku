from pathlib import Path
import os
import logging

class PathConfigError(Exception):
    """パス設定関連のエラーを表すカスタム例外クラス"""
    pass

class PathConfig:
    """パス設定を管理するクラス"""
    
    # シングルトンインスタンスを保持するクラス変数
    _instance = None
    
    def __init__(self, app_dir):
        """
        PathConfigを初期化する
        
        Args:
            app_dir (Path): アプリケーションのルートディレクトリ
        """
        self.app_dir = app_dir
        
        print('self.app_dir',self.app_dir)

        # 各種ディレクトリのパスを設定（app_dirはすでにsrcディレクトリ）
        self.templates_dir = self.app_dir / 'templates'
        self.conversations_dir = self.app_dir / 'conversations'
        self.profile_dir = self.app_dir / 'profile'
        self.prompts_dir = self.app_dir / 'prompts'
        self.saved_index_dir = self.app_dir / 'saved_index'
        self.saved_models_dir = self.app_dir / 'saved_models'
        self.temp_voice_dir = self.app_dir / 'temp_voice'
        self.api_logs_dir = self.app_dir / 'api_logs'
        self.certs_dir = self.app_dir / 'certs'
        
        # 記憶関連のディレクトリを追加
        self.memory_dir = self.app_dir / 'memory'
        self.chroma_db_dir = self.memory_dir / 'chroma_db'
        self.langmem_db_dir = self.memory_dir / 'langmem_db'
        
        # ステートログ用のディレクトリを追加
        self.state_logs_dir = self.app_dir / 'state_logs'
        
        # 設定ファイルのパスを設定
        self.settings_file = self.app_dir / 'settings.json'
        
        # 証明書と秘密鍵のパスを設定
        self.cert_file = self.certs_dir / 'cert.pem'
        self.key_file = self.certs_dir / 'key.pem'
    
    @classmethod
    def initialize(cls, app_dir):
        """
        PathConfigのインスタンスを初期化する
        既にインスタンスが存在する場合は、そのインスタンスを返す
        
        Args:
            app_dir (Path): アプリケーションのルートディレクトリ
        
        Returns:
            PathConfig: 初期化されたPathConfigインスタンス
        """
        if cls._instance is None:
            cls._instance = cls(Path(app_dir))
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """
        PathConfigのインスタンスを取得する
        インスタンスが初期化されていない場合はエラーを発生させる
        
        Returns:
            PathConfig: 初期化されたPathConfigインスタンス
            
        Raises:
            PathConfigError: インスタンスが初期化されていない場合
        """
        if cls._instance is None:
            raise PathConfigError("PathConfigが初期化されていません。app.pyから初期化してください。")
        return cls._instance
    
    def ensure_directories(self):
        """
        必要なディレクトリが存在することを確認し、存在しない場合は作成する
        
        Note:
            - 既に存在するディレクトリには影響を与えません
            - exist_ok=Trueにより、既存ディレクトリがあってもエラーになりません
            - ディレクトリ内の既存ファイルは一切変更されません
        """
        directories = [
            self.templates_dir,
            self.conversations_dir,
            self.profile_dir,
            self.prompts_dir,
            self.saved_index_dir,
            self.saved_models_dir,
            self.temp_voice_dir,
            self.api_logs_dir,
            self.certs_dir,
            self.memory_dir,
            self.chroma_db_dir,
            self.langmem_db_dir,
            self.state_logs_dir
        ]
        
        for directory in directories:
            if not directory.exists():
                print(f"ディレクトリを作成します: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
