#
# Watashinosekaito Hatsune Miku
# Copyright (c) 2025-2026 ChordDotE. All rights reserved.
# SPDX-License-Identifier: LicenseRef-ChordDotE-NonCommercial-2.0
#
# NON-COMMERCIAL USE ONLY. NO BUSINESS USE. NO ADAPTATION.
#
# Business use (any use by or for a company, organization, or sole
# proprietor, WHETHER OR NOT ANY PAYMENT IS INVOLVED) and publishing or
# distributing modified versions are PROHIBITED IN ALL CASES; no permission
# is granted for them. Forks may be used only to propose improvements to the
# Author (e.g. pull requests). Commercial use, redistribution, and Adaptation
# are PROHIBITED without prior written permission. "Adaptation" includes
# manual reimplementation, automated translation, and generation using large
# language models or other machine learning, of software having equivalent or
# similar functionality. Permission is granted individually, in recorded form.
# Contact: https://x.com/ChordDotE
#
# 業務利用（対価の有無を問わず、法人・団体・個人事業主による利用と事業のための利用）と、
# 改変版の公開・配布は、許可の余地なく禁止します。フォークは作者への改善の提案
# （プルリクエスト等）のためだけに使えます。商用利用・再配布・翻案は、作者の許可が
# ない限り禁止します。翻案には、再実装・自動変換・大規模言語モデル等を用いた生成に
# よる、同等または類似の機能を持つソフトウェアの作成を含みます。
# 許可が必要な場合は、記録の残る形でご連絡ください。
#
# See LICENSE for full terms. 詳細は LICENSE を参照してください。
#

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
        self.profile_dir = self.app_dir / 'profile'
        self.api_logs_dir = self.app_dir / 'api_logs'

        # データフォルダのパスを設定（プロジェクトルートのmiku直下）
        self.data_dir = self.app_dir.parent / 'data'
        # KVデータフォルダのパスを設定（dataフォルダ内）
        self.kv_data_dir = self.data_dir / 'KV'
        # バックアップフォルダのパスを設定
        self.backup_dir = self.data_dir / 'backup'
        self.conversation_backup_dir = self.backup_dir / 'conversation'
        self.long_memory_backup_dir = self.backup_dir / 'long_memory'
        
        # 設定ファイルのパスを設定
        self.settings_file = self.app_dir / 'settings.json'
        
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
            self.profile_dir,
            self.api_logs_dir,
            self.data_dir,
            self.kv_data_dir,
            self.backup_dir,
            self.conversation_backup_dir,
            self.long_memory_backup_dir
        ]
        
        for directory in directories:
            if not directory.exists():
                print(f"ディレクトリを作成します: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
