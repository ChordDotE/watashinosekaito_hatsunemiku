"""
プロンプト関連のユーティリティモジュール
プロンプトファイルの読み込みや管理を行う
"""
from utils.path_config import PathConfig

def load_prompt(prompt_file: str) -> str:
    """
    プロンプトファイルを読み込む関数
    
    Args:
        prompt_file (str): プロンプトファイルの名前
        
    Returns:
        str: プロンプトの内容
        
    Raises:
        FileNotFoundError: プロンプトファイルが見つからない場合
    """
    # シングルトンインスタンスを取得
    path_config = PathConfig.get_instance()
    
    # プロンプトディレクトリからファイルパスを構築
    prompt_path = path_config.prompts_dir / prompt_file
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        error_msg = f"エラー: {prompt_path}ファイルが見つかりません。"
        print(error_msg)
        raise FileNotFoundError(error_msg)
