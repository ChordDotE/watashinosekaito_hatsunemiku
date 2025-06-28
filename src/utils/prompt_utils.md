# prompt_utils.py の機能説明

## 概要

`prompt_utils.py`は、プロンプトファイルの読み込みや管理を行うユーティリティモジュールです。LLMに送信するプロンプトをファイルから読み込むことで、プロンプトの管理や更新を容易にします。

## 主要な機能

### プロンプト読み込み

`load_prompt`関数は、指定されたプロンプトファイルを読み込みます。主な処理手順は以下の通りです：

1. PathConfigが初期化されていない場合は初期化
2. プロンプトファイルのパスを構築
3. ファイルを読み込む
4. ファイルが見つからない場合はFileNotFoundErrorを発生させる

```python
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
    # 実装内容
```

## データフロー

1. **各ノードモジュール**から`load_prompt`が呼び出されます
   - 入力: プロンプトファイルの名前（例: "planner_prompt.txt"）
   - 処理: ファイルを読み込む
   - 出力: プロンプトの内容（文字列）

2. 読み込まれたプロンプトは**LLM呼び出し**に使用されます
   - システムプロンプトとして使用される場合が多い
   - ユーザープロンプトと組み合わせてLLMに送信される

## 使用例

```python
from utils.prompt_utils import load_prompt

# システムプロンプトを読み込む
system_prompt = load_prompt("planner_prompt.txt")

# ユーザープロンプトを構築
user_prompt = f"""
以下の情報を基に、次に何をすべきか判断してください。

## 最新のユーザー入力
{latest_input}

## 会話履歴
{conversation_history}

## 利用可能なノード
{available_nodes_str}
"""

# LLMを呼び出す
response = call_llm(user_prompt, system_prompt, api_name="planner_node")
```

## 特記事項

- プロンプトファイルは`prompts`ディレクトリに配置されます
- PathConfigを使用してファイルパスを管理しています
- グローバル変数`path_config`を使用して、PathConfigのインスタンスを保持しています
- ファイルが見つからない場合は明示的にエラーを発生させ、エラーメッセージを表示します
- UTF-8エンコーディングでファイルを読み込みます
