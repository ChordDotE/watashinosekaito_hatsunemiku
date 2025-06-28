# message_validator.py の機能説明

## 概要

`message_validator.py`は、LangChainのメッセージオブジェクトの検証を行うユーティリティモジュールです。各ノードから出力されるメッセージが適切な形式であることを確認し、不適合の場合はエラーを発生させます。これにより、開発時の規約違反を検知し、メッセージの一貫性を保証します。

## 主要な機能

### メッセージ検証

`MessageValidator`クラスは、以下の2つの主要なメソッドを提供します：

1. **validate_message**: 単一のメッセージを検証します
   - LangChainのBaseMessageを継承しているか確認
   - additional_kwargsが辞書型であることを確認
   - node_info情報が存在することを確認
   - node_infoが辞書型であることを確認
   - 必須フィールド（node_name, node_type, timestamp）が存在することを確認

2. **validate_messages**: メッセージリストを検証します
   - messagesがリスト型であることを確認
   - 各メッセージに対してvalidate_messageを呼び出し
   - 検証成功時にはメッセージ数を出力

### エラー処理

`MessageValidationError`例外クラスは、メッセージ検証に失敗した場合に発生します。エラーメッセージには、以下の情報が含まれます：

- 無効なメッセージの形式
- additional_kwargsの問題
- node_info情報の欠落
- node_infoの形式の問題
- 必須フィールドの欠落

## 検証ルール

### メッセージオブジェクト

- LangChainのBaseMessageを継承している必要があります（HumanMessage, AIMessage, SystemMessage, FunctionMessageなど）
- 辞書や文字列などの他の形式は許可されません

### additional_kwargs

- 辞書型である必要があります
- 存在しない場合はエラーとなります

### node_info

- additional_kwargsに含まれている必要があります
- 辞書型である必要があります
- 以下の必須フィールドを含む必要があります：
  - **node_name**: ノード名（例: "input_node", "planner_node"）
  - **node_type**: ノードタイプ（例: "user_facing", "internal"）
  - **timestamp**: 処理時刻

## 使用例

### 単一メッセージの検証

```python
from utils.message_validator import MessageValidator, MessageValidationError

try:
    MessageValidator.validate_message(message)
    print("メッセージは有効です")
except MessageValidationError as e:
    print(f"メッセージ検証エラー: {str(e)}")
```

### メッセージリストの検証

```python
from utils.message_validator import MessageValidator, MessageValidationError

try:
    MessageValidator.validate_messages(messages)
    # 成功時は自動的に "メッセージ検証成功: X件のメッセージを検証しました" と出力されます
except MessageValidationError as e:
    print(f"メッセージリスト検証エラー: {str(e)}")
```

### agent_main.pyでの使用例

agent_main.pyでは、node_wrapper関数内でメッセージ検証を行っています：

```python
def node_wrapper(node_func, node_name):
    def wrapped_func(state):     
        # ノード関数を実行
        result = node_func(state)
        
        # 処理後のメッセージ検証
        try:
            if "messages" in result and result["messages"]:
                MessageValidator.validate_messages(result["messages"])
        except MessageValidationError as e:
            error_msg = f"{node_name}の処理後にメッセージ検証エラーが発生しました: {str(e)}"
            print(f"エラー: {error_msg}")

            # エラー情報を含む状態を返す
            return {
                **result,
                "success": False,
                "error": error_msg
            }
        
        return result
    
    return wrapped_func
```

## エラーログ

検証エラーが発生した場合、以下の情報がログファイル（message_validation_errors.log）に記録されます：

- タイムスタンプ
- ノード名
- エラーメッセージ

## 拡張方法

### 追加の検証ルール

必要に応じて、以下のような追加の検証ルールを実装できます：

1. **コンテンツの検証**: メッセージのcontent属性に対する検証
2. **型固有の検証**: メッセージタイプ（human, ai, system, function）に応じた追加の検証
3. **カスタムフィールドの検証**: 特定のノードに固有のadditional_kwargsフィールドの検証

### 検証レベルの設定

環境変数やコンフィグファイルを使用して、検証レベルを設定できます：

- **strict**: すべての検証ルールを適用
- **warning**: エラーではなく警告として出力
- **disabled**: 検証を無効化（本番環境など）

## 特記事項

- このモジュールは開発時の規約違反を検知するためのものであり、本番環境では警告のみを出力するか、無効化することも検討できます
- メッセージ検証は各ノードの処理後に行われ、エラーが発生した場合はログに記録されます
- 検証成功時には、検証したメッセージ数が出力されます
- agent_main.pyの最終結果に対しても検証が行われます
