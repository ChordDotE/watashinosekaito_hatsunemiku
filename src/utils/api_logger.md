# api_logger.py の機能説明

## 概要

`api_logger.py`は、APIリクエストとレスポンスをログに記録するためのユーティリティモジュールです。OpenRouter APIなどの外部APIとの通信内容を記録し、デバッグや監査のために使用されます。

## 主要コンポーネント

### ApiLoggerクラス

APIリクエストとレスポンスをログに記録するための静的メソッドを提供するクラスです。

#### 静的メソッド

##### get_timestamp()

現在のタイムスタンプを取得します。

- **戻り値**: タイムスタンプ文字列（形式: "YYYYMMDD_HHMMSS"）

##### save_api_log()

APIリクエストとレスポンスをログファイルに保存します。

- **引数**:
  - `url`: APIのURL
  - `headers`: リクエストヘッダー（機密情報は削除済みであること）
  - `request_data`: リクエストデータ
  - `response_json`: レスポンスデータ
  - `timestamp`: タイムスタンプ（省略時は現在時刻）
  - `api_name`: API名（ログファイル名のプレフィックス）
- **戻り値**: ログファイルのパス（保存に失敗した場合はNone）

## 処理フロー

1. `save_api_log`メソッドが呼び出されます
   - 入力: URL、ヘッダー、リクエストデータ、レスポンスデータ、タイムスタンプ（オプション）、API名（オプション）
   - 処理: ログファイルにAPIリクエストとレスポンスを保存
   - 出力: ログファイルのパス

2. `PathConfig`を使用してログディレクトリを取得します
   - venvディレクトリのパスを取得
   - `PathConfig.initialize`を呼び出し
   - `api_logs_dir`を取得

3. タイムスタンプを取得します
   - 引数で指定されていない場合は`get_timestamp`メソッドを使用

4. ログファイルのパスを作成します
   - 形式: `{timestamp}_log_{api_name}.txt`

5. ログファイルにAPIリクエストとレスポンスを書き込みます
   - リクエスト情報（URL、ヘッダー、データ）
   - レスポンス情報（JSONデータ）

## ログファイルの形式

```
=== API Request ===
URL: https://api.example.com/endpoint
Headers: {
  "Content-Type": "application/json",
  "HTTP-Referer": "http://localhost:5000",
  "X-Title": "Miku Agent"
}
Data: {
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "system",
      "content": "あなたはユーザー入力を解析するアシスタントです。"
    },
    {
      "role": "user",
      "content": "こんにちは"
    }
  ]
}
==================

=== API Response ===
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20
  },
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "こんにちは！何かお手伝いできることはありますか？"
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
===================
```

## 使用例

```python
# OpenRouter APIを呼び出した後のログ保存
ApiLogger.save_api_log(
    url=api_url,
    headers={k: v for k, v in headers.items() if k != "Authorization"},  # 認証情報を除外
    request_data=data,
    response_json=result,
    api_name="openrouter_input"
)
```

## 特記事項

- 機密情報（APIキーなど）はログに保存されないように、ヘッダーから削除する必要があります
- ログファイルは`api_logs_dir`ディレクトリに保存されます
- ログファイル名には、タイムスタンプとAPI名が含まれます
- エラーハンドリングが実装されており、ログの保存に失敗した場合はNoneを返します
- デバッグ情報が出力されており、問題が発生した場合に原因を特定しやすくなっています
