# voicevox_sound.py の機能説明

## 概要

`voicevox_sound.py`は、VOICEVOXのAPIを使用してテキストを音声に変換する機能を提供するモジュールです。app.pyから呼び出され、応答テキストを音声合成して再生します。

## インポートするモジュール

```python
import requests
import json
import re
from pathlib import Path
import sys
import base64
import traceback
import MeCab
import ipadic

from utils.path_config import PathConfig
```

## 実装済みの機能

### 初期化

モジュールの初期化時に、以下の処理が行われます：

1. **出力設定**
   - 処理: printを使用して標準出力に情報を表示
   - 出力: なし

2. **VOICEVOXホスト名の設定**
   - 処理: `HOSTNAME = "127.0.0.1"`を設定
   - 出力: なし

3. **パス設定の初期化**
   - 入力: venv_dir（仮想環境のディレクトリパス）
   - 処理: `PathConfig.initialize(venv_dir)`を呼び出し、パス設定を初期化
   - 出力: path_configインスタンス

### 読み仮名変換機能

#### MeCabを使った読み仮名変換機能

`get_yomigana_with_mecab`関数は、MeCab + IPAdicを使ってテキストを読み仮名に変換します。

- 入力:
  - text（変換するテキスト）
- 処理:
  1. 読み仮名だけを抽出するフォーマット指定を設定
  2. MeCab.Taggerを初期化（ipadic.MECAB_ARGSを使用）
  3. テキストを解析して読み仮名を取得
  4. EOSを除外して結果を連結
  5. カタカナをひらがなに変換
- 出力: 読み仮名

<!-- #### LLMを使った読み仮名変換機能

`get_yomigana_from_llm`関数は、LLMを使ってテキストを読み仮名に変換します。

- 入力:
  - text（変換するテキスト）
- 処理:
  1. settings.jsonからAPIキーとモデル情報を読み込む
  2. OpenRouterのAPI情報を取得
  3. LLMを使って読み仮名を取得するためのプロンプトを作成
  4. APIリクエストを送信
  5. 結果から読み仮名を抽出
  6. APIログを保存
- 出力: 読み仮名（エラーの場合は元のテキスト） -->

### 音声合成機能

`generate_voice`関数は、テキストを音声に変換します。

- 入力:
  - text（読み上げるテキスト）
  - speaker_id（話者ID、デフォルト: 10）
  - filename（出力ファイル名、デフォルト: "temp_voice"）
- 処理:
  1. 出力ディレクトリを設定（path_config.temp_voice_dirを使用）
  2. 出力ディレクトリが存在することを確認
  3. 出力ファイルパスを設定
  4. MeCabを使ってテキスト全体の読み仮名を取得（get_yomigana_with_mecab関数を呼び出し）
  5. テキストと読み仮名を「。」「！」「？」などで分割
  6. テキストと読み仮名の数が一致しない場合は少ない方に合わせる
  7. 各文に対して音声合成を実行
     - かなテキストからアクセント句を取得（accent_phrases API）
     - 音声合成用のクエリを取得（audio_query API）
     - 音声合成クエリのアクセント句を読みがなベースのものに置き換え
     - 音声を合成（synthesis API）
     - 音声データをBase64エンコードしてリストに追加
  8. 複数の音声データがある場合は結合（connect_waves API）
  9. 音声ファイルを保存
- 出力: 処理結果を含む辞書（success, message, file_pathフィールドを含む）

### 音声再生機能

`play_audio`関数は、音声ファイルを再生します。

- 入力:
  - file_path（再生する音声ファイルのパス）
  - device_name（出力デバイス名、デフォルト: None）
- 処理:
  1. sounddeviceライブラリを使用して音声を再生
  2. デバイス名から出力デバイスIDを検索
  3. WAVファイルを読み込み
  4. 音声データをnumpy配列に変換
  5. 指定したデバイスで再生
- 出力: 処理結果を含む辞書（success, messageフィールドを含む）

## 主要な処理フロー

1. app.pyの`/generate`エンドポイントから呼び出されます。
   - 入力: response_text（応答テキスト）
   - 処理: `generate_voice`関数を呼び出し、テキストを音声に変換
   - 出力: voice_result（音声合成結果）

2. `generate_voice`内で`get_yomigana_with_mecab`関数が呼び出されます。
   - 入力: text（変換するテキスト）
   - 処理: MeCabを使ってテキストを読み仮名に変換
   - 出力: kana_all（読み仮名）

3. 音声合成が成功した場合、`play_audio`関数が呼び出されます。
   - 入力: voice_result['file_path']（音声ファイルのパス）、output_device（出力デバイス名）
   - 処理: 音声ファイルを再生
   - 出力: play_result（音声再生結果）

## エラーハンドリング

- 音声合成処理中にエラーが発生した場合、エラーメッセージが標準出力に表示され、エラー情報を含む辞書が返されます。
- 音声再生処理中にエラーが発生した場合、エラーメッセージが標準出力に表示され、エラー情報を含む辞書が返されます。

## 特記事項

- VOICEVOXのAPIは、デフォルトでは`127.0.0.1:50021`で動作することを前提としています。
- 音声合成には、accent_phrases、audio_query、synthesis、connect_wavesの4つのAPIが使用されます。
- 読み仮名変換には、MeCab + IPAdicを使用しています。LLMを使用する方法もバックアップとして残されています。
- 音声ファイルは、path_config.temp_voice_dirディレクトリに保存されます。
- 音声再生には、sounddeviceライブラリが使用されます。
