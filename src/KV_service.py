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

from flask import Flask, request, jsonify, abort
from pathlib import Path
import hashlib
import re
import threading

app = Flask(__name__)

# PathConfigとsettingsをグローバルで保持
path_config = None
kv_port = None

# キーに使える文字（英数字 + _ - のみ）
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# HTTPエラーレスポンスらしき値を拒否するパターン（例: "404 Not Found", "500 Internal Server Error"）
ERROR_VALUE_PATTERN = re.compile(r"^\s*[45]\d\d\s+[A-Za-z]")

# 読み出し〜書き込みの間に別の書き手が割り込まないようにする（同一プロセス内の直列化）。
# 別プロセスからの割り込みは If-Match（楽観ロック）で弾く
_write_lock = threading.Lock()

# 値が存在しないキーの ETag。新規作成を狙う書き手はこれを If-Match に送る
ETAG_ABSENT = '"0"'


def compute_etag(value: str) -> str:
    """値から ETag を作る（内容が同じなら同じ ETag）。

    **値の構造は一切見ない**。KV は key→text の素朴なストアのままにしておきたいので、
    「裏作業の記録」のような特定の構造を知る作りにはしない（2026-08-11 設計判断）。
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f'"{digest}"'


def current_etag(path: Path) -> str:
    """保存済みの値の ETag を返す。キーが無ければ ETAG_ABSENT。"""
    if not path.exists():
        return ETAG_ABSENT
    return compute_etag(path.read_text(encoding="utf-8"))


def initialize_kv_service(config, port):
    """
    KVサービスを初期化する
    
    Args:
        config: PathConfigインスタンス
        port: サービスのポート番号
    """
    global path_config, kv_port
    path_config = config
    kv_port = port
    
    print(f"KVサービスを初期化しました")
    print(f"  - KVデータディレクトリ: {path_config.kv_data_dir}")
    print(f"  - ポート番号: {kv_port}")


def key_to_path(key: str) -> Path:
    """
    キーからファイルパスを生成する
    
    Args:
        key: キー文字列
        
    Returns:
        Path: ファイルパス
    """
    if not KEY_PATTERN.match(key):
        abort(400, "key は英数字と _ と - のみ使用可能です")
    
    if path_config is None:
        abort(500, "KVサービスが初期化されていません")
    
    return path_config.kv_data_dir / f"{key}.txt"


@app.route("/kv", methods=["POST"])
def set_kv():
    """
    キーと値を保存するエンドポイント
    
    呼び出し例 (PORTは変数で指定されたポート番号):
    curl -X POST http://127.0.0.1:{PORT}/kv \
        -H "Content-Type: application/json" \
        -d '{"key": "user_name", "value": "Miku"}'
    
    Python例:
    import requests
    response = requests.post(f'http://127.0.0.1:{PORT}/kv',
                            json={'key': 'user_name', 'value': 'Miku'})
    print(response.json())  # {'ok': True}

    楽観ロック（任意）:
        `If-Match` ヘッダに GET で受け取った ETag を添えると、その間に他の書き手が
        更新していた場合に **409 Conflict** を返して上書きを防ぐ（読んで直して書き戻す
        処理での lost update 対策）。ヘッダを付けなければ従来どおり無条件で上書きする。
        新規作成を狙うときは `If-Match: "0"`（ETAG_ABSENT）を送る。
    """
    data = request.get_json(force=True, silent=True)
    if not data or "key" not in data or "value" not in data:
        abort(400, "JSON で key と value を指定してください")

    key = data["key"]
    value = str(data["value"])

    if ERROR_VALUE_PATTERN.match(value):
        abort(400, "HTTPエラーレスポンスらしき値は保存できません")

    path = key_to_path(key)
    if_match = request.headers.get("If-Match")

    with _write_lock:
        if if_match is not None:
            now = current_etag(path)
            if if_match != now:
                # 呼び出し側が読み直せるよう、現在の ETag を添えて返す
                return jsonify({
                    "ok": False,
                    "error": "conflict",
                    "message": "読み出し後に別の書き手が更新しました。読み直してやり直してください",
                    "etag": now,
                }), 409, {"ETag": now}
        path.write_text(value, encoding="utf-8")
        new_etag = compute_etag(value)

    return jsonify({"ok": True, "etag": new_etag}), 200, {"ETag": new_etag}


@app.route("/kv", methods=["GET"])
def get_kv():
    """
    キーから値を取得するエンドポイント
    
    呼び出し例 (PORTは変数で指定されたポート番号):
    curl -X GET "http://127.0.0.1:{PORT}/kv?key=user_name"
    
    Python例:
    import requests
    response = requests.get(f'http://127.0.0.1:{PORT}/kv',
                           params={'key': 'user_name'})
    print(response.text)  # Miku

    レスポンスには `ETag` ヘッダが付く。読んで直して書き戻す処理では、この値を
    POST の `If-Match` に添えると割り込み更新を検出できる（楽観ロック）。
    """
    key = request.args.get("key")
    if not key:
        abort(400, "key クエリパラメータが必要です")

    path = key_to_path(key)
    if not path.exists():
        abort(404, "Key not found")

    value = path.read_text(encoding="utf-8")
    return value, 200, {'Content-Type': 'text/plain; charset=utf-8',
                        'ETag': compute_etag(value)}


@app.route("/kv", methods=["DELETE"])
def delete_kv():
    """
    キーを削除するエンドポイント
    
    呼び出し例 (PORTは変数で指定されたポート番号):
    curl -X DELETE "http://127.0.0.1:{PORT}/kv?key=user_name"
    
    Python例:
    import requests
    response = requests.delete(f'http://127.0.0.1:{PORT}/kv',
                              params={'key': 'user_name'})
    print(response.json())  # {'ok': True}
    """
    key = request.args.get("key")
    if not key:
        abort(400, "key クエリパラメータが必要です")

    path = key_to_path(key)
    if not path.exists():
        abort(404, "Key not found")

    path.unlink()
    return jsonify({"ok": True})


@app.route("/list", methods=["GET"])
def list_keys():
    """
    保存されている全キーをリスト表示するエンドポイント
    
    呼び出し例 (PORTはsettings.jsonで指定されたポート番号):
    curl -X GET http://127.0.0.1:{PORT}/list
    
    Python例:
    import requests
    response = requests.get(f'http://127.0.0.1:{PORT}/list')
    print(response.json())  # {'keys': ['user_name', 'settings', ...]}
    """
    if path_config is None:
        abort(500, "KVサービスが初期化されていません")
    
    keys = [f.stem for f in path_config.kv_data_dir.glob("*.txt")]
    return jsonify({"keys": keys})


@app.route("/health", methods=["GET"])
def health_check():
    """
    ヘルスチェックエンドポイント
    """
    return jsonify({
        "status": "ok",
        "service": "KV Service",
        "initialized": path_config is not None
    })


def run_kv_service(config, port, host="127.0.0.1"):
    """
    KVサービスを実行する（gateway_service.py から呼ばれる）

    Args:
        config: PathConfigインスタンス
        port: サービスのポート番号
        host: バインドするホスト（127.0.0.1 固定で使う。認証が無いので
              同じ Wi-Fi の他の端末から会話ログを読ませない・2026-10-08）
    """
    initialize_kv_service(config, port)
    
    # Flask開発サーバーのログを抑制
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    print(f"KVサービスを起動します: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


def main() -> None:
    """単体起動の入口。配布物では `.pyd` 化されるため、起動処理は関数に置く。

    ここで `path_config` に代入してもモジュールのグローバルは書き換わらないが、
    **`run_kv_service()` が引数から `global path_config` を設定する**ので問題ない。
    """
    import sys
    from pathlib import Path
    
    # srcディレクトリをPythonパスに追加
    src_dir = Path(__file__).parent
    sys.path.insert(0, str(src_dir))
    
    # PathConfigを初期化
    from utils.path_config import PathConfig
    path_config = PathConfig.initialize(src_dir)
    path_config.ensure_directories()
    
    # settings.jsonからポート番号を取得
    import json
    try:
        with open(str(path_config.settings_file), 'r', encoding='utf-8-sig') as f:   # BOM 付きでも読む
            settings = json.load(f)
        port = settings.get('services', {}).get('kv', {}).get('port', 8010)
    except Exception as e:
        print(f"設定ファイル読み込みエラー: {e}")
        print("デフォルト設定を使用します")
        port = 8010
    
    print("=" * 60)
    print("KVサービス 単体起動")
    print("=" * 60)
    print(f"ポート: {port}")
    print(f"データディレクトリ: {path_config.kv_data_dir}")
    print("=" * 60)
    
    # サービスを実行
    run_kv_service(path_config, port)


if __name__ == "__main__":
    main()
