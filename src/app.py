from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import os
import traceback
from pathlib import Path
import sys
import threading
import time


# 注意:このファイルはvenvを起動の上、...\srcの直下で起動すること。

# 音声ファイルディレクトリをクリアする関数
def clear_voice_files():
    """音声ファイルディレクトリ内のすべてのファイルを削除"""
    try:
        # パス設定が初期化されていない場合は何もしない
        if 'path_config' not in globals():
            print("パス設定が初期化されていないため、音声ファイルのクリアをスキップします")
            return
            
        voice_dir = path_config.temp_voice_dir
        if voice_dir.exists():
            files = list(voice_dir.glob('*.wav'))
            for file_path in files:
                try:
                    os.remove(file_path)
                    print(f"音声ファイルを削除しました: {file_path}")
                except Exception as e:
                    print(f"ファイル削除エラー: {str(e)}")
            print(f"合計 {len(files)} 個の音声ファイルを削除しました")
    except Exception as e:
        print(f"音声ファイルディレクトリのクリーンアップエラー: {str(e)}")

# Flaskアプリケーションの初期化
app = Flask(__name__, static_url_path='/static', 
            static_folder='templates/materials')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PERMANENT_SESSION_LIFETIME'] = 60000  # 1000分

# SocketIOの初期化
socketio = SocketIO(app, cors_allowed_origins="*")

# 接続中のクライアントを管理する辞書
connected_clients = {}
# WebSocketセッションIDとクライアントセッションIDのマッピング
session_mapping = {}  # {socketio_session_id: client_session_id}
# グローバルタイマー管理（マスタータイマー方式）
global_timer = None
active_session_id = None
last_activity_time = None

def set_active_session(session_id):
    """
    アクティブセッションを設定し、既存のグローバルタイマーをキャンセルする関数
    
    Args:
        session_id (str): 新しいアクティブセッションID
    """
    global active_session_id, global_timer, last_activity_time
    
    try:
        # 既存のグローバルタイマーをキャンセル
        if global_timer is not None:
            global_timer.cancel()
            global_timer = None
            print("既存のグローバルタイマーをキャンセルしました")
        
        # 新しいアクティブセッションを設定
        active_session_id = session_id
        last_activity_time = time.time()
        
        print(f"アクティブセッションを設定しました: {session_id}")
        
    except Exception as e:
        print(f"アクティブセッション設定エラー: {str(e)}")

def inactivity_reminder_callback(session_id, response_text):
    """
    無応答リマインダーのコールバック関数（アクティブセッションのみに送信）
    
    Args:
        session_id (str): セッションID
        response_text (str): 生成された応答テキスト
    """
    global active_session_id
    
    try:
        # アクティブセッションのみに送信
        if session_id != active_session_id:
            print(f"非アクティブセッション {session_id} への送信をスキップしました (アクティブ: {active_session_id})")
            return
        
        print(f"無応答リマインダーを送信中 (アクティブセッション: {session_id}): {response_text}")
        
        # WebSocketを通じてアクティブセッションのみに無応答リマインダーを送信
        socketio.emit('inactivity_reminder', {
            'response': response_text,
            'session_id': session_id,
            'timestamp': time.time()
        })
        
        # 音声合成も実行
        if global_voice_generator is not None:
            try:
                global_voice_generator.generate_with_callback(
                    text=response_text,
                    speaker_id=10,
                    callback=lambda file_path, index, is_last: voice_file_callback(file_path, index, is_last, session_id),
                    play_audio=False
                )
                print(f"無応答リマインダーの音声生成を開始しました (セッション: {session_id})")
            except Exception as voice_error:
                print(f"無応答リマインダーの音声生成エラー: {str(voice_error)}")
        
    except Exception as e:
        print(f"無応答リマインダーコールバックエラー: {str(e)}")

def start_global_timer(session_id, timeout_seconds):
    """
    グローバルタイマーを開始する関数（マスタータイマー方式）
    
    Args:
        session_id (str): セッションID
        timeout_seconds (int): タイムアウト秒数（-1の場合はタイマーを設定しない）
    """
    global global_timer, active_session_id
    
    try:
        # アクティブセッションを設定（既存のグローバルタイマーもキャンセルされる）
        set_active_session(session_id)
        
        # タイムアウトが-1以下の場合はタイマーを設定しない
        if timeout_seconds <= 0:
            print(f"グローバルタイマーは設定されません (timeout: {timeout_seconds})")
            return
        
        # 新しいグローバルタイマーを作成
        def timer_callback():
            from agent_main import process_agent_request
            global global_timer
            
            try:
                # 無応答リマインダーを生成
                reminder_response = process_agent_request(
                    "", 
                    [], 
                    is_auto_response=True,
                    is_inactivity_reminder=True
                )
                
                response_text = reminder_response.get('response', 'お疲れ様。何かお手伝いできることはある？')
                # アクティブセッションのみに送信
                inactivity_reminder_callback(active_session_id, response_text)
                
            except Exception as e:
                print(f"無応答リマインダー生成エラー: {str(e)}")
                # エラー時のデフォルトメッセージ
                default_message = "お疲れ様。何かお手伝いできることはある？"
                inactivity_reminder_callback(active_session_id, default_message)
            finally:
                # タイマー実行後はグローバルタイマーをクリア
                global_timer = None
        
        global_timer = threading.Timer(timeout_seconds, timer_callback)
        global_timer.start()
        
        print(f"グローバルタイマーを開始しました ({timeout_seconds}秒, アクティブセッション: {session_id})")
        
    except Exception as e:
        print(f"グローバルタイマー開始エラー: {str(e)}")

def start_inactivity_timer(session_id, timeout_seconds):
    """
    無応答タイマーを開始する関数（マスタータイマー方式に対応）
    
    Args:
        session_id (str): セッションID
        timeout_seconds (int): タイムアウト秒数（-1の場合はタイマーを設定しない）
    """
    # マスタータイマー方式でグローバルタイマーを開始
    start_global_timer(session_id, timeout_seconds)

def cancel_inactivity_timer(session_id):
    """
    無応答タイマーをキャンセルする関数（マスタータイマー方式に対応）
    
    Args:
        session_id (str): セッションID
    """
    global global_timer
    
    try:
        # グローバルタイマーをキャンセル
        if global_timer is not None:
            global_timer.cancel()
            global_timer = None
            print(f"グローバルタイマーをキャンセルしました (要求セッション: {session_id})")
    except Exception as e:
        print(f"グローバルタイマーキャンセルエラー: {str(e)}")

# 音声ファイル生成コールバック関数
def voice_file_callback(file_path, index, is_last=False, target_session_id=None):
    """音声ファイルが生成されたときに呼び出されるコールバック関数"""
    file_name = os.path.basename(file_path)
    
    if is_last:
        print(f"🔊 最終音声ファイル生成通知: {file_name}, インデックス: {index}")
    else:
        print(f"🔊 音声ファイル生成通知: {file_name}, インデックス: {index}")
    
    # WebSocketを通じてクライアントに通知
    try:
        socketio.emit('voice_file_ready', {
            'file_name': file_name,
            'index': index,
            'is_last': is_last,
            'targetSessionId': target_session_id  # 対象のセッションIDを追加
        })
        print(f"📡 WebSocket通知送信: ファイル {file_name}" + (f" (対象セッション: {target_session_id})" if target_session_id else ""))
    except Exception as e:
        print(f"❌ WebSocket通知エラー: {str(e)}")

# 現在のファイルのディレクトリを取得
src_dir = Path(__file__).parent  # app.pyが存在するディレクトリ（src）を取得

# 必要なモジュールのインポート
# 注意: 実際の実装では、これらのモジュールを作成する必要があります
try:
    from utils.path_config import PathConfig
    from models.webrtc_handler import process_offer, initialize_webrtc
    from models.audio_manager import synthesize_and_play_audio
    from models.memory_manager import process_all_conversations  # 追加
    import json
    
    # LangGraphのインポート
    from nodes.input_node import process_input
    from nodes.output_node import process_output
    from graph import conversation_graph
    
    # パス設定の初期化
    path_config = PathConfig.initialize(src_dir)
    path_config.ensure_directories()
    
    # 音声ファイルディレクトリをクリア
    clear_voice_files()
    
    # 設定ファイルを直接読み込む
    try:
        with open(str(path_config.settings_file), 'r', encoding='utf-8') as f:
            settings = json.load(f)
        print("設定ファイルを読み込みました")
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        settings = {}
    
    # 各モジュールの初期化
    webrtc_settings = initialize_webrtc(settings.get('audio', {}).get('webrtc', {}))
    
    # 音声処理のグローバルインスタンスを初期化
    try:
        from models.voice_player_manager import VoiceStreamGenerator, VoicePlayerManager
        
        # 設定ファイルから音声出力デバイスを取得
        output_device = settings.get('audio', {}).get('output_device')
        print(f"音声出力デバイス: {output_device}")
        
        # グローバル変数として音声処理インスタンスを初期化
        global_player_manager = VoicePlayerManager(device_name=output_device)
        global_voice_generator = VoiceStreamGenerator(player_manager=global_player_manager)
        print("音声処理インスタンスを初期化しました")
        
    except Exception as e:
        print(f"音声処理インスタンス初期化エラー: {str(e)}")
        traceback.print_exc()
        global_player_manager = None
        global_voice_generator = None
    
    # 起動時に会話を登録
    try:
        print("会話ファイルの処理を開始します...")
        # PathConfigから会話ディレクトリと記憶ディレクトリのパスを取得
        conversation_dir = str(path_config.conversations_dir)
        memory_dir = str(path_config.langmem_db_dir)
        
        # 会話ファイルを処理
        process_all_conversations(conversation_dir, memory_dir)
        print("会話ファイルの処理が完了しました")
    except Exception as e:
        print(f"会話ファイルの処理中にエラーが発生しました: {e}")
        traceback.print_exc()
    
    initialization_success = True
except Exception as e:
    print(f"初期化エラー: {str(e)}")
    traceback.print_exc()
    initialization_success = False

@app.route('/', methods=['GET'])
def index():
    """メインページを表示"""
    return render_template('index.html')

@app.route('/ping', methods=['POST'])
def ping():
    """疎通確認用のエンドポイント"""
    try:
        if not initialization_success:
            return jsonify({
                'success': False,
                'error': "システムの初期化に失敗しました。ログを確認してください。"
            }), 500
            
        # LangGraphを使用して疎通確認
        result = conversation_graph.invoke({"input_text": "", "is_ping": True})
        return jsonify({
            'success': True,
            'message': 'Connection established',
            'response': result["response"]
        })
    except Exception as e:
        print(f"Ping failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/generate', methods=['POST'])
def generate():
    """テキスト入力から応答を生成するエンドポイント"""
    try:
        if not initialization_success:
            return jsonify({
                'success': False,
                'error': "システムの初期化に失敗しました。ログを確認してください。"
            }), 500
            
        # フォームからテキスト入力とセッションIDを取得
        input_text = request.form['input_text']
        client_session_id = request.form.get('clientSessionId', None)
        
        # セッションIDのログ出力とアクティブセッション設定
        if client_session_id:
            print(f"クライアントセッションID: {client_session_id}")
            # アクティブセッションを設定（既存のグローバルタイマーもキャンセルされる）
            set_active_session(client_session_id)
        else:
            print("クライアントセッションIDが指定されていません")
        
        # 添付ファイルを処理
        files_data = []
        if 'file_upload[]' in request.files:
            files = request.files.getlist('file_upload[]')
            for file in files:
                if file and file.filename:
                    # ファイル情報を出力
                    file_size = 0
                    file.seek(0, os.SEEK_END)
                    file_size = file.tell()
                    file.seek(0)  # ファイルポインタを先頭に戻す
                    
                    # ファイルの種類を判定
                    file_type = "不明"
                    if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')):
                        file_type = "画像"
                    elif file.filename.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.aac')):
                        file_type = "音声"
                    
                    # ファイル情報を詳細に表示
                    print(f"添付ファイル: {file.filename}")
                    print(f"  - 種類: {file_type}")
                    print(f"  - サイズ: {file_size} バイト ({file_size / 1024:.1f} KB)")
                    print(f"  - MIME タイプ: {file.content_type if hasattr(file, 'content_type') else '不明'}")
                    
                    # ファイル情報を保存
                    files_data.append({
                        'filename': file.filename,
                        'type': file_type,
                        'size': file_size,
                        'content_type': file.content_type if hasattr(file, 'content_type') else '不明',
                        'content': file.read()  # ファイルの内容を読み込む
                    })
                    file.seek(0)  # ファイルポインタを先頭に戻す
        
        # agent_main.pyのモデルに入力とファイルを渡す
        from agent_main import process_agent_request
        
        # モデルに入力とファイルを渡して処理
        response = process_agent_request(input_text, files_data)
        
        # 応答テキストを取得
        response_text = response.get('response', '')
        
        # VOICEVOXで音声合成
        voice_files = []
        voice_file_names = []
        try:
            # グローバルインスタンスが初期化されているか確認
            if global_voice_generator is None:
                print("音声処理インスタンスが初期化されていません")
                raise Exception("音声処理インスタンスが初期化されていません")
            
            # 音声合成を実行（コールバック関数を使用）
            global_voice_generator.generate_with_callback(
                text=response_text,
                speaker_id=10,  # デフォルトの話者ID
                callback=lambda file_path, index, is_last: voice_file_callback(file_path, index, is_last, client_session_id),
                play_audio=False  # 再生はせず、ファイルのみ生成
            )
            
            # 注: ファイル名はWebSocketで送信されるため、ここでは空のリストを返す
            voice_file_names = []
            
            print("音声生成処理を開始しました（ファイルはWebSocketで通知されます）")
                
        except Exception as voice_error:
            print(f"音声処理エラー: {str(voice_error)}")
            import traceback
            print(traceback.format_exc())
        
        # 無応答タイマーを開始
        inactivity_timeout = response.get('inactivity_timeout', 120)

        if client_session_id and inactivity_timeout > 0:
            start_inactivity_timer(client_session_id, inactivity_timeout)
        
        # 成功レスポンスを返す
        return jsonify({
            'success': True,
            'response': response_text,
            'voice_files': voice_file_names,  # 音声ファイル名のリストを追加
            'conversation_file': response.get('session_file', ''),
            'inactivity_timeout': inactivity_timeout  # タイムアウト値も返す
        })
    except Exception as e:
        print(f"Generate response failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/voice_file/<filename>', methods=['GET'])
def serve_voice_file(filename):
    """音声ファイルを提供するエンドポイント"""
    try:
        # 安全なファイルパスの構築
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(str(path_config.temp_voice_dir), safe_filename)
        
        # ファイルが存在するか確認
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
        # 音声ファイルを送信
        return send_file(
            file_path,
            mimetype='audio/wav',
            as_attachment=False
        )
    except Exception as e:
        print(f"音声ファイル提供エラー: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/offer', methods=['POST'])
def offer():
    """WebRTC接続のオファーを処理するエンドポイント"""
    try:
        if not initialization_success:
            return jsonify({
                'success': False,
                'error': "システムの初期化に失敗しました。ログを確認してください。"
            }), 500
            
        # WebRTC処理は別モジュールに移動
        return process_offer(request.get_json(), webrtc_settings)
    except Exception as e:
        print(f"WebRTC offer failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# 会話履歴ダウンロード機能（セキュリティ強化版）- 現在は無効化
'''
@app.route('/download/<path:filename>')
def download_conversation(filename):
    """会話履歴ファイルをダウンロード（セキュリティ強化版）"""
    try:
        # ファイルパスを安全に構築（パストラバーサル攻撃の防止）
        safe_path = os.path.normpath(filename)
        if os.path.isabs(safe_path) or '..' in safe_path.split(os.path.sep):
            return "Invalid file path", 400
            
        # 会話ディレクトリ内のファイルのみアクセス可能に制限
        if not safe_path.startswith(str(path_config.conversations_dir)):
            safe_path = os.path.join(str(path_config.conversations_dir), safe_path)
            
        if not os.path.exists(safe_path) or not os.path.isfile(safe_path):
            return "File not found", 404
            
        return send_file(
            safe_path,
            as_attachment=True,
            download_name=os.path.basename(safe_path)
        )
    except Exception as e:
        return str(e), 500
'''

# WebSocketイベントハンドラー
@socketio.on('connect')
def handle_connect():
    """WebSocket接続時の処理"""
    print(f"WebSocket接続: {request.sid}")
    connected_clients[request.sid] = {
        'connected_at': time.time(),
        'last_activity': time.time()
    }

@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket切断時の処理"""
    global active_session_id, global_timer, session_mapping
    
    socketio_session_id = request.sid
    print(f"WebSocket切断: {socketio_session_id}")
    
    # マッピングからクライアントセッションIDを取得
    client_session_id = session_mapping.get(socketio_session_id)
    
    if client_session_id:
        print(f"切断されたクライアントセッションID: {client_session_id}")
        
        # 切断されたセッションがアクティブセッションの場合
        if active_session_id == client_session_id:
            print(f"アクティブセッション {client_session_id} が切断されました - グローバルタイマーをキャンセルします")
            
            # グローバルタイマーをキャンセル
            if global_timer is not None:
                global_timer.cancel()
                global_timer = None
                print("WebSocket切断によりグローバルタイマーをキャンセルしました")
            
            # アクティブセッションをクリア
            active_session_id = None
        
        # マッピングから削除
        del session_mapping[socketio_session_id]
        print(f"セッションマッピング削除: {socketio_session_id}")
    else:
        print(f"警告: 切断されたWebSocketセッション {socketio_session_id} に対応するクライアントセッションIDが見つかりません")
    
    # 接続リストから削除
    if socketio_session_id in connected_clients:
        del connected_clients[socketio_session_id]

@socketio.on('session_activate')
def handle_session_activate(data):
    """セッションアクティブ化イベント（新規接続時やページロード時）"""
    try:
        # クライアントセッションIDを取得（指定がなければWebSocketセッションIDを使用）
        client_session_id = data.get('sessionId', request.sid)
        socketio_session_id = request.sid
        
        print(f"セッションアクティブ化（クライアントセッションID）: {client_session_id}")
        print(f"WebSocketセッションID: {socketio_session_id}")
        
        # WebSocketセッションIDとクライアントセッションIDのマッピングを記録
        session_mapping[socketio_session_id] = client_session_id
        print(f"セッションマッピング追加: {socketio_session_id} -> {client_session_id}")
        
        # アクティブセッションを設定
        set_active_session(client_session_id)
        
        # 確認応答を送信
        emit('session_activated', {
            'sessionId': client_session_id,
            'timestamp': time.time(),
            'message': 'セッションがアクティブになりました'
        })
        
    except Exception as e:
        print(f"セッションアクティブ化エラー: {str(e)}")
        emit('error', {'message': 'セッションアクティブ化に失敗しました'})

# アプリケーション終了時のクリーンアップ関数
def cleanup():
    global global_timer
    
    # グローバルタイマーをキャンセル
    if global_timer is not None:
        global_timer.cancel()
        global_timer = None
        print("グローバルタイマーをキャンセルしました")
    
    if 'global_player_manager' in globals() and global_player_manager is not None:
        global_player_manager.stop()
        print("音声再生を停止しました")

if __name__ == '__main__':
    # 証明書と鍵のパスを取得
    cert_path = str(path_config.cert_file)
    key_path = str(path_config.key_file)
    
    # 証明書と鍵が存在するか確認
    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        print(f"証明書または秘密鍵が見つかりません。HTTPSは無効化されます。")
        print(f"証明書を生成するには、generate_cert.pyを実行してください。")
        ssl_context = None
    else:
        # SSLコンテキストを作成
        ssl_context = (cert_path, key_path)
        print(f"HTTPSが有効化されました。")
    
    # アプリケーション終了時にクリーンアップ関数を呼び出す
    import atexit
    atexit.register(cleanup)
    
    # 注意: 本番環境では debug=False にし、必要に応じて host も変更してください
    # socketio.run(app, debug=True, host='0.0.0.0', port=5001, ssl_context=ssl_context)
    socketio.run(app, debug=False, host='0.0.0.0', port=5001)
