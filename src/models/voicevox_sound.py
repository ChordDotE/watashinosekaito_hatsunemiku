import requests
import json
import re
from pathlib import Path
import sys
import base64
import traceback
import MeCab
import ipadic
from datetime import datetime
import numpy as np
import wave
from scipy import signal

# VOICEVOXをインストールしたPCのホスト名
HOSTNAME = "127.0.0.1"

# RVCサーバーの設定
RVC_HOSTNAME = "127.0.0.1"
RVC_PORT = 18000

# venvディレクトリのパスを取得
venv_dir = Path(sys.executable).parent.parent  # Scriptsの親ディレクトリ（venv）を取得

# PathConfigをインポート
from utils.path_config import PathConfig

# パス設定の初期化
path_config = PathConfig.initialize(venv_dir)

def convert_with_rvc(input_file, output_file=None, target_sample_rate=48000):
    """
    RVCを使って音声ファイルを変換する関数
    
    Args:
        input_file (str): 入力音声ファイルのパス
        output_file (str, optional): 出力音声ファイルのパス。指定がない場合は自動生成
        target_sample_rate (int): 目標サンプリングレート（デフォルト: 48000Hz）
    
    Returns:
        dict: 処理結果を含む辞書
            - success (bool): 処理が成功したかどうか
            - message (str): 処理結果のメッセージ
            - file_path (str): 変換された音声ファイルのパス（成功時のみ）
    """
    # 処理開始時間を記録
    start_time = datetime.now()
    
    try:
        # 出力ファイルパスが指定されていない場合は自動生成
        if output_file is None:
            # 入力ファイルのパスから出力ファイルパスを生成
            input_path = Path(input_file)
            output_dir = input_path.parent
            output_file = str(output_dir / "output_voice.wav")
        
        print(f"RVC変換開始: 入力={input_file}, 出力={output_file}")
        
        # 1. ファイル読み込み
        with wave.open(input_file, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            audio_data = np.frombuffer(frames, dtype=np.int16)
        
        print(f"ファイル読み込み完了: チャンネル数={channels}, サンプル幅={sample_width}, "
              f"サンプルレート={framerate}Hz, フレーム数={n_frames}")
        
        # 2. サンプルレート変換
        if framerate != target_sample_rate:
            print(f"サンプルレート変換: {framerate}Hz → {target_sample_rate}Hz")
            # 変換後のサンプル数を計算
            target_n_frames = int(n_frames * target_sample_rate / framerate)
            # サンプルレート変換
            audio_data = signal.resample(audio_data, target_n_frames)
            # 変換後のサンプルレートを更新
            framerate = target_sample_rate
        else:
            print(f"サンプルレート変換不要: 既に{framerate}Hz")
        
        # 3. API変換
        # float32形式に変換
        audio_data_float32 = audio_data.astype(np.float32) / 32768.0
        # バイナリデータに変換
        audio_bytes = audio_data_float32.tobytes()
        
        # APIリクエスト
        files = {'waveform': ('audio.raw', audio_bytes, 'application/octet-stream')}
        headers = {'x-timestamp': '0'}
        
        # RVCサーバーのURLを構築
        rvc_url = f"http://{RVC_HOSTNAME}:{RVC_PORT}/api/voice-changer/convert_chunk"
        
        print(f"RVC APIリクエスト送信: {rvc_url}")
        response = requests.post(
            rvc_url,
            files=files,
            headers=headers
        )
        
        if response.status_code != 200:
            error_msg = f"RVC API変換失敗: ステータスコード={response.status_code}, レスポンス={response.text}"
            print(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'file_path': None
            }
        
        print(f"RVC API変換成功: レスポンスサイズ={len(response.content)}バイト")
        
        # 4. ファイル保存
        # レスポンスからバイナリデータを取得
        output_data = np.frombuffer(response.content, dtype=np.float32)
        # int16形式に変換
        output_int16 = (output_data * 32768.0).astype(np.int16)
        
        # waveファイルとして保存
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(framerate)
            wf.writeframes(output_int16.tobytes())
        
        print(f"RVC変換ファイル保存完了: {output_file}")
        
        # 処理終了時間を記録
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds() * 1000
        print(f"RVC変換全体の処理時間: {total_processing_time:.2f}ms")
        
        return {
            'success': True,
            'message': "RVC変換が完了しました",
            'file_path': output_file
        }
    
    except Exception as e:
        # 処理終了時間を記録（エラー時）
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds() * 1000
        print(f"RVC変換処理時間（エラー）: {total_processing_time:.2f}ms")
        
        error_msg = f"RVC変換エラー: {str(e)}"
        print(error_msg)
        print(f"エラー: {traceback.format_exc()}")
        
        return {
            'success': False,
            'message': error_msg,
            'file_path': None
        }

def generate_voice(text, speaker_id=10, filename="temp_voice",rvc_output_filename="output_voice"):
    """
    テキストを音声に変換する関数
    
    Args:
        text (str): 読み上げるテキスト
        speaker_id (int): 話者ID（デフォルト: 10）
        filename (str): 出力ファイル名（拡張子なし、デフォルト: "temp_voice"）
        rvc_output_filename (str, optional): RVC変換後の出力ファイル名（拡張子なし）。指定がない場合は"output_voice"
        
    Returns:
        dict: 処理結果を含む辞書
            - success (bool): 処理が成功したかどうか
            - message (str): 処理結果のメッセージ
            - file_path (str): 生成された音声ファイルのパス（成功時のみ）
    """
    # 処理開始時間を記録
    start_time = datetime.now()
    
    try:
        # 出力ディレクトリの設定（path_configから直接取得）
        output_dir = path_config.temp_voice_dir
        
        # 出力ディレクトリが存在することを確認
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 出力ファイルパスの設定
        wav_path = output_dir / f"{filename}.wav"
        
        # MeCabを使ってテキスト全体の読み仮名を取得（一度だけ呼び出し）
        print(f"MeCabを使って読み仮名を取得します: {text}")
        # 読み仮名取得の時間計測
        mecab_start_time = datetime.now()
        # kana_all = get_yomigana_from_llm(text)
        kana_all = get_yomigana_with_mecab(text)
        mecab_end_time = datetime.now()
        mecab_processing_time = (mecab_end_time - mecab_start_time).total_seconds() * 1000
        print(f"読み仮名取得処理時間: {mecab_processing_time:.2f}ms")
        
        # 「。」「！」「？」などで文章を区切り、各文の音声データを生成
        texts = re.split(r'(?<=[。！？])\s*', text)
        print(texts)
        kanas = re.split(r'(?<=[。！？])\s*', kana_all)
        print(kanas)
        
        # テキストと読み仮名の数が一致しない場合の対処
        use_kana = True
        if len(texts) != len(kanas):
            print(f"警告: テキストと読み仮名の分割数が一致しません: テキスト {len(texts)}文、読み仮名 {len(kanas)}文")
            print("テキストのみから音声を作成します")
            use_kana = False
        
        waves_data = []
        
        # 音声合成処理の時間計測（全体）
        synthesis_start_time = datetime.now()
        
        if use_kana:
            # 読み仮名を使用する場合の音声合成処理
            for i, (text_part, kana_part) in enumerate(zip(texts, kanas)):
                # 文字列が空の場合は処理しない
                if text_part == '' or kana_part == '':
                    continue
                
                # 各文の音声合成処理の時間計測
                part_start_time = datetime.now()
                    
                print(f"読み仮名を考慮して音声合成中 ({i+1}/{len(texts)}): {text_part}")
                print(f"読み仮名: {kana_part}")
                
                # ステップ1: かなテキストからアクセント句を取得
                accent_start_time = datetime.now()
                query_params = {"text": kana_part, "speaker": speaker_id}
                response = requests.post(
                    f"http://{HOSTNAME}:50021/accent_phrases", 
                    params=query_params
                )
                response.raise_for_status()
                accent_phrases = response.json()
                accent_end_time = datetime.now()
                accent_processing_time = (accent_end_time - accent_start_time).total_seconds() * 1000
                
                # ステップ2: 音声合成用のクエリを取得
                query_start_time = datetime.now()
                query_params = {"text": text_part, "speaker": speaker_id}
                response = requests.post(
                    f"http://{HOSTNAME}:50021/audio_query", 
                    params=query_params
                )
                response.raise_for_status()
                audio_query = response.json()
                query_end_time = datetime.now()
                query_processing_time = (query_end_time - query_start_time).total_seconds() * 1000
                
                # ステップ3: 音声合成クエリのアクセント句を読みがなベースのものに置き換え
                audio_query["accent_phrases"] = accent_phrases
                
                # ステップ4: 音声を合成
                synth_start_time = datetime.now()
                headers = {"Accept": "audio/wav", "Content-Type": "application/json"}
                query_params = {"speaker": speaker_id}
                response = requests.post(
                    f"http://{HOSTNAME}:50021/synthesis", 
                    headers=headers,
                    params=query_params, 
                    data=json.dumps(audio_query)
                )
                response.raise_for_status()
                synth_end_time = datetime.now()
                synth_processing_time = (synth_end_time - synth_start_time).total_seconds() * 1000
                
                # 音声データをBase64エンコードしてリストに追加
                waves_data.append(base64.b64encode(response.content).decode('utf-8'))
                
                # 各文の処理時間を出力
                part_end_time = datetime.now()
                part_processing_time = (part_end_time - part_start_time).total_seconds() * 1000
                print(f"文 {i+1}/{len(texts)} の処理時間: {part_processing_time:.2f}ms")
                # print(f"  - アクセント句取得: {accent_processing_time:.2f}ms")
                # print(f"  - クエリ取得: {query_processing_time:.2f}ms")
                # print(f"  - 音声合成: {synth_processing_time:.2f}ms")
        else:
            # 読み仮名を使用せず、テキストのみから音声を作成する場合
            for i, text_part in enumerate(texts):
                # 文字列が空の場合は処理しない
                if text_part == '':
                    continue
                
                # 各文の音声合成処理の時間計測
                part_start_time = datetime.now()
                    
                print(f"文章のみから音声合成中 ({i+1}/{len(texts)}): {text_part}")
                
                # 音声合成用のクエリを取得
                query_start_time = datetime.now()
                query_params = {"text": text_part, "speaker": speaker_id}
                response = requests.post(
                    f"http://{HOSTNAME}:50021/audio_query", 
                    params=query_params
                )
                response.raise_for_status()
                audio_query = response.json()
                query_end_time = datetime.now()
                query_processing_time = (query_end_time - query_start_time).total_seconds() * 1000
                
                # 音声を合成
                synth_start_time = datetime.now()
                headers = {"Accept": "audio/wav", "Content-Type": "application/json"}
                query_params = {"speaker": speaker_id}
                response = requests.post(
                    f"http://{HOSTNAME}:50021/synthesis", 
                    headers=headers,
                    params=query_params, 
                    data=json.dumps(audio_query)
                )
                response.raise_for_status()
                synth_end_time = datetime.now()
                synth_processing_time = (synth_end_time - synth_start_time).total_seconds() * 1000
                
                # 音声データをBase64エンコードしてリストに追加
                waves_data.append(base64.b64encode(response.content).decode('utf-8'))
                
                # 各文の処理時間を出力
                part_end_time = datetime.now()
                part_processing_time = (part_end_time - part_start_time).total_seconds() * 1000
                print(f"文 {i+1}/{len(texts)} の処理時間: {part_processing_time:.2f}ms")
                # print(f"  - クエリ取得: {query_processing_time:.2f}ms")
                # print(f"  - 音声合成: {synth_processing_time:.2f}ms")
        
        # 音声合成処理の時間計測（全体）
        synthesis_end_time = datetime.now()
        synthesis_total_time = (synthesis_end_time - synthesis_start_time).total_seconds() * 1000
        print(f"音声合成処理全体の時間: {synthesis_total_time:.2f}ms")
        
        # 音声データ結合の時間計測
        connect_start_time = datetime.now()
        
        # 音声データを結合
        if len(waves_data) > 1:
            res3 = requests.post(f'http://{HOSTNAME}:50021/connect_waves',
                                json=waves_data)
                                
            if res3.status_code != 200:
                return {
                    'success': False,
                    'message': f"connect_wavesエラー: {res3.text}",
                    'file_path': None
                }
                
            # 結果を保存
            with open(wav_path, 'wb') as f:
                f.write(res3.content)
        else:
            # 1つの音声データの場合は直接保存
            with open(wav_path, 'wb') as f:
                f.write(base64.b64decode(waves_data[0]))
        
        connect_end_time = datetime.now()
        connect_processing_time = (connect_end_time - connect_start_time).total_seconds() * 1000
        # print(f"音声データ結合処理時間: {connect_processing_time:.2f}ms")
        
        # print(f"音声ファイルを生成しました: {wav_path}")
        
        # 処理終了時間を記録
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds() * 1000
        print(f"音声生成全体の処理時間: {total_processing_time:.2f}ms")
        
        # VOICEVOXで生成した音声をRVCに投げる
        print(f"VOICEVOXで生成した音声をRVCに投げます: {wav_path}")
        
        # RVC出力ファイルパスの設定
        rvc_output_path = output_dir / f"{rvc_output_filename}.wav"
        
        # RVC変換を実行
        rvc_result = convert_with_rvc(
            input_file=str(wav_path),
            output_file=str(rvc_output_path)
        )
        
        # RVC変換が成功した場合は、RVC変換後のファイルパスを返す
        if rvc_result['success']:
            return {
                'success': True,
                'message': "音声ファイルを生成し、RVC変換しました",
                'file_path': rvc_result['file_path'],
            }
        else:
            # RVC変換に失敗した場合は、元のVOICEVOXファイルパスを返す
            print(f"警告: RVC変換に失敗しました: {rvc_result['message']}")
            return {
                'success': True,
                'message': "音声ファイルを生成しましたが、RVC変換に失敗しました",
                'file_path': str(wav_path),
            }
        
    except Exception as e:
        # 処理終了時間を記録（エラー時）
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds() * 1000
        print(f"音声生成処理時間（エラー）: {total_processing_time:.2f}ms")
        
        print(f"エラー: 音声合成エラー: {str(e)}")
        print(f"エラー: {traceback.format_exc()}")
        
        return {
            'success': False,
            'message': f"音声合成エラー: {str(e)}",
            'file_path': None
        }

def play_audio(file_path, device_name=None):
    """
    音声ファイルを再生する関数
    
    Args:
        file_path (str): 再生する音声ファイルのパス
        device_name (str): 出力デバイス名（デフォルト: None、指定がない場合はデフォルトデバイス）
        
    Returns:
        dict: 処理結果を含む辞書
            - success (bool): 処理が成功したかどうか
            - message (str): 処理結果のメッセージ
    """
    # 処理開始時間を記録
    start_time = datetime.now()
    
    try:
        import sounddevice as sd
        import wave
        import numpy as np
        
        # デバイスIDを取得
        device_id = None
        if device_name:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device_name.lower() in device['name'].lower() and device['max_output_channels'] > 0:
                    device_id = i
                    # print(f"出力デバイスを選択: {device['name']} (ID: {i})")
                    break
            if device_id is None:
                print(f"警告: 出力デバイス '{device_name}' が見つかりません。デフォルトデバイスを使用します。")

        # WAVファイルを読み込み
        with wave.open(file_path, 'rb') as wav_file:
            # パラメータを取得
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            # 音声データを読み込み
            wav_data = wav_file.readframes(wav_file.getnframes())
            # numpy配列に変換
            audio_data = np.frombuffer(wav_data, dtype=np.int16)
        
        # 音声再生の時間計測
        play_start_time = datetime.now()
        # 指定したデバイスで再生
        sd.play(audio_data, rate, device=device_id)
        sd.wait()  # 再生完了まで待機
        play_end_time = datetime.now()
        play_time = (play_end_time - play_start_time).total_seconds() * 1000
        # print(f"音声再生処理時間: {play_time:.2f}ms")
        
        # 処理終了時間を記録
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds() * 1000
        print(f"音声再生全体の処理時間: {total_processing_time:.2f}ms")
        
        return {
            'success': True,
            'message': "音声を再生しました"
        }
        
    except Exception as e:
        # 処理終了時間を記録（エラー時）
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds() * 1000
        print(f"音声再生処理時間（エラー）: {total_processing_time:.2f}ms")
        
        print(f"エラー: 音声再生エラー: {str(e)}")
        print(f"エラー: {traceback.format_exc()}")
        
        return {
            'success': False,
            'message': f"音声再生エラー: {str(e)}"
        }

# ApiLoggerをインポート
from utils.api_logger import ApiLogger

def get_yomigana_with_mecab(text):
    """
    MeCab + UniDicを使ってテキストから読み仮名を取得する関数（改良版）
    
    Args:
        text (str): 変換するテキスト
        
    Returns:
        str: 読み仮名
    """
    # 読み仮名だけを抽出するフォーマット指定
    # %f[7]は読み仮名のフィールド
    YOMI_ARGS = r' -F "%f[7]\n"'  
    YOMI_ARGS += r' -U "%m\n"'  # 未知語は表層形をそのまま出力
    
    # タガーの作成
    tagger = MeCab.Tagger(ipadic.MECAB_ARGS + YOMI_ARGS)
    
    # 解析実行
    lines = tagger.parse(text).splitlines()
    
    # EOSを除外して連結
    result = ''.join([line for line in lines if line != "EOS"])
    
    # 改行を削除して一つの文字列にする
    result = ''.join(result.split('\n'))

    # カタカナをひらがなに変換
    yomigana = ''
    for char in result:
        if 'ァ' <= char <= 'ヶ':
            # カタカナからひらがなへの変換（Unicode上で96（0x60）の差がある）
            yomigana += chr(ord(char) - 0x60)
        else:
            yomigana += char

    
    print(f"MeCabによる読み仮名変換結果: {yomigana}")
    return yomigana


def get_yomigana_from_llm(text):
    """
    LLMを使ってテキストから読み仮名を取得する関数
    
    Args:
        text (str): 変換するテキスト
        
    Returns:
        str: 読み仮名
    """
    try:
        # settings.jsonからAPIキーとモデル情報を読み込む
        # PathConfigを使用して設定ファイルのパスを取得
        settings_path = path_config.settings_file
        with open(str(settings_path), 'r') as f:
            settings = json.load(f)
        
        # OpenRouterのAPI情報を取得
        api_config = settings.get("api", {}).get("openrouter", {})
        api_url = api_config.get("url", "https://openrouter.ai/api/v1/chat/completions")
        api_key = api_config.get("api_key", "")
        model = api_config.get("models", {}).get("conversation")
        
        if not api_key:
            print("警告: OpenRouter APIキーが設定されていません。")
            return text
        
        # LLMを使って読み仮名を取得
        prompt = f"""
        以下のテキストをひらがなに変換してください。アルファベットや漢字から、ひらがなまたはカタカナに直してください。英単語やアルファベットも全てカタカナに直してください。
        意図としては、テキストを音声読み上げソフトに書けたいのですが、漢字や英単語、アルファベットは正しく発音できないためです。出てくるたびに、全てひらがな又はカタカナに直してください。
        漢字はひらがなに、英単語、アルファベットはカタカナにするのが望ましいです。
        記号については、通常読み上げるならばひらがなまたはカタカナにし、そうでないならばスキップしてください。
        逆に、それ以外の変更はひらがな又はカタカナにする以外の変更は一切加えないでください。
        例えば、!や?なども、半角/全角なども一切変えずに出力してください。
        単語の区切りにスペースをいれるなども不要です。

        変換元テキスト: {text}
        
        ひらがなのみで出力してください。余計な説明は不要です。

        # 悪い出力例（半角の「!」が全角の「！」として出力されているため）
        MORE MORE JUMP!→モアモアじゃんぷ！
        """
        
        # APIリクエストの準備
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Miku Agent"
        }
        
        messages = [
            {"role": "system", "content": "あなたはテキストをひらがなに変換するアシスタントです。"},
            {"role": "user", "content": prompt}
        ]
        
        data = {
            "model": model,
            "messages": messages
        }
        
        # APIリクエストを送信
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        # APIログを保存
        ApiLogger.save_api_log(
            url=api_url,
            headers={k: v for k, v in headers.items() if k != "Authorization"},  # 認証情報を除外
            request_data=data,
            response_json=result,
            api_name="openrouter_hiragana"
        )
        
        hiragana = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # 余計な説明などが含まれている場合は、できるだけ読み仮名のみを抽出
        # if "\n" in hiragana:
        #     hiragana = hiragana.split("\n")[0].strip()
        
        print(f"LLMによる読み仮名変換結果: {hiragana}")
        return hiragana
        
    except Exception as e:
        print(f"エラー: 読み仮名変換エラー: {str(e)}")
        print(f"エラー: {traceback.format_exc()}")
        # エラーの場合はテキストをそのまま返す
        return text
