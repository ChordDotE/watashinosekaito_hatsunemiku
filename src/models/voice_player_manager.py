import requests
import json
import re
import wave
import numpy as np
import sounddevice as sd
import threading
import time
import traceback
import os
from pathlib import Path
from datetime import datetime
import uuid
import base64
import sys
from typing import List, Optional, Dict, Any, Callable, Tuple
from scipy import signal
import MeCab
import ipadic
import concurrent.futures

# PathConfigをインポート
from utils.path_config import PathConfig

# VOICEVOXをインストールしたPCのホスト名
HOSTNAME = "127.0.0.1"

# RVCサーバーの設定
RVC_HOSTNAME = "127.0.0.1"
RVC_PORT = 18000

class OrderedVoiceQueue:
    """
    順序を保証する音声キュー管理クラス
    インデックス順に音声ファイルを再生キューに追加する
    """
    
    def __init__(self, player_manager):
        """
        初期化
        
        Args:
            player_manager (VoicePlayerManager): 音声再生マネージャー
        """
        self.player_manager = player_manager
        self.pending_files = {}  # インデックス -> ファイルパス
        self.next_index = 0  # 次に再生すべきインデックス
        self.lock = threading.Lock()
    
    def add_file(self, index: int, file_path: str) -> None:
        """
        インデックス付きでファイルを追加
        
        Args:
            index (int): ファイルのインデックス（再生順序）
            file_path (str): 音声ファイルのパス
        """
        with self.lock:
            self.pending_files[index] = file_path
            print(f"順序付きキューに追加: インデックス{index}, {file_path}")
            self._process_queue()
    
    def _process_queue(self) -> None:
        """キューを処理し、順序通りに再生"""
        while self.next_index in self.pending_files:
            file_path = self.pending_files[self.next_index]
            self.player_manager.add_file(file_path)
            print(f"順序通りに再生キューに追加: インデックス{self.next_index}, {file_path}")
            del self.pending_files[self.next_index]
            self.next_index += 1
    
    def reset(self) -> None:
        """キューをリセット"""
        with self.lock:
            self.pending_files = {}
            self.next_index = 0
            print("順序付きキューをリセットしました")


class VoicePlayerManager:
    """
    音声ファイルの再生を管理するクラス
    複数の音声ファイルを順次再生したり、再生状態を管理したりする
    """
    
    def __init__(self, device_name: Optional[str] = None):
        """
        初期化
        
        Args:
            device_name (str, optional): 出力デバイス名
        """
        self.device_name = device_name
        self.queue = []  # 再生待ちの音声ファイルキュー
        self.current_file = None  # 現在再生中のファイル
        self.is_playing = False  # 再生中かどうか
        self.device_id = self._get_device_id()
        self.play_thread = None  # 再生用スレッド
        self.lock = threading.Lock()  # スレッドセーフな操作のためのロック
        self.on_complete_callback = None  # 全ての再生が完了した時のコールバック
        
        # パス設定の初期化
        # models/の親ディレクトリ（src）を取得
        current_dir = Path(__file__).parent.parent
        self.path_config = PathConfig.initialize(current_dir)
    
    def _get_device_id(self) -> Optional[int]:
        """
        デバイス名からデバイスIDを取得
        
        Returns:
            int or None: デバイスID（見つからない場合はNone）
        """
        if not self.device_name:
            return None
            
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if self.device_name.lower() in device['name'].lower() and device['max_output_channels'] > 0:
                    print(f"出力デバイスを選択: {device['name']} (ID: {i})")
                    return i
            
            print(f"警告: 出力デバイス '{self.device_name}' が見つかりません。デフォルトデバイスを使用します。")
            return None
        except Exception as e:
            print(f"デバイスID取得エラー: {str(e)}")
            return None
    
    def add_file(self, file_path: str) -> None:
        """
        再生キューに音声ファイルを追加
        
        Args:
            file_path (str): 音声ファイルのパス
        """
        with self.lock:
            self.queue.append(file_path)
            print(f"キューに追加: {file_path}")
            
            # 再生中でなければ再生開始
            if not self.is_playing:
                self._start_playback()
    
    def add_files(self, file_paths: List[str]) -> None:
        """
        複数の音声ファイルをキューに追加
        
        Args:
            file_paths (list): 音声ファイルパスのリスト
        """
        with self.lock:
            self.queue.extend(file_paths)
            print(f"{len(file_paths)}個のファイルをキューに追加")
            
            # 再生中でなければ再生開始
            if not self.is_playing:
                self._start_playback()
    
    def _start_playback(self) -> None:
        """再生スレッドを開始"""
        if self.play_thread and self.play_thread.is_alive():
            return
            
        self.is_playing = True
        self.play_thread = threading.Thread(target=self._playback_thread)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def _playback_thread(self) -> None:
        """再生スレッドのメイン処理"""
        while True:
            # キューからファイルを取得
            with self.lock:
                if not self.queue:
                    self.is_playing = False
                    self.current_file = None
                    break
                
                self.current_file = self.queue.pop(0)
            
            # ファイルを再生
            try:
                self._play_file(self.current_file)
                
                # 再生完了後、ファイルを削除
                try:
                    if os.path.exists(self.current_file):
                        os.remove(self.current_file)
                        print(f"ファイルを削除しました: {self.current_file}")
                except Exception as e:
                    print(f"ファイル削除エラー: {str(e)}")
                    
            except Exception as e:
                print(f"再生エラー: {str(e)}")
                print(traceback.format_exc())
        
        # 全ての再生が完了したらコールバックを呼び出す
        if self.on_complete_callback:
            try:
                self.on_complete_callback()
            except Exception as e:
                print(f"コールバックエラー: {str(e)}")
    
    def _play_file(self, file_path: str) -> None:
        """
        音声ファイルを再生
        
        Args:
            file_path (str): 再生する音声ファイルのパス
        """
        try:
            # ファイル名からインデックスを抽出（temp_voice_X_YYYYYYYYまたはoutput_voice_X_YYYYYYYY形式を想定）
            file_name = os.path.basename(file_path)
            index_match = re.search(r'(?:temp|output)_voice_(\d+)_', file_name)
            index_str = index_match.group(1) if index_match else "不明"
            
            print(f"再生開始: インデックス{index_str}のファイル {file_path}")
            
            # WAVファイルを読み込み
            with wave.open(file_path, 'rb') as wav_file:
                # パラメータを取得
                rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                # 音声データを読み込み
                wav_data = wav_file.readframes(wav_file.getnframes())
                # numpy配列に変換
                audio_data = np.frombuffer(wav_data, dtype=np.int16)
            
            # 指定したデバイスで再生
            sd.play(audio_data, rate, device=self.device_id)
            sd.wait()  # 再生完了まで待機
            
            print(f"再生完了: インデックス{index_str}のファイル {file_path}")
        except Exception as e:
            print(f"ファイル再生エラー: {str(e)}")
            print(traceback.format_exc())
    
    def clear(self) -> None:
        """キューをクリア"""
        with self.lock:
            self.queue = []
            print("再生キューをクリアしました")
    
    def stop(self) -> None:
        """再生停止"""
        with self.lock:
            # 現在の再生を停止
            sd.stop()
            self.is_playing = False
            self.current_file = None
            self.queue = []
            print("再生を停止しました")
    
    def set_on_complete_callback(self, callback: Callable[[], None]) -> None:
        """
        全ての再生が完了した時のコールバックを設定
        
        Args:
            callback (callable): コールバック関数
        """
        self.on_complete_callback = callback
    
    def get_queue_length(self) -> int:
        """
        キューの長さを取得
        
        Returns:
            int: キューの長さ
        """
        with self.lock:
            return len(self.queue)
    
    def is_busy(self) -> bool:
        """
        再生中かどうかを取得
        
        Returns:
            bool: 再生中ならTrue
        """
        return self.is_playing


class VoiceStreamGenerator:
    """
    テキストから音声を生成し、ストリーミング再生するクラス
    """
    
    def __init__(self, player_manager: Optional[VoicePlayerManager] = None, hostname: str = HOSTNAME, rvc_hostname: str = RVC_HOSTNAME, rvc_port: int = RVC_PORT):
        """
        初期化
        
        Args:
            player_manager (VoicePlayerManager, optional): 音声再生マネージャー
            hostname (str): VOICEVOXサーバーのホスト名
            rvc_hostname (str): RVCサーバーのホスト名
            rvc_port (int): RVCサーバーのポート番号
        """
        self.player_manager = player_manager or VoicePlayerManager()
        self.hostname = hostname
        self.rvc_hostname = rvc_hostname
        self.rvc_port = rvc_port
        
        # 順序付きキュー管理クラスを初期化
        self.ordered_queue = OrderedVoiceQueue(self.player_manager)
        
        # パス設定の初期化
        # models/の親ディレクトリ（src）を取得
        current_dir = Path(__file__).parent.parent
        self.path_config = PathConfig.initialize(current_dir)
        
        # 出力ディレクトリの設定
        self.output_dir = self.path_config.temp_voice_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_voice_files(self, text: str, speaker_id: int = 10, play_audio: bool = True, callback=None) -> List[str]:
        """
        テキストから音声を生成する内部メソッド
        
        Args:
            text (str): 読み上げるテキスト
            speaker_id (int): 話者ID
            play_audio (bool): 音声を再生するかどうか
            callback (callable): ファイルが生成されるたびに呼び出されるコールバック関数
                                 callback(file_path, index, is_last) の形式で呼び出される
        
        Returns:
            list: 生成された音声ファイルパスのリスト
        """
        # 処理開始時間を記録
        start_time = datetime.now()
        
        try:
            # 再生する場合のみ順序付きキューをリセット
            if play_audio:
                self.ordered_queue.reset()
            
            # テキストを文に分割
            texts = re.split(r'(?<=[。！？])\s*', text)
            texts = [t for t in texts if t.strip()]  # 空の文を除外
            
            print(f"テキストを{len(texts)}個の文に分割しました")
            
            # 音声ファイルパスのリスト
            voice_files = []
            
            # 最初の文を同期的に処理して即座に再生開始
            if texts:
                first_text = texts[0]
                unique_id = str(uuid.uuid4())[:8]
                filename = f"temp_voice_0_{unique_id}"
                rvc_output_filename = f"output_voice_0_{unique_id}"
                
                print(f"最初の文の音声を生成中: {first_text}")
                first_result = self.generate_voice_part(
                    text_part=first_text,
                    speaker_id=speaker_id,
                    filename=filename,
                    rvc_output_filename=rvc_output_filename
                )
                
                if first_result['success']:
                    # 生成したファイルをリストに追加
                    voice_files.append(first_result['file_path'])
                    
                    # 再生する場合のみキューに追加
                    if play_audio:
                        # 順序付きキューにインデックス0でファイルを追加
                        self.ordered_queue.add_file(0, first_result['file_path'])
                    
                    # コールバック関数を呼び出す
                    if callback:
                        is_last = len(texts) == 1  # 最後のファイルかどうか
                        callback(first_result['file_path'], 0, is_last)
                    
                    print(f"最初の文の音声を生成しました: {first_result['file_path']}")
                else:
                    print(f"最初の文の音声生成に失敗しました: {first_result['message']}")
            
            # 残りの文を並行処理
            if len(texts) > 1:
                remaining_texts = texts[1:]
                
                # 並行処理用のExecutorを作成（最大2個までに制限）
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # 各文の音声生成をExecutorに投入
                    future_to_text = {}
                    for i, text_part in enumerate(remaining_texts, 1):  # インデックスを1から開始
                        unique_id = str(uuid.uuid4())[:8]
                        filename = f"temp_voice_{i}_{unique_id}"
                        rvc_output_filename = f"output_voice_{i}_{unique_id}"
                        
                        future = executor.submit(
                            self.generate_voice_part,
                            text_part=text_part,
                            speaker_id=speaker_id,
                            filename=filename,
                            rvc_output_filename=rvc_output_filename
                        )
                        future_to_text[future] = (i, text_part)
                    
                    # 完了した順に結果を処理
                    completed_count = 0
                    for future in concurrent.futures.as_completed(future_to_text):
                        i, text_part = future_to_text[future]
                        completed_count += 1
                        is_last = completed_count == len(remaining_texts)  # 最後のファイルかどうか
                        
                        try:
                            result = future.result()
                            if result['success']:
                                # 生成したファイルをリストに追加
                                voice_files.append(result['file_path'])
                                
                                # 再生する場合のみキューに追加
                                if play_audio:
                                    # 順序付きキューにインデックスiでファイルを追加
                                    self.ordered_queue.add_file(i, result['file_path'])
                                
                                # コールバック関数を呼び出す
                                if callback:
                                    callback(result['file_path'], i, is_last)
                                
                                print(f"文 {i+1}/{len(texts)} の音声を生成しました: {result['file_path']}")
                            else:
                                print(f"文 {i+1}/{len(texts)} の音声生成に失敗しました: {result['message']}")
                        except Exception as e:
                            print(f"文 {i+1}/{len(texts)} の処理中にエラーが発生しました: {str(e)}")
                            print(traceback.format_exc())
            
            # 処理終了時間を記録
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds() * 1000
            print(f"音声生成全体の処理時間: {total_processing_time:.2f}ms")
            
            return voice_files
            
        except Exception as e:
            # 処理終了時間を記録（エラー時）
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds() * 1000
            print(f"音声生成処理時間（エラー）: {total_processing_time:.2f}ms")
            
            print(f"エラー: 音声生成エラー: {str(e)}")
            print(traceback.format_exc())
            
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        テキストから絵文字や特殊文字を削除する
        
        Args:
            text (str): 元のテキスト
            
        Returns:
            str: クリーニングされたテキスト
        """
        import re
        
        # 元のテキストを保存
        original_text = text
        
        # 1) 絵文字（Unicode の代表的ブロック＋記号）だけを列挙
        emoji_pattern = re.compile(
            "["                                     # 開始
            "\U0001F600-\U0001F64F"                 # 1) 顔文字
            "\U0001F300-\U0001F5FF"                 # 2) 記号＆絵文字
            "\U0001F680-\U0001F6FF"                 # 3) 交通・地図
            "\U0001F700-\U0001F77F"                 # 4) 錬金術
            "\U0001F780-\U0001F7FF"                 # 5) 幾何学拡張
            "\U0001F800-\U0001F8FF"                 # 6) 補助矢印など
            "\U0001F900-\U0001F9FF"                 # 7) 補助絵文字
            "\U0001FA00-\U0001FA6F"                 # 8) 囲みキーキャップなど
            "\U0001FA70-\U0001FAFF"                 # 9) 拡張絵文字A
            "\u2600-\u26FF"                         # 10) Misc Symbols
            "\u2700-\u27BF"                         # 11) Dingbats
            "]+" ,
            flags=re.UNICODE
        )

        # 変種セレクタ・ZWJ などもあらかじめ個別に
        vs_zwj_pattern = re.compile(r"[\u200D\uFE0E\uFE0F]")

        # ---- 置換実行 ----
        cleaned_text = emoji_pattern.sub("", text)
        cleaned_text = vs_zwj_pattern.sub("", cleaned_text)

        # 制御文字を除去
        cleaned_text = re.sub(r"[\x00-\x1F\x7F]", "", cleaned_text)

        # 任意で消したい装飾記号だけ個別指定
        cleaned_text = re.sub(r"[♪♡♥❤★☆◆◇■□●○]", "", cleaned_text)

        # 余分な空白整理
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        # 変更があった場合はログに出力
        if cleaned_text != original_text:
            print(f"テキストクリーニング: 特殊文字を削除しました")
            # 長いテキストの場合は省略表示
            if len(original_text) > 100:
                print(f"  変更前: {original_text[:50]}...（省略）...{original_text[-50:]}")
            else:
                print(f"  変更前: {original_text}")
                
            if len(cleaned_text) > 100:
                print(f"  変更後: {cleaned_text[:50]}...（省略）...{cleaned_text[-50:]}")
            else:
                print(f"  変更後: {cleaned_text}")
        
        return cleaned_text
    
    def generate_with_callback(self, text: str, speaker_id: int = 10, callback=None, play_audio: bool = True) -> None:
        """
        テキストから音声を生成し、ファイルごとにコールバック関数を呼び出す
        
        Args:
            text (str): 読み上げるテキスト
            speaker_id (int): 話者ID
            callback (callable): ファイルが生成されるたびに呼び出されるコールバック関数
                                 callback(file_path, index, is_last) の形式で呼び出される
            play_audio (bool): 音声を再生するかどうか
        """
        # テキストをクリーニング
        cleaned_text = self._clean_text(text)
        
        # クリーニングされたテキストで音声生成
        self._generate_voice_files(cleaned_text, speaker_id, play_audio, callback)
    
    def generate_and_play(self, text: str, speaker_id: int = 10, play_audio: bool = True) -> List[str]:
        """
        テキストから音声を生成し、必要に応じて再生する
        並行処理を使用して複数の文を同時に処理する
        順序付きキュー管理クラスを使用して再生順序を保証する
        
        Args:
            text (str): 読み上げるテキスト
            speaker_id (int): 話者ID
            play_audio (bool): 音声を再生するかどうか（Falseの場合は生成のみ）
        
        Returns:
            list: 生成された音声ファイルパスのリスト
        """
        # テキストをクリーニング
        cleaned_text = self._clean_text(text)
        
        # クリーニングされたテキストで音声生成
        return self._generate_voice_files(cleaned_text, speaker_id, play_audio)
    
    def generate_voice_part(self, text_part: str, speaker_id: int = 10, filename: str = "temp_voice", rvc_output_filename: str = "output_voice") -> Dict[str, Any]:
        """
        テキストの一部から音声を生成する
        
        Args:
            text_part (str): 読み上げるテキスト
            speaker_id (int): 話者ID
            filename (str): 出力ファイル名（拡張子なし）
            rvc_output_filename (str): RVC変換後の出力ファイル名（拡張子なし）
        
        Returns:
            dict: 処理結果を含む辞書
                - success (bool): 処理が成功したかどうか
                - message (str): 処理結果のメッセージ
                - file_path (str): 生成された音声ファイルのパス（成功時のみ）
        """
        # 処理開始時間を記録
        start_time = datetime.now()
        
        try:
            # 出力ファイルパスの設定
            wav_path = self.output_dir / f"{filename}.wav"
            
            # MeCabを使ってテキスト全体の読み仮名を取得
            print(f"MeCabを使って読み仮名を取得します: {text_part}")
            # 読み仮名取得の時間計測
            mecab_start_time = datetime.now()
            kana_all = self.get_yomigana_with_mecab(text_part)
            mecab_end_time = datetime.now()
            mecab_processing_time = (mecab_end_time - mecab_start_time).total_seconds() * 1000
            print(f"読み仮名取得処理時間: {mecab_processing_time:.2f}ms")
            
            # 音声合成処理の時間計測（全体）
            synthesis_start_time = datetime.now()
            
            # ステップ1: かなテキストからアクセント句を取得
            accent_start_time = datetime.now()
            query_params = {"text": kana_all, "speaker": speaker_id}
            response = requests.post(
                f"http://{self.hostname}:50021/accent_phrases", 
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
                f"http://{self.hostname}:50021/audio_query", 
                params=query_params
            )
            response.raise_for_status()
            audio_query = response.json()
            query_end_time = datetime.now()
            query_processing_time = (query_end_time - query_start_time).total_seconds() * 1000
            
            # ステップ3: 音声合成クエリのアクセント句を読みがなベースのものに置き換え
            audio_query["accent_phrases"] = accent_phrases
            
            # ステップ3.5: 各文の前後に5モーラ分（約0.5秒）の無音を追加
            # audio_query["prePhonemeLength"] = 0.5  # 音声の前に5モーラ分（約0.5秒）の無音
            audio_query["postPhonemeLength"] = 0.7  # 音声の後に5モーラ分（約0.7秒）の無音
            # print(f"文の後に5モーラ分（約0.5秒）の無音を追加します")
            
            # ステップ4: 音声を合成
            synth_start_time = datetime.now()
            headers = {"Accept": "audio/wav", "Content-Type": "application/json"}
            query_params = {"speaker": speaker_id}
            response = requests.post(
                f"http://{self.hostname}:50021/synthesis", 
                headers=headers,
                params=query_params, 
                data=json.dumps(audio_query)
            )
            response.raise_for_status()
            synth_end_time = datetime.now()
            synth_processing_time = (synth_end_time - synth_start_time).total_seconds() * 1000
            
            # 音声データを保存
            with open(wav_path, 'wb') as f:
                f.write(response.content)
            
            # 音声合成処理の時間計測（全体）
            synthesis_end_time = datetime.now()
            synthesis_total_time = (synthesis_end_time - synthesis_start_time).total_seconds() * 1000
            print(f"音声合成処理全体の時間: {synthesis_total_time:.2f}ms")
            
            # 処理終了時間を記録
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds() * 1000
            print(f"音声生成全体の処理時間: {total_processing_time:.2f}ms")
            
            # VOICEVOXで生成した音声をRVCに投げる
            print(f"VOICEVOXで生成した音声をRVCに投げます: {wav_path}")
            
            # RVC出力ファイルパスの設定
            rvc_output_path = self.output_dir / f"{rvc_output_filename}.wav"
            
            # RVC変換を実行
            rvc_result = self.convert_with_rvc(
                input_file=str(wav_path),
                output_file=str(rvc_output_path)
            )
            
            # RVC変換が成功した場合は、RVC変換後のファイルパスを返す
            if rvc_result['success']:
                # 元のVOICEVOXファイルを削除
                try:
                    if os.path.exists(str(wav_path)):
                        os.remove(str(wav_path))
                        print(f"元の音声ファイルを削除しました: {wav_path}")
                except Exception as e:
                    print(f"元のファイル削除エラー: {str(e)}")
                    
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
            print(traceback.format_exc())
            
            return {
                'success': False,
                'message': f"音声合成エラー: {str(e)}",
                'file_path': None
            }
    
    def convert_with_rvc(self, input_file: str, output_file: Optional[str] = None, target_sample_rate: int = 48000) -> Dict[str, Any]:
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
            rvc_url = f"http://{self.rvc_hostname}:{self.rvc_port}/api/voice-changer/convert_chunk"
            
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
    
    def get_yomigana_with_mecab(self, text: str) -> str:
        """
        MeCab + UniDicを使ってテキストから読み仮名を取得する関数
        
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
    
    def stop(self) -> None:
        """再生停止"""
        self.player_manager.stop()
    
    def set_on_complete_callback(self, callback: Callable[[], None]) -> None:
        """
        全ての再生が完了した時のコールバックを設定
        
        Args:
            callback (callable): コールバック関数
        """
        self.player_manager.set_on_complete_callback(callback)
