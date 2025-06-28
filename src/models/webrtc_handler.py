import logging
import threading
import asyncio
import queue
import json
from fractions import Fraction
import pyaudio
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

# ロギングの設定
logging.basicConfig(level=logging.INFO)

# WebRTC関連の設定
# PyAudioの設定（デフォルト値）
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # デバイスによって上書きされる
RATE = 48000  # デバイスによって上書きされる

# WebRTCピア接続を保持する辞書
pcs = set()

class AudioStreamTrack(MediaStreamTrack):
    """音声キャプチャクラス"""
    kind = "audio"

    def __init__(self, device_name, input_channels=2):
        super().__init__()
        self.device_name = device_name
        self.input_channels = input_channels
        self.p = pyaudio.PyAudio()
        self.device_index = self._get_device_index()
        self.running = True
        self.buffer = asyncio.Queue()
        self.data_queue = queue.Queue()  # スレッドセーフなキュー
        self.thread = threading.Thread(target=self._capture_audio)
        self.thread.daemon = True
        self.thread.start()
        
        # 別スレッドでキューからデータを取り出し、非同期キューに追加するタスクを開始
        self._task = None
        
        # サンプルレートとチャンネル数を設定
        self._timestamp = 0
        self._sample_rate = RATE
        
        # パケットロス対策用の変数
        self.last_valid_frame_data = None  # 前回の有効なフレームデータ
        self.consecutive_errors = 0  # 連続エラー回数
        self.max_consecutive_errors = 5  # 最大連続エラー回数
        
        # デバイス情報をログに出力
        self._log_device_info()

    def _log_device_info(self):
        """デバイス情報をログに出力"""
        try:
            if self.device_index is not None:
                device_info = self.p.get_device_info_by_index(self.device_index)
                logging.info(f"使用するデバイス情報:")
                logging.info(f"  デバイス名: {device_info['name']}")
            else:
                logging.warning("デバイスインデックスがNoneです。デフォルトデバイスを使用します。")
        except Exception as e:
            logging.error(f"デバイス情報取得エラー: {e}")

    def _get_device_index(self):
        """デバイス名とチャンネル数から検索"""
        # デバイス名とチャンネル数から検索
        matching_devices = []
        
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            
            # デバイス名とチャンネル数でフィルタリング
            name_match = self.device_name in device_info['name'] if self.device_name else True
            channels_match = True
            
            # 入力チャンネル数でフィルタリング（指定されている場合）
            channels_match = device_info['maxInputChannels'] == self.input_channels
            
            if name_match and channels_match:
                matching_devices.append((i, device_info))
        
        # マッチするデバイスが見つかった場合
        if matching_devices:
            # 最初に見つかったデバイスを使用
            device_index, device_info = matching_devices[0]
            logging.info(f"  デバイス名: {device_info['name']}")
            return device_index
        
        # マッチするデバイスが見つからなかった場合
        if self.device_name:
            logging.warning(f"デバイス '{self.device_name}' とチャンネル数 {self.input_channels} に一致するデバイスが見つかりません。デフォルトデバイスを使用します。")
        else:
            logging.warning(f"チャンネル数 {self.input_channels} に一致するデバイスが見つかりません。デフォルトデバイスを使用します。")
        
        return None

    async def _process_queue(self):
        """スレッドセーフなキューからデータを取り出し、非同期キューに追加するタスク"""
        try:
            while self.running:
                try:
                    # キューが空でない場合、データを取り出す（タイムアウト付き）
                    data = self.data_queue.get(block=False)
                    # 非同期キューに追加
                    await self.buffer.put(data)
                except queue.Empty:
                    # キューが空の場合は少し待つ
                    await asyncio.sleep(0.01)
        except Exception as e:
            logging.error(f"キュー処理エラー: {e}")

    def _capture_audio(self):
        """音声データをキャプチャしてキューに追加するスレッド"""
        try:
            # デバイス情報を取得
            if self.device_index is not None:
                device_info = self.p.get_device_info_by_index(self.device_index)
                
                # デバイスのサポートするチャンネル数とサンプルレートを取得
                device_channels = int(device_info['maxInputChannels'])
                device_rate = int(device_info['defaultSampleRate'])
                
                # グローバル変数を更新
                global CHANNELS, RATE
                if device_channels > 0:
                    CHANNELS = device_channels
                if device_rate > 0:
                    RATE = device_rate
                
                logging.info(f"音声キャプチャを開始します:")
            
            # 音声ストリームを開く
            try:
                # 入力デバイスとして開く
                stream = self.p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=CHUNK
                )
                is_input_device = True
            except Exception as e:
                logging.warning(f"入力デバイスとして開けませんでした: {e}")
                return
            
            logging.info("音声ストリームを開きました")
            
            # 入力デバイスの場合
            data_count = 0
            while self.running:
                # 音声データを読み取る
                data = stream.read(CHUNK, exception_on_overflow=False)
                data_count += 1
                
                # 通常の音声データをキューに追加
                self.data_queue.put(data)
                
        except Exception as e:
            logging.error(f"音声キャプチャエラー: {e}")

    async def recv(self):
        """音声フレームを受信する"""
        try:
            # _taskがNoneの場合は初期化
            if self._task is None:
                self._task = asyncio.create_task(self._process_queue())
            
            # キューからデータを取得
            frame_data = await self.buffer.get()
            
            # サンプルレートを更新
            self._sample_rate = RATE
            
            # AudioFrameオブジェクトを作成
            frame = av.AudioFrame(format='s16', layout='mono' if CHANNELS == 1 else 'stereo', samples=CHUNK)
            
            # サンプルデータをコピー
            frame.planes[0].update(frame_data)
            
            # サンプルレートを設定
            frame.sample_rate = self._sample_rate
            
            # タイムスタンプを設定
            frame.pts = self._timestamp
            self._timestamp += frame.samples
            frame.time_base = Fraction(1, self._sample_rate)
            
            return frame
            
        except Exception as e:
            logging.error(f"フレーム受信エラー: {e}")
            # エラーが発生した場合は空のフレームを返す
            empty_frame = av.AudioFrame(format='s16', layout='mono' if CHANNELS == 1 else 'stereo', samples=CHUNK)
            
            # 空のデータを作成
            empty_data = bytes(CHUNK * CHANNELS * 2)  # 16ビット（2バイト）* チャンネル数 * サンプル数
            
            # サンプルデータをコピー
            empty_frame.planes[0].update(empty_data)
            empty_frame.sample_rate = self._sample_rate
            empty_frame.pts = self._timestamp
            self._timestamp += empty_frame.samples
            empty_frame.time_base = Fraction(1, self._sample_rate)
            return empty_frame

def initialize_webrtc(webrtc_settings):
    """WebRTC設定を初期化する"""
    try:
        # 設定から読み込む
        device_name = webrtc_settings.get('input_device')
        input_channels = webrtc_settings.get('input_channels')
        
        logging.info(f"WebRTC設定を読み込みました: デバイス名={device_name}, チャンネル数={input_channels}")
        
        return {
            'device_name': device_name,
            'input_channels': input_channels
        }
    except Exception as e:
        logging.error(f"WebRTC設定の初期化エラー: {e}")
        # デフォルト値を返す
        return {
            'device_name': "CABLE-A Output (VB-Audio Cable A)",
            'input_channels': 2
        }

def process_offer(request_data, webrtc_settings):
    """WebRTC接続のオファーを処理する"""
    # 結果を格納するための共有変数
    result_container = {"result": None}
    result_ready = threading.Event()
    
    # 非同期処理を別スレッドで実行
    def process_offer_thread():
        # 新しいスレッド内でイベントループを作成
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def process_offer_async():
            # コピーしたリクエストデータを使用
            params = request_data
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            
            pc = RTCPeerConnection()
            pcs.add(pc)
            
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                state = pc.connectionState
                logging.info(f"接続状態: {state}")
            
            # 音声トラックを追加
            audio_track = AudioStreamTrack(webrtc_settings['device_name'], webrtc_settings['input_channels'])
            pc.addTrack(audio_track)
            
            # FECを有効化するためにSDPを修正
            def modify_sdp_for_fec(sdp):
                lines = sdp.split('\n')
                opus_payload_type = None
                
                # まずOpusのペイロードタイプを見つける
                for i, line in enumerate(lines):
                    if 'opus/48000' in line:
                        match = line.split()
                        if len(match) >= 2:
                            opus_payload_type = match[0].split(':')[1]
                            logging.info(f"Opusペイロードタイプ: {opus_payload_type}")
                            break
                
                # 次にそのペイロードタイプのfmtp行を見つけて修正
                if opus_payload_type:
                    fmtp_prefix = f"a=fmtp:{opus_payload_type} "
                    fmtp_line_index = -1
                    
                    for i, line in enumerate(lines):
                        if line.startswith(fmtp_prefix):
                            fmtp_line_index = i
                            break
                    
                    if fmtp_line_index != -1:
                        # 既存のfmtp行を修正
                        if 'useinbandfec=1' not in lines[fmtp_line_index]:
                            lines[fmtp_line_index] += ';useinbandfec=1;usedtx=1;stereo=0;maxplaybackrate=48000;maxaveragebitrate=30000'
                            logging.info(f"既存のfmtp行を修正しました: {lines[fmtp_line_index]}")
                    else:
                        # fmtp行が存在しない場合は新しく追加
                        for i, line in enumerate(lines):
                            if f"a=rtpmap:{opus_payload_type} opus/48000" in line:
                                lines.insert(i + 1, f"{fmtp_prefix}useinbandfec=1;usedtx=1;stereo=0;maxplaybackrate=48000;maxaveragebitrate=30000")
                                logging.info(f"新しいfmtp行を追加しました: {lines[i + 1]}")
                                break
                
                return '\n'.join(lines)
            
            # オファーのSDPを修正
            offer.sdp = modify_sdp_for_fec(offer.sdp)
            logging.info(f"修正後のオファーSDP: {offer.sdp}")
            
            await pc.setRemoteDescription(offer)
            
            answer = await pc.createAnswer()
            
            # アンサーのSDPも修正
            answer.sdp = modify_sdp_for_fec(answer.sdp)
            logging.info(f"修正後のアンサーSDP: {answer.sdp}")
            
            await pc.setLocalDescription(answer)
            
            # 結果を設定
            result_container["result"] = {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
            
            # 結果が準備できたことを通知
            result_ready.set()
            
            # 接続を維持するためにループを実行し続ける
            while True:
                await asyncio.sleep(1)
        
        try:
            # 非同期処理を実行
            loop.run_until_complete(process_offer_async())
        except Exception as e:
            logging.error(f"WebRTC処理エラー: {e}")
            result_container["result"] = {"error": str(e)}
            result_ready.set()
    
    # 別スレッドで処理を実行
    thread = threading.Thread(target=process_offer_thread, daemon=True)
    thread.start()
    
    # 結果が準備できるまで待機（最大10秒）
    if not result_ready.wait(timeout=10):
        return {"error": "WebRTC処理がタイムアウトしました"}
    
    # 結果を返す
    return result_container["result"]
