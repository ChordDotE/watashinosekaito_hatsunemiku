/*
 * Watashinosekaito Hatsune Miku
 * Copyright (c) 2025-2026 ChordDotE. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-ChordDotE-NonCommercial-2.0
 *
 * NON-COMMERCIAL USE ONLY. NO BUSINESS USE. NO ADAPTATION.
 *
 * Business use (any use by or for a company, organization, or sole
 * proprietor, WHETHER OR NOT ANY PAYMENT IS INVOLVED) and publishing or
 * distributing modified versions are PROHIBITED IN ALL CASES; no permission
 * is granted for them. Forks may be used only to propose improvements to the
 * Author (e.g. pull requests). Commercial use, redistribution, and Adaptation
 * are PROHIBITED without prior written permission. "Adaptation" includes
 * manual reimplementation, automated translation, and generation using large
 * language models or other machine learning, of software having equivalent or
 * similar functionality. Permission is granted individually, in recorded form.
 * Contact: https://x.com/ChordDotE
 *
 * 業務利用（対価の有無を問わず、法人・団体・個人事業主による利用と事業のための利用）と、
 * 改変版の公開・配布は、許可の余地なく禁止します。フォークは作者への改善の提案
 * （プルリクエスト等）のためだけに使えます。商用利用・再配布・翻案は、作者の許可が
 * ない限り禁止します。翻案には、再実装・自動変換・大規模言語モデル等を用いた生成に
 * よる、同等または類似の機能を持つソフトウェアの作成を含みます。
 * 許可が必要な場合は、記録の残る形でご連絡ください。
 *
 * See LICENSE for full terms. 詳細は LICENSE を参照してください。
 */

/**
 * events.js - モジュール間イベント契約（ブラウザ側の単一ソース）
 *
 * サーバー側の「サービス間は HTTP 経由のみ」に対応するブラウザ側の規約:
 * モジュール/コンポーネント間の通信は CustomEvent（と appState）経由のみとし、
 * イベント名は必ずこのファイルの定数を使う（マジック文字列禁止）。
 *
 * 各定数のコメントが payload（e.detail）の契約。発火元・購読先の一覧は
 * docs/components/browser.md の「モジュール間カスタムイベント 一覧」を参照。
 */

// ---------------------------------------------------------------------------
// CustomEvent 名（document.dispatchEvent / addEventListener で使用）
// ---------------------------------------------------------------------------
export const EVENTS = {
    // --- 音声出力（voicePcmPlayer が発火） ---
    /** detail: { fileName? } 最初の音声再生開始（ターン1回） */
    VOICE_START: 'voice:start',
    /** detail: なし。全音声再生完了 / 強制停止 */
    VOICE_END: 'voice:end',

    // --- 歌唱（songPlayer.js が発火。歌っている区間＝マイクのミュート制御が購読する） ---
    /** detail: { title, mode: 'sing'|'play'|'preview' } 曲を歌い始めた／流し始めた */
    SONG_START: 'song:start',
    /** detail: なし。歌い終わった／中断された */
    SONG_END: 'song:end',
    /** detail: なし。歌をやめてほしい（マイクボタン＝話しかける合図。songPlayer が購読） */
    SONG_INTERRUPT: 'song:interrupt',

    // --- ターン進行（miku-chat が発火） ---
    /** detail: なし。フォーム送信（新しいターン開始） */
    THINKING_START: 'thinking:start',
    /** detail: なし。送信エラー / 音声なし応答の確定 */
    THINKING_END: 'thinking:end',
    /** detail: { audible, turnBelief, userSpeaking, userTurnOpen, prev: {同4値} }
     *  会話ターン状態の変化（turnState.js が発火。basic_design §4.6 の3パラメータ
     *  ＋ userTurnOpen）。送信ボタン表示・マイクミュート・音声再生の保留は
     *  これを購読して導出する */
    TURN_STATE_CHANGE: 'turnstate:change',
    /** detail: { turnComplete: boolean } STT の確定結果が届いた（miku-chat が発火）。
     *  turnState が「マスターの番がまだ続いているか（userTurnOpen）」の判定に使う。
     *  turnComplete は Smart Turn の終話判定で、無効時は undefined */
    STT_UTTERANCE_END: 'stt:utteranceEnd',

    // --- テキスト入力欄（miku-chat が発火） ---
    /** detail: なし */
    INPUT_FOCUS: 'input:focus',
    /** detail: なし */
    INPUT_BLUR: 'input:blur',

    // --- マイク（microphone.js が発火） ---
    /** detail: { state: 'OFF'|'LISTENING'|'RECORDING'|'PROCESSING' } */
    MIC_STATE_CHANGE: 'mic:stateChange',
    /** detail: なし。VADが発話開始を検知 */
    MIC_SPEECH_START: 'mic:speechStart',
    /** detail: { audioData: Float32Array, sampleRate: number } 発話終了 */
    MIC_SPEECH_END: 'mic:speechEnd',
    /** detail: { pcmData: Float32Array, sampleRate: number } 100ms チャンク */
    MIC_AUDIO_CHUNK: 'mic:audioChunk',
    /** detail: { rms: number } 音量（約60fps） */
    MIC_LEVEL: 'mic:level',

    // --- STT（socket.js が発火） ---
    /** detail: { text: string } バッチSTT結果 */
    STT_RESULT: 'stt:result',
    /** detail: { error: string } */
    STT_ERROR: 'stt:error',
    /** detail: なし。STTモデルロード中 */
    STT_MODEL_LOADING: 'stt:modelLoading',
    /** detail: なし。STTモデル準備完了 */
    STT_MODEL_READY: 'stt:modelReady',
    /** detail: なし。**言葉として認識された**（interim が届いた）＝マスターが本当に喋っている証拠。
     *  音量（mic:speechStart）では雑音と区別できないため、再生保留の「諦めるまで」の起点に使う */
    STT_SPEECH_EVIDENCE: 'stt:speechEvidence',

    // --- サーバー通知（socket.js が Socket.IO イベントを変換して発火） ---
    /** detail: { response, inactivity_timeout, motion, session_id } 無応答リマインダー */
    SERVER_INACTIVITY_REMINDER: 'server:inactivityReminder',
    /** detail: { motion, inactivity_timeout, has_audio, response } 最終応答確定 */
    SERVER_RESPONSE_READY: 'server:responseReady',
    /** detail: { message } generate 処理エラー */
    SERVER_GENERATE_ERROR: 'server:generateError',
    /** detail: なし。Socket.IO 切断 */
    SERVER_DISCONNECTED: 'server:disconnected',
    /** detail: なし。Socket.IO 再接続完了（audio_stream_start 再送用） */
    SERVER_SOCKET_RECONNECTED: 'server:socketReconnected',
    /** detail: { song_id, title, url, lipsync_url, mode } 曲の再生予約（すぐには鳴らない） */
    SERVER_SONG_PLAY: 'server:songPlay',
    /** detail: { sample_rate, channels, chapter } Fish Audio PCM チャプター開始 */
    SERVER_VOICE_PCM_START: 'server:voicePcmStart',
    /** detail: { chapter, data: base64 } PCM チャンク */
    SERVER_VOICE_PCM_CHUNK: 'server:voicePcmChunk',
    /** detail: { chapter, no_audio? } PCM チャプター終了 */
    SERVER_VOICE_PCM_END: 'server:voicePcmEnd',

    // --- アプリ横断（複数モジュールが発火） ---
    /** detail: { motionCode: '01'〜'19' } Live2D モーション再生要求 */
    APP_MOTION_REQUEST: 'app:motionRequest',
    /** detail: { seconds: number } 無応答タイムアウト値の更新 */
    APP_INACTIVITY_TIMEOUT: 'app:inactivityTimeout',
    /** detail: なし。サーバー疎通確認の開始（main.js が発火） */
    APP_CONNECTION_CHECKING: 'app:connectionChecking',
    /** detail: { success: boolean } サーバー疎通確認の結果 */
    APP_CONNECTION_RESULT: 'app:connectionResult',

    // --- 字幕モード（miku-chat の字幕ボタンが発火） ---
    /** detail: { enabled: boolean } ON中は音声再生・リップシンクを停止 */
    CAPTION_SET_ENABLED: 'caption:setEnabled',

    // --- 無応答リマインダー（miku-chat のリマインダーボタンが発火） ---
    /** detail: { enabled: boolean } OFF中は無応答リマインダー要求を送信しない */
    APP_REMINDER_ENABLED: 'app:reminderEnabled',

    // --- 画面ロック（miku-chat の表示メニューが発火、miku-lock が購読） ---
    /** detail: なし。ロックモード開始（miku-lock が全画面オーバーレイを表示） */
    LOCK_ENTER: 'lock:enter',
    /** detail: なし。ロック解除（miku-lock が枠を一周なぞられた時に発火） */
    LOCK_EXIT: 'lock:exit',

    // --- カメラ（CameraManager が発火） ---
    /** detail: { stream: MediaStream|null } カメラの映像が変わった（ON・切替・OFF）。
     *  gazeTracker が同じ映像を共有して顔を追う（カメラを二重に掴まないため） */
    CAMERA_STREAM_CHANGE: 'camera:streamChange',

    // --- 視線追従（表示メニューの視線ボタン／gazeTracker が発火） ---
    /** detail: { enabled: boolean } 視線追従の ON/OFF（MenuController が発火、gazeTracker が購読） */
    GAZE_SET_ENABLED: 'gaze:setEnabled',
    /** detail: { x: number, y: number } ミクが向く方向（-1〜1。x は画面の右・y は上が正。
     *  0,0 = 正面）。gazeTracker が発火し、miku-avatar-live2d が Live2D へ中継する */
    GAZE_TARGET: 'gaze:target',
};

// ---------------------------------------------------------------------------
// Socket.IO イベント名（Gateway との契約。payload は docs/components/gateway.md）
// connect / disconnect / error / reconnect は Socket.IO 組み込みのため定数化しない
// ---------------------------------------------------------------------------
export const SOCKET_EVENTS = {
    // 送信（Browser → Gateway）
    SESSION_ACTIVATE: 'session_activate',
    INACTIVITY_REMINDER_REQUEST: 'inactivity_reminder_request',
    AUDIO_STREAM_START: 'audio_stream_start',
    AUDIO_STREAM_CHUNK: 'audio_stream_chunk',
    AUDIO_STREAM_STOP: 'audio_stream_stop',
    TEXT_INTERIM: 'text_interim',
    CLIENT_LOG: 'client_log',
    USER_SPEAKING: 'user_speaking',

    // 受信（Gateway → Browser）
    SESSION_ACTIVATED: 'session_activated',
    INACTIVITY_REMINDER: 'inactivity_reminder',
    RESPONSE_READY: 'response_ready',
    GENERATE_ERROR: 'generate_error',
    STT_RESULT: 'stt_result',
    STT_ERROR: 'stt_error',
    STT_MODEL_LOADING: 'stt_model_loading',
    STT_MODEL_READY: 'stt_model_ready',
    STT_INTERIM: 'stt_interim',
    STT_FINAL: 'stt_final',
    REACTION_MOTION: 'reaction_motion',
    SONG_PLAY: 'song_play',
    VOICE_PCM_START: 'voice_pcm_start',
    VOICE_PCM_CHUNK: 'voice_pcm_chunk',
    VOICE_PCM_END: 'voice_pcm_end',
};
