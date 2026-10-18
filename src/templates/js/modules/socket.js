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
 * socket.js - WebSocket管理モジュール
 *
 * 責務: Socket.IO 接続管理とサーバーイベントの受信。
 * サーバーイベントを受け取ったら CustomEvent を発火するだけ。
 * 他モジュールへの直接呼び出しは行わない。
 *
 * 受信イベント (Socket.IO):
 *   inactivity_reminder    → server:inactivityReminder を発火
 *   stt_result             → stt:result               を発火
 *   stt_error              → stt:error                を発火
 *   stt_model_loading      → stt:modelLoading         を発火
 *   stt_model_ready        → stt:modelReady           を発火
 *   reaction_motion        → app:motionRequest        を発火（会話中リアクション）
 *   disconnect             → server:disconnected      を発火
 */

import { EVENTS, SOCKET_EVENTS } from './events.js';
import { getSessionId } from './session.js';

// Socket.IO インスタンス
let socket = null;

// スマホ等 console が見られない環境向けの画面内デバッグ表示。
// URL に ?debug=1 を付けたときだけ有効（例: https://…/?debug=1）
if (new URLSearchParams(location.search).get('debug') === '1') {
    window.__mikuDebugToast = (msg) => {
        const el = document.createElement('div');
        el.textContent = msg;
        el.style.cssText = `
            position: fixed; left: 8px; bottom: ${8 + document.querySelectorAll('.miku-debug-toast').length * 34}px;
            z-index: 9999; background: rgba(13,15,24,.88); color: #A5F8F1;
            font: 13px/1.4 monospace; padding: 6px 10px; border-radius: 8px;
            pointer-events: none;`;
        el.className = 'miku-debug-toast';
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 4000);
    };
}

/**
 * WebSocket 接続を初期化
 */
export function initSocket() {
    if (typeof io === 'undefined') {
        console.error('❌ Socket.IOが読み込まれていません');
        return;
    }

    socket = io();

    socket.on('connect',           handleConnect);
    socket.on('disconnect',        handleDisconnect);
    socket.on(SOCKET_EVENTS.SESSION_ACTIVATED, handleSessionActivated);
    socket.on('error',             handleError);
    socket.on(SOCKET_EVENTS.INACTIVITY_REMINDER, handleInactivityReminder);
    socket.on(SOCKET_EVENTS.RESPONSE_READY,    handleResponseReady);
    socket.on(SOCKET_EVENTS.GENERATE_ERROR,    handleGenerateError);
    socket.on(SOCKET_EVENTS.STT_RESULT,        handleSttResult);
    socket.on(SOCKET_EVENTS.STT_ERROR,         handleSttError);
    socket.on(SOCKET_EVENTS.STT_MODEL_LOADING, () => document.dispatchEvent(new CustomEvent(EVENTS.STT_MODEL_LOADING)));
    socket.on(SOCKET_EVENTS.STT_MODEL_READY,   () => document.dispatchEvent(new CustomEvent(EVENTS.STT_MODEL_READY)));
    socket.on(SOCKET_EVENTS.SONG_PLAY,         (data) => document.dispatchEvent(new CustomEvent(EVENTS.SERVER_SONG_PLAY, { detail: data })));
    // 会話中リアクション: 発話中の interim 判定で選ばれたモーションを即時再生
    socket.on(SOCKET_EVENTS.REACTION_MOTION, (data) => {
        if (data && data.motion && data.motion !== 'none') {
            console.log(`👂 リアクションモーション受信: ${data.motion}`);
            window.__mikuDebugToast?.(`👂 reaction ${data.motion}`);
            document.dispatchEvent(new CustomEvent(EVENTS.APP_MOTION_REQUEST, {
                detail: { motionCode: data.motion }
            }));
        }
    });

    // Fish Audio PCM ストリーミング（新フロー）
    socket.on(SOCKET_EVENTS.VOICE_PCM_START, handleVoicePcmStart);
    socket.on(SOCKET_EVENTS.VOICE_PCM_CHUNK, handleVoicePcmChunk);
    socket.on(SOCKET_EVENTS.VOICE_PCM_END,   handleVoicePcmEnd);

    // 再接続時に audio_stream_start を再送できるよう通知
    socket.io.on('reconnect', () => {
        console.log('🔄 WebSocket再接続完了 → server:socketReconnected を発火');
        document.dispatchEvent(new CustomEvent(EVENTS.SERVER_SOCKET_RECONNECTED));
    });

    console.log('WebSocket初期化完了');
}

function handleConnect() {
    console.log('✅ WebSocket接続が確立されました');
    const sessionId = getSessionId();
    if (sessionId) {
        socket.emit(SOCKET_EVENTS.SESSION_ACTIVATE, { sessionId });
        console.log(`📡 セッションアクティブ化を送信: ${sessionId}`);
    }
}

function handleDisconnect() {
    console.log('⚠️ WebSocket接続が切断されました');

    // 各モジュールへ切断を通知（直接呼び出しなし）
    document.dispatchEvent(new CustomEvent(EVENTS.SERVER_DISCONNECTED));

    setTimeout(() => {
        console.log('WebSocket再接続を試みています...');
        if (socket) socket.connect();
    }, 3000);
}

function handleSessionActivated(data) {
    console.log(`✅ セッションアクティブ化確認: ${data.sessionId}`);
}

function handleError(data) {
    console.error('❌ WebSocketエラー:', data.message);
}

/**
 * 無応答リマインダーの処理。
 * セッションIDが自分宛ての場合のみ server:inactivityReminder を発火。
 */
function handleInactivityReminder(data) {
    console.log(`🔔 無応答リマインダーを受信: ${data.response}`);

    const mySessionId = getSessionId();
    if (data.session_id && data.session_id !== mySessionId) {
        console.log(`⏭️ 他のクライアント宛ての無応答リマインダーをスキップ`);
        return;
    }

    console.log(`🔔 自分宛ての無応答リマインダーを処理`);
    document.dispatchEvent(new CustomEvent(EVENTS.SERVER_INACTIVITY_REMINDER, {
        detail: {
            response:           data.response,
            inactivity_timeout: data.inactivity_timeout,
            motion:             data.motion,
            session_id:         data.session_id,
        }
    }));

    if (data.motion && data.motion !== 'null') {
        document.dispatchEvent(new CustomEvent(EVENTS.APP_MOTION_REQUEST, {
            detail: { motionCode: data.motion }
        }));
    }
}

/**
 * 最終応答確定通知 → server:responseReady を発火
 */
function handleResponseReady(data) {
    const mySessionId = getSessionId();
    if (data.session_id && data.session_id !== mySessionId) return;

    document.dispatchEvent(new CustomEvent(EVENTS.SERVER_RESPONSE_READY, {
        detail: {
            motion:             data.motion,
            inactivity_timeout: data.inactivity_timeout,
            has_audio:          data.has_audio,
            response:           data.response,
        }
    }));
}

/**
 * generate エラー通知 → server:generateError を発火
 */
function handleGenerateError(data) {
    console.error('❌ generate エラー:', data.message);
    document.dispatchEvent(new CustomEvent(EVENTS.SERVER_GENERATE_ERROR, {
        detail: { message: data.message }
    }));
}

/**
 * STT 結果受信 → stt:result を発火
 */
function handleSttResult(data) {
    const mySessionId = getSessionId();
    if (data.targetSessionId && data.targetSessionId !== mySessionId) {
        console.log(`⏭️ 他のクライアント宛てのSTT結果をスキップ`);
        return;
    }

    console.log(`🎤 STT結果受信: "${data.text}"`);
    document.dispatchEvent(new CustomEvent(EVENTS.STT_RESULT, {
        detail: { text: data.text }
    }));
}

/**
 * STT エラー受信 → stt:error を発火
 */
function handleSttError(data) {
    console.error(`❌ STTエラー: ${data.error}`);
    document.dispatchEvent(new CustomEvent(EVENTS.STT_ERROR, {
        detail: { error: data.error }
    }));
}

// ---------------------------------------------------------------------------
// Fish Audio PCM ストリーミングハンドラ（新フロー）
// ---------------------------------------------------------------------------

/**
 * チャプター開始通知 → server:voicePcmStart を発火
 */
function handleVoicePcmStart(data) {
    const mySessionId = getSessionId();
    if (data.session_id && data.session_id !== mySessionId) return;

    console.log(`🎙️ PCMストリーム開始: チャプター${data.chapter}`);
    document.dispatchEvent(new CustomEvent(EVENTS.SERVER_VOICE_PCM_START, { detail: data }));
}

/**
 * PCM チャンク受信 → server:voicePcmChunk を発火
 */
function handleVoicePcmChunk(data) {
    document.dispatchEvent(new CustomEvent(EVENTS.SERVER_VOICE_PCM_CHUNK, { detail: data }));
}

/**
 * チャプター終了通知 → server:voicePcmEnd を発火
 */
function handleVoicePcmEnd(data) {
    console.log(`🔇 PCMストリーム終了: チャプター${data.chapter}`);
    document.dispatchEvent(new CustomEvent(EVENTS.SERVER_VOICE_PCM_END, { detail: data }));
}

/**
 * Socket インスタンスを取得
 * @returns {Object} Socket.IO インスタンス
 */
export function getSocket() {
    return socket;
}