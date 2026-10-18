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
 * remoteLog.js - ブラウザ挙動ログのサーバー転送
 *
 * スマホ実機では DevTools コンソールが見られず、タイミング系の偶発バグ
 * （例: 2026-08-04 のマイク未ミュート）が起きても記録が残らない。
 * 「偶発バグは今後も起きる」前提で、検知→診断→修正の手順の**検知**部分として、
 * タグ付きコンソールログを常時サーバー（Gateway）へ転送し日別ファイルに残す。
 * （判断の記録: docs/dev/decisions.md 2026-08-04）
 *
 * 方式:
 * - console.log/warn/error をフックし、**タグ付きの行だけ**リングバッファに積む
 *   （送信量の抑制。warn/error はタグなしでも全部拾う）
 * - 3秒ごとに Socket.IO でまとめて送信（client_log）。未接続の間はバッファ保持
 * - ページ離脱・バックグラウンド化時は sendBeacon（POST /api/client_log）で残りを送る
 * - window.onerror / unhandledrejection も捕捉
 *
 * サーバー側の書き出し先: miku/runtime/browser_YYYYMMDD.log
 */

import { SOCKET_EVENTS } from './events.js';
import { getSocket } from './socket.js';
import { getSessionId } from './session.js';

// 転送対象のタグ（会話ターン・音声まわりの挙動追跡に必要な行）
const TAG_RE = /^\[(TurnState|Fold|AutoSend|UserSpeech|VoicePcmPlayer|VoiceQueue|Microphone|MicButton|MikuChat|IdleAutosend|Camera)\]/;

const FLUSH_INTERVAL_MS = 3000;
const MAX_BUFFER = 400;   // リングバッファ上限（超過は古い行から破棄してカウント）
const MAX_BATCH = 200;    // 1回の送信上限

let _buffer = [];
let _dropped = 0;
let _initialized = false;

function _nowStr() {
    const d = new Date();
    const pad = (n, w = 2) => String(n).padStart(w, '0');
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

function _push(level, text) {
    if (_buffer.length >= MAX_BUFFER) {
        _buffer.shift();
        _dropped++;
    }
    _buffer.push({ t: _nowStr(), level, text });
}

function _shouldCapture(level, text) {
    if (level !== 'log') return true;  // warn / error は全部残す
    return TAG_RE.test(text);
}

function _hookConsole() {
    for (const level of ['log', 'warn', 'error']) {
        const orig = console[level].bind(console);
        console[level] = (...args) => {
            orig(...args);
            try {
                const text = args
                    .map((a) => (typeof a === 'string' ? a : JSON.stringify(a)))
                    .join(' ');
                if (_shouldCapture(level, text)) _push(level, text);
            } catch (_) { /* ログ変換失敗は無視（本来のログは出力済み） */ }
        };
    }
    window.addEventListener('error', (e) =>
        _push('error', `window.onerror: ${e.message} @${e.filename}:${e.lineno}`));
    window.addEventListener('unhandledrejection', (e) =>
        _push('error', `unhandledrejection: ${e.reason}`));
}

function _flush() {
    if (_buffer.length === 0) return;
    const socket = getSocket();
    if (!socket || !socket.connected) return;  // 未接続なら次回（バッファは保持）
    if (_dropped > 0) {
        _push('warn', `[RemoteLog] バッファ超過で ${_dropped} 行を破棄`);
        _dropped = 0;
    }
    const lines = _buffer.splice(0, MAX_BATCH);
    try {
        socket.emit(SOCKET_EVENTS.CLIENT_LOG, {
            session: getSessionId().slice(0, 8),
            lines,
        });
    } catch (_) {
        _buffer = lines.concat(_buffer);  // 送れなかったら戻す
    }
}

function _flushBeacon() {
    if (_buffer.length === 0) return;
    const lines = _buffer.splice(0, MAX_BATCH);
    try {
        navigator.sendBeacon(
            '/api/client_log',
            new Blob(
                [JSON.stringify({ session: getSessionId().slice(0, 8), lines })],
                { type: 'application/json' },
            ),
        );
    } catch (_) { /* 離脱時のため失敗は諦める */ }
}

/** ログ転送を開始する（main.js の初期化から呼ぶ。多重呼び出しは無視） */
export function initRemoteLog() {
    if (_initialized) return;
    _initialized = true;
    _hookConsole();
    setInterval(_flush, FLUSH_INTERVAL_MS);
    // バックグラウンド化・離脱時は Socket.IO が止まるので sendBeacon で送り切る
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') _flushBeacon();
    });
    window.addEventListener('pagehide', _flushBeacon);
}
