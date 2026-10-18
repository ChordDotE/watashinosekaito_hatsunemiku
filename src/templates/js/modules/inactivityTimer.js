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
 * inactivityTimer.js - 無応答タイマー管理モジュール
 *
 * 責務: 音声再生完了後のタイマー管理と、タイムアウト時のリマインダー要求送信。
 *
 * **状態は turnState から導出する**（basic_design §4.6 / turnState.js）。
 * ミクの声が鳴り終わったかは `turnstate:change` の `audible` の立ち下がりで判断し、
 * `voice:end` を直接購読しない。かつては直接購読しており、**歌（singing）の存在を
 * 知り得ないまま「音が止まった」と判断**して、歌の再生中に無応答リマインダーを
 * 喋り出していた（2026-08-16）。状態の集約を迂回すると、集約側が持っている文脈が
 * 落ちる——という一般則の実例なので、ここは必ず集約経由にする。
 *
 * 購読イベント（document CustomEvent）:
 *   turnstate:change         → audible ○→✕ でタイマー開始（ターン完了フラグが立っているときのみ）
 *   thinking:start           → タイマー停止 + ターン完了フラグをリセット
 *   server:responseReady     → ターン完了フラグを立てる（最終ループのみサーバーが発火）。
 *                              字幕モード中は音声再生をスキップし voice:end が来ないため、
 *                              has_audio=true ならここで直接タイマーを開始する
 *   input:focus              → タイマー一時停止
 *   input:blur               → タイマー再開
 *   server:inactivityReminder → タイムアウト値の更新（socket.js 経由）
 *   server:disconnected       → 切断時にタイマー強制停止
 *   app:inactivityTimeout     → タイムアウト値の更新（miku-chat.js 経由）
 *   app:reminderEnabled       → リマインダーのON/OFF切替（miku-chat.js のリマインダーボタン経由。
 *                              既定ON・永続化なし。OFF中は requestInactivityReminder() が要求を送らない）
 *
 * ターン完了フラグについて:
 *   ツール呼び出しを挟むターンでは、Fish Audio パスがループ（チャプター）ごとに
 *   音の停止を起こす。ループ間（ツール実行中）にタイマーを開始すると
 *   ターンの途中で無応答リマインダーが飛んでしまう。
 *   そのため「最終ループに到達した」ことを示す server:responseReady を受けるまでは
 *   タイマーを開始しない。server:responseReady は非ツールの最終ループでのみ
 *   発火し、thinking:start → responseReady →
 *   音の停止 の順序が保証される。
 */

import { EVENTS, SOCKET_EVENTS } from './events.js';
import { getSocket } from './socket.js';
import { getSessionId } from './session.js';
import { appState } from './appState.js';

// タイマー設定
let timer = null;
let timeoutSeconds = 120;  // デフォルト120秒
let remainingTime = 0;     // 一時停止時の残り時間
let timerStartTime = 0;    // タイマー開始時刻
let isPaused = false;      // 一時停止中フラグ
let turnFinalReached = false;  // 最終ループ到達フラグ（voice:end でタイマー開始してよいか）
let reminderEnabled = true;    // 無応答リマインダーのON/OFF（既定ON。ブラウザ側トグル用）

/**
 * 無応答タイマーを初期化
 */
export function initInactivityTimer() {
    // 既存イベント
    // 音の鳴り終わりは **集約された状態から導出**する（voice:end を直接聞かない。上の説明参照）
    document.addEventListener(EVENTS.TURN_STATE_CHANGE, handleTurnStateChange);
    document.addEventListener(EVENTS.THINKING_START, handleThinkingStart);
    document.addEventListener(EVENTS.SERVER_RESPONSE_READY, handleResponseReady);
    document.addEventListener(EVENTS.INPUT_FOCUS,    handleInputFocus);
    document.addEventListener(EVENTS.INPUT_BLUR,     handleInputBlur);

    // socket.js が発火する新イベント
    document.addEventListener(EVENTS.SERVER_INACTIVITY_REMINDER, handleServerInactivityReminder);
    document.addEventListener(EVENTS.SERVER_DISCONNECTED,        handleServerDisconnected);

    // miku-chat.js が発火する新イベント（/generate レスポンスや /api/config でのタイムアウト更新）
    document.addEventListener(EVENTS.APP_INACTIVITY_TIMEOUT, handleAppInactivityTimeout);

    // miku-chat.js のリマインダーボタンが発火（ON/OFF切替）
    document.addEventListener(EVENTS.APP_REMINDER_ENABLED, handleAppReminderEnabled);

    // 字幕モードON切替: voice:end 待ちのターンが残っていたらここで閉じてタイマーを開始する
    // （音声モードで responseReady を通過した後に字幕ONへ切り替えると、音声が破棄されて
    //   voice:end が来なくなるため。タイムアウト値は responseReady 時に設定済み）
    document.addEventListener(EVENTS.CAPTION_SET_ENABLED, (e) => {
        if (e.detail?.enabled && turnFinalReached) {
            turnFinalReached = false;
            console.log('🔔 字幕モードON - voice:end 待ちのターンを閉じてタイマーを開始します');
            startTimer();
        }
    });

    console.log('無応答タイマーを初期化しました');
}

// =========================================================================
// イベントハンドラ
// =========================================================================

/**
 * 会話ターン状態の変化からタイマー開始を導出する（basic_design §4.6）。
 *
 * 見るのは `audible` の **○→✕**（ミクの声が鳴り止んだ瞬間）だけ。歌かどうかは
 * turnState 側が `singing ⇒ audible` の不変条件で吸収しているので、ここでは
 * 歌の存在を知らなくてよい——それが状態を集約している理由でもある。
 */
function handleTurnStateChange(e) {
    const { audible, singing, prev } = e.detail;

    // ★**また音が鳴り出したら数えるのをやめる**（2026-09-22）。
    //   歌は「喋り終わり（voice:end）」を合図に始まるので、**歌い出す一瞬だけ
    //   audible が落ちる**。`songPlayer` の再生開始は非同期なので song:start は必ず後に来る——
    //   つまりこれは偶発的な競合ではなく**毎回そうなる**。turnState の
    //   `singing ⇒ audible` はこの順番までは吸収できない。
    //   その隙にタイマーが走り出し、**歌の最中に無応答リマインダーが飛んで、
    //   ミクが自分の歌へかぶせて喋っていた**（2026-09-21 の収録で歌い出し 65 秒後に発生）。
    if (audible && !prev.audible) {
        // 歌い終わり（song:end で audible が落ちる）に数え直せるよう、完了フラグを戻す。
        // ★戻すのは**歌のときだけ**。ツール呼び出し中のチャプター再開でも戻すと、
        //   ターンの途中でリマインダーが飛ぶ（turnFinalReached はそれを防ぐためにある）
        if (singing) {
            turnFinalReached = true;
            console.log('🔕 歌い出した - 無応答タイマーを止めます（歌い終わりから数え直す）');
        }
        stopTimer();
        return;
    }

    if (audible || !prev.audible) return;   // 「鳴っていた → 鳴り止んだ」以外は対象外

    if (!turnFinalReached) {
        // ループの途中（ツール呼び出し中のチャプター完了など）→ タイマーは開始しない
        console.log('🔕 音声再生完了（ループ途中）- 最終応答前のためタイマーは開始しません');
        return;
    }
    console.log('🔔 音声再生完了（最終応答）- 無応答タイマーを開始します');
    turnFinalReached = false;  // 消費（次ターンの responseReady まで開始しない）
    startTimer();
}

function handleThinkingStart() {
    console.log('🔔 フォーム送信 - 無応答タイマーを停止します');
    turnFinalReached = false;  // 新しいターン開始 → フラグをリセット
    stopTimer();
}

/** server:responseReady → 最終ループに到達したので voice:end でタイマー開始を許可 */
function handleResponseReady(e) {
    turnFinalReached = true;
    console.log('🔔 最終応答確定 - 次の voice:end で無応答タイマーを開始します');

    // 字幕モード中は音声再生がスキップされ voice:end が来ない → ここでタイマーを開始する。
    // has_audio=true（テキストありの最終応答）のときのみ。ツールループ中（has_audio=false）は対象外。
    // 無返信は voice:end 経由で従来どおり開始される。
    // 判定はターンラッチ（appState.isCaptionActive）を使う: ターン途中のトグルで
    // 「音声は破棄済みなのに voice:end 待ち」になる不整合を防ぐ。
    if (appState.isCaptionActive() && e.detail?.has_audio) {
        const t = e.detail.inactivity_timeout;
        if (t !== undefined) setTimeoutSeconds(t);
        turnFinalReached = false;  // 消費（万一 voice:end が来ても二重開始しない）
        console.log('🔔 字幕モード - 無応答タイマーを即時開始します');
        startTimer();
    }
}

function handleInputFocus() {
    pauseTimer();
}

function handleInputBlur() {
    resumeTimer();
}

/** server:inactivityReminder → タイムアウト値を更新 */
function handleServerInactivityReminder(e) {
    const { inactivity_timeout } = e.detail;
    if (inactivity_timeout !== undefined) {
        setTimeoutSeconds(inactivity_timeout);
        console.log(`🔔 無応答タイマーを設定: ${inactivity_timeout}秒`);
    }
}

/** server:disconnected → タイマー強制停止 */
function handleServerDisconnected() {
    forceStopTimer();
    console.log('🔔 無応答タイマーを停止しました（切断時）');
}

/** app:inactivityTimeout → タイムアウト値を更新 */
function handleAppInactivityTimeout(e) {
    const { seconds } = e.detail;
    if (seconds !== undefined) {
        setTimeoutSeconds(seconds);
    }
}

/** app:reminderEnabled → リマインダーのON/OFFを更新 */
function handleAppReminderEnabled(e) {
    reminderEnabled = e.detail?.enabled !== false;
    console.log(`🔔 無応答リマインダー: ${reminderEnabled ? '有効' : '無効'}`);
    if (!reminderEnabled) stopTimer();
}

// =========================================================================
// タイマー制御（外部公開）
// =========================================================================

/**
 * タイムアウト秒数を設定
 * @param {number} seconds - タイムアウト秒数（0以下の場合は無効化）
 */
export function setTimeoutSeconds(seconds) {
    timeoutSeconds = seconds;
    if (seconds <= 0) {
        stopTimer();
        console.log(`無応答タイマーを無効化しました（設定値: ${seconds}）`);
    } else {
        console.log(`無応答タイマーのタイムアウトを設定: ${seconds}秒`);
    }
}

/**
 * タイマーをリセット（外部から呼び出し用）
 */
export function resetTimer() {
    startTimer();
}

/**
 * タイマーを強制停止（外部から呼び出し用）
 */
export function forceStopTimer() {
    stopTimer();
}

// =========================================================================
// タイマー制御（内部）
// =========================================================================

function startTimer(seconds = null) {
    stopTimer();

    if (timeoutSeconds <= 0) {
        console.log('無応答タイマーは無効化されています');
        return;
    }

    const duration = seconds !== null ? seconds : timeoutSeconds;
    timerStartTime = Date.now();
    remainingTime = duration;
    isPaused = false;

    timer = setTimeout(() => {
        console.log('⏰ 無応答タイマーが切れました - リマインダー要求を送信します');
        requestInactivityReminder();
    }, duration * 1000);

    console.log(`無応答タイマーを開始しました（${duration}秒）`);
}

function stopTimer() {
    if (timer !== null) {
        clearTimeout(timer);
        timer = null;
        console.log('無応答タイマーを停止しました');
    }
}

function pauseTimer() {
    if (timer === null || isPaused) return;

    const elapsedMs = Date.now() - timerStartTime;
    remainingTime = Math.max(0, remainingTime - elapsedMs / 1000);

    clearTimeout(timer);
    timer = null;
    isPaused = true;

    console.log(`⏸️ 無応答タイマーを一時停止しました（残り: ${remainingTime.toFixed(1)}秒）`);
}

function resumeTimer() {
    if (!isPaused) return;

    if (remainingTime <= 0) {
        console.log('⏰ 無応答タイマーの残り時間がありません - リマインダー要求を送信します');
        isPaused = false;
        requestInactivityReminder();
        return;
    }

    isPaused = false;
    timerStartTime = Date.now();

    timer = setTimeout(() => {
        console.log('⏰ 無応答タイマーが切れました - リマインダー要求を送信します');
        requestInactivityReminder();
    }, remainingTime * 1000);

    console.log(`▶️ 無応答タイマーを再開しました（残り: ${remainingTime.toFixed(1)}秒）`);
}

/**
 * サーバーに無応答リマインダー要求を送信
 */
function requestInactivityReminder() {
    if (!reminderEnabled) {
        console.log('🔕 無応答リマインダーは無効化されています - 要求を送信しません');
        return;
    }

    const socket = getSocket();
    const sessionId = getSessionId();

    if (!socket) {
        console.error('WebSocket接続がありません');
        return;
    }

    socket.emit(SOCKET_EVENTS.INACTIVITY_REMINDER_REQUEST, {
        sessionId:     sessionId,
        timestamp:     Date.now(),
        timeout:       timeoutSeconds,
        camera_frames: appState.getCameraBuffer(),
    });

    console.log(`📡 無応答リマインダー要求を送信しました（セッション: ${sessionId}, タイムアウト: ${timeoutSeconds}秒）`);
}