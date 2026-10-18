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
 * main.js - エントリーポイント
 * 
 * アプリケーションの初期化を行う
 * - セッションID初期化
 * - WebSocket接続
 * - 音声キュー初期化
 * - サーバー接続確認
 */

// モジュールのインポート
import { EVENTS } from './modules/events.js';
import { getSessionId } from './modules/session.js';
import { initSocket } from './modules/socket.js';
import { initInactivityTimer } from './modules/inactivityTimer.js';
import { initRemoteLog } from './modules/remoteLog.js';
import { initUserSpeechNotifier } from './modules/userSpeechNotifier.js';
// Fish Audio PCM ストリーミングプレイヤー（インポートするだけでイベントリスナーが登録される）
import './modules/voicePcmPlayer.js';
// 歌唱プレイヤー（同上。曲の予約を受け、ミクが喋り終わってから鳴らす）
import './modules/songPlayer.js';
// 視線追従（同上。<miku-chat> が最初の ON/OFF を出す前に待ち受けている必要がある）
import './modules/gazeTracker.js';

// Web Components のインポート（登録のみ）
// Live2D版アバターコンポーネントを使用
import './components/miku-avatar/miku-avatar-live2d.js';
import './components/miku-chat/miku-chat.js';
import './components/miku-caption/miku-caption.js';
import './components/miku-help/miku-help.js';
import './components/miku-lock/miku-lock.js';

/**
 * アプリケーション初期化
 */
/**
 * ソフトキーボード表示時にレイアウト高を可視領域へ追従させる（iOS Safari 対応）。
 *
 * Android Chrome は viewport メタの interactive-widget=resizes-content でレイアウト自体が
 * 縮むが、iOS Safari はこのメタを無視してキーボードが画面に覆い被さる。
 * visualViewport の高さを CSS 変数 --app-height に流し込み、モバイルレイアウトの
 * body/.container がこれを参照することで、iOS でもアバター領域が縮んで全身表示が保たれる。
 */
function initViewportHeightSync() {
    if (!window.visualViewport) return;
    const apply = () => {
        document.documentElement.style.setProperty(
            '--app-height', `${Math.round(window.visualViewport.height)}px`
        );
        // キーボード表示時のページパン（入力欄へのスクロール）を打ち消す
        window.scrollTo(0, 0);
    };
    window.visualViewport.addEventListener('resize', apply);
    apply();
}

async function initApp() {
    // ブラウザ挙動ログのサーバー転送（console フックのため最初に開始する）
    initRemoteLog();
    // 考え中の発話開始・終了をサーバーへ一報する（確定保留の判断材料）
    initUserSpeechNotifier();

    console.log('🚀 Miku Agent アプリケーション初期化開始');

    // 0. ビューポート高の追従（iOS のソフトキーボード対応）
    initViewportHeightSync();

    // 1. セッションIDを取得/生成
    const sessionId = getSessionId();
    console.log(`📋 セッションID: ${sessionId}`);

    // 2. WebSocket接続を初期化
    initSocket();
    console.log('🔌 WebSocket初期化完了');

    // 3. 無応答タイマーを初期化
    initInactivityTimer();
    console.log('⏰ 無応答タイマー初期化完了');

    // 4. Live2D iframeからのpostMessageをlisten
    setupIframeMessageListener();
    console.log('📨 iframe通信リスナー初期化完了');

    // 5. サーバー接続確認
    await checkServerConnection();

    console.log('✅ Miku Agent アプリケーション初期化完了');
}

/**
 * Live2D iframeからのpostMessageを受信してCustomEventを発火
 */
function setupIframeMessageListener() {
    window.addEventListener('message', (event) => {
        // 安全チェック（同一オリジンからのメッセージのみ処理）
        if (event.origin !== window.location.origin) {
            return;
        }

        const data = event.data;
        if (!data || typeof data.type !== 'string') {
            return;
        }

        // voice:start イベント
        if (data.type === EVENTS.VOICE_START) {
            console.log('📨 iframeからvoice:start受信');
            document.dispatchEvent(new CustomEvent(EVENTS.VOICE_START, {
                bubbles: true,
                composed: true
            }));
        }

        // voice:end イベント
        if (data.type === EVENTS.VOICE_END) {
            console.log('📨 iframeからvoice:end受信');
            document.dispatchEvent(new CustomEvent(EVENTS.VOICE_END, {
                bubbles: true,
                composed: true
            }));
        }
    });
}

/**
 * サーバーとの接続確認（ping）
 */
async function checkServerConnection() {
    // ボタンの表示制御は miku-chat が app:connectionChecking / app:connectionResult を
    // 購読して行う（Shadow DOM への直接操作は境界規約違反のため廃止）。
    // miku-chat の初期化前にイベントが発火した場合は誰も購読していないだけで無害。
    document.dispatchEvent(new CustomEvent(EVENTS.APP_CONNECTION_CHECKING));

    let success = false;
    try {
        const response = await fetch('/ping', { method: 'POST' });
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || '接続に失敗しました');
        }

        console.log('✅ サーバー接続確認成功');
        success = true;

    } catch (error) {
        console.error('❌ 接続確認エラー:', error);

    } finally {
        document.dispatchEvent(new CustomEvent(EVENTS.APP_CONNECTION_RESULT, {
            detail: { success }
        }));
    }
}

// DOMContentLoadedでアプリケーションを初期化
document.addEventListener('DOMContentLoaded', initApp);

// ページ読み込み完了時の追加処理
window.addEventListener('load', () => {
    console.log('📄 ページ読み込み完了');
});