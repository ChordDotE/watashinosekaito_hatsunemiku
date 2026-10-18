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
 * uiStrings.js - UI文字列の集中管理モジュール
 * 
 * ボタン名やメッセージなどのUI文字列を一箇所で管理します。
 */

export const UI_STRINGS = {
    // 送信ボタンの状態別テキスト
    submitButton: {
        default: '話す',              // 通常時
        thinking: '考え中（追加OK）', // 送信後・応答生成中（畳み込み送信可・押せる）
        playing: '返答中...',         // 音声再生中
        singing: '♪ 歌ってる',        // 歌唱中（**押せる**＝話しかけると歌をやめて応じる）
        songPlaying: '♪ 流してる',    // 歌わずに曲を流している（インスト・作りかけ。押せるのは歌唱中と同じ）
        connecting: '接続中...'       // 起動時のサーバー疎通確認中
    },

    // プレースホルダー
    placeholder: {
        input: '話す内容を入れてください'
    },

    // マイクボタンの状態別ツールチップ（長押しヒントはマイクON中の状態にのみ付記）
    micButton: {
        OFF: '音声認識をONにする',
        LISTENING: '音声認識ON（待機中）... クリックで自動送信ON、長押しで履歴・音量パネル',
        LISTENING_AUTO: '音声認識+自動送信ON... クリックでOFF、長押しで履歴・音量パネル',
        RECORDING: '録音中... クリックでOFF、長押しで履歴・音量パネル',
        RECORDING_AUTO: '録音中（自動送信ON）... クリックでOFF、長押しで履歴・音量パネル',
        PROCESSING: '認識中...',
        MUTED: 'ミクが話し中（聞き取りを一時停止）',
    },

    // 入力アイドル自動送信トグル（STTオフ時に micButton を流用）のツールチップ
    idleAutosend: {
        OFF: '自動送信をONにする（音声入力 → 一定時間で自動送信）',
        ON:  '自動送信ON（入力が止まると自動送信）... クリックでOFF',
    },

    // 返答字幕（miku-caption）トグルのツールチップ。ON中は音声・リップシンクを停止する
    caption: {
        ON:  '字幕表示ON（音声・口の動きはオフ）... クリックで音声に戻す',
        OFF: '字幕表示に切り替える（音声を止めて文字で読む）',
    },

    // 無応答リマインダーの「ミュート」トグルのツールチップ（既定OFF=ミュートなし・永続化なし）
    // ON=ミュート中（応答しない）、OFF=リマインダー有効（従来通り話しかける）
    reminder: {
        OFF: '無応答リマインダー有効（返信がないと話しかけます）... クリックでミュート',
        ON:  '無応答リマインダーをミュート中（返信が無くても話しかけません）... クリックで解除',
    },

    // カメラボタンのツールチップ（タップでON/OFF、長押しでプレビュー）
    camera: {
        ON:  'カメラON（長押しでプレビュー）',
        OFF: 'カメラOFF（タップでON）',
    },

    // カメラ切替ボタン（表示メニュー内）のツールチップ・トースト文言
    cameraSwitch: {
        ON:     'カメラを切り替える',
        OFF:    'カメラがOFFです（先にカメラをONにしてください）',
        SINGLE: '切り替えられるカメラがありません',
    },

    // 視線追従ボタン（表示メニュー内）のツールチップ。既定ON・永続化なし。効くのはカメラONのときだけ
    gaze: {
        ON:  '視線追従ON（カメラONのとき、ミクがあなたの顔の方を向く）... クリックでOFF',
        OFF: '視線追従をONにする（カメラONのとき、ミクがあなたの顔の方を向く）',
        DISABLED: '視線追従はカメラ（インカメラ）がONのときに使えます',
    },

    // 会話メニュー（字幕・無応答リマインダーをまとめる親ボタン）のツールチップ
    conversation: {
        menu: '会話のやり取り設定（字幕・リマインダー）',
    },

    // 表示メニュー（カメラ切替／視線／画面ロック／全画面）のボタンツールチップ
    display: {
        menu:          '表示メニュー（カメラ切替・視線・画面ロック・全画面）',
        lock:          '画面ロック（放置会話・誤操作防止）',
        fullscreenOff: '全画面表示にする',
        fullscreenOn:  '全画面表示を解除する',
    },

    // 画面ロックモード（miku-lock オーバーレイ）
    lock: {
        hint: '枠の点線を一周なぞると解除（5秒でリセット）',
    },

    // エラーメッセージ
    error: {
        generateFailed: '応答の生成に失敗しました',
        generic: 'エラーが発生しました',
        micPermission: 'マイクの使用が許可されませんでした',
    },

    // 設定ページ（/settings）
    settings: {
        title: '設定',
        loading: '設定を読み込んでいます...',
        loadFailed: '設定の読み込みに失敗しました',
        save: '保存',
        saving: '保存中...',
        saved: '保存しました。反映にはシステムの再起動が必要です',
        saveFailed: '保存に失敗しました',
        noChanges: '変更はありません',
        secretPlaceholder: '未設定（入力すると保存されます）',
        secretKeepHint: '設定済み（末尾3文字のみ表示）。変更する場合のみ入力、空のままなら現在の値を維持',
        restartNote: '設定はシステム再起動時に反映されます。壊れた場合は settings.json.bak から復元できます。',
        backToChat: 'チャットに戻る',
        maintLinkNote: '※同一ネットワーク内からのみ開けます（外部アクセス時は開けません）',
        linkNeedsRunning: '起動すると開けます',
        restartNow: '今すぐ再起動して反映（記憶の整理はスキップ）',
        restarting: '再起動しています…',
        restarted: '再起動しました',
        restartFailed: '再起動に失敗しました',
    }
};
