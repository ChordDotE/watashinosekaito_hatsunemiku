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
 * turnState.js - 会話ターン状態の単一ソース（basic_design §4.6 の3パラメータ）
 *
 * 人間の会話の3軸に対応する3つの状態パラメータを一元管理する:
 *   audible      : ミクの声が実際に鳴っているか（自分の口）
 *   turnBelief   : ターン進行の認識 NO_TURN / OPEN(考え中) / CONFIRMED(応答確定)（応答を組み立て中か）
 *   userSpeaking : ユーザーがいま話しているか（相手の口・VAD観測）
 *   singing      : 曲が鳴っている最中か（audible の内訳。ボタン表示だけを分けるために持つ）
 *   songMode     : 鳴っている曲の mode（'sing' 歌う / 'play' 流す / 'preview' 作りかけ。鳴っていなければ null）。
 *                  singing の内訳で、ボタンの言い方（「♪ 歌ってる」か「♪ 流してる」か）だけに使う（2026-10-03）
 *   userTurnOpen : マスターの番がまだ続いているか（言い終わっていない）
 *
 * `userSpeaking` と `userTurnOpen` は別物:
 *   userSpeaking … VAD の観測そのもの。**無音 800ms で落ちる**（microphone.js の
 *                  silenceHangover）。助詞の前の間や考えている間でも落ちる
 *   userTurnOpen … 「まだ話が終わっていない」という**会話上の判断**。Smart Turn の
 *                  終話判定（turn_complete）で閉じ、判定が付かない場合だけ猶予で閉じる
 * 音声再生の保留に使うのは **userTurnOpen**。userSpeaking で判断すると、
 * 800ms の隙間にミクの声が差し込まれる（2026-08-13）。
 *
 * 遷移はこのモジュールだけが行う（document イベントを購読して更新）。
 * UI（送信ボタン・マイクミュート等）は TURN_STATE_CHANGE を購読して**導出**する。
 * DOM（class / disabled / dataset）を状態庫にしない。
 *
 * turnBelief はサーバー TurnHandle の**写し（belief）**であり、ズレは /generate の
 * 畳み込み分岐（folded）が照合して吸収する。サーバーへは問い合わせない。
 *
 * 状態×振る舞いの外部仕様は docs/basic_design.md §4.6 を正とする。
 */

import { EVENTS } from './events.js';
import { appState } from './appState.js';

/** turnBelief の値 */
export const TurnBelief = Object.freeze({
    NO_TURN: 'NO_TURN',       // ターンなし（待機）
    OPEN: 'OPEN',             // 考え中（畳み込み受付中）
    CONFIRMED: 'CONFIRMED',   // 最終応答確定済み（音声の消化待ち/消化済み）
});

/** Smart Turn が「まだ続く（turn_complete=false）」と言った後、次の発話が来なければ
 *  マスターの番を閉じるまでの猶予。短すぎると言い直しの間に割り込み、長すぎると
 *  黙った後にミクが喋り出すまで待たされる */
const USER_TURN_GRACE_MS = 1200;

/** interim が届いてから、対応する final を待つ上限。
 *
 *  **声が止まっても、STT がまだ確定していないうちは「喋り終わった」と見なさない**
 *  （2026-09-22）。VAD の無音検知（＋猶予 1200ms）だけで閉じると、確定処理が
 *  間に合わないうちにミクが鳴り始め、**マイクのミュートが STT ストリームごと畳んで
 *  算出済みの final（話者照合スコア付き）を落とす**——実機で発生し、最後のひと言が
 *  スコアなしの素のテキストで送られていた。
 *  確定が永久に来ない場合（接続断・認識失敗）に備えて必ず上限で諦める。 */
const FINALIZE_WAIT_MAX_MS = 2500;

class TurnState {
    constructor() {
        this._audible = false;
        this._singing = false;
        this._songMode = null;
        this._turnBelief = TurnBelief.NO_TURN;
        this._userSpeaking = false;
        this._userTurnOpen = false;
        this._graceTimer = null;
        this._initialized = false;
        // STT の確定待ち（interim は届いたが対応する final がまだ）。**内部の事実**で、
        // 購読側へは出さない——閉じるかどうかの判断はここ1か所に閉じ、
        // マイクのミュート・再生の保留・サーバーへの speaking 通知は
        // 従来どおり userTurnOpen / userSpeaking からの導出に任せる
        this._sttFinalizing = false;
        this._finalizeCapTimer = null;
        // 猶予が満了したが確定待ちで閉じられなかった（確定 or 上限で閉じ直す）
        this._closePending = false;
    }

    get audible() { return this._audible; }
    get singing() { return this._singing; }
    get songMode() { return this._songMode; }
    get turnBelief() { return this._turnBelief; }
    get userSpeaking() { return this._userSpeaking; }
    get userTurnOpen() { return this._userTurnOpen; }

    /** 現在値のスナップショット（導出の再適用などに使う） */
    snapshot() {
        return {
            audible: this._audible,
            singing: this._singing,
            songMode: this._songMode,
            turnBelief: this._turnBelief,
            userSpeaking: this._userSpeaking,
            userTurnOpen: this._userTurnOpen,
        };
    }

    _clearGrace() {
        if (this._graceTimer) {
            clearTimeout(this._graceTimer);
            this._graceTimer = null;
        }
    }

    /**
     * 猶予を置いてマスターの番を閉じる（話し始めれば取り消される）。
     *
     * **STT の確定だけに頼らない**のが要点。確定が届かなくなる状況（接続が切れた・
     * 認識に失敗した）は実際に起きており、そのとき userTurnOpen が立ちっぱなしになると
     * 以降のミクの声が毎回上限まで待たされる。VAD が無音を検知した時点でも
     * この猶予を回し、**確定が来なくても必ず閉じる**ようにする（2026-08-13 実機で発覚）。
     */
    _closeAfterGrace() {
        this._clearGrace();
        this._graceTimer = setTimeout(() => {
            this._graceTimer = null;
            // 満了時にまだ話していたら閉じない。STT の確定は**次の発話の録音中**に
            // 届くことがあり（畳み込みではむしろ普通）、そこで閉じると話している最中に
            // ミクが鳴り出す。話し終わりに MIC_STATE_CHANGE が改めて猶予を引く
            if (this._userSpeaking) return;
            // ★確定待ちなら閉じない（2026-09-22）。final が届けば STT_UTTERANCE_END が
            // 改めて閉じ、来なければ上限（FINALIZE_WAIT_MAX_MS）が諦めて閉じる
            if (this._sttFinalizing) {
                this._closePending = true;
                return;
            }
            this._set({ userTurnOpen: false });
        }, USER_TURN_GRACE_MS);
    }

    /** 確定待ちに入る（interim 到着）。上限タイマーは届くたびに引き直す。 */
    _beginFinalizing() {
        this._sttFinalizing = true;
        if (this._finalizeCapTimer) clearTimeout(this._finalizeCapTimer);
        this._finalizeCapTimer = setTimeout(() => {
            this._finalizeCapTimer = null;
            // 上限で諦める。確定は来なかったが、待ち続けると以降ミクが永久に黙る
            this._endFinalizing();
        }, FINALIZE_WAIT_MAX_MS);
    }

    /** 確定待ちを終える（final 到着 / 上限 / マイク OFF）。保留していた close をここで実行する。 */
    _endFinalizing() {
        this._sttFinalizing = false;
        if (this._finalizeCapTimer) {
            clearTimeout(this._finalizeCapTimer);
            this._finalizeCapTimer = null;
        }
        if (this._closePending) {
            this._closePending = false;
            if (!this._userSpeaking) this._set({ userTurnOpen: false });
        }
    }

    /** document イベントの購読を開始する（多重呼び出しは無視） */
    init() {
        if (this._initialized) return;
        this._initialized = true;

        // ターン開始（フォーム送信。畳み込み送信は THINKING_START を発火しないが既に OPEN）
        document.addEventListener(EVENTS.THINKING_START, () => {
            this._set({ turnBelief: TurnBelief.OPEN });
        });

        // ターン終了扱い（送信エラー / 音声なし応答の確定 / 字幕切替の後始末）
        document.addEventListener(EVENTS.THINKING_END, () => {
            this._set({ turnBelief: TurnBelief.NO_TURN });
        });

        // 最終応答確定 = サーバー側の畳み込み窓は閉じた
        document.addEventListener(EVENTS.SERVER_RESPONSE_READY, (e) => {
            const hasAudio = Boolean(e.detail?.has_audio) && !appState.isCaptionActive();
            this._set({ turnBelief: hasAudio ? TurnBelief.CONFIRMED : TurnBelief.NO_TURN });
        });

        // 可聴区間（voicePcmPlayer が「実際に音が出ている区間」で発火）
        document.addEventListener(EVENTS.VOICE_START, () => {
            this._set({ audible: true });
        });
        document.addEventListener(EVENTS.VOICE_END, (e) => {
            // **歌が鳴っている間は audible を下ろさない**（2026-08-16）。
            // 歌の直前に喋った声ぶんの voice:end がここへ届くが、それで下ろすと
            // `singing ⇒ audible` の不変条件が壊れ、§4.6 の表に無い
            // 「audible=✕ かつ singing=○」に落ちる。実際にそうなり、歌の最中に
            // マイクが開いて自分の歌を拾い、無応答タイマーまで動き出していた。
            // 歌の終了は SONG_END が閉じる（そちらで turnBelief も畳む）
            if (this._singing) return;
            // ★ツール呼び出し待ち等で同じ発話の続きがまだ来る場合
            // （voicePcmPlayer.js のデータ待ちギャップ・`moreExpected: true`）は、
            // ターン完了と誤認せず OPEN（考え中）に戻す。ここで NO_TURN/CONFIRMED
            // のままにすると、「ターンが終わった」ときと同じ濃い緑の LISTENING に
            // 見えてしまう（2026-09-22 実機で確認。data-turn-open は OPEN のときだけ立つ）。
            // 続きの本応答が来れば SERVER_RESPONSE_READY が確定した turnBelief で上書きする
            if (e.detail?.moreExpected) {
                this._set({ audible: false, turnBelief: TurnBelief.OPEN });
                return;
            }
            // CONFIRMED 中の再生終了 = ターン完了
            this._set({
                audible: false,
                ...(this._turnBelief === TurnBelief.CONFIRMED
                    ? { turnBelief: TurnBelief.NO_TURN } : {}),
            });
        });

        // 歌っている区間（songPlayer が発火）。**歌もミクの声なので audible に含める**
        // ＝マイクは既存の半二重規律のままミュートされる（自分の歌を拾わない）。
        // ただし送信ボタンは塞がない（話しかけて中断できることが中断手段そのもの）ので
        // singing を別に持ち、ボタンの導出だけ分けている（miku-chat._applyTurnState）
        // ★mode も持つ（2026-10-03）。インスト（play）・作りかけ（preview）で「♪ 歌ってる」と出さないため
        document.addEventListener(EVENTS.SONG_START, (e) => {
            this._set({ audible: true, singing: true, songMode: e.detail?.mode || 'sing' });
        });
        document.addEventListener(EVENTS.SONG_END, () => {
            // 歌い終わりがターンの終わり。VOICE_END は歌の最中は素通しにしているので、
            // CONFIRMED を畳むのはここの責務になる（音の終わりは1か所でだけ判断する）
            this._set({
                audible: false, singing: false, songMode: null,
                ...(this._turnBelief === TurnBelief.CONFIRMED
                    ? { turnBelief: TurnBelief.NO_TURN } : {}),
            });
        });

        // 字幕ON = 再生中音声の破棄（voice:end が来ないケースの保険）
        document.addEventListener(EVENTS.CAPTION_SET_ENABLED, (e) => {
            if (e.detail?.enabled) this._set({ audible: false, singing: false, songMode: null });
        });

        // 切断 = ターンの行方は不明 → 楽観的に閉じる（照合は /generate 側）
        document.addEventListener(EVENTS.SERVER_DISCONNECTED, () => {
            this._set({ audible: false, singing: false, songMode: null, turnBelief: TurnBelief.NO_TURN });
        });

        // 相手の口（VAD 観測）
        document.addEventListener(EVENTS.MIC_STATE_CHANGE, (e) => {
            const state = e.detail?.state;
            const patch = { userSpeaking: state === 'RECORDING' };
            if (state === 'RECORDING') {
                // 話し始めた → マスターの番。猶予で閉じかけていたら取り消す
                this._clearGrace();
                this._closePending = false;
                patch.userTurnOpen = true;
            } else if (state === 'OFF') {
                // マイクを切った → 以降 final も来ないので即座に閉じる
                this._clearGrace();
                this._closePending = false;
                this._endFinalizing();
                patch.userTurnOpen = false;
            } else if (this._userTurnOpen) {
                // 話し終わった（VAD が無音を検知）→ 猶予を置いて閉じる。
                // STT の確定が来ればそちらが上書きする（下の分岐）
                this._closeAfterGrace();
            }
            this._set(patch);
        });

        // STT の確定結果。マスターの番を閉じてよいかはここで決まる
        // 言葉として認識された（interim が届いた）→ ここから**確定待ち**に入る。
        // 声が止まってもこの間は「喋り終わった」と見なさない（2026-09-22）
        document.addEventListener(EVENTS.STT_SPEECH_EVIDENCE, () => {
            this._beginFinalizing();
        });

        document.addEventListener(EVENTS.STT_UTTERANCE_END, (e) => {
            // 確定が届いた → 待ちを解く（保留していた close はこの後の分岐が改めて決める）
            this._closePending = false;
            this._endFinalizing();
            if (e.detail?.turnComplete === true && !this._userSpeaking) {
                // Smart Turn が「言い終わった」と判定 → 猶予を待たず即座に閉じる。
                // ただし**今まさに次を話している最中なら閉じない**（この確定は
                // ひとつ前の発話に対するもので、目の前の発話はまだ続いている）
                this._clearGrace();
                this._set({ userTurnOpen: false });
                return;
            }
            // 「続きがありそう」または Smart Turn 無効 → 猶予を引き直してから閉じる。
            // 猶予中に話し始めれば上の RECORDING 分岐が取り消す
            this._closeAfterGrace();
        });
    }

    _set(patch) {
        const prev = this.snapshot();
        const next = { ...prev, ...patch };
        if (next.audible === prev.audible
            && next.singing === prev.singing
            && next.songMode === prev.songMode
            && next.turnBelief === prev.turnBelief
            && next.userSpeaking === prev.userSpeaking
            && next.userTurnOpen === prev.userTurnOpen) {
            return;
        }
        this._audible = next.audible;
        this._singing = next.singing;
        this._songMode = next.songMode;
        this._turnBelief = next.turnBelief;
        this._userSpeaking = next.userSpeaking;
        this._userTurnOpen = next.userTurnOpen;
        console.log(`[TurnState] belief=${next.turnBelief} audible=${next.audible} singing=${next.singing}${next.songMode ? `(${next.songMode})` : ''} `
            + `userSpeaking=${next.userSpeaking} userTurnOpen=${next.userTurnOpen}`);
        document.dispatchEvent(new CustomEvent(EVENTS.TURN_STATE_CHANGE, {
            detail: { ...next, prev },
        }));
    }
}

export const turnState = new TurnState();
