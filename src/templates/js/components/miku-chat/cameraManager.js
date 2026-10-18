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
 * cameraManager.js - カメラ制御・バッファ管理・プレビュー吹き出し
 *
 * <miku-chat> の内部部品（Shadow DOM を引数に取る。単独では動かない）。
 * 取得フレームは appState.setCameraBuffer() で共有し、inactivityTimer が参照する。
 * 映像そのもの（MediaStream）は ON・切替・OFF のたびに EVENTS.CAMERA_STREAM_CHANGE で知らせ、
 * gazeTracker（視線追従）が同じ映像を共有する。
 */

import { EVENTS } from '../../modules/events.js';
import { appState } from '../../modules/appState.js';
import { UI_STRINGS } from '../../modules/uiStrings.js';
import { attachPressGesture } from '../../modules/pressGesture.js';
export class CameraManager {
    /**
     * @param {ShadowRoot} shadow
     * @param {{ bufferSize: number }} options
     */
    constructor(shadow, options = {}) {
        this._shadow      = shadow;
        this._bufferSize  = options.bufferSize || 3;
        this._cameraOn    = false;
        this._cameraStream       = null;
        this._cameraBuffer       = [];    // Blob[]
        this._cameraSampleTimer  = null;
        this._previewVisible     = false;
        /** @type {MediaDeviceInfo[]} videoinput 一覧（カメラON後に label 込みで再列挙） */
        this._videoDevices       = [];
        /** @type {number} 現在使用中のカメラの _videoDevices 上の位置 */
        this._deviceIndex        = 0;
        /** @type {boolean} 長押し切替の多重起動防止 */
        this._switching          = false;

        // 発話中はサンプリング停止、終了後に再開
        document.addEventListener(EVENTS.VOICE_START, () => {
            if (this._cameraOn) this._stopSampling();
        });
        document.addEventListener(EVENTS.VOICE_END, () => {
            if (this._cameraOn && this._cameraStream) this._startSampling();
        });
    }

    /** カメラが使用可能か確認し、ボタンを表示してイベントを設定する */
    async init(cameraEnabledByServer) {
        const btn = this._shadow.getElementById('cameraButton');
        if (!btn) return;

        if (!cameraEnabledByServer) {
            console.log('[Camera] サーバー設定で無効 → カメラボタン非表示');
            return;
        }
        if (!navigator.mediaDevices?.getUserMedia) {
            console.log('[Camera] getUserMedia 非対応 → カメラボタン非表示');
            return;
        }

        const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
        if (!devices.some(d => d.kind === 'videoinput')) {
            console.log('[Camera] カメラデバイスなし → カメラボタン非表示');
            return;
        }

        btn.style.display = 'flex';
        console.log('[Camera] カメラ利用可 → カメラボタン表示');
        this._setupButtonEvents(btn);

        // カメラ切替ボタン（表示メニュー内）も同条件で表示する。
        // 表示/非表示＝環境として使えるか、有効/無効＝今この瞬間使えるか、で統一する
        // （既存の流儀。カメラボタン自体もサーバー無効・デバイス無しなら非表示）
        const switchBtn = this._shadow.getElementById('cameraSwitchButton');
        if (switchBtn) {
            switchBtn.style.display = 'flex';
            this._updateSwitchButtonUI();
        }
        // 視線ボタン（表示メニュー内）も同条件。視線はカメラの映像で追うので、
        // カメラが使えない環境では出さない（ON/OFF の配線は MenuController）
        const gazeBtn = this._shadow.getElementById('gazeButton');
        if (gazeBtn) gazeBtn.style.display = 'flex';
    }

    /** 映像が変わったことを知らせる（ON・切替・OFF の3か所から呼ぶ） */
    _announceStream() {
        document.dispatchEvent(new CustomEvent(EVENTS.CAMERA_STREAM_CHANGE, {
            detail: { stream: this._cameraStream },
        }));
    }

    _setupButtonEvents(btn) {
        attachPressGesture(btn, {
            onShortPress: () => {
                if (this._previewVisible) {
                    this._hidePreview();
                } else {
                    this._toggle();
                }
            },
            // 長押し = プレビュー表示（カメラ切替は表示メニューのボタンが担当。
            // 一度この長押しに切替を割り当てたが、プレビューと衝突して差し戻した・2026-08-05）
            onHoldStart: () => this._showPreview(),
        });
    }

    async _toggle() {
        if (this._cameraOn) {
            this._stopSampling();
            this._hidePreview();
            if (this._cameraStream) {
                this._cameraStream.getTracks().forEach(t => t.stop());
                this._cameraStream = null;
            }
            this._cameraBuffer = [];
            appState.clearCameraBuffer();
            this._cameraOn = false;
            this._videoDevices = [];
            this._deviceIndex = 0;
            this._updateButtonUI();
            this._updateSwitchButtonUI();
            this._announceStream();
            console.log('[Camera] OFF');
        } else {
            try {
                this._cameraStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user' }, audio: false,
                });
                this._cameraOn = true;
                await this._refreshDeviceList();
                this._updateButtonUI();
                this._updateSwitchButtonUI();
                this._startSampling();
                this._announceStream();
                console.log(`[Camera] ON (${this._currentDeviceLabel()}・${this._videoDevices.length}台)`);
            } catch (err) {
                console.error(`[Camera] カメラ起動失敗: ${err?.name}: ${err?.message}`);
                if (err.name === 'NotAllowedError') {
                    alert('カメラの使用が許可されていません。ブラウザの設定を確認してください。');
                }
            }
        }
    }

    /**
     * videoinput 一覧を取得し直し、現在のストリームの deviceId から位置を特定する。
     * label はカメラ許可後でないと取れないため、ON のタイミングで呼ぶ。
     */
    async _refreshDeviceList() {
        const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
        this._videoDevices = devices.filter(d => d.kind === 'videoinput');
        const currentId = this._cameraStream?.getVideoTracks()[0]?.getSettings().deviceId;
        const idx = this._videoDevices.findIndex(d => d.deviceId === currentId);
        this._deviceIndex = idx >= 0 ? idx : 0;
    }

    _currentDeviceLabel() {
        const d = this._videoDevices[this._deviceIndex];
        return d?.label || `カメラ${this._deviceIndex + 1}`;
    }

    /**
     * カメラ切替ボタン（表示メニュー内）の有効/無効を現在の状態から導出する。
     *
     * 有効 = カメラON かつ 2台以上（今この瞬間に切り替えられる）。
     * 無効でもボタンは残す（メニュー内の位置が動くと操作を覚えられないため）。
     */
    _updateSwitchButtonUI() {
        const btn = this._shadow.getElementById('cameraSwitchButton');
        if (!btn) return;
        const usable = this._cameraOn && this._videoDevices.length > 1;
        btn.dataset.switchState = usable ? 'ON' : 'DISABLED';
        btn.title = usable
            ? `${UI_STRINGS.cameraSwitch.ON}（${this._currentDeviceLabel()}）`
            : (this._cameraOn ? UI_STRINGS.cameraSwitch.SINGLE : UI_STRINGS.cameraSwitch.OFF);
    }

    /**
     * 次のカメラへ順次切替する（イン/アウトに限らず videoinput 一覧を巡回）。
     * 表示メニューのカメラ切替ボタンから呼ばれる。切替後はプレビューで結果を確認できる。
     *
     * 使えない状況（カメラOFF / カメラが1台）では切り替えず、理由をトーストで知らせる
     * （ボタンは disabled にせず押せるままにする。disabled だと click が発火せず理由を出せない）。
     *
     * @returns {Promise<boolean>} 実際に切り替えたか（呼び出し元がメニューを閉じるかの判断に使う）
     */
    async switchToNextCamera() {
        if (this._switching) return false;
        if (!this._cameraOn) {
            window.__mikuDebugToast?.(`📷 ${UI_STRINGS.cameraSwitch.OFF}`);
            console.log('[Camera] 切替不可: カメラOFF');
            return false;
        }
        if (this._videoDevices.length <= 1) {
            window.__mikuDebugToast?.(`📷 ${UI_STRINGS.cameraSwitch.SINGLE}`);
            console.log('[Camera] 切替不可: カメラが1台のみ');
            return false;
        }
        this._switching = true;
        const nextIndex = (this._deviceIndex + 1) % this._videoDevices.length;
        const next = this._videoDevices[nextIndex];
        const prev = this._videoDevices[this._deviceIndex];
        const open = (device) => navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: device.deviceId } }, audio: false,
        });
        // ★今のカメラを閉じてから次を開く（2026-09-27）。以前は「次を開けてから今のを閉じる」順で、
        //   2台を同時に開けないスマホ（4台の機種で確認）では次を開く時点で必ず失敗し、
        //   機能を入れた 2026-08-05 から一度も切り替わっていなかった（ブラウザのログに成功 0 件）。
        //   開けなかったときは元のカメラを開き直して、映像が消えたままにならないようにする
        this._stopSampling();
        if (this._cameraStream) this._cameraStream.getTracks().forEach(t => t.stop());
        this._cameraStream = null;
        try {
            let newStream;
            try {
                newStream = await open(next);
            } catch (err) {
                // エラーはオブジェクトのままだとログで {} になり種類が残らないので、名前と文言を文字にする
                console.error(`[Camera] カメラ切替失敗: ${err?.name}: ${err?.message}（${next.label || '次のカメラ'}）`);
                window.__mikuDebugToast?.(`📷 切替失敗: ${err?.name ?? err}`);
                try {
                    this._cameraStream = await open(prev);
                    this._startSampling();
                    this._announceStream();
                    console.log(`[Camera] 元のカメラに戻した: ${this._currentDeviceLabel()}`);
                } catch (err2) {
                    console.error(`[Camera] 元のカメラも開けなかった: ${err2?.name}: ${err2?.message} → カメラOFF`);
                    await this._toggle();   // _cameraOn が true のままなので OFF の後始末（UI・バッファ・映像の通知）が走る
                }
                return false;
            }
            this._cameraStream = newStream;
            this._deviceIndex = nextIndex;
            this._startSampling();
            this._announceStream();
            this._showPreview();
            this._updateButtonUI();
            this._updateSwitchButtonUI();
            console.log(`[Camera] カメラ切替: ${this._currentDeviceLabel()} (${nextIndex + 1}/${this._videoDevices.length})`);
            window.__mikuDebugToast?.(`📷 ${this._currentDeviceLabel()}`);
            return true;
        } catch (err) {
            console.error(`[Camera] カメラ切替の後処理で失敗: ${err?.name}: ${err?.message}`);
            window.__mikuDebugToast?.(`📷 切替失敗: ${err?.message ?? err}`);
            return false;
        } finally {
            this._switching = false;
        }
    }

    _startSampling() {
        this._stopSampling();
        this._cameraSampleTimer = setTimeout(async () => {
            const sample = async () => {
                if (!this._cameraOn || !this._cameraStream) return;
                const blob = await this._captureFrame();
                if (blob) {
                    this._cameraBuffer.push(blob);
                    if (this._cameraBuffer.length > this._bufferSize) this._cameraBuffer.shift();
                    console.log(`[Camera] フレームサンプリング: バッファ${this._cameraBuffer.length}枚`);
                    this._syncToAppState();
                    if (this._previewVisible) this._renderPreviewImages();
                }
                this._cameraSampleTimer = setTimeout(sample, 1000);
            };
            await sample();
        }, 1000);
        console.log('[Camera] サンプリング開始（1秒後）');
    }

    _stopSampling() {
        if (this._cameraSampleTimer !== null) {
            clearTimeout(this._cameraSampleTimer);
            this._cameraSampleTimer = null;
            console.log('[Camera] サンプリング停止');
        }
    }

    _syncToAppState() {
        Promise.all(
            this._cameraBuffer.map(blob => blob.arrayBuffer().then(buf =>
                'data:image/jpeg;base64,' + btoa(String.fromCharCode(...new Uint8Array(buf)))
            ))
        ).then(base64List => appState.setCameraBuffer(base64List));
    }

    _showPreview() {
        if (!this._cameraOn) return;
        const preview = this._shadow.getElementById('cameraPreview');
        if (!preview) return;
        this._previewVisible = true;
        preview.classList.add('visible');
        this._renderPreviewImages();
        console.log('[Camera] プレビュー表示');
    }

    _hidePreview() {
        const preview = this._shadow.getElementById('cameraPreview');
        if (!preview) return;
        this._previewVisible = false;
        preview.classList.remove('visible');
        console.log('[Camera] プレビュー非表示');
    }

    _renderPreviewImages() {
        const container = this._shadow.getElementById('cameraPreviewImages');
        if (!container) return;
        container.innerHTML = '';

        if (this._cameraBuffer.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'camera-preview-empty';
            empty.textContent = '撮影待ち...';
            container.appendChild(empty);
            return;
        }

        this._cameraBuffer.forEach((blob, i) => {
            const url   = URL.createObjectURL(blob);
            const thumb = document.createElement('div');
            thumb.className = 'camera-preview-thumb';

            const img = document.createElement('img');
            img.src = url;
            img.onload = () => URL.revokeObjectURL(url);

            const removeBtn = document.createElement('button');
            removeBtn.className = 'cam-remove-btn';
            removeBtn.textContent = '✕';
            removeBtn.type = 'button';
            const onRemove = (e) => {
                e.stopPropagation();
                e.preventDefault();
                this._cameraBuffer.splice(i, 1);
                this._renderPreviewImages();
            };
            removeBtn.addEventListener('click', onRemove);
            removeBtn.addEventListener('touchend', onRemove);

            thumb.appendChild(img);
            thumb.appendChild(removeBtn);
            container.appendChild(thumb);
        });
    }

    _updateButtonUI() {
        const btn = this._shadow.getElementById('cameraButton');
        if (!btn) return;
        btn.dataset.cameraState = this._cameraOn ? 'ON' : 'OFF';
        btn.title = this._cameraOn ? UI_STRINGS.camera.ON : UI_STRINGS.camera.OFF;
    }

    async _captureFrame() {
        if (!this._cameraStream) return null;
        const track = this._cameraStream.getVideoTracks()[0];
        if (!track) return null;

        if (typeof ImageCapture !== 'undefined') {
            try {
                const capture = new ImageCapture(track);
                const bitmap  = await capture.grabFrame();
                const canvas  = document.createElement('canvas');
                canvas.width  = bitmap.width;
                canvas.height = bitmap.height;
                canvas.getContext('2d').drawImage(bitmap, 0, 0);
                return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
            } catch { /* fallthrough */ }
        }

        const video = document.createElement('video');
        video.srcObject = this._cameraStream;
        video.muted = true;
        await video.play().catch(() => {});
        const canvas = document.createElement('canvas');
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;
        canvas.getContext('2d').drawImage(video, 0, 0);
        video.srcObject = null;
        return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    }

    isOn()       { return this._cameraOn; }
    getBuffer()  { return this._cameraBuffer; }
    clearBuffer(){ this._cameraBuffer = []; }
}
