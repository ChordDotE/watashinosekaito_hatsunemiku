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
 * gazeTracker.js - カメラに写ったマスターの顔を追い、ミクが向く方向を知らせる
 *
 * 動くのは「視線追従が ON」かつ「カメラが ON（インカメラ）」のときだけ。
 *   - ON/OFF は表示メニューの視線ボタン（MenuController）が EVENTS.GAZE_SET_ENABLED で知らせる
 *   - 映像は CameraManager が開いたストリームを EVENTS.CAMERA_STREAM_CHANGE で受け取って共有する
 *     （カメラを二重に掴むと、スマホで取り合いになるため自分では開かない）
 * 向く方向は EVENTS.GAZE_TARGET で出し、Live2D へは miku-avatar-live2d が中継する。
 *
 * ★映像は端末の外へ出さない（この端末の中で顔の位置を出して捨てる）。
 *   LLM へ画像を渡すのは従来どおり CameraManager の1秒1枚のサンプリングだけ。
 *
 * 顔の検出は MediaPipe Face Landmarker（vendor/mediapipe・初回 ON 時に読み込む）。
 * 追跡の回し方は presence-lab の figure/facetrack.js から移した:
 *   ★GPU で落ちる端末がある（Galaxy S25 で `ROI contains NaN values` の例外）。例外を受けたら
 *     CPU に切り替えて続ける。GPU が 6 秒たって 8 回/秒に届かないときも CPU へ
 */

import { EVENTS } from './events.js';
import { eyeMidFromLandmarks, gazeFromEye, canTrackFace } from './gazeMath.js';

const VENDOR = new URL('../vendor/mediapipe/', import.meta.url).href;

/** 顔を見失ってから正面へ戻すまで（ms）。瞬きや一瞬のうつむきで戻らないように */
const LOST_MS = 1500;

let enabled = false;
/** @type {MediaStream|null} */
let stream = null;
let running = false;
/** @type {any} */ let landmarker = null;
/** @type {Promise<any>|null} */ let loading = null;
let backend = 'GPU';
/** @type {HTMLVideoElement|null} */ let video = null;
let loopId = 0;

function dispatchGaze(x, y) {
    document.dispatchEvent(new CustomEvent(EVENTS.GAZE_TARGET, { detail: { x, y } }));
}

async function makeLandmarker(delegate) {
    const { FilesetResolver, FaceLandmarker } = await import(`${VENDOR}vision_bundle.mjs`);
    const files = await FilesetResolver.forVisionTasks(`${VENDOR}wasm`);
    return FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: `${VENDOR}face_landmarker.task`, delegate },
        runningMode: 'VIDEO', numFaces: 1,
    });
}

async function ensureLandmarker() {
    if (landmarker) return landmarker;
    if (!loading) {
        loading = (async () => {
            try { landmarker = await makeLandmarker('GPU'); backend = 'GPU'; }
            catch { landmarker = await makeLandmarker('CPU'); backend = 'CPU'; }
            console.log(`[Gaze] 顔の検出を読み込んだ（${backend}）`);
            return landmarker;
        })().catch((err) => {
            console.warn('[Gaze] 顔の検出を読み込めなかった:', err);
            window.__mikuDebugToast?.('👀 視線追従を開始できませんでした');
            loading = null;
            return null;
        });
    }
    return loading;
}

async function toCpu(why) {
    if (backend === 'CPU') return;
    backend = 'CPU';
    try { landmarker?.close(); } catch { /* 閉じられなくても続ける */ }
    landmarker = null;
    landmarker = await makeLandmarker('CPU');
    console.log(`[Gaze] 顔の検出を CPU に切り替えた（${why}）`);
}

async function start() {
    if (running) return;
    running = true;
    const myId = ++loopId;
    const lm = await ensureLandmarker();
    if (!lm || myId !== loopId || !running) return;

    if (!video) {
        video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        // 画面には出さないが、端末によっては DOM に無い video を再生しないので置いておく
        Object.assign(video.style, { position: 'fixed', width: '1px', height: '1px', opacity: '0', pointerEvents: 'none', left: '0', top: '0' });
        document.body.appendChild(video);
    }
    video.srcObject = stream;
    await video.play().catch(() => {});

    let lastT = -1, lastSeen = performance.now(), atFront = true;
    let n = 0, n0 = 0, t0 = performance.now(), rate = 0;
    const tStart = performance.now();

    const step = () => {
        if (myId !== loopId || !running) return;
        const now = performance.now();
        // 30 回/秒まで（Live2D 側で平滑化するので、それ以上は負荷を増やすだけ）
        if (landmarker && video.readyState >= 2 && now - lastT >= 33) {
            lastT = now;
            let res = null;
            try { res = landmarker.detectForVideo(video, now); }
            catch (e) { toCpu(`例外: ${String(e?.message || e).slice(0, 120)}`); }
            const L = res?.faceLandmarks?.[0];
            const p = L ? eyeMidFromLandmarks(L, video.videoWidth, video.videoHeight) : null;
            if (p) {
                const g = gazeFromEye(p);
                dispatchGaze(g.x, g.y);
                lastSeen = now; atFront = false; n++;
            } else if (!atFront && now - lastSeen > LOST_MS) {
                dispatchGaze(0, 0);
                atFront = true;
            }
            if (now - t0 > 2000) { rate = ((n - n0) * 1000) / (now - t0); n0 = n; t0 = now; }
            if (backend === 'GPU' && now - tStart > 6000 && rate > 0 && rate < 8) toCpu(`GPU が遅い（${rate.toFixed(1)} 回/秒）`);
        }
        if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(step);
        else requestAnimationFrame(step);
    };
    step();
    console.log('[Gaze] 追従開始');
}

function stop() {
    if (!running) return;
    running = false;
    loopId++;
    if (video) { video.pause(); video.srcObject = null; }
    dispatchGaze(0, 0);
    console.log('[Gaze] 追従停止');
}

function sync() {
    if (enabled && canTrackFace(stream)) {
        if (running && video && video.srcObject !== stream) { stop(); }
        start();
    } else {
        stop();
    }
}

// 読み込まれた時点で待ち受ける（main.js が components より先に import する）。
// ★init 関数にしない: MenuController は <miku-chat> の connectedCallback で作られ、
//   そこで最初の GAZE_SET_ENABLED を出す。main.js の initApp（DOMContentLoaded）では遅い
document.addEventListener(EVENTS.GAZE_SET_ENABLED, (e) => {
    enabled = !!e.detail?.enabled;
    sync();
});
document.addEventListener(EVENTS.CAMERA_STREAM_CHANGE, (e) => {
    stream = e.detail?.stream || null;
    sync();
});
