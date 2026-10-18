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
 * gazeMath.js - カメラに写った顔から、ミクが向く方向を出す（純粋関数だけ）
 *
 * 副作用なし。カメラ・MediaPipe・Live2D には触らない（それは gazeTracker.js の役目）。
 * 起動不要の単体テスト tests/browser/gaze_math.test.mjs が回す。
 *
 * 目の位置の出し方は presence-lab の figure/facetrack.js（2026-09-27 に Galaxy S25 と
 * iPad Pro で動作確認済み）から移した:
 *   両目の瞳の間隔を 63mm（成人の平均）とみなし、写った間隔（画素）から距離を出す。
 *   左右・上下のずれは「写った位置 × 距離 ÷ 焦点距離」だが、距離も同じ焦点距離から
 *   出しているので、左右・上下の角度は画角の見積もりに依存しない。
 *
 * 向きの大きさは幾何で決め、手で合わせた係数を持たない:
 *   マスターの目がカメラから見て θ 度の方向にあれば、ミクの顔も θ 度そちらを向く。
 *   Live2D のドラッグ追従は「ドラッグ値 1.0 = ParamAngleX 30 度」
 *   （Cubism SDK の lappmodel.ts: ParamAngleX += dragX * 30）なので、θ / 30 を渡す。
 *   ※ カメラはミクの顔と同じ位置には無い（画面の上端など）。そのずれは端末ごとに違い
 *     取得できないので、補正しない（＝カメラの位置から見た方向を向く）。
 */

/** Live2D のドラッグ値 1.0 に当たる顔の角度（度）。Cubism SDK の lappmodel.ts の係数 */
export const DRAG_FULL_DEG = 30;

/** 瞳の間隔（m）。成人の平均（facetrack.js と同じ） */
export const IPD_M = 0.063;

/** インカメラの対角の画角（度）の既定。スマホ・PC のインカメラはおおむね 75〜85° */
export const DFOV_DEG = 78;

/**
 * Face Landmarker の 478 点から、両目の中点の位置（m）を出す。
 *
 * @param {{x:number,y:number,z:number}[]} L - 正規化座標の顔の点（MediaPipe の faceLandmarks[0]）
 * @param {number} W - 映像の幅（画素）
 * @param {number} H - 映像の高さ（画素）
 * @param {{dfovDeg?: number, ipd?: number}} [o]
 * @returns {{x:number,y:number,z:number}|null}
 *   x = マスターから見て右・y = 上・z = カメラからの距離。壊れた値・ありえない距離は null
 */
export function eyeMidFromLandmarks(L, W, H, { dfovDeg = DFOV_DEG, ipd = IPD_M } = {}) {
    if (!L || !W || !H) return null;
    const mid = (a, b) => (a && b) ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 } : null;
    // 468 / 473 = 瞳の中心（478 点のモデル）。無ければ目頭と目尻の中点
    const a = L[468] || mid(L[33], L[133]);
    const b = L[473] || mid(L[362], L[263]);
    if (!a || !b) return null;
    const dpx = Math.hypot((a.x - b.x) * W, (a.y - b.y) * H, (a.z - b.z) * W);
    if (!(dpx > 0)) return null;
    const f = (Math.hypot(W, H) / 2) / Math.tan((dfovDeg * Math.PI) / 360);   // 焦点距離（画素）
    const z = (f * ipd) / dpx;
    // インカメラの生の映像は鏡像ではない（向かい合った人から見た絵）。マスターの右は映像の左
    const u = (a.x + b.x) / 2, v = (a.y + b.y) / 2;
    const p = { x: (-(u * W - W / 2) * z) / f, y: (-(v * H - H / 2) * z) / f, z };
    // ★壊れた値は捨てる。GPU で `ROI contains NaN` が出た端末（S25）は、CPU に切り替えた後も
    //   数でない値を返すことがあった（facetrack.js 2026-09-27）
    const fin = Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z);
    return fin && p.z > 0.05 && p.z < 3 ? p : null;
}

/**
 * この映像でマスターの顔を追えるか（カメラ ON かつアウトカメラでない）。
 * gazeTracker（追うか）と MenuController（視線ボタンを押せる見た目にするか）が同じ判定を使う。
 *
 * @param {MediaStream|null} stream
 * @returns {boolean}
 */
export function canTrackFace(stream) {
    if (!stream) return false;
    const mode = stream.getVideoTracks?.()[0]?.getSettings?.().facingMode;
    return mode !== 'environment';   // アウトカメラ（背面）には自分の顔は写らない
}

/**
 * 目の位置（m）から、Live2D のドラッグ値（-1〜1）を出す。
 * x は画面の右が正、y は上が正（Live2D のドラッグ座標と同じ向き）。
 *
 * @param {{x:number,y:number,z:number}} p
 * @returns {{x:number,y:number}}
 */
export function gazeFromEye(p) {
    const clamp = (v) => Math.max(-1, Math.min(1, v));
    const deg = (a) => (a * 180) / Math.PI;
    return {
        x: clamp(deg(Math.atan2(p.x, p.z)) / DRAG_FULL_DEG),
        y: clamp(deg(Math.atan2(p.y, p.z)) / DRAG_FULL_DEG),
    };
}
