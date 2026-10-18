/**
 * WebSocket制御モジュール（リファクタリング版）
 * 
 * 親フレームからのpostMessageを受信して、Live2Dモデルの表情・モーション・口パク・視線を制御する。
 * 
 * リファクタリング後:
 * - WebSocket接続を削除（親フレームで一元管理）
 * - 音声再生を削除（親フレームで行う）
 * - postMessage受信のみで動作
 * 
 * 配布用自作コード - CubismWebSamples/Samples/TypeScript/Demo/src/ に配置して使用
 */

import { LAppDelegate } from './lappdelegate';
import * as LAppDefine from './lappdefine';

// 口パク値のキャッシュ
let currentLipSyncValue = 0;

// 姿勢固定対象モーション（group_index形式）
// これらのモーションは終了後にIdleに戻らず、最後の姿勢を維持する
// モーションコード対応: 01=Idle_0, 02=Tap_0, 03=Tap_1, 04=Idle_1, 05=Flick_0, 06=FlickUp_0, 07=Idle_2, 08=Flick_1
const HOLD_POSE_MOTIONS: string[] = [
    'Tap_0',      // 02: 怒り・不満
    'Tap_1',      // 03: 悲しみ
    'Flick_0',    // 05: 照れ
    'FlickUp_0',  // 06: 驚き
];

/**
 * 値を0-1の範囲にクランプ
 */
function clamp01(v: number): number {
    if (!Number.isFinite(v)) return 0;
    return Math.max(0, Math.min(1, v));
}

/**
 * モデルを取得する
 */
function getModel(): any | null {
    try {
        const delegate = LAppDelegate.getInstance() as any;
        // _subdelegates配列の最初のsubdelegateからLive2DManagerを取得
        if (delegate._subdelegates && delegate._subdelegates.getSize() > 0) {
            const subdelegate = delegate._subdelegates.at(0);
            const manager = subdelegate.getLive2DManager();
            if (manager._models && manager._models.getSize() > 0) {
                return manager._models.at(0);
            }
        }
    } catch (e) {
        console.warn('[WS] モデル取得エラー:', e);
    }
    return null;
}

/**
 * postMessageリスナーを開始
 */
export function startWsControl(): void {
    console.log('[Live2D] postMessageリスナーを開始');

    // 親フレームからのメッセージを受信
    window.addEventListener('message', handleMessage);

    // 準備完了を親に通知
    try {
        window.parent.postMessage({ type: 'live2d:ready' }, '*');
    } catch (e) {
        console.warn('[Live2D] 親フレーム通知エラー:', e);
    }
}

/**
 * postMessageを処理
 */
function handleMessage(event: MessageEvent): void {
    // データが無い場合は無視
    if (!event.data || typeof event.data !== 'object') {
        return;
    }

    const data = event.data;
    const msgType = data.type;

    switch (msgType) {
        case 'lipSync':
            handleLipSync(data.value);
            break;

        case 'expression':
            handleExpression(data.name);
            break;

        case 'motion':
            handleMotion(data.group, data.index, data.priority);
            break;

        case 'gaze':
            handleGaze(data.x, data.y);
            break;

        default:
            // その他のメッセージは無視（デバッグ時のみログ出力）
            // console.log('[Live2D] 未知のメッセージ:', data);
            break;
    }
}

/**
 * マスターの顔の方を向く（視線追従・2026-09-27）
 *
 * SDK のドラッグ追従と同じ入口（setDragging）に入れる。SDK 側で平滑化され
 * （CubismTargetPoint）、顔・体・目の向きとしてモーションの上に加算される。
 * x は画面の右・y は上が正で -1〜1（1.0 = ParamAngleX 30 度）。0,0 で正面。
 * ※ 画面をドラッグしている間はそちらが優先され、指を離すと正面に戻ったあと次の値で追従が戻る
 */
function handleGaze(x: number, y: number): void {
    const model = getModel();
    if (!model) return;
    const c = (v: number) => (Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0);
    model.setDragging(c(x), c(y));
}

/**
 * 表情を変更
 */
function handleExpression(name: string): void {
    if (!name) return;

    console.log('[Live2D] 表情変更:', name);

    const model = getModel();
    if (!model) {
        console.warn('[Live2D] モデルが取得できません');
        return;
    }

    try {
        model.setExpression(name);
        console.log('[Live2D] 表情設定完了:', name);
    } catch (e) {
        console.warn('[Live2D] 表情設定エラー:', e);
        try {
            model.setRandomExpression();
            console.log('[Live2D] ランダム表情を設定');
        } catch (e2) {
            console.warn('[Live2D] ランダム表情設定エラー:', e2);
        }
    }
}

/**
 * モーションを再生
 */
function handleMotion(group: string, index: number, priority?: string): void {
    if (!group) return;

    console.log('[Live2D] モーション再生:', group, index, priority);

    const model = getModel();
    if (!model) {
        console.warn('[Live2D] モデルが取得できません');
        return;
    }

    try {
        let priorityValue = LAppDefine.PriorityNormal;
        if (priority === 'force') {
            priorityValue = LAppDefine.PriorityForce;
        } else if (priority === 'idle') {
            priorityValue = LAppDefine.PriorityIdle;
        }

        // 姿勢固定判定
        const motionKey = `${group}_${index ?? 0}`;
        const shouldHoldPose = HOLD_POSE_MOTIONS.indexOf(motionKey) !== -1;
        model._holdPose = shouldHoldPose;
        console.log('[Live2D] 姿勢固定:', shouldHoldPose ? 'ON' : 'OFF', `(${motionKey})`);

        // モーション終了時のコールバック
        const onFinished = (): void => {
            console.log('[Live2D] モーション終了:', group, index);
        };

        model.startMotion(group, index ?? 0, priorityValue, onFinished);
        console.log('[Live2D] モーション開始:', group, index);
    } catch (e) {
        console.warn('[Live2D] モーション開始エラー:', e);
        try {
            model.startRandomMotion(group, LAppDefine.PriorityNormal);
            console.log('[Live2D] ランダムモーションを開始');
        } catch (e2) {
            console.warn('[Live2D] ランダムモーション開始エラー:', e2);
        }
    }
}

/**
 * 口パク値を設定（0.0-1.0）
 */
function handleLipSync(value: number): void {
    const clampedValue = clamp01(value);
    currentLipSyncValue = clampedValue;

    const model = getModel();
    if (!model) {
        return; // モデル取得失敗時は無視（頻繁に呼ばれるためログは出さない）
    }

    try {
        model._externalLipSyncValue = clampedValue;
    } catch (e) {
        // エラーは無視（頻繁に呼ばれるため）
    }
}
