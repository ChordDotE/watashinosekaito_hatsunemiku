/**
 * WebSocket制御モジュール（リファクタリング版）
 *
 * 親フレームからのpostMessageを受信して、Live2Dモデルの表情・モーション・口パクを制御する。
 *
 * リファクタリング後:
 * - WebSocket接続を削除（親フレームで一元管理）
 * - 音声再生を削除（親フレームで行う）
 * - postMessage受信のみで動作
 *
 * 配布用自作コード - CubismWebSamples/Samples/TypeScript/Demo/src/ に配置して使用
 */
/**
 * postMessageリスナーを開始
 */
export declare function startWsControl(): void;
