/**
 * main.ts への追加コード
 *
 * このファイルの内容を、CubismWebSamples の main.ts に追記してください。
 *
 * 対象ファイル: CubismWebSamples/Samples/TypeScript/Demo/src/main.ts
 *
 * 追加場所:
 * - import文はファイルの先頭付近に追加
 * - startWsControl() は LAppDelegate.getInstance().run() の後に追加
 */
// ============================================
// ↓↓↓ ファイル先頭の import 部分に追加 ↓↓↓
// ============================================
import { startWsControl } from './ws_control';
// ============================================
// ↓↓↓ window.addEventListener('load', ...) 内の
//      LAppDelegate.getInstance().run(); の後に追加 ↓↓↓
// ============================================
// WebSocket制御を開始（サーバーからの命令を受信）
startWsControl();
// ============================================
// 完成形の例（参考）
// ============================================
/*
import { LAppDelegate } from './lappdelegate';
import * as LAppDefine from './lappdefine';
import { startWsControl } from './ws_control';  // ← 追加

window.addEventListener(
  'load',
  (): void => {
    if (!LAppDelegate.getInstance().initialize()) {
      return;
    }

    LAppDelegate.getInstance().run();
    
    // WebSocket制御を開始
    startWsControl();  // ← 追加
  },
  { passive: true }
);
*/
//# sourceMappingURL=main_patch.js.map