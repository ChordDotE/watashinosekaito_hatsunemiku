/**
 * LAppDelegate モック
 *
 * このファイルはエディタのTypeScriptエラーを抑制するためのモックです。
 * ビルド時はCubismWebSamples/Samples/TypeScript/Demo/src/の本物が使用されます。
 */
export class LAppDelegate {
    static getInstance() {
        if (!this._instance) {
            this._instance = new LAppDelegate();
        }
        return this._instance;
    }
}
//# sourceMappingURL=lappdelegate.js.map