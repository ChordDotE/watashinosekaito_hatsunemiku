/**
 * LAppDelegate モック
 * 
 * このファイルはエディタのTypeScriptエラーを抑制するためのモックです。
 * ビルド時はCubismWebSamples/Samples/TypeScript/Demo/src/の本物が使用されます。
 */

export class LAppDelegate {
    private static _instance: LAppDelegate;
    public _subdelegates: any;

    public static getInstance(): LAppDelegate {
        if (!this._instance) {
            this._instance = new LAppDelegate();
        }
        return this._instance;
    }
}
