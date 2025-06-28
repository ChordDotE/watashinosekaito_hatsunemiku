@echo off
chcp 932 >nul

REM ユーザー設定部分（ベースパスのみ変更してください）
set "RVC_PATH=ここにRVCのベースディレクトリのフルパスを記載(例 C:\MyProgram\vcclient_win_cuda_2.0.76-beta)"
set "VOICEVOX_PATH=ここにvoicevoxのディレクトリのフルパスを記載(例 C:\MyProgram\voicevox-windows-directml-0.22.4)"
set "MIKU_PATH=ここにmiku_agent-mainのディレクトリのフルパスを記載(例 C:\MyProgram\miku_agent-main)"

REM バッチ内で自動構築される実行パス
set "RVC_EXECUTE_PATH=%RVC_PATH%\dist\main"
set "VOICEVOX_EXECUTE_PATH=%VOICEVOX_PATH%\VOICEVOX\vv-engine"

REM バッチファイル起動ディレクトリを基準とした相対パス
set "MIKU_BASE=%MIKU_PATH%\watashinosekaito_hatsunemiku-main"
set "MIKU_SRC=%MIKU_BASE%\src"

REM RVC起動
cd "%RVC_EXECUTE_PATH%"
start "" "%RVC_EXECUTE_PATH%\start_http.bat"

REM voicevox起動
start "" "%VOICEVOX_EXECUTE_PATH%\run.exe"

REM 初音ミクの会話システム起動
cd "%MIKU_BASE%"
call venv\Scripts\activate.bat
cd "%MIKU_SRC%"
python app.py

REM ブラウザで会話システムを開く（Chromeの新しいウィンドウ）
REM start chrome "--new-window http://localhost:5001"
