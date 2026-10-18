#
# Watashinosekaito Hatsune Miku
# Copyright (c) 2025-2026 ChordDotE. All rights reserved.
# SPDX-License-Identifier: LicenseRef-ChordDotE-NonCommercial-2.0
#
# NON-COMMERCIAL USE ONLY. NO BUSINESS USE. NO ADAPTATION.
#
# Business use (any use by or for a company, organization, or sole
# proprietor, WHETHER OR NOT ANY PAYMENT IS INVOLVED) and publishing or
# distributing modified versions are PROHIBITED IN ALL CASES; no permission
# is granted for them. Forks may be used only to propose improvements to the
# Author (e.g. pull requests). Commercial use, redistribution, and Adaptation
# are PROHIBITED without prior written permission. "Adaptation" includes
# manual reimplementation, automated translation, and generation using large
# language models or other machine learning, of software having equivalent or
# similar functionality. Permission is granted individually, in recorded form.
# Contact: https://x.com/ChordDotE
#
# 業務利用（対価の有無を問わず、法人・団体・個人事業主による利用と事業のための利用）と、
# 改変版の公開・配布は、許可の余地なく禁止します。フォークは作者への改善の提案
# （プルリクエスト等）のためだけに使えます。商用利用・再配布・翻案は、作者の許可が
# ない限り禁止します。翻案には、再実装・自動変換・大規模言語モデル等を用いた生成に
# よる、同等または類似の機能を持つソフトウェアの作成を含みます。
# 許可が必要な場合は、記録の残る形でご連絡ください。
#
# See LICENSE for full terms. 詳細は LICENSE を参照してください。
#

# Live2D Patch Script
# Usage: powershell -ExecutionPolicy Bypass -File apply_patches.ps1 -DemoSrc "path\to\src"

param(
    [Parameter(Mandatory=$true)]
    [string]$DemoSrc
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Patching main.ts..."
try {
    $mainFile = Join-Path $DemoSrc "main.ts"
    $content = Get-Content $mainFile -Raw -Encoding UTF8
    
    if ($content -notmatch 'startWsControl') {
        # Add import
        $content = $content -replace "(import \* as LAppDefine from './lappdefine';)", "`$1`r`nimport { startWsControl } from './ws_control';"
        
        # Add startWsControl()
        $content = $content -replace "(LAppDelegate\.getInstance\(\)\.run\(\);)", "`$1`r`n`r`n    // Start WebSocket control`r`n    startWsControl();"
        
        [System.IO.File]::WriteAllText($mainFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] main.ts patched"
    } else {
        Write-Host " [OK] main.ts already patched"
    }
} catch {
    Write-Host " [ERROR] main.ts: $_"
}

Write-Host ""
Write-Host "Renaming Miku folder to miku_sample_t04..."
try {
    $resourcesBase = Join-Path (Split-Path $DemoSrc -Parent) "..\..\Resources"
    $mikuFolder = Join-Path $resourcesBase "Miku"
    $targetFolder = Join-Path $resourcesBase "miku_sample_t04"
    
    if ((Test-Path $mikuFolder) -and !(Test-Path $targetFolder)) {
        Rename-Item -Path $mikuFolder -NewName "miku_sample_t04"
        Write-Host " [OK] Renamed Miku -> miku_sample_t04"
    } elseif (Test-Path $targetFolder) {
        Write-Host " [OK] miku_sample_t04 folder already exists"
    } else {
        Write-Host " [WARN] Miku folder not found"
    }
} catch {
    Write-Host " [ERROR] Rename: $_"
}

Write-Host ""
Write-Host "Installing custom motions (React group)..."
try {
    # 自作モーション（リポジトリ管理: miku/src/live2d/motions/。制作は dev_tools/motion_lab、
    # 人間承認後にここへ確定コピーされたものが本番マスター）を公式モデルに注入する。
    # マスター（Samples/Resources）にだけ入れれば、ビルド時に copy_resources.js が
    # public/ へ、vite build が dist/ へ複製する。
    # ※新モーション追加時はこの配列（順序 = MOTION_MAP の React index）にも追記すること
    $reactMotions = @(
        'miku_nod', 'miku_tilt', 'miku_empathy', 'miku_giggle', 'miku_wink',
        'miku_celebrate', 'miku_iyaiya', 'miku_yurayura', 'miku_pyokon',
        'miku_kurutto', 'miku_blink_surprise',
        'miku_jitome', 'miku_kangae', 'miku_tameiki', 'miku_nigawarai',
        'miku_hikiwarai', 'miku_wakuwaku', 'miku_doya',
        # 27: 歌っているときの立ち姿（ループ）。**相づちではなく歌唱専用**なので
        #     schemas.py の MotionCode には入れない（LLM には選ばせない）
        'miku_sing_sway'
    )
    $resourcesBase = Join-Path (Split-Path $DemoSrc -Parent) "..\..\Resources"
    $modelDir = Join-Path $resourcesBase "miku_sample_t04"
    $approvedDir = Join-Path $PSScriptRoot "motions"

    if ((Test-Path $modelDir) -and (Test-Path $approvedDir)) {
        # motion3.json をコピー
        $copied = 0
        foreach ($name in $reactMotions) {
            $src = Join-Path $approvedDir "$name.motion3.json"
            if (Test-Path $src) {
                Copy-Item $src (Join-Path $modelDir "motion") -Force
                $copied++
            } else {
                Write-Host " [WARN] motions に $name.motion3.json がありません"
            }
        }

        # model3.json に React グループを登録（毎回リストから再構築 = 冪等）
        $modelJsonPath = Join-Path $modelDir "miku_sample_t04.model3.json"
        $model = Get-Content $modelJsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $reactEntries = @($reactMotions | Where-Object {
            Test-Path (Join-Path $modelDir "motion\$_.motion3.json")
        } | ForEach-Object { [pscustomobject]@{ File = "motion/$_.motion3.json" } })
        if ($model.FileReferences.Motions.PSObject.Properties.Name -contains 'React') {
            $model.FileReferences.Motions.React = $reactEntries
        } else {
            $model.FileReferences.Motions | Add-Member -NotePropertyName 'React' -NotePropertyValue $reactEntries
        }
        $json = $model | ConvertTo-Json -Depth 10
        [System.IO.File]::WriteAllText($modelJsonPath, $json, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] custom motions installed ($copied files, React group = $($reactEntries.Count) entries)"
    } elseif (-not (Test-Path $approvedDir)) {
        Write-Host " [WARN] live2d\motions が見つかりません（自作モーションはスキップ）"
    } else {
        Write-Host " [SKIP] miku_sample_t04 フォルダ未配置（モデル配置後に再実行してください）"
    }
} catch {
    Write-Host " [ERROR] custom motions: $_"
}

Write-Host ""
Write-Host "Patching lappdefine.ts..."
try {
    $defineFile = Join-Path $DemoSrc "lappdefine.ts"
    $content = Get-Content $defineFile -Raw -Encoding UTF8
    
    # Fixed model name for Miku sample
    $modelName = "miku_sample_t04"
    
    # Check if needs patching
    if ($content -notmatch "'$modelName'") {
        # Change ModelDir to miku_sample_t04
        $content = $content -replace "export const ModelDir: string\[\] = \[[^\]]+\];", "export const ModelDir: string[] = ['$modelName'];"
        
        [System.IO.File]::WriteAllText($defineFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] lappdefine.ts patched (ModelDir = '$modelName')"
    } else {
        Write-Host " [OK] lappdefine.ts already patched"
    }
} catch {
    Write-Host " [ERROR] lappdefine.ts: $_"
}

Write-Host ""
Write-Host "Patching lappmodel.ts..."
try {
    $modelFile = Join-Path $DemoSrc "lappmodel.ts"
    $content = Get-Content $modelFile -Raw -Encoding UTF8
    
    if ($content -notmatch '_externalLipSyncValue') {
        # Add property
        $content = $content -replace "(export class LAppModel extends CubismUserModel \{)", "`$1`r`n  _externalLipSyncValue: number = 0; // External lip sync value"
        
        # Change condition
        $content = $content -replace "(if \(this\._lipsync\) \{)", "if (this._lipsync || this._externalLipSyncValue > 0) {"
        
        # Change lip sync value processing
        $content = $content -replace "(this\._wavFileHandler\.update\(deltaTimeSeconds\);\s*\r?\n\s*value = this\._wavFileHandler\.getRms\(\);)", "if (this._externalLipSyncValue > 0) {`r`n        value = this._externalLipSyncValue;`r`n      } else {`r`n        `$1`r`n      }"
        
        [System.IO.File]::WriteAllText($modelFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] lappmodel.ts patched"
    } else {
        Write-Host " [OK] lappmodel.ts already patched"
    }

    # ★口の開きの上限を外す（2026-09-21・本人が「開き具合が小さい」と指摘）
    #   SDK 既定は `addParameterValueById(id, value, 0.8)` ＝ **重み 0.8 で加算**なので、
    #   lipsync が 1.0 でも口は 0.80 までしか開かない。1 にして頭打ちを外す。
    #   ※この指定とは別に songPlayer.js の LIPSYNC_GAMMA（0.5）が中間を持ち上げている。
    #     **両方そろって初めて大きく開く**——重みが 0.8 のままだとガンマを下げても上が伸びない。
    $content = Get-Content $modelFile -Raw -Encoding UTF8
    if ($content -match 'this\._lipSyncIds\.at\(i\), value, 0\.8') {
        $content = $content -replace "(this\._lipSyncIds\.at\(i\), value, )0\.8", "`${1}1.0"
        [System.IO.File]::WriteAllText($modelFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] lip sync weight 0.8 -> 1.0"
    } else {
        Write-Host " [OK] lip sync weight already 1.0"
    }
} catch {
    Write-Host " [ERROR] lappmodel.ts: $_"
}

Write-Host ""
Write-Host "Patching lappglmanager.ts (enable transparent background)..."
try {
    $glManagerFile = Join-Path $DemoSrc "lappglmanager.ts"
    $content = Get-Content $glManagerFile -Raw -Encoding UTF8
    
    if ($content -notmatch 'alpha: true') {
        # Add alpha option to WebGL context
        $content = $content -replace "canvas\.getContext\('webgl2'\)", "canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false }) // PATCHED: transparent background"
        
        [System.IO.File]::WriteAllText($glManagerFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] lappglmanager.ts patched (alpha enabled)"
    } else {
        Write-Host " [OK] lappglmanager.ts already patched"
    }
} catch {
    Write-Host " [ERROR] lappglmanager.ts: $_"
}

Write-Host ""
Write-Host "Patching lappsubdelegate.ts (transparent clear color)..."
try {
    $subdelegateFile = Join-Path $DemoSrc "lappsubdelegate.ts"
    $content = Get-Content $subdelegateFile -Raw -Encoding UTF8
    
    if ($content -notmatch 'clearColor\(0\.0, 0\.0, 0\.0, 0\.0\)') {
        # Change clearColor alpha to 0 (transparent)
        $content = $content -replace "gl\.clearColor\(0\.0, 0\.0, 0\.0, 1\.0\)", "gl.clearColor(0.0, 0.0, 0.0, 0.0) // PATCHED: transparent"
        
        [System.IO.File]::WriteAllText($subdelegateFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] lappsubdelegate.ts patched (transparent clear color)"
    } else {
        Write-Host " [OK] lappsubdelegate.ts already patched"
    }
} catch {
    Write-Host " [ERROR] lappsubdelegate.ts: $_"
}

Write-Host ""
Write-Host "Patching lappview.ts (disable background and gear rendering)..."
try {
    $viewFile = Join-Path $DemoSrc "lappview.ts"
    $content = Get-Content $viewFile -Raw -Encoding UTF8
    
    if ($content -notmatch '// PATCHED: disable background') {
        # 1. render() - disable background rendering
        $content = $content -replace "if \(this\._back\) \{", "// PATCHED: disable background`r`n    if (false && this._back) {"
        
        # 2. render() - disable gear rendering
        $content = $content -replace "if \(this\._gear\) \{", "// PATCHED: disable gear`r`n    if (false && this._gear) {"
        
        # 3. onTouchesEnded() - disable gear click
        $content = $content -replace "if \(this\._gear\.isHit\(posX, posY\)\) \{", "// PATCHED: disable gear click`r`n    if (false && this._gear.isHit(posX, posY)) {"
        
        [System.IO.File]::WriteAllText($viewFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] lappview.ts patched (background and gear disabled)"
    } else {
        Write-Host " [OK] lappview.ts already patched"
    }
} catch {
    Write-Host " [ERROR] lappview.ts: $_"
}

Write-Host ""
Write-Host "Patching lappmodel.ts (hold pose feature)..."
try {
    $modelFile = Join-Path $DemoSrc "lappmodel.ts"
    $content = Get-Content $modelFile -Raw -Encoding UTF8
    
    # Check if hold pose check is already applied (not just property exists)
    if ($content -notmatch 'PATCHED: hold pose check') {
        # Add _holdPose property if not exists
        if ($content -notmatch '_holdPose') {
            $content = $content -replace "(_externalLipSyncValue: number = 0;[^\r\n]*)", "`$1`r`n  _holdPose: boolean = false; // Hold pose after motion ends"
        }
        
        # Replace startRandomMotion call with hold pose check
        # Match: this.startRandomMotion(\n      LAppDefine.MotionGroupIdle,\n      LAppDefine.PriorityIdle\n    );
        $content = $content -replace "(this\.startRandomMotion\(\s*\r?\n\s*LAppDefine\.MotionGroupIdle,\s*\r?\n\s*LAppDefine\.PriorityIdle\s*\r?\n\s*\);)", "// PATCHED: hold pose check`r`n      if (!this._holdPose) {`r`n        this.startRandomMotion(`r`n          LAppDefine.MotionGroupIdle,`r`n          LAppDefine.PriorityIdle`r`n        );`r`n      }"
        
        [System.IO.File]::WriteAllText($modelFile, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host " [OK] lappmodel.ts patched (hold pose feature)"
    } else {
        Write-Host " [OK] lappmodel.ts already has hold pose feature"
    }
} catch {
    Write-Host " [ERROR] lappmodel.ts hold pose: $_"
}

Write-Host ""
Write-Host "Patching lapplive2dmanager.ts (contain fit)..."
try {
    $managerFile = Join-Path $DemoSrc "lapplive2dmanager.ts"
    $content = Get-Content $managerFile -Raw -Encoding UTF8

    if ($content -notmatch 'PATCHED: contain fit') {
        # 旧実装: 縦長画面では幅基準固定 → キーボード・添付でアバター領域の縦横比が
        # モデルより低くなると縦がはみ出して足元が切れる。制約側の軸でフィットする
        # contain 方式に置き換える（全身が常に収まる）。
        $pattern = "if \(model\.getModel\(\)\.getCanvasWidth\(\) > 1\.0 && width < height\) \{[\s\S]*?projection\.scale\(height / width, 1\.0\);\s*\r?\n\s*\}"
        $replacement = @'
// PATCHED: contain fit - 画面の縦横比によらず常に全身が収まる軸でフィットさせる
      const canvasAspect: number = height / width;
      const modelAspect: number =
        model.getModel().getCanvasHeight() / model.getModel().getCanvasWidth();
      model.getModelMatrix().setWidth(2.0);
      if (canvasAspect >= modelAspect) {
        // 縦に余裕がある → 幅基準
        projection.scale(1.0, width / height);
      } else {
        // 縦が制約 → 高さ基準に縮めて全身（足元まで）を収める
        projection.scale(canvasAspect / modelAspect, 1.0 / modelAspect);
      }
'@
        # 置換文字列に $ を含まないため単純置換で安全（PS5.1互換のため MatchEvaluator は使わない）
        $new = [regex]::Replace($content, $pattern, $replacement)
        if ($new -eq $content) {
            Write-Host " [WARN] lapplive2dmanager.ts: パッチ対象のコードが見つかりません（SDKバージョン差異の可能性）"
        } else {
            [System.IO.File]::WriteAllText($managerFile, $new, [System.Text.UTF8Encoding]::new($false))
            Write-Host " [OK] lapplive2dmanager.ts patched (contain fit)"
        }
    } else {
        Write-Host " [OK] lapplive2dmanager.ts already patched"
    }
} catch {
    Write-Host " [ERROR] lapplive2dmanager.ts: $_"
}

Write-Host ""
Write-Host "Patch completed"
