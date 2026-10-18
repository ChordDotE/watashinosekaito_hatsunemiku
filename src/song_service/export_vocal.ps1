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

# VOCALOID6 に歌声を書き出させる（人が開いて操作する工程を機械に渡す）。
#
#   powershell -ExecutionPolicy Bypass -File export_vocal.ps1 -Vpr <入力.vpr> -Out <出力.wav>
#
# 呼び出し元は engine/v6_export.py。設計と経緯は docs/components/song_service.md、
# 判断は docs/dev/decisions.md 2026-09-05。
#
# 【方針】**実行時に LLM を挟まない・座標を使わない**
#   V6 には CLI も公開 API も無い（6.13 まで確認）。UI Automation で部品を名指しする。
#   ミックスダウンのダイアログは AutomationId 付きなので、表示文言が変わっても効く:
#     xAudioMixdownDlg / xRBMasterTrack / xOpenExplorerCheckBox / xOKButton
#
# 【踏んだ落とし穴】——同じ所で悩まないように残す（2026-09-05）
#   ・V6 は**単一インスタンス**。起動済みだと、こちらが起こしたプロセスは既存側へ
#     ファイルを渡して終了する＝PID を追っても窓が無い。**先に残りが無いことを確かめる**
#   ・UIA の ControlView では**要素が 0 個に見える**（名前を持たない WPF 要素が落ちる）。
#     **RawView を使う**
#   ・WPF のメニューのポップアップは**別ウィンドウ**なので、本体窓の子孫を FindFirst しても出ない
#   ・V6 のダイアログは**別窓ではなく本体窓の中**に描かれる
#   ・保存ダイアログのファイル名欄は **UIA の ValuePattern を持たない**。窓メッセージで直に入れる
#   ・`Ctrl+E` は届かない（別プロセスからの前面化は Windows に阻まれる）
#
# 【保存形式】**BOM 付き UTF-8 で保存すること**。BOM が無いと Windows PowerShell 5.1 が
#   cp932 として読み、日本語コメントが化けて構文エラーになる。

param(
    [string]$Vpr = "",           # -CheckOnly のときは要らない
    [string]$Out = "",           # 同上
    [switch]$CheckOnly,          # **開いて確かめて閉じるだけ**（導入時の動作確認）
    [int]$UiReadySec = 120,      # UI が組み上がるまで
    [int]$RenderSec  = 900,      # 書き出しが終わるまで
    [string]$Exe = "",           # VOCALOID6.exe の実体（呼び出し元が見つけたものを渡す）
    [switch]$KeepOpen
)
$ErrorActionPreference = "Stop"
# ★**標準出力を UTF-8 で出す**（2026-09-06）。既定では PowerShell 5.1 が cp932 で書くので、
#   呼び出し元（`v6_export.export`）が UTF-8 で読むと日本語が全部 U+FFFD になり、
#   **失敗の理由が読めない**まま握り潰される。実際にこれでエラー本文を失った。
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Fg {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  // 保存ダイアログは素の Win32。UIA の ValuePattern が無いので**窓メッセージで直に**入れる
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr FindWindowEx(IntPtr p, IntPtr c, string cls, string win);
  [DllImport("user32.dll")] public static extern IntPtr GetDlgItem(IntPtr dlg, int id);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessageW(IntPtr h, uint msg, IntPtr wp, string lp);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, uint msg, IntPtr wp, IntPtr lp);
  [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr h, EnumProc cb, IntPtr p);
  public delegate bool EnumProc(IntPtr h, IntPtr p);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassNameW(IntPtr h, System.Text.StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);

  // いまの保存ダイアログ（Explorer 版）は ComboBoxEx32 がダイアログの**直下に無い**ので、
  // FindWindowEx の1段検索では届かない。**子孫を全部たどって、見えている Edit** を拾う。
  public static IntPtr FindVisibleEdit(IntPtr dlg) {
    IntPtr found = IntPtr.Zero;
    EnumChildWindows(dlg, delegate(IntPtr h, IntPtr p) {
      var sb = new System.Text.StringBuilder(256);
      GetClassNameW(h, sb, 256);
      if (sb.ToString() == "Edit" && IsWindowVisible(h)) { found = h; return false; }
      return true;
    }, IntPtr.Zero);
    return found;
  }

  // ---- 起動時のダイアログ（「バックアップから復元しますか？」など）を Win32 で払う ----
  // ★**UIA では押せない**（2026-09-06 実測）。この手のダイアログは UIA から見ると
  //   本体窓の**子**で、しかもボタンが `ControlType.Pane` に見える（Invoke を持たない）。
  //   一方 Win32 から見ると **class `#32770` のトップレベル窓**で、中身は素の `Button`。
  //   保存ダイアログと同じく**窓メッセージで直に押す**のが確実。
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowTextW(IntPtr h, System.Text.StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern int GetWindowThreadProcessId(IntPtr h, out int pid);
  static string Cls(IntPtr h){ var s = new System.Text.StringBuilder(256); GetClassNameW(h, s, 256); return s.ToString(); }
  static string Txt(IntPtr h){ var s = new System.Text.StringBuilder(512); GetWindowTextW(h, s, 512); return s.ToString(); }
  public static string LastMessage = "";

  /// **指定プロセスの** #32770 ダイアログだけを探し、「いいえ」を押す。
  /// ★pid で必ず絞る——絞らないと**他のアプリのダイアログまで勝手に押してしまう**。
  public static bool DismissNo(int pid) {
    bool hit = false;
    LastMessage = "";
    EnumWindows(delegate(IntPtr top, IntPtr p) {
      int q; GetWindowThreadProcessId(top, out q);
      if (q != pid) return true;
      if (Cls(top) != "#32770" || !IsWindowVisible(top)) return true;
      IntPtr btn = IntPtr.Zero;
      string msg = "";
      EnumChildWindows(top, delegate(IntPtr c, IntPtr p2) {
        string cls = Cls(c), t = Txt(c);
        if (cls == "Static" && t.Length > msg.Length) msg = t;
        // 「保存しない」を先に見る——保存確認は【保存する／保存しない／キャンセル】で、
        // 「いいえ」が無い代わりにこれが「進めて、かつ曲を書き換えない」選択肢になる。
        if (cls == "Button" && t.Contains("保存しない")) btn = c;
        if (cls == "Button" && btn == IntPtr.Zero &&
            (t.Contains("いいえ") || t.Contains("No"))) btn = c;
        return true;
      }, IntPtr.Zero);
      if (btn != IntPtr.Zero) {
        SendMessage(btn, 0x00F5 /* BM_CLICK */, IntPtr.Zero, IntPtr.Zero);
        LastMessage = msg;
        hit = true;
      }
      return true;
    }, IntPtr.Zero);
    return hit;
  }
}
"@

$AE = [System.Windows.Automation.AutomationElement]
$TS = [System.Windows.Automation.TreeScope]
$walker = [System.Windows.Automation.TreeWalker]::RawViewWalker
# ★**起動するパスは、存在を確かめたパスと同じものを使う**（2026-09-06）。
#   ここを決め打ちにしていたので、既定以外の場所に入れた人は「判定は通るのに書き出しで失敗」
#   という分かりにくい壊れ方をするところだった。呼び出し元（`v6_export.export`）が
#   `find_v6_exe()` で見つけた実物を -Exe で渡す。
#   単体で叩いたときのために、渡されなければ **`.vpr` の関連付け**（＝Windows がこの拡張子を
#   開くときに実際に使うコマンド）から引く。「確認」と「起動」が同じ1本になる。
$exe = $Exe
if (-not $exe) {
    $assoc = "HKLM:\SOFTWARE\Classes\VOCALOID6.vpr\shell\open\command"
    if (Test-Path $assoc) {
        $cmd = (Get-ItemProperty $assoc -EA SilentlyContinue).'(default)'
        if ($cmd -and $cmd -match '"([^"]+)"' -and (Test-Path $Matches[1])) { $exe = $Matches[1] }
    }
}
if (-not $exe) { $exe = Join-Path $env:ProgramFiles "VOCALOID6\Editor\VOCALOID6.exe" }

if (-not $CheckOnly) {
    if (-not $Vpr -or -not $Out) { throw "-Vpr と -Out が要る（確認だけなら -CheckOnly）" }
    $Vpr = (Resolve-Path $Vpr).Path
    $outDir = Split-Path $Out -Parent
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }
    $Out = Join-Path (Resolve-Path $outDir).Path (Split-Path $Out -Leaf)
    if (Test-Path $Out) { Remove-Item $Out -Force }
}

function Log($m) { Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $m) }

# ---- 探索の道具（すべて RawView。名前でも AutomationId でも引ける）----
# ★UI が組み上がる**途中**の要素を掴むと `GetFirstChild` が「認識できないエラー」を投げる
#   （要素が作り直されて手元の参照が古くなるため）。**見つからなかった扱いにして待ち直す**のが正しい
#   ——ここで止めると、起動が少し遅いだけで失敗になる。
# ★`$type` を指定できる（2026-09-06）。**名前だけで探すと同名の Text を掴む**
#   ——メニュー項目の中には同じ文字列の TextBlock がいて、そちらは `Invoke` も
#   `ExpandCollapse` も持たないので「Unsupported Pattern」で落ちる。
#   押したい相手は必ず MenuItem なので、種類まで指定して曖昧さを消す。
function Find($el, [string]$name, [string]$id, [int]$depth = 0, [int]$max = 10, [string]$type = "") {
    if ($depth -gt $max -or -not $el) { return $null }
    try {
        $c = $walker.GetFirstChild($el)
        while ($c) {
            $cur = $c.Current
            $okType = (-not $type) -or ($cur.ControlType.ProgrammaticName -match $type)
            if ($okType -and (($id -and $cur.AutomationId -eq $id) -or ($name -and $cur.Name -eq $name))) { return $c }
            $r = Find $c $name $id ($depth + 1) $max $type
            if ($r) { return $r }
            $c = $walker.GetNextSibling($c)
        }
    } catch { return $null }
    return $null
}
# ★自前の再帰探索は**1回に4秒**かかる（ツリーが1790要素あるため）。
#   UIA の `FindFirst` はネイティブ側で探すので桁違いに速い。**待ちループではこちらを使う**
#   （自前の Find は、条件に合う要素が複数ある場面の絞り込み用に残す）。
function ById($el, [string]$id) {
    try {
        $el.FindFirst($TS::Descendants,
            (New-Object System.Windows.Automation.PropertyCondition($AE::AutomationIdProperty, $id)))
    } catch { $null }
}
function ByName($el, [string]$name) {
    try {
        $el.FindFirst($TS::Descendants,
            (New-Object System.Windows.Automation.PropertyCondition($AE::NameProperty, $name)))
    } catch { $null }
}
function WaitFor($el, [string]$name, [string]$id, [int]$sec, [string]$what) {
    for ($i = 0; $i -lt $sec * 2; $i++) {
        $f = if ($id) { ById $el $id } else { ByName $el $name }
        if ($f) { return $f }
        Start-Sleep -Milliseconds 500
    }
    throw "見つからない: $what（$sec 秒待った）"
}
# ★`FindFirst` で辿れないものがある。**WPF のメニューのポップアップは別ウィンドウ**なので、
#   本体窓の子孫を native 検索しても出てこない（実際 15 秒待って見つからなかった）。
#   その手のものは **Raw ビューの再帰**で、対象の小さな部分木だけを探す。
function WaitDeep($el, [string]$name, [string]$id, [int]$sec, [string]$what) {
    for ($i = 0; $i -lt $sec; $i++) {
        $f = Find $el $name $id
        if ($f) { return $f }
        Start-Sleep -Milliseconds 500
    }
    throw "見つからない: $what（Raw で $sec 回探した）"
}
function DumpTree($el, [string]$path) {
    $lines = New-Object System.Collections.Generic.List[string]
    function D($e, $d) {
        if ($d -gt 6) { return }
        $c = $walker.GetFirstChild($e)
        while ($c) {
            $cur = $c.Current
            $t = $cur.ControlType.ProgrammaticName -replace "ControlType\.", ""
            $lines.Add("$("  " * $d)$t '$($cur.Name)' id=$($cur.AutomationId)")
            D $c ($d + 1)
            $c = $walker.GetNextSibling($c)
        }
    }
    D $el 0
    $lines | Set-Content $path -Encoding UTF8
    Log "  → 手がかりを $path に書いた（$($lines.Count) 行）"
}
function Push($el, [string]$label) {
    $el.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Log "  押した: $label"
}

# ---------------------------------------------------------------- 1. 起動
# ★V6 は**単一インスタンス**。既に起動していると、こちらが起こしたプロセスは
#   ファイルを既存側へ渡して**すぐ終了する**ので、そのPIDを追いかけても窓が1つも見つからない
#   （最初の失敗で V6 が閉じ残り、次の実行が丸ごと空振りした。2026-09-05）。
#   なので ①先に残りが無いことを確かめ ②窓はPIDでなく**プロセス名**で探す。
$already = Get-Process VOCALOID6 -ErrorAction SilentlyContinue
if ($already) {
    throw ("VOCALOID6 が既に起動している（PID $($already.Id -join ', ')）。" +
           "単一インスタンスなので自動操作と衝突する。閉じてから実行してください")
}

if ($CheckOnly) {
    Log "動作確認のため VOCALOID6 を起動します（曲は作りません）"
    $proc = Start-Process $exe -PassThru
} else {
    Log "起動: $(Split-Path $Vpr -Leaf)"
    $proc = Start-Process $exe -ArgumentList "`"$Vpr`"" -PassThru
}
$root = $AE::RootElement
function Tops {
    $ps = Get-Process VOCALOID6 -ErrorAction SilentlyContinue
    if (-not $ps) { return @() }
    $out = New-Object System.Collections.Generic.List[object]
    foreach ($p in $ps) {
        $c = New-Object System.Windows.Automation.PropertyCondition($AE::ProcessIdProperty, $p.Id)
        foreach ($e in $root.FindAll($TS::Children, $c)) { $out.Add($e) }
    }
    return $out
}

# ★**起動直後に出るダイアログを閉じる**（2026-09-06）。
#   前回 V6 が異常終了していると、次の起動で
#   「バックアップファイルが見つかりました。バックアップファイルから復元しますか？」
#   が出る。これが出ている間は**メニュー項目が押せない**（`IsEnabled=False`）ので、
#   待っても永久に進まない。60 秒＋120 秒空振りしていた原因がこれだった。
#
#   答えは常に**いいえ**——書き出す対象はこちらが作った vpr そのもので、
#   V6 の作業用バックアップに戻されると**別の内容を書き出してしまう**。
#   同じ理由で「保存しますか」の類も「いいえ」でよい。
function DismissDialogs {
    $hit = $false
    try {
        foreach ($p in @(Get-Process VOCALOID6 -ErrorAction SilentlyContinue)) {
            if ([Fg]::DismissNo($p.Id)) {
                Log "  ダイアログを閉じた（いいえ）: $([Fg]::LastMessage)"
                $hit = $true
            }
        }
        if ($hit) { Start-Sleep -Milliseconds 800 }
    } catch { }
    return $hit
}

# ★起動した直後から try に入れる——**UI が立ち上がらなかったときも finally で閉じる**ため。
#   ここを外に置いていたせいで、最初の失敗が V6 を1つ残し、以降の実行が全部空振りした。
try {

# ---------------------------------------------------------------- 動作確認だけ
# **窓が1つ出たら成功**（2026-09-06 本人判断:「起動したらどんな画面であれ即座に閉じていい。
# 要は起動エラーさえ起きなければいいわけだから」）。見ているのは**起動が通るか**だけ
# ——実体があってもライセンス認証で止まる等はここで分かる。
#
# ★**メニューが出るまで待ってはいけない**。曲を渡さずに起動すると V6 は
#   「NEW PROJECT / OPEN / RECENT OPEN」の**スタート画面**を出し、この画面には
#   メインメニューが無い。最初その条件で待って 120 秒空振りした（本人が画面を見せてくれて判明）。
if ($CheckOnly) {
    $seen = $null
    for ($i = 0; $i -lt 60 -and -not $seen; $i++) {
        Start-Sleep -Seconds 1
        try {
            foreach ($e in (Tops)) {
                if ($e.Current.Name -and $e.Current.Name -notmatch "Hidden|listener|Notification|IME") {
                    $seen = $e.Current.Name; break
                }
            }
        } catch { }
    }
    if (-not $seen) { throw "VOCALOID6 を起動できなかった（$i 秒待っても画面が出ない）" }
    Log "動作確認 OK: 「$seen」が開いた（$i 秒）。閉じます"
    return
}

$main = $null
for ($i = 0; $i -lt $UiReadySec; $i++) {
    Start-Sleep -Seconds 1
    # ★**起動時のダイアログを先に払う**。これが出ていると本体のメニューが押せない
    try { [void](DismissDialogs) } catch { }
    try {
        # ★曲を読み込むと窓の名前が「VOCALOID6 Editor - <曲名>」になるので**前方一致**で見る
        #   （固定一致にしていたら、実際の曲を開いたときに見つからなかった）
        #   ★**名前だけで決めない**（2026-09-06）。復元ダイアログも名前が
        #     「VOCALOID6 Editor」なので、名前一致＋break だと**ダイアログを掴んで**
        #     本体に一生たどり着かなかった。**メインメニューを持つ窓**を本体とする。
        foreach ($e in (Tops)) {
            if ($e.Current.Name -notlike "VOCALOID6 Editor*") { continue }
            if (Find $e $null "xMainMenu") { $main = $e; break }
        }
    } catch { $main = $null }
    if ($main) { break }
}
if (-not $main) {
    # 何が出ているのか分からないまま終わらせない（起動時のダイアログ等を疑う）
    Log "立ち上がらなかった。いま出ている窓:"
    try { foreach ($e in (Tops)) { Log "   '$($e.Current.Name)' / $($e.Current.ClassName)" } } catch {}
    throw "UI が立ち上がらない（$UiReadySec 秒）"
}
Log "UI が出来た（$i 秒）"

    # ★**メニューが生えた＝曲が読み終わった、ではない**。窓の名前が
    #   「VOCALOID6 Editor - <曲名>」になるまで待つ（読み込み中に触ると Invoke が
    #   「認識できないエラー」になる）。名前が変わらない場合も一定時間で先へ進む。
    for ($k = 0; $k -lt 60; $k++) {
        if ($main.Current.Name -match " - ") { Log "曲を読み終えた: $($main.Current.Name)"; break }
        Start-Sleep -Seconds 1
    }
    Start-Sleep -Seconds 3      # 描画が落ち着くまでの余裕

    # ------------------------------------------------------ 2. ミックスダウンを呼ぶ
    # メニューから呼ぶ（`Ctrl+E` も併記されているが、**別プロセスからのキー送信は
    # 前面化が Windows に阻まれる**ことがあり当てにならない。実際 Ctrl+E は届かなかった）。
    Log "ファイル > オーディオミックスダウン... を呼ぶ"
    # ★`FindFirst`（native）だと「ファイル」という名前の **Text** を掴む可能性がある
    #   （メニュー項目の中に同名の TextBlock がいる）。probe で通ったのは Raw の再帰なので揃える。
    # ★**メインメニューの中から探す**（2026-09-06）。窓全体から名前で引くと、
    #   同じ「ファイル」という名前の **Text**（メニュー項目の中の TextBlock）を掴むことがあり、
    #   `ExpandCollapse` も `Invoke` も持たないので「Unsupported Pattern」で落ちる。
    #   探す範囲を `xMainMenu` の下に限れば、当たるのはメニュー項目だけになる。
    $menu = WaitDeep $main $null "xMainMenu" 30 "メインメニュー"
    $file = $null
    for ($t = 0; $t -lt 20 -and -not $file; $t++) {
        $c = $walker.GetFirstChild($menu)
        while ($c) {
            $cur = $c.Current
            if ($cur.Name -eq "ファイル" -and
                $cur.ControlType.ProgrammaticName -match "MenuItem") { $file = $c; break }
            $c = $walker.GetNextSibling($c)
        }
        if (-not $file) { Start-Sleep -Milliseconds 500 }
    }
    if (-not $file) {
        DumpTree $menu (Join-Path $PSScriptRoot "logs\_v6_nomenu.txt")
        throw "ファイルメニューが見つからない"
    }
    $file.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern).Expand()
    Start-Sleep -Milliseconds 1200
    # ★**項目が「使える」状態になるまで待つ**（2026-09-06）。見つかっても `IsEnabled=False`
    #   のことがあり、そのまま押すと「Unsupported Pattern」で落ちる（無効な項目は Invoke できない）。
    #   曲の読み込み中は無効になっているらしく、**曲が長いほど待たされる**（98秒の曲で発覚）。
    #   窓の名前で読み込み完了を推測していたが、この曲では名前が変わらず判定になっていなかった。
    #   **押せるかどうかを直接見る**ほうが確実で、推測が要らない。
    # ★**待つ長さは回数でなく経過時間で決める**（2026-09-26）。以前は「0.5秒×120回＝最大60秒」の
    #   つもりだったが、**1回の探索（部品を深さ10までたどる）に4〜5秒かかる**ことがあり、
    #   実際には**9分半**待ってから開き直していた（『明日への足音』v3。開き直したらすぐ押せた）。
    $item = $null
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $lastLog = 0
    while ($sw.Elapsed.TotalSeconds -lt 60) {
        # **MenuItem に限る**（同名の Text を掴むと Invoke できずに落ちる）
        $cand = Find $file "オーディオミックスダウン..." $null 0 10 "MenuItem"
        if (-not $cand) { $cand = Find $main "オーディオミックスダウン..." $null 0 10 "MenuItem" }
        if ($cand -and $cand.Current.IsEnabled) { $item = $cand; break }
        $sec = [int]$sw.Elapsed.TotalSeconds
        if ($sec - $lastLog -ge 10) { Log "  …まだ押せない（読み込み中）: ${sec}秒"; $lastLog = $sec }
        Start-Sleep -Milliseconds 500
    }
    # ★**開き直さない**（2026-09-06）。待つ間に `Expand()` を繰り返したら
    #   **メニューが二重に開き**、同じ項目が2組ぶら下がった状態になった（ツリーに2回出る）。
    #   その状態では押せる方を掴めない。開くのは最初の1回だけにする。
    if (-not $item) {
        # 一度メニューを閉じ、開き直して**もう一度だけ**待つ（読み込みが長引いた場合の保険）
        Log "  押せないまま待ち切った。メニューを閉じて開き直す"
        try { $file.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern).Collapse() } catch {}
        Start-Sleep -Seconds 2
        try { $file.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern).Expand() } catch {}
        Start-Sleep -Milliseconds 1200
        $sw = [System.Diagnostics.Stopwatch]::StartNew()   # こちらも経過時間で60秒（上と同じ理由）
        $t = 0
        while ($sw.Elapsed.TotalSeconds -lt 60) {
            $cand = Find $file "オーディオミックスダウン..." $null 0 10 "MenuItem"
            if ($cand -and $cand.Current.IsEnabled) { $item = $cand; break }
            # ★押せないまま居座るのは、**別のダイアログが出ている**ときが多い（復元の確認など）。
            #   待つだけでは絶対に開かないので、見つけたら払う。
            if ($t % 4 -eq 3) { try { [void](DismissDialogs) } catch { } }
            $t++
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $item) {
        DumpTree $main (Join-Path $PSScriptRoot "logs\_v6_nomenu.txt")
        throw "オーディオミックスダウンの項目が押せる状態にならない（読み込みが終わっていない可能性）"
    }
    Push $item "オーディオミックスダウン..."

    # ------------------------------------------------------ 3. 設定して OK
    $dlg = WaitDeep $main $null "xAudioMixdownDlg" 30 "ミックスダウンのダイアログ"
    Log "ダイアログが出た"

    # **メイントラック**＝伴奏込みの完成ミックス（vpr は伴奏を同梱している）
    $master = Find $dlg $null "xRBMasterTrack"
    if ($master) {
        $master.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
        Log "  メイントラックを選んだ"
    }
    # 書き出し後にエクスプローラが開くと邪魔なので外す
    $chk = Find $dlg $null "xOpenExplorerCheckBox"
    if ($chk) {
        $tp = $chk.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
        if ($tp.Current.ToggleState -ne [System.Windows.Automation.ToggleState]::Off) {
            $tp.Toggle(); Log "  「書き出し後にフォルダを開く」を外した"
        }
    }

    Push (WaitFor $dlg "OK" "xOKButton" 10 "OK ボタン") "OK"

    # ------------------------------------------------------ 4. 保存先を指定
    # ★OK のあとに出る Windows 標準の保存ダイアログは、**別のトップレベル窓ではなく
    #   本体窓の中**に現れる（V6 のダイアログと同じ作り）。最初はトップレベルを探して
    #   「出ない」と誤判定した。中身は素の Win32 ダイアログで、
    #     ファイル名の欄 = AutomationId 1001 ／ 保存ボタン = AutomationId 1
    $save = $null
    for ($i = 0; $i -lt 60 -and -not $save; $i++) {
        $save = Find $main "名前を付けて保存" $null
        if (-not $save) { $save = Find $main "Save As" $null }
        if (-not $save) { Start-Sleep -Milliseconds 500 }
    }
    if (-not $save) {
        DumpTree $main (Join-Path $PSScriptRoot "logs\_v6_nosave.txt")
        throw "保存ダイアログが出ない。上のファイルで中を確認する"
    }
    Log "保存ダイアログが出た"

    # ★ファイル名欄は **UIA の ValuePattern を持っていない**（「サポートされていないパターンです」）。
    #   保存ダイアログは素の Win32 なので、**窓メッセージで直接入れる**のが確実——
    #   前面化もキー送信も要らず、他の窓に邪魔されない。
    #     ファイル名の欄 … ComboBoxEx32 → ComboBox → Edit（古典的な入れ子）
    #     保存ボタン     … ダイアログのコントロール ID 1
    $hDlg = [IntPtr]$save.Current.NativeWindowHandle
    if ($hDlg -eq [IntPtr]::Zero) { throw "保存ダイアログの窓ハンドルが取れない" }

    $hEdit = [Fg]::FindVisibleEdit($hDlg)                                      # 子孫から見えている Edit
    if ($hEdit -eq [IntPtr]::Zero) { $hEdit = [Fg]::GetDlgItem($hDlg, 1148) }   # 古い作りはこちら
    if ($hEdit -eq [IntPtr]::Zero) {
        DumpTree $save (Join-Path $PSScriptRoot "logs\_v6_savedlg.txt")
        throw "ファイル名の欄が見つからない"
    }
    [Fg]::SendMessageW($hEdit, 0x000C, [IntPtr]::Zero, $Out) | Out-Null      # WM_SETTEXT
    Log "  保存先: $Out"
    Start-Sleep -Milliseconds 500

    $hBtn = [Fg]::GetDlgItem($hDlg, 1)                                        # 「保存」
    if ($hBtn -eq [IntPtr]::Zero) { throw "保存ボタンが見つからない" }
    [Fg]::SendMessage($hBtn, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null  # BM_CLICK
    Log "  保存を押した"

    # ------------------------------------------------------ 5. 書き出しの完了を待つ
    # 進捗ダイアログの有無に頼らず、**ファイルが出来てサイズが止まるまで**見る（作りが変わっても効く）
    Log "書き出し中…（最大 $RenderSec 秒）"
    $last = -1; $stable = 0; $done = $false
    for ($i = 0; $i -lt $RenderSec; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $Out) {
            $sz = (Get-Item $Out).Length
            if ($sz -eq $last -and $sz -gt 0) { $stable++ } else { $stable = 0 }
            $last = $sz
            if ($stable -ge 5) { $done = $true; break }      # 5秒変わらなければ終わり
        }
        if ($i % 30 -eq 29) { Log "  …$($i+1) 秒経過（$([math]::Round($last/1KB)) KB）" }
    }
    if (-not $done) { throw "書き出しが終わらない（$RenderSec 秒・$last バイト）" }
    Log "書き出せた: $([math]::Round($last/1MB, 2)) MB"
}
finally {
    if (-not $KeepOpen) {
        # ★**必ず閉じきる**。閉じ残すと次の実行が単一インスタンスに阻まれて丸ごと空振りする。
        #   ここは失敗経路からも通るので、掴んだ PID ではなく**その時いる V6 全部**を相手にする。
        Log "V6 を閉じる"
        # ★動作確認は**即座に終了させる**（2026-09-06 本人「起動したらどんな画面であれ即座に
        #   閉じていい」）。曲を開いていないので失うものが無い。丁寧に閉じようとすると
        #   保存確認を探す処理がスタート画面で空回りし、**31 秒**かかっていた。
        if ($CheckOnly) {
            Get-Process VOCALOID6 -ErrorAction SilentlyContinue | Stop-Process -Force
            Start-Sleep -Milliseconds 500
        } else {
        foreach ($p in @(Get-Process VOCALOID6 -ErrorAction SilentlyContinue)) {
            $p.CloseMainWindow() | Out-Null
        }
        # 「保存しますか」が出たら**保存しない**（曲を書き換えない）
        # ★**待つ時間をけちらない**（2026-09-06）。8 秒で見切って強制終了していたが、
        #   大きい曲は閉じきるのに時間がかかる。途中で殺すと V6 が「異常終了した」と見なし、
        #   **次の起動で復元ダイアログを出して、以降の書き出しが全部失敗する**。
        #   実際にこれで詰まった（60 秒＋120 秒の空振りを繰り返した）。
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Milliseconds 700
            if (-not (Get-Process VOCALOID6 -ErrorAction SilentlyContinue)) { break }
            # ★UIA ではなく **Win32** で押す。V6 のこの手のダイアログは UIA から見ると
            #   ボタンが `Pane` に見えて Invoke できない（2026-09-06 実測）。
            [void](DismissDialogs)
        }
        Start-Sleep -Seconds 2
        $rest = Get-Process VOCALOID6 -ErrorAction SilentlyContinue
        if ($rest) { Log "  閉じないので終了させる（次回に復元ダイアログが出るかもしれない）"; $rest | Stop-Process -Force }
        }
    }
}
Log "完了: $Out"
