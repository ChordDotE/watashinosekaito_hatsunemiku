# -*- coding: utf-8 -*-
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

"""認識サービスのメンテナンス画面（HTML 生成）。

顔と声の登録状態を1画面で確認・操作する。**セットアップ経路を兼ねる**:
会話フロー（ツール）を使わずに、手元のファイルから直接登録できる唯一の入口。

2026-08-09 に face_recognition_service.py から分離した（声の追加でファイル上限
500行を超えるため）。
"""
from __future__ import annotations

from . import voice_engine

_STYLE = """
    body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }
    h1 { font-size: 1.4em; border-bottom: 2px solid #e06; padding-bottom: 8px; }
    h2.group { font-size: 1.2em; margin: 32px 0 8px; color: #e06; }
    .status { border-radius: 8px; padding: 12px 16px; margin: 16px 0; }
    .status.on { background: #e8f5e9; }
    .status.off { background: #fce4ec; }
    .photos { display: flex; gap: 24px; margin: 16px 0; flex-wrap: wrap; }
    .photo-box { text-align: center; }
    .photo-box img { max-width: 300px; max-height: 300px; border-radius: 8px; border: 1px solid #ddd; }
    .label { font-size: 0.85em; color: #666; margin: 4px 0; }
    section { margin: 16px 0; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }
    section h3 { font-size: 1.05em; margin-top: 0; }
    input[type=file] { margin: 8px 0; }
    button { background: #e06; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 8px; }
    button.danger { background: #c62828; }
    button:hover { opacity: 0.85; }
    audio { display: block; margin: 8px 0; max-width: 320px; }
"""

_SCRIPT = """
    async function fileToB64(input) {
      const file = input.files[0];
      if (!file) return null;
      return new Promise(r => {
        const reader = new FileReader();
        reader.onload = e => r(e.target.result.split(',')[1]);
        reader.readAsDataURL(file);
      });
    }

    function show(id, json) {
      const el = document.getElementById(id);
      el.style.display = 'block';
      el.textContent = JSON.stringify(json, null, 2);
    }

    async function post(url, inputId, resultId, key) {
      const b64 = await fileToB64(document.getElementById(inputId));
      if (!b64) { alert('ファイルを選択してください'); return null; }
      const res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({[key]: b64})
      });
      const json = await res.json();
      show(resultId, json);
      return json;
    }

    async function doFaceEnroll() {
      const j = await post('/face/enroll', 'faceEnrollFile', 'faceEnrollResult', 'image_b64');
      if (j && j.status === 'enrolled') setTimeout(() => location.reload(), 1000);
    }
    const doFaceVerify = () => post('/face/verify', 'faceVerifyFile', 'faceVerifyResult', 'image_b64');

    async function doVoiceEnroll() {
      const j = await post('/voice/enroll', 'voiceEnrollFile', 'voiceEnrollResult', 'audio_b64');
      if (j && j.status === 'enrolled') setTimeout(() => location.reload(), 1000);
    }
    const doVoiceVerify = () => post('/voice/verify', 'voiceVerifyFile', 'voiceVerifyResult', 'audio_b64');

    async function doDelete(kind) {
      if (!confirm('登録を削除しますか?')) return;
      const res = await fetch('/' + kind + '/enrollment', {method: 'DELETE'});
      const json = await res.json();
      show(kind + 'DeleteResult', json);
      if (json.status === 'deleted') setTimeout(() => location.reload(), 800);
    }

    document.querySelectorAll('.result-box').forEach(el => {
      Object.assign(el.style, {
        marginTop: '12px', background: '#f5f5f5', padding: '10px',
        borderRadius: '4px', fontFamily: 'monospace', whiteSpace: 'pre-wrap',
        fontSize: '0.85em', display: 'none'
      });
    });
"""


def _status_block(enrolled: bool, enrolled_at: str | None, extra: str = "") -> str:
    text = f"登録済み（{enrolled_at}）{extra}" if enrolled else "未登録"
    cls = "on" if enrolled else "off"
    return f'<div class="status {cls}"><strong>状態:</strong> {text}</div>'


def _delete_section(kind: str, enrolled: bool) -> str:
    if not enrolled:
        return ""
    return (
        f'<section><h3>登録解除</h3>'
        f'<button class="danger" onclick="doDelete(\'{kind}\')">登録を削除する</button>'
        f'<div id="{kind}DeleteResult" class="result-box"></div></section>'
    )


def render(face_enrolled_at: str | None, voice_meta: dict | None) -> str:
    """メンテナンス画面の HTML を返す。

    face_enrolled_at: 顔の登録日時（未登録なら None）
    voice_meta: 声の登録情報 {"enrolled_at": str, "seconds": float}（未登録なら None）
    """
    face_on = face_enrolled_at is not None
    voice_on = voice_meta is not None

    face_photos = ""
    if face_on:
        face_photos = """
    <div class="photos">
      <div class="photo-box">
        <p class="label">顔クロップ</p>
        <img src="/face/enrollment/photo/crop" alt="顔クロップ">
      </div>
      <div class="photo-box">
        <p class="label">登録時の画像（サムネイル）</p>
        <img src="/face/enrollment/photo/full" alt="フルサムネイル">
      </div>
    </div>"""

    voice_sample = ""
    voice_extra = ""
    if voice_on:
        sec = voice_meta.get("seconds")
        voice_extra = f"・{sec:.1f}秒の音声から" if isinstance(sec, (int, float)) else ""
        voice_sample = """
    <p class="label">登録に使った音声</p>
    <audio controls src="/voice/enrollment/sample"></audio>"""

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>ミク 認識メンテナンス</title>
  <style>{_STYLE}</style>
</head>
<body>
  <h1>🎵 ミク 認識メンテナンス（顔・声）</h1>

  <h2 class="group">顔</h2>
  {_status_block(face_on, face_enrolled_at)}
  {face_photos}
  <section>
    <h3>顔を登録する（上書き）</h3>
    <p class="label">1人だけ写った画像を選んでください。</p>
    <input type="file" id="faceEnrollFile" accept="image/*">
    <br><button onclick="doFaceEnroll()">登録</button>
    <div id="faceEnrollResult" class="result-box"></div>
  </section>
  <section>
    <h3>テスト照合</h3>
    <p class="label">confidence の見方: 本人はおおむね 0.65 以上（多くは 0.7〜0.8）・別人は 0.4 以下。
       同一人物判定の目安は 0.55 以上（実測分布に基づく）</p>
    <input type="file" id="faceVerifyFile" accept="image/*">
    <br><button onclick="doFaceVerify()">照合する</button>
    <div id="faceVerifyResult" class="result-box"></div>
  </section>
  {_delete_section("face", face_on)}

  <h2 class="group">声</h2>
  {_status_block(voice_on, (voice_meta or {}).get("enrolled_at"), voice_extra)}
  {voice_sample}
  <section>
    <h3>声を登録する（上書き）</h3>
    <p class="label">マスター1人だけが話している音声を選んでください
       （{voice_engine.RECOMMEND_ENROLL_SEC:.0f}秒以上を推奨・{voice_engine.MIN_ENROLL_SEC:.0f}秒未満は登録できません）。
       対応形式: wav / mp3 / flac / ogg</p>
    <input type="file" id="voiceEnrollFile" accept="audio/*,.wav,.mp3,.flac,.ogg">
    <br><button onclick="doVoiceEnroll()">登録</button>
    <div id="voiceEnrollResult" class="result-box"></div>
  </section>
  <section>
    <h3>テスト照合</h3>
    <p class="label">score の見方: 音声の長さで大きく動く。3秒以上なら本人 0.85 前後・同性同帯域の他人 0.45 以下。
       1秒程度だと本人でも 0.60 前後まで下がる（reliable=false は単独で測るには短いという印）</p>
    <input type="file" id="voiceVerifyFile" accept="audio/*,.wav,.mp3,.flac,.ogg">
    <br><button onclick="doVoiceVerify()">照合する</button>
    <div id="voiceVerifyResult" class="result-box"></div>
  </section>
  {_delete_section("voice", voice_on)}

  <script>{_SCRIPT}</script>
</body>
</html>"""
