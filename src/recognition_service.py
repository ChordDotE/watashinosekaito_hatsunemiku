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

"""マスター認識サービス（顔・声）。

マスター本人の同定を担う。顔は facenet-pytorch、声は speechbrain ECAPA-TDNN で、
どちらも CPU 動作・**本人判定はせず一致度だけを返す**（解釈は LLM に委ねる）。
登録は Agent ツール（face_enroll / voice_enroll）から、照合は **LLM 投入前の前処理**
（STT の `/voice/analyze`・agent_main の `/face/verify`）から HTTP で直接呼び出される。
Gateway に中継エンドポイントは設けない。

顔と声を1サービスに置く理由は「マスターの生体特徴による本人同定」という同一ドメインだから
（2026-08-09 の判断。docs/dev/decisions.md）。旧 face_recognition_service.py を改名し、
顔のエンドポイントは /face/* へ移した。

エンドポイント:
  POST   /face/enroll                 — 顔登録
  POST   /face/verify                 — 顔照合
  GET    /face/status                 — 顔の登録状態
  DELETE /face/enrollment             — 顔の登録解除
  GET    /face/enrollment/photo/crop  — 顔クロップ画像を配信
  GET    /face/enrollment/photo/full  — フルサムネイル画像を配信
  POST   /voice/enroll                — 声登録
  POST   /voice/verify                — 声照合
  GET    /voice/status                — 声の登録状態
  DELETE /voice/enrollment            — 声の登録解除
  GET    /voice/enrollment/sample     — 登録に使った音声を配信
  GET    /status                      — 顔・声をまとめた登録状態
  GET    /health                      — 疎通確認
  GET    /                            — メンテナンス画面（顔・声）
"""
from __future__ import annotations

import base64
import pickle
import threading

from flask import Flask, Response, jsonify, request, send_file

from recognition import face_engine, maintenance_page, voice_engine

app = Flask(__name__)


# ── 顔 ────────────────────────────────────────────────────────

@app.route("/face/enroll", methods=["POST"])
def face_enroll():
    body = request.get_json(force=True, silent=True) or {}
    image_b64 = body.get("image_b64", "")
    if not image_b64:
        return jsonify({"error": "image_b64_required"}), 400
    try:
        return jsonify(face_engine.enroll(image_b64)), 200
    except face_engine.FaceError as e:
        return jsonify({"error": e.code, **e.extra}), 400
    except Exception as e:
        return jsonify({"error": f"enroll_error: {e}"}), 500


@app.route("/face/verify", methods=["POST"])
def face_verify():
    body = request.get_json(force=True, silent=True) or {}
    image_b64 = body.get("image_b64", "")
    if not image_b64:
        return jsonify({"error": "image_b64_required"}), 400
    try:
        return jsonify(face_engine.verify(image_b64)), 200
    except face_engine.FaceError as e:
        return jsonify({"error": e.code, **e.extra}), 400
    except Exception as e:
        return jsonify({"error": f"image_decode_error: {e}"}), 500


@app.route("/face/status", methods=["GET"])
def face_status():
    _, enrolled_at = face_engine.load_encoding()
    if enrolled_at is None:
        return jsonify({"enrolled": False}), 200
    return jsonify({"enrolled": True, "enrolled_at": enrolled_at}), 200


@app.route("/face/enrollment", methods=["DELETE"])
def face_delete_enrollment():
    """登録データ（埋め込み＋写真）をすべて削除する。"""
    face_engine.delete_enrollment()
    return jsonify({"status": "deleted"}), 200


@app.route("/face/enrollment/photo/crop", methods=["GET"])
def face_photo_crop():
    path = face_engine.photo_crop_path()
    if not path.exists():
        return jsonify({"error": "not_found"}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/face/enrollment/photo/full", methods=["GET"])
def face_photo_full():
    path = face_engine.photo_full_path()
    if not path.exists():
        return jsonify({"error": "not_found"}), 404
    return send_file(path, mimetype="image/jpeg")


# ── 声 ────────────────────────────────────────────────────────

@app.route("/voice/enroll", methods=["POST"])
def voice_enroll():
    body = request.get_json(force=True, silent=True) or {}
    audio_b64 = body.get("audio_b64", "")
    if not audio_b64:
        return jsonify({"error": "audio_b64_required"}), 400
    try:
        data = base64.b64decode(audio_b64)
    except Exception as e:
        return jsonify({"error": f"audio_decode_error: {e}"}), 400
    try:
        return jsonify(voice_engine.enroll(data)), 200
    except voice_engine.VoiceError as e:
        # 短すぎ・読めない形式など、そのままユーザーへ見せられる理由
        return jsonify({"error": "invalid_audio", "reason": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"enroll_error: {e}"}), 500


@app.route("/voice/verify", methods=["POST"])
def voice_verify():
    body = request.get_json(force=True, silent=True) or {}
    audio_b64 = body.get("audio_b64", "")
    if not audio_b64:
        return jsonify({"error": "audio_b64_required"}), 400
    try:
        data = base64.b64decode(audio_b64)
    except Exception as e:
        return jsonify({"error": f"audio_decode_error: {e}"}), 400
    try:
        return jsonify(voice_engine.verify(data)), 200
    except voice_engine.VoiceError as e:
        return jsonify({"error": "invalid_audio", "reason": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"verify_error: {e}"}), 500


@app.route("/voice/analyze", methods=["POST"])
def voice_analyze():
    """1ターン分の音声を区間に切り、区間ごとの一致度を返す（LLM 投入前の前処理）。

    STT が ASR と**並行**に呼ぶ（照合は PCM だけで完結し ASR 結果を待たない）。
    実測 2026-08-09: ASR の裏に隠れて追加は +3〜12ms。
    """
    body = request.get_json(force=True, silent=True) or {}
    audio_b64 = body.get("audio_b64", "")
    if not audio_b64:
        return jsonify({"error": "audio_b64_required"}), 400
    try:
        data = base64.b64decode(audio_b64)
    except Exception as e:
        return jsonify({"error": f"audio_decode_error: {e}"}), 400
    try:
        return jsonify(voice_engine.analyze(data)), 200
    except voice_engine.VoiceError as e:
        return jsonify({"error": "invalid_audio", "reason": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"analyze_error: {e}"}), 500


@app.route("/voice/status", methods=["GET"])
def voice_status():
    _, enrolled_at = voice_engine.load_embedding()
    if enrolled_at is None:
        return jsonify({"enrolled": False}), 200
    return jsonify({"enrolled": True, "enrolled_at": enrolled_at}), 200


@app.route("/voice/enrollment", methods=["DELETE"])
def voice_delete_enrollment():
    voice_engine.delete_enrollment()
    return jsonify({"status": "deleted"}), 200


@app.route("/voice/enrollment/sample", methods=["GET"])
def voice_sample():
    path = voice_engine.sample_path()
    if not path.exists():
        return jsonify({"error": "not_found"}), 404
    return send_file(path, mimetype="audio/wav")


# ── 共通 ──────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    """顔・声の登録状態をまとめて返す（Gateway の system_state 用）。"""
    _, face_at = face_engine.load_encoding()
    _, voice_at = voice_engine.load_embedding()
    return jsonify({
        "face": {"enrolled": face_at is not None, "enrolled_at": face_at},
        "voice": {"enrolled": voice_at is not None, "enrolled_at": voice_at},
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def maintenance():
    _, face_at = face_engine.load_encoding()
    voice_meta = None
    path = voice_engine._EMBEDDING_PATH
    if path.exists():
        try:
            with path.open("rb") as f:
                d = pickle.load(f)
            voice_meta = {"enrolled_at": d.get("enrolled_at"), "seconds": d.get("seconds")}
        except Exception:
            voice_meta = None
    return Response(maintenance_page.render(face_at, voice_meta), mimetype="text/html")


# ── 起動エントリポイント ──────────────────────────────────────

def _preload_models() -> None:
    """顔のモデルを起動時に読み込んでおく（初回照合のタイムアウトを防ぐ）。

    facenet-pytorch の重み（約107MB）は初回の /face/verify で読まれるため、
    再起動後の1ターン目だけ 20秒のタイムアウトに掛かって顔の情報が欠けていた
    （2026-08-12 実機で発生。camera_0.jpg だけ ReadTimeout）。
    STT がモデルを事前ロードするのと同じ作法で、起動時に済ませておく。

    **声（speechbrain）は事前ロードしない**。読み込むと同一プロセスの顔が壊れるため
    無害化（voice_engine.defuse_lazy_modules）が要るが、実際に使うまで遅らせる方が
    起動が速く、顔との干渉も起きにくい（登録済みでなければ一度も読まれない）。
    """
    if face_engine.load_encoding()[1] is None:
        return                      # 未登録なら照合は走らないので読み込む必要がない
    try:
        face_engine._get_models()
        print("  顔モデルの事前ロード完了")
    except Exception as e:
        print(f"  顔モデルの事前ロードに失敗（初回照合時に読み込む）: {type(e).__name__}: {e}")


def run_recognition_service(port: int = 5007, host: str = "127.0.0.1") -> None:
    """Gateway から threading.Thread で呼び出されるエントリポイント。

    host は 127.0.0.1 固定で使う（認証が無いので、同じ Wi-Fi の他の端末から
    登録した顔写真・声を取らせない・消させない・2026-10-08）。
    """
    print(f"Recognition Serviceを起動します: http://{host}:{port}")
    print(f"  認識メンテ画面（顔・声）: http://127.0.0.1:{port}/")
    threading.Thread(target=_preload_models, daemon=True,
                     name="Recognition-Preload").start()
    app.run(host=host, port=port, debug=False, use_reloader=False)
