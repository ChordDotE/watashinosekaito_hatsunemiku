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

"""マスターの顔の登録・照合。

facenet-pytorch（MTCNN 検出 + InceptionResnetV1/VGGFace2 埋め込み・CPU 動作）。
2026-08-09 に face_recognition_service.py から分離した（声の追加でファイル上限
500行を超えるため。機能境界での分割・挙動は変えていない）。

埋め込みは L2 正規化済み（距離範囲 0〜2）。同一人物判定の目安は confidence 0.55 以上
（文献しきい値は距離 1.1＝0.45 相当だが緩すぎた。実測: 本人 0.65〜0.8 中心・別人 0.32〜0.40
〔ポスターの顔含む〕に基づき 2026-07-25 引き上げ）。**本人判定（is_master）はせず
confidence のみ返し、解釈は LLM に委ねる**（似た顔がいる場面でサービス側の二値判定が
誤動作した経緯があるため）。

2026-08-09 の実写真での実測（test/perception_latency/results_speaker.md 付録4）:
本人 0.72 / 同じ写真の別人 0.40 / マスターが写っていない写真の6人 0.32〜0.40。
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import pickle
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

_logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

EMBEDDING_DIM = 512

_VDB_BASE_URL = os.environ.get("VDB_BASE_URL", "http://127.0.0.1:8020")

_DATA_DIR = Path(
    os.environ.get(
        "FACE_ENCODING_PATH",
        # data/KV は KV サービスのテキストストア（*.txt のみ扱う）なので顔データは別置き
        # （2026-07-26 に data/KV から移動。旧環境のファイルは手動移動で移行済み）
        Path(__file__).resolve().parents[2] / "data" / "face" / "master_face_encoding.pkl",
    )
).parent

_ENCODING_PATH = _DATA_DIR / "master_face_encoding.pkl"
_PHOTO_CROP_PATH = _DATA_DIR / "master_face_photo_crop.jpg"
_PHOTO_FULL_PATH = _DATA_DIR / "master_face_photo_full.jpg"

_CROP_PADDING = 30   # 顔クロップの余白（px）
_FULL_MAX_PX  = 600  # フルサムネイルの最大辺（px）
_CROP_MAX_PX  = 300  # クロップサムネイルの最大辺（px）

_models_lock = threading.Lock()
_models: tuple | None = None


class FaceError(Exception):
    """呼び出し側がユーザーへそのまま見せられる理由つきエラー。"""

    def __init__(self, code: str, message: str = "", **extra):
        super().__init__(message or code)
        self.code = code
        self.extra = extra


# ── モデル ────────────────────────────────────────────────────

def _get_models():
    """MTCNN と InceptionResnetV1 を遅延初期化して返す（初回のみ数秒）。

    CPU 固定: 顔照合は低頻度かつ 1 枚 0.2 秒程度で足りるため、VRAM は
    STT/TTS 系サービスに譲る。学習済み重み（約107MB）はセットアップ時に
    事前ダウンロード済み（setup_common.ps1 の Install-MikuFaceWeights。保存先は TORCH_HOME=miku\\runtime\\models\\torch
    〔setup/system_start が設定・フォルダ内隔離方針〕）。
    """
    global _models
    with _models_lock:
        if _models is None:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            mtcnn = MTCNN(keep_all=True, device="cpu")
            resnet = InceptionResnetV1(pretrained="vggface2").eval()
            _models = (mtcnn, resnet)
    return _models


def detect_and_embed(img: Image.Image) -> tuple[list[tuple], list[np.ndarray]]:
    """PIL 画像から (locations, embeddings) を返す。顔なしは ([], [])。

    locations は旧 dlib 実装と互換の (top, right, bottom, left) int タプル。
    """
    import torch

    mtcnn, resnet = _get_models()
    boxes, _probs = mtcnn.detect(img)
    if boxes is None:
        return [], []
    faces = mtcnn.extract(img, boxes, save_path=None)
    with torch.no_grad():
        embs = resnet(faces).numpy()
    # MTCNN の box は画像外にはみ出しうるので画像範囲にクランプする（旧 dlib 実装と同じ挙動）
    w, h = img.size
    locations = [
        (max(0, int(y1)), min(w, int(x2)), min(h, int(y2)), max(0, int(x1)))
        for x1, y1, x2, y2 in boxes
    ]
    return locations, list(embs)


def distance_to_confidence(distance: float) -> float:
    """L2 距離（0〜2）を 0〜1 の類似度に変換する。文献しきい値 1.1 は 0.45 に相当。"""
    return round(max(0.0, min(1.0, 1.0 - distance / 2.0)), 4)


# ── 画像ユーティリティ ────────────────────────────────────────

def open_upright(fp) -> Image.Image:
    """EXIF の向き情報を適用して直立画像として開く。

    スマホ写真は生ピクセルが横倒しで EXIF Orientation により表示時に回転される形式が
    多く、未適用だと顔検出器（直立顔のみ対応）が顔を見つけられない
    （実例: Orientation=6 の2人写真が検出0件 → 適用後2件・2026-07-25）。
    """
    img = Image.open(fp)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def b64_to_image(image_b64: str) -> Image.Image:
    return open_upright(io.BytesIO(base64.b64decode(image_b64)))


def _save_photos(pil: Image.Image, location: tuple) -> None:
    """顔クロップとフルサムネイルを保存する。"""
    w, h = pil.size

    full = pil.copy()
    full.thumbnail((_FULL_MAX_PX, _FULL_MAX_PX))
    full.save(_PHOTO_FULL_PATH, "JPEG", quality=85)

    top, right, bottom, left = location
    top    = max(0, top    - _CROP_PADDING)
    right  = min(w, right  + _CROP_PADDING)
    bottom = min(h, bottom + _CROP_PADDING)
    left   = max(0, left   - _CROP_PADDING)
    crop = pil.crop((left, top, right, bottom))
    crop.thumbnail((_CROP_MAX_PX, _CROP_MAX_PX))
    crop.save(_PHOTO_CROP_PATH, "JPEG", quality=85)


# ── 埋め込みの読み書き ────────────────────────────────────────

def load_encoding() -> tuple[np.ndarray | None, str | None]:
    if not _ENCODING_PATH.exists():
        return None, None
    try:
        with _ENCODING_PATH.open("rb") as f:
            data = pickle.load(f)
        return data["encoding"], data.get("enrolled_at")
    except Exception:
        return None, None


def _save_encoding(encoding: np.ndarray) -> str:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    enrolled_at = datetime.now(_JST).isoformat()
    with _ENCODING_PATH.open("wb") as f:
        pickle.dump(
            {
                "encoding": encoding,
                "enrolled_at": enrolled_at,
                "model": "facenet-inception-resnet-v1-vggface2",
            },
            f,
        )
    return enrolled_at


def delete_enrollment() -> None:
    for p in (_ENCODING_PATH, _PHOTO_CROP_PATH, _PHOTO_FULL_PATH):
        if p.exists():
            p.unlink()


def photo_crop_path() -> Path:
    return _PHOTO_CROP_PATH


def photo_full_path() -> Path:
    return _PHOTO_FULL_PATH


# ── 旧形式（dlib 128次元）からの自動移行 ──────────────────────

def _record_migration_memory() -> None:
    """移行イベントを長期記憶（VDB long_memory）に記録する。

    ベストエフォート: VDB 未起動などで失敗しても照合処理は継続する。
    item 形式は long_memory_vdb_mcp_server.add_event と同じ。
    """
    try:
        now = datetime.now(_JST)
        start = now.strftime("%Y-%m-%d %H:%M")
        event_id = f"E{now.strftime('%Y%m%d')}-facemig"
        what = "顔認識の仕組みを新しくして、保存してあった写真からマスターの顔を覚え直した"
        row_data = {
            "id": event_id,
            "start_dt": start,
            "end_dt": start,
            "place": "",
            "what": what,
            "importance": "小",
            "data": json.dumps({"points": [], "feelings": [], "who": []}, ensure_ascii=False),
        }
        metadata = {
            "category": "events",
            "type": "event",
            "always_in_context": False,
            "data_json": json.dumps(row_data, ensure_ascii=False),
            "date": start,
            "importance": "小",
            "registered_at": now.isoformat(),
        }
        requests.post(
            f"{_VDB_BASE_URL}/vdb/long_memory/upsert",
            json={"items": [{"id": event_id, "document": f"{start}\n{what}", "metadata": metadata}]},
            timeout=10,
        )
    except Exception as e:
        _logger.warning("顔認識移行イベントの長期記憶登録に失敗: %r", e)


def _migrate_old_encoding() -> np.ndarray | None:
    """旧 dlib(128次元) の登録を、保存済みフル写真から 512 次元へ自動再生成する。

    写真が無い・顔を1つに特定できない場合は None（未登録扱い）を返す。
    成功時は長期記憶へ移行イベントを記録する（ベストエフォート）。
    """
    if not _PHOTO_FULL_PATH.exists():
        return None
    try:
        img = open_upright(_PHOTO_FULL_PATH)
        _locations, embeddings = detect_and_embed(img)
    except Exception as e:
        _logger.warning("旧顔登録の自動移行に失敗: %r", e)
        return None
    if len(embeddings) != 1:
        _logger.warning("旧顔登録の自動移行に失敗: 保存写真の顔検出数=%d", len(embeddings))
        return None
    _save_encoding(embeddings[0])
    _logger.info("旧顔登録（dlib 128次元）を facenet 512次元へ自動移行しました")
    _record_migration_memory()
    return embeddings[0]


def load_encoding_for_verify() -> np.ndarray | None:
    """照合用の埋め込みを返す。旧形式なら自動移行を試みる。"""
    encoding, _ = load_encoding()
    if encoding is None:
        return None
    if getattr(encoding, "shape", (0,))[-1] == EMBEDDING_DIM:
        return encoding
    return _migrate_old_encoding()


# ── 公開処理 ──────────────────────────────────────────────────

def enroll(image_b64: str) -> dict:
    """画像からマスターの顔を登録する。顔が1つでないときは FaceError。"""
    image = b64_to_image(image_b64)
    locations, embeddings = detect_and_embed(image)
    if len(locations) == 0:
        raise FaceError("no_face_detected", "画像に顔が検出されませんでした")
    if len(locations) > 1:
        raise FaceError("multiple_faces", "複数の顔が検出されました", count=len(locations))

    enrolled_at = _save_encoding(embeddings[0])
    try:
        _save_photos(image, locations[0])
    except Exception as e:
        _logger.warning("登録写真の保存に失敗: %r", e)
    return {"status": "enrolled", "faces_detected": 1, "enrolled_at": enrolled_at}


def verify(image_b64: str) -> dict:
    """画像に写る顔ごとの位置と一致度を返す。本人判定はしない。"""
    known = load_encoding_for_verify()
    if known is None:
        return {"enrolled": False}
    image = b64_to_image(image_b64)
    locations, embeddings = detect_and_embed(image)
    if len(locations) == 0:
        raise FaceError("no_face_detected", "画像に顔が検出されませんでした")
    faces = [
        {
            "box": {"top": top, "right": right, "bottom": bottom, "left": left},
            "confidence": distance_to_confidence(float(np.linalg.norm(known - enc))),
        }
        for (top, right, bottom, left), enc in zip(locations, embeddings)
    ]
    return {"enrolled": True, "faces": faces}
