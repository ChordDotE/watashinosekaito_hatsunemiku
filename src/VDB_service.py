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

"""Vector DB サービス（Chroma 親子チャンキング + BM25 ハイブリッド検索）。

埋め込みは **Gemini Embedding API**（`models/gemini-embedding-2`）。ローカルモデルは使わない
（2026-06-07 に bge-m3 から移行。配布時に GPU/RAM を要求しないため）。
形態素解析（MeCab）と BM25 はローカル CPU で動く。

KV_service.py と同じパターンで Flask HTTP サービスとして提供する。
gateway_service.py から別スレッドで起動するか、単体で起動する。

Endpoints (API):
  POST /vdb/upsert          親子チャンク投入
  POST /vdb/search          ハイブリッド検索（ベクトル + BM25 → RRF 統合）
  GET  /vdb/stats           件数確認
  GET  /vdb/sources         ソース一覧
  GET  /vdb/recent          直近 N 件の親チャンク（古い順、Dify「直近5会話取得」相当）
  GET  /vdb/parents         親チャンク一覧 (?source= &offset= &limit=)
  GET  /vdb/parent/<id>     親チャンク詳細 + 子チャンク一覧

Endpoints (GUI):
  GET  /                    ブラウザ用メンテナンス画面

Usage (単体起動):
  python VDB_service.py
  python VDB_service.py --port 8020
"""
from __future__ import annotations

import sys
from pathlib import Path

# src ディレクトリを path に追加（単体起動時）
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

import asyncio
import logging
import re
import threading
import time
import uuid
from typing import Any

import chromadb
from flask import Flask, Response, jsonify, request, abort

from agent.config import (
    configure_env_from_settings,
    get_embedding_config,
    get_knowledge_base_config,
    load_settings,
)
from agent.schemas import (
    Failure,
    KnowledgeSearchHit,
    KnowledgeSearchInput,
    KnowledgeSearchValue,
    KnowledgeUpsertInput,
    KnowledgeUpsertValue,
    Success,
)

_CHILD = "children"
_PARENT = "parents"
# 会話ドキュメントの本文セクションの見出し（knowledge_register.build_document_body が付ける）。
# 検索用の分割で「要約」と「本文」を切り分けるのに使う。
_BODY_MARK = "### 会話の本文"
# 長期記憶（旧 SQLite long_memory.db を統合）専用 collection。
# events / string_items（visited, user, agent, promises 等）/ records（working: todos, goals, context）/ procedural / meta
# すべてのフィールドが metadata の category で識別される。
_LONG_MEMORY = "long_memory"


class KnowledgeBase:
    """Chroma 永続化 + Gemini Embedding API を 1 プロセスで共有する容れ物。"""

    def __init__(
        self,
        path: str,
        model_name: str,
        api_key: str,
        chunk_size_child: int = 256,
    ) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._children = self._client.get_or_create_collection(_CHILD)
        self._parents = self._client.get_or_create_collection(_PARENT)
        # 旧 SQLite long_memory.db を統合した collection
        self._long_memory = self._client.get_or_create_collection(_LONG_MEMORY)
        self._model_name = model_name
        self._api_key = api_key
        self._chunk_size_child = chunk_size_child
        # BM25 キャッシュ（upsert/delete で無効化）
        self._bm25_cache: tuple[Any, list[str], list[dict]] | None = None
        self._bm25_lock = threading.Lock()
        self._bm25_dirty = True
        self._tagger: Any | None = None
        # 埋め込み API クライアント（遅延生成して**使い回す**。理由は _get_genai_client）
        self._genai_client: Any | None = None
        self._genai_lock = threading.Lock()

    def _get_genai_client(self) -> Any:
        """埋め込み API のクライアントを返す（初回だけ生成して以後は使い回す）。

        以前は `_embed` がテキスト1件ごとに `genai.Client` を作り直しており、
        **1回あたり約1.0秒を接続確立に費やしていた**（実測: 毎回生成 1.30秒 /
        使い回し 0.31秒・2026-08-26）。gemini-embedding-2 はバッチ非対応で
        1テキストずつ呼ぶため、再インデックス 1,398 件では約23分がこの無駄だった。
        """
        if self._genai_client is None:
            with self._genai_lock:
                if self._genai_client is None:      # ロック取得中に他スレッドが作った場合
                    from google import genai
                    self._genai_client = genai.Client(api_key=self._api_key)
        return self._genai_client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # gemini-embedding-2 は contents にリストを渡しても埋め込みを1件しか返さない
        # （バッチ非対応）。件数を保証するため1テキストずつ呼び出す。
        client = self._get_genai_client()
        vectors: list[list[float]] = []
        for text in texts:
            result = client.models.embed_content(
                model=self._model_name,
                contents=text,
            )
            vectors.append(list(result.embeddings[0].values))
        return vectors

    def _get_tagger(self) -> Any:
        if self._tagger is None:
            import MeCab
            self._tagger = MeCab.Tagger("-Owakati")
        return self._tagger

    def _tokenize(self, text: str) -> list[str]:
        return self._get_tagger().parse(text).strip().split()

    def invalidate_bm25(self) -> None:
        with self._bm25_lock:
            self._bm25_dirty = True

    def _get_bm25(self) -> tuple[Any, list[str], list[dict]]:
        """BM25 インデックスを返す。dirty なら全子チャンクを再取得して再構築。"""
        with self._bm25_lock:
            if not self._bm25_dirty and self._bm25_cache is not None:
                return self._bm25_cache

            result = self._children.get(include=["documents", "metadatas"])
            ids = result.get("ids") or []
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []

            if not ids:
                self._bm25_cache = (None, [], [])
                self._bm25_dirty = False
                return self._bm25_cache

            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize(d) for d in docs]
            bm25 = BM25Okapi(tokenized)
            self._bm25_cache = (bm25, ids, metas)
            self._bm25_dirty = False
            return self._bm25_cache


def _split_turns(text: str, chunk_size: int) -> list[str]:
    """発話/ターン単位に分割し、chunk_size 文字を超えたらチャンクを確定する。

    - 新形式（正式JSON配列・2026-07-31 以降。utils/conversation_log.py 参照）:
      配列の各発話レコードを 1 行のコンパクト JSON にして単位とする
    - 旧形式（疑似JSON・既存データ）: } \n\n { を区切りとして各ターンを抽出（従来どおり）
    - 単位を順に蓄積し、累積文字数が chunk_size を超えたらそのチャンクを確定する
    - 会話が短い場合は複数単位がまとめて 1 チャンクになる

    会話ドキュメントは `### 会話の要約 … ### 会話の本文 …` の形で渡ってくる。
    **全文のまま判定すると is_new_format が False になり**（先頭が `[` ではないため）、
    新形式が旧形式の正規表現へ落ちて一切分割されない。要約を独立した 1 チャンクとして切り出し、
    本文だけを発話単位の分割へ回す（2026-08-26 修正・要約は会話全体を表す単位として単独で持つ）。
    """
    from utils import conversation_log

    if _BODY_MARK in text:
        head, body = text.split(_BODY_MARK, 1)
        heads = [head.strip()] if head.strip() else []
        return heads + _split_turns(body.strip(), chunk_size)

    if conversation_log.is_new_format(text) and text.strip():
        view = conversation_log.to_llm_view(text)
        turns = [ln for ln in view.splitlines() if ln.strip()]
        if not turns:
            return [text]
    else:
        parts = re.split(r'\}\s*\n\s*\n\s*\{', text)
        turns = []
        for i, part in enumerate(parts):
            t = part.strip()
            if not t:
                continue
            if i > 0:
                t = '{' + t
            if i < len(parts) - 1:
                t = t + '}'
            turns.append(t)

    if not turns:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for turn in turns:
        current.append(turn)
        current_len += len(turn)
        if current_len >= chunk_size:
            chunks.append('\n\n'.join(current))
            current = []
            current_len = 0

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


async def run_knowledge_upsert(
    kb: KnowledgeBase, input_: KnowledgeUpsertInput
) -> Success[KnowledgeUpsertValue] | Failure:
    """source が既登録ならメタデータのみ更新（re-embed なし）、未登録なら新規挿入。"""
    try:
        common_meta = {
            "source": input_.source,
            "registered_at": input_.registered_at,
            "document_title": input_.document_title,
        }

        existing = kb._parents.get(where={"source": input_.source})
        existing_parent_ids: list[str] = existing.get("ids") or []

        if existing_parent_ids:
            # --- メタデータのみ更新（re-embed なし） ---
            kb._parents.update(
                ids=existing_parent_ids,
                metadatas=[common_meta] * len(existing_parent_ids),
            )

            existing_children = kb._children.get(
                where={"source": input_.source}, include=["metadatas"]
            )
            existing_child_ids: list[str] = existing_children.get("ids") or []
            existing_child_metas: list[dict] = existing_children.get("metadatas") or []

            if existing_child_ids:
                # parent_id を保持しつつ共通メタを上書き
                updated_child_metas = [
                    {**m, **common_meta} for m in existing_child_metas
                ]
                kb._children.update(
                    ids=existing_child_ids,
                    metadatas=updated_child_metas,
                )

            kb.invalidate_bm25()
            n_p = len(existing_parent_ids)
            n_c = len(existing_child_ids)
            return Success[KnowledgeUpsertValue](
                value=KnowledgeUpsertValue(parent_count=n_p, child_count=n_c),
                side_effects=[f"metadata_updated(parents={n_p}, children={n_c})"],
            )

        # --- 新規挿入（埋め込み + 保存） ---
        parent_id = f"p_{uuid.uuid4().hex[:16]}"
        kb._parents.upsert(
            ids=[parent_id],
            documents=[input_.document_body],
            metadatas=[common_meta],
        )

        child_texts = _split_turns(input_.document_body, kb._chunk_size_child)
        child_ids = [f"c_{uuid.uuid4().hex[:16]}" for _ in child_texts]
        child_metas = [{**common_meta, "parent_id": parent_id} for _ in child_texts]
        kb._children.upsert(
            ids=child_ids,
            documents=child_texts,
            embeddings=kb._embed(child_texts),
            metadatas=child_metas,
        )

        kb.invalidate_bm25()
        n_c = len(child_texts)
        return Success[KnowledgeUpsertValue](
            value=KnowledgeUpsertValue(parent_count=1, child_count=n_c),
            side_effects=[f"chroma_upsert(parents=1, children={n_c})"],
        )
    except Exception as exc:
        return Failure(reason=f"upsert_failed: {exc!r}", retryable=False)


_RRF_K = 60
_VEC_WEIGHT = 0.7   # ベクトル検索の重み（意味・文脈）
_BM25_WEIGHT = 0.3  # BM25の重み（キーワード一致）


def _rrf_score(cid: str, vec_rank: dict[str, int], bm25_rank: dict[str, int]) -> float:
    s = 0.0
    if cid in vec_rank:
        s += _VEC_WEIGHT / (_RRF_K + vec_rank[cid])
    if cid in bm25_rank:
        s += _BM25_WEIGHT / (_RRF_K + bm25_rank[cid])
    return s


# 検索が**この秒数を超えたときだけ**内訳を1行出す。平常時は何も出さない。
# 実測（2026-08-26）: 温まった検索 1.1〜1.2秒 / 再起動直後の初回 2.6秒 /
# VDB 単体の冷えた初回 3.6秒。5秒ならこれらでは出ず、異常時だけ出る。
_SLOW_SEARCH_SEC = 5.0


class _SearchTiming:
    """検索の各段の所要時間を測り、遅かったときだけ内訳を出す。

    **なぜ実時間と CPU 時間の両方を測るか**: 「計算していて遅い」のか
    「待たされて遅い」のかを区別するため。2026-08-26 に検索が 60 秒で
    タイムアウトしたとき、AI は「BM25 の初回構築が主因」と推測したが実際は
    **CPU 逼迫**（他の処理がコアを埋めていた）だった。実時間に対して CPU 時間が
    極端に短ければ、その時点で「自分は計算していない＝外部要因」と分かる。
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._c0 = time.process_time()
        self._last = self._t0
        self._stages: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self._stages.append((name, now - self._last))
        self._last = now

    def report(self, *, bm25_built: bool, hits: int) -> None:
        total = time.perf_counter() - self._t0
        if total < _SLOW_SEARCH_SEC:
            return                                  # 平常時はログを汚さない
        cpu = time.process_time() - self._c0
        parts = " / ".join(
            f"{n} {sec * 1000:,.0f}ms{'(構築)' if n == 'bm25' and bm25_built else ''}"
            for n, sec in self._stages
        )
        load = ""
        try:                                        # psutil は既存依存（無くても動く）
            import psutil
            load = f" / 全体CPU {psutil.cpu_percent(interval=None):.0f}% {psutil.cpu_count()}コア"
        except Exception:                           # noqa: BLE001
            pass
        print(f"[TIMING] vdb_search 遅延 {total * 1000:,.0f}ms "
              f"(CPU時間 {cpu * 1000:,.0f}ms{load}, hits={hits}): {parts}")


async def run_knowledge_search(
    kb: KnowledgeBase, input_: KnowledgeSearchInput
) -> Success[KnowledgeSearchValue] | Failure:
    try:
        top_k = input_.top_k
        _t = _SearchTiming()

        # --- ベクトル検索 ---
        q_vec = kb._embed([input_.query])
        _t.mark("embed")
        vec_res = kb._children.query(query_embeddings=q_vec, n_results=top_k)
        vec_ids: list[str] = (vec_res.get("ids") or [[]])[0]
        vec_distances: list[float] = (vec_res.get("distances") or [[]])[0]
        # child_id → コサイン類似度（0-1）
        # chromadb デフォルトは L2 距離（0-2）、正規化済みベクトルなので cosine_sim = 1 - L2/2
        cid_to_vec_score: dict[str, float] = {
            cid: max(0.0, 1.0 - float(d) / 2.0) for cid, d in zip(vec_ids, vec_distances)
        }

        _t.mark("vector")

        # --- BM25 検索 ---
        bm25_was_dirty = kb._bm25_dirty
        bm25, bm25_all_ids, _ = kb._get_bm25()
        cid_to_bm25_score: dict[str, float] = {}
        if bm25 is not None:
            q_tokens = kb._tokenize(input_.query)
            raw_scores = bm25.get_scores(q_tokens)
            max_raw = max(raw_scores) if len(raw_scores) > 0 else 1.0
            bm25_ranked_idx = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:top_k]
            bm25_top_ids = [bm25_all_ids[i] for i in bm25_ranked_idx]
            # child_id → 正規化BM25スコア（0-1）
            cid_to_bm25_score = {
                bm25_all_ids[i]: (raw_scores[i] / max_raw if max_raw > 0 else 0.0)
                for i in bm25_ranked_idx
            }
        else:
            bm25_top_ids = []

        _t.mark("bm25")

        # --- RRF 統合 ---
        vec_rank = {cid: r for r, cid in enumerate(vec_ids)}
        bm25_rank = {cid: r for r, cid in enumerate(bm25_top_ids)}

        # 両リストの和集合（順序保持で重複除去）
        seen: set[str] = set()
        candidates: list[str] = []
        for cid in vec_ids + bm25_top_ids:
            if cid not in seen:
                seen.add(cid)
                candidates.append(cid)

        candidates.sort(key=lambda c: _rrf_score(c, vec_rank, bm25_rank), reverse=True)
        candidates = candidates[:top_k]

        # 子チャンクのメタデータを取得して親 ID を解決
        got = kb._children.get(ids=candidates, include=["metadatas"])
        got_ids = got.get("ids") or []
        got_metas = got.get("metadatas") or []
        id_to_meta = {cid: m for cid, m in zip(got_ids, got_metas)}

        parent_order: list[str] = []
        parent_best: dict[str, float] = {}
        parent_vec: dict[str, float] = {}   # parent_id → 子チャンク最良ベクトルスコア
        parent_bm25: dict[str, float] = {}  # parent_id → 子チャンク最良BM25スコア
        for cid in candidates:
            md = id_to_meta.get(cid) or {}
            pid = md.get("parent_id")
            if not pid:
                continue
            score = _rrf_score(cid, vec_rank, bm25_rank)
            if pid not in parent_best or score > parent_best[pid]:
                parent_best[pid] = score
            if cid in cid_to_vec_score:
                vs = cid_to_vec_score[cid]
                if pid not in parent_vec or vs > parent_vec[pid]:
                    parent_vec[pid] = vs
            if cid in cid_to_bm25_score:
                bs = cid_to_bm25_score[cid]
                if pid not in parent_bm25 or bs > parent_bm25[pid]:
                    parent_bm25[pid] = bs
            if pid not in parent_order:
                parent_order.append(pid)

        if not parent_order:
            _t.report(bm25_built=bm25_was_dirty, hits=0)
            return Success[KnowledgeSearchValue](value=KnowledgeSearchValue(hits=[]))

        # chromadb 1.x は get() の返却順が内部ID順のため、ID→doc/meta で dict マッピング
        p_got = kb._parents.get(ids=parent_order, include=["documents", "metadatas"])
        p_ids = p_got.get("ids") or []
        p_docs = p_got.get("documents") or []
        p_metas = p_got.get("metadatas") or []
        pid_to_doc = {pid: doc for pid, doc in zip(p_ids, p_docs)}
        pid_to_meta = {pid: meta for pid, meta in zip(p_ids, p_metas)}
        hits = [
            KnowledgeSearchHit(
                parent_id=pid,
                parent_text=pid_to_doc.get(pid, ""),
                score=parent_best[pid],
                vec_score=parent_vec.get(pid),
                bm25_score=parent_bm25.get(pid),
                metadata=pid_to_meta.get(pid, {}),
            )
            for pid in parent_order
            if pid in pid_to_doc
        ]
        _t.mark("parents")
        _t.report(bm25_built=bm25_was_dirty, hits=len(hits))
        return Success[KnowledgeSearchValue](value=KnowledgeSearchValue(hits=hits))
    except Exception as exc:
        return Failure(reason=f"search_failed: {exc!r}", retryable=True)


# =============================================================================
# Long Memory（旧 SQLite long_memory.db を統合した新規 collection）
# =============================================================================
# データ構造:
#   id        ... 安定 ID（例: events なら "E20260520-01"、string_items なら "S{uuid}"、
#                 records なら "R-working-todos"、meta なら "M-version"）
#   document  ... 検索可能な文字列。事実本文をテキスト化（埋め込み・BM25 対象）
#   metadata  ... {
#       "category":      str,   # 'events' | 'visited' | 'user' | 'agent' | 'promises' |
#                               # 'agent_rules' | 'user_habits' | 'todos' | 'goals' |
#                               # 'context' | 'meta' | 'procedural' 等
#       "type":          str,   # 'event' | 'string_item' | 'record' | 'meta'
#       "data_json":     str,   # 原本データ（JSON 文字列）。検索結果の構造化返却に使う
#       "always_in_context": bool, # True なら常時 context 投入対象
#       "registered_at": str,
#       # 任意追加: date / importance / source 等
#   }
#
# 用途:
#   - 常時 context（working / semantic.user/agent/promises / meta）は
#     `where={always_in_context: True}` で全件取得して JSON 組み立て
#   - 想起 Tool（events / visited / procedural）は `query_long_memory` でベクトル検索
# =============================================================================


def long_memory_upsert(kb: KnowledgeBase, items: list[dict]) -> dict:
    """items: [{id, document, metadata}, ...] を long_memory collection に upsert する。

    re-embed コストを抑えるため、既存 id の document が同一ならメタデータのみ更新する。
    """
    if not items:
        return {"upserted": 0, "metadata_updated": 0}

    ids = [it["id"] for it in items]
    docs = [it["document"] for it in items]
    metas = [it.get("metadata", {}) for it in items]

    # 既存ドキュメントを取得して、変わっていない id をメタデータのみ更新側に振り分ける
    existing = kb._long_memory.get(ids=ids, include=["documents"])
    existing_ids: list[str] = existing.get("ids") or []
    existing_docs: list[str] = existing.get("documents") or []
    id_to_existing_doc = {i: d for i, d in zip(existing_ids, existing_docs)}

    meta_only_idx: list[int] = []
    new_or_changed_idx: list[int] = []
    for i, (id_, doc) in enumerate(zip(ids, docs)):
        if id_ in id_to_existing_doc and id_to_existing_doc[id_] == doc:
            meta_only_idx.append(i)
        else:
            new_or_changed_idx.append(i)

    if meta_only_idx:
        kb._long_memory.update(
            ids=[ids[i] for i in meta_only_idx],
            metadatas=[metas[i] for i in meta_only_idx],
        )

    if new_or_changed_idx:
        new_ids = [ids[i] for i in new_or_changed_idx]
        new_docs = [docs[i] for i in new_or_changed_idx]
        new_metas = [metas[i] for i in new_or_changed_idx]
        kb._long_memory.upsert(
            ids=new_ids,
            documents=new_docs,
            embeddings=kb._embed(new_docs),
            metadatas=new_metas,
        )

    return {
        "upserted": len(new_or_changed_idx),
        "metadata_updated": len(meta_only_idx),
    }


def long_memory_search(kb: KnowledgeBase, query: str, top_k: int = 10,
                        category: str | None = None,
                        always_in_context: bool | None = None) -> list[dict]:
    """long_memory collection をベクトル検索する。

    category 指定で metadata フィルタを併用可能（events / visited / procedural 等で絞り込み）。
    always_in_context に False を渡すと、**常時 context に入っている記憶を検索対象から外す**
    （2026-08-26 追加）。想起は「いま見えていないものを思い出す」行為なので、既に見えている
    ものが枠を埋めると本来引きたい記憶が押し出される（実測: category 未指定のとき
    「マスターの好きなもの」で 8件中8件が常時投入済みのレコードだった）。

    戻り値: [{id, document, metadata, score}, ...] スコア降順。
    """
    conds: list[dict] = []
    if category:
        conds.append({"category": category})
    if always_in_context is not None:
        conds.append({"always_in_context": always_in_context})
    if not conds:
        where = None
    elif len(conds) == 1:
        where = conds[0]
    else:
        where = {"$and": conds}
    q_vec = kb._embed([query])
    res = kb._long_memory.query(
        query_embeddings=q_vec, n_results=top_k, where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: list[dict] = []
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        # normalize_embeddings なら L2(0-2) → cos_sim = 1 - L2/2
        score = max(0.0, 1.0 - float(dist) / 2.0)
        hits.append({"id": id_, "document": doc, "metadata": meta or {}, "score": score})
    return hits


def long_memory_get_by_metadata(kb: KnowledgeBase, where: dict) -> list[dict]:
    """metadata フィルタで long_memory ドキュメントを全件取得する。

    例: where={"always_in_context": True} で常時投入セクションを一括取得。
    戻り値: [{id, document, metadata}, ...] 順序不定。
    """
    res = kb._long_memory.get(where=where, include=["documents", "metadatas"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return [
        {"id": id_, "document": doc, "metadata": meta or {}}
        for id_, doc, meta in zip(ids, docs, metas)
    ]


def long_memory_delete(kb: KnowledgeBase, ids: list[str]) -> int:
    """指定 id を long_memory から削除する。戻り値: 削除した件数。"""
    if not ids:
        return 0
    kb._long_memory.delete(ids=ids)
    return len(ids)


app = Flask(__name__)

_kb: KnowledgeBase | None = None


def initialize_vdb_service(settings: dict, root: Path) -> None:
    global _kb
    kb_cfg = get_knowledge_base_config(settings)
    emb_cfg = get_embedding_config(settings)
    _kb = KnowledgeBase(
        path=str(root / kb_cfg["path"]),
        model_name=emb_cfg["model_name"],
        api_key=emb_cfg["api_key"],
        chunk_size_child=kb_cfg["chunk_size_child"],
    )
    print(f"VDBサービスを初期化しました")
    print(f"  - Chroma path : {root / kb_cfg['path']}")
    print(f"  - 埋め込みモデル: {emb_cfg['model_name']}")


def _get_kb() -> KnowledgeBase:
    if _kb is None:
        abort(503, "VDBサービスが初期化されていません")
    return _kb


# ── ヘルパー ──────────────────────────────────────────────
def _run(coro):
    """同期コンテキストから async 関数を実行する。"""
    return asyncio.run(coro)


# ── エンドポイント ─────────────────────────────────────────

@app.post("/vdb/upsert")
def vdb_upsert():
    """
    親子チャンク投入。

    Request JSON:
      {
        "document_title": "session_20260419_073819",
        "document_body": "...",
        "source": "session_20260419_073819",
        "registered_at": "2026-04-24T12:00:00+09:00"
      }

    Response JSON:
      {"status": "success", "parent_count": 5, "child_count": 20}
      {"status": "failure", "reason": "..."}
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        abort(400, "JSON ボディが必要です")
    required = {"document_title", "document_body", "source", "registered_at"}
    missing = required - data.keys()
    if missing:
        abort(400, f"フィールドが不足しています: {missing}")

    try:
        inp = KnowledgeUpsertInput(**{k: data[k] for k in required})
    except Exception as e:
        abort(400, str(e))

    result = _run(run_knowledge_upsert(_get_kb(), inp))
    if result.status == "failure":
        return jsonify(status="failure", reason=result.reason), 500
    return jsonify(
        status="success",
        parent_count=result.value.parent_count,
        child_count=result.value.child_count,
        side_effects=result.side_effects,
    )


@app.post("/vdb/search")
def vdb_search():
    """
    意味検索。

    Request JSON:
      {"query": "サイクリング", "top_k": 10}

    Response JSON:
      {"status": "success", "hits": [{"parent_id": "...", "parent_text": "...", "score": 0.9, "metadata": {...}}, ...]}
    """
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        abort(400, "query フィールドが必要です")
    top_k = min(int(data.get("top_k", 10)), 30)

    result = _run(run_knowledge_search(_get_kb(), KnowledgeSearchInput(query=query, top_k=top_k)))
    if result.status == "failure":
        return jsonify(status="failure", reason=result.reason), 500
    return jsonify(status="success", hits=[h.model_dump() for h in result.value.hits])


@app.get("/vdb/stats")
def vdb_stats():
    """parents / children の件数を返す。"""
    kb = _get_kb()
    return jsonify(parents=kb._parents.count(), children=kb._children.count())


@app.get("/vdb/sources")
def vdb_sources():
    """登録済みソース一覧と件数を返す。"""
    kb = _get_kb()
    result = kb._parents.get(include=["metadatas"])
    metas = result.get("metadatas") or []
    counts: dict[str, int] = {}
    for m in metas:
        src = (m or {}).get("source", "")
        counts[src] = counts.get(src, 0) + 1
    sources = [{"source": k, "count": v} for k, v in sorted(counts.items())]
    return jsonify(sources)


@app.get("/vdb/recent")
def vdb_recent():
    """登録日時が最新の親チャンクを N 件返す（古い順）。

    旧 Dify「直近5会話取得コード」相当。1 source = 1 parent 前提だが、
    将来的に複数 parent / source となるケースに備えて source 単位で結合する。

    Query params:
      limit (default 5, max 20)

    Response JSON:
      {
        "documents": [
          {
            "時系列順": 1,
            "データ登録日": "2026-04-19 07:38",
            "本文": "### 会話の要約\\n...\\n### 会話の本文\\n..."
          },
          ...
        ]
      }
    """
    from datetime import datetime as _dt

    kb = _get_kb()
    limit = min(max(1, int(request.args.get("limit", 5))), 20)

    result = kb._parents.get(include=["documents", "metadatas"])
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    # source ごとに registered_at とドキュメント本文を集約
    by_source: dict[str, dict[str, Any]] = {}
    for _pid, doc, meta in zip(ids, docs, metas):
        m = meta or {}
        source = (m.get("source") or "").strip()
        registered_at = (m.get("registered_at") or "").strip()
        if not source:
            continue
        if source not in by_source:
            by_source[source] = {"registered_at": registered_at, "docs": []}
        by_source[source]["docs"].append(doc or "")

    # registered_at 降順 → 上位 N → 昇順に並び替え（古い順で返却）
    desc = sorted(by_source.items(), key=lambda kv: kv[1]["registered_at"], reverse=True)
    top = desc[:limit]
    asc = sorted(top, key=lambda kv: kv[1]["registered_at"])

    def _format_date(raw: str) -> str:
        """registered_at を YYYY-MM-DD HH:MM に整形する。失敗時は raw を返す。"""
        if not raw:
            return ""
        s = raw.replace("Z", "")
        # タイムゾーン部分を素朴に除去（"+09:00" 等）
        for sep in ("+", "T"):
            pass
        if "+" in s:
            s = s.split("+", 1)[0]
        candidates = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ]
        for fmt in candidates:
            try:
                return _dt.strptime(s, fmt).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue
        return raw

    documents = []
    for idx, (_source, info) in enumerate(asc, start=1):
        documents.append({
            "時系列順": idx,
            "データ登録日": _format_date(info["registered_at"]),
            "本文": "\n".join(info["docs"]),
        })
    return jsonify(documents=documents)


@app.get("/vdb/parents")
def vdb_parents():
    """
    親チャンク一覧。

    Query params:
      source  (optional) ソースでフィルタ
      limit   (default 30, max 100)
      offset  (default 0)
    """
    kb = _get_kb()
    source: str | None = request.args.get("source")
    limit = min(int(request.args.get("limit", 30)), 100)
    offset = int(request.args.get("offset", 0))

    kwargs: dict = {"include": ["documents", "metadatas"]}
    if source:
        kwargs["where"] = {"source": source}

    result = kb._parents.get(**kwargs)
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    total = len(ids)
    sliced = list(zip(ids, docs, metas))[offset: offset + limit]
    items = [{"id": i, "text": d, "metadata": m} for i, d, m in sliced]
    return jsonify(total=total, items=items)


@app.get("/vdb/parent/<path:pid>")
def vdb_parent(pid: str):
    """親チャンク詳細と子チャンク一覧を返す。"""
    kb = _get_kb()
    got = kb._parents.get(ids=[pid], include=["documents", "metadatas"])
    docs = got.get("documents") or [""]
    metas = got.get("metadatas") or [{}]

    c_result = kb._children.get(
        where={"parent_id": pid}, include=["documents", "metadatas"]
    )
    c_ids = c_result.get("ids") or []
    c_docs = c_result.get("documents") or []
    children = [{"id": ci, "text": cd} for ci, cd in zip(c_ids, c_docs)]

    return jsonify(text=docs[0], metadata=metas[0], children=children)


@app.put("/vdb/parent/<path:pid>")
def vdb_parent_update(pid: str):
    """親テキスト更新 → 子を再分割・再ベクトル化。"""
    kb = _get_kb()
    data = request.get_json(force=True, silent=True) or {}
    new_text = data.get("text", "").strip()
    if not new_text:
        abort(400, "text フィールドが必要です")

    got = kb._parents.get(ids=[pid], include=["metadatas"])
    metas = got.get("metadatas") or [{}]
    meta = metas[0] or {}

    # 親を更新
    kb._parents.upsert(ids=[pid], documents=[new_text], metadatas=[meta])

    # 古い子を全削除
    old_children = kb._children.get(where={"parent_id": pid})
    old_ids = old_children.get("ids") or []
    if old_ids:
        kb._children.delete(ids=old_ids)

    # 子を再分割・再ベクトル化
    child_texts = _split_turns(new_text, _get_kb()._chunk_size_child)
    child_ids = [f"c_{uuid.uuid4().hex[:16]}" for _ in child_texts]
    child_metas = [{**meta, "parent_id": pid} for _ in child_texts]
    kb._children.upsert(
        ids=child_ids,
        documents=child_texts,
        embeddings=kb._embed(child_texts),
        metadatas=child_metas,
    )
    kb.invalidate_bm25()
    return jsonify(status="success", child_count=len(child_texts))


@app.delete("/vdb/parent/<path:pid>")
def vdb_parent_delete(pid: str):
    """親チャンク + 紐づく子チャンクを全削除。"""
    kb = _get_kb()
    old_children = kb._children.get(where={"parent_id": pid})
    old_ids = old_children.get("ids") or []
    if old_ids:
        kb._children.delete(ids=old_ids)
    kb._parents.delete(ids=[pid])
    kb.invalidate_bm25()
    return jsonify(status="success", deleted_children=len(old_ids))


@app.put("/vdb/child/<path:cid>")
def vdb_child_update(cid: str):
    """子チャンクのテキスト更新 → 再ベクトル化。"""
    kb = _get_kb()
    data = request.get_json(force=True, silent=True) or {}
    new_text = data.get("text", "").strip()
    if not new_text:
        abort(400, "text フィールドが必要です")

    got = kb._children.get(ids=[cid], include=["metadatas"])
    metas = got.get("metadatas") or [{}]
    meta = metas[0] or {}

    kb._children.upsert(
        ids=[cid],
        documents=[new_text],
        embeddings=kb._embed([new_text]),
        metadatas=[meta],
    )
    kb.invalidate_bm25()
    return jsonify(status="success")


@app.delete("/vdb/child/<path:cid>")
def vdb_child_delete(cid: str):
    """子チャンク単体削除。"""
    kb = _get_kb()
    kb._children.delete(ids=[cid])
    kb.invalidate_bm25()
    return jsonify(status="success")


@app.delete("/vdb/source/<path:source>")
def vdb_source_delete(source: str):
    """source 単位で親+子を全削除。"""
    kb = _get_kb()
    p_result = kb._parents.get(where={"source": source})
    p_ids = p_result.get("ids") or []
    c_result = kb._children.get(where={"source": source})
    c_ids = c_result.get("ids") or []
    if c_ids:
        kb._children.delete(ids=c_ids)
    if p_ids:
        kb._parents.delete(ids=p_ids)
    kb.invalidate_bm25()
    return jsonify(status="success", deleted_parents=len(p_ids), deleted_children=len(c_ids))


# =============================================================================
# Long Memory エンドポイント（旧 SQLite long_memory.db を統合）
# =============================================================================


@app.post("/vdb/long_memory/upsert")
def vdb_long_memory_upsert():
    """長期記憶ドキュメントを一括 upsert する。

    Request JSON:
      {"items": [{"id": str, "document": str, "metadata": dict}, ...]}

    metadata 必須キー: category, type, always_in_context, data_json, registered_at
    任意キー: date, importance 等
    """
    kb = _get_kb()
    data = request.get_json(force=True, silent=True) or {}
    items = data.get("items") or []
    if not isinstance(items, list):
        abort(400, "items は配列である必要があります")
    result = long_memory_upsert(kb, items)
    return jsonify(status="success", **result)


@app.post("/vdb/long_memory/search")
def vdb_long_memory_search():
    """長期記憶をベクトル検索する。

    Request JSON:
      {"query": str, "top_k": int (default 10), "category": str (optional),
       "always_in_context": bool (optional)}

    always_in_context に false を渡すと**常時 context に入っている記憶を検索対象から外す**。
    想起（recall_long_memory）はこれを使う。省略時は従来どおり全件が対象。
    """
    kb = _get_kb()
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        abort(400, "query が必要です")
    top_k = min(int(data.get("top_k", 10)), 30)
    category = data.get("category") or None
    aic = data.get("always_in_context")
    hits = long_memory_search(
        kb, query, top_k=top_k, category=category,
        always_in_context=None if aic is None else bool(aic))
    return jsonify(status="success", hits=hits)


@app.get("/vdb/long_memory/by_category")
def vdb_long_memory_by_category():
    """metadata.category で全件取得する（常時 context 投入セクション用）。

    Query: ?category=working / user / agent / promises / meta / ...
           ?always_in_context=true  ← True/False のフィルタ
    複数指定時は AND 条件。
    """
    kb = _get_kb()
    where: dict = {}
    cat = request.args.get("category")
    if cat:
        where["category"] = cat
    aic = request.args.get("always_in_context")
    if aic is not None:
        where["always_in_context"] = (aic.lower() == "true")
    if not where:
        abort(400, "category または always_in_context のいずれかを指定してください")
    items = long_memory_get_by_metadata(kb, where)
    return jsonify(status="success", items=items)


@app.delete("/vdb/long_memory/<path:doc_id>")
def vdb_long_memory_delete(doc_id: str):
    """指定 id を long_memory から削除する。"""
    kb = _get_kb()
    n = long_memory_delete(kb, [doc_id])
    return jsonify(status="success", deleted=n)


@app.get("/vdb/long_memory/full_json")
def vdb_long_memory_full_json():
    """旧 long_memory_db_service の `/memory` 互換: 全 long_memory ドキュメントを
    旧 LongMemoryJSON 形式（meta/episodic/semantic/procedural/working の階層）で返す。

    メンテ画面の JSON ビュー用。
    """
    import json as _json
    kb = _get_kb()
    all_items: list[dict] = []
    for cat in ["meta", "events", "user", "agent", "promises", "visited",
                "procedural", "working"]:
        all_items.extend(long_memory_get_by_metadata(kb, where={"category": cat}))

    result: dict = {
        "meta": {},
        "episodic": {"events": []},
        "semantic": {"user": {}, "agent": {}, "promises": [], "visited": []},
        "procedural": {"habits": [], "rules": []},
        "working": {"todos": [], "goals": [], "context": ""},
    }
    for it in all_items:
        meta = it.get("metadata") or {}
        cat = meta.get("category")
        type_ = meta.get("type")
        try:
            orig = _json.loads(meta.get("data_json") or "{}")
        except _json.JSONDecodeError:
            continue
        if cat == "meta":
            result["meta"][orig.get("key", "")] = orig.get("value", "")
        elif cat == "events":
            try: inner = _json.loads(orig.get("data") or "{}")
            except _json.JSONDecodeError: inner = {}
            result["episodic"]["events"].append({
                "id": orig.get("id", ""), "start_dt": orig.get("start_dt", ""),
                "end_dt": orig.get("end_dt", ""), "place": orig.get("place", ""),
                "what": orig.get("what", ""), "importance": orig.get("importance", "中"),
                "points": inner.get("points", []), "feelings": inner.get("feelings", []),
                "who": inner.get("who", []),
            })
        elif type_ == "string_item":
            oc = meta.get("original_category", "other")
            bucket = result["semantic"]["user"] if cat == "user" else result["semantic"]["agent"]
            bucket.setdefault(oc, []).append(orig.get("value", ""))
        elif cat == "promises":
            try: d = _json.loads(orig.get("data") or "{}")
            except _json.JSONDecodeError: d = {}
            result["semantic"]["promises"].append(d)
        elif cat == "visited":
            try: d = _json.loads(orig.get("data") or "{}")
            except _json.JSONDecodeError: d = {}
            result["semantic"]["visited"].append(d)
        elif cat == "procedural":
            rt = meta.get("record_type", "")
            try: d = _json.loads(orig.get("data") or "{}")
            except _json.JSONDecodeError: d = {}
            if rt == "habit": result["procedural"]["habits"].append(d)
            elif rt == "rule": result["procedural"]["rules"].append(d)
        elif cat == "working":
            rt = meta.get("record_type", "")
            try: d = _json.loads(orig.get("data") or "{}")
            except _json.JSONDecodeError: d = {}
            if rt == "todo": result["working"]["todos"].append(d)
            elif rt == "goal": result["working"]["goals"].append(d)
            elif rt == "context":
                existing = result["working"]["context"]
                new = d.get("what", "")
                result["working"]["context"] = (existing + "\n" + new).strip() if existing else new

    return Response(_json.dumps(result, ensure_ascii=False, indent=2),
                    mimetype="application/json; charset=utf-8")


# ── GUI（メンテナンス画面）────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VDB Viewer</title>
<style>
  :root {
    --bg: #0f1117; --panel: #1a1d27; --border: #2a2d3e;
    --accent: #7c6af7; --accent2: #56c8f5;
    --text: #e2e8f0; --muted: #6b7280; --success: #34d399;
    --danger: #f87171; --warn: #fbbf24;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; height: 100vh; display: flex; flex-direction: column; }
  header { background: var(--panel); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
  .stats-badge { background: var(--border); border-radius: 20px; padding: 4px 12px; font-size: 12px; color: var(--muted); }
  .stats-badge span { color: var(--accent2); font-weight: 600; }
  .layout { flex: 1; display: grid; grid-template-columns: 280px 1fr; overflow: hidden; }
  .sidebar { border-right: 1px solid var(--border); overflow-y: auto; display: flex; flex-direction: column; }
  .sidebar-section { padding: 12px; border-bottom: 1px solid var(--border); }
  .sidebar-section h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 8px; }
  .search-box { display: flex; gap: 6px; }
  .search-box input { flex: 1; background: var(--border); border: 1px solid #3a3d4e; border-radius: 6px; padding: 7px 10px; color: var(--text); font-size: 13px; outline: none; }
  .search-box input:focus { border-color: var(--accent); }
  .btn { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 7px 14px; cursor: pointer; font-size: 13px; white-space: nowrap; }
  .btn:hover { opacity: .85; }
  .btn.secondary { background: var(--border); color: var(--text); }
  .btn.danger { background: #7f1d1d; color: #fca5a5; }
  .btn.danger:hover { background: #991b1b; }
  .btn.save { background: #14532d; color: #6ee7b7; }
  .btn.save:hover { background: #166534; }
  .source-list { list-style: none; }
  .source-item { padding: 7px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; gap: 4px; transition: background .15s; }
  .source-item:hover, .source-item.active { background: var(--border); color: var(--text); }
  .source-item .cnt { background: #2a2d3e; border-radius: 10px; padding: 1px 7px; font-size: 11px; flex-shrink: 0; }
  .source-item .del-btn { opacity: 0; font-size: 11px; padding: 1px 6px; flex-shrink: 0; }
  .source-item:hover .del-btn { opacity: 1; }
  .main { display: flex; flex-direction: column; overflow: hidden; }
  .toolbar { padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; }
  .toolbar .label { color: var(--muted); font-size: 12px; }
  .top-k-input { background: var(--border); border: 1px solid #3a3d4e; border-radius: 6px; padding: 5px 8px; color: var(--text); font-size: 12px; width: 54px; outline: none; }
  .content { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 0; overflow: hidden; }
  .list-pane, .detail-pane { overflow-y: auto; padding: 12px; }
  .list-pane { border-right: 1px solid var(--border); }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 8px; cursor: pointer; transition: border-color .15s; }
  .card:hover { border-color: var(--accent); }
  .card.selected { border-color: var(--accent); background: #1e2035; }
  .card-title { font-size: 13px; font-weight: 600; color: var(--accent2); margin-bottom: 4px; }
  .card-meta { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
  .card-score { display: inline-block; background: #1a2a1a; color: var(--success); border-radius: 4px; padding: 1px 6px; font-size: 11px; margin-bottom: 6px; }
  .card-text { font-size: 12px; color: #9ca3af; line-height: 1.6; white-space: pre-wrap; word-break: break-all; }
  .detail-header { font-size: 12px; color: var(--muted); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .detail-id { font-family: monospace; font-size: 11px; color: var(--muted); background: var(--border); padding: 2px 6px; border-radius: 4px; }
  .detail-body { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-bottom: 10px; }
  .detail-meta { font-size: 11px; color: var(--muted); margin-bottom: 8px; }
  textarea.edit-area { width: 100%; background: #12151f; border: 1px solid var(--accent); border-radius: 6px; color: var(--text); font-size: 12px; line-height: 1.7; font-family: inherit; padding: 10px; resize: vertical; min-height: 120px; outline: none; }
  .detail-body pre { white-space: pre-wrap; word-break: break-all; font-size: 12px; line-height: 1.7; font-family: inherit; }
  .children-header { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 12px 0 8px; display: flex; align-items: center; gap: 8px; }
  .child-card { background: #12151f; border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin-bottom: 6px; font-size: 12px; line-height: 1.6; }
  .child-card-header { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
  .child-id { font-family: monospace; font-size: 10px; color: var(--muted); flex: 1; }
  .child-body { white-space: pre-wrap; word-break: break-all; }
  .action-row { display: flex; gap: 6px; margin-top: 8px; }
  .loading { color: var(--muted); font-size: 13px; padding: 20px; text-align: center; }
  .empty { color: var(--muted); font-size: 13px; padding: 20px; text-align: center; }
  .error-msg { color: var(--danger); font-size: 12px; padding: 10px; background: #2a1515; border-radius: 6px; }
  .toast { position: fixed; bottom: 20px; right: 20px; background: #1a2a1a; border: 1px solid var(--success); color: var(--success); padding: 10px 16px; border-radius: 8px; font-size: 13px; z-index: 999; transition: opacity .3s; }
  .toast.error { background: #2a1515; border-color: var(--danger); color: var(--danger); }
  .search-result-count { font-size: 12px; color: var(--muted); padding: 4px 0 8px; }
  .mode-badge { font-size: 11px; padding: 3px 9px; border-radius: 10px; }
  .mode-browse { background: #1a2533; color: var(--accent2); }
  .mode-search { background: #1a2a1a; color: var(--success); }
  .pagination { display: flex; gap: 6px; align-items: center; padding: 8px 0 0; }
  .page-btn { background: var(--border); border: none; border-radius: 4px; padding: 4px 10px; color: var(--text); cursor: pointer; font-size: 12px; }
  .page-btn:disabled { opacity: .4; cursor: default; }
  .page-info { font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>🎵 VDB Viewer</h1>
  <div class="stats-badge">parents: <span id="stat-parents">-</span></div>
  <div class="stats-badge">children: <span id="stat-children">-</span></div>
  <div class="stats-badge">sources: <span id="stat-sources">-</span></div>
  <a href="/long_memory" style="margin-left:auto;color:var(--muted);font-size:12px;text-decoration:none">長期記憶ビューアへ →</a>
</header>
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-section">
      <h2>意味検索</h2>
      <div class="search-box">
        <input id="q" type="text" placeholder="検索クエリ..." onkeydown="if(event.key==='Enter')doSearch()">
        <button class="btn" onclick="doSearch()">検索</button>
      </div>
      <div style="margin-top:6px;display:flex;gap:6px;align-items:center">
        <span style="font-size:11px;color:var(--muted)">top_k:</span>
        <input class="top-k-input" id="top-k" type="number" value="10" min="1" max="30">
        <button class="btn secondary" onclick="clearSearch()" style="font-size:11px;padding:4px 10px">クリア</button>
      </div>
    </div>
    <div class="sidebar-section" style="flex:1;overflow-y:auto">
      <h2>ソース一覧</h2>
      <ul class="source-list" id="source-list"><li class="loading">読み込み中...</li></ul>
    </div>
  </div>
  <div class="main">
    <div class="toolbar">
      <span id="mode-badge" class="mode-badge mode-browse">BROWSE</span>
      <span class="label" id="toolbar-label">ソースを選択してください</span>
    </div>
    <div class="content">
      <div class="list-pane" id="list-pane"><div class="empty">左のサイドバーでソースを選ぶか、検索してください</div></div>
      <div class="detail-pane" id="detail-pane"><div class="empty">カードをクリックして詳細を表示</div></div>
    </div>
  </div>
</div>
<script>
let state = { mode: 'browse', source: null, offset: 0, total: 0, selectedId: null };
const PAGE = 30;

async function api(path, opts={}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function toast(msg, isError=false) {
  const el = document.createElement('div');
  el.className = 'toast' + (isError ? ' error' : '');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 2500);
}

async function refreshStats() {
  const stats = await api('/vdb/stats');
  document.getElementById('stat-parents').textContent = stats.parents;
  document.getElementById('stat-children').textContent = stats.children;
}

async function init() {
  const [stats, sources] = await Promise.all([api('/vdb/stats'), api('/vdb/sources')]);
  document.getElementById('stat-parents').textContent = stats.parents;
  document.getElementById('stat-children').textContent = stats.children;
  document.getElementById('stat-sources').textContent = sources.length;
  renderSourceList(sources, stats.parents);
}

function renderSourceList(sources, totalParents) {
  const ul = document.getElementById('source-list');
  ul.innerHTML = '';
  const allLi = document.createElement('li');
  allLi.className = 'source-item';
  allLi.innerHTML = `<span>すべて</span><span class="cnt">${totalParents}</span>`;
  allLi.onclick = () => browseSource(null, allLi);
  ul.appendChild(allLi);
  sources.forEach(s => {
    const li = document.createElement('li');
    li.className = 'source-item';
    li.innerHTML = `<span style="word-break:break-all;flex:1">${esc(s.source)}</span><span class="cnt">${s.count}</span><button class="btn danger del-btn" onclick="event.stopPropagation();deleteSource('${esc(s.source)}')">削除</button>`;
    li.onclick = () => browseSource(s.source, li);
    ul.appendChild(li);
  });
}

async function deleteSource(source) {
  if (!confirm(`"${source}" を完全に削除しますか？`)) return;
  try {
    const r = await api('/vdb/source/' + encodeURIComponent(source), { method: 'DELETE' });
    toast(`削除完了: parents=${r.deleted_parents}, children=${r.deleted_children}`);
    if (state.source === source) {
      state.source = null;
      document.getElementById('detail-pane').innerHTML = '<div class="empty">カードをクリックして詳細を表示</div>';
    }
    const [stats, sources] = await Promise.all([api('/vdb/stats'), api('/vdb/sources')]);
    document.getElementById('stat-parents').textContent = stats.parents;
    document.getElementById('stat-children').textContent = stats.children;
    document.getElementById('stat-sources').textContent = sources.length;
    renderSourceList(sources, stats.parents);
    if (state.source === null) document.getElementById('list-pane').innerHTML = '<div class="empty">ソースを選ぶか検索してください</div>';
    else await loadParents();
  } catch(e) { toast(e.message, true); }
}

async function browseSource(source, el) {
  document.querySelectorAll('.source-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');
  state = { ...state, mode: 'browse', source, offset: 0, selectedId: null };
  setModeBadge('browse');
  document.getElementById('detail-pane').innerHTML = '<div class="empty">カードをクリックして詳細を表示</div>';
  await loadParents();
}

async function loadParents() {
  const { source, offset } = state;
  document.getElementById('list-pane').innerHTML = '<div class="loading">読み込み中...</div>';
  const params = new URLSearchParams({ limit: PAGE, offset });
  if (source) params.set('source', source);
  const data = await api('/vdb/parents?' + params);
  state.total = data.total;
  setToolbarLabel(source ? `source: ${source}` : 'すべて', `${data.total} 件`);
  renderParentList(data.items, data.total, offset);
}

async function doSearch() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const top_k = parseInt(document.getElementById('top-k').value) || 10;
  document.querySelectorAll('.source-item').forEach(e => e.classList.remove('active'));
  state = { ...state, mode: 'search', selectedId: null };
  setModeBadge('search');
  document.getElementById('list-pane').innerHTML = '<div class="loading">検索中（初回はインデックス構築のため少し待ちます）...</div>';
  document.getElementById('detail-pane').innerHTML = '<div class="empty">カードをクリックして詳細を表示</div>';
  const t0 = performance.now();
  const data = await api('/vdb/search', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ query: q, top_k }) });
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  if (data.error) { document.getElementById('list-pane').innerHTML = `<div class="error-msg">${data.error}</div>`; return; }
  setToolbarLabel(`"${q}"`, `${data.hits.length} hits &nbsp;·&nbsp; ${elapsed}s`);
  renderSearchResults(data.hits, elapsed);
}

function clearSearch() {
  document.getElementById('q').value = '';
  state.mode = 'browse';
  setModeBadge('browse');
  document.getElementById('list-pane').innerHTML = '<div class="empty">ソースを選ぶか検索してください</div>';
  document.getElementById('detail-pane').innerHTML = '<div class="empty">カードをクリックして詳細を表示</div>';
  document.getElementById('toolbar-label').textContent = 'ソースを選択してください';
}

function renderParentList(items, total, offset) {
  const pane = document.getElementById('list-pane');
  if (!items.length) { pane.innerHTML = '<div class="empty">データなし</div>'; return; }
  let html = '';
  items.forEach(p => {
    const meta = p.metadata || {};
    html += `<div class="card" id="card-${p.id}" onclick="selectParent('${p.id}')">
      <div class="card-title">${esc(meta.document_title || p.id)}</div>
      <div class="card-meta">${esc(meta.source || '')} &nbsp;|&nbsp; ${esc(meta.registered_at || '')}</div>
      <div class="card-text">${esc((p.text||'').slice(0,160))}${(p.text||'').length>160?'…':''}</div>
    </div>`;
  });
  const pages = Math.ceil(total / PAGE);
  const cur = Math.floor(offset / PAGE);
  html += `<div class="pagination">
    <button class="page-btn" onclick="goPage(${cur-1})" ${cur===0?'disabled':''}>‹ 前</button>
    <span class="page-info">${cur+1} / ${pages} ページ（全${total}件）</span>
    <button class="page-btn" onclick="goPage(${cur+1})" ${cur>=pages-1?'disabled':''}>次 ›</button>
  </div>`;
  pane.innerHTML = html;
}

function renderSearchResults(hits, elapsed) {
  const pane = document.getElementById('list-pane');
  if (!hits.length) { pane.innerHTML = '<div class="empty">ヒットなし</div>'; return; }
  let html = `<div class="search-result-count">${hits.length} 件ヒット &nbsp;<span style="color:var(--warn)">⏱ ${elapsed}s</span></div>`;
  hits.forEach(h => {
    const meta = h.metadata || {};
    const rrf   = (h.score * 1000).toFixed(2);
    const vec   = h.vec_score  != null ? (h.vec_score  * 100).toFixed(1) + '%' : '-';
    const bm25  = h.bm25_score != null ? (h.bm25_score * 100).toFixed(1) + '%' : '-';
    html += `<div class="card" id="card-${h.parent_id}" onclick="selectParent('${h.parent_id}')">
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:5px">
        <span class="card-score" title="RRF統合スコア×1000">RRF: ${rrf}</span>
        <span class="card-score" style="background:#1a2533;color:var(--accent2)" title="ベクトル類似度（コサイン）">ベクトル: ${vec}</span>
        <span class="card-score" style="background:#1a2a1a;color:#86efac" title="BM25キーワードスコア（正規化）">BM25: ${bm25}</span>
      </div>
      <div class="card-title">${esc(meta.document_title || h.parent_id)}</div>
      <div class="card-meta">${esc(meta.source || '')} &nbsp;|&nbsp; ${esc(meta.registered_at || '')}</div>
      <div class="card-text">${esc((h.parent_text||'').slice(0,200))}${(h.parent_text||'').length>200?'…':''}</div>
    </div>`;
  });
  pane.innerHTML = html;
}

async function selectParent(id) {
  document.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));
  const el = document.getElementById('card-' + id);
  if (el) el.classList.add('selected');
  state.selectedId = id;
  await renderDetail(id);
}

async function renderDetail(id) {
  document.getElementById('detail-pane').innerHTML = '<div class="loading">読み込み中...</div>';
  const data = await api('/vdb/parent/' + encodeURIComponent(id));
  const meta = data.metadata || {};

  let html = `<div class="detail-header">
    <span class="detail-id">${esc(id)}</span>
    <button class="btn danger" style="font-size:11px;padding:3px 10px" onclick="deleteParent('${id}')">親ごと削除</button>
  </div>
  <div class="detail-body">
    <div class="detail-meta">
      <b>タイトル:</b> ${esc(meta.document_title||'-')} &nbsp;|&nbsp;
      <b>ソース:</b> ${esc(meta.source||'-')} &nbsp;|&nbsp;
      <b>登録日時:</b> ${esc(meta.registered_at||'-')}
    </div>
    <textarea class="edit-area" id="parent-text-${id}" rows="8">${esc(data.text||'')}</textarea>
    <div class="action-row">
      <button class="btn save" onclick="saveParent('${id}')">保存して再分割・再ベクトル化</button>
    </div>
  </div>`;

  if (data.children && data.children.length) {
    html += `<div class="children-header">子チャンク（${data.children.length} 件）</div>`;
    data.children.forEach((c, i) => {
      html += `<div class="child-card" id="child-card-${c.id}">
        <div class="child-card-header">
          <span class="child-id">[${i}] ${esc(c.id)}</span>
          <button class="btn danger" style="font-size:10px;padding:2px 8px" onclick="deleteChild('${c.id}','${id}')">削除</button>
        </div>
        <textarea class="edit-area" id="child-text-${c.id}" rows="4">${esc(c.text||'')}</textarea>
        <div class="action-row">
          <button class="btn save" style="font-size:11px;padding:4px 10px" onclick="saveChild('${c.id}','${id}')">保存・再ベクトル化</button>
        </div>
      </div>`;
    });
  } else {
    html += `<div class="empty" style="padding:10px 0">子チャンクなし</div>`;
  }
  document.getElementById('detail-pane').innerHTML = html;
}

async function saveParent(id) {
  const ta = document.getElementById('parent-text-' + id);
  if (!ta) return;
  const text = ta.value.trim();
  if (!text) { toast('テキストが空です', true); return; }
  try {
    const r = await api('/vdb/parent/' + encodeURIComponent(id), {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text })
    });
    toast(`保存完了（子チャンク ${r.child_count} 件に再分割）`);
    await refreshStats();
    await renderDetail(id);
  } catch(e) { toast(e.message, true); }
}

async function deleteParent(id) {
  if (!confirm('この親チャンクと紐づく子チャンクを全て削除しますか？')) return;
  try {
    const r = await api('/vdb/parent/' + encodeURIComponent(id), { method: 'DELETE' });
    toast(`削除完了（子 ${r.deleted_children} 件）`);
    document.getElementById('detail-pane').innerHTML = '<div class="empty">削除しました</div>';
    await refreshStats();
    if (state.mode === 'browse') await loadParents();
  } catch(e) { toast(e.message, true); }
}

async function saveChild(cid, pid) {
  const ta = document.getElementById('child-text-' + cid);
  if (!ta) return;
  const text = ta.value.trim();
  if (!text) { toast('テキストが空です', true); return; }
  try {
    await api('/vdb/child/' + encodeURIComponent(cid), {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text })
    });
    toast('保存・再ベクトル化完了');
  } catch(e) { toast(e.message, true); }
}

async function deleteChild(cid, pid) {
  if (!confirm('この子チャンクを削除しますか？')) return;
  try {
    await api('/vdb/child/' + encodeURIComponent(cid), { method: 'DELETE' });
    toast('子チャンク削除完了');
    await refreshStats();
    await renderDetail(pid);
  } catch(e) { toast(e.message, true); }
}

function goPage(page) { state.offset = page * PAGE; loadParents(); }
function setModeBadge(mode) {
  const el = document.getElementById('mode-badge');
  el.textContent = mode === 'search' ? 'SEARCH' : 'BROWSE';
  el.className = 'mode-badge ' + (mode === 'search' ? 'mode-search' : 'mode-browse');
}
function setToolbarLabel(label, sub='') {
  document.getElementById('toolbar-label').innerHTML = esc(label) + (sub ? ' — ' + sub : '');
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
init();
</script>
</body>
</html>"""


@app.get("/")
def viewer():
    return Response(_HTML, mimetype="text/html; charset=utf-8")


# ── Long Memory メンテ画面 ────────────────────────────────

_HTML_LONG_MEMORY_OLD_UNUSED = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long Memory Viewer</title>
<style>
  :root {
    --bg: #0f1117; --panel: #1a1d27; --border: #2a2d3e;
    --accent: #7c6af7; --accent2: #56c8f5;
    --text: #e2e8f0; --muted: #6b7280; --success: #34d399;
    --danger: #f87171; --warn: #fbbf24;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; height: 100vh; display: flex; flex-direction: column; }
  header { background: var(--panel); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
  header a { color: var(--muted); font-size: 12px; text-decoration: none; }
  header a:hover { color: var(--accent2); }
  .stats-badge { background: var(--border); border-radius: 20px; padding: 4px 12px; font-size: 12px; color: var(--muted); }
  .stats-badge span { color: var(--accent2); font-weight: 600; }
  .layout { flex: 1; display: grid; grid-template-columns: 260px 1fr; overflow: hidden; }
  .sidebar { border-right: 1px solid var(--border); overflow-y: auto; display: flex; flex-direction: column; }
  .sidebar-section { padding: 12px; border-bottom: 1px solid var(--border); }
  .sidebar-section h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 8px; }
  .search-box { display: flex; gap: 6px; }
  .search-box input { flex: 1; background: var(--border); border: 1px solid #3a3d4e; border-radius: 6px; padding: 7px 10px; color: var(--text); font-size: 13px; outline: none; }
  .search-box input:focus { border-color: var(--accent); }
  .btn { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 7px 14px; cursor: pointer; font-size: 13px; white-space: nowrap; }
  .btn:hover { opacity: .85; }
  .btn.secondary { background: var(--border); color: var(--text); }
  .btn.danger { background: #7f1d1d; color: #fca5a5; }
  .btn.danger:hover { background: #991b1b; }
  .cat-list { list-style: none; }
  .cat-item { padding: 7px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; gap: 4px; transition: background .15s; }
  .cat-item:hover, .cat-item.active { background: var(--border); color: var(--text); }
  .cat-item .cnt { background: #2a2d3e; border-radius: 10px; padding: 1px 7px; font-size: 11px; flex-shrink: 0; }
  .cat-item.aic-true { border-left: 3px solid var(--success); padding-left: 7px; }
  .cat-item.aic-false { border-left: 3px solid var(--warn); padding-left: 7px; }
  .legend { font-size: 10px; color: var(--muted); padding: 4px 0; }
  .legend span { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
  .main { display: flex; flex-direction: column; overflow: hidden; }
  .toolbar { padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; }
  .toolbar .label { color: var(--muted); font-size: 12px; }
  .content { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 0; overflow: hidden; }
  .list-pane, .detail-pane { overflow-y: auto; padding: 12px; }
  .list-pane { border-right: 1px solid var(--border); }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 8px; cursor: pointer; transition: border-color .15s; }
  .card:hover { border-color: var(--accent); }
  .card.selected { border-color: var(--accent); background: #1e2035; }
  .card-title { font-size: 13px; font-weight: 600; color: var(--accent2); margin-bottom: 4px; }
  .card-meta { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
  .card-score { display: inline-block; background: #1a2a1a; color: var(--success); border-radius: 4px; padding: 1px 6px; font-size: 11px; margin-right: 6px; }
  .card-text { font-size: 12px; color: #9ca3af; line-height: 1.6; white-space: pre-wrap; word-break: break-all; }
  /* DB view (table) */
  .view-toggle { display: inline-flex; gap: 0; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; margin-left: auto; }
  .view-toggle button { background: var(--panel); color: var(--muted); border: none; padding: 4px 12px; font-size: 11px; cursor: pointer; }
  .view-toggle button.active { background: var(--accent); color: #fff; }
  .db-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .db-table thead { background: var(--panel); position: sticky; top: 0; z-index: 1; }
  .db-table th { padding: 8px 10px; text-align: left; color: var(--accent2); font-weight: 600; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
  .db-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .db-table tr { cursor: pointer; transition: background .12s; }
  .db-table tr:hover { background: var(--panel); }
  .db-table tr.selected { background: #1e2035; }
  .db-table td.id-col { font-family: monospace; font-size: 11px; color: var(--accent2); white-space: nowrap; }
  .db-table td.date-col { font-family: monospace; font-size: 11px; color: var(--muted); white-space: nowrap; }
  .db-table td.score-col { font-family: monospace; font-size: 11px; color: var(--success); white-space: nowrap; }
  .db-table td.text-col { color: #9ca3af; max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .db-table td.text-col:hover { white-space: normal; overflow: visible; }
  .detail-header { font-size: 12px; color: var(--muted); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .detail-id { font-family: monospace; font-size: 11px; color: var(--muted); background: var(--border); padding: 2px 6px; border-radius: 4px; }
  .detail-body { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-bottom: 10px; }
  .detail-body pre { white-space: pre-wrap; word-break: break-all; font-size: 12px; line-height: 1.7; font-family: inherit; }
  .meta-table { font-size: 11px; color: var(--muted); border-collapse: collapse; width: 100%; margin-top: 8px; }
  .meta-table td { padding: 4px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .meta-table td:first-child { color: var(--accent2); width: 130px; }
  .meta-table td:last-child { word-break: break-all; font-family: monospace; font-size: 10px; }
  .action-row { display: flex; gap: 6px; margin-top: 8px; }
  .loading { color: var(--muted); font-size: 13px; padding: 20px; text-align: center; }
  .empty { color: var(--muted); font-size: 13px; padding: 20px; text-align: center; }
  .toast { position: fixed; bottom: 20px; right: 20px; background: #1a2a1a; border: 1px solid var(--success); color: var(--success); padding: 10px 16px; border-radius: 8px; font-size: 13px; z-index: 999; transition: opacity .3s; }
  .toast.error { background: #2a1515; border-color: var(--danger); color: var(--danger); }
  .mode-badge { font-size: 11px; padding: 3px 9px; border-radius: 10px; }
  .mode-browse { background: #1a2533; color: var(--accent2); }
  .mode-search { background: #1a2a1a; color: var(--success); }
</style>
</head>
<body>
<header>
  <h1>🧠 Long Memory Viewer</h1>
  <div class="stats-badge">total: <span id="stat-total">-</span></div>
  <div class="stats-badge">always_in_context: <span id="stat-aic">-</span></div>
  <div class="stats-badge">recall only: <span id="stat-recall">-</span></div>
  <a href="/" style="margin-left:auto">← 会話 VDB ビューアへ</a>
</header>
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-section">
      <h2>意味検索</h2>
      <div class="search-box">
        <input id="q" type="text" placeholder="検索クエリ..." onkeydown="if(event.key==='Enter')doSearch()">
        <button class="btn" onclick="doSearch()">検索</button>
      </div>
      <div style="margin-top:6px;display:flex;gap:6px;align-items:center">
        <span style="font-size:11px;color:var(--muted)">top_k:</span>
        <input id="top-k" type="number" value="10" min="1" max="30" style="background:var(--border);border:1px solid #3a3d4e;border-radius:6px;padding:5px 8px;color:var(--text);font-size:12px;width:54px;outline:none">
        <button class="btn secondary" onclick="clearSearch()" style="font-size:11px;padding:4px 10px">クリア</button>
      </div>
    </div>
    <div class="sidebar-section" style="flex:1;overflow-y:auto">
      <h2>カテゴリ</h2>
      <div class="legend"><span style="background:var(--success)"></span>常時投入  <span style="background:var(--warn)"></span>Tool 経由のみ</div>
      <ul class="cat-list" id="cat-list"><li class="loading">読み込み中...</li></ul>
    </div>
  </div>
  <div class="main">
    <div class="toolbar">
      <span id="mode-badge" class="mode-badge mode-browse">BROWSE</span>
      <span class="label" id="toolbar-label">カテゴリを選択してください</span>
      <div class="view-toggle">
        <button id="vt-card" class="active" onclick="setView('card')">Card</button>
        <button id="vt-db" onclick="setView('db')">DB</button>
      </div>
    </div>
    <div class="content">
      <div class="list-pane" id="list-pane"><div class="empty">左でカテゴリを選ぶか、検索してください</div></div>
      <div class="detail-pane" id="detail-pane"><div class="empty">左で項目を選択（JSON ビュー）</div></div>
    </div>
  </div>
</div>
<div id="toast" style="display:none" class="toast"></div>

<script>
// ── カテゴリ定義（VDB metadata.category と一致） ──
const CATEGORIES = [
  // [name, label, always_in_context]
  ['meta',      'meta（メタ情報）',                    true],
  ['user',      'user（マスター属性）',                true],
  ['agent',     'agent（ミク自身）',                   true],
  ['promises',  'promises（約束）',                    true],
  ['working',   'working（todos/goals/context）',      true],
  ['events',    'events（過去エピソード）',            false],
  ['visited',   'visited（訪問場所）',                 false],
  ['procedural','procedural（習慣・ルール）',          false],
];

let currentCategory = null;
let currentItems = [];
let selectedId = null;
let searchMode = false;
let currentView = 'card';  // 'card' | 'db'

// ── カテゴリ別 DB ビューカラム定義 ──
// 各カテゴリの代表的なフィールドをテーブル列として表示する。
// 各カラム: {key, label, extract(item) → string}
function extractInner(item) {
  // metadata.data_json をパースし、records は data フィールドの内側もパース
  const m = item.metadata || {};
  let parsed = {};
  let inner = {};
  try { parsed = JSON.parse(m.data_json || '{}'); } catch(e) {}
  try {
    if (typeof parsed.data === 'string') inner = JSON.parse(parsed.data || '{}');
    else if (typeof parsed.data === 'object' && parsed.data) inner = parsed.data;
  } catch(e) {}
  return {parsed, inner, m};
}

const DB_COLUMNS = {
  meta: [
    {key:'id', label:'id', cls:'id-col'},
    {key:'key', label:'key', get: it => extractInner(it).parsed.key || ''},
    {key:'value', label:'value', cls:'text-col', get: it => String(extractInner(it).parsed.value || '')},
  ],
  events: [
    {key:'id', label:'id', cls:'id-col'},
    {key:'date', label:'date', cls:'date-col', get: it => (it.metadata||{}).date || extractInner(it).parsed.start_dt || ''},
    {key:'place', label:'place', get: it => extractInner(it).parsed.place || ''},
    {key:'importance', label:'重要度', get: it => (it.metadata||{}).importance || extractInner(it).parsed.importance || ''},
    {key:'what', label:'what', cls:'text-col', get: it => extractInner(it).parsed.what || ''},
  ],
  visited: [
    {key:'id', label:'id', cls:'id-col'},
    {key:'name', label:'name', get: it => extractInner(it).inner.name || ''},
    {key:'date', label:'date', cls:'date-col', get: it => extractInner(it).inner.date || ''},
    {key:'note', label:'note', cls:'text-col', get: it => extractInner(it).inner.note || ''},
  ],
  procedural: [
    {key:'id', label:'id', cls:'id-col'},
    {key:'type', label:'type', get: it => (it.metadata||{}).record_type || ''},
    {key:'what', label:'what', cls:'text-col', get: it => extractInner(it).inner.what || ''},
    {key:'active', label:'active', get: it => String(extractInner(it).inner.active ?? '')},
    {key:'when', label:'when', get: it => {
      const w = extractInner(it).inner.when;
      return Array.isArray(w) ? w.join(', ') : String(w || '');
    }},
  ],
  user: [
    {key:'id', label:'id', cls:'id-col'},
    {key:'orig', label:'種別', get: it => (it.metadata||{}).original_category || ''},
    {key:'value', label:'value', cls:'text-col', get: it => extractInner(it).parsed.value || it.document || ''},
  ],
  agent: [
    {key:'id', label:'id', cls:'id-col'},
    {key:'orig', label:'種別', get: it => (it.metadata||{}).original_category || ''},
    {key:'value', label:'value', cls:'text-col', get: it => extractInner(it).parsed.value || it.document || ''},
  ],
  working: [
    {key:'id', label:'id', cls:'id-col'},
    {key:'type', label:'type', get: it => (it.metadata||{}).record_type || ''},
    {key:'detail', label:'内容', cls:'text-col', get: it => {
      const x = extractInner(it).inner;
      return x.title || x.what || x.content || JSON.stringify(x);
    }},
    {key:'status', label:'status', get: it => extractInner(it).inner.status || ''},
  ],
  promises: [
    {key:'id', label:'id', cls:'id-col'},
    {key:'content', label:'content', cls:'text-col', get: it => extractInner(it).inner.content || ''},
    {key:'status', label:'status', get: it => extractInner(it).inner.status || ''},
    {key:'date', label:'date', cls:'date-col', get: it => extractInner(it).inner.date || ''},
  ],
};

// 検索モード用の汎用カラム（カテゴリ横断結果向け）
const DB_COLUMNS_SEARCH = [
  {key:'score', label:'score', cls:'score-col', get: it => (typeof it.score === 'number' ? it.score.toFixed(3) : '')},
  {key:'id', label:'id', cls:'id-col'},
  {key:'cat', label:'cat', get: it => (it.metadata||{}).category || ''},
  {key:'date', label:'date', cls:'date-col', get: it => (it.metadata||{}).date || ''},
  {key:'doc', label:'document', cls:'text-col', get: it => it.document || ''},
];

function setView(v) {
  currentView = v;
  document.getElementById('vt-card').classList.toggle('active', v === 'card');
  document.getElementById('vt-db').classList.toggle('active', v === 'db');
  renderList();
}

function toast(msg, isErr) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast' + (isErr ? ' error' : '');
  t.style.display = 'block';
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => { t.style.display = 'none'; t.style.opacity = '1'; }, 300); }, 2500);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  if (r.status === 204) return null;
  return r.json();
}

// ── 起動: 各カテゴリの件数を取得してサイドバーを構築 ──
async function loadCategories() {
  const list = document.getElementById('cat-list');
  list.innerHTML = '<li class="loading">読み込み中...</li>';
  let total = 0, aic = 0, recall = 0;
  const html = [];
  for (const [cat, label, isAic] of CATEGORIES) {
    try {
      const data = await api(`/vdb/long_memory/by_category?category=${encodeURIComponent(cat)}`);
      const cnt = (data.items || []).length;
      total += cnt;
      if (isAic) aic += cnt; else recall += cnt;
      html.push(`<li class="cat-item ${isAic?'aic-true':'aic-false'}" data-cat="${cat}" onclick="selectCategory('${cat}')">
        <span>${label}</span><span class="cnt">${cnt}</span>
      </li>`);
    } catch (e) {
      html.push(`<li class="cat-item" data-cat="${cat}"><span>${label}</span><span class="cnt">err</span></li>`);
    }
  }
  list.innerHTML = html.join('');
  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-aic').textContent = aic;
  document.getElementById('stat-recall').textContent = recall;
}

async function selectCategory(cat) {
  currentCategory = cat;
  searchMode = false;
  document.getElementById('mode-badge').className = 'mode-badge mode-browse';
  document.getElementById('mode-badge').textContent = 'BROWSE';
  document.querySelectorAll('.cat-item').forEach(e => e.classList.toggle('active', e.dataset.cat === cat));
  document.getElementById('toolbar-label').textContent = `カテゴリ: ${cat}`;
  document.getElementById('list-pane').innerHTML = '<div class="loading">読み込み中...</div>';
  document.getElementById('detail-pane').innerHTML = '<div class="empty">左で項目を選択</div>';
  try {
    const data = await api(`/vdb/long_memory/by_category?category=${encodeURIComponent(cat)}`);
    currentItems = data.items || [];
    renderList();
  } catch (e) {
    document.getElementById('list-pane').innerHTML = '<div class="empty">読み込み失敗</div>';
    toast('取得失敗: ' + e.message, true);
  }
}

function _sortItems(items) {
  // 検索結果はスコア降順、それ以外は日付/id 降順
  if (searchMode) {
    return [...items].sort((a, b) => (b.score || 0) - (a.score || 0));
  }
  return [...items].sort((a, b) => {
    const da = (a.metadata && a.metadata.date) || '';
    const db = (b.metadata && b.metadata.date) || '';
    if (da !== db) return db.localeCompare(da);
    return (b.id || '').localeCompare(a.id || '');
  });
}

function renderList() {
  const pane = document.getElementById('list-pane');
  if (!currentItems.length) {
    pane.innerHTML = '<div class="empty">該当アイテムなし</div>';
    return;
  }
  const sorted = _sortItems(currentItems);
  if (currentView === 'db') {
    renderDbView(pane, sorted);
  } else {
    renderCardView(pane, sorted);
  }
}

function renderCardView(pane, sorted) {
  pane.innerHTML = sorted.map(it => {
    const m = it.metadata || {};
    const date = m.date || '';
    const imp = m.importance ? `<span class="card-meta">importance: ${m.importance}</span>` : '';
    const score = (typeof it.score === 'number') ? `<span class="card-score">score: ${it.score.toFixed(3)}</span>` : '';
    const cat = m.category ? `<span class="card-meta">cat: ${m.category}</span>` : '';
    return `<div class="card${it.id===selectedId?' selected':''}" onclick="selectItem('${escapeJs(it.id)}')">
      <div class="card-title">${escapeHtml(it.id)}</div>
      <div>${score}${cat}${date?'<span class="card-meta"> '+escapeHtml(date)+'</span>':''}${imp}</div>
      <div class="card-text">${escapeHtml((it.document||'').substring(0, 250))}${(it.document||'').length>250?'…':''}</div>
    </div>`;
  }).join('');
}

function renderDbView(pane, sorted) {
  // カラム選定: 検索モードでカテゴリ未指定なら汎用列、それ以外はカテゴリ専用列
  let cols;
  if (searchMode && !currentCategory) {
    cols = DB_COLUMNS_SEARCH;
  } else {
    const cat = currentCategory || (sorted[0] && sorted[0].metadata && sorted[0].metadata.category);
    cols = DB_COLUMNS[cat] || DB_COLUMNS_SEARCH;
    // 検索モードならスコアを先頭に追加
    if (searchMode) {
      cols = [DB_COLUMNS_SEARCH[0], ...cols];
    }
  }

  const thead = cols.map(c => `<th>${escapeHtml(c.label)}</th>`).join('');
  const tbody = sorted.map(it => {
    const tds = cols.map(c => {
      let v;
      if (c.get) v = c.get(it);
      else if (c.key === 'id') v = it.id;
      else v = String(it[c.key] || '');
      return `<td class="${c.cls || ''}">${escapeHtml(String(v))}</td>`;
    }).join('');
    return `<tr class="${it.id===selectedId?'selected':''}" onclick="selectItem('${escapeJs(it.id)}')">${tds}</tr>`;
  }).join('');

  pane.innerHTML = `<table class="db-table">
    <thead><tr>${thead}</tr></thead>
    <tbody>${tbody}</tbody>
  </table>`;
}

function selectItem(id) {
  selectedId = id;
  document.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));
  const it = currentItems.find(x => x.id === id);
  if (!it) return;
  renderDetail(it);
  renderList();  // selected ハイライト更新
}

function renderDetail(it) {
  const m = it.metadata || {};
  let metaJson = '';
  try { metaJson = m.data_json ? JSON.stringify(JSON.parse(m.data_json), null, 2) : ''; }
  catch(e) { metaJson = m.data_json || ''; }

  const html = `
    <div class="detail-header">
      <span class="detail-id">${escapeHtml(it.id)}</span>
      ${m.category?`<span>cat: <b>${escapeHtml(m.category)}</b></span>`:''}
      ${m.type?`<span>type: ${escapeHtml(m.type)}</span>`:''}
      ${m.always_in_context!==undefined?`<span>aic: <b style="color:${m.always_in_context?'var(--success)':'var(--warn)'}">${m.always_in_context}</b></span>`:''}
    </div>
    <div class="detail-body">
      <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">document（検索対象テキスト）</div>
      <pre>${escapeHtml(it.document || '')}</pre>
    </div>
    ${metaJson ? `<div class="detail-body">
      <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">data_json（構造化原本）</div>
      <pre>${escapeHtml(metaJson)}</pre>
    </div>` : ''}
    <div class="detail-body">
      <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">metadata 全フィールド</div>
      <table class="meta-table">
        ${Object.entries(m).filter(([k])=>k!=='data_json').map(([k,v]) =>
          `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v))}</td></tr>`
        ).join('')}
      </table>
    </div>
    <div class="action-row">
      <button class="btn danger" onclick="deleteItem('${escapeJs(it.id)}')">この項目を削除</button>
    </div>
  `;
  document.getElementById('detail-pane').innerHTML = html;
}

async function deleteItem(id) {
  if (!confirm(`本当に削除しますか?\n  id: ${id}`)) return;
  try {
    await api(`/vdb/long_memory/${encodeURIComponent(id)}`, {method:'DELETE'});
    toast('削除しました');
    selectedId = null;
    document.getElementById('detail-pane').innerHTML = '<div class="empty">左で項目を選択</div>';
    if (searchMode) {
      await doSearch();
    } else if (currentCategory) {
      await selectCategory(currentCategory);
    }
    await loadCategories();
  } catch (e) {
    toast('削除失敗: ' + e.message, true);
  }
}

async function doSearch() {
  const query = document.getElementById('q').value.trim();
  if (!query) { toast('クエリを入力してください', true); return; }
  const top_k = parseInt(document.getElementById('top-k').value || '10', 10);

  searchMode = true;
  document.getElementById('mode-badge').className = 'mode-badge mode-search';
  document.getElementById('mode-badge').textContent = 'SEARCH';
  document.querySelectorAll('.cat-item').forEach(e => e.classList.remove('active'));
  document.getElementById('toolbar-label').textContent = `検索: "${query}"${currentCategory?` / ${currentCategory}`:''}`;
  document.getElementById('list-pane').innerHTML = '<div class="loading">検索中...</div>';

  try {
    const body = {query, top_k};
    if (currentCategory) body.category = currentCategory;
    const data = await api('/vdb/long_memory/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    currentItems = data.hits || [];
    renderList();
  } catch (e) {
    document.getElementById('list-pane').innerHTML = '<div class="empty">検索失敗</div>';
    toast('検索失敗: ' + e.message, true);
  }
}

function clearSearch() {
  document.getElementById('q').value = '';
  searchMode = false;
  if (currentCategory) {
    selectCategory(currentCategory);
  } else {
    document.getElementById('list-pane').innerHTML = '<div class="empty">左でカテゴリを選ぶか、検索してください</div>';
    document.getElementById('detail-pane').innerHTML = '<div class="empty">左で項目を選択</div>';
    document.getElementById('mode-badge').className = 'mode-badge mode-browse';
    document.getElementById('mode-badge').textContent = 'BROWSE';
    document.getElementById('toolbar-label').textContent = 'カテゴリを選択してください';
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function escapeJs(s) { return String(s).replace(/['"\\]/g, c => '\\'+c); }

// 初期化
loadCategories();
</script>
</body>
</html>"""


# 旧 long_memory_db_service.py のメンテ画面と同じレイアウト
# （DB ビュー: events / string_items / records の 3 タブ + JSON ビュー）
_HTML_LONG_MEMORY = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long Memory Viewer</title>
<style>
:root{--bg:#0f1117;--panel:#1a1d27;--border:#2a2d3e;--accent:#a78bfa;--accent2:#56c8f5;--text:#e2e8f0;--muted:#6b7280;--success:#34d399;--danger:#f87171;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;height:100vh;display:flex;flex-direction:column;}
header{background:var(--panel);border-bottom:1px solid var(--border);padding:8px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
header h1{font-size:14px;font-weight:600;color:var(--accent);}
header a{color:var(--muted);font-size:11px;text-decoration:none;margin-left:auto;}
header a:hover{color:var(--accent2);}
.badge{background:var(--border);border-radius:20px;padding:2px 10px;font-size:11px;color:var(--muted);}
.badge span{color:var(--accent2);font-weight:600;}
.tabs{display:flex;background:var(--panel);border-bottom:1px solid var(--border);}
.tab{padding:8px 18px;font-size:12px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.tab:hover:not(.active){color:var(--text);}
.view{flex:1;display:none;overflow:hidden;}
.view.active{display:flex;}

/* DB ビュー */
#db-view{flex-direction:row;}
.sidebar{width:190px;border-right:1px solid var(--border);overflow-y:auto;flex-shrink:0;}
.sidebar-item{padding:8px 12px;cursor:pointer;font-size:12px;color:var(--muted);display:flex;justify-content:space-between;border-bottom:1px solid var(--border);}
.sidebar-item:hover,.sidebar-item.active{background:var(--border);color:var(--text);}
.sidebar-item .cnt{font-size:11px;color:var(--accent2);}
.main-panel{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.toolbar{padding:6px 12px;background:var(--panel);border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
.filter-sel{background:var(--border);border:1px solid #3a3d4e;border-radius:5px;padding:4px 7px;color:var(--text);font-size:11px;outline:none;}
.content-split{flex:1;display:grid;grid-template-columns:1fr 360px;overflow:hidden;}
.list-pane{overflow:auto;padding:0;}
.detail-pane{border-left:1px solid var(--border);overflow-y:auto;padding:12px;background:var(--panel);}

/* Excel テーブル */
.xtable-wrap{min-width:100%;overflow:auto;height:100%;}
table.xt{border-collapse:collapse;width:100%;font-size:12px;}
table.xt thead tr{position:sticky;top:0;z-index:5;}
table.xt th{background:#1e2130;border:1px solid var(--border);padding:0;white-space:nowrap;user-select:none;}
.th-inner{display:flex;align-items:center;gap:4px;padding:5px 8px;cursor:default;position:relative;}
.th-inner:hover .tooltip{display:block;}
.th-label{font-size:11px;color:var(--accent2);font-weight:600;letter-spacing:.03em;}
.th-type{font-size:10px;color:var(--muted);font-family:monospace;}
.tooltip{display:none;position:absolute;top:100%;left:0;min-width:200px;max-width:280px;background:#0a0c14;border:1px solid var(--accent);border-radius:6px;padding:7px 10px;font-size:11px;color:var(--text);white-space:normal;z-index:20;line-height:1.5;font-weight:normal;pointer-events:none;}
table.xt td{border:1px solid var(--border);padding:4px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:240px;cursor:pointer;color:var(--text);}
table.xt tbody tr:hover td{background:#1e2035;}
table.xt tbody tr.selected td{background:#252840;outline:1px solid var(--accent);}
.imp-大{color:#f472b6;font-weight:600;}
.imp-中{color:var(--accent2);}
.imp-小{color:var(--muted);}
.type-badge{font-size:10px;background:var(--border);border-radius:3px;padding:1px 5px;color:var(--muted);}

/* 詳細ペイン */
.detail-h{font-size:11px;color:var(--muted);margin-bottom:10px;border-bottom:1px solid var(--border);padding-bottom:6px;}
.df{margin-bottom:8px;}
.df label{display:block;font-size:10px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.05em;}
.df input,.df select,.df textarea{width:100%;background:#12151f;border:1px solid var(--border);border-radius:4px;padding:5px 7px;color:var(--text);font-size:11px;font-family:inherit;outline:none;resize:vertical;}
.df input:focus,.df textarea:focus{border-color:var(--accent);}
.df input[readonly]{color:var(--muted);background:#0c0e15;}
.ar{display:flex;gap:6px;margin-top:10px;}
.btn{border:none;border-radius:5px;padding:5px 12px;cursor:pointer;font-size:11px;font-family:inherit;}
.btn-p{background:var(--accent);color:#fff;}.btn-p:hover{opacity:.85;}
.btn-d{background:#7f1d1d;color:#fca5a5;}.btn-d:hover{background:#991b1b;}
.btn-n{background:var(--border);color:var(--text);}

/* JSON ビュー */
#json-view{flex-direction:column;}
.json-toolbar{padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--border);display:flex;gap:8px;align-items:center;}
#json-editor{flex:1;padding:12px;font-family:monospace;font-size:12px;background:var(--bg);color:var(--text);border:none;outline:none;resize:none;line-height:1.6;}

/* 共通 */
.empty{color:var(--muted);font-size:12px;padding:20px;text-align:center;}
.toast{position:fixed;bottom:18px;right:18px;padding:8px 14px;border-radius:7px;font-size:12px;z-index:999;transition:opacity .3s;}
.toast-ok{background:#1a2a1a;border:1px solid var(--success);color:var(--success);}
.toast-err{background:#2a1515;border:1px solid var(--danger);color:var(--danger);}
.notice{font-size:10px;color:var(--muted);font-style:italic;padding:6px 0;}
</style>
</head>
<body>
<header>
  <h1>🧠 Long Memory Viewer</h1>
  <div class="badge">events: <span id="s-ev">-</span></div>
  <div class="badge">string_items: <span id="s-si">-</span></div>
  <div class="badge">records: <span id="s-rc">-</span></div>
  <a href="/">← 会話 VDB ビューアへ</a>
</header>
<div class="tabs">
  <div class="tab active" onclick="switchTab('db')">DB ビュー</div>
  <div class="tab" onclick="switchTab('json')">JSON ビュー</div>
</div>

<div id="db-view" class="view active">
  <div class="sidebar" id="sidebar"></div>
  <div class="main-panel">
    <div class="toolbar" id="toolbar"></div>
    <div class="content-split">
      <div class="list-pane" id="list-pane"><div class="empty">← テーブルを選択</div></div>
      <div class="detail-pane" id="detail-pane"><div class="empty">行をクリックして詳細表示</div></div>
    </div>
  </div>
</div>

<div id="json-view" class="view" style="flex-direction:column;">
  <div class="json-toolbar">
    <button class="btn btn-n" onclick="loadJson()">⟳ VDB から読み込み</button>
    <span id="json-status" style="font-size:11px;color:var(--muted);margin-left:8px;"></span>
    <span style="font-size:10px;color:var(--muted);margin-left:auto;font-style:italic">読み取り専用（VDB が source of truth）</span>
  </div>
  <textarea id="json-editor" spellcheck="false" placeholder="VDB から読み込んでください" readonly></textarea>
</div>

<script>
const IMP = ['大','中','小'];
const RTYPES = ['visited','promise','habit','rule','todo','goal','context','person'];
const CAT_LABELS = {fact:'事実',like:'好み',strength:'強み',value:'価値観',
  personality:'性格',interest:'興味',skill:'スキル',
  can_do:'できること',cant_do:'できないこと',state:'現在の状態',want_to:'やりたいこと'};

// ── カラム定義（ラベル・型・ツールチップ）─────────────────────────
const COLS = {
  events: [
    {k:'id',        l:'id',        t:'TEXT PK',   d:'イベントID。日付+連番 (E20250209-01)'},
    {k:'start_dt',  l:'start_dt',  t:'DATETIME',  d:'開始日時。YYYY-MM-DDThh:mm 形式'},
    {k:'end_dt',    l:'end_dt',    t:'DATETIME',  d:'終了日時'},
    {k:'place',     l:'place',     t:'TEXT ≤60',  d:'場所。無言及時は「家」'},
    {k:'importance',l:'importance',t:'大/中/小',  d:'保存価値の大きさ'},
    {k:'what',      l:'what',      t:'TEXT ≤700', d:'出来事の要約'},
    {k:'data',      l:'data',      t:'JSON',      d:'{points:[], feelings:[], who:[]}'},
  ],
  string_items: [
    {k:'id',        l:'id',        t:'TEXT PK',   d:'string_item ID (FC-001 等)'},
    {k:'category',  l:'category',  t:'TEXT',      d:'fact|like|strength|value|personality|interest|skill|can_do|cant_do|state|want_to'},
    {k:'value',     l:'value',     t:'TEXT',      d:'値のテキスト'},
  ],
  records: [
    {k:'id',   l:'id',   t:'TEXT PK', d:'レコードID'},
    {k:'type', l:'type', t:'TEXT',    d:'visited|promise|habit|rule|todo|goal|context|person'},
    {k:'data', l:'data', t:'JSON',    d:'種別ごとの構造'},
  ],
};

let state = {table:'events', filter:'', selectedId:null, rows:[]};

function switchTab(t) {
  document.querySelectorAll('.tab').forEach((el,i)=>el.classList.toggle('active',['db','json'][i]===t));
  document.getElementById('db-view').classList.toggle('active',t==='db');
  document.getElementById('json-view').classList.toggle('active',t==='json');
  if(t==='json') loadJson();
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  if (r.status === 204) return null;
  return r.json();
}

// ── VDB → 旧 LongMemory 構造への変換 ───────────────────────
// 旧 DB の events / string_items / records 3 テーブルに対応するアダプタ。
function extractInner(item) {
  const m = item.metadata || {};
  let parsed = {}, inner = {};
  try { parsed = JSON.parse(m.data_json || '{}'); } catch(e) {}
  try {
    if (typeof parsed.data === 'string') inner = JSON.parse(parsed.data || '{}');
    else if (typeof parsed.data === 'object' && parsed.data) inner = parsed.data;
  } catch(e) {}
  return {parsed, inner, m};
}

async function fetchEvents() {
  const data = await api('/vdb/long_memory/by_category?category=events');
  return (data.items || []).map(it => {
    const {parsed, inner} = extractInner(it);
    return {
      _vdb_id: it.id,
      id: parsed.id || it.id,
      start_dt: parsed.start_dt || '',
      end_dt: parsed.end_dt || '',
      place: parsed.place || '',
      what: parsed.what || '',
      importance: parsed.importance || '中',
      data: JSON.stringify({points:inner.points||[], feelings:inner.feelings||[], who:inner.who||[]}),
    };
  });
}

async function fetchStringItems() {
  // user / agent の両方を取得して string_item のみ拾う
  const userItems = (await api('/vdb/long_memory/by_category?category=user')).items || [];
  const agentItems = (await api('/vdb/long_memory/by_category?category=agent')).items || [];
  const all = [...userItems, ...agentItems];
  return all.filter(it => (it.metadata||{}).type === 'string_item').map(it => {
    const {parsed, m} = extractInner(it);
    return {
      _vdb_id: it.id,
      id: parsed.id || it.id,
      category: m.original_category || parsed.category || '',
      value: parsed.value || it.document || '',
    };
  });
}

async function fetchRecords() {
  // type=record のものを全カテゴリから集める
  const cats = ['visited','procedural','working','user','promises'];
  const items = [];
  for (const c of cats) {
    const data = await api(`/vdb/long_memory/by_category?category=${c}`);
    for (const it of (data.items || [])) {
      if ((it.metadata||{}).type !== 'record') continue;
      const {parsed, inner, m} = extractInner(it);
      items.push({
        _vdb_id: it.id,
        id: parsed.id || it.id,
        type: m.record_type || parsed.type || '',
        data: JSON.stringify(inner, null, 0),
      });
    }
  }
  return items;
}

async function loadStats() {
  // 件数取得
  const [ev, si, rc] = await Promise.all([
    fetchEvents().catch(()=>[]),
    fetchStringItems().catch(()=>[]),
    fetchRecords().catch(()=>[]),
  ]);
  document.getElementById('s-ev').textContent = ev.length;
  document.getElementById('s-si').textContent = si.length;
  document.getElementById('s-rc').textContent = rc.length;
  const items = [
    {key:'events',       cnt:ev.length},
    {key:'string_items', cnt:si.length},
    {key:'records',      cnt:rc.length},
  ];
  document.getElementById('sidebar').innerHTML = items.map(it=>`
    <div class="sidebar-item ${state.table===it.key?'active':''}" onclick="selectTable('${it.key}')">
      <span>${it.key}</span><span class="cnt">${it.cnt}</span>
    </div>`).join('');
}

function selectTable(t) {
  state.table=t; state.filter=''; state.selectedId=null;
  document.querySelectorAll('.sidebar-item').forEach(el=>{
    el.classList.toggle('active', el.querySelector('span')?.textContent===t);
  });
  renderToolbar(); loadRows();
}

function renderToolbar() {
  const tb = document.getElementById('toolbar');
  let filterHtml = '';
  if(state.table==='events') {
    filterHtml = `<label style="font-size:11px;color:var(--muted)">importance:</label>
      <select class="filter-sel" onchange="state.filter=this.value;loadRows()">
        <option value="">すべて</option>${IMP.map(i=>`<option ${state.filter===i?'selected':''}>${i}</option>`).join('')}
      </select>`;
  } else if(state.table==='string_items') {
    filterHtml = `<label style="font-size:11px;color:var(--muted)">category:</label>
      <select class="filter-sel" onchange="state.filter=this.value;loadRows()">
        <option value="">すべて</option>${Object.keys(CAT_LABELS).map(c=>`<option value="${c}" ${state.filter===c?'selected':''}>${c}</option>`).join('')}
      </select>`;
  } else {
    filterHtml = `<label style="font-size:11px;color:var(--muted)">type:</label>
      <select class="filter-sel" onchange="state.filter=this.value;loadRows()">
        <option value="">すべて</option>${RTYPES.map(t=>`<option value="${t}" ${state.filter===t?'selected':''}>${t}</option>`).join('')}
      </select>`;
  }
  tb.innerHTML = filterHtml + `<span class="notice">VDB が source of truth。直接編集ではなく、ミクとの会話を通じて更新されます。</span>`;
}

async function loadRows() {
  try {
    let rows;
    if(state.table==='events') rows = await fetchEvents();
    else if(state.table==='string_items') rows = await fetchStringItems();
    else rows = await fetchRecords();
    // フィルタ
    if(state.filter) {
      if(state.table==='events') rows = rows.filter(r => r.importance === state.filter);
      else if(state.table==='string_items') rows = rows.filter(r => r.category === state.filter);
      else rows = rows.filter(r => r.type === state.filter);
    }
    state.rows = rows;
    renderTable(rows);
  } catch(e) {
    document.getElementById('list-pane').innerHTML = '<div class="empty">読み込み失敗: '+esc(e.message)+'</div>';
  }
}

function renderTable(rows) {
  const pane = document.getElementById('list-pane');
  if(!rows.length){pane.innerHTML='<div class="empty">データなし</div>';return;}
  const cols = COLS[state.table] || [];
  // 日付（start_dt）降順、無ければ id 降順
  const sorted = [...rows].sort((a,b) => {
    if(state.table === 'events') {
      const da = a.start_dt || '', db = b.start_dt || '';
      if(da !== db) return db.localeCompare(da);
    }
    return (b.id || '').localeCompare(a.id || '');
  });
  const thead = `<thead><tr>${cols.map(c=>`
    <th><div class="th-inner">
      <span class="th-label">${esc(c.l)}</span>
      <span class="th-type">${esc(c.t)}</span>
      <div class="tooltip">${esc(c.d)}</div>
    </div></th>`).join('')}</tr></thead>`;

  const tbody = sorted.map(r=>{
    const sel = state.selectedId===r._vdb_id ? 'selected' : '';
    const cells = cols.map(c=>{
      let v = r[c.k] ?? '';
      if(c.k==='importance') return `<td class="imp-${v}">${esc(v)}</td>`;
      if(c.k==='type') return `<td><span class="type-badge">${esc(v)}</span></td>`;
      if(c.k==='data') {
        const preview = typeof v==='string' ? v.replace(/\s+/g,' ').slice(0,60) : '';
        return `<td title="${esc(v)}">${esc(preview)}${v.length>60?'…':''}</td>`;
      }
      const s = String(v);
      return `<td title="${esc(s)}">${esc(s.slice(0,80))}${s.length>80?'…':''}</td>`;
    }).join('');
    const rowDataAttr = encodeURIComponent(JSON.stringify(r));
    return `<tr class="${sel}" data-row="${rowDataAttr}" onclick="selectRow(JSON.parse(decodeURIComponent(this.dataset.row)))">${cells}</tr>`;
  }).join('');

  pane.innerHTML = `<div class="xtable-wrap"><table class="xt">${thead}<tbody>${tbody}</tbody></table></div>`;
}

// ── 行選択・詳細ペイン ─────────────────────────────────────────
function selectRow(r) {
  state.selectedId = r._vdb_id;
  renderDetail(r);
  document.querySelectorAll('table.xt tbody tr').forEach(el=>{
    try {
      const rowData = JSON.parse(decodeURIComponent(el.dataset.row));
      el.classList.toggle('selected', rowData._vdb_id === r._vdb_id);
    } catch(e) {}
  });
}

function renderDetail(r) {
  const dp = document.getElementById('detail-pane');
  if(state.table==='events') {
    const d = (()=>{try{return JSON.parse(r.data||'{}');}catch{return {};}})();
    dp.innerHTML = `<div class="detail-h">events / ${esc(r.id)}</div>
      <div class="df"><label>id</label><input value="${esc(r.id)}" readonly></div>
      <div class="df"><label>start_dt</label><input value="${esc(r.start_dt)}" readonly></div>
      <div class="df"><label>end_dt</label><input value="${esc(r.end_dt)}" readonly></div>
      <div class="df"><label>place</label><input value="${esc(r.place)}" readonly></div>
      <div class="df"><label>importance</label><input value="${esc(r.importance)}" readonly></div>
      <div class="df"><label>what</label><textarea rows="5" readonly>${esc(r.what)}</textarea></div>
      <div class="df"><label>points</label><textarea rows="3" readonly>${esc(JSON.stringify(d.points||[],null,2))}</textarea></div>
      <div class="df"><label>feelings</label><textarea rows="2" readonly>${esc(JSON.stringify(d.feelings||[],null,2))}</textarea></div>
      <div class="df"><label>who</label><textarea rows="2" readonly>${esc(JSON.stringify(d.who||[],null,2))}</textarea></div>
      <div class="ar"><button class="btn btn-d" onclick="deleteVdbItem('${esc(r._vdb_id)}')">削除</button></div>`;
  } else if(state.table==='string_items') {
    dp.innerHTML = `<div class="detail-h">string_items / ${esc(r.id)}</div>
      <div class="df"><label>id</label><input value="${esc(r.id)}" readonly></div>
      <div class="df"><label>category</label><input value="${esc(r.category)}" readonly></div>
      <div class="df"><label>value</label><textarea rows="4" readonly>${esc(r.value)}</textarea></div>
      <div class="ar"><button class="btn btn-d" onclick="deleteVdbItem('${esc(r._vdb_id)}')">削除</button></div>`;
  } else {
    dp.innerHTML = `<div class="detail-h">records / ${esc(r.id)} [${esc(r.type)}]</div>
      <div class="df"><label>id</label><input value="${esc(r.id)}" readonly></div>
      <div class="df"><label>type</label><input value="${esc(r.type)}" readonly></div>
      <div class="df"><label>data (JSON)</label><textarea rows="12" readonly>${esc(fmtJson(r.data))}</textarea></div>
      <div class="ar"><button class="btn btn-d" onclick="deleteVdbItem('${esc(r._vdb_id)}')">削除</button></div>`;
  }
}

async function deleteVdbItem(vdbId) {
  if(!confirm(`本当に削除しますか?\n  id: ${vdbId}`))return;
  try {
    await api(`/vdb/long_memory/${encodeURIComponent(vdbId)}`, {method:'DELETE'});
    toast('削除しました');
    state.selectedId = null;
    document.getElementById('detail-pane').innerHTML = '<div class="empty">行をクリックして詳細表示</div>';
    await loadStats();
    await loadRows();
  } catch(e) {
    toast('削除失敗: ' + e.message, true);
  }
}

// ── JSON ビュー ────────────────────────────────────────────────
async function loadJson() {
  document.getElementById('json-status').textContent='読み込み中...';
  try {
    const r = await fetch('/vdb/long_memory/full_json');
    if(!r.ok) throw new Error(await r.text());
    const txt = await r.text();
    document.getElementById('json-editor').value=JSON.stringify(JSON.parse(txt),null,2);
    document.getElementById('json-status').textContent='読み込み完了';
  } catch(e) {
    toast('読み込み失敗: '+e.message, true);
    document.getElementById('json-status').textContent='エラー';
  }
}

// ── ユーティリティ ─────────────────────────────────────────────
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function fmtJson(s){try{return JSON.stringify(JSON.parse(s||'{}'),null,2);}catch{return s||'';}}
function toast(msg,err=false){
  const t=document.createElement('div');
  t.className='toast '+(err?'toast-err':'toast-ok');
  t.textContent=msg; document.body.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';setTimeout(()=>t.remove(),300);},2500);
}

// 初期化
loadStats(); renderToolbar(); loadRows();
</script>
</body>
</html>"""


@app.get("/long_memory")
def long_memory_viewer():
    """長期記憶コレクションのメンテ画面（カテゴリ別閲覧・検索・削除）。"""
    return Response(_HTML_LONG_MEMORY, mimetype="text/html; charset=utf-8")


# ── 起動 ──────────────────────────────────────────────────

def run_vdb_service(settings: dict, root: Path, port: int, host: str = "127.0.0.1") -> None:
    """gateway_service.py から別スレッドで呼ぶ公開関数。

    host は 127.0.0.1 固定で使う（認証が無いので、同じ Wi-Fi の他の端末から
    記憶を読ませない・消させない・2026-10-08）。
    """
    configure_env_from_settings(settings)
    initialize_vdb_service(settings, root)

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    print(f"VDBサービスを起動します: http://{host}:{port}")
    print(f"  会話メンテ画面: http://127.0.0.1:{port}/")
    print(f"  長期記憶メンテ画面: http://127.0.0.1:{port}/long_memory")
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


def main() -> None:
    """単体起動の入口。配布物では `.pyd` 化されるため、起動処理は関数に置く。"""
    import argparse

    p = argparse.ArgumentParser(description="VDB サービス単体起動")
    p.add_argument("--port", type=int, default=8020)
    args = p.parse_args()

    _root = _SRC.parent.parent  # miku_agent/
    _settings = load_settings(_SRC / "settings.json")
    run_vdb_service(_settings, _root, args.port)


if __name__ == "__main__":
    main()
