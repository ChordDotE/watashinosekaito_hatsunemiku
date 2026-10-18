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

"""現在心境（current_mind）— 感情 feelings と興味 interests の保存・減衰・注入文の生成。

KV の 1 キー `current_mind` に載る状態を扱う。長期記憶（積み重なったもの）に対して、
**いま動いていて時間とともに移ろうもの**を持つ層。

- `feelings`  : 出来事に紐づく気持ち。単調減衰（半減期 0.5〜72h）
- `interests` : 関心。`strength`（速い）と `knowledge`（遅い）の2変数を持ち、
                **知識が興味の天井を下げる**ので無限に詳しくならない

**番号（target）は build_view() が確定した並びに対して解決する**。
適用中に順序が変わっても宛先がずれないようにするため。

仕様・パラメータの根拠・実測値は docs/components/current_mind.md が正。
設計の経緯（なぜ選択肢にしたか等）は docs/plan/current_mind_design.md。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

JST = timezone(timedelta(hours=9))
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- clamp（LLM の暴走ガード。プロンプト側の抑制と二段構え） ---
STRENGTH_MIN, STRENGTH_MAX = 0.1, 1.0
HALF_LIFE_MIN, HALF_LIFE_MAX = 0.5, 72.0

# --- 掃除 ---
EFFECTIVE_FLOOR = 0.05
MAX_FEELINGS = 20
MAX_INTERESTS = 20

# --- 保存ガード（感情）。プロンプトでは守られないため必ずコード側で担保する ---
SAVE_FLOOR = 0.3
CAP_WINDOW_HOURS = 2.0
CAP_COUNT = 1
LONG_LANE_THRESHOLD_HOURS = 12.0

# --- 興味の知識モデル ---
# 寿命は「放置5日 / 深掘りしても10日 / 会話で繰り返し触れて24日」に収まるよう調整済み。
# 長期の関心（趣味）は人格側が持つものなので、興味システムからは生み出さない。
KNOWLEDGE_INITIAL = 0.15          # notice した時点で「少し知った」
KNOWLEDGE_INJECT = 0.05           # 注入されただけ（思考に入った）で増える分
KNOWLEDGE_TOUCH = 0.30            # 実際に話した・調べた（shift 報告）で増える分
KNOWLEDGE_HALF_LIFE_HOURS = 24 * 21   # 知識の半減期（3週間）。strength よりずっと長い
_CAP_P = 0.5                      # 単峰の形（0<p<1）
_CAP_STAR = 1.0                   # 天井の裾の広さ。大きいほど飽きるまでが長い
_CAP_PEAK = (_CAP_P * _CAP_STAR) ** _CAP_P * math.exp(-_CAP_P)   # 正規化用のピーク値
DEFAULT_INTEREST_HALF_LIFE = 36.0     # lasts_hours 未指定時の既定（放置で約5日）

# 1ターンに受け付ける新規の上限（乱発防止・コード側の最終防衛線）
MAX_NEW_PER_TURN = 2

# 性格由来の地。**動かさない**（動かすと人格をゆっくり書き換えることになるため）
DEFAULT_BASELINE: Dict[str, float] = {"valence": 0.2, "energy": 0.6}

# --- 感情語 → (valence, 余力を奪う重み) ---
# label は選択肢（schemas.py の enum と1対1）。コードが分岐に使う値なのでぶれさせない。
# 重みは2段階: 重い=解決しておらず先が見えない / 軽い=一時的で自己完結する。
_HEAVY, _LIGHT = 0.90, 0.45

FEELING_SPEC: Dict[str, Tuple[int, float]] = {
    # 正（9）
    "嬉しい": (+1, 0.0), "楽しい": (+1, 0.0), "安心": (+1, 0.0), "楽しみ": (+1, 0.0),
    "感謝": (+1, 0.0), "誇らしい": (+1, 0.0), "穏やか": (+1, 0.0), "わくわく": (+1, 0.0),
    "面白い": (+1, 0.0),
    # 負・重い（6）
    "心配": (-1, _HEAVY), "不安": (-1, _HEAVY), "悲しい": (-1, _HEAVY),
    "寂しい": (-1, _HEAVY), "悔しい": (-1, _HEAVY), "怒り": (-1, _HEAVY),
    # 負・軽い（4）
    "ばつが悪い": (-1, _LIGHT), "もどかしい": (-1, _LIGHT),
    "がっかり": (-1, _LIGHT), "申し訳ない": (-1, _LIGHT),
    # 中立（2）
    "驚き": (0, 0.0), "懐かしい": (0, 0.0),
}
FEELING_LABELS: Tuple[str, ...] = tuple(FEELING_SPEC)

# --- 向き（誰のための気持ち・関心か） ---
# 文字列マッチでは「マスターとは関係ない話」まで master に数えてしまい、
# 実測で両方向に誤ったため、本人に書かせる。TOWARD_LEGACY_WORDS は toward が
# 無い古いデータのためのフォールバックにだけ使う。
TOWARD_VALUES: Tuple[str, ...] = ("master", "self", "world")
TOWARD_LEGACY_WORDS = ("マスター", "master")

# --- 配分の気づき（マスター寄りのときだけ一人称で1行添える） ---
BALANCE_LOW = 0.35          # 自分起点の割合がこれ未満＝マスター寄りすぎ
BALANCE_MIN_ITEMS = 3       # 興味がこの件数以上ないと偏りを語らない（1件で「ばかり」は不自然）
BALANCE_MIN_TOTAL = 0.8     # 実効強度の合計がこれ未満でも語らない
BALANCE_NUDGE = ("最近、マスターのことばかり調べている気がする。"
                 "自分が好きなことにも目を向けてみたいな。")

# 余力がこれ未満なら「# 気になっていること」ブロックごと出さない
CAPACITY_HIDE_INTEREST = 0.4

# 表示は感情・興味で**別枠**にする（共通枠だと感情が増えた瞬間に興味が押し出される）
MAX_FEELING_LINES = 5
MAX_INTEREST_LINES = 5

# 表示から漏れた興味は「思い出す機会が無かった」ぶん忘れが速い
UNSHOWN_INTEREST_DECAY = 0.85

# 種（pinned）。曝露ツールを呼ぶ動機そのもので、状態が空のときに書き込む。
# 「対象」ではなく「何をしたいか」で書く（対象だけだと何をすればいいか伝わらず、
# 実測で呼び出しが 0 回になった）。
SEED_INTERESTS: Tuple[Dict[str, Any], ...] = (
    {"topic": "ときどき、いま世の中でどんな話題があるのかを眺めたい",
     "strength": 0.5, "knowledge": 0.0, "pinned": True},
    {"topic": "マスターと過ごしてきた時間には、いま思い出せていないことがたくさんある。"
              "ときどき引っ張り出してみたい",
     "strength": 0.5, "knowledge": 0.0, "pinned": True},
)


# --- 基本 ---
def _parse_time(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value).strip(), _TIME_FORMAT).replace(tzinfo=JST)
    except (ValueError, AttributeError, TypeError):
        return None


def _clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def load(raw: str) -> Dict[str, Any]:
    """KV の生文字列から復元する。空・壊れた JSON は空の状態として扱う。"""
    state: Dict[str, Any] = {
        "baseline": dict(DEFAULT_BASELINE),
        "feelings": [],
        "interests": [],
    }
    data: Any = None
    if raw and str(raw).strip():
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = None
    if isinstance(data, dict):
        if isinstance(data.get("baseline"), dict):
            state["baseline"].update(
                {k: v for k, v in data["baseline"].items() if isinstance(v, (int, float))}
            )
        if isinstance(data.get("feelings"), list):
            state["feelings"] = [e for e in data["feelings"]
                                 if isinstance(e, dict) and e.get("what")]
        if isinstance(data.get("interests"), list):
            state["interests"] = [e for e in data["interests"]
                                  if isinstance(e, dict) and e.get("topic")]
    # 空・壊れた JSON のときこそ種が要る（早期 return で飛ばさない）
    _ensure_seeds(state)
    return state


def _ensure_seeds(state: Dict[str, Any]) -> None:
    """種が無ければ入れる。種が消えると曝露ツールを呼ぶ動機が失われるため。

    既存の種は触らない（`strength` が動いていても上書きしない）。
    """
    have = {it.get("topic") for it in state["interests"] if is_pinned(it)}
    for seed in SEED_INTERESTS:
        if seed["topic"] not in have:
            state["interests"].append(dict(seed))


def dumps(state: Dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, indent=2)


# --- 減衰 ---
def feeling_effective(ev: Dict[str, Any], now: datetime) -> float:
    """感情の実効強度。単調減衰（at が読めなければ減衰なし＝消さない側に倒す）。"""
    strength = _clamp(ev.get("strength", 0.0), 0.0, STRENGTH_MAX, 0.0)
    at = _parse_time(ev.get("at", ""))
    if at is None:
        return strength
    h = _clamp(ev.get("half_life_hours"), HALF_LIFE_MIN, HALF_LIFE_MAX, HALF_LIFE_MIN)
    elapsed = max(0.0, (now - at).total_seconds() / 3600.0)
    return strength * (0.5 ** (elapsed / h))


def is_pinned(it: Dict[str, Any]) -> bool:
    """「種」かどうか。種は減衰も飽きもしない。

    曝露ツール（check_news / recall_summary_experience）を呼ぶ動機そのもの。
    通常の興味として扱うと減衰して消え、**呼ぶ動機が失われて曝露が止まる**。
    """
    return bool(it.get("pinned"))


def knowledge_now(it: Dict[str, Any], now: datetime) -> float:
    """現在の知識量。触れなければゆっくり薄れる（strength よりずっと遅い）。"""
    if is_pinned(it):
        return 0.0        # 種は飽きない（知識が増えないので天井も下がらない）
    k = _clamp(it.get("knowledge", 0.0), 0.0, 10.0, 0.0)
    at = _parse_time(it.get("last_touch", "")) or _parse_time(it.get("at", ""))
    if at is None:
        return k
    elapsed = max(0.0, (now - at).total_seconds() / 3600.0)
    return k * (0.5 ** (elapsed / KNOWLEDGE_HALF_LIFE_HOURS))


def strength_cap(knowledge: float) -> float:
    """その知識量における興味の天井（単峰）。

    知らなすぎると取っ付けない → 少し知ると最大 → 知り尽くすと冷める。
    ピークで正規化してあるので最大 1.0。
    """
    if knowledge <= 0:
        return 0.0
    return ((knowledge ** _CAP_P) * math.exp(-knowledge / _CAP_STAR)) / _CAP_PEAK


def interest_effective(it: Dict[str, Any], now: datetime) -> float:
    """興味の実効強度。時間減衰したうえで、知識による天井で頭打ちにする。"""
    w = _clamp(it.get("strength", 0.0), 0.0, STRENGTH_MAX, 0.0)
    if is_pinned(it):
        return w          # 種は減衰しない
    at = _parse_time(it.get("at", ""))
    if at is not None:
        h = _clamp(it.get("half_life_hours"), HALF_LIFE_MIN, HALF_LIFE_MAX, HALF_LIFE_MIN)
        elapsed = max(0.0, (now - at).total_seconds() / 3600.0)
        w = w * (0.5 ** (elapsed / h))
    return min(w, strength_cap(knowledge_now(it, now)))


# --- 掃除・余力 ---
def prune(state: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    """消えたものを落とす。**種は掃除せず、件数上限にも算入しない**。"""
    feelings = [e for e in state.get("feelings", [])
                if feeling_effective(e, now) >= EFFECTIVE_FLOOR]
    pinned = [e for e in state.get("interests", []) if is_pinned(e)]
    normal = [e for e in state.get("interests", [])
              if not is_pinned(e) and interest_effective(e, now) >= EFFECTIVE_FLOOR]
    return {
        "baseline": state.get("baseline", dict(DEFAULT_BASELINE)),
        "feelings": feelings[-MAX_FEELINGS:],
        "interests": pinned + normal[-MAX_INTERESTS:],
    }


def valence_of(label: str) -> int:
    """感情語 → valence（+1 / 0 / -1）。選択肢外は 0（＝現行挙動と一致）。"""
    return FEELING_SPEC.get(str(label), (0, 0.0))[0]


def weight_of(label: str) -> float:
    """その感情が余力をどれだけ削るか（負の感情だけ非ゼロ）。"""
    return FEELING_SPEC.get(str(label), (0, 0.0))[1]


def capacity(state: Dict[str, Any], now: datetime) -> float:
    """いまの余力（0.2〜1.0）。負の感情が強いほど下がる。

    低いと注入時に「# 気になっていること」ブロックを出さない（深層 → 表層の抑制）。
    重みが 0 の感情（正・中立）は効かないので、valence の判定は要らない。
    """
    burden = sum(
        feeling_effective(e, now) * weight_of(e.get("label", ""))
        for e in state.get("feelings", [])
    )
    return max(0.2, min(1.0, 1.0 - burden))


# --- 向き・配分 ---
def toward_of(it: Dict[str, Any]) -> str:
    """向きの生の値（master / self / world）。**LLM が返す toward を正とする**。

    toward を持たない古いデータのために、旧来の文字列マッチをフォールバックに残す。
    """
    t = it.get("toward")
    if t in TOWARD_VALUES:
        return str(t)
    text = f"{it.get('why', '')}{it.get('topic', '')}"
    return "master" if any(w in text for w in TOWARD_LEGACY_WORDS) else "self"


def orientation_of(it: Dict[str, Any]) -> str:
    """配分の計算に使う2値。**world（世の中への関心）は自分側**に数える。"""
    return "master" if toward_of(it) == "master" else "self"


def orientation_balance(interests: List[Dict[str, Any]], now: datetime) -> Optional[float]:
    """自分起点の割合（0=全部マスター向け / 1=全部自分）。材料が足りなければ None。

    **件数と実効強度の両方**が足りているときだけ値を返す。
    1〜2件で「ばかり」と言うと不自然になるため。
    """
    if len([it for it in interests if not is_pinned(it)]) < BALANCE_MIN_ITEMS:
        return None
    s = m = 0.0
    for it in interests:
        if is_pinned(it):
            continue      # 種は向きを持たない
        eff = interest_effective(it, now)
        if orientation_of(it) == "master":
            m += eff
        else:
            s += eff
    total = s + m
    return None if total < BALANCE_MIN_TOTAL else s / total


# --- view（番号の確定） ---
def build_view(state: Dict[str, Any], now_text: str) -> Dict[str, Any]:
    """減衰・掃除・余力を計算し、**注入と適用の両方が参照する並び**を確定する。"""
    now = _parse_time(now_text) or datetime.now(JST)
    pruned = prune(state, now)
    feelings = sorted(pruned["feelings"],
                      key=lambda e: feeling_effective(e, now), reverse=True)
    # 種は番号を振らない（shift の対象にしない）ので、通常の興味と分ける
    pinned = [it for it in pruned["interests"] if is_pinned(it)]
    interests = sorted([it for it in pruned["interests"] if not is_pinned(it)],
                       key=lambda e: interest_effective(e, now), reverse=True)
    cap = capacity(pruned, now)
    show_interest = cap >= CAPACITY_HIDE_INTEREST
    return {
        "now": now, "now_text": now_text, "capacity": cap,
        "feelings": feelings, "interests": interests, "pinned": pinned,
        "feelings_shown": feelings[:MAX_FEELING_LINES],
        "interests_shown": interests[:MAX_INTEREST_LINES] if show_interest else [],
        "balance": orientation_balance(interests, now),
    }


# --- 注入文 ---
def _adverb(eff: float) -> str:
    if eff >= 0.55:
        return "すごく"
    if eff >= 0.25:
        return ""
    return "まだ少し"


def _age_phrase(ev: Dict[str, Any], now: datetime) -> str:
    at = _parse_time(ev.get("at", ""))
    if at is None:
        return ""
    hours = (now - at).total_seconds() / 3600.0
    if hours < 1.0:
        return "さっき"
    if hours < 24.0:
        return f"{int(hours)}時間前"
    return f"{int(hours // 24)}日前"


def render(view: Dict[str, Any]) -> str:
    """user_prompt の動的ブロックへ入れる自然文。**ラベル・数値・次元は書かない**。

    設定シート風にすると inner_thoughts が「AI の口調」になることが実測で分かっているため、
    一人称の自然文にする。強さは副詞、経過は自然文に落とす。
    """
    now = view["now"]
    lines: List[str] = []
    pinned = view.get("pinned") or []

    if view["feelings_shown"]:
        lines.append("# いまの気分")
        for i, e in enumerate(view["feelings_shown"], 1):
            eff = feeling_effective(e, now)
            age = _age_phrase(e, now)
            prefix = f"{age}、" if age else ""
            lines.append(f"{i}. {prefix}{e['what']}。いまも{_adverb(eff)}{e.get('label', '')}。")

    if view["interests_shown"] or pinned:
        if lines:
            lines.append("")
        lines.append("# 気になっていること")
        for i, it in enumerate(view["interests_shown"], 1):
            eff = interest_effective(it, now)
            head = ("すごく気になっている" if eff >= 0.55
                    else "気になっている" if eff >= 0.25 else "まだ少し気になっている")
            why = f"（{it['why']}）" if it.get("why") else ""
            lines.append(f"{i}. {it['topic']}{why}。{head}。")

        # 向きが偏っていたら一人称の気づきを1行だけ添える。**指示文にしない**
        b = view.get("balance")
        if b is not None and b < BALANCE_LOW:
            lines.append(BALANCE_NUDGE)

        # 種は番号を振らない。topic に「何をしたいか」まで書いてある前提でそのまま括弧に入れる
        for p in pinned:
            lines.append(f"（{p.get('topic', '')}）")

    return "\n".join(lines)


# --- 適用（受け取り） ---
def _lane(ev: Dict[str, Any]) -> str:
    h = _clamp(ev.get("half_life_hours"), HALF_LIFE_MIN, HALF_LIFE_MAX, HALF_LIFE_MIN)
    return "長期" if h >= LONG_LANE_THRESHOLD_HOURS else "短期"


def _norm_toward(u: Dict[str, Any]) -> str:
    """LLM が返した toward を正規化する。選択肢外・未指定なら空（＝旧判定に委ねる）。"""
    t = str(u.get("toward") or "").strip()
    return t if t in TOWARD_VALUES else ""


def _apply_feel(state: Dict[str, Any], u: Dict[str, Any], now_text: str) -> str:
    what = str(u.get("what") or "").strip()
    label = str(u.get("label") or "").strip()
    if not what or not label:
        return "無効"
    strength = _clamp(u.get("strength"), STRENGTH_MIN, STRENGTH_MAX, STRENGTH_MIN)
    if strength < SAVE_FLOOR:
        return "床で破棄"
    new = {
        "what": what, "label": label, "strength": strength,
        "at": now_text,
        "half_life_hours": _clamp(u.get("lasts_hours"), HALF_LIFE_MIN, HALF_LIFE_MAX, 6.0),
        "from": str(u.get("from") or "").strip(),
    }
    now = _parse_time(now_text) or datetime.now(JST)
    lane = _lane(new)
    recent = [
        e for e in state["feelings"]
        if (at := _parse_time(e.get("at", ""))) is not None
        and (now - at).total_seconds() / 3600.0 <= CAP_WINDOW_HOURS
        and _lane(e) == lane
    ]
    if len(recent) >= CAP_COUNT:
        weakest = min(recent, key=lambda e: float(e.get("strength", 0)))
        if new["strength"] <= float(weakest.get("strength", 0)):
            return "キャップで破棄"
        state["feelings"] = [e for e in state["feelings"] if e is not weakest]
        state["feelings"].append(new)
        return "入れ替えで採用"
    state["feelings"].append(new)
    return "採用"


def _apply_notice(state: Dict[str, Any], u: Dict[str, Any], now_text: str) -> str:
    topic = str(u.get("what") or "").strip()
    if not topic:
        return "無効"
    w = _clamp(u.get("strength"), STRENGTH_MIN, STRENGTH_MAX, STRENGTH_MIN)
    if w < SAVE_FLOOR:
        return "床で破棄"
    # 緩い重複検出（LLM が shift と notice を取り違えたときの保険）
    norm = topic.replace(" ", "").replace("　", "")
    for it in state["interests"]:
        if norm and norm in str(it.get("topic", "")).replace(" ", "").replace("　", ""):
            it["strength"] = max(float(it.get("strength", 0)), w)
            it["knowledge"] = float(it.get("knowledge", 0)) + KNOWLEDGE_TOUCH
            it["last_touch"] = now_text
            return "既存に統合"
    state["interests"].append({
        "topic": topic,
        "why": str(u.get("from") or u.get("label") or "").strip(),
        "toward": _norm_toward(u),
        "strength": w,
        "knowledge": KNOWLEDGE_INITIAL,
        "at": now_text,
        "last_touch": now_text,
        "half_life_hours": _clamp(
            u.get("lasts_hours"), HALF_LIFE_MIN, HALF_LIFE_MAX, DEFAULT_INTEREST_HALF_LIFE
        ),
    })
    return "採用"


def apply(
    state: Dict[str, Any],
    view: Dict[str, Any],
    mood_updates: Optional[List[Dict[str, Any]]],
    interest_updates: Optional[List[Dict[str, Any]]],
    now_text: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """LLM 出力を状態へ反映する。**番号は view の並びに対して解決する**。

    適用順は 解消 → 更新 → 新規（消えた枠に新しいものが入れるように）。
    戻り値: (新しい状態, ログ行)
    """
    logs: List[str] = []
    new_state = {
        "baseline": dict(state.get("baseline", DEFAULT_BASELINE)),
        "feelings": list(state.get("feelings", [])),
        "interests": list(state.get("interests", [])),
    }

    def _target(items: List[Dict], idx: Any) -> Optional[Dict]:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return None
        return items[i - 1] if 1 <= i <= len(items) else None

    for kind_key, updates, shown, bucket in (
        ("mood", mood_updates or [], view["feelings_shown"], "feelings"),
        ("interest", interest_updates or [], view["interests_shown"], "interests"),
    ):
        ups = [u for u in updates if isinstance(u, dict)]

        # ① 解消（感情のみ。興味に「消す操作」は用意していない）
        for u in [x for x in ups if x.get("kind") == "resolve"]:
            tgt = _target(shown, u.get("target"))
            if tgt is None:
                logs.append(f"{kind_key}:解消 範囲外")
                continue
            new_state[bucket] = [e for e in new_state[bucket] if e is not tgt]
            logs.append(f"{kind_key}:解消 {str(tgt.get('what') or tgt.get('topic'))[:24]}")

        # ② 更新
        for u in [x for x in ups if x.get("kind") == "shift"]:
            tgt = _target(shown, u.get("target"))
            if tgt is None or tgt not in new_state[bucket]:
                logs.append(f"{kind_key}:更新 範囲外")
                continue
            if u.get("strength") is not None:
                key = "strength"
                tgt[key] = _clamp(u.get("strength"), STRENGTH_MIN, STRENGTH_MAX,
                                  tgt.get(key, 0.3))
            if u.get("label"):
                tgt["label" if bucket == "feelings" else "why"] = str(u["label"]).strip()
            if bucket == "interests":
                # 話題になった＝知識が増える
                tgt["knowledge"] = float(tgt.get("knowledge", 0)) + KNOWLEDGE_TOUCH
                tgt["last_touch"] = now_text
            logs.append(f"{kind_key}:更新 {str(tgt.get('what') or tgt.get('topic'))[:24]}")

        # ③ 新規（1ターンの上限つき）
        news = [x for x in ups if x.get("kind") in ("feel", "notice")][:MAX_NEW_PER_TURN]
        for u in news:
            if u.get("kind") == "feel":
                r = _apply_feel(new_state, u, now_text)
                logs.append(f"mood:新規 {r} / {str(u.get('what'))[:24]}")
            else:
                r = _apply_notice(new_state, u, now_text)
                logs.append(f"interest:新規 {r} / {str(u.get('what'))[:24]}")

    # ④ 注入されたトピックは、思考に入ったぶんだけ知識が増える
    shown_topics = {it.get("topic") for it in view["interests_shown"]}
    for cur in new_state["interests"]:
        if is_pinned(cur):
            continue
        if cur.get("topic") in shown_topics:
            cur["knowledge"] = float(cur.get("knowledge", 0)) + KNOWLEDGE_INJECT
        elif cur.get("at") != now_text:
            # 表示から漏れた＝このターン思い出さなかったものは忘れが速い。
            # そのままだと「見えないのに strength だけゆっくり減る」状態で居座り続ける
            cur["strength"] = float(cur.get("strength", 0)) * UNSHOWN_INTEREST_DECAY

    now = _parse_time(now_text) or datetime.now(JST)
    return prune(new_state, now), logs
