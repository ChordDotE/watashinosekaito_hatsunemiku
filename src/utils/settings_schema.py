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

"""settings.json の設定キー定義（スタート画面「設定」タブのマスター）。

設定タブはこの定義からフォームを自動生成する。
settings_sample.json にキーを追加したら、このファイルにもフィールドを1つ追加すること
（追加しないと画面に表示されず、手動編集でしか変更できない。
 漏れは tests/utils/test_settings_store.py の網羅テストが検出する）。

**文言はキー名から連想して書かない**。`docs/configuration.md` にそのキーが実際に何をするかが
書いてあるので、必ず読んでから起こす（読まずに書くと、もっともらしい嘘の説明になる。
規約: .claude/skills/miku-dev/SKILL.md「作る前・答える前に、対応するドキュメントを読む」）。
利用者は `top_k` や `compress_threshold` を知らないので、**何が起きるか**で書く。

フィールドの型（type）:
    string   テキスト
    secret   APIキー等の秘匿値。GET では末尾3文字を残してマスクされ、
             マスクのまま PUT された場合は「変更なし」として無視される
    password ログインパスワード。**保存は scrypt ハッシュ**（復元不可）で、GET では
             設定の有無だけが分かる固定マスク（末尾も文字数も出さない）。
             マスク・空のまま PUT された場合は「変更なし」として無視される
    bool     チェックボックス
    int      整数
    float    小数
    select   choices からの選択。recommended に現行標準構成の値を指定すると
             画面がその選択肢に「（推奨）」を付けて表示する
    list_str 文字列リスト（画面ではカンマ区切りで編集）

画面での扱い（ui）:
    form     通常のフォーム項目として、そのセクションに出す（既定）
    feature  「ミクにできること」で、機能の行の一部として出す
             （導入操作や状態表示と一体で描くため、汎用フォームには出さない）
    none     **画面に出さない**。settings.json を直接編集する人向けに残す。
             理由を desc に書くこと（次に触る人が「記載漏れ」と誤解して足し戻さないように）
"""
from __future__ import annotations

# GET /api/settings で秘匿値の代わりに返すマスク文字列
MASKED_VALUE = "********"

# settings_sample.json の未設定プレースホルダ（この接頭辞なら「未設定」扱い）
SECRET_PLACEHOLDER_PREFIX = "put your"


def _f(path: str, label: str, ftype: str = "string", desc: str = "",
       choices: list | None = None, recommended: str | None = None,
       maint_links: list | None = None, ui: str = "form",
       feature: str | None = None, unit: str = "") -> dict:
    """maint_links: フィールドに添えるボタン定義。2形式あり、どちらも**稼働中のみ有効**:
      - {"label", "path"}   … メンテページを開く。http://<ホスト>:<ポート値><path>（port フィールド用）
      - {"label", "action"} … コントロールパネルの API を呼ぶ（例: "unlock_auth"）

    feature: ui="feature" のとき、どの機能の行に属するかの識別子。
    unit: 数値の単位（「秒」「文字」など）。入力欄の右に添える。
    """
    field = {"path": path, "label": label, "type": ftype, "desc": desc, "ui": ui}
    if choices:
        field["choices"] = choices
    if recommended is not None:
        field["recommended"] = recommended
    if maint_links:
        field["maint_links"] = maint_links
    if feature:
        field["feature"] = feature
    if unit:
        field["unit"] = unit
    return field


# 「ミクにできること」に並べる機能。値を持たない操作（導入・生成）もここに含む。
# 実体の有無は supervisor.feature_status() が見る（設定値では判定しない）。
FEATURES: list[dict] = [
    {
        "id": "live2d",
        "name": "ミクの姿（Live2D）",
        "desc": "画面にミクが表示され、表情や口の動きがつきます。"
                "セットアップ時に導入済みです（表示がおかしいときに入れ直せます）",
        "action": "setup_live2d",
        # Live2D はセットアップ時に必須なので、ここは「追加する」場所ではなく
        # **壊れたときに入れ直す**場所。文言もそれに合わせる
        "action_label": "入れる",
        "action_label_done": "入れ直す",
    },
    {
        "id": "web_search",
        "name": "Web検索",
        "desc": "ミクが調べものをして答えられるようになります",
        "note": "無料枠があります",
        "link": "https://tavily.com/",
        "link_label": "キーの取得",
    },
    {
        "id": "email",
        "name": "メール送信",
        "desc": "ミクがあなた宛にメールを送れるようになります",
        "note": "送信元は Gmail のみに対応しています。"
                "ふだんのパスワードではなく「アプリパスワード」が必要です"
                "（Google アカウント → セキュリティ → 2段階認証を有効化 → アプリパスワード）",
        "link": "https://myaccount.google.com/apppasswords",
        "link_label": "アプリパスワードの発行",
    },
    {
        "id": "song",
        "name": "作曲機能",
        "desc": "ミクに曲を作ってもらいます（オリジナル曲と、楽譜〔MIDI〕が見つかる曲のカバーは GPU 不要。"
                "音源しか無い曲のカバーだけ NVIDIA GPU が必要）",
        "action": "install_song_service",
        "action_label": "導入する",
        "action_label_done": "入れ直す",
        # ★**GPU 必須をやめた**（2026-09-01）。オリジナル曲の経路は GPU を使っておらず、
        #   入口で一律に弾いていたせいで**できるはずの人を締め出していた**。
        #   GPU が無ければ導入は `-OriginalOnly`（約4GB・十数分）に切り替わる（supervisor が判定）。
        #   これで作れないのは「音源しか無い曲のカバー」だけ（MIDI のカバー・鼻歌の聞き取りは CPU で動く）。
        "requires_gpu": False,
        "note": "GPU が無い場合は、音源しか無い曲のカバーに使うものを除いて導入します"
                "（オリジナル曲・楽譜〔MIDI〕が見つかる曲のカバー・鼻歌の聞き取りはできます）",
    },
    {
        "id": "tunnel",
        "name": "外部アクセス",
        "desc": "スマホや外出先からミクと話せるようにします",
        "action": "toggle_tunnel",
        "action_label": "有効にする",
        "action_label_done": "無効にする",
        "note": "URLはホームタブの「アクセス先」に出ます",
    },
    {
        "id": "icon",
        "name": "アプリのアイコン",
        "desc": "ミクの姿からアイコンを作り直します",
        "action": "regenerate_icon",
        "action_label": "作る",
        "action_label_done": "作り直す",
    },
]


SETTINGS_SECTIONS: list[dict] = [
    {
        "id": "features",
        "title": "ミクにできること",
        # 唯一、既定で開くセクション。ここに来る目的がいちばん多いため
        "open": True,
        # 機能の行そのものは FEATURES が持つ。ここにあるのは行の中に描く入力欄。
        # enabled 系は**入力の有無から決める**ので画面に出さない
        # （トグルと実体が食い違う状態＝「有効なのにキーが無い」を作れなくするため）
        "fields": [
            _f("api_keys.tavily", "Tavily APIキー", "secret",
               "入力すると Web検索が使えるようになります",
               ui="feature", feature="web_search"),
            _f("web_search.enabled", "Web検索を使う", "bool",
               "画面には出さない。Tavily APIキーの有無から自動で決まる",
               ui="none"),

            _f("email.smtp_user", "送信元 Gmail", "string",
               "ミクがメールを送るときの差出人（Gmail のみ）",
               ui="feature", feature="email"),
            _f("api_keys.gmail_app_password", "アプリパスワード", "secret",
               "Google アカウントで発行した16文字のパスワード",
               ui="feature", feature="email"),
            _f("email.recipient", "送信先", "string",
               "ミクからのメールを受け取るアドレス",
               ui="feature", feature="email"),
            _f("email.enabled", "メール送信を使う", "bool",
               "画面には出さない。上の3項目が揃っているかで自動で決まる",
               ui="none"),

            _f("song_service.enabled", "作曲機能を使う", "bool",
               "画面には出さない。導入されているかで自動で決まる",
               ui="none"),
            _f("system.tunnel_enabled", "外部アクセスを使う", "bool",
               "画面には出さない。機能の行の「有効にする / 無効にする」で切り替える",
               ui="none"),
        ],
    },
    {
        "id": "voice",
        "title": "声",
        "fields": [
            _f("api_keys.fish_audio", "Fish Audio APIキー", "secret",
               "声を作るのに使います。聞き取り（マイク）でも共用します"),
        ],
    },
    {
        # 「声」が**どう聞こえるか**、こちらは**何を話すか**。並べたときに区別がつくよう
        # 「会話」ではなく「会話内容」とする
        "id": "conversation",
        "title": "会話内容",
        "fields": [
            _f("api_keys.google", "Google AI Studio APIキー", "secret",
               "ミクが考えて話すためのキーです。記憶や調べものにも使います。"
               "入れ替えるときだけ触ってください"),
        ],
    },
    {
        "id": "interest",
        "title": "ミクが眺める話題",
        "fields": [
            _f("interest.exposure.enabled", "世の中の話題を眺める", "bool",
               "ミクが自分の興味を見つけるために、ニュースの見出しだけを読みます。"
               "本文やリンク先は読みません"),
            _f("interest.exposure.categories.technology", "テクノロジー", "bool",
               "新しい技術・開発の話題"),
            _f("interest.exposure.categories.learning", "学び・科学", "bool",
               "研究・歴史・美術館の展示など"),
            _f("interest.exposure.categories.life", "暮らし", "bool",
               "料理・住まい・健康など"),
            _f("interest.exposure.categories.fun", "おもしろ", "bool",
               "雑学・体験談など軽い読みもの"),
            _f("interest.exposure.categories.anime_game", "アニメ・ゲーム", "bool",
               "作品・配信の話題"),
            _f("interest.exposure.categories.entertainment", "エンタメ・音楽", "bool",
               "音楽・芸能の話題。ミクの関心と重なりやすい"),
            _f("interest.exposure.categories.news", "時事・ニュース", "bool",
               "世の中の出来事。事件や災害の見出しも含まれます"),
        ],
    },
    {
        "id": "listening",
        "title": "聞き取り（マイク）",
        "fields": [
            _f("stt.provider", "聞き取りの方法", "select",
               "fish_audio はクラウドで聞き取ります（GPU 不要）。"
               "none にすると音声入力を使いません",
               ["none", "fish_audio"], recommended="fish_audio"),
            _f("stt.auto_send.delay_seconds", "送るまでの待ち時間", "int",
               "話し終わってからこれだけ経つと自動で送ります。"
               "声の調子から話し終わりを判断できたときは、待たずに送ります。"
               "ミクが考えている間の追加の話しかけは、これとは別に1秒で送ります",
               unit="秒"),
        ],
    },
    {
        "id": "camera",
        "title": "カメラ・会話中の反応",
        "fields": [
            _f("camera.buffer_size", "カメラの映像を手元に残す枚数", "int",
               "多くすると状況が伝わりやすくなりますが、送る量が増えます", unit="枚"),
            _f("conversation_reaction.enabled", "話している間、ミクが身ぶりで反応する", "bool",
               "こちらが話している最中に、うなずいたり心配そうな顔をしたりします。"
               "**声は出しません**（返事は話し終わってからです）"),
            _f("conversation_reaction.cooldown_seconds", "身ぶりで反応する間隔", "int",
               "前の反応からこれだけ空けます", unit="秒以上あける"),
            _f("conversation_reaction.text_input_enabled", "文字入力のときも反応する", "bool"),
        ],
    },
    {
        "id": "security",
        "title": "ログインの保護",
        "subtitle": "外に公開して使うときは設定してください",
        "fields": [
            # パスワードを入れれば保護ON・消せばOFF（有効/無効のスイッチは作らない。
            # 「スイッチは入っているのに値が無い」食い違いを作れなくする既存方針に揃える）
            _f("security.auth.password_hash", "パスワード", "password",
               "設定すると、会話画面を開くときにこのパスワードが必要になります（4文字以上）。"
               "空にすると保護しません（この PC からのアクセスは常に不要です）",
               maint_links=[{"label": "ログインロックを解除", "action": "unlock_auth"}]),
            _f("security.auth.session_hours", "入力なしで使える時間", "int",
               "一度ログインすると、この時間はブラウザを閉じても入力なしで使えます。既定 24 時間",
               unit="時間"),
            _f("security.auth.lock_threshold", "ロックまでの失敗回数", "int",
               "画面には出さない。総当たり対策の閾値（既定 10 回）", ui="none"),
            _f("security.auth.lock_minutes", "ロックする時間", "int",
               "画面には出さない。失敗が続いたときに待たせる時間（既定 5 分）", ui="none"),
        ],
    },
    {
        "id": "ports",
        "title": "サービスのポート",
        "fields": [
            _f("services.gateway.port", "会話の窓口（Gateway）", "int",
               "ブラウザで開くアドレスの番号になります"),
            _f("services.stt.port", "聞き取り", "int"),
            _f("services.recognition.port", "顔・声の認識", "int", maint_links=[
                {"label": "顔・声の登録状態を確認", "path": "/"},
            ]),
            # 待ち受けアドレス（host）は設定にしない。Gateway 以外はこの PC の中だけ（127.0.0.1）で固定
            _f("services.agent.port", "エージェント", "int"),
            _f("services.kv.port", "会話ログ（KV）", "int"),
            _f("services.vdb.port", "記憶（VDB）", "int", maint_links=[
                {"label": "知識ビューアを開く", "path": "/"},
                {"label": "長期記憶ビューアを開く", "path": "/long_memory"},
            ]),
        ],
    },
    {
        "id": "advanced",
        "title": "詳細",
        "subtitle": "ふだんは変更しません",
        "fields": [
            # --- 声 ---
            _f("tts.fish_audio.sample_rate", "音声のサンプルレート", "int",
               "既定 44100。下げると通信量は減りますが音は粗くなります", unit="Hz"),
            _f("tts.fish_audio.tone_control", "気持ちで声の調子を変える", "bool",
               "入れると、そのときの気持ちに合わせて話す速さ・声の明るさ・声の大きさが変わります。"
               "切ると常に同じ調子で話します"),
            # --- 聞き取り（クラウドで行う場合） ---
            _f("stt.smart_turn.threshold", "話し終わり判断のしきい値", "float",
               "0.0〜1.0。上げると慎重（待ちがち）、下げると即断（切りがち）。既定 0.5"),
            _f("stt.fish_audio.language", "聞き取る言語（クラウド）", "string", "例: ja"),
            _f("stt.fish_audio.silence_chunks", "話し終わりとみなす無音の長さ（クラウド）", "int",
               "1 で 0.1 秒。既定 5（＝0.5秒）の無音が続いたら、そこまでを一区切りにします",
               unit="× 0.1秒"),
            _f("stt.fish_audio.interim_interval_sec", "途中経過を出す間隔（クラウド）", "float",
               "話している最中に、認識の途中経過を入力欄へ反映する間隔です", unit="秒"),
            # --- 作曲・メール ---
            _f("song_service.max_runtime_minutes", "作曲の打ち切り時間", "int",
               "この時間を超えたら作曲を打ち切ります（歌声の書き出しまで済んでいれば、曲はできたものとして知らせます）。"
               "既定 60 分", unit="分"),
        ],
    },
    {
        "id": "file_only",
        "title": "（画面に出さない設定）",
        # 画面に出さない基準は3つ:
        #   1. 変えるとサービスとして破綻する（ホスト・サービスのON/OFF・実行デバイス等）
        #   2. 会話画面から切り替えられる（カメラ・自動送信・話し終わり判断）
        #   3. 意味を理解していないと壊せる／変える理由がほぼ無い（記憶の調整・モデル指定）
        # settings.json を直接編集する人向けに、キーの存在と意味はここに残す
        # （configuration.md にも「画面には出さない」と明記してある）
        "fields": [
            # --- 1. 変えるとサービスとして破綻するもの ---
            _f("services.gateway.host", "Gateway ホスト", "string",
               "画面には出さない。0.0.0.0 固定で運用する（変えると到達できなくなる）", ui="none"),
            # KV・VDB・聞き取り・エージェント・顔と声の認識の host は 2026-10-08 に廃止した。
            # どれも認証を持たないので、この PC の中（127.0.0.1）だけで待ち受ける（コードで固定。
            # 古い settings.json に残っていれば settings_migrate が起動時に外す）
            _f("services.kv.enabled", "会話ログのサービスを使う", "bool",
               "画面には出さない。切ると会話が保存されず、システムとして成り立たない", ui="none"),
            _f("services.vdb.enabled", "記憶のサービスを使う", "bool",
               "画面には出さない。切るとミクが何も覚えていられなくなる", ui="none"),
            _f("tts.fish_audio.backend", "音声合成のバックエンド", "string",
               "画面には出さない。提供側の都合で決まる値で、利用者が選ぶものではない", ui="none"),
            _f("tts.fish_audio.latency", "声を作るときの優先", "select",
               "画面には出さない。既定（速さ優先）から変える理由がほぼ無い",
               ["normal", "balanced"], ui="none"),
            _f("stt.fish_audio.speech_threshold", "話し始めとみなす音の大きさ（クラウド）", "float",
               "画面には出さない。**マイク長押しパネルのスライダーとは別物**（あちらはブラウザ側の"
               "送信開始判定・既定 0.015）で、こちらは中継サーバー側の文末検出にだけ使う。"
               "同名の値が2つ並ぶと混乱するため、体感の調整はスライダーに一本化する", ui="none"),
            _f("stt.fish_audio.ignore_timestamps", "タイムスタンプを受け取らない（クラウド）", "bool",
               "画面には出さない。**常に false**（話者照合が単語タイムスタンプを使うため、"
               "true にすると声の一致度がミクに届かなくなる）", ui="none"),
            _f("email.smtp_host", "SMTPホスト", "string",
               "画面には出さない。Gmail 限定なので smtp.gmail.com 固定", ui="none"),
            _f("email.smtp_port", "SMTPポート", "int",
               "画面には出さない。上と同じ理由（587 固定）", ui="none"),

            # --- 2. 会話画面から切り替えられるもの（設定で二重に持たない）---
            _f("camera.enabled", "カメラを使う", "bool",
               "画面には出さない。会話画面のカメラボタンで切り替える", ui="none"),
            _f("stt.auto_send.enabled", "話し終わったら自動で送る", "bool",
               "画面には出さない。会話画面のマイクボタンで切り替える", ui="none"),
            _f("stt.smart_turn.enabled", "話し終わりを声の調子から判断する", "bool",
               "画面には出さない。切る意味がほぼ無く、切ると待ち時間だけの判断に退化する", ui="none"),

            # --- 3. 変える理由がほぼ無い／理解していないと壊せるもの ---
            _f("llm.flash_model", "ふだんの応答に使うモデル", "string",
               "画面には出さない。動作確認済みの組み合わせから外れると壊れる", ui="none"),
            _f("llm.pro_model", "じっくり考えるときのモデル", "string",
               "画面には出さない。上と同じ理由", ui="none"),
            _f("llm.lite_model", "軽い仕事（つなぎ発話・曲の選び出しなど）に使うモデル", "string",
               "画面には出さない。上と同じ理由", ui="none"),
            _f("llm.embedding_model", "記憶の検索に使うモデル", "string",
               "画面には出さない。変えると既存の記憶を検索できなくなる（次元が変わるため）", ui="none"),
            _f("llm.filler_enabled", "考えている間に「えーと」などを挟む", "bool",
               "画面には出さない。切ると無言で待つだけになる", ui="none"),
            _f("song_service.pro_model", "作曲で pro を指定した工程に使うモデル", "string",
               "画面には出さない。会話のモデルとは別に作曲だけのモデルを決める（無ければ llm.pro_model）。"
               "2026-10-18 から pro を指定する工程は無い（大事な判断の工程も flash を指定する）。"
               "動作確認済みの組み合わせから外れると壊れる", ui="none"),
            _f("song_service.flash_model", "作曲のほぼ全部（大事な判断・進行役・点検役）に使うモデル", "string",
               "画面には出さない。上と同じ理由（無ければ llm.flash_model）", ui="none"),
            _f("song_service.lite_model", "作曲の機械的な作業に使うモデル", "string",
               "画面には出さない。上と同じ理由（無ければ llm.lite_model）", ui="none"),
            _f("tts.fish_audio.voice_id", "声のモデルID", "string",
               "画面には出さない。既定でミクの声に設定済みで、変える理由がほぼ無いため", ui="none"),
            _f("memory.compress_threshold", "記憶をまとめ直す分量（文字）", "int",
               "画面には出さない。いつも思い出せるようにしている記憶の合計がこの文字数を超えると、"
               "次の記憶整理のときに不要なものを削除・統合して約15,000字まで縮める",
               ui="none"),
            _f("memory.recent_conversation.limit", "前の会話をさかのぼって覚えている件数", "int",
               "画面には出さない。ミクが毎回思い出しながら話す、直前までの会話の件数。"
               "7 でおおよそ 2 週間ぶんにあたる", ui="none"),
            _f("memory.recent_conversation.full_text_count", "そのうち中身まで覚えている件数", "int",
               "画面には出さない。新しいものから数えてこの件数までは会話の中身ごと、"
               "それより前は要点だけを覚えている（人が昨日のことは詳しく、先週のことは"
               "ざっくり覚えているのと同じ形）。要点だけの会話も、詳しい内容が必要になれば"
               "ミクが思い出して答える", ui="none"),
            _f("memory.knowledge_base.top_k", "記憶・知識を探すときの件数", "int",
               "画面には出さない。多くすると話が広がるが応答は遅くなる", ui="none"),
            _f("memory.knowledge_base.chunk_size_child", "知識を分割する単位（文字）", "int",
               "画面には出さない。知識登録時の分割サイズ", ui="none"),
            _f("logs.archive_enabled", "古い動作ログを自動でまとめる", "bool",
               "画面には出さない。ミクの動作ログは1か月で200MB以上たまるため、"
               "30日より古いものを月ごとに1つのファイルへ自動でまとめている"
               "（中身は失われず、必要なときに取り出せる）。止める理由がほぼ無い", ui="none"),
            _f("logs.keep_days", "動作ログをそのまま残す日数", "int",
               "画面には出さない。この日数より古いログがまとめられる。"
               "**ディスク使用量はほぼこの日数で決まる**（30日でおよそ 226MB）", ui="none"),
            _f("system.skip_memory", "起動時に記憶の整理をスキップする", "bool",
               "画面には出さない（ホームタブのチェックで操作する）", ui="none"),
            _f("system.keep_tunnel_on_stop", "停止してもアクセスURLを保つ", "bool",
               "画面には出さない（ホームタブのチェックで操作する）", ui="none"),
        ],
    },
]

# path → フィールド定義 の逆引き（settings_store のバリデーションで使用）
FIELDS_BY_PATH: dict[str, dict] = {
    field["path"]: field
    for section in SETTINGS_SECTIONS
    for field in section["fields"]
}
