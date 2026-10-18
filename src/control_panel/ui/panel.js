/*
 * Watashinosekaito Hatsune Miku
 * Copyright (c) 2025-2026 ChordDotE. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-ChordDotE-NonCommercial-2.0
 *
 * NON-COMMERCIAL USE ONLY. NO BUSINESS USE. NO ADAPTATION.
 *
 * Business use (any use by or for a company, organization, or sole
 * proprietor, WHETHER OR NOT ANY PAYMENT IS INVOLVED) and publishing or
 * distributing modified versions are PROHIBITED IN ALL CASES; no permission
 * is granted for them. Forks may be used only to propose improvements to the
 * Author (e.g. pull requests). Commercial use, redistribution, and Adaptation
 * are PROHIBITED without prior written permission. "Adaptation" includes
 * manual reimplementation, automated translation, and generation using large
 * language models or other machine learning, of software having equivalent or
 * similar functionality. Permission is granted individually, in recorded form.
 * Contact: https://x.com/ChordDotE
 *
 * 業務利用（対価の有無を問わず、法人・団体・個人事業主による利用と事業のための利用）と、
 * 改変版の公開・配布は、許可の余地なく禁止します。フォークは作者への改善の提案
 * （プルリクエスト等）のためだけに使えます。商用利用・再配布・翻案は、作者の許可が
 * ない限り禁止します。翻案には、再実装・自動変換・大規模言語モデル等を用いた生成に
 * よる、同等または類似の機能を持つソフトウェアの作成を含みます。
 * 許可が必要な場合は、記録の残る形でご連絡ください。
 *
 * See LICENSE for full terms. 詳細は LICENSE を参照してください。
 */

'use strict';
/* コントロールパネルの画面制御。
   Python 側の処理は window.pywebview.api 経由でのみ呼ぶ（この層に業務ロジックを書かない）。 */

const el = (id) => document.getElementById(id);
let statusTimer = null;

// pywebview の API が使えるようになるまで待つ
function whenReady() {
  return new Promise((resolve) => {
    if (window.pywebview && window.pywebview.api) return resolve();
    window.addEventListener('pywebviewready', () => resolve(), { once: true });
  });
}

// 進捗表示 ------------------------------------------------------------------
// セットアップ系のスクリプトは各段階の頭で「[2/7] venv を構築しています」を出す。
// 導入は 40〜90分かかることがあり（作曲機能を GPU ありで入れるとき）、文字が流れるだけでは残りが分からないため、
// この印を拾ってバーと「2/7」に反映する。印のない行はそのまま下段へ流す。
const STEP_MARKER = /^\[(\d+)\/(\d+)\]\s*(.*)$/;
let stepTotal = 0;

// 起動・停止・再起動の間は進捗欄を出さない（2026-08-15 ユーザー判断）。
// これらは段階印を持たず「起動を待っています…」が数分間そのまま出るだけで、
// 進み具合が分からない割に画面下端の見切れる位置に居座っていた。
// 状態パネルの「起動しています… ⏳」だけで足りる。段階印を持つ導入処理（歌機能の
// 導入・アイコン生成など）では従来どおり出す。
// ただし**失敗の知らせだけは拾って状態パネルに残す**。起動の失敗を伝える経路はここしかなく、
// 進捗欄は処理の終わりに消える（＝一瞬で流れて気づけない）ため。
let suppressProgress = false;
let troubleMsg = null;
const TROUBLE_MARKER = /失敗|できません|できませんでした|見つかりません|タイムアウト/;

// supervisor からの進捗メッセージ（Python が evaluate_js で呼ぶ）
window.onProgress = (message) => {
  if (TROUBLE_MARKER.test(message)) {
    troubleMsg = message;
    el('state-sub').textContent = message;
  }
  if (suppressProgress) return;
  el('progress').classList.remove('hidden');
  const m = STEP_MARKER.exec(message);
  if (m) {
    const n = Number(m[1]);
    stepTotal = Number(m[2]);
    el('pfill').style.width = `${Math.round((n / stepTotal) * 100)}%`;
    el('pstep').textContent = `${n}/${stepTotal}  ${m[3]}`;
    el('pline').textContent = '';
    return;
  }
  // 段階が始まる前（段階を持たない処理）は、その行自体を見出しにする
  if (stepTotal === 0) el('pstep').textContent = message;
  else el('pline').textContent = message;
};

// 処理が終わったら消す。空欄のバーが居座ると「まだ動いている」ように見えるため
function clearProgress() {
  el('progress').classList.add('hidden');
  el('pfill').style.width = '0';
  el('pstep').textContent = '';
  el('pline').textContent = '';
  stepTotal = 0;
  suppressProgress = false;
}

// 状態パネルの3態。起動は記憶の整理を含むと数分かかるため、その間を
// 「停止中」と出すと止まって見える。処理中は Python が返す busy_action で出し分ける
const STATE_VIEW = {
  start:   { cls: 'busy', icon: '⏳', label: '起動しています…', sub: 'ミクの準備をしています' },
  restart: { cls: 'busy', icon: '⏳', label: '再起動しています…', sub: '設定を反映しています' },
  stop:    { cls: 'busy', icon: '⏳', label: '停止しています…', sub: 'ミクの記憶を保存しています' },
  on:      { cls: 'on',   icon: '🎤', label: '動作中', sub: 'ミクと会話できます' },
  off:     { cls: 'off',  icon: '💤', label: '停止中', sub: '「起動」を押すと会話をはじめられます' },
};

let lastStatus = null;   // 押した直後の表示切り替えに使う（下の showBusy）

function renderStatus(st) {
  lastStatus = st;
  const running = st.running;
  const view = STATE_VIEW[st.busy ? st.busy_action : null]
    || STATE_VIEW[running ? 'on' : 'off'];
  const box = el('status');
  box.className = 'status ' + view.cls;
  el('state-icon').textContent = view.icon;
  el('state-label').textContent = view.label;
  // 失敗の知らせは次の操作を始めるまで残す（処理の終わりに消えると気づけない）
  el('state-sub').textContent = troubleMsg || view.sub;

  el('console-note').classList.toggle('hidden', !st.console_attached);

  el('btn-start').disabled = running || st.busy;
  el('btn-stop').disabled = !running || st.busy;
  el('btn-open').disabled = !running;
  // 設定は停止中でも編集できる（APIキーの誤りで起動できない状態からの復帰路）

  el('opt-skip-memory').checked = !!st.skip_memory;
  el('opt-keep-tunnel').checked = !!st.keep_tunnel_on_stop;
  el('opt-tunnel-row').style.display = st.tunnel_enabled ? '' : 'none';

  renderUrls(st);
}

function renderUrls(st) {
  const holder = el('urls');
  holder.innerHTML = '';
  if (!st.running) {
    holder.innerHTML = '<div class="url empty">起動するとアクセス先が表示されます</div>';
    return;
  }
  addUrl(holder, 'このPC', `http://localhost:${st.port}`);
  if (st.tunnel_url) addUrl(holder, '外出先', st.tunnel_url);
}

function addUrl(holder, label, url) {
  const row = document.createElement('div');
  row.className = 'url';
  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = label;
  const code = document.createElement('code');
  code.textContent = url;
  const btn = document.createElement('button');
  btn.textContent = 'コピー';
  btn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(url);
      btn.textContent = 'コピー済';
      setTimeout(() => { btn.textContent = 'コピー'; }, 1500);
    } catch (e) { /* クリップボードが使えない環境では何もしない */ }
  };
  row.append(lbl, code, btn);
  holder.appendChild(row);
}

async function refresh() {
  const st = await window.pywebview.api.get_status();
  renderStatus(st);
}

/** 押した瞬間に「起動しています…」へ切り替える。
 *  Python 側は処理の間ずっと応答を返さないため、次の定期更新（5秒後）まで
 *  待つと押しても何も起きないように見える。 */
function showBusy(action) {
  suppressProgress = true;   // 起動・停止・再起動の間は下端の進捗欄を出さない
  troubleMsg = null;         // 前回の失敗の知らせは、次の操作を始めた時点で消す
  renderStatus({ ...(lastStatus || {}), busy: true, busy_action: action });
}

async function refreshLog() {
  el('log').textContent = await window.pywebview.api.get_log(200);
}

// 設定タブ ------------------------------------------------------------------
// 「ミクにできること」（機能の導入と、それに必要な入力）と、細かな調整を1枚にまとめる。
// 画面に出す項目・並び・文言はすべて Python 側の settings_schema が持つ（ここでは決めない）。

let settingsData = null;      // get_settings() の結果
const edited = new Map();     // 変更した項目 path -> 値

async function loadSettings() {
  const data = await window.pywebview.api.get_settings();
  if (!data.success) {
    el('settings').innerHTML = '';
    setSaveMessage(data.error || '設定を読み込めませんでした', true);
    return;
  }
  settingsData = data;
  edited.clear();
  renderSettings();
  updateSaveState();
}

function renderSettings() {
  const holder = el('settings');
  holder.innerHTML = '';
  for (const sec of settingsData.sections) {
    holder.appendChild(renderSection(sec));
  }
}

function renderSection(sec) {
  const box = document.createElement('details');
  box.className = 'group';
  // 開くのは「ミクにできること」だけ。全部開くと目的の項目に辿り着けない
  box.open = !!sec.open;

  const head = document.createElement('summary');
  head.textContent = sec.title;
  if (sec.subtitle) {
    const sub = document.createElement('span');
    sub.className = 'sub';
    sub.textContent = sec.subtitle;
    head.appendChild(sub);
  }

  const body = document.createElement('div');
  body.className = 'gbody';
  if (sec.id === 'features') {
    for (const feat of settingsData.features) {
      body.appendChild(renderFeature(feat, sec.fields.filter((f) => f.feature === feat.id)));
    }
  } else {
    for (const field of sec.fields) body.appendChild(renderField(field));
  }

  box.append(head, body);
  return box;
}

/** 機能の行。状態は設定値ではなく**実体**（導入済みか）で出す */
function renderFeature(feat, fields) {
  const st = settingsData.status;
  const row = document.createElement('div');
  row.className = 'srow';

  const state = featureState(feat, fields, st);

  const head = document.createElement('div');
  head.className = 'fhead';
  const info = document.createElement('div');
  info.className = 'finfo';
  const name = document.createElement('div');
  name.className = 'fname';
  name.textContent = feat.name;
  const desc = document.createElement('div');
  desc.className = 'fdesc';
  desc.textContent = feat.desc;
  info.append(name, desc);

  const badge = document.createElement('span');
  badge.className = 'badge ' + (state.ready ? 'ok' : (state.blocked ? 'warn' : 'no'));
  badge.textContent = state.label;
  head.append(info, badge);

  if (feat.action) {
    const btn = document.createElement('button');
    btn.textContent = state.ready ? feat.action_label_done : feat.action_label;
    btn.disabled = state.blocked;
    btn.onclick = () => runFeatureAction(feat, state, btn);
    head.appendChild(btn);
  }
  row.appendChild(head);

  for (const field of fields) row.appendChild(renderField(field, true));

  if (state.note) row.appendChild(noteEl(state.note));
  if (feat.note) row.appendChild(noteEl(feat.note));
  if (feat.link) {
    const link = document.createElement('div');
    link.className = 'flink';
    link.textContent = `${feat.link_label}: ${feat.link}`;
    row.appendChild(link);
  }
  return row;
}

/** 機能の状態（導入済みか・操作できるか・添える注意書き） */
function featureState(feat, fields, st) {
  const filled = (f) => {
    const v = edited.has(f.path) ? edited.get(f.path) : f.value;
    return typeof v === 'string' ? v.trim() !== '' : !!v;
  };
  switch (feat.id) {
    case 'live2d':
      return { ready: st.live2d_ready, label: st.live2d_ready ? '導入済み' : '未導入' };
    case 'icon':
      return {
        ready: st.icon_ready,
        blocked: !st.live2d_ready,
        label: st.icon_ready ? 'ミクの姿から作成済み' : '同梱のものを使用中',
      };
    case 'song':
      return {
        ready: st.song_ready,
        // GPU が無くても導入できる（2026-09-01 から supervisor が -OriginalOnly に切り替える。
        //   画面だけ 2026-10-03 まで押せなくしていた）。何を入れるかの説明は settings_schema の
        //   note（下に並ぶ）に1つだけ置き、ここは「GPU が無い」という事実だけを添える
        label: st.song_ready ? '導入済み' : '未導入',
        note: st.gpu ? '' : 'この環境では NVIDIA GPU が見つかりませんでした',
      };
    case 'tunnel':
      return {
        ready: st.tunnel_enabled,
        label: st.tunnel_enabled ? '有効' : '無効',
        // 有効なのに実体が無い＝起動しても外部アクセスできない。黙って落ちないよう明示する
        note: (st.tunnel_enabled && !st.tunnel_ready)
          ? '⚠ cloudflared が見つかりません。いちど「無効にする」→「有効にする」で取得してください'
          : '',
      };
    default: {
      // 入力で決まる機能（Web検索・メール送信）。すべて埋まっていれば使える
      const ready = fields.length > 0 && fields.every(filled);
      return { ready, label: ready ? '使えます' : '未設定' };
    }
  }
}

async function runFeatureAction(feat, state, btn) {
  const api = window.pywebview.api;
  btn.disabled = true;
  try {
    if (feat.action === 'toggle_tunnel') await api.set_tunnel(!state.ready);
    else if (feat.action === 'install_song_service') await api.install_song_service();
    else if (feat.action === 'regenerate_icon') await api.regenerate_icon();
    else if (feat.action === 'setup_live2d') await api.setup_live2d();
  } finally {
    clearProgress();
    settingsData.status = await api.get_features();
    renderSettings();
  }
}

function noteEl(text) {
  const note = document.createElement('div');
  note.className = 'fnote';
  note.textContent = text;
  return note;
}

/** 1項目の入力欄。型は Python 側のスキーマが決める */
function renderField(field, inFeature) {
  const row = document.createElement('div');
  row.className = inFeature ? 'field' : 'srow field';

  const line = document.createElement('div');
  line.className = 'fline';
  const label = document.createElement('label');
  label.textContent = field.label;
  line.appendChild(label);

  let input;
  if (field.type === 'bool') {
    input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = !!field.value;
    input.onchange = () => setEdited(field, input.checked);
  } else if (field.type === 'select') {
    input = document.createElement('select');
    for (const choice of field.choices || []) {
      const opt = document.createElement('option');
      opt.value = choice;
      opt.textContent = choice + (choice === field.recommended ? '（推奨）' : '');
      input.appendChild(opt);
    }
    input.value = field.value ?? '';
    input.onchange = () => setEdited(field, input.value);
  } else {
    input = document.createElement('input');
    input.type = (field.type === 'secret' || field.type === 'password') ? 'password'
      : (field.type === 'int' || field.type === 'float') ? 'number' : 'text';
    input.value = field.value ?? '';
    if (field.type === 'secret') input.placeholder = '未設定';
    if (field.type === 'password') {
      // 保存されているのはハッシュなので元の文字は出せない。設定済みかどうかだけを示す
      input.placeholder = field.value ? '設定済み（変えるときだけ入力）' : '未設定（保護しません）';
      input.autocomplete = 'new-password';
    }
    input.oninput = () => setEdited(field, input.value);
  }
  input.dataset.path = field.path;
  line.appendChild(input);

  if (field.unit) {
    const unit = document.createElement('span');
    unit.className = 'unit';
    unit.textContent = field.unit;
    line.appendChild(unit);
  }
  row.appendChild(line);

  if (field.desc) {
    const desc = document.createElement('div');
    desc.className = 'fdesc';
    desc.textContent = field.desc;
    row.appendChild(desc);
  }
  for (const link of field.maint_links || []) {
    row.appendChild(maintLink(field, link));
  }
  return row;
}

/** フィールドに添えるボタン。稼働中しか使えない。
 *  link.path があればメンテページを開き、link.action があれば API を呼ぶ */
function maintLink(field, link) {
  const a = document.createElement('button');
  a.className = 'maint';
  a.textContent = link.label;
  const running = settingsData.system_running;
  a.disabled = !running;
  a.title = running ? '' : (link.action ? '起動すると使えます' : '起動すると開けます');
  a.onclick = async () => {
    if (link.action === 'unlock_auth') {
      a.disabled = true;
      const before = a.textContent;
      a.textContent = '解除しています…';
      try {
        const res = await window.pywebview.api.unlock_auth();
        setSaveMessage(res.message || '解除しました', !res.success);
      } finally {
        a.textContent = before;
        a.disabled = false;
      }
      return;
    }
    const port = edited.has(field.path) ? edited.get(field.path) : field.value;
    window.pywebview.api.open_url(`http://127.0.0.1:${port}${link.path}`);
  };
  return a;
}

function setEdited(field, value) {
  edited.set(field.path, value);
  // 機能の状態表示（未設定→使えます）を、保存を待たずに追従させる
  if (field.ui === 'feature') {
    const sec = settingsData.sections.find((s) => s.id === 'features');
    const target = sec && sec.fields.find((f) => f.path === field.path);
    if (target) target.value = value;
  }
  updateSaveState();
}

function updateSaveState() {
  el('btn-save').disabled = edited.size === 0;
  if (edited.size > 0) setSaveMessage(`${edited.size} 件を変更しました`, false);
}

function setSaveMessage(text, isError) {
  const msg = el('save-msg');
  msg.textContent = text;
  msg.className = 'save-msg' + (isError ? ' error' : '');
}

async function saveSettings() {
  if (edited.size === 0) return;
  el('btn-save').disabled = true;
  setSaveMessage('保存しています…', false);
  const result = await window.pywebview.api.save_settings(Object.fromEntries(edited));
  if (!result.success) {
    const errors = result.errors
      ? Object.entries(result.errors).map(([k, v]) => `${k}: ${v}`).join(' / ')
      : (result.error || '保存に失敗しました');
    setSaveMessage(errors, true);
    el('btn-save').disabled = false;
    return;
  }
  await loadSettings();
  const notices = (result.notices || []).join(' / ');
  setSaveMessage('保存しました。反映にはミクの再起動が必要です' + (notices ? ` ／ ${notices}` : ''), false);
  offerRestart();
}

/** 保存後に「今すぐ再起動」を出す。停止中は出さない（起動はホームタブの操作に任せる） */
function offerRestart() {
  if (!settingsData || !settingsData.system_running) return;
  const btn = el('btn-restart');
  btn.hidden = false;
  btn.disabled = false;
  btn.textContent = '今すぐ再起動して反映（記憶の整理はスキップ）';
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = '再起動しています…';
    showBusy('restart');
    try {
      renderStatus(await window.pywebview.api.restart());
    } finally {
      clearProgress();
      btn.hidden = true;
      await loadSettings();
    }
  };
}

async function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t) => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  for (const tab of ['home', 'settings', 'log']) {
    el('tab-' + tab).classList.toggle('hidden', name !== tab);
  }
  if (name === 'log') refreshLog();
  // 開くたびに読み直す。導入状況や稼働状態（メンテページのリンク可否）を最新にするため
  if (name === 'settings') await loadSettings();
}

whenReady().then(async () => {
  document.querySelectorAll('.tab').forEach((t) => {
    t.onclick = () => switchTab(t.dataset.tab);
  });

  el('btn-start').onclick = async () => {
    el('btn-start').disabled = true;
    showBusy('start');
    try {
      renderStatus(await window.pywebview.api.start(el('opt-skip-memory').checked));
    } finally {
      clearProgress();
    }
  };
  el('btn-stop').onclick = async () => {
    el('btn-stop').disabled = true;
    showBusy('stop');
    try {
      renderStatus(await window.pywebview.api.stop(el('opt-keep-tunnel').checked));
    } finally {
      clearProgress();
    }
  };
  el('btn-open').onclick = () => window.pywebview.api.open_browser();
  el('btn-settings').onclick = () => switchTab('settings');
  el('btn-save').onclick = saveSettings;
  el('btn-reload-log').onclick = refreshLog;
  el('btn-open-log').onclick = () => window.pywebview.api.open_log_folder();

  // お問い合わせ・お知らせ（2026-09-22）。行き先は index.html の data-url が持つ。
  // **必ず外部ブラウザで開く**（パネル内で開くと戻る手段が無く、起動操作ごと見失う）
  document.querySelectorAll('.linkrow').forEach((b) => {
    b.onclick = () => window.pywebview.api.open_url(b.dataset.url);
  });

  el('opt-skip-memory').onchange = (e) =>
    window.pywebview.api.set_option('skip_memory', e.target.checked);
  el('opt-keep-tunnel').onchange = (e) =>
    window.pywebview.api.set_option('keep_tunnel_on_stop', e.target.checked);

  await refresh();
  // 別の手段（bat 等）で起動・停止された場合にも表示を追従させる
  statusTimer = setInterval(refresh, 5000);
});
