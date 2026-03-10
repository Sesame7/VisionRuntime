const REFRESH_MS = 1000;
const HEARTBEAT_STALE_POLLS = 3;
const ID_PAD_LEN = 5;
const STATUS_TIMEOUT_MS = 1200;
const TRIGGER_TIMEOUT_MS = 1500;
const RECIPE_SWITCH_TIMEOUT_MS = 12000;
const BATCH_SWITCH_TIMEOUT_MS = 3000;

const els = {
  img: document.getElementById('img'),
  titleHeader: document.getElementById('titleHeader'),
  clockTime: document.getElementById('clockTime'),
  statusTime: document.getElementById('statusTime'),
  onlineBox: document.getElementById('onlineBox'),
  onlineText: document.getElementById('onlineText'),
  triggerBtn: document.getElementById('triggerBtn'),
  batchPanel: document.getElementById('batchPanel'),
  batchCurrent: document.getElementById('batchCurrent'),
  batchInput: document.getElementById('batchInput'),
  batchSetBtn: document.getElementById('batchSetBtn'),
  batchMsg: document.getElementById('batchMsg'),
  recipeWrap: document.querySelector('.recipe-wrap'),
  recipeToggle: document.getElementById('recipeToggle'),
  recipeMenu: document.getElementById('recipeMenu'),
  statTotal: document.getElementById('statTotal'),
  statOk: document.getElementById('statOk'),
  statNg: document.getElementById('statNg'),
  statErr: document.getElementById('statErr'),
  statPass: document.getElementById('statPass'),
  tbody: document.querySelector('#records tbody'),
};

let refreshInFlight = false;
let cachedRecords = [];
let lastSeenSeq = null;
let recipeSwitchInFlight = false;
let recipeMenuOpen = false;
let batchSwitchInFlight = false;

async function fetchWithTimeout(url, options, timeoutMs){
  const ctrl = new AbortController();
  const timer = setTimeout(function(){ ctrl.abort(); }, timeoutMs);
  try{
    return await fetch(url, {...(options || {}), signal: ctrl.signal});
  } finally {
    clearTimeout(timer);
  }
}

async function readJsonSafe(response){
  try{
    return await response.json();
  } catch(_e){
    return {};
  }
}

async function postJson(path, payload, timeoutMs){
  const response = await fetchWithTimeout(
    path,
    {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    },
    timeoutMs,
  );
  const data = await readJsonSafe(response);
  return {response: response, data: data};
}

async function doTrigger(){
  try{
    const response = await fetchWithTimeout('/trigger', {method:'POST'}, TRIGGER_TIMEOUT_MS);
    if (!response.ok){
      throw new Error(`trigger_http_${response.status}`);
    }
    await readJsonSafe(response);
  } catch(e){
    console.error(e);
  }
}

async function switchRecipe(slot){
  const slotVal = Number(slot);
  if (!Number.isInteger(slotVal) || slotVal <= 0 || recipeSwitchInFlight) return;
  recipeSwitchInFlight = true;
  let switched = false;
  closeRecipeMenu();
  els.recipeToggle.disabled = true;
  try{
    const result = await postJson(
      '/recipe/select',
      {slot: slotVal},
      RECIPE_SWITCH_TIMEOUT_MS,
    );
    if (!result.response.ok || !result.data.ok){
      const msg = result.data.message || `recipe_http_${result.response.status}`;
      throw new Error(msg);
    }
    switched = true;
  } catch(e){
    console.error(e);
  } finally {
    if (switched){
      cachedRecords = [];
      lastSeenSeq = null;
      lastImgId = null;
    }
    recipeSwitchInFlight = false;
    await refresh();
  }
}

function closeRecipeMenu(){
  recipeMenuOpen = false;
  els.recipeMenu.hidden = true;
  els.recipeToggle.setAttribute('aria-expanded', 'false');
}

function openRecipeMenu(){
  if (els.recipeToggle.disabled) return;
  recipeMenuOpen = true;
  els.recipeMenu.hidden = false;
  els.recipeToggle.setAttribute('aria-expanded', 'true');
}

function toggleRecipeMenu(){
  if (recipeMenuOpen){
    closeRecipeMenu();
  } else {
    openRecipeMenu();
  }
}

function renderBatch(batchState){
  if (!batchState || typeof batchState !== 'object'){
    els.batchPanel.hidden = true;
    els.batchMsg.hidden = true;
    els.batchSetBtn.disabled = true;
    return;
  }
  els.batchPanel.hidden = false;
  const current = String(batchState.current || '');
  els.batchCurrent.innerText = current || '=@NO-BATCH_+';
  updateBatchSetButton();
}

function updateBatchSetButton(){
  const hasInput = String(els.batchInput.value || '').trim().length > 0;
  els.batchSetBtn.disabled = batchSwitchInFlight || !hasInput;
}

function showBatchMessage(message, isError){
  const text = String(message || '').trim();
  if (!text){
    els.batchMsg.hidden = true;
    els.batchMsg.innerText = '';
    els.batchMsg.className = 'batch-msg';
    return;
  }
  els.batchMsg.hidden = false;
  els.batchMsg.className = isError ? 'batch-msg err' : 'batch-msg success';
  els.batchMsg.innerText = text;
}

async function setBatch(){
  const name = String(els.batchInput.value || '').trim();
  if (!name || batchSwitchInFlight){
    return;
  }
  batchSwitchInFlight = true;
  els.batchSetBtn.disabled = true;
  showBatchMessage('', false);
  try{
    const result = await postJson(
      '/batch/select',
      {name: name},
      BATCH_SWITCH_TIMEOUT_MS,
    );
    if (!result.response.ok || !result.data.ok){
      const msg = result.data.message || `batch_http_${result.response.status}`;
      throw new Error(msg);
    }
    els.batchInput.value = '';
    showBatchMessage(result.data.message || 'batch set', false);
  } catch(e){
    console.error(e);
    showBatchMessage((e && e.message) ? e.message : 'batch set failed', true);
  } finally {
    batchSwitchInFlight = false;
    await refresh();
  }
}

function normalizeRecipeSlots(rawSlots){
  return (Array.isArray(rawSlots) ? rawSlots : []).map(function(item){
    return {
      slot: Number((item || {}).slot || 0),
      name: String((item || {}).name || ''),
      valid: Boolean((item || {}).valid),
      error: String((item || {}).error || ''),
    };
  }).filter(function(item){
    return Number.isInteger(item.slot) && item.slot > 0 && Boolean(item.name);
  });
}

function buildRecipeItem(item, currentSlot, switching){
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'recipe-item';
  if (item.slot === currentSlot){
    btn.classList.add('current');
  }
  if (!item.valid){
    btn.classList.add('invalid');
  }
  btn.disabled = switching || !item.valid;
  if (item.error){
    btn.title = item.error;
  }

  const nameEl = document.createElement('span');
  nameEl.innerText = item.name;
  btn.appendChild(nameEl);

  if (item.slot === currentSlot){
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.innerText = 'Current';
    btn.appendChild(badge);
  } else if (!item.valid){
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.innerText = 'Invalid';
    btn.appendChild(badge);
  }

  btn.addEventListener('click', function(ev){
    ev.stopPropagation();
    closeRecipeMenu();
    void switchRecipe(item.slot);
  });
  return btn;
}

function renderRecipes(recipeState){
  const state = recipeState || {};
  const slots = normalizeRecipeSlots(state.slots);
  const currentSlot = Number(state.current_slot || 0);
  const switching = Boolean(state.switching) || recipeSwitchInFlight;
  const totalSlots = Number(state.total_slots || 0);
  const menu = els.recipeMenu;
  const toggle = els.recipeToggle;

  menu.innerHTML = '';
  if (!slots.length || totalSlots <= 0){
    toggle.innerText = 'N/A';
    toggle.disabled = true;
    closeRecipeMenu();
    return;
  }

  slots.forEach(function(item){
    menu.appendChild(buildRecipeItem(item, currentSlot, switching));
  });

  const current = slots.find(function(item){ return item.slot === currentSlot; });
  if (current){
    toggle.innerText = current.name;
  } else {
    toggle.innerText = '--';
  }
  toggle.disabled = switching;
  if (switching){
    closeRecipeMenu();
  }
}
let lastImgId = null;
let lastHeartbeatSeq = null;
let hasHeartbeatProgress = false;
let staleHeartbeatPolls = HEARTBEAT_STALE_POLLS;
const baseStats = {total:0,ok:0,ng:0,error:0,pass_rate:0};
function resultClass(result){
  const r = String(result || '').toUpperCase();
  return (r === 'OK') ? 'ok' : ((r === 'NG') ? 'ng' : 'err');
}
function renderStats(stats){
  const s = stats || baseStats;
  els.statTotal.innerText = s.total;
  els.statOk.innerText = s.ok;
  els.statNg.innerText = s.ng;
  els.statErr.innerText = s.error || 0;
  els.statPass.innerText = (s.pass_rate * 100).toFixed(1);
}

function toIsoParts(ms){
  if (!Number.isFinite(ms)){
    return {date: '', time: ''};
  }
  const iso = new Date(ms).toISOString();
  return {
    date: iso.slice(0, 10),
    time: iso.slice(11, 24),
  };
}

function renderRecords(records, maxRecords){
  const limit = (typeof maxRecords === 'number' && maxRecords > 0) ? maxRecords : records.length;
  const showRecs = records.slice(0, limit);
  els.tbody.innerHTML = '';
  showRecs.forEach(function(rec){
    const id = rec.trigger_seq || 0;
    const duration = typeof rec.duration_ms==='number' ? rec.duration_ms : 0;
    const triggeredAtMs = Number(rec.triggered_at_ms);
    const parts = toIsoParts(triggeredAtMs);
    const cls = resultClass(rec.result);
    const tr=document.createElement('tr');
    tr.innerHTML = `<td>${String(id).padStart(ID_PAD_LEN,'0')}</td>
                    <td class="${cls}">${rec.result}</td>
                    <td>${parts.date}</td>
                    <td>${parts.time}</td>
                    <td>${duration.toFixed(1)}</td>`;
    els.tbody.appendChild(tr);
  });
  return showRecs;
}
function renderOnline(state){
  if (state === 'ok'){
    els.onlineBox.className = 'online ok';
    els.onlineText.innerText = 'Runtime: Online';
    return;
  }
  if (state === 'pending'){
    els.onlineBox.className = 'online';
    els.onlineText.innerText = 'Runtime: Connecting...';
    return;
  }
  els.onlineBox.className = 'online ng';
  els.onlineText.innerText = 'Runtime: Offline';
}

function staleOnlineState(){
  return hasHeartbeatProgress && staleHeartbeatPolls < HEARTBEAT_STALE_POLLS
    ? 'ok'
    : 'ng';
}

function updateHeartbeatOnline(heartbeatSeq){
  const hasSeq = typeof heartbeatSeq === 'number' && Number.isFinite(heartbeatSeq);
  if (!hasSeq){
    staleHeartbeatPolls += 1;
    renderOnline(staleOnlineState());
    return;
  }
  if (lastHeartbeatSeq === null){
    lastHeartbeatSeq = heartbeatSeq;
    staleHeartbeatPolls = 0;
    renderOnline('pending');
    return;
  }
  if (heartbeatSeq > lastHeartbeatSeq){
    lastHeartbeatSeq = heartbeatSeq;
    hasHeartbeatProgress = true;
    staleHeartbeatPolls = 0;
    renderOnline('ok');
    return;
  }
  if (heartbeatSeq < lastHeartbeatSeq){
    lastHeartbeatSeq = heartbeatSeq;
    hasHeartbeatProgress = false;
    staleHeartbeatPolls = 0;
    renderOnline('pending');
    return;
  }
  staleHeartbeatPolls += 1;
  renderOnline(staleOnlineState());
}
function tickClock(){
  els.clockTime.innerText = new Date().toLocaleTimeString([], { hour12: false });
}

function mergeStatusRecords(records, maxRecords, fullSnapshot){
  if (fullSnapshot || lastSeenSeq === null){
    cachedRecords = records.slice(0, maxRecords > 0 ? maxRecords : records.length);
    return;
  }
  if (!records.length){
    return;
  }
  const seen = new Set(cachedRecords.map(function(rec){ return rec.trigger_seq; }));
  for (let i = records.length - 1; i >= 0; i -= 1){
    const rec = records[i];
    const key = rec.trigger_seq;
    if (!seen.has(key)){
      cachedRecords.unshift(rec);
      seen.add(key);
    }
  }
  if (maxRecords > 0 && cachedRecords.length > maxRecords){
    cachedRecords = cachedRecords.slice(0, maxRecords);
  }
}

function updateLatestSeq(latestSeq){
  if (typeof latestSeq === 'number' && Number.isFinite(latestSeq)){
    lastSeenSeq = latestSeq;
    return;
  }
  if (cachedRecords.length){
    lastSeenSeq = Number(cachedRecords[0].trigger_seq || 0);
  }
}

function renderLatestRecordHeader(showRecs){
  if (!showRecs.length){
    return;
  }
  const r0 = showRecs[0];
  const id0 = r0.trigger_seq || 0;
  const triggeredAtMs = Number(r0.triggered_at_ms);
  const parts0 = toIsoParts(triggeredAtMs);
  const cls0 = resultClass(r0.result);
  els.statusTime.className = `status-time ${cls0}`;
  els.statusTime.innerText = `${String(id0).padStart(ID_PAD_LEN,'_')} ${parts0.date} ${parts0.time}`;
  els.titleHeader.className = `header ${cls0}`;
  if (id0 !== lastImgId){
    lastImgId = id0;
    els.img.src='/preview/latest?ts='+Date.now();
  }
}

function applyStatusPayload(payload){
  els.statusTime.innerText = 'Idle';
  els.statusTime.className = 'status-time';

  const records = Array.isArray(payload.records) ? payload.records : [];
  const maxRecords = Number(payload.max_records || 0);
  const fullSnapshot = Boolean(payload.full_snapshot);
  mergeStatusRecords(records, maxRecords, fullSnapshot);
  updateLatestSeq(payload.latest_seq);

  renderStats(payload.stats);
  const showRecs = renderRecords(cachedRecords, maxRecords);
  renderLatestRecordHeader(showRecs);
  updateHeartbeatOnline(payload.heartbeat_seq);
  renderRecipes(payload.recipe);
  renderBatch(payload.batch);
}

async function refresh(){
  if (refreshInFlight) return;
  refreshInFlight = true;
  try{
    const statusUrl = (lastSeenSeq === null)
      ? '/status'
      : `/status?since_seq=${encodeURIComponent(String(lastSeenSeq))}`;
    const r = await fetchWithTimeout(statusUrl, {}, STATUS_TIMEOUT_MS);
    if (!r.ok){
      throw new Error(`status_http_${r.status}`);
    }
    applyStatusPayload(await r.json());
  }catch(e){
    console.error(e);
    els.statusTime.innerText = 'Status unreachable';
    els.statusTime.className = 'status-time';
    staleHeartbeatPolls += 1;
    renderOnline(staleOnlineState());
    renderRecipes(null);
    renderBatch(null);
  } finally {
    refreshInFlight = false;
  }
}
els.triggerBtn.addEventListener('click', function(){
  void doTrigger();
});
els.recipeToggle.addEventListener('click', function(ev){
  ev.stopPropagation();
  toggleRecipeMenu();
});
els.batchInput.addEventListener('input', function(){
  showBatchMessage('', false);
  updateBatchSetButton();
});
els.batchInput.addEventListener('keydown', function(ev){
  if (ev.key === 'Enter'){
    ev.preventDefault();
    void setBatch();
  }
});
els.batchSetBtn.addEventListener('click', function(){
  void setBatch();
});
document.addEventListener('click', function(ev){
  if (!els.recipeWrap.contains(ev.target)){
    closeRecipeMenu();
  }
});
document.addEventListener('keydown', function(ev){
  if (ev.key === 'Escape'){
    closeRecipeMenu();
  }
});
window.addEventListener('blur', closeRecipeMenu);
setInterval(tickClock, REFRESH_MS);
setInterval(refresh, REFRESH_MS);
window.addEventListener('focus', function(){ void refresh(); });
tickClock();
void refresh();
