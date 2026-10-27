const $ = selector => document.querySelector(selector);
const tabsNode = $('#tabs');
const address = $('#address');
const hero = $('.hero');
const resultsNode = $('#results');
const toast = $('#toast');
let state = { tabs: [], activeId: null, recording: null, panel: null, inspection: null, adBlockEnabled: true };
let toastTimer;
let mediaItems = [];
let lastPanel = null;
let searchMode = 'web';
let imageResults = [];
let imageIndex = 0;
let searchRequestId = 0;
let historyRequestId = 0;
let historyFilters = { query: '', kind: 'all', source: 'all' };
let recentKey = '';
let lastRenderedActive = null;

async function renderRecentSearches(tab) {
  const key = `${tab?.id}:${Math.floor(Date.now() / 10000)}`;
  if (recentKey === key) return;
  recentKey = key;
  const history = await run('searchHistory', { kind: 'search', limit: 3 });
  if (!history || recentKey !== key || state.panel) return;
  const list = $('#recent-search-list'); list.replaceChildren();
  for (const entry of history.entries) {
    const button = document.createElement('button'); button.className = 'recent-search';
    button.title = `Restore saved search: ${entry.query}`;
    const query = document.createElement('strong'); query.textContent = entry.query;
    const meta = document.createElement('small'); meta.textContent = `${entry.source === 'llm' ? 'Kira' : 'Browser'} · ${entry.mode === 'images' ? 'Images' : 'Web'} · ${new Date(entry.createdAt).toLocaleDateString()}`;
    button.append(query, meta); button.onclick = () => run('revisitHistory', { historyId: entry.historyId });
    list.append(button);
  }
  if (!history.entries.length) { const empty = document.createElement('p'); empty.className = 'recent-empty'; empty.textContent = 'Your recent searches will appear here.'; list.append(empty); }
}

function notify(message, artifact) {
  clearTimeout(toastTimer);
  toast.replaceChildren();
  const label = document.createElement('span');
  label.textContent = message;
  toast.append(label);
  if (artifact) {
    const button = document.createElement('button');
    button.textContent = 'Show file ↗';
    button.style.cssText = 'margin-left:14px;background:none;border:0;color:#ffd099;font-weight:700';
    button.onclick = () => run('showArtifact', artifact);
    toast.append(button);
  }
  toast.hidden = false;
  toastTimer = setTimeout(() => { toast.hidden = true; }, 7000);
}
async function run(action, value) {
  try { return await window.diagonal.action(action, value); }
  catch (error) { notify(error.message || String(error)); return undefined; }
}
async function requestSearch(action, value) {
  const requestId = ++searchRequestId;
  $('#search-status-text').textContent = value?.mode === 'images' ? 'Searching images…' : 'Opening or searching…';
  $('#search-status').hidden = false;
  try { return await run(action, value); }
  finally { if (requestId === searchRequestId) $('#search-status').hidden = true; }
}
function renderTabs() {
  tabsNode.replaceChildren();
  for (const tab of state.tabs) {
    const item = document.createElement('div');
    item.className = `tab${tab.id === state.activeId ? ' active' : ''}`;
    item.setAttribute('role', 'tab');
    item.setAttribute('aria-selected', String(tab.id === state.activeId));
    item.tabIndex = tab.id === state.activeId ? 0 : -1;
    item.onclick = () => run('selectTab', tab.id);
    item.onkeydown = event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); run('selectTab', tab.id); }
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); const index = state.tabs.findIndex(item => item.id === tab.id); const next = (index + (event.key === 'ArrowRight' ? 1 : -1) + state.tabs.length) % state.tabs.length; run('selectTab', state.tabs[next].id); }
    };
    const dot = document.createElement('span');
    dot.className = 'tab-dot';
    dot.textContent = tab.loading ? '◌' : '◈';
    const title = document.createElement('span');
    title.className = 'tab-title';
    title.textContent = tab.title || 'New tab';
    title.title = tab.title || 'New tab';
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'tab-close';
    close.textContent = '×';
    close.title = 'Close tab';
    close.setAttribute('aria-label', `Close ${tab.title || 'New tab'}`);
    close.onclick = event => { event.stopPropagation(); run('closeTab', tab.id); };
    item.append(dot, title, close);
    tabsNode.append(item);
  }
  if (lastRenderedActive !== state.activeId) { tabsNode.querySelector('.tab.active')?.scrollIntoView({ block: 'nearest', inline: 'nearest' }); lastRenderedActive = state.activeId; }
}
function modeSwitch(onSelect) {
  const switcher = document.createElement('div');
  switcher.className = 'search-switch';
  switcher.setAttribute('role', 'tablist');
  switcher.setAttribute('aria-label', 'Search type');
  for (const [mode, label] of [['web', '⌕ Web'], ['images', '▧ Images']]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = mode === searchMode ? 'selected' : '';
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-selected', String(mode === searchMode));
    button.textContent = label;
    button.onclick = () => onSelect(mode);
    switcher.append(button);
  }
  return switcher;
}
function setSearchMode(mode, query) {
  searchMode = mode === 'images' ? 'images' : 'web';
  document.querySelectorAll('.hero .search-switch button').forEach(button => {
    const active = button.dataset.mode === searchMode;
    button.classList.toggle('selected', active);
    button.setAttribute('aria-selected', String(active));
  });
  $('#images').classList.toggle('active', searchMode === 'images');
  $('#hero-input').placeholder = searchMode === 'images' ? 'Search images across the web' : 'Search anything or paste a link';
  address.placeholder = searchMode === 'images' ? 'Search images or enter a URL' : 'Search with DDG or enter a URL';
  if (query) requestSearch('search', { query, mode: searchMode });
}
function renderImageFilters(query, filters) {
  const row = document.createElement('div');
  row.className = 'image-filters';
  const choices = [
    ['All images', {}], ['Large', { size: 'Large' }], ['Wallpaper', { size: 'Wallpaper' }],
    ['Transparent', { type_image: 'transparent' }], ['GIF', { type_image: 'gif' }],
    ['Black & white', { color: 'Monochrome' }],
  ];
  for (const [label, value] of choices) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    const active = JSON.stringify(filters || {}) === JSON.stringify(value);
    button.className = active ? 'selected' : '';
    button.setAttribute('aria-pressed', String(active));
    button.onclick = () => requestSearch('search', { query, mode: 'images', filters: value });
    row.append(button);
  }
  return row;
}
function renderImageGrid(results) {
  imageResults = results.filter(item => item.thumbnail && item.image && item.url && /^https?:\/\//i.test(item.thumbnail) && /^https?:\/\//i.test(item.image) && /^https?:\/\//i.test(item.url));
  if (!imageResults.length) {
    const empty = document.createElement('div');
    empty.className = 'search-empty';
    empty.textContent = 'No images matched this search. Try a broader phrase or choose All images.';
    resultsNode.append(empty);
    return;
  }
  const grid = document.createElement('div');
  grid.className = 'image-grid';
  imageResults.forEach((result, index) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'image-result';
    card.title = result.title || result.source;
    card.onclick = () => openImageViewer(index);
    const visual = document.createElement('span'); visual.className = 'image-result-visual';
    const image = document.createElement('img');
    image.src = result.thumbnail;
    image.alt = result.title || 'Image result';
    image.loading = 'lazy';
    image.referrerPolicy = 'no-referrer';
    const ratio = Number(result.width) / Number(result.height);
    if (Number.isFinite(ratio) && ratio > 0) image.style.setProperty('--image-ratio', String(Math.max(.85, Math.min(1.65, ratio))));
    image.onerror = () => { image.style.visibility = 'hidden'; };
    visual.append(image);
    const title = document.createElement('span'); title.className = 'image-result-title'; title.textContent = result.title || 'Untitled image';
    const source = document.createElement('span'); source.className = 'image-result-source'; source.textContent = result.source || new URL(result.url).hostname;
    card.append(visual, title, source); grid.append(card);
  });
  resultsNode.append(grid);
}
function renderResults(tab) {
  const data = tab?.searchResults;
  hero.hidden = !!data;
  resultsNode.hidden = !data;
  if (!data) return;
  searchMode = data.mode === 'images' ? 'images' : 'web';
  setSearchMode(searchMode);
  resultsNode.replaceChildren();
  const header = document.createElement('div');
  header.className = 'results-heading';
  const headingText = document.createElement('div');
  const heading = document.createElement('h2');
  heading.textContent = `${searchMode === 'images' ? 'Images' : 'Web results'} for “${data.query}”`;
  const sub = document.createElement('p');
  sub.className = 'results-sub';
  sub.textContent = searchMode === 'images'
    ? `${data.results.length} image results · Select an image to preview it and visit its source`
    : `${data.results.length} results from DuckDuckGo · Select a result to open it here`;
  if (data.savedAt) sub.textContent = `Saved ${new Date(data.savedAt).toLocaleString()} · ${data.results.length} historical results · Open a source for a fresh view`;
  headingText.append(heading, sub);
  header.append(headingText, modeSwitch(mode => setSearchMode(mode, data.query)));
  resultsNode.append(header);
  if (searchMode === 'images') {
    resultsNode.append(renderImageFilters(data.query, data.filters));
    renderImageGrid(data.results);
    return;
  }
  for (const result of data.results) {
    if (!result.url || !/^https?:\/\//i.test(result.url)) continue;
    const card = document.createElement('article');
    card.className = 'result-card';
    const title = document.createElement('button');
    title.textContent = result.title || result.url;
    title.onclick = () => run('navigate', result.url);
    const url = document.createElement('div');
    url.className = 'url';
    url.textContent = result.url;
    const snippet = document.createElement('p');
    snippet.className = 'snippet';
    snippet.textContent = result.snippet || '';
    card.append(title, url, snippet);
    resultsNode.append(card);
  }
}
function closeImageViewer() {
  $('#image-viewer').hidden = true;
  $('#image-viewer-image').removeAttribute('src');
}
function showImageViewer() {
  const item = imageResults[imageIndex];
  if (!item) return closeImageViewer();
  $('#image-viewer').hidden = false;
  $('#image-viewer-count').textContent = `${imageIndex + 1} of ${imageResults.length}`;
  $('#image-viewer-title').textContent = item.title || 'Untitled image';
  $('#image-viewer-source').textContent = `${item.source || new URL(item.url).hostname}${item.width && item.height ? ` · ${item.width} × ${item.height}` : ''}`;
  const image = $('#image-viewer-image');
  image.hidden = false;
  image.alt = item.title || 'Image preview';
  image.referrerPolicy = 'no-referrer';
  image.dataset.fallbackTried = '';
  $('#image-viewer-fallback').hidden = true;
  image.onerror = () => {
    if (!image.dataset.fallbackTried) { image.dataset.fallbackTried = '1'; image.src = item.thumbnail; }
    else { image.hidden = true; $('#image-viewer-fallback').hidden = false; }
  };
  image.src = item.image;
  $('#image-prev').disabled = imageIndex === 0;
  $('#image-next').disabled = imageIndex === imageResults.length - 1;
}
function openImageViewer(index) { imageIndex = index; showImageViewer(); }
function stepImage(direction) {
  imageIndex = Math.max(0, Math.min(imageResults.length - 1, imageIndex + direction));
  showImageViewer();
}
function render(next) {
  state = next;
  renderTabs();
  const tab = state.tabs.find(item => item.id === state.activeId);
  if (!tab?.searchResults || tab.searchResults.mode !== 'images') closeImageViewer();
  if (document.activeElement !== address) address.value = tab?.url || tab?.searchResults?.query || '';
  if (tab?.searchResults) searchMode = tab.searchResults.mode === 'images' ? 'images' : 'web';
  setSearchMode(searchMode);
  renderResults(tab);
  const onHome = !tab?.url && !tab?.searchResults && !state.panel;
  $('.home-content').hidden = !onHome;
  if (onHome) void renderRecentSearches(tab); else recentKey = '';
  $('#home-protection').textContent = state.adBlockEnabled ? 'Protected browsing' : 'Ad protection paused';
  $('#back').disabled = !tab?.canGoBack;
  $('#forward').disabled = !tab?.canGoForward;
  $('#reload').disabled = !tab?.url && !tab?.searchResults;
  $('#load-progress').hidden = !tab?.loading;
  let context = 'Ready to explore';
  if (tab?.searchResults) context = tab.searchResults.savedAt ? 'Saved search · History' : `${tab.searchResults.mode === 'images' ? 'Image' : 'Web'} results · DuckDuckGo`;
  else if (tab?.url) {
    try { const page = new URL(tab.url); context = `${page.protocol.replace(':', '').toUpperCase()} · ${page.hostname}`; }
    catch { context = tab.url; }
  }
  $('#page-context-label').textContent = context;
  $('#page-context').title = context;
  $('#record').classList.toggle('recording', !!state.recording);
  $('#record span').textContent = state.recording ? 'Stop' : 'Record';
  $('#adblock').classList.toggle('off', !state.adBlockEnabled || state.adBlockSiteEnabled === false);
  $('#adblock span').textContent = !state.adBlockEnabled ? 'Shield off' : state.adBlockSiteEnabled === false ? 'Shield paused here' : `Shield on · ${state.blockedCount || 0}`;
  for (const [id, name] of [['adblock', 'shield'], ['accounts', 'accounts'], ['history', 'history'], ['inspect', 'inspect'], ['media', 'media']]) {
    $(`#${id}`).classList.toggle('active', state.panel === name);
    $(`#${id}`).setAttribute('aria-pressed', String(state.panel === name));
  }
  if (state.panel !== lastPanel) { lastPanel = state.panel; renderPanel(); }
}

function panelButton(label, handler) {
  const button = document.createElement('button');
  button.textContent = label;
  button.onclick = handler;
  return button;
}
function section(title, content) {
  const wrapper = document.createElement('section');
  wrapper.className = 'inspect-section';
  const heading = document.createElement('h3');
  heading.textContent = title;
  wrapper.append(heading, content);
  return wrapper;
}
function list(items, formatter) {
  const ul = document.createElement('ul');
  ul.className = 'inspect-list';
  for (const item of items) {
    const li = document.createElement('li');
    li.textContent = formatter(item);
    ul.append(li);
  }
  return ul;
}
function renderInspector() {
  const data = state.inspection;
  const body = $('#panel-body');
  if (!data) { body.textContent = 'Open a website, then inspect its URL and page.'; return; }
  const { urlInfo, page } = data;
  const grid = document.createElement('div');
  grid.className = 'inspect-grid';
  const entries = [
    ['ORIGIN', urlInfo.origin], ['HOST', urlInfo.hostname], ['PROTOCOL', urlInfo.protocol.toUpperCase()],
    ['PATH', urlInfo.pathname], ['QUERY KEYS', urlInfo.queryKeys.join(', ') || 'None'],
    ['BLOCKED REQUESTS', String(data.blockedRequests)], ['PAGE WORDS', String(page.wordCount)],
    ['CANONICAL', page.canonical || 'Not declared'],
    ['LANGUAGE', page.language || 'Not declared'], ['ROBOTS', page.robots || 'Not declared'],
  ];
  for (const [label, value] of entries) {
    const card = document.createElement('div'); card.className = 'inspect-stat';
    const key = document.createElement('label'); key.textContent = label;
    const detail = document.createElement('strong'); detail.textContent = value;
    card.append(key, detail); grid.append(card);
  }
  body.append(grid);
  if (urlInfo.flags.length) {
    const flags = document.createElement('p'); flags.className = 'inspect-warning';
    flags.textContent = `Technical URL flags: ${urlInfo.flags.join(' · ')}`;
    body.append(flags);
  }
  const description = document.createElement('p'); description.textContent = page.description || 'No meta description found.';
  body.append(section('Page description', description));
  body.append(section('Headings', list(page.headings, item => `H${item.level} · ${item.text}`)));
  body.append(section('Links', list(page.links.slice(0, 40), item => `${item.text || 'Untitled'} — ${item.url}`)));
  body.append(section('Images', list(page.images.slice(0, 30), item => `${item.alt || 'No alt text'} — ${item.url}`)));
  body.append(section('Forms', list(page.forms, item => `${item.method.toUpperCase()} ${item.action} · ${item.fields.length} fields`)));
  const readable = document.createElement('pre'); readable.className = 'inspect-pre'; readable.textContent = page.text || 'No readable page text found.';
  body.append(section('Readable text for LLMs', readable));
}
function openMedia(item) {
  const body = $('#panel-body');
  body.replaceChildren();
  const viewer = document.createElement('div'); viewer.className = 'viewer';
  const header = document.createElement('div'); header.className = 'viewer-title';
  header.append(panelButton('← All media', renderMedia), document.createTextNode(item.name));
  viewer.append(header);
  const display = document.createElement(item.kind === 'video' ? 'video' : 'img');
  display.src = item.url;
  if (item.kind === 'video') { display.controls = true; display.autoplay = true; display.playsInline = true; }
  else display.alt = item.name;
  viewer.append(display);
  body.append(viewer);
}
function renderMedia() {
  const body = $('#panel-body');
  body.replaceChildren();
  if (!mediaItems.length) { body.textContent = 'No captures yet. Use Capture or Record to create one.'; return; }
  const grid = document.createElement('div'); grid.className = 'media-grid';
  for (const item of mediaItems) {
    const card = document.createElement('button'); card.className = 'media-card';
    card.onclick = () => openMedia(item);
    if (item.kind === 'image') {
      const image = document.createElement('img'); image.className = 'media-preview'; image.src = item.url; image.alt = item.name; image.loading = 'lazy';
      card.append(image);
    } else {
      const placeholder = document.createElement('div'); placeholder.className = 'video-placeholder'; placeholder.textContent = '▶'; card.append(placeholder);
    }
    const name = document.createElement('span'); name.className = 'media-label'; name.textContent = item.name;
    const meta = document.createElement('span'); meta.className = 'media-meta';
    meta.textContent = `${item.kind === 'video' ? 'WEBM VIDEO' : 'PNG IMAGE'} · ${(item.size / 1024).toFixed(0)} KB · ${new Date(item.updated).toLocaleString()}`;
    card.append(name, meta); grid.append(card);
  }
  body.append(grid);
}
function renderAccounts() {
  const body = $('#panel-body');
  body.replaceChildren();
  const note = document.createElement('p');
  note.className = 'account-note';
  note.textContent = 'Save up to three email nicknames. Choose Use on a sign-in page to fill its email field. The site still handles your password, passkey, or other sign-in step.';
  const form = document.createElement('form');
  form.className = 'account-form';
  const rows = [];
  for (let index = 0; index < 3; index++) {
    const profile = state.emailProfiles?.[index];
    const row = document.createElement('div'); row.className = 'account-row';
    const nicknameLabel = document.createElement('label'); nicknameLabel.textContent = `Nickname ${index + 1}`;
    const nickname = document.createElement('input'); nickname.value = profile?.nickname || ''; nickname.maxLength = 32; nickname.placeholder = 'Personal';
    nicknameLabel.append(nickname);
    const emailLabel = document.createElement('label'); emailLabel.textContent = 'Email';
    const email = document.createElement('input'); email.type = 'email'; email.value = profile?.email || ''; email.placeholder = 'name@example.com';
    emailLabel.append(email);
    const use = document.createElement('button'); use.type = 'button'; use.textContent = 'Use on page'; use.disabled = !profile || !state.tabs.find(item => item.id === state.activeId)?.url;
    use.onclick = async () => { const result = await run('useEmailProfile', profile.nickname); if (result) notify(`${result.nickname} filled the email field`); };
    row.append(nicknameLabel, emailLabel, use); form.append(row); rows.push({ nickname, email });
  }
  const save = document.createElement('button'); save.type = 'submit'; save.className = 'account-save'; save.textContent = 'Save email nicknames';
  form.append(save);
  form.onsubmit = async event => {
    event.preventDefault();
    const profiles = rows.map(({ nickname, email }) => ({ nickname: nickname.value.trim(), email: email.value.trim() })).filter(item => item.nickname || item.email);
    const saved = await run('saveEmailProfiles', profiles);
    if (saved) { state.emailProfiles = saved; renderAccounts(); notify('Email nicknames saved'); }
  };
  body.append(note, form);
}
function historyText(tag, className, value) {
  const node = document.createElement(tag);
  node.className = className; node.textContent = value;
  return node;
}

async function renderHistory() {
  const request = ++historyRequestId;
  const body = $('#panel-body'); body.replaceChildren();
  const form = document.createElement('form'); form.className = 'history-filters';
  const input = document.createElement('input'); input.type = 'search'; input.placeholder = 'Search queries, titles, URLs, and saved page text'; input.value = historyFilters.query;
  input.setAttribute('aria-label', 'Search saved history');
  const select = (label, key, values) => {
    const node = document.createElement('select'); node.setAttribute('aria-label', label);
    for (const [value, text] of values) { const option = document.createElement('option'); option.value = value; option.textContent = text; node.append(option); }
    node.value = historyFilters[key]; return node;
  };
  const kind = select('History type', 'kind', [['all', 'Searches + pages'], ['search', 'Searches'], ['page', 'Saved pages']]);
  const source = select('History source', 'source', [['all', 'Kira + browser'], ['llm', 'Kira only'], ['desktop', 'Browser only']]);
  const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = 'Search history';
  form.append(input, kind, source, submit);
  const status = historyText('p', 'history-meta', 'Loading saved history…'); status.setAttribute('role', 'status');
  const list = document.createElement('div'); list.className = 'history-list';
  const footer = document.createElement('div'); footer.className = 'history-pagination';
  body.append(form, status, list, footer);
  form.onsubmit = event => { event.preventDefault(); historyFilters = { query: input.value, kind: kind.value, source: source.value }; renderHistory(); };
  async function load(offset = 0) {
    const result = await run('searchHistory', { ...historyFilters, offset, limit: 20 });
    if (!result || request !== historyRequestId || state.panel !== 'history') return;
    status.textContent = result.total ? `${result.total} saved ${result.total === 1 ? 'entry' : 'entries'} · Stored locally and available to Kira` : 'No saved history matches. New searches and read pages will appear here.';
    for (const item of result.entries) {
      const card = document.createElement('article'); card.className = 'history-card';
      card.append(historyText('p', 'history-meta', `${item.source === 'llm' ? 'KIRA' : 'BROWSER'} · ${item.kind === 'search' ? `${item.mode === 'images' ? 'Image search' : 'Web search'} · ${item.resultCount} ${item.resultCount === 1 ? 'result' : 'results'}` : 'Read page'} · ${new Date(item.createdAt).toLocaleString()}`));
      card.append(historyText('h3', '', item.query || item.title || item.url));
      if (item.url) card.append(historyText('p', 'history-url', item.url));
      card.append(historyText('p', 'history-snippet', item.snippet));
      const actions = document.createElement('div'); actions.className = 'history-actions';
      actions.append(panelButton('Read saved', () => renderHistoryDetail(item.historyId)), panelButton(item.kind === 'search' ? 'Restore results' : 'Open website ↗', () => run('revisitHistory', { historyId: item.historyId })));
      card.append(actions); list.append(card);
    }
    footer.replaceChildren();
    if (result.nextOffset !== null) {
      const more = panelButton('Load more', () => { more.disabled = true; load(result.nextOffset); });
      footer.append(more);
    }
  }
  await load();
}

async function renderHistoryDetail(historyId) {
  const request = ++historyRequestId;
  const entry = await run('readHistory', { historyId });
  if (!entry || request !== historyRequestId || state.panel !== 'history') return;
  const body = $('#panel-body'); body.replaceChildren();
  const actions = document.createElement('div'); actions.className = 'history-actions';
  actions.append(panelButton('← History', renderHistory), panelButton(entry.kind === 'search' ? 'Restore results' : 'Open website ↗', () => run('revisitHistory', { historyId })));
  body.append(actions, historyText('h3', 'history-detail-title', entry.query || entry.title || entry.url),
    historyText('p', 'history-meta', `Saved ${new Date(entry.createdAt).toLocaleString()} · ${entry.source === 'llm' ? 'Kira' : 'Browser'} · ${entry.kind === 'page' ? `${entry.textLength} cached characters${entry.textTruncated ? ' (capture limit reached)' : ''}` : `${entry.resultCount} results`}`),
    historyText('p', 'history-notice', 'This is a saved copy. Website content may have changed since it was captured.'));
  const content = document.createElement('div'); content.className = 'history-list';
  const footer = document.createElement('div'); footer.className = 'history-pagination';
  body.append(content, footer);
  function appendChunk(chunk) {
    if (chunk.kind === 'page') content.append(historyText('pre', 'history-page-text', chunk.text || '(No readable page text was captured.)'));
    else for (const result of chunk.results) {
      const card = document.createElement('article'); card.className = 'history-card';
      card.append(historyText('h3', '', result.title || result.url), historyText('p', 'history-url', result.url), historyText('p', 'history-snippet', result.snippet));
      if (result.url) card.append(panelButton('Open source ↗', () => run('revisitHistory', { historyId, resultIndex: result.resultIndex })));
      content.append(card);
    }
    footer.replaceChildren();
    const next = chunk.kind === 'page' ? chunk.nextTextOffset : chunk.nextResultOffset;
    if (next !== null) {
      const more = panelButton('Read more', async () => {
        more.disabled = true;
        const nextChunk = await run('readHistory', { historyId, [chunk.kind === 'page' ? 'textOffset' : 'resultOffset']: next });
        if (nextChunk && request === historyRequestId && state.panel === 'history') appendChunk(nextChunk);
      }); footer.append(more);
    }
  }
  appendChunk(entry);
}

async function renderShield() {
  const shield = await run('adblockStatus');
  if (!shield || state.panel !== 'shield') return;
  const body = $('#panel-body'); body.replaceChildren();
  const wrapper = document.createElement('div'); wrapper.className = 'shield-settings'; body.append(wrapper);
  const summary = document.createElement('section'); summary.className = 'shield-card';
  summary.append(historyText('h3', '', shield.enabled ? 'Protection is on' : 'Protection is paused globally'),
    historyText('p', 'history-meta', `${shield.networkRules.toLocaleString()} network rules · ${shield.cosmeticRules.toLocaleString()} cosmetic rules · Updated ${new Date(shield.updatedAt).toLocaleString()}`),
    historyText('p', 'history-snippet', `${shield.blocked} ${shield.blocked === 1 ? 'request' : 'requests'} blocked or replaced in this browser session. Lists work offline and settings are shared with Kira.`));
  const controls = document.createElement('div'); controls.className = 'history-actions';
  const change = async options => { const value = await run('adblockSet', options); if (value) { await renderShield(); notify('Settings saved. Reload open pages to apply the change.'); } };
  controls.append(panelButton(shield.enabled ? 'Pause everywhere' : 'Enable ad blocker', () => change({ enabled: !shield.enabled })));
  if (shield.site) controls.append(panelButton(`${shield.siteEnabled ? 'Pause' : 'Resume'} on ${shield.site}`, () => change({ site: shield.site, siteEnabled: !shield.siteEnabled })));
  summary.append(controls);
  for (const [key, text] of [['cosmetic', 'Hide ad slots and apply site fixes'], ['autoUpdate', 'Update filter lists automatically']]) {
    const label = document.createElement('label'); label.className = 'shield-option';
    const input = document.createElement('input'); input.type = 'checkbox'; input.checked = shield[key]; input.onchange = () => change({ [key]: input.checked });
    label.append(input, document.createTextNode(text)); summary.append(label);
  }
  summary.append(historyText('p', 'history-meta', 'Reload a page after changing protection. No blocker can remove every ad, including some ads inserted directly into video streams.'));
  if (shield.updateError) summary.append(historyText('p', 'history-notice', `The last update failed. Your existing lists remain active. ${shield.updateError}`));
  wrapper.append(summary);

  const rules = document.createElement('section'); rules.className = 'shield-card'; rules.append(historyText('h3', '', 'Custom filters'));
  const form = document.createElement('form'); form.className = 'history-filters';
  const field = document.createElement('input'); field.placeholder = 'example.com##.advertisement or ||ads.example.com^'; field.setAttribute('aria-label', 'Custom adblock rule'); field.maxLength = 4000;
  const add = document.createElement('button'); add.type = 'submit'; add.textContent = 'Add rule'; form.append(field, add);
  form.onsubmit = async event => { event.preventDefault(); const result = await run('adblockRule', { operation: 'add', rule: field.value.trim() }); if (result) { renderShield(); notify('Filter saved. Reload the affected page.'); } };
  rules.append(form);
  for (const rule of shield.customRules) {
    const row = document.createElement('div'); row.className = 'shield-rule';
    const remove = panelButton('Remove', async () => { if (await run('adblockRule', { operation: 'remove', rule })) { renderShield(); notify('Filter removed. Reload the affected page.'); } });
    remove.setAttribute('aria-label', `Remove filter ${rule}`);
    row.append(historyText('code', '', rule), remove); rules.append(row);
  }
  wrapper.append(rules);

  const check = document.createElement('section'); check.className = 'shield-card'; check.append(historyText('h3', '', 'Check a request'));
  const checkForm = document.createElement('form'); checkForm.className = 'history-filters';
  const resource = document.createElement('input'); resource.type = 'url'; resource.placeholder = 'Resource URL to check'; resource.required = true; resource.setAttribute('aria-label', 'Resource URL to check');
  const source = document.createElement('input'); source.type = 'url'; source.value = state.tabs.find(tab => tab.id === state.activeId)?.url || ''; source.placeholder = 'Source page URL'; source.required = true; source.setAttribute('aria-label', 'Source page URL');
  const type = document.createElement('select'); type.setAttribute('aria-label', 'Request resource type');
  for (const name of ['script', 'image', 'stylesheet', 'xmlhttprequest', 'font', 'media', 'sub_frame', 'main_frame', 'other']) { const option = document.createElement('option'); option.value = name; option.textContent = name; type.append(option); }
  const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = 'Explain';
  const explanation = historyText('pre', 'shield-explanation', 'Check a resource URL against the active rules without requesting it.');
  checkForm.append(resource, source, type, submit);
  checkForm.onsubmit = async event => { event.preventDefault(); const result = await run('adblockCheck', { url: resource.value, pageUrl: source.value, resourceType: type.value }); if (result) explanation.textContent = JSON.stringify(result, null, 2); };
  check.append(checkForm, explanation); wrapper.append(check);

  const log = document.createElement('section'); log.className = 'shield-card'; log.append(historyText('h3', '', shield.site ? `Blocked on ${shield.site}` : 'Recent blocked requests'));
  if (!shield.recent.length) log.append(historyText('p', 'history-meta', 'No matching requests have been blocked in this session yet.'));
  for (const item of shield.recent) log.append(historyText('p', 'shield-request', `${item.action.toUpperCase()} · ${item.type}\n${item.url}\n${item.rule}`));
  wrapper.append(log);
}

function renderPanel() {
  const panel = $('#panel');
  panel.hidden = !state.panel;
  if (!state.panel) return;
  $('#panel-title').textContent = state.panel === 'shield' ? 'Privacy shield' : state.panel === 'history' ? 'Search history' : state.panel === 'media' ? 'Capture library' : state.panel === 'accounts' ? 'Email nicknames' : 'URL & page inspector';
  $('#panel-subtitle').textContent = state.panel === 'shield' ? 'Block ad requests, hide ad slots, and control protection with Kira.' : state.panel === 'history' ? 'Find what Kira searched, read saved evidence, and return to any source.' : state.panel === 'media' ? 'Your screenshots and screen recordings, ready to preview.' : state.panel === 'accounts' ? 'Quickly fill a sign-in email field on any page.' : 'Structured page details you can copy or save for Kira and other LLMs.';
  const actions = $('#panel-actions'); actions.replaceChildren();
  $('#panel-body').replaceChildren();
  if (state.panel === 'shield') {
    actions.append(panelButton('Refresh', renderShield), panelButton('Update lists', async event => {
      const button = event.currentTarget; button.disabled = true; button.textContent = 'Updating…';
      try { if (await run('adblockUpdate')) { await renderShield(); notify('Filter lists updated. Reload open pages.'); } }
      finally { button.disabled = false; button.textContent = 'Update lists'; }
    }));
    renderShield();
  } else if (state.panel === 'history') {
    actions.append(panelButton('Refresh', renderHistory));
    renderHistory();
  } else if (state.panel === 'media') {
    actions.append(panelButton('Refresh', async () => { mediaItems = await run('listMedia') || []; renderMedia(); }));
    renderMedia();
  } else if (state.panel === 'accounts') {
    renderAccounts();
  } else {
    actions.append(panelButton('Copy LLM JSON', async () => { await run('copyInspection'); notify('Page inspection copied as JSON'); }));
    actions.append(panelButton('Save JSON', async () => { const file = await run('exportInspection'); if (file) notify('Inspection saved', file); }));
    renderInspector();
  }
}
window.diagonal.onState(render);
window.diagonal.onFocusAddress(() => { address.focus(); address.select(); });
run('state').then(value => { if (value) render(value); });

$('#new-tab').onclick = () => run('newTab');
$('#back').onclick = () => run('back');
$('#forward').onclick = () => run('forward');
$('#reload').onclick = () => state.tabs.find(tab => tab.id === state.activeId)?.searchResults ? requestSearch('reload') : run('reload');
$('#home').onclick = () => run('home');
$('#address-form').onsubmit = event => { event.preventDefault(); requestSearch('submitSearch', { query: address.value, mode: searchMode }); address.blur(); };
$('#hero-search').onsubmit = event => { event.preventDefault(); requestSearch('submitSearch', { query: $('#hero-input').value, mode: searchMode }); };
document.querySelectorAll('.hero .search-switch button').forEach(button => { button.onclick = () => setSearchMode(button.dataset.mode); });
$('#images').onclick = () => {
  const query = state.tabs.find(item => item.id === state.activeId)?.searchResults?.query;
  setSearchMode('images', query);
  if (!query) { address.focus(); address.select(); }
};
document.querySelectorAll('.quick-links button').forEach(button => { button.onclick = () => run('navigate', button.dataset.url); });
$('#screenshot').onclick = async () => { const file = await run('screenshot'); if (file) notify('Screenshot saved', file); };
$('#record').onclick = async () => {
  if (state.recording) {
    const file = await run('stopRecording');
    if (file) notify('Recording saved', file);
  } else {
    const file = await run('startRecording');
    if (file) notify('Recording started. Press Stop to save the video.');
  }
};
$('#adblock').onclick = () => run('openPanel', 'shield');
$('#accounts').onclick = () => run('openPanel', 'accounts');
$('#history').onclick = () => run('openPanel', 'history');
$('#home-history').onclick = () => run('openPanel', 'history');
$('#inspect').onclick = () => run('openPanel', 'inspect');
$('#media').onclick = async () => { mediaItems = await run('openPanel', 'media') || []; renderMedia(); };
$('#panel-close').onclick = () => run('closePanel');
$('#image-viewer-close').onclick = closeImageViewer;
$('#image-viewer-backdrop').onclick = closeImageViewer;
$('#image-prev').onclick = () => stepImage(-1);
$('#image-next').onclick = () => stepImage(1);
$('#image-copy').onclick = async () => { const item = imageResults[imageIndex]; if (item) { await run('copyText', item.image); notify('Image URL copied'); } };
$('#image-original').onclick = () => { const item = imageResults[imageIndex]; if (item) { closeImageViewer(); run('newTab', item.image); } };
$('#image-source').onclick = () => { const item = imageResults[imageIndex]; if (item) { closeImageViewer(); run('newTab', item.url); } };
document.addEventListener('keydown', event => {
  if (!$('#image-viewer').hidden) {
    if (event.key === 'Escape') { event.preventDefault(); closeImageViewer(); }
    if (event.key === 'ArrowLeft') { event.preventDefault(); stepImage(-1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); stepImage(1); }
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'l') { event.preventDefault(); address.focus(); address.select(); }
  if (event.key === 'Escape' && state.panel) { event.preventDefault(); run('closePanel'); }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'h' && event.shiftKey) { event.preventDefault(); run('openPanel', 'history'); }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 't') { event.preventDefault(); run(event.shiftKey ? 'reopenClosedTab' : 'newTab'); }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'w') { event.preventDefault(); run('closeTab', state.activeId); }
});
