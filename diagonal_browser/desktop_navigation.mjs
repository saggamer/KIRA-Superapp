// Browser chrome pages and Chromium pages share a per-tab navigation timeline.
export class TabNavigation {
  constructor() { this.entries = [{ kind: 'home', title: 'New tab' }]; this.index = 0; }
  get current() { return this.entries[this.index]; }
  get canGoBack() { return this.index > 0; }
  get canGoForward() { return this.index < this.entries.length - 1; }
  push(entry) {
    this.entries.splice(this.index + 1);
    this.entries.push(entry);
    if (this.entries.length > 100) this.entries.shift();
    this.index = this.entries.length - 1;
  }
  surface(entry) {
    const previous = this.current;
    const sameSearch = entry.kind === 'search' && previous.kind === 'search' &&
      entry.searchResults.query === previous.searchResults.query && entry.searchResults.mode === previous.searchResults.mode &&
      JSON.stringify(entry.searchResults.filters || {}) === JSON.stringify(previous.searchResults.filters || {});
    const copy = structuredClone(entry);
    if ((entry.kind === 'home' && previous.kind === 'home') || sameSearch) this.entries[this.index] = copy;
    else this.push(copy);
  }
  page(entry, { replace = false } = {}) {
    const previous = this.current;
    if (replace || (previous.kind === 'page' && previous.viewId === entry.viewId && previous.nativeIndex === entry.nativeIndex)) {
      this.entries[this.index] = { kind: 'page', ...entry }; return;
    }
    // Also track history.back()/forward() invoked by a website itself.
    if (previous.kind === 'page' && previous.viewId === entry.viewId) {
      const direction = Math.sign(entry.nativeIndex - previous.nativeIndex);
      for (let index = this.index + direction; direction && index >= 0 && index < this.entries.length; index += direction) {
        const item = this.entries[index];
        if (item.kind === 'page' && item.viewId === entry.viewId && item.nativeIndex === entry.nativeIndex && item.url === entry.url) { this.index = index; this.entries[index] = { kind: 'page', ...entry }; return; }
      }
    }
    this.push({ kind: 'page', ...entry });
  }
  move(direction) {
    if (![-1, 1].includes(direction) || (direction < 0 ? !this.canGoBack : !this.canGoForward)) return null;
    this.index += direction; return this.current;
  }
}

export function liveContents(view) {
  try { const contents = view?.webContents; return contents && !contents.isDestroyed() ? contents : null; }
  catch { return null; }
}

export function countBlockedRequest(tabs, request) {
  if (!Number.isInteger(request?.tabId) || request.tabId < 0) return;
  const tab = tabs.find(item => liveContents(item.view)?.id === request.tabId);
  if (tab) tab.blocked += 1;
}
