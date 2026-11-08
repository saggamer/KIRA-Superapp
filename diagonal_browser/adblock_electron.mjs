import { ElectronBlocker } from '@ghostery/adblocker-electron';
import { webContents } from 'electron';
import { liveContents } from './desktop_navigation.mjs';

export async function protectElectronSession(session, blocker) {
  await blocker.init();
  const adapter = ElectronBlocker.empty();
  const topUrl = request => {
    if (!Number.isInteger(request.tabId) || request.tabId < 0) return request.sourceUrl;
    return liveContents({ webContents: webContents.fromId(request.tabId) })?.getURL() || request.sourceUrl;
  };
  adapter.match = request => blocker.match(request, { topUrl: topUrl(request) });
  adapter.getCSPDirectives = request => blocker.enabledFor(topUrl(request)) ? blocker.engine.getCSPDirectives(request) : undefined;
  adapter.getCosmeticsFilters = details => blocker.cosmetics(details);
  const inject = adapter.onInjectCosmeticFilters;
  adapter.onInjectCosmeticFilters = (event, _url, features) => {
    const sender = liveContents({ webContents: event.sender });
    if (!sender) return;
    let url;
    try { url = event.senderFrame?.url; } catch { return; }
    if (!url || !/^https?:\/\//i.test(url) || !blocker.enabledFor(sender.getURL())) return;
    return inject(event, url, features);
  };
  adapter.enableBlockingInSession(session);
  return adapter;
}
