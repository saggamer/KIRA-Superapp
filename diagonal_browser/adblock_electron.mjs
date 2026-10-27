import { ElectronBlocker } from '@ghostery/adblocker-electron';
import { webContents } from 'electron';

export async function protectElectronSession(session, blocker) {
  await blocker.init();
  const adapter = ElectronBlocker.empty();
  const topUrl = request => webContents.fromId(request.tabId)?.getURL() || request.sourceUrl;
  adapter.match = request => blocker.match(request, { topUrl: topUrl(request) });
  adapter.getCSPDirectives = request => blocker.enabledFor(topUrl(request)) ? blocker.engine.getCSPDirectives(request) : undefined;
  adapter.getCosmeticsFilters = details => blocker.cosmetics(details);
  const inject = adapter.onInjectCosmeticFilters;
  adapter.onInjectCosmeticFilters = (event, _url, features) => {
    const url = event.senderFrame?.url;
    if (!url || !/^https?:\/\//i.test(url) || !blocker.enabledFor(event.sender.getURL())) return;
    return inject(event, url, features);
  };
  adapter.enableBlockingInSession(session);
  return adapter;
}
