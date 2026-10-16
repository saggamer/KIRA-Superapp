const blockedHosts = [
  '2mdn.net', 'adnxs.com', 'adsrvr.org', 'amazon-adsystem.com',
  'casalemedia.com', 'criteo.com', 'criteo.net', 'doubleclick.net',
  'googlesyndication.com', 'googleadservices.com', 'google-analytics.com',
  'hotjar.com', 'moatads.com', 'outbrain.com', 'quantserve.com',
  'scorecardresearch.com', 'taboola.com', 'zedo.com',
];

export function blockedHost(url, resourceType = '') {
  if (resourceType === 'mainFrame') return null;
  let hostname;
  try { hostname = new URL(url).hostname.toLowerCase(); }
  catch { return null; }
  return blockedHosts.find(host => hostname === host || hostname.endsWith(`.${host}`)) || null;
}
