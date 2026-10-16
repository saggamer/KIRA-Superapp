export function inspectUrl(input) {
  const url = new URL(input);
  const query = [...url.searchParams.keys()];
  const flags = [];
  if (url.protocol === 'http:') flags.push('Unencrypted HTTP');
  if (url.username || url.password) flags.push('Credentials in URL');
  if (url.hostname.startsWith('xn--') || url.hostname.includes('.xn--')) flags.push('Punycode hostname');
  if (url.href.length > 2048) flags.push('Long URL');
  return {
    url: url.href, origin: url.origin, protocol: url.protocol.replace(':', ''),
    hostname: url.hostname, port: url.port || null, pathname: url.pathname,
    queryKeys: [...new Set(query)].slice(0, 50), hasFragment: !!url.hash,
    flags,
  };
}

// Runs inside a web page in both Playwright and Electron. Keep it self-contained.
export function extractPage() {
  const clean = value => String(value || '').replace(/\s+/g, ' ').trim();
  const meta = selector => document.querySelector(selector)?.getAttribute('content') || null;
  const main = document.querySelector('main, article, [role="main"]') || document.body;
  const text = clean(main?.innerText || '').slice(0, 12000);
  const nodes = selector => [...document.querySelectorAll(selector)];
  return {
    title: document.title,
    url: location.href,
    language: document.documentElement.lang || null,
    canonical: document.querySelector('link[rel="canonical"]')?.href || null,
    description: meta('meta[name="description"]'),
    robots: meta('meta[name="robots"]'),
    openGraph: { title: meta('meta[property="og:title"]'), description: meta('meta[property="og:description"]'), image: meta('meta[property="og:image"]') },
    headings: nodes('h1, h2, h3').slice(0, 60).map(node => ({ level: Number(node.tagName[1]), text: clean(node.innerText).slice(0, 250) })),
    links: nodes('a[href]').slice(0, 60).map(node => ({ text: clean(node.innerText || node.getAttribute('aria-label')).slice(0, 140), url: node.href })),
    images: nodes('img').slice(0, 30).map(node => ({ alt: clean(node.alt).slice(0, 140), url: node.currentSrc || node.src })),
    forms: nodes('form').slice(0, 20).map(node => ({ action: node.action, method: node.method, fields: [...node.querySelectorAll('input, textarea, select')].slice(0, 30).map(field => ({ name: field.name || null, type: field.type || field.tagName.toLowerCase(), label: field.getAttribute('aria-label') || field.placeholder || null })) })),
    text,
    wordCount: text ? text.split(/\s+/).length : 0,
  };
}
