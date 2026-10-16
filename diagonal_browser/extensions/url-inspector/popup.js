let inspection;
const $ = selector => document.querySelector(selector);

function readPage() {
  const main = document.querySelector('main, article, [role="main"]') || document.body;
  const clean = value => String(value || '').replace(/\s+/g, ' ').trim();
  return {
    title: document.title,
    description: document.querySelector('meta[name="description"]')?.content || null,
    canonical: document.querySelector('link[rel="canonical"]')?.href || null,
    headings: [...document.querySelectorAll('h1,h2,h3')].slice(0, 50).map(node => ({ level: Number(node.tagName[1]), text: clean(node.innerText).slice(0, 200) })),
    links: [...document.querySelectorAll('a[href]')].slice(0, 60).map(node => ({ text: clean(node.innerText).slice(0, 100), url: node.href })),
    text: clean(main?.innerText).slice(0, 15000),
  };
}

async function inspect() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !/^https?:\/\//i.test(tab.url || '')) throw new Error('Open an HTTP or HTTPS page first.');
    const url = new URL(tab.url);
    const [{ result: page }] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: readPage });
    inspection = {
      url: url.href, origin: url.origin, hostname: url.hostname,
      pathname: url.pathname, queryKeys: [...new Set(url.searchParams.keys())].slice(0, 30),
      page,
    };
    $('#origin').textContent = inspection.origin;
    $('#title').textContent = page.title || '(untitled)';
    $('#summary').textContent = `${page.headings.length} headings · ${page.links.length} links · ${page.text.split(/\s+/).length} words`;
    $('#status').textContent = 'Ready to copy structured page context.';
    $('#copy').disabled = false;
  } catch (error) {
    $('#status').textContent = error.message;
  }
}

$('#copy').onclick = async () => {
  if (!inspection) return;
  await navigator.clipboard.writeText(JSON.stringify(inspection, null, 2));
  $('#status').textContent = 'Copied JSON to clipboard.';
};
inspect();
