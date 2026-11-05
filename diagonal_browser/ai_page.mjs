// Run in an isolated world: websites cannot replace Kira's element references.
export function pageSnapshot() {
  const visible = element => { const rect = element.getBoundingClientRect(); const style = getComputedStyle(element); return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none'; };
  const nodes = [...document.querySelectorAll('a[href],button,input,textarea,select,[role="button"],[contenteditable="true"]')].filter(visible).slice(0, 160);
  globalThis.kiraElements = nodes;
  const clone = document.body?.cloneNode(true);
  clone?.querySelectorAll('input,textarea,select,[contenteditable],script,style,noscript').forEach(element => element.remove());
  return {
    url: location.href, title: document.title,
    text: (clone?.textContent || '').replace(/\s+/g, ' ').slice(0, 14000),
    elements: nodes.map((element, index) => ({ element: index + 1, tag: element.tagName.toLowerCase(), type: element.getAttribute('type') || '', label: (element.getAttribute('aria-label') || element.labels?.[0]?.innerText || element.innerText || element.getAttribute('placeholder') || '').slice(0, 140), href: element.tagName === 'A' ? element.href : undefined })),
  };
}

export function pageAction(action) {
  if (action.action === 'scroll') { window.scrollBy({ top: (action.direction === 'up' ? -1 : 1) * innerHeight * 0.75 }); return { summary: `Scrolled ${action.direction === 'up' ? 'up' : 'down'}` }; }
  const element = globalThis.kiraElements?.[action.element - 1];
  if (!element?.isConnected) throw new Error('Page changed; read it again before interacting');
  if (action.action === 'click') {
    const link = element.closest('a[href]');
    if (link) {
      const target = new URL(link.href);
      if (!['https:', 'http:'].includes(target.protocol)) throw new Error('This link is not a web page');
      if (target.origin !== location.origin || link.target === '_blank') return { permission: { action: 'open', url: target.href, reason: 'Follow this link in a new tab' } };
    }
    if (element.disabled) throw new Error('This control is disabled');
    element.scrollIntoView({ block: 'center' }); element.click(); return { summary: `Clicked element ${action.element}` };
  }
  if (action.action === 'fill') {
    if (!['INPUT', 'TEXTAREA'].includes(element.tagName) || element.disabled || element.readOnly) throw new Error('This field cannot be filled');
    const type = (element.type || '').toLowerCase();
    const sensitive = /password|cc-|one-time|credit|card.?number|cvv|cvc|secret|token/i.test([type, element.autocomplete, element.name, element.id].join(' '));
    if (sensitive || !['text', 'email', 'search', 'url', 'tel', 'textarea', 'number'].includes(type)) throw new Error('Enter sensitive credentials or payment details yourself');
    const descriptor = Object.getOwnPropertyDescriptor(element.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, 'value');
    descriptor.set.call(element, String(action.text || '')); element.dispatchEvent(new Event('input', { bubbles: true })); element.dispatchEvent(new Event('change', { bubbles: true }));
    return { summary: `Filled element ${action.element}` };
  }
  throw new Error('Unsupported page control');
}
