import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../ui.html', import.meta.url), 'utf8');
const code = html.slice(html.indexOf('        function imagePathsInMessage('),
                        html.indexOf('        function renderAIMessage('));
const element = () => ({children: [], style: {}, appendChild(child) {this.children.push(child);},
    insertBefore(child) {this.children.unshift(child);}});
const requested = [];
const context = vm.createContext({
    document: {createElement: element}, scrollToBottom() {},
    window: {pywebview: {api: {get_image_preview: async path => {
        requested.push(path);
        return {ok: true, name: 'generated image.png', width: 512, height: 512,
            data_url: 'data:image/png;base64,iVBORw0KGgo='};
    }}}},
});
vm.runInContext(code, context);
const text = 'Generated image: `/workspace/generated image.png`\nImage: `/workspace/generated image.png`';
assert.deepEqual(Array.from(context.imagePathsInMessage(text)), ['/workspace/generated image.png']);
assert.equal(context.imagePathsInMessage('![art](/workspace/image.png)')[0], '/workspace/image.png');
assert.equal(context.imagePathsInMessage('Image: /workspace/image.webp')[0], '/workspace/image.webp');
assert.equal(context.imagePathsInMessage('![x](https://example.test/x.png) `javascript:bad.png`').length, 0);
const container = element();
await context.renderImagePreviews(container, text);
assert.equal(requested.length, 1);
const [image, caption] = container.children[0].children;
assert.match(image.src, /^data:image\/png;base64,/);
assert.equal(image.alt, 'generated image.png');
assert.match(caption.textContent, /512 × 512/);
context.window.pywebview.api.get_image_preview = async () => ({ok: false, error: 'Saved image missing'});
const missing = element();
await context.renderImagePreviews(missing, text);
assert.equal(missing.children[0].children.length, 1);
assert.equal(missing.children[0].children[0].textContent, 'Saved image missing');
console.log('Inline image previews: path formats, deduplication, rendering and missing-file checks passed.');
