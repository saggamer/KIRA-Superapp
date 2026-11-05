import test from 'node:test';
import assert from 'node:assert/strict';
import { supportsKiraPopup } from '../ai_platform.mjs';

test('Kira popup is enabled only for the Apple Silicon MLX edition', () => {
  assert.equal(supportsKiraPopup('darwin', 'arm64'), true);
  for (const [platform, arch] of [['darwin', 'x64'], ['linux', 'arm64'], ['linux', 'x64'], ['win32', 'arm64'], ['win32', 'x64']]) assert.equal(supportsKiraPopup(platform, arch), false);
});
