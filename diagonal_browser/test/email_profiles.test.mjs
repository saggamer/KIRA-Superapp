import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { loadEmailProfiles, saveEmailProfiles, validateEmailProfiles, emailProfileByNickname } from '../email_profiles.mjs';

test('email nicknames are limited, validated, and saved privately', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-email-test-'));
  try {
    const profiles = await saveEmailProfiles([{ nickname: 'Work', email: 'work@example.com' }], dir);
    assert.deepEqual(await loadEmailProfiles(dir), profiles);
    assert.equal(emailProfileByNickname(profiles, 'work').email, 'work@example.com');
    assert.equal((await stat(path.join(dir, 'email_profiles.json'))).mode & 0o777, 0o600);
    assert.throws(() => validateEmailProfiles(Array(4).fill({ nickname: 'x', email: 'x@example.com' })), /three/);
    assert.throws(() => validateEmailProfiles([{ nickname: 'Work', email: 'a@example.com' }, { nickname: 'work', email: 'b@example.com' }]), /unique/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
