import { readFile, mkdir, writeFile, rename, chmod } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';

const here = path.dirname(fileURLToPath(import.meta.url));
export const defaultProfileDir = process.env.DIAGONAL_PROFILE_DIR || path.join(here, '.profile');

export function validateEmailProfiles(value) {
  if (!Array.isArray(value) || value.length > 3) throw new Error('Save no more than three email profiles');
  const seen = new Set();
  return value.map(item => {
    const nickname = String(item?.nickname || '').trim();
    const email = String(item?.email || '').trim();
    if (!nickname || nickname.length > 32 || !/^[\p{L}\p{N} _.-]+$/u.test(nickname)) throw new Error('Nicknames must be 1–32 plain characters');
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) || email.length > 254) throw new Error(`Invalid email for ${nickname}`);
    const key = nickname.toLocaleLowerCase();
    if (seen.has(key)) throw new Error('Email nicknames must be unique');
    seen.add(key);
    return { nickname, email };
  });
}

export async function loadEmailProfiles(dir = defaultProfileDir) {
  try { return validateEmailProfiles(JSON.parse(await readFile(path.join(dir, 'email_profiles.json'), 'utf8'))); }
  catch (error) { if (error.code === 'ENOENT') return []; throw error; }
}

export async function saveEmailProfiles(value, dir = defaultProfileDir) {
  const profiles = validateEmailProfiles(value);
  await mkdir(dir, { recursive: true });
  const target = path.join(dir, 'email_profiles.json');
  const temporary = path.join(dir, `.email_profiles-${randomUUID()}.tmp`);
  await writeFile(temporary, JSON.stringify(profiles, null, 2) + '\n', { mode: 0o600 });
  await rename(temporary, target);
  await chmod(target, 0o600);
  return profiles;
}

export function emailProfileByNickname(profiles, nickname) {
  const profile = profiles.find(item => item.nickname.toLocaleLowerCase() === String(nickname || '').trim().toLocaleLowerCase());
  if (!profile) throw new Error(`Unknown email nickname: ${nickname}`);
  return profile;
}
