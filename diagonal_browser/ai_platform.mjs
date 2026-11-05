// AI mode ships only with the native Apple Silicon / MLX edition.
export function supportsKiraPopup(platform = process.platform, arch = process.arch) {
  return platform === 'darwin' && arch === 'arm64';
}
