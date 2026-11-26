import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig, devices } from '@playwright/test';

// Prefer a prebuilt host (CI builds it first); a cold `cargo run` needs a much longer startup budget.
const prebuiltHost = fileURLToPath(new URL('../target/release/mw-host', import.meta.url));
const hasPrebuiltHost = existsSync(prebuiltHost);

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  reporter: [['line'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'on',
    ...devices['Desktop Chrome'],
  },
  webServer: [
    {
      command: hasPrebuiltHost
        ? './target/release/mw-host --port 7878'
        : 'cargo run --release -p mw-host -- --port 7878',
      cwd: '..',
      url: 'http://127.0.0.1:7878/v1/snapshot',
      timeout: hasPrebuiltHost ? 120_000 : 900_000,
      reuseExistingServer: false,
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 4173',
      cwd: '.',
      url: 'http://127.0.0.1:4173',
      timeout: 120_000,
      reuseExistingServer: false,
    },
  ],
});
