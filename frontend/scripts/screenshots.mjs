// Capture README screenshots of every tab from the running dev servers (API :8000, Vite :5173).
//   cd frontend && node scripts/screenshots.mjs
import { chromium } from 'playwright'

const BASE = process.env.UI_URL ?? 'http://localhost:5173'
const OUT = new URL('../../assets/screenshots/', import.meta.url).pathname
const shots = [
  { name: 'overview', path: '/overview' },
  { name: 'playground', path: '/playground', act: async (p) => { await p.getByRole('button', { name: /run/i }).first().click(); await p.waitForSelector('text=Retrieved chunks', { timeout: 120000 }) } },
  { name: 'compare', path: '/compare', act: async (p) => { await p.getByRole('button', { name: /compare 3 configs/i }).click(); await p.waitForSelector('text=Side by side', { timeout: 180000 }) } },
  { name: 'leaderboard', path: '/leaderboard', act: async (p) => { await p.waitForTimeout(2500) } },
  { name: 'chunk-explorer', path: '/chunks', act: async (p) => { await p.waitForSelector('text=chunk', { timeout: 30000 }); await p.waitForTimeout(1500) } },
  { name: 'knowledge-graph', path: '/graph', act: async (p) => { await p.waitForSelector('svg circle', { timeout: 30000 }); await p.waitForTimeout(500) } },
  { name: 'query-inspector', path: '/inspector', act: async (p) => { await p.waitForTimeout(1500); const b = p.locator('li button').first(); if (await b.count()) { await b.click(); await p.waitForTimeout(2500) } } },
]
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
for (const s of shots) {
  await page.goto(BASE + s.path, { waitUntil: 'networkidle' })
  try { if (s.act) await s.act(page) } catch (e) { console.warn(`[${s.name}] step skipped: ${e.message.split('\n')[0]}`) }
  await page.screenshot({ path: `${OUT}${s.name}.png`, fullPage: false })
  console.log('saved', s.name)
}
await browser.close()
