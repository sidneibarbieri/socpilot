/** Capture local dashboard documentation with an explicitly selected browser. */
const { chromium } = require('playwright');

async function capture() {
  const [browserPath, dashboardUrl, outputPath] = process.argv.slice(2);
  if (!browserPath || !dashboardUrl || !outputPath) {
    throw new Error('Expected browser executable, local dashboard URL and output path.');
  }
  if (new URL(dashboardUrl).protocol !== 'file:') {
    throw new Error('Dashboard must be a local file.');
  }
  const browser = await chromium.launch({ executablePath: browserPath, headless: true });
  try {
    const page = await browser.newPage({
      viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1,
      reducedMotion: 'reduce',
    });
    const browserErrors = [];
    page.on('pageerror', error => browserErrors.push(error.message));
    await page.route('**/*', route => {
      const protocol = new URL(route.request().url()).protocol;
      return protocol === 'file:' || protocol === 'data:'
        ? route.continue() : route.abort('blockedbyclient');
    });
    await page.goto(dashboardUrl, { waitUntil: 'load', timeout: 30000 });
    await page.evaluate(async () => {
      await document.fonts.ready;
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    });
    const height = await page.evaluate(() => Math.ceil(Math.max(
      document.body.scrollHeight, document.documentElement.scrollHeight)));
    if (height < 1 || height > 16000) {
      throw new Error('Dashboard height is outside the supported preview range.');
    }
    if (browserErrors.length) throw new Error(browserErrors.join('\n'));
    await page.screenshot({ path: outputPath, fullPage: true, timeout: 30000 });
    process.stdout.write(JSON.stringify({ width: 1280, height }));
  } finally {
    await browser.close();
  }
}

capture().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
