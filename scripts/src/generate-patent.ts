import puppeteer from "puppeteer-core";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const CHROMIUM_PATH = process.env.REPLIT_PLAYWRIGHT_CHROMIUM_EXECUTABLE ?? "";

if (!CHROMIUM_PATH) {
  console.error("REPLIT_PLAYWRIGHT_CHROMIUM_EXECUTABLE is not set.");
  process.exit(1);
}

const HTML_PATH = path.resolve(__dirname, "patent.html");
const OUT_PATH = path.resolve(
  __dirname,
  "../../BLawzMusicLLC_MaxBooster_Patent_Application.pdf",
);

async function main() {
  console.log("Launching browser...");
  const browser = await puppeteer.launch({
    executablePath: CHROMIUM_PATH,
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--font-render-hinting=none",
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 816, height: 1056 });

  const fileUrl = `file://${HTML_PATH}`;
  console.log(`Loading: ${fileUrl}`);
  await page.goto(fileUrl, { waitUntil: "networkidle0", timeout: 30000 });

  await page.evaluate(() => {
    document.title =
      "MaxBooster Patent Application — B-Lawz Music LLC";
  });

  console.log("Generating PDF...");
  await page.pdf({
    path: OUT_PATH,
    format: "Letter",
    printBackground: true,
    margin: { top: "0", bottom: "0", left: "0", right: "0" },
    displayHeaderFooter: false,
  });

  await browser.close();

  const size = fs.statSync(OUT_PATH).size;
  console.log(`PDF generated: ${OUT_PATH} (${(size / 1024).toFixed(1)} KB)`);
}

main().catch((err) => {
  console.error("Error generating PDF:", err);
  process.exit(1);
});
