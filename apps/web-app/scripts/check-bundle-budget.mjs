import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const distRoot = join(process.cwd(), "dist");
const assetsRoot = join(distRoot, "assets");

const jsFiles = readdirSync(assetsRoot)
  .filter((fileName) => fileName.endsWith(".js"))
  .map((fileName) => join("assets", fileName));

let totalGzipBytes = 0;
let largestGzipBytes = 0;
let largestFile = "";

for (const relativePath of jsFiles) {
  const absolutePath = join(distRoot, relativePath);
  const fileContent = readFileSync(absolutePath);
  const gzipSize = gzipSync(fileContent).length;

  totalGzipBytes += gzipSize;
  if (gzipSize > largestGzipBytes) {
    largestGzipBytes = gzipSize;
    largestFile = relativePath;
  }

  statSync(absolutePath);
}

const totalBudgetBytes = 120 * 1024;
const largestBudgetBytes = 70 * 1024;

if (totalGzipBytes > totalBudgetBytes || largestGzipBytes > largestBudgetBytes) {
  const totalKb = (totalGzipBytes / 1024).toFixed(1);
  const largestKb = (largestGzipBytes / 1024).toFixed(1);
  const totalLimitKb = (totalBudgetBytes / 1024).toFixed(1);
  const largestLimitKb = (largestBudgetBytes / 1024).toFixed(1);
  throw new Error(
    `Bundle budget exceeded. total=${totalKb}KB (limit ${totalLimitKb}KB), largest=${largestKb}KB (${largestFile}, limit ${largestLimitKb}KB).`,
  );
}

const totalKb = (totalGzipBytes / 1024).toFixed(1);
const largestKb = (largestGzipBytes / 1024).toFixed(1);
console.log(`Bundle budget OK. total=${totalKb}KB gzip, largest=${largestKb}KB gzip (${largestFile}).`);
