import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const failures = [];

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await walk(entryPath)));
    else files.push(entryPath);
  }

  return files;
}

function requireMatch(source, pattern, message) {
  if (!pattern.test(source)) failures.push(message);
}

const robots = await readFile(path.join(dist, "robots.txt"), "utf8");
const sitemapIndex = await readFile(
  path.join(dist, "sitemap-index.xml"),
  "utf8"
);
const sitemap = await readFile(path.join(dist, "sitemap-0.xml"), "utf8");

if (/Disallow:\s*\/_astro\//i.test(robots)) {
  failures.push("robots.txt blocks /_astro/ resources");
}
requireMatch(
  robots,
  /^Sitemap:\s*https:\/\/psymon-ai\.github\.io\/sitemap-index\.xml$/m,
  "robots.txt is missing the production sitemap URL"
);
requireMatch(
  sitemapIndex,
  /https:\/\/psymon-ai\.github\.io\/sitemap-0\.xml/,
  "sitemap-index.xml does not reference sitemap-0.xml"
);

const htmlFiles = (await walk(dist)).filter(file => file.endsWith(".html"));
const htmlByFile = new Map();

for (const file of htmlFiles) {
  const relativePath = path.relative(dist, file).replaceAll(path.sep, "/");
  if (relativePath === "404.html") continue;

  const html = await readFile(file, "utf8");
  htmlByFile.set(file, html);
  requireMatch(html, /<title>[^<]+<\/title>/i, `${relativePath}: missing title`);
  requireMatch(
    html,
    /<meta name="description" content="[^"]+"/i,
    `${relativePath}: missing meta description`
  );
  requireMatch(
    html,
    /<link rel="canonical" href="https:\/\/psymon-ai\.github\.io\/[^"]*"/i,
    `${relativePath}: missing production canonical URL`
  );
  requireMatch(
    html,
    /<meta name="robots" content="index, follow"/i,
    `${relativePath}: page is not explicitly indexable`
  );
}

const postFiles = [...htmlByFile.entries()]
  .filter(([, html]) => /<article\b[^>]*\bid="article"/i.test(html))
  .map(([file]) => file);

for (const file of postFiles) {
  const relativePath = path.relative(dist, file).replaceAll(path.sep, "/");
  const html = htmlByFile.get(file);
  const canonical = html.match(
    /<link rel="canonical" href="(https:\/\/psymon-ai\.github\.io\/[^"]*)"/i
  )?.[1];

  requireMatch(
    html,
    /<meta property="og:type" content="article"/i,
    `${relativePath}: missing article Open Graph type`
  );
  requireMatch(
    html,
    /"@type":"BlogPosting"/,
    `${relativePath}: missing BlogPosting structured data`
  );

  if (!canonical || !sitemap.includes(`<loc>${canonical}</loc>`)) {
    failures.push(`${relativePath}: canonical URL is missing from sitemap`);
  }
}

const home = await readFile(path.join(dist, "index.html"), "utf8");
requireMatch(
  home,
  /<title>[^<]*로컬 AI[^<]*LLM[^<]*<\/title>/i,
  "home page title must describe the Local AI and LLM topics"
);

if (failures.length > 0) {
  console.error("SEO checks failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log(
    `SEO checks passed: ${htmlFiles.length} pages, ${postFiles.length} post pages`
  );
}
