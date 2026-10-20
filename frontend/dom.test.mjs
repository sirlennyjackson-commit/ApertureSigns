// Wiring test: loads index.html + app.js in jsdom with a mocked server.
import { JSDOM } from "jsdom";
import fs from "node:fs";
import assert from "node:assert/strict";
const root = new URL("./", import.meta.url).href;
const html = fs.readFileSync(new URL("index.html", root), "utf8").replace(/<script[^>]*><\/script>/, "").replace(/<link[^>]*>/, "");
const dom = new JSDOM(html, { url: "http://localhost:8000/app/", runScripts: "outside-only", pretendToBeVisual: true });
const { window } = dom;
window.HTMLCanvasElement.prototype.getContext = () => ({});
const calls = [];
window.fetch = async (url, opt) => {
  calls.push(url);
  const body = url.endsWith("/health") ? { model_loaded: true, model_labels: 3 }
    : url.endsWith("/pipeline") ? { translation: { status: "partial", fingerspelled_words: ["wanjiku"] },
        avatar: { timeline: [{ gloss: "HELLO", type: "lexical", start_ms: 0, duration_ms: 900 }, { gloss: "FS:WANJIKU", type: "fingerspelling", letters: ["W","A"], start_ms: 1020, duration_ms: 700 }] } }
    : {};
  return { ok: true, status: 200, json: async () => body };
};
Object.assign(globalThis, { window, document: window.document, location: window.location, localStorage: window.localStorage,
  fetch: window.fetch, WebSocket: class {} });
await import(new URL("app.js", root).href);
const $ = (id) => document.getElementById(id);
await new Promise((r) => setTimeout(r, 50));
assert.equal($("chip-model").textContent, "model: 3 signs");
$("text-in").value = "Hello Wanjiku";
$("btn-translate").onclick();
await new Promise((r) => setTimeout(r, 50));
assert.equal($("chips").children.length, 2);
assert.match($("text-note").textContent, /fingerspelled: wanjiku/);
assert.ok(!$("btn-play").classList.contains("hidden"));
$("t-record").click();
assert.ok(!$("tab-record").classList.contains("hidden") && $("tab-recognize").classList.contains("hidden"));
$("btn-record").onclick(); await new Promise((r) => setTimeout(r, 10));
assert.match($("rec-status").textContent, /Start the camera first/);
console.log("DOM wiring OK; fetch calls:", calls.map((c) => c.replace("http://localhost:8000", "")).join(", "));
process.exit(0);
