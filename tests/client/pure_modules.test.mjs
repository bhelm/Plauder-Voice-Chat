// Unit smoke test for the pure split client modules (static/js/*.js).
// They are classic scripts sharing top-level bindings — evaluated here in one
// vm context, exactly like the browser's global lexical environment.
// Run via tests/test_client_js.py (pytest) or: node tests/client/pure_modules.test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const ctx = vm.createContext({ TextDecoder });
for (const f of ['i18n.js', 'markdown.js', 'vct.js']) {
  vm.runInContext(readFileSync(join(ROOT, 'static/js', f), 'utf8'), ctx, { filename: f });
}
const g = (expr) => vm.runInContext(expr, ctx);

// --- i18n: both languages exist and cover the SAME keys (CLAUDE.md rule) ---
// (spread: vm-context arrays have a foreign Array.prototype that trips
// deepStrictEqual — copy values into host-realm arrays first)
const keysEn = [...g('Object.keys(I18N.en)')];
const keysDe = [...g('Object.keys(I18N.de)')];
const missDe = keysEn.filter((k) => !keysDe.includes(k));
const missEn = keysDe.filter((k) => !keysEn.includes(k));
assert.deepEqual(missDe, [], `keys missing in I18N.de: ${missDe}`);
assert.deepEqual(missEn, [], `keys missing in I18N.en: ${missEn}`);
assert.ok(keysEn.length > 100, 'suspiciously few i18n keys');

// --- markdown: rendering + escaping ---------------------------------------
assert.match(g(`renderMarkdown('**b** und \`c\`')`), /<strong>b<\/strong>/);
assert.match(g(`renderMarkdown('**b** und \`c\`')`), /<code>c<\/code>/);
assert.match(g(`renderMarkdown('# H1')`), /<h1>H1<\/h1>/);
assert.match(g(`renderMarkdown('- a\\n- b')`), /<ul><li>a<\/li><li>b<\/li><\/ul>/);
assert.match(g(`renderMarkdown('| a | b |\\n|---|---|\\n| 1 | 2 |')`), /<table>/);
// XSS: raw HTML is escaped, javascript: URLs are not linkified.
const xss = g(`renderMarkdown('<img src=x onerror=alert(1)>')`);
assert.ok(!xss.includes('<img'), 'raw HTML must be escaped');
const evil = g(`renderMarkdown('[x](javascript:alert(1))')`);
assert.ok(!evil.includes('href="javascript:'), 'javascript: URL must not link');
assert.equal(g(`escapeHtml('<a "b" \\'c\\'>')`), '&lt;a &quot;b&quot; &#39;c&#39;&gt;');

// --- vct: frame parser roundtrips ------------------------------------------
function frame(magic, turnId, seqAndPayload) {
  const id = [...turnId].map((c) => c.charCodeAt(0));
  return Uint8Array.from([...magic.split('').map((c) => c.charCodeAt(0)),
                          id.length, ...id, ...seqAndPayload]).buffer;
}
// VCT2: seq=0x0102, one PCM16LE sample = -2 (0xFE 0xFF)
vm.runInContext('globalThis._b = null', ctx);
ctx._b = frame('VCT2', 'turn1234', [0x01, 0x02, 0xFE, 0xFF]);
let r = g('parsePcmChunk(_b)');
assert.equal(r.turnId, 'turn1234');
assert.equal(r.seq, 0x0102);
assert.equal(g('_r = parsePcmChunk(_b); _r.int16[0]'), -2);
// VCT3: seq=7, two packets [1,2,3] and [9]
ctx._b = frame('VCT3', 'ab', [0, 7, 0, 3, 1, 2, 3, 0, 1, 9]);
r = g('_r = parseOpusChunk(_b); JSON.stringify({turnId:_r.turnId, seq:_r.seq, n:_r.packets.length, p0:[..._r.packets[0]], p1:[..._r.packets[1]]})');
assert.equal(r, JSON.stringify({ turnId: 'ab', seq: 7, n: 2, p0: [1, 2, 3], p1: [9] }));
// VCT1: WAV passthrough after header
ctx._b = frame('VCT1', 'xy', [82, 73, 70, 70]); // "RIFF"
r = g('_r = parseFramedWav(_b); JSON.stringify({turnId:_r.turnId, len:_r.wav.byteLength})');
assert.equal(r, JSON.stringify({ turnId: 'xy', len: 4 }));
// Wrong magic → null
ctx._b = frame('VCT9', 'xy', [0, 0]);
assert.equal(g('parseFramedWav(_b)'), null);
assert.equal(g('parsePcmChunk(_b)'), null);
assert.equal(g('parseOpusChunk(_b)'), null);

console.log(`pure_modules ok (${keysEn.length} i18n keys, md + vct asserts passed)`);
