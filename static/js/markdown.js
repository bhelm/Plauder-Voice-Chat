/* Compact, standalone Markdown→HTML renderer for agent replies, plus the
   shared HTML-escape helper. Pure string→string (no DOM, no app state) —
   unit-tested via tests/client/pure_modules.test.mjs. */
/* ===== Markdown renderer =====
   Compact, standalone Markdown→HTML renderer.
   The agent replies are Markdown; in the UI we want to see them rendered,
   while the TTS voice already gets the cleaned plaintext on the server side.

   Safe: every raw input is HTML-escaped FIRST, after which only
   Markdown constructs are promoted to HTML. No lib, no eval, no innerHTML
   with un-escaped user content.

   Supports:
     - Fenced code blocks ```lang ... ```
     - Inline code `...`
     - Headings # through ######
     - Lists (- * +) and numbered (1. 2.)
     - Block quotes >
     - Horizontal rules ---
     - **bold**, *italic*, ~~strike~~
     - [text](url)  (only http/https/mailto)
     - Tables (GFM style with |)
     - Automatic http(s):// links
*/
const MD_PLACEHOLDER_RE = /\u0000MD(\d+)\u0000/g;
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function safeUrl(url) {
  // Allows http(s)://, mailto:, and relative #anchor / /path
  const u = String(url).trim();
  if (/^(https?:\/\/|mailto:)/i.test(u)) return u;
  if (/^[#\/]/.test(u)) return u;
  return null;
}
function renderInline(text) {
  // CAUTION: text must already be HTML-escaped!
  let s = text;
  // Inline code: `...`  — content will undergo NO further inline processing.
  // We temporarily remove it with placeholders.
  const codeSlots = [];
  s = s.replace(/`([^`\n]+)`/g, (_, code) => {
    codeSlots.push(`<code>${code}</code>`);
    return `\u0000IC${codeSlots.length - 1}\u0000`;
  });
  // Links: [text](url)
  s = s.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (m, label, url) => {
    const safe = safeUrl(url);
    if (!safe) return m; // unsafe → leave unchanged
    return `<a href="${escapeHtml(safe)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });
  // Auto-links http(s)://… (only when not already inside an href)
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)\]]+)/g, (m, pre, url) => {
    const safe = safeUrl(url);
    if (!safe) return m;
    return `${pre}<a href="${escapeHtml(safe)}" target="_blank" rel="noopener noreferrer">${url}</a>`;
  });
  // Bold: **x** or __x__
  s = s.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/__([^_\n]+?)__/g, '<strong>$1</strong>');
  // Italic: *x* or _x_  (no word-internal _)
  s = s.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
  s = s.replace(/(^|\W)_([^_\n]+?)_(?!\w)/g, '$1<em>$2</em>');
  // Strike: ~~x~~
  s = s.replace(/~~([^~\n]+?)~~/g, '<del>$1</del>');
  // Reinsert code placeholders
  s = s.replace(/\u0000IC(\d+)\u0000/g, (_, i) => codeSlots[Number(i)]);
  return s;
}
function parseTableRow(line) {
  // "| a | b | c |"  → ["a","b","c"]  (leading/trailing pipe optional)
  let cells = line.trim();
  if (cells.startsWith('|')) cells = cells.slice(1);
  if (cells.endsWith('|')) cells = cells.slice(0, -1);
  return cells.split('|').map(c => c.trim());
}
function isTableSeparator(line) {
  // "|---|---|" or "---|---"
  const row = line.trim().replace(/^\|/, '').replace(/\|$/, '');
  if (!row.includes('|') && !row.includes('-')) return false;
  return row.split('|').every(seg => /^\s*:?-{3,}:?\s*$/.test(seg));
}
function renderMarkdown(src) {
  if (!src) return '';
  // 1) Extract fenced code blocks — their content must NOT be inline-processed.
  const codeBlocks = [];
  let text = String(src).replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const langClass = lang ? ` class="lang-${escapeHtml(lang)}"` : '';
    codeBlocks.push(`<pre><code${langClass}>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`);
    return `\u0000MD${codeBlocks.length - 1}\u0000`;
  });
  // 2) Escape HTML — safe now
  text = escapeHtml(text);
  // 3) Parse line by line, block-based
  const lines = text.split('\n');
  const out = [];
  let i = 0;
  let para = [];
  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${renderInline(para.join(' ').trim())}</p>`);
      para = [];
    }
  };
  while (i < lines.length) {
    const line = lines[i];
    // Placeholder (code block) as its own line
    if (/^\u0000MD\d+\u0000\s*$/.test(line)) {
      flushPara();
      out.push(line.trim());
      i++; continue;
    }
    // Blank line → end paragraph
    if (/^\s*$/.test(line)) {
      flushPara();
      i++; continue;
    }
    // Horizontal rule
    if (/^\s*(-\s*){3,}\s*$/.test(line) || /^\s*(\*\s*){3,}\s*$/.test(line) || /^\s*(_\s*){3,}\s*$/.test(line)) {
      flushPara();
      out.push('<hr>');
      i++; continue;
    }
    // Headings
    const h = line.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (h) {
      flushPara();
      const level = h[1].length;
      out.push(`<h${level}>${renderInline(h[2])}</h${level}>`);
      i++; continue;
    }
    // Table (GFM)
    if (/\|/.test(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      flushPara();
      const headers = parseTableRow(line);
      i += 2; // skip header + separator
      const rows = [];
      while (i < lines.length && /\|/.test(lines[i]) && !/^\s*$/.test(lines[i])) {
        rows.push(parseTableRow(lines[i]));
        i++;
      }
      const thead = '<thead><tr>' + headers.map(h => `<th>${renderInline(h)}</th>`).join('') + '</tr></thead>';
      const tbody = '<tbody>' + rows.map(r => '<tr>' + r.map(c => `<td>${renderInline(c)}</td>`).join('') + '</tr>').join('') + '</tbody>';
      out.push(`<table>${thead}${tbody}</table>`);
      continue;
    }
    // Block quote — after escapeHtml ">" is now "&gt;".
    if (/^\s{0,3}&gt;\s?/.test(line)) {
      flushPara();
      const quote = [];
      while (i < lines.length && /^\s{0,3}&gt;\s?/.test(lines[i])) {
        quote.push(lines[i].replace(/^\s{0,3}&gt;\s?/, ''));
        i++;
      }
      out.push(`<blockquote>${renderInline(quote.join(' '))}</blockquote>`);
      continue;
    }
    // Lists (ul/ol)
    const ulMatch = line.match(/^(\s*)[-*+]\s+(.*)$/);
    const olMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/);
    if (ulMatch || olMatch) {
      flushPara();
      const ordered = !!olMatch;
      const items = [];
      while (i < lines.length) {
        const m1 = lines[i].match(/^(\s*)[-*+]\s+(.*)$/);
        const m2 = lines[i].match(/^(\s*)(\d+)\.\s+(.*)$/);
        const matched = ordered ? m2 : m1;
        if (!matched) break;
        items.push(renderInline(matched[ordered ? 3 : 2]));
        i++;
        // Attach continuation lines (indented) to the previous item
        while (i < lines.length && /^ {2,}\S/.test(lines[i]) && !lines[i].match(/^\s*([-*+]|\d+\.)\s+/)) {
          items[items.length - 1] += ' ' + renderInline(lines[i].trim());
          i++;
        }
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map(it => `<li>${it}</li>`).join('')}</${tag}>`);
      continue;
    }
    // Otherwise: paragraph line
    para.push(line);
    i++;
  }
  flushPara();
  // 4) Reinsert code-block placeholders
  let html = out.join('\n');
  html = html.replace(MD_PLACEHOLDER_RE, (_, idx) => codeBlocks[Number(idx)] || '');
  return html;
}
