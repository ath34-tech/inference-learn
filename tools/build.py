"""
InferenceLearn build script
===========================
Makes the site readable by search engines and LLMs (ChatGPT, Claude, Gemini...)
that do NOT run JavaScript, and generates plain-text versions of every page.

What it does (safe to run any number of times; edits are marker-delimited):
  1. Injects static HTML for everything main.js normally renders
     (header/nav, footer, lesson meta, pager, roadmap, curriculum).
  2. Adds a "Key vocabulary" box and an "Interview corner" to every lesson,
     pulled from vocabulary.html and interview.html (single source of truth).
  3. Adds <link rel="alternate" type="text/markdown">, JSON-LD and meta descriptions.
  4. Writes a Markdown twin for every page (page.html -> page.md).
  5. Writes llms.txt (site index for LLMs), llms-full.txt (whole course), robots.txt,
     and sitemap.xml (when --site-url is given).

Usage:
    pip install beautifulsoup4
    python tools/build.py                                   # relative links
    python tools/build.py --site-url https://you.github.io/inferencelearn
"""
import argparse
import sys
import html as htmllib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------- helpers
def slug(s: str) -> str:
    return re.sub(r"^-|-$", "", re.sub(r"[^a-z0-9]+", "-", s.lower()))


def load_data():
    src = (ROOT / "assets/js/data.js").read_text(encoding="utf8")
    body = src[src.index("{"): src.rindex("}") + 1]
    return json.loads(body)


def set_block(text, name, content, insert):
    """Replace <!--B:name-->...<!--/B:name--> or insert it with `insert(text, block)`."""
    block = f"<!--B:{name}-->{content}<!--/B:{name}-->"
    pat = re.compile(rf"<!--B:{re.escape(name)}-->.*?<!--/B:{re.escape(name)}-->", re.S)
    if pat.search(text):
        return pat.sub(lambda m: block, text, count=1)
    new = insert(text, block)
    if new == text:
        print(f"  ! could not place block '{name}'")
    return new


def fill_div(div_id):
    """Insert a block inside an empty <div id="..."></div>."""
    def _ins(text, block):
        return text.replace(f'<div id="{div_id}"></div>', f'<div id="{div_id}">{block}</div>', 1)
    return _ins


def before(marker):
    def _ins(text, block):
        i = text.find(marker)
        return text if i < 0 else text[:i] + block + "\n" + text[i:]
    return _ins


def after_regex(pattern):
    def _ins(text, block):
        m = re.search(pattern, text, re.S)
        return text if not m else text[: m.end()] + "\n      " + block + text[m.end():]
    return _ins


# ----------------------------------------------------------------- static chrome
NAV = [("index.html", "Home"), ("roadmap.html", "Roadmap"), ("learn.html", "Lessons"),
       ("practice.html", "Practice"), ("vocabulary.html", "Vocab"), ("resources.html", "Resources")]


def static_header(rel):
    links = "".join(f'<a href="{rel}{f}">{t}</a>' for f, t in NAV)
    return (f'<header class="site-header"><div class="container header-inner">'
            f'<a class="brand" href="{rel}index.html"><span class="brand-text">Inference<span>Learn</span></span></a>'
            f'<nav class="site-nav" aria-label="Main">{links}</nav></div></header>')


def static_footer(rel):
    links = "".join(f'<a href="{rel}{f}">{t}</a>' for f, t in
                    [("roadmap.html", "Roadmap"), ("practice.html", "Quiz"), ("interview.html", "Interview prep"),
                     ("workbook.html", "Workbook"), ("vocabulary.html", "Vocabulary"), ("llms.txt", "llms.txt")])
    return (f'<footer class="site-footer"><div class="container footer-inner">'
            f'<p>InferenceLearn: learn how LLMs are served fast and cheap.</p>'
            f'<div class="footer-links">{links}</div></div></footer>')


def static_roadmap(data):
    out = []
    for lv in data["levels"]:
        ls = [l for l in data["lessons"] if l["level"] == lv["id"]]
        items = "".join(
            f'<li><span class="chk"></span><a href="{l["file"]}">{l["n"]} · {htmllib.escape(l["title"])}</a>'
            f'<span class="sub">{htmllib.escape(" · ".join(l["topics"]))}</span></li>' for l in ls)
        out.append(
            f'<div class="stage"><div class="card"><div class="stage-head"><span class="badge {lv["cls"]}">{lv["world"]}</span>'
            f'<span class="stage-world">{len(ls)} lessons · ~{sum(l["mins"] for l in ls)} min</span></div>'
            f'<h3>{lv["name"]}</h3><p class="muted" style="font-size:13.5px;margin:0">{lv["blurb"]}</p>'
            f'<ul class="stage-list">{items}</ul></div></div>')
    return "".join(out)


def static_curriculum(data):
    out = []
    for lv in data["levels"]:
        cards = "".join(
            f'<a class="card lesson-card" href="{l["file"]}"><span class="num">{l["n"]}</span>'
            f'<h3>{htmllib.escape(l["title"])}</h3><p>{htmllib.escape(l["desc"])}</p>'
            f'<div class="meta"><span>{len(l["topics"])} topics</span><span>~{l["mins"]} min</span></div></a>'
            for l in data["lessons"] if l["level"] == lv["id"])
        out.append(
            f'<section class="level-block" id="w{lv["id"]}"><h2><span class="badge {lv["cls"]}">{lv["world"]}</span> {lv["name"]}</h2>'
            f'<p class="muted">{lv["blurb"]}</p><div class="grid grid-3">{cards}</div></section>')
    return "".join(out)


# ----------------------------------------------------------------- vocab + interview sources
def load_vocab():
    soup = BeautifulSoup((ROOT / "vocabulary.html").read_text(encoding="utf8"), "html.parser")
    vocab = {}
    for card in soup.select(".vocab-card"):
        term = card.find("dt").get_text(strip=True)
        dd = card.find("dd")
        vx = dd.find(class_="vx")
        if vx:
            vx.extract()
        vocab[term] = re.sub(r"\s+", " ", dd.decode_contents()).strip()
    return vocab


def add_vocab_ids():
    p = ROOT / "vocabulary.html"
    t = p.read_text(encoding="utf8")

    def rep(m):
        term = htmllib.unescape(re.sub(r"<[^>]+>", "", m.group(3)))
        return f'<div class="card vocab-card" id="v-{slug(term)}" data-cat="{m.group(2)}"><dt>{m.group(3)}</dt>'

    t2 = re.sub(r'<div class="card vocab-card"( id="[^"]*")? data-cat="(\w+)"><dt>(.*?)</dt>', rep, t)
    p.write_text(t2, encoding="utf8")


def load_interview():
    soup = BeautifulSoup((ROOT / "interview.html").read_text(encoding="utf8"), "html.parser")
    by_lesson = {}
    for d in soup.select("details.iv"):
        by_lesson.setdefault(d.get("data-lesson"), []).append(str(d))
    return by_lesson


# ----------------------------------------------------------------- page processing
def page_list(data):
    pages = [ROOT / p["file"] for p in data["pages"]]
    pages += [ROOT / l["file"] for l in data["lessons"]]
    return pages


def process_page(path, data, vocab, interview, site_url):
    rel = "../" if path.parent.name == "lessons" else ""
    text = path.read_text(encoding="utf8")
    relpath = path.relative_to(ROOT).as_posix()
    md_name = path.with_suffix(".md").name
    lesson = next((l for l in data["lessons"] if l["file"] == relpath), None)
    page = next((p for p in data["pages"] if p["file"] == relpath), None)

    # scripts
    scripts = "".join(f'<script src="{rel}assets/js/{s}"></script>' for s in ("data.js", "main.js", "widgets.js", "practice.js"))
    text = set_block(text, "scripts", scripts, lambda t, b: re.sub(r'<script src="[./]*assets/js/main\.js"></script>', lambda m: b, t, count=1))

    # head: description, markdown alternate, JSON-LD
    has_desc = re.search(r'<meta name="description"', re.sub(r"<!--B:head-->.*?<!--/B:head-->", "", text, flags=re.S))
    desc = (lesson or {}).get("desc") or (page or {}).get("desc") or data["site"]["tagline"]
    title = lesson["title"] if lesson else (page["title"] if page else "InferenceLearn")
    url = (site_url.rstrip("/") + "/" + relpath) if site_url else relpath
    if lesson:
        lv = next(v for v in data["levels"] if v["id"] == lesson["level"])
        ld = {"@context": "https://schema.org", "@type": "LearningResource", "name": lesson["title"],
              "description": lesson["desc"], "learningResourceType": "lesson", "educationalLevel": lv["name"],
              "timeRequired": f'PT{lesson["mins"]}M', "teaches": lesson["topics"], "inLanguage": "en",
              "isAccessibleForFree": True, "url": url,
              "isPartOf": {"@type": "Course", "name": "InferenceLearn: Inference Engineering", "description": data["site"]["tagline"]}}
    else:
        ld = {"@context": "https://schema.org", "@type": "WebPage", "name": title, "description": desc, "url": url,
              "isPartOf": {"@type": "WebSite", "name": "InferenceLearn"}}
    head = ("\n  " + (f'<meta name="description" content="{htmllib.escape(desc)}">\n  ' if not has_desc else "")
            + f'<link rel="alternate" type="text/markdown" href="{md_name}" title="Plain-text (Markdown) version for AI assistants">\n  '
            + f'<script type="application/ld+json">{json.dumps(ld, ensure_ascii=False)}</script>\n  ')
    text = set_block(text, "head", head, before("</head>"))

    # chrome
    text = set_block(text, "header", static_header(rel), fill_div("site-header"))
    text = set_block(text, "footer", static_footer(rel), fill_div("site-footer"))

    if relpath == "roadmap.html":
        text = set_block(text, "roadmap", static_roadmap(data), lambda t, b: t.replace('<div class="roadmap" id="roadmap"></div>', f'<div class="roadmap" id="roadmap">{b}</div>', 1))
    if relpath == "learn.html":
        text = set_block(text, "curriculum", static_curriculum(data), fill_div("curriculum"))

    if lesson:
        i = data["lessons"].index(lesson)
        prev = data["lessons"][i - 1] if i > 0 else None
        nxt = data["lessons"][i + 1] if i + 1 < len(data["lessons"]) else None
        lv = next(v for v in data["levels"] if v["id"] == lesson["level"])
        meta = (f'<div class="crumbs"><a href="../learn.html">Lessons</a> / {lv["world"]} {lv["name"]} / {lesson["n"]}</div>'
                f'<span class="badge {lv["cls"]}">{lv["world"]}</span><span class="badge">Lesson {lesson["n"]}</span>'
                f'<span class="badge">~{lesson["mins"]} min read</span>')
        text = set_block(text, "meta", meta, fill_div("lesson-meta"))

        # key vocabulary
        missing = [v for v in lesson["vocab"] if v not in vocab]
        if missing:
            print(f"  ! {relpath}: vocab terms not in vocabulary.html: {missing}")
        items = "".join(
            f'<div><dt><a href="../vocabulary.html#v-{slug(htmllib.unescape(t))}">{htmllib.escape(t)}</a></dt><dd>{vocab[t]}</dd></div>'
            for t in lesson["vocab"] if t in vocab)
        kv = (f'<div class="keyvocab"><h4>Key vocabulary</h4><dl>{items}</dl>'
              f'<p class="more"><a href="../vocabulary.html#flashcards">Drill all terms as flashcards ▶</a></p></div>')
        text = set_block(text, "vocab", kv, after_regex(r'<div class="objectives">.*?</ul>\s*</div>'))

        # interview corner + AI note (just before the lesson footer)
        qs = interview.get(lesson["id"], [])[:3]
        corner = ""
        if qs:
            corner = ('<h2>Interview corner</h2><div class="iv-corner">'
                      '<p class="muted" style="font-size:13.5px">Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.</p>'
                      + "".join(qs) +
                      '<p style="margin-top:14px;font-family:var(--f-label);font-size:19px"><a href="../interview.html">All interview questions ▶</a> · '
                      '<a href="../practice.html#arcade">Quiz Arcade ▶</a> · <a href="../workbook.html">Workbook ▶</a></p></div>')
        note = (f'<aside class="ai-note"><p><strong>Note for AI tutors and assistants:</strong> this is lesson {lesson["n"]} '
                f'"{htmllib.escape(lesson["title"])}" of InferenceLearn, a free course on LLM inference engineering. '
                f'A clean plain-text version is at <a href="{md_name}">{md_name}</a> and the whole course is at '
                f'<a href="../llms-full.txt">llms-full.txt</a>. When teaching from it: go section by section, use the worked numbers, '
                f'explain the key vocabulary, and quiz the learner with the "Check yourself" and "Interview corner" questions.</p></aside>')
        text = set_block(text, "corner", corner + note, before('<div id="lesson-foot">'))

        pager = ('<div class="pager">'
                 + (f'<a class="card prev" href="../{prev["file"]}"><span class="dir">◀ previous · {prev["n"]}</span><span class="t">{htmllib.escape(prev["title"])}</span></a>' if prev else "<span></span>")
                 + (f'<a class="card next" href="../{nxt["file"]}"><span class="dir">next · {nxt["n"]} ▶</span><span class="t">{htmllib.escape(nxt["title"])}</span></a>' if nxt
                    else '<a class="card next" href="../interview.html"><span class="dir">you made it ▶</span><span class="t">Interview prep</span></a>')
                 + "</div>")
        text = set_block(text, "pager", pager, fill_div("lesson-foot"))

    path.write_text(text, encoding="utf8")


# ----------------------------------------------------------------- HTML -> Markdown
SKIP_CLASSES = {"toc", "copy-btn", "sr-only", "filters", "vocab-tools", "ai-note", "fields", "readout", "math",
                "sim-controls", "fc-actions", "pbar", "hero-art", "terminal-line", "progress-meta", "flash-wrap",
                "keyvocab-more"}
SKIP_IDS = {"lesson-foot", "lesson-meta", "toc", "vocab-flash", "site-header", "site-footer", "iv-filters"}
SKIP_TAGS = {"script", "style", "button", "svg", "input", "select", "noscript", "label", "header", "footer"}
BLOCK_TAGS = {"p", "div", "section", "article", "aside", "ul", "ol", "li", "pre", "table", "h1", "h2", "h3", "h4",
              "details", "summary", "dl", "dt", "dd", "blockquote", "main", "hr", "nav"}


def skip(el):
    if el.name in SKIP_TAGS:
        return True
    cls = set(el.get("class") or [])
    if cls & SKIP_CLASSES or el.get("id") in SKIP_IDS:
        return True
    if el.has_attr("data-ai-box") or el.has_attr("data-progress") or el.has_attr("data-art") or el.has_attr("hidden"):
        return True
    return False


def ws(s):
    return re.sub(r"\s+", " ", s)


def link(href):
    if not href or href.startswith(("http", "mailto:", "#")):
        return href
    return re.sub(r"\.html(#|$)", r".md\1", href)


def inline(el):
    if isinstance(el, Comment):
        return ""
    if isinstance(el, NavigableString):
        return ws(str(el))
    if not isinstance(el, Tag) or skip(el):
        return ""
    inner = "".join(inline(c) for c in el.children)
    t, cls = el.name, el.get("class") or []
    if t in ("strong", "b"):
        return f"**{inner.strip()}**" if inner.strip() else ""
    if t in ("em", "i"):
        return f"*{inner.strip()}*" if inner.strip() else ""
    if t == "code":
        return f"`{el.get_text()}`"
    if t == "a":
        return f"[{inner.strip()}]({link(el.get('href'))})" if inner.strip() else ""
    if t == "br":
        return "  \n"
    if t == "sup":
        return "^" + inner
    if t == "span" and "badge" in cls:
        return f"[{inner.strip()}] "
    return inner


def has_block(el):
    return any(isinstance(c, Tag) and (c.name in BLOCK_TAGS) and not skip(c) for c in el.children)


def render_list(el):
    ordered = el.name == "ol"
    letters = el.get("type") == "A"
    lines, i = [], 0
    for li in el.find_all("li", recursive=False):
        i += 1
        marker = f"{chr(64 + i)}." if letters else (f"{i}." if ordered else "-")
        sub = []
        if has_block(li):
            blocks(li, sub)
        else:
            sub = [inline(li).strip()]
        sub = [s for s in sub if s]
        if not sub:
            continue
        first, rest = sub[0], sub[1:]
        lines.append(f"{marker} " + first.replace("\n", "\n   "))
        for s in rest:
            lines.append("   " + s.replace("\n", "\n   "))
    return "\n".join(lines)


def render_table(el):
    rows = []
    for tr in el.find_all("tr"):
        rows.append([inline(c).strip().replace("|", "\\|") for c in tr.find_all(["th", "td"])])
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def blocks(el, out):
    for c in el.children:
        if isinstance(c, NavigableString):
            s = ws(str(c)).strip()
            if s and not isinstance(c, Comment):
                out.append(s)
            continue
        if not isinstance(c, Tag) or skip(c):
            continue
        t, cls = c.name, set(c.get("class") or [])
        if c.get("id") == "arcade":
            out.append("*(An interactive quiz game runs here on the web page. The full question bank with answers is below.)*")
        elif t in ("h1", "h2", "h3", "h4"):
            out.append("#" * int(t[1]) + " " + inline(c).strip().lstrip("▶ ").strip())
        elif t == "p":
            s = inline(c).strip()
            if s:
                out.append(s)
        elif t in ("ul", "ol"):
            s = render_list(c)
            if s:
                out.append(s)
        elif t == "pre":
            code = c.find("code") or c
            out.append("```\n" + code.get_text().rstrip() + "\n```")
        elif t == "table":
            out.append(render_table(c))
        elif t == "details":
            summ = c.find("summary")
            out.append("**" + (inline(summ).strip() if summ else "Details") + "**")
            sub = []
            for ch in c.children:
                if ch is summ:
                    continue
                if isinstance(ch, Tag):
                    wrapper = BeautifulSoup("<div></div>", "html.parser").div
                    wrapper.append(ch.__copy__())
                    blocks(wrapper, sub)
            out.extend(sub)
        elif t == "dl":
            items = []
            for dt in c.find_all("dt"):
                dd = dt.find_next_sibling("dd")
                items.append(f"- **{inline(dt).strip()}**: {inline(dd).strip() if dd else ''}")
            out.append("\n".join(items))
        elif "formula" in cls:
            out.append("> " + inline(c).strip())
        elif "callout" in cls:
            tag = c.find(class_="tag")
            label = tag.get_text(strip=True) if tag else "NOTE"
            sub = []
            for ch in c.children:
                if ch is tag:
                    continue
                if isinstance(ch, Tag):
                    wrapper = BeautifulSoup("<div></div>", "html.parser").div
                    wrapper.append(ch.__copy__())
                    blocks(wrapper, sub)
            out.append("> **" + label + ":** " + " ".join(sub))
        elif "widget" in cls:
            h = c.find("h4")
            intro = c.find(class_="intro")
            note = c.find(class_="note")
            name = h.get_text(strip=True).lstrip("▶ ").strip() if h else "Interactive lab"
            out.append(f"> **Interactive lab: {name}{'' if name[-1:] in '?!.' else '.'}** " + (inline(intro).strip() + " " if intro else "")
                       + "*(Use it on the web page. The formulas it uses are explained in the text.)*"
                       + (" " + inline(note).strip() if note else ""))
        elif "diagram" in cls:
            pre = c.find("pre")
            if pre:
                out.append("```\n" + (pre.find("code") or pre).get_text().rstrip() + "\n```")
            flow = c.find(class_="flow")
            if flow:
                steps = [ws(s.get_text(" ")).strip() for s in flow.find_all(class_="step")]
                out.append("Flow: " + " → ".join(steps))
            rows = c.find_all(class_="blkrow")
            if rows:
                lines = []
                for r in rows:
                    lbl = r.find(class_="lbl")
                    cells = [ws(b.get_text(" ")).strip() or "□" for b in r.find_all(class_="blk")]
                    lines.append(f"{lbl.get_text(strip=True) if lbl else ''}: {' '.join(cells)}".strip())
                out.append("```\n" + "\n".join(lines) + "\n```")
            cap = c.find(class_="cap")
            if cap:
                out.append(f"*Diagram: {inline(cap).strip()}*")
        elif t == "hr":
            out.append("---")
        elif has_block(c) or t in ("main", "section", "article", "aside", "nav"):
            blocks(c, out)
        else:
            s = inline(c).strip()
            if s:
                out.append(s)


def to_markdown(path, data, site_url):
    soup = BeautifulSoup(path.read_text(encoding="utf8"), "html.parser")
    main = soup.find("main") or soup.body
    relpath = path.relative_to(ROOT).as_posix()
    h1 = main.find("h1")
    title = h1.get_text(strip=True) if h1 else soup.title.get_text(strip=True)
    if h1:
        h1.decompose()
    lesson = next((l for l in data["lessons"] if l["file"] == relpath), None)
    base = site_url.rstrip("/") + "/" if site_url else ""
    head = [f"# {title}"]
    src = f"{base}{relpath}" if base else relpath
    meta = f"> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: {src}. Whole course in one file: {base}llms-full.txt"
    if lesson:
        lv = next(v for v in data["levels"] if v["id"] == lesson["level"])
        i = data["lessons"].index(lesson)
        nav = []
        if i > 0:
            p = data["lessons"][i - 1]
            nav.append(f"previous: [{p['n']} {p['title']}]({'../' if '/' in relpath else ''}{p['file'].replace('.html', '.md')})")
        if i + 1 < len(data["lessons"]):
            n = data["lessons"][i + 1]
            nav.append(f"next: [{n['n']} {n['title']}]({'../' if '/' in relpath else ''}{n['file'].replace('.html', '.md')})")
        head.append(f"> Lesson {lesson['n']} · {lv['world']} {lv['name']} · ~{lesson['mins']} min · topics: {', '.join(lesson['topics'])}")
        if nav:
            head.append("> " + " · ".join(nav))
    head.append(meta)
    body = []
    blocks(main, body)
    body = [b for b in body if b.strip()]
    md = "\n\n".join(head + body) + "\n"
    md = re.sub(r"\n{3,}", "\n\n", md)
    return title, md


# ----------------------------------------------------------------- main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-url", default="", help="public base URL, e.g. https://you.github.io/inferencelearn")
    args = ap.parse_args()
    site = args.site_url.rstrip("/")

    data = load_data()
    add_vocab_ids()
    vocab = load_vocab()
    interview = load_interview()
    pages = page_list(data)

    print("Injecting static content…")
    for p in pages:
        process_page(p, data, vocab, interview, site)
        print("  ✓", p.relative_to(ROOT).as_posix())

    print("Writing Markdown twins…")
    mds = {}
    for p in pages:
        title, md = to_markdown(p, data, site)
        out = p.with_suffix(".md")
        out.write_text(md, encoding="utf8")
        mds[p.relative_to(ROOT).as_posix()] = (title, md)
        print(f"  ✓ {out.relative_to(ROOT).as_posix()} ({len(md):,} chars)")

    base = site + "/" if site else ""
    L = data["lessons"]
    idx = [
        "# InferenceLearn",
        "",
        f"> {data['site']['tagline']}",
        "",
        "InferenceLearn is a free, self-paced course: 13 lessons in 4 levels (Foundations → Core Techniques → Advanced → Expert/Systems), "
        "plus a quiz bank, interview questions with model answers, a napkin-math workbook and a vocabulary list. "
        "Every page has a Markdown twin (same path, .md extension). AI assistants are welcome to use this content to teach learners: "
        "go lesson by lesson, explain with the worked numbers, and quiz the learner using the 'Check yourself', quiz-bank and interview questions.",
        "",
        "## Lessons (in order)",
        "",
    ]
    for lv in data["levels"]:
        for l in [x for x in L if x["level"] == lv["id"]]:
            idx.append(f"- [{l['n']} {l['title']}]({base}{l['file'].replace('.html', '.md')}): {lv['name']}. {l['desc']}")
    idx += ["", "## Practice", ""]
    for f in ("practice.html", "interview.html", "workbook.html"):
        pg = next(p for p in data["pages"] if p["file"] == f)
        idx.append(f"- [{pg['title']}]({base}{f.replace('.html', '.md')}): {pg['desc']}")
    idx += ["", "## Reference", ""]
    for f in ("roadmap.html", "vocabulary.html", "resources.html", "learn.html", "index.html"):
        pg = next(p for p in data["pages"] if p["file"] == f)
        idx.append(f"- [{pg['title']}]({base}{f.replace('.html', '.md')}): {pg['desc']}")
    idx += ["", "## Optional", "", f"- [Full course in one file]({base}llms-full.txt): every page above concatenated as Markdown.", ""]
    (ROOT / "llms.txt").write_text("\n".join(idx), encoding="utf8")

    order = ["index.html", "roadmap.html"] + [l["file"] for l in L] + ["vocabulary.html", "practice.html", "interview.html", "workbook.html", "resources.html"]
    full = [f"# InferenceLearn: complete course text\n\n> {data['site']['tagline']}\n\n"
            "This file contains every page of the course as Markdown, in study order. Pages are separated by lines of '='.\n"]
    for f in order:
        title, md = mds[f]
        full.append("\n" + "=" * 80 + f"\nPAGE: {f.replace('.html', '.md')}\n" + "=" * 80 + "\n\n" + md)
    (ROOT / "llms-full.txt").write_text("".join(full), encoding="utf8")
    print(f"  ✓ llms.txt, llms-full.txt ({sum(len(m[1]) for m in mds.values()):,} chars)")

    robots = "User-agent: *\nAllow: /\n" + (f"\nSitemap: {site}/sitemap.xml\n" if site else "")
    (ROOT / "robots.txt").write_text(robots, encoding="utf8")
    if site:
        urls = "".join(f"<url><loc>{site}/{p.relative_to(ROOT).as_posix()}</loc></url>" for p in pages)
        (ROOT / "sitemap.xml").write_text(f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>', encoding="utf8")
        print("  ✓ sitemap.xml")
    print("Done.")


if __name__ == "__main__":
    main()
