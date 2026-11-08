# InferenceLearn

A free, pixel-styled course on LLM inference engineering. Plain HTML/CSS/JS with no framework and no build step needed to view it.

## View it
Open `index.html` in a browser.

## Structure
| Path | What |
|---|---|
| `index.html`, `roadmap.html`, `learn.html` | Home, roadmap, curriculum |
| `lessons/*.html` | 13 lessons with interactive labs |
| `practice.html` | Quiz Arcade + question bank |
| `interview.html`, `workbook.html` | Interview prep, napkin-math workbook |
| `vocabulary.html` | 129 terms + flashcards (`glossary.html` redirects here) |
| `resources.html` | Curated papers, blogs, repos |
| `assets/js/data.js` | **Single source of truth** for lessons, levels and pages |
| `assets/js/main.js` | Navigation, theme, progress, vocabulary, AI-tutor box |
| `assets/js/widgets.js` | The 6 interactive labs |
| `assets/js/practice.js` | Quiz Arcade engine (reads the static `#qbank`) |
| `tools/build.py` | Makes everything readable without JavaScript, and generates Markdown + `llms.txt` |
| `*.md`, `llms.txt`, `llms-full.txt` | Generated plain-text versions for AI assistants |

## After editing content: rebuild
```bash
pip install beautifulsoup4
python tools/build.py                                    # relative links
python tools/build.py --site-url https://YOU.github.io/inferencelearn   # once hosted (adds absolute URLs + sitemap.xml)
```
The script is safe to re-run: it only rewrites the `<!--B:...-->` marker blocks. Run it whenever you change a lesson, `data.js`, `vocabulary.html` or `interview.html`.

- **Add a lesson:** copy an existing lesson file, add an entry to `assets/js/data.js` (including its `vocab` terms, which must match `<dt>` names in `vocabulary.html`), then run the build.
- **Add interview questions to a lesson:** tag the question in `interview.html` with `data-lesson="<lesson id>"`. The first 3 appear in that lesson's "Interview corner".
- **Add quiz questions:** add an `<li>` to `#qbank` in `practice.html` (`data-topic`, `data-lesson`, `data-a`).

## Let ChatGPT / Claude / Gemini teach from it
AI assistants can only open public URLs, not `file://` paths. Host the folder on any static host:
- **GitHub Pages:** push this folder to a repo → Settings → Pages → deploy from branch.
- Or Netlify / Cloudflare Pages / Vercel: drag and drop the folder.

Then give the assistant any page link, or `.../llms-full.txt` for the whole course. Every lesson also has a "Study this with an AI tutor" box with a ready-made prompt.
