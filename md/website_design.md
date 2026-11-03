# Quadriga-Lib website overhaul: design

This document is the output of the design session (step 2 of `md/website_plan.md`). Together with the plan it is the input for the spec (step 3). It fixes the visual design, the navigation, the page templates and the components. It does not fix wording: the texts in the prototype are drafts, and section 9 lists every content gap. Step 7 deletes this document and the prototype; both stay in the git history.

The reference implementation is the static prototype in `html_builder/design_prototype/`. Where this document and the prototype differ, this document wins; where this document is silent, the prototype shows the intended result. The prototype pages were generated once from the real sources (C++ API Markdown, formats sources, publications list); they are not meant to be maintained.

## 1. Decisions in short

| Topic | Decision |
|-------|----------|
| Direction | Material style, close to Material for MkDocs and the Polars user guide: coloured top bar with a row of section tabs, white content, left sidebar on API and long text pages. Familiar to engineers who know Python and C++ documentation sites. |
| Colour | Green, to show the lineage from QuaDRiGa (quadriga-channel-model.de uses a soft green) and to sit well next to NVIDIA Sionna. Mid-green bar `#2e7d46` with white text, pale green `#eaf4ec` for the landing pitch area and note boxes, dark green-black text. |
| Dark mode | Yes, automatic: follows the device setting through `prefers-color-scheme`. No manual switch. |
| Typeface | IBM Plex Sans for text, IBM Plex Mono for code and function names. SIL Open Font Licence 1.1, self-hosted. |
| Logo | Shape unchanged (D16). One-colour SVG with `fill="currentColor"`: white on the green bar, green elsewhere. |
| API page | One page per language (D2). Sticky sidebar on wide screens, collapsible index on phones, language buttons per function, no right-hand "on this page" column. |
| Landing figures | Two figures made with the released pip package: a coverage map (`calc_diffraction_gain`) and a 3GPP panel antenna pattern (`arrayant.generate`). Placeholders until they exist; a separate session makes them (section 8). |
| JavaScript | Optional extras only (section 7). Every page works fully without it (D1). |

## 2. Prototype files

| File | Template | Content |
|------|----------|---------|
| `index.html` | Landing | Draft texts, placeholder figures and numbers |
| `cpp_api.html` | API page | The complete C++ API (137 functions), generated from `html_docu/quadriga_lib_api_cpp.md` |
| `formats.html` | Text page with index | Both formats sources from `html_builder/formats/` |
| `ai_agents.html` | Text page with index | Draft text for the AI page |
| `tutorials.html` | Tutorial index | Four dummy cards |
| `tutorial_sample.html` | Tutorial page | Dummy notebook: code cells, output figure, formula, downloads |
| `publications.html` | Publications list | The current list, regrouped |
| `impressum.html` | Legal page | Placeholders (D17) |
| `404.html` | Not found | |
| `css/site.css` | All | Tokens and components; the spec may restructure it but keeps the values |
| `js/site.js` | All | The optional extras of section 7 |
| `fonts/` | All | Seven woff2 files (Latin subset) and `LICENSE.txt` |
| `img/logo.svg` | All | Logo cleaned of editor data, recolourable |

The MATLAB and Python API pages, the download page and the FAQ page are not in the prototype. They use the API template and the text template unchanged.

To view the prototype, serve the folder with any static server, for example `python3 -m http.server --directory html_builder/design_prototype`.

## 3. Design tokens

### 3.1 Colours

Every colour is a CSS custom property on `:root`. The dark values replace the light ones inside `@media (prefers-color-scheme: dark)`. Components use only tokens, never literal colours. Contrast ratios were computed with the WCAG formula; all text pairs pass AA (4.5:1).

| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `--bg` | `#ffffff` | `#0b120e` | Page background |
| `--fg` | `#13241a` | `#e3ece5` | Body text (16.2:1 light, 15.7:1 dark) |
| `--muted` | `#4b5f52` | `#b0c2b5` | Secondary text, labels, captions (6.9:1, 10.1:1) |
| `--line` | `#dfe7e1` | `#2a3a30` | Borders, table rules, dividers |
| `--bar` | `#2e7d46` | `#24603a` | Header bar, primary button (light); white text 5.1:1 and 7.5:1 |
| `--bar-hover` | `#24603a` | `#2e7d46` | Hover of the primary button |
| `--pitch` | `#eaf4ec` | `#1d3a28` | Landing pitch area, note boxes, current item in the index |
| `--link` | `#2a7341` | `#7fd197` | Links, current tab or item, secondary button; 5.8:1 on `--bg` light, 10.4:1 dark; 5.1:1 and 6.8:1 on `--pitch` |
| `--link-hover` | `#1c5230` | `#a6e3b7` | Link hover |
| `--surface` | `#f3f6f4` | `#16221a` | Code blocks, table headers, footer |
| `--card` | `#ffffff` | `#111b15` | Cards, tab bars |
| `--btn-fg` | `#ffffff` | `#0b120e` | Text on the filled button (dark mode: the button is `--link`) |
| `--focus` | `#b8860b` | `#f2c14e` | Focus ring, 3 px; deliberately not green |
| `--warn-bg`, `--warn-fg` | `#fff1dc`, `#7a3d00` | `#3a2a12`, `#f5c98a` | "Experimental" badge, placeholder and warning notes (7.6:1, 9.0:1) |
| `--c-kw` | `#8a1f9a` | `#e3a6f0` | Code: keywords |
| `--c-ty` / `--c-fn` | `#1659b5` | `#8cc8ff` | Code: types, function names |
| `--c-st` | `#2a7341` | `#a5e8b8` | Code: strings |
| `--c-nu` | `#a3510a` | `#f3b37a` | Code: numbers |
| `--c-cm` | `#5d6b62` | `#93a598` | Code: comments, italic |

All code colours pass 5:1 on `--surface` in both themes. The prototype also has `--ph-bg` and `--ph-fg` for the yellow placeholder marker; that marker is for drafts only and must not appear on the live site.

In dark mode the header bar stays green (`--bar`), the pitch area becomes a lifted dark green clearly separated from the page background, and filled buttons switch to the light green `--link` with dark text.

### 3.2 Typography

| Role | Face | Weights | Size |
|------|------|---------|------|
| Text | IBM Plex Sans | 400, 400 italic, 500, 600, 700 | Body 16 px, line height 1.6 |
| Code, function names, version | IBM Plex Mono | 400, 500 | Code blocks 13.5 px, inline code 0.9 em |
| Fallback | `system-ui, -apple-system, "Segoe UI", sans-serif` and `ui-monospace, "SFMono-Regular", Consolas, monospace` | | |

Type scale (root 16 px): 13 px labels and captions, 14 px sidebar, tables and footer, 16 px body, 18 px lead text and function brief, 24 px section headings (`h2`), 32 px page titles (`h1`), landing headline `clamp(1.75rem, 1.2rem + 2.4vw, 2.75rem)` (28 to 44 px). Headings weight 600 (landing headline 700), line height 1.25. Uppercase labels (index headings, "DECLARATION", footer column titles) are 13 px, weight 600, letter spacing 0.06 em, colour `--muted`.

Font files: Latin subset from the `@fontsource/ibm-plex-sans` and `@fontsource/ibm-plex-mono` packages, version 5.2.5, 15 to 24 kB each, 152 kB for all seven. `font-display: swap`; the 400 Sans file is preloaded. The licence file comes from the IBM Plex repository and is published next to the fonts. The Latin subset has no Greek letters and no arrows (→); the few such characters in prose fall back to the system font, which is acceptable. Formulas are rendered by KaTeX with its own fonts.

### 3.3 Spacing, sizes, breakpoints

- Spacing scale (4 px base): 4, 8, 12, 16, 24, 32, 48, 64 px. Sibling groups use flex or grid with `gap`.
- Side gutter: 16 px below 600 px, 24 px from 600 px, 32 px from 960 px.
- Breakpoints: phone below 600 px; tablet 600 to 959 px; desktop from 960 px. Layouts are written mobile first; the smallest supported width is 360 px.
- Content width: at most 1200 px for the page frame; at most 760 px for running text and API entries.
- Header: bar 56 px; tab row 42 px (desktop only). The header is sticky. Anchor targets get `scroll-padding-top` of 72 px on phones and 122 px on desktop, so a jumped-to heading is not hidden under the header.
- Sidebar: 280 px wide, gap 48 px to the content.
- Radii: 4 px (buttons, inline code, small elements), 6 px (code blocks, tables, inputs), 10 px (cards, figures). Pills are fully rounded.
- Shadow: none on most elements; cards use a 1 px border. Dark mode uses no shadows.
- Touch targets: at least 44 px high for buttons, the menu button, the index toggle and menu entries.

## 4. Global layout

### 4.1 Header

- Green bar (`--bar`) with, from left: menu button (below 960 px only), logo (30 px, white) with the name "Quadriga-Lib", version in Plex Mono (hidden below 600 px), flexible space, "GitHub" link (hidden below 600 px; the menu has it).
- From 960 px a second row of tabs below the bar, in the same green: Home, Get started, C++ API, MATLAB API, Python API, Tutorials, Formats, Publications, AI agents, FAQ. The current page's tab is white, weight 600, with a 2 px white underline; the others are white at 88 % opacity. If the row does not fit, it scrolls sideways inside itself.
- Below 960 px the tab row is hidden. The menu button opens a full-height menu below the bar: a `<details>` element, so it works without JavaScript. Entries are grouped: Start (Home, Get started, Tutorials), API reference (C++, MATLAB, Python), More (Formats, Publications, AI agents, FAQ), Project (GitHub). The menu panel is positioned `absolute` against the sticky header (a `fixed` panel inside `<details>` was clipped in Chromium).
- A "Skip to content" link is the first focusable element.
- There is no search field in the header. Function search is the filter in the API index (section 7).

### 4.2 Footer

On `--surface` with a top border. Four columns from 600 px, two columns below: Documentation (Get started, C++ API, MATLAB / Octave API, Python API, Data formats), Learn (Tutorials, Publications, Questions, AI agents), Project (GitHub, Issues, How to cite, Apache-2.0 licence), For agents (`llms.txt`, "This page as Markdown", which links to the page's Markdown twin). Bottom line: "Quadriga-Lib v<version>", Impressum, Privacy. No build date (byte-identical builds).

### 4.3 Head of every page

`<html lang="en">`, title "<page> · Quadriga-Lib" (landing: "Quadriga-Lib"), meta description, `theme-color` for light and dark, Open Graph title, description and image, SVG favicon (the logo; the spec decides on a PNG fallback and the preview image `og-card.png`, which does not exist yet), `<link rel="alternate" type="text/markdown">` to the Markdown twin, one stylesheet, one deferred script, and an inline one-liner that sets `class="js"` on `<html>`.

## 5. Page templates

### 5.1 Landing (`index.html`)

Order of sections and their behaviour:

1. **Pitch** on `--pitch` across the full width. Headline (h1), lead text in `--muted`, two buttons (filled "Get started" to `download.html`, outlined "API reference" to the API section below), then three short proof lines with green dots: standards models, QuaDRiGa lineage, measured performance. Below 960 px the figure follows the text; from 960 px text and figure sit side by side (6 : 5). The figure is at most 520 px wide on tablets. On a 375 px phone the headline, lead, both buttons and the three proof lines fit in the first screen.
2. **Why Quadriga-Lib**: three cards (one column on phones, three from 960 px): standards models, QuaDRiGa lineage, measured performance. Only the maintainer's facts and numbers (D14).
3. **Paths**: two link cards "For engineers →" and "For team leads →" (one column on phones, two from 600 px). The section heading is visually hidden.
4. **Quick start**: `pip install quadriga-lib` in a code block, then the language tabs with the same example in Python, MATLAB / Octave and C++, with a note linking to the setup page. The antenna pattern figure sits next to it from 960 px (5 : 7), below it on smaller screens.
5. **API reference**: three link cards, one per language.
6. **What it covers**: eight features as title plus one line, no boxes; one column on phones, two from 600 px, four from 960 px. Acoustics carries the "experimental" badge (D12).
7. **Open source**: three link cards: GitHub, Publications, Use with AI agents.

### 5.2 API page (`cpp_api.html`, `mex_api.html`, `python_api.html`)

- **Page head**: breadcrumb (Home › API reference › C++), h1 "C++ API", lead with the version, a line "Same API in MATLAB / Octave · Python" and "Markdown version" (the full API Markdown file).
- **Index, desktop (from 960 px)**: sticky sidebar left of the content, scrolling on its own. From the top: the filter field (JavaScript only), "General usage notes", then one collapsible group per section (`<details>`, closed by default) with the function count; open, it lists "Section overview" and every function in Plex Mono 13 px. Below the groups: links to the other two languages and to the full Markdown file. With JavaScript, the function being read is highlighted (`--pitch` background, `--link` text) and its group opens.
- **Index, phones and tablets (below 960 px)**: the same list inside a collapsed box "Sections and functions" below the page head (`<details>`, 44 px high). The page therefore contains the list twice, once per layout; this is deliberate, because a single `<details>` cannot be open on desktop and closed on phones without JavaScript. The cost is about 3 kB compressed per API page.
- **Section**: 2 px top rule, h2 with the section title, the section description.
- **Function entry** (anchor = the function name exactly as today, e.g. `#generate_arrayant_3GPP`, `#.calc_directivity_dBi`): 1 px top rule; the name as h3 in Plex Mono 22 px with a hover "#" link; a row of pills "Also in: C++ · MATLAB · Python" (current language filled green, the others outlined links to the same function on the other page, a missing counterpart shown as a dashed "MATLAB –"), and a "Markdown" link at the right end to the section's Markdown file; the brief in 18 px; detail bullets; then sub-blocks with uppercase 13 px labels (Declaration, Inputs, Outputs, Returns, Example, See also, …).
- **Parameter lists** (Inputs, Outputs, Attributes, Arguments) are rendered as a two-column list: name in Plex Mono 14 px with "optional" in small muted text below it, description to the right (190 px name column from 600 px; stacked below 600 px). Nested tables (for example the `pol` modes) sit inside the description. Other lists (Returns, See also) stay ordinary bullet lists.
- **Code**: Declaration and Example blocks are code blocks (section 6), coloured for the page's language.
- The rule for matching functions across languages is for the spec (plan section 5.2); the prototype used exact name matches plus two hand-made pairs, so many pills show "–".

### 5.3 Text page (`formats.html`, `faq.html`, `download.html`, `ai_agents.html`)

Same frame as the API page: sidebar index "On this page" from 960 px, collapsed box below the page head on smaller screens. Content column at most 760 px. h2 headings get a top rule and 48 px space above; h3 for items within a section. Tables, code blocks, note boxes and formulas as in section 6. A short page without subsections (legal pages) uses the text column without an index.

### 5.4 Tutorial index (`tutorials.html`)

Breadcrumb, h1, lead, then cards in a grid (one column on phones, two from 600 px). Each card: tags for language (green outline) and level (grey outline), title (h3), one-line summary; the whole card is a link. While the tutorials are dummies, a warning note says so.

### 5.5 Tutorial page

Single text column (760 px), no sidebar. Breadcrumb, h1, tags (language, level, duration), download buttons for `.ipynb` and `.md`, then the notebook content: text, code cells and outputs. A code cell is a bordered box with a label "In [n]" in Plex Mono, the code, and an output area below a dashed line (text output or a figure). Formulas as in section 6. Ends with a link back to the index. The input format the notebook pipeline delivers is for the spec (plan 5.3).

### 5.6 Publications (`publications.html`)

Text column. Two groups with h2: "Quadriga-Lib" and "QuaDRiGa background", newest first. Each entry: year in Plex Mono green, title in 600 weight, authors and venue in muted 14 px, then link buttons (IEEE Xplore or IEEE Mentor, PDF, Markdown). Entries are separated by rules.

### 5.7 Legal pages (Impressum, privacy)

Text column without index. Content from the maintainer's text files (D17). Linked from every footer.

### 5.8 Not found (`404.html`)

Centred: the logo at 96 px in green, h1 "Page not found", one sentence, buttons to Home and to the API pages. Uses the normal header and footer, so all navigation is available.

## 6. Components

- **Buttons**: at least 44 px high, radius 4 px, weight 600. Filled: `--bar` background, white text (dark mode: `--link` background, dark text). Outlined: transparent, `--link` border and text.
- **Badge**: 11 px uppercase on `--warn-bg` / `--warn-fg`, used for "experimental".
- **Cards**: `--card` background, 1 px `--line` border, radius 10 px, 24 px padding. Link cards turn the border green on hover; the link label sits on top in green with "→".
- **Code block**: `--surface` background, 1 px border, radius 6 px, Plex Mono 13.5 px, line height 1.55, scrolls sideways inside itself. Colours from the `--c-*` tokens via Pygments CSS classes at build time. Tokens without a colour (whitespace, plain names, operators, punctuation) should not get a `<span>`: in the prototype they make up most of the markup, which makes the C++ page 1 MB raw (146 kB gzip, current page 125 kB).
- **Copy button** (JavaScript only): small "Copy" button in the top right corner of each code block, "Copied" for 1.5 s after a click, falls back to "Select and copy" if the clipboard is refused.
- **Language tabs**: three radio inputs with labels styled as tabs (Python, MATLAB / Octave, C++) and one panel each. Works without JavaScript; the first tab is selected by default. The selected tab is green with a 2 px underline. An optional note line sits below the panel.
- **Note box**: 4 px left border in `--link` on `--pitch` (information) or `--warn-fg` on `--warn-bg` (warning, placeholder), with a bold first line as title.
- **Table**: inside a bordered box with radius 6 px that scrolls sideways; header row on `--surface`, 14 px text, 1 px rules.
- **Formula**: block formulas centred, scroll sideways inside their box if too wide; inline formulas in the text flow. The site renders LaTeX with KaTeX at build time (D5). The prototype shows a static MathML sample instead (KaTeX needs Node, not yet installed).
- **Figure**: bordered frame, radius 10 px, fixed 4 : 3 aspect ratio, caption below in 13 px muted text.
- **Pills** (language links): 28 px high, fully rounded, 13 px.
- **Download link**: bordered box, 36 px high, label plus muted file type ("Download notebook .ipynb").
- **Focus**: every interactive element shows a 3 px `--focus` outline with 2 px offset on keyboard focus.

## 7. JavaScript extras

One small script (`js/site.js`, about 4 kB), loaded with `defer`. No third-party code. Without it nothing breaks or disappears; elements that only make sense with it carry the class `js-only` and are hidden until the inline one-liner sets `class="js"` on `<html>`.

| Extra | With JavaScript | Without JavaScript |
|-------|-----------------|--------------------|
| Copy buttons | Button on each code block | No button; code can be selected |
| Function filter | Field at the top of the API index; filters the function list as you type, Enter jumps to the first match, `/` focuses the field | Field hidden; the index lists all functions in collapsible groups |
| Current function | Sidebar highlights the function or section being read and opens its group | Groups open and close by click |
| Remembered language | After the visitor clicks a language tab, the choice is stored in `localStorage` (`qlib-lang`) and preselected on later visits (D18: only after a click) | Python is preselected |

A search across all three languages (one field that finds a function in any language) is not part of the design. If the spec adds it, it extends the filter with a small generated name index and stays optional.

## 8. Landing figures

Two figures, both made with the released pip package by scripts outside `make documentation` (D15), in a separate session before going live (section 9); the site is built with placeholders until then:

1. **Coverage map**: received power or path gain over a city block or an indoor floor, computed with `calc_diffraction_gain`. Hero figure in the pitch area.
2. **Antenna pattern**: pattern of a 3GPP panel array made with `arrayant.generate` (ideally the 2 × 2 array at 3.7 GHz from the quick-start example, or a larger panel). Placed next to the quick start.

Slot rules for both:

- Aspect ratio 4 : 3 (the frame crops nothing; the figure must fill it).
- Delivered as WebP, 1600 × 1200 px, at most 150 kB each; the scripts may also write PNG for other uses.
- One light and one dark variant per figure, embedded with `<picture>` and `<source media="(prefers-color-scheme: dark)">`. Backgrounds `#ffffff` and `#0b120e` (or transparent), axis and label text in `--fg` / `--muted` colours of the theme, rules in `--line`.
- Colour map: a sequential map that reads in both themes and does not depend on red/green distinction; the green accent `#2e7d46` / `#7fd197` may mark the antenna or transmitter.
- Text inside the figure at least 28 px at 1600 px width (readable at 375 px).
- File names `img/landing/coverage-light.webp`, `coverage-dark.webp`, `antenna-light.webp`, `antenna-dark.webp`; the scripts and their location in the repository are for the spec.
- Each figure gets an alt text that says what it shows, and a caption naming the function that made it.

The link preview image (`og-card.png`, 1200 × 630) can reuse the coverage map with the logo and name; it is made in the same session.

## 9. Gaps to settle before going live

The site is specified, built and reviewed with placeholders (steps 3 to 7). Missing content, missing figures and missing legal texts do not block any of these steps; they block only going live (plan section 7, "Going live"). The spec defines how a placeholder is marked in the sources and makes the build list every remaining placeholder, so that nothing is published by mistake.

The design session drafted texts only to fill the templates. They are not approved wording. The gaps below are filled with the maintainer in dedicated sessions after this project, before going live:

- **Content**: the texts in the table below.
- **Figures**: the two landing figures and the link-preview image (section 8). Until then the figure slots show a neutral placeholder of the same size.
- **Legal**: Impressum and privacy text from the maintainer's text files (D17). Until then both pages show a placeholder note.

| Where | Gap |
|-------|-----|
| Landing pitch | Headline and lead text (drafts in the prototype) |
| Landing pitch | Proof lines: measured performance numbers and hardware |
| Why Quadriga-Lib | Intro sentence; the three card texts: standards contribution, track record of QuaDRiGa, performance figures and how they were measured. No claims about users (D14). |
| Paths | Texts of the "For engineers" and "For team leads" cards |
| What it covers | Feature list and one-liners (taken from the current landing page; to be checked) |
| API section cards, open source cards | One-line descriptions |
| Figures | Both landing figures and the preview image (section 8) |
| AI agents page | Whole text, including the example prompt and the file table |
| Tutorials | Real tutorial list; the four cards are dummies (D11) |
| Download, FAQ | Existing sources; check against the new layout (headings for the index) |
| Impressum, privacy | Maintainer's text files (D17); going live is blocked until they are filled |
| 404 | One sentence |
| Meta descriptions | One per page |

## 10. Accessibility, print and other details

- Semantic structure: `header`, `nav` with labels (site sections, site menu, API index), `main` with `id="content"`, `footer`; one h1 per page; headings in order.
- Contrast: WCAG 2.2 AA for all text in both themes (section 3.1). The focus ring colour differs from the green so it stays visible on green.
- Keyboard: tabs, menu, index groups and language tabs are native elements (`a`, `details`/`summary`, radio inputs) and work with the keyboard.
- Reduced motion: the design has no animations apart from a border colour transition on cards.
- Print: header, footer, index, copy buttons, language pills and page tools are hidden; colours black on white; code wraps; function entries avoid page breaks inside.
- The prototype was checked in the built-in browser pane at 360, 375, 768 and 1280 px in light and dark mode: no page scrolls sideways, and the script logs no errors. The maintainer checked it on a phone.

## 11. Open points for the spec

1. Matching rule for the language pills (which C++ function corresponds to which MATLAB and Python function).
2. Section anchors: the prototype uses readable ids (`sec-array-antenna-functions`); the current pages use hash ids. Old section anchors must keep working (D3), for example with a second anchor per section.
3. Names and paths of the per-section Markdown files linked from each function ("Markdown") and from the footer ("This page as Markdown").
4. Pygments markup without spans for uncoloured tokens, and a size budget for the API pages (section 6).
5. Favicon PNG fallback and the preview image `og-card.png`.
6. Where the figure scripts live and how their images are committed (section 8).
7. The MATLAB and Python API pages use the API template unchanged; their language-specific labels (for example "MATLAB / Octave API", lower-case names) are taken from the current pages.
8. How placeholders for content, figures and legal texts are marked in the sources, how they look on the built pages, and the build's list of remaining placeholders (section 9).
