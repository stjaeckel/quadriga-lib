# Website and documentation: open items

Open work around the new documentation (`docs_builder/`, `website_builder/`) that is not a defect in the code. How the builders work is described in `docs_builder/README.md` and `website_builder/README.md`; code defects go to GitHub issues. Remove an item when it is done; delete this file when the list is empty.

## Going live

The new site is built but not yet deployed; the server still serves the old site. `python3 website_builder/build.py website --release` refuses to build while any content gap remains, so the items below block the first deploy.

- **Content gaps.** `make website` prints the gap list: draft texts and placeholders marked `%%GAP: ...%%` in `docs_builder/content/`, `website_builder/content/` and `website_builder/templates/` (landing headline, lead, proof lines and cards with the measured performance numbers and hardware; meta descriptions and leads of the content pages; the AI agents page and its `llms_summary`; the Python "General usage notes" in `docs_builder/content/api_intro_python.md`; the tutorials index texts; the 404 sentence). Fill each text and remove its marker; removing the marker is the approval.
- **Impressum and privacy.** `website_builder/content/impressum.md` and `privacy.md` are placeholders. The Impressum needs the address once it is settled; the privacy text covers the hoster and its server logs, no cookies, no analytics, no third-party requests, and the browser storage key `qlib-lang` (set only after a visitor clicks a language tab).
- **Landing figures and preview image.** Two figures made with the released pip package: a coverage map (`calc_diffraction_gain`) and a 3GPP panel antenna pattern (`arrayant.generate`), each as `website_builder/static/img/landing/<name>-light.webp` and `<name>-dark.webp` (names `coverage`, `antenna`; slot rules in `website_builder/README.md`, "Landing figures"), plus the link-preview image `website_builder/static/img/og-card.png` (1200 × 630). The scripts that make them are written in the same session; decide where they live in the repository.
- **HTTPS at the hoster.** HTTPS does not work on quadriga-lib.org yet. Once it does: switch `SITE_BASE_URL` (and the download and releases URLs) in `docs_builder/qdocs/config.py` to `https://`, run `make docs` and commit `docs/`, and enable the commented-out redirect in `website_builder/static/.htaccess`.
- **Deploy script.** Adapt the local deploy script (outside the repository) to mirror `website/` as the deploy root; `download/` and `releases/` exist only on the server and must be left alone.
- **Checks after the first deploy.** The `curl` checks in `website_builder/README.md`, "Deployment": content types of `.md` and `llms.txt`, the 404 page, and the optional compression test. Check on GitHub that the anchors in `docs/` work (for example `docs/faq.md#cite` and a function anchor such as `#.is_valid`).
- **Phone check.** Look at the landing page on a real phone before going live.

## After the merge into `main`

- **Windows.** With the first `nmake release-win` on the `main` checkout: `python docs_builder\build.py docs --check` and `python -m pytest website_builder\tests\docs` must pass, and the zip must contain `docs\quadriga_lib_api_cpp.md`. The docs builder was tested on Linux and on Python 3.10 (Ubuntu 22.04 image) only.
- **Ubuntu 22.04 package.** The 22.04 deb build fails while linking the test binary (GitHub issue #17); it fails on `main` before the documentation changes too.

## Follow-ups

- **Agent skill.** Design the content of the agent skill with the maintainer (what an agent needs, recipes, conventions, how it is tested and versioned) and replace the stub `website_builder/skill/quadriga-lib/SKILL.md`; the AI agents page then drops its placeholder box.
- **Tutorials.** Write the real tutorial notebooks in `tutorials/` (committed with outputs; format in `website_builder/README.md`, "Tutorials") and replace the dummy `pipeline_demo.ipynb`. Running and refreshing the notebooks before a release is its own small project.
- **Releases on GitHub.** Set up GitHub Releases (tags, release notes, assets, a make target that prepares the release command), fill in older versions (some have no git tag, for example v0.12.4), then switch `DOWNLOAD_BASE_URL` and `RELEASES_URL` in `docs_builder/qdocs/config.py`.
- **Documentation indexes for agents.** Register the library with documentation indexes such as Context7.
- **Development docs.** Decide after going live whether a GitHub Action builds the site on every push to `main` and publishes it on GitHub Pages, labelled as development documentation, while quadriga-lib.org keeps the release documentation.
- **Parameter-line warnings.** The docs build reports 42 warnings (`param-item`, `param-separator`) for parameter lines that do not follow the documented pattern; they render correctly. Fix them when the blocks are touched anyway.
- **Stylesheet caching.** Optional: add the library version to the stylesheet and script links in the templates (for example `site.css?v=<version>`), so browsers load the new files after a release.
