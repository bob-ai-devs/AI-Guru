# BOB AI Guru — Streamlit edition

A single-page Streamlit rewrite of the original Colab/Flask/ngrok notebook.
Generates a structured tutorial (index, chapters, trends, summary, keywords,
references) on any subject using the Gemini API, and lets you export it as
Markdown, PDF, or a zip of raw chapter files.

## ⚠️ Before you do anything else

The notebook you had contains a **live, plaintext Gemini API key**. Treat it
as compromised the moment it left your machine (including having pasted it
here):

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and **delete
   / regenerate** that key.
2. Put the *new* key only in Streamlit secrets (see below) — never in code,
   never in a notebook, never committed to GitHub.

## Project layout

```
app.py                          # the whole app
requirements.txt                # Python deps
packages.txt                    # system deps (wkhtmltopdf, for PDF export)
.streamlit/secrets.toml.example # copy to secrets.toml locally, fill in, do NOT commit
.gitignore                      # already excludes secrets.toml
```

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your real key
streamlit run app.py
```

PDF export needs the `wkhtmltopdf` binary on your PATH locally
(`apt install wkhtmltopdf` / `brew install wkhtmltopdf`). If it's missing,
the app just hides the PDF button and still offers Markdown + zip export.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (secrets.toml is gitignored — good).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   `app.py` in that repo.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-new-key"
   GEMINI_MODEL = "models/gemini-flash-lite-latest"
   ```
4. `packages.txt` tells Streamlit Cloud to `apt-get install wkhtmltopdf`
   automatically, so PDF export works out of the box there.

## What changed from the original notebook, and why

- **No Flask/ngrok/threading/sessions.** Streamlit already serves each
  visitor their own isolated script run; the old module-level globals
  (`thread_output_tutor`, `active_flag`, etc.) would have been *shared
  across every visitor* once this was deployed for more than one person —
  that's a real bug, not just unnecessary code. Everything now lives in
  `st.session_state`.
- **No files written to a shared working directory.** Chapter text, images,
  and the final PDF are all kept in memory per-session instead of on disk,
  so two people generating tutorials at the same time can't overwrite each
  other's files.
- **Secrets moved to `st.secrets`.** The API key and the Flask secret key
  are gone from the source entirely.
- **Removed `eval()` on scraped Bing data.** The original decoded a
  JS-object blob from Bing's HTML with `eval()`, which executes arbitrary
  code if that blob ever contains something unexpected. It's now parsed
  with a plain regex that only extracts the image URL.
- **Retries are now bounded.** The original had a couple of `while True`
  retry loops with no exit condition; both now retry a fixed number of
  times with exponential backoff and then surface an error instead of
  hanging forever.
- **PDF export degrades gracefully.** If `wkhtmltopdf` isn't installed, the
  app just hides that download option instead of crashing, and you still
  get Markdown + a zip of the raw chapters.
- **Added:** live progress bar and status log while generating, adjustable
  model name / image count / inter-call delay in the sidebar, a "start a
  new tutorial" reset button, and a Markdown export that has zero external
  binary dependencies.

## Ideas for further improvement

- Add per-chapter "regenerate this chapter" buttons.
- Cache generated tutorials (e.g. in `st.session_state` keyed by subject,
  or a small database) so re-visiting the same subject doesn't re-spend
  API calls.
- Replace the Bing scrape with a proper image API (Bing scraping can break
  any time Bing changes its markup, and may run against its terms of use).
