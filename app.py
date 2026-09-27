"""
BOB AI Guru — Streamlit edition
--------------------------------
Converted from a Colab/Flask/ngrok notebook to a single-page Streamlit app
that can be deployed directly on Streamlit Community Cloud.

Key changes vs. the original notebook:
  * No Flask, no ngrok, no threading, no server-side sessions/cookies.
    Streamlit already gives each visitor an isolated session — we just use
    `st.session_state` instead of module-level globals (the old globals
    would have been SHARED across every visitor, which is a real bug once
    this is hosted for more than one person at a time).
  * The Gemini API key (and everything else secret) now lives in
    `st.secrets`, never in the source. NOTE: the notebook you uploaded had
    a live-looking Gemini API key hardcoded in plain text. Treat that key
    as compromised — revoke/regenerate it in Google AI Studio and put the
    new one only in Streamlit secrets, never back in code.
  * Uses the newer `google-genai` SDK (`from google import genai`) instead
    of the older `google-generativeai` package.
  * No files are written to a shared working directory on disk. Everything
    (chapter text, images, the final PDF) is kept in memory in
    `st.session_state`, so concurrent users on the same deployment can't
    stomp on each other's output.
  * `eval()` on data scraped from Bing was replaced with a safe regex
    extraction — `eval()` on untrusted web content is a code-execution
    hole and should never be used.
  * PDF generation is a single merged document instead of per-chapter
    files stitched together, and it degrades gracefully (offers a
    Markdown/HTML download instead) if wkhtmltopdf isn't installed.
  * Added: progress bar, live status log, adjustable model/section count,
    retry/backoff on API calls, a "start over" control, and a Markdown
    export that has no external binary dependency at all.
"""

import io
import json
import random
import re
import time
import zipfile
from datetime import datetime

import requests
import streamlit as st

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="BOB AI Guru",
    page_icon="📚",
    layout="wide",
)

BING_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

# Only letters, numbers, spaces, and a small set of punctuation — mirrors the
# original client-side validation, enforced again here since Streamlit has
# no separate client/server boundary to rely on.
SUBJECT_PATTERN = re.compile(r"^[a-zA-Z0-9.#\-\s&]+$")

DEFAULT_MODEL = "models/gemini-flash-lite-latest"


# ----------------------------------------------------------------------------
# Secrets / configuration
# ----------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Read the Gemini key from Streamlit secrets — never from source."""
    return st.secrets.get("GEMINI_API_KEY")


class GeminiModel:
    """Thin wrapper around the `google-genai` client so the rest of the app
    can keep calling `model.generate_content(prompt)` exactly like it did
    with the old `google-generativeai` SDK."""

    def __init__(self, client: "genai.Client", model_name: str):
        self._client = client
        self._model_name = model_name

    def generate_content(self, prompt: str):
        return self._client.models.generate_content(model=self._model_name, contents=prompt)


def configure_genai(model_name: str) -> GeminiModel:
    api_key = get_api_key()
    if not api_key:
        st.error(
            "No Gemini API key found. Add one under **Settings → Secrets** "
            "(or in `.streamlit/secrets.toml` locally) as:\n\n"
            '```toml\nGEMINI_API_KEY = "your-key-here"\n```'
        )
        st.stop()
    if genai is None:
        st.error("The `google-genai` package isn't installed. Check requirements.txt.")
        st.stop()
    client = genai.Client(api_key=api_key)
    return GeminiModel(client, model_name)


# ----------------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------------

def generate_with_retry(model, prompt: str, max_retries: int = 4, base_delay: float = 3.0) -> str:
    """Call the model with exponential backoff instead of the original's
    unbounded `while True` retry loop (which could spin forever on a
    persistent error and never surface anything to the user)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            last_err = exc
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


def extract_chapter_count(model, index_text: str) -> int:
    """Ask the model how many chapters are in the index it just wrote, with
    a bounded number of attempts (the original looped forever until it saw
    a digit)."""
    prompt = (
        'Based on the given index tell me "How many chapters are there in the '
        'above tutorial", respond with only an integer value: ' + index_text
    )
    for _ in range(5):
        answer = generate_with_retry(model, prompt).strip()
        digits = re.sub(r"[^0-9]", "", answer)
        if digits:
            return int(digits)
    # Fall back to counting numbered lines in the index itself.
    matches = re.findall(r"^\s*\d+[\.\)]", index_text, flags=re.MULTILINE)
    return max(len(matches), 1)


def build_prompts(view: str, subject: str):
    is_overview = "overview" in view.lower()
    if is_overview:
        index_prompt = (
            "Hello, I am about to take a Project Manager role in " + subject +
            " in a company, therefore I would like you to generate a brief "
            "tutorial book for " + subject + " which can give me an overview "
            "of this subject so that I am able to understand and implement "
            "the same in projects. Please start with creating an Index having "
            "less than 5 Chapters along with the numbering of chapters, kindly "
            "use heading as 'INDEX' and do not use the word 'brief' in the "
            "INDEX or tutorial."
        )
    else:
        index_prompt = (
            "Hello, I am about to take a " + view + " role in " + subject +
            " in a company, therefore I would like you to generate an in-depth "
            "(detailed) tutorial book for " + subject + " which can make me "
            "proficient in this subject so that I am able to understand and "
            "implement the same in projects. Please start with creating an "
            "Index having less than 5 Chapters along with the numbering of "
            "chapters, kindly use heading as 'INDEX' and do not use the words "
            "'in depth' or 'detailed' in the INDEX or tutorial."
        )
    return index_prompt, is_overview


def chapter_prompt(view: str, chapter_num: int, index_text: str, is_overview: bool) -> str:
    if is_overview:
        return (
            f"Please explain chapter number {chapter_num} for {view} in a format "
            "with the help of examples, kindly use tables if applicable so that "
            "the end user can understand the information easily. Please use the "
            f"following index for tutorial generation: {index_text}"
        )
    return (
        f"Please explain chapter number {chapter_num} in depth (detailed) for "
        f"{view} in an application-oriented (applied) way with the help of "
        "examples and tables (if any) so that the Subject Matter Expert can "
        "get in-depth insights of the chapter. Please use the following index "
        f"for tutorial generation: {index_text}"
    )


# ----------------------------------------------------------------------------
# Image search (Bing) — safe JSON-ish extraction, no eval(), no disk writes
# ----------------------------------------------------------------------------

def fetch_image_urls(search_query: str, num_images: int = 6) -> list[str]:
    urls: list[str] = []
    try:
        search_url = f"https://www.bing.com/images/search?q={requests.utils.quote(search_query)}"
        resp = requests.get(search_url, headers=BING_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.warning(f"Image search skipped ({exc}).")
        return urls

    # Bing embeds a small JSON blob per result in the `m="..."` attribute.
    # Extract just the `murl` (media URL) field with a regex instead of
    # eval()-ing arbitrary scraped content.
    for match in re.finditer(r'"murl":"(https?://[^"]+)"', resp.text):
        url = match.group(1).encode().decode("unicode_escape")
        if url not in urls:
            urls.append(url)
        if len(urls) >= num_images:
            break
    return urls


# ----------------------------------------------------------------------------
# PDF export (optional — degrades gracefully if wkhtmltopdf isn't present)
# ----------------------------------------------------------------------------

def pdfkit_available() -> bool:
    try:
        import pdfkit  # noqa: F401
        import shutil as _shutil
        return _shutil.which("wkhtmltopdf") is not None
    except ImportError:
        return False


def build_pdf(subject: str, sections: list[dict], image_urls: list[str]) -> bytes | None:
    if not pdfkit_available():
        return None
    import pdfkit

    parts = [
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:Verdana,sans-serif;margin:40px;}"
        "h1{color:#333;} h2{color:#0056b3;border-bottom:1px solid #ccc;}"
        "table{border-collapse:collapse;width:100%;} "
        "td,th{border:1px solid #999;padding:6px;} "
        "img{max-width:100%;margin:8px 0;}</style></head><body>",
        f"<h1>{subject.upper()}</h1>",
        f"<p style='color:grey'>Generated by BOB AI Guru — {datetime.now():%Y-%m-%d %H:%M}</p>",
    ]
    for section in sections:
        parts.append(f"<h2>{section['title']}</h2>")
        parts.append(_markdown_to_basic_html(section["content"]))
        if section.get("show_images") and image_urls:
            for url in image_urls:
                parts.append(f"<img src='{url}' alt='illustration'/>")
    parts.append(
        f"<div style='text-align:center;font-size:12px;color:grey;'>"
        f"{subject.upper()} — Generated by BOB AI Guru</div></body></html>"
    )
    html = "\n".join(parts)

    options = {"page-size": "A4", "encoding": "UTF-8", "no-outline": None, "quiet": ""}
    try:
        pdf_bytes = pdfkit.from_string(html, False, options=options)
        return pdf_bytes
    except Exception as exc:  # noqa: BLE001
        st.warning(f"PDF generation failed ({exc}); use the Markdown download instead.")
        return None


def _markdown_to_basic_html(text: str) -> str:
    """Very small, safe markdown->HTML pass for the PDF only (the on-page
    view just uses st.markdown, which already renders this properly)."""
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.*?)`", r"<em>\1</em>", text)
    paragraphs = [f"<p>{line.strip()}</p>" for line in text.split("\n") if line.strip()]
    return "\n".join(paragraphs)


# ----------------------------------------------------------------------------
# Markdown export (no external binaries needed at all)
# ----------------------------------------------------------------------------

def build_markdown(subject: str, sections: list[dict], image_urls: list[str]) -> str:
    lines = [f"# {subject.upper()}", ""]
    for section in sections:
        lines.append(f"## {section['title']}")
        lines.append(section["content"])
        if section.get("show_images") and image_urls:
            lines.append("")
            for url in image_urls:
                lines.append(f"![illustration]({url})")
        lines.append("")
    lines.append(f"*Generated by BOB AI Guru — {datetime.now():%Y-%m-%d %H:%M}*")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

defaults = {
    "sections": None,
    "subject": "",
    "view": "Overview",
    "image_urls": [],
    "pdf_bytes": None,
    "generating": False,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# ----------------------------------------------------------------------------
# Sidebar — settings
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input(
        "Gemini model",
        value=st.secrets.get("GEMINI_MODEL", DEFAULT_MODEL),
        help="Any model string your API key has access to, e.g. "
        "`models/gemini-flash-lite-latest` or `models/gemini-1.5-pro`.",
    )
    include_images = st.checkbox("Fetch illustrative images", value=True)
    num_images = st.slider("Number of images", 0, 12, 6, disabled=not include_images)
    call_delay = st.slider(
        "Delay between API calls (seconds)",
        0.0, 10.0, 3.0, step=0.5,
        help="A small delay helps avoid hitting per-minute rate limits.",
    )
    st.divider()
    if st.button("🔄 Start a new tutorial", use_container_width=True):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.title("📚 BOB AI Guru")
st.caption("AI Tutor — generates a structured tutorial book on any subject you give it.")

# ----------------------------------------------------------------------------
# Input form (shown until a tutorial has been generated)
# ----------------------------------------------------------------------------

if not st.session_state.sections:
    with st.form("subject_form"):
        view = st.radio(
            "Select Learning Type",
            ["Overview", "Subject Matter Expert - Detailed View"],
            index=0,
            help="Overview = Leadership View, less than 5 chapters, high level.\n\n"
            "Detailed = Subject Matter Expert view, in-depth with examples.",
        )
        subject = st.text_input(
            "Subject",
            placeholder="e.g. Python Programming, Quantum Computing, Climate Risk, Digital Banking",
        )
        submitted = st.form_submit_button("Generate tutorial", type="primary")

    if submitted:
        subject = (subject or "").strip()
        if not subject:
            st.error("Please enter a subject.")
        elif not SUBJECT_PATTERN.match(subject):
            st.error(
                "Please enter a single subject using only letters, numbers, "
                "spaces, and `. # - &` — no other punctuation."
            )
        else:
            st.session_state.subject = subject
            st.session_state.view = view
            st.session_state.generating = True
            st.rerun()

# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

if st.session_state.generating:
    subject = st.session_state.subject
    view = st.session_state.view
    model = configure_genai(model_name)
    sections: list[dict] = []
    image_urls: list[str] = []

    progress = st.progress(0.0)
    with st.status(f"Writing your tutorial on **{subject}**…", expanded=True) as status:
        try:
            index_prompt, is_overview = build_prompts(view, subject.upper())
            status.write("Drafting the index…")
            index_text = generate_with_retry(model, index_prompt)
            sections.append({"title": "Index", "content": index_text})
            time.sleep(call_delay)

            status.write("Counting chapters…")
            chapter_count = extract_chapter_count(model, index_text)
            total_steps = chapter_count + 4  # chapters + trends + summary + keywords + refs
            progress.progress(1 / total_steps)

            for chapter_num in range(1, chapter_count + 1):
                status.write(f"Writing chapter {chapter_num}/{chapter_count}…")
                answer = generate_with_retry(
                    model, chapter_prompt(view, chapter_num, index_text, is_overview)
                )
                first_line = answer.split("\n", 1)[0]
                title = re.sub(r"[^\w\s.,;:\-]", "", first_line).strip() or f"Chapter {chapter_num}"
                sections.append({"title": title.upper(), "content": answer})
                progress.progress((chapter_num + 1) / total_steps)
                time.sleep(call_delay)

            status.write("Writing upcoming-trends section…")
            trends = generate_with_retry(
                model,
                "Please give a suitable header such as 'Upcoming Trends' and create "
                f"a table of upcoming trends for the following subject: {index_text}",
            )
            sections.append({"title": "Upcoming Trends", "content": trends})
            progress.progress((chapter_count + 2) / total_steps)
            time.sleep(call_delay)

            status.write("Writing summary…")
            summary = generate_with_retry(
                model,
                "Please give a suitable header such as 'Summary' and provide a brief "
                "summary of the whole subject in easy language using bullet points "
                f"from the following index: {index_text}",
            )
            sections.append({"title": "Summary", "content": summary})
            progress.progress((chapter_count + 3) / total_steps)
            time.sleep(call_delay)

            status.write("Collecting keywords" + (" and images…" if include_images else "…"))
            keywords = generate_with_retry(
                model,
                "Please give a suitable header such as 'Keywords & Diagrams' and "
                f"list the relevant keywords from the following subject: {index_text}",
            )
            if include_images and num_images > 0:
                image_urls = fetch_image_urls(f"Evolution of {subject}", num_images)
            sections.append({
                "title": "Keywords & Diagrams",
                "content": keywords,
                "show_images": bool(image_urls),
            })
            progress.progress((chapter_count + 4) / total_steps)
            time.sleep(call_delay)

            status.write("Gathering reference links…")
            references = generate_with_retry(
                model,
                "Please give a suitable header such as 'References' and provide "
                f"relevant links for further reading from the following index: {index_text}",
            )
            sections.append({"title": "References", "content": references})
            progress.progress(1.0)

            status.update(label="Tutorial ready!", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Generation failed", state="error")
            st.error(f"Something went wrong: {exc}")
            st.session_state.generating = False
            st.stop()

    st.session_state.sections = sections
    st.session_state.image_urls = image_urls
    st.session_state.pdf_bytes = None  # built lazily below
    st.session_state.generating = False
    st.rerun()

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------

if st.session_state.sections:
    subject = st.session_state.subject
    sections = st.session_state.sections
    image_urls = st.session_state.image_urls

    st.subheader(f"Tutorial: {subject.upper()}")
    st.caption(f"Learning type: {st.session_state.view}")

    for section in sections:
        with st.expander(section["title"], expanded=(section["title"] == "Index")):
            st.markdown(section["content"])
            if section.get("show_images") and image_urls:
                st.image(image_urls, width=220)

    st.divider()
    col1, col2, col3 = st.columns(3)

    markdown_doc = build_markdown(subject, sections, image_urls)
    col1.download_button(
        "⬇️ Download as Markdown",
        data=markdown_doc,
        file_name=f"{subject.replace(' ', '_')}_tutorial.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if pdfkit_available():
        if st.session_state.pdf_bytes is None:
            with st.spinner("Building PDF…"):
                st.session_state.pdf_bytes = build_pdf(subject, sections, image_urls)
        if st.session_state.pdf_bytes:
            col2.download_button(
                "⬇️ Download as PDF",
                data=st.session_state.pdf_bytes,
                file_name=f"{subject.replace(' ', '_')}_tutorial.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        col2.caption(
            "PDF export needs `wkhtmltopdf` installed on the server "
            "(see `packages.txt`). Use the Markdown download for now."
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, section in enumerate(sections, start=1):
            zf.writestr(f"{i:02d}_{section['title'][:40]}.txt", section["content"])
    col3.download_button(
        "⬇️ Download raw chapters (.zip)",
        data=zip_buffer.getvalue(),
        file_name=f"{subject.replace(' ', '_')}_chapters.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.markdown("---")
st.caption("BOB AI Guru uses Generative AI for its results. Kindly double-check its responses.")
