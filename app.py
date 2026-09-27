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
  * PDF generation is a single merged document rendered with the
    pure-Python `xhtml2pdf` (no external binary / apt dependency at all —
    the old `pdfkit` + `wkhtmltopdf` approach breaks on Streamlit Cloud's
    current base image since Debian dropped the `wkhtmltopdf` package),
    and it degrades gracefully to a Markdown/HTML download if the PDF
    library isn't installed for some reason.
  * Added: progress bar, live status log, adjustable model/section count,
    retry/backoff on API calls, a "start over" control, and a Markdown
    export that has no external binary dependency at all.
"""

import html as html_lib
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

# ==============================================================================
# BANK OF BARODA BRAND PALETTE
# ==============================================================================
BOB_ORANGE = "#F7941D"       # primary — "Baroda Sun"
BOB_ORANGE_DEEP = "#E8531B"  # sun-ray gradient end
BOB_MAROON = "#8E1B3A"       # sun-ray gradient end / accents
BOB_NAVY = "#12284C"         # wordmark / headings
BOB_NAVY_LIGHT = "#1E3E73"
BOB_CREAM = "#FFF8F1"        # page background
BOB_GREY = "#5B6675"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BOB_CREAM};
    }}
    #bob-banner {{
        background: radial-gradient(circle at 15% 50%, {BOB_ORANGE} 0%, {BOB_ORANGE_DEEP} 45%, {BOB_MAROON} 100%);
        padding: 22px 30px;
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }}
    #bob-banner h1 {{
        color: white;
        margin: 0;
        font-size: 1.9em;
        font-weight: 800;
        letter-spacing: 0.3px;
    }}
    #bob-banner p {{
        color: #FFEFE0;
        margin: 4px 0 0 0;
        font-size: 0.95em;
    }}
    h1, h2, h3 {{ color: {BOB_NAVY}; }}
    /* BOB AI Portal Card Border */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1.5px solid {BOB_ORANGE_DEEP} !important;
        border-radius: 14px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
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

DEFAULT_MODEL = "gemini-flash-lite-latest"


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
            f"following index for tutorial generation: {index_text}. No additional Commentary or information needed."
        )
    return (
        f"Please explain chapter number {chapter_num} in depth (detailed) for "
        f"{view} in an application-oriented (applied) way with the help of "
        "examples and tables (if any) so that the Subject Matter Expert can "
        "get in-depth insights of the chapter. Please use the following index "
        f"for tutorial generation: {index_text}. No additional Commentary or information needed."
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
# PDF export (pure Python — no system binary needed, unlike wkhtmltopdf)
# ----------------------------------------------------------------------------
# Streamlit Community Cloud's base image moved to Debian "trixie", which
# dropped the `wkhtmltopdf` apt package entirely (it's an unmaintained,
# ancient WebKit fork that Debian no longer ships) — so the previous
# pdfkit + packages.txt approach fails on deploy with
# "Package wkhtmltopdf is not available". xhtml2pdf is a pure-Python
# HTML->PDF renderer (built on reportlab) that needs nothing from apt at
# all, so `packages.txt` can go away completely.

try:
    from xhtml2pdf import pisa
except ImportError:  # pragma: no cover
    pisa = None


def pdf_available() -> bool:
    return pisa is not None


# def build_pdf(subject: str, sections: list[dict], image_urls: list[str]) -> bytes | None:
#     if not pdf_available():
#         return None

#     parts = [
#         "<html><head><meta charset='utf-8'>"
#         "<style>"
#         "body{font-family:Helvetica,sans-serif;margin:40px;}"
#         "h1{color:#333;}"
#         "h2{color:#0056b3;border-bottom:1px solid #ccc;}"
#         "table{border-collapse:collapse;width:100%;}"
#         "td,th{border:1px solid #999;padding:6px;}"
#         "img{max-width:100%;margin:8px 0;}"
#         ".section{page-break-before:always;}"
#         ".section:first-of-type{page-break-before:auto;}"
#         "</style></head><body>",
#         f"<h1>{subject.upper()}</h1>",
#         f"<p style='color:grey'>Generated by BOB AI Guru — {datetime.now():%Y-%m-%d %H:%M}</p>",
#     ]

#     for section in sections:
#         parts.append("<div class='section'>")
#         parts.append(f"<h2>{section['title']}</h2>")
#         parts.append(text_to_html(section["content"]))

#         if section.get("show_images") and image_urls:
#             for url in image_urls:
#                 parts.append(f"<img src='{url}' alt='illustration'/>")

#         parts.append("</div>")

#     parts.append(
#         f"<div style='text-align:center;font-size:12px;color:grey;'>"
#         f"{subject.upper()} — Generated by BOB AI Guru</div></body></html>"
#     )

#     html = "\n".join(parts)

#     buffer = io.BytesIO()
#     try:
#         result = pisa.CreatePDF(html, dest=buffer)
#         if result.err:
#             st.warning("PDF generation reported errors; use the Markdown download instead.")
#             return None
#         return buffer.getvalue()
#     except Exception as exc:  # noqa: BLE001
#         st.warning(f"PDF generation failed ({exc}); use the Markdown download instead.")
#         return None


def build_pdf(subject: str, sections: list[dict], image_urls: list[str]) -> bytes | None:
    if not pdf_available():
        return None

    parts = [
        "<style>"
            "body{font-family:Helvetica,sans-serif;margin:40px;}"
            "h1{color:#333;}"
            "h2{color:#0056b3;border-bottom:1px solid #ccc;}"
            
            "table{"
            "border-collapse:collapse;"
            "width:100%;"
            "margin:10px 0;"
            "}"
            
            "thead{display:table-header-group;}"
            
            "th{"
            "border:1px solid #999;"
            "padding:6px;"
            "background-color:#0059b3;"
            "color:#ffffff;"
            "font-weight:bold;"
            "text-align:left;"
            "}"
            
            "td{"
            "border:1px solid #999;"
            "padding:6px;"
            "color:#222222;"
            "}"
            
            "tr{page-break-inside:avoid;}"
            
            "img{max-width:100%;margin:8px 0;}"
            
            ".section{page-break-before:always;}"
            ".section:first-of-type{page-break-before:auto;}"
            
            "</style></head><body>",

        f"<h1>{subject.upper()}</h1>",
        f"<p style='color:grey'>Generated by BOB AI Guru — {datetime.now():%Y-%m-%d %H:%M}</p>",
    ]

    for section in sections:
        parts.append("<div class='section'>")
        parts.append(f"<h2>{section['title']}</h2>")
        parts.append(text_to_html(section["content"]))

        if section.get("show_images") and image_urls:
            for url in image_urls:
                parts.append(f"<img src='{url}' alt='illustration'/>")

        parts.append("</div>")

    parts.append(
        f"<div style='text-align:center;font-size:12px;color:grey;'>"
        f"{subject.upper()} — Generated by BOB AI Guru</div></body></html>"
    )

    html = "\n".join(parts)

    buffer = io.BytesIO()

    try:
        result = pisa.CreatePDF(html, dest=buffer)

        if result.err:
            st.warning(
                "PDF generation reported errors; use the Markdown download instead."
            )
            return None

        return buffer.getvalue()

    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"PDF generation failed ({exc}); use the Markdown download instead."
        )
        return None
 


def text_to_html(text: str) -> str:
    """
    Robust Gemini Markdown/HTML renderer for Streamlit's st.markdown().

    Used both for the on-page expanders (via
    `st.markdown(text_to_html(...), unsafe_allow_html=True)`) and for the
    PDF export, so both outputs render tables, code blocks, headings, etc.
    consistently instead of relying on st.markdown's plain CommonMark pass
    (which doesn't understand Gemini's occasional escaped/mixed HTML).

    Handles:
        - Normal Markdown
        - Gemini generated HTML
        - ESCAPED HTML such as \\<p> and \\<br/>
        - # / ## / ### headings
        - **bold**
        - *italic*
        - ***bold italic***
        - `inline code`
        - fenced ```code``` blocks
        - bullet lists
        - numbered lists
        - blockquotes
        - horizontal rules
        - Markdown tables
        - Markdown links
        - Gemini <b> / <span> / <strong> / <em> / <mark> formatting
        - Mixed Markdown + HTML

    IMPORTANT — why this version actually renders in Streamlit:
    st.markdown() runs everything through a CommonMark parser even with
    unsafe_allow_html=True. CommonMark treats any line indented 4+ spaces
    as a literal "indented code block" and prints it as raw text instead
    of interpreting it as HTML. Building elements with deeply indented
    triple-quoted f-strings would let chunks of the output get swallowed
    into code blocks or break the layout.

    This version collapses/strips all structural whitespace out of the
    final HTML (while fully preserving whitespace *inside* <pre><code>
    blocks) right before returning, so nothing in the output can ever be
    reinterpreted as an indented code block.
    """

    if not text:
        return ""

    # ============================================================
    # 1. NORMALIZE GEMINI ESCAPED HTML
    # ============================================================

    # Gemini sometimes returns:
    #
    # \<p style='...'>text\</p>
    # \<br/>
    #
    # Convert those back to real HTML.
    text = text.replace(r"\<", "<")
    text = text.replace(r"\>", ">")

    # Gemini's raw text frequently already contains HTML entities
    # (&amp; &lt; &gt; &quot; &#39; ...). Decode them ALL up front so our
    # own escaping later doesn't double-encode them into literal
    # "&amp;amp;"-style text in the final output.
    text = html_lib.unescape(text)

    # ============================================================
    # 1b. PROTECT GENUINE BLOCK-LEVEL HTML VERBATIM
    # ============================================================
    # Gemini sometimes hands back ALREADY-FORMED block HTML — a full
    # <table>...</table> with <thead>/<tbody>/<tr>/<td>, a real <ul>
    # with <li> children, a <blockquote>, a <pre> block, etc. — mixed in
    # with plain Markdown elsewhere in the same response. None of that
    # matches our line-by-line Markdown parser below, so without this
    # step it would fall through to the "plain paragraph" branch and
    # get HTML-escaped into visible tag soup.
    #
    # This pass finds any of those block containers (correctly handling
    # same-tag nesting, e.g. a <div> inside a <div>) and swaps the WHOLE
    # block for a placeholder token, so it is carried through untouched
    # and reinserted verbatim into the final HTML at the very end.

    protected_html = {}

    def _protect(raw_html, prefix):
        key = f"X{prefix}X{len(protected_html)}X"
        protected_html[key] = raw_html
        return key

    _BLOCK_TAGS = ("table", "ul", "ol", "blockquote", "pre", "dl")

    def _protect_block_html(raw_text):
        tag_pattern = re.compile(
            r"<(" + "|".join(_BLOCK_TAGS) + r")\b[^>]*>",
            re.IGNORECASE,
        )
        pieces = []
        pos = 0
        while True:
            m = tag_pattern.search(raw_text, pos)
            if not m:
                pieces.append(raw_text[pos:])
                break
            tag_name = m.group(1).lower()
            start = m.start()
            open_re = re.compile(rf"<{tag_name}\b[^>]*>", re.IGNORECASE)
            close_re = re.compile(rf"</{tag_name}\s*>", re.IGNORECASE)
            depth = 1
            cursor = m.end()
            end = len(raw_text)
            while cursor < len(raw_text):
                next_open = open_re.search(raw_text, cursor)
                next_close = close_re.search(raw_text, cursor)
                if not next_close:
                    end = len(raw_text)
                    break
                if next_open and next_open.start() < next_close.start():
                    depth += 1
                    cursor = next_open.end()
                else:
                    depth -= 1
                    cursor = next_close.end()
                    if depth == 0:
                        end = next_close.end()
                        break
            block_text = raw_text[start:end]
            pieces.append(raw_text[pos:start])
            pieces.append(_protect(block_text, "GEMBLOCK"))
            pos = end
        return "".join(pieces)

    text = _protect_block_html(text)

    # Real <h1>-<h6> tags (as opposed to our own "#" Markdown) get
    # converted into "# " Markdown syntax so they flow through our own
    # heading parser and pick up consistent styling. Inner inline tags
    # (e.g. a <b> inside the heading) are preserved and protected later.
    def _convert_real_headings(match):
        level = int(match.group(1))
        inner = match.group(2).strip()
        return "\n" + ("#" * level) + " " + inner + "\n"

    text = re.sub(
        r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>",
        _convert_real_headings,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # ============================================================
    # 2. REMOVE UNWANTED GEMINI WRAPPER HTML
    # ============================================================

    # If Gemini already generated <p style='margin:4px 0'>
    # we don't want nested paragraph tags inside our renderer.

    text = re.sub(r"<p\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)

    # Convert <br>, <br/>, <br /> to newline
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # <div> is used by Gemini purely as a generic paragraph wrapper
    # (table/ul/ol/blockquote/pre were already pulled out and protected
    # verbatim above, so this won't touch those).
    text = re.sub(r"</?div\b[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Gemini frequently wraps EACH table row in its own <p>...</p>, e.g.
    #   \<p>| Metric | Top | Lagging |</p>
    #
    #   \<p>| :--- | :--- | :--- |</p>
    #
    #   | **Primary Stock** | AAPL | XOM |
    # Once those <p> tags become newlines (above), the header and the
    # separator row end up with a blank line between them, which breaks
    # table detection (it requires them on consecutive lines). Collapse
    # any blank line(s) that sit between two pipe-containing rows.
    def _collapse_table_blank_lines(raw_text):
        src_lines = raw_text.split("\n")
        out_lines = []
        idx = 0
        while idx < len(src_lines):
            line = src_lines[idx]
            out_lines.append(line)
            if "|" in line.strip():
                look = idx + 1
                blanks = 0
                while look < len(src_lines) and not src_lines[look].strip():
                    blanks += 1
                    look += 1
                if (
                    blanks > 0
                    and look < len(src_lines)
                    and "|" in src_lines[look].strip()
                ):
                    idx = look
                    continue
            idx += 1
        return "\n".join(out_lines)

    text = _collapse_table_blank_lines(text)

    # ============================================================
    # 3. PROTECT INTENTIONAL GEMINI INLINE HTML
    # ============================================================

    # NOTE: these placeholder tokens deliberately contain ONLY letters
    # and digits — no underscores, asterisks, backticks, or brackets.
    # Tokens like "___INLINE_0___" would risk the bold/italic regexes
    # below (__text__, _text_) partially matching and corrupting those
    # underscore-heavy tokens before they could be restored, leaving
    # stray "INLINE0"-style fragments in the output.
    # (protected_html / _protect were already set up in step 1b above,
    # and are reused here for inline tags too.)

    # Paired inline tags, any attributes (href, style, class, etc.) —
    # not just the "style=" case.
    text = re.sub(
        r"</?(?:b|strong|i|em|u|s|del|mark|span|a|code|small|sub|sup)\b[^>]*>",
        lambda mo: _protect(mo.group(0), "GEMHTMLTAGX"),
        text,
        flags=re.IGNORECASE,
    )

    # Void/self-closing inline elements.
    text = re.sub(
        r"<(?:img|hr)\b[^>]*/?>",
        lambda mo: _protect(mo.group(0), "GEMHTMLTAGX"),
        text,
        flags=re.IGNORECASE,
    )

    # ============================================================
    # 4. INLINE MARKDOWN
    # ============================================================

    def inline_markdown(value):

        # Protect placeholders before HTML escaping
        placeholders = {}

        for key, html_tag in protected_html.items():
            placeholder = f"XINLINETOKENX{len(placeholders)}X"
            placeholders[placeholder] = html_tag
            value = value.replace(key, placeholder)

        # Escape everything else
        value = html_lib.escape(value)

        # --------------------------------------------------------
        # Markdown links
        # --------------------------------------------------------
        value = re.sub(
            r'\[([^\]]+)\]\((https?://[^\s\)]+)\)',
            r'<a href="\2" target="_blank" rel="noopener noreferrer" '
            r'style="color:#0059b3;text-decoration:none;font-weight:600;">'
            r'\1</a>',
            value,
        )

        # --------------------------------------------------------
        # Inline code
        # --------------------------------------------------------
        value = re.sub(
            r'`([^`]+)`',
            r'<code style="background:#f1f3f5;color:#7a1f1f;padding:2px 6px;'
            r'border-radius:5px;font-family:Consolas,monospace;font-size:0.90em;">'
            r'\1</code>',
            value,
        )

        # --------------------------------------------------------
        # Bold + italic (order matters: *** before ** before *)
        # --------------------------------------------------------
        value = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', value)
        value = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', value)
        value = re.sub(r'__(.+?)__', r'<strong>\1</strong>', value)
        value = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', value)
        value = re.sub(r'(?<!_)_([^_\n]+?)_(?!_)', r'<em>\1</em>', value)

        # --------------------------------------------------------
        # Strikethrough
        # --------------------------------------------------------
        value = re.sub(r'~~(.+?)~~', r'<del>\1</del>', value)

        # --------------------------------------------------------
        # Restore Gemini HTML
        # --------------------------------------------------------
        for placeholder, html_tag in placeholders.items():
            value = value.replace(placeholder, html_tag)

        return value

    # ============================================================
    # 5. TABLE FUNCTIONS
    # ============================================================

    def is_table_separator(line):
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        cells = stripped.split("|")
        if not cells:
            return False
        return all(re.match(r"^\s*:?-{3,}:?\s*$", cell) for cell in cells)

    def split_table_row(line):
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [cell.strip() for cell in line.split("|")]

    def render_table(table_lines):
        if len(table_lines) < 2:
            return None

        header = split_table_row(table_lines[0])
        separator = split_table_row(table_lines[1])

        if not is_table_separator(table_lines[1]):
            return None

        alignments = []
        for cell in separator:
            cell = cell.strip()
            if cell.startswith(":") and cell.endswith(":"):
                alignments.append("center")
            elif cell.endswith(":"):
                alignments.append("right")
            else:
                alignments.append("left")

        rows = []
        for line in table_lines[2:]:
            if not line.strip():
                continue
            if "|" not in line:
                continue
            cells = split_table_row(line)
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            elif len(cells) > len(header):
                cells = cells[:len(header)]
            rows.append(cells)

        parts = []
        parts.append(
            '<div style="width:100%;overflow-x:auto;margin:16px 0 20px 0;'
            'border:1px solid #d9dee7;border-radius:10px;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.06);">'
        )
        parts.append(
            '<table style="width:100%;border-collapse:collapse;'
            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;"
            'font-size:14px;background:#ffffff;">'
        )
        parts.append("<thead><tr>")

        for i, cell in enumerate(header):
            align = alignments[i] if i < len(alignments) else "left"
            parts.append(
                f'<th style="padding:11px 13px;text-align:{align};'
                "background:linear-gradient(135deg,#002e6e 0%,#0059b3 100%);"
                "color:#ffffff;font-weight:700;border-bottom:2px solid #f7941d;"
                f'white-space:nowrap;">{inline_markdown(cell)}</th>'
            )

        parts.append("</tr></thead><tbody>")

        for row_index, row in enumerate(rows):
            background = "#ffffff" if row_index % 2 == 0 else "#f6f8fb"
            parts.append("<tr>")
            for col_index, cell in enumerate(row):
                align = alignments[col_index] if col_index < len(alignments) else "left"
                parts.append(
                    f'<td style="padding:9px 13px;text-align:{align};'
                    f"background:{background};color:#202124;"
                    "border-bottom:1px solid #e5e7eb;vertical-align:middle;"
                    f'line-height:1.45;">{inline_markdown(cell)}</td>'
                )
            parts.append("</tr>")

        parts.append("</tbody></table></div>")

        return "".join(parts)

    # ============================================================
    # 6. MAIN PARSER
    # ============================================================

    lines = text.split("\n")
    output = []
    i = 0

    in_code = False
    code_lines = []
    code_lang = ""

    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            output.append("</ul>")
            in_ul = False
        if in_ol:
            output.append("</ol>")
            in_ol = False

    while i < len(lines):

        raw = lines[i]
        stripped = raw.strip()

        # ========================================================
        # CODE BLOCK
        # ========================================================
        if stripped.startswith("```"):

            if not in_code:
                close_lists()
                in_code = True
                code_lines = []
                code_lang = stripped[3:].strip().upper()
            else:
                code = html_lib.escape("\n".join(code_lines))
                label = code_lang if code_lang else "CODE"
                output.append(
                    '<div style="margin:14px 0;border-radius:10px;overflow:hidden;'
                    'background:#0d1117;border:1px solid #30363d;'
                    'box-shadow:0 2px 8px rgba(0,0,0,0.12);">'
                    '<div style="padding:6px 12px;background:#161b22;color:#8b949e;'
                    f'font-size:11px;font-weight:700;letter-spacing:0.5px;">{label}</div>'
                    '<pre style="margin:0;padding:14px;overflow-x:auto;color:#e6edf3;'
                    "font-family:Consolas,'Courier New',monospace;font-size:13px;"
                    f'line-height:1.55;"><code>{code}</code></pre></div>'
                )
                in_code = False
                code_lines = []
                code_lang = ""

            i += 1
            continue

        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        # ========================================================
        # BLANK LINE
        # ========================================================
        if not stripped:
            close_lists()
            i += 1
            continue

        # ========================================================
        # PROTECTED BLOCK-LEVEL HTML (already-formed <table>, <ul>,
        # <ol>, <blockquote>, <pre>, <dl> from Gemini) — output it
        # verbatim, not wrapped in a paragraph <div>.
        # ========================================================
        if re.match(r"^XGEMBLOCKX\d+X$", stripped):
            close_lists()
            output.append(inline_markdown(stripped))
            i += 1
            continue

        # ========================================================
        # TABLE
        # ========================================================
        if (
            i + 1 < len(lines)
            and "|" in stripped
            and is_table_separator(lines[i + 1])
        ):
            close_lists()

            table_lines = [lines[i], lines[i + 1]]
            j = i + 2

            while j < len(lines):
                candidate = lines[j].strip()
                if not candidate:
                    break
                if "|" not in candidate:
                    break
                table_lines.append(lines[j])
                j += 1

            rendered = render_table(table_lines)

            if rendered:
                output.append(rendered)
                i = j
                continue

        # ========================================================
        # HEADINGS
        # ========================================================
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)

        if heading:
            close_lists()

            level = len(heading.group(1))
            title = heading.group(2)

            if level == 1:
                style = (
                    "font-size:24px;color:#002e6e;border-bottom:3px solid #f7941d;"
                    "padding-bottom:8px;margin:18px 0 12px 0;"
                )
            elif level == 2:
                style = (
                    "font-size:20px;color:#002e6e;border-left:5px solid #f7941d;"
                    "padding-left:11px;margin:18px 0 10px 0;"
                )
            elif level == 3:
                style = "font-size:17px;color:#0059b3;margin:15px 0 8px 0;"
            else:
                style = "font-size:15px;color:#333333;margin:12px 0 6px 0;"

            output.append(
                f'<h{level} style="{style}font-weight:700;line-height:1.35;">'
                f"{inline_markdown(title)}</h{level}>"
            )

            i += 1
            continue

        # ========================================================
        # HORIZONTAL RULE
        # ========================================================
        if re.match(r"^([-*_])(?:\s*\1){2,}$", stripped):
            close_lists()
            output.append(
                '<div style="height:2px;margin:16px 0;background:'
                "linear-gradient(90deg,transparent,#d5dbe5,#f7941d,#d5dbe5,transparent);\"></div>"
            )
            i += 1
            continue

        # ========================================================
        # BLOCKQUOTE
        # ========================================================
        if stripped.startswith(">"):
            close_lists()
            quote = re.sub(r"^>\s?", "", stripped)
            output.append(
                '<div style="margin:10px 0;padding:11px 15px;border-left:4px solid #f7941d;'
                'background:#fff8ef;color:#4b5563;border-radius:0 8px 8px 0;line-height:1.55;">'
                f"{inline_markdown(quote)}</div>"
            )
            i += 1
            continue

        # ========================================================
        # BULLET
        # ========================================================
        bullet = re.match(r"^[-*+]\s+(.+)$", stripped)

        if bullet:
            if in_ol:
                output.append("</ol>")
                in_ol = False
            if not in_ul:
                output.append('<ul style="margin:7px 0 12px 24px;padding-left:15px;">')
                in_ul = True

            item = bullet.group(1)
            output.append(
                f'<li style="margin:5px 0;padding-left:3px;line-height:1.55;">'
                f"{inline_markdown(item)}</li>"
            )

            i += 1
            continue

        # ========================================================
        # NUMBERED LIST
        # ========================================================
        numbered = re.match(r"^\d+[.)]\s+(.+)$", stripped)

        if numbered:
            if in_ul:
                output.append("</ul>")
                in_ul = False
            if not in_ol:
                output.append('<ol style="margin:7px 0 12px 24px;padding-left:15px;">')
                in_ol = True

            item = numbered.group(1)
            output.append(
                f'<li style="margin:6px 0;padding-left:3px;line-height:1.55;">'
                f"{inline_markdown(item)}</li>"
            )

            i += 1
            continue

        # ========================================================
        # NORMAL PARAGRAPH
        # ========================================================
        close_lists()

        paragraph = inline_markdown(stripped)
        output.append(
            '<div style="margin:6px 0;color:#202124;font-size:14px;line-height:1.65;">'
            f"{paragraph}</div>"
        )

        i += 1

    # ============================================================
    # CLOSE ANY OPEN ELEMENTS
    # ============================================================
    if in_code:
        code = html_lib.escape("\n".join(code_lines))
        output.append(
            '<pre style="background:#0d1117;color:#e6edf3;padding:14px;'
            f'border-radius:8px;overflow-x:auto;">{code}</pre>'
        )

    close_lists()

    result = "\n".join(output)

    # ============================================================
    # FINAL RISK-LEVEL HIGHLIGHTING (plain-text occurrences only)
    # ============================================================
    result = re.sub(
        r"\bHIGH RISK\b",
        '<span style="display:inline-block;background:#fde8e8;color:#b42318;'
        'padding:3px 9px;border-radius:14px;font-weight:700;font-size:12px;">'
        "HIGH RISK</span>",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bMEDIUM RISK\b",
        '<span style="display:inline-block;background:#fff4d6;color:#9a6700;'
        'padding:3px 9px;border-radius:14px;font-weight:700;font-size:12px;">'
        "MEDIUM RISK</span>",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bLOW RISK\b",
        '<span style="display:inline-block;background:#e7f7ed;color:#18794e;'
        'padding:3px 9px;border-radius:14px;font-weight:700;font-size:12px;">'
        "LOW RISK</span>",
        result,
        flags=re.IGNORECASE,
    )

    # ============================================================
    # OUTER CONTAINER
    # ============================================================
    final_html = (
        '<div style="width:100%;box-sizing:border-box;'
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;"
        f'color:#202124;line-height:1.6;">{result}</div>'
    )

    # ============================================================
    # 7. STRIP STRUCTURAL WHITESPACE (the actual fix for st.markdown)
    # ============================================================
    # st.markdown() still runs a CommonMark pass even with
    # unsafe_allow_html=True. Any line starting with 4+ spaces is treated
    # as an "indented code block" and printed as literal text, which is
    # what breaks headings/tables/etc. We collapse all structural
    # newlines/indentation here, while fully preserving the exact
    # whitespace inside <pre>...</pre> code blocks.
    return _minify_preserve_pre(final_html)


def _minify_preserve_pre(html: str) -> str:
    """Strip line-leading whitespace and newlines from HTML so it can
    never be reinterpreted as a CommonMark indented code block, while
    leaving the contents of <pre>...</pre> blocks byte-for-byte intact.
    """

    pre_blocks = {}

    def protect(m):
        key = f"@@PRE_BLOCK_{len(pre_blocks)}@@"
        pre_blocks[key] = m.group(0)
        return key

    protected = re.sub(r"<pre\b.*?</pre>", protect, html, flags=re.DOTALL | re.IGNORECASE)

    lines = [line.strip() for line in protected.split("\n")]
    protected = "".join(lines)

    for key, block in pre_blocks.items():
        protected = protected.replace(key, block)

    return protected


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

# # ----------------------------------------------------------------------------
# # Header
# # ----------------------------------------------------------------------------

# st.title("📚 BOB AI Guru")
# st.caption("AI Tutor — generates a structured tutorial book on any subject you give it.")

# --------------------------------------------------------
# Header
# --------------------------------------------------------

st.markdown(
    """
    <div id="bob-banner">
        <h1>📚 BOB AI Guru</h1>
        <p>AI Tutor — generates a structured tutorial book on any subject you give it</p>
    </div>
    """,
    unsafe_allow_html=True,
)


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
            st.markdown(text_to_html(section["content"]), unsafe_allow_html=True)
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

    if pdf_available():
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
            "PDF export needs the `xhtml2pdf` package. Check requirements.txt. "
            "Use the Markdown download for now."
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
