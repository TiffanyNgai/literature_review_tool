import json
import os

# Load local environment variables from .env (kept out of git via .gitignore)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, we just rely on existing environment variables.
    pass

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(
    page_title="LitSearch Prototype",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",  # <-- minimal startup; sidebar can be opened if desired
)

# Hard-coded bibliography DB path (prototype)
DB_PATH = Path(__file__).with_name("example-bib.json")  # repo-relative (same folder as this app)
# Gemini API key lookup order:
#   1) st.secrets["GEMINI_API_KEY"] (optional)
#   2) environment variable GEMINI_API_KEY (can be populated from a local .env via python-dotenv)
# NOTE: never hard-code keys in this repo.
def get_gemini_key() -> Optional[str]:
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key


# =========================
# Utilities
# =========================
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> List[str]:
    s = _norm(s)
    toks = re.split(r"[^a-z0-9]+", s)
    return [t for t in toks if t]

def _safe_join(x: Any, sep: str = ", ") -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return sep.join(str(v) for v in x if v is not None)
    return str(x)

def _year_int(y: Any) -> Optional[int]:
    try:
        if y is None:
            return None
        return int(y)
    except Exception:
        return None

def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _bibtex_key(authors: List[str], year: Any, title: str) -> str:
    fa = authors[0] if authors else "Unknown"
    last = fa.split()[-1] if fa else "Unknown"
    y = str(year) if year is not None else "n.d."
    fw = (_tokenize(title)[:1] or ["Work"])[0].capitalize()
    return f"{last}{y}{fw}"

def _as_bibtex(entry: Dict[str, Any]) -> str:
    authors = entry.get("authors", [])
    year = entry.get("year", "")
    title = entry.get("title", "")
    journal = entry.get("journal", "")
    volume = entry.get("volume", "")
    issue = entry.get("issue", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")
    keywords = entry.get("keywords", [])
    key = _bibtex_key(authors, year, title)

    fields = {
        "author": " and ".join(authors) if isinstance(authors, list) else str(authors),
        "title": title,
        "journal": journal,
        "year": year,
        "volume": volume,
        "number": issue,
        "pages": pages,
        "doi": doi,
        "keywords": _safe_join(keywords, sep=", "),
    }
    fields = {k: v for k, v in fields.items() if str(v).strip() != ""}

    lines = [f"@article{{{key},"]
    for k, v in fields.items():
        v = str(v).replace("{", "\\{").replace("}", "\\}")
        lines.append(f"  {k} = {{{v}}},")
    lines.append("}")
    return "\n".join(lines)

def _as_ris(entry: Dict[str, Any]) -> str:
    authors = entry.get("authors", []) or []
    year = entry.get("year", "")
    title = entry.get("title", "")
    journal = entry.get("journal", "")
    volume = entry.get("volume", "")
    issue = entry.get("issue", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")
    abstract = entry.get("abstract", "")
    keywords = entry.get("keywords", []) or []

    lines = ["TY  - JOUR"]
    for a in authors:
        lines.append(f"AU  - {a}")
    if title:
        lines.append(f"TI  - {title}")
    if journal:
        lines.append(f"JO  - {journal}")
    if year:
        lines.append(f"PY  - {year}")
    if volume:
        lines.append(f"VL  - {volume}")
    if issue:
        lines.append(f"IS  - {issue}")
    if pages:
        lines.append(f"SP  - {pages}")
    if doi:
        lines.append(f"DO  - {doi}")
    if abstract:
        lines.append(f"AB  - {abstract}")
    for kw in keywords:
        lines.append(f"KW  - {kw}")
    lines.append("ER  - ")
    return "\n".join(lines)

def _score_entry(entry: Dict[str, Any], query: str, fields: List[str]) -> Tuple[float, Dict[str, float]]:
    q = _norm(query)
    if not q:
        return 0.0, {}
    q_toks = set(_tokenize(q))
    if not q_toks:
        return 0.0, {}

    per_field: Dict[str, float] = {}
    score = 0.0

    for f in fields:
        raw = entry.get(f, "")
        txt = _safe_join(raw)
        t = _norm(txt)
        if not t:
            continue

        toks = set(_tokenize(t))
        overlap = len(q_toks & toks)
        denom = max(3, len(q_toks))
        tok_score = overlap / denom
        exact_bonus = 0.7 if q in t else 0.0

        w = {
            "title": 2.0,
            "abstract": 1.2,
            "keywords": 1.4,
            "authors": 1.0,
            "journal": 0.7,
            "doi": 0.7,
        }.get(f, 1.0)

        fs = w * (tok_score + exact_bonus)
        per_field[f] = fs
        score += fs

    return score, per_field


@st.cache_data(ttl=3600, show_spinner=False)
def load_db() -> List[Dict[str, Any]]:
    data = json.loads(DB_PATH.read_text(encoding="utf-8"))
    refs = data.get("references", [])
    for r in refs:
        r.setdefault("id", None)
        r.setdefault("authors", [])
        r.setdefault("keywords", [])
        r.setdefault("abstract", "")
        r.setdefault("title", "")
        r.setdefault("journal", "")
        r.setdefault("doi", "")
        r.setdefault("year", None)
    return refs


def init_state():
    if "selected_ids" not in st.session_state:
        st.session_state.selected_ids = set()
    if "notes" not in st.session_state:
        st.session_state.notes = {}  # id -> str
    if "tags" not in st.session_state:
        st.session_state.tags = {}  # id -> list[str]
    if "mode" not in st.session_state:
        st.session_state.mode = "Normal"
    if "ai_query" not in st.session_state:
        st.session_state.ai_query = ""
    if "ai_selected_ids" not in st.session_state:
        st.session_state.ai_selected_ids = None  # None means "no AI filter applied"


def entry_to_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": entry.get("id"),
        "year": entry.get("year"),
        "title": entry.get("title", ""),
        "authors": _safe_join(entry.get("authors", []) or []),
        "journal": entry.get("journal", ""),
        "doi": entry.get("doi", ""),
        "keywords": _safe_join(entry.get("keywords", []) or []),
    }


@st.cache_resource(show_spinner=False)
def gemini_client(api_key: str):
    # New Google Gen AI SDK:
    #   pip install --upgrade google-genai
    #   from google import genai
    #   client = genai.Client(api_key=api_key)
    from google import genai  # type: ignore
    return genai.Client(api_key=api_key)


def ai_find_relevant_ids(refs: List[Dict[str, Any]], user_request: str, api_key: str) -> List[Any]:
    """
    Uses Gemini to select relevant paper IDs from the local library.
    Returns a list of ids (as they appear in the JSON). If none, returns [].
    """
    client = gemini_client(api_key)

    # Keep context compact: provide id, title, year, keywords, and a trimmed abstract
    items = []
    for r in refs:
        items.append({
            "id": r.get("id"),
            "year": r.get("year"),
            "title": r.get("title", ""),
            "keywords": r.get("keywords", []) or [],
            "abstract": (r.get("abstract", "") or "")[:600],
        })

    system = (
        "You are helping a user find relevant literature review candidates from a SMALL local library.\n"
        "You must ONLY select from the provided items.\n"
        "Return STRICT JSON with exactly this schema:\n"
        '{ "selected_ids": [<id>, <id>, ...], "reasoning": "<brief>" }\n'
        "If nothing matches, return: {\"selected_ids\": [], \"reasoning\": \"No relevant items in the library.\"}\n"
        "Do not include any extra keys. Do not wrap in markdown."
    )

    prompt = f"{system}\n\nUSER_REQUEST:\n{user_request}\n\nLIBRARY_ITEMS:\n{json.dumps(items, ensure_ascii=False)}\n"

    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    text = (getattr(resp, "text", None) or "").strip()

    # Parse JSON robustly (Gemini usually complies, but prototype should handle mild drift)
    try:
        obj = json.loads(text)
        ids = obj.get("selected_ids", [])
        if not isinstance(ids, list):
            return []
        return ids
    except Exception:
        # Attempt to extract the first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
            ids = obj.get("selected_ids", [])
            return ids if isinstance(ids, list) else []
        except Exception:
            return []


# =========================
# App
# =========================
init_state()

# --- Minimal home header (startup screen is intentionally light) ---
st.title("📚 LitSearch")
st.caption("Prototype: search + shortlist from a tiny local bibliography database.")

# --- Mode toggle on the starting page ---
mode = st.toggle("AI mode", value=(st.session_state.mode == "AI"))
st.session_state.mode = "AI" if mode else "Normal"

# Load DB (fixed backend path)
try:
    refs = load_db()
except Exception as e:
    st.error(f"Could not load the bibliography database at:\n\n{DB_PATH}\n\n{e}")
    st.stop()

# Sidebar (optional; collapsed by default)
with st.sidebar:
    st.header("Search controls")

    # Normal mode search
    query = st.text_input("Keyword query", placeholder="e.g., attention mechanisms interpretability")
    search_fields = st.multiselect(
        "Search in",
        options=["title", "abstract", "keywords", "authors", "journal", "doi"],
        default=["title", "abstract", "keywords", "authors"],
    )

    st.subheader("Filters")
    years = sorted([y for y in (_year_int(r.get("year")) for r in refs) if y is not None])
    if years:
        y_min, y_max = min(years), max(years)
    else:
        y_min, y_max = 1980, 2026

    year_min, year_max = st.slider(
        "Year range",
        min_value=min(1980, y_min),
        max_value=max(2026, y_max),
        value=(y_min, y_max),
        step=1,
    )
    require_doi = st.checkbox("Only with DOI", value=False)

    all_keywords = sorted(_unique([kw for r in refs for kw in (r.get("keywords") or []) if isinstance(kw, str)]))
    kw_filter = st.multiselect("Keyword filter (optional)", options=all_keywords, default=[])

    st.divider()
    st.header("Shortlist")
    st.write(f"Selected: **{len(st.session_state.selected_ids)}**")
    if st.button("Clear selected", use_container_width=True):
        st.session_state.selected_ids = set()

# AI mode panel (on main page)
ai_selected_ids: Optional[List[Any]] = st.session_state.ai_selected_ids

if st.session_state.mode == "AI":
    st.markdown("### AI search")
    st.caption("Describe what you want, and Gemini will select relevant items from your local library.")
    ai_query = st.text_area(
        "What are you looking for?",
        value=st.session_state.ai_query,
        height=110,
        placeholder="e.g., survey papers on transformer interpretability, with emphasis on attention attribution methods",
    )
    st.session_state.ai_query = ai_query

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        run_ai = st.button("Find in library", use_container_width=True)
    with c2:
        clear_ai = st.button("Clear AI filter", use_container_width=True)
    with c3:
        key_present = "✅ API key found" if get_gemini_key() else "⚠️ No API key configured"
        st.write(key_present)

    if clear_ai:
        st.session_state.ai_selected_ids = None
        st.rerun()

    if run_ai:
        key = get_gemini_key()
        if not key:
            st.warning("AI mode needs a Gemini API key. Set GEMINI_API_KEY in Streamlit secrets or your shell environment.")
            st.session_state.ai_selected_ids = []
        else:
            with st.spinner("Asking Gemini to search your local library…"):
                ids = ai_find_relevant_ids(refs, ai_query, key)
            st.session_state.ai_selected_ids = ids
        st.rerun()

    ai_selected_ids = st.session_state.ai_selected_ids
    if ai_selected_ids is not None:
        if len(ai_selected_ids) == 0:
            st.info("Gemini didn’t find any relevant items in this library for that request.")
        else:
            st.success(f"Gemini selected **{len(ai_selected_ids)}** candidate(s) from the library.")

st.divider()

# =========================
# Filter + rank
# =========================
filtered: List[Dict[str, Any]] = []
for r in refs:
    y = _year_int(r.get("year"))
    if y is not None and not (year_min <= y <= year_max):
        continue
    if require_doi and not str(r.get("doi", "")).strip():
        continue
    if kw_filter:
        r_kws = set(_norm(k) for k in (r.get("keywords") or []))
        if not all(_norm(k) in r_kws for k in kw_filter):
            continue
    filtered.append(r)

# Apply AI selection filter (if present)
if st.session_state.mode == "AI" and ai_selected_ids is not None:
    allowed = set(ai_selected_ids)
    filtered = [r for r in filtered if r.get("id") in allowed]

# Score (Normal mode ranking) – still used in AI mode for ordering within the selected set
query_for_scoring = query
scored = []
for r in filtered:
    s, per_field = _score_entry(r, query_for_scoring, search_fields) if query_for_scoring else (0.0, {})
    scored.append((s, per_field, r))

if query_for_scoring:
    scored.sort(key=lambda x: x[0], reverse=True)
else:
    scored.sort(key=lambda x: (_year_int(x[2].get("year")) or -1, x[2].get("id") or -1), reverse=True)

# =========================
# Main UI
# =========================
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.subheader("Results")
    st.caption(f"DB: **{len(refs)}** entries · Showing: **{len(scored)}**")

    table_rows = []
    for s, per_field, r in scored:
        row = entry_to_row(r)
        row["score"] = round(s, 3)
        row["selected"] = (r.get("id") in st.session_state.selected_ids)
        table_rows.append(row)

    df = pd.DataFrame(table_rows, columns=["selected", "score", "year", "title", "authors", "journal", "doi", "keywords", "id"])
    st.dataframe(
        df.drop(columns=["id"]),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    st.markdown("#### Browse + select")
    for s, per_field, r in scored[:50]:
        rid = r.get("id")
        title = r.get("title", "(untitled)")
        year = r.get("year", "")
        authors = r.get("authors", []) or []
        journal = r.get("journal", "")
        doi = r.get("doi", "")
        abstract = r.get("abstract", "")
        keywords = r.get("keywords", []) or []

        is_sel = rid in st.session_state.selected_ids
        head = f"{'✅' if is_sel else '⬜'} {title} ({year})"
        with st.expander(head, expanded=False):
            cols = st.columns([1, 1, 1])
            with cols[0]:
                st.markdown("**Authors**")
                st.write(_safe_join(authors))
            with cols[1]:
                st.markdown("**Venue**")
                st.write(journal or "—")
                st.markdown("**DOI**")
                st.code(doi or "—")
            with cols[2]:
                st.markdown("**Keywords**")
                st.write(", ".join(keywords) if keywords else "—")

            if abstract:
                st.markdown("**Abstract**")
                st.write(abstract)

            if query_for_scoring and per_field:
                st.markdown("**Why this matched (keyword scoring)**")
                st.json({k: round(v, 3) for k, v in sorted(per_field.items(), key=lambda kv: kv[1], reverse=True)})

            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("Add to shortlist" if not is_sel else "Remove from shortlist", key=f"toggle_{rid}"):
                    if is_sel:
                        st.session_state.selected_ids.discard(rid)
                    else:
                        st.session_state.selected_ids.add(rid)
                    st.rerun()

            with c2:
                note = st.text_area(
                    "Notes (optional)",
                    value=st.session_state.notes.get(rid, ""),
                    key=f"note_{rid}",
                    height=80,
                    placeholder="Why is this relevant? Any key claims/methods?",
                )
                st.session_state.notes[rid] = note

            tag_str = st.text_input(
                "Tags (comma-separated)",
                value=", ".join(st.session_state.tags.get(rid, [])),
                key=f"tags_{rid}",
                placeholder="e.g., survey, methods, baseline",
            )
            st.session_state.tags[rid] = [t.strip() for t in tag_str.split(",") if t.strip()]

with right:
    st.subheader("Shortlist")
    selected_entries = [r for r in refs if r.get("id") in st.session_state.selected_ids]
    if not selected_entries:
        st.info("No papers selected yet. Add some from the Results pane.")
    else:
        st.write(f"**{len(selected_entries)}** selected")

        sel_rows = []
        for r in selected_entries:
            row = entry_to_row(r)
            row["notes"] = st.session_state.notes.get(r.get("id"), "")
            row["tags"] = ", ".join(st.session_state.tags.get(r.get("id"), []))
            sel_rows.append(row)
        sel_df = pd.DataFrame(sel_rows, columns=["year", "title", "authors", "journal", "doi", "keywords", "tags", "notes", "id"])
        st.dataframe(sel_df.drop(columns=["id"]), use_container_width=True, hide_index=True, height=260)

        st.markdown("#### Export")
        as_json = {"references": selected_entries}
        bibtex_blob = "\n\n".join(_as_bibtex(r) for r in selected_entries)
        ris_blob = "\n\n".join(_as_ris(r) for r in selected_entries)

        st.download_button(
            "Download selected (JSON)",
            data=json.dumps(as_json, indent=2),
            file_name="selected_references.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Download selected (BibTeX)",
            data=bibtex_blob,
            file_name="selected_references.bib",
            mime="text/plain",
            use_container_width=True,
        )
        st.download_button(
            "Download selected (RIS)",
            data=ris_blob,
            file_name="selected_references.ris",
            mime="text/plain",
            use_container_width=True,
        )

with st.expander("Gemini API setup (quick)"):
    st.markdown(
        """
**Option A (recommended): Streamlit secrets**

Create a file named `.streamlit/secrets.toml` next to your app:

```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
```

**Option B: environment variable**

In your shell:

```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

This app will pick up the key from `st.secrets["GEMINI_API_KEY"]` first, then `GEMINI_API_KEY`.
"""
    )
