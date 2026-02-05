import json
import re
from dataclasses import dataclass
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
    initial_sidebar_state="expanded",
)

DATA_DEFAULT_PATH = Path(__file__).with_name("example-bib (1).json")  # same folder as this app by default


# =========================
# Utilities
# =========================
def _norm(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> List[str]:
    s = _norm(s)
    # keep simple: letters/numbers only, split on non-word
    toks = re.split(r"[^a-z0-9]+", s)
    return [t for t in toks if t]

def _safe_join(x: Any, sep: str = ", ") -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return sep.join(str(v) for v in x if v is not None)
    return str(x)

def _first_author(authors: List[str]) -> str:
    return authors[0] if authors else ""

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
    # Hartman2012CognitiveDrift
    fa = _first_author(authors)
    last = fa.split()[-1] if fa else "Unknown"
    y = str(year) if year is not None else "n.d."
    first_word = _tokenize(title)[:1]
    fw = first_word[0].capitalize() if first_word else "Work"
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

    # Minimal BibTeX-ish (prototype)
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
    # Drop empties
    fields = {k: v for k, v in fields.items() if str(v).strip() != ""}

    lines = [f"@article{{{key},"]
    for k, v in fields.items():
        v = str(v).replace("{", "\\{").replace("}", "\\}")
        lines.append(f"  {k} = {{{v}}},")
    lines.append("}")
    return "\n".join(lines)

def _as_ris(entry: Dict[str, Any]) -> str:
    # Very small subset (prototype)
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
    """
    Lightweight scoring:
      - token overlap between query tokens and field tokens
      - bonus for exact substring match (query in field)
    Returns: (score, per_field_scores)
    """
    q = _norm(query)
    if not q:
        return 0.0, {}

    q_toks = set(_tokenize(q))
    if not q_toks:
        return 0.0, {}

    per_field = {}
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
        tok_score = overlap / denom  # 0..~1
        exact_bonus = 0.7 if q in t else 0.0

        # Field weights
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
def load_db(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = data.get("references", [])
    # normalize a little so downstream UI is stable
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


def entry_to_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    authors = entry.get("authors", []) or []
    return {
        "id": entry.get("id"),
        "year": entry.get("year"),
        "title": entry.get("title", ""),
        "authors": _safe_join(authors),
        "journal": entry.get("journal", ""),
        "doi": entry.get("doi", ""),
        "keywords": _safe_join(entry.get("keywords", [])),
    }


# =========================
# Sidebar controls
# =========================
init_state()

st.title("📚 LitSearch Prototype")
st.caption("A tiny Streamlit prototype for searching + shortlisting literature review candidates.")

with st.sidebar:
    st.header("Data")
    st.write("This prototype loads a small JSON bibliography.")
    data_path = st.text_input(
        "JSON path",
        value=str(DATA_DEFAULT_PATH),
        help="Point this at your own export later (same schema).",
    )
    st.divider()

    st.header("Search")
    query = st.text_input("Query", placeholder="e.g., attention mechanisms interpretability")
    search_fields = st.multiselect(
        "Search in",
        options=["title", "abstract", "keywords", "authors", "journal", "doi"],
        default=["title", "abstract", "keywords", "authors"],
    )

    st.subheader("Filters")
    year_min, year_max = st.slider(
        "Year range",
        min_value=1980,
        max_value=2026,
        value=(2009, 2021),
        step=1,
    )
    require_doi = st.checkbox("Only with DOI", value=False)

    st.divider()
    st.header("Shortlist")
    st.write(f"Selected: **{len(st.session_state.selected_ids)}**")
    if st.button("Clear selected", use_container_width=True):
        st.session_state.selected_ids = set()

# =========================
# Load + prep
# =========================
try:
    refs = load_db(data_path)
except Exception as e:
    st.error(f"Could not load JSON from: {data_path}\n\n{e}")
    st.stop()

all_years = sorted([y for y in (_year_int(r.get("year")) for r in refs) if y is not None])
if all_years:
    # adjust slider defaults to the dataset range if user hasn't moved it much
    # (keep user's chosen slider as-is; just show dataset info)
    st.caption(f"DB size: **{len(refs)}** entries · Year span: **{min(all_years)}–{max(all_years)}**")
else:
    st.caption(f"DB size: **{len(refs)}** entries")

all_keywords = sorted(_unique([kw for r in refs for kw in (r.get("keywords") or []) if isinstance(kw, str)]))
kw_filter = st.multiselect("Keyword filter (optional)", options=all_keywords, default=[])

# =========================
# Search + rank
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

scored = []
for r in filtered:
    s, per_field = _score_entry(r, query, search_fields) if query else (0.0, {})
    scored.append((s, per_field, r))

# If no query, sort by year desc then id
if query:
    scored.sort(key=lambda x: x[0], reverse=True)
else:
    scored.sort(key=lambda x: (_year_int(x[2].get("year")) or -1, x[2].get("id") or -1), reverse=True)

# =========================
# Main UI
# =========================
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.subheader("Results")

    if query:
        st.write(f"Showing **{len(scored)}** results (ranked).")
    else:
        st.write(f"Showing **{len(scored)}** entries (no query; sorted by year).")

    # Compact table view
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
        height=340,
    )

    st.markdown("#### Browse + select")
    st.caption("Tip: use the expanders to inspect abstracts quickly, then add to your shortlist.")

    # Detailed cards
    for s, per_field, r in scored[:50]:  # keep UI snappy
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

            # Explain ranking a bit (for prototype transparency)
            if query and per_field:
                st.markdown("**Why this matched**")
                st.json({k: round(v, 3) for k, v in sorted(per_field.items(), key=lambda kv: kv[1], reverse=True)})

            # Select control
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

        # quick overview
        sel_rows = []
        for r in selected_entries:
            row = entry_to_row(r)
            row["notes"] = st.session_state.notes.get(r.get("id"), "")
            row["tags"] = ", ".join(st.session_state.tags.get(r.get("id"), []))
            sel_rows.append(row)
        sel_df = pd.DataFrame(sel_rows, columns=["year", "title", "authors", "journal", "doi", "keywords", "tags", "notes", "id"])
        st.dataframe(sel_df.drop(columns=["id"]), use_container_width=True, hide_index=True, height=260)

        st.markdown("#### Export")
        # Export formats
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

st.divider()
with st.expander("How to extend this prototype"):
    st.markdown(
        """
**Easy next upgrades (in roughly this order):**
1) **Better ranking** (TF-IDF cosine similarity) for larger libraries.
2) **Facet filters** (journal, author, keyword counts).
3) **PDF upload + full-text search** (local prototype) and/or link-outs to DOI.
4) **Dedup + merge** (multiple exports).
5) **Sync** with Zotero/Mendeley later.

If you paste your real schema (or export), I can adapt the loader + fields in ~10 minutes.
"""
    )
