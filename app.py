import streamlit as st

# --- INITIAL SETUP ---
# CRITICAL: This must be the very first Streamlit command.
st.set_page_config(
    page_title="ScholarLite", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

import json
import requests
import time
import os
import math
import graphviz
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BIB_FILE = "example-bib-tiff.json"

def load_data():
    """Load bibliography data from JSON file. STRICT MODE: File must exist."""
    if not os.path.exists(BIB_FILE):
        st.error(f"⚠️ Critical Error: `{BIB_FILE}` not found. Please ensure the file exists in the directory.")
        st.stop()
        
    try:
        with open(BIB_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading {BIB_FILE}: {e}")
        st.stop()

# --- STATE MANAGEMENT ---
if "bib_data" not in st.session_state:
    st.session_state.bib_data = load_data()
if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}
if "saved_papers" not in st.session_state:
    st.session_state.saved_papers = set()
if "expanded_abstracts" not in st.session_state:
    st.session_state.expanded_abstracts = set()
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# --- GEMINI API INTEGRATION ---
def get_api_key():
    """Helper to get API key from env or session."""
    return os.getenv("GEMINI_API_KEY") or st.session_state.user_api_key

def call_gemini(prompt):
    api_key = get_api_key()
    
    if not api_key:
         return "Error: API Key missing. Please enter it in the interface."

    try:
        # Initialize client dynamically with the available key
        client = genai.Client(api_key=api_key)
        
        # Exponential backoff implementation
        for delay in [1, 4]:
            try:
                response = client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=prompt
                )
                if hasattr(response, 'text'):
                    return response.text
                else:
                    return "Error: No text in response."
            except Exception as e:
                time.sleep(delay)
                
    except Exception as e:
        return f"Error initializing AI client: {str(e)}"
        
    return "Error: Could not reach AI services."

# --- UI COMPONENTS ---

# CSS for Google Scholar-like look
st.markdown("""
<style>
    .result-title { 
        font-size: 20px; 
        color: #3B82F6 !important; 
        text-decoration: none; 
        font-weight: 600; 
    }
    .result-title:hover { text-decoration: underline; }
    
    .result-meta { 
        font-size: 14px; 
        opacity: 0.8; 
        margin-bottom: 4px; 
    }
    
    .result-snippet { 
        font-size: 14px; 
        opacity: 0.9; 
        margin-bottom: 8px;
    }
    
    .stButton button { margin-top: 5px; }
    
    /* Highlight the Sidebar Radio Button for Saved Papers if active */
    div[data-testid="stRadio"] label {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR LAYOUT (Nav & Graph) ---
with st.sidebar:
    st.title("🎓 ScholarLite")
    
    # 1. Navigation with visual counter
    saved_count = len(st.session_state.saved_papers)
    st.caption(f"📚 You have {saved_count} saved paper{'s' if saved_count != 1 else ''}.")
    
    page_view = st.radio("Navigation", ["Search Library", "My Saved Papers"])
    
    st.divider()
    
    # 2. Citation Graph (Always visible)
    st.subheader("🕸️ Citation Ecosystem")
    
    if not st.session_state.saved_papers:
        st.info("💡 **Tip:** Save papers from the search results to build your personal citation map here!")
    else:
        st.caption("Visualizing connections between your saved papers.")
        
        try:
            # Create Graph
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR', size='10') # LR = Left to Right
            
            # Track added nodes to prevent duplicates in graph definition
            added_nodes = set()
            
            # Logic: 
            # 1. Add all Saved Papers (GREEN)
            # 2. Add papers cited by Saved Papers (RED, unless they are also saved)
            
            for saved_id in st.session_state.saved_papers:
                # Find paper data
                paper = next((r for r in st.session_state.bib_data["references"] if r["id"] == saved_id), None)
                if not paper: continue
                
                # Add Saved Node (Green)
                if saved_id not in added_nodes:
                    graph.node(str(saved_id), 
                              label=f"{paper['title'][:15]}...", 
                              shape='box', 
                              style='filled', 
                              fillcolor='#ccffcc', # Light Green
                              fontsize='10')
                    added_nodes.add(saved_id)
                
                # Process Citations
                citations = paper.get("citations", [])
                for cid in citations:
                    cited_paper = next((r for r in st.session_state.bib_data["references"] if r["id"] == cid), None)
                    if cited_paper:
                        # Determine color: Green if saved, Red if not
                        is_saved = cid in st.session_state.saved_papers
                        fill_color = '#ccffcc' if is_saved else '#ffcccc' # Green vs Red
                        
                        if cid not in added_nodes:
                            graph.node(str(cid), 
                                      label=f"{cited_paper['title'][:15]}...", 
                                      shape='ellipse', 
                                      style='filled', 
                                      fillcolor=fill_color,
                                      fontsize='9')
                            added_nodes.add(cid)
                        
                        # Add Edge
                        graph.edge(str(saved_id), str(cid))

            st.graphviz_chart(graph)
            
            # Legend with clear indicators
            c_leg1, c_leg2 = st.columns(2)
            c_leg1.markdown("🟢 **Saved**")
            c_leg2.markdown("🔴 **Cited Only**")
            
        except Exception as e:
            st.warning("Could not render citation graph. Ensure Graphviz is installed.")

    st.divider()
    
    # 3. API Key Settings
    st.subheader("🔑 API Settings")
    if os.getenv("GEMINI_API_KEY"):
        st.success("✅ API Key loaded from environment")
    else:
        st.session_state.user_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.user_api_key,
            type="password",
            placeholder="Paste your Google AI Studio Key",
            help="Required for AI features. Key is not saved to disk."
        )


# --- MAIN CONTENT AREA ---

# Helper to render a single document card
def render_document(doc, show_ai_options):
    # Title Link
    doi_link = f"https://doi.org/{doc.get('doi')}" if doc.get('doi') else "#"
    st.markdown(f'<a href="{doi_link}" target="_blank" class="result-title">{doc["title"]}</a>', unsafe_allow_html=True)
    
    # Meta Data
    authors = ", ".join(doc.get("authors", []))
    citation_count = len(doc.get('citations', []))
    meta_text = f"{authors} - {doc.get('journal', 'Unknown Journal')}, {doc.get('year')} &nbsp;•&nbsp; Cited by {citation_count}"
    st.markdown(f"<div class='result-meta'>{meta_text}</div>", unsafe_allow_html=True)
    
    # Keywords
    if "keywords" in doc:
        st.caption(f"Keywords: {', '.join(doc['keywords'])}")

    # Abstract Logic (Inline Expansion)
    abstract_text = doc.get('abstract', 'No abstract available.')
    is_expanded = doc['id'] in st.session_state.expanded_abstracts
    
    if is_expanded:
        st.write(abstract_text)
        if st.button("Show less", key=f"less_{doc['id']}"):
            st.session_state.expanded_abstracts.remove(doc['id'])
            st.rerun()
    else:
        words = abstract_text.split()
        snippet = " ".join(words[:15]) + "..."
        st.write(f"{snippet}")
        if st.button("Show more", key=f"more_{doc['id']}"):
            st.session_state.expanded_abstracts.add(doc['id'])
            st.rerun()

    # Save Button
    is_saved = doc['id'] in st.session_state.saved_papers
    save_label = "✅ Saved" if is_saved else "💾 Save"
    
    if st.button(save_label, key=f"save_{doc['id']}"):
        if is_saved:
            st.session_state.saved_papers.remove(doc['id'])
            st.toast("Paper removed from library.", icon="🗑️")
        else:
            st.session_state.saved_papers.add(doc['id'])
            st.toast("Paper saved! See the sidebar for your citation map.", icon="🕸️")
        st.rerun()

    # AI Features
    if show_ai_options:
        # Check if we have a key before showing AI UI
        if get_api_key():
            c1, c2 = st.columns(2)
            doc_id = doc['id']
            
            with c1:
                with st.expander("👶 Simple Summary"):
                    simple_key = f"simple_{doc_id}"
                    if simple_key not in st.session_state.ai_cache:
                        with st.spinner("Simplifying..."):
                            p = f"Explain this paper to a non-expert in simple terms:\nTitle: {doc['title']}\nAbstract: {doc['abstract']}"
                            st.session_state.ai_cache[simple_key] = call_gemini(p)
                    st.write(st.session_state.ai_cache[simple_key])

            with c2:
                with st.expander("🧐 Expert Summary"):
                    expert_key = f"expert_{doc_id}"
                    if expert_key not in st.session_state.ai_cache:
                        with st.spinner("Analyzing..."):
                            p = f"Provide a technical summary for a domain expert, focusing on methodology and contribution:\nTitle: {doc['title']}\nAbstract: {doc['abstract']}"
                            st.session_state.ai_cache[expert_key] = call_gemini(p)
                    st.write(st.session_state.ai_cache[expert_key])
        else:
            st.warning("⚠️ AI features paused. Please enter your API Key in the sidebar.")
    
    st.markdown("---")

# --- VIEW LOGIC ---

if page_view == "Search Library":
    # Layout: Top Bar for Search
    col_search_header, col_ai_toggle = st.columns([4, 1])
    with col_search_header:
        st.subheader("🔍 Search")
    with col_ai_toggle:
        ai_mode = st.toggle("✨ AI Mode", value=False)
    
    # --- API KEY CHECK LOGIC ---
    if ai_mode and not get_api_key():
        st.warning("⚠️ AI Mode enabled but no API Key found. Please configure it in the sidebar.")

    # Search Inputs
    c_search, c_sort = st.columns([3, 1])
    with c_search:
        if "search_query" not in st.session_state:
            st.session_state.search_query = ""
            
        # UPDATED: Use 'key' to bind directly to session state. 
        # Removed manual 'value' assignment and manual state update to prevent overwrite bugs.
        search_query = st.text_input(
            "Keywords, authors, or titles", 
            key="search_query",
            placeholder="Try searching for 'cognitive drift' or 'neural networks'"
        )
    with c_sort:
        sort_option = st.selectbox("Sort by", ["Relevance", "Year (Newest)", "Year (Oldest)", "Citations (Most)"])

    if search_query:
        # Filtering
        refs = st.session_state.bib_data.get("references", [])
        query_lower = search_query.lower()
        
        filtered_results = [
            r for r in refs 
            if query_lower in r.get("title", "").lower() or 
               any(query_lower in k.lower() for k in r.get("keywords", [])) or
               any(query_lower in a.lower() for a in r.get("authors", []))
        ]

        # Sorting
        if sort_option == "Year (Newest)":
            filtered_results.sort(key=lambda x: x.get('year', 0), reverse=True)
        elif sort_option == "Year (Oldest)":
            filtered_results.sort(key=lambda x: x.get('year', 0))
        elif sort_option == "Citations (Most)":
            filtered_results.sort(key=lambda x: len(x.get('citations', [])), reverse=True)

        # Pagination
        total_results = len(filtered_results)
        ITEMS_PER_PAGE = 10
        if "page" not in st.session_state: st.session_state.page = 1
        total_pages = math.ceil(total_results / ITEMS_PER_PAGE)
        if st.session_state.page > max(1, total_pages): st.session_state.page = max(1, total_pages)
        
        start_idx = (st.session_state.page - 1) * ITEMS_PER_PAGE
        current_page_results = filtered_results[start_idx : start_idx + ITEMS_PER_PAGE]

        st.markdown(f"**Found {total_results} results**")
        
        # AI Aggregated Summary
        if ai_mode and current_page_results:
            if get_api_key():
                with st.container():
                    st.markdown("### ✨ AI Overview")
                    cache_key = f"agg_{search_query}_{st.session_state.page}"
                    if cache_key not in st.session_state.ai_cache:
                        with st.spinner("Synthesizing themes..."):
                            papers_text = "\n".join([f"- {r['title']}: {r.get('abstract', '')[:200]}..." for r in current_page_results])
                            prompt = f"Provide a brief, high-level summary of the common themes found in these research papers:\n{papers_text}"
                            st.session_state.ai_cache[cache_key] = call_gemini(prompt)
                    st.info(st.session_state.ai_cache[cache_key])
                    st.divider()

        # Render List
        for doc in current_page_results:
            render_document(doc, ai_mode)

        # Pagination Buttons
        if total_pages > 1:
            c_prev, c_display, c_next = st.columns([1, 2, 1])
            with c_prev:
                if st.session_state.page > 1 and st.button("Previous"):
                    st.session_state.page -= 1
                    st.rerun()
            with c_display:
                st.markdown(f"<center>Page {st.session_state.page} of {total_pages}</center>", unsafe_allow_html=True)
            with c_next:
                if st.session_state.page < total_pages and st.button("Next"):
                    st.session_state.page += 1
                    st.rerun()
    else:
        st.info("Start typing to search your library.")
        if not st.session_state.saved_papers:
            st.markdown("""
            <div style="padding: 15px; border: 1px dashed #ccc; border-radius: 8px; text-align: center; color: #666;">
                Save papers to see them appear in the <b>Citation Ecosystem</b> on the sidebar! 👈
            </div>
            """, unsafe_allow_html=True)

elif page_view == "My Saved Papers":
    col_saved_header, col_ai_saved = st.columns([4, 1])
    with col_saved_header:
        st.subheader("💾 My Saved Papers")
    with col_ai_saved:
        ai_mode_saved = st.toggle("✨ AI Mode", value=False)

    # --- API KEY CHECK LOGIC (SAVED VIEW) ---
    if ai_mode_saved and not get_api_key():
        st.warning("⚠️ AI Mode enabled but no API Key found. Please configure it in the sidebar.")
    
    if not st.session_state.saved_papers:
        st.write("You haven't saved any papers yet. Go to Search to add some!")
    else:
        saved_docs = [
            r for r in st.session_state.bib_data["references"] 
            if r["id"] in st.session_state.saved_papers
        ]
        
        st.markdown(f"**You have {len(saved_docs)} saved articles.**")
        st.divider()
        
        for doc in saved_docs:
            render_document(doc, ai_mode_saved)