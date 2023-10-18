import streamlit as st
import os

# import extra_streamlit_components as stx

from thoughts_gpt.components.sidebar import sidebar

from thoughts_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from thoughts_gpt.core.caching import bootstrap_caching

from thoughts_gpt.core.parsing import read_file
from thoughts_gpt.core.parsing import read_url
from thoughts_gpt.core.chunking import chunk_file
from thoughts_gpt.core.embedding import embed_files
from thoughts_gpt.core.qa import query_folder
from thoughts_gpt.core.utils import get_llm
from thoughts_gpt.core.prompts import get_prompt


EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

st.set_page_config(page_title="GPT有所思", page_icon="📖", layout="wide")
st.header("🤔 GPT有所思")


# Enable caching for expensive functions
bootstrap_caching()

sidebar()

openai_api_key = st.session_state.get("OPENAI_API_KEY")
stuff_prompt = st.session_state.get("STUFF_PROMPT")
show_full_doc = st.session_state.get("SHOW_FULL_DOC", False)


if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar. You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )


tab_file, tab_url = st.tabs(["📃 File", "🕸️ URL"])

# chosen_id = stx.tab_bar(data=[
#     stx.TabBarItemData(id="file", title="📃 File", description=""),
#     stx.TabBarItemData(id="url", title="🕸️ URL", description=""),
#     stx.TabBarItemData(id="eg", title="🐦 Prompt Example", description="")]
# )

placeholder = st.container()

uploaded_file = ""
typed_url = ""
with tab_file:
    uploaded_file = st.file_uploader(
        "Upload a pdf, docx, txt, or md file",
        type=["pdf", "docx", "txt", "md"],
        help="Scanned documents are not supported yet!",
    )

with tab_url:
    typed_url = st.text_input(
        "Enter a URL", 
        placeholder="Enter URL", 
        help="The URL must be accessible without any login status.",
    )

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

# with st.expander("Advanced Options"):
#     return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    


if not uploaded_file and not typed_url:
    st.stop()

try:
    if uploaded_file:
        file = read_file(uploaded_file)
    else:
        file = read_url(typed_url)
except Exception as e:
    display_file_read_error(e, file_name=uploaded_file.name if uploaded_file else typed_url)

chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)
 
if not is_file_valid(file):
    st.stop()
 
 
if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()
 
 
with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
        openai_api_key=openai_api_key,
    )
 
with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")
 
 
if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)
 
 
if submit:
    if not is_query_valid(query):
        st.stop()
 
    # Output Columns
    answer_col, sources_col = st.columns(2)
 
    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
    prompt = get_prompt(stuff_prompt)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        llm=llm,
        stuff_prompt=prompt
    )
 
    with answer_col:
        st.markdown("#### ✨ Answer")
        st.success(result.answer)

        st.markdown("##### 💡 Suggested Questions")
        st.info('- ' + '\n- '.join(result.suggested_questions))
 
    with sources_col:
        st.markdown("#### 📄  Context")
        
        source_tabs = st.tabs([ s.metadata["source"] for s in result.sources ])

        for source, tab in zip(result.sources, source_tabs):
            with tab:
                st.info(source.page_content)
                # st.markdown(source.page_content)
                # st.markdown({source.metadata["source"]})
                # st.markdown("---")
 