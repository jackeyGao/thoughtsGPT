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
from thoughts_gpt.core.const import EMBEDDING, VECTOR_STORE, MODEL_LIST
from thoughts_gpt.core.const import PAGE_ICON, PAGE_TITLE


# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.header(f"{PAGE_ICON} {PAGE_TITLE}")


# Enable caching for expensive functions
bootstrap_caching()

sidebar()

st.markdown("""
<style>
.st-ct code.language-python {
    word-break: break-all;
    overflow: hidden;
    overflow-x: hidden;
    white-space: pre-wrap!important;
    word-wrap: break-word;
}
</style>""", unsafe_allow_html=True)

openai_api_key = st.session_state.get("OPENAI_API_KEY")
stuff_prompt = st.session_state.get("STUFF_PROMPT")
show_full_doc = st.session_state.get("SHOW_FULL_DOC", False)
similar_docs_limit = st.session_state.get("SIMILAR_DOCS_LIMIT", 5)
temperature = st.session_state.get("TEMPERRATURE", 0.1)
suggested_questions_limit = st.session_state.get("SUGGESTED_QUESTIONS_LIMIT", 5)


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
        collection_name=file.id,
    )


with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")
 
 
if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)
 
 
if submit:
    with st.spinner(f'LLM({model}) is answering, it may take a while. ⏳'):
        if not is_query_valid(query):
            st.stop()
    
        # Output Columns
        answer_col, sources_col = st.columns(2)
    
        llm = get_llm(
            model=model, 
            openai_api_key=openai_api_key, 
            temperature=temperature
        )
        
        prompt = get_prompt(stuff_prompt)
        result = query_folder(
            folder_index=folder_index,
            query=query,
            llm=llm,
            k=similar_docs_limit,
            stuff_prompt=prompt,
            suggested_questions_limit=suggested_questions_limit
        )

    with answer_col:
        st.markdown("#### ✨ Answer")
        st.success(result.answer)

        st.markdown("##### 💡 Suggested Questions")
        if result.suggested_questions:
            st.info('- ' + '\n- '.join(result.suggested_questions))
        else:
            st.warning("No suggested questions")
 
    with sources_col:
        st.markdown("#### 🍞 Context")
        
        source_tabs = st.tabs([ s.metadata["source"] for s in result.sources ] + ["🥔 Prompt", "🍟 Original"])

        for source, tab in zip(result.sources, source_tabs[0:-2]):
            with tab:
                
                # col1, col2, col3, col4 = st.columns(4)
                # col1.info(f"Source: {source.metadata['source']}")
                # col2.info(f"FileName: {source.metadata['file_name']}")
                # col3.info(f"Page: {source.metadata['page']}")
                # col4.info(f"#_id: {source.metadata['_id'][-8:]}")
                # col2.metric("Wind", "9 mph", "-8%")
                # col3.metric("Humidity", "86%", "4%")

                st.code(source.page_content, line_numbers=True)
                st.caption(
                    f":blue[source]: {source.metadata['source']}, " + 
                    f":blue[filename]: {source.metadata['file_name']}, " + 
                    f":blue[page]: {source.metadata['page']}, " + 
                    f":blue[_id]: {source.metadata['_id']}"
                )
                # st.markdown(source.page_content)
                # st.markdown({source.metadata["source"]})
                # st.markdown("---")

        with source_tabs[-2]:
            st.caption(f":blue[Token]: {result.prompt_length}")
            prompt_content = prompt.format(
                question=query, 
                suggested_questions_limit=suggested_questions_limit,
                summaries=result.summaries
            )
            st.code(prompt_content, language="python")
        
        with source_tabs[-1]:
            st.code(result.original_anwser, language="python")
 