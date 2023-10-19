import streamlit as st

from dotenv import load_dotenv
from thoughts_gpt.core.prompts import default_template
from thoughts_gpt.components.prompt_example import example
from thoughts_gpt import __version__

import os

load_dotenv()



def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
            "2. Upload a pdf, docx, or txt file📄, or enter URL.\n"
            "3. Ask a question about the document💬\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )

        st.session_state["OPENAI_API_KEY"] = api_key_input

        stuff_prompt = st.text_area(
            "Stuff Prompt",
            placeholder="stuff prompt",
            help="https://smith.langchain.com/hub/rlm/rag-prompt",  # noqa: E501
            value=st.session_state.get("STUFF_PROMPT", "")
            or default_template,
        )

        st.session_state["STUFF_PROMPT"] = stuff_prompt

        with st.expander("Advanced Options"):
            show_full_doc = st.checkbox('Show parsed contents of the document')
            temperature = st.number_input(
                label="LLM Temperature",
                step=0.1, 
                format="%.2f", 
                min_value=0., 
                max_value=2.
            )
            suggested_questions_limit = st.number_input(
                "Suggested Questions Limit", 
                max_value=10, 
                min_value=3, 
                value=5,
                help="suggested questions limit",  
            )
            similar_docs_limit = st.number_input(
                "Similar Documents Limit",
                value=5,
                min_value=1,
                max_value=12,
                help="similar docs limit",  # noqa: E501
            )


        st.session_state["SHOW_FULL_DOC"] = show_full_doc
        st.session_state["TEMPERRATURE"] = temperature
        st.session_state["SUGGESTED_QUESTIONS_LIMIT"] = suggested_questions_limit
        st.session_state["SIMILAR_DOCS_LIMIT"] = similar_docs_limit


        st.markdown(
            "## Prompt examples\n"
            "1. Copy the example prompt below 🔥\n"  # noqa: E501
            "2. Paste text into stuff prompt input \n"
        )

        example()

        st.caption(f":blue[version]: {__version__}")


    # Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').

