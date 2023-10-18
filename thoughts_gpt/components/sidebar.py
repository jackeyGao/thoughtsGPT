import streamlit as st

from dotenv import load_dotenv
from thoughts_gpt.core.prompts import default_template
from thoughts_gpt.components.prompt_example import example

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

        st.session_state["SHOW_FULL_DOC"] = show_full_doc

        st.markdown(
            "## Prompt examples\n"
            "1. Copy the example prompt below 🔥\n"  # noqa: E501
            "2. Paste text into stuff prompt input \n"
        )

        example()


    # Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').

