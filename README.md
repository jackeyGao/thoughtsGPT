## 🍔 thoughtsGPT

Knowledge document AI assistant, GPT 有所思。

[🍨 demo](https://feishu.streamlit.app/)


## used langchain module

- core/parsing.py:from langchain.docstore.document import Document
- core/parsing.py:from langchain.document_loaders.url import UnstructuredURLLoader
- core/prompts.py:from langchain.prompts import PromptTemplate
- core/prompts.py:from langchain.chains.qa_with_sources.stuff_prompt import template # this template
- core/chunking.py:from langchain.text_splitter import RecursiveCharacterTextSplitter
- core/utils.py:from langchain.chains.combine_documents.stuff import StuffDocumentsChain
- core/utils.py:from langchain.chat_models import ChatOpenAI
- core/utils.py:from langchain.chat_models.base import BaseChatModel
- core/debug.py:from langchain.vectorstores import VectorStore
- core/debug.py:from langchain.embeddings.base import Embeddings
- core/debug.py:from langchain.embeddings.fake import FakeEmbeddings as FakeEmbeddingsBase
- core/debug.py:from langchain.chat_models.fake import FakeListChatModel
- core/qa.py:from langchain.chains.qa_with_sources import load_qa_with_sources_chain
- core/qa.py:from langchain.prompts import PromptTemplate
- core/qa.py:from langchain.chat_models.base import BaseChatModel
- core/embedding.py:from langchain.vectorstores import VectorStore
- core/embedding.py:from langchain.vectorstores.faiss import FAISS
- core/embedding.py:from langchain.vectorstores.chroma import Chroma
- core/embedding.py:from langchain.embeddings import OpenAIEmbeddings
- core/embedding.py:from langchain.embeddings.base import Embeddings
- core/embedding.py:from langchain.docstore.document import Document

