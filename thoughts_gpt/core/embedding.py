__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.vectorstores import VectorStore
from thoughts_gpt.core.parsing import File
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
from chromadb.config import Settings as ChromeSettings
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from typing import List, Type
from langchain.docstore.document import Document
from thoughts_gpt.core.debug import FakeVectorStore, FakeEmbeddings
from thoughts_gpt.core.const import EMBEDDING_MAX_RETRIES
from thoughts_gpt.core.utils import hash_id


class FolderIndex:
    """Index for a collection of files (a folder)"""

    def __init__(self, files: List[File], index: VectorStore):
        self.name: str = "default"
        self.files = files
        self.index: VectorStore = index

    @staticmethod
    def _combine_files(files: List[File]) -> List[Document]:
        """Combines all the documents in a list of files into a single list."""

        all_texts, ids = [], []
        for file in files:
            for doc in file.docs:
                _id = hash_id(f"{file.id},{doc.to_json}")
                doc.metadata["file_name"] = file.name
                doc.metadata["file_id"] = file.id
                doc.metadata["_id"] = _id
                all_texts.append(doc)
                ids.append(_id)

        return all_texts, ids

    @classmethod
    def from_files(
        cls, files: List[File], embeddings: Embeddings, vector_store: Type[VectorStore],
        collection_name: str = "default"
    ) -> "FolderIndex":
        """Creates an index from files."""

        all_docs, ids = cls._combine_files(files)

        index = vector_store.from_documents(
            documents=all_docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory="./chroma", 
            client_settings=ChromeSettings(
                persist_directory="./chroma",
                is_persistent=True
            ),
            ids=ids,
        )

        return cls(files=files, index=index)


def embed_files(
    files: List[File], embedding: str, vector_store: str, collection_name: str, **kwargs
) -> FolderIndex:
    """Embeds a collection of files and stores them in a FolderIndex."""

    supported_embeddings: dict[str, Type[Embeddings]] = {
        "openai": OpenAIEmbeddings,
        "debug": FakeEmbeddings,
    }
    supported_vector_stores: dict[str, Type[VectorStore]] = {
        "faiss": FAISS,
        "chroma": Chroma,
        "debug": FakeVectorStore,
    }

    if embedding in supported_embeddings:
        _embeddings = supported_embeddings[embedding](**kwargs)
        _embeddings.max_retries = EMBEDDING_MAX_RETRIES
    else:
        raise NotImplementedError(f"Embedding {embedding} not supported.")

    if vector_store in supported_vector_stores:
        _vector_store = supported_vector_stores[vector_store]
    else:
        raise NotImplementedError(f"Vector store {vector_store} not supported.")

    return FolderIndex.from_files(
        files=files, embeddings=_embeddings, vector_store=_vector_store, collection_name=collection_name
    )
