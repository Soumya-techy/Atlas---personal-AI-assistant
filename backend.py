import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict, Annotated
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.storage import create_kv_docstore,LocalFileStore
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool

load_dotenv()
llm=ChatGroq(model='openai/gpt-oss-120b',api_key=os.getenv('GROQ_API_KEY'))
embeding_model=HuggingFaceEmbeddings(model='BAAI/bge-large-en-v1.5')
reranking_model=HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-v2-m3')


vector_store=Chroma(embedding_function=embeding_model,persist_directory='chroma_db',collection_name='children')
parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
child_splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
docstore=create_kv_docstore(LocalFileStore('parent_chunks'))
base_reteiver=ParentDocumentRetriever(vectorstore=vector_store,
                                          parent_splitter=parent_splitter,
                                          child_splitter=child_splitter,
                                          docstore=docstore,
                                          search_kwargs={'k':10})

compressor=CrossEncoderReranker(model=reranking_model,top_n=5)
main_retreiver=ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_reteiver)


def ingest_document(file_path):
    document=PyPDFLoader(file_path).load()
    base_reteiver.add_documents(document)

@tool
def rag_tool(query:str) -> str:
        """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """ 
        formated_content=[]
        answers=main_retreiver.invoke(query)
        for content in answers:
            formated_content.append(content.page_content)
        return "\n\n".join(formated_content)


