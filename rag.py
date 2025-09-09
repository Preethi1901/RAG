from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from bs4 import BeautifulSoup
import os
from langchain_openai import OpenAIEmbeddings 
load_dotenv()

model=ChatPerplexity(model="sonar")

os.environ["OPENAI_API_KEY"] = "your_openai_key_here"  
loader=WebBaseLoader("https://www.educosys.com/course/genai")

docs=loader.load()      #loader.load() returns a list of Document objects (each has page_content and metadata).

splitters=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

chunks=splitters.split_documents(docs)


vectordb=Chroma.from_documents(     #Chroma.from_documents(...) creates an in-memory Chroma collection and indexes those embeddings.
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

retriever=vectordb.as_retriever()

from langchain import hub
prompt=hub.pull("rlm/rag-prompt")

from langchain.schema.runnable import RunnablePassthrough ,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):      #format_docs converts a list of Document objects to a single string (the context passed to the prompt).
    return "\n".join(doc.page_content for doc in docs)


def print_prompt(text):
    print("-----Prompt to LLM-----")
    return text

rag_chain_with_print=({'context':retriever | format_docs,'question':RunnablePassthrough()}
           | prompt
           | RunnableLambda(print_prompt)   #RunnableLambda wraps a Python callable into a LangChain Runnable.
           | model
           | StrOutputParser()
)
result= rag_chain_with_print.invoke("are the recordings of the course available?")
print(result)
#https://smith.langchain.com/hub/rlm/rag-prompt
#print(chunks[0])
print(len(chunks))
#print(vectordb._collection.count())
#print(vectordb._collection.get())
#print(vectordb._collection.get(ids=[''],include=['embeddings','documents']))
