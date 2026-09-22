import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr
import glob


MODEL = "gemini-2.0-flash"
DB_NAME = "vector_db"
load_dotenv(override=True)


# define the knowledgebase
# Load in everything in the knowledgebase using LangChain's loaders
folders = glob.glob("knowledge-base/*")

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

# Divide into chunks using the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


# define the Embedding Model
# this generates the vector embeddings for the queries and knowledge base chunks
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# define the vector store for vector embeddings
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()


# define the auto-regressive llm
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)


# print(retriever.invoke("Who is avery?"))

SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the the given context to answer the question.
If you don't know the anser, say so.

Context:
{context}
"""

# RAG function
def rag_answer_question(question: str, history):
    # fetch the relevant context
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    # fetch llm response by providing rag context
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content

res = rag_answer_question("who is avery?", [])
print(res)
