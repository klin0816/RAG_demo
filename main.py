import os

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

loader = PyPDFLoader("./teeth.pdf")
loader.load()[0]
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
doc_split = loader.load_and_split(splitter)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=doc_split,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

client = OpenAI(
    api_key=os.getenv("API_KEY"),  # This is the default and can be omitted
)

template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)
pdf_qa = ConversationalRetrievalChain.from_llm(
    llm=client,
    condense_question_prompt=prompt,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    verbose=False,
)

result = pdf_qa.invoke(
    {"question": "牙周病該怎麼治療? 解釋後並用英文回答", "chat_history": []}
)
print(result)
