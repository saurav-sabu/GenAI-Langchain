# Import necessary modules from langchain and src.helper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.llms import CTransformers
from src.helper import *

# Define the path to the PDF file
file_path = "data\SDG.pdf"

# Load the PDF document
loader = PyPDFLoader(file_path)
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                  model_kwargs={"device":"cpu"})

# Create a FAISS vector store from the text chunks and embeddings
vector_store = FAISS.from_documents(text_chunks,embedding)

# Initialize the language model
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  # Path to the model file
                    model_type="llama",
                    config={
                        "max_new_tokens":128, # Maximum number of new tokens to generate
                        "temperature":0.3
                    })

# Define the question answering prompt
qa_prompt = PromptTemplate(template=QUESTION_ANSWERING_PROMPT,input_variables=["context","question"])


# Create a RetrievalQA chain with the specified components
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever = vector_store.as_retriever(search_kwargs={"k":2}),
                                    return_source_documents=False,
                                    chain_type_kwargs={"prompt":qa_prompt})

# Define the user's input query
input_user = "Give me summary about this paper"

# Run the chain with the user's input and print the response
response = chain.run({"query":input_user})
print(response)