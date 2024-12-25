from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load documents
loader = TextLoader('text/compatibility_supported_file_formats.txt')
documents = loader.load()

# Create a local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Load a local language model
model_id = "EleutherAI/gpt-neo-1.3B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1000)

# Create a retriever
retriever = vector_store.as_retriever()

# Create a question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=pipe),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
# query = "List up compatibility and supported file formats"
query = "What is compatibility and supported file formats for Transformers?"
result = qa_chain(query)
print(result['result'])
