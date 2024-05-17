import os
import time
import arxiv
import torch
from gritlm import GritLM
from langchain_core.prompt_values import ChatPromptValue
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables.base import Runnable
from langchain.load import dumps, loads


# Create directory if not exists
dirpath = "arxiv_papers"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Search arXiv for papers related to "Thermodynamics"
client = arxiv.Client()
search = arxiv.Search(
    query="Thermodynamics",
    max_results=10,
    sort_order=arxiv.SortOrder.Descending
)

# Download and save the papers
for result in client.results(search):
    while True:
        try:
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
            break
        except (FileNotFoundError, ConnectionResetError) as e:
            print("Error occurred:", e)
            time.sleep(5)

# Load papers from the directory
papers = []
loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
try:
    papers = loader.load()
except Exception as e:
    print(f"Error loading file: {e}")
print("Total number of pages loaded:", len(papers)) 

# Concatenate all pages' content into a single string
full_text = ''
for paper in papers:
    full_text += paper.page_content

# Remove empty lines and join lines into a single string
full_text = " ".join(line for line in full_text.splitlines() if line)
print("Total characters in the concatenated text:", len(full_text)) 

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
paper_chunks = text_splitter.create_documents([full_text])

# Create Qdrant vector store
qdrant = Qdrant.from_documents(
    documents=paper_chunks,
    embedding=GPT4AllEmbeddings(),
    path="./tmp/local_qdrant",
    collection_name="arxiv_papers",
)
retriever = qdrant.as_retriever()

# # Define prompt template
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# Initialize Ollama LLM
ollama_llm = "llama2:7b-chat"
model = Ollama(model=ollama_llm, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# # Define the processing chain
# chain = (
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#     | prompt
#     | model
#     | StrOutputParser()
# )

# Add typing for input
class Question(BaseModel):
    __root__: str

question = "what is Gibbs free energy?"


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. please generate four 
different versions of the given user question. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)


# class GPT2Generation(Runnable):
#     def __init__(self, model_name_or_path="gpt2"):
#         super().__init__()
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
#         self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)



class GRITGeneration(Runnable):
    def __init__(self, model_name_or_path="grit-model-name"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("GritLM/GritLM-7B")
        self.model = AutoModelForCausalLM.from_pretrained("PrunaAI/GritLM-GritLM-7B-bnb-8bit-smashed",trust_remote_code=True, device_map='auto')
        self.model.to(torch.device("cpu")) 
    def invoke(self, input_data):
        if isinstance(input_data, dict) and "question" in input_data:
            input_data = input_data["question"]
        input_ids = self.tokenizer(input_data, return_tensors='pt').to(torch.device("cpu"))["input_ids"]
        # input_encoded = self.model.encode(input_data, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=1000)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output
    


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
generate_queries = GRITGeneration()
query_outputs = generate_queries.invoke({"question": question})
# if not query_outputs:
#     raise ValueError("No queries returned from GenerateQueries")

# retrieval_chain = generate_queries | retriever.map() | get_unique_union
retrieval_chain = generate_queries | retriever.map() 
retrieved_docs = retrieval_chain.invoke({"question": query_outputs})

if not retrieved_docs:
    print("No documents retrieved.")
else:
    print(f"Number of documents retrieved: {len(retrieved_docs)}")


from operator import itemgetter
# from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# RAG

template2 = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template2)


if retrieved_docs:
    final_rag_chain = (prompt | model | StrOutputParser())
    result = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
    print(result)
else:
    print("No documents to process in RAG.")