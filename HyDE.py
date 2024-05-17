import os
import time
import arxiv
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables.base import Runnable


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


# # Apply input type to the chain
# chain = chain.with_types(input_type=Question)
# result = chain.invoke("Explain Gibbs free energy")
# print(result)

question = "what is Gibbs free energy?"

# HyDE document genration
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

class GPT2Generation(Runnable):
    def __init__(self, model_name_or_path="gpt2"):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# class GPT2Generation(Runnable):
#     def __init__(self, model_name_or_path="gpt2"):
#         super().__init__()
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
#         self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    def invoke(self, input_data):
        if isinstance(input_data, dict) and "question" in input_data:
            input_data = input_data["question"]
        input_ids = self.tokenizer.encode(input_data, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=100)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output



# generate_docs_for_retrieval = (
#     prompt_hyde 
#     | GPT2Generation()  
#     | StrOutputParser()
# )

class GenerateDocsForRetrieval(Runnable):
    def __init__(self):
        super().__init__()
        self.gpt2_generation = GPT2Generation()
        self.str_output_parser = StrOutputParser()

    def invoke(self, input_data, *args, **kwargs):
        # Handle additional arguments if necessary or ignore them
        if isinstance(input_data, dict) and 'question' in input_data:
            question = input_data['question']
        else:
            raise ValueError("Expected input_data to be a dictionary with a 'question' key")

        prompt_output = prompt_hyde.invoke({"question": question})
        if hasattr(prompt_output, 'content'):
            input_text = prompt_output.content
        else:
            input_text = str(prompt_output)

        gpt2_output = self.gpt2_generation.invoke(input_text)
        parsed_output = self.str_output_parser.invoke(gpt2_output)
        return parsed_output
    
# prompt_output = prompt_hyde.invoke({"question":question})
# if hasattr(prompt_output, 'content'):
#     input_text = prompt_output.content
# else:
#     input_text = str(prompt_output)
# # Ensure this is what GPT2Generation expects, if not, adapt it
# gpt2_output = GPT2Generation().invoke(input_text)
# generate_docs_for_retrieval = StrOutputParser().invoke(gpt2_output)



# generate_docs_for_retrieval = (
#     prompt_hyde 
#     | GPT2Generation()  
#     | StrOutputParser()
# )

# Run
# generate_docs_for_retrieval=generate_docs_for_retrieval.with_types(input_type=Question)
generate_docs_for_retrieval = GenerateDocsForRetrieval()
generate_docs_for_retrieval.invoke({"question":question})

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever 
retireved_docs = retrieval_chain.invoke({"question":question})

# RAG
template2 = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template2)

final_rag_chain = (
    prompt
    | model
        | StrOutputParser()
    )

result=final_rag_chain.invoke({"context":retireved_docs,"question":question})
print(result)