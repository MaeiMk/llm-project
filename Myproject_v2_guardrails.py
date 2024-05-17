import os
import time
import arxiv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from nemoguardrails import LLMRails, RailsConfig

model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model')
tokenizer = AutoTokenizer.from_pretrained('vectara/hallucination_evaluation_model')

# Create directory if not exists
dirpath = "arxiv_papers"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Search arXiv for papers related to "Thermodynamics"
client = arxiv.Client()
search = arxiv.Search(
    query="Thermodynamics",
    max_results=2,
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
paper_chunks = text_splitter.create_documents([full_text])

# Create Qdrant vector store
qdrant = Qdrant.from_documents(
    documents=paper_chunks,
    embedding=GPT4AllEmbeddings(),
    path="./tmp/local_qdrant",
    collection_name="arxiv_papers",
)
retriever = qdrant.as_retriever()

# Define prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize Ollama LLM
ollama_llm = "llama2:7b-chat"
model2 = Ollama(model=ollama_llm, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Define the processing chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model2
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str

# Apply input type to the chain
chain = chain.with_types(input_type=Question)
result = chain.invoke("Explain Gibbs free energy")
print(result)

# # Define hallucination config with modified token
# yaml_content = """
# rails:
#   output:
#     flows:
#       - self check hallucination
# """
# colang_content = """
# prompts:
#   - task: self_check_hallucination
#     content: |-
#       You are given a task to identify if the hypothesis is in agreement with the provided data below.
#       Only use the provided data and do not rely on external knowledge.
#       Answer with yes/no. "provided_data": {{ paragraph }} "hypothesis": {{ statement }} "agreement":
# """
# hallucination_config = RailsConfig.from_content(
#     yaml_content=yaml_content,
#     colang_content=colang_content
# )

# # Create hallucination rails
# hallucination_rails = LLMRails(hallucination_config)

# # Set the context variable to trigger hallucination checking
# context = {
#     "paragraph": full_text,  # Pass the retrieved document content
#     "statement": result,     # Pass the LLM2 model output
#     "check_hallucination": True
# }

# # Invoke hallucination rails to check for hallucinations
# hallucination_result = hallucination_rails.invoke(context)

# # Based on the result, take appropriate actions (blocking or warning)
# if hallucination_result:
#     # Blocking mode: Handle the message block
#     print("Output is not trustable and is based on hallucination.")
#     # Implement block logic here

# else:
#     # Warning mode: Send a warning message
#     print("The previous answer is prone to hallucination and may not be 100 percent accurate.")
#     # Send the response back to the user with a warning


# def evaluate_hallucinated_content(hallucinated_content: str) -> float:
#     """
#     Evaluate hallucinated content using the hallucination evaluation model.

#     Args:
#     - hallucinated_content (str): The hallucinated content to evaluate.

#     Returns:
#     - float: Evaluation score indicating quality or relevance.
#     """
#     # # Define pairs containing the hallucinated content and its source
#     # pairs = [[hallucinated_content, full_text]]

#     # # Tokenize the pairs
#     # inputs = tokenizer.batch_encode_plus(pairs, return_tensors='pt', padding=True)

#     # # Forward pass through the model
#     # model.eval()
#     # with torch.no_grad():
#     #     outputs = model(**inputs)
#     #     logits = outputs.logits.cpu().detach().numpy()
#     #     # Convert logits to probabilities using sigmoid function
#     #     probabilities = 1 / (1 + np.exp(-logits)).flatten()

#     # # Extract the relevant probability (probability of being factually consistent)
#     # relevant_probability = probabilities[0]

#     # return relevant_probability

def evaluate_hallucinated_content_chunks(hallucinated_content: str) -> float:
    """
    Evaluate hallucinated content chunks against the entire content and calculate the overall score.

    Args:
    - hallucinated_content (str): The hallucinated content to evaluate.
    - full_text (str): The full source document text.
    - chunk_size (int): The size of each chunk for comparison.

    Returns:
    - float: Overall score based on relevant probabilities.
    """

    relevant_probabilities = []

    # Compare each chunk with the entire hallucinated content
    for chunk in paper_chunks:
        # Define pairs containing the chunk and hallucinated content
        hallucinated_content=str(hallucinated_content)
        chunk=str(chunk)
        pairs = [[hallucinated_content, chunk]]

        # Tokenize the pairs
        inputs = tokenizer.batch_encode_plus(pairs, return_tensors='pt', padding=True)

        # Forward pass through the model
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().detach().numpy()
            # Convert logits to probabilities using sigmoid function
            probabilities = 1 / (1 + np.exp(-logits)).flatten()

        # Extract the relevant probability (probability of being factually consistent)
        relevant_probability = probabilities[0]
        relevant_probabilities.append(relevant_probability)

    # Calculate overall score (e.g., average of relevant probabilities)
    top_20_probabilities = sorted(relevant_probabilities, reverse=True)[:20]
    overall_score = np.mean(top_20_probabilities)

    return overall_score

# Test the function with a hallucinated content
hallucinated_content = result
evaluation_score = evaluate_hallucinated_content_chunks(hallucinated_content)
print(f"Hallucination Evaluation Score: {evaluation_score}")