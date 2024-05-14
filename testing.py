print("loading libraries... \n")
import urllib.request
from bs4 import BeautifulSoup
 
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
 
from googlesearch import search
 
def search_urls(query):
    urls = []
    try:
        for url in search(query, num=3, stop=3, pause=3):
            urls.append(url)
    except Exception as e:
        print("An error occurred:", e)
    return urls
 
def get_text_chunks_langchain(text, source, page):
    docs = [Document(page_content=text, metadata={"url": source})]
    return docs
 
query = input("Enter your search query: ")
search_query = f'"{query}" AND (scandal OR arrest OR lawsuit OR controversy OR bankruptcy OR fraud OR conviction OR crime OR "legal issues" OR "financial problems" OR "ethical concerns")'
 
print("searching the web...\n")
urls = search_urls(search_query)
 
all_doc = []
for idx, url in enumerate(urls, start=1):
    print(f"{idx}. {url}")
    try:
        uf = urllib.request.urlopen(url)
        html_bytes = uf.read()
        html = html_bytes.decode("utf-8")
        soup = BeautifulSoup(html, 'html.parser')
        #print(soup.get_text())
        all_doc = all_doc + get_text_chunks_langchain(soup.get_text(),url,0)
    except Exception as e:
        print("url not accessible or no internet\n")
        print("An error occurred:", e)
 
print("Loading retriever model... \n")
# MODEL_ID = 'BAAI/bge-base-en-v1.5' slower, slightly better
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_ID)
 
print("vectorize document...\n")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
texts = text_splitter.split_documents(all_doc)
 
 
# Initialize an instance of HuggingFaceInstructEmbeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=MODEL_ID,
    model_kwargs={"device": "cpu"}  # Specify the device to be used for inference (GPU - "cuda" or CPU - "cpu", M1-M4 on mac "mps")
)
 
# Define the directory where the embeddings will be stored on disk
persist_directory = 'db-webscrapping'
 
# Create a Chroma instance and generate embeddings from the supplied texts
vectordb = Chroma.from_documents(documents=texts, embedding=instructor_embeddings, persist_directory=persist_directory)
 
## Persist the database (vectordb) to disk
# vectordb.persist()
## Set the vectordb variable to None to release the memory
#vectordb = None # Release RAM
#vectordb = Chroma(persist_directory=persist_directory, embedding_function=instructor_embeddings) # Reload db next on next session # instructor_embeddings has to be the same
 
retriever = vectordb.as_retriever(search_kwargs={"k": 20}) # number of retrieved chunks. it does not affect the speed
retrieved_docs = retriever.invoke(f"what was the worst thing that {query} have done")
 
import re
 
def clean_text(text):
    # Replace sequences of newlines (with or without spaces in between) with a single newline
    text = re.sub(r'(\n\s*)+\n', '\n', text)
    # Optionally, remove multiple consecutive spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text
 
def is_relevant(text):
    # Check if any line in the text is at least 160 characters. if not, potentially not real sentences
    return any(len(line) >= 160 for line in text.split('\n'))
 
# Process each document
for doc in retrieved_docs:
    cleaned_text = clean_text(clean_text(doc.page_content))
    if is_relevant(cleaned_text):
        print(doc.metadata['url'], '\n', cleaned_text, '\n\n')