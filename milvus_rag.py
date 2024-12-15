from glob import glob
import torch.nn.functional as F
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

from LLM import LLM
from RAG import RAG

load_dotenv() 

class txt_reader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_docs(self): # TODO: docs can be changed based on input
        with open(self.file_path, "r", encoding = 'utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    
def milvus_config(filename, file_path):
    client_name = "./rag.db"
    collection_name = filename
    return collection_name, client_name

def run(filename, file_path):
    reader = txt_reader(file_path=file_path)
    raw_text = reader.read_docs()

    collection_name, client_name = milvus_config(filename=filename, file_path=file_path);

    question = input("Enter the question: ")
    gte = RAG(raw_text=raw_text, query=question)
    returned_vector = gte.milvus_query(collection_name=collection_name, client_name=client_name)

    llm = LLM(returned_vectors=returned_vector)
    return llm.LLM(question)



if __name__ == "__main__":
    file_path = input("Enter the absolute file path of your txt file: ")

    """Not yet"""
    # milvus_client = input("Enter the milvus client name: ")
    # collection_name = input("Enter the milvus collection name: ")
    # batch_size = input("Enter the batch size for text embedding: ")


    print(run_old(file_path))
    # TODO: finish orginizing code and include flexibility of parameters
    # TODO: finish call_func
    # TODO: modify OCR?py
