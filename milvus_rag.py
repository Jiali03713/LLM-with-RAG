from glob import glob
import torch.nn.functional as F
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

import LLM
import RAG
load_dotenv() 

class txt_reader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_docs(self): # TODO: docs can be changed based on input
        with open(self.file_path, "r", encoding = 'utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    

def run_new(file_path):
    reader = txt_reader(file_path=file_path)
    raw_text = reader.read_docs()

    question = input("Enter the question: ")
    gte = RAG(raw_text=raw_text, query=question)
    returned_vector = gte.milvus_query(new_collection=True)

    llm = LLM(returned_vectors=returned_vector)
    return llm.LLM(question)


def run_old(collection_name):
    question = input("Enter the question: ")
    gte = RAG(raw_text="", query=question)
    returned_vector = gte.milvus_query(new_collection=False, collection_name = collection_name)

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
