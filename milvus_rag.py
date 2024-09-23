# -*- coding: utf-8 -*-
"""Milvus_RAG.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JZ4ANw2HB2CgsCIxmlJ7UGCkoOojdDtF

### https://milvus.io/docs/build-rag-with-milvus.md
"""

"""### Load sample docs to drive"""
# !wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
# !unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
# !cp -r /content/milvus_docs /content/drive/MyDrive/RAG

from glob import glob
import getpass
import os
# from google.colab import userdata
# from posix import device_encoding
import getpass
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
import getpass
import os
# from google.colab import userdata
# from posix import device_encoding
import getpass
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tqdm import tqdm
import json


def read_docs(): # TODO: docs can be changed based on input
    text_lines = []

    # for file_path in glob("/content/drive/MyDrive/RAG/milvus_docs/en/faq/*.md", recursive=True):
    #     with open(file_path, "r") as file:
    #         file_text = file.read()

    with open("/content/drive/MyDrive/RAG/sc_sb.txt", "r") as file:
        file_text = file.read()
        text_lines += file_text.split("| ")
    
    return text_lines

"""### Embedding model"""

# Embedding Function
def emb_text(input_texts):

    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    input_texts = ["This is a sample text, it can also be in a list form. "]
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
    model = AutoModel.from_pretrained("thenlper/gte-base")
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return embeddings[0].tolist()

# def testing():
#     test_embedding = emb_text("This is a test")
#     embedding_dim = len(test_embedding)

"""### Create Milvus Collection"""

# If collection already existed, delete it and recreate
def milvus():
    milvus_client = MilvusClient(uri="./rag.db")
    collection_name = "my_rag_collection"

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=512,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )

    data = []

    text_lines = read_docs()

    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": emb_text(line), "text": line})

    milvus_client.insert(collection_name=collection_name, data=data)
    return milvus_client, collection_name

    
"""### Testing"""

def milvus_querying(question):
    milvus_client, collection_name = milvus()

    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[ emb_text(question)],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
    print(json.dumps(retrieved_lines_with_distances, indent=4))

    return retrieved_lines_with_distances

"""### Setup LLM

"""
def intergrate_LLM(question):
    os.environ["userdata.get('NVIDIA_API_KEY')"] = os.getenv('NVIDIA_API_KEY')
    # TODO: need to be able to change question
    context = "\n".join([line_with_distance[0] for line_with_distance in milvus_querying(question)])
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_AP_KEY')

    client = ChatNVIDIA(
        model="databricks/dbrx-instruct",
        api_key = os.getenv('NVIDIA_API_KEY'),
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
    )
    return context, client

def LLM(question):
    context, client = intergrate_LLM(question)

    SYSTEM_PROMPT = """
                    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
                    """
    USER_PROMPT = f"""
                    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
                    <context>{context}</context>
                    <question>{question}</question>
                    """

    result = []
    for chunk in client.stream([{"role":"user", "content":USER_PROMPT}]):
        result.add(chunk.content, end="")

    return result



if __name__ == "__main__":
    #testing, can comment out later
    print("START TESTING: ")
    milvus_querying("Hallo")

    question = input("Enter the question: ")
    result = LLM(question)
    print(result)