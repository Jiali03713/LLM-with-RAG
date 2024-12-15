
import torch
import os
import getpass
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from torch.cuda.amp import autocast
from pymilvus import MilvusClient

  # Embedding Function "thenlper/gte-base"
    # TODO: can be changed to other model to support
    # multilingual


class RAG():
    def __init__(self, raw_text, query):
        self.raw_text = raw_text
        self.question = query
        self.client_name = None
        self.collection_name = None


    def emb_text(self, input_texts): 
        """
        This function takes the raw texts from .txt file and        
        using embedding model to embed. 
        Using huggingface model: thenlper/gte-base

        Args:
            String input_texts: raw texts from read_docs()

        Returns:
            List Embeded texts: embedded texts
        """

        def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
        model = AutoModel.from_pretrained("thenlper/gte-base")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

        batch_dict = {key: value.to("cuda") for key, value in batch_dict.items()}

        with autocast():
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return embeddings[0].tolist()

    def emb_text_batch(self, lines):
        with torch.no_grad():
            embeddings = self.emb_text(lines)
        return embeddings

    def milvus_setup(self, client_name, collection_name):
        self.client_name = client_name
        self.collection_name = collection_name

        milvus_client = MilvusClient(uri=self.client_name)
        collection_name = self.collection_name

        if milvus_client.has_collection(collection_name):
            milvus_client = milvus_client
        else:
            milvus_client.create_collection(
                collection_name=collection_name,
                dimension=768,
                metric_type="IP",  # Inner product distance
                consistency_level="Strong",  # Strong consistency level
       	        params = {'efConstruction': 40, 'M': 1024}
       	    )

            data = []
            batch_size = 128

            text_lines = self.raw_text

            for i in tqdm(range(0, len(text_lines), batch_size), desc="Creating embeddings"):
                batch_lines = text_lines[i:i + batch_size]  # Get a batch of lines
                embeddings = self.emb_text_batch(batch_lines)  # Get embeddings for the batch
                for j, line in enumerate(batch_lines):
                    data.append({"id": i + j, "vector": embeddings, "text": line})


            milvus_client.insert(collection_name=collection_name, data=data)

        return milvus_client, collection_name

    def milvus_query(self, collection_name = None, client_name = None):
        """
        This function query the existing database
        created by previous function milvus()

        Args:
            Raw text of query question

        Returns:
            Raw text with distance
            for RAG context
        """

        """ Create new embedding"""

        milvus_client, collection_name = self.milvus_setup(client_name=client_name, collection_name=collection_name)

        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[self.emb_text(self.question)],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=5,  # Return top 3 results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )

        retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
        #print(json.dumps(retrieved_lines_with_distances, indent=4))
        return retrieved_lines_with_distances
