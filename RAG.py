
import torch
import os
import getpass
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from torch.cuda.amp import autocast
  # Embedding Function "thenlper/gte-base"
    # TODO: can be changed to other model to support
    # multilingual


class RAG():
    def __init__(self, raw_text, query):
        self.raw_text = raw_text
        self.question = query
        self.client_name = "./rag.db"
        self.collection_name = "one_hundred_collection"


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

        # move model to CUDA when available
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

    def milvus_setup(self):
        """
        This function 
            1. create milvus database
            2. Embed each line of raw text using embedding model
            3. add embedded line to database    
        If database need to be changed, change next function:
        milvus_querying()

        Args:
            None

        Returns:
            milvus_client:
            collection_name:
        """

        from pymilvus import MilvusClient

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

    def milvus_query(self, new_collection = False, collection_name = None):
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
        if new_collection:
            milvus_client, collection_name = self.milvus_setup()
        else:     
            from pymilvus import MilvusClient
            milvus_client = MilvusClient(uri="./rag.db")
            collection_name = collection_name


        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[self.emb_text(self.question)],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=64,  # Return top 3 results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )

        retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
        #print(json.dumps(retrieved_lines_with_distances, indent=4))
        return retrieved_lines_with_distances


  #  def pinecone_setup():
        from pinecone import Pinecone, ServerlessSpec

        index_name = 'gen-qa-openai-fast'
        pc = Pinecone(api_key = os.getenv('PINECONE_API_KEY'))

        cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
        region = os.environ.get('PINECONE_REGION') or 'us-east-1'

        spec = ServerlessSpec(cloud=cloud, region=region)
        
        
        if index_name not in pc.list_indexes().names():
        # if does not exist, create index
            pc.create_index(
                index_name,
                dimension=1536,  # dimensionality of text-embedding-ada-002
                metric='cosine',
                spec=spec
            )
        
        index = pc.Index(index_name)
        data = []

        text_lines = read_docs()

        for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
            data.append({"id": i, "values": emb_text(line), "metadata": line})

        index.upsert(data)

        return index, data
        
   # def pinecone_query(query):
        limit = 3750
        import time

        embedded_query  = emb_text(query)

        index, data = pinecone_setup()

        # get relevant contexts
        contexts = []
        time_waited = 0

        while (len(contexts) < 3 and time_waited < 60 * 12):
            res = index.query(vector=embedded_query, top_k=3, include_metadata=True)
            contexts = contexts + [
                x['metadata']for x in res['matches']
            ]
            print(f"Retrieved {len(contexts)} contexts, sleeping for 15 seconds...")
            time.sleep(15)
            time_waited += 15

        if time_waited >= 60 * 12:
            print("Timed out waiting for contexts to be retrieved.")
            contexts = ["No contexts retrieved. Try to answer the question yourself!"]


        return contexts

