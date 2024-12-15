# Stage One
1. RAG File
   - Large Language Model:
     - Language Model: "databricks/dbrx-instruct": https://huggingface.co/databricks/dbrx-instruct 
     - Nvidia Client: https://build.nvidia.com/databricks/dbrx-instruct
   - Vector Database:
     - Milvus: https://milvus.io/
     - Embedding model: https://huggingface.co/thenlper/gte-base
     - Support OS: Linux
       - Currently Does not support Windows OS because Milvus_lite does not support Windows OS
       - Will choose different database in the future in order to fix this issue
3. pdf_to_txt File
   - Current Handle:
     - pdf(text) to txt
     - Need to improve preprocessing inorder to feed to RAG model
4. Progress(10/01/2024): **Simplified version works on Linux, with one query ability**
   - (10/02/2024): **Able to reuse collection for query**
   
# Stage Two (Current)
1. Creating pdf reader using OCR
   - accept uploaded pdf
   - read using EasyOCR
   - store results in files, preferably one file for each pdf
2. RAG File supports recursive question and answer
3. Able to store historical QA in corresponding files

# Stage Three
1. Combine RAG with pdf reader
2. Support LLM long term memory
   - Add QA history to another storage
   - Combine answers with history
   
# Cleanup implementatio and Optimize
1. optimization, speed and memory

# Deployment?
   
