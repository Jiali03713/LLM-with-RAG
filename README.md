# Stage One (Current)
1. RAG File
   Large Language Model: "databricks/dbrx-instruct": https://huggingface.co/databricks/dbrx-instruct
                        Nvidia Client: https://build.nvidia.com/databricks/dbrx-instruct
   Vector Database: Milvus: https://milvus.io/
                        Embedding model: https://huggingface.co/thenlper/gte-base
3. pdf_to_txt File
   
# Stage Two
1. Improve Web Crawling file
2. Creating pdf reader using OCR
   - accept uploaded pdf
   - read using EasyOCR
   - store results in files, preferably one file for each pdf
4. RAG File supports recursive question and answer

# Stage Three
1. Combine RAG with pdf reader
2. Support LLM long term memory
   - Add QA history to another storage
   - Combine answers with history
   
# Cleanup implementatio and Optimize
1. optimization, speed and memory

# Deployment?
   
