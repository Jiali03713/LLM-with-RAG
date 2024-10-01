import milvus_rag
import pdf_to_txt


if __name__ == "__main__":
    # if user want to input new pdf
    txt_path = pdf_to_txt.run()
    milvus_rag.run(txt_path)

    #TODO: user what to query existing collection

    # milvus_client = input("Enter the milvus client name: ")
    # collection_name = input("Enter the milvus collection name: ")


