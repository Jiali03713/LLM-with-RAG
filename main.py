import milvus_rag
import pdf_to_txt
import os

if __name__ == "__main__":
    if os.getenv("NVIDIA_API_KEY") is None:
        NVIDIA_API_KEY = input("Add your NVIDIA key here: ")
        os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY

    # if user want to input new pdf
    new_pdf = input("Do you want to input a new pdf[Y/N]: ")
    if new_pdf.lower()== "y":
        txt_path = pdf_to_txt.run()
        milvus_rag.run_new(txt_path)

    elif new_pdf.lower()=="n":
        collection_name = input("Enter your collection name: ")
        milvus_rag.run_old(collection_name)
    
    else:
        # TODO implement resursive call
        print("Invalid input. Better luck next time. Bye.")


    #TODO: user what to query existing collection

    # milvus_client = input("Enter the milvus client name: ")
    # collection_name = input("Enter the milvus collection name: ")


