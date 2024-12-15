import milvus_rag
import pdf_to_txt
import os

if __name__ == "__main__":
    if os.getenv("NVIDIA_API_KEY") is None:
        NVIDIA_API_KEY = input("Add your NVIDIA key here: ")
        os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY

    # if user want to input new pdf
    file_path = input("Enter you file path: ")
    file_extension = os.path.splitext(os.path.basename(file_path))[1]
    filename = os.path.splitext(os.path.basename(file_path))[0]
    print(file_extension)
    if file_extension == '.pdf':
        filename, txt_path = pdf_to_txt.run(file_path)
    else:
        filename  = filename
        txt_path = file_path

    milvus_rag.run(filename, txt_path)
    
    # else:
    #     # TODO implement resursive call
    #     print("Invalid input. Better luck next time. Bye.")


    #TODO: user what to query existing collection

    # milvus_client = input("Enter the milvus client name: ")
    # collection_name = input("Enter the milvus collection name: ")


