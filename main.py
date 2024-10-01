import milvus_rag
import pdf_to_txt


if __name__ == "__main__":
    txt_path = pdf_to_txt.run()
    question = input("Enter the question: ")
    milvus_rag.call_func(txt_path, question)


