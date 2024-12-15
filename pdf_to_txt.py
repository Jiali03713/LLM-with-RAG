import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from contextlib import contextmanager
import pdfplumber

def save_text_to_file(texts, file_name):
    if os.path.exists(file_name + ".txt"):
        os.remove(file_name + ".txt")

    with open(file_name + ".txt", 'a', encoding='utf-8') as file:
        for text in texts:
            file.write(text + "\n")

    return file_name + ".txt"


# Extract text with pdfPlumber
def extract_text_pdfplumber(pdf_path):
    output_folder = "pdfReader_output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            count = 0
            for page in pdf.pages:
                count += 1
                page_content = page.extract_text().replace("\n", " ")
                text += page_content + " "

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    text_list = text.split(".")
    text_list = [element for element in text_list if element and isinstance(element, str) and len(element) > 10]

    save_text_to_file(text_list, os.path.join(output_folder, file_name))

    print(f"Text successfully extracted and saved to {os.path.join(output_folder, file_name)}.txt")
    output_path = os.path.join(output_folder, file_name) + ".txt"

    return file_name, output_path


@contextmanager
def use_tkinter():
    root = Tk()
    root.withdraw()
    try:
        yield root
    finally:
        root.quit()

def select_pdf():
    with use_tkinter():
        pdf_file = askopenfilename(title="Select PDF file", filetypes=[("PDF files", "*.pdf")])

        if pdf_file:
            output_path = extract_text_pdfplumber(pdf_file)
            return output_path
        else:
            print("No file selected or invalid file.")
        

def run(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = "pdfReader_output"

    output_path = f"{os.path.join(output_folder, file_name)}.txt"

    if os.path.exists(output_path):
        return file_name, output_path
    else:
        file_path = input("Enter the path of your pdf file: ")
        filename, output_path = extract_text_pdfplumber(file_path)
        return filename, output_path

if __name__ == "__main__":
    output_path = run()
    # print(output_path)
