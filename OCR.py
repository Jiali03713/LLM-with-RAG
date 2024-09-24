import PyPDF2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from contextlib import contextmanager
import pdfplumber

def save_text_to_file(text, file_name):
    with open(file_name + ".txt", 'w', encoding='utf-8') as file:
        file.write(text)


# Extract text with PyPDF2
# Not used, using pdfPlumber
def extract_text_pypdf2(pdf_path):
    output_folder = "pdfReader_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_text_to_file(text, os.path.join(output_folder, file_name))
    
    print(f"Text successfully extracted and saved to {os.path.join(output_folder, file_name)}.txt")
    return text

# Extract text with pdfPlumber
def extract_text_pdfplumber(pdf_path):
    output_folder = "pdfReader_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_text_to_file(text, os.path.join(output_folder, file_name))
    
    print(f"Text successfully extracted and saved to {os.path.join(output_folder, file_name)}.txt")
    return text


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
            extract_text_pdfplumber(pdf_file)
        else:
            print("No file selected or invalid file.")

if __name__ == "__main__":
    select_pdf()
