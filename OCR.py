# Check this out when you find time
# https://stackoverflow.com/questions/23595497/best-way-to-allow-users-to-upload-and-download-files-across-multiple-platforms

# import easyocr
import PyPDF2
# reader = easyocr.Reader(['ch_tra', 'en'])
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def save_text_to_file(text, file_name):
    with open(file_name+".txt", 'w') as file:
        file.write(text)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path, ):

    # store output into a different folder
    output_folder = "pdfReader_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the content of a pdf
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    # save to dict
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_text_to_file(text, output_folder+ "/"+file_name)

    return text


if __name__ == "__main__":
    # Open a file dialog to select the PDF file
    Tk().withdraw()  # Hide the main Tkinter window
    pdf_file = askopenfilename(title="Select PDF file", filetypes=[("PDF files", "*.pdf")])
    print(pdf_file+"_____________________")
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        print(text)
