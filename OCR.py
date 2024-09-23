# Check this out when you find time
# https://stackoverflow.com/questions/23595497/best-way-to-allow-users-to-upload-and-download-files-across-multiple-platforms

# import easyocr
import PyPDF2
# reader = easyocr.Reader(['ch_tra', 'en'])

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


if __name__ == "__main__":
    # Open a file dialog to select the PDF file
    Tk().withdraw()  # Hide the main Tkinter window
    pdf_file = askopenfilename(title="Select PDF file", filetypes=[("PDF files", "*.pdf")])
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        print(text)
