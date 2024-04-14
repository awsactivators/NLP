import gradio as gr
from pypdf import PdfReader
import docx
import os
import re

MAX_FILE_SIZE = 10000000


class TextExtractor:
    def __init__(self, doc_location: str):
        if doc_location is None:
            raise Exception(f"Please select a PDF to summarize")
        self.doc_location = doc_location

    def preprocess_text(self):
        self.text = re.sub(r"http\S+", "", self.text)
        self.text = re.sub(r"\[IMAGE:.*?\]", "", self.text)
        self.text = re.sub(r"\n", " ", self.text)
        self.text = re.sub(r"\s+", " ", self.text)

    def extract_text_from_pdf(self):
        reader = PdfReader(self.doc_location)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

        self.text = text

    def extract_text_from_doc(self):
        doc = docx.Document(self.doc_location)
        text = ""

        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        self.text = text

    def extract_text_from_txt(self):
        with open(self.doc_location, "r", encoding="utf-8") as file:
            text = file.read()
        self.text = text

    def get_text(self):
        file_extension = os.path.splitext(self.doc_location)[1]
        if file_extension == ".pdf":
            self.extract_text_from_pdf()
        elif file_extension == ".txt":
            self.extract_text_from_txt()
        elif file_extension == ".docx" or file_extension == ".doc":
            self.extract_text_from_doc()
        else:
            raise gr.Error(f"We only support .pdf, .txt, .doc and .docx files")

        self.preprocess_text()

        if len(self.text) > MAX_FILE_SIZE:
            raise gr.Error(
                f"Document exceeds the maximum supported size of {MAX_FILE_SIZE} characters."
            )

        words = self.text.split()
        self.word_count = len(words)

        return self.text, self.word_count
