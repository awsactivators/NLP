import gradio as gr
from transformers import pipeline
from pypdf import PdfReader
import docx
import os
from langdetect import detect

summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")
translator_to_french = pipeline(
    task="translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr"
)
translator_to_english = pipeline(
    task="translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en"
)

MAX_FILE_SIZE = 10000000

class TextExtractor:
    def __init__(self, doc_location: str):
        if doc_location is None:
            raise Exception(f"Please select a PDF to summarize")
        self.doc_location = doc_location

    def extract_text_from_pdf(self):
        reader = PdfReader(self.doc_location)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

        return text

    def extract_text_from_doc(self):
        doc = docx.Document(self.doc_location)
        text = ""

        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def extract_text_from_txt(self):
        with open(self.doc_location, "r", encoding="utf-8") as file:
            text = file.read()
        return text

    def extract_text_from_txt(self):
        with open(self.doc_location, "r", encoding="utf-8") as file:
            text = file.read()
        return text
    
    def text_length(self):
        words = self.text.split()
        num_words = len(words)
        return num_words

    def get_text(self) -> str:
        file_extension = os.path.splitext(self.doc_location)[1]
        if file_extension == ".pdf":
            self.text = self.extract_text_from_pdf()
        elif file_extension == ".txt":
            self.text = self.extract_text_from_txt()
        elif file_extension == ".docx" or file_extension == ".doc":
            self.text = self.extract_text_from_doc()
        else:
            raise gr.Error(f"We only support .pdf, .txt, .doc and .docx files")

        if len(self.text) > MAX_FILE_SIZE:
            raise gr.Error(
                f"Document exceeds the maximum supported size of {MAX_FILE_SIZE} characters."
            )

        return self.text




def summarize(doc: str, target_language: str) -> str:
    text_extractor = TextExtractor(doc)
    text = text_extractor.get_text()

    text_length = text_extractor.text_length()
    summary_length = int(text_length / 2)

    summary = summarizer(text, max_length=summary_length, do_sample=False)[0]["summary_text"]
    detected_lang = detect(summary)

    if target_language is None:
        pass
    elif detected_lang == "fr" and str(target_language).lower() == "english":
        summary = translator_to_english(summary)[0]["translation_text"]
    elif detected_lang == "en" and str(target_language).lower() == "french":
        summary = translator_to_french(summary)[0]["translation_text"]

    return summary


app = gr.Interface(
    fn=summarize,
    inputs=[
        gr.File(
            label="Document to summarize",
            file_types=["pdf", "docx", "doc", "txt", "odt", "dot", "dotx"],
        ),
        gr.Radio(
            label="Translate summary to", choices=["English", "French"], value="English"
        ),
    ],
    outputs=gr.Textbox(label="Summary"),
    examples=[
        ["data/pd-file-example.pdf"],
        ["data/doc-file-example.docx"],
        ["data/text-file-example.txt"],
    ],
)

if __name__ == "__main__":
    app.launch()
