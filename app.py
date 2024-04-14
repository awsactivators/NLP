import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
from text_extractor import TextExtractor

summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

translator_to_french = pipeline(
    task="translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr"
)
translator_to_english = pipeline(
    task="translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en"
)


def summarize(doc: str, target_language: str) -> str:
    text_extractor = TextExtractor(doc)
    text, word_count = text_extractor.get_text()
    summary_length = int(word_count / 2)

    try:
        summary = summarizer(text, max_length=summary_length, do_sample=False)[0][
            "summary_text"
        ]
    except Exception as ex:
        max_length = tokenizer.model_max_length
        inputs = tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt"
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=summary_length,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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
