
import tempfile
from PIL import Image

import torch
from model import LayoutLMForQuestionAnswering

from document import apply_ocr, _generate_document_output

from transformers import AutoConfig, AutoTokenizer
from transformers import pipeline as transformers_pipeline


def load_model():
    config = AutoConfig.from_pretrained("impira/layoutlm-document-qa", revision="ff904df")

    tokenizer = AutoTokenizer.from_pretrained(
            "impira/layoutlm-document-qa",
            revision="ff904df",
            config=config)

    model = LayoutLMForQuestionAnswering.from_pretrained('impira/layoutlm-document-qa', config=config, revision='ff904df')

    device = 0 if torch.cuda.is_available() else -1

    return transformers_pipeline(
        "document-question-answering",
        revision="ff904df",
        model=model,
        tokenizer=tokenizer,
        device=device)


nlp = load_model()
image = Image.open("img.jpg")


words, boxes = apply_ocr(image)
d = _generate_document_output(image, [words], [boxes])


print(nlp(question="What is the invoice number", **d))