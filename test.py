from PIL import Image

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from model import LayoutLMForQuestionAnswering

from document import apply_ocr, _generate_document_output
from ext.reg import AutoModelForDocumentQuestionAnswering
from ext.document_qa import DocumentQuestionAnsweringPipeline

from transformers import AutoConfig, AutoTokenizer
from transformers import pipeline as transformers_pipeline
from transformers.pipelines import PIPELINE_REGISTRY


def load_model():

    PIPELINE_REGISTRY.register_pipeline(
    "document-question-answering",
    pipeline_class=DocumentQuestionAnsweringPipeline,
    pt_model=AutoModelForDocumentQuestionAnswering)

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

def run_model(img,ques):
	nlp = load_model()
	image = Image.open(img)
	words, boxes = apply_ocr(image)
	d = _generate_document_output(image, [words], [boxes])
	k=nlp(question=ques, **d)
	ans=k[0]['answer']
	return ans
