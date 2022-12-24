import streamlit as st
import tempfile
from PIL import Image

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from model import LayoutLMForQuestionAnswering

import document
from ext.reg import AutoModelForDocumentQuestionAnswering
from ext.document_qa import DocumentQuestionAnsweringPipeline

from transformers import AutoConfig, AutoTokenizer
from transformers import pipeline as transformers_pipeline
from transformers.pipelines import PIPELINE_REGISTRY

nlp=None

def load_model():

    PIPELINE_REGISTRY.register_pipeline(
    "document-question-answering",
    pipeline_class=DocumentQuestionAnsweringPipeline,
    pt_model=AutoModelForDocumentQuestionAnswering, 
    )

    config = AutoConfig.from_pretrained("impira/layoutlm-document-qa", 
    revision="ff904df")

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

st.header("`Problem Statement 4 by Mukham`")

file = st.file_uploader(label="Upload your invoice.",
                        type=['jpg', 'jpeg', 'png'])
if not file:
    st.warning("Please upload the file.")
    st.stop()
else:
    submit = st.button("Submit")

if submit:
    f = file.read()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f)
    
    nlp = nlp if nlp!=None else load_model()
    
    image = Image.open(tfile.name)
    st.image(f, use_column_width=True)

    words, boxes = document.apply_ocr(image=image)
    doc = document._generate_document_output(image, [words], [boxes])

    res1 = nlp(question="What is the invoice number?", **doc)[0]
    res2 = nlp(question="What is the invoice date?", **doc)[0]
    st.success(f"**Invoice Number:** {res1['answer']}\n\n**Invoice Date:** {res2['answer']}")
