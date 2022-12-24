from collections import OrderedDict
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
        ("donut-swin", "DonutSwinModel"),
    ]
)

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
)

class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING