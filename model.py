from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import LayoutLMModel, LayoutLMPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput as QuestionAnsweringModelOutputBase

@dataclass
class QuestionAnsweringModelOutput(QuestionAnsweringModelOutputBase):
    token_logits: Optional[torch.FloatTensor] = None

class LayoutLMForQuestionAnswering(LayoutLMPreTrainedModel):
    def __init__(self, config, has_visual_segment_embedding=True):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlm = LayoutLMModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # NOTE: We have to use getattr() here because we do not patch the LayoutLMConfig
        # class to have these extra attributes, so existing LayoutLM models may not have
        # them in their configuration.
        self.token_classifier_head = None
        if getattr(self.config, "token_classification", False):
            self.token_classifier_head = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        token_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        token_logits = None
        if getattr(self.config, "token_classification", False):
            token_logits = self.token_classifier_head(sequence_output)

            if token_labels is not None:
                token_logits_reshaped = torch.movedim(token_logits, source=token_logits.ndim - 1, destination=1)
                token_loss = nn.CrossEntropyLoss(reduction=self.config.token_classifier_reduction)(
                    token_logits_reshaped, token_labels
                )

                total_loss += self.config.token_classifier_constant * token_loss

        if not return_dict:
            output = (start_logits, end_logits)
            if self.token_classification:
                output = output + (token_logits,)

            output = output + outputs[2:]

            if total_loss is not None:
                output = (total_loss,) + output

            return output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            token_logits=token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )