import loralib as lora
import torch
import torch.nn as nn
from transformers import MT5EncoderModel
from transformers.modeling_outputs import BaseModelOutput


class Args():
    def __init__(self, encoder_name, sizes_mlp, hidden_act, dropout_coef, 
                 need_lora, output_act, loss_fc):
        self.encoder_name = encoder_name
        self.sizes_mlp = sizes_mlp
        self.hidden_act = hidden_act
        self.dropout_coef = dropout_coef
        self.need_lora = need_lora
        self.output_act = output_act
        self.loss_fc = loss_fc


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class MT0Regressor(nn.Module):
    def __init__(self, config):
        super(MT0Regressor, self).__init__()

        self.llm = MT5EncoderModel.from_pretrained(
            config.encoder_name, output_attentions=True,
            output_hidden_states=True)

        dropout_coef = config.dropout_coef

        self.dropout_input = nn.Dropout(dropout_coef)
        layers = []
        for i in range(len(config.sizes_mlp) - 1):
            if config.need_lora:
                layers.append(lora.Linear(config.sizes_mlp[i],
                                          config.sizes_mlp[i + 1], r=16))
            else:
                layers.append(nn.Linear(config.sizes_mlp[i],
                                        config.sizes_mlp[i + 1]))
            if i < len(config.sizes_mlp) - 2:
                layers.append(config.hidden_act())

        layers.append(nn.Dropout(dropout_coef))

        self.mlp = nn.Sequential(*layers)
        self.output_act = config.output_act()

        self.loss_fc = config.loss_fc()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(
            outputs.last_hidden_state, outputs.attentions[-1][:, 0, :, 0]
        )

        outputs_sequence = self.dropout_input(embeddings)

        logits = self.output_act(self.mlp(outputs_sequence)) * 100

        loss = None
        if labels is not None:
            loss = self.loss_fc(logits.view(-1, 1),
                                labels.view(-1).unsqueeze(1))

        return (
            BaseModelOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ),
            logits,
            loss,
        )