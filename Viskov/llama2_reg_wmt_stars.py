import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import evaluate
import accelerate
from accelerate import DistributedType
from torch import nn
import transformers
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator
import bitsandbytes as bnb
import sys
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from functools import partial
import gc


def get_gemba_stars_prompt(langs, segs):
    source_lang, target_lang = langs
    source_seg, reference_seg, target_seg = segs
    return ' '.join([f'Score the following translation from {source_lang} to {target_lang} with respect to the human reference with one to five stars.',
            'Where one star means "Nonsense/No meaning preserved",',
            'two stars mean "Some meaning preserved, but not understandable",',
            'three stars mean "Some meaning preserved and understandable",',
            'four stars mean "Most meaning preserved with possibly few grammar mistakes",',
            'and five stars mean "Perfect meaning and grammar".\n',
            f'{source_lang} source: "{source_seg}"\n',
            f'{target_lang} human reference: "{reference_seg}"\n',
            f'{target_lang} translation: "{target_seg}"\n',
            'Stars:'])


def regression_tokenize(data, tokenizer, input_key):
    output = tokenizer(data[input_key], truncation=True)
    output['labels'] = data['raw']
    return output


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    r = spearmanr(labels, predictions)
    tau = kendalltau(labels, predictions)
    return {
        'rmse': rmse,
        'R': r,
        'tau': tau
    }

accelerator = Accelerator(gradient_accumulation_steps=8)

wmt_base = load_dataset('RicardoRei/wmt-da-human-evaluation', split='train')

translations = ['en-de', 'en-ru', 'zh-en']
id2langs = {
    'en': 'english',
    'de': 'german',
    'zh': 'chinese'
}

train_dataset = wmt_base.filter(lambda example: (example['year'] != 2022) & (example['lp'] in translations))
test_dataset = wmt_base.filter(lambda example: (example['year'] == 2022) & (example['lp'] in translations))

wmt_datasets = {}
for text_id, dset in zip(['train', 'test'], [train_dataset, test_dataset]):
    prompt_column = []
    for i in tqdm_notebook(range(len(dset))):
        point = dset[i]
        langs = point['lp'].split('-')
        segs = (point['src'], point['ref'], point['mt'])
        prompt_column.append(get_gemba_stars_prompt(langs, segs))
    wmt_datasets[text_id] = dset.add_column('gemba_stars_prompt', prompt_column)
wmt_datasets = DatasetDict(wmt_datasets)


llama2_tokenizer = AutoTokenizer.from_pretrained('./llama-2-7b-hf/')
llama2_tokenizer.pad_token = llama2_tokenizer.eos_token


peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
reg_model = AutoModelForSequenceClassification.from_pretrained('./llama-2-7b-hf/', num_labels=1)
reg_model = get_peft_model(reg_model, peft_config)
reg_model.gradient_checkpointing_enable()

llama2_wmt_reg_datasets = wmt_datasets.map(partial(regression_tokenize, tokenizer=llama2_tokenizer, input_key='gemba_stars_prompt'),
                                           batched=True)
llama2_wmt_reg_datasets = llama2_wmt_reg_datasets.remove_columns(set(llama2_wmt_reg_datasets['train'].column_names)-{'input_ids', 'attention_mask', 'labels'})
llama2_wmt_reg_datasets.set_format('torch')
llama2_wmt_reg_datasets['train'].column_names

data_collator = DataCollatorWithPadding(tokenizer=llama2_tokenizer)

train_dataloader = DataLoader(
    llama2_wmt_reg_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator
)
test_dataloader = DataLoader(
    llama2_wmt_reg_datasets['test'], batch_size=8, collate_fn=data_collator
)

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

"""
training_args = TrainingArguments(**default_args)

decay_parameters = get_parameter_names(reg_model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in reg_model.named_parameters() if n in decay_parameters],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [p for n, p in reg_model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
optimizer_kwargs["lr"] = training_args.learning_rate
optimizer = bnb.optim.Adam8bit(
    optimizer_grouped_parameters,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)
"""

optimizer = DeepSpeedCPUAdam(reg_model.parameters())

train_dl, test_dl, model, optimizer = accelerator.prepare(
    train_dataloader, test_dataloader, reg_model, optimizer
)

num_epochs = 5

num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))


for epoch in range(num_epochs):
    model.train()
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    preds = []
    targets = []
    for batch in test_dl:
        outputs = model(**batch)
        preds.append(outputs)
        targets.append(batch['labels'])
    sys.stderr.write(f'{str(compute_metrics((preds, targets)))}\n')
