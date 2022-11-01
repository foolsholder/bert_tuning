import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import datasets

from transformers import TrainingArguments
from transformers import Trainer
from torch.optim import AdamW

from typing import Dict, List, Callable, Tuple, Union, Optional, Sequence, Any

if __name__ == '__main__':
    model_name = 'gchhablani/bert-base-cased-finetuned-sst2'

    model = transformers.BertForMaskedLM.from_pretrained(model_name)
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name)

    qqp = datasets.load_dataset('SetFit/sst2', use_auth_token=True)

    MAX_LENGTH = 128

    def preprocess_function(examples):
        result = tokenizer(
            examples['text'],
            padding='max_length', max_length=MAX_LENGTH, truncation=True
        )
        labels = examples['label']
        result['labels'] = [[]] * len(labels)
        for idx, value in enumerate(labels):
            result['labels'][idx] = [value] + (MAX_LENGTH - 1) * [-100]
        return result


    qqp_preprocessed = qqp.map(preprocess_function, batched=True)

    import numpy as np

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds[:, 0], axis=1)
        return {"accuracy": (preds == p.label_ids[:, 0]).astype(np.float32).mean().item()}


    import os
    os.environ['WANDB_PROJECT'] = 'glue-cls2mlm'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_args = TrainingArguments(
        output_dir='bert-base-mlm-sst2-cls',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        eval_steps=2000,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        max_grad_norm=1.0,
        num_train_epochs=4,
        report_to=["wandb"],
        save_strategy="epoch",
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        save_steps=2000,
        logging_steps=10
    )
    optim = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    trainer = Trainer(
        model=model,
        args=train_args,
        optimizers=(optim, None),
        tokenizer=tokenizer,
        data_collator=transformers.default_data_collator,
        train_dataset=qqp_preprocessed['train'],
        eval_dataset=qqp_preprocessed['validation'],
        compute_metrics=compute_metrics
    )
    trainer.train()
    torch.save(
        model.state_dict(),
        'bert-base-mlm-sst2-cls/best_ckpt.pth'
    )

