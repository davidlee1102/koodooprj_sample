import httpx  # an HTTP client library and dependency of Prefect
import torch
import nltk
import numpy as np
import transformers

from prefect import flow, task
from koodoovoice.model_packages import constant_key
from datasets import load_dataset, load_metric, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import constant_key
from datasets import load_dataset, load_metric, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# koodoovoice.model_packages import constant_key
nltk.download('punkt')

max_input = 512
max_target = 128
metric = load_metric('rouge')
torch.mps.set_per_process_memory_fraction(0.0)


def compute_rouge(pred, tokenizer):
    predictions, labels = pred
    # decode the predictions
    decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # decode labels
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # compute results
    res = metric.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)
    # get %
    res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

    pred_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    res['gen_len'] = np.mean(pred_lens)

    return {k: round(v, 4) for k, v in res.items()}


def preprocess_data(data_to_process, tokenizer):
    inputs = [dialogue for dialogue in data_to_process['dialogue']]
    model_inputs = tokenizer(inputs, max_length=max_input, padding='max_length', truncation=True)

    with tokenizer.as_target_tokenizer():
        targets = tokenizer(data_to_process['summary'], max_length=max_target, padding='max_length', truncation=True)

    model_inputs['labels'] = targets['input_ids']
    return model_inputs


@task
def data_loader():
    data = load_dataset('samsum')
    data.save_to_disk('/Users/davidlee/PycharmProjects/KoodooProject/koodoovoice/dataset_temp/samsum')
    data = load_from_disk("/Users/davidlee/PycharmProjects/KoodooProject/koodoovoice/dataset_temp/samsum")
    return data


@task
def model_loader():
    tokenizer = transformers.AutoTokenizer.from_pretrained(constant_key.TOKENIZER_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(constant_key.MODEL_PATH)
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
    return tokenizer, model, collator


@flow(log_prints=True)
def trainer_create():
    data = data_loader()
    tokenizer, model, collator = model_loader()
    tokenize_data = data.map(preprocess_data, batched=True, remove_columns=['id', 'dialogue', 'summary'])
    train_sample = tokenize_data['train'].shuffle(seed=123).select(range(3000))
    validation_sample = tokenize_data['validation'].shuffle(seed=123).select(range(800))
    test_sample = tokenize_data['test'].shuffle(seed=123).select(range(800))

    tokenize_data['train'] = train_sample
    tokenize_data['validation'] = validation_sample
    tokenize_data['test'] = test_sample

    args = transformers.Seq2SeqTrainingArguments(
        'conversation-summ',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=2,
        predict_with_generate=True,
        eval_accumulation_steps=1,
        fp16=False,
        use_mps_device=True,
    )

    trainer = transformers.Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenize_data['train'],
        eval_dataset=tokenize_data['validation'],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_rouge
    )
    return trainer


@task(retries=2)
def get_repo_info(repo_owner: str, repo_name: str):
    """Get info about a repo - will retry twice after failing"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    api_response = httpx.get(url)
    api_response.raise_for_status()
    repo_info = api_response.json()
    return repo_info


@task
def get_contributors(repo_info: dict):
    """Get contributors for a repo"""
    contributors_url = repo_info["contributors_url"]
    response = httpx.get(contributors_url)
    response.raise_for_status()
    contributors = response.json()
    return contributors


@flow(log_prints=True)
def repo_info(repo_owner: str = "davidlee1102", repo_name: str = "Surtimesurvival"):
    """
    Given a GitHub repository, logs the number of stargazers
    and contributors for that repo.
    """
    repo_info = get_repo_info(repo_owner, repo_name)
    print(f"Stars 🌠 : {repo_info['stargazers_count']}")

    contributors = get_contributors(repo_info)
    print(f"Number of contributors 👷: {len(contributors)}")


if __name__ == "__main__":
    repo_info()
    # trainer_class = trainer_create()
    # trainer_class.train()
    # trainer_class.save_model("summary_model")
    # trainer_class.save_model("koodoovoice/model_packages/summary_model")
