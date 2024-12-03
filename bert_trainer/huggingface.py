from transformers import AutoModelForTokenClassification,AutoTokenizer,DataCollatorForTokenClassification,TrainingArguments, Trainer
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from os import path
import os, stat
import json

from copy import deepcopy


data_folder = "ulysses"

#Training
model_checkpoint = "neuralmind/bert-base-portuguese-cased"
model_name = "BERTimbau"
lr = 1e-3
batch_size = 16
num_epochs = 10
weight_decay=0.01

class Training:
    def __init__(self, data_folder, corpus_name, model_checkpoint, model_name, output_dir_list, train="train.json", test="test.json", validation="dev.json") -> None:
        self.data_folder = data_folder
        self.model_name = model_name
        self.output_dir_list = output_dir_list

        #Labels and dataset
        self.label_list, self.id2label, self.label2id = self.set_labels(self.data_folder)
        self.dataset = self.set_dataset(self.data_folder, train, test, validation)

        #Metrics
        self.seqeval = evaluate.load("seqeval")

        #Tokenizer and model
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    def _create_directory(self, ref):
        if path.exists(ref) == False:
            os.mkdir(ref)

    def _create_directory_recursive(self, root, dir_list):
        path = root

        for dir in dir_list:
            path = "{path}/{dir}".format(path=path,dir=dir)
            self._create_directory(path)

        return path

    def save_json(self, root, dict_data, file_name):
        output_dir = deepcopy(self.output_dir_list)
        output_dir.insert(0, root)
        output_dir = self._create_directory_recursive(".", output_dir)

        json_object = json.dumps(dict_data, indent=4, ensure_ascii=False)

        with open(f"{output_dir}/{file_name}", "w", encoding='utf-8') as outfile:
            outfile.write(json_object)

    def set_labels(self, data_folder):
        df_labels = pd.read_json(f"{data_folder}/labels.json")
        label_list = df_labels.labels.tolist()
        id2label = {idx: label for idx, label in enumerate(label_list)}
        label2id = {label: idx for idx, label in enumerate(label_list)}

        return label_list, id2label, label2id

    def set_dataset(self, data_folder, train, test, validation):
        df_train = pd.read_json(f"{data_folder}/{train}")
        df_test = pd.read_json(f"{data_folder}/{test}")
        df_validation = pd.read_json(f"{data_folder}/{validation}")

        df_train["ner_tokens"] = [[self.label2id[token] for token in ner_tokens] for ner_tokens in df_train["ner_tokens"].values.tolist()]
        df_test["ner_tokens"] = [[self.label2id[token] for token in ner_tokens] for ner_tokens in df_test["ner_tokens"].values.tolist()]
        df_validation["ner_tokens"] = [[self.label2id[token] for token in ner_tokens] for ner_tokens in df_validation["ner_tokens"].values.tolist()]

        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)
        validation_dataset = Dataset.from_pandas(df_validation)     
        
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
            'test': test_dataset})

        return dataset

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        
        results_dict =  {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        keys = list(set([label.replace("B-", "").replace("I-", "") for label in self.label_list if label != "O"]))

        for key in keys: 
            results_dict[f"{key}_precision"] = results[key]["precision"]
            results_dict[f"{key}_recall"] = results[key]["recall"]
            results_dict[f"{key}_f1"] = results[key]["f1"]
            results_dict[f"{key}_number"] = results[key]["number"]
            
        return results_dict
    
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], is_split_into_words=True, max_length=self.max_length, padding=self.padding, truncation=self.truncation)

        labels = []
        for i, label in enumerate(examples[f"ner_tokens"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
       

    def train(self, max_length, truncation, padding, lr, num_epochs, weight_decay, crf=False) -> None:
        #Training Hyperparameters
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay= weight_decay

        #Tokenizing
        self.tokenized = self.dataset.map(self.tokenize_and_align_labels, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        #Bert
        pretrained_model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.label_list), id2label=self.id2label, label2id=self.label2id)

        #Training
        self._create_directory("logs")

        output_dir = deepcopy(self.output_dir_list)
        output_dir = self._create_directory_recursive(".", output_dir)

        training_args = TrainingArguments(
            run_name=f"{output_dir}-training",
            output_dir= f"logs/{output_dir}-training",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        self.trainer = Trainer(
            model=pretrained_model,
            args=training_args,
            train_dataset=self.tokenized["train"],
            eval_dataset=self.tokenized["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

        #Saving model
        output_dir = deepcopy(self.output_dir_list)
        output_dir.insert(0, "models")
        output_dir = self._create_directory_recursive(".", output_dir)
        self.trainer.save_model(output_dir)
        
        #Saving training metrics
        self.save_json(dict_data=self.trainer.state.log_history, root="metrics", file_name="training.json")

    def get_and_save_metrics_test(self):
        y_probs, labels_ids, metrics = self.trainer.predict(self.tokenized["test"])

        self.save_json(dict_data=metrics, root="metrics", file_name="test.json")

        return y_probs, metrics