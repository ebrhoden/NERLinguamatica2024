import sys
sys.stderr = sys.stdout

from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.data import Sentence
from flair.models import SequenceTagger

from tqdm import tqdm
from seqeval.metrics import classification_report
from copy import deepcopy
from os import path
import os
import json



class Training:
    def __init__(self, data_folder, corpus_name, model_checkpoint, model_name, output_dir_list) -> None:
        self.model_checkpoint = model_checkpoint
        self.model_name = model_name
        self.output_dir_list = output_dir_list
        self.set_corpus(data_folder, corpus_name)
        

    def set_corpus(self, data_folder, corpus_name):
        self.corpus_name = corpus_name

        # define columns
        columns = {0: 'tokens', 1: 'ner'}

        # this is the folder in which train, test and dev files reside
        print("Getting data from:", data_folder)
        # init a corpus using column format, data folder and the names of the train, dev and test files
        self.corpus: Corpus = ColumnCorpus(data_folder, columns,
                                    train_file='train.txt',
                                    test_file='test.txt',
                                    dev_file='dev.txt')
        print(self.corpus)

        # 2. what label do we want to predict?
        label_type = 'ner'

        # 3. make the label dictionary from the corpus
        self.label_dict = self.corpus.make_label_dictionary(label_type=label_type, add_unk=False)
        print(self.label_dict)

    def _create_directory(self, ref):
        if path.exists(ref) == False:
            os.mkdir(ref)

    def _create_directory_recursive(self, root, dir_list):
        path = root

        for dir in dir_list:
            path = "{path}/{dir}".format(path=path,dir=dir)
            self._create_directory(path)

        return path

    def train(self, max_length, truncation, lr, num_epochs, use_crf, use_rnn, main_evaluation_metric):
        # 4. initialize fine-tuneable transformer embeddings WITH document context

        if use_rnn:
            layers = "all"
        else:
            layers = "-1"

        embeddings = TransformerWordEmbeddings(model=self.model_checkpoint,
                                                    layers=layers,
                                                    subtoken_pooling="first",
                                                    fine_tune= not use_rnn,
                                                    use_context=False,
                                                    #truncate=True,

                                                    #force_max_length=True,
                                                    transformers_tokenizer_kwargs={'truncation': True, 'model_max_length':512}
                                                    #truncate=truncation,
                                                    #truncate=truncation,
                                                    #model_max_length=max_length,
                                                    )

        # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
        tagger = SequenceTagger(hidden_size=256,
                                embeddings=embeddings,
                                tag_dictionary=self.label_dict,
                                tag_type='ner',
                                use_crf= use_crf,
                                use_rnn=use_rnn,
                                reproject_embeddings=use_rnn
                                )

        # 6. initialize trainer
        trainer = ModelTrainer(tagger, self.corpus)

        # 7. run fine-tuning
        output_dir = deepcopy(self.output_dir_list)
        output_dir.insert(0, "models")
        output_dir = self._create_directory_recursive(".", output_dir)

        if use_rnn:
            trainer.train()
        else:
            trainer.fine_tune(output_dir, learning_rate=lr, use_final_model_for_eval=False, max_epochs=num_epochs, main_evaluation_metric=main_evaluation_metric)

    def get_prediction(self, tagger, text):
        # make a sentence
        sentence = Sentence(text)

        # predict NER tags
        tagger.predict(sentence)

        # transfer entity labels to token level
        for entity in sentence.get_spans('ner'):
            prefix = 'B-'
            for token in entity:
                token.set_label('ner-bio', prefix + entity.tag, entity.score)
                prefix = 'I-'

        # now go through all tokens and print label
        bio_tokens = []
        for token in sentence:
            try:
                #print(token.text, token.tag)
                bio_tokens.append(token.tag)
            except:
                #print(token.text, "O")
                bio_tokens.append("O")
                
        return bio_tokens
    
    def convert_to_float(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                self.convert_to_float(value)  # chamada recursiva para dicionários internos
            else:
                try:
                    d[key] = float(value)  # tentativa de converter o valor para float
                except (ValueError, TypeError):
                    pass  # ignora valores que não podem ser convertidos para float
        return d
    
    def save_json(self, dir_list, json_object, file_name):
        path = self._create_directory_recursive(".", dir_list)

        with open(f"{path}/{file_name}", "w", encoding='utf-8') as outfile:
            json.dump(json_object, outfile)

    def get_metrics_file(self, txt_file):
        # Inicializar listas para armazenar y_true e y_pred
        y_true = []
        y_pred = []
       
        # Abrir o arquivo TXT e ler linha por linha
        with open(txt_file, 'r', encoding='utf-8') as file:
            sentence_true = []
            sentence_pred = []

            for line in file:
                # Dividir a linha em três partes usando espaços ou tabulações como delimitadores
                parts = line.strip().split()
                
                # Verificar se a linha possui três partes
                if len(parts) == 3:
                    # Adicionar tag original à lista da sentença y_true
                    sentence_true.append(parts[1])
                    # Adicionar tag predita à lista da sentença y_pred
                    sentence_pred.append(parts[2])
                else:
                    # Se a linha não tem três partes, consideramos que uma nova sentença começou
                    if sentence_true and sentence_pred:
                        y_true.append(sentence_true)
                        y_pred.append(sentence_pred)
                        # Reiniciar listas para a próxima sentença
                        sentence_true = []
                        sentence_pred = []
            
            # Adicionar a última sentença, se houver
            if sentence_true and sentence_pred:
                y_true.append(sentence_true)
                y_pred.append(sentence_pred)
        
        return y_true, y_pred

    def get_and_save_metrics_test(self):
        output_dir = deepcopy(self.output_dir_list)
        output_dir.insert(0, "models")
        output_dir = self._create_directory_recursive(".", output_dir)

        #Getting from tsv
        #new_file = open("{output_dir}/test.txt".format(output_dir=output_dir), "w+", encoding="utf8")

        with open("{output_dir}/test.tsv".format(output_dir=output_dir), "r", encoding="utf8") as file, open(f"{output_dir}/test.txt", "w", encoding="utf8") as new_file:
            for line in file:
                if line != "\n":
                    splited = line.split(" ")
                    if splited[0] != '':
                        line = line.strip()
                        spliter = line.split(" ")
                        token = spliter[0]
                        tag_1 = spliter[1]
                        tag_2 = spliter[2]
                        new_file.write(str(token)+" "+str(tag_1)+" "+str(tag_2)+"\n")
                else:
                    new_file.write("\n")

        #Metrics
        print(f'{output_dir}/test.txt')
        eval_labels, pred_labels = self.get_metrics_file('{output_dir}/test.txt'.format(output_dir=output_dir))
        metrics = classification_report(eval_labels, pred_labels, digits=4, output_dict=True)
        metrics = self.convert_to_float(metrics)

        self._create_directory("metrics")
        output_dir = deepcopy(self.output_dir_list)
        output_dir.insert(0, "metrics")

        self.save_json(json_object=metrics, dir_list=output_dir, file_name="test.json")

        return None, metrics