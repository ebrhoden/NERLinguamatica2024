import sys

root = "./bert_trainer"
sys.path.append('./bert_trainer')

sys.stderr = sys.stdout

from bert_trainer.self_learning import SelfLearning
from bert_trainer.configs import  SENTENCE_THRESHOLD, FIXED
from tools.function import create_directory_recursive, copy_and_replace

#args
model_arg = sys.argv[1]
corpus_name = sys.argv[2]
metric_name = sys.argv[3]

#Model
models_info = {
    "BERTimbau": "neuralmind/bert-large-portuguese-cased", 
    "BERTimbau-base": "neuralmind/bert-base-portuguese-cased", 
    "XLM-R-large": "FacebookAI/xlm-roberta-large", 

    "RoBERTaLexPT": "eduagarcia/RoBERTaLexPT-base", 
    "Legal-XLM-R": "joelniklaus/legal-xlm-roberta-large",
    "Legal-XLM-R-base": "joelniklaus/legal-xlm-roberta-large", 
    "Legal-portuguese-R": "joelniklaus/legal-portuguese-roberta-large", 

    "LegalBert-pt_FP": "raquelsilveira/legalbertpt_fp", 
    "LegalBert-pt_SC": "raquelsilveira/legalbertpt_sc", 

    "Jurisbert": "alfaneo/jurisbert-base-portuguese-uncased",
    "BERTimbaulaw": "alfaneo/bertimbaulaw-base-portuguese-cased",
    "BERTikal": "felipemaiapolo/legalnlp-bert",

    "Albertina-1b5": "PORTULAN/albertina-1b5-portuguese-ptbr-encoder", 
    "Albertina-1b5-256": "PORTULAN/albertina-1b5-portuguese-ptbr-encoder-256", 
    "Albertina-100m": "PORTULAN/albertina-100m-portuguese-ptbr-encoder", 
    "Albertina-900m": "PORTULAN/albertina-900m-portuguese-ptbr-encoder", 
    "Albertina-900m-brwac": "PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac", 
    
    }

metrics = {
    "micro_avg" : ("micro avg", "f1-score"),
    "macro_avg" : ("macro avg", "f1-score"),
}

model_checkpoint = models_info[model_arg]
model_name = model_arg

#Corpus
data_folder = f"corpora/{corpus_name}"

#Training hyperparameters
max_length = 512
truncation = True
lr = 2e-05
num_epochs = 10
use_rnn = False

main_evaluation_metric = metrics[metric_name]
metric = metric_name

#Technique (receive as sysarg?)

technique = "self-learning_random-dissimilar"
#technique = "self-learning-random"

#technique = "self_learning_proportional_categories_lem"
#technique = "self_learning_proportional_categories_stem"

#technique = "self_learning_disproportional_categories_lem"
#technique = "self_learning_disproportional_categories_stem"

#technique = "self_learning_uniform_categories_lem"
#technique = "self_learning_uniform_categories_stem"

#Number of folds
folds = 5

#Self-learning
input = "sentences"
output = "ner_tokens"


for use_crf in [True, False]:
    #Output
    architecture = [model_name]
    
    if use_rnn:
        architecture.append("bilstm")

    if use_crf:
        architecture.append("crf")
    else:
        architecture.append("linear")

    architecture_str = "_".join(architecture)

    #Cross-validatio
    for fold in range(folds):
        data_folder_labeled = f"{data_folder}/labeled/folds/fold{fold}"
        data_folder_unlabeled = f"{data_folder}/unlabeled/"

        #Making a copy of the training set
        copy_and_replace(f"{data_folder_labeled}/train.json", f"{data_folder_labeled}/train_original.json")
        output_dir_list = [technique, corpus_name, architecture_str, metric, f"{folds}folds", f"fold{fold}"]


        print("================================")
        print("Starting training: {architecture}, {model_checkpoint}".format(model_name=model_name, model_checkpoint=model_checkpoint, architecture="-".join(architecture)))
        print("Fold: {fold}/{folds}".format(fold=fold, folds=folds-1))
        print("================================")
        
        selfLearning = SelfLearning(input=input, output=output, 
                                    percent_sampling_random=0.05, percent_sampling_dissimilar=1, 
                                    min_size_random=2000, min_size_dissimilar=1000, 
                                    labeled_corpus_path=data_folder_labeled, unlabeled_corpus_path=data_folder_unlabeled,
                                    sentence_embedding_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
                                    model_checkpoint=model_checkpoint, model_name=model_name,
                                    corpus_name=corpus_name,
                                    technique=technique,)
        
        selfLearning.set_trainer(max_length, truncation, lr, num_epochs, use_crf, use_rnn, main_evaluation_metric)
        
        output_dir_list = selfLearning.iterations(data_folder_labeled, 100, 1, 0.99, SENTENCE_THRESHOLD, 0.005, 4, FIXED, folds, fold, output_dir_list)

        #Save the generate training set and restoring original training set
        generated_corpora_path = create_directory_recursive(".", ["generated_corpora"] + output_dir_list)
        copy_and_replace(f"{data_folder_labeled}/train.json", f"{generated_corpora_path}/train.json")
        copy_and_replace(f"{data_folder_labeled}/train_original.json", f"{data_folder_labeled}/train.json")