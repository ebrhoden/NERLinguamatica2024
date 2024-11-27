# Named Entity Recognition in Portuguese Legislative Texts

## Repository Overview

This GitHub repository contains the code and resources used for training models on the **UlyssesNER-Br** corpus. The corpus, which will be available soon in the official repository, includes CoNLL-formatted and JSON files with revised annotations, following the removal of duplicates and correction of inconsistencies.

This article is an expanded version of the paper **"A Named Entity Recognition Approach for Portuguese Legislative Texts Using Self-Learning,"** published in the proceedings of PROPOR 2024. You can find the original paper [here](https://aclanthology.org/2024.propor-1.30).

## Training Models

To train a model, use the following command:

```bash
python3 main.py [MODEL] [CORPUS] [METRIC]
```

Where:
- `[MODEL]` should be replaced with the name of the model you want to train.
- `[CORPUS]` refers to the dataset (e.g., `ulysses`), which will be available soon.
- `[METRIC]` should be either `micro_avg` or `macro_avg`, depending on the evaluation metric you wish to use.

## Running Self-Learning

To run the **self-learning** process, use the `main_self_learning.py` script with the following command:

```bash
python3 main_self_learning.py [MODEL] [CORPUS] [METRIC]
```

Where:
- `[MODEL]` is the model to be trained using the self-learning method.
- `[CORPUS]` refers to the dataset, which will be available soon.
- `[METRIC]` is either `micro_avg` or `macro_avg`, depending on the evaluation metric you wish to use.

## K-Folds and Stratified Partitions

The code to generate stratified partitions using holdout and cross-validation can be found in `tools/precompute-k-folds.ipynb`. This notebook provides the necessary scripts to create partitions suitable for training and evaluation.

## Models Configuration

The available models are defined in the `main.py` and `main_self_learning.py` files. To train new models, update the model dictionaries in these files with the desired model configurations.

## Files and Structure

- **main.py:** Contains the code for training models using the Flair library and defines the available models and their configurations.
- **main_self_learning.py:** Runs the self-learning process and trains models using self-supervised techniques.
- **tools/precompute-k-folds.ipynb:** Contains code for generating stratified partitions using holdout and cross-validation.
- **Corpus Files:**
  - **TXT (CoNLL format):** To be provided in the official corpus repository.
  - **JSON:** Mirrors the format used in the official **UlyssesNER-Br** repository.

## Additional Resources

Aqui está a versão corrigida:

- The updated **UlyssesNER-Br** corpus is described in the paper **"Named Entity Recognition and Data Leakage in Legislative Texts: A Literature Reassessment,"** which is currently under review in the journal **Linguamática**. The preprint of the paper will be available soon.
-  For more details on the original article, refer to: 

```bibtex
@inproceedings{nunes-etal-2024-named,
    title = "A Named Entity Recognition Approach for {P}ortuguese Legislative Texts Using Self-Learning",
    author = "Nunes, Rafael Oleques  and
      Balreira, Dennis Giovani  and
      Spritzer, Andr{\'e} Suslik  and
      Freitas, Carla Maria Dal Sasso",
    editor = "Gamallo, Pablo  and
      Claro, Daniela  and
      Teixeira, Ant{\'o}nio  and
      Real, Livy  and
      Garcia, Marcos  and
      Oliveira, Hugo Gon{\c{c}}alo  and
      Amaro, Raquel",
    booktitle = "Proceedings of the 16th International Conference on Computational Processing of Portuguese - Vol. 1",
    month = mar,
    year = "2024",
    address = "Santiago de Compostela, Galicia/Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.propor-1.30",
    pages = "290--300",
}
```

## BibTeX

Please cite the paper and repository using the following BibTeX entry:

```bibtex
@inproceedings{nunes2024ulyssesner,
  title = "Reconhecimento de Entidades Nomeadas e Vazamento de Dados em Textos Legislativos: Uma Reavaliação da Literatura",
  author = "Nunes, Rafael Oleques  and
      Balreira, Dennis Giovani  and
      Spritzer, Andre Suslik  and
      Freitas, Carla Maria Dal Sasso",
  year = {2024},
  pages = {},
  title = {Named Entity Recognition and Data Leakage in Legislative Texts: A Literature Reassessment},
  url={http://dx.doi.org/10.13140/RG.2.2.25781.69602},
}
```

## Test push

---
