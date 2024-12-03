import math
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from rank_bm25 import BM25Okapi

class active_sampling:
    def __init__(self, model_name: str, category_distribution_dictionary: dict, category_sampling_priority: list) -> None:
        self.model = self.__set_embedding_model(model_name)
        self.category_distribution_dictionary = category_distribution_dictionary
        self.category_sampling_priority = category_sampling_priority
        
    def __set_embedding_model(self, model_name: str):
        #if embedding_model_type == SBERT:
        model = SentenceTransformer(model_name)
        return model
    
    def feasibilize(self, df: pd.DataFrame, sample_fetch_size: int) -> int:
        """
        :param df: DataFrame to sampling
        :sample_fetch_size: desired sample fetch size
        :return int: feasible sample size 
        """

        df_len = len(df)

        if sample_fetch_size > df_len:
            return df_len
        
        return sample_fetch_size

    def random(self, df_target: pd.DataFrame, seed: int, sample_fetch_size: int) -> pd.DataFrame:
        """
        :param df: DataFrame to sampling
        :param seed: seed to the random set
        :param percent_sampling: percentage of data to sampling
        :param min_size: min size of the sample
        :return pd.DataFrame: DataFrame with the sampling
        """
        print("Random...")

        n_samples = self.feasibilize(df_target, sample_fetch_size)
        df_sample = df_target.sample(n=n_samples, random_state=seed)

        df_sample = df_sample.reset_index(drop=True)

        return df_sample
    
    
    def dissimilarity(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, percent_sampling_dissimilar:float, input: str) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        
        print("Dissimilarity...")
        print("\t=>", len(df_to_sampling[input].tolist()))
        
        # Single list of sentences
        target_sentences = df_target[input].tolist()
        sampling_sentences = df_to_sampling[input].tolist()

        # Compute embeddings
        embeddings_target = self.model.encode(target_sentences, convert_to_tensor=True)
        embeddings_sampling = self.model.encode(sampling_sentences, convert_to_tensor=True)

        # Compute cosine-similarities for each sentence in df_to_sampling with all sentences in df_target
        cosine_scores = util.cos_sim(embeddings_sampling, embeddings_target)

        # Calculate the average cosine similarity for each sentence in df_to_sampling
        average_scores = [sum(cosine_scores[i]) / len(cosine_scores[i]) for _, i in enumerate(tqdm(range(len(cosine_scores))))]

        # Sort sentences in df_to_sampling by decreasing average similarity score
        sorted_indices = sorted(range(len(average_scores)), key=lambda i: average_scores[i])

        # Always 0.5 * sample fetch size
        attempt_fetch_size = int(percent_sampling_dissimilar * len(df_to_sampling))
        n_samples = self.feasibilize(df_target, attempt_fetch_size)

        # Select the top n_samples most dissimilar sentences from df_to_sampling
        selected_indices = sorted_indices[:n_samples]          

        # Create the DataFrame of the sampled examples
        df_sample = df_to_sampling.iloc[selected_indices]
        df_sample = df_sample.reset_index(drop=True)

        return df_sample
    
    def random_dissimilarity(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, input: str, seed: int, percent_sampling_dissimilar: float, sample_fetch_size: int):
        random = self.random(df_to_sampling, seed, sample_fetch_size)
        dissimilar = self.dissimilarity(df_target, random, percent_sampling_dissimilar, input)

        return dissimilar
    
    def get_amount_of_entities(self):
        return sum(self.category_distribution_dictionary.values())
    
    def _get_top_BM_matches_by_category_lem(self,amount_of_retrieved_documents: int, category: str, unlabeled_dataframe: pd.DataFrame, labeled_dataframe: pd.DataFrame) -> list[str]:
        filtered_dataframe = labeled_dataframe[labeled_dataframe[category] > 0].sort_values(by=[category], ascending=False)
        tokenized_corpus = unlabeled_dataframe['lem_sentence'].str.split(" ")
        
        query = filtered_dataframe['lem_sentence']
        tokenized_query = query
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        top_matches = bm25.get_top_n(tokenized_query, tokenized_corpus, amount_of_retrieved_documents)
        top_matches_parsed = [' '.join(top_match) for top_match in top_matches]
        
        return top_matches_parsed
    
    def _get_top_BM_matches_by_category_stem(self, amount_of_retrieved_documents: int, category: str, unlabeled_dataframe: pd.DataFrame, labeled_dataframe: pd.DataFrame) -> list[str]:
        filtered_dataframe = labeled_dataframe[labeled_dataframe[category] > 0].sort_values(by=[category], ascending=False)
        tokenized_corpus = unlabeled_dataframe['stem_sentence'].str.split(" ")
        
        query = filtered_dataframe['stem_sentence']
        tokenized_query = query
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        top_matches = bm25.get_top_n(tokenized_query, tokenized_corpus, amount_of_retrieved_documents)
        top_matches_parsed = [' '.join(top_match) for top_match in top_matches]
        
        return top_matches_parsed
    
    def sample_proportional_categories_lemmatized(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, sample_fetch_size: int):
        print("Proportional categories lemmatized ...")
        
        df_unlabeled_copy = df_to_sampling.copy()
        final_set_sentences = set()
        
        amount_of_entities = self.get_amount_of_entities()

        n_samples = self.feasibilize(df_unlabeled_copy, sample_fetch_size)

        for category in self.category_sampling_priority:

            attempt_fetch_size = math.floor(n_samples * self.category_distribution_dictionary.get(category)/amount_of_entities)
            real_fetch_size = self.feasibilize(df_unlabeled_copy, attempt_fetch_size)

            top_matches = self._get_top_BM_matches_by_category_lem(max(1, real_fetch_size), category, df_unlabeled_copy, df_target)

            df_unlabeled_copy = df_unlabeled_copy.loc[~df_unlabeled_copy['lem_sentence'].isin(top_matches)]    
            df_unlabeled_copy = df_unlabeled_copy.reset_index(drop=True)

            for match in top_matches:
                final_set_sentences.add(match)

            if len(df_unlabeled_copy) == 0:
                break
        
        selected_samples_dataframe = df_to_sampling.copy()
        selected_samples_dataframe = selected_samples_dataframe.loc[selected_samples_dataframe['lem_sentence'].isin(final_set_sentences)]
        selected_samples_dataframe = selected_samples_dataframe.reset_index(drop=True)
        return selected_samples_dataframe 

    def sample_proportional_categories_stemmed(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, sample_fetch_size: int):
        print("Proportional categories stemmed ...")
        
        df_unlabeled_copy = df_to_sampling.copy()
        final_set_sentences = set()
        
        amount_of_entities = self.get_amount_of_entities()

        n_samples = self.feasibilize(df_unlabeled_copy, sample_fetch_size)

        for category in self.category_sampling_priority:

            attempt_fetch_size = math.floor(n_samples * self.category_distribution_dictionary.get(category)/amount_of_entities)
            real_fetch_size = self.feasibilize(df_unlabeled_copy, attempt_fetch_size)

            top_matches = self._get_top_BM_matches_by_category_stem(max(1, real_fetch_size), category, df_unlabeled_copy, df_target)

            df_unlabeled_copy = df_unlabeled_copy.loc[~df_unlabeled_copy['stem_sentence'].isin(top_matches)]    
            df_unlabeled_copy = df_unlabeled_copy.reset_index(drop=True)

            for match in top_matches:
                final_set_sentences.add(match)

            if len(df_unlabeled_copy) == 0:
                break
        
        selected_samples_dataframe = df_to_sampling.copy()
        selected_samples_dataframe = selected_samples_dataframe.loc[selected_samples_dataframe['stem_sentence'].isin(final_set_sentences)]
        selected_samples_dataframe = selected_samples_dataframe.reset_index(drop=True)
        return selected_samples_dataframe 
    
    def sample_disproportional_categories_lematized(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, sample_fetch_size: int):
        print("Disproportional categories lemmatized ...")
        
        df_unlabeled_copy = df_to_sampling.copy()
        final_set_sentences = set()
        
        amount_of_entities = self.get_amount_of_entities()

        n_samples = self.feasibilize(df_unlabeled_copy, sample_fetch_size)

        # Category sampling priority already has been reversed during instantiation!
        for category in self.category_sampling_priority:

            attempt_fetch_size = math.floor(n_samples * self.category_distribution_dictionary.get(category)/amount_of_entities)
            real_fetch_size = self.feasibilize(df_unlabeled_copy, attempt_fetch_size)

            top_matches = self._get_top_BM_matches_by_category_stem(max(1, real_fetch_size), category, df_unlabeled_copy, df_target)

            df_unlabeled_copy = df_unlabeled_copy.loc[~df_unlabeled_copy['lem_sentence'].isin(top_matches)]    
            df_unlabeled_copy = df_unlabeled_copy.reset_index(drop=True)

            for match in top_matches:
                final_set_sentences.add(match)

            if len(df_unlabeled_copy) == 0:
                break
        
        selected_samples_dataframe = df_to_sampling.copy()
        selected_samples_dataframe = selected_samples_dataframe.loc[selected_samples_dataframe['lem_sentence'].isin(final_set_sentences)]
        selected_samples_dataframe = selected_samples_dataframe.reset_index(drop=True)
        return selected_samples_dataframe 
    
    def sample_disproportional_categories_stemmed(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, sample_fetch_size: int):
        print("Disproportional categories stemmed ...")
        
        df_unlabeled_copy = df_to_sampling.copy()
        final_set_sentences = set()
        
        amount_of_entities = self.get_amount_of_entities()

        n_samples = self.feasibilize(df_unlabeled_copy, sample_fetch_size)

        # Category sampling priority already has been reversed during instantiation!
        for category in self.category_sampling_priority:

            attempt_fetch_size = math.floor(n_samples * self.category_distribution_dictionary.get(category)/amount_of_entities)
            real_fetch_size = self.feasibilize(df_unlabeled_copy, attempt_fetch_size)

            top_matches = self._get_top_BM_matches_by_category_stem(max(1, real_fetch_size), category, df_unlabeled_copy, df_target)

            df_unlabeled_copy = df_unlabeled_copy.loc[~df_unlabeled_copy['stem_sentence'].isin(top_matches)]    
            df_unlabeled_copy = df_unlabeled_copy.reset_index(drop=True)

            for match in top_matches:
                final_set_sentences.add(match)

            if len(df_unlabeled_copy) == 0:
                break
        
        selected_samples_dataframe = df_to_sampling.copy()
        selected_samples_dataframe = selected_samples_dataframe.loc[selected_samples_dataframe['stem_sentence'].isin(final_set_sentences)]
        selected_samples_dataframe = selected_samples_dataframe.reset_index(drop=True)
        return selected_samples_dataframe 
    
    def sample_uniform_categories_lematized(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, sample_fetch_size: int):
        print("Uniform categories lemmatized ...")
        
        df_unlabeled_copy = df_to_sampling.copy()
        final_set_sentences = set()
        
        amount_of_categories = len(self.category_sampling_priority)

        n_samples = self.feasibilize(df_unlabeled_copy, sample_fetch_size)

        # Category sampling priority order doesnt matter!
        for category in self.category_sampling_priority:
            attempt_fetch_size = math.floor(n_samples * 1/amount_of_categories)
            real_fetch_size = self.feasibilize(df_unlabeled_copy, attempt_fetch_size)

            top_matches = self._get_top_BM_matches_by_category_lem(max(1, real_fetch_size), category, df_unlabeled_copy, df_target)

            df_unlabeled_copy = df_unlabeled_copy.loc[~df_unlabeled_copy['lem_sentence'].isin(top_matches)]    
            df_unlabeled_copy = df_unlabeled_copy.reset_index(drop=True)

            for match in top_matches:
                final_set_sentences.add(match)

            if len(df_unlabeled_copy) == 0:
                break
        
        selected_samples_dataframe = df_to_sampling.copy()
        selected_samples_dataframe = selected_samples_dataframe.loc[selected_samples_dataframe['lem_sentence'].isin(final_set_sentences)]
        selected_samples_dataframe = selected_samples_dataframe.reset_index(drop=True)
        return selected_samples_dataframe 
    
    def sample_uniform_categories_stemmed(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, sample_fetch_size: int):
        print("Uniform categories stemmed ...")
        
        df_unlabeled_copy = df_to_sampling.copy()
        final_set_sentences = set()
        
        amount_of_categories = len(self.category_sampling_priority)

        n_samples = self.feasibilize(df_unlabeled_copy, sample_fetch_size)

        # Category sampling priority order doesnt matter!
        for category in self.category_sampling_priority:

            attempt_fetch_size = math.floor(n_samples * 1/amount_of_categories)
            real_fetch_size = self.feasibilize(df_unlabeled_copy, attempt_fetch_size)

            top_matches = self._get_top_BM_matches_by_category_stem(max(1, real_fetch_size), category, df_unlabeled_copy, df_target)

            df_unlabeled_copy = df_unlabeled_copy.loc[~df_unlabeled_copy['stem_sentence'].isin(top_matches)]    
            df_unlabeled_copy = df_unlabeled_copy.reset_index(drop=True)

            for match in top_matches:
                final_set_sentences.add(match)

            if len(df_unlabeled_copy) == 0:
                break
        
        selected_samples_dataframe = df_to_sampling.copy()
        selected_samples_dataframe = selected_samples_dataframe.loc[selected_samples_dataframe['stem_sentence'].isin(final_set_sentences)]
        selected_samples_dataframe = selected_samples_dataframe.reset_index(drop=True)
        return selected_samples_dataframe 
    