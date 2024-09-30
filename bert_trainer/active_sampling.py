import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

class active_sampling:
    def __init__(self, model_name: str) -> None:
        self.model = self.__set_embedding_model(model_name)

    def __set_embedding_model(self, model_name: str):
        #if embedding_model_type == SBERT:
        model = SentenceTransformer(model_name)
        return model

    def feasibilize(self, df: pd.DataFrame, percent_sampling: float, min_size: int) -> int:
        """
        :param df: DataFrame to sampling
        :param percent_sampling: percentage of data to sampling
        :param min_size: min size of the sample
        :return int: feasible sample size 
        """

        df_len = len(df)

        if df_len*percent_sampling < min_size:
            if min_size > df_len:
                return df_len
            else:
                return min_size
        
        return int(percent_sampling*df_len)

    def random(self, df_target: pd.DataFrame, seed: int, percent_sampling: float, min_size: int) -> pd.DataFrame:
        """
        :param df: DataFrame to sampling
        :param seed: seed to the random set
        :param percent_sampling: percentage of data to sampling
        :param min_size: min size of the sample
        :return pd.DataFrame: DataFrame with the sampling
        """
        print("Random...")
        if percent_sampling == 1:
            return df_target

        n_samples = self.feasibilize(df_target, percent_sampling, min_size)
        df_sample = df_target.sample(n=n_samples, random_state=seed)

        df_sample = df_sample.reset_index(drop=True)

        return df_sample
    
    def dissimilarity(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, percent_sampling: float, min_size: int, input: str) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        
        if percent_sampling == 1:
            return df_to_sampling
        
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

        n_samples = self.feasibilize(df_target, percent_sampling, min_size)

        # Select the top n_samples most dissimilar sentences from df_to_sampling
        selected_indices = sorted_indices[:n_samples]          

        # Create the DataFrame of the sampled examples
        df_sample = df_to_sampling.iloc[selected_indices]
        df_sample = df_sample.reset_index(drop=True)

        return df_sample
    
    def random_dissimilarity(self, df_target: pd.DataFrame, df_to_sampling: pd.DataFrame, input: str, seed: int, percent_sampling_random: float, percent_sampling_dissimilar: float, min_size_random: int, min_size_dissimilar: int):
        random = self.random(df_to_sampling, seed, percent_sampling_random, min_size_random)
        dissimilar = self.dissimilarity(df_target, random, percent_sampling_dissimilar, min_size_dissimilar, input)

        return dissimilar
