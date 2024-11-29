from active_sampling import active_sampling
import pandas as pd
from rank_bm25 import BM25Okapi

class active_sampling_strategies(active_sampling):
    def __init__(self, category_distribution_dictionary: dict, category_sampling_priority: list, model_name: str) -> None:
        super().__init__(model_name)
        self.category_distribution_dictionary = category_distribution_dictionary
        self.category_sampling_priority = category_sampling_priority
    
    def get_top_BM_matches_by_category(amount_of_retrieved_documents: int, category: str, unlabeled_dataframe: pd.DataFrame, labeled_dataframe: pd.DataFrame) -> list[str]:
        filtered_dataframe = labeled_dataframe[labeled_dataframe[category] > 0].sort_values(by=[category], ascending=False)
        tokenized_corpus = unlabeled_dataframe['sentences'].str.split(" ")
        
        query = filtered_dataframe['sentences']
        tokenized_query = query
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        top_matches = bm25.get_top_n(tokenized_query, tokenized_corpus, amount_of_retrieved_documents)
        top_matches_parsed = [' '.join(top_match) for top_match in top_matches]
        
        return top_matches_parsed
    
    def get_amount_of_entities(self):
        return sum(self.category_distribution_dictionary.values())
    
    def strategy41(self, df_labeled: pd.DataFrame, df_unlabeled: pd.DataFrame, percent_sampling: float, min_size: int) -> pd.DataFrame:
        print("Strategy 4.1 ...")
        
        df_unlabeled_copy = df_unlabeled.copy()
        final_set_sentences = set()
        
        if percent_sampling == 1:
            return df_unlabeled_copy
        
        n_samples = self.feasibilize(df_unlabeled_copy, percent_sampling, min_size)
        
        amount_of_entities = self.get_amount_of_entities()

        for category in self.category_sampling_priority:
            top_matches = self.get_top_BM_matches_by_category(max(1, round(n_samples * self.category_distribution_dictionary.get(category)/amount_of_entities)), category, df_unlabeled_copy, df_labeled)

            unlabeled_dataframe_copy = unlabeled_dataframe_copy.loc[~unlabeled_dataframe_copy['sentences'].isin(top_matches)]    
            unlabeled_dataframe_copy = unlabeled_dataframe_copy.reset_index(drop=True)

            for match in top_matches:
                final_set_sentences.add(match)

            if len(unlabeled_dataframe_copy) == 0:
                break
        
        selected_samples_dataframe = df_unlabeled.copy()
        selected_samples_dataframe = selected_samples_dataframe.loc[unlabeled_dataframe_copy['sentences'].isin(final_set_sentences)]
        selected_samples_dataframe =selected_samples_dataframe.reset_index(drop=True)
        return selected_samples_dataframe 