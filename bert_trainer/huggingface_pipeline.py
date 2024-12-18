from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer

class Pipeline:
    def __init__(self, checkpoint) -> None:
        self.tagger = pipeline("ner", 
                               model=checkpoint, 
                               tokenizer=AutoTokenizer.from_pretrained(checkpoint, model_max_length=512), 
                               device=0, # device=0 means using GPU
                               aggregation_strategy='simple')  

    def get_prediction(self, text):
        
        def bio_tagging(sentence, tokens):
            # Tokenize the sentence by spaces
            words = sentence.split()

            # Create a list of BIO tags initialized as 'O' (outside any entity)
            bio_tags = ['O'] * len(words)

            # Iterate over the tokens to assign BIO tags
            for token in tokens:
                entity_group = token['entity_group']
                word = token['word']
                start = token['start']
                end = token['end']
                
                # Get the substring of the sentence from start to end to match with the token word
                token_str = sentence[start:end]
                
                # Split the sentence into words and get the indexes of words that match the token string
                token_words = token_str.split()

                # Find the position of the token_words in the sentence
                index = 0
                for i, word in enumerate(words):
                    if word == token_words[index]:
                        # Found the start of the token, now mark it as B-<entity>
                        if index == 0:
                            bio_tags[i] = f'B-{entity_group}'
                        else:
                            bio_tags[i] = f'I-{entity_group}'
                        index += 1
                        
                        # If all words of the token are processed, break the loop
                        if index == len(token_words):
                            break

            return bio_tags
        
        def list_entity_scores(entities_dict):
            scores = []
            for entry in entities_dict:
                scores.append(entry['score'])
            return scores

        entities_dict = self.tagger(text)
        bio_tags = bio_tagging(text, entities_dict)
        scores = list_entity_scores(entities_dict)

        """
        print('Esse eh o TOKENS')
        print(entities_dict)
        
        print('Esse eh o TEXT')
        print(text)

        print('Esse eh o BIO_TAGS')
        print(bio_tags)

        print('Esse eh o SCORES')
        print(scores)
        """

        return {"entities": bio_tags, "scores": scores}
