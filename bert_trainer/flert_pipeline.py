from flair.models import SequenceTagger
from flair.data import Sentence

class Pipeline:
    def __init__(self, checkpoint) -> None:
        self.tagger = SequenceTagger.load('{checkpoint}/final-model.pt'.format(checkpoint=checkpoint))

    def split_text(self, text, max_length):
        tokens = text.split()
        sentences = []

        for i in range(0, len(tokens), max_length):
            sentences.append(' '.join(tokens[i:i + max_length]))

        return sentences
        
        
    def get_prediction(self, text, max_length=512):
        # split sentence (> 512)
        sentences_split = self.split_text(text, max_length)

        for sentence_splited in sentences_split:
            # make a sentence
            sentence = Sentence(sentence_splited)

            # predict NER tags
            self.tagger.predict(sentence)

            scores = []

            # transfer entity labels to token level
            for entity in sentence.get_spans('ner'):
                prefix = 'B-'
                for token in entity:
                    token.set_label('ner-bio', prefix + entity.tag, entity.score)
                    prefix = 'I-'
                    scores.append(entity.score)

            # now go through all tokens and print label
            bio_tokens = []
            for token in sentence:
                try:
                    #print(token.text, token.tag)
                    bio_tokens.append(token.tag)
                except:
                    #print(token.text, "O")
                    bio_tokens.append("O")
                    
        return {"entities": bio_tokens, "scores": scores}