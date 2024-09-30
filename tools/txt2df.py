import pandas as pd

def create_dataframe_from_txt(file_path, file):
    # Read the text file and split each line into token and NER class
    with open(f"{file_path}/{file}", 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if line != '' and line != '\n':
            line = line.strip().split(' ')
            token = line[0].lower()
            ner_token = line[1]
            data.append((token, ner_token))
        else:
            token = '\n'
            ner_token = '\n'
            data.append((token, ner_token))

    # Group tokens into sentences considering the '\n' line breaks
    sentences = []
    tokens = []
    ner_tokens = []
    sentence = []
    ner_token = []

    for token, ner_token_str in data:
        if token != '\n':
            sentence.append(token)
            ner_token.append(ner_token_str)
        else:
            sentences.append(' '.join(sentence))
            tokens.append(sentence)
            ner_tokens.append(ner_token)
            sentence = []
            ner_token = []

    # Create the DataFrame
    df = pd.DataFrame({
        'sentences': sentences,
        'tokens': tokens,
        'ner_tokens': ner_tokens
    })

    df['num_tokens'] = df['sentences'].str.split().apply(len)
    df = df[df['num_tokens'] > 1].drop(columns='num_tokens')
    
    return df

def create_dataframe_from_txt_results(file_path):
    # Read the text file and split each line into token and NER class
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if line != '' and line != '\n':
            line = line.strip().split(' ')
            token = line[0].lower()
            ner_token = line[1]
            model_ner_token = line[2]
            data.append((token, ner_token,model_ner_token))
        else:
            token = '\n'
            ner_token = '\n'
            model_ner_token = '\n'
            data.append((token, ner_token, model_ner_token))

    # Group tokens into sentences considering the '\n' line breaks
    sentences = []
    tokens = []
    ner_tokens = []
    model_result = []

    sentence = []
    ner_token = []
    model_results = []

    for token, ner_token_str, result in data:
        if token != '\n':
            sentence.append(token)
            ner_token.append(ner_token_str)
            model_result.append(result)
        else:
            sentences.append(' '.join(sentence))
            tokens.append(sentence)
            ner_tokens.append(ner_token)
            model_results.append(model_result)
            sentence = []
            ner_token = []
            model_result = []

    # Create the DataFrame
    df = pd.DataFrame({
        'sentences': sentences,
        'tokens': tokens,
        'ner_tokens': ner_tokens,
        'ner_tokens_results': model_results
    })

    df['num_tokens'] = df['sentences'].str.split().apply(len)
    df = df[df['num_tokens'] > 1].drop(columns='num_tokens')
    
    return df