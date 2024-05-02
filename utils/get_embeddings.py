from transformers import pipeline, AutoTokenizer, AutoModel 
import pandas as pd
import torch
from tqdm.notebook import tqdm


def load_df(path: str) -> pd.DataFrame:
    '''
    param: path - путь до датафрейма
    return: dataframe object
    '''
    df = pd.read_csv(path)
    if 'Unnamed: 0' in list(df.columns):
        df.drop(['Unnamed: 0'], axis=1, inplace=True) 
    return df

def get_pipeline(model_name: str) -> pipeline:
    '''
    param: model_name - название модели
    return: pipeline
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModel.from_pretrained(model_name)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    return pipeline('feature-extraction', model=model, tokenizer=tokenizer,
                    padding=True, truncation=True, add_special_tokens=True,
                    return_tensors='pt')

def generate_embeddings(df: pd.DataFrame, model_name: str) -> dict[list[int], torch.Tensor]:
    '''
    param: df - датафрейм с текстами
    param: model_name - название модели
    return: словарь с метками классов и эмбеддингами 
    '''
    pipeline = get_pipeline(model_name)
    text_emb = {'label': [], 'embedding': []}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        label = 0 if row['text_type'] == 'ham' else 1
        word_embeddings = pipeline(text)
        text_emb['embedding'].append(torch.mean(word_embeddings.squeeze(0), dim=0))
        text_emb['label'].append(label)
    text_emb['embedding'] = torch.stack(text_emb['embedding'])

    return text_emb
