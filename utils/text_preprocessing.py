import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def lower_case(df: pd.DataFrame) -> pd.DataFrame:
    '''
    param: df - pd.DataFrame
    return: возвращает датафрейм в нижнем регистре
    '''
    return df.applymap(lambda x: x.lower())

def remove_urls(df: pd.DataFrame) -> pd.DataFrame:
    '''
    param: df - pd.DataFrame
    return: удаляет ссылки
    '''
    url_pattern = re.compile(r'https?://\S+')

    def remove_urls(text):
        return url_pattern.sub('', text)
    
    df['text'] = df['text'].apply(remove_urls)

    return df

def tokenize(df: pd.DataFrame) -> pd.DataFrame:
    '''
    param: df - pd.DataFrame
    return: возвращает датафрейм токенами
    '''
    df['text'] = df['text'].apply(word_tokenize)
    return df

def remove_stop_words(df: pd.DataFrame) -> pd.DataFrame:
    '''
    param: df - pd.DataFrame
    return: удаляет стоп-слова
    '''
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])
    return df

def lemmatize(df: pd.DataFrame) -> pd.DataFrame:
    '''
    param: df - pd.DataFrame
    return: лемматизирует токены
    '''
    lemmatizer = WordNetLemmatizer()

    # текст должен быть токенизирован
    df['lemmatized_tokens'] = df['text'].apply(lambda x: [lemmatizer.lemmatize(el) for el in x])

    df['preprocessed_text'] = df['lemmatized_tokens'].apply(lambda x: ' '.join(x))

    return df

def full_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    '''
    param: df - pd.DataFrame
    return: применяет все функции этого скрипта
    '''
    df = lower_case(df)
    df = remove_urls(df)
    df = tokenize(df)
    df = remove_stop_words(df)
    df = lemmatize(df)
    
    return df

