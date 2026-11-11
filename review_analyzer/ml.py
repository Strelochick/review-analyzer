"""
Основные функции ML-продукта для анализа отзывов.
Используются в main.py (отчёт), compare_models.py (сравнение моделей) и app.py (веб-интерфейс).
"""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'reviews_dataset.csv')

# стоп-слова, которые не несут смысла при подсчёте частых слов
STOP_WORDS = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему когда даже ну ли если уже или ни быть был него до вас там потом
себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам без будто чего раз тоже себе под
будет тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
куда при два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя
свою этой перед иногда чуть том такой им более всегда всю между очень из это
плюсов минусов хотя""".split())


# ---------- данные ----------
def load_data(path=DATA_PATH):
    """Загрузка датасета и разметка тональности (4-5 позитив, 1-2 негатив, 3 нейтрально)."""
    df = pd.read_csv(path, dtype={'phone': str})
    df['date'] = pd.to_datetime(df['date'])
    df['sentiment'] = df['rating'].apply(lambda r: 1 if r >= 4 else (0 if r <= 2 else np.nan))
    return df


def clean_text(text):
    text = text.lower().replace('ё', 'е')
    return re.sub(r'[^а-яa-z\s]', ' ', text)


def tokenize(text):
    return [w for w in clean_text(text).split() if w not in STOP_WORDS and len(w) > 2]


# ---------- защита персональных данных ----------
def mask_phone(phone):
    phone = str(phone)
    if len(phone) >= 10:
        return phone[:4] + '***' + phone[-4:]
    return phone


def mask_name(name):
    name = str(name)
    if len(name) <= 3:
        return name[0] + '*' * (len(name) - 1)
    return name[:2] + '*' * (len(name) - 4) + name[-2:]


def protect(df):
    protected = df.copy()
    protected['user_name'] = protected['user_name'].apply(mask_name)
    protected['phone'] = protected['phone'].apply(mask_phone)
    return protected


# ---------- классификация тональности ----------
class SentimentModel:
    """
    TF-IDF + логистическая регрессия (как на лекции).
    Векторизация по символьным n-граммам (2-4 символа внутри слов): так модель узнаёт
    разные формы одного слова («разочарован», «разочаровали») в новых отзывах.
    class_weight='balanced' - чтобы модель не перекашивало в сторону позитива
    (позитивных отзывов в датасете в 2 раза больше).
    """

    def __init__(self, model=None):
        self.vectorizer = TfidfVectorizer(preprocessor=clean_text, analyzer='char_wb',
                                          ngram_range=(2, 4), sublinear_tf=True)
        self.model = model or LogisticRegression(max_iter=2000, C=5, class_weight='balanced')

    def fit(self, texts, y):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, y)
        return self

    def predict_proba(self, texts):
        return self.model.predict_proba(self.vectorizer.transform(texts))


def train_sentiment(df, test_size=0.2):
    data = df[df['sentiment'].notna()]
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['sentiment'].astype(int), test_size=test_size,
        random_state=42, stratify=data['sentiment'])
    model = SentimentModel().fit(X_train, y_train)
    return model, (X_train, X_test, y_train, y_test)


def predict_sentiment(text, model):
    """Возвращает тональность отзыва («позитивный»/«негативный») и % уверенности."""
    proba = model.predict_proba([text])[0]
    label = int(np.argmax(proba))
    return ('позитивный' if label == 1 else 'негативный'), round(float(proba[label]) * 100, 1)


# ---------- аналитика ----------
def top_products_by_positive(df, n=5):
    """Топ товаров по доле позитивных отзывов (среди позитивных и негативных)."""
    data = df[df['sentiment'].notna()]
    stats = data.groupby('product').agg(
        Отзывов=('sentiment', 'size'),
        Позитивных=('sentiment', 'sum'))
    stats['Доля позитива, %'] = (stats['Позитивных'] / stats['Отзывов'] * 100).round(1)
    stats['Позитивных'] = stats['Позитивных'].astype(int)
    return stats.sort_values('Доля позитива, %', ascending=False).head(n).reset_index()


def top_negative_words(df, n=5):
    """5 самых частых слов в негативных отзывах."""
    words = Counter()
    for text in df[df['sentiment'] == 0]['text']:
        words.update(tokenize(text))
    return words.most_common(n)


def product_stats(df):
    """Статистика по товарам: средний рейтинг, количество отзывов, доля позитива."""
    stats = df.groupby('product').agg(
        avg_rating=('rating', 'mean'),
        reviews=('rating', 'size'),
        positive=('rating', lambda r: (r >= 4).mean() * 100),
        negative=('rating', lambda r: (r <= 2).mean() * 100))
    return stats.round(2).sort_values('avg_rating', ascending=False).reset_index()


def daily_counts(df):
    daily = df.groupby(df['date'].dt.date).size().reset_index(name='reviews_count')
    daily['date'] = pd.to_datetime(daily['date'])
    full = pd.DataFrame({'date': pd.date_range(daily['date'].min(), daily['date'].max())})
    return full.merge(daily, how='left').fillna(0)


def forecast_reviews(df, days=3):
    """Прогноз количества отзывов (линейная регрессия по дню недели и лагам, как на лекции)."""
    daily = daily_counts(df)
    daily['day_of_week'] = daily['date'].dt.dayofweek
    for lag in [1, 2, 3]:
        daily[f'lag_{lag}'] = daily['reviews_count'].shift(lag)
    daily = daily.dropna()
    cols = ['day_of_week', 'lag_1', 'lag_2', 'lag_3']
    model = LinearRegression().fit(daily[cols].values[:-days], daily['reviews_count'].values[:-days])
    pred = model.predict(daily[cols].values[-days:])
    return daily['date'].values[-days:], daily['reviews_count'].values[-days:], np.maximum(pred, 0)


# ---------- рекомендации ----------
def content_recommend(df, product_name, top_n=3):
    """Контентная рекомендация похожих товаров по текстам отзывов (как на лекции)."""
    texts = df.groupby('product')['text'].apply(' '.join).reset_index()
    vectors = TfidfVectorizer(preprocessor=clean_text).fit_transform(texts['text'])
    sim = cosine_similarity(vectors)
    idx = texts[texts['product'] == product_name].index[0]
    order = [i for i in sim[idx].argsort()[::-1] if i != idx][:top_n]
    return [texts.iloc[i]['product'] for i in order]


def user_item_matrix(df):
    return df.pivot_table(index='user_id', columns='product', values='rating', aggfunc='mean')


def recommend_for_user(df, user_id, top_n=3, k=20):
    """
    User-based collaborative filtering.
    1) строим матрицу пользователь x товар с оценками;
    2) из каждой оценки вычитаем среднюю оценку товара - остаётся личное отношение
       пользователя к товару (понравился больше или меньше, чем остальным);
    3) находим k самых похожих пользователей (косинусная близость этих отклонений);
    4) прогноз = средняя оценка товара + взвешенное отклонение похожих пользователей.
    Если пользователь новый (нет оценок) — рекомендуем самые высокооценённые товары.
    """
    matrix = user_item_matrix(df)
    if user_id not in matrix.index:
        top = product_stats(df).head(top_n)
        return [(p, round(r, 2), 'популярный товар') for p, r in zip(top['product'], top['avg_rating'])]

    item_mean = matrix.mean(axis=0)
    resid = matrix.sub(item_mean, axis=1)
    sims = pd.Series(cosine_similarity(resid.fillna(0))[matrix.index.get_loc(user_id)], index=matrix.index)
    neighbours = sims.drop(user_id)
    neighbours = neighbours[neighbours > 0].sort_values(ascending=False).head(k)

    predictions = []
    for product in matrix.columns[matrix.loc[user_id].isna()]:
        score = item_mean[product]
        rated = neighbours[resid.loc[neighbours.index, product].notna()]
        if rated.sum() > 0:
            score += (resid.loc[rated.index, product] * rated).sum() / rated.sum()
        predictions.append((product, round(float(np.clip(score, 1, 5)), 2),
                            f'похожих пользователей: {len(rated)}'))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


def evaluate_recommender(df, test_share=0.2, seed=42):
    """Проверка рекомендаций: прячем часть оценок и сравниваем предсказание с реальной оценкой (MAE)."""
    test = df.sample(frac=test_share, random_state=seed)
    train = df.drop(test.index)
    errors = []
    for _, row in test.iterrows():
        recs = dict((p, s) for p, s, _ in recommend_for_user(train, row['user_id'], top_n=20))
        if row['product'] in recs:
            errors.append(abs(recs[row['product']] - row['rating']))
    item_mean = train.groupby('product')['rating'].mean()
    baseline = (test['product'].map(item_mean) - test['rating']).abs().mean()
    return round(float(np.mean(errors)), 3), round(float(baseline), 3), len(errors)
