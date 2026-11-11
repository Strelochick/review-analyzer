"""Построение графиков для веб-интерфейса и отчёта."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ml

STATIC_DIR = os.path.join(ml.BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)


def rating_distribution(df, path=None):
    path = path or os.path.join(STATIC_DIR, 'ratings.png')
    counts = df['rating'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=['#e74c3c', '#f39c12', '#f1c40f', '#7bc96f', '#27ae60'])
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y + 2, str(y), ha='center')
    ax.set_title('Распределение оценок')
    ax.set_xlabel('Оценка')
    ax.set_ylabel('Количество отзывов')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def daily_dynamics(df, path=None):
    path = path or os.path.join(STATIC_DIR, 'daily.png')
    daily = ml.daily_counts(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily['date'], daily['reviews_count'], color='#2a75bb', marker='o', markersize=3)
    ax.plot(daily['date'], daily['reviews_count'].rolling(7, min_periods=1).mean(),
            color='#e3350d', linewidth=2, label='Среднее за 7 дней')
    ax.set_title('Динамика отзывов по дням')
    ax.set_ylabel('Отзывов в день')
    ax.legend()
    ax.grid(alpha=.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def word_cloud(df, path=None, sentiment=None):
    """Облако слов (библиотека wordcloud)."""
    from wordcloud import WordCloud
    path = path or os.path.join(STATIC_DIR, 'wordcloud.png')
    data = df if sentiment is None else df[df['sentiment'] == sentiment]
    text = ' '.join(' '.join(ml.tokenize(t)) for t in data['text'])
    cloud = WordCloud(width=900, height=420, background_color='white', colormap='viridis',
                      max_words=80, random_state=42).generate(text)
    cloud.to_file(path)
    return path
