"""Хранение отзывов в SQLite (файл reviews.db)."""
import os
import sqlite3

import pandas as pd

import ml

DB_PATH = os.path.join(ml.BASE_DIR, 'reviews.db')


def connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Создание таблицы и первичная загрузка отзывов из CSV (только при первом запуске)."""
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                product TEXT NOT NULL,
                date TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT,
                phone TEXT,
                predicted TEXT
            )""")
        count = con.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
        if count == 0:
            df = pd.read_csv(ml.DATA_PATH, dtype={'phone': str})
            df['user_id'] = 'u' + df['user_id'].astype(str)
            # персональные данные сохраняем в базе только в замаскированном виде
            df['user_name'] = df['user_name'].apply(ml.mask_name)
            df['phone'] = df['phone'].apply(ml.mask_phone)
            df['predicted'] = None
            df.drop(columns='id').to_sql('reviews', con, if_exists='append', index=False)


def load_reviews():
    with connect() as con:
        df = pd.read_sql('SELECT * FROM reviews', con)
    df['date'] = pd.to_datetime(df['date'])
    df['sentiment'] = df['rating'].apply(lambda r: 1 if r >= 4 else (0 if r <= 2 else None))
    return df


def add_review(text, rating, product, user_id, predicted, date):
    with connect() as con:
        con.execute('INSERT INTO reviews (text, rating, product, date, user_id, user_name, phone, predicted) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (text, int(rating), product, date, user_id, 'Гость', None, predicted))


def user_reviews(user_id):
    with connect() as con:
        return pd.read_sql('SELECT product, rating, text, date, predicted FROM reviews '
                           'WHERE user_id = ? ORDER BY date DESC', con, params=(user_id,))
