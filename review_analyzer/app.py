"""
Веб-интерфейс ML-продукта для анализа отзывов (Flask).
Запуск: python app.py  ->  http://127.0.0.1:5000
"""
import os
import uuid
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, g, make_response

import charts
import db
import ml

app = Flask(__name__)

db.init_db()
state = {'count': None, 'model': None}


def refresh():
    """Переобучение модели и перестроение графиков, если в базе появились новые отзывы."""
    df = db.load_reviews()
    if state['count'] != len(df):
        state['model'], _ = ml.train_sentiment(df)
        charts.rating_distribution(df)
        charts.daily_dynamics(df)
        try:
            charts.word_cloud(df)
        except ImportError:
            pass  # если библиотека wordcloud не установлена, остаётся готовая картинка
        state['count'] = len(df)
    return df


@app.before_request
def identify_user():
    """Каждый посетитель получает уникальный ID, который хранится в cookies."""
    g.user_id = request.cookies.get('user_id')
    g.new_user = g.user_id is None
    if g.new_user:
        g.user_id = uuid.uuid4().hex[:12]


@app.after_request
def save_cookie(response):
    if g.get('new_user'):
        response.set_cookie('user_id', g.user_id, max_age=60 * 60 * 24 * 365)
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    df = refresh()
    result = None
    form = {'text': '', 'rating': '5', 'product': ''}
    if request.method == 'POST':
        form = {k: request.form.get(k, '') for k in form}
        text = form['text'].strip()
        if text:
            label, conf = ml.predict_sentiment(text, state['model'])
            result = {'label': label, 'conf': conf, 'saved': False}
            if request.form.get('action') == 'save' and form['product']:
                db.add_review(text, form['rating'], form['product'], g.user_id, label,
                              datetime.now().strftime('%Y-%m-%d %H:%M'))
                result['saved'] = True
                df = refresh()
    recs = ml.recommend_for_user(df, g.user_id, top_n=3)
    history = db.user_reviews(g.user_id)
    stamp = state['count']
    return render_template('index.html', page='index', result=result, form=form, recs=recs,
                           history=history.to_dict('records'), products=sorted(df['product'].unique()),
                           user_id=g.user_id, total=len(df), avg=round(df['rating'].mean(), 2),
                           positive=round((df['rating'] >= 4).mean() * 100, 1), stamp=stamp,
                           demo_users=sorted(df['user_id'].unique(), key=lambda u: (len(u), u))[:15])


@app.route('/login/<user_id>')
def login_as(user_id):
    """Демо-режим: войти под пользователем из датасета, чтобы увидеть рекомендации по его истории."""
    response = make_response(redirect(url_for('index')))
    response.set_cookie('user_id', user_id, max_age=60 * 60 * 24 * 365)
    return response


@app.route('/products')
def products():
    df = refresh()
    stats = ml.product_stats(df)
    return render_template('products.html', page='products', stats=stats.to_dict('records'),
                           top_words=ml.top_negative_words(df, 5),
                           top_pos=ml.top_products_by_positive(df, 3).to_dict('records'))


@app.route('/models')
def models():
    path = os.path.join(ml.BASE_DIR, 'models_metrics.csv')
    table = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    mae, baseline, n = ml.evaluate_recommender(refresh())
    return render_template('models.html', page='models', cols=list(table.columns),
                           rows=table.to_dict('records'), best=table.iloc[0].to_dict() if len(table) else None,
                           mae=mae, baseline=baseline, n=n)


if __name__ == '__main__':
    app.run(debug=True)
