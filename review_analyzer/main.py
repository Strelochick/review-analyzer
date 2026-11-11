"""
Практическая работа №10 - Разработка ML-продукта для анализа отзывов.
Код с лекции, адаптированный под собственный датасет и дополненный функциями:
  - predict_sentiment: «позитивный»/«негативный» и % уверенности;
  - top_products_by_positive: топ товаров по доле позитива;
  - top_negative_words: 5 самых частых слов в негативных отзывах.
Вся статистика и графики сохраняются в student_report.png
Запуск: python main.py
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import ml

warnings.filterwarnings('ignore')

print('\nЗагружаем и подготавливаем данные...')
df = ml.load_data()
print(f'Загружено {len(df)} отзывов, товаров: {df["product"].nunique()}, пользователей: {df["user_id"].nunique()}')
print(f'Период: {df["date"].min():%d.%m.%Y} - {df["date"].max():%d.%m.%Y}')

print('\n=== КЛАССИФИКАЦИЯ ТЕКСТОВ ===')
model, (X_train, X_test, y_train, y_test) = ml.train_sentiment(df)
y_pred = model.model.predict(model.vectorizer.transform(X_test))
print(f'Обучающих примеров: {len(X_train)}, тестовых: {len(X_test)}')
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-score': f1_score(y_test, y_pred),
}
for name, value in metrics.items():
    print(f'   {name:10s} {value:.2%}')
cm = confusion_matrix(y_test, y_pred)

print('\n=== ФУНКЦИЯ predict_sentiment ===')
test_phrases = [
    'Отличный ноутбук, работает быстро и тихо, рекомендую',
    'Ужасное качество, сломался через неделю, деньги на ветер',
    'Ужасные наушники, постоянно отключается блютуз',
    'Звук чистый, хотя блютуз иногда отключается',
]
predictions = []
for phrase in test_phrases:
    label, conf = ml.predict_sentiment(phrase, model)
    predictions.append((phrase, label, conf))
    print(f'   "{phrase}" -> {label} ({conf}%)')

print('\n=== ЗАЩИТА ПЕРСОНАЛЬНЫХ ДАННЫХ ===')
print(f'Телефон: {df["phone"].iloc[0]} -> {ml.mask_phone(df["phone"].iloc[0])}')
print(f'Имя: {df["user_name"].iloc[0]} -> {ml.mask_name(df["user_name"].iloc[0])}')
df_protected = ml.protect(df)
df_protected.to_csv('reviews_protected.csv', index=False, encoding='utf-8-sig')
print(f'Создана защищённая версия ({len(df_protected)} записей) -> reviews_protected.csv')

print('\n=== ВРЕМЕННЫЕ РЯДЫ (прогноз на 3 дня) ===')
dates, actual, forecast = ml.forecast_reviews(df)
for d, a, p in zip(dates, actual, forecast):
    print(f'   {str(d)[:10]}: факт = {int(a):2d} | прогноз = {p:.1f}')

print('\n=== ФУНКЦИЯ top_products_by_positive ===')
top_products = ml.top_products_by_positive(df, n=5)
print(top_products.to_string(index=False))

print('\n=== ФУНКЦИЯ top_negative_words ===')
neg_words = ml.top_negative_words(df, n=5)
for word, count in neg_words:
    print(f'   {word}: {count}')

print('\n=== РЕКОМЕНДАЦИИ (контентные, как на лекции) ===')
for p in ['Смартфон Nova X', 'Наушники SoundPro']:
    print(f'Похожие на "{p}": {", ".join(ml.content_recommend(df, p))}')

# ---------------- student_report.png ----------------
print('\n=== ФОРМИРОВАНИЕ ОТЧЁТА student_report.png ===')
fig = plt.figure(figsize=(20, 15))
fig.suptitle('Отчёт: анализ отзывов интернет-магазина электроники', fontsize=18, fontweight='bold', y=0.94)
grid = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.55)

ax = fig.add_subplot(grid[0, 0])
counts = df['rating'].value_counts().sort_index()
ax.bar(counts.index, counts.values, color='royalblue', edgecolor='black', alpha=.8)
ax.set_title('Распределение оценок')
ax.set_xlabel('Оценка')

ax = fig.add_subplot(grid[0, 1:])
daily = ml.daily_counts(df)
ax.plot(daily['date'], daily['reviews_count'], 'b-o', markersize=3, label='Факт')
ax.plot(dates, forecast, 'o--', color='orange', linewidth=2, label='Прогноз (последние 3 дня)')
ax.set_title('Динамика отзывов и прогноз')
ax.legend()
ax.tick_params(axis='x', rotation=30)

ax = fig.add_subplot(grid[1, 0])
ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Негатив', 'Позитив'])
ax.set_yticklabels(['Негатив', 'Позитив'])
ax.set_xlabel('Предсказано')
ax.set_ylabel('Факт')
ax.set_title('Матрица ошибок (LogisticRegression)')

ax = fig.add_subplot(grid[1, 1])
ax.barh(top_products['product'][::-1], top_products['Доля позитива, %'][::-1], color='seagreen')
for i, v in enumerate(top_products['Доля позитива, %'][::-1]):
    ax.text(v - 12, i, f'{v}%', va='center', color='white', fontweight='bold')
ax.set_title('Топ-5 товаров по доле позитива')
ax.set_xlim(0, 100)

ax = fig.add_subplot(grid[1, 2])
words, values = zip(*neg_words)
ax.bar(words, values, color='coral')
ax.set_title('5 самых частых слов в негативных отзывах')
ax.tick_params(axis='x', rotation=20)

ax = fig.add_subplot(grid[2, 0])
binary = df['sentiment'].value_counts().sort_index()
ax.pie(binary.values, labels=['Негатив', 'Позитив'], colors=['coral', 'lightgreen'], autopct='%1.1f%%')
ax.set_title('Соотношение тональности')

ax = fig.add_subplot(grid[2, 1:])
ax.axis('off')
lines = ['МЕТРИКИ МОДЕЛИ:  ' + ',  '.join(f'{k} = {v:.1%}' for k, v in metrics.items()), '',
         'ПРОВЕРКА predict_sentiment:']
lines += [f'   «{p}» -> {l} ({c}%)' for p, l, c in predictions]
lines += ['', 'ТОП ТОВАРОВ ПО ДОЛЕ ПОЗИТИВА: ' + ', '.join(
    f'{p} ({v}%)' for p, v in zip(top_products['product'][:3], top_products['Доля позитива, %'][:3]))]
lines += ['ЧАСТЫЕ СЛОВА В НЕГАТИВЕ: ' + ', '.join(f'{w} ({c})' for w, c in neg_words)]
lines += [f'ВСЕГО ОТЗЫВОВ: {len(df)},  СРЕДНИЙ РЕЙТИНГ: {df["rating"].mean():.2f}']
ax.text(0, 1, '\n'.join(lines), va='top', fontsize=12, family='monospace')
ax.set_title('Сводная статистика', loc='left')

fig.savefig('student_report.png', dpi=110, bbox_inches='tight')
print('Отчёт сохранён в student_report.png')
