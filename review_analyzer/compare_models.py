"""
Сравнение моделей классификации тональности: LogisticRegression, RandomForest, XGBoost.
Результат: таблица метрик в консоли, файл models_metrics.csv, график static/models.png
Запуск: python compare_models.py
"""
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

import ml
import charts

warnings.filterwarnings('ignore')

df = ml.load_data()
data = df[df['sentiment'].notna()]
X_text, y = data['text'], data['sentiment'].astype(int)

vectorizer = TfidfVectorizer(preprocessor=ml.clean_text, ngram_range=(1, 2))
X = vectorizer.fit_transform(X_text)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, C=5),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             eval_metric='logloss', random_state=42),
}

rows = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
    rows.append({
        'Модель': name,
        'Accuracy': round(accuracy_score(y_test, pred), 3),
        'Precision': round(precision_score(y_test, pred), 3),
        'Recall': round(recall_score(y_test, pred), 3),
        'F1': round(f1_score(y_test, pred), 3),
        'F1 (кросс-валидация)': round(cv_f1, 3),
    })

table = pd.DataFrame(rows).sort_values(['F1 (кросс-валидация)', 'F1'], ascending=False)
print('Сравнение моделей классификации тональности')
print(f'Обучающая выборка: {X_train.shape[0]}, тестовая: {X_test.shape[0]}, признаков TF-IDF: {X.shape[1]}\n')
print(table.to_string(index=False))
table.to_csv(os.path.join(ml.BASE_DIR, 'models_metrics.csv'), index=False, encoding='utf-8-sig')

best = table.iloc[0]
print(f'\nВывод: лучшая модель - {best["Модель"]} '
      f'(F1 на тесте = {best["F1"]}, F1 на кросс-валидации = {best["F1 (кросс-валидация)"]}).')

# график сравнения
ax = table.set_index('Модель')[['Accuracy', 'Precision', 'Recall', 'F1']].plot(
    kind='bar', figsize=(9, 4.5), rot=0, colormap='viridis')
ax.set_ylim(0.5, 1.02)
ax.set_title('Сравнение моделей классификации')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(charts.STATIC_DIR, 'models.png'), dpi=110)

# облако слов (используется в веб-интерфейсе)
charts.word_cloud(df)
charts.word_cloud(df, os.path.join(charts.STATIC_DIR, 'wordcloud_neg.png'), sentiment=0)
print('Облака слов сохранены в папку static')
