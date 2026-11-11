"""
Генерация собственного датасета отзывов интернет-магазина электроники.
Поля: id, text, rating, product, date, user_id, user_name, phone.
Запуск: python generate_dataset.py  ->  reviews_dataset.csv
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

# товар: (категория, базовое качество)
PRODUCTS = {
    'Смартфон Nova X': ('phone', 4.1),
    'Ноутбук AirBook 14': ('laptop', 4.3),
    'Наушники SoundPro': ('audio', 3.9),
    'Колонка BoomBox': ('audio', 3.4),
    'Умные часы FitTime': ('wear', 3.6),
    'Фитнес-браслет Pulse': ('wear', 2.9),
    'Планшет TabOne': ('laptop', 3.2),
    'Электронная книга ReadMe': ('reader', 4.0),
    'Роутер NetLink': ('net', 2.6),
    'Внешний аккумулятор PowerGo': ('power', 3.7),
}

POS_ASPECTS = {
    'phone': ['камера снимает отлично', 'экран яркий и сочный', 'батарея держит два дня', 'телефон работает быстро, ничего не тормозит'],
    'laptop': ['экран четкий, работать приятно', 'очень легкий и тонкий', 'работает быстро и тихо', 'клавиатура удобная'],
    'audio': ['звук чистый и объемный', 'басы мощные', 'громкости хватает с запасом', 'подключение по блютуз без проблем'],
    'wear': ['точно считает шаги и пульс', 'ремешок удобный', 'заряд держит неделю', 'приложение понятное'],
    'reader': ['экран не утомляет глаза', 'заряда хватает на месяц', 'удобно читать вечером с подсветкой', 'легкая, помещается в сумку'],
    'net': ['сигнал стабильный по всей квартире', 'скорость интернета выросла', 'настроил за пять минут', 'работает без перебоев'],
    'power': ['заряжает телефон быстро', 'емкости хватает на несколько зарядок', 'компактный и легкий', 'корпус прочный'],
}
NEG_ASPECTS = {
    'phone': ['камера снимает мыльно', 'батарея садится к обеду', 'телефон сильно греется', 'постоянно зависает'],
    'laptop': ['сильно греется и шумит', 'экран бликует', 'батареи хватает на пару часов', 'тормозит даже браузер'],
    'audio': ['звук плоский и глухой', 'постоянно отключается блютуз', 'хрипит на большой громкости', 'батарея быстро садится'],
    'wear': ['пульс считает неправильно', 'ремешок натирает руку', 'зарядка держит один день', 'приложение постоянно вылетает'],
    'reader': ['экран медленно перелистывает страницы', 'подсветка неравномерная', 'зависает при открытии книг', 'корпус скрипит'],
    'net': ['интернет постоянно пропадает', 'сигнал не доходит до соседней комнаты', 'приходится перезагружать каждый день', 'скорость сильно режет'],
    'power': ['заряжает очень медленно', 'реальная емкость меньше заявленной', 'сильно нагревается', 'сломался через неделю'],
}
POS_OPEN = ['Отличный товар!', 'Очень доволен покупкой.', 'Рекомендую!', 'Супер, всем советую.', 'Покупкой полностью доволен.',
            'Прекрасное качество.', 'Лучшая покупка за год.', 'Замечательная вещь.', 'Хороший товар за свои деньги.', 'Все понравилось.']
NEG_OPEN = ['Ужасное качество.', 'Очень разочарован.', 'Не рекомендую.', 'Деньги на ветер.', 'Плохой товар.',
            'Пожалел о покупке.', 'Кошмар, а не товар.', 'Брак, пришлось сдавать.', 'Не советую покупать.', 'Сплошное разочарование.']
NEU_OPEN = ['Нормальный товар.', 'Товар средний.', 'В целом неплохо, но есть минусы.', 'Ожидал большего, но пользоваться можно.',
            'Обычный товар, ничего особенного.']
POS_END = ['Доставка быстрая.', 'Цена отличная.', 'Буду заказывать еще.', 'Спасибо магазину!', '', '']
NEG_END = ['Доставка задержалась на неделю.', 'Поддержка не отвечает.', 'Верните деньги!', 'Больше здесь не закажу.', '', '']

FIRST = ['Анна', 'Иван', 'Мария', 'Дмитрий', 'Елена', 'Сергей', 'Ольга', 'Алексей', 'Наталья', 'Андрей',
         'Екатерина', 'Павел', 'Юлия', 'Максим', 'Ирина', 'Никита', 'Светлана', 'Артем', 'Татьяна', 'Кирилл']
LAST = ['Иванов', 'Смирнов', 'Кузнецов', 'Попов', 'Соколов', 'Лебедев', 'Козлов', 'Новиков', 'Морозов', 'Волков']

N_USERS = 90
N_REVIEWS = 600
# у каждого пользователя свои предпочтения по категориям (для коллаборативной фильтрации)
GROUPS = [
    {'audio': 1.3, 'phone': 0.8, 'laptop': -1.0, 'wear': -0.6},
    {'laptop': 1.2, 'reader': 1.0, 'audio': -1.1, 'power': -0.5},
    {'wear': 1.3, 'power': 0.9, 'net': -0.8, 'reader': -0.7},
    {'net': 1.3, 'phone': -1.0, 'reader': -0.8, 'audio': 0.5},
]


def make_user(uid):
    first = random.choice(FIRST)
    last = random.choice(LAST)
    if first[-1] in 'ая':
        last += 'а'
    phone = f"+79{random.randint(10, 99)}{random.randint(1000000, 9999999)}"
    return {'user_id': uid, 'user_name': f'{first} {last}', 'phone': phone, 'group': GROUPS[uid % len(GROUPS)]}


def make_text(category, rating):
    hard = random.random() < 0.3  # «сложный» отзыв: без явной оценки в начале и со смешанными аспектами
    if hard and rating != 3:
        good, bad = random.choice(POS_ASPECTS[category]), random.choice(NEG_ASPECTS[category])
        if rating >= 4:
            return f'{good.capitalize()}, хотя {bad}. ' + random.choice(['В целом устраивает.', 'Пользуюсь каждый день.', ''])
        return f'{bad.capitalize()}, хотя {good}. ' + random.choice(['Вернуть не получилось.', 'Не стоит своих денег.', ''])
    if rating >= 4:
        parts = [random.choice(POS_OPEN)] + [a.capitalize() + '.' for a in random.sample(POS_ASPECTS[category], 2)]
        if rating == 4 and random.random() < 0.5:
            parts.append('Из минусов: ' + random.choice(NEG_ASPECTS[category]) + '.')
        parts.append(random.choice(POS_END))
    elif rating <= 2:
        parts = [random.choice(NEG_OPEN)] + [a.capitalize() + '.' for a in random.sample(NEG_ASPECTS[category], 2)]
        if rating == 2 and random.random() < 0.4:
            parts.append('Из плюсов только то, что ' + random.choice(POS_ASPECTS[category]) + '.')
        parts.append(random.choice(NEG_END))
    else:
        parts = [random.choice(NEU_OPEN), random.choice(POS_ASPECTS[category]).capitalize() + ', но ' +
                 random.choice(NEG_ASPECTS[category]) + '.']
    return ' '.join(p for p in parts if p)


def main():
    users = [make_user(uid) for uid in range(1, N_USERS + 1)]
    end = datetime(2026, 9, 30)
    rows = []
    used = set()
    while len(rows) < N_REVIEWS:
        user = random.choice(users)
        product = random.choice(list(PRODUCTS))
        if (user['user_id'], product) in used:  # один отзыв на товар от пользователя
            continue
        used.add((user['user_id'], product))
        category, quality = PRODUCTS[product]
        value = quality + user['group'].get(category, 0) + np.random.normal(-0.3, 0.7)
        rating = int(np.clip(round(value), 1, 5))
        # больше отзывов в выходные и ближе к концу периода
        day = int(np.random.triangular(0, 90, 90))
        date = end - timedelta(days=90 - day, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        rows.append({
            'id': len(rows) + 1,
            'text': make_text(category, rating),
            'rating': rating,
            'product': product,
            'date': date.strftime('%Y-%m-%d %H:%M'),
            'user_id': user['user_id'],
            'user_name': user['user_name'],
            'phone': user['phone'],
        })
    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    df['id'] = range(1, len(df) + 1)
    df.to_csv('reviews_dataset.csv', index=False, encoding='utf-8')
    print(f'Создано {len(df)} отзывов, {df["user_id"].nunique()} пользователей, {df["product"].nunique()} товаров')
    print(df['rating'].value_counts().sort_index())


if __name__ == '__main__':
    main()
