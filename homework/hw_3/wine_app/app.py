"""Демо для HW3. Приложение дополнено с помощью ChatGPT."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


APP_DIR = Path(__file__).parent
FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино',
}

st.set_page_config(page_title='Винный помощник', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def batch_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def prepare_batch(data, feature_names):
    """Проверить CSV и расположить признаки в порядке обучения модели."""
    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise ValueError('В CSV не хватает столбцов: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('В CSV нет строк с винами.')

    X = data.loc[:, feature_names].copy()
    try:
        X = X.apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as error:
        raise ValueError('Все характеристики вина должны быть числами.') from error
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Заполните пропуски и удалите бесконечные значения в CSV.')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red допустимы только 0 (белое) и 1 (красное).')
    return X


model = load_model()
test_data = load_test_data()
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
features = test_data.drop(columns='target')
feature_names = list(features.columns)
y_test = test_data['target']
p_test = test_proba(model, features)

st.title('Винный помощник')

st.sidebar.header('Настройки отбора')
threshold = st.sidebar.slider(
    'Порог хорошего вина',
    min_value=0.0,
    max_value=1.0,
    value=float(config['threshold']),
    step=0.01,
    key='threshold',
)

tab_single, tab_metrics, tab_batch, tab_weights = st.tabs([
    'Одно вино', 'Модельные метрики', 'Загрузка CSV', 'Коэффициенты'
])

with tab_single:
    st.subheader('Характеристики вина')
    with st.form('single_wine'):
        wine = {}
        columns = st.columns(2)
        for i, name in enumerate(feature_names):
            with columns[i % 2]:
                if name == 'is_red':
                    wine[name] = int(st.toggle('Красное вино', value=bool(features[name].median())))
                else:
                    wine[name] = st.number_input(
                        FEATURE_LABELS.get(name, name),
                        min_value=float(features[name].min()),
                        max_value=float(features[name].max()),
                        value=float(features[name].median()),
                        format='%.5f',
                        key=f'feature_{name}',
                    )
        submitted = st.form_submit_button('Оценить вино', type='primary')

    if submitted:
        X_single = pd.DataFrame([wine], columns=feature_names)
        st.session_state['single_probability'] = float(model.predict_proba(X_single)[0, 1])

    if 'single_probability' in st.session_state:
        probability = st.session_state['single_probability']
        st.metric('Вероятность хорошего вина', f'{probability:.1%}')
        if probability >= threshold:
            st.success('Хорошее вино')
        else:
            st.info('Посредственное вино')

with tab_metrics:
    st.subheader('Качество на тестовой выборке')
    y_pred = (p_test >= threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    for column, label, value in zip(
        st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]
    ):
        column.metric(label, f'{value:.3f}')
    st.caption(
        f'Accuracy: {accuracy_score(y_test, y_pred):.3f} · '
        f'ROC-AUC: {roc_auc_score(y_test, p_test):.3f} · '
        f'PR-AUC: {average_precision_score(y_test, p_test):.3f}'
    )

    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Посредственное', 'Хорошее']).plot(
            ax=ax, cmap='Blues', colorbar=False
        )
        ax.set_xlabel('Предсказание модели')
        ax.set_ylabel('Настоящий класс')
        ax.set_title('Матрица ошибок')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.write(f'**В подборке:** {tp} хороших и {fp} посредственных вин.')
        st.write(f'**Пропущено хороших вин:** {fn}.')
        st.info(
            'Ниже порог — больше хороших и посредственных вин в подборке. '
            'Выше порог — строже отбор, но больше пропущенных хороших вин.'
        )

    left, right = st.columns(2)
    with left:
        fpr, tpr, _ = roc_curve(y_test, p_test)
        current_fpr = fp / (fp + tn) if fp + tn else 0.0
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label='Модель')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Случайная модель')
        ax.scatter(current_fpr, recall, s=80, color='tab:orange', zorder=3,
                   label=f'Порог {threshold:.2f}')
        ax.set(xlabel='FPR', ylabel='Recall (TPR)', title='ROC-кривая',
               xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        curve_precision, curve_recall, _ = precision_recall_curve(y_test, p_test)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(curve_recall, curve_precision, label='Модель')
        ax.axhline(float(y_test.mean()), linestyle='--', color='gray',
                   label='Случайная модель')
        point_precision = precision if tp + fp else 1.0
        ax.scatter(recall, point_precision, s=80, color='tab:orange', zorder=3,
                   label=f'Порог {threshold:.2f}')
        ax.set(xlabel='Recall', ylabel='Precision', title='PR-кривая',
               xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if tp + fp == 0:
            st.caption('Рекомендаций нет; precision принят за 0.')

with tab_batch:
    st.subheader('Оценка списка вин')
    st.download_button(
        'Скачать пример CSV',
        data=features.head(5).to_csv(index=False).encode('utf-8-sig'),
        file_name='wine_example.csv',
        mime='text/csv',
    )
    uploaded = st.file_uploader('CSV с характеристиками вин', type=['csv'])
    if uploaded is not None:
        try:
            uploaded_data = pd.read_csv(uploaded, sep=None, engine='python')
            X_batch = prepare_batch(uploaded_data, feature_names)
            probabilities = batch_proba(model, X_batch)
            result = uploaded_data.copy()
            result['probability_good'] = probabilities
            result['prediction'] = (probabilities >= threshold).astype(int)
            result['decision'] = np.where(result['prediction'] == 1, 'Хорошее', 'Посредственное')
            st.dataframe(result, hide_index=True, width='stretch')
            st.download_button(
                'Скачать результаты',
                data=result.to_csv(index=False).encode('utf-8-sig'),
                file_name='wine_predictions.csv',
                mime='text/csv',
            )
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as error:
            st.error(f'Не удалось обработать CSV. {error}')

with tab_weights:
    st.subheader('Какие признаки влияют на оценку')
    coefficients = pd.Series(model[-1].coef_[0], index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['tab:green' if value >= 0 else 'tab:red' for value in coefficients]
    ax.barh([FEATURE_LABELS.get(name, name) for name in coefficients.index],
            coefficients.values, color=colors)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент')
    ax.set_title('Влияние признаков на прогноз хорошего вина')
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    st.write('**За хорошее вино:** ' +
             ', '.join(FEATURE_LABELS.get(name, name) for name in positive.index) + '.')
    st.write('**Против:** ' +
             ', '.join(FEATURE_LABELS.get(name, name) for name in negative.index) + '.')
