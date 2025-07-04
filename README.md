# Heart Attack Prediction with FastAPI Interface

## Краткое описание
Проект реализует модель машинного обучения для предсказания риска инфаркта на основе медицинских данных.
В проекте предусмотрен удобный веб-интерфейс на FastAPI для загрузки данных и получения предсказаний.

## Основные возможности
 * Обработка и препроцессинг медицинских данных
 * Обучение и использование модели с порогом классификации
 * Веб-сервис на FastAPI для загрузки CSV-файлов и получения предсказаний
 * Экспорт результатов в формате JSON

## Структура проекта
```
Masterskaya_heart_atack
├── saved_models/ # папка с обученными объектами из analysis_and_machine_learning.ipynb
│   ├── data_preprocessor.pkl   # обученный препроцессор
│   └── model_with_threshold.pkl    # обученная модель с настроенным порогом классификации
├── .gitignore
├── README.md   # описание проекта
├── analysis_and_machine_learning.ipynb # исследование проекта и обучение моделей
├── classes.py  # файл с классами для сервиса на FastAPI
├── main.py # файл сервиса на FastAPI
├── requirements.txt # файл с настройками окружения
└── test_data_predict.csv   # файл с прогнозом модели
```

## Установка и запуск сервиса FastAPI
1) uvicorn main:app --reload
2) http://localhost:8000/docs
3) Использование сервиса:
   - загрузите CSV-файл с тестовыми данными через эндпоинт /predict/
   - получите предсказания в формате JSON

## Контакты
 * Автор: Мироненко Денис
 * Email: denmironenko@gmail.com