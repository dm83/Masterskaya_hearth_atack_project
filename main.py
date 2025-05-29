from io import StringIO
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
import joblib
import pandas as pd

# экземпляр класса FastAPI [terminal input: "uvicorn main:app --reload"]
app = FastAPI()

# экспорт классов для создания экземпляров
from classes import DataTransformer, PreprocessorWrapper, ModelWithThreshold, PredictDataFrame

# создание экземпляров классов для трансформации, обработки расчетов прогноза и построения таблицы
save_dir = "saved_models"
data_preprocessor = joblib.load(os.path.join(save_dir, "data_preprocessor.pkl"))
model_with_threshold = joblib.load(os.path.join(save_dir, "model_with_threshold.pkl"))
data_transformer = DataTransformer()
final_data_prep = PredictDataFrame()

# выбор файла и построение прогноза с выводом значений id и prediction
@app.post(
        "/predict/",
        description = 'Необходимо загрузить датасет в формате csv',
        tags=['Прогноз риска инфаркта'])
async def predict(file: UploadFile = File(..., description="CSV файл с тестовыми данными")):
    
    # проверка расширения файла (только csv)
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Требуется CSV файл")
    
    # пытаемся прочитать файл
    try:
        # если это файл 'csv' читаем содержимое файла и сохраняем в df
        contents = await file.read()
        s = contents.decode('utf-8')
        df = pd.read_csv(StringIO(s))

    except Exception as e:
        # если не csv сообщение с ошибкой
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении файла: {e}")

    # пытаемся обработать файл с построить прогноз
    try:
        # применяем трансформер, препроцессор, модель и готовим финальную таблицу
        X_test = data_transformer.transform(df)
        X_test_transformed = data_preprocessor.transform(X_test)
        predictions = model_with_threshold.predict(X_test_transformed)
        final_df = final_data_prep.transform(df=X_test, pred=predictions)
        
    except Exception as e:
        # если что-то не так -> сообщение с ошибкой
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {e}")

    # формируем ответ
    result = final_df.to_dict(orient='records')
    return {"predictions": result}