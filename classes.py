import pandas as pd

# класс трансформера датасета
class DataTransformer:
    
    # порядок столбцов в преобразованном DataFrame
    columns_order = [
        'age', 'cholesterol', 'heart_rate', 'diabetes', 'family_history',
        'smoking', 'obesity', 'alcohol_consumption', 'stress_level',
        'triglycerides', 'sleep_hours_per_day', 'gender',
        'systolic_blood_pressure', 'diastolic_blood_pressure'
    ]

    # метод преобразования имен столбцов в snake_type
    def to_snake_type(self, columns):
        return ['_'.join(w.lower().replace('-', '').split()) for w in columns]

    # метод преобразования данных
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # удаляем 'Unnamed: 0
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        # Переносим 'id' в индекс
        if 'id' in df.columns:
            df = df.set_index('id', drop=True)
        
        # Переименовываем столбцы в snake_case
        df.columns = self.to_snake_type(df.columns)

        # заменяем бинарный категориальный признак 'gender' на 1/0
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, '1.0': 1, '0.0': 0})

        # удаляем пропущенные значения
        df = df.dropna()

        # оставляем только нужные столбцы, если они есть в DataFrame
        cols_to_keep = [col for col in self.columns_order if col in df.columns]
        df = df.loc[:, cols_to_keep]

        # удаляем явные дубликаты
        df = df.drop_duplicates()

        return df
    

# класс препроцессора датасета
class PreprocessorWrapper:
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
            
    def transform(self, X):
        transformed = self.preprocessor.transform(X)
        return transformed
    

# класс модели с порогом классификации
class ModelWithThreshold:
    
    def __init__(self, model, threshold=0.2):
        self.model = model
        self.threshold = threshold
        
    def predict(self, X):
        prob = self.model.predict_proba(X)[:, 1]
        return pd.Series(prob).apply(lambda x: 1 if x > self.threshold else 0)


# класс для экспорта результатов
class PredictDataFrame:
    def __init__(self):
        pass
    
    def transform(self, df: pd.DataFrame, pred: pd.Series):
        df = df.reset_index().rename(columns = {'index': 'id'})
        df['prediction'] = pred
        df = df[['id', 'prediction']]
        return df