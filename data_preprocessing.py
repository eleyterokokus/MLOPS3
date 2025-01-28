import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess(file_path: str = "data/cars_raw.csv"):

    # Загрузка данных
    df = pd.read_csv(file_path)

    # Очистка данных: удаление пропущенных значений
    df.dropna(inplace=True)

    # Разделяем данные на признаки и целевую переменную
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Инициализируем списки для числовых и категориальных признаков
    numeric_features = []
    categorical_features = []

    # Проходим по всем столбцам DataFrame и определяем их тип
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_features.append(column)
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            categorical_features.append(column)

            # Создаем пайплайн для предобработки данных
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Заполнение пропусков средним значением
        ('scaler', StandardScaler())                   # Стандартизация числовых признаков
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Заполнение пропусков наиболее частым значением
        ('onehot', OneHotEncoder(handle_unknown='ignore'))      # Кодирование категориальных признаков
    ])

    # Объединяем трансформеры в один ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    # Применяем предобработку к данным
    new_df = preprocessor.fit_transform(df)

    # Сохранение данных
    new_df.to_csv("./data/prep_data.csv", index=False)

load_and_preprocess()