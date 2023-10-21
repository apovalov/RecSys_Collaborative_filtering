import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, linalg
import pickle
from useritemmatrix import UserItemMatrix
from tf_idf_bn25_normalization import Normalization


# Чтение данных
sales = pd.read_csv('SKU_2022.csv')
sales_df = pd.DataFrame(sales)

# Создание User-Item матрицы
itemMatrix = UserItemMatrix(sales_df)
ui_matrix = itemMatrix.csr_matrix

# Нормализация матрицы
normalized_matrix = Normalization.by_row(ui_matrix)

def items_embeddings(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
    """Build items embedding using factorization model."""
    U, s, Vt = linalg.svds(ui_matrix, k=dim)  # Используем SVD из scipy.sparse.linalg
    return Vt.T

# Получаем матрицу эмбеддингов товаров
items_vec = items_embeddings(normalized_matrix, dim=50)

# Сохраняем матрицу эмбеддингов
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(items_vec, f)
