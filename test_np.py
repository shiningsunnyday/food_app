import pandas as pd
import numpy as np

df_nutrition = pd.read_csv('./new_laplacian_matrix.csv').iloc[:-1, 1:-1]
print(df_nutrition)
df_nutrition.to_csv('./new_laplacian_matrix.csv')
