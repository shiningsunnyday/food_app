import pandas as pd
import numpy as np

df_nutrition = pd.read_csv('./new_ingredients_test.csv').loc[:,'Ingredient':]
print(df_nutrition)
