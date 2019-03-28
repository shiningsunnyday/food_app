import pandas as pd
import numpy as np

laplacian = pd.read_csv('./new_laplacian_matrix.csv').iloc[:,1:].copy()
ing_dic = dict(pd.read_csv('./new_ingredients_test.csv').copy().loc[:,"Ingredients"])

print(laplacian)
for col in range(37,len(laplacian)):

    for row in range(col):

        ing_1 = ing_dic[col]
        ing_2 = ing_dic[row]

        print(ing_1, ing_2)
        rating = int(input())
        laplacian.iloc[row, col] = rating
        laplacian.to_csv('./new_laplacian_matrix.csv')
