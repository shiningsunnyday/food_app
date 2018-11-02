from flask import jsonify
import pandas as pd
import requests
import json
import numpy as np

def add_new(x, boo):

    print(boo)
    pd.options.mode.chained_assignment = None
    attributes = ['Ingredients', 'calories', 'fat', 'protein', 'carbs', 'serving_qty', 'serving_unit', 'serving_weight_grams']

    url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    headers = {
        
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-app-id': 'e0d2583c',
            'x-app-key': '992176a868ced191c469ea0df71c7a54',
            'x-remote-user-id': '0',
        }

    data = {"query": x}
    r = requests.post(url, headers = headers, data = json.dumps(data))
    body = json.loads(r.text)

    try: 
        food = body['foods'][0]
    except KeyError:
        return jsonify("OH NO")

    dic = {'label': str(x),
           'amount': str(food["serving_qty"]) + ' ' + str(food["serving_unit"]),
           'calories': int(food["nf_calories"]),
           'protein': int(food["nf_protein"]),
           'fat': int(food["nf_total_fat"]),
           'carbs': int(food["nf_total_carbohydrate"]),
           }

    if boo:

        df_nutrition = pd.read_csv('./new_ingredients_test.csv').loc[:, "Ingredients":]
        values = [str(x), int(food["nf_calories"]), int(food["nf_total_fat"]), int(food["nf_protein"]), int(food["nf_total_carbohydrate"]), float(food["serving_qty"]), str(food["serving_unit"]), int(food["serving_weight_grams"])]
        dic = dict(zip(attributes, values))
        df_nutrition = df_nutrition.append(pd.DataFrame(dic, index=[len(df_nutrition)]))
        df_nutrition.to_csv('./new_ingredients_test.csv')
        aff = pd.read_csv('./new_laplacian_matrix.csv').iloc[:, 1:].values
        print(aff)
        aff = [np.append(x, 1) for x in aff]
        print(aff)
        aff.append([0] * len(aff[0]))

        pd.DataFrame(aff).to_csv('./new_laplacian_matrix.csv')

        
    return jsonify(dic)



