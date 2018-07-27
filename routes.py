from app import app
from flask import jsonify
import pandas as pd

df = pd.read_csv('/aifood_dfs_clean.csv')
dfs = df.loc[:, 'Ingredients':].dropna()
values = {x[0]: [x[1:], 0] for x in dfs[['Ingredients', 'calories', 'protein', 'fat', 'carbs']].values}

import random
import json
import numpy as np

@app.route('/', methods = ['GET'])
def api_root():
    return jsonify(".")

@app.route('/macros/', methods = ['GET'])
def api_macros():

    return jsonify("Hi")

@app.route('/macros', methods = ['GET'])
def api_macros2():

    return jsonify("Baby")

@app.route('/macros/<target_macros>', methods = ['GET'])
def api_macros_(target_macros):
    
    target_macros_processed = list(map(int, target_macros.split('_')))
    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3: 'carbs'}

    mcros = [0, 0, 0, 0]
    ingredients = []
    
    while True:
        
        rand = random.randint(0, len(dfs))
        ing = dfs.iloc[rand]

        if mcros[0] + ing[dic[0]] > target_macros_processed[0] * 1.1:

            pass

        else:
            
            ingredients.append([ing['Ingredients'], str(ing['serving_qty']) + ' ' + str(ing['serving_unit']),
                                {'calories': ing['calories'], 'protein': ing['protein'], 'fat': ing['fat'], 'carbs': ing['carbs']}])
                
            mcros = [mcros[i] + ing[dic[i]] for i in range(len(mcros))]

            if target_macros_processed[0] * 0.9 <= mcros[0]:

                break
   
    
    return jsonify({"requirements": [int(x) for x in mcros],
                    "listToDisplay": [
                    {
                        'label': str(ing[0]),
                        'amount': str(ing[1]),
                        'calories': ing[2]['calories'],
                        'protein': ing[2]['protein'],
                        'fat': ing[2]['fat'],
                        'carbs': ing[2]['carbs']
                    }
                        for ing in ingredients
                        ]
                    })


@app.route('/diff/<diff>', methods = ['GET'])
def api_fix(diff):

    diff_to_fix = list(map(int, diff.split('_')))
    minimal_error = sum([abs(i) for i in diff_to_fix])
    net_effect = minimal_error
    ing_to_add = ""
    
    for ing in values.keys():

        effect = sum([abs(values[ing][0][i] + diff_to_fix[i]) for i in range(1, preferences)])
        values[ing][1] = effect

        if values[ing][1] < net_effect:

            net_effect = values[ing][1]
            ing_to_add = ing
            
    ing_to_add = [ing_to_add, str(ing_to_add['serving_qty']) + ' ' + str(ing_to_add['serving_unit']),
                  ing_to_dict(zip(dic.values(), values[ing_to_add][0]))]
    return jsonify({
                        'label': str(ing_to_add[0]),
                        'amount': str(ing_to_add[1]),
                        'calories': ing_to_add[2]['calories'],
                        'protein': ing_to_add[2]['protein'],
                        'fat': ing_to_add[2]['fat'],
                        'carbs': ing_to_add[2]['carbs']
                  })
















