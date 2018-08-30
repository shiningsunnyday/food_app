from flask import Flask
from flask import jsonify
from flask import request
import pandas as pd
import random
import numpy as np
import scipy.sparse.linalg as linalg
from sklearn.cluster import KMeans

app = Flask(__name__)

df = pd.read_csv('aifood_dfs_clean.csv')
dfs = df.loc[:, 'Ingredients':].dropna()
dfs_name = dfs.set_index("Ingredients", drop = False)
values = {x[0]: x[1:] for x in dfs[['Ingredients', 'calories', 'protein', 'fat', 'carbs']].values}
values_copy = {x[0]: x[1:] for x in dfs[['Ingredients', 'calories', 'protein', 'fat', 'carbs']].values}
dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
laplacian_matrix = pd.read_csv('laplacian_matrix.csv').iloc[:,1:]

test_dic = dict(dfs.loc[:]['Ingredients'])
name_dic = {test_dic[x]: x for x in test_dic.keys()}

def laplacian(array):
    
    arr = [name_dic[x] for x in array]
    laplacian = laplacian_matrix.iloc[arr, arr].values
    return laplacian

def generate(target_macros_processed):

    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}

    mcros = [0, 0, 0, 0]
    ingredients = []
    ingredients.append("")
    ingredients = []
    
    while True:

        rand = random.randint(0, len(dfs_name))
        ing = dfs_name.iloc[rand]

        if ing['calories']/(mcros[0] + 1) < 0.1 :

            pass

        else:

            if mcros[0] + ing['calories'] < target_macros_processed[0] * 1.1:
            
                ingredients.append([ing['Ingredients'], str(ing['serving_qty']) + ' ' + str(ing['serving_unit']),
                                    {'calories': ing['calories'], 'protein': ing['protein'], 'fat': ing['fat'], 'carbs': ing['carbs']}])
                    
                mcros = [mcros[i] + ing[dic[i]] for i in range(len(mcros))]

            else:

                break

    for i in range(5):

        try:
            ingredients, mcros = iterate(ingredients, mcros, target_macros_processed)
        except KeyError or IndexError:
            pass

    return ingredients, mcros

def iterate(ingredients, mcros, target_mcros, preferences = 4):

    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
    minimal_error = sum([abs(mcros[i] - target_mcros[i]) for i in range(1, preferences)])
    ing_to_add = ""
    boo = True
    
    for ing in values.keys():

        effect = sum([abs(values[ing][i] + mcros[i] - target_mcros[i]) for i in range(1, preferences)])

        if effect < minimal_error:

            minimal_error = effect
            ing_to_add = ing

    for ing in ingredients:

        ing_name = ing[0]
        subtract_effect = sum([abs(-ing[2][dic[i]] + mcros[i] - target_mcros[i]) for i in range(1, preferences)])

        if subtract_effect < minimal_error:

            minimal_error = subtract_effect
            boo = False
            ing_to_add = ing_name

    ing_to_add = [ing_to_add, str(dfs_name.loc[ing_to_add]['serving_qty']) + ' ' + str(dfs_name.loc[ing_to_add]['serving_unit']), dict(zip(dic.values(), values[ing_to_add]))]           
            
    if boo:
        values.pop(ing_to_add[0])
        ingredients.append(ing_to_add)
    else:
        
        ingredients.remove(ing_to_add)
    
    return ingredients, [mcros[i] + ing_to_add[2][dic[i]] if boo else mcros[i] - ing_to_add[2][dic[i]] for i in range(len(dic))]

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
    avg_percent_off = 100
    closeness_measure = 0

    for i in range(10):
        
        ingredients, mcros = generate(target_macros_processed)

        x = sum([
            100 * abs(mcros[i] - target_macros_processed[i])/(target_macros_processed[i]) for i in range(len(mcros))]) / 4.0
        food_arr = laplacian([ingredient[0] for ingredient in ingredients])
        y = sum(sum(x) for x in food_arr)/len(food_arr)
        
        if x < avg_percent_off and y > closeness_measure:

            avg_percent_off = x
            closeness_measure = y
            final_mcros = mcros
            final_ingredients = ingredients


    return jsonify({"requirements": [int(x) for x in final_mcros],
                    "listToDisplay": [
                    {
                        'label': str(ing[0]),
                        'amount': str(ing[1]),
                        'calories': ing[2]['calories'],
                        'protein': ing[2]['protein'],
                        'fat': ing[2]['fat'],
                        'carbs': ing[2]['carbs']
                    }
                        for ing in final_ingredients
                        ]
                    })
    


@app.route('/diff/<diff>', methods = ['GET'])
def api_fix(diff):

    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
    diff_to_fix = list(map(int, diff.split('_')))
    minimal_error = sum([abs(i) for i in diff_to_fix])
    net_effect = minimal_error
    ing_to_add = ""
    
    for ing in values.keys():

        effect = sum([abs(values[ing][i] + diff_to_fix[i]) for i in range(4)])

        if effect < net_effect:

            net_effect = effect
            ing_to_add = ing
            
    ing_to_add = [ing_to_add, str(dfs_name.loc[ing_to_add]['serving_qty']) + ' ' + str(dfs_name.loc[ing_to_add]['serving_unit']),
                  dict(zip(dic.values(), values[ing_to_add]))]
    return jsonify({
                        'label': str(ing_to_add[0]),
                        'amount': str(ing_to_add[1]),
                        'calories': ing_to_add[2]['calories'],
                        'protein': ing_to_add[2]['protein'],
                        'fat': ing_to_add[2]['fat'],
                        'carbs': ing_to_add[2]['carbs']
                  })

def k_means(X, n_clusters):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    return kmeans.fit(X).labels_


def spectral_cluster(array, num_meals):
    
    X = laplacian(array)
    eigen_vals, eigen_vects = linalg.eigs(X, num_meals)
    X = eigen_vects.real
    rows_norm = np.linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    labels = k_means(Y, num_meals)
    dic = dict(zip(array, labels))
    unique, counts = np.unique(labels, return_counts=True)
    return [[x for x in dic.keys() if dic[x] == j] for j in range(num_meals)]


@app.route('/cluster/', methods = ['GET'])
def api_cluster():

    string = str(request.args.get('ingredients'))
    string = string.replace('_', ' ')
    num_meals = int(request.args.get('num_meals'))

    listToNames = string.split(',')
    x = spectral_cluster(listToNames, num_meals)

    
    ingredients = [[] for i in range(num_meals)]
    requirements = [[0, 0, 0, 0] for i in range(num_meals)]
    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}

    for i in range(len(x)):
        for j in x[i]:
            ing_to_add = [j, str(dfs_name.loc[j]['serving_qty']) + ' ' + str(dfs_name.loc[j]['serving_unit']),
                  dict(zip(dic.values(), values_copy[j]))]
            
            ingredients[i].append({
                        'label': str(ing_to_add[0]),
                        'amount': str(ing_to_add[1]),
                        'calories': ing_to_add[2]['calories'],
                        'protein': ing_to_add[2]['protein'],
                        'fat': ing_to_add[2]['fat'],
                        'carbs': ing_to_add[2]['carbs']
                      })
            requirements[i][0] += ing_to_add[2]['calories']
            requirements[i][1] += ing_to_add[2]['protein']
            requirements[i][2] += ing_to_add[2]['fat']
            requirements[i][3] += ing_to_add[2]['carbs']
    
    return jsonify({"listOfIngredientLists": [{

        "listToDisplay": ingredients[i],
        "requirements": [int(y) for y in requirements[i]]
        
        }
                    for i in range(len(x))]})

if __name__ == "__main__":
    app.run(host = "0.0.0.0")


