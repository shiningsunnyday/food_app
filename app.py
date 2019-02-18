from flask import Flask
from flask import jsonify
from flask import request
import pandas as pd
import random
import numpy as np
import scipy.sparse.linalg as linalg
from sklearn.cluster import KMeans

app = Flask(__name__)

df = pd.read_csv('new_ingredients_test.csv')
dfs = df.loc[:, 'Ingredients':].dropna()
dfs_name = dfs.set_index("Ingredients", drop = False)
values = {x[0]: x[1:] for x in dfs[['Ingredients', 'calories', 'protein', 'fat', 'carbs']].values}
values_copy = {x[0]: x[1:] for x in dfs[['Ingredients', 'calories', 'protein', 'fat', 'carbs']].values}
dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
laplacian_matrix = pd.read_csv('new_laplacian_matrix.csv').iloc[:,1:]
test_dic = dict(dfs.loc[:]['Ingredients'])
name_dic = {test_dic[x]: x for x in test_dic.keys()}
train = pd.read_json('meal_regression/train.json')
ings = pd.read_csv('ingredients_standardized.csv')
nutrients = ings.loc[:, 'Ingredients':].values
nutrients_dic = {nutrient[0]: nutrient[1:] for nutrient in nutrients}
recipes_nutrients = []

for recipe in train.ingredients:
    try:
        recipe_nutrients = [nutrients_dic[ing] for ing in recipe]
        recipe_nutrients.sort(key = lambda r: r[0])
        recipes_nutrients.append(np.array(recipe_nutrients))
    except KeyError:
        pass

recipes_nutrients = np.array(recipes_nutrients)

shape_div = {}
for i in range(len(recipes_nutrients)):
    try:
        shape_div[recipes_nutrients[i].shape].append(recipes_nutrients[i])
    except KeyError:
        if(recipes_nutrients[i].shape not in shape_div):
            shape_div[recipes_nutrients[i].shape] = [recipes_nutrients[i]]
shape_div = {key: np.array(shape_div[key]) for key in shape_div.keys()}

from sklearn.neighbors.kde import KernelDensity
import numpy as np

bandwidths = {1: 132.19411484660287,
 2: 533.6699231206302,
 3: 335.1602650938834,
 4: 210.49041445120218,
 5: 210.49041445120218,
 6: 335.1602650938834,
 7: 210.49041445120218,
 8: 210.49041445120218,
 9: 335.1602650938834,
 10: 210.49041445120218,
 11: 335.1602650938834,
 12: 210.49041445120218,
 13: 335.1602650938834,
 14: 335.1602650938834,
 15: 533.6699231206302,
 16: 533.6699231206302,
 17: 533.6699231206302,
 18: 533.6699231206302,
 19: 533.6699231206302,
 20: 849.7534359086438,
 21: 335.1602650938834,
 22: 1353.0477745798075,
 23: 1353.0477745798075,
 24: 1353.0477745798075}

kdes = {}

for i in range(1, 25):
    test_div = []
    for j in range(len(shape_div[(i, 12)])):
        to_append = shape_div[(i,12)][j]
        test_div.append(np.array(to_append))
    params = {'bandwidth': np.logspace(-10, 10, 100)}
    data = np.array(test_div).reshape(len(shape_div[(i, 12)]), 12 * i)
    kde = KernelDensity(kernel = 'gaussian', bandwidth = bandwidths[i])
    kde.fit(data)
    kdes[i] = kde

def get_score(recipes):
    kde = kdes[len(recipes[0])]
    data = recipes.reshape(len(recipes), -1)
    # print(data.shape)
    score = kde.score_samples(np.array(data))
    return score
    # print("recipe with score %s" % (score))

def sorted_scores(recommendations):
    return {score(rec)[0]: rec for rec in np.array(sorted(recommendations, key = lambda r: score(r)))}

def score(recommendation):
    nutrients = []
    for ingredient in recommendation:
        if(ingredient in nutrients_dic):
            nutrients.append(nutrients_dic[ingredient])
        else:
            continue
    nutrients = sorted(nutrients, key = lambda c: c[0])
    if(len(nutrients) > 0):
        return get_score(np.array([nutrients]))
    else:
        return 999

def laplacian(array, cluster):

    arr = [name_dic[x] for x in array]
    laplacian = laplacian_matrix.iloc[arr, arr].values

    laplacian = np.array([[float(x) for x in laplacian[i]] for i in range(len(laplacian))])

    return laplacian

@app.route('/returnall/', methods = ['GET'])
def returnall():

    alling = []
    for i in range(len(dfs_name)):
        ing = dfs_name.iloc[i]
        alling.append({"label": str(ing['Ingredients']), "amount": str(ing['serving_qty']) + ' ' + str(ing['serving_unit']),
                                    'calories': float(ing['calories']), 'protein': float(ing['protein']), 'fat': float(ing['fat']), 'carbs': float(ing['carbs'])})

    return jsonify(alling)

def generate(target_macros_processed):

    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}

    mcros = [0, 0, 0, 0]
    ingredients = []
    ingredients.append("")
    ingredients = []

    while True:

        current_length = len(dfs_name)
        rand = random.randint(0, current_length-1)
        ing = dfs_name.iloc[rand]

        if mcros[0] + ing['calories'] < target_macros_processed[0]:

            ingredients.append([ing['Ingredients'], str(ing['serving_qty']) + ' ' + str(ing['serving_unit']),
                                {'calories': ing['calories'], 'protein': ing['protein'], 'fat': ing['fat'], 'carbs': ing['carbs']}])
            mcros = [mcros[i] + ing[dic[i]] for i in range(len(mcros))]
        else:
            break

    for i in range(20):
        try:
            ingredients, mcros = iterate(ingredients, mcros, target_macros_processed)
        except KeyError or IndexError:
            # print("BAD ITERATE")
            pass


    return ingredients, mcros

def iterate(ingredients, mcros, target_mcros, preferences = 4):

    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
    minimal_error = sum([abs(mcros[i] - target_mcros[i]) for i in range(1, preferences)])

    # print("CURRENT ERROR IS ", minimal_error)
    ing_to_add = ""
    ing_to_remove = ""
    b = 0

    for ing in values.keys():

        effect = sum([abs(values[ing][i] + mcros[i] - target_mcros[i]) for i in range(1, preferences)])

        if effect < minimal_error:
            b = 1
            minimal_error = effect
            ing_to_add = ing
            ing_to_add = [ing_to_add, str(dfs_name.loc[ing_to_add]['serving_qty']) + ' ' + str(dfs_name.loc[ing_to_add]['serving_unit']), dict(zip(dic.values(), values[ing_to_add]))]


    for ing in ingredients:

        ing_name = ing[0]
        subtract_effect = sum([abs(-ing[2][dic[i]] + mcros[i] - target_mcros[i]) for i in range(1, preferences)])

        # print("IF we try removing", ing_name, "we can get an error of", subtract_effect)

        if subtract_effect < minimal_error:

            minimal_error = subtract_effect
            b = 2
            ing_to_remove = ing
            # print("CANDIDATE TO REMOVE", ing_to_remove)

    # print("LIKE THIS", ing_to_add)
    # print(b)
    if b == 1:

        ingredients.append(ing_to_add)
        # print("ADDED",ing_to_add)

    elif b == 2:

        # print(len(ingredients))
        ingredients.remove(ing_to_remove)
        # print(len(ingredients))
        # print("REMOVED",ing_to_remove)

    else:
        pass

    return ingredients, [mcros[i] + ing_to_add[2][dic[i]] if b == 1 else mcros[i] - ing_to_remove[2][dic[i]] if b == 2 else mcros[i] for i in range(len(dic))]

@app.route('/', methods = ['GET'])
def api_root():
    return jsonify(".")

@app.route('/macros/', methods = ['GET'])
def api_macros():

    return jsonify("Hi")

from add_new_ingredient import add_new

@app.route('/add/', methods = ['GET'])
def api_add():

    x = str(request.args.get('ingredient'))
    boo = bool(int(request.args.get('boo')))
    # print(boo)
    return add_new(x.replace('_', ' '), boo)

@app.route('/macros', methods = ['GET'])
def api_macros2():

    return jsonify("Baby")

@app.route('/rank/', methods = ['GET'])
def api_rank():
    candidates = str(request.args.get('candidates'))
    candidates = list(candidates.split('___'))
    candidates = list(map(lambda x: x.split('__'), candidates))
    candidates = list(map(lambda x: list(map(lambda y: y.replace('_', ' '), x)), candidates))
    result = sorted_scores([recipe for recipe in candidates if len(recipe) < 25 and len(recipe) > 0])
    # print(result)
    return jsonify(result)

@app.route('/macros/<target_macros>', methods = ['GET'])
def api_macros_(target_macros):

    target_macros_processed = list(map(int, target_macros.split('_')))
    avg_percent_off = 100
    closeness_measure = 0

    all_generated = []

    for i in range(20):

        #fetches ingredients and mcros with target_macros
        ingredients, mcros = generate(target_macros_processed)
        all_generated.append([ing[0] for ing in ingredients])
        if i == 0:
            final_mcros = mcros
            final_ingredients = ingredients
        #gets average error
        x = sum([
            100 * abs(mcros[i] - target_macros_processed[i])/(target_macros_processed[i]) for i in range(len(mcros))]) / 4.0
        food_arr = laplacian([ingredient[0] for ingredient in ingredients], False)
        y = sum(sum(x) for x in food_arr)/len(food_arr)

        if x < avg_percent_off:

            avg_percent_off = x
            final_mcros = mcros
            final_ingredients = ingredients

    print(sorted_scores(all_generated))

    return jsonify({"requirements": [int(x) for x in final_mcros],
                    "listToDisplay": [
                    {
                        'label': str(ing[0]),
                        'amount': str(ing[1]),
                        'calories': int(ing[2]['calories']),
                        'protein': int(ing[2]['protein']),
                        'fat': int(ing[2]['fat']),
                        'carbs': int(ing[2]['carbs'])
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

    X = laplacian(array, True)
    # print(X)
    eigen_vals, eigen_vects = linalg.eigs(X, num_meals)
    X = eigen_vects.real
    rows_norm = np.linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    labels = k_means(Y, num_meals)
    dic = dict(zip(array, labels))
    unique, counts = np.unique(labels, return_counts=True)
    return [[x for x in dic.keys() if dic[x] == j] for j in range(num_meals)]

@app.route('/returninfo/<returninfo>', methods = ['GET'])
def returninfo(returninfo):

    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
    infoList = list(map(str, returninfo.split(",")))
    for x in range(len(infoList)):

        infoList[x] = infoList[x].replace("_", " ")


    return jsonify({ing_to_add:

                   {'label': str(ing_to_add),
                'amount': str(dfs_name.loc[ing_to_add]['serving_qty']) + ' ' + str(dfs_name.loc[ing_to_add]['serving_unit']),
                'calories': dict(zip(dic.values(), values_copy[ing_to_add]))['calories'],
                'protein': dict(zip(dic.values(), values_copy[ing_to_add]))['protein'],
                'fat': dict(zip(dic.values(), values_copy[ing_to_add]))['fat'],
                'carbs': dict(zip(dic.values(), values_copy[ing_to_add]))['carbs']} for ing_to_add in infoList
                   })

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
