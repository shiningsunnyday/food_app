from flask import jsonify
import pandas as pd
import random
import requests
import json
import numpy as np
import scipy.sparse.linalg as linalg
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors.kde import KernelDensity
from food_app.to_html import to_table, to_lists, to_list_of_lists

df = pd.read_csv('food_app/new_ingredients_test.csv')
dfs = df.loc[:, 'Ingredients':].dropna()
dfs_name = dfs.set_index("Ingredients", drop = False)
values = {x[0]: x[1:] for x in dfs[['Ingredients', 'calories', 'protein', 'fat', 'carbs']].values}
values_copy = {x[0]: x[1:] for x in dfs[['Ingredients', 'calories', 'protein', 'fat', 'carbs']].values}
dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
laplacian_matrix = pd.read_csv('food_app/new_laplacian_matrix.csv').iloc[:,1:]
test_dic = dict(dfs.loc[:]['Ingredients'])
name_dic = {test_dic[x]: x for x in test_dic.keys()}
train = pd.read_json('meal_regression/train.json')
ings = pd.read_csv('food_app/ingredients_standardized.csv').loc[:,'Ingredients':].dropna()
nutrients = ings.loc[:, 'Ingredients':].values
nutrients_dic = {nutrient[0]: nutrient[1:] for nutrient in nutrients}
ordered_ings = list(nutrients_dic.keys())
ordered_ings_dic = {ordered_ings[i]: i for i in range(len(ordered_ings))}
recipes_nutrients = []

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

def add_new(x):
    global ings
    url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    headers = {

            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-app-id': '2ea6735c',
            'x-app-key': 'a387450901cb18ea23f22c01e25ad53d',
            'x-remote-user-id': '0',
        }

    data = {"query": x}
    r = requests.post(url, headers = headers, data = json.dumps(data))
    body = json.loads(r.text)
    try:
        food = body['foods'][0]
    except KeyError:
        return jsonify("OH NO")

    ratio = float(float(food["serving_weight_grams"]) / 100.0)
    desired_attrs = ['food_name', 'nf_calories', 'nf_total_fat', 'nf_protein', 'nf_total_carbohydrate', 'nf_saturated_fat', 'nf_cholesterol', 'nf_sodium', 'nf_dietary_fiber', 'nf_sugars', 'nf_potassium', 'nf_p', 'serving_weight_grams']
    lis = [food[attr] if food[attr] != None else 0 for attr in desired_attrs]
    stand_lis = [lis[0]] + list(map(lambda x: round((float(x) / (float(lis[-1]) / 100)), 2), lis[1:]))
    dic = {ings.columns[i]: stand_lis[i] for i in range(len(ings.columns))}
    df = pd.DataFrame(np.array(stand_lis).reshape(1, -1), columns = ings.columns)
    ings = ings.append(df, ignore_index = True)
    return ings.values[-1][1:]

def macros(target_macros):
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

    requirements = [int(x) for x in final_mcros]
    display_list = [
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
    return requirements, to_lists([display_list])

def iterate(ingredients, mcros, target_mcros, preferences = 4):

    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
    minimal_error = sum([abs(mcros[i] - target_mcros[i]) for i in range(1, preferences)])
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

def k_means(X, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    return kmeans.fit(X).labels_

def laplacian(array, cluster):

    arr = [name_dic[x] for x in array]
    laplacian = laplacian_matrix.iloc[arr, arr].values
    laplacian = np.array([[float(x) for x in laplacian[i]] for i in range(len(laplacian))])
    return laplacian

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

def cluster(string, num_meals):
    listToNames = [ing.strip(' ') for ing in string.strip('\"').split(',')]
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
    list_of_ingredient_lists = {"listOfIngredientLists": [{
        "listToDisplay": ingredients[i],
        "requirements": [int(y) for y in requirements[i]]
        }
                    for i in range(len(x))]}
    return requirements, to_list_of_lists(list_of_ingredient_lists)

def return_all():
    alling = []
    for i in range(len(dfs_name)):
        ing = dfs_name.iloc[i]
        alling.append(
            {"label": str(ing['Ingredients']),
             "amount": str(ing['serving_qty']) + ' ' + str(ing['serving_unit']),
             'calories': float(ing['calories']),
             'protein': float(ing['protein']),
             'fat': float(ing['fat']),
             'carbs': float(ing['carbs'])})
    return to_table(alling)

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

# returns ing:cluster_number dic
def recipe_to_labels(recipe):
    global ings
    agg_clustering = AgglomerativeClustering(n_clusters = 100).fit(ings.loc[:, 'calories':'serving_weight_grams'].values)
    return {ing: agg_clustering.labels_[ordered_ings_dic[ing]] for ing in recipe}

# replaces ings of key label with random alternative ings
def substitute_label(key, ings, labels_dic):
    subs = []
    key_array = labels_dic[key]
    for ing in ings:
        key_array.remove(ing)
    for i in range(len(ings)):
        rand_index = random.randint(0, len(ings) - 1)
        subs.append(key_array[rand_index])
        del key_array[rand_index]
    return subs

def substitute_recipe(recipe):
    recipe_ing_labels = recipe_to_labels(recipe)
    recipe_labels = recipe_to_labels(ordered_ings_dic)
    # matches to label:array_ingredients dic
    labels_dic = {key: [] for key in set(recipe_labels.values())}
    for key in recipe_labels.keys():
        labels_dic[recipe_labels[key]].append(key)
    label_freqs = {key: [ing for ing in recipe_ing_labels.keys() if recipe_ing_labels[ing] == key] for key in set(recipe_ing_labels.values())}
    new_substitute = []
    sorted_keys = sorted(label_freqs.keys(), key = lambda x: len(label_freqs[x]))
    for key in sorted_keys:
        new_substitute += substitute_label(key, label_freqs[key], labels_dic)
    return new_substitute

def check_new_ings(ings_lis):
    global ordered_ings
    global ordered_ings_dic
    global ings
    for ing in ings_lis:
        if ing not in ordered_ings_dic.keys():
            nutrients_dic[ing] = add_new(ing)
            ordered_ings = list(nutrients_dic.keys())
            ordered_ings_dic[ing] = len(ordered_ings) - 1

def substitute(ingrs):
    ings_lis = ingrs.split(', ')
    check_new_ings(ings_lis)
    return substitute_recipe(ings_lis)

def fix_diff(diff):
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

def get_info(info):
    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3:'carbs'}
    infoList = list(map(str, returninfo.split(",")))
    for x in range(len(infoList)):
        infoList[x] = infoList[x].replace("_", " ")
    return jsonify({ing_to_add: {
        'label': str(ing_to_add),
        'amount': str(dfs_name.loc[ing_to_add]['serving_qty']) + ' ' + str(dfs_name.loc[ing_to_add]['serving_unit']),
        'calories': dict(zip(dic.values(), values_copy[ing_to_add]))['calories'],
        'protein': dict(zip(dic.values(), values_copy[ing_to_add]))['protein'],
        'fat': dict(zip(dic.values(), values_copy[ing_to_add]))['fat'],
        'carbs': dict(zip(dic.values(), values_copy[ing_to_add]))['carbs']}
        for ing_to_add in infoList })
