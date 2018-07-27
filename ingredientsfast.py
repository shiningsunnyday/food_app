import pandas as pd
import random
import json
import numpy as np

def main(target_mcros):

    global dfs
    global dic
    global values
    global new_count
    global train
    
    df = pd.read_csv('/Users/shiningsunnyday/Desktop/Food/ingredients.csv').loc[1:10]
    print(df)
    train = pd.read_json('/Users/shiningsunnyday/Desktop/Food/train.json')
    ingredients = []; count = {}

    for recipe in train.values:
                
        for ingredient in recipe[2]:
            
            if ingredient not in ingredients:
                ingredients.append(ingredient)
                count[ingredient] = 0
            else:
                count[ingredient] += 1
                
    dfs = df.loc[:, 'Ingredients':].dropna()
                
    new_count = {x: count[x] for x in count.keys() if count[x] > 10}

    dfs = dfs[dfs.Ingredients.isin(new_count.keys())]
    dfs = dfs.reset_index().loc[:, 'Ingredients':]

    df_dic = {'protein': dfs.sort_values(by = ['protein']),
              'fat': dfs.sort_values(by = ['fat']),
              'carbs': dfs.sort_values(by = ['carbs'])}

    values = {x[0]: [x[1:], 0] for x in dfs[['Ingredients', 'calories', 'protein', 'fat', 'carbs']].values}
    dic = {0: 'calories', 1: 'protein', 2: 'fat', 3: 'carbs'}


    mcros, initial_list = generate([0 for x in range(len(target_mcros))], target_mcros, [])
    initial_list, mcros, error = iterate(initial_list, mcros, target_mcros)
    initial_list, mcros, error = iterate(initial_list, mcros, target_mcros)
    initial_list, mcros, error = iterate(initial_list, mcros, target_mcros)
    initial_list, mcros, error = iterate(initial_list, mcros, target_mcros)
    initial_list, mcros, error = iterate(initial_list, mcros, target_mcros)

    
    display(mcros, initial_list, sum([abs(mcros[i] - target_mcros[i]) for i in range(1, len(dic))]), target_mcros)
    return initial_list, mcros

def display(mcros, list_to_display, error, target_mcros):

    for i in range(len(list_to_display)):

        row = dfs.loc[dfs['Ingredients'] == list_to_display[i][0]]
        print("%d. " % (i+1) + row['serving_qty'].to_string(index = False) + ' ' + row['serving_unit'].to_string(index = False) + ' of ' + row['Ingredients'].to_string(index = False))


    print(" ".join(["%s: %d %s" % (dic[i], mcros[i], '(%d)' % (mcros[i] - target_mcros[i]) if target_mcros[i] >= mcros[i] - 1 else '(+%d)' % (mcros[i] - target_mcros[i])) for i in range(1, len(dic))]))
    print('\n')
    



def generate(mcros, target_mcros, ingredients):

    while True:
        
        rand = random.randint(0, len(dfs))
        ing = dfs.iloc[rand]

        if mcros[0] + ing[dic[0]] > target_mcros[0] * 1.1:

            pass

        else:
            
            ingredients.append([ing['Ingredients'], {'calories': ing['calories'], 'protein': ing['protein'], 'fat': ing['fat'], 'carbs': ing['carbs']}])
                
            mcros = [mcros[i] + ing[dic[i]] for i in range(len(mcros))]

            if target_mcros[0] * 0.9 <= mcros[0]:
                #print(ingredients)
                #print((protein_, fat_, carbs_))
                break

    return mcros, ingredients


def iterate(ingredients, mcros, target_mcros, preferences = 4):

    minimal_error = sum([abs(mcros[i] - target_mcros[i]) for i in range(1, preferences)])
    net_effect = 1000
    ing_to_add = "N"
    boo = True
    
    for ing in values.keys():

        effect = sum([abs(values[ing][0][i] + mcros[i] - target_mcros[i]) for i in range(1, preferences)]) - minimal_error
        values[ing][1] = effect

        if values[ing][1] < net_effect:

            net_effect = values[ing][1]
            ing_to_add = ing

    for ing in ingredients:

        ing = ing[0]
        subtract_effect = sum([abs(-values[ing][0][i] + mcros[i] - target_mcros[i]) for i in range(1, preferences)]) - minimal_error
        values[ing][1] = subtract_effect
        if subtract_effect < net_effect:

            net_effect = subtract_effect
            boo = False
            ing_to_add = ing

    ing_to_add = [ing_to_add, dict(zip(dic.values(), values[ing_to_add][0]))]           
            
    if boo:
        
        ingredients.append(ing_to_add)
    else:
        
        ingredients.remove(ing_to_add)
        del values[ing_to_add[0]]
    
    return ingredients, [mcros[i] + ing_to_add[1][dic[i]] if boo else mcros[i] - ing_to_add[1][dic[i]] for i in range(len(dic))], minimal_error + net_effect

def feedback(arr, initial_list, mcros):
        
    arr = [x - 1 for x in arr]

    if arr:

        for i in range(len(arr)):

            del_mcros = initial_list[arr[len(arr)-1-i]][1]
            name = initial_list[arr[len(arr)-1-i]][0]
            del initial_list[arr[len(arr)-1-i]]
            del values[name]
            mcros = [mcros[i] - del_mcros[dic[i]] for i in range(len(dic))]
            
            
        initial_list, mcros, error = iterate(initial_list, mcros, target_mcros)

        display(mcros, initial_list, error, target_mcros)

        return initial_list, mcros

    else:
        return [x[0] for x in initial_list]

target_mcros = list(map(int, input().split()))

current_list, mcros = main(target_mcros)


while True:
    arr = list(map(int, input().split()))

    if arr:
        current_list, mcros = feedback(arr, current_list, mcros)
   

    else:

        final_list = feedback(arr, current_list, mcros)
        break


print(final_list)
