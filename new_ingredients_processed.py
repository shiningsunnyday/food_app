import pandas as pd
import numpy as np
import requests
import json

df = pd.read_csv('./ingredients.csv').iloc[:,1:]
df.columns = ['banana', 'blueberries', 'apple', 'whole-wheat bread', 'brown rice',
       'oatmeal', 'sweet potatoes', 'quinoa', 'greek yogurt', 'chicken breast',
       'turkey breast', 'lean beef', 'broccoli', 'spinach', 'lettuce',
       'tomatoes', 'avocado', 'salmon', 'green tea', 'skim milk',
       'almonds']

url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
headers = {
    
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-app-id': '27103a3a',
        'x-app-key': '362533bdddd91cb1453b5d2ce28cd5b3',
        'x-remote-user-id': '0',
    }

attributes = ['Ingredients', 'calories', 'fat', 'protein', 'carbs', 'serving_qty', 'serving_unit', 'serving_weight_grams']

df_nutrition = pd.DataFrame(np.array([[0] * len(attributes)] * len(df.columns)))
df_nutrition.columns = ['Ingredients', 'calories', 'fat', 'protein', 'carbs', 'serving_qty', 'serving_unit', 'serving_weight_grams']
for i in range(len(df.columns)):
    
    ingredient = df.columns[i]
    print(ingredient)
    data = {"query": ingredient}
    r = requests.post(url, headers = headers, data = json.dumps(data))
    body = json.loads(r.text)
    food = body['foods'][0]
    df_nutrition["Ingredients"][i] = df.columns[i]
    df_nutrition["calories"][i], df_nutrition["fat"][i], df_nutrition["protein"][i], df_nutrition["carbs"][i], df_nutrition["serving_qty"][i], df_nutrition["serving_unit"][i], df_nutrition["serving_weight_grams"][i] = food["nf_calories"], food["nf_total_fat"], food["nf_protein"], food["nf_total_carbohydrate"], food["serving_qty"], food["serving_unit"], food["serving_weight_grams"]
    print(df_nutrition["calories"][i])
    df_nutrition.to_csv('./ingredients_test.csv')

df_nutrition.set_index(df.columns)
df_nutrition.to_csv('./ingredients_test.csv')

