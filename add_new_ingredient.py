from flask import jsonify
import pandas as pd
import requests
import json

def add_new(x):

    url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    headers = {
        
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-app-id': '27103a3a',
            'x-app-key': '362533bdddd91cb1453b5d2ce28cd5b3',
            'x-remote-user-id': '0',
        }

    data = {"query": x}
    r = requests.post(url, headers = headers, data = json.dumps(data))
    body = json.loads(r.text)
    food = body['foods'][0]

    dic = {'label': str(x),
           'amount': str(food["serving_qty"]) + ' of ' + str(food["serving_unit"])
           'calories': int(food["nf_calories"]),
           'protein': int(food["nf_protein"]),
           'fat': int(food["nf_total_fat"]),
           'carbs': int(food["nf_total_carbohydrate"]),
           }
    return jsonify(dic)


