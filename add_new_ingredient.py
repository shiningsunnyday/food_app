from flask import jsonify
import pandas as pd
import requests
import json

def add_new(x):

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
    return jsonify(dic)


