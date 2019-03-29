from flask import Flask, jsonify, render_template, request, redirect, session, url_for
import pandas as pd
import numpy as np
from food_app import app
import json

from food_app.functions import add_new, generate, iterate, cluster, macros, return_all, substitute, sorted_scores, get_score, fix_diff, get_info
import numpy as np

from food_app.forms import MacrosForm, IngredientsForm
# from bokeh.plotting import figure
# from bokeh.resources import CDN
# from bokeh.embed import file_html, components
# from bokeh.util.string import encode_utf8
# from bokeh.resources import INLINE

@app.route('/')
def api_root():
    # plot = figure()
    # plot.circle([1,2], [3,4])
    # js_resources = INLINE.render_js()
    # css_resources = INLINE.render_css()
    # script, div = components(plot)
    # html = render_template('base.html',
    #                         plot_div = div,
    #                         plot_script = script,
    #                         js_resources = js_resources,
    #                         css_resources = css_resources)
    # return encode_utf8(html)
    return render_template('base.html')
    # <!-- {{ js_resources|indent(4)|safe }}
    # {{ css_resources|indent(4)|safe }}
    # {{ plot_script|safe }} -->
    # <!-- {{ plot_div|indent(4)|safe }} -->

@app.route('/info/')
def app_info():
    return render_template('info.html')

@app.route('/rank/', methods = ['GET', 'POST'])
def api_rank():
    form = IngredientsForm()
    if form.validate_on_submit():
        ingredients = json.dumps(str(form.ingredients.data))
        ingredients = ingredients.strip('\"')
        candidates = ingredients.split(';')
        candidates = [[ingredient.strip(' ') for ingredient in meal.split(',')] for meal in candidates]
        result = list(sorted_scores([recipe for recipe in candidates if len(recipe) < 25 and len(recipe) > 0]))
        return render_template('rank.html', form = form, result = result)
    return render_template('rank.html', form = form)

@app.route('/input/', methods=['GET', 'POST'])
def api_input():
    form = MacrosForm()
    if form.validate_on_submit():
        calories = json.dumps(int(form.calories.data))
        protein = json.dumps(int(form.protein.data))
        fat = json.dumps(int(form.fat.data))
        carbs = json.dumps(int(form.carbs.data))
        target_macros = str(calories) + '_' + str(protein) + '_' + str(fat) + '_' + str(carbs)
        requirements, display_list = macros(target_macros)
        print(display_list)
        return render_template('input.html',
                                form = form,
                                requirements = requirements,
                                display_list = display_list)
    return render_template('input.html', form = form)

@app.route('/cluster/', methods = ['GET', 'POST'])
def api_cluster():
    requirements = []; list_of_ingredient_lists = []
    form = IngredientsForm()
    if form.validate_on_submit():
        string = json.dumps(str(form.ingredients.data))
        num_meals = int(json.dumps(int(form.num_meals.data)))
        requirements, list_of_ingredient_lists = cluster(string, num_meals)
        return render_template('cluster.html',
                                form = form,
                                requirements = requirements,
                                list_of_ingredient_lists = list_of_ingredient_lists)
    return render_template('cluster.html', form = form)

@app.route('/substitute/', methods = ['GET', 'POST'])
def api_substitute():
    form = IngredientsForm()
    if form.validate_on_submit():
        ingredients = json.dumps(str(form.ingredients.data))
        ingredients = ingredients.strip('\"')
        session['ingredients'] = ingredients
        ingredients = substitute_(ingredients)
        return render_template('substitute.html', form = form, ingredients = ingredients)
    return render_template('substitute.html', form = form)

@app.route('/info/all/', methods = ['GET'])
def api_all():
    return return_all()

@app.route('/add/', methods = ['GET'])
def api_add():
    x = str(request.args.get('ingredient'))
    boo = bool(int(request.args.get('boo')))
    return add_new(x.replace('_', ' '), boo)

@app.route('/diff/<diff>', methods = ['GET'])
def api_fix(diff):
    return fix_diff(diff)

@app.route('/returninfo/<returninfo>', methods = ['GET'])
def returninfo(returninfo):
    return get_info(returninfo)

if __name__ == "__main__":
    app.run(host = "0.0.0.0")
