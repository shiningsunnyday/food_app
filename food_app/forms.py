from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, SubmitField, TextAreaField, validators
from wtforms.validators import DataRequired

class MacrosForm(FlaskForm):
    calories = FloatField('Calories', validators=[DataRequired()])
    protein = FloatField('Protein', validators=[DataRequired()])
    fat = FloatField('Fat', validators=[DataRequired()])
    carbs = FloatField('Carbs', validators=[DataRequired()])
    submit = SubmitField('Submit')

class IngredientsForm(FlaskForm):
    ingredients = TextAreaField('Ingredients', [validators.optional(), validators.length(max=200)])
    num_meals = IntegerField('Number of meals', [validators.optional()])
    submit = SubmitField('Submit')
