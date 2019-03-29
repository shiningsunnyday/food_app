import pandas as pd
import numpy as np

def test_list(lis):

    result = np.array([[0] * len(lis)] * len(lis))
    for i in range(len(lis)):
        for j in range(i + 1, len(lis)):
            print(lis[i],"and",lis[j])
            x = int(input())
            result[i][j] = x
        df = pd.DataFrame(result)
        df.to_csv('/Users/shiningsunnyday/Desktop/ingredients.csv')
        print("Completed",lis[i])
            
    return result

lis = ['banana', 'blueberries', 'apple', 'whole-wheat bread', 'brown rice',
       'oatmeal', 'sweet potatoes', 'quinoa', 'greek yogurt', 'chicken breast',
       'turkey breast', 'lean beef', 'broccoli', 'spinach', 'lettuce',
       'tomatoes', 'avocado', 'salmon', 'green tea', 'skim milk',
       'almonds']
test_list(lis)


'''
banana and blueberries
3
banana and apple
3
banana and whole-wheat bread
2
banana and brown rice
1
banana and oatmeal
4
banana and sweet potatoes
2
banana and quinoa
1
banana and greek yogurt
4
banana and chicken breast
1
banana and turkey breast
1
banana and lean beef
1
banana and broccoli
1
banana and spinach
1
banana and lettuce
1
banana and tomatoes
1
banana and avocado
3
banana and salmon
1
banana and green tea
1
banana and skim milk
3
banana and almonds
3
Completed banana
blueberries and apple
3
blueberries and whole-wheat bread
1
blueberries and brown rice
1
blueberries and oatmeal
5
blueberries and sweet potatoes
1
blueberries and quinoa
2
blueberries and greek yogurt
4
blueberries and chicken breast
1
blueberries and turkey breast
1
blueberries and lean beef
1
blueberries and broccoli
1
blueberries and spinach
3
blueberries and lettuce
2
blueberries and tomatoes
2
blueberries and avocado
3
blueberries and salmon
1
blueberries and green tea
2
blueberries and skim milk
3
blueberries and almonds
1
Completed blueberries
apple and whole-wheat bread
1
apple and brown rice
1
apple and oatmeal
5
apple and sweet potatoes
2
apple and quinoa
2
apple and greek yogurt
1
apple and chicken breast
1
apple and turkey breast
1
apple and lean beef
1
apple and broccoli
2
apple and spinach
2
apple and lettuce
2
apple and tomatoes
2
apple and avocado
3
apple and salmon
1
apple and green tea
1
apple and skim milk
2
apple and almonds
2
Completed apple
whole-wheat bread and brown rice
1
whole-wheat bread and oatmeal
1
whole-wheat bread and sweet potatoes
1
whole-wheat bread and quinoa
1
whole-wheat bread and greek yogurt
3
whole-wheat bread and chicken breast
1
whole-wheat bread and turkey breast
1
whole-wheat bread and lean beef
1
whole-wheat bread and broccoli
1
whole-wheat bread and spinach
1
whole-wheat bread and lettuce
4
whole-wheat bread and tomatoes
4
whole-wheat bread and avocado
4
whole-wheat bread and salmon
1
whole-wheat bread and green tea
2
whole-wheat bread and skim milk
3
whole-wheat bread and almonds
3
Completed whole-wheat bread
brown rice and oatmeal
1
brown rice and sweet potatoes
2
brown rice and quinoa
2
brown rice and greek yogurt
1
brown rice and chicken breast
3
brown rice and turkey breast
3
brown rice and lean beef
3
brown rice and broccoli
4
brown rice and spinach
2
brown rice and lettuce
2
brown rice and tomatoes
1
brown rice and avocado
1
brown rice and salmon
3
brown rice and green tea
1
brown rice and skim milk
1
brown rice and almonds
1
Completed brown rice
oatmeal and sweet potatoes
2
oatmeal and quinoa
1
oatmeal and greek yogurt
3
oatmeal and chicken breast
1
oatmeal and turkey breast
1
oatmeal and lean beef
1
oatmeal and broccoli
2
oatmeal and spinach
2
oatmeal and lettuce
1
oatmeal and tomatoes
1
oatmeal and avocado
3
oatmeal and salmon
1
oatmeal and green tea
2
oatmeal and skim milk
4
oatmeal and almonds
3
Completed oatmeal
sweet potatoes and quinoa
2
sweet potatoes and greek yogurt
1
sweet potatoes and chicken breast
3
sweet potatoes and turkey breast
3
sweet potatoes and lean beef
3
sweet potatoes and broccoli
4
sweet potatoes and spinach
3
sweet potatoes and lettuce
3
sweet potatoes and tomatoes
2
sweet potatoes and avocado
1
sweet potatoes and salmon
3
sweet potatoes and green tea
2
sweet potatoes and skim milk
3
sweet potatoes and almonds
3
Completed sweet potatoes
quinoa and greek yogurt
1
quinoa and chicken breast
3
quinoa and turkey breast
3
quinoa and lean beef
3
quinoa and broccoli
4
quinoa and spinach
3
quinoa and lettuce
3
quinoa and tomatoes
3
quinoa and avocado
3
quinoa and salmon
3
quinoa and green tea
2
quinoa and skim milk
1
quinoa and almonds
3
Completed quinoa
greek yogurt and chicken breast
1
greek yogurt and turkey breast
1
greek yogurt and lean beef
1
greek yogurt and broccoli
1
greek yogurt and spinach
2
greek yogurt and lettuce
1
greek yogurt and tomatoes
1
greek yogurt and avocado
3
greek yogurt and salmon
2
greek yogurt and green tea
3
greek yogurt and skim milk
2
greek yogurt and almonds
3
Completed greek yogurt
chicken breast and turkey breast
1
chicken breast and lean beef
1
chicken breast and broccoli
4
chicken breast and spinach
3
chicken breast and lettuce
3
chicken breast and tomatoes
3
chicken breast and avocado
2
chicken breast and salmon
1
chicken breast and green tea
2
chicken breast and skim milk
1
chicken breast and almonds
1
Completed chicken breast
turkey breast and lean beef
1
turkey breast and broccoli
4
turkey breast and spinach
3
turkey breast and lettuce
3
turkey breast and tomatoes
3
turkey breast and avocado
2
turkey breast and salmon
1
turkey breast and green tea
2
turkey breast and skim milk
1
turkey breast and almonds
2
Completed turkey breast
lean beef and broccoli
4
lean beef and spinach
3
lean beef and lettuce
3
lean beef and tomatoes
3
lean beef and avocado
2
lean beef and salmon
1
lean beef and green tea
2
lean beef and skim milk
1
lean beef and almonds
1
Completed lean beef
broccoli and spinach
3
broccoli and lettuce
3
broccoli and tomatoes
3
broccoli and avocado
3
broccoli and salmon
4
broccoli and green tea
2
broccoli and skim milk
1
broccoli and almonds
2
Completed broccoli
spinach and lettuce
3
spinach and tomatoes
3
spinach and avocado
3
spinach and salmon
2
spinach and green tea
2
spinach and skim milk
3
spinach and almonds
2
Completed spinach
lettuce and tomatoes
4
lettuce and avocado
3
lettuce and salmon
2
lettuce and green tea
2
lettuce and skim milk
2
lettuce and almonds
2
Completed lettuce
tomatoes and avocado
3
tomatoes and salmon
2
tomatoes and green tea
2
tomatoes and skim milk
1
tomatoes and almonds
1
Completed tomatoes
avocado and salmon
2
avocado and green tea
3
avocado and skim milk
3
avocado and almonds
3
Completed avocado
salmon and green tea
2
salmon and skim milk
1
salmon and almonds
2
Completed salmon
green tea and skim milk
1
green tea and almonds
1
Completed green tea
skim milk and almonds
3
Completed skim milk
Completed almonds
'''
