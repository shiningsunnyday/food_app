def to_table(array_dic):
    result = "<table>"
    result += '<tr>'
    for key in array_dic[0].keys():
        result += ('<th>' + key + '</th>')
    result += '</tr>'
    for dic in array_dic:
        result += '<tr><td>'
        result += '</td><td>'.join(list(map(str, dic.values())))
        result += '</td></tr>'
    result += "</table>"
    return result

def to_lists(display_lists):
    result = ""
    for lis in display_lists:
        result += '<ul><li>'
        result += '</li><li>'.join(list(map(lambda x: x['label'], lis)))
        result += '</li></ul>'

    return result

def to_list_of_lists(list_of_lists):
    result = ""
    return to_lists([lis['listToDisplay'] for lis in list_of_lists['listOfIngredientLists']])
