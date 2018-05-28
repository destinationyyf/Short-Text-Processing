import xlrd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


f = xlrd.open_workbook('HS_CODE_Raw.xlsx')
table = f.sheet_by_index(0)
g = open('glove_test.txt','r',errors ='ignore')
list_temp = g.read().split('\n')

def check_number(string):
    if string == '': 
        return(False) # void string returns False
    elif string[0] in ['-','+']:
        return(check_number(string[1:])) # if first letter is '-' or '+', further checking
    else:
        if string.count('.') > 1:
            return(False) # two or more '.' returns False
        elif string.isdigit():
            return(True)
        else:
            new_str = string.split('.')
            if len(new_str) == 1: # zero '.' and not digit returns False
                return(False)
            else:
                if new_str[0].isdigit() and new_str[1].isdigit(): # both substring digits returns True
                    return(True)
                else:
                    return(False)

## Convert list of strings in to list of float numbers
def string_list_to_number(stringlist):
    numberlist =[]
    for l in stringlist:
        if check_number(l.replace(' ','')):
            numberlist.append(float(l))
    return(numberlist)

## Split the word (string) and the vector (float)
def deal_with_vec(string):
    word = ''
    loc = 0
    if string == '':
        return(0)
    elif not string[loc].isalpha():
        return(0)
    else:
        while(string[loc].isalpha()):
            word += string[loc]
            loc += 1
        return(word,string[loc + 1:]) # space between word and number, so starts at 'loc + 1'

## Transform txt into decent dictionary pattern
def get_vectors_pocket(filename,dim):
    dic = {}
    g = open(filename,'r',errors ='ignore')
    text = g.read()
    vec_pocket = text.split('\n')
    g.close()
    for v in vec_pocket:
        res = deal_with_vec(v)
        if deal_with_vec(v) != 0:
            numlist = string_list_to_number(res[1].split(' ')) # res[1] is the vector, res[0] is the word
            if len(numlist) == dim:
                dic[res[0]] = numlist
    return(dic)

WV_dict = get_vectors_pocket('glove.6B.50d.txt',50)

## Four levels of classification, 4 digits is the most general and 10 digits is the most specific.
## Split different levels of code into 4 sub-dictionaries.
def tidy_hs_by_digits(filename,sheet_index):
    dic4 = {}
    dic6 = {}
    dic8 = {}
    dic10 = {}
    f = xlrd.open_workbook(filename)
    table = f.sheet_by_index(sheet_index)
    n = table.nrows
    for i in range(n):
        if len(table.row(i)[0].value) == 4 :
            dic4[table.row(i)[0].value] = table.row(i)[1].value
        elif len(table.row(i)[0].value) == 6 :
            dic6[table.row(i)[0].value] = table.row(i)[1].value
        elif len(table.row(i)[0].value) == 8 :
            dic8[table.row(i)[0].value] = table.row(i)[1].value
        else:
            dic10[table.row(i)[0].value] = table.row(i)[1].value
    return(dic4,dic6,dic8,dic10)

results = tidy_hs_by_digits('HS_CODE_Raw.xlsx',0)

def construct_empty(filename):
    f = open(filename,'r')
    text = f.read()
    f.close()
    empty_dict = []
    raw = text.split('\n')
    for r in raw:
        empty_dict.extend(r.split(' '))
    return(empty_dict)

emptyset = construct_empty('Empty.txt')

def build_hs_corpus(digits,results):
    digit = [4,6,8,10]
    loc = digit.index(digits)
    dictionary_needed = results[loc]
    for i in np.arange(3,-1,-1):
        if i > loc:
            dictionary_temp = results[i]
            for d in dictionary_temp.keys():
                if d[:digits] in dictionary_needed.keys():
                    dictionary_needed[d[:digits]] += ' ' + dictionary_temp[d]
                else:
                    dictionary_needed[d[:digits]] = dictionary_temp[d]
        elif i < loc:
            dictionary_temp = results[i]
            for d in dictionary_needed.keys():
                if d[:digit[i]] in dictionary_temp.keys():
                    dictionary_needed[d] += ' ' + dictionary_temp[d[:digit[i]]]
    return(dictionary_needed)

results_2 = build_hs_corpus(6,results)

sign_pocket = [' ','\n',',','.','/','\\','?','$','#','@','!','\'',';',':','(',')','&','*','^','%','+','_','[','=',']','<','>','|','1','2','3','4','5','6','7','8','9','0']

def split_all(string,sign_pocket,emptyset):
    str_list = []
    temp = ''
    for p in string.lower():
        if p not in sign_pocket:
            temp = temp + p
        elif temp != '' and temp not in emptyset:
            str_list.append(temp)
            temp = ''
        elif temp != '' and temp in emptyset:
            temp = ''
    if temp != '' and temp not in emptyset:
        str_list.append(temp)
    return(str_list)

def get_hs_word_list(dictionary_raw,sign_pocket,emptyset):
    dict_need = {}
    for d in dictionary_raw.keys():
        dict_need[d] = split_all(dictionary_raw[d],sign_pocket,emptyset)
    return(dict_need)

dictionary = get_hs_word_list(results_2,sign_pocket,emptyset)

## Check if the word exists in the GloVe database
def Convert_WL2WV(WL,WordVector_dict):
    points = []
    for w in WL:
        if w in WordVector_dict.keys():
            points.append(WordVector_dict[w])
    return(points)

def build_HS_cluster_chrct(HS_dictionary,WordVector_dict,K):
    HS_clus = {}
    count_flag = 0
    for k in HS_dictionary.keys():
        points = Convert_WL2WV(HS_dictionary[k],WordVector_dict)
        if points != []:
            kmeans = KMeans(n_clusters = min(K,len(points)))
            kmeans.fit(points)
            centers = kmeans.cluster_centers_
            percentage = []
            for j in range(len(centers)):
                percentage.append(kmeans.labels_.tolist().count(j)/len(kmeans.labels_))
            HS_clus[k] = [centers,percentage]
        count_flag += 1
        print(count_flag,len(points))
    return(HS_clus)

HS_clus = build_HS_cluster_chrct(results_2,WV_dict,3)

def determine_weighted(matrix,Wvector):
    weighted = []
    for i in range(len(matrix)):
        m = min(matrix[i])
        loc = matrix[i].index(m)
        weighted.append(Wvector[loc]*m)
    return(weighted)

def forecast_HS_by_depict(depict_vec_list,HS_clus):
    error = -1
    for hs in HS_clus.keys():
        weighted = determine_weighted(cdist(depict_vec_list,HS_clus[hs][0],"euclidean").tolist(),HS_clus[hs][1])
        error_temp = np.mean(weighted)
        if error == -1:
            error = error_temp
            res = hs
        elif error_temp < error:
            error = error_temp
            res = hs
    return(res,error)

def forecast_HS_full(filename,HS_clus,WordVector_dict):
    f = xlrd.open_workbook(filename)
    table = f.sheet_by_index(0)
    target_dict = {}
    forecast_dict = {}
    for i in range(table.nrows):
        target_dict[i] = table.row(i)[1].value
        print(i)
    target_dict = get_hs_word_list(target_dict,sign_pocket,emptyset)
    for i in target_dict.keys():
        WV_temp = Convert_WL2WV(target_dict[i],WordVector_dict)
        if WV_temp != []:
            forecast_dict[i] = forecast_HS_by_depict(WV_temp,HS_clus)
        else:
            forecast_dict[i] = 'Null'
        print(forecast_dict[i],target_dict[i])
    return(target_dict,forecast_dict)

result0 = forecast_HS_full('effective.xls',HS_clus,WV_dict)