"""Implement ordinal encoding in Python from scratch."""

#Ordinal encoding assigns ordered integers to categories.

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def load_data(filename):
    data = pd.read_csv(filename)
    columns = [
        'age','menopause','tumor_size','inv_nodes','node_caps','deg_malig','breast_r_l','quadrant_of_breast','irradiation','class'
    ]
    data.columns = columns
    return data

def ordinal_encoding_column(column):    #COnvert one column of categories to numbers

    unique_values=[]

    for i in column:
        if i not in unique_values:
            unique_values.append(i)

    mapping={}

    for j in range(len(unique_values)):
        mapping[unique_values[j]]=j      # We assign numbers to the elemnts of the columns

    encoded_vals=[]
    for k in column:
        encoded_vals.append(mapping[k])   #Each element in the column is replaced with the number associated with the values

    return encoded_vals , mapping

def encoding_the_whole_data(data):
    encoded_data={}
    for column in data.columns:
        encoded_column,mapping=ordinal_encoding_column(data[column])
        data[column]=encoded_column
        encoded_data[column]=mapping

    return data,encoded_data



def main():
    data = load_data('breast-cancer_2.csv')
    encoded_vals, mapping = ordinal_encoding_column(data['age'])
    for col in mapping:
        print(col,mapping[col])

    encoded_data,mapping = encoding_the_whole_data(data)
    print("Encoded data :",encoded_data)


if __name__ == '__main__':
    main()
