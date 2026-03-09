"""One-hot encoding from scratch"""

import pandas as pd

def load_data(filename):
    data = pd.read_csv(filename)
    columns = [
        'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast_r_l', 'quadrant_of_breast',
        'irradiation', 'class'
    ]
    data.columns = columns

    #removing quotes from the dataset
    for col in data.columns:
        data[col] = data[col].str.replace("'","",regex=False)   #if we dont specify regex=False pandas will assume the data enclosed by the quotes to be regex

    return data

def one_hot_encoding_column(column):
    unique_values = []

    #finding unique categories
    for val in column:
        if val not in unique_values:
            unique_values.append(val)

    encoded_data={}

    for category in unique_values:

        encoded_column = []

        for val in column:

            if val == category:
                encoded_column.append(1)
            else:
                encoded_column.append(0)

        encoded_data[category] = encoded_column

    return encoded_data

def one_hot_encoding_dataset(data):
    encoded_df = pd.DataFrame()

    for column in data.columns:
        encoded_columns = one_hot_encoding_column(data[column])

        for category in encoded_columns:
            new_column_name = column + "_" + str(category)
            encoded_df[new_column_name] = encoded_columns[category]

    return encoded_df

def main():
    data = load_data("breast-cancer_2.csv")
    encoded_df = one_hot_encoding_dataset(data)
    print("\nOriginal dataset:", data)
    print("\nEncoded dataset:", encoded_df)


if __name__ == "__main__":
    main()


