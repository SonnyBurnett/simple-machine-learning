import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def read_csv_file_to_dataframe(file_name):
    df = pd.read_csv (file_name)
    return df


def extract_rows_we_need(df, number_of_classes):
    indexCat = df[(df['Category'] > number_of_classes-1)].index
    df.drop(indexCat, inplace=True)
    return df


def remove_illegal_chars(df):
    headers = ['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']
    for x in headers:
        df[x] = df[x].map(fix_deviations)
    return df


def fix_deviations(x):
    if type(x) != str:
        return float(0)
    else:
        strange = re.findall("[^0-9]", x)
        if len(strange) > 0:
            if strange == 't':
                return float(0)
            elif strange == ',':
                return format_float_number(x)
            elif strange == '-':
                index = x.find('-')
                return float(x[index:])
            elif strange == '.':
                return float(x)
            else:
                return float(0)
        else:
            return float(x)


def format_float_number(x):
    return float(re.sub("[^0-9.]", "", x))


def simplify_categories(df):
    headers = ['Category']
    for x in headers:
        df[x] = df[x].map(make_numeric_categories)

    #print(df["Category"].unique())
    #print(df["Category"].value_counts())
    return df


def make_category_dict2():
    return {0: 'Fruits', 1: 'Vegetables', 2: 'Dairy', 3: 'Meat' }


def make_category_dict():
    return {0: 'Vegetarian', 1: 'Animalian' }


def make_numeric_categories(x):
    if x.find('Fruits') > -1:
        return 0
    elif x.find('Vegetables') > -1:
        return 0
    elif x.find('Dairy') > -1:
        return 1
    elif x.find('Meat') > -1:
        return 1
    else:
        return 2


def normalize_data(df):
    d = preprocessing.normalize(df)
    scaled_df = pd.DataFrame(d)
    scaled_df.head()


def make_training_data(data_input):
    train_set = data_input.tail(len(data_input) - 50)
    train_names = train_set['Food']
    #train_predictors = train_set[['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']].to_numpy()
    train_predictors = preprocessing.normalize(train_set[['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']].to_numpy())
    train_outcome = np.asarray([int(x) for x in train_set['Category'].to_numpy()])
    return train_predictors, train_outcome


def make_test_data(data_input):
    test_set = data_input.head(50)
    test_names = test_set['Food']
    #test_predictors = test_set[['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']].to_numpy()
    test_predictors = preprocessing.normalize(test_set[['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']].to_numpy())
    test_outcome = np.asarray([int(x) for x in test_set['Category'].to_numpy()])
    return test_names, test_predictors, test_outcome


def read_and_prepare_data():
    data_source = read_csv_file_to_dataframe('nutrients.csv')
    data_source = remove_illegal_chars(data_source)
    data_numerical = simplify_categories(data_source)
    data_input = extract_rows_we_need(data_numerical, 2)
    data_input = data_input.sort_values('Food')
    return data_input
