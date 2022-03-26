import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn import preprocessing

# Load the training and test data
house_file_path = 'train.csv'
home_dataset = pd.read_csv(house_file_path)
y = home_dataset.SalePrice
home_dataset.drop(['SalePrice'],axis=1,inplace=True)
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)
# test_y = test_data.SalePrice

# ---------- Clean Training Data ----------
# drop columns that have less than 1000 data points and fill in remaining nulls with column's mode
for feature, num in home_dataset.isnull().sum().items():
    if num > 1000:
        home_dataset.drop([feature],axis=1,inplace=True)
    elif num > 0:
        home_dataset[feature] = home_dataset[feature].fillna(home_dataset[feature].mode()[0])

# DEBUG
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     file = open("nulls.txt", "w")
#     file.write(str(home_dataset.isnull().sum()))
#     file.close
# home_dataset.info()

# ---------- Clean Test Data ----------
# drop columns that have less than 1000 data points and fill in remaining nulls with column's mode
for feature, num in test_data.isnull().sum().items():
    if num > 1000:
        test_data.drop([feature],axis=1,inplace=True)
    elif num > 0:
        test_data[feature] = test_data[feature].fillna(test_data[feature].mode()[0])

# DEBUG
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     file = open("nullstest.txt", "w")
#     file.write(str(test_data.isnull().sum()))
#     file.close

# ---------- Prep Data for Training ----------
# select important features
features = home_dataset.columns #NOTE: when 'features' is printed it has a weird index thing, may cause errors and we might have to change it
string_features = [ feature  for feature, datatype in home_dataset.dtypes.items() if datatype == object]

le = preprocessing.LabelEncoder()
for string in string_features:
    home_dataset[string] = le.fit_transform(home_dataset[string])

X = home_dataset[features]

# split data into validation and training
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# ---------- Train Model ----------
model = RandomForestRegressor(random_state = 1) # CHANGE THIS!!!!!!
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)

#calculate root mean square error:
val_rms = mean_squared_error(val_predictions, val_y, squared=False)

# ---------- Plot Importances ----------
importances = model.feature_importances_
x_bar= list(range(0, len(importances)))
plt.bar(x_bar, importances)

# ---------- Retrain Model with All Data ----------
le = preprocessing.LabelEncoder()
for string in string_features:
    test_data[string] = le.fit_transform(test_data[string])

full_model = RandomForestRegressor(random_state = 1)
test_X = test_data[features]
full_model.fit(X, y)
test_preds = full_model.predict(test_X)

#calculate root mean square error:
# val_rms = mean_squared_error(test_preds, test_y, squared=False)

# ---------- Clean Results for Submission ----------
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)