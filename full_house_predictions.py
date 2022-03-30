import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

# ---------- Load the Training and Test Data ----------
house_file_path = 'train.csv'
home_dataset = pd.read_csv(house_file_path)
y = home_dataset.SalePrice
home_dataset.drop(['SalePrice'],axis=1,inplace=True)
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)

# ---------- Clean Training Data ----------
# drop columns that have less than 1000 data points and fill in remaining nulls with column's mode
for feature, num in home_dataset.isnull().sum().items():
    if num > 1000:
        home_dataset.drop([feature],axis=1,inplace=True)
    elif num > 0:
        home_dataset[feature] = home_dataset[feature].fillna(home_dataset[feature].mode()[0])

# ---------- Clean Test Data ----------
# drop columns that have less than 1000 data points and fill in remaining nulls with column's mode
for feature, num in test_data.isnull().sum().items():
    if num > 1000:
        test_data.drop([feature],axis=1,inplace=True)
    elif num > 0:
        test_data[feature] = test_data[feature].fillna(test_data[feature].mode()[0])

# ---------- Prep Data for Training ----------
# select important features
features = home_dataset.columns #NOTE: when 'features' is printed it has a weird index thing, may cause errors and we might have to change it
string_features = [ feature  for feature, datatype in home_dataset.dtypes.items() if datatype == object]

# convert string type data columns to integers
le = preprocessing.LabelEncoder()
for string in string_features:
    home_dataset[string] = le.fit_transform(home_dataset[string])

X = home_dataset[features]

# split data into validation and training
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# ---------- Create Model and Train ----------
# create model with recursive feature elimination and cross-validation to fit best model using root mean squared error
selected_features_model = RFECV(estimator=RandomForestRegressor(random_state=1), scoring = 'neg_root_mean_squared_error', cv = 10, step = 10)

# find the model with the training data to find the best features and train the model
selected_features_model.fit(X,y)

# create an array with the selected features to run with the test data
selected_features = features.copy()
selected_features = selected_features[selected_features_model.support_]

# calculate predictions from the model and find the mean squared error to determine accuracy of model
val_predictions = selected_features_model.predict(val_X)
val_rms = mean_squared_error(val_predictions, val_y, squared=False) 
print("This is the root mean square error:", val_rms)

# convert string type data columns to integers
le = preprocessing.LabelEncoder()
for string in string_features:
    test_data[string] = le.fit_transform(test_data[string])

# calculate predictions from the model and find the mean squared error to determine accuracy of model
test_X = test_data[features]
test_preds = selected_features_model.predict(test_X)

# ---------- Clean Results for Submission ----------
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)