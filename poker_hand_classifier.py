import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

# Load the datasets
training_data = pd.read_csv('C:\\Users\\hosha\\poker\\poker-hand-training.csv')

# Extract features and target from training data
X = training_data.drop(columns=['Poker Hand'])
y = training_data['Poker Hand']

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.25, random_state=42, stratify=y_res)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define the XGBoost classifier with parameters
xgb_classifier = XGBClassifier(
    n_estimators=400,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=1.0,
    min_child_weight=3,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=42,
    early_stopping_rounds=10,
)

eval_set = [(X_val, y_val)]
xgb_classifier.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Save the trained model
with open('models/xgb_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(xgb_classifier, model_file)

# Save the scaler
with open('models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
