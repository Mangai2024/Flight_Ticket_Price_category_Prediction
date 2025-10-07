**Flight Ticket Price Prediction**
I worked with the Flight Ticket Prediction Dataset, applying Regression, Classification, and Price Category Prediction approaches to analyze and model the data effectively.
**Here Flight Ticket Price Category Prediction**

This project focuses on predicting the price category (Low, Medium, High) of flight tickets based on various flight features such as airline, source city, destination city, number of stops, duration, class, and days left before departure.
Simple classification pipeline to predict flight ticket price bands — **low, medium, high.**

This repository contains data cleaning, feature engineering, and classification experiments (RandomForest / XGBoost) that transform raw ticket price into categories and train models to predict those categories from flight/meta features.

Project snapshot (final model results)

Task: Multiclass classification (3 classes: low, medium, high)

Best model: XGBoost (trained on encoded categorical features)

Accuracy:** **0.92254 (≈ 92.25%)****

Test set (support = 60031)

Classification report (summary):

Class 0: precision 0.99, recall 0.96, f1-score 0.97 (support 20030)

Class 1: precision 0.89, recall 0.93, f1-score 0.91 (support 20003)

Class 2: precision 0.89, recall 0.88, f1-score 0.88 (support 19998)

Confusion matrix:

[[19160    25   845]
 [    1 18628  1374]
 [  139  2266 17593]]

What this repo does (high level)

Loads Flight_Ticket_Prediction_Dataset.csv.

Cleans data: drop irrelevant columns, remove rows with missing price.

Creates price_category using quantile binning: pd.qcut(price, q=3, labels=['low','medium','high']).

Encodes price_category into numeric labels for training (LabelEncoder).

Encodes categorical features (LabelEncoder or pd.get_dummies) for model input.

Trains classifier(s) — RandomForest and XGBoost tested; XGBoost used for final results.

Evaluates model with accuracy, classification report and confusion matrix.

Why bin price into categories?

Converting continuous price → discrete bands (low/medium/high) turns this into a classification problem (useful when decision-makers care about ranges rather than exact price).

Use pd.qcut when you want equal-sized bins (by count). Use pd.cut if you prefer equal-width ranges.

Quick reproduction steps
1. Requirements

Create a venv or conda env and install:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn


(Or use the provided requirements.txt if included.)

2. Run the notebook / script

A typical script order:

# load
df = pd.read_csv("Flight_Ticket_Prediction_Dataset.csv")

# cleanup
df.drop(columns=['Unnamed: 0','flight'], inplace=True)  # drop if present
df.dropna(subset=['price'], inplace=True)

# create categories
df['price_category'] = pd.qcut(df['price'], q=3, labels=['low','medium','high'])

# encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['price_category'])

# drop original price to avoid leakage
X = df.drop(['price_category','price'], axis=1)

# encode categorical features
from sklearn.preprocessing import LabelEncoder
for col in X.select_dtypes(include=['object','category']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# xgboost training
from xgboost import XGBClassifier
model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
model.fit(X_train, y_train)

# evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

Suggested file structure
.
├─ data/
│  └─ Flight_Ticket_Prediction_Dataset.csv
├─ notebooks/
│  └─ EDA_and_Modeling.ipynb
├─ src/
│  ├─ preprocess.py
│  ├─ train.py
│  └─ evaluate.py
├─ requirements.txt
└─ README.md

Tips & notes (practical)

Avoid leakage: Drop the original price column when training a classifier on price_category. Otherwise the model sees the true price and trivially learns the label.

qcut vs cut: pd.qcut makes equal-frequency bins — good when you want class balance. If many identical prices cause qcut errors, use pd.cut.

Encoding: For tree models LabelEncoder is fine. For linear models use pd.get_dummies() (one-hot).

Class mapping: LabelEncoder().classes_ shows string class order. The numeric mapping is produced by fit_transform. Use np.unique(y, return_counts=True) to verify classes are present.

Model tuning: Improve performance with hyperparameter tuning (RandomizedSearchCV / GridSearchCV) and try LightGBM or CatBoost for additional gains.

Feature importance: Plot model.feature_importances_ to interpret which features drive predictions.

How to explain in an interview (2–3 lines)

“I converted continuous ticket prices into three quantile-based categories (low, medium, high) to make a robust classification pipeline. I dropped the original price to prevent leakage, encoded categorical features, and trained XGBoost — achieving ~92% accuracy with balanced per-class performance. I validated with confusion matrix and classification report.”

**Next steps & potential improvements**

Hyperparameter tuning (RandomizedSearchCV) for XGBoost parameters (n_estimators, max_depth, learning_rate etc.).

Try advanced encoders for high-cardinality categorical features (target encoding, frequency encoding).

Use time-based cross-validation if data is temporal.

Add calibration if probability estimates are needed.

Deploy as a service or create a Streamlit demo for demoing predictions.

License

Include your preferred license (e.g., MIT).
LICENSE file recommended.
