# TitanicML
Code for Titanic ML competition in Kaggle

# import required modules and classes for this project
import pandas as pd

# load data
data = pd.read_csv("C:/Users/Admin/Downloads/titanic/train.csv")
test = pd.read_csv("C:/Users/Admin/Downloads/titanic/test.csv")
test_ids = test["PassengerId"]

# Data Cleaning 
def clean(data) :
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    cols =  ["SibSp", "Parch", "Fare", "Age"]
    for col in cols :
        data[col].fillna(data[col].median(), inplace=True)
    data.Embarked.fillna("U", inplace=True)
    return data        
data = clean(data)
test = clean(test)
data.head(5)

# Data Transformation (Fron String to Integer)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

cols = ["Sex", "Embarked"]

for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
    print(le.classes_)
data.head(5)

# Separate train and test data
from sklearn.linear_model  import LogisticRegression
from sklearn. model_selection import train_test_split

y = data ["Survived"]
X = data.drop("Survived", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting up the model object
# Perform model training using train dataset from the split
model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

# Perform prediction using the Test data (choose the relevant test data)
predictions = model.predict(X_val)

# Find out the score of the model
from sklearn.metrics import accuracy_score
accuracy_score(y_val, predictions)

submission_preds = model.predict(test)

# Create submission file in csv format
df = pd.DataFrame({"PassengerId":test_ids.values, 
                   "Survived":submission_preds,
                  })             
df.to_csv("submission.csv", index=False)    
