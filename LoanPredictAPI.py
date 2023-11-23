import pandas as pd
import json
data = pd.read_csv('https://loanpredictapibucket.s3.ap-southeast-2.amazonaws.com/train_u6lujuX_CVtuZ9i.csv')
import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
data = data.drop('Loan_ID',axis=1)
columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']
data = data.dropna(subset=columns)
data['Self_Employed'].mode()[0]
data['Self_Employed'] =data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History'].mode()[0]
data['Credit_History'] =data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['Dependents'] =data['Dependents'].replace(to_replace="3+",value='4')
data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']
cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols]=st.fit_transform(X[cols])
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)
model_df
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,y)
from sklearn import svm
model = svm.SVC()
model_val(model,X,y)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,y)
from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)
from sklearn.ensemble import GradientBoostingClassifier
model =GradientBoostingClassifier()
model_val(model,X,y)
from sklearn.model_selection import RandomizedSearchCV
log_reg_grid={"C":np.logspace(-4,4,20),
             "solver":['liblinear']}
rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                   param_distributions=log_reg_grid,
                  n_iter=20,cv=5,verbose=True)
rs_log_reg.fit(X,y)
svc_grid = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}
rs_svc=RandomizedSearchCV(svm.SVC(),
                  param_distributions=svc_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)
rs_svc.fit(X,y)
RandomForestClassifier()
rf_grid={'n_estimators':np.arange(10,1000,10),
  'max_features':[None, 'sqrt', 'log2'],
 'max_depth':[None,3,5,10,20,30],
 'min_samples_split':[2,5,20,50,100],
 'min_samples_leaf':[1,2,5,10]
 }
rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                  param_distributions=rf_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)
rs_rf.fit(X,y)
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']
rf = RandomForestClassifier(n_estimators=270,
 min_samples_split=5,
 min_samples_leaf=5,
 max_features='sqrt',
 max_depth=5)
rf.fit(X,y)
import joblib
joblib.dump(rf,'loan_status_predict')
model = joblib.load('loan_status_predict')
import pandas as pd
df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])
result = model.predict(df)
if result==1:
    print("Loan Approved")
else:
    print("Loan Not Approved")
app = Flask(__name__)
model = joblib.load('loan_status_predict')
CORS(app, resources={r"/predict": {"origins": ["https://hdbank.hutech.click", "https://ebanking.hutech.click"]}})

# Đường dẫn '/predict' sẽ xử lý cả POST và GET requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify(error="No JSON data provided"), 400

    df = pd.DataFrame(data, index=[0])
    result = model.predict(df)

    if result == 1:
        return jsonify(prediction="Loan Approved")
    else:
        return jsonify(prediction="Loan Not Approved")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
