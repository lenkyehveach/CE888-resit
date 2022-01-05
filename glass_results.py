from models import DataSet
from utils import print_metrics


from sklearn import tree
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


###############################################

ds = DataSet("glass")
X, y = ds.get_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

scale_pipe = Pipeline([
  ("scaler", StandardScaler())
])

X_train_trans = scale_pipe.fit_transform(X_train)
X_test_trans = scale_pipe.transform(X_test)

#model parameters optimised using gridsearch in notebook
RFC = RandomForestClassifier(
  random_state=0,
  max_depth=10, 
  max_features=2,
  min_samples_leaf=1
  )

RFC.fit(X_train_trans, y_train)
y_hat_rfc = RFC.predict(X_test_trans)

print("RandomForestClassifier results: \n")
print_metrics(y_test, y_hat_rfc)

## prepare new labels from model 

new_y_train, new_y_test = ds.get_new_labels(RFC, X_train_trans, X_test_trans)

# train DecisionClassifier on data with new labels 

# parameters optimiseds with gridsearch in notebook
DT = tree.DecisionTreeClassifier(
  random_state=0,
  criterion="entropy", 
  max_depth=15, 
  max_features=2,
  min_samples_leaf=1
  )

DT.fit(X_train_trans, new_y_train)

y_hat_dt = DT.predict(X_test_trans)

print("Decision Tree Classifier with distilled labels results \n")
print_metrics(y_test, [ds.decoder[x] for x in y_hat_dt])

##### DTC on normal data 

DT_plain = tree.DecisionTreeClassifier(
  random_state=0,
  criterion="entropy", 
  max_depth=15, 
  max_features=2,
  min_samples_leaf=1
  ) 

DT_plain.fit(X_train_trans, y_train)
y_hat_dtp = DT_plain.predict(X_test_trans)

print("Decision Tree trained on plain data \n")
print_metrics(y_hat_dtp, y_test)