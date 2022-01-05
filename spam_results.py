from models import DataSet
from utils import print_metrics


from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

ds = DataSet("spam")
X, y = ds.get_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

params = {
          'max_features': [ 2, 4, 8, 10 ,15 ,20, 25, 30 ,40 ,50 ],
          'max_depth': [ 25, 30 ,40 ,50, 60],
          'min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5,  1]}

RFC_1 = RandomForestClassifier(n_estimators=100, random_state=0)

grid_rfc = GridSearchCV(RFC_1, params, cv=5, n_jobs=-1)

grid_rfc.fit(X_train, y_train)
print('Best score:', grid_rfc.best_score_)
print('Best params:', grid_rfc.best_params_)

y_hat = grid_rfc.predict(X_test)
print_metrics(y_test, y_hat)

new_y_train, new_y_test  = ds.get_new_labels(grid_rfc, X_train, X_test)

DT = tree.DecisionTreeClassifier(
  random_state=0,
  criterion="gini", 
  max_depth=15, 
  max_features=25,
  min_samples_leaf=1,
  max_leaf_nodes=20
  )

DT.fit(X_train, new_y_train)
y_hat_dt = DT.predict(X_test)
print("\n Decision Tree Classifier with distilled labels results")
print_metrics(y_test, [ds.decoder[x] for x in y_hat_dt])

#plain data

DT_plain = tree.DecisionTreeClassifier(
  random_state=0,
  criterion="gini", 
  max_depth=15, 
  max_features=25,
  min_samples_leaf=1,
  max_leaf_nodes=20
  )

DT_plain.fit(X_train, y_train)
y_hat_dtp = DT_plain.predict(X_test)

print("Decision Tree trained on plain data \n")
print_metrics(y_hat_dtp, y_test)