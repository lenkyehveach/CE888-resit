from utils import print_metrics

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from decimal import Decimal, getcontext

class DataSet: 
  def __init__(self, dataset):
    self.dataset = dataset

    self.X = None
    self.y = None

    self.train_labels = None
    self.test_lables = None

    self.decoder = None 

  def get_dataset(self): 
    if self.dataset == "glass": 
      cols = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Be", "Fe", "Class"]
      data = pd.read_csv("datasets/glass.data", names=cols, index_col=0)
      data["Class"] = data["Class"].apply(lambda x: 3 if x >=4 else x)

      self.X = data.drop("Class", axis=1)
      self.y = data["Class"]

      return self.X, self.y

    if self.dataset == "bank": 
      data = pd.read_csv("datasets/bank-full.csv",delimiter=";")
      categorical = []
      for col in data.columns: 
        if data[col].dtypes == "object":
          categorical.append(col)
      
      enc = OrdinalEncoder()
      enc.fit(data[categorical])
      data[categorical] = enc.transform(data[categorical])

      self.X = data.drop("y", axis=1)
      self.y = data["y"]

      return self.X, self.y

    if self.dataset == "spam": 
      columns = ["word_freq_make",
        "word_freq_address", "word_freq_all",
        "word_freq_3d", "word_freq_our",
        "word_freq_over", "word_freq_remove",
        "word_freq_internet", "word_freq_order",
        "word_freq_mail", "word_freq_receive",
        "word_freq_will", "word_freq_people",
        "word_freq_report", "word_freq_addresses",
        "word_freq_free", "word_freq_business",
        "word_freq_email", "word_freq_you",
        "word_freq_credit", "word_freq_your",
        "word_freq_font", "word_freq_000",
        "word_freq_money", "word_freq_hp",
        "word_freq_hpl", "word_freq_george",
        "word_freq_650", "word_freq_lab",
        "word_freq_labs", "word_freq_telnet",
        "word_freq_857", "word_freq_data",
        "word_freq_415", "word_freq_85",
        "word_freq_technology", "word_freq_1999",
        "word_freq_parts", "word_freq_pm",
        "word_freq_direct", "word_freq_cs",
        "word_freq_meeting", "word_freq_original",
        "word_freq_project", "word_freq_re",
        "word_freq_edu", "word_freq_table",
        "word_freq_conference", "char_freq_;",
        "char_freq_(", "char_freq_[",
        "char_freq_!", "char_freq_$",
        "char_freq_#", "capital_run_length_average",
        "capital_run_length_longest", "capital_run_length_total",
        "spam"]
      data = pd.read_csv("spambase.data", names=columns)

      self.y = data["spam"]
      self.X = data.drop("spam", axis=1)

      return self.X, self.y

  def get_new_labels(self, model, X_train, X_test): 
    train_labels = model.predict_proba(X_train)
    test_labels = model.predict_proba(X_test)

    if self.dataset == "glass": 

      encoder = LabelEncoder()
      self.decoder = {
        0: 1,
        1: 1,
        2: 2,
        3: 2, 
        4: 3, 
        5: 3
      }
      #arrange predicted classes in order of probabilty
      train_labels = ["{0}>{1}>{2}".format(*list(np.argsort(-i)),) for i in train_labels]

      self.train_labels = encoder.fit_transform(train_labels)

      self.test_labels = ["{0}>{1}>{2}".format(*list(np.argsort(-i)),) for i in test_labels]

      return self.train_labels, self.test_labels

    if self.dataset == "bank":
      getcontext().prec = 2
      beens = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

      train_preds = []
      for x in train_labels: 
        for i in range(len(beens)):
          if x[0] == beens[i]: 
            #print("equal ", x, 1)
            #print("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
            train_preds.append("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
          if x[0] > beens[i] and x[0] < beens[i + 1]:
            #print(x, beens[i])
            #print("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
            train_preds.append("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
            
      test_preds = []
      for x in test_labels: 
        for i in range(len(beens)):
          if x[0] == beens[i]: 
            #print("equal ", x, 1)
            #print("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
            test_preds.append("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
          if x[0] > beens[i] and x[0] < beens[i + 1]:
            #print(x, beens[i])
            #print("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
            test_preds.append("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))

      enc = LabelEncoder()
      
      self.decoder = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0
      }

      self.train_labels = enc.fit_transform(train_preds)
      self.test_labels = test_preds
      return self.train_labels, self.test_labels

    if self.dataset == "spam": 
      getcontext().prec = 2
      beens = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

      train_preds = []
      for x in train_labels: 
          for i in range(len(beens)):
              if x[0] == beens[i]: 
                  #print("equal ", x, 1)
                  #print("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
                  train_preds.append("{}-yes {}-no".format(beens[i], 1 - Decimal(beens[i])))
              if x[0] > beens[i] and x[0] < beens[i + 1]:
                  #print(x, beens[i])
                  #print("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
                  train_preds.append("{}-yes {}-no".format(beens[i], 1 - Decimal(beens[i])))
                  
      test_preds = []
      for x in test_labels: 
          for i in range(len(beens)):
              if x[0] == beens[i]: 
                  #print("equal ", x, 1)
                  #print("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
                  test_preds.append("{}-yes {}-no".format(beens[i], 1 - Decimal(beens[i])))
              if x[0] > beens[i] and x[0] < beens[i + 1]:
                  #print(x, beens[i])
                  #print("{}-no {}-yes".format(beens[i], 1 - Decimal(beens[i])))
                  test_preds.append("{}-yes {}-no".format(beens[i], 1 - Decimal(beens[i])))

      enc = LabelEncoder()
      
      self.decoder = {
          0: 1,
          1: 1,
          2: 1,
          3: 1,
          4: 1,
          5: 1,
          6: 0,
          7: 0,
          8: 0,
          9: 0,
          10: 0
          
      }
      
      self.train_labels = enc.fit_transform(train_preds)
      self.test_labels = test_preds

      return self.train_labels, self.test_labels
