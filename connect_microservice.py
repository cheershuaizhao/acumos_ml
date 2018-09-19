import numpy as np
#proctol buffer files are generated by Acumos platform, please replace it with your own
import sentiment_xxxxx as pb
import requests
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


restURL = "http://localhost:3330/classify"
x_test= pd.read_csv("test_data.csv", header=None, names=['review'])
y_test = pd.read_csv("test_label.csv", header=None, names=['label'])

def connector_prediction(x_test, y_true):
    data = pb.Input()
    data.review.extend(x_test)
    data.g_truth.extend(y_true)
    r = requests.post(restURL, data.SerializeToString())
    of = pb.Output()
    of.ParseFromString(r.content)
    return of.sentiment

#get purediction
yhat = connector_prediction(x_test)
yhat = list(yhat)