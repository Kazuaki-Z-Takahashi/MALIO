
import pandas as pd
import time
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "-df","--data_frame",
    default = "data_frame_all.csv",
    help = "specify input CSV file")
parser.add_argument(
    "-r","--ranking_file",
    default = "ranking.csv",
    help = "specify output ranking file name")

args = parser.parse_args()
data_frame  = args.data_frame
ranking_file = args.ranking_file

t1 = time.time()

data_frame = pd.read_csv("__tmp/" + data_frame, index_col=0)
print(data_frame.head())
print(data_frame.tail())
print(data_frame.shape)


### Split data frame for ML ###
train_x = data_frame.drop('label', axis=1)
train_y = data_frame['label']


### Import ML tools ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


### Closs validation start ###
def cross_val_wrap(clf, train_x, train_y):
 split_num = 5 # the number of doing closs validations.
 cv = StratifiedKFold(n_splits=split_num, random_state=666, shuffle=True)
 acc_all = cross_val_score(clf, train_x, train_y, cv=cv, scoring='accuracy', n_jobs=split_num)
 # "n_jobs" can be set along with the number of threads.
 return acc_all


### Evaluate classification accuracy of each LOP ###
import numpy as np
acc_data = {}
for ivar in train_x:
 train_x_1feat = train_x.loc[:, [str(ivar)]]
 clf = RandomForestClassifier(n_estimators=10)
 acc_all = cross_val_wrap(clf, train_x_1feat, train_y)
 acc_data[ivar] = np.average(acc_all)
ranking = sorted(acc_data, key=acc_data.get, reverse=True)
for i in range(min(len(ranking), 100)):
 print(i, ranking[i], acc_data[ranking[i]])
rank_data = []
for i in range(len(ranking)):
 rank_data.append([ranking[i], acc_data[ranking[i]]])
rank_pd = pd.DataFrame(rank_data)

os.makedirs("__output", exist_ok=True)
rank_pd.to_csv("__output/" + ranking_file, sep=',')

t2 = time.time() - t1
print ("### Record ### Elap.Time:{0}".format(t2) + "[sec]")
