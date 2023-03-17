import numpy as np
import pandas as pd
from feature_engineering import Fea_engineering
from feature_engineering import Model_xgb


def go():
    train_path = '../dataset/train.csv'
    labels_path = '../dataset/train_labels.csv'

    train = pd.read_csv(train_path)
    labels = pd.read_csv(labels_path)

    labels['session'] = labels['session_id'].apply(lambda x:int(x.split('_')[0]))
    labels['q'] = labels['session_id'].apply(lambda x:int(x.split('q')[1]))

    enginer = Fea_engineering()
    model = Model_xgb()

    data = enginer.fea_transform(train)
    model.xgb_score(data,labels)


if __name__ == '__main__':
    go()