import numpy as np,pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold,GroupKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class Fea_engineering(object):
    def __init__(self):
        self.nums = ['elapsed_time','level', 'hover_duration', 'elapsed_time_diff','room_coor','screen_coor']
        self.cats = ['event_name', 'fqid', 'room_fqid', 'text']
        self.events = ['navigate_click', 'person_click', 'cutscene_click', 'object_click', 'map_hover', 'notification_click','map_click', 'observation_click', 'checkpoint']

    def fea_transform(self,data):
        dfs = []
        data = data.sort_values(by=['session_id','index'])
        data['elapsed_time_diff'] = data['elapsed_time'].diff().fillna(0)
        data['room_coor'] = np.sqrt(np.power(data['room_coor_x'],2) + np.power(data['room_coor_y'],2))
        data['screen_coor'] = np.sqrt(np.power(data['screen_coor_x'],2) + np.power(data['screen_coor_y'],2))

        del data['room_coor_x'],data['room_coor_y'],data['screen_coor_x'],data['screen_coor_y']

        for c in self.cats:
            temp = data.groupby(by=['session_id','level_group'])[c].agg('nunique')
            temp.name = temp.name + '_nunique'
            dfs.append(temp)

        for n in self.nums:
            temp = data.groupby(by=['session_id','level_group'])[n].agg('mean')
            temp.name = temp.name + '_mean'
            dfs.append(temp)
        for n in self.nums:
            temp = data.groupby(by=['session_id','level_group'])[n].agg('std')
            temp.name = temp.name + '_std'
            dfs.append(temp)
        for e in self.events:
            data[e] = (data['event_name']==e).astype('int8')
        for e in self.events + ['elapsed_time']:
            temp = data.groupby(by=['session_id','level_group'])[e].agg('sum')
            temp.name = temp.name + '_sum'
            dfs.append(temp)
        data = data.drop(self.events,axis = 1)
        df = pd.concat(dfs, axis = 1)
        df = df.fillna(-1)
        df = df.reset_index()
        df = df.set_index('session_id')
        return df


class Model_xgb(object):
    def __init__(self):
        self.gkf = GroupKFold(n_splits=20)
        self.models = {}
    def xgbModel(self,train,labels):
        all_users = train.index.unique()
        features = [c for c in train.columns if c!='level_group']
        oof = pd.DataFrame(data=np.zeros((len(all_users),18)),index=all_users)

        for i ,(train_indx,val_index) in enumerate(self.gkf.split(X = train,groups = train.index)):
            print('='*30,i+1,'='*30)
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'learning_rate': 0.05,
                'max_depth': 4,
                'n_estimators': 2023,
                'early_stopping_rounds': 50,
                'tree_method': 'hist',
                'subsample': 0.8,
                'colsample_bytree': 0.4,
                'use_label_encoder': True
            }
            for t in range(1,19):
                if t<4: grp = '0-4'
                elif t<14: grp = '5-12'
                elif t<22: grp = '13-22'

                train_x = train.iloc[train_indx]
                xtrain = train_x.loc[train_x['level_group']==grp]
                train_users = xtrain.index.values
                ytrain = labels.loc[labels['q']==t].set_index('session').loc[train_users]

                val_x = train.iloc[val_index]
                xtest = val_x.loc[val_x['level_group']==grp]
                test_users = xtest.index.values
                ytest = labels[labels['q']==t].set_index('session').loc[test_users]

                clf = XGBClassifier(**xgb_params)
                clf.fit(xtrain[features].astype('float32'),ytrain['correct'],eval_set=[(xtest[features].astype('float32'),ytest['correct'])],verbose=0)
                print(f'{t} ({clf.best_ntree_limit}) ;',end='')

                self.models[f'{grp}_{t}'] = clf
                oof.loc[test_users,t-1] = clf.predict_proba(xtest[features].astype('float32'))[:,1]
        return oof, all_users

    def xgb_score(self,train,labels):
        oof, all_users = self.xgbModel(train,labels)
        val = oof.copy()
        for k in range(18):
            temp = labels.loc[labels.q == k+1].set_index('session').loc[all_users]
            val[k] = temp.correct.values
        scores = []
        thresholds = []
        best_score = 0
        best_threshold = 0
        for threshold in np.arange(0.4,0.8,0.01):
            # print(f'{threshold:.2f}, ',end='')
            preds = (oof.values.reshape((-1))>threshold).astype('int')
            m = f1_score(val.values.reshape((-1)),preds,average='macro')
            scores.append(m)
            thresholds.append(threshold)
            if m>best_score:
                best_score = m
                best_threshold = threshold
        for k in range(18):
            m = f1_score(val[k].values,(oof[k].values>best_threshold).astype('int'),average='macro')
            print(f'Q{k}: F1 = ',m)
        s = f1_score(val.values.reshape((-1)),(oof.values.reshape((-1))>best_threshold).astype('int'),average='macro')
        print('==> Overall F1_score = ',s)