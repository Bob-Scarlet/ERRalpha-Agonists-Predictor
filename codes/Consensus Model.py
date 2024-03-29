import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, \
    roc_auc_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from joblib import dump, load
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
model_path = './final_models_consensus/consensus_model.joblib'

def split_rus(df):
    X, y = df.iloc[:, 2:], df['label']
    X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.1, random_state=25)
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_1, y_1)
    return X_train, X_test, y_train, y_test

# 将数据集分为训练集和测试集,y_train1=y_train2=y_train3,y_test同理
X_train_AP2D, X_test_AP2D, y_train1, y_test1 = split_rus(pd.read_csv("fingerprints/rdkit-d+AP2D.csv"))
X_train_ECFP4, X_test_ECFP4, y_train2, y_test2 = split_rus(pd.read_csv("fingerprints/rdkit-d+ECFP4.csv"))
X_train_EState, X_test_EState, y_train3, y_test3 = split_rus(pd.read_csv("fingerprints/rdkit-d+EState.csv"))
X_train_FCFP4, X_test_FCFP4, y_train4, y_test4 = split_rus(pd.read_csv("fingerprints/rdkit-d+FCFP4.csv"))
X_train_MACCS, X_test_MACCS, y_train5, y_test5 = split_rus(pd.read_csv("fingerprints/rdkit-d+MACCS.csv"))


# 定义第一级模型（基模型路径需自行修改）
model_1 = load('./final_models/Replicate_4/lgb_rdkit-d+MACCS.joblib')
model_2 = load('./final_models/Replicate_4/lgb_rdkit-d+AP2D.joblib')
model_3 = load('./final_models/Replicate_4/xgb_rdkit-d+MACCS.joblib')
model_4 = load('./final_models/Replicate_4/lgb_rdkit-d+ECFP4.joblib')
model_5 = load('./final_models/Replicate_4/lgb_rdkit-d+FCFP4.joblib')

# 使用第一级模型对训练集进行拟合，并对测试集进行预测
y_pred_1_train = model_1.predict_proba(X_train_MACCS)[:, 1]
y_pred_1_test = model_1.predict_proba(X_test_MACCS)[:, 1]

y_pred_2_train = model_2.predict_proba(X_train_AP2D)[:, 1]
y_pred_2_test = model_2.predict_proba(X_test_AP2D)[:, 1]

y_pred_3_train = model_3.predict_proba(X_train_MACCS)[:, 1]
y_pred_3_test = model_3.predict_proba(X_test_MACCS)[:, 1]

y_pred_4_train = model_4.predict_proba(X_train_ECFP4)[:, 1]
y_pred_4_test = model_4.predict_proba(X_test_ECFP4)[:, 1]

y_pred_5_train = model_5.predict_proba(X_train_FCFP4)[:, 1]
y_pred_5_test = model_5.predict_proba(X_test_FCFP4)[:, 1]

# 使用第一级模型的预测结果作为输入
X_train = np.column_stack((y_pred_1_train, y_pred_2_train, y_pred_3_train, y_pred_4_train, y_pred_5_train))
X_test = np.column_stack((y_pred_1_test, y_pred_2_test, y_pred_3_test, y_pred_4_test, y_pred_5_test))


# 训练第二级模型
model_second = LogisticRegression(random_state=42, solver='liblinear', C=200, penalty='l1', class_weight={0:0.8, 1:0.2})

model_second.fit(X_train, y_train2)

# 保存模型
if not os.path.isdir('final_models_consensus'):
    os.makedirs('final_models_consensus')
dump(model_second, './final_models_consensus/consensus_model.joblib')


# 测试集
def test(X_test, y_test):
    # 使用第二级模型对第一级模型的预测结果进行拟合，并对测试集进行预测
    stacking_model = load('./final_models_consensus/consensus_model.joblib')
    y_pred = stacking_model.predict(X_test)
    y_proba = stacking_model.predict_proba(X_test)[:, 1]

    # 输出模型在测试集上的准确率
    def new_confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=[0, 1])

    def se(y_true, y_pred):
        cm = new_confusion_matrix(y_true, y_pred)
        return cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])

    def sp(y_true, y_pred):
        cm = new_confusion_matrix(y_true, y_pred)
        return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])

    metrics = {'AUC': roc_auc_score(y_test, y_proba),
               'ACC': accuracy_score(y_test, y_pred),
               'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
               'SE': se(y_test, y_pred),
               'SP': sp(y_test, y_pred),
               'F1': f1_score(y_test, y_pred),
               'MCC': matthews_corrcoef(y_test, y_pred)}
    print('测试集：', metrics)
    print('\n')

test(X_test, y_test1)

def outer():
    # 将数据集分成X和Y，其中X为特征，Y为目标变量，函数名不改了
    def split_rus(df):
        X, y = df.iloc[:, 2:], df['label']
        return X, y

    # 导入数据,y1=y2=...=y5
    X_AP2D, y1 = split_rus(pd.read_csv("fingerprints_outer/rdkit-d+AP2D.csv"))
    X_ECFP4, y2 = split_rus(pd.read_csv("fingerprints_outer/rdkit-d+ECFP4.csv"))
    X_EState, y3 = split_rus(pd.read_csv("fingerprints_outer/rdkit-d+EState.csv"))
    X_FCFP4, y4 = split_rus(pd.read_csv("fingerprints_outer/rdkit-d+FCFP4.csv"))
    X_MACCS, y5 = split_rus(pd.read_csv("fingerprints_outer/rdkit-d+MACCS.csv"))

    # 定义第一级模型
    model_1 = load('./final_models/lgb_rdkit-d+MACCS.joblib')
    model_2 = load('./final_models/lgb_rdkit-d+AP2D.joblib')
    model_3 = load('./final_models/xgb_rdkit-d+MACCS.joblib')
    model_4 = load('./final_models/lgb_rdkit-d+ECFP4.joblib')
    model_5 = load('./final_models/lgb_rdkit-d+FCFP4.joblib')

    # 使用第一级模型对数据集进行拟合
    y_pred_1_test = model_1.predict_proba(X_MACCS)[:, 1]

    y_pred_2_test = model_2.predict_proba(X_AP2D)[:, 1]

    y_pred_3_test = model_3.predict_proba(X_MACCS)[:, 1]

    y_pred_4_test = model_4.predict_proba(X_ECFP4)[:, 1]

    y_pred_5_test = model_5.predict_proba(X_FCFP4)[:, 1]

    # 使用第一级模型的预测结果作为输入
    X_test = np.column_stack((y_pred_1_test, y_pred_2_test, y_pred_3_test, y_pred_4_test, y_pred_5_test))

    # 导入第二级模型
    stacking_model = load('./final_models_consensus/consensus_model.joblib')

    # 外部验证集
    def test(X_test, y_test):
        # 使用第二级模型对第一级模型的预测结果进行拟合，并对外部验证集进行预测
        y_pred = stacking_model.predict(X_test)
        y_proba = stacking_model.predict_proba(X_test)[:, 1]

        # 输出模型在外部验证集上的准确率
        def new_confusion_matrix(y_true, y_pred):
            return confusion_matrix(y_true, y_pred, labels=[0, 1])

        def se(y_true, y_pred):
            cm = new_confusion_matrix(y_true, y_pred)
            return cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])

        def sp(y_true, y_pred):
            cm = new_confusion_matrix(y_true, y_pred)
            return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])

        metrics = {'AUC': roc_auc_score(y_test, y_proba),
                   'ACC': accuracy_score(y_test, y_pred),
                   'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                   'SE': se(y_test, y_pred),
                   'SP': sp(y_test, y_pred),
                   'F1': f1_score(y_test, y_pred),
                   'MCC': matthews_corrcoef(y_test, y_pred)}
        print('外部验证集：', metrics)
        print('\n')

    test(X_test, y1)
outer()