from sklearn.model_selection import train_test_split
import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, \
    roc_auc_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from joblib import dump, load
import re
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# 需在fingerprints以及fingerprints_outer中预先准备好相应的特征文件
# 特征文件命名应为'rdkit-d+AP2D','rdkit-d+ECFP4','rdkit-d+EState','rdkit-d+FCFP4','rdkit-d+MACCS'，内容参考template.csv

### 功能介绍
# 欠采样✖4种机器学习算法✖5种组合特征
# 训练、测试、外部验证一体化
# 集成了数据后处理，将训练、测试、外部验证的结果保存到一个xlsx文件的3个sheet中
# 重复10次实验，平均值和标准差同样保存至一个xlsx文件中


# 拆分+欠采样
def split_rus(df):
    X, y = df.iloc[:, 2:], df['label']
    X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.1, random_state=random)
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_1, y_1)
    return X_train, X_test, y_train, y_test

# 供outer使用
def split_outer(df):
    X, y = df.iloc[:, 2:], df['label']
    return X, y

# LightGBM,xgb,SVC,RF GridCV+5折交叉验证
lgb_params = {'random_state': [42],
              'objective': ['binary'],
              'boosting_type': ['gbdt'],
              'scale_pos_weight': [1, 1.2, 1.4, 1.6, 1.8, 2],
              'num_leaves': [i for i in range(31, 80, 16)],
              'n_estimators': [i for i in range(100, 501, 100)]}

RF_params = {'random_state': [42],
             'max_depth': [i for i in range(1, 10, 2)],
             'criterion': ['gini'],
             'class_weight': ['balanced', 'balanced_subsample'],
             'n_estimators': [i for i in range(10, 101, 10)]}

SVM_params = {'random_state': [42],
              'kernel': ['rbf'],
              'probability': [True],
              'class_weight': [None, 'balanced'],
              'C': [i * 0.1 for i in range(5, 51, 5)],
              'gamma': ['scale', 'auto', 1e-2, 5e-2, 1e-1, 5e-1]}

xgb_params = {'random_state': [42],
              'booster': ['gbtree'],
              'objective': ['binary:logistic'],
              'max_depth': [i for i in range(1, 10, 2)],
              'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
              'n_estimators': [i for i in range(10, 101, 10)]}

dict_metrics = {'AUC': 'roc_auc',
                'ACC': make_scorer(accuracy_score),
                'BA': make_scorer(balanced_accuracy_score),
                'SE': make_scorer(recall_score),
                'SP': make_scorer(recall_score, pos_label=0),
                'F1': make_scorer(f1_score),
                'MCC': make_scorer(matthews_corrcoef)}

model1 = LGBMClassifier()
model2 = RandomForestClassifier()
model3 = SVC()
model4 = XGBClassifier()

def grid_cv_lgb(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=model1, param_grid=lgb_params, cv=skf, scoring=dict_metrics, refit='AUC',
                        n_jobs=-1)
    grid.fit(X, y)
    return grid

def grid_cv_RF(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=model2, param_grid=RF_params, cv=skf, scoring=dict_metrics, refit='AUC',
                        n_jobs=-1)
    grid.fit(X, y)
    return grid

def grid_cv_SVM(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=model3, param_grid=SVM_params, cv=skf, scoring=dict_metrics, refit='AUC',
                        n_jobs=-1)
    grid.fit(X, y)
    return grid

def grid_cv_xgb(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=model4, param_grid=xgb_params, cv=skf, scoring=dict_metrics, refit='AUC',
                        n_jobs=-1)
    grid.fit(X, y)
    return grid

def grid_cvs(X_train, y_train):
    if model == 'lgb':
        grid_lgb = grid_cv_lgb(X_train, y_train)
        df_results_lgb = pd.DataFrame(grid_lgb.cv_results_)
        df_results_lgb.to_excel('./cv_results/Replicate_' + str(i) + '/lgb_rdkit-d+' + fp + '.xlsx', index=False)
        dump(grid_lgb.best_estimator_, './final_models/Replicate_' + str(i) + '/lgb_rdkit-d+' + fp + '.joblib')
        'cv_results/Replicate_' + str(i)

    elif model == 'RF':
        grid_RF = grid_cv_RF(X_train, y_train)
        df_results_RF = pd.DataFrame(grid_RF.cv_results_)
        df_results_RF.to_excel('./cv_results/Replicate_' + str(i) + '/RF_rdkit-d+' + fp + '.xlsx', index=False)
        dump(grid_RF.best_estimator_, './final_models/Replicate_' + str(i) + '/RF_rdkit-d+' + fp + '.joblib')

    elif model == 'SVM':
        grid_SVM = grid_cv_SVM(X_train, y_train)
        df_results_SVM = pd.DataFrame(grid_SVM.cv_results_)
        df_results_SVM.to_excel('./cv_results/Replicate_' + str(i) + '/SVM_rdkit-d+' + fp + '.xlsx', index=False)
        dump(grid_SVM.best_estimator_, './final_models/Replicate_' + str(i) + '/SVM_rdkit-d+' + fp + '.joblib')

    elif model == 'xgb':
        grid_xgb = grid_cv_xgb(X_train, y_train)
        df_results_xgb = pd.DataFrame(grid_xgb.cv_results_)
        df_results_xgb.to_excel('./cv_results/Replicate_' + str(i) + '/xgb_rdkit-d+' + fp + '.xlsx', index=False)
        dump(grid_xgb.best_estimator_, './final_models/Replicate_' + str(i) + '/xgb_rdkit-d+' + fp + '.joblib')

# 测试/外部验证
def test(X_test, y_test):
    model_to_test = load('./final_models/Replicate_' + str(i) + '/' + model + '_rdkit-d+' + fp + '.joblib')
    y_pred = model_to_test.predict(X_test)
    y_proba = model_to_test.predict_proba(X_test)[:, 1]

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
    return metrics


# 重复10次实验
i = 1  # 计数器
for random in (6,19,24,25,29,44,46,60,65,86):
    if not os.path.isdir('cv_results/Replicate_'+str(i)):
        os.makedirs('cv_results/Replicate_'+str(i))
    if not os.path.isdir('final_models/Replicate_'+str(i)):
        os.makedirs('final_models/Replicate_'+str(i))
    if not os.path.isdir('results_all'):
        os.makedirs('results_all')
    if not os.path.isdir('results_10to1'):
        os.makedirs('results_10to1')

    df_result_training = pd.DataFrame([])
    df_result_test = pd.DataFrame(columns=['AUC', 'ACC', 'BA', 'SE', 'SP', 'F1', 'MCC'])
    df_result_outer = pd.DataFrame(columns=['AUC', 'ACC', 'BA', 'SE', 'SP', 'F1', 'MCC'])
    models = ['lgb', 'RF', 'SVM', 'xgb']
    fps = ['AP2D', 'ECFP4', 'EState', 'FCFP4', 'MACCS']
    for model in models:
        for fp in fps:
            df = pd.read_csv("fingerprints/rdkit-d+" + fp + ".csv")
            X_train, X_test, y_train, y_test = split_rus(df)
            # 交叉验证及数据处理
            grid_cvs(X_train, y_train)
            df_cv = pd.read_excel('cv_results/Replicate_' + str(i) + '/' + model + '_rdkit-d+' + fp + '.xlsx')
            filtered_row = df_cv[df_cv["rank_test_AUC"] == 1].head(1)
            df_result_training = df_result_training._append(filtered_row, ignore_index=True)
            # 测试集及数据处理
            metrics_test = test(X_test, y_test)
            pattern = r'\d+\.\d+'  # 匹配浮点数
            numbers = [round(float(num), 3) for num in re.findall(pattern, str(metrics_test))]  # 四舍五入保留三位小数
            df_result_test = df_result_test._append(pd.Series(numbers, index=df_result_test.columns), ignore_index=True)

    # 交叉验证数据处理（续）
    df_training = pd.DataFrame(columns=['AUC', 'ACC', 'BA', 'SE', 'SP', 'F1', 'MCC'])
    metrics = ['AUC', 'ACC', 'BA', 'SE', 'SP', 'F1', 'MCC']
    for metric in metrics:
        df_training[metric] = df_result_training['mean_test_' + metric].round(3).map(str)
    df_training.to_excel('results_all/results_'+str(i)+'.xlsx', sheet_name='training', index=False)

    # 测试集数据处理（续）
    with pd.ExcelWriter('results_all/results_'+str(i)+'.xlsx', engine='openpyxl', mode='a') as writer:
        df_result_test.to_excel(writer, sheet_name='test', index=False)

    # 外部验证集及数据处理
    for model in models:
        for fp in fps:
            df = pd.read_csv("fingerprints_outer/rdkit-d+" + fp + ".csv")
            X_test, y_test = split_outer(df)
            metrics_outer = test(X_test, y_test)
            pattern = r'\d+\.\d+'  # 匹配浮点数
            numbers = [round(float(num), 3) for num in re.findall(pattern, str(metrics_outer))]  # 四舍五入保留三位小数
            df_result_outer = df_result_outer._append(pd.Series(numbers, index=df_result_outer.columns),
                                                      ignore_index=True)
    # 外部验证集数据处理（续）
    with pd.ExcelWriter('results_all/results_'+str(i)+'.xlsx', engine='openpyxl', mode='a') as writer:
        df_result_outer.to_excel(writer, sheet_name='outer', index=False)

    i = i +1

# 数据10合1
def calculate_mean_std(file_path):
    excel_files = [file for file in os.listdir(file_path) if file.endswith('.xlsx')]
    tasks = ['training', 'test', 'outer']
    for task in tasks:
        j = 1
        for file in excel_files:
            # 读取每个excel文件的数据
            if j == 1:
                df1 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 2:
                df2 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 3:
                df3 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 4:
                df4 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 5:
                df5 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 6:
                df6 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 7:
                df7 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 8:
                df8 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 9:
                df9 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            elif j == 10:
                df10 = pd.read_excel(os.path.join(file_path, file), sheet_name=task)
            j = j + 1

        df0 = (df1+df2+df3+df4+df5+df6+df7+df8+df9+df10)/10
        df_std = pow(((pow(df1-df0,2)+pow(df2-df0,2)+pow(df3-df0,2)+pow(df4-df0,2)+pow(df5-df0,2)+pow(df6-df0,2)+pow(df7-df0,2)+pow(df8-df0,2)+pow(df9-df0,2)+pow(df10-df0,2)))/10,0.5)

        # 定义model列和fp列
        df_modelandfp = pd.DataFrame(columns=['model', 'fp'])
        df_modelandfp['model'] = ['lgb'] * 5 + ['RF'] * 5 + ['SVM'] * 5 + ['xgb'] * 5
        df_modelandfp['fp'] = ['rdkit-d+AP2D', 'rdkit-d+ECFP4', 'rdkit-d+EState', 'rdkit-d+FCFP4', 'rdkit-d+MACCS'] * 4
        # 定义四舍五入的df
        new_df = pd.DataFrame(columns=['AUC', 'ACC', 'BA', 'SE', 'SP', 'F1', 'MCC'])
        new_df0 = pd.DataFrame(columns=['AUC', 'ACC', 'BA', 'SE', 'SP', 'F1', 'MCC'])
        new_df_std = pd.DataFrame(columns=['AUC', 'ACC', 'BA', 'SE', 'SP', 'F1', 'MCC'])
        # 四舍五入
        metrics = ['AUC', 'ACC', 'BA', 'SE', 'SP', 'F1', 'MCC']
        for metric in metrics:
            new_df[metric] = df0[metric].round(3).map(str) + '±' + df_std[metric].round(3).map(str)
            new_df0[metric] = df0[metric].round(3).map(str)
            new_df_std[metric] = df_std[metric].round(3).map(str)
        # 加上model列和fp列
        df_task = pd.concat([df_modelandfp, new_df], axis=1)
        df0_end = pd.concat([df_modelandfp, new_df0], axis=1)
        df_std_end = pd.concat([df_modelandfp, new_df_std], axis=1)
        # 导出至sheet
        if task == 'training':
            df_task.to_excel('results_10to1/results_mean±std.xlsx', sheet_name=task, index=False)
            df0_end.to_excel('results_10to1/results_mean.xlsx', sheet_name=task, index=False)
            df_std_end.to_excel('results_10to1/results_std.xlsx', sheet_name=task, index=False)
        else:
            with pd.ExcelWriter('results_10to1/results_mean±std.xlsx', engine='openpyxl', mode='a') as writer1:
                df_task.to_excel(writer1, sheet_name=task, index=False)
            with pd.ExcelWriter('results_10to1/results_mean.xlsx', engine='openpyxl', mode='a') as writer2:
                df0_end.to_excel(writer2, sheet_name=task, index=False)
            with pd.ExcelWriter('results_10to1/results_std.xlsx', engine='openpyxl', mode='a') as writer3:
                df_std_end.to_excel(writer3, sheet_name=task, index=False)
calculate_mean_std('results_all')