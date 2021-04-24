import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, precision_recall_curve, recall_score, f1_score, auc, roc_auc_score, plot_roc_curve
from sklearn.model_selection import KFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class MLMODEL_dr_noshow():
    def __init__(self, df_csv):
        self.df = pd.read_csv(df_csv)

    def dayDifference(self, fdate, ldate):
        delta = ldate - fdate
        return(delta.days)

    def simplify_appt_sch(self):
        self.df['scheduleDay_parse'] = self.df.ScheduledDay.apply(lambda x: x.split('T')[0].split('-'))
        self.df.scheduleDay_parse = self.df.scheduleDay_parse.apply(lambda x: date(int(x[0]), int(x[1]), int(x[2])))
        self.df['apptDay_parse'] = self.df.AppointmentDay.apply(lambda x: x.split('T')[0].split('-'))
        self.df.apptDay_parse = self.df.apptDay_parse.apply(lambda x: date(int(x[0]), int(x[1]), int(x[2])))
        self.df['Schedule_to_appt_days'] = self.df.apply(lambda x: self.dayDifference(x.scheduleDay_parse, x.apptDay_parse), axis = 1)
        self.df.drop(["scheduleDay_parse", "apptDay_parse","ScheduledDay", "AppointmentDay"], axis = 1, inplace = True)
        self.df.drop(self.df[self.df['Schedule_to_appt_days'] < 0].index, inplace = True)
        self.df.dropna(inplace=True)
        print(self.df.sample(3))

    def simplify_appt_days(self):
        bins = (-1, 0, 7, 14, 21, 28, 80)
        group_names = ["Within_1_day", "Within_1_week", "Within_1_2_weeks", "Within_2_3_weeks", "Within_3_4_weeks", "Greater_than_4_weeks"]
        categories = pd.cut(self.df.Schedule_to_appt_days, bins, labels=group_names)
        self.df.Schedule_to_appt_days = categories
        print(self.df.sample(3))

    def simplify_no_show(self):
        self.df['No-show']=pd.Series(np.where(self.df['No-show'].values == "Yes", 1, 0), self.df.index)
        print(self.df.sample(3))

    def simplify_handcap(self):
        self.df.Handcap=pd.Series(np.where(self.df.Handcap.values > 1, 1, self.df.Handcap.values), self.df.index)
        print(self.df.sample(3))

    def simplify_ages(self):
        bins = (-1, 0, 12, 18, 25, 60, 120)
        group_names = ['Unknown', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
        categories = pd.cut(self.df.Age, bins, labels=group_names)
        self.df.Age = categories
        print(self.df.sample(3))

    def drop_features(self):
        self.df.drop(['PatientId', 'AppointmentID'], axis= 1, inplace = True)
        print(self.df.sample(3))

    def plot_count_data(self, attribute):
        if attribute == "Neighbourhood":
            sns.countplot(y=attribute, hue="No-show", data=self.df)
            #plt.show()
        else:
            sns.countplot(x=attribute, hue="No-show", data=self.df)
        plt.savefig(f"ns_v_{attribute}.png")
        plt.show()
        plt.clf()

    def plot_all_data(self):
        sns.countplot(x="No-show", data=self.df)
        self.plot_count_data("Gender")
        self.plot_count_data("Scholarship")
        self.plot_count_data("Hipertension")
        self.plot_count_data("Diabetes")
        self.plot_count_data("Alcoholism")
        self.plot_count_data("Handcap")
        self.plot_count_data("SMS_received")
        self.plot_count_data("Age")
        self.plot_count_data("Neighbourhood")
        self.plot_count_data("Schedule_to_appt_days")

    def encode_features(self):
        features = ['Gender', 'Age', 'Neighbourhood','Schedule_to_appt_days']

        for feature in features:
            le = preprocessing.LabelEncoder()
            le = le.fit(self.df[feature])
            self.df[feature] = le.transform(self.df[feature])
        
        print(self.df.sample(3))

    def normalize_features(self):
        scaler = preprocessing.StandardScaler().fit(self.df)
        df_scaled = scaler.transform(self.df)
        self.df = pd.DataFrame(df_scaled, columns = self.df.columns, dtype= 'int64')

    def create_train_and_test_df(self, numtest = 0.2, rand_state = 42):
        self.X_all = self.df.drop(['PatientId', 'AppointmentID','No-show'], axis = 1)
        self.y_all = self.df['No-show']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_all, self.y_all, test_size=numtest, random_state=rand_state)


    def run_kfold(self, clf, folds = 10, name ="potato"):
        kf=KFold(n_splits=folds)
        outcomes = []
        precision_scores = []
        recall_scores = []
        f_scores = []
        aucs = []
        fig, ax = plt.subplots()
        table = {}
        fold = 0
        for train_index, test_index in kf.split(self.X_all):
            fold += 1
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.X_all.values[train_index], self.X_all.values[test_index]
            y_train, y_test = self.y_all.values[train_index], self.y_all.values[test_index]
            
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            predict_prob = clf.predict_proba(X_test)
            accuracy = accuracy_score(y_test, predictions)
            prec_score = precision_score(y_test, predictions, pos_label = 1)
            r_score = recall_score(y_test, predictions, pos_label = 1)
            f_score = f1_score(y_test, predictions, pos_label = 1)
            auc_score = roc_auc_score(y_test, predict_prob[:,1])
            clf_disp = plot_roc_curve(clf, X_test, y_test, name=f"Fold {fold}", ax=ax)
            
            #plt.savefig(f"{name}_fold{fold}_test.png")
            outcomes.append(accuracy)
            precision_scores.append(prec_score)
            recall_scores.append(r_score)
            f_scores.append(f_score)
            aucs.append(auc_score)
            print(f"Fold {fold} accuracy: {accuracy}")     
        mean_outcome = np.mean(outcomes)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_f_scores = np.mean(f_scores)
        mean_aucs = np.mean(aucs)
        table["Folds"] = np.arange(0, 10)
        table["Accuracy"] = outcomes
        table["Precision"] = precision_scores
        table["Recall"] = recall_scores
        table["F Score"] = f_scores
        table["AUC"] = aucs
        pd_table = pd.DataFrame(table)
        print(pd_table)
        pd_table.to_csv(f"{name}_table.csv")
        print(tabulate([["Mean", mean_outcome, mean_precision, mean_recall, mean_f_scores, mean_aucs]]))
        print("Mean Accuracy: {0}".format(mean_outcome))
        plt.savefig(f"{name}_test.png")


        
dr_noshow = MLMODEL_dr_noshow('dr_appt/KaggleV2-May-2016.csv')
dr_noshow.simplify_appt_sch()
dr_noshow.simplify_appt_days()
dr_noshow.simplify_handcap()
dr_noshow.simplify_ages()
dr_noshow.simplify_no_show()
print(dr_noshow.df.describe())
dr_noshow.encode_features()
#dr_noshow.normalize_features()
dr_noshow.create_train_and_test_df()

acc_scorer = make_scorer(accuracy_score)
clf_LR = LogisticRegression()
clf_LR.max_iter = 200
"""
clf.fit(dr_noshow.X_train, dr_noshow.y_train)
predictions = clf.predict(dr_noshow.X_test)
print(accuracy_score(dr_noshow.y_test, predictions))
"""
#dr_noshow.run_kfold(clf_LR, name="LogisticRegression")
clf_NB = CategoricalNB(min_categories=7)
#dr_noshow.run_kfold(clf_NB, name="Naive_Bayes")

clf_DT = DecisionTreeClassifier()
#dr_noshow.run_kfold(clf_DT, name="Decision_Tree")

clf_XGB = XGBClassifier()
dr_noshow.run_kfold(clf_XGB, name = "XGBoost")

#dr_noshow.plot_count_data("Schedule_to_appt_days")
#dr_noshow.plot_all_data()

#print(dr_noshow.df.describe())
#plt.show()        