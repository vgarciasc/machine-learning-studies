import pandas as pd
import numpy as np
import utils as u

class NaiveBayes:
    def __init__(self, df, cat_cols, num_cols, label_col):
        self.label_col = label_col
        self.possible_labels = df[label_col].unique()
        self.where_Y_is_label = {label:df.loc[df[label_col] == label] \
            for label in self.possible_labels}
        self.df_length = len(df)

        # dynamically preparing features
        self.cat_feats = {f:[df[f].unique()] for f in cat_cols}
        self.num_feats = {l:{f: \
                {'mean': self.where_Y_is_label[l].loc[:,f].mean(), \
                'var': self.where_Y_is_label[l].loc[:,f].var()} for f in num_cols}
            for l in self.possible_labels}

    def calc_posterior(self, sample, label_y):
        # rows with label 'y'
        where_Y_is_y = self.where_Y_is_label[label_y]
        label_y_count = len(where_Y_is_y)

        # P(Y=y)
        prior = len(where_Y_is_y) / self.df_length

        # P(x_i=?|Y=y) * P(x_{i+1}=?|Y=y) * ...
        where_X_cat = [where_Y_is_y.loc[where_Y_is_y[x] == sample[x]] \
            for x in self.cat_feats]
        where_X_cat = [len(x) for x in where_X_cat]
        where_X_cat = np.array(where_X_cat)
        likelihood_cat = np.prod(where_X_cat / label_y_count)

        # same as above, but with continuous variables
        where_X_num = [u.gaussian_prob( \
                self.num_feats[label_y][x]['mean'], \
                self.num_feats[label_y][x]['var'], \
                sample[x]) \
            for x in self.num_feats[label_y]]
        where_X_num = np.array(where_X_num)
        likelihood_num = np.prod(where_X_num)
            
        return prior * likelihood_cat * likelihood_num

    def predict(self, sample):
        results = {}
        for label in self.possible_labels:
            results[label] = self.calc_posterior(sample, label)
        return results

if __name__ == "__main__":
    FILENAME = "data/titanic/train.csv"
    K_FOLDS = 10

    dataframe = pd.read_csv(FILENAME)
    t_precision, t_recall, t_accuracy = [], [], []

    # K-folding
    for w in range(0, K_FOLDS):
        train_data, validate_data = u.pd_kfold_iteration_i(dataframe, w, K_FOLDS)
        nb = NaiveBayes(df=train_data, 
                        cat_cols=['Sex'],
                        num_cols=['Parch', 'SibSp', 'Fare'],
                        label_col='Survived')

        # Predicting
        t_pos, t_neg, f_pos, f_neg = 0, 0, 0, 0
        for i in range(0, len(validate_data)):
            sample = validate_data.iloc[i]
            results = nb.predict(sample)

            predicted_label = max(results, key=results.get)

            t_pos += 1 if predicted_label == 1 and sample[nb.label_col] == 1 else 0
            f_pos += 1 if predicted_label == 1 and sample[nb.label_col] != 1 else 0
            t_neg += 1 if predicted_label == 0 and sample[nb.label_col] == 0 else 0
            f_neg += 1 if predicted_label == 0 and sample[nb.label_col] != 0 else 0

        precision = (t_pos) / (t_pos + f_pos + 1e-6)
        recall = (t_pos) / (t_pos + f_neg + 1e-6)
        accuracy = (t_pos + t_neg) / len(validate_data)
        print("#", w, ": precision: ", "%.2f" % precision, ", recall: ", "%.2f" % recall)

        t_precision += [precision]
        t_recall += [recall]
        t_accuracy += [accuracy]
    
    mean_precision = np.mean(t_precision)
    mean_recall = np.mean(t_recall)
    mean_accuracy = np.mean(t_accuracy)
    f1_score = u.f1_score(mean_precision, mean_recall)

    print("---")
    print("f1 score: ", "%.2f" % f1_score)
    print("~ t_precision: ", "%.2f" % mean_precision, ", t_recall: ", "%.2f" % mean_recall)
    print("~ t_accuracy: ", "%.2f" % mean_accuracy)

