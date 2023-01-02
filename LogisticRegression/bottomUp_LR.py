import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt 
import scikitplot.metrics as skp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, accuracy_score
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


def search_grid():
    log_reg = LogisticRegression()
    # Setup grid search
    print("\nsearching grid")
    lg_param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                    'C': [0.001, 0.01, 0.1, 1],
                    'max_iter':[100, 150, 300]}
    lg_grid = GridSearchCV(cv=5, verbose=0, estimator = log_reg, param_grid = lg_param_grid)
    lg_grid.fit(X_train_final, y_train)
    print("found:", lg_grid.best_estimator_ )

def plot_feature_importance():
  feature_importances = pd.DataFrame(
  {"variables":X_train_final.columns.values,
  "Importance":log_reg.coef_[0]})
  feature_importances.sort_values(by = "Importance",inplace = True,ascending=False)
  fig, ax = plt.subplots(1,1, figsize = (20,14))
  feature_importances.plot.barh(x = "variables",y = "Importance", ax = ax)
  ax.set_title("Logisitc Regression Feature Importance",fontsize = 15,pad = 15)
  ax.tick_params(axis = "both",which = "major",labelsize = 15)
  plt.savefig(f'featImport{MODEL}.png')
  print("Saved plot!")


def processing(X, y, var_names):
    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #scale
    scaler = StandardScaler()
    scaler.fit(X_train[var_names])
    X_train[var_names] = scaler.transform(X_train[var_names])
    X_test[var_names]= scaler.transform(X_test[var_names])
    #transform nonlinear
    non_lin = ['mtld', 'nwords','lex_sim', 'lc_sim'] 
    X_train[non_lin] = np.cbrt(X_train[non_lin])
    X_test[non_lin] = np.cbrt(X_test[non_lin])
    return X_train,y_train,X_test,y_test


def logistic_regression(X_train_final, X_test_final, y_train, y_test)
    log_reg = LogisticRegression(C=0.1, solver='newton-cg') #max_iter=300
    #get LOGIT summary
    logit_model=sm.Logit(y_train,X_train_final)
    result=logit_model.fit()
    smry = result.summary2()
    file = open(f'{MODEL}_summary.txt', 'w')
    file.write(f'{MODEL}\n\nLogit Model Summary\n\n{smry}\n')
    file.close()

    # Use regressor to predict on test set
    log_reg.fit(X_train_final, y_train) ###
    y_pred = log_reg.predict(X_test_final)
    y_probas = log_reg.predict_proba(X_test_final)[:, 1]

    plot_feature_importance()

    logit_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test_final)[:,1])
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
        log_reg.score(X_test_final, y_test)))
    cr = classification_report(y_test, y_pred)
    cm = np.array2string(confusion_matrix(y_test, y_pred))

    f = open(f'{MODEL}_report.txt', 'w')
    f.write('{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\n'.format(MODEL, cr, cm))
    f.write('Log loss = {:.5f}\n'.format(log_loss(y_test, y_probas)))
    f.write('AUC = {:.5f}\n'.format(roc_auc_score(y_test, y_probas)))
    f.write('Average Precision = {:.5f}\n\n'.format(average_precision_score(y_test, y_probas)))
    f.write('\nUsing 0.5 as threshold:\n')
    f.write('Accuracy = {:.5f}\n'.format(accuracy_score(y_test, y_pred)))
    f.write('Precision = {:.5f}\n'.format(precision_score(y_test, y_pred)))
    f.write('Recall = {:.5f}\n'.format(recall_score(y_test, y_pred)))
    f.write('F1 score = {:.5f}\n'.format(f1_score(y_test, y_pred)))
    f.close()

    y_probs = log_reg.predict_proba(X_test_final)
    # set data with subplots and plot
    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(121) #.set_title('121')
    skp.plot_ks_statistic(y_test, y_probs, title='KS Statistic Plot', 
    ax=ax1, figsize=None, title_fontsize='large', text_fontsize='medium')
    ax2 = fig.add_subplot(122)
    plt.plot(fpr, tpr, label='area = %0.2f' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig(f'ROC{MODEL}.png')
    

if __name__ =="__main__":
    
    # select data
    df_raw = pd.read_csv('scoreF26_sentpairsB_norep.txt')
    df = df_raw.iloc[:,0:28]
    MODEL = 'evaluation_all'
    var_names = df.iloc[:,1:].columns.tolist()
    y=df['label']
    X= df.iloc[:,1:]

    X_train,y_train,X_test,y_test = processing(X, y, var_names)

    #select features
    #feature_selection = []

    X_train_final = X_train#[feature_selection]
    X_test_final = X_test#[feature_selection]

    #search grid for best hyperparameters
    #search_grid()
    #exit(0)

    logistic_regression(X_train_final, X_test_final, y_train, y_test)


