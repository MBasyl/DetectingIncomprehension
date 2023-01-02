import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import scikitplot.metrics as skp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
plt.rc("font", size=14)


def get_rfe():
    log_reg = LogisticRegression()
    rfe = RFE(estimator = log_reg, n_features_to_select= 10)
    rfe_sel = rfe.fit(X_train, y_train.values.ravel())
    f = rfe_sel.get_support(1) #the most important features
    features_selected = df.columns[f].tolist() 
    print(features_selected) # final features`
    return features_selected

def search_grid():
    log_reg = LogisticRegression()
    # Setup grid search
    print("\nsearching grid")
    lg_param_grid = {'solver': ['newton-cg', 'saga', 'lbfgs', 'liblinear'],
                    'C': [0.001, 0.01, 0.1, 1]}
    lg_grid = GridSearchCV(cv=5, verbose=0, estimator = log_reg, param_grid = lg_param_grid)
    lg_grid.fit(X_train_final, y_train)
    print("found:", lg_grid.best_estimator_ )


def processing(X, y, var_names):
    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #scale
    scaler = StandardScaler()
    scaler.fit(X_train[var_names])
    X_train[var_names] = scaler.transform(X_train[var_names])
    X_test[var_names]= scaler.transform(X_test[var_names])
    #transform nonlinear
    non_lin = ['lex_sim', 'lc_sim', 'mtld', 'nwords']
    X_train[non_lin] = np.cbrt(X_train[non_lin])
    X_test[non_lin] = np.cbrt(X_test[non_lin])
    
    return X_train,X_test,y_train,y_test


def logistic_regression(X_train_final, X_test_final, y_train, y_test):
    # put in selected hyperP
    log_reg = LogisticRegression(C=0.01, solver='newton-cg')
    #get LOGIT summary
    logit_model=sm.Logit(y_train,X_train_final)
    result=logit_model.fit()
    smry = result.summary2()
    file = open(f'final_results/{MODEL}_summary.txt', 'w')
    file.write(f'{MODEL}\n\nLogit Model Summary\n\n{smry}\n')
    file.close()

    # Use L regressor to predict on test set
    log_reg.fit(X_train_final, y_train) ###
    y_pred = log_reg.predict(X_test_final)
    y_probas = log_reg.predict_proba(X_test_final)
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test_final)[:,1])

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
        log_reg.score(X_test_final, y_test)))
    cr = classification_report(y_test, y_pred)
    cm = np.array2string(confusion_matrix(y_test, y_pred))
    f = open(f'final_results/{MODEL}_report.txt', 'w')
    f.write('{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(MODEL, cr, cm))
    f.close()

    # set data with subplots and plot
    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(121) #.set_title('121')
    skp.plot_ks_statistic(y_test, y_probas, title='KS Statistic Plot', 
    ax=ax1, figsize=None, title_fontsize='large', text_fontsize='medium')
    ax2 = fig.add_subplot(122)
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'final_results/SK_ROC_{MODEL}.png')
    plt.show()



if __name__ =="__main__":

    # select data
    df_raw = pd.read_csv("scoref26_2sim_sentpairs_norep.csv")
    df = df_raw.iloc[:,0:28]
    var_names = df.iloc[:,1:].columns.tolist()
    y=df['label']
    X= df.iloc[:,1:]

    X_train,X_test,y_train,y_test= processing(X, y, var_names)

    #Get Recursive Feature Elimination
    features_selected = get_rfe()

    # put in selection
    X_train_final = X_train[features_selected]
    X_test_final = X_test[features_selected]
    MODEL = 'sel_rfe_max10'

    #search grid for best hyperparameters
    #search_grid()
    #exit(0)

    logistic_regression(X_train_final, X_test_final, y_train, y_test)


