import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection, ensemble
from sklearn import svm
from sklearn.base import clone
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
import sklearn.metrics as m
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils import shuffle

seed = 42
np.random.seed(seed)
max_depth = 1
C=1
tol=1e-3
min_samples_leaf=2
min_samples_split=2
n_estimators=100

l1_logit_model = {
		  "Logistic Regression" : linear_model.LogisticRegression(
			  C=C,
			  penalty='l1',
			  solver="liblinear",
			  tol=tol,
			  random_state=seed)
		 }
rf_model = {
		  "Random Forest" : ensemble.RandomForestClassifier(
			  criterion='gini',
			  max_depth=max_depth,
			  max_features='log2',
			  min_samples_leaf=min_samples_leaf,
			  min_samples_split=min_samples_split,
			  n_estimators=n_estimators,
			  oob_score=True,
			  n_jobs=1,
			  random_state=seed)}                
nonlinear_models = {
		  "Random Forest" : ensemble.RandomForestClassifier(
			  criterion='gini',
			  max_depth=max_depth,
			  max_features='log2',
			  min_samples_leaf=min_samples_leaf,
			  min_samples_split=min_samples_split,
			  n_estimators=n_estimators,
			  oob_score=True,
			  n_jobs=1,
			  random_state=seed),
		  "Gradient Boosting Classifier" : ensemble.GradientBoostingClassifier(
			  n_estimators=n_estimators,
			  learning_rate=0.1,
			  max_depth=max_depth,
			  max_features='log2',
			  min_samples_leaf=min_samples_leaf,
			  min_samples_split=min_samples_split,
			  random_state=seed)
		 }
models = {
		  "Logistic Regression" : linear_model.LogisticRegression(
			  C=C,
			  penalty='l1',
			  solver="liblinear",
			  tol=tol,
			  random_state=seed),
		  "Random Forest" : ensemble.RandomForestClassifier(
			  criterion='gini',
			  max_depth=max_depth,
			  max_features='log2',
			  min_samples_leaf=min_samples_leaf,
			  min_samples_split=min_samples_split,
			  n_estimators=n_estimators,
			  oob_score=True,
			  n_jobs=1,
			  random_state=seed),
		  "Support Vector Machine" : svm.SVC(
			  C=C,
			  kernel="linear",
			  random_state=seed,
			  probability=True,
			  tol=tol),
		  "Gradient Boosting Classifier" : ensemble.GradientBoostingClassifier(
			  n_estimators=n_estimators,
			  learning_rate=0.1,
			  max_depth=max_depth,
			  max_features='log2',
			  min_samples_leaf=min_samples_leaf,
			  min_samples_split=min_samples_split,
			  random_state=seed)
		 }

def permute(Y,seed=42):
	"""
	shuffle sample values
	Parameters:
	----------
	Y : pandas series
		Index of samples and values are their class labels
	seed : int
		Random seed for shuffling
	Returns:
	------
	arr_shuffle: pandas series
		A shuffled Y
	"""
	arr = shuffle(Y.values,random_state=seed)
	arr_shuffle = (pd.Series(arr.reshape(1,-1)[0],index=Y.index))
	return arr_shuffle

def train_test_val_top_fold_01_within(X,Y,models,metrics=['roc_auc'],cv_split=10,seed=42,test_size=0.15,return_train_score=True,n_jobs=1,retrained_models=False,patient_level_predictions=False,return_estimator=True):
	# make sure given metrics are in list and not one metric given as a string
	if type(metrics)!=list:
		metrics = [metrics]
	# 1/ train and test split
	X = X.loc[Y.index]
	X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                            test_size=test_size,
                                                            random_state=seed,
                                                            shuffle=True)
	X_train = X_train.apply(lambda x : (x - min(x))/(max(x) - min(x)),axis=0)
	X_test = X_test.apply(lambda x : (x - min(x))/(max(x) - min(x)),axis=0)
	X_train[X_train.isna()]=0
	X_test[X_test.isna()]=0
	#define K fold splitter
	cv = StratifiedKFold(n_splits=cv_split,random_state=seed,shuffle=True)
	#Instantiate lists to collect prediction and model results
	dfs = []
	model_retrained_fits = {}
	model_confs = []
	#iterate through model dictionary
	for name,mod in models.items():
		# /2 generate model parameters and fold scores with cv splitter
		fit = cross_validate(clone(mod),X_train,y_train.values.reshape(1,-1)[0],cv=cv,scoring=metrics,
								n_jobs=n_jobs,return_train_score=return_train_score,
								return_estimator=return_estimator)
		tmp = pd.DataFrame({'fold' : range(cv_split),
                                    'model' : name},
                                   index=range(cv_split))
		#populate scores in dataframe
		cols = [k for k in fit.keys() if (k.find('test')+k.find('train'))==-1]
		for col in cols:
			tmp[col] = fit[col]
		# /3 Identify best performing model
		top_fold = np.where(fit['test_roc_auc']==fit['test_roc_auc'].max())[0][0]
		keys = [x for x in  fit.keys()]
		vals = [fit[x][top_fold] for x in keys]
		top_model_key_vals = {}
		for i in range(len(vals)):
			top_model_key_vals[keys[i]] = vals[i]
		#4/ train models on training set 
		# also get sample level predictions
		f = top_model_key_vals['estimator']
		fitted = clone(f).fit(X_train,y_train.values.reshape(1,-1)[0])
		conf = pd.DataFrame({'y_true' : y_test.values.reshape(1,-1)[0],
                                     'y_pred' : fitted.predict(X_test),
                                     'y_proba' : fitted.predict_proba(X_test)[:,1],
                                     'bootstrap' : np.repeat(seed,len(y_test.index)),
                                     'model' : np.repeat(name,len(y_test.index))},
                                    index=y_test.index)
		model_confs.append(conf)
		#do prediction for each metric
		for metric in metrics:
			tmp['validation_'+metric] = m.SCORERS[metric](fitted,X_test,y_test)
		model_retrained_fits[name] = fitted
		dfs.append(tmp.query('fold==@top_fold').drop('fold',1))
	return pd.concat(dfs,sort=True).reset_index(drop=True), model_retrained_fits, pd.concat(model_confs,sort=True)

def permuted_train_test_val_top_fold_01_within(X,Y,models,metrics=['roc_auc'],cv_split=10,seed=42,test_size=0.15,return_train_score=True,n_jobs=1,retrained_models=False,patient_level_predictions=False,return_estimator=True):
	X = X.loc[Y.index]
	Y_shuffle = permute(Y,seed=seed)
	X_shuffle = X.loc[Y_shuffle.index]
	# make sure given metrics are in list and not one metric given as a string
	if type(metrics)!=list:
		metrics = [metrics]
	# 1/ train and test split
	X_train, X_test, y_train, y_test = train_test_split(X_shuffle,Y_shuffle,
                                                            test_size=test_size,
                                                            random_state=seed,
                                                            shuffle=True)
	X_train = X_train.apply(lambda x : (x - min(x))/(max(x) - min(x)),axis=0)
	X_test = X_test.apply(lambda x : (x - min(x))/(max(x) - min(x)),axis=0)
	X_train[X_train.isna()]=0
	X_test[X_test.isna()]=0
	#define K fold splitter
	cv = StratifiedKFold(n_splits=cv_split,random_state=seed,shuffle=True)
	#Instantiate lists to collect prediction and model results
	dfs = []
	model_retrained_fits = {}
	model_confs = []
	#iterate through model dictionary
	for name,mod in models.items():
		# /2 generate model parameters and fold scores with cv splitter
		fit = cross_validate(clone(mod),X_train,y_train.values.reshape(1,-1)[0],cv=cv,scoring=metrics,
								n_jobs=n_jobs,return_train_score=return_train_score,
								return_estimator=return_estimator)
		tmp = pd.DataFrame({'fold' : range(cv_split),
                                    'model' : name},
                                   index=range(cv_split))
		#populate scores in dataframe
		cols = [k for k in fit.keys() if (k.find('test')+k.find('train'))==-1]
		for col in cols:
			tmp[col] = fit[col]
		# /3 Identify best performing model
		top_fold = np.where(fit['test_roc_auc']==fit['test_roc_auc'].max())[0][0]
		keys = [x for x in  fit.keys()]
		vals = [fit[x][top_fold] for x in keys]
		top_model_key_vals = {}
		for i in range(len(vals)):
			top_model_key_vals[keys[i]] = vals[i]
		#4/ train models on training set 
		# also get sample level predictions
		f = top_model_key_vals['estimator']
		fitted = clone(f).fit(X_train,y_train.values.reshape(1,-1)[0])
		conf = pd.DataFrame({'y_true' : y_test.values.reshape(1,-1)[0],
                                     'y_pred' : fitted.predict(X_test),
                                     'y_proba' : fitted.predict_proba(X_test)[:,1],
                                     'bootstrap' : np.repeat(seed,len(y_test.index)),
                                     'model' : np.repeat(name,len(y_test.index))},
                                    index=y_test.index)
		model_confs.append(conf)
		#do prediction for each metric
		for metric in metrics:
			tmp['validation_'+metric] = m.SCORERS[metric](fitted,X_test,y_test)
		model_retrained_fits[name] = fitted
		dfs.append(tmp.query('fold==@top_fold').drop('fold',1))
	return pd.concat(dfs,sort=True).reset_index(drop=True), model_retrained_fits, pd.concat(model_confs,sort=True)

def bootstrap_of_fcn(func=None,params={},n_jobs=4,nboot=2):
	if func==None:
		return "Need fcn to bootstrap"
	parallel = Parallel(n_jobs=n_jobs)
	return parallel(
		delayed(func)(
			seed=k,**params)
		for k in range(nboot))

def get_performance(lst):
    perf = (pd.
            concat(lst,keys=range(len(lst))).
            reset_index(level=1,drop=True).
            rename_axis('bootstrap').
            reset_index()
           )
    return perf

def model_feature_importances(boot_mods):
    dfs = []
    X = params['X'].copy()
    X.loc[:,'Intercept'] = 0
    for i in range(len(boot_mods)):
        for j in boot_mods[i].keys():
            mod = boot_mods[i][j]
            coef = []
            try:
                coef.extend([i for i in mod.feature_importances_])
            except:
                coef.extend([i for i in mod.coef_[0]])
            coef.extend(mod.intercept_)
            fs = []
            fs.extend(X.columns.values)
            df = pd.DataFrame({
                'feature' : fs,
                'importance' : coef,
                'model' : j,
                'bootstrap' : i
            })
            dfs.append(df)
    return pd.concat(dfs,sort=True)

def patient_predictions(lst,n=50,scorers={ 'roc_auc' : metrics.roc_auc_score}):
        dat = \
        (pd.
         concat(
             lst
         )
        )
        for score in scorers.keys():
            vals = []
            for b in range(n):
                x = (dat.
                     sample(n=dat.shape[0],replace=True,random_state=b)
                    )
                vals.append(scorers[score](x.y_true,x.y_proba))
            score_vals[score] = vals
        return dat, score_vals


