import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import linear_model, ensemble
from sklearn import svm
from sklearn.base import clone
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.utils import shuffle

class mccv(object):
    """
    Monte carlo cross validation results

    Dataframes, variables, and functions facilitating the processing, analysis, and results from the MCCV routine.

    Parameters
    ----------
    cv_split: int
            Number of splits for cross validation.

    test_size: float
                    Number between 0 and 1 for the proportinal size of the dataset for testing the trained machine learning model.

    n_jobs: int
    Number of cores to be used for parallel processing.

    seed: int
            Number to set random number seed generators.

    Returns
    -------

    Notes
    -----

    Examples
    --------


    """

    def __init__(self, num_bootstraps=1, n_jobs=2):

        #####SET ARGUMENTS####
        self.X = (None,)
        self.Y = (None,)
        self.all_models = {}
        self.model_names = ["Logistic Regression"]
        self.num_bootstraps = num_bootstraps
        self.cv_split = 10
        self.test_size = 0.15
        self.n_jobs = n_jobs
        self.metrics = ["roc_auc"]
        self.return_train_score = True
        self.return_estimator = True
        self.seed = 0

        #####INSTANTIATE MODELS AND PREPARE MODELS TO BE USED####
        self._set_models()
        if type(self.model_names) == str:
            self.model_names = [x for x in [self.model_names] if x in self.all_models]
        if type(self.model_names) == list:
            self.model_names = [x for x in self.model_names if x in self.all_models]
            if len(self.model_names) == 0:
                sys.exit(
                    'Please enter a prespecified model name(s) as a list: "Logistic Regression", "Random Forest", "Support Vector Machines", and/or "Gradient Boosting Machines"'
                )
        ######

        #####INSTANTIATE DATA OBJECT HOLDERS####
        self.mccv_data = None
        self.mccv_permuted_data = None
        ######

        ####Make sure given metrics are in list####
        ####and not one metric given as a string####
        if type(self.metrics) == str:
            self.metrics = [
                x for x in [self.metrics] if x in metrics.get_scorer_names()
            ]
        if type(self.metrics) == list:
            self.metrics = [x for x in self.metrics if x in metrics.get_scorer_names()]
            if len(self.metrics) == 0:
                sys.exit(
                    "Please enter a metric name from sklearn.metrics.get_scorer_names()"
                )
        if type(self.metrics) not in [str, list]:
            sys.exit(
                "Please enter a metric name from sklearn.metrics.get_scorer_names()"
            )

    def set_X(self, X):
        """
        Set predictor data.

        Set the data for the learning task (i.e. predictors)

        Parameters
        ----------
        X: dataframe
                A pandas dataframe indexed by the unique observations and columns as the predictors

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        self.X = X

    def set_Y(self, Y):
        """
        Set response data.

        Set the data for the learning task (i.e. response)

        Parameters
        ----------
        Y: dataframe
                A pandas dataframe indexed by the unique observations and column as the response

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        self.Y = Y

    def _set_models(self):
        """
        Set model dictionary.

        Set the machine learning models to  choose from

        Parameters
        ----------
        NULL

        Returns
        -------
        NULL

        Notes
        -----
        See sklearn guides for details on parameters:
        - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
        - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

        Examples
        --------
        NULL

        """

        self.all_models = {
            "Linear Regression" : linear_model.LinearRegression(
                n_jobs = self.n_jobs
            ),
            "Logistic Regression": linear_model.LogisticRegression(
                C=1000000,
                penalty="l2",
                solver="liblinear",
                tol=1e-3,
                random_state=self.seed,
            ),
            "Random Forest": ensemble.RandomForestClassifier(
                criterion="gini",
                max_depth=1,
                max_features="log2",
                min_samples_leaf=2,
                min_samples_split=2,
                n_estimators=100,
                oob_score=True,
                n_jobs=self.n_jobs,
                random_state=self.seed,
            ),
            "Support Vector Machine": svm.SVC(
                C=1, kernel="linear", random_state=self.seed, probability=True, tol=1e-3
            ),
            "Gradient Boosting Classifier": ensemble.GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=1,
                max_features="log2",
                min_samples_leaf=2,
                min_samples_split=2,
                random_state=self.seed,
            ),
        }

    def _get_models(self):
        """
        Get model dictionary.

        Get models based on model name(s)

        Parameters
        ----------
        NULL

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        dict_ = {}
        for m in self.model_names:
            dict_[m] = self.all_models[m]
        return dict_

    def mccv(self, seed):
        """
        Run MCCV routine.

        Monte Carlo Cross Validation routine


        Parameters
        ----------
        seed: int
                Random seed for random number generator

        Returns
        -------
        list
        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        # 1/ train and test split
        Y = self.Y
        X = self.X.loc[Y.index]
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=seed, shuffle=True
        )
        X_train = X_train.apply(lambda x: (x - min(x)) / (max(x) - min(x)), axis=0)
        X_test = X_test.apply(lambda x: (x - min(x)) / (max(x) - min(x)), axis=0)
        X_train[X_train.isna()] = 0
        X_test[X_test.isna()] = 0
        # define K fold splitter
        cv = StratifiedKFold(n_splits=self.cv_split, random_state=seed, shuffle=True)
        # Instantiate lists to collect prediction and model results
        dfs = []
        model_retrained_fits = {}
        model_confs = []
        # iterate through model dictionary
        for name, mod in self._get_models().items():
            # /2 generate model parameters and fold scores with cv splitter
            fit = cross_validate(
                clone(mod),
                X_train,
                y_train.values.reshape(1, -1)[0],
                cv=cv,
                scoring=self.metrics,
                n_jobs=1,
                return_train_score=self.return_train_score,
                return_estimator=self.return_estimator,
            )
            tmp = pd.DataFrame(
                {"fold": range(self.cv_split), "model": name},
                index=range(self.cv_split),
            )
            # populate scores in dataframe
            cols = [k for k in fit.keys() if (k.find("test") + k.find("train")) == -1]
            for col in cols:
                tmp[col] = fit[col]
            # /3 Identify best performing model
            top_fold = np.where(fit["test_roc_auc"] == fit["test_roc_auc"].max())[0][0]
            keys = [x for x in fit.keys()]
            vals = [fit[x][top_fold] for x in keys]
            top_model_key_vals = {}
            for i in range(len(vals)):
                top_model_key_vals[keys[i]] = vals[i]
            # 4/ train models on training set
            # also get sample level predictions
            f = top_model_key_vals["estimator"]
            fitted = clone(f).fit(X_train, y_train.values.reshape(1, -1)[0])
            conf = pd.DataFrame(
                {
                    "y_true": y_test.values.reshape(1, -1)[0],
                    "y_pred": fitted.predict(X_test),
                    "y_proba": fitted.predict_proba(X_test)[:, 1],
                    "bootstrap": np.repeat(seed, len(y_test.index)),
                    "model": np.repeat(name, len(y_test.index)),
                },
                index=y_test.index,
            )
            model_confs.append(conf)
            # do prediction for each metric
            for metric in self.metrics:
                tmp["validation_" + metric] = metrics.get_scorer(metric)(
                    fitted, X_test, y_test
                )
            model_retrained_fits[name] = fitted
            dfs.append(tmp.query("fold==@top_fold").drop(columns="fold"))
        return (
            pd.concat(dfs, sort=True).reset_index(drop=True),
            model_retrained_fits,
            pd.concat(model_confs, sort=True),
        )

    def _permute_Y(self, seed):
        """
        Shuffle Y values.

        This function permutes the response values in Y.

        Parameters:
        ----------
        Y : pandas series
                Index of samples and values are their class labels
        seed : int
                Random seed for random number generator
        Returns:
        ------
        arr_shuffle: pandas series
                A shuffled Y

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        arr = shuffle(self.Y.values, random_state=seed)
        arr_shuffle = pd.Series(arr.reshape(1, -1)[0], index=self.Y.index)
        return arr_shuffle

    def permuted_mccv(self, seed):
        """
        Run MCCV permutation routine.

        Monte Carlo Cross Validation permutation routine


        Parameters
        ----------
        seed: int
                Random seed for random number generator

        Returns
        -------
        list

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        X = self.X.loc[self.Y.index]
        Y_shuffle = self._permute_Y(seed=seed)
        X_shuffle = X.loc[Y_shuffle.index]
        # 1/ train and test split
        X = X_shuffle.loc[Y_shuffle.index]
        X_train, X_test, y_train, y_test = train_test_split(
            X_shuffle,
            Y_shuffle,
            test_size=self.test_size,
            random_state=seed,
            shuffle=True,
        )
        X_train = X_train.apply(lambda x: (x - min(x)) / (max(x) - min(x)), axis=0)
        X_test = X_test.apply(lambda x: (x - min(x)) / (max(x) - min(x)), axis=0)
        X_train[X_train.isna()] = 0
        X_test[X_test.isna()] = 0
        # define K fold splitter
        cv = StratifiedKFold(n_splits=self.cv_split, random_state=seed, shuffle=True)
        # Instantiate lists to collect prediction and model results
        dfs = []
        model_retrained_fits = {}
        model_confs = []
        # iterate through model dictionary
        for name, mod in self._get_models().items():
            # /2 generate model parameters and fold scores with cv splitter
            fit = cross_validate(
                clone(mod),
                X_train,
                y_train.values.reshape(1, -1)[0],
                cv=cv,
                scoring=self.metrics,
                n_jobs=1,
                return_train_score=self.return_train_score,
                return_estimator=self.return_estimator,
            )
            tmp = pd.DataFrame(
                {"fold": range(self.cv_split), "model": name},
                index=range(self.cv_split),
            )
            # populate scores in dataframe
            cols = [k for k in fit.keys() if (k.find("test") + k.find("train")) == -1]
            for col in cols:
                tmp[col] = fit[col]
            # /3 Identify best performing model
            top_fold = np.where(fit["test_roc_auc"] == fit["test_roc_auc"].max())[0][0]
            keys = [x for x in fit.keys()]
            vals = [fit[x][top_fold] for x in keys]
            top_model_key_vals = {}
            for i in range(len(vals)):
                top_model_key_vals[keys[i]] = vals[i]
            # 4/ train models on training set
            # also get sample level predictions
            f = top_model_key_vals["estimator"]
            fitted = clone(f).fit(X_train, y_train.values.reshape(1, -1)[0])
            conf = pd.DataFrame(
                {
                    "y_true": y_test.values.reshape(1, -1)[0],
                    "y_pred": fitted.predict(X_test),
                    "y_proba": fitted.predict_proba(X_test)[:, 1],
                    "bootstrap": np.repeat(seed, len(y_test.index)),
                    "model": np.repeat(name, len(y_test.index)),
                },
                index=y_test.index,
            )
            model_confs.append(conf)
            # do prediction for each metric
            for metric in self.metrics:
                tmp["validation_" + metric] = metrics.get_scorer(metric)(
                    fitted, X_test, y_test
                )
            model_retrained_fits[name] = fitted
            dfs.append(tmp.query("fold==@top_fold").drop(columns="fold"))
        return (
            pd.concat(dfs, sort=True).reset_index(drop=True),
            model_retrained_fits,
            pd.concat(model_confs, sort=True),
        )

    def _get_performance(self, lst):
        """
        Extract MCCV model performance.

        Extract model performance statistics from mccv data

        Parameters
        ----------
        lst: list
                List of mccv data

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        perf = (
            pd.concat(lst, keys=range(len(lst)))
            .reset_index(level=1, drop=True)
            .rename_axis("bootstrap")
            .reset_index()
        )
        return perf

    def _model_feature_importances(self, boot_mods):
        """
        Extract MCCV feature performance.

        Extract feature performance statistics from mccv data (top performing model from cross validation)

        Parameters
        ----------
        lst: list
                List of models used in mccv routine

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        dfs = []
        X = self.X.copy()
        X.loc[:, "Intercept"] = 0
        for i in range(len(boot_mods)):
            for j in boot_mods[i].keys():
                mod = boot_mods[i][j]
                coef = []
                try:
                    coef.extend([i for i in mod.feature_importances_])
                except:
                    coef.extend([i for i in mod.coef_[0]])
                try:
                    coef.extend(mod.intercept_)
                except:
                    coef.extend([np.nan])
                fs = []
                fs.extend(X.columns.values)
                df = pd.DataFrame(
                    {"feature": fs, "importance": coef, "model": j, "bootstrap": i}
                )
                dfs.append(df)
        return pd.concat(dfs, sort=True)

    def _patient_predictions(self, lst, n=50):
        """
        Extract MCCV patient performance.

        Extract performance statistics from mccv patient data

        Parameters
        ----------
        lst: list
                List of mccv data
        n: int
                Number of bootstraps of validation data

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        dat = pd.concat(lst)
        score_vals = []
        for model in dat.model.unique():
            model_dat = dat[dat.model == model]
            for score in self.metrics:
                for b in range(n):
                    x = model_dat.sample(
                        n=model_dat.shape[0], replace=True, random_state=b
                    )
                    score_vals.append(
                        [
                            model,
                            score,
                            b,
                            metrics.get_scorer(score)._score_func(x.y_true, x.y_proba),
                        ]
                    )
        return dat, pd.DataFrame(
            score_vals, columns=["model", "metric", "performance_bootstrap", "value"]
        )

    def _bootstrap_of_function(self, func, params={}):
        """
        Run MCCV routines in parallel.

        Parameters
        ----------
        func: function
                Function to perform in parallel
        params: dictionary
                Dictionary of arguments for parallelized function. Currently not used.

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """
        if func == None:
            return "Need function to bootstrap"
        parallel = Parallel(n_jobs=self.n_jobs)
        return parallel(
            delayed(func)(seed=k, **params) for k in range(self.num_bootstraps)
        )

    def run_mccv(self):
        """
        Run MCCV procedure.

        Wrapper to compute and extract statistics from MCCV routine

        Parameters
        ----------
        NULL

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """

        lst = self._bootstrap_of_function(func=self.mccv)

        perf = self._get_performance([lst[i][0] for i in range(self.num_bootstraps)])

        boot_mods = [lst[i][1] for i in range(self.num_bootstraps)]
        fimp = self._model_feature_importances(boot_mods)

        tmp = self._patient_predictions([lst[i][2] for i in range(self.num_bootstraps)])
        ppreds = tmp[0]
        ppreds_scores = tmp[1]

        self.mccv_data = {
            "Model Learning": perf,
            "Feature Importance": fimp,
            "Patient Predictions": ppreds,
            "Performance": ppreds_scores,
        }

    def run_permuted_mccv(self):
        """
        Run MCCV permutation procedure.

        Wrapper to compute and extract statistics from MCCV permutation routine

        Parameters
        ----------
        NULL

        Returns
        -------
        NULL

        Notes
        -----
        NULL

        Examples
        --------
        NULL

        """

        lst = self._bootstrap_of_function(func=self.permuted_mccv)

        perf = self._get_performance([lst[i][0] for i in range(self.num_bootstraps)])

        boot_mods = [lst[i][1] for i in range(self.num_bootstraps)]
        fimp = self._model_feature_importances(boot_mods)

        tmp = self._patient_predictions([lst[i][2] for i in range(self.num_bootstraps)])
        ppreds = tmp[0]
        ppreds_scores = tmp[1]

        self.mccv_permuted_data = {
            "Model Learning": perf,
            "Feature Importance": fimp,
            "Patient Predictions": ppreds,
            "Performance": ppreds_scores,
        }
