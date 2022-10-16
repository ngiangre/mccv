# mccv: A python package to perform Monte Carlo Cross Validation

Based on the algorithm developed and implemented by Giangreco et al. (see references).  

<object data="docs/imgs/MCCV Prediction scheme.pdf" type="application/pdf"></object>

# Objectives

1. To easily implement and perform MCCV for learning and prediction tasks.
2. To choose between different machine learning models including Logistic Regression, Random Forest, Support Vector Machines, and Gradient Boosting Machhines
3. To ingest dataframes and return dataframes from the learning task. 

# Installation

```
cd mccv/
pip3 install .
```

# Example Usage

```
#Data import
import pandas as pd
data = pd.read_csv('data/data.csv',index_col=0) # Feature column name is 'biomarker' and response column  name is 'status'


#MCCV procedure
import mccv
mccv_obj = mccv.mccv(num_bootstraps=200)
mccv_obj.set_X( data.loc[:,['biomarker']] )
mccv_obj.set_Y( data.loc[:,['status']] )
mccv_obj.run_mccv()
mccv_obj.run_permuted_mccv()

#Output
mccv_obj.mccv_data # 4 element dictionary
mccv_obj.mccv_permuted_data # 4 element dictionary

```
# Contribute

Please do! Reach out to Nick directly (nick.giangreco@gmail.com), make an issue, or make a pull request.

# License

This software is released under the MIT license, which can be found in LICENSE in the root directory of this repository.

# Citation

Giangreco, N.P., Lebreton, G., Restaino, S. et al. Alterations in the kallikrein-kinin system predict death after heart transplant. Sci Rep 12, 14167 (2022). [https://doi.org/10.1038/s41598-022-18573-2](https://doi.org/10.1038/s41598-022-18573-2)

Giangreco et al. 2021. Plasma kallikrein predicts primary graft dysfunction after heart transplant. Journal of Heart and Lung Transplantation, 40(10), 1199-1211. [https://doi.org/10.1016/j.healun.2021.07.001](https://doi.org/10.1016/j.healun.2021.07.001).


