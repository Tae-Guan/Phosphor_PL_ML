<div align="center">

![](https://www.jbnu.ac.kr/kor/images/logo_tr.png)

![](http://eai.jbnu.ac.kr/assets/images/common/header-logo(1).png)

### Machine learning investigation to predict the relationship between photoluminescence and crystalline properties of Blue Phosphor Ba0.9-xSrxMgAl10o17:Eu2+

Kim, Tae-Guan; Jurakuziev, Dadajon Boykuzi Ugli; Akhtar, M.Shaheer; Yang, O-Bong;

Graduate School of Integrated Energy-AI, Jeonbuk National University, Korea, *54896*

_Corresponding author: Akhtar, M.Shaheer (shaheerakhtar@jbnu.ac.kr) Yang, O-Bong (obyang@jbnu.ac.kr)

______________________________________________________________________

[![Python](https://img.shields.io/badge/python-3.8.12-blue)](https://www.python.org/downloads/release/python-3610/)
[![PyCaret](https://img.shields.io/badge/pycaret-2.3.9-red)](https://github.com/tensorflow/tensorflow/releases/tag/v1.12.0)
[![Flask](https://img.shields.io/badge/explainerdashboard-0.4.0-green)](https://pypi.org/project/Flask/1.1.2/)

</div>

### 1. Purpose

To predict the relationship between photoluminescence and crystalline properties of blue phosphor Ba0.9-xSrxMgAl10O17:Eu2+ using machine learning.

### 2. Data availability

Derived data supporting the findings of this study are available from the corresponding author on request.

### 3. System requirements

The source code is tested of the following 64-bit systems:
- Ubuntu 18.04 LTS
- Windows 10
- Python 3.7.13

#### Install required packages

```bash
pip install -r requirements.txt
```

### 4. Notebook contents

#### Import libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from pycaret.regression import *
import seaborn as sns
```

#### Data preparation

```python
df = pd.read_csv('./datasets/dataset.csv', encoding="UTF-8")
print(df.head())
print(df.columns)
```

#### Pearson Correlation Analysis

```python
sns.set(font_scale=1.1)
plt.figure(figsize=(9,8))
corr= df.corr()
sns.heatmap(corr, annot=True, square=False, vmin=-1.0, vmax=1.0, cmap="BuGn",annot_kws={"size": 20}); #annot parameter fills the cells with the relative correlation coefficient, which ranges from -1 to 1
plt.savefig("test.png")
```

#### Feature Engineering

```python
from pycaret.regression import *
MachineLearning_Model = setup(data = df, target = 'Wavelength', session_id=123, train_size = 0.8,
                   log_experiment = True, experiment_name = 'Crystal_Structure_PL-Prediction')
```

#### Modeling

```python
top5 = compare_models(sort='R2', n_select=5)
```

#### Blend top5 models into an ensemble Voting Regressor model

```python
blender_top5 = blend_models(estimator_list=top5)
```

#### Predicting

```python
final_model_1 = finalize_model(blender_top5)
prediction = predict_model(final_model_1)
```

### License

This project is licensed under the terms of the GNU General Public License v3.0

### Citations: tobe defined later

```bibtex
@inproceedings{???,
  title   = {Machine learning investigation to predict the relationship between photoluminescence and crystalline properties of Blue Phosphor Ba0.9-xSrxMgAl10o17:Eu2+},
  author  = {Kim, Tae-Guan; Jurakuziev, Dadajon Boykuzi Ugli; Akhtar, M.Shaheer; Yang, O-Bong;},
  year    = {2022}
}
```
