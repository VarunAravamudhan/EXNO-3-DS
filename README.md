## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="470" height="482" alt="image" src="https://github.com/user-attachments/assets/bc4a0979-c46c-452e-a7c6-999baccf8d4c" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="305" height="261" alt="image" src="https://github.com/user-attachments/assets/85a38028-c86c-4863-aad0-58e6a2f41e8c" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="491" height="497" alt="image" src="https://github.com/user-attachments/assets/eccc1fd9-4ad6-42d1-8445-a1ad108c8db8" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="531" height="494" alt="image" src="https://github.com/user-attachments/assets/da89cdf6-530a-4b89-8636-3420627c6af8" />

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="590" height="491" alt="image" src="https://github.com/user-attachments/assets/41e9ff60-efa0-431f-ad13-474ad3956879" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="895" height="508" alt="image" src="https://github.com/user-attachments/assets/03b418f5-e3eb-467e-89ab-b55c2b2d9465" />

```
!pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="955" height="576" alt="image" src="https://github.com/user-attachments/assets/f221e24d-f770-4b88-b343-6bfa4cb43b1c" />


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="874" height="518" alt="image" src="https://github.com/user-attachments/assets/122ff7cf-801a-4de1-818b-1e749a59eea0" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="1130" height="528" alt="image" src="https://github.com/user-attachments/assets/c03da130-0efb-4a60-aecf-883ce3e5fb35" />

```
df.skew()
```
<img width="412" height="284" alt="image" src="https://github.com/user-attachments/assets/88498aa1-0c23-4865-9e9e-48d1c455a2cf" />

```
np.log(df["Highly Positive Skew"])
```
<img width="445" height="585" alt="image" src="https://github.com/user-attachments/assets/3b031fe5-86e0-4bd0-87de-fc304f337d6a" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="454" height="600" alt="image" src="https://github.com/user-attachments/assets/60d41e1b-666c-4baa-ab10-5ef73ac60fb3" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="419" height="579" alt="image" src="https://github.com/user-attachments/assets/a855bdfb-dbc7-4553-9f7e-4f28ff0c9dcc" />

```
np.square(df["Highly Positive Skew"])
```
<img width="369" height="547" alt="image" src="https://github.com/user-attachments/assets/2960b218-9c8a-418b-819b-30576e9882c7" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1322" height="561" alt="image" src="https://github.com/user-attachments/assets/0d9eba0e-f377-4f08-b161-97460718af97" />

```
df.skew()
```
<img width="521" height="360" alt="image" src="https://github.com/user-attachments/assets/7da3a55a-705f-48dc-ab5e-048e88dc6ccd" />

```
df["Highly Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Highly Negative Skew"])
display(df.skew())
```




<img width="588" height="324" alt="image" src="https://github.com/user-attachments/assets/e1a122a4-a08f-42a2-8e6d-9e71a52724c9" />






```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```





<img width="1383" height="592" alt="image" src="https://github.com/user-attachments/assets/a5dea03f-aed3-49c2-a21f-d8154a80fe0b" />






```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```





<img width="875" height="622" alt="image" src="https://github.com/user-attachments/assets/10aa3106-1b43-4548-b68f-a7e1bf6048f6" />






```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```




<img width="857" height="678" alt="image" src="https://github.com/user-attachments/assets/d613d02e-3d77-47e7-bd4d-47f7017e3311" />





```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```




<img width="800" height="603" alt="image" src="https://github.com/user-attachments/assets/8ad67072-862e-403d-ab25-cb30003537e5" />




```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```



<img width="843" height="605" alt="image" src="https://github.com/user-attachments/assets/ba479afa-b298-4d4b-9fb9-6fa5c3b175ea" />




```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```



<img width="822" height="625" alt="image" src="https://github.com/user-attachments/assets/7700534b-c714-4887-9b44-56150304a8f2" />




# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
