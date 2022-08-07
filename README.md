# Machine Learning
Using the iris dataset, the data is analyzed and a model is created to derive the predicted values and compare them with the actual values.

# Step 1:
Import a pre-loaded dataset from the scikit-learn python library.
```
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.DESCR)
```
# Step 2:
Use the Pandas library to put this information into a data frame. Create a data frame to store data information about the flower characteristics.
```
import pandas as pd
data = pd.DataFrame(iris.data)
data.head()
```
# Step 3: 
Perform data processing, including removing redundant, meaningless data and organizing tables, breaking down rows and columns.
```
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#note: it is common practice to use underscores between words, and avoid spaces
data.head() 
```

# Step 4: 
Approaches to data cleaning
##Look at Data Types，Check for Missing Values，Statistical Overview
```
df.dtypes
df.isnull().sum()
df.describe()
```

#Step 5:
Visualizing
##The correlation between features is mapped through the form of a table; the larger the number, the greater the correlation between two elements.
### Use Heatmap
```
import seaborn as sns
sns.heatmap(df.corr(), annot = True);
```
### Use scatter plot
```
x_index = 0
y_index = 1
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()
```


# Step 6:
Modeling
##Create a logistic regression classification model that will predict which category a flower belongs to based on the size of its petals and sepals.
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=10)
print(np.mean(scores))
```


# Step 7：
Using the Confusion Matrix allows a closer look at the predictions made by the model
```
df_coef = pd.DataFrame(model.coef_, columns=X_train.columns)
df_coef
```
# Step 8:
Classification Report
```
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```

# Step 9:
PPT
##
https://docs.google.com/presentation/d/1GC3VDFsb75WTDRcImeYScV1fL-3Gw_JL-Vo1HDWMFQo/edit?usp=sharing

# Reference
## Classification Basics: Walk-through with the Iris Data Set
https://towardsdatascience.com/classification-basics-walk-through-with-the-iris-data-set-d46b0331bf82
