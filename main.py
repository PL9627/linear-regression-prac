import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sn

# dataset
disease_df = pd.read_csv(
    "C:/Users/Paul/linear-regress-prac/LR-prac/cardio_data_processed.csv")

# removing null/NaN values
disease_df.dropna(axis=0, inplace=True)
""" print(disease_df.head(), disease_df.shape)
print(disease_df.cardio.value_counts()) """

""" plt.figure(figsize=(7, 5))
sn.countplot(x='cardio', data=disease_df,
             palette="BuGn_r")
plt.show()

laste = disease_df['cardio'].plot()
plt.show(laste) """

X = np.array(
    disease_df[['age_years', 'gender', 'smoke', 'cholesterol', 'ap_hi', 'gluc']])
y = np.array(disease_df['cardio'])

# normalization of dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)

""" print('train set:', X_train.shape, y_train.shape)
print('test set:', y_train, y_test)
 """

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# eval and accuracy
print('')
print('Accuracy of model in jaccard similarity score is:',
      jaccard_score(y_test, y_pred))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

score = rf.score(X_test, y_test)*100
print('Accuracy of the random forest classifier model is:', score)

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm,
                           columns=['Predicted:0', 'Predicted:1'],
                           index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")

plt.show()

print('The details for confusion matrix is =')
print(classification_report(y_test, y_pred))
