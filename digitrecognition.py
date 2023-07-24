import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X,y = fetch_openml('mnist_784', version = 1, return_X_y = True)


#print(len(X))
#print(pd.Series(y).value_counts())

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

len_classes = len(classes)

sample_per_class = 5
plt.figure(figsize = (len_classes * 2, ((1+sample_per_class) * 2)))

idx_cls = 0;

for cls in classes:
  idxs = np.flatnonzero(y == cls)



  idxs = np.random.choice(idxs, sample_per_class, replace = False)

  i = 0

  for idx in idxs:
    plt_idx = i * len_classes + idx_cls + 1
    p = plt.subplot(sample_per_class, len_classes, plt_idx)
    p = sns.heatmap(np.array(X.loc[idx]).reshape(28, 28), cbar= False, cmap = plt.cm.gray, xticklabels = False, yticklabels = False)
    p = plt.axis('off')
    i+=1;

  idx_cls +=1;

print(len(X.loc[6]))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

classifier = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

