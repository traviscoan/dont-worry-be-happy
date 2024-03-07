import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score



content = pd.read_json('data/validation.json')

kf = StratifiedKFold(n_splits=5,
                     shuffle=True,
                     random_state=50)


le = preprocessing.LabelEncoder()
bioguide_class = content['bioguide_class'].tolist()
y = le.fit_transform(bioguide_class)
X = np.array(content['encodings'].tolist())

results = []
for k in range(5,20):
    neigh = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    f1_estimate = cross_val_score(neigh, X, y, cv=kf, scoring='f1_micro')
    precision_estimate = cross_val_score(neigh, X, y, cv=kf, scoring='precision_micro')
    recall_estimate = cross_val_score(neigh, X, y, cv=kf, scoring='recall_micro')
    results.append({
        'k': k,
        'f1': f1_estimate.mean(),
        'precision': precision_estimate.mean(),
        'recall': recall_estimate.mean()
    })
    
    print(f'k = {k}', f1_estimate.mean())