import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold


def get_f1_score(threshold, data):
    '''
    Calculate the F1-score for a given threshold.
    
    Args:
        threshold (float): Threshold for the cosine similarity.
        data (list): List of dictionaries with the following keys:
            - score: Cosine similarity score.
            - pred_bioguide: Predicted bioguide.
            - same_person: 'Yes' if the bioguide is the same person, 'No' otherwise.
            - bioguide: True bioguide.
    
    Returns:
        (float): F1-score for the given threshold.
    '''
    prediction = []
    truth = []
    for row in data:
        # Check prediction
        if row['score'] >= threshold:
            pred_bioguide = row['pred_bioguide']
        else:
            pred_bioguide = 'baseline'
        # Check truth
        if row['same_person'] == 'Yes':
            truth_bioguide = row['bioguide']
        else:
            truth_bioguide = 'baseline'
        # Store results
        prediction.append(pred_bioguide)
        truth.append(truth_bioguide)
    
    # F1-score
    return f1_score(truth, prediction, average='micro')
        

def inverse_f1_score(threshold, data):
    '''
    Calculate the inverse F1-score for a given threshold.
    
    Args:
        threshold (float): Threshold for the cosine similarity.
        data (list): List of dictionaries with the following keys:
            - score: Cosine similarity score.
            - pred_bioguide: Predicted bioguide.
            - same_person: 'Yes' if the bioguide is the same person, 'No' otherwise.
            - bioguide: True bioguide.

    Returns:
        (float): Inverse F1-score for the given threshold.
    '''
    f1_estimate = get_f1_score(threshold, data)
    return 1 - f1_estimate

# Load validation embeddings
df_validation = pd.read_pickle('data/validation/validation_embeddings.pkl')

# Make class variable
df_validation['bioguide_class'] = np.where(df_validation['same_person'] == 'Yes', df_validation['bioguide'], 'baseline')

# -------------------------------------------------------
# Find optimal similarity to portrait classification

# Create splits
content = df_validation.to_dict('records')
y = [row['bioguide_class'] for row in content]

# Get portrait embeddings
portrait = pd.read_pickle('data/validation/portraits_encodings.pkl')
portrait_encodings = np.array([row['encodings'] for row in portrait])
portrait_bioguides = [row['image_name'] for row in portrait]

# Calculate similarity to validation data embeddings
embeddings = np.array([row['embedding'][0] for row in content])
sim = cosine_similarity(embeddings, portrait_encodings)
idx = np.argmax(sim, axis=1)

# Make data for optimization
opt_data = []
same_person = df_validation.same_person.tolist()
bioguide =  df_validation.bioguide.tolist()
for i in range(embeddings.shape[0]):
    opt_data.append({
        'score': sim[i,idx[i]],
        'pred_bioguide': portrait_bioguides[idx[i]],
        'same_person': same_person[i],
        'bioguide': bioguide[i]
        })

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle = True, random_state=50)

# Loop over folds and calculate performance measure
results = []
for k, (train_idx, test_idx) in enumerate(kf.split(opt_data)):
    # Extract training and test data
    content_train = [opt_data[i] for i in train_idx]
    content_test = [opt_data[i] for i in test_idx]
    # Optimize F1 curve by threshold
    res = minimize(inverse_f1_score, .5, method='nelder-mead', args=(content_train),
               options={'xatol': 1e-8, 'disp': True})
    optimal_threshhold =  res['x'][0]
    
    # OOP and Write results
    result = {'fold': k,
              'threshold': optimal_threshhold,
              'f1': get_f1_score(optimal_threshhold, content_test),}
              
    results.append(result)


# Optimize F1 curve by threshold using full data
res = minimize(inverse_f1_score, .5, method='nelder-mead', args=(opt_data),
               options={'xatol': 1e-8, 'disp': True})

optimal_threshhold =  res['x'][0]
print(f'Optimal threshold = {optimal_threshhold}')
print(f'10-Fold CV F1-score = {np.mean([row["f1"] for row in results])}')

# Make data for optimization plot
start = .5
stop = .9
res = []
threshold_values = np.linspace(start, stop, num=80, endpoint=True)
for threshold in threshold_values.tolist():
    res.append(get_f1_score(threshold, opt_data))
