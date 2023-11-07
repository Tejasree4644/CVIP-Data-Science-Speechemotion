#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install librosa scikit-learn matplotlib')
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def extract_features(file_path):
    try:
        audio, _ = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file:", file_path)
        return None
    return mfccs_mean

emotions = {
    'happy': 0,
    'sad': 1,
    'angry': 2,
    'neutral': 3
}

data = []
labels = []

for emotion, label in emotions.items():
    for i in range(1, 11):  
        file_path = "VideoDemographics.csv"
        features = extract_features(file_path)
        if features is not None:
            data.append(features)
            labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42,stratify=labels)

classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, alpha=1e-4,
                           solver='sgd', verbose=10, tol=1e-4, random_state=1,
                           learning_rate_init=.1)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:




