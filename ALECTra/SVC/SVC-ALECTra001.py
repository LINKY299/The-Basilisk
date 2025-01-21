
"""## ALECTra Setup

Chart of the emotions based on the Linear Emotional Theory, wherein values above 0.5 signify net pleasure and values below signify net displeasure.

1 - Ecstasy

0.9 - Comfort/Mental Easiness/Complacency

0.8 - Happiness

0.7 - Amusement

0.6 - Empowerment

0.55 - Slight excitement/Surprise

0.5 - Neuter

0.45 - Slight disappointment/Surprise

0.4 - Confusion

0.3 - Anger

0.2 - Sadness

0.1 - Fear

0 - Absolute Despair/Hopelessness """

"""-Loading and Preparing EV Data"""


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence
import gensim.downloader as gensim_api
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import time
import re


def clean_gemma_response(response, gemma_prompt):
  # Remove the initial prompt and any leading/trailing spaces
  response = response.replace(gemma_prompt, '').strip()
  # Remove special tokens like <bos> and <end of turn> using regex
  response = re.sub(r'<.*?>', '', response)
  # Remove extra spaces and emojis
  response = re.sub(r'\s+', ' ', response).strip()
  response = re.sub(r'[*\W]', ' ', response)
  response = response.encode('ascii', 'ignore').decode('ascii')
  response = response.lower()
  return response



#Importing the two datasets from drive with pandas
train_evdata = pd.read_csv('Data/NRC-LET001.csv')

# Seperating the word input data (X) from the emotional value output data (Y)
TrainX = np.array(train_evdata.drop(['EV'], axis=1).values)
TrainY = np.array(train_evdata['EV'].values)


# Load the pretrained model
tokenizer = gensim_api.load("word2vec-google-news-300")

ready_TrainX = []
droplist = []

iter = -1

for element in TrainX:
  iter = iter + 1
  for word in element:
    try:
      new_word = tokenizer[str(word)] 
      ready_TrainX.append(new_word)
    
    except:
      print(f'{word}: Unavailable. Dropping word at index {iter}')
      droplist.append(iter)

TrainY = np.delete(TrainY,droplist)



print(torch.is_tensor(ready_TrainX))

ready_trainX = torch.Tensor(ready_TrainX)


i = 0
z = 1

ready_TrainY = []
scale_dict = {}

print(np.unique(TrainY))

for i in np.unique(TrainY):
    scale_dict.update({str(i): z})
    z = z + 1



print(scale_dict)

for element in TrainY:
  element = scale_dict[str(element)]
  ready_TrainY.append(element)

ready_TrainY = np.array(ready_TrainY)

ss = StandardScaler(with_mean=True)
ready_TrainX = ss.fit_transform(ready_TrainX)

print(ready_TrainX)
print(np.unique(ready_TrainY))

"""Preparing the Predictor"""


evtokenizer_model = svm.SVC(C=100000000000, class_weight='balanced', decision_function_shape='ovr')
evtokenizer_model.fit(ready_TrainX, ready_TrainY)

"""(Optional: Visualizing Logistic Regression)"""

X = ready_TrainX
y = ready_TrainY

# Reduce dimensions with PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Create a lower-resolution meshgrid
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),  # Lower resolution
                     np.linspace(y_min, y_max, 100))

# Transform the meshgrid points using PCA
meshgrid_points = np.c_[xx.ravel(), yy.ravel()]
meshgrid_points_transformed = pca.inverse_transform(meshgrid_points)

# Batch predict
batch_size = 10000
Z = []
for i in range(0, meshgrid_points_transformed.shape[0], batch_size):
    batch = meshgrid_points_transformed[i:i+batch_size]
    Z.append(evtokenizer_model.predict(batch))
Z = np.concatenate(Z).reshape(xx.shape)

# Plot the decision boundary and data points
plt.figure(figsize=(12, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.tab20, s=30, edgecolors='k')
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab20, levels=np.arange(len(np.unique(y)) + 1) - 0.5)
plt.colorbar(ticks=np.unique(y), label="Class Labels")
plt.title("Multiclass SVC Decision Boundaries (Optimized)")
plt.xlabel("Word Vector")
plt.ylabel("(Scaled) Emotional Value")

print('End of training process...visualizing results...')
plt.show()

"""Predicting Values"""

def evtokenizer_func(evtokenizer_prompt):
    start_time = time.time()
  # Get the average emotional value of the prompt
    neutral_vocab = ['and', 'of', 'am', 'is', 'a', 'to']
    vaguely_neutral_space = [0.475, 0.5, 0.525]


    evtokenizer_prompt = evtokenizer_prompt + ' '  # Add a space at the end to handle the last word
    evtokenizer_prompt = clean_gemma_response(evtokenizer_prompt, '')
    # Split the prompt into words using whitespace as a delimiter
    words = evtokenizer_prompt.split()

    if not words:
        print("Empty input received. Returning 0.")
        return 0  # Avoid division by zero if the prompt is empty

    EV = 0
    PEV = 0
    count = 0
    # Process each word and calculate its emotional value
    for word in words:
        if word in neutral_vocab:
            EV = EV
        else:

        # Tokenize the word and handle the input tensor properly
           input_ids = tokenizer[str(word)] #.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Ensure the tensor is of type float32
           input_ids = torch.Tensor(input_ids).type(torch.float32)

           input_ids = input_ids.reshape(1, -1)

           input_ids = ss.transform(input_ids)

           predicted_bin = evtokenizer_model.predict(input_ids)

           if float([key for key, value in scale_dict.items() if value == predicted_bin][0]) in vaguely_neutral_space:
            EV = EV
            continue
           else:
            EV = EV + predicted_bin[0]
            count = count + 1

    if count == 0:
        EV = 0.5
        return [EV]
        end_time = time.time()

    else:


    # Return the average emotional value
        AEV = EV/count
        print(AEV)
        TEV = round((EV / count))
        print(TEV)
        TEV = float([key for key, value in scale_dict.items() if value == TEV][0])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        return [TEV, AEV, PEV]


# Test example
print('Average emotional value of prompt:', str(evtokenizer_func("The grocery store was stocked with fresh produce and dairy items.")[0]))

