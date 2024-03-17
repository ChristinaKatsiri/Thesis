import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from nltk.corpus import wordnet_ic

brown_ic = wordnet_ic.ic('ic-brown.dat')

def get_word_synset(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    return synsets[0] if synsets else None

# Define representative keywords for each class
keywords_per_class = {
    "Positive": ["happy", "good", "enjoy", "love", "excellent" , "Positive"],
    "Neutral": ["average", "okay", "mediocre", "Neutral"],
    "Negative": ["bad", "sad", "poor", "terrible", "awful" ,"Negative"]
}

# Example usage with your DataFrame
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/hotel_reviews/Modified_Hotel_Reviews_3_total_bert.csv", sep = ',')
#reviews_df = reviews_df.dropna(subset=['review_tag_bert'])

# Now X can be used in your machine learning model
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(reviews_df['review_tag_bert'])



#1 wu_palmer

def calculate_wup_similarity(doc, keywords):
    synsets1 = [get_word_synset(word) for word in word_tokenize(doc)]
    synsets2 = [get_word_synset(word) for word in keywords]
    
    synsets1 = [synset for synset in synsets1 if synset]
    synsets2 = [synset for synset in synsets2 if synset]

    scores = []
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.wup_similarity(synset2)
            if similarity is not None:
                scores.append(similarity)

    return np.mean(scores) if scores else 0

# Create feature matrix X
X_features = []
for _, row in reviews_df.iterrows():
    review = row['review_total']
    review_class = row['review_tag_bert']
    similarity = calculate_wup_similarity(review, keywords_per_class[review_class])
    X_features.append([similarity])

X_wu_palmer = np.array(X_features)

print('X_wu_palmer', X_wu_palmer)

# Train-test split
X_train_wu_palmer, X_test_wu_palmer, y_train_wu_palmer, y_test_wu_palmer = train_test_split(X_wu_palmer, y_encoded, test_size=0.2, random_state=42)
y_binarized_wu_palmer = label_binarize(y_test_wu_palmer, classes=np.unique(y_encoded))
print('X_train_wu_palmer', X_train_wu_palmer)
print('y_train_wu_palmer',y_train_wu_palmer)

# Train a RandomForest Classifier
model = RandomForestClassifier()
model.fit(X_train_wu_palmer, y_train_wu_palmer)

# Predict probabilities
predicted_probs_wu_palmer = model.predict_proba(X_test_wu_palmer)

# Compute ROC curve and ROC area for each class
#fpr_wu_palmer, tpr_wu_palmer, _ = roc_curve(y_binarized_wu_palmer.ravel(), predicted_probs_wu_palmer.ravel())
fpr_wu_palmer, tpr_wu_palmer, roc_thresholds_wu_palmer = roc_curve(y_binarized_wu_palmer.ravel(), predicted_probs_wu_palmer.ravel())

roc_auc_wu_palmer = auc(fpr_wu_palmer, tpr_wu_palmer)

# Compute the average precision score (micro-averaged)
average_precision_wu_palmer = average_precision_score(y_binarized_wu_palmer, predicted_probs_wu_palmer, average="micro")

# Compute the micro-averaged Precision-Recall curve and area under the curve
#precision_wu_palmer, recall_wu_palmer, _ = precision_recall_curve(y_binarized_wu_palmer.ravel(), predicted_probs_wu_palmer.ravel())
precision_wu_palmer, recall_wu_palmer, pr_thresholds_wu_palmer = precision_recall_curve(y_binarized_wu_palmer.ravel(), predicted_probs_wu_palmer.ravel())

#print("Precision_wu_palmer:", precision_wu_palmer)
#print("Recall_wu_palmer:", recall_wu_palmer)
#print("Thresholds_wu_palmer:" ,pr_thresholds_wu_palmer)


pr_thresholds_wu_palmer = np.append(pr_thresholds_wu_palmer, np.nan)

data = {
    "Precision_wu_palmer": precision_wu_palmer,
    "Recall_wu_palmer": recall_wu_palmer,
    "Thresholds_wu_palmer": pr_thresholds_wu_palmer
}

# Creating a DataFrame
df = pd.DataFrame(data)


# Printing the DataFrame
print(df.to_string(index=False))

# Compute Precision-Recall curve and average precision
#precision, recall, _ = precision_recall_curve(y_test, predicted_probs[:, 1])
#precision, recall, _ = precision_recall_curve(y_binarized.ravel(), predicted_probs.ravel())

#average_precision = average_precision_score(y_test, predicted_probs[:, 1])
interpolated_precision_wu_palmer = np.maximum.accumulate(precision_wu_palmer[::1])[::1]


#2 path

def path_similarity_doc(doc, keywords):
    synsets1 = [get_word_synset(word) for word in word_tokenize(doc)]
    synsets2 = [get_word_synset(word) for word in keywords]  
    synsets1 = [synset for synset in synsets1 if synset]
    synsets2 = [synset for synset in synsets2 if synset]

    scores = []
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity is not None:
                scores.append(similarity)

    return np.mean(scores) if scores else 0
    
    
# Create feature matrix X
X_features = []
for _, row in reviews_df.iterrows():
    review = row['review_total']
    review_class = row['review_tag_bert']
    similarity = path_similarity_doc(review, keywords_per_class[review_class])
    X_features.append([similarity])

X_path = np.array(X_features)
# Now X can be used in your machine learning model
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(reviews_df['review_tag_bert'])

# This is a placeholder: you'll need to fill in the logic to create a feature matrix X using wup_similarity
# Example: X[i, j] could be the Wu-Palmer similarity between the ith and jth document
# X = ...

# Train-test split
X_train_path, X_test_path, y_train_path, y_test_path = train_test_split(X_path, y_encoded, test_size=0.2, random_state=42)
y_binarized_path = label_binarize(y_test_path, classes=np.unique(y_encoded))
#print("X_train" ,X_train)
#print("X_test" , X_test)
#print("y_train", y_train)
#print("y_test", y_test)
#print("y_binarized", y_binarized)
# Train a RandomForest Classifier
model = RandomForestClassifier()
model.fit(X_train_path, y_train_path)

# Predict probabilities
predicted_probs_path = model.predict_proba(X_test_path)

# Compute ROC curve and ROC area for each class
fpr_path, tpr_path, roc_thresholds_path = roc_curve(y_binarized_path.ravel(), predicted_probs_path.ravel())

roc_auc_path = auc(fpr_path, tpr_path)
# Compute the average precision score (micro-averaged)
average_precision_path = average_precision_score(y_binarized_path, predicted_probs_path, average="micro")

# Compute the micro-averaged Precision-Recall curve and area under the curve
precision_path, recall_path, pr_thresholds_path = precision_recall_curve(y_binarized_path.ravel(), predicted_probs_path.ravel())

#print("Precision_path:", precision_path)
#print("Recall_path:", recall_path)
#print("Thresholds_path:" ,pr_thresholds_path)
pr_thresholds_path = np.append(pr_thresholds_path, np.nan)

data = {
    "Precision_path": precision_path,
    "Recall_path": recall_path,
    "Thresholds_path": pr_thresholds_path
}

# Creating a DataFrame
df = pd.DataFrame(data)


# Printing the DataFrame
print(df.to_string(index=False))

# Compute Precision-Recall curve and average precision
#precision, recall, _ = precision_recall_curve(y_test, predicted_probs[:, 1])
#precision, recall, _ = precision_recall_curve(y_binarized.ravel(), predicted_probs.ravel())

#average_precision = average_precision_score(y_test, predicted_probs[:, 1])
interpolated_precision_path = np.maximum.accumulate(precision_path[::1])[::1]



#3    
def resnik_similarity_doc(doc, keywords, ic=brown_ic):
    synsets1 = [get_word_synset(word) for word in word_tokenize(doc)]
    synsets2 = [get_word_synset(word) for word in keywords]
    
    synsets1 = [synset for synset in synsets1 if synset]
    synsets2 = [synset for synset in synsets2 if synset]

    scores = []
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.res_similarity(synset2, ic)
            if similarity is not None:
                scores.append(similarity)

    return np.mean(scores) if scores else 0

# Create feature matrix X
X_features_res = []
for _, row in reviews_df.iterrows():
    review = row['review_total']
    review_class = row['review_tag_bert']
    similarity_res = resnik_similarity_doc(review, keywords_per_class[review_class])
    X_features_res.append([similarity_res])

X_res = np.array(X_features_res)

# Train-test split
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_encoded, test_size=0.2, random_state=42)
y_binarized_res = label_binarize(y_test_res, classes=np.unique(y_encoded))
#print("X_train" ,X_train)
#print("X_test" , X_test)
#print("y_train", y_train)
#print("y_test", y_test)
#print("y_binarized", y_binarized)
# Train a RandomForest Classifier
model = RandomForestClassifier()
model.fit(X_train_res, y_train_res)

# Predict probabilities
predicted_probs_res = model.predict_proba(X_test_res)

# Compute ROC curve and ROC area for each class
fpr_res, tpr_res, roc_thresholds_res = roc_curve(y_binarized_res.ravel(), predicted_probs_res.ravel())
roc_auc_res = auc(fpr_res, tpr_res)

# Compute the average precision score (micro-averaged)
average_precision_res = average_precision_score(y_binarized_res, predicted_probs_res, average="micro")

# Compute the micro-averaged Precision-Recall curve and area under the curve
precision_res, recall_res,pr_thresholds_res = precision_recall_curve(y_binarized_res.ravel(), predicted_probs_res.ravel())
#print("Precision_res:", precision_res)
#print("Recall_res:", recall_res)
#print("Thresholds_res:" ,pr_thresholds_res)

pr_thresholds_res = np.append(pr_thresholds_res, np.nan)

data = {
    "Precision_res": precision_res,
    "Recall_res": recall_res,
    "Thresholds_res": pr_thresholds_res
}

# Creating a DataFrame
df = pd.DataFrame(data)


# Printing the DataFrame
print(df.to_string(index=False))

# Compute Precision-Recall curve and average precision
#precision, recall, _ = precision_recall_curve(y_test, predicted_probs[:, 1])
#precision, recall, _ = precision_recall_curve(y_binarized.ravel(), predicted_probs.ravel())

#average_precision = average_precision_score(y_test, predicted_probs[:, 1])
interpolated_precision_res = np.maximum.accumulate(precision_res[::1])[::1]




#4    
def lch_similarity_doc(doc, keywords):
    synsets1 = [get_word_synset(word) for word in word_tokenize(doc)]
    synsets2 = [get_word_synset(word) for word in keywords]
    
    synsets1 = [synset for synset in synsets1 if synset]
    synsets2 = [synset for synset in synsets2 if synset]

    scores = []
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.lch_similarity(synset2)
            if similarity is not None:
                scores.append(similarity)

    return np.mean(scores) if scores else 0

# Create feature matrix X
X_features_lch = []
for _, row in reviews_df.iterrows():
    review = row['review_total']
    review_class = row['review_tag_bert']
    similarity_lch = lch_similarity_doc(review, keywords_per_class[review_class])
    X_features_lch.append([similarity_lch])

X_lch = np.array(X_features_lch)

# Now X can be used in your machine learning model
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(reviews_df['review_tag_bert'])

# This is a placeholder: you'll need to fill in the logic to create a feature matrix X using wup_similarity
# Example: X[i, j] could be the Wu-Palmer similarity between the ith and jth document
# X = ...

# Train-test split
X_train_lch, X_test_lch, y_train_lch, y_test_lch = train_test_split(X_lch, y_encoded, test_size=0.2, random_state=42)

y_binarized_lch = label_binarize(y_test_lch, classes=np.unique(y_encoded))

# Train a RandomForest Classifier
model = RandomForestClassifier()
model.fit(X_train_lch, y_train_lch)

# Predict probabilities
predicted_probs_lch = model.predict_proba(X_test_lch)

# Compute ROC curve and ROC area for each class
fpr_lch, tpr_lch, roc_thresholds_lch = roc_curve(y_binarized_lch.ravel(), predicted_probs_lch.ravel())
roc_auc_lch = auc(fpr_lch, tpr_lch)

# Compute the average precision score (micro-averaged)
average_precision_lch = average_precision_score(y_binarized_lch, predicted_probs_lch, average="micro")

# Compute the micro-averaged Precision-Recall curve and area under the curve
precision_lch, recall_lch,pr_thresholds_lch = precision_recall_curve(y_binarized_lch.ravel(), predicted_probs_lch.ravel())
#print("Precision_lch:", precision_lch)
#print("Recall_lch:", recall_lch)
#print("Thresholds_lch:" ,pr_thresholds_lch)


# Assuming precision_lch, recall_lch, and pr_thresholds_lch are already defined
pr_thresholds_lch = np.append(pr_thresholds_lch, np.nan)

data = {
    "Precision_lch": precision_lch,
    "Recall_lch": recall_lch,
    "Thresholds_lch": pr_thresholds_lch
}

# Creating a DataFrame
df = pd.DataFrame(data)


# Printing the DataFrame
print(df.to_string(index=False))

# Compute Precision-Recall curve and average precision
#precision, recall, _ = precision_recall_curve(y_test, predicted_probs[:, 1])
#precision, recall, _ = precision_recall_curve(y_binarized.ravel(), predicted_probs.ravel())

#average_precision = average_precision_score(y_test, predicted_probs[:, 1])
interpolated_precision_lch = np.maximum.accumulate(precision_lch[::1])[::1]






# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_lch_data = {'Precision_lch': precision_lch, 'Recall_lch': recall_lch}# , 'Thresholds_lch': pr_thresholds_lch}
roc_data_lch = {'FPR_lch': fpr_lch, 'TPR_lch': tpr_lch }#, 'Thresholds_lch': roc_thresholds_lch}

precision_recall_lch_df = pd.DataFrame(precision_recall_lch_data)
roc_lch_df = pd.DataFrame(roc_data_lch)

# Print the dataframes
print("Precision-Recall_lch Data:")
print(precision_recall_lch_df.to_markdown())

print("\nROC_lch Data:")
print(roc_lch_df.to_markdown())


# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_wu_palmer_data = {'Precision_wu_palmer': precision_wu_palmer, 'Recall_wu_palmer': recall_wu_palmer}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_wu_palmer_data = {'FPR_wu_palmer': fpr_wu_palmer, 'TPR_wu_palmer': tpr_wu_palmer}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_wu_palmer_df = pd.DataFrame(precision_recall_wu_palmer_data)
roc_wu_palmer_df = pd.DataFrame(roc_wu_palmer_data)

# Print the dataframes
print("Precision-Recall Data:")
print(precision_recall_wu_palmer_df.to_markdown())

print("\nROC Data:")
print(roc_wu_palmer_df.to_markdown())

# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_path_data = {'Precision_path': precision_path, 'Recall_path': recall_path}# , 'Thresholds_path': pr_thresholds_path}
roc_data_path = {'FPR_path': fpr_path, 'TPR_path': tpr_path }#, 'Thresholds_path': roc_thresholds_path}

precision_recall_path_df = pd.DataFrame(precision_recall_path_data)
roc_path_df = pd.DataFrame(roc_data_path)

# Print the dataframes
print("Precision-Recall Data:")
print(precision_recall_path_df.to_markdown())

print("\nROC Data:")
print(roc_path_df.to_markdown())


precision_recall_res_data = {'Precision_res': precision_res, 'Recall_res': recall_res}#, 'Thresholds_res': pr_thresholds_res}
roc_data_res = {'FPR_res': fpr_res, 'TPR_res': tpr_res}#, 'Thresholds_res': roc_thresholds_res}

# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_res_df = pd.DataFrame(precision_recall_res_data)
roc_res_df = pd.DataFrame(roc_data_res)

# Print the dataframes
print("Precision-Recall res Data:")
print(precision_recall_res_df.to_markdown())

print("\nROC res Data:")
print(roc_res_df.to_markdown())



# Print the dataframes

combined_ROC = pd.concat([roc_wu_palmer_df, roc_path_df,roc_res_df,roc_lch_df], axis=1)
print("\nROC_lch Data:")
print(combined_ROC.to_markdown())

combined_Precision_Recall = pd.concat([precision_recall_lch_df, precision_recall_res_df,precision_recall_wu_palmer_df,precision_recall_path_df], axis=1)
print("Precision-Recall_lch Data:")
print(combined_Precision_Recall.to_markdown())


# Plotting Precision-Recall Curve
plt.figure()
plt.step(recall_lch, interpolated_precision_lch-0.0003, where='post', label=f'lch Precision-Recall (AP = {average_precision_lch:.2f})')
plt.step(recall_path, interpolated_precision_path+0.0005, where='post', label=f'path Precision-Recall (AP = {average_precision_path:.2f})')
plt.step(recall_res, interpolated_precision_res-0.0008, where='post', label=f'res Precision-Recall _res (AP = {average_precision_res:.2f})')
plt.step(recall_wu_palmer, interpolated_precision_wu_palmer+0.0008, where='post', label=f'wu-palmer Precision-Recall (AP = {average_precision_wu_palmer:.2f})')

plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")



#plt.figure()
#plt.step(recall_lch, precision_lch, where='post', label=f'lch Precision-Recall (AP = {average_precision_lch:.2f})')
#plt.step(recall_path, precision_path-0.0005, where='post', label=f'path Precision-Recall (AP = {average_precision_path:.2f})')
#plt.step(recall_res, precision_res+0.0008, where='post', label=f'res Precision-Recall _res (AP = {average_precision_res:.2f})')
#plt.step(recall_wu_palmer, precision_wu_palmer-0.0008, where='post', label=f'wu-palmer Precision-Recall (AP = {average_precision_wu_palmer:.2f})')

#plt.title('Precision-Recall curve')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.legend(loc="lower right")


# Plotting ROC Curve
plt.figure()
plt.plot(fpr_lch, tpr_lch, color='blue', lw=2, label=f'lch ROC curve (area = {roc_auc_lch:.2f})')
plt.plot(fpr_path, tpr_path, color='red', lw=2, label=f'path ROC curve (area = {roc_auc_path:.2f})')
plt.plot(fpr_res, tpr_res, color='orange', lw=2, label=f'res ROC curve (area = {roc_auc_res:.2f})')
plt.plot(fpr_wu_palmer, tpr_wu_palmer, color='green', lw=2, label=f'wu-palmer ROC curve (area = {roc_auc_wu_palmer:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
