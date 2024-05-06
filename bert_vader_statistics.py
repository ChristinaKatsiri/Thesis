import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.preprocessing import label_binarize, LabelEncoder


# Load the dataset
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Hotel_Reviews_1.csv.csv", sep = ',')
reviews_df["review_total"] = reviews_df["review_total"].astype("str")
reviews_df = reviews_df.drop(['compound'], axis=1)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
#create a new column (sentiments that keeps data created by polarity feature, score based the commends)
reviews_df["sentiments"] = reviews_df["review_total"].apply(lambda x: sid.polarity_scores(x))
#create new columns from the existing sentiment one that has already info  (create new columns for each value included in sentiments)
reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)
num_unique_scores = 20


#----find the review tag based on compound-----
def find_review_tag(text):
    if text >= 0.05 :
        return("Positive")

    elif text <= - 0.05 :
        return("Negative")

    else :
        return("Neutral")

print(reviews_df["compound"])
reviews_df["review_tag_vader_new"] = reviews_df["compound"].apply(lambda x: find_review_tag(x))
predicted_labels = reviews_df["review_tag_vader_new"]

# Encode labels into numerical form
label_encoder = LabelEncoder()
reviews_df["true_labels_encoded"] = label_encoder.fit_transform(reviews_df["review_tag_vader"])
reviews_df["predicted_labels_encoded"] = label_encoder.transform(reviews_df["review_tag_vader_new"])

print('true_labels_encoded', reviews_df["true_labels_encoded"])
print('predicted_labels_encoded', reviews_df["predicted_labels_encoded"])

# Binarize true labels and predicted labels
true_labels_binarized = label_binarize(reviews_df["true_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
predicted_labels_binarized = label_binarize(reviews_df["predicted_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
print('true_labels_binarized', true_labels_binarized)
print('predicted_labels_binarized', predicted_labels_binarized)
# Calculate precision and recall for all classes combined
precision_1_vader, recall_1_vader, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_1_vader) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_1_vader) - 1, num_unique_scores, dtype=int)
    precision_1_vader = precision_1_vader[sample_indices]
    recall_1_vader = recall_1_vader[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_1_vader, tpr_1_vader, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_1_vader = auc(fpr_1_vader, tpr_1_vader)
interpolated_precision_1_vader_vader = np.maximum.accumulate(precision_1_vader[::1])[::1]

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_1_vader) - 1, num_unique_scores, dtype=int)
fpr_1_vader = fpr_1_vader[sample_indices]
tpr_1_vader = tpr_1_vader[sample_indices]

reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Hotel_Reviews_2.csv", sep = ',')
reviews_df["review_total"] = reviews_df["review_total"].astype("str")
reviews_df = reviews_df.drop(['compound'], axis=1)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
#create a new column (sentiments that keeps data created by polarity feature, score based the commends)
reviews_df["sentiments"] = reviews_df["review_total"].apply(lambda x: sid.polarity_scores(x))
#create new columns from the existing sentiment one that has already info  (create new columns for each value included in sentiments)
reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)


#----find the review tag based on compound-----
def find_review_tag(text):
    if text >= 0.05 :
        return("Positive")

    elif text <= - 0.05 :
        return("Negative")

    else :
        return("Neutral")

reviews_df["review_tag_vader_new"] = reviews_df["compound"].apply(lambda x: find_review_tag(x))
predicted_labels = reviews_df["review_tag_vader_new"]
print(predicted_labels)

# Encode labels into numerical form
label_encoder = LabelEncoder()
reviews_df["true_labels_encoded"] = label_encoder.fit_transform(reviews_df["review_tag_vader"])
reviews_df["predicted_labels_encoded"] = label_encoder.transform(reviews_df["review_tag_vader_new"])
print(reviews_df["predicted_labels_encoded"])
# Binarize true labels and predicted labels
true_labels_binarized = label_binarize(reviews_df["true_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
predicted_labels_binarized = label_binarize(reviews_df["predicted_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))

# Calculate precision and recall for all classes combined
precision_2_vader, recall_2_vader, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
interpolated_precision_2_vader = np.maximum.accumulate(precision_2_vader[::1])[::1]
# Sample the precision and recall values to get a fixed number of points
if len(recall_2_vader) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_2_vader) - 1, num_unique_scores, dtype=int)
    precision_2_vader = precision_2_vader[sample_indices]
    recall_2_vader = recall_2_vader[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_2_vader, tpr_2_vader, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_2_vader = auc(fpr_2_vader, tpr_2_vader)

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_2_vader) - 1, num_unique_scores, dtype=int)
fpr_2_vader = fpr_2_vader[sample_indices]
tpr_2_vader = tpr_2_vader[sample_indices]



#---tables
# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_1_vader_data = {'precision_1_vader': precision_1_vader, 'recall_1_vader': recall_1_vader}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_1_data_vader = {'fpr_1_vader': fpr_1_vader, 'tpr_1_vader': tpr_1_vader}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_1_vader_df = pd.DataFrame(precision_recall_1_vader_data)
roc_1_df_vader = pd.DataFrame(roc_1_data_vader)

# Print the dataframes
print("Precision-Recall Data:")
print(precision_recall_1_vader_df.to_markdown())

print("\nROC Data:")
print(roc_1_df_vader.to_markdown())


# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_2_vader_data = {'precision_2_vader': precision_2_vader, 'recall_2_vader': recall_2_vader}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_2_data_vader = {'fpr_2_vader': fpr_2_vader, 'tpr_2_vader': tpr_2_vader}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_2_vader_df = pd.DataFrame(precision_recall_2_vader_data)
roc_2_df_vader = pd.DataFrame(roc_2_data_vader)

# Print the dataframes
print("Precision-Recall Data:")
print(precision_recall_1_vader_df.to_markdown())

print("\nROC Data:")
print(roc_1_df_vader.to_markdown())




# Print the dataframes

combined_ROC = pd.concat([roc_1_df_vader, roc_2_df_vader], axis=1)
print("\nROC_vader Data:")
print(combined_ROC.to_markdown())

combined_Precision_Recall = pd.concat([precision_recall_1_vader_df, precision_recall_2_vader_df], axis=1)
print("Precision-Recall_vader Data:")
print(combined_Precision_Recall.to_markdown())


# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# Load the dataset
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Hotel_Reviews_2.csv.csv", sep = ',')
reviews_df["review_total"] = reviews_df["review_total"].astype("str")
#reviews_df = reviews_df.drop(['bert_sentiment_score'], axis=1)
# Obtain sentiment scores using BERT model
reviews_df["bert_sentiment_score"] = reviews_df["review_total"].apply(sentiment_score)
print('bert_sentiment_score' ,reviews_df["bert_sentiment_score"])

num_unique_scores = 50
#----find the review tag based on score-----
def find_review_tag_bert(score):
    if score > 3 :
        return("Positive")
    elif score < 1 :
        return("Negative")

    else :
        return("Neutral")

reviews_df["review_tag_bert_new"] = reviews_df["bert_sentiment_score"].apply(lambda x: find_review_tag_bert(x))
predicted_labels = reviews_df["review_tag_bert_new"]

# Encode labels into numerical form
label_encoder = LabelEncoder()
reviews_df["true_labels_encoded"] = label_encoder.fit_transform(reviews_df["review_tag_vader"])
reviews_df["predicted_labels_encoded"] = label_encoder.transform(reviews_df["review_tag_bert_new"])

# Binarize true labels and predicted labels
true_labels_binarized = label_binarize(reviews_df["true_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
predicted_labels_binarized = label_binarize(reviews_df["predicted_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))


# Calculate precision and recall for all classes combined
precision_1_bert, recall_1_bert, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_1_bert) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_1_bert) - 1, num_unique_scores, dtype=int)
    precision_1_bert = precision_1_bert[sample_indices]
    recall_1_bert = recall_1_bert[sample_indices]

# Calculate ROC curve and AUC for all classes combined
fpr_1_bert, tpr_1_bert, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_1_bert = auc(fpr_1_bert, tpr_1_bert)
interpolated_precision_1_bert = np.maximum.accumulate(precision_1_bert[::1])[::1]

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_1_bert) - 1, num_unique_scores, dtype=int)
fpr_1_bert = fpr_1_bert[sample_indices]
tpr_1_bert = tpr_1_bert[sample_indices]



# Load the dataset
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Hotel_Reviews_1.csv.csv", sep = ',')
reviews_df["review_total"] = reviews_df["review_total"].astype("str")
#reviews_df = reviews_df.drop(['bert_sentiment_score'], axis=1)
# Obtain sentiment scores using BERT model
reviews_df["bert_sentiment_score"] = reviews_df["review_total"].apply(sentiment_score)




#----find the review tag based on score-----
def find_review_tag_bert(score):
    if score > 3 :
        return("Positive")
    elif score < 1 :
        return("Negative")

    else :
        return("Neutral")

reviews_df["review_tag_bert_new"] = reviews_df["bert_sentiment_score"].apply(lambda x: find_review_tag_bert(x))
predicted_labels = reviews_df["review_tag_bert_new"]

# Encode labels into numerical form
label_encoder = LabelEncoder()
reviews_df["true_labels_encoded"] = label_encoder.fit_transform(reviews_df["review_tag_vader"])
reviews_df["predicted_labels_encoded"] = label_encoder.transform(reviews_df["review_tag_bert_new"])

# Binarize true labels and predicted labels
true_labels_binarized = label_binarize(reviews_df["true_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
predicted_labels_binarized = label_binarize(reviews_df["predicted_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))

# Calculate precision and recall for all classes combined
precision_2_bert, recall_2_bert, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_2_bert) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_2_bert) - 1, num_unique_scores, dtype=int)
    precision_2_bert = precision_2_bert[sample_indices]
    recall_2_bert = recall_2_bert[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_2_bert, tpr_2_bert, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_2_bert = auc(fpr_2_bert, tpr_2_bert)
interpolated_precision_2_bert = np.maximum.accumulate(precision_2_bert[::1])[::1]
# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_2_bert) - 1, num_unique_scores, dtype=int)
fpr_2_bert = fpr_2_bert[sample_indices]
tpr_2_bert = tpr_2_bert[sample_indices]

#---tables
# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_1_bert_data = {'precision_1_bert': precision_1_bert, 'recall_1_bert': recall_1_bert}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_1_data_bert = {'fpr_1_bert': fpr_1_bert, 'tpr_1_bert': tpr_1_bert}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_1_bert_df = pd.DataFrame(precision_recall_1_bert_data)
roc_1_df_bert = pd.DataFrame(roc_1_data_bert)

# Print the dataframes
print("Precision-Recall Data:")
print(precision_recall_1_bert_df.to_markdown())

print("\nROC Data:")
print(roc_1_df_bert.to_markdown())


# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_2_bert_data = {'precision_2_bert': precision_2_bert, 'recall_2_bert': recall_2_bert}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_2_data_bert = {'fpr_2_bert': fpr_2_bert, 'tpr_2_bert': tpr_2_bert}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_2_bert_df = pd.DataFrame(precision_recall_2_bert_data)
roc_2_df_bert = pd.DataFrame(roc_2_data_bert)

# Print the dataframes
print("Precision-Recall Data:")
print(precision_recall_1_bert_df.to_markdown())

print("\nROC Data:")
print(roc_1_df_bert.to_markdown())




# Print the dataframes

combined_ROC = pd.concat([roc_1_df_bert, roc_2_df_bert], axis=1)
print("\nROC_lch Data:")
print(combined_ROC.to_markdown())

combined_Precision_Recall = pd.concat([precision_recall_1_bert_df, precision_recall_2_bert_df], axis=1)
print("Precision-Recall_lch Data:")
print(combined_Precision_Recall.to_markdown())



# Plot precision-recall curve
plt.figure()
plt.plot(recall_1_bert, interpolated_precision_1_bert+0.009, color='darkorange', label='Interpolated Precision-Recall curve Bert (area = %0.2f)' % auc(recall_1_bert, precision_1_bert))
plt.plot(recall_1_vader, interpolated_precision_1_vader_vader, color='blue', label='Interpolated Precision-Recall curve Vader(area = %0.2f)' % auc(recall_1_vader, precision_1_vader))

plt.xlabel('Recall')
plt.ylabel('Interpolated Precision')
plt.title('Interpolated Precision-Recall Curve 1')
plt.legend(loc="lower right")

# Plot ROC curve
plt.figure()
plt.plot(fpr_1_vader, tpr_1_vader, color='blue', lw=2, label='ROC curve Vader(area = %0.2f)' % roc_auc_1_vader)
plt.plot(fpr_1_bert, tpr_1_bert+0.009, color='darkorange', lw=2, label='ROC curve Bert(area = %0.2f)' % roc_auc_1_bert)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve 1')
plt.legend(loc="lower right")
plt.show()

# Plot precision-recall curve
plt.figure()
plt.plot(recall_2_vader, interpolated_precision_2_vader, color='blue', label='Interpolated Precision-Recall curve Vader(area = %0.2f)' % auc(recall_2_vader, precision_2_vader))
plt.plot(recall_2_bert, interpolated_precision_2_bert+0.009, color='darkorange', label='Interpolated Precision-Recall curve Bert(area = %0.2f)' % auc(recall_2_bert, precision_2_bert))

plt.xlabel('Recall')
plt.ylabel('Interpolated Precision')
plt.title('Interpolated Precision-Recall Curve 2')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr_2_vader, tpr_2_vader, color='blue', lw=2, label='ROC curve Vader(area = %0.2f)' % roc_auc_2_vader)
plt.plot(fpr_2_bert, tpr_2_bert+0.009, color='darkorange', lw=2, label='ROC curve Bert(area = %0.2f)' % roc_auc_2_bert)


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve 2')
plt.legend(loc="lower right")
plt.show()
