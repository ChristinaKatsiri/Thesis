import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.preprocessing import label_binarize, LabelEncoder

###VADER###
# Load the dataset ----1a-vader--
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Modified_Hotel_Reviews_4_total_vader.csv", sep = ',')
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

#print(reviews_df["compound"])
reviews_df["review_tag_vader_new"] = reviews_df["compound"].apply(lambda x: find_review_tag(x))
predicted_labels = reviews_df["review_tag_vader_new"]

# Encode labels into numerical form
label_encoder = LabelEncoder()
reviews_df["true_labels_encoded"] = label_encoder.fit_transform(reviews_df["review_tag_vader"])
reviews_df["predicted_labels_encoded"] = label_encoder.transform(reviews_df["review_tag_vader_new"])




# Binarize true labels and predicted labels
true_labels_binarized = label_binarize(reviews_df["true_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
predicted_labels_binarized = label_binarize(reviews_df["predicted_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))


# Calculate precision and recall for all classes combined
precision_1a_vader, recall_1a_vader, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_1a_vader) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_1a_vader) - 1, num_unique_scores, dtype=int)
    precision_1a_vader = precision_1a_vader[sample_indices]
    recall_1a_vader = recall_1a_vader[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_1a_vader, tpr_1a_vader, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_1a_vader = auc(fpr_1a_vader, tpr_1a_vader)
interpolated_precision_1a_vader = np.maximum.accumulate(precision_1a_vader[::1])[::1]

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_1a_vader) - 1, num_unique_scores, dtype=int)
fpr_1a_vader = fpr_1a_vader[sample_indices]
tpr_1a_vader = tpr_1a_vader[sample_indices]

# Load the dataset ----1b-vader--
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Modified_Hotel_Reviews_4_total_vader.csv", sep = ',')
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
    if text >= 0.02 :
        return("Positive")

    elif text <= - 0.02 :
        return("Negative")

    else :
        return("Neutral")

#print(reviews_df["compound"])
reviews_df["review_tag_vader_new"] = reviews_df["compound"].apply(lambda x: find_review_tag(x))
predicted_labels = reviews_df["review_tag_vader_new"]

# Encode labels into numerical form
label_encoder = LabelEncoder()
reviews_df["true_labels_encoded"] = label_encoder.fit_transform(reviews_df["review_tag_vader"])
reviews_df["predicted_labels_encoded"] = label_encoder.transform(reviews_df["review_tag_vader_new"])


# Binarize true labels and predicted labels
true_labels_binarized = label_binarize(reviews_df["true_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
predicted_labels_binarized = label_binarize(reviews_df["predicted_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))

# Calculate precision and recall for all classes combined
precision_1b_vader, recall_1b_vader, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_1b_vader) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_1b_vader) - 1, num_unique_scores, dtype=int)
    precision_1b_vader = precision_1b_vader[sample_indices]
    recall_1b_vader = recall_1b_vader[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_1b_vader, tpr_1b_vader, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_1b_vader = auc(fpr_1b_vader, tpr_1b_vader)
interpolated_precision_1b_vader = np.maximum.accumulate(precision_1b_vader[::1])[::1]

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_1b_vader) - 1, num_unique_scores, dtype=int)
fpr_1b_vader = fpr_1b_vader[sample_indices]
tpr_1b_vader = tpr_1b_vader[sample_indices]




# Load the dataset ----2a-vader--

reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Modified_Hotel_Reviews_6_total_vader.csv", sep = ',')
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
#print(predicted_labels)

# Encode labels into numerical form
label_encoder = LabelEncoder()
reviews_df["true_labels_encoded"] = label_encoder.fit_transform(reviews_df["review_tag_vader"])
reviews_df["predicted_labels_encoded"] = label_encoder.transform(reviews_df["review_tag_vader_new"])
# Binarize true labels and predicted labels
true_labels_binarized = label_binarize(reviews_df["true_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
predicted_labels_binarized = label_binarize(reviews_df["predicted_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))

# Calculate precision and recall for all classes combined
precision_2a_vader, recall_2a_vader, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
interpolated_precision_2a_vader = np.maximum.accumulate(precision_2a_vader[::1])[::1]
# Sample the precision and recall values to get a fixed number of points
if len(recall_2a_vader) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_2a_vader) - 1, num_unique_scores, dtype=int)
    precision_2a_vader = precision_2a_vader[sample_indices]
    recall_2a_vader = recall_2a_vader[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_2a_vader, tpr_2a_vader, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_2a_vader = auc(fpr_2a_vader, tpr_2a_vader)

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_2a_vader) - 1, num_unique_scores, dtype=int)
fpr_2a_vader = fpr_2a_vader[sample_indices]
tpr_2a_vader = tpr_2a_vader[sample_indices]

# Load the dataset ----2b-vader--
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Modified_Hotel_Reviews_6_total_vader.csv", sep = ',')
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
    if text >= 0.02 :
        return("Positive")

    elif text <= - 0.02 :
        return("Negative")

    else :
        return("Neutral")

reviews_df["review_tag_vader_new"] = reviews_df["compound"].apply(lambda x: find_review_tag(x))
predicted_labels = reviews_df["review_tag_vader_new"]

# Encode labels into numerical form
label_encoder = LabelEncoder()
reviews_df["true_labels_encoded"] = label_encoder.fit_transform(reviews_df["review_tag_vader"])
reviews_df["predicted_labels_encoded"] = label_encoder.transform(reviews_df["review_tag_vader_new"])


# Binarize true labels and predicted labels
true_labels_binarized = label_binarize(reviews_df["true_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))
predicted_labels_binarized = label_binarize(reviews_df["predicted_labels_encoded"], classes=np.unique(reviews_df["true_labels_encoded"]))

# Calculate precision and recall for all classes combined
precision_2b_vader, recall_2b_vader, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_2b_vader) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_2b_vader) - 1, num_unique_scores, dtype=int)
    precision_2b_vader = precision_2b_vader[sample_indices]
    recall_2b_vader = recall_2b_vader[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_2b_vader, tpr_2b_vader, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_2b_vader = auc(fpr_2b_vader, tpr_2b_vader)
interpolated_precision_2b_vader = np.maximum.accumulate(precision_2b_vader[::1])[::1]

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_2b_vader) - 1, num_unique_scores, dtype=int)
fpr_2b_vader = fpr_1b_vader[sample_indices]
tpr_2b_vader = tpr_1b_vader[sample_indices]




#---tables
# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_1a_vader_data = {'precision_1a_vader': precision_1a_vader, 'recall_1a_vader': recall_1a_vader}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_1a_data_vader = {'fpr_1a_vader': fpr_1a_vader, 'tpr_1a_vader': tpr_1a_vader}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_1a_vader_df = pd.DataFrame(precision_recall_1a_vader_data)
roc_1a_df_vader = pd.DataFrame(roc_1a_data_vader)

# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_1a_vader_df.to_markdown())

#print("\nROC Data:")
#print(roc_1a_df_vader.to_markdown())

# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_1b_vader_data = {'precision_1b_vader': precision_1b_vader, 'recall_1b_vader': recall_1b_vader}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_1b_data_vader = {'fpr_1b_vader': fpr_1b_vader, 'tpr_1b_vader': tpr_1b_vader}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_1b_vader_df = pd.DataFrame(precision_recall_1b_vader_data)
roc_1b_df_vader = pd.DataFrame(roc_1b_data_vader)

# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_1b_vader_df.to_markdown())

#print("\nROC Data:")
#print(roc_1b_df_vader.to_markdown())

# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_2a_vader_data = {'precision_2a_vader': precision_2a_vader, 'recall_2a_vader': recall_2a_vader}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_2a_data_vader = {'fpr_2a_vader': fpr_2a_vader, 'tpr_2a_vader': tpr_2a_vader}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_2a_vader_df = pd.DataFrame(precision_recall_2a_vader_data)
roc_2a_df_vader = pd.DataFrame(roc_2a_data_vader)

# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_1a_vader_df.to_markdown())

#print("\nROC Data:")
#print(roc_1a_df_vader.to_markdown())


# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_2b_vader_data = {'precision_2b_vader': precision_2b_vader, 'recall_2b_vader': recall_2b_vader}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_2b_data_vader = {'fpr_2b_vader': fpr_2b_vader, 'tpr_2b_vader': tpr_2b_vader}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_2b_vader_df = pd.DataFrame(precision_recall_2b_vader_data)
roc_2b_df_vader = pd.DataFrame(roc_2b_data_vader)

# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_2a_vader_df.to_markdown())

#print("\nROC Data:")
#print(roc_2a_df_vader.to_markdown())


# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_2b_vader_df.to_markdown())

#print("\nROC Data:")
#print(roc_2b_df_vader.to_markdown())


# Print the dataframes

combined_ROC = pd.concat([roc_1a_df_vader, roc_1b_df_vader,roc_2a_df_vader,roc_2b_df_vader], axis=1)
print("\nROC_vader Data:")
print(combined_ROC.to_markdown())

combined_Precision_Recall = pd.concat([precision_recall_1a_vader_df, precision_recall_1b_vader_df ,precision_recall_2a_vader_df, precision_recall_2b_vader_df ], axis=1)
print("Precision-Recall_vader Data:")
print(combined_Precision_Recall.to_markdown())






###BERT###

# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# Load the dataset ---1a--bert
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Modified_Hotel_Reviews_6_total_vader.csv", sep = ',')
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
precision_1a_bert, recall_1a_bert, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_1a_bert) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_1a_bert) - 1, num_unique_scores, dtype=int)
    precision_1a_bert = precision_1a_bert[sample_indices]
    recall_1a_bert = recall_1a_bert[sample_indices]

# Calculate ROC curve and AUC for all classes combined
fpr_1a_bert, tpr_1a_bert, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_1a_bert = auc(fpr_1a_bert, tpr_1a_bert)
interpolated_precision_1a_bert = np.maximum.accumulate(precision_1a_bert[::1])[::1]

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_1a_bert) - 1, num_unique_scores, dtype=int)
fpr_1a_bert = fpr_1a_bert[sample_indices]
tpr_1a_bert = tpr_1a_bert[sample_indices]


# Load the dataset ---1b--bert
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Modified_Hotel_Reviews_6_total_vader.csv", sep = ',')
reviews_df["review_total"] = reviews_df["review_total"].astype("str")
#reviews_df = reviews_df.drop(['bert_sentiment_score'], axis=1)
# Obtain sentiment scores using BERT model
reviews_df["bert_sentiment_score"] = reviews_df["review_total"].apply(sentiment_score)
print('bert_sentiment_score' ,reviews_df["bert_sentiment_score"])

num_unique_scores = 50
#----find the review tag based on score-----
def find_review_tag_bert(score):
    if score > 4 :
        return("Positive")
    elif score < 2 :
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
precision_1b_bert, recall_1b_bert, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_1b_bert) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_1b_bert) - 1, num_unique_scores, dtype=int)
    precision_1b_bert = precision_1b_bert[sample_indices]
    recall_1b_bert = recall_1b_bert[sample_indices]

# Calculate ROC curve and AUC for all classes combined
fpr_1b_bert, tpr_1b_bert, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_1b_bert = auc(fpr_1b_bert, tpr_1b_bert)
interpolated_precision_1b_bert = np.maximum.accumulate(precision_1b_bert[::1])[::1]

# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_1b_bert) - 1, num_unique_scores, dtype=int)
fpr_1b_bert = fpr_1b_bert[sample_indices]
tpr_1b_bert = tpr_1b_bert[sample_indices]


# Load the dataset ----2a--bert
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Modified_Hotel_Reviews_4_total_vader.csv", sep = ',')
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
precision_2a_bert, recall_2a_bert, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_2a_bert) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_2a_bert) - 1, num_unique_scores, dtype=int)
    precision_2a_bert = precision_2a_bert[sample_indices]
    recall_2a_bert = recall_2a_bert[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_2a_bert, tpr_2a_bert, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_2a_bert = auc(fpr_2a_bert, tpr_2a_bert)
interpolated_precision_2a_bert = np.maximum.accumulate(precision_2a_bert[::1])[::1]
# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_2a_bert) - 1, num_unique_scores, dtype=int)
fpr_2a_bert = fpr_2a_bert[sample_indices]
tpr_2a_bert = tpr_2a_bert[sample_indices]


# Load the dataset ----2a--bert
reviews_df = pd.read_csv("C:/Users/katsi/Desktop/paradotea_local/Modified_Hotel_Reviews_4_total_vader.csv", sep = ',')
reviews_df["review_total"] = reviews_df["review_total"].astype("str")
#reviews_df = reviews_df.drop(['bert_sentiment_score'], axis=1)
# Obtain sentiment scores using BERT model
reviews_df["bert_sentiment_score"] = reviews_df["review_total"].apply(sentiment_score)




#----find the review tag based on score-----
def find_review_tag_bert(score):
    if score > 4 :
        return("Positive")
    elif score < 2 :
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
precision_2b_bert, recall_2b_bert, _ = precision_recall_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())

# Sample the precision and recall values to get a fixed number of points
if len(recall_2b_bert) > num_unique_scores:
    sample_indices = np.linspace(0, len(recall_2b_bert) - 1, num_unique_scores, dtype=int)
    precision_2b_bert = precision_2b_bert[sample_indices]
    recall_2b_bert = recall_2b_bert[sample_indices]
    
# Calculate ROC curve and AUC for all classes combined
fpr_2b_bert, tpr_2b_bert, _ = roc_curve(true_labels_binarized.ravel(), predicted_labels_binarized.ravel())
roc_auc_2b_bert = auc(fpr_2b_bert, tpr_2b_bert)
interpolated_precision_2b_bert = np.maximum.accumulate(precision_2b_bert[::1])[::1]
# Sample the FPR and TPR values to get a fixed number of points
sample_indices = np.linspace(0, len(fpr_2b_bert) - 1, num_unique_scores, dtype=int)
fpr_2b_bert = fpr_2b_bert[sample_indices]
tpr_2b_bert = tpr_2b_bert[sample_indices]


#---tables
# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_1a_bert_data = {'precision_1a_bert': precision_1a_bert, 'recall_1a_bert': recall_1a_bert}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_1a_data_bert = {'fpr_1a_bert': fpr_1a_bert, 'tpr_1a_bert': tpr_1a_bert}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_1a_bert_df = pd.DataFrame(precision_recall_1a_bert_data)
roc_1a_df_bert = pd.DataFrame(roc_1a_data_bert)

# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_1a_bert_df.to_markdown())

#print("\nROC Data:")
#print(roc_1a_df_bert.to_markdown())


# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_1b_bert_data = {'precision_1b_bert': precision_1b_bert, 'recall_1b_bert': recall_1b_bert}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_1b_data_bert = {'fpr_1b_bert': fpr_1b_bert, 'tpr_1b_bert': tpr_1b_bert}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_1b_bert_df = pd.DataFrame(precision_recall_1b_bert_data)
roc_1b_df_bert = pd.DataFrame(roc_1b_data_bert)

# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_1b_bert_df.to_markdown())

#print("\nROC Data:")
#print(roc_1b_df_bert.to_markdown())




# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_2a_bert_data = {'precision_2a_bert': precision_2a_bert, 'recall_2a_bert': recall_2a_bert}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_2a_data_bert = {'fpr_2a_bert': fpr_2a_bert, 'tpr_2a_bert': tpr_2a_bert}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_2a_bert_df = pd.DataFrame(precision_recall_2a_bert_data)
roc_2a_df_bert = pd.DataFrame(roc_2a_data_bert)

# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_1a_bert_df.to_markdown())

#print("\nROC Data:")
#print(roc_1a_df_bert.to_markdown())


# You can convert these data structures to dataframes for easier manipulation or export
precision_recall_2b_bert_data = {'precision_2b_bert': precision_2b_bert, 'recall_2b_bert': recall_2b_bert}# , 'Thresholds_wu_palmer': pr_thresholds_wu_palmer}
roc_2b_data_bert = {'fpr_2b_bert': fpr_2b_bert, 'tpr_2b_bert': tpr_2b_bert}#, 'Thresholds_wu_palmer': roc_thresholds_wu_palmer}

precision_recall_2b_bert_df = pd.DataFrame(precision_recall_2b_bert_data)
roc_2b_df_bert = pd.DataFrame(roc_2b_data_bert)

# Print the dataframes
#print("Precision-Recall Data:")
#print(precision_recall_1b_bert_df.to_markdown())

#print("\nROC Data:")
#print(roc_1b_df_bert.to_markdown())




# Print the dataframes

combined_ROC = pd.concat([roc_1a_df_bert,roc_1b_df_bert, roc_2a_df_bert,roc_2b_df_bert ], axis=1)
print("\nROC_lch Data:")
print(combined_ROC.to_markdown())

combined_Precision_Recall = pd.concat([precision_recall_1a_bert_df, precision_recall_1b_bert_df, precision_recall_2a_bert_df,precision_recall_2b_bert_df], axis=1)
print("Precision-Recall_lch Data:")
print(combined_Precision_Recall.to_markdown())

recall_1_bert= recall_1b_bert + recall_1a_bert
interpolated_precision_1_bert=interpolated_precision_1a_bert + interpolated_precision_1b_bert
precision_1_bert = precision_1a_bert + precision_1b_bert
recall_1_vader = recall_1b_vader + recall_1a_vader
interpolated_precision_1_vader = interpolated_precision_1a_vader + interpolated_precision_1b_vader
precision_1_vader = precision_1a_vader + precision_1b_vader
fpr_1_vader = fpr_1a_vader + fpr_1b_vader
tpr_1_vader = tpr_1a_vader + tpr_1b_vader
fpr_1_bert = fpr_1a_bert + fpr_1b_bert
tpr_1_bert = tpr_1a_bert + tpr_1b_bert



recall_2_bert = recall_2b_bert + recall_2a_bert
interpolated_precision_2_bert=interpolated_precision_2a_bert + interpolated_precision_2b_bert
precision_2_bert = precision_2a_bert + precision_2b_bert
recall_2_vader = recall_2b_vader + recall_2a_vader
interpolated_precision_2_vader = interpolated_precision_2a_vader + interpolated_precision_2b_vader
precision_2_vader = precision_2a_vader + precision_2b_vader
fpr_2_vader = fpr_2a_vader + fpr_2b_vader
tpr_2_vader = tpr_2a_vader + tpr_2b_vader
fpr_2_bert = fpr_2a_bert + fpr_2b_bert
tpr_2_bert = tpr_2a_bert + tpr_2b_bert
roc_auc_1_vader = roc_auc_1a_vader + roc_auc_1b_vader
roc_auc_2_vader = roc_auc_2a_vader + roc_auc_2b_vader
roc_auc_1_bert = roc_auc_1a_bert + roc_auc_1b_bert
roc_auc_2_bert =roc_auc_2a_bert + roc_auc_2b_bert
#roc_1_df_bert 
#roc_1_df_bert

# Print the dataframes

#combined_ROC = pd.concat([roc_1_df_bert, roc_1_df_bert], axis=1)
#print("\nROC_lch Data:")
#print(combined_ROC.to_markdown())

#combined_Precision_Recall = pd.concat([precision_recall_1_bert_df, precision_recall_2_bert_df], axis=1)
#print("Precision-Recall_lch Data:")
#print(combined_Precision_Recall.to_markdown())

# Plot precision-recall curve
plt.figure()
plt.plot(recall_1_bert, interpolated_precision_1_bert+0.009, color='darkorange', label='Interpolated Precision-Recall curve Bert (area = %0.2f)' % auc(recall_1_bert, precision_1_bert))
plt.plot(recall_1_vader, interpolated_precision_1_vader, color='blue', label='Interpolated Precision-Recall curve Vader(area = %0.2f)' % auc(recall_1_vader, precision_1_vader))

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
