import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
import warnings
from sklearn.manifold import TSNE
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import seaborn as sns
import plotly.graph_objects as go

#from sklearn.utils.fixes import signature
from tkinter import *
from ipywidgets import interact
import mplcursors
from matplotlib.container import BarContainer
import os
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification ,pipeline
import torch
import re
import networkx as nx
from neo4j import GraphDatabase ,basic_auth
import textwrap
import openai


reviews_df = pd.read_csv("Hotel_Reviews.csv", sep = ',')
reviews_df["review_total"]=reviews_df["Negative_Review"] + reviews_df["Positive_Review"]
reviews_df = reviews_df.sample(frac = 0.005, replace = False, random_state=42)
reviews_df["review_total"] = reviews_df["review_total"].fillna('')
reviews_df["Negative_Review"] = reviews_df["Negative_Review"].fillna('')
reviews_df["Positive_Review"] = reviews_df["Positive_Review"].fillna('')

    
reviews_df.dtypes


 #----- as wn
reviews_df["is_bad_review"] = reviews_df["Reviewer_Score"].apply(lambda x: 1 if x < 5 else 0)
reviews_df["is_good_review"] = reviews_df["Reviewer_Score"].apply(lambda x: 1 if x >= 5 else 0)



def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation....word.strip removes all the punctuation from each word after text.split
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop_english = stopwords.words('english')
    stop_greek = stopwords.words('greek')
    text = [x for x in text if x not in stop_english]
    text = [x for x in text if x not in stop_greek]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text ... find the grammar meaning of each word
    pos_tags = pos_tag(text)
    # lemmatize text ...meaning that words are replaced with more common with the same meaning or remove plurals or smaller words with the same meaning 
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    #print(text)
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

reviews_df["positive_clean"] = reviews_df["Positive_Review"].apply(lambda x: clean_text(x))
reviews_df["negative_clean"] = reviews_df["Negative_Review"].apply(lambda x: clean_text(x))
reviews_df["review_total_clean"]= reviews_df["review_total"].apply(lambda x: clean_text(x))

reviews_df["review_total_clean"] = reviews_df["review_total_clean"].fillna('')




categories = {
    'Cleanliness': ['clean', 'clear', 'well', 'fresh', 'bedsheet', 'towel','dirty', 'mess', 'smell', 'badly', 'iron', 'cleanness'],
    'Facilities': ['climate', 'decorated', 'spa','restaurant', 'bar', 'pool', 'Jacuzzi', 'Laundry', 'bathroom', 'food', 'breakfast','room','fridge','garden','window','aircondition', 'shower', 'facecloths', 'decoration', 'parking'],
    'Staff': ['service', 'polite','rude', 'helpful', 'pleasant'],
    'Location': ['location', 'quiet', 'close' ,'near', 'far', 'distance', 'transportation','sound', 'center', 'long', 'away', 'unsecure', 'dodgy', 'unsafe'],
    'Comfort': ['bed', 'security', 'safety','chill'],  
    'Value':['noise', 'great', 'relaxing', 'expensive', 'beautiful', 'horrible', 'superb', 'costly']   
}


terms_df= pd.DataFrame(
             [(k, val) for k, vals in categories.items() for val in vals], 
             columns=['category', 'term']
            )


a= list(terms_df.term)
  
def find_word(text):
    text = [word for word in text.split(" ")]
    text = [x for x in text if x in a]
    text = " ".join(text)
    return(text)
    

reviews_df["words"] = reviews_df["review_total_clean"].apply(lambda x: find_word(x))


def find_categories(text):
    text2=[]
    text = [word for word in text.split(" ")]
    for word in text:
        text1 = list(terms_df.query(f'term == "{word}"')['category'])
        #text1 =text.values.tolist()
        #for item in text:
        #    print(item)
        text2 += text1
    return(text2)
    

    
reviews_df["category"] = reviews_df["words"].apply(lambda x: find_categories(x))


word_list = reviews_df.words.values.tolist()
category_list = reviews_df.category.values.tolist()

def listToString(s):

    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ' ' + ele

    # return string
    return str1


def flat(lis):
    flatList = []
    # Iterate with outer list
    for element in lis:
        if type(element) is list:
            # Check if type is list than iterate through the sublist
            for item in element:
                flatList.append(item)
        else:
            flatList.append(element)
    return flatList
 
category_list1 = flat(category_list)

word_string=listToString(word_list)
categories_sting=listToString(category_list1)

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts
    


count_words=word_count(word_string)
count_categories=word_count(categories_sting)
    


#----upon user selection---

def show_neo4j_graphs(val):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth = basic_auth("neo4j","Xristina_1"))
    session = driver.session()
    reviews_df_temp=reviews_df[reviews_df["Hotel_Name"] == val]
    #-------01/10--vader

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
#create a new column (sentiments that keeps data created by polarity feature, score based the commends)
    reviews_df_temp["sentiments"] = reviews_df_temp["review_total_clean"].apply(lambda x: sid.polarity_scores(x))
#create new columns from the existing sentiment one that has already info  (create new columns for each value included in sentiments)
    reviews_df_temp = pd.concat([reviews_df_temp.drop(['sentiments'], axis=1), reviews_df_temp['sentiments'].apply(pd.Series)], axis=1)
    reviews_df_temp["Positively Rated_compound"]=reviews_df_temp['compound'].apply(lambda x: 1 if x > 0 else 0)
    reviews_df_temp["is_bad_review_compound"]=reviews_df_temp['compound'].apply(lambda x: 1 if x < 0 else 0)


#----find the review tag based on compound-----
    def find_review_tag(text):
        if text >= 0.05 :
            return("Positive")

        elif text <= - 0.05 :
            return("Negative")

        else :
            return("Neutral")



    reviews_df_temp["review_tag_vader"] = reviews_df_temp["compound"].apply(lambda x: find_review_tag(x))


    def create_nodes_and_relationships(tx, hotel_name, review_text, sentiment):
    # Create or match the hotel node based on the unique name
        tx.run("MERGE (hotel:Hotel {name: $hotel_name})", hotel_name=hotel_name)

    # Create review node and relationship with the existing hotel
        for text, tag in zip(review_text, sentiment):
            tx.run("MATCH (hotel:Hotel {name: $hotel_name}) "
               #"MERGE (review:Review {text: $text})-[:HAS_SENTIMENT {tag: $tag}]->(hotel) "
               #"MERGE (hotel:Hotel {name: $hotel_name})-[:HAS_SENTIMENT {tag: $tag}] -> (review:Review {text: $text})"
               "MERGE (review:Review {text: $text})"
               "MERGE (hotel)-[:HAS_SENTIMENT {tag: $tag}]->(review)"
               "MERGE (review)-[:HAS_SENTIMENT_TAG]->(tag:SentimentTag {name: $tag})",
                   hotel_name=hotel_name, text=text, tag=tag)

# Iterate over unique hotels in the DataFrame and create nodes and relationships
    unique_hotels = reviews_df_temp['Hotel_Name'].unique()

    with GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Xristina_1")) as driver:
        with driver.session() as session:
            for hotel_name in unique_hotels:
            # Filter data for the specific hotel
                hotel_data = reviews_df_temp[reviews_df_temp['Hotel_Name'] == hotel_name]
                # Print available columns for debugging
                print(f"Columns in 'hotel_data': {hotel_data.columns}")

            # Check if 'review_tag_vader' is in the columns
                if 'review_tag_vader' not in hotel_data.columns:
                    print(f"Column 'review_tag_vader' not found in 'hotel_data'. Check your column names.")

                review_texts = hotel_data['review_total'].tolist()
                sentiment_tags = hotel_data['review_tag_vader'].tolist()

            # Execute the transaction to create nodes and relationships
                session.execute_write(create_nodes_and_relationships, hotel_name, review_texts, sentiment_tags)

    def get_sentiment_graph(tx, hotel_name):
        result = tx.run(
            "MATCH (hotel:Hotel {name: $hotel_name})-[:HAS_SENTIMENT]->(review:Review)-[:HAS_SENTIMENT_TAG]->(tag:SentimentTag) "
            "RETURN hotel, review, tag",
            hotel_name=hotel_name)
        #print(result.data())
        return result.data()
        
    hotel_name_to_retrieve = val  # Use the 'val' variable as the hotel name

    with GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Xristina_1")) as driver:
        with driver.session() as session:
            result_data = session.execute_read(get_sentiment_graph, hotel_name_to_retrieve)
            print(result_data)
        
    neo4j_data =result_data
    

# Create a directed graph
    G = nx.DiGraph()

    # Define a dictionary to map tag_names to colors
    tag_color_mapping = {
        'Positive': 'green',  # Replace 'Positive' with the actual tag names in your data
        'Negative': 'red',    # Replace 'Negative' with the actual tag names in your data
        'Neutral': 'orange'     # Replace 'Neutral' with the actual tag names in your data
    }

# Iterate through the data and add nodes and edges to the graph
    for row in neo4j_data:
        hotel_name = row['hotel']['name']
        review_text = row['review']['text']
        tag_name = row['tag']['name']

    # Add hotel node
        G.add_node(hotel_name, name=hotel_name)


# Join the wrapped lines with '\n' to create the label
        G.add_node(review_text, name='Review', shape='ellipse', comment=review_text,color=tag_color_mapping.get(tag_name, 'gray'))  # Use the color of the corresponding tag node
    

    # Add sentiment tag node
        G.add_node(tag_name, name=tag_name,color=tag_color_mapping.get(tag_name, 'gray'))

    # Add edges between nodes
        G.add_edge(hotel_name, review_text, relationship='HAS_REVIEW')
        G.add_edge(review_text, tag_name, relationship='IS')

 #####26/11 13:35
    # Draw the graph
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'name')
    node_colors = [G.nodes[node].get('color', 'skyblue') for node in G.nodes]
    nx.draw(G, pos, with_labels=True, labels=labels, font_weight='bold', node_size=1000, node_color=node_colors, font_color='black', font_size=8, edge_color='gray', linewidths=1, arrows=True, node_shape='o')

# Add a callback for handling clicks
    def display_comment(event):
        if event.inaxes is not None:
            node = None
            for n, (x, y) in pos.items():
                radius = 0.05  # Define a small radius for the click area around each node
                if (x - radius) < event.xdata < (x + radius) and (y - radius) < event.ydata < (y + radius):
                    node = n
                    break
            if node is not None and 'comment' in G.nodes[node]:
                comment = G.nodes[node]['comment']
                wrapped_comment = '\n'.join(textwrap.wrap(comment, width=30))
                plt.annotate(
                    f"\n{wrapped_comment}",
                    xy=(x, y),
                    xytext=(x, y + 0.2),  # Adjust the position of the annotation
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.1', edgecolor='black', facecolor='white'),
                    #arrowprops=dict(arrowstyle='-', linewidth=0.5, color='black')
                )
                plt.draw()

# Connect the callback to the click event
    plt.gcf().canvas.mpl_connect('button_press_event', display_comment)

# Show the plot
    plt.show()




def show_all(val):
    def find_word(text):
        text = [word for word in text.split(" ")]
        text = [x for x in text if x in a]
        text = " ".join(text)
        return(text)
    
    reviews_df_temp=reviews_df[reviews_df['Hotel_Name']== val]
    print(reviews_df[reviews_df["Hotel_Name"] == val])
    reviews_df_temp["words"] = reviews_df_temp["review_total_clean"].apply(lambda x: find_word(x))


    def find_categories(text):
        text2=[]
        text = [word for word in text.split(" ")]
        for word in text:
            text1 = list(terms_df.query(f'term == "{word}"')['category'])
        #text1 =text.values.tolist()
        #for item in text:
        #    print(item)
            text2 += text1
        return(text2)
    

    
    reviews_df_temp["category"] = reviews_df_temp["words"].apply(lambda x: find_categories(x))


    word_list = reviews_df_temp.words.values.tolist()
    category_list = reviews_df_temp.category.values.tolist()

    def listToString(s):

    # initialize an empty string
        str1 = ""

    # traverse in the string
        for ele in s:
            str1 += ' ' + ele

    # return string
        return str1


    def flat(lis):
        flatList = []
    # Iterate with outer list
        for element in lis:
            if type(element) is list:
            # Check if type is list than iterate through the sublist
                for item in element:
                    flatList.append(item)
            else:
                flatList.append(element)
        return flatList
 
    category_list1 = flat(category_list)

    word_string=listToString(word_list)
    categories_sting=listToString(category_list1)

    def word_count(str):
        counts = dict()
        words = str.split()

        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

        return counts
    


    count_words=word_count(word_string)
    count_categories=word_count(categories_sting)

    plt.barh(list(count_words.keys()) ,list(count_words.values()), color='green')
    plt.show()

    plt.barh(list(count_categories.keys()),list(count_categories.values()) , color='orange')
    plt.show()






#-------01/10--vader

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
#create a new column (sentiments that keeps data created by polarity feature, score based the commends)
    reviews_df_temp["sentiments"] = reviews_df_temp["review_total_clean"].apply(lambda x: sid.polarity_scores(x))
#create new columns from the existing sentiment one that has already info  (create new columns for each value included in sentiments)
    reviews_df_temp = pd.concat([reviews_df_temp.drop(['sentiments'], axis=1), reviews_df_temp['sentiments'].apply(pd.Series)], axis=1)
    reviews_df_temp["Positively Rated_compound"]=reviews_df_temp['compound'].apply(lambda x: 1 if x > 0 else 0)
    reviews_df_temp["is_bad_review_compound"]=reviews_df_temp['compound'].apply(lambda x: 1 if x < 0 else 0)


#----find the review tag based on compound-----
    def find_review_tag(text):
        if text >= 0.05 :
            return("Positive")

        elif text <= - 0.05 :
            return("Negative")

        else :
            return("Neutral")



    reviews_df_temp["review_tag_vader"] = reviews_df_temp["compound"].apply(lambda x: find_review_tag(x))



#------
    def word_count_per_tag(str1,str2):
        counts_negative = dict()
        counts_positive = dict()
        counts_neutral = dict()
        words = str1.split()
        if str2 == 'Negative':
            for word in words:
                if word in counts_negative:
                    counts_negative[word] += 1
                else:
                    counts_negative[word] = 1
        elif  str2== 'Positive':
            for word in words:
                if word in counts_positive:
                   counts_positive[word] += 1
                else:
                    counts_positive[word] = 1
        else :
            for word in words:
               if word in counts_neutral:
                   counts_neutral[word] += 1
               else:
                   counts_neutral[word] = 1
        return (counts_negative , counts_positive , counts_neutral)

    results_vader=list(reviews_df_temp.apply(lambda x: word_count_per_tag(x.words, x.review_tag_vader), axis=1))

    counts_negative_words_vader = dict()
    counts_positive_words_vader = dict()
    counts_neutral_words_vader = dict()

    import tkinter as tk
    
    def show_annotation(sel):
        global y_axis_label_vader
        if type(sel.artist) == BarContainer:
            bar = sel.artist[sel.target.index]

            sel.annotation.set_text(f'Check specific reviews on click')
            sel.annotation.xy = (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2)
            sel.annotation.get_bbox_patch().set_alpha(0.8)
            y_axis_label_vader = sel.artist.get_label()
		#print(type(label))
            return (y_axis_label_vader)
    
    def on_bar_click(event):
        index = event.ydata  # Get the x-coordinate of the click
	#print(label)
        if index is not None:
            index = int(index)
            if 0 <= index < len(categories):
                word = df_words_polarity_vader.index[index]
                if y_axis_label_vader == "positive":
                    show_reviews_popup(word, "Positive")
                elif y_axis_label_vader == "negative":
                    show_reviews_popup(word, "Negative")
                else :
                    show_reviews_popup(word, "Neutral")
            
    for x in range (len(results_vader)):
        counts_negative_words_vader = {key: counts_negative_words_vader.get(key, 0) + results_vader[x][0].get(key, 0) for key in set(counts_negative_words_vader) | set(results_vader[x][0])}
    for x in range (len(results_vader)):
        counts_positive_words_vader = {key: counts_positive_words_vader.get(key, 0) + results_vader[x][1].get(key, 0) for key in set(counts_positive_words_vader) | set(results_vader[x][1])}
    for x in range (len(results_vader)):
        counts_neutral_words_vader = {key: counts_neutral_words_vader.get(key, 0) + results_vader[x][2].get(key, 0) for key in set(counts_neutral_words_vader) | set(results_vader[x][2])}
    
        
    def show_reviews_popup(word, tag):
        reviews = get_reviews_by_tag(tag, word)
        popup = tk.Toplevel()
        popup.title(f"{tag} Reviews for '{word}'")
        label = tk.Label(popup, text=f"{tag} Reviews for '{word}':")
        label.pack()
        text_widget = tk.Text(popup, height=50, width=100)
                #for review in reviews:
        for i, review in enumerate(reviews, start=1):
            review_text = f"{i}. {review}\n\n"  # Numbering each review
            text_widget.insert(tk.END, review_text)
            #text_widget.insert(tk.END, review + '\n')
        #for review in reviews:
            #text_widget.insert(tk.END, review + '\n')
        text_widget.pack()
    
        
    def get_reviews_by_tag(tag, word):
        return reviews_df_temp.query(f'review_tag_vader=="{tag}" and review_total.str.contains("{word}")')['review_total'].tolist()
    
    
    def show_reviews_in_popup(popup, label_text, reviews):
        label = tk.Label(popup, text=label_text)
        label.pack()
        text = tk.Text(popup, height=5, width=50)
        text.insert(tk.END, '\n'.join(reviews))
        text.pack()
    
    df_words_polarity_vader = pd.DataFrame({'negative' : counts_negative_words_vader,'neutral' : counts_neutral_words_vader,'positive' : counts_positive_words_vader})
    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)
    colors = ['#FF0000', '#FF8C00', '#008000']
    fig.canvas.mpl_connect('button_press_event', on_bar_click)
    df_words_polarity_vader.plot.barh(stacked=True, ax=ax,color=colors);
    ax.set_title("polarity_vader")
    ax.legend(loc='upper right')
    hfont = {'fontname':'Calibri'} # main font
    for p in ax.patches:
        if p._width>0:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.text(x+width/2, 
                y+height/2, 
                '{:.0f}'.format(width), 
                horizontalalignment='center', 
                verticalalignment='center',
                color='white',
                fontsize=8,
                **hfont)
    cursor = mplcursors.cursor(hover=True)
    cursor.connect('add', show_annotation)
    plt.show()



# add number of characters column (review)
    reviews_df_temp["nb_chars"] = reviews_df_temp["review_total"].apply(lambda x: len(x))

# add number of words column(review)
    reviews_df_temp["nb_words"] = reviews_df_temp["review_total"].apply(lambda x: len(x.split(" ")))




    fig = plt.figure(figsize = (2, 2))
 



#plt.plot(reviews_df['compound'], color='black')





    def color_func_vader(word, font_size, position, orientation, random_state=None, **kwargs):
        sentiment = sid.polarity_scores(word)['compound']
    
        if sentiment >= 0.5:
        # Positive sentiment, green
            return "green"
        elif sentiment <= -0.5:
        # Negative sentiment, red
            return "red"
        else:
        # Neutral sentiment, orange
            return "orange"

        

# Function to generate a word cloud for each review
    def generate_wordcloud_based_on_vader_sentiment(dataframe, text_column, sentiment_column, title=None):
        word_sentiments = {}  # Dictionary to store word counts for each sentiment category

    # Iterate over each row in the DataFrame
        for index, row in dataframe.iterrows():
            words = row[text_column].split()
            sentiment = row[sentiment_column]

            for word in words:
                if word not in word_sentiments:
                    word_sentiments[word] = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
                word_sentiments[word][sentiment] += 1

    # Determine predominant sentiment for each word
        predominant_sentiment = {word: max(sentiments, key=sentiments.get) for word, sentiments in word_sentiments.items()}

    # Create a WordCloud object with custom coloring function
        wordcloud = WordCloud(
            background_color='white',
            max_words=200,
            max_font_size=40,
            scale=3,
            random_state=42,
            color_func=lambda word, *args, **kwargs: color_func_based_on_vader_sentiment(word, predominant_sentiment)
        ).generate_from_frequencies({word: sum(sentiments.values()) for word, sentiments in word_sentiments.items()})

    # Display the generated word cloud using matplotlib
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        if title:
            fig.suptitle(title, fontsize=16)
        plt.show()

# Function to determine color based on predominant sentiment
    def color_func_based_on_vader_sentiment(word, predominant_sentiment):
        sentiment = predominant_sentiment.get(word, 'Neutral')  # Default to Neutral if word not found
        if sentiment == 'Positive':
            return "green"
        elif sentiment == 'Negative':
            return "red"
        else:  # Neutral
            return "orange"


    generate_wordcloud_based_on_vader_sentiment(reviews_df_temp, 'review_total', 'review_tag_vader', title='Word Cloud for Reviews')


#end show all ---


# start show review

def show_reviews(val, width=100):
    reviews_df_temp1 = reviews_df[reviews_df["Hotel_Name"] == val]


    if not reviews_df_temp1.empty:
        reviews = reviews_df_temp1["review_total"].values

        numbered_reviews = []

        for i, review in enumerate(reviews, start=1):
            wrapped_review = textwrap.fill(review, width=width)
            numbered_reviews.append(f"{i}. {wrapped_review}\n")

        return "\n".join(numbered_reviews)
    else:
        return "No reviews available for this hotel."


def show_positive_reviews(val, width=100):
    # Fetch the reviews for the specified hotel
    reviews_df_temp1 = reviews_df[reviews_df["Hotel_Name"] == val]
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
#create a new column (sentiments that keeps data created by polarity feature, score based the commends)
    reviews_df_temp1["sentiments"] = reviews_df_temp1["review_total_clean"].apply(lambda x: sid.polarity_scores(x))
#create new columns from the existing sentiment one that has already info  (create new columns for each value included in sentiments)
    reviews_df_temp1 = pd.concat([reviews_df_temp1.drop(['sentiments'], axis=1), reviews_df_temp1['sentiments'].apply(pd.Series)], axis=1)
    reviews_df_temp1["Positively Rated_compound"]=reviews_df_temp1['compound'].apply(lambda x: 1 if x > 0 else 0)
    reviews_df_temp1["is_bad_review_compound"]=reviews_df_temp1['compound'].apply(lambda x: 1 if x < 0 else 0)


#----find the review tag based on compound-----
    def find_review_tag(text):
        if text >= 0.05 :
            return("Positive")

        elif text <= - 0.05 :
            return("Negative")

        else :
            return("Neutral")



    reviews_df_temp1["review_tag_vader"] = reviews_df_temp1["compound"].apply(lambda x: find_review_tag(x))
    reviews_df_temp1 = reviews_df_temp1.query(f'review_tag_vader=="Positive"')
    # Check if there are any reviews
    if not reviews_df_temp1.empty:
        reviews = reviews_df_temp1["Positive_Review"].values
        numbered_reviews = []

        # Iterate through each review, number it, and wrap the text
        for i, review in enumerate(reviews, start=1):
            wrapped_review = textwrap.fill(review, width=width)
            numbered_reviews.append(f"{i}. {wrapped_review}\n")

        # Join all the numbered and wrapped reviews
        return "\n".join(numbered_reviews)
    else:
        # Return a default message when there are no reviews
        return "No positive reviews available for this hotel."
    


def show_negative_reviews(val, width=100):
    # Assuming openai.api_key is already set elsewhere
    reviews_df_temp1 = reviews_df[reviews_df["Hotel_Name"] == val]
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
#create a new column (sentiments that keeps data created by polarity feature, score based the commends)
    reviews_df_temp1["sentiments"] = reviews_df_temp1["review_total_clean"].apply(lambda x: sid.polarity_scores(x))
#create new columns from the existing sentiment one that has already info  (create new columns for each value included in sentiments)
    reviews_df_temp1 = pd.concat([reviews_df_temp1.drop(['sentiments'], axis=1), reviews_df_temp1['sentiments'].apply(pd.Series)], axis=1)
    reviews_df_temp1["Positively Rated_compound"]=reviews_df_temp1['compound'].apply(lambda x: 1 if x > 0 else 0)
    reviews_df_temp1["is_bad_review_compound"]=reviews_df_temp1['compound'].apply(lambda x: 1 if x < 0 else 0)


#----find the review tag based on compound-----
    def find_review_tag(text):
        if text >= 0.05 :
            return("Positive")

        elif text <= - 0.05 :
            return("Negative")

        else :
            return("Neutral")



    reviews_df_temp1["review_tag_vader"] = reviews_df_temp1["compound"].apply(lambda x: find_review_tag(x))

    reviews_df_temp1 = reviews_df_temp1.query(f'review_tag_vader=="Negative"')

    if not reviews_df_temp1.empty:
        reviews = reviews_df_temp1["Negative_Review"].values
        numbered_reviews = []
                                         
        
        for i, review in enumerate(reviews, start=1):
            wrapped_review = textwrap.fill(review, width=width)
            numbered_reviews.append(f"{i}. {wrapped_review}\n")

        return "\n".join(numbered_reviews)
    else:
        return "No negative reviews available for this hotel."
    
def show_sum_up_reviews(val, width=100):
    openai.api_key = 'YOUR_API_KEY'
    reviews_df_temp1=reviews_df[reviews_df["Hotel_Name"] == val]
    import textwrap 
    if not reviews_df_temp1.empty:
        value = str(reviews_df_temp1["review_total"].values)
        #### openai 
    
#    # Define your conversation or prompt
#    conversation = """
 #   You: Hello!
  #  AI: Hi there! How can I assist you today?
   # You: What's the weather like today?
    #AI: I'm sorry, I don't have real-time information. Can I help with something else?
 #   """

# Send the conversation to the API
#    response = openai.Completion.create(
 #       engine="davinci",  # You can choose a different engine if needed
#        prompt=conversation,
#        max_tokens=50  # Adjust the number of tokens as needed
#)

# Extract and print the AI's response
#    ai_reply = response.choices[0].text
#    print(ai_reply)
        wrapped_comment = '\n'.join(textwrap.wrap(value, width=30))
        wrapped_text = textwrap.fill(wrapped_comment, width)
        value1= 'The hotels mentioned have a mix of positive and negative aspects. Common positive points include spacious rooms, good breakfast options, free parking, and friendly staff. However, there are negative aspects such as crowded breakfast areas, distant or remote locations, issues with cleanliness, non-functioning air conditioning, and limited food options in some hotels. Despite these drawbacks, certain hotels are praised for their convenient locations, quick check-ins, and well-maintained facilities. Its recommended to consider individual preferences and priorities when choosing a hotel.'
        wrapped_comment1 = '\n'.join(textwrap.wrap(value1, width=30))
        wrapped_text1 = textwrap.fill(wrapped_comment1, width)
        return wrapped_text1
    else:
    # Return a default message when value is empty or there are no reviews

        return "No positive reviews available for this hotel."


def show_wu_palmer(val, width=100):
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

# Assuming y_true and y_scores are your true labels and predicted scores
    y_true = [0, 1, 1, 0, 1, 0]
    y_scores = [0.2, 0.8, 0.6, 0.3, 0.7, 0.4]

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()









#--------
import tkinter as tk

hotel_name=reviews_df.Hotel_Name.drop_duplicates(keep='first')
hotel_name.index = range(1, len(hotel_name) + 1)

def on_click1():
    val = selected_value
    show_all(val)
    
def on_click2():
    val = ''
    val = selected_value
    print(selected_value)
    top = Toplevel(root)
    top.geometry("950x700")
    top.title("Child Window")
    custom_font = ("Century Gothic", 12)
    Label(top, text= show_reviews(val), font=custom_font).place(x=150,y= 80)

def on_click3():
    val = ''
    val = selected_value
    top = Toplevel(root)
    top.geometry("950x700")
    top.title("Child Window")
    custom_font = ("Century Gothic", 12)
    Label(top, text= show_positive_reviews(val), font=custom_font).place(x=150,y= 80)

def on_click4():
    val = ''
    val = selected_value
    top = Toplevel(root)
    top.geometry("950x700")
    top.title("Child Window")
    custom_font = ("Century Gothic", 12)
    Label(top, text= show_negative_reviews(val), font=custom_font).place(x=150,y= 80)

def on_click5():
    val = ''
    top = Toplevel(root)
    top.geometry("950x700")
    top.title("Child Window")
    custom_font = ("Century Gothic", 12)
    Label(top, text= "εδώ θα βρείτε όλες τις απαραίτητες πληροφορίες για τα αποτελέσματα", font=custom_font).place(x=150,y= 80)



def on_click6():
    val = ''
    val = selected_value
    show_neo4j_graphs(selected_value)
# --- main ---

def on_click7():
    val = ''
    val = selected_value
    top = Toplevel(root)
    top.geometry("950x700")
    top.title("Child Window")
    custom_font = ("Century Gothic", 12)
    Label(top, text= show_sum_up_reviews(val), font=custom_font).place(x=150,y= 80)


def on_click9():
    val = ''
    val = selected_value
    how_wu_palmer(val)

    


def on_click8():
    val = ''
    val = selected2.get()
    #top = Toplevel(root)
    #top.geometry("950x700")
    #top.title("Child Window")
    custom_font = ("Century Gothic", 12)
    if val == 'view graphs':
        on_click1();
    elif val == 'view all reviews':
        on_click2();
    elif val == 'view positive reviews':
        on_click3();
    elif val == 'view negative reviews':
        on_click4();
    elif val == 'check reviews and tags':
        on_click6();
    elif val =='view sum up of the reviews':
        on_click7();
    elif val =='view sum up of the reviews':
        on_click9();
    

def on_click9():
    val = ''
    val = selected2.get()
    custom_font = ("Century Gothic", 12)
    show_wu_palmer

def Scankey(event):
	
    val = event.widget.get()
    print(val)
	

    if val == '':
        data = values1
    else:
        data = []
        for item in values1:
            if val.lower() in item.lower():
                data.append(item)				

	
    Update(data)
    listbox.bind('<<ListboxSelect>>', lambda event: on_select(event, data))

def on_select(event, data):
    global selected_value
    # Get the selected index from the event
    selected_index = listbox.curselection()

    if selected_index:
        # Get the selected value using the index
        selected_value = data[selected_index[0]]

        # Do something with the selected value
        print(f"Selected value: {selected_value}")
        return(selected_value)

def Update(data):
	

	listbox.delete(0, 'end')

	# put new data
	for item in data:
		listbox.insert('end', item)


root = tk.Tk()
root.title("tk")
root.geometry("900x900")
values1 = list(reviews_df.Hotel_Name.drop_duplicates(keep='first'))
values2=('view graphs','view all reviews','check reviews and tags','view positive reviews','view negative reviews','view sum up of the reviews')
# Add image file
from PIL import ImageTk, Image
bg = ImageTk.PhotoImage(Image.open("pexels-pixabay-87651.jpg")) 
  
# Show image using label 
label1 = Label( root, image = bg) 
label1.place(x = 0, y = 0) 
  
label2 = Label( root, text = "Welcome.\nSelect from the dropdown menu below the hotel for which you would like to get more info.Select \n-view graphs, to view info in visualize format \n-view all reviews, to have a view in all available reviews \n-view positive reviews, to check only positive reviews \n-view negative reviews, to view only negative reviews \n-check reviews and tags, for a visualization of hotel-reviews and tags and at last \n-select help to get more info for the useage of the tool") 
label2.pack(pady = 50) 


# Entry for searching
label3 = Label(root, text="Search Hotel:")
label3.pack()

entry = Entry(root)
entry.pack()
entry.bind('<KeyRelease>', Scankey)
listbox = Listbox(root)
listbox.pack()
Update(values1)

# Create Frame 
frame = Frame(root) 
frame.pack(pady = 20) 
  

selected2 = tk.StringVar()
selected2.set("Select Action")

options2 = tk.OptionMenu(root, selected2, *values2)
options2.pack()

button5 = tk.Button(root, text='help', command=on_click5)
button5.place(x=450,y=850)

# Button for searching and selecting
#button9 = tk.Button(root, text="Search and Select", command=search_and_select)
#button9.place(x=450,y=500)

button8 = tk.Button(root, text='selected action', command=on_click8)
button8.place(x=405,y=550)


root.mainloop()  
