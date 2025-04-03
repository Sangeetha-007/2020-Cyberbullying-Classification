import pandas as pd 
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline



df=pd.read_csv("/Users/sangeetha/Downloads/cyberbullying_tweets.csv")

print(df.head())

#Analyzing the shape of the dataset
print(df.shape)
#There are 4080 rows and 4 columns

#Printing the information of the dataset
print(df.info)

# Check for null values in the entire DataFrame
null_values=df.isnull().sum()
print("Null values: " ,null_values)

text = ' '.join(df['tweet_text']) 
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
#Bilinear interpolation is a method of smoothing an image when it is displayed at a resolution different from its original size.
plt.axis('off')  # Turn off axis
plt.show()

#df2=df['tweet_text']

####################################


# Function to extract polarity from TextBlob
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Apply the function to each tweet
df['polarity'] = df['tweet_text'].apply(get_polarity)

# Generate sentiment labels based on polarity
df['sentiment_label'] = df['polarity'].apply(lambda x: 'positive' if x > 0.8 else ('negative' if x < 0 else 'neutral'))
#The value 0.8 is a threshold that helps to distinguish between positive and neutral sentiment.

# Print the DataFrame to verify
print(df)

# Separate text based on sentiment labels for positive, negative, and neutral
positive_text = ' '.join(df.loc[df['sentiment_label'] == 'positive', 'tweet_text'])
negative_text = ' '.join(df.loc[df['sentiment_label'] == 'negative', 'tweet_text'])
neutral_text = ' '.join(df.loc[df['sentiment_label'] == 'neutral', 'tweet_text'])

# Generate word clouds for each sentiment label
positive_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)
negative_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)
neutral_wc = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(neutral_text)

# Display the word clouds for each sentiment label
plt.figure(figsize=(18, 6))

# Positive word cloud
plt.subplot(1, 3, 1)
plt.title("Positive Words")
plt.imshow(positive_wc, interpolation='bilinear')
plt.axis('off')

# Negative word cloud
plt.subplot(1, 3, 2)
plt.title("Negative Words")
plt.imshow(negative_wc, interpolation='bilinear')
plt.axis('off')

# Neutral word cloud
plt.subplot(1, 3, 3)
plt.title("Neutral Words")
plt.imshow(neutral_wc, interpolation='bilinear')
plt.axis('off')

plt.tight_layout()
plt.show()

# Make sure 'label' column is categorical (binary or multi-class classification)
df['sentiment_label'] = df['sentiment_label'].astype('category')

X = df['tweet_text']  # Text data
y = df['sentiment_label']     # Labels (Bullying or Not), target variable, column that you want to predict

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and Naive Bayes
# Pipeline ensures vectorization and model training are combined in one step
pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Train the Naive Bayes model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
#print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

