#https://drive.google.com/file/d/1fqNyfrcSCryh651jSO9twH1-B12kh8ZV/view?usp=drive_link

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud


# Download the VADER lexicon
nltk.download('vader_lexicon')
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Load the dataset
df = pd.read_csv('review.csv')
print(df.head())

# Check data types of columns
print(df.dtypes)
print("\n")
# Summary statistics
print(df.describe())
print("\n")
# Check for missing values
print(df.isnull().sum())
print("\n")
# Unique values in categorical columns
print(df['Author'].unique())
print(df['Reviewer Data'].unique())
print("\n")

# Histogram of ratings
plt.hist(df['Rating'])
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.show()

# Define a function to remove URLs from text
def remove_urls(text):
    if isinstance(text, str):  # Check if text is a string
        # Define the regex pattern to match URLs
        url_pattern = r'https?://\S+|www\.\S+'
        # Replace URLs with an empty string
        clean_text = re.sub(url_pattern, '', text)
        return clean_text
    else:
        return text  # Return unchanged if text is not a string

# Apply the function to the 'Review' column
df['Review'] = df['Review'].apply(remove_urls)

# Display the first few rows of the DataFrame to verify the changes
print(df.head())


# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Define a function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Tokenize the text
        tokens = word_tokenize(text)

        # Convert tokens to lowercase
        tokens = [token.lower() for token in tokens]

        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Stemming
        tokens = [stemmer.stem(token) for token in tokens]

        # Join tokens back into a single string
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    else:
        return ''  # Return an empty string if text is not a string

# Apply the preprocessing function to the 'Review' column
df['Review'] = df['Review'].apply(preprocess_text)

# Display the first few rows of the DataFrame to verify the changes
print(df.head())


# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores
def get_sentiment(text):
    # Get sentiment scores
    sentiment_scores = sid.polarity_scores(text)
    # Classify sentiment based on compound score
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to the 'Review' column
df['Sentiment'] = df['Review'].apply(get_sentiment)

# Display the first few rows of the DataFrame with sentiment
print(df.head())


# Calculate sentiment distribution
sentiment_distribution = df['Sentiment'].value_counts()

# Visualize sentiment distribution
plt.figure(figsize=(8, 6))
sentiment_distribution.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution of Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# Define a function to generate word cloud for a specific sentiment
def generate_wordcloud(sentiment):
    # Filter reviews by sentiment
    filtered_reviews = df[df['Sentiment'] == sentiment]['Review']

    # Join all reviews into a single string
    text = ' '.join(filtered_reviews)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plot word cloud
    plt.figure(figsize=(5, 2))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Reviews')
    plt.axis('off')
    plt.show()

# Generate word clouds for each sentiment category
generate_wordcloud('Positive')
generate_wordcloud('Negative')
generate_wordcloud('Neutral')


# Assuming you have a 'Date' column in your DataFrame indicating the date of each review
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Define the time interval for analysis (e.g., weekly)
time_interval = 'W'  # Weekly

# Group the data by the specified time interval and count the occurrences of each sentiment label
sentiment_counts_over_time = df.groupby(pd.Grouper(key='Date', freq=time_interval))['Sentiment'].value_counts().unstack()

# If 'Negative' sentiment label doesn't have proper data, fill NaN values with 0
sentiment_counts_over_time.fillna(0, inplace=True)

# Plot sentiment trends over time
sentiment_counts_over_time.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Temporal Sentiment Analysis')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.legend(title='Sentiment')
plt.grid(True)
plt.show()
