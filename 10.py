#https://drive.google.com/file/d/1d5GmZLsETcAMWCrosGt7gmyCXF0QrGPt/view?usp=drive_link
import pandas as pd

# Load the dataset
df = pd.read_csv("negative-data.csv")

# Display the first few rows of the dataset
print(df.head())

**Data Cleaning**

import string
# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Apply preprocessing to the 'Review_Text' column
df['Cleaned_Review_Text'] = df['Review_Text'].apply(preprocess_text)

# Display the first few rows of the dataset with cleaned text
print("\nFirst few rows with cleaned text:")
print(df[['Review_Text', 'Cleaned_Review_Text']].head())


**Sentiment Polarity**

from textblob import TextBlob

# Function to calculate sentiment polarity
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply sentiment analysis to the 'Review_Text' column
df['Sentiment_Polarity'] = df['Review_Text'].apply(calculate_sentiment)

# Calculate the average sentiment polarity of the dataset
average_sentiment = df['Sentiment_Polarity'].mean()

print(f"The average sentiment polarity of the dataset is: {average_sentiment}")
# Plot the average sentiment polarity
# Sort the DataFrame by sentiment polarity in descending order and select the top 10 reviews
top_10_reviews = df.nsmallest(10, 'Sentiment_Polarity')

# Plot the sentiment polarity of the top 10 reviews
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_reviews, x=top_10_reviews.index, y='Sentiment_Polarity', color='skyblue')
plt.title('Sentiment Polarity of Top 10 Reviews')
plt.xlabel('Review Index')
plt.ylabel('Sentiment Polarity')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.show()

**Negative Review Insights**

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Initialize CountVectorizer to convert text data into a bag-of-words representation
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Fit and transform the text data
bow_matrix = vectorizer.fit_transform(df['Cleaned_Review_Text'])

# Initialize LDA model
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)

# Fit LDA model to the bag-of-words matrix
lda_model.fit(bow_matrix)

# Display the top words for each topic
feature_names = vectorizer.get_feature_names_out()
num_top_words = 10

for topic_idx, topic in enumerate(lda_model.components_):
    print(f"Topic {topic_idx + 1}:")
    top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(top_words)

# Calculate the mean topic probability for each topic
num_topics = 5
topic_probabilities = df[[f'Topic_{i}_Probability' for i in range(1, num_topics + 1)]].mean()

# Plot the stacked bar plot
plt.figure(figsize=(10, 6))
topic_probabilities.plot(kind='bar', stacked=True, color=sns.color_palette("husl", num_topics))
plt.title('Distribution of Topics in the Dataset')
plt.xlabel('Topic')
plt.ylabel('Mean Probability')
plt.xticks(rotation=0)
plt.legend(title='Topic', loc='upper left')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of review lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['Review_Length'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


**Word Cloud**

from wordcloud import WordCloud

# Combine all review texts into a single string
all_reviews_text = ' '.join(df['Review_Text'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews_text)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Reviews')
plt.axis('off')
plt.show()
