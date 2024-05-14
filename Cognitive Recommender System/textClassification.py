import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources (you only need to do this once)
nltk.download('punkt')
nltk.download('stopwords')

# Sample data (replace this with your collected data)
titles = ["Title 1", "Title 2", "Title 3"]
posts = ["This is the content of post 1.", "Another post with some content.", "And a third post here."]

# Combine titles and posts
combined_data = [title + ' ' + post for title, post in zip(titles, posts)]

# Function to preprocess text
def preprocess_text(text):
    # Tokenize
    words = word_tokenize(text.lower())  # Convert to lowercase for consistency
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return words

# Preprocess each combined post
preprocessed_data = [preprocess_text(post) for post in combined_data]

# Print the results
for i, (title, processed_post) in enumerate(zip(titles, preprocessed_data)):
    print(f"Title {i + 1}: {title}")
    print(f"Processed Post {i + 1}: {processed_post}")
    print()
