import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearnex import patch_sklearn

# Load the CSV file into a DataFrame
df = pd.read_csv('IMDB Dataset.csv')

# Download stopwords from NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define stopwords in English
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters (keep only alphanumeric characters and spaces)
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    # Tokenize the text and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to the 'review' column in the DataFrame
df['review'] = df['review'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the text data and transform it into TF-IDF features
X = vectorizer.fit_transform(df['review'])

# Convert the 'sentiment' column to binary values: 1 for positive, 0 for negative
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the data into training and testing sets (80% training, 20% testing)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduce the size of the dataset to 40% of the original size for both training and testing sets
fraction = 0.4
num_train_samples = int(X_train_full.shape[0] * fraction)
num_test_samples = int(X_test_full.shape[0] * fraction)

# Select a subset of the training and testing data based on the reduced size
X_train = X_train_full[:num_train_samples]
y_train = y_train_full[:num_train_samples]
X_test = X_test_full[:num_test_samples]
y_test = y_test_full[:num_test_samples]

# Patch Scikit-Learn to use Intel Extension for optimized performance
patch_sklearn()

# Initialize the Support Vector Classifier model
model = SVC()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the sentiment labels for the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model with Intel Extension
accuracy_with_extension = accuracy_score(y_test, y_pred)
print(f'Accuracy with Intel Extension: {accuracy_with_extension}')


