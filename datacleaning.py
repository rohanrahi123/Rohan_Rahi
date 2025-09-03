import pandas as pd
import re
import string
import emoji
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to clean and tokenize text
def preprocess_text_return_tokens(text):
    text = str(text).lower()  # Lowercase everything

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#', '', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Normalize repeated characters (e.g. cooooool → cool)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # spaCy NLP processing
    doc = nlp(text)

    # Tokenize + Lemmatize + Remove stopwords + Keep only words (no symbols or digits)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    return tokens

# Load your Excel file
df = pd.read_excel("twitter_dataset_with_followers.csv")

# Apply preprocessing to the 'Text_cleaned' column
df['tokens'] = df['Text_cleaned'].astype(str).apply(preprocess_text_return_tokens)

# Save the new DataFrame with the tokens column
df.to_csv("tweets_cleaned_tokenized.csv", index=False)
