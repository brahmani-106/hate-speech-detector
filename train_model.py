import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    df = pd.read_csv("dataset.csv")  
except FileNotFoundError:
    print("❌ Error: 'dataset.csv' not found. Please ensure the file is in the correct folder.")
    exit()

if not {'class', 'tweet'}.issubset(df.columns):
    print("❌ Error: Dataset must contain 'class' and 'tweet' columns.")
    exit()

df = df[['class', 'tweet']]
df = df.rename(columns={'tweet': 'text', 'class': 'label'})

df['label'] = df['label'].map({0: 1, 1: 1, 2: 0})

df.dropna(subset=['text', 'label'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")
