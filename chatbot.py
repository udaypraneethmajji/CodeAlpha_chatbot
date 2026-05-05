from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ data
questions = [
    "What is AI?",
    "What is Machine Learning?",
    "What is Python?",
    "What is your name?"
]

answers = [
    "AI means Artificial Intelligence.",
    "Machine Learning is a subset of AI.",
    "Python is a programming language.",
    "I am your chatbot."
]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break
    
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    
    index = similarity.argmax()
    print("Bot:", answers[index])