import pandas as pd
import numpy as np
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from flask import Flask, request, render_template
import pickle

# Step 1: Create Synthetic Watchlist Data
watchlist = pd.read_csv('watchlist.csv')

# Step 2: Define Matching Functions
def name_similarity(name1, name2):
    return fuzz.token_sort_ratio(name1.lower(), name2.lower())

def phonetic_similarity(name1, name2):
    return int(jellyfish.soundex(name1.lower()) == jellyfish.soundex(name2.lower()))

def tfidf_cosine_similarity(name1, name2, vectorizer):
    vectors = vectorizer.transform([name1.lower(), name2.lower()])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer().fit(watchlist['name'].tolist())
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

# Step 3: Flask Web Application
app = Flask(__name__)
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def match_input_name(input_name, input_dob, input_country):
    results = []
    is_fp = True  # Assume False Positive unless a match is found
    for _, wl in watchlist.iterrows():
        name_score = name_similarity(input_name, wl['name'])
        phonetic_score = phonetic_similarity(input_name, wl['name'])
        cosine_score = tfidf_cosine_similarity(input_name, wl['name'], vectorizer)
        final_score = (name_score + phonetic_score * 100 + cosine_score * 100) / 3
        
        if final_score > 60 and input_dob == wl['dob'] and input_country.lower() == wl['country'].lower():
            is_fp = False  # Found a match, not a False Positive
            if any([wl['settlement_payment'], wl['order_to_pay'], wl['charges']]):
                category = 'TNA (True No Action)'
            else:
                category = 'TP (True Positive)'
            results.append({'watchlist_name': wl['name'], 'score': final_score, 'dob': wl['dob'], 'country': wl['country'], 'category': category})
    
    if is_fp:
        category = 'FP (False Positive)'
        results.append({'watchlist_name': input_name, 'score': 0, 'dob': input_dob, 'country': input_country, 'category': category})
    
    return results

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_name = request.form['name']
        input_dob = request.form['dob']
        input_country = request.form['country']
        matches = match_input_name(input_name, input_dob, input_country)
        return render_template('index.html', matches=matches, input_name=input_name)
    return render_template('index.html', matches=None)

if __name__ == '__main__':
    app.run(debug=True)
