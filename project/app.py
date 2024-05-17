# import nltk
# from nltk.stem import WordNetLemmatizer
# from newspaper import Article
# import joblib
# # from flask import Flask , jsonify,request  
# from flask import Flask, request, render_template
  
# app = Flask(__name__)   # Flask constructor 


# tfidf=joblib.load("C:/Users/Admin/Downloads/tfidf.pickle")
# model=joblib.load("C:/Users/Admin/Downloads/best_knnc.pickle")

# # Download NLTK resources
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

# # Initialize WordNet Lemmatizer
# wordnet_lemmatizer = WordNetLemmatizer()

# # Define a list of English stop words
# stop_words = nltk.corpus.stopwords.words('english')

# # Funtion to process texts
# def text_processing(text):

#     processed_text = text.replace("\r", " ").replace("\n", " ")
#     processed_text = processed_text.lower()  # Lowercase the text
#     punctuation_signs = list("?:!.,;") # Remove punctuation

#     for punct_sign in punctuation_signs:
#         processed_text = processed_text.replace(punct_sign, '')

#     processed_text = processed_text.replace("'s", "") # Remove "'s"

#     lemmatized_list = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in processed_text.split()]  # Lemmatization
#     processed_text = " ".join(lemmatized_list)


#     for stop_word in stop_words:   # Remove stop words

#         regex_stopword = r"\b" + stop_word + r"\b"
#         processed_text = processed_text.replace(regex_stopword, '')

#     return processed_text


# def news_scraping(url):

#     article = Article(url, language="en")
#     article.download()
#     article.parse()
#     article.nlp()

#     title=article.title
#     print("Title of the article:", title)

#     text=article.text
#     # print("Text from the article:", text)

#     summary=article.summary
#     # print("Summary of the article piece:", summary)

#     keywords=article.keywords
#     # print("Important Keywords of the article:", keywords)
    

#     return title,text,keywords,summary
    


# def classify_news(news_text):

#     preprocessed_text = text_processing(news_text)

#     # Building features from the processed text
#     vectorized_text = tfidf.transform([preprocessed_text])

#     predicted_class = model.predict(vectorized_text)[0]

#     # Converting to dense array for SVM Model
#     # vectorized_text_dense = vectorized_text.toarray()

#     # Make prediction
#     predicted_class = model.predict(vectorized_text)[0]

#     if predicted_class==0:
#       return 'business'

#     elif predicted_class==1:
#       return 'entertainment'

#     elif predicted_class==2:
#       return 'politics'

#     elif predicted_class==3:
#       return 'sports'

#     elif predicted_class==4:
#       return 'tech'

#     else:
#       return 'others'
    

# @app.route('/', methods=['GET','POST'])
# def classify():
#     if request.method == 'GET':
#         # Render the HTML form for GET requests
#         return render_template("index.html")
#     elif request.method == 'POST':
#         # Process form submission for POST requests
#         urls = request.form.get('urls')  # Access urls from form data
#         urls_list = [url.strip() for url in urls.split(',')]  # Split URLs by comma and remove leading/trailing spaces

#         results = []
#         for url in urls_list:
#             title, text, keywords,summary = news_scraping(url)
#             category = classify_news(title + text)
#             results.append({'title':title,'url': url, 'category': category, 'summary': summary, 'keywords': keywords})
        
#         return render_template("result.html", results=results)


# if __name__ == '__main__':
#     app.run(debug=True)


import nltk
from nltk.stem import WordNetLemmatizer
from newspaper import Article
import joblib
from flask import Flask, request, jsonify, render_template
import requests
from multiprocessing import Pool

app = Flask(__name__)

# Load models and initialize necessary objects
# tfidf = joblib.load("C:/Users/Admin/Downloads/tfidf.pickle")
# model = joblib.load("C:/Users/Admin/Downloads/best_knnc.pickle")
tfidf=joblib.load('tfidf.pickle')
model=joblib.load('best_knnc.pickle')

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = nltk.corpus.stopwords.words('english')


def text_processing(text):
    processed_text = text.replace("\r", " ").replace("\n", " ")
    processed_text = processed_text.lower()

    punctuation_signs = list("?:!.,;")
    for punct_sign in punctuation_signs:
        processed_text = processed_text.replace(punct_sign, '')

    processed_text = processed_text.replace("'s", "")

    lemmatized_list = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in processed_text.split()]
    processed_text = " ".join(lemmatized_list)

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        processed_text = processed_text.replace(regex_stopword, '')

    return processed_text


def news_scraping(url):
    article = Article(url, language="en")
    article.download()
    article.parse()
    article.nlp()

    title = article.title
    text = article.text
    summary = article.summary
    keywords = article.keywords

    return title, text, keywords, summary


def classify_news(news_text):
    preprocessed_text = text_processing(news_text)
    vectorized_text = tfidf.transform([preprocessed_text])
    predicted_class = model.predict(vectorized_text)[0]

    if predicted_class == 0:
        return 'Business'
    elif predicted_class == 1:
        return 'Entertainment'
    elif predicted_class == 2:
        return 'Politics'
    elif predicted_class == 3:
        return 'Sports'
    elif predicted_class == 4:
        return 'Tech'
    else:
        return 'Others'


def process_url(url):
    title, text, keywords, summary = news_scraping(url)
    category = classify_news(title + text)
    return {'title': title, 'url': url, 'category': category, 'summary': summary, 'keywords': keywords}

NEWS_API_KEY ='4557e798c48343e1af21dc38e1b27e5d'

def fetch_news_by_category(category):
    # url = f'http://newsapi.org/v2/top-headlines?category={category}&apiKey={NEWS_API_KEY}'
    url = f'http://newsapi.org/v2/top-headlines?category={category}&country=in&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])

    return articles[:4]


@app.route('/', methods=['GET', 'POST'])
def classify():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        urls = request.form.get('urls')
        urls_list = [url.strip() for url in urls.split(',')]

        # Using multiprocessing to process multiple URLs concurrently
        with Pool() as pool:
            results = pool.map(process_url, urls_list)

        recommended_stories = {}
        for result in results:
            category = result['category']
            recommended_stories[category] = fetch_news_by_category(category)

        return render_template("result.html", results=results, recommended_stories=recommended_stories)


if __name__ == '__main__':
    app.run(debug=True)

