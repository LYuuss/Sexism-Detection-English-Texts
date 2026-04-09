import nltk
from nltk.corpus import stopwords

def download_nltk_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
        print("NLTK stopwords already downloaded.")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
def set_stopwords(language):
    return set(stopwords.words(language))

# list of stopword without things like "her", "she"...
custom_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'any', 'both', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 'will', 'just', 'don', 'now','br']

def get_custom_stopwords():
    return custom_stopwords

def delete_custom_stopwords(words_to_delete):
    global custom_stopwords
    custom_stopwords = [word for word in custom_stopwords if word not in words_to_delete]
    
def add_custom_stopwords(words_to_add):
    global custom_stopwords
    for word in words_to_add:
        if word not in custom_stopwords:
            custom_stopwords.append(word)