import nltk

def download_nltk_data():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

if __name__ == '__main__':
    download_nltk_data()