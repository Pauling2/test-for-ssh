from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk,re
from gensim.models import word2vec

news=fetch_20newsgroups(subset='all')

x,y=news.data,news.target

def news_to_sentences(news):
    news_text=BeautifulSoup(news).get_text()
    tokenizer=nltk.data.load('tojenizers/punkt/english.pickle')
    raw_sentences=tokenizer.tokenize(news_text)
    sentences=[]
    for i in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]',' ',i.lower().strip()).split())
    return sentences

sentences=[]
for i in x:
    sentences+=news_to_sentences(x)
    
model=word2vec.Word2Vec(sentences,workers=2,size=300,min_count=20,window=5,sample=1e-3)
model.init_sims(replace=True)

model.most_similar('morning')
