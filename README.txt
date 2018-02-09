# Slotfilling
Using Tensorflow to train a slot-filling &amp; intent joint model


Prerequisites:
gensim
tensorflow
numpy
pandas


0. Create a folder named "ckpt"

1. Download raw data from wiki: https://dumps.wikimedia.org/backup-index.html

2. word2vec/wiki_to_txt.py
    python3 wiki_to_txt.py zhwiki-201xxxxx-pages-articles.xml.bz2
    (generated file: wiki_texts_en.txt)
    
3. word2vec/word2vec_train.py
    (generated file: en.model.bin)
    
4. model/dict.py
    - It will generate 2 csv files with intents, labels and their indexes
    
5. model/model_train.py

6. model/inference.py
    - edit line 9-10 to the checkpoint file you just trained


Ref:
[Training Word2Vec Model on English Wikipedia by Gensim]
http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim

[以 gensim 訓練中文詞向量]
http://zake7749.github.io/2016/08/28/word2vec-with-gensim/
