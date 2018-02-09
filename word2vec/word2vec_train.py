from gensim.models import word2vec
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("wiki_texts_en.txt")
    model = word2vec.Word2Vec(sentences, size=300, window=3, iter=5)

    #保存模型，供日後使用
    model.wv.save_word2vec_format(u"en.model.bin", binary=True)


    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model.bin")

if __name__ == "__main__":
    main()
