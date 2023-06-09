from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import Text_Processing_Service as TPS

vectorizer = TfidfVectorizer()

# هذا التابع بحسب tfidf لكل دوكيمنت
def tfidfVectorMatrix(corpus):   
    
    # Create a list of documents
    # array of strings //without id_document  اخدنا المحتوى فقط 
    
    documents = list(corpus.values())  

    # calc tf-idf for each doc and store them in tfidf_matrex
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return tfidf_matrix
# now tfidf_matrix its array row--> documents and column--> terms(words)(tokens)


def Write_Matrix(tfidf_matrix, filename):
    matrixJsonFile = open(filename, "wb")
    pickle.dump(tfidf_matrix, matrixJsonFile)
    
    matrixJsonFile.close()  



def Read_Matrix(filename):
 with open(filename, 'rb') as openfile:
       
        matrix = pickle.load(openfile)
        return matrix



def tfidf_For_Query(query):
    query_tfidf = vectorizer.transform([query])
    return query_tfidf



def matchies_Query(query_tfidf, tfidf_matrix):
    result = cosine_similarity(tfidf_matrix, query_tfidf).flatten()
    return result


