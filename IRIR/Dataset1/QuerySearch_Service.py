from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.sparse import csr_matrix

import numpy as np

import Cluster_Service as cs
import Inverted_Index_Service as  iis
import Text_Processing_Service as tps
import TfIdf_Vector_Service as tvs

app = Flask(__name__)
CORS(app)

###################################################################1

documents_as_dict = dict()
tsv_file = 'Dataset1\Files_Test\collection.txt'

with open(tsv_file, 'r') as file:
    for i, line in enumerate(file):
        line = line.strip().split('\t')
        doc_id = line[0]
        document = line[1]
        documents_as_dict[doc_id] = document

################################################################2

processed_docs_as_dict = tps.fileToDict("Dataset1\Files\collection_after_process.txt", 14405)
processed_documents = list(processed_docs_as_dict.values())
print(processed_documents)
print("badervalue")
processed_documents_keys = list(processed_docs_as_dict.keys())
print(processed_documents_keys)
print("baderkey")




#################################################################3
index = iis.offlineReadIndex("Dataset1\Files_Test\index.json")
matrix = tvs.offlineReadMatrix("Dataset1\Files_Test\matrix.json")
print(matrix)
print("before")
print(index)
tvs.vectorizer.fit(processed_documents)
print(matrix.shape)
cs.documentClustering(matrix)

#################################################################4
def search(query):
    # process query
    processed_query = tps.processQuery(query)
    print("YY")
    print(processed_query)
    processed_query_as_text = ' '.join(processed_query)
    print("x")
    print(processed_query_as_text)
    query_tfidf = tvs.tfidfQuery(processed_query_as_text)
    print("XX")
    print(query_tfidf)
    query_cluster = cs.queryCluserting(query_tfidf)
    print("XXX")
    print(query_cluster)    
    print("XXXXXXXXXXXXXx")
    relevant_docs = index[query_cluster[0]]
    doc_scores = tvs.matchQuery(query_tfidf,matrix[relevant_docs])

    sorted_doc_ids = np.array(relevant_docs)[doc_scores.argsort()[::-1]]

    result = []
    i = 0
    for idx in sorted_doc_ids:
        if i > 10:
            break
        result.append(documents_as_dict[processed_documents_keys[idx]])
        i+=1
    return result
#########################################################################
def docsIdsSearch(query):

    # process query 
    processed_query = tps.processQuery(query)
    processed_query_as_text = ' '.join(processed_query)
    query_tfidf = tvs.tfidfQuery(processed_query_as_text)
    query_cluster = cs.queryCluserting(query_tfidf)
    print("bjbjbj")
    relevant_docs = index[query_cluster[0]]
    doc_scores = tvs.matchQuery(query_tfidf,matrix[relevant_docs])

    sorted_doc_ids = np.array(relevant_docs)[doc_scores.argsort()[::-1]]

    result = []
    i = 0
    for idx in sorted_doc_ids:
        if i > 10:
            break
        result.append(processed_documents_keys[idx])
        i+=1
    return result


# a=search("mileage hybrid better city mileage")
# print(a)


@app.route("/search", methods=['POST','GET'])
def processData():
    query = request.json.get('query', '')
    result = search(query)
    return jsonify(result=result)

if (__name__) == "__main__":
    app.debug = True
    app.run()