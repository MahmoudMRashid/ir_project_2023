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

documents_as_dictionary = dict()
txt = 'Files_Test\collection.txt'

with open(txt, 'r') as file:
    for i, line in enumerate(file):
        line = line.strip().split('\t')
        document_id = line[0]
        document = line[1]
        documents_as_dictionary[document_id] = document

################################################################2

process_document_dictionary = tps.fileToDictionary("Files_Test\collection_after_process.txt", 15001)
processed_documents = list(process_document_dictionary.values())
print(processed_documents)
print("badervalue")
processed_documents_keys = list(process_document_dictionary.keys())
print(processed_documents_keys)
print("baderkey")




#################################################################3
index = iis.Read_Index("Files_Test\index.json")
matrix = tvs.Read_Matrix("Files_Test\matrix.json")
print(matrix)
print("before")
print(index)
tvs.vectorizer.fit(processed_documents)
print(matrix.shape)
cs.document_Cluster(matrix)

#################################################################4
def searchForQuery(query):
    # process query
    process_query = tps.processQuery(query)
    print("YY")
    print(process_query)
    processed_query_as_text = ' '.join(process_query)
    print("x")
    print(processed_query_as_text)
    query_tfidf = tvs.tfidf_For_Query(processed_query_as_text)
    print("XX")
    print(query_tfidf)
    query_cluster = cs.query_Cluser(query_tfidf)
    print("XXX")
    print(query_cluster)    
    print("XXXXXXXXXXXXXx")
    relevant_document = index[query_cluster[0]]
    document_result = tvs.matchQuery(query_tfidf,matrix[relevant_document])

    sorted_doc_ids = np.array(relevant_document)[document_result.argsort()[::-1]]

    result = []
    i = 0
    for idx in sorted_doc_ids:
        if i > 10:
            break
        result.append(documents_as_dictionary[processed_documents_keys[idx]])
        i+=1
    return result
#########################################################################
def documents_Id_from_Search(query):

    # process query 
    process_query = tps.processQuery(query)
    processed_query_as_text = ' '.join(process_query)
    query_tfidf = tvs.tfidf_For_Query(processed_query_as_text)
    query_cluster = cs.query_Cluser(query_tfidf)
    print("bjbjbj")
    relevant_document = index[query_cluster[0]]
    document_result = tvs.matchQuery(query_tfidf,matrix[relevant_document])

    sorted_doc_ids = np.array(relevant_document)[document_result.argsort()[::-1]]

    result = []
    i = 0
    for idx in sorted_doc_ids:
        if i > 10:
            break
        result.append(processed_documents_keys[idx])
        i+=1
    return result


# a=search("sun write wrtoe")
# print(a)


@app.route("/search", methods=['POST','GET'])
def processData():
    query = request.json.get('query', '')
    result = searchForQuery(query)
    return jsonify(result=result)

if (__name__) == "__main__":
    app.debug = True
    app.run()