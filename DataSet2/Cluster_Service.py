from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import TfIdf_Vector_Service as tvs
import numpy as np
from sklearn.cluster import MiniBatchKMeans  


number_clusters = 10
kmeans = KMeans(n_clusters=number_clusters, random_state=42)

def document_Cluster(docs_tfidf):
    kmeans.fit(docs_tfidf)
    doc_clusters = kmeans.labels_
    return doc_clusters





def query_Cluser(query_tfidf):
    query_cluster = kmeans.predict(query_tfidf)
    return query_cluster