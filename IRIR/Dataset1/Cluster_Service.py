from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import TfIdf_Vector_Service as tvs
import numpy as np
from sklearn.cluster import MiniBatchKMeans  
from sklearn.cluster import MiniBatchKMeans

num_clusters = 38
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

def documentClustering(docs_tfidf):
    kmeans.fit(docs_tfidf)
    doc_clusters = kmeans.labels_
    return doc_clusters





def queryCluserting(query_tfidf):
    query_cluster = kmeans.predict(query_tfidf)
    return query_cluster