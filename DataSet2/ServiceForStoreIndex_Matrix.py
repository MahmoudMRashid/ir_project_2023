import Cluster_Service as clus
import Inverted_Index_Service as  index
import Text_Processing_Service as tps
import TfIdf_Vector_Service as tvs

# doc=tps.process_documents("Files\collection.txt","Files\collection_after_process.txt",277072)  
# doc=tps.process_documents("Files_Test\collection.txt","Files_Test\collection_after_process.txt",15000)  

dic = tps.fileToDictionary("Files_Test\collection_after_process.txt", 15001)
tfidf_matrix = tvs.tfidfVectorMatrix(dic)
print("x")
documents_clusters = clus.document_Cluster(tfidf_matrix)
print("XX")
inverted_index = index.document_Index(documents_clusters)
index.Write_Index(inverted_index, 'Files_Test\index.json')
tvs.Write_Matrix(tfidf_matrix, 'Files_Test\matrix.json')