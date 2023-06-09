import Cluster_Service as clus
import Inverted_Index_Service as  index
import Text_Processing_Service as tps
import TfIdf_Vector_Service as tvs


# doc=tps.process_documents("Dataset1\Files\collection.txt","C://Users//Administrator//Desktop//test//collection_edit_process.txt",20324)  
# doc=tps.process_documents("Dataset1\Files\collection.txt","Dataset1\Files\collection_after_process.txt",373666)  
# doc=tps.process_documents("Dataset1\Files_Test\collection.txt","Dataset1\Files_Test\collection_after_process.txt",15000)  
# get the processed dataset as a dictionary
dic = tps.fileToDict('Dataset1\Files_Test\collection_after_process.txt', 14405)

tfidf_matrix = tvs.tfidfVectorMatrix(dic)

documents_clusters = clus.documentClustering(tfidf_matrix)
print("x")
inverted_index = index.docIndexing(documents_clusters)

index.offlineWriteIndex(inverted_index, 'Dataset1\Files_Test\index.json')

tvs.offlineWriteMatrix(tfidf_matrix, 'Dataset1\Files_Test\matrix.json')