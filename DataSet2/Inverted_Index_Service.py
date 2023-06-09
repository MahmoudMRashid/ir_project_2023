import pickle
import TfIdf_Vector_Service as tvs



def document_Index(document_clusters):
    document_index = {}
    for doc_id, cluster_id in enumerate(document_clusters):
        if cluster_id not in document_index:
            document_index[cluster_id] = []
        document_index[cluster_id].append(doc_id)
        
    return document_index



def Write_Index(inverted_index, filename):
    
    invertedIndexJsonFile = open(filename, "wb")
    pickle.dump(inverted_index, invertedIndexJsonFile)

    invertedIndexJsonFile.close()




def Read_Index(filename):
 with open(filename, 'rb') as openfile:
        index = pickle.load(openfile)
        return index
    
 
 


