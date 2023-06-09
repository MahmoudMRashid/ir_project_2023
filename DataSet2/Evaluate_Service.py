from flask import Flask, jsonify

app = Flask(__name__)

import QuerySearch_Service as search
import ir_datasets
dataset_writing = ir_datasets.load("lotte/writing/dev/search")



# qrels
qrel_dictaionary = {}
for qrel in dataset_writing.qrels_iter():
    # print(qrel) # namedtuple<query_id, doc_id, relevance, iteration>
    query_id = qrel[0]
    doc_id = qrel[1]
    relevance = qrel[2]
    if query_id not in qrel_dictaionary:
            qrel_dictaionary[query_id] = {}
    qrel_dictaionary[query_id][doc_id] = relevance


# get the retrieved docs for each query by our search engine
retrieved_documents = {}
for query in dataset_writing.queries_iter():
    #print(query) # namedtuple<query_id, text>
    ret_documents = search.docsIdsSearch(query.text)
    retrieved_documents[query.query_id] = ret_documents


# map

# Calculate average precision for each query
ap_list = []
for query in dataset_writing.queries_iter():
    relevant_count = 0
    precision_sum = 0.0
    for j, doc in enumerate(retrieved_documents[query.query_id]):

        if qrel_dictaionary[query.query_id].get(doc, 0) == 1:
            relevant_count += 1
            precision_sum += relevant_count / (j+1)
                        
    if relevant_count > 0:
        ap = precision_sum / relevant_count
        ap_list.append(ap)

    else: 
        ap_list.append(0)

#print (ap_list)
# Calculate mean average precision
map_score = sum(ap_list) / len(ap_list)
print("MAP:", map_score)




# mrr

# Calculate reciprocal rank for each query
rr_list = []
for query in dataset_writing.queries_iter():
    for j, doc in enumerate(retrieved_documents[query.query_id]):
        
        if qrel_dictaionary[query.query_id].get(doc, 0) == 1:
            rr_list.append(1 / (j+1))
            break
    
        rr_list.append(0)

# Calculate mean reciprocal rank
mrr_score = sum(rr_list) / len(rr_list)
print("MRR:", mrr_score)



#  precision@k 



k = 10
precision_list = []
for query in dataset_writing.queries_iter():
    relevant_count = 0
    retrieved_count = 0
    for j, doc in enumerate(retrieved_documents[query.query_id]):
        retrieved_count += 1
        
        if qrel_dictaionary[query.query_id].get(doc, 0) == 1:
            relevant_count += 1
    if retrieved_count > 0:
        precision = relevant_count / retrieved_count
        precision_list.append(precision)
    
    else: 
        precision_list.append(0)


# Calculate mean precision@k
precision_at_k = sum(precision_list) / len(precision_list)
print("Precision@k:", precision_at_k)




@app.route('/eva', methods=['GET'])
def get_numbers():
    A1 =  map_score
    A2 = mrr_score
    A3 = precision_at_k

    response = {
        'MAP': A1,
        'MRR': A2,
        'precision@k': A3
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)