
def db_upsert(index, gallery_emb:dict):

    # Convert data to Pinecone's required format
    vectors_to_upsert = [(id, vector.tolist()) for id, vector in gallery_emb.items()]   # list of tuples

    len_vec = len(vectors_to_upsert)

    # Upsert the vectors
    for i in range(0, len_vec, 50):
        index.upsert(vectors=vectors_to_upsert[i:i+50])


def db_query(index, img_emb, top_k=1):

    # return list of tuples contains img path and score of similarity
    id_scores = []

    # convert the tensor to list
    vector = img_emb.tolist()

    # make the query
    result = index.query(vector=vector, top_k=top_k, includeMetadata=True)
    
    matches = result['matches']     # list of dict
    for match in matches:
        id_scores.append((match['id'], match['score']))     

    return id_scores