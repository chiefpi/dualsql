import json

def write_prediction(
        fileptr,
        identifier,
        input_seq,
        prediction,
        gold_query,
        index_in_interaction):
    """E.g.,
        {
            "identifier": "cre_Doc_Template_Mgt/0",
            "database_id": "cre_Doc_Template_Mgt", 
            "interaction_id": "0", 
            "input_seq": ["Show", "information", "for", "all", "documents", "."], 
            "prediction": ["<bos>", "select", "documents.*", "<eos>"], 
            "flat_prediction": ["select", "documents.*"], 
            "gold_query": ["<bos>", "select", "documents.*", "<eos>"], 
            "flat_gold_queries": [["select", "documents.*"]], 
            "index_in_interaction": 0, 
        }
    """
    pred_obj = {}
    pred_obj["identifier"] = identifier
    database_id, interaction_id = identifier.split('/')
    pred_obj["database_id"] = database_id
    pred_obj["interaction_id"] = interaction_id

    pred_obj["input_seq"] = input_seq
    pred_obj["prediction"] = prediction
    pred_obj["gold_query"] = gold_query
    pred_obj["index_in_interaction"] = index_in_interaction

    fileptr.write(json.dumps(pred_obj) + "\n")