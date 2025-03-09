import train
edge_to_str_name_mapping,gF=train.main()

predicted_scores = []
true_weights = []


for edge_id, (src_name, dst_name) in edge_to_str_name_mapping.items():
    predicted_score = gF.edges[edge_id].data['score'].item()
    true_weight = gF.edges[edge_id].data['weight'].item()
    predicted_scores.append(predicted_score)
    true_weights.append(true_weight)


# Define the threshold and tolerance
threshold = 35.0
tolerance = 0.0

amb_lower = threshold - tolerance
amb_upper = threshold + tolerance

fuzzy_correct = 0
for pred, true in zip(predicted_scores, true_weights):
    true_class = 1 if true > threshold else 0
    pred_class = 1 if pred > threshold else 0

    if amb_lower <= pred <= amb_upper:
        fuzzy_correct += 1
    else:
        if pred_class == true_class:
            fuzzy_correct += 1

fuzzy_accuracy = fuzzy_correct / len(true_weights)
print(f"Classification Accuracy : {fuzzy_accuracy:.4f}")