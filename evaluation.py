import numpy as np
import pickle


def calc_raw_accuracy(preds, gts):
    raw_scores = []
    for i in range(len(preds)):
        gt = gts[i][2]
        logits = preds[i].cpu().numpy()
        prediction = np.argmax(logits)
        raw_scores.append(prediction == gt)
    return np.mean(raw_scores) * 100

def evaluate(preds):
    gts = pickle.load(open("data/val.pkl", "rb"))
    raw_accuracy = calc_raw_accuracy(preds, gts)
    max_probs = []
    for i in range(0, len(gts), 15):
        gt = gts[i][2]
        logits = np.array([preds[i + j].cpu().numpy() for j in range(15)])
        logits_summed = np.sum(logits, 0)
        prediction = np.argmax(logits_summed)
        max_probs.append(prediction == gt)
    max_prob_accuracy = np.mean(max_probs) * 100
    return raw_accuracy, max_prob_accuracy

# def evaluate(preds, gts_path):
#     """
#     Given the list of all model outputs (logits), and the path to the ground 
#     truth (val.pkl), calculate the percentage of correctly classified segments
#     (model accuracy).
#     Args:
#         preds (List[torch.Tensor]): The model ouputs (logits). This is a 
#             list of all the tensors produced by the model for all samples in
#             val.pkl. It should be a list of length 3750 (size of val). All 
#             tensors in the list should be of size 10 (number of classes).
#         gts_path (str): The path to val.pkl
#     Returns:
#         raw_score (float): A float representing the percentage of correctly
#             classified segments in val.pkl
#     """
#     gts = pickle.load(open(gts_path, 'rb')
#                       )  # Ground truth labels, pass path to val.pkl

#     raw_scores = []
#     for i in range(len(preds)):
#         # Ground truth of form (filename, spectrogram, label, samples)
#         filename, _, gt, _ = gts[i]
#         # A 10D vector that assigns probability to each class
#         logits = preds[i].cpu().numpy()
#         # Most confident class is the model prediction
#         prediction = np.argmax(logits)
#         # Boolean: 1 if prediction is correct, 0 if incorrect
#         raw_scores.append((prediction == gt))

#     print("ACCUARCY SCORES:")
#     print("-------------------------------------------------------------")
#     print()
#     print('RAW: {:.2f}'.format(np.mean(raw_scores) * 100.0))
#     print()
#     print("-------------------------------------------------------------")

#     # Return scores if you wish to save to a file
#     return np.mean(raw_scores) * 100.0
