import torch

def accuracy(logits, labels):
    """
    Computes classification accuracy.
    compares predicted answer with correct answer
    """

    predictions = torch.argmax(logits, dim=1)

    correct = (predictions == labels).sum().item()

    return correct / len(labels)


def map_at_3(logits, labels):
    """
    Computes MAP@3 (Mean Average Precision) for multiple-choice predictions.
    If correct answer at first place in the prediction then awarded 1
    If at second position then 0.5
    If at third position then 0.33
    else 0 score is awarded
    """

    # Get the indices of the three highest-scoring predictions.
    top3 = torch.topk(logits, k=3, dim=1).indices

    score = 0

    for prediction, target in zip(top3, labels):

        prediction = prediction.tolist()
        target = target.item()

        if target == prediction[0]:
            score += 1

        elif target == prediction[1]:
            score += 0.5

        elif target == prediction[2]:
            score += 1 / 3

    return score / len(labels)
