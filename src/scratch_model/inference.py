import pandas as pd
import torch

from config import Config
from data.tokenizer import Tokenizer
from data.dataset import MCQDataset
from model_src.classifier import TransformerClassifier


OPTION_MAP = ["A", "B", "C", "D", "E"]


def predict():

    device = Config.DEVICE

    test_df = pd.read_csv("dataset/test.csv")

    # Load the trained model
    checkpoint = torch.load(
        "models/best_model.pth",
        map_location=device,
        weights_only=False
    )

    #use the same vocabulary generated while training
    vocab = checkpoint["vocab"]

    tokenizer = Tokenizer()

    # Build model
    model = TransformerClassifier(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        max_seq_length=Config.MAX_LEN,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        ff_dim=Config.FF_DIM,
        dropout=Config.DROPOUT
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = MCQDataset(
        test_df,
        tokenizer,
        vocab,
        Config.MAX_LEN
    )

    predictions = []

    #disable gradient computation for inference
    with torch.no_grad():

        for sample in dataset:

            #unsqueeze(0) converts shape from (num_choice(ie.5), SEQ_LEN) to (1, 5, SEQ_LEN)
            input_ids = sample["input_ids"].unsqueeze(0).to(device)

            logits = model(input_ids)

            #get the indices for top k=3 highest scoring predictions
            top3 = torch.topk(logits, k=3, dim=1).indices.squeeze(0)


            #Convert the numerical predicitons back to letter mappings
            prediction = " ".join(OPTION_MAP[idx] for idx in top3.tolist())
            predictions.append(prediction)

    # Kaggle submission
    submission = pd.DataFrame({
        "ID": test_df["id"],
        "Prediction": predictions
    })

    submission.to_csv("submission.csv", index=False)

    print("Submission saved to submission.csv")


if __name__ == "__main__":
    predict()
