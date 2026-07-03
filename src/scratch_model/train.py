import pandas as pd
import torch
import torch.nn as nn
import wandb


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import Config
from data.tokenizer import Tokenizer
from data.vocabulary import Vocabulary
from data.dataset import MCQDataset

from model_src.classifier import TransformerClassifier
from metrics import accuracy, map_at_3


def train():

    device = Config.DEVICE

    train_df = pd.read_csv("dataset/train.csv")

    train_df, val_df = train_test_split(train_df, test_size=Config.TEST_SIZE, random_state=Config.SEED, stratify=train_df["answer"])

    # Tokenizer & Vocabulary
    tokenizer = Tokenizer()
    vocab = Vocabulary(min_freq=Config.MIN_FREQ)

    texts = []

    texts.extend(train_df["prompt"].tolist())

    for col in ["A", "B", "C", "D", "E"]:
        texts.extend(train_df[col].tolist())

    vocab.build(texts, tokenizer)

    print(f"Vocabulary Size : {len(vocab)}")

    # Dataset
    train_dataset = MCQDataset(train_df, tokenizer, vocab, Config.MAX_LEN)

    val_dataset = MCQDataset(val_df, tokenizer, vocab, Config.MAX_LEN)

    # DataLoader

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Model
    model = TransformerClassifier(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        max_seq_length=Config.MAX_LEN,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        ff_dim=Config.FF_DIM,
        dropout=Config.DROPOUT
    ).to(device)

    # Loss
    # Cross-entropy compares predicted logits with the correct answer index.
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    #Scheduler
    #Gradually decreases the LR following a cosine curve
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS
    )

    # Training Loop
    best_val_loss = float("inf")

    # wandb.init(
    #     entity="23f2001083-dl-genai-project",
    #     project="23f2001083-t22026",
    #     config={
    #         "embedding_dim": Config.EMBEDDING_DIM,
    #         "num_heads": Config.NUM_HEADS,
    #         "num_layers": Config.NUM_LAYERS,
    #         "ff_dim": Config.FF_DIM,
    #         "batch_size": Config.BATCH_SIZE,
    #         "learning_rate": Config.LR,
    #         "epochs": Config.EPOCHS,
    #         "max_len": Config.MAX_LEN,
    #         "dropout": Config.DROPOUT,
    #     }
    # )
    # wandb.watch(
    #     model,
    #     criterion,
    #     log="all",
    #     log_freq=100
    # )

    for epoch in range(Config.EPOCHS):

        model.train()

        train_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids)

            loss = criterion(logits, labels)

            loss.backward()

            #gradient clipping to prevent gradient exploding and improve training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        # Validation
        model.eval()

        val_loss = 0
        val_acc = 0
        val_map3 = 0

        # Disable gradient calculation to save memory and compute during validation
        with torch.no_grad():

            for batch in val_loader:

                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids)

                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_acc += accuracy(logits, labels)
                val_map3 += map_at_3(logits, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_map3 /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
            f" | Acc {val_acc:.4f} "
            f" | MAP@3 {val_map3:.4f}"

        )
        # wandb.log({
        #     "train_loss": train_loss,
        #     "val_loss": val_loss,
        #     "val_accuracy": val_acc,
        #     "val_map3": val_map3,
        #     "epoch": epoch + 1,
        #     "learning_rate": optimizer.param_groups[0]["lr"]

        # })   

        scheduler.step()     

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab": vocab,
                "config": Config
            }

            torch.save(checkpoint, "models/best_model.pth")
    print("Training Complete!")
    # wandb.finish()

if __name__ == "__main__":
    train()
