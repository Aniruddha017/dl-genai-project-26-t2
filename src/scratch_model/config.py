import torch


class Config:

    # Data
    MAX_LEN = 128       #Max token sequence len
    BATCH_SIZE = 32     #samples per batch
    TEST_SIZE = 0.1     # size of validation data

    # Vocabulary
    MIN_FREQ = 2        #Min word freq for a word to be included in the vocab

    # Transformer
    EMBEDDING_DIM = 256 #Dimension of token embeddings
    NUM_HEADS = 4       #Num of attention heads
    NUM_LAYERS = 2      #Num of transformer encoder blocks
    FF_DIM = 1024       #hidden dimension of feed forward network
    DROPOUT = 0.2       # dropout probability

    # Training
    EPOCHS = 3          
    LR = 3e-4           # learning rate
    WEIGHT_DECAY = 1e-4 #L2 regularization strength

    #device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # seed value for reproducability
    SEED = 42

