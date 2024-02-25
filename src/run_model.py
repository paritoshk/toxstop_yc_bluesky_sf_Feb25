import random
import torch
from torch import nn
import os
import nltk
import requests

from craft_inference import (
    batch2TrainData,
    loadPrecomputedVoc,
    WORD2INDEX_URL,
    INDEX2WORD_URL,
    FORECAST_THRESH,
    MAX_LENGTH,
    urlretrieve,
    MODEL_URL,
    unicodeToAscii,
    batchIterator,
    dialogBatch2UtteranceBatch,
)
from craft_model import EncoderRNN, ContextEncoderRNN, SingleTargetClf, Predictor


def load_model(voc, device):
    # configure model
    hidden_size = 500
    encoder_n_layers = 2
    context_encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    print("Loading saved parameters...")
    if not os.path.isfile("model.tar"):
        print("\tDownloading trained CRAFT...")
        urlretrieve(MODEL_URL, "model.tar")
        print("\t...Done!")
    checkpoint = torch.load("model.tar")
    # If running in a non-GPU environment, you need to tell PyTorch to convert the parameters to CPU tensor format.
    # To do so, replace the previous line with the following:
    # checkpoint = torch.load("model.tar", map_location=torch.device('cpu'))
    encoder_sd = checkpoint["en"]
    context_sd = checkpoint["ctx"]
    attack_clf_sd = checkpoint["atk_clf"]
    embedding_sd = checkpoint["embedding"]
    voc.__dict__ = checkpoint["voc_dict"]

    print("Building encoders, decoder, and classifier...")
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    # Initialize utterance and context encoders
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    context_encoder.load_state_dict(context_sd)
    # Initialize classifier
    attack_clf = SingleTargetClf(hidden_size, dropout)
    attack_clf.load_state_dict(attack_clf_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    context_encoder = context_encoder.to(device)
    attack_clf = attack_clf.to(device)
    print("Models built and ready to go!")

    # Set dropout layers to eval mode
    encoder.eval()
    context_encoder.eval()
    attack_clf.eval()

    # Initialize the pipeline
    predictor = Predictor(encoder, context_encoder, attack_clf)
    return predictor


def tokenize(text, voc):
    tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r"\w+|[^\w\s]")
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text.lower())

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []

    return [
        token if token in voc.word2index else "<UNK>"
        for token in tokenizer.tokenize(cleaned_text)
    ]


def setup_conversation(conversation, voc):
    """Conversation is a list of strings, each string is one turn in the conversation.

    returns context, reply tokenized
    """
    # Tokenize the input
    utterances = [tokenize(utterance, voc) for utterance in conversation]
    return utterances, [], None, None


def run(conversations, predictor):
    # Run the pipeline!
    batch_dialogs = [setup_conversation(convo, voc) for convo in conversations]
    batch_tensors = batch2TrainData(voc, batch_dialogs, already_sorted=True)
    input_batch, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, convo_ids, target_variable, mask, max_target_len = batch_tensors
    dialog_lengths_list = [len(x[0]) for x in batch_dialogs]

    input_batch = input_batch.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    # Predict future attack using predictor
    scores = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, len(batch_dialogs), max_length=MAX_LENGTH)
    predictions = (scores > 0.5).float()
    return predictions, scores


# Fix random state for reproducibility
random.seed(2019)
# First, we need to build the vocabulary so that we know how to map tokens to tensor indicies.
# For the sake of replicating the paper results, we will load the pre-computed vocabulary objects used in the paper.
voc = loadPrecomputedVoc("wikiconv", WORD2INDEX_URL, INDEX2WORD_URL)
# Tell torch to use GPU. Note that if you are running this notebook in a non-GPU environment, you can change 'cuda' to 'cpu' to get the code to run.
device = torch.device("cuda")
predictor = load_model(voc, device)