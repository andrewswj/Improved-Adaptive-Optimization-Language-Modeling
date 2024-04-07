import torch
from corpus import Corpus
from utils import batchify, get_batch, repackage_hidden, load_model, convert_to_words
from model import RNNModel
import numpy as np
import random
from criterion import SplitCrossEntropyLoss


def generate_text(model, seed_text, max_length, block_unk = False) -> None:
    # Tokenization
    seed_tokens = corpus.tokenize_words(seed_text)
    seed_input = torch.tensor(seed_tokens)
    tokens = seed_input.view(-1, 1)

    # Generate text
    generated_text = [seed_text]

    with torch.no_grad():
        for _ in range(max_length):
            # Initialize hidden state with the batch size. In this case, batch size is 1.
            hidden = model.init_hidden(1)

            # Forward pass through the model
            topk_tokens, next_token = model(tokens, hidden)

            # Check if next_token is 26
            if block_unk:
                if next_token.item() == 26:
                    selected_token = topk_tokens[0, 1]  # Select the second best token
                else:
                    selected_token = next_token
            
            else:
                selected_token = next_token
            
            # Decode the next token and add it to the generated text
            generated_tokens = [selected_token.item()]  # [token.item() for token in next_token]
            generated_words = [corpus.dictionary.idx2word[token] for token in generated_tokens]

            for word in generated_words:
                generated_text.append(word)

            # Append the generated tokens to the tensor and continue the loop
            tokens = torch.cat((tokens, torch.tensor([[next_token]])))

    print(' '.join(generated_text))


if __name__ == '__main__':
    # Set random seed for Python's built-in random module
    random.seed(123)

    # Set random seed for NumPy
    np.random.seed(123)

    # Set random seed for PyTorch
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable cudnn benchmark for deterministic behavior

    # Initialize the corpus and batchify the data
    corpus = Corpus()
    train_data = batchify(corpus.train, 10)
    valid_data = batchify(corpus.valid, 10)
    test_data = batchify(corpus.test, 10)
    ntokens = len(corpus.dictionary)

    # Load the models with separate instances
    basemodel = RNNModel(ntokens, 400, 1150, 3)
    lmmodelsqrtx = RNNModel(ntokens, 400, 1150, 3)
    lmmodelcuberoot = RNNModel(ntokens, 400, 1150, 3)

    # Define splits if necessary, and apply the SplitCrossEntropyLoss
    splits = []
    if ntokens > 500000:
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        splits = [2800, 20000, 76000]
    criterion = SplitCrossEntropyLoss(400, splits=splits, verbose=False)

    def my_function(x):  # Throwaway function to match the model parameters. Not used in evaluation.
        return x
    
    seed_text = "hesitate to finance a transaction the pilots oppose <eos> also because ual chairman stephen wolf and other ual executives have joined the pilots ' bid the board might be forced to exclude him from its deliberations in order to be fair to other bidders <eos> that could cost him the chance to influence the outcome and perhaps join the winning bidder <eos> influential members of the house ways and means"
    max_length = 100  # Maximum length of generated text

    load_model(basemodel, 'model/lstm_adabelief_3layer')
    load_model(lmmodelsqrtx, 'model/lstm_adabelief_lm_3layer_sqrtx')
    load_model(lmmodelcuberoot, 'model/lstm_adabelief_lm_3layer_cuberoot')

    # Generate text
    generate_text(basemodel, seed_text, max_length)
    generate_text(lmmodelsqrtx, seed_text, max_length)
    generate_text(lmmodelcuberoot, seed_text, max_length)