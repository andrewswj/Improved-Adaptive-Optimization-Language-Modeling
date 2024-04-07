import torch


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, mean_bptt = 70, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else mean_bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def load_model(model, file_path):
    """
    Load weights into a model from a file.

    Args:
    - model: The model instance to load weights into.
    - file_path: The path to the file containing the saved weights.

    Returns:
    - model: The model instance with loaded weights.
    """
    # Load the state dictionary
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

    # Custom loading function to handle WeightDrop
    def load_custom(model, state_dict):
        for name, param in model.named_parameters():
            if 'weight_hh' in name and 'weight_hh' not in state_dict:
                continue
            param.data.copy_(state_dict[name])

    # Load custom parameters
    load_custom(model, checkpoint['model_state_dict'])

    return model


def convert_to_words(input_tensor, output_tensor, corpus):
    # Convert input tensor to words
    input_words = [corpus.dictionary.idx2word[idx.item()] for idx in input_tensor]

    # Convert output tensor to words
    output_word_idx = output_tensor.argmax().item()
    output_word = corpus.dictionary.idx2word[output_word_idx]

    topk_words = [corpus.dictionary.idx2word[idx.item()] for idx in output_tensor]

    return input_words, output_word, topk_words