import torch
import json
from azureml.core import Model
from transformers import BertTokenizer, BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from torch.nn import Softmax


def init():
    """
    This function loads the bert model, bert tokenizer, and class names from the model
    directory. The contents of the folder can also be seen under Artifacts in the model registry.
    """

    global model
    global tokenizer
    global classes

    model_dir = Model.get_model_path(model_name='page_binary_bert', version=2)

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)


def run(texts):
    """
    This function is run when the service is called. It takes the input text (cleaned html)
    and predicts whether the page text is relevant for use in ml or not.
    :param text: list or array of strings, one or more clean html texts
    :return: output, the predicted labels
    """

    # Deserialize the input url
    texts = json.loads(texts)

    # Tokenize
    input_ids = []
    for t in texts:
        input_ids.append(tokenizer.encode(t, add_special_tokens=True))

    # Maximum input length is the length of the longest input sentence (after tokenization)
    max_len = max([len(x) for x in input_ids])
    # Max len cannot be more than the models's max, 512
    max_len = min(max_len, 512)

    # Truncate with head+tail strategy if max len is 512
    if max_len == 512:
        input_trunc = []
        for x in input_ids:
            if len(x) > max_len:
                trunc_x = x[:129] + x[-383:]
                input_trunc.append(trunc_x)
            else:
                input_trunc.append(x)
    else:
        input_trunc = input_ids

    # Pad texts shorter than max len
    input_trunc = pad_sequences(input_trunc, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")

    # Convert to PyTorch tensor
    input_tensors = torch.tensor(input_trunc)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(input_tensors)[0]
        # Get the probabilities
        softmax = Softmax(dim=1)
        softmaxed = softmax(logits)
        # Convert to numpy array
        softmaxed_array = softmaxed.cpu().detach().numpy()
        # Return the probabilities for both classes
        preds_strings = [["%.3f" % number for number in j] for j in softmaxed_array]
        labels = ['not relevant', 'relevant']
        output = [dict(zip(labels, s)) for s in preds_strings]
        return output