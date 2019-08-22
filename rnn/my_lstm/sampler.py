import numpy as np
import pickle

from my_lstm.LSTM import LSTM

if __name__ == '__main__':
    state_size = 64

    filename = "C:\\Users\\patyc\\Documents\\GitHub\\rnn-karpathy\\shakespeare.txt"

    with open(filename, 'r', encoding="utf8") as f:
        input_text = f.read()

        char_to_index = {char: i for i, char in enumerate(set(input_text))}
        index_to_char = {i: char for i, char in enumerate(set(input_text))}

    char_vocab = len(char_to_index)

    lstm = LSTM(
        V=char_vocab,
        S=state_size,
        char_to_index=char_to_index,
        index_to_char=index_to_char)

    model_filename = 'C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\data.pickle'

    lstm.load_model(model_filename)

    # lstm.model = model
    seed = [lstm.char_to_index[input_text[0]]]

    sample = lstm.sample(seed, lstm.initial_state(), 500)
    print(sample)