import numpy as np

from my_lstm.optimization import adam
from my_lstm.LSTM import LSTM

if __name__ == '__main__':
    n_iter = 1000000000
    print_after = 1000

    time_step = 10
    state_size = 64
    learning_rate = 1e-3

    # filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\abba baba.txt"
    filename = "C:\\Users\\patyc\\Documents\\GitHub\\rnn-karpathy\\shakespeare.txt"

    with open(filename, 'r', encoding="utf8") as f:
        input_text = f.read()

        char_to_index = {char: i for i, char in enumerate(set(input_text))}
        index_to_char = {i: char for i, char in enumerate(set(input_text))}

        inputs = [char_to_index[char] for char in input_text]
        targets = [char_to_index[char] for char in input_text[1:]]

        # targets must be offset by 1 wrt inputs (one timestep ahead)
        if ' ' in set(input_text):
            last_char = char_to_index[' ']
        else:
            last_char = char_to_index[list(set(input_text))[0]]
        targets.append(last_char)

        inputs = np.array(inputs)
        targets = np.array(targets)

    char_vocab = len(char_to_index)

    lstm = LSTM(
        V=char_vocab,
        S=state_size,
        char_to_index=char_to_index,
        index_to_char=index_to_char)

    adam(
        lstm,
        inputs, targets,
        learning_rate=learning_rate,
        mb_size=time_step,
        iterations=n_iter,
        print_every=print_after
    )