import numpy as np


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


class LSTM():
    def __init__(self, filename, state_size, step_size, learning_rate):
        self.input_text = open(filename, "r", encoding="utf8").read()
        self.used_chars = list(set(self.input_text))
        self.char_vocab = len(self.used_chars)
        self.char_to_index = {ch: i for i, ch in enumerate(self.used_chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.used_chars)}

        self.state_size = state_size
        self.step_size = step_size
        self.learning_rate = learning_rate

        self.W_xs = np.random.randn(self.state_size, self.char_vocab) * 0.01  # from input to state
        self.W_ss = np.random.randn(self.state_size, self.state_size) * 0.01  # from state to state
        self.W_sy = np.random.randn(self.char_vocab, self.state_size) * 0.01  # from state to output

        self.b_s = np.zeros((self.state_size, 1))
        self.b_y = np.zeros((self.char_vocab, 1))

        self.W_is = np.random.randn(self.state_size, self.state_size) * 0.01  # from state to state
        self.W_fs = np.random.randn(self.state_size, self.state_size) * 0.01  # from state to state
        self.W_os = np.random.randn(self.state_size, self.state_size) * 0.01  # from state to state
        self.W_ix = np.random.randn(self.state_size, self.char_vocab) * 0.01  # from state to state
        self.W_fx = np.random.randn(self.state_size, self.char_vocab) * 0.01  # from state to state
        self.W_ox = np.random.randn(self.state_size, self.char_vocab) * 0.01  # from state to state

        self.b_i = np.zeros((self.state_size, 1))
        self.b_o = np.zeros((self.state_size, 1))
        self.b_f = np.zeros((self.state_size, 1))

    def iteration(self, inputs, targets, initial_state):
        x = {}  # input
        y = {}  # output (raw)
        o = {}  # output (as probabilities)

        i_g = {}  # input gate (write)
        o_g = {}  # output gate (read)
        f_g = {}  # forget gate

        loss = 0

        delta_state = {}
        state = {}
        h = {}  # hidden state

        h[-1] = initial_state
        state[-1] = initial_state  # setting initial state

        # Forward pass
        for t, char_key in enumerate(inputs):
            # ~ input vector
            # eg:
            #   char_key = 1; char_vocab = 3; char_to_index = {a: 0, b: 1, c: 2}
            #   x = [0, 1, 0] (one-hot)
            x[t] = np.zeros((self.char_vocab, 1))
            x[t][char_key] = 1

            # ~ gates
            i_g[t] = sigmoid(np.dot(self.W_ix, x[t]) + np.dot(self.W_is, h[t - 1]) + self.b_i)
            o_g[t] = sigmoid(np.dot(self.W_ox, x[t]) + np.dot(self.W_os, h[t - 1]) + self.b_o)
            f_g[t] = sigmoid(np.dot(self.W_fx, x[t]) + np.dot(self.W_fs, h[t - 1]) + self.b_f)

            # ~ state
            delta_state[t] = np.tanh(np.dot(self.W_xs, x[t]) + np.dot(self.W_ss, h[t - 1]) + self.b_s)
            state[t] = np.multiply(f_g[t], state[t - 1]) + np.multiply(i_g[t], delta_state[t])

            h[t] = np.multiply(o_g[t], np.tanh(state[t]))

            # ~ output vector
            y[t] = np.dot(self.W_sy, h[t]) + self.b_y

            # ~ softmaxing and creating probabilities
            o[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))

            # ~ computing loss
            loss += - np.log(o[t][targets[t], 0])

        dW_xs = np.zeros_like(self.W_xs)
        dW_ss = np.zeros_like(self.W_ss)
        dW_sy = np.zeros_like(self.W_sy)

        db_s = np.zeros_like(self.b_s)
        db_y = np.zeros_like(self.b_y)
        ds_next = np.zeros_like(state[0])

        # Backward pass
        for t in range(len(inputs) - 1, 0, -1):
            # ~ backpropagate into y
            dy = np.copy(o[t])
            dy[targets[t]] -= 1
            dW_sy += np.dot(dy, state[t].T)
            db_y += dy

            # ~ backpropagate into state
            ds = np.dot(self.W_sy.T, dy) + ds_next
            ds_raw = (1 - state[t] * state[t]) * ds  # d(tanh)
            db_s += ds_raw
            dW_ss += np.dot(ds_raw, state[t - 1].T)
            ds_next = np.dot(self.W_ss.T, ds_raw)

            # ~ backpropagate into x
            dW_xs += np.dot(ds_raw, x[t].T)

        # Clipping gradients
        for param in [dW_xs, dW_ss, dW_sy, db_s, db_y]:
            np.clip(param, -5, 5, out=param)

        return loss, dW_xs, dW_ss, dW_sy, db_s, db_y, state[len(inputs) - 1]

    def loop(self):
        mW_xs = np.zeros_like(self.W_xs)
        mW_ss = np.zeros_like(self.W_ss)
        mW_sy = np.zeros_like(self.W_sy)

        mb_s = np.zeros_like(self.b_s)
        mb_y = np.zeros_like(self.b_y)

        current_index = 0
        current_step = 0
        smooth_loss = - np.log(1.0 / self.char_vocab) * self.step_size

        while True:
            if current_index + self.step_size + 1 >= len(self.input_text):
                # Loops over training data
                current_index = 0

            if current_index == 0:
                # Setting up initial state
                state = np.zeros((self.state_size, 1))

            # Vectorize input
            input_vector = [self.char_to_index[char] for char in
                            self.input_text[current_index: current_index + self.step_size]]
            target_vector = [self.char_to_index[char] for char in
                             self.input_text[current_index + 1: current_index + self.step_size + 1]]

            # Forward #step_size characters through the network
            loss, dW_xs, dW_ss, dW_sy, db_s, db_y, state = self.iteration(input_vector, target_vector, state)

            # Update loss
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Parameter update (Adagrad)
            for param, dparam, mem in zip([self.W_xs, self.W_ss, self.W_sy, self.b_s, self.b_y],
                                          [dW_xs, dW_ss, dW_sy, db_s, db_y],
                                          [mW_xs, mW_ss, mW_sy, mb_s, mb_y]):
                mem += dparam * dparam
                param += - self.learning_rate * dparam / np.sqrt(mem + 1e-8)

            # Managing counters
            current_step += 1
            current_index += self.step_size

            if current_step % 100 == 0:
                sample = self.sample(state, self.index_to_char[input_vector[0]], 200)
                print("------------")
                print("step #", current_step, "loss: ", smooth_loss)
                print(sample)

    def sample(self, state, seed_str, length):
        seed = [self.char_to_index[char] for char in seed_str]
        output = seed_str

        x = None

        # Advance state as if has read 'seed_str'
        for t in range(0, len(seed)):
            x = np.zeros((self.char_vocab, 1))
            x[seed[t]] = 1
            state = np.tanh(np.dot(self.W_xs, x) + np.dot(self.W_ss, state) + self.b_s)

        for t in range(0, length):
            state = np.tanh(np.dot(self.W_xs, x) + np.dot(self.W_ss, state) + self.b_s)
            y = np.dot(self.W_sy, state) + self.b_y
            o = np.exp(y) / np.sum(np.exp(y))

            sampled_letter = np.random.choice(range(self.char_vocab), p=o.ravel())
            output += self.index_to_char[sampled_letter]

            x = np.zeros((self.char_vocab, 1))
            x[sampled_letter] = 1

        return output


if __name__ == "__main__":
    # filename = "C:\\Users\\patyc\\Documents\\mablelrosk.txt"
    filename = "C:\\Users\\patyc\\Documents\\GitHub\\rnn-karpathy\\shakespeare.txt"

    rnn = LSTM(filename, state_size=512, step_size=25, learning_rate=1e-1)

    rnn.loop()
