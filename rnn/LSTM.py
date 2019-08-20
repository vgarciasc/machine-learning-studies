import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return np.tanh(x)


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
        self.W_hs = np.random.randn(self.state_size, self.state_size) * 0.01  # from state to state
        self.W_sy = np.random.randn(self.char_vocab, self.state_size) * 0.01  # from state to output

        self.b_s = np.zeros((self.state_size, 1))
        self.b_y = np.zeros((self.char_vocab, 1))

        self.W_ih = np.random.randn(self.state_size, self.state_size) * 0.01  # from state to state
        self.W_fh = np.random.randn(self.state_size, self.state_size) * 0.01  # from state to state
        self.W_oh = np.random.randn(self.state_size, self.state_size) * 0.01  # from state to state
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

        state_inc = {}
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
            i_g[t] = sigmoid(self.W_ix @ x[t] + self.W_ih @ h[t - 1] + self.b_i)
            o_g[t] = sigmoid(self.W_ox @ x[t] + self.W_oh @ h[t - 1] + self.b_o)
            f_g[t] = sigmoid(self.W_fx @ x[t] + self.W_fh @ h[t - 1] + self.b_f)

            # ~ state
            state_inc[t] = tanh(self.W_xs @ x[t] + self.W_hs @ h[t - 1] + self.b_s)
            state[t] = (f_g[t] * state[t - 1]) + (i_g[t] * state_inc[t])

            h[t] = o_g[t] * tanh(state[t])

            # ~ output vector
            y[t] = self.W_sy @ h[t] + self.b_y

            # ~ softmaxing and creating probabilities
            o[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))

            # ~ computing loss
            loss += - np.log(o[t][targets[t], 0])

        dW_sy = np.zeros_like(self.W_sy)
        db_y = np.zeros_like(self.b_y)

        dW_hs = np.zeros_like(self.W_hs)
        dW_xs = np.zeros_like(self.W_xs)
        db_s = np.zeros_like(self.b_s)

        dW_fx = np.zeros_like(self.W_fx)
        dW_fh = np.zeros_like(self.W_fh)
        db_f = np.zeros_like(self.b_f)
        dW_ix = np.zeros_like(self.W_ix)
        dW_ih = np.zeros_like(self.W_ih)
        db_i = np.zeros_like(self.b_i)
        dW_ox = np.zeros_like(self.W_ox)
        dW_oh = np.zeros_like(self.W_oh)
        db_o = np.zeros_like(self.b_o)

        ds_next = np.zeros_like(state[0])
        di_next = np.zeros_like(i_g[0])
        df_next = np.zeros_like(f_g[0])
        do_next = np.zeros_like(o_g[0])

        f_g[len(inputs)] = np.zeros_like(f_g[0])

        # Backward pass
        for t in range(len(inputs) - 1, 0, -1):
            dy = np.copy(o[t])
            dy[targets[t]] -= 1  # derivative of loss wrt output through softmax
            dW_sy += dy @ h[t].T
            db_y += dy

            dh = self.W_sy.T @ dy + \
                 self.W_hs.T @ ds_next + \
                 self.W_ih.T @ di_next + \
                 self.W_fh.T @ df_next + \
                 self.W_oh.T @ do_next
            do = dh * tanh(state[t]) * (o_g[t] * (1 - o_g[t]))
            ds = dh * o_g[t] * (1 - state[t] * state[t]) + \
                 ds_next * f_g[t + 1]
            df = ds * state[t - 1] * (f_g[t] * (1 - f_g[t]))
            di = ds * state_inc[t] * (i_g[t] * (1 - i_g[t]))
            ds_inc = ds * i_g[t] * (1 - state_inc[t] * state_inc[t])

            # W
            dW_ox += do @ x[t].T
            dW_ix += di @ x[t].T
            dW_fx += df @ x[t].T
            dW_xs += ds_inc @ x[t].T
            # R
            if t != len(inputs) - 1:
                dW_oh += do_next @ h[t].T
                dW_ih += di_next @ h[t].T
                dW_fh += df_next @ h[t].T
                dW_hs += df_next @ h[t].T
            # b
            db_o += do
            db_i += di
            db_f += df
            db_s += ds_inc

            ds_next = ds
            di_next = di
            df_next = df
            do_next = do

        # Clipping gradients
        for param in [dW_sy, db_y, dW_hs, dW_xs, db_s,
                      dW_fx, dW_fh, db_f, dW_ix, dW_ih,
                      db_i, dW_ox, dW_oh, db_o]:
            np.clip(param, -5, 5, out=param)

        return loss, dW_sy, db_y, dW_hs, dW_xs, db_s, \
               dW_fx, dW_fh, db_f, dW_ix, dW_ih, db_i, \
               dW_ox, dW_oh, db_o, state[len(inputs) - 1]

    def loop(self):
        mW_sy = np.zeros_like(self.W_sy)
        mb_y = np.zeros_like(self.b_y)

        mW_hs = np.zeros_like(self.W_hs)
        mW_xs = np.zeros_like(self.W_xs)
        mb_s = np.zeros_like(self.b_s)

        mW_fx = np.zeros_like(self.W_fx)
        mW_fh = np.zeros_like(self.W_fh)
        mb_f = np.zeros_like(self.b_f)
        mW_ix = np.zeros_like(self.W_ix)
        mW_ih = np.zeros_like(self.W_ih)
        mb_i = np.zeros_like(self.b_i)
        mW_ox = np.zeros_like(self.W_ox)
        mW_oh = np.zeros_like(self.W_oh)
        mb_o = np.zeros_like(self.b_o)

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
            loss, dW_sy, db_y, dW_hs, \
            dW_xs, db_s, dW_fx, dW_fh, \
            db_f, dW_ix, dW_ih, db_i, \
            dW_ox, dW_oh, db_o, state = self.iteration(input_vector, target_vector, state)

            # Update loss
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Parameter update (Adagrad)
            for param, dparam, mem in zip(
                    [self.W_sy, self.b_y, self.W_hs, self.W_xs, self.b_s, self.W_fx, self.W_fh, self.b_f, self.W_ix,
                     self.W_ih, self.b_i, self.W_ox, self.W_oh, self.b_o],
                    [dW_sy, db_y, dW_hs, dW_xs, db_s, dW_fx, dW_fh, db_f, dW_ix, dW_ih, db_i, dW_ox, dW_oh, db_o],
                    [mW_sy, mb_y, mW_hs, mW_xs, mb_s, mW_fx, mW_fh, mb_f, mW_ix, mW_ih, mb_i, mW_ox, mW_oh, mb_o]):
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
            state = np.tanh(np.dot(self.W_xs, x) + np.dot(self.W_hs, state) + self.b_s)

        for t in range(0, length):
            state = np.tanh(np.dot(self.W_xs, x) + np.dot(self.W_hs, state) + self.b_s)
            y = np.dot(self.W_sy, state) + self.b_y
            o = np.exp(y) / np.sum(np.exp(y))

            sampled_letter = np.random.choice(range(self.char_vocab), p=o.ravel())
            output += self.index_to_char[sampled_letter]

            x = np.zeros((self.char_vocab, 1))
            x[sampled_letter] = 1

        return output


if __name__ == "__main__":
    # filename = "C:\\Users\\patyc\\Documents\\mablelrosk.txt"
    # filename = "C:\\Users\\patyc\\Documents\\GitHub\\rnn-karpathy\\shakespeare.txt"
    filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\01011001.txt"

    rnn = LSTM(filename, state_size=100, step_size=8, learning_rate=2e-2)

    rnn.loop()
