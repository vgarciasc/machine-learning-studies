import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


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

    def iteration(self, inputs, targets, initial_state, initial_hidden_state):
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

        state[-1] = initial_state  # setting initial state
        h[-1] = initial_hidden_state

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
        dh_next = np.zeros_like(h[0])

        do = np.zeros_like(state[0])
        df = np.zeros_like(state[0])
        di = np.zeros_like(state[0])
        ds_inc = np.zeros_like(state[0])

        # Backward pass
        for t in range(len(inputs) - 1, 0, -1):
            dW_oh += do @ h[t].T
            dW_hs += ds_inc @ h[t].T
            dW_ih += di @ h[t].T
            dW_fh += df @ h[t].T

            dy = np.copy(o[t])
            dy[targets[t]] -= 1  # derivative of loss wrt output through softmax

            dW_sy += dy @ h[t].T
            db_y += dy

            dh = self.W_sy.T @ dy + dh_next

            do = dh * tanh(state[t]) * (o_g[t] * (1 - o_g[t]))
            ds = dh * o_g[t] * (1 - tanh(state[t]) * tanh(state[t])) + ds_next
            df = ds * state[t - 1] * (f_g[t] * (1 - f_g[t]))
            di = ds * state_inc[t] * (i_g[t] * (1 - i_g[t]))
            ds_inc = ds * i_g[t] * (1 - state_inc[t] * state_inc[t])

            # W
            dW_fx += df @ x[t].T
            db_f += df
            dX_f = self.W_fh.T @ df

            dW_ix += di @ x[t].T
            db_i += di
            dX_i = self.W_ih.T @ di

            dW_ox += do @ x[t].T
            db_o += do
            dX_o = self.W_oh.T @ do

            dW_xs += ds_inc @ x[t].T
            db_s += ds_inc
            dX_s = self.W_hs.T @ ds_inc

            dh_next = dX_f + dX_i + dX_o + dX_s
            ds_next = f_g[t] * ds

        # Clipping gradients
        for param in [dW_sy, db_y, dW_hs, dW_xs, db_s,
                      dW_fx, dW_fh, db_f, dW_ix, dW_ih,
                      db_i, dW_ox, dW_oh, db_o]:
            np.clip(param, -5, 5, out=param)

        return loss, dW_sy, db_y, dW_hs, dW_xs, db_s, \
               dW_fx, dW_fh, db_f, dW_ix, dW_ih, db_i, \
               dW_ox, dW_oh, db_o, \
               state[len(inputs) - 1], \
               h[len(inputs) - 1]

    def loop(self):
        mW_sy, rW_sy = np.zeros_like(self.W_sy), np.zeros_like(self.W_sy)
        mb_y, rb_y = np.zeros_like(self.b_y), np.zeros_like(self.b_y)

        mW_hs, rW_hs = np.zeros_like(self.W_hs), np.zeros_like(self.W_hs)
        mW_xs, rW_xs = np.zeros_like(self.W_xs), np.zeros_like(self.W_xs)
        mb_s, rb_s = np.zeros_like(self.b_s), np.zeros_like(self.b_s)

        mW_fx, rW_fx = np.zeros_like(self.W_fx), np.zeros_like(self.W_fx)
        mW_fh, rW_fh = np.zeros_like(self.W_fh), np.zeros_like(self.W_fh)
        mb_f, rb_f = np.zeros_like(self.b_f), np.zeros_like(self.b_f)
        mW_ix, rW_ix = np.zeros_like(self.W_ix), np.zeros_like(self.W_ix)
        mW_ih, rW_ih = np.zeros_like(self.W_ih), np.zeros_like(self.W_ih)
        mb_i, rb_i = np.zeros_like(self.b_i), np.zeros_like(self.b_i)
        mW_ox, rW_ox = np.zeros_like(self.W_ox), np.zeros_like(self.W_ox)
        mW_oh, rW_oh = np.zeros_like(self.W_oh), np.zeros_like(self.W_oh)
        mb_o, rb_o = np.zeros_like(self.b_o), np.zeros_like(self.b_o)

        current_index = 0
        current_step = 1
        smooth_loss = - np.log(1.0 / self.char_vocab) * self.step_size

        beta1 = .9
        beta2 = .999

        while True:
            if current_index + self.step_size + 1 >= len(self.input_text):
                # Loops over training data
                current_index = 0

            if current_index == 0:
                # Setting up initial state
                state = np.zeros((self.state_size, 1))
                hidden_state = np.zeros((self.state_size, 1))

            # Vectorize input
            input_vector = [self.char_to_index[char] for char in
                            self.input_text[current_index: current_index + self.step_size]]
            target_vector = [self.char_to_index[char] for char in
                             self.input_text[current_index + 1: current_index + self.step_size + 1]]

            # Forward #step_size characters through the network
            loss, dW_sy, db_y, dW_hs, \
            dW_xs, db_s, dW_fx, dW_fh, \
            db_f, dW_ix, dW_ih, db_i, \
            dW_ox, dW_oh, db_o, \
            state, hidden_state = self.iteration(input_vector, target_vector, state, hidden_state)

            # Update loss
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Parameter update (Adagrad)
            for param, dparam, mem, rmem in zip(
                    [self.W_sy, self.b_y, self.W_hs, self.W_xs, self.b_s, self.W_fx, self.W_fh, self.b_f, self.W_ix,
                     self.W_ih, self.b_i, self.W_ox, self.W_oh, self.b_o],
                    [dW_sy, db_y, dW_hs, dW_xs, db_s, dW_fx, dW_fh, db_f, dW_ix, dW_ih, db_i, dW_ox, dW_oh, db_o],
                    [mW_sy, mb_y, mW_hs, mW_xs, mb_s, mW_fx, mW_fh, mb_f, mW_ix, mW_ih, mb_i, mW_ox, mW_oh, mb_o],
                    [rW_sy, rb_y, rW_hs, rW_xs, rb_s, rW_fx, rW_fh, rb_f, rW_ix, rW_ih, rb_i, rW_ox, rW_oh, rb_o]):
                mem = exp_running_avg(mem, dparam, beta1)
                rmem = exp_running_avg(rmem, dparam ** 2, beta2)

                m_k_hat = mem / (1. - (beta1 ** (current_step)))
                r_k_hat = rmem / (1. - (beta2 ** (current_step)))

                param -= self.learning_rate * m_k_hat / (np.sqrt(r_k_hat) + 1e-8)

            # Managing counters
            current_step += 1
            current_index += self.step_size

            if current_step % 1000 == 0:
                sample = self.sample(state, hidden_state, self.index_to_char[input_vector[0]], 200)
                print("------------")
                print("step #", current_step, "loss: ", smooth_loss)
                print(sample)

    def sample(self, state, hidden_state, seed_str, length):
        seed = [self.char_to_index[char] for char in seed_str]
        output = seed_str

        x = None

        # Advance state as if has read 'seed_str'
        for t in range(0, len(seed)):
            x = np.zeros((self.char_vocab, 1))
            x[seed[t]] = 1
            i_g = sigmoid(self.W_ix @ x + self.W_ih @ hidden_state + self.b_i)
            o_g = sigmoid(self.W_ox @ x + self.W_oh @ hidden_state + self.b_o)
            f_g = sigmoid(self.W_fx @ x + self.W_fh @ hidden_state + self.b_f)

            state_inc = tanh(self.W_xs @ x + self.W_hs @ hidden_state + self.b_s)
            state = (f_g * state) + (i_g * state_inc)

            hidden_state = o_g * tanh(state)

        for t in range(0, length):
            i_g = sigmoid(self.W_ix @ x + self.W_ih @ hidden_state + self.b_i)
            o_g = sigmoid(self.W_ox @ x + self.W_oh @ hidden_state + self.b_o)
            f_g = sigmoid(self.W_fx @ x + self.W_fh @ hidden_state + self.b_f)

            state_inc = tanh(self.W_xs @ x + self.W_hs @ hidden_state + self.b_s)
            state = (f_g * state) + (i_g * state_inc)

            hidden_state = o_g * tanh(state)

            y = np.dot(self.W_sy, hidden_state) + self.b_y
            o = np.exp(y) / np.sum(np.exp(y))

            sampled_letter = np.random.choice(range(self.char_vocab), p=o.ravel())
            output += self.index_to_char[sampled_letter]

            x = np.zeros((self.char_vocab, 1))
            x[sampled_letter] = 1

        return output


if __name__ == "__main__":
    # filename = "C:\\Users\\patyc\\Documents\\mablelrosk.txt"
    # filename = "C:\\Users\\patyc\\Documents\\GitHub\\rnn-karpathy\\shakespeare.txt"
    filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\abba baba.txt"
    # filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\raposa.txt"

    rnn = LSTM(filename, state_size=64, step_size=10, learning_rate=1e-3)

    rnn.loop()
