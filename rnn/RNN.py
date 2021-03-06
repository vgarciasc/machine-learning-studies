import numpy as np


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


def get_minibatch(X, y, minibatch_size):
    minibatches = []

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches


class RNN():
    def __init__(self, filename, state_size, step_size, learning_rate):
        self.input_text = open(filename, "r", encoding="utf8").read()
        self.used_chars = list(set(self.input_text))
        self.char_vocab = len(self.used_chars)
        self.char_to_index = {ch: i for i, ch in enumerate(self.used_chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.used_chars)}

        self.state_size = state_size
        self.step_size = step_size
        self.learning_rate = learning_rate

        self.model = dict(
            W_xs=np.random.randn(self.state_size, self.char_vocab) * 0.01,  # from input to state
            W_ss=np.random.randn(self.state_size, self.state_size) * 0.01,  # from state to state
            W_sy=np.random.randn(self.char_vocab, self.state_size) * 0.01,  # from state to output
            b_s=np.zeros((self.state_size, 1)),
            b_y=np.zeros((self.char_vocab, 1))
        )

    def iteration(self, inputs, targets, initial_state):
        m = self.model
        W_xs, W_ss, W_sy, b_s, b_y = m['W_xs'], m['W_ss'], m['W_sy'], m['b_s'], m['b_y']

        x = {}
        y = {}
        o = {}  # output probabilities

        loss = 0

        state = {}
        state[-1] = initial_state  # setting initial state

        # Forward pass
        for t, char_key in enumerate(inputs):
            # ~ input vector
            # eg:
            #   char_key = 1; char_vocab = 3; char_to_index = {a: 0, b: 1, c: 2}
            #   x = [0, 1, 0] (one-hot)
            x[t] = np.zeros((self.char_vocab, 1))
            x[t][char_key] = 1

            # ~ state
            state[t] = np.tanh(np.dot(W_xs, x[t]) + np.dot(W_ss, state[t - 1]) + b_s)

            # ~ output vector
            y[t] = np.dot(W_sy, state[t]) + b_y

            # ~ softmaxing and creating probabilities
            o[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))

            # ~ computing loss
            loss += - np.log(o[t][int(targets[t]), 0])

        dW_xs = np.zeros_like(W_xs)
        dW_ss = np.zeros_like(W_ss)
        dW_sy = np.zeros_like(W_sy)

        db_s = np.zeros_like(b_s)
        db_y = np.zeros_like(b_y)
        ds_next = np.zeros_like(state[0])

        # Backward pass
        for t in range(len(inputs) - 1, 0, -1):
            # ~ backpropagate into y
            dy = np.copy(o[t])
            dy[int(targets[t])] -= 1  # derivative of loss wrt output through softmax
            dW_sy += np.dot(dy, state[t].T)
            db_y += dy

            # ~ backpropagate into state
            ds = np.dot(W_sy.T, dy) + ds_next
            ds_raw = (1 - state[t] * state[t]) * ds  # d(tanh)
            db_s += ds_raw
            dW_ss += np.dot(ds_raw, state[t - 1].T)
            ds_next = np.dot(W_ss.T, ds_raw)

            # ~ backpropagate into x
            dW_xs += np.dot(ds_raw, x[t].T)

        # Clipping gradients
        for param in [dW_xs, dW_ss, dW_sy, db_s, db_y]:
            np.clip(param, -5, 5, out=param)

        grad = dict(W_xs=dW_xs, W_ss=dW_ss, W_sy=dW_sy, b_s=db_s, b_y=db_y)

        return loss, grad, state[len(inputs) - 1]

    def adam_rnn(self, alpha=0.001, mb_size=256, n_iter=2000, print_after=100):
        M = {k: np.zeros_like(v) for k, v in self.model.items()}
        R = {k: np.zeros_like(v) for k, v in self.model.items()}
        beta1 = .9
        beta2 = .999

        X_train = np.array([self.char_to_index[char] for char in self.input_text])
        y_train = [self.char_to_index[char] for char in self.input_text[1:]]
        y_train.append(self.char_to_index[" "])
        y_train = np.array(y_train)

        minibatches = get_minibatch(X_train, y_train, mb_size)

        idx = 0
        state = np.zeros((self.state_size, 1))
        smooth_loss = - np.log(1.0 / self.char_vocab) * self.step_size

        for iter in range(1, n_iter + 1):
            t = iter

            if idx >= len(minibatches):
                idx = 0
                state = np.zeros((self.state_size, 1))

            X_mini, y_mini = minibatches[idx]
            idx += 1

            if iter % print_after == 0:
                print("=========================================================================")
                print('Iter-{} loss: {:.4f}'.format(iter, smooth_loss))
                print("=========================================================================")

                sample = self.sample(state, X_mini[0], 100)
                print(sample)

                print("=========================================================================")
                print()
                print()

            loss, grad, state = self.iteration(X_mini, y_mini, state)
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            for k in grad:
                M[k] = exp_running_avg(M[k], grad[k], beta1)
                R[k] = exp_running_avg(R[k], grad[k] ** 2, beta2)

                m_k_hat = M[k] / (1. - beta1 ** (t))
                r_k_hat = R[k] / (1. - beta2 ** (t))

                self.model[k] -= alpha * m_k_hat / (np.sqrt(r_k_hat) + 1e-8)

    def loop(self):
        # m
        mW_xs = np.zeros_like(self.W_xs)
        mW_ss = np.zeros_like(self.W_ss)
        mW_sy = np.zeros_like(self.W_sy)

        mb_s = np.zeros_like(self.b_s)
        mb_y = np.zeros_like(self.b_y)

        # r
        rW_xs = np.zeros_like(self.W_xs)
        rW_ss = np.zeros_like(self.W_ss)
        rW_sy = np.zeros_like(self.W_sy)

        rb_s = np.zeros_like(self.b_s)
        rb_y = np.zeros_like(self.b_y)

        # init
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
                # print("--~--")
                # print("dW_xs: ", dW_xs.mean(), "; dW_ss: ", dW_ss.mean(), "; dW_sy: ", dW_sy.mean())
                # print("db_y: ", db_y.mean(), "; db_s: ", db_s.mean())
                # print("--~--")
                print(sample)

    def sample(self, state, seed_str, length):
        seed = [0]
        output = ''
        m = self.model

        x = None

        # Advance state as if has read 'seed_str'
        for t in range(0, len(seed)):
            x = np.zeros((self.char_vocab, 1))
            x[seed[t]] = 1
            state = np.tanh(np.dot(m['W_xs'], x) + np.dot(m['W_ss'], state) + m['b_s'])

        for t in range(0, length):
            state = np.tanh(np.dot(m['W_xs'], x) + np.dot(m['W_ss'], state) + m['b_s'])
            y = np.dot(m['W_sy'], state) + m['b_y']
            o = np.exp(y) / np.sum(np.exp(y))

            sampled_letter = np.random.choice(range(self.char_vocab), p=o.ravel())
            output += self.index_to_char[sampled_letter]

            x = np.zeros((self.char_vocab, 1))
            x[sampled_letter] = 1

        return output


if __name__ == "__main__":
    # filename = "C:\\Users\\patyc\\Documents\\mablelrosk.txt"
    # filename = "C:\\Users\\patyc\\Documents\\GitHub\\rnn-karpathy\\shakespeare.txt"
    # filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\01011001.txt"
    filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\abba baba.txt"
    # filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\raposa.txt"

    rnn = RNN(filename, state_size=64, step_size=1, learning_rate=1e-3)

    rnn.adam_rnn()
