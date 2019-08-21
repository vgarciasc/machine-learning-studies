import numpy as np
import pickle

from my_lstm import utils as u


class LSTM:
    def __init__(self, V, S, char_to_index, index_to_char):
        self.V = V  # char_vocab
        self.S = S  # state_size
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.vocab_size = len(char_to_index)
        # stores weights and biases
        self.model = dict(
            Wsx=np.random.randn(S, V) * 0.01,
            Wsh=np.random.randn(S, S) * 0.01,
            bs=np.zeros((S, 1)),
            Wix=np.random.randn(S, V) * 0.01,
            Wih=np.random.randn(S, S) * 0.01,
            bi=np.zeros((S, 1)),
            Wfx=np.random.randn(S, V) * 0.01,
            Wfh=np.random.randn(S, S) * 0.01,
            bf=np.zeros((S, 1)),
            Wox=np.random.randn(S, V) * 0.01,
            Woh=np.random.randn(S, S) * 0.01,
            bo=np.zeros((S, 1)),
            Why=np.random.randn(V, S) * 0.01,
            by=np.zeros((V, 1)),
        )

    def get_weights_and_biases(self):
        m = self.model
        return \
            m['Wsx'], m['Wsh'], m['bs'], \
            m['Wix'], m['Wih'], m['bi'], \
            m['Wfx'], m['Wfh'], m['bf'], \
            m['Wox'], m['Woh'], m['bo'], \
            m['Why'], m['by']

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.model, state = pickle.load(f)
        return state

    def initial_state(self):
        return np.zeros((self.S, 1)), np.zeros((self.S, 1))

    def forward_pass(self, x_index, state):
        Wsx, Wsh, bs, \
        Wix, Wih, bi, \
        Wfx, Wfh, bf, \
        Wox, Woh, bo, \
        Why, by = self.get_weights_and_biases()

        h_old, s_old = state

        # ~ input vector
        x = np.zeros((self.V, 1))
        x[x_index] = 1.0

        # ~ gates
        i = u.sigmoid(Wix @ x + Wih @ h_old + bi)
        o = u.sigmoid(Wox @ x + Woh @ h_old + bo)
        f = u.sigmoid(Wfx @ x + Wfh @ h_old + bf)

        # ~ state
        s_bar = u.tanh(Wsx @ x + Wsh @ h_old + bs)
        s = f * s_old + i * s_bar

        # ~ hidden state
        h = o * u.tanh(s)

        # ~ output
        y = Why @ h + by

        # ~ output as probabilities
        prob = u.softmax(y)

        # saving variables for backpropagation
        cache = (x, h, h_old, s, s_old, s_bar, i, f, o, y, prob)

        state = (h, s)

        return y, state, cache

    def backward_pass(self, y_true, d_next, cache):
        Wsx, Wsh, bs, \
        Wix, Wih, bi, \
        Wfx, Wfh, bf, \
        Wox, Woh, bo, \
        Why, by = self.get_weights_and_biases()

        # unpacking state variables from [t + 1]
        dh_next, ds_next = d_next

        # recovering variables from forward pass
        x, h, h_old, s, s_old, s_bar, i, f, o, y, prob = cache

        # ~ output as probabilities
        dy = np.copy(prob)
        dy[y_true] -= 1

        # ~ output
        dWhy = dy @ h.T
        dby = dy

        # ~ hidden state
        delta = Why.T @ dy
        dh = dh_next + delta

        # ~ state
        ds = dh * o * (1 - u.tanh(s) ** 2) + ds_next
        ds_bar = ds * i * (1 - s_bar ** 2)

        # ~ gates
        di = ds * s_bar * (i * (1 - i))
        df = ds * s_old * (f * (1 - f))
        do = dh * u.tanh(s) * (o * (1 - o))

        # calculating gradients
        dh_acc = 0
        grad = dict(Why=dWhy, by=dby)

        for d, W, dWx, dWh, db in zip([di, df, do, ds_bar],
                                      [Wih, Wfh, Woh, Wsh],
                                      ['Wix', 'Wfx', 'Wox', 'Wsx'],
                                      ['Wih', 'Wfh', 'Woh', 'Wsh'],
                                      ['bi', 'bf', 'bo', 'bs']):
            grad[dWx] = d @ x.T
            grad[dWh] = d @ h_old.T
            grad[db] = d
            dh_acc += W.T @ d

        # saving current derivatives for [t - 1]
        dh_next = dh_acc
        ds_next = ds * f
        state = (dh_next, ds_next)

        return grad, state

    def iteration(self, inputs, targets, state):
        caches = []
        loss = 0.0

        # ~ forward pass
        for x, y_true in zip(inputs, targets):
            y, state, cache = self.forward_pass(x, state)
            loss += u.cross_entropy(u.softmax(y), y_true)

            caches.append(cache)

        # updating loss
        loss *= 100
        loss /= inputs.shape[0]

        # ~ backward pass
        d_next = self.initial_state()
        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        for y_true, cache in reversed(list(zip(targets, caches))):
            grad, d_next = self.backward_pass(y_true, d_next, cache)

            # accumulating gradients
            for k in grads.keys():
                grads[k] += grad[k]

        # gradient clipping
        for k, v in grads.items():
            grads[k] = np.clip(v, -5., 5.)

        return grads, loss, state

    def sample(self, seed, state, size=200):
        # seed = [self.char_to_index[char] for char in 'abba']

        # advancing state as if 'seed' was read
        for i in range(0, len(seed) - 2):
            x = seed[i]
            y, state, _ = self.forward_pass(x, state)

        x = seed[len(seed) - 1]
        chars = []

        # sampling
        for _ in range(size - 1):
            y, state, _ = self.forward_pass(x, state)

            prob = u.softmax(y)
            index = np.random.choice(range(self.vocab_size), p=prob.ravel())

            chars.append(self.index_to_char[index])
            x = index

        output = ''.join([self.index_to_char[index] for index in seed])
        output += " => "
        output += ''.join(chars)

        return output
