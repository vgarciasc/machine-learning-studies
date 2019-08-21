import numpy as np
from my_lstm import utils as u, layer as l, loss as loss_fun, solver


class LSTM():
    def __init__(self, D, H, char_to_index, index_to_char):
        self.D = D
        self.H = H
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.vocab_size = len(char_to_index)
        self.model = dict(
            Wsx=np.random.randn(H, D) * 0.01,
            Wsh=np.random.randn(H, H) * 0.01,
            bs=np.zeros((H, 1)),
            Wix=np.random.randn(H, D) * 0.01,
            Wih=np.random.randn(H, H) * 0.01,
            bi=np.zeros((H, 1)),
            Wfx=np.random.randn(H, D) * 0.01,
            Wfh=np.random.randn(H, H) * 0.01,
            bf=np.zeros((H, 1)),
            Wox=np.random.randn(H, D) * 0.01,
            Woh=np.random.randn(H, H) * 0.01,
            bo=np.zeros((H, 1)),
            Why=np.random.randn(D, H) * 0.01,
            by=np.zeros((D, 1)),
        )

    def initial_state(self):
        return np.zeros((self.H, 1)), np.zeros((self.H, 1))

    def forward(self, x_index, state, train=True):
        m = self.model

        Wsx, Wsh, bs = m['Wsx'], m['Wsh'], m['bs']
        Wix, Wih, bi = m['Wix'], m['Wih'], m['bi']
        Wfx, Wfh, bf = m['Wfx'], m['Wfh'], m['bf']
        Wox, Woh, bo = m['Wox'], m['Woh'], m['bo']
        Why, by = m['Why'], m['by']

        h_old, s_old = state

        x = np.zeros((self.D, 1))
        x[x_index] = 1.0

        i = u.sigmoid(Wix @ x + Wih @ h_old + bi)
        o = u.sigmoid(Wox @ x + Woh @ h_old + bo)
        f = u.sigmoid(Wfx @ x + Wfh @ h_old + bf)

        s_bar = u.tanh(Wsx @ x + Wsh @ h_old + bs)
        s = f * s_old + i * s_bar

        h = o * u.tanh(s)

        y = Why @ h + by

        prob = np.exp(y) / np.sum(np.exp(y))

        cache = (x, h, h_old, s, s_old, s_bar, i, f, o, y, prob)
        state = (h, s)

        return y, state, cache

    def backward(self, y_true, d_next, cache):
        m = self.model

        Wsx, Wsh, bs = m['Wsx'], m['Wsh'], m['bs']
        Wix, Wih, bi = m['Wix'], m['Wih'], m['bi']
        Wfx, Wfh, bf = m['Wfx'], m['Wfh'], m['bf']
        Wox, Woh, bo = m['Wox'], m['Woh'], m['bo']
        Why, by = m['Why'], m['by']

        x, h, h_old, s, s_old, s_bar, i, f, o, y, prob = cache
        dh_next, ds_next = d_next

        dy = np.copy(prob)
        dy[y_true] -= 1

        dWhy = dy @ h.T
        dby = dy

        delta = Why.T @ dy
        dh = dh_next + delta

        do = dh * u.tanh(s) * (o * (1 - o))
        ds = dh * o * (1 - u.tanh(s)**2) + ds_next
        df = ds * s_old * (f * (1 - f))
        di = ds * s_bar * (i * (1 - i))
        ds_bar = ds * i * (1 - s_bar**2)

        dWfx = df @ x.T
        dWfh = df @ h_old.T
        dbf = df
        dhf = Wfh.T @ df

        dWix = di @ x.T
        dWih = di @ h_old.T
        dbi = di
        dhi = Wih.T @ di

        dWox = do @ x.T
        dWoh = do @ h_old.T
        dbo = do
        dho = Woh.T @ do

        dWsx = ds_bar @ x.T
        dWsh = ds_bar @ h_old.T
        dbs = ds_bar
        dhs = Wsh.T @ ds_bar

        dh_next = dhf + dhi + dho + dhs
        ds_next = ds * f

        grad = dict(
            Wsx=dWsx, Wsh=dWsh, bs=dbs,
            Wix=dWix, Wih=dWih, bi=dbi,
            Wfx=dWfx, Wfh=dWfh, bf=dbf,
            Wox=dWox, Woh=dWoh, bo=dbo,
            Why=dWhy, by=dby
        )

        return grad, (dh_next, ds_next)

    def train_step(self, X_train, y_train, state):
        y_preds = []
        caches = []
        loss = 0.

        # Forward
        for x, y_true in zip(X_train, y_train):
            y, state, cache = self.forward(x, state, train=True)
            loss += loss_fun.cross_entropy(self.model, y, y_true, lam=0)

            caches.append(cache)

        loss /= X_train.shape[0]

        # Backward
        d_next = self.initial_state()
        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        for y_true, cache in reversed(list(zip(y_train, caches))):
            grad, d_next = self.backward(y_true, d_next, cache)

            for k in grads.keys():
                grads[k] += grad[k]

        for k, v in grads.items():
            grads[k] = np.clip(v, -5., 5.)

        return grads, loss, state

    def sample(self, seed, h, size=100):
        chars = [self.index_to_char[seed]]
        idx_list = list(range(self.vocab_size))
        x = seed

        for _ in range(size - 1):
            y, h, _ = self.forward(x, h, train=False)
            prob = np.exp(y) / np.sum(np.exp(y))
            idx = np.random.choice(idx_list, p=prob.ravel())
            chars.append(self.index_to_char[idx])
            x = idx

        return ''.join(chars)


if __name__ == '__main__':
    time_step = 10
    n_iter = 1000000000
    alpha = 1e-3
    print_after = 1000

    H = 64

    filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\abba baba.txt"

    with open(filename, 'r') as f:
        txt = f.read()

        X = []
        y = []

        char_to_idx = {char: i for i, char in enumerate(set(txt))}
        idx_to_char = {i: char for i, char in enumerate(set(txt))}

        X = np.array([char_to_idx[x] for x in txt])
        y = [char_to_idx[x] for x in txt[1:]]
        y.append(char_to_idx[' '])
        y = np.array(y)

    vocab_size = len(char_to_idx)

    net = LSTM(vocab_size, H=H, char_to_index=char_to_idx, index_to_char=idx_to_char)

    solver.adam(
        net, X, y,
        alpha=alpha,
        mb_size=time_step,
        n_iter=n_iter,
        print_after=print_after
    )
