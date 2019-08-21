import numpy as np
from lstm import utils as util, layer as l, loss as loss_fun, solver


class LSTMv2():
    def __init__(self, D, H, char2idx, idx2char):
        self.D = D
        self.H = H
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(char2idx)
        self._init_model(D, D, H)

    def initial_state(self):
        return (np.zeros((1, self.H)), np.zeros((1, self.H)))

    def forward(self, X, state, train=True):
        m = self.model
        Wf, Wi, Wc, Wo, Wy = m['Wf'], m['Wi'], m['Wc'], m['Wo'], m['Wy']
        bf, bi, bc, bo, by = m['bf'], m['bi'], m['bc'], m['bo'], m['by']

        h_old, c_old = state

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        X = np.column_stack((h_old, X_one_hot))

        hf, hf_cache = l.fc_forward(X, Wf, bf)
        hf, hf_sigm_cache = l.sigmoid_forward(hf)

        hi, hi_cache = l.fc_forward(X, Wi, bi)
        hi, hi_sigm_cache = l.sigmoid_forward(hi)

        ho, ho_cache = l.fc_forward(X, Wo, bo)
        ho, ho_sigm_cache = l.sigmoid_forward(ho)

        hc, hc_cache = l.fc_forward(X, Wc, bc)
        hc, hc_tanh_cache = l.tanh_forward(hc)

        c = hf * c_old + hi * hc
        c, c_tanh_cache = l.tanh_forward(c)

        h = ho * c

        y, y_cache = l.fc_forward(h, Wy, by)

        cache = (
            X, hf, hi, ho, hc, hf_cache, hf_sigm_cache, hi_cache, hi_sigm_cache, ho_cache,
            ho_sigm_cache, hc_cache, hc_tanh_cache, c_old, c, c_tanh_cache, y_cache
        )

        if not train:
            y = util.softmax(y)

        return y, (h, c), cache

    def backward(self, y_pred, y_train, d_next, cache):
        X, hf, hi, ho, hc, hf_cache, hf_sigm_cache, hi_cache, hi_sigm_cache, ho_cache, ho_sigm_cache, hc_cache, hc_tanh_cache, c_old, c, c_tanh_cache, y_cache = cache
        dh_next, dc_next = d_next

        dy = loss_fun.dcross_entropy(y_pred, y_train)

        dh, dWy, dby = l.fc_backward(dy, y_cache)
        dh += dh_next

        dho = c * dh
        dho = l.sigmoid_backward(dho, ho_sigm_cache)

        dc = ho * dh
        dc = l.tanh_backward(dc, c_tanh_cache)
        dc = dc + dc_next

        dhf = c_old * dc
        dhf = l.sigmoid_backward(dhf, hf_sigm_cache)

        dhi = hc * dc
        dhi = l.sigmoid_backward(dhi, hi_sigm_cache)

        dhc = hi * dc
        dhc = l.tanh_backward(dhc, hc_tanh_cache)

        dXo, dWo, dbo = l.fc_backward(dho, ho_cache)
        dXc, dWc, dbc = l.fc_backward(dhc, hc_cache)
        dXi, dWi, dbi = l.fc_backward(dhi, hi_cache)
        dXf, dWf, dbf = l.fc_backward(dhf, hf_cache)

        dX = dXo + dXc + dXi + dXf
        dh_next = dX[:, :self.H]
        dc_next = hf * dc

        grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)

        return grad, (dh_next, dc_next)

    def train_step(self, X_train, y_train, state):
        y_preds = []
        caches = []
        loss = 0.

        # Forward
        for x, y_true in zip(X_train, y_train):
            y, state, cache = self.forward(x, state, train=True)
            loss += loss_fun.cross_entropy(self.model, y, y_true, lam=0)

            y_preds.append(y)
            caches.append(cache)

        loss /= X_train.shape[0]

        # Backward
        dh_next = np.zeros((1, self.H))
        dc_next = np.zeros((1, self.H))
        d_next = (dh_next, dc_next)

        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        for y_pred, y_true, cache in reversed(list(zip(y_preds, y_train, caches))):
            grad, d_next = self.backward(y_pred, y_true, d_next, cache)

            for k in grads.keys():
                grads[k] += grad[k]

        for k, v in grads.items():
            grads[k] = np.clip(v, -5., 5.)

        return grads, loss, state


    def _init_model(self, D, C, H):
        Z = H + D

        self.model = dict(
            Wf=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wi=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wc=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wo=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wy=np.random.randn(H, D) / np.sqrt(D / 2.),
            bf=np.zeros((1, H)),
            bi=np.zeros((1, H)),
            bc=np.zeros((1, H)),
            bo=np.zeros((1, H)),
            by=np.zeros((1, D))
        )

    def sample(self, X_seed, h, size=100):
        chars = [self.idx2char[X_seed]]
        idx_list = list(range(self.vocab_size))
        X = X_seed

        for _ in range(size - 1):
            prob, h, _ = self.forward(X, h, train=False)
            idx = np.random.choice(idx_list, p=prob.ravel())
            chars.append(self.idx2char[idx])
            X = idx

        return ''.join(chars)

if __name__ == '__main__':
    time_step = 10
    n_iter = 1000000000
    alpha = 1e-3
    print_after = 1000

    H = 64

    filename = "C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\rnn\\raposa.txt"

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

    net = LSTMv2(vocab_size, H=H, char2idx=char_to_idx, idx2char=idx_to_char)

    solver.adam(
        net, X, y,
        alpha=alpha,
        mb_size=time_step,
        n_iter=n_iter,
        print_after=print_after
    )
