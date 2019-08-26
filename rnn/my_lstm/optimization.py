import numpy as np
import utils as util


def get_minibatches(x, y, minibatch_size):
    minibatches = []

    for i in range(0, x.shape[0], minibatch_size):
        X_mini = x[i:(i + minibatch_size)]
        y_mini = y[i:(i + minibatch_size)]

        minibatches.append((X_mini, y_mini))

    return minibatches


def adam(lstm, inputs, targets,
         learning_rate=0.001, mb_size=256,
         iterations=2000, print_every=100):
    M = {k: np.zeros_like(v) for k, v in lstm.model.items()}  # memory
    R = {k: np.zeros_like(v) for k, v in lstm.model.items()}  # running
    beta1 = 0.9
    beta2 = 0.999

    minibatches = get_minibatches(inputs, targets, mb_size)
    current_mini = 0

    smooth_loss = - np.log(1.0 / len(set(inputs)))
    state = lstm.initial_state()

    for current_step in range(1, iterations + 1):
        # if all minibatches used, return to minibatch #1
        if current_mini >= len(minibatches):
            current_mini = 0
            state = lstm.initial_state()

        x_mini, y_mini = minibatches[current_mini]
        current_mini += 1

        # iterate through 'k' timesteps (because of _truncated_ BPTT)
        grad, loss, state = lstm.iteration(x_mini, y_mini, state)

        # update loss
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        # optimize weights and biases
        for k in grad:
            M[k] = util.exp_running_avg(M[k], grad[k], beta1)
            R[k] = util.exp_running_avg(R[k], grad[k] ** 2, beta2)

            m_k_hat = M[k] / (1. - beta1 ** current_step)
            r_k_hat = R[k] / (1. - beta2 ** current_step)

            lstm.model[k] -= learning_rate * m_k_hat / (np.sqrt(r_k_hat) + 1e-8)

        # sample if it's sampling time
        if current_step % print_every == 0:
            seed = [x_mini[0]]
            sample = lstm.sample(seed, state, 600)
            print("==========================================")
            print("| step #{}".format(current_step), "; loss: {:.8f} |".format(smooth_loss))
            print("------------------------------------------")
            print("| ...", sample, "... |")

        # if (current_step < 1000 and current_step % 100 == 0) or \
        #    (current_step < 50000 and current_step % 1000 == 0) or \
        #    (current_step % 10000 == 0):
        #     model_filename = "C:\\Users\\patyc\\Documents\\GitHub\\beamer-presentations\\PESC\\TEIA\\Recurrent Neural Networks (Implementation)\\outputs\\shakespeare_pt\\models\\model_" + str(current_step) + ".pickle"
        #     lstm.save_model(model_filename)
        #
        #     seed = [x_mini[0]]
        #     sample = lstm.sample(seed, state, 2000)
        #
        #     sample_filename = "C:\\Users\\patyc\\Documents\\GitHub\\beamer-presentations\\PESC\\TEIA\\Recurrent Neural Networks (Implementation)\\outputs\\shakespeare_pt\\models\\sample_" + str(current_step) + ".txt"
        #     file = open(sample_filename, "w")
        #     file.write(sample)
        #     file.close()

    return lstm
