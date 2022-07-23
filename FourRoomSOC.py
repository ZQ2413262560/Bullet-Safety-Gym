import gym
import argparse
import numpy as np
import math
from scipy.special import expit, logsumexp
import os
import datetime
import threading
import four_room

from gym import envs
print(envs.registry.all())

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state, ])

    def __len__(self):
        return self.nstates


class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.nactions = nactions
        self.weights = 0.5 * np.ones((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi, curr_epsilon):
        if self.rng.uniform() < curr_epsilon:
            return int(self.rng.randint(self.nactions))
        return int(np.argmax(self.value(phi)))


class BoltzmannPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = 0.5 * np.ones((nfeatures, nactions))
        self.nactions = nactions
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi) / self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))


class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = 0.5 * np.ones((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate * (1. - terminate), phi


class IntraOptionQLearning:
    def __init__(self, gamma, lr, terminations, weights):
        self.lr = lr
        self.gamma = gamma
        self.terminations = terminations
        self.weights = weights

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]

    def update(self, phi, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.gamma * (
                    (1. - termination) * current_values[self.last_option] + termination * np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr * tderror
        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi

        return update_target


class IntraOptionActionQLearning:
    def __init__(self, gamma, lr, terminations, weights, qbigomega):
        self.lr = lr
        self.gamma = gamma
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega

    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)

    def start(self, phi, option, action):
        self.last_phi = phi
        self.last_option = option
        self.last_action = action

    def update(self, phi, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.gamma * (
                    (1. - termination) * current_values[self.last_option] + termination * np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_option, self.last_action] += self.lr * tderror

        self.last_phi = phi
        self.last_option = option
        self.last_action = action
        return tderror

    def get_initial_td_error(self, initial_phi, initial_option, action, reward, next_phi, psi):
        termination = self.terminations[initial_option].pmf(next_phi)
        val_next_state = self.qbigomega.value(next_phi)
        initial_td_error = reward + self.gamma * ((1. - termination) * val_next_state[initial_option] \
                                                  + termination * np.max(val_next_state)) - self.value(initial_phi,
                                                                                                       initial_option,
                                                                                                       action)

        return psi * math.pow(initial_td_error, 2.0)


class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, phi, option):
        magnitude, direction = self.terminations[option].grad(phi)
        self.terminations[option].weights[direction] -= \
            self.lr * magnitude * (self.critic.advantage(phi, option))


class IntraOptionGradient:
    def __init__(self, option_policies, lr, psi, nactions):
        self.lr = lr
        self.option_policies = option_policies
        self.psi = psi
        self.initial_actions_pmf = np.zeros(nactions)
        self.visit_phi = 1
        self.val_subtract = 0.0
        self.val_add = 0.0

    def update(self, phi, option, action, critic, initial_td_error, initial_option, initial_phi, initial_action):
        actions_pmf = self.option_policies[option].pmf(phi)
        if self.psi != 0.0:
            if self.visit_phi > 0:
                self.initial_actions_pmf = self.option_policies[initial_option].pmf(initial_phi)
                self.val_subtract = self.lr * initial_td_error
                self.val_add = self.val_subtract * self.initial_actions_pmf
                self.visit_phi = 0

            self.option_policies[initial_option].weights[initial_phi, initial_action] -= self.val_subtract
            self.option_policies[initial_option].weights[initial_phi, :] += self.val_add

        curr_val = self.lr * critic
        self.option_policies[option].weights[phi, :] -= curr_val * actions_pmf
        self.option_policies[option].weights[phi, action] += curr_val

        if (self.psi != 0.0) and (initial_option == option):
            if np.intersect1d(phi, initial_phi).size > 0:
                self.visit_phi = 1


class OutputInformation:
    def __init__(self):
        # storaging the weights of the trained model
        self.weight_intra_option = []
        self.weight_policy = []
        self.weight_termination = []
        self.history = []


def get_frozen_states():
    layout = """\
wwwwwwwwwwwww
w     w     w
w   ffwff   w
w  fffffff  w
w   ffwff   w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

    num_elem = 13
    frozen_states = []
    state_num = 0
    for line in layout.splitlines():
        for i in range(num_elem):
            if line[i] == "f":
                frozen_states.append(state_num)
            if line[i] != "w":
                state_num += 1
    return frozen_states


def save_params(args, dir_name):
    f = os.path.join(dir_name, "Params.txt")
    with open(f, "w") as f_w:
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))


def run_agent(outputinfo, features, nepisodes, frozen_states, nfeatures, nactions, num_states,
              temperature, gamma, lr_critic, lr_intra, lr_term, psi, noptions, rng):
    history = np.zeros((nepisodes, 4),
                       dtype=np.float32)  # 1. Return 2. Steps 3. TD error 1 norm 4. Q(s,w,a) value weight sum

    # store the weights of the trained model
    weight_intra_option = np.zeros((nepisodes, num_states, nactions, noptions), dtype=np.float32)
    weight_policy = np.zeros((nepisodes, num_states, noptions), dtype=np.float32)
    weight_termination = np.zeros((nepisodes, num_states, noptions), dtype=np.float32)

    # Intra-Option Policy
    option_policies = [BoltzmannPolicy(rng, nfeatures, nactions, temperature) for _ in range(noptions)]
    # The termination function are linear-sigmoid functions
    option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(noptions)]
    # Policy over options
    policy = BoltzmannPolicy(rng, nfeatures, noptions, temperature)

    # Different choices are possible for the critic. Here we learn an
    # option-value function and use the estimator for the values upon arrival
    critic = IntraOptionQLearning(gamma, lr_critic, option_terminations, policy.weights)

    # Learn Qomega separately
    action_weights = np.random.random((nfeatures, noptions, nactions))
    action_critic = IntraOptionActionQLearning(gamma, lr_critic, option_terminations, action_weights, critic)

    # Improvement of the termination functions based on gradients
    termination_improvement = TerminationGradient(option_terminations, critic, lr_term)

    # Intra-option gradient improvement with critic estimator
    intraoption_improvement = IntraOptionGradient(option_policies, lr_intra, psi, nactions)
    env = gym.make('Fourrooms-v0')
    for episode in range(nepisodes):
        return_per_episode = 0
        observation = env.reset()
        phi = features(observation)
        option = policy.sample(phi)
        action = option_policies[option].sample(phi)
        critic.start(phi, option)
        action_critic.start(phi, option, action)
        initial_phi = phi
        initial_action = action
        initial_option = option
        intraoption_improvement.visit_phi = 1

        intial_td_error = 0.0
        done = False
        step = 0
        sum_td_error = 0.0
        while done != True:
            old_phi = phi
            old_option = option
            old_action = action
            observation, reward, done, _ = env.step(action)

            # Frozen state receives a normal reward N(0, 15)
            if observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=15.0)

            phi = features(observation)
            # return calculation
            return_per_episode += pow(gamma, step) * reward

            # Termination might occur upon entering the new state
            if option_terminations[option].sample(phi):
                option = policy.sample(phi)

            action = option_policies[option].sample(phi)

            # Critic update
            update_target = critic.update(phi, option, reward, done)
            tderror = action_critic.update(phi, option, action, reward, done)
            sum_td_error += abs(tderror)

            if ((psi != 0.0) and (old_phi == initial_phi and old_option == initial_option)):
                second_phi = phi
                initial_reward = reward
                initial_action = old_action
                intial_td_error = action_critic.get_initial_td_error(initial_phi, initial_option, initial_action,
                                                                     initial_reward, second_phi, psi)

            # Intra-option policy update
            critic_feedback = action_critic.value(old_phi, old_option, old_action) - critic.value(old_phi, old_option)
            intraoption_improvement.update(old_phi, old_option, old_action, critic_feedback, intial_td_error,
                                           initial_option, initial_phi, initial_action)
            termination_improvement.update(phi, old_option)
            step += 1
            if done:
                break
        print(episode, "  end")
        history[episode, 0] = step
        history[episode, 1] = return_per_episode
        history[episode, 2] = sum_td_error
        history[episode, 3] = np.sum(action_critic.weights)

        for o in range(noptions):
            weight_intra_option[episode, :, :, o] = option_policies[o].weights
        weight_policy[episode, :, :] = policy.weights
        for o in range(noptions):
            weight_termination[episode, :, o] = option_terminations[o].weights

    outputinfo.weight_intra_option.append(weight_intra_option)
    outputinfo.weight_policy.append(weight_policy)
    outputinfo.weight_termination.append(weight_termination)
    outputinfo.history.append(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='gamma factor Gamma', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=1e-1)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-2)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=5e-1)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=10)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=500)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--psi', help="psi for controllability", type=float, default=0.0)
    parser.add_argument('--temperature', help="Temperature parameter for Boltzmann", type=float, default=1e-3)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)

    args = parser.parse_args()
    now_time = datetime.datetime.now()
    env = gym.make('Fourrooms-v0')
    outer_dir = "./results"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "FR_" + now_time.strftime("%d-%m"))
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.psi) + \
               "_LRC" + str(args.lr_critic) + "_LRIntra" + str(args.lr_intra) + "_LRT" + str(args.lr_term) + \
               "_nOpt" + str(args.noptions) + "_temp" + str(args.temperature) + "_seed" + str(args.seed)
    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)
    num_states = env.observation_space.n
    nactions = env.action_space.n
    frozen_states = get_frozen_states()
    threads = []
    features = Tabular(num_states)
    nfeatures = len(features)
    outputinfo = OutputInformation()

    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, features, args.nepisodes, frozen_states, nfeatures,
                                                     nactions, num_states, args.temperature,
                                                     args.gamma, args.lr_critic, args.lr_intra, args.lr_term, args.psi,
                                                     args.noptions, np.random.RandomState(args.seed + i),))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

    np.save(os.path.join(dir_name, 'WeightIntraOption.npy'), np.asarray(outputinfo.weight_intra_option))
    np.save(os.path.join(dir_name, 'WeightPolicy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'WeightTermination.npy'), np.asarray(outputinfo.weight_termination))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
