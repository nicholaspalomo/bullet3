"""
This is my simple implementation of PPO. The code was adapted from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py.
"""
import numpy as np
import math
from datetime import datetime
import os.path
import time
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
import gym

from tensorboardX import SummaryWriter
from tensorboard import program
import webbrowser
from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.a2c.utils import total_episode_reward_logger

torch.set_num_threads(16)

target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# He initialization for network weights
# Borrowed from: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def weights_init(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        # torch.nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
        layer.bias.data.fill_(0.)


# Borrowed from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
def init(module, weight_init, bias_init, gain=0.01):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class TensorboardPlotter:
    def __init__(self, log_dir, extra_plot_names):
        self.bool_make_tensorboard_plots = len(log_dir) > 0

        # learning visualizer
        if self.bool_make_tensorboard_plots:
            self._tensorboard_sess = program.TensorBoard()

            port = str(random.randint(6000, 9999))

            self._tensorboard_sess.configure(argv=[None, '--logdir', log_dir, '--port', port])
            self._tensorboard_sess_url = self._tensorboard_sess.launch()
            print("[RAISIM_GYM] Tensorboard session created: " + self._tensorboard_sess_url)
            webbrowser.open_new(self._tensorboard_sess_url)

            self.__writer = SummaryWriter(logdir=log_dir)
        self.__scalar_dict = {
            "ppo/avg_advantage": [0, 0],
            "ppo/avg_return": [0, 0],
            "ppo/avg_loss": [0, 0],
            "ppo/avg_surr_loss": [0, 0],
            "ppo/state_value_loss": [0, 0],
            "ppo/explained_variance": [0, 0],
            "ppo/number_of_dones": [0, 0],
            "ppo/gradient_h_infinity_actor": [0, 0],
            "ppo/gradient_h_infinity_critic": [0, 0],
            "ppo/entropy": [0, 0],
            "ppo/mse_td_target": [0, 0],
            "ppo/avg_epoch_length": [0, 0],
            "ppo/kl_divergence": [0, 0],
            "ppo/avg_policy_entropy": [0, 0]
        }

        extra_plots_dict = {}
        if extra_plot_names is not None:
            for extra_plot in extra_plot_names:
                extra_plots_dict[extra_plot] = [0, 0]

        # for now, only works for scalars
        self.__extra_plots_dict = extra_plots_dict

    def update_plot(self, name, value):
        if self.bool_make_tensorboard_plots:
            if name in self.__extra_plots_dict:
                self.__extra_plots_dict[name] = [value, self.__extra_plots_dict[name][1] + 1]
                self.__writer.add_scalar(name, self.__extra_plots_dict[name][0], self.__extra_plots_dict[name][1])
            else:
                self.__scalar_dict[name] = [value, self.__scalar_dict[name][1] + 1]
                self.__writer.add_scalar(name, self.__scalar_dict[name][0], self.__scalar_dict[name][1])

    def close_tensorboard(self):
        if self.bool_make_tensorboard_plots:
            self.__writer.close()


class BatchDataset(Dataset):
    def __init__(self, actor_inputs, critic_inputs, actions, logprobs, advantages, returns, state_values):
        self.actor_inputs = actor_inputs
        self.critic_inputs = critic_inputs
        self.actions = actions
        self.logprobs = logprobs
        self.advantages = advantages
        self.returns = returns
        self.state_values = state_values

    def __getitem__(self, index):
        return (self.actor_inputs[index], self.critic_inputs[index], self.actions[index], self.logprobs[index],
                self.advantages[index], self.returns[index], self.state_values[index], index)

    def __len__(self):
        return len(self.advantages)


class Memory:
    def __init__(self, N):
        self.actions = N * [None]
        self.actor_input = N * [None]
        self.critic_input = N * [None]
        self.logprobs = N * [None]  # logarithm of the policy distribution
        self.rewards = N * [None]
        self.is_terminal = N * [None]
        self.last_actor_input = 0
        self.last_critic_input = 0
        self.mini_batch_size = 0
        self.batch = BatchDataset(None, None, None, None, None, None, None)

    def batch_loader(self, mini_batch_size, actor_inputs, critic_inputs, actions, logprobs, rewards, advantages,
                     state_values):
        self.mini_batch_size = mini_batch_size
        self.batch.actor_inputs = actor_inputs
        self.batch.critic_inputs = critic_inputs
        self.batch.actions = actions
        self.batch.logprobs = logprobs
        self.batch.advantages = advantages
        self.batch.returns = rewards
        self.batch.state_values = state_values

        train_loader = DataLoader(dataset=self.batch, batch_size=mini_batch_size, shuffle=True, pin_memory=True)

        return iter(train_loader)

    def reset_iter(self):
        return self.batch_loader(self.mini_batch_size, self.batch.actor_inputs, self.batch.critic_inputs,
                                 self.batch.actions, self.batch.logprobs, self.batch.returns, self.batch.advantages,
                                 self.batch.state_values)


# Borrowed from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        return (x + self._bias.t().view(1, -1))


# Borrowed from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1)

    def entrop(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Borrowed from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.mean(x)
        if action_mean.is_cuda:
            action_logstd = self.logstd(torch.zeros(action_mean.size(), device=target))
        else:
            action_logstd = self.logstd(torch.zeros(action_mean.size()))

        return FixedNormal(action_mean, action_logstd.exp())

    # Actor-Critic class


class A2C(
    nn.Module):  # subclass from the Module package in PyTorch - this contains the necessary building blocks for a neural network
    def __init__(self, actor_input_dim, critic_input_dim, action_dim, actor_net_layers, critic_net_layers):
        super(A2C, self).__init__()
        # action mean range from -1 to 1

        self.actor_net_layers = actor_net_layers.copy()
        self.critic_net_layers = critic_net_layers.copy()

        # build the actor network
        if len(actor_net_layers) > 2:
            self.actor_net_ = nn.Sequential(
                *(self.build_net(actor_input_dim, action_dim, self.actor_net_layers)[:len(self.actor_net_layers) + 1]))
            self.actor_net_layers.pop()
        else:
            self.actor_net_ = nn.Sequential(*[
                nn.Linear(actor_input_dim, self.actor_net_layers[0]),
                nn.Tanh(),
                nn.Linear(self.actor_net_layers[0], self.actor_net_layers[1]),
                nn.Tanh(),
                nn.Linear(self.actor_net_layers[1], action_dim)])

        self.actor_net_.apply(weights_init)

        # build the critic network
        self.critic_net = nn.Sequential(*self.build_net(critic_input_dim, 1, self.critic_net_layers))
        self.critic_net.apply(weights_init)

        self.action_dim = action_dim

        # create a Gaussian distribution from which to sample actions
        self.dist = DiagGaussian(self.actor_net_layers[-1], action_dim)

    def MultivariateGaussian(self, state):
        output = self.actor_net_(state)

        return self.dist(output)

    def actor_net(self, state):
        dist = self.MultivariateGaussian(state)

        return dist.mode()

    def build_net(self, input_dim, output_dim, layer_dims):
        layer_dims.insert(0, input_dim)
        layer_dims.append(output_dim)
        net = []
        for i in range(len(layer_dims) - 1):
            net.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != len(layer_dims) - 1:
                net.append(nn.Tanh())
        net.pop()

        return net

    def forward(self, state):
        return self.MultivariateGaussian(state).sample()

    def act(self, state, memory, idx):
        dist = self.MultivariateGaussian(state)

        action = dist.sample()
        action_logprob = dist.log_probs(action)

        # append the states, actions, log probabilities
        memory.actor_input[idx] = state
        memory.actions[idx] = action
        memory.logprobs[idx] = action_logprob

        return action.detach().cpu().data.numpy()  # tensor.cuda() is used to move a tensor to GPU memory. tensor.cpu() moves it back to memory accessible to the CPU. <- Not sure if this is gonna work with GPU, used to be in the return statement of select_action

    # evaluate the policy and the value function at the given state
    def evaluate(self, actor_input, critic_input, action):
        dist = self.MultivariateGaussian(actor_input)

        action_logprobs = dist.log_probs(torch.squeeze(action))
        dist_entropy = dist.entrop()

        state_value = self.critic_net(critic_input)

        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(self, learning_rate, betas, gamma, lamda, alpha, delta, entropy_loss_coef, state_value_loss_coef,
                 K_epochs, eps_clip, tensorboard_log, env, env_name, num_envs, max_timestep, mini_batch_size,
                 max_grad_norm, state_value_clip_factor, lr_decay, surrogate_cost_fcn, actor_net_layers,
                 critic_net_layers, normalize_states, actor_critic_weights_init_path):
        self.action_std = np.ones((1, env.num_acts), dtype=float)
        self.learning_rate = learning_rate
        self.betas = betas
        self.gamma = gamma
        self.lamda = lamda
        self.alpha = alpha
        self.delta = delta
        self.entropy_loss_coef = entropy_loss_coef  # regularizer coefficient for entropy term in loss function
        self.state_value_loss_coef = state_value_loss_coef
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.env = env
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_timestep = max_timestep
        self.mini_batch_size = mini_batch_size  # batch_size = self.num_envs * self.max_timestep
        self.max_grad_norm = max_grad_norm
        self.state_value_clip_factor = state_value_clip_factor
        self.surrogate_cost_fcn = surrogate_cost_fcn
        self.normalize_states = normalize_states
        self.actor_critic_weights_init_path = actor_critic_weights_init_path

        state_dim = env.num_obs
        action_dim = env.num_acts

        # instantiate the extra objects that augment the policy and critic inputs
        # self.aux = aux
        # self.aux.env = env

        self.tensorboard_writer = TensorboardPlotter(tensorboard_log, None)

        # self.aux.tensorboard = self.tensorboard_writer


        # self.actor_input_dim = state_dim + aux.actor_input_extra_dim
        # self.critic_input_dim = state_dim + aux.critic_input_extra_dim

        self.actor_input_dim = state_dim
        self.critic_input_dim = state_dim

        # instantiate instance of actor-critic class
        self.policy = A2C(self.actor_input_dim, self.critic_input_dim, action_dim, actor_net_layers, critic_net_layers)

        # load the weights of the actor-critic, if present in the provided directory
        if os.path.isfile(actor_critic_weights_init_path):
            self.policy.load_state_dict(torch.load(actor_critic_weights_init_path, map_location=target))

        # instantiate optimizer for actor-critic class
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, betas=self.betas, eps=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=lr_decay)

        # state mean and standard deviation initialization
        # self.actor_input_mean_sum = np.zeros((1, self.actor_input_dim), dtype=np.float)
        # self.actor_input_std_sum = np.zeros((1, self.actor_input_dim), dtype=np.float)
        # self.critic_input_mean_sum = np.zeros((1, self.critic_input_dim), dtype=np.float)
        # self.critic_input_std_sum = np.zeros((1, self.critic_input_dim), dtype=np.float)
        # self.counter = 0
        self.critic_obs_rms = RunningMeanStd(shape=[self.actor_input_dim])
        self.actor_obs_rms = RunningMeanStd(shape=[self.critic_input_dim])
        self.clip_obs = 10.0  # for safety

        # preallocate for speed improvement

        # instantiate a memory object (definition above). this class is a container for the data coming from/going in to the simulation
        self.memory = Memory(int(self.max_timestep))
        self.advantages_batch = torch.empty((self.num_envs, int(self.max_timestep)), dtype=torch.float, device=target)
        self.state_values_batch = torch.empty((self.num_envs, int(self.max_timestep)), dtype=torch.float, device=target)

    def normalize_state(self, actor_input, critic_input, update):
        if update:
            self.actor_obs_rms.update(actor_input)
            self.critic_obs_rms.update(critic_input)

        if self.normalize_states:
            actor_input = np.clip((actor_input - self.actor_obs_rms.mean) / np.sqrt(self.actor_obs_rms.var + 1e-8),
                                  -self.clip_obs,
                                  self.clip_obs)
            critic_input = np.clip((critic_input - self.critic_obs_rms.mean) / np.sqrt(self.critic_obs_rms.var + 1e-8),
                                   -self.clip_obs,
                                   self.clip_obs)

        # if update:
        #     self.actor_input_mean_sum += np.reshape(np.sum(actor_input, axis=0), (1,-1))
        #     self.critic_input_mean_sum += np.reshape(np.sum(critic_input, axis=0), (1,-1))
        #     self.counter += self.num_envs
        #
        # actor_input_mean = self.actor_input_mean_sum / self.counter
        # critic_input_mean = self.critic_input_mean_sum / self.counter
        #
        # if update:
        #     self.actor_input_std_sum += np.reshape(np.sum((actor_input - np.tile(actor_input_mean, (self.num_envs,1)))**2, axis=0), (1,-1))
        #     self.critic_input_std_sum += np.reshape(np.sum((critic_input - np.tile(critic_input_mean, (self.num_envs,1)))**2, axis=0), (1,-1))
        #
        # actor_input_std = np.sqrt(self.actor_input_std_sum / self.counter)
        # critic_input_std = np.sqrt(self.critic_input_std_sum / self.counter)
        #

        #     return (actor_input - np.tile(actor_input_mean, (self.num_envs,1))) / (actor_input_std + 1e-8), (critic_input - np.tile(critic_input_mean, (self.num_envs,1))) / (critic_input_std + 1e-8)
        # else:
        return actor_input, critic_input

    def select_action(self, state, idx):
        state = torch.FloatTensor(state)
        # return np.clip(self.policy.act(state, memory), self.env.action_space.low, self.env.action_space.high)
        return self.policy.act(state, self.memory, idx)

    def update(self):
        print("[ppo.py] Update with {} samples".format(self.advantages_batch.numel()))
        # compute discounted reward
        advantage = torch.zeros([self.num_envs, 1], dtype=torch.float, device=target)
        self.policy = self.policy.to(target)
        for reward, critic_input, is_terminal, i in zip(reversed(self.memory.rewards),
                                                        reversed(self.memory.critic_input),
                                                        reversed(self.memory.is_terminal),
                                                        reversed(range(int(self.max_timestep)))):
            if i == int(self.max_timestep - 1):
                net_input = self.memory.last_critic_input[:].to(target)
                next_state_values = torch.squeeze(self.policy.critic_net(net_input), dim=0)
            else:
                next_state_values = self.policy.critic_net(self.memory.critic_input[i + 1].to(target))
            not_done = (~torch.from_numpy(is_terminal).view(-1, 1)).float().to(target)
            curr_state_values = self.policy.critic_net(critic_input.to(target))
            delta = torch.from_numpy(reward).view(-1, 1).to(
                target) + self.gamma * next_state_values * not_done - curr_state_values
            advantage = delta + self.gamma * self.lamda * advantage * not_done
            self.advantages_batch[:, i] = advantage.squeeze().detach()
            self.state_values_batch[:, i] = curr_state_values.squeeze().detach()

        # package the batch information for the training process and get an iterator for the dataset
        batch_iter = self.memory.batch_loader(
            self.mini_batch_size,
            torch.stack(self.memory.actor_input, dim=1).view(-1, self.actor_input_dim).detach(),  # actor input
            torch.stack(self.memory.critic_input, dim=1).view(-1, self.critic_input_dim).detach().cpu(),  # critic input
            torch.stack(self.memory.actions, dim=1).view(-1, self.env.num_acts).detach(),  # actions
            torch.squeeze(torch.stack(self.memory.logprobs, dim=1).view(-1, 1), 1).detach(),  # logprobs
            (self.advantages_batch + self.state_values_batch).view(-1, 1).detach().cpu(),  # returns
            ((self.advantages_batch - self.advantages_batch.mean()) / (self.advantages_batch.std() + 1e-5)).view(-1,
                                                                                                                 1).detach().cpu(),
            # normalized advantages
            self.state_values_batch.view(-1, 1).detach().cpu())  # state values

        # optimize policy for K epochs
        for _ in range(self.K_epochs):
            # See https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12 for details of PPO algorithm

            batch_iter = self.memory.reset_iter()

            # loop through mini batches and update network weights
            for _ in range(int(self.advantages_batch.view(-1, 1).size(0) / self.mini_batch_size)):
                # get the next mini-batch from the dataset
                old_actor_inputs, old_critic_inputs, old_actions, old_logprobs, advantages, rewards, old_state_values, _ = batch_iter.next()

                # send tensors to GPU
                old_actor_inputs = old_actor_inputs.to(target)
                old_critic_inputs = old_critic_inputs.to(target)
                old_actions = old_actions.to(target)
                old_logprobs = old_logprobs.to(target)
                advantages = advantages.to(target)
                rewards = rewards.to(target)
                old_state_values = old_state_values.to(target)

                # evaluating old actions and values every time the policy is updated
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_actor_inputs, old_critic_inputs,
                                                                            torch.unsqueeze(old_actions, dim=1))

                # normalize the logprobs and compute the KL divergence
                logprobs_kl = (logprobs - (logprobs.exp().sum() + 1e-16).log().detach()).view(-1, 1)
                old_logprobs_kl = (old_logprobs - (old_logprobs.exp().sum() + 1e-16).log()).view(-1, 1)
                D_kl = (old_logprobs_kl.exp().detach() * (old_logprobs_kl.detach() - logprobs_kl)).sum()

                # find the clipped value function MSE loss
                clipped_value = old_state_values.detach() + (state_values - old_state_values.detach()).clamp(
                    min=-self.state_value_clip_factor, max=self.state_value_clip_factor)
                state_value_loss = 0.5 * torch.max((clipped_value - rewards.detach()) ** 2,
                                                   (state_values - rewards.detach()) ** 2)

                # finding the ratio r = (pi_theta / pi_theta_old)
                ratios = torch.exp(
                    logprobs.view(-1, 1) - old_logprobs.detach().view((-1, 1)))  # since log(a / b) = log(a) - log(b)

                # compute the surrogate loss
                if self.surrogate_cost_fcn == 'truly':
                    kl_cost = self.delta * torch.ones([self.mini_batch_size, 1], dtype=torch.float)
                    for i in range(self.mini_batch_size):
                        if D_kl > self.delta and ratios[i] * advantages[i] > advantages[i]:
                            kl_cost[i] = self.alpha * D_kl

                    surr = ratios * advantages.detach() - kl_cost
                else:
                    surr1 = ratios * advantages.detach()  # L(theta) = E_t[r * A]
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                        1 + self.eps_clip) * advantages.detach()  # clip(r, 1 - eps, 1 + eps) * A

                    surr = -torch.min(surr1, surr2)

                loss = surr.mean() + self.state_value_loss_coef * state_value_loss.mean() - self.entropy_loss_coef * dist_entropy.mean()  # see Algorithm 5 in the link posted above. note that here we have a single loss function for both networks (actor and critic)

                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()  # take the mean of the loss function (cf. Algo. 5); backpropagate

                # clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

                # update plots
                # max_gradient_actor = max([
                #     self.policy.actor_net[-1].weight.grad.abs_().max().item()])
                # self.tensorboard_writer.update_plot("gradient_h_infinity_actor", max_gradient_actor)

                # max_gradient_critic = max([
                #     self.policy.critic_net[-1].weight.grad.abs_().max().item()])
                # self.tensorboard_writer.update_plot("gradient_h_infinity_critic", max_gradient_critic)

                self.tensorboard_writer.update_plot("ppo/avg_surr_loss", surr.mean().item())
                self.tensorboard_writer.update_plot("ppo/state_value_loss",
                                                    self.state_value_loss_coef * state_value_loss.mean().item())
                self.tensorboard_writer.update_plot("ppo/avg_loss", loss.mean().item())
                self.tensorboard_writer.update_plot("ppo/explained_variance",
                                                    ((1 - (rewards - state_values).std() ** 2) / (
                                                            rewards.std() ** 2 + 1e-16)).item())  # explained variance: indicator of "how good" is the value function of predicting the returns
                self.tensorboard_writer.update_plot("ppo/kl_divergence", D_kl.item())
                self.tensorboard_writer.update_plot("ppo/avg_policy_entropy", dist_entropy.mean().item())
                self.tensorboard_writer.update_plot("ppo/avg_return", rewards.mean().item())
                self.tensorboard_writer.update_plot("ppo/avg_advantage", advantages.mean().item())

                del loss

        # take one step of learning rate decrease
        if self.lr_scheduler.gamma < 1.0:
            self.lr_scheduler.step()

        # save parameters for later
        torch.save(self.policy.state_dict(), self.actor_critic_weights_init_path)

        self.policy = self.policy.to('cpu')

    def learn(self, eval_every_n, max_iteration, log_dir, record_video):
        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 1

        # training loop
        for iteration in range(0, max_iteration + 1):
            start_time = time.time()
            self.env.reset()  # reset the environment to the initial state
            for t in range(self.max_timestep):
                # self.aux.flag = time_step == 1
                # self.aux.visualizing = False
                # actor_input, critic_input = self.aux.step()
                actor_input = self.env._observation.copy()
                critic_input = self.env._observation.copy()
                actor_input, critic_input = self.normalize_state(actor_input, critic_input,
                                                                 True)  # normalize the state according to calculation of running mean and standard deviation

                # run old policy
                action = self.select_action(actor_input, t)  # samples an action from the old policy distribution
                self.memory.critic_input[t] = torch.FloatTensor(critic_input)
                state, reward, done, _ = self.env.step(action,
                                                       visualize=False)  # returns the scaled states from environment!

                self.memory.rewards[t] = reward
                self.memory.is_terminal[t] = done

                self.tensorboard_writer.update_plot("ppo/number_of_dones", sum(done))
                time_step += 1

            # update policy
            # self.aux.flag = False
            # self.aux.visualizing = False
            # actor_input, critic_input = self.aux.step()

            _, critic_input = self.normalize_state(actor_input, critic_input,
                                                   True)  # normalize the state according to calculation of running mean and standard deviation

            self.memory.last_critic_input = torch.FloatTensor(critic_input)
            self.update()

            # self.aux.flag = False
            # self.aux.visualizing = False
            # self.aux.update()

            time_step = 0

            running_reward += sum(reward) / len(
                reward)  # keep a running total of the rewards accumulated for the episode

            avg_length += t

            time_elapsed = time.time() - start_time
            print('Iteration {} done time elapsed: {}'.format(iteration, time_elapsed))

            # save every eval_every_n iterations
            if iteration % eval_every_n == 0:

                # visualize and record video in RaisimOgre
                print("[RAISIM_GYM] Visualizing in RaisimOgre")
                self.env.show_window()
                self.env.start_recording_video(
                    log_dir + "/PPO_{}_{}.mp4".format(iteration, datetime.now().strftime("%m_%d_%Y__%H_%M_%S")))
                self.env.reset()
                for _ in range(self.max_timestep):
                    # self.aux.flag = False
                    # self.aux.visualizing = True
                    # actor_input, critic_input = self.aux.step()
                    actor_input = self.env._observation.copy()
                    critic_input = self.env._observation.copy()
                    actor_input, _ = self.normalize_state(actor_input, critic_input, False)
                    actor_input = torch.FloatTensor(actor_input)
                    action = self.policy.MultivariateGaussian(actor_input).sample().detach().cpu().data.numpy()
                    state, _, _, _ = self.env.step(action, visualize=True)

                self.env.hide_window()
                self.env.stop_recording_video()

                avg_length = int(avg_length / eval_every_n)
                running_reward = int(running_reward / eval_every_n)

                running_reward = 0
                avg_length = 0

            if iteration == max_iteration:
                self.tensorboard_writer.close_tensorboard()


def torchscript(self):
    # convert the policy to a TorchScript file for inferencing in C++
    dummy_input = torch.rand(1, self.actor_input_dim)

    traced_module = torch.jit.trace(self.policy, dummy_input)

    traced_module.save("ppo_model_traced.pt")


# Must pass an instance of this class or of a class subclassed from this one when instantiating a PPO() object. 'AUX' means 'Auxiliary' because this class can be used to interact with the PPO object. For example, Aux can be used to append disturbance values to the actor/critic inputs, it can be used to create additional Tensorboard plots, or it can even be used to pass data from a higher level PPO controller to a lower level PPO controller.
class AUX:
    def __init__(self):
        self.actor_input_extra_dim = 0
        self.critic_input_extra_dim = 0
        self.extra_plots = []
        self.env = None
        self.tensorboard = None
        self.flag = False  # 'True' at the start of an episode
        self.visualizing = False  # 'True' when visualizing in RaisimGym

    # virtual method, default behavior given here for example
    def step(self):  # called every time a step of the environment is taken

        return self.env._observation, self.env._observation  # must return actor_input, critic_input (modified if necessary)

    # virtual method, default behavior given here for example
    def update(self):  # called just after PPO training

        return


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
