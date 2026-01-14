import os
import random
import itertools
from datetime import datetime, timedelta
import yaml
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib

from experience_replay import ReplayMemory
from dqn import DQN

# Set up matplotlib rendering backend for file-only graph outputs
matplotlib.use('Agg')

# Change device selection logic to always use CPU
DEVICE = torch.device("cpu")
# Directory for saving experiment data
SAVE_FOLDER = "runs"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Date formatting pattern for logging
TIME_FMT = "%m-%d %H:%M:%S"

class DeepQAgent:
    """
    Generalized DQN Agent (rewritten logic, same behavior as before).
    Handles training and evaluation of a DQN-based reinforcement learning agent.
    """
    def __init__(self, param_profile):
        # Load hyperparameters for a given profile
        with open('hyperparameters.yml', 'r') as handle:
            all_configs = yaml.safe_load(handle)
            params = all_configs[param_profile]
        
        self.config_id = param_profile
        self.env_id = params['env_id']
        self.gamma = params['discount_factor_g']
        self.alpha = params['learning_rate_a']
        self.sync_interval = params['network_sync_rate']
        self.buffer_capacity = params['replay_memory_size']
        self.batch_size = params['mini_batch_size']
        self.eps_initial = params['epsilon_init']
        self.eps_final = params['epsilon_min']
        self.eps_decay = params['epsilon_decay']
        self.reward_limit = params['stop_on_reward']
        self.hidden_neurons = params['fc1_nodes']
        self.env_extra_args = params.get('env_make_params', {})
        self.use_double = params['enable_double_dqn']
        self.use_dueling = params['enable_dueling_dqn']
        # Set save paths
        self.log_path = os.path.join(SAVE_FOLDER, f'{self.config_id}.log')
        self.model_path = os.path.join(SAVE_FOLDER, f'{self.config_id}.pt')
        self.plot_path = os.path.join(SAVE_FOLDER, f'{self.config_id}.png')
        self.loss_criterion = nn.MSELoss()
        self.optimizer = None

    def run(self, train_mode=True, visualize=False):
        if train_mode:
            self._log(f"Training started at {datetime.now().strftime(TIME_FMT)}\n", clear=True)
            graph_last_update = datetime.now()
        # Instantiate environment
        environment = gym.make(self.env_id, render_mode='human' if visualize else None, **self.env_extra_args)
        state_size = environment.observation_space.shape[0]
        action_count = environment.action_space.n
        # Set up tracking lists
        reward_history = []
        if train_mode:
            epsilon = self.eps_initial
            memory = ReplayMemory(self.buffer_capacity)
            target_net = DQN(state_size, action_count, self.hidden_neurons, self.use_dueling).to(DEVICE)
            policy_net = DQN(state_size, action_count, self.hidden_neurons, self.use_dueling).to(DEVICE)
            target_net.load_state_dict(policy_net.state_dict())
            self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.alpha)
            update_counter = 0
            epsilon_schedule = []
            best_reward = float('-inf')
        else:
            policy_net = DQN(state_size, action_count, self.hidden_neurons, self.use_dueling).to(DEVICE)
            policy_net.load_state_dict(torch.load(self.model_path))
            policy_net.eval()
        episode_idx = 0
        # Training or evaluation loop
        for episode_idx in itertools.count():
            obs, _ = environment.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            done = False
            total_reward = 0.0
            while not done and (not train_mode or total_reward < self.reward_limit):
                # Epsilon-greedy action choice
                if train_mode and random.random() < epsilon:
                    selected_action = torch.tensor(environment.action_space.sample(), dtype=torch.int64, device=DEVICE)
                else:
                    with torch.no_grad():
                        act_values = policy_net(state.unsqueeze(0))
                        selected_action = act_values.squeeze().argmax()
                next_state, reward, done, truncated, info = environment.step(selected_action.item())
                total_reward += reward
                state_next_tensor = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
                if train_mode:
                    memory.append((state, selected_action, state_next_tensor, reward_tensor, done))
                    update_counter += 1
                state = state_next_tensor
                # Learn from memory if enough experience is stored
                if train_mode and len(memory) > self.batch_size:
                    mb = memory.sample(self.batch_size)
                    self._optimize_step(mb, policy_net, target_net)
                    epsilon = max(epsilon * self.eps_decay, self.eps_final)
                    epsilon_schedule.append(epsilon)
                    if update_counter > self.sync_interval:
                        target_net.load_state_dict(policy_net.state_dict())
                        update_counter = 0
            reward_history.append(total_reward)
            # Checkpointing and graphing
            if train_mode and total_reward > best_reward:
                delta_perc = ((total_reward - best_reward) / abs(best_reward)+1e-8) * 100 if best_reward != float('-inf') else 100
                self._log(f"{datetime.now().strftime(TIME_FMT)}: Best reward {total_reward:.1f} ({delta_perc:+.1f}%) episode {episode_idx}, saving\n")
                torch.save(policy_net.state_dict(), self.model_path)
                best_reward = total_reward
            if train_mode and (datetime.now() - graph_last_update > timedelta(seconds=10)):
                self._make_plot(reward_history, epsilon_schedule)
                graph_last_update = datetime.now()

    def _log(self, message, clear=False):
        # Write log entries to file, optionally clearing
        mode = 'w' if clear else 'a'
        with open(self.log_path, mode) as logf:
            logf.write(message)
        print(message.strip())

    def _optimize_step(self, minibatch, policy_net, target_net):
        """Optimize model parameters using a randomly sampled batch"""
        # Unpack transitions
        state_b, act_b, next_state_b, reward_b, done_b = zip(*minibatch)
        states = torch.stack(state_b)
        actions = torch.stack(act_b).unsqueeze(1)
        rewards = torch.stack(reward_b)
        next_states = torch.stack(next_state_b)
        done_mask = torch.tensor(done_b, dtype=torch.bool, device=DEVICE)

        q_eval = policy_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = target_net(next_states).max(1)[0]
        target = rewards + (~done_mask).float() * self.gamma * q_next
        loss = self.loss_criterion(q_eval.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _make_plot(self, reward_series, epsilons):
        fig = plt.figure(1)
        fig.clf()
        plt.subplot(121)
        reward_avg = [np.mean(reward_series[max(0, i-99):(i+1)]) for i in range(len(reward_series))]
        plt.ylabel('Mean Rewards')
        plt.plot(reward_avg)
        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilons)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.plot_path)
        plt.close(fig)

# Alias "Agent" for compatibility with existing imports
Agent = DeepQAgent
