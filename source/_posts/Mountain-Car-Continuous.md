---
title: 连续控制—DDPG/TD3/SAC/PPO算法实现-en
date: 2024-04-11 15:42:05
tags: [Reinforcement Learning, 算法实现]
categories: Reinforcement Learning

---
# Solutions to MountainCarContinuous-v0 (DDPG / TD3 / SAC / PPO)

Codes Link: <https://github.com/xiyanzzz/RL-Implement/tree/main/Continuous%20Control>

## Mountain Car Continuous Env

### Env Introduction

<img src="https://www.gymlibrary.dev/_images/mountain_car_continuous.gif" alt="Mountain Car continu" style="zoom:67%;" />

- Introduction on OpenAI Wiki: <https://github.com/openai/gym/wiki/MountainCarContinuous-v0>
- Goal: to strategically accelerate the car to reach the goal state on top of the right hill.
- Observation: Box(2)

| Num  | Observation  |  Min  | Max  |
| :--: | :----------: | :---: | :--: |
|  0   | Car Position | -1.2  | 0.6  |
|  1   | Car Velocity | -0.07 | 0.07 |

- Action: Box(1)

| Num  |                            Action                            | Min  | Max  |
| :--: | :----------------------------------------------------------: | :--: | :--: |
|  0   | Push car to the left (negative value) or to the right (positive value) | -1.  |  1.  |
- Reward: A negative reward of $-0.1*\text{action
  }^2$ is received at each timestep to penalise for taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100 is added to the negative reward for that timestep.

>  This reward function raises an exploration challenge, because if the agent does not reach the target soon enough, it will figure out that it is better not to move, and won't find the target anymore.
>
>  Note that this reward is unusual with respect to most published work, where the goal was to reach the target as fast as possible, hence favouring a bang-bang strategy.

### Additional Notes

1. Continuous control problem: This env is belong to continous control problem whose action space is continuous and infinite. Therefore, it is unsuited for us to use the previous algorithms like DQN, RRFORCE, etc. Discretization of actions facing the dimensional catastrophe problem. A more sensible approach is to apply continuous control algorithms such as DDPG, TD3, SAC, or PPO.

2. As explained in the official documentation, in such an environment where penalties are given based on the magnitude of the action, the agent will learn to "go to seed" if the car fails to reach the target in time to get positive feedback. Therefore, increasing the exploratory nature of the strategy by adding noise is especially necessary.

3. "Full throttle" strategy (,i.e., $action \equiv  1$) can't solve this env. Moreover, a natural idea is to apply the maximum force (always doing positive work) in the direction of the velocity so that the cart can pick up speed quickly relying on inertia to reach the finish line. However, this costs extra energy and is not an optimal solution.

## Deep Deterministic Policy Gradient (DDPG)

### DDPG Introduction

The DDPG algorithm can be viewed as an extension of DQN to continuous action space. Both the algorithms output a deterministic action (no sampling operation), and are off-policy. However, since Q-learning inherently need to find the max over actions in $\max_{A\in\mathcal{A}}Q_\star(S_{t+1},A)$, it mean that DQN is adapted specifically for enviroments with discrete action space.

DDPG belongs to a class of Actor-Critic algorithm where the actor network $\mu(s;\theta)$ is parameterized by $\theta$ and outputs a continuous action, and the critic network $Q(s,a;\omega)$ learns a Q-function just like AC algorithm does.

Additinoally, both the tricks of replay buffers and target networks used in DQN  can also be employed in DDPG.

#### Value learning of DDPG

According to the Bellman equation describing the action-value function, the loss function of critic network is represented as:
$$
L(\omega)=\mathbb{E}_{(s,a,r,s')\sim \mathcal D}\left[\left(Q(s,a;\omega)-(r+\gamma\cdot \bar Q(s',\bar \mu(s';\bar\theta);\bar \omega))\right)^2\right],
$$
where $\bar Q$ and $\bar \mu$ are target networks and $\mathcal D$ is the set of previous experiences.

#### Policy learning of DDPG

Nowthat we have learned an approximator to action-value function, a natural idea is use the critic to guide the learning of policy network. The goal of policy learning is to maximize the following objective function:
$$
J(\theta)=\mathbb{E} _{s\sim\mathcal D}[Q(s,\mu(s;\theta);\omega)].
$$
Here, we assume the Q-function is differentiable with respect to action, so the loss function can be given by:
$$
L(\theta)=-\mathbb{E} _{s\sim\mathcal D}\left[ \nabla_{\theta}\mu(s;\theta) \cdot\nabla_{a}Q(s,a;\omega)|_{a=\mu(s;\theta)}\right]
$$

#### Noise

Since DDPG trains a deterministic policy in an off-policy way, to increase the exploration of behaviour policy, a time-correlated OU noise is added at the original DDPG paper by its  authors.

![image-20240326143457248](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/04/20240404-114113.png)

<center><p class="image-caption">Image from reference[1]</p></center>

The stochastic differential equation (SDE) of OU process is described as follows:
$$
\mathrm{d}x_t=\theta\cdot(\mu-x_t)\cdot\mathrm{d}t+\sigma\cdot\mathrm{d}W_t,
$$
where $\theta>0$, $\sigma>0$, $\mu$ is mean, $W_t$ is Wiener process (or Brownian motion). Discretizing the above equation (forward differencing) yields:
$$
x_{n+1}=x_n+\theta\cdot(\mu-x_n)\cdot\Delta t+\sigma\cdot\Delta W_n,
$$
where $\Delta W_n \sim N(0,\Delta t)=\sqrt{\Delta t}\cdot N(0,1)$ is independent and identically distributed (IID). It is temporally correlated that make it different from Gaussian noise,  and is therefore suitable for the systems with inertia. Here, we will apply the same treatment to the training on the MountainCarContinuous-v0 env because it favors the car exploring in the same direction.

#### Soft update

The parameters of target networks are update in a soft way, that is, weighted average with a hyerparameter $0<\tau<1$.

### DDPG Implementation

![image-20240329111840836](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/03/29/20240329-111841.png)

<center><p class="image-caption">Image from reference[1]</p></center>

The code frame:

- `DDPG.py`
  - `class Config(object):` Store configuration parameters
  - `class PolicyNet(nn.Module):`
  - `class QValueNet(nn.Module):`
  - `class DDPGAgent:`
    - `actor:PolicyNet`&`critic:QValueNet`&`config:Config`
    - `def get_action():`&`def update():`&`def soft_update():`&`def load_pretrained_model():`&`def save_trained_model():`
  - `def test(env_name:str, agent:DDPGAgent)->None:` Evaluate model (average return on specified number of episodes)
  - `def train(env_name:str, agent:DDPGAgent, config:Config)->DDPGAgent:`
- `utils.py`
  - `class OrnsteinUhlenbeckActionNoise:` Generate OU noise
  - `class ReplayBuffer:`
  - `def moving_average(a, window_size):` Smooth training curves

#### Modules

##### Libraries

```python
import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import utils
import matplotlib.pyplot as plt
import gym

# utils.py
import torch
import collections
import random
import numpy as np
import torch.nn as nn
```



##### Configuration (include hyperparameters)

```python
# Configuration
class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.algor_name = 'DDPG'
        # Env
        self.env_seed = None # random seed
        # Buffer
        self.BUFFER_SIZE = None
        self.BATCH_SIZE = None
        self.MINIMAL_SIZE = None # ready to update
        # Agent
        self.HIDDEN_DIM = None
        self.actor_lr = None
        self.critic_lr = None
        self.tau = None # soft-update factor for target network
        self.gamma = None
        self.model_path = None # model save/load path
        # OU Noise
        self.OU_Noise = False
        self.mu = None
        self.sigma = None
        self.theta = None
        # Train
        self.episode_num = None
        self.step_num = None
        # Evaluate
        self.is_load = False # whether to load pre-trained model
```

##### Ornstein-Uhlenbeck Noise (in utils.py)

```python
class OrnsteinUhlenbeckActionNoise:
    '''reference: https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py#L31'''
    def __init__(self, mu, sigma, theta=.15, dt=1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
```

![OU Noise](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/08/20240408-153117.png)

<mark>Note: Reducing the scale of the noise over the course of training may result in better training data. But it is not adopted in following  implementation.</mark>

##### Replay Buffer (in utils.py)

```python
class ReplayBuffer:
    def __init__(self, BUFFER_SIZE, BATCH_SIZE):
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.buffer = collections.deque(maxlen=BUFFER_SIZE)

    def add_experience(self, *experience):
        self.buffer.append(experience) # tuple: (state, action, reward, next_state, done)
        '''
        (np.ndarray(2,).float,  np.ndarray(1,).float, float, np.ndarray(2,).float, bool)
        '''
    
    def get_batch(self):
        transitions = random.sample(self.buffer, self.batch_size)
        batch_s, batch_a, batch_r, batch_s_, batch_done = zip(*transitions)

        batch_s_tensor = torch.tensor(batch_s, dtype=torch.float32)
        batch_a_tensor = torch.tensor(batch_a, dtype=torch.float32)
        batch_r_tensor = torch.tensor(batch_r, dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.tensor(batch_s_, dtype=torch.float32)
        batch_done_tensor = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(-1)
        
        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor, batch_done_tensor
    
    def get_size(self):
        return len(self.buffer)
```

##### DDPG Agent

``` python
# Actor Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.action_bound = action_bound
        self.actorNN = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.actorNN(state) * self.action_bound # constrain the action to requirentment of env

#Critic Network
class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.crticNN = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state, action):
        cat = torch.cat([state, action], dim = 1)
        return self.crticNN(cat)

# Agent
class DDPGAgent:
    def __init__(self, STATE_DIM, ACTION_DIM, ACTION_BOUND, config:Config):

        self.ACTION_DIM = ACTION_DIM
        self.ACTION_BOUND = ACTION_BOUND

        # Update
        self.gamma = config.gamma
        self.tau = config.tau

        # Buffer
        self.replay_buffer = utils.ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)

        # Noise
        self.Noise_OU = config.OU_Noise
        if config.OU_Noise:
            self.ou_noise = utils.OrnsteinUhlenbeckActionNoise(config.mu, config.sigma, theta=config.theta)
        else:
            self.sigma = config.sigma # Gaussian Noise
        
        self.actor = PolicyNet(STATE_DIM, ACTION_DIM, config.HIDDEN_DIM, ACTION_BOUND) # TODO
        self.critic = QValueNet(STATE_DIM, ACTION_DIM, config.HIDDEN_DIM)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.is_load:
            self.load_pretrained_model(model_path=config.model_path)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state).item()
        if self.Noise_OU:
            action += self.ou_noise() # add ou_noise
        else:
            action += self.sigma * np.random.randn(self.ACTION_DIM) # add gaussian_noise
        return np.clip(action, -self.ACTION_BOUND, self.ACTION_BOUND) # 截断，该动作参与了更新，Q(a,s)
    
    def update(self):
        b_s, b_a, b_r, b_ns, b_done = self.replay_buffer.get_batch()
        # update critic network
        next_q_values = self.target_critic(b_ns, self.target_actor(b_ns))
        target_values = b_r + self.gamma * next_q_values * (1-b_done)
        critic_loss = F.mse_loss(target_values, self.critic(b_s, b_a))
        self.critic_optimizer.zero_grad() # ^_^
        critic_loss.backward()
        self.critic_optimizer.step()

        # upadte actor network
        actor_loss = -self.critic(b_s, self.actor(b_s)).mean() # updated critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.actor, self.target_actor)
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()): # .parameters()是generator
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def load_pretrained_model(self, model_path):
        self.actor.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_parameters, model_path):
        torch.save(model_parameters, model_path)
```

##### Train Function

```python
# Train function
def train(env_name:str, agent:TD3Agent, config:Config)->TD3Agent:
    
    # env_train = gym.make(env_name, render_mode='human')
    env_train = gym.make(env_name)

    return_list = []
    for episode_i in range(1, config.episode_num + 1):
        episode_return = 0
        s, _ = env_train.reset(seed=config.env_seed)
        done = False
        for _ in range(config.step_num):
        # while not done:
            a = agent.get_action(s)
            s_, r, done, _, _ = env_train.step(a)
            agent.replay_buffer.add_experience(s, a, r, s_, done)
            s = s_
            episode_return += r

            if agent.replay_buffer.get_size() >= config.MINIMAL_SIZE:
                agent.update()

            if done: break

        return_list.append(episode_return)
        if episode_i % 10 == 0:
            print(f"Episode: {episode_i}, Avg.10_most_recent Return: {np.mean(return_list[-10:]):.2f}")

    # save trained model
    agent.save_trained_model(agent.actor.state_dict(), config.model_path)
    # plot trainning curves 
    episodes_list = list(range(len(return_list)))
    mv_return = utils.moving_average(return_list, 9)
    plt.plot(episodes_list, return_list, label="episode_return")
    plt.plot(episodes_list, mv_return, label="mv_episode_return")
    plt.legend(loc=0)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'TD3 on {env_name}')
    plt.savefig(f"./Figures/TD3 on {env_name}.png")
    plt.show()

    return agent
```

##### Evaluation Function

```python
# Evaluation
def test(env_name:str, agent:TD3Agent)->None:
    epsiode_num = 100
    # render or not
    env_test = gym.make(env_name)
    # env_test = gym.make(env_name, render_mode='human')
    return_list = []
    for i in range(epsiode_num):
        s, _ = env_test.reset()
        done = False
        return_episode = 0
        for step_i in range(500):
            a = agent.actor(torch.tensor(s, dtype=torch.float32)).detach().numpy()
            s, r, done, _, _ = env_test.step(a)
            return_episode += r
            if done or step_i == 499:
                print(f'episode.{i+1}: {return_episode}')
                return_list.append(return_episode)
                break
    print('================================================')
    print(f"average return on {epsiode_num} episodes of testing: {np.mean(return_list):.2f}")
    print('================================================')
```

##### Main

```python

if __name__ == "__main__":

    '''choose env'''
    env_name =  'MountainCarContinuous-v0'
    # env_name =  'Pendulum-v1'

    config = Config()
    '''if true: load pretrained model and test; else: train a model from 0'''
    config.is_load = True

    if env_name == 'MountainCarContinuous-v0':
        # Env
        config.env_seed = 0
        # Buffer
        config.BUFFER_SIZE = 10000
        config.BATCH_SIZE = 64
        config.MINIMAL_SIZE = 1000
        # Agent
        config.HIDDEN_DIM = 64
        config.actor_lr = 3e-4
        config.critic_lr = 3e-3
        config.tau = 0.005
        config.gamma = 0.9 # 0.98
        config.model_path = "./Models/DDPG-MountainCarContinuous.pth"
        # Noise # 参数参考自: https://github.com/samhiatt/ddpg_agent/tree/master
        config.OU_Noise = True
        config.mu = np.array([0.]) # []
        config.sigma = 0.25
        config.theta = 0.05
        # Train
        config.episode_num = 200
        config.step_num = 500

#    elif env_name == 'Pendulum-v1':
    
    np.random.seed(config.env_seed)
    torch.manual_seed(config.env_seed)

    env = gym.make(env_name)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_BOUND = env.action_space.high[0]

    agent = DDPGAgent(STATE_DIM, ACTION_DIM, ACTION_BOUND, config=config)
    env.close()

    if config.is_load:
        test(env_name, agent) # pretrained
    else:
        agent = train(env_name, agent, config) # untrained
        test(env_name, agent) # trained

'''average return on 100 episodes of testing: 90.11'''
```

#### Training Results

- Return curves:

![DDPG on MountainCarContinuous-v0](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/08/20240408-170516.png)

- Evaluation result:

> average return on 100 episodes of testing: 90.33

### References

- [1] [T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, D. Wierstra. Continuous control with deep reinforcement learning. 4th International Conference on Learning Representations, San Juan, Puerto Rico, 2016. ICLR.org, 2016](https://arxiv.org/abs/1509.02971)

- [2] <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>

- [3] <https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC13%E7%AB%A0-DDPG%E7%AE%97%E6%B3%95.ipynb>

## Twin Delayed DDPG (TD3)

### TD3 Introduction

Like the DQN algorithm, DDPG overestimates the value function unevenly, which can lead to a fragile training performance. Twin Delayed DDPG (TD3) is an algorithm that addresses this issue by introducing three critical tricks: (Reference from [here](https://spinningup.openai.com/en/latest/algorithms/td3.html))

1. **Clipped Double-Q Learning.** TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.
2. **“Delayed” Policy Updates.** TD3 updates the policy (and target networks) less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.
3. **Target Policy Smoothing.** TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action. (It can be interpreted as an increase in robustness)

#### Value learning of TD3

1. Action noise: 

$$
\bar a(s')=\mathrm{clip}\left(\bar \mu(s';\bar \theta)+\mathrm{clip}(\epsilon,-c,c),a_{Low},a_{High}\right),\quad\epsilon\sim\mathcal{N}(0,\sigma),
$$
   where $\bar\mu$ is target policy and $\epsilon$ is noise.

2. TD target: 
$$
y=r+\gamma\cdot(1-d)\cdot\min_{i=1,2}\bar Q_i(s',\bar a(s');\bar \omega_{i}),
$$
   where $\bar Q_i$ is target Q-function.

3. Loss function: 
$$
L(\omega_i)=\mathbb{E}_{(s,a,r,s')\sim \mathcal D}\left[\left(Q_i(s,a;\omega_i)-y\right)^2\right].
$$

#### Policy learning of TD3

The same with DDPG. However, in TD3, the policy is updated less frequently than the Q-functions are. 

### TD3 Implementation

The code frame:

- `TD3.py`
  - `class Config(object):` Store configuration parameters
  - `class PolicyNet(nn.Module):`
  - `class QValueNet(nn.Module):`
  - `class TD3Agent:`
    - `actor:PolicyNet`&`critic:QValueNet`&`config:Config`
    - `def get_action():`&`def update():`&`def soft_update():`&`def load_pretrained_model():`&`def save_trained_model():`
  - `def test(env_name:str, agent:TD3Agent)->None:` Evaluate model (average return on specified number of episodes)
  - `def train(env_name:str, agent:TD3Agent, config:Config)->TD3Agent:`
- `utils.py`
  - `class OrnsteinUhlenbeckActionNoise:` Generate OU noise
  - `class ReplayBuffer:`
  - `def moving_average(a, window_size):` Smooth training curves

#### Modules

Mostly the same as DDPG, here are the major modifications.

##### Configuration

```python
	# Configuration
class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.algor_name = 'TD3'
        # Env # Buffer # Agent # Train # OU Noise # Evaluate: the same with DDPG
    	# For TD3
        '''change: extra parameters'''
        self.policy_freq = None # requency of delayed policy updates
        self.policy_noise = None # Noise added to target policy during critic update
        self.noise_clip = None # Range to clip target policy noise
```

##### DDPG Agent

```python
# Actor Network
class PolicyNet(nn.Module):

# Critic Network
class QValueNet(nn.Module):
    '''change: another Q-value Network'''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.crticNN1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.crticNN2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        cat = torch.cat([state, action], dim = 1)
        q1 = self.crticNN1(cat)
        q2 = self.crticNN2(cat)
        return q1, q2
    
'''cahneg'''
class TD3Agent:
    def __init__(self, STATE_DIM, ACTION_DIM, ACTION_BOUND, config:Config):

        self.ACTION_DIM = ACTION_DIM
        self.ACTION_BOUND = ACTION_BOUND

        # Update
        self.update_count = 0
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_freq = config.policy_freq
        self.gamma = config.gamma
        self.tau = config.tau

        # Buffer
        self.replay_buffer = utils.ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)

        # Noise
        self.Noise_OU = config.OU_Noise
        if config.OU_Noise:
            self.ou_noise = utils.OrnsteinUhlenbeckActionNoise(config.mu, config.sigma, theta=config.theta)
        else:
            self.sigma = config.sigma # Gaussian Noise
            
        self.actor = PolicyNet(STATE_DIM, ACTION_DIM, config.HIDDEN_DIM, ACTION_BOUND) # TODO
        self.critic = QValueNet(STATE_DIM, ACTION_DIM, config.HIDDEN_DIM)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.is_load:
            self.load_pretrained_model(config.model_path)

    def get_action(self, state):
    
    def update(self):
        b_s, b_a, b_r, b_ns, b_done = self.replay_buffer.get_batch()
        
        # update critic networks
        '''change: Add clipped noise to taeget policy action'''
        action_noise = (torch.randn_like(b_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.target_actor(b_ns) + action_noise).clamp(-self.ACTION_BOUND, self.ACTION_BOUND)
        '''change: Choose the smaller q-value to compute td target'''
        next_q_values1, next_q_values2 = self.target_critic(b_ns, next_action)
        next_q_values = torch.min(next_q_values1, next_q_values2)

        target_values = b_r + self.gamma * next_q_values * (1-b_done)
        '''change: Count the both q'''
        current_q_values1, current_q_values2 = self.critic(b_s, b_a)
        critic_loss = F.mse_loss(target_values, current_q_values1) + F.mse_loss(target_values, current_q_values2)

        self.critic_optimizer.zero_grad() # ^_^
        critic_loss.backward()
        self.critic_optimizer.step()
        '''change: Delay policy update'''
        self.update_count += 1
        if self.update_count % self.policy_freq == 0:
            # upadte actor network
            '''change: Choose one critic between two'''
            actor_loss = -self.critic(b_s, self.actor(b_s))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            self.soft_update(self.critic, self.target_critic)
            self.soft_update(self.actor, self.target_actor)

            self.update_count = 0

    def soft_update(self, net, target_net):
        
    def load_pretrained_model(self, model_path):
        
    def save_trained_model(self, model_parameters, model_path):


```

##### Main

```python
	if env_name == 'MountainCarContinuous-v0':		
    	config.policy_noise = 0.2 # Noise added to target policy during critic update
        config.noise_clip = 0.5 # Range to clip target policy noise
        config.policy_freq = 2 # requency of delayed policy updates
```

#### Training Results

Return curves:

![TD3 on MountainCarContinuous-v0](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/09/20240409-121730.png)

Evaluation result:

> average return on 100 episodes of testing: 93.90

### References

- [1] [S. Fujimoto, H. van Hoof, D. Meger. Addressing function approximation error in actor-critic methods. 35th International Conference on Machine Learning, Stockholmsmässan, Stockholm, Sweden, 2018. PMLR.org, 2018](https://arxiv.org/abs/1802.09477)
- [2] <https://spinningup.openai.com/en/latest/algorithms/td3.html>

- [3] [Author Realization](https://github.com/sfujim/TD3/)



## Soft Actor-Critic (SAC)

**Entropy** is a measure of randomness for random variables. For example, let $a$ be a random variable with probability mass or density function $\pi$, then its entropy is definded as:
$$
H(\pi)=\mathbb{E}_{a\sim \pi}[-\log \pi(a)].
$$
**Maximum Entropy Reinforcement Learning** does not only aim to maximize the expection of cumulative reward, but also to optimize the entropy sum over time, which enables the agent to explore more and avoid converging to local optima. Therefore, the objective of policy learning is: (suppose $R_t$ is a function with respect to $s_t, a_t, s_{t+1}$)
$$
\pi^*=\arg\max_\pi\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^\infty R_t+\alpha H(\pi(\cdot|s_t))\right].
$$
There are some conceptual differences between maximum entropy and traditional RL:

- Soft state-value function:

$$
V_\pi(s_t)=\mathbb{E}_{\tau\sim\pi}\left [\sum^\infty_{l=0}\gamma^l\left(R_{t+l}+\alpha\cdot H(\pi(\cdot \mid s_{t+l}))\right) \right].
$$

- Soft action-value function:

$$
Q_\pi(s_t,a_t)=\mathbb{E}_{\tau\sim\pi}\left [\sum^\infty_{l=0}\gamma^l\left(R_{t+l}+\alpha\cdot \gamma \cdot H(\pi(\cdot \mid s_{t+l+1}))\right) \right].
$$

- Soft Bellman equations:

$$
\begin{aligned}
V(s_t)&=\mathbb{E}_{a_t\sim\pi}[Q(s_t,a_t)]+\alpha H(\pi(\cdot|s_t))\\
&=\mathbb{E}_{a_t\sim\pi}[Q(s_t,a_t)-\alpha\log\pi(a_t|s_t)]\\
\end{aligned},
$$

and
$$
Q(s_t,a_t)=\mathbb{E}_{s_{t+1}\sim P}\left[R_t+\gamma\cdot V(s_{t+1})\right].
$$


### SAC Introduction

Unlike DDPG and TD3, Soft Actor Critic (SAC) is an algorithm that learning a stochastic policy in an off-policy way, and it also belongs to the category of maximum entropy reinforcement learning.

SAC learns a policy $\pi_\theta$ and two Q-functions $Q_{\omega_{1,2}}$ with the help of two target Q-networks $\bar Q_{\bar \omega_{1,2}}$.

#### Value learning of SAC

- TD target:

$$
y(s_{t+1},a_{t+1},d)=r_t+\gamma\cdot(1-d)\cdot\left(\min_{j=1,2}\bar Q(s_{t+1},a_{t+1};\bar\omega_j)-\alpha\log\pi(a_{t+1}|s_{t+1})\right),
$$

   where $a_{t+1}\sim \pi(\cdot|s_{t+1};\theta)$.

-  Loss function:

$$
L_{Q}(\omega)=\mathbb{E}_{(s_t,a_t,r_t,s_{t+1},d)\sim \mathcal D,a_{t+1}\sim\pi_\theta(\cdot|s_{t+1})}\left[\frac{1}{2}\left(Q_\omega(s_t,a_t)-y(s_{t+1},a_{t+1},d)\right)^2\right]
$$

#### Policy learning of SAC

The policy network is updated by maximizing the expected future return plus expected future entropy, that is, maximizing $V_\pi(s_t)$ at each time step. The loss function can be written as:
$$
L_\pi(\theta)=\mathbb{E}_{s_t\sim \mathcal D,a_t\sim f_\theta}[\alpha\log(\pi_\theta(a_t|s_t))-Q_\omega(s_t,a_t)].
$$
Since the policy network outputs means $\mu_i$ and standard deviations $\sigma_i$ of Gaussian distributions to approximate the action distribution, the sampled actions cannot be directly derived with respect to parameters $\theta$. Here, **reparameterization trick** is used to comupute gradient. That is, the action is computed directly from the parameters output by the policy network with random noise:
$$
a_t=f_\theta(\mu,\sigma,s_t,\epsilon_t)=\mu(s_t)+\sigma(s_t)\cdot\epsilon_t,~~~~~\epsilon\sim\mathcal N(0,1).
$$

Moreover, the entropy regularization coefficient $\alpha$ should be automatically adjusted according to the magnitude of entropy:
$$
L(\alpha)=\mathbb{E}_{s_t\sim R,a_t\sim\pi(\cdot|s_t)}[-\alpha\log\pi(a_t|s_t)-\alpha H_0],
$$
where $H_0$ is a hyperparameter. (non-essential and no guaranteed performance improvment over fixed $\alpha$ value version)

### SAC Implementation

<mark>Once again, OU noise is employed to increase the exploration of the behavior policy (or used only to pre-populate the replay buffer).</mark>

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/09/20240409-185452.png" alt="image-20240409185451885" style="zoom:50%;" />

<center><p class="image-caption">Image from reference[2]</p></center>

The code frame:

- `SAC.py`
  - `class Config(object):` Store configuration parameters
  - `class PolicyNet(nn.Module):`
  - `class QValueNet(nn.Module):`
  - `class TD3Agent:`
    - `actor:PolicyNet`&`critic:QValueNet`&`config:Config`
    - `def get_action():`&`def calc_target()`&`def update():`&`def soft_update():`&`def load_pretrained_model():`&`def save_trained_model():`
  - `def test(env_name:str, agent:SACAgent)->None:` Evaluate model (average return on specified number of episodes)
  - `def train(env_name:str, agent:SACAgent, config:Config)->SACAgent:`
- `utils.py`
  - `class OrnsteinUhlenbeckActionNoise:` Generate OU noise
  - `class ReplayBuffer:`
  - `def moving_average(a, window_size):` Smooth training curves

#### Modules

##### Configuration

```python
# Configuration
class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.algor_name = 'SAC'
        # Env # Buffer # Agent # Train # OU Noise # Evaluate: the same with DDPG/TD3
        # SAC
        self.alpha = None
        self.target_entropy = None
        self.alpha_lr = None
        self.refactor_reward = False
```

##### SACAgent

```python
# Actor Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim) # n 个 mu 值
        self.fc_std = nn.Linear(hidden_dim, action_dim) # n 个 std 值
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) # 光滑的ReLU，确保std > 0
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样 记r.v.: u ~ mu(u|s)
        log_prob = dist.log_prob(normal_sample) # 计算采样点u的概率，取log
        action = torch.tanh(normal_sample) # 采样动作规范到(-1, 1)内, 即 a' = tanh(u) ~ pi'(a'|s)
        # 计算a'的对数概率, 即log(pi'(a'|s))
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        # a = c * a' ~ pi(a|s)
        action = action * self.action_bound
        # 计算a的对数概率, 即log(pi(a|s))
        log_prob -= torch.log(torch.tensor(self.action_bound)) # 注意下维度
        return action, log_prob

# Critic Network
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out1 = torch.nn.Linear(hidden_dim, 1)

        self.fc3 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        # two Q networks
        q1 = F.relu(self.fc1(cat))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc_out1(q1)

        q2 = F.relu(self.fc3(cat))
        q2 = F.relu(self.fc4(q2))
        q2 = self.fc_out2(q2)
        return q1, q2

# Agent
class SACAgent:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, STATE_DIM, ACTION_DIM, ACTION_BOUND, config:Config):
        self.actor = PolicyNet(STATE_DIM, config.HIDDEN_DIM, ACTION_DIM, ACTION_BOUND)
        self.critic = QValueNet(STATE_DIM, config.HIDDEN_DIM, ACTION_DIM)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.is_load:
            self.load_pretrained_model(model_path=config.model_path)

        self.log_alpha = torch.tensor(np.log(config.alpha), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.target_entropy = config.target_entropy
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=config.alpha_lr)

        # Buffer
        self.replay_buffer = utils.ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)
        
        # Noise
        self.OU_Noise = config.OU_Noise
        if config.OU_Noise:
            self.ou_noise = utils.OrnsteinUhlenbeckActionNoise(config.mu, config.sigma, theta=config.theta)
        
        # Update
        self.gamma = config.gamma
        self.tau = config.tau
        self.refactor_reward = config.refactor_reward

        self.ACTION_BOUND = ACTION_BOUND

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state)[0].detach().numpy()
        # for Mountain Car to be aggressive
        if self.OU_Noise:
            action += self.ou_noise() # array([])
            action = np.clip(action, -self.ACTION_BOUND, self.ACTION_BOUND)
        return action
    
    def calc_target(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, log_prob = self.actor(next_states)
            entropy = -log_prob
            q1_value, q2_value = self.target_critic(next_states, next_actions)
            next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    
    def update(self):
        b_s, b_a, b_r, b_ns, b_done = self.replay_buffer.get_batch()

        # update critic networks
        if self.refactor_reward:
            b_r = (b_r + 8.0) / 8.0 # only for pendulum

        target_values = self.calc_target(b_r, b_ns, b_done)
        current_q_values1, current_q_values2 = self.critic(b_s, b_a)
        critic_loss = F.mse_loss(target_values, current_q_values1) + F.mse_loss(target_values, current_q_values2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update policy network
        new_action, log_prob = self.actor(b_s)
        entropy = -log_prob
        q_value1, q_value2 = self.critic(b_s, new_action)
        actor_loss = (-self.log_alpha.exp() * entropy - torch.min(q_value1, q_value2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # update target networks
        self.soft_update(self.critic, self.target_critic)
    
    def soft_update(self, net, target_net):

    def load_pretrained_model(self, model_path):

    def save_trained_model(self, model_parameters, model_path):
```

##### Main

```python
    if env_name == 'MountainCarContinuous-v0':        
        # Agent
        config.HIDDEN_DIM = 64
        config.actor_lr = 3e-4
        config.critic_lr = 3e-3
        config.alpha_lr = 3e-4
        config.tau = 0.005
        config.gamma = 0.9 # 0.98
        config.alpha = 0.01
        config.target_entropy = -1 # -env.action_space.shape[0]
        config.model_path = "./Models/SAC-MountainCarContinuous.pth"
```

#### Training Results

- Return curves:

![SAC on MountainCarContinuous-v0](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/09/20240409-232808.png)

- Evaluation result:

> average return on 10 episodes of testing: 94.20

### References

- [1] Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." *International conference on machine learning*. PMLR, 2018.
- [2] <https://spinningup.openai.com/en/latest/algorithms/sac.html>
- [3] <https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC14%E7%AB%A0-SAC%E7%AE%97%E6%B3%95.ipynb>

## Proximal Policy Optimization (PPO)

### PPO Introduction

#### TRPO

Traditional policy-based learning algorithms, such as REINFORCE and Actor-Critic, can collapse the training curves due to the bad step size of the gradient update. **Trust Region Policy Optimization (TRPO)** is such an algorithm that updates policies over a trust region, constraining the old and new policies to be close while obtaining guaranteed performance improvement.

The optimization target of policy gradient is 
$$
\begin{aligned}
J(\theta)&=\mathbb E_{\tau\sim p_{\theta}(\tau)}\left[\sum_{t=0}^\infty\gamma^t\cdot R_t \right]\\
&=\mathbb E_{s_0\sim p_{\theta}(s_0)}\left[V^{\pi_\theta}(s_0) \right]
\end{aligned}
$$
Optimization Increment:
$$
\begin{aligned}
J(\theta')-J(\theta)&=J(\theta')-\mathbb E_{s_0\sim p_{\theta}(s_0)}\left[\sum_{t=0}^\infty\gamma^t\cdot V^{\pi_\theta}(s_t)-\sum_{t=1}^\infty\gamma^t\cdot V^{\pi_\theta}(s_t) \right]\\
&=\mathbb E_{\tau\sim p_{\theta'}(\tau)}\left[\sum_{t=0}^\infty\gamma^t\cdot R_t \right]+E_{\tau\sim p_{\theta'}(\tau)}\left[\sum_{t=0}^\infty\gamma^t\cdot(\gamma\cdot V^{\pi_\theta}(s_{t+1}-V^{\pi_\theta}(s_t)) \right]\\
&~~(\text{From the fact that initial state is independent of the policy})\\
&=\mathbb E_{\tau\sim p_{\theta'}(\tau)}\left[\sum_{t=0}^\infty\gamma^t\cdot \left(R_t+\gamma\cdot V^{\pi_\theta}(s_{t+1})-V^{\pi_\theta}(s_t)\right)\right]\\
&=\mathbb E_{\tau\sim p_{\theta'}(\tau)}\left[\sum_{t=0}^\infty\gamma^t\cdot A^{\pi_\theta}(s_t,a_t)\right]\\
&=\sum_{t=0}^\infty\gamma^t\cdot \mathbb E_{s_t\sim P_{\theta'}(s_t)}\left[\mathbb E_{a_t\sim \pi_{\theta'}(a_t|s_t)}\left[ A^{\pi_\theta}(s_t,a_t) \right] \right]\\
&~~(\text{Definition of state visition distribution})\\
&=\frac{1}{1-\gamma}\cdot\mathbb E_{s\sim \nu^{\pi_{\theta'}}(s)}\mathbb E_{a\sim \pi_{\theta}(a|s)}\left[A^{\pi_\theta}(s,a) \right]\\
&~~(\text{Importance Sampling trick})\\
&=\frac{1}{1-\gamma}\cdot\mathbb E_{s\sim \nu^{\pi_{\theta'}}(s)}\mathbb E_{a\sim \pi_{\theta}(a|s)}\left[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}\cdot A^{\pi_\theta}(s,a) \right]
\end{aligned}
$$
Furthermore, if the old and new policy distributions are close, the above equation can be approximated as:
$$
J(\theta')-J(\theta)\approx\frac{1}{1-\gamma}\cdot\mathbb E_{s\sim \nu^{\pi_{\theta}}(s)}\mathbb E_{a\sim \pi_{\theta}(a|s)}\left[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}\cdot A^{\pi_\theta}(s,a) \right].
$$
Thus, the optimization target of TRPO is given by:
$$
\theta'\leftarrow \arg \max_{\theta'} L_{\theta}(\theta')~~~~~\mathrm{s.t.}~ \mathbb E_{s\sim \nu^{\pi_{\theta}}(s)}\left[ D_{KL}(\pi_{\theta'}(\cdot|s)\parallel\pi_{\theta}(\cdot|s))\right]\le\delta,
$$
where $L_\theta(\theta')=\mathbb E_{s\sim \nu^{\pi_{\theta}}(s)}\mathbb E_{a\sim \pi_{\theta}(a|s)}\left[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}\cdot A^{\pi_\theta}(s,a) \right]$, $D_{KL}$ means Kullback-Leibler divergence.

TRPO makes an approximate solution to the above optimization problem, i.e., a first-order and second-order approximation to the objective function and constraints, respectively, and then solves the problem using the Karush-Kuhn-Tucker condition. In addition, the conjugate gradient method is used to find the inverse of the Hessian matrix and a linear search is performed to ensure that the constraints are satisfied.

#### PPO

While TRPO can theoretically guarantee monotonicity in the performance of policy learning, solving optimization problems with constraints has high complexity. **Proximal Policy Optimization (PPO)** methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

There are two primary variants of PPO: PPO-Penalty and PPO-Clip. We will focus on the latter (PPO-Clip), which has been shown to be more effective in a large number of experiments.

**PPO-Penalty** transforms the hard constraint into penalty terms in the objective function:
$$
\arg\max_{\theta'}\mathbb E_{s\sim \nu^{\pi_{\theta}}(s)}\mathbb E_{a\sim \pi_{\theta}(a|s)}\left[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}\cdot A^{\pi_{\theta}}(s,a)-\beta\cdot D_{KL}(\pi_{\theta'}(\cdot|s)\parallel\pi_{\theta}(\cdot|s))\right].
$$
**PPO-Clip** limits in the objective function to ensure that the gap between the new parameters and the old ones is not too large:
$$
\arg\max_{\theta'}\mathbb E_{s\sim \nu^{\pi_{\theta}}(s)}\mathbb E_{a\sim \pi_{\theta}(a|s)}\left[\min\left(\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}\cdot A^{\pi_{\theta}}(s,a),\mathrm{clip}\left(\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)},1-\epsilon,1+\epsilon\right)A^{\pi_{\theta}}(s,a)\right)\right].
$$
The intuitive explanation of PPO-clip is that:

- if $A^{\pi_{\theta}}(s,a)>0$, indicating that the action's value is greater than the average, maximizing this equation increases the $\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}$ but not more than $1+\epsilon$.
- if $A^{\pi_{\theta}}(s,a)<0$, maximizing this equation decreases the $\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}$ but not more than $1-\epsilon$.

![image-20240411005752022](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/11/20240411-005752.png)

<center><p class="image-caption">Image from reference [1]</p></center>

#### GAE

Denote single step TD error: $\delta_t=r_t+\gamma\cdot V(s_{t+1})-V(s_t)$.

Advantage can be estimated by:
$$
\begin{aligned}
\hat A_t^{(1)}&=-V(s_t)+r_t+\gamma V(s_{t+1})=\delta_t, \\
\hat A_t^{(2)}&=-V(s_t)+r_t+\gamma r_{t+1}+\gamma^2V(s_{t+2})=\delta_t+\gamma\cdot \delta_{t+1}, \\
&\begin{array}{cc}\vdots\\\end{array} \\
\hat A_t^{(k)}&=-V(s_t)+r_t+\gamma r_{t+1}+\ldots+\gamma^{k-1}r_{t+k-1}+\gamma^kV(s_{t+k})=\sum_{l=0}^{k-1}\gamma^l\cdot\delta_{t+l}.
\end{aligned}
$$
**Generalized Advantage Estimation** does an exponentially weighted average of the above estimates:
$$
\begin{aligned}
\hat A_{t}^{GAE}& =(1-\lambda)(\hat A_t^{(1)}+\lambda \hat A_t^{(2)}+\lambda^2\hat A_t^{(3)}+\cdots)  \\
&=(1-\lambda)(\delta_t+\lambda(\delta_t+\gamma\delta_{t+1})+\lambda^2(\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2})+\cdots) \\
&=(1-\lambda)(\delta(1+\lambda+\lambda^2+\cdots)+\gamma\delta_{t+1}(\lambda+\lambda^2+\lambda^3+\cdots)+\gamma^2\delta_{t+2}(\lambda^2+\lambda^3+\lambda^4+\cdots) \\
&=(1-\lambda)\left(\delta_t\frac1{1-\lambda}+\gamma\delta_{t+1}\frac\lambda{1-\lambda}+\gamma^2\delta_{t+2}\frac{\lambda^2}{1-\lambda}+\cdots\right) \\
&=\sum_{l=0}^\infty(\gamma\lambda)^l\delta_{t+l}
\end{aligned},
$$


where $\lambda\in[0,1]$ is a hyperparameter.

### PPO Implementation

PPO is an on-policy algorithm, so it can't ultize replay buffer trick. As a result, it is struggling to solve the `MountainCarContinuous` env.

> [PPO struggling at MountainCar whereas DDPG is solving it very easily. Any guesses as to why?](https://www.reddit.com/r/reinforcementlearning/comments/9o8ez0/ppo_struggling_at_mountaincar_whereas_ddpg_is/?rdt=44656)
>
> Sparse rewards. In OpenAI Gym `MountainCar` you only get a positive reward when you reach the top.
>
> PPO is an on-policy algorithm. It performs a policy gradient update after each episode and throws the data away. Reaching the goal in MountainCar by random actions is a pretty rare event. When it finally happens, it's very unlikely that a single policy gradient update will be enough to start reaching the goal consistently, so PPO gets stuck again with no learning signal until it reaches the goal again by chance.
>
> On the other hand, DDPG stores this event in the replay buffer so it does not forget. The TD bootstrapping of the Q function will eventually propagate the reward from the goal backwards into the Q estimate for other states near the goal.
>
> This is a big advantage of off-policy RL algorithms.
>
> Also DDPG uses an Ornstein-Uhlenbeck process for time-correlated exploration, whereas PPO samples Gaussian noise. The Ornstein-Uhlenbeck process is more likely to generate useful exploratory actions. (The exploration methods are not immutable properties of the algorithms, just the Baselines implementations.)

In the PPO algorithm implementation, the training function is somewhat different from the previous three. Specifically, we adopt `rollout` training method, that is, we update each time with a `rollout_len` number of experiences without caring about how many episodes the agent has completed in these `rollout_len` time steps. Therefore, if `rollout_len` is large enough, we can gather a handful of successes even in purely random situations, which contributes to a nice learning.

What's more, PPO's exploration comes from the randomness of its own strategy. [Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima](https://spinningup.openai.com/en/latest/algorithms/ppo.html). A entropy-regularization term is added to our actor loss function to support a more exploratory policy.

The main reference for the PPO implementation is: <https://github.com/mandrakedrink/PPO-pytorch/tree/master>

- `PPO.py`
  - `class Config(object):` Store configuration parameters
  - `class PolicyNet(nn.Module):`
  - `class ValueNet(nn.Module):`
  - `class PPOAgent:`
    - `actor:PolicyNet`&`critic:QValueNet`&`config:Config`
    - `def get_action():`&`def update():` &`def load_pretrained_model():`&`def save_trained_model():`
  - `def test(env_name:str, agent:PPOAgent)->None:` Evaluate model (average return on specified number of episodes)
  - `def train(env_name:str, agent:PPOAgent, config:Config)->PPOAgent:`
- `utils.py`
  - `def init_weights(m):` Initialize network weights
  - `class Memory:`
  - `def compute_advantage(td_delta, gamma, lambda_):` Generalized Advantage Estimation
  - `def moving_average(a, window_size):` Smooth training curves



#### Modules

##### Configuration

```python
class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.algor_name = 'PPO'
        # Env
        self.env_name = None
        self.env_seed =  None
        # Agent
        self.HIDDEN_DIM_A = None
        self.HIDDEN_DIM_C = None
        self.actor_lr = None
        self.critic_lr = None
        self.gamma = None
        self.lambda_ = None # for GAE
        self.entropy_coef = None # entropy regularization term
        self.epsilon =  None
        self.epochs = None
        self.batch_size = None
        self.model_path = None
        # Train
        self.max_rollout_num = None
        self.rollout_len = None
        self.solved_reward = None
        self.min_completed_episode_num =  None
        # Evaluate
        self.is_load = False
```

##### Memory (in utils.py)

```python
class Memory:
    """Storing the memory of the trajectory (s, a, r ...)."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_states = []

    def to_tensor(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)
        next_states = torch.tensor(self.next_states, dtype=torch.float)
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        masks = torch.tensor(self.masks, dtype=torch.float)
        return states, actions, next_states, rewards, masks

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_states = []
```

##### Advantage Estimation (in utils.py)

```python
def compute_advantage(td_delta, gamma, lambda_):
    '''GAE'''
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lambda_ * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
```

##### Weight Initialization (in utils.py)

```python
def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)
```

##### PPO Agent

```python
'''网络结构: 隐藏层数不同, std限制在了(1/e, e)内, 加了权重初始化'''
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=32):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        log_std = torch.tanh(self.fc_std(x))

        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        return action, dist
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
class PPOAgent:
    def __init__(self, STATE_DIM, ACTION_DIM, ACTION_BOUND, config:Config):
        '''初始化'''
        self.actor = PolicyNet(STATE_DIM, ACTION_DIM, ACTION_BOUND, hidden_dim=config.HIDDEN_DIM_A).apply(utils.init_weights)
        self.critic = ValueNet(STATE_DIM, hidden_dim=config.HIDDEN_DIM_C).apply(utils.init_weights)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.is_load:
            self.load_pretrained_model(model_path=config.model_path)

        self.gamma = config.gamma
        self.lambda_ = config.lambda_

        # self.memory = Memory()
        self.epochs = config.epochs
        # self.batch_size = config.batch_size
        self.entropy_coef = config.entropy_coef
        self.epsilon = config.epsilon
    
        self.action_bound = ACTION_BOUND

    def get_action(self, state): # TODO
        state = torch.tensor(state, dtype=torch.float)
        action, dist = self.actor(state)
        return action.detach().numpy(), dist

    
    def update(self, memory):
        actor_losses, critic_losses = [], []

        states, actions, next_states, rewards, masks = memory.to_tensor()

        # 1.估计优势函数 - GAE
        td_target = rewards + self.gamma * self.critic(next_states) * masks
        td_delta = td_target - self.critic(states)
        advantages = utils.compute_advantage(td_delta, self.gamma, self.lambda_)

        # 2. 存档theta_old, 即ratio的分母取对数
        action_dists =  self.actor(states)[1]
        old_log_probs = action_dists.log_prob(actions).detach() # detach()很重要，因为目标参数只能是theta'

        old_td_targets = td_target.detach()

        # 3. 以theta_old为基础, 多次更新
        for i_epoche in range(self.epochs):
            # print(i_epoche)

            # 4. tehta'，即ratio的分子取对数
            action_dists =  self.actor(states)[1]
            cur_log_probs = action_dists.log_prob(actions)

            # 5. 计算ratio
            ratio = torch.exp(cur_log_probs - old_log_probs) # 第一次循环必定为1

            # compute entropy
            entropys = action_dists.entropy().mean() # 熵正则项, 该技巧适用于多种算法

            # 6. 计算actor损失L(theta'): 截断, 比较取min
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2)) - entropys * self.entropy_coef

            # 7. 计算critic损失
            cur_values = self.critic(states)
            critic_loss = torch.mean(F.mse_loss(cur_values , old_td_targets))

            # 8. 更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # log
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)
    
    def load_pretrained_model(self, model_path):

    def save_trained_model(self, model_parameters, model_path):
```

##### Training Function

```python
def train(env_name:str, agent:PPOAgent, config:Config) -> PPOAgent:

    env_train = gym.make(env_name)
    # env_train = gym.make(env_name, render_mode='human')

    s, _ = env_train.reset(seed=config.env_seed)
    done = False
    
    memory = utils.Memory()

    return_list = []
    actor_losses_list = []
    critic_losses_list = []
    episode_return = 0
    for _ in range(config.max_rollout_num):
        '''Do not reset until done'''
        for _ in range(config.rollout_len):
            a, _ = agent.get_action(s)
            s_, r, done, _, _ = env_train.step(a)

            memory.states.append(s)
            memory.actions.append(a)
            memory.next_states.append(s_)
            memory.rewards.append([r])
            memory.masks.append([1-done])

            s = s_
            episode_return += r
            if done:
                return_list.append(episode_return)
                episode_return = 0
                # reset env
                s, _ = env_train.reset(seed=config.env_seed)

        if config.solved_reward is not None:
            '''Indicators met and training completed'''
            if len(return_list) >= config.min_completed_episode_num and np.sum(return_list[-10:]) > config.solved_reward:
                print("Congratulations, it's solved! ^_^")
                break
        # update
        actor_losses, critic_losses = agent.update(memory)

        actor_losses_list.append(actor_losses)
        critic_losses_list.append(critic_losses)

        memory.clear_memory()

        # 打印训练信息
        completed_episode_num = len(return_list)
        if  completed_episode_num % 10 == 0 and completed_episode_num >= 10:
            print(f"Episode: {completed_episode_num}, Avg.10_most_recent Return: {np.mean(return_list[-10:]):.2f}")

    agent.save_trained_model(agent.actor.state_dict(), config.model_path)
    # plot trainning curves

    return agent
```

##### Main

```python
    config = Config()
    '''if true: load pretrained model and test; else: train a model from 0'''
    # config.is_load = True

    if env_name == 'MountainCarContinuous-v0':
        # Env
        config.env_name = "MountainCarContinuous-v0"
        config.env_seed = 0 #555
        # Agent
        config.HIDDEN_DIM_A = 32
        config.HIDDEN_DIM_C = 64
        config.actor_lr = 1e-3
        config.critic_lr = 5e-3
        config.gamma = 0.95 # key 0.99/0.95 0.95/0.98
        config.lambda_ = 0.98
        config.entropy_coef = 0.01 # 0.003
        config.epsilon = 0.2
        config.epochs = 64
        config.batch_size = 1000
        config.model_path = "./Models/PPO-MountainCarContinuous.pth"
        # Train
        config.max_rollout_num = 10000
        config.rollout_len = 1000
        config.solved_reward = 92
        config.min_completed_episode_num = 500
```

#### Training Results

- Return curves

![PPO on MountainCarContinuous-v0](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/11/20240411-123905.png)

- Loss curves

![PPO_on_MountainCarContinuous-v0_loss_curves](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/11/20240411-123934.png)

- Evaluation result

> average return on 100 episodes of testing: 92.05

### References

- [1] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al. 2017
- [2] [TRPO](https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95) & [PPO](https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95) & <https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC12%E7%AB%A0-PPO%E7%AE%97%E6%B3%95.ipynb>
- [3] [OpenAI Spining Up - TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html) & [OpenAI Spining Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

- [4] https://github.com/mandrakedrink/PPO-pytorch

## Conclusion

<img src="https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2024/04/11/20240411-135408.gif" alt="MountainCarContinuous-v0_result_TD3" style="zoom: 33%;" />

<center><p class="image-caption">Solution example - Algorithm: TD3 and Return: 95.09</p></center>

