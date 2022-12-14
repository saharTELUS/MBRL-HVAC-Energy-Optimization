import numpy as np
import gym
from gym import spaces
from gym.spaces import Box, Discrete, Tuple, Dict
import time
import os
CONSTANT_NINF = -9e99

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_to_keep, lows, highs, mask):
        super().__init__(env)
        self.env = env
        self.obs_to_keep = obs_to_keep
        self.lows = lows
        self.highs = highs
        self.mask = mask
        self.observation_space = \
            spaces.Box(
                low=lows, 
                high=highs, 
                shape=((len(obs_to_keep),)), 
                dtype=self.env.observation_space.dtype
                )
    
    def observation(self, obs):
        if np.max(obs) > 1:
            print("more than 0")
        if np.min(obs) < 0:
            print("less 0 ")
        # modify obs
        return np.clip(obs[self.obs_to_keep], self.lows, self.highs)

class HNPAgent:
    def __init__(
        self,
        env,
        obs_mask,
        low,
        high,
        initial_eps =  1,
        eps_annealing = 0.9,
        eps_annealing_interval = 10,
        learning_rate: float = 0.1,
        learning_rate_annealing: float = 0.9,
        learning_rate_annealing_interval = 10,
        obs_discretization_steps = 1.0,
        gamma: float = 0.99,
    ) -> None:
        """
        Main HNP Agent Class

        :param: env: Gym environment to train the agent on
        :param: obs_mask: List of integers describing type of variables in the environment, 0 = slow moving continuous, 1 = fast moving continuous, 2 = discrete, ex: [0, 0, 1, 2]
        :param: low: List of integers describing lower bound of all variables in environment, if variable is discrete then is disregarded
        :param: high: List of integers describing upper bound of all variables in environment, if variable is discrete then is number of possible discrete values for that variable
        :param: initial_eps: Initial epsilon to use for epsilon greedy policy
        :param: eps_annealing: Float between 0 and 1 to multiply epsilon by every eps_annealing_interval timesteps
        :param: eps_annealing_interval: Number of episodes after which epsilon is multiplied by eps_annealing
        :param: learning_rate: Initial learning_rate to use for Q-Learning update
        :param: learning_rate_annealing: Float between 0 and 1 to multiply learning_rate by every learning_rate_annealing_interval timesteps
        :param: learning_rate_annealing_interval: Number of episodes after which learning_rate is multiplied by learning_rate_annealing
        :param: obs_discretization_steps: List of integers describing discretization steps for continuous variables, if variable is discrete then is disregarded
        :param: gamma: Discount factor for environment
        """

        self.env = env

        self.gamma = gamma

        self.eps = initial_eps
        self.eps_annealing = eps_annealing
        self.eps_annealing_interval = eps_annealing_interval
        self.learning_rate = learning_rate
        self.learning_rate_annealing = learning_rate_annealing
        self.learning_rate_annealing_interval = learning_rate_annealing_interval

        self.low = low
        self.high = high

        self.slow_continuous_idx = np.where(obs_mask==0)[0]
        self.fast_continuous_idx = np.where(obs_mask==1)[0]
        self.to_discretize_idx = np.hstack((self.slow_continuous_idx, self.fast_continuous_idx))# Slow first

        self.cont_low = self.low[self.to_discretize_idx]
        self.cont_high = self.high[self.to_discretize_idx]
        self.discrete_idx = np.where(obs_mask==2)[0]
        self.permutation_idx = np.hstack((self.slow_continuous_idx, self.fast_continuous_idx, self.discrete_idx))
        self.n_slow_cont = len(self.slow_continuous_idx)
        self.n_fast_discrete = len(self.fast_continuous_idx) + len(self.discrete_idx)

        self.obs_steps = obs_discretization_steps
        self.discretization_ticks = self.get_ticks(self.env.observation_space, self.obs_steps) 

        self.obs_space_shape = self.get_obs_shape()
        self.act_space_shape = self.get_act_shape()
        self.qtb = self.make_qtb()
        
        self.vtb = self.make_vtb()
        self.state_visitation = np.zeros(self.vtb.shape)
        self.rewards = []
        self.average_rewards = []

        n_dim = len(self.obs_space_shape)
        if self.n_slow_cont > 0:
            portion_index_matrix = np.vstack((np.zeros(self.n_slow_cont), np.ones(self.n_slow_cont))).T
            self.all_portion_index_combos = np.array(np.meshgrid(*portion_index_matrix), dtype=int).T.reshape(-1, self.n_slow_cont)

    def transform_obs(self, obs):
        '''
        Get permuted observation
        :param: obs: Original observation
        '''
        return obs[self.permutation_idx]

    def make_qtb(self):
        '''
        Make and return Q table
        '''
        return np.zeros((*self.obs_space_shape, self.act_space_shape))
        
    def make_vtb(self):
        '''
        Make and return V table
        '''
        return np.zeros(self.obs_space_shape)

    def get_obs_shape(self):
        '''
        Get (modified) observation space shape
        '''
        return tuple(list([len(ticks) for ticks in self.discretization_ticks]) + list(self.high[self.discrete_idx]))

    def get_act_shape(self):
        '''
        Get action space shape
        '''
        return self.env.action_space.n

    def get_ticks(self, space, steps):
        '''
        Get ticks for continuous observations
        '''
        return [np.arange(space.low[i], space.high[i] + steps[i], steps[i]) for i in self.to_discretize_idx]

    def obs_to_index_float(self, obs):
        '''
        Transform observation into index in V table (can have float index because we are using HNP)
        '''
        return (obs - self.cont_low)/(self.cont_high - self.cont_low) * (np.array(self.vtb.shape[:len(self.cont_high)]) - 1)
    
    def choose_action(self, obs_index, mode="explore"):
        '''
        Choose action based on current V table index and mode
        :param: obs_index: Current V table index
        :param: mode: Current mode to use when acting, "explore" = epsilon greedy, "greedy" = fully greedy
        '''
        if mode == "explore":
            if np.random.rand(1) < self.eps:
                return self.env.action_space.sample()
            return np.argmax(self.qtb[tuple(obs_index)])
        
        if mode == "greedy": # For evaluation purposes
            return np.argmax(self.qtb[tuple(obs_index)])

    def get_vtb_idx_from_obs(self, obs):
        '''
        Get index in V table based on observation
        '''
        obs = self.transform_obs(obs)
        cont_obs = obs[:len(self.to_discretize_idx)]

        cont_obs_index_floats = self.obs_to_index_float(cont_obs)
        cont_obs_index = np.round(cont_obs_index_floats)
        obs_index = np.hstack((cont_obs_index, obs[len(self.to_discretize_idx):])).astype(int)

        return obs_index, cont_obs_index_floats


    def get_next_value(self, obs):
        '''
        Get next value to use for Q learning target based on current observation
        '''
        full_obs_index, cont_obs_index_floats = self.get_vtb_idx_from_obs(obs)
        if self.n_slow_cont == 0: # No HNP calculation needed
            return self.vtb[tuple(full_obs_index)], full_obs_index
        slow_cont_obs_index_floats = cont_obs_index_floats[:len(self.slow_continuous_idx)]
        slow_cont_obs_index_int_below = np.floor(slow_cont_obs_index_floats).astype(np.int32)
        slow_cont_obs_index_int_above = np.ceil(slow_cont_obs_index_floats).astype(np.int32)

        if len(self.to_discretize_idx) < len(obs):
            discrete_obs = obs[len(self.to_discretize_idx) + 1:]
        vtb_index_matrix = np.vstack((slow_cont_obs_index_int_below, slow_cont_obs_index_int_above)).T
        all_vtb_index_combos = np.array(np.meshgrid(*vtb_index_matrix)).T.reshape(-1, len(slow_cont_obs_index_int_above))

        portion_below = slow_cont_obs_index_int_above - slow_cont_obs_index_floats
        portion_above = 1 - portion_below
        portion_matrix = np.vstack((portion_below, portion_above)).T

        non_hnp_index = full_obs_index[len(self.slow_continuous_idx):]
        next_value = 0
        for i, combo in enumerate(self.all_portion_index_combos):
            portions = portion_matrix[np.arange(len(slow_cont_obs_index_floats)), combo]
            value_from_vtb = self.vtb[tuple(np.hstack((all_vtb_index_combos[i], non_hnp_index)).astype(int))]
            next_value += np.prod(portions) * value_from_vtb
        
        return next_value, full_obs_index
    
    def learn(self, n_episodes) -> None:
        '''

        '''
        obs = self.env.reset()
        prev_vtb_index, _ = self.get_vtb_idx_from_obs(obs)
        episode_reward = 0
        ep_n = 0
        n_steps = 0
        while ep_n <= n_episodes: 
            ac = self.choose_action(prev_vtb_index)
            # Set value table to value of max action at that state
            self.vtb = np.nanmax(self.qtb, -1)
            obs, rew, done, info = self.env.step(ac)
            episode_reward += rew
            next_value, next_vtb_index = self.get_next_value(obs)

            # Do Q learning update
            prev_qtb_index = tuple([*prev_vtb_index, ac])
            self.state_visitation[prev_qtb_index[:-1]] += 1
            curr_q = self.qtb[prev_qtb_index]
            q_target = rew + self.gamma * next_value
            self.qtb[prev_qtb_index] = curr_q + self.learning_rate * (q_target - curr_q)
            n_steps += 1
            prev_vtb_index = next_vtb_index
            if done: # New episode
                print(f"num_timesteps: {n_steps}")
                print(f"Episode {ep_n} --- Reward: {episode_reward}, Average reward per timestep: {episode_reward/n_steps}")
                avg_reward = episode_reward/n_steps
                self.rewards.append(episode_reward)
                self.average_rewards.append(avg_reward)

                n_steps = 0
                ep_n += 1

                if ep_n % self.eps_annealing_interval == 0:
                    self.eps = self.eps * self.eps_annealing

                if ep_n % self.learning_rate_annealing_interval == 0:
                    self.learning_rate = self.learning_rate * self.learning_rate_annealing

                episode_reward = 0
                obs = self.env.reset()
    
    def save_results(self):
        today = date.today()
        day = today.strftime("%Y_%b_%d")
        now = dt.now()
        time = now.strftime("%H_%M_%S")
        dir_name = f"/root/beobench_results/{day}/results_{time}"
        os.makedirs(dir_name)

        original_stdout = sys.stdout # Save a reference to the original standard output

        with open(f"{dir_name}/params.json", 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(json.dumps(config, indent=4))
            sys.stdout = original_stdout
        print("Saving results...")

        np.savez(
            f"{dir_name}/metrics.npz", 
            qtb=self.qtb, 
            rewards=self.rewards,
            average_rewards=self.average_rewards,
            state_visitation=self.state_visitation
            )