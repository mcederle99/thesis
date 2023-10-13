import collections
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label=1)
    ax2 = fig.add_subplot(111, label=2, frame_on=False)

    ax.plot(x, epsilons, color='C0')
    ax.set_xlabel('Training Steps', color='C0')
    ax.set_ylabel('Epsilon', color='C0')
    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')

    plt.savefig(filename)

class Reward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        return obs, wrapped_reward, done, info

    def _reward(self, obs, env_reward):
        reward = [0, 0, 0, 0, 0, 0]
        agent_names = ['Agent-007', 'Agent-008', 'Agent-009', 'Agent-010', 'Agent-011', 'Agent-012']
        for i in range(len(agent_names)):
            if obs[agent_names[i]][4][1]:
                reward[i] -= 10
            
            elif obs[agent_names[i]][4][3]:
                reward[i] -= 10

            elif obs[agent_names[i]][4][4]:
                reward[i] -= 10

            elif len(obs[agent_names[i]][4][0]) != 0:
                reward[i] -= 10

            elif obs[agent_names[i]][4][2]:
                reward[i] -= 10

            elif obs[agent_names[i]][4][6]:
                reward[i] += 10

            else:
                reward[i] += env_reward[agent_names[i]]

        return np.float64(reward)

class Observation(gym.ObservationWrapper):
    def __init__(self, shape, env: gym.Env):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        agent_names = ['Agent-007', 'Agent-008', 'Agent-009', 'Agent-010', 'Agent-011', 'Agent-012']
        new_obs = {}
        for i in range(len(agent_names)):
            new_obs[agent_names[i]] = obs[agent_names[i]][13].data
            new_obs[agent_names[i]] = cv2.cvtColor(new_obs[agent_names[i]], cv2.COLOR_BGR2GRAY)
            new_obs[agent_names[i]] = cv2.resize(new_obs[agent_names[i]],self.shape[1:], interpolation=cv2.INTER_AREA)
            new_obs[agent_names[i]] = np.array(new_obs[agent_names[i]], dtype=np.uint8).reshape(self.shape)
            new_obs[agent_names[i]] = new_obs[agent_names[i]] / 255.0

        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32)
        self.stack = [collections.deque(maxlen=repeat), collections.deque(maxlen=repeat), collections.deque(maxlen=repeat), collections.deque(maxlen=repeat), collections.deque(maxlen=repeat), collections.deque(maxlen=repeat)]
    
    def reset(self):
        self.stack[0].clear()
        self.stack[1].clear()
        self.stack[2].clear()
        self.stack[3].clear()
        self.stack[4].clear()
        self.stack[5].clear()
        observation = self.env.reset()
        agent_names = ['Agent-007', 'Agent-008', 'Agent-009', 'Agent-010', 'Agent-011', 'Agent-012']
        for i in range(len(agent_names)):
            for _ in range(self.stack[i].maxlen):
                self.stack[i].append(observation[agent_names[i]])
        obs_dict = {}
        obs_dict['Agent-007'] = np.array(self.stack[0]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-008'] = np.array(self.stack[1]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-009'] = np.array(self.stack[2]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-010'] = np.array(self.stack[0]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-011'] = np.array(self.stack[1]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-012'] = np.array(self.stack[2]).reshape(self.observation_space.low.shape)
        return obs_dict

    def observation(self, observation):
        agent_names = ['Agent-007', 'Agent-008', 'Agent-009', 'Agent-010', 'Agent-011', 'Agent-012']
        for i in range(len(agent_names)):
            self.stack[i].append(observation[agent_names[i]])
        
        obs_dict = {}
        obs_dict['Agent-007'] = np.array(self.stack[0]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-008'] = np.array(self.stack[1]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-009'] = np.array(self.stack[2]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-010'] = np.array(self.stack[0]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-011'] = np.array(self.stack[1]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-012'] = np.array(self.stack[2]).reshape(self.observation_space.low.shape)
        return obs_dict

def make_env(env_name, agent_specs, scenario_path, headless) -> gym.Env:
    # Create environment
    env = gym.make(
        env_name,
        scenarios=[scenario_path],
        agent_specs=agent_specs,
        headless=headless,  # If False, enables Envision display.
        visdom=False,  # If True, enables Visdom display.
    )

    env = Reward(env=env)
    env = Observation(shape=(84,84,1), env=env)
    env = StackFrames(env, repeat=3)

    return env
