import numpy as np
import gym
import lbforaging

from learner import QMIX

env = gym.make("Foraging-8x8-2p-1f-v2")

qmix_agents = QMIX(env.observation_space[0].shape, (18,1), 6, 2)

n_agents = 2
N_GAMES = 20000
PRINT_INTERVAL = 100
best_score = -np.inf
scores = []
evaluate = False

if evaluate:
    qmix_agents.load_models()

for i in range(N_GAMES):

    obs, info = env.reset()
    obs_1 = obs
    obs_2 = np.array([obs[0], obs[1], obs[2], obs[6], obs[7], obs[8], obs[3], obs[4], obs[5]], dtype=np.float32)
    obs = (obs_1, obs_2)
    old_actions = {0:0, 1:0}

    score = 0
    episode_step = 0
    dones = [False] * 2
    while not any(dones):
        
        actions = qmix_agents.choose_actions(old_actions, obs)
        obs_, rewards, dones, info = env.step(list(actions.values()))
        reward = sum(rewards)
        score += reward

        for j in range(n_agents):
            qmix_agents.store_transition(j, i, episode_step, obs[j], actions[j], reward, obs_[j], dones[j], old_actions[j])
        
        if not evaluate:
            if (i % 200 == 0) and (i > 0):
                qmix_agents.replace_target_networks()
            qmix_agents.learn()

        obs = obs_
        old_actions = actions

        episode_step += 1
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    if not evaluate:
        if avg_score > best_score:
            qmix_agents.save_models()
            best_score = avg_score
    if i % PRINT_INTERVAL == 0:
        print('episode', i, 'average score {:.1f}'.format(avg_score))
