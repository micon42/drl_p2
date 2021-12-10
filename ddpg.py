import numpy as np
import torch
import time
from datetime import datetime
from ddpg_agent import Agent

LOAD_NETWORK_PARAMS = False


def ddpg_continuous(env, brain_name, episodes=3000):

    agent = Agent(state_size=33, action_size=4, seed=int(time.time()))

    mean_score = 0.0
    max_mean_score = 0.0
    score_list = []
    eps_start = 0.9
    eps_min = 0.01
    esp_factor = 0.995

    if LOAD_NETWORK_PARAMS:
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
        eps_start = 0.5
        eps_min = 0.01
        esp_factor = 0.995
    eps = eps_start

    for episode in range(1, episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = 0
        store = True
        eps = max(eps_min, esp_factor*eps)

        while True:
            actions = agent.act(states, eps)            # select an action
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                  # see if episode has finished
            score += np.mean(np.array(rewards))                                # update the score
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states                             # roll over the state to next time step

            if any(dones):
                # exit loop if episode finished
                mean_score = 0.99 * mean_score + 0.01 * score
                max_mean_score = max(max_mean_score, mean_score)
                score_list.append(score)
                print("\rEpisode: {}, Score: {:.4}, Mean Score: {:.4}, Max Mean Score: {:.4}, eps: {:.4}".format(
                    episode, score, mean_score, max_mean_score, eps), end="")
                break

            if 0 == episode % 300:
                if store:
                    store = False
                    torch.save(agent.qnetwork_local.state_dict(),
                               f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            else:
                store = True

        if mean_score > 30.0:
            print("\rEpisode: {}, Score: {}, Mean Score: {}, eps: {}".format(episode, score, mean_score, eps))
            torch.save(agent.qnetwork_local.state_dict(), f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
            break
    torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

    return score_list
