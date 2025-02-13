import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def run():
    env = gym.make('FrozenLake-v1',map_name='8x8',is_slippery=True)

    # Parameters
    n_episode = 10000
    alpha = 0.1
    epsilon = 0.1
    decay = 0.001

    # Q_table
    Q = np.zeros((env.observation_space.n,env.action_space.n))

    # Return
    Returns = []

    for i in range(n_episode):

        print(f"----episode: {i}")
        observation, info = env.reset()
        Return                                              = 0.
        episode_over                                        = False

        while not episode_over:

            # 1. Choose an action
            if np.random.rand() < epsilon:
                action                                      = env.action_space.sample() 
            else:
                max_index                                   = np.where(Q[observation] == np.max(Q[observation]))[0]
                action                                      = np.random.choice(max_index)

            # 2. take action and observe
            next_observation,reward,terminated,truncated,_  = env.step(action)          

            # 3. Update Rule
            Q[observation][action]                          = Q[observation][action] + alpha*(reward + max(Q[next_observation]) - Q[observation][action])

            # 4. Preparation for next iter
            Return                                          += reward               
            episode_over                                    = terminated or truncated   
            observation                                     = next_observation
            epsilon                                         *= decay
    
        # Record return after episode termination
        Returns.append(Return)
    
    env.close()

    # Plot
    plt.figure()
    plt.plot(Returns)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    plt.show()

if __name__ == '__main__':
    run()



