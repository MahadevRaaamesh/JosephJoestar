#-- Multi Arm Bandit Algorithm --#
## This is an algorithm to learn from environment
## This is a static environment where each actions give same answer
import numpy as np

#-- Envionment --#

true_probs = [0.2, 0.5, 0.3]
num_arms = len(true_probs)

#-- RL VARIABLES --#

num_steps = 10000
estimates = np.zeros(num_arms)
counts = np.zeros(num_arms)

#-- RL Action --#

def pull_arm(i):
    """Simulate environment"""
    return 1 if np.random.rand() < true_probs[i] else 0

#-- Epsilon-greedy RL strategy --#
##
# p=random()
# if(p<e) max action
# else any random action
## 

epsilon = 0.9

for t in range(num_steps):
    if np.random.rand() >= epsilon:
        action = np.random.randint(num_arms)  
    else:
        action = np.argmax(estimates)  

    reward = pull_arm(action)

    counts[action] += 1
    estimates[action] += (reward - estimates[action]) / counts[action] ## this part does running average so that past events also matters
    #We can use learning rate of alpha instead of 1/count for changing values of reward in slot
print("True probabilities:", true_probs)
print("Learned estimates:", estimates)
