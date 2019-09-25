import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

all_actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]


def cliff(state):
    if(state[1]>0 and state[1]<11):
        if(state[0]==3):
            return(True)
    return(False)

def take_step(state,index):
    reward = -1
    action=all_actions[index]
    new_state=np.add(state,action)
    if(new_state[0]<0 or new_state[0]>3 or new_state[1]<0 or new_state[1]>11):
        new_state=np.copy(state)
    if(cliff(new_state)):
        new_state=[3,0]
        reward = -100
    return(new_state,reward)

def epsilon_greedy(epsilon,state,gridworld):
    e = np.random.uniform()
    index=0
    if(e<epsilon):
        index = np.random.randint(4)
    else:
        index = np.random.choice(np.where(gridworld[state[0],state[1],:]==gridworld[state[0],state[1],:].max())[0])
    return(index)


def terminal(state):
    if(state[0]==3 and state[1]==11):
        return(True)
    return(False)


def q_learning(episodes,alpha,epsilon):
    gridworld = np.ones((4,12,4))
    gridworld[3,11,:]=0
    history = []
    for episode in range(episodes):
        state=[3,0]
        reward_sum=0
        while(not terminal(state)):
            index = epsilon_greedy(epsilon,np.copy(state),gridworld)
            next_state,reward = take_step(np.copy(state),index)
            gridworld[state[0],state[1],index]+=alpha*(reward+np.max(gridworld[next_state[0],next_state[1],:])-gridworld[state[0],state[1],index])
            state=np.copy(next_state)
            reward_sum+=reward
        history.append(reward_sum)
    return(history)

def sarsa(episodes,alpha,epsilon):
    gridworld = np.ones((4, 12, 4))
    gridworld[3, 11, :] = 0
    history = []
    for episode in range(episodes):
        state = [3, 0]
        reward_sum = 0
        index = epsilon_greedy(epsilon, np.copy(state), gridworld)

        while (not terminal(state)):
            next_state, reward = take_step(np.copy(state), index)
            next_index = epsilon_greedy(epsilon, np.copy(next_state), gridworld)
            gridworld[state[0], state[1], index] += alpha * ( reward + gridworld[next_state[0], next_state[1], next_index] - gridworld[state[0], state[1], index])
            index=next_index
            state = np.copy(next_state)
            reward_sum += reward
        history.append(reward_sum)
    return (history)


def plot():
    plt.ylim([-100,0])
    q_total = np.zeros(500)
    for i in tqdm(range(100)):
        q = np.array(q_learning(500,0.1,0.1)).reshape(500)
        q_total+=q/100
    # print(q)
    plt.plot(q_total)
    s_total = np.zeros(500)
    for i in tqdm(range(100)):
        s = np.array(sarsa(500,0.1,0.1)).reshape(500)
        s_total+=s/100
    plt.plot(s_total)
    plt.legend(['Q learning','SARSA'])
    plt.savefig('./q7.png')

plot()