import numpy as np
import matplotlib.pyplot as plt
import math

def td0(episodes, alpha=0.1):
    values = np.ones((7)) * 0.5
    values[0] = 0
    values[6] = 0
    total = []
    for episode in range(episodes):
        initial_state = 3
        state = initial_state
        while (state != 0 and state != 6):
            step = -1 if not np.random.randint(2) else 1
            next_state = state + step
            reward = 0 if next_state != 6 else 1
            values[state] += alpha * (reward + values[next_state] - values[state])
            state = next_state
        total.append(np.copy(values))

    return (values, total)


def mc(episodes, alpha):
    values = np.ones((7)) * 0.5
    values[0] = 0
    values[6] = 0
    total = []
    for episode in range(episodes):
        state = 3
        history = [3]
        while (state != 0 and state != 6):
            step = -1 if not np.random.randint(2) else 1
            state = state + step
            reward = 1 if state == 6 else 0
            history.append(state)
        # print(history[-1],reward)
        for i in reversed(history):
            values[i] += alpha * (reward - values[i])
        total.append(np.copy(values))
    return (values, total)


def plot1():
    true_values = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    zeros = np.ones((5)) * 0.5
    td_one, _ = td0(1)
    td_10, _ = td0(10)
    td_100, _ = td0(100)
    xlabels = ['A', 'B', 'C', 'D', 'E']
    # data = pd.DataFrame(np.array([true_values,zeros,td_one,td_10,td_100]).reshape(5,5).T,columns=xlabels)
    # sns.lineplot(data=data)
    plt.plot(xlabels, true_values)
    plt.plot(xlabels, zeros)
    plt.plot(xlabels, td_one[1:-1])
    plt.plot(xlabels, td_10[1:-1])
    plt.plot(xlabels, td_100[1:-1])
    plt.legend(['True', '0', '1', '10', '100'], loc='upper left')
    plt.savefig('./q6lines.png')


def plot2():
    def error(arr):
        true_values = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        arr = arr[:, 1:-1]
        errors = []
        for i in range(100):
            a = arr[i, :]
            e = math.sqrt(np.mean(np.square(a - true_values)))
            errors.append(e)
        return (np.array(errors))

    xlabels = [i for i in range(100)]

    for alpha in [0.01, 0.02, 0.03, 0.04]:
        mct = np.zeros((100))
        for j in range(100):
            _, m = mc(100, alpha)
            mct += error(np.array(m).reshape(100, 7))/100
        # mct /= 100
        plt.plot(xlabels, mct.ravel())

    for alpha in [0.05, 0.1, 0.15]:
        tdt = np.zeros((100))
        for j in range(100):
            _, t = td0(100, alpha)
            tdt += error(np.array(t).reshape(100, 7))/100
        # tdt /= 100
        plt.plot(xlabels, tdt.ravel())
    plt.legend(['MC 0.01', 'MC 0.02', 'MC 0.03', 'MC 0.04', 'TD 0.05', 'TD 0.1', 'TD 0.15'], loc='upper left')
    plt.savefig('./q6-new1.png')


# plot1()
plot2()
