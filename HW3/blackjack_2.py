import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
# initial policy for player: player_sum only
policy = np.zeros((10))

# initially stayy on 20,21 only
policy[0:8] = 1

# player, dealer, ace, action
state_action_values = np.zeros((10, 10, 2, 2))
state_action_counts = np.ones((10, 10, 2, 2))


def initial_policy(states):
    state = np.copy(states)
    state[0] -= 12
    return (policy[state[0]])


def greedy_policy(states):
    state = np.copy(states)
    state[0] -= 12
    state[1] = 1 if state[1] == 11 else state[1]
    state[1] -= 1
    state[2] = 1 if state[2] else 0
    average_returns = state_action_values[state[0], state[1], state[2], :] / state_action_counts[state[0], state[1],
                                                                             state[2], :]
    # print(average_returns,np.where(average_returns==average_returns.max()))
    return np.random.choice(np.where(average_returns == average_returns.max())[0])


# simulate the random draw of a card from Ace to King from an infinite deck
def draw_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    if (card == 1):
        card = 11
    return card


def get_random_state_action():
    player_sum = np.random.randint(12, 22)
    ace = np.random.randint(2)
    dealer = np.random.randint(2, 12)
    action = np.random.randint(2)
    return ([player_sum, dealer, ace], action)


# simulate one episode of blackjack
def episode(greedy=True):
    state, initial_action = get_random_state_action()
    # sum of player cards
    total1 = state[0]

    # sum of dealer cards
    total2 = 0

    # number of aces counted as 11s held by player
    ace1 = state[2]

    # number of aces counted as 11s held by dealer
    ace2 = 0

    history = []

    # cards held by the dealer
    dealer = [state[1]]
    dealer.append(draw_card())
    total2 = sum(dealer)
    ace2 = dealer.count(11)

    if (total2 == 22):
        total2 -= 10
        ace2 -= 1
    first = True

    # player turn
    while (True):
        # print(first,total1,state[0])

        if (first):
            action = initial_action
            first = False
        elif (not greedy):
            action = initial_policy(state)
        else:
            action = greedy_policy(state)
        history.append([state, action])

        if (action):
            curr = draw_card()
            ace1 = ace1 + 1 if curr == 11 else ace1
            total1 += curr

            # check if player can use aces as 1s
            while ((total1 >= 22) and (ace1 > 0)):
                total1 -= 10
                ace1 -= 1

            # player has gone bust and does not have aces to use as 1s
            if total1 > 21:
                return (-1, history)
            state = [total1, state[1], ace1]

        else:
            break

    # dealer's turn
    while (True):

        # dealer will hit for all sums less than 17
        if (total2 >= 17):
            break

        # if hit, get a new card
        dcurr = draw_card()
        ace2 = ace2 + 1 if dcurr == 11 else ace2
        total2 += dcurr

        while ((total2 >= 22) and (ace2 > 0)):
            total2 -= 10
            ace2 -= 1

        # dealer has gone bust
        if total2 > 21:
            return (1, history)

    reward = 1
    if (total1 == total2):
        reward -= 1
    elif (total1 < total2):
        reward -= 2
    # print(history)
    return (reward, history)


def monte_carlo_es(episodes):
    global state_action_values
    global state_action_counts

    for ep in range(episodes):
        # for each episode, use a randomly initialized state and action
        reward, history = episode(ep != 0)
        for (player_sum, dealer_card, usable_ace), action in history:
            # print(player_sum, dealer_card, usable_ace)
            usable_ace = 1 if usable_ace else 0
            player_sum -= 12
            if (dealer_card == 11):
                dealer_card = 1
            dealer_card -= 1
            action = int(action)
            # update values of state-action pairs
            state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            state_action_counts[player_sum, dealer_card, usable_ace, action] += 1

    return state_action_values / state_action_counts


def plot():
    v = monte_carlo_es(2000000)

    state1 = np.max(v[:, :, 0, :], axis=-1)[::-1, ...]
    state2 = np.max(v[:, :, 1, :], axis=-1)[::-1, ...]
    action1 = np.argmax(v[:, :, 0, :], axis=-1)[::-1, ...]
    action2 = np.argmax(v[:, :, 1, :], axis=-1)[::-1, ...]

    maps = [action2, state2, action1, state1]

    labels = ['Policy with ace', 'Value with ace', 'Policy without ace', 'Value without Ace']

    _, axes = plt.subplots(2, 2, figsize=(80, 60))
    xt = [i for i in range(1, 11)]
    yt = [i for i in range(21, 11, -1)]
    import seaborn as sns
    for m, l, a in zip(maps, labels, axes.flatten()):
        fig = sns.heatmap(m, ax=a, xticklabels=xt, yticklabels=yt)
        fig.set_ylabel('player')
        fig.set_xlabel('dealer')
        fig.set_title(l)

    plt.savefig('./figure_5_2_new.png')
    plt.close()


plot()
