import numpy as np
import matplotlib.pyplot as plt


# simulate the random draw of a card from Ace to King from an infinite deck
def draw_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    if(card==1):
        card=11
    return card




# simulate one episode of blackjack
def episode():
    # sum of player cards
    total1 = 0

    # sum of dealer cards
    total2 = 0

    # number of aces counted as 11s held by player
    ace1 = 0

    # number of aces counted as 11s held by dealer
    ace2 = 0

    # cards held by the dealer
    dealer = []

    # the states in the players episode
    history = []

    # the player makes an action according to policy only if sum is greater than 12 otherwise it is always rational to hit
    while total1 < 12:
        c = draw_card()
        if(c == 11):
            ace1 += 1
        total1 += c

    # use an ace as a 1 if he's bust
    if total1 > 21:
        total1 -= 10
        ace1 -= 1

    # dealer draws two cards, one is visible to the player
    dealer.append(draw_card())
    dealer.append(draw_card())
    total2 = sum(dealer)
    ace2 = dealer.count(11)

    if total2 > 21:
        total2 -= 10
        ace2 -= 1

    # player turn
    while(True):
        # state is player's sum, face up dealer card, and usable aces of player
        history.append([total1, dealer[0], ace1])

        # player will hit unless his sum is 20 or 21
        if(total1==20 or total1==21):
            break

        curr = draw_card()
        ace1= ace1+1 if curr==11 else ace1
        total1 += curr

        # check if player can use aces as 1s
        while((total1 >= 22) and (ace1>0)):
            total1 -= 10
            ace1 -= 1

        # player has gone bust and does not have aces to use as 1s
        if total1 > 21:
            return(-1,history)

    # dealer's turn
    while(True):

        # dealer will hit for all sums less than 17
        if(total2>=17):
            break

        # if hit, get a new card
        curr = draw_card()
        ace2 = ace2 + 1 if curr == 11 else ace2
        total2 += curr

        while((total2 >= 22) and (ace2>0)):
            total2 -= 10
            ace2 -= 1

        # dealer has gone bust
        if total2 > 21:
            return(1,history)

    reward = 1
    if(total1==total2):
        reward-=1
    elif(total1 < total2):
        reward-=2
    return(reward,history)


def mc(episodes):
    # state values and visit counts
    state_values = np.zeros((2,10, 10))
    state_counts = np.ones((2,10, 10))

    for i in range(episodes):
        reward,history = episode()
        # always add reward as reward is only received on termination
        for state in history:
            state[0] -= 12
            state[1] = 1 if(state[1]==11) else state[1]
            state[1] -= 1
            state[2] = 1 if state[2]>0 else 0
            state_counts[state[2],state[0], state[1]] += 1
            state_values[state[2],state[0], state[1]] += reward
    return(state_values/state_counts)

def plot(estimates1,estimates2):

    states = [estimates1[1][::-1,...],estimates2[1][::-1,...],estimates1[0][::-1,...],estimates2[0][::-1,...]]
    labels = ['Ace, 10000', 'Ace, 500000', 'No Ace, 10000', 'No Ace, 500000']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    import seaborn as sns
    xt = [i for i in range(1,11)]
    yt = [i for i in range(21,11,-1)]
    for m, l, a in zip(states, labels, axes.flatten()):
        fig = sns.heatmap(m, ax=a, xticklabels=xt, yticklabels=yt)
        fig.set_ylabel('player')
        fig.set_xlabel('dealer')
        fig.set_title(l)


    plt.savefig('./figure_5_1_new.png')
    plt.close()


estimates1 = mc(10000)
estimates2 = mc(500000)
plot(estimates1,estimates2)