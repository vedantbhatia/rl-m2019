import numpy as np
import matplotlib.pyplot as plt

# target policy for player: player_sum only
policy = np.ones((10))

# initially stay on 20,21 only
policy[8:10] = 0


# simulate the random draw of a card from Ace to King from an infinite deck
def draw_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    if (card == 1):
        card = 11
    return card


# simulate one episode of blackjack
def episode():
    # given state player sum 13, dealer card 2, with usable ace
    state = [13, 2, 1]

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

    # player turn
    while (True):
        # print(first,total1,state[0])

        # behaviour policy is random w equal prob
        action = np.random.randint(2)
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


def monte_carlo(episodes):

    ratios = []
    rewards = []

    for ep in range(episodes):
        # for each episode, use a randomly initialized state and action
        reward, history = episode()
        n = 1.0
        d = 1.0
        for state, action in history:
            state[0]-=12
            # print(state[0],policy[state[0]])
            target_action = policy[state[0]]
            if(action==target_action):
                d=d*0.5
            else:
                n = 0
                # print("end")
                break
        ratios.append(n/d)
        rewards.append(reward)
    numerator = [a*b for a,b in zip(ratios,rewards)]
    all_iters=[]
    d_weighted=[]
    for i in range(len(numerator)):
        all_iters.append(all_iters[i-1]+numerator[i] if (i) else numerator[i])
        d_weighted.append(d_weighted[i-1]+ratios[i] if (i) else ratios[i])

    d_ord = [i for i in range(1,len(all_iters)+1)]
    sampling1 = np.array(all_iters)/np.array(d_ord)
    sampling2 = np.array([a/b if b else 0 for a,b in zip(all_iters,d_weighted)])
    return(sampling1,sampling2)



def plot():

    e = np.zeros((2,10000))

    for i in range(0, 100):
        s1, s2 = monte_carlo(10000)
        e[0,:] += (s1 - -0.27726)**2
        e[1,:] += (s2 - -0.27726)**2


    plt.plot(e[0]/100)
    plt.plot(e[1]/100)
    plt.xlabel('Episodes')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.legend(['Ordinary Importance Sampling','Weighted Importance Sampling'])
    plt.savefig('./figure_5_3_new1.png')
    plt.close()



plot()
