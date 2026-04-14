import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict


class BlackjackEnv:

    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.deck = self._build_deck()
        self.reset()

    def _build_deck(self):
        single = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        deck = single * self.num_decks
        np.random.shuffle(deck)
        return deck

    def _draw(self):
        if len(self.deck) < 15:
            self.deck = self._build_deck()
        return self.deck.pop()


    @staticmethod
    def _hand_value(hand):
        total = sum(hand)
        usable_ace = False
        if 1 in hand and total + 10 <= 21:
            total += 10
            usable_ace = True
        return total, usable_ace

    def reset(self):
        self.player_hand = [self._draw(), self._draw()]
        self.dealer_hand = [self._draw(), self._draw()]
        return self._get_state()

    def _get_state(self):
        total, usable_ace = self._hand_value(self.player_hand)
        dealer_upcard = self.dealer_hand[0]
        return (total, dealer_upcard, usable_ace)

    def step(self, action):
        if action == 1:
            self.player_hand.append(self._draw())
            total, usable_ace = self._hand_value(self.player_hand)
            if total > 21:
                return self._get_state(), -1, True
            return self._get_state(), 0, False

        else:
            dealer_total, _ = self._hand_value(self.dealer_hand)
            while dealer_total < 17:
                self.dealer_hand.append(self._draw())
                dealer_total, _ = self._hand_value(self.dealer_hand)

            player_total, _ = self._hand_value(self.player_hand)
            if dealer_total > 21 or player_total > dealer_total:
                reward = 1
            elif player_total == dealer_total:
                reward = 0
            else:
                reward = -1

            return self._get_state(), reward, True


class BasicStrategy:
    _HARD = {
        17: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        16: [2, 3, 4, 5, 6],
        15: [2, 3, 4, 5, 6],
        14: [2, 3, 4, 5, 6],
        13: [2, 3, 4, 5, 6],
        12: [4, 5, 6],
    }

    _SOFT = {
        20: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        19: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        18: [2, 3, 4, 5, 6, 7, 8],
    }

    def act(self, state):
        total, dealer_upcard, usable_ace = state

        if total >= 21:
            return 0

        if usable_ace:
            stand_upcards = self._SOFT.get(total, [])
        else:
            stand_upcards = self._HARD.get(total, [])

        if total >= 17 and not stand_upcards:
            return 0
        if total <= 11:
            return 1

        return 0 if dealer_upcard in stand_upcards else 1


class QLearningAgent:

    def __init__(
        self,
        alpha=0.1,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9999,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(float)

    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        return self._greedy_action(state)

    def _greedy_action(self, state):
        q_hit   = self.Q[(state, 1)]
        q_stand = self.Q[(state, 0)]
        return 1 if q_hit > q_stand else 0

    def update(self, state, action, reward, next_state, done):
        """Bellman update."""
        if done:
            target = reward
        else:
            best_next = max(self.Q[(next_state, 0)], self.Q[(next_state, 1)])
            target = reward + self.gamma * best_next

        self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def policy(self, state):
        """Return greedy action (no exploration) — used at eval time."""
        return self._greedy_action(state)


def train(num_episodes=500_000):
    env   = BlackjackEnv(num_decks=6)
    agent = QLearningAgent()

    win_log = []

    wins = 0
    window = 10_000

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done  = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        if reward == 1:
            wins += 1

        if ep % window == 0:
            win_rate = wins / window
            win_log.append((ep, win_rate))
            print(f"Episode {ep:>7} | win rate (last {window}): {win_rate:.3f}"
                  f" | epsilon: {agent.epsilon:.4f}")
            wins = 0

    return agent, win_log

def evaluate(agent_or_strategy, num_episodes=100_000):
    env = BlackjackEnv(num_decks=6)
    wins = losses = pushes = 0

    for _ in range(num_episodes):
        state = env.reset()
        done  = False
        reward = 0

        while not done:
            action = agent_or_strategy.act(state) if hasattr(agent_or_strategy, 'act') \
                     else agent_or_strategy.policy(state)
            action = (agent_or_strategy.policy(state)
                      if isinstance(agent_or_strategy, QLearningAgent)
                      else agent_or_strategy.act(state))
            state, reward, done = env.step(action)

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            pushes += 1

    total = num_episodes
    return {
        "win_rate":  wins   / total,
        "loss_rate": losses / total,
        "push_rate": pushes / total,
    }


def random_agent_act(state):
    return np.random.randint(2)


def evaluate_random(num_episodes=100_000):
    env = BlackjackEnv(num_decks=6)
    wins = losses = pushes = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward = 0
        while not done:
            state, reward, done = env.step(np.random.randint(2))
        if reward == 1: wins += 1
        elif reward == -1: losses += 1
        else: pushes += 1
    total = num_episodes
    return {"win_rate": wins/total, "loss_rate": losses/total, "push_rate": pushes/total}

def compute_agreement(agent):
    bs = BasicStrategy()
    states = [
        (total, upcard, usable_ace)
        for total      in range(4, 22)
        for upcard     in range(1, 11)
        for usable_ace in [False, True]
    ]

    agree = sum(
        agent.policy(s) == bs.act(s)
        for s in states
    )
    return agree / len(states)


def plot_learning_curve(win_log, save_path="learning_curve.png"):
    episodes, rates = zip(*win_log)
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rates)
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title("Q-Learning agent win rate over training")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved learning curve to {save_path}")


def plot_policy_heatmap(agent, save_path="policy_heatmap.png"):
    bs = BasicStrategy()
    totals  = list(range(8, 21))
    upcards = list(range(2, 11)) + [1]   # 1 = Ace displayed last

    agent_grid = np.zeros((len(totals), len(upcards)))
    bs_grid    = np.zeros((len(totals), len(upcards)))

    for i, total in enumerate(totals):
        for j, upcard in enumerate(upcards):
            state = (total, upcard, False)
            agent_grid[i, j] = agent.policy(state)
            bs_grid[i, j]    = bs.act(state)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = mcolors.ListedColormap(["#d9534f", "#5cb85c"])   # red=stand, green=hit

    for ax, grid, title in zip(
        axes,
        [agent_grid, bs_grid],
        ["Q-Learning agent policy", "Basic Strategy (ground truth)"]
    ):
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(upcards)))
        ax.set_xticklabels([str(u) if u != 1 else "A" for u in upcards])
        ax.set_yticks(range(len(totals)))
        ax.set_yticklabels(totals)
        ax.set_xlabel("Dealer upcard")
        ax.set_ylabel("Player total (hard)")
        ax.set_title(title)

    fig.colorbar(im, ax=axes, ticks=[0, 1], label="0 = Stand  |  1 = Hit")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved policy heatmap to {save_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("Training Q-learning agent (500,000 episodes)...")
    print("=" * 55)
    agent, win_log = train(num_episodes=500_000)

    print("\nEvaluating agents (100,000 episodes each)...")
    results_q  = evaluate(agent,        num_episodes=100_000)
    results_bs = evaluate(BasicStrategy(), num_episodes=100_000)
    results_rn = evaluate_random(        num_episodes=100_000)

    print("\n--- Results (Table 1) ---")
    print(f"{'Agent':<20} {'Win':>6} {'Loss':>6} {'Push':>6}")
    print("-" * 42)
    for name, res in [
        ("Random baseline", results_rn),
        ("Basic Strategy",  results_bs),
        ("Q-Learning bot",  results_q),
    ]:
        print(f"{name:<20} {res['win_rate']:>6.3f} {res['loss_rate']:>6.3f} {res['push_rate']:>6.3f}")

    agreement = compute_agreement(agent)
    print(f"\nPolicy agreement with Basic Strategy: {agreement:.3f}")

    print("\nGenerating plots...")
    plot_learning_curve(win_log)
    plot_policy_heatmap(agent)

    print("\nDone.")