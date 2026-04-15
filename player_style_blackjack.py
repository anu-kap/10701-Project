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

        p_total, _ = self._hand_value(self.player_hand)
        d_total, _ = self._hand_value(self.dealer_hand)
        self.player_natural = (p_total == 21)
        self.dealer_natural = (d_total == 21)

        return self._get_state()

    def _get_state(self):
        total, usable_ace = self._hand_value(self.player_hand)
        return (total, self.dealer_hand[0], usable_ace)

    def step(self, action):
        if self.player_natural:
            if self.dealer_natural:
                return self._get_state(), 0, True   
            else:
                return self._get_state(), 1.5, True  

        if action == 1: 
            self.player_hand.append(self._draw())
            total, _ = self._hand_value(self.player_hand)
            if total > 21:
                return self._get_state(), -1, True
            return self._get_state(), 0, False

        else:  
            if self.dealer_natural:
                return self._get_state(), -1, True

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
    HARD_HIT_UPCARDS = {
        9:  [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        10: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        11: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        12: [2, 3, 7, 8, 9, 10, 1],
        13: [7, 8, 9, 10, 1],
        14: [7, 8, 9, 10, 1],
        15: [7, 8, 9, 10, 1],
        16: [7, 8, 9, 10, 1],
    }
    SOFT_HIT_UPCARDS = {
        13: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        14: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        15: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        16: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        17: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        18: [9, 10, 1],
    }

    def act(self, state):
        total, dealer_upcard, usable_ace = state
        if total >= 21:
            return 0
        if not usable_ace and total >= 17:
            return 0
        if total <= 8:
            return 1
        if usable_ace:
            if total >= 19:
                return 0
            return 1 if dealer_upcard in self.SOFT_HIT_UPCARDS.get(total, []) else 0
        return 1 if dealer_upcard in self.HARD_HIT_UPCARDS.get(total, []) else 0


PERSONAS = {
    0: {"name": "very conservative", "theta": 0.1},
    1: {"name": "conservative",      "theta": 0.3},
    2: {"name": "neutral",           "theta": 0.5},
    3: {"name": "aggressive",        "theta": 0.7},
    4: {"name": "very aggressive",   "theta": 0.9},
}
NUM_PERSONAS = len(PERSONAS)


def persona_action(state, theta, epsilon=0.0):
    total, dealer_upcard, usable_ace = state
    if total <= 8:
        return 1
    if not usable_ace and total >= 17:
        return 0
    if usable_ace and total >= 19:
        return 0
    hit_prob = theta
    if epsilon > 0:
        hit_prob = (1 - epsilon) * hit_prob + epsilon * 0.5
    return 1 if np.random.random() < hit_prob else 0


class ThetaQLearningAgent:

    def __init__(
        self,
        alpha=0.1,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(float)

    def _best_action(self, state, persona_id):
        q0 = self.Q[(state, persona_id, 0)]
        q1 = self.Q[(state, persona_id, 1)]
        return 1 if q1 > q0 else 0

    def act_training(self, state, persona_id, theta):
        if np.random.random() < self.epsilon:
            return persona_action(state, theta, epsilon=1.0)
        return self._best_action(state, persona_id)

    def act_inference(self, state, persona_id):
        return self._best_action(state, persona_id)

    def update(self, state, persona_id, action, reward, next_state, done):
        key = (state, persona_id, action)
        if done:
            target = reward
        else:
            best_next = max(
                self.Q[(next_state, persona_id, 0)],
                self.Q[(next_state, persona_id, 1)],
            )
            target = reward + self.gamma * best_next
        self.Q[key] += self.alpha * (target - self.Q[key])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def policy(self, state, persona_id):
        return self._best_action(state, persona_id)


def train(num_episodes=500_000):
    env   = BlackjackEnv(num_decks=6)
    agent = ThetaQLearningAgent()

    win_log = []
    wins = 0
    window = 10_000

    for ep in range(1, num_episodes + 1):
        persona_id = np.random.randint(NUM_PERSONAS)
        theta      = PERSONAS[persona_id]["theta"]

        state = env.reset()
        done  = False

        while not done:
            action = agent.act_training(state, persona_id, theta)
            next_state, reward, done = env.step(action)
            agent.update(state, persona_id, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()

        if reward > 0:
            wins += 1
        if ep % window == 0:
            win_rate = wins / window
            win_log.append((ep, win_rate))
            print(f"Episode {ep:>7} | win rate (last {window}): {win_rate:.3f}"
                  f" | epsilon: {agent.epsilon:.4f}")
            wins = 0

    return agent, win_log


def _eval(env, policy_fn, num_episodes):
    """
    Shared evaluation loop.
    House edge = -E[reward] since reward is from player's perspective.
    Using total_reward correctly handles the 1.5x natural payout.
    """
    wins = losses = pushes = 0
    total_reward = 0.0

    for _ in range(num_episodes):
        state = env.reset()
        done  = False
        reward = 0
        while not done:
            action = policy_fn(state)
            state, reward, done = env.step(action)
        if reward > 0:     wins += 1
        elif reward < 0:   losses += 1
        else:              pushes += 1
        total_reward += reward

    total = num_episodes
    return {
        "win_rate":   wins        / total,
        "loss_rate":  losses      / total,
        "push_rate":  pushes      / total,
        "house_edge": -total_reward / total, 
    }


def evaluate_persona(agent, persona_id, num_episodes=100_000):
    env = BlackjackEnv(num_decks=6)
    return _eval(env, lambda s: agent.act_inference(s, persona_id), num_episodes)


def evaluate_basic_strategy(num_episodes=100_000):
    env = BlackjackEnv(num_decks=6)
    bs  = BasicStrategy()
    return _eval(env, bs.act, num_episodes)


def evaluate_random(num_episodes=100_000):
    env = BlackjackEnv(num_decks=6)
    return _eval(env, lambda s: np.random.randint(2), num_episodes)


def compute_agreement(agent, persona_id):
    bs = BasicStrategy()
    states = [
        (total, upcard, usable_ace)
        for total      in range(4, 22)
        for upcard     in range(1, 11)
        for usable_ace in [False, True]
    ]
    agree = sum(agent.policy(s, persona_id) == bs.act(s) for s in states)
    return agree / len(states)


def plot_learning_curve(win_log, save_path="learning_curve.png"):
    episodes, rates = zip(*win_log)
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rates)
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title("Theta-conditioned Q-learning — win rate over training")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_policy_heatmap(agent, save_path="policy_heatmap.png"):
    """All personas side by side + basic strategy. Fixed deck: 6."""
    bs      = BasicStrategy()
    totals  = list(range(8, 21))
    upcards = list(range(2, 11)) + [1]

    fig, axes = plt.subplots(1, NUM_PERSONAS + 1, figsize=(4 * (NUM_PERSONAS + 1), 5))
    cmap = mcolors.ListedColormap(["#d9534f", "#5cb85c"])

    for pid in range(NUM_PERSONAS):
        grid = np.zeros((len(totals), len(upcards)))
        for i, total in enumerate(totals):
            for j, upcard in enumerate(upcards):
                grid[i, j] = agent.policy((total, upcard, False), pid)
        axes[pid].imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        name  = PERSONAS[pid]["name"]
        theta = PERSONAS[pid]["theta"]
        axes[pid].set_title(f"{name}\n(θ={theta})", fontsize=9)
        axes[pid].set_xticks(range(len(upcards)))
        axes[pid].set_xticklabels([str(u) if u != 1 else "A" for u in upcards], fontsize=7)
        axes[pid].set_yticks(range(len(totals)))
        axes[pid].set_yticklabels(totals, fontsize=7)
        axes[pid].set_xlabel("Dealer upcard", fontsize=8)
        if pid == 0:
            axes[pid].set_ylabel("Player total (hard)", fontsize=8)

    bs_grid = np.zeros((len(totals), len(upcards)))
    for i, total in enumerate(totals):
        for j, upcard in enumerate(upcards):
            bs_grid[i, j] = bs.act((total, upcard, False))
    im = axes[-1].imshow(bs_grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[-1].set_title("Basic Strategy\n(ground truth)", fontsize=9)
    axes[-1].set_xticks(range(len(upcards)))
    axes[-1].set_xticklabels([str(u) if u != 1 else "A" for u in upcards], fontsize=7)
    axes[-1].set_yticks(range(len(totals)))
    axes[-1].set_yticklabels(totals, fontsize=7)
    axes[-1].set_xlabel("Dealer upcard", fontsize=8)

    fig.colorbar(im, ax=axes.tolist(), ticks=[0, 1], label="0 = Stand  |  1 = Hit")
    plt.suptitle("Learned policy per persona vs basic strategy (6-deck)", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_theta_vs_house_edge(results_by_persona, results_bs,
                              save_path="theta_vs_edge.png"):
    thetas = [PERSONAS[pid]["theta"] for pid in range(NUM_PERSONAS)]
    edges  = [results_by_persona[pid]["house_edge"] * 100 for pid in range(NUM_PERSONAS)]

    plt.figure(figsize=(7, 4))
    plt.plot(thetas, edges, marker='o', label="Q-agent")
    plt.axhline(results_bs["house_edge"] * 100, color='black',
                linestyle='--', linewidth=0.9, label="Basic Strategy")
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.xlabel("Theta (player aggression)")
    plt.ylabel("House edge (%)")
    plt.title("House edge vs player aggression (theta)")
    plt.xticks(thetas, [f"{t}\n({PERSONAS[i]['name']})"
                        for i, t in enumerate(thetas)], fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("Training theta-conditioned Q-learning agent...")
    print("  5 personas sampled uniformly each episode")
    print("=" * 55)
    agent, win_log = train(num_episodes=500_000)

    print("\nEvaluating (100k episodes each)...")
    results_by_persona = {pid: evaluate_persona(agent, pid) for pid in range(NUM_PERSONAS)}
    results_bs = evaluate_basic_strategy()
    results_rn = evaluate_random()

    print("\n--- Results by persona ---")
    print(f"{'Agent':<30} {'Win':>6} {'Loss':>6} {'Push':>6} {'House Edge':>11}")
    print("-" * 64)
    print(f"{'Random baseline':<30} {results_rn['win_rate']:>6.3f} "
          f"{results_rn['loss_rate']:>6.3f} {results_rn['push_rate']:>6.3f}   {'N/A':>8}")
    print(f"{'Basic Strategy':<30} {results_bs['win_rate']:>6.3f} "
          f"{results_bs['loss_rate']:>6.3f} {results_bs['push_rate']:>6.3f}  "
          f"{results_bs['house_edge']*100:>+9.2f}%")
    for pid in range(NUM_PERSONAS):
        res   = results_by_persona[pid]
        label = f"Q-agent θ={PERSONAS[pid]['theta']} ({PERSONAS[pid]['name']})"
        print(f"{label:<30} {res['win_rate']:>6.3f} {res['loss_rate']:>6.3f} "
              f"{res['push_rate']:>6.3f}  {res['house_edge']*100:>+9.2f}%")

    print("\n--- Policy agreement with Basic Strategy per persona ---")
    for pid in range(NUM_PERSONAS):
        agr = compute_agreement(agent, pid)
        print(f"  θ={PERSONAS[pid]['theta']} ({PERSONAS[pid]['name']}): {agr:.3f}")

    print("\nGenerating plots...")
    plot_learning_curve(win_log)
    plot_policy_heatmap(agent)
    plot_theta_vs_house_edge(results_by_persona, results_bs)

    print("\nDone.")