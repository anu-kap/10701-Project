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


def persona_action(state, theta):
    total, dealer_upcard, usable_ace = state
    if total <= 8:
        return 1
    if not usable_ace and total >= 17:
        return 0
    if usable_ace and total >= 19:
        return 0
    return 1 if np.random.random() < theta else 0


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
            return persona_action(state, theta)
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


def train(num_episodes=500_000, num_decks=6, verbose=True):
    env   = BlackjackEnv(num_decks=num_decks)
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
            if verbose:
                print(f"  ep {ep:>7} | win rate: {win_rate:.3f}"
                      f" | ε: {agent.epsilon:.4f}")
            wins = 0

    return agent, win_log


def evaluate_persona(agent, persona_id, num_decks=6, num_episodes=100_000):
    env  = BlackjackEnv(num_decks=num_decks)
    wins = losses = pushes = 0
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done  = False
        reward = 0
        while not done:
            action = agent.act_inference(state, persona_id)
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


def evaluate_basic_strategy(num_decks=6, num_episodes=100_000):
    env = BlackjackEnv(num_decks=num_decks)
    bs  = BasicStrategy()
    wins = losses = pushes = 0
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done  = False
        reward = 0
        while not done:
            action = bs.act(state)
            state, reward, done = env.step(action)
        if reward > 0:   wins += 1
        elif reward < 0: losses += 1
        else:            pushes += 1
        total_reward += reward
    total = num_episodes
    return {
        "win_rate":   wins        / total,
        "loss_rate":  losses      / total,
        "push_rate":  pushes      / total,
        "house_edge": -total_reward / total,
    }


def evaluate_random(num_decks=6, num_episodes=100_000):
    env = BlackjackEnv(num_decks=num_decks)
    wins = losses = pushes = 0
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done  = False
        reward = 0
        while not done:
            state, reward, done = env.step(np.random.randint(2))
        if reward > 0:   wins += 1
        elif reward < 0: losses += 1
        else:            pushes += 1
        total_reward += reward
    total = num_episodes
    return {
        "win_rate":   wins        / total,
        "loss_rate":  losses      / total,
        "push_rate":  pushes      / total,
        "house_edge": -total_reward / total,
    }


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


def plot_learning_curves(win_logs, save_path="learning_curves_multideck.png"):
    plt.figure(figsize=(10, 5))
    for nd, win_log in win_logs.items():
        episodes, rates = zip(*win_log)
        plt.plot(episodes, rates, label=f"{nd} deck{'s' if nd > 1 else ''}")
    plt.xlabel("Episode")
    plt.ylabel("Win rate (per 10k window)")
    plt.title("Learning curves by deck configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_house_edge_by_decks(results, save_path="house_edge_by_decks.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for pid in range(NUM_PERSONAS):
        edges = [results[nd][pid]["house_edge"] * 100 for nd in DECK_CONFIGS]
        theta = PERSONAS[pid]["theta"]
        name  = PERSONAS[pid]["name"]
        ax.plot(DECK_CONFIGS, edges, marker='o', label=f"θ={theta} ({name})")
    bs_edges = [results[nd]["basic"]["house_edge"] * 100 for nd in DECK_CONFIGS]
    ax.plot(DECK_CONFIGS, bs_edges, marker='s', linestyle='--',
            color='black', label="Basic Strategy")
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel("Number of decks")
    ax.set_ylabel("House edge (%)")
    ax.set_title("House edge vs deck count by player persona")
    ax.set_xticks(DECK_CONFIGS)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_convergence_speed(win_logs, save_path="convergence_by_decks.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    convergence_eps = {}
    for nd, win_log in win_logs.items():
        episodes, rates = zip(*win_log)
        rates = np.array(rates)
        final = rates[-1]
        converged_at = episodes[-1]
        for i in range(len(rates) - 5):
            if np.all(np.abs(rates[i:] - final) < 0.01):
                converged_at = episodes[i]
                break
        convergence_eps[nd] = converged_at
    ax.bar(
        [str(nd) for nd in DECK_CONFIGS],
        [convergence_eps[nd] for nd in DECK_CONFIGS],
        color='steelblue', edgecolor='white'
    )
    ax.set_xlabel("Number of decks")
    ax.set_ylabel("Episodes to convergence")
    ax.set_title("Convergence speed by deck configuration")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_policy_heatmaps_by_deck(agents_by_deck, persona_id=2,
                                  save_path="policy_heatmaps_by_deck.png"):
    bs      = BasicStrategy()
    totals  = list(range(8, 21))
    upcards = list(range(2, 11)) + [1]
    n_cols  = len(DECK_CONFIGS) + 1

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    cmap = mcolors.ListedColormap(["#d9534f", "#5cb85c"])

    for col, nd in enumerate(DECK_CONFIGS):
        agent = agents_by_deck[nd]
        grid  = np.zeros((len(totals), len(upcards)))
        for i, total in enumerate(totals):
            for j, upcard in enumerate(upcards):
                grid[i, j] = agent.policy((total, upcard, False), persona_id)
        axes[col].imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axes[col].set_title(f"{nd} deck{'s' if nd > 1 else ''}", fontsize=9)
        axes[col].set_xticks(range(len(upcards)))
        axes[col].set_xticklabels(
            [str(u) if u != 1 else "A" for u in upcards], fontsize=7)
        axes[col].set_yticks(range(len(totals)))
        axes[col].set_yticklabels(totals, fontsize=7)
        axes[col].set_xlabel("Dealer upcard", fontsize=8)
        if col == 0:
            axes[col].set_ylabel("Player total (hard)", fontsize=8)

    bs_grid = np.zeros((len(totals), len(upcards)))
    for i, total in enumerate(totals):
        for j, upcard in enumerate(upcards):
            bs_grid[i, j] = bs.act((total, upcard, False))
    im = axes[-1].imshow(bs_grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[-1].set_title("Basic Strategy\n(ground truth)", fontsize=9)
    axes[-1].set_xticks(range(len(upcards)))
    axes[-1].set_xticklabels(
        [str(u) if u != 1 else "A" for u in upcards], fontsize=7)
    axes[-1].set_yticks(range(len(totals)))
    axes[-1].set_yticklabels(totals, fontsize=7)
    axes[-1].set_xlabel("Dealer upcard", fontsize=8)

    fig.colorbar(im, ax=axes.tolist(), ticks=[0, 1], label="0 = Stand  |  1 = Hit")
    name = PERSONAS[persona_id]["name"]
    plt.suptitle(
        f"Learned policy by deck count — {name} persona (θ={PERSONAS[persona_id]['theta']})",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def print_results_table(results):
    print(f"\n{'─'*82}")
    print(f"  {'Config':<10} {'Agent':<28} {'Win':>6} {'Loss':>6} "
          f"{'Push':>6} {'House Edge':>11}")
    print(f"{'─'*82}")
    for nd in DECK_CONFIGS:
        r = results[nd]["random"]
        print(f"  {str(nd)+' deck':<10} {'Random baseline':<28} "
              f"{r['win_rate']:>6.3f} {r['loss_rate']:>6.3f} "
              f"{r['push_rate']:>6.3f}   {'N/A':>8}")
        r = results[nd]["basic"]
        print(f"  {'':<10} {'Basic Strategy':<28} "
              f"{r['win_rate']:>6.3f} {r['loss_rate']:>6.3f} "
              f"{r['push_rate']:>6.3f}  {r['house_edge']*100:>+9.2f}%")
        for pid in range(NUM_PERSONAS):
            r     = results[nd][pid]
            name  = PERSONAS[pid]["name"]
            theta = PERSONAS[pid]["theta"]
            label = f"θ={theta} ({name})"
            print(f"  {'':<10} {label:<28} "
                  f"{r['win_rate']:>6.3f} {r['loss_rate']:>6.3f} "
                  f"{r['push_rate']:>6.3f}  {r['house_edge']*100:>+9.2f}%")
        print(f"  {'─'*80}")


DECK_CONFIGS = [1, 2, 4, 6, 8]

if __name__ == "__main__":
    NUM_EPISODES  = 500_000
    EVAL_EPISODES = 100_000

    results        = {}
    win_logs       = {}
    agents_by_deck = {}

    for nd in DECK_CONFIGS:
        print(f"\n{'='*55}")
        print(f"  Training on {nd}-deck shoe ({NUM_EPISODES:,} episodes)")
        print(f"{'='*55}")

        agent, win_log = train(num_episodes=NUM_EPISODES, num_decks=nd, verbose=True)
        agents_by_deck[nd] = agent
        win_logs[nd]       = win_log

        print(f"\n  Evaluating...")
        results[nd] = {}
        for pid in range(NUM_PERSONAS):
            results[nd][pid] = evaluate_persona(
                agent, pid, num_decks=nd, num_episodes=EVAL_EPISODES
            )
        results[nd]["basic"]     = evaluate_basic_strategy(nd, EVAL_EPISODES)
        results[nd]["random"]    = evaluate_random(nd, EVAL_EPISODES)
        results[nd]["agreement"] = compute_agreement(agent, persona_id=2)
        print(f"  Basic strategy agreement (neutral): "
              f"{results[nd]['agreement']:.3f}")

    print_results_table(results)

    print("\nGenerating plots...")
    plot_learning_curves(win_logs)
    plot_house_edge_by_decks(results)
    plot_convergence_speed(win_logs)
    plot_policy_heatmaps_by_deck(agents_by_deck, persona_id=2)

    print("\nDone.")