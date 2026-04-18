import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim


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
        self.player_hand  = [self._draw(), self._draw()]
        self.dealer_hand  = [self._draw(), self._draw()]
        self.is_first_action = True  

        p_total, _ = self._hand_value(self.player_hand)
        d_total, _ = self._hand_value(self.dealer_hand)
        self.player_natural = (p_total == 21)
        self.dealer_natural = (d_total == 21)
        return self._get_state()

    def _get_state(self):
        total, usable_ace = self._hand_value(self.player_hand)
        return (total, self.dealer_hand[0], usable_ace, self.is_first_action)

    def step(self, action):
        """
        Actions:
            0 = Stand
            1 = Hit
            2 = Double Down  

        If agent picks double when not on first action, treat as hit
        to avoid illegal moves crashing training.
        """
        if self.player_natural:
            self.is_first_action = False
            if self.dealer_natural:
                return self._get_state(), 0, True
            return self._get_state(), 1.5, True

        # Double Down
        if action == 2:
            if not self.is_first_action:
                action = 1   
            else:
                self.is_first_action = False
                self.player_hand.append(self._draw())
                p_total, _ = self._hand_value(self.player_hand)

                if p_total > 21:
                    return self._get_state(), -2.0, True  

                if self.dealer_natural:
                    return self._get_state(), -2.0, True
                dealer_total, _ = self._hand_value(self.dealer_hand)
                while dealer_total < 17:
                    self.dealer_hand.append(self._draw())
                    dealer_total, _ = self._hand_value(self.dealer_hand)
                reward = self._compare(p_total, dealer_total, multiplier=2.0)
                return self._get_state(), reward, True

        # Hit
        if action == 1:
            self.is_first_action = False
            self.player_hand.append(self._draw())
            total, _ = self._hand_value(self.player_hand)
            if total > 21:
                return self._get_state(), -1, True
            return self._get_state(), 0, False

        # Stand
        if action == 0:
            self.is_first_action = False
            if self.dealer_natural:
                return self._get_state(), -1, True
            dealer_total, _ = self._hand_value(self.dealer_hand)
            while dealer_total < 17:
                self.dealer_hand.append(self._draw())
                dealer_total, _ = self._hand_value(self.dealer_hand)
            p_total, _ = self._hand_value(self.player_hand)
            reward = self._compare(p_total, dealer_total, multiplier=1.0)
            return self._get_state(), reward, True

    @staticmethod
    def _compare(p_total, d_total, multiplier):
        if d_total > 21 or p_total > d_total:
            return multiplier
        elif p_total == d_total:
            return 0.0
        else:
            return -multiplier


class BasicStrategy:
    """
    Hit/stand/double basic strategy for 6-deck, dealer stands soft 17.
    action: 0=stand, 1=hit, 2=double
    """

    HARD_DOUBLE = {
        9:  [3, 4, 5, 6],
        10: [2, 3, 4, 5, 6, 7, 8, 9],
        11: [2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    SOFT_DOUBLE = {
        13: [5, 6],          # A+2
        14: [5, 6],          # A+3
        15: [4, 5, 6],       # A+4
        16: [4, 5, 6],       # A+5
        17: [3, 4, 5, 6],    # A+6
        18: [3, 4, 5, 6],    # A+7
    }
    HARD_HIT = {
        9:  [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        10: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        11: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        12: [2, 3, 7, 8, 9, 10, 1],
        13: [7, 8, 9, 10, 1],
        14: [7, 8, 9, 10, 1],
        15: [7, 8, 9, 10, 1],
        16: [7, 8, 9, 10, 1],
    }
    SOFT_HIT = {
        13: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        14: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        15: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        16: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        17: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
        18: [9, 10, 1],
    }

    def act(self, state):
        total, dealer_upcard, usable_ace, is_first_action = state

        if total >= 21:
            return 0
        if not usable_ace and total >= 17:
            return 0
        if total <= 8:
            return 1

        if usable_ace:
            if total >= 19:
                return 0
            if is_first_action and dealer_upcard in self.SOFT_DOUBLE.get(total, []):
                return 2
            return 1 if dealer_upcard in self.SOFT_HIT.get(total, []) else 0
        else:
            if is_first_action and dealer_upcard in self.HARD_DOUBLE.get(total, []):
                return 2
            return 1 if dealer_upcard in self.HARD_HIT.get(total, []) else 0



#  Input vector (5 features): [player_sum/21, dealer_upcard/10, usable_ace, is_first_action, theta]
#  theta is sampled from U(0,1) during training

INPUT_DIM  = 5
NUM_ACTIONS = 3   # 0=stand, 1=hit, 2=double


def encode_state(state, theta):
    total, dealer_upcard, usable_ace, is_first_action = state
    return np.array([
        total         / 21.0,
        dealer_upcard / 10.0,
        float(usable_ace),
        float(is_first_action),   
        float(theta),            
    ], dtype=np.float32)


class QNetwork(nn.Module):
    """
    5 inputs → 128 → 128 → 3 outputs (Q for stand, hit, double).
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, output_dim=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)



class ReplayBuffer:

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_vec, action, reward, next_state_vec, done):
        self.buffer.append((state_vec, action, reward, next_state_vec, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(
        self,
        lr=1e-3,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999995,
        batch_size=64,
        buffer_capacity=100_000,
        target_update_freq=500,
        min_buffer_size=1_000,
    ):
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer_size    = min_buffer_size

        self.online_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer(capacity=buffer_capacity)

    def act(self, state, theta, training=True):
        is_first = state[3]

        if training and random.random() < self.epsilon:
            if is_first:
                return random.randint(0, NUM_ACTIONS - 1)
            else:
                return random.randint(0, 1)   # only stand/hit after first action

        state_vec = encode_state(state, theta)
        with torch.no_grad():
            q_vals = self.online_net(
                torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)

        if not is_first:
            q_vals[2] = -float('inf')   # mask double when illegal

        return int(q_vals.argmax().item())

    def store(self, state, theta, action, reward, next_state, done):
        self.buffer.push(
            encode_state(state, theta),
            action,
            reward,
            encode_state(next_state, theta),
            float(done),
        )

    def train_step(self):
        if len(self.buffer) < self.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.online_net(states)
        q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1).values
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def policy(self, state, theta):
        return self.act(state, theta, training=False)


def persona_action(state, theta):
    """
    Simulate a player with continuous aggression theta.

    Unambiguous hands: everyone agrees (hit <=8, stand hard >=17)
    Ambiguous zone (9-16): theta controls hit/double probability
    Double: only considered on first action, probability scales with theta
    """
    total, dealer_upcard, usable_ace, is_first_action = state

    if total <= 8:
        return 1
    if not usable_ace and total >= 17:
        return 0
    if usable_ace and total >= 19:
        return 0

    # On first action, aggressive players occasionally double
    if is_first_action and total in range(9, 12) and theta > 0.5:
        double_prob = (theta - 0.5) * 0.6   # max 30% double prob at theta=1
        if random.random() < double_prob:
            return 2

    return 1 if random.random() < theta else 0


def train_dqn(num_episodes=500_000, num_decks=6, verbose=True):
    env   = BlackjackEnv(num_decks=num_decks)
    agent = DQNAgent()

    win_log  = []
    loss_log = []
    wins = 0
    total_loss = 0.0
    loss_steps = 0
    window = 10_000

    for ep in range(1, num_episodes + 1):
        theta = float(np.random.uniform(0, 1))

        state = env.reset()
        done  = False

        while not done:
            if random.random() < agent.epsilon:
                action = persona_action(state, theta)
            else:
                action = agent.act(state, theta, training=False)

            next_state, reward, done = env.step(action)
            agent.store(state, theta, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_steps += 1

            state = next_state

        agent.decay_epsilon()

        if ep % agent.target_update_freq == 0:
            agent.sync_target()

        if reward > 0:
            wins += 1
        if ep % window == 0:
            win_rate = wins / window
            avg_loss = total_loss / loss_steps if loss_steps > 0 else 0.0
            win_log.append((ep, win_rate))
            loss_log.append((ep, avg_loss))
            if verbose:
                print(f"  ep {ep:>7} | win rate: {win_rate:.3f}"
                      f" | ε: {agent.epsilon:.4f}"
                      f" | loss: {avg_loss:.4f}"
                      f" | buffer: {len(agent.buffer):,}")
            wins = 0
            total_loss = 0.0
            loss_steps = 0

    return agent, win_log, loss_log


def _eval(env, policy_fn, num_episodes):
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
        "win_rate":       wins        / total,
        "loss_rate":      losses      / total,
        "push_rate":      pushes      / total,
        "house_edge":    -total_reward / total,
        "expected_return": total_reward / total,
    }


def evaluate_dqn(agent, theta, num_decks=6, num_episodes=100_000):
    env = BlackjackEnv(num_decks=num_decks)
    return _eval(env, lambda s: agent.policy(s, theta), num_episodes)


def evaluate_basic_strategy(num_decks=6, num_episodes=100_000):
    env = BlackjackEnv(num_decks=num_decks)
    bs  = BasicStrategy()
    return _eval(env, bs.act, num_episodes)


def evaluate_random(num_decks=6, num_episodes=100_000):
    env = BlackjackEnv(num_decks=num_decks)
    return _eval(env, lambda s: random.randint(0, NUM_ACTIONS - 1) if s[3]
                                else random.randint(0, 1), num_episodes)


def compute_agreement(agent, theta):
    """Agreement with basic strategy at a given theta."""
    bs = BasicStrategy()
    states = [
        (total, upcard, usable_ace, is_first)
        for total      in range(4, 22)
        for upcard     in range(1, 11)
        for usable_ace in [False, True]
        for is_first   in [True, False]
    ]
    agree = sum(agent.policy(s, theta) == bs.act(s) for s in states)
    return agree / len(states)


def plot_learning_curve(win_log, loss_log, save_path="dqn_training.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    eps, rates = zip(*win_log)
    eps, losses = zip(*loss_log)
    ax1.plot(eps, rates, color="#3b82f6")
    ax1.set_ylabel("Win rate")
    ax1.set_title("DQN training (hit / stand / double, continuous θ)")
    ax2.plot(eps, losses, color="#ef4444", linewidth=0.8)
    ax2.set_ylabel("MSE loss")
    ax2.set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_theta_vs_house_edge(results_by_theta, bs_result,
                              eval_thetas, save_path="theta_vs_edge.png"):
    edges = [results_by_theta[t]["house_edge"] * 100 for t in eval_thetas]
    plt.figure(figsize=(8, 4))
    plt.plot(eval_thetas, edges, marker='o', label="DQN agent")
    plt.axhline(bs_result["house_edge"] * 100, color='black',
                linestyle='--', linewidth=1, label="Basic Strategy")
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.xlabel("Theta (player aggression)")
    plt.ylabel("House edge (%)")
    plt.title("House edge vs continuous theta — DQN with double down")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


def plot_policy_heatmap(agent, eval_thetas, save_path="policy_heatmap_dqn.png"):
    """
    One column per theta value + basic strategy.
    Green=hit, Red=stand, Blue=double.
    """
    bs      = BasicStrategy()
    totals  = list(range(8, 21))
    upcards = list(range(2, 11)) + [1]
    n_cols  = len(eval_thetas) + 1

    # 3-color map: 0=stand(red), 1=hit(green), 2=double(blue)
    cmap = mcolors.ListedColormap(["#d9534f", "#5cb85c", "#3b82f6"])

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

    for col, theta in enumerate(eval_thetas):
        grid = np.zeros((len(totals), len(upcards)))
        for i, total in enumerate(totals):
            for j, upcard in enumerate(upcards):
                grid[i, j] = agent.policy((total, upcard, False, True), theta)
        im = axes[col].imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        axes[col].set_title(f"θ={theta:.1f}", fontsize=9)
        axes[col].set_xticks(range(len(upcards)))
        axes[col].set_xticklabels([str(u) if u != 1 else "A" for u in upcards], fontsize=7)
        axes[col].set_yticks(range(len(totals)))
        axes[col].set_yticklabels(totals, fontsize=7)
        axes[col].set_xlabel("Dealer upcard", fontsize=8)
        if col == 0:
            axes[col].set_ylabel("Player total (hard)", fontsize=8)

    # Basic strategy column
    bs_grid = np.zeros((len(totals), len(upcards)))
    for i, total in enumerate(totals):
        for j, upcard in enumerate(upcards):
            bs_grid[i, j] = bs.act((total, upcard, False, True))
    im = axes[-1].imshow(bs_grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    axes[-1].set_title("Basic Strategy\n(ground truth)", fontsize=9)
    axes[-1].set_xticks(range(len(upcards)))
    axes[-1].set_xticklabels([str(u) if u != 1 else "A" for u in upcards], fontsize=7)
    axes[-1].set_yticks(range(len(totals)))
    axes[-1].set_yticklabels(totals, fontsize=7)
    axes[-1].set_xlabel("Dealer upcard", fontsize=8)

    fig.colorbar(im, ax=axes.tolist(), ticks=[0, 1, 2],
                 label="0=Stand  1=Hit  2=Double")
    plt.suptitle("DQN learned policy by theta — hard hands, first action",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# Theta values to evaluate 
EVAL_THETAS = [0.1, 0.3, 0.5, 0.7, 0.9]

if __name__ == "__main__":
    NUM_EPISODES  = 500_000
    EVAL_EPISODES = 100_000

    print("=" * 55)
    print("Training DQN (hit / stand / double, continuous θ)...")
    print("=" * 55)
    agent, win_log, loss_log = train_dqn(
        num_episodes=NUM_EPISODES, verbose=True
    )

    # Evaluate across theta values 
    print("\nEvaluating across theta values (100k episodes each)...")
    results_by_theta = {t: evaluate_dqn(agent, t, num_episodes=EVAL_EPISODES)
                        for t in EVAL_THETAS}
    bs_result = evaluate_basic_strategy(num_episodes=EVAL_EPISODES)
    rn_result = evaluate_random(num_episodes=EVAL_EPISODES)

    # Results table 
    print("\n--- Results ---")
    print(f"{'Agent':<30} {'Win':>6} {'Loss':>6} {'Push':>6} "
          f"{'Exp Return':>11} {'House Edge':>11}")
    print("-" * 74)
    print(f"{'Random baseline':<30} {rn_result['win_rate']:>6.3f} "
          f"{rn_result['loss_rate']:>6.3f} {rn_result['push_rate']:>6.3f}  "
          f"{'N/A':>10}  {'N/A':>10}")
    print(f"{'Basic Strategy':<30} {bs_result['win_rate']:>6.3f} "
          f"{bs_result['loss_rate']:>6.3f} {bs_result['push_rate']:>6.3f}  "
          f"{bs_result['expected_return']:>+10.4f}  "
          f"{bs_result['house_edge']*100:>+9.2f}%")

    for t in EVAL_THETAS:
        res = results_by_theta[t]
        print(f"{'DQN θ='+str(t):<30} {res['win_rate']:>6.3f} "
              f"{res['loss_rate']:>6.3f} {res['push_rate']:>6.3f}  "
              f"{res['expected_return']:>+10.4f}  "
              f"{res['house_edge']*100:>+9.2f}%")

    # Agreement with basic strategy 
    print("\n--- Policy agreement with Basic Strategy ---")
    for t in EVAL_THETAS:
        agr = compute_agreement(agent, t)
        print(f"  θ={t}: {agr:.3f}")

    # Plots
    print("\nGenerating plots...")
    plot_learning_curve(win_log, loss_log)
    plot_theta_vs_house_edge(results_by_theta, bs_result, EVAL_THETAS)
    plot_policy_heatmap(agent, EVAL_THETAS)

    print("\nDone.")