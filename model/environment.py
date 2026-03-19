import math
import numpy as np


class Env():
    """RL Environment for LLM-Enhanced Interactive Recommendation.

    Implements the upgraded dual reward function from Section 4.3:
      R_new(s̃_t, ã) = r_rel(s̃_t, ã) + β · ψ̂_a · r_nov(a)

    Where:
      r_rel  = Relevance reward (CF-based weighted similarity to observation window)
      r_nov  = Novelty reward = 1 / log(pop(a) + offset)
      ψ̂_a   = LLM zero-shot semantic quality score ∈ [0, 1] (from DeepSeek API)
      β      = Debiasing strength hyperparameter (default 0.4)

    Key innovation: ψ̂_a replaces the original Bayesian quality factor that was
    hopelessly dependent on |N(a)| (interaction count). Now, even a cold-start
    item with zero interactions can receive a high reward if the LLM judges
    its content quality to be excellent.
    """

    def __init__(self, user, observation_data, I,
                 item_sim_matrix, item_pop_dict, quality_dict, mask_list, K,
                 beta=0.4):
        self.observation = np.array(observation_data, dtype=int)
        self.n_observation = len(self.observation)
        self.action_space = I
        self.n_actions = len(self.action_space)
        self.user = user
        self.item_sim_matrix = item_sim_matrix
        self.item_pop_dict = item_pop_dict
        self.quality_dict = quality_dict   # ← Now contains LLM ψ̂_a scores
        self.mask_list = mask_list
        self.K = K
        self.beta = beta                   # Debiasing strength

    def reset(self, observation):
        self.observation = np.array(observation, dtype=int)
        self.n_observation = len(self.observation)
        return self.observation

    def step(self, action):
        """Execute one step: compute the upgraded dual reward and state transition.

        Reward formula:
          R_new = r_rel + β · ψ̂_a · r_nov

        Where:
          r_rel = Σ_i (0.9^i) · sim(s[-(i+1)], action)  — relevance via CF similarity
          r_nov = 1 / log10(pop(action) + 1.1)            — novelty via inverse popularity
          ψ̂_a  = quality_dict[action]                     — LLM zero-shot quality score

        Returns:
            s_: next state (int array of item IDs)
            r: reward scalar
            done: always False (episodic termination handled externally)
        """
        done = False
        s = self.observation

        if s[-1] == action:
            # Penalty for recommending the exact same item as last observation
            if str(s[-1]) in self.item_sim_matrix and str(action) in self.item_sim_matrix[str(s[-1])]:
                self.item_sim_matrix[str(s[-1])][str(action)] = 0
            r = -1
        else:
            # ── r_rel: Relevance reward (CF-based) ──
            r_rel = 0
            for i in range(self.n_observation):
                key = str(s[-(i + 1)])
                if key in self.item_sim_matrix:
                    if str(action) in self.item_sim_matrix[key]:
                        r_rel += (0.9 ** i) * self.item_sim_matrix[key][str(action)]

            # ── ψ̂_a: LLM zero-shot semantic quality score ──
            psi_a = self.quality_dict.get(str(action), 0.5)

            # ── r_nov: Novelty reward (inverse popularity) ──
            pop_a = self.item_pop_dict.get(str(action), 1)
            r_nov = 1.0 / math.log(pop_a + 1.1, 10)

            # ── R_new = r_rel + β · ψ̂_a · r_nov ──
            r = r_rel + self.beta * psi_a * r_nov

        # State transition: append action to observation window
        if r > 0:
            s_temp_ = np.append(s, action)
            observation_ = np.delete(s_temp_, 0, axis=0)
        else:
            observation_ = s
        s_ = np.array(observation_, dtype=int)
        return s_, r, done

