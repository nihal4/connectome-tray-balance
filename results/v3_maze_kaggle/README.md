# v3: Kaggle maze runs (1M steps each, one seed)

From `train_log.txt` (the notebook's filtered training output). Eval = 10 episodes,
deterministic policy, every 50k steps. `maze3h` was still running when this was saved.

| level | connectome (1000 fly neurons) | MLP (same size) |
|---|---|---|
| `maze_fixed3` (one maze) | first success at 300k; 30–40 % at 950k–1M, still rising | 100 % from 300k to 650k, then fell to 30 % (reward hack, see below) |
| `maze3` (new maze every episode) | 0 % throughout | 0 % until 850k, 30 % at 1M |

**Reading:**
- On the fixed maze the MLP learned about 3x faster. The connectome policy does learn,
  slowly: success only appears after ~700k and was climbing at the end.
- Random mazes are hard for both at 1M steps; the MLP only starts at the very end.
- One seed per run, so differences are suggestive, not conclusive.

**Reward hack (fixed after this run):** the MLP's eval return kept rising (25 → 50) while
its success rate dropped (100 % → 30 %) and episodes got longer. The maze gave a flat
+0.2 per step while the ball rested in the goal, which was worth more than finishing, so
the policy learned to hover in and out of the goal. The hold reward is now progress-based
(+0.4 per step held, all taken back if the ball leaves), so hovering earns nothing. Runs
from this version are not directly comparable with runs after the fix.
