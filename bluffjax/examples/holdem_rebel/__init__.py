"""
ReBeL for heads-up Limit and No-Limit Texas Hold'em (Brown et al., NeurIPS
2020, "Combining Deep Reinforcement Learning and Search for
Imperfect-Information Games"), shared by HUL/hul_rebel.py and
HUNL/hunl_rebel.py.

Modules
    cards   1,326 hole-card combinations, card removal, exact showdown values
    game    betting rules mirroring the envs, search-tree template and trees
    solver  value network, Linear CFR-D subgame solving, chance-node values
    agent   test-time agent (safe search), opponents, mirrored matches, checkpoints
    train   self-play data generation, replay buffer, value-network training

Algorithm (paper Algorithm 2 with the released Liar's Dice code's details)
    PBS: public state (street, board, contributions, all-in) plus each
    player's normalised reach over the 1,326 hands; the joint deal is
    proportional to x_SB(h) x_BB(h') for hands sharing no card with each other
    or the board. Players are indexed by position (the small blind acts first
    on every street, as in the envs).
    Subgames run to the end of the current betting round. Leaves are folds,
    showdowns (river, or both all-in on the river) and end-of-round PBSs,
    valued by the network. They are solved with alternating-update Linear
    CFR-D: leaf values at every iteration come from the network at the leaf
    PBS of the current policy; root values are averaged with linear weights
    and are the training targets.
    Self-play samples an iteration t with P(t) ~ t, draws a deal from the
    root PBS and walks to a leaf with pi^t; one player, chosen at random per
    subgame, plays uniformly at random with probability 0.25 at each
    decision. Beliefs follow pi^t. Games restart at the initial PBS when a
    hand ends.
    Between rounds, the PBS before the next card(s) is a chance node whose
    value is the card-removal-weighted average of the network's values over
    the next cards. This gives the network both the end-of-round layers it
    is queried at and the start-of-round layers it is trained on (the
    paper's six layers). When both players are all-in, the chance nodes are
    followed to the river, where values are exact showdowns; the network
    learns all-in values itself, as in the paper.
    Acting (safe search): at the start of every betting round the agent
    solves the subgame at its current PBS, samples t and plays pi^t for the
    round, updating both players' beliefs with pi^t after every action. It
    uses only public information and its own cards.

Deviations from the paper, and why
    - Linear CFR-D (Algorithm 2), not the paper's modified CFR-AVG, which has
      no published form for alternating Linear CFR and no proof; no policy
      network / warm start (optional in the paper).
    - Scale: a 3-layer MLP (512 units) instead of 6 x 1536; 64 CFR
      iterations per subgame (the paper uses hundreds); budgets of 5e6 / 1e7
      samples instead of billions; a 131k FIFO replay buffer. This is what
      fits a shared 10 GB GPU in a few hours per run.
    - The network outputs both players' values at once (the paper indexes
      the player in the input), and the board is a 52-dim multi-hot vector
      instead of card embeddings.
    - Preflop chance nodes average over 256 sampled flops (self-normalised
      Monte Carlo) instead of all 22,100; turn and river cards are exact.
    - The tree has the envs' action sets, without folds when checking is
      free (dominated). No-Limit allows two non-all-in raises (half pot or
      pot) per round in the tree (config max_raises; all-in, call and fold
      are always there); if an opponent raises past the cap the agent
      re-solves from just before that action. The Limit tree is exact.
    - Fixed stacks and bet sizes (those of the envs): no randomisation.
    - The initial PBS is identical in every hand, so it is solved once per
      batch of self-play steps (the network is fixed within a batch) and its
      example is added once per step instead of once per game. Each game
      still samples its own iteration, so play is unchanged.

One sample is one value-network training example: a PBS with both players'
values for all hands, from one solved subgame (betting round or chance node).
"""
