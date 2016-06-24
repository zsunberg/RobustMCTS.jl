using POMDPModels
using POMDPToolbox
using GenerativeModels
using MCTS
using RobustMCTS

mdp = GridWorld()
rmdp = RobustAdapter(mdp)
solver = RobustMCTSSolver(rollout_nature=mdp, c=10.0)
policy = solve(solver, rmdp)
state = initial_state(mdp, MersenneTwister())
a = action(policy, state)

v = MCTS.TreeVisualizer(policy, state)
json = MCTS.create_json(v)
