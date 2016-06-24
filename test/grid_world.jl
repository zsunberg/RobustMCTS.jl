using POMDPModels
using POMDPToolbox
using GenerativeModels
using MCTS

mdp = GridWorld()

rmdp = RobustAdapter(mdp)

solver = RobustMCTSSolver(rollout_nature=mdp, c=10.0)

policy = solve(solver, rmdp)

sim = RolloutSimulator(max_steps=100)
simulate(sim, mdp, policy, initial_state(mdp, MersenneTwister()))

solver2 = RobustMCTSSolver(rng=MersenneTwister(), n_iterations=1000, rollout_nature=mdp, c=10.0, depth=100)
policy2 = solve(solver2, rmdp)
mcts_solver = DPWSolver(rng=MersenneTwister(), n_iterations=1000, exploration_constant=10.0, depth=100)
mcts_policy = solve(mcts_solver, mdp)

rng = MersenneTwister(12)
N = 100
# total_rank_difference = 0
# nb_misranked = 0
total_reward_difference = 0.0
total_reward = 0.0
abs_total_reward_difference = 0.0
for i in 1:N
    s = initial_state(mdp, rng)
#     a2 = action(policy2, s)
#     mcts_a = action(mcts_policy, s)
#     print('.')
#     if a2 != mcts_a
#         nb_misranked += 1
#         ranked = MCTS.ranked_actions(policy2, s)
#         total_rank_difference += findfirst(ranked, mcts_a)
#     end
    policy2 = solve(solver2, rmdp)
    mcts_policy = solve(mcts_solver, mdp)
    sim2 = RolloutSimulator(rng=MersenneTwister(i), max_steps=100)
    sim3 = RolloutSimulator(rng=MersenneTwister(i), max_steps=100)
    r2 = simulate(sim2, mdp, policy2, s)
    r3 = simulate(sim3, mdp, mcts_policy, s)
    total_reward += r2
    total_reward_difference += r3-r2
    abs_total_reward_difference += abs(r3-r2)
    print('.')
end

# @show nb_misranked
# @show average_rank_difference = total_rank_difference/N

@show total_reward/N
@show total_reward_difference/N
@show abs_total_reward_difference/N
