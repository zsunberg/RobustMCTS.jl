module RobustMCTS

using POMDPs
using GenerativeModels
using POMDPToolbox

using MCTS
import MCTS: next_action

using Parameters

using Distributions

import POMDPs: action, solve

export
    RobustAdapter,
    RobustMCTSSolver,
    RobustMCTSPlanner,
    ModelGenerator,
    RandomModelGenerator,
    next_model,
    RandomActionGenerator,
    next_action,
    action,
    solve

include("tree.jl")
include("robust_mdp.jl")
include("nature_gen.jl")

@with_kw type RobustMCTSSolver <: Solver
    depth::Int = 10                      # search depth
    c::Float64 = 1.0                      # exploration constant- governs trade-off between exploration and exploitation in MCTS
    c_nature::Float64 = c               # exploration constant for nature
    n_iterations::Int = 100            # number of iterations
    k_action::Float64 = 10.0               # first constant controlling action generation
    alpha_action::Float64 = 0.5          # second constant controlling action generation
    k_nature::Float64 = 10.0
    alpha_nature::Float64 = 0.5
    k_state::Float64 = 10.0                # first constant controlling transition state generation
    alpha_state::Float64 = 0.5          # second constant controlling transition state generation
    rng::AbstractRNG = MersenneTwister(11)
    rollout_solver::Union{Policy,Solver} = RandomSolver(rng) # if this is a Solver, solve() will be called to get the rollout policy
                                         # if this is a Policy, it will be used for rollouts directly
    rollout_nature::MDP                  # TEMP this will probably change
    action_generator::ActionGenerator = RandomActionGenerator()
    model_generator::ModelGenerator = RandomModelGenerator()
end

type RobustMCTSPlanner{S,A} <: Policy{S}
    solver::RobustMCTSSolver
    rmdp::RobustMDP{S,A}
    tree::Dict{S,RobustStateNode{S,A}}
    rollout_policy::Policy
end

function RobustMCTSPlanner{S,A}(solver::RobustMCTSSolver, rmdp::RobustMDP{S,A})
    return RobustMCTSPlanner(solver,
                             rmdp,
                             Dict{S,RobustStateNode{S,A}}(),
                             RandomPolicy(solver.rollout_nature, rng=solver.rng))
end


include("solver.jl")
include("adapter.jl")
include("util.jl")

include("visualization.jl")

end # module
