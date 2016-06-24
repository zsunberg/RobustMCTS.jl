"""
For testing purposes - simply has a single model
"""
type RobustAdapter{S,A} <: RobustMDP{S,A}
    mdp::MDP{S,A}
end

rand_model(rmdp::RobustAdapter, rng::AbstractRNG) = rmdp.mdp

representative_mdp(r::RobustAdapter) = r.mdp
