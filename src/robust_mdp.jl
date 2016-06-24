"""
Abstract type for a Robust MDP.

Should use the same interface as an MDP for things like actions, etc. If representative_mdp is implemented, then the interface methods for that MDP will be used when available.
"""
abstract RobustMDP{S,A}

POMDPs.isterminal(r::RobustMDP, s) = isterminal(representative_mdp(r), s)
POMDPs.discount(r::RobustMDP) = discount(representative_mdp(r))
POMDPs.create_action(r::RobustMDP) = create_action(representative_mdp(r))


function next_action(gen::MCTS.RandomActionGenerator, rmdp::RobustMDP, s, ::RobustStateNode)
    if isnull(gen.action_space) 
        gen.action_space = Nullable{AbstractSpace}(actions(representative_mdp(rmdp)))
    end
    rand(gen.rng, actions(representative_mdp(rmdp), s, get(gen.action_space)))
end


type MDPCollection{S,A} <: RobustMDP{S,A}
    mdps::Vector{MDP{S,A}}
end


type MixedMDP{S,A} <: MDP{S,A}
    collection::MDPCollection{S,A}
    dist::Vector{Float64} # probability weights
end

function GenerativeModels.generate_sr{S,A}(mdp::MixedMDP{S,A}, s, a, rng::AbstractRNG)
    # select model
    r = rand(rng)
    psum = 0.0
    model = Nullable{MDP{S,A}}()
    for i in 1:length(dist)
        psum += mdp.dist[i]
        if r < psum
            model = Nullable{MDP{S,A}}(mdp.collection.mdps[i])
            break
        end
    end
    return generate_sr(get(model), s, a, rng)
end
