function POMDPs.solve(solver::RobustMCTSSolver, rmdp::RobustMDP, p::RobustMCTSPlanner=RobustMCTSPlanner(solver, rmdp, ))
    if isa(p.solver.rollout_solver, Solver) 
        p.rollout_policy = solve(p.solver.rollout_solver, solver.rollout_nature)
    else
        p.rollout_policy = p.solver.rollout_solver
    end
    return p
end

function POMDPs.action{S,A}(p::RobustMCTSPlanner{S,A}, s::S, a::A=create_action(p.rmdp))
    # This function calls simulate and chooses the approximate best action from the reward approximations
    # XXX do we need to make a copy of the state here?
    for i = 1:p.solver.n_iterations
        simulate(p, deepcopy(s), p.solver.depth)
    end
    snode = p.tree[s]
    best_Q = -Inf
    local best_a
    for (a, sanode) in snode.A
        if sanode.Q > best_Q
            best_Q = sanode.Q
            best_a = a
        end
    end
    # XXX some publications say to choose action that has been visited the most
    return best_a # choose action with highest approximate value 
end

function simulate{S,A}(p::RobustMCTSPlanner{S,A}, s::S, d::Int)
    # TODO: reimplement this as a loop instead of a recursion?

    # This function returns the reward for one iteration of MCTS 
    if d == 0 || isterminal(p.rmdp, s)
        return 0.0 # XXX is this right or should it be a rollout?
    end
    if !haskey(p.tree,s) # if state is not yet explored, add it to the set of states, perform a rollout 
        p.tree[s] = RobustStateNode{S,A}() # TODO: Mechanism to set N0
        p.tree[s].N += 1
        return estimate_value(p,s,d)
    end

    snode = p.tree[s] # save current state node so we do not have to iterate through map many times
    snode.N += 1

    # action progressive widening
    if length(snode.A) <= p.solver.k_action*snode.N^p.solver.alpha_action # criterion for new action generation
        a = next_action(p.solver.action_generator, p.rmdp, s, snode) # action generation step
        if !haskey(snode.A,a) # make sure we haven't already tried this action
            snode.A[a] = RobustActionNode{S}() # TODO: Mechanism to set N0, Q0
        end
    end

    best_UCB = -Inf
    local a
    sN = snode.N
    for (act, sanode) in snode.A
        if sN == 1 && sanode.N == 0
            UCB = sanode.Q
        else
            c = p.solver.c # for clarity
            UCB = sanode.Q + c*sqrt(log(sN)/sanode.N)
        end
        @assert !isnan(UCB)
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            a = act
        end
    end

    sanode = snode.A[a]
    sanode.N += 1

    # nature's progressive widening
    if length(sanode.nature_nodes) <= p.solver.k_nature*sanode.N^p.solver.alpha_nature
        tau = next_model(p.solver.model_generator, p.rmdp, s, a, snode, sanode)
        if !haskey(sanode.nature_nodes, tau) 
            sanode.nature_nodes[tau] = NatureNode{S}()
        end
    end

    # choose worst nature
    worst_UCB = Inf
    local tau
    saN = sanode.N
    for (mdp, nnode) in sanode.nature_nodes
        if saN == 1 && nnode.N == 0
            UCB = nnode.Q
        else
            c = p.solver.c_nature # for clarity
            UCB = nnode.Q - c*sqrt(log(saN)/nnode.N)
        end
        @assert !isnan(UCB)
        @assert !isequal(UCB, Inf)
        if UCB < worst_UCB
            worst_UCB = UCB
            tau = mdp
        end
    end

    nnode = sanode.nature_nodes[tau]

    # state progressive widening
    if length(nnode.transitions) <= p.solver.k_state*nnode.N^p.solver.alpha_state # criterion for new transition state consideration
        sp, r = generate_sr(tau, s, a, p.solver.rng) # choose a new state and get reward

        if !haskey(nnode.transitions,sp) # if transition state not yet explored, add to set and update reward
            nnode.transitions[sp] = TransitionNode() # TODO: mechanism for assigning N0
            nnode.transitions[sp].R = r
        end
        nnode.transitions[sp].N += 1

    else # sample from transition states proportional to their occurence in the past
        # warn("sampling states: |V|=$(length(nnode.V)), N=$(nnode.N)")
        total_N = reduce(add_N, 0, values(nnode.transitions))
        rn = rand(p.solver.rng, 1:total_N) # this is where Jon's bug was (I think)
        cnt = 0
        local sp, tnode
        for (sp,tnode) in nnode.transitions
            cnt += tnode.N
            if rn <= cnt
                break
            end
        end

        r = tnode.R
    end

    q = r + discount(p.rmdp)*simulate(p,sp,d-1)

    nnode.N += 1

    sanode.Q += (q - sanode.Q)/sanode.N
    nnode.Q += (q - nnode.Q)/nnode.N

    return q
end

"""
Add the N's of two sas nodes - for use in reduce
"""
add_N(a::TransitionNode, b::TransitionNode) = a.N + b.N
add_N(a::Int, b::TransitionNode) = a + b.N

"""
Estimate the value of being at state s

By default, run a rollout simulation.
"""
function estimate_value(p::RobustMCTSPlanner, s, d::Int)
    rollout(p, s, d)
end

function rollout(p::RobustMCTSPlanner, s, d::Int)
    sim = RolloutSimulator(rng=p.solver.rng, max_steps=d) # TODO(?) add a mechanism to customize this
    POMDPs.simulate(sim, p.solver.rollout_nature, p.rollout_policy, s)
end

