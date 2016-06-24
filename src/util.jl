function MCTS.ranked_actions(policy::RobustMCTSPlanner, state)
    actions = keys(policy.tree[state].A)
    q_val(a) = policy.tree[state].A[a].Q
    return sort!(collect(actions), by=q_val, rev=true)
end
