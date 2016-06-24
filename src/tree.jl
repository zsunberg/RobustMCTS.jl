type TransitionNode
    N::Int
    R::Float64
end
TransitionNode() = TransitionNode(0, 0.0)

type NatureNode{S}
    transitions::Dict{S,TransitionNode}
    N::Int
    Q::Float64 # Q(s,a,tau)
    NatureNode() = new(Dict{S,TransitionNode}(), 0, 0.0)
end

type RobustActionNode{S}
    nature_nodes::Dict{MDP, NatureNode{S}}
    N::Int
    Q::Float64 # Q(s,a) = min_tau Q(s,a,tau)
    RobustActionNode() = new(Dict{MDP, NatureNode{S}}(), 0, 0.0)
end

type RobustStateNode{S,A}
    A::Dict{A,RobustActionNode{S}}
    N::Int
    RobustStateNode() = new(Dict{A,RobustActionNode{S}}(), 0)
end
