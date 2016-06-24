import JSON
import Base: writemime
import MCTS: node_tag, tooltip_tag

function MCTS.create_json{P<:RobustMCTSPlanner}(visualizer::TreeVisualizer{P})
    root_id = -1
    next_id = 1
    node_dict = Dict{Int, Dict{UTF8String, Any}}()
    s_dict = Dict{Any, Int}()
    sa_dict = Dict{Any, Int}()
    san_dict = Dict{Any, Int}()
    for (s, sn) in visualizer.policy.tree
        # create state node
        node_dict[next_id] = sd = Dict("id"=>next_id,
                                       "type"=>:state,
                                       "children_ids"=>Array(Int,0),
                                       "tag"=>node_tag(s),
                                       "tt_tag"=>tooltip_tag(s),
                                       "N"=>sn.N
                                       )
        if s == visualizer.init_state
            root_id = next_id 
        end
        s_dict[s] = next_id
        next_id += 1

        # create action nodes
        for (a, san) in sn.A
            node_dict[next_id] = sad = Dict("id"=>next_id,
                                            "type"=>:action,
                                            "children_ids"=>Array(Int,0),
                                            "tag"=>node_tag(a),
                                            "tt_tag"=>tooltip_tag(a),
                                            "N"=>san.N,
                                            "Q"=>san.Q
                                            )
            push!(sd["children_ids"], next_id)
            sa_dict[(s,a)] = next_id
            next_id += 1

            # create nature nodes
            for (mdp, nnode) in san.nature_nodes
                node_dict[next_id] = Dict("id"=>next_id,
                                          "type"=>:action,
                                          "children_ids"=>Array(Int,0),
                                          "tag"=>node_tag(mdp),
                                          "tt_tag"=>tooltip_tag(mdp),
                                          "N"=>nnode.N,
                                          "Q"=>nnode.Q
                                          )
                push!(sad["children_ids"], next_id)
                san_dict[(s,a,mdp)] = next_id
                next_id += 1
            end
        end
    end

    if root_id < 0
        error("""
                MCTS tree visualization: Policy does not have a node for the specified state.
            """)
    end

    # go back and refill transitions
    for (s, sn) in visualizer.policy.tree
        for (a, san) in sn.A
            for (mdp, nnode) in san.nature_nodes
                for sp in keys(nnode.transitions)
                    nd = node_dict[san_dict[(s,a,mdp)]]
                    if haskey(s_dict, sp)
                        push!(nd["children_ids"], s_dict[sp])
                    else
                        node_dict[next_id] = Dict("id"=>next_id,
                                                  "type"=>:state,
                                                  "children_ids"=>Array(Int,0),
                                                  "tag"=>node_tag(sp),
                                                  "tt_tag"=>tooltip_tag(sp),
                                                  "N"=>0
                                                  )
                        push!(nd["children_ids"], next_id)
                        next_id += 1
                    end
                end
            end
        end
    end
    json = JSON.json(node_dict)
    return (json, root_id)
end
