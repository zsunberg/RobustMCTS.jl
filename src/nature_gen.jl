abstract ModelGenerator

function next_model(gen::ModelGenerator, rmdp::RobustMDP, s, a, snode::RobustStateNode, anode::RobustActionNode)
    next_model(gen, rmdp, s, a)
end

type RandomModelGenerator <: ModelGenerator
    rng::AbstractRNG
end
RandomModelGenerator() = RandomModelGenerator(MersenneTwister())

function next_model(gen::RandomModelGenerator, rmdp::RobustMDP, s, a, snode::RobustStateNode, anode::RobustActionNode)
    next_model(gen, rmdp, s, a)
end

next_model(gen::RandomModelGenerator, rmdp::RobustMDP, s, a) = rand_model(rmdp, gen.rng)

