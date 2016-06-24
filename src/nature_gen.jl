abstract ModelGenerator

type RandomModelGenerator <: ModelGenerator
    rng::AbstractRNG
end
RandomModelGenerator() = RandomModelGenerator(MersenneTwister())

function next_model(gen::RandomModelGenerator, rmdp::RobustMDP, s, a, snode::RobustStateNode, anode::RobustActionNode)
    rand_model(rmdp, gen.rng)
end
