module BasicAugmentedLangrange

    using LinearAlgebra, Optim, Random, Printf, ForwardDiff, DifferentiationInterface

    include("optimise.jl")

    export optimise

    include("optimise_with_slack.jl")

    export optimise_with_slack
end
