function optimise(f, g, x; iterations = 1, inneriterations = 1000)

    Lₐ(x; λ = λ, ρ = ρ) = f(x) + λ*g(x) + (0.5/ρ) * max(0, g(x))^2 # minimise this, equation (3)


    function minimise_Lₐ(x; λ = λ, ρ = ρ)

        local opt = Optim.Options(iterations = inneriterations, show_trace = false, show_every = 1)

        local helper(x) = Lₐ(x; λ = λ, ρ = ρ) # minimise this

        local result = optimize(helper, x, ConjugateGradient(), opt, autodiff=:forward)

        Optim.minimizer(result)

    end


    λ, ρ = 1e-3, 1.0 # initial values

    for i in 1:iterations # this is algorithm 1 in paper

        x = minimise_Lₐ(x; λ = λ, ρ = ρ) # step 2

        λ = max(0, λ + (1/ρ)*g(x))

        if g(x) > 0.0 
           
            ρ = max(0.99*ρ, 1e-6)

        end

        @printf("iterations %d: λ = %.6f, ρ = %.6f, g(x) = %.4f, Lₐ =  %.6f\n", i, λ, ρ, g(x), Lₐ(x; λ = λ, ρ = ρ))

    end

    xfinal = minimise_Lₐ(x; λ = λ, ρ = ρ)

    return xfinal

end