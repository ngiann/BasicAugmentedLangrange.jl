function optimise_with_slack(f, g, x; maxiterations = Inf, inneriterations = 100_000, ϵ_constraint = 1e-4, ϵ_λ = 1e-4, ρ_rate = 1.05, verbose = false, backend = AutoForwardDiff(chunksize=16))
    
    
    split(xext) = xext[1:end-1], exp(xext[end])

    merge(x,s) = [x; log(s)]

    opt = Optim.Options(iterations = inneriterations, show_trace = false, show_every = 1, g_tol = 1e-8)

    
    function Lₐ(x_ext; λ = λ, ρ = ρ) # minimise augmented Lagrange objective

        local x, s = split(x_ext)
        
        f(x) + λ*(g(x) + s) + (ρ/2) * (g(x) + s)^2

    end


    function minimise_Lₐ(x_ext; λ = λ, ρ = ρ)

        local helper(x′) = Lₐ(x′; λ = λ, ρ = ρ) # fix λ, ρ and minimise

        # gradhelper!(s, p) = copyto!(s, DifferentiationInterface.gradient(helper, backend, p))

        # local res = optimize(helper, gradhelper!, x_ext, LBFGS(), opt)

        local res = optimize(helper, x_ext, NelderMead(), opt) # ❗ This is temporary. Comment back in the two lines above ❗

        Optim.minimizer(res), Optim.converged(res), Optim.minimum(res)
        
    end


    λ, ρ, s = 1e-3, 1e-3, 1e-3 # initial values

    x_ext = merge(x, s)

    λprv = λ

    converged = false

    iteration = 1


    while ~converged && iteration <= maxiterations

        x_ext, converged, _ = minimise_Lₐ(x_ext; λ = λ, ρ = ρ)

        x, s = split(x_ext)

        λprv = λ

        λ = λ + ρ*(g(x) + s)


        if g(x) + s > ϵ_constraint # if constrained violated, increase penalty
           
            ρ = min(ρ_rate*ρ, 1e6)

        end


        if verbose 

            @printf("(%d): λ = %.6f, ρ = %.6f, s = %.6f, g(x) = %.4f, Lₐ =  %.6f\n", iteration, λ, ρ, s, g(x), Lₐ(x_ext; λ = λ, ρ = ρ))

        end


        # Check for convergence

        converged = converged && abs(g(x) + s) < ϵ_constraint # constraint must be satisfied within tolerance

        converged = converged && (s > 0)                      # slack variable must be positive (guaranteed by design in our case)

        converged = converged && abs(λprv - λ) < ϵ_λ          # check if the Lagrange multipliers converge

        # converged = converged && abs(s * λ) < 1e-3            # complementary slackness

        iteration += 1

    end

    return minimise_Lₐ(x_ext; λ = λ, ρ = ρ)

end