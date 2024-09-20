function optimise_with_slack(f, g, x; maxiterations = Inf, inneriterations = 10_000, ϵ_constraint = 1e-4, ϵ_λ = 1e-4, μ_rate = 1.05, verbose = false)

    D = length(x)

    
    split(xext) = xext[1:D], exp(xext[end])

    merge(x,s) = [x; log(s)]

    opt = Optim.Options(iterations = inneriterations, show_trace = false, show_every = 1, g_tol = 1e-8)

    
    function Lₐ(x_ext; λ = λ, μ = μ) # minimise augmented Lagrange objective

        local x, s = split(x_ext)
        
        f(x) + λ*g(x) + (μ/2) * (g(x) + s)^2

    end


    function minimise_Lₐ(x_ext; λ = λ, μ = μ)

        local helper(x′) = Lₐ(x′; λ = λ, μ = μ) # fix λ, μ and minimise

        local res = optimize(helper, x_ext, LBFGS(), opt, autodiff=:forward)

        Optim.minimizer(res), Optim.converged(res), Optim.minimum(res)
        
    end


    λ, μ, s = 1e-3, 1e-3, 1e-3 # initial values

    x_ext = merge(x, s)

    λprv = λ

    converged = false

    iteration = 1


    while ~converged && iteration <= maxiterations

        x_ext, converged, _ = minimise_Lₐ(x_ext; λ = λ, μ = μ)

        x, s = split(x_ext)

        λprv = λ

        λ = λ + μ*(g(x) + s)


        if g(x) + s > ϵ_constraint # if constrained violated, increase penalty
           
            μ = min(μ_rate*μ, 1e6)

        end


        if verbose 

            @printf("(%d): λ = %.6f, μ = %.6f, s = %.6f, g(x) = %.4f, Lₐ =  %.6f\n", iteration, λ, μ, s, g(x), Lₐ(x_ext; λ = λ, μ = μ))

        end


        # Check for convergence

        converged = converged && abs(g(x) + s) < ϵ_constraint # constraint must be satisfied within tolerance

        converged = converged && (s > 0)                      # slack variable must be positive (guaranteed by design in our case)

        converged = converged && abs(λprv - λ) < ϵ_λ          # check if the Lagrange multipliers converge

        # converged = converged && abs(s * λ) < 1e-3            # complementary slackness

        iteration += 1

    end

    return minimise_Lₐ(x_ext; λ = λ, μ = μ)

end