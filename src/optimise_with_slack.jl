function optimise_with_slack(f, g, x; iterations = 1, inneriterations = 10_000, ϵ = 1e-3)

    D = length(x)

    
    split(xext) = xext[1:D], exp(xext[end])

    merge(x,s) = [x; log(s)]

    opt = Optim.Options(iterations = inneriterations, show_trace = false, show_every = 1)

    
    function Lₐ(x_ext; λ = λ, μ = μ) # minimise this

        local x, s = split(x_ext)
        
        f(x) + λ*g(x) + (μ/2) * (g(x) - s)^2

    end


    function minimise_Lₐ(x_ext; λ = λ, μ = μ)

        local helper(x′) = Lₐ(x′; λ = λ, μ = μ) # fix λ, μ and minimise this

        Optim.minimizer(optimize(helper, x_ext, LBFGS(), opt, autodiff=:forward))
        
    end

    ϵ_λ = 1e-4

    λ, μ, s = 1e-3, 1e-3, 1e-3 # initial values

    x_ext = merge(x, s)

    λprv = λ

    for i in 1:iterations

        x_ext = minimise_Lₐ(x_ext; λ = λ, μ = μ)

        x, s = split(x_ext)

        λprv = λ

        λ = λ + μ*(g(x) + s)

        if g(x) + s > ϵ
           
            μ = min(1.05*μ, 1e6)

        end

        @printf("iterations %d: λ = %.6f, μ = %.6f, s = %.6f, g(x) = %.4f, Lₐ =  %.6f\n", i, λ, μ, s, g(x), Lₐ(x_ext; λ = λ, μ = μ))

        # Check for convergence

        converged = abs(g(x) + s) < ϵ

        converged = converged && (s > 0) # this should hold by design

        converged = converged && abs(λprv - λ) < ϵ_λ

        grad = ForwardDiff.gradient(_x_ext -> Lₐ(_x_ext; λ = λ, μ = μ), x_ext)

        converged = converged && norm(grad) < 1e-4

        if converged 

            @printf("Converged!\n"); break

        end

    end

    xfinal = minimise_Lₐ(x_ext; λ = λ, μ = μ)

    return xfinal

end