using QuantumToolbox


function get_overlap(func1::Matrix{Float64}, func2::Matrix{Float64})::Matrix{Float64}
    """
    Compute the overlap of two function
    Args:
        - func1: function to compare for the overlap (array)
        - func2: function to compare for the overlap (array)
    Returns: 
        - overlap: return the overlap of the two functions (array)
    """
    return min.(func1, func2)
end;

function overlap_error(state1::QuantumObject, state2::QuantumObject, dimension::Tuple{Int64,Int64}, prec::Int64)::Float64
    """
    Compute the area of the overlap of two functions
    Args:
        - state1
        - state2
        - dimension
        - prec
    Returns:
        - area of the overlap (Float)
    """

    xx = LinRange(dimension[1], dimension[2], prec)
    w1 = wigner((ptrace(state1, 1)), xx, xx)
    result1 = sum(w1, dims=2) / sum(w1)
    w2 = wigner((ptrace(state2, 1)), xx, xx)
    result2 = sum(w2, dims=2) / sum(w2)
    overlap_vector = get_overlap(result1, result2)
    return sum(overlap_vector)
end;

function distance_fidelity(state1::QuantumObject, state2::QuantumObject)::Float64
    """
    Using the formula of error Perr from https://journals.aps.org/pra/pdf/10.1103/PhysRevA.77.032311
    F = 1-Perr
    """
    return 1 - 0.5 * (1 - 0.5 * norm(state1 - state2))
end;


function simulation_separation(H_td, state0, state1, tlist, c_ops, parameters)::Float64
    output0 = mesolve(H_td, state0, tlist, c_ops, params=parameters, progress_bar=Val(false), saveat=[tau])
    output1 = mesolve(H_td, state1, tlist, c_ops, params=parameters, progress_bar=Val(false), saveat=[tau])
    return 1 - 0.5 * (1 - 0.5 * norm(ptrace(output0.states[end], 1) - ptrace(output1.states[end], 1)))
end;


function plot_wigner(output0, output1, xx)
    figure = Figure(size=(800, 400))
    w = wigner(normalize!(ptrace(output0.states[end], 1)), xx, xx)
    Axis(figure[1, 1], title="|0>")
    vbound = maximum(abs.(w))
    co = contourf!(xx, xx, w, levels=range(-vbound, vbound, length=20), colormap=:seismic)
    Colorbar(figure[1, 2], co)
    w = wigner(normalize!(ptrace(output1.states[end], 1)), xx, xx)
    Axis(figure[1, 3], title="|1>")
    vbound = maximum(abs.(w))
    co = contourf!(xx, xx, w, levels=range(-vbound, vbound, length=20), colormap=:seismic)
    Colorbar(figure[1, 4], co)
    return figure
end
function diagonalize_transmon(Nt, Nx, ec, ej)
    x_l = collect(range(-π, π, Nx + 1))
    pop!(x_l)
    dx = x_l[2] - x_l[1]
    x = Qobj(spdiagm(0 => x_l))
    p = 1 / (2 * dx) * spdiagm(-1 => -ones(ComplexF64, Nx - 1), 1 => ones(ComplexF64, Nx - 1))
    p[1, end] = 1 / (2 * dx)
    p[end, 1] = -1 / (2 * dx)
    n = -1im * p |> Qobj
    p2 = 1 / (dx^2) * spdiagm(-1 => ones(ComplexF64, Nx - 1), 0 => -2 * ones(Nx), 1 => ones(Nx - 1))
    p2[1, end] = 1 / dx^2
    p2[end, 1] = 1 / dx^2
    n2 = -p2 |> Qobj
    H = 4 * ec * n2 - ej * cos(x)
    vals_DWP, vecs_DWP, U = eigenstates(H, sparse=true, k=Nt, sigma=-ej - 0.5)
    vals_DWP = real.(vals_DWP)
    pot = -ej * cos(x)
    return vals_DWP, vecs_DWP, U, n, H, pot
end




function simpson_integrator(y, x)
    n = length(y) - 1
    n % 2 == 0 || error("`y` length (number of intervals) must be odd")
    length(x) - 1 == n || error("`x` and `y` length must be equal")
    h = (x[end] - x[1]) / n
    s = sum(y[1:2:n] + 4y[2:2:n] + y[3:2:n+1])
    return h / 3 * s
end


function overlap_err(states0, states1, tlist)
    integrated_overlap = simpson_integrator(tr.(states0 .* states1), tlist)
    return integrated_overlap / (2 * tlist[end])
end


function opposite_SNR(states_0::Vector, states_1::Vector, a, tlist, eta, kappa)::Float64
    betas_0 = expect(a, states_0)
    betas_1 = expect(a, states_1)
    diff_beta = abs.(betas_0 .- betas_1) .^ 2
    integrated_signal = simpson_integrator(diff_beta, tlist)
    return -sqrt(2 * eta * kappa * integrated_signal)
end


function relu(x)
    return maximum([0, x])
end