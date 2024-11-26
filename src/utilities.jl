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
    return min.(func1,func2);
end;

function overlap_error(state1::QuantumObject,state2::QuantumObject,dimension::Tuple{Int64,Int64},prec::Int64)::Float64
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

    xx = LinRange(dimension[1], dimension[2],prec);
    w1  =wigner((ptrace(state1,1)), xx, xx);
    result1= sum(w1, dims=2)/sum(w1);
    w2  =wigner((ptrace(state2,1)), xx, xx);
    result2= sum(w2, dims=2)/sum(w2);
    overlap_vector = get_overlap(result1,result2);
    return sum(overlap_vector);
end;

function distance_fidelity(state1::QuantumObject, state2::QuantumObject)::Float64
    """
    Using the formula of error Perr from https://journals.aps.org/pra/pdf/10.1103/PhysRevA.77.032311
    F = 1-Perr
    """
    return 1-0.5*(1-0.5*norm(state1-state2));
end;


function simulation_separation(H_td,state0,state1,tlist,c_ops,parameters)::Float64
output0 = mesolve(H_td, state0, tlist,c_ops,params = parameters,progress_bar = Val(false),saveat = [tau])
output1 = mesolve(H_td, state1, tlist,c_ops,params = parameters,progress_bar = Val(false),saveat = [tau])
return 1-0.5*(1-0.5*norm(ptrace(output0.states[end],1)-ptrace(output1.states[end],1)))
end;


function plot_wigner(output0,output1,xx)
    figure = Figure(size = (800, 400))    
    w = wigner(normalize!(ptrace(output0.states[end],1)), xx, xx)
    Axis(figure[1,1],title = L"$|0>$")
    vbound = maximum(abs.(w))
    co = contourf!(xx,xx,w, levels= range(-vbound, vbound, length = 20) ,colormap=:seismic)
    Colorbar(figure[1, 2],co)
    w = wigner(normalize!(ptrace(output1.states[end],1)), xx, xx)
    Axis(figure[1,3],title = L"$|1>$")
    vbound = maximum(abs.(w))
    co = contourf!(xx,xx,w, levels= range(-vbound, vbound, length = 20) ,colormap=:seismic)
    Colorbar(figure[1, 4],co)
return figure;
end;


function diagonalize_transmon(Nt::Int64,Nx::Int64,ec::Float64,ej::Float64,x_l)
    dx = x_l[2] - x_l[1]
    x = spdiagm(0 => ComplexF64.(cos.(x_l)))
    ∂x = 1 / (2 * dx) * spdiagm(-1 => -ones(ComplexF64, Nx - 1), 1 => ones(Nx -1))
    ∂x2 = 1 / (dx^2) * spdiagm(-1 => ones(ComplexF64, Nx - 1), 0 => -2 * ones(Nx), 1 => ones(Nx -1))
    idx = [1, Nx]; idy = [Nx, 1]; v = [1/ (dx^2) , 1/ (dx^2) ];
    ∂x2= ∂x2 + sparse(idx,idy,v)
    V = - ej * x
    H = -∂x2 * 4*ec + V
    vals_DWP, vecs_DWP, U = eigsolve(H, k=Nt, sigma=minimum(real.(V)))
    vals_DWP = real.(vals_DWP)
    x_neg_idxs = findall(x -> x < 0, x_l);
    idx = findmax(abs2,vecs_DWP[1][x_neg_idxs])[2];
    vecs_DWP[1] .*= exp(-1im * angle(vecs_DWP[1][idx]));
    vecs_DWP[2] .*= exp(-1im * angle(vecs_DWP[2][idx]));
    return vals_DWP, vecs_DWP, U,V,∂x
end


