
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
    result1= sum(w1, dims=1)/sum(w1);
    w2  =wigner((ptrace(state2,1)), xx, xx);
    result2= sum(w2, dims=1)/sum(w2);
    overlap_vector = get_overlap(result1,result2);
    return sum(overlap_vector);
end;

function distance_fidelity(state1::QuantumObject, state2::QuantumObject)::Float64
    """
    Using the formula of error Perr from https://journals.aps.org/pra/pdf/10.1103/PhysRevA.77.032311
    F = 1-Perr
    """
    return 1-0.5*(1-0.5*norm(state1-state2));
end;;
