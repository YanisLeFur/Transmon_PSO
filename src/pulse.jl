
function tanh_pulse()
    """
    Generate a smooth square pulse using an hyperbolic tan function 
    Args:
        - slope: slope of the pulse 
        - tau: time of the pulse 
        - 
    Return:
        - pulse sequence (array)
    """
    pulse = 0;
    return pulse;
end;

function square_gaussian(t::Float64,tau::Float64,sigma::Float64,risefall_sigma_ratio::Float64,amp::Float64)::Float64
    """
    Generate a smooth square pulse using a gaussian function (from https://docs.quantum.ibm.com/api/qiskit/qiskit.pulse.library.GaussianSquare)
    Args:
        - sigma: variance of the gaussian 
        - tau: time of the pulse 
        - risefall_sigma_ratio: ratio between the variance of the gaussian and the risefall of the square pulse 
        - amp: amplitude of the pulse
        - num_points: number of points that describe the pulse 
    Return:
        - pulse sequence (array)
    """

    risefall =  risefall_sigma_ratio*sigma;
    width = tau - 2 * risefall;
    if width<0
        tau = tau - width;
        risefall =  risefall_sigma_ratio*sigma;
        width = tau - 2 * risefall;
    end;
    
    function prime(t)
        if t < risefall
            f_prime = exp(-0.5*(t-risefall)^2/sigma^2);
        elseif (risefall<t) && (t < risefall+width)
            f_prime = 1;
        else
            f_prime = exp(-0.5*(t-risefall-width)^2/sigma^2);
        end;

        return f_prime;
    end;
    return amp .* (prime.(t).-prime(-1))./(1-prime(-1)) ;
end;