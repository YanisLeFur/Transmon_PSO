using QuantumToolbox


function jc_tls(delta_qr,omega_q,delta_rd,g,a,s_z,s_m)
    """
    Hamiltonian of the JC (cavity-TLS) in the rotating picture of the drive
    Args:
        - delta_qr: detuning between the qubit and resonator (Float64)
        - omega_q: frequency of the qubit (Float64)
        - delta_rd: detuning between the resonator and the drive (Float64)
        - g: cavity-qubit coupling  (Float64)
        - a: destruction operator of the cavity (QuantumObject)
        - s_z: pauli z matrix (QuantumObject)
        - s_m: pauli minus matrix (QuantumObject)

    Returns:
        - JC two-level system hamiltonian (QuantumObject)
    """
    omega_r =  omega_q - delta_qr
    omega_d = delta_rd + omega_r
    delta_qd = omega_q - omega_d
    H_0 = delta_rd*a'*a + delta_qd/2*s_z + (g'*a*s_m + g*a'*s_m');
    return dense_to_sparse(H_0);
end;


function chi_ij(g_ji::Complex64,omega_r::Float64,omega_i::Float64,omega_j::Float64)::Float64
    """ 
    Compute the dispersive shift for the dispersive hamiltonian
    Args:
        - g_ji(Complex64): coupling between the transmon state i and j
        - omega_r(Float64): frequency of the resonator
        - omega_i(Float64): frequency of the transmon state i
        - omega_j(Float64): frequency of the transmon state j
    Returns:
        - Diserpsive shift (Float64)
    
    """
    return abs2(g_ji)/(omega_j-omega_i-omega_r)    
end

function dispersive_tls(omega_r::Float64,omega_q::Float64,chi::Float64,a::QuantumObject,s_z::QuantumObject)::QuantumObject
    """
    Generate the effective TLS hamiltonian of a transmon-cavity system with dispersive coupling and RWA applied on the drive
    Args:
        - omega_r(Float64): frequency of the cavity
        - omega_q(Float64): frequency of the TLS
        - chi(Float64): dispersive shift of the TLS
        - a(QuantumObject): annihilation operator of the cavity
        - s_z(QuantumObject): pauli matrix z associated to the TLS

    Returns:
        - TLS-cavity in dispersive coupling hamiltonian (QuantumObject)
    
    """
    return omega_r*a'*a+omega_q*s_z/2 + chi*a'*a*s_z
end

function disp_c_rwa_d(omega_r::Float64,omega_d::Float64,omega_t::Vector{Float64},g_mat::QuantumObject,a::QuantumObject,Nt::Int64,Nc::Int64)::QuantumObject    
    """
    Prepare the hamiltonian of a transmon coupled to a cavity with a dispersive shift coupling and RWA applied to the drive
    Args:
        - omega_r(Float64):  frequency of the cavity
        - omega_d(Float64): frequency of the drive
        - omega_t(Vecot{Float64}): frequencies of the transmon
        - g_mat(QuantumObject): coupling matrix between the cavity and the transmon manifold
        - a(QuantumObject): annihilation operator of the cavity
        - Nt(Int64): cut-off number of the transmon
        - Nc(Int64): cut-off number of the cavity
    Returns:
        - dispersive hamiltonian with RWA on drive (QuantumObject)
    """
    
    H_0 = (omega_r-omega_d)*a'*a
    chi = [sum([chi_ij(g_mat[j,i],omega_r, vals_DWP[i], vals_DWP[j])-chi_ij(g_mat[i,j],omega_r,vals_DWP[j],vals_DWP[i]) for i in 1:Nt] ) for j in 1:Nt]
    lamb_ = [ sum([chi_ij(g_mat[j,i],omega_r,vals_DWP[j],vals_DWP[i]) for i in 1:Nt]) for j in 1:Nt]
    for i in 1:Nt
        H_0 = H_0 +(omega_t[i] + chi[i]*a'a + lamb_[i])*kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
    end
    return dense_to_sparse(H_0);
end


function no_rwa_c_no_rwa_d(omega_t::Vector{Float64},delta::Float64,g::Float64,n::QuantumObject,a::QuantumObject)::QuantumObject
    """
    Compute the transmon hamiltonian without any approximation
    Args:
        - omega_t(Vector{Float64}): frequencies of the transmon
        - delta(Float64): detuning between the transmon and the resonator
        - g(Float64): coupling value between the resonator and the transmon
        - n(QuantumObject): charge operator of the transmon manifold
        - a(QuantumObject): annihilation operator of the cavity
    Returns:
        - Transmon-cavity hamiltonian without approximation (QuantumObject)
    """
    omega_q = omega_t[2]-omega_t[1]
    omega_r = omega_q - delta
    H_0 = omega_r * a'*a - 1im*g*n*(a'-a)
    for i in 1:Nt
        H_0 = H_0 + omega_t[i] * kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
    end
    return dense_to_sparse(H_0);
end



function rwa_c_no_rwa_d(omega_t::Vector{Float64},delta::Float64,g::QuantumObject,b::QuantumObject,a::QuantumObject)::QuantumObject
    """
    Compute the transmon hamiltonian with RWA on the coupling
    Args:
        - omega_t(Vector{Float64}): frequencies of the transmon
        - delta(Float64): detuning between the transmon and the resonator
        - g(Float64): coupling value between the resonator and the transmon
        - n(QuantumObject): charge operator of the transmon manifold
        - a(QuantumObject): annihilation operator of the cavity
    Returns:
        - Transmon-cavity hamiltonian with RWA on the coupling (QuantumObject)
    """
    omega_q = omega_t[2]-omega_t[1]
    omega_r = omega_q - delta
    H_0 = omega_r * a'*a + (g'*b'*a+g*a'*b)
    for i in 1:Nt
        H_0 = H_0 + omega_t[i] * kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
    end
    return dense_to_sparse(H_0);

end



function rwa_c_rwa_d(omega_t::Vector{Float64},delta::Float64,omega_d::Float64,g::QuantumObject,b::QuantumObject,a::QuantumObject)::QuantumObject
    """
    Compute the transmon hamiltonian with RWA on the coupling and drive
    Args:
        - omega_t(Vector{Float64}): frequencies of the transmon
        - delta(Float64): detuning between the transmon and the resonator
        - g(Float64): coupling value between the resonator and the transmon
        - n(QuantumObject): charge operator of the transmon manifold
        - a(QuantumObject): annihilation operator of the cavity
    Returns:
        - Transmon-cavity hamiltonian with RWA on the coupling and drive (QuantumObject)
    """
    omega_q = omega_t[2]-omega_t[1]
    omega_r = omega_q - delta
    H_0 = (omega_r-omega_d) * a'*a + (g'*b'*a+g*a'*b)
    for i in 1:Nt
        H_0 = H_0 + (omega_t[i]-omega_d) * kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
    end
    return dense_to_sparse(H_0);

end