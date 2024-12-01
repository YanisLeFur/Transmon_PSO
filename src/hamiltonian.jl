using QuantumToolbox

Base.@kwdef mutable struct Transmon
    Nt::Integer
    Nx::Integer
    ec::Float64
    ej::Float64
    E::Vector{Float64}
    V::Vector{QuantumObject{Vector{ComplexF64}, KetQuantumObject, 1}}
    U::Matrix{ComplexF64}
    n::QuantumObject{SparseMatrixCSC{ComplexF64, Int64}, OperatorQuantumObject, 1}
    H::QuantumObject{SparseMatrixCSC{ComplexF64, Int64}, OperatorQuantumObject, 1}

end

function Transmon(Nt::Integer, Nx::Integer, ec::Float64, ej::Float64)
    E, V, U, n, H = diagonalize_transmon(Nt, Nx, ec, ej)
    return Transmon(Nt, Nx, ec, ej, E, V, dense_to_sparse(U), n, dense_to_sparse(H))
end

Base.@kwdef mutable struct Resonator
    Nr::Integer
    omega_r::Float64
    a::QuantumObject
    function Resonator(Nr::Integer, omega_r::Float64)
    new(Nc, omega_r, destroy(Nr)) 
end
end


"""
Liouvillian (time-inde) for:
    - coupling (no rwa) drive (no rwa)                      (DONE)
    - coupling (rwa) drive (no rwa)                         (DONE)
    - coupling (no rwa) drive (rwa)                         (TO DO)
    - coupling (rwa) drive (rwa)                            (DONE)
    - coupling (dispersive) drive (no rwa)                  (DONE)
    - coupling (dispersive) drive (rwa)                     (DONE)
Keep also the TLS
"""

function kron_system(transmon::Transmon,resonator::Resonator)
    nt_tr = (transmon.U)'*transmon.n.data*transmon.U |> Qobj
    Ht_truncated = (transmon.U') * transmon.H.data * transmon.U |> Qobj
    Ht_truncated = (Ht_truncated + Ht_truncated') / 2 - real(transmon.E[1])
    a = tensor(resonator.a, qeye(transmon.Nt))
    nt = kron(qeye(resonator.Nr), nt_tr)
    nt = (nt + nt') / 2 
    Ht = tensor(qeye(resonator.Nr), Ht_truncated)
    return Ht,nt,nt_tr,a
end

function L0_coupling_norwa_drive_norwa(transmon::Transmon,resonator::Resonator,g,c_coeff,op)
    Ht,nt,nt_tr,a = kron_system(transmon,resonator)
    nt_p = tensor(qeye(Nc), triu(nt_tr))
    H = Ht + resonator.omega_r * a' * a - 1im * g * nt * (a - a')
    c_ops = [sqrt(c_coeff[1])*a,sqrt(c_coeff[2])*nt_p]
    L0 = liouvillian(H, c_ops) |> op
    return L0
end

function L0_coupling_rwa_drive_norwa(transmon::Transmon,resonator::Resonator,g,c_coeff,op)
    Ht,_,nt_tr,a = kron_system(transmon,resonator)
    nt_p = tensor(qeye(Nc), triu(nt_tr))
    Hrwa = Ht + resonator.omega_r * a' * a + 1im * g * nt_p * a' - 1im * g * nt_p' * a
    c_ops = [sqrt(c_coeff[1])*a,sqrt(c_coeff[2])*nt_p]
    L0 = liouvillian(Hrwa, c_ops) |> op
    return L0
end

function L0_coupling_rwa_drive_rwa(transmon::Transmon,resonator::Resonator,g,omega_d,c_coeff,op)
    Ht,_,nt_tr,a = kron_system(transmon,resonator)
    nt_p = tensor(qeye(resonator.Nr), triu(nt_tr))
    Hrwa = Ht + resonator.omega_r * a' * a + 1im * g * nt_p * a' - 1im * g * nt_p' * a
    Hrwa = Hrwa - omega_d * a' * a - omega_d* tensor(qeye(resonator.Nr), Qobj(Diagonal(0:transmon.Nt-1)))
    c_ops = [sqrt(c_coeff[1])*a,sqrt(c_coeff[2])*nt_p]
    L0 = liouvillian(Hrwa, c_ops) |> op
    return L0
end




# function dispersive_tls(omega_r,omega_q,chi,a,s_z)
                       
#     """
#     Generate the effective TLS hamiltonian of a transmon-cavity system with dispersive coupling and RWA applied on the drive
#     Args:
#         - omega_r(Float64): frequency of the cavity
#         - omega_q(Float64): frequency of the TLS
#         - chi(Float64): dispersive shift of the TLS
#         - a(QuantumObject): annihilation operator of the cavity
#         - s_z(QuantumObject): pauli matrix z associated to the TLS

#     Returns:
#         - TLS-cavity in dispersive coupling hamiltonian (QuantumObject)
    
#     """
#     return omega_r*a'*a + omega_q*s_z/2 + chi*a'*a*s_z
# end

function dispersive_tls(omega_q::Float64,delta_rq::Float64,delta_rd::Float64,g,a::QuantumObject,s_z::QuantumObject)::QuantumObject
    """ 
    Hamiltonian of the tls coupled to a cavity in the dispersive limit
    Args:
        - omega_q: frequency of the qubit (Float64)
        - delta_rq: detuning between the resonator and cavity (Float64)
        - delta_rd: detuning between the drive and the resonator (Float64)
        - g: cavity-qubit coupling  (Float64)
        - a: destruction operator of the cavity (QuantumObject)
        - s_z: pauli z matrix (QuantumObject)
    Returns:
        - dispersive two-level system hamiltonian (QuantumObject)
    """
    chi = g^2/delta_rq;
    omega_q_prime = omega_q + chi;
    return (delta_rd + chi * s_z) * a'*a + omega_q_prime * s_z/2;
end;



# function dispersive_tls(omega_q,delta,g,ec,omega_d,a,s_z)
#     """
#     Effective TLS of the SW approximation in the dispersive regime.
#     Args:
#         - omega_q :frequency of the qubit (Float64)
#         - delta : detuning among the bare frequency of the qubit and the resonator (Float64)
#         - g : cavity-qubit coupling (Float64)
#         - ec : capacitance energy (Float64)
#         - omega_d : drive frequenvy of the laser (Float64)
#         - a : annihilation operator of the resonator (QuantumObject)
#         - s_z : pauli matrix z of the qubit (QuantumObject)
#     Returns:
#         - The effective 2-level system of the transmon-cavity hamiltonian under SW approximation (QuantumObject)
#     """
#     omega_q_prime = omega_q + g^2/delta
#     chi = (-(g^2*ec)/(delta*(delta-ec)))
#     omega_r = omega_q - delta 
#     omega_r_prime = omega_r - g^2/(delta- ec)
#     delta_rd = omega_r_prime-omega_d
#     H_0 =  (delta_rd + chi * s_z) * a'*a + omega_q_prime * s_z/2
#     return dense_to_sparse(H_0);
# end;




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


function chi_ij(g_ji,omega_r::Float64,omega_i::Float64,omega_j::Float64)::Float64
    """ 
    Compute the dispersive shift for the dispersive hamiltonian
    Args:
        - g_ji: coupling between the transmon state i and j
        - omega_r(Float64): frequency of the resonator
        - omega_i(Float64): frequency of the transmon state i
        - omega_j(Float64): frequency of the transmon state j
    Returns:
        - Diserpsive shift (Float64)
    
    """
    return abs2(g_ji)/(omega_j-omega_i-omega_r)    
end



function disp_c_rwa_d(omega_r,omega_d,omega_t,g_mat,a,Nt,Nc)    
    """
    Prepare the hamiltonian of a transmon coupled to a cavity with a dispersive shift coupling and RWA applied to the drive
    Args:
        - omega_r(Float64):  frequency of the cavity
        - omega_d(Float64): frequency of the drive
        - omega_t(Vecot{Float64}): frequencies of the transmon
        - g_ma): coupling matrix between the cavity and the transmon manifold
        - a(QuantumObject): annihilation operator of the cavity
        - N): cut-off number of the transmon
        - Nc(Int64): cut-off number of the cavity
    Returns:
        - dispersive hamiltonian with RWA on drive (QuantumObject)
    """
    
    H_0 = (omega_r-omega_d)*a'*a
    chi = [sum([chi_ij(g_mat[j,i],omega_r, omega_t[i], omega_t[j])-chi_ij(g_mat[i,j],omega_r,omega_t[j],omega_t[i]) for i in 1:Nt] ) for j in 1:Nt]
    lamb_ = [ sum([chi_ij(g_mat[j,i],omega_r,omega_t[j],omega_t[i]) for i in 1:Nt]) for j in 1:Nt]
    for i in 1:Nt
        H_0 = H_0 +(omega_t[i] + chi[i]*a'a + lamb_[i])*kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
    end
    return dense_to_sparse(H_0);
end



function disp_c_norwa_d(omega_r,omega_t,g_mat,a,Nt,Nc)    
    """
    Prepare the hamiltonian of a transmon coupled to a cavity with a dispersive shift coupling and RWA applied to the drive
    Args:
        - omega_r(Float64):  frequency of the cavity
        - omega_t(Vecot{Float64}): frequencies of the transmon
        - g_mat(QuantumObject): coupling matrix between the cavity and the transmon manifold
        - a(QuantumObject): annihilation operator of the cavity
        - Nt(Int64): cut-off number of the transmon
        - Nc(Int64): cut-off number of the cavity
    Returns:
        - dispersive hamiltonian with RWA on drive (QuantumObject)
    """
    
    H_0 = omega_r*a'*a
    chi = [sum([chi_ij(g_mat[j,i],omega_r, omega_t[i], omega_t[j])-chi_ij(g_mat[i,j],omega_r,omega_t[j],omega_t[i]) for i in 1:Nt] ) for j in 1:Nt]
    lamb_ = [ sum([chi_ij(g_mat[j,i],omega_r,omega_t[j],omega_t[i]) for i in 1:Nt]) for j in 1:Nt]
    for i in 1:Nt
        H_0 = H_0 +(omega_t[i] + chi[i]*a'a + lamb_[i])*kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
    end
    return dense_to_sparse(H_0);
end


# function no_rwa_c_no_rwa_d(omega_t::Vector{Float64},delta::Float64,g::Float64,n::QuantumObject,a::QuantumObject)::QuantumObject
#     """
#     Compute the transmon hamiltonian without any approximation
#     Args:
#         - omega_t(Vector{Float64}): frequencies of the transmon
#         - delta(Float64): detuning between the transmon and the resonator
#         - g(Float64): coupling value between the resonator and the transmon
#         - n(QuantumObject): charge operator of the transmon manifold
#         - a(QuantumObject): annihilation operator of the cavity
#     Returns:
#         - Transmon-cavity hamiltonian without approximation (QuantumObject)
#     """
#     omega_q = omega_t[2]-omega_t[1]
#     omega_r = omega_q - delta
#     H_0 = omega_r * a'*a - 1im*g*n*(a'-a)
#     for i in 1:Nt
#         H_0 = H_0 + omega_t[i] * kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
#     end
#     return dense_to_sparse(H_0);
# end



# function rwa_c_no_rwa_d(omega_t::Vector{Float64},delta::Float64,g::QuantumObject,b::QuantumObject,a::QuantumObject)::QuantumObject
#     """
#     Compute the transmon hamiltonian with RWA on the coupling
#     Args:
#         - omega_t(Vector{Float64}): frequencies of the transmon
#         - delta(Float64): detuning between the transmon and the resonator
#         - g(Float64): coupling value between the resonator and the transmon
#         - n(QuantumObject): charge operator of the transmon manifold
#         - a(QuantumObject): annihilation operator of the cavity
#     Returns:
#         - Transmon-cavity hamiltonian with RWA on the coupling (QuantumObject)
#     """
#     omega_q = omega_t[2]-omega_t[1]
#     omega_r = omega_q - delta
#     H_0 = omega_r * a'*a + (g'*b'*a+g*a'*b)
#     for i in 1:Nt
#         H_0 = H_0 + omega_t[i] * kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
#     end
#     return dense_to_sparse(H_0);

# end



# function rwa_c_rwa_d(omega_t::Vector{Float64},delta::Float64,omega_d::Float64,g::QuantumObject,b::QuantumObject,a::QuantumObject)::QuantumObject
#     """
#     Compute the transmon hamiltonian with RWA on the coupling and drive
#     Args:
#         - omega_t(Vector{Float64}): frequencies of the transmon
#         - delta(Float64): detuning between the transmon and the resonator
#         - g(Float64): coupling value between the resonator and the transmon
#         - n(QuantumObject): charge operator of the transmon manifold
#         - a(QuantumObject): annihilation operator of the cavity
#     Returns:
#         - Transmon-cavity hamiltonian with RWA on the coupling and drive (QuantumObject)
#     """
#     omega_q = omega_t[2]-omega_t[1]
#     omega_r = omega_q - delta
#     H_0 = (omega_r-omega_d) * a'*a + (g'*b'*a+g*a'*b)
#     for i in 1:Nt
#         H_0 = H_0 + (omega_t[i]-omega_d) * kron(eye(Nc),fock(Nt,i-1)*fock(Nt,i-1)')  
#     end
#     return dense_to_sparse(H_0);

# end