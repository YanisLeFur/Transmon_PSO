using QuantumToolbox


function dispersive_tls(omega_q::Float64,delta_rq::Float64,delta_rd::Float64,g::Float64,a::QuantumObject,s_z::QuantumObject)::QuantumObject
    """ 
    Hamiltonian of the tls coupled to a cavity in the dispersive limit and in the rotating picture of the drive
    Args:
        - omega_q: frequency of the qubit (Float64)
        - delta_rq: detuning between the resonator and qubit (Float64)
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

function jc_tls(delta_qr::Float64,omega_q::Float64,delta_rd::Float64,g::Float64,a::QuantumObject,s_z::QuantumObject,s_m::QuantumObject)::QuantumObject
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
        - JC two-level system hamiltonian
    """
    omega_r =  omega_q - delta_qr
    omega_d = delta_rd + omega_r
    delta_qd = omega_q - omega_d
    H_0 = delta_rd*a'*a + delta_qd/2*s_z + g * (a*s_m + a'*s_m');
    return H_0;
end;

