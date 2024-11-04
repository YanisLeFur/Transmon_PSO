using QuantumToolbox


function dispersive_tls(omega_q::Float64,delta_rq::Float64,delta_rd::Float64,g::Float64,a::QuantumObject,s_z::QuantumObject)::QuantumObject
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
