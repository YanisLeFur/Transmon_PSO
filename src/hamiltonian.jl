using QuantumToolbox
include("utilities.jl")

Base.@kwdef mutable struct Transmon
    Nt::Integer
    Nx::Integer
    ec::Float64
    ej::Float64
    E::Vector{Float64}
    V::Vector{QuantumObject{Vector{ComplexF64},KetQuantumObject,1}}
    U::Matrix{ComplexF64}
    n::QuantumObject{SparseMatrixCSC{ComplexF64,Int64},OperatorQuantumObject,1}
    H::QuantumObject{SparseMatrixCSC{ComplexF64,Int64},OperatorQuantumObject,1}

end

function Transmon(Nt::Integer, Nx::Integer, ec::Float64, ej::Float64)
    E, V, U, n, H = diagonalize_transmon(Nt, Nx, ec, ej)
    return Transmon(Nt, Nx, ec, ej, E, V, dense_to_sparse(U), n, dense_to_sparse(H))
end

Base.@kwdef mutable struct Resonator
    Nr::Integer
    omega_r::Float64
    g::Float64
    a::QuantumObject
    function Resonator(Nr::Integer, omega_r::Float64, g::Float64)
        new(Nr, omega_r, g, destroy(Nr))
    end
end


function kron_system(transmon::Transmon, resonator::Resonator)
    nt_tr = dense_to_sparse((transmon.U)' * transmon.n.data * transmon.U) |> Qobj
    Ht_truncated = (transmon.U') * transmon.H.data * transmon.U |> Qobj
    Ht_truncated = (Ht_truncated + Ht_truncated') / 2 - real(transmon.E[1])
    a = tensor(resonator.a, qeye(transmon.Nt))
    nt = kron(qeye(resonator.Nr), nt_tr)
    nt = (nt + nt') / 2
    Ht = tensor(qeye(resonator.Nr), Ht_truncated)
    return Ht, nt, nt_tr, a
end

function coupling_norwa_drive_norwa(transmon::Transmon, resonator::Resonator, c_coeff, op)
    Ht, nt, nt_tr, a = kron_system(transmon, resonator)
    nt_p = tensor(qeye(resonator.Nr), triu(nt_tr))
    H = Ht + resonator.omega_r * a' * a - 1im * resonator.g * nt * (a - a')
    c_ops = [sqrt(c_coeff[1]) * a, sqrt(c_coeff[2]) * nt_p]
    L0 = op(liouvillian(H, c_ops))

    E, V = eigenstates(H, sparse=true, k=3, sigma=-transmon.ej - 0.5)
    state_0 = Qobj((V[1])) |> op
    state_1 = Qobj((V[2])) |> op
    omega_r_dressed = E[3]
    return L0, state_0, state_1, omega_r_dressed
end

function coupling_rwa_drive_norwa(transmon::Transmon, resonator::Resonator, c_coeff, op)
    Ht, _, nt_tr, a = kron_system(transmon, resonator)
    nt_p = tensor(qeye(resonator.Nr), triu(nt_tr))
    Hrwa = Ht + resonator.omega_r * a' * a + 1im * resonator.g * nt_p * a' - 1im * resonator.g * nt_p' * a
    c_ops = [sqrt(c_coeff[1]) * a, sqrt(c_coeff[2]) * nt_p]
    L0 = liouvillian(Hrwa, c_ops) |> op
    E, V = eigenstates(Hrwa, sparse=true, k=3, sigma=-transmon.ej - 0.5)
    state_0 = Qobj((V[1])) |> op
    state_1 = Qobj((V[2])) |> op
    omega_r_dressed = E[3]
    return L0, state_0, state_1, omega_r_dressed
end

function coupling_rwa_drive_rwa(transmon::Transmon, resonator::Resonator, c_coeff, omega_d, op)
    """
    Assume that the drive is on resonance ie omega_d = omega_r_dressed 
    """
    Ht, _, nt_tr, a = kron_system(transmon, resonator)
    nt_p = tensor(qeye(resonator.Nr), triu(nt_tr))
    Hrwa = Ht + resonator.omega_r * a' * a + 1im * resonator.g * nt_p * a' - 1im * resonator.g * nt_p' * a
    E, V = eigenstates(Hrwa, sparse=true, k=3, sigma=-transmon.ej - 0.5)
    state_0 = Qobj((V[1])) |> op
    state_1 = Qobj((V[2])) |> op
    omega_r_dressed = E[3]
    Hrwa = Hrwa - omega_d * (a' * a + tensor(qeye(resonator.Nr), Qobj(Diagonal(0:transmon.Nt-1))))
    c_ops = [sqrt(c_coeff[1]) * a, sqrt(c_coeff[2]) * nt_p]
    L0 = liouvillian(Hrwa, c_ops) |> op
    return L0, state_0, state_1, omega_r_dressed
end






function dispersive_tls(omega_q::Float64, delta_rq::Float64, delta_rd::Float64, g, a::QuantumObject, s_z::QuantumObject)::QuantumObject
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
    chi = g^2 / delta_rq
    omega_q_prime = omega_q + chi
    return (delta_rd + chi * s_z) * a' * a + omega_q_prime * s_z / 2
end;


function jc_tls(delta_qr, omega_q, delta_rd, g, a, s_z, s_m)
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
    omega_r = omega_q - delta_qr
    omega_d = delta_rd + omega_r
    delta_qd = omega_q - omega_d
    H_0 = delta_rd * a' * a + delta_qd / 2 * s_z + (g' * a * s_m + g * a' * s_m')
    return dense_to_sparse(H_0)
end;


function chi_ij(g_ji, omega_r::Float64, omega_i::Float64, omega_j::Float64)::Float64
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
    return abs2(g_ji) / (omega_j - omega_i - omega_r)
end



function disp_c_rwa_d(transmon, resonator, c_coeff, omega_d, op)

    """
    Assume drive on the resonance with cavity omega_d = omega_r_dressed
    """
    Ht, _, nt_tr, a = kron_system(transmon, resonator)
    Nt = transmon.Nt
    Nr = resonator.Nr
    E = transmon.E
    omega_r = resonator.omega_r
    g_mat = resonator.g * (transmon.U)' * transmon.n.data * transmon.U
    nt_p = tensor(qeye(Nr), triu(nt_tr))


    H_0 = omega_r * a' * a
    chi = [sum([chi_ij(g_mat[j, i], omega_r, E[i], E[j]) - chi_ij(g_mat[i, j], omega_r, E[j], E[i]) for i in 1:Nt]) for j in 1:Nt]
    lamb_ = [sum([chi_ij(g_mat[j, i], omega_r, E[j], E[i]) for i in 1:Nt]) for j in 1:Nt]
    for i in 1:Nt
        H_0 = H_0 + (E[i] + chi[i] * a'a + lamb_[i]) * kron(eye(Nr), fock(Nt, i - 1) * fock(Nt, i - 1)')
    end
    E, V = eigenstates(H_0, sparse=true, k=3, sigma=-transmon.ej - 0.5)
    state_0 = Qobj((V[1])) |> op
    state_1 = Qobj((V[2])) |> op
    omega_r_dressed = E[3]
    H_0 -= omega_d * a' * a
    c_ops = [sqrt(c_coeff[1]) * a, sqrt(c_coeff[2]) * nt_p]
    L0 = liouvillian(dense_to_sparse(H_0), c_ops) |> op
    return L0, state_0, state_1, omega_r_dressed
end



function disp_c_norwa_d(transmon, resonator, c_coeff, op)

    """
    Assume drive on the resonance with cavity omega_d = omega_r_dressed
    """
    Ht, _, nt_tr, a = kron_system(transmon, resonator)
    Nt = transmon.Nt
    Nr = resonator.Nr
    E = transmon.E
    omega_r = resonator.omega_r
    g_mat = resonator.g * (transmon.U)' * transmon.n.data * transmon.U
    nt_p = tensor(qeye(Nr), triu(nt_tr))


    H_0 = omega_r * a' * a
    chi = [sum([chi_ij(g_mat[j, i], omega_r, E[i], E[j]) - chi_ij(g_mat[i, j], omega_r, E[j], E[i]) for i in 1:Nt]) for j in 1:Nt]
    lamb_ = [sum([chi_ij(g_mat[j, i], omega_r, E[j], E[i]) for i in 1:Nt]) for j in 1:Nt]
    for i in 1:Nt
        H_0 = H_0 + (E[i] + chi[i] * a' * a + lamb_[i]) * kron(eye(Nr), fock(Nt, i - 1) * fock(Nt, i - 1)')
    end

    E, V = eigenstates(H_0, sparse=true, k=3, sigma=-transmon.ej - 0.5)
    state_0 = Qobj((V[1])) |> op
    state_1 = Qobj((V[2])) |> op
    omega_r_dressed = E[3]
    c_ops = [sqrt(c_coeff[1]) * a, sqrt(c_coeff[2]) * nt_p]
    L0 = liouvillian(dense_to_sparse(H_0), c_ops) |> op
    return L0, state_0, state_1, omega_r_dressed
end



function sausage(params)

    # Preparing ingredients
    transmon = Transmon(Nt, Nx, params[1], params[2])
    resonator = Resonator(Nr, params[3], params[4])
    L0, state0, state1, omega_r_dressed = hamitlonian(transmon, resonator, [params[5], params[6]], op)
    a = kron(resonator.a, eye(transmon.Nt))
    L1 = liouvillian(a' - a) |> op
    tlist = LinRange(0.0, params[7], nsteps)
    drive(p, t) = 1im * square_gaussian(t, p.tau, p.sigma, p.risefall_sigma_ratio, p.amp) .* sin(p.omega_d * t)
    Lt = (L0, (L1, drive))
    params_drive = (tau=params[7], sigma=params[8], risefall_sigma_ratio=params[9], amp=params[10], omega_d=params[11])

    # Making the sausage
    sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive)
    sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive)
    return distance_error(ptrace(sol0.states[end], 1), ptrace(sol1.states[end], 1))
end




function sausage_specs(Nt, Nx, Nr, nsetps, params)

    # Preparing ingredients
    transmon = Transmon(Nt, Nx, params[1], params[2])
    resonator = Resonator(Nr, params[3], params[4])

    L0, state0, state1, omega_r_dressed = hamitlonian(transmon, resonator, [params[5], params[6]], op)
    a = kron(resonator.a, eye(transmon.Nt))
    L1 = liouvillian(a' - a) |> op
    a = a |> op
    n_tr = dense_to_sparse((transmon.U)' * transmon.n.data * transmon.U) |> Qobj
    n = kron(eye(resonator.Nr), n_tr) |> op
    tlist = LinRange(0.0, params[7], nsteps)
    drive(p, t) = 1im * square_gaussian(t, p.tau, p.sigma, p.risefall_sigma_ratio, p.amp) .* sin(p.omega_d * t)
    Lt = (L0, (L1, drive))
    params_drive = (tau=params[7], sigma=params[8], risefall_sigma_ratio=params[9], amp=params[10], omega_d=params[11])

    # Making the sausage
    ts = time()
    sol0 = mesolve(Lt, state0, tlist, e_ops=[a' * a, n], saveat=tlist, params=params_drive)
    sol1 = mesolve(Lt, state1, tlist, e_ops=[a' * a, n], saveat=tlist, params=params_drive)
    tf = time() - ts
    # Saving the sausage
    fidelity = zeros(6)
    fidelity[1] = distance_error(ptrace(sol0.states[end], 1), ptrace(sol1.states[end], 1))
    fidelity[2] = maximum(abs.(sol0.expect[1, :]))
    fidelity[3] = maximum(abs.(sol1.expect[1, :]))
    fidelity[4] = maximum(abs.(sol0.expect[2, :]))
    fidelity[5] = maximum(abs.(sol1.expect[2, :]))
    fidelity[6] = tf
    return fidelity
end



function sausage_specs_rwa(Nt, Nx, Nr, nsetps, params)

    # Preparing ingredients
    transmon = Transmon(Nt, Nx, params[1], params[2])
    resonator = Resonator(Nr, params[3], params[4])

    L0, state0, state1, omega_r_dressed = hamitlonian(transmon, resonator, [params[5], params[6]], params[11], op)
    a = kron(resonator.a, eye(transmon.Nt))
    L1 = liouvillian(a' - a) |> op
    a = a |> op
    n_tr = dense_to_sparse((transmon.U)' * transmon.n.data * transmon.U) |> Qobj
    n = kron(eye(resonator.Nr), n_tr) |> op

    tlist = LinRange(0.0, params[7], nsteps)
    drive_rwa(p, t) = 1im / 2 * square_gaussian(t, p.tau, p.sigma, p.risefall_sigma_ratio, p.amp)
    Lt = (L0, (L1, drive_rwa))
    params_drive = (tau=params[7], sigma=params[8], risefall_sigma_ratio=params[9], amp=params[10], omega_d=params[11])

    # Making the sausage
    ts = time()
    sol0 = mesolve(Lt, state0, tlist, e_ops=[a' * a, n], saveat=tlist, params=params_drive)
    sol1 = mesolve(Lt, state1, tlist, e_ops=[a' * a, n], saveat=tlist, params=params_drive)
    tf = time() - ts
    # Saving the sausage
    specs = zeros(6)
    specs[1] = distance_error(ptrace(sol0.states[end], 1), ptrace(sol1.states[end], 1))
    specs[2] = maximum(abs.(sol0.expect[1, :]))
    specs[3] = maximum(abs.(sol1.expect[1, :]))
    specs[4] = maximum(abs.(sol0.expect[2, :]))
    specs[5] = maximum(abs.(sol1.expect[2, :]))
    specs[6] = tf
    return specs

end



function sausage_specs_effective(Nt, Nx, Nr, nsetps, params)

    # Preparing ingredients
    transmon = Transmon(Nt, Nx, params[1], params[2])
    resonator = Resonator(Nr, params[3], params[4])

    L0, state0, state1, omega_r_dressed = hamitlonian(transmon, resonator, [params[5], params[6]], op)
    a = kron(resonator.a, eye(transmon.Nt))
    L1 = liouvillian(a' - a) |> op
    a = a |> op
    n_tr = dense_to_sparse((transmon.U)' * transmon.n.data * transmon.U) |> Qobj
    n = kron(eye(resonator.Nr), n_tr) |> op
    tlist = LinRange(0.0, params[7], nsteps)
    drive(p, t) = 1im * square_gaussian(t, p.tau, p.sigma, p.risefall_sigma_ratio, p.amp) .* sin(p.omega_d * t)
    params_drive = (tau=params[7], sigma=params[8], risefall_sigma_ratio=params[9], amp=params[10], omega_d=params[11])

    # Generate the effective ingredients
    rho_0 = ket2dm(state0)
    rho_1 = ket2dm(state1)
    m = find_smallest_square(2 * length(rho_0))
    rho_eff = vec2mat(vcat(mat2vec(rho_0.data), mat2vec(rho_1.data), zeros(m))) |> Qobj
    zero_mat = spzeros(m, m)
    op_inv = SparseMatrixCSC

    L0_eff = Qobj(blockdiag(op_inv(L0.data), op_inv(L0.data), zero_mat), type=SuperOperator) |> op
    L1_eff = Qobj(blockdiag(op_inv(L1.data), op_inv(L1.data), zero_mat), type=SuperOperator) |> op
    L_eff = QobjEvo((L0_eff, (L1_eff, drive)))


    # Making the sausage
    ts = time()
    sol_eff = mesolve(L_eff, rho_eff, tlist, saveat=tlist, params=params_drive)
    tf = time() - ts

    # Saving the sausage
    fidelity = zeros(6)
    fidelity[1] = distance_error(ptrace(extract_dm(sol_eff.states[end], a.dims, 1), 1), ptrace(extract_dm(sol_eff.states[end], a.dims, 2), 1))
    fidelity[6] = tf
    return fidelity


end

