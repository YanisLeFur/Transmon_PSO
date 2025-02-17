# Perform here the rotating picture and compare the time with the unrotated version

using Distributed
using ClusterManagers

const SLURM_NUM_TASKS = parse(Int, ENV["SLURM_NTASKS"])
const SLURM_CPUS_PER_TASK = parse(Int, ENV["SLURM_CPUS_PER_TASK"])

exeflags = ["--project=$(Base.active_project())", "-t $SLURM_CPUS_PER_TASK"]
addprocs(SlurmManager(SLURM_NUM_TASKS); exeflags=exeflags, topology=:master_worker)

@everywhere begin
    using CUDA
    using QuantumToolbox
    using JLD2

    include("../src/utilities.jl")
    include("../src/pulse.jl")
    include("../src/hamiltonian.jl")

    BLAS.set_num_threads(1)
    CUDA.allowscalar(false)
    global const op = cu


    #superparameters to be chosen
    global const Nt = 5
    global const Nx = 1000
    global const Nr = 60
    global const nsteps = 101
    global const hamitlonian = coupling_norwa_drive_norwa


    params = [44.164351879173616, 0.014831480168132807, 200.0, 0.596202092811838, 44.302671283818135]

    function simulation_unrotated(params)
        ec = 2*pi*0.315
        ej = 51*ec
        g = 2*pi*0.15
        k_qubit = 2*pi*0.000008
        transmon = Transmon(Nt,Nx,ec,ej)
        resonator = Resonator(Nr,params[1],g)
        L0,state0,state1,omega_r_dressed = hamitlonian(transmon,resonator,[params[2],k_qubit],op)
        a = kron(resonator.a,eye(transmon.Nt))
        L1 = liouvillian(a'-a) |> op
        tlist = LinRange(0.0,params[3],nsteps)
        drive(p,t) = 1im*square_gaussian(t,p.tau,p.sigma,p.risefall_sigma_ratio,p.amp).* sin(p.omega_d * t);
        Lt = (L0,(L1,drive))
        params_drive = (tau = params[3],sigma = 1.5,risefall_sigma_ratio = 4.0,amp = params[4],omega_d = params[5])

        sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive);
        sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive);

        a = cu(a)
        betas_0 = expect(a,sol0.states)
        betas_1 = expect(a,sol1.states)
        diff_beta = abs.(betas_0 .- betas_1) .^2



        return diff_beta
    end



    function simulation_rotated(params)
        ec = 2*pi*0.315
        ej = 51*ec
        g = 2*pi*0.15
        k_qubit = 2*pi*0.000008
        omega_d = params[5]

        transmon = Transmon(Nt,Nx,ec,ej)
        resonator = Resonator(Nr,params[1],g)
        Ht, nt, nt_tr, a = kron_system(transmon, resonator)
        nt_p = tensor(qeye(resonator.Nr), triu(nt_tr))
        c_ops = [sqrt(params[2])*a,sqrt(k_qubit)*nt_p]

        
        H = Ht + resonator.omega_r * a' * a - 1im * resonator.g * nt * (a - a') 
        E,V = eigenstates(H , sparse=true, k=3, sigma=-transmon.ej - 0.5)
        state0 = Qobj((V[1])) |>op
        state1 = Qobj((V[2])) |>op
        omega_r_dressed = E[3]

        H = Ht + (resonator.omega_r-omega_d) * a' * a 
        H = dense_to_sparse(H)

        amp_drive(p, t) =  1/2*square_gaussian(t,p.tau,p.sigma,p.risefall_sigma_ratio,p.amp) 
        amp_drive_exp1(p, t) =  1/2*square_gaussian(t,p.tau,p.sigma,p.risefall_sigma_ratio,p.amp) .* exp(-2im*p.omega_d*t)
        amp_drive_exp2(p, t) =  1/2*square_gaussian(t,p.tau,p.sigma,p.risefall_sigma_ratio,p.amp) .* exp(2im*p.omega_d*t)
        exp1(p, t) =  exp(-1im * p.omega_d * t)
        exp2(p, t) =  exp(1im * p.omega_d * t)

        L0 = (liouvillian(H, c_ops)) |> op
        L1 = liouvillian( -1im * resonator.g * nt * a)|> op
        L2 = liouvillian( 1im * resonator.g * nt * a')|> op
        L_drive1 = liouvillian(-(a'+a))|> op
        L_drive2 = liouvillian((a))|> op
        L_drive3 = liouvillian((a'))|> op
        Lt = QobjEvo((L0,(L1,exp1),(L2,exp2),(L_drive1,amp_drive),(L_drive2,amp_drive_exp1),(L_drive3,amp_drive_exp2)))

        tlist = LinRange(0.0,params[3],nsteps)
        params_drive = (tau = params[3],sigma = 1.5,risefall_sigma_ratio = 4.0,amp = params[4],omega_d = params[5])
        sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive);
        sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive);

        a = cu(a)
        betas_0 = expect(a,sol0.states)
        betas_1 = expect(a,sol1.states)
        diff_beta = abs.(betas_0 .- betas_1) .^2

        return diff_beta
    end

end




diff_beta_rot = simulation_rotated(params)
diff_beta = simulation_unrotated(params)
println(maximum(abs.(diff_beta_rot.-diff_beta)))
println(abs.(diff_beta_rot.-diff_beta))

jldopen("data/rotating_picture.jld2", "a+") do file
file["rot"] = diff_beta_rot
file["unrot"] = diff_beta
end
