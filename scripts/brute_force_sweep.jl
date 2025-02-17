using Distributed
using ClusterManagers
using IterTools

const SLURM_NUM_TASKS = parse(Int, ENV["SLURM_NTASKS"])
const SLURM_CPUS_PER_TASK = parse(Int, ENV["SLURM_CPUS_PER_TASK"])

exeflags = ["--project=$(Base.active_project())", "-t $SLURM_CPUS_PER_TASK"]
addprocs(SlurmManager(SLURM_NUM_TASKS); exeflags=exeflags, topology=:master_worker)

@everywhere begin
    using CUDA
    using QuantumToolbox
    using JLD2
    using SpecialFunctions
    using Dates

    include("../src/utilities.jl")
    include("../src/pulse.jl")
    include("../src/hamiltonian.jl")

    BLAS.set_num_threads(1)
    CUDA.allowscalar(false)
    global const op = cu


    #superparameters to be chosen
    global const Nt = 5
    global const Nx = 1000
    global const Nr = 400
    global const nsteps = 101
    global const lambda_1 = 0.1
        
    function sausage(ec,ej)
        g = 0.8181832903055553
        k_qubit = 2*pi*0.000008
        omega_r = 48.115365638039606
        kappa_r = 0.0113818
        tau = 71.46039823958353
        amp = 1.57
        omega_d = 47.98274583922244




        transmon = Transmon(Nt,Nx,ec,ej)
        resonator = Resonator(Nr,omega_r,g)
        g_mat = resonator.g * (transmon.U)' * transmon.n.data * transmon.U
        omega_q = transmon.E[2]-transmon.E[1]
        ncrit = abs2((omega_q-omega_r)/(2g_mat[1,2]))
        Ht, nt, nt_tr, a = kron_system(transmon, resonator)
        nt_p = tensor(qeye(resonator.Nr), triu(nt_tr))
        c_ops = [sqrt(kappa_r)*a,sqrt(k_qubit)*nt_p]

        
        H = Ht + resonator.omega_r * a' * a - 1im * resonator.g *nt*(a - a') 
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

        # state evolution
        tlist = LinRange(0.0,tau,nsteps)
        params_drive = (tau = tau,sigma = 1.5,risefall_sigma_ratio = 4.0,amp = amp,omega_d = omega_d)
        sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive);
        sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive);



        # loss function
        a = cu(a)
        SNR = -opposite_SNR(sol0.states,sol1.states,a,tlist,1.0,kappa_r)
        purity_1 = tr.(sol1.states.*sol1.states)
        purity_0 = tr.(sol0.states.*sol0.states)
        est_a0 = expect(a,sol0.states)
        est_a1 = expect(a,sol1.states)
        est_n0 = expect(a'*a,sol0.states)
        est_n1 = expect(a'*a,sol1.states)
        conv_test0 = expect(commutator(a,a'),sol0.states)
        conv_test1 = expect(commutator(a,a'),sol1.states)
        return SNR,purity_1,purity_0,est_a0,est_a1,est_n0,est_n1,conv_test0,conv_test1
    end
        


end

ec_opt = 2*pi*0.315
ej_opt = 51*ec_opt
ec_set = LinRange(ec_opt*0.95,ec_opt*1.05,10)
ej_set = LinRange(ej_opt*0.95,ej_opt*1.05,10)
ec_ej_set = [(ec, ej) for ec in ec_set, ej in ej_set]
results = pmap(ec_ej -> sausage(ec_ej[1],ec_ej[2]), ec_ej_set)

jldopen("data/mapping_ec_ej_$(Dates.now()).jld2", "w") do file
    file["results"] = results
    file["ec_set"] = ec_set
    file["ej_set"] = ej_set
end

