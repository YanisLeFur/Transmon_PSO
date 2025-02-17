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
    using SpecialFunctions
    using Dates
    include("../src/utilities.jl")
    include("../src/pulse.jl")
    include("../src/hamiltonian.jl")

    BLAS.set_num_threads(1)
    CUDA.allowscalar(false)
    global const op = cu


    #superparameters to be chosen
    global const Nt = 10
    global const Nx = 1000
    global const Nr = 200
    global const nsteps = 51
    global const lambda_ = 0.1
    global const hamitlonian = disp_c_rwa_d



    function simulation(params)
            ec = 2*pi*0.315
            ej = 51*ec
            g = params[6]
            k_qubit = 2*pi*0.000008
            omega_d = params[5]
            transmon = Transmon(Nt,Nx,ec,ej)
            resonator = Resonator(Nr,params[1],g)
            L0,state0,state1,omega_r_dressed = disp_c_rwa_d(transmon,resonator,[params[2],k_qubit],omega_d,op)        
            
            a = kron(resonator.a,eye(transmon.Nt))
            
            tlist = LinRange(0.0,params[3],nsteps)
            amp_drive(p, t) =  1/2*square_gaussian(t,p.tau,p.sigma,p.risefall_sigma_ratio,p.amp) 
            amp_drive_exp1(p, t) =  1/2*square_gaussian(t,p.tau,p.sigma,p.risefall_sigma_ratio,p.amp) .* exp(-2im*p.omega_d*t)
            amp_drive_exp2(p, t) =  1/2*square_gaussian(t,p.tau,p.sigma,p.risefall_sigma_ratio,p.amp) .* exp(2im*p.omega_d*t)
            
            L_drive1 = liouvillian(-(a'+a))|> op
            L_drive2 = liouvillian((a))|> op
            L_drive3 = liouvillian((a'))|> op

            Lt = (L0,(L_drive1,amp_drive),(L_drive2,amp_drive_exp1),(L_drive3,amp_drive_exp2))
            params_drive = (tau = params[3],sigma = 1.5,risefall_sigma_ratio = 4.0,amp = params[4],omega_d = params[5])
        
            sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive);
            sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive);
        
            return sol0.states,sol1.states
        end
        
    function simulation_rotated(params)
            ec = 2*pi*0.315
            ej = 51*ec
            k_qubit = 2*pi*0.000008
            omega_d = params[5]
    
            transmon = Transmon(Nt,Nx,ec,ej)
            resonator = Resonator(Nr,params[1],params[6])
            Ht, nt, nt_tr, a = kron_system(transmon, resonator)
            nt_p = tensor(qeye(resonator.Nr), triu(nt_tr))
            c_ops = [sqrt(params[2])*a,sqrt(k_qubit)*nt_p]
    
            
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
            tlist = LinRange(0.0,params[3],nsteps)
            params_drive = (tau = params[3],sigma = 1.5,risefall_sigma_ratio = 4.0,amp = params[4],omega_d = params[5])
            sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive);
            sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive);
            return sol0.states,sol1.states
        end
        



    function simulation_rotated_rwa(params)
        ec = 2*pi*0.315
        ej = 51*ec
        k_qubit = 2*pi*0.000008
        omega_d = params[5]

        transmon = Transmon(Nt,Nx,ec,ej)
        resonator = Resonator(Nr,params[1],params[6])
        
        
        
        Ht, nt, nt_tr, a = kron_system(transmon, resonator)
        nt_p = tensor(qeye(resonator.Nr), triu(nt_tr))
        c_ops = [sqrt(params[2])*a,sqrt(k_qubit)*nt_p]

        
        H = Ht + resonator.omega_r * a' * a - 1im * resonator.g *(nt_p'*a - nt_p*a') 
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
        L1 = liouvillian( -1im * resonator.g * nt_p' * a)|> op
        L2 = liouvillian( 1im * resonator.g * nt_p * a')|> op
        L_drive1 = liouvillian(-(a'+a))|> op
        L_drive2 = liouvillian((a))|> op
        L_drive3 = liouvillian((a'))|> op
        Lt = QobjEvo((L0,(L1,exp1),(L2,exp2),(L_drive1,amp_drive),(L_drive2,amp_drive_exp1),(L_drive3,amp_drive_exp2)))

        # state evolution
        tlist = LinRange(0.0,params[3],nsteps)
        params_drive = (tau = params[3],sigma = 1.5,risefall_sigma_ratio = 4.0,amp = params[4],omega_d = params[5])
        sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive);
        sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive);
        a  = cu(a)
        expect0 = expect(a,sol0.states)
        expect1 = expect(a,sol1.states)
        purity0 = tr.(sol0.states.*sol0.states)
        purity1 = tr.(sol1.states.*sol1.states)
        return purity0,purity1,expect0,expect1
    end

end


  
params=[48.115365638039606, 0.0113818, 71.46039823958353, 0.0, 47.98274583922244, 0.8181832903055553]#[48.115365638039606, 0.0113818, 71.46039823958353, 1.57, 47.98274583922244, 0.8181832903055553]
states0,states1 = simulation_rotated(params)
states0 = dense_to_sparse.(Matrix.(states0))
states1 = dense_to_sparse.(Matrix.(states1))
jldopen("data/simulation_no_drive$(Dates.now()).jld2", "a+") do file
    file["states0"] = states0
    file["states1"] = states1


end