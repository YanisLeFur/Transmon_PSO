using Distributed
using ClusterManagers

const SLURM_NUM_TASKS = parse(Int, ENV["SLURM_NTASKS"])
const SLURM_CPUS_PER_TASK = parse(Int, ENV["SLURM_CPUS_PER_TASK"])

exeflags = ["--project=$(Base.active_project())", "-t $SLURM_CPUS_PER_TASK"]
addprocs(SlurmManager(SLURM_NUM_TASKS); exeflags=exeflags, topology=:master_worker)


println("################")
println("Hello! You have $(nworkers()) workers with $(remotecall_fetch(Threads.nthreads, 2)) threads each.")

println("----------------")

println("################")

flush(stdout)


@everywhere begin

    using CUDA
    using QuantumToolbox
    using JLD2
    using Metaheuristics
    using SpecialFunctions
    using Dates
    
    include("../src/utilities.jl")
    include("../src/pulse.jl")
    include("../src/hamiltonian.jl")
    
    BLAS.set_num_threads(1)
    CUDA.allowscalar(false)
    global const op = cu

    # Download parameters

    #superparameters to be chosen
    global const Nt = 5
    global const Nx = 1000
    global const Nr = 160
    global const nsteps = 101
    global const hamitlonian = coupling_norwa_drive_norwa
    global const lambda_1 = 0.1

    println("################")
    println("Parameters")
    println("Nt=$Nt, Nx=$Nx, Nr=$Nr nsteps=$nsteps, hamiltonian=$hamitlonian")
    println("################")
    flush(stdout)



    function sausage_new(params)
        println(params)
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
        params_drive = (tau = params[3],sigma = 3.0,risefall_sigma_ratio = 1.0,amp = params[4],omega_d = params[5])
        
        sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive);
        sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive);

        a = kron(resonator.a,eye(transmon.Nt)) |> op
        SNR = -opposite_SNR(sol0.states,sol1.states,a,tlist,1.0,params[2])
        optimizer = (0.5*erfc(SNR/2) + overlap_err(sol0.states,sol1.states,tlist)) 
        println(optimizer)
        return optimizer
    end

    function simulation_rotated(params)
        ec = 2*pi*0.315
        ej = 51*ec
        g = params[6]
        k_qubit = 2*pi*0.000008
        omega_d = params[5]

        transmon = Transmon(Nt,Nx,ec,ej)
        resonator = Resonator(Nr,params[1],g)
        # g_mat = resonator.g * (transmon.U)' * transmon.n.data * transmon.U
        # omega_q = transmon.E[2]-transmon.E[1]
        # ncrit = abs2((omega_q-params[1])/(2g_mat[1,2]))
        
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

        # state evolution
        tlist = LinRange(0.0,params[3],nsteps)
        params_drive = (tau = params[3],sigma = 1.5,risefall_sigma_ratio = 4.0,amp = params[4],omega_d = params[5])
        sol0 = mesolve(Lt, state0, tlist, saveat=tlist, params=params_drive);
        sol1 = mesolve(Lt, state1, tlist, saveat=tlist, params=params_drive);
        
        # loss function
        a = cu(a)
        SNR = -opposite_SNR(sol0.states,sol1.states,a,tlist,1.0,params[2])
        n0 = real.(expect(a'*a,sol0.states))
        n1 = real.(expect(a'*a,sol1.states))
        relu_photons = 0.0
        if (maximum(real.(n0))>Nr/2) || (maximum(real.(n1))>Nr/2)
        println("Above critical photons")
        # relu_photons = lambda_1*(simpson_integrator(relu.(n0.-ncrit).+relu.(n1.-ncrit),tlist))/tlist[end]
        
        # We used the following to inforce the constraint on the truncation number 
        relu_photons = lambda_1*(simpson_integrator(relu.(n0.-Nr/2).+relu.(n1.-Nr/2),tlist))/tlist[end]

        end
        err_purity = simpson_integrator((2.0).-real.(tr.(sol0.states.*sol0.states)).-real.(tr.(sol1.states.*sol1.states)),tlist)/(2*tlist[end])
        optimizer = 0.5*erfc(SNR/2) + params[3]*k_qubit/2 +  relu_photons + err_purity
        return optimizer
    end

    function sausage_parallel(X)
        # unpack the livraison
        parameters = eachrow(X)
        # ask the Italians to make some sausages
        sausages = pmap(simulation_rotated,parameters)
        return sausages
    end


end

asyncmap((workers())) do p
    remotecall_wait(p) do
        println("my id  $(myid())")
        println("Worker $p uses $(p%length(CUDA.devices())) and id $(myid())")
        CUDA.device!(myid() % length(CUDA.devices()))
    end
end


date = Dates.now()
options = Options(parallel_evaluation=true,verbose=true,iterations= 50)
ts = time()
results = optimize(sausage_parallel, 
                    [43.9823  0.0113818   10.0  0.128826  43.1027 2*pi*0.1; 65.0  0.870957   200.0  1.57   66.0 2*pi*0.5],  
                    #[43.9823  0.0113818   10.0  0.128826  43.1027; 56.5487  0.870957   200.0  1.57   57.6796],
                    PSO(N=300;options))


tf = time()-ts
println(results)
println(tf)
jldopen("data/PSO_purity_optimizer_$date.jld2", "a+") do file
    file["results"] = results
    file["min"] = minimum(results)
    file["minimizer"] = minimizer(results)
    file["time"] = tf
    file["Nt"] = Nt
    file["Nr"] = Nr
    file["lambda"] = lambda_1
    file["ec"] = 2*pi*0.315
    file["ej"] = 51*2*pi*0.315
    file["g"] =  2*pi*0.15
    file["k_qubit"] = 2*pi*0.000008
    file["nsteps"] = nsteps
    end




