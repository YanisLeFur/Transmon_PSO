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

# Download parameters
include("params.jl")
global const hamitlonian = coupling_rwa_drive_rwa
Nx = 1000
nsteps = 100

start_Nr = 20
end_Nr = 100
steps_Nr = 5
Nr_set = LinRange{Int}(start_Nr,end_Nr,Integer((end_Nr-start_Nr)/steps_Nr)+1)

Nt_set = [5]

println("======START SIMULATION=====")
println("omega_r-omega_d = ",omega_r-omega_d)
println("amp = ", amp)
params = [ec,ej,omega_r,g,k_r,k_q,tau,sigma,risefall_sigma_ratio,amp,omega_d]

end



for i in 1:length(Nr_set)
    for j in 1:length(Nt_set)
        Nr = Nr_set[i]
        Nt = Nt_set[j]
        println(Nr," ",Nt)
        infidelity_normal = sausage_specs_rwa(Nt,Nx,Nr,nsteps,params)
        jldopen("data/coupling_rwa_drive_rwa_specs_amp=$amp omega_d=$omega_d.jld2", "a+") do file
            file["Nt=$Nt Nr=$Nr"] = infidelity_normal
         end
    end
end




