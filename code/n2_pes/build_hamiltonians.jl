# Convert every data/R_*/ point to PauliSum .jld2 files in all four bases with
# ../integrals/uhf_integrals_to_pauli.jl, in one Julia session.
#
#     julia --project=../integrals build_hamiltonians.jl [bases] [extra converter options...]
#     julia --project=../integrals build_hamiltonians.jl mo,lmo,no,lno
include(joinpath(@__DIR__, "..", "integrals", "uhf_integrals_to_pauli.jl"))

bases = length(ARGS) >= 1 ? split(ARGS[1], ",") : ["mo", "lmo", "no", "lno"]
extra = ARGS[2:end]
datadir = joinpath(@__DIR__, "data")
for d in sort(filter(startswith("R_"), readdir(datadir))), b in bases
    run([joinpath(datadir, d), "--basis", String(b), extra...])
end
