# Pauli Hamiltonians for the frozen-core trees, with the converter's exact
# diagonalization check on (the (5,5) sector has 3136 states).
#
#     julia --project=../../integrals build_fc_hamiltonians.jl
include(joinpath(@__DIR__, "..", "..", "integrals", "uhf_integrals_to_pauli.jl"))

for (tree, bases) in (("data_uhfcore", ["mo", "lmo"]), ("data_nocore", ["no", "lno"]), ("data_unocore", ["uno", "luno"]))
    datadir = joinpath(@__DIR__, tree)
    for d in sort(filter(startswith("R_"), readdir(datadir))), b in bases
        run([joinpath(datadir, d), "--basis", b, "--ed-max-dim", "4000"])
    end
end
