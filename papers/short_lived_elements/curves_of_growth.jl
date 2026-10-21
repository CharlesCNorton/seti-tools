# Curves of growth of the Tc I and Pm II lines in MARCS dwarf atmospheres, computed with Korg with hydrogen lines left
# out of the synthesis. Prints one row per atmosphere and line: Teff, log g, [M/H], element, air wavelength (A), log gf,
# lower level (cm^-1), then the equivalent width (mA) at each abundance A(X) = log10(N_X/N_H) + 12 of the grid.
#
#     julia curves_of_growth.jl > curves_of_growth.txt
using Korg

const CM1_TO_EV = 1.239841984e-4
const TC = [(4238.19, -0.575, 0.0), (4262.27, -0.377, 0.0), (4297.06, -0.237, 0.0)]
const PM = [(4137.95, 0.30, 2950.31), (4157.86, 0.36, 1983.52), (4186.03, 0.32, 1133.45), (4297.78, 0.08, 0.00)]
const SETS = [("Tc", 43, Korg.species"Tc I", TC), ("Pm", 61, Korg.species"Pm II", PM)]
const ABUNDANCES = -1.0:0.5:3.5

function ew(atm, line, Z, A)
    A_X = format_A_X(0.0)
    A_X[Z] = A
    wls = (line.wl * 1e8 - 1.5):0.002:(line.wl * 1e8 + 1.5)
    r = synthesize(atm, [line], A_X, wls; hydrogen_lines=false)
    return sum(1 .- r.flux ./ r.cntm) * 0.002 * 1000.0
end

println("# teff logg m_h element lambda_air loggf e_low_cm ", join(["ew_A$(a)" for a in ABUNDANCES], " "))
models = [(t, 4.55 - 0.2 * (t - 5100) / 1400, 0.0) for t in 5100:100:6500]
append!(models, [(5800, 4.45, -0.3), (5800, 4.45, 0.3)])
for (teff, logg, mh) in models
    atm = interpolate_marcs(teff, logg, mh)
    for (el, Z, spec, lines) in SETS
        for (lam_air, loggf, elo) in lines
            line = Korg.Line(Korg.air_to_vacuum(lam_air), loggf, spec, elo * CM1_TO_EV)
            ews = [ew(atm, line, Z, A) for A in ABUNDANCES]
            println("$teff $(round(logg, digits=3)) $mh $el $lam_air $loggf $elo ",
                    join([string(round(e, sigdigits=5)) for e in ews], " "))
        end
    end
end
