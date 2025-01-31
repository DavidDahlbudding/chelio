# create arrays for different parameters

# Atmosphere parameters
BOA_Ps=(1e6 1e7)
TEMPs=($(seq 100 50 350))

# Chemistry parameters
CplusOs=(1e-3 3.16e-3 1e-2 3.16e-2 1e-1 3.16e-1 1e0)
CtoOs=(0.1 0.3 0.59 0.8 1.0 1.2)
#a_Ns=0.0

# loop over all parameters

for BOA_P in "${BOA_Ps[@]}"; do
    for TEMP in "${TEMPs[@]}"; do
        for CplusO in "${CplusOs[@]}"; do
            for CtoO in "${CtoOs[@]}"; do
                bash run_coupled.bash --BOA_P $BOA_P --TEMP $TEMP --CplusO $CplusO --CtoO $CtoO
            done
        done
    done
done