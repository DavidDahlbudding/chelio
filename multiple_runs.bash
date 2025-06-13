# create arrays for different parameters

# Atmosphere parameters
BOA_Ps=(1e6 1e7 1e8)
TEMPs=($(seq 50 50 250))

# Chemistry parameters
CplusOs=(1e-3 3.16e-3 1e-2 3.16e-2 1e-1) # 3.16e-1 1e0)
CtoOs=(0.1 0.59 1.0)
#a_Ns=(1e-4 1e-2)
#ALBEDOs=(0.0 0.2 0.4 0.8)

# loop over all parameters

for BOA_P in "${BOA_Ps[@]}"; do
    for TEMP in "${TEMPs[@]}"; do
        for CplusO in "${CplusOs[@]}"; do
            for CtoO in "${CtoOs[@]}"; do
                #for ALBEDO in "${ALBEDOs[@]}"; do
                #for a_N in "${a_Ns[@]}"; do
                bash run_coupled.bash --BOA_P $BOA_P --TEMP $TEMP --CplusO $CplusO --CtoO $CtoO #--a_N $a_N #--ALBEDO $ALBEDO
                #done
            done
        done
    done
done