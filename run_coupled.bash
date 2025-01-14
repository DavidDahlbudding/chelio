
NAME=test
MIXFILE=vertical_mix

# Atmosphere parameters
TOA_P=1e-1 # both pressures
BOA_P=1e9 # in dyn/cm^2 or 1e-6 bar
TEMP=1250
ALBEDO=0.0

# Chemistry parameters
OminH=0.0
a_C=0.15
a_N=1e-3

# if current directory is not 'chelio', exit 
if [ "$(basename "$PWD")" != "chelio" ]; then
    #echo "Current directory: $(pwd)"
    echo "Please run this script from the 'chelio' directory"
    exit
fi

# create the first vertical mixing ratios file

# run python script to create abundace file for GGchem
#python3 source/calc_abundances.py --OminH $OminH --a_C $a_C --a_N $a_N
python3 source/calc_abundances_benchmark.py

# get initial (isothermal) P-T-profile
python3 source/create_pt.py --Teq $TEMP --Pmin $TOA_P --Pmax $BOA_P

# create output directory if it does not exist
mkdir -p output/${NAME}

# copy all files to output directory
cp ggchem_inputs/abundances.in output/${NAME}/.
cp ggchem_inputs/pt_helios.in output/${NAME}/${NAME}_tp_coupling_-1.dat
cp ggchem_inputs/param.in output/${NAME}/param_ggchem.in

# prepare GGchem
cp ggchem_inputs/abundances.in $GGCHEM_PATH/abund_helios.in
cp ggchem_inputs/pt_helios.in $GGCHEM_PATH/structures/pt_helios.in
cp ggchem_inputs/param.in $GGCHEM_PATH/input/param_helios.in

# run GGchem
cd $GGCHEM_PATH
./ggchem input/param_helios.in
cd $HELIOS_PATH

# prepare HELIOS
cp $CHELIO_PATH/helios_inputs/param.dat param.dat

# if convection has already started in a previous iteration, set started_convetion=1
started_convection=0
coupling_speed_up="no"

# run the iteration for a sufficient number of iterations (e.g., 10)
i_min=0
i_max=10

for i in $(seq $i_min 1 $i_max)
do

    # convert Static_Conc.dat to HELIOS format and save it in the output directory
    # if i > i_min or i == 0, i.e. if we start at i_min > 0
    if (( i > i_min || i == 0 ))
    then
        cd $CHELIO_PATH
        python3 source/convert_mixfile.py output/${NAME}/${MIXFILE}_$i.dat # only saves "optically relevant" species
        cp $GGCHEM_PATH/Static_Conc.dat output/${NAME}/Static_Conc_$i.dat # save full output for later analysis
        cd $HELIOS_PATH
    fi

    if (( $i == 0 ))
    then
        MAX_ITER=1000 # maybe lower for first iteration?
        # OR set relaxed radiative equilibrium criterion (default 1e-8), then tighten
    else
        MAX_ITER=10000
    fi

    if (( $i > 0 )) && [ -f $CHELIO_PATH/output/${NAME}/${NAME}_started_convection.dat ]
	then
        chmod +r $CHELIO_PATH/output/${NAME}/${NAME}_started_convection.dat
        started_convection=$(cat $CHELIO_PATH/output/${NAME}/${NAME}_started_convection.dat)
	fi

    if (( $i > 2 ))
    then
        coupling_speed_up="yes"
    fi

	# run HELIOS
	python3 -u ./helios.py	-name ${NAME} -output_directory $CHELIO_PATH/output/ \
				-path_to_temperature_file $CHELIO_PATH/output/${NAME}/${NAME}_tp_coupling_$(($i-1)).dat \
				-opacity_mixing on-the-fly -path_to_species_file $CHELIO_PATH/helios_inputs/species.dat \
				-file_with_vertical_mixing_ratios $CHELIO_PATH/output/${NAME}/${MIXFILE}_$i.dat \
				-coupling_mode yes -coupling_iteration_step $i -coupling_speed_up no \
                -write_tp_profile_during_run $MAX_ITER -maximum_number_of_iterations $(($MAX_ITER+1)) \
                -started_convection $started_convection -coupling_speed_up $coupling_speed_up
                 # -toa_pressure ${TOA_P} -boa_pressure ${BOA_P} -internal_temperature ${TEMP} -surface_albedo ${ALBEDO} \

    # alternative way to run HELIOS:
    # set write_tp_profile_during_run to n (e.g. 1000)
    # set maximum_number_of_iterations to n+1
    # --> HELIOS does not have to converge every time

	# stops iteration after convergence is found
	if (( $i > 0 )) && [ -f $CHELIO_PATH/output/${NAME}/${NAME}_coupling_convergence.dat ]
	then
        chmod +r $CHELIO_PATH/output/${NAME}/${NAME}_coupling_convergence.dat
        STOP=$(cat $CHELIO_PATH/output/${NAME}/${NAME}_coupling_convergence.dat)
        echo -e "--> Converged? ${STOP} (1 = yes, 0 = no)"	
        if [[ ${STOP} -eq 1 ]]
   		then
   			break
   		fi
	fi

	# run here your photochemical kinetics code
	# --> read helios_main_dir/output/test/test_tp_coupling_$i.dat
	# --> and produce vertical_mix_(($i+1)).dat so that it can be read next iteration step by HELIOS

    # copy new T(P) profile to GGchem
    cp $CHELIO_PATH/output/${NAME}/${NAME}_tp_coupling_$i.dat $GGCHEM_PATH/structures/pt_helios.in

    # run GGchem
    cd $GGCHEM_PATH
    ./ggchem input/param_helios.in
    cd $HELIOS_PATH

done

# convert Static_Conc.dat to HELIOS format and save it in the output directory one last time
cd $CHELIO_PATH
python3 source/convert_mixfile.py output/${NAME}/${MIXFILE}_$(($i+1)).dat # only saves "optically relevant" species
cp $GGCHEM_PATH/Static_Conc.dat output/${NAME}/Static_Conc_$(($i+1)).dat # save full output for later analysis
