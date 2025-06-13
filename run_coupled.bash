
# Atmosphere parameters with default values
TOA_P=1e-1
BOA_P=1e6
TEMP=200
ALBEDO=0.1
CplusO=1e-3
CtoO=0.59
a_N=0.0

i_min=0

# Parse command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --TOA_P) TOA_P="$2"; shift 2 ;;
        --BOA_P) BOA_P="$2"; shift 2 ;;
        --TEMP) TEMP="$2"; shift 2 ;;
        --ALBEDO) ALBEDO="$2"; shift 2 ;;
        --CplusO) CplusO="$2"; shift 2 ;;
        --CtoO) CtoO="$2"; shift 2 ;;
        --a_N) a_N="$2"; shift 2 ;;
        --i_min) i_min="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# if current directory is not 'chelio', exit 
if [ "$(basename "$PWD")" != "chelio" ]; then
    #echo "Current directory: $(pwd)"
    echo "Please run this script from the 'chelio' directory"
    exit
fi

OUT_DIR="output/test_newCIAs" # relative to chelio directory
NAME="Earth_P0=${BOA_P}_Tint=${TEMP}_CplusO=${CplusO}_CtoO=${CtoO}" #_aN=${a_N}" #_A=${ALBEDO}"
MIXFILE=vertical_mix

# if file MIXFILE_imin+1 exists already exit with warning
# TODO: except if not yet converged!
echo "${OUT_DIR}/${NAME}/${MIXFILE}_$(($i_min+1)).dat"
if [ -f "${OUT_DIR}/${NAME}/${MIXFILE}_$(($i_min+1)).dat" ]; then
    echo "Output ${NAME}/${MIXFILE}_$(($i_min+1)).dat already exists"
    exit
fi

echo "Running: ${NAME}"

# create the first vertical mixing ratios file

# run python script to create abundace file for GGchem
python3 source/calc_abundances.py --CplusO $CplusO --CtoO $CtoO --a_N $a_N
#python3 source/calc_abundances_benchmark.py

# get initial (isothermal) P-T-profile
#python3 source/create_pt.py --Teq $TEMP --Pmin $TOA_P --Pmax $BOA_P
python3 source/create_pt.py --Teq 500 --Pmin $TOA_P --Pmax $BOA_P # for EqCond start with higher T (such that not all species condense)

# create output directory if it does not exist
mkdir -p ${OUT_DIR}/${NAME}

# copy all files to output directory
cp ggchem_inputs/abundances.in ${OUT_DIR}/${NAME}/.
cp ggchem_inputs/pt_helios.in ${OUT_DIR}/${NAME}/${NAME}_tp_coupling_-1.dat
cp ggchem_inputs/param.in ${OUT_DIR}/${NAME}/param_ggchem.in

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
# i_min defined above
i_full=4
i_max=10

for i in $(seq $i_min 1 $i_max)
do

    # convert Static_Conc.dat to HELIOS format and save it in the output directory
    # if i > i_min or i == 0, i.e. if we start at i_min > 0
    if (( i > i_min || i == 0 ))
    then
        cd $CHELIO_PATH
        python3 source/convert_mixfile.py ${OUT_DIR}/${NAME}/${MIXFILE}_$i.dat # only saves "optically relevant" species
        cp $GGCHEM_PATH/Static_Conc.dat ${OUT_DIR}/${NAME}/Static_Conc_$i.dat # save full output for later analysis
        cd $HELIOS_PATH
    fi

    # if vertical mixing ratios file was not created, exit
    if [ ! -f $CHELIO_PATH/${OUT_DIR}/${NAME}/${MIXFILE}_$i.dat ]
    then
        echo "Vertical mixing ratios file not found"
        exit
    fi

    if (( $i == 0 ))
    then
        MAX_ITER=1000 # maybe lower for first iteration?
        # OR set relaxed radiative equilibrium criterion (default 1e-8), then tighten
    elif (( $i >= $i_full ))
    then
        MAX_ITER=30000
        coupling_speed_up="yes"
    else
        MAX_ITER=10000
    fi

    if (( $i > 0 )) && [ -f $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_started_convection.dat ]
	then
        chmod +r $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_started_convection.dat
        started_convection=$(cat $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_started_convection.dat)
	fi

	# run HELIOS
	python3 -u ./helios.py	-name ${NAME} -output_directory $CHELIO_PATH/${OUT_DIR}/ \
                -toa_pressure ${TOA_P} -boa_pressure ${BOA_P} -internal_temperature ${TEMP} -surface_albedo ${ALBEDO} \
				-path_to_temperature_file $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_tp_coupling_$(($i-1)).dat \
				-opacity_mixing on-the-fly -path_to_species_file $CHELIO_PATH/helios_inputs/species.dat \
				-file_with_vertical_mixing_ratios $CHELIO_PATH/${OUT_DIR}/${NAME}/${MIXFILE}_$i.dat \
				-coupling_mode yes -coupling_iteration_step $i -coupling_speed_up $coupling_speed_up -started_convection $started_convection\
                -write_tp_profile_during_run $MAX_ITER -maximum_number_of_iterations $(($MAX_ITER+1))


	# stops iteration after convergence is found
	if (( $i > 0 )) && [ -f $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_coupling_convergence.dat ]
	then
        chmod +r $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_coupling_convergence.dat
        STOP=$(cat $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_coupling_convergence.dat)
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
    cp $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_tp_coupling_$i.dat $GGCHEM_PATH/structures/pt_helios.in
    #python $CHELIO_PATH/source/convert_tp.py $CHELIO_PATH/${OUT_DIR}/${NAME}/${NAME}_tp_coupling_$i.dat 20 # take max(Ti, 20K) for each layer

    # run GGchem
    cd $GGCHEM_PATH
    ./ggchem input/param_helios.in
    cd $HELIOS_PATH

done

# convert Static_Conc.dat to HELIOS format and save it in the output directory one last time
cd $CHELIO_PATH
python3 source/convert_mixfile.py ${OUT_DIR}/${NAME}/${MIXFILE}_$(($i+1)).dat # only saves "optically relevant" species
cp $GGCHEM_PATH/Static_Conc.dat ${OUT_DIR}/${NAME}/Static_Conc_$(($i+1)).dat # save full output for later analysis
