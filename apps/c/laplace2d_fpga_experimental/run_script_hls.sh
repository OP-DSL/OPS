#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: ${0} <target_mode> <app_name> [<additional_args>]"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

TARGET_MODE=$1
APP_NAME=$2
PLATFORM=$3
shift 3
CXXFLAGS="$@"

PROFILE_DIR=./hls/profile_data/${TARGET_MODE}/
PROFILE_FILE=perf_profile.csv
POWER_PROFILE_FILE=hls_power_profile.csv
DEVICE_BDF=0000:c1:00.1

# Hardcoded parameter sets (sizex, sizey, iters, batch)
if [[ "${CXXFLAGS}" == *"-DPOWER_PROFILE"* ]]; then
    if [[ "${TARGET_MODE}" == "hw" ]]; then
        if [[ "${PLATFORM}" == *"u280"* ]]; then
            parameter_sets=(
                "100,100,60120,5000"
                "200,100,60120,5000"
                "200,200,60120,5000"
                "300,150,60120,2500"
                "300,300,60120,1000"
                "400,200,60120,1000"
                "400,300,60120,500"
                "400,400,60120,500"
                # Add more parameter sets here as needed
            )
        else
            parameter_sets=(
            "100,100,60032,5000"
            "200,100,60032,5000"
            "200,200,60032,5000"
            "300,150,60032,2500"
            "300,300,60032,1000"
            "400,200,60032,1000"
            "400,300,60032,500"
            "400,400,60032,500"
            # Add more parameter sets here as needed
            )
        fi
    else
        echo "Error: Cannot power profile sw_emu or hw_emu"
        exit 1
    fi
else
    if [[ "${TARGET_MODE}" == "hw" ]]; then
        if [[ "${PLATFORM}" == *"u280"* ]]; then
            parameter_sets=(
                "30,30,60120,100"
                "60,60,60120,100"
                "100,100,60120,100"
                "200,100,60120,100"
                "200,200,60120,100"
                "300,150,60120,100"
                "300,300,60120,20"
                "400,200,60120,20"
                "400,300,60120,20"
                "400,400,60120,20"
                "400,425,60120,20"
                "400,350,60120,20"
                "400,375,60120,20"
                "300,350,60120,20"
                "300,375,60120,20"
                "300,400,60120,20"
                "300,425,60120,20"
                "300,450,60120,20"
                "300,475,60120,20"
                "300,500,60120,20"
                "300,525,60120,20"
                "300,550,60120,20"
                "300,575,60032,20"
                # Add more parameter sets here as needed
                )
        else
            parameter_sets=(
                "30,30,60032,100"
                "60,60,60032,100"
                "100,100,60032,100"
                "200,100,60032,100"
                "200,200,60032,100"
                "300,150,60032,100"
                "300,300,60032,20"
                "400,200,60032,20"
                "400,300,60032,20"
                "400,400,60032,20"
                "400,425,60032,20"
                "400,350,60032,20"
                "400,375,60032,20"
                "300,350,60032,20"
                "300,375,60032,20"
                "300,400,60032,20"
                "300,425,60032,20"
                "300,450,60032,20"
                "300,475,60032,20"
                "300,500,60032,20"
                "300,525,60032,20"
                "300,550,60032,20"
                "300,575,60032,20"
                # Add more parameter sets here as needed
                )
        fi
    
    else
    parameter_sets=(
        "30,30,112,1"
        # Add more parameter sets here as needed
    )
    fi
fi

echo "Running application '${APP_NAME}' in '${TARGET_MODE}' mode with hardcoded parameters:"

for params in "${parameter_sets[@]}"; do
    IFS=',' read -r sizex sizey iters batch <<< "$params"

    if [[ -z "$sizex" || -z "$sizey" || -z "$iters" || -z "$batch" ]]; then
        echo "Warning: Skipping invalid parameter set: $params"
        continue
    fi

    # Removing previous residues
    if [ -f "${PROFILE_FILE}" ]; then
        rm ${PROFILE_FILE}
    fi
    if [ -f "${POWER_PROFILE_FILE}" ]; then
        rm ${POWER_PROFILE_FILE}
    fi


    echo "-----------------------------------------------------------------"
    echo "Running with sizex=${sizex}, sizey=${sizey}, iters=${iters}, batch=${batch}"
    echo "-----------------------------------------------------------------"


    if [[ "${CXXFLAGS}" == *"-DPOWER_PROFILE"* ]]; then
        echo "Running HW mode with power profiling"
            ${OPS_INSTALL_PATH}/../scripts/power_profile_hls.sh ${DEVICE_BDF} ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}_host ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}.xclbin -sizex="${sizex}" -sizey="${sizey}" -iters="${iters}" -piter="${batch}"

    else
        if [[ $TARGET_MODE == sw_emu || $TARGET_MODE == hw_emu ]]; then
            echo "Running in emulation mode with ${TARGET_MODE}"
            XCL_EMULATION_MODE=${TARGET_MODE} ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}_host ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}.xclbin -sizex="${sizex}" -sizey="${sizey}" -iters="${iters}" -batch="${batch}"
        else
            echo "Running HW mode"
            ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}_host ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}.xclbin -sizex="${sizex}" -sizey="${sizey}" -iters="${iters}" -batch="${batch}"
        fi
    fi

    if [ ! -d "${PROFILE_DIR}" ]; then
        echo "Directory '${PROFILE_DIR}' does not exist. Creating it..."
        mkdir -p "${PROFILE_DIR}"
    fi
    if [ -f "${PROFILE_FILE}" ]; then
        # Construct the new filename for the profile directory
        new_filename="${PROFILE_DIR}/${sizex}_${sizey}_${PROFILE_FILE}"
        echo "Moving '${PROFILE_FILE}' to '${new_filename}'"
        mv "${PROFILE_FILE}" "${new_filename}"
    else
        echo "Warning: Output file '${PROFILE_FILE}' not found after the run."
    fi
    if [ -f "${POWER_PROFILE_FILE}" ]; then
        # Construct the new filename for the profile directory
        new_filename="${PROFILE_DIR}/${sizex}_${sizey}_${POWER_PROFILE_FILE}"
        echo "Moving '${POWER_PROFILE_FILE}' to '${new_filename}'"
        mv "${POWER_PROFILE_FILE}" "${new_filename}"
    else
        echo "Warning: Output file '${POWER_PROFILE_FILE}' not found after the run."
    fi
done

echo "-----------------------------------------------------------------"
echo "Finished running all hardcoded parameter sets."

exit 0