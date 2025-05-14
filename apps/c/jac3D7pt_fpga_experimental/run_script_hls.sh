#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: ${0} <target_mode> <app_name> [<additional_args>]"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

TARGET_MODE=$1
APP_NAME=$2
shift 2
CXXFLAGS="$@"

PROFILE_DIR=./hls/profile_data/${TARGET_MODE}/
PROFILE_FILE=perf_profile.csv
POWER_PROFILE_FILE=hls_power_profile.csv
DEVICE_BDF=0000:c1:00.1

# Hardcoded parameter sets (sizex, sizey, iters, batch)
if [[ "${CXXFLAGS}" == *"-DPOWER_PROFILE"* ]]; then
    @echo "Power profiling enabled"
    if [[ "${TARGET_MODE}" == "hw" ]]; then
        if [[ "${PLATFORM}" == *"u280"* ]]; then
            parameter_sets=(
                "30,30,30,60048,2000"
                "50,50,50,60048,2000"
                "100,100,100,60048,2000"
                "150,150,150,60048,500"
                "200,200,200,60048,500"
                "250,250,250,60048,100"
                "300,300,300,60048,100"
                # Add more parameter sets here as needed
            )
        else
            parameter_sets=(
                # "30,30,30,60016,2000"
                # "50,50,50,60016,2000"
                # "100,100,100,60016,200"
                # "150,150,150,60016,100"
                # "200,200,200,60016,100"
                # "250,250,250,60016,50"
                "300,300,300,60016,10"
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
                "30,30,30,60048,50"
                "50,50,50,60048,50"
                "100,100,100,60048,20" 
                "150,150,150,60048,20"
                "200,200,200,60048,10"
                "250,250,250,60048,5"
                "300,300,300,60048,2"
                # Add more parameter sets here as needed
                )
        else
            parameter_sets=(
                "30,30,30,60016,50"
                "50,50,50,60016,50"
                "100,100,100,60016,20" 
                "150,150,150,60016,20"
                "200,200,200,60016,10"
                "250,250,250,60016,5"
                "300,300,300,60016,2"
                # Add more parameter sets here as needed
                )
        fi
    else
        if [[ "${PLATFORM}" == *"u280"* ]]; then
            parameter_sets=(
                "30,30,30,108,1"
                # Add more parameter sets here as needed
            )
        else
            parameter_sets=(
                "30,30,30,132,1"
                # Add more parameter sets here as needed
            )
        fi
    fi
fi

echo "Running application '${APP_NAME}' in '${TARGET_MODE}' mode with hardcoded parameters:"

for params in "${parameter_sets[@]}"; do
    IFS=',' read -r sizex sizey sizez iters batch <<< "$params"

    if [[ -z "$sizex" || -z "$sizey" || -z "$sizez" || -z "$iters" || -z "$batch" ]]; then
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
    echo "Running with sizex=${sizex}, sizey=${sizey}, sizez=${sizez}, iters=${iters}, batch=${batch}"
    echo "-----------------------------------------------------------------"


    if [[ "${CXXFLAGS}" == *"-DPOWER_PROFILE"* ]]; then
        echo "Running HW mode with power profiling"
            ${OPS_INSTALL_PATH}/../scripts/power_profile_hls.sh ${DEVICE_BDF} ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}_host ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}.xclbin -sizex="${sizex}" -sizey="${sizey}" -sizez="${sizez}" -iters="${iters}" -piter="${batch}"

    else
        if [[ $TARGET_MODE == sw_emu || $TARGET_MODE == hw_emu ]]; then
            echo "Running in emulation mode with ${TARGET_MODE}"
            XCL_EMULATION_MODE=${TARGET_MODE} ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}_host ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}.xclbin -sizex="${sizex}" -sizey="${sizey}" -sizez="${sizez}" -iters="${iters}" -batch="${batch}"
        else
            echo "Running HW mode"
            ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}_host ${SCRIPT_DIR}/hls/build/${TARGET_MODE}/${APP_NAME}.xclbin -sizex="${sizex}" -sizey="${sizey}" -sizez="${sizez}" -iters="${iters}" -batch="${batch}"
        fi
    fi

    if [ ! -d "${PROFILE_DIR}" ]; then
        echo "Directory '${PROFILE_DIR}' does not exist. Creating it..."
        mkdir -p "${PROFILE_DIR}"
    fi
    if [ -f "${PROFILE_FILE}" ]; then
        # Construct the new filename for the profile directory
        new_filename="${PROFILE_DIR}/${sizex}_${sizey}_${sizez}_${PROFILE_FILE}"
        echo "Moving '${PROFILE_FILE}' to '${new_filename}'"
        mv "${PROFILE_FILE}" "${new_filename}"
    else
        echo "Warning: Output file '${PROFILE_FILE}' not found after the run."
    fi
    if [ -f "${POWER_PROFILE_FILE}" ]; then
        # Construct the new filename for the profile directory
        new_filename="${PROFILE_DIR}/${sizex}_${sizey}_${sizez}_${POWER_PROFILE_FILE}"
        echo "Moving '${POWER_PROFILE_FILE}' to '${new_filename}'"
        mv "${POWER_PROFILE_FILE}" "${new_filename}"
    else
        echo "Warning: Output file '${POWER_PROFILE_FILE}' not found after the run."
    fi
done

echo "-----------------------------------------------------------------"
echo "Finished running all hardcoded parameter sets."

exit 0