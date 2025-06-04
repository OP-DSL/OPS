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
    @echo "Power profiling enabled"
    if [[ "${TARGET_MODE}" == "hw" ]]; then
        if [[ "${PLATFORM}" == *"u280"* ]]; then
            parameter_sets=(
                "30,30,30,1800,100"
                "30,30,50,1800,100"
                "50,50,16,1800,100"
                "50,50,30,1800,50"
                "50,50,50,1800,20"
                "75,75,75,1800,10"
                "100,100,100,1800,10"
                "125,125,125,1800,5"
                "150,150,150,1800,5"
                "175,175,175,1800,5"
                "200,200,200,1800,3"
                "225,225,225,1800,2"
                "250,250,250,1800,2"
                # Add more parameter sets here as needed
            )
        else
            parameter_sets=(
                "30,30,30,1800,100"
                "30,30,50,1800,100"
                "50,50,16,1800,100"
                "50,50,30,1800,50"
                "50,50,50,1800,20"
                "75,75,75,1800,10"
                "100,100,100,1800,10"
                "125,125,125,1800,5"
                "150,150,150,1800,5"
                "175,175,175,1800,5"
                "200,200,200,1800,3"
                "225,225,225,1800,2"
                "250,250,250,1800,2"
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
                "30,30,30,1800,10"
                "30,30,50,1800,10"
                "50,50,16,1800,10"
                "50,50,30,1800,5"
                "50,50,50,1800,5"
                "75,75,75,1800,5"
                "100,100,100,1800,2"
                "125,125,125,1800,2"
                "150,150,150,1800,2"
                "175,175,175,1800,2"
                "200,200,200,1800,2"
                "225,225,225,1800,2"
                "250,250,250,1800,2"
                # Add more parameter sets here as needed
                )
        else
            parameter_sets=(
                "30,30,30,1800,10"
                "30,30,50,1800,10"
                "50,50,16,1800,10"
                "50,50,30,1800,5"
                "50,50,50,1800,5"
                "75,75,75,1800,5"
                "100,100,100,1800,2"
                "125,125,125,1800,2"
                "150,150,150,1800,2"
                "175,175,175,1800,2"
                "200,200,200,1800,2"
                "225,225,225,1800,2"
                "250,250,250,1800,2"
                # Add more parameter sets here as needed
                )
        fi
    else
        if [[ "${PLATFORM}" == *"u280"* ]]; then
            parameter_sets=(
                "3,3,3,4,1"
                # Add more parameter sets here as needed
            )
        else
            parameter_sets=(
                "3,3,3,4,1"
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