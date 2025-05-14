#!/bin/bash

# Set variables
HLS_POW_PROF_LOG_FILE="hls_power_profile.csv"  # Output log file
HLS_POW_PROF_INTERVAL=0.01            # Sampling interval in seconds
# TEMP_JSON_FILE="temp.json"            # Temporary JSON file for xbutil output

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <device_bdf> <host_exec> <xclbin_file> [application_args...]"
    exit 1
fi

DEVICE_BDF=$1
HOST_EXEC=$2
XCLBIN=$3
shift 3
APP_ARGS=("$@")

# Validate BDF format: DDDD:BB:DD.F
if [[ ! "$DEVICE_BDF" =~ ^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]$ ]]; then
    echo "Error: Device BDF '$DEVICE_BDF' is not in the correct format (DDDD:BB:DD.F)"
    exit 1
fi

echo "Logging power data to $HLS_POW_PROF_LOG_FILE"
# echo "Host Executable: $HOST_EXEC" >> "$LOG_FILE"
# echo "XCLBIN File: $XCLBIN" >> "$LOG_FILE"
# echo "Arguments: ${APP_ARGS[*]}" >> "$LOG_FILE"
# echo "Timestamp: $(date)" >> "$LOG_FILE"
# echo "----------------------------------------" >> "$LOG_FILE"
echo "Timestamp,Power (W)" > "$HLS_POW_PROF_LOG_FILE"

capture_power() {
    if ! command -v xbutil &> /dev/null; then
        echo "Error: xbutil command not found. Please ensure Xilinx tools are installed and xbutil is in your PATH." >> "$LOG_FILE"
        exit 1
    fi

    xbutil examine -r electrical -d "$DEVICE_BDF" -f JSON -o "$TEMP_JSON_FILE"

    # if [[ -f "$TEMP_JSON_FILE" ]]; then
    local power_val
    power_val=$(xbutil examine -r electrical -d "$DEVICE_BDF" | awk '/^[[:space:]]*Power[[:space:]]+:[[:space:]]+[0-9.]+ Watts/ { print $3 }')
    echo "$(date +%s%3N),$power_val" >> "$HLS_POW_PROF_LOG_FILE"
    # fi

    # rm -f "$TEMP_JSON_FILE"
}

eval "$HOST_EXEC" "${XCLBIN}" "${APP_ARGS[@]}" &
APP_PID=$!

while kill -0 $APP_PID 2>/dev/null; do
    capture_power
    sleep "$HLS_POW_PROF_INTERVAL"
done

# Add command to the HLS_POW_PROF_LOG_FILE as last line
echo "Command: $HOST_EXEC ${XCLBIN} ${APP_ARGS[*]}" >> "$HLS_POW_PROF_LOG_FILE"