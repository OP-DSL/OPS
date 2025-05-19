import os
import pandas as pd
import argparse

def process_power_profiles(directory, p_batches):
    summary_data = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith("power_profile.csv"):
            file_path = os.path.join(directory, filename)
            
            # Read the CSV file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Extract the last line for the command
            command_line = lines[-1].strip()
            
            # Extract grid sizes from the command
            sizex = 1
            sizey = 1
            sizez = 1
            if "-sizex=" in command_line:
                sizex = int(command_line.split("-sizex=")[1].split()[0])
            if "-sizey=" in command_line:
                sizey = int(command_line.split("-sizey=")[1].split()[0])
            if "-sizez=" in command_line:
                sizez = int(command_line.split("-sizez=")[1].split()[0])
            
            # Extract the batch size from the command
            if "-piter=" in command_line:
                batch_size = int(command_line.split("-piter=")[1].split()[0])
            else:
                print(f"Warning: No batch size found in {filename}. Skipping.")
                continue
            
            # Read the data excluding the last line
            data = pd.read_csv(file_path, skipfooter=1, engine='python')
            
            # Calculate average power
            avg_power = data['Power (W)'].mean()
            
            # Estimate elapsed time using timestamps
            start_time = data['Timestamp'].iloc[0]
            end_time = data['Timestamp'].iloc[-1]
            elapsed_time = (end_time - start_time)  # in milliseconds
            
            # Calculate total energy for the original batch size
            total_energy = avg_power * (elapsed_time / 1000)  # Convert ms to seconds for energy calculation
            
            # Convert total energy to kilojoules
            total_energy_kj = total_energy / 1000  # in kJ
            
            # Calculate energy per batch
            energy_per_batch_kj = total_energy_kj / batch_size
            
            # Calculate estimated energy for p batches
            estimated_energy_kj = energy_per_batch_kj * p_batches
            
            # Append the results to the summary
            summary_data.append({
                "File": filename,
                "Grid Size X": sizex,
                "Grid Size Y": sizey,
                "Grid Size Z": sizez,
                "Batch Size": batch_size,
                "Average Power (W)": avg_power,
                "Elapsed Time (ms)": elapsed_time,
                "Total Energy (kJ)": total_energy_kj,
                "Energy per Batch (kJ)": energy_per_batch_kj,
                f"Estimated Energy for {p_batches} Batches (kJ)": estimated_energy_kj
            })

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary_data)

    # Sort the DataFrame by grid sizes
    summary_df = summary_df.sort_values(by=["Grid Size X", "Grid Size Y", "Grid Size Z"])

    # Save the summary to a CSV file
    output_file = os.path.join(directory, "power_profile_summary.csv")
    summary_df.to_csv(output_file, index=False)

    print(f"Power profile summary saved to {output_file}")
    
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process power profile data and calculate energy metrics.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Path to the directory containing the power profile CSV files.")
    parser.add_argument("-p", "--p_batches", type=int, required=True, help="Number of batches for energy estimation.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the power profiles
    process_power_profiles(args.directory, args.p_batches)