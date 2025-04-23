import os
import pandas as pd
import argparse

def generate_profile_summary(directory):
    # List to store grid sizes and average main times
    summary_data = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith("perf_profile.csv"):
            file_path = os.path.join(directory, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract unique grid sizes
            grid_x = df['grid_x'].iloc[0]
            grid_y = df['grid_y'].iloc[0]
            grid_z = df['grid_z'].iloc[0]
            
            # Calculate the average main_time
            avg_main_time = df['main_time'].mean()
            
            # Append the data to the summary list
            summary_data.append({
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_z": grid_z,
                "average_main_time": avg_main_time
            })

    # Create a DataFrame for the summary data
    summary_df = pd.DataFrame(summary_data)

    # Sort the DataFrame by grid sizes
    summary_df = summary_df.sort_values(by=["grid_x", "grid_y", "grid_z"])

    # Save the summary DataFrame to a new CSV file
    output_file = os.path.join(directory, "profile_summary.csv")
    summary_df.to_csv(output_file, index=False)

    print(f"Profile summary saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a profile summary CSV from a directory of CSV files.")
    parser.add_argument("-d", "--directory", type=str, help="Path to the directory containing the CSV files.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate the profile summary
    generate_profile_summary(args.directory)