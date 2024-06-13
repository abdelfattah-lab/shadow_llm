import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


for msizes in ["1.3b", "6.7b", "13b"]:
    try:
        # Define the input and output directories
        input_dir = f'opt-{msizes}'
        output_dir = f'corr_graph_opt-{msizes}'

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate over all .pkl files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.pkl') and 'trace' not in filename:
                print("At file: ", filename)
                # Construct full file path
                file_path = os.path.join(input_dir, filename)
                
                # Load the matrix from the pickle file
                with open(file_path, 'rb') as file:
                    matrix = np.asarray(pickle.load(file).detach().numpy())
                
                # Normalize the matrix
                normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
                
                # Plotting the heatmap
                plt.figure(figsize=(8, 8))
                plt.imshow(normalized_matrix, cmap='viridis', aspect='auto')
                plt.colorbar()
                
                # Save the heatmap as a .png file
                output_filename = filename.replace('.pkl', '.png')
                output_path = os.path.join(output_dir, output_filename)
                plt.savefig(output_path)
                plt.close()
    except:
        pass
