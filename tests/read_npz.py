import numpy as np

# Load the file
data = np.load('models\stop_template_spotter.npz')

# List all arrays stored in the file
print("Keys in the file:", data.files)

# Access a specific array using its key
for key in data.files:
    print(f"\nArray: {key}")
    print(data[key])