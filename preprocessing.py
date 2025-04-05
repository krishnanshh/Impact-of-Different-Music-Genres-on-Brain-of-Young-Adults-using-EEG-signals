import h5py
import numpy as np
import matplotlib.pyplot as plt
file_path = r"C:\Users\Krishnansh Verma\OneDrive\Desktop\Python and ML\Projects\Capstone Dataset\11_1_raw.mat"
with h5py.File(file_path, 'r') as f:
    eeg_data = np.array(f['X']).T  
    sampling_frequency = int(f['fs'][0][0])  

    din_metadata = f['DIN_1'][:]  
    if din_metadata.dtype == np.dtype('O'): 
        start_sample = 0  
    else:
        event_sample_points = din_metadata[:, 0].astype(int)
        start_sample = event_sample_points[0]

    # Extract 1000-sample segment
    end_sample = start_sample + 100000
    selected_eeg_data = eeg_data[:, start_sample:end_sample]

    # Plotting
    plt.figure(figsize=(15, 20))
    offset = 0
    for i in range(128):
        plt.plot(selected_eeg_data[i] + i * offset, label=f'Ch {i+1}')
    
    plt.title("EEG Channels (Unprocessed)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (offset stacked)")
    plt.yticks([])
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    