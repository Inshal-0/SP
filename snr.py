import numpy as np
import librosa

mse_list=[]
snr_list=[]

first_label=labels[0]
files=word_files[first_label][:15]

print("Using folder:",first_label)
print("Number of files:",len(files))

# Reference file (first file)
ref_signal,ref_sr=librosa.load(files[0],sr=16000)
mfcc_ref=librosa.feature.mfcc(y=ref_signal,sr=ref_sr,n_mfcc=13)

for file in files[1:]:
    signal,sr=librosa.load(file,sr=16000)
    mfcc=librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=13)

    min_frames=min(mfcc_ref.shape[1],mfcc.shape[1])

    mfcc_ref_aligned=mfcc_ref[:,:min_frames].T
    mfcc_aligned=mfcc[:,:min_frames].T

    mse=np.mean((mfcc_ref_aligned-mfcc_aligned)**2)

    signal_power=np.mean(mfcc_ref_aligned**2)
    noise_power=np.mean((mfcc_ref_aligned-mfcc_aligned)**2)
    snr=10*np.log10(signal_power/noise_power)

    mse_list.append(mse)
    snr_list.append(snr)

print(f"Average MSE (first folder, 15 files): {np.mean(mse_list):.6f}")
print(f"Average SNR (first folder, 15 files): {np.mean(snr_list):.2f} dB")
