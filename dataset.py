"""
mfcc_lab.py

Reusable code for:
- Loading .flac files
- Computing MFCCs from scratch
- Computing MFCCs with librosa
- Comparing both (MSE, SNR)
"""

import os
from typing import List,Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# ============================================================
# 1. File utilities
# ============================================================

def find_flac_files(data_dir:str)->List[str]:
    """Return a list of absolute paths to all .flac files under data_dir."""
    flac_files=[]
    for root,_,files in os.walk(data_dir):
        for file in files:
            if file.endswith(".flac"):
                flac_files.append(os.path.join(root,file))
    return flac_files

def load_audio(file_path:str,sr=None)->Tuple[np.ndarray,int]:
    """Load audio file with librosa (no resampling if sr=None)."""
    signal,sr=librosa.load(file_path,sr=sr)
    return signal,sr

# ============================================================
# 2. MFCC from scratch
# ============================================================

def pre_emphasis(signal:np.ndarray,coeff:float=0.97)->np.ndarray:
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def framing(signal:np.ndarray,sr:int,
            frame_size:float=0.025,frame_stride:float=0.01)->np.ndarray:
    frame_length=int(frame_size*sr)
    frame_step=int(frame_stride*sr)
    num_frames=int(np.ceil(float(np.abs(len(signal)-frame_length))/frame_step))

    pad_signal_length=num_frames*frame_step+frame_length
    z=np.zeros((pad_signal_length-len(signal)))
    pad_signal=np.append(signal,z)

    indices=np.tile(np.arange(0,frame_length),(num_frames,1))+ \
            np.tile(np.arange(0,num_frames*frame_step,frame_step),(frame_length,1)).T
    frames=pad_signal[indices.astype(np.int32,copy=False)]
    return frames

def hamming_window(N:int)->np.ndarray:
    return 0.54-0.46*np.cos(2*np.pi*np.arange(N)/(N-1))

def apply_window(frames:np.ndarray)->np.ndarray:
    window=hamming_window(frames.shape[1])
    return frames*window

def power_spectrum(frames:np.ndarray,NFFT:int=512)->np.ndarray:
    mag_frames=np.absolute(np.fft.rfft(frames,NFFT))
    pow_frames=(1.0/NFFT)*(mag_frames**2)
    return pow_frames

def hz_to_mel(hz:np.ndarray)->np.ndarray:
    return 2595*np.log10(1+hz/700)

def mel_to_hz(mel:np.ndarray)->np.ndarray:
    return 700*(10**(mel/2595)-1)

def mel_filterbank(pow_frames:np.ndarray,sr:int,
                   NFFT:int=512,nfilt:int=26)->np.ndarray:
    low_mel=hz_to_mel(0)
    high_mel=hz_to_mel(sr/2)
    mel_points=np.linspace(low_mel,high_mel,nfilt+2)
    hz_points=mel_to_hz(mel_points)
    bin_points=np.floor((NFFT+1)*hz_points/sr).astype(int)

    fbank=np.zeros((nfilt,int(NFFT/2+1)))
    for m in range(1,nfilt+1):
        f_m_minus=bin_points[m-1]
        f_m=bin_points[m]
        f_m_plus=bin_points[m+1]

        for k in range(f_m_minus,f_m):
            fbank[m-1,k]=(k-f_m_minus)/(f_m-f_m_minus)
        for k in range(f_m,f_m_plus):
            fbank[m-1,k]=(f_m_plus-k)/(f_m_plus-f_m)

    filter_banks=np.dot(pow_frames,fbank.T)
    filter_banks=np.where(filter_banks==0,np.finfo(float).eps,filter_banks)
    return np.log(filter_banks)

def compute_mfcc(log_energies:np.ndarray,num_ceps:int=13)->np.ndarray:
    mfcc=np.array([dct(f,type=2,norm="ortho")[:num_ceps] for f in log_energies])
    mfcc=mfcc-(np.mean(mfcc,axis=0)+1e-8)
    return mfcc

def mfcc_from_scratch(signal:np.ndarray,sr:int,
                      frame_size:float=0.025,frame_stride:float=0.01,
                      NFFT:int=512,nfilt:int=26,num_ceps:int=13,
                      pre_emph:float=0.97)->np.ndarray:
    emphasized=pre_emphasis(signal,coeff=pre_emph)
    frames=framing(emphasized,sr,frame_size=frame_size,frame_stride=frame_stride)
    windowed_frames=apply_window(frames)
    pow_frames=power_spectrum(windowed_frames,NFFT=NFFT)
    log_energies=mel_filterbank(pow_frames,sr,NFFT=NFFT,nfilt=nfilt)
    mfcc=compute_mfcc(log_energies,num_ceps=num_ceps)
    return mfcc

# ============================================================
# 3. Librosa MFCC & comparison metrics
# ============================================================

def mfcc_librosa(signal:np.ndarray,sr:int,n_mfcc:int=13)->np.ndarray:
    """Return MFCCs in shape (frames,coeffs) to match mfcc_from_scratch."""
    m=librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=n_mfcc)
    return m.T  # (frames,coeffs)

def compare_mfcc(mfcc_manual:np.ndarray,mfcc_lib:np.ndarray)->Tuple[float,float]:
    """
    Return (MSE,SNR_dB) after aligning on min number of frames.
    mfcc_manual and mfcc_lib must both be (frames,coeffs).
    """
    min_frames=min(mfcc_manual.shape[0],mfcc_lib.shape[0])
    mfcc_manual=mfcc_manual[:min_frames]
    mfcc_lib=mfcc_lib[:min_frames]

    diff=mfcc_manual-mfcc_lib
    mse=float(np.mean(diff**2))

    signal_power=float(np.mean(mfcc_lib**2))
    noise_power=float(np.mean(diff**2))
    snr_db=10*np.log10(signal_power/noise_power)
    return mse,snr_db

# ============================================================
# 4. Visualisation helpers
# ============================================================

def plot_mfccs(mfcc_manual:np.ndarray,mfcc_lib:np.ndarray,sr:int)->None:
    """Show side-by-side spectrograms of manual vs librosa MFCCs."""
    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    librosa.display.specshow(mfcc_manual.T,sr=sr,x_axis="time")
    plt.title("MFCC (From Scratch)")
    plt.colorbar()

    plt.subplot(1,2,2)
    librosa.display.specshow(mfcc_lib.T,sr=sr,x_axis="time")
    plt.title("MFCC (Librosa)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# ============================================================
# 5. Batch evaluation on multiple files
# ============================================================

def evaluate_on_files(file_paths:List[str],max_files:int=10)->Tuple[float,float]:
    """Return (avg_MSE, avg_SNR_dB) over up to max_files files."""
    mse_list=[]
    snr_list=[]
    for file in file_paths[:max_files]:
        signal,sr=load_audio(file,sr=None)
        mfcc_man=mfcc_from_scratch(signal,sr)
        mfcc_lib=mfcc_librosa(signal,sr)
        mse,snr=compare_mfcc(mfcc_man,mfcc_lib)
        mse_list.append(mse)
        snr_list.append(snr)
    return float(np.mean(mse_list)),float(np.mean(snr_list))

# ============================================================
# 6. Example usage (you can delete or adapt this part)
# ============================================================

if __name__=="__main__":
    # Change this to your dataset path
    DATA_DIR="/kaggle/input/librispeech/dev-clean/"

    flac_files=find_flac_files(DATA_DIR)
    print("Total .flac files found:",len(flac_files))
    print("Example file:",flac_files[0])

    # Single-file example
    signal,sr=load_audio(flac_files[0],sr=None)
    print("Sample rate:",sr,"| Duration (s):",len(signal)/sr)

    mfcc_man=mfcc_from_scratch(signal,sr)
    mfcc_lib=mfcc_librosa(signal,sr)
    mse,snr=compare_mfcc(mfcc_man,mfcc_lib)
    print(f"Single file -> MSE:{mse:.6f}, SNR:{snr:.2f} dB")

    # Visualise
    plot_mfccs(mfcc_man,mfcc_lib,sr)

    # Multi-file evaluation
    avg_mse,avg_snr=evaluate_on_files(flac_files,max_files=10)
    print(f"Average MSE across 10 files:{avg_mse:.6f}")
    print(f"Average SNR across 10 files:{avg_snr:.2f} dB")
