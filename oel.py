"""
speech_lab1.py  (fixed)

Reusable code for Speech Processing Lab (OEL-1)

Q1:  create tiny synthetic dataset with 2 speakers saying "aa","ee","oo"
Q2:  visualize each sample (waveform, spectrogram, MFCCs)
Q3:  spectrum of a vowel + estimated formants & harmonic frequencies
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from typing import Dict,Tuple,List

# =========================
# General helpers
# =========================

def normalize(sig:np.ndarray)->np.ndarray:
    sig=sig-np.mean(sig)
    max_abs=np.max(np.abs(sig))+1e-9
    return sig/max_abs

# =========================
# Q1 – synthetic vowel dataset
# =========================

def generate_vowel(f0:float,
                   formants:Tuple[float,float,float],
                   bandwidths:Tuple[float,float,float]=(80,90,120),
                   duration:float=0.6,
                   sr:int=16000)->np.ndarray:
    """
    Simple formant-synthesis vowel:
    sum of 3 damped sinusoids around given formant freqs.
    """
    t=np.linspace(0,duration,int(sr*duration),endpoint=False)
    sig=np.zeros_like(t)

    for f,b in zip(formants,bandwidths):
        # exponential decay envelope
        env=np.exp(-b*t)
        sig=sig+env*np.sin(2*np.pi*f*t)

    # add low-level fundamental to feel like voiced speech
    sig=sig+0.2*np.sin(2*np.pi*f0*t)
    return normalize(sig)

def create_synthetic_dataset(sr:int=16000,duration:float=0.6
                             )->Dict[str,Dict[str,np.ndarray]]:
    """
    Returns:
        {
          "speaker_1":{"aa":sig,...},
          "speaker_2":{"aa":sig,...}
        }
    Speakers differ mainly in pitch (F0) and slightly in formants.
    """
    # approximate formant sets (Hz) for male/female-ish voices
    vowels={
        "aa":(700,1220,2600),
        "ee":(300,2300,3000),
        "oo":(400,800,2600)
    }

    dataset={}

    # Speaker 1 (lower pitch)
    spk1={}
    for v,fs in vowels.items():
        spk1[v]=generate_vowel(f0=110,formants=fs,duration=duration,sr=sr)
    dataset["speaker_1"]=spk1

    # Speaker 2 (higher pitch, slightly shifted formants)
    spk2={}
    for v,fs in vowels.items():
        f_shift=1.10  # 10% higher formants
        fs2=tuple(f*f_shift for f in fs)
        spk2[v]=generate_vowel(f0=180,formants=fs2,duration=duration,sr=sr)
    dataset["speaker_2"]=spk2

    return dataset

# =========================
# Q2 – visualisation
# =========================

def visualize_sample(sig:np.ndarray,sr:int,title:str="")->None:
    """
    Plots:
      1) waveform
      2) log-power spectrogram
      3) MFCCs
    """
    fig,axes=plt.subplots(3,1,figsize=(10,9))

    # 1. waveform
    t=np.arange(len(sig))/sr
    axes[0].plot(t,sig)
    axes[0].set_title(f"Waveform – {title}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # 2. spectrogram (STFT magnitude in dB)
    D=np.abs(librosa.stft(sig,n_fft=1024,hop_length=256))**2
    S=librosa.power_to_db(D,ref=np.max)
    img1=librosa.display.specshow(S,sr=sr,hop_length=256,
                                  x_axis="time",y_axis="hz",
                                  ax=axes[1])
    axes[1].set_title("Spectrogram (dB)")
    fig.colorbar(img1,ax=axes[1],format="%+2.0f dB")

    # 3. MFCCs
    mfcc=librosa.feature.mfcc(y=sig,sr=sr,n_mfcc=13)
    img2=librosa.display.specshow(mfcc,x_axis="time",sr=sr,
                                  ax=axes[2])
    axes[2].set_title("MFCCs")
    axes[2].set_ylabel("Coeff index")
    fig.colorbar(img2,ax=axes[2])

    plt.tight_layout()
    plt.show()

# =========================
# Q3 – spectrum, formants, harmonics
# =========================

def lpc_coeffs(frame:np.ndarray,order:int)->np.ndarray:
    """
    Compute LPC coefficients using autocorrelation + Yule-Walker equations.
    Returns array a of length order+1 (a[0] = 1).
    """
    # autocorrelation of frame
    r=np.correlate(frame,frame,mode="full")
    r=r[len(frame)-1:len(frame)+order]   # r[0..order]

    # Toeplitz covariance matrix and RHS
    R=toeplitz(r[:-1])   # (order x order)
    rhs=-r[1:]

    # solve for LPC coefficients a[1:]
    a_rest=np.linalg.solve(R,rhs)
    a=np.concatenate(([1.0],a_rest))
    return a

def estimate_formants_lpc(sig:np.ndarray,sr:int,
                          lpc_order:int=12,
                          n_formants:int=3)->List[float]:
    """
    Rough LPC-based formant estimation.
    Returns first n_formants frequencies (Hz).
    """
    # pre-emphasis & short central frame
    sig=librosa.util.normalize(sig)
    sig=librosa.effects.preemphasis(sig)

    # take a stable middle section
    mid=len(sig)//2
    win_len=int(0.03*sr)  # 30 ms
    start=mid-win_len//2
    frame=sig[start:start+win_len]*np.hamming(win_len)

    # LPC (our own implementation)
    a=lpc_coeffs(frame,lpc_order)

    roots=np.roots(a)
    roots=roots[np.imag(roots)>=0.01]

    ang=np.arctan2(np.imag(roots),np.real(roots))
    freqs=ang*sr/(2*np.pi)
    freqs=freqs[(freqs>90)&(freqs<sr/2)]
    freqs=np.sort(freqs)

    return freqs[:n_formants].tolist()

def estimate_f0_and_harmonics(sig:np.ndarray,sr:int,
                              fmin:float=80,fmax:float=400,
                              max_harmonics:int=10)->Tuple[float,List[float]]:
    """
    Estimate F0 using YIN; return F0 and harmonic frequencies.
    """
    f0_track=librosa.yin(sig,fmin=fmin,fmax=fmax,sr=sr)
    f0=float(np.median(f0_track[~np.isnan(f0_track)]))
    harmonics=[n*f0 for n in range(1,max_harmonics+1)
               if n*f0<sr/2]
    return f0,harmonics

def spectrum(sig:np.ndarray,sr:int,n_fft:int=4096
             )->Tuple[np.ndarray,np.ndarray]:
    """
    Single-sided magnitude spectrum.
    """
    sig=normalize(sig)
    win=np.hamming(len(sig))
    sigw=sig*win
    spec=np.fft.rfft(sigw,n=n_fft)
    mag=np.abs(spec)
    freqs=np.fft.rfftfreq(n_fft,1/sr)
    return freqs,mag

def plot_spectrum_with_formants(sig:np.ndarray,sr:int,
                                title:str="Vowel spectrum",
                                n_formants:int=3,
                                max_harmonics:int=10)->Tuple[List[float],List[float]]:
    """
    Plots magnitude spectrum and overlays:
      - vertical red lines at formants
      - grey dashed lines at harmonics
    Returns (formant_list,harmonic_list).
    """
    freqs,mag=spectrum(sig,sr)
    formants=estimate_formants_lpc(sig,sr,lpc_order=12,n_formants=n_formants)
    f0,harmonics=estimate_f0_and_harmonics(sig,sr,max_harmonics=max_harmonics)

    plt.figure(figsize=(10,4))
    plt.plot(freqs,mag,label="Spectrum")

    # Harmonics (grey dashed)
    for h in harmonics:
        plt.axvline(h,color="gray",linestyle="--",alpha=0.6)

    # Formants (red dashed, with labels)
    max_mag=max(mag)
    for i,f in enumerate(formants,1):
        plt.axvline(f,color="red",linestyle="--")
        plt.text(f,1.02*max_mag,
                 f"F{i}={f:.0f} Hz",
                 rotation=90,verticalalignment="bottom",
                 color="red")

    plt.xlim(0,sr/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"{title}\nEstimated F0={f0:.1f} Hz")
    plt.grid(True,alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Estimated formants (Hz):",formants)
    print("Harmonic frequencies (Hz):",[round(h,1) for h in harmonics])
    return formants,harmonics

# =========================
# Example usage / quick demo
# =========================

if __name__=="__main__":
    sr=16000
    dataset=create_synthetic_dataset(sr=sr,duration=0.6)

    # --- Q1: what did we create? ---
    for spk,words in dataset.items():
        print(spk,"->",list(words.keys()))

    # pick one sample: speaker_1 saying "aa"
    sig_aa=dataset["speaker_1"]["aa"]

    # --- Q2: visualise single sample in 3 domains ---
    visualize_sample(sig_aa,sr,title="speaker_1 / 'aa'")

    # --- Q3: spectrum + formants + harmonics ---
    plot_spectrum_with_formants(sig_aa,sr,
                                title="speaker_1 / 'aa' spectrum")
