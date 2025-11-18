import numpy as np

def synth_vowel_from_formants(f0,formants,
                              duration=0.6,sr=16000,
                              bandwidths=(80,90,120)):
    """
    Low-level helper: given F0 and (F1,F2,F3) formants, 
    return a synthetic vowel signal.
    """
    t=np.linspace(0,duration,int(sr*duration),endpoint=False)
    sig=np.zeros_like(t)

    for f,b in zip(formants,bandwidths):
        env=np.exp(-b*t)                 # exponential decay envelope
        sig=sig+env*np.sin(2*np.pi*f*t)  # damped sinusoid for each formant

    # add fundamental (voicing)
    sig=sig+0.2*np.sin(2*np.pi*f0*t)

    # normalize
    sig=sig-np.mean(sig)
    sig=sig/(np.max(np.abs(sig))+1e-9)
    return sig

# Approximate male vowel formants (Hz)
MALE_VOWEL_FORMANTS={
    "aa":(730,1090,2440),   # /a/ as in 'father'
    "ee":(300,2200,2960),   # /i/ as in 'see'
    "oo":(300,870,2240),    # /u/ as in 'two'
}

def generate_male_vowel(vowel:str,
                        duration:float=0.6,
                        sr:int=16000,
                        f0:float=110.0):
    """
    Generate a synthetic male vowel.

    Parameters
    ----------
    vowel : str
        One of "aa", "ee", "oo".
    duration : float
        Length of signal in seconds (e.g., 0.6).
    sr : int
        Sample rate in Hz (e.g., 16000).
    f0 : float
        Fundamental frequency (pitch) in Hz.
        Typical male F0 ~ 90–140 Hz. Use:
            100–120 Hz for normal voice
            >140 Hz for higher-pitched male

    Returns
    -------
    y : np.ndarray
        Audio signal of shape (duration*sr,)
    sr : int
        Sample rate (returned for convenience)
    """
    if vowel not in MALE_VOWEL_FORMANTS:
        raise ValueError(f"Unknown vowel '{vowel}'. Use one of {list(MALE_VOWEL_FORMANTS.keys())}")
    formants=MALE_VOWEL_FORMANTS[vowel]
    y=synth_vowel_from_formants(f0,formants,duration=duration,sr=sr)
    return y,sr


# Female vowel formants – roughly higher than male
FEMALE_VOWEL_FORMANTS={
    "aa":(850,1220,2810),
    "ee":(370,2660,3200),
    "oo":(370,950,2670),
}

def generate_female_vowel(vowel:str,
                          duration:float=0.6,
                          sr:int=16000,
                          f0:float=220.0):
    """
    Generate a synthetic female vowel.

    Parameters
    ----------
    vowel : str
        One of "aa", "ee", "oo".
    duration : float
        Length of signal in seconds.
    sr : int
        Sample rate in Hz.
    f0 : float
        Fundamental frequency (pitch) in Hz.
        Typical female F0 ~ 180–250 Hz. Use:
            180–200 Hz for lower female
            220–240 Hz for average
            260+ Hz for higher

    Returns
    -------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    """
    if vowel not in FEMALE_VOWEL_FORMANTS:
        raise ValueError(f"Unknown vowel '{vowel}'. Use one of {list(FEMALE_VOWEL_FORMANTS.keys())}")
    formants=FEMALE_VOWEL_FORMANTS[vowel]
    y=synth_vowel_from_formants(f0,formants,duration=duration,sr=sr)
    return y,sr


# Center bands (Hz) for some consonants
CONSONANT_BANDS={
    "s": (4000,8000),  # high-frequency fricative
    "sh":(2500,6000),
    "f": (1500,4000),
    "p": (100,800),    # burst, low-mid
    "t": (2000,5000),
    "k": (500,2500),
}

def generate_consonant(symbol:str,
                       gender:str="male",
                       duration:float=0.25,
                       sr:int=16000,
                       noise_level:float=1.0):
    """
    Generate a simple synthetic consonant using band-limited noise.

    Parameters
    ----------
    symbol : str
        Consonant symbol key, e.g. "s", "sh", "f", "p", "t", "k".
    gender : str
        "male" or "female".  Female uses slightly higher band for realism.
    duration : float
        Length in seconds (shorter than vowels, e.g., 0.2–0.3).
    sr : int
        Sample rate in Hz.
    noise_level : float
        Overall amplitude scaling of the noise (1.0 is fine).

    Returns
    -------
    y : np.ndarray
        Audio signal of consonant.
    sr : int
        Sample rate.
    """
    if symbol not in CONSONANT_BANDS:
        raise ValueError(f"Unknown consonant '{symbol}'. Use one of {list(CONSONANT_BANDS.keys())}")

    low,high=CONSONANT_BANDS[symbol]

    # female bands slightly higher
    if gender.lower()=="female":
        low=int(low*1.2)
        high=int(high*1.2)

    n=int(duration*sr)
    # white noise
    noise=np.random.randn(n)

    # FFT band-pass filter (very simple)
    spec=np.fft.rfft(noise)
    freqs=np.fft.rfftfreq(n,1/sr)
    band_mask=(freqs>=low)&(freqs<=high)
    spec_filtered=np.zeros_like(spec)
    spec_filtered[band_mask]=spec[band_mask]
    y=np.fft.irfft(spec_filtered,n=n)

    # envelope so it sounds like a burst
    env=np.hanning(n)
    y=y*env*noise_level

    # normalize
    y=y/(np.max(np.abs(y))+1e-9)
    return y,sr




# Male vowels
male_aa,sr=generate_male_vowel("aa",duration=0.6,sr=16000,f0=110)
male_ee,_=generate_male_vowel("ee",f0=120)
male_oo,_=generate_male_vowel("oo",f0=100)

# Female vowels
female_aa,_=generate_female_vowel("aa",f0=210)
female_ee,_=generate_female_vowel("ee",duration=0.5,f0=230)
female_oo,_=generate_female_vowel("oo",f0=200)

# Consonants
male_s,_=generate_consonant("s",gender="male",duration=0.25)
female_sh,_=generate_consonant("sh",gender="female",duration=0.25)


visualize_sample(male_aa,sr,title="male 'aa'")
plot_spectrum_with_formants(male_aa,sr,title="male 'aa' spectrum")
