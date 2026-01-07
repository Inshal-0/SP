def narrow_spec(wavrform,sr):
  D_narrow=librosa.stft(waveform,n_fft=2048,hop_length=512)
  S_narrow=librosa.amplitude_to_db(np.abs(D_narrow),ref=np.max)

  plt.figure(figsize=(10,6))
  librosa.display.specshow(S_narrow,sr=sr,hop_length=512,x_axis='time',y_axis='hz')
  plt.colorbar(label="dB")
  plt.title("Narrowband Spectrogram (Harmonics Visible)")
  plt.ylim(0,3000)
  plt.tight_layout()
  plt.show()

def wide_spec(waveform,sr):
  D_wide=librosa.stft(waveform,n_fft=512,hop_length=128)
  S_wide=librosa.amplitude_to_db(np.abs(D_wide),ref=np.max)

  plt.figure(figsize=(10,6))
  librosa.display.specshow(S_wide,sr=sr,hop_length=128,x_axis='time',y_axis='hz')
  plt.colorbar(label="dB")
  plt.title("Wideband Spectrogram (Formant Transitions)")
  plt.ylim(0,4000)
  plt.tight_layout()
  plt.show()