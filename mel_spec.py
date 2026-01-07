def mel_spec(waveform,sr):
  D=librosa.feature.melspectrogram(y=waveform,sr=sr, n_mels=64, fmax=8000)
  S_mel_db=librosa.amplitude_to_db(np.abs(D),ref=np.max)

  plt.figure(figsize=(10,6))
  librosa.display.specshow(S_mel_db,x_axis='time', y_axis='mel', sr=sr, fmax=8000)
  plt.colorbar(label="db")
  plt.title("Log mel")
  plt.show()