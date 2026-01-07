def mfcc_librosa_frame_7(waveform,sr):
  mfcc=librosa.feature.mfcc(y=waveform,sr=sr,n_mfcc=13)
  print("Librosa MFCC shape:",mfcc.T.shape)
  if mfcc.shape[1]<=7:
    print("Frame 7 not available. Total frames:",mfcc.shape[1])
    return None
  frame7=mfcc[:,7]
  print("MFCC values for frame 7:",frame7)

  plt.figure(figsize=(10,6))
  librosa.display.specshow(mfcc, sr=sr, x_axis='time')
  plt.axvline(x=librosa.frames_to_time(7,sr=sr),color='red',linestyle='--',label='Frame 7')
  plt.title("MFCC (Librosa) with Frame 7 highlighted")
  plt.colorbar()
  plt.legend()
  plt.show()