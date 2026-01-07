Target_Width=32
Target_Height=26

def prepare_dataset(X):
  tensor_list = []
  for x in X:
    tensor = torch.tensor(x).T
    current_width = tensor.shape[1]
    if current_width > Target_Width:
      tensor = tensor[:,:Target_Width]
    elif current_width < Target_Width:
      pad_amount = Target_Width - current_width
      tensor = pad(tensor, (0,pad_amount,0,0), mode="constant", value=0)
    tensor_list.append(tensor)
  data_tensor = torch.stack(tensor_list)
  data_tensor = data_tensor.unsqueeze(1)
  mean = data_tensor.mean()
  std = data_tensor.std()
  data_tensor = ( data_tensor - mean ) / std
  return data_tensor

new_X = prepare_dataset(X)