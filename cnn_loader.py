from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

x_train,x_test,y_train,y_test = train_test_split(new_X,y_tensor,test_size=0.2,random_state = 42)

train_dataset = TensorDataset(x_train,y_train)
test_dataset = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= 64, shuffle=True)