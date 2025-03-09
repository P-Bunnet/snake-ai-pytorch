import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# ✅ Use Mac GPU (MPS) if available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth', extra_data=None):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        
        # ✅ Save model weights
        torch.save(self.state_dict(), file_name)
        
        # ✅ Save extra training parameters if provided
        if extra_data:
            torch.save(extra_data, os.path.join(model_folder_path, 'training_state.pth'))

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(device)  # ✅ Move model to MPS
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        # ✅ Move all tensors to the same device (MPS or CPU)
        state = torch.tensor(state, dtype=torch.float).clone().detach().to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).clone().detach().to(device)
        action = torch.tensor(action, dtype=torch.long).clone().detach().to(device)
        reward = torch.tensor(reward, dtype=torch.float).clone().detach().to(device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Predicted Q values with the current state
        pred = self.model(state)

        # 2. Clone the tensor before modification
        target = pred.clone().detach().to(device)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new  # Fixed indexing

        # 3. Training Step
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
