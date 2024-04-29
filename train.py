import torch
from model import *

class TrainSource():
    def __init__(
            self, 
            model: torch.nn.Module, 
            optim: torch.optim.Optimizer, 
            criterion: torch.nn.Module, 
            X: torch.Tensor, 
            y: torch.Tensor,
            configs
            ):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.X = X
        self.y = y
        self.configs = configs

    def step(self):
        self.optim.zero_grad()
        output = self.model(self.X)
        loss = self.criterion(output, self.y)
        loss.backward()
        self.optim.step()
        return loss
    
    def train(self, max_epochs):
        epoch = 0
        convergence_threshold = 1e-10
        prev_loss = 1e10
        loss = self.step()
        while (epoch < max_epochs) and (prev_loss > 1e-3) and (abs(loss.item() - prev_loss) > convergence_threshold):
            prev_loss = loss.item()
            loss = self.step()
            if epoch % 50 == 0:
                print(f"\t Epoch {epoch} loss: {loss.item(): .3f}")
            epoch += 1
        print(f"\t Epoch {epoch} loss: {loss.item(): .3f}")

class TrainTarget():
    def __init__(
            self, 
            model: torch.nn.Module, 
            representation,
            optim, 
            criterion, 
            X: torch.Tensor, 
            y: torch.Tensor,
            configs
            ):
        self.model = model
        self.B = representation
        self.optim = optim
        self.criterion = criterion
        self.X = X
        self.y = y
        self.configs = configs
        self.final_loss = None

    def step(self):
        self.optim.zero_grad()
        representation = self.X @ self.B
        prediction = self.model(representation)
        loss = self.criterion(prediction, self.y)
        loss.backward()
        self.optim.step()
        return loss
    
    def train(self, max_epochs):
        epoch = 0
        convergence_threshold = 1e-10
        prev_loss = 1e10
        loss = self.step()
        while (epoch < max_epochs) and (prev_loss > 1e-3) and (abs(loss.item() - prev_loss) > convergence_threshold):
            prev_loss = loss.item()
            loss = self.step()
            if epoch % 50 == 0:
                print(f"\t Epoch {epoch} loss: {loss.item(): .3f}")
            epoch += 1
        print(f"\t Epoch {epoch} loss: {loss.item(): .3f}")
        self.final_loss = loss.item()

def get_optimal_B(configs, X, y):
    model = MTLDLR(configs)
    source_lr = configs.get('source_lr', 0.01)
    source_max_iter = configs.get('source_max_iter', 1000)
    
    source_optim = torch.optim.Adam(model.parameters(), lr=source_lr)
    source_criterion = torch.nn.MSELoss()

    train_source = TrainSource(
        model, 
        source_optim, 
        source_criterion, 
        X, 
        y, 
        configs)
    
    print("Training source tasks")
    train_source.train(source_max_iter)
    optimal_B = train_source.model.B.detach()
    return optimal_B