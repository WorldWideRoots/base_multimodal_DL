# trying to build a class that will train the model

import torch
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer_constructor, optimizer_params, scheduler_constructor, scheduler_params, loss_function, device, saving_mode='best'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer_constructor(self.model.parameters(), **optimizer_params)
        self.scheduler = scheduler_constructor(self.optimizer, **scheduler_params)
        self.loss_function = loss_function
        self.device = device
        self.saving_mode = saving_mode
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        self.scheduler.step()
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def save_model(self, epoch, val_loss):
        if self.saving_mode == 'best':
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"best_model_epoch_{epoch}.pth")
        elif self.saving_mode == 'all':
            torch.save(self.model.state_dict(), f"model_epoch_{epoch}.pth")

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
            self.save_model(epoch, val_loss)
        self.plot_losses()

    def plot_losses(self):
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


class DynamicFrozenTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, freeze_epochs, optimizer_params_list, scheduler_params_list, loss_function, device, saving_mode='best'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.device = device
        self.saving_mode = saving_mode
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.freeze_epochs = freeze_epochs or {}
        self.val_losses = []
        self.optimizer_params_list = optimizer_params_list
        self.scheduler_params_list = scheduler_params_list
        self.current_phase = 0  # Index to track the current phase
        self._update_optimizer_and_scheduler()  # Initialize optimizer and scheduler

    def _update_optimizer_and_scheduler(self):
        # Update optimizer and scheduler based on the current phase
        optimizer_constructor, optimizer_params = self.optimizer_params_list[self.current_phase]
        scheduler_constructor, scheduler_params = self.scheduler_params_list[self.current_phase]
        self.optimizer = optimizer_constructor(self.model.parameters(), **optimizer_params)
        self.scheduler = scheduler_constructor(self.optimizer, **scheduler_params)

    def switch_phase(self, new_phase):
        # Switch to a new phase (freeze/unfreeze) and update optimizer and scheduler
        self.current_phase = new_phase
        self._update_optimizer_and_scheduler()

    def freeze_layers(self, layer_names):
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
        self.switch_phase(self.current_phase+1)  # Re-initialize optimizer and scheduler with updated parameters

    def unfreeze_layers(self, layer_names):
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
        self.switch_phase(self.current_phase+1)  # Re-initialize optimizer and scheduler with updated parameters

    def train_one_epoch(self, epoch):
        # Check if we need to freeze/unfreeze at this epoch
        if epoch in self.freeze_epochs:
            action = self.freeze_epochs[epoch]
            if action == 'freeze':
                self.freeze_model(self.freeze_epochs[epoch]['modules'])
            elif action == 'unfreeze':
                self.unfreeze_model(self.freeze_epochs[epoch]['modules'])

        # Proceed with normal training
        return super().train_one_epoch()
