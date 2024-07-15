import torch
from tqdm import tqdm
from src.evaluate import evaluate

# Get the current learning rate
def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Fit the model with learning rate scheduler
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        # Print the current learning rate
        current_lr = get_current_lr(optimizer)
        print(f'Epoch {epoch+1}/{epochs}, Learning Rate: {current_lr}')
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        # Step the scheduler
        scheduler.step(result['val_loss'])
    return history