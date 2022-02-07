import torch
from tqdm import tqdm
import numpy as np
import wandb

train_loss_history = []
val_loss_history = []
def model_train(model, optimizer, criterion, number_of_epochs, train_loader, val_loader, train_loss_history, val_loss_history,
                model_save_path):

    min_val_loss = np.inf
    for epoch in range(number_of_epochs):

        model.train()
        # Train Loop #
        epoch_loss = 0



        for feature_batch, target_batch in tqdm(train_loader):
            features, targets = feature_batch.to(torch.device('cuda:0')), target_batch.to(torch.device('cuda:0'))

            # Zero the gradients.
            optimizer.zero_grad()
            # Forward pass.
            model_outputs = model.forward(features)
            # Compute loss.
            loss = criterion(model_outputs, targets.reshape(-1, 1))
            # Backward pass. Gradients are stored along model's parameters.
            loss.backward()
            # Take a 'step' using the gradients computed.
            optimizer.step()
            # Save the loss.
            epoch_loss += loss.item()

        # Calculate loss per epoch.
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_epoch_loss)

        # Turn on evaluation mode. Forward pass the validation data and compute loss with no gradient computation.
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for val_features, val_targets in val_loader:
                val_inputs, val_targets = val_features.to(torch.device('cuda:0')), val_targets.to(torch.device('cuda:0'))

                val_outputs = model.forward(val_inputs)

                val_loss = criterion(val_outputs, val_targets.reshape(-1, 1))
                val_loss_sum += val_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(
            f"\n Starting epoch {epoch+2}, loss: {avg_epoch_loss}, val_loss: {avg_val_loss} lr= {optimizer.param_groups[0]['lr']}")

        wandb.log({"loss": avg_epoch_loss, "val_loss": avg_val_loss, "lr": optimizer.param_groups[0]['lr']})

        # Optional
        wandb.watch(model)
        # Save the model if avg. validation loss is smaller than the last.
        if min_val_loss > avg_val_loss:
            print(f"\n Validation loss {min_val_loss} --> {avg_val_loss}. Saving best model.")
            torch.save(model.state_dict(), model_save_path)
            min_val_loss = avg_val_loss