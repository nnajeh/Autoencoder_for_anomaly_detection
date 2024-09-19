from libraries import *
from model import *



criterion = nn.MSELoss()
optimizer = optim.Adadelta(net.parameters())

def train_one_epoch(epoch: int, train_losses: torch.Tensor) -> torch.Tensor:
    """This function is part of the training loop and trains the model for one epoch"""
    # Print the epoch
    print(f'Starting epoch {epoch + 1}')

    # Batch's train losses for this epoch
    batches_train_losses_for_this_epoch: List[float] = []

    # Variable use to store the running losses
    running_loss_since_last_report = 0.0

    # Iterate over the data in batches
    for i, batch_data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs: torch.Tensor = batch_data[0].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs: torch.Tensor = net(inputs)
        loss: nn.Module = criterion(outputs, inputs)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # Store the loss for this batch
        batches_train_losses_for_this_epoch.append(loss.item())

    # Store the training losses for this epoch for reports
    ## Convert the list to a tensor
    # batches_train_losses_for_this_epoch = torch.tensor(batches_train_losses_for_this_epoch, dtype=torch.float32, device = 'cpu')
    ## Compute the mean
    batches_train_losses_for_this_epoch = torch.tensor(batches_train_losses_for_this_epoch, dtype=torch.float32, device = 'cpu')
    average_loss_for_this_epoch: torch.Tensor = torch.mean(batches_train_losses_for_this_epoch)
    ## Store the epoch's train loss
    train_losses = torch.cat((train_losses, average_loss_for_this_epoch.unsqueeze(0)), 0)

    # Print the epoch's train loss
    print(f'End of training for epoch {epoch + 1} with average loss: {average_loss_for_this_epoch.item():.3f}')
    
    return train_losses




def test_one_epoch(epoch: int, test_losses: torch.Tensor) -> torch.Tensor:
    """This function is part of the training loop and is used to test the model"""

    # Run the model on the test data
    _losses, _labels, batch_losses, average_loss = run(data_loader=testloader_conform)

    # Store the training losses for this epoch for reports
    test_losses = torch.cat((test_losses, average_loss.unsqueeze(0)), 0)

    print(f'End of testing for epoch {epoch + 1} with average loss: {average_loss:.3f}')
    
    return test_losses



def is_raising(tensor: torch.Tensor, patience: int = 5, threshold: float = 1e-9) -> bool:
    """This function checks if a 1 dimensional tensor is raising.
    It is used to check if the performances are degrading on the test set.
    """
    if len(tensor) < patience:
        return False
    for i in range(1, patience + 1):
        if tensor[-i] < tensor[-i - 1] + threshold:
            return False
    return True




def run(data_loader=None, print_report = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """This function run the model and provide the loss vs the label"""

        # Prepare some variables to store the data
        ## Image level
        losses: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        ## Batch level
        batches_losses: List[torch.Tensor] = []

        # No need to store the gradients for inference
        with torch.no_grad():
            # Iterate over the data in batches
            for i, batch_data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = batch_data[0].to(device)

                # forward
                outputs = net(inputs)

                # Compute the loss for the batch
                batch_loss = criterion(outputs, inputs)
                batches_losses.append(batch_loss)

                for j, label in enumerate(batch_data[1]):
                    loss = criterion(outputs[j], inputs[j])
                     # Store the loss for this batch
                    losses.append(loss)
                    labels.append(label)

        # Stack to tensor
        losses: torch.Tensor = torch.stack(losses).cpu()
        labels: torch.Tensor = torch.stack(labels).cpu()
        batches_losses: torch.Tensor = torch.stack(batches_losses).cpu()

        # average loss
        average_loss: torch.Tensor = torch.mean(losses)

        # Print the loss
        if print_report:
            print(f'Finished running {self.p.network_name if self.p.network_name else "model"} with loss {average_loss.item():.3f}')

        # Return the loss and the labels
        return losses, labels, batches_losses, average_loss




 def train() -> Tuple[torch.Tensor, torch.Tensor]:
    """This function trains the model"""
    # We reinitialize the report containers as 1 dimensional empty tensors
    train_losses = torch.empty((0), dtype=torch.float32, device = 'cpu')
    test_losses = torch.empty((0), dtype=torch.float32, device = 'cpu')

    # Iterate over the training data multiple times
    for epoch in range(epochs):

        # Train for one epoch
        train_losses = train_one_epoch(epoch, train_losses)

        # Evaluate the model on the test set
        test_losses = test_one_epoch(epoch, test_losses)

        # Save the model if it is the best
        if epoch > 5:
            if test_losses[-1] < torch.min(test_losses[:-1]):
                torch.save(net.state_dict(), './best_model.pth')

        # Early stopping
        if is_raising(tensor=test_losses, patience=10):
            break

    print(f'Finished training')
    print(f'Training loss: {train_losses[-1].item():.3f}')
    print(f'Test loss: {test_losses[-1].item():.3f}')

    # Load the best model
    net.load_state_dict(torch.load('./best_model.pth'))
    
    # Return the losses
    return train_losses, test_losses
train_losses, test_losses = train()
