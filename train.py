import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser(description='Train a neural network on a flower dataset')
    parser.add_argument('data_directory', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Choose architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    # Data loading and preprocessing
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.Resize(255),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


    test_transform = transforms.Compose([  transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_dataset =  datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset =  datasets.ImageFolder(test_dir, transform=test_transform)
    valid_dataset =  datasets.ImageFolder(valid_dir, transform=test_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    # Model creation
    model = models.__dict__[args.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1))
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # GPU usage if available
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    epoch = args.epochs     
    steps = 0
    print_every = 103
    running_loss = 0
    training_data ={'traing_loss':[],'validation_loss':[], 'Validation_accuracy':[]}
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for batches, labels in train_loader:
            if steps %8 == 0 : print('.',end=''); 
            steps += 1
            inputs, labels = batches.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            #print(loss)
            running_loss += loss.item()

            if steps % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Step {steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}")
                training_data['traing_loss'].append(running_loss/print_every)
                running_loss = 0  # Reset running_loss after printing

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # No gradient calculations during validation
            val_loss = 0
            for batches, labels in valid_loader:  # Create a separate val_loader for validation
                inputs, labels = batches.to(device), labels.to(device)

                output = model.forward(inputs)
                loss = criterion(output, labels)

                val_loss += loss.item()

                # Calculate validation accuracy
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Validation loss: {val_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {val_accuracy:.2f}%")

        training_data['validation_loss'].append(val_loss/len(valid_loader))
        training_data['Validation_accuracy'].append(val_accuracy)

        model.train()  # Set the model back to training mode after validation
        #torch.save(model, 'checkpoint_'+str(epoch)+'_.pth')

    print("Training finished.")                                     
    # Save checkpoint
    model.class_to_idx = train_data.class_to_idx
    
    torch.save(model, args.save_dir + '/final_checkpoint.pth')

if __name__ == '__main__':
    main()
