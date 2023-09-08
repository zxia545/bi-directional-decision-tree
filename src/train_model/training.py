from torch.optim import Adam
from torch import nn, save
from tqdm import tqdm
import torch
from dataloader import load_simple_classifer_data, load_dtnn_data
from os import path
from classifer import DynamicClassifier

def train(train_csv_file_path, validation_csv_file_path, learning_rate, epochs, device_ids, dataset_name):
    # Initialize DataLoader for training and validation datasets
    train_dataloader, num_classes, feature_size = load_dtnn_data(train_csv_file_path, batch_size=16, shuffle=True)
    validation_dataloader, _, _ = load_dtnn_data(validation_csv_file_path, batch_size=16, shuffle=False)


    # Initialize the model
    model = DynamicClassifier(input_size=feature_size, hidden_size=feature_size*2, output_size=num_classes)
    
    # Device setup
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # If there are multiple GPUs, wrap the model with nn.DataParallel 
    if torch.cuda.device_count() > 1 and len(device_ids) > 1:
        print("Let's use", len(device_ids), "GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids)

    model = model.to(device)
    criterion = criterion.to(device)

    # Training loop
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        total_samples_train = 0  # Keep track of the total number of samples

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            train_label = train_label.long()
            train_input = train_input.to(device)

            # Forward pass
            output = model(train_input)

            # Compute loss
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            # Compute accuracy
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            total_samples_train += train_label.size(0)  # Add the batch size

            # Backward pass and optimization
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # Validation loop for accuracy
        model.eval()
        total_acc_val = 0
        total_samples_val = 0
        with torch.no_grad():
            for val_input, val_label in validation_dataloader:
                val_input = val_input.to(device)
                val_label = val_label.to(device).long()
                
                output = model(val_input)
                acc = (output.argmax(dim=1) == val_label).sum().item()
                
                total_acc_val += acc
                total_samples_val += val_label.size(0)
                
        val_accuracy = total_acc_val / total_samples_val
        model.train()

        print(
            f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / total_samples_train: .3f} 
            | Train Accuracy: {total_acc_train / total_samples_train: .3f}
            | Validation Accuracy: {val_accuracy: .3f}''')

    save(model.layer, path.join(path.dirname(path.abspath(__file__)), f'{dataset_name}.pt'))

            
if __name__ == "__main__":
    train_csv_file_path = '/home/data/zhengyuhu/dataset/DT_NN/datasets/optdigits_pos0.train.csv'
    validation_csv_file_path = '/home/data/zhengyuhu/dataset/DT_NN/datasets/optdigits_pos0.test.csv'
    learning_rate = 1e-4
    epochs = 10

    device_ids = [0,1]
    dataset_name = 'optdigits_pos0'

    train(train_csv_file_path, validation_csv_file_path, learning_rate, epochs, device_ids, dataset_name)