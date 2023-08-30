from torch.optim import Adam
from torch import nn, save
from tqdm import tqdm
import torch
from dataloader import load_simple_classifer_data
from os import path

def train(model, train_json_file_path, learning_rate, epochs):
    train_dataloader = load_simple_classifer_data(train_json_file_path, batch_size=16, shuffle=True)
    
    # Device setup
    device_ids = [0]
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
  
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # If there are multiple GPUs, wrap the model with nn.DataParallel 
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids)
  
    model = model.to(device)
    criterion = criterion.to(device)

    # 开始进入训练循环
    for epoch_num in range(epochs):
        #  定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        total_samples_train = 0  # Keep track of the total number of samples

        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            train_label = train_label.long()
            input_id = train_input.squeeze(1).to(device)
            # 模型计算
            output = model(input_id)
            # 计算损失
            output = output.squeeze(-1)
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            total_samples_train += train_label.size(0)  # Add the batch size

            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            
        print(
                f'''Epochs: {epoch_num + 1} 
                | Train Loss: {total_loss_train / total_samples_train: .3f} 
                | Train Accuracy: {total_acc_train / total_samples_train: .3f}''')

    return save(model.layer, path.join(path.dirname(path.abspath(__file__)), 'wall_robert.pt'))
            
if __name__ == "__main__":
    from classifer import SampleClassifer
    # NOTE: This will tell the number of hidden layers in the model
    num_of_hidden_layers = 3
    output_feature = 4
    model = SampleClassifer(hid=num_of_hidden_layers,output_feature=output_feature)
    train_json_file_path = '/home/zxia545/_Code/Research_repo/bi-directional-decision-tree/train_data/sensor_readings_2.data'
    learning_rate = 1e-4
    epochs = 50

    train(model, train_json_file_path, learning_rate, epochs)