import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImage, AsChannelFirst, ScaleIntensity, 
    RandRotate, RandFlip, RandZoom, EnsureType, Activations, AsDiscrete
)
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.data import decollate_batch
from monai.utils import set_determinism
import time

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

def densenet_model_train(traindata_x, traindata_y, validdata_x, validdata_y, model_pth, **kwargs):
    # Define default parameters
    max_epochs = kwargs.get('max_epochs', 10)
    val_interval = kwargs.get('val_interval', 1)
    batch_size = kwargs.get('batch_size', 64)
    learning_rate = kwargs.get('learning_rate', 1e-5)

    # Define transforms
    train_transforms = Compose([
        LoadImage(image_only=True),
        AsChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType()
    ])

    val_transforms = Compose([
        LoadImage(image_only=True),
        AsChannelFirst(),
        ScaleIntensity(),
        EnsureType()
    ])

    y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
    y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    set_determinism(seed=0)
    # Create datasets and data loaders
    train_ds = MedNISTDataset(traindata_x, traindata_y, train_transforms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = MedNISTDataset(validdata_x, validdata_y, val_transforms)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1)

    # Set up device, model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize metrics
    auc_metric = ROCAUCMetric()
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    start = time.time()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0][0:, 0:1, 0:, 0:].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[0][0:, 0:1, 0:, 0:].to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i).to(device) for i in decollate_batch(y)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_pth)
                    print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current AUC: {result:.4f} current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f} at epoch: {best_metric_epoch}")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    end = time.time()
    print(end - start)

    Train_result_df = pd.DataFrame({
        'Epoch': range(1, epoch + 2),
        'Train_loss': epoch_loss_values,
        'Valid_AUC': metric_values
    })
    Train_result_df.to_csv("./output/TrainResult.csv", index=False)

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Training Epoch Average Loss", fontsize=20)
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch", fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Validation AUC", fontsize=20)
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch", fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.plot(x, y)
    plt.savefig("./output/TrainResult.tiff")
    plt.show()

    return model, best_metric, best_metric_epoch


# Example usage
# model, best_metric, best_metric_epoch = train_model(traindata_x, traindata_y, validdata_x, validdata_y, "example.pth")
