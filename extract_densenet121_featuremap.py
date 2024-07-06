import torch
import pandas as pd
from monai.transforms import Compose, LoadImage, AsChannelFirst, ScaleIntensity, EnsureType
from monai.networks.nets import DenseNet121
from monai.data import decollate_batch
from train_densenet121 import MedNISTDataset  # Assuming MedNISTDataset is defined in densenet121_train.py

def extract_feature_maps(model_pth, traindata_x, traindata_y):

    val_transforms = Compose([
        LoadImage(image_only=True),
        AsChannelFirst(),
        ScaleIntensity(),
        EnsureType()
    ])

    train_ds = MedNISTDataset(traindata_x, traindata_y, val_transforms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)

    # Load pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_pth))

    # modify model layers
    del model.class_layers.out

    # Set model to evaluation mode
    model.eval()
    
    # Initialize tensors for feature maps and labels
    featuremap_trainsample = torch.tensor([], dtype=torch.float32, device=device)
    y_trainsample = torch.tensor([], dtype=torch.float32, device=device)

    # Extract feature maps
    with torch.no_grad():
        for train_data in train_loader:
            train_images, train_labels = train_data[0][0:,0:1,0:,0:].to(device), train_data[1].to(device)
            feature = model(train_images)
            featuremap_trainsample = torch.cat((featuremap_trainsample, feature), 0)
            y_trainsample = torch.cat((y_trainsample, train_labels), 0)

    # Create a DataFrame for the feature maps
    featuremap_trainsample_df = pd.DataFrame(
        featuremap_trainsample.to("cpu").numpy(),
        index=traindata_x,
        columns=["V" + str(x + 1) for x in range(featuremap_trainsample.shape[1])]
    )
    featuremap_trainsample_df = pd.concat(
        [featuremap_trainsample_df, pd.DataFrame({'label': y_trainsample.to("cpu").numpy()}, index=traindata_x)],
        axis=1
    )

    return featuremap_trainsample_df

# Example usage
# featuremap_trainsample_df = extract_feature_maps(
#     "model_pth", traindata_x, traindata_y
# )
