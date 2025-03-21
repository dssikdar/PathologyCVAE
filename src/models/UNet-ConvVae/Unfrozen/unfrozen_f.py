import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import segmentation_models_pytorch as smp

# Define transformations
target_image_size = 64
transform = transforms.Compose([
    transforms.Resize((target_image_size, target_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset class
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        all_image_paths = glob.glob(os.path.join(root_dir, "*/*/*.png"), recursive=True)
        self.transform = transform
        print(f"Found {len(all_image_paths)} images in {root_dir}")  # Debugging statement
        self.image_paths = [path for path in all_image_paths if os.path.normpath(path).split(os.sep)[-2] == '0']
        # Debugging: Count the number of images in each class
        # self.num_class_0 = sum(1 for path in self.image_paths if os.path.normpath(path).split(os.sep)[-2] == '0')
        # self.num_class_1 = sum(1 for path in self.image_paths if os.path.normpath(path).split(os.sep)[-2] == '1')
        # print(f"Number of class 0 images: {self.num_class_0}")
        print(f"Number of kept images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")
            return None  # Handle this properly in DataLoader

        label = 0  # Since we are only keeping class 0 images
        if self.transform:
            image = self.transform(image)

        return image, label

# Model Definitions
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=28, freeze_encoder=True):
        super(UNetEncoder, self).__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_channels, classes=latent_dim
        )
        self.encoder = self.unet.encoder
        self.fc_mu = nn.Sequential(
            nn.Linear(512 * (target_image_size // 32) * (target_image_size // 32), 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(512 * (target_image_size // 32) * (target_image_size // 32), 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1].view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class UNetDecoder(nn.Module):
    def __init__(self, latent_dim=28, out_channels=3):
        super(UNetDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * (target_image_size // 32) * (target_image_size // 32))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 3, 2, 1, 1), nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, target_image_size // 32, target_image_size // 32)
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=28, freeze_encoder=True):
        super(VAE, self).__init__()
        self.encoder = UNetEncoder(latent_dim=latent_dim, freeze_encoder=freeze_encoder)
        self.decoder = UNetDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

### **Ensure This is Inside `if __name__ == '__main__'`**
if __name__ == '__main__':
    # Determine dataset path
    slurm = False
    if slurm:
        dataset_path = os.path.join(os.getenv("TMPDIR", "/data/class/cs175b/xujg"), "IDC_regular_ps50_idx5")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, "IDC_regular_ps50_idx5")

    print(f"Dataset path resolved to: {dataset_path}")

    # Load dataset
    full_dataset = BreastCancerDataset(dataset_path, transform=transform)
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    use_subset = True  # Change to False for full dataset
    if use_subset:
        subset_size = 100  # Reduce dataset size for testing
        val_subset_size = int(subset_size * 0.1)
        test_subset_size = int(subset_size * 0.1)

        if len(train_dataset) > 0:
            train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
            train_dataset = Subset(train_dataset, train_indices)
        else:
            print("Warning: train_dataset is empty!")

        if len(val_dataset) > 0:
            val_indices = np.random.choice(len(val_dataset), val_subset_size, replace=False)
            val_dataset = Subset(val_dataset, val_indices)
        else:
            print("Warning: val_dataset is empty!")

        if len(test_dataset) > 0:
            test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)
            test_dataset = Subset(test_dataset, test_indices)
        else:
            print("Warning: test_dataset is empty!")

    batch_size = 128  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Final dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # script_dir = os.getcwd()
    # model_path = os.path.join(script_dir, 'finals')
    # model_path = os.path.join(script_dir, 'unfrozen_f')
    # model_path = os.path.join(script_dir, 'vae_encoder.pth')
    # print("File exists:", os.path.exists(model_path))

    # Training Loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim=96, freeze_encoder=False).to(device)
    # vae.load_state_dict(torch.load(model_path, map_location=device))
    optimizer = torch.optim.AdamW(vae.parameters(), lr=5e-4, weight_decay=1e-5)

    for epoch in range(1):
        vae.train()
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images)
            loss = loss_function(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {train_loss/len(train_loader.dataset)}")


    optimizer = torch.optim.AdamW(vae.parameters(), lr=0.0005, weight_decay=1e-5)

    epochs = 1  # Additional training
    for epoch in range(epochs):
        print(f"Epoch #{epoch+1} ", end="")

        vae.train()
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            print(".", end="")
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images)
            loss = loss_function(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f" | Loss: {train_loss/len(train_loader.dataset)}")

    vae.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images.to(device)
        recon_images, _, _ = vae(test_images)

    print(f"Shape of test_images: {test_images.shape}")
    print(f"Shape of recon_images: {recon_images.shape}")

    plt.figure(figsize=(10, 5))

    # Select 5 random indices from the batch
    num_images = min(5, test_images.shape[0])
    random_indices = random.sample(range(test_images.shape[0]), num_images)

    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)

        img = test_images[idx].cpu()
        if img.shape[0] == 1:  # Convert grayscale to RGB
            img = img.repeat(3, 1, 1)

        plt.imshow((img.permute(1, 2, 0) + 1) / 2)  # Denormalize
        plt.axis("off")

        plt.subplot(2, 5, i + 6)

        img_recon = recon_images[idx].cpu()
        if img_recon.shape[0] == 1:  # Convert grayscale to RGB
            img_recon = img_recon.repeat(3, 1, 1)

        plt.imshow((img_recon.permute(1, 2, 0) + 1) / 2)  # Denormalize
        plt.axis("off")


    output_path = os.path.join(script_dir , 'output')
    image_path = os.path.join(output_path, 'images.png')

    print("Saving images...")
    plt.savefig(image_path)
    print("File exists:", os.path.exists(image_path))
    print(f"Image saved at: {image_path}")



    vae.eval()  # Set VAE to evaluation mode

    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, _ = vae.encoder(images)  # Extract latent representation (mean)
            all_latents.append(mu.cpu().numpy())  # Move to CPU and convert to NumPy
            all_labels.append(labels.numpy())

    # Concatenate all batches
    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Latent space shape: {all_latents.shape}")  # Should be (num_samples, latent_dim)

    # Detect unique classes (since subset may have <10 classes)
    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)

    # Ensure perplexity is valid
    n_samples = all_latents.shape[0]
    perplexity = min(30, n_samples - 1)  # Perplexity must be < n_samples

    # Apply t-SNE to reduce dimensions (e.g., from 28D → 2D)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latent_2d = tsne.fit_transform(all_latents)

    # Use a colormap with `num_classes` distinct colors
    cmap = plt.get_cmap("tab10", num_classes)  # "tab10" ensures distinct colors

    # Plot t-SNE scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=all_labels, cmap=cmap, alpha=0.7)

    # Create colorbar with only detected classes
    cbar = plt.colorbar(scatter, ticks=unique_classes)  # Show only present classes
    cbar.set_label("Digit Label")

    plt.title("t-SNE Visualization of VAE Latent Space (Subset)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")


    plot_path = os.path.join(output_path, 'plot.png')

    print("Saving t-SNE plot...")
    plt.savefig(plot_path)
    print("File exists:", os.path.exists(plot_path))
    print(f"Plot saved at: {plot_path}")

    encoder_path = os.path.join(output_path, "vae_encoder.pth")

    # Save only the encoder's state_dict
    torch.save(vae.encoder.state_dict(), encoder_path)

    print(f"Encoder weights saved at: {encoder_path}")
