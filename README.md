# Investigating the outperformance of an AE for compression tasks

It is seen throughout literature and in practise that probabilistic models like the VAE and the VQ-VAE are outperformed by the basic autoencoder in purely compression tasks. This repository will contain my heuristic work to investigate this. Information:

- This is done with the RGZ108k dataset. This is the dataset my Master's project uses. [https://github.com/inigoval/byol]

## Autoencoder (AE) Compression Strategy

### Justification for Latent Dimensions
Below is how much smaller the latent dimensions are compared to the original image size (150×150×1 = 22,500 values).

| Compression Level | Latent Dim | Compression Factor |
|-------------------|-----------|--------------------|
| Minimal Compression | 8192  | 36.4% of original size|
| Balanced Compression | 4096  | 18.2% of original size|
| Higher Compression | 2048  | 9.1% of original size|

These values were chosen to ensure that the AE reduces dimensionality while retaining sufficient features for good reconstruction performance.

### 🛠️ How the AE Compresses Data
The Autoencoder reduces the original 150×150×1 input into a latent vector using three convolutional layers followed by a fully connected layer:
1. Convolutional Downsampling:
   - Extracts hierarchical features and reduces spatial dimensions.
   - The feature map (38×38×128) contains high-level representations of the input.
2. Fully Connected Compression:
   - This flattened representation is passed through a dense layer (`fc_mu`) to a latent vector of size 2048–8192.
3. Decoding & Reconstruction:
   - The decoder expands this latent vector back to a 150×150×1 image, reconstructing the input.

By using this strategy, the AE acts as an efficient lossy compressor, reducing storage needs while preserving key features for reconstruction.

