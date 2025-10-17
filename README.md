<p align="center">A project to denoise spectral data using a 1D Residual U-Net, implemented in PyTorch.<a href="https://github.com/nabhya8013/spectral_denoise/issues">Report Bug</a>·<a href="https://github.com/nabhya8013/spectral_denoise/issues">Request Feature</a></p></div><div align="center"></div><details><summary>Table of Contents</summary><ol><li><a href="#about-the-project">About The Project</a><ul><li><a href="#built-with">Built With</a></li></ul></li><li><a href="#getting-started">Getting Started</a><ul><li><a href="#prerequisites">Prerequisites</a></li><li><a href="#installation">Installation</a></li></ul></li><li><a href="#usage">Usage</a></li><li><a href="#training-and-evaluation">Training and Evaluation</a></li><li><a href="#results">Results</a></li><li><a href="#contributing">Contributing</a></li><li><a href="#license">License</a></li><li><a href="#contact">Contact</a></li></ol></details>About The ProjectThis project provides a complete workflow for training and evaluating a 1D ResUNet model for spectral denoising. The model architecture is based on the U-Net design with residual connections, which is effective for training deeper networks and achieving better performance in signal processing tasks.The key components of this project are:Data Augmentation: The training data is augmented with various types of noise, baseline shifts, and cosmic ray spikes to improve the model's robustness.1D ResUNet Model: A deep learning model tailored for 1D sequential data like spectra.Hybrid Loss Function: A combination of Mean Squared Error and Cosine Similarity is used to optimize the model for both pixel-level accuracy and structural similarity.Comprehensive Evaluation: The model's performance is assessed using multiple metrics, including MSE, PSNR, SSIM, and Pearson correlation.Built WithGetting StartedTo get a local copy up and running follow these simple example steps.PrerequisitesThis project uses conda for environment management. Make sure you have Anaconda or Miniconda installed.InstallationClone the repoBashgit clone https://github.com/nabhya8013/spectral_denoise.git
cd spectral_denoise
Create and activate the Conda environment1Bashconda create -n spectral_env python=3.10
conda activate spectral_env
Install the required packages2Bashpip install -r requirements.txt
Usage3To use the pretrained model for denoising your own spectral data, you can adapt the notebooks/demo_analysis.ipynb notebook. The basic steps are:4Load the trained model.5Load your raw spectral d6ata.Preprocess the data (e.g., resampling to the target length of 1024).Pass the data through the model to get the denoised spectrum.Here's a code snippet from the demo notebook:Pythonimport torch
import numpy as np
from scipy.signal import resample
from scripts.train_resunet import ResUNet1D

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUNet1D().to(device)
model.load_state_dict(torch.load("models/resunet1d.pth", map_location=device))
model.eval()

# Load and preprocess your data
raw_spectrum = np.loadtxt("path/to/your/spectrum.txt")
raw_spectrum_resampled = resample(raw_spectrum, 1024)
noisy_tensor = torch.tensor(raw_spectrum_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Denoise
with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

denoised_spectrum = denoised_tensor.cpu().squeeze().numpy()
Training and EvaluationTo train the model from scratch, run the train_resunet.py script. Your data should be in the data/pairs directory, with _clean.npy and _noisy.npy file pairs.Bashpython scripts/train_resunet.py
The script will:Split the data into training and validation sets.Augment the training data.Train the ResUNet1D model.Save the trained model to models/resunet1d.pth.Evaluate the model and save the metrics.ResultsThe model's performance is evaluated on a validation set, and the results are saved in results/eval_metrics.json. Here's a summary of the latest evaluation:MetricValueMean MSE0.001908Mean PSNR42.68 dBMean SSIM0.9927Mean Correlation0.9911Overall Quality95.17%ContributingContributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".Don't forget to give the project a star! Thanks again!Fork the ProjectCreate your Feature Branch (git checkout -b feature/AmazingFeature)Commit your Changes (git commit -m 'Add some AmazingFeature')Push to the Branch (git push origin feature/AmazingFeature)Open a Pull Request
