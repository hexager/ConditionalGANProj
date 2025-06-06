# BigGAN-Deep Lite: Conditional Image Generation with AugSelf & Pseudo-Augmentation

This project implements a simplified, lightweight version of the BigGAN-Deep architecture in PyTorch for conditional image generation. It focuses on stable and high-quality image synthesis, incorporating advanced training techniques like **AugSelf** (a self-supervised augmentation for the discriminator) and **Pseudo-Augmentation (APA)** to enhance performance, particularly when working with limited datasets.
The model is trained on the CIFAR-10 dataset, generating images conditioned on their respective classes.
## Features

* **Conditional Image Generation:** Generate diverse images conditioned on specific class labels (e.g., CIFAR-10 classes).
* **BigGAN-Deep Architecture:** A streamlined implementation of the powerful BigGAN-Deep generator and discriminator, featuring:
    * **Residual Blocks** 
    * **Conditional BatchNorm**
    * **Spectral Normalization**
    * **Global Sum Pooling**

* **Advanced Training Enhancements:**
    * **AugSelf (Self-Supervised Augmentation):** The discriminator is trained to classify the type of augmentation applied to real images, significantly improving its robustness and the overall GAN performance.
    * **Pseudo-Augmentation (APA):** A technique to dynamically adjust the probability of mixing real and fake images during discriminator training, further enhancing stability and mitigating mode collapse, especially on smaller datasets.

* **Hinge Loss:** Utilizes Hinge loss for both generator and discriminator, a common and effective loss function for GANs.

* **Experiment Tracking:** Integrated with `wandb` (Weights & Biases) for real-time logging of training metrics, loss curves, and generated image samples.

* **Evaluation Metrics:** Includes code for calculating:
    * **FID (Frechet Inception Distance):** A metric for assessing the quality and diversity of generated images.
    * **Inception Score (IS):** Another metric to evaluate image quality and diversity.
## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* `wandb` 
* `torchmetrics` 
* `torch-fidelity` 
