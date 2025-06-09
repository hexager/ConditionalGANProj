# Synergistic Regularization for Data-Efficient GANs

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
## References
BigGAN-deep
Andrew Brock, Jeff Donahue, Karen Simonyan.
"Large Scale GAN Training for High Fidelity Natural Image Synthesis"
International Conference on Learning Representations (ICLR), 2019
[Paper](https://arxiv.org/abs/1809.11096v2)

Augmentation-Aware Self-Supervision for Data-Efficient GAN Training
Liang Hou, Qi Cao, Huawei Shen, Siyuan Pan, Xiaoshuang Li, Xueqi Cheng
Neural Information Processing Systems, 2022
[Paper](https://arxiv.org/pdf/2205.15677)

Deceive D: Adaptive Pseudo Augmentation for GAN Training
Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy
Neural Information Processing Systems, 2021
[Paper](https://arxiv.org/abs/2111.06849)
