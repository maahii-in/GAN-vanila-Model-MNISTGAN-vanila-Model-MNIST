# Generative Adversarial Networks (GANs) Collection

A clean, easy-to-understand PyTorch implementation of various Generative Adversarial Network (GAN) architectures evaluated across multiple popular datasets (MNIST, CIFAR-10, Stanford Cars, CelebA).

---

## 🚀 Features & Architectures
* **Vanilla GAN** (Basic MLP-based GAN)
* **DCGAN** (Deep Convolutional GAN)
* **CGAN** (Conditional GAN)
* **WGAN** (Wasserstein GAN)

---

## 🛠️ Requirements & Environment
* **Python**: `3.8+` (Recommended via Anaconda)[cite: 1]
* **PyTorch**: `1.x` or higher[cite: 1]
* **Jupyter Notebook** (Optional, for running `.ipynb` files locally)[cite: 1]

---

## 📂 Dataset Sources
* **MNIST / CIFAR-10**: Automatically downloaded via `torchvision.datasets`.
* **Stanford Cars Dataset**: Can be downloaded manually from the [Stanford Car Dataset Page](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)[cite: 1]. *(Note: Requires preprocessing before training)*[cite: 1].
* **CelebA**: Can be downloaded via official channels or torchvision.

---

## 💻 How to Run

### 1. Run via Google Colab (Recommended for Quick Testing)
You can run these notebooks instantly on Google Colab with free GPU acceleration (e.g., T4/K80)[cite: 1]:
1. Open [Google Colab](https://colab.research.google.com)[cite: 1].
2. Upload the desired `.ipynb` file (e.g., `VanillaGAN.ipynb`)[cite: 1].
3. Change the runtime type to **GPU** (`Runtime > Change runtime type > Hardware accelerator > GPU`)[cite: 1].
4. Run the cells sequentially! *(Training Vanilla GAN takes approximately 10 minutes on a standard Colab GPU)*[cite: 1].

### 2. Run Locally
If Jupyter notebooks fail to render correctly directly on GitHub, clone the repository and run them on your local server[cite: 1]:

```bash
# Clone the repository
git clone [https://github.com/maahii](https://github.com/maahii)
