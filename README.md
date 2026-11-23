# Assignment 2: Image Synthesis Using GANs

**ETMMGA304 – Generative AI and Its Applications** · MCA, K.R. Mangalam University
· Lab 2 (lab manual pages 7–9)

Trains an MLP generator and discriminator on Fashion-MNIST with alternating BCEWithLogitsLoss updates, and saves grids from fixed noise at epochs 1, 5 and 10 plus a loss plot.

**Settings:** G 64→256→512→784 (LeakyReLU 0.2, Tanh), D 784→512→256→1 logit; Adam lr 2e-4, betas (0.5, 0.999), batch 128, 10 epochs, seed 42.

## Files

```
GenAI-Assignment-2/
├── README.md
├── LAB_RECORD.md        # observation tables, results, viva answers
├── requirements.txt
├── lab2_gan_fashion_mnist.py
├── lab2_gan_fashion_mnist.ipynb
├── data/                # created automatically (datasets)
└── lab_outputs/         # created automatically (results)
```

## Setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/Aman13082001/GenAI-Assignment-2.git
cd GenAI-Assignment-2
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Hardware:** Runs on CPU in about 10 minutes. A GPU (Colab T4) is faster.

### CPU or CUDA GPU

On Windows, `pip install -r requirements.txt` installs the CPU build of PyTorch.
For an NVIDIA GPU, install the CUDA build first (choose the CUDA version for
your driver at [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check with `python -c "import torch; print(torch.cuda.is_available())"`.

## Run

```bash
python lab2_gan_fashion_mnist.py    # options: --epochs 10 --batch-size 128
```

Outputs:

- `lab_outputs/gan_epoch_01.png`
- `lab_outputs/gan_epoch_05.png`
- `lab_outputs/gan_epoch_10.png`
- `lab_outputs/gan_losses.png`
- `lab_outputs/fashion_generator.pt`

## Google Colab

Open [lab2_gan_fashion_mnist.ipynb](lab2_gan_fashion_mnist.ipynb) in Colab (File → Open notebook → GitHub, then paste this repo's URL). Choose **Runtime → Change runtime type → T4 GPU** if you need a GPU, then run the cells in order. Download `lab_outputs/` before the session ends.

## Lab record

The observation tables, expected ranges and viva answers are in
[LAB_RECORD.md](LAB_RECORD.md). Fill the tables in from your own run.
