# Lab Record: Assignment 2, ETMMGA304 Generative AI and Its Applications

**Student:** ____________________  **Roll no.:** ____________  **Programme:** MCA

Fill in each blank table from **your own run**. The *expected ranges* below
are approximate values for the default settings, to help you spot a broken
run. They are not results to copy. Values change with hardware, library
versions and random seeds.

Write down the following for every experiment:

| Field | Value |
|-------|-------|
| Date | |
| Device (CPU / GPU model) | |
| Python / PyTorch / Transformers versions | |
| Seed | 42 |

---

## Lab 2: Image Synthesis Using GANs

**Aim.** Train a basic GAN on Fashion-MNIST. Watch how outputs for fixed
noise inputs change across epochs, and judge training from both the images
and the losses.

**Software / data.** PyTorch, torchvision Fashion-MNIST (60k train), pixels
normalised to [−1, 1]. Script: `lab2_gan_fashion_mnist.py`.

**Parameters.** G: 64→256→512→784 (LeakyReLU 0.2, Tanh output). D:
784→512→256→1 logit. BCEWithLogitsLoss. Adam, lr = 2e-4, β = (0.5, 0.999),
batch size 128, 10 epochs, seed 42, 25 fixed noise vectors.

### Observation table

| Epoch | D loss | G loss | Recognisable out of 25 | Repeated shapes or artefacts |
|-------|--------|--------|------------------------|------------------------------|
| 1 | | | | |
| 5 | | | | |
| 10 | | | | |

**Expected ranges.** D loss is the mean of its real and fake BCE, so it
usually settles between about 0.45 and 0.65. G loss is about 0.9–2.0 and is
often highest in the first epoch. Neither loss should go to 0 or keep
climbing. At epoch 1 the samples are noisy blobs with rough silhouettes
(0–8 of 25 recognisable). By epoch 10, trousers, shirts, sneakers and bags
are clearly visible (15–22 of 25), with some speckle noise and soft edges.

### Result

Over training, the same grid positions sharpen from blobs into clear
garment silhouettes. Flat losses that do not diverge, together with varied
classes across the grid, show that training is stable. GAN losses do not
fall steadily the way supervised losses do, because each network's loss
depends on the other network's progress.

**Limitation.** MLP networks have no spatial inductive bias, so textures
stay noisy. Partial mode collapse can appear as one silhouette repeated in
many grid positions. **Next step.** Use a DCGAN (convolutional) architecture,
train for longer, and measure FID instead of judging by eye.

### Viva answers

1. **What does each network learn during training?**
   The discriminator learns a classifier that outputs a high logit for real
   Fashion-MNIST images and a low logit for generated ones. The generator
   learns to map noise z to images that the discriminator scores as real,
   which pushes its output distribution towards the data distribution. The
   two networks play a minimax game.

2. **Why is the generated batch detached in the discriminator step?**
   `fake.detach()` cuts the computation graph, so `d_loss.backward()`
   computes gradients for D only. Without the detach, gradients would also
   flow into G, wasting compute, and could move G in the direction that helps
   D (the opposite of what G should learn). The generator step then reuses
   the non-detached `fake`, with D frozen.

3. **Why are the grid inputs kept fixed?**
   With `fixed_z` reused at every checkpoint, each grid cell shows the same
   latent point at a different stage of training. Differences between grids
   then come only from changes to the generator, not from new random noise,
   so progress and collapse are easy to compare.

4. **How can a grid reveal mode collapse?**
   With 25 different noise vectors, a healthy generator produces many
   garment classes. If many cells show nearly the same silhouette, or the
   variety shrinks from epoch 5 to epoch 10, the generator has collapsed onto
   a few modes that fool D. The losses alone can look normal while this
   happens.
