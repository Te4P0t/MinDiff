# MinDiff

A collection/playground for some minimal implementation of some diffusion models.

Try to use as less code as possible, with as simple arch as possible.

## Mnist

In this project we demo a minimal implementation of alpha-blended diffusion model which can train MNIST diffusion model with only 2 linear layer and an embedding layer for class cond.

Example result:

| lr5e-5                                         | lr2e-3                                         | lr5e-3                                         |
| ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| ![1731283536610](image/README/1731283536610.png) | ![1731283719848](image/README/1731283719848.png) | ![1731283526574](image/README/1731283526574.png) |

## Cifar

In this project we demo a minimal implementation of LDM + alpha blended diffusion model which can obtain an F4-C8 AE latent encoder/decoder with only 0.5M parameter. And a Conv+AdaLN Denoiser with only 10M params.

### AE image tokenizer/latent encoder

The arch is simple:

* Encoder: [Conv, Conv, Down]*n + [Conv, Conv]
* Decoder: [Conv, Conv, Up]*n + [Conv, Conv]

Image examples:

|                       | Input                                          | Recon                                          |
| --------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 0.5M param 1000 Epoch | ![1731282135915](image/README/1731282135915.png) | ![1731282189195](image/README/1731282189195.png) |

### Conv+AdaLN DM

For each ConvAdaNorm block we have: [GN, Conv, AdaLN Modulation, Mish, GN, Conv, residual]

The model directly use a sequence of ConvAdaNorm block instead of UNet-like arch.

Since the resolution of latent pixel is low (8x8 for Cifar10, or 16x16 when we train 64x64 images). We didn't introduce UNet-like arch which is too complicated.

Image Examples:

| 100epoch                                       | 500epoch                                       | 1000epoch                                      | 5000epoch                                      |
| ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| ![1731282575417](image/README/1731282575417.png) | ![1731282588239](image/README/1731282588239.png) | ![1731282595499](image/README/1731282595499.png) | ![1731283382791](image/README/1731283382791.png) |
