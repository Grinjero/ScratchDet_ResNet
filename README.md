# Modularized SSD implementation in TensorFlow

This repo tries to implement SSD in a modularized fashion.

Inspiration: Speed/accuracy trade-offs for modern convolutional object detectors. (arXiv:1611.10012)


## TODOs:
- [x] Rewrite core Tensorflow SSD for modularization
- [x] Connect original VGG backend for SSD
- [ ] Train VGG-SSD
- [ ] Build test routine
- [ ] Test VGG-SSD
- [ ] Connect other backends for SSD
- [ ] Test connected backends


## Acknowledgement

This repo is based on the work of
* @balancap [SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow/)
* @LevinJ [SSD_tensorflow_VOC](https://github.com/LevinJ/SSD_tensorflow_VOC)
