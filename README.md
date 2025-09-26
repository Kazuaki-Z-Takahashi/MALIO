<h1 align="center"> MALIO </h1> <br>
<p align="center">
  <a href="https://doi.org/10.1039/D2CP03696G">
    <img alt="MALIO" title="MALIO" src="https://github.com/user-attachments/assets/8cc451c4-1bf0-4cc9-992e-7e065c561f05" width="300">
  </a>
</p>

<p align="center">
  Molecular Assembly Learning for Identifying Order parameters (MALIO)
</p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Features](#features)
- [Citing MALIO](#citing-malio)
- [System Requirements](#system-requirements)
- [Build Process](#build-process)
- [Quick Start Guide](#quick-start-guide)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Features

⚠️ **Note:** This project has migrated from **Cython** to **C++**!

The main features of MALIO are described in detail in the following papers, for example:

* Kazuaki Z. Takahashi, *Molecular cluster analysis using local order parameters selected by machine learning*, **Physical Chemistry Chemical Physics** 25, 658–672, 2023. (https://doi.org/10.1039/D2CP03696G) **First paper introducing MALIO.**
* Kazuaki Z. Takahashi, Takeshi Aoyagi, and Jun-ichi Fukuda, *Multistep nucleation of anisotropic molecules*, **Nature Communications** 12, 5278, 2021. (https://doi.org/10.1038/s41467-021-25586-4) **First paper presenting the overall concept of MALIO.**
* Kazuaki Z. Takahashi, *Numerical evidence for the existence of three different stable liquid water structures as indicated by local order parameter*, **The Journal of Chemical Physics** 161, 134507, 2024. (https://doi.org/10.1063/5.0205804)
* Jun-ichi Fukuda and Kazuaki Z. Takahashi, *Direct simulation and machine learning structure identification unravel soft martensitic transformation and twinning dynamics*, **Proceedings of the National Academy of Sciences of the United States of America** 121, e2412476121, 2024. (https://doi.org/10.1073/pnas.2412476121) **First study applying MALIO to continuum simulations**

## Citing MALIO

If you publish results (including data such as figures, tables, or text) obtained using MALIO, please cite the following two references:

* Kazuaki Z. Takahashi, *Molecular cluster analysis using local order parameters selected by machine learning*, **Physical Chemistry Chemical Physics** 25, 658–672, 2023. (https://doi.org/10.1039/D2CP03696G) **First paper introducing MALIO.**
* Kazuaki Z. Takahashi, Takeshi Aoyagi, and Jun-ichi Fukuda, *Multistep nucleation of anisotropic molecules*, **Nature Communications** 12, 5278, 2021. (https://doi.org/10.1038/s41467-021-25586-4) **First paper presenting the overall concept of MALIO.**

## System Requirements

Software dependencies (tested versions):

* GNU Compiler Collection (11.5)
* MPICH (4.1.1)
* Python (3.6.13, 3.7.9)
* nose (1.3.7)
* numpy (1.16.4, 1.16.5)
* pyquaternion (0.9.5)
* sympy (1.4)
* scipy (1.5.4, 1.7.3)
* matplotlib (3.1.0, 3.4.0)
* pandas (1.1.5, 1.2.0)
* scikit-learn (0.23.2)

Operating systems tested:
* CentOS Linux 7
* Rocky Linux 8
* Rocky Linux 9

## Build Process

Run the following commands:
* $ tar xvzf voro++-0.4.6.tar.gz
* $ cd voro++-0.4.6/
* $ make
* $ cd ../malio_cpp/
* $ make
* $ cd ..

## Quick Start Guide

Run the following commands:

* $ bindir="$(pwd)/malio_cpp"
* $ cd quick_start/
* $ mpiexec -n [number_of_CPU_cores] $bindir/ml_lmp.x -os op_settings.json -fn 2 -n 3
* $ python3 $bindir/ml_learn.py

You can reduce the number of order parameters and shorten the computation time by editing op_settings.json.
* The option -fn specifies the number of files per structural motif to be input into MALIO.
* The option -n specifies the number of structural motifs to distinguish.

By following the Quick Start Guide, you can reproduce part of the results reported in [this paper](https://doi.org/10.1063/5.0205804), which is listed in the Features section.

## Acknowledgments

MALIO was developed with support from JST PRESTO (Grant No. JPMJPR22O6) and NEDO (JPNP18016).

## Contact

Kazuaki Z. Takahashi (kazu.takahashi@aist.go.jp)
