<h1 align="center"> MALIO </h1> <br>
<p align="center">
  <a href="https://doi.org/10.1039/D2CP03696G">
    <img alt="MALIO" title="MALIO" src="https://github.com/user-attachments/assets/8cc451c4-1bf0-4cc9-992e-7e065c561f05" width="300">
  </a>
</p>

<p align="center">
  Molecular Assembly structure Learning package for Identifying Order parameters (MALIO)
</p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Features](#features)
- [Citing MALIO](#citation)
- [System Requirements](#system-requirements)
- [Build Process](#build-process)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Features

The main features of MALIO are described in detail in the following papers, for example:

* [Kazuaki Z. Takahashi, Molecular cluster analysis using local order parameters selected by machine learning, Physical Chemistry Chemical Physics 25, 658-672, 2023. (First paper to publish MALIO.)](https://doi.org/10.1039/D2CP03696G)
* [Kazuaki Z. Takahashi, Takeshi Aoyagi, and Jun-ichi Fukuda, Multistep nucleation of anisotropic molecules, Nature Communications 12, 5278, 2021. (First paper using the whole concept of MALIO.)](https://doi.org/10.1038/s41467-021-25586-4)
* [Kazuaki Z. Takahashi, and Masaki Hiratsuka, Local Order Parameters Classifying Water Networks of Ice and Cyclopropane Clathrate Hydrates, Crystal Growth & Design 23, 4815-4824, 2023.](https://doi.org/10.1021/acs.cgd.2c01519)
* [Kazuaki Z. Takahashi, Numerical evidence for the existence of three different stable liquid water structures as indicated by local order parameter, The Journal of Chemical Physics 161, 134507, 2024.](https://doi.org/10.1063/5.0205804)

## Citing MALIO

When you publish findings (data such as figures and tables, as well as texts) obtained with the help of MALIO, please cite the following two references:

* [Kazuaki Z. Takahashi, Molecular cluster analysis using local order parameters selected by machine learning, Physical Chemistry Chemical Physics 25, 658-672, 2023. (First paper to publish MALIO.)](https://doi.org/10.1039/D2CP03696G)
* [Kazuaki Z. Takahashi, Takeshi Aoyagi, and Jun-ichi Fukuda, Multistep nucleation of anisotropic molecules, Nature Communications 12, 5278, 2021. (First paper using the whole concept of MALIO.)](https://doi.org/10.1038/s41467-021-25586-4)

## System Requirements

Software dependencies (version numbers tested on):

* Python (3.6.13)
* nose (1.3.7)
* numpy (1.16.4)
* pyquaternion (0.9.5)
* sympy (1.4)
* scipy (1.5.4)
* matplotlib (3.1.0)
* pandas (1.1.5)
* scikit-learn (0.23.2)
* cython (0.29.21)

Operating systems:
* CentOS Linux 7
* Rocky Linux 8
* Rocky Linux 9

## Build Process

Execute the following commands:
* $ cd malio_cy
* $ ls Makefile setup.py
* $ make

## Acknowledgments

MALIO was developed with the support of JST PRESTO (Grant No. JPMJPR22O6), and NEDO (JPNP18016).

## Contact

Kazuaki Z. Takahashi (kazu.takahashi@aist.go.jp)
