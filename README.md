# Learning of natural selection using machine learning

## Introduction

This is a joint project with [Me](github.com/Grenlex), [Mikhail Shishking]() and [Tigran Apresyan](). Also, let me say in advance that file called ml.py are stored here just as the first example of our work. It was written using **numpy** and now this file doesn't affect our main program.

## Roles

* [Alexey Shmelev](github.com/Grenlex) - code, algorithms in terms of Programing and Bioinformatics
* [Mikhail Shishking]() - algorithms in termth of Math and Bioinformatics
* [Tigran Apresyan]() - configuring of NNs

## About

Here are given three folders: *rnn*, *selam* and *tasks*. Let's define what each of them is dedicated for:

* rnn - this folder store the major part of our work, thus you can finde code of the NN and the important file with settings there
* selam - store the bash script which allow you to quickly generate data to train NN using SELAM generator
* tasks - mainly store .sbatch files which are used to run programs using SLURM

## Progress

- [ ] Create bidirectional RNN in order to find part of genome which is under natural selection
    - [X] Write working code itself
    - [ ] Opimize it for better performance in training results
- [ ] Create feedforward neural network using regression for finding generation where data was taken from
    - [X] Write working code itself
    - [ ]  Opimize it for better performance in training results
    - [ ] Find optimal mathematical model
- [X] Make program to work parallel
