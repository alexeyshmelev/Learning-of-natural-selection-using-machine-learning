# Learning of natural selection using machine learning

## Introduction

This is a project of [Me](github.com/alexeyshmelev). Also, let me say in advance that most of files in this repository are stored here just as the first example of our work. It was written using **numpy** and **pytorch** and this file doesn't affect our main program.

## Roles

* [Alexey Shmelev](github.com/Grenlex) - code, algorithms in terms of Programing and Bioinformatics
## About

Here are given three folders: *rnn*, *selam* and *tasks*. Let's define what each of them is dedicated for:

* rnn - this folder store the major part of our work, thus you can find code of the NN (and their saved states) and the important file with settings there
* selam - store the bash script which allow you to quickly generate data to train NN using SELAM generator
* tasks - mainly store .sbatch files which are used to run programs using SLURM

## Progress

- [X] Create bidirectional RNN for finding part of genome which is under natural selection
    - [X] Write working code itself
    - [X] Opimize it for better performance in training results
- [X] Create Transformer (regression task) for finding generation where data was taken from, foce of natural selection and percentage of admixture
    - [X] Write working code itself
    - [X]  Opimize it for better performance in training results
    - [X] Find optimal mathematical model
- [X] Create Transformer (binary classification) for finding if the natural selection was
- [X] Make program to work parallel
- [X] Create GUI using PyQt

## Compilation

You can compile my programm by your own using  *Pyinstaller* with the following command (file app.spec you can find in this repository)

```
pyinstaller --noconsole --onefile app.spec
```

Training results of Transformer (existence of natural selection):

<img src="https://github.com/alexeyshmelev/Learning-of-natural-selection-using-machine-learning/blob/1337dfba9f4d3986b087aae5d3ec980be8dbcc72/tof_loss.png">

<p style="position: relative; top: -50px;">
    Hello
</p>
