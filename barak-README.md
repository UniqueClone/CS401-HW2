# Supervised Learning Assignment: The “Lovey Dovey” Data

## Goal

The objective of this assignment is to train a classifier, attempting to get good generalization performance. You can use whatever tools and methods you wish (within reason), but you're expected to do the work yourself.

## Materials

These are the files provided to you.

### Training Set
* File [`train-io.txt.gz`]( train-io.txt.gz )
* Uncompress using `gunzip` or `gzcat`
* Composed of 300000 (three hundred thousand) lines.
* Each line is a single input-output pair, consisting of 11 (eleven) numbers. The first 10 (ten) of these numbers are an input vector, the last number is the target output. It is a classification problem, with two classes: `0` and `1`.
* The samples are iid.

### Test Set
* File [`test-i.txt.gz`]( test-i.txt.gz )
* Composed of 10000 (ten thousand) lines.
* Each line is a single input, consisting of 10 (ten) numbers.

### Correct Answers
* File [`test-o.txt.asc`]( test-o.txt.asc )
* Consists of 10000 lines, each containing a single character, either a `0` or a `1`, showing the true labels for the test set.
* This is encrypted! I will reveal the key after the assignment is over.

## Turn In

The report and/or code can be in a git repo, like https://gitlab.cs.nuim.ie/YOUR-USERID/cs401-hw2-great-fun/ or such. If you use the dept gitlab you can have free private repos etc, but you're welcome to use whatever, and just include the URL in a README.txt file you turn in instead of the actual materials.

### Your labelling of the test set
* File `test-o.txt`
* The labellings you are guessing for the test set, one per line, each line a single character `0` or `1`.
* This *must* be included
* With the name above.
* Ideally, I'll be able to regenerate it (perhaps with differences due to to random factors like initial weights etc) by running the code.
* It *must* be in precisely the correct format.

  This means:
  ```
  $ file test-o.txt
  test-o.txt: ASCII text

  $ wc test-o.txt
   10000 10000 20000 test-o.txt

  $ sort < test-o.txt | uniq --count
        x 0
        y 1
  ```
  where *x* and *y* are the number of test samples you classified as `0` and `1` respectively, so *x*+*y*=10000.
* If it is not in the right format, my automated tools won't handle it.
* If my automated tools won't handle it, I'll have to deal with it manually.
* I don't want to deal with it manually.
* You don't want me to have to deal with it manually.
* Don't make me deal with it manually.

###  A Report

* A document for me to read.
* Describing what you did (keeping in mind that brevity is the soul of wit)
* Can be handwritten/scanned, e.g., the drawings or some graphs, if you want. It's not a journal paper, so relax! As long as I can read it.
* For **Extra Credit**: exhibit some relevant ROC curves, e.g., of the classifier you settled on, and others

### Your Code, which I can read and run

Ideally, I should be able to:

1. copy `train.txt` and `test.txt` into the directory
2. ensure I have any required libraries/systems/etc installed as described in your README file (e.g., JAX, R, octave, whatever, plus packages they may use)
2. run `make test-o.txt` or some other command described in your README file to regenerate `test-o.txt`; need not retrain classifier although it's okay if it does if that's not too slow.
3. run `make all` or some other etc to re-train the classifier from scratch (if appropriate) and then (re)generate `test-o.txt`

### Please please please **do not** include the data files I'm distributing

- or trivially transformed versions thereof; instead, if necessary, include the transformation in your scripts for running the code as described above
- please don't include other “bloat” or derived file: things that can be generated by running `make`, that can be easily downloaded from the Internet (just include a URL, or put a `curl URL` stanza in your `Makefile`), etc. (Aside from `test-o.txt` which needs to be included.)

### Optional extra credit
- a file `code.txt` explaining the meaning of the code name “lovey dovey” for this assignment and how you figured it out
- please don't just make a random guess

## Scoring

The generalization performance of your classifier will be measured against the true classes of the test cases.

## Deadline

Due 17:00 5-Dec-2020 (automatic extensions upon request)

* If you turn in something, but put a note on the *first line* of the `README` file saying: “this is what I have now but I hope to update it by *date*”, I'll put yours at the bottom of the pile and replace it with your update when you turn that in. If the `README` in your update says what you changed (aside from `test-o.txt` of course), that might be helpful, especially if you're particularly far past the deadline. **Please don't abuse this: grading takes time.**