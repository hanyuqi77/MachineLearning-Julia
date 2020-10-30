# Assignment 3
#### Due: 11:59pm Thursday, April 16

This is the repository for assignment 2. It contains the following files:

* `.gitignore` - tells git to ignore certain kinds of files. This prevents you from submitting the auxiliary files created when producing LaTeX documents.
* `README.md` - the text you are currently reading.
* `A3.tex` LaTeX source for the writeup.
* `A3.pdf` assignment text compiled from tex. Please replace with your solutions. Feel free to use Weave.jl 
* `Project.toml` packages for the Julia environment.
* `vae.jl` starter code for assignment in Julia.
* `example_flux_model.jl` code which illustrates how to use Flux.jl
* `variational_autoencoder.py` starter code if you would like to use Python with autograd.
* `loadMNIST.py` utility functions of python loading data.

## Submitting with Github Classroom
If you are a registered student in the course, you will receive a link to accept this assignment which will `git clone` this repository through GitHub Classroom.

**If you are registered in the course and did not receive a Github Classroom assignment invitation, contact the instructors!**

You will submit your solutions for assessment by using the following `git` commands in commandline, with git integrated into your editor environment like [Atom](https://github.atom.io/), or with a dedicated [Github application](https://desktop.github.com/):

* `git status` - See what files have been changed, which have been `staged` (added and ready for a commit), or `unstaged` and will need to be added.
* `git add` - Add all the files you want assessed. Including your `.pdf` writeup and your source code (`.jl`, `.jmd`, or `.py`)
* `git commit` - Saves all the changes that were `staged` (by `git add`). 
* `git push` - Sends your changes off your machine. 

**Everything you want assessed must be committed and pushed before the assignment due date**

Please practice using git and pushing your solutions prior to the deadline. If you have any difficulties, ask on the course forum.

Keep your solutions to this assignment private and in accordance with our collaboration and academic integrity policy.
