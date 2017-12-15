# Question-Answering-NLP

Authors: Chris [github.com/chase130101] and Adarsh [github.com/jaks19]

https://www.sharelatex.com/6418673345zkgykpthzrhj

NOTE:

The hierarchy is as follows:

Code
Theory-Dump
etc

In this same folder, on your local copy, add the folder Data1 
This folder should have all the data files, extracted, so that the data processing methods work on them

I have removed them from the repo as push time was too long

REMEMBER TO MAKE A COPY OF ALL DATA FILES ON YOUR MACHINE BEFORE PULLING AS THE ONES IN YOUR WORKING FOLDER WILL GET REMOVED!


To save and load optimizer states:
https://discuss.pytorch.org/t/saving-and-loading-sgd-optimizer/2536/5
A quick workaround that fixed it for me was doing optimizer.state = defaultdict(dict, optimizer.state) after doing optimizer.load_state_dict(torch.load('optim.pth')).
