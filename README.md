Download Link: https://assignmentchef.com/product/solved-cs4342-project-1-smile-classifier
<br>
Homework 1

This homework is intended to help you (1) learn (or refresh your understanding of) how to implement linear algebraic operations in Python using numpy (to which we refer in the code below as np); (2) practice implementing one of the simpler machine learning algorithms: step-wise classification.

1           Part 1: Python and numpy

For each of the problems below, write a method (e.g., problem1) that returns the answer for the corresponding problem. Put all your methods in one file called homework1 WPIUSERNAME.py

(e.g., homework1 jrwhitehill.py, or homework1 jrwhitehill lleshin.py for a team). See the starter file homework1 template.py. In all problems, you may assume that the the dimensions of the matrices and/or vectors are compatible for the requested mathematical operations.

<strong>Note 1</strong>: In mathematical notation we usually start indices with <em>j </em>= 1. However, in numpy (and many other programming settings), it is more natural to use 0-based array indexing. When answering the questions below, do not worry about “translating” from 1-based to 0-based indexes. For example, if the (<em>i,j</em>)th element of some matrix is requested, (where <em>i,j </em>≥), you can simply write A[i,j].

<strong>Note 2</strong>: To represent and manipulate vectors and matrices, please use numpy’s array class (<em>not </em>the matrix class).

<strong>Note 3</strong>: While the difference between a row vector and a column vector is important when doing math, numpy does not care about this difference. Hence, please use a 1-dimensional numpy array for both cases.

<ol>

 <li>Given matrices <strong>A </strong>and <strong>B</strong>, compute and return an expression for <strong>A </strong>+ <strong>B</strong>. <strong>[ 2 pts ]</strong></li>

</ol>

<em>Answer </em>(freebie!): While it is completely valid to use np.add(A, B), this is unnecessarily verbose; you really should make use of the “syntactic sugar” provided by Python’s/numpy’s operator overloading and just write: A + B. Similarly, you should use the more compact (and arguably more elegant) notation for the rest of the questions as well.

<ol start="2">

 <li>Given matrices <strong>A</strong>, <strong>B</strong>, and <strong>C</strong>, compute and return <strong>AB </strong>− <strong>C </strong>(i.e., right-multiply matrix <strong>A </strong>by matrix <strong>B</strong>, and then subtract <strong>C</strong>). Use dot or dot. <strong>[ 2 pts ]</strong></li>

 <li>Given matrices <strong>A</strong>, <strong>B</strong>, and <strong>C</strong>, return <strong>A</strong>, where represents the element-wise (Hadamard) product and &gt; represents matrix transpose. In numpy, the element-wise product is obtained simply with *. <strong>[ 2 pts ]</strong></li>

 <li>Given column vectors <strong>x </strong>and <strong>y</strong>, compute the inner product of <strong>x </strong>and <strong>y </strong>(i.e., <strong>x</strong><sup>&gt;</sup><strong>y</strong>). <strong>[ 2 pts ]</strong></li>

 <li>Given matrix <strong>A</strong>, return a matrix with the same dimensions as <strong>A </strong>but that contains all zeros. Use zeros. <strong>[ 2 pts ]</strong></li>

 <li>Given matrix <strong>A</strong>, return a vector with the same number of rows as <strong>A </strong>but that contains all ones. Use ones. <strong>[ 2 pts ]</strong></li>

 <li>Given square matrix <strong>A </strong>and (scalar) <em>α</em>, compute <strong>A </strong>+ <em>α</em><strong>I</strong>, where <strong>I </strong>is the identity matrix with the same dimensions as <strong>A</strong>. Use eye. <strong>[ 2 pts ]</strong></li>

 <li>Given matrix <strong>A </strong>and integers <em>i,j</em>, return the <em>j</em>th column of the <em>i</em>th row of <strong>A</strong>, i.e., <strong>A</strong><em><sub>ij</sub></em>. <strong>[ 2 pts ]</strong></li>

 <li>Given matrix <strong>A </strong>and integer <em>i</em>, return the sum of all the entries in the <em>i</em>th row, i.e., <sup>P</sup><em><sub>j </sub></em><strong>A</strong><em><sub>ij</sub></em>. Do <strong>not </strong>use a loop, which in Python is very slow. Instead use the sum function. <strong>[ 4 pts ]</strong></li>

 <li>Given matrix <strong>A </strong>and scalars <em>c,d</em>, compute the arithmetic mean over all entries of <em>A </em>that are between <em>c </em>and <em>d </em>(inclusive). In other words, if S = {(<em>i,j</em>) : <em>c </em>≤ <strong>A</strong><em><sub>ij </sub></em>≤ <em>d</em>}, then compute <sub>|S|</sub><u><sup>1 </sup></u><sup>P</sup><sub>(<em>i,j</em>)∈S </sub><strong>A</strong><em><sub>ij</sub></em>. Use nonzero along with np.mean. <strong>[ 5 pts ]</strong></li>

 <li>Given an (<em>n </em>× <em>n</em>) matrix <strong>A </strong>and integer <em>k</em>, return an (<em>n </em>× <em>k</em>) matrix containing the right-eigenvectors of <strong>A </strong>corresponding to the <em>k </em>largest eigenvalues of <strong>A</strong>. Use linalg.eig to compute eigenvectors. <strong>[ 5 pts ]</strong></li>

 <li>Given square matrix <strong>A </strong>and column vector <strong>x</strong>, use linalg.solve to compute <strong>A</strong><sup>−1</sup><strong>x</strong>. Do <strong>not </strong>use np.linalg.inv or ** -1 to compute the inverse explicitly; this is numerically unstable and can, in some situations, give incorrect results. <strong>[ 5 pts ]</strong></li>

 <li>Given square matrix <strong>A </strong>and row vector <strong>x</strong>, use linalg.solve to compute <strong>xA</strong><sup>−1</sup>. Hint: <strong>AB </strong>=</li>

</ol>

(<strong>B</strong><sup>&gt;</sup><strong>A</strong><sup>&gt;</sup>)<sup>&gt;</sup>. Do <strong>not </strong>use np.linalg.inv or ** -1 to compute the inverse explicitly. <strong>[ 5 pts ]</strong>

2           Part 2: Step-wise Classification

For the tasks below, write your code in a file called homework1 smile WPIUSERNAME.py, and put your experimental results in homework1 smile WPIUSERNAME.pdf.

In this part of the assignment you will train a very simple smile classifier that analyzes a grayscale image <strong>x </strong>∈ R<sup>24×24 </sup>and outputs a prediction ˆ<em>y </em>∈ {0<em>,</em>1} indicating whether the image is smiling (1) or not (0). The classifier will make its decision based on very simple <strong>features </strong>of the input image consisting of <em>binary comparisons </em>between pixel values. Each feature is computed as

I[<strong>x</strong><em>r</em><sub>1</sub><em>,c</em><sub>1 </sub><em>&gt; </em><strong>x</strong><em>r</em><sub>2</sub><em>,c</em><sub>2</sub>]

where <em>r<sub>i</sub>,c<sub>i </sub></em>∈ {0<em>,</em>1<em>,</em>2<em>,…,</em>23} are the row and column indices, respectively, and I[·] is an indicator function whose value is 1 if the condition is true and 0 otherwise. In general, these features are not very good, but nonetheless they will enable the classifier to achieve an accuracy (<em>f</em><sub>PC</sub>) much better than just guessing or just choosing the dominant class. Based on these features, you should train an <em>ensemble </em>smile classifier using <strong>step-wise classification </strong>for <em>m </em>= 5 features. The output of the ensemble (1 if it thinks the image is smiling, 0 otherwise) is determined by the <em>average </em>prediction across all <em>m </em>members of the ensemble. If more than half of the <em>m </em>ensemble predictors <em>g</em><sup>(1)</sup><em>,…,g</em><sup>(<em>m</em>) </sup>think that the image is smiling, then the ensemble says it’s a smile; otherwise, the ensemble says it’s not smiling.

Step-wise classification/regression is a <strong>greedy algorithm</strong>: at each round <em>j</em>, choose the <em>j</em>th feature (<em>r</em><sub>1</sub><em>,c</em><sub>1</sub><em>,r</em><sub>2</sub><em>,c</em><sub>2</sub>) such that – when it is added to the set of <em>j </em>− 1 features that have <em>already been selected </em>– the accuracy (<em>f</em><sub>PC</sub>) of the overall classifier on the training set is maximized. More specifically, at every round <em>j</em>, consider <em>every possible </em>tuple of pixel locations (<em>r</em><sub>1</sub><em>,c</em><sub>1</sub><em>,r</em><sub>2</sub><em>,c</em><sub>2</sub>): if you construct an ensemble with <em>j </em>predictors (the <em>j </em>−1 you’ve already chosen, plus the current “candidate” (<em>r</em><sub>1</sub><em>,c</em><sub>1</sub><em>,r</em><sub>2</sub><em>,c</em><sub>2</sub>)), is the resulting ensemble more accurate (in terms of PC on training data) than <em>any other </em>tuple of pixel locations during this round? If the current tuple is the best yet (for round <em>j</em>), then save it as your “best seen yet” for round <em>j</em>. If not, ignore it. Move on to the next possible tuple of pixel locations, and repeat until you’ve searched all of them. At the end of round <em>j</em>, you will have selected the best feature for that round, and you add it to your set of selected features. Once added, it stays in the set forever – it can never be removed. (Otherwise, it wouldn’t be a greedy algorithm at all.) Now you move on to round <em>j </em>+ 1 until you’ve completed <em>m </em>= 5 rounds.

To measure the ensemble’s accuracy (<em>f</em><sub>PC</sub>), you should run it on <em>all </em>the images in the <em>training </em>set, and then compare the output of the ensemble to the corresponding ground-truth labels. At the end of the entire training procedure, you should estimate how well your “machine” (ensemble smile classifier) works on a set of images not used for training, i.e., the <em>test set</em>.

<strong>Skeleton code</strong>: While how you write your code is up to you (subject to the vectorization constraint and also basic readability); however, to get you started, we sketched in a few functions:

<ul>

 <li>fPC (y, yhat): this takes in a vector of ground-truth labels and corresponding vector of guesses, and then computes the accuracy (PC). The implementation (in vectorized form) should only take 1-line.</li>

 <li>measureAccuracyOfPredictors (predictors, X, y): this takes in a <em>set </em>of predictors, a set of images to run it on, as well as the ground-truth labels of that set. For each image in the image set, it runs the ensemble to obtain a prediction. Then, it computes and returns the accuracy (PC) of the predictions w.r.t. the ground-truth labels.</li>

 <li>stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels): I’ve included some visualization code, but otherwise it’s empty. You need to implement the step-wise classification described above.</li>

</ul>

Tasks:

<ol>

 <li>Download from Canvas the starter Python file homework1 smile.py as well as the following data files: npy,trainingLabels.npy,testingFaces.npy,testingLabels.npy.</li>

 <li>Write code to train a step-wise classifier for <em>m </em>= 5 features of the binary comparison type described above; the greedy procedure should maximize <em>f</em><sub>PC </sub>(which you will also need to implement). At each round, the code should examine <em>every possible feature </em>(<em>r</em><sub>1</sub><em>,c</em><sub>1</sub><em>,r</em><sub>2</sub><em>,c</em><sub>2</sub>)<em>.</em>. Make sure your code is <strong>vectorized </strong>to improve run-time performance (wall-clock time) <strong>[ 25 pts ]</strong>.</li>

 <li>Write code to analyze how training/testing accuracy changes as a function of number of examples<em>n </em>∈ {400<em>,</em>800<em>,</em>1200<em>,</em>1600<em>,</em>2000} (implement this in a for-loop)

  <ul>

   <li>Run your step-wise classifier training code only on the first <em>n </em>examples of the <em>training set</em>.</li>

   <li>Measure and record the <em>training accuracy </em>of the trained classifier on the <em>n </em></li>

   <li>Measure and record the <em>testing accuracy </em>of the classifier on the (entire) <em>test set</em>.</li>

  </ul></li>

</ol>

<strong>Important</strong>: you <strong>must </strong>write code (a simple loop) to do this – do <strong>not </strong>just do it manually for each value of <em>n</em>. This is good experimental practice in general and is especially important in machine learning to ensure reproducibility of results. Using the entire training set, you should achieve a test accuracy of at least 75%. <strong>[ 8 pts ]</strong>.

<ol start="4">

 <li>In a PDF document (you can use whatever tool you like – LaTeX, Word, Google Docs, etc. – but makesure you export to PDF), show the training accuracy and testing accuracy as a function of <em>n</em>. Please use the following simple format:</li>

</ol>

n trainingAccuracy testingAccuracy 400 … …

800 … …

1200 … …

1600 … …

2000 … …

Moreover, characterize in words how the training accuracy and testing accuracy changes as a function of <em>n</em>, <em>and </em>how the two curves relate to each other. What trends do you observe? <strong>[4 pts]</strong>

<ol start="5">

 <li><strong>Visualizing the learned features</strong>: It is very important in empirical machine learning work to visualize what was actually learned during training. This can be useful for debugging to make sure that your training code is working as it should, and also to make sure your training and testing sets are selected wisely. For <em>n </em>= 2000, visualize the <em>m </em>= 5 features that were learned by (a) displaying any face image from the test set; and (b) drawing a square around the specific pixel locations ((<em>r</em><sub>1</sub><em>,c</em><sub>1</sub>) and (<em>r</em><sub>2</sub><em>,c</em><sub>2</sub>)) that are examined by the feature. You can use the example code in the homework1 smile.py template to render the image. Insert the graphic (just one showing all 5 features) into your PDF file. <strong>[ 3 pts]</strong>.</li>

</ol>

Here’s an example that shows just one feature:

<strong>Tip on vectorization</strong>: Implement your training algorithm so that, for any particular feature (<em>r</em><sub>1</sub><em>,c</em><sub>1</sub><em>,r</em><sub>2</sub><em>,c</em><sub>2</sub>), <em>all </em>the feature values (over all the <em>n </em>training images) are extracted at once using numpy – do not use a loop. Even after vectorizing my own code, running the experiments required in this assignment took about 1 hour (on a single CPU of a MacBook 2016 laptop).