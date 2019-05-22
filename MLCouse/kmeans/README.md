# Programming Question (clustering with K-means) [45 points] #
(this project is a replica of a test project from Jianyang Zhang)

In class we discussed the K-means clustering algorithm. Your programming assignment this week is to implement the K-means algorithm on country survey data in the file [country.csv]:

First get the code template by running:

    download and unzip kmeans.zip
    cd kmeans

You will be modifying the file `kmean.py`. Q2.4.1 - Q2.4.3 are provided for you in `test.py`. To get the results for Q2.4.1 - 2.4.3, run:

    python test.py


Note: Please use Python 2.7 for the assignment and only modify the `kmean.py` file. Your modified `kmean.py` will be tested on the original `test.py`. Please make sure your code compiles and behaves as expected with the original `test.py` file. Report your results and plot for Q2.4.1 - Q2.4.3 and your answers for Q2.4.4-Q2.4.6 along with your written assignment (problem 1) as a PDF file.

### The Data ###
The data comes from a UN survey on people’s political priorities. We have aggregated the data across countries in the file [country.csv]. Each row lists the relative importance for each priority (between 0 and 1). 

You will cluster the data to find which countries are similar based on what the populations of those countries care about.

### The algorithm ###
Your algorithm should be implemented as follows:
1. Select k starting centers that are points from your data set. You should be able to select these centers randomly or have them given as a parameter.
2. Assign each data point to the cluster associated with the nearest of the k center points.
3. Re-calculate the centers as the mean vector of each cluster from (2).
4. Repeat steps (2) and (3) until convergence or iteration limit.

Define convergence as no change in label assignment from one step to another or you have iterated 20 times (whichever comes first). Please count your iterations as follows: after 20 iterations, you should have assigned the points 20 times.

###  Within group sum of squares ###
The goal of clustering can be thought of as minimizing the variation within groups and consequently maximizing the variation between groups. A good model has low sum of squares within each group.

We define sum of squares in the traditional way. Let C<sub>k</sub> be the kth cluster and let µ<sub>k</sub> be the empirical mean of the observations x<sub>i</sub> in cluster C<sub>k</sub>. Then the within group sum of squares for cluster C<sub>k</sub> is defined as:

(../readme_img/ss.PNG)

Please note that the term |x<sub>i</sub> − µ<sub>C<sub>k</sub></sub>| is the euclidean distance between x<sub>i</sub> and µ<sub>C<sub>k</sub></sub>, and therefore should be calculated as 

(../readme_img/dist.png)

, where d is the number of dimensions. Please note that that term is squared in SS(k).

If there are K clusters total then the “sum of within group sum of squares” is just the sum of all K of these individual SS(k) terms.

### Questions ###
1. [10pts] The values of sum of within group sum of squares for k = 5, k = 10 and k = 20. Please start your centers with the first k points in the dataset. So, if k = 5, your initial centroids will be the five countries: Afghanistan, Albania, ...
2. [5pts] The number of iterations that k-means ran for k = 5, starting the centers as in the previous item. Make sure you count the iterations correctly. If you start with iteration i = 0 and at i = 3 the cluster assignments don’t change, the number of iterations was 4, as you had to do step 2 four times to figure this out.
3. [15pts] A plot of the sum of within group sum of squares versus k for k = 1 − 50. Please start your centers randomly (choose k points from the dataset at random).
4. [5pts] Based on your plot, choose a ”best” value of k for this dataset. Explain why you chose this value.
5. [5pts] For your optimal value of k, examine the resulting clusters, and also how their clusters centers differ from the average over all countries. What general trends to you see in this data? For example, how well balanced are the clusters? Do the countries in each cluster appear to be related?
6. [5pts] Pick a country you are interested in. It could be the country you are from, somewhere you have visited, or a country you would like to learn more about. What cluster does this country belong to? What sets this cluster apart from other countries? Are the countries in this cluster related somehow (geographically, politically, economically)? Are there any unexpected countries in this cluster?

### Submitting ###
1. Submit your modified `kmean.py` file to the GauchoSpace.
2. Report your answers for Q2.4.1-Q2.4.6, and attach them to your homework (this is what you should turn in as a hardcopy in class).

