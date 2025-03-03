# Project
This repository contains the code for developing a cross-validation method aiming to validate KDE or Kernel Density Estimation as an alternative sampling technique in the context of imbalanced data.
several machine-learning algorithms  are used to assess the efficacy of the technique using SMOTE as a gold standard for comparison. 
The method runs on several binary and multiclass datasets from the [CUMIDA repository] (https://sbcb.inf.ufrgs.br/cumida#datasets) by considering a discrete imbalance ratio between the classes. The set of selected datasets for the experiment can be downloaded from the following [link] (https://drive.google.com/drive/folders/1zB5xFM9qrurKZjgKxmVpv4QKcaXUCSJY?usp=drive_link)


## Kernel Density Estimation
Kernel Density Estimation, or KDE, is a non-parametric method for estimating the probability density function of a set of random variables. Unlike parametric methods that estimate parameters by maximizing the Maximum Likelihood of obtaining the current sample, KDE estimates the density distribution directly from the data. 
<p align="center">
   $$f(x) = \frac{1}{n}\sum_{i=1}^{n}K_h(x-x_i)$$
</p>

Where, $\mathit{K}$  is the kernel function, $\mathit{h}$  is the bandwidth parameter, and $\mathit{n}$  is the number of observations. Intuitively, the true value of $\mathit{f(x)}$ is estimated as the average distance from $\mathit{x}$  to the sample data points $x_i$.  Given a continuous random variable, KDE produces a curve, which is an estimate of the underlying distribution of this data.

<p align="center">
  <img src=https://github.com/user-attachments/assets/77c6285b-f25d-4ae9-97a4-e795ce9995d5\>
</p>


Moreover, the kernel estimator depends on two parameters, i.e. the kernel function $\mathit{K}$  and the bandwidth $\mathit{h}$.
The first refers to how the data points are weighted depending on the type of kernel function. There are plenty of available kernel functions: epanechikov, biweight, triangular, gaussian, and rectangular kernels

<p align="center">
  <img src=https://github.com/user-attachments/assets/7a29f9bf-2a3f-49af-a738-f3d34c6f833f\>
</p>

## Approach
Consider a given training data set $S$ with $m$ examples (i.e., $|S| = m$), we define $S = \{(x_i,y_i\}, i = 1,...,m$, where $x_i \in X$ is an instance in the n-dimensional feature space $X = \{ f_1;f_2; ... ;f_n \}$, and $y_i \in Y = \{1,...,C\}$ is a class identity label associated with instance $x_i$. In particular, $C = 2$ represents the two-class classification problem. Furthermore, we define subsets $S_{min} \in S$ and $S_{maj} \in S$, where $S_{min}$ is the set of minority class examples in $S$, and $S_{maj}$ is the set of the majority class examples in $S$ ,so that $S_{min} \cap S_{maj}= \emptyset$ and $S_{min} \cup S_{maj}= {S}$. Lastly, any sets generated from sampling procedures on $S_{min}$ are labeled as $E$ representing the newly generated minority examples. 
Given a Kernel Density Estimation or KDE model, with a 'Gaussian' kernel function $K$ and an optimal bandwidth parameter, $h$ selected via the 'Silverman' rule of thumb. The object KDE builds an estimated probability density function $f$ over $S_{min}$. 
Moreover, KDE generates a new set of examples $E$, in quantity specified by $E = ( S_{maj} - S_{min})$ by randomly sampling from $f(S_{min})$. In this way, the number of total examples in $S_{min}$ is increased by $E$ and the class distribution balance of $S$ is adjusted accordingly. The same approach is applied to multi-class data in the same manner by oversampling each minority class by a quantity E, to adjust all classes in $|S|$ to $S_{maj}$
The implementation of SMOTE and KDE are taken respectively from imblearn and scikit-learn Python libraries with their default settings. In particular, for KDE, we
used the multivariate Gaussian KDE with its default bandwidth value determined by Silverman’s Rule of thumb (see eq  (\ref{Kernel}) - (\ref{h_silverman})) 
<p align="center">
   <img src=https://github.com/user-attachments/assets/785758b1-faee-4724-890f-8cabb23c935d\>

</p>
   
## Validation
A stratified 10-fold cross-validation schema compares a basic classifier of choice on KDE-based augmented data and SMOTE-based oversampling, along with a default not-oversampled dataset. 
In a 10-fold cross-validation, a given training set is randomly split into ten sets that are as close as equal as possible. However, in the context of class imbalance, a stratified procedure ensures that class proportions are maintained across all ten sets. Then, a model is trained on nine of these sets and validated on the tenth. This procedure is repeated for all sets until ten different performances are obtained. 
In this framework, the oversampling procedures are integrated within each training partition, thus creating for a given dataset, three parallel training sets during each cross-validation iteration.

<p align="center">
  <img src=https://github.com/user-attachments/assets/17ef6e60-56bc-4f18-896e-2fe79dabf9f9\>
\>
</p>


<!--
## 2-Dimensional KDE visualization 


![KDE_ORI20240914_1851](https://github.com/user-attachments/assets/2cbfdb55-a1d8-454c-bf16-81d5d944fd6c)

![KDE_OVSAP20240914_1748](https://github.com/user-attachments/assets/69d0549a-3e32-465c-a618-09cf2345d46d)
--!>



