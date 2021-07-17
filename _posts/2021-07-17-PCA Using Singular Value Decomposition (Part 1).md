---
title: "PCA Using Singular Value Decomposition (Part 1)"
excerpt: "What is principal component analysis and how it is related to singular value decomposition."
last_modified_at: 2021-07-17T09:45:06+06:00
tags: 
  - Principal Component Analysis 
  - Singular Value Decomposition
  - Machine Learning
  - Linear Algebra
categories:
  - Machine-Learning
toc: true
toc_label: "Contents"
toc_sticky: true 
words_per_minute: 100
---

## What is PCA
Using Principal Component Analysis (PCA) we can change the basis of data points. Suppose, we have $ m $ data points each 
with $ n $ dimension. Then, using PCA we can map the data points using $ k $ $ ( k \leq n ) $  unit vectors as the new basis. 
If $ k < n $ then we have reduced the dimension of the data points from $ n $ to $ k $, may be with some data loss. 
In Machine Learning terms, each dimension is a feature vector. Thus we can use PCA to extract the important 
feature let say, most important subsets of feature that can explain 90% of data variance. 

## Prerequisites

Before continuing how PCA works and its mathematical explanation we need to have basic understanding of 
some linear algebra concepts.

### Eigenvalue, Eigenvector and Eigen decomposition

A matrix represents linear transformation of a vector. Let $ A $ be a linear transformation square 
matrix and $ v $ be a vector, then $ y = Av $ is transformation of $ v $ using $ A $. 
Linear transformation can 
have a rotation, then a scaling then another rotation of a vector. Lets say $ v $ is vector on which 
transformation using $ A $ doesn't cause it to rotate rather just makes it scale. Then, we can say $ v $ 
is a eigenvector of matrix $ A $ and the scaling factor is a eigenvalue of matrix $ A $.
\\[
Av = λv
\\]
where $ λ \in R $, $ A \in R^{n \times n} $ and $ v \in R^{n \times 1} $.
Here $ v $ is a eigenvector and $ λ $ is eigenvalue of matrix $ A $.
If $ A $ is real symmetric matrix then it satisfies two properties of eigenvalue and eigenvector.

1. It has $ n  $ eigenvalues (not necessarily distinct) and one eigenvector for each eigenvalues.
2. These eigenvectors are mutually orthogonal.

So we have,

$$
\begin{align}
Av_1 &= λ_1v_1 \tag{1}\label{eq1} \\
Av_2 &= λ_2v_2 \tag{2}\label{eq2} \\
... \notag \\
Av_n &= λ_nv_n \tag{3}\label{eq3} 
\end{align}
$$

Putting these all together

$$
A [v_1 \ v_2 ... v_n] = [v_1 \ v_2 ... v_n] Λ \tag{4}\label{eq4}
$$

Here $ Λ $ is a diagonal matrix whose $ Λ_{ii} = λ_i $ and $ v_i $'s are column vectors 
(also eigenvectors of $ A $). We can write eqn \eqref{eq4} as,

$$
AQ = QΛ
$$

For orthogonal matrices, we know that $ M^{-1} = M^\top $. So multiplying both side by $Q^{-1}$, 
we get,

$$
\begin{align}
A &= QΛQ^{-1} \\
  &= QΛQ^\top 
\end{align}  \tag{x}\label{eqx}
$$

Decomposing A with multiplication of three matrices using eigenvalues and eigenvectors is 
the Eigendecomposition of matrix.


### Singular Value Decomposition

Lets say $ A $ is $ m \times n $ real matrix and $ v_1 $ and $ v_2 $ be two $ 1 \times n $ 
real matrix which are orthonormal. 
Then $ A $ transforms $ v_i $ (i.e. $ Av $) from $ R^n $ dimension to $ R^m $ dimension. 
Let the transformed vector 
be $ u_1 $ and $ u_2 $. If we can find such orthonormal $ v_i $ such that their 
transformed vectors $ u_i $'s are 
also orthonormal, then we can write,

$$
\begin{align}
Av_1 &= σ_1u_1 \tag{5}\label{eq5} \\
Av_2 &= σ_2u_2 \tag{6}\label{eq6} \\
... \\
Av_n &= σ_nu_n \tag{7}\label{eq7} \\
\end{align}
$$

Here $ σ_i $'s are the scaling factor. Putting these all together we have,

$$
A [v_1 \ v_2 ... v_n] = [u_1 \ u_2 ... u_n] Σ \tag{8}\label{eq8} 
$$

Here $ Σ $ is a diagonal matrix whose $ Σ_{ii} = σ_i $ and $ v_i $'s, $ u_i $'s are column vectors. 
Thus we can write eqn \eqref{eq8} as,

$$
\begin{align}
AV &= UΣ 
\end{align}
$$

Since $ U $ and $ V $ are orthonormal, thus $ V^\top = V^{-1} $ and $ V V^\top = V^\top V = I $. 
Same is true for $ U $. Now,
multiplying both side of the above equation with $ V^\top $ , we get

$$
\begin{align}
 A &= UΣV^\top 
\end{align}
$$

Here $ A $ is of dimension $ m \times n $, $ U $ is of dimension $ m \times m $, $ Σ $ is of dimension 
$ m \times n $ and $ V $ is of dimension $ n \times n $. Now to find $ U $ and $ V $ 
we can use eigendecomposition of a matrix.

$$
\begin{align}
A A^\top &= U Σ V^\top {(U Σ V^\top)}^\top \\
         &= U Σ V^\top V Σ^\top U^\top
\end{align}
$$

Since, $ V $ is orthonormal, so $ V^\top V  = I $ and we know that $ MI = M $, where $ I $ is an
identity matrix. So,

$$
\begin{align}
A A^\top &= U Σ Σ^\top U^\top 
\end{align}
$$

For diagonal matrix  $ Σ^\top=Σ $.

$$
\begin{align}
A A^\top &= U Σ^2 U^\top \tag{y}\label{eqy}
\end{align}
$$

Now comparing eqn \eqref{eqx} and eqn \eqref{eqy} can easily see that eigenvector 
matrix of $ AA^\top $ is the $ U $ in singular value decomposition. Similar to how we get eqn \eqref{eqy}, We see

$$
\begin{align}
A^\top A &= V Σ^\top U^\top U Σ V^\top \\
         &= V Σ^2 V^\top
\end{align}
$$

Now we can easily see that eigenvector matrix of $ A^\top A $ is the $ V $ in singular value 
decomposition. Also, in both cases $ Σ^2 = Λ $ of eigenvalues. Since, $ Λ $ is a diagonal matrix, 
$ Σ_{ii} = \sqrt{Λ_{ii}} $. Here, $ Σ_{ij} = 0 $ if $ i \neq j $.


## PCA using SVD

Now how does PCA relates to these concepts? Lets say we want to reduce dimension of 
our feature vectors. We have $ m $ vectors with dimension $ n $, given by a matrix $ A $ of 
dimension $ m \times n $. Let's assume the vectors are normalized such that mean of each 
dimension is $ 0 $. Now we want to reduce dimension from $ n $ to $ k \ ( k \leq n ) $. 
Let's first consider $ n= 2 \ and \ k = 1 $. Let the basis vector of the new one dimensional 
coordinate system is $ x $.

As we can see when reducing a 2 dim feature vector to 1 dim vector, among all the basis vector, 
the best basis vector has sum of distance from each point to the line is lowest. 
This is actually true for reducing any $ n $ dimension to $ k $ dimension. Sum of distance 
from each point to each basis vector is lowest. Then our dimension reduction loses 
lowest information. It is also easy to visualize this when reducing dimension from 
3 to 2. From each point sum of distance to the 2 dimensional plane is lowest. Since 
in our feature vector their relative position with each other conveys the information, 
not how they are in $ n $ dimensional plane. Thus for a specific basis vector our goal 
is to reduce the distance from a point to this vector. In one dimensional case,
Let the point be $ r $. Our basis vector is $ x $, distance vector from $ r $ to $ x $ is $ d $ and 
projection of $ r $ on $ x $ is $ p $ then,

$$
{\vert \vert r \vert \vert}^2 = {\vert \vert d \vert \vert}^2 + {\vert \vert p \vert \vert}^2
$$

But $ {\vert \vert r \vert \vert}^2 $ is fixed for a point ( This is the distance of point $ r $ from origin). 
Thus minimizing $ \vert \vert d \vert \vert $ is
equivalent to maximizing $ \vert \vert p \vert \vert $, i.e. the projection vector. 
This is actually true for any dimension, not for just 2 dimension. 
Here, normalizing the points to have mean 0 is important. 
Generalizing, If basis vector $ x $ is a unit vector and $ a $ is a point, Then,
$ \vert \vert p \vert \vert  = ax $, $ a $ have dimension $ 1 \times n $ and $ x $ have dimension $ n \times1 $.
In matrix form for each point,

$$
P = Ax
$$

here $dim(P) = m \times 1 $, $ dim(A) = m \times n $, $ dim(x) = n \times 1 $. $ P_{i1} $ represent 
the projection of vector $ A_i $ on $ x $. Our target is to maximize $ \sum_{i=1}^{m} P_{i1}^2 $ i.e.,
Find such $ x $ that $ \vert \vert P \vert \vert $ is maximized
But $ max(\vert \vert P \vert \vert) = max(\vert \vert Ax \vert \vert) $, where $ x \in R^n $
So we need to $ maximize \ \vert \vert Ax \vert \vert $. Since $ Ax $ is a column vector we can write,

$$
\begin{align}
\vert \vert Ax \vert \vert &= {(Ax)}^\top Ax \\
                           &= x^\top A^\top A x
\end{align}
$$

Let $ v_1, . . . , v_n $ be an orthonormal basis of $ R^n $ consisting of eigenvectors 
of $ A^\top A $. With eigenvalues $ λ_1 \geq λ_2 \geq ..... \geq λ_r \geq 0 \geq .. \geq 0 $. 
By the theory of singular value decomposition of $ A $ we know $ σ_i^2 = λ_i $. So we can write,

$$
x = c_1v_1 + c_2v_2 + ... + c_nv_n. 
$$

Where, $ c_i \in R $ and $ c_1^2 + .... c_n^2 \leq 1 $, Since $ x $ is a unit vector.
Now using the properties of eigenvector, we can write,

$$
A^\top A v_1 = λ_1v_1
$$

Multiplying left of both side by $ v_1^\top $ we get

$$
\begin{align}
v_1^\top A^\top A v_1 &= v_1^\top λ_1 v_1 \\
v_1^\top A^\top A v_1 &= λ_1
\end{align}
$$

Since $ v_i $'s are unit vectors, thus $ v_1^\top v_1 = 1 $
Next, Putting the value of x in $ x^\top A^\top A x $, we get

$$
\begin{align}
x^\top A^\top A x &= c_1^2 λ_1 + .... + c_n^2 λ_n \\
       &\leq λ_1( c_1^2 + .... + c_n^2) \\
       &\leq λ_1
\end{align}
$$

Equality holds if, $ x = v_1 $.
Thus $ \vert \vert Ax \vert \vert $ is maximized if $ x = v_1 $. 
Here $ v_1 $ is the corresponding eigenvector of highest eigenvalue of $ A^\top A $. 
So to reduce dimension of feature vector from $ n $ to $ k $ we can take the maximum 
$ k $ eigenvalue-eigenvector pair of $ A^\top A $ as basis vector. Interestingly, 
$ A^\top A $ is the covariance matrix of $ A $ since they have mean value of 0. 
In the next part of PCA, I will do a python implementation of PCA.




