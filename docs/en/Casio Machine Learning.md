# Casio Coding

*Contact: minhmok095@gmail.com*

## I. Introduction

This document contains multiple machine learning algorithms implemented in multiple ways using the Casio calculator. Of course, being "implemented" on the Casio is loose, because we can not fully write automated programs (or can but really tedious to re-program because the Casio does not store our "programs"), and there are memory constraints so we can not just load any arbitrary amount of training data or model we want, there will always be some manual work to do. Therefore, in this document, I will focus on what I think are the most comfortable ways to implement ML algorithms, whether it is automated on the Casio or some manual pen-and-paper work. Note that I will mainly dive into **how** to implement these algorithms, not deep ML fundamentals.

## II. Linear regression

If you don't already know, linear regression is an algorithm to find a line that best fits our given data points.

A line has the form of:
```
y = w1*x1 + w2*x2 + w3*x3 + ... + b
```

From that, we have a set of equations with different inputs:
```
y1 = w1*x1 + w2*x2 + w3*x3 + ... + b
y2 = w1*x1' + w2*x2' + w3*x3' + ... + b
y3 = w1*x1'' + w2*x2'' + w3*x3'' + ... + b
...
```

We can rewrite this as:
```
Y = X*W
```

where:
```
Y is a vector (y1, y2, y3, ...)
X is a matrix with inputs laid out row-wise:
| x1   x2   x3   ... 1 |
| x1'  x2'  x3'  ... 1 |
| x1'' x2'' x3'' ... 1 |
| ...  ...  ...  ... 1 |
W is a vector (w1, w2, w3, ..., b)
```

Our objective is to find the optimal W, and we can use the closed-form least squares formula for this:
```
W = (X^T * X)^-1 * X^T * Y
```

We can use the matrix mode to calculate this. But there is a problem: The Casio only supports matrices up to 4x4 in size, so we can only have at best 3 features with bias, or 4 features with no bias. As for the data points though, we are not limited to just 4 data rows, due to a nice property: `X^T` has size 4xn, and `X` has size nx4, but `X^T * X` has size 4x4, which is still computable in the matrix mode, same for `X^T * Y` which is `4x1` in the end, so we can think of calculating `X^T * X` and `X^T * Y` outside and use in matrix mode later.

We realize that `X^T * X` has patterns:
```
| Sum(x1^2)    Sum(x1*x2)  ...  Sum(x1*xn)  Sum(x1) |
| Sum(x1*x2)   Sum(x2^2)   ...  Sum(x2*xn)  Sum(x2) |
|    ...          ...      ...      ...       ...   |
| Sum(x1*xn)   Sum(x2*xn)  ...  Sum(xn^2)   Sum(xn) |
| Sum(x1)      Sum(x2)     ...  Sum(xn)       m     |
```

and `X^T * Y` too:
```
| Sum(x1*y) |
| Sum(x2*y) |
|    ...    |
| Sum(xn*y) |
|  Sum(y)   |
```

To calculate sums, you can input `a + b + c + d + ...` directly in your calculator, or you can setup a spreadsheet, use the `SUM` function to calculate sums, and retype in matrices in the matrix mode. Finally, create `MatA` and `MatB` matrices and type:
```
MatA^-1 * MatB
```

With `MatA` being `X^T * X` and `MatB` being `X^T * Y` behind the hood.

For example, I will try to perform linear regression on this dataset:
```
Row   x1  x2  x3  y
 1    1   2   1   7
 2    2   1   2   8
 3    1   3   2   10
 4    3   2   1   11
 5    2   3   3   14
 6    1   1   1   5
```

`X^T * X` now looks like:
```
| Sum(x1^2)   Sum(x1*x2)  Sum(x1*x3)  Sum(x1) |
| Sum(x2*x1)  Sum(x2^2)   Sum(x2*x3)  Sum(x2) |
| Sum(x3*x1)  Sum(x3*x2)  Sum(x3^2)   Sum(x3) |
| Sum(x1)     Sum(x2)     Sum(x3)     m       |
```

`X^T * Y`:
```
| Sum(x1*y) |
| Sum(x2*y) |
| Sum(x3*y) |
| Sum(y)    |
```

We have sums:
```
Sum(x1^2)  = 20
Sum(x2^2)  = 28
Sum(x3^2)  = 20
Sum(x1^x2) = 20
Sum(x1^x3) = 17
Sum(x2^x2) = 22
Sum(x1^y)  = 99
Sum(x2^y)  = 121
Sum(x3^y)  = 101
Sum(x1)    = 10
Sum(x2)    = 12
Sum(x3)    = 10
Sum(y)     = 55
```

Now create Matrix A in matrix mode:
```
| 20  20  17  10 |
| 20  28  22  12 |
| 17  22  20  10 |
| 10  12  10   6 |
```

and Matrix B:
```
|  99 |
| 121 |
| 101 |
|  55 |
```

then calculate:
```
MatA^-1 * MatB
```

we would have:
```
|  2.0652 |
|  2.0761 |
|  1.3478 |
| -0.6739 |
```

so the line we have in the end is:
```
y = 2.0652x1 + 2.0761x2 + 1.3478x3 -0.6739
```

### 1-D linear regression

An interesting thing to note is that the Casio has built-in Linear regression but just for the 1D `y = ax + b` form in the Statistics mode, so if your problem only has 1 feature and you have less than 45 rows of data, simply go into the Statistics mode, choose the 2-variable option, fill in x and y values, then choose regression calc and it will give you `a` and `b` (I don't go into specifics here because models might have different actual namings and order but you should see things similar to what I said).

### Ridge linear regression

Ridge linear regression is a variation with the formula:
```
W = (X^T * X + lambda * I)^-1 * X^T * Y
```

Where I is the identity matrix and lambda is a scalar value that you choose. This has some nice properties, first is that `X^T * X` might not be invertible, but adding `lambda * I` makes it always invertible. It also helps keep the parameters smaller, closer to 0, and prevents overfitting compared to barebone linear regression. The larger is `lambda`,  the more aggressive regularization is.

## II. K Nearest Neighbors

K Nearest Neighbors is perhaps the easiest algorithm to implement in the Casio. We lay out the data in a spreadsheet, dedicate a column for distance calculation, then manually select k nearest data rows and decide what to do from that. Some applications include classification and regression. For classification, you would get the label that showed up the most amongst the neighbors, and you would get the mean result of them for regression problems. Of course there could be more configurations and applications, but I won't go into that here.

As an example, let's solve a classification problem:
```
Fruit A: weight = 150, sweetness = 8, type = Apple
Fruit B: weight = 160, sweetness = 7, type = Apple
Fruit C: weight = 180, sweetness = 9, type = Orange
Fruit D: weight = 170, sweetness = 8, type = Orange
Fruit E: weight = 140, sweetness = 6, type = Apple
Fruit F: weight = 175, sweetness = 9, type = Orange

A new fruit has weight = 165 and sweetness = 8, what type of fruit is it?
```

Enter spreadsheet mode, fill weights in column A, sweetness in column B, and fill formula `(A1-165)^2 + (B1-8)^2` with range `C1:C6` (we don't need square root for actual distances here because it does not affect magnitude comparison). After that, we would see something like:

```
    A   B   C   D   E  
1  150  8  225  _   _
2  160  7  26   _   _
3  180  9  226  _   _
4  170  8  25   _   _
5  140  6  629  _   _
6  175  9  101  _   _
```

The closest fruits are B (apple), D (orange), F (orange), so this fruit is predicted to be an orange.
