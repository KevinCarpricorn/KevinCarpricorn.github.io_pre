---
title: R语言 dnorm, pnorm, qnorm, rnorm的区别
date: 2021-05-09 20:02:30
tags: R
---

# 前言

dnorm, pnorm, qnorm, rnorm 是R语言中常用的正态分布函数. **norm** 指的是正态分布(也可以叫高斯分布(**normal distribution**)), R语言中也有其他不同的分布操作也都类似. **p q d r** 这里分别指的是不同的函数下面将会详细简介这不同函数在正态分布中的应用以及这是个命令在R中如何使用.

## dnorm

**d** - 指的是概率密度函数(probability density function) 

正态分布的公式: 
$$
f(x|\mu, \sigma)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^{2}}{2}}
$$
<img src="https://i.loli.net/2021/05/09/oEpTxL26XQB7AFZ.png" width="75%">

dnorm实质上是正态分布概率密度函数值. 说人话就是返回上面这个函数的值.下面我们在代码中演示下:

```R
# 输出在标准正态分布下(mean = 0, standard deviation = 1) 0 的z-sore
dnorm(0, mean=0, sd=1) # 0.3989423
# 因为是标准正态分布所以mean和sd是可以省略的
dnorm(0) # 0.3989423
# 如果是一个非标准正态分布如下:
dnorm(2, mean=5, sd=3) # 0.08065691
```

## pnorm 

**p** - 指的是概率密度积分函数（从无限小到 x 的积分）(Probability density integral function)

x指的是一个z-score, 专业名词听着玄幻, 其实就是正态分布曲线下x左边的面积(概率占比), 我们知道z-score求在哪个分为数上

```R
# 标准正态分布
pnorm(0) # 0.5 (50%)
pnorm(2) # 0.9772499
# 非标准正态分布
pnorm(2, mean=5, sd=3) # 0.1586553
# 也可以求x右边的概率
pnorm(2, mean=5, sd=3, lower.tail=FALSE) # 0.81586553
# pnorm也能用来求置信区间
pnorm(3) - pnorm(1) # 0.1573054
```

<img src="https://i.loli.net/2021/05/09/UunzrTedDcxh7Vf.png">

上图用R可以这么写

```R
pnorm(2) # 0.9772499
```

## qnorm 

**q** - 指的是分位数函数(quantile function)

简单来说它就是pnorm的反函数, 通过百分比算z-score, 我知道分位数求z-score, 例如:

```R
# 在标准正态分布中求z-score
qnorm(0.5) # 0
qnorm(0.96) # 1.750686
qnorm(0.99) # 2.326348
```

## rnorm

**r** - 指的是随机数函数(random function)（常用于概率仿真）

它是用来生成一组符合正态分布的随机数, 例如:

```R
# 设置随机数种子
set.seed(1)
# 生成5个符合标准正态分布的随机数
rnorm(5) # -0.6264538  0.1836433 -0.8356286  1.5952808  0.3295078
# 生成10个mean=70, sd=5的正态分布随机数
rnorm(10, mean=70, sd=5) # 65.89766 72.43715 73.69162 72.87891 68.47306 77.55891 71.94922 66.89380 58.92650 75.62465
```



在R语言中生成别的各种分布也都是以d, p, q, r开头, 原理和正态分布相似





## references

http://www.360doc.com/content/18/0913/18/19913717_786412696.shtml

https://www.runoob.com/r/r-basic-operators.html

