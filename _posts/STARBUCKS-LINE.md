---
layout: post
title:  "Expected Length of the Starbucks Line (Queueing Theory)"
date:   2020-09-06 22:18:10 +0000
---

# Expected Length of the Starbucks Line (Queueing Theory)

It's 5:45 PM on a Sunday, and you are sitting in the university library trying to crank out the last of your weekend assignments. You've been working since noon, and you decide it's time to hotwire your exhausted brain with some sweet, sweet caffeine. It's only a short walk to the nearest coffee shop, so you pack up your laptop and notebooks and decide to go for it.

Upon arriving, you see the line, and it is very long. "Isn't coffee supposed to be a morning drink?" you think to yourself, "On the other hand, here I am..." You weigh your limited schedule with your addiction to coffee, and decide the 20 minute wait is worth it.

After standing in line, shuffling through your playlists and staring out the window, it's finally time to order. After another 10 minutes standing with your hands in your pockets in the waiting area, making a point not to look over the baristas' shoulders as they work, you grab your coffee and go. 

## Poisson processes

I've already made a blog post describing Poisson processes before, but since they are so ubiquotous in stochastic processes, it is worth another go. Let's make the following assumptions about Starbucks:

1. The customers arrive at an average rate of $\lambda$ customers per hour
2. The baristas make, on average, $\mu$ drinks per hour
3. The arrival times of any two customers are independent
4. The amount of time spent making any two drinks are independent

Then viola! the arrival of customers and creation of drinks both follow a Poisson process with rates $\lambda$ and $\mu$ respectively.

Let's denote the number of customers in the Starbucks line at time $t$ by $N(t)$. This quantity will always be a integer that is greater than or equal to zero. 

$N(t)$ increases whenever new customers enter the building and go to stand in line, and it decreases whenever the Barista finishes making a coffee. Let's make two critical assumptions about the coming and going of customers:

1. Customers enter the store according to a Poisson process with rate $\lambda$.
2. Baristas make the drinks according to a Poisson process with rate $\mu$.
3. The probability that a new customer will decide to stand in line decreases to zero as the number of customers increases arbitrarily. That is to say, the more customers are in line, the less likely new customers will decide to wait.


```python
import numpy as np
```


```python
dt = 0.01
T = np.arange(0,1000,dt)
```


```python
N = np.zeros(len(X))
```


```python
customer_rate = 1
drink_rate = 1/3

probability_fn = lambda n : 1/(n+1)

dt = 0.01
T = np.arange(0,10000,dt)

N = np.zeros(len(T))

for i in range(len(N)-1):
    
    n = N[i]
    p = probability_fn(n)
    
    N[i+1] = n
    r = np.random.random()
    
    if r < np.random.exponential(customer_rate*p*dt):
        N[i+1] += 1
        
    if r < np.random.exponential(drink_rate*dt):
        N[i+1] -= 1
    
    
plt.plot(T,N)
    
```

    c:\python36\lib\site-packages\ipykernel_launcher.py:4: RuntimeWarning: divide by zero encountered in double_scalars
      after removing the cwd from sys.path.
    




    [<matplotlib.lines.Line2D at 0x2051a279c50>]




![png](output_7_2.png)



```python
plt.hist(N,bins=[1,2,3,4,5,6,7,8,9,10])
```




    (array([188946., 311159., 278307., 141800.,  35744.,   9435.,   1712.,
              1210.,    326.]),
     array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
     <a list of 9 Patch objects>)




![png](output_8_1.png)



```python
plt.plot(T,N)
```


    ------------------------------------------

    ValueErrorTraceback (most recent call last)

    <ipython-input-37-3c5e3520bd4c> in <module>
    ----> 1 plt.plot(T,N)
    

    c:\python36\lib\site-packages\matplotlib\pyplot.py in plot(scalex, scaley, data, *args, **kwargs)
       2811     return gca().plot(
       2812         *args, scalex=scalex, scaley=scaley, **({"data": data} if data
    -> 2813         is not None else {}), **kwargs)
       2814 
       2815 
    

    c:\python36\lib\site-packages\matplotlib\__init__.py in inner(ax, data, *args, **kwargs)
       1808                         "the Matplotlib list!)" % (label_namer, func.__name__),
       1809                         RuntimeWarning, stacklevel=2)
    -> 1810             return func(ax, *args, **kwargs)
       1811 
       1812         inner.__doc__ = _add_data_doc(inner.__doc__,
    

    c:\python36\lib\site-packages\matplotlib\axes\_axes.py in plot(self, scalex, scaley, *args, **kwargs)
       1609         kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D._alias_map)
       1610 
    -> 1611         for line in self._get_lines(*args, **kwargs):
       1612             self.add_line(line)
       1613             lines.append(line)
    

    c:\python36\lib\site-packages\matplotlib\axes\_base.py in _grab_next_args(self, *args, **kwargs)
        391                 this += args[0],
        392                 args = args[1:]
    --> 393             yield from self._plot_args(this, kwargs)
        394 
        395 
    

    c:\python36\lib\site-packages\matplotlib\axes\_base.py in _plot_args(self, tup, kwargs)
        368             x, y = index_of(tup[-1])
        369 
    --> 370         x, y = self._xy_from_xy(x, y)
        371 
        372         if self.command == 'plot':
    

    c:\python36\lib\site-packages\matplotlib\axes\_base.py in _xy_from_xy(self, x, y)
        229         if x.shape[0] != y.shape[0]:
        230             raise ValueError("x and y must have same first dimension, but "
    --> 231                              "have shapes {} and {}".format(x.shape, y.shape))
        232         if x.ndim > 2 or y.ndim > 2:
        233             raise ValueError("x and y can be no greater than 2-D, but have "
    

    ValueError: x and y must have same first dimension, but have shapes (10000,) and (1000,)



![png](output_9_1.png)


## The stationary distribution

In the long-term, the number of people in line will follow the stationary distribution. This is given by a probability distrbiution on the set $\{0,1,2,\dots,\}$ which we may denote

$$ \pi = (\pi_0,\pi_1,\pi_2,\dots) $$

We have the transition matrix

$$ Q = \left[ \begin{matrix}
-\lambda p_0 & \lambda p_0 & 0 & 0 & 0 & \cdots \\
\mu & -(\mu+\lambda p_1) & \lambda p_1 & 0 & 0 & \cdots \\
0 & \mu & -(\mu+\lambda p_2) & \lambda p_2 & 0 & \cdots \\
0 & 0 & \mu & -(\mu+\lambda p_3) & \lambda p_3 & \cdots \\
0 & 0 & 0 & \mu & -(\mu+\lambda p_4) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots
\end{matrix} \right] $$

The stationary state will satisfy the equation

$$ \pi Q = 0 $$

This leads to

$$ -\lambda p_0\pi_0 + \mu\pi_1 = 0 $$
and
$$ \lambda p_i\pi_i - (\mu+\lambda p_{i+1})\pi_{i+1} + \mu\pi_{i+2} = 0 $$

Solving the first equation for $\pi_1$ we obtain
$$ \pi_1 = \frac{\lambda}{\mu}p_0\pi_0 $$
Solving the second equation for $\pi_{i+2}$ we get
$$ \pi_{i+2} = (1+\frac{\lambda}{\mu}p_{i+1})\pi_{i+1}-\frac{\lambda}{\mu} p_i\pi_i $$
Let us replace $r=\lambda/\mu$ for convenience. Then
$$ \pi_{i+2} = (1+rp_{i+1})\pi_{i+1}-rp_i\pi_i $$

Now we can solve to $\pi_2$
$$ \pi_2 = (1+rp_1)\pi_1-rp_0\pi_0 $$
Then
$$ \pi_2 = (1+rp_1)rp_0\pi_0 - rp_i\pi_i $$
So
$$ \pi_2 = r^2p_0p_1\pi_0 $$

In general, we conjecture the following:
$$ \pi_n = \left(\frac{\lambda}{\mu}\right)^n\pi_0\prod_{i=0}^{n-1}p_i $$

Since $\sum_{n=0}^{\infty}\pi_n=1$, it follows that

$$ \sum_{n=0}^{\infty}\left(\frac{\lambda}{\mu}\right)^n\pi_0\prod_{i=0}^{n-1}p_i =1 $$
So
$$ \pi_0 = \frac{1}{\sum_{n=0}^{\infty}\left(\frac{\lambda}{\mu}\right)^n\prod_{i=0}^{n-1}p_i} $$


Its expected value is given by
$$ E[N] = \sum_{n=0}^{\infty} n\left(\frac{\lambda}{\mu}\right)^n\pi_0\prod_{i=0}^{n-1}p_i $$

In the special standard case when the probabilites are all $1$ we have
$$ E[N] = \sum_{n=0}^{\infty} n \left(\frac{\lambda}{\mu}\right)^n \pi_0 $$
where
$$ \pi_0 = \frac{1}{\sum_{n=0}^{\infty}\left(\frac{\lambda}{\mu}\right)^n} = 1 - \frac{\lambda}{\mu} $$


```python

```
