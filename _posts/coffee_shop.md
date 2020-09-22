
# Waiting in Line at the Coffee Shop (Queueing Theory)

Suppose you walk into your favorite coffee shop one day, and you notice that the line is rather long today. That makes sense, since it's 10am on a Sunday. You count eight people ahead of you in line, and a quick Google search says that 90 seconds is a good estimate how long it takes a barista to make an average cup of coffee. Eight times ninety, you think, that's about twelve minutes. Not so bad!

When you already know how many people are in line ahead of you, it's easy to estimate how long you will spend in line given that you know how long it takes to make coffee. But what about before you enter the shop, when you don't know the length of the line? How can you get an estimate how long it will take to get your coffee then?

## Mathematically modeling the line

For this question, we can turn to the mathematical theory of queues, which was originally developed by Erlang in the early 20th century for telecommunications. This is a branch of the broader theory of stochastic processes, which models phenomena in terms of randomness.

For this example, there are two sources of randomness: the rate at which new customers enter the shop, and the rate at which baristas make coffee. If we denote the number of people in line (at time $t$) by $N(t)$, then $N(t)$ is a random variable in the set $\{0,1,2,\dots\}$ which increases by $1$ whenever a customer enters the shop and decreases by $1$ whenever the barista makes a coffee. Let's make the following assumptions:

1. Customers arrive in the coffee shop independently of one another with an average rate of 🧍 customers per hour.

2. Baristas prepare drinks independently of one another with an average rate of ☕ drinks per hour.

The assumption of *independence* of new customers guarantees that they arrive according to a *Poisson process* with rate 🧍. The same logic applies to the baristas making coffee, however it only follows a Poisson process with rate ☕ if the line is nonempty. Otherwise, if the line is empty, then the baristas will probably be on their phones.

Though these assumptions seem lightweight, they have actually uniquely specified the behavior of $N(t)$!  In particular, they imply that the process $\{N(t)\mid t\geq 0\}$ is a *birth-death process*, which is a special type of continuous-time Markov chain. Birth-death processes have been used to model the population of a species (hence the name), the spread of infectious disease, or in our case, the number of people in a queue.

## Simulating in Python

There are a few ways to simulate a birth-death process (as described in this blog post). The first way is to sample the duration of time between birth-death events. This can be done by sampling the amount of time until the next birth, sampling the amoun of time until the next death, and taking the minimum. This is fine to do because the exponential distribution is *memoryless*. 


```python
np.random.poisson(40)
```




    35



![a.gif](attachment:/assets/coffee_shop/a.gif)

# Deriving the stationary distribution

With simulations in hand, we have a rough idea of how the line evolves over time. In particular, we saw that if the rate of customers is less than the rate of coffee, then the average length of the line converges to a finite number. In order to make calculations, we need to step away from considering particular instances of the stochastic process. Instead, we should consider how the distribution of all possible states evolves over time.

This comes from recognizing that, for each fixed $t$, the number of people in line $N(t)$ is a *random variable*, with a particular probability associated to each of the states. For a given state $k$ in $\{0,1,2,\dots\}$, we want to know the longterm probability, call it $\pi_k$, of the line being in that state.

![stationary.gif](attachment:/assets/coffee_shop/stationary.gif)

In the long-term, the number of people in line will follow the stationary distribution. This is given by a probability distrbiution on the set $\{0,1,2,\dots,\}$ which we may denote

$$ \pi = (\pi_0,\pi_1,\pi_2,\dots) $$

We have the transition matrix

$$ Q = \left[ \begin{matrix}
-\lambda & \lambda        & 0              & 0              & \cdots \\
\mu      & -(\mu+\lambda) & \lambda        & 0              & \cdots \\
0        & \mu            & -(\mu+\lambda) & \lambda        & \cdots \\
0        & 0              & \mu            & -(\mu+\lambda) & \cdots \\
\vdots   & \vdots         & \vdots         & \vdots         & \ddots
\end{matrix} \right] $$

The stationary state will satisfy the equation

$$ \pi Q = 0 $$

This leads to $-\lambda \pi_0 + \mu\pi_1 = 0$ and $\lambda \pi_i - (\mu+\lambda)\pi_{i+1} + \mu\pi_{i+2} = 0$ for all $i\geq 0$.

Without diving into the details, this is a second order linear recurrence whose solution is given by $\pi_n = \left(1 - \frac{\lambda}{\mu}\right)\left(\frac{\lambda}{\mu}\right)^n$. (Exercise for the reader: prove it!) One could recognize that *if $\lambda<\mu$*, this is the probability mass function (pmf) of a geometric distribution with parameter $p=1-\frac{\lambda}{\mu}$.

Therefore, the average length of the line is given by

$$ E[N] = \frac{\lambda ☕}{\mu-\lambda} $$


```python
def simulate(customer_rate, barista_rate, t_max, dt=0.01):
    
    T = np.arange(start=0, stop=t_max, step=dt)
    m = len(T)
    N = np.zeros(m)
    
    for i in range(m-1):
        
        new_customer = np.random.binomial(1, customer_rate*dt)
        new_drink    = np.random.binomial(1, barista_rate*dt)
        
        N[i+1] = N[i]
        
        if new_customer:
            N[i+1] += 1
        if new_drink and N[i+1] > 0:
            N[i+1] -= 1
        
    return (T, N)
```


```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
```


```python
def simulate(customer_rate, barista_rate, t_max, dt=0.01):
    
    T = np.arange(start=0, stop=t_max, step=dt)
    m = len(T)
    N = np.zeros(m)
    
    for i in range(m-1):
        
        new_customer = np.random.binomial(1, customer_rate*dt)
        new_drink    = np.random.binomial(1, barista_rate*dt)
        
        N[i+1] = N[i]
        
        if new_customer:
            N[i+1] += 1
        if new_drink and N[i+1] > 0:
            N[i+1] -= 1
        
    return (T, N)
```


```python
def plot(T, N, t_max):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Number of 🧍')
    ax.set_title('Number of 🧍 in line over time')
    ax.set_xlim((0, t_max))

    ax.plot(T, N)
    plt.show()
```


```python
T, N = simulate(customer_rate=10, barista_rate=5, t_max=1)
plot(T, N, t_max=1)
```


![png](output_11_0.png)



```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
from matplotlib import animation, rc
from IPython.display import HTML


fig = plt.figure()
ax = plt.axes(xlim=(0, 60), ylim=(0,4))
line1, = ax.plot([], [], lw=3, color='lightblue')
line2, = ax.plot([], [], lw=3, color='tomato')

ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Number of customers')
ax.set_title('Number of customers in line over time')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line,
def animate(i):
    print('making frame'+str(i),end='\r')
    x = T[:i]
    y = N[:i]
    y_ave = A[:i]
    line1.set_data(x, y)
    line2.set_data(x, y_ave)
    return line,

frames = len(T)
print('\nwriting frames')
anim = FuncAnimation(fig, animate, init_func=init,
                               frames=np.arange(0,len(T),int(6000/300)), interval=200, blit=True, repeat=True)
print('saving file')
anim.save('a.gif', writer='imagemagick', fps=30)
```

    
    writing frames
    saving file
    making frame5980


![png](output_12_1.png)


<img src="a.gif">


```python
def plots(N_ts, t_max):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Number of customers')
    ax.set_title('Number of customers in line over time')
    ax.set_xlim((0, t_max))

    for N_t in N_ts:
        ax.plot(*np.transpose(N_t))
    plt.show()
```


```python
N_ts = [simulate(customer_rate=10, barista_rate=5, t_max=60) for i in range(10)]
plots(N_ts, t_max=60)
```


![png](output_15_0.png)



```python
N_t = simulate(customer_rate=5, barista_rate=5, t_max=60)
plot(N_t, t_max=60)
```


![png](output_16_0.png)



```python

```


```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
from matplotlib import animation, rc
from IPython.display import HTML


fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,
def animate(i):
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True, repeat=True)
```


```python
n = [0,1,2,3,4,5,6,7,8,9]

pi = [1,0,0,0,0,0,0,0,0,0]

dt = 0.01

lam = 1
mu = 5

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15.
    y += np.pi / 20.
    im = plt.imshow(f(x, y), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# ani.save('dynamic_images.mp4')

plt.show()


```


```python
N = np.zeros((100,10))
N[0] = (1,0,0,0,0,0,0,0,0,0)
pi = N[0]
pi_next = pi
pi[1] += lam*dt*pi[0]
pi[2] += lam*dt*pi[1] + mu*dt*pi[3] - lam*dt*pi[2] - mu*dt*pi[2]
...
pi[9] += lam*dt*pi[8] - mu*dt*pi[9]
```




    array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
def simulate2(customer_rate, barista_rate, t_max, dt=0.01):
    
    T = np.arange(start=0, stop=t_max, step=dt)
    m = len(T)
    no_states = 10
    N = np.zeros((m, no_states))
    N[0][0] = 1
    
    for i in range(m-1):
        
        N[i+1] = N[i]
        
        a = customer_rate
        b = barista_rate
        
        N[i+1][0] += (b*N[i][1] - a*N[i][0])*dt
        for j in range(1,no_states-1):
            N[i+1][j] += (b*N[i][j-1] - a*N[i][j+1] - (a+b)*N[i][j])*dt
        N[i+1][no_states-1] += (a*N[i][no_states-2] - b*N[i][no_states-1])*dt
        
        
    return (T, N)
```


```python
T,N = simulate2(1,2,1)
```


```python
def plot_i(T,N,i):
    plt.bar(np.arange(10),N[i])
    
def plot_i(T, N, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Number of customers')
    ax.set_title('Number of customers in line over time')
    ax.set_ylim((0, 1))

    ax.bar(np.arange(10),N[i])
    plt.show()
```


```python
len(N)
```




    100




```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
from matplotlib import animation, rc
from IPython.display import HTML


fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)


fig=plt.figure()

n=100 #Number of frames
x=range(1,6)
barcollection = plt.bar(np.arange(10),N[0])

def animate(i):
    y=N[i]
    for j, b in enumerate(barcollection):
        b.set_height(y[j])

anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=n,
                             interval=33)

anim.save('stationary.gif', writer='imagemagick', fps=10)
```


![png](output_25_0.png)



![png](output_25_1.png)



```python
plot_i(T,N,50)
```


![png](output_26_0.png)


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

Since $\lambda/\mu$ is the critical factor, let $q_n=\prod_{i=0}^{n-i}p_i$ and
$$ n^* = \inf\{q_i:q_i<\lambda<\mu\} $$
Then $E{N]

### References

Kendall, D. G. (1953). "Stochastic Processes Occurring in the Theory of Queues and their Analysis by the Method of the Imbedded Markov Chain". The Annals of Mathematical Statistics. 24 (3): 338–354. doi:10.1214/aoms/1177728975. JSTOR 2236285.


```python
!brew install imagemagick
```

    'brew' is not recognized as an internal or external command,
    operable program or batch file.
    


```python
anim.save('myAnimation.gif', writer='imagemagick', fps=30)
```

![myAnimation.gif](attachment:/assets/coffee_shop/myAnimation.gif)


```python

```
