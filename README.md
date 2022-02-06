## GANS

### Generate Results（WGAN-GP）：

<div align="center">
    <img src="images\fake1.png" height="120" width="188" >
    <img src="images\fake3.png" height="120" width="188" >
    <img src="images\fake4.png" height="120" width="188" >
    <img src="images\fake6.png" height="120" width="188" >
</div>

随着迭代次数的增加，生成的图片越来越真实！😎🎉

### GAN

GAN 的核心思想是：生成器（generator）和判别器（discriminator）的博弈！

生成器：输入按照一定分布初始化的噪声，输出一张图片

判别器：判断生成图片的真假

> 优秀生成器生成的图片会迷惑判别器的判断，使之无法准确的判断图片的真假；优秀判别器可以准确的判断图片是生成的还是真实的！

#### 学习（训练）

> 一般先训练 discriminator，再训练 generator；通常前者训练轮数大于等于1，后者只训练1轮。

- **discriminator loss**

  <img src="images\equation.png" style="zoom:70%;" />

  最大化这个损失函数。

- **generator loss**

  gen 的损失函数只看后面和 gen 有关的一项：

  <img src="images\equation_gen.png" style="zoom:70%;" />

  最小化这个损失函数。

- 对于 generator loss 可以用这个公式代替：$-\frac{1}{m}\sum^{m}_{i=1}log(D(G(z^{i})))$

代码实现 **loss**：

> 使用[torch.nn.BECLoss()](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#bceloss)实现，注意其返回值是加了负号的！

```python
criterion = nn.BCELoss()
# discriminator loss
loss_d = criterion(disc_real, torch.ones_like(disc_real)) \  # real是真实图片数据
+ criterion(disc_fake, torch.zeros_like(disc_fake))  # fake是生成图片数据
# generator loss
loss_g = criterion(gen_fake, torch.ones_like(gen_fake))
```

### WGAN & WGAN-GP

尽管存在效果不错的 **DCGAN**，但是 **GAN** 的训练过程并不容易！

**WGAN** 对 **GAN** 提出了改进（大量公式证明改进）！

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

**WGAN-GP** 对 **WGAN** 进行修改，将 Weight Clipping 改为了 Gradient Penalty！两者的参数分布对比：

![](images\gp.jpg)

#### Gradient Penalty

<img src="images\equation_gp.png" style="zoom:60%;" />

实现时，特别注意最后一项是一个导数项！借助[torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch-autograd-grad)实现！

### Reference（Thanks to!）

**GAN：**

- https://www.zhihu.com/collection/779863559

- https://zhuanlan.zhihu.com/p/266677860

**WGAN & WGAN-GP：**

- https://zhuanlan.zhihu.com/p/25071913

- https://www.zhihu.com/question/52602529/answer/158727900