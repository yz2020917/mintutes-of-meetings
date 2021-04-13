# Article

Gupta A et al., Social GAN: Socially Acceptable Trajectories with
Generative Adversarial Networks, CVPR 2018



# Data 

## Aim/Innovation Point

given observed motion trajectories of pedestrians (coordinates for the past e.g. 3.2 seconds), predict all possible future trajectories.

![1615811914(1)](./image/1615811914(1).jpg)





创新 使用gan 并引入新的池化机制。预测所有可能的轨迹

## Motivation

Robicquetet al. [38] have shown that pedestrians have multiple navigation styles in crowded scenes given a mild or aggressive style of navigation. Therefore, the forecasting task entails outputting different possible outcomes

这个任务是需要多输出的

过去的任务：

Hence, they do not have the capacity to model interactions between all people in a scene in a computationally efficient fashion.

因此，他们没有能力以计算高效的方式建模场景中所有人之间的交互

commonly used loss function that minimizes the euclidean distance between the ground truth and forecasted outputs. In contrast, we aim in learning multiple “good behaviors”, i.e., multiple socially acceptable trajectories

常用的损失函数，它最小化地面真相与预测输出之间的欧氏距离。 相反，我们的目标是学习多种“良好行为”，即多种社会可接受的轨迹。

## Contribution

 (i) we introduce a variety loss which encourages the generative network of our GAN to spread its distribution and cover the space of possible paths while being consistent with the observed inputs. 

(ii) We propose a new pooling mechanism that learns a “global” pooling vector which encodes the subtle cues for all people involved in a scene. We refer to our model as “Social GAN”.

(i)我们引入了各种损失，它鼓励我们的GAN的生成网络传播其分布，并覆盖可能路径的空间，同时与观察到的输入保持一致。(ii)我们提出了一种新的池机制，它学习一个“全局”池向量，它编码一个场景中涉及的所有人的微妙线索。我们将我们的模式称为“社会gan”。

## Conclusion



We propose a novel GAN based encoder-decoder framework for trajectory prediction capturing the multi-modality of the future prediction problem.

a recurrent sequence-to-sequence model observes motion histories and predicts future behavior, using a novel pooling mechanism to aggregate information across people

## Background

Understanding human motion behavior is critical for autonomous moving platforms (like self-driving cars and social robots) if they are to navigate human-centric environments.

了解人类的运动行为对于自动移动平台（比如自动驾驶的汽车和社交机器人）来说，是至关重要的.





## Key results







## Method





# comment



开门见题加图片

idea ： 他是找到问题的不足，对问题进行了改变，引出新的网络结构，和创新。多轨迹的预测。

改问题，改网络。问题跟网络有耦合的关系.

