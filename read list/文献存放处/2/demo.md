# Article

Pfeiffer M et al., A Data-driven Model for Interaction-aware Pedestrian 
Motion Prediction in Object Cluttered Environments, arXiv 1709.08528, 
2018

https://ieeexplore.ieee.org/abstract/document/8461157

# Data 

## Aim/Innovation

目的是解决Interaction-aware Pedestrian Motion Prediction in Object Cluttered Environments

创新点是 解决此类问题引入了静态的场景，介绍了一种处理动态对象的新方法（APG）

## Motivation

The existing approaches are limited by at least one of the following shortcomings: (i) The feature functions, which abstract agent trajectory information to an internal representation, are hand-crafted and therefore can only capture simple interactions. (ii) The approaches are not scalable to dense crowds since they use pairwise interactions between all agents [1]–[3], which leads to a quadratic complexity in number of agents, and therefore real-time computation is only feasible for a small number of agents. (iii) Static obstacles are neglected [1], [4], [5] and (iv) knowledge about a set of potential destinations is assumed [1], [3], [5]

现有的方法至少受到以下缺点之一的限制：(i)特征函数将代理轨迹信息抽象到内部表示中，它是手工制作的，因此只能捕获简单的交互。(ii)这些方法不能扩展到密集的人群，因为它们使用所有代理[1]-[3]之间的成对交互，这导致了代理数量的二次复杂度，因此实时计算只对少数代理可行。（3）静态障碍被忽略了，[1]，[4]，[5]和(iv)关于一组潜在目的地的知识被假设是[1]，[3]，[5]

## Contribution

introduce a new way of handling dynamic objects, the angular pedestrian grid (APG)

介绍了一种处理动态对象的新方法，即角度步行网格（对行人建模）

In this paper, we introduced a new approach to model pedestrian dynamics and interactions among them. The model architecture is based on an LSTM neural network which is trained from demonstration data.

在本文中，我们介绍了一种新的方法来模拟行人动力学和它们之间的相互作用。 该模型是基于一个LSTM神经网络，它是由演示数据训练的

## Conclusion

We provide an extensive evaluation and comparison against state-of-the-art approaches, both on simulated and real-world data, where we resort to a well-known publicly available dataset for pedestrian motion in public areas .We hypothesize that by taking into account both pedestrian interactions and the static obstacles, the state-of-the-art prediction approaches can be outperformed

我们提供了一个广泛的评估和比较，与最先进的方法，无论是在模拟和现实世界的数据，我们诉诸于一个众所周知的公开数据集，行人运动在公共区域。 我们假设，通过考虑行人的相互作用和静态障碍，这个方法都最好。

## Background

When navigating in such workspaces shared with humans, robots need accurate motion predictions of the surrounding pedestrians

当机器人在与人类共享的工作空间中导航时，机器人需要对周围行人进行准确的运动预测

## Key results







## Method

Angular pedestrian grid (APG)对行人的轨迹建模

lstm



![1615642367(1)](./image/1615642367(1).jpg)

# comment

 One possibility to deploy the presented model on a robotic platform is to predict trajectories of surrounding pedestrians and plan a collision-free path for the robot given the predicted trajectories.

这篇文章写的好



