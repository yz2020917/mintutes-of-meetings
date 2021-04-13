# Article

Alahi A et al., Social LSTM: Human Trajectory Prediction in Crowded Spaces, IEEE CVPR 2016





# Data 

## Aim/Innovation Point

The goal of this paper is to predict the motion dynamics in crowded scenes

本文的目的是预测拥挤场景中的运动动力学。

区别于传统的手动设计函数，利用机器的学习的函数对--动态的，相互作用的行人---人去预测他们的轨迹。

改变lstm架构 加入social pooling

## Motivation

在过去的工作中：

They use hand-crafted functions to model ”interactions” for specific settings rather than inferring them in a data-driven fashion. This results in favoring models that capture simple interactions (e.g. repulsion/attractions) and might fail to generalize for more complex crowded settings. ii) They focus on modeling interactions among people in close proximity to each other (to avoid immediate collisions). However, they do not anticipate interactions that could occur in the more distant future

前人使用手工设计的函数只能建模简单的交互，而且可能在复杂场景下失效。并且建模彼此之间临近人的交互，并没有预期更遥远的将来出现的互动。



## Contribution

用学习的方法（social-lstm）预测行人的移动

social-lstm 提出了新的池化层

## Conclusion

We have presented a LSTM-based model that can jointly reason across multiple individuals to predict human trajectories in a scene. We use one LSTM for each trajectory and share the information between the LSTMs through the introduction of a new Social pooling layer. We refer to the resulting model as the “Social” LSTM. Our proposed method outperforms state-of-the-art methods on two publicly available datasets. In addition, we qualitatively show that our Social-LSTM successfully predicts various non-linear behaviors arising from social interactions, such as a group of individuals moving together.

我们提出了一个基于LSTM的模型，它可以在多个个体之间联合推理来预测场景中的人类轨迹。 我们对每个轨迹使用一个LSTM，并通过引入一个新的社会池层在LSTM之间共享信息。 我们将得到的模型称为“社会”LSTM。 我们提出的方法在两个公开可用的数据集上优于最先进的方法。 此外，我们定性地表明，我们的Social-LSTM成功地预测了由社会互动产生的各种非线性行为，例如一群个体一起移动。



## Background

Pedestrians follow different trajectories to avoid obstacles and accommodate fellow pedestrians. Any autonomous vehicle navigating such a scene should be able to foresee the future positions of pedestrians and accordingly adjust its path to avoid collisions.

行人遵循不同的轨迹，以避免障碍物和容纳其他行人。任何在该场景中导航的自动驾驶车辆都应该能够预见到行人在未来的位置，并相应地调整其路径以避免碰撞.





## Key results

各类模型对比的结果

![1615443680(1)](D:\github\safedriving-record\read list\文献存放处\1\image\1615443680(1).jpg)

轨迹预测可视化

![1615444096(1)](D:\github\safedriving-record\read list\文献存放处\1\image\1615444096(1).jpg)

实例场景对比

![1615444210(1)](D:\github\safedriving-record\read list\文献存放处\1\image\1615444210(1).jpg)



## Method

在一些开源数据集上对视频中的人类行为建模，生成序列的数据，





# comment

重点在动态的相互作用的行人，考虑场景。

