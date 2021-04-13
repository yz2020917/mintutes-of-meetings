# Article

Sedegian A et al., SoPhie: An Attentive GAN for Predicting Paths 
Compliant to Social and Physical Constraints, arXiv 1806.01482, Sep. 
2018

# Data 

## Aim/Innovation Point

目标是解决一组代理（agent）的未来路径预测问题

将场景信息与agent状态信息共同考虑，引入注意力机制的同时使用gan进行生成轨迹。

## Motivation

过去的实验并没有将场景信息与agent的状态信息进行结合，多路径预测也很少人做（而且most of these models only incorporate the influence of few adjacent agents in a very limited search space）。

## Contribution

• Our model uses scene context information jointly with
social interactions between the agents in order to predict
future paths for each agent.
• We propose a more reliable feature extraction strategy to
encode the interactions among the agents.
• We introduce two attention mechanisms in conjunction
with an LSTM based GAN to generate more accurate and
interpretable socially and physically feasible paths.
• State-of-the-art results on multiple trajectory forecasting
benchmarks

我们的模型使用场景上下文信息与代理之间的社会交互相结合，以预测每个代理的未来路径。我们提出了一种更可靠的特征提取策略来编码代理之间的相互作用。我们引入了两种注意机制和基于LSTM的GAN相结合，以生成更准确和可解释的社会和物理上可行的路径。在多个轨迹预测基准上的最新结果。

## Conclusion





## Background

This paper addresses the problem of path prediction for multiple interacting agents in a scene, which is a crucial step for many autonomous platforms such as self-driving cars and social robots

本文解决了场景中多个交互代理的路径预测问题，这是许多自动驾驶汽车和社交机器人等自主平台的关键一步



## Key results







## Method









# comment

实验做的大

