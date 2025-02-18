# FUNCTO: Function-Centric One-Shot Imitation Learning for Tool Manipulation 

**Title (ZH)**: FUNCTO: 原点聚焦的功能导向工具操作单次模仿学习 

**Authors**: Chao Tang, Anxing Xiao, Yuhong Deng, Tianrun Hu, Wenlong Dong, Hanbo Zhang, David Hsu, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11744)  

**Abstract**: Learning tool use from a single human demonstration video offers a highly intuitive and efficient approach to robot teaching. While humans can effortlessly generalize a demonstrated tool manipulation skill to diverse tools that support the same function (e.g., pouring with a mug versus a teapot), current one-shot imitation learning (OSIL) methods struggle to achieve this. A key challenge lies in establishing functional correspondences between demonstration and test tools, considering significant geometric variations among tools with the same function (i.e., intra-function variations). To address this challenge, we propose FUNCTO (Function-Centric OSIL for Tool Manipulation), an OSIL method that establishes function-centric correspondences with a 3D functional keypoint representation, enabling robots to generalize tool manipulation skills from a single human demonstration video to novel tools with the same function despite significant intra-function variations. With this formulation, we factorize FUNCTO into three stages: (1) functional keypoint extraction, (2) function-centric correspondence establishment, and (3) functional keypoint-based action planning. We evaluate FUNCTO against exiting modular OSIL methods and end-to-end behavioral cloning methods through real-robot experiments on diverse tool manipulation tasks. The results demonstrate the superiority of FUNCTO when generalizing to novel tools with intra-function geometric variations. More details are available at this https URL. 

**Abstract (ZH)**: 从单个演示视频学习工具使用提供了一种高度直观且高效的机器人教学方法。当前的一次性模仿学习方法难以实现这一目标，主要挑战在于建立具有相同功能的演示工具与测试工具之间的功能对应关系，尤其是面对同一功能内显著的几何变异性时。为了解决这一挑战，我们提出了一种以功能为中心的一次性模仿学习方法FUNCTS（Function-Centric OSIL for Tool Manipulation），该方法通过3D功能关键点表示建立以功能为中心的对应关系，使机器人能够克服显著的功能内几何变异性，从单个演示视频中泛化出对具有相同功能的新工具的工具操作技能。我们通过多样化的工具操作任务的实地机器人实验，将FUNCTS与现有的模块化一次性模仿学习方法和端到端的行为克隆方法进行对比评估。结果表明，FUNCTS在泛化到具有功能内几何变性的新工具时表现出优越性。更多详情请见此处：this https URL。 

---
# Anti-Degeneracy Scheme for Lidar SLAM based on Particle Filter in Geometry Feature-Less Environments 

**Title (ZH)**: 基于几何特征稀疏环境的粒子滤波抗退化方案激光雷达SLAM 

**Authors**: Yanbin Li, Wei Zhang, Zhiguo Zhang, Xiaogang Shi, Ziruo Li, Mingming Zhang, Hongping Xie, Wenzheng Chi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11486)  

**Abstract**: Simultaneous localization and mapping (SLAM) based on particle filtering has been extensively employed in indoor scenarios due to its high efficiency. However, in geometry feature-less scenes, the accuracy is severely reduced due to lack of constraints. In this article, we propose an anti-degeneracy system based on deep learning. Firstly, we design a scale-invariant linear mapping to convert coordinates in continuous space into discrete indexes, in which a data augmentation method based on Gaussian model is proposed to ensure the model performance by effectively mitigating the impact of changes in the number of particles on the feature distribution. Secondly, we develop a degeneracy detection model using residual neural networks (ResNet) and transformer which is able to identify degeneracy by scrutinizing the distribution of the particle population. Thirdly, an adaptive anti-degeneracy strategy is designed, which first performs fusion and perturbation on the resample process to provide rich and accurate initial values for the pose optimization, and use a hierarchical pose optimization combining coarse and fine matching, which is able to adaptively adjust the optimization frequency and the sensor trustworthiness according to the degree of degeneracy, in order to enhance the ability of searching the global optimal pose. Finally, we demonstrate the optimality of the model, as well as the improvement of the image matrix method and GPU on the computation time through ablation experiments, and verify the performance of the anti-degeneracy system in different scenarios through simulation experiments and real experiments. This work has been submitted to IEEE for publication. Copyright may be transferred without notice, after which this version may no longer be available. 

**Abstract (ZH)**: 基于深度学习的抗退化系统： particle filtering 方法在无几何特征场景中的同时定位与映射 

---
# Learning Dexterous Bimanual Catch Skills through Adversarial-Cooperative Heterogeneous-Agent Reinforcement Learning 

**Title (ZH)**: 通过对抗-协同异构代理强化学习学习灵巧的双臂接物技能 

**Authors**: Taewoo Kim, Youngwoo Yoon, Jaehong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11437)  

**Abstract**: Robotic catching has traditionally focused on single-handed systems, which are limited in their ability to handle larger or more complex objects. In contrast, bimanual catching offers significant potential for improved dexterity and object handling but introduces new challenges in coordination and control. In this paper, we propose a novel framework for learning dexterous bimanual catching skills using Heterogeneous-Agent Reinforcement Learning (HARL). Our approach introduces an adversarial reward scheme, where a throw agent increases the difficulty of throws-adjusting speed-while a catch agent learns to coordinate both hands to catch objects under these evolving conditions. We evaluate the framework in simulated environments using 15 different objects, demonstrating robustness and versatility in handling diverse objects. Our method achieved approximately a 2x increase in catching reward compared to single-agent baselines across 15 diverse objects. 

**Abstract (ZH)**: 基于异构代理强化学习的灵巧双臂接ONUS Classe 

---
# PrivilegedDreamer: Explicit Imagination of Privileged Information for Rapid Adaptation of Learned Policies 

**Title (ZH)**: 特权梦者：显式想象特权信息以实现快速适应学习策略 

**Authors**: Morgan Byrd, Jackson Crandell, Mili Das, Jessica Inman, Robert Wright, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2502.11377)  

**Abstract**: Numerous real-world control problems involve dynamics and objectives affected by unobservable hidden pa- rameters, ranging from autonomous driving to robotic manipu- lation, which cause performance degradation during sim-to-real transfer. To represent these kinds of domains, we adopt hidden- parameter Markov decision processes (HIP-MDPs), which model sequential decision problems where hidden variables parameterize transition and reward functions. Existing ap- proaches, such as domain randomization, domain adaptation, and meta-learning, simply treat the effect of hidden param- eters as additional variance and often struggle to effectively handle HIP-MDP problems, especially when the rewards are parameterized by hidden variables. We introduce Privileged- Dreamer, a model-based reinforcement learning framework that extends the existing model-based approach by incorporating an explicit parameter estimation module. PrivilegedDreamer features its novel dual recurrent architecture that explicitly estimates hidden parameters from limited historical data and enables us to condition the model, actor, and critic networks on these estimated parameters. Our empirical analysis on five diverse HIP-MDP tasks demonstrates that PrivilegedDreamer outperforms state-of-the-art model-based, model-free, and do- main adaptation learning algorithms. Additionally, we conduct ablation studies to justify the inclusion of each component in the proposed architecture. 

**Abstract (ZH)**: Numerous 实观控制问题涉及由不可观测的隐藏参数影响的动力学和目标，从自主驾驶到机器人 manipulation 等领域，在从仿真实验到现实应用的转接过程中会导致性能下降。为了表示这类领域，我们采用了隐藏参数马尔可夫决策过程(HIP-MDP)，该模型用于建模由隐藏变量参数化的转换和奖励函数的顺序决策问题。现有的方法，如领域随机化、领域适应和元学习，简单地将隐藏参数的影响视为额外的方差，通常难以有效处理 HIP-MDP 问题，尤其是在奖励由隐藏变量参数化的情况下。我们引入了 Privileged-Dreamer，这是一种基于模型的强化学习框架，该框架通过引入显式的参数估计模块扩展了现有的基于模型的方法。Privileged-Dreamer 具有独特的双递归架构，可以从有限的历史数据中显式估计隐藏参数，并使模型、演员和批评家网络能够基于这些估计参数进行条件化。我们在五个不同的 HIP-MDP 任务上的实证分析表明，Privileged-Dreamer 在基于模型、基于策略和领域适应学习算法中表现更优。此外，我们进行了消融研究以证明所提架构中每个组件的必要性。 

---
# Robot Deformable Object Manipulation via NMPC-generated Demonstrations in Deep Reinforcement Learning 

**Title (ZH)**: 基于NMPC生成的示范在深度强化学习中的机器人可变形物体操作 

**Authors**: Haoyuan Wang, Zihao Dong, Hongliang Lei, Zejia Zhang, Weizhuang Shi, Wei Luo, Weiwei Wan, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11375)  

**Abstract**: In this work, we conducted research on deformable object manipulation by robots based on demonstration-enhanced reinforcement learning (RL). To improve the learning efficiency of RL, we enhanced the utilization of demonstration data from multiple aspects and proposed the HGCR-DDPG algorithm. It uses a novel high-dimensional fuzzy approach for grasping-point selection, a refined behavior-cloning method to enhance data-driven learning in Rainbow-DDPG, and a sequential policy-learning strategy. Compared to the baseline algorithm (Rainbow-DDPG), our proposed HGCR-DDPG achieved 2.01 times the global average reward and reduced the global average standard deviation to 45% of that of the baseline algorithm. To reduce the human labor cost of demonstration collection, we proposed a low-cost demonstration collection method based on Nonlinear Model Predictive Control (NMPC). Simulation experiment results show that demonstrations collected through NMPC can be used to train HGCR-DDPG, achieving comparable results to those obtained with human demonstrations. To validate the feasibility of our proposed methods in real-world environments, we conducted physical experiments involving deformable object manipulation. We manipulated fabric to perform three tasks: diagonal folding, central axis folding, and flattening. The experimental results demonstrate that our proposed method achieved success rates of 83.3%, 80%, and 100% for these three tasks, respectively, validating the effectiveness of our approach. Compared to current large-model approaches for robot manipulation, the proposed algorithm is lightweight, requires fewer computational resources, and offers task-specific customization and efficient adaptability for specific tasks. 

**Abstract (ZH)**: 基于演示增强强化学习的变形物体操纵研究：HGCR-DDPG算法及其应用 

---
# HI-GVF: Shared Control based on Human-Influenced Guiding Vector Fields for Human-multi-robot Cooperation 

**Title (ZH)**: HI-GVF：基于人类影响引导向量场的 humano-多机器人协同控制 

**Authors**: Pengming Zhu, Zongtan Zhou, Weijia Yao, Wei Dai, Zhiwen Zeng, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11370)  

**Abstract**: Human-multi-robot shared control leverages human decision-making and robotic autonomy to enhance human-robot collaboration. While widely studied, existing systems often adopt a leader-follower model, limiting robot autonomy to some extent. Besides, a human is required to directly participate in the motion control of robots through teleoperation, which significantly burdens the operator. To alleviate these two issues, we propose a layered shared control computing framework using human-influenced guiding vector fields (HI-GVF) for human-robot collaboration. HI-GVF guides the multi-robot system along a desired path specified by the human. Then, an intention field is designed to merge the human and robot intentions, accelerating the propagation of the human intention within the multi-robot system. Moreover, we give the stability analysis of the proposed model and use collision avoidance based on safety barrier certificates to fine-tune the velocity. Eventually, considering the firefighting task as an example scenario, we conduct simulations and experiments using multiple human-robot interfaces (brain-computer interface, myoelectric wristband, eye-tracking), and the results demonstrate that our proposed approach boosts the effectiveness and performance of the task. 

**Abstract (ZH)**: 人类与多机器人共享控制结合人类决策与机器人自主性以增强人机协作 

---
# DFM: Deep Fourier Mimic for Expressive Dance Motion Learning 

**Title (ZH)**: DFM: 深度傅里叶模拟用于表达性舞蹈动作学习 

**Authors**: Ryo Watanabe, Chenhao Li, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2502.10980)  

**Abstract**: As entertainment robots gain popularity, the demand for natural and expressive motion, particularly in dancing, continues to rise. Traditionally, dancing motions have been manually designed by artists, a process that is both labor-intensive and restricted to simple motion playback, lacking the flexibility to incorporate additional tasks such as locomotion or gaze control during dancing. To overcome these challenges, we introduce Deep Fourier Mimic (DFM), a novel method that combines advanced motion representation with Reinforcement Learning (RL) to enable smooth transitions between motions while concurrently managing auxiliary tasks during dance sequences. While previous frequency domain based motion representations have successfully encoded dance motions into latent parameters, they often impose overly rigid periodic assumptions at the local level, resulting in reduced tracking accuracy and motion expressiveness, which is a critical aspect for entertainment robots. By relaxing these locally periodic constraints, our approach not only enhances tracking precision but also facilitates smooth transitions between different motions. Furthermore, the learned RL policy that supports simultaneous base activities, such as locomotion and gaze control, allows entertainment robots to engage more dynamically and interactively with users rather than merely replaying static, pre-designed dance routines. 

**Abstract (ZH)**: 随着娱乐机器人的流行，对自然且富有表现力的运动，尤其是舞蹈，的需求持续上升。传统上，舞蹈动作由艺术家手动设计，这一过程既劳动密集又受限于简单的运动回放，缺乏在跳舞过程中同时执行其他任务如行走或眼球控制的灵活性。为克服这些挑战，我们引入了深傅里叶拟合（Deep Fourier Mimic, DFM）这一新颖方法，该方法结合了高级动作表示与强化学习（Reinforcement Learning, RL），以实现在不同动作之间的平滑过渡，并同时管理舞蹈序列中的辅助任务。虽然基于频域的动作表示成功地将舞蹈动作编码为潜在参数，但在局部层面往往强加了过于严格的周期性假设，导致跟踪精度降低和运动表现力减弱，这是娱乐机器人的重要方面。通过放松这些局部周期性约束，我们的方法不仅能提高跟踪精度，还能促进不同动作之间的平滑过渡。此外，支持同时基础活动（如行走和眼球控制）的所学习的RL策略使娱乐机器人能够更动态地与用户互动，而不仅仅是回放静态的、预先设计的舞步。 

---
# Fine-Tuning Hard-to-Simulate Objectives for Quadruped Locomotion: A Case Study on Total Power Saving 

**Title (ZH)**: Fine-Tuning 难以模拟的目标以优化四足机器人运动：关于总能耗节省的案例研究 

**Authors**: Ruiqian Nai, Jiacheng You, Liu Cao, Hanchen Cui, Shiyuan Zhang, Huazhe Xu, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10956)  

**Abstract**: Legged locomotion is not just about mobility; it also encompasses crucial objectives such as energy efficiency, safety, and user experience, which are vital for real-world applications. However, key factors such as battery power consumption and stepping noise are often inaccurately modeled or missing in common simulators, leaving these aspects poorly optimized or unaddressed by current sim-to-real methods. Hand-designed proxies, such as mechanical power and foot contact forces, have been used to address these challenges but are often problem-specific and inaccurate.
In this paper, we propose a data-driven framework for fine-tuning locomotion policies, targeting these hard-to-simulate objectives. Our framework leverages real-world data to model these objectives and incorporates the learned model into simulation for policy improvement. We demonstrate the effectiveness of our framework on power saving for quadruped locomotion, achieving a significant 24-28\% net reduction in total power consumption from the battery pack at various speeds. In essence, our approach offers a versatile solution for optimizing hard-to-simulate objectives in quadruped locomotion, providing an easy-to-adapt paradigm for continual improving with real-world knowledge. Project page this https URL. 

**Abstract (ZH)**: 腿部运动不仅关乎移动性，还涵盖了能量效率、安全性和用户体验等至关重要的目标，这些目标对于实际应用至关重要。然而，关键因素如电池耗电和步态噪音在常用模拟器中常常被不准确地建模或缺失，导致现有从仿真到现实的方法在这些方面优化不足或未予解决。手工设计的代理指标，如机械功率和足部接触力，已被用于应对这些挑战，但这些指标通常具有特定问题和准确性不足的问题。

在本文中，我们提出了一种数据驱动框架，针对这些难以仿真优化的目标进行调整。该框架利用真实世界数据建模这些目标，并将所学模型集成到仿真中以改进策略。我们展示了该框架在四足行走节能方面的有效性，实现了在不同速度下电池总能耗显著减少24-28%。本质上，我们的方法提供了一种灵活的解决方案，用于优化四足行走中难以仿真的目标，并提供了一个易于适应的范式，可不断通过真实世界知识进行改进。项目页面: [这里](这个链接无法直接显示，请手动添加)。 

---
# Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation 

**Title (ZH)**: Sim-to-Real过渡在体育运动操控中的应用 

**Authors**: Nolan Fey, Gabriel B. Margolis, Martin Peticco, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2502.10894)  

**Abstract**: Achieving athletic loco-manipulation on robots requires moving beyond traditional tracking rewards - which simply guide the robot along a reference trajectory - to task rewards that drive truly dynamic, goal-oriented behaviors. Commands such as "throw the ball as far as you can" or "lift the weight as quickly as possible" compel the robot to exhibit the agility and power inherent in athletic performance. However, training solely with task rewards introduces two major challenges: these rewards are prone to exploitation (reward hacking), and the exploration process can lack sufficient direction. To address these issues, we propose a two-stage training pipeline. First, we introduce the Unsupervised Actuator Net (UAN), which leverages real-world data to bridge the sim-to-real gap for complex actuation mechanisms without requiring access to torque sensing. UAN mitigates reward hacking by ensuring that the learned behaviors remain robust and transferable. Second, we use a pre-training and fine-tuning strategy that leverages reference trajectories as initial hints to guide exploration. With these innovations, our robot athlete learns to lift, throw, and drag with remarkable fidelity from simulation to reality. 

**Abstract (ZH)**: 实现运动型机器人操控要求超越传统的跟踪奖励，转向驱动真正动态和目标导向行为的任务奖励。诸如“尽可能远地抛球”或“尽可能快速地举起重量”之类的命令促使机器人展现出类似运动员的敏捷性和力量。然而，仅使用任务奖励进行训练会引入两个主要挑战：这些奖励容易被操纵，且探索过程可能缺乏足够的方向。为了解决这些问题，我们提出了一种两阶段的训练管道。首先，我们引入了无监督执行网络（UAN），它利用现实世界数据来弥补复杂执行机制从仿真到现实的差距，无需扭矩感知。UAN通过确保学习到的行为保持稳健性和可-transferability来减少奖励操纵。其次，我们采用了一种预训练和微调策略，利用参考轨迹作为初始提示来引导探索。通过这些创新，我们的机器人运动员能够从仿真到现实以惊人的精度学会举举重、投掷和拖拽。 

---
# Accelerated co-design of robots through morphological pretraining 

**Title (ZH)**: 通过形态预训练加速机器人协设计 

**Authors**: Luke Strgar, Sam Kriegman  

**Link**: [PDF](https://arxiv.org/pdf/2502.10862)  

**Abstract**: The co-design of robot morphology and neural control typically requires using reinforcement learning to approximate a unique control policy gradient for each body plan, demanding massive amounts of training data to measure the performance of each design. Here we show that a universal, morphology-agnostic controller can be rapidly and directly obtained by gradient-based optimization through differentiable simulation. This process of morphological pretraining allows the designer to explore non-differentiable changes to a robot's physical layout (e.g. adding, removing and recombining discrete body parts) and immediately determine which revisions are beneficial and which are deleterious using the pretrained model. We term this process "zero-shot evolution" and compare it with the simultaneous co-optimization of a universal controller alongside an evolving design population. We find the latter results in diversity collapse, a previously unknown pathology whereby the population -- and thus the controller's training data -- converges to similar designs that are easier to steer with a shared universal controller. We show that zero-shot evolution with a pretrained controller quickly yields a diversity of highly performant designs, and by fine-tuning the pretrained controller on the current population throughout evolution, diversity is not only preserved but significantly increased as superior performance is achieved. 

**Abstract (ZH)**: 通过可微模拟的基于梯度的优化实现机器人形态与神经控制的共设计：零样本进化 

---
# Motion planning for highly-dynamic unconditioned reflexes based on chained Signed Distance Functions 

**Title (ZH)**: 基于链式符号距离函数的高动态无条件反射运动规划 

**Authors**: Ken Lin, Qi Ye, Tin Lun Lam, Zhibin Li, Jiming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10734)  

**Abstract**: The unconditioned reflex (e.g., protective reflex), which is the innate reaction of the organism and usually performed through the spinal cord rather than the brain, can enable organisms to escape harms from environments. In this paper, we propose an online, highly-dynamic motion planning algorithm to endow manipulators the highly-dynamic unconditioned reflexes to humans and/or environments. Our method is based on a chained version of Signed Distance Functions (SDFs), which can be pre-computed and stored. Our proposed algorithm is divided into two stages. In the offline stage, we create 3 groups of local SDFs to store the geometric information of the manipulator and its working environment. In the online stage, the pre-computed local SDFs are chained together according the configuration of the manipulator, to provide global geometric information about the environment. While the point clouds of the dynamic objects serve as query points to look up these local SDFs for quickly generating escape velocity. Then we propose a modified geometric Jacobian matrix and use the Jacobian-pseudo-inverse method to generate real-time reflex behaviors to avoid the static and dynamic obstacles in the environment. The benefits of our method are validated in both static and dynamic scenarios. In the static scenario, our method identifies the path solutions with lower time consumption and shorter trajectory length compared to existing solutions. In the dynamic scenario, our method can reliably pursue the dynamic target point, avoid dynamic obstacles, and react to these obstacles within 1ms, which surpasses the unconditioned reflex reaction time of humans. 

**Abstract (ZH)**: 在线高度动态运动规划算法赋予 manipulator 类似人体的保护性反射能力 

---
# Reachability-Aware Reinforcement Learning for Collision Avoidance in Human-Machine Shared Control 

**Title (ZH)**: 可达性意识强化学习在人机共控避碰中的应用 

**Authors**: Shiyue Zhao, Junzhi Zhang, Neda Masoud, Jianxiong Li, Yinan Zheng, Xiaohui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2502.10610)  

**Abstract**: Human-machine shared control in critical collision scenarios aims to aid drivers' accident avoidance through intervening only when necessary. Existing methods count on replanning collision-free trajectories and imposing human-machine tracking, which usually interrupts the driver's intent and increases the risk of conflict. Additionally, the lack of guaranteed trajectory feasibility under extreme conditions can compromise safety and reliability. This paper introduces a Reachability-Aware Reinforcement Learning framework for shared control, guided by Hamilton-Jacobi (HJ) reachability analysis. Machine intervention is activated only when the vehicle approaches the Collision Avoidance Reachable Set (CARS), which represents states where collision is unavoidable. First, we precompute the reachability distributions and the CARS by solving the Bellman equation using offline data. To reduce human-machine conflicts, we develop a driver model for sudden obstacles and propose an authority allocation strategy considering key collision avoidance features. Finally, we train a reinforcement learning agent to reduce human-machine conflicts while enforcing the hard constraint of avoiding entry into the CARS. The proposed method was tested on a real vehicle platform. Results show that the controller intervenes effectively near CARS to prevent collisions while maintaining improved original driving task performance. Robustness analysis further supports its flexibility across different driver attributes. 

**Abstract (ZH)**: 基于可达性感知强化学习的关键碰撞场景人机共决策HEME 

---
# VLP: Vision-Language Preference Learning for Embodied Manipulation 

**Title (ZH)**: 视觉-语言偏好学习在嵌入式操作中的应用 

**Authors**: Runze Liu, Chenjia Bai, Jiafei Lyu, Shengjie Sun, Yali Du, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.11918)  

**Abstract**: Reward engineering is one of the key challenges in Reinforcement Learning (RL). Preference-based RL effectively addresses this issue by learning from human feedback. However, it is both time-consuming and expensive to collect human preference labels. In this paper, we propose a novel \textbf{V}ision-\textbf{L}anguage \textbf{P}reference learning framework, named \textbf{VLP}, which learns a vision-language preference model to provide preference feedback for embodied manipulation tasks. To achieve this, we define three types of language-conditioned preferences and construct a vision-language preference dataset, which contains versatile implicit preference orders without human annotations. The preference model learns to extract language-related features, and then serves as a preference annotator in various downstream tasks. The policy can be learned according to the annotated preferences via reward learning or direct policy optimization. Extensive empirical results on simulated embodied manipulation tasks demonstrate that our method provides accurate preferences and generalizes to unseen tasks and unseen language instructions, outperforming the baselines by a large margin. 

**Abstract (ZH)**: 基于视觉-语言的偏好学习框架VLP 

---
# Does Knowledge About Perceptual Uncertainty Help an Agent in Automated Driving? 

**Title (ZH)**: 知 perception 不确定性知识是否有助于自动驾驶代理？ 

**Authors**: Natalie Grabowsky, Annika Mütze, Joshua Wendland, Nils Jansen, Matthias Rottmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.11864)  

**Abstract**: Agents in real-world scenarios like automated driving deal with uncertainty in their environment, in particular due to perceptual uncertainty. Although, reinforcement learning is dedicated to autonomous decision-making under uncertainty these algorithms are typically not informed about the uncertainty currently contained in their environment. On the other hand, uncertainty estimation for perception itself is typically directly evaluated in the perception domain, e.g., in terms of false positive detection rates or calibration errors based on camera images. Its use for deciding on goal-oriented actions remains largely unstudied. In this paper, we investigate how an agent's behavior is influenced by an uncertain perception and how this behavior changes if information about this uncertainty is available. Therefore, we consider a proxy task, where the agent is rewarded for driving a route as fast as possible without colliding with other road users. For controlled experiments, we introduce uncertainty in the observation space by perturbing the perception of the given agent while informing the latter. Our experiments show that an unreliable observation space modeled by a perturbed perception leads to a defensive driving behavior of the agent. Furthermore, when adding the information about the current uncertainty directly to the observation space, the agent adapts to the specific situation and in general accomplishes its task faster while, at the same time, accounting for risks. 

**Abstract (ZH)**: 基于感知不确定性代理在实际场景中的行为研究：从不确定感知到风险考量的变化 

---
# Can you pass that tool?: Implications of Indirect Speech in Physical Human-Robot Collaboration 

**Title (ZH)**: 你能传递工具吗？间接 speech 在物理人机协作中的 implications 

**Authors**: Yan Zhang, Tharaka Sachintha Ratnayake, Cherie Sew, Jarrod Knibbe, Jorge Goncalves, Wafa Johal  

**Link**: [PDF](https://arxiv.org/pdf/2502.11720)  

**Abstract**: Indirect speech acts (ISAs) are a natural pragmatic feature of human communication, allowing requests to be conveyed implicitly while maintaining subtlety and flexibility. Although advancements in speech recognition have enabled natural language interactions with robots through direct, explicit commands--providing clarity in communication--the rise of large language models presents the potential for robots to interpret ISAs. However, empirical evidence on the effects of ISAs on human-robot collaboration (HRC) remains limited. To address this, we conducted a Wizard-of-Oz study (N=36), engaging a participant and a robot in collaborative physical tasks. Our findings indicate that robots capable of understanding ISAs significantly improve human's perceived robot anthropomorphism, team performance, and trust. However, the effectiveness of ISAs is task- and context-dependent, thus requiring careful use. These results highlight the importance of appropriately integrating direct and indirect requests in HRC to enhance collaborative experiences and task performance. 

**Abstract (ZH)**: 间接言语行为（ISAs）是人类沟通中的自然语用特征，允许通过隐含的方式传达请求，同时保持沟通的微妙性和灵活性。尽管语音识别的进步使得机器人能够通过直接、明确的命令与人类进行自然语言交互，从而提高沟通的清晰度，但大规模语言模型的发展为机器人理解和处理ISAs提供了可能。然而，关于ISAs对人类-机器人合作（HRC）影响的实证证据仍有限。为了解决这一问题，我们进行了一个知情者扮演研究（N=36），让参与者与机器人共同完成物理协作任务。研究结果表明，能够理解ISAs的机器人显著提高了参与者对机器人的拟人化感知、团队表现和信任度。然而，ISAs的有效性依赖于任务和具体情境，因此需要谨慎使用。这些结果强调了在HRC中适当整合直接和间接请求的重要性，以提升合作体验和任务表现。 

---
# Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning 

**Title (ZH)**: 记忆、基准与机器人：基于强化学习解决复杂任务的基准 

**Authors**: Egor Cherepanov, Nikita Kachaev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2502.10550)  

**Abstract**: Memory is crucial for enabling agents to tackle complex tasks with temporal and spatial dependencies. While many reinforcement learning (RL) algorithms incorporate memory, the field lacks a universal benchmark to assess an agent's memory capabilities across diverse scenarios. This gap is particularly evident in tabletop robotic manipulation, where memory is essential for solving tasks with partial observability and ensuring robust performance, yet no standardized benchmarks exist. To address this, we introduce MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a comprehensive benchmark for memory RL, with three key contributions: (1) we propose a comprehensive classification framework for memory-intensive RL tasks, (2) we collect MIKASA-Base - a unified benchmark that enables systematic evaluation of memory-enhanced agents across diverse scenarios, and (3) we develop MIKASA-Robo - a novel benchmark of 32 carefully designed memory-intensive tasks that assess memory capabilities in tabletop robotic manipulation. Our contributions establish a unified framework for advancing memory RL research, driving the development of more reliable systems for real-world applications. The code is available at this https URL. 

**Abstract (ZH)**: 记忆对于使智能体能够应对具有时间性和空间性依赖的复杂任务至关重要。虽然许多强化学习（RL）算法都包含了记忆机制，但该领域缺乏一个通用基准来评估智能体在不同场景中的记忆能力。这一差距在桌面机器人操作中尤为明显，因为在这种场景中，记忆对于解决部分可观测任务和确保稳健性能是必不可少的，但尚未存在标准化的基准测试。为解决这一问题，我们引入了MIKASA（Memory-Intensive Skills Assessment Suite for Agents），这是一个全面的记忆RL基准测试，其三大贡献为：（1）提出了一种全面的记忆密集型RL任务分类框架；（2）收集了MIKASA-Base - 一个统一的基准测试，可系统评估增强记忆能力的智能体在多种场景中的表现；（3）开发了MIKASA-Robo，这是一个新型的包含32个精心设计的记忆密集型任务的基准测试，用于评估桌面机器人操作中的记忆能力。我们的贡献建立了一个统一的框架，加速了记忆RL研究的进步，推动了更可靠系统在实际应用中的发展。代码可在以下链接获取：this https URL。 

---
# Learning to be Smooth: An End-to-End Differentiable Particle Smoother 

**Title (ZH)**: 学习平滑：端到端可微分粒子平滑器 

**Authors**: Ali Younis, Erik B. Sudderth  

**Link**: [PDF](https://arxiv.org/pdf/2502.10546)  

**Abstract**: For challenging state estimation problems arising in domains like vision and robotics, particle-based representations attractively enable temporal reasoning about multiple posterior modes. Particle smoothers offer the potential for more accurate offline data analysis by propagating information both forward and backward in time, but have classically required human-engineered dynamics and observation models. Extending recent advances in discriminative training of particle filters, we develop a framework for low-variance propagation of gradients across long time sequences when training particle smoothers. Our "two-filter'' smoother integrates particle streams that are propagated forward and backward in time, while incorporating stratification and importance weights in the resampling step to provide low-variance gradient estimates for neural network dynamics and observation models. The resulting mixture density particle smoother is substantially more accurate than state-of-the-art particle filters, as well as search-based baselines, for city-scale global vehicle localization from real-world videos and maps. 

**Abstract (ZH)**: 基于粒子的表示方法在视觉和机器人学等领域中解决具有挑战性的状态估计问题时，能够吸引人地实现对多个后验模式的时域推理。粒子平滑器通过在时间的正反两个方向传播信息，为更精准的离线数据分析提供了潜力，但传统上需要手工构建的动力学和观测模型。结合近年来粒子滤波的判别训练进展，我们开发了一种框架，用于在训练粒子平滑器时沿长时间序列进行低方差梯度传播。我们的“两滤波器”平滑器整合了沿时间正反方向传播的粒子流，并在重采样步骤中采用分层和重要性加权，以提供神经网络动力学和观测模型的低方差梯度估计。由此得到的混合密度粒子平滑器在从真实世界视频和地图中进行大规模车辆全局定位方面，比现有的粒子滤波器和基于搜索的基线更为准确。 

---
# Deep Reinforcement Learning-Based User Scheduling for Collaborative Perception 

**Title (ZH)**: 基于深度强化学习的用户调度方法研究——面向协作感知 

**Authors**: Yandi Liu, Guowei Liu, Le Liang, Hao Ye, Chongtao Guo, Shi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10456)  

**Abstract**: Stand-alone perception systems in autonomous driving suffer from limited sensing ranges and occlusions at extended distances, potentially resulting in catastrophic outcomes. To address this issue, collaborative perception is envisioned to improve perceptual accuracy by using vehicle-to-everything (V2X) communication to enable collaboration among connected and autonomous vehicles and roadside units. However, due to limited communication resources, it is impractical for all units to transmit sensing data such as point clouds or high-definition video. As a result, it is essential to optimize the scheduling of communication links to ensure efficient spectrum utilization for the exchange of perceptual data. In this work, we propose a deep reinforcement learning-based V2X user scheduling algorithm for collaborative perception. Given the challenges in acquiring perceptual labels, we reformulate the conventional label-dependent objective into a label-free goal, based on characteristics of 3D object detection. Incorporating both channel state information (CSI) and semantic information, we develop a double deep Q-Network (DDQN)-based user scheduling framework for collaborative perception, named SchedCP. Simulation results verify the effectiveness and robustness of SchedCP compared with traditional V2X scheduling methods. Finally, we present a case study to illustrate how our proposed algorithm adaptively modifies the scheduling decisions by taking both instantaneous CSI and perceptual semantics into account. 

**Abstract (ZH)**: 基于深度强化学习的协作感知V2X用户调度算法 

---
# Real Time Control of Tandem-Wing Experimental Platform Using Concerto Reinforcement Learning 

**Title (ZH)**: 使用Concerto强化学习的 tandem-wing 实验平台实时控制 

**Authors**: Zhang Minghao, Yang Xiaojun, Wang Zhihe, Wang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10429)  

**Abstract**: This paper introduces the CRL2RT algorithm, an advanced reinforcement learning method aimed at improving the real-time control performance of the Direct-Drive Tandem-Wing Experimental Platform (DDTWEP). Inspired by dragonfly flight, DDTWEP's tandem wing structure causes nonlinear and unsteady aerodynamic interactions, leading to complex load behaviors during pitch, roll, and yaw maneuvers. These complexities challenge stable motion control at high frequencies (2000 Hz). To overcome these issues, we developed the CRL2RT algorithm, which combines classical control elements with reinforcement learning-based controllers using a time-interleaved architecture and a rule-based policy composer. This integration ensures finite-time convergence and single-life adaptability. Experimental results under various conditions, including different flapping frequencies and yaw disturbances, show that CRL2RT achieves a control frequency surpassing 2500 Hz on standard CPUs. Additionally, when integrated with classical controllers like PID, Adaptive PID, and Model Reference Adaptive Control (MRAC), CRL2RT enhances tracking performance by 18.3% to 60.7%. These findings demonstrate CRL2RT's broad applicability and superior performance in complex real-time control scenarios, validating its effectiveness in overcoming existing control strategy limitations and advancing robust, efficient real-time control for biomimetic aerial vehicles. 

**Abstract (ZH)**: 本文介绍了CRL2RT算法，这是一种旨在提高直接驱动串联翼实验平台（DDTWEP）实时控制性能的先进强化学习方法。受蜻蜓飞行启发，DDTWEP的串联翼结构导致了非线性和不稳定的气动相互作用，使得在滚转、俯仰和偏航机动过程中出现复杂的载荷行为。这些复杂性对高频（2000 Hz）下的稳定运动控制构成了挑战。为了克服这些问题，我们开发了CRL2RT算法，该算法结合了经典的控制元素和基于强化学习的控制器，采用时间交错架构和基于规则的策略合成器。这种集成确保了有限时间收敛和单寿命适应性。在不同 CONDITIONS（包括不同的拍翼频率和偏航干扰）下的实验结果表明，CRL2RT在标准CPU上实现了超过2500 Hz的控制频率。此外，当与PID、自适应PID和模型参考自适应控制（MRAC）等经典控制器集成时，CRL2RT能提高跟踪性能18.3%至60.7%。这些发现展示了CRL2RT在复杂实时控制场景中的广泛应用性和优越性能，验证了其在克服现有控制策略局限性、推进仿生飞行器鲁棒高效实时控制方面的作用。 

---
# A Study on Leveraging Search and Self-Feedback for Agent Reasoning 

**Title (ZH)**: 基于搜索和自我反馈的代理推理研究 

**Authors**: Karthikeyan K, Michelle Yuan, Elman Mansimov, Katerina Margatina, Anurag Pratik, Daniele Bonadiman, Monica Sunkara, Yi Zhang, Yassine Benajiba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12094)  

**Abstract**: Recent works have demonstrated that incorporating search during inference can significantly improve reasoning capabilities of language agents. Some approaches may make use of the ground truth or rely on model's own generated feedback. The search algorithm uses this feedback to then produce values that will update its criterion for exploring and exploiting various reasoning paths. In this study, we investigate how search and model's self-feedback can be leveraged for reasoning tasks. First, we explore differences in ground-truth feedback and self-feedback during search for math reasoning. Second, we observe limitations in applying search techniques to more complex tasks like tool-calling and design domain-specific approaches to address these gaps. Our experiments reveal challenges related to generalization when solely relying on self-feedback during search. For search to work effectively, either access to the ground-truth is needed or feedback mechanisms need to be carefully designed for the specific task. 

**Abstract (ZH)**: Recent works have demonstrated that incorporating search during inference can significantly improve reasoning capabilities of language agents. Some approaches may make use of the ground truth or rely on model's own generated feedback. The search algorithm uses this feedback to then produce values that will update its criterion for exploring and exploiting various reasoning paths. In this study, we investigate how search and model's self-feedback can be leveraged for reasoning tasks. First, we explore differences in ground-truth feedback and self-feedback during search for math reasoning. Second, we observe limitations in applying search techniques to more complex tasks like tool-calling and design domain-specific approaches to address these gaps. Our experiments reveal challenges related to generalization when solely relying on self-feedback during search. For search to work effectively, either access to the ground-truth is needed or feedback mechanisms need to be carefully designed for the specific task. 

---
# Unlocking the Potential of Generative AI through Neuro-Symbolic Architectures: Benefits and Limitations 

**Title (ZH)**: 通过神经符号架构解锁生成式AI的潜力：优势与局限性 

**Authors**: Oualid Bougzime, Samir Jabbar, Christophe Cruz, Frédéric Demoly  

**Link**: [PDF](https://arxiv.org/pdf/2502.11269)  

**Abstract**: Neuro-symbolic artificial intelligence (NSAI) represents a transformative approach in artificial intelligence (AI) by combining deep learning's ability to handle large-scale and unstructured data with the structured reasoning of symbolic methods. By leveraging their complementary strengths, NSAI enhances generalization, reasoning, and scalability while addressing key challenges such as transparency and data efficiency. This paper systematically studies diverse NSAI architectures, highlighting their unique approaches to integrating neural and symbolic components. It examines the alignment of contemporary AI techniques such as retrieval-augmented generation, graph neural networks, reinforcement learning, and multi-agent systems with NSAI paradigms. This study then evaluates these architectures against comprehensive set of criteria, including generalization, reasoning capabilities, transferability, and interpretability, therefore providing a comparative analysis of their respective strengths and limitations. Notably, the Neuro > Symbolic < Neuro model consistently outperforms its counterparts across all evaluation metrics. This result aligns with state-of-the-art research that highlight the efficacy of such architectures in harnessing advanced technologies like multi-agent systems. 

**Abstract (ZH)**: 神经符号人工智能：结合深度学习和符号方法的变革性approach及其评价 

---
# NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM 

**Title (ZH)**: NavRAG: 通过检索增强的大语言模型生成用户需求指令以实现具身导航 

**Authors**: Zihan Wang, Yaohui Zhu, Gim Hee Lee, Yachun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11142)  

**Abstract**: Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models. 

**Abstract (ZH)**: 视觉-语言导航（VLN）是具身代理的一项基本技能，使其能够遵循自然语言指令在3D环境中导航。高性能导航模型需要大量的训练数据，人工标注数据的高昂成本严重阻碍了这一领域的发展。因此，一些先前的方法将轨迹视频转换为分步指令以扩展数据，但这些指令与用户简要描述目的地或表达特定需求的沟通方式不够匹配。此外，局部导航轨迹忽视了全局上下文和高层任务规划。为了应对这些问题，我们提出了NavRAG，这是一种检索增强生成（RAG）框架，用于为VLN生成用户需求指令。NavRAG 利用大型语言模型（LLM）构建从全局布局到局部细节的层次场景描述树，然后模拟具有特定需求的各种用户角色，从场景树中检索生成多样化指令。我们对861个场景中的超过200万条导航指令进行了注解，并评估了训练模型的数据质量和导航性能。 

---
# The Philosophical Foundations of Growing AI Like A Child 

**Title (ZH)**: Growing AI Like A Child：其哲学基础 

**Authors**: Dezhi Luo, Yijiang Li, Hokin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10742)  

**Abstract**: Despite excelling in high-level reasoning, current language models lack robustness in real-world scenarios and perform poorly on fundamental problem-solving tasks that are intuitive to humans. This paper argues that both challenges stem from a core discrepancy between human and machine cognitive development. While both systems rely on increasing representational power, the absence of core knowledge-foundational cognitive structures in humans-prevents language models from developing robust, generalizable abilities, where complex skills are grounded in simpler ones within their respective domains. It explores empirical evidence of core knowledge in humans, analyzes why language models fail to acquire it, and argues that this limitation is not an inherent architectural constraint. Finally, it outlines a workable proposal for systematically integrating core knowledge into future multi-modal language models through the large-scale generation of synthetic training data using a cognitive prototyping strategy. 

**Abstract (ZH)**: 尽管在高层次推理方面表现出色，当前的语言模型在现实世界场景中缺乏 robustness，在基本的问题解决任务上表现不佳，而这些任务对人类来说是直观的。本文认为，这些挑战都源自人类和机器认知发展核心差异。尽管两种系统都依赖于增加表示能力，但人类缺乏核心知识——基础认知结构，这阻碍了语言模型发展出基于各自领域内更简单技能的稳健且可泛化的技能。本文探讨了人类核心知识的经验证据，分析了语言模型为何未能获得这些知识，并认为这一局限性不是架构上的固有条件。最后，本文提出了一个可行的方案，通过大规模生成合成训练数据来系统地将核心知识整合到未来的多模态语言模型中，采用认知原型策略。 

---
# USER-VLM 360: Personalized Vision Language Models with User-aware Tuning for Social Human-Robot Interactions 

**Title (ZH)**: USER-VLM 360：面向社交人机交互的具有用户意识调整的个性化视觉语言模型 

**Authors**: Hamed Rahimi, Adil Bahaj, Mouad Abrini, Mahdi Khoramshahi, Mounir Ghogho, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2502.10636)  

**Abstract**: The integration of vision-language models into robotic systems constitutes a significant advancement in enabling machines to interact with their surroundings in a more intuitive manner. While VLMs offer rich multimodal reasoning, existing approaches lack user-specific adaptability, often relying on generic interaction paradigms that fail to account for individual behavioral, contextual, or socio-emotional nuances. When customization is attempted, ethical concerns arise from unmitigated biases in user data, risking exclusion or unfair treatment. To address these dual challenges, we propose User-VLM 360°, a holistic framework integrating multimodal user modeling with bias-aware optimization. Our approach features: (1) user-aware tuning that adapts interactions in real time using visual-linguistic signals; (2) bias mitigation via preference optimization; and (3) curated 360° socio-emotive interaction datasets annotated with demographic, emotion, and relational metadata. Evaluations across eight benchmarks demonstrate state-of-the-art results: +35.3% F1 in personalized VQA, +47.5% F1 in facial features understanding, 15% bias reduction, and 30X speedup over baselines. Ablation studies confirm component efficacy, and deployment on the Pepper robot validates real-time adaptability across diverse users. We open-source parameter-efficient 3B/10B models and an ethical verification framework for responsible adaptation. 

**Abstract (ZH)**: 用户导向的全方位视觉语言模型集成框架：结合多元用户建模与偏见意识优化 

---
# Diverse Transformer Decoding for Offline Reinforcement Learning Using Financial Algorithmic Approaches 

**Title (ZH)**: 使用金融算法方法的离线强化学习多样化变压器解码 

**Authors**: Dan Elbaz, Oren Salzman  

**Link**: [PDF](https://arxiv.org/pdf/2502.10473)  

**Abstract**: Offline Reinforcement Learning (RL) algorithms learn a policy using a fixed training dataset, which is then deployed online to interact with the environment and make decisions. Transformers, a standard choice for modeling time-series data, are gaining popularity in offline RL. In this context, Beam Search (BS), an approximate inference algorithm, is the go-to decoding method. Offline RL eliminates the need for costly or risky online data collection. However, the restricted dataset induces uncertainty as the agent may encounter unfamiliar sequences of states and actions during execution that were not covered in the training data. In this context, BS lacks two important properties essential for offline RL: It does not account for the aforementioned uncertainty, and its greedy left-right search approach often results in sequences with minimal variations, failing to explore potentially better alternatives.
To address these limitations, we propose Portfolio Beam Search (PBS), a simple-yet-effective alternative to BS that balances exploration and exploitation within a Transformer model during decoding. We draw inspiration from financial economics and apply these principles to develop an uncertainty-aware diversification mechanism, which we integrate into a sequential decoding algorithm at inference time. We empirically demonstrate the effectiveness of PBS on the D4RL locomotion benchmark, where it achieves higher returns and significantly reduces outcome variability. 

**Abstract (ZH)**: 基于Transformer的离线强化学习束搜索算法：Portfolio Beam Search 

---
# AI Alignment at Your Discretion 

**Title (ZH)**: AI对齐由你做主 

**Authors**: Maarten Buyl, Hadi Khalaf, Claudio Mayrink Verdun, Lucas Monteiro Paes, Caio C. Vieira Machado, Flavio du Pin Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2502.10441)  

**Abstract**: In AI alignment, extensive latitude must be granted to annotators, either human or algorithmic, to judge which model outputs are `better' or `safer.' We refer to this latitude as alignment discretion. Such discretion remains largely unexamined, posing two risks: (i) annotators may use their power of discretion arbitrarily, and (ii) models may fail to mimic this discretion. To study this phenomenon, we draw on legal concepts of discretion that structure how decision-making authority is conferred and exercised, particularly in cases where principles conflict or their application is unclear or irrelevant. Extended to AI alignment, discretion is required when alignment principles and rules are (inevitably) conflicting or indecisive. We present a set of metrics to systematically analyze when and how discretion in AI alignment is exercised, such that both risks (i) and (ii) can be observed. Moreover, we distinguish between human and algorithmic discretion and analyze the discrepancy between them. By measuring both human and algorithmic discretion over safety alignment datasets, we reveal layers of discretion in the alignment process that were previously unaccounted for. Furthermore, we demonstrate how algorithms trained on these datasets develop their own forms of discretion in interpreting and applying these principles, which challenges the purpose of having any principles at all. Our paper presents the first step towards formalizing this core gap in current alignment processes, and we call on the community to further scrutinize and control alignment discretion. 

**Abstract (ZH)**: 在AI对齐中，必须给予标注者（无论是人类还是算法）广泛的裁量权，以判断哪些模型输出是“更好”或“更安全”的。我们称这种裁量权为对齐裁量。这种裁量权目前尚未得到充分研究，存在两大风险：（i）标注者可能随意使用其裁量权；（ii）模型可能无法模仿这种裁量。为研究这一现象，我们借鉴法律中的裁量权概念，该概念规定了决策权如何授予和行使，尤其是在原则冲突或其应用不明确或不相关的情况下。延伸至AI对齐，当对齐原则和规则不可避免地存在冲突或不明确时，就需要这种裁量权。我们提出了一套指标，以系统分析AI对齐中何时及如何行使裁量权，从而使上述两种风险得以观察。此外，我们区分了人类和算法裁量，并分析了它们之间的差异。通过测量人类和算法在安全对齐数据集上的裁量权，我们揭示了对齐过程中此前未被考虑的多层裁量机制。此外，我们展示了这些数据集上的算法如何发展出它们自己的裁量形式，以解读和应用这些原则，这挑战了制定任何原则的初衷。我们的论文代表了朝着正式化当前对齐过程中的这一核心缺口迈出的第一步，并呼吁社区进一步审视和控制对齐裁量。 

---
# Agency in Artificial Intelligence Systems 

**Title (ZH)**: 人工智能系统的代理权 

**Authors**: Parashar Das  

**Link**: [PDF](https://arxiv.org/pdf/2502.10434)  

**Abstract**: There is a general concern that present developments in artificial intelligence (AI) research will lead to sentient AI systems, and these may pose an existential threat to humanity. But why cannot sentient AI systems benefit humanity instead? This paper endeavours to put this question in a tractable manner. I ask whether a putative AI system will develop an altruistic or a malicious disposition towards our society, or what would be the nature of its agency? Given that AI systems are being developed into formidable problem solvers, we can reasonably expect these systems to preferentially take on conscious aspects of human problem solving. I identify the relevant phenomenal aspects of agency in human problem solving. The functional aspects of conscious agency can be monitored using tools provided by functionalist theories of consciousness. A recent expert report (Butlin et al. 2023) has identified functionalist indicators of agency based on these theories. I show how to use the Integrated Information Theory (IIT) of consciousness, to monitor the phenomenal nature of this agency. If we are able to monitor the agency of AI systems as they develop, then we can dissuade them from becoming a menace to society while encouraging them to be an aid. 

**Abstract (ZH)**: 人工智能研究的发展可能导致有感知能力的AI系统出现，这些系统可能对人类构成生存威胁，但它们能否反而惠及人类？本文试图将这一问题具体化。本文探讨一个假设的AI系统将倾向于对社会产生利他主义还是恶意倾向，或者它的agency的本质是什么？考虑到AI系统正被开发成为强大的问题解决者，我们有理由预期这些系统将优先采用人类问题解决过程中的意识方面。本文识别了人类问题解决过程中的相关现象学方面的agency。功能主义意识理论提供的工具可以用于监测意识agency的功能方面。最近的一份专家报告（Butlin等人，2023）基于这些理论识别了agency的功能性指标。本文展示如何利用综合信息理论（IIT）来监测这种agency的现象学性质。如果我们能够监测AI系统在发展过程中的agency，那么我们就能阻止它们对社会构成威胁，同时鼓励它们成为一种助力。 

---
# Stonefish: Supporting Machine Learning Research in Marine Robotics 

**Title (ZH)**: 石鱼：支持海洋机器人领域机器学习研究 

**Authors**: Michele Grimaldi, Patryk Cieslak, Eduardo Ochoa, Vibhav Bharti, Hayat Rajani, Ignacio Carlucho, Maria Koskinopoulou, Yvan R. Petillot, Nuno Gracias  

**Link**: [PDF](https://arxiv.org/pdf/2502.11887)  

**Abstract**: Simulations are highly valuable in marine robotics, offering a cost-effective and controlled environment for testing in the challenging conditions of underwater and surface operations. Given the high costs and logistical difficulties of real-world trials, simulators capable of capturing the operational conditions of subsea environments have become key in developing and refining algorithms for remotely-operated and autonomous underwater vehicles. This paper highlights recent enhancements to the Stonefish simulator, an advanced open-source platform supporting development and testing of marine robotics solutions. Key updates include a suite of additional sensors, such as an event-based camera, a thermal camera, and an optical flow camera, as well as, visual light communication, support for tethered operations, improved thruster modelling, more flexible hydrodynamics, and enhanced sonar accuracy. These developments and an automated annotation tool significantly bolster Stonefish's role in marine robotics research, especially in the field of machine learning, where training data with a known ground truth is hard or impossible to collect. 

**Abstract (ZH)**: 模拟技术在海洋机器人研究中极为宝贵，提供了一种在海上和水面操作的苛刻环境下进行低成本且可控测试的环境。鉴于真实世界实验的高成本和 logistical 困难，能够捕捉海底环境操作条件的模拟器已成为开发和改进遥控和自主 underwater 车辆算法的关键工具。本文强调了对 Stonefish 模拟器的近期改进，这是一个先进的开源平台，支持海洋机器人解决方案的开发和测试。关键更新包括一系列额外的传感器，如事件驱动的相机、热像仪和光学流相机，以及视觉光通信、系缆操作支持、推进器模型改进、更灵活的水动力学以及声纳精度增强。这些改进和自动标注工具显著增强了 Stonefish 在海洋机器人研究中的作用，特别是在机器学习领域，因为此类标签数据的收集往往是困难的或不可能的。 

---
# Intuitive physics understanding emerges from self-supervised pretraining on natural videos 

**Title (ZH)**: 直觉物理理解源自自然视频的自我监督预训练 

**Authors**: Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, Yann LeCun  

**Link**: [PDF](https://arxiv.org/pdf/2502.11831)  

**Abstract**: We investigate the emergence of intuitive physics understanding in general-purpose deep neural network models trained to predict masked regions in natural videos. Leveraging the violation-of-expectation framework, we find that video prediction models trained to predict outcomes in a learned representation space demonstrate an understanding of various intuitive physics properties, such as object permanence and shape consistency. In contrast, video prediction in pixel space and multimodal large language models, which reason through text, achieve performance closer to chance. Our comparisons of these architectures reveal that jointly learning an abstract representation space while predicting missing parts of sensory input, akin to predictive coding, is sufficient to acquire an understanding of intuitive physics, and that even models trained on one week of unique video achieve above chance performance. This challenges the idea that core knowledge -- a set of innate systems to help understand the world -- needs to be hardwired to develop an understanding of intuitive physics. 

**Abstract (ZH)**: 我们在训练用于预测自然视频中遮蔽区域的一般深度神经网络模型中探讨直观物理理解的涌现。利用违反预期框架，我们发现，在学习表示空间中训练以预测结果的视频预测模型展示了对各种直观物理特性的理解，如物体恒在性和形状一致性。相比之下，在像素空间中进行视频预测以及通过文本进行推理的多模态大规模语言模型的表现接近随机水平。我们的架构比较表明，同时学习一个抽象表示空间并预测感觉输入的缺失部分，类似于预测编码，足以获得直观物理的理解，并且即使在仅训练一周的独特视频数据下，模型也能达到超过随机水平的表现。这挑战了核心知识——一套有助于理解世界的内置系统——需要硬编码才能发展出直观物理理解的观点。 

---
# Maximum Entropy Reinforcement Learning with Diffusion Policy 

**Title (ZH)**: 最大熵强化学习与扩散策略 

**Authors**: Xiaoyi Dong, Jian Cheng, Xi Sheryl Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11612)  

**Abstract**: The Soft Actor-Critic (SAC) algorithm with a Gaussian policy has become a mainstream implementation for realizing the Maximum Entropy Reinforcement Learning (MaxEnt RL) objective, which incorporates entropy maximization to encourage exploration and enhance policy robustness. While the Gaussian policy performs well on simpler tasks, its exploration capacity and potential performance in complex multi-goal RL environments are limited by its inherent unimodality. In this paper, we employ the diffusion model, a powerful generative model capable of capturing complex multimodal distributions, as the policy representation to fulfill the MaxEnt RL objective, developing a method named MaxEnt RL with Diffusion Policy (MaxEntDP). Our method enables efficient exploration and brings the policy closer to the optimal MaxEnt policy. Experimental results on Mujoco benchmarks show that MaxEntDP outperforms the Gaussian policy and other generative models within the MaxEnt RL framework, and performs comparably to other state-of-the-art diffusion-based online RL algorithms. Our code is available at this https URL. 

**Abstract (ZH)**: 基于扩散模型的MaxEnt RL方法（MaxEntDP） 

---
# Leader and Follower: Interactive Motion Generation under Trajectory Constraints 

**Title (ZH)**: 领导者与跟随者：基于轨迹约束的交互运动生成 

**Authors**: Runqi Wang, Caoyuan Ma, Jian Zhao, Hanrui Xu, Dongfang Sun, Haoyang Chen, Lin Xiong, Zheng Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.11563)  

**Abstract**: With the rapid advancement of game and film production, generating interactive motion from texts has garnered significant attention due to its potential to revolutionize content creation processes. In many practical applications, there is a need to impose strict constraints on the motion range or trajectory of virtual characters. However, existing methods that rely solely on textual input face substantial challenges in accurately capturing the user's intent, particularly in specifying the desired trajectory. As a result, the generated motions often lack plausibility and accuracy. Moreover, existing trajectory - based methods for customized motion generation rely on retraining for single - actor scenarios, which limits flexibility and adaptability to different datasets, as well as interactivity in two-actor motions. To generate interactive motion following specified trajectories, this paper decouples complex motion into a Leader - Follower dynamic, inspired by role allocation in partner dancing. Based on this framework, this paper explores the motion range refinement process in interactive motion generation and proposes a training-free approach, integrating a Pace Controller and a Kinematic Synchronization Adapter. The framework enhances the ability of existing models to generate motion that adheres to trajectory by controlling the leader's movement and correcting the follower's motion to align with the leader. Experimental results show that the proposed approach, by better leveraging trajectory information, outperforms existing methods in both realism and accuracy. 

**Abstract (ZH)**: 随着游戏和影视制作的迅速发展，从文本生成交互式运动引起了广泛关注，因其有望革命性地改变内容创作过程。在许多实际应用中，需要对虚拟角色的运动范围或轨迹施加严格约束。然而，现有仅依赖文本输入的方法在准确捕捉用户意图方面面临巨大挑战，特别是在指定期望轨迹时。因此，生成的运动往往缺乏合理性与精确性。此外，现有的基于轨迹的定制运动生成方法依赖于单人演员场景的重新训练，这限制了对不同数据集的灵活性和适应性，以及双人运动的交互性。为了根据指定的轨迹生成交互式运动，本文借鉴伴侣舞蹈中的角色分配，将复杂运动分解为领导者-跟随者动态。基于此框架，本文探索了交互式运动生成中的运动范围精炼过程，并提出了一种无需训练的方法，结合了节奏控制器和运动同步适配器。该框架通过控制领导者运动并纠正跟随者的运动以与领导者对齐，增强了现有模型生成符合轨迹运动的能力。实验结果表明，通过更好地利用轨迹信息，所提出的方法在逼真度和准确性方面优于现有方法。 

---
# $\text{M}^{\text{3}}$: A Modular World Model over Streams of Tokens 

**Title (ZH)**: $\text{M}^{\text{3}}$: 一种基于令牌流的模块化世界模型 

**Authors**: Lior Cohen, Kaixin Wang, Bingyi Kang, Uri Gadot, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2502.11537)  

**Abstract**: Token-based world models emerged as a promising modular framework, modeling dynamics over token streams while optimizing tokenization separately. While successful in visual environments with discrete actions (e.g., Atari games), their broader applicability remains uncertain. In this paper, we introduce $\text{M}^{\text{3}}$, a $\textbf{m}$odular $\textbf{w}$orld $\textbf{m}$odel that extends this framework, enabling flexible combinations of observation and action modalities through independent modality-specific components. $\text{M}^{\text{3}}$ integrates several improvements from existing literature to enhance agent performance. Through extensive empirical evaluation across diverse benchmarks, $\text{M}^{\text{3}}$ achieves state-of-the-art sample efficiency for planning-free world models. Notably, among these methods, it is the first to reach a human-level median score on Atari 100K, with superhuman performance on 13 games. We $\href{this https URL}{\text{open-source our code and weights}}$. 

**Abstract (ZH)**: 基于Token的世界模型作为一种有前景的模块化框架涌现出来，能够分别优化token化并建模token流中的动力学过程。尽管在具有离散动作的视觉环境中（例如Atari游戏）取得了成功，但其更广泛的适用性尚不确定。本文介绍了$\text{M}^{\text{3}}$，这是一种模块化世界模型，扩展了这一框架，通过独立的模态特定组件实现观察和动作模态的灵活组合。$\text{M}^{\text{3}}$整合了现有文献中的多项改进以提升代理性能。通过在多种基准上的广泛实证评估，$\text{M}^{\text{3}}$实现了无规划世界模型的样本效率新基准。特别地，这是首个在Atari 100K上达到人类级中位分并具有超人类性能的13款游戏的方法。我们开源了我们的代码和权重。 

---
# Generative Multi-Agent Collaboration in Embodied AI: A Systematic Review 

**Title (ZH)**: 基于体态人工智能的生成式多Agent协作：一项系统性综述 

**Authors**: Di Wu, Xian Wei, Guang Chen, Hao Shen, Xiangfeng Wang, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11518)  

**Abstract**: Embodied multi-agent systems (EMAS) have attracted growing attention for their potential to address complex, real-world challenges in areas such as logistics and robotics. Recent advances in foundation models pave the way for generative agents capable of richer communication and adaptive problem-solving. This survey provides a systematic examination of how EMAS can benefit from these generative capabilities. We propose a taxonomy that categorizes EMAS by system architectures and embodiment modalities, emphasizing how collaboration spans both physical and virtual contexts. Central building blocks, perception, planning, communication, and feedback, are then analyzed to illustrate how generative techniques bolster system robustness and flexibility. Through concrete examples, we demonstrate the transformative effects of integrating foundation models into embodied, multi-agent frameworks. Finally, we discuss challenges and future directions, underlining the significant promise of EMAS to reshape the landscape of AI-driven collaboration. 

**Abstract (ZH)**: 具身多智能体系统(EMAS)因其在物流和机器人等领域解决复杂现实挑战的潜力而受到越来越多的关注。基础模型的最新进展为生成式智能体进行更丰富的通信和适应性问题解决铺平了道路。本文综述了具身多智能体系统如何从这些生成式能力中受益。我们提出了一种分类法，按照系统架构和具身模态对EMAS进行分类，强调了协作如何跨越物理和虚拟环境。随后，分析了核心构建模块、感知、规划、通信和反馈，以说明生成技术如何增强系统的稳健性和灵活性。通过具体实例，展示了将基础模型整合到具身多智能体框架中的变革性影响。最后，我们讨论了挑战和未来方向，强调了EMAS在重塑以人工智能驱动的合作领域方面的巨大潜力。 

---
# Without Paired Labeled Data: An End-to-End Self-Supervised Paradigm for UAV-View Geo-Localization 

**Title (ZH)**: 无需配对标注数据：面向UAV视角地理定位的端到端自监督范式 

**Authors**: Zhongwei Chen, Zhao-Xu Yang, Hai-Jun Rong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11381)  

**Abstract**: UAV-View Geo-Localization (UVGL) aims to ascertain the precise location of a UAV by retrieving the most similar GPS-tagged satellite image. However, existing methods predominantly rely on supervised learning paradigms that necessitate annotated paired data for training, which incurs substantial annotation costs and impedes large-scale deployment. To overcome this limitation, we propose the Dynamic Memory-Driven and Neighborhood Information Learning (DMNIL) network, a lightweight end-to-end self-supervised framework for UAV-view geo-localization. The DMNIL framework utilizes a dual-path clustering-based contrastive learning architecture as its baseline to model intra-view structural relationships, enhancing feature consistency and discriminability. Additionally, a dynamic memory-driven hierarchical learning module is proposed to progressively mine local and global information, reinforcing multi-level feature associations to improve model robustness. To bridge the domain gap between UAV and satellite views, we design an information-consistent evolutionary learning mechanism that systematically explores latent correlations within intra-view neighborhoods and across cross-view domains, ultimately constructing a unified cross-view feature representation space. Extensive experiments on three benchmarks (University-1652, SUES-200, and DenseUAV) demonstrate that DMNIL achieves competitive performance against state-of-the-art supervised methods while maintaining computational efficiency. Notably, this superiority is attained without relying on paired training data, underscoring the framework's practicality for real-world deployment. Codes will be released soon. 

**Abstract (ZH)**: 基于无人机视角的动态记忆驱动和邻域信息学习地理定位（DMNIL） 

---
# Cognitive Neural Architecture Search Reveals Hierarchical Entailment 

**Title (ZH)**: 认知神经架构搜索揭示层次蕴含关系 

**Authors**: Lukas Kuhn, Sari Saba-Sadiya, Gemma Roig  

**Link**: [PDF](https://arxiv.org/pdf/2502.11141)  

**Abstract**: Recent research has suggested that the brain is more shallow than previously thought, challenging the traditionally assumed hierarchical structure of the ventral visual pathway. Here, we demonstrate that optimizing convolutional network architectures for brain-alignment via evolutionary neural architecture search results in models with clear representational hierarchies. Despite having random weights, the identified models achieve brain-alignment scores surpassing even those of pretrained classification models - as measured by both regression and representational similarity analysis. Furthermore, through traditional supervised training, architectures optimized for alignment with late ventral regions become competitive classification models. These findings suggest that hierarchical structure is a fundamental mechanism of primate visual processing. Finally, this work demonstrates the potential of neural architecture search as a framework for computational cognitive neuroscience research that could reduce the field's reliance on manually designed convolutional networks. 

**Abstract (ZH)**: 近期的研究表明，大脑的层次结构可能比以前认为的要浅，挑战了传统上假设的腹侧视觉通路的层次结构。在这里，我们证明，通过进化神经架构搜索优化卷积网络架构以实现与大脑的对齐，可以产生具有明确表示层次结构的模型。尽管这些模型的权重是随机的，但它们的脑对齐分数甚至超过了预训练分类模型的分数，通过回归和表示相似性分析进行衡量。此外，通过传统的监督训练，优化与腹侧后期区域对齐的架构可以成为有竞争力的分类模型。这些发现表明，层次结构是灵长类动物视觉处理的基本机制。最后，本研究展示了进化神经架构搜索作为计算认知神经科学研究框架的潜力，该框架可以减少该领域对手动设计卷积网络的依赖。 

---
# A Physics-Informed Machine Learning Framework for Safe and Optimal Control of Autonomous Systems 

**Title (ZH)**: 基于物理知识的机器学习框架：用于自主系统安全与最优控制 

**Authors**: Manan Tayal, Aditya Singh, Shishir Kolathaya, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.11057)  

**Abstract**: As autonomous systems become more ubiquitous in daily life, ensuring high performance with guaranteed safety is crucial. However, safety and performance could be competing objectives, which makes their co-optimization difficult. Learning-based methods, such as Constrained Reinforcement Learning (CRL), achieve strong performance but lack formal safety guarantees due to safety being enforced as soft constraints, limiting their use in safety-critical settings. Conversely, formal methods such as Hamilton-Jacobi (HJ) Reachability Analysis and Control Barrier Functions (CBFs) provide rigorous safety assurances but often neglect performance, resulting in overly conservative controllers. To bridge this gap, we formulate the co-optimization of safety and performance as a state-constrained optimal control problem, where performance objectives are encoded via a cost function and safety requirements are imposed as state constraints. We demonstrate that the resultant value function satisfies a Hamilton-Jacobi-Bellman (HJB) equation, which we approximate efficiently using a novel physics-informed machine learning framework. In addition, we introduce a conformal prediction-based verification strategy to quantify the learning errors, recovering a high-confidence safety value function, along with a probabilistic error bound on performance degradation. Through several case studies, we demonstrate the efficacy of the proposed framework in enabling scalable learning of safe and performant controllers for complex, high-dimensional autonomous systems. 

**Abstract (ZH)**: 随着自主系统在日常生活中的普及，确保在有保证的安全性下的高性能至关重要。然而，安全性和性能可能是相互竞争的目标，这使得它们的共同优化变得困难。基于学习的方法，如受限强化学习（CRL），可以获得强大的性能，但由于安全要求是以软约束的形式施加的，缺乏形式上的安全保证，限制了它们在安全关键设置中的应用。相反，形式化方法，如哈密尔顿-雅可比（HJ）可达性分析和控制屏障函数（CBFs），可以提供严格的安全保证，但通常会忽略性能，导致过于保守的控制器。为了弥合这一差距，我们将安全性和性能的共同优化形式化为状态约束最优控制问题，其中性能目标通过成本函数编码，安全性要求作为状态约束施加。我们证明了所得的价值函数满足哈密尔顿-雅可比-贝尔曼（HJB）方程，并使用一个新颖的物理知识嵌入机器学习框架对其进行有效近似。此外，我们引入了一种符合性预测为基础的验证策略来量化学习误差，恢复高可信度的安全价值函数，同时提供性能退化的一个概率误差边界。通过几个案例研究，我们展示了所提出框架的有效性，能够在复杂、高维自主系统中实现可扩展的学习安全且性能良好的控制器。 

---
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model 

**Title (ZH)**: 读取你的heartbeat：通过预训练的心电语言模型学习心电图单词和句子 

**Authors**: Jiarui Jin, Haoyu Wang, Hongyan Li, Jun Li, Jiahui Pan, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.10707)  

**Abstract**: Electrocardiogram (ECG) is essential for the clinical diagnosis of arrhythmias and other heart diseases, but deep learning methods based on ECG often face limitations due to the need for high-quality annotations. Although previous ECG self-supervised learning (eSSL) methods have made significant progress in representation learning from unannotated ECG data, they typically treat ECG signals as ordinary time-series data, segmenting the signals using fixed-size and fixed-step time windows, which often ignore the form and rhythm characteristics and latent semantic relationships in ECG signals. In this work, we introduce a novel perspective on ECG signals, treating heartbeats as words and rhythms as sentences. Based on this perspective, we first designed the QRS-Tokenizer, which generates semantically meaningful ECG sentences from the raw ECG signals. Building on these, we then propose HeartLang, a novel self-supervised learning framework for ECG language processing, learning general representations at form and rhythm levels. Additionally, we construct the largest heartbeat-based ECG vocabulary to date, which will further advance the development of ECG language processing. We evaluated HeartLang across six public ECG datasets, where it demonstrated robust competitiveness against other eSSL methods. Our data and code are publicly available at this https URL. 

**Abstract (ZH)**: 心电图（ECG）对于心律失常和其他心脏疾病的临床诊断至关重要，但由于需要高质量的注释，基于ECG的深度学习方法常常面临限制。尽管之前的心电图自我监督学习（eSSL）方法在无标注ECG数据的表征学习方面取得了显著进展，但它们通常将ECG信号视为普通的时序数据，使用固定大小和固定步长的时间窗口对信号进行分割，这往往忽略了ECG信号中的形态、节奏特征及其潜在语义关系。在本工作中，我们从一个新的视角来审视ECG信号，将心跳视为“词”，节奏视为“句子”。基于这一视角，我们首先设计了QRS-Tokenizing模块，该模块从原始ECG信号中生成语义上有意义的ECG句子。在此基础上，我们提出了HeartLang，一种新颖的心电图自我监督学习框架，用于ECG语言处理，学习形态和节奏层面的通用表征。此外，我们构建了迄今为止最大的基于心跳的心电图词汇表，这将进一步促进心电图语言处理的发展。我们在六个公开的心电图数据集中评估了HeartLang，其表现出色，与其它eSSL方法具有较强的竞争力。我们的数据和代码可在以下网址获取：this https URL。 

---
# GenComUI: Exploring Generative Visual Aids as Medium to Support Task-Oriented Human-Robot Communication 

**Title (ZH)**: GenComUI: 探索生成型视觉辅助作为支持任务导向人机通信的媒介 

**Authors**: Yate Ge, Meiying Li, Xipeng Huang, Yuanda Hu, Qi Wang, Xiaohua Sun, Weiwei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10678)  

**Abstract**: This work investigates the integration of generative visual aids in human-robot task communication. We developed GenComUI, a system powered by large language models that dynamically generates contextual visual aids (such as map annotations, path indicators, and animations) to support verbal task communication and facilitate the generation of customized task programs for the robot. This system was informed by a formative study that examined how humans use external visual tools to assist verbal communication in spatial tasks. To evaluate its effectiveness, we conducted a user experiment (n = 20) comparing GenComUI with a voice-only baseline. The results demonstrate that generative visual aids, through both qualitative and quantitative analysis, enhance verbal task communication by providing continuous visual feedback, thus promoting natural and effective human-robot communication. Additionally, the study offers a set of design implications, emphasizing how dynamically generated visual aids can serve as an effective communication medium in human-robot interaction. These findings underscore the potential of generative visual aids to inform the design of more intuitive and effective human-robot communication, particularly for complex communication scenarios in human-robot interaction and LLM-based end-user development. 

**Abstract (ZH)**: 本研究探讨了生成性视觉辅助在人类与机器人任务通信中的整合。我们开发了由大规模语言模型驱动的GenComUI系统，该系统能够动态生成上下文视觉辅助（如地图标注、路径指示和动画），以支持口头任务通信并为机器人生成定制化任务程序提供便利。该系统基于一项形成性研究的指导，该研究探讨了人类如何使用外部视觉工具来辅助空间任务中的口头通信。为了评估其 effectiveness，我们进行了一个包含20名用户的用户实验，将GenComUI与仅语音基线进行比较。研究结果表明，生成性视觉辅助通过定性和定量分析提供了持续的视觉反馈，从而促进了自然且有效的的人机通信。此外，本研究还提供了一组设计启示，强调动态生成的视觉辅助在人机交互中的有效通信媒介作用。这些发现强调了生成性视觉辅助在设计更具直观性和有效性的交互式人机通信方面的潜力，特别是在人机交互和基于LLM的用户开发中的复杂通信场景。 

---
# Leveraging Constraint Violation Signals For Action-Constrained Reinforcement Learning 

**Title (ZH)**: 利用约束违反信号进行行动受限强化学习 

**Authors**: Janaka Chathuranga Brahmanage, Jiajing Ling, Akshat Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.10431)  

**Abstract**: In many RL applications, ensuring an agent's actions adhere to constraints is crucial for safety. Most previous methods in Action-Constrained Reinforcement Learning (ACRL) employ a projection layer after the policy network to correct the action. However projection-based methods suffer from issues like the zero gradient problem and higher runtime due to the usage of optimization solvers. Recently methods were proposed to train generative models to learn a differentiable mapping between latent variables and feasible actions to address this issue. However, generative models require training using samples from the constrained action space, which itself is challenging. To address such limitations, first, we define a target distribution for feasible actions based on constraint violation signals, and train normalizing flows by minimizing the KL divergence between an approximated distribution over feasible actions and the target. This eliminates the need to generate feasible action samples, greatly simplifying the flow model learning. Second, we integrate the learned flow model with existing deep RL methods, which restrict it to exploring only the feasible action space. Third, we extend our approach beyond ACRL to handle state-wise constraints by learning the constraint violation signal from the environment. Empirically, our approach has significantly fewer constraint violations while achieving similar or better quality in several control tasks than previous best methods. 

**Abstract (ZH)**: 基于约束的强化学习中确保智能体行动符合约束的正常化流方法 

---
