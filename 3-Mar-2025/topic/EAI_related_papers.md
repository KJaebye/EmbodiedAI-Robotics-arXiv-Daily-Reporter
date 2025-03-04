# RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete 

**Title (ZH)**: RoboBrain: 从抽象到具体的机器人操作统一脑模型 

**Authors**: Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, Xinda Xue, Qinghang Su, Huaihai Lyu, Xiaolong Zheng, Jiaming Liu, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21257)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various multimodal contexts. However, their application in robotic scenarios, particularly for long-horizon manipulation tasks, reveals significant limitations. These limitations arise from the current MLLMs lacking three essential robotic brain capabilities: Planning Capability, which involves decomposing complex manipulation instructions into manageable sub-tasks; Affordance Perception, the ability to recognize and interpret the affordances of interactive objects; and Trajectory Prediction, the foresight to anticipate the complete manipulation trajectory necessary for successful execution. To enhance the robotic brain's core capabilities from abstract to concrete, we introduce ShareRobot, a high-quality heterogeneous dataset that labels multi-dimensional information such as task planning, object affordance, and end-effector trajectory. ShareRobot's diversity and accuracy have been meticulously refined by three human annotators. Building on this dataset, we developed RoboBrain, an MLLM-based model that combines robotic and general multi-modal data, utilizes a multi-stage training strategy, and incorporates long videos and high-resolution images to improve its robotic manipulation capabilities. Extensive experiments demonstrate that RoboBrain achieves state-of-the-art performance across various robotic tasks, highlighting its potential to advance robotic brain capabilities. 

**Abstract (ZH)**: 近期多模态大型语言模型（MLLMs）在各种多模态场景中的应用显示出了显著的能力，但在机器人应用场景中，特别是对于远期操作任务，其应用却暴露出显著的局限性。这些局限性源于当前MLLMs缺乏三种关键的机器人脑能力：规划能力（将复杂的操作指令分解为可管理的子任务）、可利用性感知（识别和解释交互对象的可利用性）以及轨迹预测（预见完成操作所需的完整轨迹）。为了使机器人脑的能力从抽象层面提升到具体层面，我们介绍了ShareRobot，这是一个高质量的异构数据集，标注了任务规划、物体可利用性和末端执行器轨迹等多维度信息。ShareRobot的多样性和准确性经过三位人工标注者的精细调整。基于这一数据集，我们开发了RoboBrain，这是一种结合了机器人和通用多模态数据的MLLM模型，采用多阶段训练策略，并结合长视频和高分辨率图像以提高其操作能力。广泛的实验表明，RoboBrain在各种机器人任务中达到了最先进的性能，突显了其提升机器人脑能力的潜力。 

---
# Vibrotactile information coding strategies for a body-worn vest to aid robot-human collaboration 

**Title (ZH)**: 基于穿戴式背心的振动触觉信息编码策略以辅助机器人-人类协作 

**Authors**: Adrian Vecina Tercero, Praminda Caleb-Solly  

**Link**: [PDF](https://arxiv.org/pdf/2502.21056)  

**Abstract**: This paper explores the use of a body-worn vibrotactile vest to convey real-time information from robot to operator. Vibrotactile communication could be useful in providing information without compropmising or loading a person's visual or auditory perception. This paper considers applications in Urban Search and Rescue (USAR) scenarios where a human working alongside a robot is likely to be operating in high cognitive load conditions. The focus is on understanding how best to convey information considering different vibrotactile information coding strategies to enhance scene understanding in scenarios where a robot might be operating remotely as a scout. In exploring information representation, this paper introduces Semantic Haptics, using shapes and patterns to represent certain events as if the skin was a screen, and shows how these lead to bettter learnability and interpreation accuracy. 

**Abstract (ZH)**: 本文探讨了使用穿戴式震动触觉背心来实时传输机器人到操作员的信息。触觉通信在无需牺牲或负担人员的视觉或听觉感知的情况下提供信息可能非常有用。本文考虑了在城市搜救（USAR）场景中的应用，在这些场景中，与机器人并肩工作的人员可能处于高认知负荷条件。重点是理解如何最好地传递信息，以考虑不同的触觉信息编码策略，以增强在机器人可能作为侦察员远程操作的情况下场景理解。在探索信息表示时，本文引入了语义触觉概念，使用形状和模式表示特定事件，仿佛皮肤是一个屏幕，并展示了这些方法如何提高学习能力和解释准确性。 

---
# Motion ReTouch: Motion Modification Using Four-Channel Bilateral Control 

**Title (ZH)**: 运动精修：基于四通道双边控制的运动修改 

**Authors**: Koki Inami, Sho Sakaino, Toshiaki Tsuji  

**Link**: [PDF](https://arxiv.org/pdf/2502.20982)  

**Abstract**: Recent research has demonstrated the usefulness of imitation learning in autonomous robot operation. In particular, teaching using four-channel bilateral control, which can obtain position and force information, has been proven effective. However, control performance that can easily execute high-speed, complex tasks in one go has not yet been achieved. We propose a method called Motion ReTouch, which retroactively modifies motion data obtained using four-channel bilateral control. The proposed method enables modification of not only position but also force information. This was achieved by the combination of multilateral control and motion-copying system. The proposed method was verified in experiments with a real robot, and the success rate of the test tube transfer task was improved, demonstrating the possibility of modification force information. 

**Abstract (ZH)**: Recent Research Demonstrates the Utility of Imitation Learning in Autonomous Robot Operation: Motion ReTouch method Retroactively Modifies Motion Data with Position and Force Information 

---
# DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping 

**Title (ZH)**: DexGraspVLA: 一种面向通用灵巧操作的视觉-语言-行动框架 

**Authors**: Yifan Zhong, Xuchuan Huang, Ruochong Li, Ceyao Zhang, Yitao Liang, Yaodong Yang, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20900)  

**Abstract**: Dexterous grasping remains a fundamental yet challenging problem in robotics. A general-purpose robot must be capable of grasping diverse objects in arbitrary scenarios. However, existing research typically relies on specific assumptions, such as single-object settings or limited environments, leading to constrained generalization. Our solution is DexGraspVLA, a hierarchical framework that utilizes a pre-trained Vision-Language model as the high-level task planner and learns a diffusion-based policy as the low-level Action controller. The key insight lies in iteratively transforming diverse language and visual inputs into domain-invariant representations, where imitation learning can be effectively applied due to the alleviation of domain shift. Thus, it enables robust generalization across a wide range of real-world scenarios. Notably, our method achieves a 90+% success rate under thousands of unseen object, lighting, and background combinations in a ``zero-shot'' environment. Empirical analysis further confirms the consistency of internal model behavior across environmental variations, thereby validating our design and explaining its generalization performance. We hope our work can be a step forward in achieving general dexterous grasping. Our demo and code can be found at this https URL. 

**Abstract (ZH)**: 灵巧抓取仍然是机器人技术中的一个基本而具挑战性的问题。通用机器人必须能够在任意场景中抓取多样化物体。然而，现有研究通常依赖于特定假设，如单一物体设置或有限环境，导致泛化能力受限。我们的解决方案是DexGraspVLA，这是一种层次框架，利用预训练的Vision-Language模型作为高层次任务规划器，并学习基于扩散的策略作为低层动作控制器。关键洞察在于迭代地将多样化语言和视觉输入转化为领域不变的表示，从而缓解域偏移，使得模仿学习可以有效应用。因此，它能够在广泛的实际场景中实现稳健的泛化。值得注意的是，我们的方法在“零样本”环境中，在成千上万种未见过的物体、光照和背景组合中实现了90%以上的成功率。实证分析进一步证实了模型内部行为在环境变化中的一致性，从而验证了我们的设计并解释了其泛化性能。我们希望我们的工作能够朝着实现通用灵巧抓取迈出一步。我们的演示和代码可在以下链接找到：this https URL。 

---
# Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments 

**Title (ZH)**: 非抓握式 manipulation 在通用环境中的分层与模块化网络 

**Authors**: Yoonyoung Cho, Junhyek Han, Jisu Han, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.20843)  

**Abstract**: For robots to operate in general environments like households, they must be able to perform non-prehensile manipulation actions such as toppling and rolling to manipulate ungraspable objects. However, prior works on non-prehensile manipulation cannot yet generalize across environments with diverse geometries. The main challenge lies in adapting to varying environmental constraints: within a cabinet, the robot must avoid walls and ceilings; to lift objects to the top of a step, the robot must account for the step's pose and extent. While deep reinforcement learning (RL) has demonstrated impressive success in non-prehensile manipulation, accounting for such variability presents a challenge for the generalist policy, as it must learn diverse strategies for each new combination of constraints. To address this, we propose a modular and reconfigurable architecture that adaptively reconfigures network modules based on task requirements. To capture the geometric variability in environments, we extend the contact-based object representation (CORN) to environment geometries, and propose a procedural algorithm for generating diverse environments to train our agent. Taken together, the resulting policy can zero-shot transfer to novel real-world environments and objects despite training entirely within a simulator. We additionally release a simulation-based benchmark featuring nine digital twins of real-world scenes with 353 objects to facilitate non-prehensile manipulation research in realistic domains. 

**Abstract (ZH)**: 用于在家庭等通用环境中操作的机器人必须能够执行非挟持操作，如推倒和滚动以操纵不可握住的对象。然而，当前的非挟持操作研究尚不能在具有不同几何形状的环境中泛化。主要挑战在于适应不断变化的环境约束：在柜子内，机器人必须避开墙壁和天花板；要抬起物体放到台阶顶部，机器人必须考虑台阶的姿态和范围。尽管深度强化学习（RL）已经在非挟持操作方面展示了令人印象深刻的成果，但考虑到这种变化性，通用策略的学习会面临挑战，因为它必须为每种新的约束组合学习不同的策略。为了解决这个问题，我们提出了一种模块化且可重构的架构，该架构根据任务需求自适应地重构网络模块。为了捕捉环境中的几何变化，我们扩展了基于接触的对象表示（CORN）以适应环境几何形状，并提出了生成多样化环境的方法来训练我们的代理。总体而言，最终产生的策略能够在完全在模拟器中训练的情况下，零样本迁移至新型现实世界环境和对象。此外，我们还发布了基于模拟器的基准测试，其中包括九个真实世界场景的数字孪生体和353个对象，以促进在真实场景中进行非挟持操作的研究。 

---
# CSubBT: A Self-Adjusting Execution Framework for Mobile Manipulation System 

**Title (ZH)**: CSubBT: 一种移动操作系统的自适应执行框架 

**Authors**: Huihui Guo, Huizhang Luo, Huilong Pi, Mingxing Duan, Kenli Li, Chubo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20771)  

**Abstract**: With the advancements in modern intelligent technologies, mobile robots equipped with manipulators are increasingly operating in unstructured environments. These robots can plan sequences of actions for long-horizon tasks based on perceived information. However, in practice, the planned actions often fail due to discrepancies between the perceptual information used for planning and the actual conditions. In this paper, we introduce the {\itshape Conditional Subtree} (CSubBT), a general self-adjusting execution framework for mobile manipulation tasks based on Behavior Trees (BTs). CSubBT decomposes symbolic action into sub-actions and uses BTs to control their execution, addressing any potential anomalies during the process. CSubBT treats common anomalies as constraint non-satisfaction problems and continuously guides the robot in performing tasks by sampling new action parameters in the constraint space when anomalies are detected. We demonstrate the robustness of our framework through extensive manipulation experiments on different platforms, both in simulation and real-world settings. 

**Abstract (ZH)**: 基于行为树的条件子树：移动操纵任务的自适应执行框架 

---
# A2DO: Adaptive Anti-Degradation Odometry with Deep Multi-Sensor Fusion for Autonomous Navigation 

**Title (ZH)**: 自适应抗退化里程计：基于深度多传感器融合的自主导航 

**Authors**: Hui Lai, Qi Chen, Junping Zhang, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20767)  

**Abstract**: Accurate localization is essential for the safe and effective navigation of autonomous vehicles, and Simultaneous Localization and Mapping (SLAM) is a cornerstone technology in this context. However, The performance of the SLAM system can deteriorate under challenging conditions such as low light, adverse weather, or obstructions due to sensor degradation. We present A2DO, a novel end-to-end multi-sensor fusion odometry system that enhances robustness in these scenarios through deep neural networks. A2DO integrates LiDAR and visual data, employing a multi-layer, multi-scale feature encoding module augmented by an attention mechanism to mitigate sensor degradation dynamically. The system is pre-trained extensively on simulated datasets covering a broad range of degradation scenarios and fine-tuned on a curated set of real-world data, ensuring robust adaptation to complex scenarios. Our experiments demonstrate that A2DO maintains superior localization accuracy and robustness across various degradation conditions, showcasing its potential for practical implementation in autonomous vehicle systems. 

**Abstract (ZH)**: 准确的定位对于自主车辆的安全有效导航至关重要，而同时定位与建图（SLAM）是这一过程中的核心技术。然而，在低光照、恶劣天气或传感器退化等具有挑战性条件下，SLAM系统的性能可能会下降。我们提出了一种名为A2DO的新型端到端多传感器融合里程计系统，通过深度神经网络增强在这些场景下的鲁棒性。A2DO结合了LiDAR和视觉数据，通过具有注意机制的多层多尺度特征编码模块动态减轻传感器退化的影响。该系统在覆盖广泛退化场景的模拟数据集上进行广泛的预训练，并在精心收集的真实数据集上进行微调，以确保其在复杂场景中的鲁棒适应性。我们的实验结果表明，A2DO能够在各种退化条件下保持卓越的定位准确性和鲁棒性，展示了其在自主车辆系统中的实际应用潜力。 

---
# FSMP: A Frontier-Sampling-Mixed Planner for Fast Autonomous Exploration of Complex and Large 3-D Environments 

**Title (ZH)**: FSMP：一种前沿采样混合规划器，用于快速探索复杂大型3D环境 

**Authors**: Shiyong Zhang, Xuebo Zhang, Qianli Dong, Ziyu Wang, Haobo Xi, Jing Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.20707)  

**Abstract**: In this paper, we propose a systematic framework for fast exploration of complex and large 3-D environments using micro aerial vehicles (MAVs). The key insight is the organic integration of the frontier-based and sampling-based strategies that can achieve rapid global exploration of the environment. Specifically, a field-of-view-based (FOV) frontier detector with the guarantee of completeness and soundness is devised for identifying 3-D map frontiers. Different from random sampling-based methods, the deterministic sampling technique is employed to build and maintain an incremental road map based on the recorded sensor FOVs and newly detected frontiers. With the resulting road map, we propose a two-stage path planner. First, it quickly computes the global optimal exploration path on the road map using the lazy evaluation strategy. Then, the best exploration path is smoothed for further improving the exploration efficiency. We validate the proposed method both in simulation and real-world experiments. The comparative results demonstrate the promising performance of our planner in terms of exploration efficiency, computational time, and explored volume. 

**Abstract (ZH)**: 本文提出了一种用于快速探索复杂大型3D环境的微型 aerial车辆（MAVs）系统框架。 

---
# Subtask-Aware Visual Reward Learning from Segmented Demonstrations 

**Title (ZH)**: 基于分割示例的子任务意识视觉奖励学习 

**Authors**: Changyeon Kim, Minho Heo, Doohyun Lee, Jinwoo Shin, Honglak Lee, Joseph J. Lim, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.20630)  

**Abstract**: Reinforcement Learning (RL) agents have demonstrated their potential across various robotic tasks. However, they still heavily rely on human-engineered reward functions, requiring extensive trial-and-error and access to target behavior information, often unavailable in real-world settings. This paper introduces REDS: REward learning from Demonstration with Segmentations, a novel reward learning framework that leverages action-free videos with minimal supervision. Specifically, REDS employs video demonstrations segmented into subtasks from diverse sources and treats these segments as ground-truth rewards. We train a dense reward function conditioned on video segments and their corresponding subtasks to ensure alignment with ground-truth reward signals by minimizing the Equivalent-Policy Invariant Comparison distance. Additionally, we employ contrastive learning objectives to align video representations with subtasks, ensuring precise subtask inference during online interactions. Our experiments show that REDS significantly outperforms baseline methods on complex robotic manipulation tasks in Meta-World and more challenging real-world tasks, such as furniture assembly in FurnitureBench, with minimal human intervention. Moreover, REDS facilitates generalization to unseen tasks and robot embodiments, highlighting its potential for scalable deployment in diverse environments. 

**Abstract (ZH)**: 基于示例与分割的奖励学习（REWARD LEARNING FROM DEMONSTRATION WITH SEGMENTATIONS） 

---
# LV-DOT: LiDAR-visual dynamic obstacle detection and tracking for autonomous robot navigation 

**Title (ZH)**: 基于LiDAR-视觉的动态障碍物检测与跟踪方法在自主机器人导航中的应用 

**Authors**: Zhefan Xu, Haoyu Shen, Xinming Han, Hanyu Jin, Kanlong Ye, Kenji Shimada  

**Link**: [PDF](https://arxiv.org/pdf/2502.20607)  

**Abstract**: Accurate perception of dynamic obstacles is essential for autonomous robot navigation in indoor environments. Although sophisticated 3D object detection and tracking methods have been investigated and developed thoroughly in the fields of computer vision and autonomous driving, their demands on expensive and high-accuracy sensor setups and substantial computational resources from large neural networks make them unsuitable for indoor robotics. Recently, more lightweight perception algorithms leveraging onboard cameras or LiDAR sensors have emerged as promising alternatives. However, relying on a single sensor poses significant limitations: cameras have limited fields of view and can suffer from high noise, whereas LiDAR sensors operate at lower frequencies and lack the richness of visual features. To address this limitation, we propose a dynamic obstacle detection and tracking framework that uses both onboard camera and LiDAR data to enable lightweight and accurate perception. Our proposed method expands on our previous ensemble detection approach, which integrates outputs from multiple low-accuracy but computationally efficient detectors to ensure real-time performance on the onboard computer. In this work, we propose a more robust fusion strategy that integrates both LiDAR and visual data to enhance detection accuracy further. We then utilize a tracking module that adopts feature-based object association and the Kalman filter to track and estimate detected obstacles' states. Besides, a dynamic obstacle classification algorithm is designed to robustly identify moving objects. The dataset evaluation demonstrates a better perception performance compared to benchmark methods. The physical experiments on a quadcopter robot confirms the feasibility for real-world navigation. 

**Abstract (ZH)**: 基于车载摄像头和LiDAR数据的动态障碍检测与跟踪框架 

---
# Map Space Belief Prediction for Manipulation-Enhanced Mapping 

**Title (ZH)**: 基于操作增强制图的地图空间信念预测 

**Authors**: Joao Marcos Correia Marques, Nils Dengler, Tobias Zaenker, Jesper Mucke, Shenlong Wang, Maren Bennewitz, Kris Hauser  

**Link**: [PDF](https://arxiv.org/pdf/2502.20606)  

**Abstract**: Searching for objects in cluttered environments requires selecting efficient viewpoints and manipulation actions to remove occlusions and reduce uncertainty in object locations, shapes, and categories. In this work, we address the problem of manipulation-enhanced semantic mapping, where a robot has to efficiently identify all objects in a cluttered shelf. Although Partially Observable Markov Decision Processes~(POMDPs) are standard for decision-making under uncertainty, representing unstructured interactive worlds remains challenging in this formalism. To tackle this, we define a POMDP whose belief is summarized by a metric-semantic grid map and propose a novel framework that uses neural networks to perform map-space belief updates to reason efficiently and simultaneously about object geometries, locations, categories, occlusions, and manipulation physics. Further, to enable accurate information gain analysis, the learned belief updates should maintain calibrated estimates of uncertainty. Therefore, we propose Calibrated Neural-Accelerated Belief Updates (CNABUs) to learn a belief propagation model that generalizes to novel scenarios and provides confidence-calibrated predictions for unknown areas. Our experiments show that our novel POMDP planner improves map completeness and accuracy over existing methods in challenging simulations and successfully transfers to real-world cluttered shelves in zero-shot fashion. 

**Abstract (ZH)**: 在杂乱环境中搜索物体需要选择高效的视角和操作动作以移除遮挡并降低物体位置、形状和类别不确定性的程度。在本文中，我们解决了操作增强语义映射问题，要求机器人高效地在杂乱货架上识别所有物体。尽管部分可观测马尔可夫决策过程（POMDPs）是不确定性决策的标准方法，但在这种形式isms下表示非结构化交互世界仍具挑战性。为了解决这个问题，我们定义了一个POMDP，其信念由度量语义网格图总结，并提出了一种新的框架，使用神经网络在地图空间中进行信念更新，从而高效且同时地推理物体几何形状、位置、类别、遮挡和操作物理。此外，为了实现准确的信息增益分析，所学习的信念更新应保持对不确定性的校准估计。因此，我们提出了一种校准神经加速信念更新（CNABUs）方法，学习一个能够泛化到新场景并为未知区域提供置信校准预测的信念传播模型。我们的实验表明，我们的新型POMDP规划者在具有挑战性的模拟中提高了地图的完整性和准确性，并以零样本方式成功转移到实际杂乱货架上。 

---
# Close-Proximity Satellite Operations through Deep Reinforcement Learning and Terrestrial Testing Environments 

**Title (ZH)**: 通过深度强化学习和陆基测试环境实现近距离卫星操作 

**Authors**: Henry Lei, Joshua Aurand, Zachary S. Lippay, Sean Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2502.20554)  

**Abstract**: With the increasingly congested and contested space environment, safe and effective satellite operation has become increasingly challenging. As a result, there is growing interest in autonomous satellite capabilities, with common machine learning techniques gaining attention for their potential to address complex decision-making in the space domain. However, the "black-box" nature of many of these methods results in difficulty understanding the model's input/output relationship and more specifically its sensitivity to environmental disturbances, sensor noise, and control intervention. This paper explores the use of Deep Reinforcement Learning (DRL) for satellite control in multi-agent inspection tasks. The Local Intelligent Network of Collaborative Satellites (LINCS) Lab is used to test the performance of these control algorithms across different environments, from simulations to real-world quadrotor UAV hardware, with a particular focus on understanding their behavior and potential degradation in performance when deployed beyond the training environment. 

**Abstract (ZH)**: 随着太空环境日益拥挤和竞争激烈，卫星安全有效运行面临更大挑战。因此，自主卫星能力的研究兴趣日益增长，其中常见的机器学习技术因其在空间域复杂决策方面的潜在应用而受到关注。然而，许多这些方法的“黑盒”性质使得难以理解模型的输入/输出关系及其对环境扰动、传感器噪声和控制干预的敏感性。本文探讨了在多智能体检查任务中使用深度强化学习（DRL）进行卫星控制的应用。通过Local Intelligent Network of Collaborative Satellites（LINCS）实验室，这些控制算法在从仿真到真实四旋翼无人机硬件的不同环境中进行了测试，特别是在部署到训练环境以外时对其行为和性能退化进行了重点研究。 

---
# Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction 

**Title (ZH)**: 通过学习时间抽象实现面向随机环境的大规模决策 

**Authors**: Baiting Luo, Ava Pettet, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2502.21186)  

**Abstract**: Sequential decision-making in high-dimensional continuous action spaces, particularly in stochastic environments, faces significant computational challenges. We explore this challenge in the traditional offline RL setting, where an agent must learn how to make decisions based on data collected through a stochastic behavior policy. We present \textit{Latent Macro Action Planner} (L-MAP), which addresses this challenge by learning a set of temporally extended macro-actions through a state-conditional Vector Quantized Variational Autoencoder (VQ-VAE), effectively reducing action dimensionality. L-MAP employs a (separate) learned prior model that acts as a latent transition model and allows efficient sampling of plausible actions. During planning, our approach accounts for stochasticity in both the environment and the behavior policy by using Monte Carlo tree search (MCTS). In offline RL settings, including stochastic continuous control tasks, L-MAP efficiently searches over discrete latent actions to yield high expected returns. Empirical results demonstrate that L-MAP maintains low decision latency despite increased action dimensionality. Notably, across tasks ranging from continuous control with inherently stochastic dynamics to high-dimensional robotic hand manipulation, L-MAP significantly outperforms existing model-based methods and performs on-par with strong model-free actor-critic baselines, highlighting the effectiveness of the proposed approach in planning in complex and stochastic environments with high-dimensional action spaces. 

**Abstract (ZH)**: 高维连续动作空间中在随机环境中进行序列决策存在显著的计算挑战：传统离线RL中的_latent宏动作规划者_(L-MAP)及其应用 

---
# EDENet: Echo Direction Encoding Network for Place Recognition Based on Ground Penetrating Radar 

**Title (ZH)**: EDENet：基于地面穿透雷达的回声方向编码网络在位置识别中的应用 

**Authors**: Pengyu Zhang, Xieyuanli Chen, Yuwei Chen, Beizhen Bi, Zhuo Xu, Tian Jin, Xiaotao Huang, Liang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20643)  

**Abstract**: Ground penetrating radar (GPR) based localization has gained significant recognition in robotics due to its ability to detect stable subsurface features, offering advantages in environments where traditional sensors like cameras and LiDAR may struggle. However, existing methods are primarily focused on small-scale place recognition (PR), leaving the challenges of PR in large-scale maps unaddressed. These challenges include the inherent sparsity of underground features and the variability in underground dielectric constants, which complicate robust localization. In this work, we investigate the geometric relationship between GPR echo sequences and underground scenes, leveraging the robustness of directional features to inform our network design. We introduce learnable Gabor filters for the precise extraction of directional responses, coupled with a direction-aware attention mechanism for effective geometric encoding. To further enhance performance, we incorporate a shift-invariant unit and a multi-scale aggregation strategy to better accommodate variations in di-electric constants. Experiments conducted on public datasets demonstrate that our proposed EDENet not only surpasses existing solutions in terms of PR performance but also offers advantages in model size and computational efficiency. 

**Abstract (ZH)**: 基于地面穿透雷达（GPR）的定位在机器人领域由于其能够检测稳定的地下特征，而在传统传感器如相机和LiDAR表现不佳的环境中展现出优势，获得了显著的认可。然而，现有的方法主要集中在小规模场所识别（PR）上，大型规模地图的PR挑战未得到充分解决。这些挑战包括地下特征的固有稀疏性和地下介电常数的变异性，这给稳健定位带来了复杂性。在本工作中，我们研究了GPR回波序列与地下场景之间的几何关系，利用方向特征的鲁棒性指导网络设计。我们引入可学习的Gabor滤波器来精确提取方向响应，并结合方向感知的注意机制进行有效的几何编码。为了进一步提升性能，我们加入了移不变单元和多尺度聚合策略，更好地适应介电常数的变化。在公开数据集上的实验表明，我们提出的EDENet不仅在PR性能上优于现有解决方案，还在模型大小和计算效率方面具有优势。 

---
# Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning 

**Title (ZH)**: 多模态梦境：基于世界模型的强化学习全局工作空间方法 

**Authors**: Léopold Maytié, Roland Bertin Johannet, Rufin VanRullen  

**Link**: [PDF](https://arxiv.org/pdf/2502.21142)  

**Abstract**: Humans leverage rich internal models of the world to reason about the future, imagine counterfactuals, and adapt flexibly to new situations. In Reinforcement Learning (RL), world models aim to capture how the environment evolves in response to the agent's actions, facilitating planning and generalization. However, typical world models directly operate on the environment variables (e.g. pixels, physical attributes), which can make their training slow and cumbersome; instead, it may be advantageous to rely on high-level latent dimensions that capture relevant multimodal variables. Global Workspace (GW) Theory offers a cognitive framework for multimodal integration and information broadcasting in the brain, and recent studies have begun to introduce efficient deep learning implementations of GW. Here, we evaluate the capabilities of an RL system combining GW with a world model. We compare our GW-Dreamer with various versions of the standard PPO and the original Dreamer algorithms. We show that performing the dreaming process (i.e., mental simulation) inside the GW latent space allows for training with fewer environment steps. As an additional emergent property, the resulting model (but not its comparison baselines) displays strong robustness to the absence of one of its observation modalities (images or simulation attributes). We conclude that the combination of GW with World Models holds great potential for improving decision-making in RL agents. 

**Abstract (ZH)**: 人类利用丰富的心内模型来推断未来、想象反事实场景并灵活适应新情况。在强化学习（RL）中，世界模型旨在捕捉环境在代理行动后的演变方式，从而促进规划和泛化。然而，典型的世界模型直接操作于环境变量（如像素、物理属性），这可能导致训练过程变得缓慢且复杂；相反，依赖能捕捉相关多模态变量的高级潜在维度可能更具优势。全局工作空间（GW）理论提供了一种大脑中多模态整合和信息广播的认知框架，近期研究已经开始引入高效的GW的深度学习实现。在此，我们评估了将GW与世界模型结合使用的RL系统的能力。我们将我们的GW-Dreamer与标准PPO的不同版本以及原始Dreamer算法进行了比较。我们展示了在GW的潜在空间中执行梦境过程（即心理模拟）可以减少环境步骤的数量进行训练。此外，结果模型（而非其比较基准）显示出对其中一种观测模态缺失（图像或模拟属性）的强大鲁棒性。我们得出结论，将GW与世界模型结合使用有望显著提高RL代理的决策能力。 

---
# MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts 

**Title (ZH)**: MV-MATH: 评估多模态数学推理在多视觉情境中的性能 

**Authors**: Peijie Wang, Zhongzhi Li, Fei Yin, Dekang Ran, Chenglin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20808)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown promising capabilities in mathematical reasoning within visual contexts across various datasets. However, most existing multimodal math benchmarks are limited to single-visual contexts, which diverges from the multi-visual scenarios commonly encountered in real-world mathematical applications. To address this gap, we introduce MV-MATH: a meticulously curated dataset of 2,009 high-quality mathematical problems. Each problem integrates multiple images interleaved with text, derived from authentic K-12 scenarios, and enriched with detailed annotations. MV-MATH includes multiple-choice, free-form, and multi-step questions, covering 11 subject areas across 3 difficulty levels, and serves as a comprehensive and rigorous benchmark for assessing MLLMs' mathematical reasoning in multi-visual contexts. Through extensive experimentation, we observe that MLLMs encounter substantial challenges in multi-visual math tasks, with a considerable performance gap relative to human capabilities on MV-MATH. Furthermore, we analyze the performance and error patterns of various models, providing insights into MLLMs' mathematical reasoning capabilities within multi-visual settings. 

**Abstract (ZH)**: 多模态大型语言模型在多视觉情境下的数学推理能力：MV-MATH数据集 

---
# Acquiring Grounded Representations of Words with Situated Interactive Instruction 

**Title (ZH)**: 基于定向交互指令获取词汇的接地表示 

**Authors**: Shiwali Mohan, Aaron H. Mininger, James R. Kirk, John E. Laird  

**Link**: [PDF](https://arxiv.org/pdf/2502.20754)  

**Abstract**: We present an approach for acquiring grounded representations of words from mixed-initiative, situated interactions with a human instructor. The work focuses on the acquisition of diverse types of knowledge including perceptual, semantic, and procedural knowledge along with learning grounded meanings. Interactive learning allows the agent to control its learning by requesting instructions about unknown concepts, making learning efficient. Our approach has been instantiated in Soar and has been evaluated on a table-top robotic arm capable of manipulating small objects. 

**Abstract (ZH)**: 从人机协作中获取单词的 grounded 表征的方法：基于交互式学习的多样化知识获取 

---
# ReaLJam: Real-Time Human-AI Music Jamming with Reinforcement Learning-Tuned Transformers 

**Title (ZH)**: ReaLJam: 基于强化学习调优的变压器实现实时人机音乐合奏 

**Authors**: Alexander Scarlatos, Yusong Wu, Ian Simon, Adam Roberts, Tim Cooijmans, Natasha Jaques, Cassie Tarakajian, Cheng-Zhi Anna Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21267)  

**Abstract**: Recent advances in generative artificial intelligence (AI) have created models capable of high-quality musical content generation. However, little consideration is given to how to use these models for real-time or cooperative jamming musical applications because of crucial required features: low latency, the ability to communicate planned actions, and the ability to adapt to user input in real-time. To support these needs, we introduce ReaLJam, an interface and protocol for live musical jamming sessions between a human and a Transformer-based AI agent trained with reinforcement learning. We enable real-time interactions using the concept of anticipation, where the agent continually predicts how the performance will unfold and visually conveys its plan to the user. We conduct a user study where experienced musicians jam in real-time with the agent through ReaLJam. Our results demonstrate that ReaLJam enables enjoyable and musically interesting sessions, and we uncover important takeaways for future work. 

**Abstract (ZH)**: Recent Advances in Generative Artificial Intelligence for Live Musical Jamming with Transformer-Based Agents Trained with Reinforcement Learning 

---
# XAIxArts Manifesto: Explainable AI for the Arts 

**Title (ZH)**: XAIxArts 宣言：可解释人工智能在艺术领域的应用 

**Authors**: Nick Bryan-Kinns, Shuoyang Jasper Zheng, Francisco Castro, Makayla Lewis, Jia-Rey Chang, Gabriel Vigliensoni, Terence Broad, Michael Clemens, Elizabeth Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2502.21220)  

**Abstract**: Explainable AI (XAI) is concerned with how to make AI models more understandable to people. To date these explanations have predominantly been technocentric - mechanistic or productivity oriented. This paper introduces the Explainable AI for the Arts (XAIxArts) manifesto to provoke new ways of thinking about explainability and AI beyond technocentric discourses. Manifestos offer a means to communicate ideas, amplify unheard voices, and foster reflection on practice. To supports the co-creation and revision of the XAIxArts manifesto we combine a World Café style discussion format with a living manifesto to question four core themes: 1) Empowerment, Inclusion, and Fairness; 2) Valuing Artistic Practice; 3) Hacking and Glitches; and 4) Openness. Through our interactive living manifesto experience we invite participants to actively engage in shaping this XIAxArts vision within the CHI community and beyond. 

**Abstract (ZH)**: 可解释的人工智能（XAI）旨在使人工智能模型更易于人类理解。到目前为止，这些解释主要具有技术中心倾向——机械化或生产导向。本文引入可解释的人工智能与艺术（XAIxArts）宣言，以激发关于可解释性与人工智能超越技术中心话语的新思考方式。宣言提供了一种交流思想、放大未被听见的声音并促进实践反思的手段。为了支持XAIxArts宣言的共同创作与修订，我们结合了World Café风格的讨论格式与活宣言，探讨四个核心主题：1）赋权、包容与公平；2）重视艺术实践；3）黑客与арат奇现象；4）开放性。通过我们的互动活宣言体验，我们邀请参与者积极塑造这一XAIxArts愿景，不仅限于CHI社区，还扩大至更广泛的领域。 

---
# Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? 

**Title (ZH)**: 一切万物，同时呈现：机械可解释性可识别吗？ 

**Authors**: Maxime Méloux, Silviu Maniu, François Portet, Maxime Peyrard  

**Link**: [PDF](https://arxiv.org/pdf/2502.20914)  

**Abstract**: As AI systems are used in high-stakes applications, ensuring interpretability is crucial. Mechanistic Interpretability (MI) aims to reverse-engineer neural networks by extracting human-understandable algorithms to explain their behavior. This work examines a key question: for a given behavior, and under MI's criteria, does a unique explanation exist? Drawing on identifiability in statistics, where parameters are uniquely inferred under specific assumptions, we explore the identifiability of MI explanations.
We identify two main MI strategies: (1) "where-then-what," which isolates a circuit replicating model behavior before interpreting it, and (2) "what-then-where," which starts with candidate algorithms and searches for neural activation subspaces implementing them, using causal alignment.
We test both strategies on Boolean functions and small multi-layer perceptrons, fully enumerating candidate explanations. Our experiments reveal systematic non-identifiability: multiple circuits can replicate behavior, a circuit can have multiple interpretations, several algorithms can align with the network, and one algorithm can align with different subspaces.
Is uniqueness necessary? A pragmatic approach may require only predictive and manipulability standards. If uniqueness is essential for understanding, stricter criteria may be needed. We also reference the inner interpretability framework, which validates explanations through multiple criteria. This work contributes to defining explanation standards in AI. 

**Abstract (ZH)**: 随着AI系统在高风险应用中的使用，确保可解释性至关重要。机制可解释性（MI）旨在通过提取人类可理解的算法来反向工程神经网络，以解释其行为。本文探讨了一个关键问题：对于给定的行为，以及在MI的标准下，是否唯一解释存在？我们借鉴统计学中的可识别性概念，在特定假设下唯一推断参数，探讨MI解释的可识别性。

我们识别了两种主要的MI策略：（1）“哪里-然后什么”，该策略先隔离一个复制模型行为的电路，然后再对其解释；（2）“什么-然后哪里”，该策略从候选算法开始，通过因果对齐搜索实现它们的神经激活子空间。

我们在布尔函数和小型多层感知器上测试了这两种策略，完全枚举候选解释。我们的实验揭示了系统性的非唯一性：多个电路可以复制行为，一个电路可以有多重解释，多种算法可以与网络对齐，一个算法可以与不同的子空间对齐。

唯一性是否必要？一种实用的方法可能只需要预测性和操控性标准。如果唯一性对于理解是必要的，可能需要更严格的标准。我们还参考了内解释框架，通过多个标准验证解释。本文为定义AI中的解释标准做出了贡献。 

---
# Reinforcement Learning with Curriculum-inspired Adaptive Direct Policy Guidance for Truck Dispatching 

**Title (ZH)**: 基于课程引导自适应直接策略指导的强化学习在卡车调度中的应用 

**Authors**: Shi Meng, Bin Tian, Xiaotong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20845)  

**Abstract**: Efficient truck dispatching via Reinforcement Learning (RL) in open-pit mining is often hindered by reliance on complex reward engineering and value-based methods. This paper introduces Curriculum-inspired Adaptive Direct Policy Guidance, a novel curriculum learning strategy for policy-based RL to address these issues. We adapt Proximal Policy Optimization (PPO) for mine dispatching's uneven decision intervals using time deltas in Temporal Difference and Generalized Advantage Estimation, and employ a Shortest Processing Time teacher policy for guided exploration via policy regularization and adaptive guidance. Evaluations in OpenMines demonstrate our approach yields a 10% performance gain and faster convergence over standard PPO across sparse and dense reward settings, showcasing improved robustness to reward design. This direct policy guidance method provides a general and effective curriculum learning technique for RL-based truck dispatching, enabling future work on advanced architectures. 

**Abstract (ZH)**: 基于 Curriculum 启发的自适应直接策略引导在露天矿卡车调度中的强化学习 

---
# WorldModelBench: Judging Video Generation Models As World Models 

**Title (ZH)**: WorldModelBench: 将视频生成模型评估为世界模型 

**Authors**: Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph E. Gonzalez, Ion Stoica, Song Han, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20694)  

**Abstract**: Video generation models have rapidly progressed, positioning themselves as video world models capable of supporting decision-making applications like robotics and autonomous driving. However, current benchmarks fail to rigorously evaluate these claims, focusing only on general video quality, ignoring important factors to world models such as physics adherence. To bridge this gap, we propose WorldModelBench, a benchmark designed to evaluate the world modeling capabilities of video generation models in application-driven domains. WorldModelBench offers two key advantages: (1) Against to nuanced world modeling violations: By incorporating instruction-following and physics-adherence dimensions, WorldModelBench detects subtle violations, such as irregular changes in object size that breach the mass conservation law - issues overlooked by prior benchmarks. (2) Aligned with large-scale human preferences: We crowd-source 67K human labels to accurately measure 14 frontier models. Using our high-quality human labels, we further fine-tune an accurate judger to automate the evaluation procedure, achieving 8.6% higher average accuracy in predicting world modeling violations than GPT-4o with 2B parameters. In addition, we demonstrate that training to align human annotations by maximizing the rewards from the judger noticeably improve the world modeling capability. The website is available at this https URL. 

**Abstract (ZH)**: 视频生成模型已经迅速发展，定位为能够支持如机器人和自动驾驶等决策应用的视频世界模型。然而，当前的基准未能严格评估这些声明，只关注通用视频质量，忽略了世界模型的重要因素如物理一致性。为弥补这一差距，我们提出了WorldModelBench，这是一个旨在评估视频生成模型在应用驱动领域中世界建模能力的基准。WorldModelBench提供了两项关键优势：(1) 针对细腻的世界建模违背：通过引入指令遵循和物理一致性维度，WorldModelBench能够检测细微的违背，如违反质守恒定律的物体大小不规则变化——这些问题被之前的基准所忽视。(2) 符合大规模人类偏好：我们收集了67,000个人类标签以准确度量14个前沿模型。利用高质量的人类标签，我们进一步微调了一个准确的评判器，以自动化评估流程，其在预测世界建模违背方面的平均准确率比参数量为20亿的GPT-4o高出8.6%。此外，我们展示了通过最大化评判器奖励来对齐人类注释的训练显著提高世界建模能力。网站地址为这个=https://... 

---
# Scalable Coordinated Learning for H2M/R Applications over Optical Access Networks (Invited) 

**Title (ZH)**: 面向光接入网络的H2M/R应用的可扩展协调学习（邀请报告） 

**Authors**: Sourav Mondal, Elaine Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.20598)  

**Abstract**: One of the primary research interests adhering to next-generation fiber-wireless access networks is human-to-machine/robot (H2M/R) collaborative communications facilitating Industry 5.0. This paper discusses scalable H2M/R communications across large geographical distances that also allow rapid onboarding of new machines/robots as $\sim72\%$ training time is saved through global-local coordinated learning. 

**Abstract (ZH)**: 下一代光纤无线接入网络中的主要研究兴趣之一是支持工业4.0的机对人/机器人（H2M/R）协作通信。本文讨论了跨越大地理距离的可扩展H2M/R通信，通过全局-局部协同学习节省约72%的训练时间，从而实现新机器/机器人快速上线。 

---
