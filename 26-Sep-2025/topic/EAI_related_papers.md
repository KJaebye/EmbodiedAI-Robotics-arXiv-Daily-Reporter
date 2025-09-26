# Taxonomy-aware Dynamic Motion Generation on Hyperbolic Manifolds 

**Title (ZH)**: 基于-taxonomy的双曲流形上动态运动生成 

**Authors**: Luis Augenstein, Noémie Jaquier, Tamim Asfour, Leonel Rozo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21281)  

**Abstract**: Human-like motion generation for robots often draws inspiration from biomechanical studies, which often categorize complex human motions into hierarchical taxonomies. While these taxonomies provide rich structural information about how movements relate to one another, this information is frequently overlooked in motion generation models, leading to a disconnect between the generated motions and their underlying hierarchical structure. This paper introduces the \ac{gphdm}, a novel approach that learns latent representations preserving both the hierarchical structure of motions and their temporal dynamics to ensure physical consistency. Our model achieves this by extending the dynamics prior of the Gaussian Process Dynamical Model (GPDM) to the hyperbolic manifold and integrating it with taxonomy-aware inductive biases. Building on this geometry- and taxonomy-aware frameworks, we propose three novel mechanisms for generating motions that are both taxonomically-structured and physically-consistent: two probabilistic recursive approaches and a method based on pullback-metric geodesics. Experiments on generating realistic motion sequences on the hand grasping taxonomy show that the proposed GPHDM faithfully encodes the underlying taxonomy and temporal dynamics, and generates novel physically-consistent trajectories. 

**Abstract (ZH)**: 基于几何与分类知识的人形化机器人运动生成 

---
# RetoVLA: Reusing Register Tokens for Spatial Reasoning in Vision-Language-Action Models 

**Title (ZH)**: RetoVLA: 重用寄存器令牌进行视觉-语言-动作模型的空间推理 

**Authors**: Jiyeon Koo, Taewan Cho, Hyunjoon Kang, Eunseom Pyo, Tae Gyun Oh, Taeryang Kim, Andrew Jaeyong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21243)  

**Abstract**: Recent Vision-Language-Action (VLA) models demonstrate remarkable generalization in robotics but are restricted by their substantial size and computational cost, limiting real-world deployment. However, conventional lightweighting methods often sacrifice critical capabilities, particularly spatial reasoning. This creates a trade-off between efficiency and performance. To address this challenge, our work reuses Register Tokens, which were introduced for artifact removal in Vision Transformers but subsequently discarded. We suppose that these tokens contain essential spatial information and propose RetoVLA, a novel architecture that reuses them directly by injecting them into the Action Expert.
RetoVLA maintains a lightweight structure while leveraging this repurposed spatial context to enhance reasoning. We demonstrate RetoVLA's effectiveness through a series of comprehensive experiments. On our custom-built 7-DOF robot arm, the model achieves a 17.1%p absolute improvement in success rates for complex manipulation tasks. Our results confirm that reusing Register Tokens directly enhances spatial reasoning, demonstrating that what was previously discarded as an artifact is in fact a valuable, unexplored resource for robotic intelligence. A video demonstration is available at: this https URL 

**Abstract (ZH)**: 近期的视觉-语言-动作（VLA）模型在机器人技术中展现了出色的泛化能力，但因其庞大的规模和高昂的计算成本而受到限制，阻碍了其实现真正的部署。然而，传统的轻量化方法经常会牺牲关键的能力，特别是在空间推理方面。这就造成了效率与性能之间的权衡。为了解决这一挑战，我们的工作重新利用了注册token（Register Tokens），这种token最初是在视觉变换器中引入以去除干扰物，之后被遗弃。我们假设这些token包含重要的空间信息，并提出了一种新的架构RetoVLA，该架构通过直接注入动作专家中来重新利用这些token，从而保持轻量级结构的同时提升推理能力。我们通过一系列全面的实验展示了RetoVLA的有效性。在我们自建的7-DOF机器人手臂上，该模型在复杂操作任务中的成功率绝对提高了17.1%。我们的结果证实，直接复用注册token能够增强空间推理能力，表明之前被视为干扰物的资源其实是提升机器人智能的宝贵且未被充分探索的资源。视频演示可访问：this https URL。 

---
# SEEC: Stable End-Effector Control with Model-Enhanced Residual Learning for Humanoid Loco-Manipulation 

**Title (ZH)**: SEEC：基于模型增强残差学习的稳定末端执行器控制用于类人操作与 Manipulation 

**Authors**: Jaehwi Jang, Zhuoheng Wang, Ziyi Zhou, Feiyang Wu, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21231)  

**Abstract**: Arm end-effector stabilization is essential for humanoid loco-manipulation tasks, yet it remains challenging due to the high degrees of freedom and inherent dynamic instability of bipedal robot structures. Previous model-based controllers achieve precise end-effector control but rely on precise dynamics modeling and estimation, which often struggle to capture real-world factors (e.g., friction and backlash) and thus degrade in practice. On the other hand, learning-based methods can better mitigate these factors via exploration and domain randomization, and have shown potential in real-world use. However, they often overfit to training conditions, requiring retraining with the entire body, and still struggle to adapt to unseen scenarios. To address these challenges, we propose a novel stable end-effector control (SEEC) framework with model-enhanced residual learning that learns to achieve precise and robust end-effector compensation for lower-body induced disturbances through model-guided reinforcement learning (RL) with a perturbation generator. This design allows the upper-body policy to achieve accurate end-effector stabilization as well as adapt to unseen locomotion controllers with no additional training. We validate our framework in different simulators and transfer trained policies to the Booster T1 humanoid robot. Experiments demonstrate that our method consistently outperforms baselines and robustly handles diverse and demanding loco-manipulation tasks. 

**Abstract (ZH)**: 基于模型增强残差学习的稳定末端执行器控制框架在仿人移动操作任务中的应用 

---
# Human-like Navigation in a World Built for Humans 

**Title (ZH)**: 人类导航于为人类设计的世界中 

**Authors**: Bhargav Chandaka, Gloria X. Wang, Haozhe Chen, Henry Che, Albert J. Zhai, Shenlong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21189)  

**Abstract**: When navigating in a man-made environment they haven't visited before--like an office building--humans employ behaviors such as reading signs and asking others for directions. These behaviors help humans reach their destinations efficiently by reducing the need to search through large areas. Existing robot navigation systems lack the ability to execute such behaviors and are thus highly inefficient at navigating within large environments. We present ReasonNav, a modular navigation system which integrates these human-like navigation skills by leveraging the reasoning capabilities of a vision-language model (VLM). We design compact input and output abstractions based on navigation landmarks, allowing the VLM to focus on language understanding and reasoning. We evaluate ReasonNav on real and simulated navigation tasks and show that the agent successfully employs higher-order reasoning to navigate efficiently in large, complex buildings. 

**Abstract (ZH)**: 当在未访问过的建筑物（如办公楼）中导航时，人类会表现出阅读指示牌和向他人询问方向等行为，这些行为有助于人类高效地到达目的地，减少大面积搜索的需要。现有的机器人导航系统缺乏执行此类行为的能力，因此在大型环境中的导航效率极低。我们提出了ReasonNav，这是一种模块化的导航系统，通过利用视觉语言模型（VLM）的推理能力来整合这些类似人类的导航技能。我们基于导航地标设计了紧凑的输入和输出抽象，使VLM能够专注于语言理解与推理。我们在实际和模拟的导航任务中评估了ReasonNav，并展示了该代理能够运用高级推理在大型复杂建筑中高效导航。 

---
# Rich State Observations Empower Reinforcement Learning to Surpass PID: A Drone Ball Balancing Study 

**Title (ZH)**: 丰富状态观测增强强化学习超越PID：无人机球平衡研究 

**Authors**: Mingjiang Liu, Hailong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21122)  

**Abstract**: This paper addresses a drone ball-balancing task, in which a drone stabilizes a ball atop a movable beam through cable-based interaction. We propose a hierarchical control framework that decouples high-level balancing policy from low-level drone control, and train a reinforcement learning (RL) policy to handle the high-level decision-making. Simulation results show that the RL policy achieves superior performance compared to carefully tuned PID controllers within the same hierarchical structure. Through systematic comparative analysis, we demonstrate that RL's advantage stems not from improved parameter tuning or inherent nonlinear mapping capabilities, but from its ability to effectively utilize richer state observations. These findings underscore the critical role of comprehensive state representation in learning-based systems and suggest that enhanced sensing could be instrumental in improving controller performance. 

**Abstract (ZH)**: 基于绳索交互的无人机球平衡任务的研究：一种层次控制框架及强化学习政策的训练与评估 

---
# Cross-Modal Instructions for Robot Motion Generation 

**Title (ZH)**: 跨模态指令生成机器人运动 

**Authors**: William Barron, Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21107)  

**Abstract**: Teaching robots novel behaviors typically requires motion demonstrations via teleoperation or kinaesthetic teaching, that is, physically guiding the robot. While recent work has explored using human sketches to specify desired behaviors, data collection remains cumbersome, and demonstration datasets are difficult to scale. In this paper, we introduce an alternative paradigm, Learning from Cross-Modal Instructions, where robots are shaped by demonstrations in the form of rough annotations, which can contain free-form text labels, and are used in lieu of physical motion. We introduce the CrossInstruct framework, which integrates cross-modal instructions as examples into the context input to a foundational vision-language model (VLM). The VLM then iteratively queries a smaller, fine-tuned model, and synthesizes the desired motion over multiple 2D views. These are then subsequently fused into a coherent distribution over 3D motion trajectories in the robot's workspace. By incorporating the reasoning of the large VLM with a fine-grained pointing model, CrossInstruct produces executable robot behaviors that generalize beyond the environment of in the limited set of instruction examples. We then introduce a downstream reinforcement learning pipeline that leverages CrossInstruct outputs to efficiently learn policies to complete fine-grained tasks. We rigorously evaluate CrossInstruct on benchmark simulation tasks and real hardware, demonstrating effectiveness without additional fine-tuning and providing a strong initialization for policies subsequently refined via reinforcement learning. 

**Abstract (ZH)**: 从跨模态指令学习 

---
# Normalizing Flows are Capable Visuomotor Policy Learning Models 

**Title (ZH)**: 正态流是有能力的visuomotor策略学习模型 

**Authors**: Simon Kristoffersson Lind, Jialong Li, Maj Stenmark, Volker Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2509.21073)  

**Abstract**: The field of general purpose robotics has recently embraced powerful probabilistic models, such as diffusion models, to model and learn complex behaviors. However, these models often come with significant trade-offs, namely high computational costs for inference and a fundamental inability to quantify output uncertainty. We argue that a model's trustworthiness, a critical factor for reliable, general-purpose robotics, is inherently linked to its ability to provide confidence measures.
In this work, we introduce Normalizing Flows Policy, a novel visuomotor policy learning model based on Normalizing Flows. We show that Normalizing Flows are a natural and powerful alternative to diffusion models, providing both a statistically sound measure of confidence and a highly efficient inference process. Through comprehensive experiments across four distinct simulated robotic tasks, we demonstrate that Normalizing Flows Policy achieves performance comparable to, and often surpassing, Diffusion Policy, and it does so not only with improved sample efficiency but also with up to 30 times faster inference. Additionally, our ablation study validates several key architectural and training techniques that enable Normalizing Flows to perform well in this domain. 

**Abstract (ZH)**: 通用机器人领域的研究最近采纳了强大的概率模型，如扩散模型，以建模和学习复杂行为。然而，这些模型通常伴随着显著的权衡，即推断的高计算成本和根本无法量化输出不确定性。我们认为，模型的可信度——这是可靠且通用的机器人技术的关键因素——与其提供信心度量的能力息息相关。

在本文中，我们提出了一种基于归一化流的视觉运动策略学习模型——归一化流策略。我们展示归一化流是一种自然且强大的扩散模型替代方案，提供了统计上合理的信心度量和高度高效的推理过程。通过在四个不同的模拟机器人任务上进行全面实验，我们证明归一化流策略不仅达到了与扩散策略相当的性能，而且在样本效率上有所改进，并且推断速度最快可提高30倍。此外，我们的消融研究验证了几种关键的架构和训练技术，这些技术使归一化流在该领域中表现良好。 

---
# MPC-based Deep Reinforcement Learning Method for Space Robotic Control with Fuel Sloshing Mitigation 

**Title (ZH)**: 基于MPC的深度强化学习空间机器人控制方法及燃料晃动缓解 

**Authors**: Mahya Ramezani, M. Amin Alandihallaj, Barış Can Yalçın, Miguel Angel Olivares Mendez, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2509.21045)  

**Abstract**: This paper presents an integrated Reinforcement Learning (RL) and Model Predictive Control (MPC) framework for autonomous satellite docking with a partially filled fuel tank. Traditional docking control faces challenges due to fuel sloshing in microgravity, which induces unpredictable forces affecting stability. To address this, we integrate Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) RL algorithms with MPC, leveraging MPC's predictive capabilities to accelerate RL training and improve control robustness. The proposed approach is validated through Zero-G Lab of SnT experiments for planar stabilization and high-fidelity numerical simulations for 6-DOF docking with fuel sloshing dynamics. Simulation results demonstrate that SAC-MPC achieves superior docking accuracy, higher success rates, and lower control effort, outperforming standalone RL and PPO-MPC methods. This study advances fuel-efficient and disturbance-resilient satellite docking, enhancing the feasibility of on-orbit refueling and servicing missions. 

**Abstract (ZH)**: 基于部分燃料箱的自主卫星对接，结合RL和MPC的方法研究 

---
# KeyWorld: Key Frame Reasoning Enables Effective and Efficient World Models 

**Title (ZH)**: KeyWorld: 关键帧推理使世界模型高效且有效 

**Authors**: Sibo Li, Qianyue Hao, Yu Shang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21027)  

**Abstract**: Robotic world models are a promising paradigm for forecasting future environment states, yet their inference speed and the physical plausibility of generated trajectories remain critical bottlenecks, limiting their real-world applications. This stems from the redundancy of the prevailing frame-to-frame generation approach, where the model conducts costly computation on similar frames, as well as neglecting the semantic importance of key transitions. To address this inefficiency, we propose KeyWorld, a framework that improves text-conditioned robotic world models by concentrating transformers computation on a few semantic key frames while employing a lightweight convolutional model to fill the intermediate frames. Specifically, KeyWorld first identifies significant transitions by iteratively simplifying the robot's motion trajectories, obtaining the ground truth key frames. Then, a DiT model is trained to reason and generate these physically meaningful key frames from textual task descriptions. Finally, a lightweight interpolator efficiently reconstructs the full video by inpainting all intermediate frames. Evaluations on the LIBERO benchmark demonstrate that KeyWorld achieves a 5.68$\times$ acceleration compared to the frame-to-frame generation baseline, and focusing on the motion-aware key frames further contributes to the physical validity of the generated videos, especially on complex tasks. Our approach highlights a practical path toward deploying world models in real-time robotic control and other domains requiring both efficient and effective world models. Code is released at this https URL. 

**Abstract (ZH)**: 基于关键帧的机器人世界模型：一种提高推理速度和生成物理合理轨迹的方法 

---
# AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation 

**Title (ZH)**: AnywhereVLA：语言条件化的探索与移动操作 

**Authors**: Konstantin Gubernatorov, Artem Voronov, Roman Voronov, Sergei Pasynkov, Stepan Perminov, Ziang Guo, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2509.21006)  

**Abstract**: We address natural language pick-and-place in unseen, unpredictable indoor environments with AnywhereVLA, a modular framework for mobile manipulation. A user text prompt serves as an entry point and is parsed into a structured task graph that conditions classical SLAM with LiDAR and cameras, metric semantic mapping, and a task-aware frontier exploration policy. An approach planner then selects visibility and reachability aware pre grasp base poses. For interaction, a compact SmolVLA manipulation head is fine tuned on platform pick and place trajectories for the SO-101 by TheRobotStudio, grounding local visual context and sub-goals into grasp and place proposals. The full system runs fully onboard on consumer-level hardware, with Jetson Orin NX for perception and VLA and an Intel NUC for SLAM, exploration, and control, sustaining real-time operation. We evaluated AnywhereVLA in a multi-room lab under static scenes and normal human motion. In this setting, the system achieves a $46\%$ overall task success rate while maintaining throughput on embedded compute. By combining a classical stack with a fine-tuned VLA manipulation, the system inherits the reliability of geometry-based navigation with the agility and task generalization of language-conditioned manipulation. 

**Abstract (ZH)**: AnywhereVLA：一种适用于未见和不可预测室内环境的移动操作自然语言pick-and-place框架 

---
# Autoregressive End-to-End Planning with Time-Invariant Spatial Alignment and Multi-Objective Policy Refinement 

**Title (ZH)**: 具有时间不变空间对齐的自回归端到端规划和多目标策略精炼 

**Authors**: Jianbo Zhao, Taiyu Ban, Xiangjie Li, Xingtai Gui, Hangning Zhou, Lei Liu, Hongwei Zhao, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20938)  

**Abstract**: The inherent sequential modeling capabilities of autoregressive models make them a formidable baseline for end-to-end planning in autonomous driving. Nevertheless, their performance is constrained by a spatio-temporal misalignment, as the planner must condition future actions on past sensory data. This creates an inconsistent worldview, limiting the upper bound of performance for an otherwise powerful approach. To address this, we propose a Time-Invariant Spatial Alignment (TISA) module that learns to project initial environmental features into a consistent ego-centric frame for each future time step, effectively correcting the agent's worldview without explicit future scene prediction. In addition, we employ a kinematic action prediction head (i.e., acceleration and yaw rate) to ensure physically feasible trajectories. Finally, we introduce a multi-objective post-training stage using Direct Preference Optimization (DPO) to move beyond pure imitation. Our approach provides targeted feedback on specific driving behaviors, offering a more fine-grained learning signal than the single, overall objective used in standard DPO. Our model achieves a state-of-the-art 89.8 PDMS on the NAVSIM dataset among autoregressive models. The video document is available at this https URL. 

**Abstract (ZH)**: 自回归模型固有的序列建模能力使它们成为自主驾驶端到端规划的有力基准。然而，它们的表现受限于时空错位，因为规划器必须基于过去的传感器数据来预测未来的动作。这导致了一种不一致的世界观，限制了这一原本强大方法的上界性能。为此，我们提出了一种时间不变空间对齐（TISA）模块，该模块学习将初始环境特征投影到每个未来的时光步的以自我为中心的框架中，从而有效地纠正代理的世界观，而无需显式预测未来的场景。此外，我们采用了动力学动作预测头部（即加速度和偏航角）以确保物理上可行的轨迹。最后，我们引入了使用直接偏好优化（DPO）的多目标后训练阶段，以超越单纯的模仿。我们的方法对特定驾驶行为提供有针对性的反馈，提供了比标准DPO中使用的单一总体目标更为精细的学习信号。我们的模型在NAVSIM数据集中实现了自回归模型中的最新89.8 PDMS性能。视频文档可在以下链接获取。 

---
# Efficient Differentiable Contact Model with Long-range Influence 

**Title (ZH)**: 长程影响的高效可微接触模型 

**Authors**: Xiaohan Ye, Kui Wu, Zherong Pan, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2509.20917)  

**Abstract**: With the maturation of differentiable physics, its role in various downstream applications: such as model predictive control, robotic design optimization, and neural PDE solvers, has become increasingly important. However, the derivative information provided by differentiable simulators can exhibit abrupt changes or vanish altogether, impeding the convergence of gradient-based optimizers. In this work, we demonstrate that such erratic gradient behavior is closely tied to the design of contact models. We further introduce a set of properties that a contact model must satisfy to ensure well-behaved gradient information. Lastly, we present a practical contact model for differentiable rigid-body simulators that satisfies all of these properties while maintaining computational efficiency. Our experiments show that, even from simple initializations, our contact model can discover complex, contact-rich control signals, enabling the successful execution of a range of downstream locomotion and manipulation tasks. 

**Abstract (ZH)**: 随着可微物理的发展，在不同下游应用中的作用：例如模型预测控制、机器人设计优化和神经偏微分方程求解器等方面的作用日益重要。然而，可微模拟器提供的导数信息可能会出现突然变化甚至消失，阻碍梯度基优化器的收敛。本文表明，这种不规则的梯度行为与接触模型的设计密切相关。我们进一步引入了一组确保良好行为梯度信息的接触模型性质。最后，我们提出了一种适用于可微刚体模拟器的实用接触模型，该模型满足所有这些性质同时保持计算效率。我们的实验结果表明，即使从简单的初始化开始，我们的接触模型也能发现复杂的、接触丰富的控制信号，从而使多种下游运动和操作任务的成功执行成为可能。 

---
# MTRDrive: Memory-Tool Synergistic Reasoning for Robust Autonomous Driving in Corner Cases 

**Title (ZH)**: MTRDrive: 记忆-工具协同推理在corner case中实现稳健自主驾驶 

**Authors**: Ziang Luo, Kangan Qian, Jiahua Wang, Yuechen Luo, Jinyu Miao, Zheng Fu, Yunlong Wang, Sicong Jiang, Zilin Huang, Yifei Hu, Yuhao Yang, Hao Ye, Mengmeng Yang, Xiaojian Dong, Kun Jiang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20843)  

**Abstract**: Vision-Language Models(VLMs) have demonstrated significant potential for end-to-end autonomous driving, yet a substantial gap remains between their current capabilities and the reliability necessary for real-world deployment. A critical challenge is their fragility, characterized by hallucinations and poor generalization in out-of-distribution (OOD) scenarios. To bridge this gap, we introduce MTRDrive, a novel framework that integrates procedural driving experiences with a dynamic toolkit to enhance generalization and proactive decision-making.
MTRDrive addresses these limitations through a closed-loop system that combines a memory-based experience retrieval mechanism with dynamic toolkits. This synergy enables the model to interact more effectively with its environment, improving both reasoning and decision-making capabilities with the help of our memory-tool synergistic reasoning. Additionally, we introduce a new benchmark based on complex Roadwork construction scenarios to rigorously evaluate zero-shot generalization.
Extensive experiments demonstrate the superior effectiveness of our approach. On the public NAVSIM benchmark, our 3B-parameter MTRDrive model achieves an exceptional PDMS of 88.3 without chain-of-thought and sets a state-of-the-art performance bar on high-level planning, with a driving metric score of 79.8\% and a planning accuracy of 82.6\%. Rigorous zero-shot evaluation on the new Roadwork-VLM benchmark shows a strong ability to reason robustly in unseen scenarios, achieving a driving metric score of 80.2\%. These results highlight MTRDrive's potential to advance autonomous driving toward safer and more reliable systems. 

**Abstract (ZH)**: Vision-Language模型在端到端自动驾驶中的潜在应用及其与实际部署所需的可靠性的差距：MTRDrive框架的构建与评估 

---
# ImaginationPolicy: Towards Generalizable, Precise and Reliable End-to-End Policy for Robotic Manipulation 

**Title (ZH)**: 想象策略：迈向通用、精确且可靠的端到端机器人 manipulation 策略 

**Authors**: Dekun Lu, Wei Gao, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.20841)  

**Abstract**: End-to-end robot manipulation policies offer significant potential for enabling embodied agents to understand and interact with the world. Unlike traditional modular pipelines, end-to-end learning mitigates key limitations such as information loss between modules and feature misalignment caused by isolated optimization targets. Despite these advantages, existing end-to-end neural networks for robotic manipulation--including those based on large VLM/VLA models--remain insufficiently performant for large-scale practical deployment. In this paper, we take a step towards an end-to-end manipulation policy that is generalizable, accurate and reliable. To achieve this goal, we propose a novel Chain of Moving Oriented Keypoints (CoMOK) formulation for robotic manipulation. Our formulation is used as the action representation of a neural policy, which can be trained in an end-to-end fashion. Such an action representation is general, as it extends the standard end-effector pose action representation and supports a diverse set of manipulation tasks in a unified manner. The oriented keypoint in our method enables natural generalization to objects with different shapes and sizes, while achieving sub-centimeter accuracy. Moreover, our formulation can easily handle multi-stage tasks, multi-modal robot behaviors, and deformable objects. Extensive simulated and hardware experiments demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 端到端机器人操作策略为实现具身智能体对世界的理解和交互提供了巨大潜力。与传统的模块化管线不同，端到端学习缓解了模块间信息丢失和孤立优化目标导致的特征错位等关键限制。尽管具有这些优势，现有基于端到端学习的机器人操作神经网络——包括基于大规模VLM/VLA模型的网络——在大规模实际部署中仍不够高效。本文朝着构建一个通用、准确且可靠的端到端操作策略迈进。为此，我们提出了一种新颖的移动定向关键点链（CoMOK）形式化方法，用于机器人操作。该形式化方法用作神经策略的动作表示，并可以在端到端方式进行训练。这种动作表示是通用的，因为它扩展了标准的末端执行器姿态动作表示，并以统一的方式支持多种操作任务。我们方法中的定向关键点使模型能够自然地泛化到具有不同形状和大小的物体，并且在厘米级精度下实现亚厘米级的准确性。此外，该形式化方法能够轻松处理多阶段任务、多模态机器人行为以及变形物体。大量模拟和硬件实验表明了该方法的有效性。 

---
# SemSight: Probabilistic Bird's-Eye-View Prediction of Multi-Level Scene Semantics for Navigation 

**Title (ZH)**: SemSight：多层次场景语义的概率 bird's-eye-view 预测导航 

**Authors**: Jiaxuan He, Jiamei Ren, Chongshang Yan, Wenjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.20839)  

**Abstract**: In target-driven navigation and autonomous exploration, reasonable prediction of unknown regions is crucial for efficient navigation and environment understanding. Existing methods mostly focus on single objects or geometric occupancy maps, lacking the ability to model room-level semantic structures. We propose SemSight, a probabilistic bird's-eye-view prediction model for multi-level scene semantics. The model jointly infers structural layouts, global scene context, and target area distributions, completing semantic maps of unexplored areas while estimating probability maps for target categories. To train SemSight, we simulate frontier-driven exploration on 2,000 indoor layout graphs, constructing a diverse dataset of 40,000 sequential egocentric observations paired with complete semantic maps. We adopt an encoder-decoder network as the core architecture and introduce a mask-constrained supervision strategy. This strategy applies a binary mask of unexplored areas so that supervision focuses only on unknown regions, forcing the model to infer semantic structures from the observed context. Experimental results show that SemSight improves prediction performance for key functional categories in unexplored regions and outperforms non-mask-supervised approaches on metrics such as Structural Consistency (SC) and Region Recognition Accuracy (PA). It also enhances navigation efficiency in closed-loop simulations, reducing the number of search steps when guiding robots toward target areas. 

**Abstract (ZH)**: 基于目标导向的导航和自主探索中，合理预测未知区域对于高效导航和环境理解至关重要。现有方法主要关注单个物体或几何占用地图，缺乏建模房间级语义结构的能力。我们提出SemSight，一种多级场景语义的概率bird's-eye-view预测模型。该模型联合推断结构布局、全局场景上下文和目标区域分布，构建未探索区域的语义地图并估计目标类别概率图。为了训练SemSight，我们在2,000个室内布局图上模拟目标驱动的探索，构建了一个包含40,000个序列性第一人称观察及其完整语义地图的多样化数据集。我们采用编码-解码网络作为核心架构，并引入一种基于掩码的监督策略。该策略使用未探索区域的二进制掩码，使得监督仅专注于未知区域，迫使模型从观察上下文中推断语义结构。实验结果表明，SemSight在未探索区域内关键功能类别的预测性能上有所改进，并在结构一致性（SC）和区域识别准确率（PA）等指标上优于未采用掩码监督的方法。此外，SemSight在闭环模拟中提高了导航效率，减少了引导机器人前往目标区域所需的搜索步骤。 

---
# Leveraging Temporally Extended Behavior Sharing for Multi-task Reinforcement Learning 

**Title (ZH)**: 利用扩展时间行为共享进行多任务强化学习 

**Authors**: Gawon Lee, Daesol Cho, H. Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.20766)  

**Abstract**: Multi-task reinforcement learning (MTRL) offers a promising approach to improve sample efficiency and generalization by training agents across multiple tasks, enabling knowledge sharing between them. However, applying MTRL to robotics remains challenging due to the high cost of collecting diverse task data. To address this, we propose MT-Lévy, a novel exploration strategy that enhances sample efficiency in MTRL environments by combining behavior sharing across tasks with temporally extended exploration inspired by Lévy flight. MT-Lévy leverages policies trained on related tasks to guide exploration towards key states, while dynamically adjusting exploration levels based on task success ratios. This approach enables more efficient state-space coverage, even in complex robotics environments. Empirical results demonstrate that MT-Lévy significantly improves exploration and sample efficiency, supported by quantitative and qualitative analyses. Ablation studies further highlight the contribution of each component, showing that combining behavior sharing with adaptive exploration strategies can significantly improve the practicality of MTRL in robotics applications. 

**Abstract (ZH)**: 多任务强化学习（MTRL）提供了一种通过跨多个任务训练代理来提高样本效率和泛化能力的方法，从而在它们之间共享知识。然而，将MTRL应用到机器人领域仍然具有挑战性，因为收集多样化任务数据的成本很高。为了解决这个问题，我们提出了一种新颖的探索策略MT-Lévy，它通过在任务间共享行为并受Lévy飞行启发进行时序扩展探索来增强MTRL环境中的样本效率。MT-Lévy 利用相关任务训练的策略引导探索指向关键状态，并根据任务成功率动态调整探索水平。这种方法能够在复杂机器人环境中更有效地覆盖状态空间。实验结果表明，MT-Lévy 显著提高了探索和样本效率，并通过定量和定性分析予以验证。消融研究进一步强调了各组成部分的贡献，显示将行为共享与自适应探索策略相结合可以显著提高MTRL在机器人应用中的实用性。 

---
# SLAM-Free Visual Navigation with Hierarchical Vision-Language Perception and Coarse-to-Fine Semantic Topological Planning 

**Title (ZH)**: 基于分层视觉-语言感知和从粗到细语义拓扑规划的无SLAM视觉导航 

**Authors**: Guoyang Zhao, Yudong Li, Weiqing Qi, Kai Zhang, Bonan Liu, Kai Chen, Haoang Li, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.20739)  

**Abstract**: Conventional SLAM pipelines for legged robot navigation are fragile under rapid motion, calibration demands, and sensor drift, while offering limited semantic reasoning for task-driven exploration. To deal with these issues, we propose a vision-only, SLAM-free navigation framework that replaces dense geometry with semantic reasoning and lightweight topological representations. A hierarchical vision-language perception module fuses scene-level context with object-level cues for robust semantic inference. And a semantic-probabilistic topological map supports coarse-to-fine planning: LLM-based global reasoning for subgoal selection and vision-based local planning for obstacle avoidance. Integrated with reinforcement-learning locomotion controllers, the framework is deployable across diverse legged robot platforms. Experiments in simulation and real-world settings demonstrate consistent improvements in semantic accuracy, planning quality, and navigation success, while ablation studies further showcase the necessity of both hierarchical perception and fine local planning. This work introduces a new paradigm for SLAM-free, vision-language-driven navigation, shifting robotic exploration from geometry-centric mapping to semantics-driven decision making. 

**Abstract (ZH)**: 基于视觉的无需SLAM的任务驱动探索导航框架：从几何中心映射到语义驱动决策的新范式 

---
# RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking 

**Title (ZH)**: RobotDancing: 基于残差动作强化学习的robust长时间 humanoid 运动跟踪 

**Authors**: Zhenguo Sun, Yibo Peng, Yuan Meng, Xukun Li, Bo-Sheng Huang, Zhenshan Bing, Xinlong Wang, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.20717)  

**Abstract**: Long-horizon, high-dynamic motion tracking on humanoids remains brittle because absolute joint commands cannot compensate model-plant mismatch, leading to error accumulation. We propose RobotDancing, a simple, scalable framework that predicts residual joint targets to explicitly correct dynamics discrepancies. The pipeline is end-to-end--training, sim-to-sim validation, and zero-shot sim-to-real--and uses a single-stage reinforcement learning (RL) setup with a unified observation, reward, and hyperparameter configuration. We evaluate primarily on Unitree G1 with retargeted LAFAN1 dance sequences and validate transfer on H1/H1-2. RobotDancing can track multi-minute, high-energy behaviors (jumps, spins, cartwheels) and deploys zero-shot to hardware with high motion tracking quality. 

**Abstract (ZH)**: 长时程、高动态 humanoid 运动跟踪依然脆弱，因为绝对关节命令无法补偿模型与实际系统的差异，导致误差累积。我们提出 RobotDancing，一种简单可扩展的框架，预测残差关节目标以明确修正动力学差异。该框架为端到端设计，包括从训练到模拟验证再到零样本模拟到现实的流程，并采用统一观测、奖励和超参数配置的一阶段强化学习设置。我们主要基于 Unitree G1 和重新目标跟踪的 LAFAN1 舞蹈序列进行评估，并在 H1/H1-2 上验证了其迁移性。RobotDancing 可以跟踪多分钟、高能量行为（跳跃、旋转、侧手翻）并以高质量部署到硬件。 

---
# Digital Twin-Guided Robot Path Planning: A Beta-Bernoulli Fusion with Large Language Model as a Sensor 

**Title (ZH)**: 基于数字孪生引导的机器人路径规划：Beta-Bernoulli融合与大型语言模型作为传感器 

**Authors**: Mani Amani, Reza Akhavian  

**Link**: [PDF](https://arxiv.org/pdf/2509.20709)  

**Abstract**: Integrating natural language (NL) prompts into robotic mission planning has attracted significant interest in recent years. In the construction domain, Building Information Models (BIM) encapsulate rich NL descriptions of the environment. We present a novel framework that fuses NL directives with BIM-derived semantic maps via a Beta-Bernoulli Bayesian fusion by interpreting the LLM as a sensor: each obstacle's design-time repulsive coefficient is treated as a Beta(alpha, beta) random variable and LLM-returned danger scores are incorporated as pseudo-counts to update alpha and beta. The resulting posterior mean yields a continuous, context-aware repulsive gain that augments a Euclidean-distance-based potential field for cost heuristics. By adjusting gains based on sentiment and context inferred from user prompts, our method guides robots along safer, more context-aware paths. This provides a numerically stable method that can chain multiple natural commands and prompts from construction workers and foreman to enable planning while giving flexibility to be integrated in any learned or classical AI framework. Simulation results demonstrate that this Beta-Bernoulli fusion yields both qualitative and quantitative improvements in path robustness and validity. 

**Abstract (ZH)**: 将自然语言（NL）提示融入到机器人任务规划中近年来引起了广泛关注。在建筑领域，建筑信息模型（BIM）封装了环境的丰富自然语言描述。我们提出了一种新颖的框架，通过Beta-Bernoulli贝叶斯融合将自然语言指令与由BIM衍生的语义地图相结合：将每个障碍物的设计时排斥系数视为一个Beta(alpha, beta)随机变量，将从LLM返回的危险分数作为伪计数纳入其中以更新alpha和beta。所得到的后验均值产生了一个连续的、基于上下文的排斥增益，以增强基于欧几里得距离的势场的成本启发式。根据用户提示推断出的情绪和上下文调整增益，我们的方法引导机器人沿着更加安全和上下文感知的路径。这种方法提供了一种数值稳定的手段，可以串联来自建筑工人和主管的多个自然命令和提示，以实现规划的灵活性，并能够与任何学习或经典AI框架集成。仿真结果表明，这种Beta-Bernoulli融合在路径鲁棒性和有效性方面均取得了质和量的改进。 

---
# Building Information Models to Robot-Ready Site Digital Twins (BIM2RDT): An Agentic AI Safety-First Framework 

**Title (ZH)**: Building Information Models for Robot-Ready Site Digital Twins (BIM2RDT): 一种以代理为中心的AI安全优先框架 

**Authors**: Reza Akhavian, Mani Amani, Johannes Mootz, Robert Ashe, Behrad Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2509.20705)  

**Abstract**: The adoption of cyber-physical systems and jobsite intelligence that connects design models, real-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT (Building Information Models to Robot-Ready Site Digital Twins), an agentic artificial intelligence (AI) framework designed to transform static Building Information Modeling (BIM) into dynamic, robot-ready digital twins (DTs) that prioritize safety during execution. The framework bridges the gap between pre-existing BIM data and real-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual-spatial data collected by robots during site traversal. The methodology introduces Semantic-Gravity ICP (SG-ICP), a point cloud registration algorithm that leverages large language model (LLM) reasoning. Unlike traditional methods, SG-ICP utilizes an LLM to infer object-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real-time Hand-Arm Vibration (HAV) monitoring, mapping sensor-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%--88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349-1. 

**Abstract (ZH)**: 基于BIM到机器人就绪工地数字孪生的代理人工智能框架：BIM2RDT 

---
# Joint Flow Trajectory Optimization For Feasible Robot Motion Generation from Video Demonstrations 

**Title (ZH)**: 从视频示范中生成可行机器人运动的联合流动轨迹优化 

**Authors**: Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20703)  

**Abstract**: Learning from human video demonstrations offers a scalable alternative to teleoperation or kinesthetic teaching, but poses challenges for robot manipulators due to embodiment differences and joint feasibility constraints. We address this problem by proposing the Joint Flow Trajectory Optimization (JFTO) framework for grasp pose generation and object trajectory imitation under the video-based Learning-from-Demonstration (LfD) paradigm. Rather than directly imitating human hand motions, our method treats demonstrations as object-centric guides, balancing three objectives: (i) selecting a feasible grasp pose, (ii) generating object trajectories consistent with demonstrated motions, and (iii) ensuring collision-free execution within robot kinematics. To capture the multimodal nature of demonstrations, we extend flow matching to $\SE(3)$ for probabilistic modeling of object trajectories, enabling density-aware imitation that avoids mode collapse. The resulting optimization integrates grasp similarity, trajectory likelihood, and collision penalties into a unified differentiable objective. We validate our approach in both simulation and real-world experiments across diverse real-world manipulation tasks. 

**Abstract (ZH)**: 基于视频演示的人机学习为操纵器提供了规模化替代远程操作或力觉示教的方案，但面对体化差异和关节可行性约束，带来了挑战。为此，我们提出了基于视频演示学习（LfD）框架下的关节流轨迹优化（JFTO）方法，用于抓取姿态生成和物体轨迹模仿，而不是直接模仿人的手部动作，而是将演示视为以物体为中心的指南，平衡以下三个目标：（i）选择可行的抓取姿态，（ii）生成与示范运动一致的物体轨迹，（iii）确保在机器人运动学约束下的无碰撞执行。为了捕获演示的多模态性质，我们将流匹配扩展到$\SE(3)$，以概率建模物体轨迹，使得模仿具有密度感知能力，避免模式崩溃。最终的优化将抓取相似性、轨迹似然性和碰撞惩罚整合到一个统一的不同iable目标中。我们在多种真实世界的操作任务的模拟和实际实验中验证了该方法。 

---
# RuN: Residual Policy for Natural Humanoid Locomotion 

**Title (ZH)**: RuN: 用于自然人形运动的残差策略 

**Authors**: Qingpeng Li, Chengrui Zhu, Yanming Wu, Xin Yuan, Zhen Zhang, Jian Yang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20696)  

**Abstract**: Enabling humanoid robots to achieve natural and dynamic locomotion across a wide range of speeds, including smooth transitions from walking to running, presents a significant challenge. Existing deep reinforcement learning methods typically require the policy to directly track a reference motion, forcing a single policy to simultaneously learn motion imitation, velocity tracking, and stability maintenance. To address this, we introduce RuN, a novel decoupled residual learning framework. RuN decomposes the control task by pairing a pre-trained Conditional Motion Generator, which provides a kinematically natural motion prior, with a reinforcement learning policy that learns a lightweight residual correction to handle dynamical interactions. Experiments in simulation and reality on the Unitree G1 humanoid robot demonstrate that RuN achieves stable, natural gaits and smooth walk-run transitions across a broad velocity range (0-2.5 m/s), outperforming state-of-the-art methods in both training efficiency and final performance. 

**Abstract (ZH)**: 使仿人机器人在广泛的速度范围内实现自然且动态的运动，包括从行走平滑过渡到奔跑，是一项重大挑战。现有深度强化学习方法通常要求策略直接跟踪参考运动，迫使单一策略同时学习运动模仿、速度跟踪和稳定性维护。为了解决这一问题，我们引入了 RuN，这是一种新型的解耦残差学习框架。RuN 通过将一个预训练的条件运动生成器与一个学习轻量级动态修正的强化学习策略配对，来分解控制任务，其中条件运动生成器提供一种动力学自然的运动先验。实验在 Unitree G1 仿人机器人上的模拟和现实环境中表明，RuN 在广泛的速度假区间（0-2.5 m/s）实现了稳定且自然的步伐，并且平滑的走跑过渡，其在训练效率和最终性能方面均优于现有最先进的方法。 

---
# Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation 

**Title (ZH)**: 从单张图像高效构建隐式表面模型以生成运动 

**Authors**: Wei-Teng Chu, Tianyi Zhang, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20681)  

**Abstract**: Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as \emph{NeuS} and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets. 

**Abstract (ZH)**: 基于单张图像构建隐式距离表示的快速图像到神经表面框架 

---
# EEG-Driven AR-Robot System for Zero-Touch Grasping Manipulation 

**Title (ZH)**: 基于EEG的AR机器人系统实现零触觉抓取操作 

**Authors**: Junzhe Wang, Jiarui Xie, Pengfei Hao, Zheng Li, Yi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.20656)  

**Abstract**: Reliable brain-computer interface (BCI) control of robots provides an intuitive and accessible means of human-robot interaction, particularly valuable for individuals with motor impairments. However, existing BCI-Robot systems face major limitations: electroencephalography (EEG) signals are noisy and unstable, target selection is often predefined and inflexible, and most studies remain restricted to simulation without closed-loop validation. These issues hinder real-world deployment in assistive scenarios. To address them, we propose a closed-loop BCI-AR-Robot system that integrates motor imagery (MI)-based EEG decoding, augmented reality (AR) neurofeedback, and robotic grasping for zero-touch operation. A 14-channel EEG headset enabled individualized MI calibration, a smartphone-based AR interface supported multi-target navigation with direction-congruent feedback to enhance stability, and the robotic arm combined decision outputs with vision-based pose estimation for autonomous grasping. Experiments are conducted to validate the framework: MI training achieved 93.1 percent accuracy with an average information transfer rate (ITR) of 14.8 bit/min; AR neurofeedback significantly improved sustained control (SCI = 0.210) and achieved the highest ITR (21.3 bit/min) compared with static, sham, and no-AR baselines; and closed-loop grasping achieved a 97.2 percent success rate with good efficiency and strong user-reported control. These results show that AR feedback substantially stabilizes EEG-based control and that the proposed framework enables robust zero-touch grasping, advancing assistive robotic applications and future modes of human-robot interaction. 

**Abstract (ZH)**: 可靠的大脑-计算机接口（BCI）控制的机器人提供了一种直观且易于使用的交互方式，尤其对于运动障碍患者非常有价值。然而，现有的BCI-机器人系统面临重大限制：脑电图（EEG）信号噪声大且不稳定，目标选择通常预定义且灵活性有限，多数研究仍局限于模拟而未能进行闭环验证。这些问题阻碍了在辅助场景中的实际部署。为解决这些问题，我们提出了一种闭环BCI-AR-机器人系统，该系统结合了基于 motor imagery（MI）的 EEG 解码、增强现实（AR）神经反馈以及机械臂抓取操作，实现了零接触操作。14通道EEG头盔实现了个性化的MI校准，基于智能手机的AR界面支持多目标导航并提供了方向一致的反馈以增强稳定性，机械臂结合决策输出与基于视觉的姿态估计实现自主抓取。进行了实验验证该框架：MI训练的准确率达到93.1%，平均信息传输速率为14.8 bit/min；AR神经反馈显著提高了持续控制能力（SCI = 0.210），并实现了最高的信息传输速率（21.3 bit/min），优于静态、假对照和无AR基线；闭环抓取成功率达到97.2%，具有良好的效率和强大的用户报告的控制能力。这些结果表明，AR反馈显著稳定了基于EEG的控制，并且所提出框架实现了稳健的零接触抓取，推动了辅助机器人应用和未来的人机交互模式的发展。 

---
# Suction Leap-Hand: Suction Cups on a Multi-fingered Hand Enable Embodied Dexterity and In-Hand Teleoperation 

**Title (ZH)**: 吸力Leap-Hand：多指手上的吸盘使机器人具备实体灵活性和手持远程操作能力 

**Authors**: Sun Zhaole, Xiaofeng Mao, Jihong Zhu, Yuanlong Zhang, Robert B. Fisher  

**Link**: [PDF](https://arxiv.org/pdf/2509.20646)  

**Abstract**: Dexterous in-hand manipulation remains a foundational challenge in robotics, with progress often constrained by the prevailing paradigm of imitating the human hand. This anthropomorphic approach creates two critical barriers: 1) it limits robotic capabilities to tasks humans can already perform, and 2) it makes data collection for learning-based methods exceedingly difficult. Both challenges are caused by traditional force-closure which requires coordinating complex, multi-point contacts based on friction, normal force, and gravity to grasp an object. This makes teleoperated demonstrations unstable and amplifies the sim-to-real gap for reinforcement learning. In this work, we propose a paradigm shift: moving away from replicating human mechanics toward the design of novel robotic embodiments. We introduce the \textbf{S}uction \textbf{Leap}-Hand (SLeap Hand), a multi-fingered hand featuring integrated fingertip suction cups that realize a new form of suction-enabled dexterity. By replacing complex force-closure grasps with stable, single-point adhesion, our design fundamentally simplifies in-hand teleoperation and facilitates the collection of high-quality demonstration data. More importantly, this suction-based embodiment unlocks a new class of dexterous skills that are difficult or even impossible for the human hand, such as one-handed paper cutting and in-hand writing. Our work demonstrates that by moving beyond anthropomorphic constraints, novel embodiments can not only lower the barrier for collecting robust manipulation data but also enable the stable, single-handed completion of tasks that would typically require two human hands. Our webpage is this https URL. 

**Abstract (ZH)**: 在手灵巧操作仍然是机器人技术中的一个基础挑战，进展往往受限于模仿人类手部的主流范式。这一类比人类的手部方法创造了两个关键障碍：1）它限制了机器人的能力，仅限于人类已经能完成的任务；2）它使基于学习的方法的数据采集极其困难。这两个挑战都是由传统的力闭合引起的，传统力闭合要求基于摩擦、法向力和重力协调复杂的多点接触才能抓取物体。这使得遥控演示变得不稳定，并放大了模拟到现实的差距。在本文中，我们提出了范式的转变：从复制人类力学转向设计新颖的机器人实体。我们介绍了Suction Leap-Hand（SLeap Hand），这是一种多功能的手部，配备了内置的指尖吸盘，实现了新的吸盘助力灵巧操作形式。通过用稳定的一点粘附替代复杂的力闭合抓取，我们设计从根本上简化了在手遥控操纵，并促进了高质量演示数据的收集。更重要的是，基于吸盘的设计解锁了一类新的灵巧技能，这些技能对于人类手部来说是难以实现甚至是不可能实现的，例如单手剪纸和在手写字。我们的研究显示，通过超越类人学限制，新颖的实体不仅可以降低收集稳健操作数据的门槛，还能使单手完成通常需要两只手来完成的任务变得稳定。我们的网页地址是：this https URL。 

---
# Learning Terrain-Specialized Policies for Adaptive Locomotion in Challenging Environments 

**Title (ZH)**: 学习适用于挑战性环境的terrain-specialized策略以实现自适应运动控制 

**Authors**: Matheus P. Angarola, Francisco Affonso, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.20635)  

**Abstract**: Legged robots must exhibit robust and agile locomotion across diverse, unstructured terrains, a challenge exacerbated under blind locomotion settings where terrain information is unavailable. This work introduces a hierarchical reinforcement learning framework that leverages terrain-specialized policies and curriculum learning to enhance agility and tracking performance in complex environments. We validated our method on simulation, where our approach outperforms a generalist policy by up to 16% in success rate and achieves lower tracking errors as the velocity target increases, particularly on low-friction and discontinuous terrains, demonstrating superior adaptability and robustness across mixed-terrain scenarios. 

**Abstract (ZH)**: 腿式机器人必须在多样化的无结构地形上表现出 robust 和敏捷的运动能力，而在地形信息不可用的盲运动情况下，这一挑战更加严峻。本研究引入了一种分层强化学习框架，该框架利用地形专业化策略和 Curriculum 学习来提高复杂环境中敏捷性和跟踪性能。我们在模拟中验证了该方法，结果显示，在成功率和跟踪误差方面，我们的方法分别比通用策略高出 16% 和在高速目标下表现更好，特别是在低摩擦和不连续地形上，显示出更强的适应性和鲁棒性，适用于混合地形场景。 

---
# Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation 

**Title (ZH)**: 潜在激活编辑：学习政策的推理时 refinement 以实现更安全的多机器人导航 

**Authors**: Satyajeet Das, Darren Chiu, Zhehui Huang, Lars Lindemann, Gaurav S. Sukhatme  

**Link**: [PDF](https://arxiv.org/pdf/2509.20623)  

**Abstract**: Reinforcement learning has enabled significant progress in complex domains such as coordinating and navigating multiple quadrotors. However, even well-trained policies remain vulnerable to collisions in obstacle-rich environments. Addressing these infrequent but critical safety failures through retraining or fine-tuning is costly and risks degrading previously learned skills. Inspired by activation steering in large language models and latent editing in computer vision, we introduce a framework for inference-time Latent Activation Editing (LAE) that refines the behavior of pre-trained policies without modifying their weights or architecture. The framework operates in two stages: (i) an online classifier monitors intermediate activations to detect states associated with undesired behaviors, and (ii) an activation editing module that selectively modifies flagged activations to shift the policy towards safer regimes. In this work, we focus on improving safety in multi-quadrotor navigation. We hypothesize that amplifying a policy's internal perception of risk can induce safer behaviors. We instantiate this idea through a latent collision world model trained to predict future pre-collision activations, thereby prompting earlier and more cautious avoidance responses. Extensive simulations and real-world Crazyflie experiments demonstrate that LAE achieves statistically significant reduction in collisions (nearly 90% fewer cumulative collisions compared to the unedited baseline) and substantially increases the fraction of collision-free trajectories, while preserving task completion. More broadly, our results establish LAE as a lightweight paradigm, feasible on resource-constrained hardware, for post-deployment refinement of learned robot policies. 

**Abstract (ZH)**: 强化学习在协调和导航多个四旋翼飞行器等复杂领域取得了显著进展。然而，即使训练良好的策略在障碍密集环境中仍易发生碰撞。通过重新训练或微调来解决这些罕见但关键的安全故障成本高昂，并可能削弱之前学到的技能。受大规模语言模型的激活转向和计算机视觉中的潜在编辑启发，我们提出了一个推理时潜在激活编辑（LAE）框架，该框架在不修改预训练策略的权重或架构的情况下，对其行为进行细化。该框架分两个阶段进行：（i）在线分类器监测中间激活以检测与不良行为相关的状态，（ii）一个激活编辑模块选择性地修改标记的激活，引导策略向更安全的区域转变。在本工作中，我们专注于改进多四旋翼飞行器导航中的安全性。我们假设增强策略对风险的内部感知可以诱发更安全的行为。我们通过训练一个潜在碰撞世界模型来预测预碰撞激活，从而促使更早和更加谨慎的规避响应来实现这一想法。大量模拟和现实世界的Crazyflie实验表明，LAE能显著减少碰撞（与未编辑基线相比，累积碰撞次数减少近90%）并大幅增加无碰撞轨迹的比例，同时保持任务完成。更广泛而言，我们的结果显示LAE是一种轻量级的范式，可以在资源受限的硬件上实现已部署后强化学习的机器人策略。 

---
# Selective Progress-Aware Querying for Human-in-the-Loop Reinforcement Learning 

**Title (ZH)**: 面向人类在环强化学习的选择性进步感知查询 

**Authors**: Anujith Muraleedharan, Anamika J H  

**Link**: [PDF](https://arxiv.org/pdf/2509.20541)  

**Abstract**: Human feedback can greatly accelerate robot learning, but in real-world settings, such feedback is costly and limited. Existing human-in-the-loop reinforcement learning (HiL-RL) methods often assume abundant feedback, limiting their practicality for physical robot deployment. In this work, we introduce SPARQ, a progress-aware query policy that requests feedback only when learning stagnates or worsens, thereby reducing unnecessary oracle calls. We evaluate SPARQ on a simulated UR5 cube-picking task in PyBullet, comparing against three baselines: no feedback, random querying, and always querying. Our experiments show that SPARQ achieves near-perfect task success, matching the performance of always querying while consuming about half the feedback budget. It also provides more stable and efficient learning than random querying, and significantly improves over training without feedback. These findings suggest that selective, progress-based query strategies can make HiL-RL more efficient and scalable for robots operating under realistic human effort constraints. 

**Abstract (ZH)**: 基于进展的查询策略SPARQ可加速受人性约束的强化学习 

---
# Action-Informed Estimation and Planning: Clearing Clutter on Staircases via Quadrupedal Pedipulation 

**Title (ZH)**: 基于行动指导的估计与规划：通过四足步行清除楼梯上的障碍物 

**Authors**: Prasanna Sriganesh, Barath Satheeshkumar, Anushree Sabnis, Matthew Travers  

**Link**: [PDF](https://arxiv.org/pdf/2509.20516)  

**Abstract**: For robots to operate autonomously in densely cluttered environments, they must reason about and potentially physically interact with obstacles to clear a path. Safely clearing a path on challenging terrain, such as a cluttered staircase, requires controlled interaction. For example, a quadrupedal robot that pushes objects out of the way with one leg while maintaining a stable stance with its three other legs. However, tightly coupled physical actions, such as one-legged pushing, create new constraints on the system that can be difficult to predict at design time. In this work, we present a new method that addresses one such constraint, wherein the object being pushed by a quadrupedal robot with one of its legs becomes occluded from the robot's sensors during manipulation. To address this challenge, we present a tightly coupled perception-action framework that enables the robot to perceive clutter, reason about feasible push paths, and execute the clearing maneuver. Our core contribution is an interaction-aware state estimation loop that uses proprioceptive feedback regarding foot contact and leg position to predict an object's displacement during the occlusion. This prediction guides the perception system to robustly re-detect the object after the interaction, closing the loop between action and sensing to enable accurate tracking even after partial pushes. Using this feedback allows the robot to learn from physical outcomes, reclassifying an object as immovable if a push fails due to it being too heavy. We present results of implementing our approach on a Boston Dynamics Spot robot that show our interaction-aware approach achieves higher task success rates and tracking accuracy in pushing objects on stairs compared to open-loop baselines. 

**Abstract (ZH)**: 自主导航于密集障碍环境中的机器人必须推理并可能物理互动以清除路径。在挑战性地形（如杂乱的楼梯）上安全地清除路径需要受控互动。例如，四足机器人利用一只腿推动物体，同时用其他三只腿保持稳定的姿态。然而，紧密耦合的物理动作（如单腿推动）会给系统带来新的约束，这些约束在设计时难以预测。在本工作中，我们提出了一种新方法，以解决其中一种约束：四足机器人用一只腿推动物体时，该物体会在操作过程中被遮挡，从而无法被机器人的传感器检测到。为了应对这一挑战，我们提出了一种紧密耦合的感知-动作框架，使机器人能够感知障碍物、推理可行的推动路径，并执行清除操作。我们的核心贡献是一种感知-互动感知估计回路，它利用有关足部接触和腿部位置的本体感受反馈来预测物体在遮挡期间的位移。这一预测指导感知系统在互动后 robust 地重新检测物体，从而在动作与感知之间形成闭环，即使在部分推动后也能实现精确跟踪。利用这种反馈，机器人可以从物理结果中学习，如果推动失败是因为物体太重则重新分类物体为无法移动。我们展示了在Boston Dynamics Spot机器人的实现结果，表明我们的感知-互动方法在楼梯上推动物体时比开环基线方法具有更高的任务成功率和跟踪准确性。 

---
# Boosting Zero-Shot VLN via Abstract Obstacle Map-Based Waypoint Prediction with TopoGraph-and-VisitInfo-Aware Prompting 

**Title (ZH)**: 基于抽象障碍地图的航点预测以及拓扑图和到访信息感知提示增强零样本视觉语言导航 

**Authors**: Boqi Li, Siyuan Li, Weiyi Wang, Anran Li, Zhong Cao, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20499)  

**Abstract**: With the rapid progress of foundation models and robotics, vision-language navigation (VLN) has emerged as a key task for embodied agents with broad practical applications. We address VLN in continuous environments, a particularly challenging setting where an agent must jointly interpret natural language instructions, perceive its surroundings, and plan low-level actions. We propose a zero-shot framework that integrates a simplified yet effective waypoint predictor with a multimodal large language model (MLLM). The predictor operates on an abstract obstacle map, producing linearly reachable waypoints, which are incorporated into a dynamically updated topological graph with explicit visitation records. The graph and visitation information are encoded into the prompt, enabling reasoning over both spatial structure and exploration history to encourage exploration and equip MLLM with local path planning for error correction. Extensive experiments on R2R-CE and RxR-CE show that our method achieves state-of-the-art zero-shot performance, with success rates of 41% and 36%, respectively, outperforming prior state-of-the-art methods. 

**Abstract (ZH)**: 基于基础模型和机器人技术的快速发展，视觉-语言导航（VLN）已成为具备广泛实用前景的体感Agent的关键任务。我们研究在连续环境中进行VLN，这是一种特别具有挑战性的设置，其中代理必须联合解释自然语言指令、感知周围环境并规划低层动作。我们提出了一种零样本框架，该框架结合了一个简化但有效的 waypoints 预测器和多模态大型语言模型（MLLM）。预测器基于抽象障碍地图工作，生成线性可达的waypoints，并将其整合到一个动态更新的拓扑图中，该图包含明确的访问记录。图和访问信息被编码到提示中，以支持对空间结构和探索历史的推理，从而鼓励探索，并为MLLM提供局部路径规划以进行错误校正。在R2R-CE和RxR-CE上的广泛实验表明，我们的方法实现了最先进的零样本性能，分别取得了41%和36%的成功率，优于先前的最佳方法。 

---
# Wonder Wins Ways: Curiosity-Driven Exploration through Multi-Agent Contextual Calibration 

**Title (ZH)**: 好奇心驱动的多agent情境校准探索：奇伟之路 

**Authors**: Yiyuan Pan, Zhe Liu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20648)  

**Abstract**: Autonomous exploration in complex multi-agent reinforcement learning (MARL) with sparse rewards critically depends on providing agents with effective intrinsic motivation. While artificial curiosity offers a powerful self-supervised signal, it often confuses environmental stochasticity with meaningful novelty. Moreover, existing curiosity mechanisms exhibit a uniform novelty bias, treating all unexpected observations equally. However, peer behavior novelty, which encode latent task dynamics, are often overlooked, resulting in suboptimal exploration in decentralized, communication-free MARL settings. To this end, inspired by how human children adaptively calibrate their own exploratory behaviors via observing peers, we propose a novel approach to enhance multi-agent exploration. We introduce CERMIC, a principled framework that empowers agents to robustly filter noisy surprise signals and guide exploration by dynamically calibrating their intrinsic curiosity with inferred multi-agent context. Additionally, CERMIC generates theoretically-grounded intrinsic rewards, encouraging agents to explore state transitions with high information gain. We evaluate CERMIC on benchmark suites including VMAS, Meltingpot, and SMACv2. Empirical results demonstrate that exploration with CERMIC significantly outperforms SoTA algorithms in sparse-reward environments. 

**Abstract (ZH)**: 自主探索在复杂多智能体强化学习（MARL）中的应用：稀疏奖励下的有效内在动机对于自主探索至关重要。基于观察同伴行为，我们提出一种新颖方法以增强多智能体探索。我们引入CERMIC，一种原理性的框架，使智能体能够 robust 地过滤嘈杂的惊讶信号，并通过动态校准内在好奇心与推断出的多智能体上下文来引导探索。此外，CERMIC 生成理论依据内在奖励，鼓励智能体探索具有高信息增益的状态转换。我们将在 VMAS、Meltingpot 和 SMACv2 等基准套件上评估 CERMIC。实验证明，在稀疏奖励环境中，使用 CERMIC 进行探索显著优于现有最佳算法。 

---
# Large Pre-Trained Models for Bimanual Manipulation in 3D 

**Title (ZH)**: 大型预训练模型在3D双手操作中的应用 

**Authors**: Hanna Yurchyk, Wei-Di Chang, Gregory Dudek, David Meger  

**Link**: [PDF](https://arxiv.org/pdf/2509.20579)  

**Abstract**: We investigate the integration of attention maps from a pre-trained Vision Transformer into voxel representations to enhance bimanual robotic manipulation. Specifically, we extract attention maps from DINOv2, a self-supervised ViT model, and interpret them as pixel-level saliency scores over RGB images. These maps are lifted into a 3D voxel grid, resulting in voxel-level semantic cues that are incorporated into a behavior cloning policy. When integrated into a state-of-the-art voxel-based policy, our attention-guided featurization yields an average absolute improvement of 8.2% and a relative gain of 21.9% across all tasks in the RLBench bimanual benchmark. 

**Abstract (ZH)**: 我们将注意力图集成到预训练的视觉变换器中的体素表示中，以增强双手机器人操作。具体来说，我们从自监督ViT模型DINOv2中提取注意力图，并将其解释为RGB图像上的像素级显著性评分。这些图被提升到3D体素网格中，产生体素级语义线索，这些线索被纳入行为克隆策略中。当集成到最先进的基于体素的策略中时，我们的注意力引导特征化在RLBench双手基准中的所有任务上平均绝对改善了8.2%，相对增益为21.9%。 

---
# SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent 

**Title (ZH)**: SceneWeaver：一劳永逸的3D场景合成扩展自反代理 

**Authors**: Yandan Yang, Baoxiong Jia, Shujie Zhang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20414)  

**Abstract**: Indoor scene synthesis has become increasingly important with the rise of Embodied AI, which requires 3D environments that are not only visually realistic but also physically plausible and functionally diverse. While recent approaches have advanced visual fidelity, they often remain constrained to fixed scene categories, lack sufficient object-level detail and physical consistency, and struggle to align with complex user instructions. In this work, we present SceneWeaver, a reflective agentic framework that unifies diverse scene synthesis paradigms through tool-based iterative refinement. At its core, SceneWeaver employs a language model-based planner to select from a suite of extensible scene generation tools, ranging from data-driven generative models to visual- and LLM-based methods, guided by self-evaluation of physical plausibility, visual realism, and semantic alignment with user input. This closed-loop reason-act-reflect design enables the agent to identify semantic inconsistencies, invoke targeted tools, and update the environment over successive iterations. Extensive experiments on both common and open-vocabulary room types demonstrate that SceneWeaver not only outperforms prior methods on physical, visual, and semantic metrics, but also generalizes effectively to complex scenes with diverse instructions, marking a step toward general-purpose 3D environment generation. Project website: this https URL. 

**Abstract (ZH)**: 室内场景合成随着沉浸式AI的兴起变得越来越重要，这需要既具备视觉逼真度又物理合理且功能多样的3D环境。虽然近期的方法在视觉保真度方面取得了进展，但它们往往局限于固定的场景类别，缺乏足够的物体级细节和物理一致性，并且难以与复杂的用户指令对齐。在本文中，我们提出了SceneWeaver，这是一种反思性自主框架，通过基于工具的迭代优化统一了多种场景合成范式。其核心是SceneWeaver采用基于语言模型的规划器，选择从数据驱动生成模型到基于视觉和LLM的方法等多种可扩展的场景生成工具，这些选择受到对物理合理性、视觉逼真度和语义与用户输入的一致性的自我评估指导。这种闭环的思考-行动-反思设计使代理能够识别语义不一致，激活特定的工具，并在连续迭代中更新环境。在对常见和开放式词汇房间类型的广泛实验中，SceneWeaver不仅在物理、视觉和语义指标上优于先前的方法，而且能够有效地应用于具有多种指令的复杂场景，朝着通用3D环境生成迈进。项目网站：this https URL。 

---
# SGAligner++: Cross-Modal Language-Aided 3D Scene Graph Alignment 

**Title (ZH)**: SGAligner++: 跨模态语言辅助的3D场景图对齐 

**Authors**: Binod Singh, Sayan Deb Sarkar, Iro Armeni  

**Link**: [PDF](https://arxiv.org/pdf/2509.20401)  

**Abstract**: Aligning 3D scene graphs is a crucial initial step for several applications in robot navigation and embodied perception. Current methods in 3D scene graph alignment often rely on single-modality point cloud data and struggle with incomplete or noisy input. We introduce SGAligner++, a cross-modal, language-aided framework for 3D scene graph alignment. Our method addresses the challenge of aligning partially overlapping scene observations across heterogeneous modalities by learning a unified joint embedding space, enabling accurate alignment even under low-overlap conditions and sensor noise. By employing lightweight unimodal encoders and attention-based fusion, SGAligner++ enhances scene understanding for tasks such as visual localization, 3D reconstruction, and navigation, while ensuring scalability and minimal computational overhead. Extensive evaluations on real-world datasets demonstrate that SGAligner++ outperforms state-of-the-art methods by up to 40% on noisy real-world reconstructions, while enabling cross-modal generalization. 

**Abstract (ZH)**: 跨模态语言辅助的3D场景图对齐 

---
# Embodied Representation Alignment with Mirror Neurons 

**Title (ZH)**: 镜像神经元驱动的体态表示对齐 

**Authors**: Wentao Zhu, Zhining Zhang, Yuwei Ren, Yin Huang, Hao Xu, Yizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21136)  

**Abstract**: Mirror neurons are a class of neurons that activate both when an individual observes an action and when they perform the same action. This mechanism reveals a fundamental interplay between action understanding and embodied execution, suggesting that these two abilities are inherently connected. Nonetheless, existing machine learning methods largely overlook this interplay, treating these abilities as separate tasks. In this study, we provide a unified perspective in modeling them through the lens of representation learning. We first observe that their intermediate representations spontaneously align. Inspired by mirror neurons, we further introduce an approach that explicitly aligns the representations of observed and executed actions. Specifically, we employ two linear layers to map the representations to a shared latent space, where contrastive learning enforces the alignment of corresponding representations, effectively maximizing their mutual information. Experiments demonstrate that this simple approach fosters mutual synergy between the two tasks, effectively improving representation quality and generalization. 

**Abstract (ZH)**: 镜像神经元是一种在个体观察一个动作和执行相同动作时都会激活的神经元。这一机制揭示了动作理解和体现执行之间的基本互动，表明这两种能力本质上是相连的。尽管现有的机器学习方法大多忽视了这种互动，将这些能力视为单独的任务。在本研究中，我们通过表示学习的角度提供了一种统一的建模视角。我们首先观察到它们的中间表示会自发对齐。受镜像神经元的启发，我们进一步引入了一种显式对齐观察动作和执行动作表示的方法。具体而言，我们使用两个线性层将表示映射到共享的潜在空间，在此空间中，对比学习促使相应的表示对齐，有效地最大化它们的互信息。实验表明，这种简单的做法促进了两项任务之间的相互协同作用，有效提高了表示质量和泛化能力。 

---
# Meta-Memory: Retrieving and Integrating Semantic-Spatial Memories for Robot Spatial Reasoning 

**Title (ZH)**: 元记忆：检索和整合语义-空间记忆以进行机器人空间推理 

**Authors**: Yufan Mao, Hanjing Ye, Wenlong Dong, Chengjie Zhang, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20754)  

**Abstract**: Navigating complex environments requires robots to effectively store observations as memories and leverage them to answer human queries about spatial locations, which is a critical yet underexplored research challenge. While prior work has made progress in constructing robotic memory, few have addressed the principled mechanisms needed for efficient memory retrieval and integration. To bridge this gap, we propose Meta-Memory, a large language model (LLM)-driven agent that constructs a high-density memory representation of the environment. The key innovation of Meta-Memory lies in its capacity to retrieve and integrate relevant memories through joint reasoning over semantic and spatial modalities in response to natural language location queries, thereby empowering robots with robust and accurate spatial reasoning capabilities. To evaluate its performance, we introduce SpaceLocQA, a large-scale dataset encompassing diverse real-world spatial question-answering scenarios. Experimental results show that Meta-Memory significantly outperforms state-of-the-art methods on both the SpaceLocQA and the public NaVQA benchmarks. Furthermore, we successfully deployed Meta-Memory on real-world robotic platforms, demonstrating its practical utility in complex environments. Project page: this https URL . 

**Abstract (ZH)**: 导航复杂环境要求机器人有效地存储观察作为记忆，并利用这些记忆回答关于空间位置的人类查询，这是一个关键但尚未充分探索的研究挑战。尽管之前的研究所在这方面取得进展，但很少有人解决高效记忆检索和集成的原则机制。为弥补这一差距，我们提出了一种元记忆（Meta-Memory），这是一种由大规模语言模型（LLM）驱动的代理，能够构建环境的高密度记忆表示。元记忆的关键创新在于其通过联合推理语义和空间模态来检索和整合相关记忆的能力，以响应自然语言的空间位置查询，从而赋予机器人强大的空间推理能力。为了评估其性能，我们引入了一种大规模数据集SpaceLocQA，涵盖了多种真实世界的空间问答场景。实验结果表明，元记忆在SpaceLocQA和现有的NaVQA基准测试中显著优于最先进的方法。此外，我们成功将元记忆部署到实际的机器人平台上，证明了其在复杂环境中的实用价值。项目页面：this https URL。 

---
# Fairy: Interactive Mobile Assistant to Real-world Tasks via LMM-based Multi-agent 

**Title (ZH)**: Fairy: 基于LMM的多agent交互式移动助手用于现实世界任务 

**Authors**: Jiazheng Sun, Te Yang, Jiayang Niu, Mingxuan Li, Yongyong Lu, Ruimeng Yang, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20729)  

**Abstract**: Large multi-modal models (LMMs) have advanced mobile GUI agents. However, existing methods struggle with real-world scenarios involving diverse app interfaces and evolving user needs. End-to-end methods relying on model's commonsense often fail on long-tail apps, and agents without user interaction act unilaterally, harming user experience. To address these limitations, we propose Fairy, an interactive multi-agent mobile assistant capable of continuously accumulating app knowledge and self-evolving during usage. Fairy enables cross-app collaboration, interactive execution, and continual learning through three core modules:(i) a Global Task Planner that decomposes user tasks into sub-tasks from a cross-app view; (ii) an App-Level Executor that refines sub-tasks into steps and actions based on long- and short-term memory, achieving precise execution and user interaction via four core agents operating in dual loops; and (iii) a Self-Learner that consolidates execution experience into App Map and Tricks. To evaluate Fairy, we introduce RealMobile-Eval, a real-world benchmark with a comprehensive metric suite, and LMM-based agents for automated scoring. Experiments show that Fairy with GPT-4o backbone outperforms the previous SoTA by improving user requirement completion by 33.7% and reducing redundant steps by 58.5%, showing the effectiveness of its interaction and self-learning. 

**Abstract (ZH)**: 大型多模态模型（LMMs）已推进了移动GUI代理的发展。然而，现有的方法在处理涉及多样化应用界面和不断变化的用户需求的现实场景时存在困难。依赖模型常识的端到端方法在长尾应用上往往失败，而不与用户互动的代理会单方面行动，损害用户体验。为解决这些限制，我们提出Fairy，一个具备在使用过程中连续积累应用知识和自我进化的交互多代理移动助手。Fairy通过三个核心模块实现跨应用协作、互动执行和持续学习：(i) 全局任务规划器，从跨应用视角分解用户任务；(ii) 应用级执行器，基于长期和短期记忆细化子任务为步骤和行动，在双环中通过四个核心代理实现精确执行和用户互动；和(iii) 自我学习器，将执行经验整合为应用地图和技巧。为了评估Fairy，我们引入RealMobile-Eval，一个包含全面指标套件的真实世界基准，并使用基于LMM的代理进行自动化评分。实验结果显示，以GPT-4o为骨干的Fairy在满足用户需求方面比之前的最佳方案提高了33.7%，减少了58.5%的冗余步骤，证明了其互动和自我学习的有效性。 

---
# A Compound Classification System Based on Fuzzy Relations Applied to the Noise-Tolerant Control of a Bionic Hand via EMG Signal Recognition 

**Title (ZH)**: 基于模糊关系的复合分类系统及其在 Electromyography 信号识别指导下的容噪仿生手控制 

**Authors**: Pawel Trajdos, Marek Kurzynski  

**Link**: [PDF](https://arxiv.org/pdf/2509.20523)  

**Abstract**: Modern anthropomorphic upper limb bioprostheses are typically controlled by electromyographic (EMG) biosignals using a pattern recognition scheme. Unfortunately, there are many factors originating from the human source of objects to be classified and from the human-prosthesis interface that make it difficult to obtain an acceptable classification quality. One of these factors is the high susceptibility of biosignals to contamination, which can considerably reduce the quality of classification of a recognition system.
In the paper, the authors propose a new recognition system intended for EMG based control of the hand prosthesis with detection of contaminated biosignals in order to mitigate the adverse effect of contaminations. The system consists of two ensembles: the set of one-class classifiers (OCC) to assess the degree of contamination of individual channels and the ensemble of K-nearest neighbours (KNN) classifier to recognise the patient's intent. For all recognition systems, an original, coherent fuzzy model was developed, which allows the use of a uniform soft (fuzzy) decision scheme throughout the recognition process. The experimental evaluation was conducted using real biosignals from a public repository. The goal was to provide an experimental comparative analysis of the parameters and procedures of the developed method on which the quality of the recognition system depends. The proposed fuzzy recognition system was also compared with similar systems described in the literature. 

**Abstract (ZH)**: 现代类人上肢生物假肢通常通过模式识别方案利用肌电图（EMG）生物信号进行控制。由于来自人类对象和人类-假肢接口的各种因素，很难获得可接受的分类质量。其中一个因素是生物信号对污染的高度敏感性，这可以显著降低识别系统的分类质量。

在本文中，作者提出了一种新的识别系统，旨在基于EMG控制手部假肢并检测受污染的生物信号，以减轻污染的不良影响。该系统由两个集成部分组成：一类分类器（OCC）集合作为评估各个通道污染程度的方法，以及K邻近邻居（KNN）分类器集合作为识别患者意图的方法。对于所有识别系统，开发了一个原创的一致模糊模型，这使得在整个识别过程中可以使用统一的软（模糊）决策方案。实验评估使用了一个公开数据集中的真实生物信号进行。目标是提供一种实验性的比较分析，比较所开发方法的参数和流程，这些参数和流程决定了识别系统的质量。此外，提出的模糊识别系统还与文献中描述的类似系统进行了比较。 

---
# Learning to Look: Cognitive Attention Alignment with Vision-Language Models 

**Title (ZH)**: 学习凝视：认知注意力与视觉语言模型的对齐 

**Authors**: Ryan L. Yang, Dipkamal Bhusal, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21247)  

**Abstract**: Convolutional Neural Networks (CNNs) frequently "cheat" by exploiting superficial correlations, raising concerns about whether they make predictions for the right reasons. Inspired by cognitive science, which highlights the role of attention in robust human perception, recent methods have sought to guide model attention using concept-based supervision and explanation regularization. However, these techniques depend on labor-intensive, expert-provided annotations, limiting their scalability. We propose a scalable framework that leverages vision-language models to automatically generate semantic attention maps using natural language prompts. By introducing an auxiliary loss that aligns CNN attention with these language-guided maps, our approach promotes more reliable and cognitively plausible decision-making without manual annotation. Experiments on challenging datasets, ColoredMNIST and DecoyMNIST, show that our method achieves state-of-the-art performance on ColorMNIST and remains competitive with annotation-heavy baselines on DecoyMNIST, demonstrating improved generalization, reduced shortcut reliance, and model attention that better reflects human intuition. 

**Abstract (ZH)**: 卷积神经网络（CNNs）经常通过利用表面相关性“作弊”，这引发了对其预测是否基于正确原因的质疑。受认知科学的启发，该科学强调注意在 robust 人类知觉中的作用，近年来的方法试图使用基于概念的监督和解释正则化来引导模型注意。然而，这些技术依赖于劳动密集型的、由专家提供的注释，限制了其可扩展性。我们提出了一种可扩展的框架，利用vision-language模型自动生成基于自然语言提示的语义注意图。通过引入一个辅助损失来使CNN注意与这些语言引导的图对齐，我们的方法能在无需手动注释的情况下促进更可靠且合乎认知的决策。在 ColoredMNIST 和 DecoyMNIST 等具有挑战性的数据集上的实验表明，我们的方法在 ColoredMNIST 上达到了最先进的性能，同时在 DecoyMNIST 上与注释密集型对照组保持竞争力，展示了改进的一般化能力、减少捷径依赖性和更好地反映人类直觉的模型注意。 

---
# Teaching RL Agents to Act Better: VLM as Action Advisor for Online Reinforcement Learning 

**Title (ZH)**: 教学 RL 代理更好地行动：大规模语言模型作为在线强化学习的动作顾问 

**Authors**: Xiefeng Wu, Jing Zhao, Shu Zhang, Mingyu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21126)  

**Abstract**: Online reinforcement learning in complex tasks is time-consuming, as massive interaction steps are needed to learn the optimal this http URL-language action (VLA) policies represent a promising direction for solving diverse tasks; however, their performance on low-level control remains limited, and effective deployment often requires task-specific expert demonstrations for fine-tuning. In this paper, we propose \textbf{VARL} (\textbf{V}LM as \textbf{A}ction advisor for online \textbf{R}einforcement \textbf{L}earning), a framework that leverages the domain knowledge of vision-language models (VLMs) to provide action suggestions for reinforcement learning agents. Unlike previous methods, VARL provides action suggestions rather than designing heuristic rewards, thereby guaranteeing unchanged optimality and convergence. The suggested actions increase sample diversity and ultimately improve sample efficiency, especially in sparse-reward tasks. To validate the effectiveness of VARL, we evaluate it across diverse environments and agent settings. Results show that VARL greatly improves sample efficiency without introducing significant computational overhead. These advantages make VARL a general framework for online reinforcement learning and make it feasible to directly apply reinforcement learning from scratch in real-world environments. 

**Abstract (ZH)**: 基于视觉语言模型的在线强化学习框架：VLM作为在线强化学习的动作顾问（VARL） 

---
# The Use of the Simplex Architecture to Enhance Safety in Deep-Learning-Powered Autonomous Systems 

**Title (ZH)**: Simplex架构在增强基于深度学习的自主系统安全性中的应用 

**Authors**: Federico Nesti, Niko Salamini, Mauro Marinoni, Giorgio Maria Cicero, Gabriele Serra, Alessandro Biondi, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21014)  

**Abstract**: Recently, the outstanding performance reached by neural networks in many tasks has led to their deployment in autonomous systems, such as robots and vehicles. However, neural networks are not yet trustworthy, being prone to different types of misbehavior, such as anomalous samples, distribution shifts, adversarial attacks, and other threats. Furthermore, frameworks for accelerating the inference of neural networks typically run on rich operating systems that are less predictable in terms of timing behavior and present larger surfaces for cyber-attacks.
To address these issues, this paper presents a software architecture for enhancing safety, security, and predictability levels of learning-based autonomous systems. It leverages two isolated execution domains, one dedicated to the execution of neural networks under a rich operating system, which is deemed not trustworthy, and one responsible for running safety-critical functions, possibly under a different operating system capable of handling real-time constraints.
Both domains are hosted on the same computing platform and isolated through a type-1 real-time hypervisor enabling fast and predictable inter-domain communication to exchange real-time data. The two domains cooperate to provide a fail-safe mechanism based on a safety monitor, which oversees the state of the system and switches to a simpler but safer backup module, hosted in the safety-critical domain, whenever its behavior is considered untrustworthy.
The effectiveness of the proposed architecture is illustrated by a set of experiments performed on two control systems: a Furuta pendulum and a rover. The results confirm the utility of the fall-back mechanism in preventing faults due to the learning component. 

**Abstract (ZH)**: 增强基于学习的自主系统安全性、安全性和可预测性的软件架构 

---
# Model-Based Reinforcement Learning under Random Observation Delays 

**Title (ZH)**: 基于模型的强化学习在随机观测延迟下的方法 

**Authors**: Armin Karamzade, Kyungmin Kim, JB Lanier, Davide Corsi, Roy Fox  

**Link**: [PDF](https://arxiv.org/pdf/2509.20869)  

**Abstract**: Delays frequently occur in real-world environments, yet standard reinforcement learning (RL) algorithms often assume instantaneous perception of the environment. We study random sensor delays in POMDPs, where observations may arrive out-of-sequence, a setting that has not been previously addressed in RL. We analyze the structure of such delays and demonstrate that naive approaches, such as stacking past observations, are insufficient for reliable performance. To address this, we propose a model-based filtering process that sequentially updates the belief state based on an incoming stream of observations. We then introduce a simple delay-aware framework that incorporates this idea into model-based RL, enabling agents to effectively handle random delays. Applying this framework to Dreamer, we compare our approach to delay-aware baselines developed for MDPs. Our method consistently outperforms these baselines and demonstrates robustness to delay distribution shifts during deployment. Additionally, we present experiments on simulated robotic tasks, comparing our method to common practical heuristics and emphasizing the importance of explicitly modeling observation delays. 

**Abstract (ZH)**: 随机传感器延迟在部分观测马尔可夫决策过程中的建模与处理：基于模型的过滤过程及其应用 

---
# AI-Enabled Crater-Based Navigation for Lunar Mapping 

**Title (ZH)**: 基于陨石坑的月球测绘人工智能导航 

**Authors**: Sofia McLeod, Chee-Kheng Chng, Matthew Rodda, Tat-Jun Chin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20748)  

**Abstract**: Crater-Based Navigation (CBN) uses the ubiquitous impact craters of the Moon observed on images as natural landmarks to determine the six degrees of freedom pose of a spacecraft. To date, CBN has primarily been studied in the context of powered descent and landing. These missions are typically short in duration, with high-frequency imagery captured from a nadir viewpoint over well-lit terrain. In contrast, lunar mapping missions involve sparse, oblique imagery acquired under varying illumination conditions over potentially year-long campaigns, posing significantly greater challenges for pose estimation. We bridge this gap with STELLA - the first end-to-end CBN pipeline for long-duration lunar mapping. STELLA combines a Mask R-CNN-based crater detector, a descriptor-less crater identification module, a robust perspective-n-crater pose solver, and a batch orbit determination back-end. To rigorously test STELLA, we introduce CRESENT-365 - the first public dataset that emulates a year-long lunar mapping mission. Each of its 15,283 images is rendered from high-resolution digital elevation models with SPICE-derived Sun angles and Moon motion, delivering realistic global coverage, illumination cycles, and viewing geometries. Experiments on CRESENT+ and CRESENT-365 show that STELLA maintains metre-level position accuracy and sub-degree attitude accuracy on average across wide ranges of viewing angles, illumination conditions, and lunar latitudes. These results constitute the first comprehensive assessment of CBN in a true lunar mapping setting and inform operational conditions that should be considered for future missions. 

**Abstract (ZH)**: 基于陨石坑的导航（CBN）利用月球图像中普遍存在的陨石坑作为自然 landmarks 确定航天器的六自由度姿态。迄今为止，CBN 主要集中在有动力下降和着陆任务的研究中。这些任务通常持续时间较短，从向下的视角拍摄具有良好照明条件的地形的高频率图像。相比之下，月球制图任务涉及在潜在长达一年的活动中获取稀疏、偏斜视角的图像，并且这些图像在不同的光照条件下拍摄，这给姿态估计带来了更大的挑战。我们通过 STELLA —— 首个用于长时间月球制图的端到端 CBN 管道，填补了这一空白。STELLA 结合了基于 Mask R-CNN 的陨石坑检测器、无描述子陨石坑识别模块、鲁棒的透视-n-陨石坑姿态求解器以及批量轨道确定后端。为了严格测试 STELLA，我们引入了 CRESENT-365 —— 首个模拟一年长期月球制图任务的公共数据集。该数据集的每张图像均由高分辨率数字地形模型渲染，使用 SPICE 计算的太阳角度和月球运动，提供真实的全球覆盖、照明循环和视野几何结构。在 CRESENT+ 和 CRESENT-365 上的实验表明，STELLA 在广角视野、光照条件和月球纬度变化范围内平均保持米级的位置精度和亚度的姿态精度。这些结果是首次全面评估 CBN 在真正月球制图环境中的表现，并为未来的任务应该考虑的操作条件提供了信息。 

---
