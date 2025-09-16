# Learning Contact Dynamics for Control with Action-conditioned Face Interaction Graph Networks 

**Title (ZH)**: 基于动作条件化的面部交互图网络学习接触动力学控制 

**Authors**: Zongyao Yi, Joachim Hertzberg, Martin Atzmueller  

**Link**: [PDF](https://arxiv.org/pdf/2509.12151)  

**Abstract**: We present a learnable physics simulator that provides accurate motion and force-torque prediction of robot end effectors in contact-rich manipulation. The proposed model extends the state-of-the-art GNN-based simulator (FIGNet) with novel node and edge types, enabling action-conditional predictions for control and state estimation tasks. In simulation, the MPC agent using our model matches the performance of the same controller with the ground truth dynamics model in a challenging peg-in-hole task, while in the real-world experiment, our model achieves a 50% improvement in motion prediction accuracy and 3$\times$ increase in force-torque prediction precision over the baseline physics simulator. Source code and data are publicly available. 

**Abstract (ZH)**: 我们提出一种可学习的物理仿真器，能够在接触丰富的操作中提供机器人末端执行器精确的动力学和力- torque 预测。所提出的模型在最新基于GNN的仿真器（FIGNet）的基础上引入了新的节点和边类型，使其能够在控制和状态估计任务中进行基于动作的预测。在仿真中，使用我们模型的MPC代理在一项具有挑战性的孔中穿针任务中的性能与使用真实动力学模型的相同控制器相当；而在实际实验中，我们的模型在运动预测准确性上提高了50%，在力- torque 预测精度上提高了3倍。源代码和数据已公开。 

---
# Embodied Navigation Foundation Model 

**Title (ZH)**: 具身导航基础模型 

**Authors**: Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12129)  

**Abstract**: Navigation is a fundamental capability in embodied AI, representing the intelligence required to perceive and interact within physical environments following language instructions. Despite significant progress in large Vision-Language Models (VLMs), which exhibit remarkable zero-shot performance on general vision-language tasks, their generalization ability in embodied navigation remains largely confined to narrow task settings and embodiment-specific architectures. In this work, we introduce a cross-embodiment and cross-task Navigation Foundation Model (NavFoM), trained on eight million navigation samples that encompass quadrupeds, drones, wheeled robots, and vehicles, and spanning diverse tasks such as vision-and-language navigation, object searching, target tracking, and autonomous driving. NavFoM employs a unified architecture that processes multimodal navigation inputs from varying camera configurations and navigation horizons. To accommodate diverse camera setups and temporal horizons, NavFoM incorporates identifier tokens that embed camera view information of embodiments and the temporal context of tasks. Furthermore, to meet the demands of real-world deployment, NavFoM controls all observation tokens using a dynamically adjusted sampling strategy under a limited token length budget. Extensive evaluations on public benchmarks demonstrate that our model achieves state-of-the-art or highly competitive performance across multiple navigation tasks and embodiments without requiring task-specific fine-tuning. Additional real-world experiments further confirm the strong generalization capability and practical applicability of our approach. 

**Abstract (ZH)**: 跨体态和跨任务导航基础模型（NavFoM）：多模态导航输入的统一处理与灵活观测 

---
# Gesture-Based Robot Control Integrating Mm-wave Radar and Behavior Trees 

**Title (ZH)**: 基于行为树集成毫米波雷达的手势控制机器人 

**Authors**: Yuqing Song, Cesare Tonola, Stefano Savazzi, Sanaz Kianoush, Nicola Pedrocchi, Stephan Sigg  

**Link**: [PDF](https://arxiv.org/pdf/2509.12008)  

**Abstract**: As robots become increasingly prevalent in both homes and industrial settings, the demand for intuitive and efficient human-machine interaction continues to rise. Gesture recognition offers an intuitive control method that does not require physical contact with devices and can be implemented using various sensing technologies. Wireless solutions are particularly flexible and minimally invasive. While camera-based vision systems are commonly used, they often raise privacy concerns and can struggle in complex or poorly lit environments. In contrast, radar sensing preserves privacy, is robust to occlusions and lighting, and provides rich spatial data such as distance, relative velocity, and angle. We present a gesture-controlled robotic arm using mm-wave radar for reliable, contactless motion recognition. Nine gestures are recognized and mapped to real-time commands with precision. Case studies are conducted to demonstrate the system practicality, performance and reliability for gesture-based robotic manipulation. Unlike prior work that treats gesture recognition and robotic control separately, our system unifies both into a real-time pipeline for seamless, contactless human-robot interaction. 

**Abstract (ZH)**: 随着机器人在家庭和工业环境中的应用日益广泛，对直观高效的人机交互需求不断增加。手势识别提供了一种无需物理接触设备的直观控制方法，可以利用各种传感技术实现。无线解决方案尤其灵活且微创。尽管基于摄像头的视觉系统被广泛使用，但它们往往引发隐私问题，并且在复杂或照明不良的环境中表现不佳。相比之下，雷达传感可以保护隐私，对遮挡和照明具有鲁棒性，并能提供丰富的空间数据，如距离、相对速度和角度。我们提出了一种使用毫米波雷达的手势控制机械臂，以实现可靠的无接触运动识别。该系统可以识别九种手势，并将它们精确映射到实时命令。我们进行了案例研究，以展示该系统在基于手势的机器人操作中的实用性、性能和可靠性。与以往单独处理手势识别和机器人控制的工作不同，我们的系统将两者统一到一个实时管道中，实现无缝的无接触人机交互。 

---
# Time-Constrained Intelligent Adversaries for Automation Vulnerability Testing: A Multi-Robot Patrol Case Study 

**Title (ZH)**: 时间约束下的智能对手在自动化漏洞测试中的多机器人巡逻案例研究 

**Authors**: James C. Ward, Alex Bott, Connor York, Edmund R. Hunt  

**Link**: [PDF](https://arxiv.org/pdf/2509.11971)  

**Abstract**: Simulating hostile attacks of physical autonomous systems can be a useful tool to examine their robustness to attack and inform vulnerability-aware design. In this work, we examine this through the lens of multi-robot patrol, by presenting a machine learning-based adversary model that observes robot patrol behavior in order to attempt to gain undetected access to a secure environment within a limited time duration. Such a model allows for evaluation of a patrol system against a realistic potential adversary, offering insight into future patrol strategy design. We show that our new model outperforms existing baselines, thus providing a more stringent test, and examine its performance against multiple leading decentralized multi-robot patrol strategies. 

**Abstract (ZH)**: 通过基于机器学习的对手模型模拟物理自主系统的敌对攻击，可以评估其抗攻击稳健性并指导漏洞意识设计。本文通过多机器人巡逻这一视角，提出了一种对手模型，该模型通过观察机器人巡逻行为，试图在有限时间内未被发现地访问安全环境。该模型允许对巡逻系统进行现实对手评估，为未来的巡逻策略设计提供洞见。我们展示了新模型优于现有基准，提供了更严格的测试，并对其在多个领先分散式多机器人巡逻策略上的性能进行了评估。 

---
# E2-BKI: Evidential Ellipsoidal Bayesian Kernel Inference for Uncertainty-aware Gaussian Semantic Mapping 

**Title (ZH)**: E2-BKI: 证据椭球贝叶斯核推理在不确定性感知高斯语义映射中的应用 

**Authors**: Junyoung Kim, Minsik Jeon, Jihong Min, Kiho Kwak, Junwon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11964)  

**Abstract**: Semantic mapping aims to construct a 3D semantic representation of the environment, providing essential knowledge for robots operating in complex outdoor settings. While Bayesian Kernel Inference (BKI) addresses discontinuities of map inference from sparse sensor data, existing semantic mapping methods suffer from various sources of uncertainties in challenging outdoor environments. To address these issues, we propose an uncertainty-aware semantic mapping framework that handles multiple sources of uncertainties, which significantly degrade mapping performance. Our method estimates uncertainties in semantic predictions using Evidential Deep Learning and incorporates them into BKI for robust semantic inference. It further aggregates noisy observations into coherent Gaussian representations to mitigate the impact of unreliable points, while employing geometry-aligned kernels that adapt to complex scene structures. These Gaussian primitives effectively fuse local geometric and semantic information, enabling robust, uncertainty-aware mapping in complex outdoor scenarios. Comprehensive evaluation across diverse off-road and urban outdoor environments demonstrates consistent improvements in mapping quality, uncertainty calibration, representational flexibility, and robustness, while maintaining real-time efficiency. 

**Abstract (ZH)**: 面向复杂室外环境的不确定性aware语义映射框架 

---
# VH-Diffuser: Variable Horizon Diffusion Planner for Time-Aware Goal-Conditioned Trajectory Planning 

**Title (ZH)**: 变量时间 horizon 扩散规划器：时间意识的目标条件轨迹规划 

**Authors**: Ruijia Liu, Ancheng Hou, Shaoyuan Li, Xiang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.11930)  

**Abstract**: Diffusion-based planners have gained significant recent attention for their robustness and performance in long-horizon tasks. However, most existing planners rely on a fixed, pre-specified horizon during both training and inference. This rigidity often produces length-mismatch (trajectories that are too short or too long) and brittle performance across instances with varying geometric or dynamical difficulty. In this paper, we introduce the Variable Horizon Diffuser (VHD) framework, which treats the horizon as a learned variable rather than a fixed hyperparameter. Given a start-goal pair, we first predict an instance-specific horizon using a learned Length Predictor model, which guides a Diffusion Planner to generate a trajectory of the desired length. Our design maintains compatibility with existing diffusion planners by controlling trajectory length through initial noise shaping and training on randomly cropped sub-trajectories, without requiring architectural changes. Empirically, VHD improves success rates and path efficiency in maze-navigation and robot-arm control benchmarks, showing greater robustness to horizon mismatch and unseen lengths, while keeping training simple and offline-only. 

**Abstract (ZH)**: 基于扩散的规划器在长时_horizon任务中展现出了显著的稳健性和性能，但大多数现有的规划器在训练和推理过程中都依赖于固定且预先指定的horizon。这种固有性经常会产出长度不匹配的问题（轨迹过短或过长）并且在具有不同几何或动力学难度的任务间表现出脆弱的表现。在这篇论文中，我们提出了Variable Horizon Diffuser（VHD）框架，将horizon视为一个可学习的变量而不是固定的超参数。给定起点和终点，我们首先使用一个学习到的长度预测模型来预测一个实例特定的horizon，从而引导扩散规划器生成所需长度的轨迹。我们的设计通过初始噪声塑形控制轨迹长度，并通过在随机裁剪的子轨迹上进行训练来保持与现有扩散规划器的兼容性，无需修改架构。实验结果显示，VHD在迷宫导航和机器人臂控制基准测试中提高了成功率和路径效率，对horizon不匹配和未见过的长度具有更强的鲁棒性，同时保持了简单的无监督训练。 

---
# Tenma: Robust Cross-Embodiment Robot Manipulation with Diffusion Transformer 

**Title (ZH)**: Tenma：具有扩散变压器的稳健跨体态机器人操作 

**Authors**: Travis Davies, Yiqi Huang, Yunxin Liu, Xiang Chen, Huxian Liu, Luhui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11865)  

**Abstract**: Scaling Transformer policies and diffusion models has advanced robotic manipulation, yet combining these techniques in lightweight, cross-embodiment learning settings remains challenging. We study design choices that most affect stability and performance for diffusion-transformer policies trained on heterogeneous, multimodal robot data, and introduce Tenma, a lightweight diffusion-transformer for bi-manual arm control. Tenma integrates multiview RGB, proprioception, and language via a cross-embodiment normalizer that maps disparate state/action spaces into a shared latent space; a Joint State-Time encoder for temporally aligned observation learning with inference speed boosts; and a diffusion action decoder optimized for training stability and learning capacity. Across benchmarks and under matched compute, Tenma achieves an average success rate of 88.95% in-distribution and maintains strong performance under object and scene shifts, substantially exceeding baseline policies whose best in-distribution average is 18.12%. Despite using moderate data scale, Tenma delivers robust manipulation and generalization, indicating the great potential for multimodal and cross-embodiment learning strategies for further augmenting the capacity of transformer-based imitation learning policies. 

**Abstract (ZH)**: scaling transformer策略和扩散模型已推动了机器人的操作能力，但在轻量级的跨 embodiment 学习环境中结合这些技术仍然具有挑战性。我们研究了对基于异构多模态机器人数据训练的扩散-变压器策略产生最大影响的设计选择，并引入了Tenma，一个用于双臂控制的轻量级扩散-变压器模型。Tenma 通过交叉 embodiment 正规化将不同的状态/动作空间映射到共享的潜在空间，整合多视角 RGB、 proprioception 和语言；通过联合状态-时间编码器实现时间对齐的观察学习，并加速推理速度；并通过优化训练稳定性和学习能力的动作解码器实现扩散动作解码。在基准测试中，即使在匹配的计算资源下，Tenma 的平均成功率为 88.95%，并能在对象和场景变化时保持强大的性能，大幅超过了最佳平均成功率为 18.12% 的基线策略。尽管使用了中等规模的数据，Tenma 仍实现了稳健的操作和泛化，表明多模态和跨 embodiment 学习策略在进一步增强基于变压器的模仿学习策略的能力方面的巨大潜力。 

---
# TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning 

**Title (ZH)**: TrajBooster: 以轨迹为中心的学习提升类人全身操作能力 

**Authors**: Jiacheng Liu, Pengxiang Ding, Qihang Zhou, Yuxuan Wu, Da Huang, Zimian Peng, Wei Xiao, Weinan Zhang, Lixin Yang, Cewu Lu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11839)  

**Abstract**: Imitation learning (IL) enables efficient skill acquisition from demonstrations but often struggles with long-horizon tasks and high-precision control due to compounding errors. Residual policy learning offers a promising, model-agnostic solution by refining a base policy through closed-loop corrections. However, existing approaches primarily focus on local corrections to the base policy, lacking a global understanding of state evolution, which limits robustness and generalization to unseen scenarios. To address this, we propose incorporating global dynamics modeling to guide residual policy updates. Specifically, we leverage Koopman operator theory to impose linear time-invariant structure in a learned latent space, enabling reliable state transitions and improved extrapolation for long-horizon prediction and unseen environments. We introduce KORR (Koopman-guided Online Residual Refinement), a simple yet effective framework that conditions residual corrections on Koopman-predicted latent states, enabling globally informed and stable action refinement. We evaluate KORR on long-horizon, fine-grained robotic furniture assembly tasks under various perturbations. Results demonstrate consistent gains in performance, robustness, and generalization over strong baselines. Our findings further highlight the potential of Koopman-based modeling to bridge modern learning methods with classical control theory. For more details, please refer to this https URL. 

**Abstract (ZH)**: 基于Koopman算子的全局动态建模引导残差策略在线精炼 

---
# UniPilot: Enabling GPS-Denied Autonomy Across Embodiments 

**Title (ZH)**: UniPilot: 跨载体的GPS拒止自主导航 

**Authors**: Mihir Kulkarni, Mihir Dharmadhikari, Nikhil Khedekar, Morten Nissov, Mohit Singh, Philipp Weiss, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2509.11793)  

**Abstract**: This paper presents UniPilot, a compact hardware-software autonomy payload that can be integrated across diverse robot embodiments to enable autonomous operation in GPS-denied environments. The system integrates a multi-modal sensing suite including LiDAR, radar, vision, and inertial sensing for robust operation in conditions where uni-modal approaches may fail. UniPilot runs a complete autonomy software comprising multi-modal perception, exploration and inspection path planning, and learning-based navigation policies. The payload provides robust localization, mapping, planning, and safety and control capabilities in a single unit that can be deployed across a wide range of platforms. A large number of experiments are conducted across diverse environments and on a variety of robot platforms to validate the mapping, planning, and safe navigation capabilities enabled by the payload. 

**Abstract (ZH)**: UniPilot：一种可用于各类机器人体态的紧凑型硬件-软件自主载荷，可在GPS受限环境中实现自主操作 

---
# Synthetic vs. Real Training Data for Visual Navigation 

**Title (ZH)**: 合成数据 vs. 实际数据在视觉导航中的应用 

**Authors**: Lauri Suomela, Sasanka Kuruppu Arachchige, German F. Torres, Harry Edelman, Joni-Kristian Kämäräinen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11791)  

**Abstract**: This paper investigates how the performance of visual navigation policies trained in simulation compares to policies trained with real-world data. Performance degradation of simulator-trained policies is often significant when they are evaluated in the real world. However, despite this well-known sim-to-real gap, we demonstrate that simulator-trained policies can match the performance of their real-world-trained counterparts.
Central to our approach is a navigation policy architecture that bridges the sim-to-real appearance gap by leveraging pretrained visual representations and runs real-time on robot hardware. Evaluations on a wheeled mobile robot show that the proposed policy, when trained in simulation, outperforms its real-world-trained version by 31% and the prior state-of-the-art methods by 50% in navigation success rate. Policy generalization is verified by deploying the same model onboard a drone.
Our results highlight the importance of diverse image encoder pretraining for sim-to-real generalization, and identify on-policy learning as a key advantage of simulated training over training with real data. 

**Abstract (ZH)**: 本文探讨了在仿真环境中训练的视觉导航策略与使用真实世界数据训练的策略在性能上的比较。尽管仿真训练的策略在真实世界评估时常常表现出显著的性能下降，但我们证明了仿真训练的策略能够与真实世界训练的版本相匹配。 

---
# Augmented Reality-Enhanced Robot Teleoperation for Collecting User Demonstrations 

**Title (ZH)**: 增强现实辅助的机器人远程操作以收集用户示范 

**Authors**: Shiqi Gong, Sebastian Zudaire, Chi Zhang, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11783)  

**Abstract**: Traditional industrial robot programming is often complex and time-consuming, typically requiring weeks or even months of effort from expert programmers. Although Programming by Demonstration (PbD) offers a more accessible alternative, intuitive interfaces for robot control and demonstration collection remain challenging. To address this, we propose an Augmented Reality (AR)-enhanced robot teleoperation system that integrates AR-based control with spatial point cloud rendering, enabling intuitive, contact-free demonstrations. This approach allows operators to control robots remotely without entering the workspace or using conventional tools like the teach pendant. The proposed system is generally applicable and has been demonstrated on ABB robot platforms, specifically validated with the IRB 1200 industrial robot and the GoFa 5 collaborative robot. A user study evaluates the impact of real-time environmental perception, specifically with and without point cloud rendering, on task completion accuracy, efficiency, and user confidence. Results indicate that enhanced perception significantly improves task performance by 28% and enhances user experience, as reflected by a 12% increase in the System Usability Scale (SUS) score. This work contributes to the advancement of intuitive robot teleoperation, AR interface design, environmental perception, and teleoperation safety mechanisms in industrial settings for demonstration collection. The collected demonstrations may serve as valuable training data for machine learning applications. 

**Abstract (ZH)**: 一种增强现实增强的机器人远程操作系统：基于空间点云渲染的直观无接触示教 

---
# Igniting VLMs toward the Embodied Space 

**Title (ZH)**: 点燃VLMs通往 embodied空间的路径 

**Authors**: Andy Zhai, Brae Liu, Bruno Fang, Chalse Cai, Ellie Ma, Ethan Yin, Hao Wang, Hugo Zhou, James Wang, Lights Shi, Lucy Liang, Make Wang, Qian Wang, Roy Gan, Ryan Yu, Shalfun Li, Starrick Liu, Sylas Chen, Vincent Chen, Zach Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11766)  

**Abstract**: While foundation models show remarkable progress in language and vision, existing vision-language models (VLMs) still have limited spatial and embodiment understanding. Transferring VLMs to embodied domains reveals fundamental mismatches between modalities, pretraining distributions, and training objectives, leaving action comprehension and generation as a central bottleneck on the path to AGI.
We introduce WALL-OSS, an end-to-end embodied foundation model that leverages large-scale multimodal pretraining to achieve (1) embodiment-aware vision-language understanding, (2) strong language-action association, and (3) robust manipulation capability.
Our approach employs a tightly coupled architecture and multi-strategies training curriculum that enables Unified Cross-Level CoT-seamlessly unifying instruction reasoning, subgoal decomposition, and fine-grained action synthesis within a single differentiable framework.
Our results show that WALL-OSS attains high success on complex long-horizon manipulations, demonstrates strong instruction-following capabilities, complex understanding and reasoning, and outperforms strong baselines, thereby providing a reliable and scalable path from VLMs to embodied foundation models. 

**Abstract (ZH)**: WALL-OSS：一种端到端的具身基础模型，通过大规模多模态预训练实现具身aware的视觉语言理解、强大的语言动作关联和稳健的操控能力 

---
# Adaptive Motorized LiDAR Scanning Control for Robust Localization with OpenStreetMap 

**Title (ZH)**: 基于OpenStreetMap的鲁棒定位的自适应电机化LiDAR扫描控制 

**Authors**: Jianping Li, Kaisong Zhu, Zhongyuan Liu, Rui Jin, Xinhang Xu, Pengfei Wan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.11742)  

**Abstract**: LiDAR-to-OpenStreetMap (OSM) localization has gained increasing attention, as OSM provides lightweight global priors such as building footprints. These priors enhance global consistency for robot navigation, but OSM is often incomplete or outdated, limiting its reliability in real-world deployment. Meanwhile, LiDAR itself suffers from a limited field of view (FoV), where motorized rotation is commonly used to achieve panoramic coverage. Existing motorized LiDAR systems, however, typically employ constant-speed scanning that disregards both scene structure and map priors, leading to wasted effort in feature-sparse regions and degraded localization accuracy. To address these challenges, we propose Adaptive LiDAR Scanning with OSM guidance, a framework that integrates global priors with local observability prediction to improve localization robustness. Specifically, we augment uncertainty-aware model predictive control with an OSM-aware term that adaptively allocates scanning effort according to both scene-dependent observability and the spatial distribution of OSM features. The method is implemented in ROS with a motorized LiDAR odometry backend and evaluated in both simulation and real-world experiments. Results on campus roads, indoor corridors, and urban environments demonstrate significant reductions in trajectory error compared to constant-speed baselines, while maintaining scan completeness. These findings highlight the potential of coupling open-source maps with adaptive LiDAR scanning to achieve robust and efficient localization in complex environments. 

**Abstract (ZH)**: LiDAR到OpenStreetMap (OSM)定位正逐渐获得关注，OSM提供了如建筑足迹等轻量级全局先验信息，这些先验信息增强了机器人导航的全局一致性，但OSM往往不完整或过时，限制了其在实际部署中的可靠性。同时，LiDAR自身存在有限的视野（FOV），通常通过电机驱动旋转来实现全景覆盖。然而，现有的电机驱动LiDAR系统通常采用恒定速度扫描，忽视了场景结构和地图先验，导致在特征稀疏区域浪费了扫描努力，并降低了定位准确性。为解决这些问题，我们提出了一种基于OSM指导的自适应LiDAR扫描框架，该框架将全局先验与局部可观测性预测相结合，以提高定位鲁棒性。具体而言，我们采用了不确定性感知的模型预测控制，并添加了一个OSM意识项，根据场景依赖的可观测性和OSM特征的空间分布自适应分配扫描努力。该方法在ROS中实现，并结合电机驱动LiDAR里程计后端，在仿真和实际实验中进行了评估。结果表明，在校园区道路、室内走廊和城市环境中，与恒定速度基线相比，轨迹误差显著减少，同时保持了扫描完整性。这些发现突显了将开源地图与自适应LiDAR扫描结合使用以在复杂环境中实现鲁棒且高效的定位的潜力。 

---
# From Pixels to Shelf: End-to-End Algorithmic Control of a Mobile Manipulator for Supermarket Stocking and Fronting 

**Title (ZH)**: 从像素到货架：移动 manipulator 的端到端算法控制在超市上架和前端作业中应用 

**Authors**: Davide Peron, Victor Nan Fernandez-Ayala, Lukas Segelmark  

**Link**: [PDF](https://arxiv.org/pdf/2509.11740)  

**Abstract**: Autonomous stocking in retail environments, particularly supermarkets, presents challenges due to dynamic human interactions, constrained spaces, and diverse product geometries. This paper introduces an efficient end-to-end robotic system for autonomous shelf stocking and fronting, integrating commercially available hardware with a scalable algorithmic architecture. A major contribution of this work is the system integration of off-the-shelf hardware and ROS2-based perception, planning, and control into a single deployable platform for retail environments. Our solution leverages Behavior Trees (BTs) for task planning, fine-tuned vision models for object detection, and a two-step Model Predictive Control (MPC) framework for precise shelf navigation using ArUco markers. Laboratory experiments replicating realistic supermarket conditions demonstrate reliable performance, achieving over 98% success in pick-and-place operations across a total of more than 700 stocking events. However, our comparative benchmarks indicate that the performance and cost-effectiveness of current autonomous systems remain inferior to that of human workers, which we use to highlight key improvement areas and quantify the progress still required before widespread commercial deployment can realistically be achieved. 

**Abstract (ZH)**: 零售环境中自主补货面临的挑战包括动态的人机交互、受限的空间以及多样的商品几何形状。本文介绍了一种高效的端到端机器人系统，用于自主货架补货和前臷，将商用硬件与可扩展的算法架构集成在一起。本工作的主要贡献在于将商用现成硬件和基于ROS2的感知、规划和控制相结合，形成一个适用于零售环境的可部署平台。我们的解决方案利用行为树（BTs）进行任务规划，采用细调后的视觉模型进行物体检测，并采用两步法模型预测控制（MPC）框架，利用ArUco标记实现精确的货架导航。实验室实验模拟了真实的超市条件，展示了可靠的表现，总补货事件超过700次，拣选和放置操作的成功率超过98%。然而，我们的对比基准表明，当前自主系统的性能和成本效益仍然低于人类工人的水平，这为我们指出了亟待改进的关键领域，并量化了在实现广泛商用部署之前仍需取得的进步。 

---
# Tensor Invariant Data-Assisted Control and Dynamic Decomposition of Multibody Systems 

**Title (ZH)**: 多体系统张量不变的数据辅助控制及动态分解 

**Authors**: Mostafa Eslami, Maryam Babazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.11688)  

**Abstract**: The control of robotic systems in complex, shared collaborative workspaces presents significant challenges in achieving robust performance and safety when learning from experienced or simulated data is employed in the pipeline. A primary bottleneck is the reliance on coordinate-dependent models, which leads to profound data inefficiency by failing to generalize physical interactions across different frames of reference. This forces learning algorithms to rediscover fundamental physical principles in every new orientation, artificially inflating the complexity of the learning task. This paper introduces a novel framework that synergizes a coordinate-free, unreduced multibody dynamics and kinematics model based on tensor mechanics with a Data-Assisted Control (DAC) architecture. A non-recursive, closed-form Newton-Euler model in an augmented matrix form is derived that is optimized for tensor-based control design. This structure enables a principled decomposition of the system into a structurally certain, physically grounded part and an uncertain, empirical, and interaction-focused part, mediated by a virtual port variable. Then, a complete, end-to-end tensor-invariant pipeline for modeling, control, and learning is proposed. The coordinate-free control laws for the structurally certain part provide a stable and abstract command interface, proven via Lyapunov analysis. Eventually, the model and closed-loop system are validated through simulations. This work provides a naturally ideal input for data-efficient, frame-invariant learning algorithms, such as equivariant learning, designed to learn the uncertain interaction. The synergy directly addresses the data-inefficiency problem, increases explainability and interpretability, and paves the way for more robust and generalizable robotic control in interactive environments. 

**Abstract (ZH)**: 无坐标依赖的多体动力学与 kinematics 模型结合数据辅助控制架构在复杂共享协作工作空间中的机器人系统控制 

---
# ParaEQsA: Parallel and Asynchronous Embodied Questions Scheduling and Answering 

**Title (ZH)**: ParaEQsA: 并行异步体域Question Scheduling and Answering 

**Authors**: Haisheng Wang, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11663)  

**Abstract**: This paper formulates the Embodied Questions Answering (EQsA) problem, introduces a corresponding benchmark, and proposes a system to tackle the problem. Classical Embodied Question Answering (EQA) is typically formulated as answering one single question by actively exploring a 3D environment. Real deployments, however, often demand handling multiple questions that may arrive asynchronously and carry different urgencies. We formalize this setting as Embodied Questions Answering (EQsA) and present ParaEQsA, a framework for parallel, urgency-aware scheduling and answering. ParaEQsA leverages a group memory module shared among questions to reduce redundant exploration, and a priority-planning module to dynamically schedule questions. To evaluate this setting, we contribute the Parallel Asynchronous Embodied Questions (PAEQs) benchmark containing 40 indoor scenes and five questions per scene (200 in total), featuring asynchronous follow-up questions and urgency labels. We further propose metrics for EQsA performance: Direct Answer Rate (DAR), and Normalized Urgency-Weighted Latency (NUWL), which jointly measure efficiency and responsiveness of this system. ParaEQsA consistently outperforms strong sequential baselines adapted from recent EQA systems, while reducing exploration and delay. Empirical evaluations investigate the relative contributions of priority, urgency modeling, spatial scope, reward estimation, and dependency reasoning within our framework. Together, these results demonstrate that urgency-aware, parallel scheduling is key to making embodied agents responsive and efficient under realistic, multi-question workloads. 

**Abstract (ZH)**: 基于体态的多任务问答：ParaEQsA框架 

---
# Inference-stage Adaptation-projection Strategy Adapts Diffusion Policy to Cross-manipulators Scenarios 

**Title (ZH)**: 推理阶段适配投影策略适配跨操作场景的扩散政策 

**Authors**: Xiangtong Yao, Yirui Zhou, Yuan Meng, Yanwen Liu, Liangyu Dong, Zitao Zhang, Zhenshan Bing, Kai Huang, Fuchun Sun, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.11621)  

**Abstract**: Diffusion policies are powerful visuomotor models for robotic manipulation, yet they often fail to generalize to manipulators or end-effectors unseen during training and struggle to accommodate new task requirements at inference time. Addressing this typically requires costly data recollection and policy retraining for each new hardware or task configuration. To overcome this, we introduce an adaptation-projection strategy that enables a diffusion policy to perform zero-shot adaptation to novel manipulators and dynamic task settings, entirely at inference time and without any retraining. Our method first trains a diffusion policy in SE(3) space using demonstrations from a base manipulator. During online deployment, it projects the policy's generated trajectories to satisfy the kinematic and task-specific constraints imposed by the new hardware and objectives. Moreover, this projection dynamically adapts to physical differences (e.g., tool-center-point offsets, jaw widths) and task requirements (e.g., obstacle heights), ensuring robust and successful execution. We validate our approach on real-world pick-and-place, pushing, and pouring tasks across multiple manipulators, including the Franka Panda and Kuka iiwa 14, equipped with a diverse array of end-effectors like flexible grippers, Robotiq 2F/3F grippers, and various 3D-printed designs. Our results demonstrate consistently high success rates in these cross-manipulator scenarios, proving the effectiveness and practicality of our adaptation-projection strategy. The code will be released after peer review. 

**Abstract (ZH)**: 一种在推理时进行零样本适应和投影的扩散策略：用于机器人操作的通用视觉-运动模型 

---
# AssemMate: Graph-Based LLM for Robotic Assembly Assistance 

**Title (ZH)**: AssemMate：基于图的大型语言模型在机器人装配辅助中的应用 

**Authors**: Qi Zheng, Chaoran Zhang, Zijian Liang, EnTe Lin, Shubo Cui, Qinghongbing Xie, Zhaobo Xu, Long Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.11617)  

**Abstract**: Large Language Model (LLM)-based robotic assembly assistance has gained significant research attention. It requires the injection of domain-specific knowledge to guide the assembly process through natural language interaction with humans. Despite some progress, existing methods represent knowledge in the form of natural language text. Due to the long context and redundant content, they struggle to meet the robots' requirements for real-time and precise reasoning. In order to bridge this gap, we present AssemMate, which utilizes the graph\textemdash a concise and accurate form of knowledge representation\textemdash as input. This graph-based LLM enables knowledge graph question answering (KGQA), supporting human-robot interaction and assembly task planning for specific products. Beyond interactive QA, AssemMate also supports sensing stacked scenes and executing grasping to assist with assembly. Specifically, a self-supervised Graph Convolutional Network (GCN) encodes knowledge graph entities and relations into a latent space and aligns them with LLM's representation, enabling the LLM to understand graph information. In addition, a vision-enhanced strategy is employed to address stacked scenes in grasping. Through training and evaluation, AssemMate outperforms existing methods, achieving 6.4\% higher accuracy, 3 times faster inference, and 28 times shorter context length, while demonstrating strong generalization ability on random graphs. And our approach further demonstrates superiority through robotic grasping experiments in both simulated and real-world settings. More details can be found on the project page: this https URL 

**Abstract (ZH)**: 基于大型语言模型（LLM）的机器人装配辅助：利用图表示的知识驱动交互与装配任务规划 

---
# GBPP: Grasp-Aware Base Placement Prediction for Robots via Two-Stage Learning 

**Title (ZH)**: GBPP: 基于两阶段学习的抓取意识基座位置预测方法 

**Authors**: Jizhuo Chen, Diwen Liu, Jiaming Wang, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2509.11594)  

**Abstract**: GBPP is a fast learning based scorer that selects a robot base pose for grasping from a single RGB-D snapshot. The method uses a two stage curriculum: (1) a simple distance-visibility rule auto-labels a large dataset at low cost; and (2) a smaller set of high fidelity simulation trials refines the model to match true grasp outcomes. A PointNet++ style point cloud encoder with an MLP scores dense grids of candidate poses, enabling rapid online selection without full task-and-motion optimization. In simulation and on a real mobile manipulator, GBPP outperforms proximity and geometry only baselines, choosing safer and more reachable stances and degrading gracefully when wrong. The results offer a practical recipe for data efficient, geometry aware base placement: use inexpensive heuristics for coverage, then calibrate with targeted simulation. 

**Abstract (ZH)**: GBPP是一种基于快速学习的评分器，可以从单个RGB-D快照中选择用于抓取的机器人基座姿态。该方法采用两阶段课程学习：（1）一种简单的距离-可见性规则自动标注大量数据集以降低成本；（2）一小部分高保真模拟试验精化模型以匹配真实的抓取结果。该方法使用类似于PointNet++的点云编码器和MLP对候选姿态进行评分，能够在不进行全程任务和运动优化的情况下实现快速在线选择。在模拟和实际移动 manipulator 上，GBPP 在接近性和几何学 baselines 上表现更优，选择更安全和更容易到达的姿态，在错误时平滑退化。结果提供了一种实用的基于数据高效、几何感知基座放置的食谱：使用低成本启发式方法进行覆盖，然后通过目标模拟进行校准。 

---
# Shape control of simulated multi-segment continuum robots via Koopman operators with per-segment projection 

**Title (ZH)**: 基于段落投影的科里奥兰算子控制的模拟多段连续机器人形状控制 

**Authors**: Eron Ristich, Jiahe Wang, Lei Zhang, Sultan Haidar Ali, Wanxin Jin, Yi Ren, Jiefeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.11567)  

**Abstract**: Soft continuum robots can allow for biocompatible yet compliant motions, such as the ability of octopus arms to swim, crawl, and manipulate objects. However, current state-of-the-art continuum robots can only achieve real-time task-space control (i.e., tip control) but not whole-shape control, mainly due to the high computational cost from its infinite degrees of freedom. In this paper, we present a data-driven Koopman operator-based approach for the shape control of simulated multi-segment tendon-driven soft continuum robots with the Kirchhoff rod model. Using data collected from these simulated soft robots, we conduct a per-segment projection scheme on the state of the robots allowing for the identification of control-affine Koopman models that are an order of magnitude more accurate than without the projection scheme. Using these learned Koopman models, we use a linear model predictive control (MPC) to control the robots to a collection of target shapes of varying complexity. Our method realizes computationally efficient closed-loop control, and demonstrates the feasibility of real-time shape control for soft robots. We envision this work can pave the way for practical shape control of soft continuum robots. 

**Abstract (ZH)**: 软连续机器人的生物相容性和柔顺运动，如章鱼臂的游泳、爬行和物体操作能力，可以通过软连续机器人实现。然而，当前最先进的连续机器人只能实现实时任务空间控制（即末端控制），而无法实现整体形状控制，主要原因是其无穷自由度带来的高计算成本。本文提出了一种基于Koopman算子的数据驱动方法，用于模拟多段腱驱动软连续机器人的形状控制，采用Kirchhoff杆模型。通过从这些模拟软机器人收集的数据，我们进行了一段一段的投影方案，从而识别出控制相关Koopman模型，其准确度比未使用投影方案高出一个数量级。利用这些学习到的Koopman模型，我们使用线性模型预测控制（MPC）控制机器人达到多种复杂度的目标形状。该方法实现了计算高效的闭环控制，并展示了软连续机器人实时形状控制的可行性。我们期望这项工作能够为软连续机器人的实用形状控制开辟道路。 

---
# PaiP: An Operational Aware Interactive Planner for Unknown Cabinet Environments 

**Title (ZH)**: PaiP: 一个考虑操作任务的未知衣柜环境交互式规划器 

**Authors**: Chengjin Wang, Zheng Yan, Yanmin Zhou, Runjie Shen, Zhipeng Wang, Bin Cheng, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2509.11516)  

**Abstract**: Box/cabinet scenarios with stacked objects pose significant challenges for robotic motion due to visual occlusions and constrained free space. Traditional collision-free trajectory planning methods often fail when no collision-free paths exist, and may even lead to catastrophic collisions caused by invisible objects. To overcome these challenges, we propose an operational aware interactive motion planner (PaiP) a real-time closed-loop planning framework utilizing multimodal tactile perception. This framework autonomously infers object interaction features by perceiving motion effects at interaction interfaces. These interaction features are incorporated into grid maps to generate operational cost maps. Building upon this representation, we extend sampling-based planning methods to interactive planning by optimizing both path cost and operational cost. Experimental results demonstrate that PaiP achieves robust motion in narrow spaces. 

**Abstract (ZH)**: 具有堆叠物体的盒子/柜子场景对机器人运动构成了显著挑战，由于视觉遮挡和受限的自由空间。传统的无碰撞路径规划方法在不存在无碰撞路径时往往失效，并可能导致由隐形物体引起的灾难性碰撞。为克服这些挑战，我们提出了一种操作感知交互式运动规划器（PaiP），这是一种利用多模态触觉感知的实时闭环规划框架。该框架自主通过感知交互界面的运动效果来推断物体交互特征，并将这些交互特征整合到栅格地图中生成操作成本地图。基于此表示方法，我们通过同时优化路径成本和操作成本，将基于采样的规划方法扩展到交互规划。实验结果表明，PaiP 能在狭窄空间中实现稳健的运动。 

---
# Design and Development of a Remotely Wire-Driven Walking Robot 

**Title (ZH)**: 远程线控制步行机器人设计与开发 

**Authors**: Takahiro Hattori, Kento Kawaharazuka, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2509.11506)  

**Abstract**: Operating in environments too harsh or inaccessible for humans is one of the critical roles expected of robots. However, such environments often pose risks to electronic components as well. To overcome this, various approaches have been developed, including autonomous mobile robots without electronics, hydraulic remotely actuated mobile robots, and long-reach robot arms driven by wires. Among these, electronics-free autonomous robots cannot make complex decisions, while hydraulically actuated mobile robots and wire-driven robot arms are used in harsh environments such as nuclear power plants. Mobile robots offer greater reach and obstacle avoidance than robot arms, and wire mechanisms offer broader environmental applicability than hydraulics. However, wire-driven systems have not been used for remote actuation of mobile robots. In this study, we propose a novel mechanism called Remote Wire Drive that enables remote actuation of mobile robots via wires. This mechanism is a series connection of decoupled joints, a mechanism used in wire-driven robot arms, adapted for power transmission. We experimentally validated its feasibility by actuating a wire-driven quadruped robot, which we also developed in this study, through Remote Wire Drive. 

**Abstract (ZH)**: 基于电线远程驱动的移动机器人新型远程驱动机制研究 

---
# FR-Net: Learning Robust Quadrupedal Fall Recovery on Challenging Terrains through Mass-Contact Prediction 

**Title (ZH)**: FR-网：通过质量接触预测在具有挑战性的地形上学习稳健的四足跌倒恢复 

**Authors**: Yidan Lu, Yinzhao Dong, Jiahui Zhang, Ji Ma, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11504)  

**Abstract**: Fall recovery for legged robots remains challenging, particularly on complex terrains where traditional controllers fail due to incomplete terrain perception and uncertain interactions. We present \textbf{FR-Net}, a learning-based framework that enables quadrupedal robots to recover from arbitrary fall poses across diverse environments. Central to our approach is a Mass-Contact Predictor network that estimates the robot's mass distribution and contact states from limited sensory inputs, facilitating effective recovery strategies. Our carefully designed reward functions ensure safe recovery even on steep stairs without dangerous rolling motions common to existing methods. Trained entirely in simulation using privileged learning, our framework guides policy learning without requiring explicit terrain data during deployment. We demonstrate the generalization capabilities of \textbf{FR-Net} across different quadrupedal platforms in simulation and validate its performance through extensive real-world experiments on the Go2 robot in 10 challenging scenarios. Our results indicate that explicit mass-contact prediction is key to robust fall recovery, offering a promising direction for generalizable quadrupedal skills. 

**Abstract (ZH)**: 基于学习的四足机器人任意跌倒姿态恢复方法：跨复杂环境的有效策略 

---
# RAPTOR: A Foundation Policy for Quadrotor Control 

**Title (ZH)**: RAPTOR：四旋翼飞行器控制的基石政策 

**Authors**: Jonas Eschmann, Dario Albani, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2509.11481)  

**Abstract**: Humans are remarkably data-efficient when adapting to new unseen conditions, like driving a new car. In contrast, modern robotic control systems, like neural network policies trained using Reinforcement Learning (RL), are highly specialized for single environments. Because of this overfitting, they are known to break down even under small differences like the Simulation-to-Reality (Sim2Real) gap and require system identification and retraining for even minimal changes to the system. In this work, we present RAPTOR, a method for training a highly adaptive foundation policy for quadrotor control. Our method enables training a single, end-to-end neural-network policy to control a wide variety of quadrotors. We test 10 different real quadrotors from 32 g to 2.4 kg that also differ in motor type (brushed vs. brushless), frame type (soft vs. rigid), propeller type (2/3/4-blade), and flight controller (PX4/Betaflight/Crazyflie/M5StampFly). We find that a tiny, three-layer policy with only 2084 parameters is sufficient for zero-shot adaptation to a wide variety of platforms. The adaptation through In-Context Learning is made possible by using a recurrence in the hidden layer. The policy is trained through a novel Meta-Imitation Learning algorithm, where we sample 1000 quadrotors and train a teacher policy for each of them using Reinforcement Learning. Subsequently, the 1000 teachers are distilled into a single, adaptive student policy. We find that within milliseconds, the resulting foundation policy adapts zero-shot to unseen quadrotors. We extensively test the capabilities of the foundation policy under numerous conditions (trajectory tracking, indoor/outdoor, wind disturbance, poking, different propellers). 

**Abstract (ZH)**: 人类在适应新未见条件时表现出色的数据效率，例如驾驶新汽车。相比之下，现代机器人控制系统，如使用强化学习（RL）训练的神经网络策略，高度专业化仅适用于单一环境。由于这种过度拟合，它们在面临如仿真到现实（Sim2Real）差距等细微差异时会失效，并且即使是最小的系统更改也需要进行系统识别和重新训练。在这项工作中，我们提出了RAPTOR方法，用于训练四旋翼飞行器控制的高度适应基础策略。我们的方法使训练单个端到端的神经网络策略来控制各种四旋翼飞行器成为可能。我们测试了10种不同的真实四旋翼飞行器，质量从32克到2.4公斤不等，这些飞行器在电机类型（有刷 vs 无刷）、框架类型（柔性 vs 刚性）、旋翼类型（2/3/4片桨）和飞控系统（PX4/Betaflight/Crazyflie/M5StampFly）方面也有所不同。我们发现仅含有2084个参数的三层小型策略足以实现对各种平台的零样本适应。通过在隐藏层中使用递归，使基于上下文学习的适应成为可能。该策略通过一种新颖的元模仿学习算法进行训练，其中我们采样1000个四旋翼飞行器，并为每个飞行器使用强化学习训练一个教师策略。随后，1000个教师被提炼成一个高度适应的学生策略。我们发现，该基础策略在毫秒级时间内实现了对未见四旋翼飞行器的零样本适应。我们在多种条件下（轨迹跟踪、室内/室外、风干扰、触碰、不同旋翼）广泛测试了基础策略的能力。 

---
# A Software-Only Post-Processor for Indexed Rotary Machining on GRBL-Based CNCs 

**Title (ZH)**: 基于GRBL控制的 indexed rotary 加工软件后处理器 

**Authors**: Pedro Portugal, Damian D. Venghaus, Diego Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2509.11433)  

**Abstract**: Affordable desktop CNC routers are common in education, prototyping, and makerspaces, but most lack a rotary axis, limiting fabrication of rotationally symmetric or multi-sided parts. Existing solutions often require hardware retrofits, alternative controllers, or commercial CAM software, raising cost and complexity. This work presents a software-only framework for indexed rotary machining on GRBL-based CNCs. A custom post-processor converts planar toolpaths into discrete rotary steps, executed through a browser-based interface. While not equivalent to continuous 4- axis machining, the method enables practical rotary-axis fabrication using only standard, off-the- shelf mechanics, without firmware modification. By reducing technical and financial barriers, the framework expands access to multi-axis machining in classrooms, makerspaces, and small workshops, supporting hands-on learning and rapid prototyping. 

**Abstract (ZH)**: 基于GRBL的 CNC 系统的软件化索引旋转加工框架 

---
# Enhancing Generalization in Vision-Language-Action Models by Preserving Pretrained Representations 

**Title (ZH)**: 通过保留预训练表示以增强视觉-语言-动作模型的泛化能力 

**Authors**: Shresth Grover, Akshay Gopalkrishnan, Bo Ai, Henrik I. Christensen, Hao Su, Xuanlin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11417)  

**Abstract**: Vision-language-action (VLA) models finetuned from vision-language models (VLMs) hold the promise of leveraging rich pretrained representations to build generalist robots across diverse tasks and environments. However, direct fine-tuning on robot data often disrupts these representations and limits generalization. We present a framework that better preserves pretrained features while adapting them for robot manipulation. Our approach introduces three components: (i) a dual-encoder design with one frozen vision encoder to retain pretrained features and another trainable for task adaptation, (ii) a string-based action tokenizer that casts continuous actions into character sequences aligned with the model's pretraining domain, and (iii) a co-training strategy that combines robot demonstrations with vision-language datasets emphasizing spatial reasoning and affordances. Evaluations in simulation and on real robots show that our method improves robustness to visual perturbations, generalization to novel instructions and environments, and overall task success compared to baselines. 

**Abstract (ZH)**: 基于视觉-语言-动作模型的框架在保留预训练特征的同时适应机器人操作任务 

---
# TRUST 2025: SCRITA and RTSS @ RO-MAN 2025 

**Title (ZH)**: TRUST 2025: SCRITA和RTSS @ RO-MAN 2025 

**Authors**: Alessandra Rossi, Patrick Holthaus, Gabriella Lakatos, Sílvia Moros, Ali Fallahi, Murat Kirtay, Marie Postma, Erhan Oztop  

**Link**: [PDF](https://arxiv.org/pdf/2509.11402)  

**Abstract**: The TRUST workshop is the result of a collaboration between two established workshops in the field of Human-Robot Interaction: SCRITA (Trust, Acceptance and Social Cues in Human-Robot Interaction) and RTSS (Robot Trust for Symbiotic Societies). This joint initiative brings together the complementary goals of these workshops to advance research on trust from both the human and robot perspectives.
Website: this https URL 

**Abstract (ZH)**: TRUST研讨会是人机交互领域两个成熟研讨会SCRITA（Trust, Acceptance and Social Cues in Human-Robot Interaction）和RTSS（Robot Trust for Symbiotic Societies）合作的成果。这一联合倡议将这两个研讨会互补的目标汇集起来，以从人类和机器人两个视角推进信任研究。 

---
# Quantum deep reinforcement learning for humanoid robot navigation task 

**Title (ZH)**: 量子深度强化学习在类人机器人导航任务中的应用 

**Authors**: Romerik Lokossou, Birhanu Shimelis Girma, Ozan K. Tonguz, Ahmed Biyabani  

**Link**: [PDF](https://arxiv.org/pdf/2509.11388)  

**Abstract**: Classical reinforcement learning (RL) methods often struggle in complex, high-dimensional environments because of their extensive parameter requirements and challenges posed by stochastic, non-deterministic settings. This study introduces quantum deep reinforcement learning (QDRL) to train humanoid agents efficiently. While previous quantum RL models focused on smaller environments, such as wheeled robots and robotic arms, our work pioneers the application of QDRL to humanoid robotics, specifically in environments with substantial observation and action spaces, such as MuJoCo's Humanoid-v4 and Walker2d-v4. Using parameterized quantum circuits, we explored a hybrid quantum-classical setup to directly navigate high-dimensional state spaces, bypassing traditional mapping and planning. By integrating quantum computing with deep RL, we aim to develop models that can efficiently learn complex navigation tasks in humanoid robots. We evaluated the performance of the Soft Actor-Critic (SAC) in classical RL against its quantum implementation. The results show that the quantum SAC achieves an 8% higher average return (246.40) than the classical SAC (228.36) after 92% fewer steps, highlighting the accelerated learning potential of quantum computing in RL tasks. 

**Abstract (ZH)**: 量子深度强化学习在人形机器人中的应用研究 

---
# ActivePose: Active 6D Object Pose Estimation and Tracking for Robotic Manipulation 

**Title (ZH)**: 主动姿态：面向机器人操作的主动6D物体姿态估计与跟踪 

**Authors**: Sheng Liu, Zhe Li, Weiheng Wang, Han Sun, Heng Zhang, Hongpeng Chen, Yusen Qin, Arash Ajoudani, Yizhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11364)  

**Abstract**: Accurate 6-DoF object pose estimation and tracking are critical for reliable robotic manipulation. However, zero-shot methods often fail under viewpoint-induced ambiguities and fixed-camera setups struggle when objects move or become self-occluded. To address these challenges, we propose an active pose estimation pipeline that combines a Vision-Language Model (VLM) with "robotic imagination" to dynamically detect and resolve ambiguities in real time. In an offline stage, we render a dense set of views of the CAD model, compute the FoundationPose entropy for each view, and construct a geometric-aware prompt that includes low-entropy (unambiguous) and high-entropy (ambiguous) examples. At runtime, the system: (1) queries the VLM on the live image for an ambiguity score; (2) if ambiguity is detected, imagines a discrete set of candidate camera poses by rendering virtual views, scores each based on a weighted combination of VLM ambiguity probability and FoundationPose entropy, and then moves the camera to the Next-Best-View (NBV) to obtain a disambiguated pose estimation. Furthermore, since moving objects may leave the camera's field of view, we introduce an active pose tracking module: a diffusion-policy trained via imitation learning, which generates camera trajectories that preserve object visibility and minimize pose ambiguity. Experiments in simulation and real-world show that our approach significantly outperforms classical baselines. 

**Abstract (ZH)**: 精确的6-自由度物体姿态估计与跟踪对于可靠的机器人操作至关重要。然而，零样本方法往往在视角诱导的歧义性面前失效，固定摄像头设置在物体移动或发生自遮挡时也表现不佳。为解决这些挑战，我们提出了一种结合视觉语言模型（VLM）与“机器人想象”以实时动态检测和解决歧义性的主动姿态估计管道。在离线阶段，我们渲染CAD模型的密集视图集，计算每个视图的FoundationPose熵，并构建一个几何感知提示，包括低熵（无歧义）和高熵（有歧义）示例。在运行时，系统执行以下操作：(1) 对实时图像查询VLM以获得歧义评分；(2) 如检测到歧义，通过渲染虚拟视图生成一组离散的候选相机姿态，基于VLM歧义概率和FoundationPose熵加权组合评分，并移动相机至最佳视图以获得去歧义的姿态估计。此外，由于移动物体可能离开相机的视野范围，我们引入了一种主动姿态跟踪模块：通过模仿学习训练的扩散策略，生成保持物体可见性和最小化姿态歧义的相机轨迹。在仿真和真实世界的实验中，我们的方法显著优于经典基线方法。 

---
# Brain-Robot Interface for Exercise Mimicry 

**Title (ZH)**: 脑-机器人接口用于锻炼模仿 

**Authors**: Carl Bettosi, Emilyann Nault, Lynne Baillie, Markus Garschall, Marta Romeo, Beatrix Wais-Zechmann, Nicole Binderlehner, Theodoros Georgio  

**Link**: [PDF](https://arxiv.org/pdf/2509.11306)  

**Abstract**: For social robots to maintain long-term engagement as exercise instructors, rapport-building is essential. Motor mimicry--imitating one's physical actions--during social interaction has long been recognized as a powerful tool for fostering rapport, and it is widely used in rehabilitation exercises where patients mirror a physiotherapist or video demonstration. We developed a novel Brain-Robot Interface (BRI) that allows a social robot instructor to mimic a patient's exercise movements in real-time, using mental commands derived from the patient's intention. The system was evaluated in an exploratory study with 14 participants (3 physiotherapists and 11 hemiparetic patients recovering from stroke or other injuries). We found our system successfully demonstrated exercise mimicry in 12 sessions; however, accuracy varied. Participants had positive perceptions of the robot instructor, with high trust and acceptance levels, which were not affected by the introduction of BRI technology. 

**Abstract (ZH)**: 社会机器人作为锻炼教练维持长期互动时，建立 rapport 至关重要。研究表明，实时模仿患者的运动动作的社会机器人能够在锻炼指导中增强 rapport，特别是在康复锻炼中有效。我们开发了一种新型脑-机器人接口（BRI），使社会机器人教练能够根据患者意图产生的 mental commands 实时模仿患者的锻炼动作。该系统在一项探索性研究中得到了 14 名参与者（3 名理疗师和 11 名中风或其他损伤后恢复的偏瘫患者）的评估。结果显示，系统在 12 个会话中成功展示了运动模仿，但准确性有所差异。参与者对机器人教练持正面评价，表现出较高的信任和接受度，这种接受度未受到 BRI 技术引入的影响。 

---
# Policy Learning for Social Robot-Led Physiotherapy 

**Title (ZH)**: 社会机器人引导物理治疗的策略学习 

**Authors**: Carl Bettosi, Lynne Ballie, Susan Shenkin, Marta Romeo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11297)  

**Abstract**: Social robots offer a promising solution for autonomously guiding patients through physiotherapy exercise sessions, but effective deployment requires advanced decision-making to adapt to patient needs. A key challenge is the scarcity of patient behavior data for developing robust policies. To address this, we engaged 33 expert healthcare practitioners as patient proxies, using their interactions with our robot to inform a patient behavior model capable of generating exercise performance metrics and subjective scores on perceived exertion. We trained a reinforcement learning-based policy in simulation, demonstrating that it can adapt exercise instructions to individual exertion tolerances and fluctuating performance, while also being applicable to patients at different recovery stages with varying exercise plans. 

**Abstract (ZH)**: 社会机器人提供了一种有前景的解决方案，用于自主引导患者完成理疗锻炼 session，但有效的部署需要先进的决策机制以适应患者的需求。一个关键挑战是缺乏足够的患者行为数据以开发出 robust 的策略。为了解决这一问题，我们邀请了 33 名专家医疗从业者作为患者代理，利用他们与我们机器人互动的数据来构建一个患者行为模型，该模型能够生成锻炼表现指标和主观用力感知分数。我们在仿真环境中训练了一种基于强化学习的策略，证明了该策略能够根据不同个体的用力耐受性和变动表现调整锻炼指令，并且适用于处于不同康复阶段且锻炼计划不同的患者。 

---
# Embodied Intelligence in Disassembly: Multimodal Perception Cross-validation and Continual Learning in Neuro-Symbolic TAMP 

**Title (ZH)**: 基于身体智能的拆卸：神经符号TAMP中的多模态感知交叉验证与持续学习 

**Authors**: Ziwen He, Zhigang Wang, Yanlong Peng, Pengxu Chang, Hong Yang, Ming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11270)  

**Abstract**: With the rapid development of the new energy vehicle industry, the efficient disassembly and recycling of power batteries have become a critical challenge for the circular economy. In current unstructured disassembly scenarios, the dynamic nature of the environment severely limits the robustness of robotic perception, posing a significant barrier to autonomous disassembly in industrial applications. This paper proposes a continual learning framework based on Neuro-Symbolic task and motion planning (TAMP) to enhance the adaptability of embodied intelligence systems in dynamic environments. Our approach integrates a multimodal perception cross-validation mechanism into a bidirectional reasoning flow: the forward working flow dynamically refines and optimizes action strategies, while the backward learning flow autonomously collects effective data from historical task executions to facilitate continual system learning, enabling self-optimization. Experimental results show that the proposed framework improves the task success rate in dynamic disassembly scenarios from 81.68% to 100%, while reducing the average number of perception misjudgments from 3.389 to 1.128. This research provides a new paradigm for enhancing the robustness and adaptability of embodied intelligence in complex industrial environments. 

**Abstract (ZH)**: 基于神经符号任务与动作规划的持续学习框架：提升动态环境下游动智能的适应性 

---
# CORB-Planner: Corridor as Observations for RL Planning in High-Speed Flight 

**Title (ZH)**: CORB-Planner: 航道作为观测的高速飞行路径规划方法 

**Authors**: Yechen Zhang, Bin Gao, Gang Wang, Jian Sun, Zhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11240)  

**Abstract**: Reinforcement learning (RL) has shown promise in a large number of robotic control tasks. Nevertheless, its deployment on unmanned aerial vehicles (UAVs) remains challenging, mainly because of reliance on accurate dynamic models and platform-specific sensing, which hinders cross-platform transfer. This paper presents the CORB-Planner (Corridor-as-Observations for RL B-spline planner), a real-time, RL-based trajectory planning framework for high-speed autonomous UAV flight across heterogeneous platforms. The key idea is to combine B-spline trajectory generation with the RL policy producing successive control points with a compact safe flight corridor (SFC) representation obtained via heuristic search. The SFC abstracts obstacle information in a low-dimensional form, mitigating overfitting to platform-specific details and reducing sensitivity to model inaccuracies. To narrow the sim-to-real gap, we adopt an easy-to-hard progressive training pipeline in simulation. A value-based soft decomposed-critic Q (SDCQ) algorithm is used to learn effective policies within approximately ten minutes of training. Benchmarks in simulation and real-world tests demonstrate real-time planning on lightweight onboard hardware and support maximum flight speeds up to 8.2m/s in dense, cluttered environments without external positioning. Compatibility with various UAV configurations (quadrotors, hexarotors) and modest onboard compute underlines the generality and robustness of CORB-Planner for practical deployment. 

**Abstract (ZH)**: 基于走廊观测的强化学习B样条规划器（CORB-Planner）：一种适用于异构平台高速自主 UAV 飞行的实时轨迹规划框架 

---
# MEMBOT: Memory-Based Robot in Intermittent POMDP 

**Title (ZH)**: 基于记忆的间歇性部分可观察马尔可夫决策过程机器人：MEMBOT 

**Authors**: Youzhi Liang, Eyan Noronha  

**Link**: [PDF](https://arxiv.org/pdf/2509.11225)  

**Abstract**: Robotic systems deployed in real-world environments often operate under con- ditions of partial and often intermittent observability, where sensor inputs may be noisy, occluded, or entirely unavailable due to failures or environmental con- straints. Traditional reinforcement learning (RL) approaches that assume full state observability are ill-equipped for such challenges. In this work, we introduce MEMBOT, a modular memory-based architecture designed to address intermittent partial observability in robotic control tasks. MEMBOT decouples belief inference from policy learning through a two-phase training process: an offline multi-task learning pretraining stage that learns a robust task-agnostic latent belief encoder using a reconstruction losses, followed by fine-tuning of task-specific policies using behavior cloning. The belief encoder, implemented as a state-space model (SSM) and a LSTM, integrates temporal sequences of observations and actions to infer latent state representations that persist even when observations are dropped. We train and evaluate MEMBOT on 10 robotic manipulation benchmark tasks from MetaWorld and Robomimic under varying rates of observation dropout. Results show that MEMBOT consistently outperforms both memoryless and naively recur- rent baselines, maintaining up to 80% of peak performance under 50% observation availability. These findings highlight the effectiveness of explicit belief modeling in achieving robust, transferable, and data-efficient policies for real-world partially observable robotic systems. 

**Abstract (ZH)**: 基于模块化记忆架构的MEMBOT：解决机器人控制任务中的间歇性部分可观测性问题 

---
# DreamNav: A Trajectory-Based Imaginative Framework for Zero-Shot Vision-and-Language Navigation 

**Title (ZH)**: DreamNav: 一种基于轨迹的想象框架，用于零样本视觉-语言导航 

**Authors**: Yunheng Wang, Yuetong Fang, Taowen Wang, Yixiao Feng, Yawen Tan, Shuning Zhang, Peiran Liu, Yiding Ji, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11197)  

**Abstract**: Vision-and-Language Navigation in Continuous Environments (VLN-CE), which links language instructions to perception and control in the real world, is a core capability of embodied robots. Recently, large-scale pretrained foundation models have been leveraged as shared priors for perception, reasoning, and action, enabling zero-shot VLN without task-specific training. However, existing zero-shot VLN methods depend on costly perception and passive scene understanding, collapsing control to point-level choices. As a result, they are expensive to deploy, misaligned in action semantics, and short-sighted in planning. To address these issues, we present DreamNav that focuses on the following three aspects: (1) for reducing sensory cost, our EgoView Corrector aligns viewpoints and stabilizes egocentric perception; (2) instead of point-level actions, our Trajectory Predictor favors global trajectory-level planning to better align with instruction semantics; and (3) to enable anticipatory and long-horizon planning, we propose an Imagination Predictor to endow the agent with proactive thinking capability. On VLN-CE and real-world tests, DreamNav sets a new zero-shot state-of-the-art (SOTA), outperforming the strongest egocentric baseline with extra information by up to 7.49\% and 18.15\% in terms of SR and SPL metrics. To our knowledge, this is the first zero-shot VLN method to unify trajectory-level planning and active imagination while using only egocentric inputs. 

**Abstract (ZH)**: 连续环境中基于视觉-语言导航（VLN-CE），将语言指令与现实世界的感知和控制联系起来，是体现式机器人的一项核心能力。 

---
# SAMP: Spatial Anchor-based Motion Policy for Collision-Aware Robotic Manipulators 

**Title (ZH)**: 基于空间锚点的碰撞意识运动策略SAMP 

**Authors**: Kai Chen, Zhihai Bi, Guoyang Zhao, Chunxin Zheng, Yulin Li, Hang Zhao, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.11185)  

**Abstract**: Neural-based motion planning methods have achieved remarkable progress for robotic manipulators, yet a fundamental challenge lies in simultaneously accounting for both the robot's physical shape and the surrounding environment when generating safe and feasible motions. Moreover, existing approaches often rely on simplified robot models or focus primarily on obstacle representation, which can lead to incomplete collision detection and degraded performance in cluttered scenes. To address these limitations, we propose spatial anchor-based motion policy (SAMP), a unified framework that simultaneously encodes the environment and the manipulator using signed distance field (SDF) anchored on a shared spatial grid. SAMP incorporates a dedicated robot SDF network that captures the manipulator's precise geometry, enabling collision-aware reasoning beyond coarse link approximations. These representations are fused on spatial anchors and used to train a neural motion policy that generates smooth, collision-free trajectories in the proposed efficient feature alignment strategy. Experiments conducted in both simulated and real-world environments consistently show that SAMP outperforms existing methods, delivering an 11% increase in success rate and a 7% reduction in collision rate. These results highlight the benefits of jointly modelling robot and environment geometry, demonstrating its practical value in challenging real-world environments. 

**Abstract (ZH)**: 基于神经网络的运动规划方法在机器人 manipulator 中取得了显著进展，但在生成安全可行的运动时，一项基本挑战在于同时考虑机器人的物理形状和周围环境。此外，现有方法往往依赖于简化的机器人模型或主要集中在障碍物表示上，这可能导致碰撞检测不完整并在杂乱场景中表现出较差的性能。为了克服这些限制，我们提出了一种基于空间锚的运动策略（SAMP），这是一种统一框架，可以同时使用带有共享空间网格的空间锚签量距离场（SDF）编码环境和 manipulator。SAMP 结合了一个专用的机器人 SDF 网络，该网络捕获 manipulator 的精确几何形状，从而实现超越粗略链接近似的碰撞感知推理。这些表示在空间锚上融合，并用于训练神经运动策略，以在提出的高效特征对齐策略中生成平滑且无障碍的轨迹。在模拟和真实环境中的实验结果一致显示，SAMP 比现有方法性能更优，成功率提高了 11%，碰撞率降低了 7%。这些结果突显了同时建模机器人和环境几何形状的益处，展示了其在具有挑战性的现实环境中的实际价值。 

---
# RoVerFly: Robust and Versatile Learning-based Control of Quadrotor Across Payload Configurations 

**Title (ZH)**: RoVerFly: 均衡灵活的基于学习的四旋翼无人机载重配置控制 

**Authors**: Mintae Kim, Jiaze Cai, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2509.11149)  

**Abstract**: Designing robust controllers for precise, arbitrary trajectory tracking with quadrotors is challenging due to nonlinear dynamics and underactuation, and becomes harder with flexible cable-suspended payloads that introduce extra degrees of freedom and hybridness. Classical model-based methods offer stability guarantees but require extensive tuning and often do not adapt when the configuration changes, such as when a payload is added or removed, or when the payload mass or cable length varies. We present RoVerFly, a unified learning-based control framework in which a reinforcement learning (RL) policy serves as a robust and versatile tracking controller for standard quadrotors and for cable-suspended payload systems across a range of configurations. Trained with task and domain randomization, the controller is resilient to disturbances and varying dynamics. It achieves strong zero-shot generalization across payload settings, including no payload as well as varying mass and cable length, without controller switching or re-tuning, while retaining the interpretability and structure of a feedback tracking controller. Code and supplementary materials are available at this https URL 

**Abstract (ZH)**: 基于强化学习的鲁棒控制器设计：Multipurpose轨迹跟踪控制器在四旋翼飞行器和平移悬挂载荷系统中的统一学习框架 

---
# ManiVID-3D: Generalizable View-Invariant Reinforcement Learning for Robotic Manipulation via Disentangled 3D Representations 

**Title (ZH)**: ManiVID-3D：基于解耦3D表示的具有一致视角的可泛化强化学习机器人操作方法 

**Authors**: Zheng Li, Pei Qu, Yufei Jia, Shihui Zhou, Haizhou Ge, Jiahang Cao, Jinni Zhou, Guyue Zhou, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.11125)  

**Abstract**: Deploying visual reinforcement learning (RL) policies in real-world manipulation is often hindered by camera viewpoint changes. A policy trained from a fixed front-facing camera may fail when the camera is shifted--an unavoidable situation in real-world settings where sensor placement is hard to manage appropriately. Existing methods often rely on precise camera calibration or struggle with large perspective changes. To address these limitations, we propose ManiVID-3D, a novel 3D RL architecture designed for robotic manipulation, which learns view-invariant representations through self-supervised disentangled feature learning. The framework incorporates ViewNet, a lightweight yet effective module that automatically aligns point cloud observations from arbitrary viewpoints into a unified spatial coordinate system without the need for extrinsic calibration. Additionally, we develop an efficient GPU-accelerated batch rendering module capable of processing over 5000 frames per second, enabling large-scale training for 3D visual RL at unprecedented speeds. Extensive evaluation across 10 simulated and 5 real-world tasks demonstrates that our approach achieves a 44.7% higher success rate than state-of-the-art methods under viewpoint variations while using 80% fewer parameters. The system's robustness to severe perspective changes and strong sim-to-real performance highlight the effectiveness of learning geometrically consistent representations for scalable robotic manipulation in unstructured environments. Our project website can be found in this https URL. 

**Abstract (ZH)**: 基于视觉强化学习的机器人 manipulation 在实际应用中常受相机视角变化的阻碍：一种新型 3D RL 架构 ManiVID-3D 通过自监督的解耦特征学习获得视角不变表示 

---
# FEWT: Improving Humanoid Robot Perception with Frequency-Enhanced Wavelet-based Transformers 

**Title (ZH)**: FEWT: 基于频率增强小波变换的人形机器人感知改进方法 

**Authors**: Jiaxin Huang, Hanyu Liu, Yunsheng Ma, Jian Shen, Yilin Zheng, Jiayi Wen, Baishu Wan, Pan Li, Zhigong Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.11109)  

**Abstract**: The embodied intelligence bridges the physical world and information space. As its typical physical embodiment, humanoid robots have shown great promise through robot learning algorithms in recent years. In this study, a hardware platform, including humanoid robot and exoskeleton-style teleoperation cabin, was developed to realize intuitive remote manipulation and efficient collection of anthropomorphic action data. To improve the perception representation of humanoid robot, an imitation learning framework, termed Frequency-Enhanced Wavelet-based Transformer (FEWT), was proposed, which consists of two primary modules: Frequency-Enhanced Efficient Multi-Scale Attention (FE-EMA) and Time-Series Discrete Wavelet Transform (TS-DWT). By combining multi-scale wavelet decomposition with the residual network, FE-EMA can dynamically fuse features from both time-domain and frequency-domain. This fusion is able to capture feature information across various scales effectively, thereby enhancing model robustness. Experimental performance demonstrates that FEWT improves the success rate of the state-of-the-art algorithm (Action Chunking with Transformers, ACT baseline) by up to 30% in simulation and by 6-12% in real-world. 

**Abstract (ZH)**: 具身智能连接物理世界与信息空间。作为其典型的物理体现，类人机器人通过近年来的机器人学习算法展现了巨大的潜力。在本研究中，开发了一个硬件平台，包括类人机器人和exoskeleton-style远程操作舱，以实现直观的远程操作和高效的人形动作数据采集。为提高类人机器人感知表示，提出了一种模仿学习框架，称为频率增强小波基变换器（FEWT），该框架包括两个主要模块：频率增强高效多尺度注意力（FE-EMA）和时间序列离散小波变换（TS-DWT）。通过结合多尺度波let分解与残差网络，FE-EMA能够动态融合时域和频域特征。这种融合能够有效地捕捉不同尺度下的特征信息，从而增强模型的稳健性。实验性能表明，FEWT将最新的算法（基于转换器的动作切片，ACT基线）在模拟中的成功率提高了30%，在实际应用中提高了6-12%。 

---
# Multi-objective task allocation for electric harvesting robots: a hierarchical route reconstruction approach 

**Title (ZH)**: 基于分层路径重构的多目标任务分配方法：适用于电动收割机器人的任务分配 

**Authors**: Peng Chen, Jing Liang, Hui Song, Kang-Jia Qiao, Cai-Tong Yue, Kun-Jie Yu, Ponnuthurai Nagaratnam Suganthan, Witold Pedrycz  

**Link**: [PDF](https://arxiv.org/pdf/2509.11025)  

**Abstract**: The increasing labor costs in agriculture have accelerated the adoption of multi-robot systems for orchard harvesting. However, efficiently coordinating these systems is challenging due to the complex interplay between makespan and energy consumption, particularly under practical constraints like load-dependent speed variations and battery limitations. This paper defines the multi-objective agricultural multi-electrical-robot task allocation (AMERTA) problem, which systematically incorporates these often-overlooked real-world constraints. To address this problem, we propose a hybrid hierarchical route reconstruction algorithm (HRRA) that integrates several innovative mechanisms, including a hierarchical encoding structure, a dual-phase initialization method, task sequence optimizers, and specialized route reconstruction operators. Extensive experiments on 45 test instances demonstrate HRRA's superior performance against seven state-of-the-art algorithms. Statistical analysis, including the Wilcoxon signed-rank and Friedman tests, empirically validates HRRA's competitiveness and its unique ability to explore previously inaccessible regions of the solution space. In general, this research contributes to the theoretical understanding of multi-robot coordination by offering a novel problem formulation and an effective algorithm, thereby also providing practical insights for agricultural automation. 

**Abstract (ZH)**: 农业多电气机器人任务分配（AMERTA）问题及其混合分层路径重构算法的研究 

---
# Autonomous Close-Proximity Photovoltaic Panel Coating Using a Quadcopter 

**Title (ZH)**: 自主近距光伏面板涂层四旋翼无人机系统 

**Authors**: Dimitri Jacquemont, Carlo Bosio, Teaya Yang, Ruiqi Zhang, Ozgur Orun, Shuai Li, Reza Alam, Thomas M. Schutzius, Simo A. Makiharju, Mark W. Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2509.10979)  

**Abstract**: Photovoltaic (PV) panels are becoming increasingly widespread in the domain of renewable energy, and thus, small efficiency gains can have massive effects. Anti-reflective and self-cleaning coatings enhance panel performance but degrade over time, requiring periodic reapplication. Uncrewed Aerial Vehicles (UAVs) offer a flexible and autonomous way to apply protective coatings more often and at lower cost compared to traditional manual coating methods. In this letter, we propose a quadcopter-based system, equipped with a liquid dispersion mechanism, designed to automate such tasks. The localization stack only uses onboard sensors, relying on visual-inertial odometry and the relative position of the PV panel detected with respect to the quadcopter. The control relies on a model-based controller that accounts for the ground effect and the mass decrease of the quadcopter during liquid dispersion. We validate the autonomy capabilities of our system through extensive indoor and outdoor experiments. 

**Abstract (ZH)**: 基于四旋翼无人机的自动涂覆系统在光伏面板上的应用研究 

---
# Pogosim - a Simulator for Pogobot robots 

**Title (ZH)**: Pogosim - Pogobot机器人的仿真器 

**Authors**: Leo Cazenille, Loona Macabre, Nicolas Bredeche  

**Link**: [PDF](https://arxiv.org/pdf/2509.10968)  

**Abstract**: Pogobots are a new type of open-source/open-hardware robots specifically designed for swarm robotics research. Their cost-effective and modular design, complemented by vibration-based and wheel-based locomotion, fast infrared communication and extensive software architecture facilitate the implementation of swarm intelligence algorithms. However, testing even simple distributed algorithms directly on robots is particularly labor-intensive. Scaling to more complex problems or calibrate user code parameters will have a prohibitively high strain on available resources. In this article we present Pogosim, a fast and scalable simulator for Pogobots, designed to reduce as much as possible algorithm development costs. The exact same code will be used in both simulation and to experimentally drive real robots. This article details the software architecture of Pogosim, explain how to write configuration files and user programs and how simulations approximate or differ from experiments. We describe how a large set of simulations can be launched in parallel, how to retrieve and analyze the simulation results, and how to optimize user code parameters using optimization algorithms. 

**Abstract (ZH)**: Pogobots的快速可扩展模拟器Pogosim：减少算法开发成本的研究 

---
# ImMimic: Cross-Domain Imitation from Human Videos via Mapping and Interpolation 

**Title (ZH)**: ImMimic: 通过映射和插值实现跨域 imitation 从人类视频 

**Authors**: Yangcen Liu, Woo Chul Shin, Yunhai Han, Zhenyang Chen, Harish Ravichandar, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10952)  

**Abstract**: Learning robot manipulation from abundant human videos offers a scalable alternative to costly robot-specific data collection. However, domain gaps across visual, morphological, and physical aspects hinder direct imitation. To effectively bridge the domain gap, we propose ImMimic, an embodiment-agnostic co-training framework that leverages both human videos and a small amount of teleoperated robot demonstrations. ImMimic uses Dynamic Time Warping (DTW) with either action- or visual-based mapping to map retargeted human hand poses to robot joints, followed by MixUp interpolation between paired human and robot trajectories. Our key insights are (1) retargeted human hand trajectories provide informative action labels, and (2) interpolation over the mapped data creates intermediate domains that facilitate smooth domain adaptation during co-training. Evaluations on four real-world manipulation tasks (Pick and Place, Push, Hammer, Flip) across four robotic embodiments (Robotiq, Fin Ray, Allegro, Ability) show that ImMimic improves task success rates and execution smoothness, highlighting its efficacy to bridge the domain gap for robust robot manipulation. The project website can be found at this https URL. 

**Abstract (ZH)**: 从丰富的人类视频中学习机器人操作为替代昂贵的机器人专用数据收集提供了可扩展的替代方案。然而，视觉、形态和物理方面的领域差异阻碍了直接模仿。为了有效缩小领域差距，我们提出了ImMimic，一个不依赖于实体的联合训练框架，利用人类视频和少量的远程操作机器人演示。ImMimic 使用动态时间规整（DTW），基于动作或视觉的映射来将重新定向的人类手部姿势映射到机器人关节，并在配对的人类和机器人轨迹之间进行 MixUp 插值。我们的关键见解是（1）重新定向的人类手部轨迹提供了信息性的动作标签，（2）在映射数据上进行插值创建了中间领域，有助于联合训练期间的平滑领域适应。在四个实际操作任务（拾取和放置、推送、锤击、翻转）和四种机器人实体（Robotiq、Fin Ray、Allegro、Ability）上的评估表明，ImMimic 提高了操作成功率和执行平滑度，凸显了其在增强机器人操作鲁棒性方面缩小领域差距的有效性。项目网站详见this https URL。 

---
# ViSTR-GP: Online Cyberattack Detection via Vision-to-State Tensor Regression and Gaussian Processes in Automated Robotic Operations 

**Title (ZH)**: ViSTR-GP: 自动化机器人操作中基于视觉至状态张量回归和高斯过程的在线网络攻击检测 

**Authors**: Navid Aftabi, Philip Samaha, Jin Ma, Long Cheng, Ramy Harik, Dan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.10948)  

**Abstract**: Industrial robotic systems are central to automating smart manufacturing operations. Connected and automated factories face growing cybersecurity risks that can potentially cause interruptions and damages to physical operations. Among these attacks, data-integrity attacks often involve sophisticated exploitation of vulnerabilities that enable an attacker to access and manipulate the operational data and are hence difficult to detect with only existing intrusion detection or model-based detection. This paper addresses the challenges in utilizing existing side-channels to detect data-integrity attacks in robotic manufacturing processes by developing an online detection framework, ViSTR-GP, that cross-checks encoder-reported measurements against a vision-based estimate from an overhead camera outside the controller's authority. In this framework, a one-time interactive segmentation initializes SAM-Track to generate per-frame masks. A low-rank tensor-regression surrogate maps each mask to measurements, while a matrix-variate Gaussian process models nominal residuals, capturing temporal structure and cross-joint correlations. A frame-wise test statistic derived from the predictive distribution provides an online detector with interpretable thresholds. We validate the framework on a real-world robotic testbed with synchronized video frame and encoder data, collecting multiple nominal cycles and constructing replay attack scenarios with graded end-effector deviations. Results on the testbed indicate that the proposed framework recovers joint angles accurately and detects data-integrity attacks earlier with more frequent alarms than all baselines. These improvements are most evident in the most subtle attacks. These results show that plants can detect data-integrity attacks by adding an independent physical channel, bypassing the controller's authority, without needing complex instrumentation. 

**Abstract (ZH)**: 基于视觉的在线数据完整性攻击检测框架：应用于机器人制造过程的安全性增强 

---
# Design of scalable orthogonal digital encoding architecture for large-area flexible tactile sensing in robotics 

**Title (ZH)**: 面向机器人领域的大面积柔性触觉感知可扩展正交数字编码架构设计 

**Authors**: Weijie Liu, Ziyi Qiu, Shihang Wang, Deqing Mei, Yancheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10888)  

**Abstract**: Human-like embodied tactile perception is crucial for the next-generation intelligent robotics. Achieving large-area, full-body soft coverage with high sensitivity and rapid response, akin to human skin, remains a formidable challenge due to critical bottlenecks in encoding efficiency and wiring complexity in existing flexible tactile sensors, thus significantly hinder the scalability and real-time performance required for human skin-level tactile perception. Herein, we present a new architecture employing code division multiple access-inspired orthogonal digital encoding to overcome these challenges. Our decentralized encoding strategy transforms conventional serial signal transmission by enabling parallel superposition of energy-orthogonal base codes from distributed sensing nodes, drastically reducing wiring requirements and increasing data throughput. We implemented and validated this strategy with off-the-shelf 16-node sensing array to reconstruct the pressure distribution, achieving a temporal resolution of 12.8 ms using only a single transmission wire. Crucially, the architecture can maintain sub-20ms latency across orders-of-magnitude variations in node number (to thousands of nodes). By fundamentally redefining signal encoding paradigms in soft electronics, this work opens new frontiers in developing scalable embodied intelligent systems with human-like sensory capabilities. 

**Abstract (ZH)**: 类人的身体触觉感知对于下一代智能机器人至关重要。克服现有柔性触觉传感器在编码效率和布线复杂性方面的瓶颈，以实现大面积、全身软覆盖的高灵敏度和快速响应能力，如同人类皮肤依然是一项艰巨的挑战，这严重影响了达到人类皮肤级触觉感知所需的扩展性和实时性能。在此，我们提出一种新的架构，采用基于码分多址的正交数字编码策略来克服这些挑战。我们的去中心化编码策略通过使分布式传感节点能够并行叠加能量正交的基础编码，从而大幅减少布线需求并提高数据吞吐量，重构压力分布。我们使用商用的16节点传感阵列实施并验证了这一策略，仅使用一根传输线就实现了12.8毫秒的时间分辨率。 crucial地，该架构能够保持低于20毫秒的延迟，在节点数量从数百到数千的数量级变化时依然有效。通过从根本上重新定义软电子器件中的信号编码 paradigms，本项工作为开发具有类人感知能力的可扩展体域智能系统开辟了新的前沿。 

---
# Nav-R1: Reasoning and Navigation in Embodied Scenes 

**Title (ZH)**: Nav-R1: 身体化场景中的推理与导航 

**Authors**: Qingxiang Liu, Ting Huang, Zeyu Zhang, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10884)  

**Abstract**: Embodied navigation requires agents to integrate perception, reasoning, and action for robust interaction in complex 3D environments. Existing approaches often suffer from incoherent and unstable reasoning traces that hinder generalization across diverse environments, and difficulty balancing long-horizon semantic reasoning with low-latency control for real-time navigation. To address these challenges, we propose Nav-R1, an embodied foundation model that unifies reasoning in embodied environments. We first construct Nav-CoT-110K, a large-scale dataset of step-by-step Chains-of-Thought (CoT) for embodied tasks, which enables cold-start initialization with structured reasoning. Building on this foundation, we design a GRPO-based reinforcement learning framework with three complementary rewards: format, understanding, and navigation, to improve structural adherence, semantic grounding, and path fidelity. Furthermore, we introduce a Fast-in-Slow reasoning paradigm, decoupling deliberate semantic reasoning from low-latency reactive control for efficient yet coherent navigation. Extensive evaluations on embodied AI benchmarks demonstrate that Nav-R1 consistently outperforms strong baselines, with over 8% average improvement in reasoning and navigation performance. Real-world deployment on a mobile robot further validates its robustness under limited onboard resources. Code: this https URL. Website: this https URL. 

**Abstract (ZH)**: 基于知觉的导航要求智能体整合感知、推理和行动，以在复杂的3D环境中实现稳健的交互。现有的方法往往存在不连贯且不稳定的推理轨迹，这阻碍了在多种环境之间的泛化，并且难以在长远语义推理与低延迟控制之间进行权衡，以实现实时导航。为了解决这些挑战，我们提出了一种称为Nav-R1的基于知觉的基座模型，它统一了在基于知觉环境中的推理。我们首先构建了一个包含11万步骤推理链（CoT）的大规模数据集Nav-CoT-110K，这使得智能体可以从结构化的推理开始。在此基础上，我们设计了一个基于GRPO的强化学习框架，并引入了三种互补奖励：格式、理解、导航，以提高结构一致性、语义接地和路径精度。此外，我们引入了快速内在延时推理范式，将慎重的语义推理与低延迟的反应控制脱钩，以实现高效且连贯的导航。广泛的实验表明，Nav-R1在语义推理和导航性能上均显著优于强基线模型，平均提高超过8%。实地部署在移动机器人上进一步验证了其在有限的机载资源下的鲁棒性。代码：this https URL. 网站：this https URL。 

---
# A Universal Wire Testing Machine for Enhancing the Performance of Wire-Driven Robots 

**Title (ZH)**: 用于增强 wire-driven 机器人性能的通用导线测试机 

**Authors**: Temma Suzuki, Kento Kawaharazuka, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2509.10862)  

**Abstract**: Compared with gears and linkages, wires constitute a lightweight, low-friction transmission mechanism. However, because wires are flexible materials, they tend to introduce large modeling errors, and their adoption in industrial and research robots remains this http URL this study, we built a Universal Wire Testing Machine that enables measurement and adjustment of wire characteristics to improve the performance of wire-driven mechanisms. Using this testing machine, we carried out removal of initial wire stretch, measurement of tension transmission efficiency for eight different diameters of passive pulleys, and measurement of the dynamic behavior of variable-length wires. Finally, we applied the data obtained from this testing machine to the force control of an actual wire-driven robot, reducing the end-effector force error. 

**Abstract (ZH)**: 与齿轮和连杆相比， wires 构成了一种轻量级、低摩擦的传动机制。然而，由于 wires 是一种柔性材料，它们容易引入较大的建模误差，其在工业和研究机器人中的应用尚存在局限。本研究中，我们构建了一台通用的 wire 测试机，用于测量和调整 wire 特性以提高 wire 驱动机制的性能。使用该测试机，我们进行了初始 wire 拉伸的去除、不同直径被动滑轮的张力传递效率测量以及可变长度 wire 的动态行为测量。最后，我们将该测试机所获取的数据应用于实际的 wire 驱动机器人中的力控制，减少了末端执行器力的误差。 

---
# Follow-Bench: A Unified Motion Planning Benchmark for Socially-Aware Robot Person Following 

**Title (ZH)**: Follow-Bench: 一种面向社交aware机器人行人跟随的统一运动规划基准测试 

**Authors**: Hanjing Ye, Weixi Situ, Jianwei Peng, Yu Zhan, Bingyi Xia, Kuanqi Cai, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10796)  

**Abstract**: Robot person following (RPF) -- mobile robots that follow and assist a specific person -- has emerging applications in personal assistance, security patrols, eldercare, and logistics. To be effective, such robots must follow the target while ensuring safety and comfort for both the target and surrounding people. In this work, we present the first end-to-end study of RPF, which (i) surveys representative scenarios, motion-planning methods, and evaluation metrics with a focus on safety and comfort; (ii) introduces Follow-Bench, a unified benchmark simulating diverse scenarios, including various target trajectory patterns, dynamic-crowd flows, and environmental layouts; and (iii) re-implements six popular RPF planners, ensuring that both safety and comfort are systematically considered. Moreover, we evaluate the two highest-performing planners from our benchmark on a differential-drive robot to provide insights into real-world deployment. Extensive simulation and real-world experiments provide quantitative insights into the safety-comfort trade-offs of existing planners, while revealing open challenges and future research directions. 

**Abstract (ZH)**: 机器人人群跟随（RPF）——能够跟随并协助特定人的移动机器人——在个人辅助、安全巡逻、养老服务和物流等领域具有新兴应用。为了有效工作，这类机器人必须在确保目标人物和周围人员的安全与舒适的同时跟随目标。在本项工作中，我们提出了第一个端到端的RPF研究，包括（i）回顾代表性场景、运动规划方法以及以安全和舒适为重点的评估指标；（ii）介绍Follow-Bench，一个统一基准，模拟各种场景，包括多种目标轨迹模式、动态人群流动和环境布局；（iii）重新实现六种流行的RPF规划器，确保在所有情况下都系统地考虑安全和舒适性。此外，我们在差速驱动机器人上评估了基准中最出色的两个规划器，以提供实际部署的见解。广泛的模拟和实地实验提供了有关现有规划器的安全-舒适权衡的定量见解，同时揭示了开放挑战和未来的研究方向。 

---
# RSL-RL: A Learning Library for Robotics Research 

**Title (ZH)**: RSL-RL：机器人研究的学习库 

**Authors**: Clemens Schwarke, Mayank Mittal, Nikita Rudin, David Hoeller, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2509.10771)  

**Abstract**: RSL-RL is an open-source Reinforcement Learning library tailored to the specific needs of the robotics community. Unlike broad general-purpose frameworks, its design philosophy prioritizes a compact and easily modifiable codebase, allowing researchers to adapt and extend algorithms with minimal overhead. The library focuses on algorithms most widely adopted in robotics, together with auxiliary techniques that address robotics-specific challenges. Optimized for GPU-only training, RSL-RL achieves high-throughput performance in large-scale simulation environments. Its effectiveness has been validated in both simulation benchmarks and in real-world robotic experiments, demonstrating its utility as a lightweight, extensible, and practical framework to develop learning-based robotic controllers. The library is open-sourced at: this https URL. 

**Abstract (ZH)**: RSL-RL是一个针对机器人社区特定需求的开源强化学习库。 

---
# FastTrack: GPU-Accelerated Tracking for Visual SLAM 

**Title (ZH)**: FastTrack：视觉SLAM中的GPU加速跟踪 

**Authors**: Kimia Khabiri, Parsa Hosseininejad, Shishir Gopinath, Karthik Dantu, Steven Y. Ko  

**Link**: [PDF](https://arxiv.org/pdf/2509.10757)  

**Abstract**: The tracking module of a visual-inertial SLAM system processes incoming image frames and IMU data to estimate the position of the frame in relation to the map. It is important for the tracking to complete in a timely manner for each frame to avoid poor localization or tracking loss. We therefore present a new approach which leverages GPU computing power to accelerate time-consuming components of tracking in order to improve its performance. These components include stereo feature matching and local map tracking. We implement our design inside the ORB-SLAM3 tracking process using CUDA. Our evaluation demonstrates an overall improvement in tracking performance of up to 2.8x on a desktop and Jetson Xavier NX board in stereo-inertial mode, using the well-known SLAM datasets EuRoC and TUM-VI. 

**Abstract (ZH)**: 视觉惯性SLAM系统中基于GPU的跟踪模块利用GPU计算能力加速耗时的跟踪组件，以提高跟踪性能。这些组件包括立体特征匹配和局部地图跟踪。在ORB-SLAM3跟踪过程中使用CUDA实现我们的设计。评估结果显示，在桌面和Jetson Xavier NX板的立体惯性模式下，使用著名的SLAM数据集EuRoC和TUM-VI，跟踪性能整体提升最高可达2.8倍。 

---
# Analytical Design and Development of a Modular and Intuitive Framework for Robotizing and Enhancing the Existing Endoscopic Procedures 

**Title (ZH)**: 模块化和直观框架的分析设计与开发：用于现有内镜手术的机器人化与增强 

**Authors**: Mohammad Rafiee Javazm, Yash Kulkarni, Jiaqi Xue, Naruhiko Ikoma, Farshid Alambeigi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10735)  

**Abstract**: Despite the widespread adoption of endoscopic devices for several cancer screening procedures, manual control of these devices still remains challenging for clinicians, leading to several critical issues such as increased workload, fatigue, and distractions. To address these issues, in this paper, we introduce the design and development of an intuitive, modular, and easily installable mechatronic framework. This framework includes (i) a novel nested collet-chuck gripping mechanism that can readily be integrated and assembled with the existing endoscopic devices and control their bending degrees-of-freedom (DoFs); (ii) a feeder mechanism that can control the insertion/retraction DoF of a colonoscope, and (iii) a complementary and intuitive user interface that enables simultaneous control of all DoFs during the procedure. To analyze the design of the proposed mechanisms, we also introduce a mathematical modeling approach and a design space for optimal selection of the parameters involved in the design of gripping and feeder mechanisms. Our simulation and experimental studies thoroughly demonstrate the performance of the proposed mathematical modeling and robotic framework. 

**Abstract (ZH)**: 尽管内窥镜设备在多种癌症筛查程序中的广泛应用，临床医生对手动控制这些设备仍然面临挑战，导致诸如工作量增加、疲劳和分心等一系列关键问题。为解决这些问题，本文介绍了直观、模块化、易于安装的机电一体化框架的设计与开发。该框架包括（i）一种新颖的嵌套卡盘夹持机制，可以轻松与现有的内窥镜设备集成和组装，并控制其弯曲自由度（DoF）；（ii）一种可控制结肠镜插入/撤出自由度的供料机制；以及（iii）一种互补的直观用户界面，使操作者能够在操作过程中同时控制所有自由度。为了分析所提议机制的设计，我们还引入了数学建模方法和一个设计空间，用于优化夹持和供料机制参数的选择。我们的模拟和实验研究充分展示了所提数学建模和机器人框架的性能。 

---
# A Survey on LiDAR-based Autonomous Aerial Vehicles 

**Title (ZH)**: 基于LiDAR的自主无人机综述 

**Authors**: Yunfan Ren, Yixi Cai, Haotian Li, Nan Chen, Fangcheng Zhu, Longji Yin, Fanze Kong, Rundong Li, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10730)  

**Abstract**: This survey offers a comprehensive overview of recent advancements in LiDAR-based autonomous Unmanned Aerial Vehicles (UAVs), covering their design, perception, planning, and control strategies. Over the past decade, LiDAR technology has become a crucial enabler for high-speed, agile, and reliable UAV navigation, especially in GPS-denied environments. The paper begins by examining the evolution of LiDAR sensors, emphasizing their unique advantages such as high accuracy, long-range depth measurements, and robust performance under various lighting conditions, making them particularly well-suited for UAV applications. The integration of LiDAR with UAVs has significantly enhanced their autonomy, enabling complex missions in diverse and challenging environments. Subsequently, we explore essential software components, including perception technologies for state estimation and mapping, as well as trajectory planning and control methodologies, and discuss their adoption in LiDAR-based UAVs. Additionally, we analyze various practical applications of the LiDAR-based UAVs, ranging from industrial operations to supporting different aerial platforms and UAV swarm deployments. The survey concludes by discussing existing challenges and proposing future research directions to advance LiDAR-based UAVs and enhance multi-UAV collaboration. By synthesizing recent developments, this paper aims to provide a valuable resource for researchers and practitioners working to push the boundaries of LiDAR-based UAV systems. 

**Abstract (ZH)**: LiDAR为基础的自主无人机Recent进展综述 

---
# STL-Based Motion Planning and Uncertainty-Aware Risk Analysis for Human-Robot Collaboration with a Multi-Rotor Aerial Vehicle 

**Title (ZH)**: 基于STL的运动规划及多旋翼飞行器人机协作中不确定性 Aware 风险分析 

**Authors**: Giuseppe Silano, Amr Afifi, Martin Saska, Antonio Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10692)  

**Abstract**: This paper presents a novel approach to motion planning and risk analysis for enhancing human-robot collaboration using a Multi-Rotor Aerial Vehicle (MRAV). The proposed method uses Signal Temporal Logic (STL) to encode key mission objectives, such as safety, timing, and human preferences, with a strong focus on ergonomics and comfort. An optimization framework generates dynamically feasible trajectories while considering the MRAV's physical constraints. Given the nonlinear and non-convex nature of the problem, smooth approximations and gradient-based techniques assist in handling the problem's computational complexity. Additionally, an uncertainty-aware risk analysis is incorporated to assess potential deviations from the mission specifications, providing insights into the likelihood of mission success under uncertain conditions. Further, an event-triggered replanning strategy is implemented to respond to unforeseen events and external disturbances. The approach is validated through MATLAB and Gazebo simulations, using an object handover task in a mock-up environment inspired by power line maintenance scenarios. The results highlight the method's effectiveness in achieving safe, efficient, and resilient human-robot collaboration. 

**Abstract (ZH)**: 基于多旋翼无人机的运动规划与风险分析新方法：提升人机协作安全性与舒适性 

---
# Large Foundation Models for Trajectory Prediction in Autonomous Driving: A Comprehensive Survey 

**Title (ZH)**: 大型基础模型在自动驾驶路径预测中的应用：一项综合性综述 

**Authors**: Wei Dai, Shengen Wu, Wei Wu, Zhenhao Wang, Sisuo Lyu, Haicheng Liao, Limin Yu, Weiping Ding, Runwei Guan, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.10570)  

**Abstract**: Trajectory prediction serves as a critical functionality in autonomous driving, enabling the anticipation of future motion paths for traffic participants such as vehicles and pedestrians, which is essential for driving safety. Although conventional deep learning methods have improved accuracy, they remain hindered by inherent limitations, including lack of interpretability, heavy reliance on large-scale annotated data, and weak generalization in long-tail scenarios. The rise of Large Foundation Models (LFMs) is transforming the research paradigm of trajectory prediction. This survey offers a systematic review of recent advances in LFMs, particularly Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) for trajectory prediction. By integrating linguistic and scene semantics, LFMs facilitate interpretable contextual reasoning, significantly enhancing prediction safety and generalization in complex environments. The article highlights three core methodologies: trajectory-language mapping, multimodal fusion, and constraint-based reasoning. It covers prediction tasks for both vehicles and pedestrians, evaluation metrics, and dataset analyses. Key challenges such as computational latency, data scarcity, and real-world robustness are discussed, along with future research directions including low-latency inference, causality-aware modeling, and motion foundation models. 

**Abstract (ZH)**: 大型基础模型在轨迹预测中的进展：语言与多模态方法的研究 

---
# Deceptive Risk Minimization: Out-of-Distribution Generalization by Deceiving Distribution Shift Detectors 

**Title (ZH)**: 欺骗性的风险最小化：通过欺骗分布偏移检测实现分布外泛化 

**Authors**: Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2509.12081)  

**Abstract**: This paper proposes deception as a mechanism for out-of-distribution (OOD) generalization: by learning data representations that make training data appear independent and identically distributed (iid) to an observer, we can identify stable features that eliminate spurious correlations and generalize to unseen domains. We refer to this principle as deceptive risk minimization (DRM) and instantiate it with a practical differentiable objective that simultaneously learns features that eliminate distribution shifts from the perspective of a detector based on conformal martingales while minimizing a task-specific loss. In contrast to domain adaptation or prior invariant representation learning methods, DRM does not require access to test data or a partitioning of training data into a finite number of data-generating domains. We demonstrate the efficacy of DRM on numerical experiments with concept shift and a simulated imitation learning setting with covariate shift in environments that a robot is deployed in. 

**Abstract (ZH)**: 本文提出欺骗作为一种机制来实现外部分布外（OOD）泛化：通过学习数据表示使训练数据对于观察者看来独立且同分布（iid），我们可以识别出稳定的特征以消除伪相关并泛化到未见过的领域。我们将这一原则称为欺骗性风险最小化（DRM），并借助基于一致性马氏链的检测器同时从检测视角学习消除分布偏移的特征，同时最小化特定任务的损失。与领域适应或先验不变表示学习方法不同，DRM 不需要访问测试数据或将训练数据划分为有限数量的数据生成领域。我们在概念漂移的数值实验以及机器人部署环境中由协变量漂移引起的模拟模仿学习设置中展示了DRM的有效性。 

---
# Learning to Generate 4D LiDAR Sequences 

**Title (ZH)**: 学习生成4D LiDAR序列 

**Authors**: Ao Liang, Youquan Liu, Yu Yang, Dongyue Lu, Linfeng Li, Lingdong Kong, Huaici Zhao, Wei Tsang Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11959)  

**Abstract**: While generative world models have advanced video and occupancy-based data synthesis, LiDAR generation remains underexplored despite its importance for accurate 3D perception. Extending generation to 4D LiDAR data introduces challenges in controllability, temporal stability, and evaluation. We present LiDARCrafter, a unified framework that converts free-form language into editable LiDAR sequences. Instructions are parsed into ego-centric scene graphs, which a tri-branch diffusion model transforms into object layouts, trajectories, and shapes. A range-image diffusion model generates the initial scan, and an autoregressive module extends it into a temporally coherent sequence. The explicit layout design further supports object-level editing, such as insertion or relocation. To enable fair assessment, we provide EvalSuite, a benchmark spanning scene-, object-, and sequence-level metrics. On nuScenes, LiDARCrafter achieves state-of-the-art fidelity, controllability, and temporal consistency, offering a foundation for LiDAR-based simulation and data augmentation. 

**Abstract (ZH)**: 尽管生成型世界模型在视频和占用数据合成方面取得了进展，但LiDAR生成仍然未被充分探索，尽管其对于准确的3D感知非常重要。将生成扩展到4D LiDAR数据引入了可控性、时间稳定性以及评估方面的挑战。我们介绍了一种统一框架LiDARCrafter，该框架将自由形式的语言转换为可编辑的LiDAR序列。指令被解析为以自我为中心的场景图，随后由三分支扩散模型转换为对象布局、轨迹和形状。范围图像扩散模型生成初始扫描，而自回归模块将其扩展为时间连贯的序列。明确的设计布局进一步支持对象级别的编辑，如插入或重新定位。为了实现公平的评估，我们提供了EvalSuite基准，该基准涵盖了场景、对象和序列级别的指标。在nuScenes数据集上，LiDARCrafter在保真度、可控性和时间一致性方面获得了最先进的性能，为基于LiDAR的模拟和数据增强提供了基础。 

---
# Growing Perspectives: Modelling Embodied Perspective Taking and Inner Narrative Development Using Large Language Models 

**Title (ZH)**: 扩展视角：使用大规模语言模型建模体态视角推理和内心叙事发展 

**Authors**: Sabrina Patania, Luca Annese, Anna Lambiase, Anita Pellegrini, Tom Foulsham, Azzurra Ruggeri, Silvia Rossi, Silvia Serino, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2509.11868)  

**Abstract**: Language and embodied perspective taking are essential for human collaboration, yet few computational models address both simultaneously. This work investigates the PerspAct system [1], which integrates the ReAct (Reason and Act) paradigm with Large Language Models (LLMs) to simulate developmental stages of perspective taking, grounded in Selman's theory [2]. Using an extended director task, we evaluate GPT's ability to generate internal narratives aligned with specified developmental stages, and assess how these influence collaborative performance both qualitatively (action selection) and quantitatively (task efficiency). Results show that GPT reliably produces developmentally-consistent narratives before task execution but often shifts towards more advanced stages during interaction, suggesting that language exchanges help refine internal representations. Higher developmental stages generally enhance collaborative effectiveness, while earlier stages yield more variable outcomes in complex contexts. These findings highlight the potential of integrating embodied perspective taking and language in LLMs to better model developmental dynamics and stress the importance of evaluating internal speech during combined linguistic and embodied tasks. 

**Abstract (ZH)**: 语言和具身视角转换是人类协作的关键，然而很少有计算模型能够同时处理这两方面。本研究探讨了PerspAct系统 [1]，该系统将ReAct（推理和行动）范式与大规模语言模型（LLMs）结合起来，根据Selman理论 [2] 模拟视角转换的发展阶段。通过扩展导演任务，我们评估了GPT生成与指定发展阶段一致的内部叙述的能力，并分析这些叙述在定性和定量层面上对协作性能的影响（包括动作选择和任务效率）。研究结果表明，在任务执行前，GPT能够可靠地生成符合发展阶段的叙述，但在互动过程中往往转向更高级的发展阶段，这表明语言交流有助于细化内部表征。更高的发展阶段通常可以提升协作效果，而较早的发展阶段在复杂情境下则可能导致更具有变异性的影响。这些发现突显了在LLMs中整合具身视角转换和语言的潜在价值，以更好地模拟发展阶段动力学，并强调了在结合语言和具身任务中评估内部言语的重要性。 

---
# Time to Play: Simulating Early-Life Animal Dynamics Enhances Robotics Locomotion Discovery 

**Title (ZH)**: 时间来玩：模拟早期生命动物动态增强机器人运动发现 

**Authors**: Paul Templier, Hannah Janmohamed, David Labonte, Antoine Cully  

**Link**: [PDF](https://arxiv.org/pdf/2509.11755)  

**Abstract**: Developmental changes in body morphology profoundly shape locomotion in animals, yet artificial agents and robots are typically trained under static physical parameters. Inspired by ontogenetic scaling of muscle power in biology, we propose Scaling Mechanical Output over Lifetime (SMOL), a novel curriculum that dynamically modulates robot actuator strength to mimic natural variations in power-to-weight ratio during growth and ageing. Integrating SMOL into the MAP-Elites quality-diversity framework, we vary the torque in standard robotics tasks to mimic the evolution of strength in animals as they grow up and as their body changes. Through comprehensive empirical evaluation, we show that the SMOL schedule consistently elevates both performance and diversity of locomotion behaviours across varied control scenarios, by allowing agents to leverage advantageous physics early on to discover skills that act as stepping stones when they reach their final standard body properties. Based on studies of the total power output in humans, we also implement the SMOL-Human schedule that models isometric body variations due to non-linear changes like puberty, and study its impact on robotics locomotion. 

**Abstract (ZH)**: 基于生命周期肌肉功率缩放的动态机器人训练方法 

---
# SafeDiver: Cooperative AUV-USV Assisted Diver Communication via Multi-agent Reinforcement Learning Approach 

**Title (ZH)**: SafeDiver: 多智能体强化学习辅助的AUV-USV协同潜水员通信 

**Authors**: Tinglong Deng, Hang Tao, Xinxiang Wang, Yinyan Wang, Hanjiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11508)  

**Abstract**: As underwater human activities are increasing, the demand for underwater communication service presents a significant challenge. Existing underwater diver communication methods face hurdles due to inherent disadvantages and complex underwater environments. To address this issue, we propose a scheme that utilizes maritime unmanned systems to assist divers with reliable and high-speed communication. Multiple AUVs are equipped with optical and acoustic multimodal communication devices as relay nodes, providing adaptive communication services based on changes in the diver's activity area. By using a multi-agent reinforcement learning (MARL) approach to control the cooperative movement of AUVs, high-speed and reliable data transmission between divers can be achieved. At the same time, utilizing the advantages of on-demand deployment and wide coverage of unmanned surface vehicles (USVs) as surface relay nodes to coordinate and forward information from AUVs, and controlling AUVs to adaptively select relay USV nodes for data transmission, high-quality communication between divers and surface platform can be achieved. Through simulation verification, the proposed scheme can effectively achieve reliable and high-speed communication for divers. 

**Abstract (ZH)**: 随着水下人类活动的增加，对水下通信服务的需求呈现出显著的挑战。现有水下潜水员通信方法因固有的缺点和复杂的水下环境而面临挑战。为此，我们提出了一种利用 maritime unmanned systems 来协助潜水员进行可靠和高速通信的方案。多个 AUV 装备有光学和声学多模通信设备作为中继节点，基于潜水员活动区域的变化提供适应性通信服务。通过使用多智能体强化学习（MARL）方法来控制 AUV 的协同运动，可以实现潜水员之间的高速和可靠数据传输。同时，利用无人驾驶水面车辆（USVs）的按需部署和广泛覆盖作为表面中继节点的优势，协调和转发 AUV 的信息，并控制 AUV 适配性地选择中继 USV 节点进行数据传输，从而实现潜水员与表面平台之间的高质量通信。通过仿真验证，所提出的方案可以有效实现潜水员的可靠和高速通信。 

---
# Cross-Platform Scaling of Vision-Language-Action Models from Edge to Cloud GPUs 

**Title (ZH)**: 跨平台从边缘到云GPU的视觉-语言-动作模型扩展研究 

**Authors**: Amir Taherin, Juyi Lin, Arash Akbari, Arman Akbari, Pu Zhao, Weiwei Chen, David Kaeli, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11480)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as powerful generalist policies for robotic control, yet their performance scaling across model architectures and hardware platforms, as well as their associated power budgets, remain poorly understood. This work presents an evaluation of five representative VLA models -- spanning state-of-the-art baselines and two newly proposed architectures -- targeting edge and datacenter GPU platforms. Using the LIBERO benchmark, we measure accuracy alongside system-level metrics, including latency, throughput, and peak memory usage, under varying edge power constraints and high-performance datacenter GPU configurations. Our results identify distinct scaling trends: (1) architectural choices, such as action tokenization and model backbone size, strongly influence throughput and memory footprint; (2) power-constrained edge devices exhibit non-linear performance degradation, with some configurations matching or exceeding older datacenter GPUs; and (3) high-throughput variants can be achieved without significant accuracy loss. These findings provide actionable insights when selecting and optimizing VLAs across a range of deployment constraints. Our work challenges current assumptions about the superiority of datacenter hardware for robotic inference. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型已成为机器人控制的强大通用策略，但它们在不同模型架构和硬件平台上的性能扩展及其相关的功耗预算仍 poorly understood。本工作评估了五种代表性VLA模型——涵盖最先进的基线和两种新提出的架构——针对边缘和数据中心GPU平台。使用LIBERO基准，我们在不同的边缘功耗约束和高性能数据中心GPU配置下，测量准确率和系统级指标，包括延迟、吞吐量和峰值内存使用量。我们的结果揭示了不同的扩展趋势：（1）架构选择，如动作标记化和模型骨干网络规模，强烈影响吞吐量和内存占用；（2）功耗受限的边缘设备表现出非线性的性能退化，某些配置可匹敌甚至超越较老的数据中心GPU；（3）高吞吐量变体可以在不显著损失准确性的前提下实现。这些发现为在各种部署约束下选择和优化VLA模型提供了可操作的见解。本工作挑战了数据中心硬件在机器人推断方面具有优越性的现有假设。 

---
# Beyond Frame-wise Tracking: A Trajectory-based Paradigm for Efficient Point Cloud Tracking 

**Title (ZH)**: 超越框架级跟踪：一种高效的点云轨迹导向跟踪范式 

**Authors**: BaiChen Fan, Sifan Zhou, Jian Li, Shibo Zhao, Muqing Cao, Qin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11453)  

**Abstract**: LiDAR-based 3D single object tracking (3D SOT) is a critical task in robotics and autonomous systems. Existing methods typically follow frame-wise motion estimation or a sequence-based paradigm. However, the two-frame methods are efficient but lack long-term temporal context, making them vulnerable in sparse or occluded scenes, while sequence-based methods that process multiple point clouds gain robustness at a significant computational cost. To resolve this dilemma, we propose a novel trajectory-based paradigm and its instantiation, TrajTrack. TrajTrack is a lightweight framework that enhances a base two-frame tracker by implicitly learning motion continuity from historical bounding box trajectories alone-without requiring additional, costly point cloud inputs. It first generates a fast, explicit motion proposal and then uses an implicit motion modeling module to predict the future trajectory, which in turn refines and corrects the initial proposal. Extensive experiments on the large-scale NuScenes benchmark show that TrajTrack achieves new state-of-the-art performance, dramatically improving tracking precision by 4.48% over a strong baseline while running at 56 FPS. Besides, we also demonstrate the strong generalizability of TrajTrack across different base trackers. Video is available at this https URL. 

**Abstract (ZH)**: 基于LiDAR的3D单对象跟踪（3D SOT）是机器人技术与自主系统中的关键任务。现有方法通常遵循基于帧的运动估计或基于序列的范式。然而，两帧方法虽然高效，但在稀疏或被遮挡的场景中缺乏长期的时间上下文，使其容易出错，而基于序列的方法虽然具有鲁棒性，但在处理多个点云时需要巨大的计算成本。为解决这一困境，我们提出了一种新的轨迹导向范式及其实例化方法TrajTrack。TrajTrack是一种轻量级框架，通过仅从历史边界框轨迹中隐式学习运动连续性，增强了一个基础的两帧跟踪器，而无需额外的成本高昂的点云输入。它首先生成快速明确的运动提案，然后使用隐式运动建模模块预测未来轨迹，进而细化和修正初始提案。在大规模NuScenes基准测试上的大量实验表明，TrajTrack实现了新的最先进性能，与强基准相比，精度提高了4.48%，且运行速度达到56 FPS。此外，我们还展示了TrajTrack在不同基础跟踪器上的强泛化能力。视频见此链接：[视频链接]。 

---
# Mars Traversability Prediction: A Multi-modal Self-supervised Approach for Costmap Generation 

**Title (ZH)**: 火星通行性预测：用于代价地图生成的多模态自监督方法 

**Authors**: Zongwu Xie, Kaijie Yun, Yang Liu, Yiming Ji, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11082)  

**Abstract**: We present a robust multi-modal framework for predicting traversability costmaps for planetary rovers. Our model fuses camera and LiDAR data to produce a bird's-eye-view (BEV) terrain costmap, trained self-supervised using IMU-derived labels. Key updates include a DINOv3-based image encoder, FiLM-based sensor fusion, and an optimization loss combining Huber and smoothness terms. Experimental ablations (removing image color, occluding inputs, adding noise) show only minor changes in MAE/MSE (e.g. MAE increases from ~0.0775 to 0.0915 when LiDAR is sparsified), indicating that geometry dominates the learned cost and the model is highly robust. We attribute the small performance differences to the IMU labeling primarily reflecting terrain geometry rather than semantics and to limited data diversity. Unlike prior work claiming large gains, we emphasize our contributions: (1) a high-fidelity, reproducible simulation environment; (2) a self-supervised IMU-based labeling pipeline; and (3) a strong multi-modal BEV costmap prediction model. We discuss limitations and future work such as domain generalization and dataset expansion. 

**Abstract (ZH)**: 一种鲁棒的多模态框架，用于预测行星探测车的通行性成本图：基于DINOv3的图像编码器、FiLM机制的传感器融合以及结合Huber和光滑项的优化损失 

---
# Agent-based Simulation for Drone Charging in an Internet of Things Environment System 

**Title (ZH)**: 基于代理的物联网环境下的无人机充电仿真研究 

**Authors**: Leonardo Grando, José Roberto Emiliano Leite, Edson Luiz Ursini  

**Link**: [PDF](https://arxiv.org/pdf/2509.10867)  

**Abstract**: This paper presents an agent-based simulation model for coordinating battery recharging in drone swarms, focusing on applications in Internet of Things (IoT) and Industry 4.0 environments. The proposed model includes a detailed description of the simulation methodology, system architecture, and implementation. One practical use case is explored: Smart Farming, highlighting how autonomous coordination strategies can optimize battery usage and mission efficiency in large-scale drone deployments. This work uses a machine learning technique to analyze the agent-based simulation sensitivity analysis output results. 

**Abstract (ZH)**: 基于代理的无人机群电池充电协调仿真模型：以物联网和工业4.0应用为例 

---
# Point-Plane Projections for Accurate LiDAR Semantic Segmentation in Small Data Scenarios 

**Title (ZH)**: 点面投影用于小数据场景下的LiDAR语义分割 

**Authors**: Simone Mosco, Daniel Fusaro, Wanmeng Li, Emanuele Menegatti, Alberto Pretto  

**Link**: [PDF](https://arxiv.org/pdf/2509.10841)  

**Abstract**: LiDAR point cloud semantic segmentation is essential for interpreting 3D environments in applications such as autonomous driving and robotics. Recent methods achieve strong performance by exploiting different point cloud representations or incorporating data from other sensors, such as cameras or external datasets. However, these approaches often suffer from high computational complexity and require large amounts of training data, limiting their generalization in data-scarce scenarios. In this paper, we improve the performance of point-based methods by effectively learning features from 2D representations through point-plane projections, enabling the extraction of complementary information while relying solely on LiDAR data. Additionally, we introduce a geometry-aware technique for data augmentation that aligns with LiDAR sensor properties and mitigates class imbalance. We implemented and evaluated our method that applies point-plane projections onto multiple informative 2D representations of the point cloud. Experiments demonstrate that this approach leads to significant improvements in limited-data scenarios, while also achieving competitive results on two publicly available standard datasets, as SemanticKITTI and PandaSet. The code of our method is available at this https URL 

**Abstract (ZH)**: LiDAR点云语义分割对于在自动驾驶和机器人等领域解释3D环境至关重要。近期的方法通过利用不同的点云表示或结合其他传感器（如摄像头或外部数据集）的数据，实现了较强的性能。然而，这些方法通常面临高计算复杂度和需要大量训练数据的问题，限制了它们在数据稀缺场景下的泛化能力。本文通过有效学习点平面投影的2D表示特征，提高了基于点的方法的性能，从而仅依赖LiDAR数据即可提取互补信息。此外，我们还引入了一种几何感知的数据增强技术，该技术符合LiDAR传感器的特性并缓解了类别不平衡问题。我们在多个信息性的2D点云表示上应用点平面投影的方法进行了实施和评估。实验结果表明，该方法在数据稀缺场景中取得了显著的改进，并在SemanticKITTI和PandaSet等两个公开的标准数据集上达到了具有竞争力的结果。我们的方法代码可在此处访问：this https URL。 

---
# InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts 

**Title (ZH)**: InternScenes: 一个具有真实布局的大规模可模拟室内场景数据集 

**Authors**: Weipeng Zhong, Peizhou Cao, Yichen Jin, Li Luo, Wenzhe Cai, Jingli Lin, Hanqing Wang, Zhaoyang Lyu, Tai Wang, Bo Dai, Xudong Xu, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10813)  

**Abstract**: The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community. 

**Abstract (ZH)**: InternScenes：一种新型的大型可模拟室内场景数据集 

---
# Asynchronous Gathering of Opaque Robots with Mobility Faults 

**Title (ZH)**: 具有移动故障的不透明机器人异步聚集 

**Authors**: Subhajit Pramanick, Saswata Jana, Partha Sarathi Mandal, Gokarna Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.10711)  

**Abstract**: We consider the fundamental benchmarking problem of gathering in an $(N,f)$-fault system consisting of $N$ robots, of which at most $f$ might fail at any execution, under asynchrony. Two seminal results established impossibility of a solution in the oblivious robot (OBLOT) model in a $(2,0)$-fault system under semi-synchrony and in a $(3,1)$-Byzantine fault system under asynchrony. Recently, a breakthrough result circumvented the first impossibility result by giving a deterministic algorithm in a $(2,0)$-fault system under asynchrony in the luminous robot (LUMI) model using 2-colored lights. However, a breakthrough result established impossibility of gathering in a $(2,1)$-crash system in the LUMI model under semi-synchrony. In this paper, we consider a {\em mobility fault} model in which a robot crash only impacts it mobility but not the operation of the light.
We establish four results under asynchrony in LUMI with the mobility fault model. We show that it is impossible to solve gathering in a $(2,1)$-mobility fault system using 2-colored lights, and then give a solution using 3-colored lights, which is optimal w.r.t. the number of colors. We then consider an $(N,f)$-mobility fault system, $f<N$, both $N,f$ not known, and give two deterministic algorithms that exhibit a nice time-color trade-off: The first with time $O(N)$ using 7-colored lights and the second with time $O(\max\{\ell,f\})$ using 26-colored lights, where $\ell< N$ is the number of distinct convex layers of robot positions in the initial configuration. Interestingly, for $l, f = O(1)$, our result is optimal. Our algorithms for an $(N,f)$-mobility fault system are the first to be analysed time complexity, can withstand obstructed visibility (opaque robot model) and asynchronous scheduling. 

**Abstract (ZH)**: 在具有移动故障模型的异步系统中关于$(N,f)$故障机器人的聚集团体问题 

---
# Synergetic Empowerment: Wireless Communications Meets Embodied Intelligence 

**Title (ZH)**: 协同赋能：无线通信与 embodied 智能的融合 

**Authors**: Hongtao Liang, Yihe Diao, YuHang Wu, Fuhui Zhou, Qihui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10481)  

**Abstract**: Wireless communication is evolving into an agent era, where large-scale agents with inherent embodied intelligence are not just users but active participants. The perfect combination of wireless communication and embodied intelligence can achieve a synergetic empowerment and greatly facilitate the development of agent communication. An overview of this synergetic empowerment is presented, framing it as a co-evolutionary process that transforms wireless communication from a simple utility into the digital nervous system of a collective intelligence, while simultaneously elevating isolated agents into a unified superorganism with emergent capabilities far exceeding individual contributions. Moreover, we elaborate how embodied intelligence and wireless communication mutually benefit each other through the lens of the perception-cognition-execution (PCE) loop, revealing a fundamental duality where each PCE stage both challenges network capacity and creates unprecedented opportunities for system-wide optimization. Furthermore, critical open issues and future research directions are identified. 

**Abstract (ZH)**: 无线通信正演进为智能代理时代，大规模具备内在本体智能的代理不仅是用户，更是积极参与者。无线通信与本体智能的完美结合能实现协同赋能，极大地促进智能代理通信的发展。本文概述了这种协同赋能的过程，将其视为一种共生演化过程，将无线通信从简单的工具转变为集体智能的数字神经系统，同时将孤立的代理提升为具备超越个体贡献的新兴能力的统一超有机体。此外，本文从感知-认知-执行（PCE）循环的视角阐明了本体智能与无线通信相互受益的基本二元性，每个PCE阶段既对网络容量构成挑战，又为系统级优化创造前所未有的机遇。最后，本文指出了关键的开放问题和未来研究方向。 

---
