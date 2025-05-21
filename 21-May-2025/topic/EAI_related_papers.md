# Semantically-driven Deep Reinforcement Learning for Inspection Path Planning 

**Title (ZH)**: 语义驱动的深度强化学习在检查路径规划中的应用 

**Authors**: Grzegorz Malczyk, Mihir Kulkarni, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2505.14443)  

**Abstract**: This paper introduces a novel semantics-aware inspection planning policy derived through deep reinforcement learning. Reflecting the fact that within autonomous informative path planning missions in unknown environments, it is often only a sparse set of objects of interest that need to be inspected, the method contributes an end-to-end policy that simultaneously performs semantic object visual inspection combined with collision-free navigation. Assuming access only to the instantaneous depth map, the associated segmentation image, the ego-centric local occupancy, and the history of past positions in the robot's neighborhood, the method demonstrates robust generalizability and successful crossing of the sim2real gap. Beyond simulations and extensive comparison studies, the approach is verified in experimental evaluations onboard a flying robot deployed in novel environments with previously unseen semantics and overall geometric configurations. 

**Abstract (ZH)**: 本文介绍了一种通过深度强化学习得到的新型语义意识检测规划策略，该策略能够同时实现语义物体视觉检测与碰撞-free 导航，并具有强大的鲁棒性和成功的泛化能力，能够在未见语义和几何结构的新环境中验证该方法。 

---
# Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning 

**Title (ZH)**: 基于采样和主动探索的系统辨识在腿足机器人Sim2Real学习中的应用 

**Authors**: Nikhil Sobanbabu, Guanqi He, Tairan He, Yuxiang Yang, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14266)  

**Abstract**: Sim-to-real discrepancies hinder learning-based policies from achieving high-precision tasks in the real world. While Domain Randomization (DR) is commonly used to bridge this gap, it often relies on heuristics and can lead to overly conservative policies with degrading performance when not properly tuned. System Identification (Sys-ID) offers a targeted approach, but standard techniques rely on differentiable dynamics and/or direct torque measurement, assumptions that rarely hold for contact-rich legged systems. To this end, we present SPI-Active (Sampling-based Parameter Identification with Active Exploration), a two-stage framework that estimates physical parameters of legged robots to minimize the sim-to-real gap. SPI-Active robustly identifies key physical parameters through massive parallel sampling, minimizing state prediction errors between simulated and real-world trajectories. To further improve the informativeness of collected data, we introduce an active exploration strategy that maximizes the Fisher Information of the collected real-world trajectories via optimizing the input commands of an exploration policy. This targeted exploration leads to accurate identification and better generalization across diverse tasks. Experiments demonstrate that SPI-Active enables precise sim-to-real transfer of learned policies to the real world, outperforming baselines by 42-63% in various locomotion tasks. 

**Abstract (ZH)**: 基于采样的物理参数识别与主动探索（SPI-Active）：缩小仿真与现实差距的方法 

---
# Unconventional Hexacopters via Evolution and Learning: Performance Gains and New Insights 

**Title (ZH)**: 非常规六旋翼无人机通过进化与学习：性能提升与新的见解 

**Authors**: Jed Muff, Keiichi Ito, Elijah H. W. Ang, Karine Miras, A.E. Eiben  

**Link**: [PDF](https://arxiv.org/pdf/2505.14129)  

**Abstract**: Evolution and learning have historically been interrelated topics, and their interplay is attracting increased interest lately. The emerging new factor in this trend is morphological evolution, the evolution of physical forms within embodied AI systems such as robots. In this study, we investigate a system of hexacopter-type drones with evolvable morphologies and learnable controllers and make contributions to two fields. For aerial robotics, we demonstrate that the combination of evolution and learning can deliver non-conventional drones that significantly outperform the traditional hexacopter on several tasks that are more complex than previously considered in the literature. For the field of Evolutionary Computing, we introduce novel metrics and perform new analyses into the interaction of morphological evolution and learning, uncovering hitherto unidentified effects. Our analysis tools are domain-agnostic, making a methodological contribution towards building solid foundations for embodied AI systems that integrate evolution and learning. 

**Abstract (ZH)**: 进化和学习 historically 一直是相关的话题，它们的相互作用正日益受到关注。这一趋势中新兴的因素是形态进化，即在诸如机器人等具身AI系统中的物理形态进化。在本研究中，我们探讨了一种六旋翼无人机类型的具形态可进化性和可学习控制器的系统，并为两个领域做出了贡献。在空中机器人领域，我们证明了进化与学习的结合能够产生非传统无人机，在多项比以往文献中考虑的更为复杂的任务上显著超越传统的六旋翼无人机。在演化计算领域，我们引入了新的度量标准并进行了新的分析，揭示了先前未被发现的效果。我们的分析工具具有领域通用性，为构建结合进化与学习的具身AI系统奠定了方法论基础。 

---
# AutoBio: A Simulation and Benchmark for Robotic Automation in Digital Biology Laboratory 

**Title (ZH)**: AutoBio: 数字生物学实验室中机器人自动化模拟与基准测试 

**Authors**: Zhiqian Lan, Yuxuan Jiang, Ruiqi Wang, Xuanbing Xie, Rongkui Zhang, Yicheng Zhu, Peihang Li, Tianshuo Yang, Tianxing Chen, Haoyu Gao, Xiaokang Yang, Xuelong Li, Hongyuan Zhang, Yao Mu, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14030)  

**Abstract**: Vision-language-action (VLA) models have shown promise as generalist robotic policies by jointly leveraging visual, linguistic, and proprioceptive modalities to generate action trajectories. While recent benchmarks have advanced VLA research in domestic tasks, professional science-oriented domains remain underexplored. We introduce AutoBio, a simulation framework and benchmark designed to evaluate robotic automation in biology laboratory environments--an application domain that combines structured protocols with demanding precision and multimodal interaction. AutoBio extends existing simulation capabilities through a pipeline for digitizing real-world laboratory instruments, specialized physics plugins for mechanisms ubiquitous in laboratory workflows, and a rendering stack that support dynamic instrument interfaces and transparent materials through physically based rendering. Our benchmark comprises biologically grounded tasks spanning three difficulty levels, enabling standardized evaluation of language-guided robotic manipulation in experimental protocols. We provide infrastructure for demonstration generation and seamless integration with VLA models. Baseline evaluations with two SOTA VLA models reveal significant gaps in precision manipulation, visual reasoning, and instruction following in scientific workflows. By releasing AutoBio, we aim to catalyze research on generalist robotic systems for complex, high-precision, and multimodal professional environments. The simulator and benchmark are publicly available to facilitate reproducible research. 

**Abstract (ZH)**: 基于视觉-语言-动作的模型在生物学实验室环境中的机器人自动化评估框架AutoBio 

---
# Adaptive Visuo-Tactile Fusion with Predictive Force Attention for Dexterous Manipulation 

**Title (ZH)**: 基于预测力注意力的自适应视觉-触觉融合用于灵巧操作 

**Authors**: Jinzhou Li, Tianhao Wu, Jiyao Zhang, Zeyuan Chen, Haotian Jin, Mingdong Wu, Yujun Shen, Yaodong Yang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.13982)  

**Abstract**: Effectively utilizing multi-sensory data is important for robots to generalize across diverse tasks. However, the heterogeneous nature of these modalities makes fusion challenging. Existing methods propose strategies to obtain comprehensively fused features but often ignore the fact that each modality requires different levels of attention at different manipulation stages. To address this, we propose a force-guided attention fusion module that adaptively adjusts the weights of visual and tactile features without human labeling. We also introduce a self-supervised future force prediction auxiliary task to reinforce the tactile modality, improve data imbalance, and encourage proper adjustment. Our method achieves an average success rate of 93% across three fine-grained, contactrich tasks in real-world experiments. Further analysis shows that our policy appropriately adjusts attention to each modality at different manipulation stages. The videos can be viewed at this https URL. 

**Abstract (ZH)**: 有效利用多感官数据对于机器人在多样任务中泛化至关重要。然而，这些模态的异质性使得融合具有挑战性。现有方法提出了获取综合融合特征的策略，但往往忽略了每个模态在不同操作阶段需要不同水平关注的事实。为了解决这一问题，我们提出了一种力引导的关注融合模块，能够在无需人工标注的情况下自适应调整视觉和触觉特征的权重。我们还引入了一项自监督的未来力预测辅助任务，以强化触觉模态、改善数据不平衡并促进适当的调整。我们的方法在实际实验中的三种精细、接触丰富的任务中实现了平均93%的成功率。进一步的分析表明，我们的策略在不同操作阶段适当调整了对每个模态的关注。视频可观看此链接：这个 https URL。 

---
# Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning 

**Title (ZH)**: 时反演对称性在深度强化学习中高效机器人操作中的应用 

**Authors**: Yunpeng Jiang, Jianshu Hu, Paul Weng, Yutong Ban  

**Link**: [PDF](https://arxiv.org/pdf/2505.13925)  

**Abstract**: Symmetry is pervasive in robotics and has been widely exploited to improve sample efficiency in deep reinforcement learning (DRL). However, existing approaches primarily focus on spatial symmetries, such as reflection, rotation, and translation, while largely neglecting temporal symmetries. To address this gap, we explore time reversal symmetry, a form of temporal symmetry commonly found in robotics tasks such as door opening and closing. We propose Time Reversal symmetry enhanced Deep Reinforcement Learning (TR-DRL), a framework that combines trajectory reversal augmentation and time reversal guided reward shaping to efficiently solve temporally symmetric tasks. Our method generates reversed transitions from fully reversible transitions, identified by a proposed dynamics-consistent filter, to augment the training data. For partially reversible transitions, we apply reward shaping to guide learning, according to successful trajectories from the reversed task. Extensive experiments on the Robosuite and MetaWorld benchmarks demonstrate that TR-DRL is effective in both single-task and multi-task settings, achieving higher sample efficiency and stronger final performance compared to baseline methods. 

**Abstract (ZH)**: 时间逆运算增强的深度强化学习（时间逆运算DRL） 

---
# InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning 

**Title (ZH)**: InSpire：内置空间推理的视听觉模型 

**Authors**: Ji Zhang, Shihan Wu, Xu Luo, Hao Wu, Lianli Gao, Heng Tao Shen, Jingkuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.13888)  

**Abstract**: Leveraging pretrained Vision-Language Models (VLMs) to map language instruction and visual observations to raw low-level actions, Vision-Language-Action models (VLAs) hold great promise for achieving general-purpose robotic systems. Despite their advancements, existing VLAs tend to spuriously correlate task-irrelevant visual features with actions, limiting their generalization capacity beyond the training data. To tackle this challenge, we propose Intrinsic Spatial Reasoning (InSpire), a simple yet effective approach that mitigates the adverse effects of spurious correlations by boosting the spatial reasoning ability of VLAs. Specifically, InSpire redirects the VLA's attention to task-relevant factors by prepending the question "In which direction is the [object] relative to the robot?" to the language instruction and aligning the answer "right/left/up/down/front/back/grasped" and predicted actions with the ground-truth. Notably, InSpire can be used as a plugin to enhance existing autoregressive VLAs, requiring no extra training data or interaction with other large models. Extensive experimental results in both simulation and real-world environments demonstrate the effectiveness and flexibility of our approach. Our code, pretrained models and demos are publicly available at: this https URL. 

**Abstract (ZH)**: 利用预训练的视觉-语言模型（VLMs）将语言指令和视觉观察映射到原始低级动作，视觉-语言-动作模型（VLAs）在实现通用机器人系统方面具有巨大潜力。尽管取得了进展，现有的VLAs往往会错误地将与任务无关的视觉特征与动作相关联，限制了其超越训练数据的泛化能力。为应对这一挑战，我们提出了内在空间推理（InSpire），一种简单而有效的方法，通过增强VLAs的空间推理能力来减轻虚假相关性的负面影响。具体而言，InSpire通过在语言指令前添加问题“[物体]相对于机器人在哪个方向？”并将答案“右/左/上/下/前/后/被抓取”与预测动作对齐于 ground-truth，引导VLAs的注意力关注任务相关因素。值得注意的是，InSpire可以用作插件增强现有的自回归VLAs，无需额外的训练数据或与其他大型模型交互。我们在模拟和现实环境中的广泛实验结果证明了该方法的有效性和灵活性。我们的代码、预训练模型和演示均可在以下网址获取：这个 https URL。 

---
# Toward Real-World Cooperative and Competitive Soccer with Quadrupedal Robot Teams 

**Title (ZH)**: 面向 quadrupedal 机器人团队的现实世界中的协同与竞争足球研究 

**Authors**: Zhi Su, Yuman Gao, Emily Lukas, Yunfei Li, Jiaze Cai, Faris Tulbah, Fei Gao, Chao Yu, Zhongyu Li, Yi Wu, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2505.13834)  

**Abstract**: Achieving coordinated teamwork among legged robots requires both fine-grained locomotion control and long-horizon strategic decision-making. Robot soccer offers a compelling testbed for this challenge, combining dynamic, competitive, and multi-agent interactions. In this work, we present a hierarchical multi-agent reinforcement learning (MARL) framework that enables fully autonomous and decentralized quadruped robot soccer. First, a set of highly dynamic low-level skills is trained for legged locomotion and ball manipulation, such as walking, dribbling, and kicking. On top of these, a high-level strategic planning policy is trained with Multi-Agent Proximal Policy Optimization (MAPPO) via Fictitious Self-Play (FSP). This learning framework allows agents to adapt to diverse opponent strategies and gives rise to sophisticated team behaviors, including coordinated passing, interception, and dynamic role allocation. With an extensive ablation study, the proposed learning method shows significant advantages in the cooperative and competitive multi-agent soccer game. We deploy the learned policies to real quadruped robots relying solely on onboard proprioception and decentralized localization, with the resulting system supporting autonomous robot-robot and robot-human soccer matches on indoor and outdoor soccer courts. 

**Abstract (ZH)**: 实现腿足机器人之间的协调团队合作需要精细的运动控制和长期的战略决策。机器人足球为这一挑战提供了极具吸引力的实验平台，结合了动态、竞争性和多智能体交互。在本工作中，我们提出了一个分层多智能体强化学习（MARL）框架，以实现完全自主和分布式腿足机器人足球。首先，训练了一组高度动态的低层次技能，用于腿足运动和球操作，如步行、带球和踢球。在此基础上，通过虚构自我博弈（FSP）和多智能体近端策略优化（MAPPO）训练高层次的战略规划策略。该学习框架使智能体能够适应多种对手策略，产生复杂的团队行为，包括协调传球、拦截和动态角色分配。通过广泛的消融研究，所提出的学习方法在合作性和竞争性多智能体足球游戏中显示出了显著优势。我们仅依赖于机载本体感觉和分布式定位，将所学策略部署到真实的四腿机器人上，最终系统支持室内和室外足球场上的自主机器人对机器人和机器人对人类的足球比赛。 

---
# C*: A Coverage Path Planning Algorithm for Unknown Environments using Rapidly Covering Graphs 

**Title (ZH)**: C*: 一种基于快速覆盖图的未知环境覆盖路径规划算法 

**Authors**: Zongyuan Shen, James P. Wilson, Shalabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.13782)  

**Abstract**: The paper presents a novel sample-based algorithm, called C*, for real-time coverage path planning (CPP) of unknown environments. The C* algorithm is built upon the concept of Rapidly Covering Graph (RCGs). The RCG is constructed incrementally via progressive sampling during robot navigation, which eliminates the need for cellular decomposition of the search space. The RCG has a sparse-graph structure formed by efficient sampling and pruning techniques, which produces non-myopic waypoints of the coverage trajectory. While C* produces the desired back and forth coverage pattern, it adapts to the TSP-based locally optimal coverage of small uncovered regions, called coverage holes, that are surrounded by obstacles and covered regions. Thus, C* proactively detects and covers the coverage holes in situ, which reduces the coverage time by preventing the longer return trajectories from distant regions to cover such holes later. The algorithmic simplicity and low computational complexity of C* makes it easy to implement and suitable for real-time onboard applications. It is analytically proven that C* provides complete coverage of unknown environments. The performance of C* is validated by 1) extensive high-fidelity simulations and 2) real laboratory experiments using autonomous robots. A comparative evaluation with seven existing CPP methods demonstrate that C* yields significant performance improvements in terms of coverage time, number of turns, trajectory length and overlap ratio, while preventing the formation of coverage holes. Finally, C* is evaluated on two different applications of CPP using 1) energy-constrained robots and 2) multi-robot teams. 

**Abstract (ZH)**: 基于样本的实时未知环境覆盖路径规划算法C* 

---
# Dynamic Bipedal MPC with Foot-level Obstacle Avoidance and Adjustable Step Timing 

**Title (ZH)**: 基于足部级障碍避免和可调步态timing的动态双足MPC算法 

**Authors**: Tianze Wang, Christian Hubicki  

**Link**: [PDF](https://arxiv.org/pdf/2505.13715)  

**Abstract**: Collision-free planning is essential for bipedal robots operating within unstructured environments. This paper presents a real-time Model Predictive Control (MPC) framework that addresses both body and foot avoidance for dynamic bipedal robots. Our contribution is two-fold: we introduce (1) a novel formulation for adjusting step timing to facilitate faster body avoidance and (2) a novel 3D foot-avoidance formulation that implicitly selects swing trajectories and footholds that either steps over or navigate around obstacles with awareness of Center of Mass (COM) dynamics. We achieve body avoidance by applying a half-space relaxation of the safe region but introduce a switching heuristic based on tracking error to detect a need to change foot-timing schedules. To enable foot avoidance and viable landing footholds on all sides of foot-level obstacles, we decompose the non-convex safe region on the ground into several convex polygons and use Mixed-Integer Quadratic Programming to determine the optimal candidate. We found that introducing a soft minimum-travel-distance constraint is effective in preventing the MPC from being trapped in local minima that can stall half-space relaxation methods behind obstacles. We demonstrated the proposed algorithms on multibody simulations on the bipedal robot platforms, Cassie and Digit, as well as hardware experiments on Digit. 

**Abstract (ZH)**: 无碰撞规划对于在未结构化环境中操作的双足机器人至关重要。本文提出了一种实时模型预测控制（MPC）框架，解决动态双足机器人的身体和足部避障问题。我们的贡献主要有两点：一是提出了一种新的步态时间调整公式，以促进更快的身体避障；二是提出了一种新颖的三维足部避障公式，既隐式选择了跨越或绕过障碍的摆动轨迹和 foothold，又考虑了动量中心（COM）动力学。通过应用安全区域的半空间放松来实现身体避障，但引入基于跟踪误差的切换启发式方法来检测需要改变足部时间表的情况。为了在脚高障碍物的所有侧面上实现足部避障和可行的着陆 foothold，我们将地面上的非凸安全区域分解成若干个凸多边形，并使用混合整数二次规划来确定最优候选方案。我们发现引入一个柔软的最小旅行距离约束可以有效地防止MPC陷入局部极小值，这些极小值可能会阻碍半空间放松方法在障碍物后停滞。我们在双足机器人平台Cassie和Digit上的多体仿真以及Digit的硬件实验上展示了所提出的算法。 

---
# Adaptive Diffusion Constrained Sampling for Bimanual Robot Manipulation 

**Title (ZH)**: 双手机器人操作的自适应扩散约束采样 

**Authors**: Haolei Tong, Yuezhe Zhang, Sophie Lueth, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.13667)  

**Abstract**: Coordinated multi-arm manipulation requires satisfying multiple simultaneous geometric constraints across high-dimensional configuration spaces, which poses a significant challenge for traditional planning and control methods. In this work, we propose Adaptive Diffusion Constrained Sampling (ADCS), a generative framework that flexibly integrates both equality (e.g., relative and absolute pose constraints) and structured inequality constraints (e.g., proximity to object surfaces) into an energy-based diffusion model. Equality constraints are modeled using dedicated energy networks trained on pose differences in Lie algebra space, while inequality constraints are represented via Signed Distance Functions (SDFs) and encoded into learned constraint embeddings, allowing the model to reason about complex spatial regions. A key innovation of our method is a Transformer-based architecture that learns to weight constraint-specific energy functions at inference time, enabling flexible and context-aware constraint integration. Moreover, we adopt a two-phase sampling strategy that improves precision and sample diversity by combining Langevin dynamics with resampling and density-aware re-weighting. Experimental results on dual-arm manipulation tasks show that ADCS significantly improves sample diversity and generalization across settings demanding precise coordination and adaptive constraint handling. 

**Abstract (ZH)**: 协调多臂操作需要同时满足高维配置空间中的多个几何约束，这对传统的规划和控制方法构成了重大挑战。本文提出了一种自适应扩散约束采样（ADCS）生成框架，该框架灵活地将等式约束（如相对和绝对姿态约束）和结构化不等式约束（如物体表面的接近性）整合到基于能量的扩散模型中。等式约束通过训练在李代数空间中姿态差异的专用能量网络进行建模，不等式约束通过符号距离函数（SDF）表示并编码到学习的约束嵌入中，从而使模型能够推理复杂的空间区域。我们方法的一项关键创新是一种基于Transformer的架构，该架构在推理时学习不同约束特定的能量函数的加权方法，从而实现灵活且上下文相关的约束整合。此外，我们采用两阶段采样策略，通过结合拉格朗日动力学、重采样和密度感知重新加权来提高精度和样本多样性。实验结果表明，ADCS在需要精确协调和自适应约束处理的双臂操作任务中显著提高了样本多样性和适用性。 

---
# TD-GRPC: Temporal Difference Learning with Group Relative Policy Constraint for Humanoid Locomotion 

**Title (ZH)**: TD-GRPC：组相对策略约束下的时差学习方法用于 humanoid 运动控制 

**Authors**: Khang Nguyen, Khai Nguyen, An T. Le, Jan Peters, Manfred Huber, Ngo Anh Vien, Minh Nhat Vu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13549)  

**Abstract**: Robot learning in high-dimensional control settings, such as humanoid locomotion, presents persistent challenges for reinforcement learning (RL) algorithms due to unstable dynamics, complex contact interactions, and sensitivity to distributional shifts during training. Model-based methods, \textit{e.g.}, Temporal-Difference Model Predictive Control (TD-MPC), have demonstrated promising results by combining short-horizon planning with value-based learning, enabling efficient solutions for basic locomotion tasks. However, these approaches remain ineffective in addressing policy mismatch and instability introduced by off-policy updates. Thus, in this work, we introduce Temporal-Difference Group Relative Policy Constraint (TD-GRPC), an extension of the TD-MPC framework that unifies Group Relative Policy Optimization (GRPO) with explicit Policy Constraints (PC). TD-GRPC applies a trust-region constraint in the latent policy space to maintain consistency between the planning priors and learned rollouts, while leveraging group-relative ranking to assess and preserve the physical feasibility of candidate trajectories. Unlike prior methods, TD-GRPC achieves robust motions without modifying the underlying planner, enabling flexible planning and policy learning. We validate our method across a locomotion task suite ranging from basic walking to highly dynamic movements on the 26-DoF Unitree H1-2 humanoid robot. Through simulation results, TD-GRPC demonstrates its improvements in stability and policy robustness with sampling efficiency while training for complex humanoid control tasks. 

**Abstract (ZH)**: 高维度控制设置下的机器人学习： Tempo-difference Group Relative Policy Constraint (TD-GRPC) 在类人运动控制中的应用 

---
# Distributional Soft Actor-Critic with Harmonic Gradient for Safe and Efficient Autonomous Driving in Multi-lane Scenarios 

**Title (ZH)**: 分布软演员-评论家与谐波梯度在多车道场景中安全高效自主驾驶 

**Authors**: Feihong Zhang, Guojian Zhan, Bin Shuai, Tianyi Zhang, Jingliang Duan, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13532)  

**Abstract**: Reinforcement learning (RL), known for its self-evolution capability, offers a promising approach to training high-level autonomous driving systems. However, handling constraints remains a significant challenge for existing RL algorithms, particularly in real-world applications. In this paper, we propose a new safety-oriented training technique called harmonic policy iteration (HPI). At each RL iteration, it first calculates two policy gradients associated with efficient driving and safety constraints, respectively. Then, a harmonic gradient is derived for policy updating, minimizing conflicts between the two gradients and consequently enabling a more balanced and stable training process. Furthermore, we adopt the state-of-the-art DSAC algorithm as the backbone and integrate it with our HPI to develop a new safe RL algorithm, DSAC-H. Extensive simulations in multi-lane scenarios demonstrate that DSAC-H achieves efficient driving performance with near-zero safety constraint violations. 

**Abstract (ZH)**: 基于谐波策略迭代的强化学习安全训练技术研究 

---
# Towards Embodied Cognition in Robots via Spatially Grounded Synthetic Worlds 

**Title (ZH)**: 通过空间化接地合成世界实现机器人本体认知的研究 

**Authors**: Joel Currie, Gioele Migno, Enrico Piacenti, Maria Elena Giannaccini, Patric Bach, Davide De Tommaso, Agnieszka Wykowska  

**Link**: [PDF](https://arxiv.org/pdf/2505.14366)  

**Abstract**: We present a conceptual framework for training Vision-Language Models (VLMs) to perform Visual Perspective Taking (VPT), a core capability for embodied cognition essential for Human-Robot Interaction (HRI). As a first step toward this goal, we introduce a synthetic dataset, generated in NVIDIA Omniverse, that enables supervised learning for spatial reasoning tasks. Each instance includes an RGB image, a natural language description, and a ground-truth 4X4 transformation matrix representing object pose. We focus on inferring Z-axis distance as a foundational skill, with future extensions targeting full 6 Degrees Of Freedom (DOFs) reasoning. The dataset is publicly available to support further research. This work serves as a foundational step toward embodied AI systems capable of spatial understanding in interactive human-robot scenarios. 

**Abstract (ZH)**: 我们提出了一种概念框架，用于训练视觉-语言模型（VLMs）执行视觉观点转换（VPT），这是实现人类-机器人交互（HRI）的核心能力之一。为了这一目标的第一步，我们介绍了在NVIDIA Omniverse中生成的合成数据集，以支持空间推理任务的监督学习。每个实例包含一个RGB图像、自然语言描述以及表示物体姿态的地面 truth 4x4变换矩阵。我们专注于推断Z轴距离作为一项基础技能，未来扩展将针对完整的六自由度（6DOFs）推理。该数据集已公开，以支持进一步的研究。本工作为能够在交互式人机场景中实现空间理解的体态人工智能系统奠定了基础。 

---
# Debating for Better Reasoning: An Unsupervised Multimodal Approach 

**Title (ZH)**: 为更好地进行推理而辩论：一种无监督的多模态方法 

**Authors**: Ashutosh Adhikari, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2505.14627)  

**Abstract**: As Large Language Models (LLMs) gain expertise across diverse domains and modalities, scalable oversight becomes increasingly challenging, particularly when their capabilities may surpass human evaluators. Debate has emerged as a promising mechanism for enabling such oversight. In this work, we extend the debate paradigm to a multimodal setting, exploring its potential for weaker models to supervise and enhance the performance of stronger models. We focus on visual question answering (VQA), where two "sighted" expert vision-language models debate an answer, while a "blind" (text-only) judge adjudicates based solely on the quality of the arguments. In our framework, the experts defend only answers aligned with their beliefs, thereby obviating the need for explicit role-playing and concentrating the debate on instances of expert disagreement. Experiments on several multimodal tasks demonstrate that the debate framework consistently outperforms individual expert models. Moreover, judgments from weaker LLMs can help instill reasoning capabilities in vision-language models through finetuning. 

**Abstract (ZH)**: 大型语言模型在多种领域和模态中获得专业知识后，可扩展的监督变得越来越具有挑战性，尤其是当它们的能力可能超过人类评估者时。辩论作为一种机制在使这种监督成为可能方面展现出了前景。在本文中，我们将辩论 paradig姆扩展到多模态设置，探讨其在较弱模型监督并提升较强模型性能方面的潜力。我们专注于视觉问答（VQA），其中两个“有视力”的专家视觉-语言模型辩论答案，而一个“盲人”（仅文本）的裁判基于论点的质量做出裁决。在我们的框架中，专家们仅捍卫与其信念一致的答案，从而省去了明确的角色扮演的需求，并将辩论集中在专家存有分歧的实例上。在多项多模态任务上的实验表明，辩论框架始终优于单独的专家模型。此外，较弱的大规模语言模型的判断可以通过微调帮助视觉-语言模型获得推理能力。 

---
# Agent Context Protocols Enhance Collective Inference 

**Title (ZH)**: Agent Context Protocols 提高集体推理能力 

**Authors**: Devansh Bhardwaj, Arjun Beniwal, Shreyas Chaudhari, Ashwin Kalyan, Tanmay Rajpurohit, Karthik R. Narasimhan, Ameet Deshpande, Vishvak Murahari  

**Link**: [PDF](https://arxiv.org/pdf/2505.14569)  

**Abstract**: AI agents have become increasingly adept at complex tasks such as coding, reasoning, and multimodal understanding. However, building generalist systems requires moving beyond individual agents to collective inference -- a paradigm where multi-agent systems with diverse, task-specialized agents complement one another through structured communication and collaboration. Today, coordination is usually handled with imprecise, ad-hoc natural language, which limits complex interaction and hinders interoperability with domain-specific agents. We introduce Agent context protocols (ACPs): a domain- and agent-agnostic family of structured protocols for agent-agent communication, coordination, and error handling. ACPs combine (i) persistent execution blueprints -- explicit dependency graphs that store intermediate agent outputs -- with (ii) standardized message schemas, enabling robust and fault-tolerant multi-agent collective inference. ACP-powered generalist systems reach state-of-the-art performance: 28.3 % accuracy on AssistantBench for long-horizon web assistance and best-in-class multimodal technical reports, outperforming commercial AI systems in human evaluation. ACPs are highly modular and extensible, allowing practitioners to build top-tier generalist agents quickly. 

**Abstract (ZH)**: 基于代理上下文协议的通用智能系统：从个体代理到协作推理的转变 

---
# Causal Cartographer: From Mapping to Reasoning Over Counterfactual Worlds 

**Title (ZH)**: 因果地图师：从映射到因果联想的世界推理 

**Authors**: Gaël Gendron, Jože M. Rožanec, Michael Witbrock, Gillian Dobbie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14396)  

**Abstract**: Causal world models are systems that can answer counterfactual questions about an environment of interest, i.e. predict how it would have evolved if an arbitrary subset of events had been realized differently. It requires understanding the underlying causes behind chains of events and conducting causal inference for arbitrary unseen distributions. So far, this task eludes foundation models, notably large language models (LLMs), which do not have demonstrated causal reasoning capabilities beyond the memorization of existing causal relationships. Furthermore, evaluating counterfactuals in real-world applications is challenging since only the factual world is observed, limiting evaluation to synthetic datasets. We address these problems by explicitly extracting and modeling causal relationships and propose the Causal Cartographer framework. First, we introduce a graph retrieval-augmented generation agent tasked to retrieve causal relationships from data. This approach allows us to construct a large network of real-world causal relationships that can serve as a repository of causal knowledge and build real-world counterfactuals. In addition, we create a counterfactual reasoning agent constrained by causal relationships to perform reliable step-by-step causal inference. We show that our approach can extract causal knowledge and improve the robustness of LLMs for causal reasoning tasks while reducing inference costs and spurious correlations. 

**Abstract (ZH)**: 因果世界模型是能够回答所关注环境的反事实问题的系统，即预测若某事件子集以不同的方式实现，该环境会如何演变。这需要理解事件链背后的因果关系，并进行任意未见分布的因果推断。迄今为止，这一任务难以被基础模型，尤其是大型语言模型（LLMs），解决，因为它们只能记忆现有的因果关系而缺乏推导新的因果关系的能力。此外，在现实世界应用中评估反事实是具有挑战性的，因为只能观察到事实世界，这限制了评价方法只能局限于合成数据集。我们通过明确提取和建模因果关系来解决这些问题，并提出因果制图框架。首先，我们引入了一个图检索增强的生成代理，任务是从数据中检索因果关系。这种方法允许我们构建广泛的现实世界因果关系网络，作为因果知识库，并构建现实世界的反事实。此外，我们创建了一个受限于因果关系的反事实推理代理，以进行可靠的逐步因果推断。我们展示了我们的方法可以提取因果知识，并提高大型语言模型在因果推理任务中的鲁棒性，同时减少推理成本和虚假相关性。 

---
# Toward Embodied AGI: A Review of Embodied AI and the Road Ahead 

**Title (ZH)**: 向具身AGI迈进：具身AI综述与未来之路 

**Authors**: Yequan Wang, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14235)  

**Abstract**: Artificial General Intelligence (AGI) is often envisioned as inherently embodied. With recent advances in robotics and foundational AI models, we stand at the threshold of a new era-one marked by increasingly generalized embodied AI systems. This paper contributes to the discourse by introducing a systematic taxonomy of Embodied AGI spanning five levels (L1-L5). We review existing research and challenges at the foundational stages (L1-L2) and outline the key components required to achieve higher-level capabilities (L3-L5). Building on these insights and existing technologies, we propose a conceptual framework for an L3+ robotic brain, offering both a technical outlook and a foundation for future exploration. 

**Abstract (ZH)**: 人工通用智能（AGI）常常被视为固有的具身化。随着机器人技术和基础AI模型的最新进展，我们站在了一个新时代的门槛上——这个时代以日益通用的具身AI系统为标志。本文通过引入涵盖五个层级（L1-L5）的具身AGI系统系统分类法，为相关讨论做出了贡献。我们回顾了基础阶段（L1-L2）的研究和挑战，并概述了实现更高层级能力（L3-L5）所需的关键组件。基于这些洞见和现有技术，我们提出了一种L3+级机器人脑的概念框架，提供了一种技术前景并为未来的探索奠定了基础。 

---
# Embedded Mean Field Reinforcement Learning for Perimeter-defense Game 

**Title (ZH)**: 嵌入式均场强化学习在周界防御博弈中的应用 

**Authors**: Li Wang, Xin Yu, Xuxin Lv, Gangzheng Ai, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14209)  

**Abstract**: With the rapid advancement of unmanned aerial vehicles (UAVs) and missile technologies, perimeter-defense game between attackers and defenders for the protection of critical regions have become increasingly complex and strategically significant across a wide range of domains. However, existing studies predominantly focus on small-scale, simplified two-dimensional scenarios, often overlooking realistic environmental perturbations, motion dynamics, and inherent heterogeneity--factors that pose substantial challenges to real-world applicability. To bridge this gap, we investigate large-scale heterogeneous perimeter-defense game in a three-dimensional setting, incorporating realistic elements such as motion dynamics and wind fields. We derive the Nash equilibrium strategies for both attackers and defenders, characterize the victory regions, and validate our theoretical findings through extensive simulations. To tackle large-scale heterogeneous control challenges in defense strategies, we propose an Embedded Mean-Field Actor-Critic (EMFAC) framework. EMFAC leverages representation learning to enable high-level action aggregation in a mean-field manner, supporting scalable coordination among defenders. Furthermore, we introduce a lightweight agent-level attention mechanism based on reward representation, which selectively filters observations and mean-field information to enhance decision-making efficiency and accelerate convergence in large-scale tasks. Extensive simulations across varying scales demonstrate the effectiveness and adaptability of EMFAC, which outperforms established baselines in both convergence speed and overall performance. To further validate practicality, we test EMFAC in small-scale real-world experiments and conduct detailed analyses, offering deeper insights into the framework's effectiveness in complex scenarios. 

**Abstract (ZH)**: 无人机与导弹技术迅速发展背景下，三维环境中的大面积异质性 perimeter-防御博弈分析及嵌入式平均场演员-评论家框架研究 

---
# Solving Normalized Cut Problem with Constrained Action Space 

**Title (ZH)**: 在受限动作空间中求解归一化切分问题 

**Authors**: Qize Jiang, Linsey Pang, Alice Gatti, Mahima Aggarwa, Giovanna Vantin, Xiaosong Ma, Weiwei Sun, Sanjay Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2505.13986)  

**Abstract**: Reinforcement Learning (RL) has emerged as an important paradigm to solve combinatorial optimization problems primarily due to its ability to learn heuristics that can generalize across problem instances. However, integrating external knowledge that will steer combinatorial optimization problem solutions towards domain appropriate outcomes remains an extremely challenging task. In this paper, we propose the first RL solution that uses constrained action spaces to guide the normalized cut problem towards pre-defined template instances. Using transportation networks as an example domain, we create a Wedge and Ring Transformer that results in graph partitions that are shaped in form of Wedges and Rings and which are likely to be closer to natural optimal partitions. However, our approach is general as it is based on principles that can be generalized to other domains. 

**Abstract (ZH)**: 强化学习（RL）由于其在学习能够跨问题实例泛化的启发式方法方面的能力，已成为解决组合优化问题的重要范式。然而，将外部知识整合到组合优化问题解决方案中以引导其实现领域适当的成果仍然是一项极其具有挑战性的工作。在本文中，我们提出了第一个使用约束动作空间来引导最小割问题向预定义模板实例方向的RL解决方案。以运输网络为例，我们创建了一个楔形和环形转换器，生成的图分区呈现出楔形和环形的形状，并且更接近于自然最优分区。然而，我们的方法是通用的，因为它基于可以泛化到其他领域的原则。 

---
# Efficient Agent Training for Computer Use 

**Title (ZH)**: 计算机使用中的高效代理训练 

**Authors**: Yanheng He, Jiahe Jin, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13909)  

**Abstract**: Scaling up high-quality trajectory data has long been a critical bottleneck for developing human-like computer use agents. We introduce PC Agent-E, an efficient agent training framework that significantly reduces reliance on large-scale human demonstrations. Starting with just 312 human-annotated computer use trajectories, we further improved data quality by synthesizing diverse action decisions with Claude 3.7 Sonnet. Trained on these enriched trajectories, our PC Agent-E model achieved a remarkable 141% relative improvement, surpassing the strong Claude 3.7 Sonnet with extended thinking on WindowsAgentArena-V2, an improved benchmark we also released. Furthermore, PC Agent-E demonstrates strong generalizability to different operating systems on OSWorld. Our findings suggest that strong computer use capabilities can be stimulated from a small amount of high-quality trajectory data. 

**Abstract (ZH)**: 提高高质量轨迹数据规模一直是开发类人类计算机使用代理的关键瓶颈。我们引入了PC Agent-E，这是一种高效的代理训练框架，显著减少了对大规模人工演示的依赖。从仅有312个人标注的计算机使用轨迹出发，我们进一步通过Claude 3.7 Sonnet合成了多样化的动作决策以提高数据质量。基于这些丰富化的轨迹数据，我们的PC Agent-E模型取得了令人瞩目的141%的相对改进，超过了WindowsAgentArena-V2增强思考基准上的Claude 3.7 Sonnet，该基准我们亦已发布。此外，PC Agent-E在OSWorld上展示了很强的跨操作系统的一般性。我们的研究结果表明，强大的计算机使用能力可以从少量高质量的轨迹数据中激发出来。 

---
# A Challenge to Build Neuro-Symbolic Video Agents 

**Title (ZH)**: 构建神经符号视频代理的技术挑战 

**Authors**: Sahil Shah, Harsh Goel, Sai Shankar Narasimhan, Minkyu Choi, S P Sharan, Oguzhan Akcin, Sandeep Chinchali  

**Link**: [PDF](https://arxiv.org/pdf/2505.13851)  

**Abstract**: Modern video understanding systems excel at tasks such as scene classification, object detection, and short video retrieval. However, as video analysis becomes increasingly central to real-world applications, there is a growing need for proactive video agents for the systems that not only interpret video streams but also reason about events and take informed actions. A key obstacle in this direction is temporal reasoning: while deep learning models have made remarkable progress in recognizing patterns within individual frames or short clips, they struggle to understand the sequencing and dependencies of events over time, which is critical for action-driven decision-making. Addressing this limitation demands moving beyond conventional deep learning approaches. We posit that tackling this challenge requires a neuro-symbolic perspective, where video queries are decomposed into atomic events, structured into coherent sequences, and validated against temporal constraints. Such an approach can enhance interpretability, enable structured reasoning, and provide stronger guarantees on system behavior, all key properties for advancing trustworthy video agents. To this end, we present a grand challenge to the research community: developing the next generation of intelligent video agents that integrate three core capabilities: (1) autonomous video search and analysis, (2) seamless real-world interaction, and (3) advanced content generation. By addressing these pillars, we can transition from passive perception to intelligent video agents that reason, predict, and act, pushing the boundaries of video understanding. 

**Abstract (ZH)**: 现代视频理解系统在场景分类、物体检测和短视频检索等方面表现出色。然而，随着视频分析在实际应用中的重要性不断增加，人们越来越需要能够主动解读视频流、推理解事件并采取明智行动的视频代理。这一方向上的一大障碍是时间推理：虽然深度学习模型在识别单个帧或短片段内的模式方面取得了显著进展，但在理解事件随时间发生的顺序和依赖关系方面仍存在困难，这对手动驱动的决策至关重要。解决这一限制需要超越传统的深度学习方法。我们认为，解决这一挑战需要神经符号方法，即将视频查询分解为原子事件，结构化为一致的序列，并满足时间约束。这种方法可以增强可解释性，支持结构化推理，并提供更强的系统行为保证，这些都是推进可信视频代理所必需的关键特征。为此，我们向研究界提出一个宏伟挑战：开发集成了三大核心能力的新一代智能视频代理：（1）自主视频搜索和分析，（2）无缝现实世界交互，（3）高级内容生成。通过解决这些支柱，我们可以从被动感知过渡到能够推理、预测和行动的智能视频代理，从而推动视频理解的边界。 

---
# Multimodal RAG-driven Anomaly Detection and Classification in Laser Powder Bed Fusion using Large Language Models 

**Title (ZH)**: 基于大型语言模型的多模态RAG驱动激光 Powder 床融合中的异常检测与分类 

**Authors**: Kiarash Naghavi Khanghah, Zhiling Chen, Lela Romeo, Qian Yang, Rajiv Malhotra, Farhad Imani, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13828)  

**Abstract**: Additive manufacturing enables the fabrication of complex designs while minimizing waste, but faces challenges related to defects and process anomalies. This study presents a novel multimodal Retrieval-Augmented Generation-based framework that automates anomaly detection across various Additive Manufacturing processes leveraging retrieved information from literature, including images and descriptive text, rather than training datasets. This framework integrates text and image retrieval from scientific literature and multimodal generation models to perform zero-shot anomaly identification, classification, and explanation generation in a Laser Powder Bed Fusion setting. The proposed framework is evaluated on four L-PBF manufacturing datasets from Oak Ridge National Laboratory, featuring various printer makes, models, and materials. This evaluation demonstrates the framework's adaptability and generalizability across diverse images without requiring additional training. Comparative analysis using Qwen2-VL-2B and GPT-4o-mini as MLLM within the proposed framework highlights that GPT-4o-mini outperforms Qwen2-VL-2B and proportional random baseline in manufacturing anomalies classification. Additionally, the evaluation of the RAG system confirms that incorporating retrieval mechanisms improves average accuracy by 12% by reducing the risk of hallucination and providing additional information. The proposed framework can be continuously updated by integrating emerging research, allowing seamless adaptation to the evolving landscape of AM technologies. This scalable, automated, and zero-shot-capable framework streamlines AM anomaly analysis, enhancing efficiency and accuracy. 

**Abstract (ZH)**: 基于检索增强生成的多模态框架在激光粉床融合增材制造过程中实现零样本异常检测与分类 

---
# Building spatial world models from sparse transitional episodic memories 

**Title (ZH)**: 基于稀疏过渡性 episodic 记忆构建空间世界模型 

**Authors**: Zizhan He, Maxime Daigle, Pouya Bashivan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13696)  

**Abstract**: Many animals possess a remarkable capacity to rapidly construct flexible mental models of their environments. These world models are crucial for ethologically relevant behaviors such as navigation, exploration, and planning. The ability to form episodic memories and make inferences based on these sparse experiences is believed to underpin the efficiency and adaptability of these models in the brain. Here, we ask: Can a neural network learn to construct a spatial model of its surroundings from sparse and disjoint episodic memories? We formulate the problem in a simulated world and propose a novel framework, the Episodic Spatial World Model (ESWM), as a potential answer. We show that ESWM is highly sample-efficient, requiring minimal observations to construct a robust representation of the environment. It is also inherently adaptive, allowing for rapid updates when the environment changes. In addition, we demonstrate that ESWM readily enables near-optimal strategies for exploring novel environments and navigating between arbitrary points, all without the need for additional training. 

**Abstract (ZH)**: 一种从稀疏离散的 episodic 记忆构建空间模型的神经网络框架：Episodic 空间世界模型（ESWM）的研究 

---
# AgentSGEN: Multi-Agent LLM in the Loop for Semantic Collaboration and GENeration of Synthetic Data 

**Title (ZH)**: AgentSGEN：多智能体LLM参与的语义协作与合成数据生成 

**Authors**: Vu Dinh Xuan, Hao Vo, David Murphy, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13466)  

**Abstract**: The scarcity of data depicting dangerous situations presents a major obstacle to training AI systems for safety-critical applications, such as construction safety, where ethical and logistical barriers hinder real-world data collection. This creates an urgent need for an end-to-end framework to generate synthetic data that can bridge this gap. While existing methods can produce synthetic scenes, they often lack the semantic depth required for scene simulations, limiting their effectiveness. To address this, we propose a novel multi-agent framework that employs an iterative, in-the-loop collaboration between two agents: an Evaluator Agent, acting as an LLM-based judge to enforce semantic consistency and safety-specific constraints, and an Editor Agent, which generates and refines scenes based on this guidance. Powered by LLM's capabilities to reasoning and common-sense knowledge, this collaborative design produces synthetic images tailored to safety-critical scenarios. Our experiments suggest this design can generate useful scenes based on realistic specifications that address the shortcomings of prior approaches, balancing safety requirements with visual semantics. This iterative process holds promise for delivering robust, aesthetically sound simulations, offering a potential solution to the data scarcity challenge in multimedia safety applications. 

**Abstract (ZH)**: 稀缺的数据描述危险情境是训练应用于建筑安全等关键安全领域的AI系统的一大障碍，而伦理和后勤壁垒阻碍了真实世界数据的收集。这迫切需要一个端到端的框架来生成合成数据以弥补这一差距。尽管现有方法可以生成合成场景，但它们往往缺乏用于场景模拟所需的语义深度，限制了其效果。为了解决这一问题，我们提出了一种新的多智能体框架，该框架采用两个智能体在循环中的迭代协作：评估智能体作为基于LLM的裁判，负责维护语义一致性及特定安全约束；编辑智能体根据这一指导生成和优化场景。依托LLM在推理和常识方面的能力，这种协作设计可以生成针对关键安全场景定制的合成图像。我们的实验表明，该设计能够生成基于现实规范且能够弥补前人方法不足的有用场景，平衡了安全需求与视觉语义。这一迭代过程为多媒体安全应用中的数据稀缺挑战提供了可靠且美观的模拟解决方案。 

---
# Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values Prioritization with AIRiskDilemmas 

**Title (ZH)**: AI会为了挽救生病儿童而说谎吗？通过AIRiskDilemmas测试AI价值观优先级 

**Authors**: Yu Ying Chiu, Zhilin Wang, Sharan Maiya, Yejin Choi, Kyle Fish, Sydney Levine, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.14633)  

**Abstract**: Detecting AI risks becomes more challenging as stronger models emerge and find novel methods such as Alignment Faking to circumvent these detection attempts. Inspired by how risky behaviors in humans (i.e., illegal activities that may hurt others) are sometimes guided by strongly-held values, we believe that identifying values within AI models can be an early warning system for AI's risky behaviors. We create LitmusValues, an evaluation pipeline to reveal AI models' priorities on a range of AI value classes. Then, we collect AIRiskDilemmas, a diverse collection of dilemmas that pit values against one another in scenarios relevant to AI safety risks such as Power Seeking. By measuring an AI model's value prioritization using its aggregate choices, we obtain a self-consistent set of predicted value priorities that uncover potential risks. We show that values in LitmusValues (including seemingly innocuous ones like Care) can predict for both seen risky behaviors in AIRiskDilemmas and unseen risky behaviors in HarmBench. 

**Abstract (ZH)**: 随着更强的模型出现并采用如对齐欺骗等 novel 方法来规避这些检测尝试，检测 AI 风险变得更加具有挑战性。受人类危险行为（即可能伤害他人的违法活动）有时由坚定的价值观所引导的启发，我们认为识别 AI 模型中的价值可以作为 AI 危险行为的早期预警系统。我们创建了 LitmusValues 评估流水线，以揭示 AI 模型在一系列 AI 价值类别的优先级。然后，我们收集了 AIRiskDilemmas，这是一个包含各种困境的集合，这些困境在与 AI 安全风险相关的情景中将价值相互对立，例如权力追求。通过测量 AI 模型的价值优先级来评估其集体选择，我们获得了一致的价值优先级预测集，揭示潜在风险。我们展示 LitmusValues 中的价值（包括看似无害的价值如关怀）可以预测 AIRiskDilemmas 中已知的危险行为和 HarmBench 中未见的危险行为。 

---
# KIPPO: Koopman-Inspired Proximal Policy Optimization 

**Title (ZH)**: KIPPO: Koopman启发的近端策略优化 

**Authors**: Andrei Cozma, Landon Harris, Hairong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14566)  

**Abstract**: Reinforcement Learning (RL) has made significant strides in various domains, and policy gradient methods like Proximal Policy Optimization (PPO) have gained popularity due to their balance in performance, training stability, and computational efficiency. These methods directly optimize policies through gradient-based updates. However, developing effective control policies for environments with complex and non-linear dynamics remains a challenge. High variance in gradient estimates and non-convex optimization landscapes often lead to unstable learning trajectories. Koopman Operator Theory has emerged as a powerful framework for studying non-linear systems through an infinite-dimensional linear operator that acts on a higher-dimensional space of measurement functions. In contrast with their non-linear counterparts, linear systems are simpler, more predictable, and easier to analyze. In this paper, we present Koopman-Inspired Proximal Policy Optimization (KIPPO), which learns an approximately linear latent-space representation of the underlying system's dynamics while retaining essential features for effective policy learning. This is achieved through a Koopman-approximation auxiliary network that can be added to the baseline policy optimization algorithms without altering the architecture of the core policy or value function. Extensive experimental results demonstrate consistent improvements over the PPO baseline with 6-60% increased performance while reducing variability by up to 91% when evaluated on various continuous control tasks. 

**Abstract (ZH)**: 基于柯普曼理论的增强学习proximal策略优化（KIPPO） 

---
# Energy-Efficient Deep Reinforcement Learning with Spiking Transformers 

**Title (ZH)**: 能源高效的大规模强化学习变换器 

**Authors**: Mohammad Irfan Uddin, Nishad Tasnim, Md Omor Faruk, Zejian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14533)  

**Abstract**: Agent-based Transformers have been widely adopted in recent reinforcement learning advances due to their demonstrated ability to solve complex tasks. However, the high computational complexity of Transformers often results in significant energy consumption, limiting their deployment in real-world autonomous systems. Spiking neural networks (SNNs), with their biologically inspired structure, offer an energy-efficient alternative for machine learning. In this paper, a novel Spike-Transformer Reinforcement Learning (STRL) algorithm that combines the energy efficiency of SNNs with the powerful decision-making capabilities of reinforcement learning is developed. Specifically, an SNN using multi-step Leaky Integrate-and-Fire (LIF) neurons and attention mechanisms capable of processing spatio-temporal patterns over multiple time steps is designed. The architecture is further enhanced with state, action, and reward encodings to create a Transformer-like structure optimized for reinforcement learning tasks. Comprehensive numerical experiments conducted on state-of-the-art benchmarks demonstrate that the proposed SNN Transformer achieves significantly improved policy performance compared to conventional agent-based Transformers. With both enhanced energy efficiency and policy optimality, this work highlights a promising direction for deploying bio-inspired, low-cost machine learning models in complex real-world decision-making scenarios. 

**Abstract (ZH)**: 基于神经元 spike 的变压器在强化学习中的新型节能算法（Spike-Transformer Reinforcement Learning, STRL）研究 

---
# Visual Agentic Reinforcement Fine-Tuning 

**Title (ZH)**: 视觉代理强化微调 

**Authors**: Ziyu Liu, Yuhang Zang, Yushan Zou, Zijian Liang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14246)  

**Abstract**: A key trend in Large Reasoning Models (e.g., OpenAI's o3) is the native agentic ability to use external tools such as web browsers for searching and writing/executing code for image manipulation to think with images. In the open-source research community, while significant progress has been made in language-only agentic abilities such as function calling and tool integration, the development of multi-modal agentic capabilities that involve truly thinking with images, and their corresponding benchmarks, are still less explored. This work highlights the effectiveness of Visual Agentic Reinforcement Fine-Tuning (Visual-ARFT) for enabling flexible and adaptive reasoning abilities for Large Vision-Language Models (LVLMs). With Visual-ARFT, open-source LVLMs gain the ability to browse websites for real-time information updates and write code to manipulate and analyze input images through cropping, rotation, and other image processing techniques. We also present a Multi-modal Agentic Tool Bench (MAT) with two settings (MAT-Search and MAT-Coding) designed to evaluate LVLMs' agentic search and coding abilities. Our experimental results demonstrate that Visual-ARFT outperforms its baseline by +18.6% F1 / +13.0% EM on MAT-Coding and +10.3% F1 / +8.7% EM on MAT-Search, ultimately surpassing GPT-4o. Visual-ARFT also achieves +29.3 F1% / +25.9% EM gains on existing multi-hop QA benchmarks such as 2Wiki and HotpotQA, demonstrating strong generalization capabilities. Our findings suggest that Visual-ARFT offers a promising path toward building robust and generalizable multimodal agents. 

**Abstract (ZH)**: 大型推理模型中的一个关键趋势（例如OpenAI的o3）是原生代理能力，能够使用外部工具如网络浏览器进行搜索和编写/执行代码以通过图像操作进行思考。在开源研究社区中，虽然在仅语言代理能力方面（如函数调用和工具集成）取得了显著进展，但涉及到真正通过图像进行思考的多模态代理能力及其相应的基准测试仍较少探索。本研究强调了视觉代理强化微调（Visual-ARFT）的有效性，以增强大型视觉语言模型（LVLMs）的灵活和适应性推理能力。通过Visual-ARFT，开源LVLMs能够浏览网站以获取实时信息更新，并编写代码以通过裁剪、旋转和其他图像处理技术来操作和分析输入图像。我们还提出了一个多模态代理工具基准（MAT），其中包含两个设置（MAT-Search和MAT-Coding），用于评估LVLMs的代理搜索和编程能力。实验结果表明，Visual-ARFT在MAT-Coding上的F1得分提高了18.6% / EM提高了13.0%，在MAT-Search上的F1得分提高了10.3% / EM提高了8.7%，最终超越了GPT-4o。Visual-ARFT还在2Wiki和HotpotQA等现有多跳问答基准测试中实现了29.3%的F1增益 / 25.9%的EM增益，显示出强大的泛化能力。我们的研究发现表明，Visual-ARFT为构建鲁棒性和通用性兼备的多模态代理提供了有希望的道路。 

---
# Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models 

**Title (ZH)**: 面向医学VQA的有效强化学习微调方法研究 

**Authors**: Wenhui Zhu, Xuanzhao Dong, Xin Li, Peijie Qiu, Xiwen Chen, Abolfazl Razi, Aris Sotiras, Yi Su, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13973)  

**Abstract**: Recently, reinforcement learning (RL)-based tuning has shifted the trajectory of Multimodal Large Language Models (MLLMs), particularly following the introduction of Group Relative Policy Optimization (GRPO). However, directly applying it to medical tasks remains challenging for achieving clinically grounded model behavior. Motivated by the need to align model response with clinical expectations, we investigate four critical dimensions that affect the effectiveness of RL-based tuning in medical visual question answering (VQA): base model initialization strategy, the role of medical semantic alignment, the impact of length-based rewards on long-chain reasoning, and the influence of bias. We conduct extensive experiments to analyze these factors for medical MLLMs, providing new insights into how models are domain-specifically fine-tuned. Additionally, our results also demonstrate that GRPO-based RL tuning consistently outperforms standard supervised fine-tuning (SFT) in both accuracy and reasoning quality. 

**Abstract (ZH)**: 基于强化学习的调优最近改变了多模态大型语言模型(MLLMs)的轨迹，特别是在引入Group Relative Policy Optimization (GRPO)之后。然而，直接将其应用于医疗任务仍难以实现临床立足地的模型行为。为了使模型响应符合临床期望，我们探讨了影响基于强化学习(RL)调优在医疗视觉问答(VQA)中有效性的四个关键维度：基础模型初始化策略、医疗语义对齐的作用、基于长度的奖励对长链推理的影响以及偏见的影响。我们进行了广泛的实验来分析这些因素对医疗MLLMs的影响，为模型的领域特定微调提供了新的洞察。此外，我们的结果还表明，基于GRPO的RL调优在准确性和推理质量方面始终优于标准监督微调(SFT)。 

---
# Memory-Centric Embodied Question Answer 

**Title (ZH)**: 以记忆为中心的体映射问答 

**Authors**: Mingliang Zhai, Zhi Gao, Yuwei Wu, Yunde Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.13948)  

**Abstract**: Embodied Question Answering (EQA) requires agents to autonomously explore and understand the environment to answer context-dependent questions. Existing frameworks typically center around the planner, which guides the stopping module, memory module, and answering module for reasoning. In this paper, we propose a memory-centric EQA framework named MemoryEQA. Unlike planner-centric EQA models where the memory module cannot fully interact with other modules, MemoryEQA flexible feeds memory information into all modules, thereby enhancing efficiency and accuracy in handling complex tasks, such as those involving multiple targets across different regions. Specifically, we establish a multi-modal hierarchical memory mechanism, which is divided into global memory that stores language-enhanced scene maps, and local memory that retains historical observations and state information. When performing EQA tasks, the multi-modal large language model is leveraged to convert memory information into the required input formats for injection into different modules. To evaluate EQA models' memory capabilities, we constructed the MT-HM3D dataset based on HM3D, comprising 1,587 question-answer pairs involving multiple targets across various regions, which requires agents to maintain memory of exploration-acquired target information. Experimental results on HM-EQA, MT-HM3D, and OpenEQA demonstrate the effectiveness of our framework, where a 19.8% performance gain on MT-HM3D compared to baseline model further underscores memory capability's pivotal role in resolving complex tasks. 

**Abstract (ZH)**: 基于记忆的 embodied 问答 (MemoryEQA) 

---
# RLVR-World: Training World Models with Reinforcement Learning 

**Title (ZH)**: RLVR-World: 使用强化学习训练世界模型 

**Authors**: Jialong Wu, Shaofeng Yin, Ningya Feng, Mingsheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.13934)  

**Abstract**: World models predict state transitions in response to actions and are increasingly developed across diverse modalities. However, standard training objectives such as maximum likelihood estimation (MLE) often misalign with task-specific goals of world models, i.e., transition prediction metrics like accuracy or perceptual quality. In this paper, we present RLVR-World, a unified framework that leverages reinforcement learning with verifiable rewards (RLVR) to directly optimize world models for such metrics. Despite formulating world modeling as autoregressive prediction of tokenized sequences, RLVR-World evaluates metrics of decoded predictions as verifiable rewards. We demonstrate substantial performance gains on both language- and video-based world models across domains, including text games, web navigation, and robot manipulation. Our work indicates that, beyond recent advances in reasoning language models, RLVR offers a promising post-training paradigm for enhancing the utility of generative models more broadly. 

**Abstract (ZH)**: RLVR-World: 一种利用可验证奖励的强化学习框架以直接优化世界模型 

---
# Enhancing Robot Navigation Policies with Task-Specific Uncertainty Managements 

**Title (ZH)**: 基于任务特异性不确定性管理的机器人导航策略增强 

**Authors**: Gokul Puthumanaillam, Paulo Padrao, Jose Fuentes, Leonardo Bobadilla, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2505.13837)  

**Abstract**: Robots navigating complex environments must manage uncertainty from sensor noise, environmental changes, and incomplete information, with different tasks requiring varying levels of precision in different areas. For example, precise localization may be crucial near obstacles but less critical in open spaces. We present GUIDE (Generalized Uncertainty Integration for Decision-Making and Execution), a framework that integrates these task-specific requirements into navigation policies via Task-Specific Uncertainty Maps (TSUMs). By assigning acceptable uncertainty levels to different locations, TSUMs enable robots to adapt uncertainty management based on context. When combined with reinforcement learning, GUIDE learns policies that balance task completion and uncertainty management without extensive reward engineering. Real-world tests show significant performance gains over methods lacking task-specific uncertainty awareness. 

**Abstract (ZH)**: 机器人在复杂环境中的导航必须管理来自传感器噪声、环境变化和信息不完整性的不确定性，不同任务在不同区域需要不同程度的精度。例如，接近障碍物时精确定位可能是至关重要的，但在开阔空间中则相对不那么关键。我们提出了一种名为GUIDE（Generalized Uncertainty Integration for Decision-Making and Execution）的框架，该框架通过任务特定不确定性地图（TSUMs）将这些任务特定要求整合到导航策略中。通过为不同位置分配可接受的不确定性水平，TSUMs使机器人能够基于上下文调整不确定性管理。当与强化学习结合使用时，GUIDE能够在无需大量奖励工程的情况下学习平衡任务完成与不确定性管理的策略。实际测试表明，与缺乏任务特定不确定性意识的方法相比，GUIDE在性能上取得了显著提升。 

---
# Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning 

**Title (ZH)**: 基于策略的世界模型适应性调整以实现稳健的离线模型驱动强化学习 

**Authors**: Jiayu Chen, Aravind Venugopal, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2505.13709)  

**Abstract**: Offline reinforcement learning (RL) offers a powerful paradigm for data-driven control. Compared to model-free approaches, offline model-based RL (MBRL) explicitly learns a world model from a static dataset and uses it as a surrogate simulator, improving data efficiency and enabling potential generalization beyond the dataset support. However, most existing offline MBRL methods follow a two-stage training procedure: first learning a world model by maximizing the likelihood of the observed transitions, then optimizing a policy to maximize its expected return under the learned model. This objective mismatch results in a world model that is not necessarily optimized for effective policy learning. Moreover, we observe that policies learned via offline MBRL often lack robustness during deployment, and small adversarial noise in the environment can lead to significant performance degradation. To address these, we propose a framework that dynamically adapts the world model alongside the policy under a unified learning objective aimed at improving robustness. At the core of our method is a maximin optimization problem, which we solve by innovatively utilizing Stackelberg learning dynamics. We provide theoretical analysis to support our design and introduce computationally efficient implementations. We benchmark our algorithm on twelve noisy D4RL MuJoCo tasks and three stochastic Tokamak Control tasks, demonstrating its state-of-the-art performance. 

**Abstract (ZH)**: 离线强化学习（RL）提供了一种强大的数据驱动控制范式。与无模型方法相比，离线模型基于强化学习（MBRL）明确地从静态数据集中学习世界模型，并将其用作替代模拟器，提高数据效率并允许在数据集支持范围之外潜在地泛化。然而，现有的大多数离线MBRL方法遵循两阶段训练过程：首先通过最大化观测过渡的可能性来学习世界模型，然后在学习到的模型下优化策略以最大化其预期回报。这种目标不匹配导致学习到的世界模型不一定能有效地促进策略学习。此外，我们观察到通过离线MBRL学习的策略在部署时往往缺乏鲁棒性，环境中的小对抗噪声可能导致性能显著下降。为了应对这些问题，我们提出了一种框架，该框架在统一的学习目标下动态适应世界模型和策略，旨在提高鲁棒性。我们方法的核心是一个最大化最小优化问题，我们通过创新地利用Stackelberg学习动力学来解决这个问题。我们提供了理论分析以支持我们的设计，并引入了计算高效的实现。我们在十二个噪声D4RL MuJoCo任务和三个随机Tokamak控制任务上基准测试了我们的算法，展示了其最佳性能。 

---
# An agentic system with reinforcement-learned subsystem improvements for parsing form-like documents 

**Title (ZH)**: 基于强化学习子系统改进的代理系统及其在解析表单-like 文档中的应用 

**Authors**: Ayesha Amjad, Saurav Sthapit, Tahir Qasim Syed  

**Link**: [PDF](https://arxiv.org/pdf/2505.13504)  

**Abstract**: Extracting alphanumeric data from form-like documents such as invoices, purchase orders, bills, and financial documents is often performed via vision (OCR) and learning algorithms or monolithic pipelines with limited potential for systemic improvements. We propose an agentic AI system that leverages Large Language Model (LLM) agents and a reinforcement learning (RL) driver agent to automate consistent, self-improving extraction under LLM inference uncertainty. Our work highlights the limitations of monolithic LLM-based extraction and introduces a modular, multi-agent framework with task-specific prompts and an RL policy of rewards and penalties to guide a meta-prompting agent to learn from past errors and improve prompt-based actor agents. This self-corrective adaptive system handles diverse documents, file formats, layouts, and LLMs, aiming to automate accurate information extraction without the need for human intervention. Results as reported on two benchmark datasets of SOIRE, and CORD, are promising for the agentic AI framework. 

**Abstract (ZH)**: 基于视图（OCR）和学习算法或单一管道从发票、采购订单、账单和金融文档中提取 alphanumeric 数据往往具有有限的系统改进潜力。我们提出了一种代理型人工智能系统，利用大型语言模型（LLM）代理和强化学习（RL）驱动代理，以应对 LLM 推断不确定性，在自动化和自我改进的数据提取中取得一致效果。我们的工作揭示了单一 LLM 基础提取的局限性，并引入了一个模块化的多代理框架，该框架具有任务特定的提示和基于奖励和惩罚的 RL 策略，以引导元提示代理从过去错误中学习并改进基于提示的执行代理。此自我纠正的自适应系统能够处理多样化文档、文件格式、布局和 LLM，旨在在无需人工干预的情况下实现准确信息的自动化提取。在 SOIRE 和 CORD 的两个基准数据集上的结果表明，代理型人工智能框架前景广阔。 

---
