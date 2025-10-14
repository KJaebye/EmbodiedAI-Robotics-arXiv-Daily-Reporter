# Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation 

**Title (ZH)**: Phys2Real: 结合交互式在线适应的不确定性意识模拟到现实操作 

**Authors**: Maggie Wang, Stephen Tian, Aiden Swann, Ola Shorinwa, Jiajun Wu, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2510.11689)  

**Abstract**: Learning robotic manipulation policies directly in the real world can be expensive and time-consuming. While reinforcement learning (RL) policies trained in simulation present a scalable alternative, effective sim-to-real transfer remains challenging, particularly for tasks that require precise dynamics. To address this, we propose Phys2Real, a real-to-sim-to-real RL pipeline that combines vision-language model (VLM)-inferred physical parameter estimates with interactive adaptation through uncertainty-aware fusion. Our approach consists of three core components: (1) high-fidelity geometric reconstruction with 3D Gaussian splatting, (2) VLM-inferred prior distributions over physical parameters, and (3) online physical parameter estimation from interaction data. Phys2Real conditions policies on interpretable physical parameters, refining VLM predictions with online estimates via ensemble-based uncertainty quantification. On planar pushing tasks of a T-block with varying center of mass (CoM) and a hammer with an off-center mass distribution, Phys2Real achieves substantial improvements over a domain randomization baseline: 100% vs 79% success rate for the bottom-weighted T-block, 57% vs 23% in the challenging top-weighted T-block, and 15% faster average task completion for hammer pushing. Ablation studies indicate that the combination of VLM and interaction information is essential for success. Project website: this https URL . 

**Abstract (ZH)**: 将机器人操作策略直接在现实世界中学习可能会非常昂贵且耗时。虽然在模拟中训练的强化学习（RL）策略提供了一种可扩展的替代方案，但有效的仿真实现到实际应用的转移仍然具有挑战性，尤其是在需要精确动力学的任务中。为了解决这一问题，我们提出了一种名为Phys2Real的从现实到模拟再到现实的RL管道，该管道结合了基于视觉-语言模型（VLM）推断的物理参数估计与基于不确定性意识的交互适应。我们的方法包括三个核心组件：（1）高保真度的几何重建（使用3D高斯刺针），（2）基于VLM推断的物理参数先验分布，以及（3）从交互数据中在线估计物理参数。Phys2Real根据可解释的物理参数条件策略，并通过基于集成的不确定性量化使用在线估计来改进VLM预测。在带有可变质心（CoM）的平面推T块任务和具有非中心质量分布的锤子推任务中，Phys2Real在底重T块上的成功率达到了100%，而在挑战性的顶重T块上的成功率达到了57%，锤子推的平均任务完成时间快了15%。消融研究表明，VLM与交互信息的结合对于成功至关重要。项目网站：这个 https URL 。 

---
# Ego-Vision World Model for Humanoid Contact Planning 

**Title (ZH)**: 基于自我视觉的世界模型 humanoid 接触规划 

**Authors**: Hang Liu, Yuman Gao, Sangli Teng, Yufeng Chi, Yakun Sophia Shao, Zhongyu Li, Maani Ghaffari, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2510.11682)  

**Abstract**: Enabling humanoid robots to exploit physical contact, rather than simply avoid collisions, is crucial for autonomy in unstructured environments. Traditional optimization-based planners struggle with contact complexity, while on-policy reinforcement learning (RL) is sample-inefficient and has limited multi-task ability. We propose a framework combining a learned world model with sampling-based Model Predictive Control (MPC), trained on a demonstration-free offline dataset to predict future outcomes in a compressed latent space. To address sparse contact rewards and sensor noise, the MPC uses a learned surrogate value function for dense, robust planning. Our single, scalable model supports contact-aware tasks, including wall support after perturbation, blocking incoming objects, and traversing height-limited arches, with improved data efficiency and multi-task capability over on-policy RL. Deployed on a physical humanoid, our system achieves robust, real-time contact planning from proprioception and ego-centric depth images. Website: this https URL 

**Abstract (ZH)**: Enable Humanoid Robots to Exploit Physical Contact Rather Than Simply Avoid Collisions for Autonomy in Unstructured Environments 

---
# ManiAgent: An Agentic Framework for General Robotic Manipulation 

**Title (ZH)**: ManiAgent: 一种通用机器人操作的代理框架 

**Authors**: Yi Yang, Kefan Gu, Yuqing Wen, Hebei Li, Yucheng Zhao, Tiancai Wang, Xudong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11660)  

**Abstract**: While Vision-Language-Action (VLA) models have demonstrated impressive capabilities in robotic manipulation, their performance in complex reasoning and long-horizon task planning is limited by data scarcity and model capacity. To address this, we introduce ManiAgent, an agentic architecture for general manipulation tasks that achieves end-to-end output from task descriptions and environmental inputs to robotic manipulation actions. In this framework, multiple agents involve inter-agent communication to perform environmental perception, sub-task decomposition and action generation, enabling efficient handling of complex manipulation scenarios. Evaluations show ManiAgent achieves an 86.8% success rate on the SimplerEnv benchmark and 95.8% on real-world pick-and-place tasks, enabling efficient data collection that yields VLA models with performance comparable to those trained on human-annotated this http URL project webpage is available at this https URL. 

**Abstract (ZH)**: 而视觉-语言-动作（VLA）模型在机器人操作方面展示了令人印象深刻的 capability，但在复杂推理和长期任务规划方面的表现受限于数据稀缺性和模型容量。为了解决这一问题，我们提出了 ManiAgent，这是一种适用于通用操作任务的代理架构，能够从任务描述和环境输入端到端生成机器人操作行动。在该框架中，多个代理通过内部通信协同完成环境感知、子任务分解和行动生成，从而有效处理复杂的操作场景。评估结果显示，ManiAgent 在 SimplerEnv 基准测试中取得 86.8% 的成功率，在真实的取放任务中取得 95.8% 的成功率，实现了高效的的数据收集，从而可以生成与基于人类标注训练的模型性能相当的VLA模型。该项目网页地址为：这个 <https://> 项目网页地址为：这个 <https://>。 

---
# SCOOP'D: Learning Mixed-Liquid-Solid Scooping via Sim2Real Generative Policy 

**Title (ZH)**: SCOOP'D: 学习液体-固体混合物提取的Sim2Real生成策略 

**Authors**: Kuanning Wang, Yongchong Gu, Yuqian Fu, Zeyu Shangguan, Sicheng He, Xiangyang Xue, Yanwei Fu, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2510.11566)  

**Abstract**: Scooping items with tools such as spoons and ladles is common in daily life, ranging from assistive feeding to retrieving items from environmental disaster sites. However, developing a general and autonomous robotic scooping policy is challenging since it requires reasoning about complex tool-object interactions. Furthermore, scooping often involves manipulating deformable objects, such as granular media or liquids, which is challenging due to their infinite-dimensional configuration spaces and complex dynamics. We propose a method, SCOOP'D, which uses simulation from OmniGibson (built on NVIDIA Omniverse) to collect scooping demonstrations using algorithmic procedures that rely on privileged state information. Then, we use generative policies via diffusion to imitate demonstrations from observational input. We directly apply the learned policy in diverse real-world scenarios, testing its performance on various item quantities, item characteristics, and container types. In zero-shot deployment, our method demonstrates promising results across 465 trials in diverse scenarios, including objects of different difficulty levels that we categorize as "Level 1" and "Level 2." SCOOP'D outperforms all baselines and ablations, suggesting that this is a promising approach to acquiring robotic scooping skills. Project page is at this https URL. 

**Abstract (ZH)**: 使用OmniGibson进行模拟的SCOOP'D方法：基于生成政策的自主舀取技能学习 

---
# NaviGait: Navigating Dynamically Feasible Gait Libraries using Deep Reinforcement Learning 

**Title (ZH)**: NaviGait: 使用深度强化学习导航动态可行步态库 

**Authors**: Neil C. Janwani, Varun Madabushi, Maegan Tucker  

**Link**: [PDF](https://arxiv.org/pdf/2510.11542)  

**Abstract**: Reinforcement learning (RL) has emerged as a powerful method to learn robust control policies for bipedal locomotion. Yet, it can be difficult to tune desired robot behaviors due to unintuitive and complex reward design. In comparison, offline trajectory optimization methods, like Hybrid Zero Dynamics, offer more tuneable, interpretable, and mathematically grounded motion plans for high-dimensional legged systems. However, these methods often remain brittle to real-world disturbances like external perturbations.
In this work, we present NaviGait, a hierarchical framework that combines the structure of trajectory optimization with the adaptability of RL for robust and intuitive locomotion control. NaviGait leverages a library of offline-optimized gaits and smoothly interpolates between them to produce continuous reference motions in response to high-level commands. The policy provides both joint-level and velocity command residual corrections to modulate and stabilize the reference trajectories in the gait library. One notable advantage of NaviGait is that it dramatically simplifies reward design by encoding rich motion priors from trajectory optimization, reducing the need for finely tuned shaping terms and enabling more stable and interpretable learning. Our experimental results demonstrate that NaviGait enables faster training compared to conventional and imitation-based RL, and produces motions that remain closest to the original reference. Overall, by decoupling high-level motion generation from low-level correction, NaviGait offers a more scalable and generalizable approach for achieving dynamic and robust locomotion. 

**Abstract (ZH)**: 基于轨迹优化和强化学习的鲁棒导航步态控制框架 NaviGait 

---
# Simultaneous Calibration of Noise Covariance and Kinematics for State Estimation of Legged Robots via Bi-level Optimization 

**Title (ZH)**: 基于双层优化的同时校准噪声协方差与运动学以提高腿式机器人状态估计 

**Authors**: Denglin Cheng, Jiarong Kang, Xiaobin Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.11539)  

**Abstract**: Accurate state estimation is critical for legged and aerial robots operating in dynamic, uncertain environments. A key challenge lies in specifying process and measurement noise covariances, which are typically unknown or manually tuned. In this work, we introduce a bi-level optimization framework that jointly calibrates covariance matrices and kinematic parameters in an estimator-in-the-loop manner. The upper level treats noise covariances and model parameters as optimization variables, while the lower level executes a full-information estimator. Differentiating through the estimator allows direct optimization of trajectory-level objectives, resulting in accurate and consistent state estimates. We validate our approach on quadrupedal and humanoid robots, demonstrating significantly improved estimation accuracy and uncertainty calibration compared to hand-tuned baselines. Our method unifies state estimation, sensor, and kinematics calibration into a principled, data-driven framework applicable across diverse robotic platforms. 

**Abstract (ZH)**: 精确的状态估计对于在动态、不确定环境中操作的腿足和 aerial 机器人至关重要。一个关键挑战在于指定过程和测量噪声协方差矩阵，这些矩阵通常是未知的或需要手动调整。在本文中，我们提出了一种双层优化框架，该框架以闭环方式联合校准协方差矩阵和运动学参数。上层将噪声协方差和模型参数作为优化变量，而下层执行全信息估计器。通过对估计器进行求导，可以直接优化轨迹级目标，从而获得准确一致的状态估计。我们在四足机器人和类人机器人上验证了该方法，与手动调整的基线相比，显示出显著提高的估计准确性和不确定性校准。本方法将状态估计、传感器和运动学校准统一到一个基于原理的数据驱动框架中，适用于各种不同的机器人平台。 

---
# Constraint-Aware Reinforcement Learning via Adaptive Action Scaling 

**Title (ZH)**: 基于自适应动作缩放的约束aware强化学习 

**Authors**: Murad Dawood, Usama Ahmed Siddiquie, Shahram Khorshidi, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2510.11491)  

**Abstract**: Safe reinforcement learning (RL) seeks to mitigate unsafe behaviors that arise from exploration during training by reducing constraint violations while maintaining task performance. Existing approaches typically rely on a single policy to jointly optimize reward and safety, which can cause instability due to conflicting objectives, or they use external safety filters that override actions and require prior system knowledge. In this paper, we propose a modular cost-aware regulator that scales the agent's actions based on predicted constraint violations, preserving exploration through smooth action modulation rather than overriding the policy. The regulator is trained to minimize constraint violations while avoiding degenerate suppression of actions. Our approach integrates seamlessly with off-policy RL methods such as SAC and TD3, and achieves state-of-the-art return-to-cost ratios on Safety Gym locomotion tasks with sparse costs, reducing constraint violations by up to 126 times while increasing returns by over an order of magnitude compared to prior methods. 

**Abstract (ZH)**: 安全强化学习（RL）通过降低约束违反同时保持任务性能来缓解训练过程中出现的不安全行为。现有方法通常依赖单一策略同时优化奖励和安全性，这可能导致由于目标冲突引起的不稳定性，或者使用外部安全性过滤器，后者会覆盖动作并且需要先验系统知识。在这篇论文中，我们提出了一个模块化的成本意识调节器，根据预测的约束违反来调整代理的动作，通过平滑的动作调节而不是覆盖策略来保持探索。调节器被训练以最小化约束违反并避免过度抑制动作。我们的方法能够无缝集成到如SAC和TD3等离策略RL方法中，在Sparse Costs版本的安全体操任务中实现了现有的最佳回报与成本比率，与以前的方法相比，约束违反减少了多达126倍，回报提高了超过一个数量级。 

---
# HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data 

**Title (ZH)**: HiMaCon: 从未标注多模态数据中发现层次化操作概念 

**Authors**: Ruizhe Liu, Pei Zhou, Qian Luo, Li Sun, Jun Cen, Yibing Song, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11321)  

**Abstract**: Effective generalization in robotic manipulation requires representations that capture invariant patterns of interaction across environments and tasks. We present a self-supervised framework for learning hierarchical manipulation concepts that encode these invariant patterns through cross-modal sensory correlations and multi-level temporal abstractions without requiring human annotation. Our approach combines a cross-modal correlation network that identifies persistent patterns across sensory modalities with a multi-horizon predictor that organizes representations hierarchically across temporal scales. Manipulation concepts learned through this dual structure enable policies to focus on transferable relational patterns while maintaining awareness of both immediate actions and longer-term goals. Empirical evaluation across simulated benchmarks and real-world deployments demonstrates significant performance improvements with our concept-enhanced policies. Analysis reveals that the learned concepts resemble human-interpretable manipulation primitives despite receiving no semantic supervision. This work advances both the understanding of representation learning for manipulation and provides a practical approach to enhancing robotic performance in complex scenarios. 

**Abstract (ZH)**: 有效的机器人操作泛化需要能够捕捉跨环境和任务中不变交互模式的表示。我们提出了一种自监督框架，通过跨模态感官相关性和多层次时间抽象来自学习层次化的操作概念，无需人工标注。该方法结合了一个跨模态相关网络，用于识别跨感官模态的一贯模式，以及一个多时间尺度预测器，用于在时间尺度上层次化组织表示。通过这种双重结构学习的操作概念使策略能够关注可转移的关系模式，同时保持对即时动作和长期目标的意识。在模拟基准和实际部署中的 empirical 评估表明，增强有概念的操作策略显著提高了性能。分析表明，即使没有语义监督，学习到的概念类似于可由人类解释的操作基本单元。该工作不仅推进了操作领域表示学习的理解，还提供了一种在复杂场景中提升机器人性能的实用方法。 

---
# DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation 

**Title (ZH)**: DemoHLM: 从一个演示到可泛化的类人移动 manipulatio 

**Authors**: Yuhui Fu, Feiyang Xie, Chaoyi Xu, Jing Xiong, Haoqi Yuan, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11258)  

**Abstract**: Loco-manipulation is a fundamental challenge for humanoid robots to achieve versatile interactions in human environments. Although recent studies have made significant progress in humanoid whole-body control, loco-manipulation remains underexplored and often relies on hard-coded task definitions or costly real-world data collection, which limits autonomy and generalization. We present DemoHLM, a framework for humanoid loco-manipulation that enables generalizable loco-manipulation on a real humanoid robot from a single demonstration in simulation. DemoHLM adopts a hierarchy that integrates a low-level universal whole-body controller with high-level manipulation policies for multiple tasks. The whole-body controller maps whole-body motion commands to joint torques and provides omnidirectional mobility for the humanoid robot. The manipulation policies, learned in simulation via our data generation and imitation learning pipeline, command the whole-body controller with closed-loop visual feedback to execute challenging loco-manipulation tasks. Experiments show a positive correlation between the amount of synthetic data and policy performance, underscoring the effectiveness of our data generation pipeline and the data efficiency of our approach. Real-world experiments on a Unitree G1 robot equipped with an RGB-D camera validate the sim-to-real transferability of DemoHLM, demonstrating robust performance under spatial variations across ten loco-manipulation tasks. 

**Abstract (ZH)**: 基于演示的 humanoid 动 manipulation 框架：从模拟中的单次演示实现通用 humanoid 动 manipulation 

---
# A Primer on SO(3) Action Representations in Deep Reinforcement Learning 

**Title (ZH)**: SO(3) 行动表示在深度强化学习中的入门介绍 

**Authors**: Martin Schuck, Sherif Samy, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2510.11103)  

**Abstract**: Many robotic control tasks require policies to act on orientations, yet the geometry of SO(3) makes this nontrivial. Because SO(3) admits no global, smooth, minimal parameterization, common representations such as Euler angles, quaternions, rotation matrices, and Lie algebra coordinates introduce distinct constraints and failure modes. While these trade-offs are well studied for supervised learning, their implications for actions in reinforcement learning remain unclear. We systematically evaluate SO(3) action representations across three standard continuous control algorithms, PPO, SAC, and TD3, under dense and sparse rewards. We compare how representations shape exploration, interact with entropy regularization, and affect training stability through empirical studies and analyze the implications of different projections for obtaining valid rotations from Euclidean network outputs. Across a suite of robotics benchmarks, we quantify the practical impact of these choices and distill simple, implementation-ready guidelines for selecting and using rotation actions. Our results highlight that representation-induced geometry strongly influences exploration and optimization and show that representing actions as tangent vectors in the local frame yields the most reliable results across algorithms. 

**Abstract (ZH)**: 许多机器人控制任务需要策略对姿态进行操作，但SO(3)的几何结构使这变得非平凡。由于SO(3)无法拥有全局、平滑且最小的参数化表示，常用表示方法如欧拉角、四元数、旋转矩阵和李代数坐标引入了不同的约束和失效模式。尽管这些权衡在监督学习中已有充分研究，但对于强化学习中的动作而言，其影响仍不清楚。我们系统地评估了SO(3)动作表示方法在PPO、SAC和TD3三种标准连续控制算法下，在稠密和稀疏奖励条件下的性能。我们比较了不同表示方法如何影响探索、与熵正则化的交互以及通过实验证明训练稳定，并分析了从欧几里得网络输出获取有效旋转的不同投影的含义。在一系列机器人基准测试中，我们量化了这些选择的实际影响，并提炼出选择和使用旋转动作的简单实现指南。我们的结果强调，由表示引起的几何结构强烈影响探索和优化，并显示将动作表示为局部坐标系中的切向量是最可靠的结果。 

---
# PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System 

**Title (ZH)**: PhysHSI: 面向真实世界通用且自然的人形场景交互系统 

**Authors**: Huayi Wang, Wentao Zhang, Runyi Yu, Tao Huang, Junli Ren, Feiyu Jia, Zirui Wang, Xiaojie Niu, Xiao Chen, Jiahe Chen, Qifeng Chen, Jingbo Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11072)  

**Abstract**: Deploying humanoid robots to interact with real-world environments--such as carrying objects or sitting on chairs--requires generalizable, lifelike motions and robust scene perception. Although prior approaches have advanced each capability individually, combining them in a unified system is still an ongoing challenge. In this work, we present a physical-world humanoid-scene interaction system, PhysHSI, that enables humanoids to autonomously perform diverse interaction tasks while maintaining natural and lifelike behaviors. PhysHSI comprises a simulation training pipeline and a real-world deployment system. In simulation, we adopt adversarial motion prior-based policy learning to imitate natural humanoid-scene interaction data across diverse scenarios, achieving both generalization and lifelike behaviors. For real-world deployment, we introduce a coarse-to-fine object localization module that combines LiDAR and camera inputs to provide continuous and robust scene perception. We validate PhysHSI on four representative interactive tasks--box carrying, sitting, lying, and standing up--in both simulation and real-world settings, demonstrating consistently high success rates, strong generalization across diverse task goals, and natural motion patterns. 

**Abstract (ZH)**: 将类人机器人部署到与现实环境交互——例如搬运物体或坐在椅子上——需要具有广泛适用性和逼真表现力的动作以及稳健的场景感知。尽管先前的方法在各自的能力上取得了进展，但在统一系统中结合这些能力仍然是一个持续的挑战。在本工作中，我们提出了一种物理世界类人场景交互系统PhysHSI，该系统使类人机器人能够在保持自然和逼真行为的同时自主执行多样的交互任务。PhysHSI 包括一个仿真训练流水线和一个现实世界部署系统。在仿真中，我们采用基于对抗性运动先验的策略学习来模仿跨不同场景的自然类人场景交互数据，实现了通用性和逼真行为的结合。在现实世界部署中，我们引入了一种细粒度到粗粒度物体定位模块，结合LiDAR和摄像头输入以提供连续和稳健的场景感知。我们在四种代表性交互任务——搬运箱子、坐下、躺下和站立——的仿真和现实世界设置中验证了PhysHSI，展示了高度一致的成功率、广泛的跨任务目标泛化能力和自然的运动模式。 

---
# Unveiling Uncertainty-Aware Autonomous Cooperative Learning Based Planning Strategy 

**Title (ZH)**: 揭示基于自主协同学习的不确定性意识规划策略 

**Authors**: Shiyao Zhang, Liwei Deng, Shuyu Zhang, Weijie Yuan, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11041)  

**Abstract**: In future intelligent transportation systems, autonomous cooperative planning (ACP), becomes a promising technique to increase the effectiveness and security of multi-vehicle interactions. However, multiple uncertainties cannot be fully addressed for existing ACP strategies, e.g. perception, planning, and communication uncertainties. To address these, a novel deep reinforcement learning-based autonomous cooperative planning (DRLACP) framework is proposed to tackle various uncertainties on cooperative motion planning schemes. Specifically, the soft actor-critic (SAC) with the implementation of gate recurrent units (GRUs) is adopted to learn the deterministic optimal time-varying actions with imperfect state information occurred by planning, communication, and perception uncertainties. In addition, the real-time actions of autonomous vehicles (AVs) are demonstrated via the Car Learning to Act (CARLA) simulation platform. Evaluation results show that the proposed DRLACP learns and performs cooperative planning effectively, which outperforms other baseline methods under different scenarios with imperfect AV state information. 

**Abstract (ZH)**: 未来智能交通系统中基于深度强化学习的自主协同规划（DRLACP）框架 

---
# Refinery: Active Fine-tuning and Deployment-time Optimization for Contact-Rich Policies 

**Title (ZH)**: 炼厂：具有接触丰富政策的主动微调与部署时优化 

**Authors**: Bingjie Tang, Iretiayo Akinola, Jie Xu, Bowen Wen, Dieter Fox, Gaurav S. Sukhatme, Fabio Ramos, Abhishek Gupta, Yashraj Narang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11019)  

**Abstract**: Simulation-based learning has enabled policies for precise, contact-rich tasks (e.g., robotic assembly) to reach high success rates (~80%) under high levels of observation noise and control error. Although such performance may be sufficient for research applications, it falls short of industry standards and makes policy chaining exceptionally brittle. A key limitation is the high variance in individual policy performance across diverse initial conditions. We introduce Refinery, an effective framework that bridges this performance gap, robustifying policy performance across initial conditions. We propose Bayesian Optimization-guided fine-tuning to improve individual policies, and Gaussian Mixture Model-based sampling during deployment to select initializations that maximize execution success. Using Refinery, we improve mean success rates by 10.98% over state-of-the-art methods in simulation-based learning for robotic assembly, reaching 91.51% in simulation and comparable performance in the real world. Furthermore, we demonstrate that these fine-tuned policies can be chained to accomplish long-horizon, multi-part assembly$\unicode{x2013}$successfully assembling up to 8 parts without requiring explicit multi-step training. 

**Abstract (ZH)**: 基于模拟的学习方法使得精确且接触密集型任务（如机器人装配）的策略能够在高观测噪声和控制误差下达到高达80%的成功率。尽管这种性能可能适合研究应用，但在工业化标准下却表现不足，使得策略链接变得极其脆弱。一个关键的限制是单个策略在不同初始条件下的性能差异性较高。我们引入了Refinery这一有效的框架，以弥合这一性能差距，提升策略在不同初始条件下的鲁棒性。我们提出使用贝叶斯优化引导的微调来提高单个策略的表现，并在部署时利用高斯混合模型采样来选择最大化执行成功率的初始化条件。通过Refinery，我们在基于模拟的学习方法中提高了机器人装配任务的成功率平均值10.98%，在模拟环境中达到了91.51%的成功率，并且在实际环境中表现相当。此外，我们证明了这些微调后的策略可以被链接起来完成多零件的长期装配任务，成功装配多达8个零件而无需进行显式的多步训练。 

---
# Into the Unknown: Towards using Generative Models for Sampling Priors of Environment Uncertainty for Planning in Configuration Spaces 

**Title (ZH)**: 未知之境：面向配置空间规划中环境不确定性先验采样的生成模型研究 

**Authors**: Subhransu S. Bhattacharjee, Hao Lu, Dylan Campbell, Rahul Shome  

**Link**: [PDF](https://arxiv.org/pdf/2510.11014)  

**Abstract**: Priors are vital for planning under partial observability, yet difficult to obtain in practice. We present a sampling-based pipeline that leverages large-scale pretrained generative models to produce probabilistic priors capturing environmental uncertainty and spatio-semantic relationships in a zero-shot manner. Conditioned on partial observations, the pipeline recovers complete RGB-D point cloud samples with occupancy and target semantics, formulated to be directly useful in configuration-space planning. We establish a Matterport3D benchmark of rooms partially visible through doorways, where a robot must navigate to an unobserved target object. Effective priors for this setting must represent both occupancy and target-location uncertainty in unobserved regions. Experiments show that our approach recovers commonsense spatial semantics consistent with ground truth, yielding diverse, clean 3D point clouds usable in motion planning, highlight the promise of generative models as a rich source of priors for robotic planning. 

**Abstract (ZH)**: 基于大规模预训练生成模型的采样管道在部分可观测性下的规划中至关重要但难以获取：我们提出了一种零样本方式利用大规模预训练生成模型生成捕捉环境不确定性和空间语义关系的概率先验的采样管道。该管道在部分观测条件下恢复完整的RGB-D点云样本，并且这些样本包含 occupancy 和目标语义，可以直接用于配置空间规划。我们在一个 Matterport3D 基准中建立了通过门洞部分可见的房间场景，其中机器人需要导航至一个未被观测到的目标物体。有效的先验在这种设定中必须同时表示未观测区域中的 occupancy 和目标位置的不确定性。实验结果显示，我们的方法恢复了与真实世界一致的常识性空间语义，生成了多样且干净的3D点云，可用于运动规划，突显了生成模型作为机器人规划中丰富先验来源的潜力。 

---
# RoVer: Robot Reward Model as Test-Time Verifier for Vision-Language-Action Model 

**Title (ZH)**: RoVer: 机器人奖励模型作为视觉-语言-行动模型的测试时验证器 

**Authors**: Mingtong Dai, Lingbo Liu, Yongjie Bai, Yang Liu, Zhouxia Wang, Rui SU, Chunjie Chen, Liang Lin, Xinyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10975)  

**Abstract**: Vision-Language-Action (VLA) models have become a prominent paradigm for embodied intelligence, yet further performance improvements typically rely on scaling up training data and model size -- an approach that is prohibitively expensive for robotics and fundamentally limited by data collection this http URL address this limitation with $\mathbf{RoVer}$, an embodied test-time scaling framework that uses a $\mathbf{Ro}$bot Process Reward Model (PRM) as a Test-Time $\mathbf{Ver}$ifier to enhance the capabilities of existing VLA models without modifying their architectures or weights. Specifically, RoVer (i) assigns scalar-based process rewards to evaluate the reliability of candidate actions, and (ii) predicts an action-space direction for candidate expansion/refinement. During inference, RoVer generates multiple candidate actions concurrently from the base policy, expands them along PRM-predicted directions, and then scores all candidates with PRM to select the optimal action for execution. Notably, by caching shared perception features, it can amortize perception cost and evaluate more candidates under the same test-time computational budget. Essentially, our approach effectively transforms available computing resources into better action decision-making, realizing the benefits of test-time scaling without extra training overhead. Our contributions are threefold: (1) a general, plug-and-play test-time scaling framework for VLAs; (2) a PRM that jointly provides scalar process rewards and an action-space direction to guide exploration; and (3) an efficient direction-guided sampling strategy that leverages a shared perception cache to enable scalable candidate generation and selection during inference. 

**Abstract (ZH)**: RoVer：一种用于Vision-Language-Action模型的embodied测试时扩展框架 

---
# Game-Theoretic Risk-Shaped Reinforcement Learning for Safe Autonomous Driving 

**Title (ZH)**: 基于博弈论的风险形塑强化学习的安全自主驾驶 

**Authors**: Dong Hu, Fenqing Hu, Lidong Yang, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10960)  

**Abstract**: Ensuring safety in autonomous driving (AD) remains a significant challenge, especially in highly dynamic and complex traffic environments where diverse agents interact and unexpected hazards frequently emerge. Traditional reinforcement learning (RL) methods often struggle to balance safety, efficiency, and adaptability, as they primarily focus on reward maximization without explicitly modeling risk or safety constraints. To address these limitations, this study proposes a novel game-theoretic risk-shaped RL (GTR2L) framework for safe AD. GTR2L incorporates a multi-level game-theoretic world model that jointly predicts the interactive behaviors of surrounding vehicles and their associated risks, along with an adaptive rollout horizon that adjusts dynamically based on predictive uncertainty. Furthermore, an uncertainty-aware barrier mechanism enables flexible modulation of safety boundaries. A dedicated risk modeling approach is also proposed, explicitly capturing both epistemic and aleatoric uncertainty to guide constrained policy optimization and enhance decision-making in complex environments. Extensive evaluations across diverse and safety-critical traffic scenarios show that GTR2L significantly outperforms state-of-the-art baselines, including human drivers, in terms of success rate, collision and violation reduction, and driving efficiency. The code is available at this https URL. 

**Abstract (ZH)**: 确保自动驾驶的安全性仍然是一个重大挑战，尤其是在动态复杂且存在多种交互代理和频发意外风险的交通环境中。传统的强化学习方法往往难以在安全、效率和适应性之间找到平衡，因为它们主要侧重于奖励最大化，而未明确建模风险或安全约束。为解决这些局限性，本研究提出了一种新颖的博弈论风险形强化学习（GTR2L）框架，以实现安全的自动驾驶。GTR2L结合了多层次的博弈论世界模型，该模型能够联合预测周围车辆的交互行为及其相关风险，并配备了一个能够根据预测不确定性动态调整的适应性展开 horizons。此外，还提出了一种不确定性感知的屏障机制，实现灵活的安全边界调节。同时，还提出了一种专门的风险建模方法，明确捕获先验和统计不确定性，以引导约束策略优化并增强在复杂环境中的决策能力。在多种多样且安全关键的交通场景中的广泛评估表明，GTR2L在成功率、碰撞和违规减少以及驾驶效率方面显著优于最先进的基线方法，包括人类驾驶员。相关代码可在以下链接获取。 

---
# More than A Point: Capturing Uncertainty with Adaptive Affordance Heatmaps for Spatial Grounding in Robotic Tasks 

**Title (ZH)**: 不仅仅是单一点：通过自适应 affordance 热图捕捉空间定位不确定性在机器人任务中的应用 

**Authors**: Xinyu Shao, Yanzhe Tang, Pengwei Xie, Kaiwen Zhou, Yuzheng Zhuang, Xingyue Quan, Jianye Hao, Long Zeng, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.10912)  

**Abstract**: Many language-guided robotic systems rely on collapsing spatial reasoning into discrete points, making them brittle to perceptual noise and semantic ambiguity. To address this challenge, we propose RoboMAP, a framework that represents spatial targets as continuous, adaptive affordance heatmaps. This dense representation captures the uncertainty in spatial grounding and provides richer information for downstream policies, thereby significantly enhancing task success and interpretability. RoboMAP surpasses the previous state-of-the-art on a majority of grounding benchmarks with up to a 50x speed improvement, and achieves an 82\% success rate in real-world manipulation. Across extensive simulated and physical experiments, it demonstrates robust performance and shows strong zero-shot generalization to navigation. More details and videos can be found at this https URL. 

**Abstract (ZH)**: 一种基于连续自适应可用性Heatmap的空间目标表示框架：RoboMAP及其在任务执行中的应用 

---
# Towards a Unified Understanding of Robot Manipulation: A Comprehensive Survey 

**Title (ZH)**: 面向机器人操作统一理解的研究综合综述 

**Authors**: Shuanghao Bai, Wenxuan Song, Jiayi Chen, Yuheng Ji, Zhide Zhong, Jin Yang, Han Zhao, Wanqi Zhou, Wei Zhao, Zhe Li, Pengxiang Ding, Cheng Chi, Haoang Li, Chang Xu, Xiaolong Zheng, Donglin Wang, Shanghang Zhang, Badong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10903)  

**Abstract**: Embodied intelligence has witnessed remarkable progress in recent years, driven by advances in computer vision, natural language processing, and the rise of large-scale multimodal models. Among its core challenges, robot manipulation stands out as a fundamental yet intricate problem, requiring the seamless integration of perception, planning, and control to enable interaction within diverse and unstructured environments. This survey presents a comprehensive overview of robotic manipulation, encompassing foundational background, task-organized benchmarks and datasets, and a unified taxonomy of existing methods. We extend the classical division between high-level planning and low-level control by broadening high-level planning to include language, code, motion, affordance, and 3D representations, while introducing a new taxonomy of low-level learning-based control grounded in training paradigms such as input modeling, latent learning, and policy learning. Furthermore, we provide the first dedicated taxonomy of key bottlenecks, focusing on data collection, utilization, and generalization, and conclude with an extensive review of real-world applications. Compared with prior surveys, our work offers both a broader scope and deeper insight, serving as an accessible roadmap for newcomers and a structured reference for experienced researchers. All related resources, including research papers, open-source datasets, and projects, are curated for the community at this https URL. 

**Abstract (ZH)**: embodiable 智能在近年来取得了显著进展，得益于计算机视觉、自然语言处理的进步以及大规模多模态模型的崛起。其中，机器人的操作是一个核心挑战，也是基本而又复杂的任务，需要融合感知、规划和控制以实现与多样化和非结构化环境的交互。本文综述提供了机器人操作的全面概述，包括基础知识、任务组织的基准和数据集以及现有方法的统一分类体系。我们扩展了传统意义上的高层规划与低层控制的划分，将高层规划扩展至包括语言、代码、动作、可用性和三维表示，并引入了一种新的基于输入建模、潜在学习和策略学习的低层学习控制分类体系。此外，我们提供了首个针对关键瓶颈的专门分类，关注数据的收集、利用和泛化，并详细回顾了实际应用。与之前的综述相比，我们的工作提供了更广泛的研究范围和更深刻的见解，为新手提供了一条易于理解的路线图，并为经验丰富的研究人员提供了结构化的参考。所有相关的资源，包括科研论文、开源数据集和项目，均可通过此网址 https://www.example.com 获取。 

---
# GRIP: A Unified Framework for Grid-Based Relay and Co-Occurrence-Aware Planning in Dynamic Environments 

**Title (ZH)**: GRIP：一种基于网格的中继与共现意识规划的统一框架 

**Authors**: Ahmed Alanazi, Duy Ho, Yugyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.10865)  

**Abstract**: Robots navigating dynamic, cluttered, and semantically complex environments must integrate perception, symbolic reasoning, and spatial planning to generalize across diverse layouts and object categories. Existing methods often rely on static priors or limited memory, constraining adaptability under partial observability and semantic ambiguity. We present GRIP, Grid-based Relay with Intermediate Planning, a unified, modular framework with three scalable variants: GRIP-L (Lightweight), optimized for symbolic navigation via semantic occupancy grids; GRIP-F (Full), supporting multi-hop anchor chaining and LLM-based introspection; and GRIP-R (Real-World), enabling physical robot deployment under perceptual uncertainty. GRIP integrates dynamic 2D grid construction, open-vocabulary object grounding, co-occurrence-aware symbolic planning, and hybrid policy execution using behavioral cloning, D* search, and grid-conditioned control. Empirical results on AI2-THOR and RoboTHOR benchmarks show that GRIP achieves up to 9.6% higher success rates and over $2\times$ improvement in path efficiency (SPL and SAE) on long-horizon tasks. Qualitative analyses reveal interpretable symbolic plans in ambiguous scenes. Real-world deployment on a Jetbot further validates GRIP's generalization under sensor noise and environmental variation. These results position GRIP as a robust, scalable, and explainable framework bridging simulation and real-world navigation. 

**Abstract (ZH)**: 基于网格的中间规划框架GRIP：动态、杂乱且语义复杂的环境下的导航 

---
# Preference-Conditioned Multi-Objective RL for Integrated Command Tracking and Force Compliance in Humanoid Locomotion 

**Title (ZH)**: 基于偏好条件的多目标强化学习在类人行走中的综合指令跟踪与力量合规性 

**Authors**: Tingxuan Leng, Yushi Wang, Tinglong Zheng, Changsheng Luo, Mingguo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.10851)  

**Abstract**: Humanoid locomotion requires not only accurate command tracking for navigation but also compliant responses to external forces during human interaction. Despite significant progress, existing RL approaches mainly emphasize robustness, yielding policies that resist external forces but lack compliance-particularly challenging for inherently unstable humanoids. In this work, we address this by formulating humanoid locomotion as a multi-objective optimization problem that balances command tracking and external force compliance. We introduce a preference-conditioned multi-objective RL (MORL) framework that integrates rigid command following and compliant behaviors within a single omnidirectional locomotion policy. External forces are modeled via velocity-resistance factor for consistent reward design, and training leverages an encoder-decoder structure that infers task-relevant privileged features from deployable observations. We validate our approach in both simulation and real-world experiments on a humanoid robot. Experimental results indicate that our framework not only improves adaptability and convergence over standard pipelines, but also realizes deployable preference-conditioned humanoid locomotion. 

**Abstract (ZH)**: humanoid 行走需要精确的命令跟踪以进行导航，也需要对外部力的合规响应以便于与人类的交互。尽管取得了显著进展，现有的强化学习方法主要强调鲁棒性，生成的策略能够抵抗外部力但缺乏合规性——这对于本就不稳定的人形机器人来说尤其具有挑战性。在本文中，我们将人形行走问题形式化为一个兼顾命令跟踪和外部力合规性的多目标优化问题。我们引入了一种基于偏好条件的多目标强化学习（MORL）框架，将刚性命令跟随和合规行为整合到一个全向行走策略中。通过引入速度阻力因子来建模外部力，确保奖励设计的一致性，并利用编码器-解码器结构从可部署观测中推断出与任务相关的优势特征进行训练。我们在仿真实验和现实世界的人形机器人实验中验证了该方法。实验结果表明，我们的框架不仅在标准流水线之上提高了适应性和收敛性，还实现了可部署的偏好条件人形行走。 

---
# Real2USD: Scene Representations in Universal Scene Description Language 

**Title (ZH)**: Real2USD: 全景描述语言中的场景表示 

**Authors**: Christopher D. Hsu, Pratik Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2510.10778)  

**Abstract**: Large Language Models (LLMs) can help robots reason about abstract task specifications. This requires augmenting classical representations of the environment used by robots with natural language-based priors. There are a number of existing approaches to doing so, but they are tailored to specific tasks, e.g., visual-language models for navigation, language-guided neural radiance fields for mapping, etc. This paper argues that the Universal Scene Description (USD) language is an effective and general representation of geometric, photometric and semantic information in the environment for LLM-based robotics tasks. Our argument is simple: a USD is an XML-based scene graph, readable by LLMs and humans alike, and rich enough to support essentially any task -- Pixar developed this language to store assets, scenes and even movies. We demonstrate a ``Real to USD'' system using a Unitree Go2 quadruped robot carrying LiDAR and a RGB camera that (i) builds an explicit USD representation of indoor environments with diverse objects and challenging settings with lots of glass, and (ii) parses the USD using Google's Gemini to demonstrate scene understanding, complex inferences, and planning. We also study different aspects of this system in simulated warehouse and hospital settings using Nvidia's Issac Sim. Code is available at this https URL . 

**Abstract (ZH)**: 大型语言模型（LLMs）可以帮助机器人处理抽象的任务规范。这要求将自然语言先验知识添加到机器人常用的环境经典表示中。已经有多种现有的方法来实现这一点，但它们通常针对特定任务，例如用于导航的视觉语言模型、用于建图的语言指导神经光照字段等。本文认为，通用场景描述（USD）语言是LLM基于的机器人任务中环境的几何、光度和语义信息的有效且通用的表示。我们的论点很简单：USD是一种基于XML的场景图，既可被LLM和人类阅读，又足够丰富以支持几乎所有任务——皮克斯开发此语言用于存储资产、场景甚至电影。我们演示了一个“真实到USD”系统，使用Unitree Go2四足机器人携带LiDAR和RGB相机来（i）构建包含多种物体和具有大量玻璃材质的室内环境的显式USD表示，并（ii）利用Google的Gemini解析USD来展示场景理解、复杂推理和规划。我们还在使用Nvidia的Issac Sim模拟的仓库和医院环境中研究了该系统的不同方面。代码可在以下链接获取。 

---
# Gain Tuning Is Not What You Need: Reward Gain Adaptation for Constrained Locomotion Learning 

**Title (ZH)**: 获得调整并非你需要：受约束运动学习中的奖励增益适应 

**Authors**: Arthicha Srisuchinnawong, Poramate Manoonpong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10759)  

**Abstract**: Existing robot locomotion learning techniques rely heavily on the offline selection of proper reward weighting gains and cannot guarantee constraint satisfaction (i.e., constraint violation) during training. Thus, this work aims to address both issues by proposing Reward-Oriented Gains via Embodied Regulation (ROGER), which adapts reward-weighting gains online based on penalties received throughout the embodied interaction process. The ratio between the positive reward (primary reward) and negative reward (penalty) gains is automatically reduced as the learning approaches the constraint thresholds to avoid violation. Conversely, the ratio is increased when learning is in safe states to prioritize performance. With a 60-kg quadruped robot, ROGER achieved near-zero constraint violation throughout multiple learning trials. It also achieved up to 50% more primary reward than the equivalent state-of-the-art techniques. In MuJoCo continuous locomotion benchmarks, including a single-leg hopper, ROGER exhibited comparable or up to 100% higher performance and 60% less torque usage and orientation deviation compared to those trained with the default reward function. Finally, real-world locomotion learning of a physical quadruped robot was achieved from scratch within one hour without any falls. Therefore, this work contributes to constraint-satisfying real-world continual robot locomotion learning and simplifies reward weighting gain tuning, potentially facilitating the development of physical robots and those that learn in the real world. 

**Abstract (ZH)**: 基于体感调节的目标导向增益：解决机器人运动学习中的约束满足与奖励调优问题 

---
# Deployment and Development of a Cognitive Teleoreactive Framework for Deep Sea Autonomy 

**Title (ZH)**: 深海自主认知遥控框架的部署与开发 

**Authors**: Christopher Thierauf  

**Link**: [PDF](https://arxiv.org/pdf/2510.10716)  

**Abstract**: A new AUV mission planning and execution software has been tested on AUV Sentry. Dubbed DINOS-R, it draws inspiration from cognitive architectures and AUV control systems to replace the legacy MC architecture. Unlike these existing architectures, however, DINOS-R is built from the ground-up to unify symbolic decision making (for understandable, repeatable, provable behavior) with machine learning techniques and reactive behaviors, for field-readiness across oceanographic platforms. Implemented primarily in Python3, DINOS-R is extensible, modular, and reusable, with an emphasis on non-expert use as well as growth for future research in oceanography and robot algorithms. Mission specification is flexible, and can be specified declaratively. Behavior specification is similarly flexible, supporting simultaneous use of real-time task planning and hard-coded user specified plans. These features were demonstrated in the field on Sentry, in addition to a variety of simulated cases. These results are discussed, and future work is outlined. 

**Abstract (ZH)**: 一种新的AUV任务规划与执行软件已在AUV Sentry上进行测试。该软件名为DINOS-R，受认知架构和AUV控制系统启发，用于替代现有的MC架构。与现有的架构不同，DINOS-R从头构建，旨在统一符号决策（实现可理解、可重复、可证明的行为）与机器学习技术及反应性行为，以使海洋观测平台具备现役能力。DINOS-R主要使用Python3实现，具有可扩展性、模块化和可重用性，强调非专家使用，并为未来海洋学和机器人算法研究的增长奠定基础。任务规范和行为规范都具有灵活性，支持同时使用实时任务规划和用户指定的计划。这些特性的有效性已在Sentry的实际应用和多种模拟案例中得到验证。本文讨论了这些结果，并概述了未来的工作方向。 

---
# UniCoD: Enhancing Robot Policy via Unified Continuous and Discrete Representation Learning 

**Title (ZH)**: UniCoD: 增强机器人策略的统一连续与离散表示学习 

**Authors**: Jianke Zhang, Yucheng Hu, Yanjiang Guo, Xiaoyu Chen, Yichen Liu, Wenna Chen, Chaochao Lu, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10642)  

**Abstract**: Building generalist robot policies that can handle diverse tasks in open-ended environments is a central challenge in robotics. To leverage knowledge from large-scale pretraining, prior work has typically built generalist policies either on top of vision-language understanding models (VLMs) or generative models. However, both semantic understanding from vision-language pretraining and visual dynamics modeling from visual-generation pretraining are crucial for embodied robots. Recent unified models of generation and understanding have demonstrated strong capabilities in both comprehension and generation through large-scale pretraining. We posit that robotic policy learning can likewise benefit from the combined strengths of understanding, planning and continuous future representation learning. Building on this insight, we introduce UniCoD, which acquires the ability to dynamically model high-dimensional visual features through pretraining on over 1M internet-scale instructional manipulation videos. Subsequently, UniCoD is fine-tuned on data collected from the robot embodiment, enabling the learning of mappings from predictive representations to action tokens. Extensive experiments show our approach consistently outperforms baseline methods in terms of 9\% and 12\% across simulation environments and real-world out-of-distribution tasks. 

**Abstract (ZH)**: 构建能够在开放环境中处理多样化任务的一般主义机器人政策是机器人学中的一个核心挑战。尽管前期工作的重点是基于视觉-语言理解模型（VLMs）或生成模型来构建一般主义政策，但体现式机器人所需的视觉语义理解能力和视觉动力学建模能力同样重要。最近出现的生成与理解统一模型通过大规模预训练展示了在理解和生成方面的强大能力。我们认为，机器人策略学习可以从理解、规划和连续未来表示学习的综合优势中受益。基于此见解，我们引入了UniCoD，通过在超过100万规模的互联网指令操作视频上进行预训练，实现了动态建模高维度视觉特征的能力。随后，UniCoD 在机器人实体收集的数据上进行微调，以学习从预测表示到动作标记的映射。大量实验表明，我们的方法在仿真环境和真实世界未知任务中分别比基线方法高出9%和12%的性能。 

---
# High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting 

**Title (ZH)**: 使用高保真模拟数据生成进行真实世界零样本机器人操作学习的高斯散点图方法 

**Authors**: Haoyu Zhao, Cheng Zeng, Linghao Zhuang, Yaxi Zhao, Shengke Xue, Hao Wang, Xingyue Zhao, Zhongyu Li, Kehan Li, Siteng Huang, Mingxiu Chen, Xin Li, Deli Zhao, Hua Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.10637)  

**Abstract**: The scalability of robotic learning is fundamentally bottlenecked by the significant cost and labor of real-world data collection. While simulated data offers a scalable alternative, it often fails to generalize to the real world due to significant gaps in visual appearance, physical properties, and object interactions. To address this, we propose RoboSimGS, a novel Real2Sim2Real framework that converts multi-view real-world images into scalable, high-fidelity, and physically interactive simulation environments for robotic manipulation. Our approach reconstructs scenes using a hybrid representation: 3D Gaussian Splatting (3DGS) captures the photorealistic appearance of the environment, while mesh primitives for interactive objects ensure accurate physics simulation. Crucially, we pioneer the use of a Multi-modal Large Language Model (MLLM) to automate the creation of physically plausible, articulated assets. The MLLM analyzes visual data to infer not only physical properties (e.g., density, stiffness) but also complex kinematic structures (e.g., hinges, sliding rails) of objects. We demonstrate that policies trained entirely on data generated by RoboSimGS achieve successful zero-shot sim-to-real transfer across a diverse set of real-world manipulation tasks. Furthermore, data from RoboSimGS significantly enhances the performance and generalization capabilities of SOTA methods. Our results validate RoboSimGS as a powerful and scalable solution for bridging the sim-to-real gap. 

**Abstract (ZH)**: 基于实2仿2实的机器人学习可扩展性瓶颈由现实世界数据的高成本和劳动限制。虽然仿真数据提供了可扩展的替代方案，但由于视觉外观、物理属性和物体交互的巨大差距，它往往无法泛化到现实世界中。为了解决这个问题，我们提出了一种名为RoboSimGS的新颖框架，该框架将多视角的现实世界图像转换为用于机器人操作的可扩展、高保真和物理交互的仿真环境。我们的方法使用混合表示：3D高斯点云（3DGS）捕捉环境的逼真外观，而可交互对象的网格基础确保了准确的物理仿真。关键的是，我们首次使用多模态大规模语言模型（MLLM）来自动化生成物理上可信的、具有关节的实体资产。MLLM分析视觉数据，不仅推断出物理属性（如密度、刚度），还推断出物体的复杂运动学结构（如铰链、滑动导轨）。我们证明了仅在由RoboSimGS生成的数据上训练的策略能够在多样化的现实世界操作任务中实现成功的无监督仿真实验到现实世界的任务转移。此外，来自RoboSimGS的数据显著提升了现有最佳方法的性能和泛化能力。我们的结果验证了RoboSimGS作为跨越仿真实验到现实世界差距的强大多尺度解决方案的有效性和可扩展性。 

---
# Reinforcement Learning-based Dynamic Adaptation for Sampling-Based Motion Planning in Agile Autonomous Driving 

**Title (ZH)**: 基于强化学习的采样基于运动规划的灵活自主驾驶动态自适应方法 

**Authors**: Alexander Langmann, Yevhenii Tokarev, Mattia Piccinini, Korbinian Moller, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2510.10567)  

**Abstract**: Sampling-based trajectory planners are widely used for agile autonomous driving due to their ability to generate fast, smooth, and kinodynamically feasible trajectories. However, their behavior is often governed by a cost function with manually tuned, static weights, which forces a tactical compromise that is suboptimal across the wide range of scenarios encountered in a race. To address this shortcoming, we propose using a Reinforcement Learning (RL) agent as a high-level behavioral selector that dynamically switches the cost function parameters of an analytical, low-level trajectory planner during runtime. We show the effectiveness of our approach in simulation in an autonomous racing environment where our RL-based planner achieved 0% collision rate while reducing overtaking time by up to 60% compared to state-of-the-art static planners. Our new agent now dynamically switches between aggressive and conservative behaviors, enabling interactive maneuvers unattainable with static configurations. These results demonstrate that integrating reinforcement learning as a high-level selector resolves the inherent trade-off between safety and competitiveness in autonomous racing planners. The proposed methodology offers a pathway toward adaptive yet interpretable motion planning for broader autonomous driving applications. 

**Abstract (ZH)**: 基于采样的轨迹规划器广泛用于敏捷自动驾驶，因其能够生成快速、平滑且动力学可行的轨迹。然而，它们的行为往往由具有手动调谐且静态权重的成本函数所控制，这导致了一种在竞赛中遇到的广泛情况下的次优战术妥协。为了解决这一不足，我们提出使用强化学习（RL）代理作为高级行为选择器，在运行时动态切换低级轨迹规划器的成本函数参数。我们在自主赛车环境的仿真中展示了我们方法的有效性，我们的基于RL的规划器实现了0%的碰撞率，并将超过掉头时间减少了高达60%，优于最先进的静态规划器。我们的新代理现在能够在运行时动态地在激进和保守行为之间切换，从而实现静态配置无法达成的交互式机动。这些结果表明，将强化学习作为高级选择器纳入其中解决了自主赛车规划器中固有的安全性与竞争力之间的权衡。所提出的方法为更广泛的自动驾驶应用中的适应性且可解释的运动规划提供了一条途径。 

---
# Population-Coded Spiking Neural Networks for High-Dimensional Robotic Control 

**Title (ZH)**: 高维机器人控制的 population 编码放电神经网络 

**Authors**: Kanishkha Jaisankar, Xiaoyang Jiang, Feifan Liao, Jeethu Sreenivas Amuthan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10516)  

**Abstract**: Energy-efficient and high-performance motor control remains a critical challenge in robotics, particularly for high-dimensional continuous control tasks with limited onboard resources. While Deep Reinforcement Learning (DRL) has achieved remarkable results, its computational demands and energy consumption limit deployment in resource-constrained environments. This paper introduces a novel framework combining population-coded Spiking Neural Networks (SNNs) with DRL to address these challenges. Our approach leverages the event-driven, asynchronous computation of SNNs alongside the robust policy optimization capabilities of DRL, achieving a balance between energy efficiency and control performance. Central to this framework is the Population-coded Spiking Actor Network (PopSAN), which encodes high-dimensional observations into neuronal population activities and enables optimal policy learning through gradient-based updates. We evaluate our method on the Isaac Gym platform using the PixMC benchmark with complex robotic manipulation tasks. Experimental results on the Franka robotic arm demonstrate that our approach achieves energy savings of up to 96.10% compared to traditional Artificial Neural Networks (ANNs) while maintaining comparable control performance. The trained SNN policies exhibit robust finger position tracking with minimal deviation from commanded trajectories and stable target height maintenance during pick-and-place operations. These results position population-coded SNNs as a promising solution for energy-efficient, high-performance robotic control in resource-constrained applications, paving the way for scalable deployment in real-world robotics systems. 

**Abstract (ZH)**: 高能效和高性能的电机控制依然是机器人领域中的关键挑战，特别是在资源受限的环境中执行高维连续控制任务时。尽管深度强化学习（DRL）取得了显著成果，但其计算需求和能量消耗限制了其在资源受限环境中的部署。本文提出了一种结合脉冲编码神经网络（SNN）和DRL的新框架，以应对这些挑战。我们的方法利用了SNN的事件驱动和异步计算特性，以及DRL稳健的策略优化能力，实现了能量效率和控制性能的平衡。该框架的核心是脉冲编码神经元演员网络（PopSAN），该网络将高维观测编码为神经元群体活动，并通过梯度更新实现最优策略学习。我们使用Isaac Gym平台和PixMC基准对复杂机器人操作任务进行了评估。实验结果表明，与传统的神经网络（ANNS）相比，我们的方法在保持类似控制性能的同时，实现了高达96.10%的能量节约。训练好的SNN策略在手指位置跟踪和拾放操作中表现出强大的鲁棒性，并能稳定维持目标高度。这些结果表明，脉冲编码SNN是资源受限环境中高能效和高性能机器人控制的有前途的解决方案，为实际机器人系统的可扩展部署铺平了道路。 

---
# SuperEx: Enhancing Indoor Mapping and Exploration using Non-Line-of-Sight Perception 

**Title (ZH)**: SuperEx: 使用非视距感知增强室内地图构建与探索 

**Authors**: Kush Garg, Akshat Dave  

**Link**: [PDF](https://arxiv.org/pdf/2510.10506)  

**Abstract**: Efficient exploration and mapping in unknown indoor environments is a fundamental challenge, with high stakes in time-critical settings. In current systems, robot perception remains confined to line-of-sight; occluded regions remain unknown until physically traversed, leading to inefficient exploration when layouts deviate from prior assumptions. In this work, we bring non-line-of-sight (NLOS) sensing to robotic exploration. We leverage single-photon LiDARs, which capture time-of-flight histograms that encode the presence of hidden objects - allowing robots to look around blind corners. Recent single-photon LiDARs have become practical and portable, enabling deployment beyond controlled lab settings. Prior NLOS works target 3D reconstruction in static, lab-based scenarios, and initial efforts toward NLOS-aided navigation consider simplified geometries. We introduce SuperEx, a framework that integrates NLOS sensing directly into the mapping-exploration loop. SuperEx augments global map prediction with beyond-line-of-sight cues by (i) carving empty NLOS regions from timing histograms and (ii) reconstructing occupied structure via a two-step physics-based and data-driven approach that leverages structural regularities. Evaluations on complex simulated maps and the real-world KTH Floorplan dataset show a 12% gain in mapping accuracy under < 30% coverage and improved exploration efficiency compared to line-of-sight baselines, opening a path to reliable mapping beyond direct visibility. 

**Abstract (ZH)**: 非视距感知在未知室内环境中的高效探索与制图是一项基本挑战，尤其是在时间敏感的环境中。当前系统中，机器人感知局限于视距内；被遮挡的区域在未实际穿越前保持未知，导致在布局与先验假设不符时探索效率低下。在本工作中，我们将非视距（NLOS）感知引入机器人探索。我们利用单光子LiDAR，它可以捕获飞行时间直方图，编码隐藏物体的存在——使机器人能够在视线之外进行观察。近期的单光子LiDAR已变得实用且便携，使其能够超越受控实验室环境进行部署。在此之前，NLOS 工作主要针对静态、实验室基于的场景进行3D重构，并且初始的NLOS辅助导航尝试仅考虑简化几何形状。我们引入了SuperEx框架，该框架将NLOS传感直接集成到制图-探索循环中。SuperEx通过(i) 从时间直方图中刻画空闲的NLOS区域，以及(ii) 采用基于物理和数据驱动的两步重建方法来重构占用结构，来增强全局地图预测，该方法利用结构规律性。在复杂模拟地图和实际世界中的KTH平面图数据集上的评估结果显示，在覆盖率小于30%的情况下，地图准确性提高了12%，并且与视距基准相比，探索效率有所提升，为超越直接可视性的可靠制图开辟了路径。 

---
# Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait Transitions at Varying Speeds 

**Title (ZH)**: 面向动态四足 gaits 的研究：基于对称性的 RL 等级结构使四足机器人能够在不同速度下实现自由 gaits 转换 

**Authors**: Jiayu Ding, Xulin Chen, Garrett E. Katz, Zhenyu Gan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10455)  

**Abstract**: Quadrupedal robots exhibit a wide range of viable gaits, but generating specific footfall sequences often requires laborious expert tuning of numerous variables, such as touch-down and lift-off events and holonomic constraints for each leg. This paper presents a unified reinforcement learning framework for generating versatile quadrupedal gaits by leveraging the intrinsic symmetries and velocity-period relationship of dynamic legged systems. We propose a symmetry-guided reward function design that incorporates temporal, morphological, and time-reversal symmetries. By focusing on preserved symmetries and natural dynamics, our approach eliminates the need for predefined trajectories, enabling smooth transitions between diverse locomotion patterns such as trotting, bounding, half-bounding, and galloping. Implemented on the Unitree Go2 robot, our method demonstrates robust performance across a range of speeds in both simulations and hardware tests, significantly improving gait adaptability without extensive reward tuning or explicit foot placement control. This work provides insights into dynamic locomotion strategies and underscores the crucial role of symmetries in robotic gait design. 

**Abstract (ZH)**: 四足机器人表现出广泛的可行步态，但生成特定的踏步序列往往需要专家对众多变量进行繁琐的手动调整，如接触地面和离地事件以及每条腿的动力学约束。本文提出了一种统一的强化学习框架，通过利用动态腿足系统的固有对称性和速度-周期关系来生成多样的四足步态。我们提出了一种基于对称性的奖励函数设计，结合了时间、形态和时间反转对称性。通过关注保留的对称性和自然动力学，我们的方法消除了预先定义轨迹的需要，从而能够在徒步、跃步、半跃步和飞跑等多样运动模式之间实现平滑过渡。在Unitree Go2机器人上实现后，我们的方法在模拟和硬件测试中均表现出稳健的性能，显著提高了步态适应性，而无需进行广泛的奖励调整或显式足部定位控制。本工作为动态运动策略提供了见解，并强调了对称性在机器人步态设计中的关键作用。 

---
# RobotFleet: An Open-Source Framework for Centralized Multi-Robot Task Planning 

**Title (ZH)**: RobotFleet: 一种开源的集中式多机器人任务规划框架 

**Authors**: Rohan Gupta, Trevor Asbery, Zain Merchant, Abrar Anwar, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2510.10379)  

**Abstract**: Coordinating heterogeneous robot fleets to achieve multiple goals is challenging in multi-robot systems. We introduce an open-source and extensible framework for centralized multi-robot task planning and scheduling that leverages LLMs to enable fleets of heterogeneous robots to accomplish multiple tasks. RobotFleet provides abstractions for planning, scheduling, and execution across robots deployed as containerized services to simplify fleet scaling and management. The framework maintains a shared declarative world state and two-way communication for task execution and replanning. By modularizing each layer of the autonomy stack and using LLMs for open-world reasoning, RobotFleet lowers the barrier to building scalable multi-robot systems. The code can be found here: this https URL. 

**Abstract (ZH)**: 基于LLM的协调异构机器人舰队实现多目标的集中式多机器人任务规划与调度框架 

---
# Learning to Throw-Flip 

**Title (ZH)**: 学习投掷翻转 

**Authors**: Yang Liu, Bruno Da Costa, Aude Billard  

**Link**: [PDF](https://arxiv.org/pdf/2510.10357)  

**Abstract**: Dynamic manipulation, such as robot tossing or throwing objects, has recently gained attention as a novel paradigm to speed up logistic operations. However, the focus has predominantly been on the object's landing location, irrespective of its final orientation. In this work, we present a method enabling a robot to accurately "throw-flip" objects to a desired landing pose (position and orientation). Conventionally, objects thrown by revolute robots suffer from parasitic rotation, resulting in highly restricted and uncontrollable landing poses. Our approach is based on two key design choices: first, leveraging the impulse-momentum principle, we design a family of throwing motions that effectively decouple the parasitic rotation, significantly expanding the feasible set of landing poses. Second, we combine a physics-based model of free flight with regression-based learning methods to account for unmodeled effects. Real robot experiments demonstrate that our framework can learn to throw-flip objects to a pose target within ($\pm$5 cm, $\pm$45 degrees) threshold in dozens of trials. Thanks to data assimilation, incorporating projectile dynamics reduces sample complexity by an average of 40% when throw-flipping to unseen poses compared to end-to-end learning methods. Additionally, we show that past knowledge on in-hand object spinning can be effectively reused, accelerating learning by 70% when throwing a new object with a Center of Mass (CoM) shift. A video summarizing the proposed method and the hardware experiments is available at this https URL. 

**Abstract (ZH)**: 基于动态操控的机器人目标抛翻方法 

---
# Rise of the Robochemist 

**Title (ZH)**: 机器人化学家的崛起 

**Authors**: Jihong Zhu, Kefeng Huang, Jonathon Pipe, Chris Horbaczewsky, Andy Tyrrell, Ian J. S. Fairlamb  

**Link**: [PDF](https://arxiv.org/pdf/2510.10337)  

**Abstract**: Chemistry, a long-standing discipline, has historically relied on manual and often time-consuming processes. While some automation exists, the field is now on the cusp of a significant evolution driven by the integration of robotics and artificial intelligence (AI), giving rise to the concept of the robochemist: a new paradigm where autonomous systems assist in designing, executing, and analyzing experiments. Robochemists integrate mobile manipulators, advanced perception, teleoperation, and data-driven protocols to execute experiments with greater adaptability, reproducibility, and safety. Rather than a fully automated replacement for human chemists, we envisioned the robochemist as a complementary partner that works collaboratively to enhance discovery, enabling a more efficient exploration of chemical space and accelerating innovation in pharmaceuticals, materials science, and sustainable manufacturing. This article traces the technologies, applications, and challenges that define this transformation, highlighting both the opportunities and the responsibilities that accompany the emergence of the robochemist. Ultimately, the future of chemistry is argued to lie in a symbiotic partnership where human intuition and expertise is amplified by robotic precision and AI-driven insight. 

**Abstract (ZH)**: 化学，这一历史悠久的学科，历来依赖于手工操作和常常耗时的过程。虽然一些自动化已经存在，但随着机器人技术和人工智能（AI）的融合，该领域正迎来一场显著的变革，催生出了“罗博chemist”这一新的范式：自主系统在设计、执行和分析实验中提供协助。罗博chemists融合了移动 manipulators、高级感知、远程操作以及数据驱动的协议，以更高的适应性、再现性和安全性执行实验。我们设想罗博chemist更像是人类化学家的协同伙伴，共同努力以增强发现，促进化学空间的更高效探索，加速制药、材料科学和可持续制造领域的创新。本文追溯了这一转变中涉及的技术、应用和挑战，凸显了robochemist出现所带来的机遇和伴随而来的责任。最终，化学的未来被认为在于人机共生的伙伴关系，其中人类的直觉和专业知识通过机器人精确操作和AI驱动的洞察得到强化。 

---
# Towards Safe Maneuvering of Double-Ackermann-Steering Robots with a Soft Actor-Critic Framework 

**Title (ZH)**: 基于软Actor- Critic框架的双轴线转向机器人安全机动方法 

**Authors**: Kohio Deflesselle, Mélodie Daniel, Aly Magassouba, Miguel Aranda, Olivier Ly  

**Link**: [PDF](https://arxiv.org/pdf/2510.10332)  

**Abstract**: We present a deep reinforcement learning framework based on Soft Actor-Critic (SAC) for safe and precise maneuvering of double-Ackermann-steering mobile robots (DASMRs). Unlike holonomic or simpler non-holonomic robots such as differential-drive robots, DASMRs face strong kinematic constraints that make classical planners brittle in cluttered environments. Our framework leverages the Hindsight Experience Replay (HER) and the CrossQ overlay to encourage maneuvering efficiency while avoiding obstacles. Simulation results with a heavy four-wheel-steering rover show that the learned policy can robustly reach up to 97% of target positions while avoiding obstacles. Our framework does not rely on handcrafted trajectories or expert demonstrations. 

**Abstract (ZH)**: 基于Soft Actor-Critic (SAC)的安全精准双Ackermann转向移动机器人操作的深度强化学习框架 

---
# X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model 

**Title (ZH)**: X-VLA：可扩展的跨体态视觉-语言-行动模型的软提示变换器 

**Authors**: Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10274)  

**Abstract**: Successful generalist Vision-Language-Action (VLA) models rely on effective training across diverse robotic platforms with large-scale, cross-embodiment, heterogeneous datasets. To facilitate and leverage the heterogeneity in rich, diverse robotic data sources, we propose a novel Soft Prompt approach with minimally added parameters, by infusing prompt learning concepts into cross-embodiment robot learning and introducing separate sets of learnable embeddings for each distinct data source. These embeddings serve as embodiment-specific prompts, which in unity empower VLA models with effective exploitation of varying cross-embodiment features. Our new X-VLA, a neat flow-matching-based VLA architecture, relies exclusively on soft-prompted standard Transformer encoders, enjoying both scalability and simplicity. Evaluated across 6 simulations as well as 3 real-world robots, our 0.9B instantiation-X-VLA-0.9B simultaneously achieves SOTA performance over a sweep of benchmarks, demonstrating superior results on a wide axes of capabilities, from flexible dexterity to quick adaptation across embodiments, environments, and tasks. Website: this https URL 

**Abstract (ZH)**: 成功的通用视觉-语言-动作（VLA）模型依赖于在具有大规模、跨载体和异构数据集的多样化机器人平台上进行有效的训练。为促进和利用丰富多样机器人数据源中的异质性，我们提出了一种新颖的轻量级软提示方法，通过将提示学习概念融入跨载体机器人学习，并为每个独立的数据源引入可学习嵌入，来利用这种异质性。这些嵌入作为载体特异性提示，共同赋予VLA模型有效地利用跨载体特征的能力。我们的新X-VLA是一种基于流匹配的VLA架构，仅依赖于软提示的标准Transformer编码器，兼具可扩展性和简洁性。在6个模拟平台和3个真实机器人上的评估结果表明，我们的0.9B实例X-VLA-0.9B在一系列基准测试中的表现优于现有方法，在灵活灵巧性、快速适应不同载体、环境和任务方面取得了卓越的成绩。网站: [这个链接](这个链接)。 

---
# A3RNN: Bi-directional Fusion of Bottom-up and Top-down Process for Developmental Visual Attention in Robots 

**Title (ZH)**: A3RNN：机器人发展视觉注意的自底向上与自顶向下过程双向融合模型 

**Authors**: Hyogo Hiruma, Hiroshi Ito, Hiroki Mori, Tetsuya Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2510.10221)  

**Abstract**: This study investigates the developmental interaction between top-down (TD) and bottom-up (BU) visual attention in robotic learning. Our goal is to understand how structured, human-like attentional behavior emerges through the mutual adaptation of TD and BU mechanisms over time. To this end, we propose a novel attention model $A^3 RNN$ that integrates predictive TD signals and saliency-based BU cues through a bi-directional attention architecture.
We evaluate our model in robotic manipulation tasks using imitation learning. Experimental results show that attention behaviors evolve throughout training, from saliency-driven exploration to prediction-driven direction. Initially, BU attention highlights visually salient regions, which guide TD processes, while as learning progresses, TD attention stabilizes and begins to reshape what is perceived as salient. This trajectory reflects principles from cognitive science and the free-energy framework, suggesting the importance of self-organizing attention through interaction between perception and internal prediction. Although not explicitly optimized for stability, our model exhibits more coherent and interpretable attention patterns than baselines, supporting the idea that developmental mechanisms contribute to robust attention formation. 

**Abstract (ZH)**: 本研究探讨了机器人学习中自上而下（TD）和自下而上（BU）视觉注意力的发育交互作用。我们的目标是理解通过TD和BU机制的相互适应，如何在时间进程中产生结构化的类人类注意力行为。为此，我们提出了一种新的注意力模型 $A^3 RNN$，该模型通过双向注意力架构整合了预测性TD信号和基于显著性的BU提示。我们使用模仿学习在机器人操作任务中评估了该模型。实验结果表明，注意力行为在训练过程中演化，从基于显著性的探索到基于预测的方向。起初，BU注意力突出显示视觉上显著的区域，引导TD过程，而随着学习的进展，TD注意力趋于稳定并开始重塑被认为显著的内容。这一轨迹反映了认知科学和最小自由能框架的原则，表明通过感知与内部预测之间的交互实现自我组织注意力的重要性。尽管我们的模型未明确优化稳定性，但其表现出比基线模型更加一致和可解释的注意力模式，支持发育机制在增强注意力形成 robust 性中的作用。 

---
# UF-RNN: Real-Time Adaptive Motion Generation Using Uncertainty-Driven Foresight Prediction 

**Title (ZH)**: UF-RNN：基于不确定性前瞻预测的实时自适应运动生成 

**Authors**: Hyogo Hiruma, Hiroshi Ito, Tetsuya Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2510.10217)  

**Abstract**: Training robots to operate effectively in environments with uncertain states, such as ambiguous object properties or unpredictable interactions, remains a longstanding challenge in robotics. Imitation learning methods typically rely on successful examples and often neglect failure scenarios where uncertainty is most pronounced. To address this limitation, we propose the Uncertainty-driven Foresight Recurrent Neural Network (UF-RNN), a model that combines standard time-series prediction with an active "Foresight" module. This module performs internal simulations of multiple future trajectories and refines the hidden state to minimize predicted variance, enabling the model to selectively explore actions under high uncertainty. We evaluate UF-RNN on a door-opening task in both simulation and a real-robot setting, demonstrating that, despite the absence of explicit failure demonstrations, the model exhibits robust adaptation by leveraging self-induced chaotic dynamics in its latent space. When guided by the Foresight module, these chaotic properties stimulate exploratory behaviors precisely when the environment is ambiguous, yielding improved success rates compared to conventional stochastic RNN baselines. These findings suggest that integrating uncertainty-driven foresight into imitation learning pipelines can significantly enhance a robot's ability to handle unpredictable real-world conditions. 

**Abstract (ZH)**: 训练机器人在具有不确定状态的环境（如模糊的对象属性或不可预测的交互）中有效操作仍然是机器人技术中的长期挑战。我们提出了一种不确定性驱动的前瞻性循环神经网络（UF-RNN），该模型结合了标准的时间序列预测与一个主动的“前瞻性”模块。该模块进行多个未来轨迹的内部模拟并细化隐藏状态以最小化预测的不确定性，使模型能够在高度不确定的情况下选择性地探索动作。我们在门打开任务上的模拟和真实机器人设置中评估了UF-RNN，结果表明，在缺少明确失败示范的情况下，模型通过利用其潜在空间中的自我诱导混沌动力学表现出鲁棒的适应性。当由前瞻性模块引导时，这些混沌特性能够刺激在环境不确定性高时的探索行为，从而提高成功率，优于传统的随机RNN基线。这些发现表明，在模仿学习管道中集成不确定性驱动的前瞻性可以显著提高机器人处理不可预测的现实条件的能力。 

---
# It Takes Two: Learning Interactive Whole-Body Control Between Humanoid Robots 

**Title (ZH)**: 需要双方：学习人形机器人之间的互动全身控制 

**Authors**: Zuhong Liu, Junhao Ge, Minhao Xiong, Jiahao Gu, Bowei Tang, Wei Jing, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10206)  

**Abstract**: The true promise of humanoid robotics lies beyond single-agent autonomy: two or more humanoids must engage in physically grounded, socially meaningful whole-body interactions that echo the richness of human social interaction. However, single-humanoid methods suffer from the isolation issue, ignoring inter-agent dynamics and causing misaligned contacts, interpenetrations, and unrealistic motions. To address this, we present Harmanoid , a dual-humanoid motion imitation framework that transfers interacting human motions to two robots while preserving both kinematic fidelity and physical realism. Harmanoid comprises two key components: (i) contact-aware motion retargeting, which restores inter-body coordination by aligning SMPL contacts with robot vertices, and (ii) interaction-driven motion controller, which leverages interaction-specific rewards to enforce coordinated keypoints and physically plausible contacts. By explicitly modeling inter-agent contacts and interaction-aware dynamics, Harmanoid captures the coupled behaviors between humanoids that single-humanoid frameworks inherently overlook. Experiments demonstrate that Harmanoid significantly improves interactive motion imitation, surpassing existing single-humanoid frameworks that largely fail in such scenarios. 

**Abstract (ZH)**: 人形机器人的真实潜力超越单一自主 agent：多个机器人需进行物理接地和社会意义兼具的全身交互，以模仿人类社会互动的丰富性。然而，单一机器人方法存在隔离问题，忽视了多 agent 动态，导致接触不准确、相互穿插和不现实的运动。为解决这一问题，我们提出了 Harmanoid 这一双人形机器人运动模仿框架，该框架在保留运动学保真度和物理现实性的基础上，将交互人类动作转移到两台机器人上。Harmanoid 包含两个关键组件：（i）接触感知运动重定位，通过将 SMPL 接触与机器人顶点对齐来恢复身体间协调；（ii）交互驱动运动控制器，利用特定交互奖励来确保协调的关键点和物理上合理的接触。通过明确建模多 agent 接触和交互驱动动力学，Harmanoid 捕捉到单一机器人框架固有的耦合行为。实验表明，Harmanoid 显著提高了交互运动模仿的效果，超过了在这些场景中表现不佳的现有单一机器人框架。 

---
# Dejavu: Post-Deployment Learning for Embodied Agents via Experience Feedback 

**Title (ZH)**: Dejavu: 通过经验反馈的部署后学习 for 体现代理 

**Authors**: Shaokai Wu, Yanbiao Ji, Qiuchang Li, Zhiyi Zhang, Qichen He, Wenyuan Xie, Guodong Zhang, Bayram Bayramli, Yue Ding, Hongtao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10181)  

**Abstract**: Embodied agents face a fundamental limitation: once deployed in real-world environments to perform specific tasks, they are unable to acquire new useful knowledge to enhance task performance. In this paper, we propose a general post-deployment learning framework called Dejavu, which employs an Experience Feedback Network (EFN) and augments the frozen Vision-Language-Action (VLA) policy with retrieved execution memories. EFN automatically identifies contextually successful prior action experiences and conditions action prediction on this retrieved guidance. We adopt reinforcement learning with semantic similarity rewards on EFN to ensure that the predicted actions align with past successful behaviors under current observations. During deployment, EFN continually enriches its memory with new trajectories, enabling the agent to exhibit "learning from experience" despite fixed weights. Experiments across diverse embodied tasks show that EFN significantly improves adaptability, robustness, and success rates over frozen baselines. These results highlight a promising path toward embodied agents that continually refine their behavior after deployment. 

**Abstract (ZH)**: Embodied代理面临一个根本性限制：一旦部署到实际环境执行特定任务，它们无法获取新的有用知识以增强任务性能。本文提出了一种通用的后部署学习框架Dejavu，该框架利用经验反馈网络（EFN），并结合检索到的执行记忆来增强冻结的视觉-语言-行动（VLA）策略。EFN自动识别上下文成功的历史行动经验，并以检索到的指导为条件进行行动预测。我们采用基于语义相似度奖励的强化学习方法确保EFN预测的动作与当前观测下的历史成功行为一致。在部署过程中，EFN持续丰富其记忆，使代理能够在固定权重的情况下表现出“从经验中学习”的能力。跨多种embodied任务的实验结果表明，EFN显着提高了适应性、鲁棒性和成功率，相比于冻结基线具有显著优势。这些结果突显了朝向部署后不断自我优化的embodied代理的有希望途径。 

---
# CompassNav: Steering From Path Imitation To Decision Understanding In Navigation 

**Title (ZH)**: CompassNav: 从路径模仿到导航决策理解的方向导航 

**Authors**: LinFeng Li, Jian Zhao, Yuan Xie, Xin Tan, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.10154)  

**Abstract**: The dominant paradigm for training Large Vision-Language Models (LVLMs) in navigation relies on imitating expert trajectories. This approach reduces the complex navigation task to a sequence-to-sequence replication of a single correct path, fundamentally limiting the agent's ability to explore and generalize. In this work, we argue for and introduce a new paradigm: a shift from Path Imitation to Decision Understanding. The goal of this paradigm is to build agents that do not just follow, but truly understand how to navigate. We materialize this through two core contributions: first, we introduce Compass-Data-22k, a novel 22k-trajectory this http URL Reinforcement Fine-Tuning (RFT) subset provides a panoramic view of the decision landscape by annotating all feasible actions with A* geodesic distances. Second, we design a novel gap-aware hybrid reward function that dynamically adapts its feedback to decision certainty, shifting between decisive signals for optimal actions and nuanced scores to encourage exploration. Integrated into an SFT-then-RFT recipe, our CompassNav agent is trained not to memorize static routes, but to develop an internal ``compass'' that constantly intuits the direction to the goal by evaluating the relative quality of all possible moves. This approach enables our 7B agent to set a new state-of-the-art on Goal navigation benchmarks, outperforming even larger proprietary models, and achieve robust real-world goal navigation on a physical robot. 

**Abstract (ZH)**: 训练大型视觉-语言模型（LVLMs）在导航中的主导范式依赖于模仿专家轨迹。这种方法将复杂的导航任务简化为单个正确路径的序列复制，从根本上限制了智能体的探索和泛化能力。本文我们提出并引入了一个新的范式：从路径模仿转向决策理解。这一范式的目标是构建不仅能跟随，而且真正理解如何导航的智能体。我们通过两个核心贡献实现这一点：首先，我们引入了Compass-Data-22k，一个包含22k轨迹的数据集，其中Reinforcement Fine-Tuning（RFT）子集通过标注所有可行动作的A*地理距离，提供了决策环境的全景视图。其次，我们设计了一种新的感知差距的混合奖励函数，这种奖励函数动态适应决策的确定性，从为最优动作提供决断信号到提供细微的评分鼓励探索。将这些贡献集成到SFT-then-RFT方案中，我们的CompassNav智能体被训练成不是仅仅记忆固定的路线，而是开发一种内部的“指南针”，不断通过评估所有可能动作的质量来推断目标的方向。这种方法使我们的7B智能体在目标导航基准测试中取得了新的最前沿成绩，甚至超过了更大的专有模型，并在物理机器人上实现了稳健的目标导航。 

---
# Ctrl-World: A Controllable Generative World Model for Robot Manipulation 

**Title (ZH)**: Ctrl-World: 一种可控制生成的世界模型用于机器人操作 

**Authors**: Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2510.10125)  

**Abstract**: Generalist robot policies can now perform a wide range of manipulation skills, but evaluating and improving their ability with unfamiliar objects and instructions remains a significant challenge. Rigorous evaluation requires a large number of real-world rollouts, while systematic improvement demands additional corrective data with expert labels. Both of these processes are slow, costly, and difficult to scale. World models offer a promising, scalable alternative by enabling policies to rollout within imagination space. However, a key challenge is building a controllable world model that can handle multi-step interactions with generalist robot policies. This requires a world model compatible with modern generalist policies by supporting multi-view prediction, fine-grained action control, and consistent long-horizon interactions, which is not achieved by previous works. In this paper, we make a step forward by introducing a controllable multi-view world model that can be used to evaluate and improve the instruction-following ability of generalist robot policies. Our model maintains long-horizon consistency with a pose-conditioned memory retrieval mechanism and achieves precise action control through frame-level action conditioning. Trained on the DROID dataset (95k trajectories, 564 scenes), our model generates spatially and temporally consistent trajectories under novel scenarios and new camera placements for over 20 seconds. We show that our method can accurately rank policy performance without real-world robot rollouts. Moreover, by synthesizing successful trajectories in imagination and using them for supervised fine-tuning, our approach can improve policy success by 44.7\%. 

**Abstract (ZH)**: 通用机器人策略现在可以执行广泛的操纵技能，但评估和在不熟悉的对象和指令下提升其能力仍是一项重大挑战。严格的评估需要大量的真实世界滚动测试，而系统的改进则需要额外的带有专家标记的纠正数据。这两个过程都缓慢、昂贵且难以扩展。世界模型提供了一种有前景的、可扩展的替代方案，通过使策略在想象空间内滚动来实现这一点。然而，关键挑战是构建一个可控的世界模型，可以处理通用机器人策略的多步交互。这需要一种与现代通用策略相兼容的世界模型，支持多视图预测、精细的动作控制和一致的长时交互，而这些目标此前的研究并未实现。在这篇论文中，我们通过引入一个可控的多视图世界模型向前迈出一步，该模型可以用于评估和提高通用机器人策略遵循指令的能力。我们的模型通过姿态条件化的记忆检索机制保持长时间的一致性，并通过帧级动作条件实现精确的动作控制。在DROID数据集（包含95,000条轨迹、564个场景）上训练后，我们的模型能够在新的场景和新的摄像机架设方式下生成超过20秒的空间和时间上一致的轨迹。我们展示了我们的方法可以在不需要真实世界机器人滚动测试的情况下准确排名策略性能。此外，通过在想象中合成成功的轨迹并用于监督微调，我们的方法可以将策略成功率提高44.7%。 

---
# ATRos: Learning Energy-Efficient Agile Locomotion for Wheeled-legged Robots 

**Title (ZH)**: ATRos: 学习高效灵巧运动的轮足机器人能耗优化方法 

**Authors**: Jingyuan Sun, Hongyu Ji, Zihan Qu, Chaoran Wang, Mingyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09980)  

**Abstract**: Hybrid locomotion of wheeled-legged robots has recently attracted increasing attention due to their advantages of combining the agility of legged locomotion and the efficiency of wheeled motion. But along with expanded performance, the whole-body control of wheeled-legged robots remains challenging for hybrid locomotion. In this paper, we present ATRos, a reinforcement learning (RL)-based hybrid locomotion framework to achieve hybrid walking-driving motions on the wheeled-legged robot. Without giving predefined gait patterns, our planner aims to intelligently coordinate simultaneous wheel and leg movements, thereby achieving improved terrain adaptability and improved energy efficiency. Based on RL techniques, our approach constructs a prediction policy network that could estimate external environmental states from proprioceptive sensory information, and the outputs are then fed into an actor critic network to produce optimal joint commands. The feasibility of the proposed framework is validated through both simulations and real-world experiments across diverse terrains, including flat ground, stairs, and grassy surfaces. The hybrid locomotion framework shows robust performance over various unseen terrains, highlighting its generalization capability. 

**Abstract (ZH)**: 基于强化学习的轮腿机器人混合运动框架 

---
# LLM-HBT: Dynamic Behavior Tree Construction for Adaptive Coordination in Heterogeneous Robots 

**Title (ZH)**: LLM-HBT：异构机器人自适应协调的动态行为树构建 

**Authors**: Chaoran Wang, Jingyuan Sun, Yanhui Zhang, Mingyu Zhang, Changju Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09963)  

**Abstract**: We introduce a novel framework for automatic behavior tree (BT) construction in heterogeneous multi-robot systems, designed to address the challenges of adaptability and robustness in dynamic environments. Traditional robots are limited by fixed functional attributes and cannot efficiently reconfigure their strategies in response to task failures or environmental changes. To overcome this limitation, we leverage large language models (LLMs) to generate and extend BTs dynamically, combining the reasoning and generalization power of LLMs with the modularity and recovery capability of BTs. The proposed framework consists of four interconnected modules task initialization, task assignment, BT update, and failure node detection which operate in a closed loop. Robots tick their BTs during execution, and upon encountering a failure node, they can either extend the tree locally or invoke a centralized virtual coordinator (Alex) to reassign subtasks and synchronize BTs across peers. This design enables long-term cooperative execution in heterogeneous teams. We validate the framework on 60 tasks across three simulated scenarios and in a real-world cafe environment with a robotic arm and a wheeled-legged robot. Results show that our method consistently outperforms baseline approaches in task success rate, robustness, and scalability, demonstrating its effectiveness for multi-robot collaboration in complex scenarios. 

**Abstract (ZH)**: 一种用于异构多机器人系统的自动生成行为树的新框架：克服动态环境下的适应性和鲁棒性挑战 

---
# Enhancing Diffusion Policy with Classifier-Free Guidance for Temporal Robotic Tasks 

**Title (ZH)**: 增强基于分类器-free 指导的扩散策略以应对时间敏感的机器人任务 

**Authors**: Yuang Lu, Song Wang, Xiao Han, Xuri Zhang, Yucong Wu, Zhicheng He  

**Link**: [PDF](https://arxiv.org/pdf/2510.09786)  

**Abstract**: Temporal sequential tasks challenge humanoid robots, as existing Diffusion Policy (DP) and Action Chunking with Transformers (ACT) methods often lack temporal context, resulting in local optima traps and excessive repetitive actions. To address these issues, this paper introduces a Classifier-Free Guidance-Based Diffusion Policy (CFG-DP), a novel framework to enhance DP by integrating Classifier-Free Guidance (CFG) with conditional and unconditional models. Specifically, CFG leverages timestep inputs to track task progression and ensure precise cycle termination. It dynamically adjusts action predictions based on task phase, using a guidance factor tuned to balance temporal coherence and action accuracy. Real-world experiments on a humanoid robot demonstrate high success rates and minimal repetitive actions. Furthermore, we assessed the model's ability to terminate actions and examined how different components and parameter adjustments affect its performance. This framework significantly enhances deterministic control and execution reliability for sequential robotic tasks. 

**Abstract (ZH)**: 基于分类器-free 指导的扩散策略：增强类人机器人序列任务的执行 

---
# REACT3D: Recovering Articulations for Interactive Physical 3D Scenes 

**Title (ZH)**: REACT3D: 恢复交互物理三维场景中的articulations 

**Authors**: Zhao Huang, Boyang Sun, Alexandros Delitzas, Jiaqi Chen, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2510.11340)  

**Abstract**: Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks. Our contributions include: (i) openable-object detection and segmentation to extract candidate movable parts from static scenes, (ii) articulation estimation that infers joint types and motion parameters, (iii) hidden-geometry completion followed by interactive object assembly, and (iv) interactive scene integration in widely supported formats to ensure compatibility with standard simulation platforms. We achieve state-of-the-art performance on detection/segmentation and articulation metrics across diverse indoor scenes, demonstrating the effectiveness of our framework and providing a practical foundation for scalable interactive scene generation, thereby lowering the barrier to large-scale research on articulated scene understanding. Our project page is \textit{\hypersetup{urlcolor=black}\href{this https URL}{this http URL}}. 

**Abstract (ZH)**: 交互式3D场景对于嵌入式智能至关重要，但由于现有数据集受限于标注部分分割、运动类型和运动轨迹的劳动密集型过程。我们提出了REACT3D，这是一种可扩展的零样本框架，能够将静态3D场景转换为可用于模拟的交互式复制品，具有一致的几何形状，使得可以直接应用于各种下游任务。我们的贡献包括：(i) 可开启对象的检测与分割以从静态场景中提取潜在可移动部分，(ii) 运动学估计以推断关节类型和运动参数，(iii) 隐藏几何补全跟随交互式对象装配，以及(iv) 在广泛支持的格式下实现交互式场景集成以确保与标准模拟平台的兼容性。我们在跨多种室内场景的检测/分割和运动学指标上达到了最先进的性能，展示了我们框架的有效性，并为可扩展的交互式场景生成提供了实用基础，从而降低了大规模研究 articulated 场景理解的障碍。我们的项目页面为 \textit{\href{this https URL}{this http URL}}。 

---
# TabVLA: Targeted Backdoor Attacks on Vision-Language-Action Models 

**Title (ZH)**: TabVLA: 面向视觉-语言-行动模型的 targeted 后门攻击 

**Authors**: Zonghuan Xu, Xiang Zheng, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10932)  

**Abstract**: With the growing deployment of Vision-Language-Action (VLA) models in real-world embodied AI systems, their increasing vulnerability to backdoor attacks poses a serious safety threat. A backdoored VLA agent can be covertly triggered by a pre-injected backdoor to execute adversarial actions, potentially causing system failures or even physical harm. Although backdoor attacks on VLA models have been explored, prior work has focused only on untargeted attacks, leaving the more practically threatening scenario of targeted manipulation unexamined. In this paper, we study targeted backdoor attacks on VLA models and introduce TabVLA, a novel framework that enables such attacks via black-box fine-tuning. TabVLA explores two deployment-relevant inference-time threat models: input-stream editing and in-scene triggering. It formulates poisoned data generation as an optimization problem to improve attack effectivess. Experiments with OpenVLA-7B on the LIBERO benchmark reveal that the vision channel is the principal attack surface: targeted backdoors succeed with minimal poisoning, remain robust across variations in trigger design, and are degraded only by positional mismatches between fine-tuning and inference triggers. We also investigate a potential detection-based defense against TabVLA, which reconstructs latent visual triggers from the input stream to flag activation-conditioned backdoor samples. Our work highlights the vulnerability of VLA models to targeted backdoor manipulation and underscores the need for more advanced defenses. 

**Abstract (ZH)**: 随着视觉-语言-动作（VLA）模型在实际应用中嵌入式AI系统的部署增加，它们日益增长的后门攻击脆弱性对安全性构成了严重威胁。一个植入后门的VLA代理可能通过预先注入的后门被秘密触发执行敌意操作，可能会导致系统故障甚至物理伤害。尽管已经探讨了VLA模型的后门攻击，但先前的研究仅关注无目标攻击，而没有研究更具实际威胁的目标操纵场景。本文研究了VLA模型的目标后门攻击，并引入了TabVLA这一新框架，通过黑盒微调使这些攻击成为可能。TabVLA探讨了与部署相关的两种推理时威胁模型：输入流编辑和场景触发。它将有毒数据生成建模为优化问题以提高攻击效果。在LIBERO基准上使用OpenVLA-7B进行的实验揭示，视觉通道是主要的攻击面：目标后门在极小规模的中毒下就能成功，对触发设计的变化保持稳健，并且只有在微调和推理触发之间的位置不匹配时才会受到损害。我们还研究了对TabVLA潜在的基于检测的防御措施，该措施从输入流中重建潜在的视觉触发以标记条件激活的后门样本。我们的工作突显了VLA模型对目标后门操纵的脆弱性，并强调了需要更高级防御的必要性。 

---
# The Irrational Machine: Neurosis and the Limits of Algorithmic Safety 

**Title (ZH)**: 非理性的机器：神经症与算法安全性极限 

**Authors**: Daniel Howard  

**Link**: [PDF](https://arxiv.org/pdf/2510.10823)  

**Abstract**: We present a framework for characterizing neurosis in embodied AI: behaviors that are internally coherent yet misaligned with reality, arising from interactions among planning, uncertainty handling, and aversive memory. In a grid navigation stack we catalogue recurrent modalities including flip-flop, plan churn, perseveration loops, paralysis and hypervigilance, futile search, belief incoherence, tie break thrashing, corridor thrashing, optimality compulsion, metric mismatch, policy oscillation, and limited-visibility variants. For each we give lightweight online detectors and reusable escape policies (short commitments, a margin to switch, smoothing, principled arbitration). We then show that durable phobic avoidance can persist even under full visibility when learned aversive costs dominate local choice, producing long detours despite globally safe routes. Using First/Second/Third Law as engineering shorthand for safety latency, command compliance, and resource efficiency, we argue that local fixes are insufficient; global failures can remain. To surface them, we propose genetic-programming based destructive testing that evolves worlds and perturbations to maximize law pressure and neurosis scores, yielding adversarial curricula and counterfactual traces that expose where architectural revision, not merely symptom-level patches, is required. 

**Abstract (ZH)**: 我们提出了一种表征具身人工智能神经质的框架：内部一致但与现实不一致的行为，源于规划、不确定性处理和厌恶记忆之间的交互。在网格导航堆栈中，我们列出了重复出现的模态，包括 flip-flop、计划翻滚、执着循环、瘫痪、高度警觉、无意义搜索、信念不一致、犹豫不决、走廊翻滚、优化强迫、度量匹配不一致、政策振荡以及有限视距变体。对于每种模态，我们提供了轻量级的在线检测器和可重用的逃避策略（短暂的承诺、切换的余量、平滑处理、基于原理的仲裁）。随后，我们展示即使在完全可视情况下，如果学习到的厌恶成本主导局部选择时，持久的恐惧回避仍可能持续，导致尽管全局路线安全却出现长距离绕行。使用First/Second/Third Law作为工程简写，代表安全延迟、命令合规性和资源效率，我们认为局部修复是不够的；全局失败可能仍然存在。为了揭示这些失败，我们提出了基于遗传编程的破坏性测试，通过进化世界和扰动来最大化法则压力和神经质分数，产生对抗性的课程和假设检验轨迹，以显示需要对架构进行修订，而不仅仅是症状级别的修补。 

---
# AI-Agents for Culturally Diverse Online Higher Education Environments 

**Title (ZH)**: AI代理在多元文化在线高等教育环境中的应用 

**Authors**: Fuze Sun, Paul Craig, Lingyu Li, Shixiangyue Meng, Chuxi Nan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10520)  

**Abstract**: As the global reach of online higher education continues to grow, universities are increasingly accommodating students from diverse cultural backgrounds \parencite{tereshko2024culturally}. This can present a number of challenges including linguistic barriers \parencite{ullah2021linguistic}, cultural differences in learning style \parencite{omidvar2012cultural}, cultural sensitivity in course design \parencite{nguyen2022cultural} and perceived isolation when students feel their perspectives or experiences are not reflected or valued in the learning environment \parencite{hansen2022belonging}. Ensuring active engagement and reasonable learning outcomes in such a environments requires distance educational systems that are not only adaptive but also culturally resonant \parencite{dalle2024cultural}. Both embodied and virtual AI-Agents have great potential in this regard as they can facilitate personalized learning and adapt their interactions and content delivery to align with students' cultural context. In addition Generative AI (GAI), such as, Large Language Models (LLMs) can amplify the potential for these culturally aware AI agents to address educational challenges due to their advanced capacity for understanding and generating contextually relevant content \parencite{wang2024large}. This chapter reviews existing research and suggests the usage of culturally aware AI-Agents, powered by GAI, to foster engagement and improve learning outcomes in culturally diverse online higher education environments. 

**Abstract (ZH)**: 在线高等教育全球拓展背景下文化敏感的AI代理在促进多元文化学习环境中的参与和提高学习成果中的应用 

---
# KG-MAS: Knowledge Graph-Enhanced Multi-Agent Infrastructure for coupling physical and digital robotic environments 

**Title (ZH)**: KG-MAS: 知识图谱增强的多智能体基础设施，用于耦合物理和数字机器人环境 

**Authors**: Walid Abdela  

**Link**: [PDF](https://arxiv.org/pdf/2510.10325)  

**Abstract**: The seamless integration of physical and digital environments in Cyber-Physical Systems(CPS), particularly within Industry 4.0, presents significant challenges stemming from system heterogeneity and complexity. Traditional approaches often rely on rigid, data-centric solutions like co-simulation frameworks or brittle point-to-point middleware bridges, which lack the semantic richness and flexibility required for intelligent, autonomous coordination. This report introduces the Knowledge Graph-Enhanced Multi-Agent Infrastructure(KG-MAS), as resolution in addressing such limitations. KG-MAS leverages a centralized Knowledge Graph (KG) as a dynamic, shared world model, providing a common semantic foundation for a Multi-Agent System(MAS). Autonomous agents, representing both physical and digital components, query this KG for decision-making and update it with real-time state information. The infrastructure features a model-driven architecture which facilitates the automatic generation of agents from semantic descriptions, thereby simplifying system extension and maintenance. By abstracting away underlying communication protocols and providing a unified, intelligent coordination mechanism, KG-MAS offers a robust, scalable, and flexible solution for coupling heterogeneous physical and digital robotic environments. 

**Abstract (ZH)**: Cyber-Physical系统(CPS)中物理与数字环境的无缝集成，特别是在工业4.0中，由于系统异构性和复杂性带来了显著挑战。传统方法往往依赖于像协同仿真框架或脆弱的点对点中间件桥接这样的刚性、数据为中心的解决方案，这些方法缺乏智能自主协调所需的语义丰富性和灵活性。本报告介绍了知识图谱增强的多代理基础设施(KG-MAS)，以解决这些局限性。KG-MAS利用中央化的知识图谱(KG)作为动态的共享世界模型，为多代理系统(MAS)提供一个共同的语义基础。自主代理，代表物理和数字组件，通过查询KG进行决策并实时更新其状态信息。该基础设施采用模型驱动架构，支持从语义描述自动生成代理，从而简化系统的扩展和维护。通过抽象底层通信协议并提供统一的智能协调机制，KG-MAS为异构物理和数字机器人环境的互联提供了稳健、可扩展和灵活的解决方案。 

---
# Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models 

**Title (ZH)**: 流匹配策略的强化微调用于视觉-语言-行动模型 

**Authors**: Mingyang Lyu, Yinqian Sun, Erliang Lin, Huangrui Li, Ruolin Chen, Feifei Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09976)  

**Abstract**: Vision-Language-Action (VLA) models such as OpenVLA, Octo, and $\pi_0$ have shown strong generalization by leveraging large-scale demonstrations, yet their performance is still fundamentally constrained by the quality and coverage of supervised data. Reinforcement learning (RL) provides a promising path for improving and fine-tuning VLAs through online interaction. However, conventional policy gradient methods are computationally infeasible in the context of flow-matching based models due to the intractability of the importance sampling process, which requires explicit computation of policy ratios. To overcome this limitation, we propose Flow Policy Optimization (FPO) algorithm, which reformulates importance sampling by leveraging per-sample changes in the conditional flow-matching objective. Furthermore, FPO achieves stable and scalable online reinforcement fine-tuning of the $\pi_0$ model by integrating structure-aware credit assignment to enhance gradient efficiency, clipped surrogate objectives to stabilize optimization, multi-step latent exploration to encourage diverse policy updates, and a Q-ensemble mechanism to provide robust value estimation. We evaluate FPO on the LIBERO benchmark and the ALOHA simulation task against supervised, preference-aligned, diffusion-based, autoregressive online RL, and $\pi_0$-FAST baselines, observing consistent improvements over the imitation prior and strong alternatives with stable learning under sparse rewards. In addition, ablation studies and analyses of the latent space dynamics further highlight the contributions of individual components within FPO, validating the effectiveness of the proposed computational modules and the stable convergence of the conditional flow-matching objective during online RL. 

**Abstract (ZH)**: 基于流匹配的Vision-Language-Action (VLA)模型OpenVLA、Octo和$\pi_0$通过大规模演示展示了强大的泛化能力，但其性能仍然受到监督数据质量及其覆盖范围的限制。强化学习(RL)为通过在线交互改进和微调VLAs提供了有希望的途径。然而，由于基于流匹配模型的重要性采样过程不可计算，传统的策略梯度方法在计算上是不可能实现的。为克服这一限制，我们提出了基于流的策略优化(FPO)算法，该算法通过利用条件流匹配目标的单样本变化重新表述重要性采样。此外，FPO通过结构感知的信用分配增强梯度效率，通过裁剪替代目标稳定优化，通过多步潜在探索鼓励多样性的策略更新，并通过Q-ensemble机制提供稳健的价值估计，实现了$\pi_0$模型的稳定和可扩展的在线强化学习微调。我们使用LIBERO基准和ALOHA仿真任务评估了FPO，观察到在模仿先验和强替代方法中，FPO在稀疏奖励下的稳定学习中的一致改进。此外，消融研究和潜在空间动力学的分析进一步强调了FPO中各个组件的贡献，验证了所提出计算模块的有效性和条件流匹配目标在线RL期间的稳定收敛。 

---
# OmniSAT: Compact Action Token, Faster Auto Regression 

**Title (ZH)**: 全方位SAT模型：紧凑的动作令牌，更快的自动回归 

**Authors**: Huaihai Lyu, Chaofan Chen, Senwei Xie, Pengwei Wang, Xiansheng Chen, Shanghang Zhang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09667)  

**Abstract**: Existing Vision-Language-Action (VLA) models can be broadly categorized into diffusion-based and auto-regressive (AR) approaches: diffusion models capture continuous action distributions but rely on computationally heavy iterative denoising. In contrast, AR models enable efficient optimization and flexible sequence construction, making them better suited for large-scale pretraining. To further improve AR efficiency, particularly when action chunks induce extended and high-dimensional sequences, prior work applies entropy-guided and token-frequency techniques to shorten the sequence length. However, such compression struggled with \textit{poor reconstruction or inefficient compression}. Motivated by this, we introduce an Omni Swift Action Tokenizer, which learns a compact, transferable action representation. Specifically, we first normalize value ranges and temporal horizons to obtain a consistent representation with B-Spline encoding. Then, we apply multi-stage residual quantization to the position, rotation, and gripper subspaces, producing compressed discrete tokens with coarse-to-fine granularity for each part. After pre-training on the large-scale dataset Droid, the resulting discrete tokenization shortens the training sequence by 6.8$\times$, and lowers the target entropy. To further explore the potential of OmniSAT, we develop a cross-embodiment learning strategy that builds on the unified action-pattern space and jointly leverages robot and human demonstrations. It enables scalable auxiliary supervision from heterogeneous egocentric videos. Across diverse real-robot and simulation experiments, OmniSAT encompasses higher compression while preserving reconstruction quality, enabling faster AR training convergence and model performance. 

**Abstract (ZH)**: 现有的视觉-语言-动作（VLA）模型可以大致分为基于扩散和平视自回归（AR）方法：基于扩散的模型可以捕捉连续的动作分布，但需要计算密集的去噪迭代。相比之下，AR模型能够实现高效的优化和灵活的序列构建，使它们更适合大规模预训练。为了进一步提高AR的效率，特别是在动作片段导致长且高维序列时，之前的工作应用了熵引导和token频率技术来缩短序列长度。然而，这种压缩方法在重建质量差或压缩效率低方面存在问题。受此启发，我们引入了一种全方位快速动作Tokenizer（Omni Swift Action Tokenizer），它学习到一种紧凑且可迁移的动作表示。具体来说，我们首先对值范围和时间跨度进行归一化，以获得与B-Spline编码一致的表示。然后，我们在位置、旋转和夹爪子空间应用多阶段残差量化，为每个部分生成具有粗细粒度的压缩离散token。在大型数据集Droid上进行预训练后，生成的离散token化将训练序列缩短了6.8倍，并降低了目标熵。为了进一步探索OmniSAT的潜力，我们开发了一种跨化身学习策略，它建立在一个统一的动作模式空间上，并共同利用机器人和人类的演示。它能够从异质的主观视频中实现可扩展的辅助监督。在多样化的机器人实验和仿真实验中，OmniSAT在保持重建质量的同时实现了更高的压缩比，这促进了AR训练收敛速度，并提高了模型性能。 

---
# Evolution in Simulation: AI-Agent School with Dual Memory for High-Fidelity Educational Dynamics 

**Title (ZH)**: 模拟中的进化：具有双记忆的AI代理学校，用于高保真教育动态 

**Authors**: Sheng Jin, Haoming Wang, Zhiqi Gao, Yongbo Yang, Bao Chunjia, Chengliang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11290)  

**Abstract**: Large language models (LLMs) based Agents are increasingly pivotal in simulating and understanding complex human systems and interactions. We propose the AI-Agent School (AAS) system, built around a self-evolving mechanism that leverages agents for simulating complex educational dynamics. Addressing the fragmented issues in teaching process modeling and the limitations of agents performance in simulating diverse educational participants, AAS constructs the Zero-Exp strategy, employs a continuous "experience-reflection-optimization" cycle, grounded in a dual memory base comprising experience and knowledge bases and incorporating short-term and long-term memory components. Through this mechanism, agents autonomously evolve via situated interactions within diverse simulated school scenarios. This evolution enables agents to more accurately model the nuanced, multi-faceted teacher-student engagements and underlying learning processes found in physical schools. Experiment confirms that AAS can effectively simulate intricate educational dynamics and is effective in fostering advanced agent cognitive abilities, providing a foundational stepping stone from the "Era of Experience" to the "Era of Simulation" by generating high-fidelity behavioral and interaction data. 

**Abstract (ZH)**: 基于大语言模型的智能代理在学校教育动态模拟中的应用：AI-Agent School系统及其零经验策略 

---
# PADME: Procedure Aware DynaMic Execution 

**Title (ZH)**: PADME： Procedure Awareness Dynamic Execution 

**Authors**: Deepeka Garg, Sihan Zeng, Annapoorani L. Narayanan, Sumitra Ganesh, Leo Ardon  

**Link**: [PDF](https://arxiv.org/pdf/2510.11281)  

**Abstract**: Learning to autonomously execute long-horizon procedures from natural language remains a core challenge for intelligent agents. Free-form instructions such as recipes, scientific protocols, or business workflows encode rich procedural knowledge, but their variability and lack of structure cause agents driven by large language models (LLMs) to drift or fail during execution. We introduce Procedure Aware DynaMic Execution (PADME), an agent framework that produces and exploits a graph-based representation of procedures. Unlike prior work that relies on manual graph construction or unstructured reasoning, PADME autonomously transforms procedural text into executable graphs that capture task dependencies, decision points, and reusable subroutines. Central to PADME is a two-phase methodology; Teach phase, which focuses on systematic structuring, enrichment with executable logic of procedures, followed by Execute phase, which enables dynamic execution in response to real-time inputs and environment feedback. This separation ensures quality assurance and scalability, allowing expert knowledge to be encoded once and reliably reused across varying contexts. The graph representation also provides an inductive bias that reduces error accumulation in long-horizon reasoning, underscoring the importance of structured procedure modeling for reliable agent-driven automation. Empirically, PADME achieves state-of-the-art performance on four diverse benchmarks, including ALFWorld and ScienceWorld. These results demonstrate that agents equipped with graph-based procedure representations offer a powerful intermediate abstraction for robust and generalizable execution. 

**Abstract (ZH)**: 基于图表示的程序意识动态执行（PADME）：从自然语言自主执行长期任务 

---
# Video-STR: Reinforcing MLLMs in Video Spatio-Temporal Reasoning with Relation Graph 

**Title (ZH)**: Video-STR：通过关系图增强视频时空推理的MLLMs 

**Authors**: Wentao Wang, Heqing Zou, Tianze Luo, Rui Huang, Yutian Zhao, Zhuochen Wang, Hansheng Zhang, Chengwei Qin, Yan Wang, Lin Zhao, Huaijian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10976)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has demonstrated strong semantic understanding capabilities, but struggles to perform precise spatio-temporal understanding. Existing spatio-temporal methods primarily focus on the video itself, while overlooking the physical information within the video, such as multi-object layouts and motion. Such limitations restrict the use of MLLMs in downstream applications that demand high precision, including embodied intelligence and VR. To address this issue, we present Video-STR, a novel graph-based reinforcement method for precise Video Spatio-Temporal Reasoning. Building upon the capacity of Reinforcement Learning with Verifiable Reward (RLVR) to improve model abilities, we introduce a reasoning mechanism using graph-based Group Relative Policy Optimization (GRPO) method to guide the model in inferring the underlying spatio-temporal topology of scenarios during the thinking process. To resolve the lack of spatio-temporal training data, we construct the STV-205k dataset with 205k question-answering pairs, covering dynamic multi-object scenes in both indoor and outdoor environments, to support the model training. Experiments show that Video-STR achieves state-of-the-art results on various benchmarks, outperforming the base model by 13% on STI-Bench, and demonstrating the effectiveness of our approach and dataset. Code, model, and data will be released. 

**Abstract (ZH)**: Recent Progress in Multimodal Large Language Models (MLLMs) 的空间时间理解能力取得了进展，但仍然难以进行精确的空间时间理解。现有空间时间方法主要关注视频本身，而忽略了视频中的物理信息，如多对象布局和运动。这些限制限制了MLLMs在需要高精度的下游应用中的使用，包括嵌入式智能和VR。为了解决这一问题，我们提出Video-STR，一种基于图的增强学习方法，用于精确的视频空间时间推理。基于可验证奖励的增强学习（RLVR）来提高模型能力，我们引入了一种基于图的组相对策略优化（GRPO）方法，该方法在推理过程中引导模型推断场景的潜在空间时间拓扑结构。为了解决缺乏空间时间训练数据的问题，我们构建了包含205,000个问答对的STV-205k数据集，覆盖室内外动态多对象场景，以支持模型训练。实验结果表明，Video-STR在各种基准测试中达到了最先进的性能，在STI-Bench上的表现比基础模型高出13%，证明了我们方法和数据集的有效性。代码、模型和数据将公开发布。 

---
# LLM-Empowered Agentic MAC Protocols: A Dynamic Stackelberg Game Approach 

**Title (ZH)**: LLM赋能的代理MAC协议：一种动态Stackelberg博弈方法 

**Authors**: Renxuan Tan, Rongpeng Li, Fei Wang, Chenghui Peng, Shaoyun Wu, Zhifeng Zhao, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10895)  

**Abstract**: Medium Access Control (MAC) protocols, essential for wireless networks, are typically manually configured. While deep reinforcement learning (DRL)-based protocols enhance task-specified network performance, they suffer from poor generalizability and resilience, demanding costly retraining to adapt to dynamic environments. To overcome this limitation, we introduce a game-theoretic LLM-empowered multi-agent DRL (MARL) framework, in which the uplink transmission between a base station and a varying number of user equipments is modeled as a dynamic multi-follower Stackelberg game (MFSG), capturing the network's natural hierarchical structure. Within this game, LLM-driven agents, coordinated through proximal policy optimization (PPO), synthesize adaptive, semantic MAC protocols in response to network dynamics. Protocol action grammar (PAG) is employed to ensure the reliability and efficiency of this process. Under this system, we further analyze the existence and convergence behavior in terms of a Stackelberg equilibrium by studying the learning dynamics of LLM-empowered unified policies in response to changing followers. Simulations corroborate that our framework achieves a 77.6% greater throughput and a 65.2% fairness improvement over conventional baselines. Besides, our framework generalizes excellently to a fluctuating number of users without requiring retraining or architectural changes. 

**Abstract (ZH)**: 基于游戏理论的LLM赋能多代理DRL（MARL）的MAC协议设计 

---
# Unlocking Exploration in RLVR: Uncertainty-aware Advantage Shaping for Deeper Reasoning 

**Title (ZH)**: 在RLVR中解锁探索：基于不确定性优势塑造的深层推理 

**Authors**: Can Xie, Ruotong Pan, Xiangyu Wu, Yunfei Zhang, Jiayi Fu, Tingting Gao, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.10649)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has shown significant promise for enhancing the reasoning capabilities of large language models (LLMs). However, prevailing algorithms like GRPO broadcast a uniform advantage signal across all tokens in a sequence. This coarse-grained approach overlooks the pivotal role of uncertain, high-stakes decisions during reasoning, leading to inefficient exploration and the well-documented problem of entropy collapse. To address this, we introduce UnCertainty-aware Advantage Shaping (UCAS), a model-free method that refines credit assignment by leveraging the model's internal uncertainty signals. UCAS operates in two stages: it first modulates the response-level advantage using the model's overall self-confidence, and then applies a token-level penalty based on raw logit certainty. This dual mechanism encourages exploration of high-uncertainty paths that yield correct answers while penalizing overconfident yet erroneous reasoning, effectively balancing the exploration-exploitation trade-off. Extensive experiments on five mathematical reasoning benchmarks show that UCAS significantly outperforms strong RLVR baselines across multiple model scales, including 1.5B and 7B. Our analysis confirms that UCAS not only achieves higher rewards but also promotes greater reasoning diversity and successfully mitigates entropy collapse. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）在增强大规模语言模型的推理能力方面显示出显著的前景。然而，现有的算法如GRPO会在整个序列的所有令牌中广播统一的优势信号。这种粗粒度的方法忽视了推理过程中至关重要的不确定性和高风险决策的作用，导致探索效率低下和已知的熵坍缩问题。为了解决这个问题，我们引入了基于不确定性的优势塑造（UCAS）方法，这是一种无需模型的方法，通过利用模型的内部不确定性信号来细化信用分配。UCAS分为两个阶段：首先，利用模型的整体自我信心调整响应级优势；然后，基于原始logits的确定性施加令牌级惩罚。这种双重机制鼓励探索具有高不确定性但能得出正确答案的路径，并对过于自信但错误的推理进行惩罚，从而有效平衡探索与利用之间的trade-off。在五个数学推理基准上的 extensive 实验表明，UCAS 在多个模型规模（包括1.5B和7B）下显著优于强大的RLVR基线。我们的分析证实，UCAS 不仅能够获得更高的奖励，还能促进更高的推理多样性，并成功缓解熵坍缩问题。 

---
# A Flexible Multi-Agent Deep Reinforcement Learning Framework for Dynamic Routing and Scheduling of Latency-Critical Services 

**Title (ZH)**: 一种用于低延迟关键服务动态路由和调度的灵活多agent深度强化学习框架 

**Authors**: Vincenzo Norman Vitale, Antonia Maria Tulino, Andreas F. Molisch, Jaime Llorca  

**Link**: [PDF](https://arxiv.org/pdf/2510.11535)  

**Abstract**: Timely delivery of delay-sensitive information over dynamic, heterogeneous networks is increasingly essential for a range of interactive applications, such as industrial automation, self-driving vehicles, and augmented reality. However, most existing network control solutions target only average delay performance, falling short of providing strict End-to-End (E2E) peak latency guarantees. This paper addresses the challenge of reliably delivering packets within application-imposed deadlines by leveraging recent advancements in Multi-Agent Deep Reinforcement Learning (MA-DRL). After introducing the Delay-Constrained Maximum-Throughput (DCMT) dynamic network control problem, and highlighting the limitations of current solutions, we present a novel MA-DRL network control framework that leverages a centralized routing and distributed scheduling architecture. The proposed framework leverages critical networking domain knowledge for the design of effective MA-DRL strategies based on the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) technique, where centralized routing and distributed scheduling agents dynamically assign paths and schedule packet transmissions according to packet lifetimes, thereby maximizing on-time packet delivery. The generality of the proposed framework allows integrating both data-driven \blue{Deep Reinforcement Learning (DRL)} agents and traditional rule-based policies in order to strike the right balance between performance and learning complexity. Our results confirm the superiority of the proposed framework with respect to traditional stochastic optimization-based approaches and provide key insights into the role and interplay between data-driven DRL agents and new rule-based policies for both efficient and high-performance control of latency-critical services. 

**Abstract (ZH)**: 及时交付敏感延迟信息在动态异构网络中的可靠传输对于工业自动化、自动驾驶车辆和增强现实等交互应用日益重要。然而，现有的大多数网络控制解决方案仅专注于平均延迟性能，未能提供严格的端到端（E2E）峰延迟保证。本文通过利用多智能体深度强化学习（MA-DRL）的最新进展，解决了在应用限定时间内可靠传输数据包的挑战。在介绍延迟受限最大吞吐量（DCMT）动态网络控制问题及其现有解决方案的局限性后，我们提出了一种利用集中式路由和分布式调度架构的新型MA-DRL网络控制框架。该框架基于多智能体深度确定性策略梯度（MADDPG）技术，结合关键的网络领域知识，设计有效的MA-DRL策略，通过动态分配路径和调度数据包传输，根据数据包的生命周期最大化按时传输数据包。该框架的通用性使其能够结合数据驱动的深度强化学习（DRL）代理和传统基于规则的策略，以平衡性能和学习复杂度。我们的实验结果证实了该框架优于传统的基于随机优化的方法，并提供了关于数据驱动的DRL代理和新规则基础策略在时延关键服务高效高性能控制中的角色和互动的关键见解。 

---
# People use fast, flat goal-directed simulation to reason about novel problems 

**Title (ZH)**: 人们使用快速、平坦的目标导向模拟来推理解决新型问题。 

**Authors**: Katherine M. Collins, Cedegao E. Zhang, Lionel Wong, Mauricio Barba da Costa, Graham Todd, Adrian Weller, Samuel J. Cheyette, Thomas L. Griffiths, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2510.11503)  

**Abstract**: Games have long been a microcosm for studying planning and reasoning in both natural and artificial intelligence, especially with a focus on expert-level or even super-human play. But real life also pushes human intelligence along a different frontier, requiring people to flexibly navigate decision-making problems that they have never thought about before. Here, we use novice gameplay to study how people make decisions and form judgments in new problem settings. We show that people are systematic and adaptively rational in how they play a game for the first time, or evaluate a game (e.g., how fair or how fun it is likely to be) before they have played it even once. We explain these capacities via a computational cognitive model that we call the "Intuitive Gamer". The model is based on mechanisms of fast and flat (depth-limited) goal-directed probabilistic simulation--analogous to those used in Monte Carlo tree-search models of expert game-play, but scaled down to use very few stochastic samples, simple goal heuristics for evaluating actions, and no deep search. In a series of large-scale behavioral studies with over 1000 participants and 121 two-player strategic board games (almost all novel to our participants), our model quantitatively captures human judgments and decisions varying the amount and kind of experience people have with a game--from no experience at all ("just thinking"), to a single round of play, to indirect experience watching another person and predicting how they should play--and does so significantly better than much more compute-intensive expert-level models. More broadly, our work offers new insights into how people rapidly evaluate, act, and make suggestions when encountering novel problems, and could inform the design of more flexible and human-like AI systems that can determine not just how to solve new tasks, but whether a task is worth thinking about at all. 

**Abstract (ZH)**: 游戏_long久以来都是研究自然智能和人工智能规划与推理的一个微观世界，特别是在专家级甚至超人级游戏方面。但在现实生活中，人类智能也沿着不同的前沿推进，要求人们灵活地解决前所未有的决策问题。在这里，我们使用新手玩家的游戏行为来研究人们在新问题设置中如何做出决策和形成判断。我们展示了人们在第一次玩游戏或在根本没有玩过的情况下评估游戏（例如，游戏可能有多公平、多有趣）时，表现出系统性和适应性的理性。我们通过一个被称为“直觉玩家”的计算认知模型来解释这些能力。该模型基于快速和平坦（深度受限）的目标导向概率模拟机制——类似于用于专家级游戏模拟的蒙特卡洛树搜索模型中的机制，但缩小规模以使用极少的随机样本、简单的动作目标启发式评估以及不进行深层次搜索。在涉及超过1000名参与者和121款两人制策略棋盘游戏（几乎所有游戏对参与者而言都是全新的）的一系列大规模行为研究中，我们的模型定量捕捉了人们在不同经验和类型的游戏情况下的人类判断和决策，并显著优于更耗费计算资源的专家级模型。更广泛地说，我们的工作为理解人们在遇到新问题时如何快速评估、行动和提供建议提供了新的见解，并可指导设计更具灵活性和人性化的AI系统，这些系统不仅能确定如何解决新任务，还能判断一个任务是否值得思考。 

---
# Offline Reinforcement Learning with Generative Trajectory Policies 

**Title (ZH)**: 基于生成轨迹策略的离线强化学习 

**Authors**: Xinsong Feng, Leshu Tang, Chenan Wang, Haipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11499)  

**Abstract**: Generative models have emerged as a powerful class of policies for offline reinforcement learning (RL) due to their ability to capture complex, multi-modal behaviors. However, existing methods face a stark trade-off: slow, iterative models like diffusion policies are computationally expensive, while fast, single-step models like consistency policies often suffer from degraded performance. In this paper, we demonstrate that it is possible to bridge this gap. The key to moving beyond the limitations of individual methods, we argue, lies in a unifying perspective that views modern generative models, including diffusion, flow matching, and consistency models, as specific instances of learning a continuous-time generative trajectory governed by an Ordinary Differential Equation (ODE). This principled foundation provides a clearer design space for generative policies in RL and allows us to propose Generative Trajectory Policies (GTPs), a new and more general policy paradigm that learns the entire solution map of the underlying ODE. To make this paradigm practical for offline RL, we further introduce two key theoretically principled adaptations. Empirical results demonstrate that GTP achieves state-of-the-art performance on D4RL benchmarks - it significantly outperforms prior generative policies, achieving perfect scores on several notoriously hard AntMaze tasks. 

**Abstract (ZH)**: 生成模型由于能够捕获复杂、多模态行为，已在离线 reinforcement learning 中 emerged 为一个强大的政策类别。然而，现有的方法面临着一个明显的权衡：迭代且计算成本高昂的扩散策略与其他单步且性能退化的一致性策略之间。在本文中，我们展示了一种超越这些限制的可能性。我们认为超越单一方法限制的关键在于一个统一的观点，即现代生成模型，包括扩散模型、流匹配模型和一致性模型，都是由常微分方程 (ODE) 治律的连续时间生成轨迹的具体实例。这个原则性的基础为生成策略在 RL 中的设计空间提供了更清晰的指导，并使我们能够提出生成轨迹策略（GTPs），这是一种新的更通用的策略范式，学习底层 ODE 的整个解映射。为了使这种范式在离线下能够实用，我们进一步介绍了两个重要的理论原则性改版。实验结果表明，GTP 在 D4RL 基准测试中达到了最先进的性能——它显著优于先前的生成策略，在多个 notorious 的 AntMaze 任务中获得了满分。 

---
# AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Model 

**Title (ZH)**: 安第斯VL技术报告：高效移动端多模态大语言模型 

**Authors**: Zhiwei Jin, Xiaohui Song, Nan Wang, Yafei Liu, Chao Li, Xin Li, Ruichen Wang, Zhihao Li, Qi Qi, Long Cheng, Dongze Hao, Quanlong Zheng, Yanhao Zhang, Haobo Ji, Jian Ma, Zhitong Zheng, Zhenyi Lin, Haolin Deng, Xin Zou, Xiaojie Yin, Ruilin Wang, Liankai Cai, Haijing Liu, Yuqing Qiu, Ke Chen, Zixian Li, Chi Xie, Huafei Li, Chenxing Li, Chuangchuang Wang, Kai Tang, Zhiguang Zhu, Kai Tang, Wenmei Gao, Rui Wang, Jun Wu, Chao Liu, Qin Xie, Chen Chen, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11496)  

**Abstract**: In recent years, while cloud-based MLLMs such as QwenVL, InternVL, GPT-4o, Gemini, and Claude Sonnet have demonstrated outstanding performance with enormous model sizes reaching hundreds of billions of parameters, they significantly surpass the limitations in memory, power consumption, and computing capacity of edge devices such as mobile phones. This paper introduces AndesVL, a suite of mobile-side MLLMs with 0.6B to 4B parameters based on Qwen3's LLM and various visual encoders. We comprehensively outline the model architectures, training pipeline, and training data of AndesVL, which achieves first-tier performance across a wide range of open-source benchmarks, including fields such as text-rich image understanding, reasoning and math, multi-image comprehension, general VQA, hallucination mitigation, multilingual understanding, and GUI-related tasks when compared with state-of-the-art models of a similar scale. Furthermore, we introduce a 1+N LoR 

**Abstract (ZH)**: 近年来，云-based MLLMs如QwenVL、InternVL、GPT-4o、Gemini和Claude Sonnet凭借数百亿参数的巨大模型规模展现了出色表现，但显著超越了如移动电话等边缘设备在内存、功耗和计算能力上的限制。本文介绍了一套基于Qwen3的LLM和多种视觉编码器的移动侧MLLMs AndesVL，参数范围从0.6B到4B。我们详细介绍了AndesVL的模型架构、训练流水线和训练数据，该模型在包括图文理解、推理和数学、多图理解、通用VQA、幻觉缓解、多语言理解和与GUI相关的任务在内的多种开源基准测试中达到了顶级性能，与类似规模的先进模型相比。此外，我们还介绍了1+N LoR。 

---
# FOSSIL: Harnessing Feedback on Suboptimal Samples for Data-Efficient Generalisation with Imitation Learning for Embodied Vision-and-Language Tasks 

**Title (ZH)**: FOSSIL：利用反馈优化亚优样本以提高模仿学习在实体视觉-语言任务中数据高效泛化的性能 

**Authors**: Sabrina McCallum, Amit Parekh, Alessandro Suglia  

**Link**: [PDF](https://arxiv.org/pdf/2510.11307)  

**Abstract**: Current approaches to embodied AI tend to learn policies from expert demonstrations. However, without a mechanism to evaluate the quality of demonstrated actions, they are limited to learning from optimal behaviour, or they risk replicating errors and inefficiencies. While reinforcement learning offers one alternative, the associated exploration typically results in sacrificing data efficiency. This work explores how agents trained with imitation learning can learn robust representations from both optimal and suboptimal demonstrations when given access to constructive language feedback as a means to contextualise different modes of behaviour. We directly provide language feedback embeddings as part of the input sequence into a Transformer-based policy, and optionally complement the traditional next action prediction objective with auxiliary self-supervised learning objectives for feedback prediction. We test our approach on a range of embodied Vision-and-Language tasks in our custom BabyAI-XGen environment and show significant improvements in agents' compositional generalisation abilities and robustness, suggesting that our data-efficient method allows models to successfully convert suboptimal behaviour into learning opportunities. Overall, our results suggest that language feedback is a competitive and intuitive alternative to intermediate scalar rewards for language-specified embodied tasks. 

**Abstract (ZH)**: 基于语言反馈的模仿学习在体态AI中的数据高效表示学习 

---
# VeritasFi: An Adaptable, Multi-tiered RAG Framework for Multi-modal Financial Question Answering 

**Title (ZH)**: VeritasFi：一种适应性强的多层级RAG框架用于多模态金融问答 

**Authors**: Zhenghan Tai, Hanwei Wu, Qingchen Hu, Jijun Chi, Hailin He, Lei Ding, Tung Sum Thomas Kwok, Bohuai Xiao, Yuchen Hua, Suyuchen Wang, Peng Lu, Muzhi Li, Yihong Wu, Liheng Ma, Jerry Huang, Jiayi Zhang, Gonghao Zhang, Chaolong Jiang, Jingrui Tian, Sicheng Lyu, Zeyu Li, Boyu Han, Fengran Mo, Xinyue Yu, Yufei Cui, Ling Zhou, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10828)  

**Abstract**: Retrieval-Augmented Generation (RAG) is becoming increasingly essential for Question Answering (QA) in the financial sector, where accurate and contextually grounded insights from complex public disclosures are crucial. However, existing financial RAG systems face two significant challenges: (1) they struggle to process heterogeneous data formats, such as text, tables, and figures; and (2) they encounter difficulties in balancing general-domain applicability with company-specific adaptation. To overcome these challenges, we present VeritasFi, an innovative hybrid RAG framework that incorporates a multi-modal preprocessing pipeline alongside a cutting-edge two-stage training strategy for its re-ranking component. VeritasFi enhances financial QA through three key innovations: (1) A multi-modal preprocessing pipeline that seamlessly transforms heterogeneous data into a coherent, machine-readable format. (2) A tripartite hybrid retrieval engine that operates in parallel, combining deep multi-path retrieval over a semantically indexed document corpus, real-time data acquisition through tool utilization, and an expert-curated memory bank for high-frequency questions, ensuring comprehensive scope, accuracy, and efficiency. (3) A two-stage training strategy for the document re-ranker, which initially constructs a general, domain-specific model using anonymized data, followed by rapid fine-tuning on company-specific data for targeted applications. By integrating our proposed designs, VeritasFi presents a groundbreaking framework that greatly enhances the adaptability and robustness of financial RAG systems, providing a scalable solution for both general-domain and company-specific QA tasks. Code accompanying this work is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）对于金融领域的问答（QA）变得日益重要，因为准确且上下文相关的大规模财务披露洞察至关重要。然而，现有的金融RAG系统面临两大挑战：（1）它们难以处理异构数据格式，如文本、表格和图表；（2）难以在通用领域适用性和公司特定适应性之间取得平衡。为了克服这些挑战，我们提出了VeritasFi，一种创新的混合RAG框架，结合了多模态预处理管道和最新的两阶段训练策略，用于其重排组件。VeritasFi通过三项创新提升金融问答：（1）一个无缝转换异构数据为一致、机器可读格式的多模态预处理管道；（2）一个三元混合检索引擎，同时运行，结合深层多路径检索、实时数据采集以及由专家策展的高频问题记忆库，确保全面的范围、准确性和效率；（3）两个阶段的训练策略，用于文档重排名，首先使用匿名数据构建通用领域特定模型，然后迅速在公司特定数据上进行微调以实现目标应用。通过集成我们提出的设计，VeritasFi展示了大幅增强金融RAG系统适应性和鲁棒性的突破性框架，提供了一种可扩展的解决方案，适用于通用领域和公司特定的问答任务。与此工作相关联的代码可在以下网址获取。 

---
# Therapeutic AI and the Hidden Risks of Over-Disclosure: An Embedded AI-Literacy Framework for Mental Health Privacy 

**Title (ZH)**: therapeutic AI与过度披露的隐性风险：嵌入式AI素养框架下的精神健康隐私保护 

**Authors**: Soraya S. Anvari, Rina R. Wehbe  

**Link**: [PDF](https://arxiv.org/pdf/2510.10805)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mental health contexts, from structured therapeutic support tools to informal chat-based well-being assistants. While these systems increase accessibility, scalability, and personalization, their integration into mental health care brings privacy and safety challenges that have not been well-examined. Unlike traditional clinical interactions, LLM-mediated therapy often lacks a clear structure for what information is collected, how it is processed, and how it is stored or reused. Users without clinical guidance may over-disclose personal information, which is sometimes irrelevant to their presenting concern, due to misplaced trust, lack of awareness of data risks, or the conversational design of the system. This overexposure raises privacy concerns and also increases the potential for LLM bias, misinterpretation, and long-term data misuse. We propose a framework embedding Artificial Intelligence (AI) literacy interventions directly into mental health conversational systems, and outline a study plan to evaluate their impact on disclosure safety, trust, and user experience. 

**Abstract (ZH)**: 大型语言模型（LLMs）在心理健康领域的应用从结构化的治疗支持工具扩展到了非正式的聊天式福祉助手。尽管这些系统提升了可访问性、可扩展性和个性化，但它们在心理健康护理中的集成带来了隐私和安全挑战，这些挑战尚未得到充分研究。不同于传统的临床互动，通过LLM介导的治疗往往缺乏对收集什么信息、如何处理、存储或重用这些信息的明确结构。没有临床指导的用户可能会因信任感缺失、不了解数据风险或系统的设计而过度披露个人信息，这信息有时与其当前的困扰无关。这种过度披露引发了隐私担忧，也增加了LLM偏见、误解释和长期数据滥用的风险。我们提出将人工智能（AI）素养干预直接嵌入心理健康对话系统中，并概述一项研究计划以评估其对披露安全、信任和用户体验的影响。 

---
# Towards Self-Refinement of Vision-Language Models with Triangular Consistency 

**Title (ZH)**: 面向三角一致性导向的视觉-语言模型自我精炼方法 

**Authors**: Yunlong Deng, Guangyi Chen, Tianpei Gu, Lingjing Kong, Yan Li, Zeyu Tang, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10487)  

**Abstract**: Vision-Language Models (VLMs) integrate visual knowledge with the analytical capabilities of Large Language Models (LLMs) through supervised visual instruction tuning, using image-question-answer triplets. However, the potential of VLMs trained without supervised instruction remains largely unexplored. This study validates that VLMs possess inherent self-refinement capabilities, enabling them to generate high-quality supervised data without external inputs and thereby learn autonomously. Specifically, to stimulate the self-refinement ability of VLMs, we propose a self-refinement framework based on a Triangular Consistency principle: within the image-query-answer triangle, any masked elements should be consistently and accurately reconstructed. The framework involves three steps: (1) We enable the instruction generation ability of VLMs by adding multi-task instruction tuning like image$\rightarrow$question-answer or image-answer$\rightarrow$question. (2) We generate image-query-answer triplets from unlabeled images and use the Triangular Consistency principle for filtering. (3) The model is further updated using the filtered synthetic data. To investigate the underlying mechanisms behind this self-refinement capability, we conduct a theoretical analysis from a causal perspective. Using the widely recognized LLaVA-1.5 as our baseline, our experiments reveal that the model can autonomously achieve consistent, though deliberately modest, improvements across multiple benchmarks without any external supervision, such as human annotations or environmental feedback. We expect that the insights of this study on the self-refinement ability of VLMs can inspire future research on the learning mechanism of VLMs. Code is available at this https URL. 

**Abstract (ZH)**: Vision-Language模型的内在自我精炼能力：无需监督指令的学习机制探究 

---
# FML-bench: A Benchmark for Automatic ML Research Agents Highlighting the Importance of Exploration Breadth 

**Title (ZH)**: FML-bench: 一个强调探索广度重要性的自动机器学习研究代理基准测试 

**Authors**: Qiran Zou, Hou Hei Lam, Wenhao Zhao, Yiming Tang, Tingting Chen, Samson Yu, Tianyi Zhang, Chang Liu, Xiangyang Ji, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10472)  

**Abstract**: Large language models (LLMs) have sparked growing interest in automatic machine learning research agents. Among them, agents capable of autonomously proposing ideas and conducting machine learning experiments are particularly promising, as they maximize research automation and accelerate scientific progress by iteratively refining ideas based on experimental results. However, comprehensively evaluating such agents remains challenging. Existing benchmarks tend to overemphasize engineering aspects while neglecting academic rigor, creating barriers that obscure a clear assessment of an agent's scientific capabilities in machine learning research. They also suffer from limited task diversity, an overemphasis on application-oriented tasks over fundamental research problems, and limited scalability to realistic research settings. To address these limitations, we introduce FML-bench, a benchmark designed to evaluate automatic machine learning research agents on 8 diverse and fundamental machine learning research problems. It reduces coding burden, emphasizes fundamental problems rather than specific use cases, offers high task diversity, and is extensible to real-world machine learning GitHub repositories. Furthermore, we present a unified evaluation framework with five complementary metrics, designed to comprehensively assess agent performance on our benchmark. We evaluate state-of-the-art automatic research agents on FML-bench, and find that agents employing broad research exploration strategies outperform those focusing on narrow but deep exploration. These findings suggest that emphasizing the breadth of exploration may lead to more effective research outcomes than focusing solely on incremental refinement. Our benchmark is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）引发了自动机器学习研究代理的广泛关注。其中，能够自主提出想法并开展机器学习实验的代理尤其具有前景，因为它们通过基于实验结果迭代优化想法来最大化研究自动化并加速科学进步。然而，全面评估这些代理仍然是一个挑战。现有基准倾向于过分强调工程方面而忽略学术 rigor，这创建了阻碍，使清晰评估代理在机器学习研究中的科学能力变得模糊。它们还面临着任务多样性有限、过于强调应用导向任务而非基础研究问题以及难以扩展到现实研究环境的局限。为了应对这些局限，我们引入了FML-bench，这是一个旨在评估自动机器学习研究代理在8个多样化和基础的机器学习研究问题上的基准。该基准减少了编码负担，强调基本问题而非特定用例，提供了高任务多样性，并可扩展到真实的机器学习GitHub仓库。此外，我们提出了一个统一的评估框架，包含五个互补的指标，旨在全面评估代理在本基准上的性能。我们在FML-bench上评估了最先进的自动研究代理，并发现采用广泛研究探索策略的代理优于专注于狭窄但深刻探索的代理。这些发现表明，强调探索的广度可能比仅仅关注逐步优化更为有效。我们的基准可在此处访问：this https URL。 

---
# Testing and Enhancing Multi-Agent Systems for Robust Code Generation 

**Title (ZH)**: 测试与增强多agent系统以实现稳健的代码生成 

**Authors**: Zongyi Lyu, Songqiang Chen, Zhenlan Ji, Liwen Wang, Shuai Wang, Daoyuan Wu, Wenxuan Wang, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2510.10460)  

**Abstract**: Multi-agent systems (MASs) have emerged as a promising paradigm for automated code generation, demonstrating impressive performance on established benchmarks by decomposing complex coding tasks across specialized agents with different roles. Despite their prosperous development and adoption, their robustness remains pressingly under-explored, raising critical concerns for real-world deployment. This paper presents the first comprehensive study examining the robustness of MASs for code generation through a fuzzing-based testing approach. By designing a fuzzing pipeline incorporating semantic-preserving mutation operators and a novel fitness function, we assess mainstream MASs across multiple datasets and LLMs. Our findings reveal substantial robustness flaws of various popular MASs: they fail to solve 7.9%-83.3% of problems they initially resolved successfully after applying the semantic-preserving mutations. Through comprehensive failure analysis, we identify a common yet largely overlooked cause of the robustness issue: miscommunications between planning and coding agents, where plans lack sufficient detail and coding agents misinterpret intricate logic, aligning with the challenges inherent in a multi-stage information transformation process. Accordingly, we also propose a repairing method that encompasses multi-prompt generation and introduces a new monitor agent to address this issue. Evaluation shows that our repairing method effectively enhances the robustness of MASs by solving 40.0%-88.9% of identified failures. Our work uncovers critical robustness flaws in MASs and provides effective mitigation strategies, contributing essential insights for developing more reliable MASs for code generation. 

**Abstract (ZH)**: 基于 fuzzing 的多-agent 系统代码生成鲁棒性研究 

---
# NIM: Neuro-symbolic Ideographic Metalanguage for Inclusive Communication 

**Title (ZH)**: NIM: 神经符号表意元语言促进包容性沟通 

**Authors**: Prawaal Sharma, Poonam Goyal, Navneet Goyal, Vidisha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.10459)  

**Abstract**: Digital communication has become the cornerstone of modern interaction, enabling rapid, accessible, and interactive exchanges. However, individuals with lower academic literacy often face significant barriers, exacerbating the "digital divide". In this work, we introduce a novel, universal ideographic metalanguage designed as an innovative communication framework that transcends academic, linguistic, and cultural boundaries. Our approach leverages principles of Neuro-symbolic AI, combining neural-based large language models (LLMs) enriched with world knowledge and symbolic knowledge heuristics grounded in the linguistic theory of Natural Semantic Metalanguage (NSM). This enables the semantic decomposition of complex ideas into simpler, atomic concepts. Adopting a human-centric, collaborative methodology, we engaged over 200 semi-literate participants in defining the problem, selecting ideographs, and validating the system. With over 80\% semantic comprehensibility, an accessible learning curve, and universal adaptability, our system effectively serves underprivileged populations with limited formal education. 

**Abstract (ZH)**: 数字通信已成为现代交互的基石，使快速、便捷和互动的交流成为可能。然而，低学术素养的个体往往面临显著障碍，加剧了“数字鸿沟”。在本文中，我们介绍了一种新颖的通用意象型元语言，设计为一种创新的超越学术、语言和文化边界的交流框架。我们的方法利用神经符号人工智能的原则，结合神经基础的大语言模型（LLMs），这些模型富含世界知识，并基于自然语义元语言（NSM）理论的地基符号知识启发式规则。这使复杂的概念能够分解为更简单、更基本的概念。采用以人为本、协作的方法，我们与超过200名半文盲参与者共同定义问题、选择意象并验证系统。通过超过80%的语义可理解性、易于学习的曲线和普遍适应性，我们的系统有效服务于受教育程度有限的弱势群体。 

---
# Data-driven simulator of multi-animal behavior with unknown dynamics via offline and online reinforcement learning 

**Title (ZH)**: 基于离线和在线强化学习的多动物行为数据驱动模拟器（未知动态情况） 

**Authors**: Keisuke Fujii, Kazushi Tsutsui, Yu Teshima, Makoto Itoh, Naoya Takeishi, Nozomi Nishiumi, Ryoya Tanaka, Shunsuke Shigaki, Yoshinobu Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2510.10451)  

**Abstract**: Simulators of animal movements play a valuable role in studying behavior. Advances in imitation learning for robotics have expanded possibilities for reproducing human and animal movements. A key challenge for realistic multi-animal simulation in biology is bridging the gap between unknown real-world transition models and their simulated counterparts. Because locomotion dynamics are seldom known, relying solely on mathematical models is insufficient; constructing a simulator that both reproduces real trajectories and supports reward-driven optimization remains an open problem. We introduce a data-driven simulator for multi-animal behavior based on deep reinforcement learning and counterfactual simulation. We address the ill-posed nature of the problem caused by high degrees of freedom in locomotion by estimating movement variables of an incomplete transition model as actions within an RL framework. We also employ a distance-based pseudo-reward to align and compare states between cyber and physical spaces. Validated on artificial agents, flies, newts, and silkmoth, our approach achieves higher reproducibility of species-specific behaviors and improved reward acquisition compared with standard imitation and RL methods. Moreover, it enables counterfactual behavior prediction in novel experimental settings and supports multi-individual modeling for flexible what-if trajectory generation, suggesting its potential to simulate and elucidate complex multi-animal behaviors. 

**Abstract (ZH)**: 基于深度强化学习和反事实模拟的多动物行为数据驱动仿真 

---
