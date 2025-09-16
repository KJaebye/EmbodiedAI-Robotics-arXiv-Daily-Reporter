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
# Time-Constrained Intelligent Adversaries for Automation Vulnerability Testing: A Multi-Robot Patrol Case Study 

**Title (ZH)**: 时间约束下的智能对手在自动化脆弱性测试中的多机器人巡逻案例研究 

**Authors**: James C. Ward, Alex Bott, Connor York, Edmund R. Hunt  

**Link**: [PDF](https://arxiv.org/pdf/2509.11971)  

**Abstract**: Simulating hostile attacks of physical autonomous systems can be a useful tool to examine their robustness to attack and inform vulnerability-aware design. In this work, we examine this through the lens of multi-robot patrol, by presenting a machine learning-based adversary model that observes robot patrol behavior in order to attempt to gain undetected access to a secure environment within a limited time duration. Such a model allows for evaluation of a patrol system against a realistic potential adversary, offering insight into future patrol strategy design. We show that our new model outperforms existing baselines, thus providing a more stringent test, and examine its performance against multiple leading decentralized multi-robot patrol strategies. 

**Abstract (ZH)**: 通过多机器人巡逻视角下的基于机器学习的攻击模型模拟敌对攻击可以成为评估物理自主系统抗攻击 robustness 以及指导防范设计的有效工具。在本文中，我们通过构建一个基于机器学习的对手模型来观察机器人巡逻行为，以期在有限时间内尝试在安全环境中实现未被察觉的访问。该模型允许我们评估巡逻系统对抗现实潜在对手的性能，并为未来的巡逻策略设计提供洞见。我们展示了新模型在性能上优于现有基准，并对其在多个领先的去中心化多机器人巡逻策略中的表现进行了考察。 

---
# VH-Diffuser: Variable Horizon Diffusion Planner for Time-Aware Goal-Conditioned Trajectory Planning 

**Title (ZH)**: 变量时间 horizon 扩散规划器：时间意识的目标条件轨迹规划 

**Authors**: Ruijia Liu, Ancheng Hou, Shaoyuan Li, Xiang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.11930)  

**Abstract**: Diffusion-based planners have gained significant recent attention for their robustness and performance in long-horizon tasks. However, most existing planners rely on a fixed, pre-specified horizon during both training and inference. This rigidity often produces length-mismatch (trajectories that are too short or too long) and brittle performance across instances with varying geometric or dynamical difficulty. In this paper, we introduce the Variable Horizon Diffuser (VHD) framework, which treats the horizon as a learned variable rather than a fixed hyperparameter. Given a start-goal pair, we first predict an instance-specific horizon using a learned Length Predictor model, which guides a Diffusion Planner to generate a trajectory of the desired length. Our design maintains compatibility with existing diffusion planners by controlling trajectory length through initial noise shaping and training on randomly cropped sub-trajectories, without requiring architectural changes. Empirically, VHD improves success rates and path efficiency in maze-navigation and robot-arm control benchmarks, showing greater robustness to horizon mismatch and unseen lengths, while keeping training simple and offline-only. 

**Abstract (ZH)**: 基于扩散的规划器在长时_horizon任务中展现出了显著的稳健性和性能，但大多数现有的规划器在训练和推理过程中都依赖于固定且预先指定的horizon。这种固有性经常会产出长度不匹配的问题（轨迹过短或过长）并且在具有不同几何或动力学难度的任务间表现出脆弱的表现。在这篇论文中，我们提出了Variable Horizon Diffuser（VHD）框架，将horizon视为一个可学习的变量而不是固定的超参数。给定起点和终点，我们首先使用一个学习到的长度预测模型来预测一个实例特定的horizon，从而引导扩散规划器生成所需长度的轨迹。我们的设计通过初始噪声塑形控制轨迹长度，并通过在随机裁剪的子轨迹上进行训练来保持与现有扩散规划器的兼容性，无需修改架构。实验结果显示，VHD在迷宫导航和机器人臂控制基准测试中提高了成功率和路径效率，对horizon不匹配和未见过的长度具有更强的鲁棒性，同时保持了简单的无监督训练。 

---
# Tenma: Robust Cross-Embodiment Robot Manipulation with Diffusion Transformer 

**Title (ZH)**: Tenma：基于扩散变换器的鲁棒跨身体机器人操作 

**Authors**: Travis Davies, Yiqi Huang, Yunxin Liu, Xiang Chen, Huxian Liu, Luhui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11865)  

**Abstract**: Scaling Transformer policies and diffusion models has advanced robotic manipulation, yet combining these techniques in lightweight, cross-embodiment learning settings remains challenging. We study design choices that most affect stability and performance for diffusion-transformer policies trained on heterogeneous, multimodal robot data, and introduce Tenma, a lightweight diffusion-transformer for bi-manual arm control. Tenma integrates multiview RGB, proprioception, and language via a cross-embodiment normalizer that maps disparate state/action spaces into a shared latent space; a Joint State-Time encoder for temporally aligned observation learning with inference speed boosts; and a diffusion action decoder optimized for training stability and learning capacity. Across benchmarks and under matched compute, Tenma achieves an average success rate of 88.95% in-distribution and maintains strong performance under object and scene shifts, substantially exceeding baseline policies whose best in-distribution average is 18.12%. Despite using moderate data scale, Tenma delivers robust manipulation and generalization, indicating the great potential for multimodal and cross-embodiment learning strategies for further augmenting the capacity of transformer-based imitation learning policies. 

**Abstract (ZH)**: scaling transformer策略和扩散模型促进了机器人操作，但在轻量级、跨体态学习环境中结合这些技术仍具有挑战性。我们研究了对异构多模态机器人数据训练的扩散-变压器策略影响稳定性和性能的主要设计选择，并引入了Tenma，一种轻量级的双臂控制扩散-变压器。Tenma通过跨体态归一化器将不同状态/动作空间映射到共享潜在空间；通过关节状态-时间编码器实现时间对齐的观测学习，并提升推理速度；并通过优化训练稳定性和学习能力的扩散动作解码器。在基准测试中，Tenma的平均成功率在分布内达到88.95%，并在对象和场景变化下保持了强劲表现，大幅超过了平均最佳在分布内成功率仅18.12%的基础策略。尽管使用了中等规模的数据集，Tenma展示了稳健的操作能力和泛化能力，表明了多模态和跨体态学习策略在进一步增强基于变压器的模仿学习策略能力方面的巨大潜力。 

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

**Title (ZH)**: ParaEQsA: 并行异步 embodied 问题调度与回答 

**Authors**: Haisheng Wang, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11663)  

**Abstract**: This paper formulates the Embodied Questions Answering (EQsA) problem, introduces a corresponding benchmark, and proposes a system to tackle the problem. Classical Embodied Question Answering (EQA) is typically formulated as answering one single question by actively exploring a 3D environment. Real deployments, however, often demand handling multiple questions that may arrive asynchronously and carry different urgencies. We formalize this setting as Embodied Questions Answering (EQsA) and present ParaEQsA, a framework for parallel, urgency-aware scheduling and answering. ParaEQsA leverages a group memory module shared among questions to reduce redundant exploration, and a priority-planning module to dynamically schedule questions. To evaluate this setting, we contribute the Parallel Asynchronous Embodied Questions (PAEQs) benchmark containing 40 indoor scenes and five questions per scene (200 in total), featuring asynchronous follow-up questions and urgency labels. We further propose metrics for EQsA performance: Direct Answer Rate (DAR), and Normalized Urgency-Weighted Latency (NUWL), which jointly measure efficiency and responsiveness of this system. ParaEQsA consistently outperforms strong sequential baselines adapted from recent EQA systems, while reducing exploration and delay. Empirical evaluations investigate the relative contributions of priority, urgency modeling, spatial scope, reward estimation, and dependency reasoning within our framework. Together, these results demonstrate that urgency-aware, parallel scheduling is key to making embodied agents responsive and efficient under realistic, multi-question workloads. 

**Abstract (ZH)**: 本文形式化了嵌入式疑问回答（EQsA）问题，引入了一个相应的基准，并提出了一套系统来解决该问题。 

---
# Inference-stage Adaptation-projection Strategy Adapts Diffusion Policy to Cross-manipulators Scenarios 

**Title (ZH)**: 推理阶段适配投影策略适配跨操作场景的扩散政策 

**Authors**: Xiangtong Yao, Yirui Zhou, Yuan Meng, Yanwen Liu, Liangyu Dong, Zitao Zhang, Zhenshan Bing, Kai Huang, Fuchun Sun, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.11621)  

**Abstract**: Diffusion policies are powerful visuomotor models for robotic manipulation, yet they often fail to generalize to manipulators or end-effectors unseen during training and struggle to accommodate new task requirements at inference time. Addressing this typically requires costly data recollection and policy retraining for each new hardware or task configuration. To overcome this, we introduce an adaptation-projection strategy that enables a diffusion policy to perform zero-shot adaptation to novel manipulators and dynamic task settings, entirely at inference time and without any retraining. Our method first trains a diffusion policy in SE(3) space using demonstrations from a base manipulator. During online deployment, it projects the policy's generated trajectories to satisfy the kinematic and task-specific constraints imposed by the new hardware and objectives. Moreover, this projection dynamically adapts to physical differences (e.g., tool-center-point offsets, jaw widths) and task requirements (e.g., obstacle heights), ensuring robust and successful execution. We validate our approach on real-world pick-and-place, pushing, and pouring tasks across multiple manipulators, including the Franka Panda and Kuka iiwa 14, equipped with a diverse array of end-effectors like flexible grippers, Robotiq 2F/3F grippers, and various 3D-printed designs. Our results demonstrate consistently high success rates in these cross-manipulator scenarios, proving the effectiveness and practicality of our adaptation-projection strategy. The code will be released after peer review. 

**Abstract (ZH)**: 一种在推理时进行零样本适应和投影的扩散策略：用于机器人操作的通用视觉-运动模型 

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
# FR-Net: Learning Robust Quadrupedal Fall Recovery on Challenging Terrains through Mass-Contact Prediction 

**Title (ZH)**: FR-网：通过质量接触预测在具有挑战性的地形上学习稳健的四足跌倒恢复 

**Authors**: Yidan Lu, Yinzhao Dong, Jiahui Zhang, Ji Ma, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11504)  

**Abstract**: Fall recovery for legged robots remains challenging, particularly on complex terrains where traditional controllers fail due to incomplete terrain perception and uncertain interactions. We present \textbf{FR-Net}, a learning-based framework that enables quadrupedal robots to recover from arbitrary fall poses across diverse environments. Central to our approach is a Mass-Contact Predictor network that estimates the robot's mass distribution and contact states from limited sensory inputs, facilitating effective recovery strategies. Our carefully designed reward functions ensure safe recovery even on steep stairs without dangerous rolling motions common to existing methods. Trained entirely in simulation using privileged learning, our framework guides policy learning without requiring explicit terrain data during deployment. We demonstrate the generalization capabilities of \textbf{FR-Net} across different quadrupedal platforms in simulation and validate its performance through extensive real-world experiments on the Go2 robot in 10 challenging scenarios. Our results indicate that explicit mass-contact prediction is key to robust fall recovery, offering a promising direction for generalizable quadrupedal skills. 

**Abstract (ZH)**: 基于学习的四足机器人任意跌倒姿态恢复方法：跨复杂环境的有效策略 

---
# Enhancing Generalization in Vision-Language-Action Models by Preserving Pretrained Representations 

**Title (ZH)**: 通过保留预训练表示以增强视觉-语言-动作模型的泛化能力 

**Authors**: Shresth Grover, Akshay Gopalkrishnan, Bo Ai, Henrik I. Christensen, Hao Su, Xuanlin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11417)  

**Abstract**: Vision-language-action (VLA) models finetuned from vision-language models (VLMs) hold the promise of leveraging rich pretrained representations to build generalist robots across diverse tasks and environments. However, direct fine-tuning on robot data often disrupts these representations and limits generalization. We present a framework that better preserves pretrained features while adapting them for robot manipulation. Our approach introduces three components: (i) a dual-encoder design with one frozen vision encoder to retain pretrained features and another trainable for task adaptation, (ii) a string-based action tokenizer that casts continuous actions into character sequences aligned with the model's pretraining domain, and (iii) a co-training strategy that combines robot demonstrations with vision-language datasets emphasizing spatial reasoning and affordances. Evaluations in simulation and on real robots show that our method improves robustness to visual perturbations, generalization to novel instructions and environments, and overall task success compared to baselines. 

**Abstract (ZH)**: 基于视觉-语言-动作模型的框架能够更好地保留预训练特征并适应机器人操作任务 

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

**Title (ZH)**: 社会机器人引导物理治疗的政策学习 

**Authors**: Carl Bettosi, Lynne Ballie, Susan Shenkin, Marta Romeo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11297)  

**Abstract**: Social robots offer a promising solution for autonomously guiding patients through physiotherapy exercise sessions, but effective deployment requires advanced decision-making to adapt to patient needs. A key challenge is the scarcity of patient behavior data for developing robust policies. To address this, we engaged 33 expert healthcare practitioners as patient proxies, using their interactions with our robot to inform a patient behavior model capable of generating exercise performance metrics and subjective scores on perceived exertion. We trained a reinforcement learning-based policy in simulation, demonstrating that it can adapt exercise instructions to individual exertion tolerances and fluctuating performance, while also being applicable to patients at different recovery stages with varying exercise plans. 

**Abstract (ZH)**: 社会机器人提供了自主引导患者进行物理治疗练习的一种有前景的解决方案，但有效的部署需要先进的决策能力以适应患者需求。一个关键挑战是用于开发稳健策略的患者行为数据的稀缺性。为了应对这一挑战，我们邀请了33名专家医疗从业者作为患者代理，利用他们与我们机器人交互的信息来建立一个能够生成锻炼绩效指标和主观疲劳评分的患者行为模型。我们在仿真中训练了一个基于强化学习的策略，证明它可以适应个体的投入耐受度和波动表现，并且适用于处于不同恢复阶段、具有不同锻炼计划的患者。 

---
# Embodied Intelligence in Disassembly: Multimodal Perception Cross-validation and Continual Learning in Neuro-Symbolic TAMP 

**Title (ZH)**: embodied 智能在拆卸中的应用：神经符号TAMP的多模态感知交叉验证与持续学习 

**Authors**: Ziwen He, Zhigang Wang, Yanlong Peng, Pengxu Chang, Hong Yang, Ming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11270)  

**Abstract**: With the rapid development of the new energy vehicle industry, the efficient disassembly and recycling of power batteries have become a critical challenge for the circular economy. In current unstructured disassembly scenarios, the dynamic nature of the environment severely limits the robustness of robotic perception, posing a significant barrier to autonomous disassembly in industrial applications. This paper proposes a continual learning framework based on Neuro-Symbolic task and motion planning (TAMP) to enhance the adaptability of embodied intelligence systems in dynamic environments. Our approach integrates a multimodal perception cross-validation mechanism into a bidirectional reasoning flow: the forward working flow dynamically refines and optimizes action strategies, while the backward learning flow autonomously collects effective data from historical task executions to facilitate continual system learning, enabling self-optimization. Experimental results show that the proposed framework improves the task success rate in dynamic disassembly scenarios from 81.68% to 100%, while reducing the average number of perception misjudgments from 3.389 to 1.128. This research provides a new paradigm for enhancing the robustness and adaptability of embodied intelligence in complex industrial environments. 

**Abstract (ZH)**: 基于神经符号任务与动作规划的持续学习框架：提升动态环境下单体智能系统的适应性 

---
# MEMBOT: Memory-Based Robot in Intermittent POMDP 

**Title (ZH)**: 基于记忆的间歇性部分可观马尔可夫决策过程机器人：MEMBOT 

**Authors**: Youzhi Liang, Eyan Noronha  

**Link**: [PDF](https://arxiv.org/pdf/2509.11225)  

**Abstract**: Robotic systems deployed in real-world environments often operate under con- ditions of partial and often intermittent observability, where sensor inputs may be noisy, occluded, or entirely unavailable due to failures or environmental con- straints. Traditional reinforcement learning (RL) approaches that assume full state observability are ill-equipped for such challenges. In this work, we introduce MEMBOT, a modular memory-based architecture designed to address intermittent partial observability in robotic control tasks. MEMBOT decouples belief inference from policy learning through a two-phase training process: an offline multi-task learning pretraining stage that learns a robust task-agnostic latent belief encoder using a reconstruction losses, followed by fine-tuning of task-specific policies using behavior cloning. The belief encoder, implemented as a state-space model (SSM) and a LSTM, integrates temporal sequences of observations and actions to infer latent state representations that persist even when observations are dropped. We train and evaluate MEMBOT on 10 robotic manipulation benchmark tasks from MetaWorld and Robomimic under varying rates of observation dropout. Results show that MEMBOT consistently outperforms both memoryless and naively recur- rent baselines, maintaining up to 80% of peak performance under 50% observation availability. These findings highlight the effectiveness of explicit belief modeling in achieving robust, transferable, and data-efficient policies for real-world partially observable robotic systems. 

**Abstract (ZH)**: 基于记忆的模块化架构MEMBOT在部分可观测环境下的机器人控制任务中有效应对间歇性部分可观测性 

---
# DreamNav: A Trajectory-Based Imaginative Framework for Zero-Shot Vision-and-Language Navigation 

**Title (ZH)**: DreamNav: 基于 trajectories 的想象框架，用于零样本视觉-语言导航 

**Authors**: Yunheng Wang, Yuetong Fang, Taowen Wang, Yixiao Feng, Yawen Tan, Shuning Zhang, Peiran Liu, Yiding Ji, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11197)  

**Abstract**: Vision-and-Language Navigation in Continuous Environments (VLN-CE), which links language instructions to perception and control in the real world, is a core capability of embodied robots. Recently, large-scale pretrained foundation models have been leveraged as shared priors for perception, reasoning, and action, enabling zero-shot VLN without task-specific training. However, existing zero-shot VLN methods depend on costly perception and passive scene understanding, collapsing control to point-level choices. As a result, they are expensive to deploy, misaligned in action semantics, and short-sighted in planning. To address these issues, we present DreamNav that focuses on the following three aspects: (1) for reducing sensory cost, our EgoView Corrector aligns viewpoints and stabilizes egocentric perception; (2) instead of point-level actions, our Trajectory Predictor favors global trajectory-level planning to better align with instruction semantics; and (3) to enable anticipatory and long-horizon planning, we propose an Imagination Predictor to endow the agent with proactive thinking capability. On VLN-CE and real-world tests, DreamNav sets a new zero-shot state-of-the-art (SOTA), outperforming the strongest egocentric baseline with extra information by up to 7.49\% and 18.15\% in terms of SR and SPL metrics. To our knowledge, this is the first zero-shot VLN method to unify trajectory-level planning and active imagination while using only egocentric inputs. 

**Abstract (ZH)**: Vision-and-Language Navigation in Continuous Environments: DreamNav 集成轨迹级规划与主动想象的零样本视觉-语言导航 

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
# Nav-R1: Reasoning and Navigation in Embodied Scenes 

**Title (ZH)**: Nav-R1: 身体化场景中的推理与导航 

**Authors**: Qingxiang Liu, Ting Huang, Zeyu Zhang, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10884)  

**Abstract**: Embodied navigation requires agents to integrate perception, reasoning, and action for robust interaction in complex 3D environments. Existing approaches often suffer from incoherent and unstable reasoning traces that hinder generalization across diverse environments, and difficulty balancing long-horizon semantic reasoning with low-latency control for real-time navigation. To address these challenges, we propose Nav-R1, an embodied foundation model that unifies reasoning in embodied environments. We first construct Nav-CoT-110K, a large-scale dataset of step-by-step Chains-of-Thought (CoT) for embodied tasks, which enables cold-start initialization with structured reasoning. Building on this foundation, we design a GRPO-based reinforcement learning framework with three complementary rewards: format, understanding, and navigation, to improve structural adherence, semantic grounding, and path fidelity. Furthermore, we introduce a Fast-in-Slow reasoning paradigm, decoupling deliberate semantic reasoning from low-latency reactive control for efficient yet coherent navigation. Extensive evaluations on embodied AI benchmarks demonstrate that Nav-R1 consistently outperforms strong baselines, with over 8% average improvement in reasoning and navigation performance. Real-world deployment on a mobile robot further validates its robustness under limited onboard resources. Code: this https URL. Website: this https URL. 

**Abstract (ZH)**: 基于知觉的导航要求智能体整合感知、推理和行动，以在复杂的3D环境中实现稳健的交互。现有的方法往往存在不连贯且不稳定的推理轨迹，这阻碍了在多种环境之间的泛化，并且难以在长远语义推理与低延迟控制之间进行权衡，以实现实时导航。为了解决这些挑战，我们提出了一种称为Nav-R1的基于知觉的基座模型，它统一了在基于知觉环境中的推理。我们首先构建了一个包含11万步骤推理链（CoT）的大规模数据集Nav-CoT-110K，这使得智能体可以从结构化的推理开始。在此基础上，我们设计了一个基于GRPO的强化学习框架，并引入了三种互补奖励：格式、理解、导航，以提高结构一致性、语义接地和路径精度。此外，我们引入了快速内在延时推理范式，将慎重的语义推理与低延迟的反应控制脱钩，以实现高效且连贯的导航。广泛的实验表明，Nav-R1在语义推理和导航性能上均显著优于强基线模型，平均提高超过8%。实地部署在移动机器人上进一步验证了其在有限的机载资源下的鲁棒性。代码：this https URL. 网站：this https URL。 

---
# Growing Perspectives: Modelling Embodied Perspective Taking and Inner Narrative Development Using Large Language Models 

**Title (ZH)**: 成长的视角：使用大规模语言模型建模具身换位思考和内在叙事发展 

**Authors**: Sabrina Patania, Luca Annese, Anna Lambiase, Anita Pellegrini, Tom Foulsham, Azzurra Ruggeri, Silvia Rossi, Silvia Serino, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2509.11868)  

**Abstract**: Language and embodied perspective taking are essential for human collaboration, yet few computational models address both simultaneously. This work investigates the PerspAct system [1], which integrates the ReAct (Reason and Act) paradigm with Large Language Models (LLMs) to simulate developmental stages of perspective taking, grounded in Selman's theory [2]. Using an extended director task, we evaluate GPT's ability to generate internal narratives aligned with specified developmental stages, and assess how these influence collaborative performance both qualitatively (action selection) and quantitatively (task efficiency). Results show that GPT reliably produces developmentally-consistent narratives before task execution but often shifts towards more advanced stages during interaction, suggesting that language exchanges help refine internal representations. Higher developmental stages generally enhance collaborative effectiveness, while earlier stages yield more variable outcomes in complex contexts. These findings highlight the potential of integrating embodied perspective taking and language in LLMs to better model developmental dynamics and stress the importance of evaluating internal speech during combined linguistic and embodied tasks. 

**Abstract (ZH)**: 语言和身体视角化对于人类协作至关重要，但很少有计算模型同时解决这两个方面。本研究探讨了PerspAct系统[1]，该系统将ReAct（推理与行动） paradigma与大型语言模型（LLMs）结合，模拟视角化发展的阶段，基于Selman的理论[2]。通过扩展导演任务，我们评估了GPT生成与指定发展阶段一致的内部叙事的能力，并评估这些叙事如何在定性（动作选择）和定量（任务效率）层面上影响协作性能。结果显示，GPT在任务执行前可靠地生成了发展一致的叙事，但在互动过程中往往会转向更高级的阶段，表明语言交流有助于细化内部表征。高级的发展阶段一般增强协作效果，而早期阶段在复杂情境中的结果更为多变。这些发现强调了在LLMs中结合身体视角化和语言的潜力，以更好地模拟发展动态，并强调了在联合语言和身体任务中评估内部言语的重要性。 

---
# Time to Play: Simulating Early-Life Animal Dynamics Enhances Robotics Locomotion Discovery 

**Title (ZH)**: 时间来玩：模拟早期生命动物动态增强机器人运动发现 

**Authors**: Paul Templier, Hannah Janmohamed, David Labonte, Antoine Cully  

**Link**: [PDF](https://arxiv.org/pdf/2509.11755)  

**Abstract**: Developmental changes in body morphology profoundly shape locomotion in animals, yet artificial agents and robots are typically trained under static physical parameters. Inspired by ontogenetic scaling of muscle power in biology, we propose Scaling Mechanical Output over Lifetime (SMOL), a novel curriculum that dynamically modulates robot actuator strength to mimic natural variations in power-to-weight ratio during growth and ageing. Integrating SMOL into the MAP-Elites quality-diversity framework, we vary the torque in standard robotics tasks to mimic the evolution of strength in animals as they grow up and as their body changes. Through comprehensive empirical evaluation, we show that the SMOL schedule consistently elevates both performance and diversity of locomotion behaviours across varied control scenarios, by allowing agents to leverage advantageous physics early on to discover skills that act as stepping stones when they reach their final standard body properties. Based on studies of the total power output in humans, we also implement the SMOL-Human schedule that models isometric body variations due to non-linear changes like puberty, and study its impact on robotics locomotion. 

**Abstract (ZH)**: 基于生命周期肌肉功率缩放的动态机器人训练方法 

---
# Agent-based Simulation for Drone Charging in an Internet of Things Environment System 

**Title (ZH)**: 基于代理的物联网环境下的无人机充电仿真研究 

**Authors**: Leonardo Grando, José Roberto Emiliano Leite, Edson Luiz Ursini  

**Link**: [PDF](https://arxiv.org/pdf/2509.10867)  

**Abstract**: This paper presents an agent-based simulation model for coordinating battery recharging in drone swarms, focusing on applications in Internet of Things (IoT) and Industry 4.0 environments. The proposed model includes a detailed description of the simulation methodology, system architecture, and implementation. One practical use case is explored: Smart Farming, highlighting how autonomous coordination strategies can optimize battery usage and mission efficiency in large-scale drone deployments. This work uses a machine learning technique to analyze the agent-based simulation sensitivity analysis output results. 

**Abstract (ZH)**: 基于代理的无人机群电池充电协调仿真模型：以物联网和工业4.0应用为例 

---
# InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts 

**Title (ZH)**: InternScenes: 一个具有真实布局的大规模可模拟室内场景数据集 

**Authors**: Weipeng Zhong, Peizhou Cao, Yichen Jin, Li Luo, Wenzhe Cai, Jingli Lin, Hanqing Wang, Zhaoyang Lyu, Tai Wang, Bo Dai, Xudong Xu, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10813)  

**Abstract**: The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community. 

**Abstract (ZH)**: InternScenes：一种新型的大型可模拟室内场景数据集 

---
# Synergetic Empowerment: Wireless Communications Meets Embodied Intelligence 

**Title (ZH)**: 协同赋能：无线通信与 embodied 智能的融合 

**Authors**: Hongtao Liang, Yihe Diao, YuHang Wu, Fuhui Zhou, Qihui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10481)  

**Abstract**: Wireless communication is evolving into an agent era, where large-scale agents with inherent embodied intelligence are not just users but active participants. The perfect combination of wireless communication and embodied intelligence can achieve a synergetic empowerment and greatly facilitate the development of agent communication. An overview of this synergetic empowerment is presented, framing it as a co-evolutionary process that transforms wireless communication from a simple utility into the digital nervous system of a collective intelligence, while simultaneously elevating isolated agents into a unified superorganism with emergent capabilities far exceeding individual contributions. Moreover, we elaborate how embodied intelligence and wireless communication mutually benefit each other through the lens of the perception-cognition-execution (PCE) loop, revealing a fundamental duality where each PCE stage both challenges network capacity and creates unprecedented opportunities for system-wide optimization. Furthermore, critical open issues and future research directions are identified. 

**Abstract (ZH)**: 无线通信正演进为智能代理时代，大规模具备内在本体智能的代理不仅是用户，更是积极参与者。无线通信与本体智能的完美结合能实现协同赋能，极大地促进智能代理通信的发展。本文概述了这种协同赋能的过程，将其视为一种共生演化过程，将无线通信从简单的工具转变为集体智能的数字神经系统，同时将孤立的代理提升为具备超越个体贡献的新兴能力的统一超有机体。此外，本文从感知-认知-执行（PCE）循环的视角阐明了本体智能与无线通信相互受益的基本二元性，每个PCE阶段既对网络容量构成挑战，又为系统级优化创造前所未有的机遇。最后，本文指出了关键的开放问题和未来研究方向。 

---
# Co-Alignment: Rethinking Alignment as Bidirectional Human-AI Cognitive Adaptation 

**Title (ZH)**: 共适应：重新思考 alignment 为双向人类-AI认知适应 

**Authors**: Yubo Li, Weiyi Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.12179)  

**Abstract**: Current AI alignment through RLHF follows a single directional paradigm that AI conforms to human preferences while treating human cognition as fixed. We propose a shift to co-alignment through Bidirectional Cognitive Alignment (BiCA), where humans and AI mutually adapt. BiCA uses learnable protocols, representation mapping, and KL-budget constraints for controlled co-evolution. In collaborative navigation, BiCA achieved 85.5% success versus 70.3% baseline, with 230% better mutual adaptation and 332% better protocol convergence. Emergent protocols outperformed handcrafted ones by 84%, while bidirectional adaptation unexpectedly improved safety (+23% out-of-distribution robustness). The 46% synergy improvement demonstrates optimal collaboration exists at the intersection, not union, of human and AI capabilities, validating the shift from single-directional to co-alignment paradigms. 

**Abstract (ZH)**: 当前通过RLHF实现AI对齐遵循单向范式，其中AI遵从人类偏好而将人类认知视为固定不变。我们提出转向双向认知对齐（BiCA）范式，其中人类和AI相互适应。BiCA使用可学习的协议、表示映射和KL预算约束来实现受控共生进化。在协作导航中，BiCA的成功率达到了85.5%，而基线为70.3%，显示出230%更好的相互适应和332%更好的协议收敛。自动生成的协议在性能上比手工设计的协议高出84%，而双向适应意外地提高了安全性（分布外鲁棒性提高23%）。46%的协同效应改善证明了人类和AI能力的最佳协作存在于两者的交集而非并集中，验证了从单向范式转向双向对齐范式的转变。 

---
# Neuro-Symbolic Agents with Modal Logic for Autonomous Diagnostics 

**Title (ZH)**: 模态逻辑驱动的神经符号代理在自主诊断中的应用 

**Authors**: Antonin Sulc, Thorsten Hellert  

**Link**: [PDF](https://arxiv.org/pdf/2509.11943)  

**Abstract**: The development of intelligent agents, particularly those powered by language models (LMs), has shown the critical role in various environments that require intelligent and autonomous decision. Environments are not passive testing grounds and they represent the data required for agents to learn and exhibit very challenging conditions that require adaptive, complex and autonomous capacity to make decisions. While the paradigm of scaling models and datasets has led to remarkable emergent capabilities, we argue that scaling the structure, fidelity, and logical consistency of agent reasoning within these environments is a crucial, yet underexplored, dimension of AI research. This paper introduces a neuro-symbolic multi-agent architecture where the belief states of individual agents are formally represented as Kripke models. This foundational choice enables them to reason about known concepts of \emph{possibility} and \emph{necessity} using the formal language of modal logic. In this work, we use of immutable, domain-specific knowledge to make infere information, which is encoded as logical constraints essential for proper diagnosis. In the proposed model, we show constraints that actively guide the hypothesis generation of LMs, effectively preventing them from reaching physically or logically untenable conclusions. In a high-fidelity simulated particle accelerator environment, our system successfully diagnoses complex, cascading failures by combining the powerful semantic intuition of LMs with the rigorous, verifiable validation of modal logic and a factual world model and showcasing a viable path toward more robust, reliable, and verifiable autonomous agents. 

**Abstract (ZH)**: 基于语言模型的智能代理的发展展示了其在需要智能自主决策的各种环境中的关键作用。环境不仅是被动的测试场所，它们还代表了代理学习所需的必要数据，并且环境中的条件极其挑战性，要求代理具备适应性、复杂性和自主决策能力。虽然模型和数据集的扩展增强了显著的 emergent 能力，但我们认为，在这些环境中扩展代理推理的结构、忠实度和逻辑一致性是人工智能研究中一个关键但尚未充分探索的维度。本文介绍了一种神经符号多代理架构，其中单个代理的信念状态被形式上表示为克里普克模型。这一基础选择使它们能够使用模态逻辑的形式语言来推理关于可能性和必然性的已知概念。在本文中，我们利用不可变的领域特定知识进行推理，这些知识被编码为诊断所需的重要逻辑约束。在所提出模型中，我们展示了约束条件能够积极引导语言模型的假设生成，有效地防止它们得出物理上或逻辑上不可行的结论。在高保真模拟粒子加速器环境中，我们的系统通过结合语言模型的强大语义直觉、模态逻辑的严格可验证验证和事实世界模型，成功诊断了复杂的级联故障，展示了一条通往更 robust、可靠和可验证的自主代理的有效途径。 

---
# Maestro: Self-Improving Text-to-Image Generation via Agent Orchestration 

**Title (ZH)**: Maestro：通过代理协调实现自我提升的文本到图像生成 

**Authors**: Xingchen Wan, Han Zhou, Ruoxi Sun, Hootan Nakhost, Ke Jiang, Rajarishi Sinha, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2509.10704)  

**Abstract**: Text-to-image (T2I) models, while offering immense creative potential, are highly reliant on human intervention, posing significant usability challenges that often necessitate manual, iterative prompt engineering over often underspecified prompts. This paper introduces Maestro, a novel self-evolving image generation system that enables T2I models to autonomously self-improve generated images through iterative evolution of prompts, using only an initial prompt. Maestro incorporates two key innovations: 1) self-critique, where specialized multimodal LLM (MLLM) agents act as 'critics' to identify weaknesses in generated images, correct for under-specification, and provide interpretable edit signals, which are then integrated by a 'verifier' agent while preserving user intent; and 2) self-evolution, utilizing MLLM-as-a-judge for head-to-head comparisons between iteratively generated images, eschewing problematic images, and evolving creative prompt candidates that align with user intents. Extensive experiments on complex T2I tasks using black-box models demonstrate that Maestro significantly improves image quality over initial prompts and state-of-the-art automated methods, with effectiveness scaling with more advanced MLLM components. This work presents a robust, interpretable, and effective pathway towards self-improving T2I generation. 

**Abstract (ZH)**: 基于文本到图像的Maestro自我进化图像生成系统 

---
# Approaches to Analysis and Design of AI-Based Autonomous Vehicles 

**Title (ZH)**: 基于AI的自主车辆分析与设计方法 

**Authors**: Tao Yan, Zheyu Zhang, Jingjing Jiang, Wen-Hua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12169)  

**Abstract**: Artificial intelligence (AI) models are becoming key components in an autonomous vehicle (AV), especially in handling complicated perception tasks. However, closing the loop through AI-based feedback may pose significant risks on reliability of autonomous driving due to very limited understanding about the mechanism of AI-driven perception processes. To overcome it, this paper aims to develop tools for modeling, analysis, and synthesis for a class of AI-based AV; in particular, their closed-loop properties, e.g., stability, robustness, and performance, are rigorously studied in the statistical sense. First, we provide a novel modeling means for the AI-driven perception processes by looking at their error characteristics. Specifically, three fundamental AI-induced perception uncertainties are recognized and modeled by Markov chains, Gaussian processes, and bounded disturbances, respectively. By means of that, the closed-loop stochastic stability (SS) is established in the sense of mean square, and then, an SS control synthesis method is presented within the framework of linear matrix inequalities (LMIs). Besides the SS properties, the robustness and performance of AI-based AVs are discussed in terms of a stochastic guaranteed cost, and criteria are given to test the robustness level of an AV when in the presence of AI-induced uncertainties. Furthermore, the stochastic optimal guaranteed cost control is investigated, and an efficient design procedure is developed innovatively based on LMI techniques and convex optimization. Finally, to illustrate the effectiveness, the developed results are applied to an example of car following control, along with extensive simulation. 

**Abstract (ZH)**: 基于人工智能的自动驾驶车辆建模、分析与综合方法：统计意义下的闭环稳定性和鲁棒性研究 

---
# EthicsMH: A Pilot Benchmark for Ethical Reasoning in Mental Health AI 

**Title (ZH)**: EthicsMH：心理健康AI伦理推理的试点基准 

**Authors**: Sai Kartheek Reddy Kasu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11648)  

**Abstract**: The deployment of large language models (LLMs) in mental health and other sensitive domains raises urgent questions about ethical reasoning, fairness, and responsible alignment. Yet, existing benchmarks for moral and clinical decision-making do not adequately capture the unique ethical dilemmas encountered in mental health practice, where confidentiality, autonomy, beneficence, and bias frequently intersect. To address this gap, we introduce Ethical Reasoning in Mental Health (EthicsMH), a pilot dataset of 125 scenarios designed to evaluate how AI systems navigate ethically charged situations in therapeutic and psychiatric contexts. Each scenario is enriched with structured fields, including multiple decision options, expert-aligned reasoning, expected model behavior, real-world impact, and multi-stakeholder viewpoints. This structure enables evaluation not only of decision accuracy but also of explanation quality and alignment with professional norms. Although modest in scale and developed with model-assisted generation, EthicsMH establishes a task framework that bridges AI ethics and mental health decision-making. By releasing this dataset, we aim to provide a seed resource that can be expanded through community and expert contributions, fostering the development of AI systems capable of responsibly handling some of society's most delicate decisions. 

**Abstract (ZH)**: 大型语言模型在心理健康和其他敏感领域中的部署引发了关于伦理推理、公平性和负责任对齐的迫切问题。然而，现有针对道德和临床决策的基准测试未能充分捕捉到心理健康实践中的独特伦理困境，其中保密性、自主权、有益性以及偏见经常相互交织。为弥补这一差距，我们提出了心理健康中的伦理推理（EthicsMH）数据集，该数据集包含125种场景，旨在评估AI系统如何在心理治疗和精神病学背景下应对伦理紧张的情境。每个场景都包括结构化的字段，如多种决策选项、专家一致的推理、预期模型行为、现实世界影响和多利益相关者的视角。这种结构不仅能够评估决策准确性，还能够评估解释质量和与专业规范的契合度。尽管规模较小且通过模型辅助生成，EthicsMH仍建立了一个连接AI伦理与心理健康决策的任务框架。通过发布此数据集，我们旨在提供一个种子资源，可以通过社区和专家的贡献进行扩展，促进开发能够负责任地处理社会上一些最微妙决策的AI系统。 

---
# UI-S1: Advancing GUI Automation via Semi-online Reinforcement Learning 

**Title (ZH)**: UI-S1: 基于半在线强化学习的GUI自动化先进方法 

**Authors**: Zhengxi Lu, Jiabo Ye, Fei Tang, Yongliang Shen, Haiyang Xu, Ziwei Zheng, Weiming Lu, Ming Yan, Fei Huang, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11543)  

**Abstract**: Graphical User Interface (GUI) agents have demonstrated remarkable progress in automating complex user interface interactions through reinforcement learning. However, current approaches face a fundamental dilemma: offline RL enables stable training on pre-collected trajectories, but struggles with multi-step task execution for lack of trajectory-level reward signals; online RL captures these signals through environment interaction, but suffers from sparse rewards and prohibitive deployment costs. To address it, we present Semi-online Reinforcement Learning, a novel paradigm that simulates online RL on offline trajectories. During each rollout process, we preserve the original model output within the multi-turn dialogue, where a Patch Module adaptively recovers the divergence between rollout and expert trajectories. To capture long-term training signals, Semi-online RL introduces discounted future returns into the reward computation and optimizes the policy with weighted step-level and episode-level advantages. We further introduce Semi-Online Performance (SOP), a metric that aligns better with true online performance, serving as a practical and effective proxy for real-world evaluation. Experiments show that ours Semi-online RL achieves SOTA performance among 7B models across four dynamic benchmarks, with significant gains over the base model (e.g., +12.0% on AndroidWorld, +23.8% on AITW), demonstrating significant progress in bridging the gap between offline training efficiency and online multi-turn reasoning. The code is available at this https URL. 

**Abstract (ZH)**: 半在线强化学习：一种在离线轨迹上模拟在线RL的新范式 

---
# Designing and Evaluating a Conversational Agent for Early Detection of Alzheimer's Disease and Related Dementias 

**Title (ZH)**: 设计并评估一种用于早期检测阿尔茨海默病及相关痴呆症的对话代理系统 

**Authors**: Andrew G. Breithaupt, Nayoung Choi, James D. Finch, Jeanne M. Powell, Arin L. Nelson, Oz A. Alon, Howard J. Rosen, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11478)  

**Abstract**: Early detection of Alzheimer's disease and related dementias (ADRD) is critical for timely intervention, yet most diagnoses are delayed until advanced stages. While comprehensive patient narratives are essential for accurate diagnosis, prior work has largely focused on screening studies that classify cognitive status from interactions rather than supporting the diagnostic process. We designed voice-interactive conversational agents, leveraging large language models (LLMs), to elicit narratives relevant to ADRD from patients and informants. We evaluated the agent with 30 adults with suspected ADRD through conversation analysis (n=30), user surveys (n=19), and clinical validation against blinded specialist interviews (n=24). Symptoms detected by the agent aligned well with those identified by specialists across symptoms. Users appreciated the agent's patience and systematic questioning, which supported engagement and expression of complex, hard-to-describe experiences. This preliminary work suggests conversational agents may serve as structured front-end tools for dementia assessment, highlighting interaction design considerations in sensitive healthcare contexts. 

**Abstract (ZH)**: early检测阿尔茨海默病及相关痴呆症对于及时干预至关重要，但大多数诊断直到疾病晚期才进行。虽然全面的患者叙事对于准确诊断至关重要，但此前的工作主要集中在筛查研究上，这些研究通过交流分类认知状态，而非支持诊断过程。我们设计了语音交互对话代理，利用大规模语言模型（LLMs），以从患者和观察者那里引出与阿尔茨海默病相关的历史，并提供诊断信息。我们通过对话分析（n=30）、用户调查（n=19）以及与盲审专家访谈的临床验证（n=24）评估了该代理。代理检测到的症状与专家识别的症状高度一致。用户赞赏代理的耐心和系统化的问题，这有助于促进患者的参与和表达复杂的、难以描述的经历。初步研究表明，对话代理可能作为痴呆评估的结构化前端工具发挥作用，并强调在敏感的医疗保健环境中进行交互设计时需要考虑的因素。 

---
# Detecting Model Drifts in Non-Stationary Environment Using Edit Operation Measures 

**Title (ZH)**: 使用编辑操作度量检测非稳态环境中模型漂移 

**Authors**: Chang-Hwan Lee, Alexander Shim  

**Link**: [PDF](https://arxiv.org/pdf/2509.11367)  

**Abstract**: Reinforcement learning (RL) agents typically assume stationary environment dynamics. Yet in real-world applications such as healthcare, robotics, and finance, transition probabilities or reward functions may evolve, leading to model drift. This paper proposes a novel framework to detect such drifts by analyzing the distributional changes in sequences of agent behavior. Specifically, we introduce a suite of edit operation-based measures to quantify deviations between state-action trajectories generated under stationary and perturbed conditions. Our experiments demonstrate that these measures can effectively distinguish drifted from non-drifted scenarios, even under varying levels of noise, providing a practical tool for drift detection in non-stationary RL environments. 

**Abstract (ZH)**: 强化学习（RL）代理通常假设环境动态是稳定的。但在医疗保健、机器人技术和金融等实际应用中，转换概率或奖励函数可能会发生变化，导致模型漂移。本文提出了一种新型框架，通过分析代理行为序列上的分布变化来检测此类漂移。具体而言，我们引入了一系列基于编辑操作的度量来量化在稳定和扰动条件下生成的状态-动作轨迹之间的偏差。我们的实验表明，这些度量可以在不同程度的噪声下有效地区分漂移和非漂移场景，提供了一种在非稳定RL环境中进行漂移检测的实用工具。 

---
# Gradient Free Deep Reinforcement Learning With TabPFN 

**Title (ZH)**: 无需推理直接输出：

Gradient Free Deep Reinforcement Learning With TabPFN 

**Authors**: David Schiff, Ofir Lindenbaum, Yonathan Efroni  

**Link**: [PDF](https://arxiv.org/pdf/2509.11259)  

**Abstract**: Gradient based optimization is fundamental to most modern deep reinforcement learning algorithms, however, it introduces significant sensitivity to hyperparameters, unstable training dynamics, and high computational costs. We propose TabPFN RL, a novel gradient free deep RL framework that repurposes the meta trained transformer TabPFN as a Q function approximator. Originally developed for tabular classification, TabPFN is a transformer pre trained on millions of synthetic datasets to perform inference on new unseen datasets via in context learning. Given an in context dataset of sample label pairs and new unlabeled data, it predicts the most likely labels in a single forward pass, without gradient updates or task specific fine tuning. We use TabPFN to predict Q values using inference only, thereby eliminating the need for back propagation at both training and inference. To cope with the model's fixed context budget, we design a high reward episode gate that retains only the top 5% of trajectories. Empirical evaluations on the Gymnasium classic control suite demonstrate that TabPFN RL matches or surpasses Deep Q Network on CartPole v1, MountainCar v0, and Acrobot v1, without applying gradient descent or any extensive hyperparameter tuning. We discuss the theoretical aspects of how bootstrapped targets and non stationary visitation distributions violate the independence assumptions encoded in TabPFN's prior, yet the model retains a surprising generalization capacity. We further formalize the intrinsic context size limit of in context RL algorithms and propose principled truncation strategies that enable continual learning when the context is full. Our results establish prior fitted networks such as TabPFN as a viable foundation for fast and computationally efficient RL, opening new directions for gradient free RL with large pre trained transformers. 

**Abstract (ZH)**: 基于梯度的优化是大多数现代深度强化学习算法的基础，然而它引入了对超参数的高度敏感性、不稳定的训练动态和高昂的计算成本。我们提出了TabPFN RL，一种新的无梯度深度强化学习框架，利用元训练变压器TabPFN作为Q函数近似器。TabPFN最初用于表格分类，它是在数百万个合成数据集上预训练的变压器，通过上下文学习进行新未见数据集的推理。给定一个包含样本标签对的上下文数据集和新的未标记数据，它可以在一次前向传递中预测最可能的标签，而不需要梯度更新或任务特定的微调。我们仅使用TabPFN进行推理来预测Q值，从而在训练和推理时都消除了后向传播的需要。为应对模型固定的上下文预算，我们设计了一个高奖励时间步门控机制，仅保留最顶级的5%的轨迹。在Gymnasium经典控制任务集上的实验评估表明，TabPFN RL 在CartPole v1、MountainCar v0 和 Acrobot v1 上的性能与深度Q网络相当或超越，无需应用梯度下降或任何广泛的超参数调整。我们讨论了采用自助目标和非稳定访问分布如何违反TabPFN先验中编码的独立性假设，尽管模型保留了令人惊讶的泛化能力。我们进一步形式化了上下文强化学习算法固有的上下文大小限制，并提出了原则性的截断策略，以在上下文充满时实现持续学习。我们的结果确立了先验拟合网络（如TabPFN）作为快速和计算高效的RL的基础，并为使用大型预训练变压器进行无梯度RL开辟了新方向。 

---
# ViSTR-GP: Online Cyberattack Detection via Vision-to-State Tensor Regression and Gaussian Processes in Automated Robotic Operations 

**Title (ZH)**: ViSTR-GP：基于视觉到状态张量回归和高斯过程的自动化机器人操作中的在线网络攻击检测 

**Authors**: Navid Aftabi, Philip Samaha, Jin Ma, Long Cheng, Ramy Harik, Dan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.10948)  

**Abstract**: Industrial robotic systems are central to automating smart manufacturing operations. Connected and automated factories face growing cybersecurity risks that can potentially cause interruptions and damages to physical operations. Among these attacks, data-integrity attacks often involve sophisticated exploitation of vulnerabilities that enable an attacker to access and manipulate the operational data and are hence difficult to detect with only existing intrusion detection or model-based detection. This paper addresses the challenges in utilizing existing side-channels to detect data-integrity attacks in robotic manufacturing processes by developing an online detection framework, ViSTR-GP, that cross-checks encoder-reported measurements against a vision-based estimate from an overhead camera outside the controller's authority. In this framework, a one-time interactive segmentation initializes SAM-Track to generate per-frame masks. A low-rank tensor-regression surrogate maps each mask to measurements, while a matrix-variate Gaussian process models nominal residuals, capturing temporal structure and cross-joint correlations. A frame-wise test statistic derived from the predictive distribution provides an online detector with interpretable thresholds. We validate the framework on a real-world robotic testbed with synchronized video frame and encoder data, collecting multiple nominal cycles and constructing replay attack scenarios with graded end-effector deviations. Results on the testbed indicate that the proposed framework recovers joint angles accurately and detects data-integrity attacks earlier with more frequent alarms than all baselines. These improvements are most evident in the most subtle attacks. These results show that plants can detect data-integrity attacks by adding an independent physical channel, bypassing the controller's authority, without needing complex instrumentation. 

**Abstract (ZH)**: 工业机器人系统在智能制造自动化中占据核心地位。联网和自动化工厂面临日益增长的网络安全风险，这些风险可能导致物理操作中断和损坏。在这种情况下，数据完整性攻击通常涉及利用复杂的漏洞，使攻击者能够访问并操控操作数据，因此仅依赖现有的入侵检测或模型检测方法难以检测。本文通过开发一种在线检测框架ViSTR-GP，解决了利用现有旁路信道检测机器人制造过程中数据完整性攻击的挑战。该框架通过超出控制器权限范围的上方摄像头进行的视觉估计，与编码器报告的测量值进行交叉验证。该框架首先进行一次互动分割以初始化SAM-Track生成每帧的掩码。低秩张量回归代理将每个掩码映射到测量值，同时使用矩阵多元高斯过程模型名义残差，捕捉时间结构和关节间关联性。从预测分布中得出的帧级测试统计量为在线检测器提供了可解释的阈值。我们在同步视频帧和编码器数据的现实世界机器人测试台上进行了验证，并收集了多个名义周期，构建了带有分级末端执行器偏差的回放攻击场景。测试台结果显示，所提出框架能够更准确地恢复关节角度，并比所有基线更早地检测到数据完整性攻击，频率更高。这些改进在最为微妙的攻击中最明显。这些结果表明，通过添加一个独立的物理通道，绕过控制器的权限，工厂可以检测到数据完整性攻击，无需复杂仪器。 

---
# CultureSynth: A Hierarchical Taxonomy-Guided and Retrieval-Augmented Framework for Cultural Question-Answer Synthesis 

**Title (ZH)**: CultureSynth：一种层次结构分类指导和检索增强的文化问答合成框架 

**Authors**: Xinyu Zhang, Pei Zhang, Shuang Luo, Jialong Tang, Yu Wan, Baosong Yang, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10886)  

**Abstract**: Cultural competence, defined as the ability to understand and adapt to multicultural contexts, is increasingly vital for large language models (LLMs) in global environments. While several cultural benchmarks exist to assess LLMs' cultural competence, current evaluations suffer from fragmented taxonomies, domain specificity, and heavy reliance on manual data annotation. To address these limitations, we introduce CultureSynth, a novel framework comprising (1) a comprehensive hierarchical multilingual cultural taxonomy covering 12 primary and 130 secondary topics, and (2) a Retrieval-Augmented Generation (RAG)-based methodology leveraging factual knowledge to synthesize culturally relevant question-answer pairs. The CultureSynth-7 synthetic benchmark contains 19,360 entries and 4,149 manually verified entries across 7 languages. Evaluation of 14 prevalent LLMs of different sizes reveals clear performance stratification led by ChatGPT-4o-Latest and Qwen2.5-72B-Instruct. The results demonstrate that a 3B-parameter threshold is necessary for achieving basic cultural competence, models display varying architectural biases in knowledge processing, and significant geographic disparities exist across models. We believe that CultureSynth offers a scalable framework for developing culturally aware AI systems while reducing reliance on manual annotation\footnote{Benchmark is available at this https URL.}. 

**Abstract (ZH)**: 文化适应性，定义为理解并适应多元文化环境的能力，在全球环境中对大型语言模型（LLMs）愈发重要。尽管存在多种文化基准以评估LLMs的文化适应性，但当前评估仍存在分类体系碎片化、领域特定性以及对人工数据标注的高依赖性等问题。为解决这些问题，我们提出了CultureSynth这一创新框架，该框架包括（1）涵盖12个主要和130个次要主题的全面分层多语言文化分类体系，以及（2）一种基于检索增强生成（RAG）的方法论，利用事实性知识合成文化相关的问题-答案对。CultureSynth-7合成基准包含7种语言的19,360条记录和4,149条人工验证的记录。对14种不同规模的主流LLMs进行评估显示，性能存在明显分层，以ChatGPT-4o-Latest和Qwen2.5-72B-Instruct表现领先。结果表明，基本文化适应性需要至少30亿参数，模型在知识处理上显示出不同的架构偏见，并且在不同模型间存在显著的地理差异。我们相信，CultureSynth提供了一个可扩展的框架，用于开发具有文化意识的AI系统，同时减少对人工标注的依赖。 

---
# ASL360: AI-Enabled Adaptive Streaming of Layered 360° Video over UAV-assisted Wireless Networks 

**Title (ZH)**: ASL360: 基于无人机辅助无线网络的AI使能分层360°视频自适应流传输 

**Authors**: Alireza Mohammadhosseini, Jacob Chakareski, Nicholas Mastronarde  

**Link**: [PDF](https://arxiv.org/pdf/2509.10544)  

**Abstract**: We propose ASL360, an adaptive deep reinforcement learning-based scheduler for on-demand 360° video streaming to mobile VR users in next generation wireless networks. We aim to maximize the overall Quality of Experience (QoE) of the users served over a UAV-assisted 5G wireless network. Our system model comprises a macro base station (MBS) and a UAV-mounted base station which both deploy mm-Wave transmission to the users. The 360° video is encoded into dependent layers and segmented tiles, allowing a user to schedule downloads of each layer's segments. Furthermore, each user utilizes multiple buffers to store the corresponding video layer's segments. We model the scheduling decision as a Constrained Markov Decision Process (CMDP), where the agent selects Base or Enhancement layers to maximize the QoE and use a policy gradient-based method (PPO) to find the optimal policy. Additionally, we implement a dynamic adjustment mechanism for cost components, allowing the system to adaptively balance and prioritize the video quality, buffer occupancy, and quality change based on real-time network and streaming session conditions. We demonstrate that ASL360 significantly improves the QoE, achieving approximately 2 dB higher average video quality, 80% lower average rebuffering time, and 57% lower video quality variation, relative to competitive baseline methods. Our results show the effectiveness of our layered and adaptive approach in enhancing the QoE in immersive videostreaming applications, particularly in dynamic and challenging network environments. 

**Abstract (ZH)**: ASL360：基于自适应深度强化学习的UAV辅助5G无线网络中移动VR用户按需360°视频流媒体调度器 

---
# LogGuardQ: A Cognitive-Enhanced Reinforcement Learning Framework for Cybersecurity Anomaly Detection in Security Logs 

**Title (ZH)**: LogGuardQ：一种增强认知的强化学习框架，用于安全日志中的网络安全异常检测 

**Authors**: Umberto Gonçalves de Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2509.10511)  

**Abstract**: Reinforcement learning (RL) has transformed sequential decision-making, but traditional algorithms like Deep Q-Networks (DQNs) and Proximal Policy Optimization (PPO) often struggle with efficient exploration, stability, and adaptability in dynamic environments. This study presents LogGuardQ (Adaptive Log Guard with Cognitive enhancement), a novel framework that integrates a dual-memory system inspired by human cognition and adaptive exploration strategies driven by temperature decay and curiosity. Evaluated on a dataset of 1,000,000 simulated access logs with 47.9% anomalies over 20,000 episodes, LogGuardQ achieves a 96.0% detection rate (versus 93.0% for DQN and 47.1% for PPO), with precision of 0.4776, recall of 0.9996, and an F1-score of 0.6450. The mean reward is 20.34 \pm 44.63 across all episodes (versus 18.80 \pm 43.98 for DQN and -0.17 \pm 23.79 for PPO), with an average of 5.0 steps per episode (constant across models). Graphical analyses, including learning curves smoothed with a Savgol filter (window=501, polynomial=2), variance trends, action distributions, and cumulative detections, demonstrate LogGuardQ's superior stability and efficiency. Statistical tests (Mann-Whitney U) confirm significant performance advantages (e.g., p = 0.0002 vs. DQN with negligible effect size, p < 0.0001 vs. PPO with medium effect size, and p < 0.0001 for DQN vs. PPO with small effect size). By bridging cognitive science and RL, LogGuardQ offers a scalable approach to adaptive learning in uncertain environments, with potential applications in cybersecurity, intrusion detection, and decision-making under uncertainty. 

**Abstract (ZH)**: 基于认知增强的LogGuardQ：一种适应动态环境的强化学习框架 

---
