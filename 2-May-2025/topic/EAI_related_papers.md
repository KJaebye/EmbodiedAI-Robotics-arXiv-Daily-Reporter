# Robotic Visual Instruction 

**Title (ZH)**: 机器视觉指令 

**Authors**: Yanbang Li, Ziyang Gong, Haoyang Li, Haoyang Li, Xiaoqi Huang, Haolan Kang, Guangping Bai, Xianzheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00693)  

**Abstract**: Recently, natural language has been the primary medium for human-robot interaction. However, its inherent lack of spatial precision for robotic control introduces challenges such as ambiguity and verbosity. To address these limitations, we introduce the Robotic Visual Instruction (RoVI), a novel paradigm to guide robotic tasks through an object-centric, hand-drawn symbolic representation. RoVI effectively encodes spatial-temporal information into human-interpretable visual instructions through 2D sketches, utilizing arrows, circles, colors, and numbers to direct 3D robotic manipulation. To enable robots to understand RoVI better and generate precise actions based on RoVI, we present Visual Instruction Embodied Workflow (VIEW), a pipeline formulated for RoVI-conditioned policies. This approach leverages Vision-Language Models (VLMs) to interpret RoVI inputs, decode spatial and temporal constraints from 2D pixel space via keypoint extraction, and then transform them into executable 3D action sequences. We additionally curate a specialized dataset of 15K instances to fine-tune small VLMs for edge deployment, enabling them to effectively learn RoVI capabilities. Our approach is rigorously validated across 11 novel tasks in both real and simulated environments, demonstrating significant generalization capability. Notably, VIEW achieves an 87.5% success rate in real-world scenarios involving unseen tasks that feature multi-step actions, with disturbances, and trajectory-following requirements. Code and Datasets in this paper will be released soon. 

**Abstract (ZH)**: Recent进展：基于视觉指令的机器人任务指导框架 

---
# Multi-Constraint Safe Reinforcement Learning via Closed-form Solution for Log-Sum-Exp Approximation of Control Barrier Functions 

**Title (ZH)**: 基于 Log-Sum-Exp 近似控制屏障函数的闭形式解多约束安全强化学习 

**Authors**: Chenggang Wang, Xinyi Wang, Yutong Dong, Lei Song, Xinping Guan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00671)  

**Abstract**: The safety of training task policies and their subsequent application using reinforcement learning (RL) methods has become a focal point in the field of safe RL. A central challenge in this area remains the establishment of theoretical guarantees for safety during both the learning and deployment processes. Given the successful implementation of Control Barrier Function (CBF)-based safety strategies in a range of control-affine robotic systems, CBF-based safe RL demonstrates significant promise for practical applications in real-world scenarios. However, integrating these two approaches presents several challenges. First, embedding safety optimization within the RL training pipeline requires that the optimization outputs be differentiable with respect to the input parameters, a condition commonly referred to as differentiable optimization, which is non-trivial to solve. Second, the differentiable optimization framework confronts significant efficiency issues, especially when dealing with multi-constraint problems. To address these challenges, this paper presents a CBF-based safe RL architecture that effectively mitigates the issues outlined above. The proposed approach constructs a continuous AND logic approximation for the multiple constraints using a single composite CBF. By leveraging this approximation, a close-form solution of the quadratic programming is derived for the policy network in RL, thereby circumventing the need for differentiable optimization within the end-to-end safe RL pipeline. This strategy significantly reduces computational complexity because of the closed-form solution while maintaining safety guarantees. Simulation results demonstrate that, in comparison to existing approaches relying on differentiable optimization, the proposed method significantly reduces training computational costs while ensuring provable safety throughout the training process. 

**Abstract (ZH)**: 使用强化学习方法训练任务策略及其后续应用的安全性已成为安全强化学习领域的焦点。在这一领域中，确保学习和部署过程中的安全性理论保证的建立仍然是一个核心挑战。鉴于控制 Barrier 函数（CBF）方法在多种控制仿射机器人系统中成功实施，基于 CBF 的安全强化学习在实际应用场景中展现出巨大的潜力。然而，将这两种方法结合在一起存在若干挑战。首先，将安全优化嵌入到 RL 训练管道中需要优化输出对输入参数可微分，这一条件通常被称为可微优化，解决这一问题并不容易。其次，可微优化框架在处理多约束问题时存在显著的效率问题。为了解决这些挑战，本文提出了一种基于 CBF 的安全强化学习架构，有效缓解了上述问题。所提出的方案通过单个复合 CBF 构建了多约束的连续 AND 逻辑近似。利用这一近似，我们为 RL 中的策略网络推导出了二次规划的闭式解，从而绕过了端到端安全强化学习管道中的可微优化需求。这一策略由于闭式解的存在显著降低了计算复杂度，同时保持了安全保证。仿真结果表明，与依赖于可微优化的方法相比，所提出的方法在确保整个训练过程可证明安全性的前提下，显著降低了训练的计算成本。 

---
# A Finite-State Controller Based Offline Solver for Deterministic POMDPs 

**Title (ZH)**: 基于有限状态控制器的离线求解器：确定性POMDP问题 

**Authors**: Alex Schutz, Yang You, Matias Mattamala, Ipek Caliskanelli, Bruno Lacerda, Nick Hawes  

**Link**: [PDF](https://arxiv.org/pdf/2505.00596)  

**Abstract**: Deterministic partially observable Markov decision processes (DetPOMDPs) often arise in planning problems where the agent is uncertain about its environmental state but can act and observe deterministically. In this paper, we propose DetMCVI, an adaptation of the Monte Carlo Value Iteration (MCVI) algorithm for DetPOMDPs, which builds policies in the form of finite-state controllers (FSCs). DetMCVI solves large problems with a high success rate, outperforming existing baselines for DetPOMDPs. We also verify the performance of the algorithm in a real-world mobile robot forest mapping scenario. 

**Abstract (ZH)**: 确定性部分可观测马尔可夫决策过程（DetPOMDPs） often arise in planning problems where the agent is uncertain about its environmental state but can act and observe deterministically. 在这种情况下，我们提出了DetMCVI算法，它是Monte Carlo Value Iteration (MCVI)算法对DetPOMDPs的适应，用于构建有限状态控制器（FSCs）形式的策略。DetMCVI以高成功率解决了大规模问题，超过了现有DetPOMDP基准算法。我们还在一个实际的移动机器人森林制图场景中验证了该算法的性能。 

---
# ParkDiffusion: Heterogeneous Multi-Agent Multi-Modal Trajectory Prediction for Automated Parking using Diffusion Models 

**Title (ZH)**: ParkDiffusion：基于扩散模型的自动化停车中的异构多agent多模态轨迹预测 

**Authors**: Jiarong Wei, Niclas Vödisch, Anna Rehr, Christian Feist, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.00586)  

**Abstract**: Automated parking is a critical feature of Advanced Driver Assistance Systems (ADAS), where accurate trajectory prediction is essential to bridge perception and planning modules. Despite its significance, research in this domain remains relatively limited, with most existing studies concentrating on single-modal trajectory prediction of vehicles. In this work, we propose ParkDiffusion, a novel approach that predicts the trajectories of both vehicles and pedestrians in automated parking scenarios. ParkDiffusion employs diffusion models to capture the inherent uncertainty and multi-modality of future trajectories, incorporating several key innovations. First, we propose a dual map encoder that processes soft semantic cues and hard geometric constraints using a two-step cross-attention mechanism. Second, we introduce an adaptive agent type embedding module, which dynamically conditions the prediction process on the distinct characteristics of vehicles and pedestrians. Third, to ensure kinematic feasibility, our model outputs control signals that are subsequently used within a kinematic framework to generate physically feasible trajectories. We evaluate ParkDiffusion on the Dragon Lake Parking (DLP) dataset and the Intersections Drone (inD) dataset. Our work establishes a new baseline for heterogeneous trajectory prediction in parking scenarios, outperforming existing methods by a considerable margin. 

**Abstract (ZH)**: 自动化停车是高级驾驶辅助系统（ADAS）的关键功能，其中准确的轨迹预测对于连接感知模块和规划模块至关重要。尽管其重要性不言而喻，但该领域的研究仍相对有限，大多数现有研究集中在车辆单一模态轨迹预测上。本文提出ParkDiffusion，一种新颖的方法，用于预测自动化停车场景中车辆和行人的轨迹。ParkDiffusion利用扩散模型捕捉未来轨迹的内在不确定性与多模态性，并包含几个关键创新。首先，我们提出了一种双地图编码器，通过两步交叉注意力机制处理软语义线索和硬几何约束。其次，我们引入了一种自适应实体类型嵌入模块，动态地根据车辆和行人的不同特性条件预测过程。第三，为了确保动力学可行性，我们的模型输出的控制信号随后在动力学框架中用于生成物理上可行的轨迹。我们在Dragon Lake Parking (DLP) 数据集和Intersections Drone (inD) 数据集上评估了ParkDiffusion。本文为停车场景中的异构轨迹预测建立了新的基准，明显优于现有方法。 

---
# DeCo: Task Decomposition and Skill Composition for Zero-Shot Generalization in Long-Horizon 3D Manipulation 

**Title (ZH)**: DeCo：长时 horizon 3D 操作中的零样本泛化的任务分解与技能组合 

**Authors**: Zixuan Chen, Junhui Yin, Yangtao Chen, Jing Huo, Pinzhuo Tian, Jieqi Shi, Yiwen Hou, Yinchuan Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.00527)  

**Abstract**: Generalizing language-conditioned multi-task imitation learning (IL) models to novel long-horizon 3D manipulation tasks remains a significant challenge. To address this, we propose DeCo (Task Decomposition and Skill Composition), a model-agnostic framework compatible with various multi-task IL models, designed to enhance their zero-shot generalization to novel, compositional, long-horizon 3D manipulation tasks. DeCo first decomposes IL demonstrations into a set of modular atomic tasks based on the physical interaction between the gripper and objects, and constructs an atomic training dataset that enables models to learn a diverse set of reusable atomic skills during imitation learning. At inference time, DeCo leverages a vision-language model (VLM) to parse high-level instructions for novel long-horizon tasks, retrieve the relevant atomic skills, and dynamically schedule their execution; a spatially-aware skill-chaining module then ensures smooth, collision-free transitions between sequential skills. We evaluate DeCo in simulation using DeCoBench, a benchmark specifically designed to assess zero-shot generalization of multi-task IL models in compositional long-horizon 3D manipulation. Across three representative multi-task IL models (RVT-2, 3DDA, and ARP), DeCo achieves success rate improvements of 66.67%, 21.53%, and 57.92%, respectively, on 12 novel compositional tasks. Moreover, in real-world experiments, a DeCo-enhanced model trained on only 6 atomic tasks successfully completes 9 novel long-horizon tasks, yielding an average success rate improvement of 53.33% over the base multi-task IL model. Video demonstrations are available at: this https URL. 

**Abstract (ZH)**: 面向新型长时 horizon 3D 操作任务的语言条件多任务模仿学习模型的通用化：一种任务分解与技能组合的方法 

---
# Implicit Neural-Representation Learning for Elastic Deformable-Object Manipulations 

**Title (ZH)**: 弹性变形物体操控的隐式神经表示学习 

**Authors**: Minseok Song, JeongHo Ha, Bonggyeong Park, Daehyung Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.00500)  

**Abstract**: We aim to solve the problem of manipulating deformable objects, particularly elastic bands, in real-world scenarios. However, deformable object manipulation (DOM) requires a policy that works on a large state space due to the unlimited degree of freedom (DoF) of deformable objects. Further, their dense but partial observations (e.g., images or point clouds) may increase the sampling complexity and uncertainty in policy learning. To figure it out, we propose a novel implicit neural-representation (INR) learning for elastic DOMs, called INR-DOM. Our method learns consistent state representations associated with partially observable elastic objects reconstructing a complete and implicit surface represented as a signed distance function. Furthermore, we perform exploratory representation fine-tuning through reinforcement learning (RL) that enables RL algorithms to effectively learn exploitable representations while efficiently obtaining a DOM policy. We perform quantitative and qualitative analyses building three simulated environments and real-world manipulation studies with a Franka Emika Panda arm. Videos are available at this http URL. 

**Abstract (ZH)**: 我们旨在解决在真实世界场景中操控变形物体，特别是弹性带子的问题。然而，变形物体操控（DOM）需要一个能够在由于变形物体无限自由度（DoF）而产生的大面积状态空间上工作的策略。此外，这些密集但部分的观察（例如，图像或点云）可能会增加采样复杂性和政策学习中的不确定性。为了解决这一问题，我们提出了一种新颖的隐式神经表示（INR）学习方法，称为INR-DOM。我们的方法学习与部分可观测的弹性物体相关的一致状态表示，重构一个由符号距离函数表示的完整且隐式的表面。此外，我们通过强化学习（RL）进行探究性表示微调，使RL算法能够有效地学习可利用的表示，同时高效地获得一个DOM策略。我们在三个模拟环境和使用Franka Emika Panda手臂的真实世界操控研究中进行了定量和定性的分析。视频可在以下链接获取。 

---
# MULE: Multi-terrain and Unknown Load Adaptation for Effective Quadrupedal Locomotion 

**Title (ZH)**: 多地形和未知载荷适应性四足运动控制 

**Authors**: Vamshi Kumar Kurva, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.00488)  

**Abstract**: Quadrupedal robots are increasingly deployed for load-carrying tasks across diverse terrains. While Model Predictive Control (MPC)-based methods can account for payload variations, they often depend on predefined gait schedules or trajectory generators, limiting their adaptability in unstructured environments. To address these limitations, we propose an Adaptive Reinforcement Learning (RL) framework that enables quadrupedal robots to dynamically adapt to both varying payloads and diverse terrains. The framework consists of a nominal policy responsible for baseline locomotion and an adaptive policy that learns corrective actions to preserve stability and improve command tracking under payload variations. We validate the proposed approach through large-scale simulation experiments in Isaac Gym and real-world hardware deployment on a Unitree Go1 quadruped. The controller was tested on flat ground, slopes, and stairs under both static and dynamic payload changes. Across all settings, our adaptive controller consistently outperformed the controller in tracking body height and velocity commands, demonstrating enhanced robustness and adaptability without requiring explicit gait design or manual tuning. 

**Abstract (ZH)**: 四足机器人越来越多地被部署于多种地形的负载搬运任务中。尽管基于模型预测控制(MPC)的方法可以考虑到负载变化，但这些方法往往依赖预先定义的步伐计划或轨迹生成器，限制了其在非结构化环境中的适应性。为了解决这些局限性，我们提出了一种自适应强化学习(Adaptive Reinforcement Learning, ARL)框架，使四足机器人能够动态适应变化的负载和多样的地形。该框架包括一个基线策略，负责基本的移动，以及一个自适应策略，学习纠正动作以在负载变化下保持稳定性和提高指令跟踪性能。我们通过在Isaac Gym中的大规模仿真实验和在Unitree Go1四足机器人上的实际硬件部署验证了所提出的框架。控制器在平坦地面、坡道和台阶上，在静态和动态负载变化下进行了测试。在所有设置中，我们的自适应控制器在跟踪身体高度和速度命令方面始终优于基线控制器，展示了增强的鲁棒性和适应性，无需显式的步态设计或手动调优。 

---
# Future-Oriented Navigation: Dynamic Obstacle Avoidance with One-Shot Energy-Based Multimodal Motion Prediction 

**Title (ZH)**: 面向未来的导航：基于单次能量化多模态运动预测的一次性动态障碍物规避 

**Authors**: Ze Zhang, Georg Hess, Junjie Hu, Emmanuel Dean, Lennart Svensson, Knut Åkesson  

**Link**: [PDF](https://arxiv.org/pdf/2505.00237)  

**Abstract**: This paper proposes an integrated approach for the safe and efficient control of mobile robots in dynamic and uncertain environments. The approach consists of two key steps: one-shot multimodal motion prediction to anticipate motions of dynamic obstacles and model predictive control to incorporate these predictions into the motion planning process. Motion prediction is driven by an energy-based neural network that generates high-resolution, multi-step predictions in a single operation. The prediction outcomes are further utilized to create geometric shapes formulated as mathematical constraints. Instead of treating each dynamic obstacle individually, predicted obstacles are grouped by proximity in an unsupervised way to improve performance and efficiency. The overall collision-free navigation is handled by model predictive control with a specific design for proactive dynamic obstacle avoidance. The proposed approach allows mobile robots to navigate effectively in dynamic environments. Its performance is accessed across various scenarios that represent typical warehouse settings. The results demonstrate that the proposed approach outperforms other existing dynamic obstacle avoidance methods. 

**Abstract (ZH)**: 本文提出了一种集成方法，用于在动态和不确定环境中安全高效地控制移动机器人。该方法包括两个关键步骤：一次多模态运动预测以预见动态障碍物的运动，以及模型预测控制以将这些预测纳入运动规划过程中。运动预测由基于能量的神经网络驱动，能够在单次操作中生成高分辨率的多步预测。预测结果进一步用于创建作为数学约束的几何形状。预测的障碍物不是单独处理，而是通过无监督的方式根据邻近度进行分组，以提高性能和效率。整体无碰撞导航由专门设计的模型预测控制处理，以实现主动动态障碍物避让。所提出的方法使移动机器人能够在动态环境中有效地导航。其性能在代表典型仓库设置的各种场景中进行了评估。结果表明，所提出的方法优于其他现有的动态障碍物避让方法。 

---
# AI-Enhanced Automatic Design of Efficient Underwater Gliders 

**Title (ZH)**: AI增强的高效水下航行器自动设计 

**Authors**: Peter Yichen Chen, Pingchuan Ma, Niklas Hagemann, John Romanishin, Wei Wang, Daniela Rus, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2505.00222)  

**Abstract**: The development of novel autonomous underwater gliders has been hindered by limited shape diversity, primarily due to the reliance on traditional design tools that depend heavily on manual trial and error. Building an automated design framework is challenging due to the complexities of representing glider shapes and the high computational costs associated with modeling complex solid-fluid interactions. In this work, we introduce an AI-enhanced automated computational framework designed to overcome these limitations by enabling the creation of underwater robots with non-trivial hull shapes. Our approach involves an algorithm that co-optimizes both shape and control signals, utilizing a reduced-order geometry representation and a differentiable neural-network-based fluid surrogate model. This end-to-end design workflow facilitates rapid iteration and evaluation of hydrodynamic performance, leading to the discovery of optimal and complex hull shapes across various control settings. We validate our method through wind tunnel experiments and swimming pool gliding tests, demonstrating that our computationally designed gliders surpass manually designed counterparts in terms of energy efficiency. By addressing challenges in efficient shape representation and neural fluid surrogate models, our work paves the way for the development of highly efficient underwater gliders, with implications for long-range ocean exploration and environmental monitoring. 

**Abstract (ZH)**: 基于AI增强的自动化计算框架在非平凡水下航行器外形设计中的应用：克服传统设计工具限制实现高效水下滑翔器开发 

---
# Investigating Adaptive Tuning of Assistive Exoskeletons Using Offline Reinforcement Learning: Challenges and Insights 

**Title (ZH)**: 基于离线强化学习的辅助外骨骼自适应调谐研究：挑战与见解 

**Authors**: Yasin Findik, Christopher Coco, Reza Azadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.00201)  

**Abstract**: Assistive exoskeletons have shown great potential in enhancing mobility for individuals with motor impairments, yet their effectiveness relies on precise parameter tuning for personalized assistance. In this study, we investigate the potential of offline reinforcement learning for optimizing effort thresholds in upper-limb assistive exoskeletons, aiming to reduce reliance on manual calibration. Specifically, we frame the problem as a multi-agent system where separate agents optimize biceps and triceps effort thresholds, enabling a more adaptive and data-driven approach to exoskeleton control. Mixed Q-Functionals (MQF) is employed to efficiently handle continuous action spaces while leveraging pre-collected data, thereby mitigating the risks associated with real-time exploration. Experiments were conducted using the MyoPro 2 exoskeleton across two distinct tasks involving horizontal and vertical arm movements. Our results indicate that the proposed approach can dynamically adjust threshold values based on learned patterns, potentially improving user interaction and control, though performance evaluation remains challenging due to dataset limitations. 

**Abstract (ZH)**: 基于离线强化学习优化上肢辅助外骨骼的努力阈值：减少手动校准依赖性 

---
# Deep Reinforcement Learning Policies for Underactuated Satellite Attitude Control 

**Title (ZH)**: 未 acted satelliteattitude control的深度强化学习策略 

**Authors**: Matteo El Hariry, Andrea Cini, Giacomo Mellone, Alessandro Balossino  

**Link**: [PDF](https://arxiv.org/pdf/2505.00165)  

**Abstract**: Autonomy is a key challenge for future space exploration endeavours. Deep Reinforcement Learning holds the promises for developing agents able to learn complex behaviours simply by interacting with their environment. This paper investigates the use of Reinforcement Learning for the satellite attitude control problem, namely the angular reorientation of a spacecraft with respect to an in- ertial frame of reference. In the proposed approach, a set of control policies are implemented as neural networks trained with a custom version of the Proximal Policy Optimization algorithm to maneuver a small satellite from a random starting angle to a given pointing target. In particular, we address the problem for two working conditions: the nominal case, in which all the actuators (a set of 3 reac- tion wheels) are working properly, and the underactuated case, where an actuator failure is simulated randomly along with one of the axes. We show that the agents learn to effectively perform large-angle slew maneuvers with fast convergence and industry-standard pointing accuracy. Furthermore, we test the proposed method on representative hardware, showing that by taking adequate measures controllers trained in simulation can perform well in real systems. 

**Abstract (ZH)**: 自主性是未来空间探索任务的关键挑战。深度强化学习为开发仅通过与其环境交互即可学习复杂行为的智能体带来了希望。本文探讨了使用强化学习解决卫星姿态控制问题，即航天器相对于惯性参考系的角再定向。在所提出的方案中，控制策略被实现为通过针对改良的渐进策略优化算法训练的神经网络来操作小型卫星，使其从随机初始角度转向给定的指向目标。特别地，我们针对以下两种工作条件解决了该问题：正常情况，所有执行器（一组3个反应轮）正常工作；欠驱动情况，其中随机模拟其中一个轴上的执行器故障。我们展示了智能体能够有效执行大角度指向机动，具有快速收敛性和行业标准的指向精度。此外，我们在代表性的硬件上测试了所提出的方法，表明通过采取适当措施，模拟中训练的控制器可以在实际系统中表现良好。 

---
# CoordField: Coordination Field for Agentic UAV Task Allocation In Low-altitude Urban Scenarios 

**Title (ZH)**: CoordField：低空城市场景下自主无人机任务分配的协调场 

**Authors**: Tengchao Zhang, Yonglin Tian, Fei Lin, Jun Huang, Rui Qin, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00091)  

**Abstract**: With the increasing demand for heterogeneous Unmanned Aerial Vehicle (UAV) swarms to perform complex tasks in urban environments, system design now faces major challenges, including efficient semantic understanding, flexible task planning, and the ability to dynamically adjust coordination strategies in response to evolving environmental conditions and continuously changing task requirements. To address the limitations of existing approaches, this paper proposes coordination field agentic system for coordinating heterogeneous UAV swarms in complex urban scenarios. In this system, large language models (LLMs) is responsible for interpreting high-level human instructions and converting them into executable commands for the UAV swarms, such as patrol and target tracking. Subsequently, a Coordination field mechanism is proposed to guide UAV motion and task selection, enabling decentralized and adaptive allocation of emergent tasks. A total of 50 rounds of comparative testing were conducted across different models in a 2D simulation space to evaluate their performance. Experimental results demonstrate that the proposed system achieves superior performance in terms of task coverage, response time, and adaptability to dynamic changes. 

**Abstract (ZH)**: 基于大型语言模型的协调场代理系统：用于复杂城市环境下的异构无人机群协调 

---
# Towards Autonomous Micromobility through Scalable Urban Simulation 

**Title (ZH)**: 面向自主微出行的可扩展城市仿真研究 

**Authors**: Wayne Wu, Honglin He, Chaoyuan Zhang, Jack He, Seth Z. Zhao, Ran Gong, Quanyi Li, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.00690)  

**Abstract**: Micromobility, which utilizes lightweight mobile machines moving in urban public spaces, such as delivery robots and mobility scooters, emerges as a promising alternative to vehicular mobility. Current micromobility depends mostly on human manual operation (in-person or remote control), which raises safety and efficiency concerns when navigating busy urban environments full of unpredictable obstacles and pedestrians. Assisting humans with AI agents in maneuvering micromobility devices presents a viable solution for enhancing safety and efficiency. In this work, we present a scalable urban simulation solution to advance autonomous micromobility. First, we build URBAN-SIM - a high-performance robot learning platform for large-scale training of embodied agents in interactive urban scenes. URBAN-SIM contains three critical modules: Hierarchical Urban Generation pipeline, Interactive Dynamics Generation strategy, and Asynchronous Scene Sampling scheme, to improve the diversity, realism, and efficiency of robot learning in simulation. Then, we propose URBAN-BENCH - a suite of essential tasks and benchmarks to gauge various capabilities of the AI agents in achieving autonomous micromobility. URBAN-BENCH includes eight tasks based on three core skills of the agents: Urban Locomotion, Urban Navigation, and Urban Traverse. We evaluate four robots with heterogeneous embodiments, such as the wheeled and legged robots, across these tasks. Experiments on diverse terrains and urban structures reveal each robot's strengths and limitations. 

**Abstract (ZH)**: 微移动性：利用轻型移动机器人在城市公共空间中运行的新兴技术，如配送机器人和电动滑板车，已成为车辆移动的有前景的替代方案。当前的微移动性主要依赖于人工手动操作（现场或远程控制），在繁忙且充满不可预测障碍和行人的城市环境中导航时，存在安全性和效率方面的担忧。通过AI代理辅助人类操作微移动性设备，可以有效提高安全性和效率。在本研究中，我们提出了一种可扩展的城市仿真解决方案，以推动自主微移动性的发展。首先，我们构建了URBAN-SIM——一个高性能的机器人学习平台，用于大规模训练交互式城市场景中的具身代理。URBAN-SIM包含三个关键模块：分层城市生成管道、交互动力学生成策略和异步场景抽样方案，以提高机器人在仿真中学习的多样性和真实性，提高效率。然后，我们提出了URBAN-BENCH——一套基础任务和基准测试，以评估AI代理在实现自主微移动性方面的各种能力。URBAN-BENCH包括基于代理三项核心技能的八项任务：城市运动、城市导航和城市穿越。我们在这些任务中评估了四种具有不同体态的机器人，如轮式和腿式机器人。在多样化的地形和城市结构上的实验揭示了每种机器人在能力和局限性方面的不同表现。 

---
# Emergence of Roles in Robotic Teams with Model Sharing and Limited Communication 

**Title (ZH)**: 具有模型共享和有限通信的机器人团队中的角色 emergence 在机器人团队中的角色分化：基于模型共享与有限通信 

**Authors**: Ian O'Flynn, Harun Šiljak  

**Link**: [PDF](https://arxiv.org/pdf/2505.00540)  

**Abstract**: We present a reinforcement learning strategy for use in multi-agent foraging systems in which the learning is centralised to a single agent and its model is periodically disseminated among the population of non-learning agents. In a domain where multi-agent reinforcement learning (MARL) is the common approach, this approach aims to significantly reduce the computational and energy demands compared to approaches such as MARL and centralised learning models. By developing high performing foraging agents, these approaches can be translated into real-world applications such as logistics, environmental monitoring, and autonomous exploration. A reward function was incorporated into this approach that promotes role development among agents, without explicit directives. This led to the differentiation of behaviours among the agents. The implicit encouragement of role differentiation allows for dynamic actions in which agents can alter roles dependent on their interactions with the environment without the need for explicit communication between agents. 

**Abstract (ZH)**: 一种集中学习的多Agent采集系统 reinforcement学习策略：减少计算和能源需求并促进角色分化 

---
# Variational OOD State Correction for Offline Reinforcement Learning 

**Title (ZH)**: 离线强化学习中的变分OOD状态校正 

**Authors**: Ke Jiang, Wen Jiang, Xiaoyang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00503)  

**Abstract**: The performance of Offline reinforcement learning is significantly impacted by the issue of state distributional shift, and out-of-distribution (OOD) state correction is a popular approach to address this problem. In this paper, we propose a novel method named Density-Aware Safety Perception (DASP) for OOD state correction. Specifically, our method encourages the agent to prioritize actions that lead to outcomes with higher data density, thereby promoting its operation within or the return to in-distribution (safe) regions. To achieve this, we optimize the objective within a variational framework that concurrently considers both the potential outcomes of decision-making and their density, thus providing crucial contextual information for safe decision-making. Finally, we validate the effectiveness and feasibility of our proposed method through extensive experimental evaluations on the offline MuJoCo and AntMaze suites. 

**Abstract (ZH)**: 基于密度感知的安全感知（DASP）方法在离线强化学习中处理离域状态纠正的问题 

---
# UserCentrix: An Agentic Memory-augmented AI Framework for Smart Spaces 

**Title (ZH)**: UserCentrix: 一个赋能的记忆增强AI框架用于智能空间 

**Authors**: Alaa Saleh, Sasu Tarkoma, Praveen Kumar Donta, Naser Hossein Motlagh, Schahram Dustdar, Susanna Pirttikangas, Lauri Lovén  

**Link**: [PDF](https://arxiv.org/pdf/2505.00472)  

**Abstract**: Agentic AI, with its autonomous and proactive decision-making, has transformed smart environments. By integrating Generative AI (GenAI) and multi-agent systems, modern AI frameworks can dynamically adapt to user preferences, optimize data management, and improve resource allocation. This paper introduces UserCentrix, an agentic memory-augmented AI framework designed to enhance smart spaces through dynamic, context-aware decision-making. This framework integrates personalized Large Language Model (LLM) agents that leverage user preferences and LLM memory management to deliver proactive and adaptive assistance. Furthermore, it incorporates a hybrid hierarchical control system, balancing centralized and distributed processing to optimize real-time responsiveness while maintaining global situational awareness. UserCentrix achieves resource-efficient AI interactions by embedding memory-augmented reasoning, cooperative agent negotiation, and adaptive orchestration strategies. Our key contributions include (i) a self-organizing framework with proactive scaling based on task urgency, (ii) a Value of Information (VoI)-driven decision-making process, (iii) a meta-reasoning personal LLM agent, and (iv) an intelligent multi-agent coordination system for seamless environment adaptation. Experimental results across various models confirm the effectiveness of our approach in enhancing response accuracy, system efficiency, and computational resource management in real-world application. 

**Abstract (ZH)**: 具有自主和主动决策能力的代理AI已经转型了智能环境。通过整合生成AI（GenAI）和多智能体系统，现代AI框架可以动态适应用户偏好，优化数据管理，并改进资源配置。本文介绍了一种名为UserCentrix的代理记忆增强AI框架，该框架旨在通过动态、情境感知的决策来增强智能空间。该框架整合了利用用户偏好和大型语言模型（LLM）记忆管理的个性化LLM代理，以提供主动且适应性的协助。此外，它还纳入了混合层次控制系统，平衡集中式和分布式处理，以优化实时响应能力，同时保持全局情况意识。UserCentrix通过嵌入记忆增强推理、协作智能体协商和适应性编排策略实现了高效的AI交互。我们的主要贡献包括：(i) 一个基于任务紧迫性自动扩展的自我组织框架，(ii) 一种基于信息价值（VoI）的决策过程，(iii) 一个元推理个性化LLM代理，以及(iv) 一种智能多智能体协调系统，以实现无缝的环境适应。在各种模型上的实验结果证实了该方法在增强响应准确性、系统效率和计算资源管理方面的有效性。 

---
# Position Paper: Towards Open Complex Human-AI Agents Collaboration System for Problem-Solving and Knowledge Management 

**Title (ZH)**: 位置论文：通往开放复杂人机协作系统的道路——面向问题解决和知识管理 

**Authors**: Ju Wu, Calvin K.L. Or  

**Link**: [PDF](https://arxiv.org/pdf/2505.00018)  

**Abstract**: This position paper critically surveys a broad spectrum of recent empirical developments on human-AI agents collaboration, highlighting both their technical achievements and persistent gaps. We observe a lack of a unifying theoretical framework that can coherently integrate these varied studies, especially when tackling open-ended, complex tasks. To address this, we propose a novel conceptual architecture: one that systematically interlinks the technical details of multi-agent coordination, knowledge management, cybernetic feedback loops, and higher-level control mechanisms. By mapping existing contributions, from symbolic AI techniques and connectionist LLM-based agents to hybrid organizational practices, onto this proposed framework (Hierarchical Exploration-Exploitation Net), our approach facilitates revision of legacy methods and inspires new work that fuses qualitative and quantitative paradigms. The paper's structure allows it to be read from any section, serving equally as a critical review of technical implementations and as a forward-looking reference for designing or extending human-AI symbioses. Together, these insights offer a stepping stone toward deeper co-evolution of human cognition and AI capability. 

**Abstract (ZH)**: 这篇立场论文批判性地回顾了近期人类-人工智能代理合作的广泛实证发展，突出了其技术成就和持续存在的空白。我们注意到缺乏一个统一的理论框架来综合这些多样的研究，特别是在处理开放性和复杂性任务时。为此，我们提出了一种新颖的概念架构：一种系统地将多代理协调技术细节、知识管理、控制论反馈回路和高级控制机制相互关联的架构。通过将现有的贡献，从符号人工智能技术到连接主义LLM基代理，再到混合组织实践，映射到这一提议的框架（层次探索-利用网）中，我们的方法促进了对遗留方法的修订，并启发了结合定性和定量范式的新型研究。该论文的结构使其可以从任何部分阅读，既作为对技术实现的批判性回顾，又作为设计或扩展人类-人工智能共生体的前瞻性参考。这些洞察为我们更深一步的人机认知共进化提供了基石。 

---
# Deep Reinforcement Learning for Urban Air Quality Management: Multi-Objective Optimization of Pollution Mitigation Booth Placement in Metropolitan Environments 

**Title (ZH)**: 城市空气质量管理中的深度 reinforcement 学习：大规模城市环境中污染削减装置布局的多目标优化 

**Authors**: Kirtan Rajesh, Suvidha Rupesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00668)  

**Abstract**: Urban air pollution remains a pressing global concern, particularly in densely populated and traffic-intensive metropolitan areas like Delhi, where exposure to harmful pollutants severely impacts public health. Delhi, being one of the most polluted cities globally, experiences chronic air quality issues due to vehicular emissions, industrial activities, and construction dust, which exacerbate its already fragile atmospheric conditions. Traditional pollution mitigation strategies, such as static air purifying installations, often fail to maximize their impact due to suboptimal placement and limited adaptability to dynamic urban environments. This study presents a novel deep reinforcement learning (DRL) framework to optimize the placement of air purification booths to improve the air quality index (AQI) in the city of Delhi. We employ Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm, to iteratively learn and identify high-impact locations based on multiple spatial and environmental factors, including population density, traffic patterns, industrial influence, and green space constraints. Our approach is benchmarked against conventional placement strategies, including random and greedy AQI-based methods, using multi-dimensional performance evaluation metrics such as AQI improvement, spatial coverage, population and traffic impact, and spatial entropy. Experimental results demonstrate that the RL-based approach outperforms baseline methods by achieving a balanced and effective distribution of air purification infrastructure. Notably, the DRL framework achieves an optimal trade-off between AQI reduction and high-coverage deployment, ensuring equitable environmental benefits across urban regions. The findings underscore the potential of AI-driven spatial optimization in advancing smart city initiatives and data-driven urban air quality management. 

**Abstract (ZH)**: 基于深度强化学习的城市空气污染优化治理框架：以Delhi为例 

---
# Data Therapist: Eliciting Domain Knowledge from Subject Matter Experts Using Large Language Models 

**Title (ZH)**: 数据治疗师：使用大型语言模型从领域专家处提取专业知识 

**Authors**: Sungbok Shin, Hyeon Jeon, Sanghyun Hong, Niklas Elmqvist  

**Link**: [PDF](https://arxiv.org/pdf/2505.00455)  

**Abstract**: Effective data visualization requires not only technical proficiency but also a deep understanding of the domain-specific context in which data exists. This context often includes tacit knowledge about data provenance, quality, and intended use, which is rarely explicit in the dataset itself. We present the Data Therapist, a web-based tool that helps domain experts externalize this implicit knowledge through a mixed-initiative process combining iterative Q&A with interactive annotation. Powered by a large language model, the system analyzes user-supplied datasets, prompts users with targeted questions, and allows annotation at varying levels of granularity. The resulting structured knowledge base can inform both human and automated visualization design. We evaluated the tool in a qualitative study involving expert pairs from Molecular Biology, Accounting, Political Science, and Usable Security. The study revealed recurring patterns in how experts reason about their data and highlights areas where AI support can improve visualization design. 

**Abstract (ZH)**: 有效的数据可视化不仅需要技术 proficiency，还需要对数据存在的领域特定上下文有深刻的理解。这种上下文通常包括关于数据来源、质量及预期用途的隐性知识，而这些知识在数据集本身中往往并未明确体现。我们提出了Data Therapist这一基于Web的工具，通过结合迭代问答和互动注释的混合主动过程，帮助领域专家外化这一隐性知识。该系统依托大型语言模型，分析用户提供的数据集，向用户提出针对性的问题，并允许在不同粒度级别进行注释。生成的结构化知识库可以指导人类和自动化的可视化设计。我们在涉及分子生物学、会计学、政治科学和可用安全性领域的专家配对中开展了定性研究，研究揭示了专家在处理数据时思维方式中的反复出现模式，并强调了AI支持如何改进可视化设计的领域。 

---
# Enhancing Speech-to-Speech Dialogue Modeling with End-to-End Retrieval-Augmented Generation 

**Title (ZH)**: 增强端到端检索增强生成的语音到语音对话建模 

**Authors**: Pengchao Feng, Ziyang Ma, Wenxi Chen, Yao Li, Sheng Wang, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00028)  

**Abstract**: In recent years, end-to-end speech-to-speech (S2S) dialogue systems have garnered increasing research attention due to their advantages over traditional cascaded systems, including achieving lower latency and more natural integration of nonverbal cues such as emotion and speaker identity. However, these end-to-end systems face key challenges, particularly in incorporating external knowledge, a capability commonly addressed by Retrieval-Augmented Generation (RAG) in text-based large language models (LLMs). The core difficulty lies in the modality gap between input speech and retrieved textual knowledge, which hinders effective integration. To address this issue, we propose a novel end-to-end RAG framework that directly retrieves relevant textual knowledge from speech queries, eliminating the need for intermediate speech-to-text conversion via techniques like ASR. Experimental results demonstrate that our method significantly improves the performance of end-to-end S2S dialogue systems while achieving higher retrieval efficiency. Although the overall performance still lags behind cascaded models, our framework offers a promising direction for enhancing knowledge integration in end-to-end S2S systems. We will release the code and dataset to support reproducibility and promote further research in this area. 

**Abstract (ZH)**: 近年来，端到端语音到语音（S2S）对话系统由于其优于传统级联系统的优点，包括较低的延迟和更自然地整合诸如情绪和说话人身份等非言语线索，引起了越来越多的研究关注。然而，这些端到端系统面临着关键挑战，特别是在融入外部知识方面，这一能力通常由基于文本的大语言模型（LLMs）中的检索增强生成（RAG）解决。核心难点在于输入语音与检索到的文本知识之间的模态差距，这妨碍了有效的整合。为了解决这个问题，我们提出了一种新的端到端RAG框架，该框架可以直接从语音查询中检索相关文本知识，从而消除通过ASR等技术介导的语音到文本转换的需要。实验结果表明，我们的方法显著提高了端到端S2S对话系统的性能，同时实现了更高的检索效率。虽然整体性能仍落后于级联模型，但我们的框架为增强端到端S2S系统中的知识整合提供了有前景的方向。我们将发布代码和数据集以支持可再现性并促进该领域的进一步研究。 

---
