# Neuro-Symbolic Generation of Explanations for Robot Policies with Weighted Signal Temporal Logic 

**Title (ZH)**: 基于加权信号时序逻辑的神经符号机器人策略解释生成 

**Authors**: Mikihisa Yuasa, Ramavarapu S. Sreenivas, Huy T. Tran  

**Link**: [PDF](https://arxiv.org/pdf/2504.21841)  

**Abstract**: Neural network-based policies have demonstrated success in many robotic applications, but often lack human-explanability, which poses challenges in safety-critical deployments. To address this, we propose a neuro-symbolic explanation framework that generates a weighted signal temporal logic (wSTL) specification to describe a robot policy in a interpretable form. Existing methods typically produce explanations that are verbose and inconsistent, which hinders explainability, and loose, which do not give meaningful insights into the underlying policy. We address these issues by introducing a simplification process consisting of predicate filtering, regularization, and iterative pruning. We also introduce three novel explainability evaluation metrics -- conciseness, consistency, and strictness -- to assess explanation quality beyond conventional classification metrics. Our method is validated in three simulated robotic environments, where it outperforms baselines in generating concise, consistent, and strict wSTL explanations without sacrificing classification accuracy. This work bridges policy learning with formal methods, contributing to safer and more transparent decision-making in robotics. 

**Abstract (ZH)**: 基于神经网络的政策在许多机器人应用中取得了成功，但常常缺乏人类可解释性，这在安全关键部署中提出了挑战。为应对这一问题，我们提出了一种神经符号解释框架，生成加权信号时序逻辑（wSTL）规范，以一种可解释的形式描述机器人政策。现有方法通常生成冗长且不一致的解释，妨碍了可解释性，并且宽松的解释无法提供有关潜在政策的有意义见解。我们通过引入包括谓词过滤、正则化和迭代剪枝的简化过程来解决这些问题。我们还引入了三种新的可解释性评估指标——简洁性、一致性和严格性——以超越传统分类指标来评估解释质量。该方法在三个模拟机器人环境中得到了验证，能够在不牺牲分类准确性的情况下生成简洁、一致和严格的wSTL解释，从而将策略学习与形式方法相结合，为机器人中的更安全和透明决策做出了贡献。 

---
# An Underwater, Fault-Tolerant, Laser-Aided Robotic Multi-Modal Dense SLAM System for Continuous Underwater In-Situ Observation 

**Title (ZH)**: 一种基于水下激光辅助的容错多模态密集SLAM系统，实现连续原位水下观测 

**Authors**: Yaming Ou, Junfeng Fan, Chao Zhou, Pengju Zhang, Zongyuan Shen, Yichen Fu, Xiaoyan Liu, Zengguang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21826)  

**Abstract**: Existing underwater SLAM systems are difficult to work effectively in texture-sparse and geometrically degraded underwater environments, resulting in intermittent tracking and sparse mapping. Therefore, we present Water-DSLAM, a novel laser-aided multi-sensor fusion system that can achieve uninterrupted, fault-tolerant dense SLAM capable of continuous in-situ observation in diverse complex underwater scenarios through three key innovations: Firstly, we develop Water-Scanner, a multi-sensor fusion robotic platform featuring a self-designed Underwater Binocular Structured Light (UBSL) module that enables high-precision 3D perception. Secondly, we propose a fault-tolerant triple-subsystem architecture combining: 1) DP-INS (DVL- and Pressure-aided Inertial Navigation System): fusing inertial measurement unit, doppler velocity log, and pressure sensor based Error-State Kalman Filter (ESKF) to provide high-frequency absolute odometry 2) Water-UBSL: a novel Iterated ESKF (IESKF)-based tight coupling between UBSL and DP-INS to mitigate UBSL's degeneration issues 3) Water-Stereo: a fusion of DP-INS and stereo camera for accurate initialization and tracking. Thirdly, we introduce a multi-modal factor graph back-end that dynamically fuses heterogeneous sensor data. The proposed multi-sensor factor graph maintenance strategy efficiently addresses issues caused by asynchronous sensor frequencies and partial data loss. Experimental results demonstrate Water-DSLAM achieves superior robustness (0.039 m trajectory RMSE and 100\% continuity ratio during partial sensor dropout) and dense mapping (6922.4 points/m^3 in 750 m^3 water volume, approximately 10 times denser than existing methods) in various challenging environments, including pools, dark underwater scenes, 16-meter-deep sinkholes, and field rivers. Our project is available at this https URL. 

**Abstract (ZH)**: 水下SLAM系统Water-DSLAM：一种适用于复杂水下场景的多传感器融合系统 

---
# Whleaper: A 10-DOF Flexible Bipedal Wheeled Robot 

**Title (ZH)**: Whleaper: 一种10自由度柔性 bipedal 轮式机器人 

**Authors**: Yinglei Zhu, Sixiao He, Zhenghao Qi, Zhuoyuan Yong, Yihua Qin, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21767)  

**Abstract**: Wheel-legged robots combine the advantages of both wheeled robots and legged robots, offering versatile locomotion capabilities with excellent stability on challenging terrains and high efficiency on flat surfaces. However, existing wheel-legged robots typically have limited hip joint mobility compared to humans, while hip joint plays a crucial role in locomotion. In this paper, we introduce Whleaper, a novel 10-degree-of-freedom (DOF) bipedal wheeled robot, with 3 DOFs at the hip of each leg. Its humanoid joint design enables adaptable motion in complex scenarios, ensuring stability and flexibility. This paper introduces the details of Whleaper, with a focus on innovative mechanical design, control algorithms and system implementation. Firstly, stability stems from the increased DOFs at the hip, which expand the range of possible postures and improve the robot's foot-ground contact. Secondly, the extra DOFs also augment its mobility. During walking or sliding, more complex movements can be adopted to execute obstacle avoidance tasks. Thirdly, we utilize two control algorithms to implement multimodal motion for walking and sliding. By controlling specific DOFs of the robot, we conducted a series of simulations and practical experiments, demonstrating that a high-DOF hip joint design can effectively enhance the stability and flexibility of wheel-legged robots. Whleaper shows its capability to perform actions such as squatting, obstacle avoidance sliding, and rapid turning in real-world scenarios. 

**Abstract (ZH)**: 轮腿机器人结合了轮式机器人和腿式机器人各自的优点，提供在挑战性地形中出色的稳定性和在平坦表面上的高效率的多功能移动能力。然而，现有的轮腿机器人相比人类通常具有有限的髋关节 Mobility，而髋关节在移动中起着至关重要的作用。本文介绍了Whleaper，一种新型10自由度（DOF）双足轮式机器人，每条腿具有3个髋关节自由度。其类人关节设计使其能够在复杂场景中适应运动，确保稳定性和灵活性。本文详细介绍Whleaper，重点介绍了其创新机械设计、控制算法和系统实现。首先，稳定性来自于髋关节处增加的自由度，这扩展了可能的姿态范围并提高了机器人足底着地接触的质量。其次，额外的自由度也增强了其机动性。在行走或滑行时，可以采用更复杂的动作执行避障任务。第三，我们利用两种控制算法实现了行走和滑行的多模态运动。通过控制机器人特定的自由度，我们进行了系列仿真和实验证明，高自由度髋关节设计可以有效地增强轮腿机器人的稳定性和灵活性。Whleaper展示了其在真实场景中执行蹲伏、避障滑行和快速转弯等动作的能力。 

---
# LangWBC: Language-directed Humanoid Whole-Body Control via End-to-end Learning 

**Title (ZH)**: LangWBC: 通过端到端学习的语言导向 humanoid 全身控制 

**Authors**: Yiyang Shao, Xiaoyu Huang, Bike Zhang, Qiayuan Liao, Yuman Gao, Yufeng Chi, Zhongyu Li, Sophia Shao, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2504.21738)  

**Abstract**: General-purpose humanoid robots are expected to interact intuitively with humans, enabling seamless integration into daily life. Natural language provides the most accessible medium for this purpose. However, translating language into humanoid whole-body motion remains a significant challenge, primarily due to the gap between linguistic understanding and physical actions. In this work, we present an end-to-end, language-directed policy for real-world humanoid whole-body control. Our approach combines reinforcement learning with policy distillation, allowing a single neural network to interpret language commands and execute corresponding physical actions directly. To enhance motion diversity and compositionality, we incorporate a Conditional Variational Autoencoder (CVAE) structure. The resulting policy achieves agile and versatile whole-body behaviors conditioned on language inputs, with smooth transitions between various motions, enabling adaptation to linguistic variations and the emergence of novel motions. We validate the efficacy and generalizability of our method through extensive simulations and real-world experiments, demonstrating robust whole-body control. Please see our website at this http URL for more information. 

**Abstract (ZH)**: 通用 humanoid 机器人期望能够直观地与人类交互，从而实现无缝融入日常生活。自然语言是最具可访问性的媒介。然而，将语言翻译成类人全身动作仍然是一项显著的挑战，主要是由于语言理解和物理动作之间的差距。在本文中，我们提出了一种端到端、基于语言的政策，用于现实中的类人全身控制。我们的方法结合了强化学习与策略蒸馏，使得单个神经网络能够直接解释语言命令并执行相应的物理动作。为了增强动作的多样性和组合性，我们引入了一个条件变分自编码器（CVAE）结构。最终的政策能够在语言输入的条件下实现灵活且多功能的整体动作行为，具有平滑的动作过渡，能够适应语言变化并产生新的动作。我们通过广泛的仿真实验和现实世界实验验证了该方法的有效性和普适性，展示了稳健的整体动作控制能力。更多信息请参见我们的网站：this http URL。 

---
# LLM-Empowered Embodied Agent for Memory-Augmented Task Planning in Household Robotics 

**Title (ZH)**: 基于LLM的具有记忆增强任务规划能力的物理体代理 

**Authors**: Marc Glocker, Peter Hönig, Matthias Hirschmanner, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2504.21716)  

**Abstract**: We present an embodied robotic system with an LLM-driven agent-orchestration architecture for autonomous household object management. The system integrates memory-augmented task planning, enabling robots to execute high-level user commands while tracking past actions. It employs three specialized agents: a routing agent, a task planning agent, and a knowledge base agent, each powered by task-specific LLMs. By leveraging in-context learning, our system avoids the need for explicit model training. RAG enables the system to retrieve context from past interactions, enhancing long-term object tracking. A combination of Grounded SAM and LLaMa3.2-Vision provides robust object detection, facilitating semantic scene understanding for task planning. Evaluation across three household scenarios demonstrates high task planning accuracy and an improvement in memory recall due to RAG. Specifically, Qwen2.5 yields best performance for specialized agents, while LLaMA3.1 excels in routing tasks. The source code is available at: this https URL. 

**Abstract (ZH)**: 我们提出了一种由大语言模型驱动的代理协调架构，用于自主家庭物体管理的具身机器人系统。该系统集成了增强记忆的任务规划，使机器人能够执行高层用户命令并跟踪过去的行为。系统采用三个专门的代理：路由代理、任务规划代理和知识库代理，每个代理由任务特定的LLM提供动力。通过利用上下文学习，我们的系统避免了显式模型训练的需求。基于检索增强生成（RAG），系统能够从过去的交互中检索上下文，增强长期物体跟踪。Grounded SAM与LLaMa3.2-Vision的结合为任务规划提供了 robust 的物体检测能力，促进了语义场景理解。在三个家庭场景的评估中，展示了高任务规划准确性和由于RAG导致的记忆召回改进。具体来说，Qwen2.5在专门代理方面表现最佳，而LLaMA3.1在路由任务方面表现优异。源代码可在以下网址获取：this https URL。 

---
# Multi-Goal Dexterous Hand Manipulation using Probabilistic Model-based Reinforcement Learning 

**Title (ZH)**: 基于概率模型的强化学习在多目标灵巧手操作中的应用 

**Authors**: Yingzhuo Jiang, Wenjun Huang, Rongdun Lin, Chenyang Miao, Tianfu Sun, Yunduan Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.21585)  

**Abstract**: This paper tackles the challenge of learning multi-goal dexterous hand manipulation tasks using model-based Reinforcement Learning. We propose Goal-Conditioned Probabilistic Model Predictive Control (GC-PMPC) by designing probabilistic neural network ensembles to describe the high-dimensional dexterous hand dynamics and introducing an asynchronous MPC policy to meet the control frequency requirements in real-world dexterous hand systems. Extensive evaluations on four simulated Shadow Hand manipulation scenarios with randomly generated goals demonstrate GC-PMPC's superior performance over state-of-the-art baselines. It successfully drives a cable-driven Dexterous hand, DexHand 021 with 12 Active DOFs and 5 tactile sensors, to learn manipulating a cubic die to three goal poses within approximately 80 minutes of interactions, demonstrating exceptional learning efficiency and control performance on a cost-effective dexterous hand platform. 

**Abstract (ZH)**: 基于模型的强化学习中多目标灵巧手 manipulation 任务的学习：目标条件概率模型预测控制方法的研究 

---
# RoboGround: Robotic Manipulation with Grounded Vision-Language Priors 

**Title (ZH)**: RoboGround: 基于地面视觉-语言先验的机器人操作 

**Authors**: Haifeng Huang, Xinyi Chen, Yilun Chen, Hao Li, Xiaoshen Han, Zehan Wang, Tai Wang, Jiangmiao Pang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.21530)  

**Abstract**: Recent advancements in robotic manipulation have highlighted the potential of intermediate representations for improving policy generalization. In this work, we explore grounding masks as an effective intermediate representation, balancing two key advantages: (1) effective spatial guidance that specifies target objects and placement areas while also conveying information about object shape and size, and (2) broad generalization potential driven by large-scale vision-language models pretrained on diverse grounding datasets. We introduce RoboGround, a grounding-aware robotic manipulation system that leverages grounding masks as an intermediate representation to guide policy networks in object manipulation tasks. To further explore and enhance generalization, we propose an automated pipeline for generating large-scale, simulated data with a diverse set of objects and instructions. Extensive experiments show the value of our dataset and the effectiveness of grounding masks as intermediate guidance, significantly enhancing the generalization abilities of robot policies. 

**Abstract (ZH)**: 近期机器人操作领域的进展突显了中间表示在提高策略泛化能力方面的潜力。在本文中，我们探索基于掩码的中间表示作为有效的方案，结合了两项关键优势：（1）有效的空间指导，能够指定目标对象及其放置区域，并传达对象的形状和大小信息；（2）由大规模预训练的视觉-语言模型驱动的广泛泛化能力，这些模型基于多样化的基础现实数据集。我们提出了RoboGround，这是一种基于掩码的机器人操作系统，利用掩码作为中间表示来指导物体操作任务中的策略网络。为了进一步探索和增强泛化能力，我们提出了一种自动生成多样对象和指令的大规模模拟数据的自动化管道。广泛的实验表明，我们数据集的价值以及掩码作为中间指导的有效性，显著提升了机器人策略的泛化能力。 

---
# SimPRIVE: a Simulation framework for Physical Robot Interaction with Virtual Environments 

**Title (ZH)**: SimPRIVE: 一种物理机器人与虚拟环境交互的仿真框架 

**Authors**: Federico Nesti, Gianluca D'Amico, Mauro Marinoni, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2504.21454)  

**Abstract**: The use of machine learning in cyber-physical systems has attracted the interest of both industry and academia. However, no general solution has yet been found against the unpredictable behavior of neural networks and reinforcement learning agents. Nevertheless, the improvements of photo-realistic simulators have paved the way towards extensive testing of complex algorithms in different virtual scenarios, which would be expensive and dangerous to implement in the real world.
This paper presents SimPRIVE, a simulation framework for physical robot interaction with virtual environments, which operates as a vehicle-in-the-loop platform, rendering a virtual world while operating the vehicle in the real world.
Using SimPRIVE, any physical mobile robot running on ROS 2 can easily be configured to move its digital twin in a virtual world built with the Unreal Engine 5 graphic engine, which can be populated with objects, people, or other vehicles with programmable behavior.
SimPRIVE has been designed to accommodate custom or pre-built virtual worlds while being light-weight to contain execution times and allow fast rendering. Its main advantage lies in the possibility of testing complex algorithms on the full software and hardware stack while minimizing the risks and costs of a test campaign. The framework has been validated by testing a reinforcement learning agent trained for obstacle avoidance on an AgileX Scout Mini rover that navigates a virtual office environment where everyday objects and people are placed as obstacles. The physical rover moves with no collision in an indoor limited space, thanks to a LiDAR-based heuristic. 

**Abstract (ZH)**: 机器学习在 cyber-物理系统中的应用引起了工业和学术界的兴趣。然而，尚未找到针对神经网络和强化学习代理不可预测行为的通用解决方案。尽管如此，逼真的模拟器技术的进步为在不同的虚拟场景中广泛测试复杂算法铺平了道路，而在真实世界中实现这些测试将非常昂贵且危险。

本文提出了SimPRIVE，一种用于物理机器人与虚拟环境交互的模拟框架，它作为车辆在环平台运行，在真实世界中操作车辆的同时渲染虚拟世界。

使用SimPRIVE，任何运行在ROS 2上的物理移动机器人可以轻松配置其数字双胞胎在使用Unreal Engine 5图形引擎构建的虚拟世界中移动，该虚拟世界可以包含具有可编程行为的对象、人物或其他车辆。

SimPRIVE旨在容纳自定义或预构建的虚拟世界，同时保持轻量级以控制执行时间和允许快速渲染。其主要优势在于能够在完整的软件和硬件堆栈上测试复杂算法，同时将测试活动的风险和成本降到最低。该框架通过在使用基于LiDAR的启发式算法的AgileX Scout Mini移动机器人上测试训练用于避免障碍的强化学习代理，并使其在虚拟办公环境中导航，其中放置了日常物体和人物作为障碍物，得到了验证。移动机器人在室内有限空间内成功避开了障碍物，未发生碰撞。 

---
# UAV-VLN: End-to-End Vision Language guided Navigation for UAVs 

**Title (ZH)**: UAV-VLN: 端到端视觉语言引导的无人机导航 

**Authors**: Pranav Saxena, Nishant Raghuvanshi, Neena Goveas  

**Link**: [PDF](https://arxiv.org/pdf/2504.21432)  

**Abstract**: A core challenge in AI-guided autonomy is enabling agents to navigate realistically and effectively in previously unseen environments based on natural language commands. We propose UAV-VLN, a novel end-to-end Vision-Language Navigation (VLN) framework for Unmanned Aerial Vehicles (UAVs) that seamlessly integrates Large Language Models (LLMs) with visual perception to facilitate human-interactive navigation. Our system interprets free-form natural language instructions, grounds them into visual observations, and plans feasible aerial trajectories in diverse environments.
UAV-VLN leverages the common-sense reasoning capabilities of LLMs to parse high-level semantic goals, while a vision model detects and localizes semantically relevant objects in the environment. By fusing these modalities, the UAV can reason about spatial relationships, disambiguate references in human instructions, and plan context-aware behaviors with minimal task-specific supervision. To ensure robust and interpretable decision-making, the framework includes a cross-modal grounding mechanism that aligns linguistic intent with visual context.
We evaluate UAV-VLN across diverse indoor and outdoor navigation scenarios, demonstrating its ability to generalize to novel instructions and environments with minimal task-specific training. Our results show significant improvements in instruction-following accuracy and trajectory efficiency, highlighting the potential of LLM-driven vision-language interfaces for safe, intuitive, and generalizable UAV autonomy. 

**Abstract (ZH)**: AI引导的自主性核心挑战在于使代理能够在以前未见过的环境中根据自然语言指令进行现实有效的导航。我们提出了UAV-V LN，这是一种新颖的端到端视觉-语言导航（VLN）框架，用于无人驾驶飞机（UAVs），该框架无缝地将大型语言模型（LLMs）与视觉感知结合在一起，以促进人机交互导航。该系统解释自由形式的自然语言指令，将其转化为视觉观察，并在多种环境中规划可行的空中轨迹。UAV-VLN利用大型语言模型的常识推理能力解析高层语义目标，同时视觉模型检测并定位环境中的语义相关物。通过融合这些模态，无人驾驶飞机可以推理空间关系，消解人类指令中的指代歧义，并在最少的任务特定监督下规划上下文相关的行为。为了确保稳健和可解释的决策制定，框架包括一个跨模态定位机制，使语言意图与视觉上下文对齐。我们在多种室内外导航场景中评估了UAV-VLN，展示了其在最少任务特定训练的情况下，能够泛化到新指令和新环境的能力。结果表明，指令遵循的准确性显著提高，轨迹效率也得到了提升，突显了基于大型语言模型的视觉-语言接口在安全、直观和泛化的无人驾驶飞机自主性方面的潜力。 

---
# A Koopman Operator-based NMPC Framework for Mobile Robot Navigation under Uncertainty 

**Title (ZH)**: 基于Koopman算子的移动机器人导航鲁棒NMPC框架 

**Authors**: Xiaobin Zhang, Mohamed Karim Bouafoura, Lu Shi, Konstantinos Karydis  

**Link**: [PDF](https://arxiv.org/pdf/2504.21215)  

**Abstract**: Mobile robot navigation can be challenged by system uncertainty. For example, ground friction may vary abruptly causing slipping, and noisy sensor data can lead to inaccurate feedback control. Traditional model-based methods may be limited when considering such variations, making them fragile to varying types of uncertainty. One way to address this is by leveraging learned prediction models by means of the Koopman operator into nonlinear model predictive control (NMPC). This paper describes the formulation of, and provides the solution to, an NMPC problem using a lifted bilinear model that can accurately predict affine input systems with stochastic perturbations. System constraints are defined in the Koopman space, while the optimization problem is solved in the state space to reduce computational complexity. Training data to estimate the Koopman operator for the system are given via randomized control inputs. The output of the developed method enables closed-loop navigation control over environments populated with obstacles. The effectiveness of the proposed method has been tested through numerical simulations using a wheeled robot with additive stochastic velocity perturbations, Gazebo simulations with a realistic digital twin robot, and physical hardware experiments without knowledge of the true dynamics. 

**Abstract (ZH)**: 移动机器人导航可能受到系统不确定性的影响。例如，地面摩擦可能突然变化导致打滑，且嘈杂的传感器数据可能导致不准确的反馈控制。传统的基于模型的方法在考虑此类变化时可能会受到限制，从而使其对不同类型的不确定性变得脆弱。一种解决方法是通过Koopman算子来利用学习到的预测模型来实现非线性模型预测控制（NMPC）。本文描述了使用提升的双线性模型来制定并解决一个NMPC问题，该模型可以准确预测具有随机扰动的仿射输入系统。系统约束在Koopman空间中定义，而优化问题在状态空间中求解以降低计算复杂度。通过随机控制输入来估计系统的Koopman算子。所开发方法的输出使闭环导航控制能够应用于具有障碍物的环境中。通过带有加性随机速度扰动的轮式机器人数值仿真、使用现实数字双胞胎机器人的Gazebo仿真，以及无需了解真实动力学的物理硬件实验，验证了所提出方法的有效性。 

---
# Automated Parking Trajectory Generation Using Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的自动化停车轨迹生成 

**Authors**: Zheyu Zhang, Yutong Luo, Yongzhou Chen, Haopeng Zhao, Zhichao Ma, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21071)  

**Abstract**: Autonomous parking is a key technology in modern autonomous driving systems, requiring high precision, strong adaptability, and efficiency in complex environments. This paper proposes a Deep Reinforcement Learning (DRL) framework based on the Soft Actor-Critic (SAC) algorithm to optimize autonomous parking tasks. SAC, an off-policy method with entropy regularization, is particularly well-suited for continuous action spaces, enabling fine-grained vehicle control. We model the parking task as a Markov Decision Process (MDP) and train an agent to maximize cumulative rewards while balancing exploration and exploitation through entropy maximization. The proposed system integrates multiple sensor inputs into a high-dimensional state space and leverages SAC's dual critic networks and policy network to achieve stable learning. Simulation results show that the SAC-based approach delivers high parking success rates, reduced maneuver times, and robust handling of dynamic obstacles, outperforming traditional rule-based methods and other DRL algorithms. This study demonstrates SAC's potential in autonomous parking and lays the foundation for real-world applications. 

**Abstract (ZH)**: 基于Soft Actor-Critic算法的深度强化学习自主泊车框架 

---
# Designing Control Barrier Function via Probabilistic Enumeration for Safe Reinforcement Learning Navigation 

**Title (ZH)**: 基于概率枚举的控制障碍函数设计以实现安全的强化学习导航 

**Authors**: Luca Marzari, Francesco Trotti, Enrico Marchesini, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.21643)  

**Abstract**: Achieving safe autonomous navigation systems is critical for deploying robots in dynamic and uncertain real-world environments. In this paper, we propose a hierarchical control framework leveraging neural network verification techniques to design control barrier functions (CBFs) and policy correction mechanisms that ensure safe reinforcement learning navigation policies. Our approach relies on probabilistic enumeration to identify unsafe regions of operation, which are then used to construct a safe CBF-based control layer applicable to arbitrary policies. We validate our framework both in simulation and on a real robot, using a standard mobile robot benchmark and a highly dynamic aquatic environmental monitoring task. These experiments demonstrate the ability of the proposed solution to correct unsafe actions while preserving efficient navigation behavior. Our results show the promise of developing hierarchical verification-based systems to enable safe and robust navigation behaviors in complex scenarios. 

**Abstract (ZH)**: 实现安全自主导航系统对于在动态和不确定的现实环境中部署机器人至关重要。本文提出了一种层次控制框架，利用神经网络验证技术设计控制障碍函数（CBFs）及策略修正机制，以确保安全的强化学习导航策略。我们的方法依赖于概率枚举来识别操作的不安全区域，进而构建适用于任意策略的安全CBFs控制层。我们在仿真和实际机器人上对框架进行了验证，使用标准的移动机器人基准和高度动态的水下环境监测任务。这些实验表明，所提出的方法能够纠正不安全的行为同时保持高效的导航行为。我们的结果展示了基于层次验证的系统在复杂场景中实现安全和稳健的导航行为的潜力。 

---
# Leveraging Systems and Control Theory for Social Robotics: A Model-Based Behavioral Control Approach to Human-Robot Interaction 

**Title (ZH)**: 利用系统与控制理论进行社会机器人研究：基于模型的行为控制方法在人机交互中的应用 

**Authors**: Maria Morão Patrício, Anahita Jamshidnejad  

**Link**: [PDF](https://arxiv.org/pdf/2504.21548)  

**Abstract**: Social robots (SRs) should autonomously interact with humans, while exhibiting proper social behaviors associated to their role. By contributing to health-care, education, and companionship, SRs will enhance life quality. However, personalization and sustaining user engagement remain a challenge for SRs, due to their limited understanding of human mental states. Accordingly, we leverage a recently introduced mathematical dynamic model of human perception, cognition, and decision-making for SRs. Identifying the parameters of this model and deploying it in behavioral steering system of SRs allows to effectively personalize the responses of SRs to evolving mental states of their users, enhancing long-term engagement and personalization. Our approach uniquely enables autonomous adaptability of SRs by modeling the dynamics of invisible mental states, significantly contributing to the transparency and awareness of SRs. We validated our model-based control system in experiments with 10 participants who interacted with a Nao robot over three chess puzzle sessions, 45 - 90 minutes each. The identified model achieved a mean squared error (MSE) of 0.067 (i.e., 1.675% of the maximum possible MSE) in tracking beliefs, goals, and emotions of participants. Compared to a model-free controller that did not track mental states of participants, our approach increased engagement by 16% on average. Post-interaction feedback of participants (provided via dedicated questionnaires) further confirmed the perceived engagement and awareness of the model-driven robot. These results highlight the unique potential of model-based approaches and control theory in advancing human-SR interactions. 

**Abstract (ZH)**: 社会机器人（SRs）应自主与人类互动，并表现出与其角色相关的适当社会行为。通过在健康care、教育和陪伴等方面发挥作用，SRs将提高生活质量。然而，个性化和维持用户参与度仍然是SRs面临的挑战，原因在于它们对人类心理状态的有限理解。因此，我们利用一种最近引入的人类感知、认知和决策过程的数学动态模型为SRs发挥作用。通过识别该模型的参数并在SRs的行为引导系统中部署它，可以有效地使SRs的响应个性化，以适应用户不断变化的心理状态，从而增强长期的参与度和个性化程度。我们的方法独特地使SRs具备了对无形心理状态动态建模的自主适应能力，显著提升了SRs的透明度和意识水平。我们通过与10名参与者进行实验来验证基于模型的控制系统，参与者与一个NAO机器人进行了三次国际象棋难题会话（每次会话时长45-90分钟）。识别出的模型在跟踪参与者信念、目标和情绪时的均方误差（MSE）为0.067（即最大可能MSE的1.675%）。与不跟踪参与者心理状态的无模型控制器相比，我们的方法平均增加了16%的参与度。参与者在互动后的反馈（通过专门的问卷提供）进一步证实了模型驱动机器人感知到的交互和意识。这些结果突显了基于模型的方法和控制理论在提升人类-SR互动方面的独特潜力。 

---
# PhysicsFC: Learning User-Controlled Skills for a Physics-Based Football Player Controller 

**Title (ZH)**: PhysicsFC：学习用户控制技能的物理基础足球玩家控制器 

**Authors**: Minsu Kim, Eunho Jung, Yoonsang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.21216)  

**Abstract**: We propose PhysicsFC, a method for controlling physically simulated football player characters to perform a variety of football skills--such as dribbling, trapping, moving, and kicking--based on user input, while seamlessly transitioning between these skills. Our skill-specific policies, which generate latent variables for each football skill, are trained using an existing physics-based motion embedding model that serves as a foundation for reproducing football motions. Key features include a tailored reward design for the Dribble policy, a two-phase reward structure combined with projectile dynamics-based initialization for the Trap policy, and a Data-Embedded Goal-Conditioned Latent Guidance (DEGCL) method for the Move policy. Using the trained skill policies, the proposed football player finite state machine (PhysicsFC FSM) allows users to interactively control the character. To ensure smooth and agile transitions between skill policies, as defined in the FSM, we introduce the Skill Transition-Based Initialization (STI), which is applied during the training of each skill policy. We develop several interactive scenarios to showcase PhysicsFC's effectiveness, including competitive trapping and dribbling, give-and-go plays, and 11v11 football games, where multiple PhysicsFC agents produce natural and controllable physics-based football player behaviors. Quantitative evaluations further validate the performance of individual skill policies and the transitions between them, using the presented metrics and experimental designs. 

**Abstract (ZH)**: 我们提出PhysicsFC方法，该方法基于用户输入控制物理模拟的足球球员角色执行多种足球技能——如带球、控球、移动和射门——同时无缝过渡到这些技能。关键特征包括为Dribble策略量身设计的奖励设计、Trap策略中结合轨迹动力学初始化的两阶段奖励结构以及嵌入数据的目标条件潜变量指导（DEGCL）方法。利用训练好的技能策略，提出的足球玩家有限状态机（PhysicsFC FSM）允许用户交互控制角色。为了确保有限状态机（FSM）中定义的技能策略之间的平滑和灵活过渡，我们引入了技能过渡基于初始化（STI），并将其应用于每个技能策略的训练中。我们开发了多个交互场景以展示PhysicsFC的有效性，包括竞争性控球和带球、传切配合以及11对11足球比赛，其中多个PhysicsFC代理产生自然可控的基于物理的足球玩家行为。定量评估进一步验证了每个技能策略及其过渡之间的性能，使用呈现的度量标准和实验设计。 

---
# IRL Dittos: Embodied Multimodal AI Agent Interactions in Open Spaces 

**Title (ZH)**: IRL Ditto们：开放空间中的具身多模态AI代理互动 

**Authors**: Seonghee Lee, Denae Ford, John Tang, Sasa Junuzovic, Asta Roseway, Ed Cutrell, Kori Inkpen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21347)  

**Abstract**: We introduce the In Real Life (IRL) Ditto, an AI-driven embodied agent designed to represent remote colleagues in shared office spaces, creating opportunities for real-time exchanges even in their absence. IRL Ditto offers a unique hybrid experience by allowing in-person colleagues to encounter a digital version of their remote teammates, initiating greetings, updates, or small talk as they might in person. Our research question examines: How can the IRL Ditto influence interactions and relationships among colleagues in a shared office space? Through a four-day study, we assessed IRL Ditto's ability to strengthen social ties by simulating presence and enabling meaningful interactions across different levels of social familiarity. We find that enhancing social relationships depended deeply on the foundation of the relationship participants had with the source of the IRL Ditto. This study provides insights into the role of embodied agents in enriching workplace dynamics for distributed teams. 

**Abstract (ZH)**: We introduce the In Real Life (IRL) Ditto, 一个由AI驱动的具身代理，旨在代表远程同事在共享办公空间中参与，即使在远程同事缺席的情况下，也能创造实时交流的机会。IRL Ditto提供了一种独特的混合体验，允许在场同事遇到其远程队友的数字化版本，通过问候、更新或闲聊等方式进行互动。本研究的问题是：IRL Ditto如何影响共享办公空间中同事之间的互动和关系？通过为期四天的研究，我们评估了IRL Ditto增强社会纽带、模拟存在感并促进不同熟悉程度下的有意义互动的能力。我们发现，增强社会关系在很大程度上取决于参与者与IRL Ditto来源的关系基础。本研究提供了有关具身代理在丰富分布式团队的职场动态方面作用的见解。 

---
# Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models 

**Title (ZH)**: 强化多模态大语言模型：基于RL的推理综述 

**Authors**: Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, Minghui Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21277)  

**Abstract**: The integration of reinforcement learning (RL) into the reasoning capabilities of Multimodal Large Language Models (MLLMs) has rapidly emerged as a transformative research direction. While MLLMs significantly extend Large Language Models (LLMs) to handle diverse modalities such as vision, audio, and video, enabling robust reasoning across multimodal inputs remains a major challenge. This survey systematically reviews recent advances in RL-based reasoning for MLLMs, covering key algorithmic designs, reward mechanism innovations, and practical applications. We highlight two main RL paradigms--value-free and value-based methods--and analyze how RL enhances reasoning abilities by optimizing reasoning trajectories and aligning multimodal information. Furthermore, we provide an extensive overview of benchmark datasets, evaluation protocols, and existing limitations, and propose future research directions to address current bottlenecks such as sparse rewards, inefficient cross-modal reasoning, and real-world deployment constraints. Our goal is to offer a comprehensive and structured guide to researchers interested in advancing RL-based reasoning in the multimodal era. 

**Abstract (ZH)**: 将强化学习（RL）整合到多模态大规模语言模型（MLLMs）的推理能力中，已成为一项 transformative 研究方向。尽管 MLLMs 显著扩展了大规模语言模型（LLMs），使其能够处理多种模态如视觉、音频和视频，但跨模态输入的稳健推理仍然是一个主要挑战。本文系统地回顾了基于 RL 的推理方法在 MLLMs 中的最新进展，涵盖关键算法设计、奖励机制创新和实际应用。我们着重介绍了两种主要的 RL 原理——价值自由方法和价值依赖方法，并分析了 RL 如何通过优化推理轨迹和对齐多模态信息来增强推理能力。此外，本文还提供了基准数据集、评估协议的详尽概述及现有局限性，并提出了未来研究方向，以解决当前瓶颈问题，如稀疏奖励、低效的跨模态推理和现实世界部署限制。我们的目标是为有兴趣深化多模态时代基于 RL 的推理研究的科研人员提供一份全面和结构化的指南。 

---
# Theoretical Foundations for Semantic Cognition in Artificial Intelligence 

**Title (ZH)**: 人工智能中语义认知的理论基础 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2504.21218)  

**Abstract**: This monograph presents a modular cognitive architecture for artificial intelligence grounded in the formal modeling of belief as structured semantic state. Belief states are defined as dynamic ensembles of linguistic expressions embedded within a navigable manifold, where operators enable assimilation, abstraction, nullification, memory, and introspection. Drawing from philosophy, cognitive science, and neuroscience, we develop a layered framework that enables self-regulating epistemic agents capable of reflective, goal-directed thought. At the core of this framework is the epistemic vacuum: a class of semantically inert cognitive states that serves as the conceptual origin of belief space. From this foundation, the Null Tower arises as a generative structure recursively built through internal representational capacities. The theoretical constructs are designed to be implementable in both symbolic and neural systems, including large language models, hybrid agents, and adaptive memory architectures. This work offers a foundational substrate for constructing agents that reason, remember, and regulate their beliefs in structured, interpretable ways. 

**Abstract (ZH)**: 本论著介绍了基于正式的信念结构语义建模的人工智能模块化认知架构。该架构定义了动态的语言表达嵌入可导航流形中的信念状态，并通过操作符实现同化、抽象、消除、记忆和反省。借鉴哲学、认知科学和神经科学，我们构建了一个分层框架，使能够进行反思性、目标导向思考的自主知识代理成为可能。该框架的核心是知识真空：一类语义中立的认知状态，作为信念空间的概念起点。在此基础上，虚塔作为生成结构，通过内部表征能力递归构建。这些理论构造既可用于符号系统，也可用于神经系统，包括大型语言模型、混合代理和自适应记忆架构。本研究提供了构建以结构化和可解释方式推理、记忆和调节信念的代理的基础框架。 

---
# DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition 

**Title (ZH)**: DeepSeek-Prover-V2: 通过强化学习促进子目标分解的正式数学推理优化 

**Authors**: Z.Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z.F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, Chong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21801)  

**Abstract**: We introduce DeepSeek-Prover-V2, an open-source large language model designed for formal theorem proving in Lean 4, with initialization data collected through a recursive theorem proving pipeline powered by DeepSeek-V3. The cold-start training procedure begins by prompting DeepSeek-V3 to decompose complex problems into a series of subgoals. The proofs of resolved subgoals are synthesized into a chain-of-thought process, combined with DeepSeek-V3's step-by-step reasoning, to create an initial cold start for reinforcement learning. This process enables us to integrate both informal and formal mathematical reasoning into a unified model. The resulting model, DeepSeek-Prover-V2-671B, achieves state-of-the-art performance in neural theorem proving, reaching 88.9% pass ratio on the MiniF2F-test and solving 49 out of 658 problems from PutnamBench. In addition to standard benchmarks, we introduce ProverBench, a collection of 325 formalized problems, to enrich our evaluation, including 15 selected problems from the recent AIME competitions (years 24-25). Further evaluation on these 15 AIME problems shows that the model successfully solves 6 of them. In comparison, DeepSeek-V3 solves 8 of these problems using majority voting, highlighting that the gap between formal and informal mathematical reasoning in large language models is substantially narrowing. 

**Abstract (ZH)**: DeepSeek-Prover-V2：一种基于Lean 4的开源大型语言模型及其在形式定理证明中的应用 

---
# WebThinker: Empowering Large Reasoning Models with Deep Research Capability 

**Title (ZH)**: WebThinker: 道德深厚推理能力赋能大型模型 

**Authors**: Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21776)  

**Abstract**: Large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, demonstrate impressive long-horizon reasoning capabilities. However, their reliance on static internal knowledge limits their performance on complex, knowledge-intensive tasks and hinders their ability to produce comprehensive research reports requiring synthesis of diverse web information. To address this, we propose \textbf{WebThinker}, a deep research agent that empowers LRMs to autonomously search the web, navigate web pages, and draft research reports during the reasoning process. WebThinker integrates a \textbf{Deep Web Explorer} module, enabling LRMs to dynamically search, navigate, and extract information from the web when encountering knowledge gaps. It also employs an \textbf{Autonomous Think-Search-and-Draft strategy}, allowing the model to seamlessly interleave reasoning, information gathering, and report writing in real time. To further enhance research tool utilization, we introduce an \textbf{RL-based training strategy} via iterative online Direct Preference Optimization (DPO). Extensive experiments on complex reasoning benchmarks (GPQA, GAIA, WebWalkerQA, HLE) and scientific report generation tasks (Glaive) demonstrate that WebThinker significantly outperforms existing methods and strong proprietary systems. Our approach enhances LRM reliability and applicability in complex scenarios, paving the way for more capable and versatile deep research systems. The code is available at this https URL. 

**Abstract (ZH)**: WebThinker：一种增强大型推理模型的深度研究代理 

---
# Self-Supervised Monocular Visual Drone Model Identification through Improved Occlusion Handling 

**Title (ZH)**: 自我监督单目视觉无人机模型识别通过改进遮挡处理 

**Authors**: Stavrow A. Bahnam, Christophe De Wagter, Guido C.H.E. de Croon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21695)  

**Abstract**: Ego-motion estimation is vital for drones when flying in GPS-denied environments. Vision-based methods struggle when flight speed increases and close-by objects lead to difficult visual conditions with considerable motion blur and large occlusions. To tackle this, vision is typically complemented by state estimation filters that combine a drone model with inertial measurements. However, these drone models are currently learned in a supervised manner with ground-truth data from external motion capture systems, limiting scalability to different environments and drones. In this work, we propose a self-supervised learning scheme to train a neural-network-based drone model using only onboard monocular video and flight controller data (IMU and motor feedback). We achieve this by first training a self-supervised relative pose estimation model, which then serves as a teacher for the drone model. To allow this to work at high speed close to obstacles, we propose an improved occlusion handling method for training self-supervised pose estimation models. Due to this method, the root mean squared error of resulting odometry estimates is reduced by an average of 15%. Moreover, the student neural drone model can be successfully obtained from the onboard data. It even becomes more accurate at higher speeds compared to its teacher, the self-supervised vision-based model. We demonstrate the value of the neural drone model by integrating it into a traditional filter-based VIO system (ROVIO), resulting in superior odometry accuracy on aggressive 3D racing trajectories near obstacles. Self-supervised learning of ego-motion estimation represents a significant step toward bridging the gap between flying in controlled, expensive lab environments and real-world drone applications. The fusion of vision and drone models will enable higher-speed flight and improve state estimation, on any drone in any environment. 

**Abstract (ZH)**: 基于自我监督学习的无人机 ego-运动估计训练方法 

---
# FAST-Q: Fast-track Exploration with Adversarially Balanced State Representations for Counterfactual Action Estimation in Offline Reinforcement Learning 

**Title (ZH)**: FAST-Q: 快速探索与对抗均衡状态表示在离线强化学习中用于反事实动作评估中的应用 

**Authors**: Pulkit Agrawal, Rukma Talwadker, Aditya Pareek, Tridib Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2504.21383)  

**Abstract**: Recent advancements in state-of-the-art (SOTA) offline reinforcement learning (RL) have primarily focused on addressing function approximation errors, which contribute to the overestimation of Q-values for out-of-distribution actions, a challenge that static datasets exacerbate. However, high stakes applications such as recommendation systems in online gaming, introduce further complexities due to player's psychology (intent) driven by gameplay experiences and the inherent volatility on the platform. These factors create highly sparse, partially overlapping state spaces across policies, further influenced by the experiment path selection logic which biases state spaces towards specific policies. Current SOTA methods constrain learning from such offline data by clipping known counterfactual actions as out-of-distribution due to poor generalization across unobserved states. Further aggravating conservative Q-learning and necessitating more online exploration. FAST-Q introduces a novel approach that (1) leverages Gradient Reversal Learning to construct balanced state representations, regularizing the policy-specific bias between the player's state and action thereby enabling counterfactual estimation; (2) supports offline counterfactual exploration in parallel with static data exploitation; and (3) proposes a Q-value decomposition strategy for multi-objective optimization, facilitating explainable recommendations over short and long-term objectives. These innovations demonstrate superiority of FAST-Q over prior SOTA approaches and demonstrates at least 0.15 percent increase in player returns, 2 percent improvement in lifetime value (LTV), 0.4 percent enhancement in the recommendation driven engagement, 2 percent improvement in the player's platform dwell time and an impressive 10 percent reduction in the costs associated with the recommendation, on our volatile gaming platform. 

**Abstract (ZH)**: FAST-Q：一种新颖的离线强化学习方法及其在在线游戏推荐系统中的应用 

---
# Pretraining Large Brain Language Model for Active BCI: Silent Speech 

**Title (ZH)**: 大型脑语言模型的预训练在主动BCI中的应用：沉默语音 

**Authors**: Jinzhao Zhou, Zehong Cao, Yiqun Duan, Connor Barkley, Daniel Leong, Xiaowei Jiang, Quoc-Toan Nguyen, Ziyi Zhao, Thomas Do, Yu-Cheng Chang, Sheng-Fu Liang, Chin-teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.21214)  

**Abstract**: This paper explores silent speech decoding in active brain-computer interface (BCI) systems, which offer more natural and flexible communication than traditional BCI applications. We collected a new silent speech dataset of over 120 hours of electroencephalogram (EEG) recordings from 12 subjects, capturing 24 commonly used English words for language model pretraining and decoding. Following the recent success of pretraining large models with self-supervised paradigms to enhance EEG classification performance, we propose Large Brain Language Model (LBLM) pretrained to decode silent speech for active BCI. To pretrain LBLM, we propose Future Spectro-Temporal Prediction (FSTP) pretraining paradigm to learn effective representations from unlabeled EEG data. Unlike existing EEG pretraining methods that mainly follow a masked-reconstruction paradigm, our proposed FSTP method employs autoregressive modeling in temporal and frequency domains to capture both temporal and spectral dependencies from EEG signals. After pretraining, we finetune our LBLM on downstream tasks, including word-level and semantic-level classification. Extensive experiments demonstrate significant performance gains of the LBLM over fully-supervised and pretrained baseline models. For instance, in the difficult cross-session setting, our model achieves 47.0\% accuracy on semantic-level classification and 39.6\% in word-level classification, outperforming baseline methods by 5.4\% and 7.3\%, respectively. Our research advances silent speech decoding in active BCI systems, offering an innovative solution for EEG language model pretraining and a new dataset for fundamental research. 

**Abstract (ZH)**: 本文探讨了在主动脑-计算机接口（BCI）系统中静默语音解码的应用，这些系统提供了比传统BCI应用更为自然和灵活的通信方式。我们收集了一个包含超过120小时电生理记录数据的静默语音新数据集，涵盖了24个常用英语单词，用于语言模型的预训练和解码。借鉴近年来利用自监督框架预训练大型模型以提高EEG分类性能的成功经验，我们提出了一个大型脑语言模型（LBLM），专门用于主动BCI的静默语音解码。为预训练LBLM，我们提出了未来谱时预测（FSTP）预训练框架，以从无标签的EEG数据中学习有效的表示。不同于现有主要遵循掩码重建框架的EEG预训练方法，我们提出的FSTP方法利用了时间和频率域的自回归建模来捕捉EEG信号中的时域和频域依赖关系。预训练后，我们在下游任务中对我们的LBLM进行微调，包括字节级和语义级分类。广泛的实验表明，与完全监督和预训练基线模型相比，LBLM在性能上获得了显著提升。例如，在困难的跨会话设置中，我们的模型在语义级分类中达到了47.0%的准确率，在字节级分类中达到了39.6%的准确率，分别比基线方法高出5.4%和7.3%。我们的研究推进了在主动BCI系统中静默语音解码的应用，提供了一种创新的EEG语言模型预训练解决方案，并为基本研究提供了新的数据集。 

---
# SAGA: A Security Architecture for Governing AI Agentic Systems 

**Title (ZH)**: SAGA: 一种治理AI代理系统的安全架构 

**Authors**: Georgios Syros, Anshuman Suri, Cristina Nita-Rotaru, Alina Oprea  

**Link**: [PDF](https://arxiv.org/pdf/2504.21034)  

**Abstract**: Large Language Model (LLM)-based agents increasingly interact, collaborate, and delegate tasks to one another autonomously with minimal human interaction. Industry guidelines for agentic system governance emphasize the need for users to maintain comprehensive control over their agents, mitigating potential damage from malicious agents. Several proposed agentic system designs address agent identity, authorization, and delegation, but remain purely theoretical, without concrete implementation and evaluation. Most importantly, they do not provide user-controlled agent management. To address this gap, we propose SAGA, a Security Architecture for Governing Agentic systems, that offers user oversight over their agents' lifecycle. In our design, users register their agents with a central entity, the Provider, that maintains agents contact information, user-defined access control policies, and helps agents enforce these policies on inter-agent communication. We introduce a cryptographic mechanism for deriving access control tokens, that offers fine-grained control over an agent's interaction with other agents, balancing security and performance consideration. We evaluate SAGA on several agentic tasks, using agents in different geolocations, and multiple on-device and cloud LLMs, demonstrating minimal performance overhead with no impact on underlying task utility in a wide range of conditions. Our architecture enables secure and trustworthy deployment of autonomous agents, accelerating the responsible adoption of this technology in sensitive environments. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的代理系统安全架构：用户控制下的自主代理治理与部署 

---
# ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees 

**Title (ZH)**: 形变NL2LTL：具有形变正确性保证的自然语言指令到时间逻辑公式的翻译 

**Authors**: Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2504.21022)  

**Abstract**: Linear Temporal Logic (LTL) has become a prevalent specification language for robotic tasks. To mitigate the significant manual effort and expertise required to define LTL-encoded tasks, several methods have been proposed for translating Natural Language (NL) instructions into LTL formulas, which, however, lack correctness guarantees. To address this, we introduce a new NL-to-LTL translation method, called ConformalNL2LTL, that can achieve user-defined translation success rates over unseen NL commands. Our method constructs LTL formulas iteratively by addressing a sequence of open-vocabulary Question-Answering (QA) problems with LLMs. To enable uncertainty-aware translation, we leverage conformal prediction (CP), a distribution-free uncertainty quantification tool for black-box models. CP enables our method to assess the uncertainty in LLM-generated answers, allowing it to proceed with translation when sufficiently confident and request help otherwise. We provide both theoretical and empirical results demonstrating that ConformalNL2LTL achieves user-specified translation accuracy while minimizing help rates. 

**Abstract (ZH)**: 基于自然语言到线性时序逻辑的自适应翻译：ConformalNL2LTL 

---
