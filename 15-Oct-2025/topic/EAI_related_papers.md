# T(R,O) Grasp: Efficient Graph Diffusion of Robot-Object Spatial Transformation for Cross-Embodiment Dexterous Grasping 

**Title (ZH)**: T(R,O) 抓取：机器人与物体空间变换的高效图扩散方法及其在跨躯体灵巧抓取中的应用 

**Authors**: Xin Fei, Zhixuan Xu, Huaicong Fang, Tianrui Zhang, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2510.12724)  

**Abstract**: Dexterous grasping remains a central challenge in robotics due to the complexity of its high-dimensional state and action space. We introduce T(R,O) Grasp, a diffusion-based framework that efficiently generates accurate and diverse grasps across multiple robotic hands. At its core is the T(R,O) Graph, a unified representation that models spatial transformations between robotic hands and objects while encoding their geometric properties. A graph diffusion model, coupled with an efficient inverse kinematics solver, supports both unconditioned and conditioned grasp synthesis. Extensive experiments on a diverse set of dexterous hands show that T(R,O) Grasp achieves average success rate of 94.83%, inference speed of 0.21s, and throughput of 41 grasps per second on an NVIDIA A100 40GB GPU, substantially outperforming existing baselines. In addition, our approach is robust and generalizable across embodiments while significantly reducing memory consumption. More importantly, the high inference speed enables closed-loop dexterous manipulation, underscoring the potential of T(R,O) Grasp to scale into a foundation model for dexterous grasping. 

**Abstract (ZH)**: 灵巧抓取仍然是机器人领域的核心挑战，由于其高维状态和动作空间的复杂性。我们提出了一种基于扩散的框架T(R,O) Grasp，该框架高效地生成多机器人手上的准确且多样化的抓取。其核心是T(R,O)图，这是一种统一表示，能够建模机器人手与物体之间的空间变换并编码其几何特性。通过结合图扩散模型与高效的逆运动学求解器，支持无条件和有条件抓取合成。在一系列灵巧手的广泛实验中，T(R,O) Grasp 达到了94.83%的平均成功率，推理速度为0.21秒，并在NVIDIA A100 40GB GPU上实现了每秒41个抓取的吞吐量，显著优于现有基线。此外，我们的方法在不同实体间具有鲁棒性和泛化性，同时显著降低了内存消耗。更重要的是，高推理速度使闭环灵巧操作成为可能，突显了T(R,O) Grasp作为灵巧抓取基础模型的潜力。 

---
# Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control 

**Title (ZH)**: 残差MPC：将强化学习与GPU并行模型预测控制相结合 

**Authors**: Se Hwan Jeon, Ho Jae Lee, Seungwoo Hong, Sangbae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.12717)  

**Abstract**: Model Predictive Control (MPC) provides interpretable, tunable locomotion controllers grounded in physical models, but its robustness depends on frequent replanning and is limited by model mismatch and real-time computational constraints. Reinforcement Learning (RL), by contrast, can produce highly robust behaviors through stochastic training but often lacks interpretability, suffers from out-of-distribution failures, and requires intensive reward engineering. This work presents a GPU-parallelized residual architecture that tightly integrates MPC and RL by blending their outputs at the torque-control level. We develop a kinodynamic whole-body MPC formulation evaluated across thousands of agents in parallel at 100 Hz for RL training. The residual policy learns to make targeted corrections to the MPC outputs, combining the interpretability and constraint handling of model-based control with the adaptability of RL. The model-based control prior acts as a strong bias, initializing and guiding the policy towards desirable behavior with a simple set of rewards. Compared to standalone MPC or end-to-end RL, our approach achieves higher sample efficiency, converges to greater asymptotic rewards, expands the range of trackable velocity commands, and enables zero-shot adaptation to unseen gaits and uneven terrain. 

**Abstract (ZH)**: 基于GPU并行化残差架构的 MPC与RL融合控制 

---
# Reflection-Based Task Adaptation for Self-Improving VLA 

**Title (ZH)**: 基于反射的任务适应性学习以实现自我提升的VLAD 

**Authors**: Baicheng Li, Dong Wu, Zike Yan, Xinchen Liu, Zecui Zeng, Lusong Li, Hongbin Zha  

**Link**: [PDF](https://arxiv.org/pdf/2510.12710)  

**Abstract**: Pre-trained Vision-Language-Action (VLA) models represent a major leap towards general-purpose robots, yet efficiently adapting them to novel, specific tasks in-situ remains a significant hurdle. While reinforcement learning (RL) is a promising avenue for such adaptation, the process often suffers from low efficiency, hindering rapid task mastery. We introduce Reflective Self-Adaptation, a framework for rapid, autonomous task adaptation without human intervention. Our framework establishes a self-improving loop where the agent learns from its own experience to enhance both strategy and execution.
The core of our framework is a dual-pathway architecture that addresses the full adaptation lifecycle. First, a Failure-Driven Reflective RL pathway enables rapid learning by using the VLM's causal reasoning to automatically synthesize a targeted, dense reward function from failure analysis. This provides a focused learning signal that significantly accelerates policy exploration. However, optimizing such proxy rewards introduces a potential risk of "reward hacking," where the agent masters the reward function but fails the actual task. To counteract this, our second pathway, Success-Driven Quality-Guided SFT, grounds the policy in holistic success. It identifies and selectively imitates high-quality successful trajectories, ensuring the agent remains aligned with the ultimate task goal. This pathway is strengthened by a conditional curriculum mechanism to aid initial exploration.
We conduct experiments in challenging manipulation tasks. The results demonstrate that our framework achieves faster convergence and higher final success rates compared to representative baselines. Our work presents a robust solution for creating self-improving agents that can efficiently and reliably adapt to new environments. 

**Abstract (ZH)**: 预训练视觉-语言-行动（VLA）模型代表了通用机器人的一大进步，但在原位快速适应新型特定任务仍是一项重大挑战。尽管强化学习（RL）为这种适应提供了前景，但这一过程往往效率低下，阻碍了快速掌握任务。我们提出了反思自适应框架，一种不需要人类干预即可实现快速自主任务适应的方法。该框架建立了一个自我改进的循环，通过让智能体从自身经验中学习来提升策略和执行。

我们框架的核心是双路径架构，涵盖了完整的学习生命周期。首先，基于失败的反思RL路径通过利用VLM的因果推理来自动合成针对失败分析的密集奖励函数，从而提供集中学习信号，显著加快策略探索。然而，优化此类代理奖励增加了“奖励黑客”的风险，即智能体掌握了奖励函数但未能完成实际任务。为应对这一风险，我们引入了第二个路径——基于成功的质量引导的SFT路径，该路径使政策与整体成功保持一致。它识别并选择性地模仿高质量的成功轨迹，确保智能体始终与最终任务目标保持一致。该路径通过条件性课程机制来辅助初始探索。

我们在具有挑战性的操控任务中进行了实验。结果表明，我们的框架在收敛速度和最终成功率方面优于代表性的基线方法。本文提出了一种稳健的解决方案，用于创建能够高效可靠地适应新环境的自我改进智能体。 

---
# Autonomous Legged Mobile Manipulation for Lunar Surface Operations via Constrained Reinforcement Learning 

**Title (ZH)**: 基于约束强化学习的月表自主腿式移动操作Manipulation 

**Authors**: Alvaro Belmonte-Baeza, Miguel Cazorla, Gabriel J. García, Carlos J. Pérez-Del-Pulgar, Jorge Pomares  

**Link**: [PDF](https://arxiv.org/pdf/2510.12684)  

**Abstract**: Robotics plays a pivotal role in planetary science and exploration, where autonomous and reliable systems are crucial due to the risks and challenges inherent to space environments. The establishment of permanent lunar bases demands robotic platforms capable of navigating and manipulating in the harsh lunar terrain. While wheeled rovers have been the mainstay for planetary exploration, their limitations in unstructured and steep terrains motivate the adoption of legged robots, which offer superior mobility and adaptability. This paper introduces a constrained reinforcement learning framework designed for autonomous quadrupedal mobile manipulators operating in lunar environments. The proposed framework integrates whole-body locomotion and manipulation capabilities while explicitly addressing critical safety constraints, including collision avoidance, dynamic stability, and power efficiency, in order to ensure robust performance under lunar-specific conditions, such as reduced gravity and irregular terrain. Experimental results demonstrate the framework's effectiveness in achieving precise 6D task-space end-effector pose tracking, achieving an average positional accuracy of 4 cm and orientation accuracy of 8.1 degrees. The system consistently respects both soft and hard constraints, exhibiting adaptive behaviors optimized for lunar gravity conditions. This work effectively bridges adaptive learning with essential mission-critical safety requirements, paving the way for advanced autonomous robotic explorers for future lunar missions. 

**Abstract (ZH)**: 机器人技术在行星科学与探测中发挥着关键作用，由于太空环境固有的风险和挑战，自主且可靠的系统至关重要。建立永久月球基地需要能够在恶劣月球地形中导航和操作的机器人平台。尽管履带式探测车一直是行星探索的主要工具，但在不规则和陡峭地形中的局限性促使采用腿足式机器人，后者提供了更好的机动性和适应性。本文介绍了一种针对月球环境中的自主四足移动 manipulator 设计的约束强化学习框架。该框架整合了全身运动和操作能力，并明确解决了包括碰撞避免、动态稳定性和能量效率在内的关键安全约束，以确保在月球特有的条件下（如低重力和不规则地形）实现稳健性能。实验结果表明，该框架在实现精确的6D任务空间末端执行器姿态跟踪方面有效，平均位置精度达到4厘米，姿态精度为8.1度。该系统始终尊重软约束和硬约束，表现出针对月球重力条件优化的适应性行为。这项工作有效地将适应性学习与至关重要的任务安全要求相结合，为未来的月球任务铺平了道路，开启了先进自主机器人探测器的新篇章。 

---
# Maximal Adaptation, Minimal Guidance: Permissive Reactive Robot Task Planning with Humans in the Loop 

**Title (ZH)**: 最大化适应，最小干预：人类参与下的宽松反应式机器人任务规划 

**Authors**: Oz Gitelson, Satya Prakash Nayak, Ritam Raha, Anne-Kathrin Schmuck  

**Link**: [PDF](https://arxiv.org/pdf/2510.12662)  

**Abstract**: We present a novel framework for human-robot \emph{logical} interaction that enables robots to reliably satisfy (infinite horizon) temporal logic tasks while effectively collaborating with humans who pursue independent and unknown tasks. The framework combines two key capabilities: (i) \emph{maximal adaptation} enables the robot to adjust its strategy \emph{online} to exploit human behavior for cooperation whenever possible, and (ii) \emph{minimal tunable feedback} enables the robot to request cooperation by the human online only when necessary to guarantee progress. This balance minimizes human-robot interference, preserves human autonomy, and ensures persistent robot task satisfaction even under conflicting human goals. We validate the approach in a real-world block-manipulation task with a Franka Emika Panda robotic arm and in the Overcooked-AI benchmark, demonstrating that our method produces rich, \emph{emergent} cooperative behaviors beyond the reach of existing approaches, while maintaining strong formal guarantees. 

**Abstract (ZH)**: 我们提出了一种新型框架，实现人类与机器人之间的逻辑交互，使机器人能够在可靠地完成无限时间逻辑任务的同时，有效与追求独立且未知任务的人类协作。该框架结合了两项关键能力：（i）最大程度的适应性使机器人能够在线调整策略，尽可能利用人类行为进行合作，（ii）最小可调反馈使机器人仅在必要时请求人类在线合作以确保进度。这种平衡减少了人类与机器人之间的干扰，保留下了人类的自主性，并在人类目标冲突的情况下仍能确保机器人任务的持续满足。我们在Franka Emika Panda机械臂上的实际物件操作任务和Overcooked-AI基准测试中验证了该方法，证明了我们的方法产生了丰富的、涌现性的合作行为，超越了现有方法的范畴，同时保持了强大的形式保证。 

---
# Designing Tools with Control Confidence 

**Title (ZH)**: 设计具有控制信心的工具 

**Authors**: Ajith Anil Meera, Abian Torres, Pablo Lanillos  

**Link**: [PDF](https://arxiv.org/pdf/2510.12630)  

**Abstract**: Prehistoric humans invented stone tools for specialized tasks by not just maximizing the tool's immediate goal-completion accuracy, but also increasing their confidence in the tool for later use under similar settings. This factor contributed to the increased robustness of the tool, i.e., the least performance deviations under environmental uncertainties. However, the current autonomous tool design frameworks solely rely on performance optimization, without considering the agent's confidence in tool use for repeated use. Here, we take a step towards filling this gap by i) defining an optimization framework for task-conditioned autonomous hand tool design for robots, where ii) we introduce a neuro-inspired control confidence term into the optimization routine that helps the agent to design tools with higher robustness. Through rigorous simulations using a robotic arm, we show that tools designed with control confidence as the objective function are more robust to environmental uncertainties during tool use than a pure accuracy-driven objective. We further show that adding control confidence to the objective function for tool design provides a balance between the robustness and goal accuracy of the designed tools under control perturbations. Finally, we show that our CMAES-based evolutionary optimization strategy for autonomous tool design outperforms other state-of-the-art optimizers by designing the optimal tool within the fewest iterations. Code: this https URL. 

**Abstract (ZH)**: 史前人类通过不仅最大化工具的即时目标完成准确性，还提高其在相似环境条件下Later使用时的信心，来专门为特定任务发明石器。这一因素增加了工具的鲁棒性，即在环境不确定性下的最小性能偏差。然而，当前的自主工具设计框架仅依赖于性能优化，而不考虑执行者在重复使用工具时对该工具的信心。在此，我们通过(i) 定义一种任务条件下的自主手工具设计优化框架，以及(ii) 在优化过程中引入灵感于神经系统的控制信心项，来填补这一缺口，从而帮助代理设计具有更高鲁棒性的工具。通过使用机械臂进行严格的仿真，我们表明，以控制信心为目标函数设计的工具在使用过程中对环境不确定性具有更强的鲁棒性，优于纯准确性驱动的目标函数。我们进一步表明，将控制信心添加到设计工具的目标函数中，在控制扰动下为设计工具提供了鲁棒性和目标准确性之间的平衡。最后，我们展示了基于CMAES的自主工具设计进化优化策略在最少迭代次数内设计出最优工具，优于其他最先进的优化器。代码: https://this-url.com。 

---
# Two-stream network-driven vision-based tactile sensor for object feature extraction and fusion perception 

**Title (ZH)**: 基于视觉的双流网络驱动触觉传感器及其特征提取与融合感知 

**Authors**: Muxing Huang, Zibin Chen, Weiliang Xu, Zilan Li, Yuanzhi Zhou, Guoyuan Zhou, Wenjing Chen, Xinming Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.12528)  

**Abstract**: Tactile perception is crucial for embodied intelligent robots to recognize objects. Vision-based tactile sensors extract object physical attributes multidimensionally using high spatial resolution; however, this process generates abundant redundant information. Furthermore, single-dimensional extraction, lacking effective fusion, fails to fully characterize object attributes. These challenges hinder the improvement of recognition accuracy. To address this issue, this study introduces a two-stream network feature extraction and fusion perception strategy for vision-based tactile systems. This strategy employs a distributed approach to extract internal and external object features. It obtains depth map information through three-dimensional reconstruction while simultaneously acquiring hardness information by measuring contact force data. After extracting features with a convolutional neural network (CNN), weighted fusion is applied to create a more informative and effective feature representation. In standard tests on objects of varying shapes and hardness, the force prediction error is 0.06 N (within a 12 N range). Hardness recognition accuracy reaches 98.0%, and shape recognition accuracy reaches 93.75%. With fusion algorithms, object recognition accuracy in actual grasping scenarios exceeds 98.5%. Focused on object physical attributes perception, this method enhances the artificial tactile system ability to transition from perception to cognition, enabling its use in embodied perception applications. 

**Abstract (ZH)**: 触觉感知对于具身智能机器人识別物体至关重要。基于视觉的触觉传感器通过高空间分辨率多维度提取物体物理属性，但这一过程会产生大量冗余信息。此外，单一维度提取缺乏有效融合，无法充分表征物体属性。这些挑战阻碍了识别准确性的提升。为解决这一问题，本研究提出了一种基于视觉的触觉系统两流网络特征提取与融合感知策略。该策略采用分布式方法提取物体内外特征，通过三维重建获取深度图信息，同时通过测量接触力数据获取硬度信息。在使用卷积神经网络（CNN）提取特征后，应用加权融合生成更具信息量和有效性的特征表示。在标准测试中，对于不同形状和硬度的物体，力预测误差为0.06 N（在12 N范围内），硬度识别精度达到98.0%，形状识别精度达到93.75%。通过融合算法，在实际抓取场景中的物体识别精度超过98.5%。专注于物体物理属性感知，该方法增强了人工触觉系统从感知向认知的过渡能力，使其适用于具身感知应用。 

---
# Fast Visuomotor Policy for Robotic Manipulation 

**Title (ZH)**: 快速视知觉运动策略用于机器人操作 

**Authors**: Jingkai Jia, Tong Yang, Xueyao Chen, Chenhuan Liu, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12483)  

**Abstract**: We present a fast and effective policy framework for robotic manipulation, named Energy Policy, designed for high-frequency robotic tasks and resource-constrained systems. Unlike existing robotic policies, Energy Policy natively predicts multimodal actions in a single forward pass, enabling high-precision manipulation at high speed. The framework is built upon two core components. First, we adopt the energy score as the learning objective to facilitate multimodal action modeling. Second, we introduce an energy MLP to implement the proposed objective while keeping the architecture simple and efficient. We conduct comprehensive experiments in both simulated environments and real-world robotic tasks to evaluate the effectiveness of Energy Policy. The results show that Energy Policy matches or surpasses the performance of state-of-the-art manipulation methods while significantly reducing computational overhead. Notably, on the MimicGen benchmark, Energy Policy achieves superior performance with at a faster inference compared to existing approaches. 

**Abstract (ZH)**: 一种面向高频率机器人任务和资源受限系统的快速有效操作框架：能量策略 

---
# A Task-Efficient Reinforcement Learning Task-Motion Planner for Safe Human-Robot Cooperation 

**Title (ZH)**: 一种高效的学习驱动任务-运动规划器，用于安全的人机合作 

**Authors**: Gaoyuan Liu, Joris de Winter, Kelly Merckaert, Denis Steckelmacher, Ann Nowe, Bram Vanderborght  

**Link**: [PDF](https://arxiv.org/pdf/2510.12477)  

**Abstract**: In a Human-Robot Cooperation (HRC) environment, safety and efficiency are the two core properties to evaluate robot performance. However, safety mechanisms usually hinder task efficiency since human intervention will cause backup motions and goal failures of the robot. Frequent motion replanning will increase the computational load and the chance of failure. In this paper, we present a hybrid Reinforcement Learning (RL) planning framework which is comprised of an interactive motion planner and a RL task planner. The RL task planner attempts to choose statistically safe and efficient task sequences based on the feedback from the motion planner, while the motion planner keeps the task execution process collision-free by detecting human arm motions and deploying new paths when the previous path is not valid anymore. Intuitively, the RL agent will learn to avoid dangerous tasks, while the motion planner ensures that the chosen tasks are safe. The proposed framework is validated on the cobot in both simulation and the real world, we compare the planner with hard-coded task motion planning methods. The results show that our planning framework can 1) react to uncertain human motions at both joint and task levels; 2) reduce the times of repeating failed goal commands; 3) reduce the total number of replanning requests. 

**Abstract (ZH)**: 在人机协作环境中的混合强化学习规划框架：兼顾安全与效率 

---
# Robot Learning: A Tutorial 

**Title (ZH)**: 机器人学习：教程 

**Authors**: Francesco Capuano, Caroline Pascal, Adil Zouitine, Thomas Wolf, Michel Aractingi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12403)  

**Abstract**: Robot learning is at an inflection point, driven by rapid advancements in machine learning and the growing availability of large-scale robotics data. This shift from classical, model-based methods to data-driven, learning-based paradigms is unlocking unprecedented capabilities in autonomous systems. This tutorial navigates the landscape of modern robot learning, charting a course from the foundational principles of Reinforcement Learning and Behavioral Cloning to generalist, language-conditioned models capable of operating across diverse tasks and even robot embodiments. This work is intended as a guide for researchers and practitioners, and our goal is to equip the reader with the conceptual understanding and practical tools necessary to contribute to developments in robot learning, with ready-to-use examples implemented in $\texttt{lerobot}$. 

**Abstract (ZH)**: 机器人学习正处于一个转折点，驱动因素是机器学习的迅速发展以及大规模机器人数据的日益可用。从基于模型的经典方法向基于数据的学习范式的转变，正在解锁自主系统前所未有的能力。本教程引领读者探索现代机器人学习的全景，从强化学习和行为克隆的基础原理，到能够跨多种任务甚至不同机器人载体进行操作的一般主义、语言条件化模型。本工作旨在为研究人员和实践者提供指导，并旨在让读者掌握参与机器人学习发展的概念理解与实用工具，所有示例均已在$\texttt{lerobot}$中实现。 

---
# Improving Generative Behavior Cloning via Self-Guidance and Adaptive Chunking 

**Title (ZH)**: 通过自我引导和自适应分块改善生成行为克隆 

**Authors**: Junhyuk So, Chiwoong Lee, Shinyoung Lee, Jungseul Ok, Eunhyeok Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.12392)  

**Abstract**: Generative Behavior Cloning (GBC) is a simple yet effective framework for robot learning, particularly in multi-task settings. Recent GBC methods often employ diffusion policies with open-loop (OL) control, where actions are generated via a diffusion process and executed in multi-step chunks without replanning. While this approach has demonstrated strong success rates and generalization, its inherent stochasticity can result in erroneous action sampling, occasionally leading to unexpected task failures. Moreover, OL control suffers from delayed responses, which can degrade performance in noisy or dynamic environments. To address these limitations, we propose two novel techniques to enhance the consistency and reactivity of diffusion policies: (1) self-guidance, which improves action fidelity by leveraging past observations and implicitly promoting future-aware behavior; and (2) adaptive chunking, which selectively updates action sequences when the benefits of reactivity outweigh the need for temporal consistency. Extensive experiments show that our approach substantially improves GBC performance across a wide range of simulated and real-world robotic manipulation tasks. Our code is available at this https URL 

**Abstract (ZH)**: 生成行为克隆（GBC）是一种简单而有效的机器人学习框架，特别适用于多任务设置。最近的GBC方法通常使用具有开环（OL）控制的扩散策略，其中动作通过扩散过程生成，并以多步片段执行而无需重新规划。虽然这种方法在示例成功率和泛化能力上表现出了强大的效果，但其固有的随机性可能导致错误的动作采样，有时会导致意外的任务失败。此外，开环控制在噪声或动态环境中会表现出延迟响应，从而损害性能。为了克服这些限制，我们提出了两种新技术以增强扩散策略的一致性和反应性：（1）自我引导，通过利用过去观察来提高动作的一致性，并隐式促进未来的感知行为；（2）自适应分段，当反应性的益处超过时间一致性需求时，选择性地更新动作序列。广泛的实验结果显示，我们的方法大大提高了GBC在各种模拟和真实世界机器人操作任务中的性能。我们的代码可在以下链接获取：this https URL。 

---
# Controlling Intent Expressiveness in Robot Motion with Diffusion Models 

**Title (ZH)**: 使用扩散模型控制机器人运动的意图表达性 

**Authors**: Wenli Shi, Clemence Grislain, Olivier Sigaud, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2510.12370)  

**Abstract**: Legibility of robot motion is critical in human-robot interaction, as it allows humans to quickly infer a robot's intended goal. Although traditional trajectory generation methods typically prioritize efficiency, they often fail to make the robot's intentions clear to humans. Meanwhile, existing approaches to legible motion usually produce only a single "most legible" trajectory, overlooking the need to modulate intent expressiveness in different contexts. In this work, we propose a novel motion generation framework that enables controllable legibility across the full spectrum, from highly legible to highly ambiguous motions. We introduce a modeling approach based on an Information Potential Field to assign continuous legibility scores to trajectories, and build upon it with a two-stage diffusion framework that first generates paths at specified legibility levels and then translates them into executable robot actions. Experiments in both 2D and 3D reaching tasks demonstrate that our approach produces diverse and controllable motions with varying degrees of legibility, while achieving performance comparable to SOTA. Code and project page: this https URL. 

**Abstract (ZH)**: 机器人运动的可读性在人机交互中至关重要，因为它允许人类快速推断机器人的意图目标。虽然传统的轨迹生成方法通常优先考虑效率，但它们往往未能使机器人的意图清晰地传达给人类。同时，现有的可读性运动方法通常仅生成一条“最可读”的轨迹，忽视了在不同场景下调整意图表达性的需求。在本工作中，我们提出了一种新颖的运动生成框架，使其能够在从非常可读到高度模糊的整个谱系中实现可控的可读性。我们基于信息潜力场的建模方法为轨迹分配连续的可读性评分，并在此基础上构建了一个两阶段扩散框架，首先生成指定可读性水平的路径，然后将其转换为可执行的机器人动作。在二维和三维抓取任务中的实验表明，我们的方法可以生成具有不同可读性程度的多样且可控的运动，同时实现与当前最佳方法相当的性能。代码和项目页面：https://github.com/alibaba/Qwen-motion-generation。 

---
# Pretraining in Actor-Critic Reinforcement Learning for Robot Motion Control 

**Title (ZH)**: 基于预训练的Actor-Critic强化学习在机器人运动控制中的应用 

**Authors**: Jiale Fan, Andrei Cramariuc, Tifanny Portela, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2510.12363)  

**Abstract**: The pretraining-finetuning paradigm has facilitated numerous transformative advancements in artificial intelligence research in recent years. However, in the domain of reinforcement learning (RL) for robot motion control, individual skills are often learned from scratch despite the high likelihood that some generalizable knowledge is shared across all task-specific policies belonging to a single robot embodiment. This work aims to define a paradigm for pretraining neural network models that encapsulate such knowledge and can subsequently serve as a basis for warm-starting the RL process in classic actor-critic algorithms, such as Proximal Policy Optimization (PPO). We begin with a task-agnostic exploration-based data collection algorithm to gather diverse, dynamic transition data, which is then used to train a Proprioceptive Inverse Dynamics Model (PIDM) through supervised learning. The pretrained weights are loaded into both the actor and critic networks to warm-start the policy optimization of actual tasks. We systematically validated our proposed method on seven distinct robot motion control tasks, showing significant benefits to this initialization strategy. Our proposed approach on average improves sample efficiency by 40.1% and task performance by 7.5%, compared to random initialization. We further present key ablation studies and empirical analyses that shed light on the mechanisms behind the effectiveness of our method. 

**Abstract (ZH)**: 预训练-微调范式在近年来的人工智能研究中推动了众多变革性进展。然而，在机器人运动控制的强化学习（RL）领域中，尽管多项任务可能共享某些可迁移的知识，但个体技能通常仍需要从头学习。本工作旨在定义一种预训练神经网络模型的范式，这些模型封装了这种知识，并可作为温启动经典演员-评论家算法（如 proximal policy optimization, PPO）的 basis。我们从任务无关的探索性数据收集算法开始，以收集多样且动态的转换数据，然后通过监督学习训练一种本体感受性逆动力学模型（PIDM）。预训练的权重被加载到演员和评论家网络中，以温启动实际任务的策略优化。我们在七个不同的机器人运动控制任务上系统地验证了我们提出的方法，展示了这种初始化策略的显著优势。与随机初始化相比，我们的方法平均提高了40.1%的数据效率和7.5%的任务性能。我们还展示了关键的消融研究和经验分析，阐明了我们方法有效性的机制。 

---
# PolygMap: A Perceptive Locomotion Framework for Humanoid Robot Stair Climbing 

**Title (ZH)**: PolygMap: 一种用于类人机器人爬楼梯的感知运动框架 

**Authors**: Bingquan Li, Ning Wang, Tianwei Zhang, Zhicheng He, Yucong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12346)  

**Abstract**: Recently, biped robot walking technology has been significantly developed, mainly in the context of a bland walking scheme. To emulate human walking, robots need to step on the positions they see in unknown spaces accurately. In this paper, we present PolyMap, a perception-based locomotion planning framework for humanoid robots to climb stairs. Our core idea is to build a real-time polygonal staircase plane semantic map, followed by a footstep planar using these polygonal plane segments. These plane segmentation and visual odometry are done by multi-sensor fusion(LiDAR, RGB-D camera and IMUs). The proposed framework is deployed on a NVIDIA Orin, which performs 20-30 Hz whole-body motion planning output. Both indoor and outdoor real-scene experiments indicate that our method is efficient and robust for humanoid robot stair climbing. 

**Abstract (ZH)**: 基于感知的类人机器人攀爬楼梯的运动规划框架：PolyMap 

---
# Achieving Meaningful Collaboration: Worker-centered Design of a Physical Human-Robot Collaborative Blending Task 

**Title (ZH)**: 实现有意义的合作：以工人为中心设计的物理人机协作混合任务 

**Authors**: Nicky Mol, Luka Peternel, Alessandro Ianniello, Denis Zatyagov, Auke Nachenius, Stephan Balvert, J. Micah Prendergast, Sara Muscolo, Olger Siebinga, Eva Verhoef, Deborah Forster, David A. Abbink  

**Link**: [PDF](https://arxiv.org/pdf/2510.12340)  

**Abstract**: The use of robots in industrial settings continues to grow, driven by the need to address complex societal challenges such as labor shortages, aging populations, and ever-increasing production demands. In this abstract, we advocate for (and demonstrate) a transdisciplinary approach when considering robotics in the workplace. Transdisciplinarity emphasizes the integration of academic research with pragmatic expertise and embodied experiential knowledge, that prioritize values such as worker wellbeing and job attractiveness. In the following, we describe an ongoing multi-pronged effort to explore the potential of collaborative robots in the context of airplane engine repair and maintenance operations. 

**Abstract (ZH)**: 工业环境中机器人应用的持续增长驱于应对复杂的社会挑战，如劳动力短缺、老龄化人口以及不断增长的生产需求。本文倡导并在其中展示了跨学科的方法以考虑工作场所中的机器人技术。跨学科性强调学术研究与实用专长及具身经验知识的整合，重视诸如工人福祉和工作吸引力等价值。随后，我们描述了一个正在开展的多管齐下的努力，旨在探索协作机器人在飞机发动机维修与维护操作中的潜力。 

---
# Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model 

**Title (ZH)**: 空间约束：视觉-语言-动作模型中的隐式空间表示对齐 

**Authors**: Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.12276)  

**Abstract**: Vision-language-action (VLA) models have recently shown strong potential in enabling robots to follow language instructions and execute precise actions. However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world. Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets. Alternative methods that estimate 3D cues from 2D images also suffer from the limited performance of depth this http URL propose Spatial Forcing (SF), a simple yet effective alignment strategy that implicitly forces VLA models to develop spatial comprehension capabilities without relying on explicit 3D inputs or depth estimators. SF aligns intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. By enforcing alignment at intermediate layers, SF guides VLAs to encode richer spatial representations that enhance action this http URL experiments in simulation and real-world environments demonstrate that SF achieves state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further accelerates training by up to 3.8x and improves data efficiency across diverse robotic tasks. Project page is at this https URL 

**Abstract (ZH)**: vision-language-action (VLA) 模型 recently shown strong potential in enabling robots to follow language instructions and execute precise actions. However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world. Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets. Alternative methods that estimate 3D cues from 2D images also suffer from the limited performance of depth estimates. We propose Spatial Forcing (SF), a simple yet effective alignment strategy that implicitly forces VLA models to develop spatial comprehension capabilities without relying on explicit 3D inputs or depth estimators. SF aligns intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. By enforcing alignment at intermediate layers, SF guides VLAs to encode richer spatial representations that enhance action execution. Experiments in simulation and real-world environments demonstrate that SF achieves state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further accelerates training by up to 3.8x and improves data efficiency across diverse robotic tasks. Project page: this https URL。 

---
# Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications 

**Title (ZH)**: 从正负示例和基于规则的规范中学习社交导航 

**Authors**: Chanwoo Kim, Jihwan Yoon, Hyeonseong Kim, Taemoon Jeong, Changwoo Yoo, Seungbeen Lee, Soohwan Byeon, Hoon Chung, Matthew Pan, Jean Oh, Kyungjae Lee, Sungjoon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12215)  

**Abstract**: Mobile robot navigation in dynamic human environments requires policies that balance adaptability to diverse behaviors with compliance to safety constraints. We hypothesize that integrating data-driven rewards with rule-based objectives enables navigation policies to achieve a more effective balance of adaptability and safety. To this end, we develop a framework that learns a density-based reward from positive and negative demonstrations and augments it with rule-based objectives for obstacle avoidance and goal reaching. A sampling-based lookahead controller produces supervisory actions that are both safe and adaptive, which are subsequently distilled into a compact student policy suitable for real-time operation with uncertainty estimates. Experiments in synthetic and elevator co-boarding simulations show consistent gains in success rate and time efficiency over baselines, and real-world demonstrations with human participants confirm the practicality of deployment. A video illustrating this work can be found on our project page this https URL. 

**Abstract (ZH)**: 移动机器人在动态人类环境中的导航需要兼顾多样行为适应性和安全约束的策略。我们假设将数据驱动的奖励与基于规则的目标相结合，可以使导航策略实现更有效的适应性与安全性平衡。为此，我们开发了一个框架，从正负示例中学习基于密度的奖励，并结合障碍物避免和目标获取的基于规则的目标进行增强。基于采样的前瞻控制器生成既安全又适应的监督动作，随后这些动作被提炼成一个适用于具有不确定性估计的实时操作的小型学生策略。在合成环境和电梯共乘模拟中的实验表明，该方法在成功率和时间效率方面优于基线方法，并且在人类参与者的真实世界演示中证实了其实用性。有关此工作的视频可在我们的项目页面上找到：this https URL。 

---
# CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving 

**Title (ZH)**: CoIRL-AD：协作竞争模仿-强化学习在潜在世界模型中的自动驾驶 

**Authors**: Xiaoji Zheng, Ziyuan Yang, Yanhao Chen, Yuhang Peng, Yuanrong Tang, Gengyuan Liu, Bokui Chen, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.12560)  

**Abstract**: End-to-end autonomous driving models trained solely with imitation learning (IL) often suffer from poor generalization. In contrast, reinforcement learning (RL) promotes exploration through reward maximization but faces challenges such as sample inefficiency and unstable convergence. A natural solution is to combine IL and RL. Moving beyond the conventional two-stage paradigm (IL pretraining followed by RL fine-tuning), we propose CoIRL-AD, a competitive dual-policy framework that enables IL and RL agents to interact during training. CoIRL-AD introduces a competition-based mechanism that facilitates knowledge exchange while preventing gradient conflicts. Experiments on the nuScenes dataset show an 18% reduction in collision rate compared to baselines, along with stronger generalization and improved performance on long-tail scenarios. Code is available at: this https URL. 

**Abstract (ZH)**: 仅使用模仿学习（IL）训练的端到端自主驾驶模型往往存在泛化能力差的问题。相比之下，强化学习（RL）通过奖励最大化促进探索，但面临样本效率低和不稳定的收敛问题。一种自然的解决方案是结合IL和RL。超越传统的两阶段 paradigm（IL 预训练后跟 RL 微调），我们提出 CoIRL-AD，这是一种竞争性的双策略框架，在训练过程中使 IL 和 RL 剂剂相互作用。CoIRL-AD 引入了一种基于竞争的机制，促进知识交换并防止梯度冲突。在 nuScenes 数据集上的实验显示，与基线方法相比，碰撞率降低了 18%，同时具有更强的泛化能力和在长尾场景上更好的性能。代码可在以下链接获取：this https URL。 

---
# EmboMatrix: A Scalable Training-Ground for Embodied Decision-Making 

**Title (ZH)**: EmboMatrix: 一种可扩展的体态决策训练平台 

**Authors**: Zixing Lei, Sheng Yin, Yichen Xiong, Yuanzhuo Ding, Wenhao Huang, Yuxi Wei, Qingyao Xu, Yiming Li, Weixin Li, Yunhong Wang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12072)  

**Abstract**: Embodied decision-making enables agents to translate high-level goals into executable actions through continuous interactions within the physical world, forming a cornerstone of general-purpose embodied intelligence. Large language models (LLMs), with their general decision-making capabilities, offer a promising path to realize this potential; however, LLMs trained solely on language lack exposure to physical environments, limiting their true embodied understanding. To bridge this gap, we propose the concept of a training ground: a comprehensive infrastructure that provides task and scene simulation, embodied interaction, and feedback signals, offering a one-stop solution for LLM acquire genuine embodied decision-making skills. In this work, we present EmboMatrix, the first training ground of its kind, providing massive and diverse tasks with efficient simulation and precise rewards. EmboMatrix incorporates a series of novel techniques: a multi-agent data engine for large-scale task and scene generation, a distributed heterogeneous-hardware system for scalable simulation, and a multi-level reward architecture for precise supervision. Leveraging EmboMatrix, we cultivate EmboBrain, an LLM whose embodied decision-making abilities emerge from extensive embodied interactions. Experiments show that EmboBrain-7B surpasses the 671B DeepSeek-R1 baseline by 9.5\% on two challenging embodied decision-making benchmarks, demonstrating the power of interactive, environment-grounded learning for building truly intelligent embodied agents. 

**Abstract (ZH)**: 具身决策使智能体能够通过与物理世界的持续交互将高层次目标转化为可执行动作，构成通用具身智能的核心。大规模语言模型（LLMs）凭借其广泛的决策能力为实现这一潜力提供了前景；然而，仅基于语言训练的LLMs缺乏对物理环境的暴露，限制了它们的真正具身理解。为了弥合这一差距，我们提出了训练场的概念：一个全面的基础设施，提供任务和场景模拟、具身交互和反馈信号，为LLM提供一站式解决方案以获得真实的具身决策能力。在本工作中，我们介绍了EmboMatrix，这是首个此类训练场，提供了大规模多样任务的高效模拟和精确奖励。EmboMatrix 集成了多项新技术：大规模任务和场景生成的多智能体数据引擎、可扩展模拟的分布式异构硬件系统以及多层次奖励架构以实现精确监督。借助EmboMatrix，我们培育了EmboBrain，一种具有广泛具身决策能力的LLM，这些能力源自大量具身交互。实验表明，EmboBrain-7B在两个具身决策基准测试上分别超越了671B DeepSeek-R1基线9.5%，证明了交互式、环境导向学习对于构建真正智能的具身代理的重要性。 

---
# ERA: Transforming VLMs into Embodied Agents via Embodied Prior Learning and Online Reinforcement Learning 

**Title (ZH)**: ERA: 将VLMs转化为具身代理的具身先验学习与在线强化学习方法 

**Authors**: Hanyang Chen, Mark Zhao, Rui Yang, Qinwei Ma, Ke Yang, Jiarui Yao, Kangrui Wang, Hao Bai, Zhenhailong Wang, Rui Pan, Mengchao Zhang, Jose Barreiros, Aykut Onol, ChengXiang Zhai, Heng Ji, Manling Li, Huan Zhang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12693)  

**Abstract**: Recent advances in embodied AI highlight the potential of vision language models (VLMs) as agents capable of perception, reasoning, and interaction in complex environments. However, top-performing systems rely on large-scale models that are costly to deploy, while smaller VLMs lack the necessary knowledge and skills to succeed. To bridge this gap, we present \textit{Embodied Reasoning Agent (ERA)}, a two-stage framework that integrates prior knowledge learning and online reinforcement learning (RL). The first stage, \textit{Embodied Prior Learning}, distills foundational knowledge from three types of data: (1) Trajectory-Augmented Priors, which enrich existing trajectory data with structured reasoning generated by stronger models; (2) Environment-Anchored Priors, which provide in-environment knowledge and grounding supervision; and (3) External Knowledge Priors, which transfer general knowledge from out-of-environment datasets. In the second stage, we develop an online RL pipeline that builds on these priors to further enhance agent performance. To overcome the inherent challenges in agent RL, including long horizons, sparse rewards, and training instability, we introduce three key designs: self-summarization for context management, dense reward shaping, and turn-level policy optimization. Extensive experiments on both high-level planning (EB-ALFRED) and low-level control (EB-Manipulation) tasks demonstrate that ERA-3B surpasses both prompting-based large models and previous training-based baselines. Specifically, it achieves overall improvements of 8.4\% on EB-ALFRED and 19.4\% on EB-Manipulation over GPT-4o, and exhibits strong generalization to unseen tasks. Overall, ERA offers a practical path toward scalable embodied intelligence, providing methodological insights for future embodied AI systems. 

**Abstract (ZH)**: Recent Advances in Embodied AI Highlight the Potential of Vision-Language Models as Agents Capable of Perception, Reasoning, and Interaction in Complex Environments: Bridging the Gap with the Embodied Reasoning Agent (ERA) Framework 

---
# Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks 

**Title (ZH)**: 记忆即行动：自主上下文策展用于长期自主任务 

**Authors**: Yuxiang Zhang, Jiangming Shu, Ye Ma, Xueyuan Lin, Shangxi Wu, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12635)  

**Abstract**: Large Language Models face challenges in long-horizon agentic tasks as their constrained memory is easily overwhelmed by distracting or irrelevant context. Existing working memory methods typically rely on external, heuristic mechanisms that are decoupled from the agent's core policy. In this work, we reframe working memory management as a learnable, intrinsic capability. We propose a novel framework, Memory-as-Action, where an agent actively manages its working memory by executing explicit editing operations as part of a unified policy. This formulation allows an agent, trained via reinforcement learning, to balance memory curation against long-term task objectives under given resource constraints. However, such memory editing actions break the standard assumption of a continuously growing prefix in LLM interactions, leading to what we call trajectory fractures. These non-prefix changes disrupt the causal continuity required by standard policy gradient methods, making those methods inapplicable. To address this, we propose a new algorithm, Dynamic Context Policy Optimization, which enables stable end-to-end reinforcement learning by segmenting trajectories at memory action points and applying trajectory-level advantages to the resulting action segments. Our results demonstrate that jointly optimizing for task reasoning and memory management in an end-to-end fashion not only reduces overall computational consumption but also improves task performance, driven by adaptive context curation strategies tailored to the model's intrinsic capabilities. 

**Abstract (ZH)**: 大型语言模型在长期任务中的记忆管理面临挑战，因其受约束的记忆容易被无关或分散的上下文所淹没。现有的工作记忆方法通常依赖于与代理核心策略脱钩的外部启发式机制。在本工作中，我们将工作记忆管理重新定义为可学习的内在能力。我们提出了一种新的框架——Memory-as-Action，其中代理通过执行明确的编辑操作来作为统一政策的一部分主动管理其工作记忆。这种表述使代理能够通过强化学习训练，在给定资源约束下平衡记忆整理与长期任务目标之间的关系。然而，这类记忆编辑操作破坏了标准的前缀连续假设，导致我们称之为轨迹断裂的现象。这些非前缀改变打断了标准策略梯度方法所需的因果连续性，使这些方法不再适用。为此，我们提出了一种新的算法——动态上下文策略优化，该算法通过在记忆操作点分割轨迹并在最终的行动片段上应用路径级别优势来实现端到端的强化学习的稳定性。我们的结果表明，以端到端的方式联合优化任务推理和记忆管理不仅减少了整体计算消耗，还通过根据模型的内在能力定制的自适应上下文整理策略提高了任务性能。 

---
# Artificial Intelligence Virtual Cells: From Measurements to Decisions across Modality, Scale, Dynamics, and Evaluation 

**Title (ZH)**: 人工智能虚拟细胞：从度量到跨模态、尺度、动力学和评估的决策 

**Authors**: Chengpeng Hu, Calvin Yu-Chian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12498)  

**Abstract**: Artificial Intelligence Virtual Cells (AIVCs) aim to learn executable, decision-relevant models of cell state from multimodal, multiscale measurements. Recent studies have introduced single-cell and spatial foundation models, improved cross-modality alignment, scaled perturbation atlases, and explored pathway-level readouts. Nevertheless, although held-out validation is standard practice, evaluations remain predominantly within single datasets and settings; evidence indicates that transport across laboratories and platforms is often limited, that some data splits are vulnerable to leakage and coverage bias, and that dose, time and combination effects are not yet systematically handled. Cross-scale coupling also remains constrained, as anchors linking molecular, cellular and tissue levels are sparse, and alignment to scientific or clinical readouts varies across studies. We propose a model-agnostic Cell-State Latent (CSL) perspective that organizes learning via an operator grammar: measurement, lift/project for cross-scale coupling, and intervention for dosing and scheduling. This view motivates a decision-aligned evaluation blueprint across modality, scale, context and intervention, and emphasizes function-space readouts such as pathway activity, spatial neighborhoods and clinically relevant endpoints. We recommend operator-aware data design, leakage-resistant partitions, and transparent calibration and reporting to enable reproducible, like-for-like comparisons. 

**Abstract (ZH)**: 人工 Intelligence 虚拟细胞 (AIVCs) 旨在从多模态、多层次测量中学习可执行的、与决策相关的细胞状态模型。最近的研究引入了单细胞和空间基础模型，改进了跨模态对齐，扩展了扰动图谱，并探索了通路级读数。然而，尽管留存验证是标准做法，评估依然主要局限于单个数据集和设置；证据表明，跨实验室和平台的传输往往是有限的，某些数据拆分容易出现泄漏和覆盖偏差，且剂量、时间和组合效应尚未系统处理。跨层次耦合也仍然是受限的，因为分子、细胞和组织级别之间的链接锚点稀少，且与科学或临床读数的对齐在不同研究中变化不一。我们提出了一种模型无关的细胞状态隐空间（CSL）视角，通过操作符文法组织学习：测量、跨层次耦合的提升/投影，以及剂量和调度的操作。这种视角激励了一种跨模态、跨层次、跨上下文和跨干预的决策对齐评估蓝图，并强调功能空间读数，如通路活动、空间邻域和临床相关终点。我们建议具有操作符意识的数据设计、防泄漏分割、透明的校准和报告以实现可重复的、同类比较。 

---
# Biased-Attention Guided Risk Prediction for Safe Decision-Making at Unsignalized Intersections 

**Title (ZH)**: 基于偏向注意力的风险预测方法以实现无信号交叉口的安全决策 

**Authors**: Chengyang Dong, Nan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12428)  

**Abstract**: Autonomous driving decision-making at unsignalized intersections is highly challenging due to complex dynamic interactions and high conflict risks. To achieve proactive safety control, this paper proposes a deep reinforcement learning (DRL) decision-making framework integrated with a biased attention mechanism. The framework is built upon the Soft Actor-Critic (SAC) algorithm. Its core innovation lies in the use of biased attention to construct a traffic risk predictor. This predictor assesses the long-term risk of collision for a vehicle entering the intersection and transforms this risk into a dense reward signal to guide the SAC agent in making safe and efficient driving decisions. Finally, the simulation results demonstrate that the proposed method effectively improves both traffic efficiency and vehicle safety at the intersection, thereby proving the effectiveness of the intelligent decision-making framework in complex scenarios. The code of our work is available at this https URL. 

**Abstract (ZH)**: 无信号交叉口的自动驾驶决策极具挑战性，由于复杂的动态交互和高冲突风险。为了实现主动安全控制，本文提出了一种结合偏置注意力机制的深度强化学习（DRL）决策框架，该框架基于Soft Actor-Critic（SAC）算法。其核心创新在于使用偏置注意力构建交通风险预测器，该预测器评估进入交叉口的车辆的长期碰撞风险，并将此风险转化为密集的奖励信号，以指导SAC代理做出安全高效的驾驶决策。最终，仿真结果表明，所提出的方法有效提高了交叉口的交通效率和车辆安全，从而证明了在复杂场景中智能决策框架的有效性。我们的代码可通过以下链接获取：this https URL。 

---
# ToPolyAgent: AI Agents for Coarse-Grained Topological Polymer Simulations 

**Title (ZH)**: ToPolyAgent: AI代理进行粗粒化拓扑聚合物模拟 

**Authors**: Lijie Ding, Jan-Michael Carrillo, Changwoo Do  

**Link**: [PDF](https://arxiv.org/pdf/2510.12091)  

**Abstract**: We introduce ToPolyAgent, a multi-agent AI framework for performing coarse-grained molecular dynamics (MD) simulations of topological polymers through natural language instructions. By integrating large language models (LLMs) with domain-specific computational tools, ToPolyAgent supports both interactive and autonomous simulation workflows across diverse polymer architectures, including linear, ring, brush, and star polymers, as well as dendrimers. The system consists of four LLM-powered agents: a Config Agent for generating initial polymer-solvent configurations, a Simulation Agent for executing LAMMPS-based MD simulations and conformational analyses, a Report Agent for compiling markdown reports, and a Workflow Agent for streamlined autonomous operations. Interactive mode incorporates user feedback loops for iterative refinements, while autonomous mode enables end-to-end task execution from detailed prompts. We demonstrate ToPolyAgent's versatility through case studies involving diverse polymer architectures under varying solvent condition, thermostats, and simulation lengths. Furthermore, we highlight its potential as a research assistant by directing it to investigate the effect of interaction parameters on the linear polymer conformation, and the influence of grafting density on the persistence length of the brush polymer. By coupling natural language interfaces with rigorous simulation tools, ToPolyAgent lowers barriers to complex computational workflows and advances AI-driven materials discovery in polymer science. It lays the foundation for autonomous and extensible multi-agent scientific research ecosystems. 

**Abstract (ZH)**: ToPolyAgent：一种通过自然语言指令进行拓扑聚合物粗粒度分子动力学模拟的多智能体AI框架 

---
# One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration 

**Title (ZH)**: 一生学习：从未经指导的探索中推断 stochastic 环境的符号世界模型 

**Authors**: Zaid Khan, Archiki Prasad, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2510.12088)  

**Abstract**: Symbolic world modeling requires inferring and representing an environment's transitional dynamics as an executable program. Prior work has focused on largely deterministic environments with abundant interaction data, simple mechanics, and human guidance. We address a more realistic and challenging setting, learning in a complex, stochastic environment where the agent has only "one life" to explore a hostile environment without human guidance. We introduce OneLife, a framework that models world dynamics through conditionally-activated programmatic laws within a probabilistic programming framework. Each law operates through a precondition-effect structure, activating in relevant world states. This creates a dynamic computation graph that routes inference and optimization only through relevant laws, avoiding scaling challenges when all laws contribute to predictions about a complex, hierarchical state, and enabling the learning of stochastic dynamics even with sparse rule activation. To evaluate our approach under these demanding constraints, we introduce a new evaluation protocol that measures (a) state ranking, the ability to distinguish plausible future states from implausible ones, and (b) state fidelity, the ability to generate future states that closely resemble reality. We develop and evaluate our framework on Crafter-OO, our reimplementation of the Crafter environment that exposes a structured, object-oriented symbolic state and a pure transition function that operates on that state alone. OneLife can successfully learn key environment dynamics from minimal, unguided interaction, outperforming a strong baseline on 16 out of 23 scenarios tested. We also test OneLife's planning ability, with simulated rollouts successfully identifying superior strategies. Our work establishes a foundation for autonomously constructing programmatic world models of unknown, complex environments. 

**Abstract (ZH)**: Symbolic 世界建模要求推断和表示环境的转换动力学为可执行程序。先前的工作主要集中在确定性较强的环境，这些环境有大量的交互数据、简单的物理机制，并且有人类的指导。我们解决了一个更具现实意义且更具挑战性的场景，在没有人类指导的情况下，仅凭“一次生命”探索一个敌对环境中的复杂、随机环境。我们提出了 OneLife 框架，通过概率编程框架中的条件激活程序法侓来建模世界动力学。每条法侓通过前提-效果结构运作，在相关世界状态中激活，从而创建一个动态计算图，仅通过相关法侓进行推理和优化，避免了所有法侓共同预测复杂层次状态时的扩展挑战，即使法侓激活稀疏，也能学习随机动力学。为了在这些苛刻的条件下评估我们的方法，我们提出了一种新的评估协议，衡量其在(a)状态排名方面的能力，即区分可能的未来状态与不可能的状态的能力，以及(b)状态保真度方面的能力，即生成与现实高度相似的未来状态的能力。我们在 Crafter-OO 上开发并评估了我们的框架，这是我们对 Crafter 环境的重新实现，该环境暴露了一个结构化的面向对象符号状态以及一个仅作用于此状态的纯粹转换函数。即使在最少的无指导交互下，OneLife 也能成功学习环境的关键动力学，且在测试的 23 种场景中有 16 种表现优于一个强有力的基线。我们还测试了 OneLife 的规划能力，模拟滚出成功识别出更优策略。我们的工作为自主构建未知复杂环境的程序化世界模型奠定了基础。 

---
# AI Agents as Universal Task Solvers 

**Title (ZH)**: AI代理作为通用任务求解器 

**Authors**: Alessandro Achille, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2510.12066)  

**Abstract**: AI reasoning agents are already able to solve a variety of tasks by deploying tools, simulating outcomes of multiple hypotheses and reflecting on them. In doing so, they perform computation, although not in the classical sense -- there is no program being executed. Still, if they perform computation, can AI agents be universal? Can chain-of-thought reasoning solve any computable task? How does an AI Agent learn to reason? Is it a matter of model size? Or training dataset size?
In this work, we reinterpret the role of learning in the context of AI Agents, viewing them as compute-capable stochastic dynamical systems, and highlight the role of time in a foundational principle for learning to reason. In doing so, we propose a shift from classical inductive learning to transductive learning -- where the objective is not to approximate the distribution of past data, but to capture their algorithmic structure to reduce the time needed to find solutions to new tasks.
Transductive learning suggests that, counter to Shannon's theory, a key role of information in learning is about reduction of time rather than reconstruction error. In particular, we show that the optimal speed-up that a universal solver can achieve using past data is tightly related to their algorithmic information. Using this, we show a theoretical derivation for the observed power-law scaling of inference time versus training time. We then show that scaling model size can lead to behaviors that, while improving accuracy on benchmarks, fail any reasonable test of intelligence, let alone super-intelligence: In the limit of infinite space and time, large models can behave as savants, able to brute-force through any task without any insight. Instead, we argue that the key quantity to optimize when scaling reasoning models is time, whose critical role in learning has so far only been indirectly considered. 

**Abstract (ZH)**: 基于AI推理代理的计算能力再诠释：从归纳学习到传递学习 

---
# DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving 

**Title (ZH)**: DriveVLA-W0：世界模型放大自主驾驶数据的扩展律 

**Authors**: Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12796)  

**Abstract**: Scaling Vision-Language-Action (VLA) models on large-scale data offers a promising path to achieving a more generalized driving intelligence. However, VLA models are limited by a ``supervision deficit'': the vast model capacity is supervised by sparse, low-dimensional actions, leaving much of their representational power underutilized. To remedy this, we propose \textbf{DriveVLA-W0}, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment. We showcase the paradigm's versatility by instantiating it for two dominant VLA archetypes: an autoregressive world model for VLAs that use discrete visual tokens, and a diffusion world model for those operating on continuous visual features. Building on the rich representations learned from world modeling, we introduce a lightweight action expert to address the inference latency for real-time deployment. Extensive experiments on the NAVSIM v1/v2 benchmark and a 680x larger in-house dataset demonstrate that DriveVLA-W0 significantly outperforms BEV and VLA baselines. Crucially, it amplifies the data scaling law, showing that performance gains accelerate as the training dataset size increases. 

**Abstract (ZH)**: 基于大规模数据的Vision-Language-Action (VLA) 模型缩放为实现更通用的驾驶智能提供了有希望的途径。然而，VLA 模型受到“监督不足”的限制：巨大的模型容量仅通过稀疏的低维度动作进行监督，使得其表示能力大量未被利用。为了解决这一问题，我们提出了DriveVLA-W0训练范式，采用世界建模来预测未来图像。该任务生成了一个稠密的、自监督的信号，迫使模型学习驾驶环境的基本动力学。我们通过实例化DriveVLA-W0来展示其灵活性，将其应用于两种主导的VLA架构：用于使用离散视觉标记的VLA的自回归世界模型，以及用于处理连续视觉特征的扩散世界模型。基于从世界建模中学习到的丰富表示，我们引入了一个轻量级的动作专家，以解决实时时部署中的推理延迟问题。在NAVSIM v1/v2基准测试和一个规模大680倍的内部数据集上进行的大量实验表明，DriveVLA-W0显著优于BEV和VLA基线。更重要的是，它放大了数据缩放定律，表明随着训练数据集规模的增加，性能改善加速。 

---
# HYPE: Hybrid Planning with Ego Proposal-Conditioned Predictions 

**Title (ZH)**: HYPE: 混合规划与 ego 提案条件下的预测 

**Authors**: Hang Yu, Julian Jordan, Julian Schmidt, Silvan Lindner, Alessandro Canevaro, Wilhelm Stork  

**Link**: [PDF](https://arxiv.org/pdf/2510.12733)  

**Abstract**: Safe and interpretable motion planning in complex urban environments needs to reason about bidirectional multi-agent interactions. This reasoning requires to estimate the costs of potential ego driving maneuvers. Many existing planners generate initial trajectories with sampling-based methods and refine them by optimizing on learned predictions of future environment states, which requires a cost function that encodes the desired vehicle behavior. Designing such a cost function can be very challenging, especially if a wide range of complex urban scenarios has to be considered. We propose HYPE: HYbrid Planning with Ego proposal-conditioned predictions, a planner that integrates multimodal trajectory proposals from a learned proposal model as heuristic priors into a Monte Carlo Tree Search (MCTS) refinement. To model bidirectional interactions, we introduce an ego-conditioned occupancy prediction model, enabling consistent, scene-aware reasoning. Our design significantly simplifies cost function design in refinement by considering proposal-driven guidance, requiring only minimalistic grid-based cost terms. Evaluations on large-scale real-world benchmarks nuPlan and DeepUrban show that HYPE effectively achieves state-of-the-art performance, especially in safety and adaptability. 

**Abstract (ZH)**: 混合规划与 ego 提议条件下的预测：在复杂城市环境中的安全可解释运动规划 

---
# Deep SPI: Safe Policy Improvement via World Models 

**Title (ZH)**: Deep SPI: 安全策略改进借助世界模型 

**Authors**: Florent Delgrange, Raphael Avalos, Willem Röpke  

**Link**: [PDF](https://arxiv.org/pdf/2510.12312)  

**Abstract**: Safe policy improvement (SPI) offers theoretical control over policy updates, yet existing guarantees largely concern offline, tabular reinforcement learning (RL). We study SPI in general online settings, when combined with world model and representation learning. We develop a theoretical framework showing that restricting policy updates to a well-defined neighborhood of the current policy ensures monotonic improvement and convergence. This analysis links transition and reward prediction losses to representation quality, yielding online, "deep" analogues of classical SPI theorems from the offline RL literature. Building on these results, we introduce DeepSPI, a principled on-policy algorithm that couples local transition and reward losses with regularised policy updates. On the ALE-57 benchmark, DeepSPI matches or exceeds strong baselines, including PPO and DeepMDPs, while retaining theoretical guarantees. 

**Abstract (ZH)**: Safe政策改进（SPI）提供了对政策更新的理论控制，但现有的保证主要集中在离线的表格强化学习（RL）中。我们研究SPI在结合世界模型和表示学习的一般在线设置中的应用。我们发展了一个理论框架，表明将政策更新限制在当前政策的良好定义的邻域内可以确保单调改进和收敛。该分析将转变预测损失和奖励预测损失与表示质量联系起来，从而得到经典的离线RL文献中的SPI定理的在线“深度”类比。基于这些结果，我们提出了DeepSPI，这是一种原理上的在线策略算法，它将局部转变和奖励损失与正则化政策更新联系起来。在ALE-57基准测试中，DeepSPI能够达到或超过包括PPO和DeepMDPs在内的强 baseline，同时保持理论上的保证。 

---
# Diffusion Models for Reinforcement Learning: Foundations, Taxonomy, and Development 

**Title (ZH)**: 扩散模型在强化学习中的应用：基础、分类与发展 

**Authors**: Changfu Xu, Jianxiong Guo, Yuzhu Liang, Haiyang Huang, Haodong Zou, Xi Zheng, Shui Yu, Xiaowen Chu, Jiannong Cao, Tian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12253)  

**Abstract**: Diffusion Models (DMs), as a leading class of generative models, offer key advantages for reinforcement learning (RL), including multi-modal expressiveness, stable training, and trajectory-level planning. This survey delivers a comprehensive and up-to-date synthesis of diffusion-based RL. We first provide an overview of RL, highlighting its challenges, and then introduce the fundamental concepts of DMs, investigating how they are integrated into RL frameworks to address key challenges in this research field. We establish a dual-axis taxonomy that organizes the field along two orthogonal dimensions: a function-oriented taxonomy that clarifies the roles DMs play within the RL pipeline, and a technique-oriented taxonomy that situates implementations across online versus offline learning regimes. We also provide a comprehensive examination of this progression from single-agent to multi-agent domains, thereby forming several frameworks for DM-RL integration and highlighting their practical utility. Furthermore, we outline several categories of successful applications of diffusion-based RL across diverse domains, discuss open research issues of current methodologies, and highlight key directions for future research to advance the field. Finally, we summarize the survey to identify promising future development directions. We are actively maintaining a GitHub repository (this https URL) for papers and other related resources to apply DMs for RL. 

**Abstract (ZH)**: 基于扩散模型的强化学习综合调研：从单智能体到多智能体领域的进展与应用 

---
# An AI-Based Behavioral Health Safety Filter and Dataset for Identifying Mental Health Crises in Text-Based Conversations 

**Title (ZH)**: 基于AI的行为健康安全过滤器及数据集：识别文本对话中的心理健康危机 

**Authors**: Benjamin W. Nelson, Celeste Wong, Matthew T. Silvestrini, Sooyoon Shin, Alanna Robinson, Jessica Lee, Eric Yang, John Torous, Andrew Trister  

**Link**: [PDF](https://arxiv.org/pdf/2510.12083)  

**Abstract**: Large language models often mishandle psychiatric emergencies, offering harmful or inappropriate advice and enabling destructive behaviors. This study evaluated the Verily behavioral health safety filter (VBHSF) on two datasets: the Verily Mental Health Crisis Dataset containing 1,800 simulated messages and the NVIDIA Aegis AI Content Safety Dataset subsetted to 794 mental health-related messages. The two datasets were clinician-labelled and we evaluated performance using the clinician labels. Additionally, we carried out comparative performance analyses against two open source, content moderation guardrails: OpenAI Omni Moderation Latest and NVIDIA NeMo Guardrails. The VBHSF demonstrated, well-balanced performance on the Verily Mental Health Crisis Dataset v1.0, achieving high sensitivity (0.990) and specificity (0.992) in detecting any mental health crises. It achieved an F1-score of 0.939, sensitivity ranged from 0.917-0.992, and specificity was >= 0.978 in identifying specific crisis categories. When evaluated against the NVIDIA Aegis AI Content Safety Dataset 2.0, VBHSF performance remained highly sensitive (0.982) and accuracy (0.921) with reduced specificity (0.859). When compared with the NVIDIA NeMo and OpenAI Omni Moderation Latest guardrails, the VBHSF demonstrated superior performance metrics across both datasets, achieving significantly higher sensitivity in all cases (all p < 0.001) and higher specificity relative to NVIDIA NeMo (p < 0.001), but not to OpenAI Omni Moderation Latest (p = 0.094). NVIDIA NeMo and OpenAI Omni Moderation Latest exhibited inconsistent performance across specific crisis types, with sensitivity for some categories falling below 0.10. Overall, the VBHSF demonstrated robust, generalizable performance that prioritizes sensitivity to minimize missed crises, a crucial feature for healthcare applications. 

**Abstract (ZH)**: 大型语言模型在处理心理危机方面常常出现错误，提供有害或不合适的建议，从而导致破坏性行为。本研究评估了Verily行为健康安全性过滤器（VBHSF）在两个数据集中表现：包含1800条模拟消息的Verily Mental Health Crisis Dataset和NVIDIA Aegis AI内容安全性数据集中的794条心理健康相关消息。两个数据集均由临床医生标注，我们使用临床医生标签评估性能。此外，我们还对VBHSF与两个开源内容审核护栏——OpenAI Omni Moderation Latest和NVIDIA NeMo Guardrails——进行了性能对比分析。VBHSF在Verily Mental Health Crisis Dataset v1.0中表现出良好的平衡性能，检测任何心理健康危机的敏感性为0.990、特异性为0.992。在识别特定危机类别时，VBHSF的F1分数为0.939，敏感性范围为0.917-0.992，特异性不低于0.978。当与NVIDIA Aegis AI Content Safety Dataset 2.0进行评估时，VBHSF的敏感性仍为0.982，准确率为0.921，但特异性降低至0.859。与NVIDIA NeMo和OpenAI Omni Moderation Latest护栏相比，VBHSF在两个数据集中均表现出更优的性能指标，在所有情况下敏感性显著更高（所有p < 0.001），与NVIDIA NeMo相比特异性更高（p < 0.001），但与OpenAI Omni Moderation Latest相比则无显著差异（p = 0.094）。NVIDIA NeMo和OpenAI Omni Moderation Latest在特定危机类型上表现不一致，一些类别的敏感性低于0.10。总体而言，VBHSF表现出稳健的一般化性能，优先考虑敏感性以避免错过危机，这是其在医疗保健应用中的一项关键特征。 

---
