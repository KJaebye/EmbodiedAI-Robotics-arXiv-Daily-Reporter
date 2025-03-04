# Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding 

**Title (ZH)**: 离散时间混合自动机学习：腿足运动与滑板运动的结合 

**Authors**: Hang Liu, Sangli Teng, Ben Liu, Wei Zhang, Maani Ghaffari  

**Link**: [PDF](https://arxiv.org/pdf/2503.01842)  

**Abstract**: This paper introduces Discrete-time Hybrid Automata Learning (DHAL), a framework using on-policy Reinforcement Learning to identify and execute mode-switching without trajectory segmentation or event function learning. Hybrid dynamical systems, which include continuous flow and discrete mode switching, can model robotics tasks like legged robot locomotion. Model-based methods usually depend on predefined gaits, while model-free approaches lack explicit mode-switching knowledge. Current methods identify discrete modes via segmentation before regressing continuous flow, but learning high-dimensional complex rigid body dynamics without trajectory labels or segmentation is a challenging open problem. Our approach incorporates a beta policy distribution and a multi-critic architecture to model contact-guided motions, exemplified by a challenging quadrupedal robot skateboard task. We validate our method through simulations and real-world tests, demonstrating robust performance in hybrid dynamical systems. 

**Abstract (ZH)**: 基于策略的离散时间混合自动机学习（DHAL）：无需轨迹分割或事件函数学习的模式切换识别与执行 

---
# vS-Graphs: Integrating Visual SLAM and Situational Graphs through Multi-level Scene Understanding 

**Title (ZH)**: vS-图：通过多层场景理解结合视觉SLAM和情境图谱 

**Authors**: Ali Tourani, Saad Ejaz, Hriday Bavle, David Morilla-Cabello, Jose Luis Sanchez-Lopez, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2503.01783)  

**Abstract**: Current Visual Simultaneous Localization and Mapping (VSLAM) systems often struggle to create maps that are both semantically rich and easily interpretable. While incorporating semantic scene knowledge aids in building richer maps with contextual associations among mapped objects, representing them in structured formats like scene graphs has not been widely addressed, encountering complex map comprehension and limited scalability. This paper introduces visual S-Graphs (vS-Graphs), a novel real-time VSLAM framework that integrates vision-based scene understanding with map reconstruction and comprehensible graph-based representation. The framework infers structural elements (i.e., rooms and corridors) from detected building components (i.e., walls and ground surfaces) and incorporates them into optimizable 3D scene graphs. This solution enhances the reconstructed map's semantic richness, comprehensibility, and localization accuracy. Extensive experiments on standard benchmarks and real-world datasets demonstrate that vS-Graphs outperforms state-of-the-art VSLAM methods, reducing trajectory error by an average of 3.38% and up to 9.58% on real-world data. Furthermore, the proposed framework achieves environment-driven semantic entity detection accuracy comparable to precise LiDAR-based frameworks using only visual features. A web page containing more media and evaluation outcomes is available on this https URL. 

**Abstract (ZH)**: 当前的视觉同时定位与建图(VSLAM)系统往往难以创建既富有语义信息又易于解释的地图。虽然结合语义场景知识有助于构建具有上下文关联的更丰富的地图，但将这些信息表示为场景图等结构化格式尚未广泛解决，面临地图理解复杂和扩展性差的问题。本文介绍了一种名为视觉S-图(vS-Graphs)的新颖实时VSLAM框架，该框架将基于视觉的场景理解与地图重建和易于理解的图表示结合在一起。该框架从检测到的建筑组件（如墙壁和地面表面）推断出结构元素（如房间和走廊），并将其纳入可优化的3D场景图中。此解决方案增强了重建地图的语义丰富性、可解释性和定位准确性。在标准基准数据集和真实世界数据集上的大量实验表明，vS-Graphs在平均轨迹误差减少3.38%至9.58%方面优于最先进的VSLAM方法。此外，所提出的框架使用仅视觉特征即可实现与精确LiDAR基框架相当的环境驱动语义实体检测精度。更多信息和评估结果可在以下网址找到。 

---
# No Plan but Everything Under Control: Robustly Solving Sequential Tasks with Dynamically Composed Gradient Descent 

**Title (ZH)**: 有备无患，一切尽在掌控：使用动态组合梯度下降稳健解决序列任务 

**Authors**: Vito Mengers, Oliver Brock  

**Link**: [PDF](https://arxiv.org/pdf/2503.01732)  

**Abstract**: We introduce a novel gradient-based approach for solving sequential tasks by dynamically adjusting the underlying myopic potential field in response to feedback and the world's regularities. This adjustment implicitly considers subgoals encoded in these regularities, enabling the solution of long sequential tasks, as demonstrated by solving the traditional planning domain of Blocks World - without any planning. Unlike conventional planning methods, our feedback-driven approach adapts to uncertain and dynamic environments, as demonstrated by one hundred real-world trials involving drawer manipulation. These experiments highlight the robustness of our method compared to planning and show how interactive perception and error recovery naturally emerge from gradient descent without explicitly implementing them. This offers a computationally efficient alternative to planning for a variety of sequential tasks, while aligning with observations on biological problem-solving strategies. 

**Abstract (ZH)**: 我们提出了一种基于梯度的新颖方法，通过动态调整底层短视势场以响应反馈和世界的规律来解决序列任务。这种方法隐含地考虑了这些规律中编码的子目标，从而能够解决长期序列任务，如通过免规划解决传统的规划领域Block World。与传统规划方法不同，我们的基于反馈的方法能够适应不确定和动态环境，如在涉及抽屉操作的一百次真实世界实验中所证明的那样。这些实验突显了我们方法在与规划相比的鲁棒性，并展示了如何从梯度下降自然地涌现出交互式感知和错误恢复，而无需显式实现它们。这种方法为各种序列任务提供了一种计算效率更高的替代规划方案，同时与生物问题解决策略的观察相一致。 

---
# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation 

**Title (ZH)**: 代码作为符号规划师：基于符号代码生成的基础模型机器人规划 

**Authors**: Yongchao Chen, Yilun Hao, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01700)  

**Abstract**: Recent works have shown great potentials of Large Language Models (LLMs) in robot task and motion planning (TAMP). Current LLM approaches generate text- or code-based reasoning chains with sub-goals and action plans. However, they do not fully leverage LLMs' symbolic computing and code generation capabilities. Many robot TAMP tasks involve complex optimization under multiple constraints, where pure textual reasoning is insufficient. While augmenting LLMs with predefined solvers and planners improves performance, it lacks generalization across tasks. Given LLMs' growing coding proficiency, we enhance their TAMP capabilities by steering them to generate code as symbolic planners for optimization and constraint verification. Unlike prior work that uses code to interface with robot action modules, we steer LLMs to generate code as solvers, planners, and checkers for TAMP tasks requiring symbolic computing, while still leveraging textual reasoning to incorporate common sense. With a multi-round guidance and answer evolution framework, the proposed Code-as-Symbolic-Planner improves success rates by average 24.1\% over best baseline methods across seven typical TAMP tasks and three popular LLMs. Code-as-Symbolic-Planner shows strong effectiveness and generalizability across discrete and continuous environments, 2D/3D simulations and real-world settings, as well as single- and multi-robot tasks with diverse requirements. See our project website this https URL for prompts, videos, and code. 

**Abstract (ZH)**: 近期研究表明，大型语言模型（LLMs）在机器人任务和运动规划（TAMP）方面展现了巨大的潜力。当前的LLM方法生成基于文本或代码的推理链，包含子目标和行动计划。然而，它们未能充分利用LLMs的符号计算和代码生成能力。许多机器人的TAMP任务涉及在多个约束下的复杂优化，仅凭纯粹的文字推理是不够的。通过与预定义求解器和规划器结合，虽然可以提高性能，但缺乏跨任务的一般性。鉴于LLMs编码能力的不断提升，我们通过引导它们生成代码作为符号规划器来进行优化和约束验证，增强了其TAMP能力。不同于以往使用代码与机器人动作模块接口的方法，我们引导LLMs生成作为求解器、规划器和验证器的代码，专门应用于需要符号计算的TAMP任务，同时仍利用文字推理来融入常识。通过多轮引导和答案演进框架，提出的Code-as-Symbolic-Planner方法在七种典型TAMP任务和三种流行的LLM上，平均提高了24.1%的成功率。Code-as-Symbolic-Planner在离散和连续环境、2D/3D模拟和真实世界设置，以及单个和多个机器人任务中表现出强大的效果和一般性。更多信息请访问我们的项目网站：[此 https URL]，获取提示、视频和代码。 

---
# Perceptual Motor Learning with Active Inference Framework for Robust Lateral Control 

**Title (ZH)**: 基于主动推断框架的知觉运动学习在稳健横向控制中的应用 

**Authors**: Elahe Delavari, John Moore, Junho Hong, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2503.01676)  

**Abstract**: This paper presents a novel Perceptual Motor Learning (PML) framework integrated with Active Inference (AIF) to enhance lateral control in Highly Automated Vehicles (HAVs). PML, inspired by human motor learning, emphasizes the seamless integration of perception and action, enabling efficient decision-making in dynamic environments. Traditional autonomous driving approaches--including modular pipelines, imitation learning, and reinforcement learning--struggle with adaptability, generalization, and computational efficiency. In contrast, PML with AIF leverages a generative model to minimize prediction error ("surprise") and actively shape vehicle control based on learned perceptual-motor representations. Our approach unifies deep learning with active inference principles, allowing HAVs to perform lane-keeping maneuvers with minimal data and without extensive retraining across different environments. Extensive experiments in the CARLA simulator demonstrate that PML with AIF enhances adaptability without increasing computational overhead while achieving performance comparable to conventional methods. These findings highlight the potential of PML-driven active inference as a robust alternative for real-world autonomous driving applications. 

**Abstract (ZH)**: 本文提出了一种将知觉运动学习（PML）与主动推断（AIF）结合的新型框架，以增强高度自动化车辆（HAVs）的横向控制能力。 

---
# RoboDexVLM: Visual Language Model-Enabled Task Planning and Motion Control for Dexterous Robot Manipulation 

**Title (ZH)**: RoboDexVLM：视觉语言模型驱动的灵巧机器人操作的任务规划与运动控制 

**Authors**: Haichao Liu, Sikai Guo, Pengfei Mai, Jiahang Cao, Haoang Li, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.01616)  

**Abstract**: This paper introduces RoboDexVLM, an innovative framework for robot task planning and grasp detection tailored for a collaborative manipulator equipped with a dexterous hand. Previous methods focus on simplified and limited manipulation tasks, which often neglect the complexities associated with grasping a diverse array of objects in a long-horizon manner. In contrast, our proposed framework utilizes a dexterous hand capable of grasping objects of varying shapes and sizes while executing tasks based on natural language commands. The proposed approach has the following core components: First, a robust task planner with a task-level recovery mechanism that leverages vision-language models (VLMs) is designed, which enables the system to interpret and execute open-vocabulary commands for long sequence tasks. Second, a language-guided dexterous grasp perception algorithm is presented based on robot kinematics and formal methods, tailored for zero-shot dexterous manipulation with diverse objects and commands. Comprehensive experimental results validate the effectiveness, adaptability, and robustness of RoboDexVLM in handling long-horizon scenarios and performing dexterous grasping. These results highlight the framework's ability to operate in complex environments, showcasing its potential for open-vocabulary dexterous manipulation. Our open-source project page can be found at this https URL. 

**Abstract (ZH)**: RoboDexVLM：一种面向协作 manipulator 的灵巧手任务规划与抓取检测创新框架 

---
# MapExRL: Human-Inspired Indoor Exploration with Predicted Environment Context and Reinforcement Learning 

**Title (ZH)**: MapExRL: 基于预测环境上下文和强化学习的人类启发式室内探索 

**Authors**: Narek Harutyunyan, Brady Moon, Seungchan Kim, Cherie Ho, Adam Hung, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2503.01548)  

**Abstract**: Path planning for robotic exploration is challenging, requiring reasoning over unknown spaces and anticipating future observations. Efficient exploration requires selecting budget-constrained paths that maximize information gain. Despite advances in autonomous exploration, existing algorithms still fall short of human performance, particularly in structured environments where predictive cues exist but are underutilized. Guided by insights from our user study, we introduce MapExRL, which improves robot exploration efficiency in structured indoor environments by enabling longer-horizon planning through reinforcement learning (RL) and global map predictions. Unlike many RL-based exploration methods that use motion primitives as the action space, our approach leverages frontiers for more efficient model learning and longer horizon reasoning. Our framework generates global map predictions from the observed map, which our policy utilizes, along with the prediction uncertainty, estimated sensor coverage, frontier distance, and remaining distance budget, to assess the strategic long-term value of frontiers. By leveraging multiple frontier scoring methods and additional context, our policy makes more informed decisions at each stage of the exploration. We evaluate our framework on a real-world indoor map dataset, achieving up to an 18.8% improvement over the strongest state-of-the-art baseline, with even greater gains compared to conventional frontier-based algorithms. 

**Abstract (ZH)**: 基于强化学习的结构化室内环境中的地图预测辅助路径规划 

---
# Interactive Navigation for Legged Manipulators with Learned Arm-Pushing Controller 

**Title (ZH)**: 基于学习的臂推控制器的腿足 manipulator 交互导航 

**Authors**: Zhihai Bi, Kai Chen, Chunxin Zheng, Yulin Li, Haoang Li, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.01474)  

**Abstract**: Interactive navigation is crucial in scenarios where proactively interacting with objects can yield shorter paths, thus significantly improving traversal efficiency. Existing methods primarily focus on using the robot body to relocate large obstacles (which could be comparable to the size of a robot). However, they prove ineffective in narrow or constrained spaces where the robot's dimensions restrict its manipulation capabilities. This paper introduces a novel interactive navigation framework for legged manipulators, featuring an active arm-pushing mechanism that enables the robot to reposition movable obstacles in space-constrained environments. To this end, we develop a reinforcement learning-based arm-pushing controller with a two-stage reward strategy for large-object manipulation. Specifically, this strategy first directs the manipulator to a designated pushing zone to achieve a kinematically feasible contact configuration. Then, the end effector is guided to maintain its position at appropriate contact points for stable object displacement while preventing toppling. The simulations validate the robustness of the arm-pushing controller, showing that the two-stage reward strategy improves policy convergence and long-term performance. Real-world experiments further demonstrate the effectiveness of the proposed navigation framework, which achieves shorter paths and reduced traversal time. The open-source project can be found at this https URL. 

**Abstract (ZH)**: 基于腿部 manipulator 的主动臂推送交互导航框架 

---
# CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs 

**Title (ZH)**: 认知无人机：一种适用于 UAV 实时认知任务解决与推理的长距离模型及评估基准 

**Authors**: Artem Lykov, Valerii Serpiva, Muhammad Haris Khan, Oleg Sautenkov, Artyom Myshlyaev, Grik Tadevosyan, Yasheerah Yaqoot, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01378)  

**Abstract**: This paper introduces CognitiveDrone, a novel Vision-Language-Action (VLA) model tailored for complex Unmanned Aerial Vehicles (UAVs) tasks that demand advanced cognitive abilities. Trained on a dataset comprising over 8,000 simulated flight trajectories across three key categories-Human Recognition, Symbol Understanding, and Reasoning-the model generates real-time 4D action commands based on first-person visual inputs and textual instructions. To further enhance performance in intricate scenarios, we propose CognitiveDrone-R1, which integrates an additional Vision-Language Model (VLM) reasoning module to simplify task directives prior to high-frequency control. Experimental evaluations using our open-source benchmark, CognitiveDroneBench, reveal that while a racing-oriented model (RaceVLA) achieves an overall success rate of 31.3%, the base CognitiveDrone model reaches 59.6%, and CognitiveDrone-R1 attains a success rate of 77.2%. These results demonstrate improvements of up to 30% in critical cognitive tasks, underscoring the effectiveness of incorporating advanced reasoning capabilities into UAV control systems. Our contributions include the development of a state-of-the-art VLA model for UAV control and the introduction of the first dedicated benchmark for assessing cognitive tasks in drone operations. The complete repository is available at this http URL 

**Abstract (ZH)**: 认知无人机：一种针对复杂无人机任务的新型视觉-语言-行动模型 

---
# FABG : End-to-end Imitation Learning for Embodied Affective Human-Robot Interaction 

**Title (ZH)**: FABG：端到端模仿学习在具身情感人机交互中的应用 

**Authors**: Yanghai Zhang, Changyi Liu, Keting Fu, Wenbin Zhou, Qingdu Li, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01363)  

**Abstract**: This paper proposes FABG (Facial Affective Behavior Generation), an end-to-end imitation learning system for human-robot interaction, designed to generate natural and fluid facial affective behaviors. In interaction, effectively obtaining high-quality demonstrations remains a challenge. In this work, we develop an immersive virtual reality (VR) demonstration system that allows operators to perceive stereoscopic environments. This system ensures "the operator's visual perception matches the robot's sensory input" and "the operator's actions directly determine the robot's behaviors" - as if the operator replaces the robot in human interaction engagements. We propose a prediction-driven latency compensation strategy to reduce robotic reaction delays and enhance interaction fluency. FABG naturally acquires human interactive behaviors and subconscious motions driven by intuition, eliminating manual behavior scripting. We deploy FABG on a real-world 25-degree-of-freedom (DoF) humanoid robot, validating its effectiveness through four fundamental interaction tasks: expression response, dynamic gaze, foveated attention, and gesture recognition, supported by data collection and policy training. Project website: this https URL 

**Abstract (ZH)**: 本论文提出FABG（面部情感行为生成）系统，这是一个端到端的模仿学习系统，旨在用于人机交互，以生成自然流畅的面部情感行为。在交互过程中，有效获取高品质示范仍然是一项挑战。在此项工作中，我们开发了一种沉浸式虚拟现实（VR）示范系统，允许操作员感知立体环境。该系统确保“操作员的视觉感知与机器人的感输入相匹配”且“操作员的动作直接决定机器人的行为”——仿佛操作员替换了机器人在人机交互中的角色。我们提出了一种预测驱动的延迟补偿策略，以减少机器人反应延迟并提升交互流畅性。FABG自然地获取由直觉驱动的人际互动行为和潜意识动作，消除了手动行为脚本化的需求。我们在一个实际应用的25自由度（DoF）的人形机器人上部署FABG，并通过四种基本交互任务的数据收集和策略训练验证其有效性：表情响应、动态凝视、中心视觉关注和手势识别。项目网站：this https URL。 

---
# Few-shot Sim2Real Based on High Fidelity Rendering with Force Feedback Teleoperation 

**Title (ZH)**: 基于高保真渲染和力反馈遥控的少样本Sim2Real 

**Authors**: Yanwen Zou, Junda Huang, Boyuan Liang, Honghao Guo, Zhengyang Liu, Xin Ma, Jianshu Zhou, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2503.01301)  

**Abstract**: Teleoperation offers a promising approach to robotic data collection and human-robot interaction. However, existing teleoperation methods for data collection are still limited by efficiency constraints in time and space, and the pipeline for simulation-based data collection remains unclear. The problem is how to enhance task performance while minimizing reliance on real-world data. To address this challenge, we propose a teleoperation pipeline for collecting robotic manipulation data in simulation and training a few-shot sim-to-real visual-motor policy. Force feedback devices are integrated into the teleoperation system to provide precise end-effector gripping force feedback. Experiments across various manipulation tasks demonstrate that force feedback significantly improves both success rates and execution efficiency, particularly in simulation. Furthermore, experiments with different levels of visual rendering quality reveal that enhanced visual realism in simulation substantially boosts task performance while reducing the need for real-world data. 

**Abstract (ZH)**: 基于仿真的机器人操作数据采集及少样本视觉-运动策略训练的远程操控pipeline 

---
# Diffusion Stabilizer Policy for Automated Surgical Robot Manipulations 

**Title (ZH)**: 自动化手术机器人操作的扩散稳定器策略 

**Authors**: Chonlam Ho, Jianshu Hu, Hesheng Wang, Qi Dou, Yutong Ban  

**Link**: [PDF](https://arxiv.org/pdf/2503.01252)  

**Abstract**: Intelligent surgical robots have the potential to revolutionize clinical practice by enabling more precise and automated surgical procedures. However, the automation of such robot for surgical tasks remains under-explored compared to recent advancements in solving household manipulation tasks. These successes have been largely driven by (1) advanced models, such as transformers and diffusion models, and (2) large-scale data utilization. Aiming to extend these successes to the domain of surgical robotics, we propose a diffusion-based policy learning framework, called Diffusion Stabilizer Policy (DSP), which enables training with imperfect or even failed trajectories. Our approach consists of two stages: first, we train the diffusion stabilizer policy using only clean data. Then, the policy is continuously updated using a mixture of clean and perturbed data, with filtering based on the prediction error on actions. Comprehensive experiments conducted in various surgical environments demonstrate the superior performance of our method in perturbation-free settings and its robustness when handling perturbed demonstrations. 

**Abstract (ZH)**: 基于扩散的政策学习框架（Diffusion Stabilizer Policy）：促进手术机器人领域中的鲁棒政策训练 

---
# Catching Spinning Table Tennis Balls in Simulation with End-to-End Curriculum Reinforcement Learning 

**Title (ZH)**: 使用端到端 Curriculum 强化学习在仿真中捕捉旋转乒乓球 

**Authors**: Xiaoyi Hu, Yue Mao, Gang Wang, Qingdu Li, Jianwei Zhang, Yunfeng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.01251)  

**Abstract**: The game of table tennis is renowned for its extremely high spin rate, but most table tennis robots today struggle to handle balls with such rapid spin. To address this issue, we have contributed a series of methods, including: 1. Curriculum Reinforcement Learning (RL): This method helps the table tennis robot learn to play table tennis progressively from easy to difficult tasks. 2. Analysis of Spinning Table Tennis Ball Collisions: We have conducted a physics-based analysis to generate more realistic trajectories of spinning table tennis balls after collision. 3. Definition of Trajectory States: The definition of trajectory states aids in setting up the reward function. 4. Selection of Valid Rally Trajectories: We have introduced a valid rally trajectory selection scheme to ensure that the robot's training is not influenced by abnormal trajectories. 5. Reality-to-Simulation (Real2Sim) Transfer: This scheme is employed to validate the trained robot's ability to handle spinning balls in real-world scenarios. With Real2Sim, the deployment costs for robotic reinforcement learning can be further reduced. Moreover, the trajectory-state-based reward function is not limited to table tennis robots; it can be generalized to a wide range of cyclical tasks. To validate our robot's ability to handle spinning balls, the Real2Sim experiments were conducted. For the specific video link of the experiment, please refer to the supplementary materials. 

**Abstract (ZH)**: 乒乓球游戏以其极高的旋转率而闻名，但当今大多数乒乓球机器人难以处理带有快速旋转的球。为了解决这一问题，我们贡献了一系列方法，包括：1. 进阶强化学习（Curriculum Reinforcement Learning, RL）：该方法帮助乒乓球机器人从易到难逐步学习乒乓球技能。2. 旋转乒乓球碰撞分析：我们进行了基于物理学的分析，以生成更真实的碰撞后的旋转乒乓球轨迹。3. 轨迹状态定义：轨迹状态的定义有助于设置奖励函数。4. 有效回球轨迹选择：我们引入了有效回球轨迹选择方案，以确保机器人的训练不受异常轨迹的影响。5. 现实到模拟（Real2Sim）转移：该方案用于验证训练后的机器人在现实世界场景中处理旋转球的能力。使用Real2Sim可以进一步降低机器人强化学习的部署成本。此外，基于轨迹状态的奖励函数不仅适用于乒乓球机器人，还可推广到一系列循环任务。为验证我们机器人处理旋转球的能力，进行了Real2Sim实验。具体实验视频链接请参见补充材料。 

---
# A Taxonomy for Evaluating Generalist Robot Policies 

**Title (ZH)**: 通用机器人政策评估 taxonomy 

**Authors**: Jensen Gao, Suneel Belkhale, Sudeep Dasari, Ashwin Balakrishna, Dhruv Shah, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2503.01238)  

**Abstract**: Machine learning for robotics promises to unlock generalization to novel tasks and environments. Guided by this promise, many recent works have focused on scaling up robot data collection and developing larger, more expressive policies to achieve this. But how do we measure progress towards this goal of policy generalization in practice? Evaluating and quantifying generalization is the Wild West of modern robotics, with each work proposing and measuring different types of generalization in their own, often difficult to reproduce, settings. In this work, our goal is (1) to outline the forms of generalization we believe are important in robot manipulation in a comprehensive and fine-grained manner, and (2) to provide reproducible guidelines for measuring these notions of generalization. We first propose STAR-Gen, a taxonomy of generalization for robot manipulation structured around visual, semantic, and behavioral generalization. We discuss how our taxonomy encompasses most prior notions of generalization in robotics. Next, we instantiate STAR-Gen with a concrete real-world benchmark based on the widely-used Bridge V2 dataset. We evaluate a variety of state-of-the-art models on this benchmark to demonstrate the utility of our taxonomy in practice. Our taxonomy of generalization can yield many interesting insights into existing models: for example, we observe that current vision-language-action models struggle with various types of semantic generalization, despite the promise of pre-training on internet-scale language datasets. We believe STAR-Gen and our guidelines can improve the dissemination and evaluation of progress towards generalization in robotics, which we hope will guide model design and future data collection efforts. We provide videos and demos at our website this http URL. 

**Abstract (ZH)**: 机器学习在机器人领域的应用有望解锁对新任务和环境的一般化能力。受此启发，许多近期工作集中在扩大机器人数据采集规模并开发更大、更具表达性的策略以实现这一目标。但在实践中，我们如何衡量向这一政策一般化目标的进步呢？在现代机器人领域，评估和量化一般化仍是未开发的领域，每项工作都在其自身难以复现的设置中提出和衡量不同类型的一般化。在本文中，我们的目标是（1）以全面和细腻的方式概述我们认为在机器人操作中重要的各种一般化形式，（2）提供衡量这些一般化概念的可再现指南。我们首先提出STAR-Gen，这是一个基于视觉、语义和行为一般化的机器人操作一般化分类法。我们讨论了我们的分类法如何涵盖机器人学中大多数先前的一般化概念。接下来，我们基于广泛使用的Bridge V2数据集，具体化了STAR-Gen，并提出了一个具体的现实世界基准。我们评估了多种当前最先进的模型在该基准上的表现，以展示我们分类法在实践中的用途。我们的一般化分类法可以对现有模型提供许多有趣的认识：例如，尽管互联网规模语言数据集的预训练承诺，我们观察到当前的视觉-语言-动作模型在各种语义一般化方面存在困难。我们相信STAR-Gen和我们的指南可以改进机器人领域一般化进展的传播和评估，我们希望这将指导模型设计和未来的数据采集努力。我们在网站上提供了视频和演示，链接为 this http URL。 

---
# Enhancing Deep Reinforcement Learning-based Robot Navigation Generalization through Scenario Augmentation 

**Title (ZH)**: 基于场景扩充提高深度强化学习robot导航的泛化能力 

**Authors**: Shanze Wang, Mingao Tan, Zhibo Yang, Xianghui Wang, Xiaoyu Shen, Hailong Huang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01146)  

**Abstract**: This work focuses on enhancing the generalization performance of deep reinforcement learning-based robot navigation in unseen environments. We present a novel data augmentation approach called scenario augmentation, which enables robots to navigate effectively across diverse settings without altering the training scenario. The method operates by mapping the robot's observation into an imagined space, generating an imagined action based on this transformed observation, and then remapping this action back to the real action executed in simulation. Through scenario augmentation, we conduct extensive comparative experiments to investigate the underlying causes of suboptimal navigation behaviors in unseen environments. Our analysis indicates that limited training scenarios represent the primary factor behind these undesired behaviors. Experimental results confirm that scenario augmentation substantially enhances the generalization capabilities of deep reinforcement learning-based navigation systems. The improved navigation framework demonstrates exceptional performance by producing near-optimal trajectories with significantly reduced navigation time in real-world applications. 

**Abstract (ZH)**: 本研究致力于增强基于深度强化学习的机器人导航在未见环境下的泛化性能。我们提出了一种新颖的数据增强方法——情景增强，该方法允许机器人在不改变训练情景的情况下，有效地在多种不同的环境中导航。方法通过将机器人的观察映射到一个想象的空间，基于这个转换后的观察生成想象中的动作，然后将该动作重新映射回模拟中执行的实际动作。通过情景增强，我们进行了广泛的对比实验，以探究在未见环境中产生次优化导航行为的内在原因。分析表明，有限的训练情景是这些不良行为的主要原因。实验结果证实，情景增强显著增强了基于深度强化学习的导航系统的泛化能力。改进的导航框架在实际应用中表现出色，能够生成近乎最优的轨迹，并显著减少导航时间。 

---
# Beyond Visibility Limits: A DRL-Based Navigation Strategy for Unexpected Obstacles 

**Title (ZH)**: 超越可见性限制：基于DRL的意外障碍导航策略 

**Authors**: Mingao Tan, Shanze Wang, Biao Huang, Zhibo Yang, Rongfei Chen, Xiaoyu Shen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01127)  

**Abstract**: Distance-based reward mechanisms in deep reinforcement learning (DRL) navigation systems suffer from critical safety limitations in dynamic environments, frequently resulting in collisions when visibility is restricted. We propose DRL-NSUO, a novel navigation strategy for unexpected obstacles that leverages the rate of change in LiDAR data as a dynamic environmental perception element. Our approach incorporates a composite reward function with environmental change rate constraints and dynamically adjusted weights through curriculum learning, enabling robots to autonomously balance between path efficiency and safety maximization. We enhance sensitivity to nearby obstacles by implementing short-range feature preprocessing of LiDAR data. Experimental results demonstrate that this method significantly improves both robot and pedestrian safety in complex scenarios compared to traditional DRL-based methods. When evaluated on the BARN navigation dataset, our method achieved superior performance with success rates of 94.0% at 0.5 m/s and 91.0% at 1.0 m/s, outperforming conservative obstacle expansion strategies. These results validate DRL-NSUO's enhanced practicality and safety for human-robot collaborative environments, including intelligent logistics applications. 

**Abstract (ZH)**: 基于距离的奖励机制在深度强化学习导航系统中于动态环境下的安全限制 critical 安全限制，在能见度受限的情况下经常导致碰撞。我们提出了一种新的 DRL-NSUO 导航策略，该策略利用 LiDAR 数据变化率作为动态环境感知元素以应对意外障碍。该方法结合了包含环境变化率约束的复合奖励函数，并通过课程学习动态调整权重，使机器人能够自主在路径效率和安全最大化之间取得平衡。通过实施短距离 LiDAR 数据特征预处理来增强对邻近障碍物的敏感度。实验结果表明，与传统的基于 DRL 的方法相比，该方法在复杂场景中显著提高了机器人和行人的安全性。在评估 BARN 导航数据集时，该方法在0.5 m/s 和 1.0 m/s 速度下分别取得了94.0% 和 91.0%的成功率，优于保守的障碍物扩展策略。这些结果验证了 DRL-NSUO 在人机协作环境中的增强实用性和安全性，包括智能物流应用。 

---
# TACO: General Acrobatic Flight Control via Target-and-Command-Oriented Reinforcement Learning 

**Title (ZH)**: TACO: 基于目标和指令导向的强化学习通用杂技飞行控制 

**Authors**: Zikang Yin, Canlun Zheng, Shiliang Guo, Zhikun Wang, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01125)  

**Abstract**: Although acrobatic flight control has been studied extensively, one key limitation of the existing methods is that they are usually restricted to specific maneuver tasks and cannot change flight pattern parameters online. In this work, we propose a target-and-command-oriented reinforcement learning (TACO) framework, which can handle different maneuver tasks in a unified way and allows online parameter changes. Additionally, we propose a spectral normalization method with input-output rescaling to enhance the policy's temporal and spatial smoothness, independence, and symmetry, thereby overcoming the sim-to-real gap. We validate the TACO approach through extensive simulation and real-world experiments, demonstrating its capability to achieve high-speed circular flights and continuous multi-flips. 

**Abstract (ZH)**: 尽管杂技飞行控制已经得到了广泛研究，但现有方法的一个关键局限是通常局限于特定机动任务，不能在线更改飞行模式参数。本文提出了一种目标和指令导向的强化学习（TACO）框架，该框架能够以统一方式处理不同的机动任务，并允许在线参数变化。此外，我们提出了输入输出归一化的谱正则化方法，以增强策略的时间和空间连续性、独立性和对称性，从而克服模拟与现实之间的差距。通过广泛的仿真和实地实验验证了TACO方法的能力，展示了其实现高速圆周飞行和连续多翻转的能力。 

---
# Ground contact and reaction force sensing for linear policy control of quadruped robot 

**Title (ZH)**: quadruped机器人线性策略控制的地面接触与反作用力感知 

**Authors**: Harshita Mhaske, Aniket Mandhare, Jidong Huang, Yu Bai  

**Link**: [PDF](https://arxiv.org/pdf/2503.01102)  

**Abstract**: Designing robots capable of traversing uneven terrain and overcoming physical obstacles has been a longstanding challenge in the field of robotics. Walking robots show promise in this regard due to their agility, redundant DOFs and intermittent ground contact of locomoting appendages. However, the complexity of walking robots and their numerous DOFs make controlling them extremely difficult and computation heavy. Linear policies trained with reinforcement learning have been shown to perform adequately to enable quadrupedal walking, while being computationally light weight. The goal of this research is to study the effect of augmentation of observation space of a linear policy with newer state variables on performance of the policy. Since ground contact and reaction forces are the primary means of robot-environment interaction, they are essential state variables on which the linear policy must be informed. Experimental results show that augmenting the observation space with ground contact and reaction force data trains policies with better survivability, better stability against external disturbances and higher adaptability to untrained conditions. 

**Abstract (ZH)**: 设计能够在不平地形上 traversing 并克服物理障碍的机器人在机器人学领域一直是一个长期挑战。步行机器人由于其灵活性、冗余自由度以及移动肢体的间歇性地面接触，显示出在这一方面的潜力。然而，步行机器人的复杂性和众多自由度使其实现控制极其困难且计算量大。利用强化学习训练的线性策略已被证明在使四足行走变得可行的同时，具有计算量小的优点。本研究的目标是研究将新的状态变量加入线性策略的观测空间对策略性能的影响。由于地面接触和反作用力是机器人与环境交互的主要方式，它们是线性策略必须知晓的关键状态变量。实验结果表明，通过将地面接触和反作用力数据加入观测空间来训练的策略具有更好的生存能力、更好的对外部干扰的稳定性以及更高的对未训练条件的适应性。 

---
# OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework 

**Title (ZH)**: OceanSim：一种基于GPU加速的水下机器人感知仿真框架 

**Authors**: Jingyu Song, Haoyu Ma, Onur Bagoren, Advaith V. Sethuraman, Yiting Zhang, Katherine A. Skinner  

**Link**: [PDF](https://arxiv.org/pdf/2503.01074)  

**Abstract**: Underwater simulators offer support for building robust underwater perception solutions. Significant work has recently been done to develop new simulators and to advance the performance of existing underwater simulators. Still, there remains room for improvement on physics-based underwater sensor modeling and rendering efficiency. In this paper, we propose OceanSim, a high-fidelity GPU-accelerated underwater simulator to address this research gap. We propose advanced physics-based rendering techniques to reduce the sim-to-real gap for underwater image simulation. We develop OceanSim to fully leverage the computing advantages of GPUs and achieve real-time imaging sonar rendering and fast synthetic data generation. We evaluate the capabilities and realism of OceanSim using real-world data to provide qualitative and quantitative results. The project page for OceanSim is this https URL. 

**Abstract (ZH)**: 水下模拟器为构建 robust 的水下感知解决方案提供支持。尽管已经开展了许多工作以开发新的模拟器并推进现有水下模拟器的性能，但在基于物理的水下传感器建模和渲染效率方面仍有改进空间。本文提出了一种高保真度的 GPU 加速水下模拟器 OceanSim，以解决这一研究缺口。我们提出先进的基于物理的渲染技术以减少水下图像模拟的模拟与现实差距。我们开发 OceanSim 以充分利用 GPU 的计算优势，实现实时声呐成像渲染和快速合成数据生成。我们使用真实世界的数据评估 OceanSim 的能力和真实性，提供定性和定量结果。OceanSim 的项目页面为：https://github.com/alibaba/OceanSim。 

---
# Language-Guided Object Search in Agricultural Environments 

**Title (ZH)**: 农业环境中基于语言的对象搜索 

**Authors**: Advaith Balaji, Saket Pradhan, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01068)  

**Abstract**: Creating robots that can assist in farms and gardens can help reduce the mental and physical workload experienced by farm workers. We tackle the problem of object search in a farm environment, providing a method that allows a robot to semantically reason about the location of an unseen target object among a set of previously seen objects in the environment using a Large Language Model (LLM). We leverage object-to-object semantic relationships to plan a path through the environment that will allow us to accurately and efficiently locate our target object while also reducing the overall distance traveled, without needing high-level room or area-level semantic relationships. During our evaluations, we found that our method outperformed a current state-of-the-art baseline and our ablations. Our offline testing yielded an average path efficiency of 84%, reflecting how closely the predicted path aligns with the ideal path. Upon deploying our system on the Boston Dynamics Spot robot in a real-world farm environment, we found that our system had a success rate of 80%, with a success weighted by path length of 0.67, which demonstrates a reasonable trade-off between task success and path efficiency under real-world conditions. The project website can be viewed at this https URL 

**Abstract (ZH)**: 在农场环境中创建能辅助工作的机器人可以减轻农场工人所面临的身心负担。我们解决了在农场环境中进行对象搜索的问题，提出了一种方法，通过大型语言模型（LLM），使机器人能够在一组之前在环境中看到的对象中，对一个未见的目标对象的位置进行语义推理。我们利用对象之间的语义关系来规划一条路径，该路径能准确且高效地定位目标对象，同时减少总体移动距离，而不需依赖高层的房间或区域级别的语义关系。在评估中，我们的方法优于当前最先进的基线和我们的消融实验。我们的离线测试显示路径效率平均值为84%，反映了预测路径与理想路径的接近程度。在实际农场环境中部署我们的系统到Boston Dynamics Spot机器人后，我们发现系统的成功率达到了80%，加权路径长度的成功率达到0.67，这在实际条件下展示了任务成功与路径效率之间的合理权衡。项目网站可浏览 [此链接]。 

---
# From Vague Instructions to Task Plans: A Feedback-Driven HRC Task Planning Framework based on LLMs 

**Title (ZH)**: 从模糊指令到任务计划：基于LLM的反馈驱动人机协作任务规划框架 

**Authors**: Afagh Mehri Shervedani, Matthew R. Walter, Milos Zefran  

**Link**: [PDF](https://arxiv.org/pdf/2503.01007)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated their potential as planners in human-robot collaboration (HRC) scenarios, offering a promising alternative to traditional planning methods. LLMs, which can generate structured plans by reasoning over natural language inputs, have the ability to generalize across diverse tasks and adapt to human instructions. This paper investigates the potential of LLMs to facilitate planning in the context of human-robot collaborative tasks, with a focus on their ability to reason from high-level, vague human inputs, and fine-tune plans based on real-time feedback. We propose a novel hybrid framework that combines LLMs with human feedback to create dynamic, context-aware task plans. Our work also highlights how a single, concise prompt can be used for a wide range of tasks and environments, overcoming the limitations of long, detailed structured prompts typically used in prior studies. By integrating user preferences into the planning loop, we ensure that the generated plans are not only effective but aligned with human intentions. 

**Abstract (ZH)**: 近期大型语言模型在-human-机器人协作规划中的进展：基于自然语言推理的通用能力和人机反馈动态调优框架 

---
# HWC-Loco: A Hierarchical Whole-Body Control Approach to Robust Humanoid Locomotion 

**Title (ZH)**: HWC-Loco：一种稳健的人形步行的分层次全身控制方法 

**Authors**: Sixu Lin, Guanren Qiao, Yunxin Tai, Ang Li, Kui Jia, Guiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00923)  

**Abstract**: Humanoid robots, capable of assuming human roles in various workplaces, have become essential to the advancement of embodied intelligence. However, as robots with complex physical structures, learning a control model that can operate robustly across diverse environments remains inherently challenging, particularly under the discrepancies between training and deployment environments. In this study, we propose HWC-Loco, a robust whole-body control algorithm tailored for humanoid locomotion tasks. By reformulating policy learning as a robust optimization problem, HWC-Loco explicitly learns to recover from safety-critical scenarios. While prioritizing safety guarantees, overly conservative behavior can compromise the robot's ability to complete the given tasks. To tackle this challenge, HWC-Loco leverages a hierarchical policy for robust control. This policy can dynamically resolve the trade-off between goal-tracking and safety recovery, guided by human behavior norms and dynamic constraints. To evaluate the performance of HWC-Loco, we conduct extensive comparisons against state-of-the-art humanoid control models, demonstrating HWC-Loco's superior performance across diverse terrains, robot structures, and locomotion tasks under both simulated and real-world environments. 

**Abstract (ZH)**: humanoid机器人，能够在各种工作场所扮演人类角色，已成为推进体现智能的关键。然而，作为具有复杂物理结构的机器人，学习能够在不同环境中稳健运行的控制模型仍然充满挑战，特别是在训练环境与部署环境之间的差异条件下。在本研究中，我们提出HWC-Loco，一种针对人形移动任务的稳健全身控制算法。通过将策略学习重新定义为稳健优化问题，HWC-Loco明确地学习从关键安全场景中恢复。虽然优先考虑安全保证，但过于保守的行为可能损害机器人完成给定任务的能力。为应对这一挑战，HWC-Loco采用分层策略进行稳健控制。该策略可以根据人类行为规范和动态约束动态调节目标跟踪与安全恢复之间的权衡。为了评估HWC-Loco的性能，我们在模拟和真实世界环境中对不同地形、机器人结构和移动任务的现代人形控制模型进行了广泛的比较，结果显示HWC-Loco在各种条件下表现出更优的性能。 

---
# T3: Multi-modal Tailless Triple-Flapping-Wing Robot for Efficient Aerial and Terrestrial Locomotion 

**Title (ZH)**: T3：多模态无尾三振动翼地面与空中移动机器人 

**Authors**: Xiangyu Xu, Zhi Zheng, Jin Wang, Yikai Chen, Jingyang Huang, Ruixin Wu, Huan Yu, Guodong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00805)  

**Abstract**: Flapping-wing robots offer great versatility; however, achieving efficient multi-modal locomotion remains challenging. This paper presents the design, modeling, and experimentation of T3, a novel tailless flapping-wing robot with three pairs of independently actuated wings. Inspired by juvenile water striders, T3 incorporates bio-inspired elastic passive legs that effectively transmit vibrations generated during wing flapping, enabling ground movement without additional motors. This novel mechanism facilitates efficient multi-modal locomotion while minimizing actuator usage, reducing complexity, and enhancing performance. An SE(3)-based controller ensures precise trajectory tracking and seamless mode transition. To validate T3's effectiveness, we developed a fully functional prototype and conducted targeted modeling, real-world experiments, and benchmark comparisons. The results demonstrate the robot's and controller's outstanding performance, underscoring the potential of multi-modal flapping-wing technologies for future aerial-ground robotic applications. 

**Abstract (ZH)**: 无尾拍翼机器人T3的设计、建模与实验：基于仿生弹性 Legs 的高效多模态运动 

---
# CARIL: Confidence-Aware Regression in Imitation Learning for Autonomous Driving 

**Title (ZH)**: CARIL: 带有置信度感知的 imitation 学习在自动驾驶中的回归关键技术 

**Authors**: Elahe Delavari, Aws Khalil, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2503.00783)  

**Abstract**: End-to-end vision-based imitation learning has demonstrated promising results in autonomous driving by learning control commands directly from expert demonstrations. However, traditional approaches rely on either regressionbased models, which provide precise control but lack confidence estimation, or classification-based models, which offer confidence scores but suffer from reduced precision due to discretization. This limitation makes it challenging to quantify the reliability of predicted actions and apply corrections when necessary. In this work, we introduce a dual-head neural network architecture that integrates both regression and classification heads to improve decision reliability in imitation learning. The regression head predicts continuous driving actions, while the classification head estimates confidence, enabling a correction mechanism that adjusts actions in low-confidence scenarios, enhancing driving stability. We evaluate our approach in a closed-loop setting within the CARLA simulator, demonstrating its ability to detect uncertain actions, estimate confidence, and apply real-time corrections. Experimental results show that our method reduces lane deviation and improves trajectory accuracy by up to 50%, outperforming conventional regression-only models. These findings highlight the potential of classification-guided confidence estimation in enhancing the robustness of vision-based imitation learning for autonomous driving. The source code is available at this https URL. 

**Abstract (ZH)**: 基于视觉的端到端模仿学习通过直接从专家演示中学习控制命令，在自主驾驶领域展示了 promising 的结果。然而，传统方法依赖于提供精确控制但缺乏信心估计的回归模型，或提供信心评分但因离散化而精度降低的分类模型。这一局限使得量化预测动作的可靠性并在必要时进行修正变得具有挑战性。在本工作中，我们提出了一种双头神经网络架构，将回归和分类头集成在一起以提高模仿学习中的决策可靠性。回归头预测连续驾驶动作，而分类头估计信心，使在低信心场景下能够调整动作，从而增强驾驶稳定性。我们在 CARLA 模拟器中闭环环境中评估了我们的方法，展示了其检测不确定动作、估计信心并实时修正的能力。实验结果表明，与仅使用回归模型相比，我们的方法将车道偏离减少，并且轨迹精度提高最多 50%。这些发现突显了分类引导的信心估计在增强基于视觉的模仿学习鲁棒性方面的潜力。源代码可在以下链接获取：this https URL。 

---
# Phantom: Training Robots Without Robots Using Only Human Videos 

**Title (ZH)**: 幻影：仅使用人类视频训练机器人 

**Authors**: Marion Lepert, Jiaying Fang, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2503.00779)  

**Abstract**: Scaling robotics data collection is critical to advancing general-purpose robots. Current approaches often rely on teleoperated demonstrations which are difficult to scale. We propose a novel data collection method that eliminates the need for robotics hardware by leveraging human video demonstrations. By training imitation learning policies on this human data, our approach enables zero-shot deployment on robots without collecting any robot-specific data. To bridge the embodiment gap between human and robot appearances, we utilize a data editing approach on the input observations that aligns the image distributions between training data on humans and test data on robots. Our method significantly reduces the cost of diverse data collection by allowing anyone with an RGBD camera to contribute. We demonstrate that our approach works in diverse, unseen environments and on varied tasks. 

**Abstract (ZH)**: 通过利用人类视频示范来扩展机器人数据收集对于推进通用机器人技术至关重要。我们提出了一种新颖的数据收集方法，该方法通过利用人类视频示范而不依赖于机器人硬件，从而消除了对机器人特定数据的收集需求。为了弥合人类和机器人外观之间的主体差异，我们对输入观察数据进行数据编辑，以使训练数据的人像分布与测试数据的机器人分布对齐。我们的方法通过允许任何人使用RGBD相机贡献数据，显著降低了多样数据收集的成本。我们证明了该方法在未见过的多种环境和任务中有效。 

---
# AffordGrasp: In-Context Affordance Reasoning for Open-Vocabulary Task-Oriented Grasping in Clutter 

**Title (ZH)**: AffordGrasp: 开放词汇任务导向杂乱环境中的获取能力推理 

**Authors**: Yingbo Tang, Shuaike Zhang, Xiaoshuai Hao, Pengwei Wang, Jianlong Wu, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00778)  

**Abstract**: Inferring the affordance of an object and grasping it in a task-oriented manner is crucial for robots to successfully complete manipulation tasks. Affordance indicates where and how to grasp an object by taking its functionality into account, serving as the foundation for effective task-oriented grasping. However, current task-oriented methods often depend on extensive training data that is confined to specific tasks and objects, making it difficult to generalize to novel objects and complex scenes. In this paper, we introduce AffordGrasp, a novel open-vocabulary grasping framework that leverages the reasoning capabilities of vision-language models (VLMs) for in-context affordance reasoning. Unlike existing methods that rely on explicit task and object specifications, our approach infers tasks directly from implicit user instructions, enabling more intuitive and seamless human-robot interaction in everyday scenarios. Building on the reasoning outcomes, our framework identifies task-relevant objects and grounds their part-level affordances using a visual grounding module. This allows us to generate task-oriented grasp poses precisely within the affordance regions of the object, ensuring both functional and context-aware robotic manipulation. Extensive experiments demonstrate that AffordGrasp achieves state-of-the-art performance in both simulation and real-world scenarios, highlighting the effectiveness of our method. We believe our approach advances robotic manipulation techniques and contributes to the broader field of embodied AI. Project website: this https URL. 

**Abstract (ZH)**: 基于任务的物体利用能力和抓取：一种利用视觉语言模型进行上下文推理的开放式词汇抓取框架 

---
# Shadow: Leveraging Segmentation Masks for Cross-Embodiment Policy Transfer 

**Title (ZH)**: Shadow: 利用分割掩码进行跨 bodys良政策转移 

**Authors**: Marion Lepert, Ria Doshi, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2503.00774)  

**Abstract**: Data collection in robotics is spread across diverse hardware, and this variation will increase as new hardware is developed. Effective use of this growing body of data requires methods capable of learning from diverse robot embodiments. We consider the setting of training a policy using expert trajectories from a single robot arm (the source), and evaluating on a different robot arm for which no data was collected (the target). We present a data editing scheme termed Shadow, in which the robot during training and evaluation is replaced with a composite segmentation mask of the source and target robots. In this way, the input data distribution at train and test time match closely, enabling robust policy transfer to the new unseen robot while being far more data efficient than approaches that require co-training on large amounts of data from diverse embodiments. We demonstrate that an approach as simple as Shadow is effective both in simulation on varying tasks and robots, and on real robot hardware, where Shadow demonstrates an average of over 2x improvement in success rate compared to the strongest baseline. 

**Abstract (ZH)**: 机器人领域的数据收集分布于多种硬件之上，随着新硬件的开发，这种差异将会增大。有效利用不断增加的数据体需要能够从多种机器人实体中学习的方法。我们考虑从单个机器人手臂（源）的专家轨迹训练一个策略，并在没有收集数据的不同机器人手臂（目标）上进行评估。我们提出了一种名为Shadow的数据编辑方案，在此方案中，训练和评估时的机器人被源机器人和目标机器人的复合分割掩码所替换。这样，训练和测试时的输入数据分布非常接近，能够稳健地将策略转移到新的未见机器人，且相比于需要在大量多样化实体数据上共同训练的方法，这种方法更为数据高效。我们证明，在不同的任务和机器人以及真实机器人硬件上，Shadow方法在成功率上平均优于最强基线约2倍。 

---
# Disturbance Estimation of Legged Robots: Predefined Convergence via Dynamic Gains 

**Title (ZH)**: 腿足机器人扰动估计算法：基于动态增益的预先定义收敛性 

**Authors**: Bolin Li, Peiyuan Cai, Gewei Zuo, Lijun Zhu, Han Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.00769)  

**Abstract**: In this study, we address the challenge of disturbance estimation in legged robots by introducing a novel continuous-time online feedback-based disturbance observer that leverages measurable variables. The distinct feature of our observer is the integration of dynamic gains and comparison functions, which guarantees predefined convergence of the disturbance estimation error, including ultimately uniformly bounded, asymptotic, and exponential convergence, among various types. The properties of dynamic gains and the sufficient conditions for comparison functions are detailed to guide engineers in designing desired convergence behaviors. Notably, the observer functions effectively without the need for upper bound information of the disturbance or its derivative, enhancing its engineering applicability. An experimental example corroborates the theoretical advancements achieved. 

**Abstract (ZH)**: 本研究通过引入一种新型的连续时间在线反馈型扰动观察器来应对腿式机器人中的扰动估计挑战，该观察器利用可测量变量。观察器的独特之处在于集成了动态增益和比较函数，确保扰动估计误差在不同类型中达到预定义的收敛性，包括终极一致有界、渐近和指数收敛。详细讨论了动态增益的性质和比较函数的充分条件，以指导工程师设计所需的收敛行为。值得注意的是，该观察器在无需扰动及其导数的上界信息的情况下有效运行，增强了其实用性。实验示例验证了理论进展。 

---
# TRACE: A Self-Improving Framework for Robot Behavior Forecasting with Vision-Language Models 

**Title (ZH)**: TRACE：一种基于视觉语言模型的自我提升机器人行为预测框架 

**Authors**: Gokul Puthumanaillam, Paulo Padrao, Jose Fuentes, Pranay Thangeda, William E. Schafer, Jae Hyuk Song, Karan Jagdale, Leonardo Bobadilla, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2503.00761)  

**Abstract**: Predicting the near-term behavior of a reactive agent is crucial in many robotic scenarios, yet remains challenging when observations of that agent are sparse or intermittent. Vision-Language Models (VLMs) offer a promising avenue by integrating textual domain knowledge with visual cues, but their one-shot predictions often miss important edge cases and unusual maneuvers. Our key insight is that iterative, counterfactual exploration--where a dedicated module probes each proposed behavior hypothesis, explicitly represented as a plausible trajectory, for overlooked possibilities--can significantly enhance VLM-based behavioral forecasting. We present TRACE (Tree-of-thought Reasoning And Counterfactual Exploration), an inference framework that couples tree-of-thought generation with domain-aware feedback to refine behavior hypotheses over multiple rounds. Concretely, a VLM first proposes candidate trajectories for the agent; a counterfactual critic then suggests edge-case variations consistent with partial observations, prompting the VLM to expand or adjust its hypotheses in the next iteration. This creates a self-improving cycle where the VLM progressively internalizes edge cases from previous rounds, systematically uncovering not only typical behaviors but also rare or borderline maneuvers, ultimately yielding more robust trajectory predictions from minimal sensor data. We validate TRACE on both ground-vehicle simulations and real-world marine autonomous surface vehicles. Experimental results show that our method consistently outperforms standard VLM-driven and purely model-based baselines, capturing a broader range of feasible agent behaviors despite sparse sensing. Evaluation videos and code are available at this http URL. 

**Abstract (ZH)**: 基于迭代反事实探索的反应性代理近期行为预测 

---
# CLEA: Closed-Loop Embodied Agent for Enhancing Task Execution in Dynamic Environments 

**Title (ZH)**: CLEA: 闭环身体化代理在动态环境中的任务执行增强方法 

**Authors**: Mingcong Lei, Ge Wang, Yiming Zhao, Zhixin Mai, Qing Zhao, Yao Guo, Zhen Li, Shuguang Cui, Yatong Han, Jinke Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.00729)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities in the hierarchical decomposition of complex tasks through semantic reasoning. However, their application in embodied systems faces challenges in ensuring reliable execution of subtask sequences and achieving one-shot success in long-term task completion. To address these limitations in dynamic environments, we propose Closed-Loop Embodied Agent (CLEA) -- a novel architecture incorporating four specialized open-source LLMs with functional decoupling for closed-loop task management. The framework features two core innovations: (1) Interactive task planner that dynamically generates executable subtasks based on the environmental memory, and (2) Multimodal execution critic employing an evaluation framework to conduct a probabilistic assessment of action feasibility, triggering hierarchical re-planning mechanisms when environmental perturbations exceed preset thresholds. To validate CLEA's effectiveness, we conduct experiments in a real environment with manipulable objects, using two heterogeneous robots for object search, manipulation, and search-manipulation integration tasks. Across 12 task trials, CLEA outperforms the baseline model, achieving a 67.3% improvement in success rate and a 52.8% increase in task completion rate. These results demonstrate that CLEA significantly enhances the robustness of task planning and execution in dynamic environments. 

**Abstract (ZH)**: 基于闭环控制的 embodied 代理（CLEA）：面向动态环境的任务规划与执行 

---
# From Understanding the World to Intervening in It: A Unified Multi-Scale Framework for Embodied Cognition 

**Title (ZH)**: 从理解世界到干预世界：统一的多尺度知体认知框架 

**Authors**: Maijunxian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00727)  

**Abstract**: In this paper, we propose AUKAI, an Adaptive Unified Knowledge-Action Intelligence for embodied cognition that seamlessly integrates perception, memory, and decision-making via multi-scale error feedback. Interpreting AUKAI as an embedded world model, our approach simultaneously predicts state transitions and evaluates intervention utility. The framework is underpinned by rigorous theoretical analysis drawn from convergence theory, optimal control, and Bayesian inference, which collectively establish conditions for convergence, stability, and near-optimal performance. Furthermore, we present a hybrid implementation that combines the strengths of neural networks with symbolic reasoning modules, thereby enhancing interpretability and robustness. Finally, we demonstrate the potential of AUKAI through a detailed application in robotic navigation and obstacle avoidance, and we outline comprehensive experimental plans to validate its effectiveness in both simulated and real-world environments. 

**Abstract (ZH)**: 本文提出了一种自适应统一知识-行动智能AUKAI，它通过多尺度误差反馈无缝地整合感知、记忆和决策，用于体现认知。将AUKAI视为嵌入式世界模型，我们的方法同时预测状态转换并评估干预的效用。该框架基于从收敛理论、最优控制和贝叶斯推断中汲取的严格理论分析，共同确立了收敛性、稳定性和接近最优性能的条件。此外，我们提出了一种混合实现方法，结合了神经网络的优点和符号推理模块，从而增强了可解释性和鲁棒性。最后，我们通过一个详细的机器人导航和障碍物避免应用展示了AUKAI的潜在价值，并概述了全面的实验计划，以验证其在模拟和真实世界环境中的有效性。 

---
# Learning Perceptive Humanoid Locomotion over Challenging Terrain 

**Title (ZH)**: 学习穿越挑战性地形的感知 humanoid 运动 

**Authors**: Wandong Sun, Baoshi Cao, Long Chen, Yongbo Su, Yang Liu, Zongwu Xie, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00692)  

**Abstract**: Humanoid robots are engineered to navigate terrains akin to those encountered by humans, which necessitates human-like locomotion and perceptual abilities. Currently, the most reliable controllers for humanoid motion rely exclusively on proprioception, a reliance that becomes both dangerous and unreliable when coping with rugged terrain. Although the integration of height maps into perception can enable proactive gait planning, robust utilization of this information remains a significant challenge, especially when exteroceptive perception is noisy. To surmount these challenges, we propose a solution based on a teacher-student distillation framework. In this paradigm, an oracle policy accesses noise-free data to establish an optimal reference policy, while the student policy not only imitates the teacher's actions but also simultaneously trains a world model with a variational information bottleneck for sensor denoising and state estimation. Extensive evaluations demonstrate that our approach markedly enhances performance in scenarios characterized by unreliable terrain estimations. Moreover, we conducted rigorous testing in both challenging urban settings and off-road environments, the model successfully traverse 2 km of varied terrain without external intervention. 

**Abstract (ZH)**: 类人机器人被设计为能够在类似人类遇到的地形中导航，这需要类似人类的运动能力和感知能力。目前，最可靠的类人运动控制器依赖于本体感受，但在应对崎岖地形时，这种依赖变得既危险又不可靠。虽然将高度图整合到感知中可以实现主动步态规划，但如何稳健地利用这些信息仍然是一个重大挑战，尤其是在外部感知噪声较大的情况下。为了克服这些挑战，我们提出了一种基于教师-学生蒸馏框架的解决方案。在这个框架中，先验策略访问无噪声数据以建立最优参考策略，而学生策略不仅模仿教师的动作，还同时通过变分信息瓶颈训练世界模型进行传感器去噪和状态估计。广泛的评估表明，我们的方法在地形不可靠的情况下显著提高了性能。此外，我们在具有挑战性的城市环境和非道路环境中进行了严格的测试，模型成功地在没有外部干预的情况下穿越了2公里的复杂地形。 

---
# Autonomous Dissection in Robotic Cholecystectomy 

**Title (ZH)**: 自主机器人胆囊切除术中的自动解剖分离 

**Authors**: Ki-Hwan Oh, Leonardo Borgioli, Miloš Žefran, Valentina Valle, Pier Cristoforo Giulianotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.00666)  

**Abstract**: Robotic surgery offers enhanced precision and adaptability, paving the way for automation in surgical interventions. Cholecystectomy, the gallbladder removal, is particularly well-suited for automation due to its standardized procedural steps and distinct anatomical boundaries. A key challenge in automating this procedure is dissecting with accuracy and adaptability. This paper presents a vision-based autonomous robotic dissection architecture that integrates real-time segmentation, keypoint detection, grasping and stretching the gallbladder with the left arm, and dissecting with the other. We introduce an improved segmentation dataset based on videos of robotic cholecystectomy performed by various surgeons, incorporating a new ``liver bed'' class to enhance boundary tracking after multiple rounds of dissection. Our system employs state-of-the-art segmentation models and an adaptive boundary extraction method that maintains accuracy despite tissue deformations and visual variations. Moreover, we implemented an automated grasping and pulling strategy to optimize tissue tension before dissection upon our previous work. Ex vivo evaluations on porcine livers demonstrate that our framework significantly improves dissection precision and consistency, marking a step toward fully autonomous robotic cholecystectomy. 

**Abstract (ZH)**: 基于视觉的自主机器人解剖架构：用于自动化胆囊切除术的精确和适应性解剖 

---
# CAP: A Connectivity-Aware Hierarchical Coverage Path Planning Algorithm for Unknown Environments using Coverage Guidance Graph 

**Title (ZH)**: CAP：一种基于连通性的分层覆盖路径规划算法用于未知环境的覆盖指导图方法 

**Authors**: Zongyuan Shen, Burhanuddin Shirose, Prasanna Sriganesh, Matthew Travers  

**Link**: [PDF](https://arxiv.org/pdf/2503.00647)  

**Abstract**: Efficient coverage of unknown environments requires robots to adapt their paths in real time based on on-board sensor data. In this paper, we introduce CAP, a connectivity-aware hierarchical coverage path planning algorithm for efficient coverage of unknown environments. During online operation, CAP incrementally constructs a coverage guidance graph to capture essential information about the environment. Based on the updated graph, the hierarchical planner determines an efficient path to maximize global coverage efficiency and minimize local coverage time. The performance of CAP is evaluated and compared with five baseline algorithms through high-fidelity simulations as well as robot experiments. Our results show that CAP yields significant improvements in coverage time, path length, and path overlap ratio. 

**Abstract (ZH)**: 面向连通性的分层覆盖路径规划算法CAP：未知环境的有效覆盖 

---
# Actor-Critic Cooperative Compensation to Model Predictive Control for Off-Road Autonomous Vehicles Under Unknown Dynamics 

**Title (ZH)**: 基于未知动力学的离-road自主车辆模型预测控制的Actor-Critic协同补偿方法 

**Authors**: Prakhar Gupta, Jonathon M Smereka, Yunyi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.00577)  

**Abstract**: This study presents an Actor-Critic Cooperative Compensated Model Predictive Controller (AC3MPC) designed to address unknown system dynamics. To avoid the difficulty of modeling highly complex dynamics and ensuring realtime control feasibility and performance, this work uses deep reinforcement learning with a model predictive controller in a cooperative framework to handle unknown dynamics. The model-based controller takes on the primary role as both controllers are provided with predictive information about the other. This improves tracking performance and retention of inherent robustness of the model predictive controller. We evaluate this framework for off-road autonomous driving on unknown deformable terrains that represent sandy deformable soil, sandy and rocky soil, and cohesive clay-like deformable soil. Our findings demonstrate that our controller statistically outperforms standalone model-based and learning-based controllers by upto 29.2% and 10.2%. This framework generalized well over varied and previously unseen terrain characteristics to track longitudinal reference speeds with lower errors. Furthermore, this required significantly less training data compared to purely learning-based controller, while delivering better performance even when under-trained. 

**Abstract (ZH)**: 基于Actor-Critic合作补偿模型预测控制的未知系统动力学处理方法研究 

---
# Enhancing Context-Aware Human Motion Prediction for Efficient Robot Handovers 

**Title (ZH)**: 增强上下文意识的人体运动预测以实现高效的机器人交接 

**Authors**: Gerard Gómez-Izquierdo, Javier Laplaza, Alberto Sanfeliu, Anaís Garrell  

**Link**: [PDF](https://arxiv.org/pdf/2503.00576)  

**Abstract**: Accurate human motion prediction (HMP) is critical for seamless human-robot collaboration, particularly in handover tasks that require real-time adaptability. Despite the high accuracy of state-of-the-art models, their computational complexity limits practical deployment in real-world robotic applications. In this work, we enhance human motion forecasting for handover tasks by leveraging siMLPe [1], a lightweight yet powerful architecture, and introducing key improvements. Our approach, named IntentMotion incorporates intention-aware conditioning, task-specific loss functions, and a novel intention classifier, significantly improving motion prediction accuracy while maintaining efficiency. Experimental results demonstrate that our method reduces body loss error by over 50%, achieves 200x faster inference, and requires only 3% of the parameters compared to existing state-of-the-art HMP models. These advancements establish our framework as a highly efficient and scalable solution for real-time human-robot interaction. 

**Abstract (ZH)**: 准确的人体运动预测（HMP）对于无缝的人机协作至关重要，特别是在需要实时适应性的交接任务中。尽管最先进的模型具有较高的准确性，但其计算复杂性限制了其实用部署在现实机器人应用中的应用。在本文中，我们通过利用轻量但强大的siMLPe架构并引入关键改进，增强了交接任务中的人体运动预测。我们的方法名为IntentMotion， Incorporates意图感知的条件输入、任务特定的损失函数以及一种新型的意图分类器，显著提高了运动预测准确性的同时保持了高效性。实验结果表明，我们的方法将身体损耗错误降低了超过50%，推理速度提高了200倍，并且只需要现有最先进的HMP模型3%的参数。这些进步确立了我们框架作为实时人机交互的高效且可扩展解决方案的地位。 

---
# BodyGen: Advancing Towards Efficient Embodiment Co-Design 

**Title (ZH)**: BodyGen: 向高效身躯联合设计迈进 

**Authors**: Haofei Lu, Zhe Wu, Junliang Xing, Jianshu Li, Ruoyu Li, Zhe Li, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00533)  

**Abstract**: Embodiment co-design aims to optimize a robot's morphology and control policy simultaneously. While prior work has demonstrated its potential for generating environment-adaptive robots, this field still faces persistent challenges in optimization efficiency due to the (i) combinatorial nature of morphological search spaces and (ii) intricate dependencies between morphology and control. We prove that the ineffective morphology representation and unbalanced reward signals between the design and control stages are key obstacles to efficiency. To advance towards efficient embodiment co-design, we propose BodyGen, which utilizes (1) topology-aware self-attention for both design and control, enabling efficient morphology representation with lightweight model sizes; (2) a temporal credit assignment mechanism that ensures balanced reward signals for optimization. With our findings, Body achieves an average 60.03% performance improvement against state-of-the-art baselines. We provide codes and more results on the website: this https URL. 

**Abstract (ZH)**: 基于体态的协同设计旨在同时优化机器人形态和控制策略。尽管先前的工作已经展示了其生成环境适应型机器人的潜力，但由于形态搜索空间的组合性质以及形态与控制之间的复杂依赖关系，该领域仍然面临优化效率的持续挑战。我们证明了无效的形态表示以及设计阶段和控制阶段之间不均衡的奖励信号是效率问题的关键障碍。为了实现高效的基于体态的协同设计，我们提出了BodyGen，该方法利用（1）拓扑感知自注意力机制，以轻量级模型大小实现高效形态表示；（2）时序归因机制以确保优化过程中的奖励信号平衡。基于我们的研究发现，Body在与最新基线方法相比时，平均提高了60.03%的性能。我们在网站上提供了代码和更多结果：this https URL。 

---
# HGDiffuser: Efficient Task-Oriented Grasp Generation via Human-Guided Grasp Diffusion Models 

**Title (ZH)**: HGDiffuser：高效的任务导向抓取生成通过人类引导的抓取扩散模型 

**Authors**: Dehao Huang, Wenlong Dong, Chao Tang, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00508)  

**Abstract**: Task-oriented grasping (TOG) is essential for robots to perform manipulation tasks, requiring grasps that are both stable and compliant with task-specific constraints. Humans naturally grasp objects in a task-oriented manner to facilitate subsequent manipulation tasks. By leveraging human grasp demonstrations, current methods can generate high-quality robotic parallel-jaw task-oriented grasps for diverse objects and tasks. However, they still encounter challenges in maintaining grasp stability and sampling efficiency. These methods typically rely on a two-stage process: first performing exhaustive task-agnostic grasp sampling in the 6-DoF space, then applying demonstration-induced constraints (e.g., contact regions and wrist orientations) to filter candidates. This leads to inefficiency and potential failure due to the vast sampling space. To address this, we propose the Human-guided Grasp Diffuser (HGDiffuser), a diffusion-based framework that integrates these constraints into a guided sampling process. Through this approach, HGDiffuser directly generates 6-DoF task-oriented grasps in a single stage, eliminating exhaustive task-agnostic sampling. Furthermore, by incorporating Diffusion Transformer (DiT) blocks as the feature backbone, HGDiffuser improves grasp generation quality compared to MLP-based methods. Experimental results demonstrate that our approach significantly improves the efficiency of task-oriented grasp generation, enabling more effective transfer of human grasping strategies to robotic systems. To access the source code and supplementary videos, visit this https URL. 

**Abstract (ZH)**: 面向任务的抓取（TOG）是机器人执行操作任务的关键，要求抓取既稳定又符合特定任务约束。人类在执行后续操作任务时会自然地以任务为导向抓取物体。借助人类抓取示例，当前方法可以生成高质量的机器人平行夹爪面向任务的抓取，适用于多种物体和任务。然而，这些方法在保持抓取稳定性和采样效率方面仍然面临挑战。这些方法通常依赖于两阶段过程：首先在6-DOF空间进行 exhaustive 任务无关的抓取采样，然后应用由示例诱导的约束（如接触区域和手腕方向）来筛选候选抓取。这导致了采样的低效率和潜在的失败。为了解决这一问题，我们提出了由人类指导的抓取扩散器（HGDiffuser），这是一种基于扩散的方法，将这些约束整合到引导采样过程之中。借助此方法，HGDiffuser可以直接在单阶段生成6-DOF面向任务的抓取，去掉了 exhaustive 任务无关的采样。此外，通过将扩散转换器（DiT）块作为特征骨干，HGDiffuser 提高了抓取生成质量，优于基于MLP的方法。实验结果表明，我们的方法显著提高了面向任务的抓取生成效率，使得将人类抓取策略更有效地转移到机器人系统中成为可能。要访问源代码和补充视频，访问此链接：https://this-url 

---
# Model-based optimisation for the personalisation of robot-assisted gait training 

**Title (ZH)**: 基于模型的优化方法用于机器人辅助步态训练的个性化调整 

**Authors**: Andreas Christou, Daniel F. N. Gordon, Theodoros Stouraitis, Juan C. Moreno, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.00480)  

**Abstract**: Personalised rehabilitation can be key to promoting gait independence and quality of life. Robots can enhance therapy by systematically delivering support in gait training, but often use one-size-fits-all control methods, which can be suboptimal. Here, we describe a model-based optimisation method for designing and fine-tuning personalised robotic controllers. As a case study, we formulate the objective of providing assistance as needed as an optimisation problem, and we demonstrate how musculoskeletal modelling can be used to develop personalised interventions. Eighteen healthy participants (age = 26 +/- 4) were recruited and the personalised control parameters for each were obtained to provide assistance as needed during a unilateral tracking task. A comparison was carried out between the personalised controller and the non-personalised controller. In simulation, a significant improvement was predicted when the personalised parameters were used. Experimentally, responses varied: six subjects showed significant improvements with the personalised parameters, eight subjects showed no obvious change, while four subjects performed worse. High interpersonal and intra-personal variability was observed with both controllers. This study highlights the importance of personalised control in robot-assisted gait training, and the need for a better estimation of human-robot interaction and human behaviour to realise the benefits of model-based optimisation. 

**Abstract (ZH)**: 基于模型的个性化康复机器人控制器优化方法研究：助力需求为导向的下肢功能训练与生活品质提升 

---
# Scalable Real2Sim: Physics-Aware Asset Generation Via Robotic Pick-and-Place Setups 

**Title (ZH)**: 可扩展的Real2Sim：基于机器人拾放设置的物理感知资产生成 

**Authors**: Nicholas Pfaff, Evelyn Fu, Jeremy Binagia, Phillip Isola, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2503.00370)  

**Abstract**: Simulating object dynamics from real-world perception shows great promise for digital twins and robotic manipulation but often demands labor-intensive measurements and expertise. We present a fully automated Real2Sim pipeline that generates simulation-ready assets for real-world objects through robotic interaction. Using only a robot's joint torque sensors and an external camera, the pipeline identifies visual geometry, collision geometry, and physical properties such as inertial parameters. Our approach introduces a general method for extracting high-quality, object-centric meshes from photometric reconstruction techniques (e.g., NeRF, Gaussian Splatting) by employing alpha-transparent training while explicitly distinguishing foreground occlusions from background subtraction. We validate the full pipeline through extensive experiments, demonstrating its effectiveness across diverse objects. By eliminating the need for manual intervention or environment modifications, our pipeline can be integrated directly into existing pick-and-place setups, enabling scalable and efficient dataset creation. 

**Abstract (ZH)**: 从现实世界感知模拟物体动力学在数字孪生和机器人操作中显示出巨大前景，但往往需要大量的劳动密集型测量和专业知识。我们提出了一种完全自动化的Real2Sim管道，通过机器人交互生成可用于现实世界物体的仿真资产。仅使用机器人关节扭矩传感器和外部摄像头，该管道识别视觉几何、碰撞几何以及惯性参数等物理属性。我们的方法通过采用α透明训练，结合前景遮挡与背景减法明确区分，引入了一种从光度重构技术（例如，NeRF、Gaussian Splatting）中高效提取高质量、以物体为中心的网格的通用方法。我们通过广泛的实验验证了整个管道的有效性，展示了其在多种物体上的适用性。通过消除人工干预或环境修改的需要，该管道可以直接集成到现有的抓取和放置设置中，实现可扩展且高效的数据集创建。 

---
# Legged Robot State Estimation Using Invariant Neural-Augmented Kalman Filter with a Neural Compensator 

**Title (ZH)**: 基于神经补偿器的不变神经增强卡尔曼滤波的腿足机器人状态估计 

**Authors**: Seokju Lee, Hyun-Bin Kim, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.00344)  

**Abstract**: This paper presents an algorithm to improve state estimation for legged robots. Among existing model-based state estimation methods for legged robots, the contact-aided invariant extended Kalman filter defines the state on a Lie group to preserve invariance, thereby significantly accelerating convergence. It achieves more accurate state estimation by leveraging contact information as measurements for the update step. However, when the model exhibits strong nonlinearity, the estimation accuracy decreases. Such nonlinearities can cause initial errors to accumulate and lead to large drifts over time. To address this issue, we propose compensating for errors by augmenting the Kalman filter with an artificial neural network serving as a nonlinear function approximator. Furthermore, we design this neural network to respect the Lie group structure to ensure invariance, resulting in our proposed Invariant Neural-Augmented Kalman Filter (InNKF). The proposed algorithm offers improved state estimation performance by combining the strengths of model-based and learning-based approaches. Supplementary Video: this https URL 

**Abstract (ZH)**: 基于接触辅助的李群不变广义Kalman滤波器改进腿式机器人的状态估计算法 

---
# Fast Visuomotor Policies via Partial Denoising 

**Title (ZH)**: 部分去噪实现快速视动策略 

**Authors**: Haojun Chen, Minghao Liu, Xiaojian Ma, Zailin Ma, Huimin Wu, Chengdong Ma, Yuanpei Chen, Yifan Zhong, Mingzhi Wang, Qing Li, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00339)  

**Abstract**: Diffusion policies are widely adopted in complex visuomotor tasks for their ability to capture multimodal action distributions. However, the multiple sampling steps required for action generation significantly harm real-time inference efficiency, which limits their applicability in long-horizon tasks and real-time decision-making scenarios. Existing acceleration techniques reduce sampling steps by approximating the original denoising process but inevitably introduce unacceptable performance loss. Here we propose Falcon, which mitigates this trade-off and achieves further acceleration. The core insight is that visuomotor tasks exhibit sequential dependencies between actions at consecutive time steps. Falcon leverages this property to avoid denoising from a standard normal distribution at each decision step. Instead, it starts denoising from partial denoised actions derived from historical information to significantly reduce the denoising steps while incorporating current observations to achieve performance-preserving acceleration of action generation. Importantly, Falcon is a training-free algorithm that can be applied as a plug-in to further improve decision efficiency on top of existing acceleration techniques. We validated Falcon in 46 simulated environments, demonstrating a 2-7x speedup with negligible performance degradation, offering a promising direction for efficient visuomotor policy design. 

**Abstract (ZH)**: Falcon：通过利用动作序列依赖性实现高效的视觉-运动策略设计 

---
# Peek into the `White-Box': A Field Study on Bystander Engagement with Urban Robot Uncertainty 

**Title (ZH)**: 窥视“白盒”：一项关于路人参与城市机器人不确定性交互的实地研究 

**Authors**: Xinyan Yu, Marius Hoggenmueller, Tram Thi Minh Tran, Yiyuan Wang, Qiuming Zhang, Martin Tomitsch  

**Link**: [PDF](https://arxiv.org/pdf/2503.00337)  

**Abstract**: Uncertainty inherently exists in the autonomous decision-making process of robots. Involving humans in resolving this uncertainty not only helps robots mitigate it but is also crucial for improving human-robot interactions. However, in public urban spaces filled with unpredictability, robots often face heightened uncertainty without direct human collaborators. This study investigates how robots can engage bystanders for assistance in public spaces when encountering uncertainty and examines how these interactions impact bystanders' perceptions and attitudes towards robots. We designed and tested a speculative `peephole' concept that engages bystanders in resolving urban robot uncertainty. Our design is guided by considerations of non-intrusiveness and eliciting initiative in an implicit manner, considering bystanders' unique role as non-obligated participants in relation to urban robots. Drawing from field study findings, we highlight the potential of involving bystanders to mitigate urban robots' technological imperfections to both address operational challenges and foster public acceptance of urban robots. Furthermore, we offer design implications to encourage bystanders' involvement in mitigating the imperfections. 

**Abstract (ZH)**: 不确定性固存在机器人自主决策过程中。将人类纳入解决这一不确定性不仅有助于机器人减轻这种不确定性，也是改善人机交互的关键。然而，在充满不可预测性的公共城市空间中，机器人在缺乏直接人类合作者的情况下往往面临更高的不确定性。本研究探讨了机器人在公共空间遇到不确定性时如何寻求旁观者协助，并考察了这些互动如何影响旁观者对机器人的感知和态度。我们设计并测试了一个 speculative 的“窥视孔”概念，该概念旨在让旁观者参与到解决城市机器人不确定性中。我们的设计考虑了非侵入性和以隐式方式激发主动性的因素，考虑到旁观者在与城市机器人关系中作为非强制性参与者所扮演的独特角色。基于实地研究的结果，我们强调了让旁观者参与解决城市机器人技术缺陷的潜力，以应对运营挑战并促进公众对城市机器人的接受度。此外，我们提供了设计建议，以鼓励旁观者参与解决这些缺陷。 

---
# Towards Passive Safe Reinforcement Learning: A Comparative Study on Contact-rich Robotic Manipulation 

**Title (ZH)**: 面向被动安全强化学习：接触丰富的机器人操纵比较研究 

**Authors**: Heng Zhang, Gokhan Solak, Sebastian Hjorth, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00287)  

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in various robotic tasks; however, its deployment in real-world scenarios, particularly in contact-rich environments, often overlooks critical safety and stability aspects. Policies without passivity guarantees can result in system instability, posing risks to robots, their environments, and human operators. In this work, we investigate the limitations of traditional RL policies when deployed in contact-rich tasks and explore the combination of energy-based passive control with safe RL in both training and deployment to answer these challenges. Firstly, we introduce energy-based constraints in our safe RL formulation to train \textit{passivity-aware} RL agents. Secondly, we add a passivity filter on the agent output for \textit{passivity-ensured} control during deployment. We conduct comparative studies on a contact-rich robotic maze exploration task, evaluating the effects of learning passivity-aware policies and the importance of passivity-ensured control. The experiments demonstrate that a passivity-agnostic RL policy easily violates energy constraints in deployment, even though it achieves high task completion in training. The results show that our proposed approach guarantees control stability through passivity filtering and improves the energy efficiency through passivity-aware training. A video of real-world experiments is available as supplementary material. We also release the checkpoint model and offline data for pre-training at \href{this https URL}{Hugging Face} 

**Abstract (ZH)**: 强化学习（RL）在各种机器人任务中取得了显著成功；然而，在实际应用场景中，特别是在接触密集环境中，其部署往往忽视了关键的安全与稳定性方面。没有保证被动性的策略可能导致系统不稳定，对机器人、其环境和人类操作员构成风险。在本工作中，我们探讨了传统RL策略在接触密集任务中的局限性，并探索了在训练和部署中将基于能量的被动控制与安全RL相结合的方法，以应对这些挑战。首先，我们在安全RL框架中引入能量约束以训练具备被动性的RL代理。其次，我们在部署时为代理输出添加被动性滤波器以确保被动性。我们在接触密集的机器人迷宫探索任务中进行对比研究，评估学习被动性意识策略和被动性确保控制的重要性。实验结果表明，不具备被动性的RL策略在部署时容易违反能量约束，尽管其在训练中能够实现高任务完成率。结果表明，我们提出的方法通过被动性滤波确保了控制的稳定性，并通过被动性意识训练提高了能源效率。附有现实世界实验的视频作为补充材料。我们也公布了检查点模型和离线数据以供预训练，在Hugging Face（此 https URL）可下载。 

---
# Human-Robot Collaboration: A Non-Verbal Approach with the NAO Humanoid Robot 

**Title (ZH)**: 人类与机器人协作：基于NAO人形机器人的非言语方法 

**Authors**: Maaz Qureshi, Kerstin Dautenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2503.00284)  

**Abstract**: Humanoid robots, particularly NAO, are gaining prominence for their potential to revolutionize human-robot collaboration, especially in domestic settings like kitchens. Leveraging the advantages of NAO, this research explores non-verbal communications role in enhancing human-robot interaction during meal preparation tasks. By employing gestures, body movements, and visual cues, NAO provides feedback to users, improving comprehension and safety. Our study investigates user perceptions of NAO feedback and its anthropomorphic attributes. Findings suggest that combining various non-verbal cues enhances communication effectiveness, although achieving full anthropomorphic likeness remains a challenge. Insights from this research inform the design of future robotic systems for improved human-robot collaboration. 

**Abstract (ZH)**: 类人机器人，尤其是NAO，正因其在革命人类与机器人协作方面（特别是在厨房等家庭环境中）的潜力而日益受到关注。利用NAO的优势，本研究探讨了非言语沟通在提升烹饪任务中的人机交互中的作用。通过使用手势、身体动作和视觉提示，NAO向用户提供反馈，从而提高理解和安全性。本研究考察了用户对NAO反馈及其类人属性的感知。研究发现，结合多种非言语提示可以增强沟通效果，尽管达到完全类人形态仍面临挑战。本研究的见解为设计未来的人机协作机器人系统提供了指导。 

---
# Xpress: A System For Dynamic, Context-Aware Robot Facial Expressions using Language Models 

**Title (ZH)**: Xpress：一种基于语言模型的动态、上下文感知机器人面部表情系统 

**Authors**: Victor Nikhil Antony, Maia Stiber, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00283)  

**Abstract**: Facial expressions are vital in human communication and significantly influence outcomes in human-robot interaction (HRI), such as likeability, trust, and companionship. However, current methods for generating robotic facial expressions are often labor-intensive, lack adaptability across contexts and platforms, and have limited expressive ranges--leading to repetitive behaviors that reduce interaction quality, particularly in long-term scenarios. We introduce Xpress, a system that leverages language models (LMs) to dynamically generate context-aware facial expressions for robots through a three-phase process: encoding temporal flow, conditioning expressions on context, and generating facial expression code. We demonstrated Xpress as a proof-of-concept through two user studies (n=15x2) and a case study with children and parents (n=13), in storytelling and conversational scenarios to assess the system's context-awareness, expressiveness, and dynamism. Results demonstrate Xpress's ability to dynamically produce expressive and contextually appropriate facial expressions, highlighting its versatility and potential in HRI applications. 

**Abstract (ZH)**: 面部表情在人类沟通中至关重要，并且显著影响人机交互（HRI）的结果，如好感度、信任度和陪伴感。然而，当前生成机器人面部表情的方法往往劳动密集、缺乏跨情境和平台的适应性，并且表情表达范围有限——导致重复行为，降低了交互质量，尤其是在长期场景中。我们介绍了Xpress系统，该系统利用语言模型（LMs）通过三阶段过程动态生成情境相关的面部表情：编码时间流动、依据情境条件面部表情、生成面部表情编码。我们通过两个用户研究（n=15x2）和一个关于儿童和家长的案例研究（n=13）展示了Xpress系统在故事讲述和对话场景中的情境意识、表达能力和动态性。研究结果表明，Xpress能够动态生成恰当且富有表现力的面部表情，突显其在HRI应用中的 versatility 和潜力。 

---
# Maintaining Plasticity in Reinforcement Learning: A Cost-Aware Framework for Aerial Robot Control in Non-stationary Environments 

**Title (ZH)**: 在非稳定环境中基于成本意识的无人机控制：强化学习中的塑性维持框架 

**Authors**: Ali Tahir Karasahin, Ziniu Wu, Basaran Bahadir Kocer  

**Link**: [PDF](https://arxiv.org/pdf/2503.00282)  

**Abstract**: Reinforcement learning (RL) has demonstrated the ability to maintain the plasticity of the policy throughout short-term training in aerial robot control. However, these policies have been shown to loss of plasticity when extended to long-term learning in non-stationary environments. For example, the standard proximal policy optimization (PPO) policy is observed to collapse in long-term training settings and lead to significant control performance degradation. To address this problem, this work proposes a cost-aware framework that uses a retrospective cost mechanism (RECOM) to balance rewards and losses in RL training with a non-stationary environment. Using a cost gradient relation between rewards and losses, our framework dynamically updates the learning rate to actively train the control policy in a disturbed wind environment. Our experimental results show that our framework learned a policy for the hovering task without policy collapse in variable wind conditions and has a successful result of 11.29% less dormant units than L2 regularization with PPO. 

**Abstract (ZH)**: 强化学习在空中机器人控制中的短期训练中展示了保持策略弹性的能力，但在非stationary环境中进行长期学习时，这些策略显示出弹性的损失。例如，标准的近端策略优化（PPO）策略在长期训练环境中观察到崩溃并导致控制性能显著下降。为了解决这一问题，本文提出了一种成本感知框架，该框架使用回顾性成本机制（RECOM）在非stationary环境中平衡RL训练中的奖励和损失。通过奖励和损失之间的成本梯度关系，我们的框架动态更新学习率，以在受扰动风环境中有活性地训练控制策略。实验结果表明，我们的框架在变风条件下学习悬停任务的策略，未出现策略崩溃，并且与PPO和L2正则化相比，活跃单元比例减少了11.29%。 

---
# CRADMap: Applied Distributed Volumetric Mapping with 5G-Connected Multi-Robots and 4D Radar Sensing 

**Title (ZH)**: CRADMap: 应用5G连接多机器人与4D雷达感测的分布式体积映射技术 

**Authors**: Maaz Qureshi, Alexander Werner, Zhenan Liu, Amir Khajepour, George Shaker, William Melek  

**Link**: [PDF](https://arxiv.org/pdf/2503.00262)  

**Abstract**: Sparse and feature SLAM methods provide robust camera pose estimation. However, they often fail to capture the level of detail required for inspection and scene awareness tasks. Conversely, dense SLAM approaches generate richer scene reconstructions but impose a prohibitive computational load to create 3D maps. We present a novel distributed volumetric mapping framework designated as CRADMap that addresses these issues by extending the state-of-the-art (SOTA) ORBSLAM3 [1] system with the COVINS [2] on the backend for global optimization. Our pipeline for volumetric reconstruction fuses dense keyframes at a centralized server via 5G connectivity, aggregating geometry, and occupancy information from multiple autonomous mobile robots (AMRs) without overtaxing onboard resources. This enables each AMR to independently perform mapping while the backend constructs high-fidelity 3D maps in real time. To overcome the limitation of standard visual nodes we automate a 4D mmWave radar, standalone from CRADMap, to test its capabilities for making extra maps of the hidden metallic object(s) in a cluttered environment. Experimental results Section-IV confirm that our framework yields globally consistent volumetric reconstructions and seamlessly supports applied distributed mapping in complex indoor environments. 

**Abstract (ZH)**: 稀疏特征SLAM方法提供了稳健的相机姿态估计，但往往无法捕捉到检验和场景感知任务所需的细节水平。相反，密集SLAM方法生成更为丰富的场景重建，但会带来创建3D地图的计算负担。我们提出了一种名为CRADMap的新型分布式体积映射框架，通过将最先进的ORBSLAM3系统与COVINS全局优化模块扩展结合，解决了这些问题。我们的体积重建管道利用5G连接在中央服务器上融合密集关键帧，汇总来自多个自主移动机器人（AMRs）的几何和占用信息，而不会过度占用其车载资源。这使得每个AMR能够独立进行测绘，而后端可以实现实时构建高保真的3D地图。为了克服标准视觉节点的限制，我们自动化了一种独立于CRADMap的4D毫米波雷达，以测试其为杂乱环境中的隐藏金属物体生成额外地图的能力。实验结果（第四节）证实，我们的框架能够提供全局一致的体积重建，并能无缝支持复杂室内环境中的分布式映射。 

---
# Survival of the fastest -- algorithm-guided evolution of light-powered underwater microrobots 

**Title (ZH)**: Survival of the最快者——算法引导下的水下微机器人光动力进化 

**Authors**: Mikołaj Rogóż, Zofia Dziekan, Piotr Wasylczyk  

**Link**: [PDF](https://arxiv.org/pdf/2503.00204)  

**Abstract**: Depending on environmental conditions, lightweight soft robots can exhibit various modes of locomotion that are difficult to model. As a result, optimizing their performance is complex, especially in small-scale systems characterized by low Reynolds numbers, when multiple aero- and hydrodynamical processes influence their movement. In this work, we study underwater swimmer locomotion by applying experimental results as the fitness function in two evolutionary algorithms: particle swarm optimization and genetic algorithm. Since soft, light-powered robots with different characteristics (phenotypes) can be fabricated quickly, they provide a great platform for optimisation experiments, using physical robots competing to improve swimming speed over consecutive generations. Interestingly, just like in natural evolution, unexpected gene combinations led to surprisingly good results, including several hundred percent increase in speed or the discovery of a self-oscillating underwater locomotion mode. 

**Abstract (ZH)**: 轻质软体机器人的水下游泳运动优化研究 

---
# ProDapt: Proprioceptive Adaptation using Long-term Memory Diffusion 

**Title (ZH)**: ProDapt: 使用长期记忆扩散的本体感知适应 

**Authors**: Federico Pizarro Bejarano, Bryson Jones, Daniel Pastor Moreno, Joseph Bowkett, Paul G. Backes, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2503.00193)  

**Abstract**: Diffusion models have revolutionized imitation learning, allowing robots to replicate complex behaviours. However, diffusion often relies on cameras and other exteroceptive sensors to observe the environment and lacks long-term memory. In space, military, and underwater applications, robots must be highly robust to failures in exteroceptive sensors, operating using only proprioceptive information. In this paper, we propose ProDapt, a method of incorporating long-term memory of previous contacts between the robot and the environment in the diffusion process, allowing it to complete tasks using only proprioceptive data. This is achieved by identifying "keypoints", essential past observations maintained as inputs to the policy. We test our approach using a UR10e robotic arm in both simulation and real experiments and demonstrate the necessity of this long-term memory for task completion. 

**Abstract (ZH)**: 基于长期记忆的扩散模型在仅使用本体感受信息完成任务中的应用 

---
# Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees 

**Title (ZH)**: 基于视觉的神经网络控制器的学习与半概率安全保证 

**Authors**: Xinhang Ma, Junlin Wu, Hussein Sibai, Yiannis Kantaros, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2503.00191)  

**Abstract**: Ensuring safety in autonomous systems with vision-based control remains a critical challenge due to the high dimensionality of image inputs and the fact that the relationship between true system state and its visual manifestation is unknown. Existing methods for learning-based control in such settings typically lack formal safety guarantees. To address this challenge, we introduce a novel semi-probabilistic verification framework that integrates reachability analysis with conditional generative adversarial networks and distribution-free tail bounds to enable efficient and scalable verification of vision-based neural network controllers. Next, we develop a gradient-based training approach that employs a novel safety loss function, safety-aware data-sampling strategy to efficiently select and store critical training examples, and curriculum learning, to efficiently synthesize safe controllers in the semi-probabilistic framework. Empirical evaluations in X-Plane 11 airplane landing simulation, CARLA-simulated autonomous lane following, and F1Tenth lane following in a physical visually-rich miniature environment demonstrate the effectiveness of our method in achieving formal safety guarantees while maintaining strong nominal performance. Our code is available at this https URL. 

**Abstract (ZH)**: 基于视觉的自主系统安全性保障：一种半概率验证框架及其应用 

---
# Stability Analysis of Deep Reinforcement Learning for Multi-Agent Inspection in a Terrestrial Testbed 

**Title (ZH)**: terrestrial测试床中多agent检查的深度强化学习稳定性分析 

**Authors**: Henry Lei, Zachary S. Lippay, Anonto Zaman, Joshua Aurand, Amin Maghareh, Sean Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2503.00056)  

**Abstract**: The design and deployment of autonomous systems for space missions require robust solutions to navigate strict reliability constraints, extended operational duration, and communication challenges. This study evaluates the stability and performance of a hierarchical deep reinforcement learning (DRL) framework designed for multi-agent satellite inspection tasks. The proposed framework integrates a high-level guidance policy with a low-level motion controller, enabling scalable task allocation and efficient trajectory execution. Experiments conducted on the Local Intelligent Network of Collaborative Satellites (LINCS) testbed assess the framework's performance under varying levels of fidelity, from simulated environments to a cyber-physical testbed. Key metrics, including task completion rate, distance traveled, and fuel consumption, highlight the framework's robustness and adaptability despite real-world uncertainties such as sensor noise, dynamic perturbations, and runtime assurance (RTA) constraints. The results demonstrate that the hierarchical controller effectively bridges the sim-to-real gap, maintaining high task completion rates while adapting to the complexities of real-world environments. These findings validate the framework's potential for enabling autonomous satellite operations in future space missions. 

**Abstract (ZH)**: 空间任务中自主系统的设计与部署需要应对严格可靠性的约束、长时间运行以及通信挑战的稳健解决方案。本研究评估了一种用于多智能体卫星检查任务的分层深度强化学习框架的稳定性和性能。提出的框架将高层指导策略与低层运动控制器相结合，实现可扩展的任务分配和高效的轨迹执行。在本地协作卫星的局部智能网络（LINCS）试验台上，从模拟环境到物理-计算试验台的不同保真度水平下进行了实验，评估了该框架的性能。关键指标，包括任务完成率、行驶距离和燃料消耗，展示了框架在真实世界不确定性如传感器噪声、动态扰动和运行时间保证（RTA）约束下的鲁棒性和适应性。实验结果表明，分层控制器有效地弥合了仿真与现实之间的差距，维持了高任务完成率的同时适应了真实世界环境的复杂性。这些发现验证了该框架在未来空间任务中实现自主卫星操作的潜力。 

---
# Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning 

**Title (ZH)**: 演示增强奖励、策略和世界模型的多阶段操控 

**Authors**: Adrià López Escoriza, Nicklas Hansen, Stone Tao, Tongzhou Mu, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.01837)  

**Abstract**: Long-horizon tasks in robotic manipulation present significant challenges in reinforcement learning (RL) due to the difficulty of designing dense reward functions and effectively exploring the expansive state-action space. However, despite a lack of dense rewards, these tasks often have a multi-stage structure, which can be leveraged to decompose the overall objective into manageable subgoals. In this work, we propose DEMO3, a framework that exploits this structure for efficient learning from visual inputs. Specifically, our approach incorporates multi-stage dense reward learning, a bi-phasic training scheme, and world model learning into a carefully designed demonstration-augmented RL framework that strongly mitigates the challenge of exploration in long-horizon tasks. Our evaluations demonstrate that our method improves data-efficiency by an average of 40% and by 70% on particularly difficult tasks compared to state-of-the-art approaches. We validate this across 16 sparse-reward tasks spanning four domains, including challenging humanoid visual control tasks using as few as five demonstrations. 

**Abstract (ZH)**: 长时_horizon_任务在机器人操作中的强化学习中带来了显著挑战，原因在于密集奖励函数的设计困难以及对广阔状态-动作空间的有效探索。然而，尽管缺乏密集奖励，这些任务通常具有多阶段结构，可以利用这种结构将总体目标分解为可管理的子目标。在本工作中，我们提出了DEMO3框架，该框架利用这种结构从视觉输入中高效地进行学习。具体而言，我们的方法结合了多阶段密集奖励学习、双阶段训练方案和世界模型学习，以精心设计的演示增强 RL 框架，显著减轻了长时_horizon_任务中的探索挑战。我们的评估表明，与最先进的方法相比，我们的方法在数据效率上平均提高了40%，在特别困难的任务上提高了70%。我们在包括最少仅需五次演示的人形视觉控制任务在内的四个领域中的16个稀疏奖励任务中进行了验证。 

---
# Differentiable Information Enhanced Model-Based Reinforcement Learning 

**Title (ZH)**: 可微信息增强模型导向强化学习 

**Authors**: Xiaoyuan Zhang, Xinyan Cai, Bo Liu, Weidong Huang, Song-Chun Zhu, Siyuan Qi, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01178)  

**Abstract**: Differentiable environments have heralded new possibilities for learning control policies by offering rich differentiable information that facilitates gradient-based methods. In comparison to prevailing model-free reinforcement learning approaches, model-based reinforcement learning (MBRL) methods exhibit the potential to effectively harness the power of differentiable information for recovering the underlying physical dynamics. However, this presents two primary challenges: effectively utilizing differentiable information to 1) construct models with more accurate dynamic prediction and 2) enhance the stability of policy training. In this paper, we propose a Differentiable Information Enhanced MBRL method, MB-MIX, to address both challenges. Firstly, we adopt a Sobolev model training approach that penalizes incorrect model gradient outputs, enhancing prediction accuracy and yielding more precise models that faithfully capture system dynamics. Secondly, we introduce mixing lengths of truncated learning windows to reduce the variance in policy gradient estimation, resulting in improved stability during policy learning. To validate the effectiveness of our approach in differentiable environments, we provide theoretical analysis and empirical results. Notably, our approach outperforms previous model-based and model-free methods, in multiple challenging tasks involving controllable rigid robots such as humanoid robots' motion control and deformable object manipulation. 

**Abstract (ZH)**: 不同的环境为基于梯度的方法提供了丰富的可微信息，开启了学习控制策略的新可能性。与当前占主导地位的无模型强化学习方法相比，基于模型的强化学习（MBRL）方法具备利用可微信息恢复潜在物理动态的潜力。然而，这提出了两个主要挑战：有效利用可微信息以1）构建更准确的动力学预测模型，2）提高策略训练的稳定性。本文提出了一种可微信息增强的MBRL方法MB-MIX来解决这两个挑战。首先，我们采用Sobolev模型训练方法惩罚错误的模型梯度输出，提高预测准确性并产生更精确地捕捉系统动力学的模型。其次，我们引入截断学习窗口的混合长度来减少策略梯度估计的方差，从而在策略学习过程中提高稳定性。为了验证我们在可微环境中方法的有效性，我们提供了理论分析和实证结果。值得注意的是，与先前的基于模型和无模型方法相比，我们的方法在涉及可控刚体（如类人机器人动作控制和可变形物体操作）的多个具有挑战性的任务中表现更优。 

---
# One-Shot Affordance Grounding of Deformable Objects in Egocentric Organizing Scenes 

**Title (ZH)**: 一手构建主观 organizing 场景中变形物体的功能接地 

**Authors**: Wanjun Jia, Fan Yang, Mengfei Duan, Xianchi Chen, Yinxi Wang, Yiming Jiang, Wenrui Chen, Kailun Yang, Zhiyong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.01092)  

**Abstract**: Deformable object manipulation in robotics presents significant challenges due to uncertainties in component properties, diverse configurations, visual interference, and ambiguous prompts. These factors complicate both perception and control tasks. To address these challenges, we propose a novel method for One-Shot Affordance Grounding of Deformable Objects (OS-AGDO) in egocentric organizing scenes, enabling robots to recognize previously unseen deformable objects with varying colors and shapes using minimal samples. Specifically, we first introduce the Deformable Object Semantic Enhancement Module (DefoSEM), which enhances hierarchical understanding of the internal structure and improves the ability to accurately identify local features, even under conditions of weak component information. Next, we propose the ORB-Enhanced Keypoint Fusion Module (OEKFM), which optimizes feature extraction of key components by leveraging geometric constraints and improves adaptability to diversity and visual interference. Additionally, we propose an instance-conditional prompt based on image data and task context, effectively mitigates the issue of region ambiguity caused by prompt words. To validate these methods, we construct a diverse real-world dataset, AGDDO15, which includes 15 common types of deformable objects and their associated organizational actions. Experimental results demonstrate that our approach significantly outperforms state-of-the-art methods, achieving improvements of 6.2%, 3.2%, and 2.9% in KLD, SIM, and NSS metrics, respectively, while exhibiting high generalization performance. Source code and benchmark dataset will be publicly available at this https URL. 

**Abstract (ZH)**: 面向自我中心整理场景的可变形物体一次性gradable用途 grounding 方法（OS-AGDO） 

---
# SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models 

**Title (ZH)**: SENSEI: 由基础模型引导的语义探索以学习多样化的世界模型 

**Authors**: Cansu Sancaktar, Christian Gumbsch, Andrii Zadaianchuk, Pavel Kolev, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2503.01584)  

**Abstract**: Exploration is a cornerstone of reinforcement learning (RL). Intrinsic motivation attempts to decouple exploration from external, task-based rewards. However, established approaches to intrinsic motivation that follow general principles such as information gain, often only uncover low-level interactions. In contrast, children's play suggests that they engage in meaningful high-level behavior by imitating or interacting with their caregivers. Recent work has focused on using foundation models to inject these semantic biases into exploration. However, these methods often rely on unrealistic assumptions, such as language-embedded environments or access to high-level actions. We propose SEmaNtically Sensible ExploratIon (SENSEI), a framework to equip model-based RL agents with an intrinsic motivation for semantically meaningful behavior. SENSEI distills a reward signal of interestingness from Vision Language Model (VLM) annotations, enabling an agent to predict these rewards through a world model. Using model-based RL, SENSEI trains an exploration policy that jointly maximizes semantic rewards and uncertainty. We show that in both robotic and video game-like simulations SENSEI discovers a variety of meaningful behaviors from image observations and low-level actions. SENSEI provides a general tool for learning from foundation model feedback, a crucial research direction, as VLMs become more powerful. 

**Abstract (ZH)**: 基于语义的探索：一种模型驱动的强化学习框架(SENSEI) 

---
# Multi-Agent Reinforcement Learning with Long-Term Performance Objectives for Service Workforce Optimization 

**Title (ZH)**: 面向服务劳动力优化的长期性能目标多智能体强化学习 

**Authors**: Kareem Eissa, Rayal Prasad, Sarith Mohan, Ankur Kapoor, Dorin Comaniciu, Vivek Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.01069)  

**Abstract**: Workforce optimization plays a crucial role in efficient organizational operations where decision-making may span several different administrative and time scales. For instance, dispatching personnel to immediate service requests while managing talent acquisition with various expertise sets up a highly dynamic optimization problem. Existing work focuses on specific sub-problems such as resource allocation and facility location, which are solved with heuristics like local-search and, more recently, deep reinforcement learning. However, these may not accurately represent real-world scenarios where such sub-problems are not fully independent. Our aim is to fill this gap by creating a simulator that models a unified workforce optimization problem. Specifically, we designed a modular simulator to support the development of reinforcement learning methods for integrated workforce optimization problems. We focus on three interdependent aspects: personnel dispatch, workforce management, and personnel positioning. The simulator provides configurable parameterizations to help explore dynamic scenarios with varying levels of stochasticity and non-stationarity. To facilitate benchmarking and ablation studies, we also include heuristic and RL baselines for the above mentioned aspects. 

**Abstract (ZH)**: 劳动力优化在高效组织运行中扮演着 crucial 角色，决策可能涉及多个不同层级的管理和时间尺度。例如，将人员调度到即时服务请求的同时管理具有各种专业技能的人才招聘，构成了一个高度动态的优化问题。现有研究集中在资源分配和设施位置等具体子问题上，这些问题通过局部搜索等启发式方法或近期的深度强化学习进行解决。然而，这些方法可能无法准确反映现实世界的场景，其中这些子问题不是完全独立的。我们的目的是通过创建一个模拟器来填补这一空白，以建模统一的劳动力优化问题。具体来说，我们设计了一个模块化模拟器，支持集成劳动力优化问题的强化学习方法的发展。我们重点关注三个方面：人员调度、劳动力管理以及人员定位。模拟器提供可配置的参数化设置，以帮助探索不同程度随机性和非平稳性的动态场景。为了便于基准测试和消除影响研究，我们还为上述方面包括了启发式和RL baseline。 

---
# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains 

**Title (ZH)**: NeSyC：面向开放域复杂具身任务的神经符号连续学习者 

**Authors**: Wonje Choi, Jinwoo Park, Sanghyun Ahn, Daehee Lee, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00870)  

**Abstract**: We explore neuro-symbolic approaches to generalize actionable knowledge, enabling embodied agents to tackle complex tasks more effectively in open-domain environments. A key challenge for embodied agents is the generalization of knowledge across diverse environments and situations, as limited experiences often confine them to their prior knowledge. To address this issue, we introduce a novel framework, NeSyC, a neuro-symbolic continual learner that emulates the hypothetico-deductive model by continually formulating and validating knowledge from limited experiences through the combined use of Large Language Models (LLMs) and symbolic tools. Specifically, we devise a contrastive generality improvement scheme within NeSyC, which iteratively generates hypotheses using LLMs and conducts contrastive validation via symbolic tools. This scheme reinforces the justification for admissible actions while minimizing the inference of inadmissible ones. Additionally, we incorporate a memory-based monitoring scheme that efficiently detects action errors and triggers the knowledge refinement process across domains. Experiments conducted on diverse embodied task benchmarks-including ALFWorld, VirtualHome, Minecraft, RLBench, and a real-world robotic scenario-demonstrate that NeSyC is highly effective in solving complex embodied tasks across a range of open-domain environments. 

**Abstract (ZH)**: 我们探索神经符号方法以推广可执行的知识，使具身代理能够在开放域环境中更有效地应对复杂任务。NeSyC：一种神经符号连续学习框架通过结合大规模语言模型和符号工具来模拟假设演绎模型，不断从有限经验中形成和验证知识。 

---
# Modeling Arbitrarily Applicable Relational Responding with the Non-Axiomatic Reasoning System: A Machine Psychology Approach 

**Title (ZH)**: 基于非公理化推理系统的情感泛化响应建模：一种机器心理学方法 

**Authors**: Robert Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2503.00611)  

**Abstract**: Arbitrarily Applicable Relational Responding (AARR) is a cornerstone of human language and reasoning, referring to the learned ability to relate symbols in flexible, context-dependent ways. In this paper, we present a novel theoretical approach for modeling AARR within an artificial intelligence framework using the Non-Axiomatic Reasoning System (NARS). NARS is an adaptive reasoning system designed for learning under uncertainty. By integrating principles from Relational Frame Theory - the behavioral psychology account of AARR - with the reasoning mechanisms of NARS, we conceptually demonstrate how key properties of AARR (mutual entailment, combinatorial entailment, and transformation of stimulus functions) can emerge from the inference rules and memory structures of NARS. Two theoretical experiments illustrate this approach: one modeling stimulus equivalence and transfer of function, and another modeling complex relational networks involving opposition frames. In both cases, the system logically demonstrates the derivation of untrained relations and context-sensitive transformations of stimulus significance, mirroring established human cognitive phenomena. These results suggest that AARR - long considered uniquely human - can be conceptually captured by suitably designed AI systems, highlighting the value of integrating behavioral science insights into artificial general intelligence (AGI) research. 

**Abstract (ZH)**: 任意适用的关系反应(AARR)是人类语言和推理的基石，指的是以灵活且依赖上下文的方式关联符号的能力。本文提出了一种新的理论方法，利用非公理推理系统(NARS)在人工智能框架中建模AARR。NARS是一种适应性推理系统，旨在在不确定性条件下进行学习。通过将关系框架理论——对AARR的行为心理学解释——的原则与NARS的推理机制相结合，我们概念性地展示了AARR的关键属性（相互蕴含、组合蕴含和刺激功能的转换）如何从NARS的推理法则和记忆结构中出现。两个理论实验说明了这一方法：一个模型刺激等价性和功能的转移，另一个模型涉及对立框架的复杂关系网络。在两种情况下，系统逻辑地展示了未训练关系及其上下文敏感的刺激意义转换的推导，这与已确立的人类认知现象相呼应。这些结果表明，AARR——长期被认为是人类独有的——可以通过适当设计的AI系统概念性地捕获，强调将行为科学洞察整合到通用人工智能(AGI)研究中的价值。 

---
# Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning 

**Title (ZH)**: 对抗代理：基于强化学习的黑盒逃逸攻击 

**Authors**: Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel  

**Link**: [PDF](https://arxiv.org/pdf/2503.01734)  

**Abstract**: Reinforcement learning (RL) offers powerful techniques for solving complex sequential decision-making tasks from experience. In this paper, we demonstrate how RL can be applied to adversarial machine learning (AML) to develop a new class of attacks that learn to generate adversarial examples: inputs designed to fool machine learning models. Unlike traditional AML methods that craft adversarial examples independently, our RL-based approach retains and exploits past attack experience to improve future attacks. We formulate adversarial example generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On CIFAR-10, our agent increases the success rate of adversarial examples by 19.4% and decreases the median number of victim model queries per adversarial example by 53.2% from the start to the end of training. In a head-to-head comparison with a state-of-the-art image attack, SquareAttack, our approach enables an adversary to generate adversarial examples with 13.1% more success after 5000 episodes of training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to attack ML models efficiently and at scale. 

**Abstract (ZH)**: 基于强化学习的对抗机器学习中新颖攻击方法的研究 

---
# Eau De $Q$-Network: Adaptive Distillation of Neural Networks in Deep Reinforcement Learning 

**Title (ZH)**: $Q$-网络之水：深度强化学习中神经网络的自适应_distillation 

**Authors**: Théo Vincent, Tim Faust, Yogesh Tripathi, Jan Peters, Carlo D'Eramo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01437)  

**Abstract**: Recent works have successfully demonstrated that sparse deep reinforcement learning agents can be competitive against their dense counterparts. This opens up opportunities for reinforcement learning applications in fields where inference time and memory requirements are cost-sensitive or limited by hardware. Until now, dense-to-sparse methods have relied on hand-designed sparsity schedules that are not synchronized with the agent's learning pace. Crucially, the final sparsity level is chosen as a hyperparameter, which requires careful tuning as setting it too high might lead to poor performances. In this work, we address these shortcomings by crafting a dense-to-sparse algorithm that we name Eau De $Q$-Network (EauDeQN). To increase sparsity at the agent's learning pace, we consider multiple online networks with different sparsity levels, where each online network is trained from a shared target network. At each target update, the online network with the smallest loss is chosen as the next target network, while the other networks are replaced by a pruned version of the chosen network. We evaluate the proposed approach on the Atari $2600$ benchmark and the MuJoCo physics simulator, showing that EauDeQN reaches high sparsity levels while keeping performances high. 

**Abstract (ZH)**: 最近的研究成功展示了稀疏深度强化学习代理可以与其密集 counterpart 对抗，这为在对推理时间和内存要求敏感或受硬件限制的领域应用强化学习打开了机会。直到现在，从密集到稀疏的方法依赖于与代理学习节奏不同步的手动设计稀疏计划，最关键的是，最终的稀疏水平被选作超参数，需要仔细调整，因为设置得太高可能会导致性能不佳。在这项工作中，我们通过设计一种名为Eau De $Q$-Network（EauDeQN）的从密集到稀疏算法来解决这些不足。为了在代理学习节奏下增加稀疏性，我们考虑了具有不同稀疏水平的多个在线网络，并且每个在线网络都从共享的目标网络中训练。在每次目标更新时，选择损失最小的在线网络作为下一个目标网络，而其他网络则被选择网络的剪枝版本替换。我们在Atari $2600$基准和MuJoCo物理模拟器上评估了所提出的方法，结果显示EauDeQN 达到了高稀疏水平同时保持了高性能。 

---
# Learning Actionable World Models for Industrial Process Control 

**Title (ZH)**: 工业过程控制中的可操作世界模型学习 

**Authors**: Peng Yan, Ahmed Abdulkadir, Gerrit A. Schatte, Giulia Anguzzi, Joonsu Gha, Nikola Pascher, Matthias Rosenthal, Yunlong Gao, Benjamin F. Grewe, Thilo Stadelmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.01411)  

**Abstract**: To go from (passive) process monitoring to active process control, an effective AI system must learn about the behavior of the complex system from very limited training data, forming an ad-hoc digital twin with respect to process in- and outputs that captures the consequences of actions on the process's world. We propose a novel methodology based on learning world models that disentangles process parameters in the learned latent representation, allowing for fine-grained control. Representation learning is driven by the latent factors that influence the processes through contrastive learning within a joint embedding predictive architecture. This makes changes in representations predictable from changes in inputs and vice versa, facilitating interpretability of key factors responsible for process variations, paving the way for effective control actions to keep the process within operational bounds. The effectiveness of our method is validated on the example of plastic injection molding, demonstrating practical relevance in proposing specific control actions for a notoriously unstable process. 

**Abstract (ZH)**: 从被动过程监控到主动过程控制，一种有效的AI系统必须从有限的训练数据中学习复杂系统的行为，形成一个针对过程输入和输出的临时数字孪生，捕获行动对过程世界的影响。我们提出了一种基于学习世界模型的新方法，该方法在学习到的潜在表示中解耦过程参数，从而实现精细控制。表示学习通过联合嵌入预测架构内的对比学习驱动，使输入变化对表示变化的影响可预测，反之亦然，这促进了关键因素的可解释性，这些因素负责过程变化，从而为保持过程在运行界限内提供了有效的控制措施。该方法的有效性通过塑料注射成型的示例得到验证，展示了在这一 notoriously 不稳定的工艺中提出具体控制措施的实用意义。 

---
# Behavior Preference Regression for Offline Reinforcement Learning 

**Title (ZH)**: 离线强化学习中的行为偏好回归 

**Authors**: Padmanaba Srinivasan, William Knottenbelt  

**Link**: [PDF](https://arxiv.org/pdf/2503.00930)  

**Abstract**: Offline reinforcement learning (RL) methods aim to learn optimal policies with access only to trajectories in a fixed dataset. Policy constraint methods formulate policy learning as an optimization problem that balances maximizing reward with minimizing deviation from the behavior policy. Closed form solutions to this problem can be derived as weighted behavioral cloning objectives that, in theory, must compute an intractable partition function. Reinforcement learning has gained popularity in language modeling to align models with human preferences; some recent works consider paired completions that are ranked by a preference model following which the likelihood of the preferred completion is directly increased. We adapt this approach of paired comparison. By reformulating the paired-sample optimization problem, we fit the maximum-mode of the Q function while maximizing behavioral consistency of policy actions. This yields our algorithm, Behavior Preference Regression for offline RL (BPR). We empirically evaluate BPR on the widely used D4RL Locomotion and Antmaze datasets, as well as the more challenging V-D4RL suite, which operates in image-based state spaces. BPR demonstrates state-of-the-art performance over all domains. Our on-policy experiments suggest that BPR takes advantage of the stability of on-policy value functions with minimal perceptible performance degradation on Locomotion datasets. 

**Abstract (ZH)**: 离线强化学习（RL）方法旨在通过访问固定数据集中的轨迹来学习最优策略。政策约束方法将政策学习形式化为一个优化问题，该问题平衡了最大化奖励与最小化偏离行为策略之间的偏差。这个问题的闭式解可以作为加权行为克隆目标函数导出，在理论上必须计算一个难处理的分区函数。强化学习在语言建模中获得了 popularity 以使模型与人类偏好对齐；一些近期的工作考虑了由偏好模型排名的成对完成，并直接增加了优选完成的似然性。我们采用这种成对比较的方法。通过重新形式化成对样本优化问题，我们拟合了Q函数的最大模式，同时最大化了政策行动的行为一致性。这一过程产生了我们的算法——行为偏好回归用于离线RL（BPR）。我们在广泛使用的D4RL运动和Antmaze数据集中以及更具挑战性的基于图像状态空间的V-D4RL套件上进行了实证评估，BPR在所有领域均表现出最先进的性能。我们在线策略实验表明，BPR利用了在线策略价值函数的稳定性，并在运动数据集中几乎没有明显的性能退化。 

---
# Multimodal Distillation-Driven Ensemble Learning for Long-Tailed Histopathology Whole Slide Images Analysis 

**Title (ZH)**: 多模态蒸馏驱动的集成学习在长尾病理全视野图像分析中的应用 

**Authors**: Xitong Ling, Yifeng Ping, Jiawen Li, Jing Peng, Yuxuan Chen, Minxi Ouyang, Yizhi Wang, Yonghong He, Tian Guan, Xiaoping Liu, Lianghui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00915)  

**Abstract**: Multiple Instance Learning (MIL) plays a significant role in computational pathology, enabling weakly supervised analysis of Whole Slide Image (WSI) datasets. The field of WSI analysis is confronted with a severe long-tailed distribution problem, which significantly impacts the performance of classifiers. Long-tailed distributions lead to class imbalance, where some classes have sparse samples while others are abundant, making it difficult for classifiers to accurately identify minority class samples. To address this issue, we propose an ensemble learning method based on MIL, which employs expert decoders with shared aggregators and consistency constraints to learn diverse distributions and reduce the impact of class imbalance on classifier performance. Moreover, we introduce a multimodal distillation framework that leverages text encoders pre-trained on pathology-text pairs to distill knowledge and guide the MIL aggregator in capturing stronger semantic features relevant to class information. To ensure flexibility, we use learnable prompts to guide the distillation process of the pre-trained text encoder, avoiding limitations imposed by specific prompts. Our method, MDE-MIL, integrates multiple expert branches focusing on specific data distributions to address long-tailed issues. Consistency control ensures generalization across classes. Multimodal distillation enhances feature extraction. Experiments on Camelyon+-LT and PANDA-LT datasets show it outperforms state-of-the-art methods. 

**Abstract (ZH)**: 多实例学习（MIL）在计算病理学中发挥着重要作用，使其能够在Whole Slide Image (WSI)数据集的弱监督分析中发挥作用。WSI分析领域面临着严重的长尾分布问题，这对分类器的性能产生了显著影响。长尾分布导致类别不平衡，一些类别样本稀疏而另一些类别样本丰富，使得分类器难以准确识别少数类别样本。为解决这一问题，我们提出了一种基于MIL的ensemble学习方法，该方法使用具有共享聚合器和一致性约束的专家解码器来学习多样化的分布并减少类别不平衡对分类器性能的影响。此外，我们引入了一种多模态蒸馏框架，该框架利用在病理学文本对上预训练的文本编码器，通过蒸馏知识来指导MIL聚合器捕捉与类别信息相关的更强的语义特征。为确保灵活性，我们使用可学习的提示来指导预训练文本编码器的蒸馏过程，避免特定提示的限制。我们的方法MDE-MIL结合了多个专注于特定数据分布的专家分支，以解决长尾问题。一致性控制确保了跨类别的泛化能力。多模态蒸馏增强了特征提取。实验结果表明，MDE-MIL在Camelyon+-LT和PANDA-LT数据集上优于现有最佳方法。 

---
# Brain Foundation Models: A Survey on Advancements in Neural Signal Processing and Brain Discovery 

**Title (ZH)**: 脑基础模型：神经信号处理与脑认知进展综述 

**Authors**: Xinliang Zhou, Chenyu Liu, Zhisheng Chen, Kun Wang, Yi Ding, Ziyu Jia, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00580)  

**Abstract**: Brain foundation models (BFMs) have emerged as a transformative paradigm in computational neuroscience, offering a revolutionary framework for processing diverse neural signals across different brain-related tasks. These models leverage large-scale pre-training techniques, allowing them to generalize effectively across multiple scenarios, tasks, and modalities, thus overcoming the traditional limitations faced by conventional artificial intelligence (AI) approaches in understanding complex brain data. By tapping into the power of pretrained models, BFMs provide a means to process neural data in a more unified manner, enabling advanced analysis and discovery in the field of neuroscience. In this survey, we define BFMs for the first time, providing a clear and concise framework for constructing and utilizing these models in various applications. We also examine the key principles and methodologies for developing these models, shedding light on how they transform the landscape of neural signal processing. This survey presents a comprehensive review of the latest advancements in BFMs, covering the most recent methodological innovations, novel views of application areas, and challenges in the field. Notably, we highlight the future directions and key challenges that need to be addressed to fully realize the potential of BFMs. These challenges include improving the quality of brain data, optimizing model architecture for better generalization, increasing training efficiency, and enhancing the interpretability and robustness of BFMs in real-world applications. 

**Abstract (ZH)**: 基于脑的模型（BFMs）已成为计算神经科学中的一个变革性范式，提供了处理跨不同脑相关任务的多种神经信号的革命性框架。这些模型利用大规模预训练技术，使其能够在多种场景、任务和模态中有效泛化，从而克服了传统人工智能（AI）方法在理解复杂脑数据时面临的局限性。通过利用预训练模型的力量，BFMs提供了一种更统一的方式来处理神经数据，使神经科学领域的高级分析和发现成为了可能。在这篇综述中，我们首次定义了BFMs，提供了一个清晰且简洁的框架，以便在各种应用中构建和利用这些模型。我们还探讨了开发这些模型的关键原则和方法论，揭示了它们如何改变神经信号处理的格局。这篇综述全面回顾了BFMs的最新进展，涵盖了最新的方法论创新、新的应用领域视角以及该领域的挑战。我们特别强调了实现BFMs潜力所需解决的未来方向和关键挑战，包括提高脑数据质量、优化模型架构以获得更好的泛化能力、提高训练效率以及增强BFMs在实际应用中的可解释性和鲁棒性。 

---
# Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions 

**Title (ZH)**: 永不 Too Prim to Swim：一种增强型 RL 辅助自适应 S 表面控制器在极端海况下用于 AUVs 

**Authors**: Guanwen Xie, Jingzehua Xu, Yimian Ding, Zhi Zhang, Shuai Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00527)  

**Abstract**: The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers. 

**Abstract (ZH)**: 自主 underwater 车辆 (AUVs) 的适应性和机动能力在海洋研究中引起了广泛关注，由于无法预测的干扰和 AUV 自由度间的强耦合。在本文中，我们开发了一种增强学习 (LLM-增强 RL) 算法辅助的自适应 S 表面控制器。具体来说，LLM 通过联合优化控制器参数和奖励函数来增强 RL 训练。借助多模态和结构化显式任务反馈，LLM 实现了联合调整、权衡多个目标，并提高任务导向性能和适应性。在所提控制器中，RL 策略专注于高层任务，输出任务导向的高层指令，S 表面控制器将其转换为控制信号，以确保在极端海况下消除非线性效应和不可预测的外部干扰。在复杂地形、海浪和洋流的极端海况下，所提控制器在水下目标跟踪和数据收集等高层任务中表现出优越的性能和适应性，优于传统的 PID 和 SMC 控制器。 

---
# MIRROR: Multi-Modal Pathological Self-Supervised Representation Learning via Modality Alignment and Retention 

**Title (ZH)**: MIRROR：模态对齐与保留的多模态病理自我监督表示学习 

**Authors**: Tianyi Wang, Jianan Fan, Dingxin Zhang, Dongnan Liu, Yong Xia, Heng Huang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.00374)  

**Abstract**: Histopathology and transcriptomics are fundamental modalities in oncology, encapsulating the morphological and molecular aspects of the disease. Multi-modal self-supervised learning has demonstrated remarkable potential in learning pathological representations by integrating diverse data sources. Conventional multi-modal integration methods primarily emphasize modality alignment, while paying insufficient attention to retaining the modality-specific structures. However, unlike conventional scenarios where multi-modal inputs share highly overlapping features, histopathology and transcriptomics exhibit pronounced heterogeneity, offering orthogonal yet complementary insights. Histopathology provides morphological and spatial context, elucidating tissue architecture and cellular topology, whereas transcriptomics delineates molecular signatures through gene expression patterns. This inherent disparity introduces a major challenge in aligning them while maintaining modality-specific fidelity. To address these challenges, we present MIRROR, a novel multi-modal representation learning method designed to foster both modality alignment and retention. MIRROR employs dedicated encoders to extract comprehensive features for each modality, which is further complemented by a modality alignment module to achieve seamless integration between phenotype patterns and molecular profiles. Furthermore, a modality retention module safeguards unique attributes from each modality, while a style clustering module mitigates redundancy and enhances disease-relevant information by modeling and aligning consistent pathological signatures within a clustering space. Extensive evaluations on TCGA cohorts for cancer subtyping and survival analysis highlight MIRROR's superior performance, demonstrating its effectiveness in constructing comprehensive oncological feature representations and benefiting the cancer diagnosis. 

**Abstract (ZH)**: 病理组织学和转录组学是肿瘤学中的基本模态，涵盖了疾病的空间和分子方面。多模态自监督学习通过整合多种数据源展示了在学习病理表征方面的显著潜力。传统的多模态集成方法主要强调模态对齐，而对保留模态特异性结构关注不足。然而，与传统场景中多模态输入共享高度重叠特征不同，病理组织学和转录组学表现出明显的异质性，提供的是相互补充而非重叠的见解。病理组织学提供了组织形态和空间的背景，揭示了组织结构和细胞拓扑，而转录组学则通过基因表达模式勾勒出分子特征。这种固有的差异性为在保持模态特定忠实性的同时对齐它们带来了重大挑战。为应对这些挑战，我们提出了一种名为MIRROR的新颖多模态表示学习方法，旨在促进模态对齐和保留。MIRROR采用专用编码器提取每个模态的全面特征，并通过模态对齐模块实现表型模式与分子谱型之间的无缝集成。此外，模态保留模块保护每个模态的独特属性，同时通过建模和对齐群集中一致的病理特征来缓解冗余并增强与疾病相关的信息。在TCGA队列中对癌症亚型分类和生存分析的广泛评估表明，MIRROR表现出优越的性能，证明了其在构建全面的肿瘤学特征表示和促进癌症诊断方面的有效性。 

---
# Quantifying First-Order Markov Violations in Noisy Reinforcement Learning: A Causal Discovery Approach 

**Title (ZH)**: 在嘈杂强化学习中定量分析一阶马尔可夫性的违反：一种因果发现方法 

**Authors**: Naveen Mysore  

**Link**: [PDF](https://arxiv.org/pdf/2503.00206)  

**Abstract**: Reinforcement learning (RL) methods frequently assume that each new observation completely reflects the environment's state, thereby guaranteeing Markovian (one-step) transitions. In practice, partial observability or sensor/actuator noise often invalidates this assumption. This paper proposes a systematic methodology for detecting such violations, combining a partial correlation-based causal discovery process (PCMCI) with a novel Markov Violation score (MVS). The MVS measures multi-step dependencies that emerge when noise or incomplete state information disrupts the Markov property.
Classic control tasks (CartPole, Pendulum, Acrobot) serve as examples to illustrate how targeted noise and dimension omissions affect both RL performance and measured Markov consistency. Surprisingly, even substantial observation noise sometimes fails to induce strong multi-lag dependencies in certain domains (e.g., Acrobot). In contrast, dimension-dropping investigations show that excluding some state variables (e.g., angular velocities in CartPole and Pendulum) significantly reduces returns and increases MVS, while removing other dimensions has minimal impact.
These findings emphasize the importance of locating and safeguarding the most causally essential dimensions in order to preserve effective single-step learning. By integrating partial correlation tests with RL performance outcomes, the proposed approach precisely identifies when and where the Markov assumption is violated. This framework offers a principled mechanism for developing robust policies, informing representation learning, and addressing partial observability in real-world RL scenarios. All code and experimental logs are accessible for reproducibility (this https URL). 

**Abstract (ZH)**: 基于部分相关因果发现过程和新型马尔可夫违例分数的马尔可夫性质违例检测方法 

---
# AI and Semantic Communication for Infrastructure Monitoring in 6G-Driven Drone Swarms 

**Title (ZH)**: 基于6G驱动的无人机集群中的AI与语义通信的基础设施监控 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.00053)  

**Abstract**: The adoption of unmanned aerial vehicles to monitor critical infrastructure is gaining momentum in various industrial domains. Organizational imperatives drive this progression to minimize expenses, accelerate processes, and mitigate hazards faced by inspection personnel. However, traditional infrastructure monitoring systems face critical bottlenecks-5G networks lack the latency and reliability for large-scale drone coordination, while manual inspections remain costly and slow. We propose a 6G-enabled drone swarm system that integrates ultra-reliable, low-latency communications, edge AI, and semantic communication to automate inspections. By adopting LLMs for structured output and report generation, our framework is hypothesized to reduce inspection costs and improve fault detection speed compared to existing methods. 

**Abstract (ZH)**: 6G驱动的无人机蜂群系统用于监测关键基础设施 

---
