# TWIST: Teleoperated Whole-Body Imitation System 

**Title (ZH)**: TWIST: 远程操控全身模仿系统 

**Authors**: Yanjie Ze, Zixuan Chen, João Pedro Araújo, Zi-ang Cao, Xue Bin Peng, Jiajun Wu, C. Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02833)  

**Abstract**: Teleoperating humanoid robots in a whole-body manner marks a fundamental step toward developing general-purpose robotic intelligence, with human motion providing an ideal interface for controlling all degrees of freedom. Yet, most current humanoid teleoperation systems fall short of enabling coordinated whole-body behavior, typically limiting themselves to isolated locomotion or manipulation tasks. We present the Teleoperated Whole-Body Imitation System (TWIST), a system for humanoid teleoperation through whole-body motion imitation. We first generate reference motion clips by retargeting human motion capture data to the humanoid robot. We then develop a robust, adaptive, and responsive whole-body controller using a combination of reinforcement learning and behavior cloning (RL+BC). Through systematic analysis, we demonstrate how incorporating privileged future motion frames and real-world motion capture (MoCap) data improves tracking accuracy. TWIST enables real-world humanoid robots to achieve unprecedented, versatile, and coordinated whole-body motor skills--spanning whole-body manipulation, legged manipulation, locomotion, and expressive movement--using a single unified neural network controller. Our project website: this https URL 

**Abstract (ZH)**: 全身动作模仿的远程操作人形机器人系统(TWIST)：通过全身动作模仿进行人形机器人远程操作标志着开发通用机器人智能的基础步骤，人类动作提供了控制所有自由度的理想接口。然而，当前大多数人形远程操作系统尚不能实现协调的全身行为，通常仅限于孤立的移动或操作任务。我们提出了全身动作模仿远程操作系统(TWIST)，这是一种通过全身动作模仿进行人形机器人远程操作的系统。我们首先通过将人类动作捕捉数据重新定向到人形机器人来生成参考动作片段。然后，我们开发了一个健壮、自适应和响应迅速的全身控制器，结合使用强化学习和行为克隆(RL+BC)。通过系统分析，我们展示了如何引入未来的动作帧和真实世界动作捕捉(MoCap)数据以提高跟踪精度。TWIST使现实世界的人形机器人能够实现前所未有的多功能和协调的全身运动技能，涵盖全身操作、腿式操作、移动和表情动作，仅使用一个统一的神经网络控制器。项目网站: this https URL。 

---
# Re-purposing a modular origami manipulator into an adaptive physical computer for machine learning and robotic perception 

**Title (ZH)**: 将模块化 Origami 操作臂重新用于自适应物理计算机以实现机器学习与机器人感知 

**Authors**: Jun Wang, Suyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02744)  

**Abstract**: Physical computing has emerged as a powerful tool for performing intelligent tasks directly in the mechanical domain of functional materials and robots, reducing our reliance on the more traditional COMS computers. However, no systematic study explains how mechanical design can influence physical computing performance. This study sheds insights into this question by repurposing an origami-inspired modular robotic manipulator into an adaptive physical reservoir and systematically evaluating its computing capacity with different physical configurations, input setups, and computing tasks. By challenging this adaptive reservoir computer to complete the classical NARMA benchmark tasks, this study shows that its time series emulation performance directly correlates to the Peak Similarity Index (PSI), which quantifies the frequency spectrum correlation between the target output and reservoir dynamics. The adaptive reservoir also demonstrates perception capabilities, accurately extracting its payload weight and orientation information from the intrinsic dynamics. Importantly, such information extraction capability can be measured by the spatial correlation between nodal dynamics within the reservoir body. Finally, by integrating shape memory alloy (SMA) actuation, this study demonstrates how to exploit such computing power embodied in the physical body for practical, robotic operations. This study provides a strategic framework for harvesting computing power from soft robots and functional materials, demonstrating how design parameters and input selection can be configured based on computing task requirements. Extending this framework to bio-inspired adaptive materials, prosthetics, and self-adaptive soft robotic systems could enable next-generation embodied intelligence, where the physical structure can compute and interact with their digital counterparts. 

**Abstract (ZH)**: 物理计算在功能材料和机器人机械领域直接执行智能任务中 emerge 作为一种强大工具，减少了我们对传统 COMS 计算机的依赖。然而，尚未有系统性的研究解释机械设计如何影响物理计算性能。本研究通过将受 Origami 启发的模块化机器人 manipulator 重新用于自适应物理蓄能池，并系统地评估其在不同物理配置、输入设置和计算任务下的计算能力，从而揭示了这一问题。通过要求这种自适应蓄能池完成经典的 NARMA 标准测试任务，本研究显示其时间序列模拟性能直接与峰值相似度指数 (PSI) 相关，该指数量化了目标输出和蓄能池动力学之间的频谱关联。自适应蓄能池还展示了感知能力，能够准确地从内在动力学中提取其承载物的重量和姿态信息。重要的是，这种信息提取能力可以通过蓄能池体内节点动力学的空间相关性来测量。最后，通过整合形状记忆合金 (SMA) 执行机构，本研究展示了如何利用物理身体中嵌入的这种计算能力进行实际的机器人操作。本研究为从软机器人和功能材料中获取计算能力提供了战略框架，展示了如何根据计算任务要求配置设计参数和输入选择。将该框架扩展到生物启发的自适应材料、假肢和自适应软机器人系统，可以实现下一代嵌入式智能，其中物理结构能够计算并与数字对应物进行交互。 

---
# Automated Hybrid Reward Scheduling via Large Language Models for Robotic Skill Learning 

**Title (ZH)**: 基于大型语言模型的自动化混合奖励调度在机器人技能学习中的应用 

**Authors**: Changxin Huang, Junyang Liang, Yanbin Chang, Jingzhao Xu, Jianqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02483)  

**Abstract**: Enabling a high-degree-of-freedom robot to learn specific skills is a challenging task due to the complexity of robotic dynamics. Reinforcement learning (RL) has emerged as a promising solution; however, addressing such problems requires the design of multiple reward functions to account for various constraints in robotic motion. Existing approaches typically sum all reward components indiscriminately to optimize the RL value function and policy. We argue that this uniform inclusion of all reward components in policy optimization is inefficient and limits the robot's learning performance. To address this, we propose an Automated Hybrid Reward Scheduling (AHRS) framework based on Large Language Models (LLMs). This paradigm dynamically adjusts the learning intensity of each reward component throughout the policy optimization process, enabling robots to acquire skills in a gradual and structured manner. Specifically, we design a multi-branch value network, where each branch corresponds to a distinct reward component. During policy optimization, each branch is assigned a weight that reflects its importance, and these weights are automatically computed based on rules designed by LLMs. The LLM generates a rule set in advance, derived from the task description, and during training, it selects a weight calculation rule from the library based on language prompts that evaluate the performance of each branch. Experimental results demonstrate that the AHRS method achieves an average 6.48% performance improvement across multiple high-degree-of-freedom robotic tasks. 

**Abstract (ZH)**: 基于大规模语言模型的自动混合奖励调度框架使高自由度机器人学习特定技能 

---
# Quadrupedal Spine Control Strategies: Exploring Correlations Between System Dynamic Responses and Human Perspectives 

**Title (ZH)**: 四足脊椎控制策略：探索系统动力学响应与人类视角之间的关联 

**Authors**: Nicholas Hafner, Chaoran Liu, Carlos Ishi, Hiroshi Ishiguro  

**Link**: [PDF](https://arxiv.org/pdf/2505.02414)  

**Abstract**: Unlike their biological cousins, the majority of existing quadrupedal robots are constructed with rigid chassis. This results in motion that is either beetle-like or distinctly robotic, lacking the natural fluidity characteristic of mammalian movements. Existing literature on quadrupedal robots with spinal configurations primarily focuses on energy efficiency and does not consider the effects in human-robot interaction scenarios. Our contributions include an initial investigation into various trajectory generation strategies for a quadrupedal robot with a four degree of freedom spine, and an analysis on the effect that such methods have on human perception of gait naturalness compared to a fixed spine baseline. The strategies were evaluated using videos of walking, trotting and turning simulations. Among the four different strategies developed, the optimised time varying and the foot-tracking strategies were perceived to be more natural than the baseline in a randomised trial with 50 participants. Although none of the strategies demonstrated any energy efficiency improvements over the no-spine baseline, some showed greater footfall consistency at higher speeds. Given the greater likeability drawn from the more natural locomotion patterns, this type of robot displays potential for applications in social robot scenarios such as elderly care, where energy efficiency is not a primary concern. 

**Abstract (ZH)**: 不同于其生物原型，现有大多数四足机器人采用刚性机身构造，导致其运动方式要么像甲虫，要么显得十足机械，缺乏哺乳动物运动的自然流畅性。关于具有脊柱配置的四足机器人的现有文献主要集中在能量效率上，并未考虑其在人机交互场景中的影响。本研究的贡献在于初步探讨了四足机器人四种自由度脊柱下不同轨迹生成策略，并分析了这些方法对步态自然性感知的影响，相较于固定脊柱基准而言。通过行走、慢跑和转弯的模拟视频评估了这些策略。在随机试验证实中，优化的时间变化策略和足部跟踪策略被50名参与者认为比基准更具自然性。虽然这些策略并未在无脊柱基准情况下显示出任何能量效率的提升，但其中一些在更高速度下显示了更好的足部着地一致性。鉴于更自然运动模式带来的更高受欢迎度，这种类型的机器人在老年人护理等社交机器人应用场景中具有潜力，而不以能量效率为主要考量因素。 

---
# Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety 

**Title (ZH)**: 多智能体强化学习中层次化安全约束的冲突解决 

**Authors**: Jason J. Choi, Jasmine Jerry Aloor, Jingqi Li, Maria G. Mendoza, Hamsa Balakrishnan, Claire J. Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2505.02293)  

**Abstract**: Preventing collisions in multi-robot navigation is crucial for deployment. This requirement hinders the use of learning-based approaches, such as multi-agent reinforcement learning (MARL), on their own due to their lack of safety guarantees. Traditional control methods, such as reachability and control barrier functions, can provide rigorous safety guarantees when interactions are limited only to a small number of robots. However, conflicts between the constraints faced by different agents pose a challenge to safe multi-agent coordination.
To overcome this challenge, we propose a method that integrates multiple layers of safety by combining MARL with safety filters. First, MARL is used to learn strategies that minimize multiple agent interactions, where multiple indicates more than two. Particularly, we focus on interactions likely to result in conflicting constraints within the engagement distance. Next, for agents that enter the engagement distance, we prioritize pairs requiring the most urgent corrective actions. Finally, a dedicated safety filter provides tactical corrective actions to resolve these conflicts. Crucially, the design decisions for all layers of this framework are grounded in reachability analysis and a control barrier-value function-based filtering mechanism.
We validate our Layered Safe MARL framework in 1) hardware experiments using Crazyflie drones and 2) high-density advanced aerial mobility (AAM) operation scenarios, where agents navigate to designated waypoints while avoiding collisions. The results show that our method significantly reduces conflict while maintaining safety without sacrificing much efficiency (i.e., shorter travel time and distance) compared to baselines that do not incorporate layered safety. The project website is available at \href{this https URL}{[this https URL]} 

**Abstract (ZH)**: 多层次安全多智能体强化学习框架在多机器人导航中的应用 

---
# Robust Localization, Mapping, and Navigation for Quadruped Robots 

**Title (ZH)**: 四足机器人 robust 定位、建图与导航 

**Authors**: Dyuman Aditya, Junning Huang, Nico Bohlinger, Piotr Kicki, Krzysztof Walas, Jan Peters, Matteo Luperto, Davide Tateo  

**Link**: [PDF](https://arxiv.org/pdf/2505.02272)  

**Abstract**: Quadruped robots are currently a widespread platform for robotics research, thanks to powerful Reinforcement Learning controllers and the availability of cheap and robust commercial platforms. However, to broaden the adoption of the technology in the real world, we require robust navigation stacks relying only on low-cost sensors such as depth cameras. This paper presents a first step towards a robust localization, mapping, and navigation system for low-cost quadruped robots. In pursuit of this objective we combine contact-aided kinematic, visual-inertial odometry, and depth-stabilized vision, enhancing stability and accuracy of the system. Our results in simulation and two different real-world quadruped platforms show that our system can generate an accurate 2D map of the environment, robustly localize itself, and navigate autonomously. Furthermore, we present in-depth ablation studies of the important components of the system and their impact on localization accuracy. Videos, code, and additional experiments can be found on the project website: this https URL 

**Abstract (ZH)**: 低成本四足机器人稳健定位、建图与导航系统的初步研究 

---
# Prompt-responsive Object Retrieval with Memory-augmented Student-Teacher Learning 

**Title (ZH)**: 基于记忆增强的学生-教师学习的提示响应对象检索 

**Authors**: Malte Mosbach, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2505.02232)  

**Abstract**: Building models responsive to input prompts represents a transformative shift in machine learning. This paradigm holds significant potential for robotics problems, such as targeted manipulation amidst clutter. In this work, we present a novel approach to combine promptable foundation models with reinforcement learning (RL), enabling robots to perform dexterous manipulation tasks in a prompt-responsive manner. Existing methods struggle to link high-level commands with fine-grained dexterous control. We address this gap with a memory-augmented student-teacher learning framework. We use the Segment-Anything 2 (SAM 2) model as a perception backbone to infer an object of interest from user prompts. While detections are imperfect, their temporal sequence provides rich information for implicit state estimation by memory-augmented models. Our approach successfully learns prompt-responsive policies, demonstrated in picking objects from cluttered scenes. Videos and code are available at this https URL 

**Abstract (ZH)**: 构建响应输入提示的模型代表了机器学习中的范式转变。这种范式在机器人学问题，如杂乱环境中目标操作方面具有重要的潜力。在本文中，我们提出了一种新的方法，将可提示的基础模型与强化学习相结合，使机器人能够以响应提示的方式执行灵巧操作任务。现有方法很难将高层命令与精细的灵巧控制联系起来。我们通过引入一种记忆增强的学生-教师学习框架来解决这一问题。我们使用Segment-Anything 2 (SAM 2) 模型作为感知骨干，从用户提示中推断出感兴趣的对象。尽管检测结果可能不完美，但其时间序列提供了丰富的信息用于记忆增强模型进行隐式状态估计。我们的方法成功学习了响应提示的策略，并在杂乱场景中拾取对象的任务中得到了验证。有关视频和代码可在以下网址获取。 

---
# CrayonRobo: Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation 

**Title (ZH)**: CrayonRobo: 以物体为中心的提示驱动视觉语言行动模型用于机器人 manipulation 

**Authors**: Xiaoqi Li, Lingyun Xu, Mingxu Zhang, Jiaming Liu, Yan Shen, Iaroslav Ponomarenko, Jiahui Xu, Liang Heng, Siyuan Huang, Shanghang Zhang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.02166)  

**Abstract**: In robotic, task goals can be conveyed through various modalities, such as language, goal images, and goal videos. However, natural language can be ambiguous, while images or videos may offer overly detailed specifications. To tackle these challenges, we introduce CrayonRobo that leverages comprehensive multi-modal prompts that explicitly convey both low-level actions and high-level planning in a simple manner. Specifically, for each key-frame in the task sequence, our method allows for manual or automatic generation of simple and expressive 2D visual prompts overlaid on RGB images. These prompts represent the required task goals, such as the end-effector pose and the desired movement direction after contact. We develop a training strategy that enables the model to interpret these visual-language prompts and predict the corresponding contact poses and movement directions in SE(3) space. Furthermore, by sequentially executing all key-frame steps, the model can complete long-horizon tasks. This approach not only helps the model explicitly understand the task objectives but also enhances its robustness on unseen tasks by providing easily interpretable prompts. We evaluate our method in both simulated and real-world environments, demonstrating its robust manipulation capabilities. 

**Abstract (ZH)**: 机器人领域，任务目标可以通过语言、目标图像和目标视频等多种模态传达。然而，自然语言可能存在歧义，而图像或视频可能提供过于详细的具体信息。为应对这些挑战，我们引入了CrayonRobo，该系统利用全面的多模态提示，以简单明确的方式传达低级动作和高级规划。具体而言，对于任务序列中的每个关键帧，我们的方法允许手动或自动生成简洁且富有表现力的2D视觉提示，并叠加在RGB图像上。这些提示表示所需的任务目标，例如末端执行器的姿态和接触后的期望运动方向。我们开发了一种训练策略，使模型能够解释这些视觉语言提示，并在SE(3)空间中预测相应的接触姿态和运动方向。此外，通过顺序执行所有关键帧步骤，模型可以完成长时序任务。该方法不仅有助于模型明确理解任务目标，还能通过提供易于解释的提示增强其在未见过的任务上的鲁棒性。我们分别在模拟和实际环境中评估了该方法，展示了其稳健的操作能力。 

---
# Interleave-VLA: Enhancing Robot Manipulation with Interleaved Image-Text Instructions 

**Title (ZH)**: 交错-VLA：增强机器人操作的交错图像-文本指令 

**Authors**: Cunxin Fan, Xiaosong Jia, Yihang Sun, Yixiao Wang, Jianglan Wei, Ziyang Gong, Xiangyu Zhao, Masayoshi Tomizuka, Xue Yang, Junchi Yan, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.02152)  

**Abstract**: Vision-Language-Action (VLA) models have shown great promise for generalist robotic manipulation in the physical world. However, existing models are restricted to robot observations and text-only instructions, lacking the flexibility of interleaved multimodal instructions enabled by recent advances in foundation models in the digital world. In this paper, we present Interleave-VLA, the first framework capable of comprehending interleaved image-text instructions and directly generating continuous action sequences in the physical world. It offers a flexible, model-agnostic paradigm that extends state-of-the-art VLA models with minimal modifications and strong zero-shot generalization. A key challenge in realizing Interleave-VLA is the absence of large-scale interleaved embodied datasets. To bridge this gap, we develop an automatic pipeline that converts text-only instructions from real-world datasets in Open X-Embodiment into interleaved image-text instructions, resulting in the first large-scale real-world interleaved embodied dataset with 210k episodes. Through comprehensive evaluation on simulation benchmarks and real-robot experiments, we demonstrate that Interleave-VLA offers significant benefits: 1) it improves out-of-domain generalization to unseen objects by 2-3x compared to state-of-the-art baselines, 2) supports flexible task interfaces, and 3) handles diverse user-provided image instructions in a zero-shot manner, such as hand-drawn sketches. We further analyze the factors behind Interleave-VLA's strong zero-shot performance, showing that the interleaved paradigm effectively leverages heterogeneous datasets and diverse instruction images, including those from the Internet, which demonstrates strong potential for scaling up. Our model and dataset will be open-sourced. 

**Abstract (ZH)**: 基于视觉-语言-行动的交互式框架：理解和生成物理世界的交错指令与连续动作序列 

---
# A Synergistic Framework of Nonlinear Acoustic Computing and Reinforcement Learning for Real-World Human-Robot Interaction 

**Title (ZH)**: 非线性声学计算与强化学习协同框架在实际人机交互中的应用 

**Authors**: Xiaoliang Chen, Xin Yu, Le Chang, Yunhe Huang, Jiashuai He, Shibo Zhang, Jin Li, Likai Lin, Ziyu Zeng, Xianling Tu, Shuyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01998)  

**Abstract**: This paper introduces a novel framework integrating nonlinear acoustic computing and reinforcement learning to enhance advanced human-robot interaction under complex noise and reverberation. Leveraging physically informed wave equations (e.g., Westervelt, KZK), the approach captures higher-order phenomena such as harmonic generation and shock formation. By embedding these models in a reinforcement learning-driven control loop, the system adaptively optimizes key parameters (e.g., absorption, beamforming) to mitigate multipath interference and non-stationary noise. Experimental evaluations-covering far-field localization, weak signal detection, and multilingual speech recognition-demonstrate that this hybrid strategy surpasses traditional linear methods and purely data-driven baselines, achieving superior noise suppression, minimal latency, and robust accuracy in demanding real-world scenarios. The proposed system demonstrates broad application prospects in AI hardware, robot, machine audition, artificial audition, and brain-machine interfaces. 

**Abstract (ZH)**: 本文介绍了一种将非线性声计算与强化学习结合起来的新框架，以在复杂噪声和混响环境下增强先进的机器人交互。该方法利用物理信息波动方程（如Westervelt方程、KZK方程）捕获高阶现象，如谐波生成和冲击形成。通过将这些模型嵌入到基于强化学习的控制循环中，系统能够自适应优化关键参数（如吸收、波束形成）以减轻多路径干扰和非稳定噪声。实验评估涵盖了远场定位、弱信号检测和多语种语音识别，表明这种混合策略超越了传统的线性方法和纯粹的数据驱动基准，实现了卓越的噪声抑制、最小的延迟和在苛刻的真实世界场景中的稳健准确性。所提出系统在AI硬件、机器人、机器听觉、人工听觉和脑机接口等领域展示了广泛的应用前景。 

---
# A Goal-Oriented Reinforcement Learning-Based Path Planning Algorithm for Modular Self-Reconfigurable Satellites 

**Title (ZH)**: 面向目标的基于强化学习的模块化自重构卫星路径规划算法 

**Authors**: Bofei Liu, Dong Ye, Zunhao Yao, Zhaowei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.01966)  

**Abstract**: Modular self-reconfigurable satellites refer to satellite clusters composed of individual modular units capable of altering their configurations. The configuration changes enable the execution of diverse tasks and mission objectives. Existing path planning algorithms for reconfiguration often suffer from high computational complexity, poor generalization capability, and limited support for diverse target configurations. To address these challenges, this paper proposes a goal-oriented reinforcement learning-based path planning algorithm. This algorithm is the first to address the challenge that previous reinforcement learning methods failed to overcome, namely handling multiple target configurations. Moreover, techniques such as Hindsight Experience Replay and Invalid Action Masking are incorporated to overcome the significant obstacles posed by sparse rewards and invalid actions. Based on these designs, our model achieves a 95% and 73% success rate in reaching arbitrary target configurations in a modular satellite cluster composed of four and six units, respectively. 

**Abstract (ZH)**: 模块化自重构卫星的目标导向强化学习路径规划算法 

---
# Semantic Intelligence: Integrating GPT-4 with A Planning in Low-Cost Robotics 

**Title (ZH)**: 语义智能：将GPT-4与规划集成于低成本机器人中 

**Authors**: Jesse Barkley, Abraham George, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2505.01931)  

**Abstract**: Classical robot navigation often relies on hardcoded state machines and purely geometric path planners, limiting a robot's ability to interpret high-level semantic instructions. In this paper, we first assess GPT-4's ability to act as a path planner compared to the A* algorithm, then present a hybrid planning framework that integrates GPT-4's semantic reasoning with A* on a low-cost robot platform operating on ROS2 Humble. Our approach eliminates explicit finite state machine (FSM) coding by using prompt-based GPT-4 reasoning to handle task logic while maintaining the accurate paths computed by A*. The GPT-4 module provides semantic understanding of instructions and environmental cues (e.g., recognizing toxic obstacles or crowded areas to avoid, or understanding low-battery situations requiring alternate route selection), and dynamically adjusts the robot's occupancy grid via obstacle buffering to enforce semantic constraints. We demonstrate multi-step reasoning for sequential tasks, such as first navigating to a resource goal and then reaching a final destination safely. Experiments on a Petoi Bittle robot with an overhead camera and Raspberry Pi Zero 2W compare classical A* against GPT-4-assisted planning. Results show that while A* is faster and more accurate for basic route generation and obstacle avoidance, the GPT-4-integrated system achieves high success rates (96-100%) on semantic tasks that are infeasible for pure geometric planners. This work highlights how affordable robots can exhibit intelligent, context-aware behaviors by leveraging large language model reasoning with minimal hardware and no fine-tuning. 

**Abstract (ZH)**: 经典机器人导航往往依赖于硬编码的状态机和纯粹的几何路径规划器，限制了机器人对高级语义指令的解释能力。本文首先评估GPT-4作为路径规划器的能力与A*算法相比，然后提出了一种结合GPT-4语义推理和A*算法的混合规划框架，应用于ROS2 Humble平台上的低成本机器人。我们的方法通过使用基于提示的GPT-4推理来处理任务逻辑，同时保留了A*计算的精确路径，从而消除了显式的有限状态机（FSM）编码。GPT-4模块提供了对指令和环境线索的理解（例如，识别有毒障碍物或拥挤区域以避免，或理解低电量情况需要选择替代路线），并通过障碍物缓冲动态调整机器人的占用网格，以强制执行语义约束。我们展示了多步推理以执行顺序任务，例如首先导航到资源目标，然后安全到达最终目的地。Petoi Bittle机器人的实验证明，在顶点摄像头和Raspberry Pi Zero 2W的辅助下，与经典的A*规划相比，GPT-4辅助规划提高了语义任务的成功率（96-100%），这些任务对于纯粹几何规划器来说是不可行的。本文突出了通过利用大语言模型推理，低成本机器人可以表现出智能、情境感知的行为，无需额外的硬件和微调。 

---
# ReLI: A Language-Agnostic Approach to Human-Robot Interaction 

**Title (ZH)**: ReLI: 一种语言无关的人机交互方法 

**Authors**: Linus Nwankwo, Bjoern Ellensohn, Ozan Özdenizci, Elmar Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2505.01862)  

**Abstract**: Adapting autonomous agents to industrial, domestic, and other daily tasks is currently gaining momentum. However, in the global or cross-lingual application contexts, ensuring effective interaction with the environment and executing unrestricted human task-specified instructions in diverse languages remains an unsolved problem. To address this challenge, we propose ReLI, a language-agnostic framework designed to enable autonomous agents to converse naturally, semantically reason about the environment, and to perform downstream tasks, regardless of the task instruction's linguistic origin. First, we ground large-scale pre-trained foundation models and transform them into language-to-action models that can directly provide common-sense reasoning and high-level robot control through natural, free-flow human-robot conversational interactions. Further, we perform cross-lingual grounding of the models to ensure that ReLI generalises across the global languages. To demonstrate the ReLI's robustness, we conducted extensive simulated and real-world experiments on various short- and long-horizon tasks, including zero-shot and few-shot spatial navigation, scene information retrieval, and query-oriented tasks. We benchmarked the performance on 140 languages involving over 70K multi-turn conversations. On average, ReLI achieved over 90%$\pm$0.2 accuracy in cross-lingual instruction parsing and task execution success rates. These results demonstrate the ReLI's potential to enhance natural human-robot interaction in the real world while championing linguistic diversity. Demonstrations and resources will be publicly available at this https URL. 

**Abstract (ZH)**: 面向工业、家用及其它日常任务的自主代理适应性正在逐步提升。然而，在全球或跨语言应用场景中，确保自主代理与环境有效互动并执行多语言的不限制人类任务指令仍是一个未解决的问题。为应对这一挑战，我们提出ReLI，这是一种语言无关的框架，旨在使自主代理能够进行自然对话、语义推理以及执行下游任务，而不受任务指令语言来源的影响。首先，我们基于大规模预训练基础模型并将其转化为语言到行动的模型，这些模型可以直接通过自然、流畅的人机对话互动进行常识推理和高级机器人控制。进一步地，我们对模型进行跨语言grounding，确保ReLI在多种全球语言间通用。为了证明ReLI的鲁棒性，我们在各种短期和长期任务上进行了广泛的模拟和现实世界实验，包括零样本和少样本的空间导航、场景信息检索以及查询导向任务。我们在超过70,000轮多轮对话中对140种语言进行了基准测试。结果显示，ReLI在跨语言指令解析和任务执行成功率方面的平均准确率超过90%±0.2。这些结果展示了ReLI在增强现实世界中的人机自然交互方面的潜力，同时也支持语言多样性。具体内容和资源将于以下链接公开：this https URL。 

---
# RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation 

**Title (ZH)**: RoBridge：连接认知与执行的分层架构用于通用机器人操作 

**Authors**: Kaidong Zhang, Rongtao Xu, Pengzhen Ren, Junfan Lin, Hefeng Wu, Liang Lin, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01709)  

**Abstract**: Operating robots in open-ended scenarios with diverse tasks is a crucial research and application direction in robotics. While recent progress in natural language processing and large multimodal models has enhanced robots' ability to understand complex instructions, robot manipulation still faces the procedural skill dilemma and the declarative skill dilemma in open environments. Existing methods often compromise cognitive and executive capabilities. To address these challenges, in this paper, we propose RoBridge, a hierarchical intelligent architecture for general robotic manipulation. It consists of a high-level cognitive planner (HCP) based on a large-scale pre-trained vision-language model (VLM), an invariant operable representation (IOR) serving as a symbolic bridge, and a generalist embodied agent (GEA). RoBridge maintains the declarative skill of VLM and unleashes the procedural skill of reinforcement learning, effectively bridging the gap between cognition and execution. RoBridge demonstrates significant performance improvements over existing baselines, achieving a 75% success rate on new tasks and an 83% average success rate in sim-to-real generalization using only five real-world data samples per task. This work represents a significant step towards integrating cognitive reasoning with physical execution in robotic systems, offering a new paradigm for general robotic manipulation. 

**Abstract (ZH)**: 在多样任务的开放场景中操作机器人是机器人学研究和应用的一个重要方向。尽管近期自然语言处理和大规模多模态模型的进步增强了机器人理解复杂指令的能力，但在开放环境中，机器人操作仍然面临着程序技能困境和声明技能困境。现有方法往往在认知能力和执行能力之间妥协。为应对这些挑战，本文提出RoBridge，一种用于通用机器人操作的分层智能架构。它由基于大规模预训练视觉-语言模型的高层认知规划器（HCP）、作为符号桥梁的不变可操作表示（IOR）以及通用体态代理人（GEA）组成。RoBridge保持了视觉-语言模型的声明技能，并释放了强化学习的程序技能，有效地弥合了认知与执行之间的差距。RoBridge在新任务上实现了75%的成功率，并在仅使用每个任务五个真实世界数据样本的情况下，成功地将仿真实验迁移到真实环境，平均成功率达到83%。这项工作代表着将认知推理与物理执行集成到机器人系统中的一个重要步骤，提供了通用机器人操作的新范式。 

---
# Deformable Cargo Transport in Microgravity with Astrobee 

**Title (ZH)**: 微重力环境下Astrobee的可变形载荷运输 

**Authors**: Daniel Morton, Rika Antonova, Brian Coltin, Marco Pavone, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2505.01630)  

**Abstract**: We present pyastrobee: a simulation environment and control stack for Astrobee in Python, with an emphasis on cargo manipulation and transport tasks. We also demonstrate preliminary success from a sampling-based MPC controller, using reduced-order models of NASA's cargo transfer bag (CTB) to control a high-order deformable finite element model. Our code is open-source, fully documented, and available at this https URL 

**Abstract (ZH)**: Python中的pyastrobee：Astrobee的仿真环境和控制栈，侧重于货物操作与运输任务。基于采样 MPC 控制器的初步成功演示，使用NASA货物转移袋（CTB）的降阶模型控制高阶可变形有限元模型。代码开源、完全文档化，并可从以下链接获取。 

---
# Phasing Through the Flames: Rapid Motion Planning with the AGHF PDE for Arbitrary Objective Functions and Constraints 

**Title (ZH)**: 火焰中的相位跃迁：任意目标函数和约束条件下的快速运动规划 

**Authors**: Challen Enninful Adu, César E. Ramos Chuquiure, Yutong Zhou, Pearl Lin, Ruikai Yang, Bohao Zhang, Shubham Singh, Ram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01589)  

**Abstract**: The generation of optimal trajectories for high-dimensional robotic systems under constraints remains computationally challenging due to the need to simultaneously satisfy dynamic feasibility, input limits, and task-specific objectives while searching over high-dimensional spaces. Recent approaches using the Affine Geometric Heat Flow (AGHF) Partial Differential Equation (PDE) have demonstrated promising results, generating dynamically feasible trajectories for complex systems like the Digit V3 humanoid within seconds. These methods efficiently solve trajectory optimization problems over a two-dimensional domain by evolving an initial trajectory to minimize control effort. However, these AGHF approaches are limited to a single type of optimal control problem (i.e., minimizing the integral of squared control norms) and typically require initial guesses that satisfy constraints to ensure satisfactory convergence. These limitations restrict the potential utility of the AGHF PDE especially when trying to synthesize trajectories for robotic systems. This paper generalizes the AGHF formulation to accommodate arbitrary cost functions, significantly expanding the classes of trajectories that can be generated. This work also introduces a Phase1 - Phase 2 Algorithm that enables the use of constraint-violating initial guesses while guaranteeing satisfactory convergence. The effectiveness of the proposed method is demonstrated through comparative evaluations against state-of-the-art techniques across various dynamical systems and challenging trajectory generation problems. Project Page: this https URL 

**Abstract (ZH)**: 高维约束条件下最优轨迹生成对于高性能机器人系统而言仍具有计算挑战性，现有的方法如使用Affine Geometric Heat Flow (AGHF)偏微分方程（PDE）虽能在秒内生成复杂的Digit V3人形机器人动态可行轨迹，但仅适用于特定类型的最优控制问题，并且通常需要满足约束条件的初始猜测以确保收敛性。为克服这些限制，本文将AGHF公式化扩展至任意成本函数，显著扩展了可生成的轨迹类型。此外，本文还引入了Phase1-Phase2算法，可使用违反约束的初始猜测并保证收敛性。通过与最新技术在多种动力学系统和挑战性轨迹生成问题上的对比评估，证明了所提出方法的有效性。项目页面：this https URL 

---
# ASAP-MO:Advanced Situational Awareness and Perception for Mission-critical Operations 

**Title (ZH)**: ASAP-MO：高级态势感知与关键任务操作中的感知 

**Authors**: Veronica Vannini, William Dubois, Olivier Gamache, Jean-Michel Fortin, Nicolas Samson, Effie Daum, François Pomerleau, Edith Brotherton  

**Link**: [PDF](https://arxiv.org/pdf/2505.01547)  

**Abstract**: Deploying robotic missions can be challenging due to the complexity of controlling robots with multiple degrees of freedom, fusing diverse sensory inputs, and managing communication delays and interferences. In nuclear inspection, robots can be crucial in assessing environments where human presence is limited, requiring precise teleoperation and coordination. Teleoperation requires extensive training, as operators must process multiple outputs while ensuring safe interaction with critical assets. These challenges are amplified when operating a fleet of heterogeneous robots across multiple environments, as each robot may have distinct control interfaces, sensory systems, and operational constraints. Efficient coordination in such settings remains an open problem. This paper presents a field report on how we integrated robot fleet capabilities - including mapping, localization, and telecommunication - toward a joint mission. We simulated a nuclear inspection scenario for exposed areas, using lights to represent a radiation source. We deployed two Unmanned Ground Vehicles (UGVs) tasked with mapping indoor and outdoor environments while remotely controlled from a single base station. Despite having distinct operational goals, the robots produced a unified map output, demonstrating the feasibility of coordinated multi-robot missions. Our results highlight key operational challenges and provide insights into improving adaptability and situational awareness in remote robotic deployments. 

**Abstract (ZH)**: 部署机器人任务由于多自由度控制的复杂性、多种传感器输入的融合以及通信延迟和干扰的管理而具有挑战性。在核检查中，机器人在人类存在受限的环境中可以发挥关键作用，要求精准的遥控和协调。遥控需要广泛训练，因为操作员必须处理多个输出并确保与关键资产的安全互动。当操作多种环境中的异构机器人舰队时，这些挑战会加剧，因为每台机器人都可能具有不同的控制界面、感测系统和运行约束。在这种环境中有效地协调仍然是一个开放问题。本文报告了我们如何集成机器人舰队的能力——包括测绘、定位和通信——以为联合任务做准备。我们模拟了一个暴露区域的核检查场景，使用灯光代表辐射源。我们部署了两辆无人地面车辆（UGVs），任务是在单个基站远程控制下测绘室内和室外环境。尽管具备不同的操作目标，机器人仍生成了统一的地图输出，展示了协调多机器人任务的可行性。我们的结果突显了关键的操作挑战，并提供了关于改善远程机器人部署中的适应性和情况意识的见解。 

---
# A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI 

**Title (ZH)**: embodied AI时代基于物理模拟器的机器人导航与操作综述 

**Authors**: Lik Hang Kenny Wong, Xueyang Kang, Kaixin Bai, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01458)  

**Abstract**: Navigation and manipulation are core capabilities in Embodied AI, yet training agents with these capabilities in the real world faces high costs and time complexity. Therefore, sim-to-real transfer has emerged as a key approach, yet the sim-to-real gap persists. This survey examines how physics simulators address this gap by analyzing their properties overlooked in previous surveys. We also analyze their features for navigation and manipulation tasks, along with hardware requirements. Additionally, we offer a resource with benchmark datasets, metrics, simulation platforms, and cutting-edge methods-such as world models and geometric equivariance-to help researchers select suitable tools while accounting for hardware constraints. 

**Abstract (ZH)**: 物理仿真的特性分析与导航 manipulation 任务的硬件需求：填补 sim-to-real 间隙的调研及资源指南 

---
# Giving Simulated Cells a Voice: Evolving Prompt-to-Intervention Models for Cellular Control 

**Title (ZH)**: 赋予模拟细胞发声的能力：演化提示到干预的模型以控制细胞行为 

**Authors**: Nam H. Le, Patrick Erikson, Yanbo Zhang, Michael Levin, Josh Bongard  

**Link**: [PDF](https://arxiv.org/pdf/2505.02766)  

**Abstract**: Guiding biological systems toward desired states, such as morphogenetic outcomes, remains a fundamental challenge with far-reaching implications for medicine and synthetic biology. While large language models (LLMs) have enabled natural language as an interface for interpretable control in AI systems, their use as mediators for steering biological or cellular dynamics remains largely unexplored.
In this work, we present a functional pipeline that translates natural language prompts into spatial vector fields capable of directing simulated cellular collectives. Our approach combines a large language model with an evolvable neural controller (Prompt-to-Intervention, or P2I), optimized via evolutionary strategies to generate behaviors such as clustering or scattering in a simulated 2D environment.
We demonstrate that even with constrained vocabulary and simplified cell models, evolved P2I networks can successfully align cellular dynamics with user-defined goals expressed in plain language. This work offers a complete loop from language input to simulated bioelectric-like intervention to behavioral output, providing a foundation for future systems capable of natural language-driven cellular control. 

**Abstract (ZH)**: 引导生物系统朝向所需的稳态，如形态发生结果，仍然是一个基础性挑战，对医学和合成生物学具有深远的意义。虽然大型语言模型（LLMs）已经使得自然语言成为AI系统的可解释控制界面，但它们作为引导生物或细胞动力学的中介仍然鲜有探索。

在本工作中，我们提出了一种功能管道，将自然语言提示转化为能够指导模拟细胞集群的空间向量场。我们的方法结合了大型语言模型与可进化神经控制器（Prompt-to-Intervention，或P2I），并通过进化策略进行优化，以生成如聚类或离散等行为，在模拟2D环境中。

我们证明，即使在受限的词汇量和简化的细胞模型下，进化出的P2I网络也能成功将细胞动力学与用户用自然语言定义的目标对齐。本工作提供了一个从语言输入到模拟生物电类似干预再到行为输出的完整闭环，为未来能够实现自然语言驱动细胞控制的系统奠定了基础。 

---
# MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans 

**Title (ZH)**: MetaScenes: 向自动化真实世界3D扫描的副本创建目标迈进 

**Authors**: Huangyue Yu, Baoxiong Jia, Yixin Chen, Yandan Yang, Puhao Li, Rongpeng Su, Jiaxin Li, Qing Li, Wei Liang, Song-Chun Zhu, Tengyu Liu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02388)  

**Abstract**: Embodied AI (EAI) research requires high-quality, diverse 3D scenes to effectively support skill acquisition, sim-to-real transfer, and generalization. Achieving these quality standards, however, necessitates the precise replication of real-world object diversity. Existing datasets demonstrate that this process heavily relies on artist-driven designs, which demand substantial human effort and present significant scalability challenges. To scalably produce realistic and interactive 3D scenes, we first present MetaScenes, a large-scale, simulatable 3D scene dataset constructed from real-world scans, which includes 15366 objects spanning 831 fine-grained categories. Then, we introduce Scan2Sim, a robust multi-modal alignment model, which enables the automated, high-quality replacement of assets, thereby eliminating the reliance on artist-driven designs for scaling 3D scenes. We further propose two benchmarks to evaluate MetaScenes: a detailed scene synthesis task focused on small item layouts for robotic manipulation and a domain transfer task in vision-and-language navigation (VLN) to validate cross-domain transfer. Results confirm MetaScene's potential to enhance EAI by supporting more generalizable agent learning and sim-to-real applications, introducing new possibilities for EAI research. Project website: this https URL. 

**Abstract (ZH)**: embodied AI (EAI) 研究需要高质量、多样的3D场景以有效支持技能获取、模拟到现实的转移以及泛化。然而，达到这些质量标准需要精确复现现实世界对象的多样性。现有数据集表明，这一过程高度依赖于艺术家驱动的设计，这需要大量的人力投入并呈现显著的扩展性挑战。为了能够扩展性地生成真实的、交互的3D场景，我们首先介绍了MetaScenes，这是一个大规模的、可模拟的3D场景数据集，由真实世界的扫描构建，包含15366个对象，涵盖831个细粒度类别。然后，我们引入了Scan2Sim，这是一个稳健的多模态对齐模型，能够自动进行高质量的资产替换，从而消除依赖于艺术家驱动的设计以扩展3D场景。此外，我们提出了两个基准来评估MetaScenes：一个专注于小型物品布局的详细场景合成任务，用于机器人操作，以及一个在视觉语言导航(VLN)中的领域转移任务，以验证跨域转移。结果证实MetaScenes有望通过支持更通用的代理学习和模拟到现实的应用来增强EAI，为EAI研究引入新的可能性。项目网站：this https URL。 

---
# Act Natural! Extending Naturalistic Projection to Multimodal Behavior Scenarios 

**Title (ZH)**: Act Natural！将自然投影扩展到多模态行为场景 

**Authors**: Hamzah I. Khan, David Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2505.01945)  

**Abstract**: Autonomous agents operating in public spaces must consider how their behaviors might affect the humans around them, even when not directly interacting with them. To this end, it is often beneficial to be predictable and appear naturalistic. Existing methods for this purpose use human actor intent modeling or imitation learning techniques, but these approaches rarely capture all possible motivations for human behavior and/or require significant amounts of data. Our work extends a technique for modeling unimodal naturalistic behaviors with an explicit convex set representation, to account for multimodal behavior by using multiple convex sets. This more flexible representation provides a higher degree of fidelity in data-driven modeling of naturalistic behavior that arises in real-world scenarios in which human behavior is, in some sense, discrete, e.g. whether or not to yield at a roundabout. Equipped with this new set representation, we develop an optimization-based filter to project arbitrary trajectories into the set so that they appear naturalistic to humans in the scene, while also satisfying vehicle dynamics, actuator limits, etc. We demonstrate our methods on real-world human driving data from the inD (intersection) and rounD (roundabout) datasets. 

**Abstract (ZH)**: 自主操作于公共空间的代理必须考虑其行为可能对周围的行人造成的影响，即使不直接与他们互动。为此，表现出可预测性和自然性通常是有益的。现有方法通过人类行为意图建模或仿真人学习技术来实现这一目标，但这些方法很少能够捕捉到所有可能的人类行为动机，或者需要大量数据。我们的工作扩展了一种用于建模单模态自然行为的技术，通过使用多个凸集来考虑多模态行为，从而提供更灵活的表示形式，在真实场景中以更高的保真度数据驱动建模自然行为，例如在环岛是否礼让等具有一种意义上离散的人类行为。借助这种新的集合表示法，我们开发了一种基于优化的滤波器，将任意轨迹投影到集合中，使其在场景中显得自然，同时满足车辆动力学、执行器限制等要求。我们通过inD（交叉口）和rounD（环岛）数据集中的真实人类驾驶数据，展示了我们的方法。 

---
# PhysNav-DG: A Novel Adaptive Framework for Robust VLM-Sensor Fusion in Navigation Applications 

**Title (ZH)**: PhysNav-DG: 一种新型自适应框架，用于导航应用中的鲁棒VLMI传感器融合 

**Authors**: Trisanth Srinivasan, Santosh Patapati  

**Link**: [PDF](https://arxiv.org/pdf/2505.01881)  

**Abstract**: Robust navigation in diverse environments and domains requires both accurate state estimation and transparent decision making. We present PhysNav-DG, a novel framework that integrates classical sensor fusion with the semantic power of vision-language models. Our dual-branch architecture predicts navigation actions from multi-sensor inputs while simultaneously generating detailed chain-of-thought explanations. A modified Adaptive Kalman Filter dynamically adjusts its noise parameters based on environmental context. It leverages several streams of raw sensor data along with semantic insights from models such as LLaMA 3.2 11B and BLIP-2. To evaluate our approach, we introduce the MD-NEX Benchmark, a novel multi-domain dataset that unifies indoor navigation, autonomous driving, and social navigation tasks with ground-truth actions and human-validated explanations. Extensive experiments and ablations show that PhysNav-DG improves navigation success rates by over 20% and achieves high efficiency, with explanations that are both highly grounded and clear. This work connects high-level semantic reasoning and geometric planning for safer and more trustworthy autonomous systems. 

**Abstract (ZH)**: 鲁棒的导航在多变的环境和领域中需要准确的状态估计和透明的决策过程。我们提出了PhysNav-DG这一新颖框架，将经典传感器融合与视觉语言模型的语义能力相结合。我们的双分支架构从多传感器输入中预测导航动作，同时生成详细的解释链。经过修改的自适应卡尔曼滤波器根据环境上下文动态调整其噪声参数。它利用多路原始传感器数据，并结合诸如LLaMA 3.2 11B和BLIP-2等模型的语义洞察。为了评估我们的方法，我们引入了MD-NEX基准，这是一个新型多域数据集，将室内导航、自动驾驶和社会导航任务统一起来，包括真实动作和人工验证的解释。通过广泛的实验和消融研究显示，PhysNav-DG可以提高超过20%的导航成功率，并且具有高效率，解释既详细又清晰。这项工作将高级语义推理和几何规划相结合，以实现更安全、更可信赖的自主系统。 

---
# Interactive Double Deep Q-network: Integrating Human Interventions and Evaluative Predictions in Reinforcement Learning of Autonomous Driving 

**Title (ZH)**: 交互式双深度Q网络：在自主驾驶强化学习中的人类干预与评估预测整合 

**Authors**: Alkis Sygkounas, Ioannis Athanasiadis, Andreas Persson, Michael Felsberg, Amy Loutfi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01440)  

**Abstract**: Integrating human expertise with machine learning is crucial for applications demanding high accuracy and safety, such as autonomous driving. This study introduces Interactive Double Deep Q-network (iDDQN), a Human-in-the-Loop (HITL) approach that enhances Reinforcement Learning (RL) by merging human insights directly into the RL training process, improving model performance. Our proposed iDDQN method modifies the Q-value update equation to integrate human and agent actions, establishing a collaborative approach for policy development. Additionally, we present an offline evaluative framework that simulates the agent's trajectory as if no human intervention had occurred, to assess the effectiveness of human interventions. Empirical results in simulated autonomous driving scenarios demonstrate that iDDQN outperforms established approaches, including Behavioral Cloning (BC), HG-DAgger, Deep Q-Learning from Demonstrations (DQfD), and vanilla DRL in leveraging human expertise for improving performance and adaptability. 

**Abstract (ZH)**: 将人类专业知识与机器学习相结合对于需求高准确性和安全性的应用，如自动驾驶至关重要。本研究引入了交互式双深度Q网络(iDDQN)，这是一种human-in-the-loop (HITL) 方法，通过直接将人类见解融入强化学习(Reinforcement Learning, RL)的训练过程，提高模型性能。我们提出的iDDQN方法修改了Q值更新公式，以整合人类和代理行动，建立一种策略开发的合作方法。此外，我们介绍了一个离线评估框架，模拟代理的轨迹，仿佛没有人类干预，以评估人类干预的有效性。在模拟的自动驾驶场景中的实证结果表明，iDDQN在利用人类专业知识提高性能和适应性方面优于包括行为克隆(BC)、HG-DAgger、深度Q学习从演示(DQfD)和传统的强化学习(Vanilla DRL)在内的现有方法。 

---
# FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models 

**Title (ZH)**: FormalMATH: 大型语言模型形式化数学推理benchmark评测 

**Authors**: Zhouliang Yu, Ruotian Peng, Keyi Ding, Yizhe Li, Zhongyuan Peng, Minghao Liu, Yifan Zhang, Zheng Yuan, Huajian Xin, Wenhao Huang, Yandong Wen, Ge Zhang, Weiyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02735)  

**Abstract**: Formal mathematical reasoning remains a critical challenge for artificial intelligence, hindered by limitations of existing benchmarks in scope and scale. To address this, we present FormalMATH, a large-scale Lean4 benchmark comprising 5,560 formally verified problems spanning from high-school Olympiad challenges to undergraduate-level theorems across diverse domains (e.g., algebra, applied mathematics, calculus, number theory, and discrete mathematics). To mitigate the inefficiency of manual formalization, we introduce a novel human-in-the-loop autoformalization pipeline that integrates: (1) specialized large language models (LLMs) for statement autoformalization, (2) multi-LLM semantic verification, and (3) negation-based disproof filtering strategies using off-the-shelf LLM-based provers. This approach reduces expert annotation costs by retaining 72.09% of statements before manual verification while ensuring fidelity to the original natural-language problems. Our evaluation of state-of-the-art LLM-based theorem provers reveals significant limitations: even the strongest models achieve only 16.46% success rate under practical sampling budgets, exhibiting pronounced domain bias (e.g., excelling in algebra but failing in calculus) and over-reliance on simplified automation tactics. Notably, we identify a counterintuitive inverse relationship between natural-language solution guidance and proof success in chain-of-thought reasoning scenarios, suggesting that human-written informal reasoning introduces noise rather than clarity in the formal reasoning settings. We believe that FormalMATH provides a robust benchmark for benchmarking formal mathematical reasoning. 

**Abstract (ZH)**: FormalMATH：大规模Lean4基准库及其在形式化数学推理评估中的应用 

---
# Machine-Learning-Powered Neural Interfaces for Smart Prosthetics and Diagnostics 

**Title (ZH)**: 机器学习驱动的神经接口赋能智能假肢与诊断 

**Authors**: MohammadAli Shaeri, Jinhan Liu, Mahsa Shoaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.02516)  

**Abstract**: Advanced neural interfaces are transforming applications ranging from neuroscience research to diagnostic tools (for mental state recognition, tremor and seizure detection) as well as prosthetic devices (for motor and communication recovery). By integrating complex functions into miniaturized neural devices, these systems unlock significant opportunities for personalized assistive technologies and adaptive therapeutic interventions. Leveraging high-density neural recordings, on-site signal processing, and machine learning (ML), these interfaces extract critical features, identify disease neuro-markers, and enable accurate, low-latency neural decoding. This integration facilitates real-time interpretation of neural signals, adaptive modulation of brain activity, and efficient control of assistive devices. Moreover, the synergy between neural interfaces and ML has paved the way for self-sufficient, ubiquitous platforms capable of operating in diverse environments with minimal hardware costs and external dependencies. In this work, we review recent advancements in AI-driven decoding algorithms and energy-efficient System-on-Chip (SoC) platforms for next-generation miniaturized neural devices. These innovations highlight the potential for developing intelligent neural interfaces, addressing critical challenges in scalability, reliability, interpretability, and user adaptability. 

**Abstract (ZH)**: 先进神经接口正在变革从神经科学研究到诊断工具（如心理状态识别、震颤和癫痫检测）以及仿生装置（如运动和通信恢复）的广泛应用。通过将复杂功能集成到微型神经设备中，这些系统为个性化的辅助技术和适应性治疗干预打开了重要机会。借助高密度神经记录、现场信号处理和机器学习（ML），这些接口提取关键特征、识别疾病生物标志物，并实现准确、低延迟的神经解码。这种集成促进了实时解读神经信号、适应性调节脑活动以及高效控制辅助设备。此外，神经接口与机器学习的协同作用为能够在多样环境中独立运作、硬件成本低且依赖外部设备少的平台铺平了道路。在本文中，我们回顾了旨在推动下一代微型神经设备发展的AI驱动解码算法和能效系统级芯片（SoC）平台的最新进展。这些创新突显了开发智能神经接口的潜力，旨在解决扩展性、可靠性和可解释性等关键挑战，同时提高用户适应性。 

---
# Real-time Spatial Retrieval Augmented Generation for Urban Environments 

**Title (ZH)**: 实时空间检索增强生成的城市环境模型 

**Authors**: David Nazareno Campo, Javier Conde, Álvaro Alonso, Gabriel Huecas, Joaquín Salvachúa, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2505.02271)  

**Abstract**: The proliferation of Generative Artificial Ingelligence (AI), especially Large Language Models, presents transformative opportunities for urban applications through Urban Foundation Models. However, base models face limitations, as they only contain the knowledge available at the time of training, and updating them is both time-consuming and costly. Retrieval Augmented Generation (RAG) has emerged in the literature as the preferred approach for injecting contextual information into Foundation Models. It prevails over techniques such as fine-tuning, which are less effective in dynamic, real-time scenarios like those found in urban environments. However, traditional RAG architectures, based on semantic databases, knowledge graphs, structured data, or AI-powered web searches, do not fully meet the demands of urban contexts. Urban environments are complex systems characterized by large volumes of interconnected data, frequent updates, real-time processing requirements, security needs, and strong links to the physical world. This work proposes a real-time spatial RAG architecture that defines the necessary components for the effective integration of generative AI into cities, leveraging temporal and spatial filtering capabilities through linked data. The proposed architecture is implemented using FIWARE, an ecosystem of software components to develop smart city solutions and digital twins. The design and implementation are demonstrated through the use case of a tourism assistant in the city of Madrid. The use case serves to validate the correct integration of Foundation Models through the proposed RAG architecture. 

**Abstract (ZH)**: 生成型人工智能（AI）的快速发展，尤其是大型语言模型，为城市应用通过城市基础模型提供了变革性机会。然而，基础模型面临局限性，因为它们仅包含训练时可用的知识，更新它们既耗时又昂贵。检索增强生成（RAG）已在文献中被提出作为向基础模型注入上下文信息的首选方法。在动态、实时场景如城市环境中的场景中，它优于如微调等较不有效的技术。然而，传统的RAG架构，基于语义数据库、知识图谱、结构化数据或基于AI的网络搜索，不能完全满足城市环境的需求。城市环境是复杂的系统，涵盖大量的互联数据、频繁的更新、实时处理需求、安全需求以及与物理世界的紧密联系。本文提出了一种实时空间RAG架构，定义了将生成型AI有效集成到城市中的必要组件，并通过链接数据利用了时间空间过滤能力。该提出的架构使用FIWARE生态系统开发智能城市解决方案和数字孪生。通过在马德里市旅游助手用例中的设计和实现，展示了该架构的可行性和有效性，以验证所提RAG架构中基础模型的正确集成。 

---
# Leveraging LLM Agents and Digital Twins for Fault Handling in Process Plants 

**Title (ZH)**: 利用大型语言模型代理和数字孪生进行过程 plant 故障处理 

**Authors**: Milapji Singh Gill, Javal Vyas, Artan Markaj, Felix Gehlhoff, Mehmet Mercangöz  

**Link**: [PDF](https://arxiv.org/pdf/2505.02076)  

**Abstract**: Advances in Automation and Artificial Intelligence continue to enhance the autonomy of process plants in handling various operational scenarios. However, certain tasks, such as fault handling, remain challenging, as they rely heavily on human expertise. This highlights the need for systematic, knowledge-based methods. To address this gap, we propose a methodological framework that integrates Large Language Model (LLM) agents with a Digital Twin environment. The LLM agents continuously interpret system states and initiate control actions, including responses to unexpected faults, with the goal of returning the system to normal operation. In this context, the Digital Twin acts both as a structured repository of plant-specific engineering knowledge for agent prompting and as a simulation platform for the systematic validation and verification of the generated corrective control actions. The evaluation using a mixing module of a process plant demonstrates that the proposed framework is capable not only of autonomously controlling the mixing module, but also of generating effective corrective actions to mitigate a pipe clogging with only a few reprompts. 

**Abstract (ZH)**: 自动化和人工智能的进步持续增强过程 plant 的自主处理各种操作场景的能力。然而，某些任务，如故障处理，仍然具有挑战性，因为它们高度依赖于人类专业知识。这突显了需要系统化、基于知识的方法。为此，我们提出了一种方法论框架，将大型语言模型（LLM）代理与数字孪生环境集成。LLM 代理持续解释系统状态并发起控制动作，包括对意外故障的响应，目标是使系统恢复正常运行。在此背景下，数字孪生既作为 plant 特定工程知识的结构化存储库，用于代理提示，又作为生成的纠正控制动作的系统验证和验证的仿真平台。通过过程 plant 的混合模块评估表明，所提出框架不仅能自主控制混合模块，还能通过少量重询生成有效的纠正动作来缓解管道堵塞。 

---
# From Mind to Machine: The Rise of Manus AI as a Fully Autonomous Digital Agent 

**Title (ZH)**: 从思维到机器：Manus AI作为全自主数字代理的崛起 

**Authors**: Minjie Shen, Qikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02024)  

**Abstract**: Manus AI is a general-purpose AI agent introduced in early 2025, marking a significant advancement in autonomous artificial intelligence. Developed by the Chinese startup this http URL, Manus is designed to bridge the gap between "mind" and "hand" - combining the reasoning and planning capabilities of large language models with the ability to execute complex, end-to-end tasks that produce tangible outcomes. This paper presents a comprehensive overview of Manus AI, exploring its core technical architecture, diverse applications across sectors such as healthcare, finance, manufacturing, robotics, and gaming, as well as its key strengths, current limitations, and future potential. Positioned as a preview of what lies ahead, Manus AI represents a shift toward intelligent agents that can translate high-level intentions into real-world actions, heralding a new era of human-AI collaboration. 

**Abstract (ZH)**: Manus AI是一种于2025年初推出的通用人工智能代理，标志着自主人工智能的重大进步。本文介绍了Manus AI的核心技术架构，探讨了其在医疗、金融、制造业、机器人技术和游戏等多个领域的广泛应用，以及其核心优势、当前局限性和未来潜力。Manus AI代表了智能代理向能够将高层意图转化为实际行动的方向转变，预示着人机协作新时代的到来。 

---
# Training Environment for High Performance Reinforcement Learning 

**Title (ZH)**: 高性能强化学习的训练环境 

**Authors**: Greg Search  

**Link**: [PDF](https://arxiv.org/pdf/2505.01953)  

**Abstract**: This paper presents Tunnel, a simple, open source, reinforcement learning training environment for high performance aircraft. It integrates the F16 3D nonlinear flight dynamics into OpenAI Gymnasium python package. The template includes primitives for boundaries, targets, adversaries and sensing capabilities that may vary depending on operational need. This offers mission planners a means to rapidly respond to evolving environments, sensor capabilities and adversaries for autonomous air combat aircraft. It offers researchers access to operationally relevant aircraft physics. Tunnel code base is accessible to anyone familiar with Gymnasium and/or those with basic python skills. This paper includes a demonstration of a week long trade study that investigated a variety of training methods, observation spaces, and threat presentations. This enables increased collaboration between researchers and mission planners which can translate to a national military advantage. As warfare becomes increasingly reliant upon automation, software agility will correlate with decision advantages. Airmen must have tools to adapt to adversaries in this context. It may take months for researchers to develop skills to customize observation, actions, tasks and training methodologies in air combat simulators. In Tunnel, this can be done in a matter of days. 

**Abstract (ZH)**: This paper presents Tunnel，一个简单、开源的高性能飞机强化学习训练环境，将其F16 3D非线性飞行动力学整合进OpenAI Gymnasium Python包。该模板包括边界、目标、对手和传感能力的原语，这些原语可根据操作需求有所不同。这为任务规划者提供了一种快速应对不断变化的环境、传感能力和对手的方法，以满足自主空战飞机的需求。该论文为研究人员提供了相关操作的飞机物理模型。Tunnel的代码库对任何熟悉Gymnasium或具有基本Python技能的人都开放。本文还演示了一个为期一周的权衡研究，探讨了多种训练方法、观察空间和威胁呈现方式。这使得研究人员和任务规划者之间的合作更加密切，可以转化为国家军事优势。随着战争越来越多地依赖自动化，软件敏捷性将与决策优势相关联。在这种背景下，航空人员必须具备能够适应对手的工具。研究人员可能需要数月时间来开发定制观察、动作、任务和训练方法的技能，而在Tunnel中，这一切可以在几天内完成。 

---
# Emotions in Artificial Intelligence 

**Title (ZH)**: 人工智能中的情感 

**Authors**: Hermann Borotschnig  

**Link**: [PDF](https://arxiv.org/pdf/2505.01462)  

**Abstract**: This conceptual contribution offers a speculative account of how AI systems might emulate emotions as experienced by humans and animals. It presents a thought experiment grounded in the hypothesis that natural emotions evolved as heuristics for rapid situational appraisal and action selection, enabling biologically adaptive behaviour without requiring full deliberative modeling. The text examines whether artificial systems operating in complex action spaces could similarly benefit from these principles. It is proposed that affect be interwoven with episodic memory by storing corresponding affective tags alongside all events. This allows AIs to establish whether present situations resemble past events and project the associated emotional labels onto the current context. These emotional cues are then combined with need-driven emotional hints. The combined emotional state facilitates decision-making in the present by modulating action selection. The low complexity and experiential inertness of the proposed architecture are emphasized as evidence that emotional expression and consciousness are, in principle, orthogonal-permitting the theoretical possibility of affective zombies. On this basis, the moral status of AIs emulating affective states is critically examined. It is argued that neither the mere presence of internal representations of emotion nor consciousness alone suffices for moral standing; rather, the capacity for self-awareness of inner emotional states is posited as a necessary condition. A complexity-based criterion is proposed to exclude such awareness in the presented model. Additional thought experiments are presented to test the conceptual boundaries of this framework. 

**Abstract (ZH)**: 本概念性贡献提供了一种推测性的阐释，探讨AI系统如何模仿人类和动物体验的情绪。它基于这样的假设：自然情绪在快速情境评估和行动选择中演化为启发式方法，从而在不需要完全 deliberative 模型的情况下促进生物适应性行为。文本考察了在复杂行动空间中运行的人工系统是否也能从中受益。提出了将情感与事件记忆结合的思路，通过存储对应的情感标签来记录所有事件，使AI能够判断当前情境是否类似于过去事件，并将相关的情绪标签投射到当前情景中。这些情感线索随后与需求驱动的情感提示相结合。这种组合的情绪状态通过调节行动选择来促进当前决策。强调所提出的架构的低复杂性和经验惰性，表明情感表达和意识原则上是独立的，允许情感僵尸这一理论可能性存在。在此基础上，重新审视模仿情感状态的AI的道德地位。提出，内部情感表示的存在或意识本身不足以构成道德地位，而是内在情感状态自我意识的能力被提出为必要条件。提出了基于复杂性标准的准则，排除该模型中的这种意识。还提出了进一步的思辨实验来测试该框架的边界。 

---
# Coupled Distributional Random Expert Distillation for World Model Online Imitation Learning 

**Title (ZH)**: 耦合分布随机专家蒸馏用于世界模型在线模仿学习 

**Authors**: Shangzhe Li, Zhiao Huang, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.02228)  

**Abstract**: Imitation Learning (IL) has achieved remarkable success across various domains, including robotics, autonomous driving, and healthcare, by enabling agents to learn complex behaviors from expert demonstrations. However, existing IL methods often face instability challenges, particularly when relying on adversarial reward or value formulations in world model frameworks. In this work, we propose a novel approach to online imitation learning that addresses these limitations through a reward model based on random network distillation (RND) for density estimation. Our reward model is built on the joint estimation of expert and behavioral distributions within the latent space of the world model. We evaluate our method across diverse benchmarks, including DMControl, Meta-World, and ManiSkill2, showcasing its ability to deliver stable performance and achieve expert-level results in both locomotion and manipulation tasks. Our approach demonstrates improved stability over adversarial methods while maintaining expert-level performance. 

**Abstract (ZH)**: 基于随机网络蒸馏的密度估计在线 imitation 学习 

---
# Think on your Feet: Adaptive Thinking via Reinforcement Learning for Social Agents 

**Title (ZH)**: 随机应变：通过强化学习的社会智能体自适应思考 

**Authors**: Minzheng Wang, Yongbin Li, Haobo Wang, Xinghua Zhang, Nan Xu, Bingli Wu, Fei Huang, Haiyang Yu, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.02156)  

**Abstract**: Effective social intelligence simulation requires language agents to dynamically adjust reasoning depth, a capability notably absent in current approaches. While existing methods either lack this kind of reasoning capability or enforce uniform long chain-of-thought reasoning across all scenarios, resulting in excessive token usage and inappropriate social simulation. In this paper, we propose $\textbf{A}$daptive $\textbf{M}$ode $\textbf{L}$earning ($\textbf{AML}$) that strategically selects from four thinking modes (intuitive reaction $\rightarrow$ deep contemplation) based on real-time context. Our framework's core innovation, the $\textbf{A}$daptive $\textbf{M}$ode $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{AMPO}$) algorithm, introduces three key advancements over existing methods: (1) Multi-granular thinking mode design, (2) Context-aware mode switching across social interaction, and (3) Token-efficient reasoning via depth-adaptive processing. Extensive experiments on social intelligence tasks confirm that AML achieves 15.6% higher task performance than state-of-the-art methods. Notably, our method outperforms GRPO by 7.0% with 32.8% shorter reasoning chains. These results demonstrate that context-sensitive thinking mode selection, as implemented in AMPO, enables more human-like adaptive reasoning than GRPO's fixed-depth approach 

**Abstract (ZH)**: 适配模式学习（AML）：社交智能模拟中的动态推理深度优化 

---
# Skill-based Safe Reinforcement Learning with Risk Planning 

**Title (ZH)**: 基于技能的风险规划安全强化学习 

**Authors**: Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01619)  

**Abstract**: Safe Reinforcement Learning (Safe RL) aims to ensure safety when an RL agent conducts learning by interacting with real-world environments where improper actions can induce high costs or lead to severe consequences. In this paper, we propose a novel Safe Skill Planning (SSkP) approach to enhance effective safe RL by exploiting auxiliary offline demonstration data. SSkP involves a two-stage process. First, we employ PU learning to learn a skill risk predictor from the offline demonstration data. Then, based on the learned skill risk predictor, we develop a novel risk planning process to enhance online safe RL and learn a risk-averse safe policy efficiently through interactions with the online RL environment, while simultaneously adapting the skill risk predictor to the environment. We conduct experiments in several benchmark robotic simulation environments. The experimental results demonstrate that the proposed approach consistently outperforms previous state-of-the-art safe RL methods. 

**Abstract (ZH)**: 安全强化学习（安全RL）旨在确保当RL代理通过与具有潜在高成本或严重后果的现实世界环境交互进行学习时的安全性。在本文中，我们提出了一种新颖的安全技能规划（SSkP）方法，通过利用辅助离线演示数据来增强有效的安全RL。SSkP包括两阶段过程。首先，我们采用PU学习从离线演示数据中学习技能风险预测器。然后，基于学习到的技能风险预测器，我们开发了一种新颖的风险规划过程，以增强在线安全RL并高效地通过与在线RL环境的交互学习风险规避的安全策略，同时适应环境。我们在多个基准机器人仿真环境中进行了实验。实验结果表明，所提出的方法的一致性优于以前的最先进的安全RL方法。 

---
# PIPA: A Unified Evaluation Protocol for Diagnosing Interactive Planning Agents 

**Title (ZH)**: PIPA：用于诊断交互式规划代理的一体化评估协议 

**Authors**: Takyoung Kim, Janvijay Singh, Shuhaib Mehri, Emre Can Acikgoz, Sagnik Mukherjee, Nimet Beyza Bozdag, Sumuk Shashidhar, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2505.01592)  

**Abstract**: The growing capabilities of large language models (LLMs) in instruction-following and context-understanding lead to the era of agents with numerous applications. Among these, task planning agents have become especially prominent in realistic scenarios involving complex internal pipelines, such as context understanding, tool management, and response generation. However, existing benchmarks predominantly evaluate agent performance based on task completion as a proxy for overall effectiveness. We hypothesize that merely improving task completion is misaligned with maximizing user satisfaction, as users interact with the entire agentic process and not only the end result. To address this gap, we propose PIPA, a unified evaluation protocol that conceptualizes the behavioral process of interactive task planning agents within a partially observable Markov Decision Process (POMDP) paradigm. The proposed protocol offers a comprehensive assessment of agent performance through a set of atomic evaluation criteria, allowing researchers and practitioners to diagnose specific strengths and weaknesses within the agent's decision-making pipeline. Our analyses show that agents excel in different behavioral stages, with user satisfaction shaped by both outcomes and intermediate behaviors. We also highlight future directions, including systems that leverage multiple agents and the limitations of user simulators in task planning. 

**Abstract (ZH)**: 大语言模型（LLMs）在指令遵循和语境理解能力的增长促使了具有广泛应用的智能代理时代的到来。其中，任务规划代理在涉及复杂内部管道的现实场景中尤为突出，如语境理解、工具管理和响应生成。然而，现有基准主要基于任务完成度来代理整体有效性来评估代理性能。我们假设仅仅提高任务完成度并不能最大化用户满意度，因为用户与整个代理过程互动，而不仅仅是最终结果。为解决这一问题，我们提出了PIPA，一种统一的评估协议，通过部分可观测马尔可夫决策过程（POMDP）范式来概念化交互式任务规划代理的行为过程。该提出的协议通过一系列原子评估标准提供了一个全面的代理性能评估，使研究人员和实践者能够诊断代理决策管道中的具体优势和弱点。我们的分析表明，代理在不同的行为阶段表现出色，用户满意度受到最终结果和中间行为的共同影响。我们还指出了未来的研究方向，包括利用多个代理的系统以及任务规划中用户模拟器的局限性。 

---
# Emotions in the Loop: A Survey of Affective Computing for Emotional Support 

**Title (ZH)**: 情绪在环中：情感计算在情感支持中的综述 

**Authors**: Karishma Hegde, Hemadri Jayalath  

**Link**: [PDF](https://arxiv.org/pdf/2505.01542)  

**Abstract**: In a world where technology is increasingly embedded in our everyday experiences, systems that sense and respond to human emotions are elevating digital interaction. At the intersection of artificial intelligence and human-computer interaction, affective computing is emerging with innovative solutions where machines are humanized by enabling them to process and respond to user emotions. This survey paper explores recent research contributions in affective computing applications in the area of emotion recognition, sentiment analysis and personality assignment developed using approaches like large language models (LLMs), multimodal techniques, and personalized AI systems. We analyze the key contributions and innovative methodologies applied by the selected research papers by categorizing them into four domains: AI chatbot applications, multimodal input systems, mental health and therapy applications, and affective computing for safety applications. We then highlight the technological strengths as well as the research gaps and challenges related to these studies. Furthermore, the paper examines the datasets used in each study, highlighting how modality, scale, and diversity impact the development and performance of affective models. Finally, the survey outlines ethical considerations and proposes future directions to develop applications that are more safe, empathetic and practical. 

**Abstract (ZH)**: 在技术日益融入我们日常生活体验的世界中，能够感知和响应人类情绪的系统正在提升数字交互的质量。人工智能与人机交互的交叉领域中，情感计算正在涌现新的解决方案，通过使机器能够处理和响应用户情绪而实现人性化。本文综述探讨了情感计算在情绪识别、情感分析和人格赋值方面的最新研究贡献，这些贡献基于大型语言模型（LLMs）、多模态技术以及个性化AI系统的方法。我们将所选研究论文的关键贡献和创新方法学按四个领域进行分类：AI聊天机器人应用、多模态输入系统、心理健康和治疗应用以及情感计算的安全应用领域。然后，本文还突出了这些研究的技术优势以及相关的研究空白和挑战。此外，文章还分析了每项研究中使用的数据集，强调了模态性、规模和多样性对情感模型的开发和性能的影响。最后，综述列出了伦理考量，并提出了未来发展方向，以开发更加安全、移情和实用的应用。 

---
