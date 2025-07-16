# LLM-based ambiguity detection in natural language instructions for collaborative surgical robots 

**Title (ZH)**: 基于LLM的自然语言指令中模糊性的检测方法用于协作手术机器人 

**Authors**: Ana Davila, Jacinto Colan, Yasuhisa Hasegawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.11525)  

**Abstract**: Ambiguity in natural language instructions poses significant risks in safety-critical human-robot interaction, particularly in domains such as surgery. To address this, we propose a framework that uses Large Language Models (LLMs) for ambiguity detection specifically designed for collaborative surgical scenarios. Our method employs an ensemble of LLM evaluators, each configured with distinct prompting techniques to identify linguistic, contextual, procedural, and critical ambiguities. A chain-of-thought evaluator is included to systematically analyze instruction structure for potential issues. Individual evaluator assessments are synthesized through conformal prediction, which yields non-conformity scores based on comparison to a labeled calibration dataset. Evaluating Llama 3.2 11B and Gemma 3 12B, we observed classification accuracy exceeding 60% in differentiating ambiguous from unambiguous surgical instructions. Our approach improves the safety and reliability of human-robot collaboration in surgery by offering a mechanism to identify potentially ambiguous instructions before robot action. 

**Abstract (ZH)**: 自然语言指令的歧义性在安全关键的人机交互中尤其是在手术领域中构成了重大风险。为此，我们提出了一种框架，利用大型语言模型（LLMs）进行专门针对协作手术场景的歧义检测。该方法采用由具有不同提示技术的LLM评估器组成的集成系统，以识别语言、上下文、程序和关键性歧义。包含一种推理链评估器，用于系统地分析指令结构以识别潜在问题。通过确立性预测综合各个评估器的评估结果，基于与标记校准数据集的比较，生成非协合得分。在对Llama 3.2 11B和Gemma 3 12B进行评估后，我们发现它们在区分手术指令的歧义性和非歧义性方面达到了超过60%的分类准确率。该方法通过在机器人行动前识别潜在的歧义指令，提高了手术中的人机协作的安全性和可靠性。 

---
# Robot Drummer: Learning Rhythmic Skills for Humanoid Drumming 

**Title (ZH)**: 机器人鼓手：学习类人鼓击节奏技能 

**Authors**: Asad Ali Shahid, Francesco Braghin, Loris Roveda  

**Link**: [PDF](https://arxiv.org/pdf/2507.11498)  

**Abstract**: Humanoid robots have seen remarkable advances in dexterity, balance, and locomotion, yet their role in expressive domains, such as music performance, remains largely unexplored. Musical tasks, like drumming, present unique challenges, including split-second timing, rapid contacts, and multi-limb coordination over pieces lasting minutes. In this paper, we introduce Robot Drummer, a humanoid system capable of expressive, high-precision drumming across a diverse repertoire of songs. We formulate humanoid drumming as sequential fulfillment of timed-contacts and transform drum scores in to a Rhythmic Contact Chain. To handle the long-horizon nature of musical performance, we decompose each piece into fixed-length segments and train a single policy across all segments in parallel using reinforcement learning. Through extensive experiments on over thirty popular rock, metal, and jazz tracks, our results demonstrate that Robot Drummer consistently achieves high F1 scores. The learned behaviors exhibit emergent human-like drumming strategies, such as cross-arm strikes, and adaptive sticks assignments, demonstrating the potential of reinforcement learning to bring humanoid robots into the domain of creative musical performance. Project page: \href{this https URL}{this http URL} 

**Abstract (ZH)**: 人类机器人在灵巧性、平衡性和移动性方面取得了显著进步，但在像音乐表演这样的表现领域中的作用仍然 largely unexplored。音乐任务，例如打鼓，提出了独特的挑战，包括毫秒级的节拍精度、快速接触以及多肢体在分钟级曲子中的协调。在本文中，我们介绍了机器人鼓手（Robot Drummer），这是一个能够在多种歌曲曲目中执行表达性、高精度打鼓的人形系统。我们将人形打鼓表述为按时间顺序完成接触任务，并将鼓谱转换为节律接触链。为了应对音乐表演的长时间跨度特性，我们将每首曲子分解为固定长度的段落，并利用强化学习在并行训练过程中训练一个单一的策略。通过对三十多首流行摇滚、金属和爵士乐曲的广泛实验，我们的结果表明，机器人鼓手能够一致地实现高F1分数。学习到的行为表现出类似于人类的鼓击策略，如交叉臂击打和自适应的棒分配，这展示了强化学习将人形机器人引入创意音乐表演领域的潜力。项目页面：[这个链接](这个链接)。 

---
# LF: Online Multi-Robot Path Planning Meets Optimal Trajectory Control 

**Title (ZH)**: LF: 在线多机器人路径规划与最优轨迹控制 

**Authors**: Ajay Shankar, Keisuke Okumura, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2507.11464)  

**Abstract**: We propose a multi-robot control paradigm to solve point-to-point navigation tasks for a team of holonomic robots with access to the full environment information. The framework invokes two processes asynchronously at high frequency: (i) a centralized, discrete, and full-horizon planner for computing collision- and deadlock-free paths rapidly, leveraging recent advances in multi-agent pathfinding (MAPF), and (ii) dynamics-aware, robot-wise optimal trajectory controllers that ensure all robots independently follow their assigned paths reliably. This hierarchical shift in planning representation from (i) discrete and coupled to (ii) continuous and decoupled domains enables the framework to maintain long-term scalable motion synthesis. As an instantiation of this idea, we present LF, which combines a fast state-of-the-art MAPF solver (LaCAM), and a robust feedback control stack (Freyja) for executing agile robot maneuvers. LF provides a robust and versatile mechanism for lifelong multi-robot navigation even under asynchronous and partial goal updates, and adapts to dynamic workspaces simply by quick replanning. We present various multirotor and ground robot demonstrations, including the deployment of 15 real multirotors with random, consecutive target updates while a person walks through the operational workspace. 

**Abstract (ZH)**: 我们提出了一种多机器人控制范式，用于具有全域环境信息的一组全向机器人的点对点导航任务。该框架以高频率异步调用两个过程：(i) 中心化、离散、全视野的规划器快速计算无碰撞和无死锁路径，充分利用多智能体路径寻找（MAPF）领域的最新进展；(ii) 动力学意识的、针对每个机器人的最优轨迹控制器，确保所有机器人独立可靠地遵循其分配的路径。这种从(i) 离散和耦合到(ii) 连续和解耦规划表示层次的转变，使框架能够保持长期可扩展的运动合成。作为这一思想的具体实例，我们介绍了LF，它结合了快速最先进的MAPF求解器（LaCAM）和鲁棒的反馈控制堆栈（Freyja），用于执行敏捷的机器人机动。LF为在异步和部分目标更新下提供了稳健且多功能的终身多机器人导航机制，并通过快速重新规划简单适应动态工作空间。我们展示了各种多旋翼和地面机器人的演示，包括在人员穿越操作工作空间时，部署15个实时多旋翼并随机更新目标的场景。 

---
# Human-Robot collaboration in surgery: Advances and challenges towards autonomous surgical assistants 

**Title (ZH)**: 手术中的人机协作：自主外科助手的发展与挑战 

**Authors**: Jacinto Colan, Ana Davila, Yutaro Yamada, Yasuhisa Hasegawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.11460)  

**Abstract**: Human-robot collaboration in surgery represents a significant area of research, driven by the increasing capability of autonomous robotic systems to assist surgeons in complex procedures. This systematic review examines the advancements and persistent challenges in the development of autonomous surgical robotic assistants (ASARs), focusing specifically on scenarios where robots provide meaningful and active support to human surgeons. Adhering to the PRISMA guidelines, a comprehensive literature search was conducted across the IEEE Xplore, Scopus, and Web of Science databases, resulting in the selection of 32 studies for detailed analysis. Two primary collaborative setups were identified: teleoperation-based assistance and direct hands-on interaction. The findings reveal a growing research emphasis on ASARs, with predominant applications currently in endoscope guidance, alongside emerging progress in autonomous tool manipulation. Several key challenges hinder wider adoption, including the alignment of robotic actions with human surgeon preferences, the necessity for procedural awareness within autonomous systems, the establishment of seamless human-robot information exchange, and the complexities of skill acquisition in shared workspaces. This review synthesizes current trends, identifies critical limitations, and outlines future research directions essential to improve the reliability, safety, and effectiveness of human-robot collaboration in surgical environments. 

**Abstract (ZH)**: 手术中的人机协作代表了一个重要的研究领域，由自主机器人系统能力的提升推动，这些系统能够协助外科医生进行复杂的手术程序。本系统综述基于PRISMA指南，考察了自主手术机器人助手（ASARs）的发展进步和持续挑战，重点关注机器人为外科医生提供有意义和主动支持的场景。研究识别了两种主要的协作模式：基于遥控的操作协助和直接的手动交互。研究结果揭示了对ASARs研究重点的增长，目前主要应用在内窥镜引导，并且在自主工具操作方面取得了新兴进展。阻碍更广泛采用的关键挑战包括机器人动作与外科医生偏好的对齐、自主系统内的程序意识需求、人机信息交换的无缝链接，以及共享工作空间中技能获取的复杂性。本综述汇总了当前趋势，指出了关键限制，并勾勒了未来研究方向，以提高手术环境中人机协作的可靠性和有效性。 

---
# Multi-IMU Sensor Fusion for Legged Robots 

**Title (ZH)**: 多IMU传感器融合在腿式机器人中的应用 

**Authors**: Shuo Yang, John Z. Zhang, Ibrahima Sory Sow, Zachary Manchester  

**Link**: [PDF](https://arxiv.org/pdf/2507.11447)  

**Abstract**: This paper presents a state-estimation solution for legged robots that uses a set of low-cost, compact, and lightweight sensors to achieve low-drift pose and velocity estimation under challenging locomotion conditions. The key idea is to leverage multiple inertial measurement units on different links of the robot to correct a major error source in standard proprioceptive odometry. We fuse the inertial sensor information and joint encoder measurements in an extended Kalman filter, then combine the velocity estimate from this filter with camera data in a factor-graph-based sliding-window estimator to form a visual-inertial-leg odometry method. We validate our state estimator through comprehensive theoretical analysis and hardware experiments performed using real-world robot data collected during a variety of challenging locomotion tasks. Our algorithm consistently achieves minimal position deviation, even in scenarios involving substantial ground impact, foot slippage, and sudden body rotations. A C++ implementation, along with a large-scale dataset, is available at this https URL. 

**Abstract (ZH)**: 本文提出了一种针对腿式机器人状态估计的解决方案，该方案利用一组低成本、紧凑型和轻量级传感器，在挑战性运动条件下实现低漂移姿态和速度估计。核心思想是利用机器人不同链接上的多个惯性测量单元来修正标准本体感觉里程计的主要误差源。我们通过扩展卡尔曼滤波器融合惯性传感器信息和关节编码器测量值，然后将该滤波器的速度估计值与基于因子图的滑动窗口估计算法结合，形成一种视觉-惯性-腿式里程计方法。我们通过全面的理论分析和使用多种挑战性运动任务中收集的真实机器人数据进行的硬件实验，验证了我们状态估计器的有效性。即使在涉及显著地面冲击、脚底打滑和身体突然旋转的场景中，该算法也能够实现最小的位置偏差。相关C++实现及大规模数据集可从此链接获取。 

---
# From Production Logistics to Smart Manufacturing: The Vision for a New RoboCup Industrial League 

**Title (ZH)**: 从生产物流到智能制造：新机器人世界杯工业联赛的愿景 

**Authors**: Supun Dissanayaka, Alexander Ferrein, Till Hofmann, Kosuke Nakajima, Mario Sanz-Lopez, Jesus Savage, Daniel Swoboda, Matteo Tschesche, Wataru Uemura, Tarik Viehmann, Shohei Yasuda  

**Link**: [PDF](https://arxiv.org/pdf/2507.11402)  

**Abstract**: The RoboCup Logistics League is a RoboCup competition in a smart factory scenario that has focused on task planning, job scheduling, and multi-agent coordination. The focus on production logistics allowed teams to develop highly competitive strategies, but also meant that some recent developments in the context of smart manufacturing are not reflected in the competition, weakening its relevance over the years. In this paper, we describe the vision for the RoboCup Smart Manufacturing League, a new competition designed as a larger smart manufacturing scenario, reflecting all the major aspects of a modern factory. It will consist of several tracks that are initially independent but gradually combined into one smart manufacturing scenario. The new tracks will cover industrial robotics challenges such as assembly, human-robot collaboration, and humanoid robotics, but also retain a focus on production logistics. We expect the reenvisioned competition to be more attractive to newcomers and well-tried teams, while also shifting the focus to current and future challenges of industrial robotics. 

**Abstract (ZH)**: RoboCup智能制造联盟：一个新的涵盖现代工厂各方面挑战的竞赛愿景 

---
# Acting and Planning with Hierarchical Operational Models on a Mobile Robot: A Study with RAE+UPOM 

**Title (ZH)**: 基于RAE+UPOM的移动机器人层次操作模型的执行与规划研究 

**Authors**: Oscar Lima, Marc Vinci, Sunandita Patra, Sebastian Stock, Joachim Hertzberg, Martin Atzmueller, Malik Ghallab, Dana Nau, Paolo Traverso  

**Link**: [PDF](https://arxiv.org/pdf/2507.11345)  

**Abstract**: Robotic task execution faces challenges due to the inconsistency between symbolic planner models and the rich control structures actually running on the robot. In this paper, we present the first physical deployment of an integrated actor-planner system that shares hierarchical operational models for both acting and planning, interleaving the Reactive Acting Engine (RAE) with an anytime UCT-like Monte Carlo planner (UPOM). We implement RAE+UPOM on a mobile manipulator in a real-world deployment for an object collection task. Our experiments demonstrate robust task execution under action failures and sensor noise, and provide empirical insights into the interleaved acting-and-planning decision making process. 

**Abstract (ZH)**: 由于符号计划模型与机器人实际运行的丰富控制结构之间的一致性问题，机器人任务执行面临着挑战。本文呈现了首个将层级操作模型同时应用于执行和计划的集成执行-计划系统在真实世界的物理部署。我们在一台移动 manipulator 上实现了 RAE+UPOM，用于物体收集任务。我们的实验展示了在动作失败和传感器噪声情况下的鲁棒任务执行，并提供了交错执行与规划决策过程的实证见解。 

---
# All Eyes, no IMU: Learning Flight Attitude from Vision Alone 

**Title (ZH)**: 全视觉，无IMU：仅从视觉学习飞行姿态 

**Authors**: Jesse J. Hagenaars, Stein Stroobants, Sander M. Bohte, Guido C.H.E. De Croon  

**Link**: [PDF](https://arxiv.org/pdf/2507.11302)  

**Abstract**: Vision is an essential part of attitude control for many flying animals, some of which have no dedicated sense of gravity. Flying robots, on the other hand, typically depend heavily on accelerometers and gyroscopes for attitude stabilization. In this work, we present the first vision-only approach to flight control for use in generic environments. We show that a quadrotor drone equipped with a downward-facing event camera can estimate its attitude and rotation rate from just the event stream, enabling flight control without inertial sensors. Our approach uses a small recurrent convolutional neural network trained through supervised learning. Real-world flight tests demonstrate that our combination of event camera and low-latency neural network is capable of replacing the inertial measurement unit in a traditional flight control loop. Furthermore, we investigate the network's generalization across different environments, and the impact of memory and different fields of view. While networks with memory and access to horizon-like visual cues achieve best performance, variants with a narrower field of view achieve better relative generalization. Our work showcases vision-only flight control as a promising candidate for enabling autonomous, insect-scale flying robots. 

**Abstract (ZH)**: 视觉导向的飞行控制：一种用于通用环境的纯视觉方法 

---
# Diffusion-Based Imaginative Coordination for Bimanual Manipulation 

**Title (ZH)**: 基于扩散的想象性协调实现双臂操作 

**Authors**: Huilin Xu, Jian Ding, Jiakun Xu, Ruixiang Wang, Jun Chen, Jinjie Mai, Yanwei Fu, Bernard Ghanem, Feng Xu, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2507.11296)  

**Abstract**: Bimanual manipulation is crucial in robotics, enabling complex tasks in industrial automation and household services. However, it poses significant challenges due to the high-dimensional action space and intricate coordination requirements. While video prediction has been recently studied for representation learning and control, leveraging its ability to capture rich dynamic and behavioral information, its potential for enhancing bimanual coordination remains underexplored. To bridge this gap, we propose a unified diffusion-based framework for the joint optimization of video and action prediction. Specifically, we propose a multi-frame latent prediction strategy that encodes future states in a compressed latent space, preserving task-relevant features. Furthermore, we introduce a unidirectional attention mechanism where video prediction is conditioned on the action, while action prediction remains independent of video prediction. This design allows us to omit video prediction during inference, significantly enhancing efficiency. Experiments on two simulated benchmarks and a real-world setting demonstrate a significant improvement in the success rate over the strong baseline ACT using our method, achieving a \textbf{24.9\%} increase on ALOHA, an \textbf{11.1\%} increase on RoboTwin, and a \textbf{32.5\%} increase in real-world experiments. Our models and code are publicly available at this https URL. 

**Abstract (ZH)**: 双臂 manipulation 对机器人技术至关重要，能够在工业自动化和家庭服务中执行复杂任务。然而，由于高维动作空间和复杂的协调要求，它提出了重大挑战。虽然视频预测已被研究用于表示学习和控制，并利用其捕捉丰富动态和行为信息的能力，但其在提升双臂协调方面的潜力尚未得到充分开发。为了解决这一问题，我们提出了一种统一的基于扩散的框架，用于联合优化视频和动作预测。具体来说，我们提出了一种多帧潜在预测策略，将未来状态编码在一个压缩的潜在空间中，保留任务相关特征。此外，我们引入了一种单向注意力机制，其中视频预测依赖于动作，而动作预测与视频预测独立。这种设计允许我们在推理过程中省略视频预测，显著提高效率。在两个模拟基准和一个实际场景中的实验表明，与强大的基线ACT相比，我们的方法显著提高了成功率，在ALOHA中提高了24.9%，在RoboTwin中提高了11.1%，在实际实验中提高了32.5%。我们的模型和代码已在以下网址公开。 

---
# Ocean Diviner: A Diffusion-Augmented Reinforcement Learning for AUV Robust Control in the Underwater Tasks 

**Title (ZH)**: Ocean Diviner: 基于扩散增强 reinforcement learning的自治水下车辆水下任务稳健控制方法 

**Authors**: Weiyi Liu, Jingzehua Xu, Guanwen Xie, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.11283)  

**Abstract**: This paper presents a diffusion-augmented reinforcement learning (RL) approach for robust autonomous underwater vehicle (AUV) control, addressing key challenges in underwater trajectory planning and dynamic environment adaptation. The proposed method integrates three core innovations: (1) A diffusion-based trajectory generation framework that produces physically feasible multi-step trajectories, enhanced by a high-dimensional state encoding mechanism combining current observations with historical states and actions through a novel diffusion U-Net architecture, significantly improving long-horizon planning. (2) A sample-efficient hybrid learning architecture that synergizes diffusion-guided exploration with RL policy optimization, where the diffusion model generates diverse candidate actions and the RL critic selects optimal actions, achieving higher exploration efficiency and policy stability in dynamic underwater environments. Extensive simulation experiments validating the method's superior robustness and flexibility, outperforms conventional control methods in challenging marine conditions, offering enhanced adaptability and reliability for AUV operations in the underwater tasks. 

**Abstract (ZH)**: 本文提出了一种扩散增强 reinforcement learning (RL) 方法，用于鲁棒自主水下车辆 (AUV) 控制，解决水下轨迹规划和动态环境适应的关键挑战。该提出的办法整合了三项核心创新：（1）一种基于扩散的轨迹生成框架，能够生成物理上可行的多步轨迹，并通过新颖的扩散 U-Net 架构结合当前观察和历史状态与动作的高维状态编码机制显著提高长远规划能力。（2）一种样本高效混合学习架构，协同利用扩散引导探索与 RL 策略优化，其中扩散模型生成多样化的候选动作，RL 评论家选择最优动作，在动态水下环境中实现更高的探索效率和策略稳定性。广泛的仿真实验验证了该方法的优越鲁棒性和灵活性，在恶劣海洋条件下优于传统控制方法，为 AUV 在水下任务中的操作提供增强的适应性和可靠性。 

---
# Development of an Autonomous Mobile Robotic System for Efficient and Precise Disinfection 

**Title (ZH)**: 自主移动机器人系统的发展以实现高效精准的消毒作业 

**Authors**: Ting-Wei Ou, Jia-Hao Jiang, Guan-Lin Huang, Kuu-Young Young  

**Link**: [PDF](https://arxiv.org/pdf/2507.11270)  

**Abstract**: The COVID-19 pandemic has severely affected public health, healthcare systems, and daily life, especially amid resource shortages and limited workers. This crisis has underscored the urgent need for automation in hospital environments, particularly disinfection, which is crucial to controlling virus transmission and improving the safety of healthcare personnel and patients. Ultraviolet (UV) light disinfection, known for its high efficiency, has been widely adopted in hospital settings. However, most existing research focuses on maximizing UV coverage while paying little attention to the impact of human activity on virus distribution. To address this issue, we propose a mobile robotic system for UV disinfection focusing on the virus hotspot. The system prioritizes disinfection in high-risk areas and employs an approach for optimized UV dosage to ensure that all surfaces receive an adequate level of UV exposure while significantly reducing disinfection time. It not only improves disinfection efficiency but also minimizes unnecessary exposure in low-risk areas. In two representative hospital scenarios, our method achieves the same disinfection effectiveness while reducing disinfection time by 30.7% and 31.9%, respectively. The video of the experiment is available at: this https URL. 

**Abstract (ZH)**: COVID-19 pandemic对公共健康、医疗系统和日常生活造成了严重的影响，尤其是在资源短缺和人手有限的情况下。这场危机凸显了医院环境中自动化需求的紧迫性，尤其是消毒工作，这在控制病毒传播和提高医疗人员和患者的安全方面至关重要。紫外（UV）光消毒以其高效的特性在医院环境中被广泛应用。然而，现有大多数研究侧重于最大化UV覆盖范围，而很少关注人类活动对病毒分布的影响。为解决这一问题，我们提出了一种移动机器人系统以针对病毒热点区域进行UV消毒。该系统优先对高风险区域进行消毒，并采用优化的UV剂量方法，以确保所有表面接受到足够的UV照射，同时显著减少消毒时间。它不仅提高了消毒效率，还最大限度地减少了低风险区域不必要的暴露。在两个代表性医院场景中，我们的方法在分别减少30.7%和31.9%的消毒时间的同时实现了相同的消毒效果。实验视频可在此网址查看：this https URL。 

---
# Comparison of Localization Algorithms between Reduced-Scale and Real-Sized Vehicles Using Visual and Inertial Sensors 

**Title (ZH)**: 使用视觉和惯性传感器在缩小比例模型和真实尺寸车辆之间比较定位算法性能 

**Authors**: Tobias Kern, Leon Tolksdorf, Christian Birkner  

**Link**: [PDF](https://arxiv.org/pdf/2507.11241)  

**Abstract**: Physically reduced-scale vehicles are emerging to accelerate the development of advanced automated driving functions. In this paper, we investigate the effects of scaling on self-localization accuracy with visual and visual-inertial algorithms using cameras and an inertial measurement unit (IMU). For this purpose, ROS2-compatible visual and visual-inertial algorithms are selected, and datasets are chosen as a baseline for real-sized vehicles. A test drive is conducted to record data of reduced-scale vehicles. We compare the selected localization algorithms, OpenVINS, VINS-Fusion, and RTAB-Map, in terms of their pose accuracy against the ground-truth and against data from real-sized vehicles. When comparing the implementation of the selected localization algorithms to real-sized vehicles, OpenVINS has the lowest average localization error. Although all selected localization algorithms have overlapping error ranges, OpenVINS also performs best when applied to a reduced-scale vehicle. When reduced-scale vehicles were compared to real-sized vehicles, minor differences were found in translational vehicle motion estimation accuracy. However, no significant differences were found when comparing the estimation accuracy of rotational vehicle motion, allowing RSVRs to be used as testing platforms for self-localization algorithms. 

**Abstract (ZH)**: 缩小比例的车辆正加速高级自动驾驶功能的发展。本文通过使用摄像头和惯性测量单元（IMU）的研究，考察缩放效应对基于视觉和视觉-惯性算法自定位精度的影响。选用ROS2兼容的视觉和视觉-惯性算法，并选用实际大小车辆的数据集作为基准。进行测试驱动以记录缩小比例车辆的数据。从姿态准确性角度比较选定的定位算法OpenVINS、VINS-Fusion和RTAB-Map，以及与真实大小车辆数据和地面真实值的比较。在将选定的定位算法实施到实际大小车辆进行比较时，OpenVINS具有最低的平均定位误差。尽管所有选定的定位算法具有重叠的误差范围，但当应用于缩小比例车辆时，OpenVINS表现最佳。在将缩小比例车辆与实际大小车辆进行比较时，发现平移车辆运动估计准确性存在轻微差异，但在旋转车辆运动估计准确性方面未发现显著差异，允许使用RSVR作为自定位算法的测试平台。 

---
# MPC-based Coarse-to-Fine Motion Planning for Robotic Object Transportation in Cluttered Environments 

**Title (ZH)**: 基于Model Predictive Control的面向杂乱环境的机器人物体运输粗细motion planning 

**Authors**: Chen Cai, Ernesto Dickel Saraiva, Ya-jun Pan, Steven Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11211)  

**Abstract**: This letter presents a novel coarse-to-fine motion planning framework for robotic manipulation in cluttered, unmodeled environments. The system integrates a dual-camera perception setup with a B-spline-based model predictive control (MPC) scheme. Initially, the planner generates feasible global trajectories from partial and uncertain observations. As new visual data are incrementally fused, both the environment model and motion planning are progressively refined. A vision-based cost function promotes target-driven exploration, while a refined kernel-perceptron collision detector enables efficient constraint updates for real-time planning. The framework accommodates closed-chain kinematics and supports dynamic replanning. Experiments on a multi-arm platform validate its robustness and adaptability under uncertainties and clutter. 

**Abstract (ZH)**: 本信提出了一种用于杂乱未建模环境中机器人操作的自上而下精细优化运动规划框架。该系统结合了基于B样条的模型预测控制（MPC）方案和双摄像头感知设置。初始阶段，规划器从部分和不确定的观测中生成可行的全局轨迹。随着逐步融合新视觉数据，环境模型和运动规划也随之逐步 refinement。基于视觉的代价函数促进目标导向的探索，而细化的核感知碰撞检测器则能高效更新约束以支持实时规划。该框架能够处理闭链运动学，并支持动态重规划。实验在多臂平台上的验证了其在不确定性与杂乱条件下的健壮性和适应性。 

---
# A Robust Controller based on Gaussian Processes for Robotic Manipulators with Unknown Uncertainty 

**Title (ZH)**: 基于高斯过程的鲁棒控制器：用于具有未知不确定性的机器人 manipulator 控制 

**Authors**: Giulio Giacomuzzo, Mohamed Abdelwahab, Marco Calì, Alberto Dalla Libera, Ruggero Carli  

**Link**: [PDF](https://arxiv.org/pdf/2507.11170)  

**Abstract**: In this paper, we propose a novel learning-based robust feedback linearization strategy to ensure precise trajectory tracking for an important family of Lagrangian systems. We assume a nominal knowledge of the dynamics is given but no a-priori bounds on the model mismatch are available. In our approach, the key ingredient is the adoption of a regression framework based on Gaussian Processes (GPR) to estimate the model mismatch. This estimate is added to the outer loop of a classical feedback linearization scheme based on the nominal knowledge available. Then, to compensate for the residual uncertainty, we robustify the controller including an additional term whose size is designed based on the variance provided by the GPR framework. We proved that, with high probability, the proposed scheme is able to guarantee asymptotic tracking of a desired trajectory. We tested numerically our strategy on a 2 degrees of freedom planar robot. 

**Abstract (ZH)**: 本文提出了一种基于学习的鲁棒反馈线性化策略，以确保对拉格朗日系统重要一类的精确轨迹跟踪。我们假设已给定系统的名义动力学知识，但没有关于模型不匹配先验界的信息。在我们的方法中，关键成分是采用基于高斯过程（GPR）的回归框架来估计模型不匹配。此估计值被添加到基于可用名义知识的经典反馈线性化方案的外环中。然后，为了抵消剩余的不确定性，我们使控制器具有鲁棒性，并增加了一个额外的项，其大小是基于GPR框架提供的方差进行设计的。我们证明，有高度概率性，所提出的方案能够保证对期望轨迹的渐近跟踪。我们在一个两自由度平面机器人上进行了数值测试。 

---
# Force-Based Viscosity and Elasticity Measurements for Material Biomechanical Characterisation with a Collaborative Robotic Arm 

**Title (ZH)**: 基于力反馈的材料本构特性表征中黏弹性的协作机器人测量方法 

**Authors**: Luca Beber, Edoardo Lamon, Giacomo Moretti, Matteo Saveriano, Luca Fambri, Luigi Palopoli, Daniele Fontanelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.11133)  

**Abstract**: Diagnostic activities, such as ultrasound scans and palpation, are relatively low-cost. They play a crucial role in the early detection of health problems and in assessing their progression. However, they are also error-prone activities, which require highly skilled medical staff. The use of robotic solutions can be key to decreasing the inherent subjectivity of the results and reducing the waiting list. For a robot to perform palpation or ultrasound scans, it must effectively manage physical interactions with the human body, which greatly benefits from precise estimation of the patient's tissue biomechanical properties. This paper assesses the accuracy and precision of a robotic system in estimating the viscoelastic parameters of various materials, including some tests on ex vivo tissues as a preliminary proof-of-concept demonstration of the method's applicability to biological samples. The measurements are compared against a ground truth derived from silicone specimens with different viscoelastic properties, characterised using a high-precision instrument. Experimental results show that the robotic system's accuracy closely matches the ground truth, increasing confidence in the potential use of robots for such clinical applications. 

**Abstract (ZH)**: 机器人系统在估计多种材料包括离体组织的粘弹性参数方面的准确性和精确性评估：初步概念验证表明其在生物样本中的应用可行性 

---
# Closed Form Time Derivatives of the Equations of Motion of Rigid Body Systems 

**Title (ZH)**: 刚体系统运动方程的闭式时间导数 

**Authors**: Andreas Mueller, Shivesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.11076)  

**Abstract**: Derivatives of equations of motion(EOM) describing the dynamics of rigid body systems are becoming increasingly relevant for the robotics community and find many applications in design and control of robotic systems. Controlling robots, and multibody systems comprising elastic components in particular, not only requires smooth trajectories but also the time derivatives of the control forces/torques, hence of the EOM. This paper presents the time derivatives of the EOM in closed form up to second-order as an alternative formulation to the existing recursive algorithms for this purpose, which provides a direct insight into the structure of the derivatives. The Lie group formulation for rigid body systems is used giving rise to very compact and easily parameterized equations. 

**Abstract (ZH)**: 描述刚体系统动力学的运动方程的时间导数在机器人技术领域越来越受到关注，并在机器人系统的设计与控制中找到许多应用。本文以闭式形式给出了运动方程的一阶和二阶时间导数，作为一种替代现有的递归算法的方法，这提供了一种直接洞察导数结构的方式。使用刚体系统的李群形式化描述导致了非常紧凑且易于参数化的方程。 

---
# TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update 

**Title (ZH)**: TRAN-D：基于2D高斯散斑的稀疏视图透明对象深度重构方法及其在场景更新中的物理模拟 

**Authors**: Jeongyun Kim, Seunghoon Jeong, Giseop Kim, Myung-Hwan Jeon, Eunji Jun, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.11069)  

**Abstract**: Understanding the 3D geometry of transparent objects from RGB images is challenging due to their inherent physical properties, such as reflection and refraction. To address these difficulties, especially in scenarios with sparse views and dynamic environments, we introduce TRAN-D, a novel 2D Gaussian Splatting-based depth reconstruction method for transparent objects. Our key insight lies in separating transparent objects from the background, enabling focused optimization of Gaussians corresponding to the object. We mitigate artifacts with an object-aware loss that places Gaussians in obscured regions, ensuring coverage of invisible surfaces while reducing overfitting. Furthermore, we incorporate a physics-based simulation that refines the reconstruction in just a few seconds, effectively handling object removal and chain-reaction movement of remaining objects without the need for rescanning. TRAN-D is evaluated on both synthetic and real-world sequences, and it consistently demonstrated robust improvements over existing GS-based state-of-the-art methods. In comparison with baselines, TRAN-D reduces the mean absolute error by over 39% for the synthetic TRansPose sequences. Furthermore, despite being updated using only one image, TRAN-D reaches a {\delta} < 2.5 cm accuracy of 48.46%, over 1.5 times that of baselines, which uses six images. Code and more results are available at this https URL. 

**Abstract (ZH)**: 从RGB图像理解透明物体的3D几何形状具有挑战性，因为透明物体具有固有的物理属性，如反射和折射。为了应对这些困难，特别是在稀疏视角和动态环境场景中，我们引入了TRAN-D，一种新型的基于2D高斯斑点的透明物体深度重建方法。我们的关键见解在于将透明物体与背景分离，从而使我们能够专注于优化与物体对应的高斯函数。我们通过一种对象感知的损失来减轻伪影，该损失使高斯函数定位在被遮挡的区域，从而确保覆盖看不见的表面同时减少过拟合。此外，我们结合了基于物理的模拟，在几秒钟内细化重建结果，有效处理物体去除和剩余物体连带运动，而无需重新扫描。TRAN-D在合成序列和真实世界序列上进行了评估，并且在与现有基于高斯斑点的最先进的方法相比时，表现出一致的稳健改进。相比基线方法，TRAN-D在合成TRansPose序列上的均方绝对误差降低了超过39%。此外，即使只使用了一张图像进行更新，TRAN-D的δ < 2.5 cm精度达到了48.46%，超过了使用六张图像的基线方法1.5倍以上的精度。相关代码和更多结果详见此链接。 

---
# Enhancing Autonomous Manipulator Control with Human-in-loop for Uncertain Assembly Environments 

**Title (ZH)**: 基于人为回路增强不确定装配环境中的自主 manipulator 控制 

**Authors**: Ashutosh Mishra, Shreya Santra, Hazal Gozbasi, Kentaro Uno, Kazuya Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2507.11006)  

**Abstract**: This study presents an advanced approach to enhance robotic manipulation in uncertain and challenging environments, with a focus on autonomous operations augmented by human-in-the-loop (HITL) control for lunar missions. By integrating human decision-making with autonomous robotic functions, the research improves task reliability and efficiency for space applications. The key task addressed is the autonomous deployment of flexible solar panels using an extendable ladder-like structure and a robotic manipulator with real-time feedback for precision. The manipulator relays position and force-torque data, enabling dynamic error detection and adaptive control during deployment. To mitigate the effects of sinkage, variable payload, and low-lighting conditions, efficient motion planning strategies are employed, supplemented by human control that allows operators to intervene in ambiguous scenarios. Digital twin simulation enhances system robustness by enabling continuous feedback, iterative task refinement, and seamless integration with the deployment pipeline. The system has been tested to validate its performance in simulated lunar conditions and ensure reliability in extreme lighting, variable terrain, changing payloads, and sensor limitations. 

**Abstract (ZH)**: 本研究提出了一种先进的方法，旨在在不确定和具挑战性的环境中提升机器人的操作能力，并重点关注结合有人参与环回控制（HITL）的自主操作在月球任务中的应用。通过将人类的决策与自主机器人功能相结合，该研究提高了空间应用中的任务可靠性和效率。主要任务是利用可伸展的梯形结构和具有实时反馈的机器人操作臂，实现柔性太阳电池板的自主部署。操作臂传输位置和力-扭矩数据，使部署过程中能够动态检测错误并进行自适应控制。为了应对下沉效应、可变载荷和低光照条件，研究采用了高效的运动规划策略，并结合了人类控制，以便在模棱两可的情境中进行干预。通过数字孪生模拟，该系统能够提供持续反馈、迭代的任务精细化并无缝集成到部署流程中。该系统已在模拟月球条件下进行了测试，以验证其在极端光照、可变地形、载荷变化和传感器限制条件下的可靠性。 

---
# Learning to Tune Like an Expert: Interpretable and Scene-Aware Navigation via MLLM Reasoning and CVAE-Based Adaptation 

**Title (ZH)**: 像专家一样调整：基于MLLM推理和CVAE基适应的可解释且场景感知的导航 

**Authors**: Yanbo Wang, Zipeng Fang, Lei Zhao, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11001)  

**Abstract**: Service robots are increasingly deployed in diverse and dynamic environments, where both physical layouts and social contexts change over time and across locations. In these unstructured settings, conventional navigation systems that rely on fixed parameters often fail to generalize across scenarios, resulting in degraded performance and reduced social acceptance. Although recent approaches have leveraged reinforcement learning to enhance traditional planners, these methods often fail in real-world deployments due to poor generalization and limited simulation diversity, which hampers effective sim-to-real transfer. To tackle these issues, we present LE-Nav, an interpretable and scene-aware navigation framework that leverages multi-modal large language model reasoning and conditional variational autoencoders to adaptively tune planner hyperparameters. To achieve zero-shot scene understanding, we utilize one-shot exemplars and chain-of-thought prompting strategies. Additionally, a conditional variational autoencoder captures the mapping between natural language instructions and navigation hyperparameters, enabling expert-level tuning. Experiments show that LE-Nav can generate hyperparameters achieving human-level tuning across diverse planners and scenarios. Real-world navigation trials and a user study on a smart wheelchair platform demonstrate that it outperforms state-of-the-art methods on quantitative metrics such as success rate, efficiency, safety, and comfort, while receiving higher subjective scores for perceived safety and social acceptance. Code is available at this https URL. 

**Abstract (ZH)**: 基于多模态大语言模型推理和服务变分自动编码器的可解释场景感知导航框架LE-Nav 

---
# ILCL: Inverse Logic-Constraint Learning from Temporally Constrained Demonstrations 

**Title (ZH)**: ILCL: 基于时间约束演示的逆逻辑约束学习 

**Authors**: Minwoo Cho, Jaehwi Jang, Daehyung Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.11000)  

**Abstract**: We aim to solve the problem of temporal-constraint learning from demonstrations to reproduce demonstration-like logic-constrained behaviors. Learning logic constraints is challenging due to the combinatorially large space of possible specifications and the ill-posed nature of non-Markovian constraints. To figure it out, we introduce a novel temporal-constraint learning method, which we call inverse logic-constraint learning (ILCL). Our method frames ICL as a two-player zero-sum game between 1) a genetic algorithm-based temporal-logic mining (GA-TL-Mining) and 2) logic-constrained reinforcement learning (Logic-CRL). GA-TL-Mining efficiently constructs syntax trees for parameterized truncated linear temporal logic (TLTL) without predefined templates. Subsequently, Logic-CRL finds a policy that maximizes task rewards under the constructed TLTL constraints via a novel constraint redistribution scheme. Our evaluations show ILCL outperforms state-of-the-art baselines in learning and transferring TL constraints on four temporally constrained tasks. We also demonstrate successful transfer to real-world peg-in-shallow-hole tasks. 

**Abstract (ZH)**: 我们旨在通过演示解决时间约束学习问题，以重现类似演示的逻辑约束行为。学习逻辑约束具有挑战性，因为可能的规范空间极为庞大且非马尔可夫约束问题描述不明确。为此，我们提出了一种新颖的时间约束学习方法，称为逆向逻辑约束学习（ILCL）。该方法将ILCL视为遗传算法基于时间逻辑挖掘（GA-TL-Mining）与逻辑约束强化学习（Logic-CRL）之间的零和博弈。GA-TL-Mining高效地构建了参数化截断线性时序逻辑（TLTL）的语法树，而无需预定义模板。随后，Logic-CRL通过一种新颖的约束重分布方案，在构建的TLTL约束下寻找最大化任务奖励的策略。我们的评估展示了ILCL在四个时间约束任务上学习和转移TL约束方面优于现有基准。我们还展示了其在实际应用中的成功迁移，特别是固定销入浅孔任务。 

---
# Uncertainty Aware Mapping for Vision-Based Underwater Robots 

**Title (ZH)**: 基于视觉的水下机器人不确定性感知映射 

**Authors**: Abhimanyu Bhowmik, Mohit Singh, Madhushree Sannigrahi, Martin Ludvigsen, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2507.10991)  

**Abstract**: Vision-based underwater robots can be useful in inspecting and exploring confined spaces where traditional sensors and preplanned paths cannot be followed. Sensor noise and situational change can cause significant uncertainty in environmental representation. Thus, this paper explores how to represent mapping inconsistency in vision-based sensing and incorporate depth estimation confidence into the mapping framework. The scene depth and the confidence are estimated using the RAFT-Stereo model and are integrated into a voxel-based mapping framework, Voxblox. Improvements in the existing Voxblox weight calculation and update mechanism are also proposed. Finally, a qualitative analysis of the proposed method is performed in a confined pool and in a pier in the Trondheim fjord. Experiments using an underwater robot demonstrated the change in uncertainty in the visualization. 

**Abstract (ZH)**: 基于视觉的水下机器人能够在传统传感器和预规划路径无法适用的受限空间内进行检查和探索。传感器噪声和情况变化会导致环境表示中的显著不确定性。因此，本文探讨了如何在基于视觉的传感中表示制图不一致性，并将深度估计置信度整合到制图框架中。使用RAFT-Stereo模型估计场景深度和置信度，并将它们集成到基于体素的制图框架Voxblox中。还提出了对现有Voxblox权重计算和更新机制的改进。最后，在特隆赫姆峡湾的游泳池和码头进行定性分析，实验结果表明 proposed方法在可视化中不确定性变化。 

---
# SMART-Merge Planner: A Safe Merging and Real-Time Motion Planner for Autonomous Highway On-Ramp Merging 

**Title (ZH)**: SMART-Merge 计划器：一种安全合并及实时运动规划器，应用于自主高速公路入口匝道合并 

**Authors**: Toktam Mohammadnejad, Jovin D'sa, Behdad Chalaki, Hossein Nourkhiz Mahjoub, Ehsan Moradi-Pari  

**Link**: [PDF](https://arxiv.org/pdf/2507.10968)  

**Abstract**: Merging onto a highway is a complex driving task that requires identifying a safe gap, adjusting speed, often interactions to create a merging gap, and completing the merge maneuver within a limited time window while maintaining safety and driving comfort. In this paper, we introduce a Safe Merging and Real-Time Merge (SMART-Merge) planner, a lattice-based motion planner designed to facilitate safe and comfortable forced merging. By deliberately adapting cost terms to the unique challenges of forced merging and introducing a desired speed heuristic, SMART-Merge planner enables the ego vehicle to merge successfully while minimizing the merge time. We verify the efficiency and effectiveness of the proposed merge planner through high-fidelity CarMaker simulations on hundreds of highway merge scenarios. Our proposed planner achieves the success rate of 100% as well as completes the merge maneuver in the shortest amount of time compared with the baselines, demonstrating our planner's capability to handle complex forced merge tasks and provide a reliable and robust solution for autonomous highway merge. The simulation result videos are available at this https URL. 

**Abstract (ZH)**: 基于格子的智能安全快速并线规划器（SMART-Merge）：应对复杂强制并线任务的实时并线规划 

---
# EquiContact: A Hierarchical SE(3) Vision-to-Force Equivariant Policy for Spatially Generalizable Contact-rich Tasks 

**Title (ZH)**: EquiContact: 一种层次化的SE(3)视觉到力等变策略，用于空间上通用的富含接触的任务 

**Authors**: Joohwan Seo, Arvind Kruthiventy, Soomi Lee, Megan Teng, Xiang Zhang, Seoyeon Choi, Jongeun Choi, Roberto Horowitz  

**Link**: [PDF](https://arxiv.org/pdf/2507.10961)  

**Abstract**: This paper presents a framework for learning vision-based robotic policies for contact-rich manipulation tasks that generalize spatially across task configurations. We focus on achieving robust spatial generalization of the policy for the peg-in-hole (PiH) task trained from a small number of demonstrations. We propose EquiContact, a hierarchical policy composed of a high-level vision planner (Diffusion Equivariant Descriptor Field, Diff-EDF) and a novel low-level compliant visuomotor policy (Geometric Compliant ACT, G-CompACT). G-CompACT operates using only localized observations (geometrically consistent error vectors (GCEV), force-torque readings, and wrist-mounted RGB images) and produces actions defined in the end-effector frame. Through these design choices, we show that the entire EquiContact pipeline is SE(3)-equivariant, from perception to force control. We also outline three key components for spatially generalizable contact-rich policies: compliance, localized policies, and induced equivariance. Real-world experiments on PiH tasks demonstrate a near-perfect success rate and robust generalization to unseen spatial configurations, validating the proposed framework and principles. The experimental videos can be found on the project website: this https URL 

**Abstract (ZH)**: 本文提出了一种基于视觉的机器人策略框架，用于接触丰富的 manipulation 任务，并在空间上泛化到不同的任务配置。我们专注于从少量演示中训练 peg-in-hole (PiH) 任务的策略，并实现其鲁棒的空间泛化。我们提出了 EquiContact，这是一种由高层视觉规划者（扩散等变描述子场，Diff-EDF）和新颖的低层顺应性视知觉运动策略（几何顺应性 ACT，G-CompACT）组成的层次策略。G-CompACT 仅使用局部观察（几何一致的误差向量 (GCEV)、力-扭矩读数和腕部安装的 RGB 图像）生成在末端执行器坐标系中定义的动作。通过这些设计选择，我们证明了整个 EquiContact 管道从感知到力控制都是 SE(3) 等变的。我们还概述了空间泛化的接触丰富策略的三个关键组件：顺应性、局部策略和诱导等变性。在 PiH 任务的实际实验中，展示了近乎完美的成功率和对未见的空间配置的鲁棒泛化，验证了所提出的框架和原则。实验视频可在项目网站上找到：this https URL。 

---
# Whom to Respond To? A Transformer-Based Model for Multi-Party Social Robot Interaction 

**Title (ZH)**: 与谁互动？一种基于Transformer的多机器人社会交互模型 

**Authors**: He Zhu, Ryo Miyoshi, Yuki Okafuji  

**Link**: [PDF](https://arxiv.org/pdf/2507.10960)  

**Abstract**: Prior human-robot interaction (HRI) research has primarily focused on single-user interactions, where robots do not need to consider the timing or recipient of their responses. However, in multi-party interactions, such as at malls and hospitals, social robots must understand the context and decide both when and to whom they should respond. In this paper, we propose a Transformer-based multi-task learning framework to improve the decision-making process of social robots, particularly in multi-user environments. Considering the characteristics of HRI, we propose two novel loss functions: one that enforces constraints on active speakers to improve scene modeling, and another that guides response selection towards utterances specifically directed at the robot. Additionally, we construct a novel multi-party HRI dataset that captures real-world complexities, such as gaze misalignment. Experimental results demonstrate that our model achieves state-of-the-art performance in respond decisions, outperforming existing heuristic-based and single-task approaches. Our findings contribute to the development of socially intelligent social robots capable of engaging in natural and context-aware multi-party interactions. 

**Abstract (ZH)**: 基于Transformers的多任务学习框架：提高社会机器人在多用户环境中的决策过程 

---
# Unified Modeling and Structural Optimization of Multi-magnet Embedded Soft Continuum Robots for Enhanced Kinematic Performances 

**Title (ZH)**: 多磁体嵌入软连续机器人统一建模与结构优化以提升运动性能 

**Authors**: Zhiwei Wu, Jiahao Luo, Siyi Wei, Jinhui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10950)  

**Abstract**: This paper presents a unified modeling and optimization framework to enhance the kinematic performance of multi-magnet embedded soft continuum robots (MeSCRs). To this end, we establish a differentiable system formulation based on an extended pseudo-rigid-body model. This formulation enables analysis of the equilibrium well-posedness and the geometry of the induced configuration under magnetic actuation. In particular, we show that the maximum controllable degrees of freedom of a MeSCR equal twice the number of embedded magnets. We subsequently develop a structural optimization framework based on differential geometry that links classical kinematic measures (e.g., manipulability and dexterity) to the configuration of embedded magnets. The resulting optimization condition reveals that improving local performance requires structurally modulating the spectrum of the configuration space metric to counteract its distortion. Closed-form solutions for optimal magnet configurations are derived under representative conditions, and a gradient-based numerical method is proposed for general design scenarios. Simulation studies validate the effectiveness of the proposed framework. 

**Abstract (ZH)**: 一种统一建模与优化框架以提升多磁体嵌入软连续机器人(MeSCRs)的运动性能 

---
# Fast Non-Episodic Adaptive Tuning of Robot Controllers with Online Policy Optimization 

**Title (ZH)**: 基于在线策略优化的快速非情景自适应机器人控制器调整 

**Authors**: James A. Preiss, Fengze Xie, Yiheng Lin, Adam Wierman, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.10914)  

**Abstract**: We study online algorithms to tune the parameters of a robot controller in a setting where the dynamics, policy class, and optimality objective are all time-varying. The system follows a single trajectory without episodes or state resets, and the time-varying information is not known in advance. Focusing on nonlinear geometric quadrotor controllers as a test case, we propose a practical implementation of a single-trajectory model-based online policy optimization algorithm, M-GAPS,along with reparameterizations of the quadrotor state space and policy class to improve the optimization landscape. In hardware experiments,we compare to model-based and model-free baselines that impose artificial episodes. We show that M-GAPS finds near-optimal parameters more quickly, especially when the episode length is not favorable. We also show that M-GAPS rapidly adapts to heavy unmodeled wind and payload disturbances, and achieves similar strong improvement on a 1:6-scale Ackermann-steered car. Our results demonstrate the hardware practicality of this emerging class of online policy optimization that offers significantly more flexibility than classic adaptive control, while being more stable and data-efficient than model-free reinforcement learning. 

**Abstract (ZH)**: 我们研究在线算法以在动态、策略类和最优性目标均为时变的情况下调整机器人控制器的参数。系统遵循单一轨迹而无需分段或状态重置，且时变信息事先未知。以非线性几何多旋翼控制器为例，我们提出了一种实用的基于单一轨迹的在线策略优化算法M-GAPS的实现，并改进了多旋翼状态空间和策略类的参数化以优化优化场景。在硬件实验中，我们将M-GAPS与引入人工分段的基于模型和无模型基线进行比较。结果显示，M-GAPS能够更快地找到近最优参数，尤其是在分段长度不利时更为明显。我们还展示了M-GAPS能够迅速适应强烈的未建模风力和载荷干扰，并在1:6比例的Ackermann转向汽车上实现了类似的显著改进。我们的研究结果表明，这种新兴的在线策略优化方法在提供比经典自适应控制更多灵活性的同时，比无模型强化学习更具稳定性和数据效率。 

---
# Object-Centric Mobile Manipulation through SAM2-Guided Perception and Imitation Learning 

**Title (ZH)**: 基于SAM2引导的感知与imitation学习的物体为中心的移动操作Manipulation 

**Authors**: Wang Zhicheng, Satoshi Yagi, Satoshi Yamamori, Jun Morimoto  

**Link**: [PDF](https://arxiv.org/pdf/2507.10899)  

**Abstract**: Imitation learning for mobile manipulation is a key challenge in the field of robotic manipulation. However, current mobile manipulation frameworks typically decouple navigation and manipulation, executing manipulation only after reaching a certain location. This can lead to performance degradation when navigation is imprecise, especially due to misalignment in approach angles. To enable a mobile manipulator to perform the same task from diverse orientations, an essential capability for building general-purpose robotic models, we propose an object-centric method based on SAM2, a foundation model towards solving promptable visual segmentation in images, which incorporates manipulation orientation information into our model. Our approach enables consistent understanding of the same task from different orientations. We deploy the model on a custom-built mobile manipulator and evaluate it on a pick-and-place task under varied orientation angles. Compared to Action Chunking Transformer, our model maintains superior generalization when trained with demonstrations from varied approach angles. This work significantly enhances the generalization and robustness of imitation learning-based mobile manipulation systems. 

**Abstract (ZH)**: 基于SAM2的目标为中心的方法在移动操作中的模仿学习 

---
# Mixed Discrete and Continuous Planning using Shortest Walks in Graphs of Convex Sets 

**Title (ZH)**: 混合离散与连续规划：基于凸集图上最短路径的方法 

**Authors**: Savva Morozov, Tobia Marcucci, Bernhard Paus Graesdal, Alexandre Amice, Pablo A. Parrilo, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2507.10878)  

**Abstract**: We study the Shortest-Walk Problem (SWP) in a Graph of Convex Sets (GCS). A GCS is a graph where each vertex is paired with a convex program, and each edge couples adjacent programs via additional costs and constraints. A walk in a GCS is a sequence of vertices connected by edges, where vertices may be repeated. The length of a walk is given by the cumulative optimal value of the corresponding convex programs. To solve the SWP in GCS, we first synthesize a piecewise-quadratic lower bound on the problem's cost-to-go function using semidefinite programming. Then we use this lower bound to guide an incremental-search algorithm that yields an approximate shortest walk. We show that the SWP in GCS is a natural language for many mixed discrete-continuous planning problems in robotics, unifying problems that typically require specialized solutions while delivering high performance and computational efficiency. We demonstrate this through experiments in collision-free motion planning, skill chaining, and optimal control of hybrid systems. 

**Abstract (ZH)**: 我们研究凸集图（GCS）中的最短行走问题（SWP）。凸集图（GCS）是一种图形，其中每个顶点都对应一个凸规划，每条边通过附加的成本和约束将相邻的规划联系起来。在凸集图中的行走是一条由边连接的顶点序列，顶点可以重复。行走的长度是由相应凸规划的累积最优值给出的。为了解决GCS中的最短行走问题，我们首先使用半定规划合成功维二次下界问题的成本-剩下函数。然后，我们使用这个下界来引导一种增量搜索算法，以获得近似的最短行走。我们展示了GCS中的SWP是一种自然的语言，适用于许多机器人中的混合离散-连续规划问题，统一了通常需要专门解决方案的问题，同时提供高效性能和计算效率。我们通过在无碰撞运动规划、技能链动和混合系统最优控制方面的实验进行了演示。 

---
# Versatile and Generalizable Manipulation via Goal-Conditioned Reinforcement Learning with Grounded Object Detection 

**Title (ZH)**: 基于 grounded 对象检测的目标条件强化学习的通用化 manipulation 方法 

**Authors**: Huiyi Wang, Fahim Shahriar, Alireza Azimi, Gautham Vasan, Rupam Mahmood, Colin Bellinger  

**Link**: [PDF](https://arxiv.org/pdf/2507.10814)  

**Abstract**: General-purpose robotic manipulation, including reach and grasp, is essential for deployment into households and workspaces involving diverse and evolving tasks. Recent advances propose using large pre-trained models, such as Large Language Models and object detectors, to boost robotic perception in reinforcement learning. These models, trained on large datasets via self-supervised learning, can process text prompts and identify diverse objects in scenes, an invaluable skill in RL where learning object interaction is resource-intensive. This study demonstrates how to integrate such models into Goal-Conditioned Reinforcement Learning to enable general and versatile robotic reach and grasp capabilities. We use a pre-trained object detection model to enable the agent to identify the object from a text prompt and generate a mask for goal conditioning. Mask-based goal conditioning provides object-agnostic cues, improving feature sharing and generalization. The effectiveness of the proposed framework is demonstrated in a simulated reach-and-grasp task, where the mask-based goal conditioning consistently maintains a $\sim$90\% success rate in grasping both in and out-of-distribution objects, while also ensuring faster convergence to higher returns. 

**Abstract (ZH)**: 通用型机器人操作，包括抓取和握持，对于部署到涉及多样化和不断演变任务的家庭和工作空间至关重要。近期的研究提出使用大型预训练模型，如大规模语言模型和物体检测器，以增强强化学习中的机器人感知能力。这些模型通过半监督学习在大规模数据集上训练，能够处理文本提示并识别场景中的多种物体，这一技能在强化学习中尤为重要，因为学习物体交互资源密集。本研究展示了如何将此类模型整合到目标条件强化学习中，以实现通用和多功能的机器人抓取和握持能力。我们使用预训练的物体检测模型，使智能体能够从文本提示中识别物体并生成目标条件的掩码。基于掩码的目标条件提供了物体无关的线索，提高了特征共享和泛化能力。所提出框架的有效性在模拟的抓取和握持任务中得到验证，基于掩码的目标条件在室内和室外物体抓取中保持了约90%的成功率，同时确保更快地收敛到更高的回报。 

---
# rt-RISeg: Real-Time Model-Free Robot Interactive Segmentation for Active Instance-Level Object Understanding 

**Title (ZH)**: rt-RISeg: 实时无模型机器人交互分割以实现主动实例级物体理解 

**Authors**: Howard H. Qian, Yiting Chen, Gaotian Wang, Podshara Chanrungmaneekul, Kaiyu Hang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10776)  

**Abstract**: Successful execution of dexterous robotic manipulation tasks in new environments, such as grasping, depends on the ability to proficiently segment unseen objects from the background and other objects. Previous works in unseen object instance segmentation (UOIS) train models on large-scale datasets, which often leads to overfitting on static visual features. This dependency results in poor generalization performance when confronted with out-of-distribution scenarios. To address this limitation, we rethink the task of UOIS based on the principle that vision is inherently interactive and occurs over time. We propose a novel real-time interactive perception framework, rt-RISeg, that continuously segments unseen objects by robot interactions and analysis of a designed body frame-invariant feature (BFIF). We demonstrate that the relative rotational and linear velocities of randomly sampled body frames, resulting from selected robot interactions, can be used to identify objects without any learned segmentation model. This fully self-contained segmentation pipeline generates and updates object segmentation masks throughout each robot interaction without the need to wait for an action to finish. We showcase the effectiveness of our proposed interactive perception method by achieving an average object segmentation accuracy rate 27.5% greater than state-of-the-art UOIS methods. Furthermore, although rt-RISeg is a standalone framework, we show that the autonomously generated segmentation masks can be used as prompts to vision foundation models for significantly improved performance. 

**Abstract (ZH)**: 基于新环境中的灵巧机器人操作任务成功执行，如抓取，依赖于从背景和其他物体中高效分割未见物体的能力。过去在未见物体实例分割（UOIS）方面的研究在大规模数据集上训练模型，这通常会导致对静态视觉特征的过度拟合。这种依赖导致在遇到分布外场景时泛化性能较差。为解决这一局限，我们基于视觉本质上是交互的且发生在时间上的原则重新考虑UOIS任务。我们提出了一种新颖的实时交互感知框架rt-RISeg，该框架通过机器人交互和设计的体帧不变特征（BFIF）分析不断分割未见物体。我们证明，从所选机器人交互中随机采样的体帧的相对旋转和线性速度可以用于识别物体，无需任何学习分割模型。这个完全自包含的分割管道在每次机器人交互过程中生成并更新物体分割掩码，而无需等待动作完成。我们通过将rt-RISeg方法的平均物体分割准确率提高27.5%，展示了我们提出的交互感知方法的有效性。此外，虽然rt-RISeg是一个独立框架，但我们证明自动生成的分割掩码可以作为提示用于视觉基础模型，以显著提高性能。 

---
# RCG: Safety-Critical Scenario Generation for Robust Autonomous Driving via Real-World Crash Grounding 

**Title (ZH)**: RCG：通过现实碰撞接地实现稳健自动驾驶的安全关键场景生成 

**Authors**: Benjamin Stoler, Juliet Yang, Jonathan Francis, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.10749)  

**Abstract**: Safety-critical scenarios are essential for training and evaluating autonomous driving (AD) systems, yet remain extremely rare in real-world driving datasets. To address this, we propose Real-world Crash Grounding (RCG), a scenario generation framework that integrates crash-informed semantics into adversarial perturbation pipelines. We construct a safety-aware behavior representation through contrastive pre-training on large-scale driving logs, followed by fine-tuning on a small, crash-rich dataset with approximate trajectory annotations extracted from video. This embedding captures semantic structure aligned with real-world accident behaviors and supports selection of adversary trajectories that are both high-risk and behaviorally realistic. We incorporate the resulting selection mechanism into two prior scenario generation pipelines, replacing their handcrafted scoring objectives with an embedding-based criterion. Experimental results show that ego agents trained against these generated scenarios achieve consistently higher downstream success rates, with an average improvement of 9.2% across seven evaluation settings. Qualitative and quantitative analyses further demonstrate that our approach produces more plausible and nuanced adversary behaviors, enabling more effective and realistic stress testing of AD systems. Code and tools will be released publicly. 

**Abstract (ZH)**: 基于现实碰撞事件的自动驾驶安全场景生成框架（Real-world Crash Grounding for Autonomous Driving Scenario Generation） 

---
# Exteroception through Proprioception Sensing through Improved Contact Modeling for Soft Growing Robots 

**Title (ZH)**: 通过改进接触模型实现外部感知的 proprioception 传感技术在软体生长机器人中的应用 

**Authors**: Francesco Fuentes, Serigne Diagne, Zachary Kingston, Laura H. Blumenschein  

**Link**: [PDF](https://arxiv.org/pdf/2507.10694)  

**Abstract**: Passive deformation due to compliance is a commonly used benefit of soft robots, providing opportunities to achieve robust actuation with few active degrees of freedom. Soft growing robots in particular have shown promise in navigation of unstructured environments due to their passive deformation. If their collisions and subsequent deformations can be better understood, soft robots could be used to understand the structure of the environment from direct tactile measurements. In this work, we propose the use of soft growing robots as mapping and exploration tools. We do this by first characterizing collision behavior during discrete turns, then leveraging this model to develop a geometry-based simulator that models robot trajectories in 2D environments. Finally, we demonstrate the model and simulator validity by mapping unknown environments using Monte Carlo sampling to estimate the optimal next deployment given current knowledge. Over both uniform and non-uniform environments, this selection method rapidly approaches ideal actions, showing the potential for soft growing robots in unstructured environment exploration and mapping. 

**Abstract (ZH)**: 软体生长机器人作为测绘与探索工具的被动变形作用及其应用 

---
# Vision Language Action Models in Robotic Manipulation: A Systematic Review 

**Title (ZH)**: 机器人操控中的视觉语言行动模型：一项系统性综述 

**Authors**: Muhayy Ud Din, Waseem Akram, Lyes Saad Saoud, Jan Rosell, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2507.10672)  

**Abstract**: Vision Language Action (VLA) models represent a transformative shift in robotics, with the aim of unifying visual perception, natural language understanding, and embodied control within a single learning framework. This review presents a comprehensive and forward-looking synthesis of the VLA paradigm, with a particular emphasis on robotic manipulation and instruction-driven autonomy. We comprehensively analyze 102 VLA models, 26 foundational datasets, and 12 simulation platforms that collectively shape the development and evaluation of VLAs models. These models are categorized into key architectural paradigms, each reflecting distinct strategies for integrating vision, language, and control in robotic systems. Foundational datasets are evaluated using a novel criterion based on task complexity, variety of modalities, and dataset scale, allowing a comparative analysis of their suitability for generalist policy learning. We introduce a two-dimensional characterization framework that organizes these datasets based on semantic richness and multimodal alignment, showing underexplored regions in the current data landscape. Simulation environments are evaluated for their effectiveness in generating large-scale data, as well as their ability to facilitate transfer from simulation to real-world settings and the variety of supported tasks. Using both academic and industrial contributions, we recognize ongoing challenges and outline strategic directions such as scalable pretraining protocols, modular architectural design, and robust multimodal alignment strategies. This review serves as both a technical reference and a conceptual roadmap for advancing embodiment and robotic control, providing insights that span from dataset generation to real world deployment of generalist robotic agents. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型代表了机器人领域的一项变革性转变，旨在在一个统一的学习框架中融合视觉感知、自然语言理解和实体控制。本文综述了VLA范式的全面且前瞻性的发展，特别强调了机器人操作和指令驱动的自主性。我们全面分析了102个VLA模型、26个基础数据集和12个仿真平台，这些平台共同塑造了VLA模型的发展和评估。这些模型被归类为关键的架构范式，每个范式都反映了整合视觉、语言和控制策略的不同策略。基础数据集根据任务复杂性、模态多样性以及数据集规模的新颖标准进行评估，便于对它们的一般性策略学习适合性进行比较分析。我们引入了一个二维表征框架，根据语义丰富性和多模态对齐性组织这些数据集，显示了当前数据景观中未开发的地区。仿真环境被评估其生成大规模数据的有效性以及支持从仿真到真实世界的应用能力以及所支持任务的多样性。通过综合学术和工业贡献，我们指出了当前面临的挑战，并提出了战略方向，如可扩展的预训练协议、模块化架构设计和稳健的多模态对齐策略。该综述既是一个技术参考，也是一个概念路线图，有助于推进实体性和机器人控制的发展，为从数据集生成到通用机器人代理在真实世界部署提供了见解。 

---
# Learning to Move in Rhythm: Task-Conditioned Motion Policies with Orbital Stability Guarantees 

**Title (ZH)**: 按节奏学习移动：具有轨道稳定保证的任务条件运动策略 

**Authors**: Maximilian Stölzle, T. Konstantin Rusch, Zach J. Patterson, Rodrigo Pérez-Dattari, Francesco Stella, Josie Hughes, Cosimo Della Santina, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2507.10602)  

**Abstract**: Learning from demonstration provides a sample-efficient approach to acquiring complex behaviors, enabling robots to move robustly, compliantly, and with fluidity. In this context, Dynamic Motion Primitives offer built - in stability and robustness to disturbances but often struggle to capture complex periodic behaviors. Moreover, they are limited in their ability to interpolate between different tasks. These shortcomings substantially narrow their applicability, excluding a wide class of practically meaningful tasks such as locomotion and rhythmic tool use. In this work, we introduce Orbitally Stable Motion Primitives (OSMPs) - a framework that combines a learned diffeomorphic encoder with a supercritical Hopf bifurcation in latent space, enabling the accurate acquisition of periodic motions from demonstrations while ensuring formal guarantees of orbital stability and transverse contraction. Furthermore, by conditioning the bijective encoder on the task, we enable a single learned policy to represent multiple motion objectives, yielding consistent zero-shot generalization to unseen motion objectives within the training distribution. We validate the proposed approach through extensive simulation and real-world experiments across a diverse range of robotic platforms - from collaborative arms and soft manipulators to a bio-inspired rigid-soft turtle robot - demonstrating its versatility and effectiveness in consistently outperforming state-of-the-art baselines such as diffusion policies, among others. 

**Abstract (ZH)**: 基于演示学习提供了一种高效获取复杂行为的方法，使机器人能够以稳定、顺应性和流畅的方式移动。在这种背景下，轨道稳定运动基元提供内置的稳定性和对干扰的鲁棒性，但往往难以捕捉复杂的周期性行为。此外，它们在插值不同任务方面的能力有限。这些不足之处极大地限制了它们的应用范围，排除了许多实际有意义的任务，如运动和节律性工具使用。在这项工作中，我们引入了轨道稳定运动基元（OSMPs）——一种结合学习差分编码器与潜空间中的超临界霍普夫分岔的框架，能够从演示中准确地获取周期性运动，同时确保轨道稳定性和横截收缩的正式保证。此外，通过使双射编码器依赖于任务，我们使单一学习策略能够表示多个运动目标，在训练分布内的未见运动目标上实现一致的零样本泛化。我们通过广泛的模拟和实物实验，在从协作臂和软 manipulator 到生物启发的刚-软乌龟机器人等多种机器人平台上验证了所提出的方法，展示了其多样性和有效性，并在与扩散策略等最先进的基线方法进行比较时，持续表现出优越性。 

---
# CogDDN: A Cognitive Demand-Driven Navigation with Decision Optimization and Dual-Process Thinking 

**Title (ZH)**: CogDDN：基于认知需求驱动的导航与决策优化及双过程思考方法 

**Authors**: Yuehao Huang, Liang Liu, Shuangming Lei, Yukai Ma, Hao Su, Jianbiao Mei, Pengxiang Zhao, Yaqing Gu, Yong Liu, Jiajun Lv  

**Link**: [PDF](https://arxiv.org/pdf/2507.11334)  

**Abstract**: Mobile robots are increasingly required to navigate and interact within unknown and unstructured environments to meet human demands. Demand-driven navigation (DDN) enables robots to identify and locate objects based on implicit human intent, even when object locations are unknown. However, traditional data-driven DDN methods rely on pre-collected data for model training and decision-making, limiting their generalization capability in unseen scenarios. In this paper, we propose CogDDN, a VLM-based framework that emulates the human cognitive and learning mechanisms by integrating fast and slow thinking systems and selectively identifying key objects essential to fulfilling user demands. CogDDN identifies appropriate target objects by semantically aligning detected objects with the given instructions. Furthermore, it incorporates a dual-process decision-making module, comprising a Heuristic Process for rapid, efficient decisions and an Analytic Process that analyzes past errors, accumulates them in a knowledge base, and continuously improves performance. Chain of Thought (CoT) reasoning strengthens the decision-making process. Extensive closed-loop evaluations on the AI2Thor simulator with the ProcThor dataset show that CogDDN outperforms single-view camera-only methods by 15%, demonstrating significant improvements in navigation accuracy and adaptability. The project page is available at this https URL. 

**Abstract (ZH)**: 基于VLM的认知驱动导航（CogDDN）：模仿人类认知与学习机制的框架 

---
# Task-Oriented Human Grasp Synthesis via Context- and Task-Aware Diffusers 

**Title (ZH)**: 面向任务的人手抓取合成：基于上下文和任务的扩散模型 

**Authors**: An-Lun Liu, Yu-Wei Chao, Yi-Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11287)  

**Abstract**: In this paper, we study task-oriented human grasp synthesis, a new grasp synthesis task that demands both task and context awareness. At the core of our method is the task-aware contact maps. Unlike traditional contact maps that only reason about the manipulated object and its relation with the hand, our enhanced maps take into account scene and task information. This comprehensive map is critical for hand-object interaction, enabling accurate grasping poses that align with the task. We propose a two-stage pipeline that first constructs a task-aware contact map informed by the scene and task. In the subsequent stage, we use this contact map to synthesize task-oriented human grasps. We introduce a new dataset and a metric for the proposed task to evaluate our approach. Our experiments validate the importance of modeling both scene and task, demonstrating significant improvements over existing methods in both grasp quality and task performance. See our project page for more details: this https URL 

**Abstract (ZH)**: 本文研究面向任务的人机抓取合成，这是一个既需要任务意识又需要上下文意识的新抓取合成任务。我们方法的核心是任务意识接触图。不同于传统接触图仅考虑操作对象及其与手的关系，我们的增强接触图还考虑场景和任务信息。这种综合性的接触图对于手-物交互至关重要，能够生成与任务相匹配的准确抓取姿态。我们提出了一种两阶段管道，首先基于场景和任务构建任务意识接触图，随后使用该接触图合成面向任务的人机抓取。我们引入了新的数据集和评价指标来评估所提出的方法。实验结果验证了同时建模场景和任务的重要性，在抓取质量与任务性能方面显著优于现有方法。更多详情请参见我们的项目页面：this https URL 

---
# A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge 

**Title (ZH)**: 基于领域知识的无人机群协同避碰学习框架 

**Authors**: Shuangyao Huang, Haibo Zhang, Zhiyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10913)  

**Abstract**: This paper presents a multi-agent reinforcement learning (MARL) framework for cooperative collision avoidance of UAV swarms leveraging domain knowledge-driven reward. The reward is derived from knowledge in the domain of image processing, approximating contours on a two-dimensional field. By modeling obstacles as maxima on the field, collisions are inherently avoided as contours never go through peaks or intersect. Additionally, counters are smooth and energy-efficient. Our framework enables training with large swarm sizes as the agent interaction is minimized and the need for complex credit assignment schemes or observation sharing mechanisms in state-of-the-art MARL approaches are eliminated. Moreover, UAVs obtain the ability to adapt to complex environments where contours may be non-viable or non-existent through intensive training. Extensive experiments are conducted to evaluate the performances of our framework against state-of-the-art MARL algorithms. 

**Abstract (ZH)**: 基于领域知识驱动奖励的多代理 reinforcement 学习框架：UAV 群协同避障 

---
# Offline Reinforcement Learning with Wasserstein Regularization via Optimal Transport Maps 

**Title (ZH)**: 基于最优传输映射的 Wasserstein 正则化离线强化学习 

**Authors**: Motoki Omura, Yusuke Mukuta, Kazuki Ota, Takayuki Osa, Tatsuya Harada  

**Link**: [PDF](https://arxiv.org/pdf/2507.10843)  

**Abstract**: Offline reinforcement learning (RL) aims to learn an optimal policy from a static dataset, making it particularly valuable in scenarios where data collection is costly, such as robotics. A major challenge in offline RL is distributional shift, where the learned policy deviates from the dataset distribution, potentially leading to unreliable out-of-distribution actions. To mitigate this issue, regularization techniques have been employed. While many existing methods utilize density ratio-based measures, such as the $f$-divergence, for regularization, we propose an approach that utilizes the Wasserstein distance, which is robust to out-of-distribution data and captures the similarity between actions. Our method employs input-convex neural networks (ICNNs) to model optimal transport maps, enabling the computation of the Wasserstein distance in a discriminator-free manner, thereby avoiding adversarial training and ensuring stable learning. Our approach demonstrates comparable or superior performance to widely used existing methods on the D4RL benchmark dataset. The code is available at this https URL . 

**Abstract (ZH)**: 离线强化学习（RL）的目标是通过静态数据集学习最优策略，使其在数据收集成本高昂的情景下，如机器人领域中具有特别的价值。离线RL的主要挑战之一是分布偏移，即学习到的策略与数据集分布有所偏离，可能导致不可靠的越分布外的行为。为了缓解这一问题，已经采用了正则化技术。虽然许多现有方法使用基于密度比测度的方法，如$f$-散度，进行正则化，我们提出了一种利用Wasserstein距离的方法，该方法对越分布外的数据具有鲁棒性，并能够捕捉动作之间的相似性。我们的方法利用输入凸神经网络（ICNNs）来建模最优传输映射，使得无需判别器即可计算Wasserstein距离，从而避免对抗性训练并确保学习稳定。我们的方法在D4RL基准数据集上的表现与广泛使用的方法相当或更优。代码可在以下链接获取：this https URL。 

---
# GeoHopNet: Hopfield-Augmented Sparse Spatial Attention for Dynamic UAV Site Location Problem 

**Title (ZH)**: GeoHopNet：跳层增强稀疏空间注意力在动态无人机站点选址问题中的应用 

**Authors**: Jianing Zhi, Xinghua Li, Zidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10636)  

**Abstract**: The rapid development of urban low-altitude unmanned aerial vehicle (UAV) economy poses new challenges for dynamic site selection of UAV landing points and supply stations. Traditional deep reinforcement learning methods face computational complexity bottlenecks, particularly with standard attention mechanisms, when handling large-scale urban-level location problems. This paper proposes GeoHopNet, a Hopfield-augmented sparse spatial attention network specifically designed for dynamic UAV site location problems. Our approach introduces four core innovations: (1) distance-biased multi-head attention mechanism that explicitly encodes spatial geometric information; (2) K-nearest neighbor sparse attention that reduces computational complexity from $O(N^2)$ to $O(NK)$; (3) a modern Hopfield external memory module; and (4) a memory regularization strategy. Experimental results demonstrate that GeoHopNet extends the boundary of solvable problem sizes. For large-scale instances with 1,000 nodes, where standard attention models become prohibitively slow (over 3 seconds per instance) and traditional solvers fail, GeoHopNet finds high-quality solutions (0.22\% optimality gap) in under 0.1 seconds. Compared to the state-of-the-art ADNet baseline on 100-node instances, our method improves solution quality by 22.2\% and is 1.8$\times$ faster. 

**Abstract (ZH)**: 城市低空无人驾驶航空器经济的快速发展为无人驾驶航空器着陆点和供应站的动态选址提出了新挑战。传统的深度强化学习方法在处理大规模城市级别的位置问题时面临计算复杂性瓶颈，尤其是标准注意力机制。本文提出GeoHopNet，一种针对动态无人驾驶航空器站点位置问题的Hopfield增强稀疏空间注意力网络。我们的方法引入了四项核心创新：（1）距离偏差多头注意力机制，明确编码空间几何信息；（2）K最近邻稀疏注意力，将计算复杂性从$O(N^2)$降低到$O(NK)$；（3）现代Hopfield外部记忆模块；（4）记忆正则化策略。实验结果表明，GeoHopNet扩展了解决问题规模的边界。对于具有1000个节点的大规模实例，标准注意力模型变得非常慢（每个实例超过3秒），传统求解器失效，而GeoHopNet可在不到0.1秒内找到高质量解（最优性差距为0.22%）。与100节点实例上的最新ADNet基线相比，我们的方法提高了解的质量22.2%，并且速度快1.8倍。 

---
