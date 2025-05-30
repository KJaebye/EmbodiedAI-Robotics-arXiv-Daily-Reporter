# Kinodynamic Model Predictive Control for Energy Efficient Locomotion of Legged Robots with Parallel Elasticity 

**Title (ZH)**: 基于并联弹性性的腿式机器人高效运动的运动动力学模型预测控制 

**Authors**: Yulun Zhuang, Yichen Wang, Yanran Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.05666)  

**Abstract**: In this paper, we introduce a kinodynamic model predictive control (MPC) framework that exploits unidirectional parallel springs (UPS) to improve the energy efficiency of dynamic legged robots. The proposed method employs a hierarchical control structure, where the solution of MPC with simplified dynamic models is used to warm-start the kinodynamic MPC, which accounts for nonlinear centroidal dynamics and kinematic constraints. The proposed approach enables energy efficient dynamic hopping on legged robots by using UPS to reduce peak motor torques and energy consumption during stance phases. Simulation results demonstrated a 38.8% reduction in the cost of transport (CoT) for a monoped robot equipped with UPS during high-speed hopping. Additionally, preliminary hardware experiments show a 14.8% reduction in energy consumption. Video: this https URL 

**Abstract (ZH)**: 本文介绍了一种利用单向平行弹簧（UPS）改善动态腿式机器人能量效率的kinodynamic模型预测控制（MPC）框架。提出的办法采用分层控制结构，其中简化动力学模型的MPC解用于预热odynamics MPC，该方法考虑了非线性质心动态和运动学约束。提出的办法通过使用UPS在支撑阶段减少峰值电机扭矩和能量消耗，从而实现能量高效的动态跳跃。仿真实验表明，配备UPS的单腿机器人在高速跳跃时运输成本减少了38.8%。此外，初步硬件实验显示能量消耗减少了14.8%。视频：这个链接https://这个链接结尾被截断，请访问上述链接以获取视频内容。 

---
# BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities 

**Title (ZH)**: BEHAVIOR 机器人套件：简化日常家庭活动中的全身操纵 

**Authors**: Yunfan Jiang, Ruohan Zhang, Josiah Wong, Chen Wang, Yanjie Ze, Hang Yin, Cem Gokmen, Shuran Song, Jiajun Wu, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2503.05652)  

**Abstract**: Real-world household tasks present significant challenges for mobile manipulation robots. An analysis of existing robotics benchmarks reveals that successful task performance hinges on three key whole-body control capabilities: bimanual coordination, stable and precise navigation, and extensive end-effector reachability. Achieving these capabilities requires careful hardware design, but the resulting system complexity further complicates visuomotor policy learning. To address these challenges, we introduce the BEHAVIOR Robot Suite (BRS), a comprehensive framework for whole-body manipulation in diverse household tasks. Built on a bimanual, wheeled robot with a 4-DoF torso, BRS integrates a cost-effective whole-body teleoperation interface for data collection and a novel algorithm for learning whole-body visuomotor policies. We evaluate BRS on five challenging household tasks that not only emphasize the three core capabilities but also introduce additional complexities, such as long-range navigation, interaction with articulated and deformable objects, and manipulation in confined spaces. We believe that BRS's integrated robotic embodiment, data collection interface, and learning framework mark a significant step toward enabling real-world whole-body manipulation for everyday household tasks. BRS is open-sourced at this https URL 

**Abstract (ZH)**: 家庭任务中的实际挑战为移动操作机器人带来了重大困难。现有的机器人基准分析表明，任务的成功执行取决于三项核心的整体身体控制能力：双臂协调、稳定而精确的导航以及广泛的末端执行器可达性。实现这些能力需要仔细设计硬件，但由此产生的系统复杂性进一步 complicates 视觉运动策略学习。为了应对这些挑战，我们引入了 BEHAVIOR 机器人套件（BRS），这是一个全面的家庭任务复杂操作框架。BRS 以双臂轮式机器人和4自由度躯干为基础，集成了一个经济高效的全身远程操作接口以收集数据，并提出了一种新的算法来学习全身视觉运动策略。我们在五个具有挑战性的家庭任务中评估了 BRS，这些任务不仅强调了三个核心能力，还引入了额外的复杂性，如长距离导航、与关节和变形物体的交互以及在受限空间中的操作。我们相信，BRS 的集成机器人实体、数据采集接口和学习框架标志着向实现日常生活家庭任务中全身操作能力迈出的重要一步。BRS 已开源，参见 this https URL。 

---
# dARt Vinci: Egocentric Data Collection for Surgical Robot Learning at Scale 

**Title (ZH)**: 达芬奇_egocentric数据采集：面向手术机器人规模化学习 

**Authors**: Yihao Liu, Yu-Chun Ku, Jiaming Zhang, Hao Ding, Peter Kazanzides, Mehran Armand  

**Link**: [PDF](https://arxiv.org/pdf/2503.05646)  

**Abstract**: Data scarcity has long been an issue in the robot learning community. Particularly, in safety-critical domains like surgical applications, obtaining high-quality data can be especially difficult. It poses challenges to researchers seeking to exploit recent advancements in reinforcement learning and imitation learning, which have greatly improved generalizability and enabled robots to conduct tasks autonomously. We introduce dARt Vinci, a scalable data collection platform for robot learning in surgical settings. The system uses Augmented Reality (AR) hand tracking and a high-fidelity physics engine to capture subtle maneuvers in primitive surgical tasks: By eliminating the need for a physical robot setup and providing flexibility in terms of time, space, and hardware resources-such as multiview sensors and actuators-specialized simulation is a viable alternative. At the same time, AR allows the robot data collection to be more egocentric, supported by its body tracking and content overlaying capabilities. Our user study confirms the proposed system's efficiency and usability, where we use widely-used primitive tasks for training teleoperation with da Vinci surgical robots. Data throughput improves across all tasks compared to real robot settings by 41% on average. The total experiment time is reduced by an average of 10%. The temporal demand in the task load survey is improved. These gains are statistically significant. Additionally, the collected data is over 400 times smaller in size, requiring far less storage while achieving double the frequency. 

**Abstract (ZH)**: 数据稀缺一直是机器人学习领域的一个问题。特别是在如手术应用这样的安全关键领域，获取高质量的数据尤为困难。这给致力于利用强化学习和imitation learning等最近进步的研究人员带来了挑战，这些进步极大地提高了泛化能力并使机器人能够自主执行任务。我们提出了dARt Vinci，一种适用于手术场景的机器人学习数据收集平台。该系统利用增强现实（AR）手部追踪和高度逼真的物理引擎来捕捉原始手术任务中的微妙操作：通过消除物理机器人设置的需要，并在时间、空间和硬件资源（如多视角传感器和执行器）方面提供灵活性，专门的模拟成为一种可行的替代方案。同时，AR使机器人数据收集更加以自我为中心，其身体追踪和内容叠加能力为此提供了支持。我们的用户研究证实了该系统的效率和易用性，我们使用广泛采用的原始任务来训练与da Vinci手术机器人相关的远程操作。与实际机器人设置相比，所有任务的数据传输速度平均提高了41%。实验总时间平均减少了10%。任务负载调查中的时间需求也得到了改善。这些收益具有统计显著性。此外，收集的数据大小超过400倍较小，占用的存储空间大大减少，但同时实现了两倍的频率。 

---
# Learning and generalization of robotic dual-arm manipulation of boxes from demonstrations via Gaussian Mixture Models (GMMs) 

**Title (ZH)**: 基于高斯混合模型（GMMs）从演示学习和推广双臂机器人搬运箱子的能力 

**Authors**: Qian Ying Lee, Suhas Raghavendra Kulkarni, Kenzhi Iskandar Wong, Lin Yang, Bernardo Noronha, Yongjun Wee, Tzu-Yi Hung, Domenico Campolo  

**Link**: [PDF](https://arxiv.org/pdf/2503.05619)  

**Abstract**: Learning from demonstration (LfD) is an effective method to teach robots to move and manipulate objects in a human-like manner. This is especially true when dealing with complex robotic systems, such as those with dual arms employed for their improved payload capacity and manipulability. However, a key challenge is in expanding the robotic movements beyond the learned scenarios to adapt to minor and major variations from the specific demonstrations. In this work, we propose a learning and novel generalization approach that adapts the learned Gaussian Mixture Model (GMM)-parameterized policy derived from human demonstrations. Our method requires only a small number of human demonstrations and eliminates the need for a robotic system during the demonstration phase, which can significantly reduce both cost and time. The generalization process takes place directly in the parameter space, leveraging the lower-dimensional representation of GMM parameters. With only three parameters per Gaussian component, this process is computationally efficient and yields immediate results upon request. We validate our approach through real-world experiments involving a dual-arm robotic manipulation of boxes. Starting with just five demonstrations for a single task, our approach successfully generalizes to new unseen scenarios, including new target locations, orientations, and box sizes. These results highlight the practical applicability and scalability of our method for complex manipulations. 

**Abstract (ZH)**: 基于演示学习（LfD）在复杂双臂机器人操作中的有效学习与新颖泛化方法 

---
# InDRiVE: Intrinsic Disagreement based Reinforcement for Vehicle Exploration through Curiosity Driven Generalized World Model 

**Title (ZH)**: InDRiVE: 内在分歧基于强化学习的车辆好奇心驱动 generalize世界模型探索 

**Authors**: Feeza Khan Khanzada, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2503.05573)  

**Abstract**: Model-based Reinforcement Learning (MBRL) has emerged as a promising paradigm for autonomous driving, where data efficiency and robustness are critical. Yet, existing solutions often rely on carefully crafted, task specific extrinsic rewards, limiting generalization to new tasks or environments. In this paper, we propose InDRiVE (Intrinsic Disagreement based Reinforcement for Vehicle Exploration), a method that leverages purely intrinsic, disagreement based rewards within a Dreamer based MBRL framework. By training an ensemble of world models, the agent actively explores high uncertainty regions of environments without any task specific feedback. This approach yields a task agnostic latent representation, allowing for rapid zero shot or few shot fine tuning on downstream driving tasks such as lane following and collision avoidance. Experimental results in both seen and unseen environments demonstrate that InDRiVE achieves higher success rates and fewer infractions compared to DreamerV2 and DreamerV3 baselines despite using significantly fewer training steps. Our findings highlight the effectiveness of purely intrinsic exploration for learning robust vehicle control behaviors, paving the way for more scalable and adaptable autonomous driving systems. 

**Abstract (ZH)**: 基于模型的强化学习（MBRL）在自主驾驶中 emerged as a promising paradigm where 数据效率和鲁棒性至关重要。然而，现有解决方案往往依赖于精心设计的任务特定外在奖励，限制了其对新任务或新环境的泛化能力。在本文中，我们提出了一种名为 InDRiVE（基于内在分歧的车辆探索强化学习）的方法，该方法在基于 Dreamer 的 MBRL 框架中利用纯粹内在的分歧基奖励。通过训练一组世界模型，代理能够在没有任何任务特定反馈的情况下主动探索环境中的高不确定性区域。此方法生成了任务无关的潜在表示，允许多任务的下游驾驶任务（如车道跟随和碰撞避免）的快速零样本或少样本微调。在已见和未见环境的实验结果表明，尽管使用了显著更少的训练步骤，InDRiVE 在成功率和违规次数方面优于 DreamerV2 和 DreamerV3 基线。我们的研究结果突显了纯粹内在探索在学习鲁棒车辆控制行为方面的有效性，为更可扩展和适应性强的自主驾驶系统铺平了道路。 

---
# Self-Modeling Robots by Photographing 

**Title (ZH)**: 自建模机器人：通过拍照实现 

**Authors**: Kejun Hu, Peng Yu, Ning Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05398)  

**Abstract**: Self-modeling enables robots to build task-agnostic models of their morphology and kinematics based on data that can be automatically collected, with minimal human intervention and prior information, thereby enhancing machine intelligence. Recent research has highlighted the potential of data-driven technology in modeling the morphology and kinematics of robots. However, existing self-modeling methods suffer from either low modeling quality or excessive data acquisition costs. Beyond morphology and kinematics, texture is also a crucial component of robots, which is challenging to model and remains unexplored. In this work, a high-quality, texture-aware, and link-level method is proposed for robot self-modeling. We utilize three-dimensional (3D) Gaussians to represent the static morphology and texture of robots, and cluster the 3D Gaussians to construct neural ellipsoid bones, whose deformations are controlled by the transformation matrices generated by a kinematic neural network. The 3D Gaussians and kinematic neural network are trained using data pairs composed of joint angles, camera parameters and multi-view images without depth information. By feeding the kinematic neural network with joint angles, we can utilize the well-trained model to describe the corresponding morphology, kinematics and texture of robots at the link level, and render robot images from different perspectives with the aid of 3D Gaussian splatting. Furthermore, we demonstrate that the established model can be exploited to perform downstream tasks such as motion planning and inverse kinematics. 

**Abstract (ZH)**: 基于数据的自建模技术使机器人能够构建与任务无关的形态和运动学模型，基于可自动收集的数据，无需大量人工干预和先验信息，从而提升机器智能。近期研究突显了基于数据技术在建模机器人形态和运动学方面的潜力。然而，现有自建模方法要么建模质量较低，要么数据采集成本过高。除了形态和运动学，纹理也是机器人的重要组成部分，其建模具有挑战性且尚未得到探索。本文提出了一种高质量、纹理感知且关节级的机器人自建模方法。我们使用三维高斯分布表示机器人的静态形态和纹理，并聚类三维高斯分布以构建神经椭球骨骼，其变形由由运动学神经网络生成的变换矩阵控制。三维高斯分布和运动学神经网络使用关节角度、相机参数和多视角图像（无深度信息）的数据对进行训练。通过向运动学神经网络输入关节角度，可以利用训练良好的模型在关节级描述机器人的形态、运动学和纹理，并借助三维高斯点绘制以不同视角渲染机器人图像。此外，我们证明建立的模型可用于执行诸如运动规划和逆运动学等下游任务。 

---
# CoinRobot: Generalized End-to-end Robotic Learning for Physical Intelligence 

**Title (ZH)**: CoinRobot: 通用端到端机器人学习用于物理智能 

**Authors**: Yu Zhao, Huxian Liu, Xiang Chen, Jiankai Sun, Jiahuan Yan, Luhui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05316)  

**Abstract**: Physical intelligence holds immense promise for advancing embodied intelligence, enabling robots to acquire complex behaviors from demonstrations. However, achieving generalization and transfer across diverse robotic platforms and environments requires careful design of model architectures, training strategies, and data diversity. Meanwhile existing systems often struggle with scalability, adaptability to heterogeneous hardware, and objective evaluation in real-world settings. We present a generalized end-to-end robotic learning framework designed to bridge this gap. Our framework introduces a unified architecture that supports cross-platform adaptability, enabling seamless deployment across industrial-grade robots, collaborative arms, and novel embodiments without task-specific modifications. By integrating multi-task learning with streamlined network designs, it achieves more robust performance than conventional approaches, while maintaining compatibility with varying sensor configurations and action spaces. We validate our framework through extensive experiments on seven manipulation tasks. Notably, Diffusion-based models trained in our framework demonstrated superior performance and generalizability compared to the LeRobot framework, achieving performance improvements across diverse robotic platforms and environmental conditions. 

**Abstract (ZH)**: 物理智能为推进嵌入式智能提供了巨大潜力，使机器人能够通过演示获取复杂行为。然而，要在不同类型的机器人平台和环境中实现泛化和迁移需要精心设计模型架构、训练策略和数据多样性。同时，现有系统在可扩展性、异构硬件的适应性以及现实环境中的目标评估方面经常面临挑战。我们提出了一种通用的端到端机器人学习框架，旨在弥合这一差距。该框架引入了一个统一的架构，支持跨平台适应性，能够在工业级机器人、协作臂和新型实体上无缝部署，无需针对特定任务进行修改。通过结合多任务学习和 streamlined 网络设计，该框架在保持与不同传感器配置和动作空间兼容的同时，实现了比传统方法更为稳健的性能。我们通过在七项操作任务上进行广泛的实验验证了该框架。值得注意的是，采用我们框架训练的扩散模型在多种机器人平台和环境条件下展现出优于 LeRobot 框架的性能和泛化能力。 

---
# A Helping (Human) Hand in Kinematic Structure Estimation 

**Title (ZH)**: 协助(人类)进行运动结构估计 

**Authors**: Adrian Pfisterer, Xing Li, Vito Mengers, Oliver Brock  

**Link**: [PDF](https://arxiv.org/pdf/2503.05301)  

**Abstract**: Visual uncertainties such as occlusions, lack of texture, and noise present significant challenges in obtaining accurate kinematic models for safe robotic manipulation. We introduce a probabilistic real-time approach that leverages the human hand as a prior to mitigate these uncertainties. By tracking the constrained motion of the human hand during manipulation and explicitly modeling uncertainties in visual observations, our method reliably estimates an object's kinematic model online. We validate our approach on a novel dataset featuring challenging objects that are occluded during manipulation and offer limited articulations for perception. The results demonstrate that by incorporating an appropriate prior and explicitly accounting for uncertainties, our method produces accurate estimates, outperforming two recent baselines by 195% and 140%, respectively. Furthermore, we demonstrate that our approach's estimates are precise enough to allow a robot to manipulate even small objects safely. 

**Abstract (ZH)**: 视觉不确定性，如遮挡、缺乏纹理和噪声，给安全的机器人操作获取准确的运动模型带来了巨大挑战。我们提出了一种概率实时方法，利用人类手部动作作为先验知识以减轻这些不确定性。通过在操作过程中跟踪人类手部的受限运动并明确建模视觉观测中的不确定性，该方法可以在线可靠地估算物体的运动模型。我们在一个包含操作过程中被遮挡且感知度受限的挑战性物体的新数据集上验证了该方法。结果显示，通过引入适当的先验知识并明确考虑不确定性，该方法能产生准确的估算结果，分别比两个最近的基线方法提高了195%和140%。此外，我们展示了该方法的估算精度足以使机器人能够安全操作小物体。 

---
# A Map-free Deep Learning-based Framework for Gate-to-Gate Monocular Visual Navigation aboard Miniaturized Aerial Vehicles 

**Title (ZH)**: 无地图深度学习导向的微型飞行器单目视觉导航框架 

**Authors**: Lorenzo Scarciglia, Antonio Paolillo, Daniele Palossi  

**Link**: [PDF](https://arxiv.org/pdf/2503.05251)  

**Abstract**: Palm-sized autonomous nano-drones, i.e., sub-50g in weight, recently entered the drone racing scenario, where they are tasked to avoid obstacles and navigate as fast as possible through gates. However, in contrast with their bigger counterparts, i.e., kg-scale drones, nano-drones expose three orders of magnitude less onboard memory and compute power, demanding more efficient and lightweight vision-based pipelines to win the race. This work presents a map-free vision-based (using only a monocular camera) autonomous nano-drone that combines a real-time deep learning gate detection front-end with a classic yet elegant and effective visual servoing control back-end, only relying on onboard resources. Starting from two state-of-the-art tiny deep learning models, we adapt them for our specific task, and after a mixed simulator-real-world training, we integrate and deploy them aboard our nano-drone. Our best-performing pipeline costs of only 24M multiply-accumulate operations per frame, resulting in a closed-loop control performance of 30 Hz, while achieving a gate detection root mean square error of 1.4 pixels, on our ~20k real-world image dataset. In-field experiments highlight the capability of our nano-drone to successfully navigate through 15 gates in 4 min, never crashing and covering a total travel distance of ~100m, with a peak flight speed of 1.9 m/s. Finally, to stress the generalization capability of our system, we also test it in a never-seen-before environment, where it navigates through gates for more than 4 min. 

**Abstract (ZH)**: 掌サイズ自主纳米无人机：基于单目视觉的无地图自主纳米无人机及其应用 

---
# Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects 

**Title (ZH)**: 持久对象高斯体绘制（POGS）用于追踪人类和机器人操作不规则形状物体 

**Authors**: Justin Yu, Kush Hari, Karim El-Refai, Arnav Dalal, Justin Kerr, Chung Min Kim, Richard Cheng, Muhammad Zubair Irshad, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2503.05189)  

**Abstract**: Tracking and manipulating irregularly-shaped, previously unseen objects in dynamic environments is important for robotic applications in manufacturing, assembly, and logistics. Recently introduced Gaussian Splats efficiently model object geometry, but lack persistent state estimation for task-oriented manipulation. We present Persistent Object Gaussian Splat (POGS), a system that embeds semantics, self-supervised visual features, and object grouping features into a compact representation that can be continuously updated to estimate the pose of scanned objects. POGS updates object states without requiring expensive rescanning or prior CAD models of objects. After an initial multi-view scene capture and training phase, POGS uses a single stereo camera to integrate depth estimates along with self-supervised vision encoder features for object pose estimation. POGS supports grasping, reorientation, and natural language-driven manipulation by refining object pose estimates, facilitating sequential object reset operations with human-induced object perturbations and tool servoing, where robots recover tool pose despite tool perturbations of up to 30°. POGS achieves up to 12 consecutive successful object resets and recovers from 80% of in-grasp tool perturbations. 

**Abstract (ZH)**: 在动态环境中跟踪和操控未见过的不规则形状物体对于制造、组装和物流领域的机器人应用至关重要。我们提出了一种持久对象高斯点系统（POGS），该系统将语义信息、自监督视觉特征和物体分组特征嵌入到紧凑的表示中，可以连续更新以估计扫描物体的姿态。POGS在无需昂贵的重新扫描或物体的先验CAD模型的情况下更新物体状态。在初始的多视角场景捕获和训练阶段之后，POGS使用单目立体相机结合自监督视觉编码特征来估计物体姿态。通过细化物体姿态估计，POGS支持抓取、重新定向和自然语言驱动的操控，实现基于人工干预的物体重定位操作和工具伺服控制，即使在工具姿态高达30°的干扰下，机器人也能恢复工具姿态。POGS能够连续成功进行多达12次物体重定位操作，并从80%的抓持中工具干扰中恢复。 

---
# Look Before You Leap: Using Serialized State Machine for Language Conditioned Robotic Manipulation 

**Title (ZH)**: 未雨绸缪：使用序列化状态机进行语言条件驱动的机器人操作 

**Authors**: Tong Mu, Yihao Liu, Mehran Armand  

**Link**: [PDF](https://arxiv.org/pdf/2503.05114)  

**Abstract**: Imitation learning frameworks for robotic manipulation have drawn attention in the recent development of language model grounded robotics. However, the success of the frameworks largely depends on the coverage of the demonstration cases: When the demonstration set does not include examples of how to act in all possible situations, the action may fail and can result in cascading errors. To solve this problem, we propose a framework that uses serialized Finite State Machine (FSM) to generate demonstrations and improve the success rate in manipulation tasks requiring a long sequence of precise interactions. To validate its effectiveness, we use environmentally evolving and long-horizon puzzles that require long sequential actions. Experimental results show that our approach achieves a success rate of up to 98 in these tasks, compared to the controlled condition using existing approaches, which only had a success rate of up to 60, and, in some tasks, almost failed completely. 

**Abstract (ZH)**: 基于语言模型导向的机器人操控仿存学习框架：通过序列化有限状态机生成示范以提高成功率 

---
# Multi-Robot Collaboration through Reinforcement Learning and Abstract Simulation 

**Title (ZH)**: 通过强化学习和抽象模拟的多机器人协作 

**Authors**: Adam Labiosa, Josiah P. Hanna  

**Link**: [PDF](https://arxiv.org/pdf/2503.05092)  

**Abstract**: Teams of people coordinate to perform complex tasks by forming abstract mental models of world and agent dynamics. The use of abstract models contrasts with much recent work in robot learning that uses a high-fidelity simulator and reinforcement learning (RL) to obtain policies for physical robots. Motivated by this difference, we investigate the extent to which so-called abstract simulators can be used for multi-agent reinforcement learning (MARL) and the resulting policies successfully deployed on teams of physical robots. An abstract simulator models the robot's target task at a high-level of abstraction and discards many details of the world that could impact optimal decision-making. Policies are trained in an abstract simulator then transferred to the physical robot by making use of separately-obtained low-level perception and motion control modules. We identify three key categories of modifications to the abstract simulator that enable policy transfer to physical robots: simulation fidelity enhancements, training optimizations and simulation stochasticity. We then run an empirical study with extensive ablations to determine the value of each modification category for enabling policy transfer in cooperative robot soccer tasks. We also compare the performance of policies produced by our method with a well-tuned non-learning-based behavior architecture from the annual RoboCup competition and find that our approach leads to a similar level of performance. Broadly we show that MARL can be use to train cooperative physical robot behaviors using highly abstract models of the world. 

**Abstract (ZH)**: 团队成员通过构建抽象的心理模型来协作完成复杂任务，这些模型概括了世界和代理的动力学。与机器人学习中广泛使用的高保真仿真器和强化学习（RL）获得物理机器人策略的方法不同，我们探索所谓的抽象仿真器在多智能体强化学习（MARL）中的应用及其生成的策略在物理机器人团队上的部署效果。抽象仿真器以高层次的抽象概括机器人目标任务，摒弃了许多可能影响最优决策的世界细节。策略在抽象仿真器中训练，然后通过使用分别获得的低级感知和运动控制模块转移到物理机器人。我们确定了三种关键的抽象仿真器修改类别，这些修改能促进策略向物理机器人的转移：仿真保真度提升、训练优化和仿真随机性。接着，我们进行了一系列详尽的实验消融分析，以确定每个修改类别在协同机器人足球任务中促进策略转移的价值。我们还将通过本方法生成的策略性能与年度RoboCup竞赛中 WELL 调参的非学习基于行为架构进行比较，发现我们的方法能达到相似的性能水平。总体而言，我们展示了MARL可以使用高度抽象的世界模型来训练协同物理机器人的行为。 

---
# An End-to-End Learning-Based Multi-Sensor Fusion for Autonomous Vehicle Localization 

**Title (ZH)**: 基于端到端学习的多传感器融合自主车辆定位 

**Authors**: Changhong Lin, Jiarong Lin, Zhiqiang Sui, XiaoZhi Qu, Rui Wang, Kehua Sheng, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05088)  

**Abstract**: Multi-sensor fusion is essential for autonomous vehicle localization, as it is capable of integrating data from various sources for enhanced accuracy and reliability. The accuracy of the integrated location and orientation depends on the precision of the uncertainty modeling. Traditional methods of uncertainty modeling typically assume a Gaussian distribution and involve manual heuristic parameter tuning. However, these methods struggle to scale effectively and address long-tail scenarios. To address these challenges, we propose a learning-based method that encodes sensor information using higher-order neural network features, thereby eliminating the need for uncertainty estimation. This method significantly eliminates the need for parameter fine-tuning by developing an end-to-end neural network that is specifically designed for multi-sensor fusion. In our experiments, we demonstrate the effectiveness of our approach in real-world autonomous driving scenarios. Results show that the proposed method outperforms existing multi-sensor fusion methods in terms of both accuracy and robustness. A video of the results can be viewed at this https URL. 

**Abstract (ZH)**: 多传感器融合对于自动驾驶车辆定位是必不可少的，因为它能够通过集成多种来源的数据来增强精度和可靠性。集成的位置和方向的准确性取决于不确定性建模的精度。传统的不确定性建模方法通常假设高斯分布，并涉及手动经验参数调整。然而，这些方法难以有效扩展并应对长尾场景。为了解决这些挑战，我们提出了一种基于学习的方法，使用高阶神经网络特征编码传感器信息，从而消除了不确定性估计的需求。该方法通过开发一个特定于多传感器融合的端到端神经网络，显著减少了参数微调的需要。在我们的实验中，我们展示了该方法在真实世界的自动驾驶场景中的有效性。结果显示，与现有的多传感器融合方法相比，所提出的方法在准确性和稳健性方面表现更优。结果视频请访问此链接：[此处链接]。 

---
# Perceiving, Reasoning, Adapting: A Dual-Layer Framework for VLM-Guided Precision Robotic Manipulation 

**Title (ZH)**: 感知、推理、适应：面向VLM引导的精确机器人操作的双层框架 

**Authors**: Qingxuan Jia, Guoqin Tang, Zeyuan Huang, Zixuan Hao, Ning Ji, Shihang, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05064)  

**Abstract**: Vision-Language Models (VLMs) demonstrate remarkable potential in robotic manipulation, yet challenges persist in executing complex fine manipulation tasks with high speed and precision. While excelling at high-level planning, existing VLM methods struggle to guide robots through precise sequences of fine motor actions. To address this limitation, we introduce a progressive VLM planning algorithm that empowers robots to perform fast, precise, and error-correctable fine manipulation. Our method decomposes complex tasks into sub-actions and maintains three key data structures: task memory structure, 2D topology graphs, and 3D spatial networks, achieving high-precision spatial-semantic fusion. These three components collectively accumulate and store critical information throughout task execution, providing rich context for our task-oriented VLM interaction mechanism. This enables VLMs to dynamically adjust guidance based on real-time feedback, generating precise action plans and facilitating step-wise error correction. Experimental validation on complex assembly tasks demonstrates that our algorithm effectively guides robots to rapidly and precisely accomplish fine manipulation in challenging scenarios, significantly advancing robot intelligence for precision tasks. 

**Abstract (ZH)**: Vision-Language模型在机器人精细操作中的潜力显著，但在执行复杂精细操作任务时仍面临高速度和高精度的挑战。现有的VLM方法在高层次规划方面表现出色，但在指导机器人执行精细动作的精确序列方面存在局限性。为解决这一局限性，我们提出了一种渐进式VLM规划算法，使机器人能够进行快速、精确且可纠正误差的精细操作。我们的方法将复杂任务分解为子动作，并维护三种关键数据结构：任务记忆结构、2D拓扑图和3D空间网络，实现高精度的空间语义融合。这三种组件在整个任务执行过程中持续积累和存储关键信息，为我们的任务导向VLM交互机制提供丰富的上下文。这使得VLM能够根据实时反馈动态调整指导，生成精确的动作计划并促进逐步错误纠正。在复杂装配任务上的实验验证表明，我们的算法能够有效地指导机器人在挑战性场景中快速且精确地完成精细操作，显著推进了机器人在精确任务中的智能水平。 

---
# QuietPaw: Learning Quadrupedal Locomotion with Versatile Noise Preference Alignment 

**Title (ZH)**: QuietPaw: 学习多用途噪声偏好对齐的四足运动控制 

**Authors**: Yuyou Zhang, Yihang Yao, Shiqi Liu, Yaru Niu, Changyi Lin, Yuxiang Yang, Wenhao Yu, Tingnan Zhang, Jie Tan, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05035)  

**Abstract**: When operating at their full capacity, quadrupedal robots can produce loud footstep noise, which can be disruptive in human-centered environments like homes, offices, and hospitals. As a result, balancing locomotion performance with noise constraints is crucial for the successful real-world deployment of quadrupedal robots. However, achieving adaptive noise control is challenging due to (a) the trade-off between agility and noise minimization, (b) the need for generalization across diverse deployment conditions, and (c) the difficulty of effectively adjusting policies based on noise requirements. We propose QuietPaw, a framework incorporating our Conditional Noise-Constrained Policy (CNCP), a constrained learning-based algorithm that enables flexible, noise-aware locomotion by conditioning policy behavior on noise-reduction levels. We leverage value representation decomposition in the critics, disentangling state representations from condition-dependent representations and this allows a single versatile policy to generalize across noise levels without retraining while improving the Pareto trade-off between agility and noise reduction. We validate our approach in simulation and the real world, demonstrating that CNCP can effectively balance locomotion performance and noise constraints, achieving continuously adjustable noise reduction. 

**Abstract (ZH)**: 四足机器人全速运行时会产生较大的脚步噪音，这在以人类为中心的环境，如家庭、办公室和医院中可能会造成干扰。因此，在噪声限制条件下平衡四足机器人运动性能对于其实用部署至关重要。但由于（a）敏捷性与噪声最小化之间的权衡，（b）在多样化部署条件下的一般化需求，以及（c）基于噪声要求有效调整策略的难度，实现适应性的噪声控制具有挑战性。我们提出了QuietPaw框架，该框架包含我们的条件噪声约束策略（CNCP），这是一种基于约束学习的算法，可通过根据噪声减少水平调整策略行为，实现灵活且噪声感知的运动。我们利用评论者的价值表示分解，将状态表示与条件依赖表示分离，使得单个通用策略可以在无需重新训练的情况下泛化到不同的噪声水平，并提高敏捷性与噪声减少之间的帕累托权衡。我们在仿真和实际环境中验证了这种方法，证明CNCP能够有效平衡运动性能和噪声约束，实现连续可调的噪声减少。 

---
# Multi-Agent Ergodic Exploration under Smoke-Based, Time-Varying Sensor Visibility Constraints 

**Title (ZH)**: 基于烟雾引起的时间 varying 传感器可见性约束的多智能体遍历探索 

**Authors**: Elena Wittemyer, Ananya Rao, Ian Abraham, Howie Choset  

**Link**: [PDF](https://arxiv.org/pdf/2503.04998)  

**Abstract**: In this work, we consider the problem of multi-agent informative path planning (IPP) for robots whose sensor visibility continuously changes as a consequence of a time-varying natural phenomenon. We leverage ergodic trajectory optimization (ETO), which generates paths such that the amount of time an agent spends in an area is proportional to the expected information in that area. We focus specifically on the problem of multi-agent drone search of a wildfire, where we use the time-varying environmental process of smoke diffusion to construct a sensor visibility model. This sensor visibility model is used to repeatedly calculate an expected information distribution (EID) to be used in the ETO algorithm. Our experiments show that our exploration method achieves improved information gathering over both baseline search methods and naive ergodic search formulations. 

**Abstract (ZH)**: 本研究考虑了传感器可视性因时间 varying 自然现象而连续变化的机器人多智能体信息路径规划问题。我们利用遍历轨迹优化（ETO），生成路径使得智能体在某一区域停留的时间与其预期信息量成正比。我们具体关注野火多智能体无人机搜索问题，利用烟雾扩散的时间 varying 环境过程构建传感器可视性模型。该传感器可视性模型用于反复计算预期信息分布（EID），并应用于ETO算法。实验表明，我们的探索方法在信息收集方面优于基准搜索方法和简单的遍历搜索形式。 

---
# Data-Efficient Learning from Human Interventions for Mobile Robots 

**Title (ZH)**: 基于人类干预的数据高效学习在移动机器人中的应用 

**Authors**: Zhenghao Peng, Zhizheng Liu, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04969)  

**Abstract**: Mobile robots are essential in applications such as autonomous delivery and hospitality services. Applying learning-based methods to address mobile robot tasks has gained popularity due to its robustness and generalizability. Traditional methods such as Imitation Learning (IL) and Reinforcement Learning (RL) offer adaptability but require large datasets, carefully crafted reward functions, and face sim-to-real gaps, making them challenging for efficient and safe real-world deployment. We propose an online human-in-the-loop learning method PVP4Real that combines IL and RL to address these issues. PVP4Real enables efficient real-time policy learning from online human intervention and demonstration, without reward or any pretraining, significantly improving data efficiency and training safety. We validate our method by training two different robots -- a legged quadruped, and a wheeled delivery robot -- in two mobile robot tasks, one of which even uses raw RGBD image as observation. The training finishes within 15 minutes. Our experiments show the promising future of human-in-the-loop learning in addressing the data efficiency issue in real-world robotic tasks. More information is available at: this https URL 

**Abstract (ZH)**: 移动机器人在自主配送和 Hospitality 服务等应用中至关重要。基于学习的方法在解决移动机器人任务时由于其鲁棒性和泛化能力而变得流行。传统方法如模仿学习（IL）和强化学习（RL）具有适应性，但需要大量数据集、精心设计的奖励函数，并且面临从仿真到现实的差距，使得它们在高效的现实世界部署中具有挑战性。我们提出了一种名为 PVP4Real 的在线人类在环学习方法，该方法结合了 IL 和 RL 来解决这些问题。PVP4Real 允许通过实时的人类干预和示范高效地学习策略，无需任何奖励或预训练，显著提高了数据效率和训练安全性。我们通过训练两个不同类型的机器人——一个腿足四足机器人和一个轮式配送机器人——在两种移动机器人任务中验证了该方法，其中一个任务甚至使用原始 RGBD 图像作为观测。训练时间仅需 15 分钟。我们的实验展示了人类在环学习在解决实际机器人任务中的数据效率问题方面的光明前景。更多信息请参见：this https URL 

---
# Curiosity-Driven Imagination: Discovering Plan Operators and Learning Associated Policies for Open-World Adaptation 

**Title (ZH)**: 好奇心驱动的想象：发现开放世界适应中的计划操作并学习相关策略 

**Authors**: Pierrick Lorang, Hong Lu, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04931)  

**Abstract**: Adapting quickly to dynamic, uncertain environments-often called "open worlds"-remains a major challenge in robotics. Traditional Task and Motion Planning (TAMP) approaches struggle to cope with unforeseen changes, are data-inefficient when adapting, and do not leverage world models during learning. We address this issue with a hybrid planning and learning system that integrates two models: a low level neural network based model that learns stochastic transitions and drives exploration via an Intrinsic Curiosity Module (ICM), and a high level symbolic planning model that captures abstract transitions using operators, enabling the agent to plan in an "imaginary" space and generate reward machines. Our evaluation in a robotic manipulation domain with sequential novelty injections demonstrates that our approach converges faster and outperforms state-of-the-art hybrid methods. 

**Abstract (ZH)**: 快速适应动态和不确定的环境——通常称为“开放世界”——仍然是机器人技术中的一个重大挑战。传统的任务与运动规划（TAMP）方法难以应对意外变化，适应过程数据效率低，并且在学习过程中不利用世界模型。我们通过结合一个低级别的基于神经网络的模型和一个高级别的符号规划模型来解决这个问题：低级别的模型学习随机转移并借助内在好奇心模块（ICM）驱动探索，高级别的模型使用操作符捕捉抽象转移，使代理能够在“想象空间”中进行规划并生成奖励机器。在具有序列新颖性注入的机器人操作域中的评估表明，我们的方法更快地收敛并且优于最先进的混合方法。 

---
# Modeling Dynamic Hand-Object Interactions with Applications to Human-Robot Handovers 

**Title (ZH)**: 基于动态手-物体交互建模及其在人-机器人交接中的应用 

**Authors**: Sammy Christen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04879)  

**Abstract**: Humans frequently grasp, manipulate, and move objects. Interactive systems assist humans in these tasks, enabling applications in Embodied AI, human-robot interaction, and virtual reality. However, current methods in hand-object synthesis often neglect dynamics and focus on generating static grasps. The first part of this dissertation introduces dynamic grasp synthesis, where a hand grasps and moves an object to a target pose. We approach this task using physical simulation and reinforcement learning. We then extend this to bimanual manipulation and articulated objects, requiring fine-grained coordination between hands. In the second part of this dissertation, we study human-to-robot handovers. We integrate captured human motion into simulation and introduce a student-teacher framework that adapts to human behavior and transfers from sim to real. To overcome data scarcity, we generate synthetic interactions, increasing training diversity by 100x. Our user study finds no difference between policies trained on synthetic vs. real motions. 

**Abstract (ZH)**: 人类频繁地抓握、操作和移动物体。交互系统协助人类完成这些任务，推动了具身AI、人机交互和虚拟现实等应用的发展。然而，当前的手物合成方法往往忽视了动力学效果，专注于生成静态抓握。本文的第一部分介绍了动态抓握合成方法，该方法涉及手抓住并移动物体到达目标姿态。我们采用了物理仿真和强化学习来完成这项任务。随后，我们将这种方法扩展到双臂操作和有连杆的物体，需要双手之间细致的协调。本文的第二部分研究了人向机器人传递手部动作的问题。我们整合了捕捉到的人类动作到仿真中，并引入了一种学生-教师框架，该框架能够适应人类行为并在仿真到现实场景之间进行转移。为了解决数据稀缺问题，我们生成了合成交互，将训练多样性提高了100倍。我们的用户研究发现，基于合成动作为训练政策与基于真实动作训练政策之间没有差异。 

---
# Runtime Learning of Quadruped Robots in Wild Environments 

**Title (ZH)**: 野生环境中四足机器人运行时学习 

**Authors**: Yihao Cai, Yanbing Mao, Lui Sha, Hongpeng Cao, Marco Caccamo  

**Link**: [PDF](https://arxiv.org/pdf/2503.04794)  

**Abstract**: This paper presents a runtime learning framework for quadruped robots, enabling them to learn and adapt safely in dynamic wild environments. The framework integrates sensing, navigation, and control, forming a closed-loop system for the robot. The core novelty of this framework lies in two interactive and complementary components within the control module: the high-performance (HP)-Student and the high-assurance (HA)-Teacher. HP-Student is a deep reinforcement learning (DRL) agent that engages in self-learning and teaching-to-learn to develop a safe and high-performance action policy. HA-Teacher is a simplified yet verifiable physics-model-based controller, with the role of teaching HP-Student about safety while providing a backup for the robot's safe locomotion. HA-Teacher is innovative due to its real-time physics model, real-time action policy, and real-time control goals, all tailored to respond effectively to real-time wild environments, ensuring safety. The framework also includes a coordinator who effectively manages the interaction between HP-Student and HA-Teacher. Experiments involving a Unitree Go2 robot in Nvidia Isaac Gym and comparisons with state-of-the-art safe DRLs demonstrate the effectiveness of the proposed runtime learning framework. 

**Abstract (ZH)**: 一种用于四足机器人的运行时学习框架：安全适应动态野外地形的能力 

---
# Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning 

**Title (ZH)**: Adapt3R: 适应性三维场景表示在模仿学习领域迁移中的应用 

**Authors**: Albert Wilcox, Mohamed Ghanem, Masoud Moghani, Pierre Barroso, Benjamin Joffe, Animesh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2503.04877)  

**Abstract**: Imitation Learning (IL) has been very effective in training robots to perform complex and diverse manipulation tasks. However, its performance declines precipitously when the observations are out of the training distribution. 3D scene representations that incorporate observations from calibrated RGBD cameras have been proposed as a way to improve generalizability of IL policies, but our evaluations in cross-embodiment and novel camera pose settings found that they show only modest improvement. To address those challenges, we propose Adaptive 3D Scene Representation (Adapt3R), a general-purpose 3D observation encoder which uses a novel architecture to synthesize data from one or more RGBD cameras into a single vector that can then be used as conditioning for arbitrary IL algorithms. The key idea is to use a pretrained 2D backbone to extract semantic information about the scene, using 3D only as a medium for localizing this semantic information with respect to the end-effector. We show that when trained end-to-end with several SOTA multi-task IL algorithms, Adapt3R maintains these algorithms' multi-task learning capacity while enabling zero-shot transfer to novel embodiments and camera poses. Furthermore, we provide a detailed suite of ablation and sensitivity experiments to elucidate the design space for point cloud observation encoders. 

**Abstract (ZH)**: 仿生学习（IL）在训练机器人执行复杂多样的操作任务方面非常有效。然而，当观测数据超出训练分布时，其性能会急剧下降。带有校准RGBD摄像头观测的3D场景表示已被提出以提高IL策略的泛化能力，但我们在跨体素和新型摄像头姿态设置下的评估发现，它们仅显示出适度的改进。为应对这些挑战，我们提出了自适应3D场景表示（Adapt3R），这是一种通用的3D观测编码器，使用一种新型架构将一个或多个RGBD摄像头的数据综合成一个向量，该向量可以作为任意IL算法的条件输入。核心理念是使用预训练的2D主干网络提取场景的语义信息，并仅使用3D信息作为相对于末端执行器局部化这些语义信息的媒介。我们展示了当与几种SOTA多任务IL算法端到端训练时，Adapt3R能够保持这些算法的多任务学习能力，并实现对新型体态和摄像头姿态的零样本迁移。此外，我们提供了详细的消融和敏感性实验来阐明点云观测编码器的设计空间。 

---
# High-Precision Transformer-Based Visual Servoing for Humanoid Robots in Aligning Tiny Objects 

**Title (ZH)**: 基于高精度Transformer的类人机器人微小物体对齐视觉伺服方法 

**Authors**: Jialong Xue, Wei Gao, Yu Wang, Chao Ji, Dongdong Zhao, Shi Yan, Shiwu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04862)  

**Abstract**: High-precision tiny object alignment remains a common and critical challenge for humanoid robots in real-world. To address this problem, this paper proposes a vision-based framework for precisely estimating and controlling the relative position between a handheld tool and a target object for humanoid robots, e.g., a screwdriver tip and a screw head slot. By fusing images from the head and torso cameras on a robot with its head joint angles, the proposed Transformer-based visual servoing method can correct the handheld tool's positional errors effectively, especially at a close distance. Experiments on M4-M8 screws demonstrate an average convergence error of 0.8-1.3 mm and a success rate of 93\%-100\%. Through comparative analysis, the results validate that this capability of high-precision tiny object alignment is enabled by the Distance Estimation Transformer architecture and the Multi-Perception-Head mechanism proposed in this paper. 

**Abstract (ZH)**: 基于视觉的高精度小型物体对齐框架： humanoid机器人手持工具与目标物体之间相对位置的精确估计与控制 

---
# Combined Physics and Event Camera Simulator for Slip Detection 

**Title (ZH)**: 滑动检测的联合物理与事件相机模拟器 

**Authors**: Thilo Reinold, Suman Ghosh, Guillermo Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2503.04838)  

**Abstract**: Robot manipulation is a common task in fields like industrial manufacturing. Detecting when objects slip from a robot's grasp is crucial for safe and reliable operation. Event cameras, which register pixel-level brightness changes at high temporal resolution (called ``events''), offer an elegant feature when mounted on a robot's end effector: since they only detect motion relative to their viewpoint, a properly grasped object produces no events, while a slipping object immediately triggers them. To research this feature, representative datasets are essential, both for analytic approaches and for training machine learning models. The majority of current research on slip detection with event-based data is done on real-world scenarios and manual data collection, as well as additional setups for data labeling. This can result in a significant increase in the time required for data collection, a lack of flexibility in scene setups, and a high level of complexity in the repetition of experiments. This paper presents a simulation pipeline for generating slip data using the described camera-gripper configuration in a robot arm, and demonstrates its effectiveness through initial data-driven experiments. The use of a simulator, once it is set up, has the potential to reduce the time spent on data collection, provide the ability to alter the setup at any time, simplify the process of repetition and the generation of arbitrarily large data sets. Two distinct datasets were created and validated through visual inspection and artificial neural networks (ANNs). Visual inspection confirmed photorealistic frame generation and accurate slip modeling, while three ANNs trained on this data achieved high validation accuracy and demonstrated good generalization capabilities on a separate test set, along with initial applicability to real-world data. Project page: this https URL 

**Abstract (ZH)**: 基于事件相机的机器人滑落数据生成仿真管道及其初步实验 

---
# The Society of HiveMind: Multi-Agent Optimization of Foundation Model Swarms to Unlock the Potential of Collective Intelligence 

**Title (ZH)**: HiveMind 社区：多 Agent 优化基础模型集群以释放集体智能的潜力 

**Authors**: Noah Mamie, Susie Xi Rao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05473)  

**Abstract**: Multi-agent systems address issues of accessibility and scalability of artificial intelligence (AI) foundation models, which are often represented by large language models. We develop a framework - the "Society of HiveMind" (SOHM) - that orchestrates the interaction between multiple AI foundation models, imitating the observed behavior of animal swarms in nature by following modern evolutionary theories. On the one hand, we find that the SOHM provides a negligible benefit on tasks that mainly require real-world knowledge. On the other hand, we remark a significant improvement on tasks that require intensive logical reasoning, indicating that multi-agent systems are capable of increasing the reasoning capabilities of the collective compared to the individual agents. Our findings demonstrate the potential of combining a multitude of diverse AI foundation models to form an artificial swarm intelligence capable of self-improvement through interactions with a given environment. 

**Abstract (ZH)**: 多智能体系统解决了人工智能基础模型在可访问性和可扩展性方面的限制，这些基础模型常常由大型语言模型表示。我们开发了一个框架——“蜂群社会”（SOHM）——它通过遵循现代进化理论来协调多个AI基础模型之间的交互，模仿自然界中动物集群的观察行为。一方面，我们发现SOHM在主要依赖于现实世界知识的任务上提供了微乎其微的优势。另一方面，在需要密集逻辑推理的任务上，我们注意到了显著的改进，表明多智能体系统能够提高集体的推理能力，超过个体代理的能力。我们的研究结果证明，通过与给定环境的交互，结合多种多样的人工智能基础模型，有可能形成一种能够自我改善的仿生群智能系统。 

---
# Adversarial Policy Optimization for Offline Preference-based Reinforcement Learning 

**Title (ZH)**: 基于离线偏好强化学习的对抗性策略优化 

**Authors**: Hyungkyu Kang, Min-hwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.05306)  

**Abstract**: In this paper, we study offline preference-based reinforcement learning (PbRL), where learning is based on pre-collected preference feedback over pairs of trajectories. While offline PbRL has demonstrated remarkable empirical success, existing theoretical approaches face challenges in ensuring conservatism under uncertainty, requiring computationally intractable confidence set constructions. We address this limitation by proposing Adversarial Preference-based Policy Optimization (APPO), a computationally efficient algorithm for offline PbRL that guarantees sample complexity bounds without relying on explicit confidence sets. By framing PbRL as a two-player game between a policy and a model, our approach enforces conservatism in a tractable manner. Using standard assumptions on function approximation and bounded trajectory concentrability, we derive a sample complexity bound. To our knowledge, APPO is the first offline PbRL algorithm to offer both statistical efficiency and practical applicability. Experimental results on continuous control tasks demonstrate that APPO effectively learns from complex datasets, showing comparable performance with existing state-of-the-art methods. 

**Abstract (ZH)**: 在本论文中，我们研究了基于离线偏好反馈的强化学习（PbRL），其中学习基于预先收集的轨迹对的偏好反馈。尽管离线PbRL在实验上取得了显著的成功，但现有的理论方法在确保不确定性下的保守性时面临挑战，需要构建计算上不可行的置信集。我们通过提出对抗性基于偏好的策略优化（APPO）算法解决了这一限制，该算法在不依赖显式置信集的情况下提供了样本复杂度界。通过将PbRL建模为策略和模型之间的两人游戏，我们的方法以可计算的方式保证了保守性。在函数逼近和轨迹集中性有标准假设的情况下，我们推导出了样本复杂度界。据我们所知，APPO是第一个同时提供统计效率和实际适用性的离线PbRL算法。实验结果在连续控制任务上表明，APPO能够有效地从复杂的数据集中学到，展示了与现有最先进的方法相当的性能。 

---
# Policy Constraint by Only Support Constraint for Offline Reinforcement Learning 

**Title (ZH)**: 仅基于约束的支持约束对offline reinforcement learning的策略进行限制 

**Authors**: Yunkai Gao, Jiaming Guo, Fan Wu, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05207)  

**Abstract**: Offline reinforcement learning (RL) aims to optimize a policy by using pre-collected datasets, to maximize cumulative rewards. However, offline reinforcement learning suffers challenges due to the distributional shift between the learned and behavior policies, leading to errors when computing Q-values for out-of-distribution (OOD) actions. To mitigate this issue, policy constraint methods aim to constrain the learned policy's distribution with the distribution of the behavior policy or confine action selection within the support of the behavior policy. However, current policy constraint methods tend to exhibit excessive conservatism, hindering the policy from further surpassing the behavior policy's performance. In this work, we present Only Support Constraint (OSC) which is derived from maximizing the total probability of learned policy in the support of behavior policy, to address the conservatism of policy constraint. OSC presents a regularization term that only restricts policies to the support without imposing extra constraints on actions within the support. Additionally, to fully harness the performance of the new policy constraints, OSC utilizes a diffusion model to effectively characterize the support of behavior policies. Experimental evaluations across a variety of offline RL benchmarks demonstrate that OSC significantly enhances performance, alleviating the challenges associated with distributional shifts and mitigating conservatism of policy constraints. Code is available at this https URL. 

**Abstract (ZH)**: 基于offline强化学习中的仅支持约束（Only Support Constraint，OSC）：缓解保守性并提升性能 

---
# Generative Trajectory Stitching through Diffusion Composition 

**Title (ZH)**: 通过扩散合成的生成轨迹拼接 

**Authors**: Yunhao Luo, Utkarsh A. Mishra, Yilun Du, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05153)  

**Abstract**: Effective trajectory stitching for long-horizon planning is a significant challenge in robotic decision-making. While diffusion models have shown promise in planning, they are limited to solving tasks similar to those seen in their training data. We propose CompDiffuser, a novel generative approach that can solve new tasks by learning to compositionally stitch together shorter trajectory chunks from previously seen tasks. Our key insight is modeling the trajectory distribution by subdividing it into overlapping chunks and learning their conditional relationships through a single bidirectional diffusion model. This allows information to propagate between segments during generation, ensuring physically consistent connections. We conduct experiments on benchmark tasks of various difficulties, covering different environment sizes, agent state dimension, trajectory types, training data quality, and show that CompDiffuser significantly outperforms existing methods. 

**Abstract (ZH)**: 长 horizon 规划中有效的轨迹拼接是一项重要的机器人决策挑战。虽然扩散模型在规划方面展现了潜力，但它们仅限于解决与其训练数据相似的任务。我们提出了 CompDiffuser，一种新颖的生成性方法，能够通过学习将先前见过的任务中的较短轨迹片段组合起来解决新任务。我们的关键洞察是通过将轨迹分布细分重叠片段，并通过单一双向扩散模型学习它们的条件关系来建模轨迹分布。这使得在生成过程中片段之间能够传播信息，确保物理上的一致性连接。我们在涵盖不同环境大小、代理状态维度、轨迹类型、训练数据质量等多种难度基准任务上进行了实验，并展示了 CompDiffuser 显著优于现有方法。 

---
# VQEL: Enabling Self-Developed Symbolic Language in Agents through Vector Quantization in Emergent Language Games 

**Title (ZH)**: VQEL：通过Emergent语言游戏中的向量量化实现自主开发符号语言的能力 

**Authors**: Mohammad Mahdi Samiei Paqaleh, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2503.04940)  

**Abstract**: In the field of emergent language, efforts have traditionally focused on developing communication protocols through interactions between agents in referential games. However, the aspect of internal language learning, where language serves not only as a communicative tool with others but also as a means for individual thinking, self-reflection, and problem-solving remains underexplored. Developing a language through self-play, without another agent's involvement, poses a unique challenge. It requires an agent to craft symbolic representations and train them using direct gradient methods. The challenge here is that if an agent attempts to learn symbolic representations through self-play using conventional modeling and techniques such as REINFORCE, the solution will offer no advantage over previous multi-agent approaches. We introduce VQEL, a novel method that incorporates Vector Quantization into the agents' architecture, enabling them to autonomously invent and develop discrete symbolic representations in a self-play referential game. Following the self-play phase, agents can enhance their language through reinforcement learning and interactions with other agents in the mutual-play phase. Our experiments across various datasets demonstrate that VQEL not only outperforms the traditional REINFORCE method but also benefits from improved control and reduced susceptibility to collapse, thanks to the incorporation of vector quantization. 

**Abstract (ZH)**: 在新兴语言领域，努力传统上集中在通过代理在指称游戏中相互作用来开发通信协议。然而，语言的内部学习方面仍然未被充分探索，即语言不仅作为与他人沟通的工具，也作为个体思考、自我反思和解决问题的手段。通过自我游戏发展语言而不涉及另一个代理带来了独特挑战。这要求代理构建象征性表示并使用直接梯度方法进行训练。问题是，如果代理试图通过自我游戏使用传统建模和技术（如REINFORCE）学习象征性表示，那么该解决方案将无法提供优于多代理方法的优势。我们提出了VQEL，这是一种新颖的方法，将向量量化融入代理的架构中，使代理能够自主发明和发展离散的象征性表示，在自我游戏指称游戏中。在自我游戏阶段之后，代理可以通过强化学习和在互游戏阶段与其他代理的交互来提升其语言能力。我们的跨多个数据集的实验表明，VQEL不仅优于传统的REINFORCE方法，而且由于引入了向量量化，还受益于更好的控制和较低的塌陷倾向。 

---
# An energy-efficient learning solution for the Agile Earth Observation Satellite Scheduling Problem 

**Title (ZH)**: 敏捷地球观测卫星调度问题的节能学习解决方案 

**Authors**: Antonio M. Mercado-Martínez, Beatriz Soret, Antonio Jurado-Navas  

**Link**: [PDF](https://arxiv.org/pdf/2503.04803)  

**Abstract**: The Agile Earth Observation Satellite Scheduling Problem (AEOSSP) entails finding the subset of observation targets to be scheduled along the satellite's orbit while meeting operational constraints of time, energy and memory. The problem of deciding what and when to observe is inherently complex, and becomes even more challenging when considering several issues that compromise the quality of the captured images, such as cloud occlusion, atmospheric turbulence, and image resolution. This paper presents a Deep Reinforcement Learning (DRL) approach for addressing the AEOSSP with time-dependent profits, integrating these three factors to optimize the use of energy and memory resources. The proposed method involves a dual decision-making process: selecting the sequence of targets and determining the optimal observation time for each. Our results demonstrate that the proposed algorithm reduces the capture of images that fail to meet quality requirements by > 60% and consequently decreases energy waste from attitude maneuvers by up to 78%, all while maintaining strong observation performance. 

**Abstract (ZH)**: 敏捷遥感卫星调度问题（AEOSSP）涉及在满足时间、能量和内存等操作约束条件下，确定卫星轨道上要调度的观测目标子集。决定观测什么以及何时观测问题本身是复杂的，在考虑诸如云遮挡、大气湍流和图像分辨率等因素影响图像质量的问题时，则变得更加具有挑战性。本文提出了一种基于深度强化学习（DRL）的方法来解决具有时间依赖性收益的AEOSSP问题，将这三个因素综合考虑以优化能量和内存资源的使用。所提出的方法包括双重决策过程：选择目标序列并确定每个目标的最佳观测时间。实验结果表明，所提出的算法可以将不符合质量要求的图像捕获量减少超过60%，从而将姿态机动的能量浪费降低高达78%同时保持强劲的观测性能。 

---
# Chat-GPT: An AI Based Educational Revolution 

**Title (ZH)**: Chat-GPT：基于AI的教育革命 

**Authors**: Sasa Maric, Sonja Maric, Lana Maric  

**Link**: [PDF](https://arxiv.org/pdf/2503.04758)  

**Abstract**: The AI revolution is gathering momentum at an unprecedented rate. Over the past decade, we have witnessed a seemingly inevitable integration of AI in every facet of our lives. Much has been written about the potential revolutionary impact of AI in education. AI has the potential to completely revolutionise the educational landscape as we could see entire courses and degrees developed by programs such as ChatGPT. AI has the potential to develop courses, set assignments, grade and provide feedback to students much faster than a team of teachers. In addition, because of its dynamic nature, it has the potential to continuously improve its content. In certain fields such as computer science, where technology is continuously evolving, AI based applications can provide dynamically changing, relevant material to students. AI has the potential to replace entire degrees and may challenge the concept of higher education institutions. We could also see entire new disciplines emerge as a consequence of AI. This paper examines the practical impact of ChatGPT and why it is believed that its implementation is a critical step towards a new era of education. We investigate the impact that ChatGPT will have on learning, problem solving skills and cognitive ability of students. We examine the positives, negatives and many other aspects of AI and its applications throughout this paper. 

**Abstract (ZH)**: AI革命正在以前所未有的速度加速到来。过去十年间，我们见证了AI在我们生活方方面面的似乎不可避免的融合。关于AI在教育领域潜在革命性影响的讨论甚嚣尘上。AI有潜力彻底改变我们所熟知的教育面貌，如同ChatGPT这样的程序可能开发出全新的课程和学位。AI能够比教师团队更快地开发课程、布置作业、评分并提供反馈。此外，由于其动态特性，它有潜力不断改进其内容。在计算机科学等技术不断发展的领域，基于AI的应用程序可以向学生提供动态变化的相关材料。AI有潜力取代整个学位课程，并可能挑战高等教育机构的概念。AI还可能引发全新的学科领域。本文探讨ChatGPT的实际影响，以及为什么其实施被认为是迈向教育新纪元的关键步骤。我们将研究ChatGPT对学生学习、解决问题能力和认知能力的影响。在本文中，我们还将探讨AI及其应用的诸多积极面、消极面及其他诸多方面。 

---
# Static Vs. Agentic Game Master AI for Facilitating Solo Role-Playing Experiences 

**Title (ZH)**: 静态 vs. 主动游戏大师AI在促进单人角色扮演体验中的应用 

**Authors**: Nicolai Hejlesen Jørgensen, Sarmilan Tharmabalan, Ilhan Aslan, Nicolai Brodersen Hansen, Timothy Merritt  

**Link**: [PDF](https://arxiv.org/pdf/2502.19519)  

**Abstract**: This paper presents a game master AI for single-player role-playing games. The AI is designed to deliver interactive text-based narratives and experiences typically associated with multiplayer tabletop games like Dungeons & Dragons. We report on the design process and the series of experiments to improve the functionality and experience design, resulting in two functional versions of the system. While v1 of our system uses simplified prompt engineering, v2 leverages a multi-agent architecture and the ReAct framework to include reasoning and action. A comparative evaluation demonstrates that v2 as an agentic system maintains play while significantly improving modularity and game experience, including immersion and curiosity. Our findings contribute to the evolution of AI-driven interactive fiction, highlighting new avenues for enhancing solo role-playing experiences. 

**Abstract (ZH)**: 本文提出了一种面向单人角色扮演游戏的场景主持AI。该AI旨在提供类似于多人桌面游戏（例如龙与地下城）的互动文本叙述和体验。我们报告了该设计过程以及一系列实验以改进功能和体验设计，最终实现了两个功能版本的系统。虽然系统v1采用简化的提示工程，但v2则利用多智能体架构和ReAct框架，加入了推理和行动。比较评估表明，v2作为一种自主系统在保持游戏性的同时显著提高了模块化程度和游戏体验，包括沉浸感和好奇心。我们的研究结果促进了由AI驱动的互动小说的发展，指出了增强单人角色扮演体验的新途径。 

---
