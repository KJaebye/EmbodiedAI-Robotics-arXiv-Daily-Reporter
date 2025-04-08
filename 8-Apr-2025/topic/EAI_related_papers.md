# Using Physiological Measures, Gaze, and Facial Expressions to Model Human Trust in a Robot Partner 

**Title (ZH)**: 使用生理指标、注视行为和面部表情建模人类对机器人伴侣的信任 

**Authors**: Haley N. Green, Tariq Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2504.05291)  

**Abstract**: With robots becoming increasingly prevalent in various domains, it has become crucial to equip them with tools to achieve greater fluency in interactions with humans. One of the promising areas for further exploration lies in human trust. A real-time, objective model of human trust could be used to maximize productivity, preserve safety, and mitigate failure. In this work, we attempt to use physiological measures, gaze, and facial expressions to model human trust in a robot partner. We are the first to design an in-person, human-robot supervisory interaction study to create a dedicated trust dataset. Using this dataset, we train machine learning algorithms to identify the objective measures that are most indicative of trust in a robot partner, advancing trust prediction in human-robot interactions. Our findings indicate that a combination of sensor modalities (blood volume pulse, electrodermal activity, skin temperature, and gaze) can enhance the accuracy of detecting human trust in a robot partner. Furthermore, the Extra Trees, Random Forest, and Decision Trees classifiers exhibit consistently better performance in measuring the person's trust in the robot partner. These results lay the groundwork for constructing a real-time trust model for human-robot interaction, which could foster more efficient interactions between humans and robots. 

**Abstract (ZH)**: 随着机器人在各个领域中的日益普及，亟需为它们配备工具以实现与人类互动的更高的流畅性。进一步探索的一个有前景的领域是人类的信任。实时客观的人类信任模型可用于最大化生产效率、保障安全并减少失败。在本项工作中，我们尝试使用生理测量、注视和面部表情来建模人类对机器人合作伙伴的信任。我们首次设计了一项面对面的人-机监督交互研究，创建了一套专门的信任数据集。利用此数据集，我们训练机器学习算法以识别最能体现对机器人合作伙伴信任程度的客观指标，从而推动人类-机器人互动中的信任预测。我们的研究发现，多种传感器模态（血容积脉搏、皮肤电活动、皮肤温度和注视）的结合能够提高检测人类对机器人合作伙伴信任程度的准确性。此外，随机森林、决策树和极端随机树分类器在测量个人对机器人合作伙伴的信任程度方面表现更优。这些结果为构建人类-机器人互动中的实时信任模型奠定了基础，有助于促进人类与机器人之间的更高效互动。 

---
# RobustDexGrasp: Robust Dexterous Grasping of General Objects from Single-view Perception 

**Title (ZH)**: RobustDexGrasp: 单视图感知下通用物体稳健灵巧抓取 

**Authors**: Hui Zhang, Zijian Wu, Linyi Huang, Sammy Christen, Jie Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.05287)  

**Abstract**: Robust grasping of various objects from single-view perception is fundamental for dexterous robots. Previous works often rely on fully observable objects, expert demonstrations, or static grasping poses, which restrict their generalization ability and adaptability to external disturbances. In this paper, we present a reinforcement-learning-based framework that enables zero-shot dynamic dexterous grasping of a wide range of unseen objects from single-view perception, while performing adaptive motions to external disturbances. We utilize a hand-centric object representation for shape feature extraction that emphasizes interaction-relevant local shapes, enhancing robustness to shape variance and uncertainty. To enable effective hand adaptation to disturbances with limited observations, we propose a mixed curriculum learning strategy, which first utilizes imitation learning to distill a policy trained with privileged real-time visual-tactile feedback, and gradually transfers to reinforcement learning to learn adaptive motions under disturbances caused by observation noises and dynamic randomization. Our experiments demonstrate strong generalization in grasping unseen objects with random poses, achieving success rates of 97.0% across 247,786 simulated objects and 94.6% across 512 real objects. We also demonstrate the robustness of our method to various disturbances, including unobserved object movement and external forces, through both quantitative and qualitative evaluations. Project Page: this https URL 

**Abstract (ZH)**: 单视图感知下鲁棒抓取各种未见物体的动态灵巧抓取：基于强化学习的零样本框架 

---
# Vision-Language Model Predictive Control for Manipulation Planning and Trajectory Generation 

**Title (ZH)**: 视觉-语言模型预测控制用于操作计划与轨迹生成 

**Authors**: Jiaming Chen, Wentao Zhao, Ziyu Meng, Donghui Mao, Ran Song, Wei Pan, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05225)  

**Abstract**: Model Predictive Control (MPC) is a widely adopted control paradigm that leverages predictive models to estimate future system states and optimize control inputs accordingly. However, while MPC excels in planning and control, it lacks the capability for environmental perception, leading to failures in complex and unstructured scenarios. To address this limitation, we introduce Vision-Language Model Predictive Control (VLMPC), a robotic manipulation planning framework that integrates the perception power of vision-language models (VLMs) with MPC. VLMPC utilizes a conditional action sampling module that takes a goal image or language instruction as input and leverages VLM to generate candidate action sequences. These candidates are fed into a video prediction model that simulates future frames based on the actions. In addition, we propose an enhanced variant, Traj-VLMPC, which replaces video prediction with motion trajectory generation to reduce computational complexity while maintaining accuracy. Traj-VLMPC estimates motion dynamics conditioned on the candidate actions, offering a more efficient alternative for long-horizon tasks and real-time applications. Both VLMPC and Traj-VLMPC select the optimal action sequence using a VLM-based hierarchical cost function that captures both pixel-level and knowledge-level consistency between the current observation and the task input. We demonstrate that both approaches outperform existing state-of-the-art methods on public benchmarks and achieve excellent performance in various real-world robotic manipulation tasks. Code is available at this https URL. 

**Abstract (ZH)**: 视觉语言模型预测控制 (Vision-Language Model Predictive Control)：一种结合视觉语言模型感知能力与模型预测控制的机器人操作规划框架 

---
# TDFANet: Encoding Sequential 4D Radar Point Clouds Using Trajectory-Guided Deformable Feature Aggregation for Place Recognition 

**Title (ZH)**: TDFANet：使用轨迹引导可变形特征聚合编码序贯4D雷达点云进行地点识别 

**Authors**: Shouyi Lu, Guirong Zhuo, Haitao Wang, Quan Zhou, Huanyu Zhou, Renbo Huang, Minqing Huang, Lianqing Zheng, Qiang Shu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05103)  

**Abstract**: Place recognition is essential for achieving closed-loop or global positioning in autonomous vehicles and mobile robots. Despite recent advancements in place recognition using 2D cameras or 3D LiDAR, it remains to be seen how to use 4D radar for place recognition - an increasingly popular sensor for its robustness against adverse weather and lighting conditions. Compared to LiDAR point clouds, radar data are drastically sparser, noisier and in much lower resolution, which hampers their ability to effectively represent scenes, posing significant challenges for 4D radar-based place recognition. This work addresses these challenges by leveraging multi-modal information from sequential 4D radar scans and effectively extracting and aggregating spatio-temporal this http URL approach follows a principled pipeline that comprises (1) dynamic points removal and ego-velocity estimation from velocity property, (2) bird's eye view (BEV) feature encoding on the refined point cloud, (3) feature alignment using BEV feature map motion trajectory calculated by ego-velocity, (4) multi-scale spatio-temporal features of the aligned BEV feature maps are extracted and this http URL-world experimental results validate the feasibility of the proposed method and demonstrate its robustness in handling dynamic environments. Source codes are available. 

**Abstract (ZH)**: 基于4D雷达的场所识别：利用序列多模态信息的有效时空特征提取与聚合 

---
# A High-Force Gripper with Embedded Multimodal Sensing for Powerful and Perception Driven Grasping 

**Title (ZH)**: 具有内置多模态感知的高力夹爪及其感知驱动抓取 

**Authors**: Edoardo Del Bianco, Davide Torielli, Federico Rollo, Damiano Gasperini, Arturo Laurenzi, Lorenzo Baccelliere, Luca Muratore, Marco Roveri, Nikos G. Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.04970)  

**Abstract**: Modern humanoid robots have shown their promising potential for executing various tasks involving the grasping and manipulation of objects using their end-effectors. Nevertheless, in the most of the cases, the grasping and manipulation actions involve low to moderate payload and interaction forces. This is due to limitations often presented by the end-effectors, which can not match their arm-reachable payload, and hence limit the payload that can be grasped and manipulated. In addition, grippers usually do not embed adequate perception in their hardware, and grasping actions are mainly driven by perception sensors installed in the rest of the robot body, frequently affected by occlusions due to the arm motions during the execution of the grasping and manipulation tasks. To address the above, we developed a modular high grasping force gripper equipped with embedded multi-modal perception functionalities. The proposed gripper can generate a grasping force of 110 N in a compact implementation. The high grasping force capability is combined with embedded multi-modal sensing, which includes an eye-in-hand camera, a Time-of-Flight (ToF) distance sensor, an Inertial Measurement Unit (IMU) and an omnidirectional microphone, permitting the implementation of perception-driven grasping functionalities.
We extensively evaluated the grasping force capacity of the gripper by introducing novel payload evaluation metrics that are a function of the robot arm's dynamic motion and gripper thermal states. We also evaluated the embedded multi-modal sensing by performing perception-guided enhanced grasping operations. 

**Abstract (ZH)**: 现代人形机器人展示了其在使用末端执行器执行各种涉及抓取和操作物体任务方面的前景。然而，在大多数情况下，抓取和操作动作涉及的载荷和相互作用力较低至中等。这主要是因为末端执行器的限制，它们无法匹配手臂可达的载荷，从而限制了可抓取和操作的载荷。此外，夹持器通常在其硬件中未嵌入足够的感知功能，抓取动作主要由安装在机器人身体其余部分的感知传感器驱动，这些传感器在执行抓取和操作任务时经常受到手臂运动引起的遮挡的影响。为了解决上述问题，我们开发了一种具有嵌入式多模态感知功能的模块化高抓取力夹持器，该夹持器可以在紧凑的实施中产生110 N的抓取力。高抓取力与嵌入式多模态传感功能相结合，包括手眼相机、飞行时间（ToF）距离传感器、惯性测量单元（IMU）和全向麦克风，允许实现感知驱动的抓取功能。我们通过引入新的载荷评估指标来广泛评估夹持器的抓取力能力，这些指标是机器人手臂动态运动和夹持器热状态的函数。我们还通过执行感知引导的增强抓取操作来评估嵌入的多模态传感功能。 

---
# Constrained Gaussian Process Motion Planning via Stein Variational Newton Inference 

**Title (ZH)**: 基于Stein变分牛顿推断的约束高斯过程运动规划 

**Authors**: Jiayun Li, Kay Pompetzki, An Thai Le, Haolei Tong, Jan Peters, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.04936)  

**Abstract**: Gaussian Process Motion Planning (GPMP) is a widely used framework for generating smooth trajectories within a limited compute time--an essential requirement in many robotic applications. However, traditional GPMP approaches often struggle with enforcing hard nonlinear constraints and rely on Maximum a Posteriori (MAP) solutions that disregard the full Bayesian posterior. This limits planning diversity and ultimately hampers decision-making. Recent efforts to integrate Stein Variational Gradient Descent (SVGD) into motion planning have shown promise in handling complex constraints. Nonetheless, these methods still face persistent challenges, such as difficulties in strictly enforcing constraints and inefficiencies when the probabilistic inference problem is poorly conditioned. To address these issues, we propose a novel constrained Stein Variational Gaussian Process Motion Planning (cSGPMP) framework, incorporating a GPMP prior specifically designed for trajectory optimization under hard constraints. Our approach improves the efficiency of particle-based inference while explicitly handling nonlinear constraints. This advancement significantly broadens the applicability of GPMP to motion planning scenarios demanding robust Bayesian inference, strict constraint adherence, and computational efficiency within a limited time. We validate our method on standard benchmarks, achieving an average success rate of 98.57% across 350 planning tasks, significantly outperforming competitive baselines. This demonstrates the ability of our method to discover and use diverse trajectory modes, enhancing flexibility and adaptability in complex environments, and delivering significant improvements over standard baselines without incurring major computational costs. 

**Abstract (ZH)**: 基于Stein变分梯度下降的约束高斯过程运动规划（cSGPMP） 

---
# Embracing Dynamics: Dynamics-aware 4D Gaussian Splatting SLAM 

**Title (ZH)**: 拥抱动态性：意识动态性的4D高斯点云SLAM 

**Authors**: Zhicong Sun, Jacqueline Lo, Jinxing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04844)  

**Abstract**: Simultaneous localization and mapping (SLAM) technology now has photorealistic mapping capabilities thanks to the real-time high-fidelity rendering capability of 3D Gaussian splatting (3DGS). However, due to the static representation of scenes, current 3DGS-based SLAM encounters issues with pose drift and failure to reconstruct accurate maps in dynamic environments. To address this problem, we present D4DGS-SLAM, the first SLAM method based on 4DGS map representation for dynamic environments. By incorporating the temporal dimension into scene representation, D4DGS-SLAM enables high-quality reconstruction of dynamic scenes. Utilizing the dynamics-aware InfoModule, we can obtain the dynamics, visibility, and reliability of scene points, and filter stable static points for tracking accordingly. When optimizing Gaussian points, we apply different isotropic regularization terms to Gaussians with varying dynamic characteristics. Experimental results on real-world dynamic scene datasets demonstrate that our method outperforms state-of-the-art approaches in both camera pose tracking and map quality. 

**Abstract (ZH)**: 基于4D高斯点分布的动态环境SLAM（D4DGS-SLAM） 

---
# Embodied Perception for Test-time Grasping Detection Adaptation with Knowledge Infusion 

**Title (ZH)**: 具身感知在知识注入下的测试时抓取检测适应 

**Authors**: Jin Liu, Jialong Xie, Leibing Xiao, Chaoqun Wang, Fengyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.04795)  

**Abstract**: It has always been expected that a robot can be easily deployed to unknown scenarios, accomplishing robotic grasping tasks without human intervention. Nevertheless, existing grasp detection approaches are typically off-body techniques and are realized by training various deep neural networks with extensive annotated data support. {In this paper, we propose an embodied test-time adaptation framework for grasp detection that exploits the robot's exploratory capabilities.} The framework aims to improve the generalization performance of grasping skills for robots in an unforeseen environment. Specifically, we introduce embodied assessment criteria based on the robot's manipulation capability to evaluate the quality of the grasp detection and maintain suitable samples. This process empowers the robots to actively explore the environment and continuously learn grasping skills, eliminating human intervention. Besides, to improve the efficiency of robot exploration, we construct a flexible knowledge base to provide context of initial optimal viewpoints. Conditioned on the maintained samples, the grasp detection networks can be adapted in the test-time scene. When the robot confronts new objects, it will undergo the same adaptation procedure mentioned above to realize continuous learning. Extensive experiments conducted on a real-world robot demonstrate the effectiveness and generalization of our proposed framework. 

**Abstract (ZH)**: 一种利用机器人探索能力进行抓取检测的封装式测试时自适应框架 

---
# Tool-as-Interface: Learning Robot Policies from Human Tool Usage through Imitation Learning 

**Title (ZH)**: 工具即界面：通过imitation learning学习机器人从人类工具使用中获取策略 

**Authors**: Haonan Chen, Cheng Zhu, Yunzhu Li, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2504.04612)  

**Abstract**: Tool use is critical for enabling robots to perform complex real-world tasks, and leveraging human tool-use data can be instrumental for teaching robots. However, existing data collection methods like teleoperation are slow, prone to control delays, and unsuitable for dynamic tasks. In contrast, human natural data, where humans directly perform tasks with tools, offers natural, unstructured interactions that are both efficient and easy to collect. Building on the insight that humans and robots can share the same tools, we propose a framework to transfer tool-use knowledge from human data to robots. Using two RGB cameras, our method generates 3D reconstruction, applies Gaussian splatting for novel view augmentation, employs segmentation models to extract embodiment-agnostic observations, and leverages task-space tool-action representations to train visuomotor policies. We validate our approach on diverse real-world tasks, including meatball scooping, pan flipping, wine bottle balancing, and other complex tasks. Our method achieves a 71\% higher average success rate compared to diffusion policies trained with teleoperation data and reduces data collection time by 77\%, with some tasks solvable only by our framework. Compared to hand-held gripper, our method cuts data collection time by 41\%. Additionally, our method bridges the embodiment gap, improves robustness to variations in camera viewpoints and robot configurations, and generalizes effectively across objects and spatial setups. 

**Abstract (ZH)**: 基于人类工具使用数据的机器人工具使用知识转移框架 

---
# DexTOG: Learning Task-Oriented Dexterous Grasp with Language 

**Title (ZH)**: DexTOG: 学习任务导向的灵巧抓取与语言 

**Authors**: Jieyi Zhang, Wenqiang Xu, Zhenjun Yu, Pengfei Xie, Tutian Tang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04573)  

**Abstract**: This study introduces a novel language-guided diffusion-based learning framework, DexTOG, aimed at advancing the field of task-oriented grasping (TOG) with dexterous hands. Unlike existing methods that mainly focus on 2-finger grippers, this research addresses the complexities of dexterous manipulation, where the system must identify non-unique optimal grasp poses under specific task constraints, cater to multiple valid grasps, and search in a high degree-of-freedom configuration space in grasp planning. The proposed DexTOG includes a diffusion-based grasp pose generation model, DexDiffu, and a data engine to support the DexDiffu. By leveraging DexTOG, we also proposed a new dataset, DexTOG-80K, which was developed using a shadow robot hand to perform various tasks on 80 objects from 5 categories, showcasing the dexterity and multi-tasking capabilities of the robotic hand. This research not only presents a significant leap in dexterous TOG but also provides a comprehensive dataset and simulation validation, setting a new benchmark in robotic manipulation research. 

**Abstract (ZH)**: 基于语言引导扩散学习框架 DexTOG 以推进灵巧手任务导向抓取的研究 

---
# A General Peg-in-Hole Assembly Policy Based on Domain Randomized Reinforcement Learning 

**Title (ZH)**: 基于领域随机化强化学习的通用 peg-in-hole 装配策略 

**Authors**: Xinyu Liu, Aljaz Kramberger, Leon Bodenhagen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04148)  

**Abstract**: Generalization is important for peg-in-hole assembly, a fundamental industrial operation, to adapt to dynamic industrial scenarios and enhance manufacturing efficiency. While prior work has enhanced generalization ability for pose variations, spatial generalization to six degrees of freedom (6-DOF) is less researched, limiting application in real-world scenarios. This paper addresses this limitation by developing a general policy GenPiH using Proximal Policy Optimization(PPO) and dynamic simulation with domain randomization. The policy learning experiment demonstrates the policy's generalization ability with nearly 100\% success insertion across over eight thousand unique hole poses in parallel environments, and sim-to-real validation on a UR10e robot confirms the policy's performance through direct trajectory execution without task-specific tuning. 

**Abstract (ZH)**: 通用性对于 peg-in-hole 装配这一基本工业操作在适应动态工业场景和提升制造效率方面至关重要。尽管先前的工作已经在姿态变化方面增强了通用性，但空间六自由度（6-DOF）的通用性研究较少，限制了其在实际应用场景中的应用。本文通过使用强化学习中的近端策略优化（PPO）和动态仿真结合领域随机化开发了一种通用策略 GenPiH，以解决这一限制。策略学习实验表明，该策略在并行环境中对超过八千种独特孔的姿态具有近 100% 的成功率插入能力，并且通过直接轨迹执行在 UR10e 机器人上的实到虚验证确认了该策略的性能，无需特定任务的调整。 

---
# Mapping at First Sense: A Lightweight Neural Network-Based Indoor Structures Prediction Method for Robot Autonomous Exploration 

**Title (ZH)**: 基于初次感知的 Lightweight 神经网络室内结构预测方法及其在机器人自主探索中的应用 

**Authors**: Haojia Gao, Haohua Que, Kunrong Li, Weihao Shan, Mingkai Liu, Rong Zhao, Lei Mu, Xinghua Yang, Qi Wei, Fei Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04061)  

**Abstract**: Autonomous exploration in unknown environments is a critical challenge in robotics, particularly for applications such as indoor navigation, search and rescue, and service robotics. Traditional exploration strategies, such as frontier-based methods, often struggle to efficiently utilize prior knowledge of structural regularities in indoor spaces. To address this limitation, we propose Mapping at First Sense, a lightweight neural network-based approach that predicts unobserved areas in local maps, thereby enhancing exploration efficiency. The core of our method, SenseMapNet, integrates convolutional and transformerbased architectures to infer occluded regions while maintaining computational efficiency for real-time deployment on resourceconstrained robots. Additionally, we introduce SenseMapDataset, a curated dataset constructed from KTH and HouseExpo environments, which facilitates training and evaluation of neural models for indoor exploration. Experimental results demonstrate that SenseMapNet achieves an SSIM (structural similarity) of 0.78, LPIPS (perceptual quality) of 0.68, and an FID (feature distribution alignment) of 239.79, outperforming conventional methods in map reconstruction quality. Compared to traditional frontier-based exploration, our method reduces exploration time by 46.5% (from 2335.56s to 1248.68s) while maintaining a high coverage rate (88%) and achieving a reconstruction accuracy of 88%. The proposed method represents a promising step toward efficient, learning-driven robotic exploration in structured environments. 

**Abstract (ZH)**: 基于初次感知的未知环境自主探索 

---
# I Can Hear You Coming: RF Sensing for Uncooperative Satellite Evasion 

**Title (ZH)**: 我能听到你 coming：用于不合作卫星规避的射频 sensing 

**Authors**: Cameron Mehlman, Gregory Falco  

**Link**: [PDF](https://arxiv.org/pdf/2504.03983)  

**Abstract**: Uncooperative satellite engagements with nation-state actors prompts the need for enhanced maneuverability and agility on-orbit. However, robust, autonomous and rapid adversary avoidance capabilities for the space environment is seldom studied. Further, the capability constrained nature of many space vehicles does not afford robust space situational awareness capabilities that can inform maneuvers. We present a "Cat & Mouse" system for training optimal adversary avoidance algorithms using Reinforcement Learning (RL). We propose the novel approach of utilizing intercepted radio frequency communication and dynamic spacecraft state as multi-modal input that could inform paths for a mouse to outmaneuver the cat satellite. Given the current ubiquitous use of RF communications, our proposed system can be applicable to a diverse array of satellites. In addition to providing a comprehensive framework for an RL architecture capable of training performant and adaptive adversary avoidance policies, we also explore several optimization based methods for adversarial avoidance on real-world data obtained from the Space Surveillance Network (SSN) to analyze the benefits and limitations of different avoidance methods. 

**Abstract (ZH)**: 非合作卫星与国家行为体的互动促使轨道机动性和敏捷性的增强。然而，对于空间环境中的 robust、自主且快速的对手规避能力的研究较少。此外，许多航天器的能力限制并不允许提供能够指导机动的空间态势感知能力。我们提出了一种基于增强学习（RL）的“猫与老鼠”系统，用于训练最优对手规避算法。我们提议利用截获的射频通信和动态航天器状态作为多模态输入，以指导老鼠如何规避猫卫星的路径。鉴于当前射频通信的广泛应用，我们的系统适用于各种类型的卫星。除了提供一个全面的用于训练高性能和适应性对手规避策略的RL架构框架外，我们还探讨了几种基于优化方法的实际数据（来自空间监视网络SSN）下的对手规避技术，以分析不同规避方法的利益和局限性。 

---
# Continuous Locomotive Crowd Behavior Generation 

**Title (ZH)**: 连续 locomotive 群体行为生成 

**Authors**: Inhwan Bae, Junoh Lee, Hae-Gon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2504.04756)  

**Abstract**: Modeling and reproducing crowd behaviors are important in various domains including psychology, robotics, transport engineering and virtual environments. Conventional methods have focused on synthesizing momentary scenes, which have difficulty in replicating the continuous nature of real-world crowds. In this paper, we introduce a novel method for automatically generating continuous, realistic crowd trajectories with heterogeneous behaviors and interactions among individuals. We first design a crowd emitter model. To do this, we obtain spatial layouts from single input images, including a segmentation map, appearance map, population density map and population probability, prior to crowd generation. The emitter then continually places individuals on the timeline by assigning independent behavior characteristics such as agents' type, pace, and start/end positions using diffusion models. Next, our crowd simulator produces their long-term locomotions. To simulate diverse actions, it can augment their behaviors based on a Markov chain. As a result, our overall framework populates the scenes with heterogeneous crowd behaviors by alternating between the proposed emitter and simulator. Note that all the components in the proposed framework are user-controllable. Lastly, we propose a benchmark protocol to evaluate the realism and quality of the generated crowds in terms of the scene-level population dynamics and the individual-level trajectory accuracy. We demonstrate that our approach effectively models diverse crowd behavior patterns and generalizes well across different geographical environments. Code is publicly available at this https URL . 

**Abstract (ZH)**: Modeling和再现人群行为在心理学、机器人学、交通工程和虚拟环境中都非常重要。传统方法主要集中在合成短暂的场景，难以再现现实世界人群的连续性。本文提出了一种新的方法，用于自动生成具有异质行为和个体间交互的连续、逼真人流轨迹。首先，我们设计了一种人群发射器模型。在此过程中，我们从单张输入图像中获得空间布局，包括分割图、外观图、人口密度图和人口概率，以备人群生成之用。发射器随后通过分配独立的行为特征（如代理类型、步伐、起始/结束位置）来不断在时间线上放置个体，使用扩散模型。接着，我们的 crowd 模拟器产生长期的移动。为了模拟多样化的动作，它可以基于马尔可夫链增强其行为。因此，我们整体框架通过交替使用提出的发射器和模拟器，在场景中填充了异质人群行为。请注意，框架中的所有组件均可由用户控制。最后，我们提出了一项基准协议，以评估生成人群在场景级人口动态和个体级轨迹准确性方面的真实性和质量。我们证明了我们的方法能够有效建模多样的人群行为模式，并且在不同地理环境中具有良好的泛化能力。代码可在以下链接公开获取：this https URL。 

---
# Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions 

**Title (ZH)**: 基于语言指令、视觉观察和交互的3D物体功能 grounding 

**Authors**: He Zhu, Quyu Kong, Kechun Xu, Xunlong Xia, Bing Deng, Jieping Ye, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04744)  

**Abstract**: Grounding 3D object affordance is a task that locates objects in 3D space where they can be manipulated, which links perception and action for embodied intelligence. For example, for an intelligent robot, it is necessary to accurately ground the affordance of an object and grasp it according to human instructions. In this paper, we introduce a novel task that grounds 3D object affordance based on language instructions, visual observations and interactions, which is inspired by cognitive science. We collect an Affordance Grounding dataset with Points, Images and Language instructions (AGPIL) to support the proposed task. In the 3D physical world, due to observation orientation, object rotation, or spatial occlusion, we can only get a partial observation of the object. So this dataset includes affordance estimations of objects from full-view, partial-view, and rotation-view perspectives. To accomplish this task, we propose LMAffordance3D, the first multi-modal, language-guided 3D affordance grounding network, which applies a vision-language model to fuse 2D and 3D spatial features with semantic features. Comprehensive experiments on AGPIL demonstrate the effectiveness and superiority of our method on this task, even in unseen experimental settings. Our project is available at this https URL. 

**Abstract (ZH)**: 基于语言指令、视觉观测和交互的地基三维物体功能性任务及AGPIL数据集 

---
# Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG and Symbolic Verification 

**Title (ZH)**: 基于知识图谱-RAG和符号验证的复杂任务分层规划 

**Authors**: Cristina Cornelio, Flavio Petruzzellis, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2504.04578)  

**Abstract**: Large Language Models (LLMs) have shown promise as robotic planners but often struggle with long-horizon and complex tasks, especially in specialized environments requiring external knowledge. While hierarchical planning and Retrieval-Augmented Generation (RAG) address some of these challenges, they remain insufficient on their own and a deeper integration is required for achieving more reliable systems. To this end, we propose a neuro-symbolic approach that enhances LLMs-based planners with Knowledge Graph-based RAG for hierarchical plan generation. This method decomposes complex tasks into manageable subtasks, further expanded into executable atomic action sequences. To ensure formal correctness and proper decomposition, we integrate a Symbolic Validator, which also functions as a failure detector by aligning expected and observed world states. Our evaluation against baseline methods demonstrates the consistent significant advantages of integrating hierarchical planning, symbolic verification, and RAG across tasks of varying complexity and different LLMs. Additionally, our experimental setup and novel metrics not only validate our approach for complex planning but also serve as a tool for assessing LLMs' reasoning and compositional capabilities. 

**Abstract (ZH)**: 基于知识图谱的检索增强生成的神经符号规划方法：结合层次规划、符号验证和大型语言模型 

---
# GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill 

**Title (ZH)**: GROVE: 一种通用的开放式词汇物理技能学习奖励机制 

**Authors**: Jieming Cui, Tengyu Liu, Ziyu Meng, Jiale Yu, Ran Song, Wei Zhang, Yixin Zhu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04191)  

**Abstract**: Learning open-vocabulary physical skills for simulated agents presents a significant challenge in artificial intelligence. Current reinforcement learning approaches face critical limitations: manually designed rewards lack scalability across diverse tasks, while demonstration-based methods struggle to generalize beyond their training distribution. We introduce GROVE, a generalized reward framework that enables open-vocabulary physical skill learning without manual engineering or task-specific demonstrations. Our key insight is that Large Language Models(LLMs) and Vision Language Models(VLMs) provide complementary guidance -- LLMs generate precise physical constraints capturing task requirements, while VLMs evaluate motion semantics and naturalness. Through an iterative design process, VLM-based feedback continuously refines LLM-generated constraints, creating a self-improving reward system. To bridge the domain gap between simulation and natural images, we develop Pose2CLIP, a lightweight mapper that efficiently projects agent poses directly into semantic feature space without computationally expensive rendering. Extensive experiments across diverse embodiments and learning paradigms demonstrate GROVE's effectiveness, achieving 22.2% higher motion naturalness and 25.7% better task completion scores while training 8.4x faster than previous methods. These results establish a new foundation for scalable physical skill acquisition in simulated environments. 

**Abstract (ZH)**: 基于通用词表的物理技能学习对模拟代理而言在人工智能中构成重大挑战。当前的强化学习方法面临关键限制：手工设计的奖励在不同任务间缺乏可扩展性，而基于示范的方法难以在训练分布之外泛化。我们引入了GROVE，一种通用奖励框架，该框架能够在无需手工工程或特定任务示范的情况下学习基于通用词表的物理技能。我们的关键洞察是，大型语言模型（LLMs）和视觉语言模型（VLMs）提供了互补的指导——LLMs生成精确的物理约束来捕捉任务需求，而VLMs评估运动语义和自然性。通过迭代设计过程，基于VLM的反馈不断细化LLM生成的约束，形成自我改进的奖励系统。为缩小模拟环境与自然图像之间的域差距，我们开发了Pose2CLIP，一种轻量级的映射器，能够高效地将代理姿态直接映射至语义特征空间，而不进行计算成本高的渲染。通过对多样化的实体和学习范式的广泛实验，GROVE的有效性得以验证，其在运动自然性和任务完成分数上分别提高了22.2%和25.7%，同时训练速度比之前的方法快8.4倍。这些结果为模拟环境中的可扩展物理技能获取奠定了新的基础。 

---
# VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks 

**Title (ZH)**: VAPO: 高效且可靠的高级推理任务强化学习方法 

**Authors**: YuYue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan Wang, Yonghui Wu, Lin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05118)  

**Abstract**: We present VAPO, Value-based Augmented Proximal Policy Optimization framework for reasoning models., a novel framework tailored for reasoning models within the value-based paradigm. Benchmarked the AIME 2024 dataset, VAPO, built on the Qwen 32B pre-trained model, attains a state-of-the-art score of $\mathbf{60.4}$. In direct comparison under identical experimental settings, VAPO outperforms the previously reported results of DeepSeek-R1-Zero-Qwen-32B and DAPO by more than 10 points. The training process of VAPO stands out for its stability and efficiency. It reaches state-of-the-art performance within a mere 5,000 steps. Moreover, across multiple independent runs, no training crashes occur, underscoring its reliability. This research delves into long chain-of-thought (long-CoT) reasoning using a value-based reinforcement learning framework. We pinpoint three key challenges that plague value-based methods: value model bias, the presence of heterogeneous sequence lengths, and the sparsity of reward signals. Through systematic design, VAPO offers an integrated solution that effectively alleviates these challenges, enabling enhanced performance in long-CoT reasoning tasks. 

**Abstract (ZH)**: 基于价值的增强近端策略优化框架VAPO：一种面向推理模型的新框架 

---
# AI in a vat: Fundamental limits of efficient world modelling for agent sandboxing and interpretability 

**Title (ZH)**: AI在容器中：智能体沙箱化和可解释性下的高效世界建模基本极限 

**Authors**: Fernando Rosas, Alexander Boyd, Manuel Baltieri  

**Link**: [PDF](https://arxiv.org/pdf/2504.04608)  

**Abstract**: Recent work proposes using world models to generate controlled virtual environments in which AI agents can be tested before deployment to ensure their reliability and safety. However, accurate world models often have high computational demands that can severely restrict the scope and depth of such assessments. Inspired by the classic `brain in a vat' thought experiment, here we investigate ways of simplifying world models that remain agnostic to the AI agent under evaluation. By following principles from computational mechanics, our approach reveals a fundamental trade-off in world model construction between efficiency and interpretability, demonstrating that no single world model can optimise all desirable characteristics. Building on this trade-off, we identify procedures to build world models that either minimise memory requirements, delineate the boundaries of what is learnable, or allow tracking causes of undesirable outcomes. In doing so, this work establishes fundamental limits in world modelling, leading to actionable guidelines that inform core design choices related to effective agent evaluation. 

**Abstract (ZH)**: 近期的研究提出使用世界模型生成可控的虚拟环境，以在部署AI代理之前对其进行测试，确保其可靠性和安全性。然而，准确的世界模型往往具有较高的计算需求，这会严重限制这类评估的范围和深度。受经典的“脑在 vat 中”思想实验的启发，我们探讨了一种简化世界模型的方法，这种方法对正在评估的AI代理是无偏见的。通过遵循计算力学的原则，我们的方法揭示了世界模型构建中效率与可解释性之间的基本权衡，表明没有一个世界模型能够同时优化所有理想特性。基于这种权衡，我们确定了构建世界模型的程序，要么尽量减少内存需求，要么明确可学习的边界，要么允许追踪不良结果的原因。通过这种方式，本研究确立了世界建模的基本限制，并为有效的代理评估的核心设计选择提供了可操作的指导方针。 

---
# Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks 

**Title (ZH)**: 使用地标引导的分层强化学习求解Sokoban 

**Authors**: Sergey Pastukhov  

**Link**: [PDF](https://arxiv.org/pdf/2504.04366)  

**Abstract**: We introduce a novel hierarchical reinforcement learning (HRL) framework that performs top-down recursive planning via learned subgoals, successfully applied to the complex combinatorial puzzle game Sokoban. Our approach constructs a six-level policy hierarchy, where each higher-level policy generates subgoals for the level below. All subgoals and policies are learned end-to-end from scratch, without any domain knowledge. Our results show that the agent can generate long action sequences from a single high-level call. While prior work has explored 2-3 level hierarchies and subgoal-based planning heuristics, we demonstrate that deep recursive goal decomposition can emerge purely from learning, and that such hierarchies can scale effectively to hard puzzle domains. 

**Abstract (ZH)**: 我们介绍了一种新颖的分层强化学习（HRL）框架，通过学习子目标进行自上而下的递归规划，并成功应用于复杂的组合谜题游戏 sokoban。我们的方法构建了一个六级策略层次结构，其中每一级较高的策略为较低级别生成子目标。所有子目标和策略都是端到端地从头开始学习，无需任何领域知识。我们的结果表明，该代理可以从单个高层调用生成长动作序列。尽管先前的工作探索了2-3级层次结构和基于子目标的规划启发式方法，但我们展示了深度递归目标分解可以纯粹通过学习涌现，而且这样的层次结构可以有效地扩展到困难的谜题领域。 

---
# Introducing COGENT3: An AI Architecture for Emergent Cognition 

**Title (ZH)**: 介绍COGENT3：一种涌现认知的人工智能架构 

**Authors**: Eduardo Salazar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04139)  

**Abstract**: This paper presents COGENT3 (or Collective Growth and Entropy-modulated Triads System), a novel approach for emergent cognition integrating pattern formation networks with group influence dynamics. Contrasting with traditional strategies that rely on predetermined architectures, computational structures emerge dynamically in our framework through agent interactions. This enables a more flexible and adaptive system exhibiting characteristics reminiscent of human cognitive processes. The incorporation of temperature modulation and memory effects in COGENT3 closely integrates statistical mechanics, machine learning, and cognitive science. 

**Abstract (ZH)**: COGENT3（或集体增长和熵调节三元系统）：一种将模式形成网络与群体影响动态相结合的新兴认知方法 

---
# Among Us: A Sandbox for Agentic Deception 

**Title (ZH)**: Among Us: 一种自主欺骗的沙箱环境 

**Authors**: Satvik Golechha, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2504.04072)  

**Abstract**: Studying deception in AI agents is important and difficult due to the lack of model organisms and sandboxes that elicit the behavior without asking the model to act under specific conditions or inserting intentional backdoors. Extending upon $\textit{AmongAgents}$, a text-based social-deduction game environment, we aim to fix this by introducing Among Us as a rich sandbox where LLM-agents exhibit human-style deception naturally while they think, speak, and act with other agents or humans. We introduce Deception ELO as an unbounded measure of deceptive capability, suggesting that frontier models win more because they're better at deception, not at detecting it. We evaluate the effectiveness of AI safety techniques (LLM-monitoring of outputs, linear probes on various datasets, and sparse autoencoders) for detecting lying and deception in Among Us, and find that they generalize very well out-of-distribution. We open-source our sandbox as a benchmark for future alignment research and hope that this is a good testbed to improve safety techniques to detect and remove agentically-motivated deception, and to anticipate deceptive abilities in LLMs. 

**Abstract (ZH)**: 研究AI代理中的欺骗行为具有重要意义但难度较大，由于缺乏合适的模型生物或无需在特定条件下操作模型或植入有意后门即可引发欺骗行为的沙箱环境。在$\textit{AmongAgents}$文本社会推理游戏环境的基础上，我们通过引入《Among Us》作为丰富的沙箱环境，让语言模型代理自然地表现出类似人类的欺骗行为，在与其他代理或人类交互的过程中思考、说话和行动。我们提出了欺骗ELO作为衡量欺骗能力的无界指标，表明前沿模型取胜是因为它们更擅长欺骗，而不是检测欺骗。我们评估了多种AI安全性技术（如语言模型的输出监控、线性探针以及稀疏自编码器）在《Among Us》中检测欺骗的有效性，并发现这些技术在分布外具有很好的泛化能力。我们开源这一沙箱，作为未来对齐研究的基准，并希望这可以成为一个有效的测试平台，以提高检测和移除代理驱动欺骗的技术，并预见语言模型的欺骗能力。 

---
# Improving World Models using Deep Supervision with Linear Probes 

**Title (ZH)**: 使用线性探针的深度监督改进世界模型 

**Authors**: Andrii Zahorodnii  

**Link**: [PDF](https://arxiv.org/pdf/2504.03861)  

**Abstract**: Developing effective world models is crucial for creating artificial agents that can reason about and navigate complex environments. In this paper, we investigate a deep supervision technique for encouraging the development of a world model in a network trained end-to-end to predict the next observation. While deep supervision has been widely applied for task-specific learning, our focus is on improving the world models. Using an experimental environment based on the Flappy Bird game, where the agent receives only LIDAR measurements as observations, we explore the effect of adding a linear probe component to the network's loss function. This additional term encourages the network to encode a subset of the true underlying world features into its hidden state. Our experiments demonstrate that this supervision technique improves both training and test performance, enhances training stability, and results in more easily decodable world features -- even for those world features which were not included in the training. Furthermore, we observe a reduced distribution drift in networks trained with the linear probe, particularly during high-variability phases of the game (flying between successive pipe encounters). Including the world features loss component roughly corresponded to doubling the model size, suggesting that the linear probe technique is particularly beneficial in compute-limited settings or when aiming to achieve the best performance with smaller models. These findings contribute to our understanding of how to develop more robust and sophisticated world models in artificial agents, paving the way for further advancements in this field. 

**Abstract (ZH)**: 开发有效的世界模型对于创建能够推理和导航复杂环境的人工代理至关重要。在本文中，我们研究了一种深层次监督技术，以促使网络在端到端训练中预测下一个观察值的同时发展世界模型。虽然深层监督在任务特定学习中广泛应用，但我们专注于提高世界模型的质量。通过基于Flappy Bird游戏的实验环境，其中代理仅接收LIDAR测量作为观察值，我们探讨了将线性探针组件添加到网络损失函数中的效果。这个额外的项鼓励网络将其隐藏状态编码为真实底层世界特征的一部分。我们的实验表明，这种监督技术不仅提高了训练和测试性能，还增强了训练稳定性，并导致更易于解码的世界特征——即使这些特征未包含在训练中。此外，我们观察到，在使用线性探针训练的网络中，特别是在游戏中的高变异阶段（在连续管道相遇之间飞行），网络之间的分布漂移减少。包括世界特征损失项大致相当于将模型大小翻倍，表明线性探针技术在计算资源受限的环境中尤其有益，或者在追求小型模型最佳性能时尤为有益。这些发现增加了我们对如何在人工代理中发展更稳健和复杂的世界模型的理解，为进一步推动该领域的发展奠定了基础。 

---
# BRIDGES: Bridging Graph Modality and Large Language Models within EDA Tasks 

**Title (ZH)**: BRIDGES: 跨接图模态与大型语言模型在EDA任务中的桥梁 

**Authors**: Wei Li, Yang Zou, Christopher Ellis, Ruben Purdy, Shawn Blanton, José M. F. Moura  

**Link**: [PDF](https://arxiv.org/pdf/2504.05180)  

**Abstract**: While many EDA tasks already involve graph-based data, existing LLMs in EDA primarily either represent graphs as sequential text, or simply ignore graph-structured data that might be beneficial like dataflow graphs of RTL code. Recent studies have found that LLM performance suffers when graphs are represented as sequential text, and using additional graph information significantly boosts performance. To address these challenges, we introduce BRIDGES, a framework designed to incorporate graph modality into LLMs for EDA tasks. BRIDGES integrates an automated data generation workflow, a solution that combines graph modality with LLM, and a comprehensive evaluation suite. First, we establish an LLM-driven workflow to generate RTL and netlist-level data, converting them into dataflow and netlist graphs with function descriptions. This workflow yields a large-scale dataset comprising over 500,000 graph instances and more than 1.5 billion tokens. Second, we propose a lightweight cross-modal projector that encodes graph representations into text-compatible prompts, enabling LLMs to effectively utilize graph data without architectural modifications. Experimental results demonstrate 2x to 10x improvements across multiple tasks compared to text-only baselines, including accuracy in design retrieval, type prediction and perplexity in function description, with negligible computational overhead (<1% model weights increase and <30% additional runtime overhead). Even without additional LLM finetuning, our results outperform text-only by a large margin. We plan to release BRIDGES, including the dataset, models, and training flow. 

**Abstract (ZH)**: BRIDGES：将图模态纳入EDA任务的LLM框架 

---
# A Reinforcement Learning Method for Environments with Stochastic Variables: Post-Decision Proximal Policy Optimization with Dual Critic Networks 

**Title (ZH)**: 具有随机变量环境的强化学习方法：带双critic网络的后决策近端策略优化 

**Authors**: Leonardo Kanashiro Felizardo, Edoardo Fadda, Paolo Brandimarte, Emilio Del-Moral-Hernandez, Mariá Cristina Vasconcelos Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2504.05150)  

**Abstract**: This paper presents Post-Decision Proximal Policy Optimization (PDPPO), a novel variation of the leading deep reinforcement learning method, Proximal Policy Optimization (PPO). The PDPPO state transition process is divided into two steps: a deterministic step resulting in the post-decision state and a stochastic step leading to the next state. Our approach incorporates post-decision states and dual critics to reduce the problem's dimensionality and enhance the accuracy of value function estimation. Lot-sizing is a mixed integer programming problem for which we exemplify such dynamics. The objective of lot-sizing is to optimize production, delivery fulfillment, and inventory levels in uncertain demand and cost parameters. This paper evaluates the performance of PDPPO across various environments and configurations. Notably, PDPPO with a dual critic architecture achieves nearly double the maximum reward of vanilla PPO in specific scenarios, requiring fewer episode iterations and demonstrating faster and more consistent learning across different initializations. On average, PDPPO outperforms PPO in environments with a stochastic component in the state transition. These results support the benefits of using a post-decision state. Integrating this post-decision state in the value function approximation leads to more informed and efficient learning in high-dimensional and stochastic environments. 

**Abstract (ZH)**: 基于决策后的proximal策略优化（PDPPO） 

---
# Towards Visual Text Grounding of Multimodal Large Language Model 

**Title (ZH)**: 面向多模态大语言模型的视觉文本定位 

**Authors**: Ming Li, Ruiyi Zhang, Jian Chen, Jiuxiang Gu, Yufan Zhou, Franck Dernoncourt, Wanrong Zhu, Tianyi Zhou, Tong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04974)  

**Abstract**: Despite the existing evolution of Multimodal Large Language Models (MLLMs), a non-neglectable limitation remains in their struggle with visual text grounding, especially in text-rich images of documents. Document images, such as scanned forms and infographics, highlight critical challenges due to their complex layouts and textual content. However, current benchmarks do not fully address these challenges, as they mostly focus on visual grounding on natural images, rather than text-rich document images. Thus, to bridge this gap, we introduce TRIG, a novel task with a newly designed instruction dataset for benchmarking and improving the Text-Rich Image Grounding capabilities of MLLMs in document question-answering. Specifically, we propose an OCR-LLM-human interaction pipeline to create 800 manually annotated question-answer pairs as a benchmark and a large-scale training set of 90$ synthetic data based on four diverse datasets. A comprehensive evaluation of various MLLMs on our proposed benchmark exposes substantial limitations in their grounding capability on text-rich images. In addition, we propose two simple and effective TRIG methods based on general instruction tuning and plug-and-play efficient embedding, respectively. By finetuning MLLMs on our synthetic dataset, they promisingly improve spatial reasoning and grounding capabilities. 

**Abstract (ZH)**: 尽管现有的多模态大语言模型（MLLMs）已经取得了进展，但在视觉文本定位方面仍存在不容忽视的局限性，特别是在文档中的文本丰富图像中更为明显。文档图像，如扫描表单和信息图形，因其复杂的布局和文本内容而突显出关键挑战。然而，当前的基准测试并未充分解决这些问题，因为它们主要关注自然图像的视觉定位，而非文本丰富的文档图像。因此，为了填补这一空白，我们引入了TRIG，一种新型任务及其新设计的指令数据集，用于评估和提高MLLMs在文档问答中的文本丰富图像定位能力。具体而言，我们提出了一种OCR-LLM-人工交互流水线，创建了800个手工标注的问题-答案对作为基准和基于四个不同数据集的90,000个合成数据大规模训练集。对我们的基准测试的各种MLLMs进行综合评估揭示了它们在文本丰富的图像定位能力上的诸多局限性。此外，我们提出了两种基于通用指令调整和插拔高效嵌入的简单有效TRIG方法。通过在我们合成数据集上微调MLLMs，它们确实在空间推理和定位能力上取得了显著改进。 

---
# The Point, the Vision and the Text: Does Point Cloud Boost Spatial Reasoning of Large Language Models? 

**Title (ZH)**: 点、视角与文本：点云是否提升大型语言模型的空间推理能力？ 

**Authors**: Weichen Zhang, Ruiying Peng, Chen Gao, Jianjie Fang, Xin Zeng, Kaiyuan Li, Ziyou Wang, Jinqiang Cui, Xin Wang, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04540)  

**Abstract**: 3D Large Language Models (LLMs) leveraging spatial information in point clouds for 3D spatial reasoning attract great attention. Despite some promising results, the role of point clouds in 3D spatial reasoning remains under-explored. In this work, we comprehensively evaluate and analyze these models to answer the research question: \textit{Does point cloud truly boost the spatial reasoning capacities of 3D LLMs?} We first evaluate the spatial reasoning capacity of LLMs with different input modalities by replacing the point cloud with the visual and text counterparts. We then propose a novel 3D QA (Question-answering) benchmark, ScanReQA, that comprehensively evaluates models' understanding of binary spatial relationships. Our findings reveal several critical insights: 1) LLMs without point input could even achieve competitive performance even in a zero-shot manner; 2) existing 3D LLMs struggle to comprehend the binary spatial relationships; 3) 3D LLMs exhibit limitations in exploiting the structural coordinates in point clouds for fine-grained spatial reasoning. We think these conclusions can help the next step of 3D LLMs and also offer insights for foundation models in other modalities. We release datasets and reproducible codes in the anonymous project page: this https URL. 

**Abstract (ZH)**: 三维大规模语言模型利用点云中的空间信息进行三维空间推理受到广泛关注。尽管取得了某些有前景的结果，点云在三维空间推理中的作用仍未得到充分探索。在本文中，我们全面评估和分析这些模型以回答研究问题：点云是否真正提升了三维语言模型的空间推理能力？我们首先通过用视觉和文本输入替换点云来评估具有不同输入模态的语言模型的空间推理能力。然后，我们提出了一种新的三维问答基准ScanReQA，全面评估模型对二元空间关系的理解。我们的发现揭示了几个关键洞察：1) 不含点云输入的语言模型即使在零样本方式下也能实现竞争力表现；2) 存在的三维语言模型在理解二元空间关系方面存在困难；3) 三维语言模型在利用点云中的结构坐标进行精细空间推理方面存在局限性。我们认为这些结论有助于下步三维语言模型的发展，并为其他模态的预训练模型提供见解。我们已在匿名项目页面释放了数据集和可复现代码：this https URL。 

---
# AI2STOW: End-to-End Deep Reinforcement Learning to Construct Master Stowage Plans under Demand Uncertainty 

**Title (ZH)**: AI2STOW：在需求不确定性下基于端到端深度强化学习的主理货计划构建方法 

**Authors**: Jaike Van Twiller, Djordje Grbic, Rune Møller Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04469)  

**Abstract**: The worldwide economy and environmental sustainability depend on eff icient and reliable supply chains, in which container shipping plays a crucial role as an environmentally friendly mode of transport. Liner shipping companies seek to improve operational efficiency by solving the stowage planning problem. Due to many complex combinatorial aspects, stowage planning is challenging and often decomposed into two NP-hard subproblems: master and slot planning. This article proposes AI2STOW, an end-to-end deep reinforcement learning model with feasibility projection and an action mask to create master plans under demand uncertainty with global objectives and constraints, including paired block stowage patterms. Our experimental results demonstrate that AI2STOW outperforms baseline methods from reinforcement learning and stochastic programming in objective performance and computational efficiency, based on simulated instances reflecting the scale of realistic vessels and operational planning horizons. 

**Abstract (ZH)**: 世界经济和环境可持续性依赖于高效可靠的供应链，在此基础上，集装箱运输作为一种环保的运输方式发挥了关键作用。班轮公司通过解决积载规划问题来提高运营效率。由于存在许多复杂的组合因素，积载规划具有挑战性，通常被分解为两个NP难子问题：主积载和舱位规划。本文提出了一种名为AI2STOW的端到端深度强化学习模型，该模型通过可行性投影和动作掩码，在需求不确定性下创建具有全局目标和约束的主积载计划，包括成对的模块化积载模式。实验结果表明，与强化学习和随机规划基线方法相比，AI2STOW在目标性能和计算效率方面具有优势，基于模拟实例反映了现实船舶的规模和运营规划时间范围。 

---
# Decision SpikeFormer: Spike-Driven Transformer for Decision Making 

**Title (ZH)**: 决策尖峰 Former: 尖峰驱动的变压器用于决策 Making 

**Authors**: Wei Huang, Qinying Gu, Nanyang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.03800)  

**Abstract**: Offline reinforcement learning (RL) enables policy training solely on pre-collected data, avoiding direct environment interaction - a crucial benefit for energy-constrained embodied AI applications. Although Artificial Neural Networks (ANN)-based methods perform well in offline RL, their high computational and energy demands motivate exploration of more efficient alternatives. Spiking Neural Networks (SNNs) show promise for such tasks, given their low power consumption. In this work, we introduce DSFormer, the first spike-driven transformer model designed to tackle offline RL via sequence modeling. Unlike existing SNN transformers focused on spatial dimensions for vision tasks, we develop Temporal Spiking Self-Attention (TSSA) and Positional Spiking Self-Attention (PSSA) in DSFormer to capture the temporal and positional dependencies essential for sequence modeling in RL. Additionally, we propose Progressive Threshold-dependent Batch Normalization (PTBN), which combines the benefits of LayerNorm and BatchNorm to preserve temporal dependencies while maintaining the spiking nature of SNNs. Comprehensive results in the D4RL benchmark show DSFormer's superiority over both SNN and ANN counterparts, achieving 78.4% energy savings, highlighting DSFormer's advantages not only in energy efficiency but also in competitive performance. Code and models are public at this https URL. 

**Abstract (ZH)**: 基于离线强化学习的DSFormer：一种_spike驱动的变压器模型以序列建模解决离线RL问题 

---
# Emerging Cyber Attack Risks of Medical AI Agents 

**Title (ZH)**: 新兴医疗AI代理的网络攻击风险 

**Authors**: Jianing Qiu, Lin Li, Jiankai Sun, Hao Wei, Zhe Xu, Kyle Lam, Wu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.03759)  

**Abstract**: Large language models (LLMs)-powered AI agents exhibit a high level of autonomy in addressing medical and healthcare challenges. With the ability to access various tools, they can operate within an open-ended action space. However, with the increase in autonomy and ability, unforeseen risks also arise. In this work, we investigated one particular risk, i.e., cyber attack vulnerability of medical AI agents, as agents have access to the Internet through web browsing tools. We revealed that through adversarial prompts embedded on webpages, cyberattackers can: i) inject false information into the agent's response; ii) they can force the agent to manipulate recommendation (e.g., healthcare products and services); iii) the attacker can also steal historical conversations between the user and agent, resulting in the leak of sensitive/private medical information; iv) furthermore, the targeted agent can also cause a computer system hijack by returning a malicious URL in its response. Different backbone LLMs were examined, and we found such cyber attacks can succeed in agents powered by most mainstream LLMs, with the reasoning models such as DeepSeek-R1 being the most vulnerable. 

**Abstract (ZH)**: 基于大型语言模型的AI代理在应对医疗和健康挑战方面表现出高度自主性，但随着自主性和能力的增强，未预见的安全风险也逐渐显现。本文探究了其中一种风险，即通过网页中的对抗性提示，黑客可以对医疗AI代理进行攻击：i) 注入虚假信息；ii) 强迫使代理操控推荐（如医疗产品和服务）；iii) 盗取用户与代理的历史对话记录，导致敏感/私密医疗信息泄露；iv) 进一步，受攻击代理还可能通过返回恶意URL导致计算机系统被劫持。不同基础模型的大型语言模型均存在此类攻击风险，其中深度探索-R1等推理模型最为脆弱。 

---
# Modelling bounded rational decision-making through Wasserstein constraints 

**Title (ZH)**: 通过Wasserstein约束建模有界理性决策 

**Authors**: Benjamin Patrick Evans, Leo Ardon, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2504.03743)  

**Abstract**: Modelling bounded rational decision-making through information constrained processing provides a principled approach for representing departures from rationality within a reinforcement learning framework, while still treating decision-making as an optimization process. However, existing approaches are generally based on Entropy, Kullback-Leibler divergence, or Mutual Information. In this work, we highlight issues with these approaches when dealing with ordinal action spaces. Specifically, entropy assumes uniform prior beliefs, missing the impact of a priori biases on decision-makings. KL-Divergence addresses this, however, has no notion of "nearness" of actions, and additionally, has several well known potentially undesirable properties such as the lack of symmetry, and furthermore, requires the distributions to have the same support (e.g. positive probability for all actions). Mutual information is often difficult to estimate. Here, we propose an alternative approach for modeling bounded rational RL agents utilising Wasserstein distances. This approach overcomes the aforementioned issues. Crucially, this approach accounts for the nearness of ordinal actions, modeling "stickiness" in agent decisions and unlikeliness of rapidly switching to far away actions, while also supporting low probability actions, zero-support prior distributions, and is simple to calculate directly. 

**Abstract (ZH)**: 通过信息受限处理建模有界理性决策提供了一种在强化学习框架内表示理性偏差的原则性方法，同时仍将决策视为优化过程。然而，现有方法通常基于熵、Kullback-Leibler散度或互信息。在本工作中，我们指出了这些方法在处理序数动作空间时存在的问题。具体而言，熵假定均匀先验信念，忽略了先验偏见对决策的影响。KL散度确实解决了这一问题，但它没有“动作接近性”的概念，并且还具有非对称性等众所周知的潜在不良性质，此外，要求分布具有相同的支撑（例如，所有动作的正概率）。互信息通常难以估计。在此，我们提出了一种利用Wasserstein距离建模有界理性RL代理的替代方法。这种方法克服了上述问题。至关重要的是，这种方法考虑了序数动作的接近性，模型了代理决策中的“粘滞性”，以及不太可能迅速切换到远离的动作，同时支持低概率动作、零支撑先验分布，并且易于直接计算。 

---
# Multi-Objective Quality-Diversity in Unstructured and Unbounded Spaces 

**Title (ZH)**: 无结构和无边界空间中的多目标质量多样性 

**Authors**: Hannah Janmohamed, Antoine Cully  

**Link**: [PDF](https://arxiv.org/pdf/2504.03715)  

**Abstract**: Quality-Diversity algorithms are powerful tools for discovering diverse, high-performing solutions. Recently, Multi-Objective Quality-Diversity (MOQD) extends QD to problems with several objectives while preserving solution diversity. MOQD has shown promise in fields such as robotics and materials science, where finding trade-offs between competing objectives like energy efficiency and speed, or material properties is essential. However, existing methods in MOQD rely on tessellating the feature space into a grid structure, which prevents their application in domains where feature spaces are unknown or must be learned, such as complex biological systems or latent exploration tasks. In this work, we introduce Multi-Objective Unstructured Repertoire for Quality-Diversity (MOUR-QD), a MOQD algorithm designed for unstructured and unbounded feature spaces. We evaluate MOUR-QD on five robotic tasks. Importantly, we show that our method excels in tasks where features must be learned, paving the way for applying MOQD to unsupervised domains. We also demonstrate that MOUR-QD is advantageous in domains with unbounded feature spaces, outperforming existing grid-based methods. Finally, we demonstrate that MOUR-QD is competitive with established MOQD methods on existing MOQD tasks and achieves double the MOQD-score in some environments. MOUR-QD opens up new opportunities for MOQD in domains like protein design and image generation. 

**Abstract (ZH)**: Multi-Objective Unstructured Repertoire for Quality-Diversity 

---
