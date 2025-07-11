# Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop 

**Title (ZH)**: Eye, Robot: 学习观察以行动——基于BC-RL知觉-动作环路 

**Authors**: Justin Kerr, Kush Hari, Ethan Weber, Chung Min Kim, Brent Yi, Tyler Bonnen, Ken Goldberg, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.10968)  

**Abstract**: Humans do not passively observe the visual world -- we actively look in order to act. Motivated by this principle, we introduce EyeRobot, a robotic system with gaze behavior that emerges from the need to complete real-world tasks. We develop a mechanical eyeball that can freely rotate to observe its surroundings and train a gaze policy to control it using reinforcement learning. We accomplish this by first collecting teleoperated demonstrations paired with a 360 camera. This data is imported into a simulation environment that supports rendering arbitrary eyeball viewpoints, allowing episode rollouts of eye gaze on top of robot demonstrations. We then introduce a BC-RL loop to train the hand and eye jointly: the hand (BC) agent is trained from rendered eye observations, and the eye (RL) agent is rewarded when the hand produces correct action predictions. In this way, hand-eye coordination emerges as the eye looks towards regions which allow the hand to complete the task. EyeRobot implements a foveal-inspired policy architecture allowing high resolution with a small compute budget, which we find also leads to the emergence of more stable fixation as well as improved ability to track objects and ignore distractors. We evaluate EyeRobot on five panoramic workspace manipulation tasks requiring manipulation in an arc surrounding the robot arm. Our experiments suggest EyeRobot exhibits hand-eye coordination behaviors which effectively facilitate manipulation over large workspaces with a single camera. See project site for videos: this https URL 

**Abstract (ZH)**: 基于目光行为的人工智能机器人EyeRobot：从真实世界任务中 emerges from the need to complete real-world tasks 

---
# GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation 

**Title (ZH)**: GENMANIP: 由大规模语言模型驱动的通用指令遵循操作模拟 

**Authors**: Ning Gao, Yilun Chen, Shuai Yang, Xinyi Chen, Yang Tian, Hao Li, Haifeng Huang, Hanqing Wang, Tai Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10966)  

**Abstract**: Robotic manipulation in real-world settings remains challenging, especially regarding robust generalization. Existing simulation platforms lack sufficient support for exploring how policies adapt to varied instructions and scenarios. Thus, they lag behind the growing interest in instruction-following foundation models like LLMs, whose adaptability is crucial yet remains underexplored in fair comparisons. To bridge this gap, we introduce GenManip, a realistic tabletop simulation platform tailored for policy generalization studies. It features an automatic pipeline via LLM-driven task-oriented scene graph to synthesize large-scale, diverse tasks using 10K annotated 3D object assets. To systematically assess generalization, we present GenManip-Bench, a benchmark of 200 scenarios refined via human-in-the-loop corrections. We evaluate two policy types: (1) modular manipulation systems integrating foundation models for perception, reasoning, and planning, and (2) end-to-end policies trained through scalable data collection. Results show that while data scaling benefits end-to-end methods, modular systems enhanced with foundation models generalize more effectively across diverse scenarios. We anticipate this platform to facilitate critical insights for advancing policy generalization in realistic conditions. Project Page: this https URL. 

**Abstract (ZH)**: 现实场景中的机器人操作依然具有挑战性，特别是在鲁棒泛化方面。现有的模拟平台缺乏足够的支持来探索策略如何适应多变的指令和场景。因此，它们在与越来越受到关注的指令遵循基础模型（如LLMs）的适应性进行公平比较时落后了。为弥合这一差距，我们引入了GenManip，这是一个针对策略泛化研究定制的现实桌面模拟平台。它通过LLM驱动的任务导向场景图提供了一种自动化的流水线，使用10K注释的3D对象资产来合成大规模、多样化的任务。为了系统地评估泛化能力，我们提出了GenManip-Bench，这是一个包含200个场景的基准，这些场景通过人类在环路校正进行精炼。我们评估了两种策略类型：(1) 结合基础模型的模块化操作系统，用于感知、推理和规划，以及(2) 通过可扩展的数据收集训练的端到端策略。结果表明，尽管数据量的增加有利于端到端方法，但结合基础模型的模块化系统在多样化的场景中泛化更有效。我们期望该平台能促进在现实条件下的策略泛化研究。项目页面：这个 <https://>网址。 

---
# Vib2Move: In-Hand Object Reconfiguration via Fingertip Micro-Vibrations 

**Title (ZH)**: Vib2Move：通过指尖微振动实现手内物体重构 

**Authors**: Xili Yi, Nima Fazeli  

**Link**: [PDF](https://arxiv.org/pdf/2506.10923)  

**Abstract**: We introduce Vib2Move, a novel approach for in-hand object reconfiguration that uses fingertip micro-vibrations and gravity to precisely reposition planar objects. Our framework comprises three key innovations. First, we design a vibration-based actuator that dynamically modulates the effective finger-object friction coefficient, effectively emulating changes in gripping force. Second, we derive a sliding motion model for objects clamped in a parallel gripper with two symmetric, variable-friction contact patches. Third, we propose a motion planner that coordinates end-effector finger trajectories and fingertip vibrations to achieve the desired object pose. In real-world trials, Vib2Move consistently yields final positioning errors below 6 mm, demonstrating reliable, high-precision manipulation across a variety of planar objects. For more results and information, please visit this https URL. 

**Abstract (ZH)**: Vib2Move：一种基于指尖微振动和重力的在手物体重构新方法 

---
# Modeling Trust Dynamics in Robot-Assisted Delivery: Impact of Trust Repair Strategies 

**Title (ZH)**: 基于机器人辅助配送的信任动态建模：信任修复策略的影响 

**Authors**: Dong Hae Mangalindan, Karthik Kandikonda, Ericka Rovira, Vaibhav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2506.10884)  

**Abstract**: With increasing efficiency and reliability, autonomous systems are becoming valuable assistants to humans in various tasks. In the context of robot-assisted delivery, we investigate how robot performance and trust repair strategies impact human trust. In this task, while handling a secondary task, humans can choose to either send the robot to deliver autonomously or manually control it. The trust repair strategies examined include short and long explanations, apology and promise, and denial.
Using data from human participants, we model human behavior using an Input-Output Hidden Markov Model (IOHMM) to capture the dynamics of trust and human action probabilities. Our findings indicate that humans are more likely to deploy the robot autonomously when their trust is high. Furthermore, state transition estimates show that long explanations are the most effective at repairing trust following a failure, while denial is most effective at preventing trust loss.
We also demonstrate that the trust estimates generated by our model are isomorphic to self-reported trust values, making them interpretable. This model lays the groundwork for developing optimal policies that facilitate real-time adjustment of human trust in autonomous systems. 

**Abstract (ZH)**: 随着自主系统的效率和可靠性不断提高，它们在各种任务中 becoming valuable assistants to humans。在机器人辅助配送的背景下，我们探讨了机器人性能和信任修复策略对人类信任的影响。在这一任务中，当处理次要任务时，人类可以选择让机器人自主配送或手动控制。我们研究的信任修复策略包括简短和详尽的解释、道歉和承诺以及否认。

通过人类参与者的数据，我们使用输入-输出隐马尔可夫模型（IOHMM）来建模人类行为，以捕捉信任动态和人类行动的概率。研究发现，当人类的信任度较高时，他们更倾向于让机器人自主配送。此外，状态转换估计表明，在故障后，详尽的解释是最有效的信任修复策略，而否认是最有效的防止信任损失的策略。

我们还展示了由该模型生成的信任估计值与自我报告的信任值等价，使其具有可解释性。该模型为开发促进自主系统中实时调整人类信任的最优政策奠定了基础。 

---
# Data-Driven Prediction of Dynamic Interactions Between Robot Appendage and Granular Material 

**Title (ZH)**: 基于数据驱动的机器人肢体与颗粒物质动态交互预测 

**Authors**: Guanjin Wang, Xiangxue Zhao, Shapour Azarm, Balakumar Balachandran  

**Link**: [PDF](https://arxiv.org/pdf/2506.10875)  

**Abstract**: An alternative data-driven modeling approach has been proposed and employed to gain fundamental insights into robot motion interaction with granular terrain at certain length scales. The approach is based on an integration of dimension reduction (Sequentially Truncated Higher-Order Singular Value Decomposition), surrogate modeling (Gaussian Process), and data assimilation techniques (Reduced Order Particle Filter). This approach can be used online and is based on offline data, obtained from the offline collection of high-fidelity simulation data and a set of sparse experimental data. The results have shown that orders of magnitude reduction in computational time can be obtained from the proposed data-driven modeling approach compared with physics-based high-fidelity simulations. With only simulation data as input, the data-driven prediction technique can generate predictions that have comparable accuracy as simulations. With both simulation data and sparse physical experimental measurement as input, the data-driven approach with its embedded data assimilation techniques has the potential in outperforming only high-fidelity simulations for the long-horizon predictions. In addition, it is demonstrated that the data-driven modeling approach can also reproduce the scaling relationship recovered by physics-based simulations for maximum resistive forces, which may indicate its general predictability beyond a case-by-case basis. The results are expected to help robot navigation and exploration in unknown and complex terrains during both online and offline phases. 

**Abstract (ZH)**: 一种数据驱动建模方法被提出并应用于在特定长度尺度上揭示机器人运动与颗粒型地形相互作用的基本原理。该方法基于顺序截断高阶奇异值分解、高斯过程拟似和数据同化技术（降低维数粒子滤波器）的集成。该方法在线应用，并基于离线收集的高保真模拟数据和一组稀疏实验数据。结果显示，与基于物理的高保真模拟相比，所提出的数据驱动建模方法可以获得数量级的计算时间减少。仅使用模拟数据作为输入，数据驱动预测技术可以生成具有可比准确度的预测。当同时使用模拟数据和稀疏物理实验测量作为输入，数据驱动方法结合嵌入的数据同化技术有望在长期预测中超越仅使用高保真模拟的预测。此外，本文证明数据驱动建模方法可以重现基于物理的模拟所得的最大阻力关系标度律，这可能表明其具有超越单一案例的基础预测能力。这些结果有望帮助机器人在已知和复杂地形中的导航和探索，在离线和在线阶段均如此。 

---
# Invariant Extended Kalman Filter for Autonomous Surface Vessels with Partial Orientation Measurements 

**Title (ZH)**: 部分姿态测量的自治水面船舶不变扩展卡尔曼滤波器 

**Authors**: Derek Benham, Easton Potokar, Joshua G. Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2506.10850)  

**Abstract**: Autonomous surface vessels (ASVs) are increasingly vital for marine science, offering robust platforms for underwater mapping and inspection. Accurate state estimation, particularly of vehicle pose, is paramount for precise seafloor mapping, as even small surface deviations can have significant consequences when sensing the seafloor below. To address this challenge, we propose an Invariant Extended Kalman Filter (InEKF) framework designed to integrate partial orientation measurements. While conventional estimation often relies on relative position measurements to fixed landmarks, open ocean ASVs primarily observe a receding horizon. We leverage forward-facing monocular cameras to estimate roll and pitch with respect to this horizon, which provides yaw-ambiguous partial orientation information. To effectively utilize these measurements within the InEKF, we introduce a novel framework for incorporating such partial orientation data. This approach contrasts with traditional InEKF implementations that assume full orientation measurements and is particularly relevant for planar vehicle motion constrained to a "seafaring plane." This paper details the developed InEKF framework; its integration with horizon-based roll/pitch observations and dual-antenna GPS heading measurements for ASV state estimation; and provides a comparative analysis against the InEKF using full orientation and a Multiplicative EKF (MEKF). Our results demonstrate the efficacy and robustness of the proposed partial orientation measurements for accurate ASV state estimation in open ocean environments. 

**Abstract (ZH)**: 自主水面船（ASVs）在海洋科学中日益重要，提供了一种用于水下测绘和检查的坚稳平台。精确的状态估计，尤其是车辆姿态的估计，对于精确的海底测绘至关重要，因为即使是很小的表面偏差也可能对海底传感产生重大影响。为了应对这一挑战，我们提出了一种不变广义卡尔曼滤波器（InEKF）框架，用于整合部分姿态测量。传统的估计方法通常依赖于相对于固定陆标的位置测量，而开阔海域中的ASVs主要观察的是逐渐远离的天际线。我们利用面向前方的单目摄像头来估计相对于天际线的横滚角和俯仰角，从而获得具有航向不确定性的一部分姿态信息。为了有效利用这些测量值在InEKF框架中，我们提出了一种新的整合部分姿态数据的框架。这一方法与传统的假设具有完整姿态测量的InEKF实现不同，并且特别适用于平面车辆运动受限于“航海平面”的情况。本文详细介绍了所开发的InEKF框架；其与基于天际线的横滚/俯仰观察以及双天线GPS航向测量的集成，用于ASV状态估计；并且提供了与具有完整姿态的InEKF和乘法卡尔曼滤波器（MEKF）的比较分析。我们的结果证明了所提出的部分姿态测量在开阔海域中进行准确的ASV状态估计的有效性和鲁棒性。 

---
# RationalVLA: A Rational Vision-Language-Action Model with Dual System 

**Title (ZH)**: 理性的多模态模型：带有双系统的作用视知觉模型 

**Authors**: Wenxuan Song, Jiayi Chen, Wenxue Li, Xu He, Han Zhao, Pengxiang Ding Shiyan Su, Feilong Tang, Xuelian Cheng, Donglin Wang, Zongyuan Ge, Xinhu Zheng, Zhe Liu, Hesheng Wang, Yunhui Liu, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10826)  

**Abstract**: A fundamental requirement for real-world robotic deployment is the ability to understand and respond to natural language instructions. Existing language-conditioned manipulation tasks typically assume that instructions are perfectly aligned with the environment. This assumption limits robustness and generalization in realistic scenarios where instructions may be ambiguous, irrelevant, or infeasible. To address this problem, we introduce RAtional MAnipulation (RAMA), a new benchmark that challenges models with both unseen executable instructions and defective ones that should be rejected. In RAMA, we construct a dataset with over 14,000 samples, including diverse defective instructions spanning six dimensions: visual, physical, semantic, motion, safety, and out-of-context. We further propose the Rational Vision-Language-Action model (RationalVLA). It is a dual system for robotic arms that integrates the high-level vision-language model with the low-level manipulation policy by introducing learnable latent space embeddings. This design enables RationalVLA to reason over instructions, reject infeasible commands, and execute manipulation effectively. Experiments demonstrate that RationalVLA outperforms state-of-the-art baselines on RAMA by a 14.5% higher success rate and 0.94 average task length, while maintaining competitive performance on standard manipulation tasks. Real-world trials further validate its effectiveness and robustness in practical applications. Our project page is this https URL. 

**Abstract (ZH)**: 现实世界机器人部署的基本要求是能够理解和响应自然语言指令。现有基于语言的操纵任务通常假设指令与环境完全对齐。这种假设限制了在实际场景中的鲁棒性和泛化性，因为指令可能具有模糊性、无关性或不可行性。为了解决这一问题，我们引入了RAtional MAnipulation (RAMA)，这是一个新的基准，挑战模型处理未见过的可执行指令和应被拒绝的错误指令。在RAMA中，我们构建了一个包含超过14,000个样本的数据集，涵盖六维缺陷指令：视觉、物理、语义、运动、安全和脱节。我们进一步提出了Rational Vision-Language-Action模型（RationalVLA）。这是一种双系统，通过引入可学习的潜在空间嵌入将高层视觉-语言模型与低层操纵策略相结合。这种设计使RationalVLA能够处理指令、拒绝不可行的命令并有效执行操纵。实验表明，RationalVLA在RAMA上的成功率比最先进的基线高出14.5%，平均任务长度缩短0.94个单位，并且在标准操纵任务上保持了竞争力。在现实世界的试验证实了其在实际应用中的有效性和鲁棒性。我们的项目页面为 this https URL。 

---
# In-Hand Object Pose Estimation via Visual-Tactile Fusion 

**Title (ZH)**: 基于视觉-触觉融合的手持物体姿态估计 

**Authors**: Felix Nonnengießer, Alap Kshirsagar, Boris Belousov, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.10787)  

**Abstract**: Accurate in-hand pose estimation is crucial for robotic object manipulation, but visual occlusion remains a major challenge for vision-based approaches. This paper presents an approach to robotic in-hand object pose estimation, combining visual and tactile information to accurately determine the position and orientation of objects grasped by a robotic hand. We address the challenge of visual occlusion by fusing visual information from a wrist-mounted RGB-D camera with tactile information from vision-based tactile sensors mounted on the fingertips of a robotic gripper. Our approach employs a weighting and sensor fusion module to combine point clouds from heterogeneous sensor types and control each modality's contribution to the pose estimation process. We use an augmented Iterative Closest Point (ICP) algorithm adapted for weighted point clouds to estimate the 6D object pose. Our experiments show that incorporating tactile information significantly improves pose estimation accuracy, particularly when occlusion is high. Our method achieves an average pose estimation error of 7.5 mm and 16.7 degrees, outperforming vision-only baselines by up to 20%. We also demonstrate the ability of our method to perform precise object manipulation in a real-world insertion task. 

**Abstract (ZH)**: 基于视觉和触觉信息的在手物体姿态估计方法：克服视觉遮挡挑战 

---
# Grounded Vision-Language Navigation for UAVs with Open-Vocabulary Goal Understanding 

**Title (ZH)**: 基于开放词汇目标理解的无人机接地视觉-语言导航 

**Authors**: Yuhang Zhang, Haosheng Yu, Jiaping Xiao, Mir Feroskhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10756)  

**Abstract**: Vision-and-language navigation (VLN) is a long-standing challenge in autonomous robotics, aiming to empower agents with the ability to follow human instructions while navigating complex environments. Two key bottlenecks remain in this field: generalization to out-of-distribution environments and reliance on fixed discrete action spaces. To address these challenges, we propose Vision-Language Fly (VLFly), a framework tailored for Unmanned Aerial Vehicles (UAVs) to execute language-guided flight. Without the requirement for localization or active ranging sensors, VLFly outputs continuous velocity commands purely from egocentric observations captured by an onboard monocular camera. The VLFly integrates three modules: an instruction encoder based on a large language model (LLM) that reformulates high-level language into structured prompts, a goal retriever powered by a vision-language model (VLM) that matches these prompts to goal images via vision-language similarity, and a waypoint planner that generates executable trajectories for real-time UAV control. VLFly is evaluated across diverse simulation environments without additional fine-tuning and consistently outperforms all baselines. Moreover, real-world VLN tasks in indoor and outdoor environments under direct and indirect instructions demonstrate that VLFly achieves robust open-vocabulary goal understanding and generalized navigation capabilities, even in the presence of abstract language input. 

**Abstract (ZH)**: 基于视觉-语言导航的无人飞行器框架（Vision-Language Flight for Unmanned Aerial Vehicles (VLFly)） 

---
# An $O(n$)-Algorithm for the Higher-Order Kinematics and Inverse Dynamics of Serial Manipulators using Spatial Representation of Twists 

**Title (ZH)**: 一种基于刚体运动表示的串联 manipulator 的高阶运动学和逆动力学的 O(n) 算法 

**Authors**: Andreas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2506.10686)  

**Abstract**: Optimal control in general, and flatness-based control in particular, of robotic arms necessitate to compute the first and second time derivatives of the joint torques/forces required to achieve a desired motion. In view of the required computational efficiency, recursive $O(n)$-algorithms were proposed to this end. Aiming at compact yet efficient formulations, a Lie group formulation was recently proposed, making use of body-fixed and hybrid representation of twists and wrenches. In this paper a formulation is introduced using the spatial representation. The second-order inverse dynamics algorithm is accompanied by a fourth-order forward and inverse kinematics algorithm. An advantage of all Lie group formulations is that they can be parameterized in terms of vectorial quantities that are readily available. The method is demonstrated for the 7 DOF Franka Emika Panda robot. 

**Abstract (ZH)**: 基于空间表示的冗余自由度机械臂的最优控制及基于平坦性控制 

---
# EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence 

**Title (ZH)**: EmbodiedGen: 通往生成式三维世界引擎的体态智能之路 

**Authors**: Wang Xinjie, Liu Liu, Cao Yu, Wu Ruiqi, Qin Wenkang, Wang Dehui, Sui Wei, Su Zhizhong  

**Link**: [PDF](https://arxiv.org/pdf/2506.10600)  

**Abstract**: Constructing a physically realistic and accurately scaled simulated 3D world is crucial for the training and evaluation of embodied intelligence tasks. The diversity, realism, low cost accessibility and affordability of 3D data assets are critical for achieving generalization and scalability in embodied AI. However, most current embodied intelligence tasks still rely heavily on traditional 3D computer graphics assets manually created and annotated, which suffer from high production costs and limited realism. These limitations significantly hinder the scalability of data driven approaches. We present EmbodiedGen, a foundational platform for interactive 3D world generation. It enables the scalable generation of high-quality, controllable and photorealistic 3D assets with accurate physical properties and real-world scale in the Unified Robotics Description Format (URDF) at low cost. These assets can be directly imported into various physics simulation engines for fine-grained physical control, supporting downstream tasks in training and evaluation. EmbodiedGen is an easy-to-use, full-featured toolkit composed of six key modules: Image-to-3D, Text-to-3D, Texture Generation, Articulated Object Generation, Scene Generation and Layout Generation. EmbodiedGen generates diverse and interactive 3D worlds composed of generative 3D assets, leveraging generative AI to address the challenges of generalization and evaluation to the needs of embodied intelligence related research. Code is available at this https URL. 

**Abstract (ZH)**: 构建一个物理真实且准确缩放的三维世界对于训练和评估具身智能任务至关重要。三维数据资产的多样性、逼真度、低成本和易获取性对于实现具身人工智能的泛化和可扩展性至关重要。然而，当前大多数具身智能任务仍 heavily 依赖于手工创建和标注的传统三维计算机图形资产，这些资产存在高昂的制作成本和有限的逼真度问题。这些局限性显著阻碍了数据驱动方法的可扩展性。我们提出 EmbodiedGen，一个交互式三维世界生成的基础平台。它能够以低成本生成高质量、可控制和照片级真实的三维资产，这些资产具有准确的物理属性和真实世界比例，符合统一机器人描述格式（URDF）。这些资产可以直接导入各种物理仿真引擎，以支持精确的物理控制，从而支持训练和评估中的下游任务。EmbodiedGen 是一个易于使用且功能齐全的工具包，由六个关键模块组成：图像到三维、文本到三维、纹理生成、连杆对象生成、场景生成和布局生成。EmbodiedGen 利用生成式人工智能生成多样化的交互式三维世界，以应对具身智能相关研究中泛化和评估的挑战。代码可从以下链接获取。 

---
# Are We Generalizing from the Exception? An In-the-Wild Study on Group-Sensitive Conversation Design in Human-Agent Interactions 

**Title (ZH)**: 我们从例外中泛化吗？关于人类-代理交互中分组敏感对话设计的野外研究 

**Authors**: Ana Müller, Sabina Jeschke, Anja Richert  

**Link**: [PDF](https://arxiv.org/pdf/2506.10462)  

**Abstract**: This paper investigates the impact of a group-adaptive conversation design in two socially interactive agents (SIAs) through two real-world studies. Both SIAs - Furhat, a social robot, and MetaHuman, a virtual agent - were equipped with a conversational artificial intelligence (CAI) backend combining hybrid retrieval and generative models. The studies were carried out in an in-the-wild setting with a total of $N = 188$ participants who interacted with the SIAs - in dyads, triads or larger groups - at a German museum. Although the results did not reveal a significant effect of the group-sensitive conversation design on perceived satisfaction, the findings provide valuable insights into the challenges of adapting CAI for multi-party interactions and across different embodiments (robot vs.\ virtual agent), highlighting the need for multimodal strategies beyond linguistic pluralization. These insights contribute to the fields of Human-Agent Interaction (HAI), Human-Robot Interaction (HRI), and broader Human-Machine Interaction (HMI), providing insights for future research on effective dialogue adaptation in group settings. 

**Abstract (ZH)**: 本文通过两项实地研究，探讨了一种群体自适应对话设计在两个社会互动代理（SIAs）中的影响。在德国博物馆，共有188名参与者与Furhat（一种社会机器人）和MetaHuman（一种虚拟代理）等SIAs进行了二元、三元或更大规模的互动。尽管研究结果未能显著显示群体敏感对话设计对满意度感知的影响，但 findings 为多方面适应会话人工智能（CAI）以及不同类型实体（机器人 vs. 虚拟代理）之间的交互挑战提供了有价值的见解，突显了超越语言多样化的多模态策略的必要性。这些见解为人类-代理交互（HAI）、人类-机器人交互（HRI）和更广泛的人机交互（HMI）领域未来关于群组环境中有效对话适应的研究提供了指导。 

---
# RICE: Reactive Interaction Controller for Cluttered Canopy Environment 

**Title (ZH)**: 稻草人控制器：杂乱植被环境下的反应性交互控制 

**Authors**: Nidhi Homey Parayil, Thierry Peynot, Chris Lehnert  

**Link**: [PDF](https://arxiv.org/pdf/2506.10383)  

**Abstract**: Robotic navigation in dense, cluttered environments such as agricultural canopies presents significant challenges due to physical and visual occlusion caused by leaves and branches. Traditional vision-based or model-dependent approaches often fail in these settings, where physical interaction without damaging foliage and branches is necessary to reach a target. We present a novel reactive controller that enables safe navigation for a robotic arm in a contact-rich, cluttered, deformable environment using end-effector position and real-time tactile feedback. Our proposed framework's interaction strategy is based on a trade-off between minimizing disturbance by maneuvering around obstacles and pushing through them to move towards the target. We show that over 35 trials in 3 experimental plant setups with an occluded target, the proposed controller successfully reached the target in all trials without breaking any branch and outperformed the state-of-the-art model-free controller in robustness and adaptability. This work lays the foundation for safe, adaptive interaction in cluttered, contact-rich deformable environments, enabling future agricultural tasks such as pruning and harvesting in plant canopies. 

**Abstract (ZH)**: 密集遮蔽环境中基于机器人的导航研究：面向农业冠层的鲁棒适应性交互控制 

---
# Towards more efficient quantitative safety validation of residual risk for assisted and automated driving 

**Title (ZH)**: 面向辅助和自动驾驶残余风险更高效的定量安全验证 

**Authors**: Daniel Betschinske, Malte Schrimpf, Steven Peters, Kamil Klonecki, Jan Peter Karch, Moritz Lippert  

**Link**: [PDF](https://arxiv.org/pdf/2506.10363)  

**Abstract**: The safety validation of Advanced Driver Assistance Systems (ADAS) and Automated Driving Systems (ADS) increasingly demands efficient and reliable methods to quantify residual risk while adhering to international standards such as ISO 21448. Traditionally, Field Operational Testing (FOT) has been pivotal for macroscopic safety validation of automotive driving functions up to SAE automation level 2. However, state-of-the-art derivations for empirical safety demonstrations using FOT often result in impractical testing efforts, particularly at higher automation levels. Even at lower automation levels, this limitation - coupled with the substantial costs associated with FOT - motivates the exploration of approaches to enhance the efficiency of FOT-based macroscopic safety validation. Therefore, this publication systematically identifies and evaluates state-of-the-art Reduction Approaches (RAs) for FOT, including novel methods reported in the literature. Based on an analysis of ISO 21448, two models are derived: a generic model capturing the argumentation components of the standard, and a base model, exemplarily applied to Automatic Emergency Braking (AEB) systems, establishing a baseline for the real-world driving requirement for a Quantitative Safety Validation of Residual Risk (QSVRR). Subsequently, the RAs are assessed using four criteria: quantifiability, threats to validity, missing links, and black box compatibility, highlighting potential benefits, inherent limitations, and identifying key areas for further research. Our evaluation reveals that, while several approaches offer potential, none are free from missing links or other substantial shortcomings. Moreover, no identified alternative can fully replace FOT, reflecting its crucial role in the safety validation of ADAS and ADS. 

**Abstract (ZH)**: 高级驾驶辅助系统（ADAS）和自动驾驶系统（ADS）的安全验证越来越多地需要符合国际标准（如ISO 21448）的高效可靠方法来量化剩余风险。传统的场操作测试（FOT）在宏观层面验证自动驾驶功能（至SAE自动化水平2）方面一直至关重要。然而，用于FOT的先进经验性安全演示推导往往导致不切实际的测试努力，特别是在更高自动化水平时。即使在较低自动化水平，FOT的这一局限性以及其高昂的成本促使探索提高FOT宏层面安全验证效率的方法。因此，本文系统地识别和评估了FOT的最新减小型方案（RAs），包括文献中报道的新方法。基于对ISO 21448的分析，提出了两个模型：一个泛化模型捕捉标准的论点组件，以及一个基础模型，用于例示自动紧急制动（AEB）系统的实车安全性要求，建立量化剩余风险（QSVRR）的实际驾驶需求基准。随后，使用四个标准（可量化性、有效性的威胁、缺失链接和黑盒兼容性）评估RAs，强调潜在益处、固有的局限性，并确定需要进一步研究的关键领域。我们的评估表明，尽管一些方法具有潜力，但没有一个方法是完美的，更没有一种方法能够完全取代FOT在ADAS和ADS安全性验证中的关键作用。 

---
# Demonstrating Multi-Suction Item Picking at Scale via Multi-Modal Learning of Pick Success 

**Title (ZH)**: 基于多模态学习的多吸盘抓取物体大规模拣选演示 

**Authors**: Che Wang, Jeroen van Baar, Chaitanya Mitash, Shuai Li, Dylan Randle, Weiyao Wang, Sumedh Sontakke, Kostas E. Bekris, Kapil Katyal  

**Link**: [PDF](https://arxiv.org/pdf/2506.10359)  

**Abstract**: This work demonstrates how autonomously learning aspects of robotic operation from sparsely-labeled, real-world data of deployed, engineered solutions at industrial scale can provide with solutions that achieve improved performance. Specifically, it focuses on multi-suction robot picking and performs a comprehensive study on the application of multi-modal visual encoders for predicting the success of candidate robotic picks. Picking diverse items from unstructured piles is an important and challenging task for robot manipulation in real-world settings, such as warehouses. Methods for picking from clutter must work for an open set of items while simultaneously meeting latency constraints to achieve high throughput. The demonstrated approach utilizes multiple input modalities, such as RGB, depth and semantic segmentation, to estimate the quality of candidate multi-suction picks. The strategy is trained from real-world item picking data, with a combination of multimodal pretrain and finetune. The manuscript provides comprehensive experimental evaluation performed over a large item-picking dataset, an item-picking dataset targeted to include partial occlusions, and a package-picking dataset, which focuses on containers, such as boxes and envelopes, instead of unpackaged items. The evaluation measures performance for different item configurations, pick scenes, and object types. Ablations help to understand the effects of in-domain pretraining, the impact of different modalities and the importance of finetuning. These ablations reveal both the importance of training over multiple modalities but also the ability of models to learn during pretraining the relationship between modalities so that during finetuning and inference, only a subset of them can be used as input. 

**Abstract (ZH)**: 基于稀疏标注实际数据的工业规模自主学习在机器人操作中的应用：以多吸盘机器人拣选为例 

---
# Using Language and Road Manuals to Inform Map Reconstruction for Autonomous Driving 

**Title (ZH)**: 使用语言和道路手册指导自动驾驶中的地图重建 

**Authors**: Akshar Tumu, Henrik I. Christensen, Marcell Vazquez-Chanlatte, Chikao Tsuchiya, Dhaval Bhanderi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10317)  

**Abstract**: Lane-topology prediction is a critical component of safe and reliable autonomous navigation. An accurate understanding of the road environment aids this task. We observe that this information often follows conventions encoded in natural language, through design codes that reflect the road structure and road names that capture the road functionality. We augment this information in a lightweight manner to SMERF, a map-prior-based online lane-topology prediction model, by combining structured road metadata from OSM maps and lane-width priors from Road design manuals with the road centerline encodings. We evaluate our method on two geo-diverse complex intersection scenarios. Our method shows improvement in both lane and traffic element detection and their association. We report results using four topology-aware metrics to comprehensively assess the model performance. These results demonstrate the ability of our approach to generalize and scale to diverse topologies and conditions. 

**Abstract (ZH)**: 基于道路拓扑预测的安全可靠自主导航的关键组件：通过结合OSM地图结构化道路元数据和道路设计手册中的车道宽度先验知识来增强SMERF模型，以轻量级方式补充道路拓扑信息，并在两个地理多样化的复杂交叉口场景中进行评估。我们使用四种拓扑感知指标全面评估模型性能，结果表明该方法具有泛化能力和适应多样拓扑结构和条件的能力。 

---
# Multi-Timescale Dynamics Model Bayesian Optimization for Plasma Stabilization in Tokamaks 

**Title (ZH)**: 多时间尺度动力学模型贝叶斯优化方法在托卡马克中的等离子体稳定化 

**Authors**: Rohit Sonker, Alexandre Capone, Andrew Rothstein, Hiro Josep Farre Kaga, Egemen Kolemen, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2506.10287)  

**Abstract**: Machine learning algorithms often struggle to control complex real-world systems. In the case of nuclear fusion, these challenges are exacerbated, as the dynamics are notoriously complex, data is poor, hardware is subject to failures, and experiments often affect dynamics beyond the experiment's duration. Existing tools like reinforcement learning, supervised learning, and Bayesian optimization address some of these challenges but fail to provide a comprehensive solution. To overcome these limitations, we present a multi-scale Bayesian optimization approach that integrates a high-frequency data-driven dynamics model with a low-frequency Gaussian process. By updating the Gaussian process between experiments, the method rapidly adapts to new data, refining the predictions of the less reliable dynamical model. We validate our approach by controlling tearing instabilities in the DIII-D nuclear fusion plant. Offline testing on historical data shows that our method significantly outperforms several baselines. Results on live experiments on the DIII-D tokamak, conducted under high-performance plasma scenarios prone to instabilities, shows a 50% success rate, marking a 117% improvement over historical outcomes. 

**Abstract (ZH)**: 机器学习算法在控制复杂现实系统时常常面临挑战。在核聚变领域，这些挑战更为严峻，因为动力学极其复杂、数据质量较差、硬件易故障，且实验往往会影响实验持续时间之外的动力学。现有的工具如强化学习、监督学习和贝叶斯优化部分解决了这些问题，但未能提供全面的解决方案。为克服这些限制，我们提出了一种多尺度贝叶斯优化方法，该方法将高频数据驱动的动力学模型与低频高斯过程结合。通过在实验之间更新高斯过程，该方法能够迅速适应新数据，改进可靠性较低的动力学模型的预测。我们通过控制DIII-D核聚变装置中的剥离不稳定性来验证我们的方法。离线历史数据测试表明，我们的方法显著优于几种基线方法。在DIII-D托卡马克进行的实时实验中，在高度易发生不稳定的高性能等离子体场景下，成功率达到50%，比历史结果提高了117%。 

---
# Learning Safe Control via On-the-Fly Bandit Exploration 

**Title (ZH)**: 基于即时bandit探索的.safe控制学习 

**Authors**: Alexandre Capone, Ryan Cosner, Aaaron Ames, Sandra Hirche  

**Link**: [PDF](https://arxiv.org/pdf/2506.10279)  

**Abstract**: Control tasks with safety requirements under high levels of model uncertainty are increasingly common. Machine learning techniques are frequently used to address such tasks, typically by leveraging model error bounds to specify robust constraint-based safety filters. However, if the learned model uncertainty is very high, the corresponding filters are potentially invalid, meaning no control input satisfies the constraints imposed by the safety filter. While most works address this issue by assuming some form of safe backup controller, ours tackles it by collecting additional data on the fly using a Gaussian process bandit-type algorithm. We combine a control barrier function with a learned model to specify a robust certificate that ensures safety if feasible. Whenever infeasibility occurs, we leverage the control barrier function to guide exploration, ensuring the collected data contributes toward the closed-loop system safety. By combining a safety filter with exploration in this manner, our method provably achieves safety in a setting that allows for a zero-mean prior dynamics model, without requiring a backup controller. To the best of our knowledge, it is the first safe learning-based control method that achieves this. 

**Abstract (ZH)**: 在高模型不确定性下执行具有安全要求的任务正变得越来越常见。机器学习技术通常被用来处理这类任务，通常通过利用模型误差边界来指定鲁棒的基于约束的安全过滤器。然而，如果学习到的模型不确定性非常高，相应的过滤器可能是无效的，这意味着没有控制输入能够满足安全过滤器施加的约束。大多数工作通过假设某种形式的安全备用控制器来解决这个问题，而我们则通过使用高斯过程宝瓶式的算法收集额外数据来处理这一问题。我们将控制屏障函数与学习到的模型相结合，以指定一个鲁棒证书，该证书确保如果可行，能确保安全性。每当不可行性发生时，我们利用控制屏障函数来引导探索，确保收集的数据有助于闭环系统的安全性。通过以这种方式结合安全过滤器和探索性，我们的方法在允许零均值先验动力学模型的环境中能够证明实现安全性，而无需备用控制器。据我们所知，这是第一个能够在不需要备用控制器的情况下实现这一目标的安全学习控制方法。 

---
# A Novel Feedforward Youla Parameterization Method for Avoiding Local Minima in Stereo Image Based Visual Servoing Control 

**Title (ZH)**: 基于立体图像视觉伺服控制中避免局部极小值的新型前馈Youla参数化方法 

**Authors**: Rongfei Li, Francis Assadian  

**Link**: [PDF](https://arxiv.org/pdf/2506.10252)  

**Abstract**: In robot navigation and manipulation, accurately determining the camera's pose relative to the environment is crucial for effective task execution. In this paper, we systematically prove that this problem corresponds to the Perspective-3-Point (P3P) formulation, where exactly three known 3D points and their corresponding 2D image projections are used to estimate the pose of a stereo camera. In image-based visual servoing (IBVS) control, the system becomes overdetermined, as the 6 degrees of freedom (DoF) of the stereo camera must align with 9 observed 2D features in the scene. When more constraints are imposed than available DoFs, global stability cannot be guaranteed, as the camera may become trapped in a local minimum far from the desired configuration during servoing. To address this issue, we propose a novel control strategy for accurately positioning a calibrated stereo camera. Our approach integrates a feedforward controller with a Youla parameterization-based feedback controller, ensuring robust servoing performance. Through simulations, we demonstrate that our method effectively avoids local minima and enables the camera to reach the desired pose accurately and efficiently. 

**Abstract (ZH)**: 机器人导航与操作中，准确确定相机相对于环境的姿态对于有效执行任务至关重要。本文系统地证明了这一问题等同于基于三点视角（P3P）的求解方式，即利用三个已知的3D点及其对应的2D图像投影来估计立体相机的姿态。在基于图像的视觉伺服（IBVS）控制中，系统变得过定，因为立体相机的6个自由度必须与场景中观察到的9个2D特征对齐。当施加的约束条件超过可自由度时，全局稳定性不能得到保证，相机在伺服过程中可能会被困在远离期望配置的局部极小值中。为解决这一问题，我们提出了一种新的用于精确定位校准立体相机的控制策略。该方法将前馈控制器与基于Youla参数化的反馈控制器相结合，确保视觉伺服性能的鲁棒性。通过仿真实验，我们展示了该方法有效地避免了局部极小值，并使相机能够准确高效地达到期望姿态。 

---
# Innovative Adaptive Imaged Based Visual Servoing Control of 6 DoFs Industrial Robot Manipulators 

**Title (ZH)**: 基于图像的自适应视觉伺服控制的6轴工业机器人 manipulator创新技术 

**Authors**: Rongfei Li, Francis Assadian  

**Link**: [PDF](https://arxiv.org/pdf/2506.10240)  

**Abstract**: Image-based visual servoing (IBVS) methods have been well developed and used in many applications, especially in pose (position and orientation) alignment. However, most research papers focused on developing control solutions when 3D point features can be detected inside the field of view. This work proposes an innovative feedforward-feedback adaptive control algorithm structure with the Youla Parameterization method. A designed feature estimation loop ensures stable and fast motion control when point features are outside the field of view. As 3D point features move inside the field of view, the IBVS feedback loop preserves the precision of the pose at the end of the control period. Also, an adaptive controller is developed in the feedback loop to stabilize the system in the entire range of operations. The nonlinear camera and robot manipulator model is linearized and decoupled online by an adaptive algorithm. The adaptive controller is then computed based on the linearized model evaluated at current linearized point. The proposed solution is robust and easy to implement in different industrial robotic systems. Various scenarios are used in simulations to validate the effectiveness and robust performance of the proposed controller. 

**Abstract (ZH)**: 基于图像的视觉伺服（IBVS）方法已在许多应用中得到发展和使用，尤其是在姿态（位置和方向）对准方面。然而，大多数研究论文集中在可以在视野内检测到3D点特征时开发控制解决方案。本文提出了一个创新的前馈-反馈自适应控制算法结构，并采用了尤拉参数化方法。设计的特征估计环路确保了当点特征位于视野外时的稳定和快速运动控制。当3D点特征移动到视野内时，IBVS反馈环路在控制期末期保持姿态精度。此外，在反馈环路中开发了自适应控制器，以在操作的整个范围内稳定系统。非线性的相机和机器人 manipulator 模型在线上通过自适应算法进行线性化和解耦。然后基于在当前线性化点上评价的线性化模型计算自适应控制器。所提出的解决方案在不同工业机器人系统中具有鲁棒性和易于实现的特点。通过各种场景在 simulations 中验证了所提出控制器的有效性和鲁棒性能。 

---
# A Unified Framework for Probabilistic Dynamic-, Trajectory- and Vision-based Virtual Fixtures 

**Title (ZH)**: 一种统一框架：基于概率动态、轨迹和视觉的虚拟 fixtures 

**Authors**: Maximilian Mühlbauer, Freek Stulp, Sylvain Calinon, Alin Albu-Schäffer, João Silvério  

**Link**: [PDF](https://arxiv.org/pdf/2506.10239)  

**Abstract**: Probabilistic Virtual Fixtures (VFs) enable the adaptive selection of the most suitable haptic feedback for each phase of a task, based on learned or perceived uncertainty. While keeping the human in the loop remains essential, for instance, to ensure high precision, partial automation of certain task phases is critical for productivity. We present a unified framework for probabilistic VFs that seamlessly switches between manual fixtures, semi-automated fixtures (with the human handling precise tasks), and full autonomy. We introduce a novel probabilistic Dynamical System-based VF for coarse guidance, enabling the robot to autonomously complete certain task phases while keeping the human operator in the loop. For tasks requiring precise guidance, we extend probabilistic position-based trajectory fixtures with automation allowing for seamless human interaction as well as geometry-awareness and optimal impedance gains. For manual tasks requiring very precise guidance, we also extend visual servoing fixtures with the same geometry-awareness and impedance behaviour. We validate our approach experimentally on different robots, showcasing multiple operation modes and the ease of programming fixtures. 

**Abstract (ZH)**: 概率虚拟 fixtures (VFs) 允许根据学习到的或感知到的不确定性，在任务的各个阶段选择最合适的触觉反馈。尽管保持人类在环内对于确保高精度仍然是必要的，但对某些任务阶段的部分自动化对于提高生产力至关重要。我们提出了一种统一的概率虚拟 fixtures 框架，该框架可以在手动 fixtures、半自动化 fixtures（人类处理精确任务）和完全自主之间无缝切换。我们引入了一种基于概率动力系统的新颖虚拟 fixtures，用于粗略指导，使机器人能够在保持人类操作员在环内的同时自主完成某些任务阶段。对于需要精确指导的任务，我们扩展了基于概率位置的轨迹 fixtures，引入了自动化功能，使其能够无缝地与人类交互，并具备几何感知能力和最优阻抗增益。对于需要非常精确指导的手动任务，我们还扩展了视觉伺服 fixtures，使其具备相同的几何感知能力和阻抗行为。我们在不同的机器人上实验验证了我们的方法，展示了多种操作模式以及 fixtures 编程的简单性。 

---
# A Navigation Framework Utilizing Vision-Language Models 

**Title (ZH)**: 利用视觉-语言模型的导航框架 

**Authors**: Yicheng Duan, Kaiyu tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10172)  

**Abstract**: Vision-and-Language Navigation (VLN) presents a complex challenge in embodied AI, requiring agents to interpret natural language instructions and navigate through visually rich, unfamiliar environments. Recent advances in large vision-language models (LVLMs), such as CLIP and Flamingo, have significantly improved multimodal understanding but introduced new challenges related to computational cost and real-time deployment. In this project, we propose a modular, plug-and-play navigation framework that decouples vision-language understanding from action planning. By integrating a frozen vision-language model, Qwen2.5-VL-7B-Instruct, with lightweight planning logic, we aim to achieve flexible, fast, and adaptable navigation without extensive model fine-tuning. Our framework leverages prompt engineering, structured history management, and a two-frame visual input strategy to enhance decision-making continuity across navigation steps. We evaluate our system on the Room-to-Room benchmark within the VLN-CE setting using the Matterport3D dataset and Habitat-Lab simulation environment. Although our initial results reveal challenges in generalizing to unseen environments under strict evaluation settings, our modular approach lays a foundation for scalable and efficient navigation systems, highlighting promising directions for future improvement through enhanced environmental priors and expanded multimodal input integration. 

**Abstract (ZH)**: 视觉和语言导航（VLN）在体现式AI中提出了一项复杂的挑战，要求代理解析自然语言指令并导航通过视觉丰富且不熟悉的环境。近期大规模视觉语言模型（LVLM）的进步，如CLIP和Flamingo，显著提升了多模态理解能力，但也带来了计算成本和实时部署方面的新的挑战。在本项目中，我们提出了一种模块化且插拔式的导航框架，将视觉语言理解与行动规划分离。通过整合冻结的视觉语言模型Qwen2.5-VL-7B-Instruct和轻量级规划逻辑，我们旨在实现灵活、快速且适应性强的导航，而无需广泛的模型微调。我们的框架利用指令工程、结构化历史管理以及两帧视觉输入策略，以增强导航步骤间的决策连续性。我们使用Matterport3D数据集和Habitat-Lab模拟环境，在VLN-CE设置下的Room-to-Room基准上评估了我们的系统。尽管我们的初步结果显示在严格评估设置下泛化到未见过的环境存在挑战，但我们的模块化方法为可扩展且高效的导航系统奠定了基础，并指出通过增强环境先验知识和扩展多模态输入集成而改善方向的前景。 

---
# One For All: LLM-based Heterogeneous Mission Planning in Precision Agriculture 

**Title (ZH)**: 一站式解决：基于LLM的精准农业异构任务规划 

**Authors**: Marcos Abel Zuzuárregui, Mustafa Melih Toslak, Stefano Carpin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10106)  

**Abstract**: Artificial intelligence is transforming precision agriculture, offering farmers new tools to streamline their daily operations. While these technological advances promise increased efficiency, they often introduce additional complexity and steep learning curves that are particularly challenging for non-technical users who must balance tech adoption with existing workloads. In this paper, we present a natural language (NL) robotic mission planner that enables non-specialists to control heterogeneous robots through a common interface. By leveraging large language models (LLMs) and predefined primitives, our architecture seamlessly translates human language into intermediate descriptions that can be executed by different robotic platforms. With this system, users can formulate complex agricultural missions without writing any code. In the work presented in this paper, we extend our previous system tailored for wheeled robot mission planning through a new class of experiments involving robotic manipulation and computer vision tasks. Our results demonstrate that the architecture is both general enough to support a diverse set of robots and powerful enough to execute complex mission requests. This work represents a significant step toward making robotic automation in precision agriculture more accessible to non-technical users. 

**Abstract (ZH)**: 人工智能正在 transforming 精准农业，为农民提供了新的工具以简化日常运营。尽管这些技术进步承诺了更高的效率，但它们常常引入额外的复杂性和陡峭的学习曲线，这对必须在技术采用与现有工作负担之间取得平衡的非技术人员来说尤其具有挑战性。在本文中，我们提出了一种自然语言（NL）机器人任务规划器，使非专家能够通过一个共同界面控制异构机器人。通过利用大规模语言模型（LLMs）和预定义的原子操作，我们的架构能够无缝地将人类语言转换为可在不同机器人平台上执行的中间描述。通过该系统，用户可以在不需要编写任何代码的情况下制定复杂的农业任务。在本文中，我们扩展了我们之前专为轮式机器人任务规划设计的系统，通过涉及机器人操作和计算机视觉任务的一类新实验。我们的结果表明，该架构既通用到足以支持一系列不同的机器人，又强大到足以执行复杂的任务请求。这项工作朝着使精准农业中的机器人自动化对非技术人员更加可及迈出了重要一步。 

---
# Estimating the Joint Probability of Scenario Parameters with Gaussian Mixture Copula Models 

**Title (ZH)**: 基于高斯混合 copula 模型的场景参数联合概率估计算法 

**Authors**: Christian Reichenbächer, Philipp Rank, Jochen Hipp, Oliver Bringmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.10098)  

**Abstract**: This paper presents the first application of Gaussian Mixture Copula Models to the statistical modeling of driving scenarios for the safety validation of automated driving systems. Knowledge of the joint probability distribution of scenario parameters is essential for scenario-based safety assessment, where risk quantification depends on the likelihood of concrete parameter combinations. Gaussian Mixture Copula Models bring together the multimodal expressivity of Gaussian Mixture Models and the flexibility of copulas, enabling separate modeling of marginal distributions and dependencies. We benchmark Gaussian Mixture Copula Models against previously proposed approaches - Gaussian Mixture Models and Gaussian Copula Models - using real-world driving data drawn from scenarios defined in United Nations Regulation No. 157. Our evaluation across 18 million scenario instances demonstrates that Gaussian Mixture Copula Models provide a better fit to the data in terms of both likelihood and Sinkhorn distance. These results suggest that Gaussian Mixture Copula Models are a compelling foundation for future scenario-based validation frameworks. 

**Abstract (ZH)**: 基于高斯混合 copula 模型的自动驾驶场景统计建模及其安全验证应用 

---
# Leveraging LLMs for Mission Planning in Precision Agriculture 

**Title (ZH)**: 利用大规模语言模型进行精准农业的任务规划 

**Authors**: Marcos Abel Zuzuárregui, Stefano Carpin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10093)  

**Abstract**: Robotics and artificial intelligence hold significant potential for advancing precision agriculture. While robotic systems have been successfully deployed for various tasks, adapting them to perform diverse missions remains challenging, particularly because end users often lack technical expertise. In this paper, we present an end-to-end system that leverages large language models (LLMs), specifically ChatGPT, to enable users to assign complex data collection tasks to autonomous robots using natural language instructions. To enhance reusability, mission plans are encoded using an existing IEEE task specification standard, and are executed on robots via ROS2 nodes that bridge high-level mission descriptions with existing ROS libraries. Through extensive experiments, we highlight the strengths and limitations of LLMs in this context, particularly regarding spatial reasoning and solving complex routing challenges, and show how our proposed implementation overcomes them. 

**Abstract (ZH)**: 机器人技术和人工智能在推进精准农业方面具有显著潜力。虽然已经成功部署了各种机器人系统，但将这些系统适应多种任务仍然具有挑战性，尤其是因为最终用户往往缺乏技术专业知识。在这篇论文中，我们提出了一套端到端系统，利用大型语言模型（LLMs），具体是ChatGPT，使用户能够使用自然语言指令将复杂的数据收集任务分配给自主机器人。为了提高可重用性，任务计划使用现有的IEEE任务规范标准进行编码，并通过ROS2节点执行，这些节点将高级任务描述与现有的ROS库桥接起来。通过广泛实验，我们强调了LLMs在这种情境下的优点和局限性，特别是在空间推理和解决复杂路径规划问题方面，并展示了我们提出的实现方法如何克服这些局限性。 

---
# Provable Sim-to-Real Transfer via Offline Domain Randomization 

**Title (ZH)**: 可验证的离线领域随机化实现从模拟到现实的迁移 

**Authors**: Arnaud Fickinger, Abderrahim Bendahi, Stuart Russell  

**Link**: [PDF](https://arxiv.org/pdf/2506.10133)  

**Abstract**: Reinforcement-learning agents often struggle when deployed from simulation to the real-world. A dominant strategy for reducing the sim-to-real gap is domain randomization (DR) which trains the policy across many simulators produced by sampling dynamics parameters, but standard DR ignores offline data already available from the real system. We study offline domain randomization (ODR), which first fits a distribution over simulator parameters to an offline dataset. While a growing body of empirical work reports substantial gains with algorithms such as DROPO, the theoretical foundations of ODR remain largely unexplored. In this work, we (i) formalize ODR as a maximum-likelihood estimation over a parametric simulator family, (ii) prove consistency of this estimator under mild regularity and identifiability conditions, showing it converges to the true dynamics as the dataset grows, (iii) derive gap bounds demonstrating ODRs sim-to-real error is up to an O(M) factor tighter than uniform DR in the finite-simulator case (and analogous gains in the continuous setting), and (iv) introduce E-DROPO, a new version of DROPO which adds an entropy bonus to prevent variance collapse, yielding broader randomization and more robust zero-shot transfer in practice. 

**Abstract (ZH)**: Offline Domain Randomizationagt;往往在从模拟环境部署到真实世界时难以适应。我们研究了Offline Domain Randomization（ODR），首先根据离线数据集拟合模拟器参数的分布。尽管有关ODR的实证研究显示了显著的改进，但其理论基础仍相对未被探索。在本文中，我们（i）将ODR形式化为参数化模拟器族的最大似然估计，（ii）在温和的正则性和可识别性条件下证明了该估计器的一致性，展示了随着数据集规模的增长，该估计器收敛于真实动力学，（iii）推导出间隙界，表明在有限模拟器情形下ODR的sim-to-real误差最多比均匀DR小O(M)倍（在连续情形下也有类似的优势），并（iv）引入了E-DROPO，这是一种改进的DROPO版本，增加了熵奖励以防止方差崩溃，实测中提供了更广泛的随机化和更强鲁棒性的零 shot 转移。 

---
# Cybernetic Marionette: Channeling Collective Agency Through a Wearable Robot in a Live Dancer-Robot Duet 

**Title (ZH)**: 机械傀儡控制装置：通过穿戴式机器人在真人与机器人舞者双人舞中引导集体行为 

**Authors**: Anup Sathya, Jiasheng Li, Zeyu Yan, Adriane Fang, Bill Kules, Jonathan David Martin, Huaishu Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10079)  

**Abstract**: We describe DANCE^2, an interactive dance performance in which audience members channel their collective agency into a dancer-robot duet by voting on the behavior of a wearable robot affixed to the dancer's body. At key moments during the performance, the audience is invited to either continue the choreography or override it, shaping the unfolding interaction through real-time collective input. While post-performance surveys revealed that participants felt their choices meaningfully influenced the performance, voting data across four public performances exhibited strikingly consistent patterns. This tension between what audience members do, what they feel, and what actually changes highlights a complex interplay between agentive behavior, the experience of agency, and power. We reflect on how choreography, interaction design, and the structure of the performance mediate this relationship, offering a live analogy for algorithmically curated digital systems where agency is felt, but not exercised. 

**Abstract (ZH)**: DANCE^2：一种通过投票控制舞者-机器人搭档行为的互动舞蹈表演 

---
# Impacts between multibody systems and deformable structures 

**Title (ZH)**: 多体系统与可变形结构之间的相互作用影响 

**Authors**: Lipinski Krzysztof  

**Link**: [PDF](https://arxiv.org/pdf/2506.10034)  

**Abstract**: Collisions and impacts are the principal reasons for impulsive motions, which we frequently see in dynamic responses of systems. Precise modelling of impacts is a challenging problem due to the lack of the accurate and commonly accepted constitutive law that governs their mechanics. Rigid-body approach and soft contact methods are discussed in this paper and examined in the presented numerical examples. The main focus is set to impacts in systems with multiple unilateral contacts and collisions with elastic elements of the reference. Parameters of interconnecting unilateral springs are under discussion. 

**Abstract (ZH)**: 碰撞和冲击是引起瞬态运动的主要原因，我们经常在系统的动态响应中见到它们。准确模拟能动体的碰撞是一个难题，主要原因是缺乏能够准确描述其力学规律的公认的本构关系。本文讨论了刚体方法和软接触方法，并通过示例数值分析进行了检验。重点放在涉及多个单向接触的系统及其与参考弹性元件的碰撞上，探讨了相互连接的单向弹簧参数。 

---
