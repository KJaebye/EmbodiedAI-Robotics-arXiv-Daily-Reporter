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
# Speech-to-Trajectory: Learning Human-Like Verbal Guidance for Robot Motion 

**Title (ZH)**: 语音转换为运动：学习类人语音指导以实现机器人运动 

**Authors**: Eran Beeri Bamani, Eden Nissinman, Rotem Atari, Nevo Heimann Saadon, Avishai Sintov  

**Link**: [PDF](https://arxiv.org/pdf/2504.05084)  

**Abstract**: Full integration of robots into real-life applications necessitates their ability to interpret and execute natural language directives from untrained users. Given the inherent variability in human language, equivalent directives may be phrased differently, yet require consistent robot behavior. While Large Language Models (LLMs) have advanced language understanding, they often falter in handling user phrasing variability, rely on predefined commands, and exhibit unpredictable outputs. This letter introduces the Directive Language Model (DLM), a novel speech-to-trajectory framework that directly maps verbal commands to executable motion trajectories, bypassing predefined phrases. DLM utilizes Behavior Cloning (BC) on simulated demonstrations of human-guided robot motion. To enhance generalization, GPT-based semantic augmentation generates diverse paraphrases of training commands, labeled with the same motion trajectory. DLM further incorporates a diffusion policy-based trajectory generation for adaptive motion refinement and stochastic sampling. In contrast to LLM-based methods, DLM ensures consistent, predictable motion without extensive prompt engineering, facilitating real-time robotic guidance. As DLM learns from trajectory data, it is embodiment-agnostic, enabling deployment across diverse robotic platforms. Experimental results demonstrate DLM's improved command generalization, reduced dependence on structured phrasing, and achievement of human-like motion. 

**Abstract (ZH)**: 全集成到现实应用中的机器人需要能够理解和执行未训练用户发出的自然语言指令。鉴于人类语言的固有变异性，等效的指令可能有不同的表达方式，但需要一致的机器人行为。虽然大型语言模型（LLMs）提升了语言理解能力，但在处理用户表达的变异性方面通常表现不佳，依赖预定义命令，并表现出不可预测的输出。本信介绍了一种新型指令语言模型（DLM），该模型直接将口头命令映射到可执行的运动轨迹，省去了预定义的命令。DLM 利用模拟的人引导机器人运动示范进行行为克隆（BC）。为了增强泛化能力，基于GPT的语义增强生成多样化的训练指令同义句，标注相同的运动轨迹。DLM 进一步结合了基于扩散策略的轨迹生成，实现适应性运动细化和随机采样。与基于LLM的方法不同，DLM 确保了运动的一致性和可预测性，无需大量的提示工程，促进实时的机器人引导。由于DLM从轨迹数据中学习，因此具有体表无关性，可在多种机器人平台上部署。实验结果表明，DLM 提高了命令泛化能力，减少了对结构化表达的依赖，并实现了类人运动。 

---
# Segmented Trajectory Optimization for Autonomous Parking in Unstructured Environments 

**Title (ZH)**: 非结构化环境中自主泊车分段轨迹优化 

**Authors**: Hang Yu, Renjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.05041)  

**Abstract**: This paper presents a Segmented Trajectory Optimization (STO) method for autonomous parking, which refines an initial trajectory into a dynamically feasible and collision-free one using an iterative SQP-based approach. STO maintains the maneuver strategy of the high-level global planner while allowing curvature discontinuities at switching points to improve maneuver efficiency. To ensure safety, a convex corridor is constructed via GJK-accelerated ellipse shrinking and expansion, serving as safety constraints in each iteration. Numerical simulations in perpendicular and reverse-angled parking scenarios demonstrate that STO enhances maneuver efficiency while ensuring safety. Moreover, computational performance confirms its practicality for real-world applications. 

**Abstract (ZH)**: 基于迭代SQP方法的分段轨迹优化在自动驾驶泊车中的应用 

---
# CloSE: A Compact Shape- and Orientation-Agnostic Cloth State Representation 

**Title (ZH)**: CloSE: 一种紧凑的不依赖形状和方向的衣物状态表示方法 

**Authors**: Jay Kamat, Júlia Borràs, Carme Torras  

**Link**: [PDF](https://arxiv.org/pdf/2504.05033)  

**Abstract**: Cloth manipulation is a difficult problem mainly because of the non-rigid nature of cloth, which makes a good representation of deformation essential. We present a new representation for the deformation-state of clothes. First, we propose the dGLI disk representation, based on topological indices computed for segments on the edges of the cloth mesh border that are arranged on a circular grid. The heat-map of the dGLI disk uncovers patterns that correspond to features of the cloth state that are consistent for different shapes, sizes of positions of the cloth, like the corners and the fold locations. We then abstract these important features from the dGLI disk onto a circle, calling it the Cloth StatE representation (CloSE). This representation is compact, continuous, and general for different shapes. Finally, we show the strengths of this representation in two relevant applications: semantic labeling and high- and low-level planning. The code, the dataset and the video can be accessed from : this https URL 

**Abstract (ZH)**: 布料操纵是一个困难的问题，主要是由于布料的非刚性性质，使得变形的良好表示至关重要。我们提出了一种新的布料变形状态表示方法。首先，我们基于布料网格边界边缘上的段在圆形网格上的排列，提出了dGLI圆盘表示法。dGLI圆盘的热图揭示了与不同形状、大小和位置的布料状态特征一致的模式，如角落和折叠位置。然后，我们将这些重要特征从dGLI圆盘抽象到一个圆上，称为布料状态表示（CloSE）。这种表示方法紧凑、连续且适用于不同形状。最后，我们展示了此表示方法在两个相关应用中的优势：语义标注和高低级规划。代码、数据集和视频可以从以下链接访问：this https URL 

---
# CONCERT: a Modular Reconfigurable Robot for Construction 

**Title (ZH)**: CONCERT：一种可重构模块化建筑机器人 

**Authors**: Luca Rossini, Edoardo Romiti, Arturo Laurenzi, Francesco Ruscelli, Marco Ruzzon, Luca Covizzi, Lorenzo Baccelliere, Stefano Carrozzo, Michael Terzer, Marco Magri, Carlo Morganti, Maolin Lei, Liana Bertoni, Diego Vedelago, Corrado Burchielli, Stefano Cordasco, Luca Muratore, Andrea Giusti, Nikos Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.04998)  

**Abstract**: This paper presents CONCERT, a fully reconfigurable modular collaborative robot (cobot) for multiple on-site operations in a construction site. CONCERT has been designed to support human activities in construction sites by leveraging two main characteristics: high-power density motors and modularity. In this way, the robot is able to perform a wide range of highly demanding tasks by acting as a co-worker of the human operator or by autonomously executing them following user instructions. Most of its versatility comes from the possibility of rapidly changing its kinematic structure by adding or removing passive or active modules. In this way, the robot can be set up in a vast set of morphologies, consequently changing its workspace and capabilities depending on the task to be executed. In the same way, distal end-effectors can be replaced for the execution of different operations. This paper also includes a full description of the software pipeline employed to automatically discover and deploy the robot morphology. Specifically, depending on the modules installed, the robot updates the kinematic, dynamic, and geometric parameters, taking into account the information embedded in each module. In this way, we demonstrate how the robot can be fully reassembled and made operational in less than ten minutes. We validated the CONCERT robot across different use cases, including drilling, sanding, plastering, and collaborative transportation with obstacle avoidance, all performed in a real construction site scenario. We demonstrated the robot's adaptivity and performance in multiple scenarios characterized by different requirements in terms of power and workspace. CONCERT has been designed and built by the Humanoid and Human-Centered Mechatronics Laboratory (HHCM) at the Istituto Italiano di Tecnologia in the context of the European Project Horizon 2020 CONCERT. 

**Abstract (ZH)**: CONCERT：一种用于施工现场多任务操作的全可重构模块化协作机器人 

---
# Wavelet Policy: Imitation Policy Learning in Frequency Domain with Wavelet Transforms 

**Title (ZH)**: 小波策略：基于小波变换的频域imitation策略学习 

**Authors**: Changchuan Yang, Yuhang Dong, Guanzhong Tian, Haizhou Ge, Hongrui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04991)  

**Abstract**: Recent imitation learning policies, often framed as time series prediction tasks, directly map robotic observations-such as high-dimensional visual data and proprioception-into the action space. While time series prediction primarily relies on spatial domain modeling, the underutilization of frequency domain analysis in robotic manipulation trajectory prediction may lead to neglecting the inherent temporal information embedded within action sequences. To address this, we reframe imitation learning policies through the lens of the frequency domain and introduce the Wavelet Policy. This novel approach employs wavelet transforms (WT) for feature preprocessing and extracts multi-scale features from the frequency domain using the SE2MD (Single Encoder to Multiple Decoder) architecture. Furthermore, to enhance feature mapping in the frequency domain and increase model capacity, we introduce a Learnable Frequency-Domain Filter (LFDF) after each frequency decoder, improving adaptability under different visual conditions. Our results show that the Wavelet Policy outperforms state-of-the-art (SOTA) end-to-end methods by over 10% on four challenging robotic arm tasks, while maintaining a comparable parameter count. In long-range settings, its performance declines more slowly as task volume increases. The code will be publicly available. 

**Abstract (ZH)**: 基于频域的imitation学习策略：Wavelet Policy及其应用 

---
# A High-Force Gripper with Embedded Multimodal Sensing for Powerful and Perception Driven Grasping 

**Title (ZH)**: 嵌入多模态传感的高抓力 gripper 及其驱动感知的强力抓取 

**Authors**: Edoardo Del Bianco, Davide Torielli, Federico Rollo, Damiano Gasperini, Arturo Laurenzi, Lorenzo Baccelliere, Luca Muratore, Marco Roveri, Nikos G. Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.04970)  

**Abstract**: Modern humanoid robots have shown their promising potential for executing various tasks involving the grasping and manipulation of objects using their end-effectors. Nevertheless, in the most of the cases, the grasping and manipulation actions involve low to moderate payload and interaction forces. This is due to limitations often presented by the end-effectors, which can not match their arm-reachable payload, and hence limit the payload that can be grasped and manipulated. In addition, grippers usually do not embed adequate perception in their hardware, and grasping actions are mainly driven by perception sensors installed in the rest of the robot body, frequently affected by occlusions due to the arm motions during the execution of the grasping and manipulation tasks. To address the above, we developed a modular high grasping force gripper equipped with embedded multi-modal perception functionalities. The proposed gripper can generate a grasping force of 110 N in a compact implementation. The high grasping force capability is combined with embedded multi-modal sensing, which includes an eye-in-hand camera, a Time-of-Flight (ToF) distance sensor, an Inertial Measurement Unit (IMU) and an omnidirectional microphone, permitting the implementation of perception-driven grasping functionalities.
We extensively evaluated the grasping force capacity of the gripper by introducing novel payload evaluation metrics that are a function of the robot arm's dynamic motion and gripper thermal states. We also evaluated the embedded multi-modal sensing by performing perception-guided enhanced grasping operations. 

**Abstract (ZH)**: 现代人形机器人展示了通过末端执行器抓取和操作物体执行各种任务的潜在能力。然而，在大多数情况下，抓取和操作动作涉及的负载和交互力较低到中等水平。这主要是由于末端执行器的限制，它们无法匹配手臂可触及的负载，从而限制了可抓取和操作的负载量。此外，夹爪通常在硬件上缺乏足够的感知能力，抓取动作主要由安装在机器人身体其余部分的感知传感器驱动，这些传感器经常受到手臂运动期间执行抓取和操作任务时遮挡的影响。为解决上述问题，我们开发了一款模块化高抓取力夹爪，配备了嵌入式多模态感知功能。所提出的夹爪能够在紧凑的实施中产生110 N的抓取力。该高抓取力能力与嵌入式多模态感知相结合，包括手眼相机、飞行时间（ToF）距离传感器、惯性测量单元（IMU）和全景麦克风，允许实现感知驱动的抓取功能。我们通过引入新的基于机器人手臂动态运动和夹爪热状态的抓取力评价指标，广泛评估了夹爪的抓取力能力，并通过执行感知导向的增强抓取操作来评估嵌入式多模态感知功能。 

---
# A Taxonomy of Self-Handover 

**Title (ZH)**: 自我切换的分类学 

**Authors**: Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.04939)  

**Abstract**: Self-handover, transferring an object between one's own hands, is a common but understudied bimanual action. While it facilitates seamless transitions in complex tasks, the strategies underlying its execution remain largely unexplored. Here, we introduce the first systematic taxonomy of self-handover, derived from manual annotation of over 12 hours of cooking activity performed by 21 participants. Our analysis reveals that self-handover is not merely a passive transition, but a highly coordinated action involving anticipatory adjustments by both hands. As a step toward automated analysis of human manipulation, we further demonstrate the feasibility of classifying self-handover types using a state-of-the-art vision-language model. These findings offer fresh insights into bimanual coordination, underscoring the role of self-handover in enabling smooth task transitions-an ability essential for adaptive dual-arm robotics. 

**Abstract (ZH)**: 自我交接：在双手之间转移物体的共同但研究不足的双手动作及其执行策略的系统分类 

---
# Constrained Gaussian Process Motion Planning via Stein Variational Newton Inference 

**Title (ZH)**: 基于Stein变分牛顿推断的约束高斯过程运动规划 

**Authors**: Jiayun Li, Kay Pompetzki, An Thai Le, Haolei Tong, Jan Peters, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.04936)  

**Abstract**: Gaussian Process Motion Planning (GPMP) is a widely used framework for generating smooth trajectories within a limited compute time--an essential requirement in many robotic applications. However, traditional GPMP approaches often struggle with enforcing hard nonlinear constraints and rely on Maximum a Posteriori (MAP) solutions that disregard the full Bayesian posterior. This limits planning diversity and ultimately hampers decision-making. Recent efforts to integrate Stein Variational Gradient Descent (SVGD) into motion planning have shown promise in handling complex constraints. Nonetheless, these methods still face persistent challenges, such as difficulties in strictly enforcing constraints and inefficiencies when the probabilistic inference problem is poorly conditioned. To address these issues, we propose a novel constrained Stein Variational Gaussian Process Motion Planning (cSGPMP) framework, incorporating a GPMP prior specifically designed for trajectory optimization under hard constraints. Our approach improves the efficiency of particle-based inference while explicitly handling nonlinear constraints. This advancement significantly broadens the applicability of GPMP to motion planning scenarios demanding robust Bayesian inference, strict constraint adherence, and computational efficiency within a limited time. We validate our method on standard benchmarks, achieving an average success rate of 98.57% across 350 planning tasks, significantly outperforming competitive baselines. This demonstrates the ability of our method to discover and use diverse trajectory modes, enhancing flexibility and adaptability in complex environments, and delivering significant improvements over standard baselines without incurring major computational costs. 

**Abstract (ZH)**: 基于Stein变分梯度下降的约束高斯过程运动规划（cSGPMP） 

---
# On Scenario Formalisms for Automated Driving 

**Title (ZH)**: 面向自动驾驶的场景形式化方法研究 

**Authors**: Christian Neurohr, Lukas Westhofen, Tjark Koopmann, Eike Möhlmann, Eckard Böde, Axel Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2504.04868)  

**Abstract**: The concept of scenario and its many qualifications -- specifically logical and abstract scenarios -- have emerged as a foundational element in safeguarding automated driving systems. However, the original linguistic definitions of the different scenario qualifications were often applied ambiguously, leading to a divergence between scenario description languages proposed or standardized in practice and their terminological foundation. This resulted in confusion about the unique features as well as strengths and weaknesses of logical and abstract scenarios. To alleviate this, we give clear linguistic definitions for the scenario qualifications concrete, logical, and abstract scenario and propose generic, unifying formalisms using curves, mappings to sets of curves, and temporal logics, respectively. We demonstrate that these formalisms allow pinpointing strengths and weaknesses precisely by comparing expressiveness, specification complexity, sampling, and monitoring of logical and abstract scenarios. Our work hence enables the practitioner to comprehend the different scenario qualifications and identify a suitable formalism. 

**Abstract (ZH)**: 场景及其多种限定概念——尤其是逻辑和抽象场景——已成为保障自动驾驶系统安全的基础元素。然而，不同场景限定概念的原初语言定义常常被模糊应用，导致实践中提出的或标准化的场景描述语言与其术语基础之间出现分歧。这造成了对逻辑和抽象场景的独特特征及其优势和劣势的混淆。为了解决这一问题，我们明确了具体的、逻辑的和抽象的场景的限定概念，并提出了分别基于曲线、曲线集合的映射和时序逻辑的一般统一形式化方法。我们展示了这些形式化方法可以通过比较表达能力、规范复杂性、抽样和监控逻辑和抽象场景的能力，精确指出其优势和劣势。因此，我们的工作使实践者能够理解不同的场景限定概念，并识别出合适的形式化方法。 

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
# BayesCPF: Enabling Collective Perception in Robot Swarms with Degrading Sensors 

**Title (ZH)**: BayesCPF：在传感器退化的条件下使机器人 Swarm 实现集体感知 

**Authors**: Khai Yi Chin, Carlo Pinciroli  

**Link**: [PDF](https://arxiv.org/pdf/2504.04774)  

**Abstract**: The collective perception problem -- where a group of robots perceives its surroundings and comes to a consensus on an environmental state -- is a fundamental problem in swarm robotics. Past works studying collective perception use either an entire robot swarm with perfect sensing or a swarm with only a handful of malfunctioning members. A related study proposed an algorithm that does account for an entire swarm of unreliable robots but assumes that the sensor faults are known and remain constant over time. To that end, we build on that study by proposing the Bayes Collective Perception Filter (BayesCPF) that enables robots with continuously degrading sensors to accurately estimate the fill ratio -- the rate at which an environmental feature occurs. Our main contribution is the Extended Kalman Filter within the BayesCPF, which helps swarm robots calibrate for their time-varying sensor degradation. We validate our method across different degradation models, initial conditions, and environments in simulated and physical experiments. Our findings show that, regardless of degradation model assumptions, fill ratio estimation using the BayesCPF is competitive to the case if the true sensor accuracy is known, especially when assumptions regarding the model and initial sensor accuracy levels are preserved. 

**Abstract (ZH)**: 集体感知问题——一群机器人感知其周围环境并就环境状态达成一致的过程是群集机器人领域的一个基本问题。现有的集体感知研究要么假设整个机器人群具备完美的传感能力，要么仅考虑少量功能失效的成员。相关研究提出了一种算法来处理整个由不可靠机器人组成的群集，但假设传感器故障是已知且恒定的。在此基础上，我们提出了一种称为贝叶斯集体感知滤波器（BayesCPF）的方法，该方法能够使具有持续退化传感器的机器人准确估计填充率——即环境特征出现的速率。我们的主要贡献是BayesCPF中的扩展Kalman滤波器，它帮助群集机器人校准其时变传感器退化情况。我们在模拟和物理实验中对不同退化模型、初始条件和环境进行了验证。研究结果表明，无论假设何种退化模型，使用BayesCPF进行填充率估计都能在已知真实传感器精度的情况下保持竞争力，尤其是在模型和初始传感器精度水平的假设保持一致的情况下。 

---
# Extended URDF: Accounting for parallel mechanism in robot description 

**Title (ZH)**: 扩展的URDF：考虑并联机构的机器人描述 

**Authors**: Virgile Batto, Ludovic de Matteïs, Nicolas Mansard  

**Link**: [PDF](https://arxiv.org/pdf/2504.04767)  

**Abstract**: Robotic designs played an important role in recent advances by providing powerful robots with complex mechanics. Many recent systems rely on parallel actuation to provide lighter limbs and allow more complex motion. However, these emerging architectures fall outside the scope of most used description formats, leading to difficulties when designing, storing, and sharing the models of these systems. This paper introduces an extension to the widely used Unified Robot Description Format (URDF) to support closed-loop kinematic structures. Our approach relies on augmenting URDF with minimal additional information to allow more efficient modeling of complex robotic systems while maintaining compatibility with existing design and simulation frameworks. This method sets the basic requirement for a description format to handle parallel mechanisms efficiently. We demonstrate the applicability of our approach by providing an open-source collection of parallel robots, along with tools for generating and parsing this extended description format. The proposed extension simplifies robot modeling, reduces redundancy, and improves usability for advanced robotic applications. 

**Abstract (ZH)**: 机器人设计在 recent advances 中发挥了重要作用，通过提供具有复杂机械结构的强大机器人。许多近期系统依赖并行驱动以实现更轻的肢体和更复杂的操作。然而，这些新兴架构超出了大多数使用描述格式的范围，导致在设计、存储和分享这些系统的模型时遇到困难。本文介绍了对广泛使用的统一机器人描述格式（URDF）的一个扩展，以支持闭环运动学结构。我们的方法通过在URDF中添加最少的额外信息来提高复杂机器人系统建模效率，同时保持与现有设计和仿真框架的兼容性。该方法为高效处理并行机构设定了基本要求。我们通过提供一个开源的并行机器人集合以及生成和解析此扩展描述格式的工具，展示了我们方法的应用性。提出的扩展简化了机器人建模，减少了冗余，并提高了高级机器人应用的易用性。 

---
# Tool-as-Interface: Learning Robot Policies from Human Tool Usage through Imitation Learning 

**Title (ZH)**: 工具作为界面：通过模仿学习从人类工具使用中学习机器人策略 

**Authors**: Haonan Chen, Cheng Zhu, Yunzhu Li, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2504.04612)  

**Abstract**: Tool use is critical for enabling robots to perform complex real-world tasks, and leveraging human tool-use data can be instrumental for teaching robots. However, existing data collection methods like teleoperation are slow, prone to control delays, and unsuitable for dynamic tasks. In contrast, human natural data, where humans directly perform tasks with tools, offers natural, unstructured interactions that are both efficient and easy to collect. Building on the insight that humans and robots can share the same tools, we propose a framework to transfer tool-use knowledge from human data to robots. Using two RGB cameras, our method generates 3D reconstruction, applies Gaussian splatting for novel view augmentation, employs segmentation models to extract embodiment-agnostic observations, and leverages task-space tool-action representations to train visuomotor policies. We validate our approach on diverse real-world tasks, including meatball scooping, pan flipping, wine bottle balancing, and other complex tasks. Our method achieves a 71\% higher average success rate compared to diffusion policies trained with teleoperation data and reduces data collection time by 77\%, with some tasks solvable only by our framework. Compared to hand-held gripper, our method cuts data collection time by 41\%. Additionally, our method bridges the embodiment gap, improves robustness to variations in camera viewpoints and robot configurations, and generalizes effectively across objects and spatial setups. 

**Abstract (ZH)**: 基于人类数据的工具使用知识转移框架：用于机器人复杂现实任务的高效学习方法 

---
# Diffusion-Based Approximate MPC: Fast and Consistent Imitation of Multi-Modal Action Distributions 

**Title (ZH)**: 基于扩散的近似MPC：快速且一致地模仿多模态动作分布 

**Authors**: Pau Marquez Julbe, Julian Nubert, Henrik Hose, Sebastian Trimpe, Katherine J. Kuchenbecker  

**Link**: [PDF](https://arxiv.org/pdf/2504.04603)  

**Abstract**: Approximating model predictive control (MPC) using imitation learning (IL) allows for fast control without solving expensive optimization problems online. However, methods that use neural networks in a simple L2-regression setup fail to approximate multi-modal (set-valued) solution distributions caused by local optima found by the numerical solver or non-convex constraints, such as obstacles, significantly limiting the applicability of approximate MPC in practice. We solve this issue by using diffusion models to accurately represent the complete solution distribution (i.e., all modes) at high control rates (more than 1000 Hz). This work shows that diffusion based AMPC significantly outperforms L2-regression-based approximate MPC for multi-modal action distributions. In contrast to most earlier work on IL, we also focus on running the diffusion-based controller at a higher rate and in joint space instead of end-effector space. Additionally, we propose the use of gradient guidance during the denoising process to consistently pick the same mode in closed loop to prevent switching between solutions. We propose using the cost and constraint satisfaction of the original MPC problem during parallel sampling of solutions from the diffusion model to pick a better mode online. We evaluate our method on the fast and accurate control of a 7-DoF robot manipulator both in simulation and on hardware deployed at 250 Hz, achieving a speedup of more than 70 times compared to solving the MPC problem online and also outperforming the numerical optimization (used for training) in success ratio. 

**Abstract (ZH)**: 使用扩散模型进行多模式近似模型预测控制 

---
# B4P: Simultaneous Grasp and Motion Planning for Object Placement via Parallelized Bidirectional Forests and Path Repair 

**Title (ZH)**: B4P: 同时进行物体放置时的抓取与运动规划方法_via_并行双向森林及路径修复_ 

**Authors**: Benjamin H. Leebron, Kejia Ren, Yiting Chen, Kaiyu Hang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04598)  

**Abstract**: Robot pick and place systems have traditionally decoupled grasp, placement, and motion planning to build sequential optimization pipelines with the assumption that the individual components will be able to work together. However, this separation introduces sub-optimality, as grasp choices may limit or even prohibit feasible motions for a robot to reach the target placement pose, particularly in cluttered environments with narrow passages. To this end, we propose a forest-based planning framework to simultaneously find grasp configurations and feasible robot motions that explicitly satisfy downstream placement configurations paired with the selected grasps. Our proposed framework leverages a bidirectional sampling-based approach to build a start forest, rooted at the feasible grasp regions, and a goal forest, rooted at the feasible placement regions, to facilitate the search through randomly explored motions that connect valid pairs of grasp and placement trees. We demonstrate that the framework's inherent parallelism enables superlinear speedup, making it scalable for applications for redundant robot arms (e.g., 7 Degrees of Freedom) to work efficiently in highly cluttered environments. Extensive experiments in simulation demonstrate the robustness and efficiency of the proposed framework in comparison with multiple baselines under diverse scenarios. 

**Abstract (ZH)**: 基于森林的规划框架：同时寻找满足选握态与下游放置态的握持配置和可行机器人运动 

---
# DexTOG: Learning Task-Oriented Dexterous Grasp with Language 

**Title (ZH)**: DexTOG: 学习任务导向的灵巧抓取与语言 

**Authors**: Jieyi Zhang, Wenqiang Xu, Zhenjun Yu, Pengfei Xie, Tutian Tang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04573)  

**Abstract**: This study introduces a novel language-guided diffusion-based learning framework, DexTOG, aimed at advancing the field of task-oriented grasping (TOG) with dexterous hands. Unlike existing methods that mainly focus on 2-finger grippers, this research addresses the complexities of dexterous manipulation, where the system must identify non-unique optimal grasp poses under specific task constraints, cater to multiple valid grasps, and search in a high degree-of-freedom configuration space in grasp planning. The proposed DexTOG includes a diffusion-based grasp pose generation model, DexDiffu, and a data engine to support the DexDiffu. By leveraging DexTOG, we also proposed a new dataset, DexTOG-80K, which was developed using a shadow robot hand to perform various tasks on 80 objects from 5 categories, showcasing the dexterity and multi-tasking capabilities of the robotic hand. This research not only presents a significant leap in dexterous TOG but also provides a comprehensive dataset and simulation validation, setting a new benchmark in robotic manipulation research. 

**Abstract (ZH)**: 基于语言引导扩散学习框架 DexTOG 以推进灵巧手任务导向抓取的研究 

---
# Planning Safety Trajectories with Dual-Phase, Physics-Informed, and Transportation Knowledge-Driven Large Language Models 

**Title (ZH)**: 基于双阶段、物理信息及交通知识驱动的大语言模型规划安全轨迹 

**Authors**: Rui Gan, Pei Li, Keke Long, Bocheng An, Junwei You, Keshu Wu, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2504.04562)  

**Abstract**: Foundation models have demonstrated strong reasoning and generalization capabilities in driving-related tasks, including scene understanding, planning, and control. However, they still face challenges in hallucinations, uncertainty, and long inference latency. While existing foundation models have general knowledge of avoiding collisions, they often lack transportation-specific safety knowledge. To overcome these limitations, we introduce LetsPi, a physics-informed, dual-phase, knowledge-driven framework for safe, human-like trajectory planning. To prevent hallucinations and minimize uncertainty, this hybrid framework integrates Large Language Model (LLM) reasoning with physics-informed social force dynamics. LetsPi leverages the LLM to analyze driving scenes and historical information, providing appropriate parameters and target destinations (goals) for the social force model, which then generates the future trajectory. Moreover, the dual-phase architecture balances reasoning and computational efficiency through its Memory Collection phase and Fast Inference phase. The Memory Collection phase leverages the physics-informed LLM to process and refine planning results through reasoning, reflection, and memory modules, storing safe, high-quality driving experiences in a memory bank. Surrogate safety measures and physics-informed prompt techniques are introduced to enhance the LLM's knowledge of transportation safety and physical force, respectively. The Fast Inference phase extracts similar driving experiences as few-shot examples for new scenarios, while simplifying input-output requirements to enable rapid trajectory planning without compromising safety. Extensive experiments using the HighD dataset demonstrate that LetsPi outperforms baseline models across five safety this http URL PDF for project Github link. 

**Abstract (ZH)**: 基于物理的双阶段知识驱动框架Let们Pi实现安全的人类LIKE轨迹规划 

---
# DexSinGrasp: Learning a Unified Policy for Dexterous Object Singulation and Grasping in Cluttered Environments 

**Title (ZH)**: DexSinGrasp: 学习在拥挤环境中进行灵巧物体分拣和抓取的统一策略 

**Authors**: Lixin Xu, Zixuan Liu, Zhewei Gui, Jingxiang Guo, Zeyu Jiang, Zhixuan Xu, Chongkai Gao, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04516)  

**Abstract**: Grasping objects in cluttered environments remains a fundamental yet challenging problem in robotic manipulation. While prior works have explored learning-based synergies between pushing and grasping for two-fingered grippers, few have leveraged the high degrees of freedom (DoF) in dexterous hands to perform efficient singulation for grasping in cluttered settings. In this work, we introduce DexSinGrasp, a unified policy for dexterous object singulation and grasping. DexSinGrasp enables high-dexterity object singulation to facilitate grasping, significantly improving efficiency and effectiveness in cluttered environments. We incorporate clutter arrangement curriculum learning to enhance success rates and generalization across diverse clutter conditions, while policy distillation enables a deployable vision-based grasping strategy. To evaluate our approach, we introduce a set of cluttered grasping tasks with varying object arrangements and occlusion levels. Experimental results show that our method outperforms baselines in both efficiency and grasping success rate, particularly in dense clutter. Codes, appendix, and videos are available on our project website this https URL. 

**Abstract (ZH)**: 在杂乱环境中的物体抓取仍然是机器人操作中的一个基础但具有挑战性的问题。尽管先前的工作探索了推拿和抓取之间基于学习的协同作用以适用于两指夹持器，但很少有工作利用灵巧手的高自由度来高效地实现杂乱环境中的物体分离与抓取。在本文中，我们引入了DexSinGrasp，一种统一的灵巧物体分离与抓取策略。DexSinGrasp通过高灵巧度的物体分离来促进抓取，显著提高了杂乱环境中的效率和有效性。我们通过杂乱排列课程学习来增强成功率，并在多种杂乱条件下实现泛化，同时策略蒸馏使基于视觉的抓取策略可部署。为了评估我们的方法，我们引入了一组具有不同物体排列和遮挡程度的抓取任务。实验结果表明，我们的方法在效率和抓取成功率方面均优于基线方法，特别是在密集杂乱环境中表现尤为突出。代码、附录和视频可在我们的项目网站上获得：this https URL。 

---
# SELC: Self-Supervised Efficient Local Correspondence Learning for Low Quality Images 

**Title (ZH)**: SELCl：自监督高效局部对应学习在低质量图像中的应用 

**Authors**: Yuqing Wang, Yan Wang, Hailiang Tang, Xiaoji Niu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04497)  

**Abstract**: Accurate and stable feature matching is critical for computer vision tasks, particularly in applications such as Simultaneous Localization and Mapping (SLAM). While recent learning-based feature matching methods have demonstrated promising performance in challenging spatiotemporal scenarios, they still face inherent trade-offs between accuracy and computational efficiency in specific settings. In this paper, we propose a lightweight feature matching network designed to establish sparse, stable, and consistent correspondence between multiple frames. The proposed method eliminates the dependency on manual annotations during training and mitigates feature drift through a hybrid self-supervised paradigm. Extensive experiments validate three key advantages: (1) Our method operates without dependency on external prior knowledge and seamlessly incorporates its hybrid training mechanism into original datasets. (2) Benchmarked against state-of-the-art deep learning-based methods, our approach maintains equivalent computational efficiency at low-resolution scales while achieving a 2-10x improvement in computational efficiency for high-resolution inputs. (3) Comparative evaluations demonstrate that the proposed hybrid self-supervised scheme effectively mitigates feature drift in long-term tracking while maintaining consistent representation across image sequences. 

**Abstract (ZH)**: 准确且稳定的特征匹配对于计算机视觉任务至关重要，特别是在Simultaneous Localization and Mapping (SLAM)等应用中。尽管近年来基于学习的特征匹配方法在复杂的时空场景中展现了令人鼓舞的性能，但在特定情况下它们仍然面临准确性和计算效率之间的固有权衡。在本文中，我们提出了一种轻量级特征匹配网络，旨在建立多帧之间的稀疏、稳定且一致的对应关系。所提出的方法在训练过程中消除对外部先验知识的依赖，并通过混合自监督范式来减轻特征漂移。广泛的实验验证了三个关键优势：(1) 该方法不依赖外部先验知识，并能无缝将其实现机制整合到原始数据集中。(2) 与最先进的基于深度学习的方法相比，我们的方法在低分辨率尺度下保持了相当的计算效率，并且在高分辨率输入下可实现2-10倍的计算效率提升。(3) 比较评估表明，提出的混合自监督方案有效地减轻了长时间跟踪中的特征漂移，并保持了图像序列中的一致表示。 

---
# eKalibr-Stereo: Continuous-Time Spatiotemporal Calibration for Event-Based Stereo Visual Systems 

**Title (ZH)**: eKalibr-立体视觉系统基于事件的连续时空校准 

**Authors**: Shuolong Chen, Xingxing Li, Liu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04451)  

**Abstract**: The bioinspired event camera, distinguished by its exceptional temporal resolution, high dynamic range, and low power consumption, has been extensively studied in recent years for motion estimation, robotic perception, and object detection. In ego-motion estimation, the stereo event camera setup is commonly adopted due to its direct scale perception and depth recovery. For optimal stereo visual fusion, accurate spatiotemporal (extrinsic and temporal) calibration is required. Considering that few stereo visual calibrators orienting to event cameras exist, based on our previous work eKalibr (an event camera intrinsic calibrator), we propose eKalibr-Stereo for accurate spatiotemporal calibration of event-based stereo visual systems. To improve the continuity of grid pattern tracking, building upon the grid pattern recognition method in eKalibr, an additional motion prior-based tracking module is designed in eKalibr-Stereo to track incomplete grid patterns. Based on tracked grid patterns, a two-step initialization procedure is performed to recover initial guesses of piece-wise B-splines and spatiotemporal parameters, followed by a continuous-time batch bundle adjustment to refine the initialized states to optimal ones. The results of extensive real-world experiments show that eKalibr-Stereo can achieve accurate event-based stereo spatiotemporal calibration. The implementation of eKalibr-Stereo is open-sourced at (this https URL) to benefit the research community. 

**Abstract (ZH)**: 受生物启发的事件相机因其卓越的时间分辨率、高动态范围和低功耗，在近期被广泛研究用于运动估计、机器人感知和物体检测。在自我运动估计中，由于其直接尺度感知和深度恢复能力，立体事件相机设置被广泛应用。为了实现最佳立体视觉融合，需要进行精确的空间-时间（外在和时间）校准。鉴于针对事件相机的立体视觉校准工具较少，基于我们之前的工作eKalibr（事件相机内在校准器），我们提出eKalibr-Stereo用于立体事件视觉系统的精确空间-时间校准。为了提高网格图案跟踪的连续性，基于eKalibr中的网格图案识别方法，在eKalibr-Stereo中设计了一个基于运动先验的跟踪模块来跟踪不完整的网格图案。基于跟踪的网格图案，我们执行两步初始化程序来恢复分段B样条和空间-时间参数的初始猜测，然后进行连续时间批量多项式调整以优化初始化状态。广泛的实地实验结果表明，eKalibr-Stereo可以实现精确的事件驱动立体空间-时间校准。eKalibr-Stereo的实现已经开源（https://this-url/），以造福研究界。 

---
# A Convex and Global Solution for the P$n$P Problem in 2D Forward-Looking Sonar 

**Title (ZH)**: 2D 前向声纳 P^nP 问题的凸全局解 

**Authors**: Jiayi Su, Jingyu Qian, Liuqing Yang, Yufan Yuan, Yanbing Fu, Jie Wu, Yan Wei, Fengzhong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04445)  

**Abstract**: The perspective-$n$-point (P$n$P) problem is important for robotic pose estimation. It is well studied for optical cameras, but research is lacking for 2D forward-looking sonar (FLS) in underwater scenarios due to the vastly different imaging principles. In this paper, we demonstrate that, despite the nonlinearity inherent in sonar image formation, the P$n$P problem for 2D FLS can still be effectively addressed within a point-to-line (PtL) 3D registration paradigm through orthographic approximation. The registration is then resolved by a duality-based optimal solver, ensuring the global optimality. For coplanar cases, a null space analysis is conducted to retrieve the solutions from the dual formulation, enabling the methods to be applied to more general cases. Extensive simulations have been conducted to systematically evaluate the performance under different settings. Compared to non-reprojection-optimized state-of-the-art (SOTA) methods, the proposed approach achieves significantly higher precision. When both methods are optimized, ours demonstrates comparable or slightly superior precision. 

**Abstract (ZH)**: 基于点到线（PtL）三维配准范式的2D前向声呐（FLS）视角-n点（P$n$P）问题研究 

---
# Deliberate Planning of 3D Bin Packing on Packing Configuration Trees 

**Title (ZH)**: 3D货物配置树上的故意规划装箱ترتيب التخطيط المقصود ل burial فيrees ثلاثي الأبعاد على نCarouselءات الترتيبثلاثي الأبعاد 

**Authors**: Hang Zhao, Juzhan Xu, Kexiong Yu, Ruizhen Hu, Chenyang Zhu, Kai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04421)  

**Abstract**: Online 3D Bin Packing Problem (3D-BPP) has widespread applications in industrial automation. Existing methods usually solve the problem with limited resolution of spatial discretization, and/or cannot deal with complex practical constraints well. We propose to enhance the practical applicability of online 3D-BPP via learning on a novel hierarchical representation, packing configuration tree (PCT). PCT is a full-fledged description of the state and action space of bin packing which can support packing policy learning based on deep reinforcement learning (DRL). The size of the packing action space is proportional to the number of leaf nodes, making the DRL model easy to train and well-performing even with continuous solution space. We further discover the potential of PCT as tree-based planners in deliberately solving packing problems of industrial significance, including large-scale packing and different variations of BPP setting. A recursive packing method is proposed to decompose large-scale packing into smaller sub-trees while a spatial ensemble mechanism integrates local solutions into global. For different BPP variations with additional decision variables, such as lookahead, buffering, and offline packing, we propose a unified planning framework enabling out-of-the-box problem solving. Extensive evaluations demonstrate that our method outperforms existing online BPP baselines and is versatile in incorporating various practical constraints. The planning process excels across large-scale problems and diverse problem variations. We develop a real-world packing robot for industrial warehousing, with careful designs accounting for constrained placement and transportation stability. Our packing robot operates reliably and efficiently on unprotected pallets at 10 seconds per box. It achieves averagely 19 boxes per pallet with 57.4% space utilization for relatively large-size boxes. 

**Abstract (ZH)**: 在线3D容器打包问题（3D-BPP）在工业自动化中具有广泛的应用。现有的方法通常在有限的空间离散化分辨率下解决问题，或者无法很好地处理复杂的实际约束。我们提出了一种通过学习新提出的分层表示——打包配置树（PCT）来增强在线3D-BPP的实际适用性。PCT是对容器打包状态和动作空间的全面描述，可以支持基于深度强化学习（DRL）的打包策略学习。打包动作空间的大小与叶节点数量成正比，使DRL模型易于训练，即使在连续的解空间中也能表现出色。我们进一步发现PCT作为基于树的规划器的潜在能力，专门解决具有工业意义的打包问题，包括大规模打包和不同时效的BPP设置变体。提出了一种递归打包方法来将大规模打包分解为较小的子树，而空间ensemble机制将局部解集成到全局。对于不同的具有额外决策变量的BPP变体，如前瞻、缓存和离线打包，我们提出了一种统一的规划框架，使问题解决更加方便。广泛评估表明，我们的方法优于现有的在线BPP基线，并且能够灵活地整合各种实际约束。规划过程在大规模问题和多变的问题类型中表现出色。我们为工业仓储开发了一台实际应用的打包机器人，设计考虑了受限放置和运输稳定性。该打包机器人在无保护托盘上以每箱10秒的速度可靠且高效地运行，平均每托盘放置19箱，空间利用率为57.4%。 

---
# Driving-RAG: Driving Scenarios Embedding, Search, and RAG Applications 

**Title (ZH)**: 驾驶场景嵌入、搜索与RAG应用 

**Authors**: Cheng Chang, Jingwei Ge, Jiazhe Guo, Zelin Guo, Binghong Jiang, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04419)  

**Abstract**: Driving scenario data play an increasingly vital role in the development of intelligent vehicles and autonomous driving. Accurate and efficient scenario data search is critical for both online vehicle decision-making and planning, and offline scenario generation and simulations, as it allows for leveraging the scenario experiences to improve the overall performance. Especially with the application of large language models (LLMs) and Retrieval-Augmented-Generation (RAG) systems in autonomous driving, urgent requirements are put forward. In this paper, we introduce the Driving-RAG framework to address the challenges of efficient scenario data embedding, search, and applications for RAG systems. Our embedding model aligns fundamental scenario information and scenario distance metrics in the vector space. The typical scenario sampling method combined with hierarchical navigable small world can perform efficient scenario vector search to achieve high efficiency without sacrificing accuracy. In addition, the reorganization mechanism by graph knowledge enhances the relevance to the prompt scenarios and augment LLM generation. We demonstrate the effectiveness of the proposed framework on typical trajectory planning task for complex interactive scenarios such as ramps and intersections, showcasing its advantages for RAG applications. 

**Abstract (ZH)**: 驾驶场景数据在智能车辆和自动驾驶的发展中扮演着越来越重要的角色。准确高效的场景数据搜索对于在线车辆决策和规划以及离线场景生成和模拟至关重要，因为它允许利用场景经验来提高整体性能。特别是在将大型语言模型（LLMs）和检索增强生成（RAG）系统应用于自动驾驶后，提出了迫切的需求。本文我们介绍了一种Driving-RAG框架，以应对RAG系统中高效场景数据嵌入、搜索和应用的挑战。我们嵌入模型在向量空间中对基本场景信息和场景距离度量进行对齐。结合典型的场景抽样方法和分层可导航小世界图技术可以实现高效的场景向量搜索，从而在不牺牲准确性的前提下提高效率。此外，通过图知识的重组机制增强了对提示场景的相关性，并增强LLM生成。我们在复杂交互场景（如匝道和交叉口）的典型轨迹规划任务中展示了所提出框架的有效性，突显了其在RAG应用中的优势。 

---
# Data Scaling Laws for End-to-End Autonomous Driving 

**Title (ZH)**: 端到端自动驾驶的数据标度律 

**Authors**: Alexander Naumann, Xunjiang Gu, Tolga Dimlioglu, Mariusz Bojarski, Alperen Degirmenci, Alexander Popov, Devansh Bisla, Marco Pavone, Urs Müller, Boris Ivanovic  

**Link**: [PDF](https://arxiv.org/pdf/2504.04338)  

**Abstract**: Autonomous vehicle (AV) stacks have traditionally relied on decomposed approaches, with separate modules handling perception, prediction, and planning. However, this design introduces information loss during inter-module communication, increases computational overhead, and can lead to compounding errors. To address these challenges, recent works have proposed architectures that integrate all components into an end-to-end differentiable model, enabling holistic system optimization. This shift emphasizes data engineering over software integration, offering the potential to enhance system performance by simply scaling up training resources. In this work, we evaluate the performance of a simple end-to-end driving architecture on internal driving datasets ranging in size from 16 to 8192 hours with both open-loop metrics and closed-loop simulations. Specifically, we investigate how much additional training data is needed to achieve a target performance gain, e.g., a 5% improvement in motion prediction accuracy. By understanding the relationship between model performance and training dataset size, we aim to provide insights for data-driven decision-making in autonomous driving development. 

**Abstract (ZH)**: 自主驾驶系统堆栈传统上依赖于分解的方法，每个模块分别进行感知、预测和规划。然而，这种设计在模块间通信时会导致信息丢失，增加计算负担，并可能导致累积错误。为了解决这些问题，近期的研究提出了将所有组件整合到端到端可微分模型中的架构，从而实现整体系统优化。这种转变更注重数据工程而非软件集成，可以通过简单地扩展训练资源来提升系统性能。在本工作中，我们评估了一种简单的端到端驾驶架构在从16小时到8192小时不等内部驾驶数据集上的性能，包括开环度量和闭环仿真。具体地，我们探讨了实现目标性能提升（例如在运动预测准确性方面提高5%）所需的额外训练数据量。通过理解模型性能与训练数据集规模之间的关系，我们旨在为自主驾驶开发中的数据驱动决策提供见解。 

---
# A Self-Supervised Learning Approach with Differentiable Optimization for UAV Trajectory Planning 

**Title (ZH)**: 基于可微优化的自监督学习方法在无人机轨迹规划中的应用 

**Authors**: Yufei Jiang, Yuanzhu Zhan, Harsh Vardhan Gupta, Chinmay Borde, Junyi Geng  

**Link**: [PDF](https://arxiv.org/pdf/2504.04289)  

**Abstract**: While Unmanned Aerial Vehicles (UAVs) have gained significant traction across various fields, path planning in 3D environments remains a critical challenge, particularly under size, weight, and power (SWAP) constraints. Traditional modular planning systems often introduce latency and suboptimal performance due to limited information sharing and local minima issues. End-to-end learning approaches streamline the pipeline by mapping sensory observations directly to actions but require large-scale datasets, face significant sim-to-real gaps, or lack dynamical feasibility. In this paper, we propose a self-supervised UAV trajectory planning pipeline that integrates a learning-based depth perception with differentiable trajectory optimization. A 3D cost map guides UAV behavior without expert demonstrations or human labels. Additionally, we incorporate a neural network-based time allocation strategy to improve the efficiency and optimality. The system thus combines robust learning-based perception with reliable physics-based optimization for improved generalizability and interpretability. Both simulation and real-world experiments validate our approach across various environments, demonstrating its effectiveness and robustness. Our method achieves a 31.33% improvement in position tracking error and 49.37% reduction in control effort compared to the state-of-the-art. 

**Abstract (ZH)**: 基于自我监督的无人机轨迹规划管道：结合基于学习的距离感知和可微轨迹优化 

---
# ORCA: An Open-Source, Reliable, Cost-Effective, Anthropomorphic Robotic Hand for Uninterrupted Dexterous Task Learning 

**Title (ZH)**: ORCA：一种开放源代码、可靠、低成本的人类仿生机器人手，用于不间断的灵巧任务学习 

**Authors**: Clemens C. Christoph, Maximilian Eberlein, Filippos Katsimalis, Arturo Roberti, Aristotelis Sympetheros, Michel R. Vogt, Davide Liconti, Chenyu Yang, Barnabas Gavin Cangan, Ronan J. Hinchet, Robert K. Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.04259)  

**Abstract**: General-purpose robots should possess humanlike dexterity and agility to perform tasks with the same versatility as us. A human-like form factor further enables the use of vast datasets of human-hand interactions. However, the primary bottleneck in dexterous manipulation lies not only in software but arguably even more in hardware. Robotic hands that approach human capabilities are often prohibitively expensive, bulky, or require enterprise-level maintenance, limiting their accessibility for broader research and practical applications. What if the research community could get started with reliable dexterous hands within a day? We present the open-source ORCA hand, a reliable and anthropomorphic 17-DoF tendon-driven robotic hand with integrated tactile sensors, fully assembled in less than eight hours and built for a material cost below 2,000 CHF. We showcase ORCA's key design features such as popping joints, auto-calibration, and tensioning systems that significantly reduce complexity while increasing reliability, accuracy, and robustness. We benchmark the ORCA hand across a variety of tasks, ranging from teleoperation and imitation learning to zero-shot sim-to-real reinforcement learning. Furthermore, we demonstrate its durability, withstanding more than 10,000 continuous operation cycles - equivalent to approximately 20 hours - without hardware failure, the only constraint being the duration of the experiment itself. All design files, source code, and documentation will be available at this https URL. 

**Abstract (ZH)**: 通用机器人应具备人类般的灵巧性和敏捷性，以便像人类一样执行多样化任务。类似人类的手形进一步使人类手部互动的大量数据集的利用成为可能。然而，灵巧操作的主要瓶颈不仅在于软件，也许甚至更多地在于硬件。接近人类能力的手部机器人往往价格昂贵、笨重，或者需要企业级维护，限制了其在更广泛研究和实际应用中的 accessibility。如果研究社区能在一天内开始使用可靠的灵巧手部，该有多好？我们介绍了开源ORCA手部，这是一种可靠且仿人化的17自由度肌腱驱动的灵巧手，集成了触觉传感器，并可在不到八小时内完全组装完成，材料成本低于2000瑞士法郎。我们展示了ORCA的关键设计功能，如弹出关节、自校准系统和张力系统，这些功能大幅降低了复杂性，同时提高了可靠性和精度。我们跨多种任务benchmark了ORCA手部，从遥控操作和模仿学习到零样本模拟到现实的强化学习。此外，我们展示了其耐用性，在超过10,000次连续操作循环（约20小时）中未发生硬件故障，唯一的限制是实验本身的持续时间。所有设计文件、源代码和文档将在以下网址获得。 

---
# An Optimized Density-Based Lane Keeping System for A Cost-Efficient Autonomous Vehicle Platform: AurigaBot V1 

**Title (ZH)**: 一种面向经济型自主车辆平台的优化密度基车道保持系统：AurigaBot V1 

**Authors**: Farbod Younesi, Milad Rabiei, Soroush Keivanfard, Mohsen Sharifi, Marzieh Ghayour Najafabadi, Bahar Moadeli, Arshia Jafari, Mohammad Hossein Moaiyeri  

**Link**: [PDF](https://arxiv.org/pdf/2504.04217)  

**Abstract**: The development of self-driving cars has garnered significant attention from researchers, universities, and industries worldwide. Autonomous vehicles integrate numerous subsystems, including lane tracking, object detection, and vehicle control, which require thorough testing and validation. Scaled-down vehicles offer a cost-effective and accessible platform for experimentation, providing researchers with opportunities to optimize algorithms under constraints of limited computational power. This paper presents a four-wheeled autonomous vehicle platform designed to facilitate research and prototyping in autonomous driving. Key contributions include (1) a novel density-based clustering approach utilizing histogram statistics for landmark tracking, (2) a lateral controller, and (3) the integration of these innovations into a cohesive platform. Additionally, the paper explores object detection through systematic dataset augmentation and introduces an autonomous parking procedure. The results demonstrate the platform's effectiveness in achieving reliable lane tracking under varying lighting conditions, smooth trajectory following, and consistent object detection performance. Though developed for small-scale vehicles, these modular solutions are adaptable for full-scale autonomous systems, offering a versatile and cost-efficient framework for advancing research and industry applications. 

**Abstract (ZH)**: 自动驾驶汽车的发展已引起全球研究人员、大学和工业界的广泛关注。自动驾驶车辆集成了多个子系统，包括车道跟踪、物体检测和车辆控制，这些系统需要进行全面的测试和验证。缩小规模的车辆提供了一种经济且易于访问的实验平台，使研究人员能够在计算资源有限的情况下优化算法。本文介绍了用于促进自动驾驶研究和原型设计的四轮自动驾驶车辆平台。主要贡献包括：（1）一种基于直方图统计的新型基于密度的聚类方法用于地标跟踪，（2）一种横向控制器，以及（3）将这些创新整合到一个协调平台中。此外，本文还探讨了通过系统数据集扩增进行物体检测，并介绍了自动驾驶泊车程序。实验结果表明，该平台在不同光照条件下的车道跟踪可靠性、平滑轨迹跟踪以及一致的物体检测性能方面具有有效性。尽管是为小型车辆开发的，但这些模块化解决方案也可适应全尺寸自动驾驶系统，提供了一个灵活且成本效益高的研究和工业应用框架。 

---
# Passive Luminescent Bellows Mechanism 

**Title (ZH)**: 被动发光 bellows 机制 

**Authors**: Naoto Kikuta, Issei Onda, Kazuki Abe, Masahiro Watanabe, Kenjiro Tadakuma  

**Link**: [PDF](https://arxiv.org/pdf/2504.04194)  

**Abstract**: The use of robots in disaster sites has rapidly expanded, with soft robots attracting particular interest due to their flexibility and adaptability. They can navigate through narrow spaces and debris, facilitating efficient and safe operations. However, low visibility in such environments remains a challenge. This study aims to enhance the visibility of soft robots by developing and evaluating a passive luminescent exible actuator activated by a black light. Using Ecoex mixed with phosphorescent powder, we fabricated an actuator and confirmed its fluorescence phosphorescence and deformation ability. Furthermore the effects of the mixing ratio on optical and mechanical properties were assessed. 

**Abstract (ZH)**: 软体机器人在灾害现场的应用拓展及其被动发光柔性执行机构的研究：黑光激活下的荧光磷光和变形能力评估 

---
# A General Peg-in-Hole Assembly Policy Based on Domain Randomized Reinforcement Learning 

**Title (ZH)**: 基于领域随机化强化学习的通用 peg-in-hole 装配策略 

**Authors**: Xinyu Liu, Aljaz Kramberger, Leon Bodenhagen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04148)  

**Abstract**: Generalization is important for peg-in-hole assembly, a fundamental industrial operation, to adapt to dynamic industrial scenarios and enhance manufacturing efficiency. While prior work has enhanced generalization ability for pose variations, spatial generalization to six degrees of freedom (6-DOF) is less researched, limiting application in real-world scenarios. This paper addresses this limitation by developing a general policy GenPiH using Proximal Policy Optimization(PPO) and dynamic simulation with domain randomization. The policy learning experiment demonstrates the policy's generalization ability with nearly 100\% success insertion across over eight thousand unique hole poses in parallel environments, and sim-to-real validation on a UR10e robot confirms the policy's performance through direct trajectory execution without task-specific tuning. 

**Abstract (ZH)**: 通用性对于 peg-in-hole 装配这一基本工业操作在适应动态工业场景和提升制造效率方面至关重要。尽管先前的工作已经在姿态变化方面增强了通用性，但空间六自由度（6-DOF）的通用性研究较少，限制了其在实际应用场景中的应用。本文通过使用强化学习中的近端策略优化（PPO）和动态仿真结合领域随机化开发了一种通用策略 GenPiH，以解决这一限制。策略学习实验表明，该策略在并行环境中对超过八千种独特孔的姿态具有近 100% 的成功率插入能力，并且通过直接轨迹执行在 UR10e 机器人上的实到虚验证确认了该策略的性能，无需特定任务的调整。 

---
# Mapping at First Sense: A Lightweight Neural Network-Based Indoor Structures Prediction Method for Robot Autonomous Exploration 

**Title (ZH)**: 初级感知映射：一种轻量级神经网络导向的室内结构预测方法用于机器人自主探索 

**Authors**: Haojia Gao, Haohua Que, Kunrong Li, Weihao Shan, Mingkai Liu, Rong Zhao, Lei Mu, Xinghua Yang, Qi Wei, Fei Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04061)  

**Abstract**: Autonomous exploration in unknown environments is a critical challenge in robotics, particularly for applications such as indoor navigation, search and rescue, and service robotics. Traditional exploration strategies, such as frontier-based methods, often struggle to efficiently utilize prior knowledge of structural regularities in indoor spaces. To address this limitation, we propose Mapping at First Sense, a lightweight neural network-based approach that predicts unobserved areas in local maps, thereby enhancing exploration efficiency. The core of our method, SenseMapNet, integrates convolutional and transformerbased architectures to infer occluded regions while maintaining computational efficiency for real-time deployment on resourceconstrained robots. Additionally, we introduce SenseMapDataset, a curated dataset constructed from KTH and HouseExpo environments, which facilitates training and evaluation of neural models for indoor exploration. Experimental results demonstrate that SenseMapNet achieves an SSIM (structural similarity) of 0.78, LPIPS (perceptual quality) of 0.68, and an FID (feature distribution alignment) of 239.79, outperforming conventional methods in map reconstruction quality. Compared to traditional frontier-based exploration, our method reduces exploration time by 46.5% (from 2335.56s to 1248.68s) while maintaining a high coverage rate (88%) and achieving a reconstruction accuracy of 88%. The proposed method represents a promising step toward efficient, learning-driven robotic exploration in structured environments. 

**Abstract (ZH)**: 自主探索未知环境是机器人技术中的一个关键挑战，特别是在室内导航、搜索与救援以及服务机器人等领域应用尤为突出。传统的探索策略，如基于前沿的方法，往往难以有效利用室内空间结构规律的先验知识。为了解决这一局限性，我们提出了一种名为“先行感知制图”的轻量级神经网络方法，该方法能够预测局部地图中的未观测区域，从而提高探索效率。该方法的核心SenseMapNet结合了卷积和基于转换器的架构，能够在保持计算效率的同时推断被遮挡的区域，适用于资源受限机器人上的实时部署。此外，我们还引入了SenseMapDataset数据集，该数据集基于KTH和HouseExpo环境构建，有助于室内探索神经模型的训练与评估。实验结果表明，SenseMapNet在结构相似度(SSIM)方面达到0.78，在感知质量(LPIPS)方面达到0.68，在特征分布对齐(FID)方面达到239.79，相较于传统方法，在图重建质量上表现出色。与传统的基于前沿的探索方法相比，我们的方法将探索时间减少了46.5%(从2335.56秒减少到1248.68秒)，同时保持了88%的高覆盖率，并实现了88%的重建准确率。所提出的方法代表了在结构化环境中实现高效、学习驱动的机器人探索的一个有前景的步骤。 

---
# CORTEX-AVD: CORner Case Testing & EXploration for Autonomous Vehicles Development 

**Title (ZH)**: CORTEX-AVD: 角案例测试与探索在自主车辆开发中的应用 

**Authors**: Gabriel Shimanuki, Alexandre Nascimento, Lucio Vismari, Joao Camargo Jr, Jorge Almeida Jr, Paulo Cugnasca  

**Link**: [PDF](https://arxiv.org/pdf/2504.03989)  

**Abstract**: Autonomous Vehicles (AVs) aim to improve traffic safety and efficiency by reducing human error. However, ensuring AVs reliability and safety is a challenging task when rare, high-risk traffic scenarios are considered. These 'Corner Cases' (CC) scenarios, such as unexpected vehicle maneuvers or sudden pedestrian crossings, must be safely and reliable dealt by AVs during their operations. But they arehard to be efficiently generated. Traditional CC generation relies on costly and risky real-world data acquisition, limiting scalability, and slowing research and development progress. Simulation-based techniques also face challenges, as modeling diverse scenarios and capturing all possible CCs is complex and time-consuming. To address these limitations in CC generation, this research introduces CORTEX-AVD, CORner Case Testing & EXploration for Autonomous Vehicles Development, an open-source framework that integrates the CARLA Simulator and Scenic to automatically generate CC from textual descriptions, increasing the diversity and automation of scenario modeling. Genetic Algorithms (GA) are used to optimize the scenario parameters in six case study scenarios, increasing the occurrence of high-risk events. Unlike previous methods, CORTEX-AVD incorporates a multi-factor fitness function that considers variables such as distance, time, speed, and collision likelihood. Additionally, the study provides a benchmark for comparing GA-based CC generation methods, contributing to a more standardized evaluation of synthetic data generation and scenario assessment. Experimental results demonstrate that the CORTEX-AVD framework significantly increases CC incidence while reducing the proportion of wasted simulations. 

**Abstract (ZH)**: 自动驾驶车辆（AV）的目标是通过减少人为错误来提高交通安全和效率。然而，在考虑罕见的高风险交通场景时，确保AV的可靠性和安全性是一项挑战。这些“边缘案例”（CC）场景，如意外的车辆操作或突然的行人横穿，必须在AV运行过程中被安全且可靠地处理，但这些场景难以高效生成。传统的CC生成依赖于成本高昂且风险高的现实世界数据采集，限制了其扩展性，减缓了研究和开发进度。基于模拟的方法也面临挑战，因为建模多样场景并捕捉所有可能的CC是复杂且耗时的。为解决CC生成的这些局限性，本研究引入了CORTEX-AVD框架，即“边缘案例测试与探索以促进自动驾驶车辆开发”，该框架结合了CARLA模拟器和Scenic，可以从文本描述中自动生成CC，增加了场景建模的多样性和自动化程度。遗传算法（GA）用于在六个案例研究场景中优化场景参数，增加了高风险事件的发生率。与之前的方法不同，CORTEX-AVD结合了一个多因素适应度函数，考虑了距离、时间、速度和碰撞可能性等变量。此外，本研究还提供了基于GA的CC生成方法的基准，有助于更标准化地评估合成数据生成和场景评估。实验结果表明，CORTEX-AVD框架显著增加了CC的发生率，同时减少了无效模拟的比例。 

---
# Bistable SMA-driven engine for pulse-jet locomotion in soft aquatic robots 

**Title (ZH)**: 双稳态SMC驱动发动机在软水下机器人中的脉冲喷气驱动推进 

**Authors**: Graziella Bedenik, Antonio Morales, Supun Pieris, Barbara da Silva, John W. Kurelek, Melissa Greeff, Matthew Robertson  

**Link**: [PDF](https://arxiv.org/pdf/2504.03988)  

**Abstract**: This paper presents the design and experimental validation of a bio-inspired soft aquatic robot, the DilBot, which uses a bistable shape memory alloy-driven engine for pulse-jet locomotion. Drawing inspiration from the efficient swimming mechanisms of box jellyfish, the DilBot incorporates antagonistic shape memory alloy springs encapsulated in silicone insulation to achieve high-power propulsion. The innovative bistable mechanism allows continuous swimming cycles by storing and releasing energy in a controlled manner. Through free-swimming experiments and force characterization tests, we evaluated the DilBot's performance, achieving a peak speed of 158 mm/s and generating a maximum thrust of 5.59 N. This work demonstrates a novel approach to enhancing the efficiency of shape memory alloy actuators in aquatic environments. It presents a promising pathway for future applications in underwater environmental monitoring using robotic swarms. 

**Abstract (ZH)**: 基于 bistable 形状记忆合金驱动的仿生软水下机器人 DilBot 的设计与实验验证 

---
# I Can Hear You Coming: RF Sensing for Uncooperative Satellite Evasion 

**Title (ZH)**: 我能听到你 coming：用于不合作卫星规避的射频 sensing 

**Authors**: Cameron Mehlman, Gregory Falco  

**Link**: [PDF](https://arxiv.org/pdf/2504.03983)  

**Abstract**: Uncooperative satellite engagements with nation-state actors prompts the need for enhanced maneuverability and agility on-orbit. However, robust, autonomous and rapid adversary avoidance capabilities for the space environment is seldom studied. Further, the capability constrained nature of many space vehicles does not afford robust space situational awareness capabilities that can inform maneuvers. We present a "Cat & Mouse" system for training optimal adversary avoidance algorithms using Reinforcement Learning (RL). We propose the novel approach of utilizing intercepted radio frequency communication and dynamic spacecraft state as multi-modal input that could inform paths for a mouse to outmaneuver the cat satellite. Given the current ubiquitous use of RF communications, our proposed system can be applicable to a diverse array of satellites. In addition to providing a comprehensive framework for an RL architecture capable of training performant and adaptive adversary avoidance policies, we also explore several optimization based methods for adversarial avoidance on real-world data obtained from the Space Surveillance Network (SSN) to analyze the benefits and limitations of different avoidance methods. 

**Abstract (ZH)**: 非合作卫星与国家行为体的互动促使轨道机动性和敏捷性的增强。然而，对于空间环境中的 robust、自主且快速的对手规避能力的研究较少。此外，许多航天器的能力限制并不允许提供能够指导机动的空间态势感知能力。我们提出了一种基于增强学习（RL）的“猫与老鼠”系统，用于训练最优对手规避算法。我们提议利用截获的射频通信和动态航天器状态作为多模态输入，以指导老鼠如何规避猫卫星的路径。鉴于当前射频通信的广泛应用，我们的系统适用于各种类型的卫星。除了提供一个全面的用于训练高性能和适应性对手规避策略的RL架构框架外，我们还探讨了几种基于优化方法的实际数据（来自空间监视网络SSN）下的对手规避技术，以分析不同规避方法的利益和局限性。 

---
# Deep Learning-Enhanced Robotic Subretinal Injection with Real-Time Retinal Motion Compensation 

**Title (ZH)**: 基于深度学习的实时视网膜运动补偿视网膜下注射机器人增强技术 

**Authors**: Tianle Wu, Mojtaba Esfandiari, Peiyao Zhang, Russell H. Taylor, Peter Gehlbach, Iulian Iordachita  

**Link**: [PDF](https://arxiv.org/pdf/2504.03939)  

**Abstract**: Subretinal injection is a critical procedure for delivering therapeutic agents to treat retinal diseases such as age-related macular degeneration (AMD). However, retinal motion caused by physiological factors such as respiration and heartbeat significantly impacts precise needle positioning, increasing the risk of retinal pigment epithelium (RPE) damage. This paper presents a fully autonomous robotic subretinal injection system that integrates intraoperative optical coherence tomography (iOCT) imaging and deep learning-based motion prediction to synchronize needle motion with retinal displacement. A Long Short-Term Memory (LSTM) neural network is used to predict internal limiting membrane (ILM) motion, outperforming a Fast Fourier Transform (FFT)-based baseline model. Additionally, a real-time registration framework aligns the needle tip position with the robot's coordinate frame. Then, a dynamic proportional speed control strategy ensures smooth and adaptive needle insertion. Experimental validation in both simulation and ex vivo open-sky porcine eyes demonstrates precise motion synchronization and successful subretinal injections. The experiment achieves a mean tracking error below 16.4 {\mu}m in pre-insertion phases. These results show the potential of AI-driven robotic assistance to improve the safety and accuracy of retinal microsurgery. 

**Abstract (ZH)**: 基于光相干断层成像和深度学习的自主视网膜注射机器人系统：运动预测与精准运动同步 

---
# Energy Efficient Planning for Repetitive Heterogeneous Tasks in Precision Agriculture 

**Title (ZH)**: 精准农业中重复异构任务的能源高效规划 

**Authors**: Shuangyu Xie, Ken Goldberg, Dezhen Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.03938)  

**Abstract**: Robotic weed removal in precision agriculture introduces a repetitive heterogeneous task planning (RHTP) challenge for a mobile manipulator. RHTP has two unique characteristics: 1) an observe-first-and-manipulate-later (OFML) temporal constraint that forces a unique ordering of two different tasks for each target and 2) energy savings from efficient task collocation to minimize unnecessary movements. RHTP can be framed as a stochastic renewal process. According to the Renewal Reward Theorem, the expected energy usage per task cycle is the long-run average. Traditional task and motion planning focuses on feasibility rather than optimality due to the unknown object and obstacle position prior to execution. However, the known target/obstacle distribution in precision agriculture allows minimizing the expected energy usage. For each instance in this renewal process, we first compute task space partition, a novel data structure that computes all possibilities of task multiplexing and its probabilities with robot reachability. Then we propose a region-based set-coverage problem to formulate the RHTP as a mixed-integer nonlinear programming. We have implemented and solved RHTP using Branch-and-Bound solver. Compared to a baseline in simulations based on real field data, the results suggest a significant improvement in path length, number of robot stops, overall energy usage, and number of replans. 

**Abstract (ZH)**: 精准农业中机器人除草任务规划引入了一种重复异构任务规划（RHTP）挑战，为移动 manipulator 提出新的任务规划问题。RHTP 具有两大特性：1) 观察先行和操作后续（OFML）的时间约束，要求每个目标任务按特定顺序执行；2) 通过高效的任务共位减少不必要的移动从而节省能量。RHTP 可以作为随机更新过程进行建模。依据更新报酬定理，每轮任务的预期能耗是长期平均值。传统任务与运动规划主要关注可行性而非最优性，因为执行前目标和障碍物的位置未知。然而，精准农业中已知的目标/障碍物分布允许最大限度地减少预期能耗。对于这个随机更新过程中的每一个实例，首先计算任务空间分区，这是一种新的数据结构，计算所有任务复用的可能性及其概率，并结合机器人的可达性。然后提出基于区域的集合覆盖问题，将RHTP形式化为混合整数非线性规划问题。我们使用分支定界求解器实施和解决了RHTP。与基于实际农田数据的基线在仿真中的比较结果表明，在路径长度、机器人停顿次数、总体能耗和重新规划次数方面有显著改进。 

---
# Reducing the Communication of Distributed Model Predictive Control: Autoencoders and Formation Control 

**Title (ZH)**: 减少分布式模型预测控制中的通信开销：自编码器与 formation 控制 

**Authors**: Torben Schiz, Henrik Ebel  

**Link**: [PDF](https://arxiv.org/pdf/2504.05223)  

**Abstract**: Communication remains a key factor limiting the applicability of distributed model predictive control (DMPC) in realistic settings, despite advances in wireless communication. DMPC schemes can require an overwhelming amount of information exchange between agents as the amount of data depends on the length of the predication horizon, for which some applications require a significant length to formally guarantee nominal asymptotic stability. This work aims to provide an approach to reduce the communication effort of DMPC by reducing the size of the communicated data between agents. Using an autoencoder, the communicated data is reduced by the encoder part of the autoencoder prior to communication and reconstructed by the decoder part upon reception within the distributed optimization algorithm that constitutes the DMPC scheme. The choice of a learning-based reduction method is motivated by structure inherent to the data, which results from the data's connection to solutions of optimal control problems. The approach is implemented and tested at the example of formation control of differential-drive robots, which is challenging for optimization-based control due to the robots' nonholonomic constraints, and which is interesting due to the practical importance of mobile robotics. The applicability of the proposed approach is presented first in form of a simulative analysis showing that the resulting control performance yields a satisfactory accuracy. In particular, the proposed approach outperforms the canonical naive way to reduce communication by reducing the length of the prediction horizon. Moreover, it is shown that numerical experiments conducted on embedded computation hardware, with real distributed computation and wireless communication, work well with the proposed way of reducing communication even in practical scenarios in which full communication fails. 

**Abstract (ZH)**: 基于自编码器的数据压缩方法在分布式模型预测控制中的通信效率提升 

---
# Stereo-LiDAR Fusion by Semi-Global Matching With Discrete Disparity-Matching Cost and Semidensification 

**Title (ZH)**: 基于半全局匹配的离散视差匹配代价与半稀疏化立体-LiDAR融合 

**Authors**: Yasuhiro Yao, Ryoichi Ishikawa, Takeshi Oishi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05148)  

**Abstract**: We present a real-time, non-learning depth estimation method that fuses Light Detection and Ranging (LiDAR) data with stereo camera input. Our approach comprises three key techniques: Semi-Global Matching (SGM) stereo with Discrete Disparity-matching Cost (DDC), semidensification of LiDAR disparity, and a consistency check that combines stereo images and LiDAR data. Each of these components is designed for parallelization on a GPU to realize real-time performance. When it was evaluated on the KITTI dataset, the proposed method achieved an error rate of 2.79\%, outperforming the previous state-of-the-art real-time stereo-LiDAR fusion method, which had an error rate of 3.05\%. Furthermore, we tested the proposed method in various scenarios, including different LiDAR point densities, varying weather conditions, and indoor environments, to demonstrate its high adaptability. We believe that the real-time and non-learning nature of our method makes it highly practical for applications in robotics and automation. 

**Abstract (ZH)**: 我们提出了一种实时、非学习的深度估计方法，该方法结合了Light Detection and Ranging (LiDAR) 数据和立体相机输入。该方法包含三种关键技术：半全局匹配（SGM）立体视觉与离散视差匹配成本（DDC）、LiDAR视差半稠密化以及一个结合立体图像和LiDAR数据的一致性检查。这些组件均设计为可在GPU上并行化以实现实时性能。当在KITTI数据集上进行评估时，所提出的方法实现了2.79%的误差率，优于之前实时立体视觉-LiDAR融合的最佳方法（误差率为3.05%）。此外，我们在不同的LiDAR点密度、不同天气条件以及室内环境中测试了所提出的方法，以展示其高适应性。我们认为，本方法的实时和非学习特性使其在机器人技术和自动化领域具有很高的实用性。 

---
# GAMDTP: Dynamic Trajectory Prediction with Graph Attention Mamba Network 

**Title (ZH)**: GAMDTP：基于图注意力Mamba网络的动力学轨迹预测 

**Authors**: Yunxiang Liu, Hongkuo Niu, Jianlin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04862)  

**Abstract**: Accurate motion prediction of traffic agents is crucial for the safety and stability of autonomous driving systems. In this paper, we introduce GAMDTP, a novel graph attention-based network tailored for dynamic trajectory prediction. Specifically, we fuse the result of self attention and mamba-ssm through a gate mechanism, leveraging the strengths of both to extract features more efficiently and accurately, in each graph convolution layer. GAMDTP encodes the high-definition map(HD map) data and the agents' historical trajectory coordinates and decodes the network's output to generate the final prediction results. Additionally, recent approaches predominantly focus on dynamically fusing historical forecast results and rely on two-stage frameworks including proposal and refinement. To further enhance the performance of the two-stage frameworks we also design a scoring mechanism to evaluate the prediction quality during the proposal and refinement processes. Experiments on the Argoverse dataset demonstrates that GAMDTP achieves state-of-the-art performance, achieving superior accuracy in dynamic trajectory prediction. 

**Abstract (ZH)**: 基于图注意机制的GAMDTP网络在交通代理动态轨迹预测中的应用 

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

**Abstract (ZH)**: 基于语言指令、视觉观测和交互的3D物体操作能力 grounding任务及其数据集和网络模型 

---
# Inverse++: Vision-Centric 3D Semantic Occupancy Prediction Assisted with 3D Object Detection 

**Title (ZH)**: Inverse++: 以视觉为中心的3D语义占有预测辅助以3D物体检测 

**Authors**: Zhenxing Ming, Julie Stephany Berrio, Mao Shan, Stewart Worrall  

**Link**: [PDF](https://arxiv.org/pdf/2504.04732)  

**Abstract**: 3D semantic occupancy prediction aims to forecast detailed geometric and semantic information of the surrounding environment for autonomous vehicles (AVs) using onboard surround-view cameras. Existing methods primarily focus on intricate inner structure module designs to improve model performance, such as efficient feature sampling and aggregation processes or intermediate feature representation formats. In this paper, we explore multitask learning by introducing an additional 3D supervision signal by incorporating an additional 3D object detection auxiliary branch. This extra 3D supervision signal enhances the model's overall performance by strengthening the capability of the intermediate features to capture small dynamic objects in the scene, and these small dynamic objects often include vulnerable road users, i.e. bicycles, motorcycles, and pedestrians, whose detection is crucial for ensuring driving safety in autonomous vehicles. Extensive experiments conducted on the nuScenes datasets, including challenging rainy and nighttime scenarios, showcase that our approach attains state-of-the-art results, achieving an IoU score of 31.73% and a mIoU score of 20.91% and excels at detecting vulnerable road users (VRU). The code will be made available at:this https URL 

**Abstract (ZH)**: 基于车载全景相机的3D语义 occupancy 预测旨在利用自动驾驶车辆上的全景相机，预测周围环境的详细几何和语义信息。现有方法主要集中在精细的内部结构模块设计以提高模型性能，例如高效的特征采样和聚合过程或中间特征表示格式。在本文中，我们通过引入附加的3D监督信号，探索多任务学习，该附加信号通过结合一个额外的3D物体检测辅助分支来实现。该附加的3D监督信号通过增强中间特征捕捉场景中小动态物体的能力而增强了模型的整体性能，这些小动态物体通常包括弱势道路使用者，即自行车、摩托车和行人，其检测对于确保自动驾驶车辆的驾驶安全至关重要。在nuScenes数据集上的广泛实验，包括恶劣的雨天和夜间场景，展示了我们的方法取得了最先进的成果，实现了IoU分数为31.73%和mIoU分数为20.91%，并擅长检测弱势道路使用者（VRU）。代码将在该链接处提供：this https URL。 

---
# Modeling, Translation, and Analysis of Different examples using Simulink, Stateflow, SpaceEx, and FlowStar 

**Title (ZH)**: Simulink、Stateflow、SpaceEx和FlowStar的不同示例建模、翻译与分析 

**Authors**: Yogesh Gajula, Ravi Varma Lingala  

**Link**: [PDF](https://arxiv.org/pdf/2504.04638)  

**Abstract**: This report details the translation and testing of multiple benchmarks, including the Six Vehicle Platoon, Two Bouncing Ball, Three Tank System, and Four-Dimensional Linear Switching, which represent continuous and hybrid systems. These benchmarks were gathered from past instances involving diverse verification tools such as SpaceEx, Flow*, HyST, MATLAB-Simulink, Stateflow, etc. They cover a range of systems modeled as hybrid automata, providing a comprehensive set for analysis and evaluation. Initially, we created models for all four systems using various suitable tools. Subsequently, these models were converted to the SpaceEx format and then translated into different formats compatible with various verification tools. Adapting our approach to the dynamic characteristics of each system, we performed reachability analysis using the respective verification tools. 

**Abstract (ZH)**: 本报告详细介绍了对六辆车队、两颗滚动球、三个坦克系统和四维线性切换等多个基准的翻译和测试，这些基准代表了连续和混合系统。这些基准来源于以往涉及多种验证工具（如SpaceEx、Flow*、HyST、MATLAB-Simulink、Stateflow等）的应用实例，涵盖了作为混合自动机建模的各类系统，提供了一套全面的分析和评估集。首先，我们使用各种合适的工具为所有四个系统创建了模型。随后，将这些模型转换为SpaceEx格式，并翻译成各种验证工具兼容的不同格式。根据每个系统的动态特性，我们使用相应的验证工具进行了可达性分析。 

---
# Nonlinear Robust Optimization for Planning and Control 

**Title (ZH)**: 非线性鲁棒优化在规划与控制中的应用 

**Authors**: Arshiya Taj Abdul, Augustinos D. Saravanos, Evangelos A. Theodorou  

**Link**: [PDF](https://arxiv.org/pdf/2504.04605)  

**Abstract**: This paper presents a novel robust trajectory optimization method for constrained nonlinear dynamical systems subject to unknown bounded disturbances. In particular, we seek optimal control policies that remain robustly feasible with respect to all possible realizations of the disturbances within prescribed uncertainty sets. To address this problem, we introduce a bi-level optimization algorithm. The outer level employs a trust-region successive convexification approach which relies on linearizing the nonlinear dynamics and robust constraints. The inner level involves solving the resulting linearized robust optimization problems, for which we derive tractable convex reformulations and present an Augmented Lagrangian method for efficiently solving them. To further enhance the robustness of our methodology on nonlinear systems, we also illustrate that potential linearization errors can be effectively modeled as unknown disturbances as well. Simulation results verify the applicability of our approach in controlling nonlinear systems in a robust manner under unknown disturbances. The promise of effectively handling approximation errors in such successive linearization schemes from a robust optimization perspective is also highlighted. 

**Abstract (ZH)**: 一种用于受未知有界干扰约束非线性动力学系统的新型鲁棒轨迹优化方法 

---
# Modeling of AUV Dynamics with Limited Resources: Efficient Online Learning Using Uncertainty 

**Title (ZH)**: 基于有限资源的自主 underwater 车辆动力学建模：利用不确定性进行高效在线学习 

**Authors**: Michal Tešnar, Bilal Wehbe, Matias Valdenegro-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2504.04583)  

**Abstract**: Machine learning proves effective in constructing dynamics models from data, especially for underwater vehicles. Continuous refinement of these models using incoming data streams, however, often requires storage of an overwhelming amount of redundant data. This work investigates the use of uncertainty in the selection of data points to rehearse in online learning when storage capacity is constrained. The models are learned using an ensemble of multilayer perceptrons as they perform well at predicting epistemic uncertainty. We present three novel approaches: the Threshold method, which excludes samples with uncertainty below a specified threshold, the Greedy method, designed to maximize uncertainty among the stored points, and Threshold-Greedy, which combines the previous two approaches. The methods are assessed on data collected by an underwater vehicle Dagon. Comparison with baselines reveals that the Threshold exhibits enhanced stability throughout the learning process and also yields a model with the least cumulative testing loss. We also conducted detailed analyses on the impact of model parameters and storage size on the performance of the models, as well as a comparison of three different uncertainty estimation methods. 

**Abstract (ZH)**: 机器学习在从数据构建水下车辆动力学模型中证明有效，然而，使用有限存储能力下的连续数据流细化这些模型往往需要存储大量冗余数据。本工作研究在存储能力受限时，在在线学习中使用不确定性选择数据点进行重新训练的方法。模型使用多层感知机集成进行学习，因为它们在预测认识不确定性方面表现良好。我们提出了三种新颖的方法：阈值方法（排除低于指定阈值不确定性的样本）、贪心方法（旨在最大化存储点中的不确定性），以及结合前两者的方法（阈值-贪心）。这些方法在由水下车辆Dagon采集的数据上进行评估。与基线方法的比较表明，阈值方法在整个学习过程中表现出增强的稳定性，并且产生的模型具有最小的累积测试损失。我们还详细分析了模型参数和存储大小对模型性能的影响，并比较了三种不同的不确定性估计方法。 

---
# Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG and Symbolic Verification 

**Title (ZH)**: 基于知识图谱-RAG的层次规划及其符号验证方法 

**Authors**: Cristina Cornelio, Flavio Petruzzellis, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2504.04578)  

**Abstract**: Large Language Models (LLMs) have shown promise as robotic planners but often struggle with long-horizon and complex tasks, especially in specialized environments requiring external knowledge. While hierarchical planning and Retrieval-Augmented Generation (RAG) address some of these challenges, they remain insufficient on their own and a deeper integration is required for achieving more reliable systems. To this end, we propose a neuro-symbolic approach that enhances LLMs-based planners with Knowledge Graph-based RAG for hierarchical plan generation. This method decomposes complex tasks into manageable subtasks, further expanded into executable atomic action sequences. To ensure formal correctness and proper decomposition, we integrate a Symbolic Validator, which also functions as a failure detector by aligning expected and observed world states. Our evaluation against baseline methods demonstrates the consistent significant advantages of integrating hierarchical planning, symbolic verification, and RAG across tasks of varying complexity and different LLMs. Additionally, our experimental setup and novel metrics not only validate our approach for complex planning but also serve as a tool for assessing LLMs' reasoning and compositional capabilities. 

**Abstract (ZH)**: 基于知识图谱增强的神经符号规划方法：结合层级规划和检索增强生成 

---
# Advancing Egocentric Video Question Answering with Multimodal Large Language Models 

**Title (ZH)**: 基于多模态大型语言模型的以自我为中心的视频问答研究 

**Authors**: Alkesh Patel, Vibhav Chitalia, Yinfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04550)  

**Abstract**: Egocentric Video Question Answering (QA) requires models to handle long-horizon temporal reasoning, first-person perspectives, and specialized challenges like frequent camera movement. This paper systematically evaluates both proprietary and open-source Multimodal Large Language Models (MLLMs) on QaEgo4Dv2 - a refined dataset of egocentric videos derived from QaEgo4D. Four popular MLLMs (GPT-4o, Gemini-1.5-Pro, Video-LLaVa-7B and Qwen2-VL-7B-Instruct) are assessed using zero-shot and fine-tuned approaches for both OpenQA and CloseQA settings. We introduce QaEgo4Dv2 to mitigate annotation noise in QaEgo4D, enabling more reliable comparison. Our results show that fine-tuned Video-LLaVa-7B and Qwen2-VL-7B-Instruct achieve new state-of-the-art performance, surpassing previous benchmarks by up to +2.6% ROUGE/METEOR (for OpenQA) and +13% accuracy (for CloseQA). We also present a thorough error analysis, indicating the model's difficulty in spatial reasoning and fine-grained object recognition - key areas for future improvement. 

**Abstract (ZH)**: 自视点视频问答要求模型处理长时序推理、第一人称视角以及频繁的摄像机运动等专门挑战。本文系统性地评估了私有和开源多模态大型语言模型（MLLMs）在QaEgo4Dv2上的性能，该数据集是基于QaEgo4D精选而来。四种流行的MLLMs（GPT-4o、Gemini-1.5-Pro、Video-LLaVa-7B和Qwen2-VL-7B-Instruct）在开放式和封闭式问答设置中分别采用零样本和微调方法进行评估。我们引入QaEgo4Dv2以减轻QaEgo4D中的注释噪声，从而实现更可靠地比较。结果显示，微调后的Video-LLaVa-7B和Qwen2-VL-7B-Instruct在开放式和封闭式问答设置中分别取得了新的最佳性能，分别比前一基准提高了高达2.6%的ROUGE/METEOR和13%的准确率。此外，我们还进行了详细错误分析，指出了模型在空间推理和细粒度物体识别等方面的困难，这是未来需要改进的关键领域。 

---
# The Mediating Effects of Emotions on Trust through Risk Perception and System Performance in Automated Driving 

**Title (ZH)**: 情绪通过风险感知和系统性能在自动驾驶中对信任的中介效应 

**Authors**: Lilit Avetisyan, Emmanuel Abolarin, Vanik Zakarian, X. Jessie Yang, Feng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.04508)  

**Abstract**: Trust in automated vehicles (AVs) has traditionally been explored through a cognitive lens, but growing evidence highlights the significant role emotions play in shaping trust. This study investigates how risk perception and AV performance (error vs. no error) influence emotional responses and trust in AVs, using mediation analysis to examine the indirect effects of emotions. In this study, 70 participants (42 male, 28 female) watched real-life recorded videos of AVs operating with or without errors, coupled with varying levels of risk information (high, low, or none). They reported their anticipated emotional responses using 19 discrete emotion items, and trust was assessed through dispositional, learned, and situational trust measures. Factor analysis identified four key emotional components, namely hostility, confidence, anxiety, and loneliness, that were influenced by risk perception and AV performance. The linear mixed model showed that risk perception was not a significant predictor of trust, while performance and individual differences were. Mediation analysis revealed that confidence was a strong positive mediator, while hostile and anxious emotions negatively impacted trust. However, lonely emotions did not significantly mediate the relationship between AV performance and trust. The results show that real-time AV behavior is more influential on trust than pre-existing risk perceptions, indicating trust in AVs might be more experience-based than shaped by prior beliefs. Our findings also underscore the importance of fostering positive emotional responses for trust calibration, which has important implications for user experience design in automated driving. 

**Abstract (ZH)**: 自动化车辆中信任的情感作用：风险感知和车辆性能的影响 

---
# Nonlinear Observer Design for Landmark-Inertial Simultaneous Localization and Mapping 

**Title (ZH)**: 地标-惯性同时定位与建图的非线性观测器设计 

**Authors**: Mouaad Boughellaba, Soulaimane Berkane, Abdelhamid Tayebi  

**Link**: [PDF](https://arxiv.org/pdf/2504.04239)  

**Abstract**: This paper addresses the problem of Simultaneous Localization and Mapping (SLAM) for rigid body systems in three-dimensional space. We introduce a new matrix Lie group SE_{3+n}(3), whose elements are composed of the pose, gravity, linear velocity and landmark positions, and propose an almost globally asymptotically stable nonlinear geometric observer that integrates Inertial Measurement Unit (IMU) data with landmark measurements. The proposed observer estimates the pose and map up to a constant position and a constant rotation about the gravity direction. Numerical simulations are provided to validate the performance and effectiveness of the proposed observer, demonstrating its potential for robust SLAM applications. 

**Abstract (ZH)**: 这篇论文解决了三维空间中刚体系统的同时定位与建图（SLAM）问题。我们引入了一个新的矩阵李群SE_{3+n}(3)，其元素由姿态、重力、线性速度和地标位置组成，并提出了一种几乎全局渐近稳定的非线性几何观测器，该观测器将惯性测量单元（IMU）数据与地标测量数据进行整合。所提出的观测器估计姿态和地图，相对于重力方向有一个固定的平移和旋转。提供了数值仿真来验证所提观测器的性能和有效性，展示了其在鲁棒SLAM应用中的潜力。 

---
# GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill 

**Title (ZH)**: GROVE: 一种通用的开放式词汇物理技能学习奖励机制 

**Authors**: Jieming Cui, Tengyu Liu, Ziyu Meng, Jiale Yu, Ran Song, Wei Zhang, Yixin Zhu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04191)  

**Abstract**: Learning open-vocabulary physical skills for simulated agents presents a significant challenge in artificial intelligence. Current reinforcement learning approaches face critical limitations: manually designed rewards lack scalability across diverse tasks, while demonstration-based methods struggle to generalize beyond their training distribution. We introduce GROVE, a generalized reward framework that enables open-vocabulary physical skill learning without manual engineering or task-specific demonstrations. Our key insight is that Large Language Models(LLMs) and Vision Language Models(VLMs) provide complementary guidance -- LLMs generate precise physical constraints capturing task requirements, while VLMs evaluate motion semantics and naturalness. Through an iterative design process, VLM-based feedback continuously refines LLM-generated constraints, creating a self-improving reward system. To bridge the domain gap between simulation and natural images, we develop Pose2CLIP, a lightweight mapper that efficiently projects agent poses directly into semantic feature space without computationally expensive rendering. Extensive experiments across diverse embodiments and learning paradigms demonstrate GROVE's effectiveness, achieving 22.2% higher motion naturalness and 25.7% better task completion scores while training 8.4x faster than previous methods. These results establish a new foundation for scalable physical skill acquisition in simulated environments. 

**Abstract (ZH)**: 基于通用词表的物理技能学习对模拟代理而言在人工智能中构成重大挑战。当前的强化学习方法面临关键限制：手工设计的奖励在不同任务间缺乏可扩展性，而基于示范的方法难以在训练分布之外泛化。我们引入了GROVE，一种通用奖励框架，该框架能够在无需手工工程或特定任务示范的情况下学习基于通用词表的物理技能。我们的关键洞察是，大型语言模型（LLMs）和视觉语言模型（VLMs）提供了互补的指导——LLMs生成精确的物理约束来捕捉任务需求，而VLMs评估运动语义和自然性。通过迭代设计过程，基于VLM的反馈不断细化LLM生成的约束，形成自我改进的奖励系统。为缩小模拟环境与自然图像之间的域差距，我们开发了Pose2CLIP，一种轻量级的映射器，能够高效地将代理姿态直接映射至语义特征空间，而不进行计算成本高的渲染。通过对多样化的实体和学习范式的广泛实验，GROVE的有效性得以验证，其在运动自然性和任务完成分数上分别提高了22.2%和25.7%，同时训练速度比之前的方法快8.4倍。这些结果为模拟环境中的可扩展物理技能获取奠定了新的基础。 

---
# Learning about the Physical World through Analytic Concepts 

**Title (ZH)**: 通过分析概念学习物质世界 

**Authors**: Jianhua Sun, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04170)  

**Abstract**: Reviewing the progress in artificial intelligence over the past decade, various significant advances (e.g. object detection, image generation, large language models) have enabled AI systems to produce more semantically meaningful outputs and achieve widespread adoption in internet scenarios. Nevertheless, AI systems still struggle when it comes to understanding and interacting with the physical world. This reveals an important issue: relying solely on semantic-level concepts learned from internet data (e.g. texts, images) to understand the physical world is far from sufficient -- machine intelligence currently lacks an effective way to learn about the physical world. This research introduces the idea of analytic concept -- representing the concepts related to the physical world through programs of mathematical procedures, providing machine intelligence a portal to perceive, reason about, and interact with the physical world. Except for detailing the design philosophy and providing guidelines for the application of analytic concepts, this research also introduce about the infrastructure that has been built around analytic concepts. I aim for my research to contribute to addressing these questions: What is a proper abstraction of general concepts in the physical world for machine intelligence? How to systematically integrate structured priors with neural networks to constrain AI systems to comply with physical laws? 

**Abstract (ZH)**: 过去十年人工 intelligence 进展回顾：从语义级概念到物理世界的分析概念——机器对物理世界的理解与交互 

---
# Risk-Aware Robot Control in Dynamic Environments Using Belief Control Barrier Functions 

**Title (ZH)**: 使用信念控制屏障函数的动态环境下具风险意识的机器人控制 

**Authors**: Shaohang Han, Matti Vahs, Jana Tumova  

**Link**: [PDF](https://arxiv.org/pdf/2504.04097)  

**Abstract**: Ensuring safety for autonomous robots operating in dynamic environments can be challenging due to factors such as unmodeled dynamics, noisy sensor measurements, and partial observability. To account for these limitations, it is common to maintain a belief distribution over the true state. This belief could be a non-parametric, sample-based representation to capture uncertainty more flexibly. In this paper, we propose a novel form of Belief Control Barrier Functions (BCBFs) specifically designed to ensure safety in dynamic environments under stochastic dynamics and a sample-based belief about the environment state. Our approach incorporates provable concentration bounds on tail risk measures into BCBFs, effectively addressing possible multimodal and skewed belief distributions represented by samples. Moreover, the proposed method demonstrates robustness against distributional shifts up to a predefined bound. We validate the effectiveness and real-time performance (approximately 1kHz) of the proposed method through two simulated underwater robotic applications: object tracking and dynamic collision avoidance. 

**Abstract (ZH)**: 确保自主机器人在动态环境中操作的安全性因未建模的动力学、嘈杂的传感器测量和部分可观测性等因素而具有挑战性。为了应对这些限制，通常需要维护对真实状态的信念分布。这种信念可以是非参数的样本基表示，以更灵活地捕捉不确定性。在本文中，我们提出了一种新型的信念控制屏障函数（Belief Control Barrier Functions，BCBFs），专门设计用于在随机动力学和基于样本的环境状态信念下确保动态环境中的安全性。我们的方法将可证明的尾部风险度量的集中界引入到BCBFs中，有效地解决了由样本表示的可能的多模态和偏斜信念分布。此外，提出的方法在预定义的界内表现出对分布偏移的鲁棒性。我们通过两个模拟的水下机器人应用（物体跟踪和动态避碰）验证了所提出方法的有效性和实时性能（约1kHz）。 

---
# ADAPT: Actively Discovering and Adapting to Preferences for any Task 

**Title (ZH)**: ADAPT: 主动发现和适应任何任务的偏好 

**Authors**: Maithili Patel, Xavier Puig, Ruta Desai, Roozbeh Mottaghi, Sonia Chernova, Joanne Truong, Akshara Rai  

**Link**: [PDF](https://arxiv.org/pdf/2504.04040)  

**Abstract**: Assistive agents should be able to perform under-specified long-horizon tasks while respecting user preferences. We introduce Actively Discovering and Adapting to Preferences for any Task (ADAPT) -- a benchmark designed to evaluate agents' ability to adhere to user preferences across various household tasks through active questioning. Next, we propose Reflection-DPO, a novel training approach for adapting large language models (LLMs) to the task of active questioning. Reflection-DPO finetunes a 'student' LLM to follow the actions of a privileged 'teacher' LLM, and optionally ask a question to gather necessary information to better predict the teacher action. We find that prior approaches that use state-of-the-art LLMs fail to sufficiently follow user preferences in ADAPT due to insufficient questioning and poor adherence to elicited preferences. In contrast, Reflection-DPO achieves a higher rate of satisfying user preferences, outperforming a zero-shot chain-of-thought baseline by 6.1% on unseen users. 

**Abstract (ZH)**: 助手中应能够在尊重用户偏好的同时执行未指定的长期任务。我们引入了Actively Discovering and Adapting to Preferences for any Task (ADAPT)——一个用于评估代理在通过主动提问的方式适应各种家务任务时遵循用户偏好的基准。随后，我们提出了一种新的训练方法——反思-DPO，用于将大规模语言模型（LLMs）适应主动提问的任务。反思-DPO将一个“学生”LLM微调为遵循“教师”LLM的动作，并可选地提问以收集必要信息以更准确地预测教师行为。我们发现，以前使用最先进的LLM的方法在ADAPT中未能充分遵循用户偏好，因为提问不足且对引发的偏好遵守不良。相比之下，反思-DPO能够在未见过的用户上将遵循用户偏好的成功率提高6.1%，超过了零shot思维链基线。 

---
# WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments 

**Title (ZH)**: WildGS-SLAM: 动态环境中的单目高斯点云SLAM 

**Authors**: Jianhao Zheng, Zihan Zhu, Valentin Bieri, Marc Pollefeys, Songyou Peng, Iro Armeni  

**Link**: [PDF](https://arxiv.org/pdf/2504.03886)  

**Abstract**: We present WildGS-SLAM, a robust and efficient monocular RGB SLAM system designed to handle dynamic environments by leveraging uncertainty-aware geometric mapping. Unlike traditional SLAM systems, which assume static scenes, our approach integrates depth and uncertainty information to enhance tracking, mapping, and rendering performance in the presence of moving objects. We introduce an uncertainty map, predicted by a shallow multi-layer perceptron and DINOv2 features, to guide dynamic object removal during both tracking and mapping. This uncertainty map enhances dense bundle adjustment and Gaussian map optimization, improving reconstruction accuracy. Our system is evaluated on multiple datasets and demonstrates artifact-free view synthesis. Results showcase WildGS-SLAM's superior performance in dynamic environments compared to state-of-the-art methods. 

**Abstract (ZH)**: WildGS-SLAM：一种通过利用不确定性意识几何映射来处理动态环境的鲁棒高效单目RGB SLAM系统 

---
# Hierarchically Encapsulated Representation for Protocol Design in Self-Driving Labs 

**Title (ZH)**: 自驾驶实验室中协议设计的分层封装表示 

**Authors**: Yu-Zhe Shi, Mingchen Liu, Fanxu Meng, Qiao Xu, Zhangqian Bi, Kun He, Lecheng Ruan, Qining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03810)  

**Abstract**: Self-driving laboratories have begun to replace human experimenters in performing single experimental skills or predetermined experimental protocols. However, as the pace of idea iteration in scientific research has been intensified by Artificial Intelligence, the demand for rapid design of new protocols for new discoveries become evident. Efforts to automate protocol design have been initiated, but the capabilities of knowledge-based machine designers, such as Large Language Models, have not been fully elicited, probably for the absence of a systematic representation of experimental knowledge, as opposed to isolated, flatten pieces of information. To tackle this issue, we propose a multi-faceted, multi-scale representation, where instance actions, generalized operations, and product flow models are hierarchically encapsulated using Domain-Specific Languages. We further develop a data-driven algorithm based on non-parametric modeling that autonomously customizes these representations for specific domains. The proposed representation is equipped with various machine designers to manage protocol design tasks, including planning, modification, and adjustment. The results demonstrate that the proposed method could effectively complement Large Language Models in the protocol design process, serving as an auxiliary module in the realm of machine-assisted scientific exploration. 

**Abstract (ZH)**: 自驾驶实验室已经开始用以执行单个实验技能或预设的实验协议代替人类实验员。然而，随着人工智能加速了科学研究中的创意迭代，为新发现快速设计新协议的需求变得明显。虽然已经开始尝试自动化协议设计，但基于知识的机器设计师，如大型语言模型的能力尚未完全发挥，可能是因为缺乏对实验知识系统化的表示，而只是孤立的扁平化信息。为解决这一问题，我们提出了一种多层次、多尺度的表示方法，其中实例动作、泛化的操作和产品流程模型通过领域特定语言进行分层封装。我们进一步开发了一种基于非参数建模的数据驱动算法，能够自主适配这些表示以特定领域。所提出的表示方法配备了各种机器设计师，用于管理协议设计任务，包括规划、修改和调整。实验结果表明，所提出的方法可以在协议设计过程中有效补充大型语言模型，作为一种辅助模块存在于机器辅助的科学探索领域。 

---
# Optimal Sensor Placement Using Combinations of Hybrid Measurements for Source Localization 

**Title (ZH)**: 基于混合测量组合的源定位传感器优化布设 

**Authors**: Kang Tang, Sheng Xu, Yuqi Yang, He Kong, Yongsheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03769)  

**Abstract**: This paper focuses on static source localization employing different combinations of measurements, including time-difference-of-arrival (TDOA), received-signal-strength (RSS), angle-of-arrival (AOA), and time-of-arrival (TOA) measurements. Since sensor-source geometry significantly impacts localization accuracy, the strategies of optimal sensor placement are proposed systematically using combinations of hybrid measurements. Firstly, the relationship between sensor placement and source estimation accuracy is formulated by a derived Cramér-Rao bound (CRB). Secondly, the A-optimality criterion, i.e., minimizing the trace of the CRB, is selected to calculate the smallest reachable estimation mean-squared-error (MSE) in a unified manner. Thirdly, the optimal sensor placement strategies are developed to achieve the optimal estimation bound. Specifically, the specific constraints of the optimal geometries deduced by specific measurement, i.e., TDOA, AOA, RSS, and TOA, are found and discussed theoretically. Finally, the new findings are verified by simulation studies. 

**Abstract (ZH)**: 本文聚焦于利用不同组合的测量数据进行静态源定位，包括时间到达差（TDOA）、接收信号强度（RSS）、到达角（AOA）和到达时间（TOA）测量。由于传感器-源几何结构显著影响定位精度，提出了系统性的最优传感器布局策略，利用组合混和测量数据。首先，通过推导出的克拉默-拉奥下界（CRB）制定了传感器布局与源估计精度之间的关系。其次，选择了A-最优标准，即最小化CRB的迹，以此统一计算最小可达到的均方误差（MSE）。第三，发展了最优传感器布局策略以达到最优估计界限。具体而言，探讨了特定测量数据，即TDOA、AOA、RSS和TOA，所推导出的最佳几何结构的具体约束条件。最后，通过仿真研究验证了新发现。 

---
# A Geometric Approach For Pose and Velocity Estimation Using IMU and Inertial/Body-Frame Measurements 

**Title (ZH)**: 基于IMU和体帧测量的几何方法用于姿态和速度估计 

**Authors**: Sifeddine Benahmed, Soulaimane Berkane, Tarek Hamel  

**Link**: [PDF](https://arxiv.org/pdf/2504.03764)  

**Abstract**: This paper addresses accurate pose estimation (position, velocity, and orientation) for a rigid body using a combination of generic inertial-frame and/or body-frame measurements along with an Inertial Measurement Unit (IMU). By embedding the original state space, $\so \times \R^3 \times \R^3$, within the higher-dimensional Lie group $\sefive$, we reformulate the vehicle dynamics and outputs within a structured, geometric framework. In particular, this embedding enables a decoupling of the resulting geometric error dynamics: the translational error dynamics follow a structure similar to the error dynamics of a continuous-time Kalman filter, which allows for a time-varying gain design using the Riccati equation. Under the condition of uniform observability, we establish that the proposed observer design on $\sefive$ guarantees almost global asymptotic stability. We validate the approach in simulations for two practical scenarios: stereo-aided inertial navigation systems (INS) and GPS-aided INS. The proposed method significantly simplifies the design of nonlinear geometric observers for INS, providing a generalized and robust approach to state estimation. 

**Abstract (ZH)**: 基于广义惯性框架和/或体框架测量及惯性测量单元的刚体精确姿态估计 

---
# EDRF: Enhanced Driving Risk Field Based on Multimodal Trajectory Prediction and Its Applications 

**Title (ZH)**: 基于多模态轨迹预测的增强驾驶风险场及其应用 

**Authors**: Junkai Jiang, Zeyu Han, Yuning Wang, Mengchi Cai, Qingwen Meng, Qing Xu, Jianqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2410.14996)  

**Abstract**: Driving risk assessment is crucial for both autonomous vehicles and human-driven vehicles. The driving risk can be quantified as the product of the probability that an event (such as collision) will occur and the consequence of that event. However, the probability of events occurring is often difficult to predict due to the uncertainty of drivers' or vehicles' behavior. Traditional methods generally employ kinematic-based approaches to predict the future trajectories of entities, which often yield unrealistic prediction results. In this paper, the Enhanced Driving Risk Field (EDRF) model is proposed, integrating deep learning-based multimodal trajectory prediction results with Gaussian distribution models to quantitatively capture the uncertainty of traffic entities' behavior. The applications of the EDRF are also proposed. It is applied across various tasks (traffic risk monitoring, ego-vehicle risk analysis, and motion and trajectory planning) through the defined concept Interaction Risk (IR). Adequate example scenarios are provided for each application to illustrate the effectiveness of the model. 

**Abstract (ZH)**: 基于增强驾驶风险场的驾驶风险评估 

---
