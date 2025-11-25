# RynnVLA-002: A Unified Vision-Language-Action and World Model 

**Title (ZH)**: RynnVLA-002: 统一的视觉-语言-动作与世界模型 

**Authors**: Jun Cen, Siteng Huang, Yuqian Yuan, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Kehan Li, Hao Luo, Fan Wang, Xin Li, Deli Zhao, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.17502)  

**Abstract**: We introduce RynnVLA-002, a unified Vision-Language-Action (VLA) and world model. The world model leverages action and visual inputs to predict future image states, learning the underlying physics of the environment to refine action generation. Conversely, the VLA model produces subsequent actions from image observations, enhancing visual understanding and supporting the world model's image generation. The unified framework of RynnVLA-002 enables joint learning of environmental dynamics and action planning. Our experiments show that RynnVLA-002 surpasses individual VLA and world models, demonstrating their mutual enhancement. We evaluate RynnVLA-002 in both simulation and real-world robot tasks. RynnVLA-002 achieves 97.4% success rate on the LIBERO simulation benchmark without pretraining, while in real-world LeRobot experiments, its integrated world model boosts the overall success rate by 50%. 

**Abstract (ZH)**: 我们介绍了RynnVLA-002，一个统一的视觉-语言-动作（VLA）和世界模型。世界模型利用动作和视觉输入来预测未来的图像状态，学习环境的底层物理规律以精化动作生成。相反，VLA模型根据图像观察生成后续动作，增强视觉理解并支持世界模型的图像生成。RynnVLA-002的统一框架使环境动力学和动作规划能够协同学习。我们的实验表明，RynnVLA-002超越了单独的VLA和世界模型，展示了它们的相互增强效果。我们在仿真和真实世界机器人任务中评估了RynnVLA-002。在无需预训练的情况下，RynnVLA-002在LIBERO仿真基准测试中的成功率达到了97.4%，而在真实世界LeRobot实验中，其集成世界模型将整体成功率提高了50%。 

---
# HALO: High-Altitude Language-Conditioned Monocular Aerial Exploration and Navigation 

**Title (ZH)**: HALO: 高海拔语言条件单目空中探索与导航 

**Authors**: Yuezhan Tao, Dexter Ong, Fernando Cladera, Jason Hughes, Camillo J. Taylor, Pratik Chaudhari, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.17497)  

**Abstract**: We demonstrate real-time high-altitude aerial metric-semantic mapping and exploration using a monocular camera paired with a global positioning system (GPS) and an inertial measurement unit (IMU). Our system, named HALO, addresses two key challenges: (i) real-time dense 3D reconstruction using vision at large distances, and (ii) mapping and exploration of large-scale outdoor environments with accurate scene geometry and semantics. We demonstrate that HALO can plan informative paths that exploit this information to complete missions with multiple tasks specified in natural language. In simulation-based evaluation across large-scale environments of size up to 78,000 sq. m., HALO consistently completes tasks with less exploration time and achieves up to 68% higher competitive ratio in terms of the distance traveled compared to the state-of-the-art semantic exploration baseline. We use real-world experiments on a custom quadrotor platform to demonstrate that (i) all modules can run onboard the robot, and that (ii) in diverse environments HALO can support effective autonomous execution of missions covering up to 24,600 sq. m. area at an altitude of 40 m. Experiment videos and more details can be found on our project page: this https URL. 

**Abstract (ZH)**: 我们采用单目摄像头配以全球定位系统（GPS）和惯性测量单元（IMU），展示了实时高空航拍度量语义地图构建与探索。我们的系统名为HALO，解决了两个关键挑战：（i）远距离视觉下的实时密集三维重建，以及(ii) 在具有精确场景几何结构和语义信息的大规模室外环境中进行探索。我们证明HALO能够规划具有信息利用性的路径，以完成指定自然语言的多重任务。在最大规模为78,000平方米的仿真环境中，HALO在探索时间和行驶距离方面均优于最先进的语义探索基准，最高可提升68%的竞争比。我们在自定义四旋翼平台上的实地实验表明，(i) 所有模块均可以在机器人上运行，且(ii) 在多种环境中，HALO可支持在40米高度覆盖最多24,600平方米区域的自主任务执行。更多信息和实验视频请参见我们的项目页面：this https URL。 

---
# MDG: Masked Denoising Generation for Multi-Agent Behavior Modeling in Traffic Environments 

**Title (ZH)**: MDG: 遮蔽去噪生成在交通环境中的多-agent 行为建模 

**Authors**: Zhiyu Huang, Zewei Zhou, Tianhui Cai, Yun Zhang, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.17496)  

**Abstract**: Modeling realistic and interactive multi-agent behavior is critical to autonomous driving and traffic simulation. However, existing diffusion and autoregressive approaches are limited by iterative sampling, sequential decoding, or task-specific designs, which hinder efficiency and reuse. We propose Masked Denoising Generation (MDG), a unified generative framework that reformulates multi-agent behavior modeling as the reconstruction of independently noised spatiotemporal tensors. Instead of relying on diffusion time steps or discrete tokenization, MDG applies continuous, per-agent and per-timestep noise masks that enable localized denoising and controllable trajectory generation in a single or few forward passes. This mask-driven formulation generalizes across open-loop prediction, closed-loop simulation, motion planning, and conditional generation within one model. Trained on large-scale real-world driving datasets, MDG achieves competitive closed-loop performance on the Waymo Sim Agents and nuPlan Planning benchmarks, while providing efficient, consistent, and controllable open-loop multi-agent trajectory generation. These results position MDG as a simple yet versatile paradigm for multi-agent behavior modeling. 

**Abstract (ZH)**: Masked Denoising Generation: A Unified Framework for Efficient and Controllable Multi-Agent Behavior Modeling 

---
# RoboCOIN: An Open-Sourced Bimanual Robotic Data COllection for INtegrated Manipulation 

**Title (ZH)**: RoboCOIN: 一种开源的双臂机器人数据收集系统以集成操作研究 

**Authors**: Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, Zhaoye Long, Yue Wang, Chong Liu, Dihan Wang, Ziqiang Ni, Xiang Yang, You Liu, Ruoxuan Feng, Runtian Xu, Lei Zhang, Denghang Huang, Chenghao Jin, Anlan Yin, Xinlong Wang, Zhenguo Sun, Junkai Zhao, Mengfei Du, Mingyu Cao, Xiansheng Chen, Hongyang Cheng, Xiaojie Zhang, Yankai Fu, Ning Chen, Cheng Chi, Sixiang Chen, Huaihai Lyu, Xiaoshuai Hao, Yankai Fu, Yequan Wang, Bo Lei, Dong Liu, Xi Yang, Yance Jiao, Tengfei Pan, Yunyan Zhang, Songjing Wang, Ziqian Zhang, Xu Liu, Ji Zhang, Caowei Meng, Zhizheng Zhang, Jiyang Gao, Song Wang, Xiaokun Leng, Zhiqiang Xie, Zhenzhen Zhou, Peng Huang, Wu Yang, Yandong Guo, Yichao Zhu, Suibing Zheng, Hao Cheng, Xinmin Ding, Yang Yue, Huanqian Wang, Chi Chen, Jingrui Pang, YuXi Qian, Haoran Geng, Lianli Gao, Haiyuan Li, Bin Fang, Gao Huang, Yaodong Yang, Hao Dong, He Wang, Hang Zhao, Yadong Mu, Di Hu, Hao Zhao, Tiejun Huang, Shanghang Zhang, Yonghua Lin, Zhongyuan Wang, Guocai Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.17441)  

**Abstract**: Bimanual manipulation is essential for achieving human-like dexterity in robots, but the large-scale and diverse bimanual robot datasets remain scarce due to hardware heterogeneity across robotic platforms. To address the challenge, we present RoboCOIN, a comprehensive multi-embodiment bimanual manipulation dataset with over 180,000 demonstrations collected from 15 distinct robotic platforms. The dataset covers 16 scenarios, including residential, commercial, and working environments, with 421 tasks systematically organized by bimanual coordination patterns and object properties. Our key innovation is a hierarchical capability pyramid that provides multi-level annotations, spanning trajectory-level concepts, segment-level subtasks, and frame-level kinematics. We further develop CoRobot, a comprehensive processing framework featuring Robot Trajectory Markup Language (RTML) for quality assessment, automated annotation generation, and unified multi-embodiment management. Extensive experiments demonstrate the reliability and effectiveness of RoboCOIN in multi-embodiment bimanual learning, with significant performance improvements across various model architectures and robotic platforms. The complete dataset and framework are open-sourced and publicly available for further research purposes. Project website: this https URL. 

**Abstract (ZH)**: 双臂操作对于实现类人灵巧性至关重要，但由于不同机器人平台上硬件异质性导致的大规模和多样化的双臂机器人数据集仍然匮乏，为此我们提出RoboCOIN，这是一个包含超过180,000个演示的全面双臂操作多体征数据集，收集自15种不同的机器人平台。该数据集覆盖了16种场景，包括住宅、商业和工作环境，且421项任务通过双臂协调模式和物体属性系统化组织。我们的关键创新是一个分层能力金字塔，提供了多层次注解，涵盖轨迹级概念、段级子任务和帧级运动学。此外，我们进一步开发了CoRobot，这是一个全面的处理框架，其中包括用于质量评估、自动化注解生成和统一多体征管理的Robot Trajectory Markup Language (RTML)。广泛的实验表明，RoboCOIN 在多体征双臂学习中具有可靠性与有效性，并在多种模型架构和机器人平台上取得了显著性能提升。整个数据集和框架已开源并公开提供，以便于进一步研究。项目网址：this https URL。 

---
# SPEAR-1: Scaling Beyond Robot Demonstrations via 3D Understanding 

**Title (ZH)**: SPEAR-1: 通过三维理解超越机器人示范的扩展学习 

**Authors**: Nikolay Nikolov, Giuliano Albanese, Sombit Dey, Aleksandar Yanev, Luc Van Gool, Jan-Nico Zaech, Danda Pani Paudel  

**Link**: [PDF](https://arxiv.org/pdf/2511.17411)  

**Abstract**: Robotic Foundation Models (RFMs) hold great promise as generalist, end-to-end systems for robot control. Yet their ability to generalize across new environments, tasks, and embodiments remains limited. We argue that a major bottleneck lies in their foundations: most RFMs are built by fine-tuning internet-pretrained Vision-Language Models (VLMs). However, these VLMs are trained on 2D image-language tasks and lack the 3D spatial reasoning inherently required for embodied control in the 3D world. Bridging this gap directly with large-scale robotic data is costly and difficult to scale. Instead, we propose to enrich easy-to-collect non-robotic image data with 3D annotations and enhance a pretrained VLM with 3D understanding capabilities. Following this strategy, we train SPEAR-VLM, a 3D-aware VLM that infers object coordinates in 3D space from a single 2D image. Building on SPEAR-VLM, we introduce our main contribution, $~\textbf{SPEAR-1}$: a robotic foundation model that integrates grounded 3D perception with language-instructed embodied control. Trained on $\sim$45M frames from 24 Open X-Embodiment datasets, SPEAR-1 outperforms or matches state-of-the-art models such as $\pi_0$-FAST and $\pi_{0.5}$, while it uses 20$\times$ fewer robot demonstrations. This carefully-engineered training strategy unlocks new VLM capabilities and as a consequence boosts the reliability of embodied control beyond what is achievable with only robotic data. We make our model weights and 3D-annotated datasets publicly available. 

**Abstract (ZH)**: 基于3D理解的Robotic Foundation Models：SPEAR-1 

---
# Feasibility of Embodied Dynamics Based Bayesian Learning for Continuous Pursuit Motion Control of Assistive Mobile Robots in the Built Environment 

**Title (ZH)**: 基于体 packageNameodied 动力学的贝叶斯学习在建筑物环境中的辅助移动机器人连续追踪运动控制可行性研究 

**Authors**: Xiaoshan Zhou, Carol C. Menassa, Vineet R. Kamat  

**Link**: [PDF](https://arxiv.org/pdf/2511.17401)  

**Abstract**: Non-invasive electroencephalography (EEG)-based brain-computer interfaces (BCIs) offer an intuitive means for individuals with severe motor impairments to independently operate assistive robotic wheelchairs and navigate built environments. Despite considerable progress in BCI research, most current motion control systems are limited to discrete commands, rather than supporting continuous pursuit, where users can freely adjust speed and direction in real time. Such natural mobility control is, however, essential for wheelchair users to navigate complex public spaces, such as transit stations, airports, hospitals, and indoor corridors, to interact socially with the dynamic populations with agility, and to move flexibly and comfortably as autonomous driving is refined to allow movement at will. In this study, we address the gap of continuous pursuit motion control in BCIs by proposing and validating a brain-inspired Bayesian inference framework, where embodied dynamics in acceleration-based motor representations are decoded. This approach contrasts with conventional kinematics-level decoding and deep learning-based methods. Using a public dataset with sixteen hours of EEG from four subjects performing motor imagery-based target-following, we demonstrate that our method, utilizing Automatic Relevance Determination for feature selection and continual online learning, reduces the normalized mean squared error between predicted and true velocities by 72% compared to autoregressive and EEGNet-based methods in a session-accumulative transfer learning setting. Theoretically, these findings empirically support embodied cognition theory and reveal the brain's intrinsic motor control dynamics in an embodied and predictive nature. Practically, grounding EEG decoding in the same dynamical principles that govern biological motion offers a promising path toward more stable and intuitive BCI control. 

**Abstract (ZH)**: 非侵入性脑电图（EEG）为基础的脑-计算机接口（BCI）为严重运动障碍个体提供了直观方式，使其独立操作辅助轮椅并导航人造环境。尽管在BCI研究方面取得了显著进展，但大多数当前的运动控制系统仅支持离散命令，而未能支持连续追踪，使用户无法实时自由调整速度和方向。然而，这种自然的运动控制对于轮椅使用者在复杂的公共交通场所、机场、医院和室内走廊等各种环境中导航，与动态的人群进行社交互动以及灵活舒适地移动至关重要，特别随着自动驾驶技术的发展，使人们能够自由移动。本研究通过提出并验证一种受脑启发的贝叶斯推断框架，以解决BCI中连续追踪运动控制的空白，在基于加速度的运动表征中解码体素动力学。该方法与传统的运动学水平解码和深度学习方法形成对比。利用四名受试者进行运动想象目标跟随实验的公开数据集，包含16小时的EEG记录，我们展示了该方法利用自动相关性确定性进行特征选择和持续在线学习，在会话累积性迁移学习设置中，与自回归和EEGNet方法相比，减少了预测速度与真实速度之间的归一化均方误差72%。理论上，这些发现支持了体验认知理论，并揭示了大脑在体验性和预测性方面的内在运动控制动力学。在实践上，将EEG解码建立在控制生物运动相同的动力学原理之上，提供了一条走向更稳定和直观BCI控制的有希望的道路。 

---
# Human Imitated Bipedal Locomotion with Frequency Based Gait Generator Network 

**Title (ZH)**: 基于频率的步态生成网络仿人类双足运动 

**Authors**: Yusuf Baran Ates, Omer Morgul  

**Link**: [PDF](https://arxiv.org/pdf/2511.17387)  

**Abstract**: Learning human-like, robust bipedal walking remains difficult due to hybrid dynamics and terrain variability. We propose a lightweight framework that combines a gait generator network learned from human motion with Proximal Policy Optimization (PPO) controller for torque control. Despite being trained only on flat or mildly sloped ground, the learned policies generalize to steeper ramps and rough surfaces. Results suggest that pairing spectral motion priors with Deep Reinforcement Learning (DRL) offers a practical path toward natural and robust bipedal locomotion with modest training cost. 

**Abstract (ZH)**: 学习类人、鲁棒的 bipedal 行走由于混合动力学和地形变化依然具有挑战性。我们提出了一种轻量级框架，将从人类动作中学得的步伐生成网络与近端策略优化（PPO）控制器结合用于扭矩控制。尽管仅在平坦或轻微斜坡的地面上进行训练，学到的策略仍能在陡峭的斜坡和粗糙的地面上泛化。结果表明，将频谱运动先验与深度强化学习（DRL）结合是一种实用路径，以实现自然且鲁棒的 bipedal 运动，同时训练成本较低。 

---
# IndustryNav: Exploring Spatial Reasoning of Embodied Agents in Dynamic Industrial Navigation 

**Title (ZH)**: IndustryNav: 探索动态工业导航中具身代理的空间推理 

**Authors**: Yifan Li, Lichi Li, Anh Dao, Xinyu Zhou, Yicheng Qiao, Zheda Mai, Daeun Lee, Zichen Chen, Zhen Tan, Mohit Bansal, Yu Kong  

**Link**: [PDF](https://arxiv.org/pdf/2511.17384)  

**Abstract**: While Visual Large Language Models (VLLMs) show great promise as embodied agents, they continue to face substantial challenges in spatial reasoning. Existing embodied benchmarks largely focus on passive, static household environments and evaluate only isolated capabilities, failing to capture holistic performance in dynamic, real-world complexity. To fill this gap, we present IndustryNav, the first dynamic industrial navigation benchmark for active spatial reasoning. IndustryNav leverages 12 manually created, high-fidelity Unity warehouse scenarios featuring dynamic objects and human movement. Our evaluation employs a PointGoal navigation pipeline that effectively combines egocentric vision with global odometry to assess holistic local-global planning. Crucially, we introduce the "collision rate" and "warning rate" metrics to measure safety-oriented behaviors and distance estimation. A comprehensive study of nine state-of-the-art VLLMs (including models such as GPT-5-mini, Claude-4.5, and Gemini-2.5) reveals that closed-source models maintain a consistent advantage; however, all agents exhibit notable deficiencies in robust path planning, collision avoidance and active exploration. This highlights a critical need for embodied research to move beyond passive perception and toward tasks that demand stable planning, active exploration, and safe behavior in dynamic, real-world environment. 

**Abstract (ZH)**: 视觉大型语言模型在空间推理中的动态工业导航基准 

---
# Vector Cost Behavioral Planning for Autonomous Robotic Systems with Contemporary Validation Strategies 

**Title (ZH)**: 面向自主机器人系统的矢量成本行为规划及其当代验证策略 

**Authors**: Benjamin R. Toaz, Quentin Goss, John Thompson, Seta Boğosyan, Shaunak D. Bopardikar, Mustafa İlhan Akbaş, Metin Gökaşan  

**Link**: [PDF](https://arxiv.org/pdf/2511.17375)  

**Abstract**: The vector cost bimatrix game is a method for multi-objective decision making that enables autonomous robotic systems to optimize for multiple goals at once while avoiding worst-case scenarios in neglected objectives. We expand this approach to arbitrary numbers of objectives and compare its performance to scalar weighted sum methods during competitive motion planning. Explainable Artificial Intelligence (XAI) software is used to aid in the analysis of high dimensional decision-making data. State-space Exploration of Multidimensional Boundaries using Adherence Strategies (SEMBAS) is applied to explore performance modes in the parameter space as a sensitivity study for the baseline and proposed frameworks. While some works have explored aspects of game theoretic planning and intelligent systems validation separately, we combine each of these into a novel and comprehensive simulation pipeline. This integration demonstrates a dramatic improvement of the vector cost method over scalarization and offers an interpretable and generalizable framework for robotic behavioral planning. Code available at this https URL. The video companion to this work is available at this https URL. 

**Abstract (ZH)**: 多目标向量成本双矩阵博弈是一种同时优化多个目标并避免次要目标最坏情况的自主机器人系统优化方法。我们将其扩展到任意数量的目标，并将其性能与标量加权和方法在竞争性运动规划中的表现进行比较。可解释的人工智能（XAI）软件用于分析高维度决策数据。采用状态空间探索多维度边界策略（SEMBAS）对基线和提议框架的参数空间进行灵敏度研究，以探索性能模式。虽然一些研究分别探讨了博弈论规划和智能系统验证的方面，我们将其整合到一个新颖且综合的模拟管道中。这种方法展示了向量成本方法相对于标量化方法的巨大改进，并提供了一个可解释且可泛化的机器人行为规划框架。相关代码可在以下链接获取：this https URL。此工作的视频辅助资料可在以下链接获取：this https URL。 

---
# Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data 

**Title (ZH)**: 敏捷性与稳定性相结合：异构数据驱动的通用 humanoid 控制 

**Authors**: Yixuan Pan, Ruoyi Qiao, Li Chen, Kashyap Chitta, Liang Pan, Haoguang Mai, Qingwen Bu, Hao Zhao, Cunyuan Zheng, Ping Luo, Hongyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.17373)  

**Abstract**: Humanoid robots are envisioned to perform a wide range of tasks in human-centered environments, requiring controllers that combine agility with robust balance. Recent advances in locomotion and whole-body tracking have enabled impressive progress in either agile dynamic skills or stability-critical behaviors, but existing methods remain specialized, focusing on one capability while compromising the other. In this work, we introduce AMS (Agility Meets Stability), the first framework that unifies both dynamic motion tracking and extreme balance maintenance in a single policy. Our key insight is to leverage heterogeneous data sources: human motion capture datasets that provide rich, agile behaviors, and physically constrained synthetic balance motions that capture stability configurations. To reconcile the divergent optimization goals of agility and stability, we design a hybrid reward scheme that applies general tracking objectives across all data while injecting balance-specific priors only into synthetic motions. Further, an adaptive learning strategy with performance-driven sampling and motion-specific reward shaping enables efficient training across diverse motion distributions. We validate AMS extensively in simulation and on a real Unitree G1 humanoid. Experiments demonstrate that a single policy can execute agile skills such as dancing and running, while also performing zero-shot extreme balance motions like Ip Man's Squat, highlighting AMS as a versatile control paradigm for future humanoid applications. 

**Abstract (ZH)**: humanoid机器人在以人为中心的环境中被构想用于执行广泛的任务，需要结合灵活和稳固的控制器。近期在移动性和全身跟踪方面的进展使得在敏捷动态技能或稳定性关键行为方面取得了显著进展，但现有方法仍然专业化，专注于一种能力而牺牲了另一种能力。在这项工作中，我们提出了AMS（敏捷性与稳定性相结合），这是首个在单一策略中统一动态运动跟踪和极端平衡维护的框架。我们的关键见解是利用异质数据源：提供丰富灵活行为的人类运动捕捉数据集，以及受到物理约束的合成平衡动作数据集，这些数据集捕捉了稳定性配置。为了在敏捷性和稳定性之间达成一致的优化目标，我们设计了一种混合奖励方案，该方案在所有数据上应用通用跟踪目标，而在合成运动中仅注入平衡特定的先验知识。此外，一种基于性能的采样和动作特定的奖励塑形的自适应学习策略使得在多样化的运动分布中高效训练成为可能。在仿真和实际的Unitree G1人型机器人上对AMS进行了广泛验证。实验结果表明，单一策略不仅能够执行敏捷技能如跳舞和奔跑，还能执行零样本的极端平衡动作如Ip Man的深蹲，突显了AMS作为未来人型机器人应用中多功能控制范式的潜力。 

---
# METIS: Multi-Source Egocentric Training for Integrated Dexterous Vision-Language-Action Model 

**Title (ZH)**: METIS：多源自视训练赋能统一灵巧视觉-语言-行动模型 

**Authors**: Yankai Fu, Ning Chen, Junkai Zhao, Shaozhe Shan, Guocai Yao, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17366)  

**Abstract**: Building a generalist robot that can perceive, reason, and act across diverse tasks remains an open challenge, especially for dexterous manipulation. A major bottleneck lies in the scarcity of large-scale, action-annotated data for dexterous skills, as teleoperation is difficult and costly. Human data, with its vast scale and diverse manipulation behaviors, provides rich priors for learning robotic actions. While prior works have explored leveraging human demonstrations, they are often constrained by limited scenarios and a large visual gap between human and robots. To eliminate these limitations, we propose METIS, a vision-language-action (VLA) model for dexterous manipulation pretrained on multi-source egocentric datasets. We first construct EgoAtlas, which integrates large-scale human and robotic data from multiple sources, all unified under a consistent action space. We further extract motion-aware dynamics, a compact and discretized motion representation, which provides efficient and expressive supervision for VLA training. Built upon them, METIS integrates reasoning and acting into a unified framework, enabling effective deployment to downstream dexterous manipulation tasks. Our method demonstrates exceptional dexterous manipulation capabilities, achieving highest average success rate in six real-world tasks. Experimental results also highlight the superior generalization and robustness to out-of-distribution scenarios. These findings emphasize METIS as a promising step toward a generalist model for dexterous manipulation. 

**Abstract (ZH)**: 构建能够在多样化任务中进行感知、推理和操作的一般ist机器人，特别是在灵巧 manipulation 方面，仍是一个开放性的挑战。主要瓶颈在于灵巧技能数据稀缺，因为远程操作困难且成本高。人类数据因其大规模和多样的操作行为，为学习机器人的操作提供了丰富的先验知识。尽管先前的工作探索了利用人类示范，但它们往往受到场景有限和人机视觉差距大的限制。为了消除这些限制，我们提出了一种基于多源第一人称数据集预训练的视觉-语言-动作（VLA）模型 METIS，用于灵巧 manipulation。我们首先构建了 EgoAtlas，将多源的大规模人类和机器人数据统一在一个一致的动作空间下。进一步提出了运动感知动力学，这是一种紧凑且离散的运动表示，为 VLA 训练提供高效的表达监督。基于此，METIS 将推理和操作整合到一个统一框架中，使其能够有效部署到下游的灵巧 manipulation 任务中。我们的方法展示了卓越的灵巧 manipulation 能力，在六个实际任务中平均成功率最高。实验结果还突显了其在分布外场景下的优越泛化能力和鲁棒性。这些发现强调了 METIS 作为灵巧 manipulation 通用模型的前景。 

---
# Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM 

**Title (ZH)**: 基于长上下文Q-Former的多模态大语言模型的机器人确认生成与动作规划 

**Authors**: Chiori Hori, Yoshiki Masuyama, Siddarth Jain, Radu Corcodel, Devesh Jha, Diego Romeres, Jonathan Le Roux  

**Link**: [PDF](https://arxiv.org/pdf/2511.17335)  

**Abstract**: Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3. 

**Abstract (ZH)**: 人类与机器人共同朝着共同目标进行合作需要机器人理解人类的动作及其与周围环境的互动。本文 focuses 人类-机器人互动（HRI）基于人类-机器人对话，依赖于多模态场景理解进行机器人的动作确认和动作步骤生成。最先进的方法使用多模态变压器从包含多个微观步骤的单个剪辑任务中生成与机器人动作确认对齐的机器人动作步骤。尽管长时序任务中的动作在整个视频过程中相互依赖，但当前的方法主要关注剪辑级别的处理，并未利用长上下文信息。本文提出了一种结合全视频左、右上下文依赖性的长上下文 Q-former。此外，本文提出了一种文本条件化方法，直接将文本嵌入输入到LLM解码器中，以减轻Q-former对文本信息的高度抽象。实验结果表明，确认生成的准确性是动作规划性能的主要因素。此外，我们证明了结合VideoLLaMA3的长上下文Q-former能够改进确认和动作规划。 

---
# FORWARD: Dataset of a forwarder operating in rough terrain 

**Title (ZH)**: FORWARD: 严峻地形中传送器的数据集 

**Authors**: Mikael Lundbäck, Erik Wallin, Carola Häggström, Mattias Nyström, Andreas Grönlund, Mats Richardson, Petrus Jönsson, William Arnvik, Lucas Hedström, Arvid Fälldin, Martin Servin  

**Link**: [PDF](https://arxiv.org/pdf/2511.17318)  

**Abstract**: We present FORWARD, a high-resolution multimodal dataset of a cut-to-length forwarder operating in rough terrain on two harvest sites in the middle part of Sweden. The forwarder is a large Komatsu model equipped with a variety of sensors, including RTK-GNSS, 360-camera, operator vibration sensors, internal CAN-bus signal recording, and multiple IMUs. The data includes event time logs recorded in 5 Hz with e.g., driving speed, fuel consumption, vehicle position with centimeter accuracy, and crane use while the vehicle operates in forest areas laser-scanned with very high-resolution, $\sim$1500 points per square meter. Production log files (StanForD standard) with time-stamped machine events, extensive video material, and terrain data in various formats are included as well. About 18 hours of regular wood extraction work during three days is annotated from 360-video material into individual work elements and included in the dataset. We also include scenario specifications of conducted experiments on forest roads and in terrain. Scenarios include repeatedly driving the same routes with and without steel tracks, different load weight, and different target driving speeds. The dataset is intended for developing models and algorithms for trafficability, perception, and autonomous control of forest machines using artificial intelligence, simulation, and experiments on physical testbeds. In part, we focus on forwarders traversing terrain, avoiding obstacles, and loading or unloading logs, with consideration for efficiency, fuel consumption, safety, and environmental impact. Other benefits of the open dataset include the ability to explore auto-generation and calibration of forestry machine simulators and automation scenario descriptions using the data recorded in the field. 

**Abstract (ZH)**: 我们提出FORWARD，一个高分辨率多模态数据集，该数据集记录了森林中部地区两个采伐现场的切割至长度的前移器在崎岖地形上的操作情况。前移器配备有各种传感器，包括RTK-GNSS、360度摄像头、操作员振动传感器、车内CAN总线信号记录以及多个IMU。数据包含以5 Hz频率记录的事件时间日志，例如行驶速度、燃料消耗、厘米级精确的车辆位置以及操作期间的吊臂使用情况。车辆在森林区域进行激光扫描，分辨率非常高，约为每平方米1500个点。数据集还包括带有时间戳的机器事件生产日志文件（StanForD标准）、广泛的视频材料以及多种形式的地形数据。约18小时为期三天的正常木材采伐工作被从360度视频材料中标注为个体工作元素，并包含在数据集中。我们还包含关于在森林道路和地形上进行的试验中的场景规范。场景包括多次重复驾驶相同路线，有无钢制履带，不同负载重量和不同目标行驶速度。该数据集旨在用于利用人工智能、模拟和物理试验台上的实验开发森林机械的通行性、感知和自主控制模型和算法。我们重点关注前移器穿越地形、避开障碍物、装车或卸车的情况，同时考虑到效率、燃料消耗、安全和环境影响。开放数据集的其他好处包括利用田间记录的数据探索和校准森林机械模拟器以及自动化场景描述的能力。 

---
# MonoSpheres: Large-Scale Monocular SLAM-Based UAV Exploration through Perception-Coupled Mapping and Planning 

**Title (ZH)**: MonoSpheres：基于感知耦合建图与规划的大规模单目SLAM无人机探索 

**Authors**: Tomáš Musil, Matěj Petrlík, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2511.17299)  

**Abstract**: Autonomous exploration of unknown environments is a key capability for mobile robots, but it is largely unsolved for robots equipped with only a single monocular camera and no dense range sensors. In this paper, we present a novel approach to monocular vision-based exploration that can safely cover large-scale unstructured indoor and outdoor 3D environments by explicitly accounting for the properties of a sparse monocular SLAM frontend in both mapping and planning. The mapping module solves the problems of sparse depth data, free-space gaps, and large depth uncertainty by oversampling free space in texture-sparse areas and keeping track of obstacle position uncertainty. The planning module handles the added free-space uncertainty through rapid replanning and perception-aware heading control. We further show that frontier-based exploration is possible with sparse monocular depth data when parallax requirements and the possibility of textureless surfaces are taken into account. We evaluate our approach extensively in diverse real-world and simulated environments, including ablation studies. To the best of the authors' knowledge, the proposed method is the first to achieve 3D monocular exploration in real-world unstructured outdoor environments. We open-source our implementation to support future research. 

**Abstract (ZH)**: 单目视觉导向的自主未知环境探索：面向稀疏单目SLAM前端的大规模未结构化室内和室外3D环境安全覆盖方法 

---
# Leveraging CVAE for Joint Configuration Estimation of Multifingered Grippers from Point Cloud Data 

**Title (ZH)**: 利用条件变分自编码器进行多指灵巧手点云数据的联合配置估计 

**Authors**: Julien Merand, Boris Meden, Mathieu Grossard  

**Link**: [PDF](https://arxiv.org/pdf/2511.17276)  

**Abstract**: This paper presents an efficient approach for determining the joint configuration of a multifingered gripper solely from the point cloud data of its poly-articulated chain, as generated by visual sensors, simulations or even generative neural networks. Well-known inverse kinematics (IK) techniques can provide mathematically exact solutions (when they exist) for joint configuration determination based solely on the fingertip pose, but often require post-hoc decision-making by considering the positions of all intermediate phalanges in the gripper's fingers, or rely on algorithms to numerically approximate solutions for more complex kinematics. In contrast, our method leverages machine learning to implicitly overcome these challenges. This is achieved through a Conditional Variational Auto-Encoder (CVAE), which takes point cloud data of key structural elements as input and reconstructs the corresponding joint configurations. We validate our approach on the MultiDex grasping dataset using the Allegro Hand, operating within 0.05 milliseconds and achieving accuracy comparable to state-of-the-art methods. This highlights the effectiveness of our pipeline for joint configuration estimation within the broader context of AI-driven techniques for grasp planning. 

**Abstract (ZH)**: 本文提出了一种有效的方法，仅从视觉传感器、仿真或生成神经网络生成的多关节链的点云数据中确定多指 gripper 的联合配置。已知的逆向运动学（IK）技术可以在指尖姿态的基础上提供数学上精确的解决方案（当存在时），但通常需要通过考虑 gripper 指节中所有中间指骨的位置来进行后续决策，或者依赖于算法来近似求解更复杂的运动学。相比之下，我们的方法利用机器学习隐式地克服了这些挑战。这通过 Conditional Variational Auto-Encoder (CVAE) 实现，CVAE 以关键结构元素的点云数据作为输入，重构相应的关节配置。我们在使用 Allegro Hand 的 MultiDex 抓取数据集中验证了该方法，在 0.05 毫秒内运行，并实现了与当前最先进的方法相媲美的准确度。这突显了我们用于关节配置估计的流水线在更广泛的基于 AI 的抓取规划技术中的有效性。 

---
# Simulation of Active Soft Nets for Capture of Space Debris 

**Title (ZH)**: 主动软网捕获空间碎片仿真 

**Authors**: Leone Costi, Dario Izzo  

**Link**: [PDF](https://arxiv.org/pdf/2511.17266)  

**Abstract**: In this work, we propose a simulator, based on the open-source physics engine MuJoCo, for the design and control of soft robotic nets for the autonomous removal of space debris. The proposed simulator includes net dynamics, contact between the net and the debris, self-contact of the net, orbital mechanics, and a controller that can actuate thrusters on the four satellites at the corners of the net. It showcases the case of capturing Envisat, a large ESA satellite that remains in orbit as space debris following the end of its mission. This work investigates different mechanical models, which can be used to simulate the net dynamics, simulating various degrees of compliance, and different control strategies to achieve the capture of the debris, depending on the relative position of the net and the target. Unlike previous works on this topic, we do not assume that the net has been previously ballistically thrown toward the target, and we start from a relatively static configuration. The results show that a more compliant net achieves higher performance when attempting the capture of Envisat. Moreover, when paired with a sliding mode controller, soft nets are able to achieve successful capture in 100% of the tested cases, whilst also showcasing a higher effective area at contact and a higher number of contact points between net and Envisat. 

**Abstract (ZH)**: 基于开源物理引擎MuJoCo的软机器人网自主清除空间碎片设计与控制仿真 

---
# A ROS2 Interface for Universal Robots Collaborative Manipulators Based on ur_rtde 

**Title (ZH)**: 基于ur_rtde的ROS2接口研究：面向通用机器人协作执行器 

**Authors**: Alessio Saccuti, Riccardo Monica, Jacopo Aleotti  

**Link**: [PDF](https://arxiv.org/pdf/2511.17237)  

**Abstract**: In this paper a novel ROS2 driver for UR robot manipulators is presented, based on the ur_rtde C++ library. The proposed driver aims to be a flexible solution, adaptable to a wide range of applications. The driver exposes the high-level commands of Universal Robots URScripts, and custom commands can be added using a plugin system. Several commands have been implemented, including motion execution along a waypoint-based path. The driver is published as open source. 

**Abstract (ZH)**: 本文提出了一种基于ur_rtde C++库的新型ROS2驱动程序，用于UR机器人 manipulator，旨在提供一种灵活的解决方案，适用于广泛的应用场景。该驱动程序暴露了通用机器人URScripts的高级命令，并可通过插件系统添加自定义命令。实现了多种命令，包括基于waypoint的路径运动执行。该驱动程序已开源。 

---
# TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making 

**Title (ZH)**: TP-MDDN: 任务偏好型多需求驱动导航与自主决策 

**Authors**: Shanshan Li, Da Huang, Yu He, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.17225)  

**Abstract**: In daily life, people often move through spaces to find objects that meet their needs, posing a key challenge in embodied AI. Traditional Demand-Driven Navigation (DDN) handles one need at a time but does not reflect the complexity of real-world tasks involving multiple needs and personal choices. To bridge this gap, we introduce Task-Preferenced Multi-Demand-Driven Navigation (TP-MDDN), a new benchmark for long-horizon navigation involving multiple sub-demands with explicit task preferences. To solve TP-MDDN, we propose AWMSystem, an autonomous decision-making system composed of three key modules: BreakLLM (instruction decomposition), LocateLLM (goal selection), and StatusMLLM (task monitoring). For spatial memory, we design MASMap, which combines 3D point cloud accumulation with 2D semantic mapping for accurate and efficient environmental understanding. Our Dual-Tempo action generation framework integrates zero-shot planning with policy-based fine control, and is further supported by an Adaptive Error Corrector that handles failure cases in real time. Experiments demonstrate that our approach outperforms state-of-the-art baselines in both perception accuracy and navigation robustness. 

**Abstract (ZH)**: 日常生活中的物体搜索任务为身体化AI带来了关键挑战：面向任务的多需求驱动导航 

---
# Efficient Robot Design with Multi-Objective Black-Box Optimization and Large Language Models 

**Title (ZH)**: 基于多目标黑盒优化和大型语言模型的高效机器人设计 

**Authors**: Kento Kawaharazuka, Yoshiki Obinata, Naoaki Kanazawa, Haoyu Jia, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2511.17178)  

**Abstract**: Various methods for robot design optimization have been developed so far. These methods are diverse, ranging from numerical optimization to black-box optimization. While numerical optimization is fast, it is not suitable for cases involving complex structures or discrete values, leading to frequent use of black-box optimization instead. However, black-box optimization suffers from low sampling efficiency and takes considerable sampling iterations to obtain good solutions. In this study, we propose a method to enhance the efficiency of robot body design based on black-box optimization by utilizing large language models (LLMs). In parallel with the sampling process based on black-box optimization, sampling is performed using LLMs, which are provided with problem settings and extensive feedback. We demonstrate that this method enables more efficient exploration of design solutions and discuss its characteristics and limitations. 

**Abstract (ZH)**: 基于大型语言模型提升黑盒优化在机器人机体设计中的效率方法 

---
# Reflection-Based Relative Localization for Cooperative UAV Teams Using Active Markers 

**Title (ZH)**: 基于反射的相对定位方法及其在使用主动标志的合作无人机团队中的应用 

**Authors**: Tim Lakemann, Daniel Bonilla Licea, Viktor Walter, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2511.17166)  

**Abstract**: Reflections of active markers in the environment are a common source of ambiguity in onboard visual relative localization. This work presents a novel approach for onboard relative localization in multi-robot teams that exploits these typically unwanted reflections of active markers in the environment. It operates without prior knowledge of robot size or predefined marker configurations and remains independent of surface properties, an essential feature for heterogeneous micro-aerial swarms cooperating in unknown environments. It explicitly accounts for uncertainties caused by non-flat surfaces, with a particular focus on dynamic water surfaces, which are especially relevant for marine deployments. We validated the approach in both indoor and outdoor experiments, demonstrating that the proposed reflection-based localization system operates reliably without prior knowledge of team member size and achieves greater effective range (above 30 m) and accuracy than state-of-the-art methods. The video and source code of this work will be made publicly available after publication. 

**Abstract (ZH)**: 环境中的主动标记反射是机载视觉相对定位中常见的来源模糊。本文提出了一种新颖的方法，用于在多机器人团队中进行机载相对定位，该方法利用了环境中通常被视为不必要的主动标记的反射。该方法无需预先知道机器人尺寸或预定义的标记配置，并且独立于表面特性，这是异质微空中飞行群在未知环境中合作的一项重要特征。该方法明确考虑了由非平坦表面引起的不确定性，并特别关注动态水表面，这对海洋部署尤其相关。我们在室内和室外实验中验证了该方法，结果表明，所提出的基于反射的定位系统在无需了解团队成员尺寸的情况下可靠地运行，并实现了比最新方法更大的有效范围（超过30米）和更高的精度。该工作的视频和源代码将在发表后公开。 

---
# Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation 

**Title (ZH)**: Progress-Think: 语义进步推理在视觉语言导航中的应用 

**Authors**: Shuo Wang, Yucheng Wang, Guoxin Lian, Yongcai Wang, Maiyue Chen, Kaihui Wang, Bo Zhang, Zhizhong Su, Yutian Zhou, Wanting Li, Deying Li, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.17097)  

**Abstract**: Vision-Language Navigation requires agents to act coherently over long horizons by understanding not only local visual context but also how far they have advanced within a multi-step instruction. However, recent Vision-Language-Action models focus on direct action prediction and earlier progress methods predict numeric achievements; both overlook the monotonic co-progression property of the observation and instruction sequences. Building on this insight, Progress-Think introduces semantic progress reasoning, predicting instruction-style progress from visual observations to enable more accurate navigation. To achieve this without expensive annotations, we propose a three-stage framework. In the initial stage, Self-Aligned Progress Pretraining bootstraps a reasoning module via a novel differentiable alignment between visual history and instruction prefixes. Then, Progress-Guided Policy Pretraining injects learned progress states into the navigation context, guiding the policy toward consistent actions. Finally, Progress-Policy Co-Finetuning jointly optimizes both modules with tailored progress-aware reinforcement objectives. Experiments on R2R-CE and RxR-CE show state-of-the-art success and efficiency, demonstrating that semantic progress yields a more consistent representation of navigation advancement. 

**Abstract (ZH)**: Vision-Language Navigation 需要代理在长时间范围内一致地行动，不仅理解局部视觉上下文，还理解自己在多步指令中的进展。然而，最近的 Vision-Language-Action 模型专注于直接动作预测，而早期的方法预测数值成就；两者都忽视了观察和指令序列的单调共同进步特性。基于这一洞见，Progress-Think 引入了语义进展推理，预测从视觉观察到指令风格进展，以实现更准确的导航。为了在不使用昂贵标注的情况下实现这一点，我们提出了一种三阶段框架。在初始阶段，Self-Aligned Progress Pretraining 通过新颖的视觉历史与指令前缀的可微对齐来引导推理模块。随后，Progress-Guided Policy Pretraining 将学习到的进展状态注入导航上下文，引导策略采取一致的动作。最后，Progress-Policy Co-Finetuning 通过定制化的进展意识强化学习目标联合优化两个模块。在 R2R-CE 和 RxR-CE 上的实验显示了最先进的成功率和效率，证明了语义进展为导航进展提供了更一致的表示。 

---
# H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation 

**Title (ZH)**: H-GAR：一种基于目标驱动的观察-动作 refinement 的层级交互框架用于机器人操作 

**Authors**: Yijie Zhu, Rui Shao, Ziyang Liu, Jie He, Jizhihui Liu, Jiuru Wang, Zitong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17079)  

**Abstract**: Unified video and action prediction models hold great potential for robotic manipulation, as future observations offer contextual cues for planning, while actions reveal how interactions shape the environment. However, most existing approaches treat observation and action generation in a monolithic and goal-agnostic manner, often leading to semantically misaligned predictions and incoherent behaviors. To this end, we propose H-GAR, a Hierarchical interaction framework via Goal-driven observation-Action this http URL anchor prediction to the task objective, H-GAR first produces a goal observation and a coarse action sketch that outline a high-level route toward the goal. To enable explicit interaction between observation and action under the guidance of the goal observation for more coherent decision-making, we devise two synergistic modules. (1) Goal-Conditioned Observation Synthesizer (GOS) synthesizes intermediate observations based on the coarse-grained actions and the predicted goal observation. (2) Interaction-Aware Action Refiner (IAAR) refines coarse actions into fine-grained, goal-consistent actions by leveraging feedback from the intermediate observations and a Historical Action Memory Bank that encodes prior actions to ensure temporal consistency. By integrating goal grounding with explicit action-observation interaction in a coarse-to-fine manner, H-GAR enables more accurate manipulation. Extensive experiments on both simulation and real-world robotic manipulation tasks demonstrate that H-GAR achieves state-of-the-art performance. 

**Abstract (ZH)**: 基于目标驱动的层次交互框架H-GAR：通过观察-动作合成与交互意识的动作精炼实现统一的视频和动作预测 

---
# MfNeuPAN: Proactive End-to-End Navigation in Dynamic Environments via Direct Multi-Frame Point Constraints 

**Title (ZH)**: MfNeuPAN: 面向动态环境的直接多帧点约束端到端主动导航 

**Authors**: Yiwen Ying, Hanjing Ye, Senzi Luo, Luyao Liu, Yu Zhan, Li He, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17013)  

**Abstract**: Obstacle avoidance in complex and dynamic environments is a critical challenge for real-time robot navigation. Model-based and learning-based methods often fail in highly dynamic scenarios because traditional methods assume a static environment and cannot adapt to real-time changes, while learning-based methods rely on single-frame observations for motion constraint estimation, limiting their adaptability. To overcome these limitations, this paper proposes a novel framework that leverages multi-frame point constraints, including current and future frames predicted by a dedicated module, to enable proactive end-to-end navigation. By incorporating a prediction module that forecasts the future path of moving obstacles based on multi-frame observations, our method allows the robot to proactively anticipate and avoid potential dangers. This proactive planning capability significantly enhances navigation robustness and efficiency in unknown dynamic environments. Simulations and real-world experiments validate the effectiveness of our approach. 

**Abstract (ZH)**: 复杂和动态环境下基于多帧点约束的障碍物规避方法 

---
# Stable Offline Hand-Eye Calibration for any Robot with Just One Mark 

**Title (ZH)**: 仅使用一个标记的任意机器人离线手眼标定方法 

**Authors**: Sicheng Xie, Lingchen Meng, Zhiying Du, Shuyuan Tu, Haidong Cao, Jiaqi Leng, Zuxuan Wu, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17001)  

**Abstract**: Imitation learning has achieved remarkable success in a variety of robotic tasks by learning a mapping function from camera-space observations to robot-space actions. Recent work indicates that the use of robot-to-camera transformation information ({\ie}, camera extrinsics) benefits the learning process and produces better results. However, camera extrinsics are oftentimes unavailable and estimation methods usually suffer from local minima and poor generalizations. In this paper, we present CalibAll, a simple yet effective method that \textbf{requires only a single mark} and performs training-free, stable, and accurate camera extrinsic estimation across diverse robots and datasets through a coarse-to-fine calibration pipeline. In particular, we annotate a single mark on an end-effector (EEF), and leverage the correspondence ability emerged from vision foundation models (VFM) to automatically localize the corresponding mark across robots in diverse datasets. Using this mark, together with point tracking and the 3D EEF trajectory, we obtain a coarse camera extrinsic via temporal Perspective-n-Point (PnP). This estimate is further refined through a rendering-based optimization that aligns rendered and ground-true masks, yielding accurate and stable camera extrinsic. Experimental results demonstrate that our method outperforms state-of-the-art approaches, showing strong robustness and general effectiveness across three robot platforms. It also produces useful auxiliary annotations such as depth maps, link-wise masks, and end-effector 2D trajectories, which can further support downstream tasks. 

**Abstract (ZH)**: 基于单标记的无需训练的机器人摄像头外参精细标定方法 

---
# MobileOcc: A Human-Aware Semantic Occupancy Dataset for Mobile Robots 

**Title (ZH)**: MobileOcc：面向移动机器人的具有人类意识的语义占有数据集 

**Authors**: Junseo Kim, Guido Dumont, Xinyu Gao, Gang Chen, Holger Caesar, Javier Alonso-Mora  

**Link**: [PDF](https://arxiv.org/pdf/2511.16949)  

**Abstract**: Dense 3D semantic occupancy perception is critical for mobile robots operating in pedestrian-rich environments, yet it remains underexplored compared to its application in autonomous driving. To address this gap, we present MobileOcc, a semantic occupancy dataset for mobile robots operating in crowded human environments. Our dataset is built using an annotation pipeline that incorporates static object occupancy annotations and a novel mesh optimization framework explicitly designed for human occupancy modeling. It reconstructs deformable human geometry from 2D images and subsequently refines and optimizes it using associated LiDAR point data. Using MobileOcc, we establish benchmarks for two tasks, i) Occupancy prediction and ii) Pedestrian velocity prediction, using different methods including monocular, stereo, and panoptic occupancy, with metrics and baseline implementations for reproducible comparison. Beyond occupancy prediction, we further assess our annotation method on 3D human pose estimation datasets. Results demonstrate that our method exhibits robust performance across different datasets. 

**Abstract (ZH)**: 密集的三维语义占用感知对于在行人密集环境中操作的移动机器人至关重要，但与其实现自主驾驶的应用相比，这一领域仍处于探索不足的状态。为弥补这一差距，我们提出了MobileOcc，这是一种用于在拥挤的人群环境中操作的移动机器人的语义占用数据集。该数据集使用一个注释管道构建，该管道结合了静态物体占用标注和一种专门为人类占用建模设计的新型网格优化框架。它从2D图像中重建可变形的人类几何结构，然后使用关联的LiDAR点数据对其进行细化和优化。利用MobileOcc，我们使用不同的方法（包括单目、立体和全景占用）对两种任务进行基准测试，即占用预测和行人速度预测，并提供可重现比较的度量和基准实现。除了占用预测外，我们还进一步评估了我们的注释方法在三维人体姿态估计数据集上的表现。结果表明，我们的方法在不同数据集上表现出稳健的性能。 

---
# Multi-UAV Swarm Obstacle Avoidance Based on Potential Field Optimization 

**Title (ZH)**: 基于potential field优化的多无人机群障碍避让 

**Authors**: Yendo Hu, Yiliang Wu, Weican Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16911)  

**Abstract**: In multi UAV scenarios,the traditional Artificial Potential Field (APF) method often leads to redundant flight paths and frequent abrupt heading changes due to unreasonable obstacle avoidance path planning,and is highly prone to inter UAV collisions during the obstacle avoidance this http URL address these issues,this study proposes a novel hybrid algorithm that combines the improved Multi-Robot Formation Obstacle Avoidance (MRF IAPF) algorithm with an enhanced APF optimized for single UAV path this http URL core ideas are as follows:first,integrating three types of interaction forces from MRF IAPF obstacle repulsion force,inter UAV interaction force,and target attraction force;second,incorporating a refined single UAV path optimization mechanism,including collision risk assessment and an auxiliary sub goal this http URL a UAV faces a high collision threat,temporary waypoints are generated to guide obstacle avoidance,ensuring eventual precise arrival at the actual this http URL results demonstrate that compared with traditional APF based formation algorithms,the proposed algorithm achieves significant improvements in path length optimization and heading stability,can effectively avoid obstacles and quickly restore the formation configuration,thus verifying its applicability and effectiveness in static environments with unknown obstacles. 

**Abstract (ZH)**: 在多无人机场景中，传统的潜在场方法由于不合理的目标避障路径规划往往导致多余的飞行路径和频繁的突然航向改变，并且在目标避障过程中容易发生无人机之间的碰撞。为了应对这些问题，本研究提出了一种新的混合算法，该算法结合了改进的多机器人编队避障(MRF IAPF)算法与针对单无人机路径优化的增强潜在场方法。该算法的核心思想包括：首先，整合从MRF IAPF中提取的三种类型交互力，包括目标排斥力、无人机间交互力和目标吸引力；其次，融入了更为精细的单无人机路径优化机制，包括碰撞风险评估和辅助子目标生成。当无人机面临高碰撞威胁时，生成临时航点以引导避障，确保最终精确到达实际目标。研究结果表明，与基于传统潜在场的编队算法相比，所提出算法在路径长度优化和航向稳定性方面取得了显著改进，能够有效避障并快速恢复编队配置，从而验证了其在具有未知障碍的静态环境中的适用性和有效性。 

---
# Single-Pixel Tactile Skin via Compressive Sampling 

**Title (ZH)**: 单像素触觉皮肤基于压缩采样 

**Authors**: Ariel Slepyan, Laura Xing, Rudy Zhang, Nitish Thakor  

**Link**: [PDF](https://arxiv.org/pdf/2511.16898)  

**Abstract**: Development of large-area, high-speed electronic skins is a grand challenge for robotics, prosthetics, and human-machine interfaces, but is fundamentally limited by wiring complexity and data bottlenecks. Here, we introduce Single-Pixel Tactile Skin (SPTS), a paradigm that uses compressive sampling to reconstruct rich tactile information from an entire sensor array via a single output channel. This is achieved through a direct circuit-level implementation where each sensing element, equipped with a miniature microcontroller, contributes a dynamically weighted analog signal to a global sum, performing distributed compressed sensing in hardware. Our flexible, daisy-chainable design simplifies wiring to a few input lines and one output, and significantly reduces measurement requirements compared to raster scanning methods. We demonstrate the system's performance by achieving object classification at an effective 3500 FPS and by capturing transient dynamics, resolving an 8 ms projectile impact into 23 frames. A key feature is the support for adaptive reconstruction, where sensing fidelity scales with measurement time. This allows for rapid contact localization using as little as 7% of total data, followed by progressive refinement to a high-fidelity image - a capability critical for responsive robotic systems. This work offers an efficient pathway towards large-scale tactile intelligence for robotics and human-machine interfaces. 

**Abstract (ZH)**: 单像素触觉皮肤：通过压缩采样实现大规模高速触觉信息重建 

---
# A*-based Temporal Logic Path Planning with User Preferences on Relaxed Task Satisfaction 

**Title (ZH)**: 基于A*的时序逻辑路径规划方法，考虑用户对任务满意度的放宽偏好 

**Authors**: Disha Kamale, Xi Yu, Cristian-Ioan Vasile  

**Link**: [PDF](https://arxiv.org/pdf/2511.16844)  

**Abstract**: In this work, we consider the problem of planning for temporal logic tasks in large robot environments. When full task compliance is unattainable, we aim to achieve the best possible task satisfaction by integrating user preferences for relaxation into the planning process. Utilizing the automata-based representations for temporal logic goals and user preferences, we propose an A*-based planning framework. This approach effectively tackles large-scale problems while generating near-optimal high-level trajectories. To facilitate this, we propose a simple, efficient heuristic that allows for planning over large robot environments in a fraction of time and search memory as compared to uninformed search algorithms. We present extensive case studies to demonstrate the scalability, runtime analysis as well as empirical bounds on the suboptimality of the proposed heuristic. 

**Abstract (ZH)**: 在大型机器人环境中的时序逻辑任务规划：集成用户偏好实现最佳任务满意度 

---
# QueryOcc: Query-based Self-Supervision for 3D Semantic Occupancy 

**Title (ZH)**: 基于查询的自监督三维语义占用表示：QueryOcc 

**Authors**: Adam Lilja, Ji Lan, Junsheng Fu, Lars Hammarstrand  

**Link**: [PDF](https://arxiv.org/pdf/2511.17221)  

**Abstract**: Learning 3D scene geometry and semantics from images is a core challenge in computer vision and a key capability for autonomous driving. Since large-scale 3D annotation is prohibitively expensive, recent work explores self-supervised learning directly from sensor data without manual labels. Existing approaches either rely on 2D rendering consistency, where 3D structure emerges only implicitly, or on discretized voxel grids from accumulated lidar point clouds, limiting spatial precision and scalability. We introduce QueryOcc, a query-based self-supervised framework that learns continuous 3D semantic occupancy directly through independent 4D spatio-temporal queries sampled across adjacent frames. The framework supports supervision from either pseudo-point clouds derived from vision foundation models or raw lidar data. To enable long-range supervision and reasoning under constant memory, we introduce a contractive scene representation that preserves near-field detail while smoothly compressing distant regions. QueryOcc surpasses previous camera-based methods by 26% in semantic RayIoU on the self-supervised Occ3D-nuScenes benchmark while running at 11.6 FPS, demonstrating that direct 4D query supervision enables strong self-supervised occupancy learning. this https URL 

**Abstract (ZH)**: 从图像中学习3D场景几何结构和语义是计算机视觉中的一个核心挑战，也是自动驾驶的关键能力。由于大规模3D注解成本高昂，近期的工作探索直接从传感器数据中进行自我监督学习，而无需人工标签。现有方法要么依赖于2D渲染一致性，其中3D结构只能隐式出现，要么基于累积的激光雷达点云离散体素网格，限制了空间精度和可扩展性。我们引入了QueryOcc，这是一种基于查询的自我监督框架，通过跨相邻帧独立的4D时空查询直接学习连续的3D语义占有。该框架可以从从视觉基础模型派生的伪点云或原始激光雷达数据中获得监督。为了在常驻内存下实现长距离监督和推理，我们引入了一种收缩场景表示，该表示保持近场细节的同时平滑压缩远地区域。QueryOcc在自我监督的Occ3D-nuScenes基准测试中，在语义RayIoU上的性能超过了先前基于相机的方法26%，并在运行速度上达到了11.6 FPS，表明直接的4D查询监督能够实现强大的自我监督占有学习。 

---
# SING3R-SLAM: Submap-based Indoor Monocular Gaussian SLAM with 3D Reconstruction Priors 

**Title (ZH)**: SING3R-SLAM：基于子地图的室内单目高斯SLAM与三维重建先验 

**Authors**: Kunyi Li, Michael Niemeyer, Sen Wang, Stefano Gasperini, Nassir Navab, Federico Tombari  

**Link**: [PDF](https://arxiv.org/pdf/2511.17207)  

**Abstract**: Recent advances in dense 3D reconstruction enable the accurate capture of local geometry; however, integrating them into SLAM is challenging due to drift and redundant point maps, which limit efficiency and downstream tasks, such as novel view synthesis. To address these issues, we propose SING3R-SLAM, a globally consistent and compact Gaussian-based dense RGB SLAM framework. The key idea is to combine locally consistent 3D reconstructions with a unified global Gaussian representation that jointly refines scene geometry and camera poses, enabling efficient and versatile 3D mapping for multiple downstream applications. SING3R-SLAM first builds locally consistent submaps through our lightweight tracking and reconstruction module, and then progressively aligns and fuses them into a global Gaussian map that enforces cross-view geometric consistency. This global map, in turn, provides feedback to correct local drift and enhance the robustness of tracking. Extensive experiments demonstrate that SING3R-SLAM achieves state-of-the-art tracking, 3D reconstruction, and novel view rendering, resulting in over 12% improvement in tracking and producing finer, more detailed geometry, all while maintaining a compact and memory-efficient global representation on real-world datasets. 

**Abstract (ZH)**: 近期密集三维重建的进展使得局部几何的准确捕获成为可能；然而，将其集成到SLAM中由于漂移和冗余点云图的挑战，限制了效率和下游任务，如新型视图合成。为解决这些问题，我们提出了一种全局一致且紧凑的高斯为基础的密集RGB SLAM框架SING3R-SLAM。核心思想是将局部一致的三维重建与统一的全局高斯表示相结合，共同精化场景几何和相机姿态，从而实现高效的多功能三维建图以支持多种下游应用。SING3R-SLAM首先通过轻量级跟踪与重建模块构建局部一致的子地图，并逐步将其对齐并融合到一个强制跨视图几何一致性的全局高斯地图中。该全局地图反过来提供了反馈以纠正局部漂移并增强跟踪的鲁棒性。广泛实验表明，SING3R-SLAM在跟踪、三维重建和新型视图渲染方面均达到了最先进的水平，在实际数据集上实现了超过12%的跟踪提升，并生成更精细、更详细的几何结构，同时保持了一个紧凑且节省内存的全局表示。 

---
# A segment anchoring-based balancing algorithm for agricultural multi-robot task allocation with energy constraints 

**Title (ZH)**: 基于段锚定的能源约束农业多机器人任务分配平衡算法 

**Authors**: Peng Chen, Jing Liang, Kang-Jia Qiao, Hui Song, Tian-lei Ma, Kun-Jie Yu, Cai-Tong Yue, Ponnuthurai Nagaratnam Suganthan, Witold Pedryc  

**Link**: [PDF](https://arxiv.org/pdf/2511.17076)  

**Abstract**: Multi-robot systems have emerged as a key technology for addressing the efficiency and cost challenges in labor-intensive industries. In the representative scenario of smart farming, planning efficient harvesting schedules for a fleet of electric robots presents a highly challenging frontier problem. The complexity arises not only from the need to find Pareto-optimal solutions for the conflicting objectives of makespan and transportation cost, but also from the necessity to simultaneously manage payload constraints and finite battery capacity. When robot loads are dynamically updated during planned multi-trip operations, a mandatory recharge triggered by energy constraints introduces an unscheduled load reset. This interaction creates a complex cascading effect that disrupts the entire schedule and renders traditional optimization methods ineffective. To address this challenge, this paper proposes the segment anchoring-based balancing algorithm (SABA). The core of SABA lies in the organic combination of two synergistic mechanisms: the sequential anchoring and balancing mechanism, which leverages charging decisions as `anchors' to systematically reconstruct disrupted routes, while the proportional splitting-based rebalancing mechanism is responsible for the fine-grained balancing and tuning of the final solutions' makespans. Extensive comparative experiments, conducted on a real-world case study and a suite of benchmark instances, demonstrate that SABA comprehensively outperforms 6 state-of-the-art algorithms in terms of both solution convergence and diversity. This research provides a novel theoretical perspective and an effective solution for the multi-robot task allocation problem under energy constraints. 

**Abstract (ZH)**: 多机器人系统在劳动密集型行业中的效率和成本挑战中 emerged as a key technology 也被视为一种关键技术。在智能农业的典型场景中，为智能电机器人队列规划高效的采收时间表 presents a highly challenging frontier problem 成为一个高度具有挑战性的前沿问题。SABA 基于区间锚定的平衡算法 proposes the segment anchoring-based balancing algorithm (SABA) 提出了区间锚定平衡算法 (SABA)。SABA 的核心 lies in the organic combination of two synergistic mechanisms 在于两种协同机制的有机结合：基于顺序锚定和平衡的机制，通过充电决策作为“锚”系统地重构被破坏的路线，而基于比例分割的再平衡机制则负责最终解决方案的历时精细平衡和调节。本研究通过在真实案例和一系列基准实例上进行的广泛比较实验，证明 SABA 在解的收敛性和多样性方面全面优于 6 种最先进的算法。所提供的理论视角和解决方案为带能源约束下的多机器人任务分配问题提供了新的有效方法。 

---
# BOP-ASK: Object-Interaction Reasoning for Vision-Language Models 

**Title (ZH)**: BOP-ASK：物体交互推理对于视觉语言模型 

**Authors**: Vineet Bhat, Sungsu Kim, Valts Blukis, Greg Heinrich, Prashanth Krishnamurthy, Ramesh Karri, Stan Birchfield, Farshad Khorrami, Jonathan Tremblay  

**Link**: [PDF](https://arxiv.org/pdf/2511.16857)  

**Abstract**: Vision Language Models (VLMs) have achieved impressive performance on spatial reasoning benchmarks, yet these evaluations mask critical weaknesses in understanding object interactions. Current benchmarks test high level relationships ('left of,' 'behind', etc.) but ignore fine-grained spatial understanding needed for real world applications: precise 3D localization, physical compatibility between objects, object affordances and multi step spatial planning. In this work, we present BOP-ASK, a novel large scale dataset for object interaction reasoning for both training and benchmarking. Our data generation pipeline leverages 6D object poses from the Benchmark for Object Pose Estimation (BOP) datasets from which we derive fine grained annotations such as grasp poses, referred object poses, path planning trajectories, relative spatial and depth relationships, and object-to-object relationships. BOP-ASK comprises over 150k images and 33M question answer pairs spanning six tasks (four novel), providing a rich resource for training and evaluating VLMs. We evaluate proprietary and open sourced VLMs, and conduct human evaluations on BOP-ASK-core, a contributed test benchmark. We also release BOP-ASK-lab, an out-of-distribution benchmark with images not sourced from BOP, enabling testing of generalization. Our experiments demonstrate that models trained on BOP-ASK outperform baselines and exhibit emergent capabilities such as precise object and grasp pose estimation, trajectory planning, and fine-grained object-centric spatial reasoning in cluttered environments. We will publicly release our datasets and dataset generation pipeline. 

**Abstract (ZH)**: 物体交互推理的大规模数据集BOP-ASK 

---
