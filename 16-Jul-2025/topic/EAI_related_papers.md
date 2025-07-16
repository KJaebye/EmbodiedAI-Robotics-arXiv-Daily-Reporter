# Human-Robot collaboration in surgery: Advances and challenges towards autonomous surgical assistants 

**Title (ZH)**: 手术中的人机协作：自主外科助手的发展与挑战 

**Authors**: Jacinto Colan, Ana Davila, Yutaro Yamada, Yasuhisa Hasegawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.11460)  

**Abstract**: Human-robot collaboration in surgery represents a significant area of research, driven by the increasing capability of autonomous robotic systems to assist surgeons in complex procedures. This systematic review examines the advancements and persistent challenges in the development of autonomous surgical robotic assistants (ASARs), focusing specifically on scenarios where robots provide meaningful and active support to human surgeons. Adhering to the PRISMA guidelines, a comprehensive literature search was conducted across the IEEE Xplore, Scopus, and Web of Science databases, resulting in the selection of 32 studies for detailed analysis. Two primary collaborative setups were identified: teleoperation-based assistance and direct hands-on interaction. The findings reveal a growing research emphasis on ASARs, with predominant applications currently in endoscope guidance, alongside emerging progress in autonomous tool manipulation. Several key challenges hinder wider adoption, including the alignment of robotic actions with human surgeon preferences, the necessity for procedural awareness within autonomous systems, the establishment of seamless human-robot information exchange, and the complexities of skill acquisition in shared workspaces. This review synthesizes current trends, identifies critical limitations, and outlines future research directions essential to improve the reliability, safety, and effectiveness of human-robot collaboration in surgical environments. 

**Abstract (ZH)**: 手术中的人机协作代表了一个重要的研究领域，由自主机器人系统能力的提升推动，这些系统能够协助外科医生进行复杂的手术程序。本系统综述基于PRISMA指南，考察了自主手术机器人助手（ASARs）的发展进步和持续挑战，重点关注机器人为外科医生提供有意义和主动支持的场景。研究识别了两种主要的协作模式：基于遥控的操作协助和直接的手动交互。研究结果揭示了对ASARs研究重点的增长，目前主要应用在内窥镜引导，并且在自主工具操作方面取得了新兴进展。阻碍更广泛采用的关键挑战包括机器人动作与外科医生偏好的对齐、自主系统内的程序意识需求、人机信息交换的无缝链接，以及共享工作空间中技能获取的复杂性。本综述汇总了当前趋势，指出了关键限制，并勾勒了未来研究方向，以提高手术环境中人机协作的可靠性和有效性。 

---
# All Eyes, no IMU: Learning Flight Attitude from Vision Alone 

**Title (ZH)**: 全视觉，无IMU：仅从视觉学习飞行姿态 

**Authors**: Jesse J. Hagenaars, Stein Stroobants, Sander M. Bohte, Guido C.H.E. De Croon  

**Link**: [PDF](https://arxiv.org/pdf/2507.11302)  

**Abstract**: Vision is an essential part of attitude control for many flying animals, some of which have no dedicated sense of gravity. Flying robots, on the other hand, typically depend heavily on accelerometers and gyroscopes for attitude stabilization. In this work, we present the first vision-only approach to flight control for use in generic environments. We show that a quadrotor drone equipped with a downward-facing event camera can estimate its attitude and rotation rate from just the event stream, enabling flight control without inertial sensors. Our approach uses a small recurrent convolutional neural network trained through supervised learning. Real-world flight tests demonstrate that our combination of event camera and low-latency neural network is capable of replacing the inertial measurement unit in a traditional flight control loop. Furthermore, we investigate the network's generalization across different environments, and the impact of memory and different fields of view. While networks with memory and access to horizon-like visual cues achieve best performance, variants with a narrower field of view achieve better relative generalization. Our work showcases vision-only flight control as a promising candidate for enabling autonomous, insect-scale flying robots. 

**Abstract (ZH)**: 视觉导向的飞行控制：一种用于通用环境的纯视觉方法 

---
# Ocean Diviner: A Diffusion-Augmented Reinforcement Learning for AUV Robust Control in the Underwater Tasks 

**Title (ZH)**: Ocean Diviner: 基于扩散增强 reinforcement learning的自治水下车辆水下任务稳健控制方法 

**Authors**: Weiyi Liu, Jingzehua Xu, Guanwen Xie, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.11283)  

**Abstract**: This paper presents a diffusion-augmented reinforcement learning (RL) approach for robust autonomous underwater vehicle (AUV) control, addressing key challenges in underwater trajectory planning and dynamic environment adaptation. The proposed method integrates three core innovations: (1) A diffusion-based trajectory generation framework that produces physically feasible multi-step trajectories, enhanced by a high-dimensional state encoding mechanism combining current observations with historical states and actions through a novel diffusion U-Net architecture, significantly improving long-horizon planning. (2) A sample-efficient hybrid learning architecture that synergizes diffusion-guided exploration with RL policy optimization, where the diffusion model generates diverse candidate actions and the RL critic selects optimal actions, achieving higher exploration efficiency and policy stability in dynamic underwater environments. Extensive simulation experiments validating the method's superior robustness and flexibility, outperforms conventional control methods in challenging marine conditions, offering enhanced adaptability and reliability for AUV operations in the underwater tasks. 

**Abstract (ZH)**: 本文提出了一种扩散增强 reinforcement learning (RL) 方法，用于鲁棒自主水下车辆 (AUV) 控制，解决水下轨迹规划和动态环境适应的关键挑战。该提出的办法整合了三项核心创新：（1）一种基于扩散的轨迹生成框架，能够生成物理上可行的多步轨迹，并通过新颖的扩散 U-Net 架构结合当前观察和历史状态与动作的高维状态编码机制显著提高长远规划能力。（2）一种样本高效混合学习架构，协同利用扩散引导探索与 RL 策略优化，其中扩散模型生成多样化的候选动作，RL 评论家选择最优动作，在动态水下环境中实现更高的探索效率和策略稳定性。广泛的仿真实验验证了该方法的优越鲁棒性和灵活性，在恶劣海洋条件下优于传统控制方法，为 AUV 在水下任务中的操作提供增强的适应性和可靠性。 

---
# Learning to Tune Like an Expert: Interpretable and Scene-Aware Navigation via MLLM Reasoning and CVAE-Based Adaptation 

**Title (ZH)**: 像专家一样调整：基于MLLM推理和CVAE基适应的可解释且场景感知的导航 

**Authors**: Yanbo Wang, Zipeng Fang, Lei Zhao, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11001)  

**Abstract**: Service robots are increasingly deployed in diverse and dynamic environments, where both physical layouts and social contexts change over time and across locations. In these unstructured settings, conventional navigation systems that rely on fixed parameters often fail to generalize across scenarios, resulting in degraded performance and reduced social acceptance. Although recent approaches have leveraged reinforcement learning to enhance traditional planners, these methods often fail in real-world deployments due to poor generalization and limited simulation diversity, which hampers effective sim-to-real transfer. To tackle these issues, we present LE-Nav, an interpretable and scene-aware navigation framework that leverages multi-modal large language model reasoning and conditional variational autoencoders to adaptively tune planner hyperparameters. To achieve zero-shot scene understanding, we utilize one-shot exemplars and chain-of-thought prompting strategies. Additionally, a conditional variational autoencoder captures the mapping between natural language instructions and navigation hyperparameters, enabling expert-level tuning. Experiments show that LE-Nav can generate hyperparameters achieving human-level tuning across diverse planners and scenarios. Real-world navigation trials and a user study on a smart wheelchair platform demonstrate that it outperforms state-of-the-art methods on quantitative metrics such as success rate, efficiency, safety, and comfort, while receiving higher subjective scores for perceived safety and social acceptance. Code is available at this https URL. 

**Abstract (ZH)**: 基于多模态大语言模型推理和服务变分自动编码器的可解释场景感知导航框架LE-Nav 

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
# Object-Centric Mobile Manipulation through SAM2-Guided Perception and Imitation Learning 

**Title (ZH)**: 基于SAM2引导的感知与imitation学习的物体为中心的移动操作Manipulation 

**Authors**: Wang Zhicheng, Satoshi Yagi, Satoshi Yamamori, Jun Morimoto  

**Link**: [PDF](https://arxiv.org/pdf/2507.10899)  

**Abstract**: Imitation learning for mobile manipulation is a key challenge in the field of robotic manipulation. However, current mobile manipulation frameworks typically decouple navigation and manipulation, executing manipulation only after reaching a certain location. This can lead to performance degradation when navigation is imprecise, especially due to misalignment in approach angles. To enable a mobile manipulator to perform the same task from diverse orientations, an essential capability for building general-purpose robotic models, we propose an object-centric method based on SAM2, a foundation model towards solving promptable visual segmentation in images, which incorporates manipulation orientation information into our model. Our approach enables consistent understanding of the same task from different orientations. We deploy the model on a custom-built mobile manipulator and evaluate it on a pick-and-place task under varied orientation angles. Compared to Action Chunking Transformer, our model maintains superior generalization when trained with demonstrations from varied approach angles. This work significantly enhances the generalization and robustness of imitation learning-based mobile manipulation systems. 

**Abstract (ZH)**: 基于SAM2的目标为中心的方法在移动操作中的模仿学习 

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

**Title (ZH)**: 学习 rhythm 中的运动：具有轨道稳定保证的任务条件运动策略 

**Authors**: Maximilian Stölzle, T. Konstantin Rusch, Zach J. Patterson, Rodrigo Pérez-Dattari, Francesco Stella, Josie Hughes, Cosimo Della Santina, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2507.10602)  

**Abstract**: Learning from demonstration provides a sample-efficient approach to acquiring complex behaviors, enabling robots to move robustly, compliantly, and with fluidity. In this context, Dynamic Motion Primitives offer built - in stability and robustness to disturbances but often struggle to capture complex periodic behaviors. Moreover, they are limited in their ability to interpolate between different tasks. These shortcomings substantially narrow their applicability, excluding a wide class of practically meaningful tasks such as locomotion and rhythmic tool use. In this work, we introduce Orbitally Stable Motion Primitives (OSMPs) - a framework that combines a learned diffeomorphic encoder with a supercritical Hopf bifurcation in latent space, enabling the accurate acquisition of periodic motions from demonstrations while ensuring formal guarantees of orbital stability and transverse contraction. Furthermore, by conditioning the bijective encoder on the task, we enable a single learned policy to represent multiple motion objectives, yielding consistent zero-shot generalization to unseen motion objectives within the training distribution. We validate the proposed approach through extensive simulation and real-world experiments across a diverse range of robotic platforms - from collaborative arms and soft manipulators to a bio-inspired rigid-soft turtle robot - demonstrating its versatility and effectiveness in consistently outperforming state-of-the-art baselines such as diffusion policies, among others. 

**Abstract (ZH)**: 基于演示学习提供了高效样本获取复杂行为的方法，使机器人能够以稳定性、鲁棒性和流畅性移动。在此背景下，轨道稳定运动基元具有内置的稳定性和对干扰的鲁棒性，但往往难以捕捉复杂周期行为，且在不同任务之间的插值能力有限。这些不足极大地限制了其应用范围，排除了许多实际意义重大的任务，如运动和节律性工具使用。在本文中，我们提出了轨道稳定运动基元（OSMPs）——一种结合学习差分编码器与潜在空间超临界霍普极限环的框架，能够从演示中准确获取周期性运动，并确保轨道稳定性和横向收缩的正式保证。此外，通过将双射编码器与任务相关联，我们使得单一学习策略能够表示多个运动目标，在训练分布内的一次性泛化到未见过的运动目标。我们通过广泛的仿真实验和实际世界实验，在多种不同的机器人平台上验证了所提出的方法，从协作臂和软 manipulators 到生物启发的刚体软体龟机器人，展示了其多样性和有效性，其性能在与现有的先进技术如扩散策略等的对比中持续表现更优。 

---
# CogDDN: A Cognitive Demand-Driven Navigation with Decision Optimization and Dual-Process Thinking 

**Title (ZH)**: CogDDN：基于认知需求驱动的导航与决策优化及双过程思维方法 

**Authors**: Yuehao Huang, Liang Liu, Shuangming Lei, Yukai Ma, Hao Su, Jianbiao Mei, Pengxiang Zhao, Yaqing Gu, Yong Liu, Jiajun Lv  

**Link**: [PDF](https://arxiv.org/pdf/2507.11334)  

**Abstract**: Mobile robots are increasingly required to navigate and interact within unknown and unstructured environments to meet human demands. Demand-driven navigation (DDN) enables robots to identify and locate objects based on implicit human intent, even when object locations are unknown. However, traditional data-driven DDN methods rely on pre-collected data for model training and decision-making, limiting their generalization capability in unseen scenarios. In this paper, we propose CogDDN, a VLM-based framework that emulates the human cognitive and learning mechanisms by integrating fast and slow thinking systems and selectively identifying key objects essential to fulfilling user demands. CogDDN identifies appropriate target objects by semantically aligning detected objects with the given instructions. Furthermore, it incorporates a dual-process decision-making module, comprising a Heuristic Process for rapid, efficient decisions and an Analytic Process that analyzes past errors, accumulates them in a knowledge base, and continuously improves performance. Chain of Thought (CoT) reasoning strengthens the decision-making process. Extensive closed-loop evaluations on the AI2Thor simulator with the ProcThor dataset show that CogDDN outperforms single-view camera-only methods by 15%, demonstrating significant improvements in navigation accuracy and adaptability. The project page is available at this https URL. 

**Abstract (ZH)**: 基于VLM的认知驱动导航（CogDDN）框架：模拟人类认知与学习机制以实现需求导向的环境交互 

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

**Abstract (ZH)**: 离线强化学习（RL）旨在从静态数据集中学习最优策略，使其在数据收集成本较高的场景下（如机器人领域）尤为重要。离线RL的一个主要挑战是分布偏移，即学习到的策略与数据集分布不一致，可能导致无法泛化的动作。为缓解这一问题，引入了正则化技术。虽然许多现有方法采用基于密度比的度量，如$f$散度进行正则化，我们提出了一种利用Wasserstein距离的方法，Wasserstein距离对异常数据具有鲁棒性，并能够捕捉动作之间的相似性。该方法使用输入凸神经网络（ICNNs）来建模最优传输映射，通过鉴别器-Free的方式计算Wasserstein距离，从而避免 adversarial 训练并确保学习的稳定性。我们的方法在D4RL基准数据集上展示了与现有广泛使用的方法相当或更优的性能。代码可在以下链接获取：this https URL。 

---
# Illuminating the Three Dogmas of Reinforcement Learning under Evolutionary Light 

**Title (ZH)**: 在进化视角下揭示强化学习的三大 dogma 

**Authors**: Mani Hamidi, Terrence W. Deacon  

**Link**: [PDF](https://arxiv.org/pdf/2507.11482)  

**Abstract**: Three core tenets of reinforcement learning (RL)--concerning the definition of agency, the objective of learning, and the scope of the reward hypothesis--have been highlighted as key targets for conceptual revision, with major implications for theory and application. We propose a framework, inspired by open-ended evolutionary theory, to reconsider these three "dogmas." We revisit each assumption and address related concerns raised alongside them. To make our arguments relevant to RL as a model of biological learning, we first establish that evolutionary dynamics can plausibly operate within living brains over an individual's lifetime, and are not confined to cross-generational processes. We begin by revisiting the second dogma, drawing on evolutionary insights to enrich the "adaptation-rather-than-search" view of learning. We then address the third dogma regarding the limits of the reward hypothesis, using analogies from evolutionary fitness to illuminate the scalar reward vs. multi-objective debate. After discussing practical implications for exploration in RL, we turn to the first--and arguably most fundamental--issue: the absence of a formal account of agency. We argue that unlike the other two problems, the evolutionary paradigm alone cannot resolve the agency question, though it gestures in a productive direction. We advocate integrating ideas from origins-of-life theory, where the thermodynamics of sustenance and replication offer promising foundations for understanding agency and resource-constrained reinforcement learning in biological systems. 

**Abstract (ZH)**: 强化学习（RL）的三大核心原则——关于代理的定义、学习的目标以及奖励假设的范围——已被视为概念修订的关键目标，对理论和应用具有重要影响。我们提出一个框架，受到开放性进化理论的启发，重新考虑这三种“教条”。我们重新审视每项假设，并解决与它们相关的顾虑。为使我们的论点与RL作为生物学习模型相关，我们首先确立进化动力学在个体生命期内合理地在活脑中运行，而不局限于代际过程。我们首先重新审视第二个教条，借鉴进化生物学的见解，丰富“适应而非搜索”的学习视图。接着，我们解决关于奖励假设限度的第三个教条，利用进化适应性的类比来阐明标量奖励与多目标奖励之间的争论。在讨论探索在RL中的实际影响之后，我们转向第一个——或许是最重要的——问题：缺乏对代理的正式描述。我们认为，不同于其他两个问题，进化范式本身无法解决代理问题，但它指出了一个有成效的方向。我们倡导整合生命起源理论中的观点，其中维持和复制的热力学原理为理解代理和资源约束下的生物系统强化学习提供了有希望的基础。 

---
# Tactical Decision for Multi-UGV Confrontation with a Vision-Language Model-Based Commander 

**Title (ZH)**: 基于视觉-语言模型指挥官的多无人地面车辆战术决策 

**Authors**: Li Wang, Qizhen Wu, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11079)  

**Abstract**: In multiple unmanned ground vehicle confrontations, autonomously evolving multi-agent tactical decisions from situational awareness remain a significant challenge. Traditional handcraft rule-based methods become vulnerable in the complicated and transient battlefield environment, and current reinforcement learning methods mainly focus on action manipulation instead of strategic decisions due to lack of interpretability. Here, we propose a vision-language model-based commander to address the issue of intelligent perception-to-decision reasoning in autonomous confrontations. Our method integrates a vision language model for scene understanding and a lightweight large language model for strategic reasoning, achieving unified perception and decision within a shared semantic space, with strong adaptability and interpretability. Unlike rule-based search and reinforcement learning methods, the combination of the two modules establishes a full-chain process, reflecting the cognitive process of human commanders. Simulation and ablation experiments validate that the proposed approach achieves a win rate of over 80% compared with baseline models. 

**Abstract (ZH)**: 在多智能地面车辆对抗中，从态势感知自主演化多代理战术决策仍然是一项重大挑战。传统手工规则方法在复杂多变的战场环境中变得脆弱，当前的强化学习方法主要侧重于动作操控而非战略决策，缺乏可解释性。在此，我们提出一种基于视觉语言模型的指挥官，以解决自主对抗中的智能感知到决策推理问题。我们的方法结合视觉语言模型进行场景理解以及轻量级大型语言模型进行战略推理，在共享语义空间中实现统一的感知与决策，具有较强的适应性和可解释性。与基于规则的搜索和强化学习方法不同，两模块的结合建立了完整的链路过程，反映了人类指挥官的认知过程。仿真实验和消融实验验证了所提出的方法在基线模型上的胜率超过80%。 

---
# NavComposer: Composing Language Instructions for Navigation Trajectories through Action-Scene-Object Modularization 

**Title (ZH)**: NavComposer: 通过动作-场景-对象模块化组成导航指令的trajectory生成方法 

**Authors**: Zongtao He, Liuyi Wang, Lu Chen, Chengju Liu, Qijun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10894)  

**Abstract**: Language-guided navigation is a cornerstone of embodied AI, enabling agents to interpret language instructions and navigate complex environments. However, expert-provided instructions are limited in quantity, while synthesized annotations often lack quality, making them insufficient for large-scale research. To address this, we propose NavComposer, a novel framework for automatically generating high-quality navigation instructions. NavComposer explicitly decomposes semantic entities such as actions, scenes, and objects, and recomposes them into natural language instructions. Its modular architecture allows flexible integration of state-of-the-art techniques, while the explicit use of semantic entities enhances both the richness and accuracy of instructions. Moreover, it operates in a data-agnostic manner, supporting adaptation to diverse navigation trajectories without domain-specific training. Complementing NavComposer, we introduce NavInstrCritic, a comprehensive annotation-free evaluation system that assesses navigation instructions on three dimensions: contrastive matching, semantic consistency, and linguistic diversity. NavInstrCritic provides a holistic evaluation of instruction quality, addressing limitations of traditional metrics that rely heavily on expert annotations. By decoupling instruction generation and evaluation from specific navigation agents, our method enables more scalable and generalizable research. Extensive experiments provide direct and practical evidence for the effectiveness of our method. 

**Abstract (ZH)**: 基于语言的导航是实体AI的基石，使代理能够解析语言指令并导航复杂环境。然而，专家提供的指令数量有限，而合成的注释往往质量不足，无法满足大规模研究的需要。为了解决这一问题，我们提出NavComposer，一种自动生成高质量导航指令的新型框架。NavComposer明确分解了语义实体，如动作、场景和对象，并重新组合成自然语言指令。其模块化架构允许灵活集成当前最先进的技术，而明确使用语义实体增强了指令的丰富性和准确性。此外，它以数据无关的方式运行，支持针对多样化的导航轨迹进行适应，无需特定领域的训练。与NavComposer互补，我们引入了NavInstrCritic，这是一种全面的无注释评估系统，从对比匹配、语义一致性和语言多样性三个维度评估导航指令。NavInstrCritic提供了一种整体性的指令质量评估，克服了传统评估指标严重依赖专家注释的局限性。通过将指令生成和评估与特定导航代理解耦，我们的方法能够促进更大规模和更通用的研究。广泛的实验提供了直接且实用的有效性证据。 

---
# From Semantic Web and MAS to Agentic AI: A Unified Narrative of the Web of Agents 

**Title (ZH)**: 从语义 web 和MAS到有能动性的AI：代理网络中的统一叙事 

**Authors**: Tatiana Petrova, Aleksandr Puzikov, Boris Bliznukov, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2507.10644)  

**Abstract**: The concept of the Web of Agents (WoA), which transforms the static, document-centric Web into an environment of autonomous agents acting on users' behalf, has attracted growing interest as large language models (LLMs) become more capable. However, research in this area is still fragmented across different communities. Contemporary surveys catalog the latest LLM-powered frameworks, while the rich histories of Multi-Agent Systems (MAS) and the Semantic Web are often treated as separate, legacy domains. This fragmentation obscures the intellectual lineage of modern systems and hinders a holistic understanding of the field's trajectory. We present the first comprehensive evolutionary overview of the WoA. We show that modern protocols like A2A and the MCP, are direct evolutionary responses to the well-documented limitations of earlier standards like FIPA standards and OWL-based semantic agents. To systematize this analysis, we introduce a four-axis taxonomy (semantic foundation, communication paradigm, locus of intelligence, discovery mechanism). This framework provides a unified analytical lens for comparing agent architectures across all generations, revealing a clear line of descent where others have seen a disconnect. Our analysis identifies a paradigm shift in the 'locus of intelligence': from being encoded in external data (Semantic Web) or the platform (MAS) to being embedded within the agent's core model (LLM). This shift is foundational to modern Agentic AI, enabling the scalable and adaptive systems the WoA has long envisioned. We conclude that while new protocols are essential, they are insufficient for building a robust, open, trustworthy ecosystem. Finally, we argue that the next research frontier lies in solving persistent socio-technical challenges, and we map out a new agenda focused on decentralized identity, economic models, security, and governance for the emerging WoA. 

**Abstract (ZH)**: Web of Agents (WoA)的演化综述：从多代理系统到语义web的智能代理架构演变 

---
# Orchestrator-Agent Trust: A Modular Agentic AI Visual Classification System with Trust-Aware Orchestration and RAG-Based Reasoning 

**Title (ZH)**: Orchestrator-Agent Trust: 一种具有信任意识编排和基于RAG的推理的模块化代理AI视觉分类系统 

**Authors**: Konstantinos I. Roumeliotis, Ranjan Sapkota, Manoj Karkee, Nikolaos D. Tselikas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10571)  

**Abstract**: Modern Artificial Intelligence (AI) increasingly relies on multi-agent architectures that blend visual and language understanding. Yet, a pressing challenge remains: How can we trust these agents especially in zero-shot settings with no fine-tuning? We introduce a novel modular Agentic AI visual classification framework that integrates generalist multimodal agents with a non-visual reasoning orchestrator and a Retrieval-Augmented Generation (RAG) module. Applied to apple leaf disease diagnosis, we benchmark three configurations: (I) zero-shot with confidence-based orchestration, (II) fine-tuned agents with improved performance, and (III) trust-calibrated orchestration enhanced by CLIP-based image retrieval and re-evaluation loops. Using confidence calibration metrics (ECE, OCR, CCC), the orchestrator modulates trust across agents. Our results demonstrate a 77.94\% accuracy improvement in the zero-shot setting using trust-aware orchestration and RAG, achieving 85.63\% overall. GPT-4o showed better calibration, while Qwen-2.5-VL displayed overconfidence. Furthermore, image-RAG grounded predictions with visually similar cases, enabling correction of agent overconfidence via iterative re-evaluation. The proposed system separates perception (vision agents) from meta-reasoning (orchestrator), enabling scalable and interpretable multi-agent AI. This blueprint is extensible to diagnostics, biology, and other trust-critical domains. All models, prompts, results, and system components including the complete software source code are openly released to support reproducibility, transparency, and community benchmarking at Github: this https URL 

**Abstract (ZH)**: 现代人工智能（AI）越来越多地依赖融合视觉和语言理解的多agent架构。然而，一个紧迫的挑战依然存在：我们在零样本设置下，即没有任何微调的情况下，如何信任这些agent？我们提出了一种新颖的模块化Agentic AI视觉分类框架，该框架结合了一般主义多模态agent、非视觉推理协调器和检索增强生成（RAG）模块。应用于苹果叶病诊断，我们对三种配置进行了基准测试：（I）基于信心的零样本协调、（II）微调后的agent性能得到提升、（III）通过基于CLIP的图像检索和再评价循环增强的信任校准协调。使用信任校准指标（ECE、OCR、CCC），协调器在agent之间调整信任。我们的结果显示，在使用信任感知协调和RAG的零样本设置中，准确率提高了77.94%，整体达到85.63%。GPT-4o展示了更好的校准，而Qwen-2.5-VL则表现出过大的自信。此外，通过迭代再评价将图像-RAG与视觉上相似的案例结合起来，能够纠正agent的过自信。该系统将感知（视觉agent）与元推理（协调器）分离，使多agent AI具备可扩展性和可解释性。该蓝图可以扩展到诊断、生物学和其他依赖信任的领域。所有模型、提示、结果和系统组件包括完整的软件源代码均已在Github公开发布，以支持可重复性、透明性和社区基准测试：this https URL。 

---
# AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems 

**Title (ZH)**: AI母语：通过内生符号系统在多智能体 reinforcement 学习中实现自主沟通 

**Authors**: Hung Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10566)  

**Abstract**: In Decentralized Multi-Agent Reinforcement Learning (MARL), the development of Emergent Communication has long been constrained by the ``Joint Exploration Dilemma'', leading agents to fall into a ``Communication Vacuum Equilibrium'' . Traditional methods address this by introducing inductive biases to facilitate communication emergence . This study fundamentally questions whether such artificial inductive biases are, in fact, over-engineering. Through experiments with the ``AI Mother Tongue'' (AIM) framework, based on a Vector Quantized Variational Autoencoder (VQ-VAE), we demonstrate that when agents possess an endogenous symbol system, their neural representations naturally exhibit spontaneous semantic compression and Nash equilibrium-driven semantic convergence, achieving effective symbolic communication without external inductive biases. This aligns with recent neuroscience findings suggesting that the human brain does not directly use human language for internal thought , and resonates with research on ``soft thinking'' capabilities in Large Language Models (LLMs) . Compared to traditional explicit communication methods, AIM demonstrates stronger generality and efficiency. The interpretable analysis toolkit developed in this study confirms that symbol usage exhibits a significant power-law distribution, leading to three major theoretical insights: the ``Neural Communication Hypothesis'', the ``Tool-First Principle'', and the ``Semantic Interpretability Paradigm''. Future research will explore the integration of Hierarchical Quantized Variational Autoencoders (HQ-VAE) to enhance AIM's complex expressive capabilities and investigate the potential for ``Reinforcement Learning (RL) Low-Level Pre-training''. This discovery offers new avenues for bridging symbolism and connectionism. 

**Abstract (ZH)**: 在分布式多智能体 reinforcement 学习（MARL）中，自发通信的发展长期以来受到“联合探索困境”的限制，导致智能体陷入“通信真空均衡”。传统方法通过引入归纳偏置来促进通信的自发出现。本研究从根本上质疑这种人工归纳偏置是否是一种过度工程。通过基于向量量化变分自编码器（VQ-VAE）的“AI 母语”（AIM）框架的实验，我们证明，在智能体具备内生符号系统的情况下，它们的神经表示自然展现出自发语义压缩和纳什均衡驱动的意义收敛，能够实现有效的象征性通信，无需外部归纳偏置。这与近期的神经科学发现相一致，表明人类大脑不直接使用人类语言进行内部思考，并与大型语言模型（LLMs）的“软思考”能力研究相呼应。与传统的显式通信方法相比，AIM 展现出更强的通用性和效率。本研究开发的可解释分析工具包证实符号使用表现出显著的幂律分布，从而产生三个主要的理论洞察：“神经通信假说”、“工具优先原则”和“语义可解释性范式”。未来的研究将探讨将分层量化变分自编码器（HQ-VAE）集成以增强 AIM 的复杂表达能力，并研究“强化学习（RL）低级预训练”的潜在可能性。这一发现为沟通符号主义和连接主义的连通提供了新途径。 

---
# React to This (RTT): A Nonverbal Turing Test for Embodied AI 

**Title (ZH)**: 基于身体的AI的非言语图灵测试： React to This (RTT) 

**Authors**: Chuxuan Zhang, Yasaman Etesam, Angelica Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.10812)  

**Abstract**: We propose an approach to test embodied AI agents for interaction awareness and believability, particularly in scenarios where humans push them to their limits. Turing introduced the Imitation Game as a way to explore the question: "Can machines think?" The Total Turing Test later expanded this concept beyond purely verbal communication, incorporating perceptual and physical interaction. Building on this, we propose a new guiding question: "Can machines react?" and introduce the React to This (RTT) test for nonverbal behaviors, presenting results from an initial experiment. 

**Abstract (ZH)**: 我们提出一种测试具身AI代理在交互意识和可信度方面的方法，特别是在人类将其推至极限的情景中。图灵引入了模仿游戏来探索“机器能思考吗？”这一问题。随后的全面图灵测试将这一概念扩展到不仅仅是言语交流，还包括感知和物理交互。在此基础上，我们提出了一个新的指导问题：“机器能作出反应吗？”并引入了RTT测试来评估非言语行为，展示了一个初步实验的结果。 

---
# Ground-Compose-Reinforce: Tasking Reinforcement Learning Agents through Formal Language 

**Title (ZH)**: 基于地面-组成-强化：通过形式语言任务化 reinforcement learning 代理 

**Authors**: Andrew C. Li, Toryn Q. Klassen, Andrew Wang, Parand A. Alamdari, Sheila A. McIlraith  

**Link**: [PDF](https://arxiv.org/pdf/2507.10741)  

**Abstract**: Grounding language in complex perception (e.g. pixels) and action is a key challenge when building situated agents that can interact with humans via language. In past works, this is often solved via manual design of the language grounding or by curating massive datasets relating language to elements of the environment. We propose Ground-Compose-Reinforce, a neurosymbolic framework for grounding formal language from data, and eliciting behaviours by directly tasking RL agents through this language. By virtue of data-driven learning, our framework avoids the manual design of domain-specific elements like reward functions or symbol detectors. By virtue of compositional formal language semantics, our framework achieves data-efficient grounding and generalization to arbitrary language compositions. Experiments on an image-based gridworld and a MuJoCo robotics domain show that our approach reliably maps formal language instructions to behaviours with limited data while end-to-end, data-driven approaches fail. 

**Abstract (ZH)**: 将语言嵌入复杂感知（例如像素）和行动中是构建能够通过语言与人类交互的智能体的一个关键挑战。以往的工作常常通过人工设计语言嵌入或构建大量将语言与环境元素关联的数据集来解决这一问题。我们提出了一种神经符号框架——Ground-Compose-Reinforce，用于从数据中进行形式语言的嵌入，并通过这种语言直接对RL智能体进行任务指派以引出其行为。得益于数据驱动的学习，我们的框架避免了人工设计领域特定的元素，如奖励函数或符号检测器。得益于组合理式语言语义，我们的框架实现了数据高效嵌入和对任意语言组合的泛化。在基于图像的格世界和MuJoCo机器人学领域中的实验表明，我们的方法能够在有限的数据下可靠地将形式语言指令映射为行为，而端到端的数据驱动方法则失败了。 

---
# Meta-Reinforcement Learning for Fast and Data-Efficient Spectrum Allocation in Dynamic Wireless Networks 

**Title (ZH)**: 元强化学习在动态无线网络中快速高效频谱分配中的应用 

**Authors**: Oluwaseyi Giwa, Tobi Awodunmila, Muhammad Ahmed Mohsin, Ahsan Bilal, Muhammad Ali Jamshed  

**Link**: [PDF](https://arxiv.org/pdf/2507.10619)  

**Abstract**: The dynamic allocation of spectrum in 5G / 6G networks is critical to efficient resource utilization. However, applying traditional deep reinforcement learning (DRL) is often infeasible due to its immense sample complexity and the safety risks associated with unguided exploration, which can cause severe network interference. To address these challenges, we propose a meta-learning framework that enables agents to learn a robust initial policy and rapidly adapt to new wireless scenarios with minimal data. We implement three meta-learning architectures, model-agnostic meta-learning (MAML), recurrent neural network (RNN), and an attention-enhanced RNN, and evaluate them against a non-meta-learning DRL algorithm, proximal policy optimization (PPO) baseline, in a simulated dynamic integrated access/backhaul (IAB) environment. Our results show a clear performance gap. The attention-based meta-learning agent reaches a peak mean network throughput of 48 Mbps, while the PPO baseline decreased drastically to 10 Mbps. Furthermore, our method reduces SINR and latency violations by more than 50% compared to PPO. It also shows quick adaptation, with a fairness index 0.7, showing better resource allocation. This work proves that meta-learning is a very effective and safer option for intelligent control in complex wireless systems. 

**Abstract (ZH)**: 5G/6G网络中基于元学习的频谱动态分配：一种高效安全的资源管理方法 

---
# Truth Sleuth and Trend Bender: AI Agents to fact-check YouTube videos and influence opinions 

**Title (ZH)**: 真相探查者与趋势颠覆者：用于检查YouTube视频事实并影响观点的AI代理 

**Authors**: Logé Cécile, Ghori Rehan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10577)  

**Abstract**: Misinformation poses a significant threat in today's digital world, often spreading rapidly through platforms like YouTube. This paper introduces a novel approach to combating misinformation by developing an AI-powered system that not only fact-checks claims made in YouTube videos but also actively engages users in the comment section and challenge misleading narratives. Our system comprises two main agents: Truth Sleuth and Trend Bender.
Truth Sleuth extracts claims from a YouTube video, uses a Retrieval-Augmented Generation (RAG) approach - drawing on sources like Wikipedia, Google Search, Google FactCheck - to accurately assess their veracity and generates a nuanced and comprehensive report. Through rigorous prompt engineering, Trend Bender leverages this report along with a curated corpus of relevant articles to generate insightful and persuasive comments designed to stimulate a productive debate. With a carefully set up self-evaluation loop, this agent is able to iteratively improve its style and refine its output.
We demonstrate the system's capabilities through experiments on established benchmark datasets and a real-world deployment on YouTube, showcasing its potential to engage users and potentially influence perspectives. Our findings highlight the high accuracy of our fact-checking agent, and confirm the potential of AI-driven interventions in combating misinformation and fostering a more informed online space. 

**Abstract (ZH)**: 人工智能赋能的系统在YouTube上对抗 misinformation及其应用研究 

---
