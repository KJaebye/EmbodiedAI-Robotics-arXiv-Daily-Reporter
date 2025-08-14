# Masquerade: Learning from In-the-wild Human Videos using Data-Editing 

**Title (ZH)**: 欺瞒：利用数据编辑从wild数据中学习 

**Authors**: Marion Lepert, Jiaying Fang, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2508.09976)  

**Abstract**: Robot manipulation research still suffers from significant data scarcity: even the largest robot datasets are orders of magnitude smaller and less diverse than those that fueled recent breakthroughs in language and vision. We introduce Masquerade, a method that edits in-the-wild egocentric human videos to bridge the visual embodiment gap between humans and robots and then learns a robot policy with these edited videos. Our pipeline turns each human video into robotized demonstrations by (i) estimating 3-D hand poses, (ii) inpainting the human arms, and (iii) overlaying a rendered bimanual robot that tracks the recovered end-effector trajectories. Pre-training a visual encoder to predict future 2-D robot keypoints on 675K frames of these edited clips, and continuing that auxiliary loss while fine-tuning a diffusion policy head on only 50 robot demonstrations per task, yields policies that generalize significantly better than prior work. On three long-horizon, bimanual kitchen tasks evaluated in three unseen scenes each, Masquerade outperforms baselines by 5-6x. Ablations show that both the robot overlay and co-training are indispensable, and performance scales logarithmically with the amount of edited human video. These results demonstrate that explicitly closing the visual embodiment gap unlocks a vast, readily available source of data from human videos that can be used to improve robot policies. 

**Abstract (ZH)**: 机器人操作研究仍受到显著的数据稀缺性困扰：即使最大的机器人数据集也比 recent 语言和视觉领域的突破所依赖的数据小几个数量级且不那么多样化。我们提出了 Masquerade 方法，该方法编辑野外第一人称人类视频以弥合人类与机器人之间的视觉实体差距，然后使用这些编辑后的视频学习一种机器人策略。我们的管道通过以下步骤将每段人类视频转换为机器人化演示：(i) 估计 3D 手部姿势，(ii) 填充人类手臂，(iii) 覆盖渲染的双臂机器人，使其跟踪恢复的末端执行器轨迹。在 675,000 帧这些编辑片段的基础上预训练一个视觉编码器以预测未来 2D 机器人关键点，并在仅使用 50 个机器人演示每任务的情况下微调扩散策略头的同时继续使用该辅助损失，可以比之前的工作获得显著更好的泛化性能。在三个长时距的双臂厨房任务中评估，其中每个任务有三个未见过的场景，Masquerade 的表现比基线高出 5-6 倍。消融实验表明，机器人覆盖和联合训练都是必不可少的，性能随编辑的人类视频量按对数关系增长。这些结果表明，明确地弥合视觉实体差距可以解锁大量现成的数据来源，这些数据可以从人类视频中获取并用于提高机器人策略。 

---
# GBC: Generalized Behavior-Cloning Framework for Whole-Body Humanoid Imitation 

**Title (ZH)**: Generalized 行为克隆框架：全身 humanoid 仿真人机交互 

**Authors**: Yifei Yao, Chengyuan Luo, Jiaheng Du, Wentao He, Jun-Guo Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09960)  

**Abstract**: The creation of human-like humanoid robots is hindered by a fundamental fragmentation: data processing and learning algorithms are rarely universal across different robot morphologies. This paper introduces the Generalized Behavior Cloning (GBC) framework, a comprehensive and unified solution designed to solve this end-to-end challenge. GBC establishes a complete pathway from human motion to robot action through three synergistic innovations. First, an adaptive data pipeline leverages a differentiable IK network to automatically retarget any human MoCap data to any humanoid. Building on this foundation, our novel DAgger-MMPPO algorithm with its MMTransformer architecture learns robust, high-fidelity imitation policies. To complete the ecosystem, the entire framework is delivered as an efficient, open-source platform based on Isaac Lab, empowering the community to deploy the full workflow via simple configuration scripts. We validate the power and generality of GBC by training policies on multiple heterogeneous humanoids, demonstrating excellent performance and transfer to novel motions. This work establishes the first practical and unified pathway for creating truly generalized humanoid controllers. 

**Abstract (ZH)**: 人类类人机器人创建受制于根本性的碎片化问题：不同的机器人形态通常缺乏普适的数据处理和学习算法。本文提出了通用行为克隆（GBC）框架，这是一种端到端的综合统一解决方案，通过三种协同创新建立了从人类运动到机器人动作的完整路径。首先，自适应数据管道利用可微逆运动学网络自动将任何人的动捕数据重定向到任何类人机器人。在此基础上，我们的新型DAgger-MMPPO算法及其MMTransformer架构学习出鲁棒且高保真的模仿策略。为了构建完整的生态系统，整个框架基于Isaac Lab以高效开源的形式提供，使社区能够通过简单的配置脚本部署完整的操作流程。我们通过在多种异构类人机器人上训练策略，验证了GBC的强大力量和泛化能力，并展示了其对新颖运动的出色适应性。这项工作确立了创建真正通用类人控制器的第一个可行且统一的途径。 

---
# PPL: Point Cloud Supervised Proprioceptive Locomotion Reinforcement Learning for Legged Robots in Crawl Spaces 

**Title (ZH)**: PPL： crawl space中腿式机器人基于点云监督的本体感受性强化学习 

**Authors**: Bida Ma, Nuo Xu, Chenkun Qi, Xin Liu, Yule Mo, Jinkai Wang, Chunpeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09950)  

**Abstract**: The legged locomotion in spatially constrained structures (called crawl spaces) is challenging. In crawl spaces, current exteroceptive locomotion learning methods are limited by large noises and errors of the sensors in possible low visibility conditions, and current proprioceptive locomotion learning methods are difficult in traversing crawl spaces because only ground features are inferred. In this study, a point cloud supervised proprioceptive locomotion reinforcement learning method for legged robots in crawl spaces is proposed. A state estimation network is designed to estimate the robot's surrounding ground and spatial features as well as the robot's collision states using historical proprioceptive sensor data. The point cloud is represented in polar coordinate frame and a point cloud processing method is proposed to efficiently extract the ground and spatial features that are used to supervise the state estimation network learning. Comprehensive reward functions that guide the robot to traverse through crawl spaces after collisions are designed. Experiments demonstrate that, compared to existing methods, our method exhibits more agile locomotion in crawl spaces. This study enhances the ability of legged robots to traverse spatially constrained environments without requiring exteroceptive sensors. 

**Abstract (ZH)**: 基于空间受限结构中足式运动的点云监督 proprioceptive 运动强化学习方法 

---
# Toward Human-Robot Teaming: Learning Handover Behaviors from 3D Scenes 

**Title (ZH)**: 向着人机协同的目标：从三维场景学习交接行为 

**Authors**: Yuekun Wu, Yik Lung Pang, Andrea Cavallaro, Changjae Oh  

**Link**: [PDF](https://arxiv.org/pdf/2508.09855)  

**Abstract**: Human-robot teaming (HRT) systems often rely on large-scale datasets of human and robot interactions, especially for close-proximity collaboration tasks such as human-robot handovers. Learning robot manipulation policies from raw, real-world image data requires a large number of robot-action trials in the physical environment. Although simulation training offers a cost-effective alternative, the visual domain gap between simulation and robot workspace remains a major limitation. We introduce a method for training HRT policies, focusing on human-to-robot handovers, solely from RGB images without the need for real-robot training or real-robot data collection. The goal is to enable the robot to reliably receive objects from a human with stable grasping while avoiding collisions with the human hand. The proposed policy learner leverages sparse-view Gaussian Splatting reconstruction of human-to-robot handover scenes to generate robot demonstrations containing image-action pairs captured with a camera mounted on the robot gripper. As a result, the simulated camera pose changes in the reconstructed scene can be directly translated into gripper pose changes. Experiments in both Gaussian Splatting reconstructed scene and real-world human-to-robot handover experiments demonstrate that our method serves as a new and effective representation for the human-to-robot handover task, contributing to more seamless and robust HRT. 

**Abstract (ZH)**: 基于RGB图像训练人机协作政策：专注于人对机器人手递任务 

---
# Whole-Body Bilateral Teleoperation with Multi-Stage Object Parameter Estimation for Wheeled Humanoid Locomanipulation 

**Title (ZH)**: 全身体部位双边遥控操作及其多阶段物体参数估计在轮式类人机器人手上操作中的应用 

**Authors**: Donghoon Baek, Amartya Purushottam, Jason J. Choi, Joao Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2508.09846)  

**Abstract**: This paper presents an object-aware whole-body bilateral teleoperation framework for wheeled humanoid loco-manipulation. This framework combines whole-body bilateral teleoperation with an online multi-stage object inertial parameter estimation module, which is the core technical contribution of this work. The multi-stage process sequentially integrates a vision-based object size estimator, an initial parameter guess generated by a large vision-language model (VLM), and a decoupled hierarchical sampling strategy. The visual size estimate and VLM prior offer a strong initial guess of the object's inertial parameters, significantly reducing the search space for sampling-based refinement and improving the overall estimation speed. A hierarchical strategy first estimates mass and center of mass, then infers inertia from object size to ensure physically feasible parameters, while a decoupled multi-hypothesis scheme enhances robustness to VLM prior errors. Our estimator operates in parallel with high-fidelity simulation and hardware, enabling real-time online updates. The estimated parameters are then used to update the wheeled humanoid's equilibrium point, allowing the operator to focus more on locomotion and manipulation. This integration improves the haptic force feedback for dynamic synchronization, enabling more dynamic whole-body teleoperation. By compensating for object dynamics using the estimated parameters, the framework also improves manipulation tracking while preserving compliant behavior. We validate the system on a customized wheeled humanoid with a robotic gripper and human-machine interface, demonstrating real-time execution of lifting, delivering, and releasing tasks with a payload weighing approximately one-third of the robot's body weight. 

**Abstract (ZH)**: 本文提出了一种针对轮式类人仿生移动与操作的物体感知整体体双边远程操作框架。该框架结合 �整合了整体体双边远程操作与基于视觉的多网格格参数估计的在线阶段过程。该该多阶段过程依次结合由基于视觉的物体感知估计算法、通过大规模视觉-语言模型（VLM）生成的初步参数猜测，以及解耦的分层次抽样策略。视觉估计和 VLM � 前知共同提供了物体惯性性参数的强先验估计，，为此基上了奠定了格步优化基础并提高整体的速度 on。首先的多次层次定义算法估计质量并 以及重心，，然后从质量推到重心推导惯性，，并进而推导出物理上可行的参数。同时，解耦的多多多层次假设方案增强了了对 VLM 基先估计的鲁棒性性。我们的估计算法与高保真度的仿真和硬件并行运行 on，实现实时在线线线线线线线线线线更新。估计的参数 on则通过上更新轮式类人人人人类人人的平衡点以进一步集中在移动和操作上。该集成改善了触觉反馈以动态同步 on从而实现了更动态的整体体远程操作。通过使用所估计算拟物体动力学， on改善了操作追踪同时保留了顺应行为。我们在一架定制的轮式类ononon on类on类人人的设计上面上集带了机器人手爪，并上展示了实时执行提升、输送和放置任务的能力，其中负重约为机器人重量的三分之一。 

---
# Embodied Tactile Perception of Soft Objects Properties 

**Title (ZH)**: 软物体质地的 embodied 触觉感知 

**Authors**: Anirvan Dutta, Alexis WM Devillard, Zhihuan Zhang, Xiaoxiao Cheng, Etienne Burdet  

**Link**: [PDF](https://arxiv.org/pdf/2508.09836)  

**Abstract**: To enable robots to develop human-like fine manipulation, it is essential to understand how mechanical compliance, multi-modal sensing, and purposeful interaction jointly shape tactile perception. In this study, we use a dedicated modular e-Skin with tunable mechanical compliance and multi-modal sensing (normal, shear forces and vibrations) to systematically investigate how sensing embodiment and interaction strategies influence robotic perception of objects. Leveraging a curated set of soft wave objects with controlled viscoelastic and surface properties, we explore a rich set of palpation primitives-pressing, precession, sliding that vary indentation depth, frequency, and directionality. In addition, we propose the latent filter, an unsupervised, action-conditioned deep state-space model of the sophisticated interaction dynamics and infer causal mechanical properties into a structured latent space. This provides generalizable and in-depth interpretable representation of how embodiment and interaction determine and influence perception. Our investigation demonstrates that multi-modal sensing outperforms uni-modal sensing. It highlights a nuanced interaction between the environment and mechanical properties of e-Skin, which should be examined alongside the interaction by incorporating temporal dynamics. 

**Abstract (ZH)**: 使机器人具备类人的精细操作能力，理解机械柔顺性、多模态感知和目的性交互如何共同塑造触觉感知至关重要。在本研究中，我们使用具有可调机械柔顺性和多模态感知（正常力、切力和振动）的专用模块化电子皮肤，系统地研究了感知体态和交互策略如何影响机器人的物体感知。利用一组具有可控粘弹性及表面性质的软波形物体，我们探索了包括按压、旋转变换和滑动等多种丰富的触觉基本操作，这些操作可变化深度、频率和方向性。此外，我们提出了潜在滤波器，这是一种无监督的、基于动作条件的深度状态空间模型，用于推断复杂的交互动力学，并将因果机械特性映射到结构化的潜在空间。这提供了感知体态和交互如何决定和影响感知的一般化和深入解释的表示。我们的研究展示了多模态感知优于单模态感知，并强调了环境与电子皮肤机械性质之间微妙的交互作用，这种交互作用应在考虑时间动态的同时进行研究。 

---
# FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning 

**Title (ZH)**: FLARE: 通过强化学习的四旋翼缆挂载荷系统敏捷飞行方法 

**Authors**: Dongcheng Cao, Jin Zhou, Xian Wang, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09797)  

**Abstract**: Agile flight for the quadrotor cable-suspended payload system is a formidable challenge due to its underactuated, highly nonlinear, and hybrid dynamics. Traditional optimization-based methods often struggle with high computational costs and the complexities of cable mode transitions, limiting their real-time applicability and maneuverability exploitation. In this letter, we present FLARE, a reinforcement learning (RL) framework that directly learns agile navigation policy from high-fidelity simulation. Our method is validated across three designed challenging scenarios, notably outperforming a state-of-the-art optimization-based approach by a 3x speedup during gate traversal maneuvers. Furthermore, the learned policies achieve successful zero-shot sim-to-real transfer, demonstrating remarkable agility and safety in real-world experiments, running in real time on an onboard computer. 

**Abstract (ZH)**: 基于强化学习的四旋翼悬挂载荷系统的敏捷飞行：FLARE框架 

---
# Immersive Teleoperation of Beyond-Human-Scale Robotic Manipulators: Challenges and Future Directions 

**Title (ZH)**: 沉浸式远程操作超人类规模机器人 manipulator 删除器：挑战与未来方向 

**Authors**: Mahdi Hejrati, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2508.09700)  

**Abstract**: Teleoperation of beyond-human-scale robotic manipulators (BHSRMs) presents unique challenges that differ fundamentally from conventional human-scale systems. As these platforms gain relevance in industrial domains such as construction, mining, and disaster response, immersive interfaces must be rethought to support scalable, safe, and effective human-robot collaboration. This paper investigates the control, cognitive, and interface-level challenges of immersive teleoperation in BHSRMs, with a focus on ensuring operator safety, minimizing sensorimotor mismatch, and enhancing the sense of embodiment. We analyze design trade-offs in haptic and visual feedback systems, supported by early experimental comparisons of exoskeleton- and joystick-based control setups. Finally, we outline key research directions for developing new evaluation tools, scaling strategies, and human-centered safety models tailored to large-scale robotic telepresence. 

**Abstract (ZH)**: 超越人类规模的机器人操作手远程操作：超越人类规模的机器人操作手(BHSRMs)的远程操作提出了与传统人类规模系统根本不同的独特挑战。随着这些平台在建筑、采矿和灾难响应等工业领域中的重要性不断提高，沉浸式界面必须重新设计以支持可扩展、安全和有效的有人-机器人协作。本文探讨了BHSRMs中沉浸式远程操作的控制、认知和界面级挑战，重点关注确保操作员安全、减少感觉运动不匹配以及增强沉浸感。我们分析了触觉和视觉反馈系统的设计权衡，并通过早期的外骨骼和joystick操作设置的实验比较加以支持。最后，我们概述了开发新的评估工具、扩展策略和以人为核心的安全模型的关键研究方向，这些模型适用于大规模机器人远程操作。 

---
# Interpretable Robot Control via Structured Behavior Trees and Large Language Models 

**Title (ZH)**: 结构化行为树与大型语言模型下的可解释机器人控制 

**Authors**: Ingrid Maéva Chekam, Ines Pastor-Martinez, Ali Tourani, Jose Andres Millan-Romera, Laura Ribeiro, Pedro Miguel Bastos Soares, Holger Voos, Jose Luis Sanchez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2508.09621)  

**Abstract**: As intelligent robots become more integrated into human environments, there is a growing need for intuitive and reliable Human-Robot Interaction (HRI) interfaces that are adaptable and more natural to interact with. Traditional robot control methods often require users to adapt to interfaces or memorize predefined commands, limiting usability in dynamic, unstructured environments. This paper presents a novel framework that bridges natural language understanding and robotic execution by combining Large Language Models (LLMs) with Behavior Trees. This integration enables robots to interpret natural language instructions given by users and translate them into executable actions by activating domain-specific plugins. The system supports scalable and modular integration, with a primary focus on perception-based functionalities, such as person tracking and hand gesture recognition. To evaluate the system, a series of real-world experiments was conducted across diverse environments. Experimental results demonstrate that the proposed approach is practical in real-world scenarios, with an average cognition-to-execution accuracy of approximately 94%, making a significant contribution to HRI systems and robots. The complete source code of the framework is publicly available at this https URL. 

**Abstract (ZH)**: 随着智能机器人越来越多地融入人类环境，需要更为直观可靠且适应性强的人机交互（HRI）界面，以便与之进行自然的交互。传统的机器人控制方法往往要求用户适应接口或记忆预定义的命令，这在动态的非结构化环境中限制了其实用性。本文提出了一种新型框架，将大规模语言模型（LLMs）与行为树结合，以实现自然语言理解和机器人执行的融合。该集成能够使机器人解析由用户给出的自然语言指令，并通过激活领域特定插件将其转化为可执行的动作。该系统支持可扩展和模块化的集成，主要关注基于感知的功能，如人像跟踪和手势识别。为了评估系统，在多种环境中进行了实地实验。实验结果表明，所提出的方法在实际应用场景中是可行的，认知到执行的平均准确率为约94%，对HRI系统和机器人做出了重要贡献。该框架的完整源代码可从此链接访问。 

---
# BEAVR: Bimanual, multi-Embodiment, Accessible, Virtual Reality Teleoperation System for Robots 

**Title (ZH)**: BEAVR: 双手、多身体、Accessible、虚拟现实远程操作机器人系统 

**Authors**: Alejandro Posadas-Nava, Alejandro Carrasco, Richard Linares  

**Link**: [PDF](https://arxiv.org/pdf/2508.09606)  

**Abstract**: \textbf{BEAVR} is an open-source, bimanual, multi-embodiment Virtual Reality (VR) teleoperation system for robots, designed to unify real-time control, data recording, and policy learning across heterogeneous robotic platforms. BEAVR enables real-time, dexterous teleoperation using commodity VR hardware, supports modular integration with robots ranging from 7-DoF manipulators to full-body humanoids, and records synchronized multi-modal demonstrations directly in the LeRobot dataset schema. Our system features a zero-copy streaming architecture achieving $\leq$35\,ms latency, an asynchronous ``think--act'' control loop for scalable inference, and a flexible network API optimized for real-time, multi-robot operation. We benchmark BEAVR across diverse manipulation tasks and demonstrate its compatibility with leading visuomotor policies such as ACT, DiffusionPolicy, and SmolVLA. All code is publicly available, and datasets are released on Hugging Face\footnote{Code, datasets, and VR app available at this https URL. 

**Abstract (ZH)**: BEAVR是一个开源的双臂多躯体虚拟现实(VR)遥控系统，用于机器人，旨在统一异构机器人平台上的实时控制、数据记录和策略学习。BEAVR使用户能够使用商用VR硬件进行实时、灵巧的遥控操作，支持从7自由度 manipulator 到全身类人机器人等多种型号的机器人模块化集成，并直接在LeRobot数据集模式中记录同步的多模态演示。该系统具有零拷贝流式架构，延迟≤35毫秒，采用异步“思考-行动”控制环路进行可扩展推理，以及针对实时多机器人操作优化的灵活网络API。我们在多种操作任务中对BEAVR进行了基准测试，并展示了其与领先的视觉-运动策略（如ACT、DiffusionPolicy和SmolVLA）的兼容性。所有代码均可公开访问，数据集在Hugging Face上发布。 

---
# DAgger Diffusion Navigation: DAgger Boosted Diffusion Policy for Vision-Language Navigation 

**Title (ZH)**: DAgger扩散导航：增强扩散策略的DAgger视觉语言导航 

**Authors**: Haoxiang Shi, Xiang Deng, Zaijing Li, Gongwei Chen, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.09444)  

**Abstract**: Vision-Language Navigation in Continuous Environments (VLN-CE) requires agents to follow natural language instructions through free-form 3D spaces. Existing VLN-CE approaches typically use a two-stage waypoint planning framework, where a high-level waypoint predictor generates the navigable waypoints, and then a navigation planner suggests the intermediate goals in the high-level action space. However, this two-stage decomposition framework suffers from: (1) global sub-optimization due to the proxy objective in each stage, and (2) a performance bottleneck caused by the strong reliance on the quality of the first-stage predicted waypoints. To address these limitations, we propose DAgger Diffusion Navigation (DifNav), an end-to-end optimized VLN-CE policy that unifies the traditional two stages, i.e. waypoint generation and planning, into a single diffusion policy. Notably, DifNav employs a conditional diffusion policy to directly model multi-modal action distributions over future actions in continuous navigation space, eliminating the need for a waypoint predictor while enabling the agent to capture multiple possible instruction-following behaviors. To address the issues of compounding error in imitation learning and enhance spatial reasoning in long-horizon navigation tasks, we employ DAgger for online policy training and expert trajectory augmentation, and use the aggregated data to further fine-tune the policy. This approach significantly improves the policy's robustness and its ability to recover from error states. Extensive experiments on benchmark datasets demonstrate that, even without a waypoint predictor, the proposed method substantially outperforms previous state-of-the-art two-stage waypoint-based models in terms of navigation performance. Our code is available at: this https URL. 

**Abstract (ZH)**: 连续环境中的视觉-语言导航（VLN-CE）要求代理遵循自然语言指令通过自由形式的3D空间导航。现有的VLN-CE方法通常采用两阶段的航点规划框架，其中高层航点预测器生成可导航的航点，然后导航规划器在高层动作空间中建议中间目标。然而，这种两阶段分解框架面临以下局限性：（1）由于每个阶段的代理目标而产生的全局次优性，以及（2）性能瓶颈，由于对第一阶段预测航点质量的强烈依赖。为了解决这些问题，我们提出了面向扩散的航向导航（DifNav），这是一种端到端优化的VLN-CE策略，将传统的两个阶段，即航点生成和规划，统一成一个单差分策略。值得注意的是，DifNav 使用条件扩散策略直接建模连续导航空间中未来动作的多模态动作分布，从而消除航点预测器的需要，同时使代理能够捕捉到多种可能的指令遵循行为。为了应对模仿学习中的累积误差问题，并增强长期导航任务中的空间推理能力，我们采用了DAgger进行在线策略训练和专家轨迹增强，并使用汇总数据进一步微调策略。该方法显著提高了策略的鲁棒性及其从错误状态中恢复的能力。基准数据集上的广泛实验表明，即使没有航点预测器，所提出的方法在导航性能上也显著优于之前的基于两阶段航点预测的最新模型。我们的代码可在以下地址获得：this https URL。 

---
# CLF-RL: Control Lyapunov Function Guided Reinforcement Learning 

**Title (ZH)**: CLF-RL: 控制李雅普诺夫函数引导的强化学习 

**Authors**: Kejun Li, Zachary Olkin, Yisong Yue, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2508.09354)  

**Abstract**: Reinforcement learning (RL) has shown promise in generating robust locomotion policies for bipedal robots, but often suffers from tedious reward design and sensitivity to poorly shaped objectives. In this work, we propose a structured reward shaping framework that leverages model-based trajectory generation and control Lyapunov functions (CLFs) to guide policy learning. We explore two model-based planners for generating reference trajectories: a reduced-order linear inverted pendulum (LIP) model for velocity-conditioned motion planning, and a precomputed gait library based on hybrid zero dynamics (HZD) using full-order dynamics. These planners define desired end-effector and joint trajectories, which are used to construct CLF-based rewards that penalize tracking error and encourage rapid convergence. This formulation provides meaningful intermediate rewards, and is straightforward to implement once a reference is available. Both the reference trajectories and CLF shaping are used only during training, resulting in a lightweight policy at deployment. We validate our method both in simulation and through extensive real-world experiments on a Unitree G1 robot. CLF-RL demonstrates significantly improved robustness relative to the baseline RL policy and better performance than a classic tracking reward RL formulation. 

**Abstract (ZH)**: 基于模型的轨迹生成和控制李雅普un夫函数的强化学习奖励塑造框架：提高双足机器人稳健性的方法 

---
# How Safe Will I Be Given What I Saw? Calibrated Prediction of Safety Chances for Image-Controlled Autonomy 

**Title (ZH)**: 给定所见保障安全的概率：图像控制自主性的校准预测 

**Authors**: Zhenjiang Mao, Mrinall Eashaan Umasudhan, Ivan Ruchkin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09346)  

**Abstract**: Autonomous robots that rely on deep neural network controllers pose critical challenges for safety prediction, especially under partial observability and distribution shift. Traditional model-based verification techniques are limited in scalability and require access to low-dimensional state models, while model-free methods often lack reliability guarantees. This paper addresses these limitations by introducing a framework for calibrated safety prediction in end-to-end vision-controlled systems, where neither the state-transition model nor the observation model is accessible. Building on the foundation of world models, we leverage variational autoencoders and recurrent predictors to forecast future latent trajectories from raw image sequences and estimate the probability of satisfying safety properties. We distinguish between monolithic and composite prediction pipelines and introduce a calibration mechanism to quantify prediction confidence. In long-horizon predictions from high-dimensional observations, the forecasted inputs to the safety evaluator can deviate significantly from the training distribution due to compounding prediction errors and changing environmental conditions, leading to miscalibrated risk estimates. To address this, we incorporate unsupervised domain adaptation to ensure robustness of safety evaluation under distribution shift in predictions without requiring manual labels. Our formulation provides theoretical calibration guarantees and supports practical evaluation across long prediction horizons. Experimental results on three benchmarks show that our UDA-equipped evaluators maintain high accuracy and substantially lower false positive rates under distribution shift. Similarly, world model-based composite predictors outperform their monolithic counterparts on long-horizon tasks, and our conformal calibration provides reliable statistical bounds. 

**Abstract (ZH)**: 基于深度神经网络控制器的自主机器人在部分可观测性和分布偏移条件下的安全性预测校准框架 

---
# Human-Aligned Procedural Level Generation Reinforcement Learning via Text-Level-Sketch Shared Representation 

**Title (ZH)**: 基于文本级草图共享表示的人类齐平程序化层级生成强化学习 

**Authors**: In-Chang Baek, Seoyoung Lee, Sung-Hyun Kim, Geumhwan Hwang, KyungJoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.09860)  

**Abstract**: Human-aligned AI is a critical component of co-creativity, as it enables models to accurately interpret human intent and generate controllable outputs that align with design goals in collaborative content creation. This direction is especially relevant in procedural content generation via reinforcement learning (PCGRL), which is intended to serve as a tool for human designers. However, existing systems often fall short of exhibiting human-centered behavior, limiting the practical utility of AI-driven generation tools in real-world design workflows. In this paper, we propose VIPCGRL (Vision-Instruction PCGRL), a novel deep reinforcement learning framework that incorporates three modalities-text, level, and sketches-to extend control modality and enhance human-likeness. We introduce a shared embedding space trained via quadruple contrastive learning across modalities and human-AI styles, and align the policy using an auxiliary reward based on embedding similarity. Experimental results show that VIPCGRL outperforms existing baselines in human-likeness, as validated by both quantitative metrics and human evaluations. The code and dataset will be available upon publication. 

**Abstract (ZH)**: 基于人类导向的AI是协创作中关键的组成部分，使模型能够准确地理解人类意图并生成与目标一致的可控输出。这一方向尤其适用于通过强化学习（PCGRL）生成过程性
user
基于人类导向的AI是协创作过程中的关键组成部分，通过准确地理解人类意图并生成与预定目标一致的可控输出，此方向尤其适用于通过强化学习（PCGRL）生成内容的场景，其中内容旨在作为人类设计师的工具。然而，现有方法往往缺乏以人为中心的行为，限制了基于AI驱动的辅助工具在实际设计工作流中的实用性。因此，我们提出了一种基于视图指示的PCGRL（视觉指示PCGRL），这是一种结合了文本、语义和素描三种模态的深度强化学习框架，旨在提高可控性和增强人性相似度我们通过四模态和人机风格四种对比学习来训练了共享嵌入并并使用基于嵌入相似性的辅助奖励对策略进行调整。实验表明基于视觉指示的的PCGRL在人性相似度现有基线上线上表现出更优的性能；该方法已通过定量评价和人工评估得到验证。源代码和数据集将在不久后提供。 

---
# An Automated Multi-Modal Evaluation Framework for Mobile Intelligent Assistants 

**Title (ZH)**: 移动智能助手的自动化多模态评估框架 

**Authors**: Meiping Wang, Jian Zhong, Rongduo Han, Liming Kang, Zhengkun Shi, Xiao Liang, Xing Lin, Nan Gao, Haining Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09507)  

**Abstract**: With the rapid development of mobile intelligent assistant technologies, multi-modal AI assistants have become essential interfaces for daily user interactions. However, current evaluation methods face challenges including high manual costs, inconsistent standards, and subjective bias. This paper proposes an automated multi-modal evaluation framework based on large language models and multi-agent collaboration. The framework employs a three-tier agent architecture consisting of interaction evaluation agents, semantic verification agents, and experience decision agents. Through supervised fine-tuning on the Qwen3-8B model, we achieve a significant evaluation matching accuracy with human experts. Experimental results on eight major intelligent agents demonstrate the framework's effectiveness in predicting users' satisfaction and identifying generation defects. 

**Abstract (ZH)**: 基于大规模语言模型和多agent协作的多模态自动化评估框架 

---
# Perceptual Reality Transformer: Neural Architectures for Simulating Neurological Perception Conditions 

**Title (ZH)**: 感知现实变换器：模拟神经感知条件的神经架构 

**Authors**: Baihan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09852)  

**Abstract**: Neurological conditions affecting visual perception create profound experiential divides between affected individuals and their caregivers, families, and medical professionals. We present the Perceptual Reality Transformer, a comprehensive framework employing six distinct neural architectures to simulate eight neurological perception conditions with scientifically-grounded visual transformations. Our system learns mappings from natural images to condition-specific perceptual states, enabling others to experience approximations of simultanagnosia, prosopagnosia, ADHD attention deficits, visual agnosia, depression-related changes, anxiety tunnel vision, and Alzheimer's memory effects. Through systematic evaluation across ImageNet and CIFAR-10 datasets, we demonstrate that Vision Transformer architectures achieve optimal performance, outperforming traditional CNN and generative approaches. Our work establishes the first systematic benchmark for neurological perception simulation, contributes novel condition-specific perturbation functions grounded in clinical literature, and provides quantitative metrics for evaluating simulation fidelity. The framework has immediate applications in medical education, empathy training, and assistive technology development, while advancing our fundamental understanding of how neural networks can model atypical human perception. 

**Abstract (ZH)**: 神经系统条件影响视觉感知，在受这些条件影响的个体与其护理人员、家庭成员和医疗专业人员之间创造出深刻的经验差异。我们提出了一种全面的框架——感知现实转换器，该框架采用六种不同的神经架构来模拟八种神经系统感知条件的科学依据视觉变换。我们的系统学习从自然图像到特定感知状态的映射，使他人能够体验似动性失认、面孔失认、注意缺陷多动障碍注意力缺陷、视觉失认、与抑郁相关的改变、焦虑导致的隧道视野以及阿尔茨海默病的记忆效应。通过对ImageNet和CIFAR-10数据集的系统评估，我们证明了视觉变换器架构实现了最优性能，超过了传统的CNN和生成方法。我们的工作建立了首个系统的神经系统感知模拟基准，贡献了基于临床文献的新颖条件特定的扰动函数，并提供了评估模拟保真度的定量指标。该框架在医疗教育、共情训练和辅助技术开发中具有即刻应用价值，同时推进了我们对神经网络如何模拟异常人类感知的基本理解。 

---
# Goal Discovery with Causal Capacity for Efficient Reinforcement Learning 

**Title (ZH)**: 因果容量驱动的目标发现与高效强化学习 

**Authors**: Yan Yu, Yaodong Yang, Zhengbo Lu, Chengdong Ma, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09624)  

**Abstract**: Causal inference is crucial for humans to explore the world, which can be modeled to enable an agent to efficiently explore the environment in reinforcement learning. Existing research indicates that establishing the causality between action and state transition will enhance an agent to reason how a policy affects its future trajectory, thereby promoting directed exploration. However, it is challenging to measure the causality due to its intractability in the vast state-action space of complex scenarios. In this paper, we propose a novel Goal Discovery with Causal Capacity (GDCC) framework for efficient environment exploration. Specifically, we first derive a measurement of causality in state space, \emph{i.e.,} causal capacity, which represents the highest influence of an agent's behavior on future trajectories. After that, we present a Monte Carlo based method to identify critical points in discrete state space and further optimize this method for continuous high-dimensional environments. Those critical points are used to uncover where the agent makes important decisions in the environment, which are then regarded as our subgoals to guide the agent to make exploration more purposefully and efficiently. Empirical results from multi-objective tasks demonstrate that states with high causal capacity align with our expected subgoals, and our GDCC achieves significant success rate improvements compared to baselines. 

**Abstract (ZH)**: 因果推理对于人类探索世界至关重要，可以被建模以使代理在强化学习中高效地探索环境。现有研究指出，建立动作与状态转换之间的因果关系可以增强代理对其未来轨迹如何受策略影响的推理能力，从而促进定向探索。然而，由于在复杂场景的巨大状态-动作空间中测量因果关系具有不可行性，这颇具挑战。在本文中，我们提出了一种新的因果容量引导的目标发现（GDCC）框架，用于高效的环境探索。具体而言，我们首先在状态空间中推导出因果性的度量，即因果容量，这代表了代理行为对未来轨迹的最高影响。之后，我们提出了一种基于蒙特卡洛的方法来识别离散状态空间中的关键点，并进一步优化该方法以适应连续高维度环境。这些关键点用于揭示代理在环境中的重要决策位置，然后将这些决策位置视为我们的子目标，以引导代理有目的、高效地进行探索。多目标任务的实验结果表明，具有高因果容量的状态与我们期望的子目标相符，且我们的GDCC相对于基线实现了显著的成功率提升。 

---
# GoViG: Goal-Conditioned Visual Navigation Instruction Generation 

**Title (ZH)**: GoViG: 基于目标的视觉导航指令生成 

**Authors**: Fengyi Wu, Yifei Dong, Zhi-Qi Cheng, Yilong Dai, Guangyu Chen, Hang Wang, Qi Dai, Alexander G. Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.09547)  

**Abstract**: We introduce Goal-Conditioned Visual Navigation Instruction Generation (GoViG), a new task that aims to autonomously generate precise and contextually coherent navigation instructions solely from egocentric visual observations of initial and goal states. Unlike conventional approaches that rely on structured inputs such as semantic annotations or environmental maps, GoViG exclusively leverages raw egocentric visual data, substantially improving its adaptability to unseen and unstructured environments. Our method addresses this task by decomposing it into two interconnected subtasks: (1) visual forecasting, which predicts intermediate visual states bridging the initial and goal views; and (2) instruction generation, which synthesizes linguistically coherent instructions grounded in both observed and anticipated visuals. These subtasks are integrated within an autoregressive multimodal large language model trained with tailored objectives to ensure spatial accuracy and linguistic clarity. Furthermore, we introduce two complementary multimodal reasoning strategies, one-pass and interleaved reasoning, to mimic incremental human cognitive processes during navigation. To evaluate our method, we propose the R2R-Goal dataset, combining diverse synthetic and real-world trajectories. Empirical results demonstrate significant improvements over state-of-the-art methods, achieving superior BLEU-4 and CIDEr scores along with robust cross-domain generalization. 

**Abstract (ZH)**: 基于目标条件的视觉导航指令生成 (GoViG) 

---
# COMPEER: Controllable Empathetic Reinforcement Reasoning for Emotional Support Conversation 

**Title (ZH)**: COMPEER: �,
可控共情强化推理对话支持系统 

**Authors**: Yunxiao Wang, Meng Liu, Wenqi Liu, Kaiyu Jiang, Bin Wen, Fan Yang, Tingting Gao, Guorui Zhou, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.09521)  

**Abstract**: Emotional support conversations are crucial for promoting emotional well-being, yet current models often lack deep empathetic reasoning grounded in psychological principles. To address this, we propose controllable empathetic reasoning, which combines natural language reasoning with structured psychological steps. We construct a fine-grained dataset annotated with reasoning correctness and response preferences to enable this capability. To further enhance training, we employ reinforcement learning with a unified process-outcome reward model that delivers precise feedback. To mitigate response repetitiveness from entropy collapse, we introduce personality-based dialogue rewriting and a redundancy-aware reward reweighting strategy. Our approach significantly improves model's emotional support ability, advancing the development of empathetic, human-like support systems. 

**Abstract (ZH)**: 情感支持对话对于促进情感福祉至关重要，然而当前模型往往缺乏基于心理原理的深刻共情推理能力。为了应对这一问题，我们提出可控共情推理，它将自然语言推理与结构化的心理步骤相结合。我们构建了一个细粒度的数据集，标注了推理正确性和响应偏好，以支持这种能力。为进一步提高训练效果，我们采用强化学习，并使用统一的过程-结果奖励模型提供精确反馈。为缓解由于熵塌缩导致的回应重复性，我们引入了基于个性的对话重写和重复性感知的奖励重赋值策略。我们的方法显著提升了模型的情感支持能力，推动了具有共情和人性化支持系统的开发。 

---
# Episodic Memory Representation for Long-form Video Understanding 

**Title (ZH)**: 长视频理解中的情景记忆表示 

**Authors**: Yun Wang, Long Zhang, Jingren Liu, Jiaqi Yan, Zhanjie Zhang, Jiahao Zheng, Xun Yang, Dapeng Wu, Xiangyu Chen, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09486)  

**Abstract**: Video Large Language Models (Video-LLMs) excel at general video understanding but struggle with long-form videos due to context window limits. Consequently, recent approaches focus on keyframe retrieval, condensing lengthy videos into a small set of informative frames. Despite their practicality, these methods simplify the problem to static text image matching, overlooking spatio temporal relationships crucial for capturing scene transitions and contextual continuity, and may yield redundant keyframes with limited information, diluting salient cues essential for accurate video question answering. To address these limitations, we introduce Video-EM, a training free framework inspired by the principles of human episodic memory, designed to facilitate robust and contextually grounded reasoning. Rather than treating keyframes as isolated visual entities, Video-EM explicitly models them as temporally ordered episodic events, capturing both spatial relationships and temporal dynamics necessary for accurately reconstructing the underlying narrative. Furthermore, the framework leverages chain of thought (CoT) thinking with LLMs to iteratively identify a minimal yet highly informative subset of episodic memories, enabling efficient and accurate question answering by Video-LLMs. Extensive evaluations on the Video-MME, EgoSchema, HourVideo, and LVBench benchmarks confirm the superiority of Video-EM, which achieves highly competitive results with performance gains of 4-9 percent over respective baselines while utilizing fewer frames. 

**Abstract (ZH)**: 基于人类情景记忆原则的Video-EM：一种无需训练的框架，适用于视频理解与高效问答 

---
# What-Meets-Where: Unified Learning of Action and Contact Localization in a New Dataset 

**Title (ZH)**: 何遇何地：新数据集中的动作与接触定位联合学习 

**Authors**: Yuxiao Wang, Yu Lei, Wolin Liang, Weiying Xue, Zhenao Wei, Nan Zhuang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09428)  

**Abstract**: People control their bodies to establish contact with the environment. To comprehensively understand actions across diverse visual contexts, it is essential to simultaneously consider \textbf{what} action is occurring and \textbf{where} it is happening. Current methodologies, however, often inadequately capture this duality, typically failing to jointly model both action semantics and their spatial contextualization within scenes. To bridge this gap, we introduce a novel vision task that simultaneously predicts high-level action semantics and fine-grained body-part contact regions. Our proposed framework, PaIR-Net, comprises three key components: the Contact Prior Aware Module (CPAM) for identifying contact-relevant body parts, the Prior-Guided Concat Segmenter (PGCS) for pixel-wise contact segmentation, and the Interaction Inference Module (IIM) responsible for integrating global interaction relationships. To facilitate this task, we present PaIR (Part-aware Interaction Representation), a comprehensive dataset containing 13,979 images that encompass 654 actions, 80 object categories, and 17 body parts. Experimental evaluation demonstrates that PaIR-Net significantly outperforms baseline approaches, while ablation studies confirm the efficacy of each architectural component. The code and dataset will be released upon publication. 

**Abstract (ZH)**: 人体控制其动作以与环境建立联系。为了全面理解跨多样化视觉场景的动作，同时考虑动作的类型及其发生的位置至关重要。现有方法往往未能充分捕捉到这一双重性，通常无法同时建模动作语义及其在场景中的空间上下文。为弥补这一差距，我们提出了一个新的视觉任务，同时预测高层次的动作语义和细粒度的身体部位接触区域。我们提出的PaIR-Net框架包括三个关键组件：接触先验感知模块（CPAM）用于识别与接触相关的身体部位、先验引导 CONCAT 聚合同步分割器（PGCS）用于像素级接触分割，以及交互推理模块（IIM）用于整合全局交互关系。为了促进此任务，我们发布了PaIR（部分感知交互表示）数据集，包含13,979张图像，涵盖了654种动作、80种物体类别和17种身体部位。实验评价表明，PaIR-Net 显著优于基线方法，且消融研究证实了每个架构组件的有效性。代码和数据集将在发表时公开。 

---
# Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models: A Unified and Accurate Approach 

**Title (ZH)**: 在大型视觉-语言模型中学习检测未知越狱攻击：一种统一且准确的方法 

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09201)  

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. Although recent detection works have shifted to internal representations due to their rich cross-modal information, most methods rely on heuristic rules rather than principled objectives, resulting in suboptimal performance. To address these limitations, we propose Learning to Detect (LoD), a novel unsupervised framework that formulates jailbreak detection as anomaly detection. LoD introduces two key components: Multi-modal Safety Concept Activation Vectors (MSCAV), which capture layer-wise safety-related representations across modalities, and the Safety Pattern Auto-Encoder, which models the distribution of MSCAV derived from safe inputs and detects anomalies via reconstruction errors. By training the auto-encoder (AE) solely on safe samples without attack labels, LoD naturally identifies jailbreak inputs as distributional anomalies, enabling accurate and unified detection of jailbreak attacks. Comprehensive experiments on three different LVLMs and five benchmarks demonstrate that LoD achieves state-of-the-art performance, with an average AUROC of 0.9951 and an improvement of up to 38.89% in the minimum AUROC over the strongest baselines. 

**Abstract (ZH)**: 尽管进行了广泛的努力对齐，大型视觉-语言模型（LVLMs）仍然容易受到劫持攻击，这带来了严重的安全风险。尽管最近的检测工作转向了内部表示，以利用其丰富的跨模态信息，但大多数方法仍然依赖于启发式规则而不是原理性的目标，导致性能不佳。为解决这些限制，我们提出了学习检测（LoD），这是一种新颖的无监督框架，将劫持检测公式化为异常检测。LoD 引入了两个关键组件：多模态安全性概念激活向量（MSCAV），用于捕捉各模态的分层安全性相关表示，以及安全性模式自编码器，该自编码器建模来自安全输入的 MSCAV 的分布，并通过重构误差检测异常。通过仅使用安全样本而不使用攻击标签训练自编码器（AE），LoD 自然而然地将劫持输入识别为分布异常，从而实现对劫持攻击的准确和统一检测。针对三个不同 LVLM 和五个基准的全面实验表明，LoD 达到了最先进的性能，平均 AUCROC 为 0.9951，并且相对于最强基线在最小 AUCROC 上的提升最高达 38.89%。 

---
# MX-AI: Agentic Observability and Control Platform for Open and AI-RAN 

**Title (ZH)**: MX-AI: 为开放和AI-RAN的代理可观测性和控制平台 

**Authors**: Ilias Chatzistefanidis, Andrea Leone, Ali Yaghoubian, Mikel Irazabal, Sehad Nassim, Lina Bariah, Merouane Debbah, Navid Nikaein  

**Link**: [PDF](https://arxiv.org/pdf/2508.09197)  

**Abstract**: Future 6G radio access networks (RANs) will be artificial intelligence (AI)-native: observed, reasoned about, and re-configured by autonomous agents cooperating across the cloud-edge continuum. We introduce MX-AI, the first end-to-end agentic system that (i) instruments a live 5G Open RAN testbed based on OpenAirInterface (OAI) and FlexRIC, (ii) deploys a graph of Large-Language-Model (LLM)-powered agents inside the Service Management and Orchestration (SMO) layer, and (iii) exposes both observability and control functions for 6G RAN resources through natural-language intents. On 50 realistic operational queries, MX-AI attains a mean answer quality of 4.1/5.0 and 100 % decision-action accuracy, while incurring only 8.8 seconds end-to-end latency when backed by GPT-4.1. Thus, it matches human-expert performance, validating its practicality in real settings. We publicly release the agent graph, prompts, and evaluation harness to accelerate open research on AI-native RANs. A live demo is presented here: this https URL 

**Abstract (ZH)**: 未来的6G无线接入网络（RAN）将具有人工智能（AI）原生特性：通过跨云边 continuum 的自主代理进行观察、推理和重构。我们引入了MX-AI，这是一个端到端的自主系统，它在基于OpenAirInterface (OAI) 和 FlexRIC 的实时5G Open RAN试验床上进行仪器化，部署了Service Management and Orchestration (SMO) 层内的Large-Language-Model (LLM) 动力代理图，并通过自然语言意图暴露6G RAN资源的可观测性和控制功能。在50个现实的操作查询中，MX-AI的平均回答质量为4.1/5.0，决策-行动准确率为100%，并在支持GPT-4.1的情况下仅产生8.8秒的端到端延迟，从而达到了人类专家的性能水平，验证了其在实际环境中的实用性。我们公开发布代理图、提示和评估框架，以加速对AI原生RAN的开放研究。演示链接：this https URL 

---
# EvaDrive: Evolutionary Adversarial Policy Optimization for End-to-End Autonomous Driving 

**Title (ZH)**: EvaDrive: 进化对抗策略优化在端到端自动驾驶中的应用 

**Authors**: Siwen Jiao, Kangan Qian, Hao Ye, Yang Zhong, Ziang Luo, Sicong Jiang, Zilin Huang, Yangyi Fang, Jinyu Miao, Zheng Fu, Yunlong Wang, Kun Jiang, Diange Yang, Rui Fan, Baoyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09158)  

**Abstract**: Autonomous driving faces significant challenges in achieving human-like iterative decision-making, which continuously generates, evaluates, and refines trajectory proposals. Current generation-evaluation frameworks isolate trajectory generation from quality assessment, preventing iterative refinement essential for planning, while reinforcement learning methods collapse multi-dimensional preferences into scalar rewards, obscuring critical trade-offs and yielding scalarization this http URL overcome these issues, we present EvaDrive, a novel multi-objective reinforcement learning framework that establishes genuine closed-loop co-evolution between trajectory generation and evaluation via adversarial optimization. EvaDrive frames trajectory planning as a multi-round adversarial game. In this game, a hierarchical generator continuously proposes candidate paths by combining autoregressive intent modeling for temporal causality with diffusion-based refinement for spatial flexibility. These proposals are then rigorously assessed by a trainable multi-objective critic that explicitly preserves diverse preference structures without collapsing them into a single scalarization this http URL adversarial interplay, guided by a Pareto frontier selection mechanism, enables iterative multi-round refinement, effectively escaping local optima while preserving trajectory this http URL experiments on NAVSIM and Bench2Drive benchmarks demonstrate SOTA performance, achieving 94.9 PDMS on NAVSIM v1 (surpassing DiffusionDrive by 6.8, DriveSuprim by 5.0, and TrajHF by 0.9) and 64.96 Driving Score on Bench2Drive. EvaDrive generates diverse driving styles via dynamic weighting without external preference data, introducing a closed-loop adversarial framework for human-like iterative decision-making, offering a novel scalarization-free trajectory optimization approach. 

**Abstract (ZH)**: 自主驾驶在实现类人类的迭代决策方面面临重大挑战，这需要不断生成、评估和改进轨迹提案。当前的生成-评估框架将轨迹生成与质量评估隔离，阻碍了规划中必不可少的迭代细化。而强化学习方法将多维偏好压缩为单一标量奖励，模糊了关键权衡，导致标量化。为解决这些问题，我们提出了EvaDrive，一种新颖的多目标强化学习框架，通过对抗优化在轨迹生成和评估之间建立真实的闭环共进化。EvaDrive将轨迹规划框架化为多轮对抗游戏。在此游戏中，层次生成器通过结合自回归意图建模来实现时间因果性，结合基于扩散的精细调整来实现空间灵活性，不断提出候选路径。这些提案然后由可训练的多目标批评家严格评估，后者明确保留了多样的偏好结构，而不会将其压缩成单一标量化。这一对抗互动，由帕累托前沿选择机制引导，实现迭代多轮细化，有效逃脱局部最优解，同时保留轨迹。在NAVSIM和Bench2Drive基准测试上的实验展示了SOTA性能，其中EvaDrive在NAVSIM v1上的PDMS得分为94.9（超越DiffusionDrive 6.8、DriveSuprim 5.0和TrajHF 0.9），Bench2Drive上的驾驶得分为64.96。EvaDrive通过动态权重生成多样化的驾驶风格，无需外部偏好数据，引入了一种闭环对抗框架，用于类人的迭代决策，提供了一种无标量化轨迹优化的新方法。 

---
# User-Intent-Driven Semantic Communication via Adaptive Deep Understanding 

**Title (ZH)**: 基于用户意图的自适应深度语义通信 

**Authors**: Peigen Ye, Jingpu Duan, Hongyang Du, Yulan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05884)  

**Abstract**: Semantic communication focuses on transmitting task-relevant semantic information, aiming for intent-oriented communication. While existing systems improve efficiency by extracting key semantics, they still fail to deeply understand and generalize users' real intentions. To overcome this, we propose a user-intention-driven semantic communication system that interprets diverse abstract intents. First, we integrate a multi-modal large model as semantic knowledge base to generate user-intention prior. Next, a mask-guided attention module is proposed to effectively highlight critical semantic regions. Further, a channel state awareness module ensures adaptive, robust transmission across varying channel conditions. Extensive experiments demonstrate that our system achieves deep intent understanding and outperforms DeepJSCC, e.g., under a Rayleigh channel at an SNR of 5 dB, it achieves improvements of 8%, 6%, and 19% in PSNR, SSIM, and LPIPS, respectively. 

**Abstract (ZH)**: 面向意图的语义通信系统研究：基于多元模态大模型的抽象意图解释与适应性鲁棒传输 

---
