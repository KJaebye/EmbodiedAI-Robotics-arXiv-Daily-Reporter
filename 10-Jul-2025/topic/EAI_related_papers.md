# ULC: A Unified and Fine-Grained Controller for Humanoid Loco-Manipulation 

**Title (ZH)**: ULC：统一细粒度 humanoid 人類抓举控制单元 

**Authors**: Wandong Sun, Luying Feng, Baoshi Cao, Yang Liu, Yaochu Jin, Zongwu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.06905)  

**Abstract**: Loco-Manipulation for humanoid robots aims to enable robots to integrate mobility with upper-body tracking capabilities. Most existing approaches adopt hierarchical architectures that decompose control into isolated upper-body (manipulation) and lower-body (locomotion) policies. While this decomposition reduces training complexity, it inherently limits coordination between subsystems and contradicts the unified whole-body control exhibited by humans. We demonstrate that a single unified policy can achieve a combination of tracking accuracy, large workspace, and robustness for humanoid loco-manipulation. We propose the Unified Loco-Manipulation Controller (ULC), a single-policy framework that simultaneously tracks root velocity, root height, torso rotation, and dual-arm joint positions in an end-to-end manner, proving the feasibility of unified control without sacrificing performance. We achieve this unified control through key technologies: sequence skill acquisition for progressive learning complexity, residual action modeling for fine-grained control adjustments, command polynomial interpolation for smooth motion transitions, random delay release for robustness to deploy variations, load randomization for generalization to external disturbances, and center-of-gravity tracking for providing explicit policy gradients to maintain stability. We validate our method on the Unitree G1 humanoid robot with 3-DOF (degrees-of-freedom) waist. Compared with strong baselines, ULC shows better tracking performance to disentangled methods and demonstrating larger workspace coverage. The unified dual-arm tracking enables precise manipulation under external loads while maintaining coordinated whole-body control for complex loco-manipulation tasks. 

**Abstract (ZH)**: 人形机器人的结合操作与移动控制研究：统一策略实现精确操作与广泛工作空间 

---
# Hierarchical Reinforcement Learning for Articulated Tool Manipulation with Multifingered Hand 

**Title (ZH)**: 多指手驱动 articulated 工具 manipulation 的分层强化学习 

**Authors**: Wei Xu, Yanchao Zhao, Weichao Guo, Xinjun Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06822)  

**Abstract**: Manipulating articulated tools, such as tweezers or scissors, has rarely been explored in previous research. Unlike rigid tools, articulated tools change their shape dynamically, creating unique challenges for dexterous robotic hands. In this work, we present a hierarchical, goal-conditioned reinforcement learning (GCRL) framework to improve the manipulation capabilities of anthropomorphic robotic hands using articulated tools. Our framework comprises two policy layers: (1) a low-level policy that enables the dexterous hand to manipulate the tool into various configurations for objects of different sizes, and (2) a high-level policy that defines the tool's goal state and controls the robotic arm for object-picking tasks. We employ an encoder, trained on synthetic pointclouds, to estimate the tool's affordance states--specifically, how different tool configurations (e.g., tweezer opening angles) enable grasping of objects of varying sizes--from input point clouds, thereby enabling precise tool manipulation. We also utilize a privilege-informed heuristic policy to generate replay buffer, improving the training efficiency of the high-level policy. We validate our approach through real-world experiments, showing that the robot can effectively manipulate a tweezer-like tool to grasp objects of diverse shapes and sizes with a 70.8 % success rate. This study highlights the potential of RL to advance dexterous robotic manipulation of articulated tools. 

**Abstract (ZH)**: 操纵articulated工具（如镊子或剪刀）的研究在以往的工作中较少探索。与刚性工具不同，articulated工具会动态改变形状，为灵巧的手部机器人带来了独特的挑战。在这项工作中，我们提出了一种分层的目标条件强化学习（GCRL）框架，以提高类人手部机器人使用articulated工具的操纵能力。该框架包含两个策略层：（1）一个低层策略，使灵巧的手部能够根据不同大小的对象调整工具的配置；（2）一个高层策略，定义工具的目标状态并控制机器臂进行物体拾取任务。我们利用一个在合成点云上训练的编码器，从输入点云中估计工具的功能状态，特别是不同工具配置（例如镊子张开的角度）如何使不同大小的对象抓取成为可能，从而实现精确的工具操纵。我们还利用基于特权的启发式策略生成回放缓冲区，提高了高层策略的训练效率。通过现实世界的实验验证了我们的方法，结果显示机器人能够以70.8%的成功率有效地操纵类似镊子的工具来抓取各种形状和大小的对象。本研究突显了RL在推进articulated工具的灵巧机器人操控方面的潜力。 

---
# Distributed Fault-Tolerant Multi-Robot Cooperative Localization in Adversarial Environments 

**Title (ZH)**: 分布式鲁棒多机器人协同定位技术在对抗环境中 

**Authors**: Tohid Kargar Tasooji, Ramviyas Parasuraman  

**Link**: [PDF](https://arxiv.org/pdf/2507.06750)  

**Abstract**: In multi-robot systems (MRS), cooperative localization is a crucial task for enhancing system robustness and scalability, especially in GPS-denied or communication-limited environments. However, adversarial attacks, such as sensor manipulation, and communication jamming, pose significant challenges to the performance of traditional localization methods. In this paper, we propose a novel distributed fault-tolerant cooperative localization framework to enhance resilience against sensor and communication disruptions in adversarial environments. We introduce an adaptive event-triggered communication strategy that dynamically adjusts communication thresholds based on real-time sensing and communication quality. This strategy ensures optimal performance even in the presence of sensor degradation or communication failure. Furthermore, we conduct a rigorous analysis of the convergence and stability properties of the proposed algorithm, demonstrating its resilience against bounded adversarial zones and maintaining accurate state estimation. Robotarium-based experiment results show that our proposed algorithm significantly outperforms traditional methods in terms of localization accuracy and communication efficiency, particularly in adversarial settings. Our approach offers improved scalability, reliability, and fault tolerance for MRS, making it suitable for large-scale deployments in real-world, challenging environments. 

**Abstract (ZH)**: 多机器人系统中基于对抗环境的分布式容错协同定位框架 

---
# LOVON: Legged Open-Vocabulary Object Navigator 

**Title (ZH)**: LOVON: 腿足开放式词汇对象导航器 

**Authors**: Daojie Peng, Jiahang Cao, Qiang Zhang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.06747)  

**Abstract**: Object navigation in open-world environments remains a formidable and pervasive challenge for robotic systems, particularly when it comes to executing long-horizon tasks that require both open-world object detection and high-level task planning. Traditional methods often struggle to integrate these components effectively, and this limits their capability to deal with complex, long-range navigation missions. In this paper, we propose LOVON, a novel framework that integrates large language models (LLMs) for hierarchical task planning with open-vocabulary visual detection models, tailored for effective long-range object navigation in dynamic, unstructured environments. To tackle real-world challenges including visual jittering, blind zones, and temporary target loss, we design dedicated solutions such as Laplacian Variance Filtering for visual stabilization. We also develop a functional execution logic for the robot that guarantees LOVON's capabilities in autonomous navigation, task adaptation, and robust task completion. Extensive evaluations demonstrate the successful completion of long-sequence tasks involving real-time detection, search, and navigation toward open-vocabulary dynamic targets. Furthermore, real-world experiments across different legged robots (Unitree Go2, B2, and H1-2) showcase the compatibility and appealing plug-and-play feature of LOVON. 

**Abstract (ZH)**: 开放世界环境中的物体导航仍然是机器人系统面临的一项艰巨且普遍的挑战，尤其是在执行长时任务时，这些任务需要开放世界物体检测和高级任务规划的结合。传统方法往往难以有效集成这些组件，从而限制了其处理复杂、长距离导航任务的能力。本文提出了一种新的框架LOVON，该框架结合了层次任务规划的大语言模型（LLMs）与面向开放词汇视觉检测的模型，专门针对动态、非结构化环境中的有效长距离物体导航。为了应对包括视觉抖动、盲区和目标暂时丢失在内的现实世界挑战，我们设计了专门的解决方案，如拉普拉斯方差滤波用于视觉稳定。我们还为机器人开发了功能执行逻辑，以确保LOVON在自主导航、任务适应和稳健任务完成方面的能力。广泛的评估表明，LOVON能够成功完成涉及实时检测、搜索和导航至开放词汇动态目标的长时间序列任务。此外，跨不同腿足机器人（Unitree Go2、B2和H1-2）的实际实验展示了LOVON的兼容性和方便的即插即用特性。 

---
# Spatial-Temporal Aware Visuomotor Diffusion Policy Learning 

**Title (ZH)**: 空间-时间知觉运动扩散策略学习 

**Authors**: Zhenyang Liu, Yikai Wang, Kuanning Wang, Longfei Liang, Xiangyang Xue, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06710)  

**Abstract**: Visual imitation learning is effective for robots to learn versatile tasks. However, many existing methods rely on behavior cloning with supervised historical trajectories, limiting their 3D spatial and 4D spatiotemporal awareness. Consequently, these methods struggle to capture the 3D structures and 4D spatiotemporal relationships necessary for real-world deployment. In this work, we propose 4D Diffusion Policy (DP4), a novel visual imitation learning method that incorporates spatiotemporal awareness into diffusion-based policies. Unlike traditional approaches that rely on trajectory cloning, DP4 leverages a dynamic Gaussian world model to guide the learning of 3D spatial and 4D spatiotemporal perceptions from interactive environments. Our method constructs the current 3D scene from a single-view RGB-D observation and predicts the future 3D scene, optimizing trajectory generation by explicitly modeling both spatial and temporal dependencies. Extensive experiments across 17 simulation tasks with 173 variants and 3 real-world robotic tasks demonstrate that the 4D Diffusion Policy (DP4) outperforms baseline methods, improving the average simulation task success rate by 16.4% (Adroit), 14% (DexArt), and 6.45% (RLBench), and the average real-world robotic task success rate by 8.6%. 

**Abstract (ZH)**: 4D扩散策略(DP4)：一种融入时空意识的视觉模仿学习方法 

---
# Integrating Perceptions: A Human-Centered Physical Safety Model for Human-Robot Interaction 

**Title (ZH)**: 集成感知：以人为本的物理安全模型在人机交互中的应用 

**Authors**: Pranav Pandey, Ramviyas Parasuraman, Prashant Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06700)  

**Abstract**: Ensuring safety in human-robot interaction (HRI) is essential to foster user trust and enable the broader adoption of robotic systems. Traditional safety models primarily rely on sensor-based measures, such as relative distance and velocity, to assess physical safety. However, these models often fail to capture subjective safety perceptions, which are shaped by individual traits and contextual factors. In this paper, we introduce and analyze a parameterized general safety model that bridges the gap between physical and perceived safety by incorporating a personalization parameter, $\rho$, into the safety measurement framework to account for individual differences in safety perception. Through a series of hypothesis-driven human-subject studies in a simulated rescue scenario, we investigate how emotional state, trust, and robot behavior influence perceived safety. Our results show that $\rho$ effectively captures meaningful individual differences, driven by affective responses, trust in task consistency, and clustering into distinct user types. Specifically, our findings confirm that predictable and consistent robot behavior as well as the elicitation of positive emotional states, significantly enhance perceived safety. Moreover, responses cluster into a small number of user types, supporting adaptive personalization based on shared safety models. Notably, participant role significantly shapes safety perception, and repeated exposure reduces perceived safety for participants in the casualty role, emphasizing the impact of physical interaction and experiential change. These findings highlight the importance of adaptive, human-centered safety models that integrate both psychological and behavioral dimensions, offering a pathway toward more trustworthy and effective HRI in safety-critical domains. 

**Abstract (ZH)**: 确保人机交互中的安全性对于培养用户信任并促进机器人系统的广泛应用至关重要。传统的安全性模型主要依赖于基于传感器的措施，如相对距离和速度，来评估物理安全性。然而，这些模型往往未能捕捉到主观的安全感受，后者是由个体特质和情境因素形成的。在本文中，我们引入并分析了一个参数化的通用安全性模型，通过在安全性测量框架中引入个人化参数 $\rho$ 来弥合物理安全与感知安全之间的差距，以考虑安全感知的个体差异。通过一系列基于假设的人类主体研究，在模拟的救援场景中，我们探讨了情绪状态、信任和机器人行为如何影响感知安全。研究结果表明，$\rho$ 有效地捕捉到了由情感反应、任务一致性的信任以及用户类型分群驱动的有意义的个体差异。具体而言，我们的研究结果证实，可预测和一致的机器人行为以及唤起积极情感状态，显著增强了感知安全。此外，响应呈现出少量用户类型的分群，支持基于共享安全性模型的适应性个性化。值得注意的是，参与者角色显著影响了安全性感知，反复接触降低了伤亡角色参与者感知的安全性，突显了物理交互和体验变化的影响。这些发现强调了整合心理和行为维度的适应性和以人为中心的安全模型的重要性，为在关键安全领域实现更值得信赖和有效的HRI提供了途径。 

---
# Multi-Task Multi-Agent Reinforcement Learning via Skill Graphs 

**Title (ZH)**: 基于技能图的多任务多代理强化学习 

**Authors**: Guobin Zhu, Rui Zhou, Wenkang Ji, Hongyin Zhang, Donglin Wang, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06690)  

**Abstract**: Multi-task multi-agent reinforcement learning (MT-MARL) has recently gained attention for its potential to enhance MARL's adaptability across multiple tasks. However, it is challenging for existing multi-task learning methods to handle complex problems, as they are unable to handle unrelated tasks and possess limited knowledge transfer capabilities. In this paper, we propose a hierarchical approach that efficiently addresses these challenges. The high-level module utilizes a skill graph, while the low-level module employs a standard MARL algorithm. Our approach offers two contributions. First, we consider the MT-MARL problem in the context of unrelated tasks, expanding the scope of MTRL. Second, the skill graph is used as the upper layer of the standard hierarchical approach, with training independent of the lower layer, effectively handling unrelated tasks and enhancing knowledge transfer capabilities. Extensive experiments are conducted to validate these advantages and demonstrate that the proposed method outperforms the latest hierarchical MAPPO algorithms. Videos and code are available at this https URL 

**Abstract (ZH)**: 多任务多智能体强化学习（MT-MARL）近年来因其在多任务环境下的适应性增强而受到关注。然而，现有的多任务学习方法难以处理复杂问题，因为它们无法处理无关任务并且知识迁移能力有限。本文提出了一种分层方法来有效应对这些挑战。高层次模块利用技能图，低层次模块采用标准的多智能体强化学习算法。本文方法做出了两个贡献。首先，我们在无关任务的背景下考虑MT-MARL问题，拓展了多任务强化学习迁移学习（MTRL）的范围。其次，技能图作为标准分层方法的高层，训练与低层独立，有效地处理无关任务并增强知识迁移能力。大量的实验证明了这些优势，并展示了所提出的方法在与最新分层MAPPO算法相比的优越性。有关视频和代码可在以下链接获取。 

---
# AI Space Cortex: An Experimental System for Future Era Space Exploration 

**Title (ZH)**: AI太空 cortex：未来时代太空探索的实验系统 

**Authors**: Thomas Touma, Ersin Daş, Erica Tevere, Martin Feather, Ksenia Kolcio, Maurice Prather, Alberto Candela, Ashish Goel, Erik Kramer, Hari Nayar, Lorraine Fesq, Joel W. Burdick  

**Link**: [PDF](https://arxiv.org/pdf/2507.06574)  

**Abstract**: Our Robust, Explainable Autonomy for Scientific Icy Moon Operations (REASIMO) effort contributes to NASA's Concepts for Ocean worlds Life Detection Technology (COLDTech) program, which explores science platform technologies for ocean worlds such as Europa and Enceladus. Ocean world missions pose significant operational challenges. These include long communication lags, limited power, and lifetime limitations caused by radiation damage and hostile conditions. Given these operational limitations, onboard autonomy will be vital for future Ocean world missions. Besides the management of nominal lander operations, onboard autonomy must react appropriately in the event of anomalies. Traditional spacecraft rely on a transition into 'safe-mode' in which non-essential components and subsystems are powered off to preserve safety and maintain communication with Earth. For a severely time-limited Ocean world mission, resolutions to these anomalies that can be executed without Earth-in-the-loop communication and associated delays are paramount for completion of the mission objectives and science goals. To address these challenges, the REASIMO effort aims to demonstrate a robust level of AI-assisted autonomy for such missions, including the ability to detect and recover from anomalies, and to perform missions based on pre-trained behaviors rather than hard-coded, predetermined logic like all prior space missions. We developed an AI-assisted, personality-driven, intelligent framework for control of an Ocean world mission by combining a mix of advanced technologies. To demonstrate the capabilities of the framework, we perform tests of autonomous sampling operations on a lander-manipulator testbed at the NASA Jet Propulsion Laboratory, approximating possible surface conditions such a mission might encounter. 

**Abstract (ZH)**: 我们鲁棒可解释的冰卫星科学探测自主性（REASIMO）项目为NASA的海洋世界生命探测技术（COLDTech）计划贡献力量，该计划探索诸如欧罗巴和恩赛lasses的海洋世界科学平台技术。海洋世界任务面临着重大操作挑战，包括长通信延迟、有限的电力供应以及由辐射损伤和恶劣环境引起的寿命限制。鉴于这些操作限制，未来的海洋世界任务将依赖于机载自主性。除了常规着陆器操作的管理之外，机载自主性还必须在出现异常时适当地作出反应。传统航天器依赖于一种“安全模式”转移，其中非必需的组件和子系统会被断电以保持安全并维持与地球的通信。对于时间极其有限的海洋世界任务，无需地球干预即可执行的异常解决策略对于完成任务目标和科学目标至关重要。为此，REASIMO项目旨在展示一种适用于此类任务的鲁棒人工智能辅助自主性，包括检测和恢复异常的能力，以及基于预先训练的行为而非所有先前太空任务中的硬编码预定逻辑来进行任务。我们通过结合多种先进技术，开发了一种人工智能辅助、个性驱动的智能控制框架来管理海洋世界任务。为了展示该框架的能力，我们在NASA喷气推进实验室的着陆器操作测试台上进行自主采样操作测试，模拟此类任务可能遇到的表面条件。 

---
# SkyVLN: Vision-and-Language Navigation and NMPC Control for UAVs in Urban Environments 

**Title (ZH)**: SkyVLN: 无人机在城市环境中的视觉-语言导航与NMPC控制 

**Authors**: Tianshun Li, Tianyi Huai, Zhen Li, Yichun Gao, Haoang Li, Xinhu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06564)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) have emerged as versatile tools across various sectors, driven by their mobility and adaptability. This paper introduces SkyVLN, a novel framework integrating vision-and-language navigation (VLN) with Nonlinear Model Predictive Control (NMPC) to enhance UAV autonomy in complex urban environments. Unlike traditional navigation methods, SkyVLN leverages Large Language Models (LLMs) to interpret natural language instructions and visual observations, enabling UAVs to navigate through dynamic 3D spaces with improved accuracy and robustness. We present a multimodal navigation agent equipped with a fine-grained spatial verbalizer and a history path memory mechanism. These components allow the UAV to disambiguate spatial contexts, handle ambiguous instructions, and backtrack when necessary. The framework also incorporates an NMPC module for dynamic obstacle avoidance, ensuring precise trajectory tracking and collision prevention. To validate our approach, we developed a high-fidelity 3D urban simulation environment using AirSim, featuring realistic imagery and dynamic urban elements. Extensive experiments demonstrate that SkyVLN significantly improves navigation success rates and efficiency, particularly in new and unseen environments. 

**Abstract (ZH)**: 无人机（UAVs）已作为一种多功能工具在各个领域中涌现，得益于其移动性和适应性。本文介绍了SkyVLN，这是一种将视觉-语言导航（VLN）与非线性模型预测控制（NMPC）集成的新 framework，以增强无人机在复杂城市环境中的自主性。与传统的导航方法不同，SkyVLN 利用大型语言模型（LLMs）解释自然语言指令和视觉观察，使无人机能够在动态3D空间中导航，提高导航的准确性和鲁棒性。本文提出了一种多模态导航代理，配备了精细的空间语言化器和历史路径记忆机制。这些组件使无人机能够消除空间歧义、处理模糊指令并在必要时回退。该框架还集成了一个动态障碍规避的NMPC模块，确保精确轨迹跟踪和防碰撞功能。为了验证我们的方法，我们使用AirSim开发了一个高保真3D城市仿真环境，含有逼真图像和动态城市元素。大量实验表明，SkyVLN 显著提高了导航成功率和效率，尤其是在新的和未见过的环境中。 

---
# KLEIYN : A Quadruped Robot with an Active Waist for Both Locomotion and Wall Climbing 

**Title (ZH)**: KLEIYN：具备主动腰部用于行进和攀壁的四足机器人 

**Authors**: Keita Yoneda, Kento Kawaharazuka, Temma Suzuki, Takahiro Hattori, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2507.06562)  

**Abstract**: In recent years, advancements in hardware have enabled quadruped robots to operate with high power and speed, while robust locomotion control using reinforcement learning (RL) has also been realized. As a result, expectations are rising for the automation of tasks such as material transport and exploration in unknown environments. However, autonomous locomotion in rough terrains with significant height variations requires vertical movement, and robots capable of performing such movements stably, along with their control methods, have not yet been fully established. In this study, we developed the quadruped robot KLEIYN, which features a waist joint, and aimed to expand quadruped locomotion by enabling chimney climbing through RL. To facilitate the learning of vertical motion, we introduced Contact-Guided Curriculum Learning (CGCL). As a result, KLEIYN successfully climbed walls ranging from 800 mm to 1000 mm in width at an average speed of 150 mm/s, 50 times faster than conventional robots. Furthermore, we demonstrated that the introduction of a waist joint improves climbing performance, particularly enhancing tracking ability on narrow walls. 

**Abstract (ZH)**: 近年来，硬件的进步使四足机器人能够以高功率和速度运行，同时利用强化学习（RL）实现 robust 的运动控制也已成为现实。因此，对诸如材料运输和未知环境探索等任务的自动化期望正在上升。然而，在具有显著高度变化的崎岖地形上实现自主运动仍然需要垂直运动，而能够稳定执行此类运动的机器人及其控制方法尚未完全建立。在这项研究中，我们开发了配备了腰部关节的四足机器人 KLEIYN，并通过 RL 实现烟囱攀爬，旨在通过 RL 扩展四足运动。为了促进垂直运动的学习，我们引入了接触引导 Curriculum Learning（CGCL）。结果显示，KLEIYN 成功以 150 mm/s 的平均速度攀爬了宽度从 800 mm 到 1000 mm 的墙面，比传统机器人快 50 倍。此外，我们展示了腰部关节的引入提高了攀爬性能，尤其是在狭壁跟踪能力方面的提升。 

---
# Evaluating Robots Like Human Infants: A Case Study of Learned Bipedal Locomotion 

**Title (ZH)**: 评价机器人如人类婴儿般：双足运动学习的案例研究 

**Authors**: Devin Crowley, Whitney G. Cole, Christina M. Hospodar, Ruiting Shen, Karen E. Adolph, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2507.06426)  

**Abstract**: Typically, learned robot controllers are trained via relatively unsystematic regimens and evaluated with coarse-grained outcome measures such as average cumulative reward. The typical approach is useful to compare learning algorithms but provides limited insight into the effects of different training regimens and little understanding about the richness and complexity of learned behaviors. Likewise, human infants and other animals are "trained" via unsystematic regimens, but in contrast, developmental psychologists evaluate their performance in highly-controlled experiments with fine-grained measures such as success, speed of walking, and prospective adjustments. However, the study of learned behavior in human infants is limited by the practical constraints of training and testing babies. Here, we present a case study that applies methods from developmental psychology to study the learned behavior of the simulated bipedal robot Cassie. Following research on infant walking, we systematically designed reinforcement learning training regimens and tested the resulting controllers in simulated environments analogous to those used for babies--but without the practical constraints. Results reveal new insights into the behavioral impact of different training regimens and the development of Cassie's learned behaviors relative to infants who are learning to walk. This interdisciplinary baby-robot approach provides inspiration for future research designed to systematically test effects of training on the development of complex learned robot behaviors. 

**Abstract (ZH)**: 通常，机器人控制器是通过相对非系统的训练程序进行训练，并且通过粗糙的结果指标，如平均累积奖励来进行评估。通常的方法有助于比较学习算法，但提供了有限的关于不同训练程序效果的洞察，并且对学到的行为的丰富性和复杂性缺乏理解。同样，人类婴儿和其他动物也是通过非系统的程序进行“训练”，但相比之下，发展心理学家通过精细测量，如成功、行走速度和前瞻性调整，在高度控制的实验中评估他们的表现。然而，对人类婴儿学到的行为的研究受限于培训和测试婴儿的实际约束。在这里，我们呈现了一个案例研究，将发展心理学的方法应用于研究模拟双足机器人Cassie学到的行为。借鉴婴儿行走的研究，我们系统地设计了强化学习训练程序，并在模拟环境中测试了生成的控制器，这些模拟环境类似于婴儿使用的环境——但没有实际约束。结果揭示了不同训练程序对行为影响的新见解，以及Cassie学到的行为与其正在学习行走的婴儿相比的发展情况。这种跨学科的婴儿-机器人方法为未来旨在系统测试训练对复杂机器人行为发展影响的研究提供了 inspiration。 

---
# Learning to Evaluate Autonomous Behaviour in Human-Robot Interaction 

**Title (ZH)**: 学习评估自主行为的人机交互 

**Authors**: Matteo Tiezzi, Tommaso Apicella, Carlos Cardenas-Perez, Giovanni Fregonese, Stefano Dafarra, Pietro Morerio, Daniele Pucci, Alessio Del Bue  

**Link**: [PDF](https://arxiv.org/pdf/2507.06404)  

**Abstract**: Evaluating and comparing the performance of autonomous Humanoid Robots is challenging, as success rate metrics are difficult to reproduce and fail to capture the complexity of robot movement trajectories, critical in Human-Robot Interaction and Collaboration (HRIC). To address these challenges, we propose a general evaluation framework that measures the quality of Imitation Learning (IL) methods by focusing on trajectory performance. We devise the Neural Meta Evaluator (NeME), a deep learning model trained to classify actions from robot joint trajectories. NeME serves as a meta-evaluator to compare the performance of robot control policies, enabling policy evaluation without requiring human involvement in the loop. We validate our framework on ergoCub, a humanoid robot, using teleoperation data and comparing IL methods tailored to the available platform. The experimental results indicate that our method is more aligned with the success rate obtained on the robot than baselines, offering a reproducible, systematic, and insightful means for comparing the performance of multimodal imitation learning approaches in complex HRI tasks. 

**Abstract (ZH)**: 评估和比较自主 humanoid 机器人的性能具有挑战性，因为成功率指标难以重现且无法捕捉人类-机器人交互与协作（HRIC）中机器人运动轨迹的复杂性。为应对这些挑战，我们提出了一种通用评估框架，通过关注轨迹性能来衡量模仿学习（IL）方法的质量。我们设计了神经元元评估器（NeME），这是一种用于分类机器人关节轨迹的动作的深度学习模型。NeME 作为元评估器，用于比较机器人控制策略的性能，使其能够在循环中无需人类参与即可进行策略评估。我们在使用遥控操作数据和针对可用平台定制的 IL 方法进行比较的 ergoCub 人形机器人上验证了该框架。实验结果表明，我们的方法与机器人上获得的成功率更为一致，提供了一种可重现、系统且具有洞察力的方法，用于比较复杂HRIC任务中多模态模仿学习方法的性能。 

---
# A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding 

**Title (ZH)**: 基于LLM驱动空间推理的神经表示框架在开放式词汇3D视觉接地中的应用 

**Authors**: Zhenyang Liu, Sixiao Zheng, Siyu Chen, Cairong Zhao, Longfei Liang, Xiangyang Xue, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06719)  

**Abstract**: Open-vocabulary 3D visual grounding aims to localize target objects based on free-form language queries, which is crucial for embodied AI applications such as autonomous navigation, robotics, and augmented reality. Learning 3D language fields through neural representations enables accurate understanding of 3D scenes from limited viewpoints and facilitates the localization of target objects in complex environments. However, existing language field methods struggle to accurately localize instances using spatial relations in language queries, such as ``the book on the chair.'' This limitation mainly arises from inadequate reasoning about spatial relations in both language queries and 3D scenes. In this work, we propose SpatialReasoner, a novel neural representation-based framework with large language model (LLM)-driven spatial reasoning that constructs a visual properties-enhanced hierarchical feature field for open-vocabulary 3D visual grounding. To enable spatial reasoning in language queries, SpatialReasoner fine-tunes an LLM to capture spatial relations and explicitly infer instructions for the target, anchor, and spatial relation. To enable spatial reasoning in 3D scenes, SpatialReasoner incorporates visual properties (opacity and color) to construct a hierarchical feature field. This field represents language and instance features using distilled CLIP features and masks extracted via the Segment Anything Model (SAM). The field is then queried using the inferred instructions in a hierarchical manner to localize the target 3D instance based on the spatial relation in the language query. Extensive experiments show that our framework can be seamlessly integrated into different neural representations, outperforming baseline models in 3D visual grounding while empowering their spatial reasoning capability. 

**Abstract (ZH)**: 基于开放词汇的3D视觉定位旨在根据自由形式的语言查询定位目标物体，这对于自主导航、机器人技术和增强现实等嵌入式AI应用至关重要。通过神经表示学习3D语言场能够从有限视角准确理解3D场景，并简化在复杂环境中定位目标物体的过程。然而，现有语言场方法难以使用语言查询中的空间关系（如“书在椅子上”）准确定位实例。这一局限主要源于对语言查询和3D场景中空间关系推理的不足。在本文中，我们提出SpatialReasoner，这是一种基于神经表示的新颖框架，采用大型语言模型（LLM）驱动的空间推理来构建增强视觉属性的分层次特征场以实现开放词汇的3D视觉定位。为了在语言查询中进行空间推理，SpatialReasoner对LLM进行微调以捕捉空间关系并明确推断目标、锚点和空间关系的指令。为了在3D场景中进行空间推理，SpatialReasoner将视觉属性（不透明度和颜色）纳入分层次特征场的构建中。该场使用从中断的CLIP特征和通过Segment Anything Model (SAM)提取的掩码表示语言和实例特征，并以分层次方式查询这些指令以基于语言查询中的空间关系定位目标3D实例。广泛实验表明，我们的框架可以无缝集成到不同的神经表示中，在3D视觉定位方面优于基线模型，同时增强了它们的空间推理能力。 

---
# VisioPath: Vision-Language Enhanced Model Predictive Control for Safe Autonomous Navigation in Mixed Traffic 

**Title (ZH)**: VisioPath: 视觉语言增强的模型预测控制方法以实现混合交通中的安全自主导航 

**Authors**: Shanting Wang, Panagiotis Typaldos, Chenjun Li, Andreas A. Malikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2507.06441)  

**Abstract**: In this paper, we introduce VisioPath, a novel framework combining vision-language models (VLMs) with model predictive control (MPC) to enable safe autonomous driving in dynamic traffic environments. The proposed approach leverages a bird's-eye view video processing pipeline and zero-shot VLM capabilities to obtain structured information about surrounding vehicles, including their positions, dimensions, and velocities. Using this rich perception output, we construct elliptical collision-avoidance potential fields around other traffic participants, which are seamlessly integrated into a finite-horizon optimal control problem for trajectory planning. The resulting trajectory optimization is solved via differential dynamic programming with an adaptive regularization scheme and is embedded in an event-triggered MPC loop. To ensure collision-free motion, a safety verification layer is incorporated in the framework that provides an assessment of potential unsafe trajectories. Extensive simulations in Simulation of Urban Mobility (SUMO) demonstrate that VisioPath outperforms conventional MPC baselines across multiple metrics. By combining modern AI-driven perception with the rigorous foundation of optimal control, VisioPath represents a significant step forward in safe trajectory planning for complex traffic systems. 

**Abstract (ZH)**: 基于视觉-语言模型与模型预测控制的VisioPath框架：动态交通环境中的安全自主驾驶 

---
# DeepRetro: Retrosynthetic Pathway Discovery using Iterative LLM Reasoning 

**Title (ZH)**: DeepRetro: 使用迭代LLM推理发现逆合成反应路径 

**Authors**: Shreyas Vinaya Sathyanarayana, Rahil Shah, Sharanabasava D. Hiremath, Rishikesh Panda, Rahul Jana, Riya Singh, Rida Irfan, Ashwin Murali, Bharath Ramsundar  

**Link**: [PDF](https://arxiv.org/pdf/2507.07060)  

**Abstract**: Retrosynthesis, the identification of precursor molecules for a target compound, is pivotal for synthesizing complex molecules, but faces challenges in discovering novel pathways beyond predefined templates. Recent large language model (LLM) approaches to retrosynthesis have shown promise but effectively harnessing LLM reasoning capabilities for effective multi-step planning remains an open question. To address this challenge, we introduce DeepRetro, an open-source, iterative, hybrid LLM-based retrosynthetic framework. Our approach integrates the strengths of conventional template-based/Monte Carlo tree search tools with the generative power of LLMs in a step-wise, feedback-driven loop. Initially, synthesis planning is attempted with a template-based engine. If this fails, the LLM subsequently proposes single-step retrosynthetic disconnections. Crucially, these suggestions undergo rigorous validity, stability, and hallucination checks before the resulting precursors are recursively fed back into the pipeline for further evaluation. This iterative refinement allows for dynamic pathway exploration and correction. We demonstrate the potential of this pipeline through benchmark evaluations and case studies, showcasing its ability to identify viable and potentially novel retrosynthetic routes. In particular, we develop an interactive graphical user interface that allows expert human chemists to provide human-in-the-loop feedback to the reasoning algorithm. This approach successfully generates novel pathways for complex natural product compounds, demonstrating the potential for iterative LLM reasoning to advance state-of-art in complex chemical syntheses. 

**Abstract (ZH)**: 基于大规模语言模型的迭代混合 retrosynthesis 框架：DeepRetro 

---
# What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models 

**Title (ZH)**: 基于归纳偏置探究基础模型发现的世界模型 

**Authors**: Keyon Vafa, Peter G. Chang, Ashesh Rambachan, Sendhil Mullainathan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06952)  

**Abstract**: Foundation models are premised on the idea that sequence prediction can uncover deeper domain understanding, much like how Kepler's predictions of planetary motion later led to the discovery of Newtonian mechanics. However, evaluating whether these models truly capture deeper structure remains a challenge. We develop a technique for evaluating foundation models that examines how they adapt to synthetic datasets generated from some postulated world model. Our technique measures whether the foundation model's inductive bias aligns with the world model, and so we refer to it as an inductive bias probe. Across multiple domains, we find that foundation models can excel at their training tasks yet fail to develop inductive biases towards the underlying world model when adapted to new tasks. We particularly find that foundation models trained on orbital trajectories consistently fail to apply Newtonian mechanics when adapted to new physics tasks. Further analysis reveals that these models behave as if they develop task-specific heuristics that fail to generalize. 

**Abstract (ZH)**: 基础模型的前提在于序列预测能够揭示更深层次的领域理解，类似于开普勒对行星运动的预测后来促成了牛顿力学的发现。然而，评估这些模型是否真正捕捉到更深层次的结构仍然是一个挑战。我们开发了一种技术，该技术通过检查基础模型对从某个假设世界模型生成的合成数据集的适应性来评估基础模型。该技术衡量基础模型的归纳偏见是否与世界模型一致，因此我们将其称为归纳偏见探针。在多个领域中，我们发现基础模型在训练任务上表现优异，但在适应新任务时，未能发展出与底层世界模型相一致的归纳偏见。特别地，我们发现训练于轨道轨迹数据的基础模型在适应新的物理任务时，总是不能应用牛顿力学。进一步分析表明，这些模型的行为似乎表明它们发展了特定于任务的经验法则，无法实现泛化。 

---
# VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation 

**Title (ZH)**: 视觉陷阱：通过视觉接地操纵实现GUI代理的隐蔽后门攻击 

**Authors**: Ziang Ye, Yang Zhang, Wentao Shi, Xiaoyu You, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2507.06899)  

**Abstract**: Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the visual grounding of GUI agent-mapping textual plans to GUI elements-can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent's behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose VisualTrap, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, e.g., being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents. 

**Abstract (ZH)**: 由大规模视觉-语言模型驱动的图形用户界面（GUI）代理的漏洞及其后门攻击风险研究 

---
# Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning 

**Title (ZH)**: 人工通用智能：使用强化学习掌握Generals.io游戏技巧 

**Authors**: Matej Straka, Martin Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2507.06825)  

**Abstract**: We introduce a real-time strategy game environment built on this http URL, a game that hosts thousands of active players each week across multiple game formats. Our environment is fully compatible with Gymnasium and PettingZoo, capable of running thousands of frames per second on commodity hardware. Our reference agent -- trained with supervised pre-training and self-play -- hits the top 0.003\% of the 1v1 human leaderboard after just 36 hours on a single H100 GPU. To accelerate learning, we incorporate potential-based reward shaping and memory features. Our contributions -- a modular RTS benchmark and a competitive, state-of-the-art baseline agent -- provide an accessible yet challenging platform for advancing multi-agent reinforcement learning research. 

**Abstract (ZH)**: 基于这个网址构建的实时战略游戏环境：数千名玩家每周在多种游戏格式中活跃参与。我们的环境与Gymnasium和PettingZoo完全兼容，在普通硬件上可以每秒运行数千帧。参考代理——通过监督预训练和自我博弈训练，在单个H100 GPU上运行36小时后达到1v1人类排行榜的前0.003%。为了加速学习，我们加入了基于潜力的奖励塑造和记忆特征。我们的贡献包括一个模块化的RTS基准和一个竞争力强、处于最新技术水平的基线多代理强化学习代理，为推进多代理强化学习研究提供了一个易于访问但具有挑战性的平台。 

---
# HeLo: Heterogeneous Multi-Modal Fusion with Label Correlation for Emotion Distribution Learning 

**Title (ZH)**: HeLo：具有标签相关性的异构多模态融合情感分布学习 

**Authors**: Chuhang Zheng, Chunwei Tian, Jie Wen, Daoqiang Zhang, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06821)  

**Abstract**: Multi-modal emotion recognition has garnered increasing attention as it plays a significant role in human-computer interaction (HCI) in recent years. Since different discrete emotions may exist at the same time, compared with single-class emotion recognition, emotion distribution learning (EDL) that identifies a mixture of basic emotions has gradually emerged as a trend. However, existing EDL methods face challenges in mining the heterogeneity among multiple modalities. Besides, rich semantic correlations across arbitrary basic emotions are not fully exploited. In this paper, we propose a multi-modal emotion distribution learning framework, named HeLo, aimed at fully exploring the heterogeneity and complementary information in multi-modal emotional data and label correlation within mixed basic emotions. Specifically, we first adopt cross-attention to effectively fuse the physiological data. Then, an optimal transport (OT)-based heterogeneity mining module is devised to mine the interaction and heterogeneity between the physiological and behavioral representations. To facilitate label correlation learning, we introduce a learnable label embedding optimized by correlation matrix alignment. Finally, the learnable label embeddings and label correlation matrices are integrated with the multi-modal representations through a novel label correlation-driven cross-attention mechanism for accurate emotion distribution learning. Experimental results on two publicly available datasets demonstrate the superiority of our proposed method in emotion distribution learning. 

**Abstract (ZH)**: 多模态情感分布学习在近年的人机交互（HCI）中引起了广泛关注。现有的情感分布学习方法在挖掘多模态之间的异质性方面面临挑战，同时未能充分利用基本情感之间的丰富语义关联。为了解决这些问题，我们提出了一种名为HeLo的多模态情感分布学习框架，旨在充分探索多模态情感数据及其标签在混合基本情感中的异质性和互补信息。具体地，我们首先采用跨注意力机制有效融合生理数据，然后设计了一种基于最优传输（OT）的异质性挖掘模块，以挖掘生理和行为表示之间的交互和异质性。为了促进标签关联学习，我们引入了通过相关矩阵对齐优化的可学习标签嵌入。最后，通过一种新颖的情感分布驱动的跨注意力机制，将可学习标签嵌入和标签相关矩阵与多模态表示结合起来，实现精确的情感分布学习。在两个公开数据集上的实验结果表明，我们提出的方法在情感分布学习方面具有优越性。 

---
# Goal-Oriented Skill Abstraction for Offline Multi-Task Reinforcement Learning 

**Title (ZH)**: 面向目标的技能抽象在offline多任务 reinforcement learning中的应用 

**Authors**: Jinmin He, Kai Li, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06628)  

**Abstract**: Offline multi-task reinforcement learning aims to learn a unified policy capable of solving multiple tasks using only pre-collected task-mixed datasets, without requiring any online interaction with the environment. However, it faces significant challenges in effectively sharing knowledge across tasks. Inspired by the efficient knowledge abstraction observed in human learning, we propose Goal-Oriented Skill Abstraction (GO-Skill), a novel approach designed to extract and utilize reusable skills to enhance knowledge transfer and task performance. Our approach uncovers reusable skills through a goal-oriented skill extraction process and leverages vector quantization to construct a discrete skill library. To mitigate class imbalances between broadly applicable and task-specific skills, we introduce a skill enhancement phase to refine the extracted skills. Furthermore, we integrate these skills using hierarchical policy learning, enabling the construction of a high-level policy that dynamically orchestrates discrete skills to accomplish specific tasks. Extensive experiments on diverse robotic manipulation tasks within the MetaWorld benchmark demonstrate the effectiveness and versatility of GO-Skill. 

**Abstract (ZH)**: 面向 Offline 多任务强化学习的目标导向技能抽象 

---
# Efficient Multi-Task Reinforcement Learning with Cross-Task Policy Guidance 

**Title (ZH)**: 跨任务策略指导的高效多任务强化学习 

**Authors**: Jinmin He, Kai Li, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06615)  

**Abstract**: Multi-task reinforcement learning endeavors to efficiently leverage shared information across various tasks, facilitating the simultaneous learning of multiple tasks. Existing approaches primarily focus on parameter sharing with carefully designed network structures or tailored optimization procedures. However, they overlook a direct and complementary way to exploit cross-task similarities: the control policies of tasks already proficient in some skills can provide explicit guidance for unmastered tasks to accelerate skills acquisition. To this end, we present a novel framework called Cross-Task Policy Guidance (CTPG), which trains a guide policy for each task to select the behavior policy interacting with the environment from all tasks' control policies, generating better training trajectories. In addition, we propose two gating mechanisms to improve the learning efficiency of CTPG: one gate filters out control policies that are not beneficial for guidance, while the other gate blocks tasks that do not necessitate guidance. CTPG is a general framework adaptable to existing parameter sharing approaches. Empirical evaluations demonstrate that incorporating CTPG with these approaches significantly enhances performance in manipulation and locomotion benchmarks. 

**Abstract (ZH)**: 跨任务策略指导的多任务 reinforcement 学习 

---
# Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning 

**Title (ZH)**: 视频RTS：重新思考高效增强视频推理中的强化学习和测试时扩展问题 

**Authors**: Ziyang Wang, Jaehong Yoon, Shoubin Yu, Md Mohaiminul Islam, Gedas Bertasius, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2507.06485)  

**Abstract**: Despite advances in reinforcement learning (RL)-based video reasoning with large language models (LLMs), data collection and finetuning remain significant challenges. These methods often rely on large-scale supervised fine-tuning (SFT) with extensive video data and long Chain-of-Thought (CoT) annotations, making them costly and hard to scale. To address this, we present Video-RTS, a new approach to improve video reasoning capability with drastically improved data efficiency by combining data-efficient RL with a video-adaptive test-time scaling (TTS) strategy. Based on observations about the data scaling of RL samples, we skip the resource-intensive SFT step and employ efficient pure-RL training with output-based rewards, requiring no additional annotations or extensive fine-tuning. Furthermore, to utilize computational resources more efficiently, we introduce a sparse-to-dense video TTS strategy that improves inference by iteratively adding frames based on output consistency. We validate our approach on multiple video reasoning benchmarks, showing that Video-RTS surpasses existing video reasoning models by an average of 2.4% in accuracy using only 3.6% training samples. For example, Video-RTS achieves a 4.2% improvement on Video-Holmes, a recent and challenging video reasoning benchmark, and a 2.6% improvement on MMVU. Notably, our pure RL training and adaptive video TTS offer complementary strengths, enabling Video-RTS's strong reasoning performance. 

**Abstract (ZH)**: 基于大规模语言模型的强化学习在视频推理中的数据收集和微调仍然是显著挑战。尽管在基于强化学习(RL)的视频推理方法中取得了进展，但仍面临数据收集和微调的巨大挑战。这些方法通常依赖于大规模监督微调(SFT)和广泛的视频数据以及长的推理链(Chain-of-Thought, CoT)注释，使其成本高昂且不易扩展。为了解决这一问题，我们提出了Video-RTS，这是一种通过结合数据高效RL和视频自适应测试时缩放(TTS)策略来大幅提高数据效率的新方法。基于对RL样本数据放大的观察，我们跳过了资源密集型的SFT步骤，采用基于输出的奖励的高效纯RL训练，无需额外注释或大规模微调。此外，为了更有效地利用计算资源，我们引入了一种稀疏到密集的视频TTS策略，通过迭代添加帧并基于输出一致性逐步改善推理。我们在多个视频推理基准上验证了该方法，结果显示，使用仅3.6%的训练样本，Video-RTS在准确率上平均超过了现有视频推理模型2.4%。例如，Video-RTS在最近提出的具有挑战性的视频推理基准Video-Holmes上实现了4.2%的提升，在MMVU上实现了2.6%的提升。值得注意的是，我们的纯RL训练和自适应视频TTS互补性强，使Video-RTS在推理性能上表现出色。 

---
# An AI-Driven Thermal-Fluid Testbed for Advanced Small Modular Reactors: Integration of Digital Twin and Large Language Models 

**Title (ZH)**: 面向先进小模块反应堆的AI驱动热流测试床：数字孪生与大型语言模型的集成 

**Authors**: Doyeong Lim, Yang Liu, Zavier Ndum Ndum, Christian Young, Yassin Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06399)  

**Abstract**: This paper presents a multipurpose artificial intelligence (AI)-driven thermal-fluid testbed designed to advance Small Modular Reactor technologies by seamlessly integrating physical experimentation with advanced computational intelligence. The platform uniquely combines a versatile three-loop thermal-fluid facility with a high-fidelity digital twin and sophisticated AI frameworks for real-time prediction, control, and operational assistance. Methodologically, the testbed's digital twin, built upon the System Analysis Module code, is coupled with a Gated Recurrent Unit (GRU) neural network. This machine learning model, trained on experimental data, enables faster-than-real-time simulation, providing predictive insights into the system's dynamic behavior. The practical application of this AI integration is showcased through case studies. An AI-driven control framework where the GRU model accurately forecasts future system states and the corresponding control actions required to meet operational demands. Furthermore, an intelligent assistant, powered by a large language model, translates complex sensor data and simulation outputs into natural language, offering operators actionable analysis and safety recommendations. Comprehensive validation against experimental transients confirms the platform's high fidelity, with the GRU model achieving a temperature prediction root mean square error of 1.42 K. This work establishes an integrated research environment at the intersection of AI and thermal-fluid science, showcasing how AI-driven methodologies in modeling, control, and operator support can accelerate the innovation and deployment of next-generation nuclear systems. 

**Abstract (ZH)**: 一种基于人工智能的多功能热流试验台：先进计算智能与物理实验的无缝集成以推动小模块反应堆技术发展 

---
# VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting 

**Title (ZH)**: VOTE：具有轨迹集成投票的视觉-语言-动作优化 

**Authors**: Juyi Lin, Amir Taherin, Arash Akbari, Arman Akbari, Lei Lu, Guangyu Chen, Taskin Padir, Xiaomeng Yang, Weiwei Chen, Yiqian Li, Xue Lin, David Kaeli, Pu Zhao, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05116)  

**Abstract**: Recent large-scale Vision Language Action (VLA) models have shown superior performance in robotic manipulation tasks guided by natural language. However, their generalization remains limited when applied to novel objects or unfamiliar environments that lie outside the training distribution. To address this, many existing approaches integrate additional components such as depth estimation, segmentation, or even diffusion to improve generalization, at the cost of adding significant computation overhead, resulting in low efficiency. This motivates the exploration of efficient action prediction methods, which are independent of additional high-level visual representations or diffusion techniques. In this work, we propose VOTE, an efficient and general framework for the optimization and acceleration of VLA models. In details, we propose a novel tokenizer-free fine-tuning approach for parallel accurate action prediction, which reduces computational overhead and accelerates inference speed. Additionally, we adopt an ensemble voting strategy for the action sampling, which significantly improves model performance and enhances generalization. Experimental results show that our method achieves state-of-the-art performance with 35$\times$ faster inference and 145 Hz throughput. All the details and codes will be open-sourced. 

**Abstract (ZH)**: 近期大规模视觉语言动作（VLA）模型在由自然语言指导的机器人操作任务中展现了 superior 性能。然而，当应用于训练分布之外的新型物体或不熟悉环境时，其泛化能力仍然有限。为解决这一问题，许多现有方法通过集成额外组件，如深度估计、分割或乃至扩散技术来提升泛化能力，但这些方法会显著增加计算开销，导致效率低下。这促使我们探索独立于额外高层视觉表示或扩散技术的高效动作预测方法。在本工作中，我们提出 VOTE，一种优化和加速 VLA 模型的高效且通用框架。具体而言，我们提出了一种新的无标记符的并行准确动作预测微调方法，减少了计算开销并加速了推理速度。同时，我们采用了一种集成投票策略进行动作采样，显著提高了模型性能并增强了泛化能力。实验结果表明，我们的方法以 35 倍更快的推理速度和 145 Hz 的吞吐量达到了当前最优性能。所有细节和代码将开源。 

---
