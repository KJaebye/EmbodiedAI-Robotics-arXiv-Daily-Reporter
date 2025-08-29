# Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation 

**Title (ZH)**: 随需学习：通过可微模拟实现快速策略适应 

**Authors**: Jiahe Pan, Jiaxu Xing, Rudolf Reiter, Yifan Zhai, Elie Aljalbout, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2508.21065)  

**Abstract**: Learning control policies in simulation enables rapid, safe, and cost-effective development of advanced robotic capabilities. However, transferring these policies to the real world remains difficult due to the sim-to-real gap, where unmodeled dynamics and environmental disturbances can degrade policy performance. Existing approaches, such as domain randomization and Real2Sim2Real pipelines, can improve policy robustness, but either struggle under out-of-distribution conditions or require costly offline retraining. In this work, we approach these problems from a different perspective. Instead of relying on diverse training conditions before deployment, we focus on rapidly adapting the learned policy in the real world in an online fashion. To achieve this, we propose a novel online adaptive learning framework that unifies residual dynamics learning with real-time policy adaptation inside a differentiable simulation. Starting from a simple dynamics model, our framework refines the model continuously with real-world data to capture unmodeled effects and disturbances such as payload changes and wind. The refined dynamics model is embedded in a differentiable simulation framework, enabling gradient backpropagation through the dynamics and thus rapid, sample-efficient policy updates beyond the reach of classical RL methods like PPO. All components of our system are designed for rapid adaptation, enabling the policy to adjust to unseen disturbances within 5 seconds of training. We validate the approach on agile quadrotor control under various disturbances in both simulation and the real world. Our framework reduces hovering error by up to 81% compared to L1-MPC and 55% compared to DATT, while also demonstrating robustness in vision-based control without explicit state estimation. 

**Abstract (ZH)**: 在仿真中学习控制策略使得快速、安全且经济有效地开发先进机器人能力成为可能。然而，由于仿真实践与真实世界之间的差距，即未建模动态和环境扰动可能导致策略性能下降，将这些策略转移到现实世界仍然存在困难。现有的方法，如域随机化和Real2Sim2Real流水线，可以提高策略的鲁棒性，但在处理异常分布情况时可能表现不佳，或者需要昂贵的离线重新训练。在本工作中，我们从不同的角度解决这些问题。我们不依赖于部署前的多样化训练条件，而是专注于以在线方式快速适应在真实世界中学习到的策略。为此，我们提出了一种新颖的在线自适应学习框架，该框架将残差动力学学习与实时策略适应统一在可微分仿真中。从一个简单的动力学模型开始，我们的框架不断用现实世界的数据对模型进行细化，以捕捉未建模效应和干扰，如载荷变化和风。细化的动力学模型嵌入到一个可微分仿真框架中，从而可以在包含动力学梯度回传的情况下，实现超越传统RL方法（如PPO）的快速、样本高效策略更新。我们系统的各个组件都旨在实现快速适应，使策略在5秒的训练时间内即可调整以应对未见的干扰。我们在各种干扰下于仿真和真实世界中验证了该方法，我们的框架将悬停误差降低了81%（相对于L1-MPC）和55%（相对于DATT），同时在基于视觉的控制中也展示了无显式状态估计的鲁棒性。 

---
# Prompt-to-Product: Generative Assembly via Bimanual Manipulation 

**Title (ZH)**: Prompt-to-Product: 生成组装通过双臂操作 

**Authors**: Ruixuan Liu, Philip Huang, Ava Pun, Kangle Deng, Shobhit Aggarwal, Kevin Tang, Michelle Liu, Deva Ramanan, Jun-Yan Zhu, Jiaoyang Li, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21063)  

**Abstract**: Creating assembly products demands significant manual effort and expert knowledge in 1) designing the assembly and 2) constructing the product. This paper introduces Prompt-to-Product, an automated pipeline that generates real-world assembly products from natural language prompts. Specifically, we leverage LEGO bricks as the assembly platform and automate the process of creating brick assembly structures. Given the user design requirements, Prompt-to-Product generates physically buildable brick designs, and then leverages a bimanual robotic system to construct the real assembly products, bringing user imaginations into the real world. We conduct a comprehensive user study, and the results demonstrate that Prompt-to-Product significantly lowers the barrier and reduces manual effort in creating assembly products from imaginative ideas. 

**Abstract (ZH)**: 基于自然语言提示的自动装配产品生成pipeline降低了从想象概念创建装配产品的门槛并减少了手动 effort。 

---
# HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning 

**Title (ZH)**: HITTER：基于层次规划与学习的类人乒乓球机器人 

**Authors**: Zhi Su, Bike Zhang, Nima Rahmanian, Yuman Gao, Qiayuan Liao, Caitlin Regan, Koushil Sreenath, S. Shankar Sastry  

**Link**: [PDF](https://arxiv.org/pdf/2508.21043)  

**Abstract**: Humanoid robots have recently achieved impressive progress in locomotion and whole-body control, yet they remain constrained in tasks that demand rapid interaction with dynamic environments through manipulation. Table tennis exemplifies such a challenge: with ball speeds exceeding 5 m/s, players must perceive, predict, and act within sub-second reaction times, requiring both agility and precision. To address this, we present a hierarchical framework for humanoid table tennis that integrates a model-based planner for ball trajectory prediction and racket target planning with a reinforcement learning-based whole-body controller. The planner determines striking position, velocity and timing, while the controller generates coordinated arm and leg motions that mimic human strikes and maintain stability and agility across consecutive rallies. Moreover, to encourage natural movements, human motion references are incorporated during training. We validate our system on a general-purpose humanoid robot, achieving up to 106 consecutive shots with a human opponent and sustained exchanges against another humanoid. These results demonstrate real-world humanoid table tennis with sub-second reactive control, marking a step toward agile and interactive humanoid behaviors. 

**Abstract (ZH)**: 基于模型的轨迹预测和强化学习的全身控制器结合的类人乒乓球框架：实现实秒级反应控制的类人乒乓球行为 

---
# Rapid Mismatch Estimation via Neural Network Informed Variational Inference 

**Title (ZH)**: 基于神经网络引导的变分推断的快速不匹配估计 

**Authors**: Mateusz Jaszczuk, Nadia Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2508.21007)  

**Abstract**: With robots increasingly operating in human-centric environments, ensuring soft and safe physical interactions, whether with humans, surroundings, or other machines, is essential. While compliant hardware can facilitate such interactions, this work focuses on impedance controllers that allow torque-controlled robots to safely and passively respond to contact while accurately executing tasks. From inverse dynamics to quadratic programming-based controllers, the effectiveness of these methods relies on accurate dynamics models of the robot and the object it manipulates. Any model mismatch results in task failures and unsafe behaviors. Thus, we introduce Rapid Mismatch Estimation (RME), an adaptive, controller-agnostic, probabilistic framework that estimates end-effector dynamics mismatches online, without relying on external force-torque sensors. From the robot's proprioceptive feedback, a Neural Network Model Mismatch Estimator generates a prior for a Variational Inference solver, which rapidly converges to the unknown parameters while quantifying uncertainty. With a real 7-DoF manipulator driven by a state-of-the-art passive impedance controller, RME adapts to sudden changes in mass and center of mass at the end-effector in $\sim400$ ms, in static and dynamic settings. We demonstrate RME in a collaborative scenario where a human attaches an unknown basket to the robot's end-effector and dynamically adds/removes heavy items, showcasing fast and safe adaptation to changing dynamics during physical interaction without any external sensory system. 

**Abstract (ZH)**: 随着机器人越来越多地在以人类为中心的环境中操作，确保与其进行软而安全的物理交互（无论是与人类、环境还是其他机器的交互）至关重要。虽然顺应性硬件可以促进这种交互，但本文的重点在于允许扭矩控制机器人安全且被动地响应接触的同时准确执行任务的阻抗控制器。从逆动力学到基于二次规划的控制器，这些方法的有效性依赖于对机器人及其操作对象的精确动力学模型。任何模型不匹配都会导致任务失败和不安全的行为。因此，我们提出了快速不匹配估计（RME）方法，这是一种自适应的、控制器无关的概率框架，能够在不依赖外部力-扭矩传感器的情况下在线估计末端执行器动力学不匹配。从机器人的本体感受反馈出发，神经网络模型不匹配估计器生成变分推断求解器的先验信息，该解算器能够快速收敛到未知参数的同时量化不确定性。我们使用一个由先进被动阻抗控制器驱动的真实7自由度 manipulator，RME 在大约400毫秒内适应末端执行器处突然的质量和质心变化，无论是静态还是动态情况下均能实现这一点。我们在一个协作场景中展示了 RME，其中人类将一个未知篮子附加到机器人的末端执行器上，并动态添加/移除重物，证明了在物理交互过程中能够快速且安全地适应不断变化的动力学，无需任何外部感知系统。 

---
# UltraTac: Integrated Ultrasound-Augmented Visuotactile Sensor for Enhanced Robotic Perception 

**Title (ZH)**: UltraTac：集成超声增强的视触觉传感器以提高机器人感知能力 

**Authors**: Junhao Gong, Kit-Wa Sou, Shoujie Li, Changqing Guo, Yan Huang, Chuqiao Lyu, Ziwu Song, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.20982)  

**Abstract**: Visuotactile sensors provide high-resolution tactile information but are incapable of perceiving the material features of objects. We present UltraTac, an integrated sensor that combines visuotactile imaging with ultrasound sensing through a coaxial optoacoustic architecture. The design shares structural components and achieves consistent sensing regions for both modalities. Additionally, we incorporate acoustic matching into the traditional visuotactile sensor structure, enabling integration of the ultrasound sensing modality without compromising visuotactile performance. Through tactile feedback, we dynamically adjust the operating state of the ultrasound module to achieve flexible functional coordination. Systematic experiments demonstrate three key capabilities: proximity sensing in the 3-8 cm range ($R^2=0.90$), material classification (average accuracy: 99.20%), and texture-material dual-mode object recognition achieving 92.11% accuracy on a 15-class task. Finally, we integrate the sensor into a robotic manipulation system to concurrently detect container surface patterns and internal content, which verifies its potential for advanced human-machine interaction and precise robotic manipulation. 

**Abstract (ZH)**: 基于共轴光学声学架构的综合触觉-视觉传感器UltraTac：接近感知、材料分类与纹理-材料双重模式物体识别 

---
# ActLoc: Learning to Localize on the Move via Active Viewpoint Selection 

**Title (ZH)**: ActLoc: 在移动过程中通过主动视角选择进行 localization 

**Authors**: Jiajie Li, Boyang Sun, Luca Di Giammarino, Hermann Blum, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2508.20981)  

**Abstract**: Reliable localization is critical for robot navigation, yet most existing systems implicitly assume that all viewing directions at a location are equally informative. In practice, localization becomes unreliable when the robot observes unmapped, ambiguous, or uninformative regions. To address this, we present ActLoc, an active viewpoint-aware planning framework for enhancing localization accuracy for general robot navigation tasks. At its core, ActLoc employs a largescale trained attention-based model for viewpoint selection. The model encodes a metric map and the camera poses used during map construction, and predicts localization accuracy across yaw and pitch directions at arbitrary 3D locations. These per-point accuracy distributions are incorporated into a path planner, enabling the robot to actively select camera orientations that maximize localization robustness while respecting task and motion constraints. ActLoc achieves stateof-the-art results on single-viewpoint selection and generalizes effectively to fulltrajectory planning. Its modular design makes it readily applicable to diverse robot navigation and inspection tasks. 

**Abstract (ZH)**: 可靠的定位对于机器人导航至关重要，然而现有的大多数系统隐含地假设位置的所有视向角度都具有相同的信息量。实际上，当机器人观察未映射、模棱两可或无信息的区域时，定位变得不可靠。为了解决这一问题，我们提出了ActLoc，一种主动视角感知规划框架，旨在提高通用机器人导航任务中的定位准确性。ActLoc的核心在于使用大型训练的基于注意力的模型进行视角选择。该模型编码了用于地图构建的度量地图和相机姿态，并预测在任意3D位置的偏航和俯仰方向上的定位准确性。这些点级别的准确性分布被整合到路径规划器中，从而使机器人能够主动选择最大化定位鲁棒性的相机姿态，同时遵守任务和运动约束。ActLoc在单视角选择任务上达到了最先进的性能，并且在全程路径规划中表现出有效的泛化能力。其模块化设计使其易于应用于各种机器人导航和检测任务。 

---
# Scaling Fabric-Based Piezoresistive Sensor Arrays for Whole-Body Tactile Sensing 

**Title (ZH)**: 基于织物的压阻式传感器阵列放大技术及其在全身触觉感知中的应用 

**Authors**: Curtis C. Johnson, Daniel Webb, David Hill, Marc D. Killpack  

**Link**: [PDF](https://arxiv.org/pdf/2508.20959)  

**Abstract**: Scaling tactile sensing for robust whole-body manipulation is a significant challenge, often limited by wiring complexity, data throughput, and system reliability. This paper presents a complete architecture designed to overcome these barriers. Our approach pairs open-source, fabric-based sensors with custom readout electronics that reduce signal crosstalk to less than 3.3% through hardware-based mitigation. Critically, we introduce a novel, daisy-chained SPI bus topology that avoids the practical limitations of common wireless protocols and the prohibitive wiring complexity of USB hub-based systems. This architecture streams synchronized data from over 8,000 taxels across 1 square meter of sensing area at update rates exceeding 50 FPS, confirming its suitability for real-time control. We validate the system's efficacy in a whole-body grasping task where, without feedback, the robot's open-loop trajectory results in an uncontrolled application of force that slowly crushes a deformable cardboard box. With real-time tactile feedback, the robot transforms this motion into a gentle, stable grasp, successfully manipulating the object without causing structural damage. This work provides a robust and well-characterized platform to enable future research in advanced whole-body control and physical human-robot interaction. 

**Abstract (ZH)**: 扩展触觉传感以实现稳健的全身操作是一个重大挑战，常受到线缆复杂性、数据吞吐量和系统可靠性的限制。本文提出了一种完整的架构以克服这些障碍。我们的方法将开源的织物基传感器与定制的读出电子设备配对，通过硬件基础的缓解措施将信号串扰降至少于3.3%。关键的是，我们引入了一种新型的级联SPI总线拓扑结构，避免了常见无线协议的实用性限制，并克服了基于USB集线器系统的繁琐线缆复杂性。该架构在超过一平方米的传感区域内，以超过50 FPS的更新率同步传输来自超过8,000个触点的数据，证实其适用于实时控制。我们在一项全身抓取任务中验证了该系统的有效性：在没有反馈的情况下，机器人的开环轨迹会导致对可变形纸板箱的无控施力，逐渐将其压扁；而在实时触觉反馈下，机器人将此运动转化为一种柔和、稳定的抓取，成功操纵物体而不会造成结构性损坏。该工作提供了一个稳健且性能良好的平台，以促进先进全身控制和物理人机交互的未来研究。 

---
# PLUME: Procedural Layer Underground Modeling Engine 

**Title (ZH)**: PLUME: 基于过程的地下分层建模引擎 

**Authors**: Gabriel Manuel Garcia, Antoine Richard, Miguel Olivares-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2508.20926)  

**Abstract**: As space exploration advances, underground environments are becoming increasingly attractive due to their potential to provide shelter, easier access to resources, and enhanced scientific opportunities. Although such environments exist on Earth, they are often not easily accessible and do not accurately represent the diversity of underground environments found throughout the solar system. This paper presents PLUME, a procedural generation framework aimed at easily creating 3D underground environments. Its flexible structure allows for the continuous enhancement of various underground features, aligning with our expanding understanding of the solar system. The environments generated using PLUME can be used for AI training, evaluating robotics algorithms, 3D rendering, and facilitating rapid iteration on developed exploration algorithms. In this paper, it is demonstrated that PLUME has been used along with a robotic simulator. PLUME is open source and has been released on Github. this https URL 

**Abstract (ZH)**: 随着太空探索的进展，地下环境因可能提供的庇护所、更便捷的资源获取途径以及增强的科研机会而变得越来越有吸引力。尽管在地球上存在这样的环境，但它们往往难以访问且不能准确代表太阳系中发现的地下环境的多样性。本文介绍了一种名为PLUME的程序生成框架，旨在轻松创建3D地下环境。其灵活结构允许不断增强各种地下特征，以适应我们对太阳系理解的扩展。使用PLUME生成的环境可用于AI训练、机器人算法评估、3D渲染以及促进开发中的探索算法的快速迭代。本文展示了PLUME已与机器人模拟器结合使用。PLUME是开源的，并已在Github上发布。更多详情请参见：这个链接。 

---
# Language-Enhanced Mobile Manipulation for Efficient Object Search in Indoor Environments 

**Title (ZH)**: 基于语言增强的移动 manipulation 技术在室内环境中的高效物体搜寻 

**Authors**: Liding Zhang, Zeqi Li, Kuanqi Cai, Qian Huang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.20899)  

**Abstract**: Enabling robots to efficiently search for and identify objects in complex, unstructured environments is critical for diverse applications ranging from household assistance to industrial automation. However, traditional scene representations typically capture only static semantics and lack interpretable contextual reasoning, limiting their ability to guide object search in completely unfamiliar settings. To address this challenge, we propose a language-enhanced hierarchical navigation framework that tightly integrates semantic perception and spatial reasoning. Our method, Goal-Oriented Dynamically Heuristic-Guided Hierarchical Search (GODHS), leverages large language models (LLMs) to infer scene semantics and guide the search process through a multi-level decision hierarchy. Reliability in reasoning is achieved through the use of structured prompts and logical constraints applied at each stage of the hierarchy. For the specific challenges of mobile manipulation, we introduce a heuristic-based motion planner that combines polar angle sorting with distance prioritization to efficiently generate exploration paths. Comprehensive evaluations in Isaac Sim demonstrate the feasibility of our framework, showing that GODHS can locate target objects with higher search efficiency compared to conventional, non-semantic search strategies. Website and Video are available at: this https URL 

**Abstract (ZH)**: 提高机器人在复杂无结构环境中的物体搜索和识别效率对于从家庭辅助到工业自动化等多种应用至关重要。然而，传统的场景表示通常只能捕捉静态语义信息，并缺乏可解释的上下文推理能力，限制了其在完全陌生环境中的物体搜索引导能力。为应对这一挑战，我们提出了一种语言增强的分层导航框架，该框架紧密整合了语义感知和空间推理。我们的方法，目标导向动态启发式引导分层搜索（GODHS），利用大规模语言模型（LLMs）推断场景语义并通过多级决策层次引导搜索过程。通过在每一级层次中应用结构化提示和逻辑约束来实现推理的可靠性。针对移动操作的具体挑战，我们引入了一种基于启发式的运动规划器，结合极角排序和距离优先级，以高效生成探索路径。在Isaac Sim中的综合评估表明，我们的框架可行，GODHS相较于传统的非语义搜索策略，可以在更高的搜索效率下定位目标物体。更多信息请访问：this https URL 

---
# CoCoL: A Communication Efficient Decentralized Collaborative Method for Multi-Robot Systems 

**Title (ZH)**: CoCoL: 多机器人系统中高效的去中心化协作方法 

**Authors**: Jiaxi Huang, Yan Huang, Yixian Zhao, Wenchao Meng, Jinming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20898)  

**Abstract**: Collaborative learning enhances the performance and adaptability of multi-robot systems in complex tasks but faces significant challenges due to high communication overhead and data heterogeneity inherent in multi-robot tasks. To this end, we propose CoCoL, a Communication efficient decentralized Collaborative Learning method tailored for multi-robot systems with heterogeneous local datasets. Leveraging a mirror descent framework, CoCoL achieves remarkable communication efficiency with approximate Newton-type updates by capturing the similarity between objective functions of robots, and reduces computational costs through inexact sub-problem solutions. Furthermore, the integration of a gradient tracking scheme ensures its robustness against data heterogeneity. Experimental results on three representative multi robot collaborative learning tasks show the superiority of the proposed CoCoL in significantly reducing both the number of communication rounds and total bandwidth consumption while maintaining state-of-the-art accuracy. These benefits are particularly evident in challenging scenarios involving non-IID (non-independent and identically distributed) data distribution, streaming data, and time-varying network topologies. 

**Abstract (ZH)**: 协作学习提升了多机器人系统在复杂任务中的性能和适应性，但由于多机器人任务固有的高通信开销和数据异质性，面临显著挑战。为此，我们提出了一种通信高效的分布式协作学习方法CoCoL，该方法针对具有异质本地数据集的多机器人系统进行优化。利用镜像下降框架，CoCoL通过捕捉机器人目标函数之间的相似性实现了近似的牛顿型更新，从而实现高效的通信，并通过近似子问题求解降低计算成本。此外，梯度跟踪方案的集成确保了其在数据异质性下的鲁棒性。实验结果表明，CoCoL在保持先进准确度的同时，显著减少了通信轮次和总带宽消耗，特别是在涉及非IID数据分布、流式数据和时间varying网络拓扑的挑战性场景中表现尤为突出。 

---
# Deep Fuzzy Optimization for Batch-Size and Nearest Neighbors in Optimal Robot Motion Planning 

**Title (ZH)**: 基于模糊优化的批量大小和最近邻在最优机器人运动规划中的应用 

**Authors**: Liding Zhang, Qiyang Zong, Yu Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.20884)  

**Abstract**: Efficient motion planning algorithms are essential in robotics. Optimizing essential parameters, such as batch size and nearest neighbor selection in sampling-based methods, can enhance performance in the planning process. However, existing approaches often lack environmental adaptability. Inspired by the method of the deep fuzzy neural networks, this work introduces Learning-based Informed Trees (LIT*), a sampling-based deep fuzzy learning-based planner that dynamically adjusts batch size and nearest neighbor parameters to obstacle distributions in the configuration spaces. By encoding both global and local ratios via valid and invalid states, LIT* differentiates between obstacle-sparse and obstacle-dense regions, leading to lower-cost paths and reduced computation time. Experimental results in high-dimensional spaces demonstrate that LIT* achieves faster convergence and improved solution quality. It outperforms state-of-the-art single-query, sampling-based planners in environments ranging from R^8 to R^14 and is successfully validated on a dual-arm robot manipulation task. A video showcasing our experimental results is available at: this https URL 

**Abstract (ZH)**: 基于学习的知情树（LIT*）：一种用于配置空间中动态调整batch大小和最近邻参数的采样基于深度模糊学习规划算法 

---
# Genetic Informed Trees (GIT*): Path Planning via Reinforced Genetic Programming Heuristics 

**Title (ZH)**: 遗传信息树（GIT*）：基于增强遗传编程启发式的路径规划 

**Authors**: Liding Zhang, Kuanqi Cai, Zhenshan Bing, Chaoqun Wang, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.20871)  

**Abstract**: Optimal path planning involves finding a feasible state sequence between a start and a goal that optimizes an objective. This process relies on heuristic functions to guide the search direction. While a robust function can improve search efficiency and solution quality, current methods often overlook available environmental data and simplify the function structure due to the complexity of information relationships. This study introduces Genetic Informed Trees (GIT*), which improves upon Effort Informed Trees (EIT*) by integrating a wider array of environmental data, such as repulsive forces from obstacles and the dynamic importance of vertices, to refine heuristic functions for better guidance. Furthermore, we integrated reinforced genetic programming (RGP), which combines genetic programming with reward system feedback to mutate genotype-generative heuristic functions for GIT*. RGP leverages a multitude of data types, thereby improving computational efficiency and solution quality within a set timeframe. Comparative analyses demonstrate that GIT* surpasses existing single-query, sampling-based planners in problems ranging from R^4 to R^16 and was tested on a real-world mobile manipulation task. A video showcasing our experimental results is available at this https URL 

**Abstract (ZH)**: 基于遗传信息树的最优路径规划方法：集成广泛环境数据的探索优化 

---
# Learning Primitive Embodied World Models: Towards Scalable Robotic Learning 

**Title (ZH)**: 基于基础体态世界模型的学习：通往可扩展机器人学习的道路 

**Authors**: Qiao Sun, Liujia Yang, Wei Tang, Wei Huang, Kaixin Xu, Yongchao Chen, Mingyu Liu, Jiange Yang, Haoyi Zhu, Yating Wang, Tong He, Yilun Chen, Xili Dai, Nanyang Ye, Qinying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20840)  

**Abstract**: While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a "GPT moment" in the embodied domain. There is a naive observation: the diversity of embodied data far exceeds the relatively small space of possible primitive motions. Based on this insight, we propose a novel paradigm for world modeling--Primitive Embodied World Models (PEWM). By restricting video generation to fixed short horizons, our approach 1) enables fine-grained alignment between linguistic concepts and visual representations of robotic actions, 2) reduces learning complexity, 3) improves data efficiency in embodied data collection, and 4) decreases inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence. 

**Abstract (ZH)**: 基于视频生成的 embodied 世界模型虽然逐渐受到关注，但其对大规模 embodied 交互数据的依赖仍然是一个关键瓶颈。由于 embodied 数据的稀缺性、收集难度和高维度性，根本限制了语言与动作之间的对齐粒度，并加剧了长时序视频生成的挑战——阻碍生成模型在 embodied 领域实现类似“GPT时刻”的突破。基于一个简单的观察：embodied 数据的多样性远超过可能的基本动作空间。基于此见解，我们提出了一种新的世界建模范式——基本要素 embodied 世界模型（PEWM）。通过限制视频生成在固定短时序内，我们的方法 1) 促进语言概念与机器人动作的视觉表示之间的精细对齐，2) 减少学习复杂性，3) 提高 embodied 数据收集的数据效率，4) 减少推理延迟。通过配备模块化视觉-语言模型（VLM）规划器和起始目标热图引导机制（SGG），PEWM 进一步支持灵活的闭环控制，并在长期复杂的任务中实现基本级别策略的组合泛化。我们的框架利用视频模型中的时空视觉先验和 VLM 的语义意识，弥合精细物理交互与高层推理之间的差距，朝着可扩展、可解释和通用的 embodied 智能迈进。 

---
# Model-Free Hovering and Source Seeking via Extremum Seeking Control: Experimental Demonstration 

**Title (ZH)**: 模型无关的悬浮与源搜索通过极值搜索控制：实验演示 

**Authors**: Ahmed A. Elgohary, Rohan Palanikumar, Sameh A. Eisa  

**Link**: [PDF](https://arxiv.org/pdf/2508.20836)  

**Abstract**: In a recent effort, we successfully proposed a categorically novel approach to mimic the phenomenoa of hovering and source seeking by flapping insects and hummingbirds using a new extremum seeking control (ESC) approach. Said ESC approach was shown capable of characterizing the physics of hovering and source seeking by flapping systems, providing at the same time uniquely novel opportunity for a model-free, real-time biomimicry control design. In this paper, we experimentally test and verify, for the first time in the literature, the potential of ESC in flapping robots to achieve model-free, real-time controlled hovering and source seeking. The results of this paper, while being restricted to 1D, confirm the premise of introducing ESC as a natural control method and biomimicry mechanism to the field of flapping flight and robotics. 

**Abstract (ZH)**: 一种新的 extremum seeking 控制方法在仿生扑翼飞行机器人悬停和源寻求中的实验验证与理论确认 

---
# A Soft Fabric-Based Thermal Haptic Device for VR and Teleoperation 

**Title (ZH)**: 基于软织物的热触觉装置：适用于VR和远程操作 

**Authors**: Rui Chen, Domenico Chiaradia, Antonio Frisoli, Daniele Leonardis  

**Link**: [PDF](https://arxiv.org/pdf/2508.20831)  

**Abstract**: This paper presents a novel fabric-based thermal-haptic interface for virtual reality and teleoperation. It integrates pneumatic actuation and conductive fabric with an innovative ultra-lightweight design, achieving only 2~g for each finger unit. By embedding heating elements within textile pneumatic chambers, the system delivers modulated pressure and thermal stimuli to fingerpads through a fully soft, wearable interface.
Comprehensive characterization demonstrates rapid thermal modulation with heating rates up to 3$^{\circ}$C/s, enabling dynamic thermal feedback for virtual or teleoperation interactions. The pneumatic subsystem generates forces up to 8.93~N at 50~kPa, while optimization of fingerpad-actuator clearance enhances cooling efficiency with minimal force reduction. Experimental validation conducted with two different user studies shows high temperature identification accuracy (0.98 overall) across three thermal levels, and significant manipulation improvements in a virtual pick-and-place tasks. Results show enhanced success rates (88.5\% to 96.4\%, p = 0.029) and improved force control precision (p = 0.013) when haptic feedback is enabled, validating the effectiveness of the integrated thermal-haptic approach for advanced human-machine interaction applications. 

**Abstract (ZH)**: 基于织物的新型热触觉界面：适用于虚拟现实和远程操作的新设计 

---
# Uncertainty Aware-Predictive Control Barrier Functions: Safer Human Robot Interaction through Probabilistic Motion Forecasting 

**Title (ZH)**: 不确定性aware预测控制障碍函数：基于概率运动预测的安全Human-机器人交互 

**Authors**: Lorenzo Busellato, Federico Cunico, Diego Dall'Alba, Marco Emporio, Andrea Giachetti, Riccardo Muradore, Marco Cristani  

**Link**: [PDF](https://arxiv.org/pdf/2508.20812)  

**Abstract**: To enable flexible, high-throughput automation in settings where people and robots share workspaces, collaborative robotic cells must reconcile stringent safety guarantees with the need for responsive and effective behavior. A dynamic obstacle is the stochastic, task-dependent variability of human motion: when robots fall back on purely reactive or worst-case envelopes, they brake unnecessarily, stall task progress, and tamper with the fluidity that true Human-Robot Interaction demands. In recent years, learning-based human-motion prediction has rapidly advanced, although most approaches produce worst-case scenario forecasts that often do not treat prediction uncertainty in a well-structured way, resulting in over-conservative planning algorithms, limiting their flexibility. We introduce Uncertainty-Aware Predictive Control Barrier Functions (UA-PCBFs), a unified framework that fuses probabilistic human hand motion forecasting with the formal safety guarantees of Control Barrier Functions. In contrast to other variants, our framework allows for dynamic adjustment of the safety margin thanks to the human motion uncertainty estimation provided by a forecasting module. Thanks to uncertainty estimation, UA-PCBFs empower collaborative robots with a deeper understanding of future human states, facilitating more fluid and intelligent interactions through informed motion planning. We validate UA-PCBFs through comprehensive real-world experiments with an increasing level of realism, including automated setups (to perform exactly repeatable motions) with a robotic hand and direct human-robot interactions (to validate promptness, usability, and human confidence). Relative to state-of-the-art HRI architectures, UA-PCBFs show better performance in task-critical metrics, significantly reducing the number of violations of the robot's safe space during interaction with respect to the state-of-the-art. 

**Abstract (ZH)**: 适应人机共存 workspace 的灵活高通量自动化协作机器人细胞必须在确保严格安全保证的同时满足响应性和有效性的需求。基于不确定性预测的控制屏障函数（UA-PCBFs）：一种将概率性的手部运动预测与控制屏障函数的形式安全保证统一的框架 

---
# Non-expert to Expert Motion Translation Using Generative Adversarial Networks 

**Title (ZH)**: 非专家到专家级运动转换的生成对抗网络方法 

**Authors**: Yuki Tanaka, Seiichiro Katsura  

**Link**: [PDF](https://arxiv.org/pdf/2508.20740)  

**Abstract**: Decreasing skilled workers is a very serious problem in the world. To deal with this problem, the skill transfer from experts to robots has been researched. These methods which teach robots by human motion are called imitation learning. Experts' skills generally appear in not only position data, but also force data. Thus, position and force data need to be saved and reproduced. To realize this, a lot of research has been conducted in the framework of a motion-copying system. Recent research uses machine learning methods to generate motion commands. However, most of them could not change tasks by following human intention. Some of them can change tasks by conditional training, but the labels are limited. Thus, we propose the flexible motion translation method by using Generative Adversarial Networks. The proposed method enables users to teach robots tasks by inputting data, and skills by a trained model. We evaluated the proposed system with a 3-DOF calligraphy robot. 

**Abstract (ZH)**: 熟练工人的减少是一个非常严重的世界性问题。为解决这一问题，专家技能向机器人转移的研究已经展开。通过人类动作来教导机器人的方法被称为模仿学习。专家的技能不仅体现在位置数据中，还体现在力数据中。因此，需要保存并再现位置和力数据。为实现这一目标，运动复制系统框架内的大量研究已经进行。最近的研究使用机器学习方法生成运动指令，但大多数方法无法根据人类意图改变任务。其中一些方法可以通过条件训练改变任务，但标签有限。因此，我们提出了使用生成对抗网络的灵活运动转换方法。该方法使用户能够通过输入数据和训练模型来教导机器人任务。我们使用一个3-DOF书法机器人评估了提出系统。 

---
# Task Allocation for Autonomous Machines using Computational Intelligence and Deep Reinforcement Learning 

**Title (ZH)**: 使用计算智能和深度强化学习的自主机器任务分配 

**Authors**: Thanh Thi Nguyen, Quoc Viet Hung Nguyen, Jonathan Kua, Imran Razzak, Dung Nguyen, Saeid Nahavandi  

**Link**: [PDF](https://arxiv.org/pdf/2508.20688)  

**Abstract**: Enabling multiple autonomous machines to perform reliably requires the development of efficient cooperative control algorithms. This paper presents a survey of algorithms that have been developed for controlling and coordinating autonomous machines in complex environments. We especially focus on task allocation methods using computational intelligence (CI) and deep reinforcement learning (RL). The advantages and disadvantages of the surveyed methods are analysed thoroughly. We also propose and discuss in detail various future research directions that shed light on how to improve existing algorithms or create new methods to enhance the employability and performance of autonomous machines in real-world applications. The findings indicate that CI and deep RL methods provide viable approaches to addressing complex task allocation problems in dynamic and uncertain environments. The recent development of deep RL has greatly contributed to the literature on controlling and coordinating autonomous machines, and it has become a growing trend in this area. It is envisaged that this paper will provide researchers and engineers with a comprehensive overview of progress in machine learning research related to autonomous machines. It also highlights underexplored areas, identifies emerging methodologies, and suggests new avenues for exploration in future research within this domain. 

**Abstract (ZH)**: Enables 多个自主机器可靠地执行任务需要开发高效的协同控制算法。本文综述了用于在复杂环境中控制和协调自主机器的算法，特别关注使用计算智能(CI)和深度强化学习(RL)的任务分配方法。我们详细分析了所综述方法的优势和劣势，并提出了改进现有算法或创造新方法以提高自主机器在实际应用中的适用性和性能的多种未来研究方向。这些发现表明，计算智能和深度RL方法为解决动态和不确定性环境中的复杂任务分配问题提供了可行的途径。近年来深度RL的发展极大地推动了自主机器控制和协调研究文献的发展，已成为该领域的研究趋势。本文旨在为研究人员和工程师提供自主机器相关的机器学习研究进展的综合性概述，同时也指出了未充分开发的领域，识别了新兴方法论，并为未来研究指出了新的探索方向。 

---
# Task-Oriented Edge-Assisted Cross-System Design for Real-Time Human-Robot Interaction in Industrial Metaverse 

**Title (ZH)**: 面向任务的边缘辅助跨系统设计：工业元宇宙中实时人机交互 

**Authors**: Kan Chen, Zhen Meng, Xiangmin Xu, Jiaming Yang, Emma Li, Philip G. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20664)  

**Abstract**: Real-time human-device interaction in industrial Metaverse faces challenges such as high computational load, limited bandwidth, and strict latency. This paper proposes a task-oriented edge-assisted cross-system framework using digital twins (DTs) to enable responsive interactions. By predicting operator motions, the system supports: 1) proactive Metaverse rendering for visual feedback, and 2) preemptive control of remote devices. The DTs are decoupled into two virtual functions-visual display and robotic control-optimizing both performance and adaptability. To enhance generalizability, we introduce the Human-In-The-Loop Model-Agnostic Meta-Learning (HITL-MAML) algorithm, which dynamically adjusts prediction horizons. Evaluation on two tasks demonstrates the framework's effectiveness: in a Trajectory-Based Drawing Control task, it reduces weighted RMSE from 0.0712 m to 0.0101 m; in a real-time 3D scene representation task for nuclear decommissioning, it achieves a PSNR of 22.11, SSIM of 0.8729, and LPIPS of 0.1298. These results show the framework's capability to ensure spatial precision and visual fidelity in real-time, high-risk industrial environments. 

**Abstract (ZH)**: 工业元宇宙中基于边缘协助的任务驱动数字孪生框架及其应用 

---
# Traversing the Narrow Path: A Two-Stage Reinforcement Learning Framework for Humanoid Beam Walking 

**Title (ZH)**: 穿越狭窄路径：类人摆动走行的两阶段强化学习框架 

**Authors**: TianChen Huang, Wei Gao, Runchen Xu, Shiwu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20661)  

**Abstract**: Traversing narrow beams is challenging for humanoids due to sparse, safety-critical contacts and the fragility of purely learned policies. We propose a physically grounded, two-stage framework that couples an XCoM/LIPM footstep template with a lightweight residual planner and a simple low-level tracker. Stage-1 is trained on flat ground: the tracker learns to robustly follow footstep targets by adding small random perturbations to heuristic footsteps, without any hand-crafted centerline locking, so it acquires stable contact scheduling and strong target-tracking robustness. Stage-2 is trained in simulation on a beam: a high-level planner predicts a body-frame residual (Delta x, Delta y, Delta psi) for the swing foot only, refining the template step to prioritize safe, precise placement under narrow support while preserving interpretability. To ease deployment, sensing is kept minimal and consistent between simulation and hardware: the planner consumes compact, forward-facing elevation cues together with onboard IMU and joint signals. On a Unitree G1, our system reliably traverses a 0.2 m-wide, 3 m-long beam. Across simulation and real-world studies, residual refinement consistently outperforms template-only and monolithic baselines in success rate, centerline adherence, and safety margins, while the structured footstep interface enables transparent analysis and low-friction sim-to-real transfer. 

**Abstract (ZH)**: 基于物理的双阶段框架实现人体形机器人跨窄_beam_行走的挑战与解决方案 

---
# SimShear: Sim-to-Real Shear-based Tactile Servoing 

**Title (ZH)**: SimShear: 基于剪切的模拟到现实的触觉伺服控制 

**Authors**: Kipp McAdam Freud, Yijiong Lin, Nathan F. Lepora  

**Link**: [PDF](https://arxiv.org/pdf/2508.20561)  

**Abstract**: We present SimShear, a sim-to-real pipeline for tactile control that enables the use of shear information without explicitly modeling shear dynamics in simulation. Shear, arising from lateral movements across contact surfaces, is critical for tasks involving dynamic object interactions but remains challenging to simulate. To address this, we introduce shPix2pix, a shear-conditioned U-Net GAN that transforms simulated tactile images absent of shear, together with a vector encoding shear information, into realistic equivalents with shear deformations. This method outperforms baseline pix2pix approaches in simulating tactile images and in pose/shear prediction. We apply SimShear to two control tasks using a pair of low-cost desktop robotic arms equipped with a vision-based tactile sensor: (i) a tactile tracking task, where a follower arm tracks a surface moved by a leader arm, and (ii) a collaborative co-lifting task, where both arms jointly hold an object while the leader follows a prescribed trajectory. Our method maintains contact errors within 1 to 2 mm across varied trajectories where shear sensing is essential, validating the feasibility of sim-to-real shear modeling with rigid-body simulators and opening new directions for simulation in tactile robotics. 

**Abstract (ZH)**: SimShear：一种无需显式建模摩擦动力学的摩擦信息仿真到现实的管道 

---
# SPGrasp: Spatiotemporal Prompt-driven Grasp Synthesis in Dynamic Scenes 

**Title (ZH)**: SPGrasp: 动态场景中基于空间时间提示的抓取合成 

**Authors**: Yunpeng Mei, Hongjie Cao, Yinqiu Xia, Wei Xiao, Zhaohan Feng, Gang Wang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20547)  

**Abstract**: Real-time interactive grasp synthesis for dynamic objects remains challenging as existing methods fail to achieve low-latency inference while maintaining promptability. To bridge this gap, we propose SPGrasp (spatiotemporal prompt-driven dynamic grasp synthesis), a novel framework extending segment anything model v2 (SAMv2) for video stream grasp estimation. Our core innovation integrates user prompts with spatiotemporal context, enabling real-time interaction with end-to-end latency as low as 59 ms while ensuring temporal consistency for dynamic objects. In benchmark evaluations, SPGrasp achieves instance-level grasp accuracies of 90.6% on OCID and 93.8% on Jacquard. On the challenging GraspNet-1Billion dataset under continuous tracking, SPGrasp achieves 92.0% accuracy with 73.1 ms per-frame latency, representing a 58.5% reduction compared to the prior state-of-the-art promptable method RoG-SAM while maintaining competitive accuracy. Real-world experiments involving 13 moving objects demonstrate a 94.8% success rate in interactive grasping scenarios. These results confirm SPGrasp effectively resolves the latency-interactivity trade-off in dynamic grasp synthesis. Code is available at this https URL. 

**Abstract (ZH)**: 实时交互动态物体抓取合成：时空提示驱动的方法 

---
# Learning Fast, Tool aware Collision Avoidance for Collaborative Robots 

**Title (ZH)**: 学习快速的工具感知碰撞避免方法——协作机器人应用 

**Authors**: Joonho Lee, Yunho Kim, Seokjoon Kim, Quan Nguyen, Youngjin Heo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20457)  

**Abstract**: Ensuring safe and efficient operation of collaborative robots in human environments is challenging, especially in dynamic settings where both obstacle motion and tasks change over time. Current robot controllers typically assume full visibility and fixed tools, which can lead to collisions or overly conservative behavior. In our work, we introduce a tool-aware collision avoidance system that adjusts in real time to different tool sizes and modes of tool-environment interaction. Using a learned perception model, our system filters out robot and tool components from the point cloud, reasons about occluded area, and predicts collision under partial observability. We then use a control policy trained via constrained reinforcement learning to produce smooth avoidance maneuvers in under 10 milliseconds. In simulated and real-world tests, our approach outperforms traditional approaches (APF, MPPI) in dynamic environments, while maintaining sub-millimeter accuracy. Moreover, our system operates with approximately 60% lower computational cost compared to a state-of-the-art GPU-based planner. Our approach provides modular, efficient, and effective collision avoidance for robots operating in dynamic environments. We integrate our method into a collaborative robot application and demonstrate its practical use for safe and responsive operation. 

**Abstract (ZH)**: 确保协作机器人在人类环境中的安全高效运行具有挑战性，特别是在动态环境中，障碍物和任务会随时间发生变化。当前的机器人控制器通常假设完全可见性和固定工具，这可能导致碰撞或过于保守的行为。在我们的工作中，我们引入了一种工具感知的碰撞避免系统，该系统能够实时调整不同的工具尺寸和工具-环境互动模式。借助学习到的感知模型，我们的系统过滤掉机器人和工具组件，推理出被遮挡的区域，并在部分可观测性下预测碰撞。然后，我们使用通过约束强化学习训练的控制策略，在不到10毫秒的时间内生成平滑的避免动作。在模拟和现实世界的测试中，我们的方法在动态环境中比传统方法（如APF、MPPI）表现出更优的性能，同时保持亚毫米级的精度。此外，与最先进的基于GPU的规划器相比，我们的系统计算成本大约降低了60%。我们的方法为动态环境中的机器人提供了模块化、高效和有效的碰撞避免。我们将该方法集成到协作机器人应用中，并展示了其在安全和响应性操作中的实际应用。 

---
# CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification 

**Title (ZH)**: CogVLA：指令驱动的路由与稀疏化认知对齐的视觉-语言-行动模型 

**Authors**: Wei Li, Renshan Zhang, Rui Shao, Jie He, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.21046)  

**Abstract**: Recent Vision-Language-Action (VLA) models built on pre-trained Vision-Language Models (VLMs) require extensive post-training, resulting in high computational overhead that limits scalability and this http URL propose CogVLA, a Cognition-Aligned Vision-Language-Action framework that leverages instruction-driven routing and sparsification to improve both efficiency and performance. CogVLA draws inspiration from human multimodal coordination and introduces a 3-stage progressive architecture. 1) Encoder-FiLM based Aggregation Routing (EFA-Routing) injects instruction information into the vision encoder to selectively aggregate and compress dual-stream visual tokens, forming a instruction-aware latent representation. 2) Building upon this compact visual encoding, LLM-FiLM based Pruning Routing (LFP-Routing) introduces action intent into the language model by pruning instruction-irrelevant visually grounded tokens, thereby achieving token-level sparsity. 3) To ensure that compressed perception inputs can still support accurate and coherent action generation, we introduce V-L-A Coupled Attention (CAtten), which combines causal vision-language attention with bidirectional action parallel decoding. Extensive experiments on the LIBERO benchmark and real-world robotic tasks demonstrate that CogVLA achieves state-of-the-art performance with success rates of 97.4% and 70.0%, respectively, while reducing training costs by 2.5-fold and decreasing inference latency by 2.8-fold compared to OpenVLA. CogVLA is open-sourced and publicly available at this https URL. 

**Abstract (ZH)**: 基于预训练视觉语言模型的 Recent Vision-Language-Action (VLA) 模型在后训练过程中需要大量的计算，导致计算开销高并限制了其可扩展性，本文提出CogVLA：一种认知对齐的视觉语言行动框架，通过指令驱动的路由和稀疏化提高效率与性能 

---
# Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees 

**Title (ZH)**: 一次训练，处处规划：基于扩散树的运动规划 

**Authors**: Yaniv Hassidof, Tom Jurgenson, Kiril Solovey  

**Link**: [PDF](https://arxiv.org/pdf/2508.21001)  

**Abstract**: Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a \emph{provably-generalizable} framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield \emph{provably-safe} solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a \emph{single environment}. In comprehensive evaluations on OOD scenarios, % DiTree has comparable runtimes to a standalone DP (3x faster than classical SBPs), while improving the average success rate over DP and SBPs. DiTree is on average 3x faster than classical SBPs, and outperforms all other approaches by achieving roughly 30\% higher success rate. Project webpage: this https URL. 

**Abstract (ZH)**: 基于扩散策略的可证明通用化动规树规划方法：DiTree 

---
# COMETH: Convex Optimization for Multiview Estimation and Tracking of Humans 

**Title (ZH)**: COMETH：多视图人类估计与跟踪的凸优化方法 

**Authors**: Enrico Martini, Ho Jin Choi, Nadia Figueroa, Nicola Bombieri  

**Link**: [PDF](https://arxiv.org/pdf/2508.20920)  

**Abstract**: In the era of Industry 5.0, monitoring human activity is essential for ensuring both ergonomic safety and overall well-being. While multi-camera centralized setups improve pose estimation accuracy, they often suffer from high computational costs and bandwidth requirements, limiting scalability and real-time applicability. Distributing processing across edge devices can reduce network bandwidth and computational load. On the other hand, the constrained resources of edge devices lead to accuracy degradation, and the distribution of computation leads to temporal and spatial inconsistencies. We address this challenge by proposing COMETH (Convex Optimization for Multiview Estimation and Tracking of Humans), a lightweight algorithm for real-time multi-view human pose fusion that relies on three concepts: it integrates kinematic and biomechanical constraints to increase the joint positioning accuracy; it employs convex optimization-based inverse kinematics for spatial fusion; and it implements a state observer to improve temporal consistency. We evaluate COMETH on both public and industrial datasets, where it outperforms state-of-the-art methods in localization, detection, and tracking accuracy. The proposed fusion pipeline enables accurate and scalable human motion tracking, making it well-suited for industrial and safety-critical applications. The code is publicly available at this https URL. 

**Abstract (ZH)**: 在 Industry 5.0 时代，监控人类活动对于确保人体工程学安全和整体福祉至关重要。虽然多相机集中设置可以提高姿态估计准确性，但往往会受到高计算成本和带宽需求的影响，限制了其 scalability 和实时适用性。将处理分散到边缘设备可以降低网络带宽和计算负载。另一方面，边缘设备资源受限会导致准确性下降，而计算的分散会导致时间和空间不一致性。我们通过提出 COMETH（Convex Optimization for Multiview Estimation and Tracking of Humans）来应对这一挑战，这是一种依赖于三个概念的轻量化实时多视角人类姿态融合算法：它结合运动学和生物力学约束以提高关节定位准确性；利用基于凸优化的逆运动学进行空间融合；并实施状态观察器以提高时间一致性。我们在公共和工业数据集上评估了 COMETH，在定位、检测和跟踪准确性方面均优于现有方法。所提出的融合流水线实现了准确且可扩展的人体运动追踪，使其适用于工业和安全关键应用。代码已在以下网址开源：this https URL。 

---
# To New Beginnings: A Survey of Unified Perception in Autonomous Vehicle Software 

**Title (ZH)**: 新的起点：自主车辆软件中统一感知的综述 

**Authors**: Loïc Stratil, Felix Fent, Esteban Rivera, Markus Lienkamp  

**Link**: [PDF](https://arxiv.org/pdf/2508.20892)  

**Abstract**: Autonomous vehicle perception typically relies on modular pipelines that decompose the task into detection, tracking, and prediction. While interpretable, these pipelines suffer from error accumulation and limited inter-task synergy. Unified perception has emerged as a promising paradigm that integrates these sub-tasks within a shared architecture, potentially improving robustness, contextual reasoning, and efficiency while retaining interpretable outputs. In this survey, we provide a comprehensive overview of unified perception, introducing a holistic and systemic taxonomy that categorizes methods along task integration, tracking formulation, and representation flow. We define three paradigms -Early, Late, and Full Unified Perception- and systematically review existing methods, their architectures, training strategies, datasets used, and open-source availability, while highlighting future research directions. This work establishes the first comprehensive framework for understanding and advancing unified perception, consolidates fragmented efforts, and guides future research toward more robust, generalizable, and interpretable perception. 

**Abstract (ZH)**: 自主驾驶车辆感知通常依赖于模块化的管道，将任务分解为检测、跟踪和预测。虽然具有可解释性，但这些管道易发生错误累积且各任务之间的协同作用有限。统一感知作为一种有前景的范式，将这些子任务整合到共享架构中，有可能提高鲁棒性、上下文推理能力和效率，同时保持可解释的输出。在本文综述中，我们提供了一个全面的统一感知综述，引入了一个整体和系统的分类框架，按照任务整合、跟踪公式化和表示流对方法进行分类。我们定义了三种范式——早期统一感知、晚期统一感知和全统一感知——并对现有方法、架构、训练策略、使用的数据集以及开源可用性进行了系统性回顾，同时指出了未来的研究方向。这项工作建立了第一个全面的统一感知理解与推进框架，整合了分散的努力，并为未来研究指明了更鲁棒、更通用和更可解释的感知方向。 

---
# Encoding Tactile Stimuli for Organoid Intelligence in Braille Recognition 

**Title (ZH)**: 用于盲文识别的组织体触觉编码智能 

**Authors**: Tianyi Liu, Hemma Philamore, Benjamin Ward-Cherrier  

**Link**: [PDF](https://arxiv.org/pdf/2508.20850)  

**Abstract**: This study proposes a generalizable encoding strategy that maps tactile sensor data to electrical stimulation patterns, enabling neural organoids to perform an open-loop artificial tactile Braille classification task. Human forebrain organoids cultured on a low-density microelectrode array (MEA) are systematically stimulated to characterize the relationship between electrical stimulation parameters (number of pulse, phase amplitude, phase duration, and trigger delay) and organoid responses, measured as spike activity and spatial displacement of the center of activity. Implemented on event-based tactile inputs recorded from the Evetac sensor, our system achieved an average Braille letter classification accuracy of 61 percent with a single organoid, which increased significantly to 83 percent when responses from a three-organoid ensemble were combined. Additionally, the multi-organoid configuration demonstrated enhanced robustness against various types of artificially introduced noise. This research demonstrates the potential of organoids as low-power, adaptive bio-hybrid computational elements and provides a foundational encoding framework for future scalable bio-hybrid computing architectures. 

**Abstract (ZH)**: 本研究提出了一种可推广的编码策略，将触觉传感器数据映射到电刺激模式，使神经器官球能够执行开放环人工触觉盲文分类任务。在低密度微电极阵列(MEA)上培养的人类前脑器官球系统地受到刺激，以表征电刺激参数（脉冲数、相位幅度、相位持续时间和触发延迟）与器官球反应之间的关系，器官球反应通过尖峰活动和活动中心的空间位移进行测量。在事件驱动的触觉输入从Evetac传感器记录的数据上实现时，该系统在单个器官球上的平均盲文字母分类准确率为61%，当结合三个器官球的反应时，分类准确率显著提高至83%。此外，多器官球配置展示了对各种人工引入噪声的增强鲁棒性。该研究展示了器官球作为低能耗、自适应的生物杂合计算元件的潜力，并为未来的可扩展生物杂合计算架构提供了一种基础编码框架。 

---
# SKGE-SWIN: End-To-End Autonomous Vehicle Waypoint Prediction and Navigation Using Skip Stage Swin Transformer 

**Title (ZH)**: SKGE-SWIN：基于跳层Swin变换器的端到端自主车辆 waypoints 预测与导航 

**Authors**: Fachri Najm Noer Kartiman, Rasim, Yaya Wihardi, Nurul Hasanah, Oskar Natan, Bambang Wahono, Taufik Ibnu Salim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20762)  

**Abstract**: Focusing on the development of an end-to-end autonomous vehicle model with pixel-to-pixel context awareness, this research proposes the SKGE-Swin architecture. This architecture utilizes the Swin Transformer with a skip-stage mechanism to broaden feature representation globally and at various network levels. This approach enables the model to extract information from distant pixels by leveraging the Swin Transformer's Shifted Window-based Multi-head Self-Attention (SW-MSA) mechanism and to retain critical information from the initial to the final stages of feature extraction, thereby enhancing its capability to comprehend complex patterns in the vehicle's surroundings. The model is evaluated on the CARLA platform using adversarial scenarios to simulate real-world conditions. Experimental results demonstrate that the SKGE-Swin architecture achieves a superior Driving Score compared to previous methods. Furthermore, an ablation study will be conducted to evaluate the contribution of each architectural component, including the influence of skip connections and the use of the Swin Transformer, in improving model performance. 

**Abstract (ZH)**: 基于像素到像素的上下文感知，本研究提出了一种SKGE-Swin架构，该架构利用具有跳跃阶段机制的Swin Transformer以全局和各种网络层次扩展特征表示。通过利用Swin Transformer基于移窗的多头自注意力（SW-MSA）机制，模型能够从远处的像素中提取信息，并在特征提取的初始阶段到最终阶段保留关键信息，从而增强其理解车辆周围复杂模式的能力。该模型在CARLA平台上使用对抗场景进行评估，以模拟现实世界条件。实验结果表明，SKGE-Swin架构在驾驶得分方面优于先前的方法。此外，还将进行消融研究以评估每个架构组件的贡献，包括跳跃连接和使用Swin Transformer对模型性能的改善影响。 

---
# Regulation-Aware Game-Theoretic Motion Planning for Autonomous Racing 

**Title (ZH)**: 基于博弈论的自主赛车运动规划，考虑监管要求 

**Authors**: Francesco Prignoli, Francesco Borrelli, Paolo Falcone, Mark Pustilnik  

**Link**: [PDF](https://arxiv.org/pdf/2508.20203)  

**Abstract**: This paper presents a regulation-aware motion planning framework for autonomous racing scenarios. Each agent solves a Regulation-Compliant Model Predictive Control problem, where racing rules - such as right-of-way and collision avoidance responsibilities - are encoded using Mixed Logical Dynamical constraints. We formalize the interaction between vehicles as a Generalized Nash Equilibrium Problem (GNEP) and approximate its solution using an Iterative Best Response scheme. Building on this, we introduce the Regulation-Aware Game-Theoretic Planner (RA-GTP), in which the attacker reasons over the defender's regulation-constrained behavior. This game-theoretic layer enables the generation of overtaking strategies that are both safe and non-conservative. Simulation results demonstrate that the RA-GTP outperforms baseline methods that assume non-interacting or rule-agnostic opponent models, leading to more effective maneuvers while consistently maintaining compliance with racing regulations. 

**Abstract (ZH)**: 基于规则意识的自主赛车运动规划框架 

---
