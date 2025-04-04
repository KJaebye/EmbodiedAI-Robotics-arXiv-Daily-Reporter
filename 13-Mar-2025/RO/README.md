# Action-Aware Pro-Active Safe Exploration for Mobile Robot Mapping 

**Title (ZH)**: 基于动作意识的主动安全探索方法在移动机器人建图中的应用 

**Authors**: Aykut İşleyen, René van de Molengraft, Ömür Arslan  

**Link**: [PDF](https://arxiv.org/pdf/2503.09515)  

**Abstract**: Safe autonomous exploration of unknown environments is an essential skill for mobile robots to effectively and adaptively perform environmental mapping for diverse critical tasks. Due to its simplicity, most existing exploration methods rely on the standard frontier-based exploration strategy, which directs a robot to the boundary between the known safe and the unknown unexplored spaces to acquire new information about the environment. This typically follows a recurrent persistent planning strategy, first selecting an informative frontier viewpoint, then moving the robot toward the selected viewpoint until reaching it, and repeating these steps until termination. However, exploration with persistent planning may lack adaptivity to continuously updated maps, whereas highly adaptive exploration with online planning often suffers from high computational costs and potential issues with livelocks. In this paper, as an alternative to less-adaptive persistent planning and costly online planning, we introduce a new proactive preventive replanning strategy for effective exploration using the immediately available actionable information at a viewpoint to avoid redundant, uninformative last-mile exploration motion. We also use the actionable information of a viewpoint as a systematic termination criterion for exploration. To close the gap between perception and action, we perform safe and informative path planning that minimizes the risk of collision with detected obstacles and the distance to unexplored regions, and we apply action-aware viewpoint selection with maximal information utility per total navigation cost. We demonstrate the effectiveness of our action-aware proactive exploration method in numerical simulations and hardware experiments. 

**Abstract (ZH)**: 安全自主探索未知环境是移动机器人有效适应性执行环境建图以完成多样关键任务的一项基本技能。大多数现有探索方法依赖于标准的前沿基探索策略，该策略指导机器人前往已知安全区域与未知未探索区域的边界，以获取有关环境的新信息。这种探索通常遵循循环坚持规划策略，首先选择一个有信息价值的前沿视角，然后将机器人移动到选定的视角，直到到达，然后重复这些步骤直到终止。然而，坚持规划的探索缺乏对持续更新地图的适应性，而在线规划的高适应性探索往往面临较高的计算成本和潜在死锁问题。在本文中，作为少适应性坚持规划和高成本在线规划的替代方案，我们引入了一种基于即时可用可操作信息的主动预防性重规划策略，以有效探索并避免冗余、无信息的最后阶段探索运动。我们还使用视点的可操作信息作为探索的系统终止标准。为了弥合感知与行动之间的差距，我们执行了安全且信息丰富的路径规划，以最小化与检测到的障碍物碰撞的风险和到未探索区域的距离，并应用了具有最大信息效用的感知行动视点选择，以最小化总导航成本。我们通过数值仿真和硬件实验展示了我们感知行动的主动探索方法的有效性。 

---
# Neural reservoir control of a soft bio-hybrid arm 

**Title (ZH)**: 软生物杂合臂的神经蓄积控制 

**Authors**: Noel Naughton, Arman Tekinalp, Keshav Shivam, Seung Hung Kim, Volodymyr Kindratenko, Mattia Gazzola  

**Link**: [PDF](https://arxiv.org/pdf/2503.09477)  

**Abstract**: A long-standing engineering problem, the control of soft robots is difficult because of their highly non-linear, heterogeneous, anisotropic, and distributed nature. Here, bridging engineering and biology, a neural reservoir is employed for the dynamic control of a bio-hybrid model arm made of multiple muscle-tendon groups enveloping an elastic spine. We show how the use of reservoirs facilitates simultaneous control and self-modeling across a set of challenging tasks, outperforming classic neural network approaches. Further, by implementing a spiking reservoir on neuromorphic hardware, energy efficiency is achieved, with nearly two-orders of magnitude improvement relative to standard CPUs, with implications for the on-board control of untethered, small-scale soft robots. 

**Abstract (ZH)**: 软体机器人的动态控制是一个长期存在的工程问题，由于其高度非线性、异质性、各向异性以及分布特性，控制非常困难。本文通过融合工程学和生物学，采用神经水库对由多组肌肉-肌腱群包裹弹性脊柱构成的生物-混合模型臂进行动力学控制。我们展示了如何通过使用神经水库实现对一系列具有挑战性的任务同时进行控制和自我建模，超越了经典神经网络方法。此外，通过在神经形态硬件上实现脉冲神经水库，实现了能效提升，相比于普通CPU提高了近两个数量级，对无缆操控小型软体机器人的机载控制具有重要意义。 

---
# Neural-Augmented Incremental Nonlinear Dynamic Inversion for Quadrotors with Payload Adaptation 

**Title (ZH)**: 基于神经增强增量非线性动态反转的载荷自适应四旋翼控制 

**Authors**: Eckart Cobo-Briesewitz, Khaled Wahba, Wolfgang Hönig  

**Link**: [PDF](https://arxiv.org/pdf/2503.09441)  

**Abstract**: The increasing complexity of multirotor applications has led to the need of more accurate flight controllers that can reliably predict all forces acting on the robot. Traditional flight controllers model a large part of the forces but do not take so called residual forces into account. A reason for this is that accurately computing the residual forces can be computationally expensive. Incremental Nonlinear Dynamic Inversion (INDI) is a method that computes the difference between different sensor measurements in order to estimate these residual forces. The main issue with INDI is it's reliance on special sensor measurements which can be very noisy. Recent work has also shown that residual forces can be predicted using learning-based methods. In this work, we demonstrate that a learning algorithm can predict a smoother version of INDI outputs without requiring additional sensor measurements. In addition, we introduce a new method that combines learning based predictions with INDI. We also adapt the two approaches to work on quadrotors carrying a slung-type payload. The results show that using a neural network to predict residual forces can outperform INDI while using the combination of neural network and INDI can yield even better results than each method individually. 

**Abstract (ZH)**: 多旋翼应用复杂性的增加促使需要更准确的飞行控制器以可靠地预测作用于机器人上的所有力。传统的飞行控制器建模了大部分力，但没有考虑所谓的残余力。这主要是因为准确计算残余力可能是计算上昂贵的。增量非线性动态逆（INDI）是一种方法，通过计算不同传感器测量值之间的差异来估计这些残余力。INDI的主要问题是依赖于特殊的传感器测量，这些测量可能会非常嘈杂。最近的研究还表明，可以使用基于学习的方法来预测残余力。在这项工作中，我们证明了一种学习算法可以在不需要额外传感器测量的情况下预测INDI输出的平滑版本。此外，我们提出了一种新方法，将基于学习的预测与INDI相结合。我们还将两种方法适应于悬挂载荷的四旋翼。结果显示，使用神经网络预测残余力可以优于INDI，而将神经网络与INDI结合使用则可以比各自的方法获得更好的结果。 

---
# Efficient Alignment of Unconditioned Action Prior for Language-conditioned Pick and Place in Clutter 

**Title (ZH)**: 语言条件下的拾取和放置在杂乱环境中的无条件动作先验高效对齐 

**Authors**: Kechun Xu, Xunlong Xia, Kaixuan Wang, Yifei Yang, Yunxuan Mao, Bing Deng, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09423)  

**Abstract**: We study the task of language-conditioned pick and place in clutter, where a robot should grasp a target object in open clutter and move it to a specified place. Some approaches learn end-to-end policies with features from vision foundation models, requiring large datasets. Others combine foundation models in a zero-shot setting, suffering from cascading errors. In addition, they primarily leverage vision and language foundation models, focusing less on action priors. In this paper, we aim to develop an effective policy by integrating foundation priors from vision, language, and action. We propose A$^2$, an action prior alignment method that aligns unconditioned action priors with 3D vision-language priors by learning one attention layer. The alignment formulation enables our policy to train with less data and preserve zero-shot generalization capabilities. We show that a shared policy for both pick and place actions enhances the performance for each task, and introduce a policy adaptation scheme to accommodate the multi-modal nature of actions. Extensive experiments in simulation and the real-world show that our policy achieves higher task success rates with fewer steps for both pick and place tasks in clutter, effectively generalizing to unseen objects and language instructions. 

**Abstract (ZH)**: 基于视觉-语言先验的混合行动先验对齐方法 kobold A$^2$及其在杂乱环境中抓取与放置任务中的应用 

---
# AI-based Framework for Robust Model-Based Connector Mating in Robotic Wire Harness Installation 

**Title (ZH)**: 基于AI的鲁棒模型导向连接器对接框架在机器人线束安装中的应用 

**Authors**: Claudius Kienle, Benjamin Alt, Finn Schneider, Tobias Pertlwieser, Rainer Jäkel, Rania Rayyes  

**Link**: [PDF](https://arxiv.org/pdf/2503.09409)  

**Abstract**: Despite the widespread adoption of industrial robots in automotive assembly, wire harness installation remains a largely manual process, as it requires precise and flexible manipulation. To address this challenge, we design a novel AI-based framework that automates cable connector mating by integrating force control with deep visuotactile learning. Our system optimizes search-and-insertion strategies using first-order optimization over a multimodal transformer architecture trained on visual, tactile, and proprioceptive data. Additionally, we design a novel automated data collection and optimization pipeline that minimizes the need for machine learning expertise. The framework optimizes robot programs that run natively on standard industrial controllers, permitting human experts to audit and certify them. Experimental validations on a center console assembly task demonstrate significant improvements in cycle times and robustness compared to conventional robot programming approaches. Videos are available under this https URL. 

**Abstract (ZH)**: 尽管工业机器人在汽车装配中得到广泛应用，线束安装过程仍主要依赖手动操作，因为它需要精确且灵活的操作。为应对这一挑战，我们设计了一种基于AI的新型框架，通过结合力控制与深度视触觉学习来实现电缆连接器的自动对接。该系统使用多元模态变换器架构对视觉、触觉和本体感受数据进行训练，以第一阶优化方法优化搜索和插入策略。此外，我们还设计了一种新型的自动化数据采集和优化管道，以减少对机器学习专业知识的需求。该框架优化的机器人程序可以在标准工业控制器上原生运行，允许人类专家审核和认证这些程序。在中心控制台装配任务上的实验验证表明，与传统的机器人编程方法相比，该框架能在循环时间和鲁棒性方面取得显著改进。相关视频可访问此链接。 

---
# Robust Self-Reconfiguration for Fault-Tolerant Control of Modular Aerial Robot Systems 

**Title (ZH)**: 模块化空中机器人系统容错控制的鲁棒自重构方法 

**Authors**: Rui Huang, Siyu Tang, Zhiqian Cai, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09376)  

**Abstract**: Modular Aerial Robotic Systems (MARS) consist of multiple drone units assembled into a single, integrated rigid flying platform. With inherent redundancy, MARS can self-reconfigure into different configurations to mitigate rotor or unit failures and maintain stable flight. However, existing works on MARS self-reconfiguration often overlook the practical controllability of intermediate structures formed during the reassembly process, which limits their applicability. In this paper, we address this gap by considering the control-constrained dynamic model of MARS and proposing a robust and efficient self-reconstruction algorithm that maximizes the controllability margin at each intermediate stage. Specifically, we develop algorithms to compute optimal, controllable disassembly and assembly sequences, enabling robust self-reconfiguration. Finally, we validate our method in several challenging fault-tolerant self-reconfiguration scenarios, demonstrating significant improvements in both controllability and trajectory tracking while reducing the number of assembly steps. The videos and source code of this work are available at this https URL 

**Abstract (ZH)**: 模块化 aerial 机器人系统（MARS）由多个无人机单元组装成一个单一的集成刚性飞行平台。具有固有的冗余性，MARS 可以自我重构为不同的配置以减轻旋翼或单元故障并保持稳定的飞行。然而，现有的 MARS 自我重构工作往往忽略了重组过程中形成的中间结构的可操作性，这限制了其应用范围。在本文中，我们通过考虑 MARS 的控制约束动态模型，并提出一种 robust 和高效的自我重构算法来填补这一空白，该算法在每个中间阶段最大化可操作性裕度。具体而言，我们开发了算法来计算最优且可操作的拆卸和组装序列，从而实现 robust 自我重构。最后，我们在几个具有挑战性的容错自我重构场景中验证了我们的方法，证明了在可操作性和轨迹跟踪方面有显著改进，同时减少了组装步骤的数量。本工作的视频和源代码可从以下网址获取。 

---
# Robust Fault-Tolerant Control and Agile Trajectory Planning for Modular Aerial Robotic Systems 

**Title (ZH)**: 模块化空中机器人系统的鲁棒容错控制与敏捷轨迹规划 

**Authors**: Rui Huang, Zhenyu Zhang, Siyu Tang, Zhiqian Cai, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09351)  

**Abstract**: Modular Aerial Robotic Systems (MARS) consist of multiple drone units that can self-reconfigure to adapt to various mission requirements and fault conditions. However, existing fault-tolerant control methods exhibit significant oscillations during docking and separation, impacting system stability. To address this issue, we propose a novel fault-tolerant control reallocation method that adapts to arbitrary number of modular robots and their assembly formations. The algorithm redistributes the expected collective force and torque required for MARS to individual unit according to their moment arm relative to the center of MARS mass. Furthermore, We propose an agile trajectory planning method for MARS of arbitrary configurations, which is collision-avoiding and dynamically feasible. Our work represents the first comprehensive approach to enable fault-tolerant and collision avoidance flight for MARS. We validate our method through extensive simulations, demonstrating improved fault tolerance, enhanced trajectory tracking accuracy, and greater robustness in cluttered environments. The videos and source code of this work are available at this https URL 

**Abstract (ZH)**: 模块化空中机器人系统（MARS）由多个能够自重构以适应各种任务要求和故障状态的无人机单元组成。然而，现有的容错控制方法在对接和分离过程中表现出显著的振荡，影响系统稳定性。为解决这一问题，我们提出了一种新的容错控制重新分配方法，该方法适用于任意数量的模块化机器人及其组装形态。该算法根据各单元相对于MARS质心的力臂重新分配MARS所需的预期集体力和力矩。此外，我们还提出了一种适用于任意配置的MARS的敏捷轨迹规划方法，该方法具有避碰能力和动态可行性。我们的工作代表了首次全面的方法，以实现MARS的容错和碰撞避让飞行。我们通过广泛的仿真验证了该方法，展示了改进的容错性能、提高的轨迹跟踪精度以及在复杂环境中的更大鲁棒性。该工作的视频和源代码可在以下链接获取：this https URL 

---
# NVP-HRI: Zero Shot Natural Voice and Posture-based Human-Robot Interaction via Large Language Model 

**Title (ZH)**: NVP-HRI: 零样本基于自然语音和姿态的人机交互 via 大型语言模型 

**Authors**: Yuzhi Lai, Shenghai Yuan, Youssef Nassar, Mingyu Fan, Thomas Weber, Matthias Rätsch  

**Link**: [PDF](https://arxiv.org/pdf/2503.09335)  

**Abstract**: Effective Human-Robot Interaction (HRI) is crucial for future service robots in aging societies. Existing solutions are biased toward only well-trained objects, creating a gap when dealing with new objects. Currently, HRI systems using predefined gestures or language tokens for pretrained objects pose challenges for all individuals, especially elderly ones. These challenges include difficulties in recalling commands, memorizing hand gestures, and learning new names. This paper introduces NVP-HRI, an intuitive multi-modal HRI paradigm that combines voice commands and deictic posture. NVP-HRI utilizes the Segment Anything Model (SAM) to analyze visual cues and depth data, enabling precise structural object representation. Through a pre-trained SAM network, NVP-HRI allows interaction with new objects via zero-shot prediction, even without prior knowledge. NVP-HRI also integrates with a large language model (LLM) for multimodal commands, coordinating them with object selection and scene distribution in real time for collision-free trajectory solutions. We also regulate the action sequence with the essential control syntax to reduce LLM hallucination risks. The evaluation of diverse real-world tasks using a Universal Robot showcased up to 59.2\% efficiency improvement over traditional gesture control, as illustrated in the video this https URL. Our code and design will be openly available at this https URL. 

**Abstract (ZH)**: 面向老龄化社会的有效人机交互对于未来服务机器人至关重要。现有解决方案偏向于已训练良好的对象，对于新对象则存在差距。目前，使用预定义手势或语言标记的HRI系统对所有个体，尤其是老年人，构成了挑战。这些挑战包括命令回忆困难、手势记忆和新名字学习的难题。本文介绍了一种直观的多模态HRI范式NVP-HRI，结合了语音命令和指示性姿态。NVP-HRI利用Segment Anything Model (SAM) 分析视觉线索和深度数据，实现精确的结构化对象表示。通过预训练的SAM网络，NVP-HRI即使在没有先验知识的情况下也能通过零样本预测与新对象交互。NVP-HRI还集成了大规模语言模型（LLM）进行多模态命令协调，实时配合对象选择和场景分配以实现无碰撞轨迹解决方案。我们通过使用Universal Robot进行多样化的实际任务评估，结果显示NVP-HRI较传统手势控制的效率提升了高达59.2%，如视频所示：this https URL。我们的代码和设计将在此 https URL 公开可供。 

---
# MonoSLAM: Robust Monocular SLAM with Global Structure Optimization 

**Title (ZH)**: 单目SLAM：带全局结构优化的鲁棒单目SLAM 

**Authors**: Bingzheng Jiang, Jiayuan Wang, Han Ding, Lijun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09296)  

**Abstract**: This paper presents a robust monocular visual SLAM system that simultaneously utilizes point, line, and vanishing point features for accurate camera pose estimation and mapping. To address the critical challenge of achieving reliable localization in low-texture environments, where traditional point-based systems often fail due to insufficient visual features, we introduce a novel approach leveraging Global Primitives structural information to improve the system's robustness and accuracy performance. Our key innovation lies in constructing vanishing points from line features and proposing a weighted fusion strategy to build Global Primitives in the world coordinate system. This strategy associates multiple frames with non-overlapping regions and formulates a multi-frame reprojection error optimization, significantly improving tracking accuracy in texture-scarce scenarios. Evaluations on various datasets show that our system outperforms state-of-the-art methods in trajectory precision, particularly in challenging environments. 

**Abstract (ZH)**: 一种同时利用点、线和消失点特征的鲁棒单目视觉SLAM系统 

---
# GarmentPile: Point-Level Visual Affordance Guided Retrieval and Adaptation for Cluttered Garments Manipulation 

**Title (ZH)**: 服装堆叠：点级视觉功能引导的杂乱服装操作检索与适应 

**Authors**: Ruihai Wu, Ziyu Zhu, Yuran Wang, Yue Chen, Jiarui Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.09243)  

**Abstract**: Cluttered garments manipulation poses significant challenges due to the complex, deformable nature of garments and intricate garment relations. Unlike single-garment manipulation, cluttered scenarios require managing complex garment entanglements and interactions, while maintaining garment cleanliness and manipulation stability. To address these demands, we propose to learn point-level affordance, the dense representation modeling the complex space and multi-modal manipulation candidates, while being aware of garment geometry, structure, and inter-object relations. Additionally, as it is difficult to directly retrieve a garment in some extremely entangled clutters, we introduce an adaptation module, guided by learned affordance, to reorganize highly-entangled garments into states plausible for manipulation. Our framework demonstrates effectiveness over environments featuring diverse garment types and pile configurations in both simulation and the real world. Project page: this https URL. 

**Abstract (ZH)**: 杂乱衣物操作由于衣物的复杂可变形性质和复杂的衣物关系，提出了显著的挑战。与单件衣物操作不同，杂乱场景需要管理复杂的衣物缠绕和相互作用，同时保持衣物清洁和操作稳定性。为应对这些需求，我们提出学习点级功能，这是一种密集表示，模型复杂的空间和多模态的操作候选者，同时考虑到衣物的几何形状、结构以及物体间的关系。此外，由于在某些极其缠结的杂乱中直接检索衣物困难，我们引入了一个根据学习到的功能进行引导的适应模块，重新组织高度缠结的衣物，使其处于可行的操作状态。我们的框架在包含各种衣物类型和堆积配置的模拟和真实世界环境中都展示了有效性。项目页面：这个 https URL。 

---
# MarineGym: A High-Performance Reinforcement Learning Platform for Underwater Robotics 

**Title (ZH)**: MarineGym: 一种高性能的水下机器人强化学习平台 

**Authors**: Shuguang Chu, Zebin Huang, Yutong Li, Mingwei Lin, Ignacio Carlucho, Yvan R. Petillot, Canjun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09203)  

**Abstract**: This work presents the MarineGym, a high-performance reinforcement learning (RL) platform specifically designed for underwater robotics. It aims to address the limitations of existing underwater simulation environments in terms of RL compatibility, training efficiency, and standardized benchmarking. MarineGym integrates a proposed GPU-accelerated hydrodynamic plugin based on Isaac Sim, achieving a rollout speed of 250,000 frames per second on a single NVIDIA RTX 3060 GPU. It also provides five models of unmanned underwater vehicles (UUVs), multiple propulsion systems, and a set of predefined tasks covering core underwater control challenges. Additionally, the DR toolkit allows flexible adjustments of simulation and task parameters during training to improve Sim2Real transfer. Further benchmark experiments demonstrate that MarineGym improves training efficiency over existing platforms and supports robust policy adaptation under various perturbations. We expect this platform could drive further advancements in RL research for underwater robotics. For more details about MarineGym and its applications, please visit our project page: this https URL. 

**Abstract (ZH)**: MarineGym：一种用于水下机器人领域的高性能强化学习平台 

---
# Rethinking Bimanual Robotic Manipulation: Learning with Decoupled Interaction Framework 

**Title (ZH)**: 重新思考双臂机器人操作：基于解耦互动框架的学习 

**Authors**: Jian-Jian Jiang, Xiao-Ming Wu, Yi-Xiang He, Ling-An Zeng, Yi-Lin Wei, Dandan Zhang, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.09186)  

**Abstract**: Bimanual robotic manipulation is an emerging and critical topic in the robotics community. Previous works primarily rely on integrated control models that take the perceptions and states of both arms as inputs to directly predict their actions. However, we think bimanual manipulation involves not only coordinated tasks but also various uncoordinated tasks that do not require explicit cooperation during execution, such as grasping objects with the closest hand, which integrated control frameworks ignore to consider due to their enforced cooperation in the early inputs. In this paper, we propose a novel decoupled interaction framework that considers the characteristics of different tasks in bimanual manipulation. The key insight of our framework is to assign an independent model to each arm to enhance the learning of uncoordinated tasks, while introducing a selective interaction module that adaptively learns weights from its own arm to improve the learning of coordinated tasks. Extensive experiments on seven tasks in the RoboTwin dataset demonstrate that: (1) Our framework achieves outstanding performance, with a 23.5% boost over the SOTA method. (2) Our framework is flexible and can be seamlessly integrated into existing methods. (3) Our framework can be effectively extended to multi-agent manipulation tasks, achieving a 28% boost over the integrated control SOTA. (4) The performance boost stems from the decoupled design itself, surpassing the SOTA by 16.5% in success rate with only 1/6 of the model size. 

**Abstract (ZH)**: 双臂机器人操作是机器人领域的一个新兴且关键课题。 

---
# Long-Term Planning Around Humans in Domestic Environments with 3D Scene Graphs 

**Title (ZH)**: 基于3D场景图的家庭环境中长期规划人类活动 

**Authors**: Ermanno Bartoli, Dennis Rotondi, Kai O. Arras, Iolanda Leite  

**Link**: [PDF](https://arxiv.org/pdf/2503.09173)  

**Abstract**: Long-term planning for robots operating in domestic environments poses unique challenges due to the interactions between humans, objects, and spaces. Recent advancements in trajectory planning have leveraged vision-language models (VLMs) to extract contextual information for robots operating in real-world environments. While these methods achieve satisfying performance, they do not explicitly model human activities. Such activities influence surrounding objects and reshape spatial constraints. This paper presents a novel approach to trajectory planning that integrates human preferences, activities, and spatial context through an enriched 3D scene graph (3DSG) representation. By incorporating activity-based relationships, our method captures the spatial impact of human actions, leading to more context-sensitive trajectory adaptation. Preliminary results demonstrate that our approach effectively assigns costs to spaces influenced by human activities, ensuring that the robot trajectory remains contextually appropriate and sensitive to the ongoing environment. This balance between task efficiency and social appropriateness enhances context-aware human-robot interactions in domestic settings. Future work includes implementing a full planning pipeline and conducting user studies to evaluate trajectory acceptability. 

**Abstract (ZH)**: 长期规划家用环境中机器人操作的独特挑战在于人类、物体和空间之间的交互。最近在轨迹规划方面的进展利用了视觉-语言模型（VLMs）来提取机器人在真实环境中操作所需的空间上下文信息。尽管这些方法取得了令人满意的性能，但它们并未明确建模人类活动。这些活动会影响周围的物体并重塑空间约束。本文提出了一种新的轨迹规划方法，通过丰富三维场景图（3DSG）表示整合人类偏好、活动和空间上下文。通过引入基于活动的关系，我们的方法捕捉了人类行为的空间影响，使得轨迹适应更具有上下文相关性。初步结果显示，我们的方法有效地为受人类活动影响的空间分配了成本，确保了机器人的轨迹在上下文中保持适宜性并对当前环境具有敏感性。这种在任务效率和社会适宜性之间的平衡增强了家用环境中的人机交互。未来工作将包括实现完整的规划管道并开展用户研究以评估轨迹的接受度。 

---
# Predictor-Based Time Delay Control of A Hex-Jet Unmanned Aerial Vehicle 

**Title (ZH)**: 基于预测的时间延迟控制六旋翼无人机 

**Authors**: Junning Liang, Haowen Zheng, Yuying Zhang, Yongzhuo Gao, Wei Dong, Ximin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09148)  

**Abstract**: Turbojet-powered VTOL UAVs have garnered increased attention in heavy-load transport and emergency services, due to their superior power density and thrust-to-weight ratio compared to existing electronic propulsion systems. The main challenge with jet-powered UAVs lies in the complexity of thrust vectoring mechanical systems, which aim to mitigate the slow dynamics of the turbojet. In this letter, we introduce a novel turbojet-powered UAV platform named Hex-Jet. Our concept integrates thrust vectoring and differential thrust for comprehensive attitude control. This approach notably simplifies the thrust vectoring mechanism. We utilize a predictor-based time delay control method based on the frequency domain model in our Hex-Jet controller design to mitigate the delay in roll attitude control caused by turbojet dynamics. Our comparative studies provide valuable insights for the UAV community, and flight tests on the scaled prototype demonstrate the successful implementation and verification of the proposed predictor-based time delay control technique. 

**Abstract (ZH)**: 基于涡喷发动机的垂直起降无人机在重型负载运输和紧急服务领域引起了广泛关注，由于其比现有电动推进系统具有更高的功率密度和推重比。喷气动力无人机的主要挑战在于推进矢量机机械系统的复杂性，旨在缓解涡喷发动机的缓慢动态特性。本文介绍了一种新型涡喷发动机动力垂直起降无人机平台Hex-Jet，该平台将推进矢量控制与差动推力集成，全面实现姿态控制。该方法显著简化了推进矢量控制机制。我们使用基于频域模型的预测式时间延迟控制方法，在Hex-Jet控制器设计中减轻了由涡喷发动机动态特性引起的滚转姿态控制的延迟。我们的比较研究为无人机社区提供了有价值的见解，并对缩小比例原型机的飞行测试证明了所提出的预测式时间延迟控制技术的成功实施和验证。 

---
# Tacchi 2.0: A Low Computational Cost and Comprehensive Dynamic Contact Simulator for Vision-based Tactile Sensors 

**Title (ZH)**: Tacchi 2.0：一种低计算成本和综合动态接触模拟器，用于基于视觉的触觉传感器 

**Authors**: Yuhao Sun, Shixin Zhang, Wenzhuang Li, Jie Zhao, Jianhua Shan, Zirong Shen, Zixi Chen, Fuchun Sun, Di Guo, Bin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09100)  

**Abstract**: With the development of robotics technology, some tactile sensors, such as vision-based sensors, have been applied to contact-rich robotics tasks. However, the durability of vision-based tactile sensors significantly increases the cost of tactile information acquisition. Utilizing simulation to generate tactile data has emerged as a reliable approach to address this issue. While data-driven methods for tactile data generation lack robustness, finite element methods (FEM) based approaches require significant computational costs. To address these issues, we integrated a pinhole camera model into the low computational cost vision-based tactile simulator Tacchi that used the Material Point Method (MPM) as the simulated method, completing the simulation of marker motion images. We upgraded Tacchi and introduced Tacchi 2.0. This simulator can simulate tactile images, marked motion images, and joint images under different motion states like pressing, slipping, and rotating. Experimental results demonstrate the reliability of our method and its robustness across various vision-based tactile sensors. 

**Abstract (ZH)**: 随着机器人技术的发展，一些触觉传感器，如基于视觉的传感器，已被应用于接触密集型机器人任务。然而，基于视觉的触觉传感器的耐用性显著增加了触觉信息获取的成本。利用仿真生成触觉数据已成为解决这一问题的可靠方法。虽然基于数据驱动的方法在触觉数据生成中缺乏稳定性，但基于有限元方法（FEM）的方法需要巨大的计算成本。为了解决这些问题，我们将针孔相机模型集成到使用物质点方法（MPM）进行仿真的低计算成本视觉触觉模拟器Tacchi中，完成了标记运动图像的模拟。我们升级了Tacchi并推出了Tacchi 2.0。该模拟器可以在不同的运动状态下（如按压、滑动和旋转）模拟触觉图像、标记运动图像和关节图像。实验结果证明了我们方法的可靠性和在各种视觉触觉传感器上的稳定性。 

---
# Sequential Multi-Object Grasping with One Dexterous Hand 

**Title (ZH)**: 单灵巧手的序列多对象抓取 

**Authors**: Sicheng He, Zeyu Shangguan, Kuanning Wang, Yongchong Gu, Yuqian Fu, Yanwei Fu, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2503.09078)  

**Abstract**: Sequentially grasping multiple objects with multi-fingered hands is common in daily life, where humans can fully leverage the dexterity of their hands to enclose multiple objects. However, the diversity of object geometries and the complex contact interactions required for high-DOF hands to grasp one object while enclosing another make sequential multi-object grasping challenging for robots. In this paper, we propose SeqMultiGrasp, a system for sequentially grasping objects with a four-fingered Allegro Hand. We focus on sequentially grasping two objects, ensuring that the hand fully encloses one object before lifting it and then grasps the second object without dropping the first. Our system first synthesizes single-object grasp candidates, where each grasp is constrained to use only a subset of the hand's links. These grasps are then validated in a physics simulator to ensure stability and feasibility. Next, we merge the validated single-object grasp poses to construct multi-object grasp configurations. For real-world deployment, we train a diffusion model conditioned on point clouds to propose grasp poses, followed by a heuristic-based execution strategy. We test our system using $8 \times 8$ object combinations in simulation and $6 \times 3$ object combinations in real. Our diffusion-based grasp model obtains an average success rate of 65.8% over 1600 simulation trials and 56.7% over 90 real-world trials, suggesting that it is a promising approach for sequential multi-object grasping with multi-fingered hands. Supplementary material is available on our project website: this https URL. 

**Abstract (ZH)**: 基于四指 Allegro 手的序列多物 grasping 系统：SeqMultiGrasp 

---
# ManeuverGPT Agentic Control for Safe Autonomous Stunt Maneuvers 

**Title (ZH)**: ManeuverGPT自主特技机动的代理控制以确保安全 

**Authors**: Shawn Azdam, Pranav Doma, Aliasghar Moj Arab  

**Link**: [PDF](https://arxiv.org/pdf/2503.09035)  

**Abstract**: The next generation of active safety features in autonomous vehicles should be capable of safely executing evasive hazard-avoidance maneuvers akin to those performed by professional stunt drivers to achieve high-agility motion at the limits of vehicle handling. This paper presents a novel framework, ManeuverGPT, for generating and executing high-dynamic stunt maneuvers in autonomous vehicles using large language model (LLM)-based agents as controllers. We target aggressive maneuvers, such as J-turns, within the CARLA simulation environment and demonstrate an iterative, prompt-based approach to refine vehicle control parameters, starting tabula rasa without retraining model weights. We propose an agentic architecture comprised of three specialized agents (1) a Query Enricher Agent for contextualizing user commands, (2) a Driver Agent for generating maneuver parameters, and (3) a Parameter Validator Agent that enforces physics-based and safety constraints. Experimental results demonstrate successful J-turn execution across multiple vehicle models through textual prompts that adapt to differing vehicle dynamics. We evaluate performance via established success criteria and discuss limitations regarding numeric precision and scenario complexity. Our findings underscore the potential of LLM-driven control for flexible, high-dynamic maneuvers, while highlighting the importance of hybrid approaches that combine language-based reasoning with algorithmic validation. 

**Abstract (ZH)**: 下一代自动驾驶车辆的主动安全功能应能够安全执行类似专业特技驾驶员在车辆 handling 极限下实现高敏捷运动的避险机动动作。本文提出了一种新型框架 ManeuverGPT，使用基于大型语言模型的智能体作为控制器，以生成并执行自动驾驶车辆中的高动态特技机动。我们针对 CARLA 模拟环境中的激进机动，如 J-turn，采用迭代提示方法逐步调整车辆控制参数，无需重新训练模型权重。我们提出了一种智能体架构，包括三个专门的智能体：（1）查询增强智能体，用于上下文化用户命令，（2）驾驶员智能体，用于生成机动参数，以及（3）参数验证智能体，用于强制执行基于物理和安全的约束。实验结果表明，通过适应不同车辆动力学的文本提示成功执行了多种车型的 J-turn。我们通过既定的成功标准评估性能，并讨论了在数值精度和场景复杂性方面的局限性。我们的研究结果突显了基于大型语言模型控制在灵活、高动态机动中的潜力，同时强调了结合基于语言的推理与算法验证的混合方法的重要性。 

---
# RFUAV: A Benchmark Dataset for Unmanned Aerial Vehicle Detection and Identification 

**Title (ZH)**: RFUAV：无人机检测与识别基准数据集 

**Authors**: Rui Shi, Xiaodong Yu, Shengming Wang, Yijia Zhang, Lu Xu, Peng Pan, Chunlai Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.09033)  

**Abstract**: In this paper, we propose RFUAV as a new benchmark dataset for radio-frequency based (RF-based) unmanned aerial vehicle (UAV) identification and address the following challenges: Firstly, many existing datasets feature a restricted variety of drone types and insufficient volumes of raw data, which fail to meet the demands of practical applications. Secondly, existing datasets often lack raw data covering a broad range of signal-to-noise ratios (SNR), or do not provide tools for transforming raw data to different SNR levels. This limitation undermines the validity of model training and evaluation. Lastly, many existing datasets do not offer open-access evaluation tools, leading to a lack of unified evaluation standards in current research within this field. RFUAV comprises approximately 1.3 TB of raw frequency data collected from 37 distinct UAVs using the Universal Software Radio Peripheral (USRP) device in real-world environments. Through in-depth analysis of the RF data in RFUAV, we define a drone feature sequence called RF drone fingerprint, which aids in distinguishing drone signals. In addition to the dataset, RFUAV provides a baseline preprocessing method and model evaluation tools. Rigorous experiments demonstrate that these preprocessing methods achieve state-of-the-art (SOTA) performance using the provided evaluation tools. The RFUAV dataset and baseline implementation are publicly available at this https URL. 

**Abstract (ZH)**: 本文提出RFUAV作为基于射频的无人机识别的新基准数据集，并解决了以下挑战：首先，许多现有数据集的无人机类型有限且原始数据量不足，无法满足实际应用的需求。其次，现有数据集往往缺乏覆盖广泛信噪比（SNR）范围的原始数据，或未提供将原始数据转换为不同SNR水平的工具，这限制了模型训练和评估的有效性。最后，许多现有数据集未提供开源评估工具，导致当前研究领域缺乏统一的评价标准。RFUAV包含约1.3 TB由37架不同无人机在实际环境中使用通用软件射频外围设备（USRP）收集的原始射频频段数据。通过对RFUAV中的射频数据进行深入分析，定义了一种无人机特征序列，称为射频无人机指纹，有助于区分无人机信号。除了数据集外，RFUAV还提供了基线预处理方法和模型评估工具。严格实验表明，这些预处理方法在提供的评估工具下实现了当前最佳性能（SOTA）。RFUAV数据集和基线实现可在https://... 公开获取。 

---
# Traffic Regulation-aware Path Planning with Regulation Databases and Vision-Language Models 

**Title (ZH)**: 基于交通管制数据库和视觉语言模型的路径规划与交通管制意识规划 

**Authors**: Xu Han, Zhiwen Wu, Xin Xia, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.09024)  

**Abstract**: This paper introduces and tests a framework integrating traffic regulation compliance into automated driving systems (ADS). The framework enables ADS to follow traffic laws and make informed decisions based on the driving environment. Using RGB camera inputs and a vision-language model (VLM), the system generates descriptive text to support a regulation-aware decision-making process, ensuring legal and safe driving practices. This information is combined with a machine-readable ADS regulation database to guide future driving plans within legal constraints. Key features include: 1) a regulation database supporting ADS decision-making, 2) an automated process using sensor input for regulation-aware path planning, and 3) validation in both simulated and real-world environments. Particularly, the real-world vehicle tests not only assess the framework's performance but also evaluate the potential and challenges of VLMs to solve complex driving problems by integrating detection, reasoning, and planning. This work enhances the legality, safety, and public trust in ADS, representing a significant step forward in the field. 

**Abstract (ZH)**: 本文介绍并测试了一种将交通法规遵守整合到自动驾驶系统（ADS）中的框架。该框架使ADS能够遵循交通法规并基于驾驶环境做出明智的决策。利用RGB摄像头输入和视觉语言模型（VLM），系统生成描述性文本以支持法规意识决策过程，确保合法和安全的驾驶实践。这些信息与可机器读取的ADS法规数据库相结合，以在法律约束内指导未来的驾驶计划。关键功能包括：1）支持ADS决策的法规数据库，2）基于传感器输入的自动化过程进行法规意识路径规划，3）在模拟和实际环境中进行验证。特别地，实际车辆测试不仅评估了该框架的性能，还评估了VLMs通过集成检测、推理和规划来解决复杂驾驶问题的潜力和挑战。本文增强了ADS的合法性、安全性和公众信任，代表了该领域的一项重要进展。 

---
# Feasibility-aware Imitation Learning from Observations through a Hand-mounted Demonstration Interface 

**Title (ZH)**: 基于手部穿戴示范接口的观察模仿学习的可行性意识化研究 

**Authors**: Kei Takahashi, Hikaru Sasaki, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2503.09018)  

**Abstract**: Imitation learning through a demonstration interface is expected to learn policies for robot automation from intuitive human demonstrations. However, due to the differences in human and robot movement characteristics, a human expert might unintentionally demonstrate an action that the robot cannot execute. We propose feasibility-aware behavior cloning from observation (FABCO). In the FABCO framework, the feasibility of each demonstration is assessed using the robot's pre-trained forward and inverse dynamics models. This feasibility information is provided as visual feedback to the demonstrators, encouraging them to refine their demonstrations. During policy learning, estimated feasibility serves as a weight for the demonstration data, improving both the data efficiency and the robustness of the learned policy. We experimentally validated FABCO's effectiveness by applying it to a pipette insertion task involving a pipette and a vial. Four participants assessed the impact of the feasibility feedback and the weighted policy learning in FABCO. Additionally, we used the NASA Task Load Index (NASA-TLX) to evaluate the workload induced by demonstrations with visual feedback. 

**Abstract (ZH)**: 通过演示界面进行模仿学习有望从直观的人类演示中学习用于机器人自动化的策略。然而，由于人类和机器人运动特征的差异，人类专家可能会无意中演示机器人无法执行的动作。我们提出了基于观察的可行性感知行为克隆（FABCO）。在FABCO框架中，使用机器人预训练的正向和逆向动力学模型来评估每项演示的可行性。该可行性信息作为视觉反馈提供给演示者，鼓励他们改进演示。在策略学习过程中，估计的可行性作为演示数据的权重，提高了学习策略的数据效率和鲁棒性。我们通过将FABCO应用于涉及移液管和瓶的移液任务，实验验证了其有效性。四位参与者评估了可行性反馈和带有加权策略学习的FABCO的影响。此外，我们使用NASA任务负荷指数（NASA-TLX）评估了带有视觉反馈的演示引起的负荷。 

---
# Natural Humanoid Robot Locomotion with Generative Motion Prior 

**Title (ZH)**: 自然 humanoid 机器人运动的生成运动先验 

**Authors**: Haodong Zhang, Liang Zhang, Zhenghan Chen, Lu Chen, Yue Wang, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.09015)  

**Abstract**: Natural and lifelike locomotion remains a fundamental challenge for humanoid robots to interact with human society. However, previous methods either neglect motion naturalness or rely on unstable and ambiguous style rewards. In this paper, we propose a novel Generative Motion Prior (GMP) that provides fine-grained motion-level supervision for the task of natural humanoid robot locomotion. To leverage natural human motions, we first employ whole-body motion retargeting to effectively transfer them to the robot. Subsequently, we train a generative model offline to predict future natural reference motions for the robot based on a conditional variational auto-encoder. During policy training, the generative motion prior serves as a frozen online motion generator, delivering precise and comprehensive supervision at the trajectory level, including joint angles and keypoint positions. The generative motion prior significantly enhances training stability and improves interpretability by offering detailed and dense guidance throughout the learning process. Experimental results in both simulation and real-world environments demonstrate that our method achieves superior motion naturalness compared to existing approaches. Project page can be found at this https URL 

**Abstract (ZH)**: 自然且逼真的运动对于人形机器人与人类社会互动仍然是一个基本挑战。然而，先前的方法要么忽视运动的自然性，要么依赖于不穩定和含糊的风格奖励。本文 propose 一种新颖的生成运动先验（GMP），为自然人形机器人运动提供细粒度的动力学级监督。为了利用自然的人运动，我们首先使用全身运动重定向有效地将人运动转移到机器人上。随后，我们训练一个生成模型，在条件变分自编码器的基础上预测机器人未来自然的参考运动。在策略训练过程中，生成运动先验作为冻结的在线运动生成器，提供轨迹级的精确且全面的监督，包括关节角度和关键点位置。生成运动先验在整个学习过程中提供了详细且密集的指导，显著增强了训练的稳定性并提高了可解释性。实验结果在仿真和真实环境中均表明，我们的方法在运动自然性方面优于现有方法。更多信息请参见项目页面：this https URL。 

---
# HumanoidPano: Hybrid Spherical Panoramic-LiDAR Cross-Modal Perception for Humanoid Robots 

**Title (ZH)**: 类人全景：混合球面全景-LiDAR 跨模态感知技术用于类人机器人 

**Authors**: Qiang Zhang, Zhang Zhang, Wei Cui, Jingkai Sun, Jiahang Cao, Yijie Guo, Gang Han, Wen Zhao, Jiaxu Wang, Chenghao Sun, Lingfeng Zhang, Hao Cheng, Yujie Chen, Lin Wang, Jian Tang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09010)  

**Abstract**: The perceptual system design for humanoid robots poses unique challenges due to inherent structural constraints that cause severe self-occlusion and limited field-of-view (FOV). We present HumanoidPano, a novel hybrid cross-modal perception framework that synergistically integrates panoramic vision and LiDAR sensing to overcome these limitations. Unlike conventional robot perception systems that rely on monocular cameras or standard multi-sensor configurations, our method establishes geometrically-aware modality alignment through a spherical vision transformer, enabling seamless fusion of 360 visual context with LiDAR's precise depth measurements. First, Spherical Geometry-aware Constraints (SGC) leverage panoramic camera ray properties to guide distortion-regularized sampling offsets for geometric alignment. Second, Spatial Deformable Attention (SDA) aggregates hierarchical 3D features via spherical offsets, enabling efficient 360°-to-BEV fusion with geometrically complete object representations. Third, Panoramic Augmentation (AUG) combines cross-view transformations and semantic alignment to enhance BEV-panoramic feature consistency during data augmentation. Extensive evaluations demonstrate state-of-the-art performance on the 360BEV-Matterport benchmark. Real-world deployment on humanoid platforms validates the system's capability to generate accurate BEV segmentation maps through panoramic-LiDAR co-perception, directly enabling downstream navigation tasks in complex environments. Our work establishes a new paradigm for embodied perception in humanoid robotics. 

**Abstract (ZH)**: 基于人体形机器人的知觉系统设计由于固有的结构约束导致严重的自遮挡和有限的视野。我们提出了HumanoidPano，这是一种新颖的混合跨模态感知框架，结合了全景视觉和LiDAR感应以克服这些限制。 

---
# Unified Locomotion Transformer with Simultaneous Sim-to-Real Transfer for Quadrupeds 

**Title (ZH)**: 四足机器人统一运动变压器的同步从仿真到现实的迁移 

**Authors**: Dikai Liu, Tianwei Zhang, Jianxiong Yin, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2503.08997)  

**Abstract**: Quadrupeds have gained rapid advancement in their capability of traversing across complex terrains. The adoption of deep Reinforcement Learning (RL), transformers and various knowledge transfer techniques can greatly reduce the sim-to-real gap. However, the classical teacher-student framework commonly used in existing locomotion policies requires a pre-trained teacher and leverages the privilege information to guide the student policy. With the implementation of large-scale models in robotics controllers, especially transformers-based ones, this knowledge distillation technique starts to show its weakness in efficiency, due to the requirement of multiple supervised stages. In this paper, we propose Unified Locomotion Transformer (ULT), a new transformer-based framework to unify the processes of knowledge transfer and policy optimization in a single network while still taking advantage of privilege information. The policies are optimized with reinforcement learning, next state-action prediction, and action imitation, all in just one training stage, to achieve zero-shot deployment. Evaluation results demonstrate that with ULT, optimal teacher and student policies can be obtained at the same time, greatly easing the difficulty in knowledge transfer, even with complex transformer-based models. 

**Abstract (ZH)**: 四足机器人在复杂地形穿越能力上取得了 rapid advancement。采用深度强化学习（RL）、变压器和各种知识迁移技术可以大幅缩小模拟与现实之间的差距。然而，现有运动策略中常用的经典教师-学生框架需要预先训练教师，并利用特权信息来指导学生策略。随着大规模模型在机器人控制器中的实施，特别是基于变压器的模型，这种知识蒸馏技术开始在效率方面显示其弱点，因为需要多个监督阶段。在本文中，我们提出了一种统一的运动变压器（ULT）框架，该框架在一个网络中统一了知识迁移和策略优化的过程，同时仍然利用特权信息。通过强化学习、下一步状态动作预测和动作模仿优化策略，仅在一个训练阶段即可实现零样本部署。评估结果表明，通过ULT，可以同时获得最优的教师和学生策略，大大简化了知识迁移的难度，即使对于复杂的基于变压器的模型也是如此。 

---
# TetraGrip: Sensor-Driven Multi-Suction Reactive Object Manipulation in Cluttered Scenes 

**Title (ZH)**: TetraGrip: 基于传感器的多吸附反应式物体 manipulation 在杂乱场景中的应用 

**Authors**: Paolo Torrado, Joshua Levin, Markus Grotz, Joshua Smith  

**Link**: [PDF](https://arxiv.org/pdf/2503.08978)  

**Abstract**: Warehouse robotic systems equipped with vacuum grippers must reliably grasp a diverse range of objects from densely packed shelves. However, these environments present significant challenges, including occlusions, diverse object orientations, stacked and obstructed items, and surfaces that are difficult to suction. We introduce \tetra, a novel vacuum-based grasping strategy featuring four suction cups mounted on linear actuators. Each actuator is equipped with an optical time-of-flight (ToF) proximity sensor, enabling reactive grasping.
We evaluate \tetra in a warehouse-style setting, demonstrating its ability to manipulate objects in stacked and obstructed configurations. Our results show that our RL-based policy improves picking success in stacked-object scenarios by 22.86\% compared to a single-suction gripper. Additionally, we demonstrate that TetraGrip can successfully grasp objects in scenarios where a single-suction gripper fails due to physical limitations, specifically in two cases: (1) picking an object occluded by another object and (2) retrieving an object in a complex scenario. These findings highlight the advantages of multi-actuated, suction-based grasping in unstructured warehouse environments. The project website is available at: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 装备有真空吸盘的仓库机器人系统必须可靠地从密集排列的货架上抓取多种多样的物体。然而，这些环境带来了显著的挑战，包括遮挡、多样的物体朝向、堆叠和阻挡的物品，以及难以吸盘吸附的表面。我们引入了\tetra，一种基于真空的新型抓取策略，采用线性执行器上安装的四个吸盘。每个执行器配备了一个光学飞行时间（ToF）距离传感器，使得抓取具有反应性。 

---
# FP3: A 3D Foundation Policy for Robotic Manipulation 

**Title (ZH)**: FP3: 一种用于机器人操作的3D基础策略 

**Authors**: Rujia Yang, Geng Chen, Chuan Wen, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08950)  

**Abstract**: Following its success in natural language processing and computer vision, foundation models that are pre-trained on large-scale multi-task datasets have also shown great potential in robotics. However, most existing robot foundation models rely solely on 2D image observations, ignoring 3D geometric information, which is essential for robots to perceive and reason about the 3D world. In this paper, we introduce FP3, a first large-scale 3D foundation policy model for robotic manipulation. FP3 builds on a scalable diffusion transformer architecture and is pre-trained on 60k trajectories with point cloud observations. With the model design and diverse pre-training data, FP3 can be efficiently fine-tuned for downstream tasks while exhibiting strong generalization capabilities. Experiments on real robots demonstrate that with only 80 demonstrations, FP3 is able to learn a new task with over 90% success rates in novel environments with unseen objects, significantly surpassing existing robot foundation models. 

**Abstract (ZH)**: 随着其在自然语言处理和计算机视觉领域的成功，预训练于大规模多任务数据集的基座模型在机器人领域也显示出了巨大潜力。然而，现有的大多数机器人基座模型仅依赖于2D图像观测，忽视了对于机器人感知和推理三维世界至关重要的3D几何信息。本文介绍了一种新的大规模3D基座策略模型FP3，它基于可扩展的扩散变换器架构，并在包含点云观测的60,000条轨迹上进行预训练。通过模型设计和多样化的预训练数据，FP3能够高效地微调以适应下游任务，同时表现出强大的泛化能力。实验证实在真实机器人上的结果显示，仅需80个演示，FP3便能在未见物体的新环境中以超过90%的成功率学习新任务，显著优于现有机器人基座模型。 

---
# Mutual Adaptation in Human-Robot Co-Transportation with Human Preference Uncertainty 

**Title (ZH)**: 基于人类偏好不确定性的人机协同运输适配机制 

**Authors**: Al Jaber Mahmud, Weizi Li, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08895)  

**Abstract**: Mutual adaptation can significantly enhance overall task performance in human-robot co-transportation by integrating both the robot's and human's understanding of the environment. While human modeling helps capture humans' subjective preferences, two challenges persist: (i) the uncertainty of human preference parameters and (ii) the need to balance adaptation strategies that benefit both humans and robots. In this paper, we propose a unified framework to address these challenges and improve task performance through mutual adaptation. First, instead of relying on fixed parameters, we model a probability distribution of human choices by incorporating a range of uncertain human parameters. Next, we introduce a time-varying stubbornness measure and a coordination mode transition model, which allows either the robot to lead the team's trajectory or, if a human's preferred path conflicts with the robot's plan and their stubbornness exceeds a threshold, the robot to transition to following the human. Finally, we introduce a pose optimization strategy to mitigate the uncertain human behaviors when they are leading. To validate the framework, we design and perform experiments with real human feedback. We then demonstrate, through simulations, the effectiveness of our models in enhancing task performance with mutual adaptation and pose optimization. 

**Abstract (ZH)**: 人类与机器人共运送任务中的相互适应可显著提升整体任务性能：一种统一框架及其实验验证 

---
# SICNav-Diffusion: Safe and Interactive Crowd Navigation with Diffusion Trajectory Predictions 

**Title (ZH)**: SICNav-扩散：基于扩散轨迹预测的安全交互式人群导航 

**Authors**: Sepehr Samavi, Anthony Lem, Fumiaki Sato, Sirui Chen, Qiao Gu, Keijiro Yano, Angela P. Schoellig, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2503.08858)  

**Abstract**: To navigate crowds without collisions, robots must interact with humans by forecasting their future motion and reacting accordingly. While learning-based prediction models have shown success in generating likely human trajectory predictions, integrating these stochastic models into a robot controller presents several challenges. The controller needs to account for interactive coupling between planned robot motion and human predictions while ensuring both predictions and robot actions are safe (i.e. collision-free). To address these challenges, we present a receding horizon crowd navigation method for single-robot multi-human environments. We first propose a diffusion model to generate joint trajectory predictions for all humans in the scene. We then incorporate these multi-modal predictions into a SICNav Bilevel MPC problem that simultaneously solves for a robot plan (upper-level) and acts as a safety filter to refine the predictions for non-collision (lower-level). Combining planning and prediction refinement into one bilevel problem ensures that the robot plan and human predictions are coupled. We validate the open-loop trajectory prediction performance of our diffusion model on the commonly used ETH/UCY benchmark and evaluate the closed-loop performance of our robot navigation method in simulation and extensive real-robot experiments demonstrating safe, efficient, and reactive robot motion. 

**Abstract (ZH)**: 基于扩散模型的单机器人多人群体导航方法 

---
# Geometric Data-Driven Multi-Jet Locomotion Inspired by Salps 

**Title (ZH)**: 几何数据驱动的多喷流运动受沙虱启发 

**Authors**: Yanhao Yang, Nina L. Hecht, Yousef Salaman-Maclara, Nathan Justus, Zachary A. Thomas, Farhan Rozaidi, Ross L. Hatton  

**Link**: [PDF](https://arxiv.org/pdf/2503.08817)  

**Abstract**: Salps are marine animals consisting of chains of jellyfish-like units. Their capacity for effective underwater undulatory locomotion through coordinating multi-jet propulsion has aroused significant interest in the field of robotics and inspired extensive research including design, modeling, and control. In this paper, we conduct a comprehensive analysis of the locomotion of salp-like systems using the robotic platform "LandSalp" based on geometric mechanics, including mechanism design, dynamic modeling, system identification, and motion planning and control. Our work takes a step toward a better understanding of salps' underwater locomotion and provides a clear path for extending these insights to more complex and capable underwater robotic systems. Furthermore, this study illustrates the effectiveness of geometric mechanics in bio-inspired robots for efficient data-driven locomotion modeling, demonstrated by learning the dynamics of LandSalp from only 3 minutes of experimental data. Lastly, we extend the geometric mechanics principles to multi-jet propulsion systems with stability considerations and validate the theory through experiments on the LandSalp hardware. 

**Abstract (ZH)**: 珊懑是由一系列类似水母的单元组成的marine动物，它们通过协调多喷射推进实现有效的水下波浪式运动，这一特性在机器人学领域引起了广泛关注，并激发了大量关于设计、建模和控制的研究。在本文中，我们基于几何力学对“LandSalp”机器人平台上的类似珊懑系统的运动进行了全面分析，包括机构设计、动态建模、系统辨识以及运动规划与控制。我们的工作为进一步理解珊懑的水下运动提供了新的见解，并为将这些见解扩展到更加复杂和高性能的水下机器人系统指明了路径。此外，本研究展示了几何力学在生物启发机器人中实现高效数据驱动运动建模的有效性，仅通过3分钟的实验数据就学习到了LandSalp的动力学特性。最后，我们将几何力学原理扩展到具有稳定性考虑的多喷射推进系统，并通过在LandSalp硬件上的实验验证了理论。 

---
# SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment 

**Title (ZH)**: SimLingo: 仅视觉闭环自主驾驶与语言-动作对齐 

**Authors**: Katrin Renz, Long Chen, Elahe Arani, Oleg Sinavski  

**Link**: [PDF](https://arxiv.org/pdf/2503.09594)  

**Abstract**: Integrating large language models (LLMs) into autonomous driving has attracted significant attention with the hope of improving generalization and explainability. However, existing methods often focus on either driving or vision-language understanding but achieving both high driving performance and extensive language understanding remains challenging. In addition, the dominant approach to tackle vision-language understanding is using visual question answering. However, for autonomous driving, this is only useful if it is aligned with the action space. Otherwise, the model's answers could be inconsistent with its behavior. Therefore, we propose a model that can handle three different tasks: (1) closed-loop driving, (2) vision-language understanding, and (3) language-action alignment. Our model SimLingo is based on a vision language model (VLM) and works using only camera, excluding expensive sensors like LiDAR. SimLingo obtains state-of-the-art performance on the widely used CARLA simulator on the Bench2Drive benchmark and is the winning entry at the CARLA challenge 2024. Additionally, we achieve strong results in a wide variety of language-related tasks while maintaining high driving performance. 

**Abstract (ZH)**: 将大型语言模型（LLMs）集成到自动驾驶中以提高泛化能力和解释性吸引了大量关注，但现有方法往往侧重于自动驾驶或视觉语言理解，同时实现高驾驶性能和广泛的语言理解仍然具有挑战性。此外，应对视觉语言理解的主要方法是使用视觉问答。然而，对于自动驾驶而言，只有当视觉问答与行为空间对齐时，这种方法才是有用的。否则，模型的回答可能与其行为不一致。因此，我们提出一个可以处理三个不同任务的模型：（1）闭环自动驾驶，（2）视觉语言理解，以及（3）语言-行为对齐。我们的模型SimLingo基于视觉语言模型（VLM），仅使用摄像头而无需昂贵的传感器如LiDAR。SimLingo在广泛应用的CARLA模拟器上的Bench2Drive基准测试中获得了最先进的性能，并在2024年CARLA挑战赛中获得冠军。此外，我们还在多种语言相关任务中取得了优异的结果，同时保持了高驾驶性能。 

---
# Online Language Splatting 

**Title (ZH)**: 在线语言斑图化 

**Authors**: Saimouli Katragadda, Cho-Ying Wu, Yuliang Guo, Xinyu Huang, Guoquan Huang, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.09447)  

**Abstract**: To enable AI agents to interact seamlessly with both humans and 3D environments, they must not only perceive the 3D world accurately but also align human language with 3D spatial representations. While prior work has made significant progress by integrating language features into geometrically detailed 3D scene representations using 3D Gaussian Splatting (GS), these approaches rely on computationally intensive offline preprocessing of language features for each input image, limiting adaptability to new environments. In this work, we introduce Online Language Splatting, the first framework to achieve online, near real-time, open-vocabulary language mapping within a 3DGS-SLAM system without requiring pre-generated language features. The key challenge lies in efficiently fusing high-dimensional language features into 3D representations while balancing the computation speed, memory usage, rendering quality and open-vocabulary capability. To this end, we innovatively design: (1) a high-resolution CLIP embedding module capable of generating detailed language feature maps in 18ms per frame, (2) a two-stage online auto-encoder that compresses 768-dimensional CLIP features to 15 dimensions while preserving open-vocabulary capabilities, and (3) a color-language disentangled optimization approach to improve rendering quality. Experimental results show that our online method not only surpasses the state-of-the-art offline methods in accuracy but also achieves more than 40x efficiency boost, demonstrating the potential for dynamic and interactive AI applications. 

**Abstract (ZH)**: 在线语言平滑：一种无需先验语言特征的3DGS-SLAM系统中实现即时开放词汇语言映射的框架 

---
# PCLA: A Framework for Testing Autonomous Agents in the CARLA Simulator 

**Title (ZH)**: PCLA: 在CARLA模拟器中测试自主代理的框架 

**Authors**: Masoud Jamshidiyan Tehrani, Jinhan Kim, Paolo Tonella  

**Link**: [PDF](https://arxiv.org/pdf/2503.09385)  

**Abstract**: Recent research on testing autonomous driving agents has grown significantly, especially in simulation environments. The CARLA simulator is often the preferred choice, and the autonomous agents from the CARLA Leaderboard challenge are regarded as the best-performing agents within this environment. However, researchers who test these agents, rather than training their own ones from scratch, often face challenges in utilizing them within customized test environments and scenarios. To address these challenges, we introduce PCLA (Pretrained CARLA Leaderboard Agents), an open-source Python testing framework that includes nine high-performing pre-trained autonomous agents from the Leaderboard challenges. PCLA is the first infrastructure specifically designed for testing various autonomous agents in arbitrary CARLA environments/scenarios. PCLA provides a simple way to deploy Leaderboard agents onto a vehicle without relying on the Leaderboard codebase, it allows researchers to easily switch between agents without requiring modifications to CARLA versions or programming environments, and it is fully compatible with the latest version of CARLA while remaining independent of the Leaderboard's specific CARLA version. PCLA is publicly accessible at this https URL. 

**Abstract (ZH)**: Recent研究中对自动驾驶代理的测试显著增长，尤其是在仿真环境中。CARLA模拟器通常是首选，来自CARLA Leaderboard挑战的自主代理被认为是该环境内性能最好的代理。然而，测试这些代理而非从头开始训练自己的代理的研究人员往往会在将它们用于定制的测试环境和场景中遇到挑战。为了解决这些挑战，我们引入了PCLA（预训练CARLA Leaderboard代理），这是一个开源的Python测试框架，其中包括了九个来自Leaderboard挑战的高性能预训练自主代理。PCLA是专门为在任意CARLA环境/场景中测试各种自主代理设计的第一个基础设施。PCLA提供了将Leaderboard代理部署到车辆上的一种简单方式，无需依赖Leaderboard代码库，它允许研究人员在不修改CARLA版本或编程环境的情况下轻松切换代理，并且PCLA与最新版本的CARLA完全兼容，同时保持独立于Leaderboard的特定CARLA版本。PCLA可在以下网址公开访问：https://example.com。 

---
# 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos 

**Title (ZH)**: 2HandedAfforder: 从人类视频中学习精确可执行的双手 affordances 

**Authors**: Marvin Heidinger, Snehal Jauhri, Vignesh Prasad, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.09320)  

**Abstract**: When interacting with objects, humans effectively reason about which regions of objects are viable for an intended action, i.e., the affordance regions of the object. They can also account for subtle differences in object regions based on the task to be performed and whether one or two hands need to be used. However, current vision-based affordance prediction methods often reduce the problem to naive object part segmentation. In this work, we propose a framework for extracting affordance data from human activity video datasets. Our extracted 2HANDS dataset contains precise object affordance region segmentations and affordance class-labels as narrations of the activity performed. The data also accounts for bimanual actions, i.e., two hands co-ordinating and interacting with one or more objects. We present a VLM-based affordance prediction model, 2HandedAfforder, trained on the dataset and demonstrate superior performance over baselines in affordance region segmentation for various activities. Finally, we show that our predicted affordance regions are actionable, i.e., can be used by an agent performing a task, through demonstration in robotic manipulation scenarios. 

**Abstract (ZH)**: 基于视觉语言模型的双手法具预测框架：从人类活动视频中提取手部互动区域标注 

---
# Learning Appearance and Motion Cues for Panoptic Tracking 

**Title (ZH)**: 学习外观和运动线索进行全景跟踪 

**Authors**: Juana Valeria Hurtado, Sajad Marvi, Rohit Mohan, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2503.09191)  

**Abstract**: Panoptic tracking enables pixel-level scene interpretation of videos by integrating instance tracking in panoptic segmentation. This provides robots with a spatio-temporal understanding of the environment, an essential attribute for their operation in dynamic environments. In this paper, we propose a novel approach for panoptic tracking that simultaneously captures general semantic information and instance-specific appearance and motion features. Unlike existing methods that overlook dynamic scene attributes, our approach leverages both appearance and motion cues through dedicated network heads. These interconnected heads employ multi-scale deformable convolutions that reason about scene motion offsets with semantic context and motion-enhanced appearance features to learn tracking embeddings. Furthermore, we introduce a novel two-step fusion module that integrates the outputs from both heads by first matching instances from the current time step with propagated instances from previous time steps and subsequently refines associations using motion-enhanced appearance embeddings, improving robustness in challenging scenarios. Extensive evaluations of our proposed \netname model on two benchmark datasets demonstrate that it achieves state-of-the-art performance in panoptic tracking accuracy, surpassing prior methods in maintaining object identities over time. To facilitate future research, we make the code available at this http URL 

**Abstract (ZH)**: 全景跟踪通过结合全景分割中的实例跟踪，实现视频的像素级场景解释。这为机器人提供了时空环境理解能力，是其在动态环境中操作的关键属性。在本文中，我们提出了一种新颖的全景跟踪方法，同时捕捉通用语义信息和实例特定的外观和运动特征。与现有方法忽略动态场景属性不同，我们的方法通过专用网络头利用外观和运动线索。这些相互连接的头使用多尺度可变形卷积，结合语义上下文和运动增强的外观特征来学习跟踪嵌入。此外，我们引入了一种新的两步融合模块，首先通过匹配当前时间步的实例与之前时间步传播的实例，然后使用运动增强的外观嵌入进一步细化关联，增强在挑战性场景中的鲁棒性。我们在两个基准数据集上的广泛评估表明，我们的\netname模型在全景跟踪准确性上达到了最先进的性能，优于先前方法在保持对象身份方面的表现。为了促进未来研究，我们已在此网址处提供了代码。 

---
# Motion Blender Gaussian Splatting for Dynamic Reconstruction 

**Title (ZH)**: 动态重建中的运动混合高斯绘制 

**Authors**: Xinyu Zhang, Haonan Chang, Yuhan Liu, Abdeslam Boularias  

**Link**: [PDF](https://arxiv.org/pdf/2503.09040)  

**Abstract**: Gaussian splatting has emerged as a powerful tool for high-fidelity reconstruction of dynamic scenes. However, existing methods primarily rely on implicit motion representations, such as encoding motions into neural networks or per-Gaussian parameters, which makes it difficult to further manipulate the reconstructed motions. This lack of explicit controllability limits existing methods to replaying recorded motions only, which hinders a wider application. To address this, we propose Motion Blender Gaussian Splatting (MB-GS), a novel framework that uses motion graph as an explicit and sparse motion representation. The motion of graph links is propagated to individual Gaussians via dual quaternion skinning, with learnable weight painting functions determining the influence of each link. The motion graphs and 3D Gaussians are jointly optimized from input videos via differentiable rendering. Experiments show that MB-GS achieves state-of-the-art performance on the iPhone dataset while being competitive on HyperNeRF. Additionally, we demonstrate the application potential of our method in generating novel object motions and robot demonstrations through motion editing. Video demonstrations can be found at this https URL. 

**Abstract (ZH)**: 高保真动态场景重建的一种强大工具：高斯点积已 emerge 作为一种强大的工具，用于高保真重建动态场景。然而，现有方法主要依赖于隐式运动表示，如将运动编码到神经网络或每个高斯参数中，这使得难以进一步操控重建的运动。这种缺乏显式可控性限制了现有方法只能回放录制的运动，这妨碍了其更广泛的应用。为了解决这一问题，我们提出了运动混合高斯点积（Motion Blender Gaussian Splatting，MB-GS）这一新颖框架，该框架使用运动图作为显式且稀疏的运动表示。通过双四元数皮肤技术，运动图边的运动传播到个体高斯点上，可学习的权重绘画函数确定每个边的影响。通过可微渲染，运动图和3D高斯点共同从输入视频中进行优化。实验结果显示，MB-GS 在 iPhone 数据集上达到了最先进的性能，同时在 HyperNeRF 上竞争力强。此外，我们展示了该方法在通过运动编辑生成新型物体运动和机器人演示方面的应用潜力。相关视频演示可访问以下链接：this https URL。 

---
# Accurate Control under Voltage Drop for Rotor Drones 

**Title (ZH)**: 准确控制下的电压降对旋翼无人机的影响 

**Authors**: Yuhang Liu, Jindou Jia, Zihan Yang, Kexin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.09017)  

**Abstract**: This letter proposes an anti-disturbance control scheme for rotor drones to counteract voltage drop (VD) disturbance caused by voltage drop of the battery, which is a common case for long-time flight or aggressive maneuvers. Firstly, the refined dynamics of rotor drones considering VD disturbance are presented. Based on the dynamics, a voltage drop observer (VDO) is developed to accurately estimate the VD disturbance by decoupling the disturbance and state information of the drone, reducing the conservativeness of conventional disturbance observers. Subsequently, the control scheme integrates the VDO within the translational loop and a fixed-time sliding mode observer (SMO) within the rotational loop, enabling it to address force and torque disturbances caused by voltage drop of the battery. Sufficient real flight experiments are conducted to demonstrate the effectiveness of the proposed control scheme under VD disturbance. 

**Abstract (ZH)**: 本论文提出了一种反干扰控制方案，用于抵消电池电压下降（VD）引起的电压下降干扰，该干扰常见于长时间飞行或激烈机动。首先，考虑VD干扰的旋翼无人机精细化动力学模型被呈现。基于该动力学模型，开发了一种电压下降观察器（VDO），通过解耦无人机的扰动和状态信息来精确估计VD干扰，从而减少传统扰动观察器的保守性。随后，控制方案将VDO集成到平移回路中，并将固定时间滑模观察器（SMO）集成到旋转回路中，使其能够处理由电池电压下降引起的力和力矩干扰。进行了充分的实飞实验以证明在VD干扰下所提控制方案的有效性。 

---
# Simulator Ensembles for Trustworthy Autonomous Driving Testing 

**Title (ZH)**: 可信自动驾驶测试的模拟器集合 

**Authors**: Lev Sorokin, Matteo Biagiola, Andrea Stocco  

**Link**: [PDF](https://arxiv.org/pdf/2503.08936)  

**Abstract**: Scenario-based testing with driving simulators is extensively used to identify failing conditions of automated driving assistance systems (ADAS) and reduce the amount of in-field road testing. However, existing studies have shown that repeated test execution in the same as well as in distinct simulators can yield different outcomes, which can be attributed to sources of flakiness or different implementations of the physics, among other factors. In this paper, we present MultiSim, a novel approach to multi-simulation ADAS testing based on a search-based testing approach that leverages an ensemble of simulators to identify failure-inducing, simulator-agnostic test scenarios. During the search, each scenario is evaluated jointly on multiple simulators. Scenarios that produce consistent results across simulators are prioritized for further exploration, while those that fail on only a subset of simulators are given less priority, as they may reflect simulator-specific issues rather than generalizable failures. Our case study, which involves testing a deep neural network-based ADAS on different pairs of three widely used simulators, demonstrates that MultiSim outperforms single-simulator testing by achieving on average a higher rate of simulator-agnostic failures by 51%. Compared to a state-of-the-art multi-simulator approach that combines the outcome of independent test generation campaigns obtained in different simulators, MultiSim identifies 54% more simulator-agnostic failing tests while showing a comparable validity rate. An enhancement of MultiSim that leverages surrogate models to predict simulator disagreements and bypass executions does not only increase the average number of valid failures but also improves efficiency in finding the first valid failure. 

**Abstract (ZH)**: 基于多仿真的场景测试：一种利用搜索测试方法识别模拟器无关故障的新方法 

---
# Acoustic Neural 3D Reconstruction Under Pose Drift 

**Title (ZH)**: 基于姿态漂移的声学神经3D重建 

**Authors**: Tianxiang Lin, Mohamad Qadri, Kevin Zhang, Adithya Pediredla, Christopher A. Metzler, Michael Kaess  

**Link**: [PDF](https://arxiv.org/pdf/2503.08930)  

**Abstract**: We consider the problem of optimizing neural implicit surfaces for 3D reconstruction using acoustic images collected with drifting sensor poses. The accuracy of current state-of-the-art 3D acoustic modeling algorithms is highly dependent on accurate pose estimation; small errors in sensor pose can lead to severe reconstruction artifacts. In this paper, we propose an algorithm that jointly optimizes the neural scene representation and sonar poses. Our algorithm does so by parameterizing the 6DoF poses as learnable parameters and backpropagating gradients through the neural renderer and implicit representation. We validated our algorithm on both real and simulated datasets. It produces high-fidelity 3D reconstructions even under significant pose drift. 

**Abstract (ZH)**: 我们考虑使用漂移传感器姿态收集的声学图像优化3D重建中的神经隐式表面的问题。当前最先进的3D声学建模算法的准确性高度依赖于姿态估计的准确性；传感器姿态中的小误差可能导致严重的重建伪影。在本文中，我们提出了一种联合优化神经场景表示和声纳姿态的算法。该算法通过将6自由度姿态参数化为可学习的参数，并通过神经渲染器和隐式表示反向传播梯度来实现。我们在真实数据集和模拟数据集上验证了该算法，即使在显著的姿态漂移下也能生成高保真的3D重建。 

---
# HessianForge: Scalable LiDAR reconstruction with Physics-Informed Neural Representation and Smoothness Energy Constraints 

**Title (ZH)**: HessianForge：具有物理指导神经表示和平滑能量约束的大规模LiDAR重建 

**Authors**: Hrishikesh Viswanath, Md Ashiqur Rahman, Chi Lin, Damon Conover, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2503.08929)  

**Abstract**: Accurate and efficient 3D mapping of large-scale outdoor environments from LiDAR measurements is a fundamental challenge in robotics, particularly towards ensuring smooth and artifact-free surface reconstructions. Although the state-of-the-art methods focus on memory-efficient neural representations for high-fidelity surface generation, they often fail to produce artifact-free manifolds, with artifacts arising due to noisy and sparse inputs. To address this issue, we frame surface mapping as a physics-informed energy optimization problem, enforcing surface smoothness by optimizing an energy functional that penalizes sharp surface ridges. Specifically, we propose a deep learning based approach that learns the signed distance field (SDF) of the surface manifold from raw LiDAR point clouds using a physics-informed loss function that optimizes the $L_2$-Hessian energy of the surface. Our learning framework includes a hierarchical octree based input feature encoding and a multi-scale neural network to iteratively refine the signed distance field at different scales of resolution. Lastly, we introduce a test-time refinement strategy to correct topological inconsistencies and edge distortions that can arise in the generated mesh. We propose a \texttt{CUDA}-accelerated least-squares optimization that locally adjusts vertex positions to enforce feature-preserving smoothing. We evaluate our approach on large-scale outdoor datasets and demonstrate that our approach outperforms current state-of-the-art methods in terms of improved accuracy and smoothness. Our code is available at \href{this https URL}{this https URL} 

**Abstract (ZH)**: 从LiDAR测量构建大规模室外环境的准确且高效的3D地图是一项根本性的挑战，尤其是在确保无artifact的表面重建方面。尽管最先进的方法关注于高保真表面生成的内存高效神经表示，但它们往往无法生成无artifact的安全流形，因为artifacts源于嘈杂和稀疏的输入。为了解决这一问题，我们将表面映射框架化为一个物理启发的能量优化问题，通过优化一个惩罚尖锐表面脊的能量泛函来保证表面光滑性。具体来说，我们提出了一种基于深度学习的方法，该方法从原始LiDAR点云中学习表面流形的符号距离场（SDF），使用一个物理启发的损失函数来优化表面的$L_2$-Hessian能量。我们的学习框架包括基于层次八叉树的输入特征编码以及多尺度神经网络，以在不同分辨率尺度上迭代细化符号距离场。最后，我们引入了一种测试时的细化策略，以纠正生成网格中可能出现的拓扑不一致性和边缘失真。我们提出了一种CUDA加速的最小二乘优化，局部调整顶点位置以实现特征保的平滑性。我们在大规模室外数据集上评估了我们的方法，并证明在精度和光滑性方面我们的方法优于当前最先进的方法。我们的代码可在\href{this https URL}{this https URL}获得。 

---
# Real-time simulation enabled navigation control of magnetic soft continuum robots in confined lumens 

**Title (ZH)**: 受限管道中基于实时模拟的磁软连续机器人导航控制 

**Authors**: Dezhong Tong, Zhuonan Hao, Jiyu Li, Boxi Sun, Mingchao Liu, Liu Wang, Weicheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08864)  

**Abstract**: Magnetic soft continuum robots (MSCRs) have emerged as a promising technology for minimally invasive interventions, offering enhanced dexterity and remote-controlled navigation in confined lumens. Unlike conventional guidewires with pre-shaped tips, MSCRs feature a magnetic tip that actively bends under applied magnetic fields. Despite extensive studies in modeling and simulation, achieving real-time navigation control of MSCRs in confined lumens remains a significant challenge. The primary reasons are due to robot-lumen contact interactions and computational limitations in modeling MSCR nonlinear behavior under magnetic actuation. Existing approaches, such as Finite Element Method (FEM) simulations and energy-minimization techniques, suffer from high computational costs and oversimplified contact interactions, making them impractical for real-world applications. In this work, we develop a real-time simulation and navigation control framework that integrates hard-magnetic elastic rod theory, formulated within the Discrete Differential Geometry (DDG) framework, with an order-reduced contact handling strategy. Our approach captures large deformations and complex interactions while maintaining computational efficiency. Next, the navigation control problem is formulated as an inverse design task, where optimal magnetic fields are computed in real time by minimizing the constrained forces and enhancing navigation accuracy. We validate the proposed framework through comprehensive numerical simulations and experimental studies, demonstrating its robustness, efficiency, and accuracy. The results show that our method significantly reduces computational costs while maintaining high-fidelity modeling, making it feasible for real-time deployment in clinical settings. 

**Abstract (ZH)**: 基于磁性的软连续体机器人（MSCRs）已成为一种有前景的微创介入技术，能够在狭小的腔道中提供增强的操作灵活性和远程控制导航。与具有预成型尖端的常规导丝不同，MSCRs配备了一个在施加磁场下主动弯曲的磁性尖端。尽管在建模和仿真方面进行了大量研究，但在狭小腔道中实现MSCRs的实时导航控制仍然是一项重大挑战。主要原因在于机器人-腔道接触交互和在磁场驱动下建模MSCRs非线性行为的计算限制。现有的方法，如有限元方法（FEM）仿真和能量最小化技术，由于计算成本高和接触交互的简化，使其在实际应用中不可行。在本工作中，我们开发了一种实时仿真和导航控制框架，该框架结合了在离散微分几何（DDG）框架内制定的硬磁弹性杆理论，并采用了降阶接触处理策略。我们的方法能够捕捉到大面积变形和复杂交互，同时保持计算效率。然后，将导航控制问题形式化为逆设计任务，通过在实时计算约束力最小化和提升导航精度来确定最优磁场。我们通过全面的数值仿真和实验研究验证了所提出的框架，证明了其鲁棒性、效率和准确性。结果显示，我们的方法显著降低了计算成本，同时保持了高保真建模，使其在临床设置中实时部署成为可能。 

---
# Keypoint Semantic Integration for Improved Feature Matching in Outdoor Agricultural Environments 

**Title (ZH)**: 户外农业环境中超連結语义关键点集成以改进特征匹配 

**Authors**: Rajitha de Silva, Jonathan Cox, Marija Popovic, Cesar Cadena, Cyrill Stachniss, Riccardo Polvara  

**Link**: [PDF](https://arxiv.org/pdf/2503.08843)  

**Abstract**: Robust robot navigation in outdoor environments requires accurate perception systems capable of handling visual challenges such as repetitive structures and changing appearances. Visual feature matching is crucial to vision-based pipelines but remains particularly challenging in natural outdoor settings due to perceptual aliasing. We address this issue in vineyards, where repetitive vine trunks and other natural elements generate ambiguous descriptors that hinder reliable feature matching. We hypothesise that semantic information tied to keypoint positions can alleviate perceptual aliasing by enhancing keypoint descriptor distinctiveness. To this end, we introduce a keypoint semantic integration technique that improves the descriptors in semantically meaningful regions within the image, enabling more accurate differentiation even among visually similar local features. We validate this approach in two vineyard perception tasks: (i) relative pose estimation and (ii) visual localisation. Across all tested keypoint types and descriptors, our method improves matching accuracy by 12.6%, demonstrating its effectiveness over multiple months in challenging vineyard conditions. 

**Abstract (ZH)**: 户外环境中的鲁棒机器人导航需要能够处理重复结构和变化外观等视觉挑战的准确感知系统。基于视觉的特征匹配在自然户外环境中尤为关键，但由于感知混叠问题，仍然面临巨大挑战。在葡萄园中，重复的葡萄藤主干和其他自然元素会产生模糊的描述符，阻碍可靠的特征匹配。我们假设与关键点位置相关的语义信息能够通过增强描述符的独特性来缓解感知混叠。为此，我们提出了一种关键点语义集成技术，能够在图像中的语义有意义的区域改进描述符，即使在视觉上相似的局部特征之间也能实现更准确的区分。我们在两个葡萄园感知任务中验证了这种方法：（i）相对姿态估计和（ii）视觉定位。在所有测试的关键点类型和描述符中，我们的方法将匹配准确性提高12.6%，证明了其在挑战性葡萄园条件下的有效性。 

---
# Cooperative Bearing-Only Target Pursuit via Multiagent Reinforcement Learning: Design and Experiment 

**Title (ZH)**: 多智能体强化学习引导下的协作基于 bearing 的目标追击：设计与实验 

**Authors**: Jianan Li, Zhikun Wang, Susheng Ding, Shiliang Guo, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08740)  

**Abstract**: This paper addresses the multi-robot pursuit problem for an unknown target, encompassing both target state estimation and pursuit control. First, in state estimation, we focus on using only bearing information, as it is readily available from vision sensors and effective for small, distant targets. Challenges such as instability due to the nonlinearity of bearing measurements and singularities in the two-angle representation are addressed through a proposed uniform bearing-only information filter. This filter integrates multiple 3D bearing measurements, provides a concise formulation, and enhances stability and resilience to target loss caused by limited field of view (FoV). Second, in target pursuit control within complex environments, where challenges such as heterogeneity and limited FoV arise, conventional methods like differential games or Voronoi partitioning often prove inadequate. To address these limitations, we propose a novel multiagent reinforcement learning (MARL) framework, enabling multiple heterogeneous vehicles to search, localize, and follow a target while effectively handling those challenges. Third, to bridge the sim-to-real gap, we propose two key techniques: incorporating adjustable low-level control gains in training to replicate the dynamics of real-world autonomous ground vehicles (AGVs), and proposing spectral-normalized RL algorithms to enhance policy smoothness and robustness. Finally, we demonstrate the successful zero-shot transfer of the MARL controllers to AGVs, validating the effectiveness and practical feasibility of our approach. The accompanying video is available at this https URL. 

**Abstract (ZH)**: 本文解决了未知目标的多机器人捕获问题，涵盖了目标状态估计和捕获控制。首先，在状态估计中，我们专注于使用方位信息，因为方位信息来自视觉传感器且适用于小型远距离目标。通过提出一种统一的方位-only信息滤波器，解决了由于方位测量的非线性引起的不稳定性和两角表示中的奇异性问题。该滤波器整合了多个3D方位测量值，提供了简洁的形式，并增强了在视场（FoV）受限导致的目标丢失时的稳定性和鲁棒性。其次，在复杂环境下的目标捕获控制中，由于异质性和有限的视场等因素，传统的差分博弈或Voronoi划分方法通常效果不佳。为了解决这些问题，我们提出了一种新颖的多Agent强化学习（MARL）框架，使多个异构车辆能够搜索、定位并跟随目标，有效应对这些挑战。第三，为了弥合仿真与现实之间的差距，我们提出了两种关键技术：在训练中引入可调低级控制增益以模拟现实世界自主地面车辆（AGVs）的动力学特性，以及提出谱正则化RL算法以提升策略的平滑性和鲁棒性。最后，我们展示了MARL控制器成功实现了对AGVs的零样本迁移，验证了我们方法的有效性和实用性。相关视频可在以下链接获取。 

---
# Out-of-Distribution Segmentation in Autonomous Driving: Problems and State of the Art 

**Title (ZH)**: 自动驾驶中未知分布分割：问题与最新进展 

**Authors**: Youssef Shoeb, Azarm Nowzad, Hanno Gottschalk  

**Link**: [PDF](https://arxiv.org/pdf/2503.08695)  

**Abstract**: In this paper, we review the state of the art in Out-of-Distribution (OoD) segmentation, with a focus on road obstacle detection in automated driving as a real-world application. We analyse the performance of existing methods on two widely used benchmarks, SegmentMeIfYouCan Obstacle Track and LostAndFound-NoKnown, highlighting their strengths, limitations, and real-world applicability. Additionally, we discuss key challenges and outline potential research directions to advance the field. Our goal is to provide researchers and practitioners with a comprehensive perspective on the current landscape of OoD segmentation and to foster further advancements toward safer and more reliable autonomous driving systems. 

**Abstract (ZH)**: 本文回顾了异常分布（OoD）分割的最新进展，并重点分析了在自动驾驶中道路障碍检测的实际应用。我们在SegmentMeIfYouCan Obstacle Track和LostAndFound-NoKnown两个广泛使用的基准上评估现有方法的表现，指出了它们的优势、局限性和实际应用前景。此外，我们讨论了关键挑战，并概述了潜在的研究方向以推进该领域的发展。本文旨在为研究人员和实践者提供当前异常分布分割领域全面的视角，并促进向更安全可靠的自动驾驶系统的进一步发展。 

---
