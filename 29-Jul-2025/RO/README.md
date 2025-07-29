# PixelNav: Towards Model-based Vision-Only Navigation with Topological Graphs 

**Title (ZH)**: PixelNav: 向基于拓扑图的纯视觉导航模型方向发展 

**Authors**: Sergey Bakulin, Timur Akhtyamov, Denis Fatykhov, German Devchich, Gonzalo Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2507.20892)  

**Abstract**: This work proposes a novel hybrid approach for vision-only navigation of mobile robots, which combines advances of both deep learning approaches and classical model-based planning algorithms. Today, purely data-driven end-to-end models are dominant solutions to this problem. Despite advantages such as flexibility and adaptability, the requirement of a large amount of training data and limited interpretability are the main bottlenecks for their practical applications. To address these limitations, we propose a hierarchical system that utilizes recent advances in model predictive control, traversability estimation, visual place recognition, and pose estimation, employing topological graphs as a representation of the target environment. Using such a combination, we provide a scalable system with a higher level of interpretability compared to end-to-end approaches. Extensive real-world experiments show the efficiency of the proposed method. 

**Abstract (ZH)**: 本研究提出了一种新颖的混合方法，用于仅依靠视觉的移动机器人导航，该方法结合了深度学习方法和经典模型导向规划算法的最新进展。 

---
# A Human-in-the-loop Approach to Robot Action Replanning through LLM Common-Sense Reasoning 

**Title (ZH)**: 基于LLM常识推理的人在回环中的机器人动作重规划方法 

**Authors**: Elena Merlo, Marta Lagomarsino, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2507.20870)  

**Abstract**: To facilitate the wider adoption of robotics, accessible programming tools are required for non-experts. Observational learning enables intuitive human skills transfer through hands-on demonstrations, but relying solely on visual input can be inefficient in terms of scalability and failure mitigation, especially when based on a single demonstration. This paper presents a human-in-the-loop method for enhancing the robot execution plan, automatically generated based on a single RGB video, with natural language input to a Large Language Model (LLM). By including user-specified goals or critical task aspects and exploiting the LLM common-sense reasoning, the system adjusts the vision-based plan to prevent potential failures and adapts it based on the received instructions. Experiments demonstrated the framework intuitiveness and effectiveness in correcting vision-derived errors and adapting plans without requiring additional demonstrations. Moreover, interactive plan refinement and hallucination corrections promoted system robustness. 

**Abstract (ZH)**: 面向非专家的易用编程工具是促进机器人更广泛采用的关键。观察学习可以通过亲手示范实现直观的人类技能转移，但仅依赖视觉输入在可扩展性和故障缓解方面可能效率低下，尤其是在基于单次示范时。本文提出一种在环人类辅助方法，通过自然语言输入到大型语言模型（LLM），增强基于单个RGB视频自动生成的机器人执行计划。通过纳入用户指定的目标或关键任务方面，并利用LLM的常识推理，系统调整基于视觉的计划以防止潜在失败，并根据接收到的指令进行适应。实验表明，该框架在纠正基于视觉的错误和不需额外示范的情况下调整计划方面具有直观性和有效性。此外，交互式计划细化和幻觉修正提升了系统的鲁棒性。 

---
# Uncertainty-aware Planning with Inaccurate Models for Robotized Liquid Handling 

**Title (ZH)**: 基于不准确模型的aware不确定性规划与机器人液体处理 

**Authors**: Marco Faroni, Carlo Odesco, Andrea Zanchettin, Paolo Rocco  

**Link**: [PDF](https://arxiv.org/pdf/2507.20861)  

**Abstract**: Physics-based simulations and learning-based models are vital for complex robotics tasks like deformable object manipulation and liquid handling. However, these models often struggle with accuracy due to epistemic uncertainty or the sim-to-real gap. For instance, accurately pouring liquid from one container to another poses challenges, particularly when models are trained on limited demonstrations and may perform poorly in novel situations. This paper proposes an uncertainty-aware Monte Carlo Tree Search (MCTS) algorithm designed to mitigate these inaccuracies. By incorporating estimates of model uncertainty, the proposed MCTS strategy biases the search towards actions with lower predicted uncertainty. This approach enhances the reliability of planning under uncertain conditions. Applied to a liquid pouring task, our method demonstrates improved success rates even with models trained on minimal data, outperforming traditional methods and showcasing its potential for robust decision-making in robotics. 

**Abstract (ZH)**: 基于物理的模拟和基于学习的模型对于复杂机器人任务如可变形物体操作和液体处理至关重要。然而，这些模型往往由于认识不确定性或仿真到真实世界的差距而准确性不足。例如，精确地将液体从一个容器倒入另一个容器时会面临挑战，特别是在模型基于有限演示训练的情况下，在新颖情况下表现较差。本文提出了一种意识到不确定性的蒙特卡洛树搜索（MCTS）算法，旨在减轻这些不准确性。通过融入模型不确定性估计，所提出的MCTS策略倾向于那些预测不确定性较低的动作。这种方法在不确定条件下增强了规划的可靠性。应用于液体倾倒任务时，我们的方法即使在使用少量数据训练的模型上也表现出更高的成功率，优于传统方法，并展示了其在机器人中进行稳健决策的潜力。 

---
# Free Energy-Inspired Cognitive Risk Integration for AV Navigation in Pedestrian-Rich Environments 

**Title (ZH)**: 自由能启发的认知风险集成在行人密集环境中自动驾驶车辆导航 

**Authors**: Meiting Dang, Yanping Wu, Yafei Wang, Dezong Zhao, David Flynn, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.20850)  

**Abstract**: Recent advances in autonomous vehicle (AV) behavior planning have shown impressive social interaction capabilities when interacting with other road users. However, achieving human-like prediction and decision-making in interactions with vulnerable road users remains a key challenge in complex multi-agent interactive environments. Existing research focuses primarily on crowd navigation for small mobile robots, which cannot be directly applied to AVs due to inherent differences in their decision-making strategies and dynamic boundaries. Moreover, pedestrians in these multi-agent simulations follow fixed behavior patterns that cannot dynamically respond to AV actions. To overcome these limitations, this paper proposes a novel framework for modeling interactions between the AV and multiple pedestrians. In this framework, a cognitive process modeling approach inspired by the Free Energy Principle is integrated into both the AV and pedestrian models to simulate more realistic interaction dynamics. Specifically, the proposed pedestrian Cognitive-Risk Social Force Model adjusts goal-directed and repulsive forces using a fused measure of cognitive uncertainty and physical risk to produce human-like trajectories. Meanwhile, the AV leverages this fused risk to construct a dynamic, risk-aware adjacency matrix for a Graph Convolutional Network within a Soft Actor-Critic architecture, allowing it to make more reasonable and informed decisions. Simulation results indicate that our proposed framework effectively improves safety, efficiency, and smoothness of AV navigation compared to the state-of-the-art method. 

**Abstract (ZH)**: 近年来，自动驾驶车辆（AV）行为规划方面的进展在与其他道路使用者互动时展现了令人印象深刻的社交互动能力。然而，在复杂多agents交互环境中与弱势道路使用者进行人类般的预测和决策仍然是一项关键挑战。现有研究主要集中在小型移动机器人 crowd navigation 上，由于其决策策略和动态边界的本质差异，无法直接应用于AV。此外，在这些多agents模拟中，行人的行为模式是固定的，不能动态响应AV的动作。为了克服这些局限性，本文提出了一种用于建模AV与多名行人间交互的新框架。在此框架中，借鉴Free Energy Principle的认知过程建模方法被整合到AV和行人的模型中，以模拟更真实的交互动力学。具体而言，所提出的行人认知-风险社会力模型通过结合认知不确定性和物理风险的综合衡量来调整目标向和排斥力，从而产生类似人类的轨迹。与此同时，AV利用这种综合风险构建Soft Actor-Critic架构内的Graph Convolutional Network的动态、风险意识邻接矩阵，使其能够做出更加合理和知情的决策。仿真结果表明，与最新方法相比，我们提出的新框架有效提高了AV导航的安全性、效率和流畅度。 

---
# Hanging Around: Cognitive Inspired Reasoning for Reactive Robotics 

**Title (ZH)**: 悬挂于此：认知启发式的反应式机器人推理 

**Authors**: Mihai Pomarlan, Stefano De Giorgis, Rachel Ringe, Maria M. Hedblom, Nikolaos Tsiogkas  

**Link**: [PDF](https://arxiv.org/pdf/2507.20832)  

**Abstract**: Situationally-aware artificial agents operating with competence in natural environments face several challenges: spatial awareness, object affordance detection, dynamic changes and unpredictability. A critical challenge is the agent's ability to identify and monitor environmental elements pertinent to its objectives. Our research introduces a neurosymbolic modular architecture for reactive robotics. Our system combines a neural component performing object recognition over the environment and image processing techniques such as optical flow, with symbolic representation and reasoning. The reasoning system is grounded in the embodied cognition paradigm, via integrating image schematic knowledge in an ontological structure. The ontology is operatively used to create queries for the perception system, decide on actions, and infer entities' capabilities derived from perceptual data. The combination of reasoning and image processing allows the agent to focus its perception for normal operation as well as discover new concepts for parts of objects involved in particular interactions. The discovered concepts allow the robot to autonomously acquire training data and adjust its subsymbolic perception to recognize the parts, as well as making planning for more complex tasks feasible by focusing search on those relevant object parts. We demonstrate our approach in a simulated world, in which an agent learns to recognize parts of objects involved in support relations. While the agent has no concept of handle initially, by observing examples of supported objects hanging from a hook it learns to recognize the parts involved in establishing support and becomes able to plan the establishment/destruction of the support relation. This underscores the agent's capability to expand its knowledge through observation in a systematic way, and illustrates the potential of combining deep reasoning [...]. 

**Abstract (ZH)**: 情景感知的人工智能代理在自然环境中高效运作面临的挑战：空间意识、物体操作检测、动态变化和不确定性。一个关键挑战是代理识别和监控与其目标相关环境元素的能力。我们的研究提出了一种神经符号模块化架构以应对反应式机器人面临的挑战。该系统结合了神经网络执行的环境物体识别和光学流等图像处理技术，同时使用符号表示和推理。推理系统通过将图像示意性知识整合到本体结构中，基于体表认知范式运行。本体用于为感知系统创建查询、决定行动以及从感知数据中推断实体的能力。推理与图像处理的结合使代理能够在正常操作时集中感知，并且能够发现与特定交互相关的对象部分的新概念。发现的概念使机器人能够自主获取训练数据，调整其次符号感知以识别这些部分，并通过聚焦于相关对象部分的搜索使复杂任务的规划成为可能。我们在一个模拟世界中展示了该方法，一个代理学会了识别涉及支持关系的对象部分。在最初没有把手的概念的情况下，通过观察悬挂于挂钩上的支撑对象示例，代理学会了识别建立支持的部分，并能够规划支持关系的建立或破坏。这强调了代理通过系统观察扩展知识的能力，并展示了结合深度推理 […] 的潜力。 

---
# LanternNet: A Novel Hub-and-Spoke System to Seek and Suppress Spotted Lanternfly Populations 

**Title (ZH)**: LanternNet: 一种寻求并抑制星天牛种群的新颖辐辏系统 

**Authors**: Vinil Polepalli  

**Link**: [PDF](https://arxiv.org/pdf/2507.20800)  

**Abstract**: The invasive spotted lanternfly (SLF) poses a significant threat to agriculture and ecosystems, causing widespread damage. Current control methods, such as egg scraping, pesticides, and quarantines, prove labor-intensive, environmentally hazardous, and inadequate for long-term SLF suppression. This research introduces LanternNet, a novel autonomous robotic Hub-and-Spoke system designed for scalable detection and suppression of SLF populations. A central, tree-mimicking hub utilizes a YOLOv8 computer vision model for precise SLF identification. Three specialized robotic spokes perform targeted tasks: pest neutralization, environmental monitoring, and navigation/mapping. Field deployment across multiple infested sites over 5 weeks demonstrated LanternNet's efficacy. Quantitative analysis revealed significant reductions (p < 0.01, paired t-tests) in SLF populations and corresponding improvements in tree health indicators across the majority of test sites. Compared to conventional methods, LanternNet offers substantial cost advantages and improved scalability. Furthermore, the system's adaptability for enhanced autonomy and targeting of other invasive species presents significant potential for broader ecological impact. LanternNet demonstrates the transformative potential of integrating robotics and AI for advanced invasive species management and improved environmental outcomes. 

**Abstract (ZH)**: 入侵的星天牛（SLF）对农业和生态系统构成了重大威胁，造成广泛破坏。现有的控制方法，如卵剥离、农药和隔离措施，证明劳动密集、环境有害且不足以长期抑制SLF。本研究介绍了一种新型自主 robotic 集中式辐辏系统LanternNet，用于 scalability 的SLF种群检测和抑制。中央树状仿生中心利用YOLOv8计算机视觉模型进行精准的SLF识别。三个专业化的机器人辐条分别执行特定任务：害虫中和、环境监测和导航/制图。在多个受侵害地点的田间部署5周显示了LanternNet的有效性。定量分析表明，在绝大多数试验地点，SLF种群显著减少（p < 0.01，配对t检验），树健康指标也有相应改善。与传统方法相比，LanternNet在成本优势和可扩展性方面具有明显优势。此外，系统增强了自主性和针对其他入侵物种的针对性，展现出对更广泛生态影响的巨大潜力。LanternNet展示了将机器人技术和AI集成以实现进阶入侵物种管理和改善环境结果的潜在变革性。 

---
# A Strawberry Harvesting Tool with Minimal Footprint 

**Title (ZH)**: 一种占用空间最小的草莓采摘工具 

**Authors**: Mohamed Sorour, Mohamed Heshmat, Khaled Elgeneidy, Pål Johan From  

**Link**: [PDF](https://arxiv.org/pdf/2507.20784)  

**Abstract**: In this paper, a novel prototype for harvesting table-top grown strawberries is presented, that is minimalist in its footprint interacting with the fruit. In our methodology, a smooth trapper manipulates the stem into a precise groove location at which a distant laser beam is focused. The tool reaches temperatures as high as 188° Celsius and as such killing germs and preventing the spread of local plant diseases. The burnt stem wound preserves water content and in turn the fruit shelf life. Cycle and cut times achieved are 5.56 and 2.88 seconds respectively in successful in-door harvesting demonstration. Extensive experiments are performed to optimize the laser spot diameter and lateral speed against the cutting time. 

**Abstract (ZH)**: 本文提出了一种用于收获桌面种植草莓的简约型采集原型，通过平滑的捕捉装置将草莓茎引导至精确的沟槽位置，随后聚焦远处的激光束进行切割。该工具最高可达到188摄氏度的温度，从而杀死细菌并防止局部植物疾病传播。烧焦的茎伤口保留水分，进而延长果实保质期。室内收获演示中，切割周期和切割时间分别达到5.56秒和2.88秒。进行了大量实验以优化激光斑点直径和横向速度与切割时间的关系。 

---
# FMimic: Foundation Models are Fine-grained Action Learners from Human Videos 

**Title (ZH)**: FMimic: 基础模型是从人类视频中细粒度学习动作的模型 

**Authors**: Guangyan Chen, Meiling Wang, Te Cui, Yao Mu, Haoyang Lu, Zicai Peng, Mengxiao Hu, Tianxing Zhou, Mengyin Fu, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.20622)  

**Abstract**: Visual imitation learning (VIL) provides an efficient and intuitive strategy for robotic systems to acquire novel skills. Recent advancements in foundation models, particularly Vision Language Models (VLMs), have demonstrated remarkable capabilities in visual and linguistic reasoning for VIL tasks. Despite this progress, existing approaches primarily utilize these models for learning high-level plans from human demonstrations, relying on pre-defined motion primitives for executing physical interactions, which remains a major bottleneck for robotic systems. In this work, we present FMimic, a novel paradigm that harnesses foundation models to directly learn generalizable skills at even fine-grained action levels, using only a limited number of human videos. Extensive experiments demonstrate that our FMimic delivers strong performance with a single human video, and significantly outperforms all other methods with five videos. Furthermore, our method exhibits significant improvements of over 39% and 29% in RLBench multi-task experiments and real-world manipulation tasks, respectively, and exceeds baselines by more than 34% in high-precision tasks and 47% in long-horizon tasks. 

**Abstract (ZH)**: 视觉模仿学习（VIL）为机器人系统获取新型技能提供了一种高效直观的策略。基于基础模型的最新进展，尤其是视觉语言模型（VLMs），展示了在VIL任务中视觉和语言推理的非凡能力。尽管取得了这些进展，现有的方法主要利用这些模型从人类示范中学习高级规划，并依赖预定义的运动基元来执行物理交互，这是机器人系统的一个主要瓶颈。在此项工作中，我们提出了FMimic，这是一种全新的范式，利用基础模型直接在极细粒度的动作层面学习可泛化的技能，仅使用少量的人类视频。 extensive实验表明，我们的FMimic仅使用单个人类视频就能达到出色的性能，并且在使用五个视频时显著优于所有其他方法。此外，在RLBench多任务实验和实际操作任务中，我们的方法分别表现出超过39%和29%的改进，并在高精度任务和长时序任务中分别超过基线方法34%和47%。 

---
# Methods for the Segmentation of Reticular Structures Using 3D LiDAR Data: A Comparative Evaluation 

**Title (ZH)**: 使用3D LiDAR数据分割网状结构的方法：一种比较评估 

**Authors**: Francisco J. Soler Mora, Adrián Peidró Vidal, Marc Fabregat-Jaén, Luis Payá Castelló, Óscar Reinoso García  

**Link**: [PDF](https://arxiv.org/pdf/2507.20589)  

**Abstract**: Reticular structures form the backbone of major infrastructure like bridges, pylons, and airports, but their inspection and maintenance are costly and hazardous, often requiring human intervention. While prior research has focused on fault detection via images or robotic platform design, the autonomous navigation of robots within these structures is less explored. This study addresses that gap by proposing methods to detect navigable surfaces in truss structures, enhancing the autonomy of climbing robots. The paper introduces several approaches for binary segmentation of navigable surfaces versus background from 3D point clouds of metallic trusses. These methods fall into two categories: analytical algorithms and deep learning models. The analytical approach features a custom algorithm that segments structures by analyzing the eigendecomposition of planar patches in the point cloud. In parallel, advanced deep learning models PointNet, PointNet++, MinkUNet34C, and PointTransformerV3 are trained and evaluated for the same task. Comparative analysis shows that the analytical algorithm offers easier parameter tuning and performance comparable to deep learning models, which, while more computationally intensive, excel in segmentation accuracy. Notably, PointTransformerV3 achieves a Mean Intersection Over Union (mIoU) of about 97%. The study demonstrates the promise of both analytical and deep learning methods for improving autonomous navigation in complex truss environments. The results highlight the trade-offs between computational efficiency and segmentation performance, providing valuable guidance for future research and practical applications in autonomous infrastructure inspection and maintenance. 

**Abstract (ZH)**: 网状结构构成桥梁、电线杆和机场等重大基础设施的骨干，但其检测和维护成本高且危险，常需人工干预。尽管以往研究主要集中在通过图像进行故障检测或机器人平台设计，但在这些结构中自主导航机器人的探索较少。本研究通过提出检测桁架结构中可通行表面的方法，增强了攀爬机器人的自主性。论文介绍了从金属桁架的3D点云中分割可通行表面与背景的二元分割方法，这些方法分为两类：解析算法和深度学习模型。解析方法包含一个定制算法，通过分析点云中平面片段的特征值分解来进行结构分割。同时，PointNet、PointNet++、MinkUNet34C 和 PointTransformerV3 等高级深度学习模型被训练和评估以完成相同任务。比较分析表明，解析算法具有更易调参且性能与深度学习模型相当的特点，尽管深度学习模型在计算上更为密集，但在分割准确性方面表现出色。特别地，PointTransformerV3 达到了约 97% 的平均交并比 (mIoU)。研究展示了解析方法和深度学习方法在复杂桁架环境下的自主导航中的潜力。结果突显了计算效率与分割性能之间的权衡，为未来研究和自主基础设施检测与维护的实际应用提供了宝贵指导。 

---
# Uni-Mapper: Unified Mapping Framework for Multi-modal LiDARs in Complex and Dynamic Environments 

**Title (ZH)**: Uni-Mapper: 统一多模态LiDAR在复杂动态环境中的mapping框架 

**Authors**: Gilhwan Kang, Hogyun Kim, Byunghee Choi, Seokhwan Jeong, Young-Sik Shin, Younggun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2507.20538)  

**Abstract**: The unification of disparate maps is crucial for enabling scalable robot operation across multiple sessions and collaborative multi-robot scenarios. However, achieving a unified map robust to sensor modalities and dynamic environments remains a challenging problem. Variations in LiDAR types and dynamic elements lead to differences in point cloud distribution and scene consistency, hindering reliable descriptor generation and loop closure detection essential for accurate map alignment. To address these challenges, this paper presents Uni-Mapper, a dynamic-aware 3D point cloud map merging framework for multi-modal LiDAR systems. It comprises dynamic object removal, dynamic-aware loop closure, and multi-modal LiDAR map merging modules. A voxel-wise free space hash map is built in a coarse-to-fine manner to identify and reject dynamic objects via temporal occupancy inconsistencies. The removal module is integrated with a LiDAR global descriptor, which encodes preserved static local features to ensure robust place recognition in dynamic environments. In the final stage, multiple pose graph optimizations are conducted for both intra-session and inter-map loop closures. We adopt a centralized anchor-node strategy to mitigate intra-session drift errors during map merging. In the final stage, centralized anchor-node-based pose graph optimization is performed to address intra- and inter-map loop closures for globally consistent map merging. Our framework is evaluated on diverse real-world datasets with dynamic objects and heterogeneous LiDARs, showing superior performance in loop detection across sensor modalities, robust mapping in dynamic environments, and accurate multi-map alignment over existing methods. Project Page: this https URL. 

**Abstract (ZH)**: 多模态激光雷达系统中动态感知的3D点云地图融合框架 

---
# Large-Scale LiDAR-Inertial Dataset for Degradation-Robust High-Precision Mapping 

**Title (ZH)**: 大规模激光雷达-惯性数据集以实现鲁棒高精度映射 

**Authors**: Xiaofeng Jin, Ningbo Bu, Shijie Wang, Jianfei Ge, Jiangjian Xiao, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2507.20516)  

**Abstract**: This paper introduces a large-scale, high-precision LiDAR-Inertial Odometry (LIO) dataset, aiming to address the insufficient validation of LIO systems in complex real-world scenarios in existing research. The dataset covers four diverse real-world environments spanning 60,000 to 750,000 square meters, collected using a custom backpack-mounted platform equipped with multi-beam LiDAR, an industrial-grade IMU, and RTK-GNSS modules. The dataset includes long trajectories, complex scenes, and high-precision ground truth, generated by fusing SLAM-based optimization with RTK-GNSS anchoring, and validated for trajectory accuracy through the integration of oblique photogrammetry and RTK-GNSS. This dataset provides a comprehensive benchmark for evaluating the generalization ability of LIO systems in practical high-precision mapping scenarios. 

**Abstract (ZH)**: 本文介绍了大规模高精度激光雷达-惯性里程计（LIO）数据集，旨在解决现有研究中LIO系统在复杂真实场景下验证不足的问题。数据集涵盖了四个多样化的现实环境，面积从60,000到750,000平方米不等，采用定制的背负式平台收集数据，该平台配备有多束激光雷达、工业级IMU和RTK-GNSS模块。数据集包括长轨迹、复杂场景和通过将SLAM优化与RTK-GNSS锚定融合生成的高精度真值，并通过倾斜摄影测量和RTK-GNSS综合验证轨迹准确性。该数据集为评估LIO系统在实际高精度测绘场景下的泛化能力提供了全面的基准。 

---
# LLMs-guided adaptive compensator: Bringing Adaptivity to Automatic Control Systems with Large Language Models 

**Title (ZH)**: 基于LLMs的自适应补偿器：大型语言模型赋能的自适应自动控制系统 

**Authors**: Zhongchao Zhou, Yuxi Lu, Yaonan Zhu, Yifan Zhao, Bin He, Liang He, Wenwen Yu, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.20509)  

**Abstract**: With rapid advances in code generation, reasoning, and problem-solving, Large Language Models (LLMs) are increasingly applied in robotics. Most existing work focuses on high-level tasks such as task decomposition. A few studies have explored the use of LLMs in feedback controller design; however, these efforts are restricted to overly simplified systems, fixed-structure gain tuning, and lack real-world validation. To further investigate LLMs in automatic control, this work targets a key subfield: adaptive control. Inspired by the framework of model reference adaptive control (MRAC), we propose an LLM-guided adaptive compensator framework that avoids designing controllers from scratch. Instead, the LLMs are prompted using the discrepancies between an unknown system and a reference system to design a compensator that aligns the response of the unknown system with that of the reference, thereby achieving adaptivity. Experiments evaluate five methods: LLM-guided adaptive compensator, LLM-guided adaptive controller, indirect adaptive control, learning-based adaptive control, and MRAC, on soft and humanoid robots in both simulated and real-world environments. Results show that the LLM-guided adaptive compensator outperforms traditional adaptive controllers and significantly reduces reasoning complexity compared to the LLM-guided adaptive controller. The Lyapunov-based analysis and reasoning-path inspection demonstrate that the LLM-guided adaptive compensator enables a more structured design process by transforming mathematical derivation into a reasoning task, while exhibiting strong generalizability, adaptability, and robustness. This study opens a new direction for applying LLMs in the field of automatic control, offering greater deployability and practicality compared to vision-language models. 

**Abstract (ZH)**: 大型语言模型在机器人领域自适应控制中的作用探索：基于模型参考自适应控制的LLM引导自适应补偿器框架 

---
# Learning Physical Interaction Skills from Human Demonstrations 

**Title (ZH)**: 从人类示范中学习物理交互技能 

**Authors**: Tianyu Li, Hengbo Ma, Sehoon Ha, Kwonjoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.20445)  

**Abstract**: Learning physical interaction skills, such as dancing, handshaking, or sparring, remains a fundamental challenge for agents operating in human environments, particularly when the agent's morphology differs significantly from that of the demonstrator. Existing approaches often rely on handcrafted objectives or morphological similarity, limiting their capacity for generalization. Here, we introduce a framework that enables agents with diverse embodiments to learn wholebbody interaction behaviors directly from human demonstrations. The framework extracts a compact, transferable representation of interaction dynamics, called the Embedded Interaction Graph (EIG), which captures key spatiotemporal relationships between the interacting agents. This graph is then used as an imitation objective to train control policies in physics-based simulations, allowing the agent to generate motions that are both semantically meaningful and physically feasible. We demonstrate BuddyImitation on multiple agents, such as humans, quadrupedal robots with manipulators, or mobile manipulators and various interaction scenarios, including sparring, handshaking, rock-paper-scissors, or dancing. Our results demonstrate a promising path toward coordinated behaviors across morphologically distinct characters via cross embodiment interaction learning. 

**Abstract (ZH)**: 一种框架使不同形态的代理可以直接从人类示范中学习全身交互行为，以解决在人类环境中操作时形态差异显著的代理学习物理互动技能的问题。该框架提取了一个紧凑且可转移的交互动力学表示——嵌入式交互图（EIG），以捕捉交互代理间的时空关系。随后，该图被用作模仿目标，用于基于物理的模拟中训练控制策略，从而使代理能够生成既具有语义意义又符合物理可能性的运动。我们展示了BuddyImitation在多种代理上，如人类、具有 manipulator 的四足机器人或移动 manipulator，以及多种交互场景，包括对打、握手、石头剪刀布或跳舞中的应用。我们的结果展示了一种通过跨形态互动学习实现不同形态角色协调行为的有希望的途径。 

---
# Model-Structured Neural Networks to Control the Steering Dynamics of Autonomous Race Cars 

**Title (ZH)**: 基于模型结构的神经网络控制自主赛车转向动力学 

**Authors**: Mattia Piccinini, Aniello Mungiello, Georg Jank, Gastone Pietro Rosati Papini, Francesco Biral, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2507.20427)  

**Abstract**: Autonomous racing has gained increasing attention in recent years, as a safe environment to accelerate the development of motion planning and control methods for autonomous driving. Deep learning models, predominantly based on neural networks (NNs), have demonstrated significant potential in modeling the vehicle dynamics and in performing various tasks in autonomous driving. However, their black-box nature is critical in the context of autonomous racing, where safety and robustness demand a thorough understanding of the decision-making algorithms. To address this challenge, this paper proposes MS-NN-steer, a new Model-Structured Neural Network for vehicle steering control, integrating the prior knowledge of the nonlinear vehicle dynamics into the neural architecture. The proposed controller is validated using real-world data from the Abu Dhabi Autonomous Racing League (A2RL) competition, with full-scale autonomous race cars. In comparison with general-purpose NNs, MS-NN-steer is shown to achieve better accuracy and generalization with small training datasets, while being less sensitive to the weights' initialization. Also, MS-NN-steer outperforms the steering controller used by the A2RL winning team. Our implementation is available open-source in a GitHub repository. 

**Abstract (ZH)**: 自主赛车在近年来获得了越来越多的关注，作为加速自主驾驶运动规划与控制方法发展的安全环境。基于神经网络（NNs）的深度学习模型在建模车辆动力学和执行各种自主驾驶任务方面展现了显著潜力。然而，在自主赛车的背景下，由于安全性与鲁棒性要求对决策算法有全面的理解，其黑盒性质成为关键问题。为应对这一挑战，本文提出MS-NN-steer，这是一种新的模型结构化神经网络，将非线性车辆动力学的先验知识集成到神经网络架构中。所提出的控制器使用阿联酋自主赛车联赛（A2RL）比赛中的全尺寸自主赛车进行验证，结果显示与通用神经网络相比，MS-NN-steer在小规模训练数据集上具有更好的准确性和泛化能力，并且对权重初始化的敏感性较低。此外，MS-NN-steer在性能上优于A2RL获胜团队使用的转向控制器。我们的实现已在GitHub仓库中开源。 

---
# Bipedalism for Quadrupedal Robots: Versatile Loco-Manipulation through Risk-Adaptive Reinforcement Learning 

**Title (ZH)**: 四足机器人的人行 :";
user
请纠正这个标题：Bipedalism for Quadrupedal Robots: Versatile Loco-Manipulation through Risk-Adaptive Reinforcement Learning

四足机器人的人行走态：基于风险自适应强化学习的多功能运动操控 

**Authors**: Yuyou Zhang, Radu Corcodel, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20382)  

**Abstract**: Loco-manipulation of quadrupedal robots has broadened robotic applications, but using legs as manipulators often compromises locomotion, while mounting arms complicates the system. To mitigate this issue, we introduce bipedalism for quadrupedal robots, thus freeing the front legs for versatile interactions with the environment. We propose a risk-adaptive distributional Reinforcement Learning (RL) framework designed for quadrupedal robots walking on their hind legs, balancing worst-case conservativeness with optimal performance in this inherently unstable task. During training, the adaptive risk preference is dynamically adjusted based on the uncertainty of the return, measured by the coefficient of variation of the estimated return distribution. Extensive experiments in simulation show our method's superior performance over baselines. Real-world deployment on a Unitree Go2 robot further demonstrates the versatility of our policy, enabling tasks like cart pushing, obstacle probing, and payload transport, while showcasing robustness against challenging dynamics and external disturbances. 

**Abstract (ZH)**: 四足机器人 bipedalism 的引入拓宽了机器人的应用范围，通过使前腿自由，实现了与环境的多样互动。我们提出一种基于分布的风险适应性强化学习（RL）框架，该框架适用于四足机器人后腿行走的任务，能够在这一固有的不稳定任务中实现最坏情况下的保守性和最优性能的最佳平衡。在训练过程中，根据估计回报分布的标准差系数动态调整适应的风险偏好。仿真中的广泛实验显示，我们的方法在基准方法之上表现出更优的性能。在 Unitree Go2 网格机器人的实际部署中，进一步展示了该策略的灵活性，能够执行小车推拉、障碍探查和负载运输等任务，并展示了对复杂动态和外部干扰的鲁棒性。 

---
# Advancing Shared and Multi-Agent Autonomy in Underwater Missions: Integrating Knowledge Graphs and Retrieval-Augmented Generation 

**Title (ZH)**: 推进海底任务中共享与多智能体自主性：集成知识图谱与检索增强生成 

**Authors**: Michele Grimaldi, Carlo Cernicchiaro, Sebastian Realpe Rua, Alaaeddine El-Masri-El-Chaarani, Markus Buchholz, Loizos Michael, Pere Ridao Rodriguez, Ignacio Carlucho, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2507.20370)  

**Abstract**: Robotic platforms have become essential for marine operations by providing regular and continuous access to offshore assets, such as underwater infrastructure inspection, environmental monitoring, and resource exploration. However, the complex and dynamic nature of underwater environments, characterized by limited visibility, unpredictable currents, and communication constraints, presents significant challenges that demand advanced autonomy while ensuring operator trust and oversight. Central to addressing these challenges are knowledge representation and reasoning techniques, particularly knowledge graphs and retrieval-augmented generation (RAG) systems, that enable robots to efficiently structure, retrieve, and interpret complex environmental data. These capabilities empower robotic agents to reason, adapt, and respond effectively to changing conditions. The primary goal of this work is to demonstrate both multi-agent autonomy and shared autonomy, where multiple robotic agents operate independently while remaining connected to a human supervisor. We show how a RAG-powered large language model, augmented with knowledge graph data and domain taxonomy, enables autonomous multi-agent decision-making and facilitates seamless human-robot interaction, resulting in 100\% mission validation and behavior completeness. Finally, ablation studies reveal that without structured knowledge from the graph and/or taxonomy, the LLM is prone to hallucinations, which can compromise decision quality. 

**Abstract (ZH)**: 机器人平台已成为海上操作不可或缺的部分，通过提供定期且连续的离岸资产访问，如水下基础设施 inspection、环境监测和资源勘探。然而，海底环境的复杂和动态特性，包括有限的能见度、不可预测的洋流和通信限制，提出了重大挑战，要求先进的自主性以确保操作员的信任和监督。解决这些挑战的关键在于知识表示和推理技术，尤其是知识图谱和检索增强生成（RAG）系统，这些技术使机器人能够高效地结构化、检索和解释复杂的环境数据。这些能力使机器人代理能够进行推理、适应并有效应对变化的条件。本文的主要目标是展示多代理自主性和共享自主性，其中多个机器人代理独立操作同时保持与人类监督员的连接。我们展示了如何通过增强的大语言模型和知识图谱数据及领域分类，实现自主多代理决策并促进无缝的人机交互，结果验证率为100%，行为完备性为100%。最后，消融研究显示，没有图表结构化知识和/或分类，大语言模型容易产生幻觉，这可能会影响决策质量。 

---
# Decentralized Uncertainty-Aware Multi-Agent Collision Avoidance With Model Predictive Path Integral 

**Title (ZH)**: 去中心化感知不确定性多智能体碰撞 avoidance ewith 模型预测路径积分 

**Authors**: Stepan Dergachev, Konstantin Yakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2507.20293)  

**Abstract**: Decentralized multi-agent navigation under uncertainty is a complex task that arises in numerous robotic applications. It requires collision avoidance strategies that account for both kinematic constraints, sensing and action execution noise. In this paper, we propose a novel approach that integrates the Model Predictive Path Integral (MPPI) with a probabilistic adaptation of Optimal Reciprocal Collision Avoidance. Our method ensures safe and efficient multi-agent navigation by incorporating probabilistic safety constraints directly into the MPPI sampling process via a Second-Order Cone Programming formulation. This approach enables agents to operate independently using local noisy observations while maintaining safety guarantees. We validate our algorithm through extensive simulations with differential-drive robots and benchmark it against state-of-the-art methods, including ORCA-DD and B-UAVC. Results demonstrate that our approach outperforms them while achieving high success rates, even in densely populated environments. Additionally, validation in the Gazebo simulator confirms its practical applicability to robotic platforms. 

**Abstract (ZH)**: 在不确定性下的去中心化多代理导航是一个复杂的任务，广泛应用于各种机器人应用中。它需要考虑到运动约束、感测和动作执行噪声的碰撞避免策略。在本文中，我们提出了一种新型方法，将模型预测路径积分（MPPI）与最优相互碰撞避免的概率适应方法相结合。通过使用圆锥二次规划形式，我们的方法直接将概率安全约束集成到MPPI采样过程中，从而确保代理在利用局部噪声观测独立操作的同时保持安全保证。我们通过广泛的仿真验证了该算法，并将其与当今最先进的方法（包括ORCA-DD和B-UAVC）进行了基准测试。结果表明，我们的方法在不同环境中实现了更高的成功率，并且在Gazebo仿真器中的验证也证明了其在机器人平台上的实际适用性。 

---
# Tactile-Guided Robotic Ultrasound: Mapping Preplanned Scan Paths for Intercostal Imaging 

**Title (ZH)**: 触觉引导的机器人超声：胸间区域成像的预规划扫描路径映射 

**Authors**: Yifan Zhang, Dianye Huang, Nassir Navab, Zhongliang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20282)  

**Abstract**: Medical ultrasound (US) imaging is widely used in clinical examinations due to its portability, real-time capability, and radiation-free nature. To address inter- and intra-operator variability, robotic ultrasound systems have gained increasing attention. However, their application in challenging intercostal imaging remains limited due to the lack of an effective scan path generation method within the constrained acoustic window. To overcome this challenge, we explore the potential of tactile cues for characterizing subcutaneous rib structures as an alternative signal for ultrasound segmentation-free bone surface point cloud extraction. Compared to 2D US images, 1D tactile-related signals offer higher processing efficiency and are less susceptible to acoustic noise and artifacts. By leveraging robotic tracking data, a sparse tactile point cloud is generated through a few scans along the rib, mimicking human palpation. To robustly map the scanning trajectory into the intercostal space, the sparse tactile bone location point cloud is first interpolated to form a denser representation. This refined point cloud is then registered to an image-based dense bone surface point cloud, enabling accurate scan path mapping for individual patients. Additionally, to ensure full coverage of the object of interest, we introduce an automated tilt angle adjustment method to visualize structures beneath the bone. To validate the proposed method, we conducted comprehensive experiments on four distinct phantoms. The final scanning waypoint mapping achieved Mean Nearest Neighbor Distance (MNND) and Hausdorff distance (HD) errors of 3.41 mm and 3.65 mm, respectively, while the reconstructed object beneath the bone had errors of 0.69 mm and 2.2 mm compared to the CT ground truth. 

**Abstract (ZH)**: 基于触觉线索的肋骨结构表征在无创骨表面点云提取中的应用：一种用于胸间影像的新方法 

---
# Humanoid Occupancy: Enabling A Generalized Multimodal Occupancy Perception System on Humanoid Robots 

**Title (ZH)**: humanoid 住人率: 使仿人机器人具备通用多模态住人感知系统 

**Authors**: Wei Cui, Haoyu Wang, Wenkang Qin, Yijie Guo, Gang Han, Wen Zhao, Jiahang Cao, Zhang Zhang, Jiaru Zhong, Jingkai Sun, Pihai Sun, Shuai Shi, Botuo Jiang, Jiahao Ma, Jiaxu Wang, Hao Cheng, Zhichao Liu, Yang Wang, Zheng Zhu, Guan Huang, Jian Tang, Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20217)  

**Abstract**: Humanoid robot technology is advancing rapidly, with manufacturers introducing diverse heterogeneous visual perception modules tailored to specific scenarios. Among various perception paradigms, occupancy-based representation has become widely recognized as particularly suitable for humanoid robots, as it provides both rich semantic and 3D geometric information essential for comprehensive environmental understanding. In this work, we present Humanoid Occupancy, a generalized multimodal occupancy perception system that integrates hardware and software components, data acquisition devices, and a dedicated annotation pipeline. Our framework employs advanced multi-modal fusion techniques to generate grid-based occupancy outputs encoding both occupancy status and semantic labels, thereby enabling holistic environmental understanding for downstream tasks such as task planning and navigation. To address the unique challenges of humanoid robots, we overcome issues such as kinematic interference and occlusion, and establish an effective sensor layout strategy. Furthermore, we have developed the first panoramic occupancy dataset specifically for humanoid robots, offering a valuable benchmark and resource for future research and development in this domain. The network architecture incorporates multi-modal feature fusion and temporal information integration to ensure robust perception. Overall, Humanoid Occupancy delivers effective environmental perception for humanoid robots and establishes a technical foundation for standardizing universal visual modules, paving the way for the widespread deployment of humanoid robots in complex real-world scenarios. 

**Abstract (ZH)**: 基于 occupancy 表征的类人机器人多模态环境感知系统 

---
# A real-time full-chain wearable sensor-based musculoskeletal simulation: an OpenSim-ROS Integration 

**Title (ZH)**: 基于穿戴传感器的实时全链条肌骨仿真：OpenSim-ROS集成 

**Authors**: Frederico Belmonte Klein, Zhaoyuan Wan, Huawei Wang, Ruoli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20049)  

**Abstract**: Musculoskeletal modeling and simulations enable the accurate description and analysis of the movement of biological systems with applications such as rehabilitation assessment, prosthesis, and exoskeleton design. However, the widespread usage of these techniques is limited by costly sensors, laboratory-based setups, computationally demanding processes, and the use of diverse software tools that often lack seamless integration. In this work, we address these limitations by proposing an integrated, real-time framework for musculoskeletal modeling and simulations that leverages OpenSimRT, the robotics operating system (ROS), and wearable sensors. As a proof-of-concept, we demonstrate that this framework can reasonably well describe inverse kinematics of both lower and upper body using either inertial measurement units or fiducial markers. Additionally, we show that it can effectively estimate inverse dynamics of the ankle joint and muscle activations of major lower limb muscles during daily activities, including walking, squatting and sit to stand, stand to sit when combined with pressure insoles. We believe this work lays the groundwork for further studies with more complex real-time and wearable sensor-based human movement analysis systems and holds potential to advance technologies in rehabilitation, robotics and exoskeleton designs. 

**Abstract (ZH)**: 基于OpenSimRT、ROS和可穿戴传感器的集成实时肌骨建模与仿真框架及其应用 

---
# Digital and Robotic Twinning for Validation of Proximity Operations and Formation Flying 

**Title (ZH)**: 数字孪生与机器人孪生验证近距离操作与编队飞行 

**Authors**: Aviad Golan, Gregory Zin, Zahra Ahmed, Emily Bates, Toby Bell, Pol Francesch Huc, Samuel Y. W. Low, Juergen Bosse, Simone D'Amico  

**Link**: [PDF](https://arxiv.org/pdf/2507.20034)  

**Abstract**: In spacecraft Rendezvous, Proximity Operations (RPO), and Formation Flying (FF), the Guidance Navigation and Control (GNC) system is safety-critical and must meet strict performance requirements. However, validating such systems is challenging due to the complexity of the space environment, necessitating a verification and validation (V&V) process that bridges simulation and real-world behavior. The key contribution of this paper is a unified, end-to-end digital and robotic twinning framework that enables software- and hardware-in-the-loop testing for multi-modal GNC systems. The robotic twin includes three testbeds at Stanford's Space Rendezvous Laboratory (SLAB): the GNSS and Radiofrequency Autonomous Navigation Testbed for Distributed Space Systems (GRAND) to validate RF-based navigation techniques, and the Testbed for Rendezvous and Optical Navigation (TRON) and Optical Stimulator (OS) to validate vision-based methods. The test article for this work is an integrated multi-modal GNC software stack for RPO and FF developed at SLAB. This paper introduces the hybrid framework and summarizes calibration and error characterization for the robotic twin. Then, the GNC stack's performance and robustness is characterized using the integrated digital and robotic twinning pipeline for a full-range RPO mission scenario in Low-Earth Orbit (LEO). The results shown in the paper demonstrate consistency between digital and robotic twins, validating the hybrid twinning pipeline as a reliable framework for realistic assessment and verification of GNC systems. 

**Abstract (ZH)**: 基于数字与机器人镜像的综合GNC系统验证与验证框架：低地球轨道 proximity 操作任务场景验证 

---
# When Engineering Outruns Intelligence: A Re-evaluation of Instruction-Guided Navigation 

**Title (ZH)**: 当工程超越智能：指令引导导航的重新评估 

**Authors**: Matin Aghaei, Mohammad Ali Alomrani, Yingxue Zhang, Mahdi Biparva  

**Link**: [PDF](https://arxiv.org/pdf/2507.20021)  

**Abstract**: Large language models (LLMs) are often credited with recent leaps in ObjectGoal Navigation, yet the extent to which they improve planning remains unclear. We revisit this question on the HM3D-v1 validation split. First, we strip InstructNav of its Dynamic Chain-of-Navigation prompt, open-vocabulary GLEE detector and Intuition saliency map, and replace them with a simple Distance-Weighted Frontier Explorer (DWFE). This geometry-only heuristic raises Success from 58.0% to 61.1% and lifts SPL from 20.9% to 36.0% over 2 000 validation episodes, outperforming all previous training-free baselines. Second, we add a lightweight language prior (SHF); on a 200-episode subset this yields a further +2% Success and +0.9% SPL while shortening paths by five steps on average. Qualitative trajectories confirm the trend: InstructNav back-tracks and times-out, DWFE reaches the goal after a few islands, and SHF follows an almost straight route. Our results indicate that frontier geometry, not emergent LLM reasoning, drives most reported gains, and suggest that metric-aware prompts or offline semantic graphs are necessary before attributing navigation success to "LLM intelligence." 

**Abstract (ZH)**: 大型语言模型在对象目标导航领域的规划改进程度尚不明确：HM3D-v1验证集上的 revisit 

---
# SuperMag: Vision-based Tactile Data Guided High-resolution Tactile Shape Reconstruction for Magnetic Tactile Sensors 

**Title (ZH)**: SuperMag：基于视觉的触觉数据指导的磁性触觉传感器高分辨率触觉形状重建 

**Authors**: Peiyao Hou, Danning Sun, Meng Wang, Yuzhe Huang, Zeyu Zhang, Hangxin Liu, Wanlin Li, Ziyuan Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20002)  

**Abstract**: Magnetic-based tactile sensors (MBTS) combine the advantages of compact design and high-frequency operation but suffer from limited spatial resolution due to their sparse taxel arrays. This paper proposes SuperMag, a tactile shape reconstruction method that addresses this limitation by leveraging high-resolution vision-based tactile sensor (VBTS) data to supervise MBTS super-resolution. Co-designed, open-source VBTS and MBTS with identical contact modules enable synchronized data collection of high-resolution shapes and magnetic signals via a symmetric calibration setup. We frame tactile shape reconstruction as a conditional generative problem, employing a conditional variational auto-encoder to infer high-resolution shapes from low-resolution MBTS inputs. The MBTS achieves a sampling frequency of 125 Hz, whereas the shape reconstruction sustains an inference time within 2.5 ms. This cross-modality synergy advances tactile perception of the MBTS, potentially unlocking its new capabilities in high-precision robotic tasks. 

**Abstract (ZH)**: 基于磁性的触觉传感器（MBTS）结合了紧凑设计和高频操作的优势，但由于传感器单元稀疏排列导致空间分辨率受限。本文提出了一种名为SuperMag的触觉形状重建方法，通过利用高分辨率视觉导向的触觉传感器（VBTS）数据来监督MBTS的超分辨率，从而解决这一限制。设计一致的开源VBTS和MBTS配合对称标定设置，实现同步采集高分辨率形状和磁信号的数据。我们将触觉形状重建框架化为一个条件生成问题，采用条件变分自编码器从低分辨率MBTS输入中推断出高分辨率形状。MBTS实现了125 Hz的采样频率，而形状重建的推理时间保持在2.5 ms以内。这种跨模态协同作用提升了一体化触觉传感器的触觉感知能力，可能为其在高精度机器人任务中解锁新的能力。 

---
# Robot Excavation and Manipulation of Geometrically Cohesive Granular Media 

**Title (ZH)**: 机器人挖掘与操纵几何结合性粒状介质 

**Authors**: Laura Treers, Daniel Soto, Joonha Hwang, Michael A. D. Goodisman, Daniel I. Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2507.19999)  

**Abstract**: Construction throughout history typically assumes that its blueprints and building blocks are pre-determined. However, recent work suggests that alternative approaches can enable new paradigms for structure formation. Aleatory architectures, or those which rely on the properties of their granular building blocks rather than pre-planned design or computation, have thus far relied on human intervention for their creation. We imagine that robotic swarms could be valuable to create such aleatory structures by manipulating and forming structures from entangled granular materials. To discover principles by which robotic systems can effectively manipulate soft matter, we develop a robophysical model for interaction with geometrically cohesive granular media composed of u-shape particles. This robotic platform uses environmental signals to autonomously coordinate excavation, transport, and deposition of material. We test the effect of substrate initial conditions by characterizing robot performance in two different material compaction states and observe as much as a 75% change in transported mass depending on initial substrate compressive loading. These discrepancies suggest the functional role that material properties such as packing and cohesion/entanglement play in excavation and construction. To better understand these material properties, we develop an apparatus for tensile testing of the geometrically cohesive substrates, which reveals how entangled material strength responds strongly to initial compressive loading. These results explain the variation observed in robotic performance and point to future directions for better understanding robotic interaction mechanics with entangled materials. 

**Abstract (ZH)**: 历史上的建筑通常假设其Blueprints和Building Blocks是预先确定的。然而，近期的研究表明，替代方法可以为结构形成开辟新的范式。依赖于其颗粒构建块的性质而非预先计划的设计或计算的Aleatory架构，迄今为止仍需要人工干预来创建。我们设想，通过操纵和形成纠缠的颗粒材料，机器人 swarm 可以成为创建此类Aleatory结构的有价值的工具。为了发现机器人系统有效操纵软物质的原则，我们建立了一个与几何结合的颗粒介质相互作用的robophysical模型，这些介质由U形颗粒组成。这个机器人平台使用环境信号自主协调材料的挖掘、运输和沉积。通过表征机器人在两种不同的材料密实状态下的性能，我们观察到所运输的物质质量最多可变化75%，这表明初态基材压缩载荷在挖掘和建造中的功能作用。为了更好地理解这些材料性质，我们开发了一种几何结合基材的拉伸测试装置，揭示了纠缠材料强度对初态压缩载荷的强烈响应。这些结果解释了观察到的机器人性能变化，并指出了未来更好地理解机器人与纠缠材料相互作用力学的途径。 

---
# CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints 

**Title (ZH)**: CLASP: 基于语义关键点的一般性衣物 manipulation 方法 

**Authors**: Yuhong Deng, Chao Tang, Cunjun Yu, Linfeng Li, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19983)  

**Abstract**: Clothes manipulation, such as folding or hanging, is a critical capability for home service robots. Despite recent advances, most existing methods remain limited to specific tasks and clothes types, due to the complex, high-dimensional geometry of clothes. This paper presents CLothes mAnipulation with Semantic keyPoints (CLASP), which aims at general-purpose clothes manipulation over different clothes types, T-shirts, shorts, skirts, long dresses, ... , as well as different tasks, folding, flattening, hanging, ... . The core idea of CLASP is semantic keypoints -- e.g., ''left sleeve'', ''right shoulder'', etc. -- a sparse spatial-semantic representation that is salient for both perception and action. Semantic keypoints of clothes can be reliably extracted from RGB-D images and provide an effective intermediate representation of clothes manipulation policies. CLASP uses semantic keypoints to bridge high-level task planning and low-level action execution. At the high level, it exploits vision language models (VLMs) to predict task plans over the semantic keypoints. At the low level, it executes the plans with the help of a simple pre-built manipulation skill library. Extensive simulation experiments show that CLASP outperforms state-of-the-art baseline methods on multiple tasks across diverse clothes types, demonstrating strong performance and generalization. Further experiments with a Franka dual-arm system on four distinct tasks -- folding, flattening, hanging, and placing -- confirm CLASP's performance on a real robot. 

**Abstract (ZH)**: Clothes mAnipulation with Semantic KeyPoints (CLASP):通用衣物操作方法 

---
# A roadmap for AI in robotics 

**Title (ZH)**: AI在机器人领域的应用 roadmap 

**Authors**: Aude Billard, Alin Albu-Schaeffer, Michael Beetz, Wolfram Burgard, Peter Corke, Matei Ciocarlie, Ravinder Dahiya, Danica Kragic, Ken Goldberg, Yukie Nagai, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2507.19975)  

**Abstract**: AI technologies, including deep learning, large-language models have gone from one breakthrough to the other. As a result, we are witnessing growing excitement in robotics at the prospect of leveraging the potential of AI to tackle some of the outstanding barriers to the full deployment of robots in our daily lives. However, action and sensing in the physical world pose greater and different challenges than analysing data in isolation. As the development and application of AI in robotic products advances, it is important to reflect on which technologies, among the vast array of network architectures and learning models now available in the AI field, are most likely to be successfully applied to robots; how they can be adapted to specific robot designs, tasks, environments; which challenges must be overcome. This article offers an assessment of what AI for robotics has achieved since the 1990s and proposes a short- and medium-term research roadmap listing challenges and promises. These range from keeping up-to-date large datasets, representatives of a diversity of tasks robots may have to perform, and of environments they may encounter, to designing AI algorithms tailored specifically to robotics problems but generic enough to apply to a wide range of applications and transfer easily to a variety of robotic platforms. For robots to collaborate effectively with humans, they must predict human behavior without relying on bias-based profiling. Explainability and transparency in AI-driven robot control are not optional but essential for building trust, preventing misuse, and attributing responsibility in accidents. We close on what we view as the primary long-term challenges, that is, to design robots capable of lifelong learning, while guaranteeing safe deployment and usage, and sustainable computational costs. 

**Abstract (ZH)**: 人工智能技术，包括深度学习和大型语言模型，已从一个突破走向另一个突破。随着人工智能潜力在机器人领域得到更广泛的应用，我们目睹了机器人领域日益增长的兴奋感。然而，物理世界的行动与感知比单独分析数据带来了更大的挑战。随着人工智能在机器人产品中的发展和应用，重要的是反思在当前人工智能领域中可用的各种网络架构和学习模型中，哪些技术最有可能成功应用于机器人；如何适应特定的机器人设计、任务和环境；以及必须克服哪些挑战。本文评估了自20世纪90年代以来机器人领域人工智能的发展成就，并提出了一份短期和中期研究路线图，列出了挑战和前景。这些挑战和前景范围从更新大型数据集，包括机器人可能执行的各种任务和可能遇到的各种环境的代表，到为机器人问题量身定制的AI算法，同时具有广泛的适用性和易于转移到各种机器人平台的能力。为了使机器人有效地与人类协作，它们必须预测人类行为，而不依赖于基于偏见的画像。以人工智能驱动的机器人控制的可解释性和透明性不仅是可选的，而是建立信任、防止误用和在事故中归咎责任必不可少的。最后，我们概述了我们认为主要的长期挑战，即设计能够实现终身学习的机器人，同时确保安全部署和使用，并具有可持续的计算成本。 

---
# Spatial Language Likelihood Grounding Network for Bayesian Fusion of Human-Robot Observations 

**Title (ZH)**: 基于空间语言likelihood grounding的贝叶斯融合人类-机器人观测网络 

**Authors**: Supawich Sitdhipol, Waritwong Sukprasongdee, Ekapol Chuangsuwanich, Rina Tse  

**Link**: [PDF](https://arxiv.org/pdf/2507.19947)  

**Abstract**: Fusing information from human observations can help robots overcome sensing limitations in collaborative tasks. However, an uncertainty-aware fusion framework requires a grounded likelihood representing the uncertainty of human inputs. This paper presents a Feature Pyramid Likelihood Grounding Network (FP-LGN) that grounds spatial language by learning relevant map image features and their relationships with spatial relation semantics. The model is trained as a probability estimator to capture aleatoric uncertainty in human language using three-stage curriculum learning. Results showed that FP-LGN matched expert-designed rules in mean Negative Log-Likelihood (NLL) and demonstrated greater robustness with lower standard deviation. Collaborative sensing results demonstrated that the grounded likelihood successfully enabled uncertainty-aware fusion of heterogeneous human language observations and robot sensor measurements, achieving significant improvements in human-robot collaborative task performance. 

**Abstract (ZH)**: 融合人类观察信息可以幫助机器人在协作任务中克服传感限制。然而，一个具备不确定性的融合框架需要一个基于地面的likelihood来表示人类输入的不确定性。本文提出了一种特征金字塔可能性接地网络（FP-LGN），该网络通过学习与空间关系语义相关的地图图像特征及其关系来接地空间语言。模型利用三阶段 Curriculum 学习作为概率估计器，捕捉人类语言中的偶然不确定性。结果显示，FP-LGN 在平均负对数似然（NLL）上与专家设计的规则相匹配，并且表现出更低的标准差，从而提高了鲁棒性。协作感知结果表明，接地的似然性成功地使异质人类语言观察和机器人传感器测量的不确定性感知融合成为可能，显著提高了人机协作任务的性能。 

---
# High-Speed Event Vision-Based Tactile Roller Sensor for Large Surface Measurements 

**Title (ZH)**: 基于事件视觉的高速触觉滚轮传感器用于大型表面测量 

**Authors**: Akram Khairi, Hussain Sajwani, Abdallah Mohammad Alkilany, Laith AbuAssi, Mohamad Halwani, Islam Mohamed Zaid, Ahmed Awadalla, Dewald Swart, Abdulla Ayyad, Yahya Zweiri  

**Link**: [PDF](https://arxiv.org/pdf/2507.19914)  

**Abstract**: Inspecting large-scale industrial surfaces like aircraft fuselages for quality control requires capturing their precise 3D surface geometry at high resolution. Vision-based tactile sensors (VBTSs) offer high local resolution but require slow 'press-and-lift' measurements stitched for large areas. Approaches with sliding or roller/belt VBTS designs provide measurements continuity. However, they face significant challenges respectively: sliding struggles with friction/wear and both approaches are speed-limited by conventional camera frame rates and motion blur, making large-area scanning time consuming. Thus, a rapid, continuous, high-resolution method is needed. We introduce a novel tactile sensor integrating a neuromorphic camera in a rolling mechanism to achieve this. Leveraging its high temporal resolution and robustness to motion blur, our system uses a modified event-based multi-view stereo approach for 3D reconstruction. We demonstrate state-of-the-art scanning speeds up to 0.5 m/s, achieving Mean Absolute Error below 100 microns -- 11 times faster than prior continuous tactile sensing methods. A multi-reference Bayesian fusion strategy enhances accuracy (reducing MAE by 25.2\% compared to EMVS) and mitigates curvature errors. We also validate high-speed feature recognition via Braille reading 2.6 times faster than previous approaches. 

**Abstract (ZH)**: 大规模工业表面如飞机 fuselages 的质量控制需要捕获其高分辨率的精确 3D 表面几何。视觉基触觉传感器（VBTS）提供局部高分辨率，但需要缓慢的“压下并抬起”测量，并通过拼接来进行大面积测量。滑动或滚轮/皮带 VBTS 设计的方案可以提供连续测量，但分别面临显著挑战：滑动难以处理摩擦/磨损，两种方法均由于传统相机帧率和运动模糊速度受限，导致大面积扫描耗时。因此，需要一种快速、连续、高分辨率的方法。我们提出了一种结合神经形态相机的滚动机制触觉传感器，以实现这一目标。利用其高时间分辨率和对运动模糊的鲁棒性，我们的系统采用改进的事件驱动多视图立体匹配方法进行 3D 重建。我们展示了高达 0.5 m/s 的扫描速度，平均绝对误差低于 100 微米——比先前的连续触觉传感方法快 11 倍。多参考贝叶斯融合策略提高了准确性（将 MAE 减少 25.2% 相较于 EMVS），并减轻了曲率误差。我们还通过布莱叶盲文阅读验证了高速特征识别，速度比之前的方法快 2.6 倍。 

---
# Bridging Simulation and Usability: A User-Friendly Framework for Scenario Generation in CARLA 

**Title (ZH)**: Simulation与可用性之间的桥梁：一个用户友好的CARLA场景生成框架 

**Authors**: Ahmed Abouelazm, Mohammad Mahmoud, Conrad Walter, Oleksandr Shchetsura, Erne Hussong, Helen Gremmelmaier, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2507.19883)  

**Abstract**: Autonomous driving promises safer roads, reduced congestion, and improved mobility, yet validating these systems across diverse conditions remains a major challenge. Real-world testing is expensive, time-consuming, and sometimes unsafe, making large-scale validation impractical. In contrast, simulation environments offer a scalable and cost-effective alternative for rigorous verification and validation. A critical component of the validation process is scenario generation, which involves designing and configuring traffic scenarios to evaluate autonomous systems' responses to various events and uncertainties. However, existing scenario generation tools often require programming knowledge, limiting accessibility for non-technical users. To address this limitation, we present an interactive, no-code framework for scenario generation. Our framework features a graphical interface that enables users to create, modify, save, load, and execute scenarios without needing coding expertise or detailed simulation knowledge. Unlike script-based tools such as Scenic or ScenarioRunner, our approach lowers the barrier to entry and supports a broader user base. Central to our framework is a graph-based scenario representation that facilitates structured management, supports both manual and automated generation, and enables integration with deep learning-based scenario and behavior generation methods. In automated mode, the framework can randomly sample parameters such as actor types, behaviors, and environmental conditions, allowing the generation of diverse and realistic test datasets. By simplifying the scenario generation process, this framework supports more efficient testing workflows and increases the accessibility of simulation-based validation for researchers, engineers, and policymakers. 

**Abstract (ZH)**: 自主驾驶 promises 更安全的道路、减少拥堵和提高流动性，然而在多样条件下验证这些系统依然是一项重大挑战。现实世界的测试昂贵、耗时且有时不安全，使得大规模验证变得不切实际。相比之下，仿真环境提供了严格的验证和验证的可扩展且成本效益高的替代方案。验证过程中的关键组成部分是情景生成，这涉及设计和配置交通情景以评估自主系统对各种事件和不确定性的响应。然而，现有的情景生成工具通常需要编程知识，限制了非技术人员的访问。为了克服这一限制，我们提出了一种无需编码的交互式框架来进行情景生成。该框架的特点是图形界面，允许用户无需编程专业知识或详细的仿真知识即可创建、修改、保存、加载和执行情景。与基于脚本的工具如Scenic或ScenarioRunner不同，我们的方法降低了入门门槛并支持更广泛的用户群体。我们框架的核心是一个基于图的情景表示，有助于结构化管理，支持手工和自动生成，并允许与基于深度学习的情景和行为生成方法集成。在自动模式下，该框架可以随机采样诸如角色类型、行为和环境条件等参数，从而生成多样且真实的测试数据集。通过简化情景生成过程，该框架支持更高效的测试工作流并增加了基于仿真的验证方法在研究人员、工程师和决策者中的可访问性。 

---
# Homotopy-aware Multi-agent Navigation via Distributed Model Predictive Control 

**Title (ZH)**: 基于同伦感知的分布式模型预测控制多agent导航 

**Authors**: Haoze Dong, Meng Guo, Chengyi He, Zhongkui Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.19860)  

**Abstract**: Multi-agent trajectory planning requires ensuring both safety and efficiency, yet deadlocks remain a significant challenge, especially in obstacle-dense environments. Such deadlocks frequently occur when multiple agents attempt to traverse the same long and narrow corridor simultaneously. To address this, we propose a novel distributed trajectory planning framework that bridges the gap between global path and local trajectory cooperation. At the global level, a homotopy-aware optimal path planning algorithm is proposed, which fully leverages the topological structure of the environment. A reference path is chosen from distinct homotopy classes by considering both its spatial and temporal properties, leading to improved coordination among agents globally. At the local level, a model predictive control-based trajectory optimization method is used to generate dynamically feasible and collision-free trajectories. Additionally, an online replanning strategy ensures its adaptability to dynamic environments. Simulations and experiments validate the effectiveness of our approach in mitigating deadlocks. Ablation studies demonstrate that by incorporating time-aware homotopic properties into the underlying global paths, our method can significantly reduce deadlocks and improve the average success rate from 4%-13% to over 90% in randomly generated dense scenarios. 

**Abstract (ZH)**: 多智能体轨迹规划需要同时保证安全性和效率，但在障碍密集环境中，死锁仍然是一个重大挑战。当多个智能体试图同时穿越相同的长而窄的走廊时，死锁经常发生。为此，我们提出了一种新颖的分布式轨迹规划框架，以弥合全局路径和局部轨迹合作之间的差距。在全局层面，提出了一种同伦意识最优路径规划算法，充分利用环境的拓扑结构。通过同时考虑参考路径的空间和时间特性，从不同的同伦类中选择路径，从而实现智能体在全局范围内的更好协调。在局部层面，采用模型预测控制为基础的轨迹优化方法生成动态可行且无碰撞的轨迹。此外，采用在线重规划策略确保其在动态环境中的适应性。仿真和实验验证了我们方法在缓解死锁方面的有效性。消融研究显示，通过将时间意识的同伦特性融入底层全局路径中，我们的方法可以显著减少死锁，并将随机生成的密集场景中的平均成功率从4%-13%提高到超过90%。 

---
# Think, Act, Learn: A Framework for Autonomous Robotic Agents using Closed-Loop Large Language Models 

**Title (ZH)**: 思考、行动、学习：一种基于闭环大型语言模型的自主机器人框架 

**Authors**: Anjali R. Menon, Rohit K. Sharma, Priya Singh, Chengyu Wang, Aurora M. Ferreira, Mateja Novak  

**Link**: [PDF](https://arxiv.org/pdf/2507.19854)  

**Abstract**: The integration of Large Language Models (LLMs) into robotics has unlocked unprecedented capabilities in high-level task planning. However, most current systems operate in an open-loop fashion, where LLMs act as one-shot planners, rendering them brittle and unable to adapt to unforeseen circumstances in dynamic physical environments. To overcome this limitation, this paper introduces the "Think, Act, Learn" (T-A-L) framework, a novel architecture that enables an embodied agent to autonomously learn and refine its policies through continuous interaction. Our framework establishes a closed-loop cycle where an LLM first "thinks" by decomposing high-level commands into actionable plans. The robot then "acts" by executing these plans while gathering rich, multimodal sensory feedback. Critically, the "learn" module processes this feedback to facilitate LLM-driven self-reflection, allowing the agent to perform causal analysis on its failures and generate corrective strategies. These insights are stored in an experiential memory to guide future planning cycles. We demonstrate through extensive experiments in both simulation and the real world that our T-A-L agent significantly outperforms baseline methods, including open-loop LLMs, Behavioral Cloning, and traditional Reinforcement Learning. Our framework achieves over a 97% success rate on complex, long-horizon tasks, converges to a stable policy in an average of just 9 trials, and exhibits remarkable generalization to unseen tasks. This work presents a significant step towards developing more robust, adaptive, and truly autonomous robotic agents. 

**Abstract (ZH)**: Large Language Models Integrated into Robotics: The "Think, Act, Learn" Framework for Autonomous Policy Refinement 

---
# PlaneHEC: Efficient Hand-Eye Calibration for Multi-view Robotic Arm via Any Point Cloud Plane Detection 

**Title (ZH)**: PlaneHEC: 基于任意点云平面检测的多视点机器人手臂高效手眼标定 

**Authors**: Ye Wang, Haodong Jing, Yang Liao, Yongqiang Ma, Nanning Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19851)  

**Abstract**: Hand-eye calibration is an important task in vision-guided robotic systems and is crucial for determining the transformation matrix between the camera coordinate system and the robot end-effector. Existing methods, for multi-view robotic systems, usually rely on accurate geometric models or manual assistance, generalize poorly, and can be very complicated and inefficient. Therefore, in this study, we propose PlaneHEC, a generalized hand-eye calibration method that does not require complex models and can be accomplished using only depth cameras, which achieves the optimal and fastest calibration results using arbitrary planar surfaces like walls and tables. PlaneHEC introduces hand-eye calibration equations based on planar constraints, which makes it strongly interpretable and generalizable. PlaneHEC also uses a comprehensive solution that starts with a closed-form solution and improves it withiterative optimization, which greatly improves accuracy. We comprehensively evaluated the performance of PlaneHEC in both simulated and real-world environments and compared the results with other point-cloud-based calibration methods, proving its superiority. Our approach achieves universal and fast calibration with an innovative design of computational models, providing a strong contribution to the development of multi-agent systems and embodied intelligence. 

**Abstract (ZH)**: 平面Hand-eye标定方法：PlaneHEC 

---
# Feeling the Force: A Nuanced Physics-based Traversability Sensor for Navigation in Unstructured Vegetation 

**Title (ZH)**: 感知力：一种细腻的基于物理的可通行性传感器，用于不规则植被中的导航 

**Authors**: Zaar Khizar, Johann Laconte, Roland Lenain, Romuald Aufrere  

**Link**: [PDF](https://arxiv.org/pdf/2507.19831)  

**Abstract**: In many applications, robots are increasingly deployed in unstructured and natural environments where they encounter various types of vegetation. Vegetation presents unique challenges as a traversable obstacle, where the mechanical properties of the plants can influence whether a robot can safely collide with and overcome the obstacle. A more nuanced approach is required to assess the safety and traversability of these obstacles, as collisions can sometimes be safe and necessary for navigating through dense or unavoidable vegetation. This paper introduces a novel sensor designed to directly measure the applied forces exerted by vegetation on a robot: by directly capturing the push-back forces, our sensor provides a detailed understanding of the interactions between the robot and its surroundings. We demonstrate the sensor's effectiveness through experimental validations, showcasing its ability to measure subtle force variations. This force-based approach provides a quantifiable metric that can inform navigation decisions and serve as a foundation for developing future learning algorithms. 

**Abstract (ZH)**: 基于植被的推力测量的新型传感器及其应用：评估机器人穿越植被障碍的安全性和可行性 

---
# A 4D Radar Camera Extrinsic Calibration Tool Based on 3D Uncertainty Perspective N Points 

**Title (ZH)**: 基于三维不确定视角N点的4D雷达摄像机外参标定工具 

**Authors**: Chuan Cao, Xiaoning Wang, Wenqian Xi, Han Zhang, Weidong Chen, Jingchuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19829)  

**Abstract**: 4D imaging radar is a type of low-cost millimeter-wave radar(costing merely 10-20$\%$ of lidar systems) capable of providing range, azimuth, elevation, and Doppler velocity information. Accurate extrinsic calibration between millimeter-wave radar and camera systems is critical for robust multimodal perception in robotics, yet remains challenging due to inherent sensor noise characteristics and complex error propagation. This paper presents a systematic calibration framework to address critical challenges through a spatial 3d uncertainty-aware PnP algorithm (3DUPnP) that explicitly models spherical coordinate noise propagation in radar measurements, then compensating for non-zero error expectations during coordinate transformations. Finally, experimental validation demonstrates significant performance improvements over state-of-the-art CPnP baseline, including improved consistency in simulations and enhanced precision in physical experiments. This study provides a robust calibration solution for robotic systems equipped with millimeter-wave radar and cameras, tailored specifically for autonomous driving and robotic perception applications. 

**Abstract (ZH)**: 4D成像雷达是一种低成本毫米波雷达（成本仅为激光雷达系统的10-20%），能够提供距离、方位、仰角和多普勒速度信息。毫米波雷达与摄像头系统的准确外部校准对于机器人鲁棒多模感知至关重要，但由于固有传感器噪声特性及复杂误差传播，这一任务仍然具有挑战性。本文提出了一种系统化的校准框架，通过空间三维不确定性感知PnP算法（3DUPnP），明确建模雷达测距中的球坐标噪声传播，并在坐标变换过程中补偿非零误差期望。实验验证显示，该方法在仿真和物理实验中均显著优于最先进的CPnP基线，包括一致性改进和精度提高。本研究为配备毫米波雷达和摄像头的机器人系统提供了一种鲁棒校准解决方案，特别适用于自主驾驶和机器人感知应用。 

---
# Ag2x2: Robust Agent-Agnostic Visual Representations for Zero-Shot Bimanual Manipulation 

**Title (ZH)**: Ag2x2: 基于稳健的、代理无关的视觉表示的零样本双臂操作 

**Authors**: Ziyin Xiong, Yinghan Chen, Puhao Li, Yixin Zhu, Tengyu Liu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19817)  

**Abstract**: Bimanual manipulation, fundamental to human daily activities, remains a challenging task due to its inherent complexity of coordinated control. Recent advances have enabled zero-shot learning of single-arm manipulation skills through agent-agnostic visual representations derived from human videos; however, these methods overlook crucial agent-specific information necessary for bimanual coordination, such as end-effector positions. We propose Ag2x2, a computational framework for bimanual manipulation through coordination-aware visual representations that jointly encode object states and hand motion patterns while maintaining agent-agnosticism. Extensive experiments demonstrate that Ag2x2 achieves a 73.5% success rate across 13 diverse bimanual tasks from Bi-DexHands and PerAct2, including challenging scenarios with deformable objects like ropes. This performance outperforms baseline methods and even surpasses the success rate of policies trained with expert-engineered rewards. Furthermore, we show that representations learned through Ag2x2 can be effectively leveraged for imitation learning, establishing a scalable pipeline for skill acquisition without expert supervision. By maintaining robust performance across diverse tasks without human demonstrations or engineered rewards, Ag2x2 represents a step toward scalable learning of complex bimanual robotic skills. 

**Abstract (ZH)**: 双臂操作，对于人类日常活动至关重要，但由于其固有的协调控制复杂性，仍然是一项具有挑战性的任务。近期进展通过从人类视频中提取的代理无关视觉表示使单臂操作技能实现了零样本学习；然而，这些方法忽略了对于双臂协调至关重要的代理特定信息，如末端执行器位置。我们提出了一种名为Ag2x2的计算框架，该框架通过感知代理意识的视觉表示同时编码物体状态和手部运动模式，同时保持代理无关性。大量的实验表明，Ag2x2在Bi-DexHands和PerAct2的13种不同双臂任务中取得了73.5%的成功率，包括涉及可变形物体（如绳索）的具有挑战性的场景。该性能超越了基线方法，并甚至超过了使用专家设计奖励训练的策略的成功率。此外，我们展示了通过Ag2x2学习的表示可以有效应用于模仿学习，建立了一种在无专家监督的情况下可扩展的技能获取管道。通过在不同任务中保持稳健的性能，无需人类示范或工程化奖励，Ag2x2代表了复杂双臂机器人技能可扩展学习的一个重要进展。 

---
# Skin-Machine Interface with Multimodal Contact Motion Classifier 

**Title (ZH)**: 多模态接触运动分类的皮肤-机器接口 

**Authors**: Alberto Confente, Takanori Jin, Taisuke Kobayashi, Julio Rogelio Guadarrama-Olvera, Gordon Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19760)  

**Abstract**: This paper proposes a novel framework for utilizing skin sensors as a new operation interface of complex robots. The skin sensors employed in this study possess the capability to quantify multimodal tactile information at multiple contact points. The time-series data generated from these sensors is anticipated to facilitate the classification of diverse contact motions exhibited by an operator. By mapping the classification results with robot motion primitives, a diverse range of robot motions can be generated by altering the manner in which the skin sensors are interacted with. In this paper, we focus on a learning-based contact motion classifier employing recurrent neural networks. This classifier is a pivotal factor in the success of this framework. Furthermore, we elucidate the requisite conditions for software-hardware designs. Firstly, multimodal sensing and its comprehensive encoding significantly contribute to the enhancement of classification accuracy and learning stability. Utilizing all modalities simultaneously as inputs to the classifier proves to be an effective approach. Secondly, it is essential to mount the skin sensors on a flexible and compliant support to enable the activation of three-axis accelerometers. These accelerometers are capable of measuring horizontal tactile information, thereby enhancing the correlation with other modalities. Furthermore, they serve to absorb the noises generated by the robot's movements during deployment. Through these discoveries, the accuracy of the developed classifier surpassed 95 %, enabling the dual-arm mobile manipulator to execute a diverse range of tasks via the Skin-Machine Interface. this https URL 

**Abstract (ZH)**: 本文提出了一种利用皮肤传感器作为复杂机器人新型操作界面的新型框架。研究中使用的皮肤传感器具备在多个接触点量化多模态触觉信息的能力。这些传感器生成的时间序列数据有望便于识别操作员展现的多种接触运动。通过将分类结果映射到机器人运动基本元素，可以通过改变与皮肤传感器的交互方式生成多种多样的机器人运动。在本文中，我们专注于利用循环神经网络的基于学习的接触运动分类器。该分类器是该框架成功的关键因素之一。此外，我们阐述了软硬件设计的必要条件。首先，多模态感知及其综合编码显著提高了分类准确率和学习稳定性。将所有模态同时作为分类器的输入证明是一种有效的方法。其次，将皮肤传感器安装在柔性且顺应性好的支撑上，可以激活三轴加速度计，这些加速度计能够测量水平触觉信息，从而增强与其他模态的关联性。此外，它们还能够吸收机器人在部署过程中产生的噪音。通过这些发现，开发的分类器准确性超过了95%，使得双臂移动操作器能够通过皮肤-机器界面执行多样化的任务。this https.URL 

---
# DOA: A Degeneracy Optimization Agent with Adaptive Pose Compensation Capability based on Deep Reinforcement Learning 

**Title (ZH)**: DOA：一种基于深度强化学习的自适应姿态补偿退化优化代理 

**Authors**: Yanbin Li, Canran Xiao, Hongyang He, Shenghai Yuan, Zong Ke, Jiajie Yu, Zixiong Qin, Zhiguo Zhang, Wenzheng Chi, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19742)  

**Abstract**: Particle filter-based 2D-SLAM is widely used in indoor localization tasks due to its efficiency. However, indoor environments such as long straight corridors can cause severe degeneracy problems in SLAM. In this paper, we use Proximal Policy Optimization (PPO) to train an adaptive degeneracy optimization agent (DOA) to address degeneracy problem. We propose a systematic methodology to address three critical challenges in traditional supervised learning frameworks: (1) data acquisition bottlenecks in degenerate dataset, (2) inherent quality deterioration of training samples, and (3) ambiguity in annotation protocol design. We design a specialized reward function to guide the agent in developing perception capabilities for degenerate environments. Using the output degeneracy factor as a reference weight, the agent can dynamically adjust the contribution of different sensors to pose optimization. Specifically, the observation distribution is shifted towards the motion model distribution, with the step size determined by a linear interpolation formula related to the degeneracy factor. In addition, we employ a transfer learning module to endow the agent with generalization capabilities across different environments and address the inefficiency of training in degenerate environments. Finally, we conduct ablation studies to demonstrate the rationality of our model design and the role of transfer learning. We also compare the proposed DOA with SOTA methods to prove its superior degeneracy detection and optimization capabilities across various environments. 

**Abstract (ZH)**: 基于粒子滤波的2D-SLAM在室内定位任务中广泛应用，但由于其高效性。然而，如长直走廊等室内环境会导致SLAM严重退化问题。本文使用临近策略优化(PPO)训练自适应退化优化代理(DOA)来解决退化问题。我们提出了一种系统方法来应对传统监督学习框架中的三个关键挑战：(1)退化数据集的数据获取瓶颈，(2)训练样本固有的质量退化，以及(3)标注协议设计的模糊性。我们设计了专门的奖励函数来引导代理在退化环境中的感知能力发展。利用输出退化因子作为参考权重，代理可以动态调整不同传感器在姿态优化中的贡献。具体而言，观测分布朝向运动模型分布移动，步长由与退化因子相关的线性插值公式确定。此外，我们采用迁移学习模块赋予代理跨不同环境的泛化能力，并解决退化环境中的训练效率低下问题。最后，我们进行消融研究以证明我们模型设计的合理性及其迁移学习的作用。我们还将所提出的DOA与当前最先进的方法进行比较，以证明其在各种环境中的优越的退化检测和优化能力。 

---
# PhysVarMix: Physics-Informed Variational Mixture Model for Multi-Modal Trajectory Prediction 

**Title (ZH)**: PhysVarMix：物理知情的变分混合模型多模态轨迹预测 

**Authors**: Haichuan Li, Tomi Westerlund  

**Link**: [PDF](https://arxiv.org/pdf/2507.19701)  

**Abstract**: Accurate prediction of future agent trajectories is a critical challenge for ensuring safe and efficient autonomous navigation, particularly in complex urban environments characterized by multiple plausible future scenarios. In this paper, we present a novel hybrid approach that integrates learning-based with physics-based constraints to address the multi-modality inherent in trajectory prediction. Our method employs a variational Bayesian mixture model to effectively capture the diverse range of potential future behaviors, moving beyond traditional unimodal assumptions. Unlike prior approaches that predominantly treat trajectory prediction as a data-driven regression task, our framework incorporates physical realism through sector-specific boundary conditions and Model Predictive Control (MPC)-based smoothing. These constraints ensure that predicted trajectories are not only data-consistent but also physically plausible, adhering to kinematic and dynamic principles. Furthermore, our method produces interpretable and diverse trajectory predictions, enabling enhanced downstream decision-making and planning in autonomous driving systems. We evaluate our approach on two benchmark datasets, demonstrating superior performance compared to existing methods. Comprehensive ablation studies validate the contributions of each component and highlight their synergistic impact on prediction accuracy and reliability. By balancing data-driven insights with physics-informed constraints, our approach offers a robust and scalable solution for navigating the uncertainties of real-world urban environments. 

**Abstract (ZH)**: 基于学习与物理约束的未来代理轨迹准确预测方法：实现复杂城市环境下的高效自主导航 

---
# RAKOMO: Reachability-Aware K-Order Markov Path Optimization for Quadrupedal Loco-Manipulation 

**Title (ZH)**: RAKOMO：可达性感知的四元 Markov 路径优化方法用于四足Manipulation 

**Authors**: Mattia Risiglione, Abdelrahman Abdalla, Victor Barasuol, Kim Tien Ly, Ioannis Havoutis, Claudio Semini  

**Link**: [PDF](https://arxiv.org/pdf/2507.19652)  

**Abstract**: Legged manipulators, such as quadrupeds equipped with robotic arms, require motion planning techniques that account for their complex kinematic constraints in order to perform manipulation tasks both safely and effectively. However, trajectory optimization methods often face challenges due to the hybrid dynamics introduced by contact discontinuities, and tend to neglect leg limitations during planning for computational reasons. In this work, we propose RAKOMO, a path optimization technique that integrates the strengths of K-Order Markov Optimization (KOMO) with a kinematically-aware criterion based on the reachable region defined as reachability margin. We leverage a neural-network to predict the margin and optimize it by incorporating it in the standard KOMO formulation. This approach enables rapid convergence of gradient-based motion planning -- commonly tailored for continuous systems -- while adapting it effectively to legged manipulators, successfully executing loco-manipulation tasks. We benchmark RAKOMO against a baseline KOMO approach through a set of simulations for pick-and-place tasks with the HyQReal quadruped robot equipped with a Kinova Gen3 robotic arm. 

**Abstract (ZH)**: 装有机械臂的腿足 manipulator，如四足机器人，为了安全高效地执行操作任务，需要考虑其复杂的运动约束的动力学规划技术。然而，由于接触断点引入的混合动力学，轨迹优化方法往往会遇到挑战，并且在计算原因上倾向于在规划过程中忽视腿的限制。在这种背景下，我们提出了RAKOMO，一种结合了K-Order Markov Optimization (KOMO) 强点并与可达区域定义为基础的动力学感知标准相结合的方法。我们利用神经网络预测可达区域并将其纳入标准的KOMO公式中进行优化。这种方法能够使基于梯度的运动规划快速收敛，同时有效地适应腿足 manipulator，成功执行行进操作任务。我们通过一系列针对HyQReal四足机器人配以Kinova Gen3机械臂的放置任务仿真实验，将RAKOMO与基准KOMO方法进行对比评估。 

---
# GABRIL: Gaze-Based Regularization for Mitigating Causal Confusion in Imitation Learning 

**Title (ZH)**: GABRIL: 基于凝视正则化的方法以减轻模仿学习中的因果混淆 

**Authors**: Amin Banayeeanzade, Fatemeh Bahrani, Yutai Zhou, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2507.19647)  

**Abstract**: Imitation Learning (IL) is a widely adopted approach which enables agents to learn from human expert demonstrations by framing the task as a supervised learning problem. However, IL often suffers from causal confusion, where agents misinterpret spurious correlations as causal relationships, leading to poor performance in testing environments with distribution shift. To address this issue, we introduce GAze-Based Regularization in Imitation Learning (GABRIL), a novel method that leverages the human gaze data gathered during the data collection phase to guide the representation learning in IL. GABRIL utilizes a regularization loss which encourages the model to focus on causally relevant features identified through expert gaze and consequently mitigates the effects of confounding variables. We validate our approach in Atari environments and the Bench2Drive benchmark in CARLA by collecting human gaze datasets and applying our method in both domains. Experimental results show that the improvement of GABRIL over behavior cloning is around 179% more than the same number for other baselines in the Atari and 76% in the CARLA setup. Finally, we show that our method provides extra explainability when compared to regular IL agents. 

**Abstract (ZH)**: 基于注视的强化学习正则化在模仿学习中的应用（GAze-Based Regularization in Imitation Learning, GABRIL） 

---
# Reward-Augmented Reinforcement Learning for Continuous Control in Precision Autonomous Parking via Policy Optimization Methods 

**Title (ZH)**: 基于策略优化方法的精确自主停车连续控制的奖励增强强化学习 

**Authors**: Ahmad Suleman, Misha Urooj Khan, Zeeshan Kaleem, Ali H. Alenezi, Iqra Shabbir Sinem Coleri, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19642)  

**Abstract**: Autonomous parking (AP) represents a critical yet complex subset of intelligent vehicle automation, characterized by tight spatial constraints, frequent close-range obstacle interactions, and stringent safety margins. However, conventional rule-based and model-predictive methods often lack the adaptability and generalization needed to handle the nonlinear and environment-dependent complexities of AP. To address these limitations, we propose a reward-augmented learning framework for AP (RARLAP), that mitigates the inherent complexities of continuous-domain control by leveraging structured reward design to induce smooth and adaptable policy behavior, trained entirely within a high-fidelity Unity-based custom 3D simulation environment. We systematically design and assess three structured reward strategies: goal-only reward (GOR), dense proximity reward (DPR), and milestone-augmented reward (MAR), each integrated with both on-policy and off-policy optimization paradigms. Empirical evaluations demonstrate that the on-policy MAR achieves a 91\% success rate, yielding smoother trajectories and more robust behavior, while GOR and DPR fail to guide effective learning. Convergence and trajectory analyses demonstrate that the proposed framework enhances policy adaptability, accelerates training, and improves safety in continuous control. Overall, RARLAP establishes that reward augmentation effectively addresses complex autonomous parking challenges, enabling scalable and efficient policy optimization with both on- and off-policy methods. To support reproducibility, the code accompanying this paper is publicly available. 

**Abstract (ZH)**: 自主泊车(Autonomous Parking, AP)代表了一种关键但复杂的智能车辆自动化子集，其特征包括严格的 spatial 约束、频繁的近距离障碍交互以及严格的安全裕度。然而，传统的基于规则的方法和模型预测方法往往缺乏处理 AP 非线性和环境依赖性复杂性的适应性和泛化能力。为了解决这些限制，我们提出了一种增强奖励的学习框架（Reward-Augmented Learning Framework for Autonomous Parking, RARLAP），该框架通过利用结构化奖励设计来减轻连续域控制的内在复杂性，并在高保真度的 Unity 基础自定义 3D 仿真环境中完全训练，诱导平稳且适应性强的策略行为。我们系统地设计并评估了三种结构化奖励策略：仅目标奖励（Goal-Only Reward, GOR）、密集距离奖励（Dense Proximity Reward, DPR）以及里程碑增强奖励（Milestone-Augmented Reward, MAR），每种策略都与在线策略优化和离线策略优化范式相结合。实证评估表明，仅目标增强奖励（on-policy MAR）实现了 91% 的成功率，产生更平稳的轨迹和更稳健的行为，而 GOR 和 DPR 未能引导有效的学习。收敛性和轨迹分析表明，所提出的框架增强了策略适应性，加速了训练，并提高了连续控制中的安全性。总体而言，RARLAP 证实了奖励增强有效地解决了复杂的自主泊车挑战，使得使用在线和离线方法进行可扩展且高效的策略优化成为可能。为了支持可重复性，与本文配套的代码已公开。 

---
# Extending Group Relative Policy Optimization to Continuous Control: A Theoretical Framework for Robotic Reinforcement Learning 

**Title (ZH)**: 将组相对策略优化扩展到连续控制：机器人强化学习的理论框架 

**Authors**: Rajat Khanda, Mohammad Baqar, Sambuddha Chakrabarti, Satyasaran Changdar  

**Link**: [PDF](https://arxiv.org/pdf/2507.19555)  

**Abstract**: Group Relative Policy Optimization (GRPO) has shown promise in discrete action spaces by eliminating value function dependencies through group-based advantage estimation. However, its application to continuous control remains unexplored, limiting its utility in robotics where continuous actions are essential. This paper presents a theoretical framework extending GRPO to continuous control environments, addressing challenges in high-dimensional action spaces, sparse rewards, and temporal dynamics. Our approach introduces trajectory-based policy clustering, state-aware advantage estimation, and regularized policy updates designed for robotic applications. We provide theoretical analysis of convergence properties and computational complexity, establishing a foundation for future empirical validation in robotic systems including locomotion and manipulation tasks. 

**Abstract (ZH)**: 基于群体的优势估计连续控制的组相对策略优化（GRPO）在离散动作空间中显示出潜力，并通过群体优势估计消除了价值函数的依赖性。然而，其在连续控制中的应用尚未探索，限制了其在需要连续动作的机器人领域的应用。本文提出了一种扩展GRPO到连续控制环境的理论框架，解决高维度动作空间、稀疏奖励以及时间动态带来的挑战。我们的方法引入了基于轨迹的策略聚类、状态感知优势估计以及针对机器人应用的正则化策略更新。我们提供了收敛性质和计算复杂性的理论分析，为未来的机器人系统中的运动和操作任务的实证验证奠定了基础。 

---
# Flow Matching Policy Gradients 

**Title (ZH)**: 流匹配策略梯度 

**Authors**: David McAllister, Songwei Ge, Brent Yi, Chung Min Kim, Ethan Weber, Hongsuk Choi, Haiwen Feng, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.21053)  

**Abstract**: Flow-based generative models, including diffusion models, excel at modeling continuous distributions in high-dimensional spaces. In this work, we introduce Flow Policy Optimization (FPO), a simple on-policy reinforcement learning algorithm that brings flow matching into the policy gradient framework. FPO casts policy optimization as maximizing an advantage-weighted ratio computed from the conditional flow matching loss, in a manner compatible with the popular PPO-clip framework. It sidesteps the need for exact likelihood computation while preserving the generative capabilities of flow-based models. Unlike prior approaches for diffusion-based reinforcement learning that bind training to a specific sampling method, FPO is agnostic to the choice of diffusion or flow integration at both training and inference time. We show that FPO can train diffusion-style policies from scratch in a variety of continuous control tasks. We find that flow-based models can capture multimodal action distributions and achieve higher performance than Gaussian policies, particularly in under-conditioned settings. 

**Abstract (ZH)**: 基于流的生成模型，包括扩散模型，在高维空间中 excel 于建模连续分布。在本文中，我们引入了流策略优化（FPO），这是一种将流匹配引入策略梯度框架的简单在线策略强化学习算法。FPO 将策略优化视为最大化由条件流匹配损失计算的优势加权比，以与流行的 PPO-clip 框架兼容。它避免了精确似然计算的需要，同时保留了基于流的模型的生成能力。与基于扩散的强化学习的先前方法不同，FPO 在训练和推理时对扩散或流的集成选择不敏感。我们展示了 FPO 可以从各种连续控制任务中训练出扩散风格的策略。我们发现，基于流的模型可以捕捉多模态动作分布，并且在欠约束设置中，其性能优于高斯策略。 

---
# Partially Observable Monte-Carlo Graph Search 

**Title (ZH)**: 部分可观测蒙特卡洛图搜索 

**Authors**: Yang You, Vincent Thomas, Alex Schutz, Robert Skilton, Nick Hawes, Olivier Buffet  

**Link**: [PDF](https://arxiv.org/pdf/2507.20951)  

**Abstract**: Currently, large partially observable Markov decision processes (POMDPs) are often solved by sampling-based online methods which interleave planning and execution phases. However, a pre-computed offline policy is more desirable in POMDP applications with time or energy constraints. But previous offline algorithms are not able to scale up to large POMDPs. In this article, we propose a new sampling-based algorithm, the partially observable Monte-Carlo graph search (POMCGS) to solve large POMDPs offline. Different from many online POMDP methods, which progressively develop a tree while performing (Monte-Carlo) simulations, POMCGS folds this search tree on the fly to construct a policy graph, so that computations can be drastically reduced, and users can analyze and validate the policy prior to embedding and executing it. Moreover, POMCGS, together with action progressive widening and observation clustering methods provided in this article, is able to address certain continuous POMDPs. Through experiments, we demonstrate that POMCGS can generate policies on the most challenging POMDPs, which cannot be computed by previous offline algorithms, and these policies' values are competitive compared with the state-of-the-art online POMDP algorithms. 

**Abstract (ZH)**: 当前，大型部分可观察马尔可夫决策过程（POMDP）通常通过混合规划和执行阶段的采样基于在线方法来求解。然而，在时间或能量受限的POMDP应用中，一个预先计算的离线策略更为 desirable。但是，先前的离线算法无法扩展到大型POMDP。在本文中，我们提出了一种新的采样基于算法——部分可观察蒙特卡洛图搜索（POMCGS），以解决大型POMDP的离线问题。与许多在执行（蒙特卡洛）模拟过程中逐步构建树的在线POMDP方法不同，POMCGS 在运行时动态折叠搜索树以构建策略图，从而大幅减少计算量，并允许用户在嵌入和执行策略之前对其进行分析和验证。此外，POMCGS 结合本文提供的动作逐步扩展和观测聚类方法，能够解决某些连续的POMDP。通过实验，我们证明POMCGS 可以生成在最具挑战性的POMDP上计算策略，这些策略的价值与最先进的在线POMDP算法相当。 

---
# Beyond Line-of-Sight: Cooperative Localization Using Vision and V2X Communication 

**Title (ZH)**: 超越视线：基于视觉和V2X通信的协同定位 

**Authors**: Annika Wong, Zhiqi Tang, Frank J. Jiang, Karl H. Johansson, Jonas Mårtensson  

**Link**: [PDF](https://arxiv.org/pdf/2507.20772)  

**Abstract**: Accurate and robust localization is critical for the safe operation of Connected and Automated Vehicles (CAVs), especially in complex urban environments where Global Navigation Satellite System (GNSS) signals are unreliable. This paper presents a novel vision-based cooperative localization algorithm that leverages onboard cameras and Vehicle-to-Everything (V2X) communication to enable CAVs to estimate their poses, even in occlusion-heavy scenarios such as busy intersections. In particular, we propose a novel decentralized observer for a group of connected agents that includes landmark agents (static or moving) in the environment with known positions and vehicle agents that need to estimate their poses (both positions and orientations). Assuming that (i) there are at least three landmark agents in the environment, (ii) each vehicle agent can measure its own angular and translational velocities as well as relative bearings to at least three neighboring landmarks or vehicles, and (iii) neighboring vehicles can communicate their pose estimates, each vehicle can estimate its own pose using the proposed decentralized observer. We prove that the origin of the estimation error is locally exponentially stable under the proposed observer, provided that the minimal observability conditions are satisfied. Moreover, we evaluate the proposed approach through experiments with real 1/10th-scale connected vehicles and large-scale simulations, demonstrating its scalability and validating the theoretical guarantees in practical scenarios. 

**Abstract (ZH)**: 基于视觉的合作定位算法：面向复杂城市环境中的连接和自动驾驶车辆的鲁棒定位 

---
# AQUA: A Large Language Model for Aquaculture & Fisheries 

**Title (ZH)**: AQUA：用于水产养殖与渔业的大型语言模型 

**Authors**: Praneeth Narisetty, Uday Kumar Reddy Kattamanchi, Lohit Akshant Nimma, Sri Ram Kaushik Karnati, Shiva Nagendra Babu Kore, Mounika Golamari, Tejashree Nageshreddy  

**Link**: [PDF](https://arxiv.org/pdf/2507.20520)  

**Abstract**: Aquaculture plays a vital role in global food security and coastal economies by providing sustainable protein sources. As the industry expands to meet rising demand, it faces growing challenges such as disease outbreaks, inefficient feeding practices, rising labor costs, logistical inefficiencies, and critical hatchery issues, including high mortality rates and poor water quality control. Although artificial intelligence has made significant progress, existing machine learning methods fall short of addressing the domain-specific complexities of aquaculture. To bridge this gap, we introduce AQUA, the first large language model (LLM) tailored for aquaculture, designed to support farmers, researchers, and industry practitioners. Central to this effort is AQUADAPT (Data Acquisition, Processing and Tuning), an Agentic Framework for generating and refining high-quality synthetic data using a combination of expert knowledge, largescale language models, and automated evaluation techniques. Our work lays the foundation for LLM-driven innovations in aquaculture research, advisory systems, and decision-making tools. 

**Abstract (ZH)**: 水产养殖在通过提供可持续蛋白质来源保障全球食品 security 和沿海经济体方面发挥着重要作用。随着该行业扩大以满足不断增长的需求，它面临着诸如疾病暴发、喂养效率低下、劳动成本上升、物流效率低下以及关键苗圃问题（包括高死亡率和水质控制差）等诸多挑战。尽管人工智能取得了显著进步，现有的机器学习方法仍无法解决水产养殖领域的特定复杂性。为弥补这一差距，我们提出了AQUA，这是首个针对水产养殖定制的大语言模型（LLM），旨在支持农民、研究人员和行业从业者。该努力的核心是AQUADAPT（数据获取、处理和调整）框架，该框架结合使用专家知识、大规模语言模型和自动化评估技术生成和改进高质量的合成数据。我们的工作为基础研究、咨询系统和决策工具有了基于大语言模型的创新奠定了基础。 

---
# ACCESS-AV: Adaptive Communication-Computation Codesign for Sustainable Autonomous Vehicle Localization in Smart Factories 

**Title (ZH)**: ACCESS-AV：面向智能工厂自主车辆定位的自适应通信-计算协同设计 

**Authors**: Rajat Bhattacharjya, Arnab Sarkar, Ish Kool, Sabur Baidya, Nikil Dutt  

**Link**: [PDF](https://arxiv.org/pdf/2507.20399)  

**Abstract**: Autonomous Delivery Vehicles (ADVs) are increasingly used for transporting goods in 5G network-enabled smart factories, with the compute-intensive localization module presenting a significant opportunity for optimization. We propose ACCESS-AV, an energy-efficient Vehicle-to-Infrastructure (V2I) localization framework that leverages existing 5G infrastructure in smart factory environments. By opportunistically accessing the periodically broadcast 5G Synchronization Signal Blocks (SSBs) for localization, ACCESS-AV obviates the need for dedicated Roadside Units (RSUs) or additional onboard sensors to achieve energy efficiency as well as cost reduction. We implement an Angle-of-Arrival (AoA)-based estimation method using the Multiple Signal Classification (MUSIC) algorithm, optimized for resource-constrained ADV platforms through an adaptive communication-computation strategy that dynamically balances energy consumption with localization accuracy based on environmental conditions such as Signal-to-Noise Ratio (SNR) and vehicle velocity. Experimental results demonstrate that ACCESS-AV achieves an average energy reduction of 43.09% compared to non-adaptive systems employing AoA algorithms such as vanilla MUSIC, ESPRIT, and Root-MUSIC. It maintains sub-30 cm localization accuracy while also delivering substantial reductions in infrastructure and operational costs, establishing its viability for sustainable smart factory environments. 

**Abstract (ZH)**: 基于5G基础设施的自主配送车辆（ADVs）能量高效V2I定位框架：ACCESS-AV 

---
# Hypo-paradoxical Linkages: Linkages That Should Move-But Don't 

**Title (ZH)**: 超悖论性链接：本应变动却未变动的链接 

**Authors**: Nir Shvalb, Oded Medina  

**Link**: [PDF](https://arxiv.org/pdf/2507.20371)  

**Abstract**: While paradoxical linkages famously violate the Chebyshev-Grubler-Kutzbach criterion by exhibiting unexpected mobility, we identify an opposing phenomenon: a class of linkages that appear mobile according to the same criterion, yet are in fact rigid. We refer to these as hypo-paradoxical linkages, and proceed to analyze and illustrate their behavior. We use the same tools to further explain the unexpected positive mobility of Bennet mechanism. 

**Abstract (ZH)**: 而悖论性连杆机构虽然违反了Chebyshev-Grubler-Kutzbach准则，表现出意想不到的自由度，我们发现一种 opposing 现象：一类连杆机构根据同一准则显示出 Mobility，但实际上是刚性的。我们将这些机构称为假设悖论性连杆机构，并进一步分析和说明其行为。我们使用相同的工具进一步解释Bennet机构的意想不到的正自由度。 

---
# VLMPlanner: Integrating Visual Language Models with Motion Planning 

**Title (ZH)**: VLMPlanner：将视觉语言模型与运动规划集成 

**Authors**: Zhipeng Tang, Sha Zhang, Jiajun Deng, Chenjie Wang, Guoliang You, Yuting Huang, Xinrui Lin, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20342)  

**Abstract**: Integrating large language models (LLMs) into autonomous driving motion planning has recently emerged as a promising direction, offering enhanced interpretability, better controllability, and improved generalization in rare and long-tail scenarios. However, existing methods often rely on abstracted perception or map-based inputs, missing crucial visual context, such as fine-grained road cues, accident aftermath, or unexpected obstacles, which are essential for robust decision-making in complex driving environments. To bridge this gap, we propose VLMPlanner, a hybrid framework that combines a learning-based real-time planner with a vision-language model (VLM) capable of reasoning over raw images. The VLM processes multi-view images to capture rich, detailed visual information and leverages its common-sense reasoning capabilities to guide the real-time planner in generating robust and safe trajectories. Furthermore, we develop the Context-Adaptive Inference Gate (CAI-Gate) mechanism that enables the VLM to mimic human driving behavior by dynamically adjusting its inference frequency based on scene complexity, thereby achieving an optimal balance between planning performance and computational efficiency. We evaluate our approach on the large-scale, challenging nuPlan benchmark, with comprehensive experimental results demonstrating superior planning performance in scenarios with intricate road conditions and dynamic elements. Code will be available. 

**Abstract (ZH)**: 将大规模语言模型（LLMs）集成到自主驾驶运动规划中已成为一个有前景的方向，能够提高可解释性、更好的可控性和在稀有和长尾场景中的泛化能力。然而，现有方法往往依赖于抽象的感知或基于地图的输入，缺失了重要的视觉上下文，如细粒度的道路提示、事故后的状况或意外障碍物，这些都是在复杂驾驶环境中实现稳健决策必不可少的。为弥补这一差距，我们提出了一种名为VLMPlanner的混合框架，该框架结合了基于学习的实时规划器和一个能够对原始图像进行推理的视觉语言模型（VLM）。通过处理多视角图像，VLM捕获丰富的详细视觉信息，并利用其常识推理能力引导实时规划器生成稳健且安全的轨迹。此外，我们开发了上下文自适应推断门控机制（CAI-Gate），该机制使VLM能够根据场景复杂度动态调整其推理频率，从而在规划性能和计算效率之间实现最优平衡。我们在大规模挑战性的nuPlan基准上评估了该方法，实验结果全面展示了在复杂道路条件和动态元素场景中的优越规划性能。代码将开源。 

---
# Optimizing Spreading Factor Selection for Mobile LoRa Gateways Using Single-Channel Hardware 

**Title (ZH)**: 使用单通道硬件优化移动LoRa网关的扩频因子选择 

**Authors**: W. A. Sasindu Wijesuriya  

**Link**: [PDF](https://arxiv.org/pdf/2507.19938)  

**Abstract**: The deployment of mobile LoRa gateways using low-cost single-channel hardware presents a significant challenge in maintaining reliable communication due to the lack of dynamic configuration support. In traditional LoRaWAN networks, Adaptive Data Rate (ADR) mechanisms optimize communication parameters in real time. However, such features are typically supported only by expensive multi-channel gateways. This study proposes a cost-effective and energy-efficient solution by statically selecting the optimal Spreading Factor (SF) using a two-phase algorithm. The method first applies rule-based exclusion to eliminate SFs that violate constraints related to distance, data rate, link margin, and regulatory limits. Remaining candidates are then evaluated using a weighted scoring model incorporating Time-on-Air, energy consumption, data rate, and link robustness. The proposed algorithm was validated through extensive field tests and NS-3 simulations under line-of-sight conditions. Results demonstrate that the selected SF matched the optimal SF in over 92% of cases across 672 simulated scenarios, confirming the algorithm's effectiveness. This approach offers a scalable alternative to dynamic protocols, enabling reliable mobile LoRa deployments in cost-sensitive environments such as agriculture and rural sensing applications. 

**Abstract (ZH)**: 使用低成本单通道硬件部署移动LoRa网关因其缺乏动态配置支持而在保持可靠通信方面面临重大挑战。通过一种两阶段算法静态选择最优扩频因子(SF)，本研究提出了一种低成本和节能的解决方案。该方法首先采用基于规则的排除法来消除违反距离、数据率、链路余量和监管限制的SF。剩余候选SF再通过综合考虑空中时间、能耗、数据率和链路稳健性的加权评分模型进行评估。所提算法通过视线条件下的广泛现场测试和NS-3仿真进行了验证。结果表明，在672个模拟场景中，所选SF与最优SF匹配的比例超过92%，证明了该算法的有效性。该方法为动态协议提供了一种可扩展的替代方案，在农业和农村传感等成本敏感环境中实现可靠的移动LoRa部署。 

---
# Efficient Self-Supervised Neuro-Analytic Visual Servoing for Real-time Quadrotor Control 

**Title (ZH)**: 高效自监督神经分析视觉伺服控制实时四旋翼飞行器控制 

**Authors**: Sebastian Mocanu, Sebastian-Ion Nae, Mihai-Eugen Barbu, Marius Leordeanu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19878)  

**Abstract**: This work introduces a self-supervised neuro-analytical, cost efficient, model for visual-based quadrotor control in which a small 1.7M parameters student ConvNet learns automatically from an analytical teacher, an improved image-based visual servoing (IBVS) controller. Our IBVS system solves numerical instabilities by reducing the classical visual servoing equations and enabling efficient stable image feature detection. Through knowledge distillation, the student model achieves 11x faster inference compared to the teacher IBVS pipeline, while demonstrating similar control accuracy at a significantly lower computational and memory cost. Our vision-only self-supervised neuro-analytic control, enables quadrotor orientation and movement without requiring explicit geometric models or fiducial markers. The proposed methodology leverages simulation-to-reality transfer learning and is validated on a small drone platform in GPS-denied indoor environments. Our key contributions include: (1) an analytical IBVS teacher that solves numerical instabilities inherent in classical approaches, (2) a two-stage segmentation pipeline combining YOLOv11 with a U-Net-based mask splitter for robust anterior-posterior vehicle segmentation to correctly estimate the orientation of the target, and (3) an efficient knowledge distillation dual-path system, which transfers geometric visual servoing capabilities from the analytical IBVS teacher to a compact and small student neural network that outperforms the teacher, while being suitable for real-time onboard deployment. 

**Abstract (ZH)**: 一种自监督的神经分析控制方法：基于视觉的四旋翼飞行器低成本控制模型 

---
# Co-Win: Joint Object Detection and Instance Segmentation in LiDAR Point Clouds via Collaborative Window Processing 

**Title (ZH)**: Co-Win: 联合点云目标检测与实例分割的协作窗口处理方法 

**Authors**: Haichuan Li, Tomi Westerlund  

**Link**: [PDF](https://arxiv.org/pdf/2507.19691)  

**Abstract**: Accurate perception and scene understanding in complex urban environments is a critical challenge for ensuring safe and efficient autonomous navigation. In this paper, we present Co-Win, a novel bird's eye view (BEV) perception framework that integrates point cloud encoding with efficient parallel window-based feature extraction to address the multi-modality inherent in environmental understanding. Our method employs a hierarchical architecture comprising a specialized encoder, a window-based backbone, and a query-based decoder head to effectively capture diverse spatial features and object relationships. Unlike prior approaches that treat perception as a simple regression task, our framework incorporates a variational approach with mask-based instance segmentation, enabling fine-grained scene decomposition and understanding. The Co-Win architecture processes point cloud data through progressive feature extraction stages, ensuring that predicted masks are both data-consistent and contextually relevant. Furthermore, our method produces interpretable and diverse instance predictions, enabling enhanced downstream decision-making and planning in autonomous driving systems. 

**Abstract (ZH)**: 准确感知和理解复杂城市环境是确保自主导航安全和高效的关键挑战。本文提出了一种新颖的鸟瞰视图（BEV）感知框架Co-Win，该框架结合点云编码与高效的并行窗口特征提取，以应对环境理解中的多模态性。本文方法采用包含专用编码器、基于窗口的主干和查询式解码头的分层架构，以有效地捕捉多样的空间特征和对象关系。与以往将感知简单视为回归任务的方法不同，我们的框架结合了变分方法和基于掩码的实例分割，实现精细粒度的场景分解和理解。Co-Win架构通过逐步特征提取阶段处理点云数据，确保预测掩码既与数据一致又具有上下文相关性。此外，我们的方法能够生成可解释且多样的实例预测，从而增强自主驾驶系统中的下游决策和规划能力。 

---
