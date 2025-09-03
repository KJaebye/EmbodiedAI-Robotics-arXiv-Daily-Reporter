# Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots 

**Title (ZH)**: 模拟中的操控： enabling robots to achieve准确的几何感知 

**Authors**: Minghuan Liu, Zhengbang Zhu, Xiaoshen Han, Peng Hu, Haotong Lin, Xinyao Li, Jingxiao Chen, Jiafeng Xu, Yichu Yang, Yunfeng Lin, Xinghang Li, Yong Yu, Weinan Zhang, Tao Kong, Bingyi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02530)  

**Abstract**: Modern robotic manipulation primarily relies on visual observations in a 2D color space for skill learning but suffers from poor generalization. In contrast, humans, living in a 3D world, depend more on physical properties-such as distance, size, and shape-than on texture when interacting with objects. Since such 3D geometric information can be acquired from widely available depth cameras, it appears feasible to endow robots with similar perceptual capabilities. Our pilot study found that using depth cameras for manipulation is challenging, primarily due to their limited accuracy and susceptibility to various types of noise. In this work, we propose Camera Depth Models (CDMs) as a simple plugin on daily-use depth cameras, which take RGB images and raw depth signals as input and output denoised, accurate metric depth. To achieve this, we develop a neural data engine that generates high-quality paired data from simulation by modeling a depth camera's noise pattern. Our results show that CDMs achieve nearly simulation-level accuracy in depth prediction, effectively bridging the sim-to-real gap for manipulation tasks. Notably, our experiments demonstrate, for the first time, that a policy trained on raw simulated depth, without the need for adding noise or real-world fine-tuning, generalizes seamlessly to real-world robots on two challenging long-horizon tasks involving articulated, reflective, and slender objects, with little to no performance degradation. We hope our findings will inspire future research in utilizing simulation data and 3D information in general robot policies. 

**Abstract (ZH)**: 基于深度相机的Camera Depth Models (CDMs)提升机器人 manipulation 精度：从模拟到现实的桥梁 

---
# Fault-tolerant Model Predictive Control for Spacecraft 

**Title (ZH)**: 空间飞行器容错模型预测控制 

**Authors**: Raphael Stöckner, Pedro Roque, Maria Charitidou, Dimos V. Dimarogonas  

**Link**: [PDF](https://arxiv.org/pdf/2509.02527)  

**Abstract**: Given the cost and critical functions of satellite constellations, ensuring mission longevity and safe decommissioning is essential for space sustainability. This article presents a Model Predictive Control for spacecraft trajectory and setpoint stabilization under multiple actuation failures. The proposed solution allows us to efficiently control the faulty spacecraft enabling safe navigation towards servicing or collision-free trajectories. The proposed scheme ensures closed-loop asymptotic stability and is shown to be recursively feasible. We demonstrate its efficacy through open-source numerical results and realistic experiments using the ATMOS platform. 

**Abstract (ZH)**: 基于多执行机构故障的卫星轨迹和设定点稳定性的模型预测控制方法：确保空间可持续性的故障应付与安全拆解 

---
# Classification of Vision-Based Tactile Sensors: A Review 

**Title (ZH)**: 基于视觉的触觉传感器分类：一个综述 

**Authors**: Haoran Li, Yijiong Lin, Chenghua Lu, Max Yang, Efi Psomopoulou, Nathan F Lepora  

**Link**: [PDF](https://arxiv.org/pdf/2509.02478)  

**Abstract**: Vision-based tactile sensors (VBTS) have gained widespread application in robotic hands, grippers and prosthetics due to their high spatial resolution, low manufacturing costs, and ease of customization. While VBTSs have common design features, such as a camera module, they can differ in a rich diversity of sensing principles, material compositions, multimodal approaches, and data interpretation methods. Here, we propose a novel classification of VBTS that categorizes the technology into two primary sensing principles based on the underlying transduction of contact into a tactile image: the Marker-Based Transduction Principle and the Intensity-Based Transduction Principle. Marker-Based Transduction interprets tactile information by detecting marker displacement and changes in marker density. In contrast, Intensity-Based Transduction maps external disturbances with variations in pixel values. Depending on the design of the contact module, Marker-Based Transduction can be further divided into two subtypes: Simple Marker-Based (SMB) and Morphological Marker-Based (MMB) mechanisms. Similarly, the Intensity-Based Transduction Principle encompasses the Reflective Layer-based (RLB) and Transparent Layer-Based (TLB) mechanisms. This paper provides a comparative study of the hardware characteristics of these four types of sensors including various combination types, and discusses the commonly used methods for interpreting tactile information. This~comparison reveals some current challenges faced by VBTS technology and directions for future research. 

**Abstract (ZH)**: 基于视觉的触觉传感器（Vision-based Tactile Sensors, VBTS）在机器人手、夹持器和假肢中的广泛应用得益于其高空间分辨率、低制造成本和易定制性。尽管VBTSs具有共同的设计特征，如相机模块，但它们在传感原理、材料组成、多模态方法和数据解释方法上呈现丰富的多样性。在这里，我们提出了一种新颖的VBTS分类方法，根据接触转换为触觉图像的基本原理，将技术分为两大类：标志基转换原理和强度基转换原理。标志基转换通过检测标志物的位移和标志密度的变化来解释触觉信息。相比之下，强度基转换通过像素值的变化映射外部扰动。根据接触模块的设计，标志基转换可以进一步分为简单标志基（SMB）机制和形态学标志基（MMB）机制。同样，强度基转换原理包括反射层基（RLB）机制和透明层基（TLB）机制。本文对这四种类型传感器的硬件特性及其各种组合类型进行了比较研究，并讨论了常用的数据解释方法。比较揭示了VBTS技术目前面临的某些挑战及未来研究的方向。 

---
# Coral: A Unifying Abstraction Layer for Composable Robotics Software 

**Title (ZH)**: Coral: 一种统合的机器人软件模块化抽象层 

**Authors**: Steven Swanbeck, Mitch Pryor  

**Link**: [PDF](https://arxiv.org/pdf/2509.02453)  

**Abstract**: Despite the multitude of excellent software components and tools available in the robotics and broader software engineering communities, successful integration of software for robotic systems remains a time-consuming and challenging task for users of all knowledge and skill levels. And with robotics software often being built into tightly coupled, monolithic systems, even minor alterations to improve performance, adjust to changing task requirements, or deploy to new hardware can require significant engineering investment. To help solve this problem, this paper presents Coral, an abstraction layer for building, deploying, and coordinating independent software components that maximizes composability to allow for rapid system integration without modifying low-level code. Rather than replacing existing tools, Coral complements them by introducing a higher-level abstraction that constrains the integration process to semantically meaningful choices, reducing the configuration burden without limiting adaptability to diverse domains, systems, and tasks. We describe Coral in detail and demonstrate its utility in integrating software for scenarios of increasing complexity, including LiDAR-based SLAM and multi-robot corrosion mitigation tasks. By enabling practical composability in robotics software, Coral offers a scalable solution to a broad range of robotics system integration challenges, improving component reusability, system reconfigurability, and accessibility to both expert and non-expert users. We release Coral open source. 

**Abstract (ZH)**: 尽管机器人学和更广泛的软件工程社区提供了众多优秀的软件组件和工具，但将软件成功集成到机器人系统中仍然是各个知识和技能水平用户都面临的一项耗时且具有挑战性的任务。由于机器人软件通常构建为紧密耦合的大型系统，即使是对性能进行小幅改进、适应变化的任务要求或部署到新硬件所需的小规模修改，也可能需要大量的工程投入。为了解决这一问题，本文介绍了Coral，一个用于构建、部署和协调独立软件组件的抽象层，最大限度地提高可组合性，以便在不修改底层代码的情况下快速实现系统集成。Coral 不会取代现有的工具，而是通过引入更高层次的抽象来补充它们，将集成过程限制在语义上具有意义的选择，从而减少配置负担，同时不限制对不同领域、系统和任务的适应性。我们详细描述了Coral，并展示了其在集成从简单到复杂场景的软件方面的实用性，包括基于LiDAR的SLAM和多机器人腐蚀防护任务。通过在机器人软件中实现实用的可组合性，Coral 提供了一个可扩展的解决方案，以应对广泛的机器人系统集成挑战，提高了组件的重用性、系统的重新配置能力和对专家及非专家用户的可访问性。我们已开源Coral。 

---
# U-ARM : Ultra low-cost general teleoperation interface for robot manipulation 

**Title (ZH)**: U-ARM : 超低成本通用机器人操作接口 

**Authors**: Yanwen Zou, Zhaoye Zhou, Chenyang Shi, Zewei Ye, Junda Huang, Yan Ding, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.02437)  

**Abstract**: We propose U-Arm, a low-cost and rapidly adaptable leader-follower teleoperation framework designed to interface with most of commercially available robotic arms. Our system supports teleoperation through three structurally distinct 3D-printed leader arms that share consistent control logic, enabling seamless compatibility with diverse commercial robot configurations. Compared with previous open-source leader-follower interfaces, we further optimized both the mechanical design and servo selection, achieving a bill of materials (BOM) cost of only \$50.5 for the 6-DoF leader arm and \$56.8 for the 7-DoF version. To enhance usability, we mitigate the common challenge in controlling redundant degrees of freedom by %engineering methods mechanical and control optimizations. Experimental results demonstrate that U-Arm achieves 39\% higher data collection efficiency and comparable task success rates across multiple manipulation scenarios compared with Joycon, another low-cost teleoperation interface. We have open-sourced all CAD models of three configs and also provided simulation support for validating teleoperation workflows. We also open-sourced real-world manipulation data collected with U-Arm. The project website is this https URL. 

**Abstract (ZH)**: 我们提出了一种低成本且易于快速适应的Leader-Follower远程操作框架U-Arm，设计用于与大多数商用机器人臂兼容。该系统通过三个结构不同的3D打印Leader臂支持远程操作，这些Leader臂共享一致的控制逻辑，使其能够与各种商业机器人配置无缝兼容。与之前的开源Leader-Follower接口相比，我们进一步优化了机械设计和伺服选择，6-DoF Leader臂的成本仅为50.5美元，7-DoF版本的成本为56.8美元。为了提高易用性，我们通过工程方法和优化解耦冗余自由度的控制问题。实验结果表明，与另一种低成本远程操作接口Joycon相比，U-Arm在多个操作场景中的数据收集效率提高39%，任务成功率相当。我们已开源三个配置的全部CAD模型，并提供了验证远程操作流程的仿真支持。此外，我们还开源了使用U-Arm收集的实地操作数据。该项目网站为：this https URL。 

---
# OpenGuide: Assistive Object Retrieval in Indoor Spaces for Individuals with Visual Impairments 

**Title (ZH)**: OpenGuide: 为视障个体在室内空间中提供辅助对象检索 

**Authors**: Yifan Xu, Qianwei Wang, Vineet Kamat, Carol Menassa  

**Link**: [PDF](https://arxiv.org/pdf/2509.02425)  

**Abstract**: Indoor built environments like homes and offices often present complex and cluttered layouts that pose significant challenges for individuals who are blind or visually impaired, especially when performing tasks that involve locating and gathering multiple objects. While many existing assistive technologies focus on basic navigation or obstacle avoidance, few systems provide scalable and efficient multi-object search capabilities in real-world, partially observable settings. To address this gap, we introduce OpenGuide, an assistive mobile robot system that combines natural language understanding with vision-language foundation models (VLM), frontier-based exploration, and a Partially Observable Markov Decision Process (POMDP) planner. OpenGuide interprets open-vocabulary requests, reasons about object-scene relationships, and adaptively navigates and localizes multiple target items in novel environments. Our approach enables robust recovery from missed detections through value decay and belief-space reasoning, resulting in more effective exploration and object localization. We validate OpenGuide in simulated and real-world experiments, demonstrating substantial improvements in task success rate and search efficiency over prior methods. This work establishes a foundation for scalable, human-centered robotic assistance in assisted living environments. 

**Abstract (ZH)**: 室内建筑设计，如住宅和办公室，常常具有复杂且杂乱的布局，这对盲人或视觉受损者在执行查找和收集多个物体的任务时构成了重大挑战。虽然许多现有的辅助技术侧重于基本导航或避障，但在部分可观测的真实世界环境中，很少有系统能够提供可扩展且高效的多目标搜索能力。为解决这一问题，我们介绍了一种名为OpenGuide的辅助移动机器人系统，该系统结合了自然语言理解、视觉-语言基础模型（VLM）、边疆导向探索以及部分可观测马尔可夫决策过程（POMDP）规划。OpenGuide能够解析开放词汇请求，推断物体-场景关系，并在新型环境中自适应导航和定位多个目标物品。我们的方法通过价值衰减和信念空间推理实现出色的错误恢复，从而实现更有效的探索和物体定位。我们通过仿真和现实世界实验验证了OpenGuide，证明其在任务成功率和搜索效率方面较以前的方法有显著提升。这项工作为在辅助生活环境中提供可扩展且以人为中心的机器人辅助奠定了基础。 

---
# Physics-Informed Machine Learning with Adaptive Grids for Optical Microrobot Depth Estimation 

**Title (ZH)**: 基于物理 informant 的自适应网格机器学习方法用于光学微机器人深度估计 

**Authors**: Lan Wei, Lou Genoud, Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02343)  

**Abstract**: Optical microrobots actuated by optical tweezers (OT) offer great potential for biomedical applications such as cell manipulation and microscale assembly. These tasks demand accurate three-dimensional perception to ensure precise control in complex and dynamic biological environments. However, the transparent nature of microrobots and low-contrast microscopic imaging challenge conventional deep learning methods, which also require large annotated datasets that are costly to obtain. To address these challenges, we propose a physics-informed, data-efficient framework for depth estimation of optical microrobots. Our method augments convolutional feature extraction with physics-based focus metrics, such as entropy, Laplacian of Gaussian, and gradient sharpness, calculated using an adaptive grid strategy. This approach allocates finer grids over microrobot regions and coarser grids over background areas, enhancing depth sensitivity while reducing computational complexity. We evaluate our framework on multiple microrobot types and demonstrate significant improvements over baseline models. Specifically, our approach reduces mean squared error (MSE) by over 60% and improves the coefficient of determination (R^2) across all test cases. Notably, even when trained on only 20% of the available data, our model outperforms ResNet50 trained on the full dataset, highlighting its robustness under limited data conditions. Our code is available at: this https URL. 

**Abstract (ZH)**: 光学镊子驱动的光学微机器人三维深度估计的物理导向高效框架 

---
# Language-Guided Long Horizon Manipulation with LLM-based Planning and Visual Perception 

**Title (ZH)**: 基于LLM的规划与视觉感知的语言指导长时 horizons 操作 

**Authors**: Changshi Zhou, Haichuan Xu, Ningquan Gu, Zhipeng Wang, Bin Cheng, Pengpeng Zhang, Yanchao Dong, Mitsuhiro Hayashibe, Yanmin Zhou, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2509.02324)  

**Abstract**: Language-guided long-horizon manipulation of deformable objects presents significant challenges due to high degrees of freedom, complex dynamics, and the need for accurate vision-language grounding. In this work, we focus on multi-step cloth folding, a representative deformable-object manipulation task that requires both structured long-horizon planning and fine-grained visual perception. To this end, we propose a unified framework that integrates a Large Language Model (LLM)-based planner, a Vision-Language Model (VLM)-based perception system, and a task execution module. Specifically, the LLM-based planner decomposes high-level language instructions into low-level action primitives, bridging the semantic-execution gap, aligning perception with action, and enhancing generalization. The VLM-based perception module employs a SigLIP2-driven architecture with a bidirectional cross-attention fusion mechanism and weight-decomposed low-rank adaptation (DoRA) fine-tuning to achieve language-conditioned fine-grained visual grounding. Experiments in both simulation and real-world settings demonstrate the method's effectiveness. In simulation, it outperforms state-of-the-art baselines by 2.23, 1.87, and 33.3 on seen instructions, unseen instructions, and unseen tasks, respectively. On a real robot, it robustly executes multi-step folding sequences from language instructions across diverse cloth materials and configurations, demonstrating strong generalization in practical scenarios. Project page: this https URL 

**Abstract (ZH)**: 基于语言指导的长时序变形物体 manipulation 面临极大挑战，主要原因包括高自由度、复杂动力学以及精确的视觉-语言匹配需求。本文聚焦于多步布料折叠任务，这是一种既需要结构化长时序规划又需要精细视觉感知的典型变形物体 manipulation 任务。为此，我们提出了一种统一框架，该框架整合了基于大型语言模型 (LLM) 的规划器、基于视觉-语言模型 (VLM) 的感知系统以及任务执行模块。具体来说，基于 LLM 的规划器将高层语言指令分解为底层动作 primitive，从而弥合语义执行差距、使感知与动作相协调，并增强泛化能力。基于 VLM 的感知模块采用由 SigLIP2 驱动的架构，并结合双向跨注意力融合机制和基于权重分解的低秩适应（DoRA）微调，以实现语言条件下的精细视觉匹配。在仿真和现实环境中的实验均证明了该方法的有效性。在仿真环境中，该方法分别在已见过的指令、未见过的指令和未见过的任务上，比最先进的基线方法高出 2.23、1.87 和 33.3。在现实机器人上，该方法能够从语言指令中稳健地执行跨不同布料材料和配置的多步折叠序列，展示了在实际场景中的强大泛化能力。 

---
# Sem-RaDiff: Diffusion-Based 3D Radar Semantic Perception in Cluttered Agricultural Environments 

**Title (ZH)**: Sem-RaDiff: 基于扩散的复杂农业环境三维雷达语义感知 

**Authors**: Ruibin Zhang, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.02283)  

**Abstract**: Accurate and robust environmental perception is crucial for robot autonomous navigation. While current methods typically adopt optical sensors (e.g., camera, LiDAR) as primary sensing modalities, their susceptibility to visual occlusion often leads to degraded performance or complete system failure. In this paper, we focus on agricultural scenarios where robots are exposed to the risk of onboard sensor contamination. Leveraging radar's strong penetration capability, we introduce a radar-based 3D environmental perception framework as a viable alternative. It comprises three core modules designed for dense and accurate semantic perception: 1) Parallel frame accumulation to enhance signal-to-noise ratio of radar raw data. 2) A diffusion model-based hierarchical learning framework that first filters radar sidelobe artifacts then generates fine-grained 3D semantic point clouds. 3) A specifically designed sparse 3D network optimized for processing large-scale radar raw data. We conducted extensive benchmark comparisons and experimental evaluations on a self-built dataset collected in real-world agricultural field scenes. Results demonstrate that our method achieves superior structural and semantic prediction performance compared to existing methods, while simultaneously reducing computational and memory costs by 51.3% and 27.5%, respectively. Furthermore, our approach achieves complete reconstruction and accurate classification of thin structures such as poles and wires-which existing methods struggle to perceive-highlighting its potential for dense and accurate 3D radar perception. 

**Abstract (ZH)**: 精准且稳健的环境感知对于机器人自主导航至关重要。现有方法通常采用光学传感器（如摄像头、激光雷达）作为主要传感模态，但其对视觉遮挡的敏感性往往会降低性能或导致系统完全失效。本文着重于农业场景，其中机器人面临车载传感器污染的风险。借助雷达强大的穿透能力，本文介绍了一种基于雷达的3D环境感知框架作为可行替代方案。该框架包含三个核心模块，用于进行密集且精确的语义感知：1）并行帧累积以增强雷达原始数据的信噪比。2）基于扩散模型的分层学习框架，首先过滤雷达旁瓣伪影，然后生成精细粒度的3D语义点云。3）专门设计的稀疏3D网络，优化大型雷达原始数据处理。我们在实际农业场景自建数据集上进行了广泛的基准比较和实验评估。结果表明，我们的方法在结构和语义预测性能上优于现有方法，同时分别减少了51.3%和27.5%的计算和内存成本。此外，我们的方法能够实现对诸如杆和电线等薄结构的完整重建和准确分类，而现有方法难以感知，突显了其在密集且精确3D雷达感知方面的潜力。 

---
# Human-Inspired Soft Anthropomorphic Hand System for Neuromorphic Object and Pose Recognition Using Multimodal Signals 

**Title (ZH)**: 基于多模态信号的人类启发式软类人手系统及其在神经形态物体与姿态识别中的应用 

**Authors**: Fengyi Wang, Xiangyu Fu, Nitish Thakor, Gordon Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.02275)  

**Abstract**: The human somatosensory system integrates multimodal sensory feedback, including tactile, proprioceptive, and thermal signals, to enable comprehensive perception and effective interaction with the environment. Inspired by the biological mechanism, we present a sensorized soft anthropomorphic hand equipped with diverse sensors designed to emulate the sensory modalities of the human hand. This system incorporates biologically inspired encoding schemes that convert multimodal sensory data into spike trains, enabling highly-efficient processing through Spiking Neural Networks (SNNs). By utilizing these neuromorphic signals, the proposed framework achieves 97.14% accuracy in object recognition across varying poses, significantly outperforming previous studies on soft hands. Additionally, we introduce a novel differentiator neuron model to enhance material classification by capturing dynamic thermal responses. Our results demonstrate the benefits of multimodal sensory fusion and highlight the potential of neuromorphic approaches for achieving efficient, robust, and human-like perception in robotic systems. 

**Abstract (ZH)**: 人类本体感觉系统整合多模态感觉反馈，包括触觉、本体感觉和温度信号，以实现对环境的全面感知和有效交互。受生物机制启发，我们提出了一种装备多种传感器的人工仿生软手，旨在模拟人手的感觉模态。该系统结合了生物启发的编码方案，将多模态感觉数据转换为尖锐脉冲序列，通过脉冲神经网络（SNNs）实现高效处理。利用这些类神经形态信号，所提出的框架在不同姿态下实现了97.14%的物体识别准确率，显著优于先前关于软手的研究。此外，我们引入了一种新的微分神经元模型，通过捕捉动态的温度响应来增强材料分类能力。实验结果表明，多模态感觉融合的益处，并突显了神经形态方法在实现机器人系统中高效、鲁棒且类人感知方面的潜力。 

---
# Adaptive Navigation Strategy for Low-Thrust Proximity Operations in Circular Relative Orbit 

**Title (ZH)**: 低推力近距离轨道操作的自适应导航策略 

**Authors**: Dario Ruggiero, Mauro Mancini, Elisa Capello  

**Link**: [PDF](https://arxiv.org/pdf/2509.02204)  

**Abstract**: This paper presents an adaptive observer-based navigation strategy for spacecraft in Circular Relative Orbit (CRO) scenarios, addressing challenges in proximity operations like formation flight and uncooperative target inspection. The proposed method adjusts observer gains based on the estimated state to achieve fast convergence and low noise sensitivity in state estimation. A Lyapunov-based analysis ensures stability and accuracy, while simulations using vision-based sensor data validate the approach under realistic conditions. Compared to classical observers with time-invariant gains, the proposed method enhances trajectory tracking precision and reduces control input switching, making it a promising solution for autonomous spacecraft localization and control. 

**Abstract (ZH)**: 基于自适应观测器的航天器环路相对轨道导航策略及其在接近操作中的应用 

---
# Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety 

**Title (ZH)**: 增强包含大型语言模型的机器人系统可靠性：安全性与可靠性统一方法 

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Bräunl, Jin B. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.02163)  

**Abstract**: Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at this https URL 

**Abstract (ZH)**: 将大规模语言模型（LLMs）整合到机器人系统中已革新了具身人工智能，使高级决策能力和适应性成为可能。然而，确保可靠性，包括对抗性攻击的安全性和复杂环境中的安全性，仍然是一个关键挑战。为了解决这一问题，我们提出了一种统一框架，该框架通过鲁棒验证机制减轻提示注入攻击并确保操作安全性。我们的方法结合了提示构建、状态管理以及安全性验证，并通过性能和安全性指标进行了评估。实验结果显示，在注入攻击下性能提升了30.8%，在对抗性条件下复杂环境设置下性能提升了325%，相较于基线场景。该工作在基于LLM的机器人系统中填补了安全性和安全性之间的空白，提供了在实际环境中部署可靠的LLM集成移动机器人的可操作见解。该框架在以下网址开源，并提供了模拟和物理部署的演示：https://this-url.com 

---
# Systematic Evaluation of Trade-Offs in Motion Planning Algorithms for Optimal Industrial Robotic Work Cell Design 

**Title (ZH)**: 运动规划算法在最优工业机器人工作站设计中的权衡系统评价 

**Authors**: G. de Mathelin, C. Hartl-Nesic, A. Kugi  

**Link**: [PDF](https://arxiv.org/pdf/2509.02146)  

**Abstract**: The performance of industrial robotic work cells depends on optimizing various hyperparameters referring to the cell layout, such as robot base placement, tool placement, and kinematic design. Achieving this requires a bilevel optimization approach, where the high-level optimization adjusts these hyperparameters, and the low-level optimization computes robot motions. However, computing the optimal robot motion is computationally infeasible, introducing trade-offs in motion planning to make the problem tractable. These trade-offs significantly impact the overall performance of the bilevel optimization, but their effects still need to be systematically evaluated. In this paper, we introduce metrics to assess these trade-offs regarding optimality, time gain, robustness, and consistency. Through extensive simulation studies, we investigate how simplifications in motion-level optimization affect the high-level optimization outcomes, balancing computational complexity with solution quality. The proposed algorithms are applied to find the time-optimal kinematic design for a modular robot in two palletization scenarios. 

**Abstract (ZH)**: 工业机器人工作单元的性能取决于优化各种超参数，如单元布局、机器人基座位置、工具位置和 kinematic 设计。这需要一种多层次优化方法，其中高层优化调整这些超参数，低层优化计算机器人运动。然而，计算最优的机器人运动在计算上是不可行的，引入了在运动规划中进行权衡以使问题可解。这些权衡显著影响多层次优化的整体性能，但它们的影响仍需要系统评估。在本文中，我们引入了评估这些权衡的指标，包括最优性、时间增益、鲁棒性和一致性。通过广泛的研究模拟，我们探讨了在运动层优化中进行简化的效果，平衡计算复杂度与解的质量。所提出的算法应用于在两种装盘场景下找到模块化机器人的时间最优 kinematic 设计。 

---
# Learning Social Heuristics for Human-Aware Path Planning 

**Title (ZH)**: 基于社会启发式的面向人类路径规划 

**Authors**: Andrea Eirale, Matteo Leonetti, Marcello Chiaberge  

**Link**: [PDF](https://arxiv.org/pdf/2509.02134)  

**Abstract**: Social robotic navigation has been at the center of numerous studies in recent years. Most of the research has focused on driving the robotic agent along obstacle-free trajectories, respecting social distances from humans, and predicting their movements to optimize navigation. However, in order to really be socially accepted, the robots must be able to attain certain social norms that cannot arise from conventional navigation, but require a dedicated learning process. We propose Heuristic Planning with Learned Social Value (HPLSV), a method to learn a value function encapsulating the cost of social navigation, and use it as an additional heuristic in heuristic-search path planning. In this preliminary work, we apply the methodology to the common social scenario of joining a queue of people, with the intention of generalizing to further human activities. 

**Abstract (ZH)**: 社会机器人导航在近年来的研究中处于中心地位。大多数研究集中在引导机器人沿无障碍轨迹行驶，遵守与人类的社会距离，并预测人类的移动以优化导航。然而，为了真正被社会接受，机器人必须具备某些不能通过传统导航获得的社会规范，而需要一个专门的学习过程。我们提出了一种启发式规划与学习的社会价值方法（HPLSV），用于学习一个包含社会导航成本的价值函数，并将其用作启发式搜索路径规划中的附加启发式方法。在本初步工作中，我们将该方法应用于人们常见的排队场景，旨在进一步推广到更多的人类活动。 

---
# A Geometric Method for Base Parameter Analysis in Robot Inertia Identification Based on Projective Geometric Algebra 

**Title (ZH)**: 基于射影几何代数的机器人惯性参数基元分析的几何方法 

**Authors**: Guangzhen Sun, Ye Ding, Xiangyang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02071)  

**Abstract**: This paper proposes a novel geometric method for analytically determining the base inertial parameters of robotic systems. The rigid body dynamics is reformulated using projective geometric algebra, leading to a new identification model named ``tetrahedral-point (TP)" model. Based on the rigid body TP model, coefficients in the regresoor matrix of the identification model are derived in closed-form, exhibiting clear geometric interpretations. Building directly from the dynamic model, three foundational principles for base parameter analysis are proposed: the shared points principle, fixed points principle, and planar rotations principle. With these principles, algorithms are developed to automatically determine all the base parameters. The core algorithm, referred to as Dynamics Regressor Nullspace Generator (DRNG), achieves $O(1)$-complexity theoretically following an $O(N)$-complexity preprocessing stage, where $N$ is the number of rigid bodies. The proposed method and algorithms are validated across four robots: Puma560, Unitree Go2, a 2RRU-1RRS parallel kinematics mechanism (PKM), and a 2PRS-1PSR PKM. In all cases, the algorithms successfully identify the complete set of base parameters. Notably, the approach demonstrates high robustness and computational efficiency, particularly in the cases of PKMs. Through the comprehensive demonstrations, the method is shown to be general, robust, and efficient. 

**Abstract (ZH)**: 一种基于几何的方法用于机器人系统基座惯性参数的解析确定 

---
# Align-Then-stEer: Adapting the Vision-Language Action Models through Unified Latent Guidance 

**Title (ZH)**: 对齐然后定向：通过统一潜在指导调整视觉-语言行动模型 

**Authors**: Yang Zhang, Chenwei Wang, Ouyang Lu, Yuan Zhao, Yunfei Ge, Zhenglong Sun, Xiu Li, Chi Zhang, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02055)  

**Abstract**: Vision-Language-Action (VLA) models pre-trained on large, diverse datasets show remarkable potential for general-purpose robotic manipulation. However, a primary bottleneck remains in adapting these models to downstream tasks, especially when the robot's embodiment or the task itself differs from the pre-training data. This discrepancy leads to a significant mismatch in action distributions, demanding extensive data and compute for effective fine-tuning. To address this challenge, we introduce \textbf{Align-Then-stEer (\texttt{ATE})}, a novel, data-efficient, and plug-and-play adaptation framework. \texttt{ATE} first aligns disparate action spaces by constructing a unified latent space, where a variational autoencoder constrained by reverse KL divergence embeds adaptation actions into modes of the pre-training action latent distribution. Subsequently, it steers the diffusion- or flow-based VLA's generation process during fine-tuning via a guidance mechanism that pushes the model's output distribution towards the target domain. We conduct extensive experiments on cross-embodiment and cross-task manipulation in both simulation and real world. Compared to direct fine-tuning of representative VLAs, our method improves the average multi-task success rate by up to \textbf{9.8\%} in simulation and achieves a striking \textbf{32\% success rate gain} in a real-world cross-embodiment setting. Our work presents a general and lightweight solution that greatly enhances the practicality of deploying VLA models to new robotic platforms and tasks. 

**Abstract (ZH)**: Vision-语言-行动（VLA）模型在大规模多样的数据集上预训练后，在通用机器人操作方面显示出显著的潜力。然而，将这些模型适应下游任务的主要瓶颈在于，当机器人本体或任务本身与预训练数据不同步时。这种差异导致了行动分布的显著不匹配，从而要求大量的数据和计算资源才能有效进行微调。为了解决这一挑战，我们提出了**Align-Then-stEer（ATE）**，一种新颖的数据高效且即插即用的适应框架。ATE首先通过构建统一的潜空间来对齐不同的行动空间，在该空间中，受限于逆KL散度的变分自编码器将适应动作嵌入到预训练行动潜分布的模式中。随后，它通过一种引导机制，在微调过程中引导以流或扩散为基础的VLA生成过程，将模型的输出分布推向目标领域。我们分别在仿真和真实世界中进行了跨越本体和任务的操纵的广泛实验。与直接微调代表性VLA相比，我们的方法在仿真中的多任务成功率平均提高了**9.8%**，在真实世界的跨本体设置中成功率达到惊人的**32%**的提升。我们的工作提供了一个通用且轻量级的解决方案，大大增强了将VLA模型部署到新机器人平台和任务的实际可行性。 

---
# Generalizing Unsupervised Lidar Odometry Model from Normal to Snowy Weather Conditions 

**Title (ZH)**: 从普通天气条件到雪天条件的无监督激光里程计模型泛化研究 

**Authors**: Beibei Zhou, Zhiyuan Zhang, Zhenbo Song, Jianhui Guo, Hui Kong  

**Link**: [PDF](https://arxiv.org/pdf/2509.02011)  

**Abstract**: Deep learning-based LiDAR odometry is crucial for autonomous driving and robotic navigation, yet its performance under adverse weather, especially snowfall, remains challenging. Existing models struggle to generalize across conditions due to sensitivity to snow-induced noise, limiting real-world use. In this work, we present an unsupervised LiDAR odometry model to close the gap between clear and snowy weather conditions. Our approach focuses on effective denoising to mitigate the impact of snowflake noise and outlier points on pose estimation, while also maintaining computational efficiency for real-time applications.
To achieve this, we introduce a Patch Spatial Measure (PSM) module that evaluates the dispersion of points within each patch, enabling effective detection of sparse and discrete noise.
We further propose a Patch Point Weight Predictor (PPWP) to assign adaptive point-wise weights, enhancing their discriminative capacity within local regions. To support real-time performance, we first apply an intensity threshold mask to quickly suppress dense snowflake clusters near the LiDAR, and then perform multi-modal feature fusion to refine the point-wise weight prediction, improving overall robustness under adverse weather. Our model is trained in clear weather conditions and rigorously tested across various scenarios, including snowy and dynamic. Extensive experimental results confirm the effectiveness of our method, demonstrating robust performance in both clear and snowy weather. This advancement enhances the model's generalizability and paves the way for more reliable autonomous systems capable of operating across a wider range of environmental conditions. 

**Abstract (ZH)**: 基于深度学习的LiDAR里程计对于自动驾驶和机器人导航至关重要，但在恶劣天气尤其是雪天下的性能仍具挑战性。现有模型由于对雪诱导噪声的敏感性难以在不同条件下泛化，限制了其在实际中的应用。本文提出了一种无监督的LiDAR里程计模型，以缩小晴天和雪天条件之间的差距。该方法重点在于有效的去噪，以减轻雪花噪声和离群点对姿态估计的影响，同时保持对实时应用的计算效率。

为此，我们引入了补丁空间度量（PSM）模块，该模块评估每个补丁内点的分散度，以有效检测稀疏且离散的噪声。

我们还提出了补丁点权重预测器（PPWP），以分配自适应的点权重，增强其在局部区域的判别能力。为支持实时性能，我们首先应用强度阈值掩码快速抑制LiDAR附近的密集雪花簇，然后进行多模态特征融合以细化点权重预测，从而在恶劣天气下提高整体鲁棒性。我们的模型在晴朗天气条件下训练，并在各种场景下进行严格的测试，包括雪天和动态场景。广泛的实验结果证实了该方法的有效性，展示了模型在晴天和雪天条件下的稳健性能。这一进展提升了模型的泛化能力，并为在更广泛环境条件下运行的更可靠自主系统铺平了道路。 

---
# MIRAGE: Multimodal Intention Recognition and Admittance-Guided Enhancement in VR-based Multi-object Teleoperation 

**Title (ZH)**: MIRAGE: 多模态意图识别与接纳指导增强在基于VR的多对象远程操控中的应用 

**Authors**: Chi Sun, Xian Wang, Abhishek Kumar, Chengbin Cui, Lik-Hang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01996)  

**Abstract**: Effective human-robot interaction (HRI) in multi-object teleoperation tasks faces significant challenges due to perceptual ambiguities in virtual reality (VR) environments and the limitations of single-modality intention recognition. This paper proposes a shared control framework that combines a virtual admittance (VA) model with a Multimodal-CNN-based Human Intention Perception Network (MMIPN) to enhance teleoperation performance and user experience. The VA model employs artificial potential fields to guide operators toward target objects by adjusting admittance force and optimizing motion trajectories. MMIPN processes multimodal inputs, including gaze movement, robot motions, and environmental context, to estimate human grasping intentions, helping to overcome depth perception challenges in VR. Our user study evaluated four conditions across two factors, and the results showed that MMIPN significantly improved grasp success rates, while the VA model enhanced movement efficiency by reducing path lengths. Gaze data emerged as the most crucial input modality. These findings demonstrate the effectiveness of combining multimodal cues with implicit guidance in VR-based teleoperation, providing a robust solution for multi-object grasping tasks and enabling more natural interactions across various applications in the future. 

**Abstract (ZH)**: 多模态 CNN 基于的人意图感知网络与虚拟阻抗模型的结合在多对象远程操作中的有效人机交互 

---
# Geometric Control of Mechanical Systems with Symmetries Based on Sliding Modes 

**Title (ZH)**: 基于滑模的具有对称性的机械系统几何控制 

**Authors**: Eduardo Espindola, Yu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01985)  

**Abstract**: In this paper, we propose a framework for designing sliding mode controllers for a class of mechanical systems with symmetry, both unconstrained and constrained, that evolve on principal fiber bundles. Control laws are developed based on the reduced motion equations by exploring symmetries, leading to a sliding mode control strategy where the reaching stage is executed on the base space, and the sliding stage is performed on the structure group. Thus, design complexity is reduced, and difficult choices for coordinate representations when working with a particular Lie group are avoided. For this purpose, a sliding subgroup is constructed on the structure group based on a kinematic controller, and the sliding variable will converge to the identity of the state manifold upon reaching the sliding subgroup. A reaching law based on a general sliding vector field is then designed on the base space using the local form of the mechanical connection to drive the sliding variable to the sliding subgroup, and its time evolution is given according to the appropriate covariant derivative. Almost global asymptotic stability and local exponential stability are demonstrated using a Lyapunov analysis. We apply the results to a fully actuated system (a rigid spacecraft actuated by reaction wheels) and a subactuated nonholonomic system (unicycle mobile robot actuated by wheels), which is also simulated for illustration. 

**Abstract (ZH)**: 本文提出了一类在主纤维丛上运动的对称机械系统滑模控制器设计框架，包括无约束和受约束系统。基于运动方程的降维并通过利用对称性开发控制律，提出了一种滑模控制策略，在基空间执行到达阶段，在结构群上执行滑模阶段，从而降低了设计复杂性，避免了在特定李群下使用特定坐标表示时的困难选择。为此，基于动力学控制器构造滑动子群，到达滑动子群后，滑动变量收敛于状态流形上的单位元。利用机械联络的局部形式在基空间设计基于通用滑动向量场的到达定律，驱动滑动变量到达滑动子群，并根据适当的协变导数给出其时间演变。通过李雅普unov分析证明了几乎全局渐近稳定性和局部指数稳定性。将结果应用于完全驱动系统（由反应轮驱动的刚体航天器）和部分驱动非完整系统（由车轮驱动的单轨移动机器人），并进行了仿真以进行说明。 

---
# Hybrid Autonomy Framework for a Future Mars Science Helicopter 

**Title (ZH)**: 未来火星科学旋翼机的混合自主体系架构 

**Authors**: Luca Di Pierno, Robert Hewitt, Stephan Weiss, Roland Brockers  

**Link**: [PDF](https://arxiv.org/pdf/2509.01980)  

**Abstract**: Autonomous aerial vehicles, such as NASA's Ingenuity, enable rapid planetary surface exploration beyond the reach of ground-based robots. Thus, NASA is studying a Mars Science Helicopter (MSH), an advanced concept capable of performing long-range science missions and autonomously navigating challenging Martian terrain. Given significant Earth-Mars communication delays and mission complexity, an advanced autonomy framework is required to ensure safe and efficient operation by continuously adapting behavior based on mission objectives and real-time conditions, without human intervention. This study presents a deterministic high-level control framework for aerial exploration, integrating a Finite State Machine (FSM) with Behavior Trees (BTs) to achieve a scalable, robust, and computationally efficient autonomy solution for critical scenarios like deep space exploration. In this paper we outline key capabilities of a possible MSH and detail the FSM-BT hybrid autonomy framework which orchestrates them to achieve the desired objectives. Monte Carlo simulations and real field tests validate the framework, demonstrating its robustness and adaptability to both discrete events and real-time system feedback. These inputs trigger state transitions or dynamically adjust behavior execution, enabling reactive and context-aware responses. The framework is middleware-agnostic, supporting integration with systems like F-Prime and extending beyond aerial robotics. 

**Abstract (ZH)**: 自主飞行器如NASA的Ingenuity使超越地面机器人范围的行星表面快速探索成为可能。因此，NASA正在研究火星科学旋翼飞机（MSH）这一先进概念，该概念能够执行远程科学任务并自主导航火星复杂地形。鉴于地火通信延迟和任务的复杂性，需要一个高级自主框架，以在不依赖人类干预的情况下，根据任务目标和实时条件持续适应行为，确保安全高效的操作。本文提出了一种确定性的高层控制框架，用于空中探索，将有限状态机（FSM）与行为树（BTs）结合，以实现适用于关键场景（如深空探索）的可扩展、可靠且计算高效的自主解决方案。本文概述了一种可能的MSH的关键能力，并详细描述了FSM-BT混合自主框架，该框架协调这些能力以实现 desired 目标。蒙特卡洛模拟和实地测试验证了该框架，证明了其对离散事件和实时系统反馈的高度稳健性和适应性。这些输入触发状态转换或动态调整行为执行，实现反应性和上下文感知的响应。该框架是一种中间件agnostic，支持与F-Prime等系统集成，并超越空中机器人领域。 

---
# AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving 

**Title (ZH)**: AutoDrive-R$^2$: 激励VLA模型在自动驾驶中进行推理和自我反思能力 

**Authors**: Zhenlong Yuan, Jing Tang, Jinguo Luo, Rui Chen, Chengxuan Qian, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.01944)  

**Abstract**: Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method. 

**Abstract (ZH)**: 基于视觉-语言-行动（VLA）模型的自动驾驶系统通过多模态感知与决策能力的结合展现出变革性的潜力，然而其决策过程的可解释性和行动序列的连贯性及合理性仍需进一步探索。为解决这些问题，我们提出了一种名为AutoDrive-R$^2$的新型VLA框架，通过链式思考（CoT）处理和强化学习（RL）增强自动驾驶系统的推理和自我反思能力。具体地，我们首先提出了一种创新的CoT数据集nuScenesR$^2$-6K，用于监督微调，该数据集通过四步逻辑链和自我反思有效地建立了输入信息与输出轨迹之间的认知桥梁。此外，为了在RL阶段最大化推理和自我反思能力，我们进一步采用基于物理的奖励框架中的Group Relative Policy Optimization（GRPO）算法，该框架整合了空间对齐、车辆动力学和时间连贯性的标准，以确保可靠的和现实的轨迹规划。我们的方法在nuScenes和Waymo数据集上的广泛评估结果展示了其领先性能和强大的泛化能力。 

---
# AI-Driven Marine Robotics: Emerging Trends in Underwater Perception and Ecosystem Monitoring 

**Title (ZH)**: AI驱动的海洋机器人：水上探测与生态系统监测的新兴趋势 

**Authors**: Scarlett Raine, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.01878)  

**Abstract**: Marine ecosystems face increasing pressure due to climate change, driving the need for scalable, AI-powered monitoring solutions. This paper examines the rapid emergence of underwater AI as a major research frontier and analyzes the factors that have transformed marine perception from a niche application into a catalyst for AI innovation. We identify three convergent drivers: environmental necessity for ecosystem-scale monitoring, democratization of underwater datasets through citizen science platforms, and researcher migration from saturated terrestrial computer vision domains. Our analysis reveals how unique underwater challenges - turbidity, cryptic species detection, expert annotation bottlenecks, and cross-ecosystem generalization - are driving fundamental advances in weakly supervised learning, open-set recognition, and robust perception under degraded conditions. We survey emerging trends in datasets, scene understanding and 3D reconstruction, highlighting the paradigm shift from passive observation toward AI-driven, targeted intervention capabilities. The paper demonstrates how underwater constraints are pushing the boundaries of foundation models, self-supervised learning, and perception, with methodological innovations that extend far beyond marine applications to benefit general computer vision, robotics, and environmental monitoring. 

**Abstract (ZH)**: 海洋生态系统由于气候变化面临不断增加的压力，推动了可扩展的AI驱动监测解决方案的需求。本文探讨了水下AI作为重要研究前沿的迅猛发展，并分析了将其从 niche 应用转变为AI创新催化剂的因素。我们确定了三种相互作用的驱动力：环境需求促使生态系统规模的监测，通过公民科学平台普及水下数据集，以及研究人员从饱和的陆地计算机视觉领域转向。我们的分析揭示了水下独特挑战——浑浊度、隐匿物种检测、专家注释瓶颈以及跨生态系统的一般化——如何推动了弱监督学习、开放集识别和退化条件下鲁棒感知的基石性进步。我们概述了新兴趋势，包括数据集、场景理解和3D重建，展示了从被动观察向AI驱动的目标干预能力的范式转变。本文展示了水下约束如何推动基础模型、自监督学习和感知的边界，其中方法上的创新远不止于海洋应用，还能惠及通用计算机视觉、机器人技术和环境监测。 

---
# Multi-vessel Interaction-Aware Trajectory Prediction and Collision Risk Assessment 

**Title (ZH)**: 多血管交互aware轨迹预测与碰撞风险评估 

**Authors**: Md Mahbub Alam, Jose F. Rodrigues-Jr, Gabriel Spadon  

**Link**: [PDF](https://arxiv.org/pdf/2509.01836)  

**Abstract**: Accurate vessel trajectory prediction is essential for enhancing situational awareness and preventing collisions. Still, existing data-driven models are constrained mainly to single-vessel forecasting, overlooking vessel interactions, navigation rules, and explicit collision risk assessment. We present a transformer-based framework for multi-vessel trajectory prediction with integrated collision risk analysis. For a given target vessel, the framework identifies nearby vessels. It jointly predicts their future trajectories through parallel streams encoding kinematic and derived physical features, causal convolutions for temporal locality, spatial transformations for positional encoding, and hybrid positional embeddings that capture both local motion patterns and long-range dependencies. Evaluated on large-scale real-world AIS data using joint multi-vessel metrics, the model demonstrates superior forecasting capabilities beyond traditional single-vessel displacement errors. By simulating interactions among predicted trajectories, the framework further quantifies potential collision risks, offering actionable insights to strengthen maritime safety and decision support. 

**Abstract (ZH)**: 基于变压器的多船轨迹预测及其碰撞风险分析框架 

---
# ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training 

**Title (ZH)**: ManiFlow：一致性流训练的通用机器人 manipulation 策略 

**Authors**: Ge Yan, Jiyue Zhu, Yuquan Deng, Shiqi Yang, Ri-Zhao Qiu, Xuxin Cheng, Marius Memmel, Ranjay Krishna, Ankit Goyal, Xiaolong Wang, Dieter Fox  

**Link**: [PDF](https://arxiv.org/pdf/2509.01819)  

**Abstract**: This paper introduces ManiFlow, a visuomotor imitation learning policy for general robot manipulation that generates precise, high-dimensional actions conditioned on diverse visual, language and proprioceptive inputs. We leverage flow matching with consistency training to enable high-quality dexterous action generation in just 1-2 inference steps. To handle diverse input modalities efficiently, we propose DiT-X, a diffusion transformer architecture with adaptive cross-attention and AdaLN-Zero conditioning that enables fine-grained feature interactions between action tokens and multi-modal observations. ManiFlow demonstrates consistent improvements across diverse simulation benchmarks and nearly doubles success rates on real-world tasks across single-arm, bimanual, and humanoid robot setups with increasing dexterity. The extensive evaluation further demonstrates the strong robustness and generalizability of ManiFlow to novel objects and background changes, and highlights its strong scaling capability with larger-scale datasets. Our website: this http URL. 

**Abstract (ZH)**: 本文介绍了ManiFlow，一种基于视觉和运动模仿学习的通用机器人 manipulation 策略，能够根据多样化视觉、语言和本体感受输入生成精确的高维动作。我们利用流匹配和一致性训练，使机器人能够在仅1-2个推理步骤中生成高质量的灵巧动作。为有效处理多种输入模态，我们提出了一种名为DiT-X的扩散变压器架构，该架构具有自适应交叉注意力和AdaLN-Zero条件，能够实现动作标记与多模态观测之间的细粒度特征交互。ManiFlow在多种模拟基准测试中表现出一致的改进，并且在单臂、双手和类人机器人配置中成功执行各种任务，成功率几乎翻倍，随着灵巧程度的增加而提高。广泛的评估进一步证明了ManiFlow对新物体和背景变化的强健性和泛化能力，并突显了其在更大规模数据集上的强大扩展能力。我们的网站：[此处填网址]。 

---
# Non-conflicting Energy Minimization in Reinforcement Learning based Robot Control 

**Title (ZH)**: 基于强化学习的机器人控制中非冲突的能量最小化 

**Authors**: Skand Peri, Akhil Perincherry, Bikram Pandit, Stefan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01765)  

**Abstract**: Efficient robot control often requires balancing task performance with energy expenditure. A common approach in reinforcement learning (RL) is to penalize energy use directly as part of the reward function. This requires carefully tuning weight terms to avoid undesirable trade-offs where energy minimization harms task success. In this work, we propose a hyperparameter-free gradient optimization method to minimize energy expenditure without conflicting with task performance. Inspired by recent works in multitask learning, our method applies policy gradient projection between task and energy objectives to derive policy updates that minimize energy expenditure in ways that do not impact task performance. We evaluate this technique on standard locomotion benchmarks of DM-Control and HumanoidBench and demonstrate a reduction of 64% energy usage while maintaining comparable task performance. Further, we conduct experiments on a Unitree GO2 quadruped showcasing Sim2Real transfer of energy efficient policies. Our method is easy to implement in standard RL pipelines with minimal code changes, is applicable to any policy gradient method, and offers a principled alternative to reward shaping for energy efficient control policies. 

**Abstract (ZH)**: 无需超参数的梯度优化方法在不冲突于任务性能的前提下最小化能量消耗 

---
# Fail2Progress: Learning from Real-World Robot Failures with Stein Variational Inference 

**Title (ZH)**: 从Stein变分推断学习现实世界机器人故障 

**Authors**: Yixuan Huang, Novella Alvina, Mohanraj Devendran Shanthi, Tucker Hermans  

**Link**: [PDF](https://arxiv.org/pdf/2509.01746)  

**Abstract**: Skill effect models for long-horizon manipulation tasks are prone to failures in conditions not covered by training data distributions. Therefore, enabling robots to reason about and learn from failures is necessary. We investigate the problem of efficiently generating a dataset targeted to observed failures. After fine-tuning a skill effect model on this dataset, we evaluate the extent to which the model can recover from failures and minimize future failures. We propose Fail2Progress, an approach that leverages Stein variational inference to generate multiple simulation environments in parallel, enabling efficient data sample generation similar to observed failures. Our method is capable of handling several challenging mobile manipulation tasks, including transporting multiple objects, organizing a constrained shelf, and tabletop organization. Through large-scale simulation and real-world experiments, we demonstrate that our approach excels at learning from failures across different numbers of objects. Furthermore, we show that Fail2Progress outperforms several baselines. 

**Abstract (ZH)**: 长时操作任务的能力影响模型在未覆盖训练数据分布的条件下容易发生故障，因此让机器人能够推理和学习故障是必要的。我们研究了生成针对观察到的故障的目标化数据集的问题。在对这个数据集微调能力影响模型后，我们评估了模型从故障中恢复以及减少未来故障的程度。我们提出了Fail2Progress方法，该方法利用Stein变分推断并行生成多个仿真环境，实现类似于观察到的故障的高效数据样本生成。我们的方法能够处理多个具有挑战性的移动操作任务，包括多物体搬运、受限货架整理和桌面整理。通过大规模仿真和实际实验，我们展示了我们的方法在不同物体数量下从故障中学习方面的优越性。此外，我们表明Fail2Progress优于几个基准方法。 

---
# Constrained Decoding for Robotics Foundation Models 

**Title (ZH)**: 受限解码for机器人基础模型 

**Authors**: Parv Kapoor, Akila Ganlath, Changliu Liu, Sebastian Scherer, Eunsuk Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01728)  

**Abstract**: Recent advances in the development of robotic foundation models have led to promising end-to-end and general-purpose capabilities in robotic systems. These models are pretrained on vast datasets of robot trajectories to process multi- modal inputs and directly output a sequence of action that the system then executes in the real world. Although this approach is attractive from the perspective of im- proved generalization across diverse tasks, these models are still data-driven and, therefore, lack explicit notions of behavioral correctness and safety constraints. We address these limitations by introducing a constrained decoding framework for robotics foundation models that enforces logical constraints on action trajec- tories in dynamical systems. Our method ensures that generated actions provably satisfy signal temporal logic (STL) specifications at runtime without retraining, while remaining agnostic of the underlying foundation model. We perform com- prehensive evaluation of our approach across state-of-the-art navigation founda- tion models and we show that our decoding-time interventions are useful not only for filtering unsafe actions but also for conditional action-generation. Videos available on our website: this https URL 

**Abstract (ZH)**: 近期在机器人基础模型发展的进步已经引发了在机器人系统中端到端和通用能力的有希望的成果。这些模型在大规模机器人轨迹数据集上进行预训练，以处理多模式输入并直接输出系统在真实世界中执行的一系列动作。尽管从增强跨多种任务的一般化角度来看这种方法具有吸引力，但这些模型仍然是数据驱动的，因此缺乏明确的行为正确性和安全约束的概念。我们通过引入一种受限解码框架来解决这些限制，该框架在动态系统中对动作轨迹施加逻辑约束。我们的方法确保在运行时生成的动作可以证明满足信号时序逻辑（STL）规范，而无需重新训练，并且对底层基础模型保持无偏见。我们对最先进的导航基础模型进行了全面评估，并表明我们的解码时干预不仅有助于筛选出不安全的动作，还可以用于条件动作生成。更多视频请参见我们的网站: this https URL 

---
# Articulated Object Estimation in the Wild 

**Title (ZH)**: 野外 articulated 对象估计 

**Authors**: Abdelrhman Werby, Martin Büchner, Adrian Röfer, Chenguang Huang, Wolfram Burgard, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2509.01708)  

**Abstract**: Understanding the 3D motion of articulated objects is essential in robotic scene understanding, mobile manipulation, and motion planning. Prior methods for articulation estimation have primarily focused on controlled settings, assuming either fixed camera viewpoints or direct observations of various object states, which tend to fail in more realistic unconstrained environments. In contrast, humans effortlessly infer articulation by watching others manipulate objects. Inspired by this, we introduce ArtiPoint, a novel estimation framework that can infer articulated object models under dynamic camera motion and partial observability. By combining deep point tracking with a factor graph optimization framework, ArtiPoint robustly estimates articulated part trajectories and articulation axes directly from raw RGB-D videos. To foster future research in this domain, we introduce Arti4D, the first ego-centric in-the-wild dataset that captures articulated object interactions at a scene level, accompanied by articulation labels and ground-truth camera poses. We benchmark ArtiPoint against a range of classical and learning-based baselines, demonstrating its superior performance on Arti4D. We make code and Arti4D publicly available at this https URL. 

**Abstract (ZH)**: 理解 articulated 对象的 3D 运动在机器人场景理解、移动操作和运动规划中至关重要。以往的articulation 估计方法主要集中在受控环境中，假设要么是固定的相机视角，要么是对各种对象状态的直接观察，这些方法在更具现实感的非受限环境中往往失效。与此相反，人类通过观看他人操作对象而轻松推断出articulation。受此启发，我们提出了一种新的估计框架 ArtiPoint，该框架能够在动态相机运动和部分可观测性条件下推断 articulated 对象模型。通过将深度点跟踪与因子图优化框架相结合，ArtiPoint 直接从原始 RGB-D 视频中稳健地估计出 articulated 部分轨迹和 articulation 轴。为了促进该领域的未来研究，我们引入了 Arti4D，这是第一个以第一人称视角捕捉场景级别 articulated 对象交互的野外数据集，并附带articulation 标签和真实相机姿态。我们在 Arti4D 上对 ArtiPoint 进行基准测试，展示了其优于多种经典和学习基线的方法性能。我们已在以下网址公开发布代码和 Arti4D：this https URL。 

---
# MoTo: A Zero-shot Plug-in Interaction-aware Navigation for General Mobile Manipulation 

**Title (ZH)**: MoTo: 零样本插件式交互感知导航用于通用移动操作 

**Authors**: Zhenyu Wu, Angyuan Ma, Xiuwei Xu, Hang Yin, Yinan Liang, Ziwei Wang, Jiwen Lu, Haibin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01658)  

**Abstract**: Mobile manipulation stands as a core challenge in robotics, enabling robots to assist humans across varied tasks and dynamic daily environments. Conventional mobile manipulation approaches often struggle to generalize across different tasks and environments due to the lack of large-scale training. However, recent advances in manipulation foundation models demonstrate impressive generalization capability on a wide range of fixed-base manipulation tasks, which are still limited to a fixed setting. Therefore, we devise a plug-in module named MoTo, which can be combined with any off-the-shelf manipulation foundation model to empower them with mobile manipulation ability. Specifically, we propose an interaction-aware navigation policy to generate robot docking points for generalized mobile manipulation. To enable zero-shot ability, we propose an interaction keypoints framework via vision-language models (VLM) under multi-view consistency for both target object and robotic arm following instructions, where fixed-base manipulation foundation models can be employed. We further propose motion planning objectives for the mobile base and robot arm, which minimize the distance between the two keypoints and maintain the physical feasibility of trajectories. In this way, MoTo guides the robot to move to the docking points where fixed-base manipulation can be successfully performed, and leverages VLM generation and trajectory optimization to achieve mobile manipulation in a zero-shot manner, without any requirement on mobile manipulation expert data. Extensive experimental results on OVMM and real-world demonstrate that MoTo achieves success rates of 2.68% and 16.67% higher than the state-of-the-art mobile manipulation methods, respectively, without requiring additional training data. 

**Abstract (ZH)**: 移动操作是机器人学中的核心挑战，使机器人能够在多种任务和动态日常环境中辅助人类。传统的移动操作方法由于缺乏大规模训练数据，往往难以在不同任务和环境中泛化。然而，近期的操纵基础模型进展展示了在一系列固定基座操作任务上出色的泛化能力，这些任务仍局限于静态环境。因此，我们设计了一个名为MoTo的插件模块，可以与任何现成的操纵基础模型结合，赋予其实现移动操作的能力。具体而言，我们提出了一种交互感知导航策略以生成通用移动操作的机器人对接点。为了实现零样本能力，我们通过多视角一致性下的视觉-语言模型（VLM）提出了交互关键点框架，用于目标对象和机器人臂跟随指令的场景，可以利用固定的基座操作基础模型。我们进一步提出了移动基座和机器人臂的动力学规划目标，最小化两个关键点之间的距离并保持轨迹的物理可行性。通过这种方式，MoTo指导机器人移动到可以成功执行固定基座操作的对接点，并利用VLM生成和轨迹优化以零样本方式实现移动操作，无需移动操作专家数据。在OVMM和实际环境中的广泛实验结果显示，MoTo分别比最先进的移动操作方法成功率达到2.68%和16.67%的提升，且无需额外的训练数据。 

---
# Data Retrieval with Importance Weights for Few-Shot Imitation Learning 

**Title (ZH)**: 带有重要性加权的数据检索在少样本模仿学习中的应用 

**Authors**: Amber Xie, Rahul Chand, Dorsa Sadigh, Joey Hejna  

**Link**: [PDF](https://arxiv.org/pdf/2509.01657)  

**Abstract**: While large-scale robot datasets have propelled recent progress in imitation learning, learning from smaller task specific datasets remains critical for deployment in new environments and unseen tasks. One such approach to few-shot imitation learning is retrieval-based imitation learning, which extracts relevant samples from large, widely available prior datasets to augment a limited demonstration dataset. To determine the relevant data from prior datasets, retrieval-based approaches most commonly calculate a prior data point's minimum distance to a point in the target dataset in latent space. While retrieval-based methods have shown success using this metric for data selection, we demonstrate its equivalence to the limit of a Gaussian kernel density (KDE) estimate of the target data distribution. This reveals two shortcomings of the retrieval rule used in prior work. First, it relies on high-variance nearest neighbor estimates that are susceptible to noise. Second, it does not account for the distribution of prior data when retrieving data. To address these issues, we introduce Importance Weighted Retrieval (IWR), which estimates importance weights, or the ratio between the target and prior data distributions for retrieval, using Gaussian KDEs. By considering the probability ratio, IWR seeks to mitigate the bias of previous selection rules, and by using reasonable modeling parameters, IWR effectively smooths estimates using all data points. Across both simulation environments and real-world evaluations on the Bridge dataset we find that our method, IWR, consistently improves performance of existing retrieval-based methods, despite only requiring minor modifications. 

**Abstract (ZH)**: 基于检索的少样本模仿学习中重要性加权检索方法的研究 

---
# Speculative Design of Equitable Robotics: Queer Fictions and Futures 

**Title (ZH)**: 推测性设计公平机器人：奇异fiction与未来 

**Authors**: Minja Axelsson  

**Link**: [PDF](https://arxiv.org/pdf/2509.01643)  

**Abstract**: This paper examines the speculative topic of equitable robots through an exploratory essay format. It focuses specifically on robots by and for LGBTQ+ populations. It aims to provoke thought and conversations in the field about what aspirational queer robotics futures may look like, both in the arts and sciences. First, it briefly reviews the state-of-the-art of queer robotics in fiction and science, drawing together threads from each. Then, it discusses queering robots through three speculative design proposals for queer robot roles: 1) reflecting the queerness of their ''in-group'' queer users, building and celebrating ''in-group'' identity, 2) a new kind of queer activism by implementing queer robot identity performance to interact with ''out-group'' users, with a goal of reducing bigotry through familiarisation, and 3) a network of queer-owned robots, through which the community could reach each other, and distribute and access important resources. The paper then questions whether robots should be queered, and what ethical implications this raises. Finally, the paper makes suggestions for what aspirational queer robotics futures may look like, and what would be required to get there. 

**Abstract (ZH)**: 本文通过探索性散文的形式考察了公平机器人这一富有争议的话题，重点关注为LGBTQ+群体服务的机器人。它旨在引发学术界关于理想中的异性恋机器人未来的思考，涵盖艺术和科学领域。首先，本文简要回顾了虚构和科学中异性恋机器人领域的现状，梳理了两者的相关线索。然后，通过三个关于同性恋机器人角色的设想讨论了如何“同性恋化”机器人：1）反映其“小群体”同性恋用户的身份特征，构建和庆祝“小群体”身份；2）一种新的同性恋主义形式，通过实施同性恋机器人身份表演与“非小群体”用户互动，目标是通过熟悉度减少偏见；3）一个由同性恋所有者经营的机器人网络，使得社群能够相互联系，并分配和获取重要的资源。本文随后探讨了是否应该“同性恋化”机器人，以及这一做法带来的伦理问题。最后，本文提出了理想中的同性恋机器人未来的愿景，以及实现这些愿景所需的前提条件。 

---
# A Hybrid Input based Deep Reinforcement Learning for Lane Change Decision-Making of Autonomous Vehicle 

**Title (ZH)**: 基于混合输入的深度强化学习的自主车辆变道决策方法 

**Authors**: Ziteng Gao, Jiaqi Qu, Chaoyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.01611)  

**Abstract**: Lane change decision-making for autonomous vehicles is a complex but high-reward behavior. In this paper, we propose a hybrid input based deep reinforcement learning (DRL) algorithm, which realizes abstract lane change decisions and lane change actions for autonomous vehicles within traffic flow. Firstly, a surrounding vehicles trajectory prediction method is proposed to reduce the risk of future behavior of surrounding vehicles to ego vehicle, and the prediction results are input into the reinforcement learning model as additional information. Secondly, to comprehensively leverage environmental information, the model extracts feature from high-dimensional images and low-dimensional sensor data simultaneously. The fusion of surrounding vehicle trajectory prediction and multi-modal information are used as state space of reinforcement learning to improve the rationality of lane change decision. Finally, we integrate reinforcement learning macro decisions with end-to-end vehicle control to achieve a holistic lane change process. Experiments were conducted within the CARLA simulator, and the results demonstrated that the utilization of a hybrid state space significantly enhances the safety of vehicle lane change decisions. 

**Abstract (ZH)**: 自主驾驶车辆车道变更决策是一种复杂但高收益的行为：基于混合输入的深度强化学习算法及其应用研究 

---
# Aleatoric Uncertainty from AI-based 6D Object Pose Predictors for Object-relative State Estimation 

**Title (ZH)**: 基于AI的6D物体姿态预测器的 aleatoric 不确定性在物体相对状态估计中的应用 

**Authors**: Thomas Jantos, Stephan Weiss, Jan Steinbrener  

**Link**: [PDF](https://arxiv.org/pdf/2509.01583)  

**Abstract**: Deep Learning (DL) has become essential in various robotics applications due to excelling at processing raw sensory data to extract task specific information from semantic objects. For example, vision-based object-relative navigation relies on a DL-based 6D object pose predictor to provide the relative pose between the object and the robot as measurements to the robot's state estimator. Accurately knowing the uncertainty inherent in such Deep Neural Network (DNN) based measurements is essential for probabilistic state estimators subsequently guiding the robot's tasks. Thus, in this letter, we show that we can extend any existing DL-based object-relative pose predictor for aleatoric uncertainty inference simply by including two multi-layer perceptrons detached from the translational and rotational part of the DL predictor. This allows for efficient training while freezing the existing pre-trained predictor. We then use the inferred 6D pose and its uncertainty as a measurement and corresponding noise covariance matrix in an extended Kalman filter (EKF). Our approach induces minimal computational overhead such that the state estimator can be deployed on edge devices while benefiting from the dynamically inferred measurement uncertainty. This increases the performance of the object-relative state estimation task compared to a fix-covariance approach. We conduct evaluations on synthetic data and real-world data to underline the benefits of aleatoric uncertainty inference for the object-relative state estimation task. 

**Abstract (ZH)**: 基于深度学习的对象相对姿态预测中 aleatoric 不确定性推断在机器人状态估计中的应用 

---
# FGO-SLAM: Enhancing Gaussian SLAM with Globally Consistent Opacity Radiance Field 

**Title (ZH)**: FGO-SLAM：增强Gaussian SLAM的全局一致透明辐射场方法 

**Authors**: Fan Zhu, Yifan Zhao, Ziyu Chen, Biao Yu, Hui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01547)  

**Abstract**: Visual SLAM has regained attention due to its ability to provide perceptual capabilities and simulation test data for Embodied AI. However, traditional SLAM methods struggle to meet the demands of high-quality scene reconstruction, and Gaussian SLAM systems, despite their rapid rendering and high-quality mapping capabilities, lack effective pose optimization methods and face challenges in geometric reconstruction. To address these issues, we introduce FGO-SLAM, a Gaussian SLAM system that employs an opacity radiance field as the scene representation to enhance geometric mapping performance. After initial pose estimation, we apply global adjustment to optimize camera poses and sparse point cloud, ensuring robust tracking of our approach. Additionally, we maintain a globally consistent opacity radiance field based on 3D Gaussians and introduce depth distortion and normal consistency terms to refine the scene representation. Furthermore, after constructing tetrahedral grids, we identify level sets to directly extract surfaces from 3D Gaussians. Results across various real-world and large-scale synthetic datasets demonstrate that our method achieves state-of-the-art tracking accuracy and mapping performance. 

**Abstract (ZH)**: 视觉SLAM由于其为感知能力以及实体AI的仿真测试数据提供支持而重新获得了关注。然而，传统的SLAM方法难以满足高质量场景重建的需求，尽管高斯SLAM系统具有快速渲染和高质量建图能力，但缺乏有效的姿态优化方法，并在几何重建方面面临挑战。为此，我们引入了FGO-SLAM，这是一种采用不透明辐射场作为场景表示的高斯SLAM系统，以提高几何建图性能。在初始姿态估计之后，我们应用全局调整来优化相机姿态和稀疏点云，确保我们的方法具有鲁棒的跟踪能力。此外，基于三维高斯分布，我们维护全局一致的不透明辐射场，并引入深度失真和法线一致性项以细化场景表示。进一步地，在构建四面体网格之后，我们识别等值集以直接从三维高斯分布中提取表面。在多种真实世界和大规模合成数据集上的结果表明，我们的方法达到了最先进的跟踪精度和建图性能。 

---
# Analyzing Reluctance to Ask for Help When Cooperating With Robots: Insights to Integrate Artificial Agents in HRC 

**Title (ZH)**: 分析合作过程中拒绝向机器人求助的倾向：整合人工智能代理于人机协作中的见解 

**Authors**: Ane San Martin, Michael Hagenow, Julie Shah, Johan Kildal, Elena Lazkano  

**Link**: [PDF](https://arxiv.org/pdf/2509.01450)  

**Abstract**: As robot technology advances, collaboration between humans and robots will become more prevalent in industrial tasks. When humans run into issues in such scenarios, a likely future involves relying on artificial agents or robots for aid. This study identifies key aspects for the design of future user-assisting agents. We analyze quantitative and qualitative data from a user study examining the impact of on-demand assistance received from a remote human in a human-robot collaboration (HRC) assembly task. We study scenarios in which users require help and we assess their experiences in requesting and receiving assistance. Additionally, we investigate participants' perceptions of future non-human assisting agents and whether assistance should be on-demand or unsolicited. Through a user study, we analyze the impact that such design decisions (human or artificial assistant, on-demand or unsolicited help) can have on elicited emotional responses, productivity, and preferences of humans engaged in HRC tasks. 

**Abstract (ZH)**: 随着机器人技术的发展，人类与机器人在工业任务中的协作将更加普遍。当人类在这些场景中遇到问题时，未来很可能依赖人工代理或机器人提供帮助。本研究识别了未来用户辅助代理设计的关键方面。我们通过一项用户研究分析了在人机协作（HRC）装配任务中，从远程人类获得按需协助的影响，研究用户需要帮助的场景，并评估他们请求和接受帮助的经历。此外，我们探讨了参与者对未来非人类辅助代理的看法，以及协助应该是按需提供的还是未经请求的。通过用户研究，我们分析了这些设计决策（人类或人工助手，按需或未经请求的帮助）对参与HRC任务的人所引发的情感反应、生产率和偏好可能产生的影响。 

---
# TopoNav: Topological Graphs as a Key Enabler for Advanced Object Navigation 

**Title (ZH)**: TopoNav: 抽象拓扑图作为高级对象导航的关键使能器 

**Authors**: Peiran Liu, Qiang Zhang, Daojie Peng, Lingfeng Zhang, Yihao Qin, Hang Zhou, Jun Ma, Renjing Xu, Yiding Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.01364)  

**Abstract**: Object Navigation (ObjectNav) has made great progress with large language models (LLMs), but still faces challenges in memory management, especially in long-horizon tasks and dynamic scenes. To address this, we propose TopoNav, a new framework that leverages topological structures as spatial memory. By building and updating a topological graph that captures scene connections, adjacency, and semantic meaning, TopoNav helps agents accumulate spatial knowledge over time, retrieve key information, and reason effectively toward distant goals. Our experiments show that TopoNav achieves state-of-the-art performance on benchmark ObjectNav datasets, with higher success rates and more efficient paths. It particularly excels in diverse and complex environments, as it connects temporary visual inputs with lasting spatial understanding. 

**Abstract (ZH)**: 基于拓扑结构的空间记忆物体导航（TopoNav）在大型语言模型（LLMs）的驱动下取得了显著进展，但仍面临记忆管理的挑战，尤其是在长期任务和动态场景中。为解决这一问题，我们提出了一种新的框架TopoNav，该框架利用拓扑结构作为空间记忆。通过构建并更新一个捕获场景连接、相邻关系和语义意义的拓扑图，TopoNav帮助代理积累空间知识，检索关键信息，并有效地朝着远距离目标进行推理。我们的实验表明，TopoNav在基准物体导航数据集上取得了最先进的性能，具有更高的成功率和更高效的路径。特别是在多样性和复杂性环境中，TopoNav特别出色，因为它将临时的视觉输入与持久的空间理解相连接。 

---
# Disentangled Multi-Context Meta-Learning: Unlocking robust and Generalized Task Learning 

**Title (ZH)**: 解耦多上下文元学习：解锁稳健且通用的任务学习 

**Authors**: Seonsoo Kim, Jun-Gill Kang, Taehong Kim, Seongil Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01297)  

**Abstract**: In meta-learning and its downstream tasks, many methods rely on implicit adaptation to task variations, where multiple factors are mixed together in a single entangled representation. This makes it difficult to interpret which factors drive performance and can hinder generalization. In this work, we introduce a disentangled multi-context meta-learning framework that explicitly assigns each task factor to a distinct context vector. By decoupling these variations, our approach improves robustness through deeper task understanding and enhances generalization by enabling context vector sharing across tasks with shared factors. We evaluate our approach in two domains. First, on a sinusoidal regression task, our model outperforms baselines on out-of-distribution tasks and generalizes to unseen sine functions by sharing context vectors associated with shared amplitudes or phase shifts. Second, in a quadruped robot locomotion task, we disentangle the robot-specific properties and the characteristics of the terrain in the robot dynamics model. By transferring disentangled context vectors acquired from the dynamics model into reinforcement learning, the resulting policy achieves improved robustness under out-of-distribution conditions, surpassing the baselines that rely on a single unified context. Furthermore, by effectively sharing context, our model enables successful sim-to-real policy transfer to challenging terrains with out-of-distribution robot-specific properties, using just 20 seconds of real data from flat terrain, a result not achievable with single-task adaptation. 

**Abstract (ZH)**: 在元学习及其下游任务中，许多方法依赖于隐式的适应任务变化，其中多个因素在单一纠缠表示中混合。这使得难以解释哪些因素驱动性能提升，并且会阻碍泛化能力。在本研究中，我们提出了一种解纠缠多上下文元学习框架，明确将每个任务因素分配到一个独特的上下文向量中。通过分离这些变化，我们的方法通过对任务的更深层次理解提高稳健性，并通过允许具有共享因素的任务之间共享上下文向量来增强泛化能力。我们在两个领域评估了该方法。首先，在一个正弦回归任务上，我们的模型在分布外任务上优于基线模型，并通过共享相关共享幅度或相位平移的上下文向量实现了对未见过的正弦函数的泛化。其次，在四足机器人运动任务中，我们解纠缠了机器人的特性和地形在动力学模型中的特性。通过将动力学模型中获得的解纠缠上下文向量转移到强化学习中，所得策略在分布外条件下表现出增强的稳健性，超越了依赖单一统一上下文的基线方法。此外，通过有效共享上下文，我们的模型能够使用仅20秒的平地数据实现实验到现实的策略转移，适用于具有分布外机器人特性的挑战性地形，这是单任务适应无法实现的。 

---
# Toward a Holistic Multi-Criteria Trajectory Evaluation Framework for Autonomous Driving in Mixed Traffic Environment 

**Title (ZH)**: 面向混合交通环境自主驾驶轨迹综合多准则评价框架 

**Authors**: Nouhed Naidja, Stéphane Font, Marc Revilloud, Guillaume Sandou  

**Link**: [PDF](https://arxiv.org/pdf/2509.01291)  

**Abstract**: This paper presents a unified framework for the evaluation and optimization of autonomous vehicle trajectories, integrating formal safety, comfort, and efficiency criteria. An innovative geometric indicator, based on the analysis of safety zones using adaptive ellipses, is used to accurately quantify collision risks. Our method applies the Shoelace formula to compute the intersection area in the case of misaligned and time-varying configurations. Comfort is modeled using indicators centered on longitudinal and lateral jerk, while efficiency is assessed by overall travel time. These criteria are aggregated into a comprehensive objective function solved using a PSO based algorithm. The approach was successfully validated under real traffic conditions via experiments conducted in an urban intersection involving an autonomous vehicle interacting with a human-operated vehicle, and in simulation using data recorded from human driving in real traffic. 

**Abstract (ZH)**: 本文提出了一种综合框架，用于自动驾驶车辆轨迹的评估与优化，整合了形式化安全、舒适性和效率标准。该方法采用基于自适应椭圆分析安全区域的创新几何指标，准确量化碰撞风险。舒适性通过纵向和横向加速度指标建模，效率通过总体行驶时间评估。这些标准被汇总成一个综合目标函数，使用基于PSO的算法求解。该方法在城市交叉口的真实交通条件下，通过涉及自动驾驶车辆与人为操作车辆互动的实验，以及使用真实交通中人类驾驶数据进行的模拟实验中得到验证。 

---
# Towards Data-Driven Metrics for Social Robot Navigation Benchmarking 

**Title (ZH)**: 面向数据驱动的社会机器人导航基准评估指标研究 

**Authors**: Pilar Bachiller-Burgos, Ulysses Bernardet, Luis V. Calderita, Pranup Chhetri, Anthony Francis, Noriaki Hirose, Noé Pérez, Dhruv Shah, Phani T. Singamaneni, Xuesu Xiao, Luis J. Manso  

**Link**: [PDF](https://arxiv.org/pdf/2509.01251)  

**Abstract**: This paper presents a joint effort towards the development of a data-driven Social Robot Navigation metric to facilitate benchmarking and policy optimization. We provide our motivations for our approach and describe our proposal for storing rated social navigation trajectory datasets. Following these guidelines, we compiled a dataset with 4427 trajectories -- 182 real and 4245 simulated -- and presented it to human raters, yielding a total of 4402 rated trajectories after data quality assurance. We also trained an RNN-based baseline metric on the dataset and present quantitative and qualitative results. All data, software, and model weights are publicly available. 

**Abstract (ZH)**: 本文提出了一种基于数据驱动的社会机器人导航度量方法，旨在促进基准测试和政策优化。我们阐述了采用此种方法的原因，并描述了我们提出的用于存储评分社会导航轨迹数据集的方案。根据这些指南，我们编译了一个包含4427条轨迹的数据集——其中182条是真实的，4245条是模拟的——并将其呈现给人类评估者，经过数据质量保证后，最终获得了4402条评分轨迹。我们还在数据集上训练了一个基于RNN的基本度量方法，并展示了定量和定性的结果。所有数据、软件和模型权重均已公开。 

---
# OpenMulti: Open-Vocabulary Instance-Level Multi-Agent Distributed Implicit Mapping 

**Title (ZH)**: OpenMulti: 开放词汇实例级多Agent分布式隐式映射 

**Authors**: Jianyu Dou, Yinan Deng, Jiahui Wang, Xingsi Tang, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.01228)  

**Abstract**: Multi-agent distributed collaborative mapping provides comprehensive and efficient representations for robots. However, existing approaches lack instance-level awareness and semantic understanding of environments, limiting their effectiveness for downstream applications. To address this issue, we propose OpenMulti, an open-vocabulary instance-level multi-agent distributed implicit mapping framework. Specifically, we introduce a Cross-Agent Instance Alignment module, which constructs an Instance Collaborative Graph to ensure consistent instance understanding across agents. To alleviate the degradation of mapping accuracy due to the blind-zone optimization trap, we leverage Cross Rendering Supervision to enhance distributed learning of the scene. Experimental results show that OpenMulti outperforms related algorithms in both fine-grained geometric accuracy and zero-shot semantic accuracy. In addition, OpenMulti supports instance-level retrieval tasks, delivering semantic annotations for downstream applications. The project website of OpenMulti is publicly available at this https URL. 

**Abstract (ZH)**: 多智能体分布式协作建图提供了全面而高效的机器人表示。然而，现有方法缺乏对环境的实例级感知和语义理解，限制了其在下游应用中的效果。为了解决这一问题，我们提出了OpenMulti，一个开放词汇量的实例级多智能体分布式隐式建图框架。具体而言，我们引入了跨智能体实例对齐模块，构建实例协作图以确保各智能体之间的一致实例理解。为了解决由于盲区优化陷阱导致的建图精度下降问题，我们利用跨渲染监督来增强场景的分布式学习能力。实验结果表明，OpenMulti在细粒度几何精度和零样本语义精度上均优于相关算法。此外，OpenMulti支持实例级检索任务，为下游应用提供语义标注。OpenMulti项目的官方网站可在此 https URL 访问。 

---
# Novel bio-inspired soft actuators for upper-limb exoskeletons: design, fabrication and feasibility study 

**Title (ZH)**: 仿生启发的新型软执行器设计、制造及其在上肢外骨骼中的可行性研究 

**Authors**: Haiyun Zhang, Gabrielle Naquila, Jung Hyun Bae, Zonghuan Wu, Ashwin Hingwe, Ashish Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2509.01145)  

**Abstract**: Soft robots have been increasingly utilized as sophisticated tools in physical rehabilitation, particularly for assisting patients with neuromotor impairments. However, many soft robotics for rehabilitation applications are characterized by limitations such as slow response times, restricted range of motion, and low output force. There are also limited studies on the precise position and force control of wearable soft actuators. Furthermore, not many studies articulate how bellow-structured actuator designs quantitatively contribute to the robots' capability. This study introduces a paradigm of upper limb soft actuator design. This paradigm comprises two actuators: the Lobster-Inspired Silicone Pneumatic Robot (LISPER) for the elbow and the Scallop-Shaped Pneumatic Robot (SCASPER) for the shoulder. LISPER is characterized by higher bandwidth, increased output force/torque, and high linearity. SCASPER is characterized by high output force/torque and simplified fabrication processes. Comprehensive analytical models that describe the relationship between pressure, bending angles, and output force for both actuators were presented so the geometric configuration of the actuators can be set to modify the range of motion and output forces. The preliminary test on a dummy arm is conducted to test the capability of the actuators. 

**Abstract (ZH)**: 软体机器人在物理康复中的应用越来越广泛， Particularly for 助残神经肌肉功能障碍患者的辅助治疗。然而，许多康复应用中的软体机器人受到响应时间缓慢、活动范围受限和输出力低等限制。此外，关于可穿戴软执行器的精确位置和力控制的研究也比较有限。同时，很少有研究明确阐述Bellows结构执行器设计如何定量提升机器人的能力。本研究引入了上肢软执行器的设计范式。该范式包括两个执行器：以龙虾为灵感的硅气动机器人（Lobster-Inspired Silicone Pneumatic Robot, LISPER）用于肘部，以及扇贝形气动机器人（Scallop-Shaped Pneumatic Robot, SCASPER）用于肩部。LISPER具有更高的带宽、更大的输出力/扭矩和高线性度。SCASPER具有更大的输出力/扭矩和简化了的制造工艺。两种执行器之间的压力、弯曲角度和输出力的关系的全面分析模型被提出，以便通过调整执行器的几何配置来修改活动范围和输出力。初步测试在假肢上进行，以测试执行器的能力。 

---
# A novel parameter estimation method for pneumatic soft hand control applying logarithmic decrement for pseudo rigid body modeling 

**Title (ZH)**: 基于对数 decrement 的伪刚体建模的新型气动软手控制参数估计方法 

**Authors**: Haiyun Zhang, Kelvin HoLam Heung, Gabrielle J. Naquila, Ashwin Hingwe, Ashish D. Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2509.01113)  

**Abstract**: The rapid advancement in physical human-robot interaction (HRI) has accelerated the development of soft robot designs and controllers. Controlling soft robots, especially soft hand grasping, is challenging due to their continuous deformation, motivating the use of reduced model-based controllers for real-time dynamic performance. Most existing models, however, suffer from computational inefficiency and complex parameter identification, limiting their real-time applicability. To address this, we propose a paradigm coupling Pseudo-Rigid Body Modeling with the Logarithmic Decrement Method for parameter estimation (PRBM plus LDM). Using a soft robotic hand test bed, we validate PRBM plus LDM for predicting position and force output from pressure input and benchmark its performance. We then implement PRBM plus LDM as the basis for closed-loop position and force controllers. Compared to a simple PID controller, the PRBM plus LDM position controller achieves lower error (average maximum error across all fingers: 4.37 degrees versus 20.38 degrees). For force control, PRBM plus LDM outperforms constant pressure grasping in pinching tasks on delicate objects: potato chip 86 versus 82.5, screwdriver 74.42 versus 70, brass coin 64.75 versus 35. These results demonstrate PRBM plus LDM as a computationally efficient and accurate modeling technique for soft actuators, enabling stable and flexible grasping with precise force regulation. 

**Abstract (ZH)**: 基于伪刚体模型与对数 decrement方法的参数估计在软手抓取控制中的应用 

---
# SR-SLAM: Scene-reliability Based RGB-D SLAM in Diverse Environments 

**Title (ZH)**: SR-SLAM: 基于场景可靠性的RGB-D SLAM在多样环境中 

**Authors**: Haolan Zhang, Chenghao Li, Thanh Nguyen Canh, Lijun Wang, Nak Young Chong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01111)  

**Abstract**: Visual simultaneous localization and mapping (SLAM) plays a critical role in autonomous robotic systems, especially where accurate and reliable measurements are essential for navigation and sensing. In feature-based SLAM, the quantityand quality of extracted features significantly influence system performance. Due to the variations in feature quantity and quality across diverse environments, current approaches face two major challenges: (1) limited adaptability in dynamic feature culling and pose estimation, and (2) insufficient environmental awareness in assessment and optimization strategies. To address these issues, we propose SRR-SLAM, a scene-reliability based framework that enhances feature-based SLAM through environment-aware processing. Our method introduces a unified scene reliability assessment mechanism that incorporates multiple metrics and historical observations to guide system behavior. Based on this assessment, we develop: (i) adaptive dynamic region selection with flexible geometric constraints, (ii) depth-assisted self-adjusting clustering for efficient dynamic feature removal in high-dimensional settings, and (iii) reliability-aware pose refinement that dynamically integrates direct methods when features are insufficient. Furthermore, we propose (iv) reliability-based keyframe selection and a weighted optimization scheme to reduce computational overhead while improving estimation accuracy. Extensive experiments on public datasets and real world scenarios show that SRR-SLAM outperforms state-of-the-art dynamic SLAM methods, achieving up to 90% improvement in accuracy and robustness across diverse environments. These improvements directly contribute to enhanced measurement precision and reliability in autonomous robotic sensing systems. 

**Abstract (ZH)**: 基于场景可靠性的SLAM（视觉同步定位与建图）在自主机器人系统中的关键作用及其环境感知处理框架 

---
# Model Predictive Control for a Soft Robotic Finger with Stochastic Behavior based on Fokker-Planck Equation 

**Title (ZH)**: 基于Fokker-Planck方程的具有随机行为软机器人手指的模型预测控制 

**Authors**: Sumitaka Honji, Takahiro Wada  

**Link**: [PDF](https://arxiv.org/pdf/2509.01065)  

**Abstract**: The inherent flexibility of soft robots offers numerous advantages, such as enhanced adaptability and improved safety. However, this flexibility can also introduce challenges regarding highly uncertain and nonlinear motion. These challenges become particularly problematic when using open-loop control methods, which lack a feedback mechanism and are commonly employed in soft robot control. Though one potential solution is model-based control, typical deterministic models struggle with uncertainty as mentioned above. The idea is to use the Fokker-Planck Equation (FPE), a master equation of a stochastic process, to control not the state of soft robots but the probabilistic distribution. In this study, we propose and implement a stochastic-based control strategy, termed FPE-based Model Predictive Control (FPE-MPC), for a soft robotic finger. Two numerical simulation case studies examine the performance and characteristics of this control method, revealing its efficacy in managing the uncertainty inherent in soft robotic systems. 

**Abstract (ZH)**: 软体机器人内在的柔韧性提供了诸多优势，如增强的适应性和提高的安全性。然而，这种柔韧性也可能引入高度不确定性和非线性运动的挑战。这些挑战在使用开环控制方法时尤为突出，这类方法缺乏反馈机制，常被用于软体机器人控制中。尽管一种潜在的解决方案是基于模型的控制，但典型的确定性模型难以处理上述的不确定性。本研究提出并实现了基于随机性的控制策略，称为Fokker-Planck方程（FPE）为基础的模型预测控制（FPE-MPC），应用于软体机器人手指。通过两个数值仿真案例研究，探讨了该控制方法的性能和特性，展示了其在管理软体机器人系统固有的不确定性方面的有效性。 

---
# A Reactive Grasping Framework for Multi-DoF Grippers via Task Space Velocity Fields and Joint Space QP 

**Title (ZH)**: 基于任务空间速度场和关节空间QP的多自由度抓取框架 

**Authors**: Yonghyeon Lee, Tzu-Yuan Lin, Alexander Alexiev, Sangbae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.01044)  

**Abstract**: We present a fast and reactive grasping framework for multi-DoF grippers that combines task-space velocity fields with a joint-space Quadratic Program (QP) in a hierarchical structure. Reactive, collision-free global motion planning is particularly challenging for high-DoF systems, since simultaneous increases in state dimensionality and planning horizon trigger a combinatorial explosion of the search space, making real-time planning intractable. To address this, we plan globally in a lower-dimensional task space, such as fingertip positions, and track locally in the full joint space while enforcing all constraints. This approach is realized by constructing velocity fields in multiple task-space coordinates (or in some cases a subset of joint coordinates) and solving a weighted joint-space QP to compute joint velocities that track these fields with appropriately assigned priorities. Through simulation experiments with privileged knowledge and real-world tests using the recent pose-tracking algorithm FoundationPose, we verify that our method enables high-DoF arm-hand systems to perform real-time, collision-free reaching motions while adapting to dynamic environments and external disturbances. 

**Abstract (ZH)**: 多自由度机械手快速反应抓取框架：结合任务空间速度场与关节空间二次规划的分层结构 

---
# TARA: A Low-Cost 3D-Printed Robotic Arm for Accessible Robotics Education 

**Title (ZH)**: TARA：一种低成本的3D打印机器人手臂，用于无障碍机器人教育 

**Authors**: Thays Leach Mitre  

**Link**: [PDF](https://arxiv.org/pdf/2509.01043)  

**Abstract**: The high cost of robotic platforms limits students' ability to gain practical skills directly applicable in real-world scenarios. To address this challenge, this paper presents TARA, a low-cost, 3D-printed robotic arm designed for accessible robotics education. TARA includes an open-source repository with design files, assembly instructions, and baseline code, enabling users to build and customize the platform. The system balances affordability and functionality, offering a highly capable robotic arm for approximately 200 USD, significantly lower than industrial systems that often cost thousands of dollars. Experimental validation confirmed accurate performance in basic manipulation tasks. Rather than focusing on performance benchmarking, this work prioritizes educational reproducibility, providing a platform that students and educators can reliably replicate and extend. 

**Abstract (ZH)**: 高成本的机器人平台限制了学生直接获得应用于实际场景的实践技能的能力。为应对这一挑战，本文介绍了一种低成本的3D打印机器人臂TARA，旨在推动可访问的机器人教育。TARA包括一个开源资源库，其中包含设计文件、组装说明和基础代码，使用户能够构建和定制该平台。该系统在保持低成本的同时兼顾功能性，提供了一个大约200美元的高性能机器人臂，远低于常常需要数以千计美元的工业系统。实验验证证实了其在基本操作任务中具有准确的表现。本文优先考虑教育再现性，而非性能基准测试，提供了一个学生和教育者可以可靠地复制和扩展的平台。 

---
# A Robust Numerical Method for Solving Trigonometric Equations in Robotic Kinematics 

**Title (ZH)**: 一种求解机器人运动学中三角方程的稳健数值方法 

**Authors**: Hai-Jun Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.01010)  

**Abstract**: This paper presents a robust numerical method for solving systems of trigonometric equations commonly encountered in robotic kinematics. Our approach employs polynomial substitution techniques combined with eigenvalue decomposition to handle singular matrices and edge cases effectively. The method demonstrates superior numerical stability compared to traditional approaches and has been implemented as an open-source Python package. For non-singular matrices, we employ Weierstrass substitution to transform the system into a quartic polynomial, ensuring all analytical solutions are found. For singular matrices, we develop specialized geometric constraint methods using SVD analysis. The solver demonstrates machine precision accuracy ($< 10^{-15}$ error) with 100\% success rate on extensive test cases, making it particularly valuable for robotics applications such as inverse kinematics problems. 

**Abstract (ZH)**: 本文提出了一种稳健的数值方法，用于解决机器人运动学中常见的三角方程系统。该方法结合多项式替换技术和特征值分解来有效处理奇异矩阵和边界情况。与传统方法相比，该方法显示出更优越的数值稳定性，并已被实现为开源Python包。对于非奇异矩阵，采用魏尔斯特拉斯替换将系统转换为四次多项式，确保找到所有解析解。对于奇异矩阵，采用基于SVD分析的专门几何约束方法。求解器在大量测试案例中实现了机器精度精度（误差小于$10^{-15}$）和100%的成功率，特别适用于机器人应用，如逆运动学问题。 

---
# Enhanced Mean Field Game for Interactive Decision-Making with Varied Stylish Multi-Vehicles 

**Title (ZH)**: 增强的均场游戏及其在多元風格 Vehicles 的交互决策中的应用 

**Authors**: Liancheng Zheng, Zhen Tian, Yangfan He, Shuo Liu, Ke Gong, Huilin Chen, Zhihao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.00981)  

**Abstract**: This paper presents an MFG-based decision-making framework for autonomous driving in heterogeneous traffic. To capture diverse human behaviors, we propose a quantitative driving style representation that maps abstract traits to parameters such as speed, safety factors, and reaction time. These parameters are embedded into the MFG through a spatial influence field model. To ensure safe operation in dense traffic, we introduce a safety-critical lane-changing algorithm that leverages dynamic safety margins, time-to-collision analysis, and multi-layered constraints. Real-world NGSIM data is employed for style calibration and empirical validation. Experimental results demonstrate zero collisions across six style combinations, two 15-vehicle scenarios, and NGSIM-based trials, consistently outperforming conventional game-theoretic baselines. Overall, our approach provides a scalable, interpretable, and behavior-aware planning framework for real-world autonomous driving applications. 

**Abstract (ZH)**: 基于MFG的异构交通自主驾驶决策框架 

---
# One-Step Model Predictive Path Integral for Manipulator Motion Planning Using Configuration Space Distance Fields 

**Title (ZH)**: 基于配置空间距离场的一步模型预测路径积分用于 manipulator 运动规划 

**Authors**: Yulin Li, Tetsuro Miyazaki, Kenji Kawashima  

**Link**: [PDF](https://arxiv.org/pdf/2509.00836)  

**Abstract**: Motion planning for robotic manipulators is a fundamental problem in robotics. Classical optimization-based methods typically rely on the gradients of signed distance fields (SDFs) to impose collision-avoidance constraints. However, these methods are susceptible to local minima and may fail when the SDF gradients vanish. Recently, Configuration Space Distance Fields (CDFs) have been introduced, which directly model distances in the robot's configuration space. Unlike workspace SDFs, CDFs are differentiable almost everywhere and thus provide reliable gradient information. On the other hand, gradient-free approaches such as Model Predictive Path Integral (MPPI) control leverage long-horizon rollouts to achieve collision avoidance. While effective, these methods are computationally expensive due to the large number of trajectory samples, repeated collision checks, and the difficulty of designing cost functions with heterogeneous physical units. In this paper, we propose a framework that integrates CDFs with MPPI to enable direct navigation in the robot's configuration space. Leveraging CDF gradients, we unify the MPPI cost in joint-space and reduce the horizon to one step, substantially cutting computation while preserving collision avoidance in practice. We demonstrate that our approach achieves nearly 100% success rates in 2D environments and consistently high success rates in challenging 7-DOF Franka manipulator simulations with complex obstacles. Furthermore, our method attains control frequencies exceeding 750 Hz, substantially outperforming both optimization-based and standard MPPI baselines. These results highlight the effectiveness and efficiency of the proposed CDF-MPPI framework for high-dimensional motion planning. 

**Abstract (ZH)**: 基于配置空间距离场的模型预测路径积分运动规划框架 

---
# An Effective Trajectory Planning and an Optimized Path Planning for a 6-Degree-of-Freedom Robot Manipulator 

**Title (ZH)**: 一种有效的六自由度机器人 manipulator 轨迹规划及优化路径规划 

**Authors**: Takumu Okazaki, Akira Terui, Masahiko Mikawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.00828)  

**Abstract**: An effective method for optimizing path planning for a specific model of a 6-degree-of-freedom (6-DOF) robot manipulator is presented as part of the motion planning of the manipulator using computer algebra. We assume that we are given a path in the form of a set of line segments that the end-effector should follow. We also assume that we have a method to solve the inverse kinematic problem of the manipulator at each via-point of the trajectory. The proposed method consists of three steps. First, we calculate the feasible region of the manipulator under a specific configuration of the end-effector. Next, we aim to find a trajectory on the line segments and a sequence of joint configurations the manipulator should follow to move the end-effector along the specified trajectory. Finally, we find the optimal combination of solutions to the inverse kinematic problem at each via-point along the trajectory by reducing the problem to a shortest-path problem of the graph and applying Dijkstra's algorithm. We show the effectiveness of the proposed method by experiments. 

**Abstract (ZH)**: 一种基于计算机代数的特定6自由度（6-DOF）机器人 manipulator 路径规划优化方法 

---
# Inverse Kinematics for a 6-Degree-of-Freedom Robot Manipulator Using Comprehensive Gröbner Systems 

**Title (ZH)**: 使用综合格消系统求解六自由度机器人 manipulator 的逆向动力学 

**Authors**: Takumu Okazaki, Akira Terui, Masahiko Mikawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.00823)  

**Abstract**: We propose an effective method for solving the inverse kinematic problem of a specific model of 6-degree-of-freedom (6-DOF) robot manipulator using computer algebra. It is known that when the rotation axes of three consecutive rotational joints of a manipulator intersect at a single point, the inverse kinematics problem can be divided into determining position and orientation. We extend this method to more general manipulators in which the rotational axes of two consecutive joints intersect. This extension broadens the class of 6-DOF manipulators for which the inverse kinematics problem can be solved, and is expected to enable more efficient solutions. The inverse kinematic problem is solved using the Comprehensive Gröbner System (CGS) with joint parameters of the robot appearing as parameters in the coefficients to prevent repetitive calculations of the Gröbner bases. The effectiveness of the proposed method is shown by experiments. 

**Abstract (ZH)**: 我们提出了一种有效的方法，使用计算机代数解决特定6自由度（6-DOF）机器人 manipulator 的逆运动学问题。当 manipulator 连续三个旋转关节的旋转轴在单一点相交时，逆运动学问题可以分解为确定位置和姿态。我们将此方法扩展到连续两个关节的旋转轴相交的更通用的 manipulator。这种扩展扩大了可以使用逆运动学问题解决方案的6-DOF manipulator 类别，并有望实现更高效的解决方案。通过使用综合格罗布ner系统（CGS）并使机器人关节参数出现在系数中作为参数，以防止格罗布ner基的重复计算，逆运动学问题得到解决。所提出方法的有效性通过实验得到了证实。 

---
# DyPho-SLAM : Real-time Photorealistic SLAM in Dynamic Environments 

**Title (ZH)**: DyPho-SLAM : 实时高保真动态环境SLAM 

**Authors**: Yi Liu, Keyu Fan, Bin Lan, Houde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00741)  

**Abstract**: Visual SLAM algorithms have been enhanced through the exploration of Gaussian Splatting representations, particularly in generating high-fidelity dense maps. While existing methods perform reliably in static environments, they often encounter camera tracking drift and fuzzy mapping when dealing with the disturbances caused by moving objects. This paper presents DyPho-SLAM, a real-time, resource-efficient visual SLAM system designed to address the challenges of localization and photorealistic mapping in environments with dynamic objects. Specifically, the proposed system integrates prior image information to generate refined masks, effectively minimizing noise from mask misjudgment. Additionally, to enhance constraints for optimization after removing dynamic obstacles, we devise adaptive feature extraction strategies significantly improving the system's resilience. Experiments conducted on publicly dynamic RGB-D datasets demonstrate that the proposed system achieves state-of-the-art performance in camera pose estimation and dense map reconstruction, while operating in real-time in dynamic scenes. 

**Abstract (ZH)**: 视觉SLAM算法通过探索高保真密集地图生成的Gaussian Splatting表示得到了增强，特别是DyPho-SLAM：一种针对动态环境的实时高效视觉SLAM系统 

---
# CARIS: A Context-Adaptable Robot Interface System for Personalized and Scalable Human-Robot Interaction 

**Title (ZH)**: CARIS: 一种适用于个性化和可扩展人机交互的上下文自适应机器人接口系统 

**Authors**: Felipe Arias-Russi, Yuanchen Bai, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2509.00660)  

**Abstract**: The human-robot interaction (HRI) field has traditionally used Wizard-of-Oz (WoZ) controlled robots to explore navigation, conversational dynamics, human-in-the-loop interactions, and more to explore appropriate robot behaviors in everyday settings. However, existing WoZ tools are often limited to one context, making them less adaptable across different settings, users, and robotic platforms. To mitigate these issues, we introduce a Context-Adaptable Robot Interface System (CARIS) that combines advanced robotic capabilities such teleoperation, human perception, human-robot dialogue, and multimodal data recording. Through pilot studies, we demonstrate the potential of CARIS to WoZ control a robot in two contexts: 1) mental health companion and as a 2) tour guide. Furthermore, we identified areas of improvement for CARIS, including smoother integration between movement and communication, clearer functionality separation, recommended prompts, and one-click communication options to enhance the usability wizard control of CARIS. This project offers a publicly available, context-adaptable tool for the HRI community, enabling researchers to streamline data-driven approaches to intelligent robot behavior. 

**Abstract (ZH)**: 人类与机器人交互领域中的适场景可调机器人接口系统（CARIS）：一种结合了远程操作、人类感知、人机对话和多模态数据记录的工具 

---
# A Risk-aware Spatial-temporal Trajectory Planning Framework for Autonomous Vehicles Using QP-MPC and Dynamic Hazard Fields 

**Title (ZH)**: 基于QP-MPC和动态危险场的自驾车风险意识时空轨迹规划框架 

**Authors**: Zhen Tian, Zhihao Lin, Dezong Zhao, Christos Anagnostopoulos, Qiyuan Wang, Wenjing Zhao, Xiaodan Wang, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.00643)  

**Abstract**: Trajectory planning is a critical component in ensuring the safety, stability, and efficiency of autonomous vehicles. While existing trajectory planning methods have achieved progress, they often suffer from high computational costs, unstable performance in dynamic environments, and limited validation across diverse scenarios. To overcome these challenges, we propose an enhanced QP-MPC-based framework that incorporates three key innovations: (i) a novel cost function designed with a dynamic hazard field, which explicitly balances safety, efficiency, and comfort; (ii) seamless integration of this cost function into the QP-MPC formulation, enabling direct optimization of desired driving behaviors; and (iii) extensive validation of the proposed framework across complex tasks. The spatial safe planning is guided by a dynamic hazard field (DHF) for risk assessment, while temporal safe planning is based on a space-time graph. Besides, the quintic polynomial sampling and sub-reward of comforts are used to ensure comforts during lane-changing. The sub-reward of efficiency is used to maintain driving efficiency. Finally, the proposed DHF-enhanced objective function integrates multiple objectives, providing a proper optimization tasks for QP-MPC. Extensive simulations demonstrate that the proposed framework outperforms benchmark optimization methods in terms of efficiency, stability, and comfort across a variety of scenarios likes lane-changing, overtaking, and crossing intersections. 

**Abstract (ZH)**: 轨迹规划是确保自主车辆安全、稳定和高效的关键组件。为克服现有方法的高计算成本、动态环境下的不稳定性能以及跨多样场景的有限验证等问题，我们提出了一种增强的QP-MPC框架，结合了三项创新：（i）一种基于动态危险场的新成本函数，显式平衡安全、效率和舒适性；（ii）将此成本函数无缝集成到QP-MPC公式中，实现对期望驾驶行为的直接优化；（iii）在复杂任务中全面验证所提出的框架。通过动态危险场（DHF）进行空间安全性规划以进行风险评估，基于时空图进行时间安全性规划。此外，使用五次多项式采样和舒适度亚奖励确保变道过程中的舒适性，使用效率亚奖励保持驾驶效率。最后，提出的DHF增强目标函数结合了多个目标，为QP-MPC提供适当的优化任务。广泛仿真实验表明，所提出的框架在多种场景（如变道、超车和穿越交叉口）下在效率、稳定性和舒适性方面优于基准优化方法。 

---
# Vehicle-in-Virtual-Environment (VVE) Method for Developing and Evaluating VRU Safety of Connected and Autonomous Driving with Focus on Bicyclist Safety 

**Title (ZH)**: 基于虚拟环境的车辆（VVE方法）在连接和自主驾驶中发展和评估VRU安全性，重点关注自行车骑行者安全 

**Authors**: Haochong Chen, Xincheng Cao, Bilin Aksun-Guvenc, Levent Guvenc  

**Link**: [PDF](https://arxiv.org/pdf/2509.00624)  

**Abstract**: Extensive research has already been conducted in the autonomous driving field to help vehicles navigate safely and efficiently. At the same time, plenty of current research on vulnerable road user (VRU) safety is performed which largely concentrates on perception, localization, or trajectory prediction of VRUs. However, existing research still exhibits several gaps, including the lack of a unified planning and collision avoidance system for autonomous vehicles, limited investigation into delay tolerant control strategies, and the absence of an efficient and standardized testing methodology. Ensuring VRU safety remains one of the most pressing challenges in autonomous driving, particularly in dynamic and unpredictable environments. In this two year project, we focused on applying the Vehicle in Virtual Environment (VVE) method to develop, evaluate, and demonstrate safety functions for Vulnerable Road Users (VRUs) using automated steering and braking of ADS. In this current second year project report, our primary focus was on enhancing the previous year results while also considering bicyclist safety. 

**Abstract (ZH)**: 自动驾驶领域已开展了大量研究以帮助车辆安全高效地导航。同时，大量关于脆弱道路使用者（VRU）安全的研究也集中在感知、定位或轨迹预测等方面。然而，现有研究仍存在一些差距，包括缺乏统一的规划和碰撞 avoidance 系统、对容错控制策略的调查有限，以及缺乏有效的标准化测试方法。确保 VRU 安全仍是自动驾驶中最 pressing 的挑战之一，特别是在动态和不可预测的环境中。在本两年期项目中，我们专注于采用车辆在虚拟环境（VVE）方法开发、评估和展示使用自动转向和制动的自动驾驶系统（ADS）的安全功能，特别是在骑自行车者安全方面的应用。在当前的第二年项目报告中，我们主要致力于增强去年的结果，同时考虑骑自行车者安全。 

---
# Safe and Efficient Lane-Changing for Autonomous Vehicles: An Improved Double Quintic Polynomial Approach with Time-to-Collision Evaluation 

**Title (ZH)**: 基于时间碰撞评估的改进双 quintic 多项式方法：自主车辆安全高效变道研究 

**Authors**: Rui Bai, Rui Xu, Teng Rui, Jiale Liu, Qi Wei Oung, Hoi Leong Lee, Zhen Tian, Fujiang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00582)  

**Abstract**: Autonomous driving technology has made significant advancements in recent years, yet challenges remain in ensuring safe and comfortable interactions with human-driven vehicles (HDVs), particularly during lane-changing maneuvers. This paper proposes an improved double quintic polynomial approach for safe and efficient lane-changing in mixed traffic environments. The proposed method integrates a time-to-collision (TTC) based evaluation mechanism directly into the trajectory optimization process, ensuring that the ego vehicle proactively maintains a safe gap from surrounding HDVs throughout the maneuver. The framework comprises state estimation for both the autonomous vehicle (AV) and HDVs, trajectory generation using double quintic polynomials, real-time TTC computation, and adaptive trajectory evaluation. To the best of our knowledge, this is the first work to embed an analytic TTC penalty directly into the closed-form double-quintic polynomial solver, enabling real-time safety-aware trajectory generation without post-hoc validation. Extensive simulations conducted under diverse traffic scenarios demonstrate the safety, efficiency, and comfort of the proposed approach compared to conventional methods such as quintic polynomials, Bezier curves, and B-splines. The results highlight that the improved method not only avoids collisions but also ensures smooth transitions and adaptive decision-making in dynamic environments. This work bridges the gap between model-based and adaptive trajectory planning approaches, offering a stable solution for real-world autonomous driving applications. 

**Abstract (ZH)**: 自主驾驶技术在近年来取得了显著进展，但仍面临与人工驾驶车辆（HDVs）在变道时确保安全舒适交互的挑战。本文提出了一种改进的双五次多项式方法，以实现混合交通环境中安全和高效的变道。该方法将基于碰撞时间（TTC）的评估机制直接融入轨迹优化过程，确保自主车辆在整个变道过程中主动与周围HDVs保持安全距离。该框架包括自主车辆（AV）和HDVs的状态估计、使用双五次多项式生成轨迹、实时计算TTC以及适应性轨迹评估。据我们所知，这是首次将解析TTC惩罚直接嵌入闭式双五次多项式求解器中，从而实现无需事后验证的实时安全意识轨迹生成。在多种交通场景下进行的广泛仿真表明，与传统的五次多项式、贝兹曲线和B样条等方法相比，所提出的方法在安全性、效率和舒适性方面具有显著优势。这些结果表明，改进的方法不仅避免了碰撞，还确保了在动态环境中平滑过渡和适应性决策。本文填补了基于模型和自适应轨迹规划方法之间的空白，提供了一种适用于实际自主驾驶应用的稳定解决方案。 

---
# Galaxea Open-World Dataset and G0 Dual-System VLA Model 

**Title (ZH)**: Galaxea 开放世界数据集和 G0 双系统超分辨率模型 

**Authors**: Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.00576)  

**Abstract**: We present Galaxea Open-World Dataset, a large-scale, diverse collection of robot behaviors recorded in authentic human living and working environments. All demonstrations are gathered using a consistent robotic embodiment, paired with precise subtask-level language annotations to facilitate both training and evaluation. Building on this dataset, we introduce G0, a dual-system framework that couples a Vision-Language Model (VLM) for multimodal planning with a Vision-Language-Action (VLA) model for fine-grained execution. G0 is trained using a three-stage curriculum: cross-embodiment pre-training, single-embodiment pre-training, and task-specific post-training. A comprehensive benchmark spanning tabletop manipulation, few-shot learning, and long-horizon mobile manipulation, demonstrates the effectiveness of our approach. In particular, we find that the single-embodiment pre-training stage, together with the Galaxea Open-World Dataset, plays a critical role in achieving strong performance. 

**Abstract (ZH)**: Galaxea 开放世界数据集及其在 G0 双系统框架中的应用 

---
# Learning Dolly-In Filming From Demonstration Using a Ground-Based Robot 

**Title (ZH)**: 基于地面机器人从示范中学习多利拍摄方法 

**Authors**: Philip Lorimer, Alan Hunter, Wenbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00574)  

**Abstract**: Cinematic camera control demands a balance of precision and artistry - qualities that are difficult to encode through handcrafted reward functions. While reinforcement learning (RL) has been applied to robotic filmmaking, its reliance on bespoke rewards and extensive tuning limits creative usability. We propose a Learning from Demonstration (LfD) approach using Generative Adversarial Imitation Learning (GAIL) to automate dolly-in shots with a free-roaming, ground-based filming robot. Expert trajectories are collected via joystick teleoperation in simulation, capturing smooth, expressive motion without explicit objective design.
Trained exclusively on these demonstrations, our GAIL policy outperforms a PPO baseline in simulation, achieving higher rewards, faster convergence, and lower variance. Crucially, it transfers directly to a real-world robot without fine-tuning, achieving more consistent framing and subject alignment than a prior TD3-based method. These results show that LfD offers a robust, reward-free alternative to RL in cinematic domains, enabling real-time deployment with minimal technical effort. Our pipeline brings intuitive, stylized camera control within reach of creative professionals, bridging the gap between artistic intent and robotic autonomy. 

**Abstract (ZH)**: 电影摄像控制需要精确与艺术的平衡——一种难以通过手工制作的奖励函数编码的品质。虽然强化学习（RL）已被应用于机器人电影制作，但其依赖于定制的奖励函数和大量的调优限制了其创意的实用性。我们提出了一种使用生成对抗模仿学习（GAIL）的示例学习（LfD）方法，以自动实现自由移动地面拍摄机器人的推进入镜头。专家轨迹通过仿真中的手柄遥操作采集，捕捉到平滑、表现力强的运动，而无需明确的目标设计。
仅通过这些示例训练，我们的GAIL策略在仿真中优于PPO基线，获得更高的奖励、更快的收敛性和更低的方差。 crucial 地，它无需微调即可直接转移到实际机器人上，实现比先前基于TD3的方法更一致的构图和主题对齐。这些结果表明，示例学习提供了在电影领域中RL的一种稳健的、无需奖励的替代方案，能够实现最小技术努力下的实时部署。我们的流程将直观的、风格化的摄像控制带给了创意专业人士，弥合了艺术意图与机器人自主之间的差距。 

---
# Gray-Box Computed Torque Control for Differential-Drive Mobile Robot Tracking 

**Title (ZH)**: 灰盒差动驱动移动机器人轨迹跟踪计算扭矩控制 

**Authors**: Arman Javan Sekhavat Pishkhani  

**Link**: [PDF](https://arxiv.org/pdf/2509.00571)  

**Abstract**: This study presents a learning-based nonlinear algorithm for tracking control of differential-drive mobile robots. The Computed Torque Method (CTM) suffers from inaccurate knowledge of system parameters, while Deep Reinforcement Learning (DRL) algorithms are known for sample inefficiency and weak stability guarantees. The proposed method replaces the black-box policy network of a DRL agent with a gray-box Computed Torque Controller (CTC) to improve sample efficiency and ensure closed-loop stability. This approach enables finding an optimal set of controller parameters for an arbitrary reward function using only a few short learning episodes. The Twin-Delayed Deep Deterministic Policy Gradient (TD3) algorithm is used for this purpose. Additionally, some controller parameters are constrained to lie within known value ranges, ensuring the RL agent learns physically plausible values. A technique is also applied to enforce a critically damped closed-loop time response. The controller's performance is evaluated on a differential-drive mobile robot simulated in the MuJoCo physics engine and compared against the raw CTC and a conventional kinematic controller. 

**Abstract (ZH)**: 基于学习的非线性算法在差速驱动移动机器人跟踪控制中的研究 

---
# ConceptBot: Enhancing Robot's Autonomy through Task Decomposition with Large Language Models and Knowledge Graph 

**Title (ZH)**: ConceptBot: 通过任务分解增强机器人自主性的大语言模型与知识图谱 

**Authors**: Alessandro Leanza, Angelo Moroncelli, Giuseppe Vizzari, Francesco Braghin, Loris Roveda, Blerina Spahiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00570)  

**Abstract**: ConceptBot is a modular robotic planning framework that combines Large Language Models and Knowledge Graphs to generate feasible and risk-aware plans despite ambiguities in natural language instructions and correctly analyzing the objects present in the environment - challenges that typically arise from a lack of commonsense reasoning. To do that, ConceptBot integrates (i) an Object Property Extraction (OPE) module that enriches scene understanding with semantic concepts from ConceptNet, (ii) a User Request Processing (URP) module that disambiguates and structures instructions, and (iii) a Planner that generates context-aware, feasible pick-and-place policies. In comparative evaluations against Google SayCan, ConceptBot achieved 100% success on explicit tasks, maintained 87% accuracy on implicit tasks (versus 31% for SayCan), reached 76% on risk-aware tasks (versus 15%), and outperformed SayCan in application-specific scenarios, including material classification (70% vs. 20%) and toxicity detection (86% vs. 36%). On SafeAgentBench, ConceptBot achieved an overall score of 80% (versus 46% for the next-best baseline). These results, validated in both simulation and laboratory experiments, demonstrate ConceptBot's ability to generalize without domain-specific training and to significantly improve the reliability of robotic policies in unstructured environments. Website: this https URL 

**Abstract (ZH)**: ConceptBot是一个模块化的机器人规划框架，结合了大规模语言模型和知识图谱，以生成可行性高且风险意识强的计划，即使在自然语言指令存在歧义的情况下也能进行正确的环境物体分析——这通常源于常识推理的不足。该框架通过集成（i）对象属性提取（OPE）模块，（ii）用户请求处理（URP）模块，以及（iii）规划器来实现这一目标。在与Google SayCan的比较评估中，ConceptBot在显式任务中实现了100%的成功率，在隐式任务中的准确率为87%（而SayCan为31%），在风险意识任务中的准确率为76%（而SayCan为15%），并且在特定应用场景中优于SayCan，包括材料分类（70%对20%）和毒性检测（86%对36%）。在SafeAgentBench上，ConceptBot的整体得分为80%，而下一个最佳基线得分为46%。这些结果在仿真和实验室实验中的验证证明了ConceptBot的能力，即无需领域特定训练即可泛化，并显著提高了机器人策略在非结构化环境中的可靠性。网站：this https URL。 

---
# Reinforcement Learning of Dolly-In Filming Using a Ground-Based Robot 

**Title (ZH)**: 基于地面机器人学习黛西影视拍摄方法 

**Authors**: Philip Lorimer, Jack Saunders, Alan Hunter, Wenbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00564)  

**Abstract**: Free-roaming dollies enhance filmmaking with dynamic movement, but challenges in automated camera control remain unresolved. Our study advances this field by applying Reinforcement Learning (RL) to automate dolly-in shots using free-roaming ground-based filming robots, overcoming traditional control hurdles. We demonstrate the effectiveness of combined control for precise film tasks by comparing it to independent control strategies. Our robust RL pipeline surpasses traditional Proportional-Derivative controller performance in simulation and proves its efficacy in real-world tests on a modified ROSBot 2.0 platform equipped with a camera turret. This validates our approach's practicality and sets the stage for further research in complex filming scenarios, contributing significantly to the fusion of technology with cinematic creativity. This work presents a leap forward in the field and opens new avenues for research and development, effectively bridging the gap between technological advancement and creative filmmaking. 

**Abstract (ZH)**: 自由漫游地景机器人增强电影制作动态运动，但自动化摄像控制仍面临挑战。通过应用强化学习（RL）自动化自由漫游地面拍摄机器人进行地景推拉镜头，本研究克服了传统控制难题。我们通过与独立控制策略的比较，展示了结合控制在精确电影任务中的有效性。我们的稳健RL管道在仿真中超过传统比例-微分控制器性能，并在改装的ROSBot 2.0平台上进行实际测试，验证了其有效性，为复杂拍摄场景的研究奠定了基础，显著促进了技术与 Cinematic 创意的融合。此项工作在该领域取得了突破，并为研究与发展开辟了新途径，有效弥合了技术进步与创意 filmmaking 之间的差距。 

---
# Needle Biopsy And Fiber-Optic Compatible Robotic Insertion Platform 

**Title (ZH)**: 针吸活检及光纤兼容的机器人插入平台 

**Authors**: Fanxin Wang, Yikun Cheng, Chuyuan Tao, Rohit Bhargava, Thenkurussi Kesavadas  

**Link**: [PDF](https://arxiv.org/pdf/2509.00530)  

**Abstract**: Tissue biopsy is the gold standard for diagnosing many diseases, involving the extraction of diseased tissue for histopathology analysis by expert pathologists. However, this procedure has two main limitations: 1) Manual sampling through tissue biopsy is prone to inaccuracies; 2) The extraction process is followed by a time-consuming pathology test. To address these limitations, we present a compact, accurate, and maneuverable robotic insertion platform to overcome the limitations in traditional histopathology. Our platform is capable of steering a variety of tools with different sizes, including needle for tissue extraction and optical fibers for vibrational spectroscopy applications. This system facilitates the guidance of end-effector to the tissue and assists surgeons in navigating to the biopsy target area for multi-modal diagnosis. In this paper, we outline the general concept of our device, followed by a detailed description of its mechanical design and control scheme. We conclude with the validation of the system through a series of tests, including positioning accuracy, admittance performance, and tool insertion efficacy. 

**Abstract (ZH)**: 组织活检是诊断多种疾病的标准方法，涉及通过获取病变组织供专家病理学家进行组织病理学分析。然而，这一过程有两个主要限制：1）手动获取组织活检容易出现误差；2）获取过程后需进行耗时的病理测试。为解决这些限制，我们提出了一种紧凑、准确且操作灵活的机器人插入平台，以克服传统组织病理学的限制。该平台能够引导不同尺寸工具，包括用于组织获取的针和用于振动光谱学应用的光学纤维。此系统便于端执行器引导至组织并帮助外科医生导航至活检目标区域进行多模态诊断。在本文中，我们概述了该设备的一般概念，随后详细描述了其机械设计和控制方案，并通过一系列测试验证了系统的性能，包括定位精度、顺应性能和工具插入效果。 

---
# NeuralSVCD for Efficient Swept Volume Collision Detection 

**Title (ZH)**: 基于神经网络的高效扫掠体积碰撞检测 NeuralSVCD 

**Authors**: Dongwon Son, Hojin Jung, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.00499)  

**Abstract**: Robot manipulation in unstructured environments requires efficient and reliable Swept Volume Collision Detection (SVCD) for safe motion planning. Traditional discrete methods potentially miss collisions between these points, whereas SVCD continuously checks for collisions along the entire trajectory. Existing SVCD methods typically face a trade-off between efficiency and accuracy, limiting practical use. In this paper, we introduce NeuralSVCD, a novel neural encoder-decoder architecture tailored to overcome this trade-off. Our approach leverages shape locality and temporal locality through distributed geometric representations and temporal optimization. This enhances computational efficiency without sacrificing accuracy. Comprehensive experiments show that NeuralSVCD consistently outperforms existing state-of-the-art SVCD methods in terms of both collision detection accuracy and computational efficiency, demonstrating its robust applicability across diverse robotic manipulation scenarios. Code and videos are available at this https URL. 

**Abstract (ZH)**: 无结构环境中机器人操作需要高效的可靠 Swept Volume Collision Detection (SVCD) 以实现安全运动规划。 

---
# FLUID: A Fine-Grained Lightweight Urban Signalized-Intersection Dataset of Dense Conflict Trajectories 

**Title (ZH)**: FLUID：稠密冲突轨迹的细粒度轻量级城市信号交叉口数据集 

**Authors**: Yiyang Chen, Zhigang Wu, Guohong Zheng, Xuesong Wu, Liwen Xu, Haoyuan Tang, Zhaocheng He, Haipeng Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00497)  

**Abstract**: The trajectory data of traffic participants (TPs) is a fundamental resource for evaluating traffic conditions and optimizing policies, especially at urban intersections. Although data acquisition using drones is efficient, existing datasets still have limitations in scene representativeness, information richness, and data fidelity. This study introduces FLUID, comprising a fine-grained trajectory dataset that captures dense conflicts at typical urban signalized intersections, and a lightweight, full-pipeline framework for drone-based trajectory processing. FLUID covers three distinct intersection types, with approximately 5 hours of recording time and featuring over 20,000 TPs across 8 categories. Notably, the dataset averages two vehicle conflicts per minute, involving roughly 25% of all motor vehicles. FLUID provides comprehensive data, including trajectories, traffic signals, maps, and raw videos. Comparison with the DataFromSky platform and ground-truth measurements validates its high spatio-temporal accuracy. Through a detailed classification of motor vehicle conflicts and violations, FLUID reveals a diversity of interactive behaviors, demonstrating its value for human preference mining, traffic behavior modeling, and autonomous driving research. 

**Abstract (ZH)**: 交通参与者 trajectories 数据 (TPs) 是评估交通状况和优化政策的基础资源，尤其是在城市交叉口。尽管无人机数据采集效率高，现有数据集在场景代表性、信息丰富性和数据保真度方面仍存在局限性。本研究介绍了 FLUID，包含了一个细粒度的轨迹数据集，该数据集捕捉到了典型城市信号交叉口的密集冲突，并且提供了一个轻量级的端到端框架用于无人机轨迹处理。FLUID 包括三种不同类型的交叉口，记录时长约 5 小时，涵盖了超过 20,000 个交通参与者，分为 8 个类别。值得注意的是，每分钟平均有两次车辆冲突，涉及大约 25% 的所有机动车。FLUID 提供了全面的数据，包括轨迹、交通信号、地图和原始视频。与 DataFromSky 平台和实地测量的对比验证了其高时空准确性。通过详细分类车辆冲突和违规行为，FLUID 揭示了多样化的互动行为，展示了其在人类偏好挖掘、交通行为建模和自动驾驶研究中的价值。 

---
# Extended Diffeomorphism for Real-Time Motion Replication in Workspaces with Different Spatial Arrangements 

**Title (ZH)**: 扩展 diffeomorphism 在不同空间排列工作空间中的实时运动复制 

**Authors**: Masaki Saito, Shunki Itadera, Toshiyuki Murakami  

**Link**: [PDF](https://arxiv.org/pdf/2509.00491)  

**Abstract**: This paper presents two types of extended diffeomorphism designs to compensate for spatial placement differences between robot workspaces. Teleoperation of multiple robots is attracting attention to expand the utilization of the robot embodiment. Real-time reproduction of robot motion would facilitate the efficient execution of similar tasks by multiple robots. A challenge in the motion reproduction is compensating for the spatial arrangement errors of target keypoints in robot workspaces. This paper proposes a methodology for smooth mappings that transform primary robot poses into follower robot poses based on the predefined key points in each workspace. Through a picking task experiment using a dual-arm UR5 robot, this study demonstrates that the proposed mapping generation method can balance lower mapping errors for precise operation and lower mapping gradients for smooth replicated movement. 

**Abstract (ZH)**: 本文介绍了两种扩展 diffeomorphism 设计，以弥补机器人工作空间中空间位置差异。多机器人远程操作正受到关注，以扩展机器人本体的利用。实时再现机器人运动有助于多个机器人高效执行类似任务。运动再现中的挑战之一是补偿目标关键点在机器人工作空间中的空间布局误差。本文提出了一种基于每个工作空间中预定义的关键点将主机器人姿态映射到从动机器人姿态的方法学。通过使用双臂 UR5 机器人进行拾取任务实验，本研究证明了所提出的映射生成方法可以在精确操作中保持较低的映射误差和平滑复制运动中较低的映射梯度之间的平衡。 

---
# Embodied Spatial Intelligence: from Implicit Scene Modeling to Spatial Reasoning 

**Title (ZH)**: embodied空间智能：从隐式场景建模到空间推理 

**Authors**: Jiading Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00465)  

**Abstract**: This thesis introduces "Embodied Spatial Intelligence" to address the challenge of creating robots that can perceive and act in the real world based on natural language instructions. To bridge the gap between Large Language Models (LLMs) and physical embodiment, we present contributions on two fronts: scene representation and spatial reasoning. For perception, we develop robust, scalable, and accurate scene representations using implicit neural models, with contributions in self-supervised camera calibration, high-fidelity depth field generation, and large-scale reconstruction. For spatial reasoning, we enhance the spatial capabilities of LLMs by introducing a novel navigation benchmark, a method for grounding language in 3D, and a state-feedback mechanism to improve long-horizon decision-making. This work lays a foundation for robots that can robustly perceive their surroundings and intelligently act upon complex, language-based commands. 

**Abstract (ZH)**: 本论文引入“具身空间智能”以应对基于自然语言指令在真实世界中感知与行动的挑战。为弥合大型语言模型与物理具身之间的gap，我们在场景表示和空间推理两个方面提出了贡献：在感知方面，我们使用隐式神经表示开发了稳健、可扩展且准确的场景表示方法，并对自监督相机标定、高保真深度场生成和大规模重建等方面进行了贡献；在空间推理方面，我们通过引入新颖的导航基准、语言在三维空间中的定位方法以及改进长期决策的状态反馈机制来增强大型语言模型的空间能力。本研究为能稳健感知环境并在复杂语言指令下智能行动的机器人奠定了基础。 

---
# Generative Visual Foresight Meets Task-Agnostic Pose Estimation in Robotic Table-Top Manipulation 

**Title (ZH)**: 生成式视觉预见与任务无关的Pose估计在机器人桌面Manipulation中的融合 

**Authors**: Chuye Zhang, Xiaoxiong Zhang, Wei Pan, Linfang Zheng, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00361)  

**Abstract**: Robotic manipulation in unstructured environments requires systems that can generalize across diverse tasks while maintaining robust and reliable performance. We introduce {GVF-TAPE}, a closed-loop framework that combines generative visual foresight with task-agnostic pose estimation to enable scalable robotic manipulation. GVF-TAPE employs a generative video model to predict future RGB-D frames from a single side-view RGB image and a task description, offering visual plans that guide robot actions. A decoupled pose estimation model then extracts end-effector poses from the predicted frames, translating them into executable commands via low-level controllers. By iteratively integrating video foresight and pose estimation in a closed loop, GVF-TAPE achieves real-time, adaptive manipulation across a broad range of tasks. Extensive experiments in both simulation and real-world settings demonstrate that our approach reduces reliance on task-specific action data and generalizes effectively, providing a practical and scalable solution for intelligent robotic systems. 

**Abstract (ZH)**: 在未结构化环境中进行机器人操作需要能够泛化到多种任务并保持鲁棒性和可靠性能的系统。我们引入了GVF-TAPE，这是一种闭环框架，结合了生成视觉预见性和任务无关的手位估计，以实现可扩展的机器人操作。GVF-TAPE采用生成视频模型，从单张侧视RGB图像和任务描述中预测未来RGB-D帧，提供视觉计划以指导机器人动作。然后，解耦的手位估计模型从预测的帧中提取末端执行器手位，并通过低级控制器将其转换为可执行命令。通过在闭环中迭代集成视觉预见性和手位估计，GVF-TAPE实现了对多种任务的实时、自适应操作。在仿真和实际环境中的 extensive 实验表明，我们的方法减少了对特定任务动作数据的依赖并有效地泛化，为智能机器人系统提供了实用和可扩展的解决方案。 

---
# Autonomous Aggregate Sorting in Construction and Mining via Computer Vision-Aided Robotic Arm Systems 

**Title (ZH)**: 基于计算机视觉辅助 robotic arm 系统的建筑与矿业自动分选 

**Authors**: Md. Taherul Islam Shawon, Yuan Li, Yincai Cai, Junjie Niu, Ting Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00339)  

**Abstract**: Traditional aggregate sorting methods, whether manual or mechanical, often suffer from low precision, limited flexibility, and poor adaptability to diverse material properties such as size, shape, and lithology. To address these limitations, this study presents a computer vision-aided robotic arm system designed for autonomous aggregate sorting in construction and mining applications. The system integrates a six-degree-of-freedom robotic arm, a binocular stereo camera for 3D perception, and a ROS-based control framework. Core techniques include an attention-augmented YOLOv8 model for aggregate detection, stereo matching for 3D localization, Denavit-Hartenberg kinematic modeling for arm motion control, minimum enclosing rectangle analysis for size estimation, and hand-eye calibration for precise coordinate alignment. Experimental validation with four aggregate types achieved an average grasping and sorting success rate of 97.5%, with comparable classification accuracy. Remaining challenges include the reliable handling of small aggregates and texture-based misclassification. Overall, the proposed system demonstrates significant potential to enhance productivity, reduce operational costs, and improve safety in aggregate handling, while providing a scalable framework for advancing smart automation in construction, mining, and recycling industries. 

**Abstract (ZH)**: 传统的集料分拣方法，无论是人工的还是机械的，往往精度较低、灵活性有限，并且难以适应不同材料的性质，如尺寸、形状和地层。为了解决这些限制，本研究提出了一种计算机视觉辅助机械臂系统，用于建筑和采矿应用中的自主集料分拣。该系统集成了六自由度机械臂、双目立体相机进行三维感知以及基于ROS的控制框架。核心技术包括注意力增强的YOLOv8模型进行集料检测、立体匹配进行三维定位、Denavit-Hartenberg运动学模型进行机械臂运动控制、最小包围矩形分析进行尺寸估计以及手眼标定进行精确坐标对齐。使用四种集料类型进行的实验验证实现了平均抓取和分拣成功率97.5%，分类准确性相当。剩余的挑战包括可靠处理小集料和基于纹理的分类错误。总体而言，所提出的系统显示出显著的潜力，可以通过提高集料处理的生产率、降低操作成本和提高安全性来提升施工、采矿和回收行业的智能自动化水平，同时提供一个可扩展的框架以推进这些行业的智能自动化。 

---
# Jacobian Exploratory Dual-Phase Reinforcement Learning for Dynamic Endoluminal Navigation of Deformable Continuum Robots 

**Title (ZH)**: 渐变探索双重阶段强化学习驱动的柔顺连续机器人动态内腔导航 

**Authors**: Yu Tian, Chi Kit Ng, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.00329)  

**Abstract**: Deformable continuum robots (DCRs) present unique planning challenges due to nonlinear deformation mechanics and partial state observability, violating the Markov assumptions of conventional reinforcement learning (RL) methods. While Jacobian-based approaches offer theoretical foundations for rigid manipulators, their direct application to DCRs remains limited by time-varying kinematics and underactuated deformation dynamics. This paper proposes Jacobian Exploratory Dual-Phase RL (JEDP-RL), a framework that decomposes planning into phased Jacobian estimation and policy execution. During each training step, we first perform small-scale local exploratory actions to estimate the deformation Jacobian matrix, then augment the state representation with Jacobian features to restore approximate Markovianity. Extensive SOFA surgical dynamic simulations demonstrate JEDP-RL's three key advantages over proximal policy optimization (PPO) baselines: 1) Convergence speed: 3.2x faster policy convergence, 2) Navigation efficiency: requires 25% fewer steps to reach the target, and 3) Generalization ability: achieve 92% success rate under material property variations and achieve 83% (33% higher than PPO) success rate in the unseen tissue environment. 

**Abstract (ZH)**: 可变形连续机器人（DCRs）由于非线性变形力学和部分状态可观测性，提出了独特的规划挑战，违背了传统强化学习（RL）方法的马尔可夫假设。虽然雅各宾方法为刚性 manipulator 提供了理论基础，但它们直接应用于 DCRs 仍受到时间变化的动力学和欠驱动变形动力学的限制。本文提出了一种雅各宾探索双阶段 RL（JEDP-RL）框架，将规划分解为雅各宾估计阶段和策略执行阶段。在每次训练步骤中，首先执行小规模的局部探索动作以估计变形雅各宾矩阵，然后通过增加雅各宾特征来补充状态表示，以恢复近似的马尔可夫性。广泛使用的 SOFA 手术动力学模拟显示，JEDP-RL 相对于代理策略优化（PPO）基线具有三大关键优势：1）收敛速度：政策收敛速度快 3.2 倍，2）导航效率：所需步骤减少 25% 以达到目标，3）泛化能力：在材料性质变化下实现 92% 的成功率，在未见过的组织环境中实现 83%（比 PPO 高 33%）的成功率。 

---
# Mechanistic interpretability for steering vision-language-action models 

**Title (ZH)**: 基于机理的可解释性以指导视觉-语言-动作模型 

**Authors**: Bear Häon, Kaylene Stocking, Ian Chuang, Claire Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2509.00328)  

**Abstract**: Vision-Language-Action (VLA) models are a promising path to realizing generalist embodied agents that can quickly adapt to new tasks, modalities, and environments. However, methods for interpreting and steering VLAs fall far short of classical robotics pipelines, which are grounded in explicit models of kinematics, dynamics, and control. This lack of mechanistic insight is a central challenge for deploying learned policies in real-world robotics, where robustness and explainability are critical. Motivated by advances in mechanistic interpretability for large language models, we introduce the first framework for interpreting and steering VLAs via their internal representations, enabling direct intervention in model behavior at inference time. We project feedforward activations within transformer layers onto the token embedding basis, identifying sparse semantic directions - such as speed and direction - that are causally linked to action selection. Leveraging these findings, we introduce a general-purpose activation steering method that modulates behavior in real time, without fine-tuning, reward signals, or environment interaction. We evaluate this method on two recent open-source VLAs, Pi0 and OpenVLA, and demonstrate zero-shot behavioral control in simulation (LIBERO) and on a physical robot (UR5). This work demonstrates that interpretable components of embodied VLAs can be systematically harnessed for control - establishing a new paradigm for transparent and steerable foundation models in robotics. 

**Abstract (ZH)**: 基于视觉-语言-动作(Vision-Language-Action, VLA)模型的可解释操控与 steering 技术：迈向透明可控的机器人基础模型 

---
# Contact-Aided Navigation of Flexible Robotic Endoscope Using Deep Reinforcement Learning in Dynamic Stomach 

**Title (ZH)**: 基于深度强化学习的动态胃部环境下柔性内窥镜接触辅助导航 

**Authors**: Chi Kit Ng, Huxin Gao, Tian-Ao Ren, Jiewen Lai, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.00319)  

**Abstract**: Navigating a flexible robotic endoscope (FRE) through the gastrointestinal tract is critical for surgical diagnosis and treatment. However, navigation in the dynamic stomach is particularly challenging because the FRE must learn to effectively use contact with the deformable stomach walls to reach target locations. To address this, we introduce a deep reinforcement learning (DRL) based Contact-Aided Navigation (CAN) strategy for FREs, leveraging contact force feedback to enhance motion stability and navigation precision. The training environment is established using a physics-based finite element method (FEM) simulation of a deformable stomach. Trained with the Proximal Policy Optimization (PPO) algorithm, our approach achieves high navigation success rates (within 3 mm error between the FRE's end-effector and target) and significantly outperforms baseline policies. In both static and dynamic stomach environments, the CAN agent achieved a 100% success rate with 1.6 mm average error, and it maintained an 85% success rate in challenging unseen scenarios with stronger external disturbances. These results validate that the DRL-based CAN strategy substantially enhances FRE navigation performance over prior methods. 

**Abstract (ZH)**: 基于接触辅助导航的深度强化学习柔性内窥镜在消化道中的灵活导航 

---
# A Framework for Task and Motion Planning based on Expanding AND/OR Graphs 

**Title (ZH)**: 基于扩展AND/OR图的任务与运动规划框架 

**Authors**: Fulvio Mastrogiovanni, Antony Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2509.00317)  

**Abstract**: Robot autonomy in space environments presents unique challenges, including high perception and motion uncertainty, strict kinematic constraints, and limited opportunities for human intervention. Therefore, Task and Motion Planning (TMP) may be critical for autonomous servicing, surface operations, or even in-orbit missions, just to name a few, as it models tasks as discrete action sequencing integrated with continuous motion feasibility assessments. In this paper, we introduce a TMP framework based on expanding AND/OR graphs, referred to as TMP-EAOG, and demonstrate its adaptability to different scenarios. TMP-EAOG encodes task-level abstractions within an AND/OR graph, which expands iteratively as the plan is executed, and performs in-the-loop motion planning assessments to ascertain their feasibility. As a consequence, TMP-EAOG is characterised by the desirable properties of (i) robustness to a certain degree of uncertainty, because AND/OR graph expansion can accommodate for unpredictable information about the robot environment, (ii) controlled autonomy, since an AND/OR graph can be validated by human experts, and (iii) bounded flexibility, in that unexpected events, including the assessment of unfeasible motions, can lead to different courses of action as alternative paths in the AND/OR graph. We evaluate TMP-EAOG on two benchmark domains. We use a simulated mobile manipulator as a proxy for space-grade autonomous robots. Our evaluation shows that TMP-EAOG can deal with a wide range of challenges in the benchmarks. 

**Abstract (ZH)**: 空间环境中机器人的自主性面临着独特挑战，包括高感知和运动不确定性、严格的运动学约束以及有限的人工干预机会。因此，任务与运动规划（TMP）可能是自治服务、表层操作或在轨任务等任务的关键，因为它将任务建模为离散动作序列与连续运动可行性评估的集成。本文介绍了一种基于扩展AND/OR图的TMP框架，称为TMP-EAOG，并展示了其在不同场景下的适应性。TMP-EAOG在执行计划时迭代扩展AND/OR图，并进行在环运动规划评估以确定其可行性。因此，TMP-EAOG具有以下特点：（i）一定程度上对不确定性具有鲁棒性，因为AND/OR图的扩展可以适应对机器人环境的不可预测信息；（ii）可控的自主性，因为AND/OR图可以由人类专家验证；（iii）在动态环境中具有边界灵活性，意外事件，包括不可行运动的评估，可以导致AND/OR图中的不同行动路线。我们使用仿真移动 manipulator 作为太空级自主机器人代理，并在两个基准域上评估了TMP-EAOG。我们的评估结果表明，TMP-EAOG能够应对基准域中的各种挑战。 

---
# TReF-6: Inferring Task-Relevant Frames from a Single Demonstration for One-Shot Skill Generalization 

**Title (ZH)**: TReF-6: 从单次示范中推断任务相关帧以实现一次性技能泛化 

**Authors**: Yuxuan Ding, Shuangge Wang, Tesca Fitzgerald  

**Link**: [PDF](https://arxiv.org/pdf/2509.00310)  

**Abstract**: Robots often struggle to generalize from a single demonstration due to the lack of a transferable and interpretable spatial representation. In this work, we introduce TReF-6, a method that infers a simplified, abstracted 6DoF Task-Relevant Frame from a single trajectory. Our approach identifies an influence point purely from the trajectory geometry to define the origin for a local frame, which serves as a reference for parameterizing a Dynamic Movement Primitive (DMP). This influence point captures the task's spatial structure, extending the standard DMP formulation beyond start-goal imitation. The inferred frame is semantically grounded via a vision-language model and localized in novel scenes by Grounded-SAM, enabling functionally consistent skill generalization. We validate TReF-6 in simulation and demonstrate robustness to trajectory noise. We further deploy an end-to-end pipeline on real-world manipulation tasks, showing that TReF-6 supports one-shot imitation learning that preserves task intent across diverse object configurations. 

**Abstract (ZH)**: 机器人往往难以从单次示范中泛化，因为缺乏可转移和可解释的空间表示。在这项工作中，我们提出了TReF-6方法，该方法可以从单次轨迹中推断出一个简化的、抽象的6DoF任务相关帧。我们的方法仅从轨迹几何中识别出一个影响点，以此定义局部坐标系的原点，该局部坐标系作为动态运动本体（DMP）参数化的参考。该影响点捕获了任务的空间结构，超越了标准的DMP表达范式，使其能够进行除起点和终点模仿之外的任务模仿。通过视觉语言模型，推断出的坐标系具备语义基础，并通过Grounded-SAM在新场景中进行局部化，从而实现功能一致的能力泛化。我们在仿真中验证了TReF-6，并展示了其对轨迹噪声的鲁棒性。此外，我们在真实世界的操作任务中部署了端到端的管道，证明了TReF-6支持保留任务意图的一次性模仿学习，在不同的物体配置中均表现良好。 

---
# Learn from What We HAVE: History-Aware VErifier that Reasons about Past Interactions Online 

**Title (ZH)**: 从前有据可循：在线推理过往交互的历史意识验证器 

**Authors**: Yishu Li, Xinyi Mao, Ying Yuan, Kyutae Sim, Ben Eisner, David Held  

**Link**: [PDF](https://arxiv.org/pdf/2509.00271)  

**Abstract**: We introduce a novel History-Aware VErifier (HAVE) to disambiguate uncertain scenarios online by leveraging past interactions. Robots frequently encounter visually ambiguous objects whose manipulation outcomes remain uncertain until physically interacted with. While generative models alone could theoretically adapt to such ambiguity, in practice they obtain suboptimal performance in ambiguous cases, even when conditioned on action history. To address this, we propose explicitly decoupling action generation from verification: we use an unconditional diffusion-based generator to propose multiple candidate actions and employ our history-aware verifier to select the most promising action by reasoning about past interactions. Through theoretical analysis, we demonstrate that employing a verifier significantly improves expected action quality. Empirical evaluations and analysis across multiple simulated and real-world environments including articulated objects, multi-modal doors, and uneven object pick-up confirm the effectiveness of our method and improvements over baselines. Our project website is available at: this https URL 

**Abstract (ZH)**: 我们提出了一种新的历史意识验证器（HAVEN），通过利用过往交互来在线消歧不确定性场景。机器人经常遇到视觉上具有歧义的物体，其操作结果在实际物理交互之前是不确定的。尽管生成模型理论上能够适应这种不确定性，在实际中，它们在具有歧义性的案例中性能不佳，即使考虑了动作历史。为了解决这一问题，我们提出显式地将动作生成与验证脱钩：我们使用一个无条件的扩散生成器提出多个候选动作，并利用我们的历史意识验证器通过推理过往交互来选择最有前途的动作。通过理论分析，我们证明了采用验证器能够显著提高预期动作质量。在多个模拟和真实世界环境中对具关节物体、多模态门和不规则物体拾取的实验评估与分析证实了我们方法的有效性及优于基线模型的改进。我们的项目网站可在以下链接访问：this https URL。 

---
# Embodied AI in Social Spaces: Responsible and Adaptive Robots in Complex Setting - UKAIRS 2025 (Copy) 

**Title (ZH)**: 嵌入式人工智能在社交空间中的应用：复杂环境中的负责任且适应性强的机器人-UKAIRS 2025 

**Authors**: Aleksandra Landowska, Aislinn D Gomez Bergin, Ayodeji O. Abioye, Jayati Deshmukh, Andriana Bouadouki, Maria Wheadon, Athina Georgara, Dominic Price, Tuyen Nguyen, Shuang Ao, Lokesh Singh, Yi Long, Raffaele Miele, Joel E. Fischer, Sarvapali D. Ramchurn  

**Link**: [PDF](https://arxiv.org/pdf/2509.00218)  

**Abstract**: This paper introduces and overviews a multidisciplinary project aimed at developing responsible and adaptive multi-human multi-robot (MHMR) systems for complex, dynamic settings. The project integrates co-design, ethical frameworks, and multimodal sensing to create AI-driven robots that are emotionally responsive, context-aware, and aligned with the needs of diverse users. We outline the project's vision, methodology, and early outcomes, demonstrating how embodied AI can support sustainable, ethical, and human-centred futures. 

**Abstract (ZH)**: 本文介绍了旨在为复杂动态环境开发负责任和适应性强的多人类多机器人(MHMR)系统的跨学科项目，概述了该项目的设计理念、方法论及早期成果，展示了具身AI如何支持可持续、伦理性和以人类为中心的未来。 

---
# First Order Model-Based RL through Decoupled Backpropagation 

**Title (ZH)**: 基于解耦反向传播的第一阶模型RL方法 

**Authors**: Joseph Amigo, Rooholla Khorrambakht, Elliot Chane-Sane, Nicolas Mansard, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2509.00215)  

**Abstract**: There is growing interest in reinforcement learning (RL) methods that leverage the simulator's derivatives to improve learning efficiency. While early gradient-based approaches have demonstrated superior performance compared to derivative-free methods, accessing simulator gradients is often impractical due to their implementation cost or unavailability. Model-based RL (MBRL) can approximate these gradients via learned dynamics models, but the solver efficiency suffers from compounding prediction errors during training rollouts, which can degrade policy performance. We propose an approach that decouples trajectory generation from gradient computation: trajectories are unrolled using a simulator, while gradients are computed via backpropagation through a learned differentiable model of the simulator. This hybrid design enables efficient and consistent first-order policy optimization, even when simulator gradients are unavailable, as well as learning a critic from simulation rollouts, which is more accurate. Our method achieves the sample efficiency and speed of specialized optimizers such as SHAC, while maintaining the generality of standard approaches like PPO and avoiding ill behaviors observed in other first-order MBRL methods. We empirically validate our algorithm on benchmark control tasks and demonstrate its effectiveness on a real Go2 quadruped robot, across both quadrupedal and bipedal locomotion tasks. 

**Abstract (ZH)**: 利用模拟器梯度提高学习效率的方法：解耦轨迹生成与梯度计算的混合模型强化学习 

---
# Poke and Strike: Learning Task-Informed Exploration Policies 

**Title (ZH)**: 戳击策略：学习任务导向的探索策略 

**Authors**: Marina Y. Aoyama, Joao Moura, Juan Del Aguila Ferrandis, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.00178)  

**Abstract**: In many dynamic robotic tasks, such as striking pucks into a goal outside the reachable workspace, the robot must first identify the relevant physical properties of the object for successful task execution, as it is unable to recover from failure or retry without human intervention. To address this challenge, we propose a task-informed exploration approach, based on reinforcement learning, that trains an exploration policy using rewards automatically generated from the sensitivity of a privileged task policy to errors in estimated properties. We also introduce an uncertainty-based mechanism to determine when to transition from exploration to task execution, ensuring sufficient property estimation accuracy with minimal exploration time. Our method achieves a 90% success rate on the striking task with an average exploration time under 1.2 seconds, significantly outperforming baselines that achieve at most 40% success or require inefficient querying and retraining in a simulator at test time. Additionally, we demonstrate that our task-informed rewards capture the relative importance of physical properties in both the striking task and the classical CartPole example. Finally, we validate our approach by demonstrating its ability to identify object properties and adjust task execution in a physical setup using the KUKA iiwa robot arm. 

**Abstract (ZH)**: 在许多动态机器人任务中，如将冰球击入远离可达工作空间的目标门，机器人必须首先识别出与任务成功执行相关的物理特性，因为它无法在没有人类干预的情况下从失败中恢复或重试。为解决这一挑战，我们提出了一种基于强化学习的任务导向探索方法，通过使用来自优先级任务策略对估计特性误差敏感性的自动奖励来训练探索策略。同时，我们引入了一种基于不确定性机制来确定何时从探索过渡到任务执行的方法，确保在最小的探索时间内获得足够的特性估计精度。该方法在打击任务中实现了90%的成功率，平均探索时间不到1.2秒，显著优于仅能达到40%成功率或在测试时需要在模拟器中进行低效查询和重新训练的基线方法。此外，我们展示了任务导向的奖励能够捕捉打击任务和经典的CartPole示例中物理特性的相对重要性。最后，通过在物理设置中使用KUKA iiwa机器人臂验证该方法的能力，展示了其识别物体特性和调整任务执行的能力。 

---
# A Comparative Study of Spline-Based Trajectory Reconstruction Methods Across Varying Automatic Vehicle Location Data Densities 

**Title (ZH)**: 基于样条的轨迹重建方法在不同自动车辆定位数据密度下的比较研究 

**Authors**: Jake Robbennolt, Sirajum Munira, Stephen D. Boyles  

**Link**: [PDF](https://arxiv.org/pdf/2509.00119)  

**Abstract**: Automatic vehicle location (AVL) data offers insights into transit dynamics, but its effectiveness is often hampered by inconsistent update frequencies, necessitating trajectory reconstruction. This research evaluates 13 trajectory reconstruction methods, including several novel approaches, using high-resolution AVL data from Austin, Texas. We examine the interplay of four critical factors -- velocity, position, smoothing, and data density -- on reconstruction performance. A key contribution of this study is evaluation of these methods across sparse and dense datasets, providing insights into the trade-off between accuracy and resource allocation. Our evaluation framework combines traditional mathematical error metrics for positional and velocity with practical considerations, such as physical realism (e.g., aligning velocity and acceleration with stopped states, deceleration rates, and speed variability). In addition, we provide insight into the relative value of each method in calculating realistic metrics for infrastructure evaluations. Our findings indicate that velocity-aware methods consistently outperform position-only approaches. Interestingly, we discovered that smoothing-based methods can degrade overall performance in complex, congested urban environments, although enforcing monotonicity remains critical. The velocity constrained Hermite interpolation with monotonicity enforcement (VCHIP-ME) yields optimal results, offering a balance between high accuracy and computational efficiency. Its minimal overhead makes it suitable for both historical analysis and real-time applications, providing significant predictive power when combined with dense datasets. These findings offer practical guidance for researchers and practitioners implementing trajectory reconstruction systems and emphasize the importance of investing in higher-frequency AVL data collection for improved analysis. 

**Abstract (ZH)**: 自动车辆定位(AVL)数据提供了公交动态的洞察，但由于更新频率不一致，其有效性受到限制，需进行轨迹重构。本研究使用德克萨斯州奥斯汀的高分辨率AVL数据评估了13种轨迹重构方法，包括若干新型方法，探讨了速度、位置、平滑处理和数据密度等四个关键因素对重构性能的影响。本研究的一大贡献是对这些方法在稀疏和密集数据集上的评估，提供了准确性和资源分配之间权衡的见解。评估框架结合了传统的数学误差指标（如位置和速度误差）以及实际考虑（如物理现实性，例如速度和加速度与停止状态、减速度率和速度变异性的对齐）。此外，本研究还提供了每种方法在计算基础设施评估中的真实度量指标方面的相对价值的见解。研究结果表明，速度感知方法始终优于仅基于位置的方法。有趣的是，我们发现，在复杂、拥堵的市区环境中，基于平滑处理的方法可能会降低总体性能，但确保单调性的约束仍然是关键。速度约束的Hermite插值法（VCHIP-ME）在性能最佳，兼顾高精度和计算效率。其最小的开销使其适用于历史分析和实时应用，与密集数据集结合使用时提供了显著的预测能力。这些发现为实现轨迹重构系统的研究人员和实践者提供了实用指导，并强调了收集更高频率的AVL数据以提高分析效果的重要性。 

---
# Hybrid Perception and Equivariant Diffusion for Robust Multi-Node Rebar Tying 

**Title (ZH)**: 混合感知与同构扩散在鲁棒多节点钢筋绑扎中的应用 

**Authors**: Zhitao Wang, Yirong Xiong, Roberto Horowitz, Yanke Wang, Yuxing Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.00065)  

**Abstract**: Rebar tying is a repetitive but critical task in reinforced concrete construction, typically performed manually at considerable ergonomic risk. Recent advances in robotic manipulation hold the potential to automate the tying process, yet face challenges in accurately estimating tying poses in congested rebar nodes. In this paper, we introduce a hybrid perception and motion planning approach that integrates geometry-based perception with Equivariant Denoising Diffusion on SE(3) (Diffusion-EDFs) to enable robust multi-node rebar tying with minimal training data. Our perception module utilizes density-based clustering (DBSCAN), geometry-based node feature extraction, and principal component analysis (PCA) to segment rebar bars, identify rebar nodes, and estimate orientation vectors for sequential ranking, even in complex, unstructured environments. The motion planner, based on Diffusion-EDFs, is trained on as few as 5-10 demonstrations to generate sequential end-effector poses that optimize collision avoidance and tying efficiency. The proposed system is validated on various rebar meshes, including single-layer, multi-layer, and cluttered configurations, demonstrating high success rates in node detection and accurate sequential tying. Compared with conventional approaches that rely on large datasets or extensive manual parameter tuning, our method achieves robust, efficient, and adaptable multi-node tying while significantly reducing data requirements. This result underscores the potential of hybrid perception and diffusion-driven planning to enhance automation in on-site construction tasks, improving both safety and labor efficiency. 

**Abstract (ZH)**: 基于几何感知与SE(3)不变去噪扩散的钢筋绑扎混合感知与运动规划方法 

---
# OpenTie: Open-vocabulary Sequential Rebar Tying System 

**Title (ZH)**: 开 Vinci: 开放词汇顺序钢筋绑扎系统 

**Authors**: Mingze Liu, Sai Fan, Haozhen Li, Haobo Liang, Yixing Yuan, Yanke Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00064)  

**Abstract**: Robotic practices on the construction site emerge as an attention-attracting manner owing to their capability of tackle complex challenges, especially in the rebar-involved scenarios. Most of existing products and research are mainly focused on flat rebar setting with model training demands. To fulfill this gap, we propose OpenTie, a 3D training-free rebar tying framework utilizing a RGB-to-point-cloud generation and an open-vocabulary detection. We implements the OpenTie via a robotic arm with a binocular camera and guarantees a high accuracy by applying the prompt-based object detection method on the image filtered by our propose post-processing procedure based a image to point cloud generation framework. The system is flexible for horizontal and vertical rebar tying tasks and the experiments on the real-world rebar setting verifies that the effectiveness of the system in practice. 

**Abstract (ZH)**: 基于机器人技术的3D无标注钢筋绑扎框架：利用RGB到点云生成和开放词汇检测 

---
# Correspondence-Free, Function-Based Sim-to-Real Learning for Deformable Surface Control 

**Title (ZH)**: 无需对应关系的功能导向从仿真到现实的学习：可变形表面控制 

**Authors**: Yingjun Tian, Guoxin Fang, Renbo Su, Aoran Lyu, Neelotpal Dutta, Simeon Gill, Andrew Weightman, Charlie C.L. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00060)  

**Abstract**: This paper presents a correspondence-free, function-based sim-to-real learning method for controlling deformable freeform surfaces. Unlike traditional sim-to-real transfer methods that strongly rely on marker points with full correspondences, our approach simultaneously learns a deformation function space and a confidence map -- both parameterized by a neural network -- to map simulated shapes to their real-world counterparts. As a result, the sim-to-real learning can be conducted by input from either a 3D scanner as point clouds (without correspondences) or a motion capture system as marker points (tolerating missed markers). The resultant sim-to-real transfer can be seamlessly integrated into a neural network-based computational pipeline for inverse kinematics and shape control. We demonstrate the versatility and adaptability of our method on both vision devices and across four pneumatically actuated soft robots: a deformable membrane, a robotic mannequin, and two soft manipulators. 

**Abstract (ZH)**: 基于函数的无对应关系的模拟到现实学习方法：控制自由曲面变形 

---
# U2UData-2: A Scalable Swarm UAVs Autonomous Flight Dataset for Long-horizon Tasks 

**Title (ZH)**: U2UData-2: 一项适用于长时间任务的大规模 swarm 航空器自主飞行数据集 

**Authors**: Tongtong Feng, Xin Wang, Feilin Han, Leping Zhang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00055)  

**Abstract**: Swarm UAV autonomous flight for Long-Horizon (LH) tasks is crucial for advancing the low-altitude economy. However, existing methods focus only on specific basic tasks due to dataset limitations, failing in real-world deployment for LH tasks. LH tasks are not mere concatenations of basic tasks, requiring handling long-term dependencies, maintaining persistent states, and adapting to dynamic goal shifts. This paper presents U2UData-2, the first large-scale swarm UAV autonomous flight dataset for LH tasks and the first scalable swarm UAV data online collection and algorithm closed-loop verification platform. The dataset is captured by 15 UAVs in autonomous collaborative flights for LH tasks, comprising 12 scenes, 720 traces, 120 hours, 600 seconds per trajectory, 4.32M LiDAR frames, and 12.96M RGB frames. This dataset also includes brightness, temperature, humidity, smoke, and airflow values covering all flight routes. The platform supports the customization of simulators, UAVs, sensors, flight algorithms, formation modes, and LH tasks. Through a visual control window, this platform allows users to collect customized datasets through one-click deployment online and to verify algorithms by closed-loop simulation. U2UData-2 also introduces an LH task for wildlife conservation and provides comprehensive benchmarks with 9 SOTA models. U2UData-2 can be found at this https URL. 

**Abstract (ZH)**: 大规模自主飞行无人机群数据集U2UData-2及其在长周期任务中的应用平台 

---
# Robotic Fire Risk Detection based on Dynamic Knowledge Graph Reasoning: An LLM-Driven Approach with Graph Chain-of-Thought 

**Title (ZH)**: 基于动态知识图谱推理的机器人火灾风险检测：一种以图链式思维为核心的LLM驱动方法 

**Authors**: Haimei Pan, Jiyun Zhang, Qinxi Wei, Xiongnan Jin, Chen Xinkai, Jie Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00054)  

**Abstract**: Fire is a highly destructive disaster, but effective prevention can significantly reduce its likelihood of occurrence. When it happens, deploying emergency robots in fire-risk scenarios can help minimize the danger to human responders. However, current research on pre-disaster warnings and disaster-time rescue still faces significant challenges due to incomplete perception, inadequate fire situational awareness, and delayed response. To enhance intelligent perception and response planning for robots in fire scenarios, we first construct a knowledge graph (KG) by leveraging large language models (LLMs) to integrate fire domain knowledge derived from fire prevention guidelines and fire rescue task information from robotic emergency response documents. We then propose a new framework called Insights-on-Graph (IOG), which integrates the structured fire information of KG and Large Multimodal Models (LMMs). The framework generates perception-driven risk graphs from real-time scene imagery to enable early fire risk detection and provide interpretable emergency responses for task module and robot component configuration based on the evolving risk situation. Extensive simulations and real-world experiments show that IOG has good applicability and practical application value in fire risk detection and rescue decision-making. 

**Abstract (ZH)**: 基于图的洞见框架（IOG）在火灾场景中的智能感知与响应规划 

---
# Ensemble-Based Event Camera Place Recognition Under Varying Illumination 

**Title (ZH)**: 基于事件相机的自适应光照条件下的场所识别集成方法 

**Authors**: Therese Joseph, Tobias Fischer, Michael Milford  

**Link**: [PDF](https://arxiv.org/pdf/2509.01968)  

**Abstract**: Compared to conventional cameras, event cameras provide a high dynamic range and low latency, offering greater robustness to rapid motion and challenging lighting conditions. Although the potential of event cameras for visual place recognition (VPR) has been established, developing robust VPR frameworks under severe illumination changes remains an open research problem. In this paper, we introduce an ensemble-based approach to event camera place recognition that combines sequence-matched results from multiple event-to-frame reconstructions, VPR feature extractors, and temporal resolutions. Unlike previous event-based ensemble methods, which only utilise temporal resolution, our broader fusion strategy delivers significantly improved robustness under varied lighting conditions (e.g., afternoon, sunset, night), achieving a 57% relative improvement in Recall@1 across day-night transitions. We evaluate our approach on two long-term driving datasets (with 8 km per traverse) without metric subsampling, thereby preserving natural variations in speed and stop duration that influence event density. We also conduct a comprehensive analysis of key design choices, including binning strategies, polarity handling, reconstruction methods, and feature extractors, to identify the most critical components for robust performance. Additionally, we propose a modification to the standard sequence matching framework that enhances performance at longer sequence lengths. To facilitate future research, we will release our codebase and benchmarking framework. 

**Abstract (ZH)**: 事件 cameras 在视觉局部场景识别中的基于集成的方法：在极端光照变化下的鲁棒性增强 

---
# Robustness Enhancement for Multi-Quadrotor Centralized Transportation System via Online Tuning and Learning 

**Title (ZH)**: 基于在线调优与学习的多无人机集中式运输系统鲁棒性增强 

**Authors**: Tianhua Gao, Kohji Tomita, Akiya Kamimura  

**Link**: [PDF](https://arxiv.org/pdf/2509.01952)  

**Abstract**: This paper introduces an adaptive-neuro geometric control for a centralized multi-quadrotor cooperative transportation system, which enhances both adaptivity and disturbance rejection. Our strategy is to coactively tune the model parameters and learn the external disturbances in real-time. To realize this, we augmented the existing geometric control with multiple neural networks and adaptive laws, where the estimated model parameters and the weights of the neural networks are simultaneously tuned and adjusted online. The Lyapunov-based adaptation guarantees bounded estimation errors without requiring either pre-training or the persistent excitation (PE) condition. The proposed control system has been proven to be stable in the sense of Lyapunov under certain preconditions, and its enhanced robustness under scenarios of disturbed environment and model-unmatched plant was demonstrated by numerical simulations. 

**Abstract (ZH)**: 一种用于集中式多旋翼协同运输系统的自适应神经几何控制方法 

---
# Online Identification using Adaptive Laws and Neural Networks for Multi-Quadrotor Centralized Transportation System 

**Title (ZH)**: 基于自适应律和神经网络的多旋翼集中运输系统在线辨识 

**Authors**: Tianhua Gao, Kohji Tomita, Akiya Kamimura  

**Link**: [PDF](https://arxiv.org/pdf/2509.01951)  

**Abstract**: This paper introduces an adaptive-neuro identification method that enhances the robustness of a centralized multi-quadrotor transportation system. This method leverages online tuning and learning on decomposed error subspaces, enabling efficient real-time compensation to time-varying disturbances and model uncertainties acting on the payload. The strategy is to decompose the high-dimensional error space into a set of low-dimensional subspaces. In this way, the identification problem for unseen features is naturally transformed into submappings (``slices'') addressed by multiple adaptive laws and shallow neural networks, which are updated online via Lyapunov-based adaptation without requiring persistent excitation (PE) and offline training. Due to the model-free nature of neural networks, this approach can be well adapted to highly coupled and nonlinear centralized transportation systems. It serves as a feedforward compensator for the payload controller without explicitly relying on the dynamics coupled with the payload, such as cables and quadrotors. The proposed control system has been proven to be stable in the sense of Lyapunov, and its enhanced robustness under time-varying disturbances and model uncertainties was demonstrated by numerical simulations. 

**Abstract (ZH)**: 本文介绍了一种自适应神经识别方法，该方法增强了集中式多旋翼运输系统的鲁棒性。该方法通过在线调整和学习分解后的误差子空间，实现了对时间varying干扰和作用于载荷的模型不确定性进行高效的实时补偿。该策略是将高维误差空间分解为一系列低维子空间。这样一来，对于未见过的特征，识别问题自然地被转换为由多个自适应律和浅层神经网络处理的子映射（“切片”），这些网络通过基于Lyapunov的方法在线更新，无需持续激励（PE）和离线训练。由于神经网络的无模态特性，该方法可以很好地适应高度耦合和非线性的集中式运输系统。它作为载荷控制器的前馈补偿器工作，而无需显式依赖于与载荷耦合的动力学，如缆绳和旋翼机。所提出的控制系统在Lyapunov意义下被证明是稳定的，并通过数值仿真展示了其在时间varying干扰和模型不确定性下的增强鲁棒性。 

---
# Nonlinear Model Predictive Control-Based Reverse Path-Planning and Path-Tracking Control of a Vehicle with Trailer System 

**Title (ZH)**: 基于非线性模型预测控制的带挂车车辆反向路径规划与路径跟踪控制 

**Authors**: Xincheng Cao, Haochong Chen, Bilin Aksun-Guvenc, Levent Guvenc, Brian Link, Peter J Richmond, Dokyung Yim, Shihong Fan, John Harber  

**Link**: [PDF](https://arxiv.org/pdf/2509.01820)  

**Abstract**: Reverse parking maneuvers of a vehicle with trailer system is a challenging task to complete for human drivers due to the unstable nature of the system and unintuitive controls required to orientate the trailer properly. This paper hence proposes an optimization-based automation routine to handle the path-planning and path-tracking control process of such type of maneuvers. The proposed approach utilizes nonlinear model predictive control (NMPC) to robustly guide the vehicle-trailer system into the desired parking space, and an optional forward repositioning maneuver can be added as an additional stage of the parking process to obtain better system configurations, before backward motion can be attempted again to get a good final pose. The novelty of the proposed approach is the simplicity of its formulation, as the path-planning and path-tracking operations are only conducted on the trailer being viewed as a standalone vehicle, before the control inputs are propagated to the tractor vehicle via inverse kinematic relationships also derived in this paper. Simulation case studies and hardware-in-the-loop tests are performed, and the results demonstrate the efficacy of the proposed approach. 

**Abstract (ZH)**: 带有挂车系统的车辆倒车入库操作是一个对人类驾驶员而言具有挑战性任务，由于该系统的不稳定性质和对挂车正确定向所需的非直观控制。因此，本文提出了一种基于优化的自动化流程来处理此类操作的路径规划和路径跟踪控制过程。所提出的方法利用非线性模型预测控制（NMPC） robustly 将车辆-挂车系统引导至目标停车位，并可根据需要添加一个额外的前向重新定位操作阶段来获得更好的系统配置，然后再次尝试后向运动以获得良好的最终姿态。所提出方法的创新之处在于其公式化的简洁性，路径规划和路径跟踪操作仅在将挂车视为独立车辆进行后，通过在本文推导的逆 kinematic 关系将控制输入传递至牵引车。进行了仿真案例研究和硬件在环测试，结果表明所提出方法的有效性。 

---
# EgoTouch: On-Body Touch Input Using AR/VR Headset Cameras 

**Title (ZH)**: 基于身体的触觉输入：使用AR/VR头显摄像头的交互方式 

**Authors**: Vimal Mollyn, Chris Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2509.01786)  

**Abstract**: In augmented and virtual reality (AR/VR) experiences, a user's arms and hands can provide a convenient and tactile surface for touch input. Prior work has shown on-body input to have significant speed, accuracy, and ergonomic benefits over in-air interfaces, which are common today. In this work, we demonstrate high accuracy, bare hands (i.e., no special instrumentation of the user) skin input using just an RGB camera, like those already integrated into all modern XR headsets. Our results show this approach can be accurate, and robust across diverse lighting conditions, skin tones, and body motion (e.g., input while walking). Finally, our pipeline also provides rich input metadata including touch force, finger identification, angle of attack, and rotation. We believe these are the requisite technical ingredients to more fully unlock on-skin interfaces that have been well motivated in the HCI literature but have lacked robust and practical methods. 

**Abstract (ZH)**: 在增强现实和虚拟现实（AR/VR）体验中，用户的胳膊和手可以提供一种便捷且具有触觉反馈的表面用于触控输入。以往研究表明，在体输入相较于当前常见的空中界面具有显著的速度、准确性和人体工程学优势。在这项工作中，我们利用一个像现代XR头显中已集成的RGB摄像头这样的普通摄像头，展示了仅通过裸手皮肤输入实现高精度输入的方法。我们的结果显示，这种方法在不同光照条件、不同肤色和身体运动（如行走时输入）下具有鲁棒性。最后，我们的处理管道还提供了丰富的输入元数据，包括触控力度、手指识别、攻击角度和旋转。我们认为这些是充分利用已被人机交互文献充分证明但缺乏稳健和实用方法的在体界面所需的技术要件。 

---
# Learning to Coordinate: Distributed Meta-Trajectory Optimization Via Differentiable ADMM-DDP 

**Title (ZH)**: 学习协调：基于可微ADMM-DDP的分布式元轨迹优化 

**Authors**: Bingheng Wang, Yichao Gao, Tianchen Sun, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.01630)  

**Abstract**: Distributed trajectory optimization via ADMM-DDP is a powerful approach for coordinating multi-agent systems, but it requires extensive tuning of tightly coupled hyperparameters that jointly govern local task performance and global coordination. In this paper, we propose Learning to Coordinate (L2C), a general framework that meta-learns these hyperparameters, modeled by lightweight agent-wise neural networks, to adapt across diverse tasks and agent configurations. L2C differentiates end-to-end through the ADMM-DDP pipeline in a distributed manner. It also enables efficient meta-gradient computation by reusing DDP components such as Riccati recursions and feedback gains. These gradients correspond to the optimal solutions of distributed matrix-valued LQR problems, coordinated across agents via an auxiliary ADMM framework that becomes convex under mild assumptions. Training is further accelerated by truncating iterations and meta-learning ADMM penalty parameters optimized for rapid residual reduction, with provable Lipschitz-bounded gradient errors. On a challenging cooperative aerial transport task, L2C generates dynamically feasible trajectories in high-fidelity simulation using IsaacSIM, reconfigures quadrotor formations for safe 6-DoF load manipulation in tight spaces, and adapts robustly to varying team sizes and task conditions, while achieving up to $88\%$ faster gradient computation than state-of-the-art methods. 

**Abstract (ZH)**: 分布式轨迹优化 via ADMM-DDP 是一个多智能体系统协调的一种强大方法，但需要对紧密耦合的超参数进行广泛的调优，这些超参数同时管理局部任务性能和全局协调。本文提出了一种名为 Learning to Coordinate (L2C) 的通用框架，该框架通过轻量级的智能体神经网络元学习这些超参数，以适应多样化的任务和智能体配置。L2C 通过分布式的方式在 ADMM-DDP 管道中进行端到端的微分。它还通过重用诸如 Riccati 递推和反馈增益等 DDP 组件来实现高效的元梯度计算。这些梯度对应于通过辅助的 ADMM 框架协调的分布式矩阵值 LQR 问题的最优解，在轻微假设下成为凸问题。通过截断迭代次数和优化 ADMM 惩罚参数加速训练，这些参数旨在快速减少余差，并具有可证明的 Lipschitz 上界的梯度误差。在一项具有挑战性的协同空中运输任务中，L2C 使用 IsaacSIM 进行高保真模拟以生成动态可行的轨迹，重新配置四旋翼机队形以在狭窄空间内安全执行 6 自由度载荷操作，并在团队规模和任务条件变化时表现出高度鲁棒性，同时梯度计算速度比现有方法快达 88%。 

---
# TransForSeg: A Multitask Stereo ViT for Joint Stereo Segmentation and 3D Force Estimation in Catheterization 

**Title (ZH)**: TransForSeg: 一种用于导管操作中联合立体分割和三维力估计的多任务立体ViT 

**Authors**: Pedram Fekri, Mehrdad Zadeh, Javad Dargahi  

**Link**: [PDF](https://arxiv.org/pdf/2509.01605)  

**Abstract**: Recently, the emergence of multitask deep learning models has enhanced catheterization procedures by providing tactile and visual perception data through an end-to-end architec- ture. This information is derived from a segmentation and force estimation head, which localizes the catheter in X-ray images and estimates the applied pressure based on its deflection within the image. These stereo vision architectures incorporate a CNN- based encoder-decoder that captures the dependencies between X-ray images from two viewpoints, enabling simultaneous 3D force estimation and stereo segmentation of the catheter. With these tasks in mind, this work approaches the problem from a new perspective. We propose a novel encoder-decoder Vision Transformer model that processes two input X-ray images as separate sequences. Given sequences of X-ray patches from two perspectives, the transformer captures long-range dependencies without the need to gradually expand the receptive field for either image. The embeddings generated by both the encoder and decoder are fed into two shared segmentation heads, while a regression head employs the fused information from the decoder for 3D force estimation. The proposed model is a stereo Vision Transformer capable of simultaneously segmenting the catheter from two angles while estimating the generated forces at its tip in 3D. This model has undergone extensive experiments on synthetic X-ray images with various noise levels and has been compared against state-of-the-art pure segmentation models, vision-based catheter force estimation methods, and a multitask catheter segmentation and force estimation approach. It outperforms existing models, setting a new state-of-the-art in both catheter segmentation and force estimation. 

**Abstract (ZH)**: 最近，多任务深度学习模型的出现通过端到端架构提供了触觉和视觉感知数据，从而增强了导管插入程序。这些信息来自一个分割和力估计算法头部，该头部在X射线图像中定位导管并基于其在图像中的偏转估计所施加的压力。这些立体视觉架构包含一个基于CNN的编码器-解码器，能够捕获来自两个视点的X射线图像之间的依赖关系，从而实现同时进行3D力估计和立体分割。基于此，本研究从一个新视角来解决该问题。我们提出了一种新的编码器-解码器视觉变换器模型，将两幅输入的X射线图像分别处理为独立的序列。给定来自两个视角的X射线补丁序列，变换器在不需要逐步扩大任一图像的感受野的情况下捕获远程依赖性。编码器和解码器生成的嵌入分别输入到两个共享的分割头部，而回归头部则利用解码器融合的信息进行3D力估计。所提出的模型是一个立体视觉变换器，能够同时从两个角度分割导管，并在3D中估计其尖端产生的力。该模型已在不同噪声水平的合成X射线图像上进行了广泛实验，并与最先进的纯分割模型、基于视觉的导管力估计方法以及多任务导管分割与力估计方法进行了比较。它在导管分割和力估计方面均优于现有模型，建立了新的最先进的水平。 

---
# Quantum game models for interaction-aware decision-making in automated driving 

**Title (ZH)**: 量子博弈模型在自动化驾驶中的交互感知决策making 

**Authors**: Karim Essalmi, Fernando Garrido, Fawzi Nashashibi  

**Link**: [PDF](https://arxiv.org/pdf/2509.01582)  

**Abstract**: Decision-making in automated driving must consider interactions with surrounding agents to be effective. However, traditional methods often neglect or oversimplify these interactions because they are difficult to model and solve, which can lead to overly conservative behavior of the ego vehicle. To address this gap, we propose two quantum game models, QG-U1 (Quantum Game - Unitary 1) and QG-G4 (Quantum Game - Gates 4), for interaction-aware decision-making. These models extend classical game theory by incorporating principles of quantum mechanics, such as superposition, interference, and entanglement. Specifically, QG-U1 and QG-G4 are designed for two-player games with two strategies per player and can be executed in real time on a standard computer without requiring quantum hardware. We evaluate both models in merging and roundabout scenarios and compare them with classical game-theoretic methods and baseline approaches (IDM, MOBIL, and a utility-based technique). Results show that QG-G4 achieves lower collision rates and higher success rates compared to baseline methods, while both quantum models yield higher expected payoffs than classical game approaches under certain parameter settings. 

**Abstract (ZH)**: 自动化驾驶中的决策必须考虑与周围代理的互动以实现有效性。然而，传统方法往往因为这些互动难以建模和解决而忽视或简化这些互动，这可能导致自我车辆表现出过度保守的行为。为此，我们提出了两种量子博弈模型，QG-U1（量子博弈-幺正1）和QG-G4（量子博弈-门4），以实现互动感知的决策。这些模型将经典博弈理论扩展到包含量子力学原则（如叠加、干涉和纠缠）。具体而言，QG-U1和QG-G4适用于每方有两个策略的两玩家博弈，并且可以在标准计算机上实时运行而无需量子硬件。我们在这两种模型在并道和环岛场景中进行了评估，并将它们与经典博弈理论方法和基线方法（IDM、MOBIL和基于效用的技术）进行了比较。结果表明，在某些参数设置下，QG-G4相比基线方法具有更低的碰撞率和更高的成功率，而两种量子模型在与经典博弈方法相比时具有更高的预期收益。 

---
# End-to-End Low-Level Neural Control of an Industrial-Grade 6D Magnetic Levitation System 

**Title (ZH)**: 端到端低级神经控制的工业级6D磁悬浮系统 

**Authors**: Philipp Hartmann, Jannick Stranghöner, Klaus Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2509.01388)  

**Abstract**: Magnetic levitation is poised to revolutionize industrial automation by integrating flexible in-machine product transport and seamless manipulation. It is expected to become the standard drive for automated manufacturing. However, controlling such systems is inherently challenging due to their complex, unstable dynamics. Traditional control approaches, which rely on hand-crafted control engineering, typically yield robust but conservative solutions, with their performance closely tied to the expertise of the engineering team. In contrast, neural control learning presents a promising alternative. This paper presents the first neural controller for 6D magnetic levitation. Trained end-to-end on interaction data from a proprietary controller, it directly maps raw sensor data and 6D reference poses to coil current commands. The neural controller can effectively generalize to previously unseen situations while maintaining accurate and robust control. These results underscore the practical feasibility of learning-based neural control in complex physical systems and suggest a future where such a paradigm could enhance or even substitute traditional engineering approaches in demanding real-world applications. The trained neural controller, source code, and demonstration videos are publicly available at this https URL. 

**Abstract (ZH)**: 磁悬浮技术有望通过集成灵活的在机产品运输和无缝操作来革新工业自动化，并预期将成为自动化制造的标准驱动。然而，由于其复杂的不稳定动态，控制此类系统本身具有固有的挑战性。传统的控制方法依赖于手工设计的控制工程，通常会产生稳健但保守的解决方案，其性能与工程团队的专业知识密切相关。相比之下，神经控制学习提供了有 promise 的替代方案。本文提出了首个用于6D磁悬浮的神经控制器，该控制器端到端地在专有控制器的交互数据上进行训练，直接将原始传感器数据和6D参考姿态映射到线圈电流命令。神经控制器可以在维护精确和鲁棒控制的同时，有效泛化到未见过的情况。这些结果强调了基于学习的神经控制在复杂物理系统中的实际可行性，并暗示了一种未来，其中此类范式可能在苛刻的实际应用中增强甚至替代传统工程方法。已训练的神经控制器、源代码和演示视频可在以下网址公开获取：this https URL。 

---
# Metamorphic Testing of Multimodal Human Trajectory Prediction 

**Title (ZH)**: 多模态人类轨迹预测的 metamorphic 测试 

**Authors**: Helge Spieker, Nadjib Lazaar, Arnaud Gotlieb, Nassim Belmecheri  

**Link**: [PDF](https://arxiv.org/pdf/2509.01294)  

**Abstract**: Context: Predicting human trajectories is crucial for the safety and reliability of autonomous systems, such as automated vehicles and mobile robots. However, rigorously testing the underlying multimodal Human Trajectory Prediction (HTP) models, which typically use multiple input sources (e.g., trajectory history and environment maps) and produce stochastic outputs (multiple possible future paths), presents significant challenges. The primary difficulty lies in the absence of a definitive test oracle, as numerous future trajectories might be plausible for any given scenario. Objectives: This research presents the application of Metamorphic Testing (MT) as a systematic methodology for testing multimodal HTP systems. We address the oracle problem through metamorphic relations (MRs) adapted for the complexities and stochastic nature of HTP. Methods: We present five MRs, targeting transformations of both historical trajectory data and semantic segmentation maps used as an environmental context. These MRs encompass: 1) label-preserving geometric transformations (mirroring, rotation, rescaling) applied to both trajectory and map inputs, where outputs are expected to transform correspondingly. 2) Map-altering transformations (changing semantic class labels, introducing obstacles) with predictable changes in trajectory distributions. We propose probabilistic violation criteria based on distance metrics between probability distributions, such as the Wasserstein or Hellinger distance. Conclusion: This study introduces tool, a MT framework for the oracle-less testing of multimodal, stochastic HTP systems. It allows for assessment of model robustness against input transformations and contextual changes without reliance on ground-truth trajectories. 

**Abstract (ZH)**: 基于元模型测试的无注解多模态人类轨迹预测系统测试方法 

---
# An AI-Based Shopping Assistant System to Support the Visually Impaired 

**Title (ZH)**: 基于AI的购物助理系统以支持视力障碍者 

**Authors**: Larissa R. de S. Shibata, Ankit A. Ravankar, Jose Victorio Salazar Luces, Yasuhisa Hirata  

**Link**: [PDF](https://arxiv.org/pdf/2509.01246)  

**Abstract**: Shopping plays a significant role in shaping consumer identity and social integration. However, for individuals with visual impairments, navigating in supermarkets and identifying products can be an overwhelming and challenging experience. This paper presents an AI-based shopping assistant prototype designed to enhance the autonomy and inclusivity of visually impaired individuals in supermarket environments. The system integrates multiple technologies, including computer vision, speech recognition, text-to-speech synthesis, and indoor navigation, into a single, user-friendly platform. Using cameras for ArUco marker detection and real-time environmental scanning, the system helps users navigate the store, identify product locations, provide real-time auditory guidance, and gain context about their surroundings. The assistant interacts with the user through voice commands and multimodal feedback, promoting a more dynamic and engaging shopping experience. The system was evaluated through experiments, which demonstrated its ability to guide users effectively and improve their shopping experience. This paper contributes to the development of inclusive AI-driven assistive technologies aimed at enhancing accessibility and user independence for the shopping experience. 

**Abstract (ZH)**: 视觉障碍人士超市购物辅助系统的AI原型设计与实现 

---
# Robix: A Unified Model for Robot Interaction, Reasoning and Planning 

**Title (ZH)**: Robix：统一的机器人交互、推理和规划模型 

**Authors**: Huang Fang, Mengxi Zhang, Heng Dong, Wei Li, Zixuan Wang, Qifeng Zhang, Xueyun Tian, Yucheng Hu, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.01106)  

**Abstract**: We introduce Robix, a unified model that integrates robot reasoning, task planning, and natural language interaction within a single vision-language architecture. Acting as the high-level cognitive layer in a hierarchical robot system, Robix dynamically generates atomic commands for the low-level controller and verbal responses for human interaction, enabling robots to follow complex instructions, plan long-horizon tasks, and interact naturally with human within an end-to-end framework. Robix further introduces novel capabilities such as proactive dialogue, real-time interruption handling, and context-aware commonsense reasoning during task execution. At its core, Robix leverages chain-of-thought reasoning and adopts a three-stage training strategy: (1) continued pretraining to enhance foundational embodied reasoning abilities including 3D spatial understanding, visual grounding, and task-centric reasoning; (2) supervised finetuning to model human-robot interaction and task planning as a unified reasoning-action sequence; and (3) reinforcement learning to improve reasoning-action consistency and long-horizon task coherence. Extensive experiments demonstrate that Robix outperforms both open-source and commercial baselines (e.g., GPT-4o and Gemini 2.5 Pro) in interactive task execution, demonstrating strong generalization across diverse instruction types (e.g., open-ended, multi-stage, constrained, invalid, and interrupted) and various user-involved tasks such as table bussing, grocery shopping, and dietary filtering. 

**Abstract (ZH)**: Robix：统一的机器人推理、任务规划和自然语言交互模型 

---
# Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Movable Obstacles 

**Title (ZH)**: 符号规划与移动障碍物条件下极度密集环境中的多代理路径寻找 

**Authors**: Bo Fu, Zhe Chen, Rahul Chandan, Alex Barbosa, Michael Caldara, Joey Durham, Federico Pecora  

**Link**: [PDF](https://arxiv.org/pdf/2509.01022)  

**Abstract**: We introduce the Block Rearrangement Problem (BRaP), a challenging component of large warehouse management which involves rearranging storage blocks within dense grids to achieve a target state. We formally define the BRaP as a graph search problem. Building on intuitions from sliding puzzle problems, we propose five search-based solution algorithms, leveraging joint configuration space search, classical planning, multi-agent pathfinding, and expert heuristics. We evaluate the five approaches empirically for plan quality and scalability. Despite the exponential relation between search space size and block number, our methods demonstrate efficiency in creating rearrangement plans for deeply buried blocks in up to 80x80 grids. 

**Abstract (ZH)**: 块重排问题：仓储管理中的挑战性组件及其图搜索问题形式化定义与解决方案算法研究 

---
# AI-driven Dispensing of Coral Reseeding Devices for Broad-scale Restoration of the Great Barrier Reef 

**Title (ZH)**: AI驱动的珊瑚重播设备分发以实现大堡礁的大规模修复 

**Authors**: Scarlett Raine, Benjamin Moshirian, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.01019)  

**Abstract**: Coral reefs are on the brink of collapse, with climate change, ocean acidification, and pollution leading to a projected 70-90% loss of coral species within the next decade. Restoration efforts are crucial, but their success hinges on introducing automation to upscale efforts. We present automated deployment of coral re-seeding devices powered by artificial intelligence, computer vision, and robotics. Specifically, we perform automated substrate classification, enabling detection of areas of the seafloor suitable for coral growth, thus significantly reducing reliance on human experts and increasing the range and efficiency of restoration. Real-world testing of the algorithms on the Great Barrier Reef leads to deployment accuracy of 77.8%, sub-image patch classification of 89.1%, and real-time model inference at 5.5 frames per second. Further, we present and publicly contribute a large collection of annotated substrate image data to foster future research in this area. 

**Abstract (ZH)**: 珊瑚礁正处于崩溃的边缘，气候变化、海洋酸化和污染导致未来十年珊瑚物种可能会减少70-90%。恢复工作至关重要，但其成功取决于引入自动化技术以扩大努力规模。我们提出了一种由人工智能、计算机视觉和机器人技术驱动的自动投放珊瑚重新播种设备的方法。具体而言，我们实现了自动底质分类，能够检测适合珊瑚生长的海底区域，从而大大减少了对人类专家的依赖，并提高了恢复工作的范围和效率。在大堡礁的实际测试中，算法的部署准确率为77.8%，子图像patch分类准确率为89.1%，实时模型推理速率为每秒5.5帧。此外，我们还提供并公开贡献了大量的标注底质图像数据，以促进该领域的未来研究。 

---
# ER-LoRA: Effective-Rank Guided Adaptation for Weather-Generalized Depth Estimation 

**Title (ZH)**: ER-LoRA: 基于有效秩引导的天气通用深度估计适应性方法 

**Authors**: Weilong Yan, Xin Zhang, Robby T. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00665)  

**Abstract**: Monocular depth estimation under adverse weather conditions (e.g.\ rain, fog, snow, and nighttime) remains highly challenging due to the lack of reliable ground truth and the difficulty of learning from unlabeled real-world data. Existing methods often rely on synthetic adverse data with pseudo-labels, which suffer from domain gaps, or employ self-supervised learning, which violates photometric assumptions in adverse scenarios. In this work, we propose to achieve weather--generalized depth estimation by Parameter--Efficient Fine--Tuning (PEFT) of Vision Foundation Models (VFMs), using only a small amount of high--visibility (normal) data. While PEFT has shown strong performance in semantic tasks such as segmentation, it remains underexplored for geometry--centric tasks like depth estimation -- especially in terms of balancing effective adaptation with the preservation of pretrained knowledge. To this end, we introduce the Selecting--Tuning--Maintaining (STM) strategy, which structurally decomposes the pretrained weights of VFMs based on two kinds of effective ranks (entropy--rank and stable--rank). In the tuning phase, we adaptively select the proper rank number as well as the task--aware singular directions for initialization, based on the entropy--rank and full--tuned weight; while in the maintaining stage, we enforce a principal direction regularization based on the stable--rank. This design guarantees flexible task adaptation while preserving the strong generalization capability of the pretrained VFM. Extensive experiments on four real--world benchmarks across diverse weather conditions demonstrate that STM not only outperforms existing PEFT methods and full fine--tuning but also surpasses methods trained with adverse synthetic data, and even the depth foundation model 

**Abstract (ZH)**: 单目深度估计在恶劣天气条件（如雨、雾、雪和夜间）下仍极具挑战性，由于缺乏可靠的地面真实数据和从未标记的真实世界数据中学习的难度。现有方法通常依赖合成的恶劣数据和伪标签，这会导致域间差异，或者采用自监督学习，但在恶劣场景下这会违反光度假设。在这项工作中，我们提出通过Vision Foundation Models (VFMs)的参数高效微调（PEFT）来实现天气通用的深度估计，仅使用少量高可见度（正常）数据。虽然PEFT在语义任务（如分割）中表现出色，但在几何中心任务（如深度估计）中的探索仍处于初级阶段，特别地，在有效适应和预训练知识保留间取得平衡方面尚未得到充分探索。为此，我们引入了Selecting-Tuning-Maintaining（STM）策略，该策略基于两种有效秩（熵秩和稳定秩）对VFMs的预训练权重进行结构分解。在微调阶段，我们基于熵秩和全微调权重自适应地选择合适的秩数以及任务感知的奇异方向进行初始化；而在维护阶段，我们基于稳定秩施加主方向正则化。这种设计确保了任务的灵活适应能力，同时保留了预训练VFMs的强大泛化能力。在四种不同天气条件的现实世界基准上的广泛实验表明，STM不仅优于现有的PEFT方法和全微调方法，还超越了使用恶劣合成数据训练的方法，甚至超过了深度基础模型。 

---
# MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation 

**Title (ZH)**: MV-SSM：多视图状态空间建模在3D人体姿态估计中的应用 

**Authors**: Aviral Chharia, Wenbo Gou, Haoye Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00649)  

**Abstract**: While significant progress has been made in single-view 3D human pose estimation, multi-view 3D human pose estimation remains challenging, particularly in terms of generalizing to new camera configurations. Existing attention-based transformers often struggle to accurately model the spatial arrangement of keypoints, especially in occluded scenarios. Additionally, they tend to overfit specific camera arrangements and visual scenes from training data, resulting in substantial performance drops in new settings. In this study, we introduce a novel Multi-View State Space Modeling framework, named MV-SSM, for robustly estimating 3D human keypoints. We explicitly model the joint spatial sequence at two distinct levels: the feature level from multi-view images and the person keypoint level. We propose a Projective State Space (PSS) block to learn a generalized representation of joint spatial arrangements using state space modeling. Moreover, we modify Mamba's traditional scanning into an effective Grid Token-guided Bidirectional Scanning (GTBS), which is integral to the PSS block. Multiple experiments demonstrate that MV-SSM achieves strong generalization, outperforming state-of-the-art methods: +10.8 on AP25 (+24%) on the challenging three-camera setting in CMU Panoptic, +7.0 on AP25 (+13%) on varying camera arrangements, and +15.3 PCP (+38%) on Campus A1 in cross-dataset evaluations. Project Website: this https URL 

**Abstract (ZH)**: 多视角状态空间建模：一种稳健的3D人体关键点估计方法 

---
# AGS: Accelerating 3D Gaussian Splatting SLAM via CODEC-Assisted Frame Covisibility Detection 

**Title (ZH)**: AGS: 通过CODEC辅助帧共视性检测加速3D高斯体绘制SLAM 

**Authors**: Houshu He, Naifeng Jing, Li Jiang, Xiaoyao Liang, Zhuoran Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.00433)  

**Abstract**: Simultaneous Localization and Mapping (SLAM) is a critical task that enables autonomous vehicles to construct maps and localize themselves in unknown environments. Recent breakthroughs combine SLAM with 3D Gaussian Splatting (3DGS) to achieve exceptional reconstruction fidelity. However, existing 3DGS-SLAM systems provide insufficient throughput due to the need for multiple training iterations per frame and the vast number of Gaussians.
In this paper, we propose AGS, an algorithm-hardware co-design framework to boost the efficiency of 3DGS-SLAM based on the intuition that SLAM systems process frames in a streaming manner, where adjacent frames exhibit high similarity that can be utilized for acceleration. On the software level: 1) We propose a coarse-then-fine-grained pose tracking method with respect to the robot's movement. 2) We avoid redundant computations of Gaussians by sharing their contribution information across frames. On the hardware level, we propose a frame covisibility detection engine to extract intermediate data from the video CODEC. We also implement a pose tracking engine and a mapping engine with workload schedulers to efficiently deploy the AGS algorithm. Our evaluation shows that AGS achieves up to $17.12\times$, $6.71\times$, and $5.41\times$ speedups against the mobile and high-end GPUs, and a state-of-the-art 3DGS accelerator, GSCore. 

**Abstract (ZH)**: 三维Gauss散点表示同步定位与Mapping (AGS):一种算法-硬件协同设计框架 

---
# Domain Adaptation-Based Crossmodal Knowledge Distillation for 3D Semantic Segmentation 

**Title (ZH)**: 基于领域适应的跨模态知识精炼方法及其在3D语义分割中的应用 

**Authors**: Jialiang Kang, Jiawen Wang, Dingsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00379)  

**Abstract**: Semantic segmentation of 3D LiDAR data plays a pivotal role in autonomous driving. Traditional approaches rely on extensive annotated data for point cloud analysis, incurring high costs and time investments. In contrast, realworld image datasets offer abundant availability and substantial scale. To mitigate the burden of annotating 3D LiDAR point clouds, we propose two crossmodal knowledge distillation methods: Unsupervised Domain Adaptation Knowledge Distillation (UDAKD) and Feature and Semantic-based Knowledge Distillation (FSKD). Leveraging readily available spatio-temporally synchronized data from cameras and LiDARs in autonomous driving scenarios, we directly apply a pretrained 2D image model to unlabeled 2D data. Through crossmodal knowledge distillation with known 2D-3D correspondence, we actively align the output of the 3D network with the corresponding points of the 2D network, thereby obviating the necessity for 3D annotations. Our focus is on preserving modality-general information while filtering out modality-specific details during crossmodal distillation. To achieve this, we deploy self-calibrated convolution on 3D point clouds as the foundation of our domain adaptation module. Rigorous experimentation validates the effectiveness of our proposed methods, consistently surpassing the performance of state-of-the-art approaches in the field. 

**Abstract (ZH)**: 基于跨模态知识蒸馏的3D LiDAR语义分割方法 

---
# A Layered Control Perspective on Legged Locomotion: Embedding Reduced Order Models via Hybrid Zero Dynamics 

**Title (ZH)**: 基于分层控制视角的足式运动控制：通过混合零动力学嵌入降阶模型 

**Authors**: Sergio A. Esteban, Max H. Cohen, Adrian B. Ghansah, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2509.00294)  

**Abstract**: Reduced-order models (ROMs) provide a powerful means of synthesizing dynamic walking gaits on legged robots. Yet this approach lacks the formal guarantees enjoyed by methods that utilize the full-order model (FOM) for gait synthesis, e.g., hybrid zero dynamics. This paper aims to unify these approaches through a layered control perspective. In particular, we establish conditions on when a ROM of locomotion yields stable walking on the full-order hybrid dynamics. To achieve this result, given an ROM we synthesize a zero dynamics manifold encoding the behavior of the ROM -- controllers can be synthesized that drive the FOM to this surface, yielding hybrid zero dynamics. We prove that a stable periodic orbit in the ROM implies an input-to-state stable periodic orbit of the FOM's hybrid zero dynamics, and hence the FOM dynamics. This result is demonstrated in simulation on a linear inverted pendulum ROM and a 5-link planar walking FOM. 

**Abstract (ZH)**: Reduced-order模型（ROMs）为合成有腿机器人的动态行走步态提供了一种强大的手段。然而，这种方法缺乏使用完整阶数模型（FOM）进行步态合成的方法所享有的正式保证，例如混合零动态。本文通过分层控制视角旨在统一这两种方法。具体而言，我们建立了当运动学ROM导致在完整阶数混合动力学中稳定行走时的条件。为了实现这一结果，给定一个ROM，我们合成了一个零动态流形，编码ROM的行为——可以合成控制器将FOM驱动到这个表面上，从而得到混合零动态。我们证明了ROM中的稳定周期轨道意味着FOM的混合零动态及其动力学的输入到状态稳定的周期轨道，这一结果在仿真的线性倒摆ROM和5连杆平面行走FOM上得到了验证。 

---
# Embodied AI: Emerging Risks and Opportunities for Policy Action 

**Title (ZH)**: 具身人工智能：新兴风险与政策行动机会 

**Authors**: Jared Perlo, Alexander Robey, Fazl Barez, Luciano Floridi, Jakob Mökander  

**Link**: [PDF](https://arxiv.org/pdf/2509.00117)  

**Abstract**: The field of embodied AI (EAI) is rapidly advancing. Unlike virtual AI, EAI can exist in, learn from, reason about, and act in the physical world. Given recent innovations in large language and multimodal models, along with increasingly advanced and responsive hardware, EAI systems are rapidly growing in capabilities and operational domains. These advances present significant risks, including physical harm from malicious use, mass surveillance, and economic and societal disruption. However, these risks have been severely overlooked by policymakers. Existing policies, such as international standards for industrial robots or statutes governing autonomous vehicles, are insufficient to address the full range of concerns. While lawmakers are increasingly focused on AI, there is now an urgent need to extend and adapt existing frameworks to account for the unique risks of EAI. To help bridge this gap, this paper makes three contributions: first, we provide a foundational taxonomy of key physical, informational, economic, and social EAI risks. Secondly, we analyze policies in the US, EU, and UK to identify how existing frameworks address these risks and where these policies leave critical gaps. We conclude by offering concrete policy recommendations to address the coming wave of EAI innovation, including mandatory testing and certification for EAI systems, clarified liability frameworks, and forward-looking strategies to manage and prepare for transformative economic and societal impacts. 

**Abstract (ZH)**: embodied AI领域正 rapid地进步。与虚拟AI不同，embodied AI可以在物理世界中存在、学习、推理和行动。鉴于大型语言模型和多模态模型的Recent创新，以及不断进步且响应迅速的硬件，embodied AI系统的能力和操作领域正迅速扩展。这些进展带来了重大的风险，包括恶意使用造成的物理伤害、大规模监视、以及经济和社会的颠覆。然而，这些风险已经被政策制定者严重忽视。现有的政策，比如针对工业机器人的国际标准或自动驾驶汽车的相关法规，不足以应对这些担忧的各个方面。虽然立法者越来越关注AI，但是现在迫切需要扩展和适应现有的框架，以考虑embodied AI的独特风险。为了帮助弥合这一差距，本文做出了三项贡献：首先，我们提供了一个基础性的embodied AI风险分类，涵盖物理、信息、经济和社会等方面的关键风险。其次，我们分析了美国、欧盟和英国的政策，以识别现有框架如何应对这些风险，并指出政策中存在的关键缺口。最后，我们提出了具体的政策建议，以应对即将到来的embodied AI创新浪潮，包括强制性测试和认证、清晰化的责任框架，以及前瞻性的策略来管理并准备应对变革性的经济和社会影响。 

---
# Design and Testing of a Low-Cost 3D-Printed Servo Gimbal for Thrust Vector Control in Model Rockets 

**Title (ZH)**: 低成本3D打印伺服陀螺仪设计与 Thrust Vector Control 在模型火箭中的测试 

**Authors**: Ekansh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2509.00061)  

**Abstract**: Thrust vector control (TVC) is a key mechanism for stabilizing rockets during flight, yet conventional implementations remain costly and technically inaccessible to students and hobbyists. This paper presents the design, fabrication, and testing of a low-cost, 3D-printed, servo-driven two-dimensional gimbal developed for model rocket applications. The gimbal underwent more than 60 CAD iterations, with servo selection guided by torque, response time, and stability requirements. A high-speed camera and Fusion 360 parameter simulations were used to emulate dynamic instability, enabling evaluation of angular deflection, servo responsiveness, and structural durability. The results demonstrated stable actuation within plus or minus 5 degrees, with response times on the average order of 44.5 ms, while limitations included servo fatigue and pin-joint stress under extended loading. The project highlights the feasibility of student-accessible thrust vector control systems and their potential as a reproducible platform for STEM education and experimental aerospace research. 

**Abstract (ZH)**: 低成本3D打印伺服驱动二维陀螺仪设计与测试：面向模型火箭的应用 

---
# Harnessing ADAS for Pedestrian Safety: A Data-Driven Exploration of Fatality Reduction 

**Title (ZH)**: 基于ADAS提升行人安全：一种数据驱动的致命事故减少探索 

**Authors**: Methusela Sulle, Judith Mwakalonge, Gurcan Comert, Saidi Siuhi, Nana Kankam Gyimah  

**Link**: [PDF](https://arxiv.org/pdf/2509.00048)  

**Abstract**: Pedestrian fatalities continue to rise in the United States, driven by factors such as human distraction, increased vehicle size, and complex traffic environments. Advanced Driver Assistance Systems (ADAS) offer a promising avenue for improving pedestrian safety by enhancing driver awareness and vehicle responsiveness. This study conducts a comprehensive data-driven analysis utilizing the Fatality Analysis Reporting System (FARS) to quantify the effectiveness of specific ADAS features like Pedestrian Automatic Emergency Braking (PAEB), Forward Collision Warning (FCW), and Lane Departure Warning (LDW), in lowering pedestrian fatalities. By linking vehicle specifications with crash data, we assess how ADAS performance varies under different environmental and behavioral conditions, such as lighting, weather, and driver/pedestrian distraction. Results indicate that while ADAS can reduce crash severity and prevent some fatalities, its effectiveness is diminished in low-light and adverse weather. The findings highlight the need for enhanced sensor technologies and improved driver education. This research informs policymakers, transportation planners, and automotive manufacturers on optimizing ADAS deployment to improve pedestrian safety and reduce traffic-related deaths. 

**Abstract (ZH)**: 行人死亡率在美国持续上升，受人为分心、车辆尺寸增加及复杂交通环境等因素驱动。高级驾驶辅助系统（ADAS）通过增强驾驶员意识和车辆反应性，为提高行人安全提供了前景广阔的方法。本研究利用致命事故报告系统（FARS）进行全面的数据驱动分析，量化行人自动紧急制动（PAEB）、前方碰撞预警（FCW）和车道偏离预警（LDW）等特定ADAS功能在降低行人死亡率方面的有效性。通过将车辆规格与碰撞数据相链接，我们评估了在不同环境和行为条件下（如光照、天气和驾驶员/行人的分心）ADAS的性能差异。研究结果表明，虽然ADAS可以减轻碰撞严重程度并预防某些死亡事件，但在低光照和不利天气条件下其有效性减弱。研究结果强调了增强传感器技术和改善驾驶员教育的必要性。本研究为政策制定者、交通规划者和汽车制造商提供了优化ADAS部署以提高行人安全和减少交通相关死亡的信息。 

---
# Curve-based slicer for multi-axis DLP 3D printing 

**Title (ZH)**: 基于曲线的多轴DLP 3D打印切割器 

**Authors**: Chengkai Dai, Tao Liu, Dezhao Guo, Binzhi Sun, Guoxin Fang, Yeung Yam, Charlie C.L. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00040)  

**Abstract**: This paper introduces a novel curve-based slicing method for generating planar layers with dynamically varying orientations in digital light processing (DLP) 3D printing. Our approach effectively addresses key challenges in DLP printing, such as regions with large overhangs and staircase artifacts, while preserving its intrinsic advantages of high resolution and fast printing speeds. We formulate the slicing problem as an optimization task, in which parametric curves are computed to define both the slicing layers and the model partitioning through their tangent planes. These curves inherently define motion trajectories for the build platform and can be optimized to meet critical manufacturing objectives, including collision-free motion and floating-free deposition. We validate our method through physical experiments on a robotic multi-axis DLP printing setup, demonstrating that the optimized curves can robustly guide smooth, high-quality fabrication of complex geometries. 

**Abstract (ZH)**: 本文介绍了一种基于曲线的切片方法，用于在数字光处理（DLP）三维打印中生成具有动态变化方向的平面层。该方法有效解决了DLP打印中的关键挑战，如大面积悬臂结构和阶梯状缺陷，同时保持其固有的高分辨率和快速打印速度的优势。我们将切片问题形式化为优化任务，通过计算参数化曲线来定义切片层和模型分区的切平面。这些曲线天然定义了构建平台的运动轨迹，并可优化以满足无碰撞运动和无浮置沉积等关键制造目标。我们在一台机械多轴DLP打印设备上进行物理实验，验证了该方法，结果显示优化后的曲线能够稳健地引导复杂几何结构的高质量、平滑制造。 

---
