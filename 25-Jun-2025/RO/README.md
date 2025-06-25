# ManiGaussian++: General Robotic Bimanual Manipulation with Hierarchical Gaussian World Model 

**Title (ZH)**: ManiGaussian++: 通用的基于分层高斯世界模型的双臂 manipulation 技术 

**Authors**: Tengbo Yu, Guanxing Lu, Zaijia Yang, Haoyuan Deng, Season Si Chen, Jiwen Lu, Wenbo Ding, Guoqiang Hu, Yansong Tang, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19842)  

**Abstract**: Multi-task robotic bimanual manipulation is becoming increasingly popular as it enables sophisticated tasks that require diverse dual-arm collaboration patterns. Compared to unimanual manipulation, bimanual tasks pose challenges to understanding the multi-body spatiotemporal dynamics. An existing method ManiGaussian pioneers encoding the spatiotemporal dynamics into the visual representation via Gaussian world model for single-arm settings, which ignores the interaction of multiple embodiments for dual-arm systems with significant performance drop. In this paper, we propose ManiGaussian++, an extension of ManiGaussian framework that improves multi-task bimanual manipulation by digesting multi-body scene dynamics through a hierarchical Gaussian world model. To be specific, we first generate task-oriented Gaussian Splatting from intermediate visual features, which aims to differentiate acting and stabilizing arms for multi-body spatiotemporal dynamics modeling. We then build a hierarchical Gaussian world model with the leader-follower architecture, where the multi-body spatiotemporal dynamics is mined for intermediate visual representation via future scene prediction. The leader predicts Gaussian Splatting deformation caused by motions of the stabilizing arm, through which the follower generates the physical consequences resulted from the movement of the acting arm. As a result, our method significantly outperforms the current state-of-the-art bimanual manipulation techniques by an improvement of 20.2% in 10 simulated tasks, and achieves 60% success rate on average in 9 challenging real-world tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 基于多任务双臂 manipulation 的多体时空动态建模方法进展：ManiGaussian++ 

---
# Look to Locate: Vision-Based Multisensory Navigation with 3-D Digital Maps for GNSS-Challenged Environments 

**Title (ZH)**: 面向定位的基于视觉的多感官导航：针对GNSS受限制环境的三维数字地图导航 

**Authors**: Ola Elmaghraby, Eslam Mounier, Paulo Ricardo Marques de Araujo, Aboelmagd Noureldin  

**Link**: [PDF](https://arxiv.org/pdf/2506.19827)  

**Abstract**: In Global Navigation Satellite System (GNSS)-denied environments such as indoor parking structures or dense urban canyons, achieving accurate and robust vehicle positioning remains a significant challenge. This paper proposes a cost-effective, vision-based multi-sensor navigation system that integrates monocular depth estimation, semantic filtering, and visual map registration (VMR) with 3-D digital maps. Extensive testing in real-world indoor and outdoor driving scenarios demonstrates the effectiveness of the proposed system, achieving sub-meter accuracy of 92% indoors and more than 80% outdoors, with consistent horizontal positioning and heading average root mean-square errors of approximately 0.98 m and 1.25 °, respectively. Compared to the baselines examined, the proposed solution significantly reduced drift and improved robustness under various conditions, achieving positioning accuracy improvements of approximately 88% on average. This work highlights the potential of cost-effective monocular vision systems combined with 3D maps for scalable, GNSS-independent navigation in land vehicles. 

**Abstract (ZH)**: 在全球导航卫星系统（GNSS）受限环境中，如室内停车场或密集的城市峡谷，实现准确可靠的车辆定位仍然是一项重大挑战。本文提出了一种经济有效的基于视觉的多传感器导航系统，该系统结合了单目深度估计、语义过滤和视觉地图注册（VMR）以及3D数字地图。在实际的室内外驾驶场景中的广泛测试显示，该系统具有良好的效果，室内实现了92%以上的亚米级准确度，室外超过80%，水平定位和航向的一致均方根误差分别为约0.98米和1.25°。与基准方法相比，在各种条件下，提出的解决方案显著减少了漂移并提高了鲁棒性，平均定位精度提高了约88%。这项工作突显了低成本单目视觉系统与3D地图结合在地面车辆中实现可扩展且GNSS独立导航的潜在能力。 

---
# CronusVLA: Transferring Latent Motion Across Time for Multi-Frame Prediction in Manipulation 

**Title (ZH)**: CronusVLA：跨时间转移潜在运动以进行多帧预测的操作控制 

**Authors**: Hao Li, Shuai Yang, Yilun Chen, Yang Tian, Xiaoda Yang, Xinyi Chen, Hanqing Wang, Tai Wang, Feng Zhao, Dahua Lin, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19816)  

**Abstract**: Recent vision-language-action (VLA) models built on pretrained vision-language models (VLMs) have demonstrated strong generalization across manipulation tasks. However, they remain constrained by a single-frame observation paradigm and cannot fully benefit from the motion information offered by aggregated multi-frame historical observations, as the large vision-language backbone introduces substantial computational cost and inference latency. We propose CronusVLA, a unified framework that extends single-frame VLA models to the multi-frame paradigm through an efficient post-training stage. CronusVLA comprises three key components: (1) single-frame pretraining on large-scale embodied datasets with autoregressive action tokens prediction, which establishes an embodied vision-language foundation; (2) multi-frame encoding, adapting the prediction of vision-language backbones from discrete action tokens to motion features during post-training, and aggregating motion features from historical frames into a feature chunking; (3) cross-frame decoding, which maps the feature chunking to accurate actions via a shared decoder with cross-attention. By reducing redundant token computation and caching past motion features, CronusVLA achieves efficient inference. As an application of motion features, we further propose an action adaptation mechanism based on feature-action retrieval to improve model performance during finetuning. CronusVLA achieves state-of-the-art performance on SimplerEnv with 70.9% success rate, and 12.7% improvement over OpenVLA on LIBERO. Real-world Franka experiments also show the strong performance and robustness. 

**Abstract (ZH)**: Recent Vision-Language-Action (VLA) Models Built on Pretrained Vision-Language Models (VLMs) Have Demonstrated Strong Generalization Across Manipulation Tasks. However, They Remain Constrained by a Single-Frame Observation Paradigm and Cannot Fully Benefit from the Motion Information Offered by Aggregated Multi-Frame Historical Observations Due to the Substantial Computational Cost and Inference Latency Introduced by the Large Vision-Language Backbone. We Propose CronusVLA, a Unified Framework That Extends Single-Frame VLA Models to the Multi-Frame Paradigm Through an Efficient Post-Training Stage. 

---
# ReactEMG: Zero-Shot, Low-Latency Intent Detection via sEMG 

**Title (ZH)**: ReactEMG: 零样本、低延迟意图检测基于sEMG 

**Authors**: Runsheng Wang, Xinyue Zhu, Ava Chen, Jingxi Xu, Lauren Winterbottom, Dawn M. Nilsen, Joel Stein, Matei Ciocarlie  

**Link**: [PDF](https://arxiv.org/pdf/2506.19815)  

**Abstract**: Surface electromyography (sEMG) signals show promise for effective human-computer interfaces, particularly in rehabilitation and prosthetics. However, challenges remain in developing systems that respond quickly and reliably to user intent, across different subjects and without requiring time-consuming calibration. In this work, we propose a framework for EMG-based intent detection that addresses these challenges. Unlike traditional gesture recognition models that wait until a gesture is completed before classifying it, our approach uses a segmentation strategy to assign intent labels at every timestep as the gesture unfolds. We introduce a novel masked modeling strategy that aligns muscle activations with their corresponding user intents, enabling rapid onset detection and stable tracking of ongoing gestures. In evaluations against baseline methods, considering both accuracy and stability for device control, our approach surpasses state-of-the-art performance in zero-shot transfer conditions, demonstrating its potential for wearable robotics and next-generation prosthetic systems. Our project page is available at: this https URL 

**Abstract (ZH)**: 表面肌电图(sEMG)信号在人机接口中显示出有效的前景，尤其是在康复和假肢领域。然而，在开发能快速可靠地响应用户意图的系统方面仍面临挑战，特别是在不同受试者之间无需长时间校准的情况下。在这项工作中，我们提出了一种基于肌电图的意图检测框架，以应对这些挑战。与传统的手势识别模型在手势完成后再进行分类不同，我们的方法使用分割策略，在手势展开的每个时间步骤都分配意图标签。我们引入了一种新颖的掩码建模策略，将肌肉激活与相应的用户意图对齐，从而实现快速起始检测并稳定追踪正在进行的手势。在基线方法的评估中，考虑设备控制的准确性和稳定性，我们的方法在零样本迁移条件下超越了现有最佳性能，展示了其在可穿戴机器人和下一代假肢系统中的潜力。更多详情请参见我们的项目页面：this https URL 

---
# The Starlink Robot: A Platform and Dataset for Mobile Satellite Communication 

**Title (ZH)**: Starlink机器人平台及移动卫星通信数据集 

**Authors**: Boyi Liu, Qianyi Zhang, Qiang Yang, Jianhao Jiao, Jagmohan Chauhan, Dimitrios Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2506.19781)  

**Abstract**: The integration of satellite communication into mobile devices represents a paradigm shift in connectivity, yet the performance characteristics under motion and environmental occlusion remain poorly understood. We present the Starlink Robot, the first mobile robotic platform equipped with Starlink satellite internet, comprehensive sensor suite including upward-facing camera, LiDAR, and IMU, designed to systematically study satellite communication performance during movement. Our multi-modal dataset captures synchronized communication metrics, motion dynamics, sky visibility, and 3D environmental context across diverse scenarios including steady-state motion, variable speeds, and different occlusion conditions. This platform and dataset enable researchers to develop motion-aware communication protocols, predict connectivity disruptions, and optimize satellite communication for emerging mobile applications from smartphones to autonomous vehicles. The project is available at this https URL. 

**Abstract (ZH)**: Satellite Communication Integration into Mobile Devices: Performance Study Using the Starlink Robot Platform and Dataset 

---
# Estimating Spatially-Dependent GPS Errors Using a Swarm of Robots 

**Title (ZH)**: 使用机器人群估计空间相关的GPS误差 

**Authors**: Praneeth Somisetty, Robert Griffin, Victor M. Baez, Miguel F. Arevalo-Castiblanco, Aaron T. Becker, Jason M. O'Kane  

**Link**: [PDF](https://arxiv.org/pdf/2506.19712)  

**Abstract**: External factors, including urban canyons and adversarial interference, can lead to Global Positioning System (GPS) inaccuracies that vary as a function of the position in the environment. This study addresses the challenge of estimating a static, spatially-varying error function using a team of robots. We introduce a State Bias Estimation Algorithm (SBE) whose purpose is to estimate the GPS biases. The central idea is to use sensed estimates of the range and bearing to the other robots in the team to estimate changes in bias across the environment. A set of drones moves in a 2D environment, each sampling data from GPS, range, and bearing sensors. The biases calculated by the SBE at estimated positions are used to train a Gaussian Process Regression (GPR) model. We use a Sparse Gaussian process-based Informative Path Planning (IPP) algorithm that identifies high-value regions of the environment for data collection. The swarm plans paths that maximize information gain in each iteration, further refining their understanding of the environment's positional bias landscape. We evaluated SBE and IPP in simulation and compared the IPP methodology to an open-loop strategy. 

**Abstract (ZH)**: 外部因素，包括城市峡谷效应和对抗性干扰，会导致全球定位系统（GPS）的不准确性，这种不准确性随着环境位置的变化而变化。本研究旨在利用机器人团队解决估计静态、空间变化误差函数的挑战。我们介绍了一种状态偏差估计算法（SBE），其目的是估计GPS偏差。核心思想是利用团队中其他机器人到自身的距离和方位角感应估计值，来估计环境中偏差的变化。一组无人机在二维环境中移动，每个无人机从GPS、距离和方位传感器采集数据。由SBE在估计位置计算的偏差用于训练高斯过程回归（GPR）模型。我们使用基于稀疏高斯过程的信息路径规划（IPP）算法来识别环境中的高价值区域以采集数据。根据每次迭代中获取的最大信息增益规划路径，进一步精确定义环境位置偏差景观。我们通过仿真评估了SBE和IPP，并将IPP方法与开环策略进行了比较。 

---
# UniTac-NV: A Unified Tactile Representation For Non-Vision-Based Tactile Sensors 

**Title (ZH)**: UniTac-NV: 一种基于非视觉触觉传感器的统一触觉表示 

**Authors**: Jian Hou, Xin Zhou, Qihan Yang, Adam J. Spiers  

**Link**: [PDF](https://arxiv.org/pdf/2506.19699)  

**Abstract**: Generalizable algorithms for tactile sensing remain underexplored, primarily due to the diversity of sensor modalities. Recently, many methods for cross-sensor transfer between optical (vision-based) tactile sensors have been investigated, yet little work focus on non-optical tactile sensors. To address this gap, we propose an encoder-decoder architecture to unify tactile data across non-vision-based sensors. By leveraging sensor-specific encoders, the framework creates a latent space that is sensor-agnostic, enabling cross-sensor data transfer with low errors and direct use in downstream applications. We leverage this network to unify tactile data from two commercial tactile sensors: the Xela uSkin uSPa 46 and the Contactile PapillArray. Both were mounted on a UR5e robotic arm, performing force-controlled pressing sequences against distinct object shapes (circular, square, and hexagonal prisms) and two materials (rigid PLA and flexible TPU). Another more complex unseen object was also included to investigate the model's generalization capabilities. We show that alignment in latent space can be implicitly learned from joint autoencoder training with matching contacts collected via different sensors. We further demonstrate the practical utility of our approach through contact geometry estimation, where downstream models trained on one sensor's latent representation can be directly applied to another without retraining. 

**Abstract (ZH)**: 可泛化的触觉感知算法研究：基于非视觉传感器的统一表示与跨传感器转移 

---
# A Verification Methodology for Safety Assurance of Robotic Autonomous Systems 

**Title (ZH)**: 一种用于保障机器人自主系统安全性的验证方法学 

**Authors**: Mustafa Adam, David A. Anisi, Pedro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.19622)  

**Abstract**: Autonomous robots deployed in shared human environments, such as agricultural settings, require rigorous safety assurance to meet both functional reliability and regulatory compliance. These systems must operate in dynamic, unstructured environments, interact safely with humans, and respond effectively to a wide range of potential hazards. This paper presents a verification workflow for the safety assurance of an autonomous agricultural robot, covering the entire development life-cycle, from concept study and design to runtime verification. The outlined methodology begins with a systematic hazard analysis and risk assessment to identify potential risks and derive corresponding safety requirements. A formal model of the safety controller is then developed to capture its behaviour and verify that the controller satisfies the specified safety properties with respect to these requirements. The proposed approach is demonstrated on a field robot operating in an agricultural setting. The results show that the methodology can be effectively used to verify safety-critical properties and facilitate the early identification of design issues, contributing to the development of safer robots and autonomous systems. 

**Abstract (ZH)**: 自主机器人部署在共享人类环境中的农业设置等场景下，需要严格的安全性保证以满足功能性可靠性和监管合规性要求。这些系统必须在动态、非结构化的环境中操作，安全地与人类互动，并有效地应对各种潜在的危害。本文提出了一种自主农业机器人安全保证的验证工作流，涵盖从概念研究和设计到运行时验证的整个开发生命周期。该概述的方法从系统化的危害分析和风险评估开始，以识别潜在风险并推导出相应的安全要求。然后开发形式化的安全控制器模型以捕捉其行为，并验证控制器是否满足这些要求所规定的安全属性。所提出的的方法在一种田间操作的农业机器人中进行了演示。结果表明，该方法可以有效地用于验证安全关键属性，并有助于早期识别设计问题，从而促进更安全的机器人和自主系统的开发。 

---
# Probabilistic modelling and safety assurance of an agriculture robot providing light-treatment 

**Title (ZH)**: 农业机器人提供光照处理的概率建模与安全保证 

**Authors**: Mustafa Adam, Kangfeng Ye, David A. Anisi, Ana Cavalcanti, Jim Woodcock, Robert Morris  

**Link**: [PDF](https://arxiv.org/pdf/2506.19620)  

**Abstract**: Continued adoption of agricultural robots postulates the farmer's trust in the reliability, robustness and safety of the new technology. This motivates our work on safety assurance of agricultural robots, particularly their ability to detect, track and avoid obstacles and humans. This paper considers a probabilistic modelling and risk analysis framework for use in the early development phases. Starting off with hazard identification and a risk assessment matrix, the behaviour of the mobile robot platform, sensor and perception system, and any humans present are captured using three state machines. An auto-generated probabilistic model is then solved and analysed using the probabilistic model checker PRISM. The result provides unique insight into fundamental development and engineering aspects by quantifying the effect of the risk mitigation actions and risk reduction associated with distinct design concepts. These include implications of adopting a higher performance and more expensive Object Detection System or opting for a more elaborate warning system to increase human awareness. Although this paper mainly focuses on the initial concept-development phase, the proposed safety assurance framework can also be used during implementation, and subsequent deployment and operation phases. 

**Abstract (ZH)**: 农业机器人持续采用预示农民对其可靠性和安全性信托，这促使我们关注农业机器人安全性保障，特别是其检测、跟踪和避开障碍物及人类的能力。本文提出一种概率建模和风险管理框架，适用于早期开发阶段。通过危害识别和风险评估矩阵，利用三个状态机捕捉移动机器人平台、传感器和感知系统，以及任何现场人类的行为。然后自动生成的概率模型通过PRISM概率模型检查器求解和分析。结果通过量化风险缓解措施和不同设计概念相关的风险降低，为基本开发和工程方面提供独特的见解。这些包括采用性能更高、成本更高的目标检测系统或选择更复杂的警告系统以提升人类意识的涵义。尽管本文主要集中在初始概念开发阶段，但提出的安全保障框架也可用于实施、后续部署和运行阶段。 

---
# Soft Robotic Delivery of Coiled Anchors for Cardiac Interventions 

**Title (ZH)**: 心脏介入手术用螺旋锚的软体机器人递送技术 

**Authors**: Leonardo Zamora Yanez, Jacob Rogatinsky, Dominic Recco, Sang-Yoep Lee, Grace Matthews, Andrew P. Sabelhaus, Tommaso Ranzani  

**Link**: [PDF](https://arxiv.org/pdf/2506.19602)  

**Abstract**: Trans-catheter cardiac intervention has become an increasingly available option for high-risk patients without the complications of open heart surgery. However, current catheterbased platforms suffer from a lack of dexterity, force application, and compliance required to perform complex intracardiac procedures. An exemplary task that would significantly ease minimally invasive intracardiac procedures is the implantation of anchor coils, which can be used to fix and implant various devices in the beating heart. We introduce a robotic platform capable of delivering anchor coils. We develop a kineto-statics model of the robotic platform and demonstrate low positional error. We leverage the passive compliance and high force output of the actuator in a multi-anchor delivery procedure against a motile in-vitro simulator with millimeter level accuracy. 

**Abstract (ZH)**: 经导管心脏干预已成为一种越来越可供高风险患者选择的选项，且避免了开胸手术的并发症。然而，当前的基于导管的平台在执行复杂的心内操作时缺乏所需的柔顺性、力应用和顺应性。一个能够显著简化微创心内操作的范例任务是植入锚线圈，这些锚线圈可用于在跳动的心脏中定位和植入各种设备。我们介绍了一个能够输送锚线圈的机器人平台。我们建立了该机器人平台的运动静力学模型，并展示了低位置误差。我们利用执行器的被动顺应性和高力输出，针对一个移动的体外模拟器进行了多锚输送操作，实现了毫米级的精确度。 

---
# Robotics Under Construction: Challenges on Job Sites 

**Title (ZH)**: 机器人技术在工地上的挑战 

**Authors**: Haruki Uchiito, Akhilesh Bhat, Koji Kusaka, Xiaoya Zhang, Hiraku Kinjo, Honoka Uehara, Motoki Koyama, Shinji Natsume  

**Link**: [PDF](https://arxiv.org/pdf/2506.19597)  

**Abstract**: As labor shortages and productivity stagnation increasingly challenge the construction industry, automation has become essential for sustainable infrastructure development. This paper presents an autonomous payload transportation system as an initial step toward fully unmanned construction sites. Our system, based on the CD110R-3 crawler carrier, integrates autonomous navigation, fleet management, and GNSS-based localization to facilitate material transport in construction site environments. While the current system does not yet incorporate dynamic environment adaptation algorithms, we have begun fundamental investigations into external-sensor based perception and mapping system. Preliminary results highlight the potential challenges, including navigation in evolving terrain, environmental perception under construction-specific conditions, and sensor placement optimization for improving autonomy and efficiency. Looking forward, we envision a construction ecosystem where collaborative autonomous agents dynamically adapt to site conditions, optimizing workflow and reducing human intervention. This paper provides foundational insights into the future of robotics-driven construction automation and identifies critical areas for further technological development. 

**Abstract (ZH)**: 随着劳动力短缺和生产率停滞日益挑战建筑行业，自动化已成为可持续基础设施发展的 essential。本文介绍了基于CD110R-3履带载体的自主载荷运输系统，作为迈向完全无人施工现场的第一步。该系统集成了自主导航、车队管理和基于GNSS的定位技术，以促进施工现场的材料运输。当前系统尚未包含动态环境适应算法，但我们已经开始对外部传感器感知与建图系统的基础研究。初步结果强调了潜在挑战，包括在动态地形中的导航、在特定施工条件下进行环境感知以及优化传感器布局以提高自主性和效率。展望未来，我们设想一个协作自主代理能够动态适应施工现场条件，优化工作流程并减少人为干预的建筑生态系统。本文为机器人驱动的建筑自动化未来提供了基础洞察，并指出了需要进一步技术发展的关键领域。 

---
# Fake or Real, Can Robots Tell? Evaluating Embodied Vision-Language Models on Real and 3D-Printed Objects 

**Title (ZH)**: 假的还是真的，机器人能分辨吗？基于实体物体的视觉-语言模型评价 

**Authors**: Federico Tavella, Kathryn Mearns, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19579)  

**Abstract**: Robotic scene understanding increasingly relies on vision-language models (VLMs) to generate natural language descriptions of the environment. In this work, we present a comparative study of captioning strategies for tabletop scenes captured by a robotic arm equipped with an RGB camera. The robot collects images of objects from multiple viewpoints, and we evaluate several models that generate scene descriptions. We compare the performance of various captioning models, like BLIP and VLMs. Our experiments examine the trade-offs between single-view and multi-view captioning, and difference between recognising real-world and 3D printed objects. We quantitatively evaluate object identification accuracy, completeness, and naturalness of the generated captions. Results show that VLMs can be used in robotic settings where common objects need to be recognised, but fail to generalise to novel representations. Our findings provide practical insights into deploying foundation models for embodied agents in real-world settings. 

**Abstract (ZH)**: 机器人场景理解越来越多地依赖于视觉-语言模型（VLMs）生成环境的自然语言描述。在本工作中，我们对配备RGB相机的机器人手臂拍摄的 tabletop场景的配图策略进行了比较研究。机器人从多个视角收集物体图像，并评估生成场景描述的各种模型。我们将比较BLIP和VLM等多种配图模型的性能。我们的实验研究了单视角与多视角配图之间的权衡，并探讨了识别真实世界物体与3D打印物体之间的差异。我们从对象识别准确性、完整性和生成描述的自然度方面定量评估结果。结果显示，VLMs可以在需要识别常见物体的机器人环境中使用，但不能泛化到新的表示。我们的研究结果为在现实环境中部署基础模型应用于体现智能体提供了实用见解。 

---
# T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with Vision-Language Models 

**Title (ZH)**: T-Rex：基于视觉-语言模型的任务自适应空间表示提取在机器人操作中的应用 

**Authors**: Yiteng Chen, Wenbo Li, Shiyi Wang, Huiping Zhuang, Qingyao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19498)  

**Abstract**: Building a general robotic manipulation system capable of performing a wide variety of tasks in real-world settings is a challenging task. Vision-Language Models (VLMs) have demonstrated remarkable potential in robotic manipulation tasks, primarily due to the extensive world knowledge they gain from large-scale datasets. In this process, Spatial Representations (such as points representing object positions or vectors representing object orientations) act as a bridge between VLMs and real-world scene, effectively grounding the reasoning abilities of VLMs and applying them to specific task scenarios. However, existing VLM-based robotic approaches often adopt a fixed spatial representation extraction scheme for various tasks, resulting in insufficient representational capability or excessive extraction time. In this work, we introduce T-Rex, a Task-Adaptive Framework for Spatial Representation Extraction, which dynamically selects the most appropriate spatial representation extraction scheme for each entity based on specific task requirements. Our key insight is that task complexity determines the types and granularity of spatial representations, and Stronger representational capabilities are typically associated with Higher overall system operation costs. Through comprehensive experiments in real-world robotic environments, we show that our approach delivers significant advantages in spatial understanding, efficiency, and stability without additional training. 

**Abstract (ZH)**: 构建一种能够执行多种任务的一般机器人操作系统是在实际环境中具有挑战性的任务。基于视觉-语言模型（VLMs）的机器人操作任务表现出显著潜力，主要归因于它们从大规模数据集中获得的广泛世界知识。在这个过程中，空间表示（如表示物体位置的点或表示物体方向的向量）充当了VLMs与实际场景之间的桥梁，有效grounded VLMs的推理能力并应用于特定任务场景中。然而，现有的基于VLM的机器人方法通常采用固定的空间表示提取方案处理多种任务，导致空间表示能力不足或提取时间过长。在本文中，我们提出了T-Rex，一种基于任务的空间表示提取自适应框架，根据特定任务需求动态选择最合适的空间表示提取方案。我们的核心洞察是，任务复杂性决定了空间表示的类型和粒度，更强的空间表示能力通常伴随着更高的系统操作成本。通过在真实世界机器人环境中的全面实验，我们展示了该方法在空间理解、效率和稳定性方面具有显著优势，而无需额外训练。 

---
# Ground-Effect-Aware Modeling and Control for Multicopters 

**Title (ZH)**: 地面效应对多旋翼飞行器的建模与控制 

**Authors**: Tiankai Yang, Kaixin Chai, Jialin Ji, Yuze Wu, Chao Xu, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19424)  

**Abstract**: The ground effect on multicopters introduces several challenges, such as control errors caused by additional lift, oscillations that may occur during near-ground flight due to external torques, and the influence of ground airflow on models such as the rotor drag and the mixing matrix. This article collects and analyzes the dynamics data of near-ground multicopter flight through various methods, including force measurement platforms and real-world flights. For the first time, we summarize the mathematical model of the external torque of multicopters under ground effect. The influence of ground airflow on rotor drag and the mixing matrix is also verified through adequate experimentation and analysis. Through simplification and derivation, the differential flatness of the multicopter's dynamic model under ground effect is confirmed. To mitigate the influence of these disturbance models on control, we propose a control method that combines dynamic inverse and disturbance models, ensuring consistent control effectiveness at both high and low altitudes. In this method, the additional thrust and variations in rotor drag under ground effect are both considered and compensated through feedforward models. The leveling torque of ground effect can be equivalently represented as variations in the center of gravity and the moment of inertia. In this way, the leveling torque does not explicitly appear in the dynamic model. The final experimental results show that the method proposed in this paper reduces the control error (RMSE) by \textbf{45.3\%}. Please check the supplementary material at: this https URL. 

**Abstract (ZH)**: 地面效应对多旋翼飞行器的影响及其控制方法研究：通过地面气流对旋翼阻力和混合矩阵的影响的分析与验证 

---
# A Survey on Soft Robot Adaptability: Implementations, Applications, and Prospects 

**Title (ZH)**: 软体机器人适应性综述：实现、应用与前景 

**Authors**: Zixi Chen, Di Wu, Qinghua Guan, David Hardman, Federico Renda, Josie Hughes, Thomas George Thuruthel, Cosimo Della Santina, Barbara Mazzolai, Huichan Zhao, Cesare Stefanini  

**Link**: [PDF](https://arxiv.org/pdf/2506.19397)  

**Abstract**: Soft robots, compared to rigid robots, possess inherent advantages, including higher degrees of freedom, compliance, and enhanced safety, which have contributed to their increasing application across various fields. Among these benefits, adaptability is particularly noteworthy. In this paper, adaptability in soft robots is categorized into external and internal adaptability. External adaptability refers to the robot's ability to adjust, either passively or actively, to variations in environments, object properties, geometries, and task dynamics. Internal adaptability refers to the robot's ability to cope with internal variations, such as manufacturing tolerances or material aging, and to generalize control strategies across different robots. As the field of soft robotics continues to evolve, the significance of adaptability has become increasingly pronounced. In this review, we summarize various approaches to enhancing the adaptability of soft robots, including design, sensing, and control strategies. Additionally, we assess the impact of adaptability on applications such as surgery, wearable devices, locomotion, and manipulation. We also discuss the limitations of soft robotics adaptability and prospective directions for future research. By analyzing adaptability through the lenses of implementation, application, and challenges, this paper aims to provide a comprehensive understanding of this essential characteristic in soft robotics and its implications for diverse applications. 

**Abstract (ZH)**: 软体机器人相较于刚性机器人具有更高的自由度、顺应性和增强的安全性等固有优势，这些优势使其在多个领域中的应用日益增多。在这其中，适应性尤为重要。本文将软体机器人的适应性分为外部适应性和内部适应性。外部适应性指的是机器人能够被动或主动地适应环境变化、对象性质、几何形态和任务动态的变化。内部适应性指的是机器人能够应对制造公差或材料老化等内部变化，并在不同机器人上泛化控制策略的能力。随着软体机器人领域的不断发展，适应性的重要性日益凸显。在本文的综述中，我们总结了提高软体机器人适应性的各种方法，包括设计、传感和控制策略。此外，我们还评估了适应性对诸如手术、可穿戴设备、运动和操纵等领域应用的影响。我们还讨论了软体机器人适应性的局限性及未来研究的潜在方向。通过从实现、应用和挑战三个视角分析适应性，本文旨在提供对这一软体机器人关键特性及其对多种应用的潜在影响的全面理解。 

---
# Zero-Shot Parameter Learning of Robot Dynamics Using Bayesian Statistics and Prior Knowledge 

**Title (ZH)**: 基于贝叶斯统计和先验知识的机器人动力学零样本参数学习 

**Authors**: Carsten Reiners, Minh Trinh, Lukas Gründel, Sven Tauchmann, David Bitterolf, Oliver Petrovic, Christian Brecher  

**Link**: [PDF](https://arxiv.org/pdf/2506.19350)  

**Abstract**: Inertial parameter identification of industrial robots is an established process, but standard methods using Least Squares or Machine Learning do not consider prior information about the robot and require extensive measurements. Inspired by Bayesian statistics, this paper presents an identification method with improved generalization that incorporates prior knowledge and is able to learn with only a few or without additional measurements (Zero-Shot Learning). Furthermore, our method is able to correctly learn not only the inertial but also the mechanical and base parameters of the MABI Max 100 robot while ensuring physical feasibility and specifying the confidence intervals of the results. We also provide different types of priors for serial robots with 6 degrees of freedom, where datasheets or CAD models are not available. 

**Abstract (ZH)**: 基于贝叶斯统计的工业机器人惯性参数识别方法：融合先验知识的零样本学习 

---
# Robotic Perception with a Large Tactile-Vision-Language Model for Physical Property Inference 

**Title (ZH)**: 基于大型触觉-视觉-语言模型的机器人感知与物理属性推理 

**Authors**: Zexiang Guo, Hengxiang Chen, Xinheng Mai, Qiusang Qiu, Gan Ma, Zhanat Kappassov, Qiang Li, Nutan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19303)  

**Abstract**: Inferring physical properties can significantly enhance robotic manipulation by enabling robots to handle objects safely and efficiently through adaptive grasping strategies. Previous approaches have typically relied on either tactile or visual data, limiting their ability to fully capture properties. We introduce a novel cross-modal perception framework that integrates visual observations with tactile representations within a multimodal vision-language model. Our physical reasoning framework, which employs a hierarchical feature alignment mechanism and a refined prompting strategy, enables our model to make property-specific predictions that strongly correlate with ground-truth measurements. Evaluated on 35 diverse objects, our approach outperforms existing baselines and demonstrates strong zero-shot generalization. Keywords: tactile perception, visual-tactile fusion, physical property inference, multimodal integration, robot perception 

**Abstract (ZH)**: 物理属性推理可以显著增强机器人操作，通过适应性抓取策略使机器人能够安全高效地处理物体。以往的方法通常依赖于触觉或视觉数据，限制了其全面捕获属性的能力。我们提出了一种新的跨模态感知框架，将视觉观察与触觉表征结合在多模态视觉-语言模型中。我们的物理推理框架采用分层特征对齐机制和精细的提示策略，使我们的模型能够做出与真实测量高度相关的属性特定预测。在35种不同物体的评估中，我们的方法优于现有基线，并显示出强大的零样本通用性。关键词：触觉感知、视觉-触觉融合、物理属性推理、多模态集成、机器人感知。 

---
# Ontology Neural Network and ORTSF: A Framework for Topological Reasoning and Delay-Robust Control 

**Title (ZH)**: 本体神经网络与ORTSF：拓扑推理与延迟鲁棒控制的框架 

**Authors**: Jaehong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.19277)  

**Abstract**: The advancement of autonomous robotic systems has led to impressive capabilities in perception, localization, mapping, and control. Yet, a fundamental gap remains: existing frameworks excel at geometric reasoning and dynamic stability but fall short in representing and preserving relational semantics, contextual reasoning, and cognitive transparency essential for collaboration in dynamic, human-centric environments. This paper introduces a unified architecture comprising the Ontology Neural Network (ONN) and the Ontological Real-Time Semantic Fabric (ORTSF) to address this gap. The ONN formalizes relational semantic reasoning as a dynamic topological process. By embedding Forman-Ricci curvature, persistent homology, and semantic tensor structures within a unified loss formulation, ONN ensures that relational integrity and topological coherence are preserved as scenes evolve over time. The ORTSF transforms reasoning traces into actionable control commands while compensating for system delays. It integrates predictive and delay-aware operators that ensure phase margin preservation and continuity of control signals, even under significant latency conditions. Empirical studies demonstrate the ONN + ORTSF framework's ability to unify semantic cognition and robust control, providing a mathematically principled and practically viable solution for cognitive robotics. 

**Abstract (ZH)**: 自主robotic系统的发展已经在感知、定位、制图和控制方面取得了令人印象深刻的成果。然而，仍存在一个根本性的差距：现有的框架在几何推理和动态稳定性方面表现出色，但在表示和保留对动态、以人为本的环境中协作至关重要的关系语义、上下文推理和认知透明性方面却有所不足。本文介绍了一种统一架构，包括本体神经网络（ONN）和本体实时语义 Fabric（ORTSF），以填补这一空白。ONN 将关系语义推理形式化为动态拓扑过程。通过在统一的损失公式中嵌入 Forman-Ricci 曲率、持久同调和语义张量结构，ONN 确保随场景随时间演变，关系完整性和拓扑一致性得以保持。ORTSF 将推理轨迹转化为可执行的控制命令，同时补偿系统时延。它集成了预测性和时延感知操作符，确保相位裕量的保留和控制信号的连续性，即使在显著时延条件下也是如此。实证研究表明，ONN + ORTSF 框架能够统一语义认知和鲁棒控制，提供了一个数学原理上和实践上可行的认知机器人解决方案。 

---
# AnchorDP3: 3D Affordance Guided Sparse Diffusion Policy for Robotic Manipulation 

**Title (ZH)**: AnchorDP3：基于3D可操作性引导的稀疏扩散策略用于机器人 manipulation 

**Authors**: Ziyan Zhao, Ke Fan, He-Yang Xu, Ning Qiao, Bo Peng, Wenlong Gao, Dongjiang Li, Hui Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19269)  

**Abstract**: We present AnchorDP3, a diffusion policy framework for dual-arm robotic manipulation that achieves state-of-the-art performance in highly randomized environments. AnchorDP3 integrates three key innovations: (1) Simulator-Supervised Semantic Segmentation, using rendered ground truth to explicitly segment task-critical objects within the point cloud, which provides strong affordance priors; (2) Task-Conditioned Feature Encoders, lightweight modules processing augmented point clouds per task, enabling efficient multi-task learning through a shared diffusion-based action expert; (3) Affordance-Anchored Keypose Diffusion with Full State Supervision, replacing dense trajectory prediction with sparse, geometrically meaningful action anchors, i.e., keyposes such as pre-grasp pose, grasp pose directly anchored to affordances, drastically simplifying the prediction space; the action expert is forced to predict both robot joint angles and end-effector poses simultaneously, which exploits geometric consistency to accelerate convergence and boost accuracy. Trained on large-scale, procedurally generated simulation data, AnchorDP3 achieves a 98.7% average success rate in the RoboTwin benchmark across diverse tasks under extreme randomization of objects, clutter, table height, lighting, and backgrounds. This framework, when integrated with the RoboTwin real-to-sim pipeline, has the potential to enable fully autonomous generation of deployable visuomotor policies from only scene and instruction, totally eliminating human demonstrations from learning manipulation skills. 

**Abstract (ZH)**: AnchorDP3：一种在高随机化环境下的双臂机器人操纵扩散策略框架 

---
# Scaffolding Dexterous Manipulation with Vision-Language Models 

**Title (ZH)**: 基于视觉语言模型的灵巧操作辅助 

**Authors**: Vincent de Bakker, Joey Hejna, Tyler Ga Wei Lum, Onur Celik, Aleksandar Taranovic, Denis Blessing, Gerhard Neumann, Jeannette Bohg, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2506.19212)  

**Abstract**: Dexterous robotic hands are essential for performing complex manipulation tasks, yet remain difficult to train due to the challenges of demonstration collection and high-dimensional control. While reinforcement learning (RL) can alleviate the data bottleneck by generating experience in simulation, it typically relies on carefully designed, task-specific reward functions, which hinder scalability and generalization. Thus, contemporary works in dexterous manipulation have often bootstrapped from reference trajectories. These trajectories specify target hand poses that guide the exploration of RL policies and object poses that enable dense, task-agnostic rewards. However, sourcing suitable trajectories - particularly for dexterous hands - remains a significant challenge. Yet, the precise details in explicit reference trajectories are often unnecessary, as RL ultimately refines the motion. Our key insight is that modern vision-language models (VLMs) already encode the commonsense spatial and semantic knowledge needed to specify tasks and guide exploration effectively. Given a task description (e.g., "open the cabinet") and a visual scene, our method uses an off-the-shelf VLM to first identify task-relevant keypoints (e.g., handles, buttons) and then synthesize 3D trajectories for hand motion and object motion. Subsequently, we train a low-level residual RL policy in simulation to track these coarse trajectories or "scaffolds" with high fidelity. Across a number of simulated tasks involving articulated objects and semantic understanding, we demonstrate that our method is able to learn robust dexterous manipulation policies. Moreover, we showcase that our method transfers to real-world robotic hands without any human demonstrations or handcrafted rewards. 

**Abstract (ZH)**: 灵巧机器人手对手部执行复杂操作任务至关重要，但由于演示数据收集的挑战和高维控制的难度，训练起来仍然很困难。虽然强化学习（RL）可以通过模拟生成经验数据来缓解数据瓶颈，但它通常依赖于精心设计的任务特定奖励函数，这阻碍了其可扩展性和泛化能力。因此，当前的灵巧操作研究往往基于参考轨迹进行。这些轨迹指定目标手部姿态以引导RL策略的探索，并指定物体姿态以提供密集的任务无关奖励。然而，获取合适的轨迹，特别是对于灵巧的手部来说，仍然是一个巨大挑战。尽管如此，明确的参考轨迹中的细节对于RL并不是必要的，因为最终它会优化运动。我们的核心洞察是现代视觉-语言模型（VLMs）已经编码了指定任务和有效引导探索所需的常识空间和语义知识。给出一个任务描述（例如，“打开柜门”）和一个视觉场景，我们的方法首先使用一个现成的VLM识别与任务相关的关键点（例如，把手、按钮），然后合成手部运动和物体运动的三维轨迹。随后，我们在模拟中训练一个低级别的残差RL策略以高保真度跟踪这些粗略的轨迹或“支架”。在涉及 articulated 物体和语义理解的多个模拟任务中，我们证明了我们的方法能够学习稳健的灵巧操作策略。此外，我们展示了我们的方法可以无缝转移到现实世界的机器人手中，无需任何人类演示或人工设计的奖励。 

---
# Preserving Sense of Agency: User Preferences for Robot Autonomy and User Control across Household Tasks 

**Title (ZH)**: 保持主动感: 用户在家庭任务中对机器人自主性和用户控制的偏好 

**Authors**: Claire Yang, Heer Patel, Max Kleiman-Weiner, Maya Cakmak  

**Link**: [PDF](https://arxiv.org/pdf/2506.19202)  

**Abstract**: Roboticists often design with the assumption that assistive robots should be fully autonomous. However, it remains unclear whether users prefer highly autonomous robots, as prior work in assistive robotics suggests otherwise. High robot autonomy can reduce the user's sense of agency, which represents feeling in control of one's environment. How much control do users, in fact, want over the actions of robots used for in-home assistance? We investigate how robot autonomy levels affect users' sense of agency and the autonomy level they prefer in contexts with varying risks. Our study asked participants to rate their sense of agency as robot users across four distinct autonomy levels and ranked their robot preferences with respect to various household tasks. Our findings revealed that participants' sense of agency was primarily influenced by two factors: (1) whether the robot acts autonomously, and (2) whether a third party is involved in the robot's programming or operation. Notably, an end-user programmed robot highly preserved users' sense of agency, even though it acts autonomously. However, in high-risk settings, e.g., preparing a snack for a child with allergies, they preferred robots that prioritized their control significantly more. Additional contextual factors, such as trust in a third party operator, also shaped their preferences. 

**Abstract (ZH)**: 机器人研究人员常常假设辅助机器人应该完全自主。然而，目前尚不清楚用户是否更偏好高度自主的机器人，因为之前关于辅助机器人研究的成果表明并非如此。高度自主的机器人可以减少用户的自主感，即感觉自己能够控制环境。实际上，用户希望对用于家庭辅助的机器人动作拥有多少控制？我们研究了不同自主水平的机器人如何影响用户的自主感，以及在不同风险情境下用户偏好何种自主水平。研究要求参与者在四种不同的自主水平下评估其作为机器人用户时的自主感，并根据不同家务任务对他们的机器人偏好进行排序。研究发现，用户的自主感主要受两个因素影响：（1）机器人是否自主行动；（2）第三方是否参与机器人编程或操作。值得注意的是，即使机器人自主行事，由最终用户编程的机器人也能最大程度地保留用户的自主感。但在高风险情境下，例如为有食物过敏的孩子准备零食时，参与者更倾向于机器人更多地优先考虑他们的控制权。此外，第三方操作者的可信度等因素也影响了他们的偏好。 

---
# The MOTIF Hand: A Robotic Hand for Multimodal Observations with Thermal, Inertial, and Force Sensors 

**Title (ZH)**: MOTIF 手部：一种配备热感、惯性及力敏传感器的多功能观察机器人手 

**Authors**: Hanyang Zhou, Haozhe Lou, Wenhao Liu, Enyu Zhao, Yue Wang, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2506.19201)  

**Abstract**: Advancing dexterous manipulation with multi-fingered robotic hands requires rich sensory capabilities, while existing designs lack onboard thermal and torque sensing. In this work, we propose the MOTIF hand, a novel multimodal and versatile robotic hand that extends the LEAP hand by integrating: (i) dense tactile information across the fingers, (ii) a depth sensor, (iii) a thermal camera, (iv), IMU sensors, and (v) a visual sensor. The MOTIF hand is designed to be relatively low-cost (under 4000 USD) and easily reproducible. We validate our hand design through experiments that leverage its multimodal sensing for two representative tasks. First, we integrate thermal sensing into 3D reconstruction to guide temperature-aware, safe grasping. Second, we show how our hand can distinguish objects with identical appearance but different masses - a capability beyond methods that use vision only. 

**Abstract (ZH)**: 增强多指灵巧 manipulative 能力需要丰富的感知能力，而现有设计缺乏在手内进行热和扭矩感知的能力。本文提出 MOTIF 手，一种通过整合（i）手指上密集的触觉信息，（ii）深度传感器，（iii）热成像相机，（iv）惯性测量单元传感器，以及（v）视觉传感器，扩展 LEAP 手的新颖的多模态和多功能机械手。MOTIF 手旨在较低成本（低于4000美元）且易于复现。我们通过利用其多模态感知进行的实验验证了我们的手设计，完成两项代表性任务。首先，我们将热感知整合到3D重建中，以指导温度感知的安全抓取。其次，我们展示了我们的手如何区分外观相同但质量不同的物体——这超出了仅使用视觉方法的能力。 

---
# Situated Haptic Interaction: Exploring the Role of Context in Affective Perception of Robotic Touch 

**Title (ZH)**: 情境化触觉交互：探索触觉感知中环境作用的研究 

**Authors**: Qiaoqiao Ren, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2506.19179)  

**Abstract**: Affective interaction is not merely about recognizing emotions; it is an embodied, situated process shaped by context and co-created through interaction. In affective computing, the role of haptic feedback within dynamic emotional exchanges remains underexplored. This study investigates how situational emotional cues influence the perception and interpretation of haptic signals given by a robot. In a controlled experiment, 32 participants watched video scenarios in which a robot experienced either positive actions (such as being kissed), negative actions (such as being slapped) or neutral actions. After each video, the robot conveyed its emotional response through haptic communication, delivered via a wearable vibration sleeve worn by the participant. Participants rated the robot's emotional state-its valence (positive or negative) and arousal (intensity)-based on the video, the haptic feedback, and the combination of the two. The study reveals a dynamic interplay between visual context and touch. Participants' interpretation of haptic feedback was strongly shaped by the emotional context of the video, with visual context often overriding the perceived valence of the haptic signal. Negative haptic cues amplified the perceived valence of the interaction, while positive cues softened it. Furthermore, haptics override the participants' perception of arousal of the video. Together, these results offer insights into how situated haptic feedback can enrich affective human-robot interaction, pointing toward more nuanced and embodied approaches to emotional communication with machines. 

**Abstract (ZH)**: 情感交互不仅仅是情绪识别；它是一个受情境影响的身体化、情境性过程，并且是通过互动共同创造的。在情感计算中，动态情绪交流中触觉反馈的作用仍亟待探索。本研究探讨情境性情感提示如何影响参与者对机器人触觉信号的感知和解释。在受控实验中，32名参与者观看了机器人经历正面行为（如接吻）、负面行为（如被打）或中性行为（如握手）的视频场景。之后，机器人通过穿戴在参与者身上的振动袖口传达其情绪反应。参与者根据视频、触觉反馈以及两者结合对机器人的情绪状态（正向或负向的效价以及强度）进行了评估。研究揭示了视觉情境和触觉之间的动态互动。参与者对触觉反馈的解释强烈受到视频中情绪情境的影响，视觉情境通常会抵消参与者感知到的触觉信号的效价。负面的触觉提示会放大交互的效价感知，而正面的提示则会软化它。此外，触觉会超越参与者对视频情绪强度的感知。综上所述，这些结果提供了关于情境性触觉反馈如何丰富情感人机交互的见解，指出了对机器情感交流进行更细致入微和身体化的处理方法。 

---
# CUPID: Curating Data your Robot Loves with Influence Functions 

**Title (ZH)**: CUPID: 精挑细选机器人喜爱的数据——基于影响函数的方法 

**Authors**: Christopher Agia, Rohan Sinha, Jingyun Yang, Rika Antonova, Marco Pavone, Haruki Nishimura, Masha Itkina, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2506.19121)  

**Abstract**: In robot imitation learning, policy performance is tightly coupled with the quality and composition of the demonstration data. Yet, developing a precise understanding of how individual demonstrations contribute to downstream outcomes - such as closed-loop task success or failure - remains a persistent challenge. We propose CUPID, a robot data curation method based on a novel influence function-theoretic formulation for imitation learning policies. Given a set of evaluation rollouts, CUPID estimates the influence of each training demonstration on the policy's expected return. This enables ranking and selection of demonstrations according to their impact on the policy's closed-loop performance. We use CUPID to curate data by 1) filtering out training demonstrations that harm policy performance and 2) subselecting newly collected trajectories that will most improve the policy. Extensive simulated and hardware experiments show that our approach consistently identifies which data drives test-time performance. For example, training with less than 33% of curated data can yield state-of-the-art diffusion policies on the simulated RoboMimic benchmark, with similar gains observed in hardware. Furthermore, hardware experiments show that our method can identify robust strategies under distribution shift, isolate spurious correlations, and even enhance the post-training of generalist robot policies. Additional materials are made available at: this https URL. 

**Abstract (ZH)**: 在机器人模仿学习中，策略性能与示范数据的质量和组成紧密相关。然而，如何精确理解单个示范如何影响下游结果（如闭环任务的成功或失败）仍然是一个持久的挑战。我们提出了CUPID，一种基于新颖的影响函数理论形式化方法的机器人数据整理方法。给定一组评估轨迹，CUPID估计每个训练示范对策略期望回报的影响。这使得可以根据其对策略闭环性能的影响来对示范进行排名和选择。我们使用CUPID进行数据整理，包括1) 过滤掉对策略性能有害的训练示范，2) 选择新收集的轨迹，这些轨迹将最大程度地提高策略。广泛的仿真和硬件实验表明，我们的方法始终能够识别哪些数据驱动测试时的性能。例如，使用不到33%整理后的数据进行训练，可以在仿真RoboMimic基准上获得最先进的扩散策略表现，硬件实验中也观察到类似的改进。此外，硬件实验表明，我们的方法可以识别鲁棒策略，隔离虚假关联，并且甚至可以增强通用机器人策略的后训练。更多材料可在以下链接获取：this https URL。 

---
# Analysis and experiments of the dissipative Twistcar: direction reversal and asymptotic approximations 

**Title (ZH)**: 耗散Twistcar的方向反转及渐近逼近分析与实验 

**Authors**: Rom Levy, Ari Dantus, Zitao Yu, Yizhar Or  

**Link**: [PDF](https://arxiv.org/pdf/2506.19112)  

**Abstract**: Underactuated wheeled vehicles are commonly studied as nonholonomic systems with periodic actuation. Twistcar is a classical example inspired by a riding toy, which has been analyzed using a planar model of a dynamical system with nonholonomic constraints. Most of the previous analyses did not account for energy dissipation due to friction. In this work, we study a theoretical two-link model of the Twistcar while incorporating dissipation due to rolling resistance. We obtain asymptotic expressions for the system's small-amplitude steady-state periodic dynamics, which reveals the possibility of reversing the direction of motion upon varying the geometric and mass properties of the vehicle. Next, we design and construct a robotic prototype of the Twistcar whose center-of-mass position can be shifted by adding and removing a massive block, enabling demonstration of the Twistcar's direction reversal phenomenon. We also conduct parameter fitting for the frictional resistance in order to improve agreement with experiments. 

**Abstract (ZH)**: 欠驱动轮式车辆通常被研究为具有周期性激励的非完整系统。Twistcar 是一个源于一种儿童玩具的经典例子，它已经被用平面动力学系统的非完整约束模型进行分析。之前的大多数分析都没有考虑到摩擦引起的能量损耗。在本文中，我们考虑滚动阻力引起的耗散，研究了一个Twistcar的两环节理论模型，并获得了系统的小振幅稳态周期动力学的渐近表达式，揭示了通过改变车辆的几何和质量属性可以逆转运动方向的可能性。接下来，我们设计并构建了一个能够通过添加和移除重块改变质心位置的Twistcar机器人原型，以便演示Twistcar的运动方向反转现象。我们还进行了摩擦阻力的参数拟合，以提高与实验的吻合度。 

---
# Multimodal Anomaly Detection with a Mixture-of-Experts 

**Title (ZH)**: 多模态专家混合异常检测 

**Authors**: Christoph Willibald, Daniel Sliwowski, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.19077)  

**Abstract**: With a growing number of robots being deployed across diverse applications, robust multimodal anomaly detection becomes increasingly important. In robotic manipulation, failures typically arise from (1) robot-driven anomalies due to an insufficient task model or hardware limitations, and (2) environment-driven anomalies caused by dynamic environmental changes or external interferences. Conventional anomaly detection methods focus either on the first by low-level statistical modeling of proprioceptive signals or the second by deep learning-based visual environment observation, each with different computational and training data requirements. To effectively capture anomalies from both sources, we propose a mixture-of-experts framework that integrates the complementary detection mechanisms with a visual-language model for environment monitoring and a Gaussian-mixture regression-based detector for tracking deviations in interaction forces and robot motions. We introduce a confidence-based fusion mechanism that dynamically selects the most reliable detector for each situation. We evaluate our approach on both household and industrial tasks using two robotic systems, demonstrating a 60% reduction in detection delay while improving frame-wise anomaly detection performance compared to individual detectors. 

**Abstract (ZH)**: 随着部署在多样化应用中的机器人数量增加，鲁棒多模态异常检测变得越来越重要。在机器人操作中，故障通常源自于（1）由不充分的任务模型或硬件限制引起的机器人驱动异常，以及（2）由动态环境变化或外部干扰引起的环境驱动异常。传统的异常检测方法要么通过低级别统计建模 proprioceptive 信号来关注前者，要么通过基于深度学习的视觉环境观察来关注后者，每种方法都有不同的计算和训练数据要求。为了有效捕捉来自两个源的异常，我们提出了一种专家混合框架，该框架将视觉语言模型用于环境监控和基于高斯混合回归的检测器用于跟踪作用力和机器人运动中的偏差相结合，并引入了一种基于置信度的融合机制，以动态选择每种情况下最可靠的检测器。我们在两种机器人系统上对家庭和工业任务进行了评估，结果显示与单一检测器相比，检测延迟降低了60%，同时帧级异常检测性能得到提高。 

---
# Faster Motion Planning via Restarts 

**Title (ZH)**: 基于重启的更快运动规划 

**Authors**: Nancy Amato, Stav Ashur, Sariel Har-Peled%  

**Link**: [PDF](https://arxiv.org/pdf/2506.19016)  

**Abstract**: Randomized methods such as PRM and RRT are widely used in motion planning. However, in some cases, their running-time suffers from inherent instability, leading to ``catastrophic'' performance even for relatively simple instances. We apply stochastic restart techniques, some of them new, for speeding up Las Vegas algorithms, that provide dramatic speedups in practice (a factor of $3$ [or larger] in many cases).
Our experiments demonstrate that the new algorithms have faster runtimes, shorter paths, and greater gains from multi-threading (when compared with straightforward parallel implementation). We prove the optimality of the new variants. Our implementation is open source, available on github, and is easy to deploy and use. 

**Abstract (ZH)**: 随机方法如PRM和RRT在运动规划中广泛应用。然而，在某些情况下，它们的运行时间会受到内在不稳定性的负面影响，导致“灾难性”的性能下降，即使是对相对简单的实例也是如此。我们应用了包括一些新方法在内的随机重启技术，加速了Las Vegas算法，这些技术在实践中提供了显著的速度提升（在许多情况下，速度提升了3倍或更多）。

我们的实验表明，新的算法具有更快的运行时间、更短的路径，并且在多线程处理中具有更大的优势（与直接的并行实现相比）。我们证明了新变体的最优性。我们的实现是开源的，可在GitHub上获取，并且易于部署和使用。 

---
# FORTE: Tactile Force and Slip Sensing on Compliant Fingers for Delicate Manipulation 

**Title (ZH)**: FORTE：具有良好触觉力感知和打滑检测的柔顺手指精细操作技术 

**Authors**: Siqi Shang, Mingyo Seo, Yuke Zhu, Lilly Chin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18960)  

**Abstract**: Handling delicate and fragile objects remains a major challenge for robotic manipulation, especially for rigid parallel grippers. While the simplicity and versatility of parallel grippers have led to widespread adoption, these grippers are limited by their heavy reliance on visual feedback. Tactile sensing and soft robotics can add responsiveness and compliance. However, existing methods typically involve high integration complexity or suffer from slow response times. In this work, we introduce FORTE, a tactile sensing system embedded in compliant gripper fingers. FORTE uses 3D-printed fin-ray grippers with internal air channels to provide low-latency force and slip feedback. FORTE applies just enough force to grasp objects without damaging them, while remaining easy to fabricate and integrate. We find that FORTE can accurately estimate grasping forces from 0-8 N with an average error of 0.2 N, and detect slip events within 100 ms of occurring. We demonstrate FORTE's ability to grasp a wide range of slippery, fragile, and deformable objects. In particular, FORTE grasps fragile objects like raspberries and potato chips with a 98.6% success rate, and achieves 93% accuracy in detecting slip events. These results highlight FORTE's potential as a robust and practical solution for enabling delicate robotic manipulation. Project page: this https URL 

**Abstract (ZH)**: Handling Delicate and Fragile Objects Remains a Major Challenge for Robotic Manipulation, Especially for Rigid Parallel Grippers. Tactile Sensing and Soft Robotics Can Add Responsiveness and Compliance, but Existing Methods Usually Involve High Integration Complexity or Suffer from Slow Response Times. In This Work, We Introduce FORTE, a Tactile Sensing System Embedded in Compliant Gripper Fingers. 

---
# Unified Vision-Language-Action Model 

**Title (ZH)**: 统一的视觉-语言-动作模型 

**Authors**: Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19850)  

**Abstract**: Vision-language-action models (VLAs) have garnered significant attention for their potential in advancing robotic manipulation. However, previous approaches predominantly rely on the general comprehension capabilities of vision-language models (VLMs) to generate action signals, often overlooking the rich temporal and causal structure embedded in visual observations. In this paper, we present UniVLA, a unified and native multimodal VLA model that autoregressively models vision, language, and action signals as discrete token sequences. This formulation enables flexible multimodal tasks learning, particularly from large-scale video data. By incorporating world modeling during post-training, UniVLA captures causal dynamics from videos, facilitating effective transfer to downstream policy learning--especially for long-horizon tasks. Our approach sets new state-of-the-art results across several widely used simulation benchmarks, including CALVIN, LIBERO, and Simplenv-Bridge, significantly surpassing previous methods. For example, UniVLA achieves 95.5% average success rate on LIBERO benchmark, surpassing pi0-FAST's 85.5%. We further demonstrate its broad applicability on real-world ALOHA manipulation and autonomous driving. 

**Abstract (ZH)**: Vision-language-action模型（VLAs）在推动机器人操作方面展现出巨大潜力，但以往的方法主要依赖于视觉语言模型（VLMs）的一般理解能力生成行动信号，往往忽视了视觉观察中丰富的时序和因果结构。本文介绍了UniVLA，一种统一且原生的多模态VLA模型，以自回归方式建模视觉、语言和行动信号的离散标记序列。该建模方法使得从大规模视频数据中学习灵活的多模态任务成为可能。通过在后训练期间引入世界建模，UniVLA从视频中捕捉因果动态，促进下游策略学习的有效转移，尤其是在长时序任务方面。我们的方法在包括CALVIN、LIBERO和Simplenv-Bridge等多个广泛使用的模拟基准测试中设置了新的最佳结果，显著超越了以往的方法。例如，UniVLA在LIBERO基准测试中的平均成功率达到95.5%，超过pi0-FAST的85.5%。我们进一步展示了其在真实世界ALOHA操作和自动驾驶中的广泛适用性。 

---
# Systematic Comparison of Projection Methods for Monocular 3D Human Pose Estimation on Fisheye Images 

**Title (ZH)**: 鱼眼图像中单目3D人体姿态估计的投影方法系统比较 

**Authors**: Stephanie Käs, Sven Peter, Henrik Thillmann, Anton Burenko, David Benjamin Adrian, Dennis Mack, Timm Linder, Bastian Leibe  

**Link**: [PDF](https://arxiv.org/pdf/2506.19747)  

**Abstract**: Fisheye cameras offer robots the ability to capture human movements across a wider field of view (FOV) than standard pinhole cameras, making them particularly useful for applications in human-robot interaction and automotive contexts. However, accurately detecting human poses in fisheye images is challenging due to the curved distortions inherent to fisheye optics. While various methods for undistorting fisheye images have been proposed, their effectiveness and limitations for poses that cover a wide FOV has not been systematically evaluated in the context of absolute human pose estimation from monocular fisheye images. To address this gap, we evaluate the impact of pinhole, equidistant and double sphere camera models, as well as cylindrical projection methods, on 3D human pose estimation accuracy. We find that in close-up scenarios, pinhole projection is inadequate, and the optimal projection method varies with the FOV covered by the human pose. The usage of advanced fisheye models like the double sphere model significantly enhances 3D human pose estimation accuracy. We propose a heuristic for selecting the appropriate projection model based on the detection bounding box to enhance prediction quality. Additionally, we introduce and evaluate on our novel dataset FISHnCHIPS, which features 3D human skeleton annotations in fisheye images, including images from unconventional angles, such as extreme close-ups, ground-mounted cameras, and wide-FOV poses, available at: this https URL 

**Abstract (ZH)**: 鱼眼相机为机器人提供了捕捉人类动作的 ability，其视野远大于标准针孔相机，特别是在人类-机器人交互和汽车领域具有重要作用。然而，由于鱼眼光学固有的曲率失真，准确检测鱼眼图像中的人体姿态具有挑战性。虽然已经提出了多种鱼眼图像去畸变的方法，但在单目鱼眼图像绝对人体姿态估计中的有效性及其局限性尚未系统性评估。为解决这一问题，我们评估了针孔、等距和双球相机模型，以及圆柱投影方法对三维人体姿态估计精度的影响。我们发现，在近距离场景中，针孔投影不足，最佳投影方法随人体姿态覆盖的视野变化而变化。使用如双球模型等高级鱼眼模型显著提高了三维人体姿态估计的精度。我们提出了一种基于检测边界框选择适当投影模型的启发式方法，以提高预测质量。此外，我们引入并评估了我们的新型数据集FISHnCHIPS，该数据集包含鱼眼图像中的人体骨架标注，包括极近距离拍摄、地埋相机拍摄和大视野姿态等非常规角度的图像。地址如下：this https URL。 

---
# ReLink: Computational Circular Design of Planar Linkage Mechanisms Using Available Standard Parts 

**Title (ZH)**: ReLink：使用标准部件进行平面连杆机构的计算圆整设计 

**Authors**: Maxime Escande, Kristina Shea  

**Link**: [PDF](https://arxiv.org/pdf/2506.19657)  

**Abstract**: The Circular Economy framework emphasizes sustainability by reducing resource consumption and waste through the reuse of components and materials. This paper presents ReLink, a computational framework for the circular design of planar linkage mechanisms using available standard parts. Unlike most mechanism design methods, which assume the ability to create custom parts and infinite part availability, ReLink prioritizes the reuse of discrete, standardized components, thus minimizing the need for new parts. The framework consists of two main components: design generation, where a generative design algorithm generates mechanisms from an inventory of available parts, and inverse design, which uses optimization methods to identify designs that match a user-defined trajectory curve. The paper also examines the trade-offs between kinematic performance and CO2 footprint when incorporating new parts. Challenges such as the combinatorial nature of the design problem and the enforcement of valid solutions are addressed. By combining sustainability principles with kinematic synthesis, ReLink lays the groundwork for further research into computational circular design to support the development of systems that integrate reused components into mechanical products. 

**Abstract (ZH)**: 循环经济框架通过重复使用组件和材料来减少资源消耗和废物，强调可持续性。本文介绍了一种计算框架——ReLink，用于使用现有标准部件进行平面连杆机构的循环设计。与大多数假定能够创建自定义部件并且部件无限可用的机构设计方法不同，ReLink 优先考虑重复使用离散的标准部件，从而最大限度地减少对新部件的需求。该框架由两个主要组件组成：设计生成，其中生成设计算法从现有部件库存中生成机构；逆设计，使用优化方法来识别与用户定义轨迹曲线相匹配的设计。本文还探讨了在采用新部件时动能性能与二氧化碳足迹之间的权衡。面对设计问题的组合性质以及有效解决方案的实现等挑战，本文进行了分析。通过将可持续性原则与运动综合相结合，ReLink 为计算循环经济设计的研究奠定了基础，以支持将重复使用部件纳入机械产品的系统开发。 

---
# Adaptive Domain Modeling with Language Models: A Multi-Agent Approach to Task Planning 

**Title (ZH)**: 基于语言模型的自适应领域建模：任务规划的多智能体方法 

**Authors**: Harisankar Babu, Philipp Schillinger, Tamim Asfour  

**Link**: [PDF](https://arxiv.org/pdf/2506.19592)  

**Abstract**: We introduce TAPAS (Task-based Adaptation and Planning using AgentS), a multi-agent framework that integrates Large Language Models (LLMs) with symbolic planning to solve complex tasks without the need for manually defined environment models. TAPAS employs specialized LLM-based agents that collaboratively generate and adapt domain models, initial states, and goal specifications as needed using structured tool-calling mechanisms. Through this tool-based interaction, downstream agents can request modifications from upstream agents, enabling adaptation to novel attributes and constraints without manual domain redefinition. A ReAct (Reason+Act)-style execution agent, coupled with natural language plan translation, bridges the gap between dynamically generated plans and real-world robot capabilities. TAPAS demonstrates strong performance in benchmark planning domains and in the VirtualHome simulated real-world environment. 

**Abstract (ZH)**: 基于任务的代理规划与适应框架TAPAS：语言模型与符号规划的结合 

---
# EvDetMAV: Generalized MAV Detection from Moving Event Cameras 

**Title (ZH)**: EvDetMAV: 从移动事件相机中的一般化MAV检测 

**Authors**: Yin Zhang, Zian Ning, Xiaoyu Zhang, Shiliang Guo, Peidong Liu, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19416)  

**Abstract**: Existing micro aerial vehicle (MAV) detection methods mainly rely on the target's appearance features in RGB images, whose diversity makes it difficult to achieve generalized MAV detection. We notice that different types of MAVs share the same distinctive features in event streams due to their high-speed rotating propellers, which are hard to see in RGB images. This paper studies how to detect different types of MAVs from an event camera by fully exploiting the features of propellers in the original event stream. The proposed method consists of three modules to extract the salient and spatio-temporal features of the propellers while filtering out noise from background objects and camera motion. Since there are no existing event-based MAV datasets, we introduce a novel MAV dataset for the community. This is the first event-based MAV dataset comprising multiple scenarios and different types of MAVs. Without training, our method significantly outperforms state-of-the-art methods and can deal with challenging scenarios, achieving a precision rate of 83.0\% (+30.3\%) and a recall rate of 81.5\% (+36.4\%) on the proposed testing dataset. The dataset and code are available at: this https URL. 

**Abstract (ZH)**: 现有微空中无人机（MAV）检测方法主要依赖RGB图像中的目标外观特征，这些特征的多样性使得实现泛化 MAV 检测变得困难。我们注意到，由于其高速旋转的螺旋桨，不同类型的 MAV 在事件流中具有相同的独特特征，而在 RGB 图像中这些特征难以观察到。本文研究如何通过充分利用事件流中原始螺旋桨特征来检测不同类型 MAV，同时过滤掉背景物体和相机运动的噪声。由于目前不存在基于事件的 MAV 数据集，我们为社区引入了一个新的 MAV 数据集。这是首个包含多种场景和不同类型 MAV 的基于事件的 MAV 数据集。在无需训练的情况下，我们的方法显著优于现有最先进的方法，并能够处理具有挑战性的场景，在提出的测试数据集上实现了 83.0%（+30.3%）的精确率和 81.5%（+36.4%）的召回率。数据集和代码可在以下链接访问：this https URL。 

---
# Is an object-centric representation beneficial for robotic manipulation ? 

**Title (ZH)**: 对象中心表示对机器人操作有利吗？ 

**Authors**: Alexandre Chapin, Emmanuel Dellandrea, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19408)  

**Abstract**: Object-centric representation (OCR) has recently become a subject of interest in the computer vision community for learning a structured representation of images and videos. It has been several times presented as a potential way to improve data-efficiency and generalization capabilities to learn an agent on downstream tasks. However, most existing work only evaluates such models on scene decomposition, without any notion of reasoning over the learned representation. Robotic manipulation tasks generally involve multi-object environments with potential inter-object interaction. We thus argue that they are a very interesting playground to really evaluate the potential of existing object-centric work. To do so, we create several robotic manipulation tasks in simulated environments involving multiple objects (several distractors, the robot, etc.) and a high-level of randomization (object positions, colors, shapes, background, initial positions, etc.). We then evaluate one classical object-centric method across several generalization scenarios and compare its results against several state-of-the-art hollistic representations. Our results exhibit that existing methods are prone to failure in difficult scenarios involving complex scene structures, whereas object-centric methods help overcome these challenges. 

**Abstract (ZH)**: 基于对象的表示（OCR）在计算机视觉领域的图像和视频的结构化表示学习中 recently 成为一个研究热点，并被多次视为提高数据效率和泛化能力、在下游任务中学习代理的潜在方式。然而，现有大多数工作仅在场景分解方面评估了此类模型，而没有对所学习的表示进行推理的概念。机器人操作任务通常涉及具有潜在对象间交互的多对象环境。因此，我们认为它们是真正评估现有基于对象的表示潜力的非常有趣的实验平台。为了实现这一目标，我们创建了多个涉及多个对象（多个干扰物、机器人等）和高随机性（物体位置、颜色、形状、背景、初始位置等）的机器人操作任务，在模拟环境中进行评估。我们随后对一种经典的对象中心表示方法在多个泛化场景下进行了评估，并将其结果与几种最先进的整体表示方法进行了比较。我们的结果表明，现有方法在涉及复杂场景结构的困难场景中容易失败，而基于对象的方法有助于克服这些挑战。 

---
# Da Yu: Towards USV-Based Image Captioning for Waterway Surveillance and Scene Understanding 

**Title (ZH)**: 大禹：基于USV的水道监控与场景理解中的图像captioning研究 

**Authors**: Runwei Guan, Ningwei Ouyang, Tianhao Xu, Shaofeng Liang, Wei Dai, Yafeng Sun, Shang Gao, Songning Lai, Shanliang Yao, Xuming Hu, Ryan Wen Liu, Yutao Yue, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.19288)  

**Abstract**: Automated waterway environment perception is crucial for enabling unmanned surface vessels (USVs) to understand their surroundings and make informed decisions. Most existing waterway perception models primarily focus on instance-level object perception paradigms (e.g., detection, segmentation). However, due to the complexity of waterway environments, current perception datasets and models fail to achieve global semantic understanding of waterways, limiting large-scale monitoring and structured log generation. With the advancement of vision-language models (VLMs), we leverage image captioning to introduce WaterCaption, the first captioning dataset specifically designed for waterway environments. WaterCaption focuses on fine-grained, multi-region long-text descriptions, providing a new research direction for visual geo-understanding and spatial scene cognition. Exactly, it includes 20.2k image-text pair data with 1.8 million vocabulary size. Additionally, we propose Da Yu, an edge-deployable multi-modal large language model for USVs, where we propose a novel vision-to-language projector called Nano Transformer Adaptor (NTA). NTA effectively balances computational efficiency with the capacity for both global and fine-grained local modeling of visual features, thereby significantly enhancing the model's ability to generate long-form textual outputs. Da Yu achieves an optimal balance between performance and efficiency, surpassing state-of-the-art models on WaterCaption and several other captioning benchmarks. 

**Abstract (ZH)**: 自动水道环境感知对于使无人驾驶水面船舶（USVs）理解其周围环境并作出明智决策至关重要。现有的水道感知模型主要集中在实例级对象感知范式（如检测、分割）上。然而，由于水道环境的复杂性，当前的感知数据集和模型未能实现对水道的全局语义理解，限制了大规模监测和结构化日志生成。随着视觉语言模型（VLMs）的发展，我们利用图像标题生成引入了WaterCaption，这是一个专门针对水道环境的数据集。WaterCaption专注于细粒度、多区域的长文本描述，为视觉地理理解和空间场景认知提供了新的研究方向。该数据集包含20200张图像-文本对数据，词汇量达180万。此外，我们提出了一种名为Nano Transformer Adaptor (NTA) 的新颖的视觉到语言投影器，用于USVs的边缘部署多模态大语言模型Da Yu。NTA有效地在计算效率与视觉特征的全局和细粒度局部建模能力之间取得了平衡，从而显著增强了模型生成长文本输出的能力。Da Yu在WaterCaption和多个其他图像标题生成基准测试中均实现了性能与效率的最佳平衡，超越了现有的最先进的模型。 

---
# AirV2X: Unified Air-Ground Vehicle-to-Everything Collaboration 

**Title (ZH)**: AirV2X: 统一的空地车辆到每一切协作 

**Authors**: Xiangbo Gao, Yuheng Wu, Xuewen Luo, Keshu Wu, Xinghao Chen, Yuping Wang, Chenxi Liu, Yang Zhou, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19283)  

**Abstract**: While multi-vehicular collaborative driving demonstrates clear advantages over single-vehicle autonomy, traditional infrastructure-based V2X systems remain constrained by substantial deployment costs and the creation of "uncovered danger zones" in rural and suburban areas. We present AirV2X-Perception, a large-scale dataset that leverages Unmanned Aerial Vehicles (UAVs) as a flexible alternative or complement to fixed Road-Side Units (RSUs). Drones offer unique advantages over ground-based perception: complementary bird's-eye-views that reduce occlusions, dynamic positioning capabilities that enable hovering, patrolling, and escorting navigation rules, and significantly lower deployment costs compared to fixed infrastructure. Our dataset comprises 6.73 hours of drone-assisted driving scenarios across urban, suburban, and rural environments with varied weather and lighting conditions. The AirV2X-Perception dataset facilitates the development and standardized evaluation of Vehicle-to-Drone (V2D) algorithms, addressing a critical gap in the rapidly expanding field of aerial-assisted autonomous driving systems. The dataset and development kits are open-sourced at this https URL. 

**Abstract (ZH)**: AirV2X-Perception：基于无人机的大型数据集用于空中辅助自动驾驶系统中车辆到无人机（V2D）算法的研究与标准化评估 

---
# Low-Cost Infrastructure-Free 3D Relative Localization with Sub-Meter Accuracy in Near Field 

**Title (ZH)**: 低成本无基础设施近场亚米级相对定位三维定位 

**Authors**: Qiangsheng Gao, Ka Ho Cheng, Li Qiu, Zijun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2506.19199)  

**Abstract**: Relative localization in the near-field scenario is critically important for unmanned vehicle (UxV) applications. Although related works addressing 2D relative localization problem have been widely studied for unmanned ground vehicles (UGVs), the problem in 3D scenarios for unmanned aerial vehicles (UAVs) involves more uncertainties and remains to be investigated. Inspired by the phenomenon that animals can achieve swarm behaviors solely based on individual perception of relative information, this study proposes an infrastructure-free 3D relative localization framework that relies exclusively on onboard ultra-wideband (UWB) sensors. Leveraging 2D relative positioning research, we conducted feasibility analysis, system modeling, simulations, performance evaluation, and field tests using UWB sensors. The key contributions of this work include: derivation of the Cramér-Rao lower bound (CRLB) and geometric dilution of precision (GDOP) for near-field scenarios; development of two localization algorithms -- one based on Euclidean distance matrix (EDM) and another employing maximum likelihood estimation (MLE); comprehensive performance comparison and computational complexity analysis against state-of-the-art methods; simulation studies and field experiments; a novel sensor deployment strategy inspired by animal behavior, enabling single-sensor implementation within the proposed framework for UxV applications. The theoretical, simulation, and experimental results demonstrate strong generalizability to other 3D near-field localization tasks, with significant potential for a cost-effective cross-platform UxV collaborative system. 

**Abstract (ZH)**: 无基础设施的基于UWB的无人驾驶空中车辆（UxV）三维相对定位框架 

---
# Correspondence-Free Multiview Point Cloud Registration via Depth-Guided Joint Optimisation 

**Title (ZH)**: 基于深度引导联合优化的无对应点多元视图点云注册 

**Authors**: Yiran Zhou, Yingyu Wang, Shoudong Huang, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18922)  

**Abstract**: Multiview point cloud registration is a fundamental task for constructing globally consistent 3D models. Existing approaches typically rely on feature extraction and data association across multiple point clouds; however, these processes are challenging to obtain global optimal solution in complex environments. In this paper, we introduce a novel correspondence-free multiview point cloud registration method. Specifically, we represent the global map as a depth map and leverage raw depth information to formulate a non-linear least squares optimisation that jointly estimates poses of point clouds and the global map. Unlike traditional feature-based bundle adjustment methods, which rely on explicit feature extraction and data association, our method bypasses these challenges by associating multi-frame point clouds with a global depth map through their corresponding poses. This data association is implicitly incorporated and dynamically refined during the optimisation process. Extensive evaluations on real-world datasets demonstrate that our method outperforms state-of-the-art approaches in accuracy, particularly in challenging environments where feature extraction and data association are difficult. 

**Abstract (ZH)**: 多视图点云注册是构建全局一致的3D模型的基础任务。现有方法通常依赖于多点云间的特征提取和数据关联；然而，在复杂环境中获得全局最优解极具挑战性。本文介绍了一种新型的无需对应关系的多视图点云注册方法。具体地，我们将全局地图表示为深度图，并利用原始深度信息来形式化一个非线性最小二乘优化问题，该问题联合估计点云的姿态和全局地图。与依赖显式特征提取和数据关联的传统基于特征的束调整方法不同，我们的方法通过点云的姿态将多帧点云与全局深度图关联起来，从而避开这些挑战。数据关联在优化过程中隐式纳入并动态优化。在真实数据集上的广泛评估表明，在特征提取和数据关联困难的复杂环境中，我们的方法在精度方面优于现有最先进的方法。 

---
