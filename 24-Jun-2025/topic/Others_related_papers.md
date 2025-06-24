# Improvement on LiDAR-Camera Calibration Using Square Targets 

**Title (ZH)**: LiDAR-相机标定基于方靶标的改进 

**Authors**: Zhongyuan Li, Honggang Gou, Ping Li, Jiaotong Guo, Mao Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.18294)  

**Abstract**: Precise sensor calibration is critical for autonomous vehicles as a prerequisite for perception algorithms to function properly. Rotation error of one degree can translate to position error of meters in target object detection at large distance, leading to improper reaction of the system or even safety related issues. Many methods for multi-sensor calibration have been proposed. However, there are very few work that comprehensively consider the challenges of the calibration procedure when applied to factory manufacturing pipeline or after-sales service scenarios. In this work, we introduce a fully automatic LiDAR-camera extrinsic calibration algorithm based on targets that is fast, easy to deploy and robust to sensor noises such as missing data. The core of the method include: (1) an automatic multi-stage LiDAR board detection pipeline using only geometry information with no specific material requirement; (2) a fast coarse extrinsic parameter search mechanism that is robust to initial extrinsic errors; (3) a direct optimization algorithm that is robust to sensor noises. We validate the effectiveness of our methods through experiments on data captured in real world scenarios. 

**Abstract (ZH)**: 精确传感器标定对于自动驾驶车辆至关重要，是感知算法正常工作的前提。旋转误差一度在远距离目标检测中可能导致米级的位置误差，引发系统不当反应甚至安全问题。许多多传感器标定方法已被提出，但很少有工作全面考虑将标定程序应用于工厂制造管道或售后服务中心场景中的挑战。在本研究中，我们介绍了一种基于目标的全自动化LiDAR-相机外参标定算法，该算法快速、易部署并且对如缺失数据等传感器噪声具有鲁棒性。该方法的核心包括：（1）仅使用几何信息，无需特定材料要求的多阶段LiDAR板自动检测管道；（2）一种对初始外参误差具有鲁棒性的快速粗略外参参数搜索机制；（3）一种对传感器噪声具有鲁棒性的直接优化算法。我们通过在真实场景中捕获的数据进行实验，验证了该方法的有效性。 

---
# ADA-DPM: A Neural Descriptors-based Adaptive Noise Point Filtering Strategy for SLAM 

**Title (ZH)**: ADA-DPM：一种基于神经描述子的自适应噪声点过滤策略用于SLAM 

**Authors**: Yongxin Shao, Binrui Wang, Aihong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18016)  

**Abstract**: LiDAR SLAM has demonstrated significant application value in various fields, including mobile robot navigation and high-precision map construction. However, existing methods often need to make a trade-off between positioning accuracy and system robustness when faced with dynamic object interference, point cloud noise, and unstructured environments. To address this challenge, we propose an adaptive noise filtering SLAM strategy-ADA-DPM, achieving excellent preference in both aspects. We design the Dynamic Segmentation Head to predict the category of feature points belonging to dynamic points, to eliminate dynamic feature points; design the Global Importance Scoring Head to adaptively select feature points with higher contribution and features while suppressing noise interference; and construct the Cross Layer Intra-Graph Convolution Module (GLI-GCN) to fuse multi-scale neighborhood structures, thereby enhancing the discriminative ability of overlapping features. Finally, to further validate the effectiveness of our method, we tested it on several publicly available datasets and achieved outstanding results. 

**Abstract (ZH)**: LiDAR SLAM已在移动机器人导航和高精度地图构建等多个领域展示了显著的应用价值。然而，现有方法在面对动态物体干扰、点云噪声和非结构化环境时，往往需要在定位精度和系统鲁棒性之间做出权衡。为解决这一挑战，我们提出了一种自适应噪声过滤SLAM策略—ADA-DPM，在这两方面均实现了优异的表现。我们设计了动态分割头来预测特征点属于动态点的类别，以消除动态特征点；设计了全局重要性评分头以自适应选择具有更高贡献度和特征的特征点，同时抑制噪声干扰；构建了跨层内图卷积模块（GLI-GCN）以融合多尺度局部结构，从而增强重叠特征的辨别能力。最后，为了进一步验证本方法的有效性，我们在多个公开数据集上进行了测试，并取得了优异的结果。 

---
# Optimizing Exploration with a New Uncertainty Framework for Active SLAM Systems 

**Title (ZH)**: 基于新不确定性框架的主动SLAM系统式探索优化 

**Authors**: Sebastian Sansoni, Javier Gimenez, Gastón Castro, Santiago Tosetti, Flavio Craparo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17775)  

**Abstract**: Accurate reconstruction of the environment is a central goal of Simultaneous Localization and Mapping (SLAM) systems. However, the agent's trajectory can significantly affect estimation accuracy. This paper presents a new method to model map uncertainty in Active SLAM systems using an Uncertainty Map (UM). The UM uses probability distributions to capture where the map is uncertain, allowing Uncertainty Frontiers (UF) to be defined as key exploration-exploitation objectives and potential stopping criteria. In addition, the method introduces the Signed Relative Entropy (SiREn), based on the Kullback-Leibler divergence, to measure both coverage and uncertainty together. This helps balance exploration and exploitation through an easy-to-understand parameter. Unlike methods that depend on particular SLAM setups, the proposed approach is compatible with different types of sensors, such as cameras, LiDARs, and multi-sensor fusion. It also addresses common problems in exploration planning and stopping conditions. Furthermore, integrating this map modeling approach with a UF-based planning system enables the agent to autonomously explore open spaces, a behavior not previously observed in the Active SLAM literature. Code and implementation details are available as a ROS node, and all generated data are openly available for public use, facilitating broader adoption and validation of the proposed approach. 

**Abstract (ZH)**: 基于不确定性图的主动SLAM中环境重建不确定性建模方法 

---
# DiLQR: Differentiable Iterative Linear Quadratic Regulator via Implicit Differentiation 

**Title (ZH)**: 可微迭代线性二次调节器：通过隐式求导实现 

**Authors**: Shuyuan Wang, Philip D. Loewen, Michael Forbes, Bhushan Gopaluni, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17473)  

**Abstract**: While differentiable control has emerged as a powerful paradigm combining model-free flexibility with model-based efficiency, the iterative Linear Quadratic Regulator (iLQR) remains underexplored as a differentiable component. The scalability of differentiating through extended iterations and horizons poses significant challenges, hindering iLQR from being an effective differentiable controller. This paper introduces DiLQR, a framework that facilitates differentiation through iLQR, allowing it to serve as a trainable and differentiable module, either as or within a neural network. A novel aspect of this framework is the analytical solution that it provides for the gradient of an iLQR controller through implicit differentiation, which ensures a constant backward cost regardless of iteration, while producing an accurate gradient. We evaluate our framework on imitation tasks on famous control benchmarks. Our analytical method demonstrates superior computational performance, achieving up to 128x speedup and a minimum of 21x speedup compared to automatic differentiation. Our method also demonstrates superior learning performance ($10^6$x) compared to traditional neural network policies and better model loss with differentiable controllers that lack exact analytical gradients. Furthermore, we integrate our module into a larger network with visual inputs to demonstrate the capacity of our method for high-dimensional, fully end-to-end tasks. Codes can be found on the project homepage this https URL. 

**Abstract (ZH)**: 不同iable DiLQR：一种可微分的迭代线性二次调节器框架 

---
# A workflow for generating synthetic LiDAR datasets in simulation environments 

**Title (ZH)**: 仿真环境生成合成LiDAR数据集的工作流 

**Authors**: Abhishek Phadke, Shakib Mahmud Dipto, Pratip Rana  

**Link**: [PDF](https://arxiv.org/pdf/2506.17378)  

**Abstract**: This paper presents a simulation workflow for generating synthetic LiDAR datasets to support autonomous vehicle perception, robotics research, and sensor security analysis. Leveraging the CoppeliaSim simulation environment and its Python API, we integrate time-of-flight LiDAR, image sensors, and two dimensional scanners onto a simulated vehicle platform operating within an urban scenario. The workflow automates data capture, storage, and annotation across multiple formats (PCD, PLY, CSV), producing synchronized multimodal datasets with ground truth pose information. We validate the pipeline by generating large-scale point clouds and corresponding RGB and depth imagery. The study examines potential security vulnerabilities in LiDAR data, such as adversarial point injection and spoofing attacks, and demonstrates how synthetic datasets can facilitate the evaluation of defense strategies. Finally, limitations related to environmental realism, sensor noise modeling, and computational scalability are discussed, and future research directions, such as incorporating weather effects, real-world terrain models, and advanced scanner configurations, are proposed. The workflow provides a versatile, reproducible framework for generating high-fidelity synthetic LiDAR datasets to advance perception research and strengthen sensor security in autonomous systems. Documentation and examples accompany this framework; samples of animated cloud returns and image sensor data can be found at this Link. 

**Abstract (ZH)**: 本文提出了一个用于生成合成LiDAR数据集的模拟工作流，以支持自主车辆感知、机器人研究和传感器安全分析。利用CoppeliaSim模拟环境及其Python API，我们将飞行时间LiDAR、图像传感器和二维扫描仪集成到一个配备模拟车辆平台的都市场景中。该工作流自动捕获、存储和标注多种格式（PCD、PLY、CSV）的数据，生成同步多模态数据集，并包含真实姿态信息。通过生成大规模点云和相应的RGB及深度图像，验证了该管道。研究探讨了LiDAR数据中的潜在安全漏洞，如对抗性点注入和欺骗攻击，并展示了合成数据集如何促进防御策略的评估。最后，讨论了环境现实性、传感器噪声建模和计算扩展性的相关限制，并提出了结合天气效应、现实地形模型和高级扫描器配置等未来研究方向。该工作流提供了一个多功能且可复现的框架，用于生成高保真合成LiDAR数据集，促进感知研究并增强自主系统中的传感器安全性。框架包括文档和示例；动画点云返回和图像传感器数据样本可在该链接处找到。 

---
# MCN-SLAM: Multi-Agent Collaborative Neural SLAM with Hybrid Implicit Neural Scene Representation 

**Title (ZH)**: MCN-SLAM: 多agent协作神经SLAM与混合隐式神经场景表示 

**Authors**: Tianchen Deng, Guole Shen, Xun Chen, Shenghai Yuan, Hongming Shen, Guohao Peng, Zhenyu Wu, Jingchuan Wang, Lihua Xie, Danwei Wang, Hesheng Wang, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18678)  

**Abstract**: Neural implicit scene representations have recently shown promising results in dense visual SLAM. However, existing implicit SLAM algorithms are constrained to single-agent scenarios, and fall difficulties in large-scale scenes and long sequences. Existing NeRF-based multi-agent SLAM frameworks cannot meet the constraints of communication bandwidth. To this end, we propose the first distributed multi-agent collaborative neural SLAM framework with hybrid scene representation, distributed camera tracking, intra-to-inter loop closure, and online distillation for multiple submap fusion. A novel triplane-grid joint scene representation method is proposed to improve scene reconstruction. A novel intra-to-inter loop closure method is designed to achieve local (single-agent) and global (multi-agent) consistency. We also design a novel online distillation method to fuse the information of different submaps to achieve global consistency. Furthermore, to the best of our knowledge, there is no real-world dataset for NeRF-based/GS-based SLAM that provides both continuous-time trajectories groundtruth and high-accuracy 3D meshes groundtruth. To this end, we propose the first real-world Dense slam (DES) dataset covering both single-agent and multi-agent scenarios, ranging from small rooms to large-scale outdoor scenes, with high-accuracy ground truth for both 3D mesh and continuous-time camera trajectory. This dataset can advance the development of the research in both SLAM, 3D reconstruction, and visual foundation model. Experiments on various datasets demonstrate the superiority of the proposed method in both mapping, tracking, and communication. The dataset and code will open-source on this https URL. 

**Abstract (ZH)**: 分布式多智能体协作神经SLAM框架：混合场景表示、分布式相机追踪、局部到全局回环闭合及在线蒸馏 

---
# Crowdsourcing Ubiquitous Indoor Localization with Non-Cooperative Wi-Fi Ranging 

**Title (ZH)**: 基于非协作Wi-Fi测距的众包室内广泛定位 

**Authors**: Emerson Sie, Enguang Fan, Federico Cifuentes-Urtubey, Deepak Vasisht  

**Link**: [PDF](https://arxiv.org/pdf/2506.18317)  

**Abstract**: Indoor localization opens the path to potentially transformative applications. Although many indoor localization methods have been proposed over the years, they remain too impractical for widespread deployment in the real world. In this paper, we introduce PeepLoc, a deployable and scalable Wi-Fi-based solution for indoor localization that relies only on pre-existing devices and infrastructure. Specifically, PeepLoc works on any mobile device with an unmodified Wi-Fi transceiver and in any indoor environment with a sufficient number of Wi-Fi access points (APs) and pedestrian traffic. At the core of PeepLoc is (a) a mechanism which allows any Wi-Fi device to obtain non-cooperative time-of-flight (ToF) to any Wi-Fi AP and (b) a novel bootstrapping mechanism that relies on pedestrian dead reckoning (PDR) and crowdsourcing to opportunistically initialize pre-existing APs as anchor points within an environment. We implement PeepLoc using commodity hardware and evaluate it extensively across 4 campus buildings. We show PeepLoc leads to a mean and median positional error of 3.41 m and 3.06 m respectively, which is superior to existing deployed indoor localization systems and is competitive with commodity GPS in outdoor environments. 

**Abstract (ZH)**: 基于Wi-Fi的室内定位技术PeepLoc：可部署且可扩展的解决方案 

---
# Leveraging Cloud-Fog Automation for Autonomous Collision Detection and Classification in Intelligent Unmanned Surface Vehicles 

**Title (ZH)**: 基于云-雾自动化技术的自主碰撞检测与分类在智能无人水面车辆中的应用 

**Authors**: Thien Tran, Quang Nguyen, Jonathan Kua, Minh Tran, Toan Luu, Thuong Hoang, Jiong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18024)  

**Abstract**: Industrial Cyber-Physical Systems (ICPS) technologies are foundational in driving maritime autonomy, particularly for Unmanned Surface Vehicles (USVs). However, onboard computational constraints and communication latency significantly restrict real-time data processing, analysis, and predictive modeling, hence limiting the scalability and responsiveness of maritime ICPS. To overcome these challenges, we propose a distributed Cloud-Edge-IoT architecture tailored for maritime ICPS by leveraging design principles from the recently proposed Cloud-Fog Automation paradigm. Our proposed architecture comprises three hierarchical layers: a Cloud Layer for centralized and decentralized data aggregation, advanced analytics, and future model refinement; an Edge Layer that executes localized AI-driven processing and decision-making; and an IoT Layer responsible for low-latency sensor data acquisition. Our experimental results demonstrated improvements in computational efficiency, responsiveness, and scalability. When compared with our conventional approaches, we achieved a classification accuracy of 86\%, with an improved latency performance. By adopting Cloud-Fog Automation, we address the low-latency processing constraints and scalability challenges in maritime ICPS applications. Our work offers a practical, modular, and scalable framework to advance robust autonomy and AI-driven decision-making and autonomy for intelligent USVs in future maritime ICPS. 

**Abstract (ZH)**: 工业 cyber-物理系统 (ICPS) 技术是推动海洋自主航行，特别是无人驾驶水面车辆 (USVs) 自动化的基础。然而，船上计算限制和通信延迟显著限制了实时数据处理、分析和预测建模，从而限制了海洋 ICPS 的可扩展性和响应性。为克服这些挑战，我们提出了一种针对海洋 ICPS 的分布式云-边缘-IoT 架构，通过利用最近提出的云-雾自动化范式的设计理念。我们提出的设计包括三个层次：云层进行集中和分散的数据聚合、高级分析和未来模型优化；边缘层执行本地化的AI驱动的处理和决策；物联网层负责低延迟传感器数据采集。我们的实验结果表明，在计算效率、响应性和可扩展性方面均有所提升。与传统方法相比，我们实现了86%的分类准确性，并改进了延迟性能。通过采用云-雾自动化，我们的工作解决了海洋 ICPS 应用中低延迟处理和可扩展性的挑战。我们的研究提供了一种实用的、模块化的、可扩展的方法，以促进智能 USVs 在未来海洋 ICPS 中的稳健自主和AI驱动的决策和自主性。 

---
# DRAMA-X: A Fine-grained Intent Prediction and Risk Reasoning Benchmark For Driving 

**Title (ZH)**: DRAMA-X: 一种细粒度意图预测和风险推理基准驾驶数据集 

**Authors**: Mihir Godbole, Xiangbo Gao, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17590)  

**Abstract**: Understanding the short-term motion of vulnerable road users (VRUs) like pedestrians and cyclists is critical for safe autonomous driving, especially in urban scenarios with ambiguous or high-risk behaviors. While vision-language models (VLMs) have enabled open-vocabulary perception, their utility for fine-grained intent reasoning remains underexplored. Notably, no existing benchmark evaluates multi-class intent prediction in safety-critical situations, To address this gap, we introduce DRAMA-X, a fine-grained benchmark constructed from the DRAMA dataset via an automated annotation pipeline. DRAMA-X contains 5,686 accident-prone frames labeled with object bounding boxes, a nine-class directional intent taxonomy, binary risk scores, expert-generated action suggestions for the ego vehicle, and descriptive motion summaries. These annotations enable a structured evaluation of four interrelated tasks central to autonomous decision-making: object detection, intent prediction, risk assessment, and action suggestion. As a reference baseline, we propose SGG-Intent, a lightweight, training-free framework that mirrors the ego vehicle's reasoning pipeline. It sequentially generates a scene graph from visual input using VLM-backed detectors, infers intent, assesses risk, and recommends an action using a compositional reasoning stage powered by a large language model. We evaluate a range of recent VLMs, comparing performance across all four DRAMA-X tasks. Our experiments demonstrate that scene-graph-based reasoning enhances intent prediction and risk assessment, especially when contextual cues are explicitly modeled. 

**Abstract (ZH)**: 理解易受损道路使用者（如行人和骑行者）的短期运动对于安全的自动驾驶至关重要，特别是在具有模棱两可或高风险行为的城市场景中。虽然视觉-语言模型(VLMs)已经实现了开放词汇感知，但它们在细粒度意图推理方面的应用仍然不够探索。值得注意的是，目前没有任何基准数据集评估安全关键情况下多类意图预测。为填补这一空白，我们介绍了DRAMA-X，这是一个通过自动化注释流水线从DRAMA数据集构建的细粒度基准数据集。DRAMA-X包含5,686个易发生事故的帧，标记有对象边界框，九类方向意图分类，二元风险评分，为ego车辆生成的专家建议动作，以及描述性运动总结。这些注释使得能够对四个与自主决策相关的任务进行结构化的评估：对象检测、意图预测、风险评估和动作建议。作为参考基准，我们提出了SGG-Intent，这是一个轻量级、无需训练的框架，模仿ego车辆的推理流程。它依次从视觉输入生成场景图，使用VLM支持的检测器推断意图、评估风险，并通过大型语言模型驱动的组合推理阶段推荐动作。我们评估了多种最新的VLM，比较了它们在DRAMA-X四项任务上的性能。我们的实验表明，基于场景图的推理增强意图预测和风险评估，尤其是在对上下文线索进行了明确建模的情况下。 

---
# On the Power of Spatial Locality on Online Routing Problems 

**Title (ZH)**: 空间局部性对在线路由问题的影响 

**Authors**: Swapnil Guragain, Gokarna Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.17517)  

**Abstract**: We consider the online versions of two fundamental routing problems, traveling salesman (TSP) and dial-a-ride (DARP), which have a variety of relevant applications in logistics and robotics. The online versions of these problems concern with efficiently serving a sequence of requests presented in a real-time on-line fashion located at points of a metric space by servers (salesmen/vehicles/robots). In this paper, motivated from real-world applications, such as Uber/Lyft rides, where some limited knowledge is available on the future requests, we propose the {\em spatial locality} model that provides in advance the distance within which new request(s) will be released from the current position of server(s). We study the usefulness of this advanced information on achieving the improved competitive ratios for both the problems with $k\geq 1$ servers, compared to the competitive results established in the literature without such spatial locality consideration. We show that small locality is indeed useful in obtaining improved competitive ratios irrespective of the metric space. 

**Abstract (ZH)**: 考虑物流与机器人领域中两类基本路由问题的在线版本：旅行销售商问题(TSP)和预约车辆问题(DARP)的实时在线版本。本文受到实际应用的启发，如Uber/Lyft乘车服务，其中对于未来请求有一些有限的已知信息，我们提出了空间局部性模型，该模型提前提供了新请求将从服务器当前位置释放出来的距离范围。我们研究了这种先进信息在具有$k \geq 1$台服务器的情况下，如何提高两类问题的竞争比，相比文献中未考虑空间局部性的情况，我们证明了即使在不同的度量空间中，小的空间局部性也可以提高竞争比。 

---
# Public Perceptions of Autonomous Vehicles: A Survey of Pedestrians and Cyclists in Pittsburgh 

**Title (ZH)**: 无人驾驶车辆的公众 perception：匹兹堡行人和骑自行车者的调查 

**Authors**: Rudra Y. Bedekar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17513)  

**Abstract**: This study investigates how autonomous vehicle(AV) technology is perceived by pedestrians and bicyclists in Pittsburgh. Using survey data from over 1200 respondents, the research explores the interplay between demographics, AV interactions, infrastructural readiness, safety perceptions, and trust. Findings highlight demographic divides, infrastructure gaps, and the crucial role of communication and education in AV adoption. 

**Abstract (ZH)**: 本研究调查了匹兹堡行人和骑行者对自动驾驶车辆（AV）技术的感知，基于超过1200名受访者的调查数据，研究探讨了人口统计学、AV互动、基础设施准备情况、安全感知和信任之间的相互作用。研究发现突显了人口统计学差异、基础设施缺口以及在自动驾驶车辆采纳中沟通和教育的关键作用。 

---
# ConciseHint: Boosting Efficient Reasoning via Continuous Concise Hints during Generation 

**Title (ZH)**: ConciseHint: 通过生成过程中连续简洁提示增强高效推理 

**Authors**: Siao Tang, Xinyin Ma, Gongfan Fang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18810)  

**Abstract**: Recent advancements in large reasoning models (LRMs) like DeepSeek-R1 and OpenAI o1 series have achieved notable performance enhancements on complex reasoning tasks by scaling up the generation length by Chain-of-Thought (CoT). However, an emerging issue is their inclination to produce excessively verbose reasoning processes, leading to the inefficiency problem. Existing literature on improving efficiency mainly adheres to the before-reasoning paradigms such as prompting and reasoning or fine-tuning and reasoning, but ignores the promising direction of directly encouraging the model to speak concisely by intervening during the generation of reasoning. In order to fill the blank, we propose a framework dubbed ConciseHint, which continuously encourages the reasoning model to speak concisely by injecting the textual hint (manually designed or trained on the concise data) during the token generation of the reasoning process. Besides, ConciseHint is adaptive to the complexity of the query by adaptively adjusting the hint intensity, which ensures it will not undermine model performance. Experiments on the state-of-the-art LRMs, including DeepSeek-R1 and Qwen-3 series, demonstrate that our method can effectively produce concise reasoning processes while maintaining performance well. For instance, we achieve a reduction ratio of 65\% for the reasoning length on GSM8K benchmark with Qwen-3 4B with nearly no accuracy loss. 

**Abstract (ZH)**: Recent advancements in large reasoning models (LRMs) like DeepSeek-R1和OpenAI o1系列通过扩展Chain-of-Thought (CoT)生成长度实现了复杂推理任务上的显著性能提升。然而，一个新兴的问题是这些模型倾向于生成过于冗长的推理过程，导致效率低下。现有提高效率的研究主要集中在推理之前的提示和微调方法，但忽略了直接在推理生成过程中干预以鼓励模型简洁表达的有前途的方向。为了填补这一空白，我们提出了一种名为ConciseHint的框架，该框架在推理过程的 token 生成过程中注入文本提示（手动设计或基于简洁数据训练），持续鼓励模型简洁表达。此外，ConciseHint可以根据查询的复杂度自适应调整提示强度，确保不会损害模型性能。实验表明，该方法可以在保持性能的同时有效生成简洁的推理过程。例如，使用Qwen-3 4B模型在GSM8K基准上实现了65%的推理长度减少，几乎没有任何准确率损失。 

---
# Dual-level Behavioral Consistency for Inter-group and Intra-group Coordination in Multi-Agent Systems 

**Title (ZH)**: 多层级行为一致性在多agent系统中实现组内与组间协调 

**Authors**: Shuocun Yang, Huawen Hu, Enze Shi, Shu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18651)  

**Abstract**: Behavioral diversity in Multi-agent reinforcement learning(MARL) represents an emerging and promising research area. Prior work has largely centered on intra-group behavioral consistency in multi-agent systems, with limited attention given to behavioral consistency in multi-agent grouping scenarios. In this paper, we introduce Dual-Level Behavioral Consistency (DLBC), a novel MARL control method designed to explicitly regulate agent behaviors at both intra-group and inter-group levels. DLBC partitions agents into distinct groups and dynamically modulates behavioral diversity both within and between these groups. By dynamically modulating behavioral diversity within and between these groups, DLBC achieves enhanced division of labor through inter-group consistency, which constrains behavioral strategies across different groups. Simultaneously, intra-group consistency, achieved by aligning behavioral strategies within each group, fosters stronger intra-group cooperation. Crucially, DLBC's direct constraint of agent policy functions ensures its broad applicability across various algorithmic frameworks. Experimental results in various grouping cooperation scenarios demonstrate that DLBC significantly enhances both intra-group cooperative performance and inter-group task specialization, yielding substantial performance improvements. DLBC provides new ideas for behavioral consistency control of multi-intelligent body systems, and its potential for application in more complex tasks and dynamic environments can be further explored in the future. 

**Abstract (ZH)**: 多智能体强化学习中的行为多样性在多个代理层次上的一致性（Dual-Level Behavioral Consistency in Multi-agent Reinforcement Learning） 

---
# Airalogy: AI-empowered universal data digitization for research automation 

**Title (ZH)**: Airalogy: AI赋能的通用数据数字化研究自动化平台 

**Authors**: Zijie Yang, Qiji Zhou, Fang Guo, Sijie Zhang, Yexun Xi, Jinglei Nie, Yudian Zhu, Liping Huang, Chou Wu, Yonghe Xia, Xiaoyu Ma, Yingming Pu, Panzhong Lu, Junshu Pan, Mingtao Chen, Tiannan Guo, Yanmei Dou, Hongyu Chen, Anping Zeng, Jiaxing Huang, Tian Xu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18586)  

**Abstract**: Research data are the foundation of Artificial Intelligence (AI)-driven science, yet current AI applications remain limited to a few fields with readily available, well-structured, digitized datasets. Achieving comprehensive AI empowerment across multiple disciplines is still out of reach. Present-day research data collection is often fragmented, lacking unified standards, inefficiently managed, and difficult to share. Creating a single platform for standardized data digitization needs to overcome the inherent challenge of balancing between universality (supporting the diverse, ever-evolving needs of various disciplines) and standardization (enforcing consistent formats to fully enable AI). No existing platform accommodates both facets. Building a truly multidisciplinary platform requires integrating scientific domain knowledge with sophisticated computing skills. Researchers often lack the computational expertise to design customized and standardized data recording methods, whereas platform developers rarely grasp the intricate needs of multiple scientific domains. These gaps impede research data standardization and hamper AI-driven progress. In this study, we address these challenges by developing Airalogy (this https URL), the world's first AI- and community-driven platform that balances universality and standardization for digitizing research data across multiple disciplines. Airalogy represents entire research workflows using customizable, standardized data records and offers an advanced AI research copilot for intelligent Q&A, automated data entry, analysis, and research automation. Already deployed in laboratories across all four schools of Westlake University, Airalogy has the potential to accelerate and automate scientific innovation in universities, industry, and the global research community-ultimately benefiting humanity as a whole. 

**Abstract (ZH)**: AI-和社区驱动的跨学科研究数据标准化平台：Airalogy 

---
# T-CPDL: A Temporal Causal Probabilistic Description Logic for Developing Logic-RAG Agent 

**Title (ZH)**: T-CPDL：一种用于开发逻辑-RAG代理的时间因果概率描述逻辑 

**Authors**: Hong Qing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18559)  

**Abstract**: Large language models excel at generating fluent text but frequently struggle with structured reasoning involving temporal constraints, causal relationships, and probabilistic reasoning. To address these limitations, we propose Temporal Causal Probabilistic Description Logic (T-CPDL), an integrated framework that extends traditional Description Logic with temporal interval operators, explicit causal relationships, and probabilistic annotations. We present two distinct variants of T-CPDL: one capturing qualitative temporal relationships through Allen's interval algebra, and another variant enriched with explicit timestamped causal assertions. Both variants share a unified logical structure, enabling complex reasoning tasks ranging from simple temporal ordering to nuanced probabilistic causation. Empirical evaluations on temporal reasoning and causal inference benchmarks confirm that T-CPDL substantially improves inference accuracy, interpretability, and confidence calibration of language model outputs. By delivering transparent reasoning paths and fine-grained temporal and causal semantics, T-CPDL significantly enhances the capability of language models to support robust, explainable, and trustworthy decision-making. This work also lays the groundwork for developing advanced Logic-Retrieval-Augmented Generation (Logic-RAG) frameworks, potentially boosting the reasoning capabilities and efficiency of knowledge graph-enhanced RAG systems. 

**Abstract (ZH)**: 大型语言模型在生成流畅文本方面表现出色，但经常在涉及时间约束、因果关系和概率推理的结构化推理任务中遇到困难。为了解决这些局限性，我们提出了一种时空因果概率描述逻辑（T-CPDL）集成框架，该框架在传统描述逻辑中扩展了时间区间操作符、显式因果关系和概率注释。我们提出了T-CPDL的两种不同变体：一种通过Allen区间代数捕获定性时间关系，另一种带有明确的时间戳因果断言。两种变体共享统一的逻辑结构，能够支持从简单的时序排序到复杂的概率因果推理的复杂推理任务。在时间推理和因果推断基准测试中的实证评估证实，T-CPDL显著提高了语言模型输出的推理准确性、可解释性和置信度校准。通过提供透明的推理路径和精细粒度的时间和因果语义，T-CPDL极大地提升了语言模型支持稳健、可解释和可信决策的能力。此外，本工作也为开发高级逻辑-检索-增强生成（Logic-RAG）框架奠定了基础，有可能提升知识图谱增强的RAG系统推理能力和效率。 

---
# A Question Bank to Assess AI Inclusivity: Mapping out the Journey from Diversity Errors to Inclusion Excellence 

**Title (ZH)**: 用于评估AI包容性的题库：从多样性错误到包容卓越的旅程映射 

**Authors**: Rifat Ara Shams, Didar Zowghi, Muneera Bano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18538)  

**Abstract**: Ensuring diversity and inclusion (D&I) in artificial intelligence (AI) is crucial for mitigating biases and promoting equitable decision-making. However, existing AI risk assessment frameworks often overlook inclusivity, lacking standardized tools to measure an AI system's alignment with D&I principles. This paper introduces a structured AI inclusivity question bank, a comprehensive set of 253 questions designed to evaluate AI inclusivity across five pillars: Humans, Data, Process, System, and Governance. The development of the question bank involved an iterative, multi-source approach, incorporating insights from literature reviews, D&I guidelines, Responsible AI frameworks, and a simulated user study. The simulated evaluation, conducted with 70 AI-generated personas related to different AI jobs, assessed the question bank's relevance and effectiveness for AI inclusivity across diverse roles and application domains. The findings highlight the importance of integrating D&I principles into AI development workflows and governance structures. The question bank provides an actionable tool for researchers, practitioners, and policymakers to systematically assess and enhance the inclusivity of AI systems, paving the way for more equitable and responsible AI technologies. 

**Abstract (ZH)**: 确保人工智能中的多样性和包容性（D&I）对于缓解偏见和促进公平决策至关重要。然而，现有的人工智能风险评估框架往往忽视了包容性，缺乏衡量人工智能系统与D&I原则一致性的标准化工具。本文介绍了一套结构化的人工智能包容性问题库，包含253个问题，旨在从人类、数据、过程、系统和治理五个支柱方面全面评估人工智能的包容性。问题库的开发采用了迭代的多源方法，整合了文献综述、D&I准则、负责任的人工智能框架以及模拟用户研究的见解。模拟评估使用与不同人工智能岗位相关的70个人工智能生成的角色进行，评估了问题库在多样角色和应用场景中的相关性和有效性。研究结果强调了将D&I原则整合到人工智能开发工作流程和治理结构中的重要性。问题库为研究者、从业者和政策制定者提供了一个可操作的工具，以系统地评估和提升人工智能系统的包容性，铺就更公平和负责任的人工智能技术之路。 

---
# Standard Applicability Judgment and Cross-jurisdictional Reasoning: A RAG-based Framework for Medical Device Compliance 

**Title (ZH)**: 标准适用性判断与跨境推理：基于RAG的医疗器械合规性框架 

**Authors**: Yu Han, Aaron Ceross, Jeroen H.M. Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.18511)  

**Abstract**: Identifying the appropriate regulatory standard applicability remains a critical yet understudied challenge in medical device compliance, frequently necessitating expert interpretation of fragmented and heterogeneous documentation across different jurisdictions. To address this challenge, we introduce a modular AI system that leverages a retrieval-augmented generation (RAG) pipeline to automate standard applicability determination. Given a free-text device description, our system retrieves candidate standards from a curated corpus and uses large language models to infer jurisdiction-specific applicability, classified as Mandatory, Recommended, or Not Applicable, with traceable justifications. We construct an international benchmark dataset of medical device descriptions with expert-annotated standard mappings, and evaluate our system against retrieval-only, zero-shot, and rule-based baselines. The proposed approach attains a classification accuracy of 73% and a Top-5 retrieval recall of 87%, demonstrating its effectiveness in identifying relevant regulatory standards. We introduce the first end-to-end system for standard applicability reasoning, enabling scalable and interpretable AI-supported regulatory science. Notably, our region-aware RAG agent performs cross-jurisdictional reasoning between Chinese and U.S. standards, supporting conflict resolution and applicability justification across regulatory frameworks. 

**Abstract (ZH)**: 确定合适的监管标准适用性仍然是医疗器械合规中的一个重要但研究不足的挑战，经常需要专家对跨不同司法管辖区的碎片化和异质化文件进行解释。为应对这一挑战，我们提出了一种模块化人工智能系统，利用检索增强生成（RAG）管道自动确定标准适用性。给定一个自由文本的设备描述，我们的系统从精心编纂的语料库中检索候选标准，并使用大规模语言模型推断出特定于司法管辖区的适用性分类为强制、推荐或不适用，并提供可追溯的依据。我们构建了一个包含医学设备描述和专家注释标准映射的国际基准数据集，并将我们的系统与检索仅限、零样本和基于规则的基线进行了评估。所提出的方法在分类准确性上达到了73%，Top-5检索召回率为87%，证明了其在识别相关监管标准方面的有效性。我们首次提出了标准适用性推理的端到端系统，使其能够支持可扩展和可解释的人工智能辅助监管科学。值得注意的是，我们的区域感知RAG代理在中英文标准之间进行跨境推理，支持不同监管框架下的冲突解决和适用性解释。 

---
# A Conceptual Framework for AI Capability Evaluations 

**Title (ZH)**: 人工智能能力评估的概念框架 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Francisca Gauna Selasco, Luca Nicolás Forziati Gangi, Matheo Sandleris Musa, Lola Ramos Pereyra, Mario Leiva, Juan Gustavo Corvalan, María Vanina Martinez, Gerardo Simari  

**Link**: [PDF](https://arxiv.org/pdf/2506.18213)  

**Abstract**: As AI systems advance and integrate into society, well-designed and transparent evaluations are becoming essential tools in AI governance, informing decisions by providing evidence about system capabilities and risks. Yet there remains a lack of clarity on how to perform these assessments both comprehensively and reliably. To address this gap, we propose a conceptual framework for analyzing AI capability evaluations, offering a structured, descriptive approach that systematizes the analysis of widely used methods and terminology without imposing new taxonomies or rigid formats. This framework supports transparency, comparability, and interpretability across diverse evaluations. It also enables researchers to identify methodological weaknesses, assists practitioners in designing evaluations, and provides policymakers with an accessible tool to scrutinize, compare, and navigate complex evaluation landscapes. 

**Abstract (ZH)**: 随着AI系统的发展和社会整合，设计良好且透明的评估成为AI治理的重要工具，通过提供关于系统能力和风险的证据来指导决策。然而，如何进行全面和可靠地进行这些评估仍然缺乏清晰性。为了解决这一问题，我们提出了一种分析AI能力评估的概念框架，提供了一种结构化、描述性的方法来系统化分析广泛使用的方法和术语，而不强加新的分类或严格的格式。该框架支持评估的透明性、可比性和可解释性，同时也使研究人员能够识别方法论的弱点，帮助实践者设计评估，并为政策制定者提供一个易于使用的工具，以审查、比较和导航复杂的评估景观。 

---
# The Impact of Medication Non-adherence on Adverse Outcomes: Evidence from Schizophrenia Patients via Survival Analysis 

**Title (ZH)**: 药物依从性对不良 outcomes 的影响：基于精神分裂症患者的生存分析证据 

**Authors**: Shahriar Noroozizadeh, Pim Welle, Jeremy C. Weiss, George H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18187)  

**Abstract**: This study quantifies the association between non-adherence to antipsychotic medications and adverse outcomes in individuals with schizophrenia. We frame the problem using survival analysis, focusing on the time to the earliest of several adverse events (early death, involuntary hospitalization, jail booking). We extend standard causal inference methods (T-learner, S-learner, nearest neighbor matching) to utilize various survival models to estimate individual and average treatment effects, where treatment corresponds to medication non-adherence. Analyses are repeated using different amounts of longitudinal information (3, 6, 9, and 12 months). Using data from Allegheny County in western Pennsylvania, we find strong evidence that non-adherence advances adverse outcomes by approximately 1 to 4 months. Ablation studies confirm that county-provided risk scores adjust for key confounders, as their removal amplifies the estimated effects. Subgroup analyses by medication formulation (injectable vs. oral) and medication type consistently show that non-adherence is associated with earlier adverse events. These findings highlight the clinical importance of adherence in delaying psychiatric crises and show that integrating survival analysis with causal inference tools can yield policy-relevant insights. We caution that although we apply causal inference, we only make associative claims and discuss assumptions needed for causal interpretation. 

**Abstract (ZH)**: 本研究量化了抗精神病药物不依从与精神分裂症患者不良结局之间的关联。我们使用生存分析方法，重点关注最早出现的多种不良事件（早期死亡、强制住院、被捕入狱）的时间。我们扩展了标准因果推断方法（T-学习者、S-学习者、最近邻匹配），利用各种生存模型估计个体和平均治疗效果，其中治疗定义为药物不依从。分析使用不同持续时间的纵向信息（3, 6, 9, 和 12个月）重复进行。借助宾夕法尼亚州西部阿勒格尼县的数据，我们发现药物不依从显著加快了不良结局约1到4个月。消融研究证实，由县政府提供的风险评分能够调整关键混杂因素，因为这些评分的移除会放大估计效果。通过药物剂型（注射 vs. 口服）和药物类型分层分析，一致表明药物不依从与更早发生的不良事件相关。这些发现强调了在精神科危机中提高依从性的临床重要性，并展示了将生存分析与因果推断工具结合使用的政策相关见解。我们谨告诫，尽管我们应用了因果推断方法，但我们仅作出关联性声明，并讨论了进行因果解释所需的前提条件。 

---
# Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know? 

**Title (ZH)**: 关于不确定性推理：推理模型-know其不知的能力吗？ 

**Authors**: Zhiting Mei, Christina Zhang, Tenny Yin, Justin Lidard, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18183)  

**Abstract**: Reasoning language models have set state-of-the-art (SOTA) records on many challenging benchmarks, enabled by multi-step reasoning induced using reinforcement learning. However, like previous language models, reasoning models are prone to generating confident, plausible responses that are incorrect (hallucinations). Knowing when and how much to trust these models is critical to the safe deployment of reasoning models in real-world applications. To this end, we explore uncertainty quantification of reasoning models in this work. Specifically, we ask three fundamental questions: First, are reasoning models well-calibrated? Second, does deeper reasoning improve model calibration? Finally, inspired by humans' innate ability to double-check their thought processes to verify the validity of their answers and their confidence, we ask: can reasoning models improve their calibration by explicitly reasoning about their chain-of-thought traces? We introduce introspective uncertainty quantification (UQ) to explore this direction. In extensive evaluations on SOTA reasoning models across a broad range of benchmarks, we find that reasoning models: (i) are typically overconfident, with self-verbalized confidence estimates often greater than 85% particularly for incorrect responses, (ii) become even more overconfident with deeper reasoning, and (iii) can become better calibrated through introspection (e.g., o3-Mini and DeepSeek R1) but not uniformly (e.g., Claude 3.7 Sonnet becomes more poorly calibrated). Lastly, we conclude with important research directions to design necessary UQ benchmarks and improve the calibration of reasoning models. 

**Abstract (ZH)**: 推理语言模型在多步骤推理的驱动下，在许多具有挑战性的基准测试中取得了最新的性能记录，这些推理是由强化学习引起的。然而，就像之前的语言模型一样，推理模型容易生成自信但错误的合理响应（幻觉）。了解何时以及多大程度上信任这些模型对于推理模型在实际应用中的安全部署至关重要。基于此，我们在本工作中探索了推理模型的不确定性量化。具体地，我们提出了三个基本问题：首先，推理模型是否校准良好？其次，更深的推理是否有助于模型校准的改进？最后，借鉴人类校验其思维过程以验证答案的正确性和信心的能力，我们提出：推理模型是否可以通过明证地推理其思维过程轨迹来改善其校准？我们引入了反省不确定性量化（UQ）来探索这一方向。我们在广泛基准测试上的实证评估中发现：（i）推理模型通常过于自信，自我验证的置信度估计值往往超过85%，尤其是在错误的响应中，（ii）随着推理过程的加深，模型的过度自信程度增加，（iii）通过反省（如o3-Mini和DeepSeek R1），模型可以变得更好校准，但并非一概而然（如Claude 3.7 Sonnet变得校准更差）。最后，我们提出了重要的研究方向，以设计必要的不确定性量化基准，从而改善推理模型的校准。 

---
# Chain-of-Memory: Enhancing GUI Agents for Cross-Application Navigation 

**Title (ZH)**: 记忆链：增强跨应用程序导航的GUI代理 

**Authors**: Xinzge Gao, Chuanrui Hu, Bin Chen, Teng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18158)  

**Abstract**: Multimodal large language models (MLLMs) are attracting growing attention in the development of Graphical User Interface (GUI) agents. Existing approaches often rely on historical screenshots or actions to implicitly represent the task state. This reliance poses challenges for GUI agents in accurately understanding task states and underscores the absence of effective mechanisms to store critical information in complex and lengthy cross-app tasks. To address these challenges, we propose Chain-of-Memory (CoM), a novel approach for explicitly modeling short-term and long-term memory in GUI agents. CoM achieves this by capturing action descriptions, integrating task-relevant screen information, and maintaining a dedicated memory module to store and manage this information. By leveraging explicit memory representations, CoM enables GUI agents to better understand task states and retain critical historical information persistently. To equip GUI agents with memory management capabilities and evaluate the effectiveness of CoM, we developed the GUI Odyssey-CoM, a dataset comprising 111k screen-action pairs annotated with Chain-of-Memory. Experimental results demonstrate that CoM significantly improves GUI agents' performance in cross-application tasks. Additionally, GUI Odyssey-CoM enables 7B models to achieve memory management capabilities comparable to 72B models. The dataset and code will be open-sourced. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在图形用户界面（GUI）代理开发中的应用正日益受到关注。现有的方法通常依赖于历史截图或操作来隐式表示任务状态。这种依赖性给GUI代理准确理解任务状态带来了挑战，并突显了在复杂和漫长的跨应用任务中缺乏有效的机制来存储关键信息。为了解决这些挑战，我们提出了Chain-of-Memory（CoM），一种在GUI代理中显式建模短期和长期记忆的新方法。CoM通过捕获操作描述、整合任务相关的屏幕信息，并维护一个专用的记忆模块来存储和管理这些信息，从而实现这一目标。通过利用显式记忆表示，CoM使GUI代理能够更好地理解任务状态并持久地保留关键的历史信息。为了给GUI代理配备记忆管理能力并评估CoM的效果，我们开发了包含111,000个屏幕-操作对的GUI Odyssey-CoM数据集，这些数据对都标注了Chain-of-Memory信息。实验结果表明，CoM显著提高了GUI代理在跨应用任务中的性能。此外，GUI Odyssey-CoM使7B规模的模型能够获得与72B规模模型相当的记忆管理能力。该数据集和代码将开源。 

---
# SE-Merging: A Self-Enhanced Approach for Dynamic Model Merging 

**Title (ZH)**: SE-合并：一种自我增强的动态模型合并方法 

**Authors**: Zijun Chen, Zhanpeng Zhou, Bo Zhang, Weinan Zhang, Xi Sun, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18135)  

**Abstract**: Model merging has gained increasing attention due to its intriguing property: interpolating the parameters of different task-specific fine-tuned models leads to multi-task abilities. However, despite its empirical success, the underlying mechanisms of model merging remain poorly understood. In this work, we delve into the mechanism behind model merging from a representation perspective. Our analysis reveals that model merging achieves multi-task abilities through two key capabilities: i) distinguishing samples from different tasks, and ii) adapting to the corresponding expert model for each sample. These two capabilities allow the merged model to retain task-specific expertise, enabling efficient multi-task adaptation. Building on these insights, we propose \texttt{SE-Merging}, a self-enhanced model merging framework that leverages these two characteristics to dynamically identify the corresponding task for each sample and then adaptively rescales the merging coefficients to further enhance task-specific expertise in the merged model. Notably, \texttt{SE-Merging} achieves dynamic model merging without additional training. Extensive experiments demonstrate that \texttt{SE-Merging} achieves significant performance improvements while remaining compatible with existing model merging techniques. 

**Abstract (ZH)**: 模型融合因其独特的属性而引起了越来越多的关注：插值不同任务特定微调模型的参数可以实现多任务能力。然而，尽管模型融合在实践中取得了成功，其背后的机理仍然知之甚少。在本文中，我们从表示的角度探索了模型融合的机理。我们的分析揭示了模型融合通过两种关键能力实现多任务能力：一是区分不同任务的数据样本，二是根据不同样本适应相应的专家模型。这两种能力使得融合模型能够保留任务特定的专业知识，从而实现高效的多任务适应。基于这些见解，我们提出了一种名为\texttt{SE-Merging}的自我增强模型融合框架，该框架利用这两种特性动态识别每个样本对应的任务，并自适应地重新调整融合系数，以进一步增强融合模型中的任务特定专业知识。值得注意的是，\texttt{SE-Merging}实现了动态模型融合而不需额外训练。大量实验证明，\texttt{SE-Merging}在保持与现有模型融合技术兼容的同时，显著提升了性能。 

---
# Weighted Assumption Based Argumentation to reason about ethical principles and actions 

**Title (ZH)**: 基于加权假设计论的伦理原则与行为推理 

**Authors**: Paolo Baldi, Fabio Aurelio D'Asaro, Abeer Dyoub, Francesca Alessandra Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18056)  

**Abstract**: We augment Assumption Based Argumentation (ABA for short) with weighted argumentation. In a nutshell, we assign weights to arguments and then derive the weight of attacks between ABA arguments. We illustrate our proposal through running examples in the field of ethical reasoning, and present an implementation based on Answer Set Programming. 

**Abstract (ZH)**: 基于权重的论证扩展假设论辩学（ABA）：通过在伦理推理领域运行示例来阐述我们的提案，并基于回答集编程进行实现。 

---
# Action Language BC+ 

**Title (ZH)**: 行为语言BC+ 

**Authors**: Joseph Babb, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.18044)  

**Abstract**: Action languages are formal models of parts of natural language that are designed to describe effects of actions. Many of these languages can be viewed as high level notations of answer set programs structured to represent transition systems. However, the form of answer set programs considered in the earlier work is quite limited in comparison with the modern Answer Set Programming (ASP) language, which allows several useful constructs for knowledge representation, such as choice rules, aggregates, and abstract constraint atoms. We propose a new action language called BC+, which closes the gap between action languages and the modern ASP language. The main idea is to define the semantics of BC+ in terms of general stable model semantics for propositional formulas, under which many modern ASP language constructs can be identified with shorthands for propositional formulas. Language BC+ turns out to be sufficiently expressive to encompass the best features of other action languages, such as languages B, C, C+, and BC. Computational methods available in ASP solvers are readily applicable to compute BC+, which led to an implementation of the language by extending system cplus2asp. 

**Abstract (ZH)**: 行动语言是一类形式化的自然语言部分，用于描述行动的效果。这些语言中的许多可以被视为结构化以表示转换系统的高级记法的回答集程序。然而，早期工作中考虑的回答集程序的形式与现代回答集编程（ASP）语言相比非常有限，后者允许多种用于知识表示的有用构造，如选择规则、聚合和抽象约束原子。我们提出了一种新的行动语言BC+，它弥合了行动语言与现代ASP语言之间的差距。其主要思想是通过使用命题公式的一般稳定模型语义来定义BC+的语义，这样可以将现代ASP语言中的许多构造识别为命题公式的简称。BC+语言证明具有足够的表达能力，可以涵盖其他行动语言（如B、C、C+和BC）的最佳特征。可用的ASP求解器计算方法可以直接应用于计算BC+，从而通过扩展cplus2asp系统实现了该语言的实现。 

---
# medicX-KG: A Knowledge Graph for Pharmacists' Drug Information Needs 

**Title (ZH)**: medicX-KG: 供药师用药信息需求的知识图谱 

**Authors**: Lizzy Farrugia, Lilian M. Azzopardi, Jeremy Debattista, Charlie Abela  

**Link**: [PDF](https://arxiv.org/pdf/2506.17959)  

**Abstract**: The role of pharmacists is evolving from medicine dispensing to delivering comprehensive pharmaceutical services within multidisciplinary healthcare teams. Central to this shift is access to accurate, up-to-date medicinal product information supported by robust data integration. Leveraging artificial intelligence and semantic technologies, Knowledge Graphs (KGs) uncover hidden relationships and enable data-driven decision-making. This paper presents medicX-KG, a pharmacist-oriented knowledge graph supporting clinical and regulatory decisions. It forms the semantic layer of the broader medicX platform, powering predictive and explainable pharmacy services. medicX-KG integrates data from three sources, including, the British National Formulary (BNF), DrugBank, and the Malta Medicines Authority (MMA) that addresses Malta's regulatory landscape and combines European Medicines Agency alignment with partial UK supply dependence. The KG tackles the absence of a unified national drug repository, reducing pharmacists' reliance on fragmented sources. Its design was informed by interviews with practicing pharmacists to ensure real-world applicability. We detail the KG's construction, including data extraction, ontology design, and semantic mapping. Evaluation demonstrates that medicX-KG effectively supports queries about drug availability, interactions, adverse reactions, and therapeutic classes. Limitations, including missing detailed dosage encoding and real-time updates, are discussed alongside directions for future enhancements. 

**Abstract (ZH)**: 药师角色从药物分发向多学科健康Care团队提供全面药物服务的转变：基于Knowledge Graphs的支持与应用 

---
# Learning, Reasoning, Refinement: A Framework for Kahneman's Dual-System Intelligence in GUI Agents 

**Title (ZH)**: 学习、推理、修正：Kahneman的双系统智能在GUI代理中的框架 

**Authors**: Jinjie Wei, Jiyao Liu, Lihao Liu, Ming Hu, Junzhi Ning, Mingcheng Li, Weijie Yin, Junjun He, Xiao Liang, Chao Feng, Dingkang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17913)  

**Abstract**: Graphical User Interface (GUI) agents have made significant progress in automating digital tasks through the utilization of computer vision and language models. Nevertheless, existing agent systems encounter notable limitations. Firstly, they predominantly depend on trial and error decision making rather than progressive reasoning, thereby lacking the capability to learn and adapt from interactive encounters. Secondly, these systems are assessed using overly simplistic single step accuracy metrics, which do not adequately reflect the intricate nature of real world GUI interactions. In this paper, we present CogniGUI, a cognitive framework developed to overcome these limitations by enabling adaptive learning for GUI automation resembling human-like behavior. Inspired by Kahneman's Dual Process Theory, our approach combines two main components: (1) an omni parser engine that conducts immediate hierarchical parsing of GUI elements through quick visual semantic analysis to identify actionable components, and (2) a Group based Relative Policy Optimization (GRPO) grounding agent that assesses multiple interaction paths using a unique relative reward system, promoting minimal and efficient operational routes. This dual-system design facilitates iterative ''exploration learning mastery'' cycles, enabling the agent to enhance its strategies over time based on accumulated experience. Moreover, to assess the generalization and adaptability of agent systems, we introduce ScreenSeek, a comprehensive benchmark that includes multi application navigation, dynamic state transitions, and cross interface coherence, which are often overlooked challenges in current benchmarks. Experimental results demonstrate that CogniGUI surpasses state-of-the-art methods in both the current GUI grounding benchmarks and our newly proposed benchmark. 

**Abstract (ZH)**: 图形用户界面（GUI）代理通过利用计算机视觉和语言模型，在自动化数字任务方面取得了显著进展。然而，现有的代理系统存在明显局限性。首先，它们主要依赖于试探和错误决策，而不是逐步推理，因此缺乏从交互经历中学习和适应的能力。其次，这些系统仅通过过于简单的单一步骤准确性指标进行评估，这未能充分反映现实世界GUI交互的复杂性。本文提出了一种名为CogniGUI的认知框架，旨在通过使GUI自动化能够适应性学习，从而克服这些局限性，使其行为更接近人类。受到Kahneman的双重过程理论启发，我们的方法结合了两个主要组件：（1）一个全能解析引擎，通过快速视觉语义分析对GUI元素进行即时分层解析，以识别可操作组件；（2）基于组的相对策略优化（GRPO）接地代理，使用独特的相对奖励系统评估多条交互路径，促进最小化和高效的操作路径。该双系统设计促进了迭代的“探索学习掌握”循环，使代理能够基于积累的经验增强其策略。此外，为评估代理系统的泛化能力和适应性，我们引入了ScreenSeek这一全面基准，包括多应用导航、动态状态转换和跨界面一致性，这些都是当前基准中常被忽视的挑战。实验结果表明，CogniGUI在现有的GUI接地基准和我们新提出的基准中都超越了最先进的方法。 

---
# Reflective Verbal Reward Design for Pluralistic Alignment 

**Title (ZH)**: 多元共识下的反思性语言奖励设计 

**Authors**: Carter Blair, Kate Larson, Edith Law  

**Link**: [PDF](https://arxiv.org/pdf/2506.17834)  

**Abstract**: AI agents are commonly aligned with "human values" through reinforcement learning from human feedback (RLHF), where a single reward model is learned from aggregated human feedback and used to align an agent's behavior. However, human values are not homogeneous--different people hold distinct and sometimes conflicting values. Aggregating feedback into a single reward model risks disproportionately suppressing minority preferences. To address this, we present a novel reward modeling approach for learning individualized reward models. Our approach uses a language model to guide users through reflective dialogues where they critique agent behavior and construct their preferences. This personalized dialogue history, containing the user's reflections and critiqued examples, is then used as context for another language model that serves as an individualized reward function (what we call a "verbal reward model") for evaluating new trajectories. In studies with 30 participants, our method achieved a 9-12% improvement in accuracy over non-reflective verbal reward models while being more sample efficient than traditional supervised learning methods. 

**Abstract (ZH)**: 基于反思对话的学习个性化奖励模型方法 

---
# Efficient Strategy Synthesis for MDPs via Hierarchical Block Decomposition 

**Title (ZH)**: 基于分层块分解的MDPs高效策略合成 

**Authors**: Alexandros Evangelidis, Gricel Vázquez, Simos Gerasimou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17792)  

**Abstract**: Software-intensive systems, such as software product lines and robotics, utilise Markov decision processes (MDPs) to capture uncertainty and analyse sequential decision-making problems. Despite the usefulness of conventional policy synthesis methods, they fail to scale to large state spaces. Our approach addresses this issue and accelerates policy synthesis in large MDPs by dynamically refining the MDP and iteratively selecting the most fragile MDP regions for refinement. This iterative procedure offers a balance between accuracy and efficiency, as refinement occurs only when necessary. Through a comprehensive empirical evaluation comprising diverse case studies and MDPs up to 1M states, we demonstrate significant performance improvements yielded by our approach compared to the leading probabilistic model checker PRISM (up to 2x), thus offering a very competitive solution for real-world policy synthesis tasks in larger MDPs. 

**Abstract (ZH)**: 基于软件的系统，如软件产品线和机器人技术，利用马尔可夫决策过程（MDPs）来捕捉不确定性并分析 sequential 决策问题。尽管传统的策略合成方法在很多方面都很有用，但它们无法扩展到大规模状态空间。我们的方法解决了这一问题，并通过动态细化 MDP 和迭代选择最脆弱的 MDP 区域进行细化来加速大规模 MDP 的策略合成。这种迭代过程在必要时才进行细化，平衡了准确性和效率。通过涵盖多种案例研究和多达 100 万状态的 MDP 的全面实验证明，与领先的概率模型检测工具 PRISM 相比，我们的方法在性能上取得了显著改进（最高可达 2 倍），从而为更大规模 MDP 中的实际策略合成任务提供了一个极具竞争力的解决方案。 

---
# PhysUniBench: An Undergraduate-Level Physics Reasoning Benchmark for Multimodal Models 

**Title (ZH)**: PhysUniBench: 本科生水平物理推理多模态模型基准 

**Authors**: Lintao Wang, Encheng Su, Jiaqi Liu, Pengze Li, Peng Xia, Jiabei Xiao, Wenlong Zhang, Xinnan Dai, Xi Chen, Yuan Meng, Mingyu Ding, Lei Bai, Wanli Ouyang, Shixiang Tang, Aoran Wang, Xinzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.17667)  

**Abstract**: Physics problem-solving is a challenging domain for large AI models, requiring integration of conceptual understanding, mathematical reasoning, and interpretation of physical diagrams. Current evaluation methodologies show notable limitations in capturing the breadth and complexity of undergraduate-level physics, underscoring the need for more rigorous assessments. To this end, we present PhysUniBench, a large-scale multimodal benchmark designed to evaluate and improve the reasoning capabilities of multimodal large language models (MLLMs) specifically on undergraduate-level physics problems. PhysUniBench consists of 3,304 physics questions spanning 8 major sub-disciplines of physics, each accompanied by one visual diagrams. The benchmark includes both open-ended and multiple-choice questions, systematically curated and difficulty-rated through an iterative model-in-the-loop process. The benchmark's construction involved a rigorous multi-stage process, including multiple roll-outs, expert-level evaluation, automated filtering of easily solved problems, and a nuanced difficulty grading system with five levels. Through extensive experiments, we observe that current state-of-the-art models encounter substantial challenges in physics reasoning. For example, GPT-4o mini achieves only about 34.2\% accuracy in the proposed PhysUniBench. These results highlight that current MLLMs struggle with advanced physics reasoning, especially on multi-step problems and those requiring precise diagram interpretation. By providing a broad and rigorous assessment tool, PhysUniBench aims to drive progress in AI for Science, encouraging the development of models with stronger physical reasoning, problem-solving skills, and multimodal understanding. The benchmark and evaluation scripts are available at this https URL. 

**Abstract (ZH)**: 物理学问题求解是大规模AI模型的一个具有挑战性的领域，需要结合概念理解、数学推理和物理图表的解释。当前的评估方法在捕捉本科物理学的广度和复杂性方面显示出明显局限性，强调了更严格评估的需求。为此，我们提出了PhysUniBench，这是一个大规模多模态基准，旨在评估和提高多模态大规模语言模型（MLLMs）在本科物理学问题上的推理能力。PhysUniBench 包含3304道物理题目，涵盖了8个主要的物理子学科，每题配有1个视觉图表。基准测试包括开放性和选择性问题，历经迭代模型循环过程，系统地进行了分类和难度评级。基准测试的构建过程包括多个阶段，包括多次滚动发布、专家级评估、自动过滤易解问题以及五级细致难度分级系统。通过大量实验，我们发现当前最先进的模型在物理学推理方面面临重大挑战。例如，GPT-4o mini 在提出的PhysUniBench上的准确率仅约为34.2%。这些结果表明当前的MLLM在高级物理学推理方面面临困难，尤其是在多步问题和需要精确图表解读的问题上。通过提供一个广泛而严格的评估工具，PhysUniBench旨在推动科学领域中AI的发展，鼓励开发出具有更强物理推理、问题解决能力和多模态理解能力的模型。基准测试和评估脚本可访问此链接。 

---
# Kaleidoscopic Teaming in Multi Agent Simulations 

**Title (ZH)**: 多智能体仿真中的 Kaleidoscopic 配合模式 

**Authors**: Ninareh Mehrabi, Tharindu Kumarage, Kai-Wei Chang, Aram Galstyan, Rahul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.17514)  

**Abstract**: Warning: This paper contains content that may be inappropriate or offensive.
AI agents have gained significant recent attention due to their autonomous tool usage capabilities and their integration in various real-world applications. This autonomy poses novel challenges for the safety of such systems, both in single- and multi-agent scenarios. We argue that existing red teaming or safety evaluation frameworks fall short in evaluating safety risks in complex behaviors, thought processes and actions taken by agents. Moreover, they fail to consider risks in multi-agent setups where various vulnerabilities can be exposed when agents engage in complex behaviors and interactions with each other. To address this shortcoming, we introduce the term kaleidoscopic teaming which seeks to capture complex and wide range of vulnerabilities that can happen in agents both in single-agent and multi-agent scenarios. We also present a new kaleidoscopic teaming framework that generates a diverse array of scenarios modeling real-world human societies. Our framework evaluates safety of agents in both single-agent and multi-agent setups. In single-agent setup, an agent is given a scenario that it needs to complete using the tools it has access to. In multi-agent setup, multiple agents either compete against or cooperate together to complete a task in the scenario through which we capture existing safety vulnerabilities in agents. We introduce new in-context optimization techniques that can be used in our kaleidoscopic teaming framework to generate better scenarios for safety analysis. Lastly, we present appropriate metrics that can be used along with our framework to measure safety of agents. Utilizing our kaleidoscopic teaming framework, we identify vulnerabilities in various models with respect to their safety in agentic use-cases. 

**Abstract (ZH)**: 警告：本文包含可能不适合或具有冒犯性的内容。
AI智能体由于其自主工具使用能力和在各种现实世界应用中的集成而近期获得了广泛关注。这种自主性为这些系统带来了新的安全挑战，无论是单智能体还是多智能体场景。我们argue现有的红队或安全评估框架在评估智能体在复杂行为、思维过程和行动中的安全风险方面存在不足。此外，它们未能考虑多智能体配置中的风险，在这种配置中，当智能体进行复杂行为和相互作用时，各种漏洞会被暴露。为了解决这一不足，我们引入了“变彩编队”的概念，旨在捕捉智能体在单智能体和多智能体场景中可能发生的各种复杂和广泛的漏洞。我们也提出了一种新的变彩编队框架，用于生成模拟现实人类社会的多样化场景。该框架评估智能体在单智能体和多智能体配置中的安全性。在单智能体配置中，智能体需要使用其可访问的工具完成一个场景。在多智能体配置中，多个智能体要么相互竞争，要么合作完成场景中的任务，从而捕捉智能体现有的安全漏洞。我们介绍了可以在我们的变彩编队框架中使用的新的上下文优化技术，以生成更好的场景进行安全性分析。最后，我们提出了适当的度量标准，用于与我们的框架结合以衡量智能体的安全性。利用我们的变彩编队框架，我们针对智能体使用场景中的安全性识别了各种模型的漏洞。 

---
# Keeping Medical AI Healthy: A Review of Detection and Correction Methods for System Degradation 

**Title (ZH)**: 保持医疗AI健康：系统退化检测与修正方法综述 

**Authors**: Hao Guan, David Bates, Li Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17442)  

**Abstract**: Artificial intelligence (AI) is increasingly integrated into modern healthcare, offering powerful support for clinical decision-making. However, in real-world settings, AI systems may experience performance degradation over time, due to factors such as shifting data distributions, changes in patient characteristics, evolving clinical protocols, and variations in data quality. These factors can compromise model reliability, posing safety concerns and increasing the likelihood of inaccurate predictions or adverse outcomes. This review presents a forward-looking perspective on monitoring and maintaining the "health" of AI systems in healthcare. We highlight the urgent need for continuous performance monitoring, early degradation detection, and effective self-correction mechanisms. The paper begins by reviewing common causes of performance degradation at both data and model levels. We then summarize key techniques for detecting data and model drift, followed by an in-depth look at root cause analysis. Correction strategies are further reviewed, ranging from model retraining to test-time adaptation. Our survey spans both traditional machine learning models and state-of-the-art large language models (LLMs), offering insights into their strengths and limitations. Finally, we discuss ongoing technical challenges and propose future research directions. This work aims to guide the development of reliable, robust medical AI systems capable of sustaining safe, long-term deployment in dynamic clinical settings. 

**Abstract (ZH)**: 人工智能（AI）越来越多地融入现代医疗，为临床决策提供强大的支持。然而，在实际应用中，由于数据分布变化、患者特征变化、临床流程演变以及数据质量差异等因素，AI系统可能会随着时间的推移出现性能下降。这些因素会削弱模型的可靠性，引发行安全风险并增加不准确预测或不良后果的可能性。本文从前瞻性的角度探讨了在医疗保健中监控和维护AI系统“健康状况”的必要性。我们强调了持续性能监控、早期性能下降检测以及有效自我修正机制的迫切需求。文章首先回顾了数据和模型层面常见性能下降的原因。然后总结了检测数据和模型漂移的关键技术，并深入探讨了根本原因分析。进一步审查了从模型重训练到测试时适应的各种纠正策略。本文涵盖了传统机器学习模型和最新大型语言模型（LLMs），提供了它们优缺点的见解。最后，讨论了当前的技术挑战并提出了未来的研究方向。本工作旨在指导开发可靠的、健壮的医疗AI系统，使其能够在动态临床环境中安全、长期部署。 

---
# Resource Rational Contractualism Should Guide AI Alignment 

**Title (ZH)**: 资源理性契约论应指导AI对齐 

**Authors**: Sydney Levine, Matija Franklin, Tan Zhi-Xuan, Secil Yanik Guyot, Lionel Wong, Daniel Kilov, Yejin Choi, Joshua B. Tenenbaum, Noah Goodman, Seth Lazar, Iason Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17434)  

**Abstract**: AI systems will soon have to navigate human environments and make decisions that affect people and other AI agents whose goals and values diverge. Contractualist alignment proposes grounding those decisions in agreements that diverse stakeholders would endorse under the right conditions, yet securing such agreement at scale remains costly and slow -- even for advanced AI. We therefore propose Resource-Rational Contractualism (RRC): a framework where AI systems approximate the agreements rational parties would form by drawing on a toolbox of normatively-grounded, cognitively-inspired heuristics that trade effort for accuracy. An RRC-aligned agent would not only operate efficiently, but also be equipped to dynamically adapt to and interpret the ever-changing human social world. 

**Abstract (ZH)**: 资源理性契约主义：一种AI系统框架 

---
# Individual Causal Inference with Structural Causal Model 

**Title (ZH)**: 基于结构因果模型的个体因果推断 

**Authors**: Daniel T. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17300)  

**Abstract**: Individual causal inference (ICI) uses causal inference methods to understand and predict the effects of interventions on individuals, considering their specific characteristics / facts. It aims to estimate individual causal effect (ICE), which varies across individuals. Estimating ICE can be challenging due to the limited data available for individuals, and the fact that most causal inference methods are population-based. Structural Causal Model (SCM) is fundamentally population-based. Therefore, causal discovery (structural learning and parameter learning), association queries and intervention queries are all naturally population-based. However, exogenous variables (U) in SCM can encode individual variations and thus provide the mechanism for individualized population per specific individual characteristics / facts. Based on this, we propose ICI with SCM as a "rung 3" causal inference, because it involves "imagining" what would be the causal effect of a hypothetical intervention on an individual, given the individual's observed characteristics / facts. Specifically, we propose the indiv-operator, indiv(W), to formalize/represent the population individualization process, and the individual causal query, P(Y | indiv(W), do(X), Z), to formalize/represent ICI. We show and argue that ICI with SCM is inference on individual alternatives (possible), not individual counterfactuals (non-actual). 

**Abstract (ZH)**: 个体因果推断（ICI）使用因果推断方法来理解并预测干预措施对个体的影响，考虑到个体的具体特征/事实。它旨在估计个体因果效应（ICE），而这在不同个体间会有所不同。由于可用的个体数据有限且大多数因果推断方法基于总体，估计ICE具有挑战性。结构因果模型（SCM）本质上是基于总体的，因此因果发现（结构学习和参数学习）、关联查询和干预查询都是基于总体的。然而，SCM中的外生变量（U）可以编码个体差异，从而为特定个体特征/事实下的个体化总体提供机制。基于此，我们提议使用SCM进行个体因果推断（ICI）作为一种“第三级”因果推断方法，因为它涉及“设想”给定个体观察到的特征/事实时，假设干预措施的因果效应。具体地，我们提出了个体算子indiv(W)来形式化/表示人口个体化过程，以及个体因果查询P(Y | indiv(W), do(X), Z)来形式化/表示ICI。我们展示了并论证了使用SCM进行个体因果推断是一种针对个体替代（潜在的）而非个体反事实（非实际的）的推断。 

---
# Mechanistic Interpretability Needs Philosophy 

**Title (ZH)**: 机制可解释性需要哲学 

**Authors**: Iwan Williams, Ninell Oldenburg, Ruchira Dhar, Joshua Hatherley, Constanza Fierro, Nina Rajcic, Sandrine R. Schiller, Filippos Stamatiou, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2506.18852)  

**Abstract**: Mechanistic interpretability (MI) aims to explain how neural networks work by uncovering their underlying causal mechanisms. As the field grows in influence, it is increasingly important to examine not just models themselves, but the assumptions, concepts and explanatory strategies implicit in MI research. We argue that mechanistic interpretability needs philosophy: not as an afterthought, but as an ongoing partner in clarifying its concepts, refining its methods, and assessing the epistemic and ethical stakes of interpreting AI systems. Taking three open problems from the MI literature as examples, this position paper illustrates the value philosophy can add to MI research, and outlines a path toward deeper interdisciplinary dialogue. 

**Abstract (ZH)**: 机制可解释性（MI）旨在通过揭示其潜在的因果机制来解释神经网络的工作原理。随着该领域的影响力日益增强，不仅需要考察模型本身，还需要审视隐含在MI研究中的假设、概念和解释策略。我们认为，机制可解释性需要哲学：不应仅将其视为附带事项，而应将其作为持续的伙伴，用于澄清概念、改进方法，并评估解释AI系统所带来的认识论和伦理学风险。通过MI文献中的三个开放式问题为例，本文阐述哲学能够为MI研究带来的价值，并概述一条通向更深入跨学科对话的道路。 

---
# Shift Happens: Mixture of Experts based Continual Adaptation in Federated Learning 

**Title (ZH)**: 变化不可避免：基于专家混合的联邦学习连续适应 

**Authors**: Rahul Atul Bhope, K.R. Jayaram, Praveen Venkateswaran, Nalini Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.18789)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized clients without sharing raw data, yet faces significant challenges in real-world settings where client data distributions evolve dynamically over time. This paper tackles the critical problem of covariate and label shifts in streaming FL environments, where non-stationary data distributions degrade model performance and require adaptive middleware solutions. We introduce ShiftEx, a shift-aware mixture of experts framework that dynamically creates and trains specialized global models in response to detected distribution shifts using Maximum Mean Discrepancy for covariate shifts. The framework employs a latent memory mechanism for expert reuse and implements facility location-based optimization to jointly minimize covariate mismatch, expert creation costs, and label imbalance. Through theoretical analysis and comprehensive experiments on benchmark datasets, we demonstrate 5.5-12.9 percentage point accuracy improvements and 22-95 % faster adaptation compared to state-of-the-art FL baselines across diverse shift scenarios. The proposed approach offers a scalable, privacy-preserving middleware solution for FL systems operating in non-stationary, real-world conditions while minimizing communication and computational overhead. 

**Abstract (ZH)**: 联邦学习中的协变量和标签转移问题在流式环境中动态分布变化下的应对方法 

---
# ContinualFlow: Learning and Unlearning with Neural Flow Matching 

**Title (ZH)**: 持续流动：神经流匹配中的学习与遗忘 

**Authors**: Lorenzo Simone, Davide Bacciu, Shuangge Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.18747)  

**Abstract**: We introduce ContinualFlow, a principled framework for targeted unlearning in generative models via Flow Matching. Our method leverages an energy-based reweighting loss to softly subtract undesired regions of the data distribution without retraining from scratch or requiring direct access to the samples to be unlearned. Instead, it relies on energy-based proxies to guide the unlearning process. We prove that this induces gradients equivalent to Flow Matching toward a soft mass-subtracted target, and validate the framework through experiments on 2D and image domains, supported by interpretable visualizations and quantitative evaluations. 

**Abstract (ZH)**: 我们介绍了一种名为ContinualFlow的方法，这是一种通过Flow Matching进行生成模型目标性忘存的原理性框架。该方法利用基于能量的重加权损失柔和地减去数据分布中不必要的区域，而无需从头重新训练或直接访问要忘存的数据样本。相反，它依赖基于能量的代理来引导忘存过程。我们证明了这会诱导出与针对柔和减去质量目标的Flow Matching梯度等价的梯度，并通过2D和图像域的实验对其进行验证，这些实验由可解释的可视化和定性评估支持。 

---
# On the Existence of Universal Simulators of Attention 

**Title (ZH)**: 关于通用注意力模拟器的存在性 

**Authors**: Debanjan Dutta, Faizanuddin Ansari, Anish Chakrabarty, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.18739)  

**Abstract**: Prior work on the learnability of transformers has established its capacity to approximate specific algorithmic patterns through training under restrictive architectural assumptions. Fundamentally, these arguments remain data-driven and therefore can only provide a probabilistic guarantee. Expressivity, on the contrary, has theoretically been explored to address the problems \emph{computable} by such architecture. These results proved the Turing-completeness of transformers, investigated bounds focused on circuit complexity, and formal logic. Being at the crossroad between learnability and expressivity, the question remains: \emph{can transformer architectures exactly simulate an arbitrary attention mechanism, or in particular, the underlying operations?} In this study, we investigate the transformer encoder's ability to simulate a vanilla attention mechanism. By constructing a universal simulator $\mathcal{U}$ composed of transformer encoders, we present algorithmic solutions to identically replicate attention outputs and the underlying elementary matrix and activation operations via RASP, a formal framework for transformer computation. Our proofs, for the first time, show the existence of an algorithmically achievable data-agnostic solution, previously known to be approximated only by learning. 

**Abstract (ZH)**: Transformer架构可精确模拟任意注意力机制及其基本矩阵和激活操作的研究 

---
# Deep CNN Face Matchers Inherently Support Revocable Biometric Templates 

**Title (ZH)**: 深度CNN面部匹配器本质上支持可撤回生物特征模板 

**Authors**: Aman Bhatta, Michael C. King, Kevin W. Bowyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.18731)  

**Abstract**: One common critique of biometric authentication is that if an individual's biometric is compromised, then the individual has no recourse. The concept of revocable biometrics was developed to address this concern. A biometric scheme is revocable if an individual can have their current enrollment in the scheme revoked, so that the compromised biometric template becomes worthless, and the individual can re-enroll with a new template that has similar recognition power. We show that modern deep CNN face matchers inherently allow for a robust revocable biometric scheme. For a given state-of-the-art deep CNN backbone and training set, it is possible to generate an unlimited number of distinct face matcher models that have both (1) equivalent recognition power, and (2) strongly incompatible biometric templates. The equivalent recognition power extends to the point of generating impostor and genuine distributions that have the same shape and placement on the similarity dimension, meaning that the models can share a similarity threshold for a 1-in-10,000 false match rate. The biometric templates from different model instances are so strongly incompatible that the cross-instance similarity score for images of the same person is typically lower than the same-instance similarity score for images of different persons. That is, a stolen biometric template that is revoked is of less value in attempting to match the re-enrolled identity than the average impostor template. We also explore the feasibility of using a Vision Transformer (ViT) backbone-based face matcher in the revocable biometric system proposed in this work and demonstrate that it is less suitable compared to typical ResNet-based deep CNN backbones. 

**Abstract (ZH)**: 现代深度CNN面部匹配器中可撤销生物识别方案的实现探究：基于视效变换器的可行性分析 

---
# A Study of Dynamic Stock Relationship Modeling and S&P500 Price Forecasting Based on Differential Graph Transformer 

**Title (ZH)**: 基于差分图变换器的动态股票关系建模与S&P500价格预测研究 

**Authors**: Linyue Hu, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18717)  

**Abstract**: Stock price prediction is vital for investment decisions and risk management, yet remains challenging due to markets' nonlinear dynamics and time-varying inter-stock correlations. Traditional static-correlation models fail to capture evolving stock relationships. To address this, we propose a Differential Graph Transformer (DGT) framework for dynamic relationship modeling and price prediction. Our DGT integrates sequential graph structure changes into multi-head self-attention via a differential graph mechanism, adaptively preserving high-value connections while suppressing noise. Causal temporal attention captures global/local dependencies in price sequences. We further evaluate correlation metrics (Pearson, Mutual Information, Spearman, Kendall's Tau) across global/local/dual scopes as spatial-attention priors. Using 10 years of S&P 500 closing prices (z-score normalized; 64-day sliding windows), DGT with spatial priors outperformed GRU baselines (RMSE: 0.24 vs. 0.87). Kendall's Tau global matrices yielded optimal results (MAE: 0.11). K-means clustering revealed "high-volatility growth" and "defensive blue-chip" stocks, with the latter showing lower errors (RMSE: 0.13) due to stable correlations. Kendall's Tau and Mutual Information excelled in volatile sectors. This study innovatively combines differential graph structures with Transformers, validating dynamic relationship modeling and identifying optimal correlation metrics/scopes. Clustering analysis supports tailored quantitative strategies. Our framework advances financial time-series prediction through dynamic modeling and cross-asset interaction analysis. 

**Abstract (ZH)**: 动态关系建模与价格预测的差分图变换器框架 

---
# Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement 

**Title (ZH)**: 基于频率加权训练损失的音素级DNN语音增强 

**Authors**: Nasser-Eddine Monir, Paul Magron, Romain Serizel  

**Link**: [PDF](https://arxiv.org/pdf/2506.18714)  

**Abstract**: Recent advances in deep learning have significantly improved multichannel speech enhancement algorithms, yet conventional training loss functions such as the scale-invariant signal-to-distortion ratio (SDR) may fail to preserve fine-grained spectral cues essential for phoneme intelligibility. In this work, we propose perceptually-informed variants of the SDR loss, formulated in the time-frequency domain and modulated by frequency-dependent weighting schemes. These weights are designed to emphasize time-frequency regions where speech is prominent or where the interfering noise is particularly strong. We investigate both fixed and adaptive strategies, including ANSI band-importance weights, spectral magnitude-based weighting, and dynamic weighting based on the relative amount of speech and noise. We train the FaSNet multichannel speech enhancement model using these various losses. Experimental results show that while standard metrics such as the SDR are only marginally improved, their perceptual frequency-weighted counterparts exhibit a more substantial improvement. Besides, spectral and phoneme-level analysis indicates better consonant reconstruction, which points to a better preservation of certain acoustic cues. 

**Abstract (ZH)**: 近期深度学习的进展显著提高了多通道语音增强算法的效果，但传统的训练损失函数，如无量纲信噪比（SDR），可能无法保留对音素可懂度至关重要的细粒度频谱线索。在这项工作中，我们提出了感知导向的SDR损失变体，这些变体在时频域中表述，并通过频率依赖的加权方案进行调制。这些权重旨在强调语音突出或干扰噪声特别强烈的时频区域。我们研究了固定和自适应策略，包括ANSI带权重要性加权、基于频谱幅度的加权以及基于语音和噪声相对量的动态加权。我们使用这些不同损失函数来训练FaSNet多通道语音增强模型。实验结果表明，虽然标准指标如SDR仅轻微改善，但其感知频率加权版本显示出更大的改进。此外，频谱和音素级别的分析表明更好的辅音重建，这表明某些声学线索得到了更好的保留。 

---
# Benchmarking histopathology foundation models in a multi-center dataset for skin cancer subtyping 

**Title (ZH)**: 多中心数据集中的皮肤癌亚型分类基础模型benchmark研究 

**Authors**: Pablo Meseguer, Rocío del Amor, Valery Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18668)  

**Abstract**: Pretraining on large-scale, in-domain datasets grants histopathology foundation models (FM) the ability to learn task-agnostic data representations, enhancing transfer learning on downstream tasks. In computational pathology, automated whole slide image analysis requires multiple instance learning (MIL) frameworks due to the gigapixel scale of the slides. The diversity among histopathology FMs has highlighted the need to design real-world challenges for evaluating their effectiveness. To bridge this gap, our work presents a novel benchmark for evaluating histopathology FMs as patch-level feature extractors within a MIL classification framework. For that purpose, we leverage the AI4SkIN dataset, a multi-center cohort encompassing slides with challenging cutaneous spindle cell neoplasm subtypes. We also define the Foundation Model - Silhouette Index (FM-SI), a novel metric to measure model consistency against distribution shifts. Our experimentation shows that extracting less biased features enhances classification performance, especially in similarity-based MIL classifiers. 

**Abstract (ZH)**: 大规模、领域内的预训练使病理学基础模型能够学习任务无关的数据表示，提高了下游任务的迁移学习能力。在计算病理学中，由于玻片的巨像素规模，全玻片图像的自动化分析需要多实例学习（MIL）框架。病理学基础模型之间的多样性凸显了设计实际挑战以评估其有效性的需求。为弥合这一差距，我们的工作呈现了一个新的基准，用于评估病理学基础模型作为MIL分类框架内的patch级特征提取器的有效性。为此，我们利用AI4SkIN数据集，这是一个多中心队列，包含具有挑战性的皮肤梭形细胞肿瘤亚型的玻片。我们还定义了基础模型轮廓指数（FM-SI），这是一种新型度量标准，用于衡量模型在分布偏移下的一致性。我们的实验表明，提取更无偏的特征可以提高分类性能，尤其是在基于相似性的MIL分类器中。 

---
# Federated Loss Exploration for Improved Convergence on Non-IID Data 

**Title (ZH)**: federatedLoss探索以改善非iid数据上的收敛性 

**Authors**: Christian Internò, Markus Olhofer, Yaochu Jin, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.18640)  

**Abstract**: Federated learning (FL) has emerged as a groundbreaking paradigm in machine learning (ML), offering privacy-preserving collaborative model training across diverse datasets. Despite its promise, FL faces significant hurdles in non-identically and independently distributed (non-IID) data scenarios, where most existing methods often struggle with data heterogeneity and lack robustness in performance. This paper introduces Federated Loss Exploration (FedLEx), an innovative approach specifically designed to tackle these challenges. FedLEx distinctively addresses the shortcomings of existing FL methods in non-IID settings by optimizing its learning behavior for scenarios in which assumptions about data heterogeneity are impractical or unknown. It employs a federated loss exploration technique, where clients contribute to a global guidance matrix by calculating gradient deviations for model parameters. This matrix serves as a strategic compass to guide clients' gradient updates in subsequent FL rounds, thereby fostering optimal parameter updates for the global model. FedLEx effectively navigates the complex loss surfaces inherent in non-IID data, enhancing knowledge transfer in an efficient manner, since only a small number of epochs and small amount of data are required to build a strong global guidance matrix that can achieve model convergence without the need for additional data sharing or data distribution statics in a large client scenario. Our extensive experiments with state-of-the art FL algorithms demonstrate significant improvements in performance, particularly under realistic non-IID conditions, thus highlighting FedLEx's potential to overcome critical barriers in diverse FL applications. 

**Abstract (ZH)**: 联邦学习损失探索（FedLEx）：一种应对非同质数据挑战的方法 

---
# Granular-Ball-Induced Multiple Kernel K-Means 

**Title (ZH)**: 由粒球诱导的多重核K均值 

**Authors**: Shuyin Xia, Yifan Wang, Lifeng Shen, Guoyin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18637)  

**Abstract**: Most existing multi-kernel clustering algorithms, such as multi-kernel K-means, often struggle with computational efficiency and robustness when faced with complex data distributions. These challenges stem from their dependence on point-to-point relationships for optimization, which can lead to difficulty in accurately capturing data sets' inherent structure and diversity. Additionally, the intricate interplay between multiple kernels in such algorithms can further exacerbate these issues, effectively impacting their ability to cluster data points in high-dimensional spaces. In this paper, we leverage granular-ball computing to improve the multi-kernel clustering framework. The core of granular-ball computing is to adaptively fit data distribution by balls from coarse to acceptable levels. Each ball can enclose data points based on a density consistency measurement. Such ball-based data description thus improves the computational efficiency and the robustness to unknown noises. Specifically, based on granular-ball representations, we introduce the granular-ball kernel (GBK) and its corresponding granular-ball multi-kernel K-means framework (GB-MKKM) for efficient clustering. Using granular-ball relationships in multiple kernel spaces, the proposed GB-MKKM framework shows its superiority in efficiency and clustering performance in the empirical evaluation of various clustering tasks. 

**Abstract (ZH)**: 大多数现有的多核聚类算法，如多核K均值，往往在面对复杂数据分布时遇到计算效率和鲁棒性的问题。这些问题源自于它们依赖于点对点关系进行优化，这可能导致难以准确捕获数据集固有的结构和多样性。此外，此类算法中多个核函数之间的复杂相互作用会进一步加剧这些问题，从而影响其在高维空间中聚类数据点的能力。本文利用粒度球计算改进多核聚类框架。粒度球计算的核心是通过从较为粗糙到较为合适的层级适应地拟合数据分布。每个球可以根据密度一致性度量包含数据点。基于这种基于球的数据描述，可以提高计算效率和对未知噪声的鲁棒性。具体而言，基于粒度球表示引入粒度球核（GBK）及其对应的粒度球多核K均值框架（GB-MKKM），以实现高效聚类。基于多个核函数空间中的粒度球关系，所提出的GB-MKKM框架在多种聚类任务的实证评估中显示出其在效率和聚类性能上的优越性。 

---
# Multi-Agent Reinforcement Learning for Inverse Design in Photonic Integrated Circuits 

**Title (ZH)**: 多代理 reinforcement 学习在光子集成电路逆向设计中的应用 

**Authors**: Yannik Mahlau, Maximilian Schier, Christoph Reinders, Frederik Schubert, Marco Bügling, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2506.18627)  

**Abstract**: Inverse design of photonic integrated circuits (PICs) has traditionally relied on gradientbased optimization. However, this approach is prone to end up in local minima, which results in suboptimal design functionality. As interest in PICs increases due to their potential for addressing modern hardware demands through optical computing, more adaptive optimization algorithms are needed. We present a reinforcement learning (RL) environment as well as multi-agent RL algorithms for the design of PICs. By discretizing the design space into a grid, we formulate the design task as an optimization problem with thousands of binary variables. We consider multiple two- and three-dimensional design tasks that represent PIC components for an optical computing system. By decomposing the design space into thousands of individual agents, our algorithms are able to optimize designs with only a few thousand environment samples. They outperform previous state-of-the-art gradient-based optimization in both twoand three-dimensional design tasks. Our work may also serve as a benchmark for further exploration of sample-efficient RL for inverse design in photonics. 

**Abstract (ZH)**: 光电集成电路（PICs）的逆向设计传统上依赖于基于梯度的优化。然而，这种方法容易陷入局部极小值，导致设计功能不佳。随着对PICs的兴趣增加，由于其在光学计算中解决现代硬件需求的潜力，需要更具适应性的优化算法。我们提出了一个强化学习（RL）环境以及多智能体RL算法用于PICs的设计。通过将设计空间离散化为网格，我们将设计任务形式化为具有数千个二进制变量的优化问题。我们考虑了多项代表光学计算系统中PIC组件的二维和三维设计任务。通过将设计空间分解为数千个单一代理，我们的算法能够在少量环境样本的情况下优化设计。它们在二维和三维设计任务中均优于先前最先进的基于梯度的优化方法。我们的工作也可能为进一步探索光子学逆向设计中的样本高效RL提供一个基准。 

---
# Frequency Control in Microgrids: An Adaptive Fuzzy-Neural-Network Virtual Synchronous Generator 

**Title (ZH)**: 微电网中的频率控制：自适应模糊神经网络虚拟同步发电机 

**Authors**: Waleed Breesam, Rezvan Alamian, Nima Tashakor, Brahim Elkhalil Youcefa, Stefan M. Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2506.18611)  

**Abstract**: The reliance on distributed renewable energy has increased recently. As a result, power electronic-based distributed generators replaced synchronous generators which led to a change in the dynamic characteristics of the microgrid. Most critically, they reduced system inertia and damping. Virtual synchronous generators emulated in power electronics, which mimic the dynamic behaviour of synchronous generators, are meant to fix this problem. However, fixed virtual synchronous generator parameters cannot guarantee a frequency regulation within the acceptable tolerance range. Conversely, a dynamic adjustment of these virtual parameters promises robust solution with stable frequency. This paper proposes a method to adapt the inertia, damping, and droop parameters dynamically through a fuzzy neural network controller. This controller trains itself online to choose appropriate values for these virtual parameters. The proposed method can be applied to a typical AC microgrid by considering the penetration and impact of renewable energy sources. We study the system in a MATLAB/Simulink model and validate it experimentally in real time using hardware-in-the-loop based on an embedded ARM system (SAM3X8E, Cortex-M3). Compared to traditional and fuzzy logic controller methods, the results demonstrate that the proposed method significantly reduces the frequency deviation to less than 0.03 Hz and shortens the stabilizing/recovery time. 

**Abstract (ZH)**: 基于分布式可再生能源的依赖性增加，最近促进了电力电子基分布式发电系统取代同步发电机，导致微电网动态特性发生变化。关键地，这减少了系统的惯性和阻尼。为了应对这一问题，通过电力电子模拟同步发电机动态特性的虚拟同步发电机被提出。然而，固定的虚拟同步发电机参数无法保证频率调节在可接受的误差范围内。相反，动态调整这些虚拟参数能够提供具有稳定频率的稳健解决方案。本文提出了一种方法，通过模糊神经网络控制器动态调整惯性、阻尼和分量参数。该控制器在线训练以选择这些虚拟参数的适当值。所提出的方案可应用于考虑可再生能源渗透及其影响的典型AC微电网。我们在MATLAB/Simulink模型中研究了该系统，并基于嵌入式ARM系统（SAM3X8E，Cortex-M3）的硬件在环进行实时实验验证。与传统和模糊逻辑控制器方法相比，结果表明所提出的方法显著减少了频率偏差至小于0.03 Hz，并缩短了稳定/恢复时间。 

---
# Simulation-Free Differential Dynamics through Neural Conservation Laws 

**Title (ZH)**: 无模拟的差分动力学通过神经守恒律 

**Authors**: Mengjian Hua, Eric Vanden-Eijnden, Ricky T.Q. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18604)  

**Abstract**: We present a novel simulation-free framework for training continuous-time diffusion processes over very general objective functions. Existing methods typically involve either prescribing the optimal diffusion process -- which only works for heavily restricted problem formulations -- or require expensive simulation to numerically obtain the time-dependent densities and sample from the diffusion process. In contrast, we propose a coupled parameterization which jointly models a time-dependent density function, or probability path, and the dynamics of a diffusion process that generates this probability path. To accomplish this, our approach directly bakes in the Fokker-Planck equation and density function requirements as hard constraints, by extending and greatly simplifying the construction of Neural Conservation Laws. This enables simulation-free training for a large variety of problem formulations, from data-driven objectives as in generative modeling and dynamical optimal transport, to optimality-based objectives as in stochastic optimal control, with straightforward extensions to mean-field objectives due to the ease of accessing exact density functions. We validate our method in a diverse range of application domains from modeling spatio-temporal events to learning optimal dynamics from population data. 

**Abstract (ZH)**: 一种新型无模拟框架：面向非常通用的目标函数训练连续时间扩散过程 

---
# Optimization-Induced Dynamics of Lipschitz Continuity in Neural Networks 

**Title (ZH)**: 优化诱导的Lipschitz连续性动力学在神经网络中的研究 

**Authors**: Róisín Luo, James McDermott, Christian Gagné, Qiang Sun, Colm O'Riordan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18588)  

**Abstract**: Lipschitz continuity characterizes the worst-case sensitivity of neural networks to small input perturbations; yet its dynamics (i.e. temporal evolution) during training remains under-explored. We present a rigorous mathematical framework to model the temporal evolution of Lipschitz continuity during training with stochastic gradient descent (SGD). This framework leverages a system of stochastic differential equations (SDEs) to capture both deterministic and stochastic forces. Our theoretical analysis identifies three principal factors driving the evolution: (i) the projection of gradient flows, induced by the optimization dynamics, onto the operator-norm Jacobian of parameter matrices; (ii) the projection of gradient noise, arising from the randomness in mini-batch sampling, onto the operator-norm Jacobian; and (iii) the projection of the gradient noise onto the operator-norm Hessian of parameter matrices. Furthermore, our theoretical framework sheds light on such as how noisy supervision, parameter initialization, batch size, and mini-batch sampling trajectories, among other factors, shape the evolution of the Lipschitz continuity of neural networks. Our experimental results demonstrate strong agreement between the theoretical implications and the observed behaviors. 

**Abstract (ZH)**: Lipschitz连续性表征了小输入扰动下神经网络的最坏情况敏感性；然而其在训练期间的动力学（即时间演变）尚未得到充分探索。我们提出了一种严谨的数学框架，用于建模使用随机梯度下降（SGD）训练期间Lipschitz连续性的时变演化。该框架利用随机微分方程（SDEs）系统来捕捉确定性和随机性力量。我们的理论分析确定了驱动演化过程的三个主要因素：（i）由最优化动态引发的梯度流在参数矩阵操作范数雅可比矩阵上的投影；（ii）由批量采样中的随机性导致的梯度噪声在操作范数雅可比矩阵上的投影；以及（iii）由梯度噪声在操作范数海森矩阵上的投影。此外，我们的理论框架揭示了诸如嘈杂的监督、参数初始化、批量大小、批量采样轨迹等因素如何塑造神经网络Lipschitz连续性的演化。我们的实验结果表明，理论推导的含义与观察到的行为之间存在很强的一致性。 

---
# PuckTrick: A Library for Making Synthetic Data More Realistic 

**Title (ZH)**: PuckTrick: 一个使合成数据更加真实的库 

**Authors**: Alessandra Agostini, Andrea Maurino, Blerina Spahiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18499)  

**Abstract**: The increasing reliance on machine learning (ML) models for decision-making requires high-quality training data. However, access to real-world datasets is often restricted due to privacy concerns, proprietary restrictions, and incomplete data availability. As a result, synthetic data generation (SDG) has emerged as a viable alternative, enabling the creation of artificial datasets that preserve the statistical properties of real data while ensuring privacy compliance. Despite its advantages, synthetic data is often overly clean and lacks real-world imperfections, such as missing values, noise, outliers, and misclassified labels, which can significantly impact model generalization and robustness. To address this limitation, we introduce Pucktrick, a Python library designed to systematically contaminate synthetic datasets by introducing controlled errors. The library supports multiple error types, including missing data, noisy values, outliers, label misclassification, duplication, and class imbalance, offering a structured approach to evaluating ML model resilience under real-world data imperfections. Pucktrick provides two contamination modes: one for injecting errors into clean datasets and another for further corrupting already contaminated datasets. Through extensive experiments on real-world financial datasets, we evaluate the impact of systematic data contamination on model performance. Our findings demonstrate that ML models trained on contaminated synthetic data outperform those trained on purely synthetic, error-free data, particularly for tree-based and linear models such as SVMs and Extra Trees. 

**Abstract (ZH)**: 不断增加对机器学习（ML）模型的依赖要求高质量的训练数据。但由于隐私顾虑、专有限制和数据不完整性，获取真实世界数据集往往受到限制。因此，合成数据生成（SDG）已成为一种可行的替代方案，能够创建保留真实数据统计属性的人工数据集，同时确保合规性。尽管具有优势，合成数据往往过于干净，缺乏真实世界的瑕疵，如缺失值、噪声、异常值和标签错误分类，这些瑕疵可能严重影响模型的泛化能力和鲁棒性。为解决这一局限性，我们引入了Pucktrick，这是一个设计用于系统性地通过引入受控错误污染合成数据集的Python库。该库支持多种错误类型，包括缺失数据、噪声值、异常值、标签错误分类、重复和类别不平衡，提供了一种结构化方法来评估在真实世界数据瑕疵下的ML模型鲁棒性。Pucktrick提供了两种污染模式：一种用于向干净数据集注入错误，另一种用于进一步污染已受污染的数据集。通过在实际金融数据集上的广泛实验，我们评估了系统性数据污染对模型性能的影响。我们的研究结果表明，使用受污染合成数据训练的ML模型优于使用纯粹合成且无错误数据训练的模型，特别是对于基于树和线性模型如SVM和Extra Trees而言。 

---
# AI-Generated Song Detection via Lyrics Transcripts 

**Title (ZH)**: 基于歌词转录的AI生成歌曲检测 

**Authors**: Markus Frohmann, Elena V. Epure, Gabriel Meseguer-Brocal, Markus Schedl, Romain Hennequin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18488)  

**Abstract**: The recent rise in capabilities of AI-based music generation tools has created an upheaval in the music industry, necessitating the creation of accurate methods to detect such AI-generated content. This can be done using audio-based detectors; however, it has been shown that they struggle to generalize to unseen generators or when the audio is perturbed. Furthermore, recent work used accurate and cleanly formatted lyrics sourced from a lyrics provider database to detect AI-generated music. However, in practice, such perfect lyrics are not available (only the audio is); this leaves a substantial gap in applicability in real-life use cases. In this work, we instead propose solving this gap by transcribing songs using general automatic speech recognition (ASR) models. We do this using several detectors. The results on diverse, multi-genre, and multi-lingual lyrics show generally strong detection performance across languages and genres, particularly for our best-performing model using Whisper large-v2 and LLM2Vec embeddings. In addition, we show that our method is more robust than state-of-the-art audio-based ones when the audio is perturbed in different ways and when evaluated on different music generators. Our code is available at this https URL. 

**Abstract (ZH)**: 基于AI的音乐生成工具 Recent Capabilities 对音乐产业造成的影响 necessitating 准确检测 AI 生成内容方法的创建。这可以通过使用基于音频的检测器来实现；然而，研究表明它们在面对未见的生成器或音频被扰动时难以泛化。此外，近期的研究使用来自歌词提供商数据库的准确且格式整洁的歌词来检测 AI 生成的音乐。然而，在实践中，这样的完美歌词并不可得（只有音频）；这在实际应用场景中留下了一个巨大的缺口。在本工作中，我们 Instead 提出通过使用通用自动语音识别（ASR）模型进行歌词转写来解决这一缺口。我们使用多种检测器进行这一工作。结果显示，在多种语言和多种流派的歌词上，我们的检测性能普遍强劲，尤其是使用 Whisper large-v2 和 LLM2Vec 向量的最佳模型。此外，我们展示了当音频以不同方式被扰动并在不同的音乐生成器上评估时，我们的方法比最先进的基于音频的方法更具鲁棒性。我们的代码可通过以下链接获取。 

---
# Benchmarking Foundation Models and Parameter-Efficient Fine-Tuning for Prognosis Prediction in Medical Imaging 

**Title (ZH)**: 医学影像中病程预测的基准模型及参数高效微调研究 

**Authors**: Filippo Ruffini, Elena Mulero Ayllon, Linlin Shen, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18434)  

**Abstract**: Artificial Intelligence (AI) holds significant promise for improving prognosis prediction in medical imaging, yet its effective application remains challenging. In this work, we introduce a structured benchmark explicitly designed to evaluate and compare the transferability of Convolutional Neural Networks and Foundation Models in predicting clinical outcomes in COVID-19 patients, leveraging diverse publicly available Chest X-ray datasets. Our experimental methodology extensively explores a wide set of fine-tuning strategies, encompassing traditional approaches such as Full Fine-Tuning and Linear Probing, as well as advanced Parameter-Efficient Fine-Tuning methods including Low-Rank Adaptation, BitFit, VeRA, and IA3. The evaluations were conducted across multiple learning paradigms, including both extensive full-data scenarios and more clinically realistic Few-Shot Learning settings, which are critical for modeling rare disease outcomes and rapidly emerging health threats. By implementing a large-scale comparative analysis involving a diverse selection of pretrained models, including general-purpose architectures pretrained on large-scale datasets such as CLIP and DINOv2, to biomedical-specific models like MedCLIP, BioMedCLIP, and PubMedCLIP, we rigorously assess each model's capacity to effectively adapt and generalize to prognosis tasks, particularly under conditions of severe data scarcity and pronounced class imbalance. The benchmark was designed to capture critical conditions common in prognosis tasks, including variations in dataset size and class distribution, providing detailed insights into the strengths and limitations of each fine-tuning strategy. This extensive and structured evaluation aims to inform the practical deployment and adoption of robust, efficient, and generalizable AI-driven solutions in real-world clinical prognosis prediction workflows. 

**Abstract (ZH)**: 人工智能（AI）在医学影像中改善預後預測方面具備重要的潛力，但其有效應用仍然挑戰重重。本研究引入了一個結構化基准，旨在評估和比較卷積神經網絡和Foundation Models在預測COVID-19患者臨床預後方面的轉移能力，利用多種公開可用的胸部X光數據集。實驗方法學 slee 生動探索了大量的調參策略，涵蓋傳統方法如全調參和線性探針，以及先進的參數高效調參方法，包括低秩適應、BitFit、VeRA和IA3。評估在多個學習框架下進行，包括廣泛的全數據場景和更符合臨床現實的少樣本學習設置，這些對於建模罕見疾病的預後和快速出現的健康威脅至關重要。通過大規模比較分析，涉及一系列預訓練模型，包括通用架構如CLIP和DINOv2以及醫學專用模型如MedCLIP、BioMedCLIP和PubMedCLIP，我們嚴格評估每種模型在嚴重數據匱乏和顯著類別不平衡條件下的有效適應和泛化能力，特別是對於預後任務。Benchmark設計用以捕捉預後任務中常見的關鍵條件，包括數據集大小和類別分佈的變異，提供對每種調參策略的強點和局限性的詳細洞察。該廣泛且結構化的評估旨在指導robust、高效和普適的人工智能驅動解決方案在真實世界臨床預後預測流程中的實用部署和採用。 

---
# Latent Space Analysis for Melanoma Prevention 

**Title (ZH)**: latent空间分析在黑色素瘤预防中的应用 

**Authors**: Ciro Listone, Aniello Murano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18414)  

**Abstract**: Melanoma represents a critical health risk due to its aggressive progression and high mortality, underscoring the need for early, interpretable diagnostic tools. While deep learning has advanced in skin lesion classification, most existing models provide only binary outputs, offering limited clinical insight. This work introduces a novel approach that extends beyond classification, enabling interpretable risk modelling through a Conditional Variational Autoencoder. The proposed method learns a structured latent space that captures semantic relationships among lesions, allowing for a nuanced, continuous assessment of morphological differences. An SVM is also trained on this representation effectively differentiating between benign nevi and melanomas, demonstrating strong and consistent performance. More importantly, the learned latent space supports visual and geometric interpretation of malignancy, with the spatial proximity of a lesion to known melanomas serving as a meaningful indicator of risk. This approach bridges predictive performance with clinical applicability, fostering early detection, highlighting ambiguous cases, and enhancing trust in AI-assisted diagnosis through transparent and interpretable decision-making. 

**Abstract (ZH)**: 黑色素瘤由于其侵袭性的进展和高死亡率构成了严重的健康风险，强调了早期、可解释诊断工具的需要。虽然深度学习在皮肤病变分类上取得了进展，但大多数现有模型仅提供二元输出，临床洞察有限。本研究引入了一种新方法，超越分类，通过条件变分自编码器实现可解释的风险建模。所提方法学习一个结构化的潜在空间，捕捉病变间的语义关系，从而实现形态学差异的细致、连续评估。此外，还在该表示上训练了一个SVM，有效地区分良性痣和黑色素瘤，展示了强大的一致性能。更重要的是，学习到的潜在空间支持视觉和几何上的恶性解释，病变的空间位置接近已知黑色素瘤作为风险的有意义指标。该方法将预测性能与临床应用相结合，促进早期检测，突出模糊病例，并通过透明和可解释的决策增强AI辅助诊断的信任度。 

---
# ADNF-Clustering: An Adaptive and Dynamic Neuro-Fuzzy Clustering for Leukemia Prediction 

**Title (ZH)**: ADNF-聚类：一种适用于白血病预测的自适应动态神经模糊聚类 

**Authors**: Marco Aruta, Ciro Listone, Giuseppe Murano, Aniello Murano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18396)  

**Abstract**: Leukemia diagnosis and monitoring rely increasingly on high-throughput image data, yet conventional clustering methods lack the flexibility to accommodate evolving cellular patterns and quantify uncertainty in real time. We introduce Adaptive and Dynamic Neuro-Fuzzy Clustering, a novel streaming-capable framework that combines Convolutional Neural Network-based feature extraction with an online fuzzy clustering engine. ADNF initializes soft partitions via Fuzzy C-Means, then continuously updates micro-cluster centers, densities, and fuzziness parameters using a Fuzzy Temporal Index (FTI) that measures entropy evolution. A topology refinement stage performs density-weighted merging and entropy-guided splitting to guard against over- and under-segmentation. On the C-NMC leukemia microscopy dataset, our tool achieves a silhouette score of 0.51, demonstrating superior cohesion and separation over static baselines. The method's adaptive uncertainty modeling and label-free operation hold immediate potential for integration within the INFANT pediatric oncology network, enabling scalable, up-to-date support for personalized leukemia management. 

**Abstract (ZH)**: 白血病诊断与监测越来越依赖高通量图像数据，但传统聚类方法缺乏适应演化细胞模式和实时量化不确定性的能力。我们提出了一种名为自适应和动态神经模糊聚类（ADNF）的新型流处理框架，该框架结合了基于卷积神经网络的特征提取与在线模糊聚类引擎。ADNF通过模糊C均值初始化软分区，然后使用模糊时间索引（FTI）连续更新微聚类中心、密度和模糊性参数，该索引衡量熵的演变。拓扑优化阶段执行密度加权合并和基于熵的分裂操作，以防止过度分割和欠分割。在C-NMC白血病显微镜数据集上，我们的工具获得了0.51的轮廓分数，显示出比静态基线更好的凝聚性和分离性。该方法的自适应不确定性建模和无标签操作具有立即整合到INFANT儿童肿瘤网络中的潜力，可实现个性化白血病管理的可扩展和及时支持。 

---
# PERSCEN: Learning Personalized Interaction Pattern and Scenario Preference for Multi-Scenario Matching 

**Title (ZH)**: PERSCEN: 学习个性化交互模式和场景偏好以实现多场景匹配 

**Authors**: Haotong Du, Yaqing Wang, Fei Xiong, Lei Shao, Ming Liu, Hao Gu, Quanming Yao, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18382)  

**Abstract**: With the expansion of business scales and scopes on online platforms, multi-scenario matching has become a mainstream solution to reduce maintenance costs and alleviate data sparsity. The key to effective multi-scenario recommendation lies in capturing both user preferences shared across all scenarios and scenario-aware preferences specific to each scenario. However, existing methods often overlook user-specific modeling, limiting the generation of personalized user representations. To address this, we propose PERSCEN, an innovative approach that incorporates user-specific modeling into multi-scenario matching. PERSCEN constructs a user-specific feature graph based on user characteristics and employs a lightweight graph neural network to capture higher-order interaction patterns, enabling personalized extraction of preferences shared across scenarios. Additionally, we leverage vector quantization techniques to distil scenario-aware preferences from users' behavior sequence within individual scenarios, facilitating user-specific and scenario-aware preference modeling. To enhance efficient and flexible information transfer, we introduce a progressive scenario-aware gated linear unit that allows fine-grained, low-latency fusion. Extensive experiments demonstrate that PERSCEN outperforms existing methods. Further efficiency analysis confirms that PERSCEN effectively balances performance with computational cost, ensuring its practicality for real-world industrial systems. 

**Abstract (ZH)**: 基于用户特定建模的多场景匹配方法PERSCEN 

---
# Controlled Generation with Equivariant Variational Flow Matching 

**Title (ZH)**: 具等变变异流匹配的可控生成 

**Authors**: Floor Eijkelboom, Heiko Zimmermann, Sharvaree Vadgama, Erik J Bekkers, Max Welling, Christian A. Naesseth, Jan-Willem van de Meent  

**Link**: [PDF](https://arxiv.org/pdf/2506.18340)  

**Abstract**: We derive a controlled generation objective within the framework of Variational Flow Matching (VFM), which casts flow matching as a variational inference problem. We demonstrate that controlled generation can be implemented two ways: (1) by way of end-to-end training of conditional generative models, or (2) as a Bayesian inference problem, enabling post hoc control of unconditional models without retraining. Furthermore, we establish the conditions required for equivariant generation and provide an equivariant formulation of VFM tailored for molecular generation, ensuring invariance to rotations, translations, and permutations. We evaluate our approach on both uncontrolled and controlled molecular generation, achieving state-of-the-art performance on uncontrolled generation and outperforming state-of-the-art models in controlled generation, both with end-to-end training and in the Bayesian inference setting. This work strengthens the connection between flow-based generative modeling and Bayesian inference, offering a scalable and principled framework for constraint-driven and symmetry-aware generation. 

**Abstract (ZH)**: 我们基于变分流匹配（VFM）框架推导出一种受控生成目标，将流匹配问题视为变分推断问题。我们展示了受控生成可以采用两种方式实现：（1）通过端到端训练条件生成模型，或（2）将其视为贝叶斯推理问题，从而在无需重新训练的情况下对无条件模型进行事后控制。此外，我们建立了等变生成所需的条件，并为分子生成提供了一个面向等变生成的VFM形式化表达，确保对旋转、平移和置换的不变性。我们在无控制和受控分子生成方面进行了评估，实现了无控制生成的顶级性能，并在受控生成中（无论是在端到端训练还是在贝叶斯推理设置中）超过了最先进的模型。本工作加强了基于流的生成建模与贝叶斯推理之间的联系，提供了一个适用于约束驱动和对称感知生成的可扩展且原理性的框架。 

---
# Structured Kolmogorov-Arnold Neural ODEs for Interpretable Learning and Symbolic Discovery of Nonlinear Dynamics 

**Title (ZH)**: 结构化柯尔莫哥洛夫-阿诺尔德神经ODEs及其在可解释学习和非线性动力学的符号发现中的应用 

**Authors**: Wei Liu, Kiran Bacsa, Loon Ching Tang, Eleni Chatzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18339)  

**Abstract**: Understanding and modeling nonlinear dynamical systems is a fundamental problem across scientific and engineering domains. While deep learning has demonstrated remarkable potential for learning complex system behavior, achieving models that are both highly accurate and physically interpretable remains a major challenge. To address this, we propose Structured Kolmogorov-Arnold Neural ODEs (SKANODEs), a novel framework that integrates structured state-space modeling with the Kolmogorov-Arnold Network (KAN). SKANODE first employs a fully trainable KAN as a universal function approximator within a structured Neural ODE framework to perform virtual sensing, recovering latent states that correspond to physically interpretable quantities such as positions and velocities. Once this structured latent representation is established, we exploit the symbolic regression capability of KAN to extract compact and interpretable expressions for the system's governing dynamics. The resulting symbolic expression is then substituted back into the Neural ODE framework and further calibrated through continued training to refine its coefficients, enhancing both the precision of the discovered equations and the predictive accuracy of system responses. Extensive experiments on both simulated and real-world systems demonstrate that SKANODE achieves superior performance while offering interpretable, physics-consistent models that uncover the underlying mechanisms of nonlinear dynamical systems. 

**Abstract (ZH)**: 理解和建模非线性动力系统是科学和工程领域的一项基础问题。尽管深度学习展示了学习复杂系统行为的巨大潜力，但实现既高度准确又具有物理可解释性的模型仍是一项重大挑战。为此，我们提出了一种新颖的框架——结构化柯尔莫哥洛夫-阿诺尔德神经常微分方程（SKANODEs），该框架将结构化状态空间建模与柯尔莫哥洛夫-阿诺尔德网络（KAN）相结合。SKANODE首先利用一个完全可训练的KAN作为结构化神经常微分方程框架内的通用函数逼近器，进行虚拟传感，恢复与位置、速度等物理可解释量相对应的潜在状态。一旦建立这种结构化的潜在表示，我们利用KAN的符号回归能力提取系统的支配动力学的紧凑且可解释的表达式。由此产生的符号表达式随后被重新代入神经常微分方程框架，并通过继续训练进一步校准其系数，从而提高发现方程的精度和系统响应的预测准确性。在仿真和实际系统上的广泛实验表明，SKANODE在保持模型可解释性的同时，实现了物理一致的高性能模型，揭示了非线性动力系统的内在机制。 

---
# Bias vs Bias -- Dawn of Justice: A Fair Fight in Recommendation Systems 

**Title (ZH)**: 偏见 vs 偏见 —— 公正的曙光：推荐系统中的公平争斗 

**Authors**: Tahsin Alamgir Kheya, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2506.18327)  

**Abstract**: Recommendation systems play a crucial role in our daily lives by impacting user experience across various domains, including e-commerce, job advertisements, entertainment, etc. Given the vital role of such systems in our lives, practitioners must ensure they do not produce unfair and imbalanced recommendations. Previous work addressing bias in recommendations overlooked bias in certain item categories, potentially leaving some biases unaddressed. Additionally, most previous work on fair re-ranking focused on binary-sensitive attributes. In this paper, we address these issues by proposing a fairness-aware re-ranking approach that helps mitigate bias in different categories of items. This re-ranking approach leverages existing biases to correct disparities in recommendations across various demographic groups. We show how our approach can mitigate bias on multiple sensitive attributes, including gender, age, and occupation. We experimented on three real-world datasets to evaluate the effectiveness of our re-ranking scheme in mitigating bias in recommendations. Our results show how this approach helps mitigate social bias with little to no degradation in performance. 

**Abstract (ZH)**: 推荐系统在电商、招聘信息、娱乐等领域通过影响用户经验发挥着关键作用。鉴于这类系统在生活中的重要性，从业者必须确保它们不会生成不公平和不平衡的推荐。先前针对推荐偏见的研究忽略了某些项目类别中的偏见，可能导致某些偏见未得以解决。此外，大多数关于公平重新排名的工作主要关注二元敏感属性。在本文中，我们通过提出一种公平意识下的重新排名方法来解决这些问题，该方法有助于在不同类别的项目中减轻偏见。该重新排名方法利用现有偏见来纠正不同 demographic 组群在推荐中的差异。我们展示了该方法如何在多个敏感属性（包括性别、年龄和职业）中减轻偏见。我们在三个实际数据集中进行了实验，以评估该重新排名方案在减轻推荐中的偏见方面的有效性。我们的结果显示，该方法在几乎不牺牲性能的情况下有助于减轻社会偏见。 

---
# A Multi-Scale Spatial Attention-Based Zero-Shot Learning Framework for Low-Light Image Enhancement 

**Title (ZH)**: 多尺度空间注意力机制的零样本学习框架用于低光照图像增强 

**Authors**: Muhammad Azeem Aslam, Hassan Khalid, Nisar Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.18323)  

**Abstract**: Low-light image enhancement remains a challenging task, particularly in the absence of paired training data. In this study, we present LucentVisionNet, a novel zero-shot learning framework that addresses the limitations of traditional and deep learning-based enhancement methods. The proposed approach integrates multi-scale spatial attention with a deep curve estimation network, enabling fine-grained enhancement while preserving semantic and perceptual fidelity. To further improve generalization, we adopt a recurrent enhancement strategy and optimize the model using a composite loss function comprising six tailored components, including a novel no-reference image quality loss inspired by human visual perception. Extensive experiments on both paired and unpaired benchmark datasets demonstrate that LucentVisionNet consistently outperforms state-of-the-art supervised, unsupervised, and zero-shot methods across multiple full-reference and no-reference image quality metrics. Our framework achieves high visual quality, structural consistency, and computational efficiency, making it well-suited for deployment in real-world applications such as mobile photography, surveillance, and autonomous navigation. 

**Abstract (ZH)**: 低光照图像增强仍然是一个具有挑战性的任务，尤其是在缺乏配对训练数据的情况下。本文提出LucentVisionNet，这是一种新颖的零样本学习框架，旨在解决传统和基于深度学习的增强方法的局限性。所提出的方法将多尺度空间注意力与深度曲线估计网络结合，实现了精细的增强同时保持语义和感知保真度。为进一步提高泛化能力，我们采用了递归增强策略，并使用包含六个定制组件的复合损失函数进行优化，其中包括一种由人类视觉感知启发的新颖无参考图像质量损失。在配对和非配对基准数据集上的 extensive 实验表明，LucentVisionNet 在多个全参考和无参考图像质量指标上始终优于最先进的监督学习、无监督学习和零样本方法。该框架实现了高质量的视觉效果、结构一致性以及计算效率，使其适用于移动摄影、监控和自主导航等实际应用。 

---
# Spiffy: Efficient Implementation of CoLaNET for Raspberry Pi 

**Title (ZH)**: Spiffy: 为Raspberry Pi高效实现CoLaNET 

**Authors**: Andrey Derzhavin, Denis Larionov  

**Link**: [PDF](https://arxiv.org/pdf/2506.18306)  

**Abstract**: This paper presents a lightweight software-based approach for running spiking neural networks (SNNs) without relying on specialized neuromorphic hardware or frameworks. Instead, we implement a specific SNN architecture (CoLaNET) in Rust and optimize it for common computing platforms. As a case study, we demonstrate our implementation, called Spiffy, on a Raspberry Pi using the MNIST dataset. Spiffy achieves 92% accuracy with low latency - just 0.9 ms per training step and 0.45 ms per inference step. The code is open-source. 

**Abstract (ZH)**: 本文提出了一种基于轻量级软件的解决方案，用于运行脉冲神经网络（SNNs），而不依赖于专门的神经形态硬件或框架。我们使用Rust实现了一种特定的SNN架构（CoLaNET），并对其进行了优化，以适应常见的计算平台。作为案例研究，我们在Raspberry Pi上使用MNIST数据集展示了我们的实现，名为Spiffy。Spiffy在训练步骤中实现了92%的准确率，延迟仅为0.9毫秒，在推理步骤中延迟为0.45毫秒。该代码是开源的。 

---
# GeNeRT: A Physics-Informed Approach to Intelligent Wireless Channel Modeling via Generalizable Neural Ray Tracing 

**Title (ZH)**: GeNeRT: 一种基于物理的通用神经射线跟踪方法应用于智能无线信道建模 

**Authors**: Kejia Bian, Meixia Tao, Shu Sun, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18295)  

**Abstract**: Neural ray tracing (RT) has emerged as a promising paradigm for channel modeling by combining physical propagation principles with neural networks. It enables high modeling accuracy and efficiency. However, current neural RT methods face two key limitations: constrained generalization capability due to strong spatial dependence, and weak adherence to electromagnetic laws. In this paper, we propose GeNeRT, a Generalizable Neural RT framework with enhanced generalization, accuracy and efficiency. GeNeRT supports both intra-scenario spatial transferability and inter-scenario zero-shot generalization. By incorporating Fresnel-inspired neural network design, it also achieves higher accuracy in multipath component (MPC) prediction. Furthermore, a GPU-tensorized acceleration strategy is introduced to improve runtime efficiency. Extensive experiments conducted in outdoor scenarios demonstrate that GeNeRT generalizes well across untrained regions within a scenario and entirely unseen environments, and achieves superior accuracy in MPC prediction compared to baselines. Moreover, it outperforms Wireless Insite in runtime efficiency, particularly in multi-transmitter settings. Ablation experiments validate the effectiveness of the network architecture and training strategy in capturing physical principles of ray-surface interactions. 

**Abstract (ZH)**: 基于 Fresnel 启发的通用神经射线追踪框架 GeNeRT：增强的泛化能力、准确性和效率 

---
# Selective Social-Interaction via Individual Importance for Fast Human Trajectory Prediction 

**Title (ZH)**: 基于个体重要性的选择性社会互动快速人体轨迹预测 

**Authors**: Yota Urano, Hiromu Taketsugu, Norimichi Ukita  

**Link**: [PDF](https://arxiv.org/pdf/2506.18291)  

**Abstract**: This paper presents an architecture for selecting important neighboring people to predict the primary person's trajectory. To achieve effective neighboring people selection, we propose a people selection module called the Importance Estimator which outputs the importance of each neighboring person for predicting the primary person's future trajectory. To prevent gradients from being blocked by non-differentiable operations when sampling surrounding people based on their importance, we employ the Gumbel Softmax for training. Experiments conducted on the JRDB dataset show that our method speeds up the process with competitive prediction accuracy. 

**Abstract (ZH)**: 本文提出了一种架构，用于选择重要临近人群以预测主要人群的轨迹。为了实现有效的临近人群选择，我们提出了一种称为重要性估计器的人群选择模块，该模块输出每个临近人群对未来主要人群轨迹预测的重要性。为了在基于重要性采样周围人群时防止梯度被非可微操作阻断，我们采用了Gumbel Softmax进行训练。实验在JRDB数据集上的结果显示，我们的方法能够在保持竞争力的预测精度的同时加快过程。 

---
# Tu(r)ning AI Green: Exploring Energy Efficiency Cascading with Orthogonal Optimizations 

**Title (ZH)**: 转向绿色AI：探索正交优化下的能源效率 cascading 效应 

**Authors**: Saurabhsingh Rajput, Mootez Saad, Tushar Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.18289)  

**Abstract**: AI's exponential growth intensifies computational demands and energy challenges. While practitioners employ various optimization techniques, that we refer as "knobs" in this paper, to tune model efficiency, these are typically afterthoughts and reactive ad-hoc changes applied in isolation without understanding their combinatorial effects on energy efficiency. This paper emphasizes on treating energy efficiency as the first-class citizen and as a fundamental design consideration for a compute-intensive pipeline. We show that strategic selection across five AI pipeline phases (data, model, training, system, inference) creates cascading efficiency. Experimental validation shows orthogonal combinations reduce energy consumption by up to $94.6$% while preserving $95.95$% of the original F1 score of non-optimized pipelines. This curated approach provides actionable frameworks for informed sustainable AI that balance efficiency, performance, and environmental responsibility. 

**Abstract (ZH)**: AI的指数级增长加剧了计算需求和能源挑战。本文强调将能源效率视为首要考虑因素和计算密集型流程的基本设计考量。我们展示出在五个AI流程阶段（数据、模型、训练、系统、推理）中进行策略性选择可以产生级联的效率提升。实验验证表明，正交组合可以减少高达94.6%的能耗，同时保持95.95%的原始非优化流程的F1分数。这种精心设计的方法为平衡效率、性能和环境责任的可持续AI提供了可操作的框架。 

---
# Learning Causal Graphs at Scale: A Foundation Model Approach 

**Title (ZH)**: 大规模学习因果图：一种基础模型方法 

**Authors**: Naiyu Yin, Tian Gao, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18285)  

**Abstract**: Due to its human-interpretability and invariance properties, Directed Acyclic Graph (DAG) has been a foundational tool across various areas of AI research, leading to significant advancements. However, DAG learning remains highly challenging, due to its super-exponential growth in computational cost and identifiability issues, particularly in small-sample regimes. To address these two challenges, in this work we leverage the recent success of linear transformers and develop a foundation model approach for discovering multiple order-consistent DAGs across tasks. In particular, we propose Attention-DAG (ADAG), a novel attention-mechanism-based architecture for learning multiple linear Structural Equation Models (SEMs). ADAG learns the mapping from observed data to both graph structure and parameters via a nonlinear attention-based kernel, enabling efficient multi-task estimation of the underlying linear SEMs. By formulating the learning process across multiple tasks as a continuous optimization problem, the pre-trained ADAG model captures the common structural properties as a shared low-dimensional prior, thereby reducing the ill-posedness of downstream DAG learning tasks in small-sample regimes. We evaluate our proposed approach on benchmark synthetic datasets and find that ADAG achieves substantial improvements in both DAG learning accuracy and zero-shot inference efficiency. To the best of our knowledge, this is the first practical approach for pre-training a foundation model specifically designed for DAG learning, representing a step toward more efficient and generalizable down-stream applications in causal discovery. 

**Abstract (ZH)**: 基于注意力机制的多任务DAG学习：一种用于因果发现的基础模型方法 

---
# Open Set Recognition for Endoscopic Image Classification: A Deep Learning Approach on the Kvasir Dataset 

**Title (ZH)**: 基于Kvasir数据集的端oscopic图像开放集识别：一种深度学习方法 

**Authors**: Kasra Moazzami, Seoyoun Son, John Lin, Sun Min Lee, Daniel Son, Hayeon Lee, Jeongho Lee, Seongji Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.18284)  

**Abstract**: Endoscopic image classification plays a pivotal role in medical diagnostics by identifying anatomical landmarks and pathological findings. However, conventional closed-set classification frameworks are inherently limited in open-world clinical settings, where previously unseen conditions can arise andcompromise model reliability. To address this, we explore the application of Open Set Recognition (OSR) techniques on the Kvasir dataset, a publicly available and diverse endoscopic image collection. In this study, we evaluate and compare the OSR capabilities of several representative deep learning architectures, including ResNet-50, Swin Transformer, and a hybrid ResNet-Transformer model, under both closed-set and open-set conditions. OpenMax is adopted as a baseline OSR method to assess the ability of these models to distinguish known classes from previously unseen categories. This work represents one of the first efforts to apply open set recognition to the Kvasir dataset and provides a foundational benchmark for evaluating OSR performance in medical image analysis. Our results offer practical insights into model behavior in clinically realistic settings and highlight the importance of OSR techniques for the safe deployment of AI systems in endoscopy. 

**Abstract (ZH)**: 基于Open Set Recognition的Kvasir数据集内窥图像分类研究 

---
# Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models 

**Title (ZH)**: Morse: 双采样加速扩散模型的无损加速方法 

**Authors**: Chao Li, Jiawei Fan, Anbang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18251)  

**Abstract**: In this paper, we present Morse, a simple dual-sampling framework for accelerating diffusion models losslessly. The key insight of Morse is to reformulate the iterative generation (from noise to data) process via taking advantage of fast jump sampling and adaptive residual feedback strategies. Specifically, Morse involves two models called Dash and Dot that interact with each other. The Dash model is just the pre-trained diffusion model of any type, but operates in a jump sampling regime, creating sufficient space for sampling efficiency improvement. The Dot model is significantly faster than the Dash model, which is learnt to generate residual feedback conditioned on the observations at the current jump sampling point on the trajectory of the Dash model, lifting the noise estimate to easily match the next-step estimate of the Dash model without jump sampling. By chaining the outputs of the Dash and Dot models run in a time-interleaved fashion, Morse exhibits the merit of flexibly attaining desired image generation performance while improving overall runtime efficiency. With our proposed weight sharing strategy between the Dash and Dot models, Morse is efficient for training and inference. Our method shows a lossless speedup of 1.78X to 3.31X on average over a wide range of sampling step budgets relative to 9 baseline diffusion models on 6 image generation tasks. Furthermore, we show that our method can be also generalized to improve the Latent Consistency Model (LCM-SDXL, which is already accelerated with consistency distillation technique) tailored for few-step text-to-image synthesis. The code and models are available at this https URL. 

**Abstract (ZH)**: Morse：一种加速扩散模型的简单双采样框架 

---
# Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability 

**Title (ZH)**: 面向语义结构的生成型攻击以增强对抗性转移性 

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.18248)  

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR). 

**Abstract (ZH)**: 生成式对抗攻击训练一种白盒代理模型上的扰动生成器，并 subsequently 应用于未见过的黑盒受害者模型。与迭代攻击相比，这些方法在推理时效率更高、更具扩展性和迁移性；然而，现有研究尚未充分利用生成模型的表征能力以保留和利用语义信息。具体来说，生成器的中间激活包含了丰富的语义特征——物体边界和粗略形状——这些特征目前被严重忽视，限制了扰动与对齐物体显著区域的能力，后者对于对抗迁移性至关重要。为了解决这一问题，我们引入了一种基于 Mean Teacher 的语义结构感知攻击框架，它作为时间平滑的特征参考。借助这种平滑的参考，我们进一步通过特征蒸馏在学生模型的早期层激活和语义丰富的教师模型的激活之间引导语义一致性。基于实证发现，我们的方法将扰动合成锚定在生成器内的语义显著早期中间块上，从而逐步对那些显著增强对抗迁移性的区域施加对抗扰动。我们在不同的模型、领域和任务上进行了广泛的实验，以展示相对于最先进的生成式攻击的一致改进，并全面使用常规指标和我们提出的新颖的意外纠正率（ACR）进行评估。 

---
# Quantum-Classical Hybrid Quantized Neural Network 

**Title (ZH)**: 量子-经典混合量化神经网络 

**Authors**: Wenxin Li, Chuan Wang, Hongdong Zhu, Qi Gao, Yin Ma, Hai Wei, Kai Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18240)  

**Abstract**: Here in this work, we present a novel Quadratic Binary Optimization (QBO) model for quantized neural network training, enabling the use of arbitrary activation and loss functions through spline interpolation. We introduce Forward Interval Propagation (FIP), a method designed to tackle the challenges of non-linearity and the multi-layer composite structure in neural networks by discretizing activation functions into linear subintervals. This approach preserves the universal approximation properties of neural networks while allowing complex nonlinear functions to be optimized using quantum computers, thus broadening their applicability in artificial intelligence. We provide theoretical upper bounds on the approximation error and the number of Ising spins required, by deriving the sample complexity of the empirical risk minimization problem, from an optimization perspective. A significant challenge in solving the associated Quadratic Constrained Binary Optimization (QCBO) model on a large scale is the presence of numerous constraints. When employing the penalty method to handle these constraints, tuning a large number of penalty coefficients becomes a critical hyperparameter optimization problem, increasing computational complexity and potentially affecting solution quality. To address this, we employ the Quantum Conditional Gradient Descent (QCGD) algorithm, which leverages quantum computing to directly solve the QCBO problem. We prove the convergence of QCGD under a quantum oracle with randomness and bounded variance in objective value, as well as under limited precision constraints in the coefficient matrix. Additionally, we provide an upper bound on the Time-To-Solution for the QCBO solving process. Experimental results using a coherent Ising machine (CIM) demonstrate a 94.95% accuracy on the Fashion MNIST classification task, with only 1.1-bit precision. 

**Abstract (ZH)**: 一种基于插值的量化神经网络训练的二次二进制优化模型 

---
# These are Not All the Features You are Looking For: A Fundamental Bottleneck In Supervised Pretraining 

**Title (ZH)**: 这些未必都是您在寻找的特征：监督预训练中的根本瓶颈 

**Authors**: Xingyu Alice Yang, Jianyu Zhang, Léon Bottou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18221)  

**Abstract**: Transfer learning is a cornerstone of modern machine learning, promising a way to adapt models pretrained on a broad mix of data to new tasks with minimal new data. However, a significant challenge remains in ensuring that transferred features are sufficient to handle unseen datasets, amplified by the difficulty of quantifying whether two tasks are "related". To address these challenges, we evaluate model transfer from a pretraining mixture to each of its component tasks, assessing whether pretrained features can match the performance of task-specific direct training. We identify a fundamental limitation in deep learning models -- an "information saturation bottleneck" -- where networks fail to learn new features once they encode similar competing features during training. When restricted to learning only a subset of key features during pretraining, models will permanently lose critical features for transfer and perform inconsistently on data distributions, even components of the training mixture. Empirical evidence from published studies suggests that this phenomenon is pervasive in deep learning architectures -- factors such as data distribution or ordering affect the features that current representation learning methods can learn over time. This study suggests that relying solely on large-scale networks may not be as effective as focusing on task-specific training, when available. We propose richer feature representations as a potential solution to better generalize across new datasets and, specifically, present existing methods alongside a novel approach, the initial steps towards addressing this challenge. 

**Abstract (ZH)**: 迁移学习是现代机器学习的基石， promise 了一种通过在广泛数据上预训练模型，以最少的新数据适应新任务的方法。然而，确保转移特征足以处理未见过的数据集仍然是一个重大挑战，特别是量化两个任务是否“相关”的难度。为应对这些挑战，我们评估了从预训练混合模型到其各个组件任务的模型迁移，考察预训练特征是否能匹配任务特定直接训练的性能。我们识别出深度学习模型的一个基本局限性——“信息饱和瓶颈”——其中，网络在训练过程中一旦编码了类似的竞争特征，就会无法学习新的特征。当模型仅在预训练过程中学习关键特征的子集时，它们会永久性地失去对于迁移至关重要的特征，并在数据分布上表现不一致，即使是在预训练数据混合的组成部分上。已发表研究的实证证据表明，这一现象在深度学习架构中普遍存在——因素如数据分布或排序会影响当前表示学习方法随时间能够学习的特征。本研究建议，在可用时，仅依赖大规模网络可能不如专注于任务特定训练有效。我们提出了更丰富的特征表示作为更好泛化到新数据集的潜在解决方案，并具体介绍了现有方法和一个新颖方法——初期步骤以应对这一挑战。 

---
# Two Sonification Methods for the MindCube 

**Title (ZH)**: 基于MindCube的两种听觉化方法 

**Authors**: Fangzheng Liu, Lancelot Blanchard, Don D. Haddad, Joseph A. Paradiso  

**Link**: [PDF](https://arxiv.org/pdf/2506.18196)  

**Abstract**: In this work, we explore the musical interface potential of the MindCube, an interactive device designed to study emotions. Embedding diverse sensors and input devices, this interface resembles a fidget cube toy commonly used to help users relieve their stress and anxiety. As such, it is a particularly well-suited controller for musical systems that aim to help with emotion regulation. In this regard, we present two different mappings for the MindCube, with and without AI. With our generative AI mapping, we propose a way to infuse meaning within a latent space and techniques to navigate through it with an external controller. We discuss our results and propose directions for future work. 

**Abstract (ZH)**: MindCube在情绪调节音乐系统中的潜力探索：基于AI的生成映射研究 

---
# Wisdom of Crowds Through Myopic Self-Confidence Adaptation 

**Title (ZH)**: 众人的智慧通过短视自信心适应 

**Authors**: Giacomo Como, Fabio Fagnani, Anton Proskurnikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.18195)  

**Abstract**: The wisdom of crowds is an umbrella term for phenomena suggesting that the collective judgment or decision of a large group can be more accurate than the individual judgments or decisions of the group members. A well-known example illustrating this concept is the competition at a country fair described by Galton, where the median value of the individual guesses about the weight of an ox resulted in an astonishingly accurate estimate of the actual weight. This phenomenon resembles classical results in probability theory and relies on independent decision-making. The accuracy of the group's final decision can be significantly reduced if the final agents' opinions are driven by a few influential agents.
In this paper, we consider a group of agents who initially possess uncorrelated and unbiased noisy measurements of a common state of the world. Assume these agents iteratively update their estimates according to a simple non-Bayesian learning rule, commonly known in mathematical sociology as the French-DeGroot dynamics or iterative opinion pooling. As a result of this iterative distributed averaging process, each agent arrives at an asymptotic estimate of the state of the world, with the variance of this estimate determined by the matrix of weights the agents assign to each other. Every agent aims at minimizing the variance of her asymptotic estimate of the state of the world; however, such variance is also influenced by the weights allocated by other agents. To achieve the best possible estimate, the agents must then solve a game-theoretic, multi-objective optimization problem defined by the available sets of influence weights. We characterize both the Pareto frontier and the set of Nash equilibria in the resulting game. Additionally, we examine asynchronous best-response dynamics for the group of agents and prove their convergence to the set of strict Nash equilibria. 

**Abstract (ZH)**: 群体的智慧是群体的集体判断或决策比个体判断或决策更准确的一种现象的统称。一个著名的例子是由高尔顿描述的乡村 fair 上的竞赛，个体对牛的重量的猜测中位数惊人的准确地估计了实际重量。这一现象类似于概率论中的经典结果，并依赖于独立的决策。如果最终决策者的意见受到少数有影响力的决策者的影响，群体最终决策的准确性会显著降低。

在本文中，我们考虑一群初始时具有关于世界公共状态的不相关且无偏的噪声测量值的代理。假设这些代理根据一个简单的非贝叶斯学习规则迭代地更新其估算值，这种学习规则在数学社会学中广为人知，称为法国-德格鲁 Physics 或迭代意见聚合。由于这一迭代分布式平均过程，每个代理都会达到世界状态的渐进估算，估算的方差由代理相互分配的权重矩阵决定。每个代理都力求最小化其关于世界状态的渐进估算的方差；然而，这种方差也受到其他代理分配权重的影响。为了获得最佳估算，代理必须解一个由可用影响力的权重集合定义的多重目标博弈论优化问题。我们界定了由此产生的博弈的帕累托前沿和纳什均衡集。此外，我们研究了代理组的异步最佳反应动态，并证明它们收敛到严格的纳什均衡集。 

---
# DeInfoReg: A Decoupled Learning Framework for Better Training Throughput 

**Title (ZH)**: DeInfoReg：一种解耦学习框架，以提高训练吞吐量 

**Authors**: Zih-Hao Huang, You-Teng Lin, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18193)  

**Abstract**: This paper introduces Decoupled Supervised Learning with Information Regularization (DeInfoReg), a novel approach that transforms a long gradient flow into multiple shorter ones, thereby mitigating the vanishing gradient problem. Integrating a pipeline strategy, DeInfoReg enables model parallelization across multiple GPUs, significantly improving training throughput. We compare our proposed method with standard backpropagation and other gradient flow decomposition techniques. Extensive experiments on diverse tasks and datasets demonstrate that DeInfoReg achieves superior performance and better noise resistance than traditional BP models and efficiently utilizes parallel computing resources. The code for reproducibility is available at: this https URL. 

**Abstract (ZH)**: 本文介绍了去耦合监督学习与信息正则化（DeInfoReg）方法，该方法将长梯度流转换为多个较短的梯度流，从而减轻梯度消失问题。通过集成管道策略，DeInfoReg 支持在多块 GPU 上进行模型并行化，显著提高训练吞吐量。我们将提出的算法与标准反向传播以及其他梯度流分解技术进行比较。在多种任务和数据集上的广泛实验表明，DeInfoReg 较之传统反向传播模型在性能上更优且具有更好的噪声抗性，并能有效利用并行计算资源。代码可供复现下载：https://github.com/Qwen-Model/DeInfoReg 

---
# Call Me Maybe: Enhancing JavaScript Call Graph Construction using Graph Neural Networks 

**Title (ZH)**: 呼叫我吧：使用图神经网络增强JavaScript调用图构建 

**Authors**: Masudul Hasan Masud Bhuiyan, Gianluca De Stefano, Giancarlo Pellegrino, Cristian-Alexandru Staicu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18191)  

**Abstract**: Static analysis plays a key role in finding bugs, including security issues. A critical step in static analysis is building accurate call graphs that model function calls in a program. However, due to hard-to-analyze language features, existing call graph construction algorithms for JavaScript are neither sound nor complete. Prior work shows that even advanced solutions produce false edges and miss valid ones. In this work, we assist these tools by identifying missed call edges. Our main idea is to frame the problem as link prediction on full program graphs, using a rich representation with multiple edge types. Our approach, GRAPHIA, leverages recent advances in graph neural networks to model non-local relationships between code elements. Concretely, we propose representing JavaScript programs using a combination of syntactic- and semantic-based edges. GRAPHIA can learn from imperfect labels, including static call edges from existing tools and dynamic edges from tests, either from the same or different projects. Because call graphs are sparse, standard machine learning metrics like ROC are not suitable. Instead, we evaluate GRAPHIA by ranking function definitions for each unresolved call site. We conduct a large-scale evaluation on 50 popular JavaScript libraries with 163K call edges (150K static and 13K dynamic). GRAPHIA builds program graphs with 6.6M structural and 386K semantic edges. It ranks the correct target as the top candidate in over 42% of unresolved cases and within the top 5 in 72% of cases, reducing the manual effort needed for analysis. Our results show that learning-based methods can improve the recall of JavaScript call graph construction. To our knowledge, this is the first work to apply GNN-based link prediction to full multi-file program graphs for interprocedural analysis. 

**Abstract (ZH)**: 静态分析在发现包括安全问题在内的漏洞中起着关键作用。构建准确的调用图是静态分析中建模程序中函数调用的一个关键步骤。然而，由于难以分析的语言特性，现有的 JavaScript 调用图构建算法既不完全也不可靠。先前的工作表明，即使是高级解决方案也会产生虚假边并忽略有效的边。在这项工作中，我们通过识别遗漏的调用边来协助这些工具。我们的主要思路是将问题重新表述为程序图上的链接预测问题，使用带有多种边类型的丰富表示。我们的方法 GRAPHIA 利用图神经网络的最新进展来建模代码元素之间的非局部关系。具体而言，我们提出使用基于语法和语义的边来表示 JavaScript 程序。GRAPHIA 可以从不完美的标签中学习，包括现有工具中的静态调用边和测试中的动态边，既可以来自同一项目，也可以来自不同的项目。由于调用图稀疏，标准的机器学习评价度量如 ROC 并不适用。相反，我们通过排名每个未解决的调用位置的函数定义来评估 GRAPHIA。我们在 50 个流行的 JavaScript 库上进行了大规模评估，这些库包含 163K 个调用边（其中150K 个是静态边，13K 个是动态边）。GRAPHIA 构建了具有 6.6M 结构边和 386K 语义边的程序图。在超过 42% 的未解决案例中，GRAPHIA 将正确的目标作为首选项，在 72% 的案例中将其置于前五位，从而减少了分析所需的手动努力。我们的结果显示，基于学习的方法可以提高 JavaScript 调用图构建的召回率。据我们所知，这是首次将基于 GNN 的链接预测应用于用于跨过程分析的完整多文件程序图的工作。 

---
# CareLab at #SMM4H-HeaRD 2025: Insomnia Detection and Food Safety Event Extraction with Domain-Aware Transformers 

**Title (ZH)**: CareLab在#SMM4H-HeaRD 2025上的失眠检测与食品安全事件提取：基于领域意识的变换器模型 

**Authors**: Zihan Liang, Ziwen Pan, Sumon Kanti Dey, Azra Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2506.18185)  

**Abstract**: This paper presents our system for the SMM4H-HeaRD 2025 shared tasks, specifically Task 4 (Subtasks 1, 2a, and 2b) and Task 5 (Subtasks 1 and 2). Task 4 focused on detecting mentions of insomnia in clinical notes, while Task 5 addressed the extraction of food safety events from news articles. We participated in all subtasks and report key findings across them, with particular emphasis on Task 5 Subtask 1, where our system achieved strong performance-securing first place with an F1 score of 0.958 on the test set. To attain this result, we employed encoder-based models (e.g., RoBERTa), alongside GPT-4 for data augmentation. This paper outlines our approach, including preprocessing, model architecture, and subtask-specific adaptations 

**Abstract (ZH)**: 本文介绍了我们参加SMM4H-HeaRD 2025共享任务的系统，具体包括Task 4（子任务1、2a和2b）和Task 5（子任务1和2）。Task 4专注于在临床笔记中检测失眠的提及，而Task 5则处理从新闻文章中抽取食品安全事件的问题。我们参与了所有子任务，并报告了其中的关键发现，特别是在Task 5子任务1中，我们的系统取得出色性能，在测试集上获得了F1分数0.958，位居榜首。为了达到这一结果，我们使用了基于编码器的模型（如RoBERTa），并结合了GPT-4进行数据增强。本文概述了我们的方法，包括预处理、模型架构和针对每个子任务的具体适应性。 

---
# Non-equilibrium Annealed Adjoint Sampler 

**Title (ZH)**: 非平衡 annealed adjoint 采样器 

**Authors**: Jaemoo Choi, Yongxin Chen, Molei Tao, Guan-Horng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18165)  

**Abstract**: Recently, there has been significant progress in learning-based diffusion samplers, which aim to sample from a given unnormalized density. These methods typically follow one of two paradigms: (i) formulating sampling as an unbiased stochastic optimal control (SOC) problem using a canonical reference process, or (ii) refining annealed path measures through importance-weighted sampling. Although annealing approaches have advantages in guiding samples toward high-density regions, reliance on importance sampling leads to high variance and limited scalability in practice. In this paper, we introduce the \textbf{Non-equilibrium Annealed Adjoint Sampler (NAAS)}, a novel SOC-based diffusion sampler that leverages annealed reference dynamics without resorting to importance sampling. NAAS employs a lean adjoint system inspired by adjoint matching, enabling efficient and scalable training. We demonstrate the effectiveness of our approach across a range of tasks, including sampling from classical energy landscapes and molecular Boltzmann distribution. 

**Abstract (ZH)**: 非平衡退火伴随采样器（NAAS）：一种基于伴随的退火差分采样方法 

---
# QuranMorph: Morphologically Annotated Quranic Corpus 

**Title (ZH)**: QuranMorph: 基于形态标注的古兰经语料库 

**Authors**: Diyam Akra, Tymaa Hammouda, Mustafa Jarrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18148)  

**Abstract**: We present the QuranMorph corpus, a morphologically annotated corpus for the Quran (77,429 tokens). Each token in the QuranMorph was manually lemmatized and tagged with its part-of-speech by three expert linguists. The lemmatization process utilized lemmas from Qabas, an Arabic lexicographic database linked with 110 lexicons and corpora of 2 million tokens. The part-of-speech tagging was performed using the fine-grained SAMA/Qabas tagset, which encompasses 40 tags. As shown in this paper, this rich lemmatization and POS tagset enabled the QuranMorph corpus to be inter-linked with many linguistic resources. The corpus is open-source and publicly available as part of the SinaLab resources at (this https URL) 

**Abstract (ZH)**: 我们呈现了QuranMorph语料库，这是一个包含77,429个词素的希伯来语语料库，每条词素都由三位专家语言学家手工词根还原并标注了词性。词根还原过程利用了与110个词典和200万词的语料库链接的Qabas阿拉伯语词表数据库中的词根。词性标注使用了细粒度的SAMA/Qabas标签集，包含40个标签。如本文所示，丰富的词根还原和词性标注标签集使得QuranMorph语料库能够与许多语言资源相互链接。该语料库是开源的，并作为SinaLab资源的一部分公开可用（请参见此链接：[这个链接](this https URL)）。 

---
# Routing Mamba: Scaling State Space Models with Mixture-of-Experts Projection 

**Title (ZH)**: Routing 猫鼬：通过混合专家投影扩展状态空间模型 

**Authors**: Zheng Zhan, Liliang Ren, Shuohang Wang, Liyuan Liu, Yang Liu, Yeyun Gong, Yanzhi Wang, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18145)  

**Abstract**: Linear State Space Models (SSMs) offer remarkable performance gains in efficient sequence modeling, with constant inference-time computation and memory complexity. Recent advances, such as Mamba, further enhance SSMs with input-dependent gating and hardware-aware implementations, positioning them as strong alternatives to Transformers for long sequence modeling. However, efficiently scaling the expressive power of SSMs, particularly with Mixture of Experts (MoE), remains challenging, as naive integration attempts often falter or degrade performance. In this work, we introduce Routing Mamba (RoM), a novel approach that scales SSM parameters using sparse mixtures of linear projection experts. By sharing routing decisions between projection layers and lightweight sub-modules within Mamba across experts, RoM leverages synergies among linear projection experts for effective and efficient sparse scaling of Mamba layers. At a scale of 1.3B active parameters (10B total) and 16K training sequence length, RoM achieves language modeling performance equivalent to a dense Mamba model requiring over 2.3x more active parameters, and demonstrates consistent perplexity across context lengths. Experimental results further show RoM effectively scales hybrid language models, yielding a 23% FLOPS saving compared to dense Mamba scaling for similar performance. 

**Abstract (ZH)**: 基于线性状态空间模型的路由Mamba（RoM）：一种有效的稀疏扩展方法 

---
# AI Harmonizer: Expanding Vocal Expression with a Generative Neurosymbolic Music AI System 

**Title (ZH)**: AI Harmonizer: 通过生成神经符号音乐AI系统扩展 vocal 表达 

**Authors**: Lancelot Blanchard, Cameron Holt, Joseph A. Paradiso  

**Link**: [PDF](https://arxiv.org/pdf/2506.18143)  

**Abstract**: Vocals harmonizers are powerful tools to help solo vocalists enrich their melodies with harmonically supportive voices. These tools exist in various forms, from commercially available pedals and software to custom-built systems, each employing different methods to generate harmonies. Traditional harmonizers often require users to manually specify a key or tonal center, while others allow pitch selection via an external keyboard-both approaches demanding some degree of musical expertise. The AI Harmonizer introduces a novel approach by autonomously generating musically coherent four-part harmonies without requiring prior harmonic input from the user. By integrating state-of-the-art generative AI techniques for pitch detection and voice modeling with custom-trained symbolic music models, our system arranges any vocal melody into rich choral textures. In this paper, we present our methods, explore potential applications in performance and composition, and discuss future directions for real-time implementations. While our system currently operates offline, we believe it represents a significant step toward AI-assisted vocal performance and expressive musical augmentation. We release our implementation on GitHub. 

**Abstract (ZH)**: 声乐和声器是帮助独唱歌手丰富旋律、添加和声支持声音的强大工具。这些工具以多种形式存在，从商业可购的踏板和软件到自定义系统，每种工具都采用了不同方法来生成和声。传统和声器通常需要用户手动指定一个键或调中心，而其他一些工具则允许通过外部键盘选择音高——这两种方法都需要一定的音乐专业知识。AI 和声器通过自主生成音乐连贯的四部和声，无需用户先提供和声输入，引入了一种新的方法。通过将最先进的生成AI技术与音高检测和声模型结合，并结合自定义训练的符号音乐模型，我们的系统将任何 vocal 熔调排列成丰富的合唱纹理。在本文中，我们介绍了我们的方法，探讨了在表演和创作中的潜在应用，并讨论了实时实现的未来方向。尽管我们的系统目前是离线运行的，但我们认为它代表了AI辅助声乐表演和表达性音乐增强的重要一步。我们在GitHub上发布了我们的实现。 

---
# Conceptualization, Operationalization, and Measurement of Machine Companionship: A Scoping Review 

**Title (ZH)**: 机器同伴概念化、操作化及测量：一项范围性回顾 

**Authors**: Jaime Banks, Zhixin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18119)  

**Abstract**: The notion of machine companions has long been embedded in social-technological imaginaries. Recent advances in AI have moved those media musings into believable sociality manifested in interfaces, robotic bodies, and devices. Those machines are often referred to colloquially as "companions" yet there is little careful engagement of machine companionship (MC) as a formal concept or measured variable. This PRISMA-guided scoping review systematically samples, surveys, and synthesizes current scholarly works on MC (N = 71; 2017-2025), to that end. Works varied widely in considerations of MC according to guiding theories, dimensions of a-priori specified properties (subjectively positive, sustained over time, co-active, autotelic), and in measured concepts (with more than 50 distinct measured variables). WE ultimately offer a literature-guided definition of MC as an autotelic, coordinated connection between human and machine that unfolds over time and is subjectively positive. 

**Abstract (ZH)**: 机器伴侣概念长久以来根植于社会技术幻想之中。近年来，AI的发展将这些媒体想象物表现成了可信的社会互动，体现在界面、机器人身体和设备中。这些机器通常被通俗地称为“伴侣”，但鲜有严谨地将机器伴侣ship（MC）作为一个正式的概念或量化的变量进行探讨。本研究遵循PRISMA指南，系统性地抽样、调查和综述了2017-2025年间关于MC的现有学术作品（N=71），旨在定义MC。作品在指导理论、先验规定属性（主观积极、持续时间、协同作用、终末学的）以及测量概念方面差异广泛（超过50个不同的测量变量）。最终，我们提供了一个基于文献界定的MC定义，即一种随时间展开的、主观上积极的人机协同连接。 

---
# RL for Reasoning by Adaptively Revealing Rationales 

**Title (ZH)**: 基于自适应揭示推理依据的强化学习方法 

**Authors**: Mohammad Hossein Amani, Aryo Lotfi, Nicolas Mario Baldwin, Samy Bengio, Mehrdad Farajtabar, Emmanuel Abbe, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2506.18110)  

**Abstract**: We propose that reinforcement learning (RL) from partial expert demonstrations is not merely a training heuristic, but a promising framework for solving complex sequence generation tasks. Supervised fine-tuning (SFT) relies on dense ground-truth labels, which become increasingly costly as sequence length grows. RL, on the other hand, struggles with sparse rewards and a combinatorially large output space. We address this by introducing adaptive backtracking (AdaBack), a per-sample curriculum learning algorithm that reveals only a partial prefix of the target output during training. The supervision length is adjusted dynamically for each sample based on the model's past reward signal, allowing it to incrementally learn to complete reasoning chains by conditioning on correct partial solutions. We investigate this intermediate regime between SFT and RL and argue that per-sample curriculum learning is more than a trade-off between efficiency and generality, it can succeed in tasks with long sequences of latent dependencies where SFT and RL both fail to generalize. Using a synthetic task with latent parity constraints, we show that our adaptive curriculum over partial answers reliably solves problems that are otherwise intractable. On mathematical reasoning benchmarks (MATH, GSM8k), we find that curriculum learning enables models to solve problems that RL alone cannot, acquiring new reasoning capabilities through incremental exposure to partial solutions. 

**Abstract (ZH)**: 从部分专家演示中学习的强化学习：一种解决复杂序列生成任务的有前途的框架 

---
# Distributionally robust minimization in meta-learning for system identification 

**Title (ZH)**: 元学习中的分布鲁棒最小化在系统识别中的应用 

**Authors**: Matteo Rufolo, Dario Piga, Marco Forgione  

**Link**: [PDF](https://arxiv.org/pdf/2506.18074)  

**Abstract**: Meta learning aims at learning how to solve tasks, and thus it allows to estimate models that can be quickly adapted to new scenarios. This work explores distributionally robust minimization in meta learning for system identification. Standard meta learning approaches optimize the expected loss, overlooking task variability. We use an alternative approach, adopting a distributionally robust optimization paradigm that prioritizes high-loss tasks, enhancing performance in worst-case scenarios. Evaluated on a meta model trained on a class of synthetic dynamical systems and tested in both in-distribution and out-of-distribution settings, the proposed approach allows to reduce failures in safety-critical applications. 

**Abstract (ZH)**: 元学习旨在学习如何解决任务，从而能够估算出可以快速适应新场景的模型。本文探讨了在系统识别中的元学习中的分布鲁棒最小化方法。标准的元学习方法优化期望损失，忽略了任务的变异性。我们采用一种替代方法，采用分布鲁棒优化范式，优先考虑高损失任务，从而在最坏情况下提高性能。在基于一类合成动力学系统训练的元模型上进行训练并在分布内和分布外设置下进行测试，所提出的方法能够减少在安全关键应用中的失败情况。 

---
# Pathwise Explanation of ReLU Neural Networks 

**Title (ZH)**: ReLU神经网络的路径解释 

**Authors**: Seongwoo Lim, Won Jo, Joohyung Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18037)  

**Abstract**: Neural networks have demonstrated a wide range of successes, but their ``black box" nature raises concerns about transparency and reliability. Previous research on ReLU networks has sought to unwrap these networks into linear models based on activation states of all hidden units. In this paper, we introduce a novel approach that considers subsets of the hidden units involved in the decision making path. This pathwise explanation provides a clearer and more consistent understanding of the relationship between the input and the decision-making process. Our method also offers flexibility in adjusting the range of explanations within the input, i.e., from an overall attribution input to particular components within the input. Furthermore, it allows for the decomposition of explanations for a given input for more detailed explanations. Experiments demonstrate that our method outperforms others both quantitatively and qualitatively. 

**Abstract (ZH)**: 神经网络在广泛领域取得了成功，但其“黑箱”性质引发了透明度和可靠性方面的担忧。前人在ReLU网络上的研究试图通过所有隐藏单元的激活状态将这些网络展开为线性模型。在本文中，我们提出了一种新的方法，考虑参与决策路径的隐藏单元子集。这种路径解释方法为输入与决策过程之间的关系提供了更清晰、更一致的理解。该方法还提供了在输入范围内调整解释范围的灵活性，即从整体归因输入到输入的具体组件。此外，它还允许对给定输入的解释进行分解，以提供更详细的解释。实验表明，我们的方法在定量和定性方面都优于其他方法。 

---
# Probing the Embedding Space of Transformers via Minimal Token Perturbations 

**Title (ZH)**: 通过最小토큰扰动探究变压器的嵌入空间 

**Authors**: Eddie Conti, Alejandro Astruc, Alvaro Parafita, Axel Brando  

**Link**: [PDF](https://arxiv.org/pdf/2506.18011)  

**Abstract**: Understanding how information propagates through Transformer models is a key challenge for interpretability. In this work, we study the effects of minimal token perturbations on the embedding space. In our experiments, we analyze the frequency of which tokens yield to minimal shifts, highlighting that rare tokens usually lead to larger shifts. Moreover, we study how perturbations propagate across layers, demonstrating that input information is increasingly intermixed in deeper layers. Our findings validate the common assumption that the first layers of a model can be used as proxies for model explanations. Overall, this work introduces the combination of token perturbations and shifts on the embedding space as a powerful tool for model interpretability. 

**Abstract (ZH)**: 理解信息在Transformer模型中的传播机制是可解释性的关键挑战。在本工作中，我们研究了最小token扰动对嵌入空间的影响。在我们的实验中，我们分析了导致最小位移的token频率，指出罕见token通常会导致更大的位移。此外，我们研究了扰动在层间传播的方式，证明了输入信息在更深的层中越来越多地混合在一起。我们的发现验证了模型早期层可以用作模型解释的代理的common假设。总体而言，本工作引入了令牌扰动和嵌入空间中的位移组合，作为模型可解释性的一个强大工具。 

---
# h-calibration: Rethinking Classifier Recalibration with Probabilistic Error-Bounded Objective 

**Title (ZH)**: h-校准：基于概率误差界目标重思考分类器再校准 

**Authors**: Wenjian Huang, Guiping Cao, Jiahao Xia, Jingkun Chen, Hao Wang, Jianguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17968)  

**Abstract**: Deep neural networks have demonstrated remarkable performance across numerous learning tasks but often suffer from miscalibration, resulting in unreliable probability outputs. This has inspired many recent works on mitigating miscalibration, particularly through post-hoc recalibration methods that aim to obtain calibrated probabilities without sacrificing the classification performance of pre-trained models. In this study, we summarize and categorize previous works into three general strategies: intuitively designed methods, binning-based methods, and methods based on formulations of ideal calibration. Through theoretical and practical analysis, we highlight ten common limitations in previous approaches. To address these limitations, we propose a probabilistic learning framework for calibration called h-calibration, which theoretically constructs an equivalent learning formulation for canonical calibration with boundedness. On this basis, we design a simple yet effective post-hoc calibration algorithm. Our method not only overcomes the ten identified limitations but also achieves markedly better performance than traditional methods, as validated by extensive experiments. We further analyze, both theoretically and experimentally, the relationship and advantages of our learning objective compared to traditional proper scoring rule. In summary, our probabilistic framework derives an approximately equivalent differentiable objective for learning error-bounded calibrated probabilities, elucidating the correspondence and convergence properties of computational statistics with respect to theoretical bounds in canonical calibration. The theoretical effectiveness is verified on standard post-hoc calibration benchmarks by achieving state-of-the-art performance. This research offers valuable reference for learning reliable likelihood in related fields. 

**Abstract (ZH)**: 深度神经网络在众多学习任务中展现了出色的性能，但往往存在校准不足的问题，导致概率输出不可靠。为此，许多最近的工作集中在通过后处理校准方法减轻校准不足，这些方法旨在获得校准概率而不牺牲预训练模型的分类性能。本研究总结并分类了以往工作为三大类策略：直观设计方法、分箱方法以及基于理想校准形式的方法。通过理论和实践分析，我们指出了以往方法中的十个常见局限性。针对这些局限性，我们提出了一种称为h-校准的概率学习框架，理论上为标准校准构建了一个带有边界条件的等效学习形式。在此基础上，我们设计了一个简单而有效的后处理校准算法。我们的方法不仅克服了所识别的十种局限性，而且在广泛的实验中表现出色，远优于传统方法。我们进一步从理论上和实验上分析了我们的学习目标与传统适当评分规则的关系和优势。总体而言，我们提出的方法为基础校准提供了一个近似等价可微目标，揭示了计算统计与标准校准理论边界特性之间的对应性和收敛性。我们的研究成果在标准后处理校准基准测试中达到了最先进的性能，验证了其实用性。这项研究为相关领域学习可靠的似然性提供了有价值的参考。 

---
# OmniESI: A unified framework for enzyme-substrate interaction prediction with progressive conditional deep learning 

**Title (ZH)**: 全方位ESI：一种渐进条件深度学习的统一框架用于酶-底物相互作用预测 

**Authors**: Zhiwei Nie, Hongyu Zhang, Hao Jiang, Yutian Liu, Xiansong Huang, Fan Xu, Jie Fu, Zhixiang Ren, Yonghong Tian, Wen-Bin Zhang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17963)  

**Abstract**: Understanding and modeling enzyme-substrate interactions is crucial for catalytic mechanism research, enzyme engineering, and metabolic engineering. Although a large number of predictive methods have emerged, they do not incorporate prior knowledge of enzyme catalysis to rationally modulate general protein-molecule features that are misaligned with catalytic patterns. To address this issue, we introduce a two-stage progressive framework, OmniESI, for enzyme-substrate interaction prediction through conditional deep learning. By decomposing the modeling of enzyme-substrate interactions into a two-stage progressive process, OmniESI incorporates two conditional networks that respectively emphasize enzymatic reaction specificity and crucial catalysis-related interactions, facilitating a gradual feature modulation in the latent space from general protein-molecule domain to catalysis-aware domain. On top of this unified architecture, OmniESI can adapt to a variety of downstream tasks, including enzyme kinetic parameter prediction, enzyme-substrate pairing prediction, enzyme mutational effect prediction, and enzymatic active site annotation. Under the multi-perspective performance evaluation of in-distribution and out-of-distribution settings, OmniESI consistently delivered superior performance than state-of-the-art specialized methods across seven benchmarks. More importantly, the proposed conditional networks were shown to internalize the fundamental patterns of catalytic efficiency while significantly improving prediction performance, with only negligible parameter increases (0.16%), as demonstrated by ablation studies on key components. Overall, OmniESI represents a unified predictive approach for enzyme-substrate interactions, providing an effective tool for catalytic mechanism cracking and enzyme engineering with strong generalization and broad applicability. 

**Abstract (ZH)**: 全方位酶-底物相互作用预测框架OmniESI：基于条件深度学习的方法 

---
# Greedy Selection under Independent Increments: A Toy Model Analysis 

**Title (ZH)**: 贪婪选择下的独立增量：一种玩具模型分析 

**Authors**: Huitao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17941)  

**Abstract**: We study an iterative selection problem over N i.i.d. discrete-time stochastic processes with independent increments. At each stage, a fixed number of processes are retained based on their observed values. Under this simple model, we prove that the optimal strategy for selecting the final maximum-value process is to apply greedy selection at each stage. While the result relies on strong independence assumptions, it offers a clean justification for greedy heuristics in multi-stage elimination settings and may serve as a toy example for understanding related algorithms in high-dimensional applications. 

**Abstract (ZH)**: 我们研究了在N个独立同分布的离散时间随机过程上的迭代选择问题。在每一阶段，根据观察到的值保留固定数量的过程。在这一简单模型下，我们证明了选择最终最大值过程的最优策略是在每一阶段应用贪婪选择。虽然该结果依赖于强烈的独立性假设，但它为多阶段淘汰设置中的贪婪启发式方法提供了清晰的解释，并可能作为理解相关算法在高维应用中的一个示例。 

---
# Software Reuse in the Generative AI Era: From Cargo Cult Towards AI Native Software Engineering 

**Title (ZH)**: 生成式AI时代软件重用：从cargo cult到AI原生软件工程 

**Authors**: Tommi Mikkonen, Antero Taivalsaari  

**Link**: [PDF](https://arxiv.org/pdf/2506.17937)  

**Abstract**: Software development is currently under a paradigm shift in which artificial intelligence and generative software reuse are taking the center stage in software creation. Consequently, earlier software reuse practices and methods are rapidly being replaced by AI-assisted approaches in which developers place their trust on code that has been generated by artificial intelligence. This is leading to a new form of software reuse that is conceptually not all that different from cargo cult development. In this paper we discuss the implications of AI-assisted generative software reuse in the context of emerging "AI native" software engineering, bring forth relevant questions, and define a tentative research agenda and call to action for tackling some of the central issues associated with this approach. 

**Abstract (ZH)**: 人工智能辅助生成式软件重用在新兴“AI本位”软件工程中的影响：相关问题探讨与研究议程 

---
# When concept-based XAI is imprecise: Do people distinguish between generalisations and misrepresentations? 

**Title (ZH)**: 基于概念的解释性人工智能不精确时：人们能否区分一般化和误表征？ 

**Authors**: Romy Müller  

**Link**: [PDF](https://arxiv.org/pdf/2506.17936)  

**Abstract**: Concept-based explainable artificial intelligence (C-XAI) can help reveal the inner representations of AI models. Understanding these representations is particularly important in complex tasks like safety evaluation. Such tasks rely on high-level semantic information (e.g., about actions) to make decisions about abstract categories (e.g., whether a situation is dangerous). In this context, it may desirable for C-XAI concepts to show some variability, suggesting that the AI is capable of generalising beyond the concrete details of a situation. However, it is unclear whether people recognise and appreciate such generalisations and can distinguish them from other, less desirable forms of imprecision. This was investigated in an experimental railway safety scenario. Participants evaluated the performance of a simulated AI that evaluated whether traffic scenes involving people were dangerous. To explain these decisions, the AI provided concepts in the form of similar image snippets. These concepts differed in their match with the classified image, either regarding a highly relevant feature (i.e., relation to tracks) or a less relevant feature (i.e., actions). Contrary to the hypotheses, concepts that generalised over less relevant features led to ratings that were lower than for precisely matching concepts and comparable to concepts that systematically misrepresented these features. Conversely, participants were highly sensitive to imprecisions in relevant features. These findings cast doubts on whether people spontaneously recognise generalisations. Accordingly, they might not be able to infer from C-XAI concepts whether AI models have gained a deeper understanding of complex situations. 

**Abstract (ZH)**: 基于概念的可解释人工智能（C-XAI）可以揭示AI模型的内在表示。在复杂的任务如安全性评估中，理解这些表示尤为重要。此类任务依赖于高层次的语义信息（例如，关于行动的信息）来对抽象类别（例如，情况是否危险）作出决策。在这种情境下，C-XAI的概念可能需要表现出一定的灵活性，表明AI能够超越具体情境的细节进行泛化。然而，尚不清楚人们是否能够认识到并欣赏这些泛化，以及是否能够将其与其它形式的不精确区分开来。这一问题在一个实验性的铁路安全场景中进行了探究。参与者评估了一个模拟AI的表现，该AI评估交通场景（涉及人员）是否危险。为解释这些决策，AI提供了以类似图像片段形式呈现的概念。这些概念在与分类图像的匹配程度上有所不同，前者是关于高度相关的特征（即，轨道的关系），后者是关于较不相关的特征（即，行动）。与假设相反，泛化于较不相关特征的概念导致的评分低于精确匹配的概念，并且这些评分与系统地误导这些特征的概念相当。相反，参与者对相关特征的不精确性非常敏感。这些发现对人们是否自发地认识到泛化的能力提出了疑问。因此，他们可能无法从C-XAI概念中推断出AI模型是否对复杂情境有了更深入的理解。 

---
# A GenAI System for Improved FAIR Independent Biological Database Integration 

**Title (ZH)**: 一个用于改善FAIR独立生物数据库集成的GenAI系统 

**Authors**: Syed N. Sakib, Kallol Naha, Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17934)  

**Abstract**: Life sciences research increasingly requires identifying, accessing, and effectively processing data from an ever-evolving array of information sources on the Linked Open Data (LOD) network. This dynamic landscape places a significant burden on researchers, as the quality of query responses depends heavily on the selection and semantic integration of data sources --processes that are often labor-intensive, error-prone, and costly. While the adoption of FAIR (Findable, Accessible, Interoperable, and Reusable) data principles has aimed to address these challenges, barriers to efficient and accurate scientific data processing persist.
In this paper, we introduce FAIRBridge, an experimental natural language-based query processing system designed to empower scientists to discover, access, and query biological databases, even when they are not FAIR-compliant. FAIRBridge harnesses the capabilities of AI to interpret query intents, map them to relevant databases described in scientific literature, and generate executable queries via intelligent resource access plans. The system also includes robust tools for mitigating low-quality query processing, ensuring high fidelity and responsiveness in the information delivered.
FAIRBridge's autonomous query processing framework enables users to explore alternative data sources, make informed choices at every step, and leverage community-driven crowd curation when needed. By providing a user-friendly, automated hypothesis-testing platform in natural English, FAIRBridge significantly enhances the integration and processing of scientific data, offering researchers a powerful new tool for advancing their inquiries. 

**Abstract (ZH)**: 生命科学研究 increasingly requires identifying, accessing, and effectively processing data from an ever-evolving array of information sources on the Linked Open Data (LOD) network. This dynamic landscape places a significant burden on researchers, as the quality of query responses depends heavily on the selection and semantic integration of data sources --processes that are often labor-intensive, error-prone, and costly. While the adoption of FAIR (Findable, Accessible, Interoperable, and Reusable) data principles has aimed to address these challenges, barriers to efficient and accurate scientific data processing persist.
在这种动态的环境中，研究人员需要不断识别、访问并有效地处理链接开放数据（LOD）网络上不断演化的各种信息源的数据。高质量的查询响应依赖于数据来源的选择和语义集成——这些过程往往是劳动密集型、易出错和昂贵的。虽然采用可获取性（Findable）、可访问性（Accessible）、互操作性（Interoperable）和可重用性（Reusable）（FAIR）的数据原则旨在解决这些挑战，但高效的和准确的科学数据处理仍面临障碍。
In this paper, we introduce FAIRBridge, an experimental natural language-based query processing system designed to empower scientists to discover, access, and query biological databases, even when they are not FAIR-compliant. FAIRBridge harnesses the capabilities of AI to interpret query intents, map them to relevant databases described in scientific literature, and generate executable queries via intelligent resource access plans. The system also includes robust tools for mitigating low-quality query processing, ensuring high fidelity and responsiveness in the information delivered.
在这篇论文中，我们介绍了FAIRBridge，这是一种实验性的基于自然语言的查询处理系统，旨在使科学家能够发现、访问和查询生物数据库，即使这些数据库不符合FAIR准则也是如此。FAIRBridge利用人工智能的能力来解释查询意图，将其映射到科学文献中描述的相关数据库，并通过智能资源访问计划生成可执行查询。该系统还包含强大的工具来降低低质量查询处理的影响，确保交付的信息具有高保真度和高响应性。
FAIRBridge's autonomous query processing framework enables users to explore alternative data sources, make informed choices at every step, and leverage community-driven crowd curation when needed. By providing a user-friendly, automated hypothesis-testing platform in natural English, FAIRBridge significantly enhances the integration and processing of scientific data, offering researchers a powerful new tool for advancing their inquiries.
FAIRBridge的自动查询处理框架使用户能够探索替代数据源，在每一步做出知情选择，并在需要时利用社区驱动的众包进行内容审查。通过提供一个用户友好、自动化的自然英文假设检验平台，FAIRBridge显著增强了科学数据的集成和处理，为研究人员提供了促进其研究的强大新工具。 

---
# IDAL: Improved Domain Adaptive Learning for Natural Images Dataset 

**Title (ZH)**: IDAL: 改进的领域自适应学习方法用于自然图像数据集 

**Authors**: Ravi Kant Gupta, Shounak Das, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17931)  

**Abstract**: We present a novel approach for unsupervised domain adaptation (UDA) for natural images. A commonly-used objective for UDA schemes is to enhance domain alignment in representation space even if there is a domain shift in the input space. Existing adversarial domain adaptation methods may not effectively align different domains of multimodal distributions associated with classification problems. Our approach has two main features. Firstly, its neural architecture uses the deep structure of ResNet and the effective separation of scales of feature pyramidal network (FPN) to work with both content and style features. Secondly, it uses a combination of a novel loss function and judiciously selected existing loss functions to train the network architecture. This tailored combination is designed to address challenges inherent to natural images, such as scale, noise, and style shifts, that occur on top of a multi-modal (multi-class) distribution. The combined loss function not only enhances model accuracy and robustness on the target domain but also speeds up training convergence. Our proposed UDA scheme generalizes better than state-of-the-art for CNN-based methods on Office-Home, Office-31, and VisDA-2017 datasets and comaparable for DomainNet dataset. 

**Abstract (ZH)**: 我们提出了一种新的无监督领域适应（UDA）方法用于自然图像。 

---
# ASTER: Adaptive Spatio-Temporal Early Decision Model for Dynamic Resource Allocation 

**Title (ZH)**: ASTER：适应性时空早期决策模型用于动态资源分配 

**Authors**: Shulun Chen, Wei Shao, Flora D. Salim, Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.17929)  

**Abstract**: Supporting decision-making has long been a central vision in the field of spatio-temporal intelligence. While prior work has improved the timeliness and accuracy of spatio-temporal forecasting, converting these forecasts into actionable strategies remains a key challenge. A main limitation is the decoupling of the prediction and the downstream decision phases, which can significantly degrade the downstream efficiency. For example, in emergency response, the priority is successful resource allocation and intervention, not just incident prediction. To this end, it is essential to propose an Adaptive Spatio-Temporal Early Decision model (ASTER) that reforms the forecasting paradigm from event anticipation to actionable decision support. This framework ensures that information is directly used for decision-making, thereby maximizing overall effectiveness. Specifically, ASTER introduces a new Resource-aware Spatio-Temporal interaction module (RaST) that adaptively captures long- and short-term dependencies under dynamic resource conditions, producing context-aware spatiotemporal representations. To directly generate actionable decisions, we further design a Preference-oriented decision agent (Poda) based on multi-objective reinforcement learning, which transforms predictive signals into resource-efficient intervention strategies by deriving optimal actions under specific preferences and dynamic constraints. Experimental results on four benchmark datasets demonstrate the state-of-the-art performance of ASTER in improving both early prediction accuracy and resource allocation outcomes across six downstream metrics. 

**Abstract (ZH)**: 支持决策长久以来一直是时空智能领域的核心愿景。尽管前期工作在提升时空预测的时效性和准确性方面取得进展，但将这些预测转化为 actionable 策略仍然是一项关键挑战。主要限制在于预测阶段与下游决策阶段的脱节，这会显著降低下游效率。例如，在应急响应中，优先级是成功分配和干预资源，而不仅仅是事件预测。为此，提出一种自适应时空早期决策模型（ASTER）以改革预测范式，从事件预见到行动支持决策。该框架确保信息直接用于决策，从而最大化整体效果。具体而言，ASTER 引入了一种资源感知时空交互模块（RaST），能够在动态资源条件下自适应捕捉长期和短期依赖性，生成上下文感知的时空表示。为进一步生成 actionable 决策，设计了一种基于多目标强化学习的偏好导向决策智能体（Poda），通过在特定偏好和动态约束下推导出最优行动，将预测信号转化为高效的资源干预策略。在四个基准数据集上的实验结果表明，ASTER 在六个下游指标上均能提高早期预测准确性和资源分配效果，展现出最先进的性能。 

---
# Permutation Equivariant Model-based Offline Reinforcement Learning for Auto-bidding 

**Title (ZH)**: 基于置换不变模型的离线强化学习自动出价方法 

**Authors**: Zhiyu Mou, Miao Xu, Wei Chen, Rongquan Bai, Chuan Yu, Jian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17919)  

**Abstract**: Reinforcement learning (RL) for auto-bidding has shifted from using simplistic offline simulators (Simulation-based RL Bidding, SRLB) to offline RL on fixed real datasets (Offline RL Bidding, ORLB). However, ORLB policies are limited by the dataset's state space coverage, offering modest gains. While SRLB expands state coverage, its simulator-reality gap risks misleading policies. This paper introduces Model-based RL Bidding (MRLB), which learns an environment model from real data to bridge this gap. MRLB trains policies using both real and model-generated data, expanding state coverage beyond ORLB. To ensure model reliability, we propose: 1) A permutation equivariant model architecture for better generalization, and 2) A robust offline Q-learning method that pessimistically penalizes model errors. These form the Permutation Equivariant Model-based Offline RL (PE-MORL) algorithm. Real-world experiments show that PE-MORL outperforms state-of-the-art auto-bidding methods. 

**Abstract (ZH)**: 基于模型的强化学习自动竞价：从模拟基在线学习到模型基离线强化学习 

---
# Cause-Effect Driven Optimization for Robust Medical Visual Question Answering with Language Biases 

**Title (ZH)**: 基于因果驱动的优化方法以应对语言偏差的鲁棒医学视觉问答 

**Authors**: Huanjia Zhu, Yishu Liu, Xiaozhao Fang, Guangming Lu, Bingzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17903)  

**Abstract**: Existing Medical Visual Question Answering (Med-VQA) models often suffer from language biases, where spurious correlations between question types and answer categories are inadvertently established. To address these issues, we propose a novel Cause-Effect Driven Optimization framework called CEDO, that incorporates three well-established mechanisms, i.e., Modality-driven Heterogeneous Optimization (MHO), Gradient-guided Modality Synergy (GMS), and Distribution-adapted Loss Rescaling (DLR), for comprehensively mitigating language biases from both causal and effectual perspectives. Specifically, MHO employs adaptive learning rates for specific modalities to achieve heterogeneous optimization, thus enhancing robust reasoning capabilities. Additionally, GMS leverages the Pareto optimization method to foster synergistic interactions between modalities and enforce gradient orthogonality to eliminate bias updates, thereby mitigating language biases from the effect side, i.e., shortcut bias. Furthermore, DLR is designed to assign adaptive weights to individual losses to ensure balanced learning across all answer categories, effectively alleviating language biases from the cause side, i.e., imbalance biases within datasets. Extensive experiments on multiple traditional and bias-sensitive benchmarks consistently demonstrate the robustness of CEDO over state-of-the-art competitors. 

**Abstract (ZH)**: 基于因果驱动优化的医疗视觉问答语言偏差缓解框架：MHO、GMS和DLR联合优化(CEDO) 

---
# NestQuant: Post-Training Integer-Nesting Quantization for On-Device DNN 

**Title (ZH)**: NestQuant: On-Device DNN整数嵌套量化后训练方法 

**Authors**: Jianhang Xie, Chuntao Ding, Xiaqing Li, Shenyuan Ren, Yidong Li, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17870)  

**Abstract**: Deploying quantized deep neural network (DNN) models with resource adaptation capabilities on ubiquitous Internet of Things (IoT) devices to provide high-quality AI services can leverage the benefits of compression and meet multi-scenario resource requirements. However, existing dynamic/mixed precision quantization requires retraining or special hardware, whereas post-training quantization (PTQ) has two limitations for resource adaptation: (i) The state-of-the-art PTQ methods only provide one fixed bitwidth model, which makes it challenging to adapt to the dynamic resources of IoT devices; (ii) Deploying multiple PTQ models with diverse bitwidths consumes large storage resources and switching overheads. To this end, this paper introduces a resource-friendly post-training integer-nesting quantization, i.e., NestQuant, for on-device quantized model switching on IoT devices. The proposed NestQuant incorporates the integer weight decomposition, which bit-wise splits quantized weights into higher-bit and lower-bit weights of integer data types. It also contains a decomposed weights nesting mechanism to optimize the higher-bit weights by adaptive rounding and nest them into the original quantized weights. In deployment, we can send and store only one NestQuant model and switch between the full-bit/part-bit model by paging in/out lower-bit weights to adapt to resource changes and reduce consumption. Experimental results on the ImageNet-1K pretrained DNNs demonstrated that the NestQuant model can achieve high performance in top-1 accuracy, and reduce in terms of data transmission, storage consumption, and switching overheads. In particular, the ResNet-101 with INT8 nesting INT6 can achieve 78.1% and 77.9% accuracy for full-bit and part-bit models, respectively, and reduce switching overheads by approximately 78.1% compared with diverse bitwidths PTQ models. 

**Abstract (ZH)**: 在物联网设备上部署具有资源适应能力的量化深度神经网络模型以提供高质量的AI服务可以利用压缩带来的好处并满足多场景的资源需求。然而，现有的动态/混合精度量化需要进行重新训练或特殊的硬件支持，而后训练量化（PTQ）方法有两个限制：(i) 当前最先进的PTQ方法只能提供一种固定的位宽模型，这使得适应物联网设备的动态资源具有挑战性；(ii) 部署具有多种位宽的多个PTQ模型消耗大量的存储资源和切换开销。为此，本文介绍了一种资源友好的后训练整数嵌套量化方法，即NestQuant，用于物联网设备上的量化模型在设备上的切换。提出的NestQuant结合了整数权重分解，按位将量化权重拆分为较高位和较低位的整数数据类型权重。它还包含一个拆分权重嵌套机制，通过自适应舍入优化较高位权重，并将其嵌套到原始量化权重中。在部署过程中，我们可以发送和存储一个NestQuant模型，并通过调入/调出较低位权重在全精度/部分精度模型之间切换，以适应资源变化并减少消耗。实验结果表明，NestQuant模型在Top-1准确率、数据传输量、存储消耗和切换开销方面均可实现高性能。特别是，ResNet-101在嵌套INT8为INT6的配置下，全精度和部分精度模型的准确率分别为78.1%和77.9%，并且与具有多种位宽的PTQ模型相比，切换开销减少了约78.1%。 

---
# In-Context Learning Strategies Emerge Rationally 

**Title (ZH)**: 上下文学习策略理性涌现 

**Authors**: Daniel Wurgaft, Ekdeep Singh Lubana, Core Francisco Park, Hidenori Tanaka, Gautam Reddy, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17859)  

**Abstract**: Recent work analyzing in-context learning (ICL) has identified a broad set of strategies that describe model behavior in different experimental conditions. We aim to unify these findings by asking why a model learns these disparate strategies in the first place. Specifically, we start with the observation that when trained to learn a mixture of tasks, as is popular in the literature, the strategies learned by a model for performing ICL can be captured by a family of Bayesian predictors: a memorizing predictor, which assumes a discrete prior on the set of seen tasks, and a generalizing predictor, wherein the prior matches the underlying task distribution. Adopting the lens of rational analysis from cognitive science, where a learner's behavior is explained as an optimal adaptation to data given computational constraints, we develop a hierarchical Bayesian framework that almost perfectly predicts Transformer next token predictions throughout training without assuming access to its weights. Under this framework, pretraining is viewed as a process of updating the posterior probability of different strategies, and its inference-time behavior as a posterior-weighted average over these strategies' predictions. Our framework draws on common assumptions about neural network learning dynamics, which make explicit a tradeoff between loss and complexity among candidate strategies: beyond how well it explains the data, a model's preference towards implementing a strategy is dictated by its complexity. This helps explain well-known ICL phenomena, while offering novel predictions: e.g., we show a superlinear trend in the timescale for transition to memorization as task diversity is increased. Overall, our work advances an explanatory and predictive account of ICL grounded in tradeoffs between strategy loss and complexity. 

**Abstract (ZH)**: 近期关于上下文学习（ICL）的研究已识别出一系列描述模型行为的策略。我们旨在通过探讨模型为何会在一开始就学习这些不同的策略来统一这些发现。具体而言，我们从文献中流行的做法——即训练模型学习任务混合开始，观察到模型在进行ICL时所学到的策略可以用一组贝叶斯预测器来捕捉：一种记忆预测器，假设看到的任务集合具有离散prior；一种泛化预测器，其中prior与任务分布匹配。借鉴认知科学中的理性分析视角，即在计算资源受限的情况下，学习者的行为被视为对数据的最佳适应，我们开发了一个分层贝叶斯框架，几乎可以在训练过程中完美预测Transformer的下一个标记预测，而不假设对权重的访问。在此框架下，预训练被视为更新不同策略后验概率的过程，而推理时的行为则视为这些策略预测的后验加权平均。我们的框架借鉴了关于神经网络学习动力学的常见假设，明确展示了候选策略之间的损失与复杂性之间的权衡：模型偏好实现某一策略不仅取决于其解释数据的能力，还取决于其复杂性。这有助于解释已知的ICL现象，并提出新的预测，如随着任务多样性增加，过渡到记忆的比例呈现超线性趋势。总体而言，我们的工作为ICL提供了一个基于策略损失与复杂性权衡的解释性和预测性框架。 

---
# Pathway-based Progressive Inference (PaPI) for Energy-Efficient Continual Learning 

**Title (ZH)**: 基于路径的渐进推理（PaPI）for 能效持续学习 

**Authors**: Suyash Gaurav, Jukka Heikkonen, Jatin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2506.17848)  

**Abstract**: Continual learning systems face the dual challenge of preventing catastrophic forgetting while maintaining energy efficiency, particularly in resource-constrained environments. This paper introduces Pathway-based Progressive Inference (PaPI), a novel theoretical framework that addresses these challenges through a mathematically rigorous approach to pathway selection and adaptation. We formulate continual learning as an energy-constrained optimization problem and provide formal convergence guarantees for our pathway routing mechanisms. Our theoretical analysis demonstrates that PaPI achieves an $\mathcal{O}(K)$ improvement in the stability-plasticity trade-off compared to monolithic architectures, where $K$ is the number of pathways. We derive tight bounds on forgetting rates using Fisher Information Matrix analysis and prove that PaPI's energy consumption scales with the number of active parameters rather than the total model size. Comparative theoretical analysis shows that PaPI provides stronger guarantees against catastrophic forgetting than Elastic Weight Consolidation (EWC) while maintaining better energy efficiency than both EWC and Gradient Episodic Memory (GEM). Our experimental validation confirms these theoretical advantages across multiple benchmarks, demonstrating PaPI's effectiveness for continual learning in energy-constrained settings. Our codes are available at this https URL. 

**Abstract (ZH)**: 基于路径的渐进推理（PaPI）：在能量约束环境中预防灾难性遗忘与保持能量效率的新理论框架 

---
# A Comparative Study of Open-Source Libraries for Synthetic Tabular Data Generation: SDV vs. SynthCity 

**Title (ZH)**: 开源库合成表格数据生成比较研究：SDV vs. SynthCity 

**Authors**: Cristian Del Gobbo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17847)  

**Abstract**: High-quality training data is critical to the performance of machine learning models, particularly Large Language Models (LLMs). However, obtaining real, high-quality data can be challenging, especially for smaller organizations and early-stage startups. Synthetic data generators provide a promising solution by replicating the statistical and structural properties of real data while preserving privacy and scalability. This study evaluates the performance of six tabular synthetic data generators from two widely used open-source libraries: SDV (Gaussian Copula, CTGAN, TVAE) and Synthicity (Bayesian Network, CTGAN, TVAE). Using a real-world dataset from the UCI Machine Learning Repository, comprising energy consumption and environmental variables from Belgium, we simulate a low-data regime by training models on only 1,000 rows. Each generator is then tasked with producing synthetic datasets under two conditions: a 1:1 (1,000 rows) and a 1:10 (10,000 rows) input-output ratio. Evaluation is conducted using two criteria: statistical similarity, measured via classical statistics and distributional metrics; and predictive utility, assessed using a "Train on Synthetic, Test on Real" approach with four regression models. While statistical similarity remained consistent across models in both scenarios, predictive utility declined notably in the 1:10 case. The Bayesian Network from Synthicity achieved the highest fidelity in both scenarios, while TVAE from SDV performed best in predictive tasks under the 1:10 setting. Although no significant performance gap was found between the two libraries, SDV stands out for its superior documentation and ease of use, making it more accessible for practitioners. 

**Abstract (ZH)**: 高质量训练数据对机器学习模型，特别是大型语言模型(LLMs)的性能至关重要。然而，获取真实的高质量数据对于较小的组织和早期初创企业来说颇具挑战。合成数据生成器通过复制真实数据的统计和结构属性，同时保护隐私和可扩展性，提供了一个有前景的解决方案。本研究评估了来自两个广泛使用的开源库SDV（Gaussian Copula、CTGAN、TVAE）和Synthicity（Bayesian Network、CTGAN、TVAE）的六种表结构合成数据生成器的表现。使用来自UCI机器学习仓库的实际数据集，该数据集包含来自比利时的能源消耗和环境变量，模拟低数据环境，仅使用1,000行进行模型训练。然后在1:1（1,000行）和1:10（10,000行）输入-输出比率的条件下，每种生成器生成合成数据集。评估标准包括统计相似性，通过经典统计方法和分布度量测量；以及预测效用，使用“在合成数据上训练，在实际数据上测试”的方法，评估四种回归模型。尽管两种情况下模型的统计相似性保持一致，但在1:10情况下预测效用显著下降。Synthicity的Bayesian Network在两种情况下均表现出最高的保真度，而SDV的TVAE在1:10设置下的预测任务中表现最佳。虽然两个库之间未发现显著性能差异，但SDV因其更好的文档和易用性脱颖而出，使其更适用于实践者。 

---
# THCM-CAL: Temporal-Hierarchical Causal Modelling with Conformal Calibration for Clinical Risk Prediction 

**Title (ZH)**: THCM-CAL: 时间分层因果建模结合校准化验证的临床风险预测 

**Authors**: Xin Zhang, Qiyu Wei, Yingjie Zhu, Fanyi Wu, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17844)  

**Abstract**: Automated clinical risk prediction from electronic health records (EHRs) demands modeling both structured diagnostic codes and unstructured narrative notes. However, most prior approaches either handle these modalities separately or rely on simplistic fusion strategies that ignore the directional, hierarchical causal interactions by which narrative observations precipitate diagnoses and propagate risk across admissions. In this paper, we propose THCM-CAL, a Temporal-Hierarchical Causal Model with Conformal Calibration. Our framework constructs a multimodal causal graph where nodes represent clinical entities from two modalities: Textual propositions extracted from notes and ICD codes mapped to textual descriptions. Through hierarchical causal discovery, THCM-CAL infers three clinically grounded interactions: intra-slice same-modality sequencing, intra-slice cross-modality triggers, and inter-slice risk propagation. To enhance prediction reliability, we extend conformal prediction to multi-label ICD coding, calibrating per-code confidence intervals under complex co-occurrences. Experimental results on MIMIC-III and MIMIC-IV demonstrate the superiority of THCM-CAL. 

**Abstract (ZH)**: 电子健康记录（EHRs）中的自动化临床风险预测需要同时建模结构化诊断代码和非结构化病历笔记。然而，大多数先前的方法要么单独处理这些模态，要么依赖于忽略叙述性观察如何引发诊断及其在住院之间传播风险的简单的融合策略。本文提出了一种名为THCM-CAL的时空层次因果模型，该模型构建了一个多模态因果图，通过层次因果发现推断出三条临床相关交互：同一切片内同模态序列、同一切片内跨模态触发以及跨切片的风险传播。为了增强预测可靠性，我们将校准预测扩展到多标签ICD编码，针对复杂共现现象calibrate每个代码的置信区间。在MIMIC-III和MIMIC-IV上的实验结果表明THCM-CAL的优越性。 

---
# Causal Spherical Hypergraph Networks for Modelling Social Uncertainty 

**Title (ZH)**: 因果球形超图网络用于建模社会不确定性 

**Authors**: Anoushka Harit, Zhongtian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.17840)  

**Abstract**: Human social behaviour is governed by complex interactions shaped by uncertainty, causality, and group dynamics. We propose Causal Spherical Hypergraph Networks (Causal-SphHN), a principled framework for socially grounded prediction that jointly models higher-order structure, directional influence, and epistemic uncertainty. Our method represents individuals as hyperspherical embeddings and group contexts as hyperedges, capturing semantic and relational geometry. Uncertainty is quantified via Shannon entropy over von Mises-Fisher distributions, while temporal causal dependencies are identified using Granger-informed subgraphs. Information is propagated through an angular message-passing mechanism that respects belief dispersion and directional semantics. Experiments on SNARE (offline networks), PHEME (online discourse), and AMIGOS (multimodal affect) show that Causal-SphHN improves predictive accuracy, robustness, and calibration over strong baselines. Moreover, it enables interpretable analysis of influence patterns and social ambiguity. This work contributes a unified causal-geometric approach for learning under uncertainty in dynamic social environments. 

**Abstract (ZH)**: 基于因果关系的球面超图网络：在动态社交环境中的不确定性学习 

---
# Actionable Interpretability via Causal Hypergraphs: Unravelling Batch Size Effects in Deep Learning 

**Title (ZH)**: 基于因果超图的行为可解释性：揭开深度学习中的批量大小效应 

**Authors**: Zhongtian Sun, Anoushka Harit, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2506.17826)  

**Abstract**: While the impact of batch size on generalisation is well studied in vision tasks, its causal mechanisms remain underexplored in graph and text domains. We introduce a hypergraph-based causal framework, HGCNet, that leverages deep structural causal models (DSCMs) to uncover how batch size influences generalisation via gradient noise, minima sharpness, and model complexity. Unlike prior approaches based on static pairwise dependencies, HGCNet employs hypergraphs to capture higher-order interactions across training dynamics. Using do-calculus, we quantify direct and mediated effects of batch size interventions, providing interpretable, causally grounded insights into optimisation. Experiments on citation networks, biomedical text, and e-commerce reviews show that HGCNet outperforms strong baselines including GCN, GAT, PI-GNN, BERT, and RoBERTa. Our analysis reveals that smaller batch sizes causally enhance generalisation through increased stochasticity and flatter minima, offering actionable interpretability to guide training strategies in deep learning. This work positions interpretability as a driver of principled architectural and optimisation choices beyond post hoc analysis. 

**Abstract (ZH)**: 基于超图的因果框架HGCNet：通过梯度噪声、最小值锋利度和模型复杂性探究批量大小对泛化的影响 

---
# CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning 

**Title (ZH)**: 文化MERT：跨文化音乐表示学习的持续预训练 

**Authors**: Angelos-Nikolaos Kanatas, Charilaos Papaioannou, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2506.17818)  

**Abstract**: Recent advances in music foundation models have improved audio representation learning, yet their effectiveness across diverse musical traditions remains limited. We introduce CultureMERT-95M, a multi-culturally adapted foundation model developed to enhance cross-cultural music representation learning and understanding. To achieve this, we propose a two-stage continual pre-training strategy that integrates learning rate re-warming and re-decaying, enabling stable adaptation even with limited computational resources. Training on a 650-hour multi-cultural data mix, comprising Greek, Turkish, and Indian music traditions, results in an average improvement of 4.9% in ROC-AUC and AP across diverse non-Western music auto-tagging tasks, surpassing prior state-of-the-art, with minimal forgetting on Western-centric benchmarks. We further investigate task arithmetic, an alternative approach to multi-cultural adaptation that merges single-culture adapted models in the weight space. Task arithmetic performs on par with our multi-culturally trained model on non-Western auto-tagging tasks and shows no regression on Western datasets. Cross-cultural evaluation reveals that single-culture models transfer with varying effectiveness across musical traditions, whereas the multi-culturally adapted model achieves the best overall performance. To support research on world music representation learning, we publicly release CultureMERT-95M and CultureMERT-TA-95M, fostering the development of more culturally aware music foundation models. 

**Abstract (ZH)**: Recent Advances in Multi-Culturally Adapted Music Foundation Models: Enhancing Cross-Cultural Music Representation Learning and Understanding 

---
# Reimagining Parameter Space Exploration with Diffusion Models 

**Title (ZH)**: 重塑参数空间探索：基于扩散模型的方法 

**Authors**: Lijun Zhang, Xiao Liu, Hui Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17807)  

**Abstract**: Adapting neural networks to new tasks typically requires task-specific fine-tuning, which is time-consuming and reliant on labeled data. We explore a generative alternative that produces task-specific parameters directly from task identity, eliminating the need for task-specific training. To this end, we propose using diffusion models to learn the underlying structure of effective task-specific parameter space and synthesize parameters on demand. Once trained, the task-conditioned diffusion model can generate specialized weights directly from task identifiers. We evaluate this approach across three scenarios: generating parameters for a single seen task, for multiple seen tasks, and for entirely unseen tasks. Experiments show that diffusion models can generate accurate task-specific parameters and support multi-task interpolation when parameter subspaces are well-structured, but fail to generalize to unseen tasks, highlighting both the potential and limitations of this generative solution. 

**Abstract (ZH)**: 将神经网络适应新任务通常需要针对特定任务进行微调，这耗时且依赖于标记数据。我们探索了一种生成性替代方案，可以从任务身份直接生成任务特定参数，从而消除针对特定任务的训练需求。为此，我们提出使用扩散模型来学习有效的任务特定参数空间的基本结构，并按需合成参数。训练完成后，条件于任务的扩散模型可以直接从任务标识符生成专门的权重。我们在此方法中评估了三种场景：为单个已见过的任务生成参数、为多个已见过的任务生成参数以及为完全未见过的任务生成参数。实验结果显示，当参数子空间结构良好时，扩散模型可以生成准确的任务特定参数，并支持多任务插值，但无法泛化到未见过的任务，这突显了该生成性解决方案的潜力和局限性。 

---
# Toward Autonomous UI Exploration: The UIExplorer Benchmark 

**Title (ZH)**: 向自主UI探索迈进：UIExplorer基准测试 

**Authors**: Andrei Cristian Nica, Akshaya Vishnu Kudlu Shanbhogue, Harshil Shah, Aleix Cambray, Tudor Berariu, Lucas Maystre, David Barber  

**Link**: [PDF](https://arxiv.org/pdf/2506.17779)  

**Abstract**: Autonomous agents must know how to explore user interfaces (UIs) for reliable task solving, yet systematic evaluation of this crucial phase is lacking. We introduce UIExplore-Bench, the first benchmark explicitly dedicated to UI exploration. The benchmark evaluates agents with either Structured mode (granting access to layout information like DOM trees) or Screen mode (relying on GUI-only observations such as screenshots and human-like mouse/keyboard interactions) across three levels in a standardized GitLab sandbox environment. We formalize exploration as the process of maximizing the set of actionable UI components discovered and propose a metric, human-normalized UI-Functionalities Observed (hUFO), to quantify the effectiveness of exploration. Our results show that UIExplore-AlGo achieves the leading mean hUFO scores, reaching up to 77.2% of human performance in Structured mode and 59.0% in Screen mode at 2,000 steps, particularly excelling at the Sparse level. The results highlight the relevance of our benchmark, as current agents show a substantial performance gap compared to one hour of human expert exploration, indicating ample room for future advancements. We publicly release the benchmark environment, an exploration dataset, and an evaluation suite to catalyze research into efficient UI exploration strategies and their downstream applications, such as experience-driven task completion and automated training data generation. 

**Abstract (ZH)**: UIExplore-Bench：面向UI探索的首个基准 

---
# Machine Learning Model Integration with Open World Temporal Logic for Process Automation 

**Title (ZH)**: 基于开放世界时序逻辑的机器学习模型集成与过程自动化 

**Authors**: Dyuman Aditya, Colton Payne, Mario Leiva, Paulo Shakarian  

**Link**: [PDF](https://arxiv.org/pdf/2506.17776)  

**Abstract**: Recent advancements in Machine Learning (ML) have yielded powerful models capable of extracting structured information from diverse and complex data sources. However, a significant challenge lies in translating these perceptual or extractive outputs into actionable, reasoned decisions within complex operational workflows. To address these challenges, this paper introduces a novel approach that integrates the outputs from various machine learning models directly with the PyReason framework, an open-world temporal logic programming reasoning engine. PyReason's foundation in generalized annotated logic allows for the seamless incorporation of real-valued outputs (e.g., probabilities, confidence scores) from diverse ML models, treating them as truth intervals within its logical framework. Crucially, PyReason provides mechanisms, implemented in Python, to continuously poll ML model outputs, convert them into logical facts, and dynamically recompute the minimal model, ensuring real-tine adaptive decision-making. Furthermore, its native support for temporal reasoning, knowledge graph integration, and fully explainable interface traces enables sophisticated analysis over time-sensitive process data and existing organizational knowledge. By combining the strengths of perception and extraction from ML models with the logical deduction and transparency of PyReason, we aim to create a powerful system for automating complex processes. This integration finds utility across numerous domains, including manufacturing, healthcare, and business operations. 

**Abstract (ZH)**: 近期机器学习领域的进展产生了强大的模型，能够从多样化和复杂的数据源中提取结构化信息。然而，在复杂的操作工作流中将这些感知或提取输出转化为可行的决策仍面临重大挑战。为应对这些挑战，本文介绍了一种新的方法，该方法将各种机器学习模型的输出直接集成到PyReason框架中，PyReason是一个开放世界的时序逻辑编程推理引擎。PyReason基于通用标注逻辑，允许无缝地将来自不同机器学习模型的实值输出（例如，概率、置信分数）纳入其逻辑框架中，视作其逻辑框架中的真度区间。PyReason提供了用Python实现的机制，可以持续轮询机器学习模型输出，将其转换为逻辑事实，并动态重新计算最小模型，从而确保实时自适应决策。此外，其原生支持时间推理、知识图谱整合以及完全可解释的接口跟踪特性，使其能够对时间敏感的过程数据和现有组织知识进行复杂的分析。通过结合机器学习模型感知和提取的优点以及PyReason的逻辑推理和透明性，我们旨在创建一个强大的自动化复杂过程系统。这一整合在制造业、医疗保健和业务运营等多个领域具有广泛的应用。 

---
# Residual Connection-Enhanced ConvLSTM for Lithium Dendrite Growth Prediction 

**Title (ZH)**: 基于残差连接增强的ConvLSTM锂枝晶生长预测 

**Authors**: Hosung Lee, Byeongoh Hwang, Dasan Kim, Myungjoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17756)  

**Abstract**: The growth of lithium dendrites significantly impacts the performance and safety of rechargeable batteries, leading to short circuits and capacity degradation. This study proposes a Residual Connection-Enhanced ConvLSTM model to predict dendrite growth patterns with improved accuracy and computational efficiency. By integrating residual connections into ConvLSTM, the model mitigates the vanishing gradient problem, enhances feature retention across layers, and effectively captures both localized dendrite growth dynamics and macroscopic battery behavior. The dataset was generated using a phase-field model, simulating dendrite evolution under varying conditions. Experimental results show that the proposed model achieves up to 7% higher accuracy and significantly reduces mean squared error (MSE) compared to conventional ConvLSTM across different voltage conditions (0.1V, 0.3V, 0.5V). This highlights the effectiveness of residual connections in deep spatiotemporal networks for electrochemical system modeling. The proposed approach offers a robust tool for battery diagnostics, potentially aiding in real-time monitoring and optimization of lithium battery performance. Future research can extend this framework to other battery chemistries and integrate it with real-world experimental data for further validation 

**Abstract (ZH)**: 锂枝晶生长对可充电电池的性能和安全性有显著影响，导致短路和容量衰退。本研究提出了一种残差连接增强的ConvLSTM模型，以提高枝晶生长模式预测的准确性和计算效率。通过将残差连接整合到ConvLSTM中，该模型缓解了梯度消失问题，增强了跨层特征保留，并有效地捕获了局部枝晶生长动力学和宏观电池行为。数据集使用相场模型生成，模拟了不同条件下枝晶的演变。实验结果表明，所提出模型在不同电压条件下（0.1V、0.3V、0.5V）的准确率最高可提高7%，并显著降低了均方误差（MSE）与传统ConvLSTM相比。这强调了残差连接在电化学系统建模的深时空网络中的有效性。所提出的方法为电池诊断提供了 robust 工具，可能有助于实时监测和优化锂离子电池的性能。未来的研究可以将此框架扩展到其他电池化学体系，并将其与实际实验数据集成以进行进一步验证。 

---
# Resolving the Ti-V Phase Diagram Discrepancy with First-Principles Calculations and Bayesian Learning 

**Title (ZH)**: 基于第一性原理计算和贝叶斯学习解决Ti-V相图差异问题 

**Authors**: Timofei Miryashkin, Olga Klimanova, Alexander Shapeev  

**Link**: [PDF](https://arxiv.org/pdf/2506.17719)  

**Abstract**: Conflicting experiments disagree on whether the titanium-vanadium (Ti-V) binary alloy exhibits a body-centred cubic (BCC) miscibility gap or remains completely soluble. A leading hypothesis attributes the miscibility gap to oxygen contamination during alloy preparation. To resolve this controversy, we use an ab initio + machine-learning workflow that couples an actively-trained Moment Tensor Potential to Bayesian thermodynamic inference. Using this workflow, we obtain Ti-V binary system across the entire composition range, together with confidence intervals in the thermodynamic limit. The resulting diagram reproduces all experimental features, demonstrating the robustness of our approach, and clearly favors the variant with a BCC miscibility gap terminating at T = 980 K and c = 0.67. Because oxygen was excluded from simulations, the gap cannot be attributed to impurity effects, contradicting recent CALPHAD reassessments. 

**Abstract (ZH)**: 钛-钒（Ti-V）二元合金是否存在体心立方（BCC）共熔区间仍存在实验争议：从第一性原理+机器学习工作流探究Ti-V二元系统的相图及其热力学稳定性分析 

---
# Aged to Perfection: Machine-Learning Maps of Age in Conversational English 

**Title (ZH)**: 完美老化：对话英语中的年龄机器学习地图 

**Authors**: MingZe Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17708)  

**Abstract**: The study uses the British National Corpus 2014, a large sample of contemporary spoken British English, to investigate language patterns across different age groups. Our research attempts to explore how language patterns vary between different age groups, exploring the connection between speaker demographics and linguistic factors such as utterance duration, lexical diversity, and word choice. By merging computational language analysis and machine learning methodologies, we attempt to uncover distinctive linguistic markers characteristic of multiple generations and create prediction models that can consistently estimate the speaker's age group from various aspects. This work contributes to our knowledge of sociolinguistic diversity throughout the life of modern British speech. 

**Abstract (ZH)**: 本研究使用2014年英国国家语料库，这一当代英式英语的大规模样本，探讨不同年龄组的语言模式。我们的研究尝试探索不同年龄组之间语言模式的差异，探究讲话者人口统计学特征与语用长度、词汇多样性、词汇选择等语言因素之间的联系。通过结合计算语言分析和机器学习方法，我们试图发现多个代际特征性的语言标记，并建立可以从多方面一致估计讲话者年龄组的预测模型。本项工作增进了我们对现代英式口语 生命周期中社会语言多样性 的理解。 

---
# Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems 

**Title (ZH)**: 强化多场景学习中用户的兴趣演化推荐系统 

**Authors**: Zhijian Feng, Wenhao Zheng, Xuanji Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17682)  

**Abstract**: In real-world recommendation systems, users would engage in variety scenarios, such as homepages, search pages, and related recommendation pages. Each of these scenarios would reflect different aspects users focus on. However, the user interests may be inconsistent in different scenarios, due to differences in decision-making processes and preference expression. This variability complicates unified modeling, making multi-scenario learning a significant challenge. To address this, we propose a novel reinforcement learning approach that models user preferences across scenarios by modeling user interest evolution across multiple scenarios. Our method employs Double Q-learning to enhance next-item prediction accuracy and optimizes contrastive learning loss using Q-value to make model performance better. Experimental results demonstrate that our approach surpasses state-of-the-art methods in multi-scenario recommendation tasks. Our work offers a fresh perspective on multi-scenario modeling and highlights promising directions for future research. 

**Abstract (ZH)**: 在现实世界的推荐系统中，用户会在多种场景中互动，如首页、搜索页面和相关推荐页面。每个场景都会反映用户关注的不同方面。然而，由于决策过程和偏好表达的不同，用户的兴趣在不同场景中可能不一致。这种变化性使统一建模变得复杂，使多场景学习成为一个重大挑战。为应对这一挑战，我们提出了一种新颖的强化学习方法，通过建模跨场景的用户兴趣演变来建模用户的偏好。我们的方法采用双重Q学习提高下一项预测的准确性，并使用Q值优化对比学习损失以提升模型性能。实验结果表明，我们的方法在多场景推荐任务中超过了最先进的方法。我们的工作为多场景建模提供了新的视角，并指出了未来研究的有前景方向。 

---
# Enhancing Stress-Strain Predictions with Seq2Seq and Cross-Attention based on Small Punch Test 

**Title (ZH)**: 基于小冲程试验的Seq2Seq与跨注意力机制增强应力-应变预测 

**Authors**: Zhengni Yang, Rui Yang, Weijian Han, Qixin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17680)  

**Abstract**: This paper introduces a novel deep-learning approach to predict true stress-strain curves of high-strength steels from small punch test (SPT) load-displacement data. The proposed approach uses Gramian Angular Field (GAF) to transform load-displacement sequences into images, capturing spatial-temporal features and employs a Sequence-to-Sequence (Seq2Seq) model with an LSTM-based encoder-decoder architecture, enhanced by multi-head cross-attention to improved accuracy. Experimental results demonstrate that the proposed approach achieves superior prediction accuracy, with minimum and maximum mean absolute errors of 0.15 MPa and 5.58 MPa, respectively. The proposed method offers a promising alternative to traditional experimental techniques in materials science, enhancing the accuracy and efficiency of true stress-strain relationship predictions. 

**Abstract (ZH)**: 本文提出了一种新颖的深度学习方法，从小凸缘试验(SPT)载荷-位移数据预测高强度钢的真实应力-应变曲线。该方法使用Gramian角场(GAF)将载荷-位移序列转换为图像，捕捉空间-时间特征，并采用基于LSTM的编码器-解码器架构的Sequence-to-Sequence (Seq2Seq)模型，通过多头交叉注意力机制提高预测精度。实验结果表明，所提出的方法在预测精度方面表现出色，最小和最大均方绝对误差分别为0.15 MPa和5.58 MPa。所提出的方法为材料科学中的传统实验技术提供了有希望的替代方案，提高了真实应力-应变关系预测的准确性和效率。 

---
# FaithfulSAE: Towards Capturing Faithful Features with Sparse Autoencoders without External Dataset Dependencies 

**Title (ZH)**: FaithfulSAE：如何在无需外部数据集依赖的情况下捕捉忠实特征的稀疏自编码器 

**Authors**: Seonglae Cho, Harryn Oh, Donghyun Lee, Luis Eduardo Rodrigues Vieira, Andrew Bermingham, Ziad El Sayed  

**Link**: [PDF](https://arxiv.org/pdf/2506.17673)  

**Abstract**: Sparse Autoencoders (SAEs) have emerged as a promising solution for decomposing large language model representations into interpretable features. However, Paulo and Belrose (2025) have highlighted instability across different initialization seeds, and Heap et al. (2025) have pointed out that SAEs may not capture model-internal features. These problems likely stem from training SAEs on external datasets - either collected from the Web or generated by another model - which may contain out-of-distribution (OOD) data beyond the model's generalisation capabilities. This can result in hallucinated SAE features, which we term "Fake Features", that misrepresent the model's internal activations. To address these issues, we propose FaithfulSAE, a method that trains SAEs on the model's own synthetic dataset. Using FaithfulSAEs, we demonstrate that training SAEs on less-OOD instruction datasets results in SAEs being more stable across seeds. Notably, FaithfulSAEs outperform SAEs trained on web-based datasets in the SAE probing task and exhibit a lower Fake Feature Ratio in 5 out of 7 models. Overall, our approach eliminates the dependency on external datasets, advancing interpretability by better capturing model-internal features while highlighting the often neglected importance of SAE training datasets. 

**Abstract (ZH)**: FaithfulSAE: Training Sparse Autoencoders on Model-Specific Synthetic Data for Enhanced Interpretability 

---
# Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection 

**Title (ZH)**: 自适应多提示对比网络用于少样本域外检测 

**Authors**: Xiang Fang, Arvind Easwaran, Blaise Genest  

**Link**: [PDF](https://arxiv.org/pdf/2506.17633)  

**Abstract**: Out-of-distribution (OOD) detection attempts to distinguish outlier samples to prevent models trained on the in-distribution (ID) dataset from producing unavailable outputs. Most OOD detection methods require many IID samples for training, which seriously limits their real-world applications. To this end, we target a challenging setting: few-shot OOD detection, where {Only a few {\em labeled ID} samples are available.} Therefore, few-shot OOD detection is much more challenging than the traditional OOD detection setting. Previous few-shot OOD detection works ignore the distinct diversity between different classes. In this paper, we propose a novel network: Adaptive Multi-prompt Contrastive Network (AMCN), which adapts the ID-OOD separation boundary by learning inter- and intra-class distribution. To compensate for the absence of OOD and scarcity of ID {\em image samples}, we leverage CLIP, connecting text with images, engineering learnable ID and OOD {\em textual prompts}. Specifically, we first generate adaptive prompts (learnable ID prompts, label-fixed OOD prompts and label-adaptive OOD prompts). Then, we generate an adaptive class boundary for each class by introducing a class-wise threshold. Finally, we propose a prompt-guided ID-OOD separation module to control the margin between ID and OOD prompts. Experimental results show that AMCN outperforms other state-of-the-art works. 

**Abstract (ZH)**: 离分布（OOD）检测旨在区分异常样本，防止在分布内（ID）数据集上训练的模型产生不可用的输出。大多数OOD检测方法需要大量独立同分布（IID）样本进行训练，这严重限制了它们的实际应用。为此，我们针对一个具有挑战性的场景进行研究：少量样本的OOD检测，其中仅可用少量标记的ID样本。因此，少量样本的OOD检测比传统的OOD检测场景更具挑战性。以往的少量样本的OOD检测工作忽略了不同类别之间的独特多样性。在本文中，我们提出了一种新颖的网络：自适应多提示对比网络（AMCN），通过学习类别间的和类别内的分布来适应ID-OOD分离边界。为了弥补OOD样本不足和ID图像样本稀缺的问题，我们利用CLIP，将文本与图像进行连接，工程化生成可学习的ID和OOD文本提示。具体来说，我们首先生成自适应提示（可学习的ID提示、标签固定的不同类别自适应OOD提示）。然后，我们通过引入类别的阈值为每个类别生成自适应类边界。最后，我们提出了一种提示引导的ID-OOD分离模块，以控制ID提示与OOD提示之间的差距。实验结果显示，AMCN优于其他现有最佳方法。 

---
# Exploiting Efficiency Vulnerabilities in Dynamic Deep Learning Systems 

**Title (ZH)**: 利用动态深度学习系统中的效率漏洞 

**Authors**: Ravishka Rathnasuriya, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17621)  

**Abstract**: The growing deployment of deep learning models in real-world environments has intensified the need for efficient inference under strict latency and resource constraints. To meet these demands, dynamic deep learning systems (DDLSs) have emerged, offering input-adaptive computation to optimize runtime efficiency. While these systems succeed in reducing cost, their dynamic nature introduces subtle and underexplored security risks. In particular, input-dependent execution pathways create opportunities for adversaries to degrade efficiency, resulting in excessive latency, energy usage, and potential denial-of-service in time-sensitive deployments. This work investigates the security implications of dynamic behaviors in DDLSs and reveals how current systems expose efficiency vulnerabilities exploitable by adversarial inputs. Through a survey of existing attack strategies, we identify gaps in the coverage of emerging model architectures and limitations in current defense mechanisms. Building on these insights, we propose to examine the feasibility of efficiency attacks on modern DDLSs and develop targeted defenses to preserve robustness under adversarial conditions. 

**Abstract (ZH)**: 动态深度学习系统中动态行为的安全性影响及其防御策略 

---
# Optimizing Mastery Learning by Fast-Forwarding Over-Practice Steps 

**Title (ZH)**: 优化掌握学习通过跳过过度练习步骤实现快速增长 

**Authors**: Meng Xia, Robin Schmucker, Conrad Borchers, Vincent Aleven  

**Link**: [PDF](https://arxiv.org/pdf/2506.17577)  

**Abstract**: Mastery learning improves learning proficiency and efficiency. However, the overpractice of skills--students spending time on skills they have already mastered--remains a fundamental challenge for tutoring systems. Previous research has reduced overpractice through the development of better problem selection algorithms and the authoring of focused practice tasks. However, few efforts have concentrated on reducing overpractice through step-level adaptivity, which can avoid resource-intensive curriculum redesign. We propose and evaluate Fast-Forwarding as a technique that enhances existing problem selection algorithms. Based on simulation studies informed by learner models and problem-solving pathways derived from real student data, Fast-Forwarding can reduce overpractice by up to one-third, as it does not require students to complete problem-solving steps if all remaining pathways are fully mastered. Fast-Forwarding is a flexible method that enhances any problem selection algorithm, though its effectiveness is highest for algorithms that preferentially select difficult problems. Therefore, our findings suggest that while Fast-Forwarding may improve student practice efficiency, the size of its practical impact may also depend on students' ability to stay motivated and engaged at higher levels of difficulty. 

**Abstract (ZH)**: Mastery学习提高学习成效和效率，但技能的过度练习——学生花费时间在已掌握的技能上——仍然是辅导系统的一个基本挑战。先前的研究通过开发更好的问题选择算法和编写针对性练习任务来减少过度练习。然而，较少的研究集中于通过步骤级别的适应性来减少过度练习，这种方式可以避免耗费资源的课程再设计。我们提出并评估了“快进”作为一种增强现有问题选择算法的技术。基于由学习者模型和真实学生数据推导出的学习路径的仿真研究，“快进”可以通过跳过所有剩余路径均已完全掌握的问题解决步骤来减少多达三分之一的过度练习。作为一种灵活的方法，“快进”可以增强任何问题选择算法，尽管其有效性在优先选择困难问题的算法中最高。因此，我们的研究结果表明，“快进”可能会提高学生练习效率，但其实际影响的大小也可能取决于学生是否能在较高难度水平上保持动力和参与。 

---
# Towards Zero-Shot Coordination between Teams of Agents: The N-XPlay Framework 

**Title (ZH)**: 零样本智能体团队之间的协调：N-XPlay框架 

**Authors**: Ava Abderezaei, Chi-Hui Lin, Joseph Miceli, Naren Sivagnanadasan, Stéphane Aroca-Ouellette, Jake Brawer, Alessandro Roncone  

**Link**: [PDF](https://arxiv.org/pdf/2506.17560)  

**Abstract**: Zero-shot coordination (ZSC) -- the ability to collaborate with unfamiliar partners -- is essential to making autonomous agents effective teammates. Existing ZSC methods evaluate coordination capabilities between two agents who have not previously interacted. However, these scenarios do not reflect the complexity of real-world multi-agent systems, where coordination often involves a hierarchy of sub-groups and interactions between teams of agents, known as Multi-Team Systems (MTS). To address this gap, we first introduce N-player Overcooked, an N-agent extension of the popular two-agent ZSC benchmark, enabling evaluation of ZSC in N-agent scenarios. We then propose N-XPlay for ZSC in N-agent, multi-team settings. Comparison against Self-Play across two-, three- and five-player Overcooked scenarios, where agents are split between an ``ego-team'' and a group of unseen collaborators shows that agents trained with N-XPlay are better able to simultaneously balance ``intra-team'' and ``inter-team'' coordination than agents trained with SP. 

**Abstract (ZH)**: 零样本协调（ZSC）——与不熟悉的合作伙伴协作的能力——是使自主代理成为有效队友的关键。现有的ZSC方法评估的是两个未曾互动的代理之间的协作能力。然而，这些场景未能反映现实世界多代理系统中的复杂性，在这些系统中，协调往往涉及子组层次结构和团队间代理的互动，称为多团队系统（MTS）。为了解决这一差距，我们首先引入了N-player Overcooked，这是流行的两代理ZSC基准的N代理扩展，使ZSC在N代理场景中的评估成为可能。然后，我们提出了N-XPlay，用于N代理和多团队设置中的ZSC。在两个、三个和五个玩家的Overcooked场景中，将代理分配给“自我团队”和一组未见合作者，相比使用自我博弈（Self-Play, SP）训练的代理，使用N-XPlay训练的代理能更好地同时平衡“团队内”和“团队间”协调。 

---
# ConsumerBench: Benchmarking Generative AI Applications on End-User Devices 

**Title (ZH)**: ConsumerBench: 在终端用户设备上基准测试生成式AI应用 

**Authors**: Yile Gu, Rohan Kadekodi, Hoang Nguyen, Keisuke Kamahori, Yiyu Liu, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2506.17538)  

**Abstract**: The recent shift in Generative AI (GenAI) applications from cloud-only environments to end-user devices introduces new challenges in resource management, system efficiency, and user experience. This paper presents ConsumerBench, a comprehensive benchmarking framework designed to evaluate the system efficiency and response time of GenAI models running on end-user devices. Unlike existing benchmarks that assume exclusive model access on dedicated GPUs, ConsumerBench simulates realistic multi-application scenarios executing concurrently on constrained hardware. Furthermore, ConsumerBench supports customizable workflows that simulate complex tasks requiring coordination among multiple applications. ConsumerBench captures both application-level metrics, including latency and Service Level Objective (SLO) attainment, and system-level metrics like CPU/GPU utilization and memory bandwidth. Through extensive experiments, ConsumerBench reveals inefficiencies in resource sharing, unfair scheduling under greedy allocation, and performance pitfalls of static model server configurations. The paper also provides practical insights for model developers and system designers, highlighting the benefits of custom kernels tailored to consumer-grade GPU architectures and the value of implementing SLO-aware scheduling strategies. 

**Abstract (ZH)**: 近期生成式人工智能（GenAI）应用从云环境向终端用户设备的转移引入了新的资源管理、系统效率和用户体验挑战。本文介绍了一种名为ConsumerBench的全面基准测试框架，用于评估运行在终端用户设备上的GenAI模型的系统效率和响应时间。与现有假设独家模型访问专用GPU的基准不同，ConsumerBench模拟了在受约束硬件上并发执行多个应用程序的现实场景。此外，ConsumerBench支持可定制的工作流，模拟需要多个应用程序协调的复杂任务。ConsumerBench捕获应用程序级别的指标，包括延迟和SLI（服务级别指标）达成情况，以及系统级别的指标，如CPU/GPU利用率和内存带宽。通过广泛的实验，ConsumerBench揭示了资源共享的低效性、贪婪分配下的不公平调度以及静态模型服务器配置的性能陷阱。本文还为模型开发人员和系统设计师提供了实用见解，强调了针对消费级GPU架构的定制内核以及实施具有SLI意识的调度策略的价值。 

---
# Exploring Strategies for Personalized Radiation Therapy Part I Unlocking Response-Related Tumor Subregions with Class Activation Mapping 

**Title (ZH)**: 探索个性化放疗策略 第一部分 通过类别激活映射解锁与响应相关的肿瘤亚区域 

**Authors**: Hao Peng, Steve Jiang, Robert Timmerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17536)  

**Abstract**: Personalized precision radiation therapy requires more than simple classification, it demands the identification of prognostic, spatially informative features and the ability to adapt treatment based on individual response. This study compares three approaches for predicting treatment response: standard radiomics, gradient based features, and convolutional neural networks enhanced with Class Activation Mapping. We analyzed 69 brain metastases from 39 patients treated with Gamma Knife radiosurgery. An integrated autoencoder classifier model was used to predict whether tumor volume would shrink by more than 20 percent at a three months follow up, framed as a binary classification task. The results highlight their strength in hierarchical feature extraction and the classifiers discriminative capacity. Among the models, pixel wise CAM provides the most detailed spatial insight, identifying lesion specific regions rather than relying on fixed patterns, demonstrating strong generalization. In non responding lesions, the activated regions may indicate areas of radio resistance. Pixel wise CAM outperformed both radiomics and gradient based methods in classification accuracy. Moreover, its fine grained spatial features allow for alignment with cellular level data, supporting biological validation and deeper understanding of heterogeneous treatment responses. Although further validation is necessary, these findings underscore the promise in guiding personalized and adaptive radiotherapy strategies for both photon and particle therapies. 

**Abstract (ZH)**: 个性化精准放疗要求的不仅仅是简单的分类，还需识别预后性、空间信息丰富的特征，并能够根据个体反应调整治疗方案。本研究比较了三种预测治疗反应的方法：标准放射omics、梯度特征以及结合Class Activation Mapping的卷积神经网络。我们分析了39名患者经伽玛刀放射手术治疗的69个脑转移瘤，使用集成自编码分类模型预测肿瘤体积在三个月随访时是否减少超过20%，将其视为二分类任务。结果强调了这些方法在层次特征提取和分类器判别能力方面的优势。在各种模型中，像素级CAM提供了最详细的空间洞察，能够识别病变特异性区域，而不是依赖固定模式，显示出较强的泛化能力。在未响应的病灶中，激活区域可能表明存在放疗抵抗。像素级CAM在分类准确性上优于放射omics和基于梯度的方法，其细粒度的空间特征允许与细胞水平数据对齐，支持生物学验证和对异质治疗反应的深入理解。尽管需要进一步验证，但这些发现强调了指导个性化和自适应放疗策略的潜力，适用于光子和粒子疗法。 

---
# Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning 

**Title (ZH)**: 多语言语音数据中的数据质量问题：需要社会语言学意识和主动语言规划 

**Authors**: Mingfei Lau, Qian Chen, Yeming Fang, Tingting Xu, Tongzhou Chen, Pavel Golik  

**Link**: [PDF](https://arxiv.org/pdf/2506.17525)  

**Abstract**: Our quality audit for three widely used public multilingual speech datasets - Mozilla Common Voice 17.0, FLEURS, and VoxPopuli - shows that in some languages, these datasets suffer from significant quality issues. We believe addressing these issues will make these datasets more useful as training and evaluation sets, and improve downstream models. We divide these quality issues into two categories: micro-level and macro-level. We find that macro-level issues are more prevalent in less institutionalized, often under-resourced languages. We provide a case analysis of Taiwanese Southern Min (nan_tw) that highlights the need for proactive language planning (e.g. orthography prescriptions, dialect boundary definition) and enhanced data quality control in the process of Automatic Speech Recognition (ASR) dataset creation. We conclude by proposing guidelines and recommendations to mitigate these issues in future dataset development, emphasizing the importance of sociolinguistic awareness in creating robust and reliable speech data resources. 

**Abstract (ZH)**: 我们对广泛使用的三个公共多语言语音数据集——Mozilla Common Voice 17.0、FLEURS和VoxPopuli的质量审计显示，在某些语言中，这些数据集存在显著的质量问题。我们认为解决这些问题将使这些数据集在作为训练和评估集时更具有用途，并改善下游模型。我们将这些质量问题分为微观层面和宏观层面两类。我们发现，宏观层面的问题在制度化程度较低、often under-resourced 的语言中更为常见。我们通过对台语（nan_tw）进行案例分析，强调了自动语音识别（ASR）数据集创建过程中需要积极的语言规划（例如拼写规范、方言边界定义）和增强的数据质量控制。最后，我们提出了指导原则和建议，以在未来的数据集开发中减少这些问题，并强调在创造稳健可靠的声音数据资源时需要提高社会语言学意识。 

---
# A Survey of State Representation Learning for Deep Reinforcement Learning 

**Title (ZH)**: 深度强化学习中状态表示学习综述 

**Authors**: Ayoub Echchahed, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2506.17518)  

**Abstract**: Representation learning methods are an important tool for addressing the challenges posed by complex observations spaces in sequential decision making problems. Recently, many methods have used a wide variety of types of approaches for learning meaningful state representations in reinforcement learning, allowing better sample efficiency, generalization, and performance. This survey aims to provide a broad categorization of these methods within a model-free online setting, exploring how they tackle the learning of state representations differently. We categorize the methods into six main classes, detailing their mechanisms, benefits, and limitations. Through this taxonomy, our aim is to enhance the understanding of this field and provide a guide for new researchers. We also discuss techniques for assessing the quality of representations, and detail relevant future directions. 

**Abstract (ZH)**: 无监督表示学习方法在无模型在线设置下处理顺序决策问题中复杂观测空间挑战的应用：方法分类与前景展望 

---
# Mapping the Evolution of Research Contributions using KnoVo 

**Title (ZH)**: 使用KnoVo映射研究贡献的演化 

**Authors**: Sajratul Y. Rubaiat, Syed N. Sakib, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17508)  

**Abstract**: This paper presents KnoVo (Knowledge Evolution), an intelligent framework designed for quantifying and analyzing the evolution of research novelty in the scientific literature. Moving beyond traditional citation analysis, which primarily measures impact, KnoVo determines a paper's novelty relative to both prior and subsequent work within its multilayered citation network. Given a target paper's abstract, KnoVo utilizes Large Language Models (LLMs) to dynamically extract dimensions of comparison (e.g., methodology, application, dataset). The target paper is then compared to related publications along these same extracted dimensions. This comparative analysis, inspired by tournament selection, yields quantitative novelty scores reflecting the relative improvement, equivalence, or inferiority of the target paper in specific aspects. By aggregating these scores and visualizing their progression, for instance, through dynamic evolution graphs and comparative radar charts, KnoVo facilitates researchers not only to assess originality and identify similar work, but also to track knowledge evolution along specific research dimensions, uncover research gaps, and explore cross-disciplinary connections. We demonstrate these capabilities through a detailed analysis of 20 diverse papers from multiple scientific fields and report on the performance of various open-source LLMs within the KnoVo framework. 

**Abstract (ZH)**: 基于知识进化的智能框架：科研创新演化的量化与分析 

---
# From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training 

**Title (ZH)**: 从通见到精通：通过大规模预训练实现作曲家风格的符号音乐生成 

**Authors**: Mingyang Yao, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17497)  

**Abstract**: Despite progress in controllable symbolic music generation, data scarcity remains a challenge for certain control modalities. Composer-style music generation is a prime example, as only a few pieces per composer are available, limiting the modeling of both styles and fundamental music elements (e.g., melody, chord, rhythm). In this paper, we investigate how general music knowledge learned from a broad corpus can enhance the mastery of specific composer styles, with a focus on piano piece generation. Our approach follows a two-stage training paradigm. First, we pre-train a REMI-based music generation model on a large corpus of pop, folk, and classical music. Then, we fine-tune it on a small, human-verified dataset from four renowned composers, namely Bach, Mozart, Beethoven, and Chopin, using a lightweight adapter module to condition the model on style indicators. To evaluate the effectiveness of our approach, we conduct both objective and subjective evaluations on style accuracy and musicality. Experimental results demonstrate that our method outperforms ablations and baselines, achieving more precise composer-style modeling and better musical aesthetics. Additionally, we provide observations on how the model builds music concepts from the generality pre-training and refines its stylistic understanding through the mastery fine-tuning. 

**Abstract (ZH)**: 尽管在可控符号音乐生成方面取得了进展，但某些控制模式仍面临数据稀缺的挑战。作曲家风格的音乐生成就是一个典型的例子，每个作曲家的乐曲数量有限，限制了风格和基本音乐元素（如旋律、和弦、节奏）的建模。在本文中，我们研究广泛乐曲知识如何增强特定作曲家风格的专业技能，重点关注钢琴曲的生成。我们的方法采用两阶段训练 paradigm。首先，我们基于大量流行、民间和古典音乐的语料库对一种基于REMI的音乐生成模型进行预训练。然后，我们使用一个轻量级适配模块对模型进行微调，以便根据风格指标对模型进行条件限制，微调数据集来自四位著名的作曲家：巴赫、莫扎特、贝多芬和肖邦，由人类验证。为了评估我们方法的有效性，我们在风格准确性和音乐性方面进行了客观和主观评估。实验结果表明，我们的方法优于简化模型和基线，实现了更精确的作曲家风格建模和更好的音乐美学。此外，我们提供了关于模型如何从广泛预训练构建音乐概念并在专业微调中细化其风格理解的观察。 

---
# Exploring Strategies for Personalized Radiation Therapy Part II Predicting Tumor Drift Patterns with Diffusion Models 

**Title (ZH)**: 探索个性化学术放射治疗策略 II 基于扩散模型预测肿瘤位移模式 

**Authors**: Hao Peng, Steve Jiang, Robert Timmerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17491)  

**Abstract**: Radiation therapy outcomes are decided by two key parameters, dose and timing, whose best values vary substantially across patients. This variability is especially critical in the treatment of brain cancer, where fractionated or staged stereotactic radiosurgery improves safety compared to single fraction approaches, but complicates the ability to predict treatment response. To address this challenge, we employ Personalized Ultra-fractionated Stereotactic Adaptive Radiotherapy (PULSAR), a strategy that dynamically adjusts treatment based on how each tumor evolves over time. However, the success of PULSAR and other adaptive approaches depends on predictive tools that can guide early treatment decisions and avoid both overtreatment and undertreatment. However, current radiomics and dosiomics models offer limited insight into the evolving spatial and temporal patterns of tumor response. To overcome these limitations, we propose a novel framework using Denoising Diffusion Implicit Models (DDIM), which learns data-driven mappings from pre to post treatment imaging. In this study, we developed single step and iterative denoising strategies and compared their performance. The results show that diffusion models can effectively simulate patient specific tumor evolution and localize regions associated with treatment response. The proposed strategy provides a promising foundation for modeling heterogeneous treatment response and enabling early, adaptive interventions, paving the way toward more personalized and biologically informed radiotherapy. 

**Abstract (ZH)**: 个性化超分割立体适形自适应放疗（PULSAR）及其在放疗中的应用：基于去噪扩散隐模型的数据驱动肿瘤演变模拟 

---
# FedNAMs: Performing Interpretability Analysis in Federated Learning Context 

**Title (ZH)**: FedNAMs：在联邦学习背景下进行可解释性分析 

**Authors**: Amitash Nanda, Sree Bhargavi Balija, Debashis Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17466)  

**Abstract**: Federated learning continues to evolve but faces challenges in interpretability and explainability. To address these challenges, we introduce a novel approach that employs Neural Additive Models (NAMs) within a federated learning framework. This new Federated Neural Additive Models (FedNAMs) approach merges the advantages of NAMs, where individual networks concentrate on specific input features, with the decentralized approach of federated learning, ultimately producing interpretable analysis results. This integration enhances privacy by training on local data across multiple devices, thereby minimizing the risks associated with data centralization and improving model robustness and generalizability. FedNAMs maintain detailed, feature-specific learning, making them especially valuable in sectors such as finance and healthcare. They facilitate the training of client-specific models to integrate local updates, preserve privacy, and mitigate concerns related to centralization. Our studies on various text and image classification tasks, using datasets such as OpenFetch ML Wine, UCI Heart Disease, and Iris, show that FedNAMs deliver strong interpretability with minimal accuracy loss compared to traditional Federated Deep Neural Networks (DNNs). The research involves notable findings, including the identification of critical predictive features at both client and global levels. Volatile acidity, sulfates, and chlorides for wine quality. Chest pain type, maximum heart rate, and number of vessels for heart disease. Petal length and width for iris classification. This approach strengthens privacy and model efficiency and improves interpretability and robustness across diverse datasets. Finally, FedNAMs generate insights on causes of highly and low interpretable features. 

**Abstract (ZH)**: federated learning在可解释性和可说明性方面不断演进但仍面临挑战。为应对这些挑战，我们提出了一种新颖的方法，该方法在联邦学习框架中采用了神经加性模型（NAMs）。这一新的联邦神经加性模型（FedNAMs）方法结合了NAMs的优势，即各个网络专注于特定的输入特征，以及联邦学习的分散化方法，从而产生可解释的分析结果。这种整合通过在多设备上的本地数据上进行训练增强了隐私性，减少了数据集中化带来的风险，并提高了模型的稳健性和泛化能力。FedNAMs保持了详细的、特征特定的训练，使其在金融和医疗保健等行业尤为有价值。它们促进了客户端特定模型的训练，以整合局部更新、保护隐私并缓解集中化的关切。我们在各种文本和图像分类任务上进行的研究，使用了如OpenFetch ML Wine、UCI Heart Disease和Iris等数据集，表明FedNAMs在可解释性方面表现出色，且相对于传统的联邦深度神经网络（DNNs）仅有轻微的准确性损失。这项研究包括重要发现，如识别出葡萄酒质量的临界预测特征（挥发酸、硫酸盐和氯化物），心脏病的特征（胸痛类型、最大心率和血管数量），以及鸢尾花分类的特征（花瓣长度和宽度）。这种方法增强了隐私和模型效率，并在多种数据集上改进了可解释性和鲁棒性。最后，FedNAMs还提供了关于高可解释性和低可解释性特征原因的洞见。 

---
# AI based Content Creation and Product Recommendation Applications in E-commerce: An Ethical overview 

**Title (ZH)**: 基于AI的内容创作与产品推荐应用在电子商务中的伦理概述 

**Authors**: Aditi Madhusudan Jain, Ayush Jain  

**Link**: [PDF](https://arxiv.org/pdf/2506.17370)  

**Abstract**: As e-commerce rapidly integrates artificial intelligence for content creation and product recommendations, these technologies offer significant benefits in personalization and efficiency. AI-driven systems automate product descriptions, generate dynamic advertisements, and deliver tailored recommendations based on consumer behavior, as seen in major platforms like Amazon and Shopify. However, the widespread use of AI in e-commerce raises crucial ethical challenges, particularly around data privacy, algorithmic bias, and consumer autonomy. Bias -- whether cultural, gender-based, or socioeconomic -- can be inadvertently embedded in AI models, leading to inequitable product recommendations and reinforcing harmful stereotypes. This paper examines the ethical implications of AI-driven content creation and product recommendations, emphasizing the need for frameworks to ensure fairness, transparency, and need for more established and robust ethical standards. We propose actionable best practices to remove bias and ensure inclusivity, such as conducting regular audits of algorithms, diversifying training data, and incorporating fairness metrics into AI models. Additionally, we discuss frameworks for ethical conformance that focus on safeguarding consumer data privacy, promoting transparency in decision-making processes, and enhancing consumer autonomy. By addressing these issues, we provide guidelines for responsibly utilizing AI in e-commerce applications for content creation and product recommendations, ensuring that these technologies are both effective and ethically sound. 

**Abstract (ZH)**: 电子商务中基于人工智能的内容创作与产品推荐的伦理影响：公平、透明与包容的最佳实践 

---
# Speeding up Local Optimization in Vehicle Routing with Tensor-based GPU Acceleration 

**Title (ZH)**: 基于张量的GPU加速在车辆路线问题中的局部优化加速 

**Authors**: Zhenyu Lei, Jin-Kao Hao, Qinghua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17357)  

**Abstract**: Local search plays a central role in many effective heuristic algorithms for the vehicle routing problem (VRP) and its variants. However, neighborhood exploration is known to be computationally expensive and time consuming, especially for large instances or problems with complex constraints. In this study, we explore a promising direction to address this challenge by introducing an original tensor-based GPU acceleration method designed to speed up the commonly used local search operators in vehicle routing. By using an attribute-based representation, the method offers broad extensibility, making it applicable to different VRP variants. Its low-coupling architecture, with intensive computations completely offloaded to the GPU, ensures seamless integration in various local search-based algorithms and frameworks, leading to significant improvements in computational efficiency and potentially improved solution quality. Through comparative experiments on benchmark instances of three routing problems, we demonstrate the substantial computational advantages of the proposed approach over traditional CPU-based implementations. We also provide a detailed analysis of the strengths and limitations of the method, providing valuable insights into its performance characteristics and identifying potential bottlenecks in practical applications. These findings contribute to a better understanding and suggest directions for future improvements. 

**Abstract (ZH)**: 局部搜索在车辆路线问题及其变体的许多有效启发式算法中发挥着核心作用。然而，邻域探索计算昂贵且耗时，尤其是在大型实例或具有复杂约束的问题中。本研究通过引入一种基于张量的GPU加速方法，旨在加速车辆路线中常用的地方搜索操作符，以解决这一挑战。该方法通过属性表示提供了广泛的可扩展性，使其适用于不同的车辆路线变体。其低耦合架构将密集计算完全卸载到GPU上，确保在各种基于局部搜索的算法和框架中无缝集成，从而大大提高计算效率并有可能提高解的质量。通过在三种路由问题的标准测试实例上进行比较实验，我们展示了所提出方法相对于传统CPU实现的显著计算优势。我们还详细分析了该方法的优势和局限性，提供了对其性能特征的见解，并确定了其实用应用中的潜在瓶颈。这些发现有助于更好地理解并为未来改进指明方向。 

---
# CUBA: Controlled Untargeted Backdoor Attack against Deep Neural Networks 

**Title (ZH)**: CUBA: 受控无目标后门攻击针对深度神经网络 

**Authors**: Yinghao Wu, Liyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17350)  

**Abstract**: Backdoor attacks have emerged as a critical security threat against deep neural networks in recent years. The majority of existing backdoor attacks focus on targeted backdoor attacks, where trigger is strongly associated to specific malicious behavior. Various backdoor detection methods depend on this inherent property and shows effective results in identifying and mitigating such targeted attacks. However, a purely untargeted attack in backdoor scenarios is, in some sense, self-weakening, since the target nature is what makes backdoor attacks so powerful. In light of this, we introduce a novel Constrained Untargeted Backdoor Attack (CUBA), which combines the flexibility of untargeted attacks with the intentionality of targeted attacks. The compromised model, when presented with backdoor images, will classify them into random classes within a constrained range of target classes selected by the attacker. This combination of randomness and determinedness enables the proposed untargeted backdoor attack to natively circumvent existing backdoor defense methods. To implement the untargeted backdoor attack under controlled flexibility, we propose to apply logit normalization on cross-entropy loss with flipped one-hot labels. By constraining the logit during training, the compromised model will show a uniform distribution across selected target classes, resulting in controlled untargeted attack. Extensive experiments demonstrate the effectiveness of the proposed CUBA on different datasets. 

**Abstract (ZH)**: 受约束的未 targeted 黑盒攻击 (CUBA): 结合灵活性与意图性以绕过现有防御方法 

---
# Advanced Game-Theoretic Frameworks for Multi-Agent AI Challenges: A 2025 Outlook 

**Title (ZH)**: 2025年视角下的高级博弈论框架与多智能体AI挑战 

**Authors**: Pavel Malinovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17348)  

**Abstract**: This paper presents a substantially reworked examination of how advanced game-theoretic paradigms can serve as a foundation for the next-generation challenges in Artificial Intelligence (AI), forecasted to arrive in or around 2025. Our focus extends beyond traditional models by incorporating dynamic coalition formation, language-based utilities, sabotage risks, and partial observability. We provide a set of mathematical formalisms, simulations, and coding schemes that illustrate how multi-agent AI systems may adapt and negotiate in complex environments. Key elements include repeated games, Bayesian updates for adversarial detection, and moral framing within payoff structures. This work aims to equip AI researchers with robust theoretical tools for aligning strategic interaction in uncertain, partially adversarial contexts. 

**Abstract (ZH)**: 这篇论文提出了一种大幅修改后的研究，探讨了先进的博弈理论范式如何为基础人工智能（AI）领域的下一代挑战提供基础，预计这些挑战将在2025年或其前后出现。我们的研究超越了传统模型，纳入了动态联盟形成、基于语言的效用、破坏风险以及部分可观测性。我们提供了一套数学形式化、仿真和编码方案，展示了多智能体AI系统如何在复杂环境中适应和谈判。关键要素包括重复博弈、贝叶斯更新以检测对手，以及在收益结构中的道德框架。本研究旨在为AI研究人员提供强大的理论工具，以在不确定且部分对抗的环境下对齐战略互动。 

---
# Distinguishing Predictive and Generative AI in Regulation 

**Title (ZH)**: 区分预测型和生成型人工智能在监管中的应用 

**Authors**: Jennifer Wang, Andrew Selbst, Solon Barocas, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.17347)  

**Abstract**: Over the past decade, policymakers have developed a set of regulatory tools to ensure AI development aligns with key societal goals. Many of these tools were initially developed in response to concerns with predictive AI and therefore encode certain assumptions about the nature of AI systems and the utility of certain regulatory approaches. With the advent of generative AI, however, some of these assumptions no longer hold, even as policymakers attempt to maintain a single regulatory target that covers both types of AI.
In this paper, we identify four distinct aspects of generative AI that call for meaningfully different policy responses. These are the generality and adaptability of generative AI that make it a poor regulatory target, the difficulty of designing effective evaluations, new legal concerns that change the ecosystem of stakeholders and sources of expertise, and the distributed structure of the generative AI value chain.
In light of these distinctions, policymakers will need to evaluate where the past decade of policy work remains relevant and where new policies, designed to address the unique risks posed by generative AI, are necessary. We outline three recommendations for policymakers to more effectively identify regulatory targets and leverage constraints across the broader ecosystem to govern generative AI. 

**Abstract (ZH)**: 过去十年，政策制定者开发了一套监管工具以确保人工智能的发展符合关键的社会目标。许多这些工具最初是针对预测型人工智能的担忧而设计的，因此包含了对人工智能系统性质和某些监管方法效益的假设。然而，伴随着生成型人工智能的到来，这些假设已不再适用，尽管政策制定者试图维持一个同时涵盖两类人工智能的单一监管目标。
在本文中，我们识别出生成型人工智能的四个独特方面，这需要有意义地不同的政策回应。这些方面包括生成型人工智能的一般性和适应性导致其成为不良的监管目标、设计有效评估的难度、新的法律问题改变了利益相关者和专业知识来源的生态系统、以及生成型人工智能价值链条的分布式结构。
鉴于这些区别，政策制定者需要评估过去十年政策工作的相关性和必要性更新政策，以应对生成型人工智能带来的独特风险。我们提出了三项针对政策制定者，旨在更有效地确定监管目标并利用更广泛生态系统中的制约因素来治理生成型人工智能的建议。 

---
# A Novel Multi-layer Task-centric and Data Quality Framework for Autonomous Driving 

**Title (ZH)**: 一种新型多层任务导向和数据质量框架 for 自动驾驶 

**Authors**: Yuhan Zhou, Haihua Chen, Kewei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2506.17346)  

**Abstract**: The next-generation autonomous vehicles (AVs), embedded with frequent real-time decision-making, will rely heavily on a large volume of multisource and multimodal data. In real-world settings, the data quality (DQ) of different sources and modalities usually varies due to unexpected environmental factors or sensor issues. However, both researchers and practitioners in the AV field overwhelmingly concentrate on models/algorithms while undervaluing the DQ. To fulfill the needs of the next-generation AVs with guarantees of functionality, efficiency, and trustworthiness, this paper proposes a novel task-centric and data quality vase framework which consists of five layers: data layer, DQ layer, task layer, application layer, and goal layer. The proposed framework aims to map DQ with task requirements and performance goals. To illustrate, a case study investigating redundancy on the nuScenes dataset proves that partially removing redundancy on multisource image data could improve YOLOv8 object detection task performance. Analysis on multimodal data of image and LiDAR further presents existing redundancy DQ issues. This paper opens up a range of critical but unexplored challenges at the intersection of DQ, task orchestration, and performance-oriented system development in AVs. It is expected to guide the AV community toward building more adaptive, explainable, and resilient AVs that respond intelligently to dynamic environments and heterogeneous data streams. Code, data, and implementation details are publicly available at: this https URL. 

**Abstract (ZH)**: 下一代自主车辆（AV）嵌入了频繁的实时决策，将高度依赖大量多源和多模态数据。在实际应用中，由于不可预测的环境因素或传感器问题，不同来源和模态的数据质量（DQ）通常会有所不同。然而，自动驾驶领域的研究人员和从业人员普遍重视模型/算法，而忽视了数据质量。为了确保下一代AV的功能、效率和可靠性，本文提出了一种新的以任务为中心的数据质量框架，该框架由五层组成：数据层、数据质量层、任务层、应用层和目标层。该提出的框架旨在将数据质量与任务要求和性能目标映射起来。通过在nuScenes数据集上的案例研究，部分移除多源图像数据中的冗余可提高YOLOv8目标检测任务的性能。对图像和LiDAR多模态数据的分析进一步揭示了现有冗余数据质量的问题。本文在数据质量、任务协调和面向性能的系统开发在自主车辆中的交叉点上揭示了一系列重要的但未被探索的挑战。它有望指导自动驾驶社区构建更具适应性、可解释性和韧性的自主车辆，这些车辆能够智能地响应动态环境和异构数据流。相关代码、数据和实现细节可在以下链接获取：this https URL。 

---
# Adaptive Social Metaverse Streaming based on Federated Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: 基于联邦多代理深度强化学习的自适应社会元宇宙流媒体 

**Authors**: Zijian Long, Haopeng Wang, Haiwei Dong, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2506.17342)  

**Abstract**: The social metaverse is a growing digital ecosystem that blends virtual and physical worlds. It allows users to interact socially, work, shop, and enjoy entertainment. However, privacy remains a major challenge, as immersive interactions require continuous collection of biometric and behavioral data. At the same time, ensuring high-quality, low-latency streaming is difficult due to the demands of real-time interaction, immersive rendering, and bandwidth optimization. To address these issues, we propose ASMS (Adaptive Social Metaverse Streaming), a novel streaming system based on Federated Multi-Agent Proximal Policy Optimization (F-MAPPO). ASMS leverages F-MAPPO, which integrates federated learning (FL) and deep reinforcement learning (DRL) to dynamically adjust streaming bit rates while preserving user privacy. Experimental results show that ASMS improves user experience by at least 14% compared to existing streaming methods across various network conditions. Therefore, ASMS enhances the social metaverse experience by providing seamless and immersive streaming, even in dynamic and resource-constrained networks, while ensuring that sensitive user data remains on local devices. 

**Abstract (ZH)**: 社交元宇宙是一种融合虚拟和物理世界的 growing 数字生态系统，允许用户进行社交互动、工作、购物和享受娱乐。然而，隐私仍然是一个主要挑战，因为沉浸式互动需要持续收集生物特征和行为数据。同时，由于实时交互、沉浸式渲染和带宽优化的需求，确保高质量、低延迟的流媒体传输也极具挑战性。为了解决这些问题，我们提出了 ASMS（自适应社交元宇宙流媒体）系统，这是一种基于联邦多代理近端策略优化（F-MAPPO）的创新流媒体系统。ASMS 利用了 F-MAPPO，该系统结合了联邦学习（FL）和深度强化学习（DRL），以动态调整流媒体比特率同时保护用户隐私。实验结果表明，与现有流媒体方法相比，ASMS 在多种网络条件下将用户体验至少提升了 14%。因此，ASMS 通过提供无缝和沉浸式的流媒体体验，即使在动态和资源受限的网络环境下也能增强社交元宇宙体验，同时确保敏感用户数据保留在本地设备上。 

---
# PBFT-Backed Semantic Voting for Multi-Agent Memory Pruning 

**Title (ZH)**: 基于PBFT的语义投票多agent内存剪枝 

**Authors**: Duong Bach  

**Link**: [PDF](https://arxiv.org/pdf/2506.17338)  

**Abstract**: The proliferation of multi-agent systems (MAS) in complex, dynamic environments necessitates robust and efficient mechanisms for managing shared knowledge. A critical challenge is ensuring that distributed memories remain synchronized, relevant, and free from the accumulation of outdated or inconsequential data - a process analogous to biological forgetting. This paper introduces the Co-Forgetting Protocol, a novel, comprehensive framework designed to address this challenge by enabling synchronized memory pruning in MAS. The protocol integrates three key components: (1) context-aware semantic voting, where agents utilize a lightweight DistilBERT model to assess the relevance of memory items based on their content and the current operational context; (2) multi-scale temporal decay functions, which assign diminishing importance to memories based on their age and access frequency across different time horizons; and (3) a Practical Byzantine Fault Tolerance (PBFT)-based consensus mechanism, ensuring that decisions to retain or discard memory items are agreed upon by a qualified and fault-tolerant majority of agents, even in the presence of up to f Byzantine (malicious or faulty) agents in a system of N greater than or equal to 3f+1 agents. The protocol leverages gRPC for efficient inter-agent communication and Pinecone for scalable vector embedding storage and similarity search, with SQLite managing metadata. Experimental evaluations in a simulated MAS environment with four agents demonstrate the protocol's efficacy, achieving a 52% reduction in memory footprint over 500 epochs, 88% voting accuracy in forgetting decisions against human-annotated benchmarks, a 92% PBFT consensus success rate under simulated Byzantine conditions, and an 82% cache hit rate for memory access. 

**Abstract (ZH)**: 多代理系统中分布式记忆同步与精简的Co-Forgetting协议 

---
# On the Performance of Cyber-Biomedical Features for Intrusion Detection in Healthcare 5.0 

**Title (ZH)**: 面向 healthcare 5.0 的网络生物医学特征入侵检测性能研究 

**Authors**: Pedro H. Lui, Lucas P. Siqueira, Juliano F. Kazienko, Vagner E. Quincozes, Silvio E. Quincozes, Daniel Welfer  

**Link**: [PDF](https://arxiv.org/pdf/2506.17329)  

**Abstract**: Healthcare 5.0 integrates Artificial Intelligence (AI), the Internet of Things (IoT), real-time monitoring, and human-centered design toward personalized medicine and predictive diagnostics. However, the increasing reliance on interconnected medical technologies exposes them to cyber threats. Meanwhile, current AI-driven cybersecurity models often neglect biomedical data, limiting their effectiveness and interpretability. This study addresses this gap by applying eXplainable AI (XAI) to a Healthcare 5.0 dataset that integrates network traffic and biomedical sensor data. Classification outputs indicate that XGBoost achieved 99% F1-score for benign and data alteration, and 81% for spoofing. Explainability findings reveal that network data play a dominant role in intrusion detection whereas biomedical features contributed to spoofing detection, with temperature reaching a Shapley values magnitude of 0.37. 

**Abstract (ZH)**: Healthcare 5.0融合了人工智能、物联网、实时监控和以人为本的设计，旨在实现个性化医疗和预测诊断。然而，对互联医疗技术的日益依赖使其面临网络安全威胁。当前的AI驱动的网络安全模型往往忽视生物医学数据，限制了其有效性和可解释性。本研究通过将可解释人工智能（XAI）应用于整合网络流量和生物医学传感器数据的Healthcare 5.0数据集，填补了这一缺口。分类输出表明，XGBoost在良性行为和数据篡改检测上达到了99%的F1分数，在冒充检测上达到了81%。可解释性研究表明，网络数据在网络入侵检测中发挥主导作用，而生物医学特征对冒充检测做出了贡献，其中体温的Shapley值为0.37。 

---
# Context manipulation attacks : Web agents are susceptible to corrupted memory 

**Title (ZH)**: 上下文操作攻击：网页代理易受污染内存的影响 

**Authors**: Atharv Singh Patlan, Ashwin Hebbar, Pramod Viswanath, Prateek Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17318)  

**Abstract**: Autonomous web navigation agents, which translate natural language instructions into sequences of browser actions, are increasingly deployed for complex tasks across e-commerce, information retrieval, and content discovery. Due to the stateless nature of large language models (LLMs), these agents rely heavily on external memory systems to maintain context across interactions. Unlike centralized systems where context is securely stored server-side, agent memory is often managed client-side or by third-party applications, creating significant security vulnerabilities. This was recently exploited to attack production systems.
We introduce and formalize "plan injection," a novel context manipulation attack that corrupts these agents' internal task representations by targeting this vulnerable context. Through systematic evaluation of two popular web agents, Browser-use and Agent-E, we show that plan injections bypass robust prompt injection defenses, achieving up to 3x higher attack success rates than comparable prompt-based attacks. Furthermore, "context-chained injections," which craft logical bridges between legitimate user goals and attacker objectives, lead to a 17.7% increase in success rate for privacy exfiltration tasks. Our findings highlight that secure memory handling must be a first-class concern in agentic systems. 

**Abstract (ZH)**: 自主网络导航代理将自然语言指令转化为浏览器操作序列，在电子商务、信息检索和内容发现等领域中越来越多地被用于复杂任务。由于大型语言模型（LLMs）缺乏状态维持能力，这些代理高度依赖外部内存系统来维持交互过程中的上下文。与将上下文安全地存储在服务器端的集中式系统不同，代理的内存往往在客户端或第三方应用程序中管理，从而创造出重大的安全漏洞。这最近被利用来攻击生产系统。

我们提出了并形式化了“计划注入”这一新颖的上下文操控攻击，通过针对这一脆弱的上下文，篡改这些代理内部的任务表示。通过系统性地评估两个流行的网络代理Browser-use和Agent-E，我们显示计划注入绕过了稳健的提示注入防御，其攻击成功率比同类提示基攻击高3倍。此外，通过在合法用户目标和攻击者目标之间构建逻辑桥梁的“上下文链接注入”，前所未有地将隐私泄露任务的成功率提高了17.7%。我们的研究结果表明，安全的内存处理必须在代理系统中被视为头等大事。 

---
# Heterogeneous Temporal Hypergraph Neural Network 

**Title (ZH)**: 异构时序超图神经网络 

**Authors**: Huan Liu, Pengfei Jiao, Mengzhou Gao, Chaochao Chen, Di Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17312)  

**Abstract**: Graph representation learning (GRL) has emerged as an effective technique for modeling graph-structured data. When modeling heterogeneity and dynamics in real-world complex networks, GRL methods designed for complex heterogeneous temporal graphs (HTGs) have been proposed and have achieved successful applications in various fields. However, most existing GRL methods mainly focus on preserving the low-order topology information while ignoring higher-order group interaction relationships, which are more consistent with real-world networks. In addition, most existing hypergraph methods can only model static homogeneous graphs, limiting their ability to model high-order interactions in HTGs. Therefore, to simultaneously enable the GRL model to capture high-order interaction relationships in HTGs, we first propose a formal definition of heterogeneous temporal hypergraphs and $P$-uniform heterogeneous hyperedge construction algorithm that does not rely on additional information. Then, a novel Heterogeneous Temporal HyperGraph Neural network (HTHGN), is proposed to fully capture higher-order interactions in HTGs. HTHGN contains a hierarchical attention mechanism module that simultaneously performs temporal message-passing between heterogeneous nodes and hyperedges to capture rich semantics in a wider receptive field brought by hyperedges. Furthermore, HTHGN performs contrastive learning by maximizing the consistency between low-order correlated heterogeneous node pairs on HTG to avoid the low-order structural ambiguity issue. Detailed experimental results on three real-world HTG datasets verify the effectiveness of the proposed HTHGN for modeling high-order interactions in HTGs and demonstrate significant performance improvements. 

**Abstract (ZH)**: 基于异构时变超图的图表示学习方法：捕获高阶交互关系的新范式 

---
# AlgoSelect: Universal Algorithm Selection via the Comb Operator 

**Title (ZH)**: AlgoSelect: 统一的算法选择方法通过Combing运算符 

**Authors**: Jasper Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17304)  

**Abstract**: We introduce AlgoSelect, a principled framework for learning optimal algorithm selection from data, centered around the novel Comb Operator. Given a set of algorithms and a feature representation of problems, AlgoSelect learns to interpolate between diverse computational approaches. For pairs of algorithms, a simple sigmoid-gated selector, an instance of the Comb Operator, facilitates this interpolation. We extend this to an N-Path Comb for multiple algorithms. We prove that this framework is universal (can approximate any algorithm selector), information-theoretically optimal in its learnability (thresholds for selection converge almost surely, demonstrated via Borel-Cantelli arguments), computationally efficient, and robust. Key theoretical contributions include: (1) a universal approximation theorem demonstrating that Comb-based selectors can achieve arbitrary accuracy; (2) information-theoretic learnability for selection thresholds; (3) formalization of the Comb Operator within linear operator theory, detailing its boundedness and spectral properties; (4) an N-Path Comb generalization for multi-algorithm selection; and (5) a practical learning framework for the adaptive seeding functions that guide the Comb Operator. Empirical validation on a comprehensive 20$\times$20 problem-algorithm study demonstrates near-perfect selection (99.9\%+ accuracy) with remarkably few samples and rapid convergence, revealing that $H(\text{Algorithm}|\text{Problem}) \approx 0$ in structured domains. AlgoSelect provides a theoretically grounded, practically deployable solution to automated algorithm selection with provable optimality and learnability guarantees, with significant implications for AI and adaptive systems. 

**Abstract (ZH)**: 我们介绍了一种基于新颖Comb操作器的原理性框架AlgoSelect，用于从数据中学习最优算法选择。该框架通过插值多种计算方法，针对算法集和问题特征表示，学习如何进行选择。对于成对的算法，Comb操作器的一种简单Sigmoid门选路器促进了这种插值。我们将这一方法扩展到支持多算法的N-Path Comb。我们证明了该框架是通用的（能够逼近任何算法选择器）、信息论上的可学习性最优（选择阈值几乎必然收敛，通过博雷尔-坎特利论证证明）、计算效率高且稳健。关键的理论贡献包括：（1）一个通用逼近定理，证明Comb基于的选择器可以达到任意精度；（2）选择阈值的信息论可学习性；（3）在线性算子理论中形式化Comb操作器，详细描述其有界性和频谱性质；（4）支持多算法选择的N-Path Comb推广；以及（5）指导Comb操作器的自适应初始化函数的实用学习框架。在全面的20$\times$20问题-算法研究中进行的经验验证表明，在惊人少量的样本和快速收敛下，选择精度接近完美（99.9%以上），揭示出在结构域中$H(\text{Algorithm}|\text{Problem}) \approx 0$。AlgoSelect提供了一种理论依据且实用部署的自动化算法选择解决方案，带有可证明的最优性和可学习性保证，对AI和自适应系统具有重大影响。 

---
# SafeRL-Lite: A Lightweight, Explainable, and Constrained Reinforcement Learning Library 

**Title (ZH)**: SafeRL-Lite: 一种轻量级、可解释且受约束的强化学习库 

**Authors**: Satyam Mishra, Phung Thao Vi, Shivam Mishra, Vishwanath Bijalwan, Vijay Bhaskar Semwal, Abdul Manan Khan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17297)  

**Abstract**: We introduce SafeRL-Lite, an open-source Python library for building reinforcement learning (RL) agents that are both constrained and explainable. Existing RL toolkits often lack native mechanisms for enforcing hard safety constraints or producing human-interpretable rationales for decisions. SafeRL-Lite provides modular wrappers around standard Gym environments and deep Q-learning agents to enable: (i) safety-aware training via constraint enforcement, and (ii) real-time post-hoc explanation via SHAP values and saliency maps. The library is lightweight, extensible, and installable via pip, and includes built-in metrics for constraint violations. We demonstrate its effectiveness on constrained variants of CartPole and provide visualizations that reveal both policy logic and safety adherence. The full codebase is available at: this https URL. 

**Abstract (ZH)**: 我们介绍SafeRL-Lite，一个开源Python库，用于构建既受约束又可解释的强化学习（RL）代理。现有的RL工具包通常缺乏强制执行严格安全约束或生成人类可解释决策理由的内置机制。SafeRL-Lite通过模块化包装标准Gym环境和深度Q学习代理，实现了：(i) 通过约束强制执行进行安全意识训练，以及(ii) 通过SHAP值和可解释性地图进行实时事后解释。该库轻量级、可扩展，并可通过pip安装，内置了约束违反的度量标准。我们展示了其在受约束的CartPole变体上的有效性，并提供了可视化，揭示了策略逻辑和安全性遵循情况。完整代码库可在以下链接获取：this https URL。 

---
# AI-Generated Game Commentary: A Survey and a Datasheet Repository 

**Title (ZH)**: AI生成的游戏评论：综述与数据集仓库 

**Authors**: Qirui Zheng, Xingbo Wang, Keyuan Cheng, Yunlong Lu, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17294)  

**Abstract**: AI-Generated Game Commentary (AIGGC) has gained increasing attention due to its market potential and inherent technical challenges. As a comprehensive multimodal Natural Language Processing (NLP) task, AIGGC imposes substantial demands on language models, including factual accuracy, logical reasoning, expressive text generation, generation speed, and context management. In this paper, we introduce a general framework for AIGGC and present a comprehensive survey of 45 existing game commentary dataset and methods according to key challenges they aim to address in this domain. We further classify and compare various evaluation metrics commonly used in this domain. To support future research and benchmarking, we also provide a structured datasheet summarizing the essential attributes of these datasets in appendix, which is meanwhile publicly available in an open repository. 

**Abstract (ZH)**: AI生成的游戏评论（AIGGC）因其市场潜力和固有的技术挑战而日益受到关注。作为一种综合性的多模态自然语言处理任务，AIGGC对语言模型提出了包括事实准确性、逻辑推理、表达性文本生成、生成速度和上下文管理等方面的重大需求。本文介绍了一种AIGGC的通用框架，并根据它们在这领域中试图解决的关键挑战，综述了45个现有的游戏评论数据集和方法。此外，我们还对这个领域中常用的各种评估指标进行了分类和比较。为了支持未来的研究和基准测试，我们还在附录中提供了一份结构化的数据表，总结了这些数据集的关键属性，并且该数据表同时在开放仓库中公开可用。 

---
# SlimRAG: Retrieval without Graphs via Entity-Aware Context Selection 

**Title (ZH)**: SlimRAG：无需图的实体意识上下文选择检索 

**Authors**: Jiale Zhang, Jiaxiang Chen, Zhucong Li, Jie Ding, Kui Zhao, Zenglin Xu, Xin Pang, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17288)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language models by incorporating external knowledge at inference time. However, graph-based RAG systems often suffer from structural overhead and imprecise retrieval: they require costly pipelines for entity linking and relation extraction, yet frequently return subgraphs filled with loosely related or tangential content. This stems from a fundamental flaw -- semantic similarity does not imply semantic relevance. We introduce SlimRAG, a lightweight framework for retrieval without graphs. SlimRAG replaces structure-heavy components with a simple yet effective entity-aware mechanism. At indexing time, it constructs a compact entity-to-chunk table based on semantic embeddings. At query time, it identifies salient entities, retrieves and scores associated chunks, and assembles a concise, contextually relevant input -- without graph traversal or edge construction. To quantify retrieval efficiency, we propose Relative Index Token Utilization (RITU), a metric measuring the compactness of retrieved content. Experiments across multiple QA benchmarks show that SlimRAG outperforms strong flat and graph-based baselines in accuracy while reducing index size and RITU (e.g., 16.31 vs. 56+), highlighting the value of structure-free, entity-centric context selection. The code will be released soon. this https URL 

**Abstract (ZH)**: 基于检索的生成（RAG）通过在推理时 incorporare 外部知识来增强语言模型。然而，基于图的 RAG 系统往往受到结构开销和检索不精确的问题：它们需要昂贵的实体链接和关系抽取管道，但经常返回包含松散相关或不相关内容的子图。这源于一个根本性的缺陷——语义相似性不等于语义相关性。我们提出了 SlimRAG，一个轻量级的不基于图的检索框架。SlimRAG 用简单有效的实体感知机制取代了结构密集的组件。在索引时，它基于语义嵌入构建紧凑的实体到片段表。在查询时，它识别显著实体、检索和评分相关片段，并组装成简洁且上下文相关的内容，无需进行图遍历或边构建。为了量化检索效率，我们提出了相对索引_token_利用度 (RITU) 的度量标准，衡量检索内容的紧凑性。多项 QA 基准实验显示，SlimRAG 在准确率上优于强大的平铺和基于图的基线模型的同时减小了索引大小和 RITU（例如，16.31 对比 56+），突显了无结构、以实体为中心的上下文选择的价值。代码将于近期发布。 

---
# A Theoretical Framework for Virtual Power Plant Integration with Gigawatt-Scale AI Data Centers: Multi-Timescale Control and Stability Analysis 

**Title (ZH)**: 兆瓦级人工智能数据中心与虚拟电厂集成的理论框架：多时间尺度控制与稳定性分析 

**Authors**: Ali Peivandizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.17284)  

**Abstract**: The explosive growth of artificial intelligence has created gigawatt-scale data centers that fundamentally challenge power system operation, exhibiting power fluctuations exceeding 500 MW within seconds and millisecond-scale variations of 50-75% of thermal design power. This paper presents a comprehensive theoretical framework that reconceptualizes Virtual Power Plants (VPPs) to accommodate these extreme dynamics through a four-layer hierarchical control architecture operating across timescales from 100 microseconds to 24 hours.
We develop control mechanisms and stability criteria specifically tailored to converter-dominated systems with pulsing megawatt-scale loads. We prove that traditional VPP architectures, designed for aggregating distributed resources with response times of seconds to minutes, cannot maintain stability when confronted with AI data center dynamics exhibiting slew rates exceeding 1,000 MW/s at gigawatt scale.
Our framework introduces: (1) a sub-millisecond control layer that interfaces with data center power electronics to actively dampen power oscillations; (2) new stability criteria incorporating protection system dynamics, demonstrating that critical clearing times reduce from 150 ms to 83 ms for gigawatt-scale pulsing loads; and (3) quantified flexibility characterization showing that workload deferability enables 30% peak reduction while maintaining AI service availability above 99.95%.
This work establishes the mathematical foundations necessary for the stable integration of AI infrastructure that will constitute 50-70% of data center electricity consumption by 2030. 

**Abstract (ZH)**: 人工智能的爆炸性增长创建了 gigawatt 规模的数据中心，从根本上挑战了电力系统的运行，展示出秒级内超过 500 MW 的功率波动和毫秒级范围内 50-75% 的热设计功率变化。本文提出了一种综合理论框架，通过一种四层分级控制架构重新概念化虚拟电厂（VPP），该架构的时间跨度从 100 微秒到 24 小时，以适应这些极端动态。 

---
# Chunk Twice, Embed Once: A Systematic Study of Segmentation and Representation Trade-offs in Chemistry-Aware Retrieval-Augmented Generation 

**Title (ZH)**: 两次切分，一次嵌入：化学意识检索增强生成中切分与表示权衡的系统研究 

**Authors**: Mahmoud Amiri, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.17277)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly vital for navigating the ever-expanding body of scientific literature, particularly in high-stakes domains such as chemistry. Despite the promise of RAG, foundational design choices -- such as how documents are segmented and represented -- remain underexplored in domain-specific contexts. This study presents the first large-scale, systematic evaluation of chunking strategies and embedding models tailored to chemistry-focused RAG systems. We investigate 25 chunking configurations across five method families and evaluate 48 embedding models on three chemistry-specific benchmarks, including the newly introduced QuestChemRetrieval dataset. Our results reveal that recursive token-based chunking (specifically R100-0) consistently outperforms other approaches, offering strong performance with minimal resource overhead. We also find that retrieval-optimized embeddings -- such as Nomic and Intfloat E5 variants -- substantially outperform domain-specialized models like SciBERT. By releasing our datasets, evaluation framework, and empirical benchmarks, we provide actionable guidelines for building effective and efficient chemistry-aware RAG systems. 

**Abstract (ZH)**: 基于检索增强生成的化学专注系统的大规模系统性评估 

---
# Modal Logic for Stratified Becoming: Actualization Beyond Possible Worlds 

**Title (ZH)**: 分层生成的模态逻辑：超越可能世界的实现 

**Authors**: Alexandre Le Nepvou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17276)  

**Abstract**: This article develops a novel framework for modal logic based on the idea of stratified actualization, rather than the classical model of global possible worlds. Traditional Kripke semantics treat modal operators as quantification over fully determinate alternatives, neglecting the local, dynamic, and often asymmetric nature of actualization processes. We propose a system Stratified Actualization Logic (SAL) in which modalities are indexed by levels of ontological stability, interpreted as admissibility regimes. Each modality operates over a structured layer of possibility, grounded in the internal coherence of transitions between layers. We formally define the syntax and semantics of SAL, introduce its axioms, and prove soundness and completeness. Applications are discussed in connection with temporal becoming, quantum decoherence domains, and modal metaphysics. The result is a logic that captures the ontological structure of actualization without recourse to abstract possible worlds, offering a stratified alternative to standard modal realism. 

**Abstract (ZH)**: 基于分层实现的新范式模态逻辑 

---
# QUST_NLP at SemEval-2025 Task 7: A Three-Stage Retrieval Framework for Monolingual and Crosslingual Fact-Checked Claim Retrieval 

**Title (ZH)**: QUST_NLP 在 SemEval-2025 任务 7 中的三级检索框架：单语和跨语言事实核查声明检索 

**Authors**: Youzheng Liu, Jiyan Liu, Xiaoman Xu, Taihang Wang, Yimin Wang, Ye Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17272)  

**Abstract**: This paper describes the participation of QUST_NLP in the SemEval-2025 Task 7. We propose a three-stage retrieval framework specifically designed for fact-checked claim retrieval. Initially, we evaluate the performance of several retrieval models and select the one that yields the best results for candidate retrieval. Next, we employ multiple re-ranking models to enhance the candidate results, with each model selecting the Top-10 outcomes. In the final stage, we utilize weighted voting to determine the final retrieval outcomes. Our approach achieved 5th place in the monolingual track and 7th place in the crosslingual track. We release our system code at: this https URL 

**Abstract (ZH)**: 本论文描述了UESTC_NLP在SemEval-2025 Task 7中的参与情况。我们提出了一种专门设计的事实核验声明检索的三阶段检索框架。首先，我们评估了几种检索模型的性能，并选择了性能最佳的候选检索模型。接着，我们采用了多种重排序模型来提升候选结果，每种模型选取Top-10结果。在最终阶段，我们利用加权投票来确定最终的检索结果。我们的方法在单语轨道中获得第5名，在跨语言轨道中获得第7名。我们已将系统代码发布在以下链接：this https URL。 

---
# Memory Allocation in Resource-Constrained Reinforcement Learning 

**Title (ZH)**: 资源受限强化学习中的内存分配 

**Authors**: Massimiliano Tamborski, David Abel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17263)  

**Abstract**: Resource constraints can fundamentally change both learning and decision-making. We explore how memory constraints influence an agent's performance when navigating unknown environments using standard reinforcement learning algorithms. Specifically, memory-constrained agents face a dilemma: how much of their limited memory should be allocated to each of the agent's internal processes, such as estimating a world model, as opposed to forming a plan using that model? We study this dilemma in MCTS- and DQN-based algorithms and examine how different allocations of memory impact performance in episodic and continual learning settings. 

**Abstract (ZH)**: 资源约束可以根本上改变学习和决策的方式。我们探究了当使用标准强化学习算法在未知环境中导航时，记忆约束如何影响智能体的性能。具体而言，记忆受限的智能体面临一个困境：它们有限的记忆应分配给内部过程（如构建世界模型的估计）还是利用该模型形成计划？我们在这类基于MCTS和DQN的算法中研究这一困境，并考察不同记忆分配对 episodic 学习和持续学习设置中性能的影响。 

---
# MS-TVNet:A Long-Term Time Series Prediction Method Based on Multi-Scale Dynamic Convolution 

**Title (ZH)**: MS-TVNet：基于多尺度动态卷积的长期时间序列预测方法 

**Authors**: Chenghan Li, Mingchen Li, Yipu Liao, Ruisheng Diao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17253)  

**Abstract**: Long-term time series prediction has predominantly relied on Transformer and MLP models, while the potential of convolutional networks in this domain remains underexplored. To address this gap, we introduce a novel multi-scale time series reshape module, which effectively captures the relationships among multi-period patches and variable dependencies. Building upon this module, we propose MS-TVNet, a multi-scale 3D dynamic convolutional neural network. Through comprehensive evaluations on diverse datasets, MS-TVNet demonstrates superior performance compared to baseline models, achieving state-of-the-art (SOTA) results in long-term time series prediction. Our findings highlight the effectiveness of leveraging convolutional networks for capturing complex temporal patterns, suggesting a promising direction for future research in this this http URL code is realsed on this https URL. 

**Abstract (ZH)**: 长期时间序列预测主要依赖于Transformer和MLP模型，而卷积网络在这一领域中的潜力尚未充分探索。为填补这一空白，我们提出了一种新颖的多尺度时间序列重构模块，该模块有效地捕捉了多周期片段之间的关系和可变依赖性。基于该模块，我们提出了MS-TVNet，一种多尺度3D动态卷积神经网络。通过在多种数据集上的全面评估，MS-TVNet展示了相对于基线模型的优越性能，并在长期时间序列预测中取得了最优结果。我们的研究结果强调了利用卷积网络捕捉复杂时间模式的有效性，为未来研究提供了有前途的方向。代码已在以下链接发布：https://github.com/your-repo-address。 

---
# Towards Interpretable Adversarial Examples via Sparse Adversarial Attack 

**Title (ZH)**: 面向可解释的对抗样本：基于稀疏对抗攻击的方法 

**Authors**: Fudong Lin, Jiadong Lou, Hao Wang, Brian Jalaian, Xu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17250)  

**Abstract**: Sparse attacks are to optimize the magnitude of adversarial perturbations for fooling deep neural networks (DNNs) involving only a few perturbed pixels (i.e., under the l0 constraint), suitable for interpreting the vulnerability of DNNs. However, existing solutions fail to yield interpretable adversarial examples due to their poor sparsity. Worse still, they often struggle with heavy computational overhead, poor transferability, and weak attack strength. In this paper, we aim to develop a sparse attack for understanding the vulnerability of CNNs by minimizing the magnitude of initial perturbations under the l0 constraint, to overcome the existing drawbacks while achieving a fast, transferable, and strong attack to DNNs. In particular, a novel and theoretical sound parameterization technique is introduced to approximate the NP-hard l0 optimization problem, making directly optimizing sparse perturbations computationally feasible. Besides, a novel loss function is designed to augment initial perturbations by maximizing the adversary property and minimizing the number of perturbed pixels simultaneously. Extensive experiments are conducted to demonstrate that our approach, with theoretical performance guarantees, outperforms state-of-the-art sparse attacks in terms of computational overhead, transferability, and attack strength, expecting to serve as a benchmark for evaluating the robustness of DNNs. In addition, theoretical and empirical results validate that our approach yields sparser adversarial examples, empowering us to discover two categories of noises, i.e., "obscuring noise" and "leading noise", which will help interpret how adversarial perturbation misleads the classifiers into incorrect predictions. Our code is available at this https URL. 

**Abstract (ZH)**: 稀疏攻击优化少量受扰像素（即在l0约束下）的对抗扰动幅度，以迷惑深度神经网络（DNNs），适合作为解释DNNs脆弱性的工具。然而，现有解决方案由于稀疏性较差，无法生成可解释的对抗样本。更糟糕的是，它们通常面临着计算开销重、迁移性差以及攻击强度弱的问题。本文旨在通过在l0约束下最小化初始扰动幅度来开发一种稀疏攻击，以克服现有方法的不足，同时实现快速、可迁移且具有强大攻击性的对抗DNNs的方法。特别地，提出了一种新的且理论上合理的参数化技术来近似NP难的l0优化问题，使得直接优化稀疏扰动在计算上可行。此外，设计了一种新的损失函数，通过同时最大化对抗特性并最小化受扰像素的数量来增强初始扰动。大量实验显示，在计算开销、迁移性和攻击强度方面，我们的方法都优于现有最先进的稀疏攻击方法，并期待成为评估DNNs鲁棒性的一个基准。此外，理论和实验证据表明，我们的方法生成了更稀疏的对抗样本，帮助我们发现两类噪声，即“遮蔽噪声”和“引导噪声”，这将有助于解释对抗扰动是如何误导分类器产生错误预测的。我们的代码可在以下链接获取。 

---
# Recursive Learning-Based Virtual Buffering for Analytical Global Placement 

**Title (ZH)**: 基于递归学习的虚拟缓冲全局布线方法 

**Authors**: Andrew B. Kahng, Yiting Liu, Zhiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17247)  

**Abstract**: Due to the skewed scaling of interconnect versus cell delay in modern technology nodes, placement with buffer porosity (i.e., cell density) awareness is essential for timing closure in physical synthesis flows. However, existing approaches face two key challenges: (i) traditional van Ginneken-Lillis-style buffering approaches are computationally expensive during global placement; and (ii) machine learning-based approaches, such as BufFormer, lack a thorough consideration of Electrical Rule Check (ERC) violations and fail to "close the loop" back into the physical design flow. In this work, we propose MLBuf-RePlAce, the first open-source learning-driven virtual buffering-aware analytical global placement framework, built on top of the OpenROAD infrastructure. MLBuf-RePlAce adopts an efficient recursive learning-based generative buffering approach to predict buffer types and locations, addressing ERC violations during global placement. We compare MLBuf-RePlAce against the default virtual buffering-based timing-driven global placer in OpenROAD, using open-source testcases from the TILOS MacroPlacement and OpenROAD-flow-scripts repositories. Without degradation of post-route power, MLBuf-RePlAce achieves (maximum, average) improvements of (56%, 31%) in total negative slack (TNS) within the open-source OpenROAD flow. When evaluated by completion in a commercial flow, MLBuf-RePlAce achieves (maximum, average) improvements of (53%, 28%) in TNS with an average of 0.2% improvement in post-route power. 

**Abstract (ZH)**: 基于学习的虚拟缓存aware全局布图设计框架MLBuf-RePlAce 

---
# Graph Neural Networks in Multi-Omics Cancer Research: A Structured Survey 

**Title (ZH)**: 图神经网络在多组学癌症研究中的应用：一个结构化的综述 

**Authors**: Payam Zohari, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2506.17234)  

**Abstract**: The task of data integration for multi-omics data has emerged as a powerful strategy to unravel the complex biological underpinnings of cancer. Recent advancements in graph neural networks (GNNs) offer an effective framework to model heterogeneous and structured omics data, enabling precise representation of molecular interactions and regulatory networks. This systematic review explores several recent studies that leverage GNN-based architectures in multi-omics cancer research. We classify the approaches based on their targeted omics layers, graph neural network structures, and biological tasks such as subtype classification, prognosis prediction, and biomarker discovery. The analysis reveals a growing trend toward hybrid and interpretable models, alongside increasing adoption of attention mechanisms and contrastive learning. Furthermore, we highlight the use of patient-specific graphs and knowledge-driven priors as emerging directions. This survey serves as a comprehensive resource for researchers aiming to design effective GNN-based pipelines for integrative cancer analysis, offering insights into current practices, limitations, and potential future directions. 

**Abstract (ZH)**: 多组学数据集成的任务作为揭示癌症复杂生物学机制的一种强大策略已经显现出来。基于图神经网络（GNN）的最新进展为建模异构和结构化的多组学数据提供了有效框架，能够精确表示分子互作和调控网络。本系统综述探讨了若干采用基于GNN架构的近期研究在多组学癌症研究中的应用。我们将方法根据靶向的组学层次、图神经网络结构以及亚型分类、预后预测和生物标志物发现等生物任务进行分类。分析显示，混合和可解释模型的趋势日益增长，同时注意力机制和对比学习的采用也在增加。此外，我们强调了患者特定图和知识驱动先验作为新兴方向的应用。本文综述为旨在设计有效的基于GNN的集成癌症分析管道的研究人员提供了一项全面资源，提出了当前做法、局限性和潜在未来方向的见解。 

---
# MMET: A Multi-Input and Multi-Scale Transformer for Efficient PDEs Solving 

**Title (ZH)**: 多输入与多尺度变压器：一种高效的偏微分方程求解方法 

**Authors**: Yichen Luo, Jia Wang, Dapeng Lan, Yu Liu, Zhibo Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17230)  

**Abstract**: Partial Differential Equations (PDEs) are fundamental for modeling physical systems, yet solving them in a generic and efficient manner using machine learning-based approaches remains challenging due to limited multi-input and multi-scale generalization capabilities, as well as high computational costs. This paper proposes the Multi-input and Multi-scale Efficient Transformer (MMET), a novel framework designed to address the above challenges. MMET decouples mesh and query points as two sequences and feeds them into the encoder and decoder, respectively, and uses a Gated Condition Embedding (GCE) layer to embed input variables or functions with varying dimensions, enabling effective solutions for multi-scale and multi-input problems. Additionally, a Hilbert curve-based reserialization and patch embedding mechanism decrease the input length. This significantly reduces the computational cost when dealing with large-scale geometric models. These innovations enable efficient representations and support multi-scale resolution queries for large-scale and multi-input PDE problems. Experimental evaluations on diverse benchmarks spanning different physical fields demonstrate that MMET outperforms SOTA methods in both accuracy and computational efficiency. This work highlights the potential of MMET as a robust and scalable solution for real-time PDE solving in engineering and physics-based applications, paving the way for future explorations into pre-trained large-scale models in specific domains. This work is open-sourced at this https URL. 

**Abstract (ZH)**: 多输入多尺度高效变压器（MMET）：面向物理系统的偏微分方程求解 

---
