# Investigating Sensors and Methods in Grasp State Classification in Agricultural Manipulation 

**Title (ZH)**: 研究农业 manipulation 中抓取状态分类的传感器与方法 

**Authors**: Benjamin Walt, Jordan Westphal, Girish Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11588)  

**Abstract**: Effective and efficient agricultural manipulation and harvesting depend on accurately understanding the current state of the grasp. The agricultural environment presents unique challenges due to its complexity, clutter, and occlusion. Additionally, fruit is physically attached to the plant, requiring precise separation during harvesting. Selecting appropriate sensors and modeling techniques is critical for obtaining reliable feedback and correctly identifying grasp states. This work investigates a set of key sensors, namely inertial measurement units (IMUs), infrared (IR) reflectance, tension, tactile sensors, and RGB cameras, integrated into a compliant gripper to classify grasp states. We evaluate the individual contribution of each sensor and compare the performance of two widely used classification models: Random Forest and Long Short-Term Memory (LSTM) networks. Our results demonstrate that a Random Forest classifier, trained in a controlled lab environment and tested on real cherry tomato plants, achieved 100% accuracy in identifying slip, grasp failure, and successful picks, marking a substantial improvement over baseline performance. Furthermore, we identify a minimal viable sensor combination, namely IMU and tension sensors that effectively classifies grasp states. This classifier enables the planning of corrective actions based on real-time feedback, thereby enhancing the efficiency and reliability of fruit harvesting operations. 

**Abstract (ZH)**: 有效的农业操作和收获依赖于对当前握持状态的准确理解。农业环境因其复杂性、混乱和遮挡而提出独特的挑战。此外，水果物理上附着在植物上，要求在收获过程中进行精确分离。选择适当的传感器和建模技术对于获取可靠反馈和正确识别握持状态至关重要。本研究探讨了一组关键传感器，即惯性测量单元（IMUs）、红外（IR）反射率、张力、触觉传感器和RGB相机，集成到顺应性夹爪中以分类握持状态。我们评估了每个传感器的个体贡献，并比较了两种广泛使用的分类模型：随机森林和长短期记忆（LSTM）网络的性能。我们的结果显示，随机森林分类器在可控实验室环境中训练，在实际樱桃番茄植株上测试，实现了对滑落、握持失败和成功收获的100%准确识别，显著优于基线性能。此外，我们确定了一种最小可行的传感器组合，即IMU和张力传感器，能够有效分类握持状态。该分类器使得基于实时反馈规划纠正措施成为可能，从而提高了水果收获操作的效率和可靠性。 

---
# Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks 

**Title (ZH)**: 视觉感知引擎：面向机器人视觉任务的快速灵活多头推断 

**Authors**: Jakub Łucki, Jonathan Becktor, Georgios Georgakis, Robert Royce, Shehryar Khattak  

**Link**: [PDF](https://arxiv.org/pdf/2508.11584)  

**Abstract**: Deploying multiple machine learning models on resource-constrained robotic platforms for different perception tasks often results in redundant computations, large memory footprints, and complex integration challenges. In response, this work presents Visual Perception Engine (VPEngine), a modular framework designed to enable efficient GPU usage for visual multitasking while maintaining extensibility and developer accessibility. Our framework architecture leverages a shared foundation model backbone that extracts image representations, which are efficiently shared, without any unnecessary GPU-CPU memory transfers, across multiple specialized task-specific model heads running in parallel. This design eliminates the computational redundancy inherent in feature extraction component when deploying traditional sequential models while enabling dynamic task prioritization based on application demands. We demonstrate our framework's capabilities through an example implementation using DINOv2 as the foundation model with multiple task (depth, object detection and semantic segmentation) heads, achieving up to 3x speedup compared to sequential execution. Building on CUDA Multi-Process Service (MPS), VPEngine offers efficient GPU utilization and maintains a constant memory footprint while allowing per-task inference frequencies to be adjusted dynamically during runtime. The framework is written in Python and is open source with ROS2 C++ (Humble) bindings for ease of use by the robotics community across diverse robotic platforms. Our example implementation demonstrates end-to-end real-time performance at $\geq$50 Hz on NVIDIA Jetson Orin AGX for TensorRT optimized models. 

**Abstract (ZH)**: 基于资源共享的模块化视觉感知引擎（VPEngine）：在资源受限的机器人平台上实现高效视觉多任务处理 

---
# Nominal Evaluation Of Automatic Multi-Sections Control Potential In Comparison To A Simpler One- Or Two-Sections Alternative With Predictive Spray Switching 

**Title (ZH)**: 名义评估自动多段控制潜力与简单的一段或多段替代方案（带有预测性喷雾切换）的比较 

**Authors**: Mogens Plessen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11573)  

**Abstract**: Automatic Section Control (ASC) is a long-standing trend for spraying in agriculture. It promises to minimise spray overlap areas. The core idea is to (i) switch off spray nozzles on areas that have already been sprayed, and (ii) to dynamically adjust nozzle flow rates along the boom bar that holds the spray nozzles when velocities of boom sections vary during turn maneuvers. ASC is not possible without sensors, in particular for accurate positioning data. Spraying and the movement of modern wide boom bars are highly dynamic processes. In addition, many uncertainty factors have an effect such as cross wind drift, boom height, nozzle clogging in open-field conditions, and so forth. In view of this complexity, the natural question arises if a simpler alternative exist. Therefore, an Automatic Multi-Sections Control method is compared to a proposed simpler one- or two-sections alternative that uses predictive spray switching. The comparison is provided under nominal conditions. Agricultural spraying is intrinsically linked to area coverage path planning and spray switching logic. Combinations of two area coverage path planning and switching logics as well as three sections-setups are compared. The three sections-setups differ by controlling 48 sections, 2 sections or controlling all nozzles uniformly with the same control signal as one single section. Methods are evaluated on 10 diverse real-world field examples, including non-convex field contours, freeform mainfield lanes and multiple obstacle areas. A preferred method is suggested that (i) minimises area coverage pathlength, (ii) offers intermediate overlap, (iii) is suitable for manual driving by following a pre-planned predictive spray switching logic for an area coverage path plan, and (iv) and in contrast to ASC can be implemented sensor-free and therefore at low cost. 

**Abstract (ZH)**: 自动多区域控制（AMC）方法及其预测喷洒切换的简单替代方案比较 

---
# Towards Fully Onboard State Estimation and Trajectory Tracking for UAVs with Suspended Payloads 

**Title (ZH)**: 面向悬挂载荷的无人机的完全载荷状态下状态估计与轨迹跟踪研究 

**Authors**: Martin Jiroušek, Tomáš Báča, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2508.11547)  

**Abstract**: This paper addresses the problem of tracking the position of a cable-suspended payload carried by an unmanned aerial vehicle, with a focus on real-world deployment and minimal hardware requirements. In contrast to many existing approaches that rely on motion-capture systems, additional onboard cameras, or instrumented payloads, we propose a framework that uses only standard onboard sensors--specifically, real-time kinematic global navigation satellite system measurements and data from the onboard inertial measurement unit--to estimate and control the payload's position. The system models the full coupled dynamics of the aerial vehicle and payload, and integrates a linear Kalman filter for state estimation, a model predictive contouring control planner, and an incremental model predictive controller. The control architecture is designed to remain effective despite sensing limitations and estimation uncertainty. Extensive simulations demonstrate that the proposed system achieves performance comparable to control based on ground-truth measurements, with only minor degradation (< 6%). The system also shows strong robustness to variations in payload parameters. Field experiments further validate the framework, confirming its practical applicability and reliable performance in outdoor environments using only off-the-shelf aerial vehicle hardware. 

**Abstract (ZH)**: 本文解决了由无人驾驶航空器承载的悬挂电缆载荷位置跟踪问题，着重于实际部署并与最少硬件要求相结合。与许多现有的依赖于运动捕捉系统、附加机载摄像头或配备传感器的载荷的方法不同，我们提出了一种仅使用标准机载传感器（具体而言是实时动态全球定位系统测量和机载惯性测量单元的数据）来估计和控制载荷位置的框架。该系统模型涵盖了空中飞行器和载荷的完整耦合动态，集成了线性卡尔曼滤波器进行状态估计、模型预测包络控制规划器以及增量模型预测控制器。控制架构设计旨在即使在存在传感限制和估计不确定性的情况下仍能保持有效性。大量仿真表明，所提出系统在基于地面真实测量的控制性能方面具有可比性，仅轻微降解 (< 6%)。该系统还表现出对载荷参数变化的强大鲁棒性。现场实验进一步验证了该框架，在仅使用即插即用的飞行器硬件的户外环境中证实了其实用可行性和可靠性能。 

---
# MultiPark: Multimodal Parking Transformer with Next-Segment Prediction 

**Title (ZH)**: 多模态停车变换器：基于下一段落预测的多模态停车模型 

**Authors**: Han Zheng, Zikang Zhou, Guli Zhang, Zhepei Wang, Kaixuan Wang, Peiliang Li, Shaojie Shen, Ming Yang, Tong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11537)  

**Abstract**: Parking accurately and safely in highly constrained spaces remains a critical challenge. Unlike structured driving environments, parking requires executing complex maneuvers such as frequent gear shifts and steering saturation. Recent attempts to employ imitation learning (IL) for parking have achieved promising results. However, existing works ignore the multimodal nature of parking behavior in lane-free open space, failing to derive multiple plausible solutions under the same situation. Notably, IL-based methods encompass inherent causal confusion, so enabling a neural network to generalize across diverse parking scenarios is particularly difficult. To address these challenges, we propose MultiPark, an autoregressive transformer for multimodal parking. To handle paths filled with abrupt turning points, we introduce a data-efficient next-segment prediction paradigm, enabling spatial generalization and temporal extrapolation. Furthermore, we design learnable parking queries factorized into gear, longitudinal, and lateral components, parallelly decoding diverse parking behaviors. To mitigate causal confusion in IL, our method employs target-centric pose and ego-centric collision as outcome-oriented loss across all modalities beyond pure imitation loss. Evaluations on real-world datasets demonstrate that MultiPark achieves state-of-the-art performance across various scenarios. We deploy MultiPark on a production vehicle, further confirming our approach's robustness in real-world parking environments. 

**Abstract (ZH)**: 多模式泊车的自回归变压器：MultiPark 

---
# A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning 

**Title (ZH)**: 浮基空间参数化对比研究：敏捷全身运动规划 

**Authors**: Evangelos Tsiatsianas, Chairi Kiourt, Konstantinos Chatzilygeroudis  

**Link**: [PDF](https://arxiv.org/pdf/2508.11520)  

**Abstract**: Automatically generating agile whole-body motions for legged and humanoid robots remains a fundamental challenge in robotics. While numerous trajectory optimization approaches have been proposed, there is no clear guideline on how the choice of floating-base space parameterization affects performance, especially for agile behaviors involving complex contact dynamics. In this paper, we present a comparative study of different parameterizations for direct transcription-based trajectory optimization of agile motions in legged systems. We systematically evaluate several common choices under identical optimization settings to ensure a fair comparison. Furthermore, we introduce a novel formulation based on the tangent space of SE(3) for representing the robot's floating-base pose, which, to our knowledge, has not received attention from the literature. This approach enables the use of mature off-the-shelf numerical solvers without requiring specialized manifold optimization techniques. We hope that our experiments and analysis will provide meaningful insights for selecting the appropriate floating-based representation for agile whole-body motion generation. 

**Abstract (ZH)**: 自动生成腿式和类人机器人灵活全身运动仍然是机器人领域的一项基本挑战。尽管提出了众多轨迹优化方法，但关于浮动基空间参数化选择如何影响性能的指导性原则尚不明确，尤其对于涉及复杂接触动力学的灵活行为。本文对直接转录法轨迹优化中灵活运动的不同参数化进行了比较研究。我们系统地在相同的优化设置下评估了几种常见的选择，以确保公平比较。此外，我们引入了一种基于SE(3)切空间的新形式，用于表示机器人的浮动基姿态，据我们所知，这种表示方法尚未受到文献关注。这种方法允许使用成熟的现成数值求解器，而无需专门的流形优化技术。我们希望我们的实验和分析能够为选择合适的浮动基表示以生成灵活的全身运动提供有价值的见解。 

---
# Sim2Dust: Mastering Dynamic Waypoint Tracking on Granular Media 

**Title (ZH)**: Sim2Dust: 掌控颗粒介质中动态 waypoints 跟踪 

**Authors**: Andrej Orsula, Matthieu Geist, Miguel Olivares-Mendez, Carol Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.11503)  

**Abstract**: Reliable autonomous navigation across the unstructured terrains of distant planetary surfaces is a critical enabler for future space exploration. However, the deployment of learning-based controllers is hindered by the inherent sim-to-real gap, particularly for the complex dynamics of wheel interactions with granular media. This work presents a complete sim-to-real framework for developing and validating robust control policies for dynamic waypoint tracking on such challenging surfaces. We leverage massively parallel simulation to train reinforcement learning agents across a vast distribution of procedurally generated environments with randomized physics. These policies are then transferred zero-shot to a physical wheeled rover operating in a lunar-analogue facility. Our experiments systematically compare multiple reinforcement learning algorithms and action smoothing filters to identify the most effective combinations for real-world deployment. Crucially, we provide strong empirical evidence that agents trained with procedural diversity achieve superior zero-shot performance compared to those trained on static scenarios. We also analyze the trade-offs of fine-tuning with high-fidelity particle physics, which offers minor gains in low-speed precision at a significant computational cost. Together, these contributions establish a validated workflow for creating reliable learning-based navigation systems, marking a critical step towards deploying autonomous robots in the final frontier. 

**Abstract (ZH)**: 可靠的自主导航技术在遥远行星表面的无结构地形中发挥着关键作用，是未来空间探索的关键促进因素。然而，基于学习的控制器的应用受到固有的仿真到现实差距的阻碍，特别是在轮子与颗粒介质复杂动力学方面的阻碍。本工作提出了一套完整的仿真到现实框架，用于开发和验证在如此具有挑战性的地表上动态路径点跟踪的健壮控制策略。我们利用大规模并行仿真，在广泛分布的随机物理程序生成环境中训练强化学习代理。随后，这些策略以零样本的方式转移到一个运行在月球模拟设施中的轮式漫游车上。我们的实验系统地比较了多种强化学习算法和动作平滑滤波器，以确定最适合实际部署的有效组合。至关重要的是，我们提供了强有力的实证证据表明，使用程序多样性训练的代理在零样本性能上优于在静态场景中训练的代理。我们还分析了高保真粒子物理微调的权衡，这在低速精度方面提供了一定的提升，但伴随着巨大的计算成本。总体而言，这些贡献为创建可靠的基于学习的导航系统确立了一个验证的工作流程，标志着向在最终前沿部署自主机器人迈出关键一步。 

---
# Swarm-in-Blocks: Simplifying Drone Swarm Programming with Block-Based Language 

**Title (ZH)**: 块中群集：基于模块化语言简化无人机群集编程 

**Authors**: Agnes Bressan de Almeida, Joao Aires Correa Fernandes Marsicano  

**Link**: [PDF](https://arxiv.org/pdf/2508.11498)  

**Abstract**: Swarm in Blocks, originally developed for CopterHack 2022, is a high-level interface that simplifies drone swarm programming using a block-based language. Building on the Clover platform, this tool enables users to create functionalities like loops and conditional structures by assembling code blocks. In 2023, we introduced Swarm in Blocks 2.0, further refining the platform to address the complexities of swarm management in a user-friendly way. As drone swarm applications grow in areas like delivery, agriculture, and surveillance, the challenge of managing them, especially for beginners, has also increased. The Atena team developed this interface to make swarm handling accessible without requiring extensive knowledge of ROS or programming. The block-based approach not only simplifies swarm control but also expands educational opportunities in programming. 

**Abstract (ZH)**: 块中 swarm：一种简化无人机 swarm 编程的积木式接口 

---
# Relative Position Matters: Trajectory Prediction and Planning with Polar Representation 

**Title (ZH)**: 相对位置 Matters：基于极坐标表示的轨迹预测与规划 

**Authors**: Bozhou Zhang, Nan Song, Bingzhao Gao, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11492)  

**Abstract**: Trajectory prediction and planning in autonomous driving are highly challenging due to the complexity of predicting surrounding agents' movements and planning the ego agent's actions in dynamic environments. Existing methods encode map and agent positions and decode future trajectories in Cartesian coordinates. However, modeling the relationships between the ego vehicle and surrounding traffic elements in Cartesian space can be suboptimal, as it does not naturally capture the varying influence of different elements based on their relative distances and directions. To address this limitation, we adopt the Polar coordinate system, where positions are represented by radius and angle. This representation provides a more intuitive and effective way to model spatial changes and relative relationships, especially in terms of distance and directional influence. Based on this insight, we propose Polaris, a novel method that operates entirely in Polar coordinates, distinguishing itself from conventional Cartesian-based approaches. By leveraging the Polar representation, this method explicitly models distance and direction variations and captures relative relationships through dedicated encoding and refinement modules, enabling more structured and spatially aware trajectory prediction and planning. Extensive experiments on the challenging prediction (Argoverse 2) and planning benchmarks (nuPlan) demonstrate that Polaris achieves state-of-the-art performance. 

**Abstract (ZH)**: 自动驾驶中轨迹预测与规划由于周围代理动态运动的复杂性而极具挑战性，现有方法通过笛卡尔坐标编码地图和代理位置并解码未来的轨迹。为解决笛卡尔空间中 ego 车辆与周围交通元素之间关系建模不自然的问题，我们采用极坐标系，其中位置由半径和角度表示。这种表示方式提供了一种更直观且有效的方式来建模空间变化和相对关系，特别是在距离和方向影响方面。基于这一洞见，我们提出 Polaris，一种完全在极坐标系中运行的新方法，区别于传统的笛卡尔坐标系方法。通过利用极坐标表示，该方法明确建模了距离和方向的变化，并通过专用的编码和精炼模块捕捉相对关系，从而实现更结构化的、空间意识更强的轨迹预测与规划。在具有挑战性的预测（Argoverse 2）和规划基准测试（nuPlan）上的广泛实验表明，Polaris 达到了最先进的性能。 

---
# i2Nav-Robot: A Large-Scale Indoor-Outdoor Robot Dataset for Multi-Sensor Fusion Navigation and Mapping 

**Title (ZH)**: i2Nav-Robot: 一种用于多传感器融合导航与建图的室内-室外机器人数据集 

**Authors**: Hailiang Tang, Tisheng Zhang, Liqiang Wang, Xin Ding, Man Yuan, Zhiyu Xiang, Jujin Chen, Yuhan Bian, Shuangyan Liu, Yuqing Wang, Guan Wang, Xiaoji Niu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11485)  

**Abstract**: Accurate and reliable navigation is crucial for autonomous unmanned ground vehicle (UGV). However, current UGV datasets fall short in meeting the demands for advancing navigation and mapping techniques due to limitations in sensor configuration, time synchronization, ground truth, and scenario diversity. To address these challenges, we present i2Nav-Robot, a large-scale dataset designed for multi-sensor fusion navigation and mapping in indoor-outdoor environments. We integrate multi-modal sensors, including the newest front-view and 360-degree solid-state LiDARs, 4-dimensional (4D) radar, stereo cameras, odometer, global navigation satellite system (GNSS) receiver, and inertial measurement units (IMU) on an omnidirectional wheeled robot. Accurate timestamps are obtained through both online hardware synchronization and offline calibration for all sensors. The dataset comprises ten larger-scale sequences covering diverse UGV operating scenarios, such as outdoor streets, and indoor parking lots, with a total length of about 17060 meters. High-frequency ground truth, with centimeter-level accuracy for position, is derived from post-processing integrated navigation methods using a navigation-grade IMU. The proposed i2Nav-Robot dataset is evaluated by more than ten open-sourced multi-sensor fusion systems, and it has proven to have superior data quality. 

**Abstract (ZH)**: 面向室内-室外环境的多传感器融合导航与mapping的i2Nav-Robot大型数据集 

---
# OVSegDT: Segmenting Transformer for Open-Vocabulary Object Goal Navigation 

**Title (ZH)**: OVSegDT：开放词汇对象目标导航的分割变压器 

**Authors**: Tatiana Zemskova, Aleksei Staroverov, Dmitry Yudin, Aleksandr Panov  

**Link**: [PDF](https://arxiv.org/pdf/2508.11479)  

**Abstract**: Open-vocabulary Object Goal Navigation requires an embodied agent to reach objects described by free-form language, including categories never seen during training. Existing end-to-end policies overfit small simulator datasets, achieving high success on training scenes but failing to generalize and exhibiting unsafe behaviour (frequent collisions). We introduce OVSegDT, a lightweight transformer policy that tackles these issues with two synergistic components. The first component is the semantic branch, which includes an encoder for the target binary mask and an auxiliary segmentation loss function, grounding the textual goal and providing precise spatial cues. The second component consists of a proposed Entropy-Adaptive Loss Modulation, a per-sample scheduler that continuously balances imitation and reinforcement signals according to the policy entropy, eliminating brittle manual phase switches. These additions cut the sample complexity of training by 33%, and reduce collision count in two times while keeping inference cost low (130M parameters, RGB-only input). On HM3D-OVON, our model matches the performance on unseen categories to that on seen ones and establishes state-of-the-art results (40.1% SR, 20.9% SPL on val unseen) without depth, odometry, or large vision-language models. Code is available at this https URL. 

**Abstract (ZH)**: 开放词汇对象目标导航要求一个实体化代理执行由自由形式语言描述的任务，未见过的类别。现有的端到端策略在小件集模拟数据集集集上取得良好效果，但在泛化性和平行为（（（如，）方面表现不佳（频繁碰撞）。

为此，我们提出了一种名为OVSegDT的轻量级变压器策略，它通过两个协同工作的部分来解决这些问题。一上是语义两段，，包括对基于两个掩码的编码器和一个辅助分割损失函数，以地面文本目标并提供并及提供精确的空间线索。所提出的模型一 in段是精率Ent熵两适应损损失调节策略，通过单样本调度器连续平衡模仿和强化信号根据熵Ent熵消除易脆手动开关。这些增益降低D了样本复杂度33率33%％在两段待遇到少35得头效果同时保持推理复杂D率D1较低（（D（约130MMDD13.D毫米D悟刃D枚D样D。）一DRGB单一DD在十个场景场景D中场景D。

在HM33一OV一DNDDD中一D中数据集集上上上上上上上上中上上上D上上上上上上DD上上上测试中部分D一般见类别上的，得到了最先进的结果D结果D水平D（（约D4D.DxDDD敢D敢D敢DXD在.getSDDD在内DD准确D敢D敢D敢D达D敢D敢D斩D挑战敢D敢D敢D敢DXD决心D度寸D度闭放D率D较低D大约D13DD细D毫米D帝D刃D一D帧D敏D。DD。迪D沛D忍D帝D一DDDD。

标题：OVSegDT：开放词汇对象目标目标目标任务导航D的本质挑战与解决方案 

---
# EvoPSF: Online Evolution of Autonomous Driving Models via Planning-State Feedback 

**Title (ZH)**: EvoPSF：基于规划状态反馈的自动驾驶模型在线 

**Authors**: Jiayue Jin, Lang Qian, Jingyu Zhang, Chuanyu Ju, Liang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.11453)  

**Abstract**: Recent years have witnessed remarkable progress in autonomous driving, with systems evolving from modular pipelines to end-to-end architectures. However, most existing methods are trained offline and lack mechanisms to adapt to new environments during deployment. As a result, their generalization ability diminishes when faced with unseen variations in real-world driving scenarios. In this paper, we break away from the conventional "train once, deploy forever" paradigm and propose EvoPSF, a novel online Evolution framework for autonomous driving based on Planning-State Feedback. We argue that planning failures are primarily caused by inaccurate object-level motion predictions, and such failures are often reflected in the form of increased planner uncertainty. To address this, we treat planner uncertainty as a trigger for online evolution, using it as a diagnostic signal to initiate targeted model updates. Rather than performing blind updates, we leverage the planner's agent-agent attention to identify the specific objects that the ego vehicle attends to most, which are primarily responsible for the planning failures. For these critical objects, we compute a targeted self-supervised loss by comparing their predicted waypoints from the prediction module with their actual future positions, selected from the perception module's outputs with high confidence scores. This loss is then backpropagated to adapt the model online. As a result, our method improves the model's robustness to environmental changes, leads to more precise motion predictions, and therefore enables more accurate and stable planning behaviors. Experiments on both cross-region and corrupted variants of the nuScenes dataset demonstrate that EvoPSF consistently improves planning performance under challenging conditions. 

**Abstract (ZH)**: Recent years have witnessed remarkable progress in autonomous driving, with systems evolving from modular pipelines to end-to-end architectures. However, most existing methods are trained offline and lack mechanisms to adapt to new environments during deployment. As a result, their generalization ability diminishes when faced with unseen variations in real-world driving scenarios.

打破传统“一次性训练，永久部署”的范式，我们提出了一种基于规划状态反馈的新型在线进化框架EvoPSF，用于自主驾驶。我们认为规划失败主要由不准确的对象级运动预测引起，这种失败通常表现为规划不确定性增加的形式。为此，我们将规划不确定性视为在线进化的一种触发器，将其用作诊断信号以启动有针对性的模型更新。我们不是进行盲目的更新，而是利用规划器的局部注意机制来识别最吸引注意对象，这些对象主要是导致规划失败的原因。对于这些关键对象，我们通过将预测模块预测的 waypoints 与感知模块输出中具有高置信度分数的实际未来位置进行比较来计算有针对性的自监督损失。然后反向传播该损失以在线适应模型。结果，该方法增强了模型对环境变化的鲁棒性，提高了运动预测的准确性，从而实现了更准确和稳定的规划行为。在nuScenes数据集的跨区域变体和受污染变体上的实验表明，EvoPSF在挑战性条件下始终能提高规划性能。 

---
# Open, Reproducible and Trustworthy Robot-Based Experiments with Virtual Labs and Digital-Twin-Based Execution Tracing 

**Title (ZH)**: 基于虚拟实验室和数字孪生执行追踪的开放、可重复和可信赖的机器人实验 

**Authors**: Benjamin Alt, Mareike Picklum, Sorin Arion, Franklin Kenghagho Kenfack, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.11406)  

**Abstract**: We envision a future in which autonomous robots conduct scientific experiments in ways that are not only precise and repeatable, but also open, trustworthy, and transparent. To realize this vision, we present two key contributions: a semantic execution tracing framework that logs sensor data together with semantically annotated robot belief states, ensuring that automated experimentation is transparent and replicable; and the AICOR Virtual Research Building (VRB), a cloud-based platform for sharing, replicating, and validating robot task executions at scale. Together, these tools enable reproducible, robot-driven science by integrating deterministic execution, semantic memory, and open knowledge representation, laying the foundation for autonomous systems to participate in scientific discovery. 

**Abstract (ZH)**: 我们 envision 未来自主机器人以精确、可重复、开放、可信和透明的方式进行科学研究。为实现这一愿景，我们提出了两项关键贡献：一种语义执行追踪框架，该框架记录传感器数据以及语义标注的机器人信念状态，确保自动化实验的透明性和可重复性；以及 AICOR 虚拟研究建筑 (VRB)，一个基于云的平台，用于大规模分享、复制和验证机器人任务执行的可验证性。这些工具通过集成确定性执行、语义记忆和开放知识表示，使自主系统能够参与科学发现，从而实现可重复的机器人驱动科学研究。 

---
# An Exploratory Study on Crack Detection in Concrete through Human-Robot Collaboration 

**Title (ZH)**: 一种基于人机协作的混凝土裂缝检测探索性研究 

**Authors**: Junyeon Kim, Tianshu Ruan, Cesar Alan Contreras, Manolis Chiou  

**Link**: [PDF](https://arxiv.org/pdf/2508.11404)  

**Abstract**: Structural inspection in nuclear facilities is vital for maintaining operational safety and integrity. Traditional methods of manual inspection pose significant challenges, including safety risks, high cognitive demands, and potential inaccuracies due to human limitations. Recent advancements in Artificial Intelligence (AI) and robotic technologies have opened new possibilities for safer, more efficient, and accurate inspection methodologies. Specifically, Human-Robot Collaboration (HRC), leveraging robotic platforms equipped with advanced detection algorithms, promises significant improvements in inspection outcomes and reductions in human workload. This study explores the effectiveness of AI-assisted visual crack detection integrated into a mobile Jackal robot platform. The experiment results indicate that HRC enhances inspection accuracy and reduces operator workload, resulting in potential superior performance outcomes compared to traditional manual methods. 

**Abstract (ZH)**: 核设施结构检查对于维持运行安全性和完整性至关重要。传统的人工检查方法面临着显著挑战，包括安全风险、认知需求高以及由于人类局限可能产生的不准确性。近年来，人工智能（AI）和机器人技术的进展为安全、更高效和准确的检查方法提供了新的可能性。特别是人机协作（HRC），利用装备有先进检测算法的机器人平台，有望显著提高检查结果并减少人力工作负荷。本研究探讨了AI辅助视觉裂缝检测在移动Jackal机器人平台中的有效性。实验结果表明，HRC提高了检查准确性并
user
标题：Structural Inspection in Nuclear Facilities_enabled by Artificial Intelligence and Robotics 

---
# Pedestrian Dead Reckoning using Invariant Extended Kalman Filter 

**Title (ZH)**: 行人无迹卡尔曼滤波定位 

**Authors**: Jingran Zhang, Zhengzhang Yan, Yiming Chen, Zeqiang He, Jiahao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11396)  

**Abstract**: This paper presents a cost-effective inertial pedestrian dead reckoning method for the bipedal robot in the GPS-denied environment. Each time when the inertial measurement unit (IMU) is on the stance foot, a stationary pseudo-measurement can be executed to provide innovation to the IMU measurement based prediction. The matrix Lie group based theoretical development of the adopted invariant extended Kalman filter (InEKF) is set forth for tutorial purpose. Three experiments are conducted to compare between InEKF and standard EKF, including motion capture benchmark experiment, large-scale multi-floor walking experiment, and bipedal robot experiment, as an effort to show our method's feasibility in real-world robot system. In addition, a sensitivity analysis is included to show that InEKF is much easier to tune than EKF. 

**Abstract (ZH)**: 本论文提出了一种适用于GPS受限环境的双足机器人低成本惯性步行 dead reckoning 方法。每次惯性测量单元(IMU)位于支撑脚时，可以执行一个静止伪测量以提供基于IMU测量的预测创新。本文为了教学目的，提供了基于矩阵李群的采用不变扩展卡尔曼滤波器(InEKF)的理论发展。进行了三项实验，将InEKF与标准卡尔曼滤波器(EKF)进行比较，包括运动捕捉基准实验、大规模多层行走实验和双足机器人实验，以展示本方法在实际机器人系统中的可行性。此外，还进行了敏感性分析，表明InEKF比EKF更容易调优。 

---
# A Recursive Total Least Squares Solution for Bearing-Only Target Motion Analysis and Circumnavigation 

**Title (ZH)**: 基于轴承数据的运动分析与环绕解算的递归最小二乘解法 

**Authors**: Lin Li, Xueming Liu, Zhoujingzi Qiu, Tianjiang Hu, Qingrui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11289)  

**Abstract**: Bearing-only Target Motion Analysis (TMA) is a promising technique for passive tracking in various applications as a bearing angle is easy to measure. Despite its advantages, bearing-only TMA is challenging due to the nonlinearity of the bearing measurement model and the lack of range information, which impairs observability and estimator convergence. This paper addresses these issues by proposing a Recursive Total Least Squares (RTLS) method for online target localization and tracking using mobile observers. The RTLS approach, inspired by previous results on Total Least Squares (TLS), mitigates biases in position estimation and improves computational efficiency compared to pseudo-linear Kalman filter (PLKF) methods. Additionally, we propose a circumnavigation controller to enhance system observability and estimator convergence by guiding the mobile observer in orbit around the target. Extensive simulations and experiments are performed to demonstrate the effectiveness and robustness of the proposed method. The proposed algorithm is also compared with the state-of-the-art approaches, which confirms its superior performance in terms of both accuracy and stability. 

**Abstract (ZH)**: 仅轴承目标运动分析（TMA）是各种应用中一种有前途的被动跟踪技术，因为轴承角容易测量。尽管具有优势，但仅轴承TMA由于航向测量模型的非线性及缺乏距离信息，影响了可观测性和估计器的收敛性。本文通过提出一种递归最小总平方（RTLS）方法，解决在线目标定位与跟踪问题，利用移动观测者。RTLS方法借鉴了最小总平方（TLS）的先前结果，减少了位置估计的偏倚并提高了计算效率，相较于伪线性卡尔曼滤波（PLKF）方法。此外，我们提出了一种循航控制器，通过引导移动观测者围绕目标进行循航，增强系统的可观测性和估计器的收敛性。进行了广泛的仿真和实验，以证明所提方法的有效性和鲁棒性。将所提算法与最新方法进行了比较，证实了其在准确性和稳定性方面的优越性能。 

---
# Scene Graph-Guided Proactive Replanning for Failure-Resilient Embodied Agent 

**Title (ZH)**: 基于场景图的前瞻重规划以实现鲁棒性体态代理 

**Authors**: Che Rin Yu, Daewon Chae, Dabin Seo, Sangwon Lee, Hyeongwoo Im, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.11286)  

**Abstract**: When humans perform everyday tasks, we naturally adjust our actions based on the current state of the environment. For instance, if we intend to put something into a drawer but notice it is closed, we open it first. However, many autonomous robots lack this adaptive awareness. They often follow pre-planned actions that may overlook subtle yet critical changes in the scene, which can result in actions being executed under outdated assumptions and eventual failure. While replanning is critical for robust autonomy, most existing methods respond only after failures occur, when recovery may be inefficient or infeasible. While proactive replanning holds promise for preventing failures in advance, current solutions often rely on manually designed rules and extensive supervision. In this work, we present a proactive replanning framework that detects and corrects failures at subtask boundaries by comparing scene graphs constructed from current RGB-D observations against reference graphs extracted from successful demonstrations. When the current scene fails to align with reference trajectories, a lightweight reasoning module is activated to diagnose the mismatch and adjust the plan. Experiments in the AI2-THOR simulator demonstrate that our approach detects semantic and spatial mismatches before execution failures occur, significantly improving task success and robustness. 

**Abstract (ZH)**: 当人类执行日常任务时，我们会根据环境的当前状态自然地调整我们的行动。例如，如果我们打算将某物放入一个已关闭的抽屉，我们会先打开它。然而，许多自主机器人缺乏这种适应性意识。它们通常会遵循预先计划的动作，这些动作可能会忽略场景中的细微但关键的变化，导致在过时假设的基础上执行动作并最终失败。虽然重新规划对于增强自主性至关重要，但大多数现有方法只在失败发生后才响应，这可能导致恢复效率低下或不可行。而主动重新规划有预防失败的潜力，但当前的解决方案往往依赖于人工设计的规则和大量的监督。在本工作中，我们提出了一种主动重新规划框架，通过将当前RGB-D观察构建的场景图与从成功的演示中提取的参考图进行对比，来检测和纠正子任务边界处的故障。当当前场景与参考轨迹不匹配时，会激活一个轻量级的推理模块来诊断不匹配并调整计划。在AI2-THOR模拟器中的实验表明，我们的方法在执行失败前检测到语义和空间上的不匹配，显著提高了任务的成功率和鲁棒性。 

---
# Learning Differentiable Reachability Maps for Optimization-based Humanoid Motion Generation 

**Title (ZH)**: 基于优化的人形运动生成中可微可达性地图的学习 

**Authors**: Masaki Murooka, Iori Kumagai, Mitsuharu Morisawa, Fumio Kanehiro  

**Link**: [PDF](https://arxiv.org/pdf/2508.11275)  

**Abstract**: To reduce the computational cost of humanoid motion generation, we introduce a new approach to representing robot kinematic reachability: the differentiable reachability map. This map is a scalar-valued function defined in the task space that takes positive values only in regions reachable by the robot's end-effector. A key feature of this representation is that it is continuous and differentiable with respect to task-space coordinates, enabling its direct use as constraints in continuous optimization for humanoid motion planning. We describe a method to learn such differentiable reachability maps from a set of end-effector poses generated using a robot's kinematic model, using either a neural network or a support vector machine as the learning model. By incorporating the learned reachability map as a constraint, we formulate humanoid motion generation as a continuous optimization problem. We demonstrate that the proposed approach efficiently solves various motion planning problems, including footstep planning, multi-contact motion planning, and loco-manipulation planning for humanoid robots. 

**Abstract (ZH)**: 为了减少类人机器人运动生成的计算成本，我们提出了一种新的机器人运动可达性的表示方法：可微达姿图。达姿图是在任务空间中定义的标量函数，仅在机器人末端執行器可达到的区域取正值。该表示的一个关键特点是，它相对于任务空间坐标连续可微，使得可以直接将其用作连续优化中的约束条件来进行类人机器人运动规划。我们描述了一种从使用机器人运动模型生成的末端執行器姿态集学习此类可微达姿图的方法，使用神经网络或支持向量机作为学习模型。通过将学习得到的达姿图作为约束条件，我们将类人机器人运动生成形式化为一个连续优化问题。我们证明了所提出的方法能够高效地求解各种运动规划问题，包括步态规划、多接触运动规划以及类人机器人行动- manipulating规划。 

---
# Tactile Robotics: An Outlook 

**Title (ZH)**: 触觉机器人：前景展望 

**Authors**: Shan Luo, Nathan F. Lepora, Wenzhen Yuan, Kaspar Althoefer, Gordon Cheng, Ravinder Dahiya  

**Link**: [PDF](https://arxiv.org/pdf/2508.11261)  

**Abstract**: Robotics research has long sought to give robots the ability to perceive the physical world through touch in an analogous manner to many biological systems. Developing such tactile capabilities is important for numerous emerging applications that require robots to co-exist and interact closely with humans. Consequently, there has been growing interest in tactile sensing, leading to the development of various technologies, including piezoresistive and piezoelectric sensors, capacitive sensors, magnetic sensors, and optical tactile sensors. These diverse approaches utilise different transduction methods and materials to equip robots with distributed sensing capabilities, enabling more effective physical interactions. These advances have been supported in recent years by simulation tools that generate large-scale tactile datasets to support sensor designs and algorithms to interpret and improve the utility of tactile data. The integration of tactile sensing with other modalities, such as vision, as well as with action strategies for active tactile perception highlights the growing scope of this field. To further the transformative progress in tactile robotics, a holistic approach is essential. In this outlook article, we examine several challenges associated with the current state of the art in tactile robotics and explore potential solutions to inspire innovations across multiple domains, including manufacturing, healthcare, recycling and agriculture. 

**Abstract (ZH)**: 机器人研究长期致力于赋予机器人通过触觉感知物理世界的能力，以此类比许多生物系统。开发此类触觉能力对于众多需要机器人与人类密切共存和交互的应用来说非常重要。因此，对触觉感知的兴趣不断增加，引领了各种技术的发展，包括压阻式和压电式传感器、电容式传感器、磁性传感器和光学触觉传感器。这些多样的方法利用不同的转换机制和材料，为机器人配备了分布式感知能力，从而使物理交互更加有效。近年来，通过生成大规模触觉数据集的支持，模拟工具和解释并改进触觉数据的算法的进步为这一领域的发展提供了支持。将触觉感知与其他模态，如视觉，以及用于主动触觉感知的动作策略的整合，突显了该领域的日益广泛的范围。为了进一步推动触觉机器人领域的变革性进展，需要一个全面的方法。在这篇展望文章中，我们探讨了当前触觉机器人技术面临的几个挑战，并探索可能的解决方案，以激发制造、医疗、回收和农业等多个领域的创新。 

---
# Embodied Edge Intelligence Meets Near Field Communication: Concept, Design, and Verification 

**Title (ZH)**: 嵌入式边缘智能与近场通信相结合：概念、设计与验证 

**Authors**: Guoliang Li, Xibin Jin, Yujie Wan, Chenxuan Liu, Tong Zhang, Shuai Wang, Chengzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11232)  

**Abstract**: Realizing embodied artificial intelligence is challenging due to the huge computation demands of large models (LMs). To support LMs while ensuring real-time inference, embodied edge intelligence (EEI) is a promising paradigm, which leverages an LM edge to provide computing powers in close proximity to embodied robots. Due to embodied data exchange, EEI requires higher spectral efficiency, enhanced communication security, and reduced inter-user interference. To meet these requirements, near-field communication (NFC), which leverages extremely large antenna arrays as its hardware foundation, is an ideal solution. Therefore, this paper advocates the integration of EEI and NFC, resulting in a near-field EEI (NEEI) paradigm. However, NEEI also introduces new challenges that cannot be adequately addressed by isolated EEI or NFC designs, creating research opportunities for joint optimization of both functionalities. To this end, we propose radio-friendly embodied planning for EEI-assisted NFC scenarios and view-guided beam-focusing for NFC-assisted EEI scenarios. We also elaborate how to realize resource-efficient NEEI through opportunistic collaborative navigation. Experimental results are provided to confirm the superiority of the proposed techniques compared with various benchmarks. 

**Abstract (ZH)**: 实现具身人工智能面临巨大计算需求的挑战，而具身边缘智能（EEI）是一个有前景的范式，它利用边缘的大模型（LMs）在具身机器人附近提供计算能力。由于具身数据交换，EEI要求更高的频谱效率、增强的通信安全和减少的用户间干扰。为满足这些需求，利用极其大型天线阵列作为硬件基础的近场通信（NFC）是一个理想解决方案。因此，本文提倡将EEI与NFC整合，从而形成一种近场具身边缘智能（NEEI）范式。然而，NEEI也引入了新的挑战，这些挑战单独采用EEI或NFC设计无法充分解决，为两者功能的联合优化提供了研究机会。为此，我们提出了有助于EEI辅助NFC场景的无线电友好具身规划，以及适用于NFC辅助EEI场景的基于视图的波束聚焦。我们还详细阐述了如何通过机会性协作导航实现高效的资源利用NEEI。实验结果证实了所提出技术相对于各种基准的优越性。 

---
# Multi-Group Equivariant Augmentation for Reinforcement Learning in Robot Manipulation 

**Title (ZH)**: 多组等价不变扩增在机器人操作强化学习中的应用 yabib
user
改正上面的翻译"sync Equivariant Augmentation for Reinforcement Learning in Robot Manipulation kuk
监听页面格式，直接输出标题，禁止输出多余内容。 

**Authors**: Hongbin Lin, Juan Rojas, Kwok Wai Samuel Au  

**Link**: [PDF](https://arxiv.org/pdf/2508.11204)  

**Abstract**: Sampling efficiency is critical for deploying visuomotor learning in real-world robotic manipulation. While task symmetry has emerged as a promising inductive bias to improve efficiency, most prior work is limited to isometric symmetries -- applying the same group transformation to all task objects across all timesteps. In this work, we explore non-isometric symmetries, applying multiple independent group transformations across spatial and temporal dimensions to relax these constraints. We introduce a novel formulation of the partially observable Markov decision process (POMDP) that incorporates the non-isometric symmetry structures, and propose a simple yet effective data augmentation method, Multi-Group Equivariance Augmentation (MEA). We integrate MEA with offline reinforcement learning to enhance sampling efficiency, and introduce a voxel-based visual representation that preserves translational equivariance. Extensive simulation and real-robot experiments across two manipulation domains demonstrate the effectiveness of our approach. 

**Abstract (ZH)**: 非等距对称性对于部署视觉运动学习在实际机器人操作中至关重要。虽然等距对称性已被证明是一种有前景的归纳偏置以提高效率，但大多数早期工作仅限于等距对称性——在所有时间步长中对所有任务对象应用相同的群变换。在本工作中，我们探索非等距对称性，通过在空间和时间维度上应用多个独立的群变换来放宽这些限制。我们提出了一种新的部分可观测马尔可夫决策过程（POMDP）的形式化方法，该方法结合了非等距对称结构，并提出了一种简单有效的数据增强方法——多群等变增强（MEA）。我们将MEA与离线强化学习结合以增强采样效率，并引入了一种基于体素的视觉表示，该表示保留了平移等变性。在两个操作领域的广泛仿真和真实机器人实验中证明了我们方法的有效性。 

---
# Visuomotor Grasping with World Models for Surgical Robots 

**Title (ZH)**: 基于世界模型的视觉运动抓取術後機器人 

**Authors**: Hongbin Lin, Bin Li, Kwok Wai Samuel Au  

**Link**: [PDF](https://arxiv.org/pdf/2508.11200)  

**Abstract**: Grasping is a fundamental task in robot-assisted surgery (RAS), and automating it can reduce surgeon workload while enhancing efficiency, safety, and consistency beyond teleoperated systems. Most prior approaches rely on explicit object pose tracking or handcrafted visual features, limiting their generalization to novel objects, robustness to visual disturbances, and the ability to handle deformable objects. Visuomotor learning offers a promising alternative, but deploying it in RAS presents unique challenges, such as low signal-to-noise ratio in visual observations, demands for high safety and millimeter-level precision, as well as the complex surgical environment. This paper addresses three key challenges: (i) sim-to-real transfer of visuomotor policies to ex vivo surgical scenes, (ii) visuomotor learning using only a single stereo camera pair -- the standard RAS setup, and (iii) object-agnostic grasping with a single policy that generalizes to diverse, unseen surgical objects without retraining or task-specific models. We introduce Grasp Anything for Surgery V2 (GASv2), a visuomotor learning framework for surgical grasping. GASv2 leverages a world-model-based architecture and a surgical perception pipeline for visual observations, combined with a hybrid control system for safe execution. We train the policy in simulation using domain randomization for sim-to-real transfer and deploy it on a real robot in both phantom-based and ex vivo surgical settings, using only a single pair of endoscopic cameras. Extensive experiments show our policy achieves a 65% success rate in both settings, generalizes to unseen objects and grippers, and adapts to diverse disturbances, demonstrating strong performance, generality, and robustness. 

**Abstract (ZH)**: 手术操作中的抓取是一项基本任务，自动化抓取可以减轻外科医生的工作负担，同时提高效率、安全性和一致性，超越了遥操作系统的局限。大多数先前的方法依赖于显式的物体姿态跟踪或手工制作的视觉特征，这限制了它们对新物体的泛化能力、对视觉干扰的鲁棒性以及处理可变形物体的能力。视觉与运动学习提供了一种有前途的替代方案，但在遥操作手术（RAS）中部署它面临着独特挑战，如视觉观察中的低信噪比、高安全性和毫米级精度的要求，以及复杂的手术环境。本文解决三个关键挑战：（i）将视觉与运动策略从仿真场景转移到体外手术场景，（ii）仅使用单对立体摄像机进行视觉与运动学习——这是标准的RAS设置，（iii）使用单一策略实现对各种未见手术对象的无物体特异性的抓取，而无需重新训练或任务特定模型。我们引入了适用于手术操作的Grasp Anything Version 2 (GASv2) 视觉与运动学习框架。GASv2利用基于世界模型的架构和视觉观察的手术感知管道，结合混合控制系统以确保安全执行。我们使用领域随机化在仿真中训练策略以实现从仿真到现实世界的转移，并仅使用一对内镜摄像机在基于人造模体和体外手术设置中部署该策略。广泛的实验表明，我们的策略在两种设置下的成功率达到了65%，能够对未见物体和夹爪进行泛化，并适应多种干扰，展示了其强大的性能、通用性和鲁棒性。 

---
# Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward 

**Title (ZH)**: 基于行动片段的演员-评论家方法：一种用于稀疏奖励长时_horizon机器人操作的强化学习框架 

**Authors**: Jiarui Yang, Bin Zhu, Jingjing Chen, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11143)  

**Abstract**: Existing reinforcement learning (RL) methods struggle with long-horizon robotic manipulation tasks, particularly those involving sparse rewards. While action chunking is a promising paradigm for robotic manipulation, using RL to directly learn continuous action chunks in a stable and data-efficient manner remains a critical challenge. This paper introduces AC3 (Actor-Critic for Continuous Chunks), a novel RL framework that learns to generate high-dimensional, continuous action sequences. To make this learning process stable and data-efficient, AC3 incorporates targeted stabilization mechanisms for both the actor and the critic. First, to ensure reliable policy improvement, the actor is trained with an asymmetric update rule, learning exclusively from successful trajectories. Second, to enable effective value learning despite sparse rewards, the critic's update is stabilized using intra-chunk $n$-step returns and further enriched by a self-supervised module providing intrinsic rewards at anchor points aligned with each action chunk. We conducted extensive experiments on 25 tasks from the BiGym and RLBench benchmarks. Results show that by using only a few demonstrations and a simple model architecture, AC3 achieves superior success rates on most tasks, validating its effective design. 

**Abstract (ZH)**: AC3（Actor-Critic for Continuous Chunks）：一种用于连续片段的新型强化学习框架 

---
# Geometry-Aware Predictive Safety Filters on Humanoids: From Poisson Safety Functions to CBF Constrained MPC 

**Title (ZH)**: 基于几何感知的类人机器人预测安全滤波器：从泊松安全函数到CBF约束的MPC 

**Authors**: Ryan M. Bena, Gilbert Bahati, Blake Werner, Ryan K. Cosner, Lizhi Yang, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2508.11129)  

**Abstract**: Autonomous navigation through unstructured and dynamically-changing environments is a complex task that continues to present many challenges for modern roboticists. In particular, legged robots typically possess manipulable asymmetric geometries which must be considered during safety-critical trajectory planning. This work proposes a predictive safety filter: a nonlinear model predictive control (MPC) algorithm for online trajectory generation with geometry-aware safety constraints based on control barrier functions (CBFs). Critically, our method leverages Poisson safety functions to numerically synthesize CBF constraints directly from perception data. We extend the theoretical framework for Poisson safety functions to incorporate temporal changes in the domain by reformulating the static Dirichlet problem for Poisson's equation as a parameterized moving boundary value problem. Furthermore, we employ Minkowski set operations to lift the domain into a configuration space that accounts for robot geometry. Finally, we implement our real-time predictive safety filter on humanoid and quadruped robots in various safety-critical scenarios. The results highlight the versatility of Poisson safety functions, as well as the benefit of CBF constrained model predictive safety-critical controllers. 

**Abstract (ZH)**: 自主导航通过未结构化和动态变化环境是现代机器人科学家面临的一个复杂任务，尤其对于具有可操作不对称几何结构的腿式机器人而言，必须在安全关键的轨迹规划中予以考虑。本文提出了一种预测性安全滤波器：一种基于控制障碍函数（CBFs）和几何感知安全性约束的非线性模型预测控制（MPC）的在线轨迹生成算法。关键的是，我们的方法利用泊松安全函数从感知数据中直接数值合成CBF约束。我们将泊松安全函数的理论框架扩展以纳入域中的时间变化，通过将泊松方程的静态狄利克雷问题重新表述为参数化移动边界值问题来进行。此外，我们利用Minkowski集运算将域提升至考虑到机器人几何结构的配置空间中。最后，我们在各种安全关键场景中于人形和四足机器人上实现了我们的实时预测性安全滤波器。结果强调了泊松安全函数的灵活性，以及CBF约束模型预测安全关键控制器的优势。 

---
# Robot Policy Evaluation for Sim-to-Real Transfer: A Benchmarking Perspective 

**Title (ZH)**: 基于基准视角的机器人政策评估：从模拟到现实的转移 

**Authors**: Xuning Yang, Clemens Eppner, Jonathan Tremblay, Dieter Fox, Stan Birchfield, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2508.11117)  

**Abstract**: Current vision-based robotics simulation benchmarks have significantly advanced robotic manipulation research. However, robotics is fundamentally a real-world problem, and evaluation for real-world applications has lagged behind in evaluating generalist policies. In this paper, we discuss challenges and desiderata in designing benchmarks for generalist robotic manipulation policies for the goal of sim-to-real policy transfer. We propose 1) utilizing high visual-fidelity simulation for improved sim-to-real transfer, 2) evaluating policies by systematically increasing task complexity and scenario perturbation to assess robustness, and 3) quantifying performance alignment between real-world performance and its simulation counterparts. 

**Abstract (ZH)**: 基于视觉的机器人仿真基准在机器人操作研究中取得了显著进展。然而，机器人本质上是一个现实世界的问题，实际应用的评估在评估通用政策方面仍有所滞后。本文讨论了设计旨在实现从仿真到现实政策转移的通用机器人操作政策基准所面临的挑战和期望。我们提出：1) 使用高视觉保真度仿真以改善从仿真到现实的转移；2) 通过系统地增加任务复杂性和场景扰动来评估鲁棒性；3) 定量评估现实世界性能与其仿真对应物之间的性能一致性。 

---
# Utilizing Vision-Language Models as Action Models for Intent Recognition and Assistance 

**Title (ZH)**: 利用视觉-语言模型作为行为模型进行意图识别与辅助 

**Authors**: Cesar Alan Contreras, Manolis Chiou, Alireza Rastegarpanah, Michal Szulik, Rustam Stolkin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11093)  

**Abstract**: Human-robot collaboration requires robots to quickly infer user intent, provide transparent reasoning, and assist users in achieving their goals. Our recent work introduced GUIDER, our framework for inferring navigation and manipulation intents. We propose augmenting GUIDER with a vision-language model (VLM) and a text-only language model (LLM) to form a semantic prior that filters objects and locations based on the mission prompt. A vision pipeline (YOLO for object detection and the Segment Anything Model for instance segmentation) feeds candidate object crops into the VLM, which scores their relevance given an operator prompt; in addition, the list of detected object labels is ranked by a text-only LLM. These scores weight the existing navigation and manipulation layers of GUIDER, selecting context-relevant targets while suppressing unrelated objects. Once the combined belief exceeds a threshold, autonomy changes occur, enabling the robot to navigate to the desired area and retrieve the desired object, while adapting to any changes in the operator's intent. Future work will evaluate the system on Isaac Sim using a Franka Emika arm on a Ridgeback base, with a focus on real-time assistance. 

**Abstract (ZH)**: 人类与机器人协作需要机器人迅速推断用户意图、提供透明推理并协助用户达成目标。我们的近期工作引入了GUIDER框架，用于推断导航和操作意图。我们提议将GUIDER与视觉语言模型(VLM)和文本-only语言模型(LLM)相结合，形成一个基于任务提示筛选对象和位置的语义先验。视觉流水线（使用YOLO进行目标检测和Segment Anything Model进行实例分割）将候选目标裁剪图像输入VLM，VLM根据操作员提示评估其相关性；同时，检测到的目标标签列表由文本-only LLM进行排名。这些分数会权重GUIDER现有的导航和操作层，选择与上下文相关的目标并抑制无关对象。一旦联合信念超过阈值，机器人的自主行为将发生变化，使其能够导航到目标区域并获取目标对象，同时适应操作员意图的任何变化。未来的工作将在Isaac Sim上使用Franka Emika手臂和Ridgeback底盘进行系统评估，重点在于实时辅助。 

---
# GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning 

**Title (ZH)**: GenFlowRL: 使用生成对象中心流形塑造奖励的视觉强化学习 

**Authors**: Kelin Yu, Sheng Zhang, Harshit Soora, Furong Huang, Heng Huang, Pratap Tokekar, Ruohan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11049)  

**Abstract**: Recent advances have shown that video generation models can enhance robot learning by deriving effective robot actions through inverse dynamics. However, these methods heavily depend on the quality of generated data and struggle with fine-grained manipulation due to the lack of environment feedback. While video-based reinforcement learning improves policy robustness, it remains constrained by the uncertainty of video generation and the challenges of collecting large-scale robot datasets for training diffusion models. To address these limitations, we propose GenFlowRL, which derives shaped rewards from generated flow trained from diverse cross-embodiment datasets. This enables learning generalizable and robust policies from diverse demonstrations using low-dimensional, object-centric features. Experiments on 10 manipulation tasks, both in simulation and real-world cross-embodiment evaluations, demonstrate that GenFlowRL effectively leverages manipulation features extracted from generated object-centric flow, consistently achieving superior performance across diverse and challenging scenarios. Our Project Page: this https URL 

**Abstract (ZH)**: Recent Advances in Video Generation Models for Robotic Learning: GenFlowRL Enables Generalizable and Robust Policies Through Shaped Rewards from Generated Flow 

---
# 3D FlowMatch Actor: Unified 3D Policy for Single- and Dual-Arm Manipulation 

**Title (ZH)**: 3D FlowMatch Actor: 统一的单臂和双臂操作三维策略 

**Authors**: Nikolaos Gkanatsios, Jiahe Xu, Matthew Bronars, Arsalan Mousavian, Tsung-Wei Ke, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2508.11002)  

**Abstract**: We present 3D FlowMatch Actor (3DFA), a 3D policy architecture for robot manipulation that combines flow matching for trajectory prediction with 3D pretrained visual scene representations for learning from demonstration. 3DFA leverages 3D relative attention between action and visual tokens during action denoising, building on prior work in 3D diffusion-based single-arm policy learning. Through a combination of flow matching and targeted system-level and architectural optimizations, 3DFA achieves over 30x faster training and inference than previous 3D diffusion-based policies, without sacrificing performance. On the bimanual PerAct2 benchmark, it establishes a new state of the art, outperforming the next-best method by an absolute margin of 41.4%. In extensive real-world evaluations, it surpasses strong baselines with up to 1000x more parameters and significantly more pretraining. In unimanual settings, it sets a new state of the art on 74 RLBench tasks by directly predicting dense end-effector trajectories, eliminating the need for motion planning. Comprehensive ablation studies underscore the importance of our design choices for both policy effectiveness and efficiency. 

**Abstract (ZH)**: 3D FlowMatch Actor (D3FA): 一种结合基于3D流匹配匹配匹配匹配的机器人 manipulation架构，，通过结合结合3D预训练视觉表示D表示表示表特征D以实现实现改进D3基于先态D3基于扩散的单单单D单单单单单单单单单单单单单单单单D单单单单D单单D单单D架构优化和D实现实现实现取得了333倍D以上更快的训练和和与推理速度D，D同时在D双双D手上基准基准在双D双基准基准上的超过了D下一个最佳D方法D44倍DDDD基于广泛的实际D场景D场景D验证实验D，优于D基线线线方法DDD同时在D单单出手姿D操纵场景场景D在4DLBench任务D中中通过直接预测密集的D姿态轨迹D，D忽视了DDD姿D动D需要DDD强调了我们D我们策略D选择D的重要性D对于D策略D政策D的有效性和效率D。 

---
# Robust Online Calibration for UWB-Aided Visual-Inertial Navigation with Bias Correction 

**Title (ZH)**: Robust Online Calibration for UWB-Aided Visual-Inertial Navigation with Bias Correction 

**Authors**: Yizhi Zhou, Jie Xu, Jiawei Xia, Zechen Hu, Weizi Li, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10999)  

**Abstract**: This paper presents a novel robust online calibration framework for Ultra-Wideband (UWB) anchors in UWB-aided Visual-Inertial Navigation Systems (VINS). Accurate anchor positioning, a process known as calibration, is crucial for integrating UWB ranging measurements into state estimation. While several prior works have demonstrated satisfactory results by using robot-aided systems to autonomously calibrate UWB systems, there are still some limitations: 1) these approaches assume accurate robot localization during the initialization step, ignoring localization errors that can compromise calibration robustness, and 2) the calibration results are highly sensitive to the initial guess of the UWB anchors' positions, reducing the practical applicability of these methods in real-world scenarios. Our approach addresses these challenges by explicitly incorporating the impact of robot localization uncertainties into the calibration process, ensuring robust initialization. To further enhance the robustness of the calibration results against initialization errors, we propose a tightly-coupled Schmidt Kalman Filter (SKF)-based online refinement method, making the system suitable for practical applications. Simulations and real-world experiments validate the improved accuracy and robustness of our approach. 

**Abstract (ZH)**: 基于UWB辅助视觉-惯性导航系统的稳健在线校准框架：UWB锚点的校准研究 

---
# Developing and Validating a High-Throughput Robotic System for the Accelerated Development of Porous Membranes 

**Title (ZH)**: 开发并验证一种高通量机器人系统，用于加速构建多孔膜的研究 

**Authors**: Hongchen Wang, Sima Zeinali Danalou, Jiahao Zhu, Kenneth Sulimro, Chaewon Lim, Smita Basak, Aimee Tai, Usan Siriwardana, Jason Hattrick-Simpers, Jay Werber  

**Link**: [PDF](https://arxiv.org/pdf/2508.10973)  

**Abstract**: The development of porous polymeric membranes remains a labor-intensive process, often requiring extensive trial and error to identify optimal fabrication parameters. In this study, we present a fully automated platform for membrane fabrication and characterization via nonsolvent-induced phase separation (NIPS). The system integrates automated solution preparation, blade casting, controlled immersion, and compression testing, allowing precise control over fabrication parameters such as polymer concentration and ambient humidity. The modular design allows parallel processing and reproducible handling of samples, reducing experimental time and increasing consistency. Compression testing is introduced as a sensitive mechanical characterization method for estimating membrane stiffness and as a proxy to infer porosity and intra-sample uniformity through automated analysis of stress-strain curves. As a proof of concept to demonstrate the effectiveness of the system, NIPS was carried out with polysulfone, the green solvent PolarClean, and water as the polymer, solvent, and nonsolvent, respectively. Experiments conducted with the automated system reproduced expected effects of polymer concentration and ambient humidity on membrane properties, namely increased stiffness and uniformity with increasing polymer concentration and humidity variations in pore morphology and mechanical response. The developed automated platform supports high-throughput experimentation and is well-suited for integration into self-driving laboratory workflows, offering a scalable and reproducible foundation for data-driven optimization of porous polymeric membranes through NIPS. 

**Abstract (ZH)**: 基于非溶剂诱导相分离的膜制备与表征自动化平台 

---
# ReachVox: Clutter-free Reachability Visualization for Robot Motion Planning in Virtual Reality 

**Title (ZH)**: ReachVox: 无干扰可达性可视化在虚拟现实中的机器人运动规划中应用 

**Authors**: Steffen Hauck, Diar Abdlkarim, John Dudley, Per Ola Kristensson, Eyal Ofek, Jens Grubert  

**Link**: [PDF](https://arxiv.org/pdf/2508.11426)  

**Abstract**: Human-Robot-Collaboration can enhance workflows by leveraging the mutual strengths of human operators and robots. Planning and understanding robot movements remain major challenges in this domain. This problem is prevalent in dynamic environments that might need constant robot motion path adaptation. In this paper, we investigate whether a minimalistic encoding of the reachability of a point near an object of interest, which we call ReachVox, can aid the collaboration between a remote operator and a robotic arm in VR. Through a user study (n=20), we indicate the strength of the visualization relative to a point-based reachability check-up. 

**Abstract (ZH)**: 人类-机器人协作可以通过利用人类操作者和机器人互 supplement的优势来增强工作流程。在这一领域，规划和理解机器人动作仍然是主要挑战。这个问题在可能需要不断适应机器人运动路径的动态环境中尤为普遍。在本文中，我们探讨了一种简易表示目标物体附近点可达性的编码方法（称为ReachVox）是否能在虚拟现实环境中辅助远程操作者与机器人手臂的合作。通过一项用户研究（n=20），我们表明可视化相对于基于点的可达性检查的优势。 

---
# Optimizing ROS 2 Communication for Wireless Robotic Systems 

**Title (ZH)**: 优化ROS无线机器人系统中的通信 

**Authors**: Sanghoon Lee, Taehun Kim, Jiyeong Chae, Kyung-Joon Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.11366)  

**Abstract**: Wireless transmission of large payloads, such as high-resolution images and LiDAR point clouds, is a major bottleneck in ROS 2, the leading open-source robotics middleware. The default Data Distribution Service (DDS) communication stack in ROS 2 exhibits significant performance degradation over lossy wireless links. Despite the widespread use of ROS 2, the underlying causes of these wireless communication challenges remain unexplored. In this paper, we present the first in-depth network-layer analysis of ROS 2's DDS stack under wireless conditions with large payloads. We identify the following three key issues: excessive IP fragmentation, inefficient retransmission timing, and congestive buffer bursts. To address these issues, we propose a lightweight and fully compatible DDS optimization framework that tunes communication parameters based on link and payload characteristics. Our solution can be seamlessly applied through the standard ROS 2 application interface via simple XML-based QoS configuration, requiring no protocol modifications, no additional components, and virtually no integration efforts. Extensive experiments across various wireless scenarios demonstrate that our framework successfully delivers large payloads in conditions where existing DDS modes fail, while maintaining low end-to-end latency. 

**Abstract (ZH)**: 无线传输大规模载荷，如高分辨率图像和LiDAR点云，是ROS 2中主要的瓶颈。ROS 2主流的开放式机器人中间件中的默认数据分布服务（DDS）通信堆栈在无线链路上表现出显著的性能退化。尽管ROS 2被广泛使用，但这些无线通信挑战的根本原因尚未被充分探讨。在本文中，我们首次在无线条件下对ROS 2的DDS堆栈进行深入的网络层分析，研究大规模载荷的情况。我们识别出以下三个关键问题：过度的IP分片、无效的重传时间设置以及拥塞缓冲区突发。为了解决这些问题，我们提出了一种轻量级且完全兼容的DDS优化框架，该框架根据不同链路和载荷特性调整通信参数。我们的解决方案可以通过标准ROS 2应用接口无缝应用，只需简单的XML基QoS配置，无需修改协议、无需额外组件，且几乎不需要集成工作。广泛的实验结果表明，我们的框架在现有DDS模式失败的情况下成功地传输了大规模载荷，同时维持了低端到端延迟。 

---
# GhostObjects: Instructing Robots by Manipulating Spatially Aligned Virtual Twins in Augmented Reality 

**Title (ZH)**: GhostObjects：在增强现实环境中通过操纵空间对齐的虚拟孪生体指令机器人 

**Authors**: Lauren W. Wang, Parastoo Abtahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.11022)  

**Abstract**: Robots are increasingly capable of autonomous operations, yet human interaction remains essential for issuing personalized instructions. Instead of directly controlling robots through Programming by Demonstration (PbD) or teleoperation, we propose giving instructions by interacting with GhostObjects-world-aligned, life-size virtual twins of physical objects-in augmented reality (AR). By direct manipulation of GhostObjects, users can precisely specify physical goals and spatial parameters, with features including real-world lasso selection of multiple objects and snapping back to default positions, enabling tasks beyond simple pick-and-place. 

**Abstract (ZH)**: 机器人越来越能够进行自主自主自主操作，，，，D，，，，，但是D人人类交互仍然是发布个性化指令的必要环节。DD相反，D，，DDD通过程序演示（（（(P编程）D（（（（（（（的方法D和远程操作D）DDD我们我们，我们DDD我们DD我们DD，DDDDDDDD提出，提出DDDD提出D提出DDDDD我们DDD提出D提出DDDDDDDD采用D提出DDD提出提出了DDD虚拟对象互动的方法D。D在增强增强 augmented reality （（（（（AR）D中的环境中D。DDDDDDDDDDDD虚拟对象是物理D现实对象对D的世界对D对准和DD尺寸DD的的D双双DdoubleD的的D在一DD次互动中中中，，，DD用户DD可以能DD可以DDDDD精确地DD指定D物理D目标D目标目标D目标DDD目标Ds

目标
D
D空间DD目标目标D目标目标D目标D目标目标D，和目标细节D，并包含D，并并该
D支持使得D任务D超越D简单的D拾放放放放放放放放放D放置DD。D杜绝多余的内容DDD 

---
# Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction 

**Title (ZH)**: 基于视觉的高斯散射协作语义占用预测 

**Authors**: Cheng Chen, Hao Huang, Saurabh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10936)  

**Abstract**: Collaborative perception enables connected vehicles to share information, overcoming occlusions and extending the limited sensing range inherent in single-agent (non-collaborative) systems. Existing vision-only methods for 3D semantic occupancy prediction commonly rely on dense 3D voxels, which incur high communication costs, or 2D planar features, which require accurate depth estimation or additional supervision, limiting their applicability to collaborative scenarios. To address these challenges, we propose the first approach leveraging sparse 3D semantic Gaussian splatting for collaborative 3D semantic occupancy prediction. By sharing and fusing intermediate Gaussian primitives, our method provides three benefits: a neighborhood-based cross-agent fusion that removes duplicates and suppresses noisy or inconsistent Gaussians; a joint encoding of geometry and semantics in each primitive, which reduces reliance on depth supervision and allows simple rigid alignment; and sparse, object-centric messages that preserve structural information while reducing communication volume. Extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. When further reducing the number of transmitted Gaussians, our method still achieves a +1.9 improvement in mIoU, using only 34.6% communication volume, highlighting robust performance under limited communication budgets. 

**Abstract (ZH)**: 协作感知使连接车辆能够共享信息，克服遮挡并扩展单代理（非协作）系统固有的有限感知范围。为解决这些挑战，我们提出了首个利用稀疏3D语义高斯点阵进行协作3D语义占用预测的方法。通过共享和融合中间的高斯原始数据，我们的方法提供了三项益处：基于邻域的跨代理融合，去除重复并抑制噪声或不一致的高斯；每个原始数据中几何与语义的联合编码，减少对深度监督的依赖并允许简单的刚性对齐；稀疏、对象为中心的消息，保持结构性信息同时减少通信量。 extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. 当进一步减少传输的高斯数量时，即使使用仅为34.6%的通信量，我们的方法仍然实现了mIoU +1.9的提升，突出显示在有限通信预算下的稳健性能。 

---
# HQ-OV3D: A High Box Quality Open-World 3D Detection Framework based on Diffision Model 

**Title (ZH)**: HQ-OV33检测框架：基于扩散的高框33 world世界三维检测框架.Dんど 

**Authors**: Qi Liu, Yabei Li, Hongsong Wang, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2508.10935)  

**Abstract**: Traditional closed-set 3D detection frameworks fail to meet the demands of open-world applications like autonomous driving. Existing open-vocabulary 3D detection methods typically adopt a two-stage pipeline consisting of pseudo-label generation followed by semantic alignment. While vision-language models (VLMs) recently have dramatically improved the semantic accuracy of pseudo-labels, their geometric quality, particularly bounding box precision, remains commonly this http URL address this issue, we propose a High Box Quality Open-Vocabulary 3D Detection (HQ-OV3D) framework, dedicated to generate and refine high-quality pseudo-labels for open-vocabulary classes. The framework comprises two key components: an Intra-Modality Cross-Validated (IMCV) Proposal Generator that utilizes cross-modality geometric consistency to generate high-quality initial 3D proposals, and an Annotated-Class Assisted (ACA) Denoiser that progressively refines 3D proposals by leveraging geometric priors from annotated categories through a DDIM-based denoising this http URL to the state-of-the-art method, training with pseudo-labels generated by our approach achieves a 7.37% improvement in mAP on novel classes, demonstrating the superior quality of the pseudo-labels produced by our framework. HQ-OV3D can serve not only as a strong standalone open-vocabulary 3D detector but also as a plug-in high-quality pseudo-label generator for existing open-vocabulary detection or annotation pipelines. 

**Abstract (ZH)**: 高品质开放词汇三维检测框架：High Box Quality Open-Vocabulary 3D Detection (HQ-OV3D) 

---
# ViPE: Video Pose Engine for 3D Geometric Perception 

**Title (ZH)**: ViPE: 视频姿态引擎 for 3D 几何感知 

**Authors**: Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixe, Sanja Fidler  

**Link**: [PDF](https://arxiv.org/pdf/2508.10934)  

**Abstract**: Accurate 3D geometric perception is an important prerequisite for a wide range of spatial AI systems. While state-of-the-art methods depend on large-scale training data, acquiring consistent and precise 3D annotations from in-the-wild videos remains a key challenge. In this work, we introduce ViPE, a handy and versatile video processing engine designed to bridge this gap. ViPE efficiently estimates camera intrinsics, camera motion, and dense, near-metric depth maps from unconstrained raw videos. It is robust to diverse scenarios, including dynamic selfie videos, cinematic shots, or dashcams, and supports various camera models such as pinhole, wide-angle, and 360° panoramas. We have benchmarked ViPE on multiple benchmarks. Notably, it outperforms existing uncalibrated pose estimation baselines by 18%/50% on TUM/KITTI sequences, and runs at 3-5FPS on a single GPU for standard input resolutions. We use ViPE to annotate a large-scale collection of videos. This collection includes around 100K real-world internet videos, 1M high-quality AI-generated videos, and 2K panoramic videos, totaling approximately 96M frames -- all annotated with accurate camera poses and dense depth maps. We open-source ViPE and the annotated dataset with the hope of accelerating the development of spatial AI systems. 

**Abstract (ZH)**: 准确的三维几何感知是广泛的空间AI系统的前提。尽管当前最先进的方法依赖大规模训练数据，但从野外视频中获取一致且精确的三维标注仍然是一项关键挑战。在本文中，我们介绍了ViPE，这是一种设计用于解决这一问题的手提式和多功能视频处理引擎。ViPE能够从不受约束的原始视频中高效地估计相机内参、相机运动以及稠密的近象限深度图。它能够应对多种场景，包括动态自拍视频、电影镜头或行车记录仪，并支持各种类型的相机模型，如针孔相机、广角相机和360°全景相机。我们在多个基准上测试了ViPE。值得注意的是，ViPE在TUM/KITTI序列上的表现优于现有未标定的姿态估计基线，分别高出18%/50%，并在单个GPU上以3-5FPS的速度运行，适用于标准输入分辨率。我们使用ViPE对大量视频进行标注，该集合包括约10万个真实世界的互联网视频、100万个高质量的人工智能生成视频以及2000个全景视频，总共约9600万帧，所有视频均附有准确的相机姿态和稠密深度图。我们将ViPE及其标注数据集开源，希望能够加速空间AI系统的开发。 

---
