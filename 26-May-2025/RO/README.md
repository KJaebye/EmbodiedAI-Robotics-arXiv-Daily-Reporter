# ExoGait-MS: Learning Periodic Dynamics with Multi-Scale Graph Network for Exoskeleton Gait Recognition 

**Title (ZH)**: ExoGait-MS：基于多尺度图网络学习周期动力学的外骨骼步态识别方法 

**Authors**: Lijiang Liu, Junyu Shi, Yong Sun, Zhiyuan Zhang, Jinni Zhou, Shugen Ma, Qiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.18018)  

**Abstract**: Current exoskeleton control methods often face challenges in delivering personalized treatment. Standardized walking gaits can lead to patient discomfort or even injury. Therefore, personalized gait is essential for the effectiveness of exoskeleton robots, as it directly impacts their adaptability, comfort, and rehabilitation outcomes for individual users. To enable personalized treatment in exoskeleton-assisted therapy and related applications, accurate recognition of personal gait is crucial for implementing tailored gait control. The key challenge in gait recognition lies in effectively capturing individual differences in subtle gait features caused by joint synergy, such as step frequency and step length. To tackle this issue, we propose a novel approach, which uses Multi-Scale Global Dense Graph Convolutional Networks (GCN) in the spatial domain to identify latent joint synergy patterns. Moreover, we propose a Gait Non-linear Periodic Dynamics Learning module to effectively capture the periodic characteristics of gait in the temporal domain. To support our individual gait recognition task, we have constructed a comprehensive gait dataset that ensures both completeness and reliability. Our experimental results demonstrate that our method achieves an impressive accuracy of 94.34% on this dataset, surpassing the current state-of-the-art (SOTA) by 3.77%. This advancement underscores the potential of our approach to enhance personalized gait control in exoskeleton-assisted therapy. 

**Abstract (ZH)**: 当前的外骨骼控制方法往往难以提供个性化的治疗。标准化的行走模式可能导致患者不适甚至受伤。因此，个性化的步态对于外骨骼机器人的有效性至关重要，因为它直接影响其适应性、舒适性和个体用户的康复效果。为了在外骨骼辅助治疗及相关应用中实现个性化的治疗，准确识别个人步态至关重要，以便实施定制化的步态控制。步态识别的关键挑战在于有效地捕捉由关节协同作用引起的细微步态特征差异，如步频和步长。为了解决这一问题，我们提出了一种新型方法，该方法在空间域使用多尺度全局密集图形卷积网络（GCN）来识别潜在的关节协同模式。此外，我们提出了一种步态非线性周期动态学习模块，以有效地在时间域捕捉步态的周期特性。为了支持我们的个体步态识别任务，我们构建了一个全面的步态数据集，确保其完整性和可靠性。实验结果表明，我们的方法在该数据集上的准确率达到94.34%，超过当前最先进的方法（SOTA）3.77%。这一进展突显了我们方法在增强外骨骼辅助治疗中个性化步态控制方面的潜力。 

---
# Classification of assembly tasks combining multiple primitive actions using Transformers and xLSTMs 

**Title (ZH)**: 使用Transformer和xLSTM结合多种基本动作进行装配任务分类 

**Authors**: Miguel Neves, Pedro Neto  

**Link**: [PDF](https://arxiv.org/pdf/2505.18012)  

**Abstract**: The classification of human-performed assembly tasks is essential in collaborative robotics to ensure safety, anticipate robot actions, and facilitate robot learning. However, achieving reliable classification is challenging when segmenting tasks into smaller primitive actions is unfeasible, requiring us to classify long assembly tasks that encompass multiple primitive actions. In this study, we propose classifying long assembly sequential tasks based on hand landmark coordinates and compare the performance of two well-established classifiers, LSTM and Transformer, as well as a recent model, xLSTM. We used the HRC scenario proposed in the CT benchmark, which includes long assembly tasks that combine actions such as insertions, screw fastenings, and snap fittings. Testing was conducted using sequences gathered from both the human operator who performed the training sequences and three new operators. The testing results of real-padded sequences for the LSTM, Transformer, and xLSTM models was 72.9%, 95.0% and 93.2% for the training operator, and 43.5%, 54.3% and 60.8% for the new operators, respectively. The LSTM model clearly underperformed compared to the other two approaches. As expected, both the Transformer and xLSTM achieved satisfactory results for the operator they were trained on, though the xLSTM model demonstrated better generalization capabilities to new operators. The results clearly show that for this type of classification, the xLSTM model offers a slight edge over Transformers. 

**Abstract (ZH)**: 基于手部关键点坐标的人工装配任务分类在协作机器人中的应用：LSTM、Transformer和xLSTM模型的性能比较 

---
# Is Single-View Mesh Reconstruction Ready for Robotics? 

**Title (ZH)**: 单视图网格重建准备好应对机器人技术挑战了吗？ 

**Authors**: Frederik Nolte, Bernhard Schölkopf, Ingmar Posner  

**Link**: [PDF](https://arxiv.org/pdf/2505.17966)  

**Abstract**: This paper evaluates single-view mesh reconstruction models for creating digital twin environments in robot manipulation. Recent advances in computer vision for 3D reconstruction from single viewpoints present a potential breakthrough for efficiently creating virtual replicas of physical environments for robotics contexts. However, their suitability for physics simulations and robotics applications remains unexplored. We establish benchmarking criteria for 3D reconstruction in robotics contexts, including handling typical inputs, producing collision-free and stable reconstructions, managing occlusions, and meeting computational constraints. Our empirical evaluation using realistic robotics datasets shows that despite success on computer vision benchmarks, existing approaches fail to meet robotics-specific requirements. We quantitively examine limitations of single-view reconstruction for practical robotics implementation, in contrast to prior work that focuses on multi-view approaches. Our findings highlight critical gaps between computer vision advances and robotics needs, guiding future research at this intersection. 

**Abstract (ZH)**: 本文评估了单视角网格重建模型在机器人操作中创建数字孪生环境的应用。随着计算机视觉在单视角三维重建方面的 recent 进展，为机器人学上下文高效创建物理环境的虚拟副本提供了潜在突破。然而，现有方法在物理仿真和机器人应用中的适用性尚未得到探索。我们为机器人学上下文的三维重建建立了基准评估标准，包括处理典型输入、生成无碰撞且稳定的重建、管理遮挡以及满足计算约束。使用现实的机器人数据集进行的经验评估表明，尽管在计算机视觉基准测试中取得成功，现有方法仍无法满足机器人特定的要求。我们定量分析了单视角重建在实际机器人实施中的局限性，与之前主要关注多视角方法的研究不同。我们的发现突显了计算机视觉进展与机器人需求之间的关键差距，为该交叉领域的未来研究指明了方向。 

---
# Object Classification Utilizing Neuromorphic Proprioceptive Signals in Active Exploration: Validated on a Soft Anthropomorphic Hand 

**Title (ZH)**: 利用神经形态本体感受信号在主动探索中进行物体分类：以软类人手为例验证 

**Authors**: Fengyi Wang, Xiangyu Fu, Nitish Thakor, Gordon Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17738)  

**Abstract**: Proprioception, a key sensory modality in haptic perception, plays a vital role in perceiving the 3D structure of objects by providing feedback on the position and movement of body parts. The restoration of proprioceptive sensation is crucial for enabling in-hand manipulation and natural control in the prosthetic hand. Despite its importance, proprioceptive sensation is relatively unexplored in an artificial system. In this work, we introduce a novel platform that integrates a soft anthropomorphic robot hand (QB SoftHand) with flexible proprioceptive sensors and a classifier that utilizes a hybrid spiking neural network with different types of spiking neurons to interpret neuromorphic proprioceptive signals encoded by a biological muscle spindle model. The encoding scheme and the classifier are implemented and tested on the datasets we collected in the active exploration of ten objects from the YCB benchmark. Our results indicate that the classifier achieves more accurate inferences than existing learning approaches, especially in the early stage of the exploration. This system holds the potential for development in the areas of haptic feedback and neural prosthetics. 

**Abstract (ZH)**: 本体感受，作为触觉感知的关键感觉模态，通过提供身体部位位置和运动的反馈，在感知物体的3D结构中发挥着重要作用。恢复本体感受知觉对于在假手内进行物体操控和实现自然控制至关重要。尽管其重要性不言而喻，但本体感受在人工系统中的研究相对较少。本文介绍了一个新的平台，该平台结合了软类人机器人手（QB SoftHand）和柔性的本体感觉传感器，以及一个利用混合神经脉冲网络和不同类型的神经脉冲细胞进行自然界本体感觉信号解码的分类器。该编码方案和分类器已在我们通过主动探索十个YCB基准物体收集的数据集中实现和测试。我们的结果表明，分类器在探索的早期阶段比现有学习方法提供了更准确的推断。该系统在触觉反馈和神经假肢领域具有发展潜力。 

---
# A Bio-mimetic Neuromorphic Model for Heat-evoked Nociceptive Withdrawal Reflex in Upper Limb 

**Title (ZH)**: 生物仿生类神经模型在上肢热诱发痛觉撤离反射中的应用 

**Authors**: Fengyi Wang, J. Rogelio Guadarrama Olvera, Nitish Thako, Gordon Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17724)  

**Abstract**: The nociceptive withdrawal reflex (NWR) is a mechanism to mediate interactions and protect the body from damage in a potentially dangerous environment. To better convey warning signals to users of prosthetic arms or autonomous robots and protect them by triggering a proper NWR, it is useful to use a biological representation of temperature information for fast and effective processing. In this work, we present a neuromorphic spiking network for heat-evoked NWR by mimicking the structure and encoding scheme of the reflex arc. The network is trained with the bio-plausible reward modulated spike timing-dependent plasticity learning algorithm. We evaluated the proposed model and three other methods in recent studies that trigger NWR in an experiment with radiant heat. We found that only the neuromorphic model exhibits the spatial summation (SS) effect and temporal summation (TS) effect similar to humans and can encode the reflex strength matching the intensity of the stimulus in the relative spike latency online. The improved bio-plausibility of this neuromorphic model could improve sensory feedback in neural prostheses. 

**Abstract (ZH)**: 基于神经形态尖峰网络的热诱发痛觉回避反射研究 

---
# Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling 

**Title (ZH)**: Plan-R1: 安全可行的轨迹规划作为语言建模 

**Authors**: Xiaolong Tang, Meina Kan, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17659)  

**Abstract**: Safe and feasible trajectory planning is essential for real-world autonomous driving systems. However, existing learning-based planning methods often rely on expert demonstrations, which not only lack explicit safety awareness but also risk inheriting unsafe behaviors such as speeding from suboptimal human driving data. Inspired by the success of large language models, we propose Plan-R1, a novel two-stage trajectory planning framework that formulates trajectory planning as a sequential prediction task, guided by explicit planning principles such as safety, comfort, and traffic rule compliance. In the first stage, we train an autoregressive trajectory predictor via next motion token prediction on expert data. In the second stage, we design rule-based rewards (e.g., collision avoidance, speed limits) and fine-tune the model using Group Relative Policy Optimization (GRPO), a reinforcement learning strategy, to align its predictions with these planning principles. Experiments on the nuPlan benchmark demonstrate that our Plan-R1 significantly improves planning safety and feasibility, achieving state-of-the-art performance. 

**Abstract (ZH)**: 基于明确规划原则的安全可行路径规划对于实际自主驾驶系统至关重要。然而，现有的基于学习的规划方法往往依赖于专家示范，这不仅缺乏明确的安全意识，还可能继承如超速等不安全行为。受大型语言模型成功的启发，我们提出了一种新颖的两阶段路径规划框架Plan-R1，将其形式化为一个由明确规划原则（如安全、舒适和交通规则遵守）引导的序列预测任务。在第一阶段，我们通过下一个运动标记预测对专家数据进行自回归路径预测器的训练。在第二阶段，我们设计基于规则的奖励（如碰撞避免、限速），并通过Group Relative Policy Optimization (GRPO)强化学习策略对模型进行微调，使其预测与这些规划原则保持一致。在nuPlan基准上的实验表明，我们的Plan-R1在规划安全性和可行性方面取得了显著提升，实现了最先进的性能。 

---
# H2-COMPACT: Human-Humanoid Co-Manipulation via Adaptive Contact Trajectory Policies 

**Title (ZH)**: H2-COMPACT: 人类-类人机器人协同操作的自适应接触轨迹策略 

**Authors**: Geeta Chandra Raju Bethala, Hao Huang, Niraj Pudasaini, Abdullah Mohamed Ali, Shuaihang Yuan, Congcong Wen, Anthony Tzes, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17627)  

**Abstract**: We present a hierarchical policy-learning framework that enables a legged humanoid to cooperatively carry extended loads with a human partner using only haptic cues for intent inference. At the upper tier, a lightweight behavior-cloning network consumes six-axis force/torque streams from dual wrist-mounted sensors and outputs whole-body planar velocity commands that capture the leader's applied forces. At the lower tier, a deep-reinforcement-learning policy, trained under randomized payloads (0-3 kg) and friction conditions in Isaac Gym and validated in MuJoCo and on a real Unitree G1, maps these high-level twists to stable, under-load joint trajectories. By decoupling intent interpretation (force -> velocity) from legged locomotion (velocity -> joints), our method combines intuitive responsiveness to human inputs with robust, load-adaptive walking. We collect training data without motion-capture or markers, only synchronized RGB video and F/T readings, employing SAM2 and WHAM to extract 3D human pose and velocity. In real-world trials, our humanoid achieves cooperative carry-and-move performance (completion time, trajectory deviation, velocity synchrony, and follower-force) on par with a blindfolded human-follower baseline. This work is the first to demonstrate learned haptic guidance fused with full-body legged control for fluid human-humanoid co-manipulation. Code and videos are available on the H2-COMPACT website. 

**Abstract (ZH)**: 基于触觉线索的人类伙伴协同负载搬运的分层策略学习框架 

---
# CU-Multi: A Dataset for Multi-Robot Data Association 

**Title (ZH)**: CU-Multi: 多机器人数据关联数据集 

**Authors**: Doncey Albin, Miles Mena, Annika Thomas, Harel Biggie, Xuefei Sun, Dusty Woods, Steve McGuire, Christoffer Heckman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17576)  

**Abstract**: Multi-robot systems (MRSs) are valuable for tasks such as search and rescue due to their ability to coordinate over shared observations. A central challenge in these systems is aligning independently collected perception data across space and time, i.e., multi-robot data association. While recent advances in collaborative SLAM (C-SLAM), map merging, and inter-robot loop closure detection have significantly progressed the field, evaluation strategies still predominantly rely on splitting a single trajectory from single-robot SLAM datasets into multiple segments to simulate multiple robots. Without careful consideration to how a single trajectory is split, this approach will fail to capture realistic pose-dependent variation in observations of a scene inherent to multi-robot systems. To address this gap, we present CU-Multi, a multi-robot dataset collected over multiple days at two locations on the University of Colorado Boulder campus. Using a single robotic platform, we generate four synchronized runs with aligned start times and deliberate percentages of trajectory overlap. CU-Multi includes RGB-D, GPS with accurate geospatial heading, and semantically annotated LiDAR data. By introducing controlled variations in trajectory overlap and dense lidar annotations, CU-Multi offers a compelling alternative for evaluating methods in multi-robot data association. Instructions on accessing the dataset, support code, and the latest updates are publicly available at this https URL 

**Abstract (ZH)**: 多机器人系统（MRSs）在搜索与救援等任务中因其在共享观察信息上的协同能力而具有价值。这些系统中的核心挑战在于对独立收集的感知数据进行空间和时间上的对齐，即多机器人数据关联。虽然在协作SLAM（C-SLAM）、地图合并以及机器人间循环闭合检测方面取得了显著进展，但评估策略仍未摆脱将单个机器人SLAM数据集中的单个轨迹拆分成多个段落来模拟多个机器人的方法。若不仔细考虑轨迹拆分的方式，这种方法将无法捕捉多机器人系统中场景观察中固有的、依赖于姿态的真实变化。为解决这一问题，我们介绍了CU-Multi，这是一个在科罗拉多大学博尔德分校两个地点多天收集的多机器人数据集。使用单一机器人平台，我们生成了四个同步运行，具有对齐的起始时间和故意的轨迹重叠比例。CU-Multi 包含RGB-D、GPS（带精确地理方向性）和语义标注的LiDAR数据。通过引入轨迹重叠的可控变化和密集的LiDAR注释，CU-Multi 为评估多机器人数据关联方法提供了极具吸引力的替代方案。有关数据集的访问说明、支持代码以及最新更新的详细信息，请访问 <https://this.is.public>。 

---
# DTRT: Enhancing Human Intent Estimation and Role Allocation for Physical Human-Robot Collaboration 

**Title (ZH)**: DTRT: 提升物理人机协作中的人类意图估计与角色分配 

**Authors**: Haotian Liu, Yuchuang Tong, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17490)  

**Abstract**: In physical Human-Robot Collaboration (pHRC), accurate human intent estimation and rational human-robot role allocation are crucial for safe and efficient assistance. Existing methods that rely on short-term motion data for intention estimation lack multi-step prediction capabilities, hindering their ability to sense intent changes and adjust human-robot assignments autonomously, resulting in potential discrepancies. To address these issues, we propose a Dual Transformer-based Robot Trajectron (DTRT) featuring a hierarchical architecture, which harnesses human-guided motion and force data to rapidly capture human intent changes, enabling accurate trajectory predictions and dynamic robot behavior adjustments for effective collaboration. Specifically, human intent estimation in DTRT uses two Transformer-based Conditional Variational Autoencoders (CVAEs), incorporating robot motion data in obstacle-free case with human-guided trajectory and force for obstacle avoidance. Additionally, Differential Cooperative Game Theory (DCGT) is employed to synthesize predictions based on human-applied forces, ensuring robot behavior align with human intention. Compared to state-of-the-art (SOTA) methods, DTRT incorporates human dynamics into long-term prediction, providing an accurate understanding of intention and enabling rational role allocation, achieving robot autonomy and maneuverability. Experiments demonstrate DTRT's accurate intent estimation and superior collaboration performance. 

**Abstract (ZH)**: 基于物理的人机协作中的Dual Transformer-based Robot Trajectron (DTRT)：融合人类引导的运动和力数据实现鲁棒的人意估计和动态角色分配 

---
# HEPP: Hyper-efficient Perception and Planning for High-speed Obstacle Avoidance of UAVs 

**Title (ZH)**: HEPP: 超高效感知与规划以实现无人机高速障碍 avoidance 

**Authors**: Minghao Lu, Xiyu Fan, Bowen Xu, Zexuan Yan, Rui Peng, Han Chen, Lixian Zhang, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17438)  

**Abstract**: High-speed obstacle avoidance of uncrewed aerial vehicles (UAVs) in cluttered environments is a significant challenge. Existing UAV planning and obstacle avoidance systems can only fly at moderate speeds or at high speeds over empty or sparse fields. In this article, we propose a hyper-efficient perception and planning system for the high-speed obstacle avoidance of UAVs. The system mainly consists of three modules: 1) A novel incremental robocentric mapping method with distance and gradient information, which takes 89.5% less time compared to existing methods. 2) A novel obstacle-aware topological path search method that generates multiple distinct paths. 3) An adaptive gradient-based high-speed trajectory generation method with a novel time pre-allocation algorithm. With these innovations, the system has an excellent real-time performance with only milliseconds latency in each iteration, taking 79.24% less time than existing methods at high speeds (15 m/s in cluttered environments), allowing UAVs to fly swiftly and avoid obstacles in cluttered environments. The planned trajectory of the UAV is close to the global optimum in both temporal and spatial domains. Finally, extensive validations in both simulation and real-world experiments demonstrate the effectiveness of our proposed system for high-speed navigation in cluttered environments. 

**Abstract (ZH)**: 无crewed航空器在复杂环境下的高速障碍避让：一种高效感知与规划系统 

---
# Dynamic Manipulation of Deformable Objects in 3D: Simulation, Benchmark and Learning Strategy 

**Title (ZH)**: 三维可变形对象的动态操纵：仿真、基准测试与学习策略 

**Authors**: Guanzhou Lan, Yuqi Yang, Anup Teejo Mathew, Feiping Nie, Rong Wang, Xuelong Li, Federico Renda, Bin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17434)  

**Abstract**: Goal-conditioned dynamic manipulation is inherently challenging due to complex system dynamics and stringent task constraints, particularly in deformable object scenarios characterized by high degrees of freedom and underactuation. Prior methods often simplify the problem to low-speed or 2D settings, limiting their applicability to real-world 3D tasks. In this work, we explore 3D goal-conditioned rope manipulation as a representative challenge. To mitigate data scarcity, we introduce a novel simulation framework and benchmark grounded in reduced-order dynamics, which enables compact state representation and facilitates efficient policy learning. Building on this, we propose Dynamics Informed Diffusion Policy (DIDP), a framework that integrates imitation pretraining with physics-informed test-time adaptation. First, we design a diffusion policy that learns inverse dynamics within the reduced-order space, enabling imitation learning to move beyond naïve data fitting and capture the underlying physical structure. Second, we propose a physics-informed test-time adaptation scheme that imposes kinematic boundary conditions and structured dynamics priors on the diffusion process, ensuring consistency and reliability in manipulation execution. Extensive experiments validate the proposed approach, demonstrating strong performance in terms of accuracy and robustness in the learned policy. 

**Abstract (ZH)**: 基于动力学指导的三维绳索操作是一项由于复杂系统动力学和严格的任务约束而固有的挑战性任务，尤其是在自由度高且欠驱动的变形物体场景中。先前的方法往往将问题简化为低速或二维设置，限制了它们在真实世界三维任务中的适用性。在本工作中，我们探索三维目标条件下的绳索操作作为具有代表性的挑战。为缓解数据稀缺问题，我们引入了一种基于降阶动力学的新型仿真框架和基准，这能够实现紧凑的状态表示并促进高效的策略学习。在此基础上，我们提出了动力学指导扩散策略（DIDP）框架，该框架结合了模仿预训练与基于物理的测试时适应。首先，我们设计了一种扩散策略，在降阶空间中学习逆动力学，从而使模仿学习能够超越简单的数据拟合并捕获潜在的物理结构。其次，我们提出了一种基于物理的测试时适应方案，在扩散过程中施加动力学边界条件和结构化动力学先验，确保操作执行的一致性和可靠性。广泛的实验验证了所提出的方案，在学习策略的准确性和鲁棒性方面表现出强劲性能。 

---
# Bootstrapping Imitation Learning for Long-horizon Manipulation via Hierarchical Data Collection Space 

**Title (ZH)**: 基于分层数据收集空间的长时域操作imitation learning自举方法 

**Authors**: Jinrong Yang, Kexun Chen, Zhuoling Li, Shengkai Wu, Yong Zhao, Liangliang Ren, Wenqiu Luo, Chaohui Shang, Meiyu Zhi, Linfeng Gao, Mingshan Sun, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17389)  

**Abstract**: Imitation learning (IL) with human demonstrations is a promising method for robotic manipulation tasks. While minimal demonstrations enable robotic action execution, achieving high success rates and generalization requires high cost, e.g., continuously adding data or incrementally conducting human-in-loop processes with complex hardware/software systems. In this paper, we rethink the state/action space of the data collection pipeline as well as the underlying factors responsible for the prediction of non-robust actions. To this end, we introduce a Hierarchical Data Collection Space (HD-Space) for robotic imitation learning, a simple data collection scheme, endowing the model to train with proactive and high-quality data. Specifically, We segment the fine manipulation task into multiple key atomic tasks from a high-level perspective and design atomic state/action spaces for human demonstrations, aiming to generate robust IL data. We conduct empirical evaluations across two simulated and five real-world long-horizon manipulation tasks and demonstrate that IL policy training with HD-Space-based data can achieve significantly enhanced policy performance. HD-Space allows the use of a small amount of demonstration data to train a more powerful policy, particularly for long-horizon manipulation tasks. We aim for HD-Space to offer insights into optimizing data quality and guiding data scaling. project page: this https URL. 

**Abstract (ZH)**: 基于人类演示的层次化数据采集空间在机器人模仿学习中的应用：促进高效和鲁棒的数据收集与模型训练 

---
# UAV Control with Vision-based Hand Gesture Recognition over Edge-Computing 

**Title (ZH)**: 基于边缘计算的视觉手势识别无人机控制 

**Authors**: Sousannah Abdalla, Sabur Baidya  

**Link**: [PDF](https://arxiv.org/pdf/2505.17303)  

**Abstract**: Gesture recognition presents a promising avenue for interfacing with unmanned aerial vehicles (UAVs) due to its intuitive nature and potential for precise interaction. This research conducts a comprehensive comparative analysis of vision-based hand gesture detection methodologies tailored for UAV Control. The existing gesture recognition approaches involving cropping, zooming, and color-based segmentation, do not work well for this kind of applications in dynamic conditions and suffer in performance with increasing distance and environmental noises. We propose to use a novel approach leveraging hand landmarks drawing and classification for gesture recognition based UAV control. With experimental results we show that our proposed method outperforms the other existing methods in terms of accuracy, noise resilience, and efficacy across varying distances, thus providing robust control decisions. However, implementing the deep learning based compute intensive gesture recognition algorithms on the UAV's onboard computer is significantly challenging in terms of performance. Hence, we propose to use a edge-computing based framework to offload the heavier computing tasks, thus achieving closed-loop real-time performance. With implementation over AirSim simulator as well as over a real-world UAV, we showcase the advantage of our end-to-end gesture recognition based UAV control system. 

**Abstract (ZH)**: 基于手势识别的无人机控制：一种利用手部关键点检测与分类的方法 

---
# ScanBot: Towards Intelligent Surface Scanning in Embodied Robotic Systems 

**Title (ZH)**: ScanBot: 让虚拟智能体表面扫描更智能的研究 

**Authors**: Zhiling Chen, Yang Zhang, Fardin Jalil Piran, Qianyu Zhou, Jiong Tang, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2505.17295)  

**Abstract**: We introduce ScanBot, a novel dataset designed for instruction-conditioned, high-precision surface scanning in robotic systems. In contrast to existing robot learning datasets that focus on coarse tasks such as grasping, navigation, or dialogue, ScanBot targets the high-precision demands of industrial laser scanning, where sub-millimeter path continuity and parameter stability are critical. The dataset covers laser scanning trajectories executed by a robot across 12 diverse objects and 6 task types, including full-surface scans, geometry-focused regions, spatially referenced parts, functionally relevant structures, defect inspection, and comparative analysis. Each scan is guided by natural language instructions and paired with synchronized RGB, depth, and laser profiles, as well as robot pose and joint states. Despite recent progress, existing vision-language action (VLA) models still fail to generate stable scanning trajectories under fine-grained instructions and real-world precision demands. To investigate this limitation, we benchmark a range of multimodal large language models (MLLMs) across the full perception-planning-execution loop, revealing persistent challenges in instruction-following under realistic constraints. 

**Abstract (ZH)**: 我们介绍ScanBot，一个用于指令条件下的高精度表面扫描的新颖数据集。与现有的专注于抓取、导航或对话等粗粒度任务的机器人学习数据集不同，ScanBot 针对工业激光扫描的高精度需求，其中毫米级路径连续性和参数稳定性至关重要。该数据集涵盖了机器人在12种不同对象和6种任务类型上执行的激光扫描轨迹，包括完整表面扫描、几何区域扫描、空间参考部件、功能相关结构、缺陷检测和对比分析。每项扫描由自然语言指令引导，并配有同步的RGB、深度和激光配置文件，以及机器人姿态和关节状态。尽管最近取得了进展，现有的视觉-语言行动（VLA）模型仍无法在细粒度指令和现实世界的高精度需求下生成稳定的扫描轨迹。为了探讨这一限制，我们在感知-规划-执行的完整流程中对一系列多模态大型语言模型（MLLMs）进行了基准测试，揭示了在实际约束条件下指令遵循的持续挑战。 

---
# Construction of an Impedance Control Test Bench 

**Title (ZH)**: 阻抗控制试验台的构建 

**Authors**: Elisa G. Vergamini, Leonardo F. Dos Santos, Cícero Zanette, Yecid Moreno, Felix M. Escalante, Thiago Boaventura  

**Link**: [PDF](https://arxiv.org/pdf/2505.17278)  

**Abstract**: Controlling the physical interaction with the environment or objects, as humans do, is a shared requirement across different types of robots. To effectively control this interaction, it is necessary to control the power delivered to the load, that is, the interaction force and the interaction velocity. However, it is not possible to control these two quantities independently at the same time. An alternative is to control the relation between them, with Impedance and Admittance control, for example. The Impedance Control 2 Dimensions (IC2D) bench is a test bench designed to allow the performance analysis of different actuators and controllers at the joint level. Therefore, it was designed to be as versatile as possible, to allow the combination of linear and/or rotational motions, to use electric and/or hydraulic actuators, with loads known and defined by the user. The bench adheres to a set of requirements defined by the demands of the research group, to be a reliable, backlash-free mechatronic system to validate system dynamics models and controller designs, as well as a valuable experimental setup for benchmarking electric and hydraulic actuators. This article presents the mechanical, electrical, and hydraulic configurations used to ensure the robustness and reliability of the test bench. Benches similar to this one are commonly found in robotics laboratories around the world. However, the IC2D stands out for its versatility and reliability, as well as for supporting hydraulic and electric actuators. 

**Abstract (ZH)**: 基于阻抗和 admittance 控制的双向交互力控制平台 IC2D 的机械、电气与液压配置研究 

---
# ConvoyNext: A Scalable Testbed Platform for Cooperative Autonomous Vehicle Systems 

**Title (ZH)**: ConvoyNext：一个可扩展的协作自动驾驶车辆系统测试床平台 

**Authors**: Hossein Maghsoumi, Yaser Fallah  

**Link**: [PDF](https://arxiv.org/pdf/2505.17275)  

**Abstract**: The advancement of cooperative autonomous vehicle systems depends heavily on effective coordination between multiple agents, aiming to enhance traffic efficiency, fuel economy, and road safety. Despite these potential benefits, real-world testing of such systems remains a major challenge and is essential for validating control strategies, trajectory modeling methods, and communication robustness across diverse environments. To address this need, we introduce ConvoyNext, a scalable, modular, and extensible platform tailored for the real-world evaluation of cooperative driving behaviors. We demonstrate the capabilities of ConvoyNext through a series of experiments involving convoys of autonomous vehicles navigating complex trajectories. These tests highlight the platform's robustness across heterogeneous vehicle configurations and its effectiveness in assessing convoy behavior under varying communication conditions, including intentional packet loss. Our results validate ConvoyNext as a comprehensive, open-access testbed for advancing research in cooperative autonomous vehicle systems. 

**Abstract (ZH)**: 合作自动驾驶车辆系统的发展高度依赖于多智能体之间的有效协调，旨在提高交通效率、燃油经济性和道路安全。尽管这些潜在好处显著，但在实际环境中的测试仍然是一个重大挑战，对于验证控制策略、轨迹建模方法和通信鲁棒性至关重要。为应对这一需求，我们引入了ConvoyNext平台，这是一个可扩展、模块化和可扩展的平台，专为合作驾驶行为的实际评估而设计。通过涉及自动驾驶车辆编队导航复杂轨迹的一系列实验，我们展示了ConvoyNext的功能。这些测试强调了该平台在异构车辆配置下的鲁棒性，并评估了在不同通信条件下编队行为的有效性，包括有意的包丢失。我们的结果验证了ConvoyNext作为合作自动驾驶车辆系统研究综合性开放式测试平台的有效性。 

---
# LiloDriver: A Lifelong Learning Framework for Closed-loop Motion Planning in Long-tail Autonomous Driving Scenarios 

**Title (ZH)**: LiloDriver：长尾自动驾驶场景中闭环运动规划的终身学习框架 

**Authors**: Huaiyuan Yao, Pengfei Li, Bu Jin, Yupeng Zheng, An Liu, Lisen Mu, Qing Su, Qian Zhang, Yilun Chen, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17209)  

**Abstract**: Recent advances in autonomous driving research towards motion planners that are robust, safe, and adaptive. However, existing rule-based and data-driven planners lack adaptability to long-tail scenarios, while knowledge-driven methods offer strong reasoning but face challenges in representation, control, and real-world evaluation. To address these challenges, we present LiloDriver, a lifelong learning framework for closed-loop motion planning in long-tail autonomous driving scenarios. By integrating large language models (LLMs) with a memory-augmented planner generation system, LiloDriver continuously adapts to new scenarios without retraining. It features a four-stage architecture including perception, scene encoding, memory-based strategy refinement, and LLM-guided reasoning. Evaluated on the nuPlan benchmark, LiloDriver achieves superior performance in both common and rare driving scenarios, outperforming static rule-based and learning-based planners. Our results highlight the effectiveness of combining structured memory and LLM reasoning to enable scalable, human-like motion planning in real-world autonomous driving. Our code is available at this https URL. 

**Abstract (ZH)**: 近年来自主驾驶研究在鲁棒、安全和自适应运动规划方面的进展。然而，现有的基于规则和基于数据的规划者缺乏对长尾场景的适应性，而基于知识的方法虽然提供了强大的推理能力，但在表示、控制和现实世界评估方面面临挑战。为应对这些挑战，我们提出了一种名为LiloDriver的终身学习框架，用于长尾自主驾驶场景下的闭环运动规划。通过将大型语言模型（LLMs）与记忆增强的规划生成系统集成，LiloDriver无需重新训练即可不断适应新的场景。LiloDriver具有四阶段架构，包括感知、场景编码、基于记忆的策略细化和LLM引导的推理。在nuPlan基准测试上，LiloDriver在常见和罕见驾驶场景中均表现出色，超越了静态基于规则和基于学习的规划者。我们的结果强调了结合结构化记忆和LLM推理的有效性，以实现可扩展的人类级运动规划。有关代码可访问此链接：this https URL。 

---
# Rotational Multi-material 3D Printing of Soft Robotic Matter with Asymmetrical Embedded Pneumatics 

**Title (ZH)**: 具有异构嵌入气动结构的旋转多材料3D打印软机器人材料 

**Authors**: Jackson K. Wilt, Natalie M. Larson, Jennifer A. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2505.18095)  

**Abstract**: The rapid design and fabrication of soft robotic matter is of growing interest for shape morphing, actuation, and wearable devices. Here, we report a facile fabrication method for creating soft robotic materials with embedded pneumatics that exhibit programmable shape morphing behavior. Using rotational multi-material 3D printing, asymmetrical core-shell filaments composed of elastomeric shells and fugitive cores are patterned in 1D and 2D motifs. By precisely controlling the nozzle design, rotation rate, and print path, one can control the local orientation, shape, and cross-sectional area of the patterned fugitive core along each printed filament. Once the elastomeric matrix is cured, the fugitive cores are removed, leaving behind embedded conduits that facilitate pneumatic actuation. Using a connected Fermat spirals pathing approach, one can automatically generate desired print paths required for more complex soft robots, such as hand-inspired grippers. Our integrated design and printing approach enables one to rapidly build soft robotic matter that exhibits myriad shape morphing transitions on demand. 

**Abstract (ZH)**: 软机器人材料的快速设计与制造：嵌入式气动编程形状变形行为的研究 

---
# What Do You Need for Diverse Trajectory Stitching in Diffusion Planning? 

**Title (ZH)**: 你需要什么来进行多样化的轨迹缝合以规划扩散？ 

**Authors**: Quentin Clark, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2505.18083)  

**Abstract**: In planning, stitching is an ability of algorithms to piece together sub-trajectories of data they are trained on to generate new and diverse behaviours. While stitching is historically a strength of offline reinforcement learning, recent generative behavioural cloning (BC) methods have also shown proficiency at stitching. However, the main factors behind this are poorly understood, hindering the development of new algorithms that can reliably stitch. Focusing on diffusion planners trained via BC, we find two properties are needed to compose: \emph{positional equivariance} and \emph{local receptiveness}. We use these two properties to explain architecture, data, and inference choices in existing generative BC methods based on diffusion planning, including replanning frequency, data augmentation, and data scaling. Experimental comparisions show that (1) while locality is more important than positional equivariance in creating a diffusion planner capable of composition, both are crucial (2) enabling these properties through relatively simple architecture choices can be competitive with more computationally expensive methods such as replanning or scaling data, and (3) simple inpainting-based guidance can guide architecturally compositional models to enable generalization in goal-conditioned settings. 

**Abstract (ZH)**: 规划中的缝合能力：基于扩散规划的生成行为克隆方法的综合研究 

---
# Linear Mixture Distributionally Robust Markov Decision Processes 

**Title (ZH)**: 线性混合分布鲁棒马尔可夫决策过程 

**Authors**: Zhishuai Liu, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18044)  

**Abstract**: Many real-world decision-making problems face the off-dynamics challenge: the agent learns a policy in a source domain and deploys it in a target domain with different state transitions. The distributionally robust Markov decision process (DRMDP) addresses this challenge by finding a robust policy that performs well under the worst-case environment within a pre-specified uncertainty set of transition dynamics. Its effectiveness heavily hinges on the proper design of these uncertainty sets, based on prior knowledge of the dynamics. In this work, we propose a novel linear mixture DRMDP framework, where the nominal dynamics is assumed to be a linear mixture model. In contrast with existing uncertainty sets directly defined as a ball centered around the nominal kernel, linear mixture DRMDPs define the uncertainty sets based on a ball around the mixture weighting parameter. We show that this new framework provides a more refined representation of uncertainties compared to conventional models based on $(s,a)$-rectangularity and $d$-rectangularity, when prior knowledge about the mixture model is present. We propose a meta algorithm for robust policy learning in linear mixture DRMDPs with general $f$-divergence defined uncertainty sets, and analyze its sample complexities under three divergence metrics instantiations: total variation, Kullback-Leibler, and $\chi^2$ divergences. These results establish the statistical learnability of linear mixture DRMDPs, laying the theoretical foundation for future research on this new setting. 

**Abstract (ZH)**: 一种新的线性混合分布鲁棒马尔可夫决策过程框架及其统计可学习性分析 

---
# Knot So Simple: A Minimalistic Environment for Spatial Reasoning 

**Title (ZH)**: 结不那么简单：一个简洁的空间推理环境 

**Authors**: Zizhao Chen, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18028)  

**Abstract**: We propose KnotGym, an interactive environment for complex, spatial reasoning and manipulation. KnotGym includes goal-oriented rope manipulation tasks with varying levels of complexity, all requiring acting from pure image observations. Tasks are defined along a clear and quantifiable axis of complexity based on the number of knot crossings, creating a natural generalization test. KnotGym has a simple observation space, allowing for scalable development, yet it highlights core challenges in integrating acute perception, spatial reasoning, and grounded manipulation. We evaluate methods of different classes, including model-based RL, model-predictive control, and chain-of-thought reasoning, and illustrate the challenges KnotGym presents. KnotGym is available at this https URL. 

**Abstract (ZH)**: 我们提出KnotGym，一个用于复杂空间推理和操作的互动环境。 

---
# AI-Driven Robotics for Free-Space Optics 

**Title (ZH)**: AI驱动的自由空间光学机器人技术 

**Authors**: Shiekh Zia Uddin, Sachin Vaidya, Shrish Choudhary, Zhuo Chen, Raafat K. Salib, Luke Huang, Dirk R. Englund, Marin Soljačić  

**Link**: [PDF](https://arxiv.org/pdf/2505.17985)  

**Abstract**: Tabletop optical experiments are foundational to research in many areas of science, including photonics, quantum optics, materials science, metrology, and biomedical imaging. However these experiments remain fundamentally reliant on manual design, assembly, and alignment, limiting throughput and reproducibility. Optics currently lacks generalizable robotic systems capable of operating across a diverse range of setups in realistic laboratory environments. Here we present OptoMate, an autonomous platform that integrates generative AI, computer vision, and precision robotics to enable automation of free-space optics experiments. Our platform interprets user-defined goals to generate valid optical setups using a fine-tuned large language model (LLM), assembles the setup via robotic pick-and-place with sub-millimeter accuracy, and performs fine alignment using a robot-deployable tool. The system then executes a range of automated measurements, including laser beam characterization, polarization mapping, and spectroscopy tasks. This work demonstrates the first flexible, AI-driven automation platform for optics, offering a path toward remote operation, cloud labs, and high-throughput discovery in the optical sciences. 

**Abstract (ZH)**: 桌面光学实验是许多科学领域，包括光子学、量子光学、材料科学、计量学和生物医学成像研究的基础。然而，这些实验仍然主要依赖于手工设计、组装和对准，限制了 throughput 和可再现性。光学目前缺乏能够在现实实验室环境中跨多种设置进行通用操作的机器人系统。本文介绍了一种自主平台 OptoMate，该平台整合生成式 AI、计算机视觉和精密机器人技术，以实现自由空间光学实验的自动化。该平台根据用户定义的目标，使用微调的大语言模型生成有效的光学设置，通过机器人拾放组装设置，并使用机器人部署工具进行精细对准。然后，该系统执行一系列自动化测量任务，包括激光束表征、偏振映射和光谱学任务。本研究展示了第一个灵活的、由 AI 驱动的光学自动化平台，为远程操作、云实验室和光学科学中的高通量发现提供了途径。 

---
# Feasible Action Space Reduction for Quantifying Causal Responsibility in Continuous Spatial Interactions 

**Title (ZH)**: 连续空间交互中因果责任量化可行动作空间减维 

**Authors**: Ashwin George, Luciano Cavalcante Siebert, David A. Abbink, Arkady Zgonnikov  

**Link**: [PDF](https://arxiv.org/pdf/2505.17739)  

**Abstract**: Understanding the causal influence of one agent on another agent is crucial for safely deploying artificially intelligent systems such as automated vehicles and mobile robots into human-inhabited environments. Existing models of causal responsibility deal with simplified abstractions of scenarios with discrete actions, thus, limiting real-world use when understanding responsibility in spatial interactions. Based on the assumption that spatially interacting agents are embedded in a scene and must follow an action at each instant, Feasible Action-Space Reduction (FeAR) was proposed as a metric for causal responsibility in a grid-world setting with discrete actions. Since real-world interactions involve continuous action spaces, this paper proposes a formulation of the FeAR metric for measuring causal responsibility in space-continuous interactions. We illustrate the utility of the metric in prototypical space-sharing conflicts, and showcase its applications for analysing backward-looking responsibility and in estimating forward-looking responsibility to guide agent decision making. Our results highlight the potential of the FeAR metric for designing and engineering artificial agents, as well as for assessing the responsibility of agents around humans. 

**Abstract (ZH)**: 理解一个智能体对另一个智能体的因果影响对于安全部署自动车辆和移动机器人等人工智能系统进入人类居住环境至关重要。现有因果责任模型处理的是具有离散动作的简化场景抽象，这限制了在理解空间交互责任时的实际应用。基于空间交互智能体嵌入场景并在每一时刻必须采取行动的假设，提出了可行动作空间缩减（FeAS）作为网格世界中离散动作环境下的因果责任度量标准。由于实际交互涉及连续动作空间，本文提出了一个在连续空间交互中衡量因果责任的FeAS度量标准的建模方法。我们通过典型的空间共享冲突示例展示了该度量标准的应用价值，并展示了其在分析后向责任和估计前瞻性责任以指导智能体决策中的应用。我们的研究结果突显了FeAS度量标准在设计和工程化人工智能系统以及评估人类周围智能体的责任方面的潜在价值。 

---
# MinkUNeXt-SI: Improving point cloud-based place recognition including spherical coordinates and LiDAR intensity 

**Title (ZH)**: MinkUNeXt-SI: 提高基于点云的地点识别性能，包括球坐标和激光雷达强度 

**Authors**: Judith Vilella-Cantos, Juan José Cabrera, Luis Payá, Mónica Ballesta, David Valiente  

**Link**: [PDF](https://arxiv.org/pdf/2505.17591)  

**Abstract**: In autonomous navigation systems, the solution of the place recognition problem is crucial for their safe functioning. But this is not a trivial solution, since it must be accurate regardless of any changes in the scene, such as seasonal changes and different weather conditions, and it must be generalizable to other environments. This paper presents our method, MinkUNeXt-SI, which, starting from a LiDAR point cloud, preprocesses the input data to obtain its spherical coordinates and intensity values normalized within a range of 0 to 1 for each point, and it produces a robust place recognition descriptor. To that end, a deep learning approach that combines Minkowski convolutions and a U-net architecture with skip connections is used. The results of MinkUNeXt-SI demonstrate that this method reaches and surpasses state-of-the-art performance while it also generalizes satisfactorily to other datasets. Additionally, we showcase the capture of a custom dataset and its use in evaluating our solution, which also achieves outstanding results. Both the code of our solution and the runs of our dataset are publicly available for reproducibility purposes. 

**Abstract (ZH)**: 在自主导航系统中，位置识别问题的解决方案对于其安全运行至关重要。但这不是一个简单的解决方案，因为它必须在任何场景变化（如季节变化和不同天气条件）下保持准确，并且能够泛化到其他环境中。本文介绍了我们的方法MinkUNeXt-SI，该方法从激光雷达点云出发，预处理输入数据以获取每个点的球坐标及其在0到1范围内的强度值，并生成一个稳健的位置识别描述符。为此，我们使用了结合Minkowski卷积和具有跳连接的U-net架构的深度学习方法。MinkUNeXt-SI的结果表明，该方法不仅达到了最先进的性能，而且还能够在其他数据集中泛化得当。此外，我们展示了自定义数据集的采集及其在评估我们解决方案中的应用，该解决方案也取得了优异的结果。我们的解决方案代码和数据集运行结果均已公开，以便于可再现研究。 

---
# Distance Estimation in Outdoor Driving Environments Using Phase-only Correlation Method with Event Cameras 

**Title (ZH)**: 使用事件摄像头的相位-only相关性方法在户外驾驶环境中进行距离估计 

**Authors**: Masataka Kobayashi, Shintaro Shiba, Quan Kong, Norimasa Kobori, Tsukasa Shimizu, Shan Lu, Takaya Yamazato  

**Link**: [PDF](https://arxiv.org/pdf/2505.17582)  

**Abstract**: With the growing adoption of autonomous driving, the advancement of sensor technology is crucial for ensuring safety and reliable operation. Sensor fusion techniques that combine multiple sensors such as LiDAR, radar, and cameras have proven effective, but the integration of multiple devices increases both hardware complexity and cost. Therefore, developing a single sensor capable of performing multiple roles is highly desirable for cost-efficient and scalable autonomous driving systems.
Event cameras have emerged as a promising solution due to their unique characteristics, including high dynamic range, low latency, and high temporal resolution. These features enable them to perform well in challenging lighting conditions, such as low-light or backlit environments. Moreover, their ability to detect fine-grained motion events makes them suitable for applications like pedestrian detection and vehicle-to-infrastructure communication via visible light.
In this study, we present a method for distance estimation using a monocular event camera and a roadside LED bar. By applying a phase-only correlation technique to the event data, we achieve sub-pixel precision in detecting the spatial shift between two light sources. This enables accurate triangulation-based distance estimation without requiring stereo vision. Field experiments conducted in outdoor driving scenarios demonstrated that the proposed approach achieves over 90% success rate with less than 0.5-meter error for distances ranging from 20 to 60 meters.
Future work includes extending this method to full position estimation by leveraging infrastructure such as smart poles equipped with LEDs, enabling event-camera-based vehicles to determine their own position in real time. This advancement could significantly enhance navigation accuracy, route optimization, and integration into intelligent transportation systems. 

**Abstract (ZH)**: 随着自动驾驶的广泛采用，传感器技术的进步对于确保安全和可靠运行至关重要。结合多种传感器（如LiDAR、雷达和摄像头）的传感器融合技术已被证明是有效的，但多设备的集成增加了硬件复杂性和成本。因此，开发能够执行多重角色的单个传感器对于成本效益和可扩展的自动驾驶系统来说是非常 desirable 的。

事件相机由于其独特的特性，如高动态范围、低延迟和高时间分辨率，而被视为一种有前途的解决方案。这些特性使它们能够在低光照或背光等具有挑战性的光照条件下表现出色。此外，它们检测精细运动事件的能力使其适用于行人检测和基于可见光的车路通信等应用。

在本研究中，我们提出了一种使用单目事件相机和路边LED条形灯进行距离估算的方法。通过将相位唯一相关技术应用于事件数据，我们实现了在检测两个光源之间空间偏移方面的亚像素精度。这使我们能够实现基于三角测量的距离准确估算，而无需立体视觉技术。在户外驾驶场景下的实地实验表明，所提出的方法在20至60米的距离范围内误差小于0.5米的条件下，成功率超过90%。

未来的工作包括利用配备LED的智能路灯等基础设施将此方法扩展到全位置估计，使事件相机为基础的车辆能够实时确定自己的位置。这一进展可以显著提高导航准确性、路线优化，并实现与智能交通系统的集成。 

---
# Navigating Polytopes with Safety: A Control Barrier Function Approach 

**Title (ZH)**: 带安全性的多面体导航：基于控制障碍函数的方法 

**Authors**: Tamas G. Molnar  

**Link**: [PDF](https://arxiv.org/pdf/2505.17270)  

**Abstract**: Collision-free motion is a fundamental requirement for many autonomous systems. This paper develops a safety-critical control approach for the collision-free navigation of polytope-shaped agents in polytope-shaped environments. A systematic method is proposed to generate control barrier function candidates in closed form that lead to controllers with formal safety guarantees. The proposed approach is demonstrated through simulation, with obstacle avoidance examples in 2D and 3D, including dynamically changing environments. 

**Abstract (ZH)**: 多面体形代理在多面体形环境中的碰撞-free导航的安全关键控制方法 

---
# Lightweight Multispectral Crop-Weed Segmentation for Precision Agriculture 

**Title (ZH)**: 轻量级多光谱作物-杂草分割技术及其在精准农业中的应用 

**Authors**: Zeynep Galymzhankyzy, Eric Martinson  

**Link**: [PDF](https://arxiv.org/pdf/2505.07444)  

**Abstract**: Efficient crop-weed segmentation is critical for site-specific weed control in precision agriculture. Conventional CNN-based methods struggle to generalize and rely on RGB imagery, limiting performance under complex field conditions. To address these challenges, we propose a lightweight transformer-CNN hybrid. It processes RGB, Near-Infrared (NIR), and Red-Edge (RE) bands using specialized encoders and dynamic modality integration. Evaluated on the WeedsGalore dataset, the model achieves a segmentation accuracy (mean IoU) of 78.88%, outperforming RGB-only models by 15.8 percentage points. With only 8.7 million parameters, the model offers high accuracy, computational efficiency, and potential for real-time deployment on Unmanned Aerial Vehicles (UAVs) and edge devices, advancing precision weed management. 

**Abstract (ZH)**: 高效的作物-杂草分割对于精准农业中的定点杂草控制至关重要。传统的基于CNN的方法难以泛化且依赖RGB图像，这在复杂田间条件下限制了其性能。为解决这些挑战，我们提出了一种轻量级的Transformer-CNN混合模型。该模型使用专门的编码器处理RGB、近红外（NIR）和红边（RE）波段，并实现动态模态集成。在WeedsGalore数据集上，该模型实现了78.88%的分割准确率（mean IoU），比仅使用RGB的数据模型高出15.8个百分点。该模型仅有870万参数，提供高精度、计算效率，并具有在无人驾驶飞机（UAV）和边缘设备上实时部署的潜力，推动精准杂草管理的发展。 

---
