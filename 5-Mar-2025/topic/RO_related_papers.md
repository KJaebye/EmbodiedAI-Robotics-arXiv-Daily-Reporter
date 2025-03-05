# Bridging VLM and KMP: Enabling Fine-grained robotic manipulation via Semantic Keypoints Representation 

**Title (ZH)**: 连接VLM和KMP：通过语义关键点表示实现精细的机器人操作 

**Authors**: Junjie Zhu, Huayu Liu, Jin Wang, Bangrong Wen, Kaixiang Huang, Xiaofei Li, Haiyun Zhan, Guodong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02748)  

**Abstract**: From early Movement Primitive (MP) techniques to modern Vision-Language Models (VLMs), autonomous manipulation has remained a pivotal topic in robotics. As two extremes, VLM-based methods emphasize zero-shot and adaptive manipulation but struggle with fine-grained planning. In contrast, MP-based approaches excel in precise trajectory generalization but lack decision-making ability. To leverage the strengths of the two frameworks, we propose VL-MP, which integrates VLM with Kernelized Movement Primitives (KMP) via a low-distortion decision information transfer bridge, enabling fine-grained robotic manipulation under ambiguous situations. One key of VL-MP is the accurate representation of task decision parameters through semantic keypoints constraints, leading to more precise task parameter generation. Additionally, we introduce a local trajectory feature-enhanced KMP to support VL-MP, thereby achieving shape preservation for complex trajectories. Extensive experiments conducted in complex real-world environments validate the effectiveness of VL-MP for adaptive and fine-grained manipulation. 

**Abstract (ZH)**: 从早期的运动原型技术到现代的视觉-语言模型，自主操作一直是机器人研究中的核心议题。基于视觉-语言模型的方法强调零样本和适应性操作，但在精细操作规划上存在局限。相比之下，基于运动原型的方法在精确轨迹泛化方面表现出色，但在决策制定方面存在不足。为了融合两者的长处，我们提出了VL-MP，该方法通过低失真决策信息传递桥梁将视觉-语言模型与核化运动原型（KMP）相结合，从而在模糊情况下实现精细的机器人操作。VL-MP的关键在于通过语义关键点约束准确表示任务决策参数，从而实现更精确的任务参数生成。此外，我们还引入了一种局部轨迹特征增强的KMP，以支持VL-MP，从而实现复杂轨迹的形状保真。在复杂现实环境中的广泛实验验证了VL-MP在适应性和精细操作方面的有效性。 

---
# Variable-Friction In-Hand Manipulation for Arbitrary Objects via Diffusion-Based Imitation Learning 

**Title (ZH)**: 基于扩散推究学习的任意物体自手内操纵摩擦变量控制 

**Authors**: Qiyang Yan, Zihan Ding, Xin Zhou, Adam J. Spiers  

**Link**: [PDF](https://arxiv.org/pdf/2503.02738)  

**Abstract**: Dexterous in-hand manipulation (IHM) for arbitrary objects is challenging due to the rich and subtle contact process. Variable-friction manipulation is an alternative approach to dexterity, previously demonstrating robust and versatile 2D IHM capabilities with only two single-joint fingers. However, the hard-coded manipulation methods for variable friction hands are restricted to regular polygon objects and limited target poses, as well as requiring the policy to be tailored for each object. This paper proposes an end-to-end learning-based manipulation method to achieve arbitrary object manipulation for any target pose on real hardware, with minimal engineering efforts and data collection. The method features a diffusion policy-based imitation learning method with co-training from simulation and a small amount of real-world data. With the proposed framework, arbitrary objects including polygons and non-polygons can be precisely manipulated to reach arbitrary goal poses within 2 hours of training on an A100 GPU and only 1 hour of real-world data collection. The precision is higher than previous customized object-specific policies, achieving an average success rate of 71.3% with average pose error being 2.676 mm and 1.902 degrees. 

**Abstract (ZH)**: 基于端到端学习的任意物体在手灵巧操作方法 

---
# Vibration-Assisted Hysteresis Mitigation for Achieving High Compensation Efficiency 

**Title (ZH)**: 振动辅助滞回回程抑制以实现高补偿效率 

**Authors**: Myeongbo Park, Chunggil An, Junhyun Park, Jonghyun Kang, Minho Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02720)  

**Abstract**: Tendon-sheath mechanisms (TSMs) are widely used in minimally invasive surgical (MIS) applications, but their inherent hysteresis-caused by friction, backlash, and tendon elongation-leads to significant tracking errors. Conventional modeling and compensation methods struggle with these nonlinearities and require extensive parameter tuning. To address this, we propose a vibration-assisted hysteresis compensation approach, where controlled vibrational motion is applied along the tendon's movement direction to mitigate friction and reduce dead zones. Experimental results demonstrate that the exerted vibration consistently reduces hysteresis across all tested frequencies, decreasing RMSE by up to 23.41% (from 2.2345 mm to 1.7113 mm) and improving correlation, leading to more accurate trajectory tracking. When combined with a Temporal Convolutional Network (TCN)-based compensation model, vibration further enhances performance, achieving an 85.2% reduction in MAE (from 1.334 mm to 0.1969 mm). Without vibration, the TCN-based approach still reduces MAE by 72.3% (from 1.334 mm to 0.370 mm) under the same parameter settings. These findings confirm that vibration effectively mitigates hysteresis, improving trajectory accuracy and enabling more efficient compensation models with fewer trainable parameters. This approach provides a scalable and practical solution for TSM-based robotic applications, particularly in MIS. 

**Abstract (ZH)**: 腱鞘机制(TSMs)在微创手术(MIS)应用中广泛使用，但由于摩擦、间隙和肌腱伸长导致的固有滞后性，造成了显著的跟踪误差。传统建模和补偿方法难以应对这些非线性特性，并需要大量参数调整。为解决这一问题，我们提出了一种振动辅助滞后补偿方法，该方法通过在肌腱运动方向施加受控的振动运动来缓解摩擦并减少无效区域。实验结果表明，施加的振动在所有测试频率下都能一致地减少滞后性，RMSE最多降低23.41%（从2.2345 mm降至1.7113 mm），并提高了相关性，从而实现更准确的轨迹跟踪。当与基于时序卷积网络(TCN)的补偿模型结合使用时，振动进一步提升了性能，MAE降低了85.2%（从1.334 mm降至0.1969 mm）。在没有振动的情况下，基于TCN的方法在同一参数设置下仍能使MAE降低72.3%（从1.334 mm降至0.370 mm）。这些发现证实了振动有效地缓解了滞后性，提高了轨迹准确性，并使补偿模型更加高效，具有更少的可训练参数。该方法为基于TSM的机器人应用提供了可扩展且实用的解决方案，特别是在微创手术中。 

---
# Learning-Based Passive Fault-Tolerant Control of a Quadrotor with Rotor Failure 

**Title (ZH)**: 基于学习的旋翼失效条件下 quadrotor 的被动容错控制 

**Authors**: Jiehao Chen, Kaidong Zhao, Zihan Liu, YanJie Li, Yunjiang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02649)  

**Abstract**: This paper proposes a learning-based passive fault-tolerant control (PFTC) method for quadrotor capable of handling arbitrary single-rotor failures, including conditions ranging from fault-free to complete rotor failure, without requiring any rotor fault information or controller switching. Unlike existing methods that treat rotor faults as disturbances and rely on a single controller for multiple fault scenarios, our approach introduces a novel Selector-Controller network structure. This architecture integrates fault detection module and the controller into a unified policy network, effectively combining the adaptability to multiple fault scenarios of PFTC with the superior control performance of active fault-tolerant control (AFTC). To optimize performance, the policy network is trained using a hybrid framework that synergizes reinforcement learning (RL), behavior cloning (BC), and supervised learning with fault information. Extensive simulations and real-world experiments validate the proposed method, demonstrating significant improvements in fault response speed and position tracking performance compared to state-of-the-art PFTC and AFTC approaches. 

**Abstract (ZH)**: 基于学习的被动容错控制方法：适用于处理任意单旋翼故障的四旋翼无人机容错控制 

---
# Impact of Temporal Delay on Radar-Inertial Odometry 

**Title (ZH)**: 雷达-惯性里程计中时间延迟的影响 

**Authors**: Vlaho-Josip Štironja, Luka Petrović, Juraj Peršić, Ivan Marković, Ivan Petrović  

**Link**: [PDF](https://arxiv.org/pdf/2503.02509)  

**Abstract**: Accurate ego-motion estimation is a critical component of any autonomous system. Conventional ego-motion sensors, such as cameras and LiDARs, may be compromised in adverse environmental conditions, such as fog, heavy rain, or dust. Automotive radars, known for their robustness to such conditions, present themselves as complementary sensors or a promising alternative within the ego-motion estimation frameworks. In this paper we propose a novel Radar-Inertial Odometry (RIO) system that integrates an automotive radar and an inertial measurement unit. The key contribution is the integration of online temporal delay calibration within the factor graph optimization framework that compensates for potential time offsets between radar and IMU measurements. To validate the proposed approach we have conducted thorough experimental analysis on real-world radar and IMU data. The results show that, even without scan matching or target tracking, integration of online temporal calibration significantly reduces localization error compared to systems that disregard time synchronization, thus highlighting the important role of, often neglected, accurate temporal alignment in radar-based sensor fusion systems for autonomous navigation. 

**Abstract (ZH)**: 基于雷达和惯性测量单元的在线时间延迟校准里程计系统（Radar-Inertial Odometry with Online Temporal Delay Calibration） 

---
# SEB-Naver: A SE(2)-based Local Navigation Framework for Car-like Robots on Uneven Terrain 

**Title (ZH)**: SEB-Naver：一种基于SE(2)的地面不平环境下类汽车机器人本地导航框架 

**Authors**: Xiaoying Li, Long Xu, Xiaolin Huang, Donglai Xue, Zhihao Zhang, Zhichao Han, Chao Xu, Yanjun Cao, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02412)  

**Abstract**: Autonomous navigation of car-like robots on uneven terrain poses unique challenges compared to flat terrain, particularly in traversability assessment and terrain-associated kinematic modelling for motion planning. This paper introduces SEB-Naver, a novel SE(2)-based local navigation framework designed to overcome these challenges. First, we propose an efficient traversability assessment method for SE(2) grids, leveraging GPU parallel computing to enable real-time updates and maintenance of local maps. Second, inspired by differential flatness, we present an optimization-based trajectory planning method that integrates terrain-associated kinematic models, significantly improving both planning efficiency and trajectory quality. Finally, we unify these components into SEB-Naver, achieving real-time terrain assessment and trajectory optimization. Extensive simulations and real-world experiments demonstrate the effectiveness and efficiency of our approach. The code is at this https URL. 

**Abstract (ZH)**: 非平坦地形上类似汽车的机器人自主导航面临独特的挑战，特别是在可达性评估和与地形相关的运动规划中的运动学建模方面。本文介绍了一种新的基于SE(2)的局部导航框架SEB-Naver，旨在克服这些挑战。首先，我们提出了一种高效的SE(2)网格可达性评估方法，利用GPU并行计算实现局部地图的实时更新和维护。其次，受微分平坦性启发，我们提出了一种基于优化的轨迹规划方法，结合了与地形相关的运动学模型，显著提高了规划效率和轨迹质量。最后，我们将这些组件统一整合为SEB-Naver，实现了实时地形评估和轨迹优化。广泛的仿真实验和现实世界实验验证了我们方法的有效性和效率。代码链接见https URL。 

---
# Predictive Kinematic Coordinate Control for Aerial Manipulators based on Modified Kinematics Learning 

**Title (ZH)**: 基于改进动力学学习的空中 manipulator 预测运动坐标控制 

**Authors**: Zhengzhen Li, Jiahao Shen, Mengyu Ji, Huazi Cao, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02408)  

**Abstract**: High-precision manipulation has always been a developmental goal for aerial manipulators. This paper investigates the kinematic coordinate control issue in aerial manipulators. We propose a predictive kinematic coordinate control method, which includes a learning-based modified kinematic model and a model predictive control (MPC) scheme based on weight allocation. Compared to existing methods, our proposed approach offers several attractive features. First, the kinematic model incorporates closed-loop dynamics characteristics and online residual learning. Compared to methods that do not consider closed-loop dynamics and residuals, our proposed method has improved accuracy by 59.6$\%$. Second, a MPC scheme that considers weight allocation has been proposed, which can coordinate the motion strategies of quadcopters and manipulators. Compared to methods that do not consider weight allocation, the proposed method can meet the requirements of more tasks. The proposed approach is verified through complex trajectory tracking and moving target tracking experiments. The results validate the effectiveness of the proposed method. 

**Abstract (ZH)**: 高精度操作一直是空中 manipulator 发展的目标。本文研究了空中 manipulator 的运动坐标控制问题。我们提出了一种预测性运动坐标控制方法，该方法包括基于学习的改进运动学模型和基于权重分配的模型预测控制（MPC）方案。与现有方法相比，我们提出的方法具有多个吸引特点。首先，运动学模型包含了闭环动力学特性和在线残差学习。与不考虑闭环动力学和残差的方法相比，我们提出的方法准确性提高了59.6%。其次，我们提出了一个考虑权重分配的MPC方案，可以协调四旋翼和 manipulator 的运动策略。与不考虑权重分配的方法相比，所提出的方法可以满足更多的任务需求。所提出的方法通过复杂轨迹跟踪和移动目标跟踪实验得到了验证，结果验证了所提出方法的有效性。 

---
# Model-Based Capacitive Touch Sensing in Soft Robotics: Achieving Robust Tactile Interactions for Artistic Applications 

**Title (ZH)**: 基于模型的柔性机器人电容式触觉传感：实现艺术应用中的稳健触觉交互 

**Authors**: Carolina Silva-Plata, Carlos Rosel, Barnabas Gavin Cangan, Hosam Alagi, Björn Hein, Robert K. Katzschmann, Rubén Fernández, Yosra Mojtahedi, Stefan Escaida Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2503.02280)  

**Abstract**: In this paper, we present a touch technology to achieve tactile interactivity for human-robot interaction (HRI) in soft robotics. By combining a capacitive touch sensor with an online solid mechanics simulation provided by the SOFA framework, contact detection is achieved for arbitrary shapes. Furthermore, the implementation of the capacitive touch technology presented here is selectively sensitive to human touch (conductive objects), while it is largely unaffected by the deformations created by the pneumatic actuation of our soft robot. Multi-touch interactions are also possible. We evaluated our approach with an organic soft robotics sculpture that was created by a visual artist. In particular, we evaluate that the touch localization capabilities are robust under the deformation of the device. We discuss the potential this approach has for the arts and entertainment as well as other domains. 

**Abstract (ZH)**: 本文介绍了一种触觉技术，用于实现软机器人中的人机交互（HRI）的触觉互动。通过将电容式触摸传感器与由SOFA框架提供的在线固体力学模拟结合，实现了任意形状的接触检测。此外，本文呈现的电容式触摸技术对人类触摸（导电物体）具有选择性敏感性，而对由我们软机器人气动驱动产生的形变影响较小。多点触摸交互也是可能的。我们利用一位视觉艺术家创作的有机软机器人雕塑评估了该方法。特别地，我们评估了在设备形变情况下触摸定位能力的鲁棒性。我们探讨了该方法在艺术与娱乐以及其他领域的潜在应用价值。 

---
# ForaNav: Insect-inspired Online Target-oriented Navigation for MAVs in Tree Plantations 

**Title (ZH)**: ForaNav：基于昆虫启发的面向目标的 MAVs 树植造林在线导航 

**Authors**: Weijie Kuang, Hann Woei Ho, Ye Zhou, Shahrel Azmin Suandi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02275)  

**Abstract**: Autonomous Micro Air Vehicles (MAVs) are becoming essential in precision agriculture to enhance efficiency and reduce labor costs through targeted, real-time operations. However, existing unmanned systems often rely on GPS-based navigation, which is prone to inaccuracies in rural areas and limits flight paths to predefined routes, resulting in operational inefficiencies. To address these challenges, this paper presents ForaNav, an insect-inspired navigation strategy for autonomous navigation in plantations. The proposed method employs an enhanced Histogram of Oriented Gradient (HOG)-based tree detection approach, integrating hue-saturation histograms and global HOG feature variance with hierarchical HOG extraction to distinguish oil palm trees from visually similar objects. Inspired by insect foraging behavior, the MAV dynamically adjusts its path based on detected trees and employs a recovery mechanism to stay on course if a target is temporarily lost. We demonstrate that our detection method generalizes well to different tree types while maintaining lower CPU usage, lower temperature, and higher FPS than lightweight deep learning models, making it well-suited for real-time applications. Flight test results across diverse real-world scenarios show that the MAV successfully detects and approaches all trees without prior tree location, validating its effectiveness for agricultural automation. 

**Abstract (ZH)**: 自主微型空中 Vehicles (MAVs) 在精准农业中的应用通过目标导向的实时操作提高了效率并减少了劳动力成本。然而，现有的无人系统通常依赖于基于GPS的导航，这在农村区域容易出现不准确性，并限制飞行路径为预定义路线，导致操作效率低下。为了解决这些挑战，本文提出了一种受昆虫启发的导航策略ForaNav，以实现植物园中的自主导航。该方法采用改进的基于方向梯度直方图（HOG）的树木检测方法，结合色调饱和度直方图和全局HOG特征方差及分层HOG提取，以区分油棕榈树与其他视觉相似的物体。受昆虫觅食行为的启发，MAV根据检测到的树木动态调整其路径，并采用恢复机制以防止单一目标暂时丢失。我们证明，我们的检测方法在不同树木类型上具有良好的泛化能力，同时具有较低的CPU使用率、较低的温度和更高的FPS，使其适用于实时应用。在多种真实世界场景下的飞行测试结果表明，MAV能够成功地检测并接近所有树木，无需事先知道树木位置，验证了其在农业自动化中的有效性。 

---
# Towards Fluorescence-Guided Autonomous Robotic Partial Nephrectomy on Novel Tissue-Mimicking Hydrogel Phantoms 

**Title (ZH)**: 面向荧光引导的自主机器人部分肾切除手术新型组织模拟水凝胶phantom研究 

**Authors**: Ethan Kilmer, Joseph Chen, Jiawei Ge, Preksha Sarda, Richard Cha, Kevin Cleary, Lauren Shepard, Ahmed Ezzat Ghazi, Paul Maria Scheikl, Axel Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2503.02265)  

**Abstract**: Autonomous robotic systems hold potential for improving renal tumor resection accuracy and patient outcomes. We present a fluorescence-guided robotic system capable of planning and executing incision paths around exophytic renal tumors with a clinically relevant resection margin. Leveraging point cloud observations, the system handles irregular tumor shapes and distinguishes healthy from tumorous tissue based on near-infrared imaging, akin to indocyanine green staining in partial nephrectomy. Tissue-mimicking phantoms are crucial for the development of autonomous robotic surgical systems for interventions where acquiring ex-vivo animal tissue is infeasible, such as cancer of the kidney and renal pelvis. To this end, we propose novel hydrogel-based kidney phantoms with exophytic tumors that mimic the physical and visual behavior of tissue, and are compatible with electrosurgical instruments, a common limitation of silicone-based phantoms. In contrast to previous hydrogel phantoms, we mix the material with near-infrared dye to enable fluorescence-guided tumor segmentation. Autonomous real-world robotic experiments validate our system and phantoms, achieving an average margin accuracy of 1.44 mm in a completion time of 69 sec. 

**Abstract (ZH)**: 自主机器人系统有潜力提高肾肿瘤切除准确性和患者 outcomes。我们提出了一种荧光引导的机器人系统，能够规划并执行围绕外生性肾肿瘤的切口路径，同时保持临床相关的切除边缘。该系统利用点云观察，处理不规则的肿瘤形状，并根据近红外成像区分健康组织和肿瘤组织，类似于肾部分切除术中的吲哚菁绿染色。对于在获取离体动物组织不可行的情况下进行的干预，如肾癌和肾盂癌，仿组织水凝胶假体对于自主机器人外科手术系统的发展至关重要。为此，我们提出了一种新型水凝胶基肾脏假体，具有外生性肿瘤，能够模拟组织的物理和视觉行为，并与电外科器械兼容，后者是基于硅胶的假体的常见限制。与之前的水凝胶假体不同，我们将材料与近红外染料混合，以实现荧光引导的肿瘤分割。自主现实世界机器人实验验证了我们的系统和假体，在完成时间为69秒的情况下，平均边缘准确度为1.44毫米。 

---
# Zero-Shot Sim-to-Real Visual Quadrotor Control with Hard Constraints 

**Title (ZH)**: 零样本仿真实践视觉四旋翼控制带硬约束 

**Authors**: Yan Miao, Will Shen, Sayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2503.02198)  

**Abstract**: We present the first framework demonstrating zero-shot sim-to-real transfer of visual control policies learned in a Neural Radiance Field (NeRF) environment for quadrotors to fly through racing gates. Robust transfer from simulation to real flight poses a major challenge, as standard simulators often lack sufficient visual fidelity. To address this, we construct a photorealistic simulation environment of quadrotor racing tracks, called FalconGym, which provides effectively unlimited synthetic images for training. Within FalconGym, we develop a pipelined approach for crossing gates that combines (i) a Neural Pose Estimator (NPE) coupled with a Kalman filter to reliably infer quadrotor poses from single-frame RGB images and IMU data, and (ii) a self-attention-based multi-modal controller that adaptively integrates visual features and pose estimation. This multi-modal design compensates for perception noise and intermittent gate visibility. We train this controller purely in FalconGym with imitation learning and deploy the resulting policy to real hardware with no additional fine-tuning. Simulation experiments on three distinct tracks (circle, U-turn and figure-8) demonstrate that our controller outperforms a vision-only state-of-the-art baseline in both success rate and gate-crossing accuracy. In 30 live hardware flights spanning three tracks and 120 gates, our controller achieves a 95.8% success rate and an average error of just 10 cm when flying through 38 cm-radius gates. 

**Abstract (ZH)**: 我们提出了第一个框架，该框架展示了通过神经辐射场（NeRF）环境学习的视觉控制策略在quadrotor从仿真到現實飞行中的零样本迁移。我们构建了名为FalconGym的真实感仿真实验环境，提供了无限的合成图像用于训练。在FalconGym中，我们开发了一种流水线方法来穿越门，该方法结合了（i）一种与卡尔曼滤波器耦合的神经位姿估计器（NPE），用于可靠地从单帧RGB图像和IMU数据中推断quadrotor位姿；和（ii）一种基于自我注意力的多模态控制器，该控制器能够自适应地整合视觉特征和位姿估计。这种多模态设计补偿了感知噪声和门的间歇性可见性。我们仅使用演示学习在FalconGym中训练该控制器，并在无需额外微调的情况下将其部署到实际硬件中。针对三个不同赛道（圆形、U型和8字形）的仿真实验表明，我们的控制器在成功率和门穿越精度方面均优于现有的仅基于视觉的先进基线。在跨越38厘米半径的120个门的30次实际硬件飞行中，我们的控制器实现了95.8%的成功率和平均每错误位移仅10厘米。标题：

零样本视觉控制策略从仿真到现实飞行的quadrotor门穿越迁移 

---
# RPF-Search: Field-based Search for Robot Person Following in Unknown Dynamic Environments 

**Title (ZH)**: 基于字段的搜索：未知动态环境中机器人跟随人员的搜索方法 

**Authors**: Hanjing Ye, Kuanqi Cai, Yu Zhan, Bingyi Xia, Arash Ajoudani, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02188)  

**Abstract**: Autonomous robot person-following (RPF) systems are crucial for personal assistance and security but suffer from target loss due to occlusions in dynamic, unknown environments. Current methods rely on pre-built maps and assume static environments, limiting their effectiveness in real-world settings. There is a critical gap in re-finding targets under topographic (e.g., walls, corners) and dynamic (e.g., moving pedestrians) occlusions. In this paper, we propose a novel heuristic-guided search framework that dynamically builds environmental maps while following the target and resolves various occlusions by prioritizing high-probability areas for locating the target. For topographic occlusions, a belief-guided search field is constructed and used to evaluate the likelihood of the target's presence, while for dynamic occlusions, a fluid-field approach allows the robot to adaptively follow or overtake moving occluders. Past motion cues and environmental observations refine the search decision over time. Our results demonstrate that the proposed method outperforms existing approaches in terms of search efficiency and success rates, both in simulations and real-world tests. Our target search method enhances the adaptability and reliability of RPF systems in unknown and dynamic environments to support their use in real-world applications. Our code, video, experimental results and appendix are available at this https URL. 

**Abstract (ZH)**: 自主机器人跟踪系统中的路径跟随（RPF）对于个人辅助和安全至关重要，但在动态、未知环境中由于遮挡会失去目标。当前方法依赖预构建的地图并假设静态环境，限制了其在现实环境中的有效性。在地形遮挡（例如，墙壁、角落）和动态遮挡（例如，移动行人）下重新找到目标存在关键空白。本文提出了一种新的启发式引导搜索框架，在跟随目标的同时动态构建环境地图，并通过优先搜索高概率区域来解决各种遮挡问题。对于地形遮挡，构建信念引导的搜索字段来评估目标存在的可能性；对于动态遮挡，流场方法使机器人能够适应性地跟随或超越移动遮挡物。过往运动线索和环境观察随着时间的推移细化搜索决策。我们的实验结果表明，所提出的方法在搜索效率和成功率上优于现有方法，无论是仿真还是实地测试。我们的目标搜索方法增强了RPF系统在未知和动态环境中的适应性和可靠性，支持其实用应用。我们的代码、视频、实验结果和附录可在以下链接获取：this https URL。 

---
# Design and Control of A Tilt-Rotor Tailsitter Aircraft with Pivoting VTOL Capability 

**Title (ZH)**: 具有pivot VTOL能力的傾轉旋翼垂直起降飞机的设计与控制 

**Authors**: Ziqing Ma, Ewoud J.J. Smeur, Guido C.H.E. de Croon  

**Link**: [PDF](https://arxiv.org/pdf/2503.02158)  

**Abstract**: Tailsitter aircraft attract considerable interest due to their capabilities of both agile hover and high speed forward flight. However, traditional tailsitters that use aerodynamic control surfaces face the challenge of limited control effectiveness and associated actuator saturation during vertical flight and transitions. Conversely, tailsitters relying solely on tilting rotors have the drawback of insufficient roll control authority in forward flight. This paper proposes a tilt-rotor tailsitter aircraft with both elevons and tilting rotors as a promising solution. By implementing a cascaded weighted least squares (WLS) based incremental nonlinear dynamic inversion (INDI) controller, the drone successfully achieved autonomous waypoint tracking in outdoor experiments at a cruise airspeed of 16 m/s, including transitions between forward flight and hover without actuator saturation. Wind tunnel experiments confirm improved roll control compared to tilt-rotor-only configurations, while comparative outdoor flight tests highlight the vehicle's superior control over elevon-only designs during critical phases such as vertical descent and transitions. Finally, we also show that the tilt-rotors allow for an autonomous takeoff and landing with a unique pivoting capability that demonstrates stability and robustness under wind disturbances. 

**Abstract (ZH)**: 兼具升降舵和倾斜旋翼的尾座式无人机由于其在悬停和高速前进飞行中的能力而引起广泛关注。然而，传统尾座式无人机采用气动控制舵面，在垂直飞行和转换过程中面临控制效果有限和相关作动器饱和的挑战。相反，仅依靠倾斜旋翼的尾座式无人机在前进飞行中存在滚转控制权威不足的问题。本文提出了一种兼具升降舵和倾斜旋翼的尾座式无人机作为潜在解决方案。通过实施基于加权最小二乘法的嵌套非线性动态反演控制方法，无人机在外场试验中成功实现了自主航点跟踪，巡航空速为16 m/s，包括前进飞行与悬停之间的转换且未出现作动器饱和。风洞试验验证了与仅依靠倾斜旋翼配置相比改进了滚转控制，而在关键阶段如垂直下降和转换过程中，与仅依靠升降舵的设计相比，该无人机显示出了更优越的控制性能。最后，我们还展示了倾斜旋翼的倾斜能力允许该无人机自主起飞和降落，证明了在风干扰下的稳定性和鲁棒性。 

---
# NavG: Risk-Aware Navigation in Crowded Environments Based on Reinforcement Learning with Guidance Points 

**Title (ZH)**: NavG：基于强化学习和引导点的风险感知导航在拥挤环境中的应用 

**Authors**: Qianyi Zhang, Wentao Luo, Boyi Liu, Ziyang Zhang, Yaoyuan Wang, Jingtai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02111)  

**Abstract**: Motion planning in navigation systems is highly susceptible to upstream perceptual errors, particularly in human detection and tracking. To mitigate this issue, the concept of guidance points--a novel directional cue within a reinforcement learning-based framework--is introduced. A structured method for identifying guidance points is developed, consisting of obstacle boundary extraction, potential guidance point detection, and redundancy elimination. To integrate guidance points into the navigation pipeline, a perception-to-planning mapping strategy is proposed, unifying guidance points with other perceptual inputs and enabling the RL agent to effectively leverage the complementary relationships among raw laser data, human detection and tracking, and guidance points. Qualitative and quantitative simulations demonstrate that the proposed approach achieves the highest success rate and near-optimal travel times, greatly improving both safety and efficiency. Furthermore, real-world experiments in dynamic corridors and lobbies validate the robot's ability to confidently navigate around obstacles and robustly avoid pedestrians. 

**Abstract (ZH)**: 基于强化学习的导航系统中的运动规划易受上游感知错误的影响，特别是在人类检测和跟踪方面。为缓解这一问题，提出了指导点——一种新的方向性提示——这一概念并融入到基于强化学习的框架中。开发了一种结构化的指导点识别方法，包括障碍边界提取、潜在指导点检测和冗余消除。为了将指导点整合到导航管道中，提出了一种感知到规划的映射策略，将指导点与其他感知输入统一起来，使RL代理能够有效利用原始激光数据、人类检测与跟踪以及指导点之间的互补关系。定性和定量的仿真实验表明，所提出的方法实现了最高的成功率和接近最优的旅行时间，大大提高了安全性和效率。此外，在动态走廊和lobby的实际实验中验证了机器人能够自信地绕过障碍物并 robust 地避开行人。 

---
# Velocity-free task-space regulator for robot manipulators with external disturbances 

**Title (ZH)**: 基于外部干扰的机器人 manipulators 无速度任务空间调节器 

**Authors**: Haiwen Wu, Bayu Jayawardhana, Dabo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02634)  

**Abstract**: This paper addresses the problem of task-space robust regulation of robot manipulators subject to external disturbances. A velocity-free control law is proposed by combining the internal model principle and the passivity-based output-feedback control approach. The developed output-feedback controller ensures not only asymptotic convergence of the regulation error but also suppression of unwanted external step/sinusoidal disturbances. The potential of the proposed method lies in its simplicity, intuitively appealing, and simple gain selection criteria for synthesis of multi-joint robot manipulator control systems. 

**Abstract (ZH)**: 本文解决了机器人 manipulator 在外部干扰下的任务空间鲁棒调节问题。通过结合内部模型原则和基于耗散性输出反馈控制方法，提出了一种无速度控制律。所发展的输出反馈控制器不仅确保调节误差的渐近收敛，而且还抑制了不必要的外部阶跃/正弦干扰。所提出方法的潜力在于其简洁性、直观性和多关节机器人控制系统合成中的简单增益选择准则。 

---
# Optimizing Robot Programming: Mixed Reality Gripper Control 

**Title (ZH)**: 机器人编程优化：混合现实抓手控制 

**Authors**: Maximilian Rettinger, Leander Hacker, Philipp Wolters, Gerhard Rigoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.02042)  

**Abstract**: Conventional robot programming methods are complex and time-consuming for users. In recent years, alternative approaches such as mixed reality have been explored to address these challenges and optimize robot programming. While the findings of the mixed reality robot programming methods are convincing, most existing methods rely on gesture interaction for robot programming. Since controller-based interactions have proven to be more reliable, this paper examines three controller-based programming methods within a mixed reality scenario: 1) Classical Jogging, where the user positions the robot's end effector using the controller's thumbsticks, 2) Direct Control, where the controller's position and orientation directly corresponds to the end effector's, and 3) Gripper Control, where the controller is enhanced with a 3D-printed gripper attachment to grasp and release objects. A within-subjects study (n = 30) was conducted to compare these methods. The findings indicate that the Gripper Control condition outperforms the others in terms of task completion time, user experience, mental demand, and task performance, while also being the preferred method. Therefore, it demonstrates promising potential as an effective and efficient approach for future robot programming. Video available at this https URL. 

**Abstract (ZH)**: 传统的机器人编程方法对用户来说复杂且耗时。近年来，混合现实等替代方法被积极探索以应对这些挑战并优化机器人编程。尽管混合现实机器人编程方法的研究结果令人信服，但大多数现有方法仍依赖手势交互进行机器人编程。由于基于控制器的交互已被证明更可靠，本文在混合现实场景下研究了三种基于控制器的编程方法：1）经典 Jogging 方法，其中用户使用控制器的拇指摇杆定位机器人的末端执行器；2）直接控制方法，其中控制器的位置和方向直接对应末端执行器的位置和方向；3）夹爪控制方法，其中控制器配备了3D打印的夹爪附件以抓取和释放物体。进行了一个被试内研究（n=30）来比较这些方法。研究发现，夹爪控制条件在任务完成时间、用户体验、心理需求和任务性能方面优于其他方法，并且是用户偏好方法，因此证明了其作为未来机器人编程的有效且高效方法的有希望的潜力。视频链接：此 https URL。 

---
# Minimum-Length Coordinated Motions For Two Convex Centrally-Symmetric Robots 

**Title (ZH)**: 两个凸中心对称机器人 的最小协调运动 

**Authors**: David Kirkpatrick, Paul Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02010)  

**Abstract**: We study the problem of determining coordinated motions, of minimum total length, for two arbitrary convex centrally-symmetric (CCS) robots in an otherwise obstacle-free plane. Using the total path length traced by the two robot centres as a measure of distance, we give an exact characterization of a (not necessarily unique) shortest collision-avoiding motion for all initial and goal configurations of the robots. The individual paths are composed of at most six convex pieces, and their total length can be expressed as a simple integral with a closed form solution depending only on the initial and goal configuration of the robots. The path pieces are either straight segments or segments of the boundary of the Minkowski sum of the two robots (circular arcs, in the special case of disc robots). Furthermore, the paths can be parameterized in such a way that (i) only one robot is moving at any given time (decoupled motion), or (ii) the orientation of the robot configuration changes monotonically. 

**Abstract (ZH)**: 我们研究了在无障碍平面内确定两个任意中心对称凸机器人（CCS）的协调运动问题，使得总长度最小。使用两个机器人中心所追踪的总路径长度作为距离的度量，给出了所有初始和目标配置下机器人避免碰撞的最短运动的精确描述（不一定唯一）。各个路径最多由六段凸部分组成，其总长度可以表示为仅依赖于机器人初始和目标配置的简单闭合形式积分。路径片段要么是直线段，要么是两个机器人Minkowski和的边界部分（在特殊情况下为圆弧段）。进一步地，路径可以参数化为如下方式：（i）任何时候只有单个机器人移动（解耦运动），或（ii）机器人配置的朝向单调变化。 

---
# Tracking Control of Euler-Lagrangian Systems with Prescribed State, Input, and Temporal Constraints 

**Title (ZH)**: 带有预先指定状态、输入和时间约束的Euler-Lagrangian系统跟踪控制 

**Authors**: Chidre Shravista Kashyap, Pushpak Jagtap, Jishnu Keshavan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01866)  

**Abstract**: The synthesis of a smooth tracking control policy for Euler-Lagrangian (EL) systems with stringent regions of operation induced by state, input and temporal (SIT) constraints is a very challenging task. Most existing solutions rely on prior information of the parameters of the nominal EL dynamics together with bounds on system uncertainty, and incorporate either state or input constraints to guarantee tracking error convergence in a prescribed settling time. Contrary to these approaches, this study proposes an approximation-free adaptive barrier function-based control policy for achieving local prescribed-time convergence of the tracking error to a prescribed-bound in the presence of state and input constraints. This is achieved by imposing time-varying bounds on the filtered tracking error to confine the states within their respective bounds, while also incorporating a saturation function to limit the magnitude of the proposed control action that leverages smooth time-based generator functions for ensuring tracking error convergence within the prescribed-time. Importantly, corresponding feasibility conditions pertaining to the minimum control authority, maximum disturbance rejection capability of the control policy, and the viable set of initial conditions are derived, illuminating the narrow operating domain of the EL systems arising from the interplay of SIT constraints. Numerical validation studies with three different robotic manipulators are employed to demonstrate the efficacy of the proposed scheme. A detailed performance comparison study with leading alternative designs is also undertaken to illustrate the superior performance of the proposed scheme. 

**Abstract (ZH)**: 基于状态和输入时变约束的埃尔朗gen-拉格朗日系统平滑跟踪控制策略的研究 

---
# A strictly predefined-time convergent and anti-noise fractional-order zeroing neural network for solving time-variant quadratic programming in kinematic robot control 

**Title (ZH)**: 严格预定义时间收敛且抗噪声的分数阶零点神经网络在Kinematic机器人控制中求解时变二次规划 

**Authors**: Yi Yang, Xiao Li, Xuchen Wang, Mei Liu, Junwei Yin, Weibing Li, Richard M. Voyles, Xin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.01857)  

**Abstract**: This paper proposes a strictly predefined-time convergent and anti-noise fractional-order zeroing neural network (SPTC-AN-FOZNN) model, meticulously designed for addressing time-variant quadratic programming (TVQP) problems. This model marks the first variable-gain ZNN to collectively manifest strictly predefined-time convergence and noise resilience, specifically tailored for kinematic motion control of robots. The SPTC-AN-FOZNN advances traditional ZNNs by incorporating a conformable fractional derivative in accordance with the Leibniz rule, a compliance not commonly achieved by other fractional derivative definitions. It also features a novel activation function designed to ensure favorable convergence independent of the model's order. When compared to five recently published recurrent neural networks (RNNs), the SPTC-AN-FOZNN, configured with $0<\alpha\leq 1$, exhibits superior positional accuracy and robustness against additive noises for TVQP applications. Extensive empirical evaluations, including simulations with two types of robotic manipulators and experiments with a Flexiv Rizon robot, have validated the SPTC-AN-FOZNN's effectiveness in precise tracking and computational efficiency, establishing its utility for robust kinematic control. 

**Abstract (ZH)**: 严格预定义时间收敛和抗噪声分数阶零阶神经网络模型：针对时变二次规划问题的严格预定义时间收敛和抗噪声可变增益零阶神经网络模型 

---
# Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation 

**Title (ZH)**: 反应扩散策略：接触丰富操作中的慢-快视觉-触觉策略学习 

**Authors**: Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02881)  

**Abstract**: Humans can accomplish complex contact-rich tasks using vision and touch, with highly reactive capabilities such as quick adjustments to environmental changes and adaptive control of contact forces; however, this remains challenging for robots. Existing visual imitation learning (IL) approaches rely on action chunking to model complex behaviors, which lacks the ability to respond instantly to real-time tactile feedback during the chunk execution. Furthermore, most teleoperation systems struggle to provide fine-grained tactile / force feedback, which limits the range of tasks that can be performed. To address these challenges, we introduce TactAR, a low-cost teleoperation system that provides real-time tactile feedback through Augmented Reality (AR), along with Reactive Diffusion Policy (RDP), a novel slow-fast visual-tactile imitation learning algorithm for learning contact-rich manipulation skills. RDP employs a two-level hierarchy: (1) a slow latent diffusion policy for predicting high-level action chunks in latent space at low frequency, (2) a fast asymmetric tokenizer for closed-loop tactile feedback control at high frequency. This design enables both complex trajectory modeling and quick reactive behavior within a unified framework. Through extensive evaluation across three challenging contact-rich tasks, RDP significantly improves performance compared to state-of-the-art visual IL baselines through rapid response to tactile / force feedback. Furthermore, experiments show that RDP is applicable across different tactile / force sensors. Code and videos are available on this https URL. 

**Abstract (ZH)**: 人类可以使用视觉和触觉完成复杂的接触密集型任务，并具备快速适应环境变化和调整接触力的高反应能力；然而这对机器人来说仍然具有挑战性。现有的视觉imitation learning（IL）方法依赖于动作切片来建模复杂行为，在动作切片执行过程中缺乏对实时触觉反馈的即时响应能力。此外，大多数远程操作系统难以提供精细的触觉/力反馈，这限制了可执行任务的范围。为了解决这些挑战，我们引入了TactAR，这是一种通过增强现实（AR）提供实时触觉反馈的低成本远程操作系统，以及用于学习接触密集型操作技能的新型慢速-快速视觉-触觉imitation learning算法Reactive Diffusion Policy（RDP）。RDP采用两层层次结构：（1）低频的慢速潜在扩散策略用于预测低维潜在空间中的高层动作切片；（2）高频的不对称标记器用于闭环触觉反馈控制。这一设计使得复杂轨迹建模和快速反应行为可以在统一框架内实现。通过在三个具有挑战性的接触密集型任务上的广泛评估，RDP相较于最先进的视觉IL基准实现了显著的性能提升，通过快速响应触觉/力反馈。此外，实验表明RDP适用于不同类型的触觉/力传感器。相关代码和视频可在以下链接获得。 

---
